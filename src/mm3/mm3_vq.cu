// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.63).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/mm3/mm3_vq.cu - the Baekpica VQ weights on the device (see mm3_vq.cuh for the format and the blob law).
//
// Phase A of the VQ lane: EXACT arithmetic throughout - the activations stay F32 (no Q8_K rounding), the dot is
// an F32 FMA chain over the decoded E4M3 values, the row gain applied once at the row's end.  The VQ dequant is
// cheap enough that the CPU lane can run the same F32 math bit-compatibly (a gather plus an FMA), so the lane's
// GPU/CPU parity does not need gg's shared-rounding trick; when a CPU lane lands it implements THIS arithmetic.
// The kernels read a blob's GVQ1 payload through its own 128-byte header (the host pass validated it), so the
// device geometry comes from the data, not a second copy of it.
#include "mm3_vq.cuh"

#include "mm3_env.hpp"   // (the debug gates read MM3_VQ_DEBUG)

#include "mm3_gguf.cuh"   // act_quant forwards to the GGUF lane's shared rounding

#include <assert.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <atomic>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

namespace mm3::vq {

using x3::ExpertLayout;
using x3::Mat;
using x3::MoeArgs;
using x3::SetArgs;
using x3::WinArgs;

// ------------------------------------------------------------------ housekeeping

namespace {
std::atomic<int> g_launch_errors{0};
std::atomic<unsigned> g_wait_timeouts{0};
void fail(const char* what) {
    std::fprintf(stderr, "mm3 vq %s: launch failed\n", what);
    g_launch_errors.fetch_add(1, std::memory_order_relaxed);
}
void check(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mm3 vq %s: %s\n", what, cudaGetErrorString(e));
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    }
}
constexpr int kMaxEmbd = 8192, kMaxFf = 4096, kMaxK = 8;   // the geometry() guards (M3.1: 6144 / 3072 / 4)
bool geometry(const char* what, int k, int n_embd, int n_ff) {
    if (k < 1 || k > kMaxK || n_embd < 128 || n_embd > kMaxEmbd || n_ff < 128 || n_ff > kMaxFf ||
        n_embd % 128 || n_ff % 128) {
        std::fprintf(stderr, "mm3 vq %s: geometry not covered (k %d, n_embd %d, n_ff %d)\n", what, k, n_embd, n_ff);
        return false;
    }
    return true;
}

struct Scratch {
    float* h = nullptr;        // [kMaxK][kMaxFf]: a token's experts' activations (moe_gate_up's out)
    float* part = nullptr;     // [kMaxK][kMaxEmbd]: the experts' down outputs before the combine
    uint16_t* wb = nullptr;    // the GEMM bands' decoded FP16 weights
    cublasHandle_t blas = nullptr;
};
std::mutex g_mu;
std::map<cudaStream_t, Scratch*> g_scratch;

constexpr size_t kBand = 64u << 20;   // the FP16 band budget (gg's)

Scratch* scratch(cudaStream_t s) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scratch.find(s);
    if (it != g_scratch.end()) return it->second;
    Scratch* sc = new Scratch();
    if (cudaMalloc(&sc->h, sizeof(float) * kMaxK * kMaxFf) != cudaSuccess ||
        cudaMalloc(&sc->part, sizeof(float) * kMaxK * kMaxEmbd) != cudaSuccess ||
        cudaMalloc(&sc->wb, kBand) != cudaSuccess ||
        cublasCreate(&sc->blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(sc->blas, s) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "mm3 vq: scratch allocation failed\n");
        delete sc;
        g_scratch[s] = nullptr;
        return nullptr;
    }
    g_scratch[s] = sc;
    return sc;
}
}   // namespace

bool prepare(cudaStream_t s) { return scratch(s) != nullptr; }
size_t scratch_bytes() { return sizeof(float) * kMaxK * (kMaxFf + kMaxEmbd) + kBand; }
int launch_errors() { return g_launch_errors.load(std::memory_order_relaxed); }
namespace { int g_trace_layer = -1; }
void trace_layer(int li) { g_trace_layer = li; }
unsigned wait_timeouts() { return g_wait_timeouts.load(std::memory_order_relaxed); }


namespace {
// The payloads live on the device; the host passes (validation, band sizing) read the 128-byte header
// through a staging copy - a host dereference of a device pointer is the segfault this helper removes.
bool host_header(const void* d_payload, GvH& out, cudaStream_t s) {
    uint8_t h[128];
    if (cudaMemcpyAsync(h, d_payload, 128, cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess)
        return false;
    out.rows = *(const uint32_t*) (h + 8);
    out.cols = *(const uint32_t*) (h + 12);
    out.entries = *(const uint32_t*) (h + 16);
    out.gain_off = *(const uint64_t*) (h + 32);
    out.index_off = *(const uint64_t*) (h + 40);
    out.high_off = *(const uint64_t*) (h + 48);
    out.total = *(const uint64_t*) (h + 56);
    return true;
}
}   // namespace

// ------------------------------------------------------------------ decode helpers (host side of the device law)

namespace {
// The three streams of a GVQ1 payload + the residual's, resolved once a kernel walks a matrix.
struct Views {
    const uint16_t* gain;
    const uint8_t *low12, *high1;
    const uint8_t* idx8;      // the GVR8 residual's index bytes (null: none)
};
__device__ Views views(const void* payload, const void* res) {
    Views v;
    const uint8_t* p = (const uint8_t*) payload;
    const GvH h = gv_header(payload);
    v.gain = (const uint16_t*) (p + h.gain_off);
    v.low12 = p + h.index_off;
    v.high1 = h.high_off ? p + h.high_off : nullptr;
    v.idx8 = nullptr;
    if (res != nullptr) {
        const uint8_t* r = (const uint8_t*) res;
        const GvH hr = gv_header(res);
        // The GVR8 wire re-uses the 128-byte layout but not the GVQ1 field roles: a residual carries no
        // gains (the row gain is shared with its GVQ1 matrix), so the u64 at 32 is the INDEX offset -
        // 128 on the real wire (verified: layer-16's three residuals read u64@32=128, u64@40=0,
        // u64@48=0).  The proven reference reads exactly this field (ds4_glmvq_gpu.cuh gvq_link_res:
        // index = gvq_u64(h, 32), required == 128, index bytes at data + index); the GVQ1-named
        // index_off/high_off read 0 and are not the residual's law.
        v.idx8 = r + hr.gain_off;
    }
    return v;
}
}   // namespace

// ------------------------------------------------------------------ mv (a dense GEMV; ncol activations a job)

namespace {
constexpr int kMvThreads = 128;   // 4 warps; a warp owns a row, its 32 lanes walk the vectors

__global__ void mv_kernel(const x3::MvJob* jobs, int n, int ncol) {
    const int job = blockIdx.z;
    if (job >= n) return;
    const Mat m = *jobs[job].m;   // one read per block
    const GvH h = gv_header(m.raw);
    const int cols = (int) h.vecs();   // the wire's cols is k (values); the index streams are per VECTOR
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.y * 4 + warp;
    if (row >= (int) h.rows) return;
    const Views v = views(m.raw, m.vq_res);
    const float g = __half2float(__ushort_as_half(v.gain[row]));
    const size_t xcs = jobs[job].xf_cs ? jobs[job].xf_cs / 4 : (size_t) m.k;
    const size_t ycs = jobs[job].y_cs ? jobs[job].y_cs / 4 : (size_t) m.n;
    for (int c = 0; c < ncol; ++c) {
        const float* x = jobs[job].xf ? jobs[job].xf + c * xcs : nullptr;
        float acc = 0.0f;
        for (int vv = lane; vv < cols; vv += 32) {
            const uint64_t j = (uint64_t) row * cols + vv;
            float w[8];
            w_vec(v.low12, v.high1, v.idx8, (const uint8_t*) m.vq_cb, (const uint8_t*) m.vq_cb2, j, w);
            const float* xc = x ? x + 8 * vv : nullptr;
            if (xc != nullptr) {
                #pragma unroll
                for (int d = 0; d < 8; ++d) acc = __fmaf_rn(w[d], xc[d], acc);
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            (jobs[job].y + c * ycs)[row] = __fmul_rn(__fmul_rn(g, acc), jobs[job].alpha) +
                                           (jobs[job].bias ? jobs[job].bias[row] : 0.0f);
        }
    }
}
}   // namespace

bool mv(const x3::MvJob* jobs, int n, cudaStream_t s) {
    if (n < 1 || n > x3::kMaxMvJobs) {
        fail("mv: job count");
        return false;
    }
    for (int i = 0; i < n; ++i)
        if (jobs[i].xf == nullptr) {   // phase A: F32 activations (xq lands with the shared-rounding pass)
            fail("mv: xf required");
            return false;
        }
    int max_rows = 0, ncol = 1;
    for (int i = 0; i < n; ++i) {
        const Mat& m = *jobs[i].m;
        if (m.fmt != kFmtVQ || m.raw == nullptr || m.vq_cb == nullptr) {
            fail("mv: not a VQ matrix");
            return false;
        }
        GvH h;
        if (!host_header(m.raw, h, s) || h.cols != (uint32_t) m.k || h.rows != (uint32_t) m.n) {
            fail("mv: header/geometry mismatch");
            return false;
        }
        max_rows = std::max(max_rows, m.n);
        ncol = std::max(ncol, jobs[i].ncol);
    }
    // the jobs' `m` points at HOST Mats; the kernel needs device-side Mat copies.  Stage both (a decode-path
    // launch, not the hot token loop, which reaches the experts through moe/win below).
    Mat* dm = nullptr;
    x3::MvJob* dj = nullptr;
    std::vector<Mat> hm(n);
    for (int i = 0; i < n; ++i) hm[i] = *jobs[i].m;
    if (cudaMallocAsync(&dm, sizeof(Mat) * n, s) != cudaSuccess ||
        cudaMallocAsync(&dj, sizeof(x3::MvJob) * n, s) != cudaSuccess) {
        fail("mv: stage");
        return false;
    }
    cudaMemcpyAsync(dm, hm.data(), sizeof(Mat) * n, cudaMemcpyHostToDevice, s);
    std::vector<x3::MvJob> hj(jobs, jobs + n);
    for (int i = 0; i < n; ++i) hj[i].m = dm + i;
    cudaMemcpyAsync(dj, hj.data(), sizeof(x3::MvJob) * n, cudaMemcpyHostToDevice, s);
    const int warps4 = (max_rows + 3) / 4;
    dim3 grid(1, warps4, n);
    mv_kernel<<<grid, kMvThreads, 0, s>>>(dj, n, ncol);
    cudaFreeAsync(dm, s);
    cudaFreeAsync(dj, s);
    check("mv");
    return true;
}

// ------------------------------------------------------------------ f16_rows + gemm_rows

namespace {
constexpr int kF16Block = 128;   // a block a row; lanes walk the vectors

__global__ void f16_rows_kernel(const void* payload, const void* res, const uint8_t* cb, const uint8_t* cb2,
                                int row0, int rows, int cols, uint16_t* dst, int ldx) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int row = row0 + r;
    const Views v = views(payload, res);
    const float g = __half2float(__ushort_as_half(v.gain[row]));
    uint16_t* out = dst + (size_t) r * ldx;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int vv = warp * 32 + lane; vv < cols; vv += kF16Block) {
        const uint64_t j = (uint64_t) row * cols + vv;
        float w[8];
        w_vec(v.low12, v.high1, v.idx8, cb, cb2, j, w);
        #pragma unroll
        for (int d = 0; d < 8; ++d) out[8 * vv + d] = __half_as_ushort(__float2half_rn(__fmul_rn(g, w[d])));
    }
}
}   // namespace

void f16_rows(const Mat& m, int row0, int rows, uint16_t* dst, int ldx, cudaStream_t s) {
    GvH h;
    if (!host_header(m.raw, h, s)) {
        fail("f16_rows: header");
        return;
    }
    const uint32_t vecs = h.vecs();
    f16_rows_kernel<<<rows, kF16Block, 0, s>>>(m.raw, m.vq_res, (const uint8_t*) m.vq_cb,
                                               (const uint8_t*) m.vq_cb2, row0, rows, (int) vecs, dst, ldx);
    check("f16_rows");
}

void gemm_rows(const Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
               cudaStream_t s) {
    if (rows <= 0) return;
    Scratch* sc = scratch(s);
    GvH h;
    if (sc == nullptr || m.fmt != kFmtVQ || m.raw == nullptr || m.vq_cb == nullptr ||
        !host_header(m.raw, h, s) || h.cols != (uint32_t) m.k) {
        fail("gemm_rows: not a VQ matrix the kernels take");
        return;
    }
    // the weight's rows (the outputs) in FP16 bands, each decoded once and multiplied by every input row - gg's
    // band law and cuBLAS call, the orientation identical (Y^T [nc][rows] = W [nc][k] x X^T)
    const int band = (int) std::max<size_t>(256, kBand / ((size_t) m.k * 2) / 256 * 256);
    const float one = 1.0f;
    for (int c0 = 0; c0 < m.n; c0 += band) {
        const int nc = std::min(band, m.n - c0);
        f16_rows(m, c0, nc, sc->wb, m.k, s);
        if (cublasGemmEx(sc->blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, m.k, &one, sc->wb, CUDA_R_16F, m.k, x16,
                         CUDA_R_16F, ldx, &beta, y + c0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
            fail("gemm_rows: cublasGemmEx");
            return;
        }
    }
    check("gemm_rows");
}

// ------------------------------------------------------------------ the routed experts

namespace {
// MiniMax's swigluoai (mm3_exl3.hpp's file comment): g = min(g, limit), u = clamp(u, -limit, limit),
// h = g * sigmoid(alpha g) * (u + 1); limit 0: no clamp.
__device__ __forceinline__ float oai(float g, float u, float alpha, float limit) {
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    return __fmul_rn(__fmul_rn(g, 1.0f / (1.0f + __expf(-__fmul_rn(alpha, g)))), u + 1.0f);
}

// One expert's blob -> its three GVQ1 payloads (+ residuals) + codebooks.
struct Blob {
    const uint8_t *w1, *w3, *w2, *r1, *r3, *r2, *cb, *cb2;
};
__device__ Blob blob_of(const ExpertLayout& lay, const void* base) {
    const uint8_t* b = (const uint8_t*) base;
    Blob o;
    o.w1 = b + lay.trellis[0];
    o.w3 = b + lay.trellis[1];
    o.w2 = b + lay.trellis[2];
    o.r1 = lay.vq_res[0] ? b + lay.vq_res[0] : nullptr;
    o.r3 = lay.vq_res[1] ? b + lay.vq_res[1] : nullptr;
    o.r2 = lay.vq_res[2] ? b + lay.vq_res[2] : nullptr;
    o.cb = b + lay.vq_cb;
    o.cb2 = lay.vq_cb2 ? b + lay.vq_cb2 : nullptr;
    return o;
}

// h[i][r] = oai(gate_i(x)[r], up_i(x)[r]) for the token's k experts; a warp a row of each expert.
__global__ void moe_gu_kernel(const unsigned long long* plan_ptr, const float* x, float* h, int k, int n_embd,
                              int n_ff, float alpha, float limit, ExpertLayout lay) {
    const int i = blockIdx.z;
    if (i >= k || plan_ptr[i] == 0) return;
    const Blob o = blob_of(lay, (const void*) plan_ptr[i]);
    const GvH hg = gv_header(o.w1);
#ifdef MM3_VQ_KERNEL_DEBUG
    if (threadIdx.x == 0 && (hg.rows != (uint32_t) n_ff || hg.cols != (uint32_t) n_embd ||
                             hg.high_off != 3545216ull || hg.index_off != 6272ull)) {
        printf("[gu-geo] i %d z%d y%d w1 %p rows %u (want %d) cols %u (want %d) hi %llu (want 3545216) "
               "idx %llu (want 6272) plan %llu t0 %zu -> pb %llu\n",
               i, (int) blockIdx.z, (int) blockIdx.y, (void*) o.w1, hg.rows, n_ff, hg.cols, n_embd,
               (unsigned long long) hg.high_off, (unsigned long long) hg.index_off, plan_ptr[i],
               lay.trellis[0], plan_ptr[i]);
        assert(false && "w1 header law");
    }
#endif
    const int cols = (int) hg.vecs();   // the wire's cols is k; the streams are per vector
    const Views vg = views(o.w1, o.r1), vu = views(o.w3, o.r3);
#ifdef MM3_VQ_KERNEL_DEBUG
    if (i == 2 && blockIdx.y % 256 == 0 && threadIdx.x == 0)   // (expert 2's blobs: the fault's home)
        printf("[gu2] y%d blob %llu w1 %p w2 %p r1 %p cb %p cols %d\n", (int) blockIdx.y,
               (unsigned long long) plan_ptr[i], (void*) o.w1, (void*) o.w2, (void*) o.r1, (void*) o.cb, cols);
    if (o.r1 != nullptr && threadIdx.x == 0)   // (a residual layout in play: the class the fault rides)
        printf("[gu] blk(%d,%d,%d) i %d row %d cols %d w1 %p w3 %p w2 %p r1 %p cb %p hg{%u,%u} vu.gain %p vu.low %p\n",
               (int) blockIdx.x, (int) blockIdx.y, (int) blockIdx.z, i,
               (int) (blockIdx.y * 4 + (threadIdx.x >> 5)), cols, (void*) o.w1, (void*) o.w3, (void*) o.w2,
               (void*) o.r1, (void*) o.cb, hg.rows, hg.cols, (void*) vu.gain, (void*) vu.low12);
#endif
    const int row = blockIdx.y * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_ff) return;
    const float gg = __half2float(__ushort_as_half(vg.gain[row]));
    const float gu = __half2float(__ushort_as_half(vu.gain[row]));
    float ag = 0.0f, au = 0.0f;
    for (int vv = lane; vv < cols; vv += 32) {
        // j is the flat VECTOR index: rows advance by `cols` vectors (hg.cols is VALUES, 8 a vector -
        // scaling by it walks 8x past every stream, the OOB the compute-sanitizer fault rode).
        const uint64_t j = (uint64_t) row * cols + vv;
        float wg[8], wu[8];
        w_vec(vg.low12, vg.high1, vg.idx8, o.cb, o.cb2, j, wg);
        w_vec(vu.low12, vu.high1, vu.idx8, o.cb, o.cb2, j, wu);
        const float* xc = x + 8 * vv;
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            ag = __fmaf_rn(wg[d], xc[d], ag);
            au = __fmaf_rn(wu[d], xc[d], au);
        }
    }
    #pragma unroll
    for (int o16 = 16; o16 > 0; o16 >>= 1) {
        ag += __shfl_down_sync(0xffffffffu, ag, o16);
        au += __shfl_down_sync(0xffffffffu, au, o16);
    }
    if (lane == 0) h[(size_t) i * n_ff + row] = oai(__fmul_rn(gg, ag), __fmul_rn(gu, au), alpha, limit);
}

// part[i][r] = down_i(h[i])[r]; a warp a row.
__global__ void moe_down_kernel(const unsigned long long* plan_ptr, const float* h, float* part, int k,
                                int n_embd, int n_ff, ExpertLayout lay) {
    const int i = blockIdx.z;
    if (i >= k || plan_ptr[i] == 0) return;
    const Blob o = blob_of(lay, (const void*) plan_ptr[i]);
    const GvH hd = gv_header(o.w2);
#ifdef MM3_VQ_KERNEL_DEBUG
    if (threadIdx.x == 0 && blockIdx.y % 1477 == 0)
        printf("[dn2] z%d y%d i %d plan %llu w2 %p rows %u cols %u\n", (int) blockIdx.z, (int) blockIdx.y,
               i, plan_ptr[i], (void*) o.w2, hd.rows, hd.cols);
    if (threadIdx.x == 0 &&
        (hd.rows != (uint32_t) n_embd || hd.cols != (uint32_t) n_ff ||
         hd.high_off != 3551360ull || hd.index_off != 12416ull))
        printf("[dn-geo] i %d z%d y%d w2 %p rows %u (want %d) cols %u (want %d) hi %llu (want 3551360) "
               "idx %llu (want 12416) plan %llu t2 %zu -> pb %llu\n",
               i, (int) blockIdx.z, (int) blockIdx.y, (void*) o.w2, hd.rows, n_embd, hd.cols, n_ff,
               (unsigned long long) hd.high_off, (unsigned long long) hd.index_off, plan_ptr[i],
               lay.trellis[2], plan_ptr[i]);
#endif
    const int cols = (int) hd.vecs();
    const Views vd = views(o.w2, o.r2);
    const int row = blockIdx.y * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_embd) return;
    const float g = __half2float(__ushort_as_half(vd.gain[row]));
    const float* hi = h + (size_t) i * n_ff;
    float acc = 0.0f;
    for (int vv = lane; vv < cols; vv += 32) {
        // the flat VECTOR index (see moe_gu_kernel - hd.cols is VALUES, not vectors)
        const uint64_t j = (uint64_t) row * cols + vv;
        float w[8];
        w_vec(vd.low12, vd.high1, vd.idx8, o.cb, o.cb2, j, w);
        const float* hc = hi + 8 * vv;
        #pragma unroll
        for (int d = 0; d < 8; ++d) acc = __fmaf_rn(w[d], hc[d], acc);
    }
    #pragma unroll
    for (int o16 = 16; o16 > 0; o16 >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o16);
    if (lane == 0) part[(size_t) i * n_embd + row] = __fmul_rn(g, acc);
}

// out[r] = sum_i w_i part_i[r] (+ cpu_part) (+ sh_out) - gg's Comb law.
__global__ void combine_kernel(const unsigned long long* plan_ptr, const float* plan_w, const float* part,
                               const float* cpu_part, const int* cpu_flag, const float* sh_out, float* out,
                               int k, int n_embd) {
    const int r = blockIdx.x * kF16Block + threadIdx.x;
    if (r >= n_embd) return;
    float m = 0.0f;
    for (int i = 0; i < k; ++i)
        if (plan_ptr[i] != 0) m = __fmaf_rn(plan_w[i], part[(size_t) i * n_embd + r], m);
    if (cpu_flag != nullptr && *cpu_flag) m = __fadd_rn(m, cpu_part[r]);
    out[r] = __fadd_rn(m, sh_out != nullptr ? sh_out[r] : 0.0f);
}
}   // namespace

namespace {
// (always on until first tokens: every blob a launch will touch must open GVQ1 - the win path's direct
// launches bypass moe_gate_up's law, so the check lives here for them too)
void check_blobs(const char* who, const unsigned long long* plan_d, int n, const ExpertLayout& lay,
                 cudaStream_t s) {
    const int tl = g_trace_layer;
    // the plan was written on the caller's stream (the same one the kernels run on) - syncing the default
    // stream here races that write and reads the PREVIOUS layer's plan (its plain blobs pass the GVQ1 law
    // by luck; the GVR1 law reads +vq_res[0], past a plain blob's end, and aborts on zeros)
    cudaStreamSynchronize(s);
    if (lay.vq_res[0] != 0) {   // (an R13+8 layout: every blob must carry GVR1 at +vq_res[0]; a plain blob
        unsigned long long pl2[8 * 16];   // mispaired here reads garbage-index header -> the idx8 read flies)
        if (cudaMemcpy(pl2, plan_d, sizeof(unsigned long long) * (size_t) n, cudaMemcpyDeviceToHost) == cudaSuccess)
            for (int i = 0; i < n; ++i) {
                if (pl2[i] == 0) continue;
                uint8_t rm[4] = {0, 0, 0, 0};
                const unsigned long long rp = pl2[i] + (unsigned long long) lay.vq_res[0];
                if (cudaMemcpy(rm, (const void*) rp, 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
                    !(rm[0] == 'G' && rm[1] == 'V' && rm[2] == 'R' && rm[3] == '1')) {
                    std::fprintf(stderr, "vq BAD RESIDUAL(%s) i%d ptr %llu res %llu magic %02x%02x%02x%02x\n",
                                 who, i, pl2[i], rp, rm[0], rm[1], rm[2], rm[3]);
                    std::fflush(stderr);
                    abort();
                }
            }
    }
    unsigned long long pl[8 * 16];   // (a win's [T][8] - the call passes T*k entries)
    if (cudaMemcpy(pl, plan_d, sizeof(unsigned long long) * (size_t) n, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "vq DEAD PLAN %s n %d\n", who, n);
        abort();
    }
    for (int i = 0; i < n; ++i) {
        const unsigned long long pb = pl[i];
        if (pb == 0) continue;
        uint8_t magic[4] = {0, 0, 0, 0};
        const unsigned long long w1p = pb + (unsigned long long) lay.trellis[0];   // (blob base = the codebook; GVQ1 sits at trellis[0])
        if (cudaMemcpy(magic, (const void*) w1p, 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
            !(magic[0] == 'G' && magic[1] == 'V' && magic[2] == 'Q' && magic[3] == '1')) {
            std::fprintf(stderr, "vq BAD BLOB(%s) L%d i%d ptr %llu magic %02x%02x%02x%02x "
                                 "t{%zu,%zu,%zu} res{%zu,%zu,%zu}\n",
                         who, tl, i, pb, magic[0], magic[1], magic[2], magic[3],
                         lay.trellis[0], lay.trellis[1], lay.trellis[2],
                         (size_t) lay.vq_res[0], (size_t) lay.vq_res[1], (size_t) lay.vq_res[2]);
            uint8_t dump[64] = {0};
            cudaMemcpy(dump, (const void*) pb, 64, cudaMemcpyDeviceToHost);
            for (int b = 0; b < 64; ++b) std::fprintf(stderr, "%02x", dump[b]);
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
            abort();
        }
    }
}
}   // namespace

void moe_gate_up(const MoeArgs& a_in, const float* x, float* h, cudaStream_t s) {
    static const bool no_res = ::mm3::env("MM3_VQ_NO_RES") != nullptr;   // (diagnostic: residuals as absent)
    MoeArgs a = a_in;
    if (no_res) a.lay.vq_res[0] = a.lay.vq_res[1] = a.lay.vq_res[2] = 0;
    if (!geometry("moe_gate_up", a.k, a.n_embd, a.n_ff)) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr || a.lay.fmt != kFmtVQ) {
        fail("moe_gate_up: not covered");
        return;
    }
    float* hh = h != nullptr ? h : sc->h;
    check_blobs("gate_up", a.plan_ptr, a.k, a.lay, s);   // (w1 + residual magic at every served blob)

    dim3 grid(1, (a.n_ff + 3) / 4, a.k);
    moe_gu_kernel<<<grid, kMvThreads, 0, s>>>(a.plan_ptr, x, hh, a.k, a.n_embd, a.n_ff, x3::kSwigluAlpha,
                                              a.limit, a.lay);
    check("moe_gate_up");
}

void moe_down(const MoeArgs& a_in, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    static const bool no_res = ::mm3::env("MM3_VQ_NO_RES") != nullptr;
    MoeArgs a = a_in;
    if (no_res) a.lay.vq_res[0] = a.lay.vq_res[1] = a.lay.vq_res[2] = 0;
    if (!geometry("moe_down", a.k, a.n_embd, a.n_ff)) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr || a.lay.fmt != kFmtVQ) {
        fail("moe_down: not covered");
        return;
    }
    check_blobs("down", a.plan_ptr, a.k, a.lay, s);
    const float* hh = h != nullptr ? h : sc->h;
    dim3 grid(1, (a.n_embd + 3) / 4, a.k);
    moe_down_kernel<<<grid, kMvThreads, 0, s>>>(a.plan_ptr, hh, sc->part, a.k, a.n_embd, a.n_ff, a.lay);
    combine_kernel<<<(a.n_embd + kF16Block - 1) / kF16Block, kF16Block, 0, s>>>(
        a.plan_ptr, a.plan_w, sc->part, a.cpu_part, a.cpu_flag, sh_out, out, a.k, a.n_embd);
    check("moe_down");
}



bool win_experts(const WinArgs& a_in, cudaStream_t s) {
    static const bool no_res = ::mm3::env("MM3_VQ_NO_RES") != nullptr;   // (diagnostic: covers the win path too)
    WinArgs a = a_in;
    if (no_res) a.lay.vq_res[0] = a.lay.vq_res[1] = a.lay.vq_res[2] = 0;
    if (a.T < 1 || !geometry("win_experts", a.k, a.n_embd, a.n_ff) || a.h == nullptr || a.out == nullptr) {
        fail("win_experts: not covered");
        return false;
    }
    check_blobs("win", a.plan_ptr, a.T * a.k, a.lay, s);
    Scratch* sc = scratch(s);
    if (sc == nullptr || a.lay.fmt != kFmtVQ) return false;
    static bool dbg = ::mm3::env("MM3_VQ_DEBUG") != nullptr;
    if (dbg) {
        unsigned long long pl[8];
        cudaMemcpy(pl, a.plan_ptr, sizeof(pl), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "vq win: T %d k %d plan {%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu}"
                             " t {%zu,%zu,%zu} res {%zu,%zu,%zu} cb %zu/%zu grp %d\n",
                     a.T, a.k, pl[0], pl[1], pl[2], pl[3], pl[4], pl[5], pl[6], pl[7],
                     a.lay.trellis[0], a.lay.trellis[1], a.lay.trellis[2],
                     (size_t) a.lay.vq_res[0], (size_t) a.lay.vq_res[1], (size_t) a.lay.vq_res[2],
                     (size_t) a.lay.vq_cb, (size_t) a.lay.vq_cb2, (int) (a.n_groups != nullptr));
        dbg = false;
    }
    // Phase A: each (token, expert) pair computes from its own plan slot - the grouping's re-read savings land
    // with the optimized kernel; every token's output is the per-token path's, bit for bit.
    for (int t = 0; t < a.T; ++t) {
        // WinArgs' plan is [T][8] (x3::kMaxMvCols' fixed width, whatever k is) - the stride is 8, not k
        const unsigned long long* plan_t = a.plan_ptr + (size_t) t * 8;
        const float* w_t = a.plan_w + (size_t) t * 8;
        dim3 gu(1, (a.n_ff + 3) / 4, a.k);
        moe_gu_kernel<<<gu, kMvThreads, 0, s>>>(plan_t, a.x + (size_t) t * (a.x_stride / 4),
                                                a.h + (size_t) t * a.k * a.n_ff, a.k, a.n_embd, a.n_ff,
                                                x3::kSwigluAlpha, a.limit, a.lay);
        dim3 dn(1, (a.n_embd + 3) / 4, a.k);
        moe_down_kernel<<<dn, kMvThreads, 0, s>>>(plan_t, a.h + (size_t) t * a.k * a.n_ff,
                                                  sc->part, a.k, a.n_embd, a.n_ff, a.lay);
        combine_kernel<<<(a.n_embd + kF16Block - 1) / kF16Block, kF16Block, 0, s>>>(
            plan_t, w_t, sc->part,
            a.cpu_part != nullptr ? a.cpu_part + (size_t) t * (a.cpu_stride / 4) : nullptr,
            a.cpu_flag != nullptr ? a.cpu_flag + t : nullptr,
            a.sh_out != nullptr ? a.sh_out + (size_t) t * a.n_embd : nullptr,
            a.out + (size_t) t * (a.out_stride / 4), a.k, a.n_embd);
    }
    check("win_experts");
    return true;
}

void set_gate_up(const SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s) {
    (void) a;
    // Phase A: per expert, per sorted row, decode gate/up rows directly (the mma-tile kernels are the
    // optimization pass; the outputs are the per-row law's).
    (void) x; (void) ldx; (void) row_idx; (void) h16; (void) ldh;
    fail("set_gate_up: phase B (the set kernels land with the prompt-path integration)");
}

void set_down(const SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s) {
    (void) a;
    fail("set_down: phase B");
}

void act_quant(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    // the GGUF lane's shared rounding (a CPU lane's parity needs it; the VQ math itself stays F32)
    gg::act_quant(x, rows, k, q, d, s);
}

}   // namespace mm3::vq
