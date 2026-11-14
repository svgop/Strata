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

#include "mm3_gguf.cuh"   # act_quant forwards to the GGUF lane's shared rounding

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
        v.idx8 = r + hr.index_off;   // (8-bit indices: the index region is the whole stream)
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
    const int cols = (int) h.cols;
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
        if (!host_header(m.raw, h, s) || h.cols * 8 != (uint32_t) m.k || h.rows != (uint32_t) m.n) {
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
    f16_rows_kernel<<<rows, kF16Block, 0, s>>>(m.raw, m.vq_res, (const uint8_t*) m.vq_cb,
                                               (const uint8_t*) m.vq_cb2, row0, rows, (int) h.cols, dst, ldx);
    check("f16_rows");
}

void gemm_rows(const Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
               cudaStream_t s) {
    if (rows <= 0) return;
    Scratch* sc = scratch(s);
    GvH h;
    if (sc == nullptr || m.fmt != kFmtVQ || m.raw == nullptr || m.vq_cb == nullptr ||
        !host_header(m.raw, h, s) || h.cols * 8 != (uint32_t) m.k) {
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
    const Views vg = views(o.w1, o.r1), vu = views(o.w3, o.r3);
    const int row = blockIdx.y * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_ff) return;
    const float gg = __half2float(__ushort_as_half(vg.gain[row]));
    const float gu = __half2float(__ushort_as_half(vu.gain[row]));
    float ag = 0.0f, au = 0.0f;
    for (int vv = lane; vv < (int) hg.cols; vv += 32) {
        const uint64_t j = (uint64_t) row * hg.cols + vv;
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
    const Views vd = views(o.w2, o.r2);
    const int row = blockIdx.y * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_embd) return;
    const float g = __half2float(__ushort_as_half(vd.gain[row]));
    const float* hi = h + (size_t) i * n_ff;
    float acc = 0.0f;
    for (int vv = lane; vv < (int) hd.cols; vv += 32) {
        const uint64_t j = (uint64_t) row * hd.cols + vv;
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

void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    if (!geometry("moe_gate_up", a.k, a.n_embd, a.n_ff)) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr || a.lay.fmt != kFmtVQ) {
        fail("moe_gate_up: not covered");
        return;
    }
    float* hh = h != nullptr ? h : sc->h;
    dim3 grid(1, (a.n_ff + 3) / 4, a.k);
    moe_gu_kernel<<<grid, kMvThreads, 0, s>>>(a.plan_ptr, x, hh, a.k, a.n_embd, a.n_ff, x3::kSwigluAlpha,
                                              a.limit, a.lay);
    check("moe_gate_up");
}

void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    if (!geometry("moe_down", a.k, a.n_embd, a.n_ff)) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr || a.lay.fmt != kFmtVQ) {
        fail("moe_down: not covered");
        return;
    }
    const float* hh = h != nullptr ? h : sc->h;
    dim3 grid(1, (a.n_embd + 3) / 4, a.k);
    moe_down_kernel<<<grid, kMvThreads, 0, s>>>(a.plan_ptr, hh, sc->part, a.k, a.n_embd, a.n_ff, a.lay);
    combine_kernel<<<(a.n_embd + kF16Block - 1) / kF16Block, kF16Block, 0, s>>>(
        a.plan_ptr, a.plan_w, sc->part, a.cpu_part, a.cpu_flag, sh_out, out, a.k, a.n_embd);
    check("moe_down");
}

bool win_experts(const WinArgs& a, cudaStream_t s) {
    if (a.T < 1 || !geometry("win_experts", a.k, a.n_embd, a.n_ff) || a.h == nullptr || a.out == nullptr) {
        fail("win_experts: not covered");
        return false;
    }
    Scratch* sc = scratch(s);
    if (sc == nullptr || a.lay.fmt != kFmtVQ) return false;
    // Phase A: each (token, expert) pair computes from its own plan slot - the grouping's re-read savings land
    // with the optimized kernel; every token's output is the per-token path's, bit for bit.
    for (int t = 0; t < a.T; ++t) {
        dim3 gu(1, (a.n_ff + 3) / 4, a.k);
        moe_gu_kernel<<<gu, kMvThreads, 0, s>>>(a.plan_ptr + (size_t) t * a.k, a.x + (size_t) t * (a.x_stride / 4),
                                                a.h + (size_t) t * a.k * a.n_ff, a.k, a.n_embd, a.n_ff,
                                                x3::kSwigluAlpha, a.limit, a.lay);
        dim3 dn(1, (a.n_embd + 3) / 4, a.k);
        moe_down_kernel<<<dn, kMvThreads, 0, s>>>(a.plan_ptr + (size_t) t * a.k, a.h + (size_t) t * a.k * a.n_ff,
                                                  sc->part, a.k, a.n_embd, a.n_ff, a.lay);
        combine_kernel<<<(a.n_embd + kF16Block - 1) / kF16Block, kF16Block, 0, s>>>(
            a.plan_ptr + (size_t) t * a.k, a.plan_w + (size_t) t * a.k, sc->part,
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
