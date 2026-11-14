// src/mm3/mm3_nvfp4.cu - NVFP4 expert weights on the device (mm3_nvfp4.cuh): the activations' rounding (mm3_gguf.cu's,
// so the CPU lane's sums are the same bits), the dense GEMVs, the prompt path's FP16-band GEMMs, the routed experts'
// passes and their combine.  The decode math is the vendored llama.cpp pairing (third_party/llama.cpp/ggml/src/
// ggml-cuda/vecdotq.cuh's vec_dot_nvfp4_q8_1, common.cuh's ue4m3_to_fp32): the E2M1 codebook as integers doubled and
// the scale byte as E4M3 halved, a 16-value sub-block's dot exact in integers through dp4a, one F32 sum a sub-block,
// the blob's F32 global multiplier at the row's end.
#include "mm3_nvfp4.cuh"

#include "mm3_gguf.cuh"   // (the activations' rounding: the experts' and the dense GEMVs' - the same bits both sides)

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace mm3::nf {

namespace {

std::atomic<int> g_errors{0};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mm3 nvfp4 %s: %s\n", what, cudaGetErrorString(e));
        g_errors.fetch_add(1, std::memory_order_relaxed);
    }
}
void fail(const char* what) {
    std::fprintf(stderr, "mm3 nvfp4 %s\n", what);
    g_errors.fetch_add(1, std::memory_order_relaxed);
}

constexpr int kWarps = 8, kThreads = kWarps * 32;
constexpr int kMaxEnt = 16;          // a window's tokens at most (mm3_model.cu's kWin)
constexpr int kMaxEmbd = 8192;       // n_embd and n_ff at most (MiniMax-M3.1: 6144, 3072)
constexpr int kMaxQK = 16384;        // a GEMV's inputs at most
constexpr int kWin = 1024;           // the set kernels' staging window (values)
constexpr size_t kBand = 64u << 20;  // a GEMM's FP16 weight band (bytes): the rows decoded at a time

// the E2M1 codebook, integers doubled (ggml-common.h's kvalues_fp4), four uint32 of int8 bytes: the byte_perm lookups
// below pick four decoded values a call, in the dp4a lane order the blob's Q4 nibble order gives
__constant__ unsigned kNF4T[4] = {0x03020100u, 0x0c080604u, 0xfdfeff00u, 0xf4f8fafcu};
// ... and the values themselves, halves (the FP16-band decode; sign bit 3, magnitude bits 0..2)
__constant__ float kNF4F[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

/// The scale byte as E4M3 halved (the vendored ue4m3_to_fp32: exp (b >> 3) & 15, man b & 7, exp 0: man * 2^-9, else
/// (1 + man / 8) * 2^(exp - 7), then the half - the doubled table's pair; bit 7 unused, 0 and the NaNs 0x7F / 0xFF
/// decode to 0).
__device__ __forceinline__ float nf_scale(unsigned b) {
    if (b == 0 || b == 0x7F || b == 0xFF) return 0.0f;
    const int e = (int) ((b >> 3) & 0xF), m = (int) (b & 7);
    const float raw = e == 0 ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
    return raw * 0.5f;
}

/// One 16-value sub-block: the decoded E2M1 halves (int8) dotted against four int32 words of the int8 activations,
/// exact.  The byte_perm trick is vecdotq.cuh's get_int_from_table_16 over the blob's plain Q4 nibble order (value c
/// in nibble c of the row), so the four dp4a operands come out consecutive - no regroup of theirs needed.
__device__ __forceinline__ int nf_dot16(unsigned wa, unsigned wb, const int8_t* q8) {
    unsigned tmp[4];
    const unsigned w[2] = {wa, wb};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        // the word's low nibbles decode values 4i .. 4i+3, the high ones 4i+4 .. 4i+7 (the blob's nibbles sit in
        // plain Q4 order - the lookups shift with the half, the regroup keeps the signs per nibble)
        const unsigned sel = 0x32103210u | ((w[i] & 0x88888888u) >> 1);
        const unsigned lo = __byte_perm(kNF4T[0], kNF4T[1], w[i]);
        const unsigned hi = __byte_perm(kNF4T[2], kNF4T[3], w[i]);
        const unsigned lo2 = __byte_perm(kNF4T[0], kNF4T[1], w[i] >> 16);
        const unsigned hi2 = __byte_perm(kNF4T[2], kNF4T[3], w[i] >> 16);
        tmp[2 * i + 0] = __byte_perm(lo, hi, sel & 0xffffu);
        tmp[2 * i + 1] = __byte_perm(lo2, hi2, sel >> 16);
    }
    int sum = 0;
    const int* q4 = (const int*) q8;
#pragma unroll
    for (int i = 0; i < 4; ++i) sum = __dp4a((int) tmp[i], q4[i], sum);
    return sum;
}

/// One weight row's dot against one int8 activation row (xq, a scale a 32 in xd): each lane walks the 16-value
/// sub-blocks lane, lane + 32, ... in order, the sub-blocks' F32 sums accumulated with fmaf, the warp's butterfly at
/// the end - the same sums on every path that calls it (mv, the experts' passes, the sets).
__device__ __forceinline__ float nf_dot_row(const uint8_t* nib, const uint8_t* sb, const int8_t* xq, const float* xd,
                                            int k, int lane) {
    const unsigned* nw = (const unsigned*) nib;
    float acc = 0.0f;
    for (int b = lane; b < k / 16; b += 32)
        acc = fmaf(nf_scale(sb[b]) * xd[b >> 1], (float) nf_dot16(nw[2 * b], nw[2 * b + 1], xq + (size_t) b * 16), acc);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

/// MiniMax's swigluoai (ds4_iq.cu's, mm3_exl3.hpp's kSwigluAlpha - the same operations).
constexpr float kAlpha = strata::mm3::exl3::kSwigluAlpha;
__device__ __forceinline__ float swigluoai(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(__fmul_rn(-kAlpha, gv)))), __fadd_rn(uv, 1.0f));
}

// one blob role's planes (trellis[r] = the matrix's offset; the header F32, the nibbles [n][k / 2], the scales
// [n][k / 16] - mm3_nvfp4.cuh's blob)
struct Role {
    const float* gs;     // the matrix's F32 multiplier (the blob header)
    const uint8_t* nib;  // the nibble plane's row 0
    const uint8_t* sb;   // the scale plane's row 0
};
__device__ __forceinline__ Role role_at(const uint8_t* blob, size_t off, int n_rows, int k) {
    Role r;
    r.gs = (const float*) (blob + off);
    r.nib = blob + off + 4;
    r.sb = r.nib + (size_t) n_rows * (size_t) (k / 2);
    return r;
}

// ---- the routed experts' sum (mm3_exl3.cu's combine, mm3_gguf.cu's copy): the experts in plan order (one off the
// device adds 0), then the CPU lane's part and the shared expert's
struct Comb {
    const unsigned long long* plan_ptr;   // [T][8]
    const float* plan_w;                  // [T][8]
    const float* part;                    // [T * 8][n_embd]
    const float* cpu_part;
    size_t cpu_stride;
    const int* cpu_flag;
    const float* sh_out;                  // [T][n_embd]
    float* out;
    size_t out_stride;
    int k, n_embd;
};
__global__ void __launch_bounds__(128) comb_kernel(const Comb a) {
    const int t = blockIdx.y, r = blockIdx.x * 128 + threadIdx.x;
    if (r >= a.n_embd) return;
    float m = 0.0f;
    for (int i = 0; i < a.k; ++i) {
        const int e = t * 8 + i;
        const float p = a.plan_ptr[e] != 0ull ? a.part[(size_t) e * a.n_embd + r] : 0.0f;
        m = __fmaf_rn(a.plan_w[e], p, m);
    }
    if (a.cpu_flag != nullptr && a.cpu_flag[t]) m = __fadd_rn(m, a.cpu_part[(size_t) t * a.cpu_stride + r]);
    a.out[(size_t) t * a.out_stride + r] =
        __fadd_rn(m, a.sh_out != nullptr ? a.sh_out[(size_t) t * a.n_embd + r] : 0.0f);
}
void combine(const unsigned long long* plan_ptr, const float* plan_w, const float* part, const float* cpu_part,
             size_t cpu_stride, const int* cpu_flag, const float* sh_out, float* out, size_t out_stride, int T, int k,
             int n_embd, cudaStream_t s) {
    const Comb c{plan_ptr, plan_w, part, cpu_part, cpu_stride, cpu_flag, sh_out, out, out_stride, k, n_embd};
    comb_kernel<<<dim3((unsigned) ((n_embd + 127) / 128), (unsigned) T), 128, 0, s>>>(c);
}

// ---- the device's wait for the host's word (the CPU lane's parts), as mm3_gguf.cu's: it gives up after 120 s
__device__ unsigned g_timeouts = 0;
__global__ void wait_kernel(const unsigned* p, unsigned v, unsigned long long limit) {
    if (threadIdx.x != 0) return;
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));   // (ns)
    while (*(volatile const unsigned*) p != v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 > limit) {
            atomicAdd(&g_timeouts, 1u);
            break;
        }
    }
    __threadfence_system();
}
unsigned long long wait_limit() { return 120ull * 1000000000ull; }

// ---- per-stream scratch
struct Scratch {
    float* part = nullptr;   // [8][kMaxEmbd]: a token's experts' down outputs
    float* h = nullptr;      // [8][kMaxEmbd]: a token's experts' swigluoai activations (when the caller gives none)
    int8_t *xq = nullptr, *hq = nullptr, *mq = nullptr;   // int8 rows: x [kMaxEnt], h [kMaxEnt * 8], the GEMVs'
    float *xd = nullptr, *hd = nullptr, *md = nullptr;    //   [8 * 16] inputs; their scales
    uint16_t* wb = nullptr;  // a GEMM's FP16 weight band (kBand bytes)
    cublasHandle_t blas = nullptr;
};
std::mutex g_mu;
std::unordered_map<cudaStream_t, Scratch> g_scratch;
constexpr size_t kNx = (size_t) kMaxEnt * kMaxEmbd, kNh = (size_t) kMaxEnt * 8 * kMaxEmbd,
                 kNm = (size_t) x3::kMaxMvJobs * x3::kMaxMvCols * kMaxQK;

Scratch* scratch(cudaStream_t s) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scratch.find(s);
    if (it != g_scratch.end()) return &it->second;
    Scratch sc;
    if (cudaMalloc(&sc.part, (size_t) 8 * kMaxEmbd * 4) != cudaSuccess ||
        cudaMalloc(&sc.h, (size_t) 8 * kMaxEmbd * 4) != cudaSuccess || cudaMalloc(&sc.xq, kNx) != cudaSuccess ||
        cudaMalloc(&sc.xd, kNx / 32 * 4) != cudaSuccess || cudaMalloc(&sc.hq, kNh) != cudaSuccess ||
        cudaMalloc(&sc.hd, kNh / 32 * 4) != cudaSuccess || cudaMalloc(&sc.mq, kNm) != cudaSuccess ||
        cudaMalloc(&sc.md, kNm / 32 * 4) != cudaSuccess || cudaMalloc(&sc.wb, kBand) != cudaSuccess) {
        cudaGetLastError();
        fail("scratch: out of device memory");
        return nullptr;
    }
    if (cublasCreate(&sc.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(sc.blas, s) != CUBLAS_STATUS_SUCCESS) {
        fail("scratch: cuBLAS");
        return nullptr;
    }
    return &g_scratch.emplace(s, sc).first->second;
}

bool geometry(const char* who, int k, int n_embd, int n_ff) {
    if (k < 1 || k > 8 || n_embd % 32 != 0 || n_ff % 32 != 0 || n_embd > kMaxEmbd || n_ff > kMaxEmbd) {
        std::fprintf(stderr, "mm3 nvfp4 %s: geometry not covered (k %d n_embd %d n_ff %d)\n", who, k, n_embd, n_ff);
        g_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

/// A whole fmt-6 matrix (ld != 0 is refused: no views yet - mm3_nvfp4.cuh's blob).
bool mat_ok(const char* who, const x3::Mat& m) {
    if (m.fmt != kFmtNVFP4 || m.raw == nullptr || m.ld != 0 || m.k < 16 || m.k % 16 != 0 || m.k > kMaxQK ||
        ((uintptr_t) m.raw & 3) != 0) {
        std::fprintf(stderr, "mm3 nvfp4 %s: not a whole NVFP4 matrix (k %d n %d ld %d)\n", who, m.k, m.n, m.ld);
        g_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

bool lay_ok(const char* who, const x3::ExpertLayout& lay) {
    if (lay.fmt != kFmtNVFP4) {
        fail(who);
        return false;
    }
    for (int r = 0; r < 3; ++r)
        if ((lay.trellis[r] & 3) != 0) {
            std::fprintf(stderr, "mm3 nvfp4 %s: role %d's offset not 4-byte aligned\n", who, r);
            g_errors.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    return true;
}

}  // namespace

// ---------------------------------------------------------------- the dense GEMVs
namespace {

struct MvDev {
    const uint8_t* nib;   // the Mat's planes (raw's header skipped)
    const uint8_t* sb;
    const float* gs;
    const int8_t* xq;    // [ncol][k] (the scratch's rows)
    const float* xd;     // [ncol][k / 32]
    float* y;
    const float* bias;
    float alpha;
    int k, n, ncol;
    size_t ycs;
};
struct MvArgs {
    MvDev j[x3::kMaxMvJobs];
    int n;
};
__global__ void __launch_bounds__(kThreads) mv_kernel(const MvArgs a) {
    const int jj = blockIdx.y;
    if (jj >= a.n) return;
    const MvDev& d = a.j[jj];
    const int row = blockIdx.x * kWarps + (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (row >= d.n) return;
    const uint8_t* nib = d.nib + (size_t) row * (size_t) (d.k >> 1);
    const uint8_t* sb = d.sb + (size_t) row * (size_t) (d.k / 16);
    for (int c = 0; c < d.ncol; ++c) {
        const float acc = nf_dot_row(nib, sb, d.xq + (size_t) c * d.k, d.xd + (size_t) c * (d.k / 32), d.k, lane);
        if (lane == 0)
            d.y[(size_t) c * d.ycs + row] =
                __fmaf_rn(d.alpha, __fmul_rn(*d.gs, acc), d.bias != nullptr ? d.bias[row] : 0.0f);
    }
}

}  // namespace

bool mv(const x3::MvJob* jobs, int n, cudaStream_t s) {
    if (n <= 0 || n > x3::kMaxMvJobs) return false;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    MvArgs A;
    A.n = n;
    int maxn = 0;
    for (int i = 0; i < n; ++i) {
        const x3::MvJob& q = jobs[i];
        const x3::Mat& m = *q.m;
        if (q.xf == nullptr || q.y == nullptr || q.ncol < 1 || q.ncol > x3::kMaxMvCols || m.k % 32 != 0 ||
            !mat_ok("mv", m)) {
            std::fprintf(stderr, "mm3 nvfp4 mv: job %d not covered (%d activations, xf %d y %d)\n", i, q.ncol,
                         q.xf != nullptr, q.y != nullptr);
            g_errors.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        size_t at = (size_t) i * x3::kMaxMvCols * kMaxQK;
        // (a job over the same activations as an earlier one - q / k / v, the shared expert's gate / up - takes that
        //  one's int8 rows: the same values, one rounding launch instead of several)
        int same = -1;
        for (int j = 0; j < i && same < 0; ++j)
            if (jobs[j].xf == q.xf && jobs[j].ncol == q.ncol && jobs[j].m->k == q.m->k &&
                (q.ncol == 1 || jobs[j].xf_cs == q.xf_cs))
                same = j;
        if (same >= 0) at = (size_t) same * x3::kMaxMvCols * kMaxQK;
        else
            mm3::gg::q8_quant(q.xf, q.ncol > 1 ? q.xf_cs : (size_t) m.k, q.ncol, m.k, sc->mq + at, sc->md + at / 32, s);
        MvDev& d = A.j[i];
        d.nib = (const uint8_t*) m.raw + 4;
        d.sb = d.nib + (size_t) m.n * (size_t) (m.k >> 1);
        d.gs = (const float*) m.raw;
        d.xq = sc->mq + at;
        d.xd = sc->md + at / 32;
        d.y = q.y;
        d.bias = q.bias;
        d.alpha = q.alpha;
        d.k = m.k;
        d.n = m.n;
        d.ncol = q.ncol;
        d.ycs = q.ncol > 1 ? q.y_cs : (size_t) m.n;
        maxn = std::max(maxn, m.n);
    }
    mv_kernel<<<dim3((unsigned) ((maxn + kWarps - 1) / kWarps), (unsigned) n), kThreads, 0, s>>>(A);
    check("mv");
    return true;
}

// ---------------------------------------------------------------- the prompt path's GEMMs
namespace {
// thread per (band row, 16-value block): the band's rows (the outputs) decoded once to FP16 - the product in double,
// the half at the end, the reference's bits
__global__ void __launch_bounds__(256) f16_rows_kernel(const uint8_t* nib0, const uint8_t* sb0, int k,
                                                       const float* gs, uint16_t* w, int ldw) {
    const int r = blockIdx.y, b = blockIdx.x * 256 + threadIdx.x;
    if (b * 16 >= k) return;
    const unsigned* nw = (const unsigned*) (nib0 + (size_t) r * (size_t) (k >> 1));
    const unsigned wa = nw[2 * b], wb = nw[2 * b + 1];
    const double sc2 = (double) nf_scale(sb0[(size_t) r * (size_t) (k / 16) + b]) * 2.0 * (double) *gs;
    uint16_t h[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned byte = (i < 4 ? (wa >> (8 * i)) : (wb >> (8 * (i - 4)))) & 0xFF;
        h[2 * i] = __half_as_ushort(__float2half_rn((float) ((double) kNF4F[byte & 0xF] * sc2)));
        h[2 * i + 1] = __half_as_ushort(__float2half_rn((float) ((double) kNF4F[byte >> 4] * sc2)));
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) ((uint2 *) (w + (size_t) r * ldw + b * 16))[i] = ((const uint2 *) h)[i];
}
}  // namespace

void gemm_rows(const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s) {
    if (rows <= 0) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr || !mat_ok("gemm_rows", m)) return;
    // the weight's rows (the outputs) in bands of the FP16 scratch, each decoded once and multiplied by every row
    const int band = (int) std::max<size_t>(256, kBand / ((size_t) m.k * 2) / 256 * 256);
    const float one = 1.0f;
    for (int c0 = 0; c0 < m.n; c0 += band) {
        const int nc = std::min(band, m.n - c0);
        f16_rows_kernel<<<dim3((unsigned) ((m.k / 16 + 255) / 256), (unsigned) nc), 256, 0, s>>>(
            (const uint8_t*) m.raw + 4 + (size_t) c0 * (m.k / 2),
            (const uint8_t*) m.raw + 4 + (size_t) m.n * (m.k / 2) + (size_t) c0 * (m.k / 16), m.k,
            (const float*) m.raw, sc->wb, m.k);
        // Y^T [nc][rows] (column-major, ldy) = W [nc][k] (row-major: A^T) x X^T [k][rows] (ldx)
        if (cublasGemmEx(sc->blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, m.k, &one, sc->wb, CUDA_R_16F, m.k, x16,
                         CUDA_R_16F, ldx, &beta, y + c0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
            fail("gemm_rows: cublasGemmEx");
            return;
        }
    }
    check("gemm_rows");
}

// ---------------------------------------------------------------- the routed experts
namespace {

// the gate / up pass: blob g of gridDim.y (the plan's expert on the token path; a window's group past *n_groups
// returns), each row of gate and up over the int8 activations, h[e][row] the F32 swigluoai (the token path's e = the
// plan index; a window's e = the entry = token * 8 + pick)
struct GUArgs {
    const unsigned long long* plan;    // the token path's [8] (grp null)
    const unsigned long long* grp;     // a window's [groups]
    const int *grp_start, *grp_ent, *n_groups;
    const int8_t* xq;                  // [T][n_embd] (the scratch's rows)
    const float* xd;
    float* h;                          // [entries][n_ff]
    size_t trg, tru;                   // gate and up's offsets in a blob
    int n_embd, n_ff;
    float limit;
};
__global__ void __launch_bounds__(kThreads) gu_kernel(const GUArgs a) {
    const int g = blockIdx.y;
    const uint8_t* blob;
    int e0 = 0, e1 = 1;
    if (a.grp != nullptr) {
        if (g >= *a.n_groups) return;
        blob = (const uint8_t*) a.grp[g];
        e0 = a.grp_start[g];
        e1 = a.grp_start[g + 1];
    } else {
        const unsigned long long p = a.plan[g];
        if (p == 0ull) return;
        blob = (const uint8_t*) p;
        e0 = g;
        e1 = g + 1;
    }
    const int row = blockIdx.x * kWarps + (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (row >= a.n_ff) return;
    const Role rg = role_at(blob, a.trg, a.n_ff, a.n_embd);
    const Role ru = role_at(blob, a.tru, a.n_ff, a.n_embd);
    const uint8_t* gn = rg.nib + (size_t) row * (size_t) (a.n_embd / 2);
    const uint8_t* gs_ = rg.sb + (size_t) row * (size_t) (a.n_embd / 16);
    const uint8_t* un = ru.nib + (size_t) row * (size_t) (a.n_embd / 2);
    const uint8_t* us = ru.sb + (size_t) row * (size_t) (a.n_embd / 16);
    for (int e = e0; e < e1; ++e) {
        const int t = a.grp_ent != nullptr ? (a.grp_ent[e] >> 3) : 0;
        const int8_t* xq = a.xq + (size_t) t * a.n_embd;
        const float* xd = a.xd + (size_t) t * (a.n_embd / 32);
        const float gv = nf_dot_row(gn, gs_, xq, xd, a.n_embd, lane);
        const float uv = nf_dot_row(un, us, xq, xd, a.n_embd, lane);
        if (lane == 0)
            a.h[(size_t) (a.grp_ent != nullptr ? a.grp_ent[e] : e) * a.n_ff + row] =
                swigluoai(*rg.gs * gv, *ru.gs * uv, a.limit);
    }
}

// the down pass: part[e][row] = the down row's dot over entry e's int8 activations (the swigluoai rows, quantized
// outside), times the role's F32
struct DNArgs {
    const unsigned long long* plan;
    const unsigned long long* grp;
    const int *grp_start, *grp_ent, *n_groups;
    const int8_t* hq;    // [entries][n_ff]
    const float* hd;
    float* part;         // [entries][n_embd]
    size_t trd;
    int n_embd, n_ff;
};
__global__ void __launch_bounds__(kThreads) dn_kernel(const DNArgs a) {
    const int g = blockIdx.y;
    const uint8_t* blob;
    int e0 = 0, e1 = 1;
    if (a.grp != nullptr) {
        if (g >= *a.n_groups) return;
        blob = (const uint8_t*) a.grp[g];
        e0 = a.grp_start[g];
        e1 = a.grp_start[g + 1];
    } else {
        const unsigned long long p = a.plan[g];
        if (p == 0ull) return;
        blob = (const uint8_t*) p;
        e0 = g;
        e1 = g + 1;
    }
    const int row = blockIdx.x * kWarps + (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (row >= a.n_embd) return;
    const Role rd = role_at(blob, a.trd, a.n_embd, a.n_ff);
    const uint8_t* dn = rd.nib + (size_t) row * (size_t) (a.n_ff / 2);
    const uint8_t* ds = rd.sb + (size_t) row * (size_t) (a.n_ff / 16);
    for (int e = e0; e < e1; ++e) {
        const int er = a.grp_ent != nullptr ? a.grp_ent[e] : e;
        const float acc = nf_dot_row(dn, ds, a.hq + (size_t) er * a.n_ff, a.hd + (size_t) er * (a.n_ff / 32), a.n_ff,
                                     lane);
        if (lane == 0) a.part[(size_t) er * a.n_embd + row] = *rd.gs * acc;
    }
}

}  // namespace

void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    if (a.mix.n > 0) {
        fail("moe_gate_up: mixed classes not covered (one NVFP4 layout a layer)");
        return;
    }
    if (!geometry("moe_gate_up", a.k, a.n_embd, a.n_ff) || !lay_ok("moe_gate_up", a.lay)) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return;
    mm3::gg::act_quant(x, 1, a.n_embd, sc->xq, sc->xd, s);
    GUArgs gu;
    gu.plan = a.plan_ptr;
    gu.grp = nullptr;
    gu.grp_start = gu.grp_ent = gu.n_groups = nullptr;
    gu.xq = sc->xq;
    gu.xd = sc->xd;
    gu.h = h != nullptr ? h : sc->h;
    gu.trg = a.lay.trellis[0];
    gu.tru = a.lay.trellis[1];
    gu.n_embd = a.n_embd;
    gu.n_ff = a.n_ff;
    gu.limit = a.limit;
    gu_kernel<<<dim3((unsigned) (a.n_ff / kWarps), (unsigned) a.k), kThreads, 0, s>>>(gu);
    check("moe_gate_up");
}

void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    if (a.mix.n > 0) {
        fail("moe_down: mixed classes not covered (one NVFP4 layout a layer)");
        return;
    }
    if (!geometry("moe_down", a.k, a.n_embd, a.n_ff) || !lay_ok("moe_down", a.lay)) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return;
    mm3::gg::act_quant(h != nullptr ? h : sc->h, a.k, a.n_ff, sc->hq, sc->hd, s);
    DNArgs dn;
    dn.plan = a.plan_ptr;
    dn.grp = nullptr;
    dn.grp_start = dn.grp_ent = dn.n_groups = nullptr;
    dn.hq = sc->hq;
    dn.hd = sc->hd;
    dn.part = sc->part;
    dn.trd = a.lay.trellis[2];
    dn.n_embd = a.n_embd;
    dn.n_ff = a.n_ff;
    dn_kernel<<<dim3((unsigned) (a.n_embd / kWarps), (unsigned) a.k), kThreads, 0, s>>>(dn);
    combine(a.plan_ptr, a.plan_w, sc->part, a.cpu_part, (size_t) a.n_embd, a.cpu_flag, sh_out, out, (size_t) a.n_embd,
            1, a.k, a.n_embd, s);
    check("moe_down");
}

bool win_experts(const x3::WinArgs& a, cudaStream_t s) {
    if (a.T < 1 || a.T > kMaxEnt || a.h == nullptr || a.part == nullptr || a.out == nullptr ||
        !geometry("win_experts", a.k, a.n_embd, a.n_ff) || !lay_ok("win_experts", a.lay)) {
        std::fprintf(stderr, "mm3 nvfp4 win_experts: not covered (T %d)\n", a.T);
        return false;
    }
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    const int groups = a.T * a.k;   // at most (the blocks past *n_groups return at once)
    mm3::gg::act_quant(a.x, a.T, a.n_embd, sc->xq, sc->xd, s);
    GUArgs gu;
    gu.plan = nullptr;
    gu.grp = a.grp_ptr;
    gu.grp_start = a.grp_start;
    gu.grp_ent = a.grp_ent;
    gu.n_groups = a.n_groups;
    gu.xq = sc->xq;
    gu.xd = sc->xd;
    gu.h = a.h;
    gu.trg = a.lay.trellis[0];
    gu.tru = a.lay.trellis[1];
    gu.n_embd = a.n_embd;
    gu.n_ff = a.n_ff;
    gu.limit = a.limit;
    gu_kernel<<<dim3((unsigned) (a.n_ff / kWarps), (unsigned) groups), kThreads, 0, s>>>(gu);
    mm3::gg::act_quant(a.h, a.T * 8, a.n_ff, sc->hq, sc->hd, s);   // (the rows of picks not on the device: unused)
    DNArgs dn;
    dn.plan = nullptr;
    dn.grp = a.grp_ptr;
    dn.grp_start = a.grp_start;
    dn.grp_ent = a.grp_ent;
    dn.n_groups = a.n_groups;
    dn.hq = sc->hq;
    dn.hd = sc->hd;
    dn.part = a.part;
    dn.trd = a.lay.trellis[2];
    dn.n_embd = a.n_embd;
    dn.n_ff = a.n_ff;
    dn_kernel<<<dim3((unsigned) (a.n_embd / kWarps), (unsigned) groups), kThreads, 0, s>>>(dn);
    if (a.wait_ptr != nullptr) wait_kernel<<<1, 32, 0, s>>>(a.wait_ptr, a.wait_val, wait_limit());
    combine(a.plan_ptr, a.plan_w, a.part, a.cpu_part, a.cpu_stride != 0 ? a.cpu_stride : (size_t) a.n_embd, a.cpu_flag,
            a.sh_out, a.out, a.out_stride, a.T, a.k, a.n_embd, s);
    check("win_experts");
    return true;
}

// ---------------------------------------------------------------- the prompt sets
namespace {

// one warp a weight row of one expert: the warp loops its expert's local rows, staging each row's activations (or the
// FP16 rows' values) in shared as int8 a 32 and dotting its own weight row's sub-blocks over each window - the same
// per-lane sub-block order as nf_dot_row, the sums the token path's order too.  No cross-warp traffic: __syncwarp only.
__global__ void __launch_bounds__(kThreads) set_gu_kernel(const x3::SetArgs a, const float* x, int ldx,
                                                          const int* row_idx, uint16_t* h16, int ldh) {
    __shared__ int8_t sq[kWarps][kWin];
    __shared__ float sd[kWarps][kWin / 32];
    const int e = blockIdx.y, ff = blockIdx.x * kWarps + (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31),
              w = (int) (threadIdx.x >> 5);
    if (ff >= a.n_ff) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    if (lo >= hi) return;
    const Role rg = role_at(blob, a.lay.trellis[0], a.n_ff, a.n_embd);
    const Role ru = role_at(blob, a.lay.trellis[1], a.n_ff, a.n_embd);
    const uint8_t* gn = rg.nib + (size_t) ff * (size_t) (a.n_embd / 2);
    const uint8_t* gs_ = rg.sb + (size_t) ff * (size_t) (a.n_embd / 16);
    const uint8_t* un = ru.nib + (size_t) ff * (size_t) (a.n_embd / 2);
    const uint8_t* us = ru.sb + (size_t) ff * (size_t) (a.n_embd / 16);
    for (int lr = lo; lr < hi; ++lr) {
        const int src = row_idx[lr];
        float ag = 0.0f, au = 0.0f;
        for (int w0 = 0; w0 < a.n_embd; w0 += kWin) {
            const int wn = kWin < a.n_embd - w0 ? kWin : a.n_embd - w0, ng = wn / 32;
            if (lane < ng) {
                const float* xr = x + (size_t) src * ldx + w0 + lane * 32;
                float am = 0.0f;
                for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(xr[j]));
                am /= 127.0f;
                for (int j = 0; j < 32; ++j)
                    sq[w][lane * 32 + j] = (int8_t) (am > 0.0f ? __float2int_rn(xr[j] / am) : 0);
                sd[w][lane] = am;
            }
            __syncwarp();
            const unsigned* gw = (const unsigned*) (gn + (size_t) w0 / 2), *uw = (const unsigned*) (un + (size_t) w0 / 2);
            for (int b = lane; b < wn / 16; b += 32) {
                ag = fmaf(nf_scale(gs_[(size_t) w0 / 16 + b]) * sd[w][b >> 1],
                          (float) nf_dot16(gw[2 * b], gw[2 * b + 1], sq[w] + b * 16), ag);
                au = fmaf(nf_scale(us[(size_t) w0 / 16 + b]) * sd[w][b >> 1],
                          (float) nf_dot16(uw[2 * b], uw[2 * b + 1], sq[w] + b * 16), au);
            }
            __syncwarp();
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            ag += __shfl_xor_sync(0xffffffffu, ag, o);
            au += __shfl_xor_sync(0xffffffffu, au, o);
        }
        if (lane == 0) {
            h16[(size_t) lr * ldh + ff] =
                __half_as_ushort(__float2half_rn(swigluoai(*rg.gs * ag, *ru.gs * au, a.limit)));
        }
    }
}

__global__ void __launch_bounds__(kThreads) set_dn_kernel(const x3::SetArgs a, const uint16_t* h16, int ldh, float* y,
                                                          int ldy) {
    __shared__ int8_t sq[kWarps][kWin];
    __shared__ float sd[kWarps][kWin / 32];
    const int e = blockIdx.y, ff = blockIdx.x * kWarps + (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31),
              w = (int) (threadIdx.x >> 5);
    if (ff >= a.n_embd) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    if (lo >= hi) return;
    const Role rd = role_at(blob, a.lay.trellis[2], a.n_embd, a.n_ff);
    const uint8_t* dn = rd.nib + (size_t) ff * (size_t) (a.n_ff / 2);
    const uint8_t* ds = rd.sb + (size_t) ff * (size_t) (a.n_ff / 16);
    for (int lr = lo; lr < hi; ++lr) {
        float acc = 0.0f;
        for (int w0 = 0; w0 < a.n_ff; w0 += kWin) {
            const int wn = kWin < a.n_ff - w0 ? kWin : a.n_ff - w0, ng = wn / 32;
            if (lane < ng) {
                const uint16_t* hr = h16 + (size_t) lr * ldh + w0 + lane * 32;
                float am = 0.0f;
                for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(__half2float(*(const __half*) (hr + j))));
                am /= 127.0f;
                for (int j = 0; j < 32; ++j)
                    sq[w][lane * 32 + j] =
                        (int8_t) (am > 0.0f ? __float2int_rn(__half2float(*(const __half*) (hr + j)) / am) : 0);
                sd[w][lane] = am;
            }
            __syncwarp();
            const unsigned* dw = (const unsigned*) (dn + (size_t) w0 / 2);
            for (int b = lane; b < wn / 16; b += 32)
                acc = fmaf(nf_scale(ds[(size_t) w0 / 16 + b]) * sd[w][b >> 1],
                           (float) nf_dot16(dw[2 * b], dw[2 * b + 1], sq[w] + b * 16), acc);
            __syncwarp();
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[(size_t) lr * ldy + ff] = *rd.gs * acc;
    }
}

}  // namespace

void set_gate_up(const x3::SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s) {
    if (a.n_exp <= 0 || a.max_rows <= 0) return;
    if (a.mix.n > 0) {
        fail("set_gate_up: mixed classes not covered (one NVFP4 layout a layer)");
        return;
    }
    if (!geometry("set_gate_up", 1, a.n_embd, a.n_ff) || !lay_ok("set_gate_up", a.lay) || a.wbase == nullptr ||
        a.bounds == nullptr)
        return;
    set_gu_kernel<<<dim3((unsigned) ((a.n_ff + kWarps - 1) / kWarps), (unsigned) a.n_exp), kThreads, 0, s>>>(
        a, x, ldx, row_idx, h16, ldh);
    check("set_gate_up");
}

void set_down(const x3::SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s) {
    if (a.n_exp <= 0 || a.max_rows <= 0) return;
    if (a.mix.n > 0) {
        fail("set_down: mixed classes not covered (one NVFP4 layout a layer)");
        return;
    }
    if (!geometry("set_down", 1, a.n_embd, a.n_ff) || !lay_ok("set_down", a.lay) || a.wbase == nullptr ||
        a.bounds == nullptr)
        return;
    set_dn_kernel<<<dim3((unsigned) ((a.n_embd + kWarps - 1) / kWarps), (unsigned) a.n_exp), kThreads, 0, s>>>(
        a, h16, ldh, y, ldy);
    check("set_down");
}

// ---------------------------------------------------------------- the family's counters
bool prepare(cudaStream_t s) { return scratch(s) != nullptr; }

size_t scratch_bytes() {
    const size_t n8 = kNx + kNh + kNm;
    return 2 * (size_t) 8 * kMaxEmbd * 4 + n8 + n8 / 32 * 4 + kBand + (16u << 20);   // (+ cuBLAS' workspace)
}

int launch_errors() { return g_errors.load(std::memory_order_relaxed); }

unsigned wait_timeouts() {
    unsigned n = 0;
    if (cudaMemcpyFromSymbol(&n, g_timeouts, sizeof n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

}  // namespace mm3::nf
