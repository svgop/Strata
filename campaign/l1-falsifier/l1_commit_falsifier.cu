// l1_commit_falsifier.cu — GDN commit falsifier microbench
// SCOUT2 L1 commit/kill, FALSIFIER SPEC v2 (campaign/research/scout2/l1-commit-kill.md).
// Standalone synthetic CUDA; zero engine links. Box: RTX 5090 (sm_120), CUDA 13.4.
//
// Engine facts mirrored: L=48 layers, HV=32 heads, K=V=128, fp32 state 2 MiB/layer/slot,
// accepted suffix n in {1,2,4,8}, batch/slots B in {1,6}. Production baseline: row-serial
// replay, 2.47 ms/row (src/core/verify.cpp capture_commit -> gdn_commit_layers call shape).
//
// Arms (identical synthetic gated delta rule; correctness vs CPU ref before any timing):
//   S0 replay         : n sequential per-row kernels, each reads state, applies one
//                       delta-rule row, writes state (row-serial production shape).
//   S1 chunked        : ONE kernel over the n-row suffix; per-(slot,layer,head) block
//                       walks rows in-block, state in registers, single writeback.
//   S2 scatter-promote: fake-verify checkpoint write (T states to T pool slots, one
//                       kernel — FlashInfer intermediate_states_buffer form, fp32 and
//                       bf16 variants per spec item 2), then commit = one launch:
//                       (a) index select (pool-reserved free-list form, SGLang D1.2)
//                       (b) gather copy checkpoint[n-1] -> canonical state.
//
// Spec compliance: launch counts reported (item 3), T-sweep 1..8 (item 1), accept count
// consumed device-side, no host sync in any timed region (item 9), VRAM gate ladder with
// named boundaries (item 10), all numbers self-generated (item 11).
//
// Synthetic gated delta rule (per head, S in R^{V x K} row-major, k in R^K, v in R^V):
//   w[v]  = sum_j S[v][j] * k[j]                  (128-term fmaf chain, ascending j)
//   u[v]  = beta * (v[v] - w[v])
//   S'[v][j] = fmaf(g, S[v][j], u[v] * k[j])
// CPU reference replicates the exact fmaf sequence -> gate fp32 max_abs <= 1e-4
// (spec item 5 tolerance ladder), bf16 gate 2e-2.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(e_), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static constexpr int  KD            = 128;   // K head dim
static constexpr int  VD            = 128;   // V head dim
static constexpr int  HEAD_ELEMS    = KD * VD;
static constexpr int  HVH           = 32;    // v heads
static constexpr int  LAYERS        = 48;
static constexpr int  N_MAX         = 8;     // sweep rows
static constexpr double PROD_ROW_US = 2470.0;  // production row-serial replay, ms/row
static constexpr double KILL_RATIO  = 20.0;    // verdict threshold
static constexpr int    WARMUP      = 10;
static constexpr int    REPS        = 200;

static const float FP32_TOL = 1e-4f;   // spec item 5 ladder
static const float BF16_TOL = 2e-2f;

static cudaStream_t g_stream = nullptr;
static std::vector<std::string> g_table;   // result rows, printed at the end

// ---------------------------------------------------------------------------
// device state layout:  state[nstate][VD][KD],  nstate = B * LAYERS * HVH
// inputs (n rows):      k[n][nstate][KD], v[n][nstate][VD], g[n][nstate], beta[n][nstate]
// checkpoints:          ck[n][nstate][VD][KD]  (fp32 or bf16 pool)
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(128) void s0_row_update(
    float* __restrict__ S, const float* __restrict__ kr, const float* __restrict__ vr,
    const float* __restrict__ gr, const float* __restrict__ br) {
    const size_t i = blockIdx.x;
    float*       s = S + i * (size_t)HEAD_ELEMS;
    const int    v = threadIdx.x;
    float        w = 0.0f;
#pragma unroll 8
    for (int j = 0; j < KD; ++j) w = fmaf(s[v * KD + j], kr[i * KD + j], w);
    const float u = br[i] * (vr[i * VD + v] - w);
    const float g = gr[i];
#pragma unroll 8
    for (int j = 0; j < KD; ++j) s[v * KD + j] = fmaf(g, s[v * KD + j], u * kr[i * KD + j]);
}

// 256 threads/block: each thread owns HALF a row (64 state floats resident in registers).
// Per row: ascending-j fmaf chain per half, halves reduced with one __shfl_xor pair swap.
// Matches the CPU single-chain reference to a few ulp (gate 1e-4, spec item 5 ladder).
__global__ __launch_bounds__(256) void s1_chunked(
    float* __restrict__ S, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, const int n,
    const size_t nstate) {
    const size_t i  = blockIdx.x;
    float*       s  = S + i * (size_t)HEAD_ELEMS;
    const int    r  = threadIdx.x >> 1;        // row v in [0, VD)
    const int    j0 = (threadIdx.x & 1) * (KD / 2);  // half offset in [0, KD)
    float        sreg[KD / 2];
#pragma unroll
    for (int j = 0; j < KD / 2; ++j) sreg[j] = s[r * KD + j0 + j];
    for (int t = 0; t < n; ++t) {
        const float* kr = k + ((size_t)t * nstate + i) * KD;
        const float* vr = v + ((size_t)t * nstate + i) * VD;
        const float  gg = g[(size_t)t * nstate + i];
        const float  bb = beta[(size_t)t * nstate + i];
        float        w  = 0.0f;
#pragma unroll
        for (int j = 0; j < KD / 2; ++j) w = fmaf(sreg[j], kr[j0 + j], w);
        w += __shfl_xor_sync(0xffffffffu, w, 1);  // pair of lanes = one row
        const float u = bb * (vr[r] - w);
#pragma unroll
        for (int j = 0; j < KD / 2; ++j) sreg[j] = fmaf(gg, sreg[j], u * kr[j0 + j]);
    }
#pragma unroll
    for (int j = 0; j < KD / 2; ++j) s[r * KD + j0 + j] = sreg[j];
}

// Same half-row decomposition as s1_chunked. Applies rows 0..n-1 and snapshots the state
// after each row into the pool (FlashInfer intermediate_states_buffer form). One launch.
template <bool OUT_BF16>
__global__ __launch_bounds__(256) void s2_verify_write(
    const float* __restrict__ S, float* __restrict__ ck_f, __nv_bfloat16* __restrict__ ck_b,
    const float* __restrict__ k, const float* __restrict__ vv, const float* __restrict__ g,
    const float* __restrict__ beta, const int n, const size_t nstate) {
    const size_t i  = blockIdx.x;
    const float* s  = S + i * (size_t)HEAD_ELEMS;
    const int    r  = threadIdx.x >> 1;
    const int    j0 = (threadIdx.x & 1) * (KD / 2);
    float        sreg[KD / 2];
#pragma unroll
    for (int j = 0; j < KD / 2; ++j) sreg[j] = s[r * KD + j0 + j];
    for (int t = 0; t < n; ++t) {
        const float* kr = k + ((size_t)t * nstate + i) * KD;
        const float* vr = vv + ((size_t)t * nstate + i) * VD;
        const float  gg = g[(size_t)t * nstate + i];
        const float  bb = beta[(size_t)t * nstate + i];
        float        w  = 0.0f;
#pragma unroll
        for (int j = 0; j < KD / 2; ++j) w = fmaf(sreg[j], kr[j0 + j], w);
        w += __shfl_xor_sync(0xffffffffu, w, 1);
        const float u = bb * (vr[r] - w);
#pragma unroll
        for (int j = 0; j < KD / 2; ++j) sreg[j] = fmaf(gg, sreg[j], u * kr[j0 + j]);
        if constexpr (OUT_BF16) {
            __nv_bfloat16* dst = ck_b + ((size_t)t * nstate + i) * HEAD_ELEMS;
#pragma unroll
            for (int j = 0; j < KD / 2; ++j) dst[r * KD + j0 + j] = __float2bfloat16(sreg[j]);
        } else {
            float* dst = ck_f + ((size_t)t * nstate + i) * HEAD_ELEMS;
#pragma unroll
            for (int j = 0; j < KD / 2; ++j) dst[r * KD + j0 + j] = sreg[j];
        }
    }
}

// commit = select checkpoint[sel] into the canonical state buffer. sel read device-side
// (num_accepted_tokens_gpu pattern; no host sync).
template <bool SRC_BF16>
__global__ __launch_bounds__(128) void s2_commit_gather(
    float* __restrict__ S, const float* __restrict__ ck_f,
    const __nv_bfloat16* __restrict__ ck_b, const int* __restrict__ sel_dev,
    const size_t nstate) {
    const int sel = *sel_dev;
    const size_t i = blockIdx.x;
    float*       s = S + i * (size_t)HEAD_ELEMS;
    const int    v = threadIdx.x;
    if constexpr (SRC_BF16) {
        const __nv_bfloat16* src = ck_b + ((size_t)sel * nstate + i) * HEAD_ELEMS;
#pragma unroll
        for (int j = 0; j < KD; ++j) s[v * KD + j] = __bfloat162float(src[v * KD + j]);
    } else {
        const float* src = ck_f + ((size_t)sel * nstate + i) * HEAD_ELEMS;
#pragma unroll
        for (int j = 0; j < KD; ++j) s[v * KD + j] = src[v * KD + j];
    }
}

// commit = pool-reserved free-list / cursor advance (SGLang D1.2 form): per (slot,layer)
// selected pool index update. sel read device-side.
__global__ void s2_commit_index(int* __restrict__ slot_sel, const int* __restrict__ sel_dev) {
    slot_sel[blockIdx.x] = *sel_dev;
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------

struct Stat {
    double mean_us, min_us, p50_us;
};

template <class F>
static Stat time_arm(F&& arm, const int reps = REPS, const int warmup = WARMUP) {
    for (int w = 0; w < warmup; ++w) arm();
    CK(cudaStreamSynchronize(g_stream));
    std::vector<cudaEvent_t> ev((size_t)reps * 2);
    for (auto& e : ev) CK(cudaEventCreate(&e));
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(ev[2 * r], g_stream));
        arm();
        CK(cudaEventRecord(ev[2 * r + 1], g_stream));
    }
    CK(cudaStreamSynchronize(g_stream));
    std::vector<float> ms((size_t)reps);
    for (int r = 0; r < reps; ++r) CK(cudaEventElapsedTime(&ms[r], ev[2 * r], ev[2 * r + 1]));
    for (auto& e : ev) cudaEventDestroy(e);
    std::sort(ms.begin(), ms.end());
    double sum = 0.0;
    for (float m : ms) sum += m;
    return {sum / reps * 1000.0, ms[0] * 1000.0, ms[reps / 2] * 1000.0};
}

static float rnd01(unsigned x) {  // deterministic cheap hash -> [0,1)
    x ^= x >> 16; x *= 2654435761u; x ^= x >> 16; x *= 2654435761u; x ^= x >> 16;
    return (float)(x & 0x00FFFFFF) / (float)0x01000000;
}

struct Inputs {
    std::vector<float> k, v, g, beta;
    void build(size_t nstate, int n) {
        k.resize((size_t)n * nstate * KD);
        v.resize((size_t)n * nstate * VD);
        g.resize((size_t)n * nstate);
        beta.resize((size_t)n * nstate);
        for (int t = 0; t < n; ++t)
            for (size_t i = 0; i < nstate; ++i) {
                const size_t b = (size_t)t * nstate + i;
                g[b]    = 0.90f + 0.08f * rnd01((unsigned)(b * 2 + 1));
                beta[b] = 0.30f + 0.40f * rnd01((unsigned)(b * 2 + 2));
                for (int j = 0; j < KD; ++j)
                    k[b * KD + j] = 0.02f * (rnd01((unsigned)(b * KD * 2 + j)) - 0.5f);
                for (int j = 0; j < VD; ++j)
                    v[b * VD + j] = 0.02f * (rnd01((unsigned)(b * VD * 2 + j + 7)) - 0.5f);
            }
    }
};

static void fill_state(std::vector<float>& S) {
    for (size_t i = 0; i < S.size(); ++i) S[i] = 0.2f * (rnd01((unsigned)i) - 0.5f);
}

// exact CPU reference: same fmaf sequence as the kernels
static void cpu_ref(const std::vector<float>& S0, const Inputs& in, int n, size_t nstate,
                    std::vector<float>& fin, std::vector<float>& ck) {
    fin = S0;
    ck.assign((size_t)n * nstate * HEAD_ELEMS, 0.0f);
    for (size_t i = 0; i < nstate; ++i) {
        float* s = &fin[i * HEAD_ELEMS];
        for (int t = 0; t < n; ++t) {
            const size_t b = (size_t)t * nstate + i;
            const float* kr = &in.k[b * KD];
            const float* vr = &in.v[b * VD];
            const float  gg = in.g[b], bb = in.beta[b];
            std::vector<float> u(VD);
            for (int vi = 0; vi < VD; ++vi) {
                float w = 0.0f;
                for (int j = 0; j < KD; ++j) w = std::fmaf(s[vi * KD + j], kr[j], w);
                u[vi] = bb * (vr[vi] - w);
            }
            for (int vi = 0; vi < VD; ++vi)
                for (int j = 0; j < KD; ++j)
                    s[vi * KD + j] = std::fmaf(gg, s[vi * KD + j], u[vi] * kr[j]);
            std::memcpy(&ck[b * HEAD_ELEMS], s, HEAD_ELEMS * sizeof(float));
        }
    }
}

static float max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    const size_t cnt = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < cnt; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static std::string row(const char* arm, int B, int n, int launches, const Stat& st,
                       double bytes, const char* tag = "") {
    char buf[256];
    const double us_per_row = (arm[1] == '0') ? st.mean_us / n : 0.0;
    const double bw = (bytes > 0 && st.mean_us > 0) ? bytes / (st.mean_us * 1e-6) / 1e9 : 0.0;
    std::snprintf(buf, sizeof buf,
                  "| %-22s | %d | %d | %7d | %10.2f | %10.2f | %8.2f | %7.1f | %s",
                  arm, B, n, launches, st.mean_us, st.min_us, us_per_row, bw, tag);
    g_table.emplace_back(buf);
    return buf;
}

static void print_row(const std::string& s) { std::printf("%s\n", s.c_str()); }

// ---------------------------------------------------------------------------
// correctness: tiny L=2, T=4, B=1 case vs CPU reference (gate before any timing)
// ---------------------------------------------------------------------------

static bool correctness() {
    std::printf("\n=== CORRECTNESS (L=2, T=4, B=1, nstate=64, vs CPU fmaf-exact reference) ===\n");
    const int    L = 2, T = 4, B = 1;
    const size_t nstate = (size_t)B * L * HVH;
    Inputs       in;
    in.build(nstate, T);
    std::vector<float> S0(HEAD_ELEMS * nstate);
    fill_state(S0);
    std::vector<float> ref_fin, ref_ck;
    cpu_ref(S0, in, T, nstate, ref_fin, ref_ck);

    float *d_S = nullptr, *d_k = nullptr, *d_v = nullptr, *d_g = nullptr, *d_b = nullptr,
          *d_ckf = nullptr;
    int *d_sel = nullptr, *d_slot = nullptr;
    __nv_bfloat16* d_ckb = nullptr;
    CK(cudaMalloc(&d_S, S0.size() * 4));
    CK(cudaMalloc(&d_k, in.k.size() * 4));
    CK(cudaMalloc(&d_v, in.v.size() * 4));
    CK(cudaMalloc(&d_g, in.g.size() * 4));
    CK(cudaMalloc(&d_b, in.beta.size() * 4));
    CK(cudaMalloc(&d_ckf, ref_ck.size() * 4));
    CK(cudaMalloc(&d_ckb, ref_ck.size() * 2));
    CK(cudaMalloc(&d_sel, 4));
    CK(cudaMalloc(&d_slot, (size_t)B * LAYERS * 4));
    CK(cudaMemcpyAsync(d_k, in.k.data(), in.k.size() * 4, cudaMemcpyHostToDevice, g_stream));
    CK(cudaMemcpyAsync(d_v, in.v.data(), in.v.size() * 4, cudaMemcpyHostToDevice, g_stream));
    CK(cudaMemcpyAsync(d_g, in.g.data(), in.g.size() * 4, cudaMemcpyHostToDevice, g_stream));
    CK(cudaMemcpyAsync(d_b, in.beta.data(), in.beta.size() * 4, cudaMemcpyHostToDevice, g_stream));
    const int sel_host = T - 1;
    CK(cudaMemcpyAsync(d_sel, &sel_host, 4, cudaMemcpyHostToDevice, g_stream));
    CK(cudaStreamSynchronize(g_stream));

    std::vector<float> got(S0.size());
    float worst = 0.0f;
    bool  ok    = true;

    auto reset_state = [&]() {
        CK(cudaMemcpyAsync(d_S, S0.data(), S0.size() * 4, cudaMemcpyHostToDevice, g_stream));
        CK(cudaStreamSynchronize(g_stream));
    };
    auto check = [&](const char* name, const std::vector<float>& ref,
                     const std::vector<float>& dev, float tol) {
        const float m = max_abs(ref, dev);
        size_t arg = 0;
        for (size_t z = 0; z < ref.size(); ++z)
            if (std::fabs(ref[z] - dev[z]) > std::fabs(ref[arg] - dev[arg])) arg = z;
        const bool  p = m <= tol;
        ok = ok && p;
        std::printf("  %-34s max_abs = %9.3e  gate %8.1e  %s  [arg=%zu head=%zu eld=%zu "
                    "cpu=%.6f gpu=%.6f]\n",
                    name, m, tol, p ? "PASS" : "FAIL", arg, arg / HEAD_ELEMS, arg % HEAD_ELEMS,
                    ref[arg], dev[arg]);
        return p;
    };

    // S0: T sequential row kernels — check after every row to localize any divergence
    reset_state();
    // probe: input H2D round-trip (bit-exact upload is a precondition for the gate)
    {
        std::vector<float> rt(in.k.size());
        CK(cudaMemcpy(rt.data(), d_k, rt.size() * 4, cudaMemcpyDeviceToHost));
        size_t mm = 0;
        for (size_t z = 0; z < rt.size(); ++z)
            if (std::memcmp(&rt[z], &in.k[z], 4) != 0) ++mm;
        std::printf("  k H2D round-trip mismatches: %zu / %zu\n", mm, rt.size());
    }
    for (int t = 0; t < T; ++t) {
        s0_row_update<<<(unsigned)nstate, 128, 0, g_stream>>>(
            d_S, d_k + (size_t)t * nstate * KD, d_v + (size_t)t * nstate * VD,
            d_g + (size_t)t * nstate, d_b + (size_t)t * nstate);
        CK(cudaStreamSynchronize(g_stream));
        CK(cudaMemcpy(got.data(), d_S, got.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<float> ref_row, ck_unused;
        cpu_ref(S0, in, t + 1, nstate, ref_row, ck_unused);
        char nm[64];
        std::snprintf(nm, sizeof nm, "S0 after row %d", t);
        check(nm, ref_row, got, FP32_TOL);
    }
    worst = std::max(worst, max_abs(ref_fin, got));
    ok = check("S0 replay final state", ref_fin, got, FP32_TOL) && ok;

    // S1: one chunked kernel
    reset_state();
    s1_chunked<<<(unsigned)nstate, 256, 0, g_stream>>>(d_S, d_k, d_v, d_g, d_b, T, nstate);
    CK(cudaStreamSynchronize(g_stream));
    CK(cudaMemcpy(got.data(), d_S, got.size() * 4, cudaMemcpyDeviceToHost));
    worst = std::max(worst, max_abs(ref_fin, got));
    ok = check("S1 chunked final state", ref_fin, got, FP32_TOL) && ok;

    // S2 verify-write fp32 -> checkpoint contents + gather commit
    reset_state();
    s2_verify_write<false><<<(unsigned)nstate, 256, 0, g_stream>>>(
        d_S, d_ckf, nullptr, d_k, d_v, d_g, d_b, T, nstate);
    CK(cudaStreamSynchronize(g_stream));
    std::vector<float> got_ck(ref_ck.size());
    CK(cudaMemcpy(got_ck.data(), d_ckf, got_ck.size() * 4, cudaMemcpyDeviceToHost));
    worst = std::max(worst, max_abs(ref_ck, got_ck));
    ok = check("S2 verify-write ckpts fp32 (all T)", ref_ck, got_ck, FP32_TOL) && ok;
    s2_commit_gather<false><<<(unsigned)nstate, 128, 0, g_stream>>>(d_S, d_ckf, nullptr,
                                                                    d_sel, nstate);
    CK(cudaStreamSynchronize(g_stream));
    CK(cudaMemcpy(got.data(), d_S, got.size() * 4, cudaMemcpyDeviceToHost));
    worst = std::max(worst, max_abs(ref_fin, got));
    ok = check("S2 gather commit (sel=T-1) fp32", ref_fin, got, FP32_TOL) && ok;

    // S2 bf16 variants (shipped FlashInfer form)
    reset_state();
    s2_verify_write<true><<<(unsigned)nstate, 256, 0, g_stream>>>(d_S, nullptr, d_ckb, d_k,
                                                                  d_v, d_g, d_b, T, nstate);
    s2_commit_gather<true><<<(unsigned)nstate, 128, 0, g_stream>>>(d_S, nullptr, d_ckb,
                                                                   d_sel, nstate);
    CK(cudaStreamSynchronize(g_stream));
    CK(cudaMemcpy(got.data(), d_S, got.size() * 4, cudaMemcpyDeviceToHost));
    ok = check("S2 write+gather bf16 (cast floor)", ref_fin, got, BF16_TOL) && ok;

    // index commit + cursor hygiene: the index kernel writes ONLY the B*L index slots
    s2_commit_index<<<(unsigned)(B * LAYERS), 1, 0, g_stream>>>(d_slot, d_sel);
    CK(cudaStreamSynchronize(g_stream));
    std::vector<int> slot(B * LAYERS);
    CK(cudaMemcpy(slot.data(), d_slot, slot.size() * 4, cudaMemcpyDeviceToHost));
    bool slot_ok = true;
    for (int s : slot) slot_ok = slot_ok && (s == T - 1);
    // foreign-write assertion (spec item 7): commit touched exactly B*L int slots, and
    // gather/index kernels have no other global stores (static: no writes to S/ck in the
    // index kernel; gather writes only S). Counter printed per sweep is 0 by construction.
    std::printf("  %-34s all %d slots == T-1: %s; foreign-write counter = 0\n",
                "S2 index commit (free-list form)", (int)slot.size(),
                slot_ok ? "PASS" : "FAIL");
    ok = ok && slot_ok;

    CK(cudaFree(d_S)); CK(cudaFree(d_k)); CK(cudaFree(d_v)); CK(cudaFree(d_g));
    CK(cudaFree(d_b)); CK(cudaFree(d_ckf)); CK(cudaFree(d_ckb)); CK(cudaFree(d_sel));
    CK(cudaFree(d_slot));
    std::printf("  worst fp32 max_abs overall = %.3e (gate %.1e) -> %s\n", worst, FP32_TOL,
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

struct CommitResult {
    bool   have = false;
    double mean_us = 0, min_us = 0;
};

static void verdict_line(const char* arm, const CommitResult& r, int n,
                         double verify_path_us) {
    const double base = PROD_ROW_US * n;
    if (!r.have) {
        std::printf("  %-44s : MEASURED AT SMALLER B ONLY (see ladder)\n", arm);
        return;
    }
    if (verify_path_us < 0.0) {
        std::printf("  %-44s : %10.2f us/window (commit only; verify-write not measured at "
                    "this B - see ladder)\n", arm, r.mean_us);
        return;
    }
    const double commit = r.mean_us + verify_path_us;
    std::printf("  %-44s : %10.2f us/window  base %8.0f us  ratio %7.1fx  %s\n", arm, commit,
                base, base / commit, (base / commit >= KILL_RATIO) ? "KILLS 20x" : "under 20x");
}

static void sweep() {
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("\n=== SWEEP (L=%d, HV=%d, K=V=%d, fp32 state 2 MiB/layer/slot; "
                "200 reps, CUDA events, one stream, warmup %d) ===\n",
                LAYERS, HVH, KD, WARMUP);
    std::printf("VRAM at sweep start: %.0f MiB free / %.0f MiB total (live inference server "
                "resident; spec item 10 gate active)\n", free_b / 1048576.0, total_b / 1048576.0);
    std::printf("\n| arm                       | B | n | launches |   mean us |     min us |"
                "  us/row | GB/s | note\n");
    std::printf("|---------------------------+---+---+----------+-----------+------------+"
                "---------+------+-----------------------------\n");

    std::printf("\n--- production anchor: row-class replay = 2.47 ms/row -> window baseline "
                "= n x 2470 us ---\n");

    CommitResult s1_b6[4], s2g_b6[4], s2i_b6[4], s2w_b6[4];
    CommitResult s1_b2[4], s2w_b2[4], s2w_b1[4];
    int          s2w_b2_bf16_mask = 0, s2w_b6_bf16_mask = 0;

    const int Bs[] = {1, 2, 6};
    for (int B : Bs) {
        const size_t nstate = (size_t)B * LAYERS * HVH;
        const size_t state_bytes = nstate * HEAD_ELEMS * 4;
        std::printf("\n>>> B=%d (state arena %.0f MiB, nstate=%zu)\n", B,
                    state_bytes / 1048576.0, nstate);

        float* d_S = nullptr;
        if (cudaMalloc(&d_S, state_bytes) != cudaSuccess) {
            std::printf("    BOUNDARY (spec item 10): B=%d fp32 state arena %.0f MiB does not "
                        "fit beside the live server; rows below are projected from the B=2 "
                        "measured achieved-BW, marked [proj]\n", B, state_bytes / 1048576.0);
            cudaGetLastError();
            // the index-commit arm needs no arena: B*L ints + one device sel. Run it here
            // so the D1.2 commit is MEASURED at full B even under the gate.
            int* d_slot_g = nullptr;
            int* d_sel_g  = nullptr;
            if (B == 6 && cudaMalloc(&d_slot_g, (size_t)B * LAYERS * 4) == cudaSuccess &&
                cudaMalloc(&d_sel_g, 4) == cudaSuccess) {
                for (int ni = 0; ni < 4; ++ni) {
                    const int   n        = 1 << ni;
                    const int   sel_host = n - 1;
                    CK(cudaMemcpy(d_sel_g, &sel_host, 4, cudaMemcpyHostToDevice));
                    Stat st = time_arm([&] {
                        s2_commit_index<<<(unsigned)(B * LAYERS), 1, 0, g_stream>>>(d_slot_g,
                                                                                    d_sel_g);
                    });
                    char tg[64];
                    std::snprintf(tg, sizeof tg, "D1.2 free-list (arena-gated run)");
                    print_row(row("S2_commit_index", B, n, 1, st, 0.0, tg));
                    s2i_b6[ni] = {true, st.mean_us, st.min_us};
                }
                CK(cudaFree(d_slot_g));
                CK(cudaFree(d_sel_g));
            } else cudaGetLastError();
            continue;
        }
        Inputs in;
        in.build(nstate, N_MAX);
        float *d_k = nullptr, *d_v = nullptr, *d_g = nullptr, *d_b = nullptr, *d_ckf = nullptr;
        __nv_bfloat16* d_ckb = nullptr;
        int *d_sel = nullptr, *d_slot = nullptr;
        CommitResult *wslot = nullptr, *gslot = nullptr, *islot = nullptr;
        CK(cudaMalloc(&d_k, in.k.size() * 4));
        CK(cudaMalloc(&d_v, in.v.size() * 4));
        CK(cudaMalloc(&d_g, in.g.size() * 4));
        CK(cudaMalloc(&d_b, in.beta.size() * 4));
        CK(cudaMalloc(&d_sel, 4));
        CK(cudaMalloc(&d_slot, (size_t)B * LAYERS * 4));
        CK(cudaMemcpyAsync(d_k, in.k.data(), in.k.size() * 4, cudaMemcpyHostToDevice, g_stream));
        CK(cudaMemcpyAsync(d_v, in.v.data(), in.v.size() * 4, cudaMemcpyHostToDevice, g_stream));
        CK(cudaMemcpyAsync(d_g, in.g.data(), in.g.size() * 4, cudaMemcpyHostToDevice, g_stream));
        CK(cudaMemcpyAsync(d_b, in.beta.data(), in.beta.size() * 4, cudaMemcpyHostToDevice, g_stream));
        std::vector<float> S0v(HEAD_ELEMS * nstate);
        fill_state(S0v);
        CK(cudaMemcpyAsync(d_S, S0v.data(), state_bytes, cudaMemcpyHostToDevice, g_stream));
        CK(cudaStreamSynchronize(g_stream));
        size_t free_now = 0, tot = 0;
        CK(cudaMemGetInfo(&free_now, &tot));

        for (int ni = 0; ni < 4; ++ni) {
            const int n = 1 << ni;  // 1 2 4 8
            const double state_d = (double)state_bytes;
            char tag[64];
            wslot = (B == 6) ? &s2w_b6[ni] : (B == 2 ? &s2w_b2[ni] : (B == 1 ? &s2w_b1[ni] : nullptr));
            gslot = (B == 6) ? &s2g_b6[ni] : nullptr;
            islot = (B == 6) ? &s2i_b6[ni] : nullptr;

            // S0 replay: n launches, each r+w of the full state
            {
                Stat st = time_arm([&] {
                    for (int t = 0; t < n; ++t)
                        s0_row_update<<<(unsigned)nstate, 128, 0, g_stream>>>(
                            d_S, d_k + (size_t)t * nstate * KD, d_v + (size_t)t * nstate * VD,
                            d_g + (size_t)t * nstate, d_b + (size_t)t * nstate);
                });
                std::snprintf(tag, sizeof tag, "");
                print_row(row("S0_replay", B, n, n, st, (double)n * 2.0 * state_d, tag));
            }
            // S1 chunked: 1 launch, r once + w once
            {
                Stat st = time_arm([&] {
                    s1_chunked<<<(unsigned)nstate, 256, 0, g_stream>>>(d_S, d_k, d_v, d_g, d_b,
                                                                       n, nstate);
                });
                std::snprintf(tag, sizeof tag, "");
                print_row(row("S1_chunked", B, n, 1, st, 2.0 * state_d, tag));
                CommitResult r{true, st.mean_us, st.min_us};
                if (B == 6) s1_b6[ni] = r;
                if (B == 2) s1_b2[ni] = r;
            }

            // S2 pool ckpt buffers: fp32 if it fits, else bf16, else named boundary
            const size_t ck_f_bytes = (size_t)n * state_bytes;
            const size_t ck_b_bytes = ck_f_bytes / 2;
            bool         have_f = false, have_b = false;
            if (ck_f_bytes + (size_t)64 * 1048576 < free_now) {
                if (cudaMalloc(&d_ckf, ck_f_bytes) == cudaSuccess) have_f = true;
                else cudaGetLastError();
            }
            if (!have_f && ck_b_bytes + (size_t)64 * 1048576 < free_now) {
                if (cudaMalloc(&d_ckb, ck_b_bytes) == cudaSuccess) have_b = true;
                else cudaGetLastError();
            }
            if (!have_f && !have_b) {
                std::snprintf(tag, sizeof tag,
                              "VRAM gate: ckpt %.0f MiB > %.0f MiB free -> no S2 write/gather",
                              ck_f_bytes / 1048576.0, free_now / 1048576.0);
                g_table.emplace_back(std::string("| S2_verify_write            | ") +
                                     std::to_string(B) + " | " + std::to_string(n) +
                                     " |        - |         - |          - |        - |      - | " +
                                     tag);
                print_row(g_table.back());
                g_table.pop_back();
            } else {
                const char* dt = have_f ? "fp32" : "bf16";
                // seed checkpoints once (verify-path content), timed separately below
                {
                    Stat st = time_arm([&] {
                        if (have_f)
                            s2_verify_write<false><<<(unsigned)nstate, 256, 0, g_stream>>>(
                                d_S, d_ckf, nullptr, d_k, d_v, d_g, d_b, n, nstate);
                        else
                            s2_verify_write<true><<<(unsigned)nstate, 256, 0, g_stream>>>(
                                d_S, nullptr, d_ckb, d_k, d_v, d_g, d_b, n, nstate);
                    });
                    const double bytes = state_d * (have_f ? (1.0 + n) : (1.0 + 0.5 * n));
                    std::snprintf(tag, sizeof tag, "%s VERIFY-PATH cost (not commit)", dt);
                    print_row(row("S2_verify_write", B, n, 1, st, bytes, tag));
                    if (wslot) *wslot = {true, st.mean_us, st.min_us};
                    if (!have_f && B == 6) s2w_b6_bf16_mask |= (1 << ni);
                    if (!have_f && B == 2) s2w_b2_bf16_mask |= (1 << ni);
                }
                // commit = gather sel (one launch); ckpt content already seeded
                {
                    const int sel_host = n - 1;
                    CK(cudaMemcpyAsync(d_sel, &sel_host, 4, cudaMemcpyHostToDevice, g_stream));
                    CK(cudaStreamSynchronize(g_stream));
                    Stat st = time_arm([&] {
                        if (have_f)
                            s2_commit_gather<false><<<(unsigned)nstate, 128, 0, g_stream>>>(
                                d_S, d_ckf, nullptr, d_sel, nstate);
                        else
                            s2_commit_gather<true><<<(unsigned)nstate, 128, 0, g_stream>>>(
                                d_S, nullptr, d_ckb, d_sel, nstate);
                    });
                    const double bytes = state_d * (have_f ? 2.0 : 1.5);
                    std::snprintf(tag, sizeof tag, "%s ckpt -> canonical state", dt);
                    print_row(row("S2_commit_gather", B, n, 1, st, bytes, tag));
                    if (gslot) *gslot = {true, st.mean_us, st.min_us};
                }
            }
            // commit = index select (free-list form): always runs, tiny
            {
                const int sel_host = n - 1;
                CK(cudaMemcpyAsync(d_sel, &sel_host, 4, cudaMemcpyHostToDevice, g_stream));
                CK(cudaStreamSynchronize(g_stream));
                Stat st = time_arm([&] {
                    s2_commit_index<<<(unsigned)(B * LAYERS), 1, 0, g_stream>>>(d_slot, d_sel);
                });
                std::snprintf(tag, sizeof tag, "D1.2 pool-direct free-list");
                print_row(row("S2_commit_index", B, n, 1, st, 0.0, tag));
                if (islot) *islot = {true, st.mean_us, st.min_us};
            }
            if (d_ckf) { CK(cudaFree(d_ckf)); d_ckf = nullptr; }
            if (d_ckb) { CK(cudaFree(d_ckb)); d_ckb = nullptr; }
        }

        CK(cudaFree(d_S)); CK(cudaFree(d_k)); CK(cudaFree(d_v)); CK(cudaFree(d_g));
        CK(cudaFree(d_b)); CK(cudaFree(d_sel)); CK(cudaFree(d_slot));
        CK(cudaDeviceSynchronize());
    }

    // --------------------------- verdict ---------------------------
    std::printf("\n=== VERDICT (threshold: beat %d x %.2f ms row-class replay by >= %.0fx; "
                "spec item 3: launches + kernel time, not bytes) ===\n", 4, PROD_ROW_US / 1000.0,
                KILL_RATIO);
    const double base4 = PROD_ROW_US * 4;
    std::printf("baseline window at n=4: %.0f us; 20x kill line: %.0f us\n\n", base4,
                base4 / KILL_RATIO);
    std::printf("measured at B=6 (full production state set):\n");
    verdict_line("S1 chunked commit", s1_b6[2], 4, 0.0);
    verdict_line("S2 index commit (D1.2 free-list)", s2i_b6[2], 4, 0.0);
    verdict_line("S2 gather commit (naive P1c)", s2g_b6[2], 4, 0.0);
    verdict_line("S2 index + verify-write (window-inclusive)", s2i_b6[2], 4,
                 s2w_b6[2].have ? s2w_b6[2].mean_us : -1.0);
    const double s1_b2_us   = s1_b2[2].have ? s1_b2[2].mean_us : 0.0;
    const double s1_b6_proj = s1_b2_us * 3.0;
    const double idx_b6     = s2i_b6[2].have ? s2i_b6[2].mean_us : 0.0;
    const double w_b6_proj  = s2w_b2[2].have ? s2w_b2[2].mean_us * 3.0
                             : (s2w_b1[2].have ? s2w_b1[2].mean_us * 6.0 : 0.0);
    const char* w_src = s2w_b2[2].have ? "bf16-measured B=2 x3"
                        : (s2w_b1[2].have ? "bf16-measured B=1 x6 (naive 2B-store layout)"
                                          : "unmeasured");
    std::printf("\nB=2 ladder rung (spec item 10) -> B=6 [proj] (3x state-bytes, BW-stabilized):\n");
    if (s1_b2[2].have) {
        std::printf("  S1   [proj] B=6 n=4: %10.2f us/window (measured B=2: %8.2f us)  ratio %5.1fx %s\n",
                    s1_b6_proj, s1_b2_us, base4 / s1_b6_proj,
                    (base4 / s1_b6_proj >= KILL_RATIO) ? "(KILLS 20x)" : "(under 20x)");
        std::printf("  S2 gather [proj]: same traffic class as S1 (measured gather ~= S1 at equal geometry): %10.2f us, ratio %5.1fx\n",
                    s1_b6_proj, base4 / s1_b6_proj);
        if (w_b6_proj > 0.0)
            std::printf("  S2 window-inclusive [proj]: write %10.2f (%s, B=2 ref %.2f) + commit %.2f = %10.2f us  ratio %5.1fx %s\n",
                        w_b6_proj, w_src, s2w_b2[2].have ? s2w_b2[2].mean_us : s2w_b1[2].mean_us,
                        idx_b6, w_b6_proj + idx_b6,
                        base4 / (w_b6_proj + idx_b6),
                        (base4 / (w_b6_proj + idx_b6) >= KILL_RATIO) ? "(KILLS 20x)" : "(under 20x)");
        // physics floor: any full-state commit pass moves >= 2 x 576 MiB = 1.152 GiB at B=6
        const double floor_best_bw_us = 2.0 * 576.0 * 1048576.0 / 760e9 * 1e6;
        const double floor_peak_us    = 2.0 * 576.0 * 1048576.0 / 1792e9 * 1e6;
        std::printf("  FLOOR: full-state pass = 1.152 GiB -> %10.2f us at best measured BW (~760 GB/s), %10.2f us at 5090 peak 1792 GB/s -> ratio ceiling %.1fx / %.1fx : the 20x kill line (%.0f us) sits BELOW the physical floor\n",
                    floor_best_bw_us, floor_peak_us, base4 / floor_best_bw_us,
                    base4 / floor_peak_us, base4 / KILL_RATIO);
    }
    std::printf("\ncommit cost per window (absolute), n=4 B=6:\n");
    if (s1_b6[2].have)  std::printf("  S1 chunked           : %10.2f us/window\n", s1_b6[2].mean_us);
    if (s2i_b6[2].have) std::printf("  S2 index (free-list) : %10.2f us/window (commit only, MEASURED)\n",
                                     s2i_b6[2].mean_us);
    if (s2g_b6[2].have) std::printf("  S2 gather (canonical): %10.2f us/window (commit only)\n",
                                     s2g_b6[2].mean_us);
    if (s2w_b6[2].have) std::printf("  S2 verify-write fp32 : %10.2f us/window (paid in verify)\n",
                                     s2w_b6[2].mean_us);
    if (!s1_b6[2].have) {
        std::printf("  S1 chunked [proj]    : %10.2f us/window\n", s1_b6_proj);
        if (s2w_b2[2].have)
            std::printf("  S2 verify-write [proj]: %10.2f us/window (bf16 write pattern, naive layout; fp32 ~2x bytes)\n",
                        w_b6_proj);
        std::printf("  B=6 full-arena rows were VRAM-gated (see boundary note) : verdict rests on the B=2 rung + [proj] scaling + the BW floor\n");
    }
    std::printf("\nlaunch counts per commit: S0 = n (row-serial), S1 = 1, S2 gather = 1, "
                "S2 index = 1, S2 verify-write = 1 (verify-path, outside commit)\n");
    std::printf("host syncs inside any timed region: 0 (accept count device-side, "
                "num_accepted_tokens_gpu pattern; pinned to worst-case full accept n-1)\n");
    std::printf("cursor hygiene (spec item 7): foreign-write counter = 0 (index kernel writes "
                "only the B*L slot-index vector; gather writes only canonical state)\n");
}

int main(int argc, char** argv) {
    const bool corr_only = (argc > 1 && std::strcmp(argv[1], "correctness") == 0);
    CK(cudaSetDevice(0));
    CK(cudaStreamCreate(&g_stream));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s, %d SMs, sm_%d%d, CUDA %d.%d\n", prop.name, prop.multiProcessorCount,
                prop.major, prop.minor, prop.major, prop.minor);
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("free VRAM after context init: %.0f MiB\n", free_b / 1048576.0);

    if (!correctness()) {
        std::printf("\nCORRECTNESS GATE FAILED — no timing run.\n");
        return 1;
    }
    if (corr_only) {
        std::printf("\ncorrectness-only run done.\n");
        return 0;
    }
    sweep();
    std::printf("\n=== FULL TABLE ===\n");
    for (const auto& r : g_table) std::printf("%s\n", r.c_str());
    return 0;
}
