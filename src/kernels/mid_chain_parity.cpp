// src/kernels/mid_chain_parity.cpp - T38 V4's mid_chain parity rig (the dm_plan_parity pattern; the donor runs
// STANDALONE on our card against our standing spine, before any engine port is considered).
//
// The donor (jesdga95/basalt @ fb0944c7, vendored under third_party/basalt-fb0944c7; MIT) computes a GDN layer's
// middle as ONE persistent kernel: qkv + z projections (dense_tc_pair's integer-mma path) -> recurrence ->
// out projection.  Our standing spine computes the same layer with native_quantize_q8_1 + native_mmvq
// projections + the verify_kernels rec trio, and the campaign law is parity BEFORE battery: a ported kernel
// must be bitwise against the spine it replaces, or it is an M3b/KL-gated candidate (F-230 law) - never shipped
// as a near-miss.
//
// Phase A (this file, the STOP/GO decider): the donor's projection PAIR, launched exactly as mid_chain launches
// it (`pair_only`), against OUR standing calls for the same tensors:
//     native_quantize_q8_1(x) -> xq        (shared: the donor reads the very bytes our kernel wrote)
//     native_mmvq(type_q, wq, xq, qkv_ref, N, C,  T)     qkv
//     native_mmvq(type_z, wz, xq, z_ref,   N, ZV, T)     z (the gate projection)
// Bitwise per byte, over the pack's real (type_q, type_z, type_o) triples at T = 1..4, over the same weight
// bytes (random finite super-blocks of the types, the donor's own harness shape).  The type triples are the 17
// distinct ones the two GGUF shards declare for the 36 GDN layers (attn_qkv / attn_gate / ssm_out), so the
// exercising set is exactly what the engine's M1/pair windows (mean 2.47 rows, kMidChainMaxT = 4) would run.
// Phase B (the rec trio and the out projection, the full-chain comparison) extends this file only after Phase A
// is bitwise-green: the law is bitwise-first, and a divergence in the pair alone already ends the port.
//
//     build/mid_chain_parity [--case q,z,o] [--iters N]
//
// VERDICT AT fb0944c7 (T38, 2026-10-23, RTX 5090 / CUDA 13.4): NOT BITWISE - the port is STOPPED.  68
// (case, T) cases x 3 iterations, 0 bitwise iterations; the deterministic surface is rel ~1.0-1.3e-7 (fp32
// summation order: ~19% of output bytes differ, e.g. 2,109,184/11,141,120 in the iters=3 sweep) on both qkv
// and z for every one of the 17 real triples.  Evidence that this is the donor's persistent kernel, not the
// harness: OUR port of the donor's dense_tc.cu (same file, fb0944c7) is bitwise against OUR native_mmvq on
// every shape and type (build/dense_tc_parity: 23 kernel cases + 12 pair cases, 0 differing values), and our
// port's device code equals the donor's dense_tc.cu device code (the only textual delta is the pdl rename).
// So the donor's mid_chain pair does NOT reproduce dense_tc_pair's bits at this revision - mid_chain.hpp's
// "the same operations (and bits) as dense_tc_pair" claim fails on our shapes; the constant pair_xfrag_cg /
// pair_terms_xr path attn_chain_kernel.cuh:689 gives the persistent kernel its own x-fragment/term order.
// Secondary finding (same sweep): rare nondeterministic unwritten outputs (2/68 cases, z only, 100 and 2,628
// of 18,432 floats left at the harness fill) - a robustness anomaly of the standalone persistent launch, not
// investigated further because Phase A already ended the port.  Raw output:
// campaign/results/t38-mid-chain-parity-a.out
//
// Exit: 0 all bitwise, 1 a divergence (the surface printed), 77 the device does not cover the mid chain.
#include "strata/kernels/native_mmvq.hpp"     // our standing projections (the reference side)

#include "basalt/kernels/hc_fused.hpp"        // hc_fused_init / hc_fused_supported (mid_chain_init's gate)
#include "basalt/kernels/mid_chain.hpp"       // the donor under test

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = basalt::kernels;
namespace S = strata::kernels;

namespace {

constexpr int N = 2560;               // n_embd
constexpr int NQ = 10240;             // qkv rows (P_NQ)
constexpr int NZ = 6144;              // z rows (P_NZ)
constexpr int C = 2 * 16 * 128 + 48 * 128;   // 10240: the rec's qkv width
constexpr int ZV = 48 * 128;          // 6144: z width / h_v * S
constexpr int TM = K::kMidChainMaxT;  // 4
constexpr int KT = 10;                // top-k experts (the hc args' shape only; the pair does not route)

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s @%s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

// the donor's sb_bytes (a 256-value super-block's bytes)
int sb_bytes(int type) {
    switch (type) {
        case 12: return 144;   // Q4_K
        case 13: return 176;   // Q5_K
        case 14: return 210;   // Q6_K
        case 21: return 110;   // IQ3_S
        default: return 136;   // IQ4_XS (23)
    }
}
uint16_t sane_half(std::mt19937& r) {
    const uint32_t v = r();
    return (uint16_t) (((v >> 31) << 15) | ((5u + (v >> 10) % 5u) << 10) | (v & 0x3ffu));
}
// random super-blocks of a type with finite scales (the donor's harness shape: d, dmin for Q4_K / Q5_K; Q6_K's
// d is its last 2 bytes; IQ4_XS keeps its 2-byte d at 0)
std::vector<uint8_t> rand_type_host(int type, size_t nb, std::mt19937& r) {
    const int sb = sb_bytes(type);
    std::vector<uint8_t> w(nb * (size_t) sb);
    for (auto& v : w) v = (uint8_t) (r() & 0xffu);
    for (size_t k = 0; k < nb; ++k) {
        const uint16_t h = sane_half(r), h2 = sane_half(r);
        if (type == 14) std::memcpy(&w[k * (size_t) sb + 208], &h, 2);
        else {
            std::memcpy(&w[k * (size_t) sb], &h, 2);
            if (type == 12 || type == 13) std::memcpy(&w[k * (size_t) sb + 2], &h2, 2);
        }
    }
    return w;
}
template <typename V>
V* upload(const std::vector<V>& h) {
    V* d = nullptr;
    CK(cudaMalloc((void**) &d, h.size() * sizeof(V) + 64));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice));
    return d;
}
float* zeros(size_t n) {
    float* d = nullptr;
    CK(cudaMalloc((void**) &d, n * sizeof(float) + 64));
    CK(cudaMemset(d, 0, n * sizeof(float)));
    return d;
}

struct Diff {
    size_t bytes = 0, differ = 0, first = (size_t) -1, untouched = 0;
    double maxd = 0.0, maxr = 0.0, rmsd = 0.0, rmsr = 0.0;
};
// bitwise, per byte, of two device buffers of `n` floats; the magnitudes beside it (rel/rms ~1e-7 = summation
// order; larger = a real arithmetic difference).  untouched: donor elements still exactly the 0xAA fill (the
// kernel never wrote them) - a race or a skipped tile, not a rounding difference.
Diff compare(const float* a, const float* b, size_t n) {
    std::vector<float> va(n), vb(n);
    CK(cudaMemcpy(va.data(), a, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vb.data(), b, n * 4, cudaMemcpyDeviceToHost));
    Diff d;
    d.bytes = n * 4;
    const uint8_t* pa = reinterpret_cast<const uint8_t*>(va.data());
    const uint8_t* pb = reinterpret_cast<const uint8_t*>(vb.data());
    for (size_t i = 0; i < n * 4; ++i) {
        if (pa[i] != pb[i]) { ++d.differ; if (d.first == (size_t) -1) d.first = i; }
    }
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &va[i], 4);
        if (bits == 0xAAAAAAAAu) ++d.untouched;
        const double x = va[i], y = vb[i];
        const double ad = std::fabs(x - y);
        d.maxd = std::max(d.maxd, ad);
        d.maxr = std::max(d.maxr, std::fabs(x));
        d.rmsd += (x - y) * (x - y);
        d.rmsr += x * x;
    }
    d.rmsd = std::sqrt(d.rmsd / std::max<size_t>(n, 1));
    d.rmsr = std::sqrt(d.rmsr / std::max<size_t>(n, 1));
    return d;
}

struct Case {
    int tq, tz, to;
    const char* note;
};
// the 17 distinct (attn_qkv, attn_gate, ssm_out) type triples of the pack's 36 GDN layers, parsed from the two
// GGUF v3 shard headers (types: 12 Q4_K, 13 Q5_K, 14 Q6_K, 23 IQ4_XS; counts x layers in the comment)
const Case CASES[] = {
    {14, 14, 14, "x7"},  {14, 12, 14, "x6"},  {13, 14, 14, "x3"},  {14, 23, 14, "x3"},  {13, 23, 14, "x3"},
    {12, 12, 14, "x2"},  {23, 14, 14, "x2"},  {14, 12, 12, "x1"},  {14, 23, 12, "x1"},  {14, 13, 12, "x1"},
    {14, 23, 13, "x1"},  {14, 14, 13, "x1"},  {13, 14, 13, "x1"},  {14, 13, 14, "x1"},  {13, 13, 14, "x1"},
    {13, 12, 14, "x1"},  {13, 13, 13, "x1"},
};

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int iters = 1, only_q = -1, only_z = -1, only_o = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--case" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%d,%d,%d", &only_q, &only_z, &only_o) != 3) {
                std::printf("usage: mid_chain_parity [--iters N] [--case q,z,o]\n");
                return 2;
            }
        } else {
            std::printf("usage: mid_chain_parity [--iters N] [--case q,z,o]\n");
            return 2;
        }
    }
    int dev = 0, sms = 0, optin = 0;
    CK(cudaGetDevice(&dev));
    K::hc_fused_init();
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (!K::hc_fused_supported() || sms < K::kMidStampCtas || optin < 101376) {   // kMidStampCtas == R_CTA0 + R_HV = 170 CTAs
        std::printf("mid_chain_parity: device not covered (%d SMs, %d B optin smem, hc %d): the mid chain cannot "
                    "run here\n", sms, optin, (int) K::hc_fused_supported());
        return 77;
    }
    K::mid_chain_init();
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    std::mt19937 rng(38);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t) TM * N);
    for (auto& v : hx) v = nd(rng);
    float* x = upload(hx);
    uint8_t* xq = nullptr;   // OUR quantizer's bytes; the donor reads the same buffer
    CK(cudaMalloc((void**) &xq, (size_t) TM * (N / 32) * 36 + 64));
    S::native_quantize_q8_1(x, xq, N, TM, s);   // all TM rows: T-row runs read a prefix
    CK(cudaStreamSynchronize(s));

    // the rec args the pair phase touches (its rec-role CTAs prefetch the state and read x/w_alpha/w_beta for
    // the a/b heads); sized as the kernel expects, values finite and small
    const size_t state_floats = (size_t) 128 * 48 * 128 + (size_t) C * 3;
    std::vector<float> hstate(state_floats, 0.f);
    for (size_t i = 0; i < state_floats; ++i) hstate[i] = 0.01f * nd(rng);
    std::vector<float> hconv((size_t) C * 4);
    for (auto& v : hconv) v = 0.5f * nd(rng);
    std::vector<float> hdt(48), hsa(48), hgm(128);
    for (auto& v : hdt) v = nd(rng);
    for (auto& v : hsa) v = -std::exp(nd(rng));
    for (auto& v : hgm) v = 1.f + 0.1f * nd(rng);
    std::vector<uint16_t> hwa((size_t) 48 * N), hwb((size_t) 48 * N);
    for (auto& v : hwa) v = (uint16_t) ((((rng() >> 31) << 15) | ((5u + (rng() >> 10) % 5u) << 10) | (rng() & 0x3ffu)));
    for (auto& v : hwb) v = (uint16_t) ((((rng() >> 31) << 15) | ((5u + (rng() >> 10) % 5u) << 10) | (rng() & 0x3ffu)));
    const float* d_state = upload(hstate);
    const float* d_conv = upload(hconv);
    const float* d_dt = upload(hdt);
    const float* d_sa = upload(hsa);
    const float* d_gm = upload(hgm);
    const uint16_t* d_wa = upload(hwa);
    const uint16_t* d_wb = upload(hwb);
    float* d_h = zeros((size_t) TM * C);
    float* d_gate = zeros((size_t) TM * 48);
    float* d_beta = zeros((size_t) TM * 48);
    float* d_y = zeros((size_t) TM * ZV);
    uint8_t* d_yq = nullptr;
    CK(cudaMalloc((void**) &d_yq, (size_t) TM * (ZV / 32) * 36 + 64));

    float* d_qkv_donor = zeros((size_t) TM * C);
    float* d_z_donor = zeros((size_t) TM * ZV);
    float* d_qkv_ref = zeros((size_t) TM * C);
    float* d_z_ref = zeros((size_t) TM * ZV);
    float* d_wo_dummy = zeros(4096);   // pair_only never reads it; supported() only wants a non-null aligned one

    size_t cases = 0, bad = 0, total_bytes = 0, total_differ = 0;
    bool ok = true;
    for (const Case& c : CASES) {
        if (only_q >= 0 && (c.tq != only_q || c.tz != only_z || c.to != only_o)) continue;
        if (!S::native_mmvq_supported(c.tq) || !S::native_mmvq_supported(c.tz)) {
            std::printf("types %2d/%2d/%2d: our mmvq does not support the pair; skipped\n", c.tq, c.tz, c.to);
            continue;
        }
        const size_t nbq = (size_t) NQ * (N / 256), nbz = (size_t) NZ * (N / 256);
        std::vector<uint8_t> hwq = rand_type_host(c.tq, nbq, rng);
        std::vector<uint8_t> hwz = rand_type_host(c.tz, nbz, rng);
        const uint8_t* d_wq = upload(hwq);
        const uint8_t* d_wz = upload(hwz);
        for (int T = 1; T <= TM; ++T) {
            K::MidChainArgs a;
            a.T = T;
            a.xq = xq;
            a.wq = d_wq;
            a.wz = d_wz;
            a.wo = d_wo_dummy;
            a.type_q = c.tq;
            a.type_z = c.tz;
            a.type_o = c.to;
            K::DecRecArgs& r = a.rec;
            r.state = d_state;
            r.qkv = d_qkv_donor;
            r.conv_w = d_conv;
            r.x = x;
            r.w_alpha = d_wa;
            r.w_beta = d_wb;
            r.dt = d_dt;
            r.ssm_a = d_sa;
            r.z = d_z_donor;
            r.gamma = d_gm;
            r.h = d_h;
            r.gate = d_gate;
            r.beta = d_beta;
            r.y = d_y;
            r.yq = d_yq;
            r.C = C;
            r.n_embd = N;
            r.h_k = 16;
            r.h_v = 48;
            r.eps = 1e-6f;
            a.bo = d_wo_dummy;
            a.ord = 0;   // the first launch zeroes the hand-off counters
            a.pair_only = true;
            if (!K::mid_chain_supported(a)) {
                std::printf("types %2d/%2d/%2d T=%d: mid_chain_supported says no; skipped\n", c.tq, c.tz, c.to, T);
                continue;
            }
            bool case_ok = true;
            Diff dq{}, dz{};
            int matched = 0;
            for (int it = 0; it < iters; ++it) {
                CK(cudaMemset(d_qkv_donor, 0xAA, (size_t) T * C * 4));
                CK(cudaMemset(d_z_donor, 0xAA, (size_t) T * ZV * 4));
                K::mid_chain(a, s);   // the donor's own launch, pair_only: the projection pair exactly as it runs
                S::native_mmvq(c.tq, d_wq, xq, d_qkv_ref, N, NQ, T, s);
                S::native_mmvq(c.tz, d_wz, xq, d_z_ref, N, NZ, T, s);
                CK(cudaStreamSynchronize(s));
                const Diff q = compare(d_qkv_donor, d_qkv_ref, (size_t) T * C);
                const Diff z = compare(d_z_donor, d_z_ref, (size_t) T * ZV);
                dq = q;
                dz = z;
                matched += (q.differ == 0 && z.differ == 0);
                case_ok = case_ok && q.differ == 0 && z.differ == 0;
            }
            ++cases;
            total_bytes += dq.bytes + dz.bytes;
            total_differ += dq.differ + dz.differ;
            if (!case_ok) { ++bad; ok = false; }
            std::printf("types %2d/%2d/%2d %s T=%d: qkv %8zu/%9zu bytes differ  z %8zu/%9zu bytes differ  "
                        "| donor qkv vs ref: max|d| %.3e rel %.3e rms %.3e; z: max|d| %.3e rel %.3e rms %.3e  "
                        "[last of %d iters, %d matched; untouched-0xAA qkv %zu z %zu]\n",
                        c.tq, c.tz, c.to, c.note, T, dq.differ, dq.bytes, dz.differ, dz.bytes,
                        dq.maxd, dq.maxd / std::max(dq.maxr, 1e-30), dq.rmsd / std::max(dq.rmsr, 1e-30),
                        dz.maxd, dz.maxd / std::max(dz.maxr, 1e-30), dz.rmsd / std::max(dz.rmsr, 1e-30),
                        iters, matched, dq.untouched, dz.untouched);
            if (!case_ok) {
                // the divergence surface: the first differing float of each output (byte offset -> element)
                std::printf("    first qkv diff at byte %zu (element %zu of %zu): see above\n",
                            dq.first, dq.first / 4, dq.bytes / 4);
                std::printf("    first z   diff at byte %zu (element %zu of %zu)\n", dz.first, dz.first / 4, dz.bytes / 4);
            }
        }
        CK(cudaFree((void*) d_wq));
        CK(cudaFree((void*) d_wz));
    }
    std::printf("mid_chain_parity phase A: %zu cases, %zu diverging, %zu/%zu bytes differ (bitwise %s); PHASE B "
                "(rec trio + out) stays gated on this verdict\n",
                cases, bad, total_differ, total_bytes, ok ? "yes" : "NO");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}