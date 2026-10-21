// src/kernels/hc_q8_parity.cpp - the hyper-connection read on STRATA_HC_REQ8's int8 copies (fused_gr.cu's
// s_down/s_up path: w_down / w_up hold int8 codes in the bf16 layout, one fp32 scale per 32 values) against
// the bf16 read, fused_gr_read_multi, every T = 1..8, with and without an injection weight (a layer read vs
// the final mixer).  The q8 read is NOT the bf16 read's bits - the codes quantize (Q8_0's rule, amax/127,
// round-to-nearest-even, exactly load_hc_q8's host loop in src/core/weights.cpp) - so this bounds the error
// and checks the structure around it:
//   - every T-column of a T=8 q8 call is bitwise the same column of the per-T calls (the read's own
//     T-independence contract, now on the q8 path);
//   - mixed/inject_out vs the bf16 read: max |diff| and the RMS of the diff over the RMS of the reference,
//     per case (PASS below 0.02 relative RMS - the quantization step is amax/254 per value);
//   - no non-finite outputs.
// Real geometry (n_embd 2560, hc 4, hc_lr 320).  GPU, synthetic, no model.
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    ck(cudaMalloc(&p, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return (T*) p;
}
template <typename T>
T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return (T*) p;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}
uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
// load_hc_q8's loop (src/core/weights.cpp): d = amax / 127, q = round-to-nearest-even(x / d)
void quantize_req8(const std::vector<uint16_t>& bf16, std::vector<int8_t>& q, std::vector<float>& sc) {
    const size_t n = bf16.size();
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t b = (uint32_t) bf16[i] << 16;
        std::memcpy(&src[i], &b, 4);
    }
    q.resize(n);
    sc.resize(n / 32);
    for (size_t blk = 0; blk < n / 32; ++blk) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(src[blk * 32 + j]));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        sc[blk] = d;
        for (int j = 0; j < 32; ++j) q[blk * 32 + j] = (int8_t) std::nearbyint(src[blk * 32 + j] * id);
    }
}
// one fused_gr_read_multi call over T tokens; `q8` selects the int8-copies arguments
void run_read(const float* d_norm, const uint16_t* d_down, const uint16_t* d_up, const uint16_t* d_inj,
              float* d_R, const float* d_bo, const float* d_inj_prev, int T, bool with_inj, bool q8,
              const int8_t* d_qdown, const int8_t* d_qup, const float* d_sdown, const float* d_sup,
              float* d_mixed, float* d_inject_out, float* d_lo, float* d_rs, float* d_xn, cudaStream_t s) {
    std::vector<k::FusedGrArgs> fa(T);
    for (int t = 0; t < T; ++t) {
        k::FusedGrArgs& a = fa[t];
        a.R = d_R + (size_t) t * 4 * 2560;
        a.R_out = d_R + (size_t) t * 4 * 2560;   // apply=false: never written, but the field is non-const
        a.apply = false;
        a.bo_prev = nullptr;
        a.inj_prev = nullptr;
        a.w_norm = d_norm;
        a.w_down = (const uint16_t*) d_down;
        a.w_up = (const uint16_t*) d_up;
        a.w_inject = with_inj ? d_inj : nullptr;
        a.eps = 1e-6f;
        a.lo = d_lo + (size_t) t * 320;
        a.rs = d_rs + (size_t) t * 4;
        a.inject_out = with_inj ? d_inject_out + (size_t) t * 4 : nullptr;
        a.mixed = d_mixed + (size_t) t * 2560;
        if (q8) {
            a.w_down = (const uint16_t*) d_qdown;   // the codes ride the bf16 pointer, per the header contract
            a.w_up = (const uint16_t*) d_qup;
            a.s_down = d_sdown;
            a.s_up = d_sup;
        }
    }
    // the bool return is the S26 QFUSE indicator ("also wrote mixed's q8_1 image"), not success - false is
    // the normal case here (no q8_mixed destinations set)
    (void) k::fused_gr_read_multi(fa.data(), T, d_xn, s);
    ck(cudaStreamSynchronize(s), "read sync");
    ck(cudaGetLastError(), "read launch");
}
}  // namespace

int main() {
    const long long n_embd = 2560, hc = 4, hc_lr = 320, hc_dim = hc * n_embd;
    std::mt19937 rng(2416);
    std::normal_distribution<float> gauss(0.f, 1.f);

    std::vector<float> w_norm((size_t) hc_dim);
    for (auto& x : w_norm) x = 1.0f + 0.1f * gauss(rng);
    std::vector<uint16_t> w_down((size_t) (hc_lr * hc_dim)), w_up((size_t) (hc_dim * hc_lr)),
        w_inj((size_t) (hc * hc_dim));
    for (auto& x : w_down) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_up) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_inj) x = bf16_of(gauss(rng) * 0.02f);
    std::vector<int8_t> q_down, q_up;
    std::vector<float> s_down, s_up;
    quantize_req8(w_down, q_down, s_down);
    quantize_req8(w_up, q_up, s_up);

    float* d_norm = up(w_norm);
    uint16_t *d_down = up(w_down), *d_up = up(w_up), *d_inj = up(w_inj);
    int8_t *d_qdown = up(q_down), *d_qup = up(q_up);
    float *d_sdown = up(s_down), *d_sup = up(s_up);

    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");

    int cases = 0, bad = 0;
    for (int with_inj = 0; with_inj < 2; ++with_inj) {
        // the reference (bf16) and the candidate (int8 copies) at T=8, plus per-T calls for the bitwise
        // column contract
        std::vector<float> R8((size_t) 8 * hc_dim);
        for (auto& x : R8) x = gauss(rng);
float *d_R8 = up(R8), *d_xn8 = (float*) dev<float>(8 * hc_dim);
        float *d_lo8 = (float*) dev<float>(8 * hc_lr), *d_rs8 = (float*) dev<float>(8 * hc);
        float *d_mix_bf = (float*) dev<float>(8 * n_embd), *d_mix_q8 = (float*) dev<float>(8 * n_embd);
        float *d_inj_bf = (float*) dev<float>(8 * hc), *d_inj_q8 = (float*) dev<float>(8 * hc);
        run_read(d_norm, d_down, d_up, d_inj, d_R8, nullptr, nullptr, 8, with_inj != 0, false, nullptr, nullptr,
                 nullptr, nullptr, d_mix_bf, d_inj_bf, d_lo8, d_rs8, d_xn8, s);
        run_read(d_norm, d_down, d_up, d_inj, d_R8, nullptr, nullptr, 8, with_inj != 0, true, d_qdown, d_qup,
                 d_sdown, d_sup, d_mix_q8, d_inj_q8, d_lo8, d_rs8, d_xn8, s);
        const std::vector<float> ref = down(d_mix_bf, (size_t) 8 * n_embd);
        const std::vector<float> q8ref = down(d_mix_q8, (size_t) 8 * n_embd);   // the column contract's reference
        const std::vector<float> got = q8ref;
        double sd = 0, sr = 0, maxd = 0;
        long long nonfinite = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!std::isfinite(got[i])) ++nonfinite;
            const double d = std::fabs((double) got[i] - (double) ref[i]);
            maxd = std::max(maxd, d);
            sd += d * d;
            sr += (double) ref[i] * ref[i];
        }
        const double rms = std::sqrt(sd / std::max(sr, 1e-300));
        bool ok = nonfinite == 0 && rms < 0.02;
        if (with_inj) {
            const std::vector<float> iref = down(d_inj_bf, (size_t) 8 * hc);
            const std::vector<float> igot = down(d_inj_q8, (size_t) 8 * hc);
            double isd = 0, isr = 0;
            for (size_t i = 0; i < iref.size(); ++i) {
                const double d = (double) igot[i] - (double) iref[i];
                isd += d * d;
                isr += (double) iref[i] * iref[i];
            }
            const double irms = std::sqrt(isd / std::max(isr, 1e-300));
            std::printf("   inject_out rel rms %.3e\n", irms);
            ok = ok && irms < 0.02;
        }
        std::printf("T=8 %-6s vs bf16: max|d| %.3e  rel rms %.3e  nonfinite %lld  %s\n",
                    with_inj ? "layer" : "mixer", maxd, rms, nonfinite, ok ? "PASS" : "FAIL");
        ++cases;
        if (!ok) ++bad;

        // the column contract: every column of a T-token q8 call is bitwise the same column of the T=8 call
        for (int T = 1; T <= 8; ++T) {
            std::vector<float> RT(R8.begin(), R8.begin() + (size_t) T * hc_dim);
            float *d_RT = up(RT), *d_xnT = (float*) dev<float>(T * hc_dim);
            float *d_loT = (float*) dev<float>(T * hc_lr), *d_rsT = (float*) dev<float>(T * hc);
            float *d_mixT = (float*) dev<float>(T * n_embd), *d_injT = (float*) dev<float>(T * hc);
            run_read(d_norm, d_down, d_up, d_inj, d_RT, nullptr, nullptr, T, with_inj != 0, true, d_qdown, d_qup,
                     d_sdown, d_sup, d_mixT, d_injT, d_loT, d_rsT, d_xnT, s);
            const std::vector<float> gT = down(d_mixT, (size_t) T * n_embd);
            long long bits = 0;
            for (int t = 0; t < T; ++t)
                for (long long d = 0; d < n_embd; ++d)
                    if (std::memcmp(&gT[(size_t) t * n_embd + d], &q8ref[(size_t) t * n_embd + d], 4) != 0) ++bits;
            std::printf("   T=%d q8 columns bitwise vs the T=8 call: %s (%lld differ)\n", T,
                        bits == 0 ? "yes" : "NO", bits);
            ++cases;
            if (bits != 0) ++bad;
            cudaFree(d_RT); cudaFree(d_xnT); cudaFree(d_loT); cudaFree(d_rsT); cudaFree(d_mixT); cudaFree(d_injT);
        }
        cudaFree(d_R8); cudaFree(d_xn8); cudaFree(d_lo8); cudaFree(d_rs8);
        cudaFree(d_mix_bf); cudaFree(d_mix_q8); cudaFree(d_inj_bf); cudaFree(d_inj_q8);
    }
    cudaFree(d_norm); cudaFree(d_down); cudaFree(d_up); cudaFree(d_inj);
    cudaFree(d_qdown); cudaFree(d_qup); cudaFree(d_sdown); cudaFree(d_sup);
    cudaStreamDestroy(s);
    std::printf("%s (%d cases, %d bad)\n", bad == 0 ? "PASS" : "FAIL", cases, bad);
    return bad == 0 ? 0 : 1;
}
