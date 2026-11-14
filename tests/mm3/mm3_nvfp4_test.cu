// tests/mm3/mm3_nvfp4_test.cu - maya-mm3's NVFP4 weights (src/mm3/mm3_nvfp4.cu) on random blobs, no model: the
// host-side E2M1 / E4M3 decode written from the bit definitions against the device's ported llama.cpp pairing:
//   * the dense GEMVs (1, 3, 16 activations) against a double reference on the int8-rounded activations;
//   * the prompt path's GEMMs (FP16 bands, one and several) against a double reference on the FP16 values;
//   * the routed experts with MiniMax's swigluoai: the token path (an expert off the device, the CPU lane's part,
//     the shared expert), a verify window (bit for bit the token path's), the prompt sets (gathered rows);
//   * the manifest vocabulary: type_of / row_bytes.
// mm3_nvfp4_test [--mv | --gemm | --experts | --sets]
#include "../src/mm3/mm3_exl3.hpp"
#include "../src/mm3/mm3_gguf.cuh"
#include "../src/mm3/mm3_nvfp4.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace x3 = strata::mm3::exl3;
namespace gg = mm3::gg;
namespace nf = mm3::nf;

namespace {

int g_fail = 0;
std::mt19937 rng(20261114);

#define CU(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);        \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

float frand(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> h(n);
    CU(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

void expect(const char* what, const float* got, const double* want, size_t n, double rtol) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            ++g_fail;
            std::printf("%-52s FAIL  not finite (at %zu: %.6g vs %.6g)\n", what, i, (double) got[i], want[i]);
            return;
        }
    double worst = 0, scale = 1e-30;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs(want[i]));
    size_t at = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(got[i] - want[i]) / scale;
        if (e > worst) {
            worst = e;
            at = i;
        }
    }
    const bool ok = worst <= rtol;
    if (!ok) ++g_fail;
    std::printf("%-52s %s  max rel err %.2e (at %zu: %.6g vs %.6g)\n", what, ok ? "ok  " : "FAIL", worst, at,
                (double) got[at], want[at]);
}
void same(const char* what, const void* a, const void* b, size_t bytes) {
    const bool ok = std::memcmp(a, b, bytes) == 0;
    if (!ok) ++g_fail;
    std::printf("%-52s %s\n", what, ok ? "ok    bit for bit" : "FAIL  differs");
}

float h2f(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}
uint16_t f2h(float f) {
    const __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
float f16r(float f) { return h2f(f2h(f)); }

// ---------------------------------------------------------------- the CPU reference decode
// E2M1 (sign bit 3, magnitude bits 0..2) x E4M3FN scale byte (bits 0..6, bit 7 unused, the NaNs 0x7F / 0xFF -> 0)
// x the blob's F32 multiplier - the exact decode the format promises (the device pairs it as the doubled table
// times the halved scale: the same product).
double e4m3f(unsigned b) {
    b &= 0x7F;
    if (b == 0x7F) return 0.0;   // (the NaN byte: the format decodes it 0)
    const int e = (int) ((b >> 3) & 0xF), m = (int) (b & 7);
    return e == 0 ? std::ldexp((double) m, -9) : (1.0 + (double) m / 8.0) * std::ldexp(1.0, e - 7);
}
const double kTab[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

// a matrix blob (mm3_nvfp4.cuh's layout): F32 multiplier, nibbles [n][k / 2] (value c in nibble c of its row),
// scales [n][k / 16]; the nibbles and scale bytes random, the multiplier small positive
struct Blob {
    std::vector<uint8_t> b;
    int n = 0, k = 0;
    size_t nib_at() const { return 4; }
    size_t sc_at() const { return 4 + (size_t) n * (k / 2); }
    double w(int r, int c) const {   // the decoded weight of row r (an output), value c (an input)
        const unsigned nib = (b[nib_at() + (size_t) r * (k / 2) + (c >> 1)] >> ((c & 1) * 4)) & 0xF;
        const unsigned sc = b[sc_at() + (size_t) r * (k / 16) + (c >> 4)];
        float gs;
        std::memcpy(&gs, b.data(), 4);
        return kTab[nib] * e4m3f(sc) * (double) gs;
    }
};
Blob blob_rand(int n, int k) {
    Blob t;
    t.n = n;
    t.k = k;
    t.b.assign(4 + (size_t) n * (k / 2) + (size_t) n * (k / 16), 0);
    const float gs = 0.25f + 0.5f * std::fabs(frand());
    std::memcpy(t.b.data(), &gs, 4);
    for (size_t i = 4; i < t.b.size(); ++i) t.b[i] = (uint8_t) (rng() & 0xF);   // (nibbles 0..15; scales fixed below)
    for (size_t i = t.sc_at(); i < t.b.size(); ++i)
        t.b[i] = (uint8_t) (((5 + (rng() % 5)) << 3) | (rng() & 7));   // (E4M3 exp 5..9, man 0..7: scales 2^-1..2^2.875)
    return t;
}

// x rounded as the dense GEMVs round it: a scale a 32, max|x| / 127 (the device's __float2int_rn: nearest-even)
std::vector<float> q8_round(const float* x, int k) {
    std::vector<float> v(k);
    for (int b = 0; b < k / 32; ++b) {
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(x[b * 32 + j]));
        const float sc = am / 127.0f;
        for (int j = 0; j < 32; ++j)
            v[b * 32 + j] = am > 0.0f ? (float) ((double) std::nearbyint(x[b * 32 + j] / sc) * (double) sc) : 0.0f;
    }
    return v;
}
// rows rounded by the device as the routed experts' activations are (gg::act_quant: ggml's Q8_K when k % 256 == 0)
std::vector<float> act_dev(const float* dx, int rows, int k, cudaStream_t s) {
    int8_t* q = nullptr;
    float* d = nullptr;
    CU(cudaMalloc(&q, (size_t) rows * k));
    CU(cudaMalloc(&d, (size_t) rows * (k / 32) * 4));
    gg::act_quant(dx, rows, k, q, d, s);
    CU(cudaStreamSynchronize(s));
    const std::vector<int8_t> hq = host(q, (size_t) rows * k);
    const std::vector<float> hd = host(d, (size_t) rows * (k / 32));
    std::vector<float> v((size_t) rows * k);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float) ((double) hq[i] * (double) hd[i / 32]);
    cudaFree(q);
    cudaFree(d);
    return v;
}
double dot(const float* w, const float* x, int k) {
    double s = 0;
    for (int i = 0; i < k; ++i) s += (double) w[i] * x[i];
    return s;
}
double row_dot(const Blob& t, int r, const float* x, int k) {
    std::vector<float> wv(k);
    for (int c = 0; c < k; ++c) wv[c] = (float) t.w(r, c);
    return dot(wv.data(), x, k);
}
double oai(double g, double u, double lim) {
    g = std::min(g, lim);
    u = std::min(std::max(u, -lim), lim);
    return g / (1.0 + std::exp(-1.702 * g)) * (u + 1.0);
}

// ---------------------------------------------------------------- the dense GEMVs
void test_mv(cudaStream_t s) {
    const int k = 512, n = 96;
    const Blob t = blob_rand(n, k);
    uint8_t* dw = dev(t.b);
    x3::Mat m;
    m.fmt = nf::kFmtNVFP4;
    m.raw = dw;
    m.k = k;
    m.n = n;
    m.ld = 0;
    for (int nc : {1, 3, 16}) {
        const size_t xcs = (size_t) k + 32;
        std::vector<float> x(xcs * nc), bias(n);
        for (auto& v : x) v = frand(2.0f);
        for (auto& v : bias) v = frand(0.1f);
        float *dx = dev(x), *db = dev(bias), *dy = dev(std::vector<float>((size_t) n * nc));
        x3::MvJob j;
        j.m = &m;
        j.xf = dx;
        j.y = dy;
        j.bias = db;
        j.alpha = 0.75f;
        j.ncol = nc;
        j.xf_cs = xcs;
        j.y_cs = n;
        if (!x3::mv(&j, 1, s)) ++g_fail;
        CU(cudaStreamSynchronize(s));
        const std::vector<float> y = host(dy, (size_t) n * nc);
        std::vector<double> ref((size_t) n * nc);
        for (int c = 0; c < nc; ++c) {
            const std::vector<float> xr = q8_round(x.data() + (size_t) c * xcs, k);
            for (int r = 0; r < n; ++r)
                ref[(size_t) c * n + r] = 0.75 * row_dot(t, r, xr.data(), k) + bias[r];
        }
        char what[96];
        std::snprintf(what, sizeof what, "mv nvfp4, %d activations", nc);
        expect(what, y.data(), ref.data(), ref.size(), 2e-5);
        cudaFree(dx);
        cudaFree(db);
        cudaFree(dy);
    }
    cudaFree(dw);
}

// ---------------------------------------------------------------- the prompt path's GEMMs
void test_gemm(cudaStream_t s) {
    const int k = 512, n = 128, rows = 40;
    const Blob t = blob_rand(n, k);
    uint8_t* dw = dev(t.b);
    x3::Mat m;
    m.fmt = nf::kFmtNVFP4;
    m.raw = dw;
    m.k = k;
    m.n = n;
    m.ld = 0;
    for (float beta : {0.0f, 1.0f}) {
        const int ldx = k + 16, ldy = n + 8;
        std::vector<uint16_t> x16((size_t) rows * ldx);
        for (auto& v : x16) v = f2h(frand(2.0f));
        std::vector<float> y0((size_t) rows * ldy);
        for (auto& v : y0) v = frand();
        uint16_t* dx = dev(x16);
        float* dy = dev(y0);
        x3::gemm_rows(m, dx, ldx, rows, dy, ldy, beta, s);
        CU(cudaStreamSynchronize(s));
        const std::vector<float> y = host(dy, y0.size());
        std::vector<float> got;
        std::vector<double> ref;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < n; ++c) {
                double a = (double) beta * y0[(size_t) r * ldy + c];
                for (int i = 0; i < k; ++i) a += (double) f16r((float) t.w(c, i)) * h2f(x16[(size_t) r * ldx + i]);
                ref.push_back(a);
                got.push_back(y[(size_t) r * ldy + c]);
            }
        char what[96];
        std::snprintf(what, sizeof what, "gemm_rows nvfp4, %d rows, beta %g", rows, beta);
        expect(what, got.data(), ref.data(), ref.size(), 1e-4);
        cudaFree(dx);
        cudaFree(dy);
    }
    cudaFree(dw);
    {   // a matrix past one FP16 band (64 MB): its rows in several (k 8192 x n 4608 needs two)
        const int k2 = 8192, n2 = 4608, rows2 = 4;
        const Blob t2 = blob_rand(n2, k2);
        uint8_t* dw2 = dev(t2.b);
        x3::Mat m2;
        m2.fmt = nf::kFmtNVFP4;
        m2.raw = dw2;
        m2.k = k2;
        m2.n = n2;
        m2.ld = 0;
        std::vector<uint16_t> x16((size_t) rows2 * k2);
        for (auto& v : x16) v = f2h(frand());
        uint16_t* dx = dev(x16);
        float* dy = dev(std::vector<float>((size_t) rows2 * n2));
        x3::gemm_rows(m2, dx, k2, rows2, dy, n2, 0.0f, s);
        CU(cudaStreamSynchronize(s));
        const std::vector<float> y = host(dy, (size_t) rows2 * n2);
        std::vector<double> ref((size_t) rows2 * n2);
        for (int r = 0; r < rows2; ++r)
            for (int c = 0; c < n2; ++c) {
                double a = 0;
                for (int i = 0; i < k2; ++i) a += (double) f16r((float) t2.w(c, i)) * h2f(x16[(size_t) r * k2 + i]);
                ref[(size_t) r * n2 + c] = a;
            }
        expect("gemm_rows nvfp4 8192 x 4608 (two FP16 bands)", y.data(), ref.data(), ref.size(), 1e-4);
        cudaFree(dw2);
        cudaFree(dx);
        cudaFree(dy);
    }
}

// ---------------------------------------------------------------- the routed experts
struct QE {
    int D, F, n_exp;
    x3::ExpertLayout lay;
    size_t step = 0;
    std::vector<uint8_t> h;
    std::vector<Blob> w;   // [e * 3 + role]: the decoded reference, rows = outputs
    uint8_t* d = nullptr;
    const Blob& row(int e, int r) const { return w[(size_t) e * 3 + r]; }
};
QE make(int D, int F, int n_exp) {
    QE E;
    E.D = D;
    E.F = F;
    E.n_exp = n_exp;
    size_t off = 0;
    for (int r = 0; r < 3; ++r) {
        E.lay.trellis[r] = off;
        off += (size_t(4) + size_t(r < 2 ? F : D) * (size_t)((r < 2 ? D : F) / 2) +
                size_t(r < 2 ? F : D) * (size_t)((r < 2 ? D : F) / 16) + 255) / 256 * 256;
    }
    E.lay.fmt = nf::kFmtNVFP4;
    E.step = off;
    E.h.assign(E.step * (size_t) n_exp, 0);
    for (int e = 0; e < n_exp; ++e)
        for (int r = 0; r < 3; ++r) {
            const int rows = r < 2 ? F : D, k = r < 2 ? D : F;
            Blob t = blob_rand(rows, k);
            std::memcpy(E.h.data() + (size_t) e * E.step + E.lay.trellis[r], t.b.data(), t.b.size());
            E.w.push_back(std::move(t));
        }
    E.d = dev(E.h);
    return E;
}

void test_experts(cudaStream_t s) {
    const int D = 512, F = 256, n_exp = 10, K = 4, T = 3;
    const float lim = 7.0f;
    QE E = make(D, F, n_exp);
    // the plans: token 1's third pick off the device, token 2 with the CPU lane's part
    std::vector<int> pick((size_t) T * 8, 0);
    std::vector<float> pw((size_t) T * 8, 0.0f);
    std::vector<unsigned long long> pp((size_t) T * 8, 0ull);
    for (int t = 0; t < T; ++t) {
        std::vector<int> all(n_exp);
        for (int i = 0; i < n_exp; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        for (int i = 0; i < K; ++i) {
            pick[t * 8 + i] = all[i];
            pw[t * 8 + i] = 0.2f + 0.1f * std::fabs(frand());
            pp[t * 8 + i] = t == 1 && i == 2 ? 0ull : (unsigned long long) (E.d + (size_t) all[i] * E.step);
        }
    }
    std::vector<float> x((size_t) T * D), sh((size_t) T * D), cpu((size_t) T * D);
    for (auto& v : x) v = frand();
    for (auto& v : sh) v = frand(0.1f);
    for (auto& v : cpu) v = frand(0.1f);
    std::vector<int> cflag(T, 0);
    cflag[2] = 1;
    float *dx = dev(x), *dsh = dev(sh), *dcpu = dev(cpu), *dpw = dev(pw);
    unsigned long long* dpp = dev(pp);
    int* dflag = dev(cflag);
    float* dh = dev(std::vector<float>((size_t) T * 8 * F));
    std::vector<float> tok((size_t) T * D), gh((size_t) T * 8 * F);
    for (int t = 0; t < T; ++t) {
        x3::MoeArgs a;
        a.plan_ptr = dpp + (size_t) t * 8;
        a.plan_w = dpw + (size_t) t * 8;
        a.cpu_part = dcpu + (size_t) t * D;
        a.cpu_flag = dflag + t;
        a.lay = E.lay;
        a.k = K;
        a.n_embd = D;
        a.n_ff = F;
        a.limit = lim;
        float* dout = dev(std::vector<float>(D));
        x3::moe_gate_up(a, dx + (size_t) t * D, dh, s);
        x3::moe_down(a, dh, dsh + (size_t) t * D, dout, s);
        CU(cudaStreamSynchronize(s));
        CU(cudaMemcpy(tok.data() + (size_t) t * D, dout, (size_t) D * 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(gh.data() + (size_t) t * 8 * F, dh, (size_t) K * F * 4, cudaMemcpyDeviceToHost));
        cudaFree(dout);
    }
    // the reference: gate / up in double on the device's rounded x, the down on the device's own h re-rounded
    const std::vector<float> xr = act_dev(dx, T, D, s);
    std::vector<double> want((size_t) T * D, 0.0), hwant;
    std::vector<float> hgot;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < K; ++i) {
            if (pp[t * 8 + i] == 0ull) continue;
            const int e = pick[t * 8 + i];
            for (int f = 0; f < F; ++f) {
                const double g = row_dot(E.row(e, 0), f, xr.data() + (size_t) t * D, D);
                const double u = row_dot(E.row(e, 1), f, xr.data() + (size_t) t * D, D);
                hwant.push_back(oai(g, u, lim));
                hgot.push_back(gh[(size_t)(t * 8 + i) * F + f]);
            }
        }
    }
    // (the down pass per entry: the device's own h rows, re-rounded as the experts' activations are)
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i) {
            if (pp[t * 8 + i] == 0ull) continue;
            const int e = pick[t * 8 + i];
            float* dhi = dev(std::vector<float>(gh.begin() + (long)((size_t)(t * 8 + i) * F),
                                                gh.begin() + (long)((size_t)(t * 8 + i + 1) * F)));
            const std::vector<float> hr = act_dev(dhi, 1, F, s);
            cudaFree(dhi);
            for (int d = 0; d < D; ++d)
                want[(size_t) t * D + d] += (double) pw[t * 8 + i] * row_dot(E.row(e, 2), d, hr.data(), F);
        }
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            want[(size_t) t * D + d] += (cflag[t] ? cpu[(size_t) t * D + d] : 0.0f) + sh[(size_t) t * D + d];
    char what[128];
    std::snprintf(what, sizeof what, "nvfp4 token experts: h (swigluoai)");
    expect(what, hgot.data(), hwant.data(), hwant.size(), 1e-4);
    std::snprintf(what, sizeof what, "nvfp4 token experts: out");
    expect(what, tok.data(), want.data(), want.size(), 1e-4);

    // the window: each distinct expert once for its tokens - bit for bit the token path's
    std::vector<unsigned long long> gptr;
    std::vector<int> gstart, gent;
    for (int e0 = 0; e0 < n_exp; ++e0) {
        std::vector<int> ents;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < K; ++i)
                if (pick[t * 8 + i] == e0 && pp[t * 8 + i] != 0ull) ents.push_back(t * 8 + i);
        if (ents.empty()) continue;
        gptr.push_back((unsigned long long) (E.d + (size_t) e0 * E.step));
        gstart.push_back((int) gent.size());
        for (int e : ents) gent.push_back(e);
    }
    gstart.push_back((int) gent.size());
    std::vector<int> ng(1, (int) gptr.size());
    unsigned long long* dgp = dev(gptr);
    int *dgs = dev(gstart), *dge = dev(gent), *dng = dev(ng);
    float* dwh = dev(std::vector<float>((size_t) T * 8 * F));
    float* dpart = dev(std::vector<float>((size_t) T * 8 * D));
    float* dwo = dev(std::vector<float>((size_t) T * D));
    x3::WinArgs w;
    w.grp_ptr = dgp;
    w.grp_start = dgs;
    w.grp_ent = dge;
    w.n_groups = dng;
    w.plan_ptr = dpp;
    w.plan_w = dpw;
    w.cpu_part = dcpu;
    w.cpu_stride = D;
    w.cpu_flag = dflag;
    w.x = dx;
    w.x_stride = D;
    w.h = dwh;
    w.part = dpart;
    w.sh_out = dsh;
    w.out = dwo;
    w.out_stride = D;
    w.lay = E.lay;
    w.T = T;
    w.k = K;
    w.n_embd = D;
    w.n_ff = F;
    w.limit = lim;
    if (!x3::win_experts(w, s)) ++g_fail;
    CU(cudaStreamSynchronize(s));
    const std::vector<float> win = host(dwo, (size_t) T * D);
    std::snprintf(what, sizeof what, "nvfp4 window (T %d) = the token path's", T);
    same(what, win.data(), tok.data(), win.size() * 4);

    // the prompt sets: a few owners' gathered rows against the double reference (the set kernels' own per-32
    // staging: the reference rounds x the same way)
    for (int R : {60, 200}) {
        std::vector<int> owner(R), idx(R);
        for (int r = 0; r < R; ++r) owner[r] = (int) (rng() % n_exp);
        std::sort(owner.begin(), owner.end());
        std::vector<int> bounds(n_exp + 1, 0);
        for (int e = 0, r = 0; e <= n_exp; ++e) {
            while (r < R && owner[r] < e) ++r;
            bounds[e] = r;
        }
        int max_rows = 0;
        for (int e = 0; e < n_exp; ++e) max_rows = std::max(max_rows, bounds[e + 1] - bounds[e]);
        std::vector<float> xs((size_t) 30 * D);
        for (auto& v : xs) v = frand();
        for (int r = 0; r < R; ++r) idx[r] = (int) (rng() % 30);
        float* dxs = dev(xs);
        int *dbd = dev(bounds), *didx = dev(idx);
        uint16_t* dh16 = dev(std::vector<uint16_t>((size_t) R * F));
        float* dy = dev(std::vector<float>((size_t) R * D));
        x3::SetArgs sa;
        sa.wbase = E.d;
        sa.stride = E.step;
        sa.lay = E.lay;
        sa.n_exp = n_exp;
        sa.bounds = dbd;
        sa.row0 = 0;
        sa.max_rows = max_rows;
        sa.n_embd = D;
        sa.n_ff = F;
        sa.limit = lim;
        x3::set_gate_up(sa, dxs, D, didx, dh16, F, s);
        x3::set_down(sa, dh16, F, dy, D, s);
        CU(cudaStreamSynchronize(s));
        const std::vector<uint16_t> h16 = host(dh16, (size_t) R * F);
        const std::vector<float> y = host(dy, (size_t) R * D);
        std::vector<double> hw((size_t) R * F), yw((size_t) R * D);
        std::vector<float> hg((size_t) R * F);
        for (int r = 0; r < R; ++r) {
            const int e = owner[r];
            std::vector<float> xi(xs.begin() + (long) idx[r] * D, xs.begin() + (long) (idx[r] + 1) * D);
            xi = q8_round(xi.data(), D);
            for (int f = 0; f < F; ++f) {
                hw[(size_t) r * F + f] = oai(row_dot(E.row(e, 0), f, xi.data(), D),
                                             row_dot(E.row(e, 1), f, xi.data(), D), lim);
                hg[(size_t) r * F + f] = h2f(h16[(size_t) r * F + f]);
            }
            std::vector<float> hr(F);
            for (int f = 0; f < F; ++f) hr[f] = h2f(h16[(size_t) r * F + f]);
            hr = q8_round(hr.data(), F);
            for (int d = 0; d < D; ++d) yw[(size_t) r * D + d] = row_dot(E.row(e, 2), d, hr.data(), F);
        }
        std::snprintf(what, sizeof what, "nvfp4 set gate/up (F16 h), %d rows", R);
        expect(what, hg.data(), hw.data(), hw.size(), 2e-3);
        std::snprintf(what, sizeof what, "nvfp4 set down, %d rows", R);
        expect(what, y.data(), yw.data(), yw.size(), 1e-4);
        cudaFree(dxs);
        cudaFree(dbd);
        cudaFree(didx);
        cudaFree(dh16);
        cudaFree(dy);
    }
    cudaFree(E.d);
    cudaFree(dx);
    cudaFree(dsh);
    cudaFree(dcpu);
    cudaFree(dpw);
    cudaFree(dpp);
    cudaFree(dflag);
    cudaFree(dh);
    cudaFree(dgp);
    cudaFree(dgs);
    cudaFree(dge);
    cudaFree(dng);
    cudaFree(dwh);
    cudaFree(dpart);
    cudaFree(dwo);
}

}  // namespace

int main(int argc, char** argv) {
    cudaStream_t s;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const std::string only = argc >= 2 ? argv[1] : "";
    // the manifest vocabulary: "NVFP4" -> 40 (upstream's claim; the dispatch keys on fmt 6), 4.0625 B a value row
    if (nf::type_of("NVFP4") != 40 || nf::type_of("Q4_K") != -1 || nf::row_bytes(512) != 256 + 32) {
        std::printf("%-52s FAIL\n", "type_of / row_bytes vocabulary");
        ++g_fail;
    } else {
        std::printf("%-52s ok\n", "type_of / row_bytes vocabulary");
    }
    if (!nf::prepare(s)) {
        std::printf("%-52s FAIL  prepare\n", "nf::prepare");
        ++g_fail;
    }
    if (only.empty() || only == "--mv") test_mv(s);
    if (only.empty() || only == "--gemm") test_gemm(s);
    if (only.empty() || only == "--experts") test_experts(s);
    const int errs = x3::launch_errors();
    if (errs != 0) {
        std::printf("launch errors: %d\n", errs);
        ++g_fail;
    }
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
