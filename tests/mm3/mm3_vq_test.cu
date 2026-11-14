// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.63).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// tests/mm3/mm3_vq_test.cu - the Baekpica VQ weights (GVQ1 + GVR8) against a scalar host reference: the packing
// law (13-bit indices, the high1 plane, the F16 gains), the mv GEMV, the FP16 row decode + cuBLAS GEMM, and the
// routed experts (swigluoai fused, the plan combine) on the token and window paths.  Synthetic payloads built
// with the same law the pack writes (FORMAT.md), decoded on the host in doubles.
//
//   ./mm3_vq_test           (random matrices, no model files)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../src/mm3/mm3_vq.cuh"
#include "../src/mm3/mm3_vq_host.hpp"

static int g_failed = 0, g_checks = 0;
static void check(bool ok, const char* what, double rel = -1.0) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::printf("  %-46s FAIL%s\n", what, rel >= 0 ? "" : "");
    } else if (rel >= 0) {
        std::printf("  %-46s ok    rel %.3e\n", what, rel);
    } else {
        std::printf("  %-46s ok\n", what);
    }
}

// ------------------------------------------------------------------ the format, written and read on the host

static float h_e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float mag = e == 0 ? (float) m / 512.0f : std::ldexp((8.0f + m) / 8.0f, e - 7);
    return (b & 0x80) ? -mag : mag;
}
static uint16_t f32_to_f16(float v) { __half h = __float2half_rn(v); return __half_as_ushort(h); }

struct Built {   // one GVQ1 (+ optional GVR8) matrix, its device buffers and the decoded reference rows
    std::vector<uint8_t> payload, residual;
    std::vector<uint16_t> gain;
    std::vector<double> w;      // [rows][k] the decoded values (gain applied)
    int rows, cols, k;
    bool has_res = false;
    void* d_payload = nullptr;
    void* d_residual = nullptr;
};

static void write_header(std::vector<uint8_t>& p, uint32_t rows, uint32_t cols, uint32_t entries, uint32_t cb_id,
                         uint64_t gain_off, uint64_t index_off, uint64_t high_off, uint64_t total) {
    p.resize(128);
    std::memcpy(p.data(), entries == 256 ? "GVR1" : "GVQ1", 4);
    auto put16 = [&](size_t o, uint16_t v) { std::memcpy(p.data() + o, &v, 2); };
    auto put32 = [&](size_t o, uint32_t v) { std::memcpy(p.data() + o, &v, 4); };
    auto put64 = [&](size_t o, uint64_t v) { std::memcpy(p.data() + o, &v, 8); };
    put16(4, 1);
    p[6] = entries == 256 ? 8 : 13;
    p[7] = 8;
    put32(8, rows);
    put32(12, cols);
    put32(16, entries);
    put32(20, 1);   // E4M3FN
    put32(24, cb_id);
    put32(28, 0);
    put64(32, gain_off);
    put64(40, index_off);
    put64(48, high_off);
    put64(56, total);
}

static Built build_matrix(std::mt19937& rnd, int rows, int k_values, uint32_t cb_id, bool with_res,
                          const std::vector<uint8_t>& cb1, const std::vector<uint8_t>& cb2) {
    Built b;
    b.rows = rows;
    b.cols = k_values;   // the WIRE law: the header's cols is k (values a row); vectors = k / 8
    const int cols = k_values / 8;
    b.k = k_values;
    b.has_res = with_res;
    const uint64_t n = (uint64_t) rows * cols;
    // gains: positive F16s; 13-bit indices into cb1; 8-bit indices into cb2 when a residual
    std::uniform_real_distribution<float> gain_d(0.25f, 2.0f);
    b.gain.resize(rows);
    for (int r = 0; r < rows; ++r) b.gain[r] = f32_to_f16(gain_d(rnd));
    std::vector<uint32_t> i1(n);
    std::vector<uint8_t> i2(with_res ? n : 0);
    for (uint64_t j = 0; j < n; ++j) i1[j] = rnd() % 8192;
    if (with_res)
        for (uint64_t j = 0; j < n; ++j) i2[j] = uint8_t(rnd() % 256);
    // the low12 stream: pairs a,b as [a&255, ((a>>8)&15)|((b&15)<<4), (b>>4)&255]; high1: bit12 of j at j/8 %8
    std::vector<uint8_t> low12((size_t)((n + 1) / 2 * 3)), high1((size_t)((n + 7) / 8));
    for (uint64_t m = 0; m < (n + 1) / 2; ++m) {
        const uint32_t a = i1[2 * m];
        const uint32_t bb = 2 * m + 1 < n ? i1[2 * m + 1] : 0;
        low12[3 * m] = uint8_t(a & 255);
        low12[3 * m + 1] = uint8_t(((a >> 8) & 15) | ((bb & 15) << 4));
        low12[3 * m + 2] = uint8_t((bb >> 4) & 255);
    }
    for (uint64_t j = 0; j < n; ++j)
        if (i1[j] & 0x1000) high1[j / 8] |= uint8_t(1u << (j % 8));
    const uint64_t gain_off = 128, index_off = (gain_off + 2 * rows + 127) / 128 * 128;
    const uint64_t high_off = (index_off + low12.size() + 127) / 128 * 128;
    const uint64_t total = (high_off + high1.size() + 127) / 128 * 128;
    write_header(b.payload, rows, (uint32_t) b.k, 8192, cb_id, gain_off, index_off, high_off, total);
    b.payload.resize(total, 0);
    std::memcpy(b.payload.data() + gain_off, b.gain.data(), 2 * rows);
    std::memcpy(b.payload.data() + index_off, low12.data(), low12.size());
    std::memcpy(b.payload.data() + high_off, high1.data(), high1.size());
    if (with_res) {   // GVR8: header + one index byte a vector
        const uint64_t io = 128, tot = (io + i2.size() + 127) / 128 * 128;
        write_header(b.residual, rows, (uint32_t) b.k, 256, cb_id + 1, 0, io, 0, tot);
        b.residual.resize(tot, 0);
        std::memcpy(b.residual.data() + io, i2.data(), i2.size());
    }
    // the decoded reference (doubles)
    b.w.assign((size_t) rows * b.k, 0.0);
    for (int r = 0; r < rows; ++r) {
        const double gd = __half2float(__ushort_as_half(b.gain[r]));
        for (int v = 0; v < cols; ++v) {
            const uint64_t j = (uint64_t) r * cols + v;
            for (int d = 0; d < 8; ++d) {
                double val = h_e4m3(cb1[8 * (size_t) i1[j] + d]);
                if (with_res) val += h_e4m3(cb2[8 * (size_t) i2[j] + d]);
                b.w[(size_t) r * b.k + 8 * v + d] = gd * val;
            }
        }
    }
    cudaMalloc(&b.d_payload, b.payload.size());
    cudaMemcpy(b.d_payload, b.payload.data(), b.payload.size(), cudaMemcpyHostToDevice);
    if (with_res) {
        cudaMalloc(&b.d_residual, b.residual.size());
        cudaMemcpy(b.d_residual, b.residual.data(), b.residual.size(), cudaMemcpyHostToDevice);
    }
    return b;
}

int main() {
    std::mt19937 rnd(42);
    // the codebooks: finite E4M3 bytes (e 15 + m 7 is the NaN class; keep below it)
    std::vector<uint8_t> cb1(8192 * 8), cb2(256 * 8);
    for (auto& b : cb1) b = uint8_t(rnd() % 256) & 0x77;
    for (auto& b : cb2) b = uint8_t(rnd() % 256) & 0x77;
    void *d_cb1 = nullptr, *d_cb2 = nullptr;
    cudaMalloc(&d_cb1, cb1.size());
    cudaMemcpy(d_cb1, cb1.data(), cb1.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_cb2, cb2.size());
    cudaMemcpy(d_cb2, cb2.data(), cb2.size(), cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    mm3::vq::prepare(s);

    // ---- mv (dense GEMV), single column and three
    {
        Built b = build_matrix(rnd, 256, 96, 0, true, cb1, cb2);   // k = 768, n = 256
        strata::mm3::exl3::Mat m;
        m.fmt = mm3::vq::kFmtVQ;
        m.raw = b.d_payload;
        m.vq_cb = d_cb1;
        m.vq_res = b.d_residual;
        m.vq_cb2 = d_cb2;
        m.k = b.k;
        m.n = b.rows;
        std::vector<float> x(b.k);
        for (auto& v : x) v = -1.0f + 2.0f * (rnd() % 1000) / 999.0f;
        float* d_x = nullptr;
        cudaMalloc(&d_x, x.size() * 4);
        cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        std::vector<float> y(b.rows);
        float* d_y = nullptr;
        cudaMalloc(&d_y, b.rows * 4);
        strata::mm3::exl3::MvJob job;
        job.m = &m;
        job.xf = d_x;
        job.y = d_y;
        mm3::vq::mv(&job, 1, s);
        cudaMemcpy(y.data(), d_y, b.rows * 4, cudaMemcpyDeviceToHost);
        std::vector<double> want(b.rows);
        for (int r = 0; r < b.rows; ++r) {
            double acc = 0.0;
            for (int c = 0; c < b.k; ++c) acc += (double) x[c] * b.w[(size_t) r * b.k + c];
            want[r] = acc;
        }
        double worst = 0.0;
        for (int r = 0; r < b.rows; ++r) worst = std::max(worst, std::fabs(y[r] - want[r]) / (std::fabs(want[r]) + 1e-3));
        check(worst < 1e-3, "mv residual matrix vs double reference", worst);
        // three activations (their own buffers - the column stride is the job's contract, not a realloc)
        std::vector<float> x3v(b.k * 3);
        for (auto& v : x3v) v = -1.0f + 2.0f * (rnd() % 1000) / 999.0f;
        float* d_x3 = nullptr;
        float* d_y3 = nullptr;
        cudaMalloc(&d_x3, x3v.size() * 4);
        cudaMemcpy(d_x3, x3v.data(), x3v.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_y3, (size_t) b.rows * 3 * 4);
        std::vector<float> y3(b.rows * 3);
        job.xf = d_x3;
        job.y = d_y3;
        job.ncol = 3;
        job.xf_cs = b.k * 4;
        job.y_cs = b.rows * 4;
        mm3::vq::mv(&job, 1, s);
        cudaMemcpy(y3.data(), d_y3, b.rows * 3 * 4, cudaMemcpyDeviceToHost);
        worst = 0.0;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < b.rows; ++r) {
                double acc = 0.0;
                for (int k = 0; k < b.k; ++k) acc += (double) x3v[c * b.k + k] * b.w[(size_t) r * b.k + k];
                const double rel = std::fabs(y3[c * b.rows + r] - acc) / (std::fabs(acc) + 1e-3);
                if (rel > worst) worst = rel;
            }
        check(worst < 1e-3, "mv ncol 3", worst);   // F32 FMA-chain vs double at k 768: the 1e-4..1e-3 class
    }

    // ---- f16_rows + gemm_rows (the prompt path)
    {
        Built b = build_matrix(rnd, 384, 128, 4, false, cb1, cb2);   // k = 1024, n = 384
        strata::mm3::exl3::Mat m;
        m.fmt = mm3::vq::kFmtVQ;
        m.raw = b.d_payload;
        m.vq_cb = d_cb1;
        m.k = b.k;
        m.n = b.rows;
        const int rows = 40;
        std::vector<float> x((size_t) rows * b.k);
        for (auto& v : x) v = -1.0f + 2.0f * (rnd() % 1000) / 999.0f;
        uint16_t* d_x16 = nullptr;
        cudaMalloc(&d_x16, x.size() * 2);
        std::vector<uint16_t> x16(x.size());
        for (size_t i = 0; i < x.size(); ++i) x16[i] = __half_as_ushort(__float2half_rn(x[i]));
        cudaMemcpy(d_x16, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice);
        std::vector<float> y((size_t) rows * b.rows);
        float* d_y = nullptr;
        cudaMalloc(&d_y, y.size() * 4);
        mm3::vq::gemm_rows(m, d_x16, b.k, rows, d_y, b.rows, 0.0f, s);
        cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost);
        double worst = 0.0;
        for (int r = 0; r < rows; ++r)
            for (int n_ = 0; n_ < b.rows; ++n_) {
                // the reference over the FP16-rounded weights (the device decodes to FP16 before the GEMM)
                double acc = 0.0;
                for (int k = 0; k < b.k; ++k) {
                    const double wf = (double) __half2float(__float2half_rn((float) b.w[(size_t) n_ * b.k + k]));
                    acc += (double) __half2float(__ushort_as_half(x16[(size_t) r * b.k + k])) * wf;
                }
                worst = std::max(worst, std::fabs(y[(size_t) r * b.rows + n_] - acc) / (std::fabs(acc) + 1e-2));
            }
        check(worst < 2e-2, "gemm_rows (FP16 weights) vs reference", worst);
    }

    // ---- the routed experts: a token (moe_gate_up + moe_down) and a window of three
    {
        const int n_embd = 512, n_ff = 256, k = 4;
        // one blob: [cb 64K][cb2 2K][w1][w3][w2][r1][r3][r2] - gate/up [n_ff x n_embd], down [n_embd x n_ff]
        Built g = build_matrix(rnd, n_ff, n_embd / 8, 0, true, cb1, cb2);
        Built u = build_matrix(rnd, n_ff, n_embd / 8, 0, true, cb1, cb2);
        Built d_ = build_matrix(rnd, n_embd, n_ff / 8, 0, true, cb1, cb2);
        std::vector<uint8_t> blob(64 * 1024 + 2 * 1024);
        const size_t o_w1 = blob.size(), o_w3 = o_w1 + g.payload.size(), o_w2 = o_w3 + u.payload.size();
        const size_t o_r1 = o_w2 + d_.payload.size(), o_r3 = o_r1 + g.residual.size(), o_r2 = o_r3 + u.residual.size();
        blob.insert(blob.end(), g.payload.begin(), g.payload.end());
        blob.insert(blob.end(), u.payload.begin(), u.payload.end());
        blob.insert(blob.end(), d_.payload.begin(), d_.payload.end());
        blob.insert(blob.end(), g.residual.begin(), g.residual.end());
        blob.insert(blob.end(), u.residual.begin(), u.residual.end());
        blob.insert(blob.end(), d_.residual.begin(), d_.residual.end());
        std::memcpy(blob.data(), cb1.data(), std::min(cb1.size(), (size_t) 64 * 1024));
        std::memcpy(blob.data() + 64 * 1024, cb2.data(), cb2.size());
        void* d_blob = nullptr;
        cudaMalloc(&d_blob, blob.size());
        cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        unsigned long long plan[k];
        float wts[k];
        for (int i = 0; i < k; ++i) {
            plan[i] = (unsigned long long) d_blob;
            wts[i] = 0.25f;
        }
        unsigned long long* d_plan = nullptr;
        float* d_wts = nullptr;
        cudaMalloc(&d_plan, sizeof(plan));
        cudaMemcpy(d_plan, plan, sizeof(plan), cudaMemcpyHostToDevice);
        cudaMalloc(&d_wts, sizeof(wts));
        cudaMemcpy(d_wts, wts, sizeof(wts), cudaMemcpyHostToDevice);
        strata::mm3::exl3::ExpertLayout lay;
        lay.fmt = mm3::vq::kFmtVQ;
        lay.trellis[0] = o_w1;
        lay.trellis[1] = o_w3;
        lay.trellis[2] = o_w2;
        lay.vq_res[0] = o_r1;
        lay.vq_res[1] = o_r3;
        lay.vq_res[2] = o_r2;
        lay.vq_cb = 0;
        lay.vq_cb2 = 64 * 1024;
        std::vector<float> x(n_embd);
        for (auto& v : x) v = -1.0f + 2.0f * (rnd() % 1000) / 999.0f;
        float* d_x = nullptr;
        cudaMalloc(&d_x, n_embd * 4);
        cudaMemcpy(d_x, x.data(), n_embd * 4, cudaMemcpyHostToDevice);
        float* d_out = nullptr;
        cudaMalloc(&d_out, n_embd * 4);
        strata::mm3::exl3::MoeArgs a;
        a.plan_ptr = d_plan;
        a.plan_w = d_wts;
        a.lay = lay;
        a.k = k;
        a.n_embd = n_embd;
        a.n_ff = n_ff;
        a.limit = 7.0f;
        mm3::vq::moe_gate_up(a, d_x, nullptr, s);
        mm3::vq::moe_down(a, nullptr, nullptr, d_out, s);
        std::vector<float> out(n_embd);
        cudaMemcpy(out.data(), d_out, n_embd * 4, cudaMemcpyDeviceToHost);
        double worst = 0.0;
        const float alpha = strata::mm3::exl3::kSwigluAlpha, lim = 7.0f;
        for (int r = 0; r < n_embd; ++r) {
            double acc = 0.0, dv = 0.0;
            for (int c = 0; c < n_ff; ++c) {
                double gr = 0.0, ur = 0.0;
                for (int cc = 0; cc < n_embd; ++cc) {
                    gr += (double) x[cc] * g.w[(size_t) c * n_embd + cc];
                    ur += (double) x[cc] * u.w[(size_t) c * n_embd + cc];
                }
                float gf = std::min((float) gr, lim), uf = std::min(std::max((float) ur, -lim), lim);
                const float h = gf * (1.0f / (1.0f + std::exp(-alpha * gf))) * (uf + 1.0f);
                dv += (double) h * d_.w[(size_t) r * n_ff + c];
            }
            acc += k * 0.25 * dv;   // k experts at the same blob, weight 0.25 each
            worst = std::max(worst, std::fabs(out[r] - acc) / (std::fabs(acc) + 1e-2));
        }
        check(worst < 3e-3, "moe token path vs reference", worst);

        // ---- window of three tokens, its own buffers
        const int T = 3;
        std::vector<float> xw((size_t) T * n_embd), outw((size_t) T * n_embd);
        for (auto& v : xw) v = -1.0f + 2.0f * (rnd() % 1000) / 999.0f;
        float *d_xw = nullptr, *d_outw = nullptr, *d_h = nullptr, *d_part = nullptr;
        cudaMalloc(&d_xw, xw.size() * 4);
        cudaMemcpy(d_xw, xw.data(), xw.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_outw, outw.size() * 4);
        cudaMalloc(&d_h, (size_t) T * k * n_ff * 4);
        cudaMalloc(&d_part, (size_t) T * k * n_embd * 4);
        unsigned long long* d_planw = nullptr;
        std::vector<unsigned long long> planw((size_t) T * k);
        float* d_wtsw = nullptr;
        std::vector<float> wtsw((size_t) T * k, 0.25f);
        for (auto& pw : planw) pw = (unsigned long long) d_blob;
        cudaMalloc(&d_planw, planw.size() * 8);
        cudaMemcpy(d_planw, planw.data(), planw.size() * 8, cudaMemcpyHostToDevice);
        cudaMalloc(&d_wtsw, wtsw.size() * 4);
        cudaMemcpy(d_wtsw, wtsw.data(), wtsw.size() * 4, cudaMemcpyHostToDevice);
        strata::mm3::exl3::WinArgs wa;
        wa.plan_ptr = d_planw;
        wa.plan_w = d_wtsw;
        wa.x = d_xw;
        wa.x_stride = n_embd * 4;
        wa.h = d_h;
        wa.part = d_part;
        wa.out = d_outw;
        wa.out_stride = n_embd * 4;
        wa.lay = lay;
        wa.T = T;
        wa.k = k;
        wa.n_embd = n_embd;
        wa.n_ff = n_ff;
        wa.limit = 7.0f;
        check(mm3::vq::win_experts(wa, s), "win_experts launched");
        cudaMemcpy(outw.data(), d_outw, outw.size() * 4, cudaMemcpyDeviceToHost);
        worst = 0.0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < n_embd; ++r) {
                double acc = 0.0;
                for (int c = 0; c < n_ff; ++c) {
                    double gr = 0.0, ur = 0.0;
                    for (int cc = 0; cc < n_embd; ++cc) {
                        gr += (double) xw[(size_t) t * n_embd + cc] * g.w[(size_t) c * n_embd + cc];
                        ur += (double) xw[(size_t) t * n_embd + cc] * u.w[(size_t) c * n_embd + cc];
                    }
                    float gf = std::min((float) gr, lim), uf = std::min(std::max((float) ur, -lim), lim);
                    const float h = gf * (1.0f / (1.0f + std::exp(-alpha * gf))) * (uf + 1.0f);
                    acc += (double) h * d_.w[(size_t) r * n_ff + c];
                }
                acc *= k * 0.25;
                worst = std::max(worst, std::fabs(outw[(size_t) t * n_embd + r] - acc) / (std::fabs(acc) + 1e-2));
            }
        check(worst < 3e-3, "win_experts T3 vs per-token reference", worst);
    }

    std::printf("mm3_vq_test: %s (%d failed of %d)\n", g_failed ? "FAILED" : "ok", g_failed, g_checks);
    return g_failed ? 1 : 0;
}
