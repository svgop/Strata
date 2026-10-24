// campaign/headport/head_requant_probe.cpp — the F-298 head-requant offline falsifier.
// Loads the REAL head tensor (head_q6k.bin, Q6_K [2560 x 248320] extracted from the GGUF),
// dequantizes every row, packs a Q4_0 copy (4.5 bits/w vs Q6_K's 6.6 — the quality-floored
// lower bound; Q4_K would sit above it), and measures, for N random N(0,1) activation
// vectors x: the exact (Q6_K) logits vs the Q4_0 logits — top-1 flip rate (the emitted-token
// contract), KL(exact || requant), and the max logit delta.  CPU-only, zero boots.
//   head_requant_probe [rows] [xsamples]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

constexpr int kNin = 2560, kBpr = 10, kBlk = 210, kQ4Blk = 32;

static void dequant_row_q6k(const uint8_t* row, float* out) {
    for (int b = 0; b < kBpr; ++b) {
        const uint8_t* bp = row + (size_t) b * kBlk;
        const uint8_t *ql = bp, *qh = bp + 128;
        const int8_t* sc = (const int8_t*) (bp + 192);
        uint16_t dh; std::memcpy(&dh, bp + 208, 2);
        const float d = __half2float(*(__half*) &dh);
        for (int n = 0; n < 2; ++n)
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[64 * n + l] & 0xF) | (((qh[32 * n + l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[64 * n + l + 32] & 0xF) | (((qh[32 * n + l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[64 * n + l] >> 4) | (((qh[32 * n + l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[64 * n + l + 32] >> 4) | (((qh[32 * n + l] >> 6) & 3) << 4)) - 32;
                float* y = out + (size_t) b * 256 + 128 * n;
                y[l]      = d * sc[8 * n + is] * q1;
                y[l + 32] = d * sc[8 * n + is + 2] * q2;
                y[l + 64] = d * sc[8 * n + is + 4] * q3;
                y[l + 96] = d * sc[8 * n + is + 6] * q4;
            }
    }
}

// ggml Q4_0: per 32 weights, fp16 d + 16 bytes of 4-bit pairs (row-major blocks)
struct Q4Row { __half d[kNin / kQ4Blk]; uint8_t qs[kNin / 2]; };

static void quant_row_q4_0(const float* w, Q4Row* out) {
    uint8_t* q = (uint8_t*) out;
    for (int blk = 0; blk < kNin / kQ4Blk; ++blk) {
        const float* x = w + (size_t) blk * kQ4Blk;
        float amax = 0; for (int i = 0; i < kQ4Blk; ++i) amax = std::fmax(amax, std::fabs(x[i]));
        const float d = amax / -8;               // ggml convention: q in [-8, 7], d negative-signed
        (__half&) q[blk * 18] = __float2half(d ? d : 1e-12f);
        uint8_t* qs = q + (size_t) blk * 18 + 2;
        for (int i = 0; i < 16; ++i) {
            int q0 = d ? std::lround(x[i] / d) : 0;
            int q1 = d ? std::lround(x[16 + i] / d) : 0;
            q0 = std::min(15, std::max(0, q0 + 8));
            q1 = std::min(15, std::max(0, q1 + 8));
            qs[i] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}
static void dequant_row_q4_0(const Q4Row* r, float* out) {
    const uint8_t* q = (const uint8_t*) r;
    for (int blk = 0; blk < kNin / kQ4Blk; ++blk) {
        const float d = __half2float((const __half&) q[blk * 18]);
        const uint8_t* qs = q + (size_t) blk * 18 + 2;
        for (int i = 0; i < 16; ++i) {
            const int q0 = (qs[i] & 0xF) - 8, q1 = (qs[i] >> 4) - 8;
            out[(size_t) blk * 32 + i] = d * q0;
            out[(size_t) blk * 32 + 16 + i] = d * q1;
        }
    }
}

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 248320;
    const int xs = argc > 2 ? std::atoi(argv[2]) : 8;
    const char* path = argc > 3 ? argv[3] : "campaign/headport/head_q6k.bin";
    const size_t wbytes = (size_t) rows * kBpr * kBlk;
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "open fail %s\n", path); return 2; }
    std::vector<uint8_t> q6(wbytes);
    if (std::fread(q6.data(), 1, wbytes, f) != wbytes) { std::fprintf(stderr, "read fail\n"); return 2; }
    std::fclose(f);
    std::fprintf(stderr, "loaded %s: %zu bytes (%d rows)\n", path, wbytes, rows);

    // dequantize + pack Q4_0 row by row (streamed; ~2.5 GB peak for the Q4 copy)
    std::vector<Q4Row> q4((size_t) rows);
    {
        std::vector<float> w(kNin);
        for (int r = 0; r < rows; ++r) {
            dequant_row_q6k(q6.data() + (size_t) r * kBpr * kBlk, w.data());
            quant_row_q4_0(w.data(), &q4[(size_t) r]);
        }
    }
    std::fprintf(stderr, "packed Q4_0 (%.1f MB vs %.1f MB Q6_K = %.2fx)\n",
                 rows * sizeof(Q4Row) / 1e6, wbytes / 1e6, wbytes / (double) (rows * sizeof(Q4Row)));

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(kNin), w(kNin), wq(kNin);
    long flips = 0, tested = 0; double kl_sum = 0, maxdelta = 0;
    std::vector<std::pair<float, int>> lexact((size_t) rows), lq4((size_t) rows);
    for (int s = 0; s < xs; ++s) {
        for (auto& v : x) v = nd(rng);
        for (int r = 0; r < rows; ++r) {
            dequant_row_q6k(q6.data() + (size_t) r * kBpr * kBlk, w.data());
            dequant_row_q4_0(&q4[(size_t) r], wq.data());
            float e = 0, q = 0;
            for (int i = 0; i < kNin; ++i) { e += w[i] * x[i]; q += wq[i] * x[i]; }
            lexact[(size_t) r] = {e, r}; lq4[(size_t) r] = {q, r};
            maxdelta = std::fmax(maxdelta, std::fabs(e - q));
        }
        // top-1 flip
        const int t1e = std::max_element(lexact.begin(), lexact.end())->second;
        const int t1q = std::max_element(lq4.begin(), lq4.end())->second;
        flips += (t1e != t1q); ++tested;
        // KL(exact || q4) over softmax (stable, temperature 1)
        float me = -1e30f, mq = -1e30f;
        for (auto& p : lexact) me = std::fmax(me, p.first);
        for (auto& p : lq4) mq = std::fmax(mq, p.first);
        double se = 0, sq = 0;
        for (size_t r = 0; r < lexact.size(); ++r) { lexact[r].first -= me; lq4[r].first -= mq; se += std::exp(lexact[r].first); sq += std::exp(lq4[r].first); }
        double kl = 0;
        for (size_t r = 0; r < lexact.size(); ++r) {
            const double pe = std::exp(lexact[r].first) / se, pq = std::exp(lq4[r].first) / sq;
            if (pe > 1e-12) kl += pe * std::log(pe / std::fmax(pq, 1e-300));
        }
        kl_sum += kl;
        std::fprintf(stderr, "x[%d]: top1 exact=%d q4=%d %s | KL %.6f\n", s, t1e, t1q, t1e == t1q ? "MATCH" : "FLIP", kl);
    }
    std::printf("HEAD REQUANT PROBE (Q6_K vs Q4_0, %d rows, %d xs): top1 flips %ld/%d, mean KL %.6f, max |dlogit| %.3f\n",
                rows, xs, flips, tested, kl_sum / xs, maxdelta);
    return 0;
}
