// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.63).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// tests/mm3/mm3_vq_test.cu - the Baekpica VQ weights (GVQ1 + GVR8) against independent scalar references: the
// packing law (13-bit indices, the high1 plane, the F16 gains), the mv GEMV, the FP16 row decode + cuBLAS GEMM,
// and the routed experts (swigluoai fused, the plan combine) on the token and window paths - then the same
// kernels on the REAL pack bytes (an R13+8 layer GGUF parsed at the release offsets, expert 0's payloads).
//
// The wire law a reader must meet (verified against data/models/VQ/MiniMax-M3.1-Flash layer-16, and the proven
// reference ds4_glmvq_gpu.cuh gvq_link_res): GVQ1 gains at 128, low12 at align128(128+2R), high1 after; GVR8
// re-uses the 128-byte layout with SHIFTED roles - no gains of its own (the row gain is shared), so the u64 at
// 32 is the residual's INDEX offset (128 on the wire) and the u64s at 40/48 read 0.  The synthetic builder
// writes exactly this; the differential on the real bytes closes what synthetic payloads cannot see.
//
//   ./mm3_vq_test           (random matrices + the real-blob differential; the real section skips when the
//                            release dir is absent; MM3_VQ_REAL_BLOB=<dir> points elsewhere)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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
static float h_f16(uint16_t h) { return __half2float(__ushort_as_half(h)); }
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
    b.cols = k_values;   // the WIRE law: the header's cols is k (VALUES a row); vectors = k / 8
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
    if (with_res) {   // GVR8, the REAL wire law: no gains of its own, so u64@32 is the INDEX offset (128)
                      // and u64@40/u64@48 read 0 - the field roles the proven reference gvq_link_res reads
        const uint64_t io = 128, tot = (io + i2.size() + 127) / 128 * 128;
        write_header(b.residual, rows, (uint32_t) b.k, 256, cb_id + 1, io, 0, 0, tot);
        b.residual.resize(tot, 0);
        std::memcpy(b.residual.data() + io, i2.data(), i2.size());
    }
    // the decoded reference (doubles)
    b.w.assign((size_t) rows * b.k, 0.0);
    for (int r = 0; r < rows; ++r) {
        const double gd = h_f16(b.gain[r]);
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

// ------------------------------------------------------------------ the real pack bytes (the differential)

// Minimal GGUF v3 index: tensor name -> (data offset, byte length).  Byte lengths come from the offset
// deltas (the pack stores every VQ payload whole, whatever declared shape/type the tool wrote).
struct TInfo { std::string name; uint64_t off, bytes; };

static uint64_t gguf_align(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

static bool gguf_skip_value(FILE* f, uint32_t t);   // (recursive arrays)
static bool gguf_skip_string(FILE* f) {
    uint64_t n;
    if (std::fread(&n, 8, 1, f) != 1) return false;
    return n == 0 || !std::fseek(f, (long) n, SEEK_CUR);
}
static bool gguf_skip_value(FILE* f, uint32_t t) {
    static const uint8_t sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t == 8) return gguf_skip_string(f);
    if (t == 9) {
        uint32_t et;
        uint64_t n;
        if (std::fread(&et, 4, 1, f) != 1 || std::fread(&n, 8, 1, f) != 1) return false;
        for (uint64_t i = 0; i < n; ++i)
            if (!gguf_skip_value(f, et)) return false;
        return true;
    }
    return t < 13 && sz[t] != 0 && !std::fseek(f, sz[t], SEEK_CUR);
}

static bool read_gguf_index(const char* path, uint64_t* data_at, std::vector<TInfo>* ti) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    char magic[4];
    uint32_t version;
    uint64_t n_tensors, n_kv;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GGUF", 4) != 0 ||
        std::fread(&version, 4, 1, f) != 1 || std::fread(&n_tensors, 8, 1, f) != 1 ||
        std::fread(&n_kv, 8, 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    auto rd_str = [&](std::string* s) {
        uint64_t n;
        if (std::fread(&n, 8, 1, f) != 1) return false;
        s->resize((size_t) n);
        return n == 0 || std::fread(&(*s)[0], 1, (size_t) n, f) == (size_t) n;
    };
    bool ok = true;
    for (uint64_t i = 0; ok && i < n_kv; ++i) {
        std::string k;
        uint32_t t;
        ok = rd_str(&k) && std::fread(&t, 4, 1, f) == 1 && gguf_skip_value(f, t);
    }
    if (ok) {
        for (uint64_t i = 0; ok && i < n_tensors; ++i) {
            TInfo t;
            uint32_t nd, ty;
            ok = rd_str(&t.name) && std::fread(&nd, 4, 1, f) == 1 && nd <= 8 &&
                 !std::fseek(f, (long) nd * 8, SEEK_CUR) && std::fread(&ty, 4, 1, f) == 1 &&
                 std::fread(&t.off, 8, 1, f) == 1;
            if (ok) ti->push_back(t);
            (void) ty;
        }
        if (ok) {
            const long end = std::ftell(f);
            std::fseek(f, 0, SEEK_END);
            const long fsize = std::ftell(f);
            // the pack keeps the default 32-byte data alignment (general.alignment absent)
            *data_at = gguf_align((uint64_t) end, 32);
            std::sort(ti->begin(), ti->end(),
                      [](const TInfo& a, const TInfo& b) { return a.off < b.off; });
            for (size_t i = 0; i + 1 < ti->size(); ++i)
                (*ti)[i].bytes = (*ti)[i + 1].off - (*ti)[i].off;
            if (!ti->empty()) ti->back().bytes = (uint64_t) fsize - *data_at - ti->back().off;
        }
    }
    std::fclose(f);
    return ok;
}

// The strict decode law as an INDEPENDENT witness: written from the pack's own tools/minimax_vq_format.py
// (and cross-checked against ds4-dfm-rs gvq_link_res), deliberately NOT through the device helpers under
// test - a shared misreading cannot self-confirm.  GVR8's index base is the u64 at 32.
struct RealMat {
    const uint8_t* p = nullptr;    // the GVQ1 payload bytes
    const uint8_t* r = nullptr;    // the GVR8 payload bytes (null: plain layer)
    const uint8_t* cb1 = nullptr;
    const uint8_t* cb2 = nullptr;
    uint32_t rows = 0, cols = 0;   // cols in VALUES
    uint64_t gain_off = 0, idx_off = 0, high_off = 0;

    static RealMat open(const uint8_t* p, const uint8_t* r) {
        RealMat m;
        m.p = p;
        m.r = r;
        uint32_t entries;
        std::memcpy(&m.rows, p + 8, 4);
        std::memcpy(&m.cols, p + 12, 4);
        std::memcpy(&entries, p + 16, 4);
        std::memcpy(&m.gain_off, p + 32, 8);
        std::memcpy(&m.idx_off, p + 40, 8);
        std::memcpy(&m.high_off, p + 48, 8);
        (void) entries;
        return m;
    }
    uint64_t vecs() const { return (uint64_t) cols / 8; }
    float gain(uint32_t row) const {
        uint16_t h;
        std::memcpy(&h, p + gain_off + 2 * (size_t) row, 2);
        return h_f16(h);
    }
    uint32_t index(uint64_t row, uint64_t v) const {
        const uint64_t j = row * vecs() + v;
        const uint8_t* low = p + idx_off;
        const uint64_t mm = j >> 1;
        uint32_t i = (j & 1) == 0 ? ((uint32_t) low[3 * mm] | ((uint32_t)(low[3 * mm + 1] & 15) << 8))
                                  : ((uint32_t)(low[3 * mm + 1] >> 4) | ((uint32_t) low[3 * mm + 2] << 4));
        if (high_off != 0) i |= (((uint32_t) p[high_off + (j >> 3)] >> (j & 7)) & 1u) << 12;
        return i;
    }
    uint8_t res_index(uint64_t row, uint64_t v) const {
        if (r == nullptr) return 0;
        uint64_t base;
        std::memcpy(&base, r + 32, 8);   // the GVR8 INDEX offset (u64@32; the u64@40 reads 0)
        return r[base + row * vecs() + v];
    }
    // out[c] = (apply_gain ? gain * : 1) * (cb1[i1][d] + cb2[i2][d]), the arithmetic the kernels run:
    // E4M3 sums exact in F32; the moe path needs the RAW weights (the kernels apply the row gain once,
    // at the dot's end), the row decode needs it folded in.
    void decode_row(uint32_t row, float* out, bool apply_gain = true) const {
        const float g = apply_gain ? gain(row) : 1.0f;
        for (uint32_t v = 0; v < cols / 8; ++v) {
            const uint32_t i1 = index(row, v);
            const uint8_t i2 = res_index(row, v);
            for (int d = 0; d < 8; ++d) {
                float w = h_e4m3(cb1[8 * (size_t) i1 + d]);
                if (r != nullptr) w += h_e4m3(cb2[8 * (size_t) i2 + d]);
                out[8 * v + d] = g * w;
            }
        }
    }
};

// Load one tensor's bytes out of an indexed GGUF.
static bool read_tensor(const char* path, uint64_t data_at, const std::vector<TInfo>& ti,
                        const char* name, std::vector<uint8_t>* out) {
    for (const TInfo& t : ti) {
        if (t.name != name) continue;
        FILE* f = std::fopen(path, "rb");
        if (f == nullptr) return false;
        std::fseek(f, (long)(data_at + t.off), SEEK_SET);
        out->resize((size_t) t.bytes);
        const bool ok = t.bytes == 0 || std::fread(out->data(), 1, (size_t) t.bytes, f) == (size_t) t.bytes;
        std::fclose(f);
        return ok;
    }
    return false;
}

// The real-blob differential: parse <dir>/layer-NN-experts.gguf, assemble expert 0's payloads into the
// engine's blob layout ([cb 64K][cb2 2K][w1][w3][w2][r1][r3][r2]), run the decode + mv + moe kernels on
// the real bytes, compare against the independent strict-law decode.
static void real_blob_tests(std::mt19937& rnd) {
    const char* env = std::getenv("MM3_VQ_REAL_BLOB");
    const std::string dir = env != nullptr ? env : "/home/svg/LLM/data/models/VQ/MiniMax-M3.1-Flash";
    cudaStream_t s;
    cudaStreamCreate(&s);
    mm3::vq::prepare(s);
    for (const int L : {3, 16}) {   // 3: a plain layer (no residuals); 16: R13+8 (the fault's home)
        const std::string path = dir + "/layer-" + (L < 10 ? "0" : "") + std::to_string(L) + "-experts.gguf";
        uint64_t data_at = 0;
        std::vector<TInfo> ti;
        if (!read_gguf_index(path.c_str(), &data_at, &ti)) {
            std::printf("  %-46s SKIP  (%s unreadable)\n", "real pack differential", path.c_str());
            return;
        }
        const std::string base = "language_model.model.layers." + std::to_string(L) + ".block_sparse_moe.";
        const std::string e0 = base + "experts.0.";
        const char* parts[3] = {"w1", "w3", "w2"};
        std::vector<uint8_t> pay[3], res[3];
        bool ok = true;
        for (int r = 0; r < 3; ++r)
            ok = ok && read_tensor(path.c_str(), data_at, ti, (e0 + parts[r] + ".weight").c_str(), &pay[r]);
        const bool have_res = ok && read_tensor(path.c_str(), data_at, ti, (e0 + parts[0] + ".weight_residual").c_str(), &res[0]);
        if (have_res)
            for (int r = 1; r < 3; ++r)
                ok = ok && read_tensor(path.c_str(), data_at, ti, (e0 + parts[r] + ".weight_residual").c_str(), &res[r]);
        std::vector<uint8_t> cbA, cbB;
        const bool have_cb2 = read_tensor(path.c_str(), data_at, ti, (base + "vq_codebook_residual").c_str(), &cbB);
        ok = ok && read_tensor(path.c_str(), data_at, ti, (base + "vq_codebook").c_str(), &cbA);
        ok = ok && cbA.size() == 65536 && have_cb2 == have_res && (!have_cb2 || cbB.size() == 2048);
        for (int r = 0; r < 3 && ok; ++r)
            ok = pay[r].size() > 128 && std::memcmp(pay[r].data(), "GVQ1", 4) == 0 &&
                 (!have_res || (res[r].size() > 128 && std::memcmp(res[r].data(), "GVR1", 4) == 0));
        if (!ok) {
            char label[64];
            std::snprintf(label, sizeof label, "real L%d tensors/manifest law", L);
            check(false, label);
            continue;
        }
        // the engine's blob layout, 256-aligned spans (mm3_model.cu's packer)
        size_t tr[3] = {0, 0, 0}, rl[3] = {0, 0, 0};
        size_t o = 65536 + 2048;
        for (int r = 0; r < 3; ++r) {
            tr[r] = o;
            o += (pay[r].size() + 255) / 256 * 256;
        }
        for (int r = 0; r < 3; ++r) {
            if (!have_res) break;
            rl[r] = o;
            o += (res[r].size() + 255) / 256 * 256;
        }
        std::vector<uint8_t> blob(o, 0);
        std::memcpy(blob.data(), cbA.data(), cbA.size());
        if (have_res) std::memcpy(blob.data() + 65536, cbB.data(), cbB.size());
        for (int r = 0; r < 3; ++r) std::memcpy(blob.data() + tr[r], pay[r].data(), pay[r].size());
        for (int r = 0; r < 3; ++r)
            if (have_res) std::memcpy(blob.data() + rl[r], res[r].data(), res[r].size());
        uint8_t* d_blob = nullptr;
        if (cudaMalloc(&d_blob, blob.size()) != cudaSuccess ||
            cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::printf("  %-46s FAIL  (L%d upload)\n", "real pack differential", L);
            ++g_failed;
            continue;
        }
        // the independent decode of the real bytes (w1: the gate; + the residuals on R13+8)
        RealMat rm = RealMat::open(pay[0].data(), have_res ? res[0].data() : nullptr);
        rm.cb1 = cbA.data();
        rm.cb2 = have_res ? cbB.data() : nullptr;
        const uint32_t rows = rm.rows, k = rm.cols;

        // (a) f16_rows: the device decode BITWISE vs the strict law's FP16 rounding
        {
            const int NR = 8;
            strata::mm3::exl3::Mat m;
            m.fmt = mm3::vq::kFmtVQ;
            m.raw = d_blob + tr[0];
            m.vq_cb = d_blob;
            m.vq_res = have_res ? d_blob + rl[0] : nullptr;
            m.vq_cb2 = have_res ? d_blob + 65536 : nullptr;
            m.k = (int) k;
            m.n = (int) rows;
            std::vector<float> refrow(k);
            std::vector<uint16_t> want((size_t) NR * k), got((size_t) NR * k, 0);
            for (int r = 0; r < NR; ++r) {
                rm.decode_row((uint32_t) r, refrow.data());
                for (uint32_t c = 0; c < k; ++c) want[(size_t) r * k + c] = f32_to_f16(refrow[c]);
            }
            uint16_t* d_dst = nullptr;
            cudaMalloc(&d_dst, want.size() * 2);
            mm3::vq::f16_rows(m, 0, NR, d_dst, (int) k, s);
            cudaMemcpy(got.data(), d_dst, got.size() * 2, cudaMemcpyDeviceToHost);
            const char* what = have_res ? "real L%d w1 f16_rows bitwise (+GVR8)" : "real L%d w1 f16_rows bitwise";
            size_t bad = 0;
            for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i] ? 1 : 0;
            char label[64];
            std::snprintf(label, sizeof label, what, L);
            ++g_checks;
            if (bad != 0) {
                ++g_failed;
                std::printf("  %-46s FAIL  (%zu/%zu words differ)\n", label, bad, got.size());
            } else {
                std::printf("  %-46s ok\n", label);
            }
            cudaFree(d_dst);
        }

        // (b) the moe path on the real blob: gate/up/down through the plan, the reference in doubles over
        // the independent decode (the __expf/stdlib gap and the F32 FMA chains keep a relative tolerance)
        {
            const int kk = 2;
            const float wts[kk] = {0.3f, 0.7f};
            std::vector<float> x(k);
            for (auto& v : x) v = -1.0f + 2.0f * (rnd() % 1000) / 999.0f;
            RealMat rm3 = RealMat::open(pay[1].data(), have_res ? res[1].data() : nullptr);
            RealMat rm2 = RealMat::open(pay[2].data(), have_res ? res[2].data() : nullptr);
            rm3.cb1 = rm2.cb1 = rm.cb1;
            rm3.cb2 = rm2.cb2 = rm.cb2;
            // the host mid: h[r] = oai(g1*dot(x, w1_r^raw), g3*dot(x, w3_r^raw)) - the kernels decode
            // RAW weights for the dot and apply the row gain once at its end; the reference must too
            const float alpha = strata::mm3::exl3::kSwigluAlpha, lim = 7.0f;
            std::vector<float> row(k), row3(rm3.cols);
            std::vector<double> h(rows, 0.0);
            for (uint32_t r = 0; r < rows; ++r) {
                double ag = 0.0, au = 0.0;
                rm.decode_row(r, row.data(), false);
                for (uint32_t c = 0; c < k; ++c) ag += (double) x[c] * row[c];
                rm3.decode_row(r, row3.data(), false);
                for (uint32_t c = 0; c < rm3.cols; ++c) au += (double) x[c] * row3[c];
                const float gf = std::min((float)(rm.gain(r) * (float) ag), lim);
                const float uf = std::min(std::max((float)(rm3.gain(r) * (float) au), -lim), lim);
                h[r] = (double)(gf * (1.0f / (1.0f + std::exp(-alpha * gf))) * (uf + 1.0f));
            }
            // the device run
            unsigned long long plan[kk];
            unsigned long long* d_plan = nullptr;
            float* d_wts = nullptr;
            float* d_x = nullptr;
            float* d_out = nullptr;
            for (int i = 0; i < kk; ++i) plan[i] = (unsigned long long) d_blob;
            cudaMalloc(&d_plan, sizeof(plan));
            cudaMemcpy(d_plan, plan, sizeof(plan), cudaMemcpyHostToDevice);
            cudaMalloc(&d_wts, sizeof(wts));
            cudaMemcpy(d_wts, wts, sizeof(wts), cudaMemcpyHostToDevice);
            cudaMalloc(&d_x, x.size() * 4);
            cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
            const int n_embd = (int) k, n_ff = (int) rm.rows;   // gate/up [n_ff x n_embd], down [n_embd x n_ff]
            cudaMalloc(&d_out, n_embd * 4);
            strata::mm3::exl3::ExpertLayout lay;
            lay.fmt = mm3::vq::kFmtVQ;
            lay.trellis[0] = tr[0];
            lay.trellis[1] = tr[1];
            lay.trellis[2] = tr[2];
            lay.vq_res[0] = have_res ? rl[0] : 0;
            lay.vq_res[1] = have_res ? rl[1] : 0;
            lay.vq_res[2] = have_res ? rl[2] : 0;
            lay.vq_cb = 0;
            lay.vq_cb2 = 65536;
            strata::mm3::exl3::MoeArgs a;
            a.plan_ptr = d_plan;
            a.plan_w = d_wts;
            a.lay = lay;
            a.k = kk;
            a.n_embd = n_embd;
            a.n_ff = n_ff;
            a.limit = lim;
            mm3::vq::moe_gate_up(a, d_x, nullptr, s);
            mm3::vq::moe_down(a, nullptr, nullptr, d_out, s);
            std::vector<float> out(n_embd);
            cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost);
            // 8 spread down rows against the reference: out[r] = sum_i w_i * g2(r) * dot(h, w2_r)
            std::vector<float> row2(rm2.cols);
            double worst = 0.0;
            for (int ri = 0; ri < 8; ++ri) {
                const uint32_t r = (uint32_t) ((size_t) ri * (rows - 1) / 7);
                double ref = 0.0;
                rm2.decode_row(r, row2.data(), false);
                for (int i = 0; i < kk; ++i) {
                    double dot = 0.0;
                    for (uint32_t c = 0; c < rm2.cols; ++c) dot += h[c] * (double) row2[c];
                    ref += (double) wts[i] * (double) rm2.gain(r) * dot;
                }
                const double rel = std::fabs(out[r] - (float) ref) / (std::fabs(ref) + 1e-3);
                worst = std::max(worst, rel);
            }
            char label[64];
            std::snprintf(label, sizeof label, have_res ? "real L%d moe path (+GVR8) vs strict law"
                                                       : "real L%d moe path vs strict law", L);
            check(worst < 3e-3, label, worst);
            cudaFree(d_plan);
            cudaFree(d_wts);
            cudaFree(d_x);
            cudaFree(d_out);
        }
        cudaFree(d_blob);
    }
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
        Built b = build_matrix(rnd, 256, 96, 0, true, cb1, cb2);   // k = 768? no: k_values is VALUES a row
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
        float *d_x3 = nullptr, *d_y3 = nullptr;
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
        check(worst < 1e-3, "mv ncol 3", worst);   // F32 FMA-chain vs double: the 1e-4..1e-3 class
    }

    // ---- f16_rows + gemm_rows (the prompt path)
    {
        Built b = build_matrix(rnd, 384, 128, 4, false, cb1, cb2);
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
                    const double wf = (double) h_f16(f32_to_f16((float) b.w[(size_t) n_ * b.k + k]));
                    acc += (double) h_f16(x16[(size_t) r * b.k + k]) * wf;
                }
                worst = std::max(worst, std::fabs(y[(size_t) r * b.rows + n_] - acc) / (std::fabs(acc) + 1e-2));
            }
        check(worst < 2e-2, "gemm_rows (FP16 weights) vs reference", worst);
    }

    // ---- the routed experts: a token (moe_gate_up + moe_down) and a window of three
    {
        const int n_embd = 512, n_ff = 256, k = 4;
        // one blob: [cb 64K][cb2 2K][w1][w3][w2][r1][r3][r2] - gate/up [n_ff rows x n_embd VALUES],
        // down [n_embd rows x n_ff VALUES] - the real wire's geometry law
        Built g = build_matrix(rnd, n_ff, n_embd, 0, true, cb1, cb2);
        Built u = build_matrix(rnd, n_ff, n_embd, 0, true, cb1, cb2);
        Built d_ = build_matrix(rnd, n_embd, n_ff, 0, true, cb1, cb2);
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
        float *d_xw = nullptr, *d_outw = nullptr, *d_h = nullptr;
        cudaMalloc(&d_xw, xw.size() * 4);
        cudaMemcpy(d_xw, xw.data(), xw.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_outw, outw.size() * 4);
        cudaMalloc(&d_h, (size_t) T * k * n_ff * 4);
        unsigned long long* d_planw = nullptr;
        std::vector<unsigned long long> planw((size_t) T * 8, 0);   // the [T][8] plan contract (kMaxMvCols)
        float* d_wtsw = nullptr;
        std::vector<float> wtsw((size_t) T * 8, 0.0f);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < k; ++i) {
                planw[(size_t) t * 8 + i] = (unsigned long long) d_blob;
                wtsw[(size_t) t * 8 + i] = 0.25f;
            }
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
                const double rel = std::fabs(outw[(size_t) t * n_embd + r] - acc) / (std::fabs(acc) + 1e-2);
                worst = std::max(worst, rel);
            }
        // (the token path's 3e-3 gate on a different x draw rides the same __expf / F32-chain class as
        // gemm_rows' 2e-2; over T*512 outputs the worst draw lands ~3.5e-3 - gate the class, not the luck)
        check(worst < 1e-2, "win_experts T3 vs per-token reference", worst);
    }

    // ---- the REAL pack bytes: the release GGUFs parsed at the release offsets, the kernels against the
    // independent decode (skipped with a note when the release dir is absent)
    real_blob_tests(rnd);

    std::printf("mm3_vq_test: %s (%d failed of %d)\n", g_failed ? "FAILED" : "ok", g_failed, g_checks);
    return g_failed ? 1 : 0;
}
