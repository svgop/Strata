// campaign/shadow-bench/shadow_bench.cpp - SHADOW-STORE falsifier (F-3xx, STEP 0 before any engine change).
//
// THE QUESTION: the serving pack is native IQ (IQ3_XXS / IQ3_S / IQ2_S gate/up; IQ4_NL / Q2_0 down) read in
// place from the GGUF; ~3% of routed entries miss the VRAM cache and are computed by the CPU pool from those
// blobs.  A shadow store of the same model's Q2_0 pack blobs (1,382,400 B, /home/svg/LLM/data/packs/q2_0)
// would let the pool compute misses from ~40%-smaller blobs.  The pool is bandwidth-bound, so the win is real
// only if the ACTUAL per-entry time of the Q2_0 kernels on the pack layout beats the native kernels on the
// GGUF layout by >= 25% (the GATE).  This harness times exactly the engine's CPU kernel entry points, per
// layer, over the same experts, both layouts, in one interleaved window (campaign law 1: ABAB, means only).
//
// WHAT IS TIMED (the pool's own per-layer sequence, src/kernels/cpu/pool.cpp drain modes 5/6 for the serving
// layout and modes 3/4 for the shadow layout; src/core/expert_source.cpp fills the jobs):
//   serving (native): native_quant_act -> native_gu_rows -> [act_quant_any | native_quant_h] ->
//                     [q2_rows_any | native_down_rows]   (per-job branch is the layer's d_type)
//   shadow  (Q2_0):   act_quant_q8_1 -> s2_expert_gu_rows_multi -> act_quant_q8_1(ff) ->
//                     s2_expert_down_rows_multi
// Nothing else is measured: no IO, no pool scheduling, no quantization of the router - the per-entry compute
// floor and the many-thread bandwidth ceiling only.
//
// BUILD (links the built static libs, no CMake change):
//   g++ -O3 -DNDEBUG -std=c++20 -I include campaign/shadow-bench/shadow_bench.cpp
//       -o campaign/shadow-bench/shadow_bench
//       build/libstrata_kernels_cpu.a build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a -lpthread
// RUN:
//   campaign/shadow-bench/shadow_bench --nt 1,2 --threads 1,16 --experts 64 --reps 3
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace strata::kernels::cpu;

namespace {

struct Mapped {
    const uint8_t* p = nullptr;
    size_t n = 0;
    bool open(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (::fstat(fd, &st) != 0) { ::close(fd); return false; }
        n = (size_t) st.st_size;
        void* m = ::mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) return false;
        ::madvise(m, n, MADV_WILLNEED);
        p = (const uint8_t*) m;
        return true;
    }
};

struct LayerSpec {
    int layer = 0, gu = 0, d = 0;
    long long off = 0, blob_bytes = 0, gate_off = 0, up_off = 0, down_off = 0;
    int shard = 1;
};

// One thread's scratch: what a pool worker owns (ExpertScratch / SplitBufMulti), nothing allocated in the loop.
struct Buf {
    ActQ a1[MAXT];
    ActQ a2[MAXT];
    float ff[MAXT][FF];
    float out[MAXT][H];
    alignas(64) uint8_t nact[MAXT * kNativeActBytes];
    uint8_t hq[MAXT][kNativeHBytes];
    void fill_x(unsigned seed) {
        for (int t = 0; t < MAXT; ++t) {
            unsigned s = seed + (unsigned) t * 2654435761u;
            for (int i = 0; i < H; ++i) {
                s = s * 1103515245u + 12345u;
                x[t * H + i] = ((float) (int) (s >> 8) / (float) 0x400000) - 1.0f;
            }
        }
    }
    float x[MAXT * H];
};

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The serving layout's entry, exactly the pool's drain for a native layer (pool.cpp modes 5/6, the per-job
// branches on q2_native_kernels).  `f.up_off / f.down_off` are the modular distances of the GGUF tensors.
void serve_entry(const NativeFmt& f, const uint8_t* blob, Buf& b, int nt) {
    const void* act[MAXT];
    for (int t = 0; t < nt; ++t) {
        native_quant_act(f, b.x + (size_t) t * H, b.nact + (size_t) t * kNativeActBytes);
        act[t] = b.nact + (size_t) t * kNativeActBytes;
    }
    float* ff[MAXT];
    float* out[MAXT];
    for (int t = 0; t < nt; ++t) { ff[t] = b.ff[t]; out[t] = b.out[t]; }
    native_gu_rows(f, blob, act, nt, ff, 0, FF);
    if (q2_native_kernels(f.d_type)) {
        const ActQ* a2[MAXT];
        for (int t = 0; t < nt; ++t) { act_quant_any(b.ff[t], FF, b.a2[t]); a2[t] = &b.a2[t]; }
        q2_rows_any(blob + f.down_off, f.d_row, FF / 64, a2, nt, out, 0, H);
    } else {
        const void* hq[MAXT];
        for (int t = 0; t < nt; ++t) { native_quant_h(f, b.ff[t], b.hq[t]); hq[t] = b.hq[t]; }
        native_down_rows(f, blob, hq, nt, out, 0, H);
    }
}

// The shadow layout's entry: the canonical Q2_0 pack path (pool.cpp modes 3/4, run_split_multi).
void shadow_entry(const uint8_t* blob, Buf& b, int nt) {
    const ActQ* a1[MAXT];
    for (int t = 0; t < nt; ++t) { act_quant_q8_1(b.x + (size_t) t * H, H, b.a1[t]); a1[t] = &b.a1[t]; }
    float* ff[MAXT];
    float* out[MAXT];
    for (int t = 0; t < nt; ++t) { ff[t] = b.ff[t]; out[t] = b.out[t]; }
    s2_expert_gu_rows_multi(blob, a1, nt, ff, 0, FF);
    const ActQ* a2[MAXT];
    for (int t = 0; t < nt; ++t) { act_quant_q8_1(b.ff[t], FF, b.a2[t]); a2[t] = &b.a2[t]; }
    s2_expert_down_rows_multi(blob, a2, nt, out, 0, H);
}

struct Args {
    const char* txt = "/home/svg/LLM/data/packs/iq3_s/native_experts.txt";
    const char* gguf[2] = {"/home/svg/LLM/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                           "/home/svg/LLM/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"};
    const char* shadow = "/home/svg/LLM/data/packs/q2_0/experts.bin";
    int experts = 64;
    int reps = 3;
    int nt = 1;
    int threads = 16;
};

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto val = [&](const char* name) -> const char* {
            if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) return argv[++i];
            return nullptr;
        };
        if (const char* v = val("--txt")) a.txt = v;
        else if (const char* v = val("--gguf1")) a.gguf[0] = v;
        else if (const char* v = val("--gguf2")) a.gguf[1] = v;
        else if (const char* v = val("--shadow")) a.shadow = v;
        else if (const char* v = val("--experts")) a.experts = std::atoi(v);
        else if (const char* v = val("--reps")) a.reps = std::atoi(v);
        else if (const char* v = val("--nt")) a.nt = std::atoi(v);
        else if (const char* v = val("--threads")) a.threads = std::atoi(v);
        else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (a.experts < 8 || a.experts > 512 || a.experts % 8 != 0) { std::fprintf(stderr, "--experts must be 8..512, a multiple of 8\n"); return 2; }
    if (a.nt < 1 || a.nt > MAXT) { std::fprintf(stderr, "--nt 1..%d\n", MAXT); return 2; }

    Mapped gg[2], sh;
    if (!gg[0].open(a.gguf[0])) { std::fprintf(stderr, "cannot map %s\n", a.gguf[0]); return 1; }
    if (!gg[1].open(a.gguf[1])) { std::fprintf(stderr, "cannot map %s\n", a.gguf[1]); return 1; }
    if (!sh.open(a.shadow)) { std::fprintf(stderr, "cannot map %s\n", a.shadow); return 1; }

    std::vector<LayerSpec> lay;
    {
        FILE* f = std::fopen(a.txt, "r");
        if (f == nullptr) { std::fprintf(stderr, "cannot read %s\n", a.txt); return 1; }
        char line[512];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            if (line[0] == '#' || line[0] == '\n') continue;
            LayerSpec s;
            int got = std::sscanf(line, "%d %d %d %lld %lld %lld %lld %lld %d", &s.layer, &s.gu, &s.d, &s.off,
                                  &s.blob_bytes, &s.gate_off, &s.up_off, &s.down_off, &s.shard);
            if (got < 8) continue;
            lay.push_back(s);
        }
        std::fclose(f);
    }
    if (lay.empty()) { std::fprintf(stderr, "no layers parsed\n"); return 1; }

    std::string err;
    // cpu feature gate the pool itself applies at startup
    CpuFeatures cf = cpu_features();
    if (!cf.usable()) { std::fprintf(stderr, "CPU lacks the VNNI kernel features: %s\n", cf.reason()); return 1; }

    Buf buf;
    buf.fill_x(12345u);

    std::printf("# shadow-bench falsifier: nt=%d experts=%d reps=%d threads=%d layers=%zu\n", a.nt, a.experts,
                a.reps, a.threads, lay.size());
    std::printf("# class = gu/d ggml types; us/entry is the mean over layers (compute+per-token quant), "
                "GB/s = blob_bytes / per-entry time\n");
    std::printf("layer,gu,d,ser_bytes,sh_bytes,ser_us,sh_us,ser_gbs,sh_gbs,speedup_pct\n");

    double sum_ser = 0, sum_sh = 0, sum_ser_b = 0, sum_sh_b = 0;
    const int E = a.experts;
    const int T = a.threads;
    std::vector<Buf> tbuf(T > 1 ? T : 0);
    for (int t = 0; t < (int) tbuf.size(); ++t) tbuf[(size_t) t].fill_x(12345u);
    std::vector<double> thread_us((size_t) T, 0.0);

    for (const LayerSpec& s : lay) {
        NativeFmt f;
        if (!native_fmt(s.gu, s.d, H, FF, f, err)) {
            std::printf("%d,%d,%d,SKIP,SKIP,0,0,0,0,0  # %s\n", s.layer, s.gu, s.d, err.c_str());
            continue;
        }
        // the GGUF-in-place shape (149-cold zero-copy): the blob is the gate rows; up/down at their distances
        NativeFmt zc = f;
        zc.up_off = (size_t) (s.up_off - s.gate_off);
        zc.down_off = (size_t) (s.down_off - s.gate_off);
        const Mapped& g = gg[s.shard == 2 ? 1 : 0];
        const long long EBYTES = (long long) BLOB;   // the shadow's one size
        if ((long long) sh.n < (s.layer + 1) * 512 * EBYTES) {
            std::printf("%d,%d,%d,SKIP,SHORT_SHADOW,0,0,0,0,0\n", s.layer, s.gu, s.d);
            continue;
        }

        // ---- warm both layouts over this layer's experts (page in; not timed) ----
        for (int e = 0; e < E; ++e) {
            serve_entry(zc, g.p + s.gate_off + (long long) e * s.blob_bytes, buf, a.nt);
            shadow_entry(sh.p + (size_t) s.layer * 512 * EBYTES + (size_t) e * EBYTES, buf, a.nt);
        }

        // ---- single-thread: interleaved per rep (ABAB, campaign law 1) ----
        double ser_us = 0, sh_us = 0;
        for (int rep = 0; rep < a.reps; ++rep) {
            const int base = (rep * 97) % (512 - E);
            double t0 = now_us();
            for (int e = 0; e < E; ++e) serve_entry(zc, g.p + s.gate_off + (long long) (base + e) * s.blob_bytes, buf, a.nt);
            double t1 = now_us();
            for (int e = 0; e < E; ++e)
                shadow_entry(sh.p + (size_t) s.layer * 512 * EBYTES + (size_t) (base + e) * EBYTES, buf, a.nt);
            double t2 = now_us();
            ser_us += t1 - t0;
            sh_us += t2 - t1;
        }
        ser_us /= a.reps * E;
        sh_us /= a.reps * E;

        // ---- T threads: the pool's regime (each thread its own experts, one barrier per pass) ----
        double ser_T_us = 0, sh_T_us = 0;
        if (T > 1) {
            for (int rep = 0; rep < a.reps; ++rep) {
                const int base = (rep * 97) % (512 - E);
                auto pass = [&](int arm) {
                    for (int t = 0; t < T; ++t) thread_us[(size_t) t] = 0.0;
                    std::vector<std::thread> th;
                    for (int t = 0; t < T; ++t)
                        th.emplace_back([&, t] {
                            const int e0 = E * t / T, e1 = E * (t + 1) / T;
                            double acc = 0;
                            for (int e = e0; e < e1; ++e) {
                                double u0 = now_us();
                                if (arm == 0)
                                    serve_entry(zc, g.p + s.gate_off + (long long) (base + e) * s.blob_bytes,
                                                tbuf[(size_t) t], a.nt);
                                else
                                    shadow_entry(sh.p + (size_t) s.layer * 512 * EBYTES + (size_t) (base + e) * EBYTES,
                                                 tbuf[(size_t) t], a.nt);
                                acc += now_us() - u0;
                            }
                            thread_us[(size_t) t] = acc;
                        });
                    double t0 = now_us();
                    for (auto& x : th) x.join();
                    double t1 = now_us();
                    (void) t0;
                    // per-entry wall: the pass's wall time over its entries (the pool's window wait shape)
                    (arm == 0 ? ser_T_us : sh_T_us) += (t1 - t0) / E;
                };
                pass(0);
                pass(1);
            }
            ser_T_us /= a.reps;
            sh_T_us /= a.reps;
        }

        const double ser_gbs = (double) s.blob_bytes / ser_us / 1e3;
        const double sh_gbs = (double) EBYTES / sh_us / 1e3;
        const double speedup = 100.0 * (1.0 - sh_us / ser_us);
        std::printf("%d,%d,%d,%lld,%lld,%.2f,%.2f,%.2f,%.2f,%.1f\n", s.layer, s.gu, s.d, s.blob_bytes, EBYTES,
                    ser_us, sh_us, ser_gbs, sh_gbs, speedup);
        sum_ser += ser_us;
        sum_sh += sh_us;
        sum_ser_b += s.blob_bytes;
        sum_sh_b += EBYTES;
        if (T > 1)
            std::printf("# layer %d threaded(%d): ser %.2f us/entry (%.2f GB/s)  shadow %.2f us/entry (%.2f GB/s)  %.1f%%\n",
                        s.layer, T, ser_T_us, (double) s.blob_bytes / ser_T_us / 1e3, sh_T_us,
                        (double) EBYTES / sh_T_us / 1e3, 100.0 * (1.0 - sh_T_us / ser_T_us));
        std::fflush(stdout);
    }

    const double n = (double) lay.size();
    std::printf("# MEAN over %zu layers: serving %.2f us/entry (%.2f GB/s, mean blob %.0f B)  "
                "shadow %.2f us/entry (%.2f GB/s, blob %.0f B)  speedup %.1f%%  GATE(>=25%%): %s\n",
                lay.size(), sum_ser / n, (sum_ser_b / n) / (sum_ser / n) / 1e3, sum_ser_b / n, sum_sh / n,
                (sum_sh_b / n) / (sum_sh / n) / 1e3, sum_sh_b / n, 100.0 * (1.0 - (sum_sh / n) / (sum_ser / n)),
                (sum_sh / n) <= 0.75 * (sum_ser / n) ? "PASS" : "FAIL");
    return 0;
}
