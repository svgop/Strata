// src/kernels/cpu/pool_bench.cpp - F-458: what limits the CPU expert pool's split kernel at the concurrent lane's shape.
//
//   pool_bench [units=30] [name]     CPU only, synthetic experts, no model and no GPU.  `name` keeps only the formats
//                                    whose name contains it (a smoke run).  POOL_BENCH_FIXTURES=<dir> keeps the experts
//                                    there, so a measured run starts on an idle box and a repeat costs seconds.
//   --prefetch                       F-460's experiment (section F-460 below) instead of tables 1-4; the default run
//                                    does not change.
//
// Timed: ExpertPool::run_split_multi_native (a native pack's layer) and ExpertPool::run_split_multi (the shadow Q2_0
// store), the engine's own calls (src/core/expert_source.cpp, expert_pool_dispatch_multi), over N experts of T rows each.
// Reported: us per expert-call, the median of `units` timed units after warm-up, with the pool's own phase split
// (gate/up rows, the host-serial quantization, down rows) and the aggregate weight-byte rate.
//
// THE FIXTURE mirrors the standing pack (config/custom-iq3_s-8081.json -> packs/iq3_s, Qwen3.8-Flash-Next: 48 layers,
// 512 experts, n_embd 2560 x n_ff 640): its seven native layer types with their layer counts (native_experts.txt v3)
// and the shadow store's canonical blob.  The weights are synthetic in each type's own encoding (the reference
// quantizers over uniform-random floats; the shadow blob as expert_multi_test builds it): the numbers' meaning does
// not matter, only the bytes decoded and the arithmetic, and a well-formed encoding keeps NaNs and denormals out of
// the timing.
//
// WHAT SEPARATES THE HYPOTHESES (15 workers + the draining host thread unless a table says otherwise):
//   cold    the unit's experts are the next N of a 320-expert store (442 MB at least, far over every cache this CPU
//           has): every weight comes from DRAM.
//   hot     the same N experts every unit.  The 16 threads span two CCDs whose L3 differ (a 9950X3D: 96 MiB on the
//           CCD of CPUs 0-7, 32 MiB on the other) and every thread reads every expert's rows, so a set is resident
//           only while it fits the SMALLER one: N = 8 (16 MB) is; N = 16 and 32 exceed it and are partly cold.
//   T       rows per expert.  An ExpertJobMulti carries MAXT = 8 rows, the engine refuses a larger window
//           (expert_source.cpp, dispatch_begin), and the row kernels are written for 8 (iq_avx512.cpp gu_rows_nt;
//           iq_avx2.cpp iq4nl256_down_rows_v fills a y[8]), so T > 8 is TWO serial calls with the rows split evenly
//           (12 = 6 + 6): every weight is decoded twice, and dispatch and barriers are paid twice - as for one expert
//           seen by both cohorts.
//   gap     the host's idle time before every call.  The standing recipe parks its workers after 100 us
//           (STRATA_POOL_SPIN_US, launch-custom-iq3s-8081.sh) and the engine's host waits on the GPU between layers, so
//           there a call wakes sleepers - inside the call, where the engine's telemetry times it.
//   threads 4, 8, 15 pool workers (each + the host), and one thread calling the row kernels directly: the per-core rate
//           the pool is measured against.
//   stream  16 threads, pinned as the pool pins its threads, read 1 GiB they have not touched: the box's streaming
//           read rate, the bandwidth ceiling.
//   --prefetch   F-460: before each timed call a pass reads every byte of THAT call's experts, the host idles for a gap,
//           one thread reads X MiB of unrelated memory in the gap, and the call is timed against a cold control that
//           carries the same gap and traffic behind a sham pass.  Who reads: the host thread, one thread beside the
//           host, 16 threads beside the pool's, or one thread per CCD.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kH = cpu::H, kFF = cpu::FF, kMaxT = cpu::MAXT;
constexpr int kRows = 16;      // rows per expert the buffers hold (the largest T)
constexpr int kStore = 320;    // cold experts per format: 320 x 1,382,400 B = 442 MB at least
constexpr int kThreads = 16;   // the pool's threads on this lane's CPU: 15 workers + the host
constexpr int kWarm = 5;       // untimed units before a point's timed ones
constexpr int kTs[] = {1, 4, 6, 8, 12, 16};
constexpr int kNs[] = {1, 4, 8, 16, 32};
constexpr int kTCount = (int) (sizeof kTs / sizeof kTs[0]);
constexpr int kNCount = (int) (sizeof kNs / sizeof kNs[0]);
constexpr int kT12 = 4;        // kTs[4] = 12; kTs[0..3] are the one-call points
constexpr int kN8 = 2;         // kNs[2] = 8

struct Format {
    const char* name;
    int gu, d;      // ggml types of gate/up and of down; gu < 0: the shadow Q2_0 store (the canonical blob)
    int layers;     // layers of the pack in this form
};
constexpr Format kFormats[] = {
    {"IQ2_S/IQ4_NL", GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, 15},
    {"IQ3_XXS/IQ4_NL", GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, 14},
    {"IQ3_S/IQ4_NL", GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, 9},
    {"IQ2_S/Q2_0", GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, 5},
    {"IQ3_XXS/Q2_0", GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0, 3},
    {"IQ4_XS/IQ4_NL", GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, 1},
    {"IQ3_S/Q2_0", GGML_TYPE_IQ3_S, GGML_TYPE_Q2_0, 1},
    {"shadow-Q2_0", -1, -1, 0},
};
constexpr int kFormatCount = (int) (sizeof kFormats / sizeof kFormats[0]);
constexpr int kPackLayers = 15 + 14 + 9 + 5 + 3 + 1 + 1;
static_assert(kPackLayers == 48, "the layer counts are the standing pack's 48");

void pause_cpu() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

void set_env(const char* key, const char* value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    ::setenv(key, value, 1);
#endif
}

bool env_on(const char* key) { const char* const v = std::getenv(key); return v != nullptr && std::atoi(v) != 0; }

uint64_t rng(uint64_t& s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1Dull;
}
/// Uniform in [-1, 1): a 24-bit mantissa.
float uniform(uint64_t& s) { return (float) (rng(s) >> 40) * (1.0f / 8388608.0f) - 1.0f; }

double us_of(Clock::duration d) { return std::chrono::duration<double, std::micro>(d).count(); }

double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

std::string cpu_model() {
#if defined(_WIN32)
    return "unknown";
#else
    std::FILE* const f = std::fopen("/proc/cpuinfo", "r");
    if (f == nullptr) return "unknown";
    char line[512];
    std::string model = "unknown";
    while (std::fgets(line, sizeof line, f) != nullptr) {
        if (std::strncmp(line, "model name", 10) != 0) continue;
        const char* const colon = std::strchr(line, ':');
        model = colon != nullptr ? colon + 2 : line;
        while (!model.empty() && (model.back() == '\n' || model.back() == '\r')) model.pop_back();
        break;
    }
    std::fclose(f);
    return model;
#endif
}

/// The kernel's 1-minute load average (/proc/loadavg); -1 where there is none.
double loadavg() {
#if defined(_WIN32)
    return -1.0;
#else
    std::FILE* const f = std::fopen("/proc/loadavg", "r");
    if (f == nullptr) return -1.0;
    double v = -1.0;
    if (std::fscanf(f, "%lf", &v) != 1) v = -1.0;
    std::fclose(f);
    return v;
#endif
}

/// POOL_BENCH_FIXTURES: one file of experts per format under its directory.
std::string fixture_path(const char* dir, const Format& fmt) {
    std::string name(fmt.name);
    for (char& c : name)
        if (c == '/') c = '_';
    return std::string(dir) + "/" + name + ".bin";
}

std::string t_label(int ti) {
    std::string s = std::to_string(kTs[ti]);
    if (kTs[ti] > kMaxT) s += " (" + std::to_string(kTs[ti] / 2) + "+" + std::to_string(kTs[ti] / 2) + ", two calls)";
    return s;
}

// ---------------------------------------------------------------------------------------------------------------
// The store: 320 synthetic experts of one layer form, and 16 quantized activations.

struct Store {
    const Format* fmt = nullptr;
    cpu::NativeFmt nf;             // native: the layer's geometry (the shadow store leaves it default)
    size_t bytes = 0;              // per expert
    std::vector<uint8_t> mem;      // kStore experts
    std::vector<uint8_t> nact;     // kRows native activations, kNativeActBytes each
    std::vector<cpu::ActQ> qact;   // kRows activations in the pool's planar form
    const uint8_t* expert(int e) const { return mem.data() + (size_t) e * bytes; }
};

// expert_multi_test's blob: the codes at random, the scales fp16 in 0.0039-0.0156.
void make_shadow_blob(uint8_t* b, uint64_t& s) {
    for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng(s);
    for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
        const uint16_t h = (uint16_t) (0x1C00 + rng(s) % 0x0800);
        std::memcpy(b + i, &h, 2);
    }
}

// Experts [e0, e1): pool_tasks_test's fixture - each projection is its type's reference quantizer over random floats.
void make_experts(Store& st, int e0, int e1, uint64_t seed) {
    std::vector<float> w((size_t) kH * kFF);
    for (int e = e0; e < e1; ++e) {
        uint8_t* const blob = st.mem.data() + (size_t) e * st.bytes;
        if (st.fmt->gu < 0) {
            make_shadow_blob(blob, seed);
            continue;
        }
        for (float& v : w) v = uniform(seed);
        ggml_quantize_chunk((ggml_type) st.fmt->gu, w.data(), blob, 0, kFF, kH, nullptr);
        ggml_quantize_chunk((ggml_type) st.fmt->gu, w.data(), blob + st.nf.up_off, 0, kFF, kH, nullptr);
        ggml_quantize_chunk((ggml_type) st.fmt->d, w.data(), blob + st.nf.down_off, 0, kH, kFF, nullptr);
    }
}

void build_store(Store& st, const Format& fmt, const char* keep) {
    st.fmt = &fmt;
    std::string err;
    if (fmt.gu >= 0) {
        if (!cpu::native_fmt(fmt.gu, fmt.d, kH, kFF, st.nf, err)) {
            std::fprintf(stderr, "pool_bench: %s: %s\n", fmt.name, err.c_str());
            std::exit(1);
        }
        // the direct path below mirrors the pool's branches for this pack's forms only
        if (cpu::q2_native_kernels(st.nf.gu_type) || st.nf.tail_off != 0 || st.nf.had_block != 0) {
            std::fprintf(stderr, "pool_bench: %s: a form this bench does not mirror (Q2_0 gate/up, NVFP4 tail, Hadamard)\n",
                         fmt.name);
            std::exit(1);
        }
        st.bytes = st.nf.bytes;
    } else {
        st.bytes = cpu::BLOB;
    }
    // POOL_BENCH_FIXTURES=<dir>: every store is kept there, one file per format.  The generation is 16 threads for
    // ~90 s; a measured run that loads its experts starts on an idle box instead of under its own generation's load,
    // and a repeat costs seconds.
    const std::string path = keep != nullptr && keep[0] != '\0' ? fixture_path(keep, fmt) : std::string();
    const double t0 = now_ms();
    st.mem.assign((size_t) kStore * st.bytes, 0);
    bool loaded = false;
    if (!path.empty()) {
        if (std::FILE* const f = std::fopen(path.c_str(), "rb")) {
            loaded = std::fread(st.mem.data(), 1, st.mem.size(), f) == st.mem.size();
            std::fclose(f);
        }
    }
    if (!loaded) {
        const int threads = (std::max)(1, (std::min)(16, (int) std::thread::hardware_concurrency()));
        const int per = (kStore + threads - 1) / threads;
        std::vector<std::thread> gen;
        for (int t = 1; t < threads; ++t) {
            gen.emplace_back([&st, t, per] {
                make_experts(st, t * per, (std::min)(kStore, (t + 1) * per),
                             0x9E3779B97F4A7C15ull ^ (uint64_t) t * 0xD1B54A32D192ED03ull);
            });
        }
        make_experts(st, 0, (std::min)(kStore, per), 0x9E3779B97F4A7C15ull);
        for (std::thread& th : gen) th.join();
        if (!path.empty()) {
            if (std::FILE* const f = std::fopen(path.c_str(), "wb")) {
                const bool kept = std::fwrite(st.mem.data(), 1, st.mem.size(), f) == st.mem.size();
                std::fclose(f);
                if (!kept) std::fprintf(stderr, "pool_bench: could not keep %s\n", path.c_str());
            } else {
                std::fprintf(stderr, "pool_bench: could not open %s to keep the experts\n", path.c_str());
            }
        }
    }
    st.nact.assign((size_t) kRows * cpu::kNativeActBytes, 0);
    st.qact.assign((size_t) kRows, cpu::ActQ{});
    std::vector<float> x((size_t) kH);
    uint64_t seed = 0xA24BAED4963EE407ull;
    for (int r = 0; r < kRows; ++r) {
        for (float& v : x) v = uniform(seed);
        if (fmt.gu >= 0) cpu::native_quant_act(st.nf, x.data(), st.nact.data() + (size_t) r * cpu::kNativeActBytes);
        cpu::act_quant_any(x.data(), kH, st.qact[(size_t) r]);
    }
    std::printf("[store] %-14s blob %8zu B x %d = %5zu MB, %s in %.0f ms%s\n", fmt.name, st.bytes, kStore,
                ((size_t) st.bytes * kStore) >> 20, loaded ? "loaded" : "built", now_ms() - t0,
                fmt.gu >= 0 ? "" : " (the shadow store's canonical blob)");
}

// ---------------------------------------------------------------------------------------------------------------
// One unit: N experts x T rows, as `calls` serial calls of `per` rows (two when T > MAXT: 12 = 6 + 6, 16 = 8 + 8).

struct Pt {         // a unit, in us
    double us = 0, gu = 0, q = 0, down = 0;
    void add(const Pt& o) { us += o.us; gu += o.gu; q += o.q; down += o.down; }
    void scale(double k) { us *= k; gu *= k; q *= k; down *= k; }
};
struct Series {
    Pt mean, median;         // phases: means over the units; us: the mean's and the median unit's
    double p25 = 0, p75 = 0;
};

template <class Run>
Series measure(int units, Run&& run) {
    for (int i = 0; i < kWarm; ++i) run();               // the workers start, the rows are touched
    std::vector<Pt> pts((size_t) units);
    std::vector<double> us((size_t) units);
    for (int i = 0; i < units; ++i) {
        pts[(size_t) i] = run();
        us[(size_t) i] = pts[(size_t) i].us;
    }
    Series s;
    for (const Pt& p : pts) s.mean.add(p);
    s.mean.scale(1.0 / units);
    std::sort(us.begin(), us.end());
    const auto quantile = [&us](double f) {
        const double pos = f * (double) (us.size() - 1);
        const size_t lo = (size_t) pos, hi = (std::min)(us.size() - 1, lo + 1);
        return us[lo] + (us[hi] - us[lo]) * (pos - (double) lo);
    };
    s.median = s.mean;
    s.median.us = quantile(0.5);
    s.p25 = quantile(0.25);
    s.p75 = quantile(0.75);
    return s;
}

void spin_until(Clock::time_point until) {   // the engine's host spins on the GPU's flags between two calls
    while (Clock::now() < until) pause_cpu();
}

struct Unit {
    const Store* st = nullptr;
    int n = 0, rows = 0, calls = 1, per = 0;
    bool hot = false;
    int cursor = 0;                       // the next cold expert
    Clock::time_point last = Clock::now();
    std::vector<int> index;               // the unit's experts, into the store
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<float> out;               // n experts x kRows rows x H
    // the direct path's buffers
    alignas(64) float dff[kRows][kFF];
    cpu::ActQ da2[kRows];
    alignas(64) uint8_t dhq[kRows][cpu::kNativeHBytes];

    /// `split`: T > MAXT is two calls, as the pool's API and the row kernels both require (iq_avx512.cpp gu_rows_nt
    /// takes 1..8 rows; iq_avx2.cpp iq4nl256_down_rows_v fills a y[8]): 12 = 6 + 6, 16 = 8 + 8.
    void configure(const Store& store, int experts, int rows_, bool hot_, bool split) {
        if (rows_ > kRows || (rows_ > kMaxT && (!split || rows_ % 2 != 0))) {
            std::fprintf(stderr, "pool_bench: %d rows are not a shape this bench runs\n", rows_);
            std::exit(1);
        }
        st = &store;
        n = experts;
        rows = rows_;
        hot = hot_;
        per = split && rows_ > kMaxT ? rows_ / 2 : rows_;
        calls = rows_ / per;
        index.assign((size_t) n, 0);
        jobs.assign((size_t) n, cpu::ExpertJobMulti{});
        out.assign((size_t) n * kRows * kH, 0.f);
    }

    void select() {                        // cold: the next n of the store; hot: always the first n
        for (int e = 0; e < n; ++e) index[(size_t) e] = hot ? e : (cursor + e) % kStore;
    }
    void advance() {
        if (!hot) cursor = (cursor + n) % kStore;
    }
    void point(int c) {                    // the engine's job lists (expert_pool_dispatch_multi)
        for (int e = 0; e < n; ++e) {
            cpu::ExpertJobMulti& j = jobs[(size_t) e];
            j.blob = st->expert(index[(size_t) e]);
            j.nt = per;
            j.fmt = nullptr;
            for (int t = 0; t < per; ++t) {
                const int r = c * kMaxT + t;
                j.act[t] = &st->qact[(size_t) r];
                j.nact[t] = st->fmt->gu >= 0 ? (const void*) (st->nact.data() + (size_t) r * cpu::kNativeActBytes) : nullptr;
                j.out[t] = out.data() + ((size_t) e * kRows + r) * kH;
            }
        }
    }

    // The engine's own call, timed as its telemetry times it (inside the call; a gap is the host's idle before it).
    Pt run_pool(cpu::ExpertPool& pool, int gap_us) {
        if (per > kMaxT) {
            std::fprintf(stderr, "pool_bench: the pool's job takes %d rows at most\n", kMaxT);
            std::exit(1);
        }
        Pt pt;
        select();
        for (int c = 0; c < calls; ++c) {
            if (gap_us > 0) {
                const Clock::time_point until = last + std::chrono::microseconds(gap_us);
                spin_until(until);
                last = until;
            }
            point(c);
            const double g0 = pool.ms_multi_gu, q0 = pool.ms_multi_q, d0 = pool.ms_multi_down;
            const Clock::time_point t0 = Clock::now();
            if (st->fmt->gu >= 0) pool.run_split_multi_native(st->nf, jobs.data(), n);
            else pool.run_split_multi(jobs.data(), n);
            last = Clock::now();
            pt.us += us_of(last - t0);
            pt.gu += 1e3 * (pool.ms_multi_gu - g0);
            pt.q += 1e3 * (pool.ms_multi_q - q0);
            pt.down += 1e3 * (pool.ms_multi_down - d0);
        }
        advance();
        return pt;
    }

    // One expert's whole rows on the calling thread: the pool's drain modes 5/6 (native) and 3/4 (shadow), no pool.
    // At most MAXT rows: the kernels' own limit, the same guard the pool's job has.
    void direct_expert(const uint8_t* blob, float* out0, int per_, int row0) {
        if (per_ > kMaxT) {
            std::fprintf(stderr, "pool_bench: the row kernels take %d rows at most\n", kMaxT);
            std::exit(1);
        }
        float* ff[kRows];
        float* outp[kRows];
        const void* act[kRows];
        const void* hq[kRows];
        const cpu::ActQ* a2[kRows];
        for (int t = 0; t < per_; ++t) {
            ff[t] = dff[t];
            outp[t] = out0 + (size_t) t * kH;
            a2[t] = &da2[t];
            hq[t] = dhq[t];
            if (st->fmt->gu >= 0) act[t] = st->nact.data() + (size_t) (row0 + t) * cpu::kNativeActBytes;
            else a2[t] = &st->qact[(size_t) (row0 + t)];
        }
        if (st->fmt->gu < 0) {
            cpu::s2_expert_gu_rows_multi(blob, a2, per_, ff, 0, kFF);
            for (int t = 0; t < per_; ++t) {
                cpu::act_quant_q8_1(dff[t], kFF, da2[t]);
                a2[t] = &da2[t];
            }
            cpu::s2_expert_down_rows_multi(blob, a2, per_, outp, 0, kH);
            return;
        }
        cpu::native_gu_rows(st->nf, blob, act, per_, ff, 0, kFF);
        if (cpu::q2_native_kernels(st->nf.d_type)) {     // a Q2_0 down: the planar activation and the AVX-512 kernel
            for (int t = 0; t < per_; ++t) {
                cpu::act_quant_any(dff[t], kFF, da2[t]);
                a2[t] = &da2[t];
            }
            cpu::q2_rows_any(blob + st->nf.down_off, st->nf.d_row, kFF / 64, a2, per_, outp, 0, kH);
            return;
        }
        for (int t = 0; t < per_; ++t) cpu::native_quant_h(st->nf, dff[t], dhq[t]);
        cpu::native_down_rows(st->nf, blob, hq, per_, outp, 0, kH);
    }

    Pt run_direct() {
        Pt pt;
        select();
        const Clock::time_point t0 = Clock::now();
        for (int e = 0; e < n; ++e)
            for (int c = 0; c < calls; ++c)
                direct_expert(st->expert(index[(size_t) e]), out.data() + (size_t) e * kRows * kH, per, c * kMaxT);
        last = Clock::now();
        pt.us = us_of(last - t0);
        advance();
        return pt;
    }
};

// The engine pins its host thread AFTER the pool is built (generate.cpp, measure_pool_affinity), and so must this: the
// pool reads the CPU layout from the affinity of the thread that constructs it, and a host pinned to one processor
// first leaves it that processor and no workers.
struct HostPin {
    cpu::ThreadAffinity prev;
    explicit HostPin(int core) {
        if (core >= 0) prev = cpu::pin_current_thread(core);
    }
    ~HostPin() {
        if (prev.valid) cpu::restore_thread_affinity(prev);
    }
    HostPin(const HostPin&) = delete;
    HostPin& operator=(const HostPin&) = delete;
};

// The box's streaming read rate with the pool's own layout: 16 threads, each pinned where the pool pins a thread
// (15 workers, then the host), each reading a 64 MiB region it has not just read; then one thread alone.
//
// THE READS MUST NOT BE FOLDABLE.  A reader that can see its buffer being filled has every load folded to the fill
// value: this test's first form (a zero-filled vector, the loop where the fill was visible) "read" 612 GB/s on a box
// whose DRAM streams 50.  So the pattern is not a constant, the readers reach it through a volatile pointer, and the
// sum comes back from a function the compiler may not inline.  The checksum is printed: it is the data's.
#if defined(_MSC_VER)
#define POOL_BENCH_NOINLINE __declspec(noinline)
#else
#define POOL_BENCH_NOINLINE __attribute__((noinline))
#endif
const uint64_t* volatile g_stream = nullptr;       // the readers' only view of the buffer

POOL_BENCH_NOINLINE uint64_t stream_sum(const uint64_t* p, size_t n) {
    uint64_t sum = 0;
    for (size_t k = 0; k < n; ++k) sum += p[k];
    return sum;
}

/// One rep of `threads` readers: thread i takes region (i + rep) mod 16 - with 16 threads every region exactly once.
uint64_t stream_rep(const int* cores, size_t per, int rep, int threads, double* seconds) {
    std::atomic<uint64_t> sink{0};
    const auto reader = [per, &sink, rep](int i) {
        sink.fetch_add(stream_sum(g_stream + ((size_t) (i + rep) % (size_t) kThreads) * per, per),
                       std::memory_order_relaxed);
    };
    std::vector<std::thread> th;
    const Clock::time_point t0 = Clock::now();
    for (int i = 1; i < threads; ++i) {
        th.emplace_back([&reader, cores, i] {
            cpu::pin_current_thread(cores[i - 1]);
            reader(i);
        });
    }
    reader(0);                                       // the main thread, on the host core
    for (std::thread& t : th) t.join();
    *seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return sink.load();
}

/// `worker_cores`: the pool's physical cores, detected while no thread was pinned (a pinned caller reads one
/// processor and none besides - the crash this function's first form died of).
void stream_roofline(int reps, int host, const std::vector<int>& worker_cores) {
    if (worker_cores.size() + 1 < (size_t) kThreads) {
        std::fprintf(stderr, "pool_bench: %zu worker cores for %d threads\n", worker_cores.size(), kThreads);
        std::exit(1);
    }
    const HostPin host_pin(host);
    const size_t per = (size_t) 1 << 23;                       // 64 MiB of uint64
    std::vector<uint64_t> buf(per * (size_t) kThreads);        // 1 GiB
    for (size_t k = 0; k < buf.size(); ++k) buf[k] = (uint64_t) k * 0x9E3779B97F4A7C15ull + 1u;
    g_stream = buf.data();
    const int* const cores = worker_cores.data();
    std::vector<double> all((size_t) reps), one((size_t) reps);
    uint64_t checksum = 0;
    for (int rep = 0; rep < reps; ++rep) {
        const uint64_t got = stream_rep(cores, per, rep, kThreads, &all[(size_t) rep]);
        if (rep == 0) checksum = got;
        if (got != checksum) {
            std::fprintf(stderr, "pool_bench: the stream readers' checksum moved between reps\n");
            std::exit(1);
        }
    }
    for (int rep = 0; rep < reps; ++rep)                    // then one thread alone: regions the 16-thread reps streamed
        stream_rep(cores, per, rep, 1, &one[(size_t) rep]);
    g_stream = nullptr;
    std::sort(all.begin(), all.end());                      // seconds: the shortest is the best rate
    std::sort(one.begin(), one.end());
    const double all_bytes = (double) (buf.size() * sizeof(uint64_t)), one_bytes = (double) (per * sizeof(uint64_t));
    std::printf("stream: %d threads read %.0f MB they had not touched: %.1f GB/s (best of %d), %.1f GB/s (median); one thread, %.0f MB:\n"
                "        %.1f GB/s (best), %.1f GB/s (median); checksum %016llx\n",
                kThreads, all_bytes / 1e6, all_bytes / all.front() / 1e9, reps,
                all_bytes / all[(size_t) (reps / 2)] / 1e9, one_bytes / 1e6, one_bytes / one.front() / 1e9,
                one_bytes / one[(size_t) (reps / 2)] / 1e9, (unsigned long long) checksum);
}

// ---------------------------------------------------------------------------------------------------------------
// Results and tables.

struct Result {
    Series tn[2][kTCount][kNCount];         // 15 workers + host, gap 0: [cold, hot][T][N]
    Series small[2][2];                     // T=12, N=8, gap 0: [cold, hot][8 workers, 4 workers]
    Series direct_t[2][kTCount];            // N=8, one thread, no pool: [cold, hot][T] (T > 8 as 6+6, 8+8)
    Series sync[4];                         // cold, T=12, N=8: gap 150 | gap 2000 | gap 150 spin 20 ms | gap 150 diet 0
};

/// us per expert-call: a unit holds n experts, each in every one of its calls.
double per_expert(double unit_us, int n) { return unit_us / n; }

void print_phases(const Pt& pt, int n) { std::printf("[%5.1f %4.1f %5.1f] ", pt.gu / n, pt.q / n, pt.down / n); }

/// `blob`: one expert's bytes; `calls`: how many times an expert is read in a unit.
struct Fit {
    double a, b, r2;
};
void print_rates(double us_per_expert, double blob, int calls, int threads) {
    const double s = us_per_expert * 1e-6;
    std::printf("%6.1f %7.2f", blob * calls / s / 1e9, 3.0 * kH * kFF * calls / s / threads / 1e9);
}

// ---------------------------------------------------------------------------------------------------------------
// F-460: the prefetch experiment (`--prefetch`).  The default run does not enter this section.
//
// THE QUESTION.  A DRAM-resident call is bound by DRAM reads and a cache-resident one is cheaper (tables 1 and 3).  Does
// a read-only pass over the NEXT call's experts, made while the pool idles, hand the call the hot rate - after a gap,
// under other traffic, with the pool's threads spread over both CCDs - and what does the pass itself cost?
//
// THE UNIT.  Select the call's N experts (cold: the next N of the store) -> THE PASS: every byte of those N experts is
// read once through the not-foldable pattern above; its checksum is checked against per-expert sums, so the pass read
// exactly the bytes the call will read, each once -> the host idles for the gap G while ONE thread reads X MiB of
// unrelated memory -> the timed call (the engine's own call, timed inside, as its telemetry times it) -> the traffic
// is waited for, so no unit overlaps the next.
//
// WHO READS.  The pool claims its tasks at run time (pool.cpp: "one claim per iteration, so a slow worker takes fewer
// experts"), so no pass can place a line in the core that will read it - only in a CCD's L3:
//   host  the host thread itself reads everything: no wake, no second thread
//   one   one thread beside the host (its SMT sibling): every line lands in the host's CCD's L3, and the other CCD's
//         threads read their share across the fabric
//   all   16 threads beside the pool's 16, each a contiguous 1/16 of the call's bytes: each line in one CCD's L3
//   both  one thread per CCD, each reading everything: every line in both L3s (each line is read twice)
//
// THE CONTROLS.  A call after a gap wakes parked workers (table 4), and the pass itself streams the call's bytes through
// the L3s.  So every cold control runs the same gap and the same traffic behind a SHAM pass: the same readers reading
// the same number of bytes from unrelated memory.  `hot` is the same N experts every unit, no pass, under the same gap
// and traffic: what an already-resident set survives of them.

constexpr int kPfRows = 6;
constexpr int kPfNs[] = {2, 3, 4, 8};
constexpr int kPfNCount = (int) (sizeof kPfNs / sizeof kPfNs[0]);
constexpr int kPfMaxN = 8;
constexpr double kPfGapsMs[] = {0.0, 0.1, 0.25, 0.5};      // the host's idle time between the pass and the call
constexpr int kPfGCount = (int) (sizeof kPfGapsMs / sizeof kPfGapsMs[0]);
constexpr int kPfTrafficMib[] = {0, 8, 32};                // what the traffic thread reads during the gap
constexpr int kPfXCount = (int) (sizeof kPfTrafficMib / sizeof kPfTrafficMib[0]);
constexpr int kPfStopG = 2, kPfStopX = 1;                   // the stop rule's cell: G = 0.25 ms, 8 MiB of traffic
constexpr int kPfReaders = 16;                             // beside the pool's threads: the host, then the 15 workers
enum PfWho { kPfHost = 0, kPfOne, kPfAll, kPfBoth, kPfWhoCount };
constexpr const char* kPfWhoName[kPfWhoCount] = {"host", "one", "all", "both"};
constexpr size_t kPfTrafficBytes = (size_t) 512 << 20;     // the traffic's region: 16 units of 32 MiB before a wrap
constexpr size_t kPfShamBytes = (size_t) 256 << 20;        // the sham pass reads its own region, apart from it

std::string read_sysfs(const std::string& path) {
#if defined(_WIN32)
    (void) path;
    return std::string();
#else
    std::FILE* const f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return std::string();
    char buf[256];
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[n] = '\0';
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
#endif
}

/// The SMT sibling of logical processor `cpu` (/sys/devices/system/cpu/cpuN/topology/thread_siblings_list, "0,16" or
/// "0-3,4-7"): -1 where there is none or it is not told.
int smt_sibling(int cpu) {
    const std::string list = read_sysfs("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
    size_t i = 0;
    while (i < list.size()) {
        const size_t j = list.find(',', i);
        const std::string token = list.substr(i, (j == std::string::npos ? list.size() : j) - i);
        const size_t dash = token.find('-');
        const int a = std::atoi(token.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(token.c_str() + dash + 1);
        for (int c = a; c <= b; ++c)
            if (c != cpu) return c;
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return -1;
}

/// `cpu`'s last-level cache: the logical processors that share it (its domain) and its size in MiB.
struct L3 {
    std::string cpus;
    int mib = 0;
};
L3 l3_of(int cpu) {
    L3 l;
    for (int i = 0; i < 6 && l.cpus.empty(); ++i) {
        const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index" + std::to_string(i) + "/";
        if (read_sysfs(dir + "level") != "3") continue;
        l.cpus = read_sysfs(dir + "shared_cpu_list");
        l.mib = (int) (std::atof(read_sysfs(dir + "size").c_str()) / 1024.0 + 0.5);   // "98304K"
    }
    return l;
}

const uint8_t* volatile g_prefetch = nullptr;      // the pass's only view of the expert store
const uint8_t* volatile g_scratch = nullptr;       // the traffic thread's and the sham pass's only view of their buffer

/// Every byte of [p, p + n), the 8-aligned body through the not-foldable word sum.
uint64_t bytes_sum(const uint8_t* p, size_t n) {
    uint64_t sum = 0;
    size_t i = 0;
    for (; i < n && (reinterpret_cast<uintptr_t>(p + i) & 7u) != 0; ++i) sum += p[i];
    const size_t words = (n - i) >> 3;
    if (words > 0) sum += stream_sum(reinterpret_cast<const uint64_t*>(p + i), words);
    for (i += words << 3; i < n; ++i) sum += p[i];
    return sum;
}

struct Seg { size_t off, len; };                   // the bytes [off, off + len) from the pass's base

/// Reader `pos` of `k`: the linear slice [pos*T/k, (pos+1)*T/k) over the segments' concatenated T bytes, or all of them
/// when `whole`.
uint64_t read_part(const uint8_t* base, const Seg* segs, int nsegs, int pos, int k, bool whole) {
    size_t total = 0;
    for (int i = 0; i < nsegs; ++i) total += segs[i].len;
    size_t lo = 0, hi = total;
    if (!whole) {
        lo = total * (size_t) pos / (size_t) k;
        hi = total * (size_t) (pos + 1) / (size_t) k;
    }
    uint64_t sum = 0;
    size_t lin = 0;
    for (int i = 0; i < nsegs; ++i) {
        const size_t a = (std::max)(lo, lin), b = (std::min)(hi, lin + segs[i].len);
        if (a < b) sum += bytes_sum(base + segs[i].off + (a - lin), b - a);
        lin += segs[i].len;
    }
    return sum;
}

/// The pass's readers: persistent threads, each pinned to the SMT sibling of one pool thread (the pool's threads keep
/// their own logical processors), asleep on their own counter between passes.  A pass wakes the readers of its set;
/// its time is the wake and the reads.
struct Readers {
    struct Job {
        const uint8_t* const volatile* base = nullptr;   // &g_prefetch or &g_scratch
        Seg segs[kPfMaxN] = {};
        int nsegs = 0, k = 0;
        bool whole = true;
        int pos[kPfReaders] = {};                        // a reader's place in the pass (-1: not in it)
    };
    Job job = {};
    std::atomic<uint32_t> go[kPfReaders] = {};
    std::atomic<uint32_t> done{0};
    std::atomic<bool> parked[kPfReaders] = {};         // the reader has left its CPU's run queue for the wait
    std::atomic<bool> stop{false};
    std::atomic<int64_t> woke_ns[kPfReaders] = {};     // when the reader began its slice (ns since the epoch)
    uint64_t last[kPfReaders] = {};                      // the readers' sums of the last pass they ran
    int core[kPfReaders] = {};                           // where reader i is pinned
    std::vector<std::thread> th;

    void start() {
        for (int i = 0; i < kPfReaders; ++i) {
            th.emplace_back([this, i] {
                cpu::pin_current_thread(core[i]);
                uint32_t seen = 0;
                for (;;) {
                    go[i].wait(seen, std::memory_order_acquire);
                    seen = go[i].load(std::memory_order_relaxed);
                    if (stop.load(std::memory_order_relaxed)) break;
                    woke_ns[i].store(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(),
                                     std::memory_order_relaxed);
                    const uint8_t* const base = *job.base;   // through the volatile pointer: the loads are not foldable
                    last[i] = read_part(base, job.segs, job.nsegs, job.pos[i], job.k, job.whole);
                    done.fetch_add(1, std::memory_order_release);
                    parked[i].store(true, std::memory_order_release);
                }
            });
        }
    }
    /// One pass by readers ids[0..k): each its slice, or every byte when `whole`.  `wake` receives the slowest reader's
    /// time from the signal to its first read; the return is the pass's own us, the wake included.  The pass ends when
    /// every reader has parked again, so none of it runs beside the traffic thread that follows.
    double run(const int* ids, int k, bool whole, const uint8_t* const volatile* base, const Seg* segs, int nsegs,
               double& wake) {
        job.base = base;
        job.nsegs = nsegs;
        job.k = k;
        job.whole = whole;
        for (int i = 0; i < kPfReaders; ++i) job.pos[i] = -1;
        for (int i = 0; i < nsegs; ++i) job.segs[i] = segs[i];
        for (int r = 0; r < k; ++r) {
            job.pos[ids[r]] = r;
            parked[ids[r]].store(false, std::memory_order_relaxed);
        }
        done.store(0, std::memory_order_relaxed);
        const Clock::time_point t0 = Clock::now();
        for (int r = 0; r < k; ++r) {
            go[ids[r]].fetch_add(1, std::memory_order_release);
            go[ids[r]].notify_one();
        }
        while (done.load(std::memory_order_acquire) < (uint32_t) k) pause_cpu();
        const double us = us_of(Clock::now() - t0);
        wake = 0.0;
        for (int r = 0; r < k; ++r) {
            const double w = std::chrono::duration<double, std::micro>(
                Clock::time_point(std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(
                    woke_ns[ids[r]].load(std::memory_order_relaxed)))) - t0).count();
            wake = (std::max)(wake, w);
        }
        for (int r = 0; r < k; ++r)
            while (!parked[ids[r]].load(std::memory_order_acquire)) pause_cpu();
        return us;
    }
    uint64_t sum(const int* ids, int k) const {
        uint64_t s = 0;
        for (int r = 0; r < k; ++r) s += last[ids[r]];
        return s;
    }
    void join() {
        stop.store(true, std::memory_order_relaxed);
        for (int i = 0; i < kPfReaders; ++i) {
            go[i].fetch_add(1, std::memory_order_release);
            go[i].notify_one();
        }
        for (std::thread& t : th) t.join();
    }
};

/// One thread reading X MiB of the traffic region while the host idles: the engine's host work and DMA staging, which
/// pass through the memory system while the pool does nothing.  Pinned beside the host, where reader 0 lives too; the
/// two never run at once, and each is waited into its park before the other is woken.  It shares a processor with no
/// pool thread: its effect on a call is the memory system's.
struct Traffic {
    std::atomic<uint32_t> go{0}, done{0};
    std::atomic<bool> parked{false}, stop{false};
    std::atomic<size_t> off{0}, bytes{0};
    std::atomic<int64_t> end_ns{0};
    uint64_t sum = 0;                           // what it read: written by the thread, read after the join
    std::thread th;
    size_t cursor = 0;                          // the host's side: where the next unit's traffic begins
    uint32_t ticket = 0;
    Clock::time_point t_start = Clock::now();

    void start(int core) {
        th = std::thread([this, core] {
            cpu::pin_current_thread(core);
            uint32_t seen = 0;
            for (;;) {
                go.wait(seen, std::memory_order_acquire);
                seen = go.load(std::memory_order_relaxed);
                if (stop.load(std::memory_order_relaxed)) break;
                const uint8_t* const base = g_scratch;
                const size_t n = bytes.load(std::memory_order_relaxed), o = off.load(std::memory_order_relaxed);
                sum += stream_sum(reinterpret_cast<const uint64_t*>(base + o), n >> 3);
                end_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(),
                             std::memory_order_release);
                done.fetch_add(1, std::memory_order_release);
                parked.store(true, std::memory_order_release);
            }
        });
    }
    /// This unit's traffic: n bytes of the region [0, region).  Returns the ticket `wait` takes.
    uint32_t begin(size_t n, size_t region) {
        if (cursor + n > region) cursor = 0;
        off.store(cursor, std::memory_order_relaxed);
        bytes.store(n, std::memory_order_relaxed);
        cursor += n;
        parked.store(false, std::memory_order_relaxed);
        t_start = Clock::now();
        go.fetch_add(1, std::memory_order_release);
        go.notify_one();
        return ++ticket;
    }
    /// The traffic of ticket `t` has ended and its thread has left its processor (which it shares with reader 0).
    void wait(uint32_t t) {
        while (done.load(std::memory_order_acquire) < t) pause_cpu();
        while (!parked.load(std::memory_order_acquire)) pause_cpu();
    }
    void join() {
        stop.store(true, std::memory_order_relaxed);
        go.fetch_add(1, std::memory_order_release);
        go.notify_one();
        th.join();
    }
};

struct PfPt { double us = 0, gu = 0, q = 0, down = 0, pf = 0, wake = 0, traffic = 0, overrun = 0; };
struct PfSeries {
    double call = 0, p25 = 0, p75 = 0;          // the call's us per unit of N experts: median, quartiles
    double pf = 0, pf_p25 = 0, pf_p75 = 0;      // the pass's own us (0: no pass)
    double wake = 0;                            // median: the slowest reader's time from the signal to its first read
    double traffic = 0, overrun = 0;            // medians: the traffic's duration, and its overrun past the gap's end
    Pt mean;                                    // the call's phases, means
};

template <class Run>
PfSeries measure_pf(int units, Run&& run) {
    for (int i = 0; i < kWarm; ++i) run();
    std::vector<PfPt> pts((size_t) units);
    for (int i = 0; i < units; ++i) pts[(size_t) i] = run();
    const auto quantile = [&pts, units](double PfPt::*field, double f) {
        std::vector<double> v((size_t) units);
        for (int i = 0; i < units; ++i) v[(size_t) i] = pts[(size_t) i].*field;
        std::sort(v.begin(), v.end());
        const double pos = f * (double) (units - 1);
        const size_t lo = (size_t) pos, hi = (std::min)((size_t) (units - 1), lo + 1);
        return v[lo] + (v[hi] - v[lo]) * (pos - (double) lo);
    };
    PfSeries s;
    s.call = quantile(&PfPt::us, 0.5);
    s.p25 = quantile(&PfPt::us, 0.25);
    s.p75 = quantile(&PfPt::us, 0.75);
    s.pf = quantile(&PfPt::pf, 0.5);
    s.pf_p25 = quantile(&PfPt::pf, 0.25);
    s.pf_p75 = quantile(&PfPt::pf, 0.75);
    s.wake = quantile(&PfPt::wake, 0.5);
    s.traffic = quantile(&PfPt::traffic, 0.5);
    s.overrun = quantile(&PfPt::overrun, 0.5);
    for (const PfPt& p : pts) {
        s.mean.us += p.us / units;
        s.mean.gu += p.gu / units;
        s.mean.q += p.q / units;
        s.mean.down += p.down / units;
    }
    return s;
}

struct PfRes {
    PfSeries cold0, hot0;                                   // the anchors: gap 0, no pass, no traffic
    PfSeries hot[kPfGCount][kPfXCount];                     // no pass, under the gap and the traffic
    PfSeries cold[kPfWhoCount][kPfGCount][kPfXCount];       // behind a sham pass by `who`
    PfSeries pf[kPfWhoCount][kPfGCount][kPfXCount];         // behind the real pass by `who`
};

int run(int argc, char** argv) {
    if (const cpu::CpuFeatures feat = cpu::cpu_features(); !feat.usable()) {   // the AVX-512 kernels: nothing to time here
        std::printf("pool_bench: CPU lacks %s: SKIPPED\n", feat.reason());
        return 77;
    }
    // F-460: `--prefetch` is an option anywhere on the line; the positional arguments are unchanged.
    std::vector<std::string> positional;
    bool prefetch = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--prefetch") == 0) prefetch = true;
        else positional.emplace_back(argv[i]);
    }
    const int units = !positional.empty() ? std::atoi(positional[0].c_str()) : 30;
    const std::string only = positional.size() > 1 ? positional[1] : std::string();
    std::setvbuf(stdout, nullptr, _IOLBF, 0);      // a run that dies must not take its tables with it
    if (units < 5) {
        std::fprintf(stderr, "usage: pool_bench [units >= 5] [format-name] [--prefetch]\n");
        return 2;
    }
    // The layout is read before anything is pinned, and nothing is detected again afterwards: a pinned thread's mask
    // is one processor.  A process restricted to fewer physical cores than the lane's pool uses cannot measure it:
    // refuse, the way the engine would have measured nothing.
    const int host = cpu::planned_host_core(cpu::PoolAffinity::All);
    const std::vector<int> worker_cores = cpu::physical_cores(true, cpu::PoolAffinity::All);
    if (host < 0 || worker_cores.size() < 15) {
        std::fprintf(stderr, "pool_bench: this process reaches %zu physical cores besides the host's (logical %d); the lane's "
                             "pool needs 15 - run it with the whole CPU allowed\n",
                     worker_cores.size(), host);
        return 3;
    }
    // The standing recipe's pool environment (launch-custom-iq3s-8081.sh: STRATA_POOL_DIET=1, STRATA_POOL_SPIN_US=100),
    // set here so a run does not depend on the shell.  Table 4's arms override it around a pool's construction.
    set_env("STRATA_POOL_DIET", "1");
    set_env("STRATA_POOL_SPIN_US", "100");

    std::vector<Store> stores;
    std::vector<int> live;                   // the formats measured, into kFormats
    std::vector<Result> res;
    stores.reserve((size_t) kFormatCount);
    for (int f = 0; f < kFormatCount; ++f) {
        if (!only.empty() && std::string(kFormats[f].name).find(only) == std::string::npos) continue;
        live.push_back(f);
        stores.emplace_back();
        build_store(stores.back(), kFormats[f], std::getenv("POOL_BENCH_FIXTURES"));
    }
    if (live.empty()) {
        std::fprintf(stderr, "pool_bench: no format's name contains \"%s\"\n", only.c_str());
        return 2;
    }
    res.assign((size_t) kFormatCount, Result{});
    const auto store_of = [&stores, &live](int f) -> const Store& {
        return stores[(size_t) (std::find(live.begin(), live.end(), f) - live.begin())];
    };
    int native_layers = 0;
    double native_bytes = 0;                 // the pack's layer-weighted blob
    double resident = 0;
    for (int f : live) {
        resident += (double) ((size_t) kStore * store_of(f).bytes);
        if (kFormats[f].layers > 0) {
            native_layers += kFormats[f].layers;
            native_bytes += (double) kFormats[f].layers * store_of(f).bytes;
        }
    }
    const double mean_blob = native_bytes / (std::max)(1, native_layers);
    double smallest_store = 1e300;
    for (int f : live) smallest_store = (std::min)(smallest_store, (double) ((size_t) kStore * store_of(f).bytes));

    std::printf("# F-458 pool_bench: ExpertPool::run_split_multi[_native] per expert at the concurrent lane's shape\n");
    std::printf("cpu: %s | %u logical processors\n", cpu_model().c_str(), std::thread::hardware_concurrency());
    std::printf("fixture: %zu formats, %.2f GB of synthetic experts (%d per format; the smallest store is %.0f MB, far over any cache)\n",
                live.size(), resident / 1e9, kStore, smallest_store / 1e6);
    std::printf("env: STRATA_POOL_DIET=1 STRATA_POOL_SPIN_US=100 (the standing recipe) | units per point: %d (the median; %d warm-up) | %.3f M weights per expert-call (3 x %d x %d)\n",
                units, kWarm, 3.0 * kH * kFF / 1e6, kH, kFF);
    stream_roofline(7, host, worker_cores);
    std::printf("layout: the host on logical processor %d; workers pinned to the %zu physical cores after it, in the OS's order\n",
                host, worker_cores.size());

    // One point: `pool` null = the direct path (one thread).  Pool points are always `split` (the API's MAXT rows).
    const auto run_point = [&](Unit& u, const Store& st, cpu::ExpertPool* pool, int experts, int rows, bool hot, int gap_us,
                               bool split) {
        u.configure(st, experts, rows, hot, split);
        if (pool == nullptr) return measure(units, [&] { return u.run_direct(); });
        return measure(units, [&] { return u.run_pool(*pool, gap_us); });
    };
    const auto make_pool = [](int workers) {
        return std::unique_ptr<cpu::ExpertPool>(
            new cpu::ExpertPool(workers, /*pin=*/true, /*host_works=*/true, cpu::PoolAffinity::All, /*tasks=*/0));
    };

    // ---- F-460: the prefetch experiment (--prefetch); then the run ends
    if (prefetch) {
        // The readers go beside the pool's threads, on each one's SMT sibling, so a pool thread keeps its own logical
        // processor.  The topology is read before the host is pinned: a pinned thread's mask is one processor.
        int beside[kPfReaders];
        beside[0] = host;
        for (int i = 1; i < kPfReaders; ++i) beside[i] = worker_cores[(size_t) (i - 1)];
        for (int i = 0; i < kPfReaders; ++i)
            if (const int s = smt_sibling(beside[i]); s >= 0) beside[i] = s;
        L3 l3[kPfReaders];
        for (int i = 0; i < kPfReaders; ++i) l3[i] = l3_of(beside[i]);
        int other = 0;                                       // `both`: a reader whose L3 is not the host's
        for (int i = 1; i < kPfReaders; ++i)
            if (!l3[i].cpus.empty() && l3[i].cpus != l3[0].cpus) { other = i; break; }
        int ids_all[kPfReaders];
        for (int i = 0; i < kPfReaders; ++i) ids_all[i] = i;
        const int ids_one[1] = {0}, ids_both[2] = {0, other};
        const int* const ids[kPfWhoCount] = {ids_one, ids_one, ids_all, ids_both};
        const int count[kPfWhoCount] = {1, 1, kPfReaders, other != 0 ? 2 : 0};
        const bool whole[kPfWhoCount] = {true, true, false, true};
        const bool has[kPfWhoCount] = {true, true, true, other != 0};

        Readers readers;
        for (int i = 0; i < kPfReaders; ++i) readers.core[i] = beside[i];
        std::vector<uint8_t> scratch(kPfTrafficBytes + kPfShamBytes);
        for (size_t k = 0; k < scratch.size(); ++k) scratch[k] = (uint8_t) (k * 131u + 7u);
        g_scratch = scratch.data();
        const std::unique_ptr<cpu::ExpertPool> pool = make_pool(15);   // built unpinned, as above; then the host takes its core
        const HostPin host_pin(host);
        readers.start();
        Traffic traffic;
        // POOL_BENCH_PF_TRAFFIC_CCD=B: the traffic beside a worker of the other CCD, so it streams through that CCD's L3
        // (the 32 MiB one on this CPU) instead of the host's.  It then shares a processor with that worker's reader; the
        // park handshakes keep the two from running at once.
        const char* const traffic_ccd = std::getenv("POOL_BENCH_PF_TRAFFIC_CCD");
        const bool traffic_b = other != 0 && traffic_ccd != nullptr && traffic_ccd[0] == 'B';
        const int traffic_core = traffic_b ? beside[other] : beside[0];
        traffic.start(traffic_core);

        std::printf("\n# F-460 --prefetch: T=%d, N in {2, 3, 4, 8}, cold experts cycling through the store, %d units per point (the median)\n",
                    kPfRows, units);
        std::printf("pool: %d workers + the draining host; the host on logical processor %d, its L3 %d MiB shared with CPUs %s\n",
                    pool->workers(), host, l3[0].mib, l3[0].cpus.c_str());
        std::printf("readers: host = the host thread itself; one = a thread beside the host, on logical processor %d; all = %d threads beside\n"
                    "         the pool's threads (beside the host, then beside each worker in the OS's order); their L3 domains A/B:",
                    beside[0], kPfReaders);
        for (int i = 0; i < kPfReaders; ++i) std::printf(" %c", l3[i].cpus == l3[0].cpus ? 'A' : 'B');
        std::printf("  (B = CPUs %s, %d MiB)\n", l3[other].cpus.c_str(), l3[other].mib);
        if (other != 0)
            std::printf("         both = the thread beside the host and the one beside worker %d, on logical processor %d\n", other - 1, beside[other]);
        else
            std::printf("         both = not run: every pool thread shares one L3\n");
        std::printf("traffic: one thread%s (logical processor %d, L3 %d MiB shared with CPUs %s) reads X MiB of a %zu MiB region\n"
                    "         during the gap; the sham passes read a separate %zu MiB region.  The workers park as the standing\n"
                    "         recipe has them (STRATA_POOL_SPIN_US=100).\n",
                    traffic_b ? " beside a worker on the other CCD (POOL_BENCH_PF_TRAFFIC_CCD=B)" : " beside the host", traffic_core,
                    l3_of(traffic_core).mib, l3_of(traffic_core).cpus.c_str(), kPfTrafficBytes >> 20, kPfShamBytes >> 20);

        const double load_start = loadavg();
        const bool trace = env_on("POOL_BENCH_PF_TRACE");
        std::vector<PfRes> pf((size_t) live.size() * (size_t) kPfNCount);
        size_t sham_cursor = 0;
        uint64_t sham_sum = 0, checks = 0;
        Unit u;
        for (size_t li = 0; li < live.size(); ++li) {
            const int f = live[li];
            const Store& st = store_of(f);
            g_prefetch = st.mem.data();
            std::vector<uint64_t> esum((size_t) kStore);           // every expert's byte sum: what a pass over it must return
            for (int e = 0; e < kStore; ++e) esum[(size_t) e] = bytes_sum(st.mem.data() + (size_t) e * st.bytes, st.bytes);
            const auto fail = [&](int n, int who, uint64_t got, uint64_t expected) {
                std::fprintf(stderr, "pool_bench: --prefetch: %s N=%d who=%s: the pass returned %016llx, the experts' bytes sum to %016llx -"
                                     " it did not read exactly their bytes, each once\n",
                             kFormats[f].name, n, kPfWhoName[who], (unsigned long long) got, (unsigned long long) expected);
                std::exit(2);
            };
            for (int ni = 0; ni < kPfNCount; ++ni) {
                const int n = kPfNs[ni];
                PfRes& r = pf[li * (size_t) kPfNCount + (size_t) ni];
                // One point: `who` -1 = no pass (the anchors, and hot); `sham` = the pass reads the same byte count from
                // unrelated memory.  The workers park as they do in the engine between two layers.
                const auto series = [&](bool hot, int who, int gi, int xi, bool sham) {
                    u.configure(st, n, kPfRows, hot, true);
                    return measure_pf(units, [&] {
                        PfPt pt;
                        u.select();
                        const int gap_us = (int) (kPfGapsMs[gi] * 1000.0 + 0.5);
                        const size_t xbytes = (size_t) kPfTrafficMib[xi] << 20;
                        Seg segs[kPfMaxN];
                        int nsegs = 0;
                        uint64_t expected = 0;
                        const uint8_t* const volatile* base = &g_prefetch;
                        if (who >= 0 && sham) {
                            if (sham_cursor + (size_t) n * st.bytes > kPfShamBytes) sham_cursor = 0;
                            segs[0] = {kPfTrafficBytes + sham_cursor, (size_t) n * st.bytes};
                            sham_cursor += (size_t) n * st.bytes;
                            nsegs = 1;
                            base = &g_scratch;
                        } else if (who >= 0) {
                            for (int e = 0; e < n; ++e) {
                                segs[e] = {(size_t) u.index[(size_t) e] * st.bytes, st.bytes};
                                expected += esum[(size_t) u.index[(size_t) e]];
                            }
                            nsegs = n;
                        }
                        // the pass: the host thread itself, or the readers of `who`
                        uint64_t got = 0;
                        if (who == kPfHost) {
                            const Clock::time_point t0 = Clock::now();
                            got = read_part(*base, segs, nsegs, 0, 1, true);
                            pt.pf = us_of(Clock::now() - t0);
                        } else if (who >= 0) {
                            pt.pf = readers.run(ids[who], count[who], whole[who], base, segs, nsegs, pt.wake);
                            got = readers.sum(ids[who], count[who]);
                        }
                        if (who >= 0) {
                            if (sham) sham_sum += got;
                            else if (got != (whole[who] ? (uint64_t) count[who] : 1u) * expected) fail(n, who, got, expected);
                            else ++checks;
                        }
                        // the gap, with the traffic in it
                        const Clock::time_point t_gap = Clock::now();
                        uint32_t ticket = 0;
                        if (xbytes > 0) ticket = traffic.begin(xbytes, kPfTrafficBytes);
                        const Clock::time_point gap_end = t_gap + std::chrono::microseconds(gap_us);
                        if (gap_us > 0) spin_until(gap_end);
                        // the call, timed inside as the engine's telemetry times it
                        u.point(0);
                        const double g0 = pool->ms_multi_gu, q0 = pool->ms_multi_q, d0 = pool->ms_multi_down;
                        const Clock::time_point t0 = Clock::now();
                        if (st.fmt->gu >= 0) pool->run_split_multi_native(st.nf, u.jobs.data(), n);
                        else pool->run_split_multi(u.jobs.data(), n);
                        const Clock::time_point t1 = Clock::now();
                        pt.us = us_of(t1 - t0);
                        pt.gu = 1e3 * (pool->ms_multi_gu - g0);
                        pt.q = 1e3 * (pool->ms_multi_q - q0);
                        pt.down = 1e3 * (pool->ms_multi_down - d0);
                        if (xbytes > 0) {                          // no unit overlaps the next: the traffic is waited for
                            traffic.wait(ticket);
                            const Clock::time_point t_end = Clock::time_point(std::chrono::duration_cast<Clock::duration>(
                                std::chrono::nanoseconds(traffic.end_ns.load(std::memory_order_relaxed))));
                            pt.traffic = us_of(t_end - traffic.t_start);
                            pt.overrun = (std::max)(0.0, us_of(t_end - gap_end));
                        }
                        u.advance();
                        if (trace)   // POOL_BENCH_PF_TRACE=1: every unit, to read a series' shape
                            std::fprintf(stderr, "unit %-14s N=%d %-4s G=%.2f X=%2d%s: pass %6.1f (wake %5.1f) call %6.1f traffic %6.1f overrun %6.1f\n",
                                         kFormats[f].name, n, who >= 0 ? kPfWhoName[who] : "none", kPfGapsMs[gi], kPfTrafficMib[xi],
                                         sham ? " sham" : "", pt.pf, pt.wake, pt.us, pt.traffic, pt.overrun);
                        return pt;
                    });
                };
                r.cold0 = series(false, -1, 0, 0, false);
                r.hot0 = series(true, -1, 0, 0, false);
                for (int gi = 0; gi < kPfGCount; ++gi)
                    for (int xi = 0; xi < kPfXCount; ++xi) {
                        r.hot[gi][xi] = series(true, -1, gi, xi, false);
                        for (int w = 0; w < kPfWhoCount; ++w) {
                            if (!has[w]) continue;
                            r.cold[w][gi][xi] = series(false, w, gi, xi, true);
                            r.pf[w][gi][xi] = series(false, w, gi, xi, false);
                        }
                    }
                std::printf("[prefetch] %-14s N=%d: cold %.1f us | hot %.1f us | one, gap 0, no traffic: pass %.1f us, call %.1f us (%.2f of its cold)\n",
                            kFormats[f].name, n, r.cold0.call, r.hot0.call, r.pf[kPfOne][0][0].pf, r.pf[kPfOne][0][0].call,
                            r.pf[kPfOne][0][0].call / r.cold[kPfOne][0][0].call);
            }
        }
        readers.join();
        traffic.join();
        g_prefetch = nullptr;
        g_scratch = nullptr;
        std::printf("\nload: the kernel's 1-minute average %.2f when the measured phase began, %.2f when it ended; %llu passes returned their\n"
                    "     experts' sum (exactly their bytes, each once); the sham passes read %016llx, the traffic %016llx\n",
                    load_start, loadavg(), (unsigned long long) checks, (unsigned long long) sham_sum, (unsigned long long) traffic.sum);

        // ---- tables P1, P2: per format, then the pack's layer-weighted native mix.  `at(ni, kind, who, gi, xi)`: kind 0 the
        // cold anchor, 1 the hot anchor, 2 hot under the gap and traffic, 3 the cold control of `who`, 4 the pass of `who`.
        const auto cell = [&](size_t li, int ni, int kind, int w, int gi, int xi) -> const PfSeries& {
            const PfRes& r = pf[li * (size_t) kPfNCount + (size_t) ni];
            switch (kind) {
                case 0: return r.cold0;
                case 1: return r.hot0;
                case 2: return r.hot[gi][xi];
                case 3: return r.cold[w][gi][xi];
                default: return r.pf[w][gi][xi];
            }
        };
        const auto mix_cell = [&](int ni, int kind, int w, int gi, int xi) -> PfSeries {   // unit us, layer-weighted
            PfSeries m;
            double layers = 0;
            for (size_t li = 0; li < live.size(); ++li) {
                const int of = kFormats[live[li]].layers;
                if (of == 0) continue;
                const PfSeries& s = cell(li, ni, kind, w, gi, xi);
                m.call += of * s.call;
                m.p25 += of * s.p25;
                m.p75 += of * s.p75;
                m.pf += of * s.pf;
                m.pf_p25 += of * s.pf_p25;
                m.pf_p75 += of * s.pf_p75;
                m.wake += of * s.wake;
                m.traffic += of * s.traffic;
                m.overrun += of * s.overrun;
                Pt t = s.mean;
                t.scale((double) of);
                m.mean.add(t);
                layers += of;
            }
            const double k = layers > 0 ? 1.0 / layers : 0.0;
            m.call *= k, m.p25 *= k, m.p75 *= k, m.pf *= k, m.pf_p25 *= k, m.pf_p75 *= k, m.wake *= k, m.traffic *= k, m.overrun *= k;
            m.mean.scale(k);
            return m;
        };
        const auto print_tables = [&](const char* title, double blob, const auto& at) {
            for (int ni = 0; ni < kPfNCount; ++ni) {
                const int n = kPfNs[ni];
                const PfSeries c0 = at(ni, 0, 0, 0, 0), h0 = at(ni, 1, 0, 0, 0);
                std::printf("\n## P1. %s, N=%d, T=%d, 15 workers + host: us per call of N experts, the median of %d units\n", title, n, kPfRows, units);
                std::printf("##     cold = behind a sham pass by the same readers (the same bytes from unrelated memory), then the gap, then the\n"
                            "##     call; hot = no pass, the same N experts every unit; pass = the pass's own time, its readers' wake included;\n"
                            "##     traffic = the thread's read of X MiB during the gap: its duration (its overrun past the gap's end)\n");
                std::printf("anchors, gap 0, no pass, no traffic: cold %.1f us | hot %.1f us | the call reads %.1f MB\n", c0.call, h0.call, blob * n / 1e6);
                std::printf("%5s %4s |%s |%s |%s |%s | %s\n", "G ms", "MiB", " call:  host   one   all  both", " cold:  host   one   all  both",
                            "   hot", " pass:  host   one   all  both", "traffic us (overrun)");
                for (int gi = 0; gi < kPfGCount; ++gi)
                    for (int xi = 0; xi < kPfXCount; ++xi) {
                        const PfSeries h = at(ni, 2, 0, gi, xi);
                        std::printf("%5.2f %4d |", kPfGapsMs[gi], kPfTrafficMib[xi]);
                        for (int kind = 4; kind >= 3; --kind) {              // the prefetched call, then its cold control
                            std::printf("      ");
                            for (int w = 0; w < kPfWhoCount; ++w) {
                                if (has[w]) std::printf(" %5.1f", at(ni, kind, w, gi, xi).call);
                                else std::printf("    na");
                            }
                            std::printf(" |");
                        }
                        std::printf(" %5.1f |", h.call);
                        std::printf("      ");
                        for (int w = 0; w < kPfWhoCount; ++w) {
                            if (has[w]) std::printf(" %5.1f", at(ni, 4, w, gi, xi).pf);
                            else std::printf("    na");
                        }
                        std::printf(" | %6.0f (%5.0f)\n", h.traffic, h.overrun);
                    }
                std::printf("\n## P2. %s, N=%d: the pass's rate (GB/s = bytes read / its time; both reads every line twice) and the call against\n"
                            "##     its controls: pf/cold (its own cold control: the same readers, gap and traffic) | pf/hot (no pass, the same gap\n"
                            "##     and traffic)\n", title, n);
                std::printf("%5s %4s |", "G ms", "MiB");
                for (int w = 0; w < kPfWhoCount; ++w) std::printf(" %-5s %6s %7s %7s |", kPfWhoName[w], "GB/s", "pf/cold", "pf/hot");
                std::printf("\n");
                for (int gi = 0; gi < kPfGCount; ++gi)
                    for (int xi = 0; xi < kPfXCount; ++xi) {
                        std::printf("%5.2f %4d |", kPfGapsMs[gi], kPfTrafficMib[xi]);
                        for (int w = 0; w < kPfWhoCount; ++w) {
                            if (!has[w]) {
                                std::printf("%30s|", "n/a");
                                continue;
                            }
                            const PfSeries c = at(ni, 3, w, gi, xi), p = at(ni, 4, w, gi, xi), h = at(ni, 2, 0, gi, xi);
                            const double gbs = p.pf > 0 ? blob * n * (whole[w] ? count[w] : 1) / (p.pf * 1e-6) / 1e9 : 0.0;
                            std::printf("       %6.1f %7.2f %7.2f |", gbs, p.call / c.call, p.call / h.call);
                        }
                        std::printf("\n");
                    }
                std::printf("\n## P3. %s, N=%d: the pass's wake, us (the median): the slowest reader's time from the host's signal to its first read;\n"
                            "##     the pass is the wake and the reads (P1)\n", title, n);
                std::printf("%5s %4s |%s\n", "G ms", "MiB", "   host    one    all   both");
                for (int gi = 0; gi < kPfGCount; ++gi)
                    for (int xi = 0; xi < kPfXCount; ++xi) {
                        std::printf("%5.2f %4d |", kPfGapsMs[gi], kPfTrafficMib[xi]);
                        for (int w = 0; w < kPfWhoCount; ++w) {
                            if (has[w]) std::printf(" %6.1f", at(ni, 4, w, gi, xi).wake);
                            else std::printf("     na");
                        }
                        std::printf("\n");
                    }
            }
        };
        for (size_t li = 0; li < live.size(); ++li)
            print_tables(kFormats[live[li]].name, (double) store_of(live[li]).bytes,
                         [&](int ni, int kind, int w, int gi, int xi) -> PfSeries { return cell(li, ni, kind, w, gi, xi); });
        if (native_layers > 0)
            print_tables("native mix", mean_blob,
                         [&](int ni, int kind, int w, int gi, int xi) -> PfSeries { return mix_cell(ni, kind, w, gi, xi); });

        // ---- table S: the stop rule, the best cell of each reader set, the pass against the idle time, the price
        std::printf("\n## S. the stop rule: call/cold at G = %.2f ms with %d MiB of traffic; the best cell of each reader set over G and X\n"
                    "##    (who, G ms, MiB, call/cold); the pass's own time at the stop-rule cell against the engine's idle time per layer\n"
                    "##    (250 us solo, ~320 us concurrent); the price if the stop-rule cell held for every call: (1 - r) x 0.46 x 0.715 of the\n"
                    "##    concurrent round's window\n", kPfGapsMs[kPfStopG], kPfTrafficMib[kPfStopX]);
        std::printf("%-15s %3s |%s | %-24s |%s |%s\n", "format", "N", " call/cold:  host    one    all   both", "best cell",
                    " pass us:   host    one    all   both", " price %:   host    one    all   both");
        const auto summary_rows = [&](const char* name, const auto& at) {
            for (int ni = 0; ni < kPfNCount; ++ni) {
                std::printf("%-15s %3d |", ni == 0 ? name : "", kPfNs[ni]);
                for (int w = 0; w < kPfWhoCount; ++w)
                    if (has[w]) std::printf(" %6.2f", at(ni, 4, w, kPfStopG, kPfStopX).call / at(ni, 3, w, kPfStopG, kPfStopX).call);
                int bg = 0, bx = 0, bw = 0;
                double best = 1e9;
                for (int w = 0; w < kPfWhoCount; ++w) {
                    if (!has[w]) continue;
                    for (int gi = 0; gi < kPfGCount; ++gi)
                        for (int xi = 0; xi < kPfXCount; ++xi) {
                            const double r = at(ni, 4, w, gi, xi).call / at(ni, 3, w, gi, xi).call;
                            if (r < best) best = r, bg = gi, bx = xi, bw = w;
                        }
                }
                std::printf(" | %-5s %.2f %2d MiB %5.2f |", kPfWhoName[bw], kPfGapsMs[bg], kPfTrafficMib[bx], best);
                for (int w = 0; w < kPfWhoCount; ++w)
                    if (has[w]) std::printf(" %6.1f", at(ni, 4, w, kPfStopG, kPfStopX).pf);
                for (int w = 0; w < kPfWhoCount; ++w)
                    if (has[w])
                        std::printf(" %+6.1f", 100.0 * (1.0 - at(ni, 4, w, kPfStopG, kPfStopX).call / at(ni, 3, w, kPfStopG, kPfStopX).call) * 0.46 * 0.715);
                std::printf("\n");
            }
        };
        for (size_t li = 0; li < live.size(); ++li)
            summary_rows(kFormats[live[li]].name,
                         [&](int ni, int kind, int w, int gi, int xi) -> PfSeries { return cell(li, ni, kind, w, gi, xi); });
        if (native_layers > 0)
            summary_rows("native mix",
                         [&](int ni, int kind, int w, int gi, int xi) -> PfSeries { return mix_cell(ni, kind, w, gi, xi); });
        std::printf("END pool_bench\n");
        return 0;
    }

    // ---- tables 1-3: the standing park environment, no gap
    const double load_start = loadavg();
    std::vector<Unit> work((size_t) live.size());     // one unit per format: each keeps its own cold cursor
    for (const int workers : {15, 8, 4}) {
        const std::unique_ptr<cpu::ExpertPool> pool = make_pool(workers);   // built unpinned: the CPU layout is whole
        const HostPin host_pin(host);                                       // then the host takes its core
        if (workers == 15)
            std::printf("pool: %d workers + the draining host, %d tasks per phase, repark waits %s\n", pool->workers(),
                        3 * (pool->workers() + 1), env_on("STRATA_POOL_DIET") ? "skipped (diet)" : "kept");
        for (size_t i = 0; i < live.size(); ++i) {
            const int f = live[i];
            const Store& st = store_of(f);
            Unit& u = work[i];
            if (workers != 15) {                                   // table 2
                for (int h = 0; h < 2; ++h)
                    res[f].small[h][workers == 8 ? 0 : 1] = run_point(u, st, pool.get(), 8, 12, h != 0, 0, true);
                continue;
            }
            for (int h = 0; h < 2; ++h) {
                for (int ti = 0; ti < kTCount; ++ti)
                    for (int ni = 0; ni < kNCount; ++ni)
                        res[f].tn[h][ti][ni] = run_point(u, st, pool.get(), kNs[ni], kTs[ti], h != 0, 0, true);
                for (int ti = 0; ti < kTCount; ++ti)               // one thread, no pool: one call up to MAXT rows
                    res[f].direct_t[h][ti] = run_point(u, st, nullptr, 8, kTs[ti], h != 0, 0, true);
            }
        }
    }

    // ---- table 4: the park environment and the host's cadence (cold, T=12, N=8, 15 workers + host)
    struct Arm {
        const char *spin_us, *diet;
        int gap_us, slot;
    };
    const Arm kArms[] = {
        {"100", "1", 150, 0}, {"100", "1", 2000, 1}, {"20000", "1", 150, 2}, {"100", "0", 150, 3},
    };
    for (const Arm& arm : kArms) {
        set_env("STRATA_POOL_SPIN_US", arm.spin_us);
        set_env("STRATA_POOL_DIET", arm.diet);
        const std::unique_ptr<cpu::ExpertPool> pool = make_pool(15);
        const HostPin host_pin(host);
        for (size_t i = 0; i < live.size(); ++i)
            res[live[i]].sync[arm.slot] = run_point(work[i], store_of(live[i]), pool.get(), 8, 12, false, arm.gap_us, true);
    }
    set_env("STRATA_POOL_SPIN_US", "100");
    set_env("STRATA_POOL_DIET", "1");
    std::printf("\nload: the kernel's 1-minute average %.2f when the measured phase began, %.2f when it ended (a run that built its\n"
                "     experts carries their generation in the first; POOL_BENCH_FIXTURES=<dir> keeps them for the next run)\n",
                load_start, loadavg());

    // ---- table 1: residency
    std::printf("\n## 1. residency: T=12 (6+6, two calls) N=8, 15 workers + host, gap 0\n"
                "##    us per expert-call: median of %d units [p25..p75]; phases are means [gu q down]: gate/up rows, host-serial\n"
                "##    quantization, down rows.  GB/s = expert bytes / time over all threads; Gw/s/thr = weights decoded per second\n"
                "##    per thread (%d threads)\n", units, kThreads);
    std::printf("%-15s %6s %9s | %-24s%-22s%-15s | %-24s%-22s%-15s | %s\n", "format", "layers", "blob B",
                "cold: us/exp", "[gu q down]   (mean)", "GB/s Gw/s/thr", "hot: us/exp", "[gu q down]   (mean)",
                "GB/s Gw/s/thr", "cold/hot");
    const auto cell1 = [&](double blob, const Series& s) {
        std::printf("%7.1f [%6.1f..%-6.1f] ", per_expert(s.median.us, 8), per_expert(s.p25, 8), per_expert(s.p75, 8));
        print_phases(s.mean, 8);
        print_rates(per_expert(s.median.us, 8), blob, 2, kThreads);
    };
    for (int f : live) {
        const Series &c = res[f].tn[0][kT12][kN8], &h = res[f].tn[1][kT12][kN8];
        std::printf("%-15s %6d %9zu | ", kFormats[f].name, kFormats[f].layers, store_of(f).bytes);
        cell1((double) store_of(f).bytes, c);
        std::printf(" | ");
        cell1((double) store_of(f).bytes, h);
        std::printf(" | %.2f\n", per_expert(c.median.us, 8) / per_expert(h.median.us, 8));
    }
    // The pack's native mix of a point: layer-weighted UNIT us and phases - a table's cells divide by the unit's
    // experts, so this must not (its first form did both and printed the mix 8x small).
    const auto mix_series = [&](const auto& pick) {
        Series m;
        for (int f : live) {
            if (kFormats[f].layers == 0) continue;
            const Series& s = pick(f);
            Pt p = s.mean;
            p.us = s.median.us;
            p.scale((double) kFormats[f].layers / native_layers);
            m.median.us += p.us;
            m.mean.add(p);
        }
        return m;
    };
    if (native_layers > 0) {
        const Series mc = mix_series([&](int f) -> const Series& { return res[f].tn[0][kT12][kN8]; });
        const Series mh = mix_series([&](int f) -> const Series& { return res[f].tn[1][kT12][kN8]; });
        std::printf("%-15s %6d %9.0f | ", "native mix", native_layers, mean_blob);
        std::printf("%7.1f                ", per_expert(mc.median.us, 8));   // a mix has no quartiles
        print_phases(mc.mean, 8);
        print_rates(per_expert(mc.median.us, 8), mean_blob, 2, kThreads);
        std::printf(" | %7.1f                ", per_expert(mh.median.us, 8));
        print_phases(mh.mean, 8);
        print_rates(per_expert(mh.median.us, 8), mean_blob, 2, kThreads);
        std::printf(" | %.2f  (layer-weighted over the %d native formats' %d layers)\n",
                    per_expert(mc.median.us, 8) / per_expert(mh.median.us, 8), kFormatCount - 1, native_layers);
    }

    // ---- table 2: threads
    std::printf("\n## 2. threads: T=12 (6+6) N=8, gap 0.  threads = pool workers + the draining host; \"1 direct\" = one thread\n"
                "##    calling the row kernels, no pool (the per-core rate).  us per expert-call, cold | hot\n");
    std::printf("%-15s %8s | %-9s%-15s | %-9s%-15s\n", "format", "threads", "cold us/e", "GB/s Gw/s/thr", "hot us/e",
                "GB/s Gw/s/thr");
    const auto row2 = [&](const char* name, double blob, const Series& c, const Series& h, int threads) {
        const std::string label = threads == 1 ? std::string("1 direct") : std::to_string(threads);
        std::printf("%-15s %8s | %9.1f", name, label.c_str(), per_expert(c.median.us, 8));
        print_rates(per_expert(c.median.us, 8), blob, 2, threads);
        std::printf(" | %9.1f", per_expert(h.median.us, 8));
        print_rates(per_expert(h.median.us, 8), blob, 2, threads);
        std::printf("\n");
    };
    for (int f : live) {
        const double blob = (double) store_of(f).bytes;
        row2(kFormats[f].name, blob, res[f].direct_t[0][kT12], res[f].direct_t[1][kT12], 1);
        row2("", blob, res[f].small[0][1], res[f].small[1][1], 5);
        row2("", blob, res[f].small[0][0], res[f].small[1][0], 9);
        row2("", blob, res[f].tn[0][kT12][kN8], res[f].tn[1][kT12][kN8], 16);
    }
    if (native_layers > 0) {
        // arm 0: one thread, no pool; 1: 4 workers; 2: 8 workers; 3: 15 workers (each pool + the host).  Unit us,
        // layer-weighted; row2 divides by the unit's 8 experts.
        const auto mix_arm = [&](int arm, int h) {
            Series m;
            for (int f : live) {
                if (kFormats[f].layers == 0) continue;
                const Series& s = arm == 0 ? res[f].direct_t[h][kT12] : arm == 3 ? res[f].tn[h][kT12][kN8]
                                         : res[f].small[h][arm == 2 ? 0 : 1];
                m.median.us += (double) kFormats[f].layers * s.median.us;
            }
            m.median.us /= native_layers;
            return m;
        };
        row2("native mix", mean_blob, mix_arm(0, 0), mix_arm(0, 1), 1);
        row2("", mean_blob, mix_arm(1, 0), mix_arm(1, 1), 5);
        row2("", mean_blob, mix_arm(2, 0), mix_arm(2, 1), 9);
        row2("", mean_blob, mix_arm(3, 0), mix_arm(3, 1), 16);
    }
    std::printf("\n## 2b. one thread, no pool, N=8: us per expert - T <= 8 in ONE call; T = 12, 16 as the kernels require (6+6, 8+8)\n");
    std::printf("%-15s %8s %8s %8s %8s %8s %8s  |%8s %8s %8s %8s %8s %8s\n", "format", "T=1", "T=4", "T=6", "T=8", "T=12",
                "T=16", "T=1", "T=4", "T=6", "T=8", "T=12", "T=16");
    for (int f : live) {
        std::printf("%-15s", kFormats[f].name);
        for (int h = 0; h < 2; ++h) {
            for (int ti = 0; ti < kTCount; ++ti) std::printf(" %8.1f", per_expert(res[f].direct_t[h][ti].median.us, 8));
            if (h == 0) std::printf("  |");
        }
        std::printf("\n");
    }
    if (native_layers > 0) {
        std::printf("%-15s", "native mix");
        for (int h = 0; h < 2; ++h) {
            for (int ti = 0; ti < kTCount; ++ti) {
                const Series m = mix_series([&](int f) -> const Series& { return res[f].direct_t[h][ti]; });
                std::printf(" %8.1f", per_expert(m.median.us, 8));
            }
            if (h == 0) std::printf("  |");
        }
        std::printf("\n");
    }

    // ---- table 3: rows x experts
    for (int h = 0; h < 2; ++h) {
        std::printf("\n## 3%s. rows x experts, 15 workers + host, gap 0: us per expert-call, median - %s\n", h == 0 ? "a" : "b",
                    h == 0 ? "COLD (every expert's weights from DRAM)" : "HOT (the same experts every unit: resident only while they fit every CCD's L3)");
        std::printf("%-15s %-20s %8s %8s %8s %8s %8s\n", "format", "T \\ N", "1", "4", "8", "16", "32");
        const auto rows3 = [&](const char* name, const auto& pick) {
            for (int ti = 0; ti < kTCount; ++ti) {
                std::printf("%-15s %-20s", ti == 0 ? name : "", t_label(ti).c_str());
                for (int ni = 0; ni < kNCount; ++ni) std::printf(" %8.1f", per_expert(pick(ti, ni), kNs[ni]));
                std::printf("\n");
            }
        };
        for (int f : live)
            rows3(kFormats[f].name, [&](int ti, int ni) { return res[f].tn[h][ti][ni].median.us; });
        if (native_layers > 0)
            rows3("native mix", [&](int ti, int ni) {
                double v = 0;
                for (int f : live)
                    if (kFormats[f].layers > 0) v += (double) kFormats[f].layers * res[f].tn[h][ti][ni].median.us;
                return v / native_layers;
            });
    }
    std::printf("\n## 3c. N=8, 15 workers + host: us per expert-call (median) [gu q down] (means) against T - cold | hot\n%-15s",
                "format");
    for (int ti = 0; ti < kTCount; ++ti) std::printf(" %26s", ("T=" + std::to_string(kTs[ti])).c_str());
    std::printf("\n");
    for (int f : live)
        for (int h = 0; h < 2; ++h) {
            std::printf("%-15s", h == 0 ? kFormats[f].name : "");
            for (int ti = 0; ti < kTCount; ++ti) {
                std::printf(" ");
                std::printf("%7.1f ", per_expert(res[f].tn[h][ti][kN8].median.us, 8));
                print_phases(res[f].tn[h][ti][kN8].mean, 8);
            }
            std::printf("%s\n", h == 0 ? "  |" : "");
        }

    // ---- table 3d: what a row costs against what a weight costs
    std::printf("\n## 3d. least squares over the one-call points T=1,4,6,8 at N=8: us per expert-call = a + b*T\n"
                "##     a: the per-weight and per-call part, b: the per-row part (us per expert-call per row)\n"
                "%-15s | %-24s | %-24s\n", "format", "cold:  a      b     R^2", "hot:   a      b     R^2");
    const auto fit = [&](const auto& at) {      // at(ti): the point of T = kTs[ti] at N = 8
        double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        const int m = kT12;                                                 // kTs[0..3] = 1, 4, 6, 8
        for (int i = 0; i < m; ++i) {
            const double x = kTs[i], y = per_expert(at(i).median.us, 8);
            sx += x, sy += y, sxx += x * x, sxy += x * y, syy += y * y;
        }
        const double b = (m * sxy - sx * sy) / (m * sxx - sx * sx), a = (sy - b * sx) / m;
        double ss_res = 0;
        for (int i = 0; i < m; ++i) {
            const double e = per_expert(at(i).median.us, 8) - (a + b * kTs[i]);
            ss_res += e * e;
        }
        const double ss_tot = syy - sy * sy / m;
        return Fit{a, b, ss_tot > 0 ? 1.0 - ss_res / ss_tot : 1.0};
    };
    for (int f : live) {
        const auto at = [&](int h) { return [&, h](int ti) -> const Series& { return res[f].tn[h][ti][kN8]; }; };
        const auto at_direct = [&](int h) { return [&, h](int ti) -> const Series& { return res[f].direct_t[h][ti]; }; };
        const Fit a = fit(at(0)), b = fit(at(1)), d0 = fit(at_direct(0)), d1 = fit(at_direct(1));
        std::printf("%-15s | %8.1f %6.2f %6.3f | %8.1f %6.2f %6.3f  | one thread, no pool: cold a %.1f b %.2f, hot a %.1f b %.2f\n",
                    kFormats[f].name, a.a, a.b, a.r2, b.a, b.b, b.r2, d0.a, d0.b, d1.a, d1.b);
    }

    // ---- table 4: the pool's own synchronization
    std::printf("\n## 4. the pool's own synchronization: T=12 (6+6) N=8 COLD, 15 workers + host.  gap = the host's idle before every\n"
                "##    call; spin = STRATA_POOL_SPIN_US (a worker parks after that long without work); diet = STRATA_POOL_DIET\n"
                "##    (the trailing repark waits skipped).  us per expert-call (median) [gu q down] (means)\n");
    std::printf("%-15s | %-38s | %-38s | %-38s | %-38s | %-38s\n", "format",
                "gap 0    spin 100   diet 1", "gap 150  spin 100   diet 1", "gap 2000 spin 100   diet 1",
                "gap 150  spin 20000 diet 1", "gap 150  spin 100   diet 0");
    const auto cell4 = [&](const Series& s) {
        std::printf(" %7.1f ", per_expert(s.median.us, 8));
        print_phases(s.mean, 8);
        std::printf("|");
    };
    const auto mix4 = [&](int s) {
        return mix_series([&](int f) -> const Series& { return s < 0 ? res[f].tn[0][kT12][kN8] : res[f].sync[s]; });
    };
    for (int f : live) {
        std::printf("%-15s |", kFormats[f].name);
        cell4(res[f].tn[0][kT12][kN8]);
        for (int s = 0; s < 4; ++s) cell4(res[f].sync[s]);
        std::printf("\n");
    }
    if (native_layers > 0) {
        std::printf("%-15s |", "native mix");
        cell4(mix4(-1));
        for (int s = 0; s < 4; ++s) cell4(mix4(s));
        std::printf("\n");
    }
    std::printf("END pool_bench\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) { return run(argc, argv); }
