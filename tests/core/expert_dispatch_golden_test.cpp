// The Golden Master for BOTH expert dispatch entry points, neither of which any test executed before
// F-438: `expert_pool_dispatch_multi` (the verify-window path, generate.cpp:1250, its only production
// caller) and `expert_pool_dispatch` (the single-token decode lane).  F-432..F-436 split the multi
// god-function into dispatch_begin / plan_distinct_groups / plan_build / plan_admit_fetch /
// cpu_miss_pipeline; this test freezes the seam those helpers form on the real CPU path, with a
// synthetic canonical pack:
//
//   - the fingerprint (FNV-1a 64) is over the full `out` window after a fixed 8-token x 10-expert
//     window with fixed activations, fixed routing (two resident experts, five misses, duplicates in
//     the router's order) and a counting GpuPlanSink;
//   - the counters, the plan's groups/entries/starts/ptrs and the publish/fetch call counts are
//     asserted by value, not by hash, so a failure names the field that moved;
//   - the refusal paths (window too large, a pre-failed dispatch, a routed id out of range) freeze
//     the exact `d.fail` strings - error states are part of the contract.
//
// Determinism pins: STRATA_COLD_OVERLAP=0 (the file default would arm the async cold-overlap batches
// whose landing order is timing) and STRATA_ROUTE_TAIL_SKIP=0 (the serve start-up's CUDA default 7
// changes the output by design).  Both are read once per process, so the pin must precede the first
// call.  The fingerprint is machine-pinned (VNNI arithmetic): a box with a different ISA may print a
// different constant - re-record there; the A/B against the pre-refactor binary is the oracle.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;
using namespace strata::kernels::cpu;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        path = fs::temp_directory_path() / ("strata-dispatch-golden-" +
                                            std::to_string(std::random_device{}()));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// A deterministic byte pattern per blob: identical on every run, distinct per (layer, expert), so a
// dispatch that reads the wrong expert's bytes moves the fingerprint.
void write_blob_pattern(const fs::path& file, int64_t layer, int32_t expert) {
    const uint64_t off = (uint64_t) ((size_t) layer * NE + (size_t) expert) * BLOB;
    std::fstream out(file, std::ios::binary | std::ios::in | std::ios::out);
    require((bool) out, "could not reopen the synthetic pack for a pattern");
    out.seekp((std::streamoff) off);
    uint32_t lcg = 0x9e3779b9u ^ (uint32_t) (layer * 31 + expert * 2246822519u);
    std::vector<char> buf((size_t) BLOB);
    for (char& b : buf) {
        lcg = lcg * 1664525u + 1013904223u;
        b = (char) ((lcg >> 24) & 0x3f);   // keeps the S2 scales tame: finite floats only
    }
    out.write(buf.data(), (std::streamoff) buf.size());
    require((bool) out, "could not write the synthetic expert pattern");
}

uint64_t fnv1a(const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

struct SinkState {
    int publish_calls = 0;
    int fetch_calls = 0;
    int fetch_entries = -1;
};

void sink_publish(void* ctx) { ++((SinkState*) ctx)->publish_calls; }
void sink_fetch(void* ctx, const uint8_t* const*, int n, size_t) {
    SinkState* s = (SinkState*) ctx;
    ++s->fetch_calls;
    s->fetch_entries = n;
}

// The window under test: 8 tokens (MAXT) x the model's 10 routed experts, each token a rotation of
// one permutation of ten DISTINCT experts (a real router never repeats an expert inside a token, and
// the job tables' MAXT bound is exactly "once per token").  Experts 0 and 1 are resident (host_res
// slots 0 and 1); 2, 3, 5, 7, 9, 11, 13, 15 are misses.  The rotation varies each token's
// first-occurrence order, exercising the group linking.
constexpr int64_t kTokens = 8;
constexpr int64_t kTop = 10;
constexpr int32_t kPattern[kTop] = {5, 0, 7, 1, 9, 3, 2, 11, 13, 15};

struct Window {
    std::vector<int32_t> ids;
    int resident_entries = 0;
    int cpu_entries = 0;
    Window() : ids((size_t) (kTokens * kTop)) {
        for (int64_t t = 0; t < kTokens; ++t)
            for (int64_t j = 0; j < kTop; ++j) {
                const int32_t e = kPattern[(size_t) ((j + t) % kTop)];
                ids[(size_t) (t * kTop + j)] = e;
                if (e == 0 || e == 1) ++resident_entries;
                else ++cpu_entries;
            }
    }
};

// One dispatchable run: fresh ExpertDispatch, fresh plan arrays, sentinel-filled output.
struct Run {
    ExpertDispatch d;
    GpuPlanSink sink;
    SinkState st;
    std::vector<int32_t> counts, start, dst, tok, start2;
    std::vector<unsigned long long> ptr, ptr2, dst2;
    std::vector<float> out;

    Run(ExpertPool& pool, FileExpertSource& src, const std::vector<int32_t>& host_res,
        std::vector<float>&& out_buf)
        : counts(3, -1), start(129, -1), dst(128, -1), tok(128, -1), start2(8, -1),
          ptr(128, 0), ptr2(4, 0), dst2(4, 0), out(std::move(out_buf)) {
        sink.counts = counts.data();
        sink.start = start.data();
        sink.dst = dst.data();
        sink.tok = tok.data();
        sink.ptr = ptr.data();
        sink.start2 = start2.data();
        sink.ptr2 = ptr2.data();
        sink.dst2 = dst2.data();
        sink.staging = 0xfeed00000000ull;
        sink.staging_cap = 4;
        sink.cap = 128;   // kMaxWindowEntries (expert_source.cpp): MAXT * 10 fits
        sink.publish = sink_publish;
        sink.fetch = sink_fetch;
        sink.ctx = &st;
        d.pool = &pool;
        d.src = &src;
        d.plan = &sink;
        d.host_res = host_res.data();
        d.cache_blob = (int64_t) BLOB;   // the slot stride for the resident groups' ptr values
    }
};

// The frozen fingerprint, recorded 2026-11-20 on the campaign box (RTX 5090 host, VNNI) and proven
// identical on the pre-extraction binary e0f4c57bc and the post-F-436 tree: the A/B that certified
// the F-432..F-436 decomposition by execution.  A different ISA may legitimately print a different
// constant - re-record it there and note the box; the determinism and value assertions still hold.
constexpr uint64_t kGoldenHash = 0xc90ded77662c664dull;

// The single-token (decode-lane) dispatch's fingerprint, recorded the same day on 37e2d2cfe.  That
// function is byte-identical before and after this session's extractions (verified by extraction and
// diff against e0f4c57bc), so the pin's provenance is textual - no A/B rebuild was needed.
constexpr uint64_t kSingleGoldenHash = 0xf37183119d4d14faull;

// The no-plan arm (plan == nullptr: a session without a GPU plan) takes plan_build's fallback kind
// scan, and because gpu_zeroes_hits is false there the HOST zeroes the resident rows itself - the
// row-ownership split inverts relative to the plan arm.  Recorded the same day, same provenance.
constexpr uint64_t kNoPlanGoldenHash = 0xec8d5c9cdddc564dull;

std::vector<float> sentinel_window() {
    return std::vector<float>((size_t) (kTokens * kTop) * (size_t) H, -1.2345e33f);
}

}  // namespace

int main() {
    // Read once per process: pin both before the first dispatch call.
#if defined(_WIN32)
    _putenv_s("STRATA_COLD_OVERLAP", "0");
    _putenv_s("STRATA_ROUTE_TAIL_SKIP", "0");
#else
    setenv("STRATA_COLD_OVERLAP", "0", 1);
    setenv("STRATA_ROUTE_TAIL_SKIP", "0", 1);
#endif

    const strata::kernels::cpu::CpuFeatures feat = strata::kernels::cpu::cpu_features();
    if (!feat.usable()) {
        std::printf("  CPU lacks %s - the VNNI path cannot run here; dispatch golden SKIPPED\n",
                    feat.reason());
        std::printf("\ndispatch golden: SKIPPED\n");
        return 0;
    }

    // ---- the synthetic canonical pack: 2 layers x 512 experts, sparse except the routed blobs
    TempDirectory dir;
    constexpr int64_t kLayers = 2;
    const uint64_t total = (uint64_t) kLayers * (uint64_t) NE * (uint64_t) BLOB;
    {
        const fs::path file = dir.path / "experts.bin";
        {
            std::ofstream out(file, std::ios::binary | std::ios::trunc);
            require((bool) out, "could not create the synthetic pack");
            out.seekp((std::streamoff) (total - 1));
            out.put('\0');
        }
        for (int32_t e : {0, 1, 2, 3, 5, 7, 9, 11, 13, 15}) write_blob_pattern(file, 0, e);
        write_blob_pattern(file, 1, 5);   // layer 1's copy differs: a cross-layer read moves the hash
    }
    std::string err;
    require(expert_layout_load(dir.path.string(), kLayers, NE, err), "layout: " + err);
    FileExpertSource src;
    require(src.open(dir.path.string(), kLayers, NE, err), "open: " + err);

    // ---- fixed activations, fixed routing, two residents on layer 0
    const Window w;
    require(w.resident_entries + w.cpu_entries == kTokens * kTop, "window accounting is wrong");
    std::mt19937 rng(0x5eed1a7eu);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x_f((size_t) kTokens * (size_t) H);
    for (float& v : x_f) v = gauss(rng);
    std::vector<int32_t> host_res((size_t) (kLayers * NE), -1);
    host_res[0] = 0;
    host_res[1] = 1;

    ExpertPool pool;

    // ---- deterministic twice: identical inputs, identical bits
    Run a(pool, src, host_res, sentinel_window());
    expert_pool_dispatch_multi(a.d, x_f.data(), w.ids.data(), kTokens, kTop, a.out.data());
    Run b(pool, src, host_res, sentinel_window());
    expert_pool_dispatch_multi(b.d, x_f.data(), w.ids.data(), kTokens, kTop, b.out.data());

    require(!a.d.failed, std::string("dispatch failed: ") + (a.d.fail ? a.d.fail : "?"));
    require(!b.d.failed, "the second dispatch failed");
    const uint64_t h1 = fnv1a(a.out.data(), a.out.size() * sizeof(float));
    const uint64_t h2 = fnv1a(b.out.data(), b.out.size() * sizeof(float));
    require(h1 == h2, "two identical windows produced different bits");
    require(h1 == kGoldenHash,
            "the dispatch fingerprint moved (or this is a new ISA: re-record kGoldenHash and note "
            "the box)");

    // ---- the plan: experts 0 and 1 are the only resident groups, first-occurrence order
    require(a.st.publish_calls == 1, "publish was not called exactly once");
    require(a.st.fetch_calls == 1 && a.st.fetch_entries == 0, "the copy engine saw a nonzero fetch");
    require(a.counts[0] == 2, "resident group count is wrong");
    require(a.counts[1] == w.resident_entries, "resident entry count is wrong");
    require(a.counts[2] == 0, "an unfetchable miss was staged");
    require(a.start[0] == 0 && a.start[2] == a.counts[1], "the plan's start table is inconsistent");
    // cache_base is null in this harness, so a resident group's ptr is its slot's byte offset.
    require(a.ptr[0] == 0 && a.ptr[1] == (unsigned long long) BLOB,
            "the resident groups' blob addresses are wrong");

    // ---- the rows: GPU-owned entries keep the sentinel (dec_batch zeroes them on the device, not
    // here); CPU entries were computed by the pool
    int sentinel_rows = 0, computed_rows = 0;
    for (int64_t i = 0; i < kTokens * kTop; ++i) {
        const float* row = a.out.data() + (size_t) i * (size_t) H;
        const bool untouched = row[0] == -1.2345e33f && row[(size_t) H - 1] == -1.2345e33f;
        const int32_t e = w.ids[(size_t) i];
        if (e == 0 || e == 1) {
            if (untouched) ++sentinel_rows;
        } else if (!untouched)
            ++computed_rows;
    }
    require(sentinel_rows == w.resident_entries, "the host touched a GPU-owned row");
    require(computed_rows == w.cpu_entries, "a CPU entry was never computed");

    // ---- the counters
    require(a.d.experts == kTokens * kTop, "the expert counter is wrong");
    require(a.d.multi_misses == 8, "the distinct-CPU-expert count is wrong");
    require(a.d.cache_refused == w.cpu_entries, "the refused counter is wrong");

    // ---- the refusal paths (exact strings frozen)
    {
        Run f(pool, src, host_res, sentinel_window());
        expert_pool_dispatch_multi(f.d, x_f.data(), w.ids.data(), MAXT + 1, kTop, f.out.data());
        require(f.d.failed &&
                std::strcmp(f.d.fail, "a verify window has more tokens than the multi-token "
                                      "expert kernel takes") == 0,
                "the oversized window was not refused with its string");
    }
    {
        Run f(pool, src, host_res, sentinel_window());
        f.d.failed = true;   // a previous layer already failed: the dispatch must do nothing
        const int64_t before = f.d.experts;
        expert_pool_dispatch_multi(f.d, x_f.data(), w.ids.data(), kTokens, kTop, f.out.data());
        require(f.d.experts == before && f.d.cache_refused == 0, "a pre-failed dispatch did work");
    }
    {
        Run f(pool, src, host_res, sentinel_window());
        std::vector<int32_t> bad = w.ids;
        bad[3] = 999;
        expert_pool_dispatch_multi(f.d, x_f.data(), bad.data(), kTokens, kTop, f.out.data());
        require(f.d.failed && std::strcmp(f.d.fail, "a routed expert id is out of range") == 0 &&
                    f.d.fail_expert == 999,
                "the out-of-range id was not refused with its string");
    }

    // ---- the single-token (decode-lane) dispatch: graph-hits mode (host_res wired), residents
    // zeroed by the host and owned by the GPU, misses computed into jobs
    {
        Run f(pool, src, host_res, sentinel_window());
        expert_pool_dispatch((void*) &f.d, x_f.data(), w.ids.data(), nullptr, H, kTop, f.out.data());
        require(!f.d.failed, std::string("single-token dispatch failed: ") +
                                 (f.d.fail ? f.d.fail : "?"));
        int zeroed = 0, computed = 0;
        for (int64_t i = 0; i < kTop; ++i) {
            const float* row = f.out.data() + (size_t) i * (size_t) H;
            const bool zero = row[0] == 0.0f && row[(size_t) H - 1] == 0.0f;
            const bool untouched = row[0] == -1.2345e33f && row[(size_t) H - 1] == -1.2345e33f;
            const int32_t e = w.ids[(size_t) i];
            if (e == 0 || e == 1) {
                if (zero) ++zeroed;
            } else if (!untouched)
                ++computed;
        }
        require(zeroed == 2, "the host did not zero the GPU-owned rows");
        require(computed == kTop - 2, "a single-token miss was never computed");
        require(f.d.cache_hits == 2, "the hit counter is wrong");
        require(f.d.cache_refused == kTop - 2, "the refused counter is wrong");
        require(f.d.experts == kTop && f.d.layers == 1, "the single-token counters are wrong");
        const uint64_t hs = fnv1a(f.out.data(), f.out.size() * sizeof(float));
        std::fprintf(stderr, "  dispatch golden (single token): hash %016llx, 2 rows zeroed, %d computed\n",
                     (unsigned long long) hs, computed);
        std::fflush(nullptr);
        require(hs == kSingleGoldenHash,
                "the single-token fingerprint moved (or this is a new ISA: re-record "
                "kSingleGoldenHash and note the box)");
    }

    // ---- the no-plan arm: plan == nullptr takes plan_build's fallback kind scan, and the HOST
    // zeroes the resident rows (gpu_zeroes_hits is false without a plan)
    {
        Run f(pool, src, host_res, sentinel_window());
        f.d.plan = nullptr;
        expert_pool_dispatch_multi(f.d, x_f.data(), w.ids.data(), kTokens, kTop, f.out.data());
        require(!f.d.failed, std::string("no-plan dispatch failed: ") +
                                 (f.d.fail ? f.d.fail : "?"));
        int zeroed = 0, computed = 0;
        for (int64_t i = 0; i < kTokens * kTop; ++i) {
            const float* row = f.out.data() + (size_t) i * (size_t) H;
            const bool zero = row[0] == 0.0f && row[(size_t) H - 1] == 0.0f;
            const bool untouched = row[0] == -1.2345e33f && row[(size_t) H - 1] == -1.2345e33f;
            const int32_t e = w.ids[(size_t) i];
            if (e == 0 || e == 1) {
                if (zero) ++zeroed;
            } else if (!untouched)
                ++computed;
        }
        require(zeroed == w.resident_entries, "the host did not zero the residents without a plan");
        require(computed == w.cpu_entries, "a no-plan CPU entry was never computed");
        require(f.d.cache_refused == w.cpu_entries && f.d.multi_misses == 8,
                "the no-plan counters are wrong");
        const uint64_t hn = fnv1a(f.out.data(), f.out.size() * sizeof(float));
        std::fprintf(stderr, "  dispatch golden (no plan): hash %016llx, %d rows zeroed, %d computed\n",
                     (unsigned long long) hn, zeroed, computed);
        std::fflush(nullptr);
        require(hn == kNoPlanGoldenHash,
                "the no-plan fingerprint moved (or this is a new ISA: re-record kNoPlanGoldenHash)");
    }

    std::printf("  dispatch golden: hash %016llx, %d resident rows kept, %d CPU rows computed\n",
                (unsigned long long) h1, sentinel_rows, computed_rows);
    std::printf("\ndispatch golden: PASS (hash %016llx)\n", (unsigned long long) h1);
    return 0;
}
