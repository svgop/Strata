// src/kernels/sampler_fastk_parity.cpp - F-453: the split sampler's one-pass warp top-k (STRATA_SAMPLER_FASTK) against the
// k argmax rounds it can replace, LIST FOR LIST.
//
// sampler_parity judges the sampled TOKEN (top_k -> top_p -> min_p -> temperature -> draw against a host reference); a
// token is the tail's pick, so it sees the top_k list only through top_p's cut.  The claim here is stronger - the two
// selections return the same list, entry for entry - and it is checked that way: both kernels run on the same row and
// every block's candidate list is compared with memcmp (not a tolerance).
//
// The fixtures put the order's corner cases INTO the lists on purpose, each placed where the top-k will reach it:
//   - -inf and NaN (never candidates: the rounds' first round admits every value below +inf, and `> -inf` fails for
//     both), +inf (a candidate, first in the order);
//   - -0.0 beside +0.0 (the rounds compare them EQUAL, so the smaller id leads; a key built from the raw bits would
//     rank +0.0 above -0.0 whatever the ids);
//   - duplicate values (ties by id, across lanes and across the four warps of a block);
//   - a vocabulary that is not a multiple of the 4,096-logit block (the block's tail is -inf padding);
//   - fewer candidates than k (the list ends in sentinels).
// Before the kernel is judged, the comparison is shown to see a one-entry difference.
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr float kNegInf = -__builtin_huge_valf();
constexpr float kNaN = __builtin_nanf("");

// the largest block count x k the fixtures reach: 38 blocks x 64 entries
constexpr int kMaxBlocks = 64, kK = 64;

// runs the row through both selections and returns the number of mismatched entries over every block's list
int mismatched_entries(const float* d_logits, int nv, int k, int2* d_rounds, int2* d_fast, std::vector<int2>& rounds,
                       std::vector<int2>& fast) {
    const int nb = strata::kernels::sampler_split_part_both_for_test(d_logits, nv, k, d_rounds, d_fast, 3, nullptr);
    check(cudaGetLastError(), "sampler_split_part_both_for_test");
    check(cudaDeviceSynchronize(), "sampler_split_part_both_for_test sync");
    rounds.resize((size_t) nb * k);
    fast.resize((size_t) nb * k);
    check(cudaMemcpy(rounds.data(), d_rounds, rounds.size() * sizeof(int2), cudaMemcpyDeviceToHost), "rounds down");
    check(cudaMemcpy(fast.data(), d_fast, fast.size() * sizeof(int2), cudaMemcpyDeviceToHost), "fast down");
    int bad = 0;
    for (size_t e = 0; e < rounds.size(); ++e)
        if (std::memcmp(&rounds[e], &fast[e], sizeof(int2)) != 0) ++bad;
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else if (std::string(argv[i]) == "--bench") bench = true;
        else { std::fprintf(stderr, "usage: sampler_fastk_parity [--selftest] [--bench]\n"); return 2; }
    }
    int dev = 0;
    check(cudaGetDevice(&dev), "cudaGetDevice");
    int major = 0, minor = 0;
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc major");
    check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev), "cc minor");
    std::printf("sampler_fastk_parity: device %d (compute capability %d.%d)\n", dev, major, minor);
    const int nv_max = 151936;   // the artifact's vocabulary
    float* d_logits = nullptr;
    int2 *d_rounds = nullptr, *d_fast = nullptr;
    check(cudaMalloc(&d_logits, (size_t) nv_max * sizeof(float)), "logits");
    check(cudaMalloc(&d_rounds, (size_t) kMaxBlocks * kK * sizeof(int2)), "rounds list");
    check(cudaMalloc(&d_fast, (size_t) kMaxBlocks * kK * sizeof(int2)), "fast list");
    std::vector<int2> rounds, fast;
    int bad = 0, cases = 0;

    if (selftest || !bench) {
        std::mt19937 rng(20261125u);
        std::normal_distribution<float> gauss(0.0f, 4.0f);
        const int vocabs[] = {4096, 8209, 40960, 151936};        // one block; a ragged tail; the drafter's; the artifact's
        const int ks[] = {1, 20, 40, 64};                        // the engine's default, the widest list, both ends
        std::vector<float> logits;
        for (int nv : vocabs)
            for (int k : ks)
                for (int rep = 0; rep < 12; ++rep) {
                    logits.resize((size_t) nv);
                    for (auto& v : logits) v = gauss(rng);
                    // a block with fewer candidates than k: everything past three values is -inf
                    if (rep == 11) {
                        for (auto& v : logits) v = kNegInf;
                        logits[17] = 2.5f; logits[(size_t) nv / 2 + 3] = 2.5f; logits[(size_t) nv - 1] = -7.0f;   // two ties, in different blocks when there are enough
                    }
                    // the corner cases, dense enough that the top-k of some block holds several of each
                    const auto place = [&](float v, int count) {
                        for (int c = 0; c < count; ++c) logits[(size_t) (rng() % (unsigned) nv)] = v;
                    };
                    place(9.0f, 30);                                 // duplicates at the top: ties by id
                    place(8.5f, 3);                                  // a few more, one per block in the 40,960 case
                    place(12.0f, 2); place(kNegInf, 40); place(kNaN, 9); place(-kNaN, 3);
                    place(__builtin_huge_valf(), 2); place(-0.0f, 6); place(0.0f, 5);
                    logits[33] = -0.0f; logits[65] = 0.0f;          // -0.0 with the smaller id beside +0.0
                    logits[97] = 0.0f; logits[129] = -0.0f;         // and the other way round
                    logits[(size_t) nv - 1] = 1.0f;                 // the vocabulary's last id is a candidate
                    check(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "logits up");
                    const int n = mismatched_entries(d_logits, nv, k, d_rounds, d_fast, rounds, fast);
                    if (n != 0) {
                        if (bad == 0) {
                            size_t e = 0;
                            while (std::memcmp(&rounds[e], &fast[e], sizeof(int2)) == 0) ++e;
                            std::printf("    nv %d k %d rep %d: entry %zu (block %zu, rank %zu): rounds (id %d, bits %08x), one-pass (id %d, bits %08x)\n",
                                        nv, k, rep, e, e / (size_t) k, e % (size_t) k, rounds[e].x, (unsigned) rounds[e].y, fast[e].x,
                                        (unsigned) fast[e].y);
                            const size_t b0 = (e / (size_t) k) * (size_t) k;   // the failing block's lists, in rank order
                            std::printf("      rank:  id (value bits)   rounds | one-pass\n");
                            for (int r = 0; r < 10 && (size_t) r < (size_t) k; ++r)
                                std::printf("      %4d: %6d (%08x) | %6d (%08x)%s\n", r, rounds[b0 + r].x, (unsigned) rounds[b0 + r].y,
                                            fast[b0 + r].x, (unsigned) fast[b0 + r].y,
                                            std::memcmp(&rounds[b0 + r], &fast[b0 + r], sizeof(int2)) ? "   <--" : "");
                            // the host's truth for the failing block: each warp's first two in the order, and the two ids above
                            const size_t b_lo = (b0 / (size_t) k) * 4096u;
                            auto host_order = [&](size_t w) {   // the warp's candidates in the order (stable: ties by id)
                                std::vector<std::pair<float, int>> v;
                                for (size_t i = b_lo + w * 1024u; i < b_lo + (w + 1) * 1024u && i < logits.size(); ++i)
                                    if (logits[i] == logits[i] && logits[i] > kNegInf) v.emplace_back(logits[i], (int) i);
                                std::stable_sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
                                return v;
                            };
                            for (size_t w = 0; w < 4; ++w) {
                                const auto v = host_order(w);
                                std::printf("      host warp %zu: first two (id, bits): ", w);
                                for (size_t r = 0; r < 2 && r < v.size(); ++r) {
                                    unsigned int b = 0;
                                    std::memcpy(&b, &v[r].first, sizeof b);
                                    std::printf("(%d, %08x) ", v[r].second, b);
                                }
                                std::printf("\n");
                            }
                        }
                        ++bad;
                    }
                    ++cases;
                }
        std::printf("\n  %-52s %s (%d cases)\n", "one-pass top-k == k argmax rounds", bad ? "*** NO ***" : "bitwise", cases);
        // the comparison is observable: a one-entry change in a list is seen
        logits.resize((size_t) 4096);
        for (auto& v : logits) v = gauss(rng);
        check(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "logits up");
        (void) mismatched_entries(d_logits, 4096, 64, d_rounds, d_fast, rounds, fast);
        fast[5].y ^= 1;                                            // one mantissa bit of one entry
        const int seen = (int) (std::memcmp(rounds.data(), fast.data(), rounds.size() * sizeof(int2)) != 0);
        std::printf("  %-52s %s\n", "the comparison sees a one-entry difference", seen ? "yes" : "*** NO ***");
        if (!seen) ++bad;
    }

    if (bench) {
        // the drafter's shape (the ~40k draft vocabulary at the widest list) and the main sampler's (the artifact's
        // vocabulary at the engine's default top_k); each selection alone, 300 launches, CUDA events
        std::mt19937 rng(1u);
        std::normal_distribution<float> gauss(0.0f, 4.0f);
        std::vector<float> logits((size_t) nv_max);
        for (auto& v : logits) v = gauss(rng);
        check(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "logits up");
        struct Shape { int nv, k; const char* what; };
        for (Shape sh : {Shape{40960, 64, "the drafter's shape"}, Shape{151936, 20, "the main sampler's shape"}}) {
            float us[2] = {0.0f, 0.0f};
            for (int sel = 0; sel < 2; ++sel) {
                cudaEvent_t a, b;
                check(cudaEventCreate(&a), "event"); check(cudaEventCreate(&b), "event");
                for (int w = 0; w < 20; ++w) strata::kernels::sampler_split_part_both_for_test(d_logits, sh.nv, sh.k, d_rounds, d_fast, 1 << sel, nullptr);
                check(cudaDeviceSynchronize(), "warm");
                check(cudaEventRecord(a), "record");
                for (int i = 0; i < 300; ++i) strata::kernels::sampler_split_part_both_for_test(d_logits, sh.nv, sh.k, d_rounds, d_fast, 1 << sel, nullptr);
                check(cudaEventRecord(b), "record");
                check(cudaEventSynchronize(b), "sync");
                float ms = 0.0f;
                check(cudaEventElapsedTime(&ms, a, b), "elapsed");
                us[sel] = ms * 1000.0f / 300.0f;
                check(cudaEventDestroy(a), "destroy"); check(cudaEventDestroy(b), "destroy");
            }
            std::printf("  bench: nv %d, k %d (%s): k argmax rounds %6.1f us, one-pass top-k %6.1f us per call\n", sh.nv, sh.k, sh.what, us[0], us[1]);
        }
    }

    check(cudaFree(d_logits), "free"); check(cudaFree(d_rounds), "free"); check(cudaFree(d_fast), "free");
    if (bad) std::printf("  *** %d failing cases ***\n", bad);
    return bad ? 1 : 0;
}
