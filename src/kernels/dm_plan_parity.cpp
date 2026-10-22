// src/kernels/dm_plan_parity.cpp - the dm (basalt dec_moe) port's read-only parity boot (T26-1/T27-1).
//
//     build/dm_plan_parity
//
// Before any dm kernel is ported, this proves on OUR expert shapes that the algebra the port must reproduce
// already matches, and pins the two places it deliberately does not:
//   - OUR resident_plan kernel (the device_plan_ path's plan builder, verify_kernels.cu) against HOST
//     transcriptions of the donor's two plan algebras - dm_group (dec_moe.cuh:43-79, the route form's
//     first/rank/cnt pass) and dm_glue_gu_kernel's per-expert token MASK form (dec_moe.cuh:444-514) - field
//     by field (counts, start[], dst[], tok[], ptr[]), bitwise, over the pack's window shapes (T = 1..8,
//     k = 10, n_expert 512, capx 80) and routing regimes (all-resident, mixed residency in the partial mode,
//     cross-token shared experts, per-call G = 2 splits).  T26-1's "algebraically identical: first-occurrence
//     groups, entry order" claim, made executable.
//   - OUR combine's sum order (combine_k10_vec4 / _hits / the ZADD variant) against dm_combine4_kernel's
//     (donor native_moe.cu:127-198): first product rounded, one FMA per expert j = 1..9 in order, shared
//     added once; the VRAM rows' +0.0f (the donor's __fadd_rn(0.0f, p)) bitwise equal to our zeroed-row hit
//     add, including the -0 edge.  Also measured: the donor's source form UNcontracted (product rounded, then
//     added) - what nvcc's default fmad=true must keep contracting to for the port to stay bit-identical.
//   - the pack's seven (gate/up, down) pairs against a mirror of the donor's DM_PAIRS table.
// The documented divergences (asserted to be EXACTLY these, not failures): a repeat within a token goes to
// the CPU pool in dm (kind -1) while our plan keeps it (our router never emits one - top-k indices are
// distinct); an out-of-range id makes our kernel fall back to the host plan (skip = 0) while dm plans the
// rest into its VRAM tier.  Not covered here (needs the ported kernels): the expert row math itself - the
// donor's dm_test_kernel over the seven pairs runs after dec_moe.cuh/gu2.cuh land.
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N_EMBD = 2560, N_FF = 640, N_EXPERT = 512, K = 10, CAPX = 8 * K;   // our pack's MoE geometry
constexpr int K_DM_CAP = 128, K_DM_GLUE_EXPERTS = 1024, K_DM_GLUE_TOKENS = 16;   // the donor's bounds (dec_moe.hpp:22, dec_moe.cuh:418-419)
static_assert(CAPX <= K_DM_CAP && N_EXPERT <= K_DM_GLUE_EXPERTS && CAPX / K <= K_DM_GLUE_TOKENS,
              "our windows fit the donor's dm plan and glue bounds");

// our pack, native_experts.txt: 48 layers, all seven pairs inside the donor's DM_PAIRS (T26-1 section 2)
struct Pair { int gu, down, layers; const char* note; };
const Pair PACK_PAIRS[] = {
    {18, 20, 14, ""},
    {18, 42, 3, ""},
    {21, 20, 9, ""},
    {21, 42, 1, ""},
    {22, 20, 15, ""},
    {22, 42, 5, ""},
    {23, 20, 1, "route form only: 23 is in DM_PAIRS but not dec_moe_glue_ok's gate/up set"},
};

// the donor's DM_PAIRS (dec_moe.cuh:1024-1027), as gu*100 + down
const unsigned DM_PAIRS_DONOR[] = {
    16 * 100 + 20, 17 * 100 + 20, 18 * 100 + 20, 21 * 100 + 20, 22 * 100 + 20, 23 * 100 + 20, 29 * 100 + 20,
    42 * 100 + 20, 16 * 100 + 42, 17 * 100 + 42, 18 * 100 + 42, 21 * 100 + 42, 22 * 100 + 42, 23 * 100 + 42,
    29 * 100 + 42, 42 * 100 + 42, 40 * 100 + 40, 12 * 100 + 7, 12 * 100 + 8, 13 * 100 + 8, 14 * 100 + 8, 8 * 100 + 8,
};

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// ---------------------------------------------------------------- the donor's DmPlan, VRAM tier (host references)
struct DmPlanRef {
    int counts[4] = {};                  // [0] VRAM groups, [1] VRAM entries, [2] peer (0: no peer in our port), [3] CPU
    int start[K_DM_CAP + 1] = {};
    int dst[K_DM_CAP] = {}, tok[K_DM_CAP] = {}, kind[K_DM_CAP] = {};
    unsigned long long ptr[K_DM_CAP] = {};

    // bitwise over every field the plan carries (kind[] only vs the other reference: ours has no tier array)
    bool equal(const DmPlanRef& o, int n, const char** why) const {
        const auto bad = [&](const char* w) { if (why != nullptr) *why = w; return false; };
        if (counts[0] != o.counts[0] || counts[1] != o.counts[1] || counts[2] != o.counts[2] || counts[3] != o.counts[3])
            return bad("counts[]");
        for (int g = 0; g <= counts[0]; ++g)
            if (start[g] != o.start[g]) return bad("start[]");
        for (int i = 0; i < counts[1]; ++i)
            if (dst[i] != o.dst[i] || tok[i] != o.tok[i]) return bad("dst[]/tok[]");
        for (int i = 0; i < n; ++i)
            if (kind[i] != o.kind[i]) return bad("kind[]");
        for (int g = 0; g < counts[0]; ++g)
            if (ptr[g] != o.ptr[g]) return bad("ptr[]");
        return true;
    }
};

using blob_t = long long;

// dm_group (dec_moe.cuh:43-79) with the route kernel's tier rule (:159-171: a valid id that is not repeated
// within its token and has res[e] >= 0 is the VRAM tier; the peer tier does not exist in our port).
// `dup_rule`: a repeat within its token leaves the VRAM tier (the donor's defensive rule; our router never
// emits one).
DmPlanRef dm_route_ref(const int32_t* ids, int n, int k, const int32_t* res, int n_expert,
                       const uint8_t* cache_base, const unsigned long long* slot_off, blob_t blob, bool dup_rule) {
    DmPlanRef p;
    std::vector<int> kind(n, -1);
    for (int i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        bool ok = e >= 0 && e < n_expert;
        if (ok && dup_rule)
            for (int j = i - i % k; ok && j < i; ++j) ok = ids[j] != e;
        if (ok && res[e] >= 0) kind[i] = 0;
        p.kind[i] = kind[i];
    }
    struct ER { int first, rank, cnt; };
    std::vector<ER> er(n, ER{0, 0, 0});
    for (int i = 0; i < n; ++i) {
        if (kind[i] != 0) continue;
        er[i] = ER{i, 0, 0};
        for (int j = 0; j < n; ++j)
            if (ids[j] == ids[i] && kind[j] == 0) {
                if (j < er[i].first) er[i].first = j;
                if (j < i) ++er[i].rank;
                ++er[i].cnt;
            }
    }
    int ng = 0, ne = 0;
    std::vector<int> gstart_of_first(n, -1);
    for (int i = 0; i < n; ++i) {   // the openers, in entry order (= dm_group's gidx/gstart scans)
        if (kind[i] != 0 || er[i].first != i) continue;
        gstart_of_first[i] = ne;
        p.start[ng] = ne;
        const int32_t slot = res[ids[i]];
        p.ptr[ng] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        ++ng;
        ne += er[i].cnt;
    }
    for (int i = 0; i < n; ++i) {   // every kind-0 entry at s_pos = gstart[first] + rank (dec_moe.cuh:222-223)
        if (kind[i] != 0) continue;
        const int pos = gstart_of_first[er[i].first] + er[i].rank;
        p.dst[pos] = i;
        p.tok[pos] = i / k;
    }
    p.start[ng] = ne;
    p.counts[0] = ng;
    p.counts[1] = ne;
    p.counts[2] = 0;
    for (int i = 0; i < n; ++i)
        if (kind[i] != 0) ++p.counts[3];   // the CPU pool's share
    return p;
}

// dm_glue_gu_kernel's grouping from the per-expert token mask (dec_moe.cuh:444-514), host transcription:
// bits[e] = the tokens routing to e; an entry is VRAM when its id is in range, the expert is resident (the
// glue's ptab[e] != 0), and - when `dup_rule` - no earlier entry of its token repeated it; is_first = low
// (the earlier tokens' bits) == 0; the group's size = popc(bits), its tokens the mask's set bits ascending;
// group index and start = the warp scans of is_first / sizes over the entry order.
DmPlanRef dm_glue_mask_ref(const int32_t* ids, int n, int k, const int32_t* res, int n_expert,
                           const uint8_t* cache_base, const unsigned long long* slot_off, blob_t blob,
                           bool dup_rule) {
    DmPlanRef p;
    std::vector<uint32_t> bits(n_expert, 0);
    std::vector<char> dup(n, 0);
    for (int i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert) continue;
        const uint32_t bit = 1u << (i / k);
        if ((bits[e] & bit) != 0u) dup[i] = (char) dup_rule;   // atomicOr's old bit set: s_dup; the repeat is not VRAM
        else bits[e] |= bit;
    }
    std::vector<int> gstart_e(n_expert, -1);
    int ng = 0, ne = 0;
    for (int i = 0; i < n; ++i) {   // the openers off the mask (the warp scans, in entry order)
        const int32_t e = ids[i];
        const bool inr = e >= 0 && e < n_expert;
        const bool valid = inr && res[e] >= 0 && !dup[i];
        const uint32_t b = inr ? bits[e] : 0u;
        const uint32_t low = b & ((1u << (i / k)) - 1u);
        if (!(valid && low == 0u)) continue;
        gstart_e[e] = ne;
        p.start[ng] = ne;
        int r = 0;
        for (uint32_t m = b; m != 0u; m &= m - 1u) p.tok[ne + r++] = __builtin_ctz(m);   // s_tok, tokens ascending
        const int32_t slot = res[e];
        p.ptr[ng] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        ++ng;
        ne += __builtin_popcount(b);
    }
    for (int i = 0; i < n; ++i) {   // every valid entry: pos = its group's start + popc(low)
        const int32_t e = ids[i];
        const bool inr = e >= 0 && e < n_expert;
        const bool valid = inr && res[e] >= 0 && !dup[i];
        const uint32_t b = inr ? bits[e] : 0u;
        const uint32_t low = b & ((1u << (i / k)) - 1u);
        p.kind[i] = valid ? 0 : -1;
        if (!valid) continue;
        const int pos = gstart_e[e] + __builtin_popcount(low);
        p.dst[pos] = i;
        p.tok[pos] = i / k;
    }
    p.start[ng] = ne;
    p.counts[0] = ng;
    p.counts[1] = ne;
    p.counts[2] = 0;
    for (int i = 0; i < n; ++i)
        if (p.kind[i] != 0) ++p.counts[3];
    return p;
}

// ---------------------------------------------------------------- our plan, off the device
struct OurPlan {
    int counts[3] = {};   // groups, entries, PCIe groups (a device plan carries none)
    std::vector<int32_t> start, dst, tok;
    std::vector<unsigned long long> ptr;
    uint32_t skip = 0;

    bool equal(const DmPlanRef& r, std::string* why) const {
        const auto bad = [&](const char* w) { if (why != nullptr) *why = w; return false; };
        if (counts[0] != r.counts[0]) return bad("counts[0] (groups)");
        if (counts[1] != r.counts[1]) return bad("counts[1] (entries)");
        if (counts[2] != 0) return bad("counts[2] (a device plan has no PCIe groups)");
        for (int g = 0; g <= counts[0]; ++g)
            if (start[g] != r.start[g]) return bad("start[]");
        for (int i = 0; i < counts[1]; ++i)
            if (dst[i] != r.dst[i] || tok[i] != r.tok[i]) return bad("dst[]/tok[]");
        for (int g = 0; g < counts[0]; ++g)
            if (ptr[g] != r.ptr[g]) return bad("ptr[]");
        return true;
    }
};

// decode our plan buffer (counts | start | dst | tok | pad | ptr | ptr2 | start2; verify.cpp:2062-2070)
OurPlan our_plan(const int32_t* pl, long long capx, uint32_t skip) {
    OurPlan o;
    o.counts[0] = pl[0];
    o.counts[1] = pl[1];
    o.counts[2] = pl[2];
    const int32_t* start = pl + 4;
    const int32_t* dst = start + capx + 1;
    const int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const unsigned long long* ptr = (const unsigned long long*) (pl + ptr_off);
    o.start.assign(start, start + o.counts[0] + 1);
    o.dst.assign(dst, dst + o.counts[1]);
    o.tok.assign(tok, tok + o.counts[1]);
    o.ptr.assign(ptr, ptr + o.counts[0]);
    o.skip = skip;
    return o;
}

long long plan_words(long long capx) {
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    return ptr_off + 4 * capx + (capx + 1) + 2;   // verify.cpp:797's size, +2 slack
}

std::vector<std::unique_ptr<std::vector<uint64_t>>> g_plan_kept;   // the plans live as long as the process
int32_t* pl_words(long long capx) {
    auto buf = std::make_unique<std::vector<uint64_t>>((size_t) ((plan_words(capx) + 1) & ~1ll), 0);
    g_plan_kept.emplace_back(std::move(buf));
    return (int32_t*) g_plan_kept.back()->data();   // uint64_t storage: pl + ptr_off is 8-byte aligned
}

// ---------------------------------------------------------------- routing regimes
enum Regime { RG_RANDOM, RG_POOL, RG_DUP, RG_INVALID };

std::vector<int32_t> gen_ids(int T, Regime rg, std::mt19937& rng) {
    std::vector<int32_t> ids((size_t) T * K, -1);
    std::vector<int32_t> pool(N_EXPERT);
    for (int e = 0; e < N_EXPERT; ++e) pool[e] = e;
    if (rg == RG_POOL) std::shuffle(pool.begin(), pool.end(), rng);   // ONE shared K-subset: cross-token groups
    for (int t = 0; t < T; ++t) {
        if (rg != RG_POOL) std::shuffle(pool.begin(), pool.end(), rng);
        for (int j = 0; j < K; ++j) ids[(size_t) t * K + j] = pool[j];   // K distinct ids in the token
        if (rg == RG_DUP && t == 0) ids[1] = ids[0];   // a repeat within token 0
        if (rg == RG_INVALID) {
            if (t == 0) ids[3] = -1;                                       // below the table
            if (t > 0) ids[(size_t) t * K + 2] = (int32_t) N_EXPERT + 7;   // past it
        }
    }
    return ids;
}

// per-expert slots: distinct nonzero slots for the resident experts, -1 for the misses (the residency row is
// per expert, as the kernel's res[] and dm's res0[] are)
std::vector<int32_t> gen_res(double hit_rate, std::mt19937& rng) {
    std::vector<int32_t> res(N_EXPERT, -1);
    std::vector<int32_t> slots(N_EXPERT);
    for (int i = 0; i < N_EXPERT; ++i) slots[i] = i * 3 + 1;
    std::shuffle(slots.begin(), slots.end(), rng);
    int s = 0;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int e = 0; e < N_EXPERT; ++e)
        if (hit_rate >= 1.0 || u(rng) < hit_rate) res[e] = slots[s++];
    return res;
}

// ---------------------------------------------------------------- one plan case
int g_cases = 0;
long long g_field_checks = 0;

bool plan_case(const char* name, int T, Regime rg, double hit_rate, bool with_slot_off, std::mt19937& rng,
               cudaStream_t s, uint8_t* d_cache, bool partial_call) {
    ++g_cases;
    const std::vector<int32_t> ids = gen_ids(T, rg, rng);
    const int n = T * K;
    const std::vector<int32_t> res = gen_res(hit_rate, rng);
    bool all_resident = true;
    for (int i = 0; i < n; ++i) all_resident &= ids[i] >= 0 && ids[i] < N_EXPERT && res[ids[i]] >= 0;
    std::vector<unsigned long long> slot_off;
    unsigned long long* d_slot_off = nullptr;
    if (with_slot_off) {
        slot_off.resize(4096);
        for (auto& v : slot_off) v = (unsigned long long) (rng() & 0x3ffff) * 256ull;
        if (!ck(cudaMalloc((void**) &d_slot_off, slot_off.size() * 8), "slot_off malloc")) return false;
        if (!ck(cudaMemcpy(d_slot_off, slot_off.data(), slot_off.size() * 8, cudaMemcpyHostToDevice), "slot_off H2D"))
            return false;
    }
    int32_t *d_ids = nullptr, *d_res = nullptr, *pl = pl_words(CAPX);
    uint32_t* d_skip = nullptr;
    bool ok = ck(cudaMalloc((void**) &d_ids, n * 4), "ids malloc") && ck(cudaMalloc((void**) &d_res, N_EXPERT * 4), "res malloc") &&
              ck(cudaMalloc((void**) &d_skip, 64), "skip malloc") && ck(cudaMemset(d_skip, 0, 64), "skip zero");
    if (ok) ok = ck(cudaMemcpy(d_ids, ids.data(), n * 4, cudaMemcpyHostToDevice), "ids H2D") &&
                 ck(cudaMemcpy(d_res, res.data(), N_EXPERT * 4, cudaMemcpyHostToDevice), "res H2D");
    if (!ok) return false;

    // the calls the device_plan_ path makes (verify.cpp:1970-1975): the plain ring skip, and T1.1's partial
    // one.  The ring value is arbitrary (any value the windows' per-layer ring counter reaches).
    constexpr uint32_t ring = 37u;
    strata::kernels::resident_plan(d_ids, n, K, d_res, N_EXPERT, d_cache, d_slot_off,
                                   (blob_t) N_FF * N_EMBD * 2, pl, CAPX, d_skip, ring, s, nullptr,
                                   partial_call ? strata::kernels::kPlanPartialSkip : 0u);
    if (!ck(cudaStreamSynchronize(s), "plan run")) return false;
    uint32_t hskip = 0;
    ck(cudaMemcpy(&hskip, d_skip, 4, cudaMemcpyDeviceToHost), "skip D2H");
    std::vector<int32_t> hpl((size_t) plan_words(CAPX), 0);
    ck(cudaMemcpy(hpl.data(), pl, hpl.size() * 4, cudaMemcpyDeviceToHost), "plan D2H");
    const OurPlan ours = our_plan(hpl.data(), CAPX, hskip);

    // the host references: dm_group (the route form) and the glue's mask form, each with the donor's dup rule
    // (kind -1 for a repeat within its token) and without it (our kernel's rule: keep it - the router never
    // emits one; the divergence is asserted exactly on the dup case below)
    const uint8_t* cache_base = d_cache;   // the value only: both sides evaluate the same slot expression
    const unsigned long long* so = with_slot_off ? slot_off.data() : nullptr;
    const blob_t blob = (blob_t) N_FF * N_EMBD * 2;
    const DmPlanRef ref_group = dm_route_ref(ids.data(), n, K, res.data(), N_EXPERT, cache_base, so, blob, true);
    const DmPlanRef ref_group_nodup = dm_route_ref(ids.data(), n, K, res.data(), N_EXPERT, cache_base, so, blob, false);
    const DmPlanRef ref_mask = dm_glue_mask_ref(ids.data(), n, K, res.data(), N_EXPERT, cache_base, so, blob, true);
    const DmPlanRef ref_mask_nodup = dm_glue_mask_ref(ids.data(), n, K, res.data(), N_EXPERT, cache_base, so, blob, false);

    auto fail = [&](const std::string& w) {
        std::printf("   FAIL %-34s T=%d: %s\n", name, T, w.c_str());
        return false;
    };
    // the donor's two forms agree with each other wherever both are defined (T26-1: "algebraically
    // identical").  On a distinct-in-token id set (the router's output) that holds with the dup rule off
    // too; a within-token repeat has no no-dup-rule mask form at all (the mask holds one bit per token).
    {
        const char* why = nullptr;
        if (!ref_group.equal(ref_mask, n, &why)) return fail(std::string("dm_group vs dm_glue mask: ") + why);
        if (rg != RG_DUP && !ref_group_nodup.equal(ref_mask_nodup, n, &why))
            return fail(std::string("dm_group vs dm_glue mask (dup rule off): ") + why);
    }

    if (rg == RG_INVALID) {
        // ours falls back to the host plan (skip = 0) in both modes; dm plans the rest into its VRAM tier.
        // The documented divergence - unreachable on our stack (the router's top-k ids are in [0, 512)).
        if (hskip != 0) return fail("an invalid id did not fall back to the host plan");
        std::printf("   %-34s T=%d skip=0 (host fallback; dm would plan %d VRAM entries)  note-only\n", name, T,
                    ref_group.counts[1]);
    } else if (partial_call) {
        if (hskip != strata::kernels::kPlanPartialSkip) return fail("the partial call did not take the partial skip");
        std::string why;
        if (!ours.equal(ref_group_nodup, &why)) return fail("ours vs dm_group: " + why);
        if (rg != RG_DUP && !ours.equal(ref_mask_nodup, &why)) return fail("ours vs dm_glue mask: " + why);
        g_field_checks += 2 * (4 + ref_group_nodup.counts[0] + 1 + 2 * ref_group_nodup.counts[1] + ref_group_nodup.counts[0]);
        std::printf("   %-34s T=%d %s groups=%d entries=%d (CPU %d)  bitwise vs dm_group + dm_glue-mask\n", name, T,
                    all_resident ? "full  " : "mixed ", ref_group_nodup.counts[0], ref_group_nodup.counts[1],
                    ref_group_nodup.counts[3]);
    } else {
        if (all_resident) {
            if (hskip != ring) return fail("a fully resident group was not device-planned");
            std::string why;
            if (!ours.equal(ref_group_nodup, &why)) return fail("ours vs dm_group: " + why);
            if (rg != RG_DUP && !ours.equal(ref_mask_nodup, &why)) return fail("ours vs dm_glue mask: " + why);
            g_field_checks += 2 * (4 + ref_group_nodup.counts[0] + 1 + 2 * ref_group_nodup.counts[1] + ref_group_nodup.counts[0]);
            std::printf("   %-34s T=%d full   groups=%d entries=%d  bitwise vs dm_group + dm_glue-mask\n", name, T,
                        ref_group_nodup.counts[0], ref_group_nodup.counts[1]);
        } else {
            if (hskip != 0) return fail("a non-plannable group did not fall back (skip != 0)");
            std::printf("   %-34s T=%d mixed  skip=0 (host fallback, today's default)  note-only\n", name, T);
        }
    }
    if (rg == RG_DUP && all_resident && !partial_call) {
        // the documented divergence, asserted exactly: ours keeps the RESIDENT repeated entry (our router
        // never emits one); dm demotes it to the CPU pool.  ours == the no-dup-rule references, and !=
        // the true ones.
        if (ours.equal(ref_group, nullptr) || ours.equal(ref_mask, nullptr))
            return fail("the dup divergence did not show (reference bug)");
        std::printf("      dup note: ours plans %d entries, dm plans %d (the in-token repeat goes to the CPU pool)\n",
                    ref_group_nodup.counts[1], ref_group.counts[1]);
    }
    if (d_slot_off) ck(cudaFree(d_slot_off), "slot_off free");
    ck(cudaFree(d_ids), "ids free");
    ck(cudaFree(d_res), "res free");
    ck(cudaFree(d_skip), "skip free");
    return ok;
}

// ---------------------------------------------------------------- the combine's sum order
static float fl(float v) { volatile float x = v; return x; }   // round to float; blocks contraction across it

// ours / the donor as nvcc's default fmad=true compiles it: first product rounded, one FMA per expert in order
static float chain_fma(const float* p, const float* w, float sh) {
    float sum = fl(p[0] * w[0]);
    for (int j = 1; j < K; ++j) sum = std::fma(p[j], w[j], sum);
    return fl(sum + sh);
}
// the donor's source as written, uncontracted: every product rounds on its own before the add
static float chain_uncontracted(const float* p, const float* w, float sh) {
    float sum = fl(p[0] * w[0]);
    for (int j = 1; j < K; ++j) sum = fl(sum + fl(p[j] * w[j]));
    return fl(sum + sh);
}

bool combine_parity(cudaStream_t s) {
    constexpr int T = 8;
    const int64_t N = N_EMBD;
    std::mt19937 rng(20261021u);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> coin(0, 9);
    std::vector<float> parts((size_t) T * K * N), hits((size_t) T * K * N), w((size_t) T * K), sh((size_t) T * N);
    for (auto& v : parts) v = nd(rng);
    for (auto& v : hits) v = coin(rng) == 0 ? 0.0f : (coin(rng) == 1 ? -0.0f : nd(rng));
    for (auto& v : w) v = nd(rng);
    for (auto& v : sh) v = coin(rng) == 0 ? -0.0f : nd(rng);
    // the hit rows (dst/count): a random subset of each token's experts - the entries the VRAM tier computed.
    // count is ONE device int (the plan's entries total) and dst lists token*K + expert (the kernel's map).
    std::vector<int32_t> dst;
    std::vector<char> hit_mask((size_t) T * K, 0);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < K; ++j)
            if (coin(rng) < 7) {   // ~70% hits, the live config's regime
                hit_mask[(size_t) t * K + j] = 1;
                dst.push_back(t * K + j);
            }
    float *dp = nullptr, *dh = nullptr, *dw = nullptr, *dsh = nullptr, *da = nullptr, *db = nullptr, *dc = nullptr;
    int32_t *ddst = nullptr, *dcount = nullptr;
    const int32_t total_hits = (int32_t) dst.size();
    bool ok = ck(cudaMalloc((void**) &dp, parts.size() * 4), "malloc") && ck(cudaMalloc((void**) &dh, hits.size() * 4), "malloc") &&
              ck(cudaMalloc((void**) &dw, w.size() * 4), "malloc") && ck(cudaMalloc((void**) &dsh, sh.size() * 4), "malloc") &&
              ck(cudaMalloc((void**) &da, parts.size() * 4), "malloc") && ck(cudaMalloc((void**) &db, parts.size() * 4), "malloc") &&
              ck(cudaMalloc((void**) &dc, parts.size() * 4), "malloc") &&
              ck(cudaMalloc((void**) &ddst, std::max<size_t>(dst.size(), 1) * 4), "malloc") &&
              ck(cudaMalloc((void**) &dcount, 4), "malloc");
    if (ok) ok = ck(cudaMemcpy(dp, parts.data(), parts.size() * 4, cudaMemcpyHostToDevice), "H2D") &&
                 ck(cudaMemcpy(dh, hits.data(), hits.size() * 4, cudaMemcpyHostToDevice), "H2D") &&
                 ck(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "H2D") &&
                 ck(cudaMemcpy(dsh, sh.data(), sh.size() * 4, cudaMemcpyHostToDevice), "H2D") &&
                 ck(cudaMemcpy(ddst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "H2D") &&
                 ck(cudaMemcpy(dcount, &total_hits, 4, cudaMemcpyHostToDevice), "H2D");
    if (!ok) return false;

    // A: the plain multi combine over the parts.  B: moe_hit_add's fold - zeroed part rows with the hits
    // added for the plan's rows (the engine's exact pre-combine state).  C: the ZADD variant, every part
    // read as 0.0f + hit (dm_combine4's __fadd_rn(0.0f, p) on its VRAM rows, -0 edge included).
    std::vector<float> parts_zero(parts.size(), 0.0f);
    ck(cudaMemcpy(dp, parts_zero.data(), parts.size() * 4, cudaMemcpyHostToDevice), "zero parts");
    strata::kernels::native_moe_combine_multi_hits_added(dp, dh, ddst, dcount, dw, dsh, db, N, K, T, s);
    ck(cudaMemcpy(dp, parts.data(), parts.size() * 4, cudaMemcpyHostToDevice), "restore parts");
    strata::kernels::native_moe_combine_multi(dp, dw, dsh, da, N, K, T, s);
    strata::kernels::native_moe_combine_multi_hits(dh, dw, dsh, dc, N, K, T, s);
    if (!ck(cudaStreamSynchronize(s), "combine run")) return false;
    std::vector<float> ga((size_t) T * N), gb((size_t) T * N), gc((size_t) T * N);
    ck(cudaMemcpy(ga.data(), da, ga.size() * 4, cudaMemcpyDeviceToHost), "D2H");
    ck(cudaMemcpy(gb.data(), db, gb.size() * 4, cudaMemcpyDeviceToHost), "D2H");
    ck(cudaMemcpy(gc.data(), dc, gc.size() * 4, cudaMemcpyDeviceToHost), "D2H");

    long long bad_a = 0, bad_b = 0, bad_c = 0, uncontracted_diff = 0, total = 0;
    for (int t = 0; t < T; ++t)
        for (int64_t c = 0; c < N; ++c) {
            const float* p = parts.data() + ((size_t) t * K) * N + c;   // [t][j][N + c] at p[j * N]
            const float* h = hits.data() + ((size_t) t * K) * N + c;
            const float* ww = w.data() + (size_t) t * K;
            const float shv = sh[(size_t) t * N + c];
            const float ea = chain_fma(p, ww, shv);
            float eb_parts[K];
            for (int j = 0; j < K; ++j)
                eb_parts[j] = hit_mask[(size_t) t * K + j] ? fl(0.0f + h[(int64_t) j * N]) : 0.0f;
            const float eb = chain_fma(eb_parts, ww, shv);
            float ec_parts[K];
            for (int j = 0; j < K; ++j) ec_parts[j] = fl(0.0f + h[(int64_t) j * N]);
            const float ec = chain_fma(ec_parts, ww, shv);
            const float ed = chain_uncontracted(p, ww, shv);
            ++total;
            if (std::memcmp(&ea, &ga[(size_t) t * N + c], 4) != 0) ++bad_a;
            if (std::memcmp(&eb, &gb[(size_t) t * N + c], 4) != 0) ++bad_b;
            if (std::memcmp(&ec, &gc[(size_t) t * N + c], 4) != 0) ++bad_c;
            if (std::memcmp(&ea, &ed, 4) != 0) ++uncontracted_diff;
        }
    // the -0 edge of the VRAM rows' add: __fadd_rn(0.0f, p) flips -0 to +0, exactly the host's add here
    bool zero_edge = true;
    for (float v : {0.0f, -0.0f}) {
        const float r = fl(0.0f + v);
        uint32_t b;
        std::memcpy(&b, &r, 4);
        if (b != 0u) zero_edge = false;
    }
    std::printf("   combine (T=%d, N=%d, K=%d, %lld columns): multi %s, hits-added %s, zadd %s | donor form "
                "uncontracted differs on %lld/%lld columns (%.1f%%)%s\n",
                T, (int) N, K, total, bad_a == 0 ? "bitwise" : "MISMATCH", bad_b == 0 ? "bitwise" : "MISMATCH",
                bad_c == 0 ? "bitwise" : "MISMATCH", uncontracted_diff, total,
                100.0 * (double) uncontracted_diff / (double) total, zero_edge ? "" : " | -0 EDGE BROKE");
    std::printf("      port note: dm_combine4's sums must compile with fmad on (nvcc's default): the source's\n"
                "      'sx += v[j].x * w[j]' contracts to our fmaf chain - uncontracted it is a DIFFERENT sum\n");
    ck(cudaFree(dp), "free"); ck(cudaFree(dh), "free"); ck(cudaFree(dw), "free"); ck(cudaFree(dsh), "free");
    ck(cudaFree(da), "free"); ck(cudaFree(db), "free"); ck(cudaFree(dc), "free");
    ck(cudaFree(ddst), "free"); ck(cudaFree(dcount), "free");
    return bad_a == 0 && bad_b == 0 && bad_c == 0 && zero_edge;
}

bool pairs_covered() {
    int total = 0;
    bool ok = true;
    for (const Pair& pr : PACK_PAIRS) {
        total += pr.layers;
        const unsigned key = (unsigned) (pr.gu * 100 + pr.down);
        bool in = false;
        for (unsigned d : DM_PAIRS_DONOR) in |= d == key;
        std::printf("   pair (%d,%d) x%d: %s%s%s\n", pr.gu, pr.down, pr.layers, in ? "in DM_PAIRS" : "MISSING FROM DM_PAIRS",
                    pr.note[0] ? " - " : "", pr.note);
        ok &= in;
    }
    std::printf("   %d layers covered, n_ff %d, n_embd %d (glue bounds: %d tokens x %d experts)\n", total, N_FF, N_EMBD,
                K_DM_GLUE_TOKENS, K_DM_GLUE_EXPERTS);
    return ok && total == 48;
}

}  // namespace

int main() {
    cudaStream_t s;
    if (!ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream")) return 1;
    bool ok = true;

    std::printf("dm_plan_parity: the pack's pairs vs the donor's DM_PAIRS\n");
    ok = pairs_covered() && ok;

    // the plan algebra: the GPU kernel (our resident_plan, the device_plan_ path) vs the donor's two host
    // transcriptions, over the window shapes and routing regimes; the G = 2 split's halves included (the
    // engine plans each half by its own call, verify.cpp:1961/1971)
    uint8_t* d_cache = nullptr;
    if (!ck(cudaMalloc((void**) &d_cache, 256), "cache malloc")) return 1;
    std::printf("plan algebra: ours (resident_plan) vs dm_group and dm_glue's mask form, bitwise\n");
    std::mt19937 rng(777u);
    for (int T = 1; T <= 8; ++T) {
        ok = plan_case("all-resident", T, RG_RANDOM, 1.0, (T & 1) != 0, rng, s, d_cache, false) && ok;
        ok = plan_case("all-resident (partial call)", T, RG_RANDOM, 1.0, (T & 1) == 0, rng, s, d_cache, true) && ok;
        ok = plan_case("mixed ~0.7 (partial call)", T, RG_RANDOM, 0.7, true, rng, s, d_cache, true) && ok;
        ok = plan_case("mixed ~0.7 (plain call)", T, RG_RANDOM, 0.7, false, rng, s, d_cache, false) && ok;
        ok = plan_case("cross-token shared experts", T, RG_POOL, 1.0, (T % 2) != 0, rng, s, d_cache, false) && ok;
        if (T >= 2) ok = plan_case("shared experts (partial call)", T, RG_POOL, 0.5, true, rng, s, d_cache, true) && ok;
    }
    ok = plan_case("dup within token (doc)", 4, RG_DUP, 1.0, true, rng, s, d_cache, false) && ok;
    ok = plan_case("invalid id (doc)", 4, RG_INVALID, 1.0, true, rng, s, d_cache, false) && ok;
    ok = plan_case("G2 half A (T=4)", 4, RG_RANDOM, 1.0, true, rng, s, d_cache, false) && ok;
    ok = plan_case("G2 half B (T=4)", 4, RG_RANDOM, 0.7, true, rng, s, d_cache, true) && ok;

    std::printf("combine order: ours (combine_k10_vec4 family) vs dm_combine4's sums\n");
    ok = combine_parity(s) && ok;

    std::printf("plan cases %d, plan fields compared >= %lld\n", g_cases, g_field_checks);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    ck(cudaFree(d_cache), "cache free");
    return ok ? 0 : 1;
}
