---
type: research-digest
id: 2026-11-07-preboot-ds-pool-sizing
date: 2026-11-07
scope: the ONE number for /etc/sysctl.d/91-strata-hugepages.conf before the reboot — the DeepSeek engine's CPU-lane copy arena (DS4_COPY_HUGETLB / DS4_COPY_PIN semantics) sized against the standing Qwen pool consumers
subject: pre-boot hugetlb pool sizing; dsv41 copy-arena defaults; partial-pool and pin behavior; post-boot DS battery plan
method: local code study (src/dsv41), log mining (logs/custom-dsv41-*), pack/manifest measurement, live /proc/meminfo. No web sources. No code changes.
---

# Pre-boot DS pool sizing — the decision

## Verdict (read this first)

**Recommendation: `vm.nr_hugepages=54272` (106 GiB, 54,272 × 2 MiB pages = 113.82 GB), replacing the armed 45056.** This is the smallest whole-GiB pool that carries the DS copy arena's measured worst case (110.85 GB = 52,860 pages) plus 2 GiB slack, keeps every standing Qwen consumer fully pool-backed (worst concurrent draw 80.4 GiB), and covers UD-IQ4_XS whole (94 GiB = 48,128 pages).

**Option (a) — keeping 45056 — is not "status quo"; it is a DS regression.** At 45056 the pool (94.49 GB) is smaller than `all` (110.7-110.8 GB), so `huge_arena` defaults OFF, and then the budget's `can` loses the pool term: post-boot `can ≈ avail − 8.59 GB ≈ 22-31 GB` < 110.8 GB → auto mode's all-or-none budget goes to 0 → **the DS copy does not happen at all** and the CPU lane falls to in-place streaming (decode precedent: copies beat in-place by +10-25%; on this box the cache window is far smaller than the 251 GB reference). The pool arming already broke the copy TODAY (pre-reboot): MemAvailable right now is 78.02 GB → `can = 69.43 GB` < `all`. Raising the pool is what repairs DS, not an optimization on top.

---

## 1. Code semantics (file:line)

### 1.1 The huge-arena default — `copy_hugetlb_default`

`src/dsv41/ds4_experts_rule.hpp:172-177`:

```cpp
// DS4_COPY_HUGETLB (the copy arena on explicit 2 MB pages): env -1 unset, else 0 / 1; 0.2.6: opt-in; 0.2.7: on when the reserved pool already holds the whole copy.
inline bool copy_hugetlb_default(int env, double pool_free_bytes, double copy_bytes, bool v026 = false) {
    if (env >= 0) return env != 0;
    if (v026) return false;
    return pool_free_bytes > 0.0 && pool_free_bytes >= copy_bytes;
}
```

- `huge_arena` is true iff `HugePages_Free × Hugepagesize ≥ all` (the whole CPU-lane copy, spares included) — or an explicit `DS4_COPY_HUGETLB=1`. **Partial pool ⇒ default OFF.** Called at `src/dsv41/ds4_model.cu:4668-4669` with `pool_free = os::huge_free_bytes()` and `all` computed at `ds4_model.cu:4654-4660` (sum of each routed expert's blob, `+1 spare` slot on a layer's first copy).
- `huge_free_bytes()` (`src/dsv41/ds4_os.hpp:404-420`) reads `HugePages_Free × Hugepagesize` from `/proc/meminfo` — live pool state at load, not the armed target.
- When the default fires, the engine prints the witness line (`ds4_model.cu:4670-4672`): `"ds4: the reserved huge-page pool (%.1f GB free) holds the whole copy (%.1f GB): the arena on 2 MB pages (DS4_COPY_HUGETLB=0 keeps the ordinary arena)"`.

### 1.2 Partial pool inside the load — per-layer fallback

`ds4_model.cu:4737-4752`: each layer's slot block is `os::huge_alloc(bytes)` when `huge_arena`; on the first `nullptr` (pool exhausted mid-load):

- warn-once: `"ds4: the 2 MB huge-page arena was asked for but the pool ran out (reserve more: sysctl vm.nr_hugepages=N); using the ordinary arena"` (`:4745-4746`);
- that layer (and the rest) fall back to `node_map` (ordinary anon, `MADV_HUGEPAGE`, NUMA-bound), and `arena_all_huge = false` **sticky for the whole load** (`:4749`, set at `:4673` with the comment "(every chunk really from the pool: the pin below needs that)").

So a partial pool that trips mid-load yields a mixed arena (some layers on pool pages, the rest THP-anon) — the worst case, deliberately steered away from the pin.

### 1.3 The pin — `copy_pin_default` with a partially-huge arena

`src/dsv41/ds4_experts_rule.hpp:164-170`:

```cpp
// DS4_COPY_PIN ... 0.2.7: on when the whole arena is on explicit huge pages OR (RDNA4 and the cards hold at least half the experts).
inline bool copy_pin_default(int env, bool arena_all_huge, bool rdna4, double share, bool have_chunks, bool v026 = false) {
    if (env >= 0) return env != 0;
    if (v026) return false;
    return (arena_all_huge || (rdna4 && share >= 0.5)) && have_chunks;
}
```

Called at `ds4_model.cu:4835-4836`. **On this box the pin can only ever fire through `arena_all_huge`**: the build is CUDA (RTX 5090; `lib_dirs: cuda-13.4` in `config/custom-dsv41-8081.json`), so `rdna4()` is false; and the GPU tier holds ~2,560 of 15,360 routed experts (16.7%) so `share ≈ 0.17 < 0.5` regardless. Partial pool ⇒ `arena_all_huge = false` ⇒ **pin OFF** ⇒ the prompt path stages experts through the pinned staging ring — the path the comments measure at 6.8 GB/s streamed vs 24/28 GB/s probes (`ds4_model.cu:4822-4834`) and the erratic-pin lesson at `:4734-4736`:

> "a THP arena pinned by cudaHostRegister runs in 4 KB pages: measured 11.9 s one prompt, 30.4 s the next for the same 140 GB"

**Conclusion: the full pool is required to light BOTH mechanisms (2 MiB arena + pinned DMA). A partial pool lights neither and keeps the ordinary-arena staging path.**

### 1.4 The budget — why a small pool also kills the copy itself

`ds4_model.cu:4662-4680`:

```cpp
const uint64_t avail = ram_avail();
// DS4_COPY_HUGETLB: the arena comes from the kernel's huge-page pool, which MemAvailable does not count: add
// its free bytes so the full copy is taken when the pool holds it.
const uint64_t pool_free = os::huge_free_bytes();
const bool huge_arena = ...;
const double pool = huge_arena ? (double) pool_free : 0.0;
const double can = std::min(avail == ~uint64_t{0} ? 1e300 : std::max(0.0, (double) avail - (double) kRamMargin) + pool, commit_room());
...
double budget = o.mmap_experts == 0 ? std::min(all, can) : auto_copy_all && all <= can ? all : 0.0;   // (auto: all or none)
```

- `kRamMargin = 8 GiB` (`ds4_model.cu:950`; printed as "9 GB" decimal in the log lines).
- Standing mode is `--mmap-experts auto` (no flag in the launcher/config) ⇒ **all or none**: `can < all` ⇒ budget 0 ⇒ `copy_mode` never set ⇒ the CPU lane reads the model files in place. The failure prints (`:4682-4684`): `"ds4: the CPU lane's experts (%.1f GB) do not all fit the RAM this process can get (%.1f GB beside %.0f GB for the rest): read in place from the model's files"`.
- `ram_avail()` = MemAvailable (`ds4_model.cu:180-183` → `src/core/host_mem.cpp:255-257`, cgroup/operator cap aside).
- The `+ pool` term exists precisely so a full pool makes the copy possible on a pool-heavy box (comment `:4664-4665`). **This is the design intent the pool size must honor.**

## 2. DS's actual copy arena `all` (measured)

Source: `logs/custom-dsv41-8081.log` (the standing config: pack `deepseek-v4.1-flash-ist-das-2.00bpw`, `--experts 4`, `--dspark auto`, max-context 262144; `config/custom-dsv41-8081.json`). Last five boots (file mtime 2026-11-05):

> `ds4: --mmap-experts auto: the CPU lane's experts (110.8 GB) fit the RAM this process can get (126.2 GB beside 9 GB for the rest): copied in at load, no demand store (--mmap-experts on keeps them in place)`
> `ds4: 12800 of the CPU lane's 12800 experts copied into RAM (110.1 GB, 1 NUMA part) in 37.2 s`

- **`all` = 110.7-110.8 GB** across the last five boots (printed at 1 decimal ⇒ true value < 110.85 GB). The copied total (excluding spare-slot bytes) is 109.8-110.1 GB over 12,794-12,800 experts.
- Older boots (Oct 24-Nov 3, also in this log and `custom-dsv41-8081-4e.log`/`custom-dsv41-t54.log`) range 110.3-110.6 GB — the band is CPU-lane placement variance (12,712-12,800 experts), not drift.
- `--experts 5` (`logs/custom-dsv41-e5-8081.log`, Oct 28) measured `all` = 110.5 GB — same class; the expert count at 5-vs-4 does not grow `all` on this pack/card.
- Pack cross-check (`/home/svg/LLM/data/models/GSQ-RCO/Deepseek-v4.1-Flash/IQ2-XXS/`): 4 GGUF shards = 131,836 MiB + Engram-IQ4_NL = 105,472 MiB (103.0 GiB — the expert mass), mmproj 928 MiB; total 245,842 MiB ≈ 240 GiB, matching "~240 GiB streamed from page cache". Per-expert blob ≈ 110.8 GB / 12,800 ≈ 8.66 MB.
- There is no DS shadow store (launcher exports only `DS4_ATTN_SPLIT=1` and `STRATA_DS4_SLOT_DIR`; slot cache measured 339 MB on disk).

## 3. The arithmetic (one page = 2 MiB = 2,097,152 B; 1 GiB = 512 pages; engine prints are decimal GB)

Live `/proc/meminfo` (2026-11-07): `MemTotal: 193105792 kB` = 184.19 GiB = 197.66 GB; `HugePages_Total: 24369` (47.59 GiB = 51.10 GB locked out, runtime-aged arm of the 45056 target); `MemAvailable: 78018720 kB` = 78.02 GB; `Hugepagesize: 2048 kB`; `Shmem: 45.42 GiB` (≈ store 38 + slots ~5.8 + misc — consistent with the Qwen consumers below).

### 3.1 Minimum pool for DS's auto-huge rule

```
all (worst printed)          110.85 GB      (110.8 at 1 decimal, bounded above)
pages = ceil(110.85e9 / 2,097,152) = 52,860 pages   (52,860 × 2,097,152 = 110.855 GB)
+ 2 GiB slack                  = +1024 pages  → 53,884 pages
round up to whole GiB (512-page steps):
  105 GiB = 53,760 pages   < 53,884   ✗ (124 pages short of worst case + slack)
  106 GiB = 54,272 pages   ≥ 53,884   ✓  headroom = 113.82 GB − 110.85 GB = 2.97 GB
```

53,760 (105 GiB) sits **below** the worst-case minimum+slack — against an all-or-none cliff whose failure mode is "no copy at all". 106 GiB clears it with ~3 GB headroom ≈ 6× the observed `all` variance band (110.3→110.8). In-pool rounding waste is bounded at ≤ 2 MiB × 40 MoE layers ≈ 80 MiB (one rounded block per layer, `NH=1`), already inside the headroom.

### 3.2 Option (a): 45056 pages (88.0 GiB = 94.49 GB) — the regression

```
huge_arena: 94.49 GB (HugePages_Free) ≥ 110.85 GB?  NO → OFF → pool term = 0
can = avail − 8.59 GB (kRamMargin) + 0
  today (pre-reboot):      78.02 − 8.59 = 69.43 GB  < 110.85  → budget 0 → NO COPY (already broken)
  post-boot DS-boot window: Nov-5 boots saw avail 125.0-127.1 GB with no pool;
                            125.0 − 94.49 ≈ 30.5 GB avail → can ≈ 21.9 GB → NO COPY
DS then runs the CPU lane in place: file streaming through a ~20-30 GB cache window for a
110.8 GB working set; header precedent (ds4_experts_rule.hpp:86): copies beat in-place +10-25% decode.
```

Qwen at 45056 keeps working (its concurrent draw 80.4 GiB < 88 GiB) — option (a) trades DS away to keep the status quo that no longer exists.

### 3.3 Option (b): 54,272 pages (106 GiB) — coexistence checks

```
DS:            huge_arena ON (113.82 ≥ 110.85); pin ON (arena_all_huge stays true — headroom
               covers the ≤80 MiB rounding, so no mid-load fallback to arena_all_huge=false)
               can = avail − 8.59 + 113.82 ≥ 110.85  ⇔  avail ≥ 5.62 GB at DS load
               (boot-window estimate 12-31 GB → passes with 2-6× margin)
Qwen standing: concurrent pool draw = store 38 + largest single arena 39.4 (UD-IQ2_XS;
               GSQ complement 37.6 is smaller) + KV host shadow ~3 = 80.4 GiB = 41,165 pages
               ≤ 54,272 ✓ (pool spare 13,107 pages = 25.6 GiB)
               hugetlbfs store threshold ≥40,000 pages ✓ (54,272 ≥ 40,000 → store lands on 2 MiB)
UD-IQ4_XS:     arena 56 + store 38 = 94 GiB = 48,128 pages ≤ 54,272 ✓ fully pool-backed
               (≥49,152-page requirement exceeded)
General RAM:   184.19 − 106 = 78.19 GiB
               KV host shadows 23.4 + slots 5.8 + desktop ~6 = 35.2 GiB
               → page-cache room ≈ 43 GiB (≈ 26-43 GiB if today's measured non-shmem working
                 set 22.7 GiB is counted separately) — positive either way; organ reads no longer
                 touch general RAM, and DS's engram streaming stops once the copy is in.
Boot arming:   +9,216 pages over the proven 45056 boot design; pristine-buddy law unchanged
               (HOSTMEM-NO-REBOOT.md M5); if short, the documented compact/re-raise steps apply.
```

Why not larger (108-112 GiB): buys nothing for DS (the `can` cliff already clears at avail ≥ 5.6 GB), and each +1 GiB is page-cache room the 80-94 GiB Qwen packs need. Why engines can coexist with the pool at all: they don't run simultaneously (port law, `launch-custom-dsv41-8081.sh`: "Alternate occupant of 8081"), and "engines stop/restart = perfect pool re-arm" is the measured property that makes alternation clean. If DS and Qwen ever DO overlap, DS loses the copy (pool under 110.85 GB → OFF → `can` tiny) — keep the alternation law.

## 4. `huge_alloc` without `MAP_HUGE_2MB` — correct as-is

`src/dsv41/ds4_os.hpp:426-438`:

```cpp
inline void* huge_alloc(size_t bytes, size_t* got = nullptr) {
    const size_t hp = (size_t) 2 << 20;
    const size_t hb = (bytes + hp - 1) / hp * hp;                 // round up to 2 MiB
    void* m = ::mmap(nullptr, hb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    if (got != nullptr) *got = hb;
    return m;
}
```

- Plain `MAP_HUGETLB` selects the kernel's **default** huge size; this kernel reports `Hugepagesize: 2048 kB` (`/proc/meminfo:56`) — the default IS 2 MiB, so the flag-less call draws exactly from the 2 MiB pool the sysctl arms. `MAP_HUGE_2MB` would select the same hstate — a redundant no-op. **No fix needed on kernel 7.2.12 fc45.**
- The rounded length `hb` is what gets registered by `cudaHostRegister` (`ram_chunk_bytes.push_back(got)`, `ds4_model.cu:4756`; register at `:4841-4843`), so the pin covers whole pages. ✓
- Latent boundary (name, don't fix here): if a future cmdline ever sets `default_hugepagesz=1G`, plain `MAP_HUGETLB` silently retargets 1 GiB pages; the 2 MiB-rounded `hb` would then fail (`EINVAL` → `nullptr` → the `:4742-4752` ordinary-arena fallback — safe degradation, pool unusable by DS). A one-line hardening for some future pass: pass `MAP_HUGE_2MB` explicitly.

## 5. Recommendation and the post-boot DS battery

**Set `vm.nr_hugepages=54272` in `/etc/sysctl.d/91-strata-hugepages.conf` (one line, plus its comment: this digest is the "DS boots sized at their ladder legs" decision that file defers to).** Do it before the reboot; at 45056 the first DS boot after rebooting lands in-place and the bank numbers will lie about the huge-page change.

Expected witnesses in `logs/custom-dsv41-8081.log` (first DS boot after reboot):

1. `ds4: the reserved huge-page pool (113.8 GB free) holds the whole copy (110.8 GB): the arena on 2 MB pages (DS4_COPY_HUGETLB=0 keeps the ordinary arena)` — the auto-ON line (`ds4_model.cu:4670-4672`).
2. `ds4: --mmap-experts auto: the CPU lane's experts (110.8 GB) fit the RAM this process can get (~118 GB beside 9 GB for the rest)` — `can` now includes the pool term (the ~118 estimate: avail ~12.7 + pool 113.8 − margin 8.59).
3. `ds4: 127xx of the CPU lane's 127xx experts copied into RAM (110.x GB, 1 NUMA part) in N s`.
4. NEW for this box: `ds4: the copied experts pinned in X.X s (read-write): the prompt path DMAs them in place` (`:4846-4847`) — pin first fires here, via the `arena_all_huge` branch.
5. `/proc/meminfo` after load: `HugePages_Free` 54,272 → ≈ 1,370 (106 GiB − ~103.3 GiB consumed incl. rounding).

Lanes and measurements (vs standing banks **52.3 solo / 51.6 pair / 57.5 conc** decode t/s, `--experts 4`):

- **Decode (mechanism: TLB).** The standing arena is `node_map` + `MADV_HUGEPAGE`-class THP (`ds4_os.hpp:442-458`) on an aged buddy — effectively 4 KB (HOSTMEM doc: `AnonHugePages` 120 MB of 133 GiB; `pinned.cu:262`: ~750 TLB misses per 3 MB expert). 2 MiB backing cuts expert-lane TLB misses ~512×; the Qwen precedent is +4.9-7.2% solo on DRAM-ceiling-bound lanes, and the DS CPU lane sits at 93-96% of the plain-read ceiling (46-50 GB/s). Expect solo ≈ 55-56 t/s if the lane behaves like Qwen's; **fail line: anything ≤ 52.3**.
- **Prefill (mechanism: pin).** With the pin ON, the per-chunk expert stream DMAs from the arena in place instead of the staging ring (staged 6.8 GB/s vs 24/28 GB/s probes, `:4822-4829`). Measure warm-8K ppt AND the second prompt of the process specifically (the 11.9 s → 30.4 s erratic-pin signature was a 4 KB pinned arena; on 2 MiB pages the second prompt should match the first — instability means a partial pool ate the pin).
- **DS4_PROF**: `cpu_experts` stage ms/token before/after; DRAM read rate during decode.
- **Conc/pair legs** against the same-hour controls (the standing banks), plus a long-context prefill leg.
- **Quality gate**: pin and huge pages move bytes, not values — run the standing greedy-ids/KL gate anyway (T53 precedent).
- **Alternation drill**: DS stop → pool back to 54,272 free → Qwen boot (organs draw ≤ 80.4 GiB; store lands on `/mnt/strata-huge`) → stop → DS boot again; confirm witnesses 1-4 on every leg.
- **Fail lines to grep for** (either means the pool didn't hold): `the 2 MB huge-page arena was asked for but the pool ran out` (`:4745`) and `the CPU lane's experts (... GB) do not all fit the RAM this process can get` (`:4683`).

Boundaries of this recommendation: sized for THIS pack (`all` ≤ 110.85 GB) and `--experts 4/5`; a denser pack or a future `all` growth past ~113 GB re-opens the decision. Assumes the standing alternation law (no DS+Qwen overlap) and a whole boot-time arm; a runtime top-up after boot must follow the compact-and-step procedure in the sysctl conf comment.
