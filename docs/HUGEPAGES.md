# The reserved-page memory architecture (hugepages)

**Status:** engine capability shipped; the standing-pool era RETIRED 2026-11-09 — a 106 GiB pool
armed at every boot was the biggest proven tax of the regression era (see *Arming*). The organs
stay; the standing state is gone. Owner: `src/core/pinned.cu`
(`reserve()`, `announce_page_class`), `include/strata/platform/hugepage.hpp` (`huge_alloc`,
`huge_pool_covers`), `src/core/shadow_experts.cpp` (the memfd store), `src/core/expert_source.cpp`
(the paced pool path). Research: `.docs/research/digest/2026-11-07-hostmem-*.md` (three digests)
and `2026-11-07-preboot-*.md` (six more). Evidence: `campaign/HOSTMEM-NO-REBOOT.md`,
`campaign/results/hostmem-day1-verdict.md`.

## The problem it removes

MoE models served on one GPU stream their expert weights from host RAM at memory speed: the CPU
expert lane is DRAM-bandwidth-bound, so *how* that RAM is paged is a first-order performance term.
With the default 4 KiB pages, every 3 MB expert costs ~750 TLB lookups on the streaming lane
(`pinned.cu`), and — worse — the kernel can only build a 2 MiB page where it finds 512 *physically
adjacent* free pages. Engines that allocate and release tens of GiB per start scatter physical
memory until, after days of uptime, no contiguous runs remain (measured on this box: 71% of kernel
compaction attempts failing, `AnonHugePages` at 120 MB of 133 GiB resident). The hot pools silently
fall back to 4 KiB and the lanes lose 10–25%. The only cure was a reboot — because a reboot was the
only thing that re-shelved the library.

This is not a quirk of our engine: it is why PostgreSQL (`huge_pages`), Oracle (SGA), DPDK
(mempools), vLLM (`cpu_hugepage_block_size`) and SGLang (`SGLANG_HUGEPAGE_SIZE`) all reserve
explicit huge-page pools at boot. The transparent-THP alternative ("soft" binding) is the measured
disease: best-effort binding degrades exactly the same way on a fragmented box.

## The architecture, in one paragraph

The kernel carves a **reserved pool of 2 MiB pages** (a fixed count; armed per-session at runtime
on this estate — see *Arming*). The engine draws its large
streaming organs from that pool with `MAP_HUGETLB` — whole pages in, whole pages out. Pool pages
cannot fragment (they are never split), cannot swap, and cannot age. Engine stop returns every page
to the pool intact; engine start re-arms perfectly. Nothing accumulates across restarts, so nothing
ever needs a reboot to clean up. Reserved-but-idle pages are free for nothing else — the pool is a
dedicated shelf — which is a deliberate trade: the shelf is sized to the model set, and the rest of
the RAM serves the desktop, page cache, and everything else normally.

## The organs, and how each is backed

| organ | size class | path | backing |
|---|---|---|---|
| expert arena (UD packs) | 39–56 GiB | `PinnedArena` → `reserve()` (`pinned.cu`) | `MAP_HUGETLB` first; guarded THP fallback; loud banner either way |
| pinned cache complement (GSQ) | ~38 GiB | `expert_source.cpp` paced path → `huge_alloc` | pool-first when `huge_pool_covers()`; `cudaHostAlloc` one-shot otherwise |
| shadow expert store | ~32–38 GiB | `shadow_experts.cpp` arm() | **hugetlb-backed memfd** (`memfd_create(MFD_HUGETLB)`), populated from the pack file at arm (~2–3 s at memcpy speed), released with the process; falls back to the tmpfs/disk tiers |
| KV host shadow, session slots | < 1 GiB each | gated OUT of the pool by the priority rule | ordinary pages (status quo; `STRATA_POOL_SMALL_ARENAS=1` opts in when slack exists) |

**Pool priority** (`reserve()` and `huge_alloc` both): only organs ≥ 8 GiB draw from the pool by
default. Without this, ~90 small per-layer allocations (~23 GiB total) win the race and starve the
big streaming organ — measured, then fixed. Small mappings keep the ordinary paths.

**The measured composition proof:** `cudaHostRegister(PORTABLE)` over pool-backed regions holds
(5/5 boots at 37–39 GiB on driver 615.x; production-proven upstream by SGLang HiCache and FlexKV).
The registration must come *after* the 2 MiB pages are faulted — pre-faulted registration is
~130 GiB/s, lazy-fault registration ~6× slower.

## Boot truth: every organ announces its page class

Every ≥ 1 GiB arena prints its class at boot:

```
strata: ARENA PAGES: 2 MiB (37.6 GiB arena, large-page backed)
strata: hugepages: the expert complement: 37.59 GiB of 37.59 GiB in 2 MiB pages (HUGETLB POOL; ...)
strata: STRATA_SHADOW_EXPERTS=...: shadow Q2_0 expert store mapped - 31.64 GiB; 2 MiB pool pages (memfd, populated in 3.4 s)
```

A degraded boot says so, in the same voice: `ARENA PAGES: 4K - DEGRADED CLASS (...)`. Benches can
refuse to measure a silently-4K boot: `STRATA_ARENA_PAGES=strict` exits 70 instead of serving
degraded. Silent fallbacks are the bug class this campaign killed; loudness is the invariant.

## Sizing the pool

The reservation must cover **the largest single-draw configuration** of the model set, plus slack,
within the RAM left for everything else (rule: never reserve past the point where general RAM
cannot hold the OS + page cache + desktop + KV). Worked example (this box, 184 GiB RAM):

- DS CPU-lane expert copy, all-or-none: 110.85 GB → the binding constraint
- Qwen worst concurrent draw: store + largest arena + KV ≈ 80.4 GiB
- UD-IQ4_XS whole config: 56 + 32 ≈ 88 GiB
- chosen: **54272 pages (106 GiB)** = DS + 2 GiB slack; ~78 GiB general remains

All-or-none consumers (DS's `DS4_COPY_HUGETLB` arms only when `HugePages_Free ≥ all`) make slack
non-optional: a pool 124 pages short of the worst case silently disarms the whole mechanism for
that model.

## Arming (per DS session, runtime-only — the standing era is over)

The standing install this doc originally taught (grubby args + `/etc/sysctl.d/91`, armed at every
boot) was **convicted 2026-11-09**: a 106 GiB pool at 50–60% of RAM taxes *every* workload on the
box — measured −12–15% on this engine and −20–22% on a stock engine that never touched a hugepage
(capacity loss, not page class). The conf was deleted. The doctrine since:

- **Standing state: RELEASED.** No hugepages kernel args, no sysctl.d file. Strip any legacy boot
  entry: `sudo grubby --update-kernel=ALL --remove-args="default_hugepagesz=2M hugepagesz=2M hugepages=54272 hugetlb_free_vmemmap=on"`.
  A reboot lands released — that is correct, not a gap.
- **DS session arm: runtime-only** — `sudo bash campaign/results/ds-pool-topup-block.sh`
  (engines down → drop cache → compact → raise in steps → **READ BACK OR FAIL**; runtime
  reservation is silently partial on a fragmented buddy — measured 24369 of 45056 — which is what
  the read-back law exists for).
- **Release after the session:** `echo 0 > /proc/sys/vm/nr_hugepages`, or just reboot (clean by default).
- Qwen/UD rotations run released by design — the record class (231/225/315) is a released-box
  number; the organs fall back to THP-madvise/4K without the pool.
- `memlock unlimited` (`limits.d` + the user-service drop-in) stays either way — the working-set
  locks are engine-level, pool or no pool.

## Rollbacks (every one is a single switch)

- `STRATA_HUGEPAGE_ARENAS=0` — the complement back to one-shot `cudaHostAlloc`
- `STRATA_SHADOW_MEMFD=0` — the store back to file mapping
- `STRATA_NO_LARGEPAGES=1` — no pool attempts anywhere (the A/B switch every battery uses)
- `STRATA_POOL_SMALL_ARENAS=1` — small arenas into the pool when slack exists
- delete the cmdline args — the pool never arms; the engine serves 4 KiB exactly as before

## What it is not

Not a capability change (context windows, packs, vision, spec, cache: untouched — law 16). Not a
per-app tax (one pool per device; every engine and the store draw from the same shelf; the
precedent systems run this way for decades). Not dynamic — by design: both dynamic variants (THP
soft-binding and `nr_overcommit_hugepages` surplus) fail the same contiguity requirement at
runtime, and the surplus flavor fails mid-serve. The ledger was never the problem; 512× fewer
ledger entries is.
