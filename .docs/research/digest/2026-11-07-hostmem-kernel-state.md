---
type: research-digest
date: 2026-11-07
topic: state of the art for keeping multi-GB, long-lived, sequentially-streamed host pools TLB-stable and physically coalesced on a 24/7 Linux box, online, without reboot
kernel: 7.2.12-300.fc45.x86_64 (Fedora 45)
method: primary sources only — kernel admin-guide docs, LWN.net, man pages, LKML pulls/commits, plus live /proc + /sys measurements on this box
---

# Host memory management for streamed host pools: kernel state of the art (2026-11-07)

## 0. Method and applicability grading

Cited features carry a merge-version grade. Confidence grades: **High** (primary kernel doc or measured on this box), **Medium** (LWN/commit evidence, semantics clear, magnitude uncertain), **Low** (engineering estimate or unverified-on-7.2).

Applicability to 7.2.12: every load-bearing mechanism cited here (anon mTHP 6.8, shmem mTHP 6.13, MADV_COLLAPSE 6.1, proactive compaction 5.13, large-folio swap-in 6.13, ext4 large folios 6.16) predates this kernel. Verified live: `/sys/kernel/mm/transparent_hugepage/hugepages-2048kB/shmem_enabled` exists with `[inherit] within_size advise` — the 6.13 shmem-mTHP surface is present on this box. The newest items (khugepaged mTHP collapse, ~6.18/6.19-era late 2025; CONFIG_READ_ONLY_THP_FOR_FS removal, 2026) are graded separately where cited.

## 0.1 Measured state of this box (2026-11-07, degraded regime)

| Signal | Value | Meaning |
|---|---|---|
| Shmem | 112,557,336 kB (~107 GiB) | placement cache in tmpfs |
| ShmemHugePages / ShmemPmdMapped | 0 / 0 | 100% of the store on 4K pages |
| AnonPages / AnonHugePages | ~17 GiB / ~120 MB | anon pools also on 4K |
| THP enabled / defrag | madvise / madvise | nothing in userspace calls MADV_HUGEPAGE |
| shmem_enabled | never | tmpfs huge pages off |
| thp_fault_alloc / thp_fault_fallback | 1.24M / 1.11M | 47% of anon THP fault attempts failed and fell back |
| compact_stall / compact_fail / compact_success | 1.66M / 1.18M / 0.48M | 71% of direct compaction stalls fail |
| pgsteal_proactive | 0 | proactive compaction has reclaimed nothing (default 20) |
| SwapTotal / Used (zram) | 8 GiB / 6.4 GiB | zram near full |
| ulimit -l (memlock) | 8192 kB | blocks mlock-scale strategies until raised |
| hugepages-2048kB/shmem_enabled | inherit | mTHP shmem knob present, inheriting "never" |

Reading: the store is 4K-paged because both gates are closed (`enabled=madvise` with no callers, `shmem_enabled=never`). Fragmentation counters show the compaction machinery losing (71% stall-fail) while the 107 GiB of 4K backing keeps ~215 MiB of unmovable page-table pages scattered through physical memory. This is the degradation signature; fresh boot resets it by re-faulting the pool early in uptime.

---

## 1. THP for anonymous memory: madvise vs always

**Answer:** Keep `enabled=madvise` and make the engine opt in per-VMA with `madvise(MADV_HUGEPAGE)`. `always` is not justified on this box.

- `madvise` restricts THP to MADV_HUGEPAGE regions "to avoid the risk of consuming more memory resources" (kernel admin guide). [High](https://docs.kernel.org/admin-guide/mm/transhuge.html)
- MADV_HUGEPAGE (since 2.6.38): marks the VMA; kernel faults huge pages where naturally aligned and khugepaged backfills by scanning. Works on anonymous, shmem, and (constrained) file-backed memory since 5.4. [High](https://man7.org/linux/man-pages/man2/madvise.2.html)
- defrag=madvise (this box's setting, also kernel default): direct reclaim/compaction on allocation failure only inside MADV_HUGEPAGE regions; other allocations fall back without stalling. defrag=always makes every THP fault candidate stall in direct compaction — that is the classic latency-spike pathology. [High](https://docs.kernel.org/admin-guide/mm/transhuge.html)
- Streaming effectiveness: 2MB pages cut TLB translations 512x and remove per-4K-page fault/rmap work. For a pool faulted once and streamed repeatedly, fault-time THP is the effective moment; khugepaged backfill later. Mechanism [High]; magnitude arch-dependent [Medium].
- Known pathologies, all documented or covered by LWN: (a) memory bloat — "an application may mmap a large region but only touch 1 byte of it, in that case a 2M page might be allocated" [High]; (b) defrag=always fault stalls [High]; (c) khugepaged/kcompactd CPU wars on fragmented boxes — Red Hat BZ 1694305 (kcompactd0 100% CPU), VMware-vs-kcompactd reports [Medium](https://bugzilla.redhat.com/show_bug.cgi?id=1694305); (d) the mTHP-era consensus: David Hildenbrand argued at LPC 2024 that automatic (`always`) huge-page use still risks server regressions; kernel mitigations landed instead — underused-THP splitting (`shrink_underused`, present on this box) and per-size sysfs controls. [Medium](https://lwn.net/Articles/975938/)
- Box evidence: THP=madvise with no MADV_HUGEPAGE callers gives AnonHugePages 120 MB of 17 GiB anon; 47% of the THP attempts that did happen failed under fragmentation. [High, measured]
- zram interaction: THP/mTHP swap-out works as multi-page swap entries; large-folio swap-in and zram large-folio compression landed across 6.12–6.13 (Andrew Morton's 6.13 MM pull; LWN swap rework coverage). [Medium-High](https://lkml.iu.edu/2411.2/02192.html) (https://lwn.net/Articles/1016136/)
- Setting changes affect only future faults; engines must restart to pick them up (doc: "only affect future behavior"). [High]

## 2. THP for shmem/tmpfs: shmem_enabled semantics, huge=, mTHP status

**Answer:** This is the primary lever for the 107 GiB store and it is mature. Three layers: global `shmem_enabled`, per-mount `huge=`, per-size mTHP knobs (6.13+).

- `shmem_enabled` values (admin guide, verbatim semantics): `always` — allocate huge on every new page, PMD first with fallback; `within_size` — "only allocate huge page if it will be fully within i_size", honors madvise; `advise` — only on MADV_HUGEPAGE; `never` — off (MADV_COLLAPSE still works); `deny` — "for use in emergencies, to force the huge option off from all mounts"; `force` — force on everywhere, testing artifact. [High](https://docs.kernel.org/admin-guide/mm/transhuge.html)
- `huge=` tmpfs mount option: `never` (default), `always`, `within_size`, `advise`. Per-mount choice wins over the global default; only global `deny`/`force` override mounts. So this box can keep `shmem_enabled=never` system-wide and turn huge pages on just for the placement-cache mount. [High](https://docs.kernel.org/filesystems/tmpfs.html)
- Layering/maturity: PMD-sized huge tmpfs landed 4.8 (2016, Hugh Dickins series). mTHP (multi-size, 16k–1M under PMD) for anonymous shmem landed **6.13** with `thp_shmem=` boot parameter and per-size knobs `hugepages-NkB/shmem_enabled` (values always/inherit/never/within_size/advise; deny/force dropped per-size). Verified present on this box. [High](https://lwn.net/Articles/974630/)
- LSFMM 2024 caveats before the mTHP landing: swap path split large shmem folios on reclaim; Hildenbrand warned shmem "works in a weird space" between anon and page cache and large folios could waste scarce memory; an unexplained tmpfs-vs-XFS perf anomaly where large folios made tmpfs *worse*, suspected compaction cost during large-folio allocation. Treat `within_size` (bounds waste to fully-used ranges) as the safer default for a placement cache. [Medium](https://lwn.net/Articles/974630/)
- Cost of 107 GiB in 4K pages vs 2 MB folios (measured base 112,557,336 kB):
  - Page count: 28.1M 4K pages vs ~55K 2MB folios — a 512x reduction in managed units, rmap entries, and TLB translations.
  - Page tables: 4K mapping needs one 4 KiB PTE page per 2 MiB → ~55K PTE pages ≈ **215 MiB** of kernel page-table memory. PTE pages are unmovable scatter; this alone poisons 2MB-block availability across the box and is the leading suspect behind the 71% compact_fail rate. Removing it via PMD mapping eliminates both the memory and the scatter. [High on arithmetic; Medium on the causal fragmentation share]
  - TLB at ~50 GB/s sequential: 4K paging streams 12.2M new pages/s; an L1 dTLB (~64–96 entries) covers well under 0.5 MiB, so essentially every 4K page is an L1 dTLB miss (L2 dTLB hit or page walk). 2MB paging drops that to ~24K misses/s. Expect a single-digit-percent throughput gain on a DRAM-bandwidth-bound lane — real but secondary to removing fragmentation-driven fallback behavior. [Low-Medium, engineering estimate; no kernel citation for magnitude]
- Online path for shmem huge pages: MADV_COLLAPSE supports "private anonymous pages, shmem pages, and file-backed pages" and ignores enabled/huge= settings (still respects VM_NOHUGEPAGE). Landed 6.1 (Zach O'Keefe; LWN coverage + man page). Restriction: PMD-size only, no mTHP collapse. [High](https://man7.org/linux/man-pages/man2/madvise.2.html) (https://lwn.net/Articles/892727/)
- khugepaged also collapses file/shmem to PMD THP ("khugepaged currently only searches for opportunities to collapse file/shmem to PMD-sized THP"), gated by khugepaged/defrag. mTHP-sized collapse for shmem is newer work (~6.18/6.19-era, Baolin Wang series) — likely present on 7.2.12, unverified. [Medium](https://docs.kernel.org/admin-guide/mm/transhuge.html)

## 3. Proactive compaction (vm.compaction_proactiveness)

**Answer:** Real and mature (5.13+), but it is a background pressure-reducer, not a guarantee. On this box it has effectively not been doing useful work (pgsteal_proactive 0) at default 20 while fragmentation accumulated.

- Design (LWN 2020, Nitin Gupta): per-node kcompactd computes a fragmentation score (free-area coverage at PMD order) and compacts in the background toward a target derived from the tunable; with proactiveness=20 the target band is 80–90. Backs off when kswapd runs, locks are contended, or a round shows no progress. [High](https://lwn.net/Articles/817905/) (author's writeup: https://nitingupta.dev/post/proactive-compaction)
- Measured effect in author's tests (1 TB 2-node, fragmented): 95th-percentile huge-page allocation latency 33,799 us → 429 us (~78x); 700 GB Java heap alloc 27 min → 4 min. [High on reported numbers; Medium on transferability]
- Does it keep THP success high on long-uptime churn boxes? It helps when the memory that scatters is *movable* (anon with swap slots, page cache, shmem). It cannot fix scatter from unmovable kernel allocations (slab, page tables, pinned buffers). Field reports show the failure mode: kcompactd looping at 100% CPU when fragmentation persists (Red Hat BZ 1694305; VMware tug-of-war reports; mitigations include lowering proactiveness or disabling THP). [Medium](https://bugzilla.redhat.com/show_bug.cgi?id=1694305)
- Tuning guidance (sysctl doc): range [0,100], default 20; nonzero values trigger immediate compaction; 0 disables; >80 "will trigger more often, but reduce fragmentation by a smaller amount"; "be careful ... extreme values like 100"; compaction "could also lead to latency spikes in unsuspecting applications". [High](https://docs.kernel.org/admin-guide/sysctl/vm.html)
- Verdict for this box: raise to ~40–50 and watch kcompactd CPU + `compact_daemon_*` vmstat counters; it complements, never replaces, restart-time huge-page allocation. [Medium]

## 4. On-demand defragmentation: /proc/sys/vm/compact_memory

**Answer:** A valid online maintenance trigger; present on this box (`/proc/sys/vm/compact_memory`).

- Semantics: "When 1 is written to the file, all zones are compacted such that free memory is available in contiguous blocks where possible" — motivated explicitly by huge-page allocation. Per-node variant exists (`/sys/devices/system/node/nodeN/compact`). [High](https://docs.kernel.org/admin-guide/sysctl/vm.html)
- What it can reclaim/move: migratable pages (anon with swap space, page cache, shmem, mlocked pages if `compact_unevictable_allowed=1`, the default — "the default behavior is to let unevictable pages be moved"). What it cannot: unmovable kernel memory — slab, page tables, pinned/PFN mappings. On a box with 133 GiB resident, those unmovable scatter points cap how many 2MB blocks it can free. [High on doc semantics; Medium on the achieved fraction, workload-dependent]
- Duration: the write is synchronous (compaction runs inside the syscall). No published bound; expect seconds to minutes on 184 GiB under resident pressure — measure first, run it from a timer, not from the request path. [Low-Medium, no citation]
- What it restores: 2MB availability for *future* allocations. Existing 4K mappings stay 4K; converting the resident pool needs khugepaged or MADV_COLLAPSE. Pair: compact_memory (make blocks) → engine restart or MADV_COLLAPSE (use blocks). [High]

## 5. hugetlbfs reserved pool

**Answer:** Strongest guarantee on the table, mature since forever, but static. Reserved at boot it is the only mechanism that is fragmentation-proof by construction.

- Boot vs runtime: `hugepagesz=2M hugepages=N` at boot "is the most reliable method of allocating huge pages as memory has not yet become fragmented" (admin guide, verbatim). Runtime writes to `/proc/sys/vm/nr_hugepages` attempt allocation and can partially fail: "If the kernel is unable to allocate huge pages from some nodes ... it will attempt to make up the difference by allocating extra pages on other nodes"; nodes without contiguous memory "will be silently skipped". So runtime growth on a 6-day-fragmented box is best-effort, not guaranteed — allocate the whole pool at boot or accept partial results (the sysctl returns what it got). [High](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)
- `nr_overcommit_hugepages`: surplus pages grown on demand from the normal pool, freed back when unused. This reintroduces fault-time allocation failure (and OOM risk) — it trades guarantee for elasticity. [High, same doc]
- Unswappable by construction: "Huge pages cannot be swapped out under memory pressure." They never touch zram; reclaim cannot split them; contiguity is definitional (each page IS a contiguous 2MB block). Migration exists only for admin operations (hotplug/CMA/NUMA balancing). [High]
- mlock: not needed — hugetlb pages are outside vmscan entirely; locking is redundant (except partially-mapped-THP semantics which do not apply here). [High]
- memfd_create(MFD_HUGETLB) (4.14; MFD_ALLOW_SEALING combo 4.16; MFD_HUGE_2MB/1GB selectors): hugetlbfs-backed memfd, no mount needed — the clean engine-side primitive for pool-style allocations. EPERM without CAP_IPC_LOCK unless in hugetlb_shm_group. [High](https://man7.org/linux/man-pages/man2/memfd_create.2.html)
- Costs: a 107 GiB reservation pins 58% of 184 GiB regardless of engine state (that is the guarantee); engines restart fine against a persistent reservation; sizing must cover peak, and misjudgment causes OOM elsewhere because the reserved pages are unusable for anything else. Requires engine cooperation (memfd/mmap MAP_HUGETLB) or an interposer. No cgroup-memory reclaim safety net exists for it — the hugetlb controller bounds usage, it does not rescue you. [High on mechanics; Medium on ops judgment]

## 6. File-backed alternative: MAP_SHARED read-only + page cache + mlock

**Answer:** Architecturally clean (read-only weight files dedupe across restarts — the page cache survives engine restarts), but the large-folio guarantee today is weak. Fine as a complement, not the primary architecture.

- mlock/MAP_LOCKED semantics: mlock moves pages to the Unevictable LRU — "to hide these folios from vmscan" (kernel doc); mlocked file/page-cache pages become unevictable too. MAP_LOCKED is weaker: "the mmap() call doesn't fail with ENOMEM if this fails ... the semantic is not as strong as mlock(2)" — it can silently lock less and leave major faults. Use explicit `mmap()+mlock2()` and raise RLIMIT_MEMLOCK (this box: 8 MiB; set systemd `LimitMEMLOCK=` on the engine unit). [High](https://docs.kernel.org/mm/unevictable-lru.html) (https://man7.org/linux/man-pages/man2/mmap.2.html)
- Do locked file pages get large folios? Not guaranteed:
  - Historical file-THP path: khugepaged-only collapse of read-only file mappings, gated by CONFIG_READ_ONLY_THP_FOR_FS (marked experimental, typically off in distro kernels) and requiring the mapping to be executable and the file not open for write. [High](https://cateee.net/lkddb/web-lkddb/READ_ONLY_THP_FOR_FS.html) (https://blogs.oracle.com/linux/transparent-hugepages-for-text-mappings)
  - 2026 work proposes removing the config gate and allowing writable-file THP — newest-item grade: not verifiable on 7.2.12 without checking the config. [Medium](https://lwn.net/Articles/1070484/)
  - Page-cache large folios themselves: XFS 5.18, ext4 regular files 6.16 — readahead allocates multi-page folios opportunistically, but there is no PMD-order guarantee at fault time for arbitrary read mappings. [Medium](https://www.phoronix.com/news/EXT4-Large-Folios-Performance) (https://lwn.net/Articles/1009548)
- Thrash: working set > page cache without mlock → refault storm (and re-read from disk). With mlock → no thrash, but the pages are hidden from reclaim; oversubscription shifts pressure to everything unprotected (pair with cgroup limits). mlocked pages remain migratable by compaction (default), which helps defrag. [High on semantics; Medium on behavior under oversubscription]
- Distinct advantage vs tmpfs: file pages survive engine restarts (no re-fault, no re-read) and dedupe naturally if two engines map the same weights. For a 107 GiB cache that is *rewritten* by the engine (tmpfs placement cache), this advantage does not apply — the tmpfs path is the right shape there; MAP_SHARED+mlock is the right shape for immutable on-disk weight files. [Medium-High, reasoned]

## 7. cgroup v2 protection for the serving pool

**Answer:** Yes — memory.min/low stop the kernel from reclaiming/swap-cleaning the pool; memory.swap.max=0 opts the pool out of zram entirely. It does not stop compaction or khugepaged scanning (and should not).

- memory.min: "the cgroup's memory won't be reclaimed under any conditions" within the effective boundary; only global exhaustion past all unprotected memory leads to OOM. Overcommitting min across cgroups is documented as a constant-OOM anti-pattern. [High](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- memory.low: best-effort; reclaim sweeps protected memory last and proportionally above the boundary. Use low as the primary dial, min only for the true irreducible core. [High]
- memory.swap.max=0: "anonymous memory of the cgroup will not be swapped out." zram is the swap device, so this is a hard opt-out of zram for the pool: no pool pages pushed into an already-full zram, no THP split-to-4K churn on swapout, and zram stays reserved for everything else. The docs also note zswap.writeback semantics differ — not relevant here. [High]
- Interaction with the degradation: this box's zram is at 6.4/8 GiB and pgscan counters are enormous (5.3G kswapd scans) — scan pressure is landing on the pool today. Protection moves that pressure to genuinely reclaimable memory or, if none exists, makes the shortage explicit (memory.events, OOM) instead of silently degrading the streamed lane. [High on mechanism; the pairing with correct sizing is ops judgment, Medium]
- Compaction/khugepaged are not reclaim: memory.min does not exempt pool pages from migration by compaction (they are movable) or from khugepaged collapse scans. Both are desirable here. [High]

## 8. The honest comparative for this workload

Workload shape: huge file-shaped store (rewritten by the engine, ~107 GiB tmpfs) + anon pools, cycled several times daily, streamed at the DRAM ceiling, 24/7 no-reboot target.

| Architecture | Guarantee of 2MB residency | Maturity | Ops cost | Fits cycled pool? |
|---|---|---|---|---|
| tmpfs huge= / shmem mTHP (advise or within_size) | High when blocks available; backed by khugepaged + MADV_COLLAPSE repair | PMD shmem THP 4.8, mTHP 6.13 | Low | Yes — faults fresh at every restart |
| hugetlbfs boot reservation (memfd MFD_HUGETLB) | Absolute (unswappable, contiguous by construction) | Decades-old | High (static pin, sizing discipline) | Yes, but no elasticity |
| page-cache MAP_SHARED + mlock | Weak (folio size not guaranteed; file-THP config-gated) | Large-folio infra mature, PMD file THP not | Medium (memlock limits, sizing) | Only for immutable weight files |
| THP=always system-wide | High attempt-rate, no guarantee; bloat + stall risk | Mature but consensus says not for automatic use (LPC 2024) | Low | Poor fit with full zram + mixed workloads |

**Why tmpfs huge= wins the primary slot:** the degradation on this box is caused by the pool being *allocated* as 4K in the first place (28.1M pages, ~215 MiB of unmovable PTE pages, 512x TLB amplification). Every restart re-faults the whole store; if the mount says `huge=within_size` (or the engine calls MADV_HUGEPAGE under `huge=advise`), each restart becomes a fresh attempt to build the pool on 2MB folios — the daily cycle converts from a fragmentation source into a defrag opportunity. Online repair for a pool that already degraded is MADV_COLLAPSE (engine-issued; process_madvise does not carry MADV_COLLAPSE cross-process) plus compact_memory to supply blocks. [High on mechanism; Medium on remount semantics — apply `huge=` via remount or at next mount+restart]

**Why not THP=always:** it fixes anon, which is the small part of this box (17 GiB anon vs 107 GiB shmem), and it imports bloat and defrag-stall risk onto a box whose zram is already nearly full and whose compact_fail rate is 71%. Per-VMA madvise achieves the same coverage for the pools with none of the blast radius. [Medium-High]

**Why not hugetlbfs as primary:** nothing in the kernel can take 107 GiB back from a reservation, and the pool composition changes with engine releases. Correct use: reserve the provably-constant core (if any survives sizing analysis) at boot with `hugepagesz=2M hugepages=N`, exposed to the engine via memfd_create(MFD_HUGETLB). Keep `nr_overcommit_hugepages` at 0 — its surplus mode reintroduces exactly the fault-time failure this architecture exists to remove. [Medium-High]

**Page-cache+mlock:** adopt for immutable on-disk weight files if/when the engine reads them directly (restart-surviving, dedup across engines), with mmap()+mlock2() and raised LimitMEMLOCK. Do not base the 107 GiB placement cache on it — tmpfs huge= gives the same fault-fresh allocation with an actual PMD guarantee path. [Medium]

### Ranked no-reboot architecture (recommended)

1. **Placement cache (107 GiB tmpfs): mount `huge=within_size`** (or `advise` + engine MADV_HUGEPAGE on the mapping). Keep global `shmem_enabled=never` as the safety default; optionally set `hugepages-2048kB/shmem_enabled=within_size` instead for box-wide effect. Verify ShmemPmdMapped ~107 GiB after the first engine restart.
2. **Anon pools: THP=madvise (keep) + defrag=madvise (keep) + engine MADV_HUGEPAGE on pool VMAs.** AnonHugePages should rise from 120 MB toward the pool working set; watch thp_fault_fallback.
3. **Online maintenance:** vm.compaction_proactiveness 40–50 (watch kcompactd CPU); systemd timer running `echo 1 > /proc/sys/vm/compact_memory` after engine restart waves (measure duration first); MADV_COLLAPSE pass inside the engine after pool load. This triple is the online substitute for the reboot: compact supplies 2MB blocks, restart/MADV_COLLAPSE re-page the pool onto them.
4. **cgroup protection:** engine slice `memory.low` ≈ steady-state pool size (min only for the irreducible core), `memory.swap.max=0`. Confirm via memory.events that pressure events stop touching the pool.
5. **Optional endgame for the constant core:** boot-time hugetlbfs reservation + memfd MFD_HUGETLB in the engine; only for the subset that sizing analysis proves constant.
6. **Immutable weight files (if mapped directly):** MAP_SHARED read-only + mmap/mlock2 with LimitMEMLOCK=infinity on the unit; accept folio-size as best-effort today.

Everything above is online and reversible; no step requires a reboot. The reboot-equivalent state (pool on 2MB folios, low compact_fail) is reachable by: set tunables → compact_memory → restart engines → verify counters.

### Verification commands

- `/proc/meminfo`: ShmemPmdMapped (target ~107 GiB), AnonHugePages, FileHugePages.
- `/proc/vmstat`: thp_fault_fallback rate, compact_fail delta per day, pgsteal_proactive > 0 after tuning, khugepaged full_scans advancing.
- `/sys/kernel/mm/transparent_hugepage/hugepages-2048kB/stats` per-size mTHP stats.
- Lane throughput A/B: pre/post across one restart cycle.

## Sources

- THP admin guide: https://docs.kernel.org/admin-guide/mm/transhuge.html
- tmpfs doc (huge=): https://docs.kernel.org/filesystems/tmpfs.html
- hugetlbpage admin guide: https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
- vm sysctl doc (compact_memory, compaction_proactiveness): https://docs.kernel.org/admin-guide/sysctl/vm.html
- Unevictable LRU: https://docs.kernel.org/mm/unevictable-lru.html
- cgroup v2: https://docs.kernel.org/admin-guide/cgroup-v2.html
- madvise(2) (MADV_HUGEPAGE, MADV_COLLAPSE): https://man7.org/linux/man-pages/man2/madvise.2.html
- mmap(2) (MAP_LOCKED): https://man7.org/linux/man-pages/man2/mmap.2.html
- memfd_create(2) (MFD_HUGETLB): https://man7.org/linux/man-pages/man2/memfd_create.2.html
- LWN proactive compaction: https://lwn.net/Articles/817905/ ; author writeup: https://nitingupta.dev/post/proactive-compaction
- LWN shmem/tmpfs large folios (LSFMM 2024): https://lwn.net/Articles/974630/
- LWN mTHP background: https://lwn.net/Articles/937239/ ; LPC 2024 mTHP perf talks: https://lwn.net/Articles/975938/
- LWN MADV_COLLAPSE: https://lwn.net/Articles/892727/
- LWN swap rework / large-folio swap: https://lwn.net/Articles/1016136/ ; 6.13 MM pull: https://lkml.iu.edu/2411.2/02192.html
- LWN file-THP config removal (2026): https://lwn.net/Articles/1070484/
- khugepaged mTHP collapse (late-2025 era): per-size knobs and docs above; presence on 7.2.12 to verify via khugepaged sysfs counters
- Red Hat BZ 1694305 (kcompactd 100% CPU): https://bugzilla.redhat.com/show_bug.cgi?id=1694305
- CONFIG_READ_ONLY_THP_FOR_FS: https://cateee.net/lkddb/web-lkddb/READ_ONLY_THP_FOR_FS.html ; Oracle THP text mappings: https://blogs.oracle.com/linux/transparent-hugepages-for-text-mappings
- ext4 large folios (6.16): https://www.phoronix.com/news/EXT4-Large-Folios-Performance ; XFS/large-block-size: https://lwn.net/Articles/1009548/ (https://kernelnewbies.org/KernelProjects/large-block-size)
