---
type: research-digest
date: 2026-11-07
topic: second-pass preboot review of the hugetlb-pool plan — memcg charging of hugetlb under cgroup v2, boot-time reservation reliability, overcommit surplus, proactive compaction with half of RAM reserved, mTHP for the residual 4K mass, zram policy, and 7.x-era mm supersets
kernel: 7.2.12-300.fc45.x86_64 (Fedora 45), 184 GiB RAM (MemTotal 193105792 kB), zram 8G swap, cgroup v2 mounted with memory_recursiveprot + memory_hugetlb_accounting
pool-state: 24369/45056 pages achieved at runtime today (two compact_memory assists) — a live specimen of silent partial allocation; 45056 (88 GiB) armed via /etc/sysctl.d/91-strata-hugepages.conf for next boot
method: primary sources — kernel admin-guide docs fetched 2026-11-07 (hugetlbpage, cgroup-v2, sysctl vm, damon/reclaim), LWN, Oracle kernel blog, Nhat Pham's hugetlb-memcg patch series, NVIDIA developer forums/blog, Red Hat docs — plus live /proc, /sys, systemctl, cgroup-file measurements on this box. Every claim carries a source and a confidence grade.
grades: High = primary kernel doc or measured on this box. Medium = LWN/commit-era evidence, semantics clear, magnitude or 7.2-presence uncertain. Low = engineering estimate.
---

# Digest 2026-11-07 — Hugetlb pool round 2: charging, boot guarantee, surplus, compaction, mTHP residual, zram

## 0. Measured state of this box today (inputs to every answer)

| Signal | Value | Meaning |
|---|---|---|
| cgroup2 mount options | `memory_recursiveprot,memory_hugetlb_accounting` | unified memcg charging of hugetlb is LIVE — features analyzed in T1 apply |
| HugePages_Total / Free | 24369 / 24369 | today's runtime attempt came up 54% short (20687 pages missing) and NOTHING complained — silent partiality is proven on this box |
| nr_overcommit_hugepages | 0 | no surplus path armed |
| cmdline | no `hugepages=`, no `crashkernel=`, kdump inactive | no firmware/kdump competitor for the reservation; strongest method unused |
| user-1000.slice + user@1000.service memory.max/high | `max` / `max` | no memory limits anywhere on the user charge path (T1) |
| user@1000.service LimitMEMLOCK soft / hard | 8388608 (8 MiB) / 24717541376 (~23 GiB) | cudaHostRegister pinning of 37–56 GiB arenas will FAIL on the soft limit as-is (T1) |
| user@1000.service memory.stat | anon 6.0 GiB, file 42.8 GiB | the ~38 GiB tmpfs store is currently charged as `file`; after migration it re-appears as `hugetlb` |
| vmstat | thp_fault_alloc 1.24M / thp_fault_fallback 1.22M (49%); compact_daemon_wake 335K; pgsteal_proactive 0 | the degraded-regime signature from round 1, unchanged |
| mTHP knobs | hugepages-8kB…2048kB present; ALL sub-PMD sizes `[never]`; 2048kB `[inherit]`; 8kB knob reads empty | anon mTHP surface exists at every size down to 16k; nothing enabled (T5) |
| AnonHugePages | ~10 MiB | in practice nothing on this box calls MADV_HUGEPAGE today (T5) |
| hugetlbfs mounts | /dev/hugepages pagesize=2M; **strata-hugepages → /mnt/strata-huget, pagesize=2M, size=42949672960, uid=1000** | the 40 GiB file-store mount is already armed (T1/T6) |
| zram | lzo-rle, 8G disksize, DATA 7.9G, COMPR 5.1G (ratio 1.55) | near full; fedora default `zram-size = min(ram, 8192)` confirmed in /usr/lib/systemd/zram-generator.conf (T6) |
| swappiness / compaction_proactiveness | 10 / 40 | already tuned; both evaluated below (T4/T6) |
| kernel config | TRANSPARENT_HUGEPAGE_MADVISE=y; SHMEM_TMPFS_HUGE_NEVER=y; LRU_GEN_ENABLED=y; HUGETLB_PAGE_OPTIMIZE_VMEMMAP=y (DEFAULT_ON unset; /proc/sys/vm/hugetlb_optimize_vmemmap=0) | HVO available but off — ~1.2 GiB left on the table (T2) |
| buddyinfo | Node 0 only; Normal zone holds 771 order-9 + 9 order-10 blocks in the degraded state | single NUMA node; per-node allocation path is trivial (T2) |

---

## 1. memcg hugetlb accounting: does the pool charge to user@1000.service, and what can refuse the allocation

**Answer:** Yes — hugetlb pages charge to the faulting process's cgroup at fault time. On this box **no memory limit** exists anywhere on the user path, so nothing will refuse the charge. The real refusal risk for a user-session engine is not memory.max — it is **RLIMIT_MEMLOCK, measured at 8 MiB soft on user@1000.service**, which breaks cudaHostRegister. Fix that one knob.

**Mechanism, primary source.** cgroup-v2 doc, `memory_hugetlb_accounting` mount option (verbatim, fetched 2026-11-07): "Count HugeTLB memory usage towards the cgroup's overall memory usage for the memory controller"; "This is a new behavior that could regress existing setups, so it must be explicitly opted in with this mount option." Charging semantics: "There is no HugeTLB pool management involved in the memory controller. The pre-allocated pool does not belong to anyone. It is only charged to a cgroup when it is actually used (for e.g at page fault time)." [High](https://docs.kernel.org/admin-guide/cgroup-v2.html)

The mount option exists since 6.7 (Nhat Pham's "account hugetlb-backed memory in memory controller" series; rationale: hugetlb was previously invisible to the memory controller, so cgroups got overprotected and unaccounted hugetlb could defeat limits). [High](https://patchew.org/linux/20231006184629.155543-1-nphamcs@gmail.com/20231006184629.155543-4-nphamcs@gmail.com/)

**Charge timing consequence:** reservation (mmap of MAP_HUGETLB) charges nothing; population charges. Engines pin arenas with cudaHostRegister → every page faults → the full 37–56 GiB lands in `user@1000.service` memory.current and memory.stat (the store's 40 GiB likewise; today it is charged as `file` under tmpfs, so the box's total memcg accounting does not grow, it just re-labels). Expect a `hugetlb` entry in memory.stat after engine start — that is the verification probe. [High]

**Can a default Fedora user slice refuse it?** No. Measured on this box: `/sys/fs/cgroup/user.slice/user-1000.slice/memory.max` = `max`, `memory.high` = `max`; same for `user@1000.service`. systemd does not set MemoryMax/MemoryHigh on user slices by default (default is unset/infinite, systemd.resource-control). The only memory knobs present are systemd-managed protections: `MemoryMin=250 MiB`, `MemoryLow=0` (system.control drop-ins) — protections, not limits; they cannot refuse an allocation. [High, measured]

**The documented failure mode if a limit is ever set — and it is ugly.** cgroup-v2 doc: "Failure to charge a HugeTLB folio to the memory controller results in SIGBUS. This could happen even if the HugeTLB pool still has pages available (but the cgroup limit is hit and reclaim attempt fails)." Two extra twists make this worse than a generic OOM: (a) hugetlb is not reclaimable, so hitting memory.max with a mostly-hugetlb cgroup gives no reclaim escape hatch — every page beyond the limit is a fault-time SIGBUS; (b) charging happens on faults during engine warm-up, so a limit sized by eyeball manifests as SIGBUS mid-load, with a half-populated arena. The doc also warns: "Charging HugeTLB memory towards the memory controller affects memory protection and reclaim dynamics" — userspace tuning of min/low must budget for the hugetlb mass. [High](https://docs.kernel.org/admin-guide/cgroup-v2.html)

**The gate that WILL fire today: RLIMIT_MEMLOCK.** CUDA pins host memory by page-locking it (cudaHostRegister "page-locks the memory range"; NVIDIA's own guidance for large pinned allocations is to raise memlock to unlimited — documented in NVIDIA developer forums where cudaMallocHost fails beyond RLIMIT_MEMLOCK, and in the pinned-memory optimization guide). Measured: user@1000.service `LimitMEMLOCKSoft=8388608` (8 MiB), hard ~23 GiB (24717541376, a systemd-262 default, well above soft but below arena sizes). A 37–56 GiB cudaHostRegister in a user session fails today. [High, measured + NVIDIA](https://forums.developer.nvidia.com/t/pinned-memory-limit/39777) (https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cpp/)

**Verify/set list (T1):**
1. `/etc/systemd/system/user@.service.d/90-memlock.conf` → `[Service] LimitMEMLOCK=infinity`; `systemctl daemon-reload`. This covers every process spawned by the user manager, including user services and session scopes under it. For raw-login shells, also set `memlock` soft/hard in `/etc/security/limits.conf` (PAM path bypasses user@.service rlimits). Verify in-process: `grep MemLock /proc/<engine-pid>/limits` shows unlimited.
2. Never set memory.max / memory.high on `user.slice`, `user-1000.slice`, or `user@1000.service` below peak hugetlb usage + 20% margin. Codify as an invariant comment next to the sysctl.d pool file: hugetlb faults fail with SIGBUS, not OOM, when the charge fails.
3. Post-start probe: `grep -E '^hugetlb' /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.stat` — should show ~arena+store bytes.
4. systemd-oomd is active on this box but `ManagedOOMSwap=auto` / `ManagedOOMMemoryPressure=auto` on user@1000.service — no kill policy configured on the user tree; nothing for oomd to trip on (and hugetlb cannot swap anyway). Leave it. [High, measured]

Confidence: High (doc semantics verbatim + every gate measured live).

---

## 2. Boot-time reservation: sysctl vs kernel cmdline, failure modes, partial-allocation visibility

**Answer:** Move the reservation to the kernel command line (`default_hugepagesz=2M hugepagesz=2M hugepages=45056`). Kernel docs rate cmdline allocation "the most reliable method" because memory is allocated before any fragmentation can exist; the sysctl.d write runs after initrd and can silently come up short — this box is the proof (24369/45056 at runtime today, no error anywhere). Keep the sysctl.d file as a self-healing reconciliation write: once the pool is at 45056 the write is a no-op, and if the cmdline ever misses it still tops up.

**Kernel-doc evidence.** hugetlbpage.rst: "The administrator can allocate persistent huge pages on the kernel boot command line by specifying the 'hugepages=N' parameter. This is the most reliable method of allocating huge pages as memory has not yet become fragmented." Parameter semantics: `hugepagesz`/`hugepages` pairs per size ("can only be specified once on the command line for a specific huge page size"); `hugepages=256 hugepagesz=2M hugepages=512` allocates 256 and **ignores the 512 with a warning** — ordering matters, put `hugepagesz` before `hugepages`; `default_hugepagesz` once only; `hugepage_alloc_threads=N` parallelizes boot-time allocation of non-gigantic pages (relevant at 45056 pages). [High](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)

**Known failure modes for the sysctl.d-at-boot approach (what you have now):**
- **Ordering:** systemd-sysctl.service applies /etc/sysctl.d in the real root, after the initrd has run. Dracut's module loads, root-device probing, and tmpfs churn allocate from the buddy before the pool write — the window is small but exactly the fragmentation exposure the docs warn about. Cmdline allocation happens inside kernel init, before initrd unpack and before any userspace. [High, doc + boot ordering]
- **Silent partiality:** for a sysctl write, "nodes without contiguous memory will be silently skipped," and the achieved count is simply what `nr_hugepages` ends up holding — systemd-sysctl logs nothing at failure. Measured specimen: today's runtime raise stalled at 24369 with two `compact_memory` assists and zero complaints. The boot-time sysctl write can fail the same way if anything (initrd residue, firmware-reserved regions) chopped MAX_ORDER blocks. [High](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)
- **crashkernel:** absent on this box (no `crashkernel=` in cmdline, kdump inactive, no `Crash kernel` range in /proc/iomem) — zero interference. On boxes that carry `crashkernel=auto`, the kexec reservation is also a memblock-era carve-out and does not fragment the pool, it merely consumes address space; nothing to do here. [High, measured]
- **movablecore/kinplace:** not set on this box; no interaction. [High, measured]

**Partial allocation visibility + watching.** The achieved count is always visible at `/proc/sys/vm/nr_hugepages` and `/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages`; dmesg logs per-hstate boot allocation results. But visibility is not enforcement — nothing in Fedora watches it, and an engine that starts against a short pool gets a clean ENOMEM at mmap (acceptable, fail-fast) or, worse, an overcommit/surplus SIGBUS later (T3). Add a oneshot verify service (after systemd-sysctl, before multi-user): assert `nr_hugepages >= 45056`, otherwise fail loudly and drop a marker file. The engine-side equivalent — fail fast at mmap time — is already the correct second line of defense. [High]

**Free upgrade on the same cmdline: HVO.** `CONFIG_HUGETLB_PAGE_OPTIMIZE_VMEMMAP=y` is compiled in but off (`DEFAULT_ON` unset, `/proc/sys/vm/hugetlb_optimize_vmemmap=0`). Without it, 45056 huge pages burn 512 struct pages × 64 B = 32 KiB of vmemmap each ≈ **1.44 GiB**, and those vmemmap pages are unmovable scatter spread through memory. `hugetlb_free_vmemmap=on` frees all but ~one 4 KiB page per huge page ≈ **1.2 GiB back** and removes ~360K scattered unmovable pages. Note the doc caveat that freeing huge pages while HVO is on "may fail under memory pressure — please try again later" (teardown only; irrelevant for a 24/7 pool). [High on arithmetic + config; Medium-High on HVO stability era]

**Recommended boot config:** `default_hugepagesz=2M hugepagesz=2M hugepages=45056 hugetlb_free_vmemmap=on` (+ optional `hugepage_alloc_threads=8`), via GRUB_CMDLINE_LINUX + `grub2-mkconfig`; keep `91-strata-hugepages.conf` as reconciliation. Confidence: High.

---

## 3. nr_overcommit_hugepages surplus: elasticity or the same trap?

**Answer:** Surplus is real elasticity with zero upfront cost, but its allocation happens **at fault time on the running box** — physically identical exposure to runtime nr_hugepages growth — and its failure mode is strictly worse: mmap succeeds, the engine serves, then faults start SIGBUSing when the buddy cannot yield 2 MiB blocks. Sound as a sized safety net; unsound as the primary mechanism for the 94 GiB config.

**Kernel behavior.** hugetlbpage.rst: a non-zero `nr_overcommit_hugepages` lets hugetlb "try to obtain that number of 'surplus' huge pages" from the normal page pool when the persistent pool is exhausted; "As these surplus huge pages become unused, they are freed back to the kernel's normal page pool." The allocation therefore needs exactly what a runtime sysctl raise needs — physically contiguous memory present at allocation time — except it is attempted per-fault, potentially mid-request, with whatever compaction/reclaim the fault path can do under allocation pressure. Success "depends on the amount of physically contiguous memory" (same doc). The mmap-time reservation check passes on free-pool + overcommit headroom, which is why the failure moves from clean startup ENOMEM to runtime SIGBUS. [High](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)

**Corroborating era evidence:** Oracle's kernel blog documents the surplus model as of v6.19 ("HugeTLB Gigantic Page Overcommit"), and surplus accounting was still receiving correctness patches into 2026 (LKML, "mm/hugetlb: Try to respect mempolicy when calculating surplus pages", July 2026) — an active, best-effort subsystem, not a guarantee surface. [Medium](https://blogs.oracle.com/linux/hugetlb-gigantic-page-overcommit)

**Second trap, documented:** shrinking `nr_hugepages` below the in-use count "will convert the balance of the in-use huge pages to surplus huge pages" — potentially exceeding the overcommit allowance and blocking further surplus until in-use drops. Relevant to any future pool-resize playbook. [High](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)

**On this box specifically.** The gap the 94 GiB config needs over the 88 GiB pool is 6 GiB = 3072 pages. Post-reboot the buddy holds ~96 GiB and the engines start early, so surplus allocation of 3072 pages will nearly always succeed — early. The exposure re-appears exactly when the box is old and busy, which is the state this whole architecture exists to escape. Also: surplus pages are charged to user@1000.service under memory_hugetlb_accounting like any hugetlb folio (no interaction, memory.max=max today). If elasticity is wanted, size it to the actual gap — `nr_overcommit_hugepages=3072`, not 8192 — and keep the engine's mmap path fail-fast. The stronger alternative: boot a second GRUB entry with `hugepages=48128` (94 GiB) for the bigger config, or just accept the 6 GiB idle cost permanently (3.2% of RAM, ~90 GiB still free for everything else). [Medium-High]

Confidence: High on mechanism (docs); Medium on the "fails exactly like runtime growth" physics (strong inference from doc semantics + allocator reality, not a lab repro on this box).

---

## 4. kcompactd / proactive compaction with ~46% of RAM in an unmovable pool

**Answer:** The reservation reduces fragmentation pressure; it does not starve compaction. Pool pages are invisible to the page allocator, so kcompactd and the fragmentation score never see the 88 GiB at all — they see a smaller, cleaner movable mass (~96 GiB) with the biggest unmovable scatterer of the degraded regime (107 GiB of tmpfs 4K backing + ~215 MiB of scattered PTE pages) gone into a pool that cannot fragment. Keep `compaction_proactiveness=40` through the reboot window; expect it to matter less afterward, and re-derive from measurements in a week.

**Mechanism.** Reserved hugetlb pages are carved out of buddy management ("Pages that are used as huge pages are reserved inside the kernel and cannot be used for other purposes" — hugetlbpage.rst); compaction operates on buddy free blocks only, and the proactive-compaction fragmentation score is computed over zone free areas. Fewer managed pages and fewer unmovable objects in those pages = lower scores and less work per scan. The order-9 demand that made round-1's box thrash (71% compact_fail) came from the tmpfs store and 4K masses competing for 2 MiB buddy blocks; the pool removes the largest competitor (it cannot scatter — that is its contract), and THP=madvise with no madvise callers makes the residual order-9 demand small. [High on mechanism](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)

**Knob semantics, primary source.** vm.rst: `compaction_proactiveness` "takes a value in the range [0, 100] with a default value of 20" and "determines how aggressively compaction is done in the background"; above 80 the kernel becomes "more sensitive to increases in fragmentation," triggering more often with smaller per-run reductions, and extremes "risk excessive background compaction activity"; compaction itself "could also lead to latency spikes in unsuspecting applications." Proactive compaction (fragmentation-score-driven kcompactd work) merged in 5.13 after Nitin Gupta's ~8-round series — mature, and on this box it has reclaimed nothing (`pgsteal_proactive=0`) while kcompactd woke 335K times in the degraded regime. [High](https://docs.kernel.org/admin-guide/sysctl/vm.html) ([LWN](https://lwn.net/Articles/817905/), [author writeup](https://nitingupta.dev/post/proactive-compaction))

**Guidance:**
1. Keep 40 for the reboot window: engines re-fault ~90+ GiB of arenas into a pristine buddy and the remaining masses (shadow store until its own migration, tmpfs residue) rebuild; the round-1 rationale ("refill 2 MiB supply between engine cycles") still applies until the shadows are collapsed/mTHP'd (T5).
2. Post-migration the knob's job shrinks: the pool can't fragment, anon arenas are hugetlb-mapped, and remaining order-9 demand is the shadow collapse path. Re-measure at one week: `compact_stall`/`compact_fail` rates, kcompactd CPU under idle. If both are quiet, dropping to 20 is free headroom back; if compact_fail is still significant, the residual cause is the 4K masses (T5), not the knob.
3. Pool resize (freeing pages back) returns 2 MiB blocks to the buddy in max-order chunks — teardown does not re-fragment; `compact_memory` assists remain a valid manual lever for runtime top-ups. [High]

Confidence: High on mechanism; Medium on post-migration magnitudes (needs the week of measurements).

---

## 5. The residual 4K mass: MADV_HUGEPAGE/MADV_COLLAPSE vs mTHP vs leave-4K

**Answer:** Do the 2 MiB path first (MADV_HUGEPAGE + khugepaged, or MADV_COLLAPSE at shadow-build), enable mTHP `hugepages-64kB/enabled=inherit` as the fallback layer for what cannot collapse, and leave 16k/32k off. Post-reservation the "no supply" premise is stale: anon 2 MiB THP draws from the remaining ~96 GiB buddy, not from the pool — and this box's root probe already measured anon MADV_COLLAPSE at 100% promotion.

**Current state, measured.** All mTHP knobs exist down to 8 kB; every sub-PMD size is `[never]`; 2048 kB is `[inherit]` (follows `enabled=madvise`); AnonHugePages ≈ 10 MiB — i.e., in practice nothing calls MADV_HUGEPAGE on this box today (consistent with round 1; the "hugepage.hpp fallback" call sites, if they exist, are not being exercised — verify them as a checklist item rather than assuming). thp_fault_fallback 49% shows how hostile the old regime was to fault-time THP. [High, measured]

**Maturity of mTHP.** Anon mTHP merged in 6.8 (Ryan Roberts series; per-size sysfs `hugepages-*/enabled` with always/inherit/madvise/never; fault-path allocation of large folios between 4K and PMD). [High](https://lwn.net/Articles/956435/) (earlier coverage: [Large folios for anonymous memory](https://lwn.net/Articles/927544/)) Shmem mTHP landed 6.13 (round-1 digest; knob verified present on this box). The backfill path — khugepaged and MADV_COLLAPSE collapsing to non-PMD mTHP sizes — is the newest piece: Dev Jain's "khugepaged: mTHP support" series targeted the 6.15/6.16 window (khugepaged scans PMD-sized chunks, drops to the largest enabled mTHP size when PMD collapse is impossible; MADV_COLLAPSE gained the same; known limitation: regions with shared or swapped-out pages are not collapsed). On a 7.2-era kernel this should be present — grade Medium until a small test collapse confirms on-box. [Medium](https://lwn.net/Articles/1048881/) Era churn warning: 2026 saw a 57-patch RFC (Meta) proposing to replace khugepaged's anon collapse wholesale, with pushback — do not build strategy on automatic mTHP collapse alone; MADV_COLLAPSE at load time is the stable API. [Medium]

**TLB math for a 50 GiB/s stream (honest bounds).** 4 KiB pages: 50 GiB/s ÷ 4 KiB = **12.8M page-crossings/s**; a 23 GiB shadow pass over an L2 STLB of ~2K entries (~8 MiB coverage on common x86) rewrites the whole TLB ~2900x per pass → every page takes an L2-TLB refill per pass. Upper-bound cost: 12.8M misses/s; hardware page-walk caches (PDE/PTE caches) make sequential walks cheap (~10–20 cycles), so the realistic tax on a **pure sequential** stream is a few percent of bandwidth plus walk-traffic (12.8M × ~2 walk lines ≈ ~1.6 GB/s stolen); the tax grows with strided/layer-scattered access where walks miss caches. 64 KiB folios: **16x** fewer translations and 16x less walk traffic — coverage per entry rises 8 MiB → 128 MiB equivalent, which covers a whole per-layer shadow (0.26 GiB) in 4 entries. 2 MiB: 512x — this is what the pool buys for arenas. mTHP 64K is meaningful but second-order; 2M is the win. Internal-fragmentation risk at 64K granularity for fully-populated shadows: zero (0.26 GiB is 64K-aligned usage). [Medium on percentages — arch-dependent; High on ratios]

**Cost side of leaving 4K:** 23 GiB of 4 KiB mappings = ~11.8K PTE pages ≈ **46 MiB** of unmovable page-table scatter, and ~5.9M rmap/fault units. This is round-1's pathology at one-fifth scale — worth removing given the supply now exists. [High on arithmetic]

**Verdict:** (a) engines add MADV_HUGEPAGE on shadow VMAs (khugepaged/defrag=1 already on) and/or MADV_COLLAPSE at shadow-build — deterministic, supply-backed; (b) set `hugepages-2048kB/enabled=inherit` stays, add `echo inherit > /sys/kernel/mm/transparent_hugepage/hugepages-64kB/enabled` (sysfs, non-persistent — add a systemd oneshot or sysfsutils unit; inherit gates it to MADV_HUGEPAGE VMAs, keeping the box-wide default conservative); (c) skip 16k/32k — ratios drop to 4–8x while folio-management overhead per folio stays, and the 8 kB knob on this box reads empty (not usable). Confidence: High on state and ratios; Medium on khugepaged-mTHP presence on 7.2 and TLB percentages.

---

## 6. zram policy with 88 GiB unswappable

**Answer:** Zero interaction, confirmed at the doc level — hugetlb has no swap path, so zram serves only the general anon/tmpfs remainder. Keep zram as the only swap; grow it to 16G and switch to zstd; do not add disk-backed swap or zswap on this box. Swappiness: current 10 is safe; the kernel doc explicitly endorses >100 for zram-class swap, and the right value after migration is an empirical call, not 0.

**The no-interaction proof.** hugetlbpage.rst, verbatim: "Huge pages cannot be swapped out under memory pressure." This covers pool pages, MAP_PRIVATE hugetlb COW, and hugetlbfs file pages (no backing-device writeback; freed only on truncate) — nothing in the 88 GiB pool or the 38–40 GiB store can ever reach the swap layer, so no zram policy choice can touch them and no hugetlb state can fragment zram. [High](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)

**Sizing.** Fedora's shipped default is `zram-size = min(ram, 8192)` (confirmed in /usr/lib/systemd/zram-generator.conf) → 8G. Measured today: DATA 7.9G / 8G — full, with lzo-rle at a 1.55 ratio (5.1G compressed). Post-migration, the store leaves the swappable class (hugetlbfs), so demand drops — but the dam should still cover the residual tmpfs + anon spikes of user slices: `zram-size = min(ram, 16384)` and `compression-algorithm = zstd` in /etc/systemd/zram-generator.conf. Disksize is a cap, not an allocation (memory is consumed only by compressed content, as zramctl's COMPR column shows), so growing costs nothing idle. [High]

**Disk-backed zswap? No.** zswap needs a backing swap device; adding a disk swapfile reintroduces swap-IO jitter onto a latency-critical box precisely to insure against a scenario (general-anon overflow) that is bounded and rare post-migration. zram's decompress path is CPU-only. If reclaim pressure against the user slices ever materializes, cgroup memory.high + MGLRU + a bigger zram is the whole toolchain. [High on mechanics; judgment]

**Swappiness.** Measured 10. vm.rst: swappiness sets "the rough relative IO cost of swapping and filesystem paging"; default 60; "for zram/zswap or fast swap, values over 100 are reasonable (e.g., 2x-faster swap → 133)"; at 0 the kernel won't swap "until the amount of free and file-backed pages is less than the high watermark" — avoid 0. Since hugetlb is invisible to every reclaim path, swappiness only steers the residual anon-vs-file decision; with MGLRU active (CONFIG_LRU_GEN_ENABLED=y, measured) both 10 and 100 are defensible — 10 biases toward keeping file cache, 100 leans into cheap zram when anon pressure exists. Recommendation: keep 10 now; revisit toward 100 only if measurements show anon reclaim pressure post-migration. [High](https://docs.kernel.org/admin-guide/sysctl/vm.html)

Confidence: High.

---

## 7. 7.x-era mm: what supersedes, what to ignore

Graded against this workload (fixed 88 GiB pool + 38–40 GiB store + ~23 GiB residual 4K shadows, single node, 50 GB/s streams):

| Item | Maturity | Verdict for this box |
|---|---|---|
| **Anon mTHP per-size knobs** (6.8) | Mature — knobs verified on box, all `[never]` | Adopt for the 64K shadow fallback (T5). Grade A. [High](https://lwn.net/Articles/956435/) |
| **Shmem mTHP** (6.13) | Mature surface, present on box — yet this box's root probe measured shmem huge folios NEVER materializing even at huge=always and via MADV_COLLAPSE (0 kB). That is an anomaly against documented semantics (round-1: `never` still allows MADV_COLLAPSE). Moot after the hugetlbfs decision — but re-probe post-reboot as a regression guard; if a 7.2 bug is hiding there, the residual tmpfs masses are affected too. | Grade B, curiosity + guard. [High, probe measured](https://lwn.net/Articles/974630/) |
| **khugepaged / MADV_COLLAPSE mTHP collapse** (6.15/16-era) | New-ish; 2026 RFC churn (khugepaged replacement proposal) signals design motion | Use MADV_COLLAPSE at load (stable API), treat automatic mTHP collapse as bonus. Grade B+. [Medium](https://lwn.net/Articles/1048881/) |
| **MGLRU** (CONFIG_LRU_GEN_ENABLED=y, measured) | Mature, active on this box | Already working; pairs with cgroup high for slice-level reclaim. No action. Grade A. |
| **DAMON_RECLAIM** (docs fetched at 7.4-rc1: "static kernel module… proactive and lightweight reclamation under light memory pressure", quota-lipped at 128 MiB/window default, "selectively used", cannot co-run with other DAMON modules) | Mature but advisory-grade | Skip. Paging out cold shadow layers at 4K granularity reintroduces the exact churn the pool removes, and cgroup memory.high + zram already implement the same policy simpler. Revisit only if cold-layer sprawl reappears. Grade C. [High, doc](https://docs.kernel.org/admin-guide/mm/damon/reclaim.html) |
| **Memory tiering / demotion, CXL tiers** | Mature upstream, irrelevant | Single NUMA node (buddyinfo, measured). N/A. |
| **HVO vmemmap optimization** (6.7-era, present in config, off) | Mature | Enable with the pool at boot — ~1.2 GiB back and ~360K fewer scattered unmovable pages (T2). Grade B+. |
| **Proactive compaction** (5.13) | Mature | Keep 40, re-derive post-migration (T4). Grade A. [Medium](https://lwn.net/Articles/817905/) |
| **Large-folio swap-in + zram large-folio compression** (6.12/6.13, round-1 cites) | Mature | Marginal here — swap is a dam, not a tier; benefits only the residual anon that ever swaps. No action. |
| **Surplus/overcommit accounting fixes** (v6.19 Oracle writeup; 2026 mempolicy surplus patch) | Active bug-fix surface | Confirms T3's "best-effort, not guarantee" grading. [Medium](https://blogs.oracle.com/linux/hugetlb-gigantic-page-overcommit) |

**Nothing supersedes the plan.** The plan already IS the modern stack: boot-reserved hugetlb pool + unified memcg accounting (6.7) + mTHP knobs (6.8/6.13) + MGLRU + MADV_COLLAPSE. The only supersets worth adopting are HVO (T2) and the mTHP-64K shadow fallback (T5).

---

## Do these 10 things (before / after the reboot)

**Before reboot:**
1. **Cmdline reservation** — GRUB_CMDLINE_LINUX += `default_hugepagesz=2M hugepagesz=2M hugepages=45056 hugetlb_free_vmemmap=on`; `grub2-mkconfig -o /boot/grub2/grub.cfg`. Keep `/etc/sysctl.d/91-strata-hugepages.conf` as a no-op reconciliation top-up.
2. **Fix the pin gate** — `/etc/systemd/system/user@.service.d/90-memlock.conf`: `[Service] LimitMEMLOCK=infinity`; daemon-reload. (Measured soft limit 8 MiB WILL fail cudaHostRegister of 37–56 GiB arenas; also set memlock in limits.conf for raw-login shells.)
3. **Codify the charge invariant** — comment in 91-strata-hugepages.conf: never set memory.max/high below arena-peak+20% on user.slice / user-1000.slice / user@1000.service — hugetlb charge failure = fault-time SIGBUS, not OOM (cgroup-v2 doc, verbatim).
4. **Size surplus to the gap** — `vm.nr_overcommit_hugepages=3072` (the 94 GiB config's 6 GiB gap), not 8192; engine mmap path stays fail-fast.
5. **Boot watcher** — oneshot systemd service after systemd-sysctl: assert `/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages >= 45056`, fail loudly + marker file on shortfall (silent partiality is measured fact on this box).

**After reboot (verify + tune):**
6. **Verify the pool** — nr_hugepages=45056, HugePages_Free=45056, dmesg hugepage lines; `grep MemLock /proc/<engine-pid>/limits` = unlimited; after engine start `grep hugetlb memory.stat` under user@1000.service shows arena+store bytes; `/mnt/strata-huget` store fills with 2M-backed files.
7. **Shadow path** — enable `hugepages-64kB/enabled=inherit` (persisted via oneshot/unit), verify engine MADV_HUGEPAGE/MADV_COLLAPSE call sites actually execute (AnonHugePages should climb from ~10 MiB), measure TLB/throughput delta on the 50 GB/s lane.
8. **zram resize** — `/etc/systemd/zram-generator.conf`: `zram-size = min(ram, 16384)` + `compression-algorithm = zstd`; confirm DATA post-migration drops well below 8G.
9. **kcompactd re-derivation** — one week at proactiveness=40: compact_stall/fail rates + kcompactd idle CPU; drop to 20 if quiet (the pool made the knob's job smaller).
10. **Re-probe the shmem-huge anomaly** — post-reboot, tiny tmpfs huge=always + MADV_COLLAPSE test; 0 kB again = file a kernel-bug note next to the store decision.

## Sources

- HugeTLB admin guide (boot vs runtime reliability, surplus semantics, swappability, boot params, HVO caveat): https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
- Control Group v2 (memory_hugetlb_accounting semantics, fault-time charging, SIGBUS on failed charge): https://docs.kernel.org/admin-guide/cgroup-v2.html
- hugetlb memcg accounting series (Nhat Pham, 6.7): https://patchew.org/linux/20231006184629.155543-1-nphamcs@gmail.com/20231006184629.155543-4-nphamcs@gmail.com/
- Sysctl vm doc (compaction_proactiveness, nr_hugepages/overcommit, swappiness, hugetlb_optimize_vmemmap): https://docs.kernel.org/admin-guide/sysctl/vm.html
- Transparent Hugepage doc (madvise semantics, mTHP knobs, MADV_COLLAPSE): https://docs.kernel.org/admin-guide/mm/transhuge.html
- Proactive compaction (LWN; merged 5.13): https://lwn.net/Articles/817905/ ; https://nitingupta.dev/post/proactive-compaction
- Anon mTHP merge (LWN, 6.8): https://lwn.net/Articles/956435/ ; https://lwn.net/Articles/927544/
- Shmem mTHP (LWN, 6.13): https://lwn.net/Articles/974630/
- khugepaged mTHP collapse (LWN, 6.15/16-era): https://lwn.net/Articles/1048881/
- hugetlb surplus model as of v6.19 (Oracle blog): https://blogs.oracle.com/linux/hugetlb-gigantic-page-overcommit
- hugetlbfs unification / reservation mechanics (LWN): https://lwn.net/Articles/972943/
- Huge pages series (LWN 2010): https://lwn.net/Articles/374407/
- Red Hat huge pages tuning (nr_overcommit_hugepages semantics): https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/7/html/performance_tuning_guide/sect-red_hat_enterprise_linux-performance_tuning_guide-memory-configuring-huge-pages
- NVIDIA pinned memory vs RLIMIT_MEMLOCK: https://forums.developer.nvidia.com/t/pinned-memory-limit/39777 ; https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cpp/
- DAMON_RECLAIM admin doc: https://docs.kernel.org/admin-guide/mm/damon/reclaim.html
- zram-generator (Fedora default config): https://github.com/systemd/zram-generator
- systemd resource control (MemoryMax unset default, LimitMEMLOCK): https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html ; https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html
- Round-1 digests (measured degraded-regime baseline): .docs/research/digest/2026-11-07-hostmem-kernel-state.md, 2026-11-07-hostmem-inference-servers.md
