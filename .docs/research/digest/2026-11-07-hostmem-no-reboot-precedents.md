---
type: research-digest
id: 2026-11-07-hostmem-no-reboot-precedents
date: 2026-11-07
scope: how latency-critical production systems keep host-RAM streaming performance stable for months without performance reboots — the reserved-arena pattern and its variants: PostgreSQL, Oracle, DPDK/SPDK, HFT/market-data, Varnish/nginx/Redis/jemalloc, uptime-degradation post-mortems, online-compaction alternatives
subject: Linux host-memory reservation precedents (hugetlb pool semantics, slab/arena allocators, THP policy, compaction sysctls)
method: web research 2026-11-07 against primary sources (kernel.org docs, postgresql.org, redis.io, oracle.com, docs.oracle.com, mongodb.com, splunk.com, dpdk.org, doc.dpdk.org, hudsonrivertrading.com, rigtorp.se, varnish-software, linkedin engineering). All verbatim quotes come from fetched pages; search-summary-only facts are graded B or C.
---

# Host-memory stability without reboots — the reserved-arena precedents

## Verdict (read this first)

1. **The pattern is real, uniform, and named:** every system that streams GB/s from host RAM for months owns its hot-path memory as a **boot-reserved, kernel-pinned, huge-page-backed pool**, carved by the process's **own bounded-waste arena**, with **zero allocation on the hot path** and **page tables frozen after warm-up**. PostgreSQL, Oracle, DPDK, and the HFT stack all converge on this shape independently.
2. **Reservation is armed at boot, not at demand.** The kernel hugetlb doc states the boot cmdline is "the most reliable method of allocating huge pages as memory has not yet become fragmented." PostgreSQL's own doc admits runtime reservation can fail: "Sometimes the kernel is not able to allocate the desired number of huge pages immediately due to fragmentation, so it might be necessary to repeat the command or to reboot." No cited system re-acquires a fragmented pool at runtime; they never let it fragment.
3. **The pool is exempt from the VM's normal life cycle.** Kernel doc: huge pages "cannot be swapped out under memory pressure" and are "reserved inside the kernel and cannot be used for other purposes." Compaction never migrates them; swap never captures them; the buddy allocator never sees them again. This is why DPDK never defrags and Oracle never tunes for it — the failure mode is excluded by construction.
4. **Recycle without reboot is process-level, and it is normal.** Oracle and PostgreSQL instances restart daily in production; freed huge pages return to the hugetlb pool on process exit, so the next instance re-arms instantly onto a pristine pool. "Never reboot" really means "never reboot the kernel." Our engine already restarts several times daily — that half of the pattern is already in place. The defect is that our pool is anon/tmpfs 4K, so each re-arm re-rolls physical layout and the layout degrades under it.
5. **THP is the adversarial version of the same idea, and every DB class of system bans it.** Splunk measured "a minimum of a 30% degradation" with THP active and warns that THP **or** direct compaction alone degrades performance; Oracle alert MOS 1557478.1 ties THP to "performance issues and system reboots"; Redis's latency doc requires `never`; MongoDB required it through 7.0 (and only re-enabled it in 8.0 with a rebuilt allocator plus `khugepaged/max_ptes_none=0`); PostgreSQL docs say THP use "is currently discouraged (unlike explicit use of huge_pages)".
6. **Our kernel version is inside a documented regression class.** AWS reported PostgreSQL at 0.51x baseline throughput on Linux 7.0 without huge pages (Phoronix, 2026-04). Our box runs 7.2.12 with ~99.9% of a 133 GiB resident pool on 4K pages — the same failure surface, plus a full zram and 71% failed compaction attempts.
7. **Online compaction exists but no vendor publishes it as the fix.** `vm.compact_memory` is a synchronous, all-zone stall; `vm.compaction_proactiveness` (default 20) is background kcompactd with an explicit latency-spike warning; Alibaba cloud guidance is the only cited shop scheduling `compact_memory` off-peak (grade C). The dominant operational answer is: reserve at boot, keep the pool out of the kernel's hands, and recycle the process — not the kernel.
8. **The sizing/arming recipe recurs across every system:** pool = max steady-state working set + ~10–20% headroom; reserve at boot cmdline (or earliest rc); persist in sysctl; fail fast on shortfall (`huge_pages=on`, no silent 4K fallback); leave the rest of RAM to the OS; verify 100% of the mapping landed in the pool at start (Oracle alert log prints it, `HugePages_Rsvd` exposes it); monitor pool-free, `/proc/vmstat compact_*`, and TLB miss rate.

Confidence scale used below:
- **A** — primary source fetched this pass; quotes verbatim from the page.
- **B** — reputable source seen via search summary only (Phoronix, Percona, Red Hat, MOS note titles, OpenResty blog).
- **C** — single-source or community practice; treat as hypothesis.

---

## 1. PostgreSQL (grade A unless noted)

**Mechanism.** `huge_pages = try|on|off` controls whether the main shared memory area is backed by huge pages; "At present, this setting is supported only on Linux and Windows. The setting is ignored on other systems when set to try. On Linux, it is only supported when shared_memory_type is set to mmap (the default)." Stated rationale: "The use of huge pages results in smaller page tables and less CPU time spent on memory management, increasing performance."
- Source: <https://www.postgresql.org/docs/current/runtime-config-resource.html>

**Fail-fast vs silent fallback.** "With huge_pages set to try, the server will try to request huge pages, but fall back to the default if that fails. With on, failure to request huge pages will prevent the server from starting up." DBA practice is `on` precisely to kill the silent 4K fallback (grade B: Cybertec pgconfigurator + postgresql.org thread).

**THP stance.** "That feature has been known to cause performance degradation with PostgreSQL for some users on some Linux versions, so its use is currently discouraged (unlike explicit use of huge_pages)."

**Sizing and arming (PG ≥15).** `postgres -D $PGDATA -C shared_memory_size_in_huge_pages` prints the required page count before start (doc example: 3170 pages × 2 MiB). Reserve with `sysctl -w vm.nr_hugepages=3170`, persist in `/etc/sysctl.conf`, or boot-reserve with `hugepagesz=2M hugepages=3170`. Permission knobs: `vm.hugetlb_shm_group`, `ulimit -l`.
- Source: <https://www.postgresql.org/docs/current/kernel-resources.html> (§18.4.5 Linux Huge Pages)

**Scope of the pool.** Huge pages back only the shared buffer pool — not `work_mem`/`maintenance_work_mem`; `min_dynamic_shared_memory` (PG 14+) is allocated alongside the pool and inherits huge pages. The pool "must be reserved as a continuous memory area"; a short `HugePages_Total` means fewer pages than configured. Over-sizing the pool can OOM the box. Sizing formula (grade A, Cybertec): `((shm size + (BLKSZ - (shm size % BLKSZ))) / (huge page size * 1024)) + 1`.
- Source: <https://www.cybertec-postgresql.com/en/huge-pages-postgresql/>

**shared_buffers sizing.** "A reasonable starting value for shared_buffers is 25% of the memory in your system... it is unlikely that an allocation of more than 40% of RAM to shared_buffers will work better."
- Source: <https://www.postgresql.org/docs/current/runtime-config-resource.html>

**What happens without it on long uptime / new kernels.** Phoronix (2026-04-04): an AWS engineer reported PostgreSQL throughput at ~0.51x baseline on the Linux 7.0 development kernel, root cause in TLB-flush/page-fault handling, "mitigated by enabling huge pages" (grade B). Our host runs 7.2.12 — inside this family. With ~99.9% of a 133 GiB pool on 4K pages (AnonHugePages 120 MB), every 50 GB/s streaming pass runs a page-walk storm: 3–5 dependent memory loads per translation when the TLB (≈1500–2000 entries/core ≈ 6–8 MiB of 4K coverage) misses.

## 2. Oracle (grade A unless noted)

**The 4K-SGA cost, quantified.** Oracle's own article: an 8 GB SGA per process = "8 GB/4 KB or 2,097,152 records" of page-table entries; 100 sessions multiply it; a listing shows `PageTables: 766680 kB` and the author: "I have personally seen Page Tables entries in the gigabytes." With HugePages the per-record count drops "to 4,096 from 2,097,152" and the note: "The performance improvement when enabling HugePages can be dramatic."
- Source: <https://www.oracle.com/technical-resources/articles/it-infrastructure/dev-hugepages.html>

**Reservation semantics — the load-bearing quote.** `HugePages_Rsvd`: "Reserved HugePages guarantee that an application will be able to allocate HugePages from the pool of HugePages at the time they are requested, even if the system has been up for a while." This is the exact anti-fragmentation guarantee our anon pool lacks.

**Arming and sizing.** `vm.nr_hugepages` set at boot; MOS 401749.1 ships the sizing script; MOS 361468.1/361323.1 are the canonical guides. Automatic memory management (AMM) is "not compatible with Linux HugePages" — ASMM is the required configuration (MOS 1134002.1). Verification is printed into the alert log at startup: `Total Shared Global Region in Large Pages = 28 GB (100%)`.

**lock_sga / pinning.** Oracle 18c docs: "Oracle recommends specifying enough large pages to contain the entire SGA. The Oracle database instance attempts to allocate large pages when the LOCK_SGA parameter is set to true." On AIX, 16 MiB pages "are always pinned, and cannot be used for standard memory"; if pinning capacity is short, "the portion of the SGA exceeding these sizes is allocated to ordinary shared memory" — the documented silent-fallback hazard.
- Source: <https://docs.oracle.com/en/database/oracle/oracle-database/18/unxar/tuning-oracle-database.html>

**THP ban.** "The use of transparent HugePages in conjunction with the use of traditional HugePages is currently causing problems that can result in performance issues and system reboots" — alert MOS 1557478.1.

**Hugepages-only DBA practice** (grade B): size the pool to SGA + headroom, leave the general allocator nothing it can steal; resize requires reconfiguring the pool, which is why the pool is sized once, at boot.

## 3. DPDK / SPDK (grade A)

**Why hugepages are mandatory.** "With hugepages, fewer pages are needed, and therefore less Translation Lookaside Buffers (TLBs)" — without them "high TLB miss rates would occur with the standard 4k page size, slowing performance."
- Source: <https://doc.dpdk.org/guides/linux_gsg/sys_reqs.html>

**Boot-time reservation is the documented default.** Boot-time reserving is preferred "in use cases where having lots of physically contiguous memory is required," since it prevents "physical memory from becoming heavily fragmented." 2 MiB: `hugepages=1024`; 1 GiB: `default_hugepagesz=1G hugepagesz=1G hugepages=4`. Note: "Some kernel versions may not allow reserving 1 GB hugepages at run time." 1 GiB pages are recommended when the platform supports them. NUMA runtime reservation splits across nodes; per-node sysfs targets one node.

**Why DPDK never defrags or reboots for memory.** DPDK does not malloc: it "allocates huge pages and creates a heap out of this memory." Memory is pinned: since DPDK "uses pinned memory, generally in the form of huge pages," the "physical address of the underlying memory area is not expected to change" — hardware DMAs against it directly. Mempool buffers are a fixed ring of pointers over that heap; freeing an mbuf returns it to the pool's stack/ring, never to the kernel. Physical contiguity is guaranteed by construction (1 GiB page = 1 GiB contiguous IOVA), so there is nothing to defrag. Freed huge pages persist in the kernel pool across restarts.
- Sources: <https://www.dpdk.org/memory-in-dpdk-part-1-general-concepts/>, <https://doc.dpdk.org/guides/linux_gsg/sys_reqs.html>

**SPDK** (grade B): `spdk_memzone_reserve_aligned` with 1 GiB hugepages for >1 GiB contiguous buffers — same reservation, same reasoning (<https://spdk.io>).

## 4. HFT / market-data (grade A)

**Hudson River Trading (Guillaume Morin), parts 1–2.**
- TLB arithmetic: "recent server x86 CPUs come with a TLB of about 1500-2000 entries per core"; with 4K pages that covers "~8MiB worth of memory" vs "~4GiB" with 2 MiB pages. A 4K page walk is "at least 3 dependent memory loads" (~210 ns uncached). Benchmark: 32 GiB array, random access "runtime is decreased by a factor of 4.5", linear init 40% faster.
- Allocation mechanics: creating one huge page needs "512 contiguous 4KiB pages, starting at an address that's a multiple of 2MiB"; when free memory is fragmented "defragmenting can be quite disruptive" (process suspension). Hugetlbfs pool: minimum "pre-allocated", maximum is a hard limit; pool hit "as fast as allocating a 4KiB page"; pool miss at max → SIGBUS; pool pages "cannot be swapped out". Exhaustion under COW fork is the trap — "it's safer not to use hugetlbfs in a process that needs to spawn child processes."
- Allocators: "most of the high performance memory allocators (jemalloc, tcmalloc, mimalloc, etc.) support allocating 2MiB pages backed by either THP, hugetlbfs, or both"; glibc ≥2.35 has `glibc.malloc.hugetlb`.
- Fine print: keep some hot pages 4K — the L1 dTLB has separate 4K/2M/1G slots (Cascade Lake: 32×2M, 64×4K, 4×1G entries), so blanket hugepaging can evict the hottest small objects from L1 TLB slots.
- Sources: <https://www.hudsonrivertrading.com/hrtbeat/low-latency-optimization-part-1>, <https://www.hudsonrivertrading.com/hrtbeat/low-latency-optimization-part-2>

**Erik Rigtorp, "Latency implications of virtual memory" — the never-malloc-on-the-hot-path rules, verbatim:**
- "Minimize page faults by pre-faulting, locking and pre-allocating needed memory. Disable swap."
- "Reduce TLB misses by minimizing your working set memory and utilizing huge pages."
- "Prevent TLB shootdowns by not modifying your programs page tables after startup."
- "Disable Linux transparent huge pages (THP)." Also disable KSM and automatic NUMA balancing.
- Shootdown sources named: "munmap, mprotect, malloc/free internals, THP, compaction, KSM, NUMA balancing, page migration, page cache writeback."
- Fix: "map all needed memory at program startup and avoid calling any functions that modifies the page table after that." mimalloc knobs: `MIMALLOC_RESERVE_HUGE_OS_PAGES=N` (reserve at startup), `MIMALLOC_PAGE_RESET=0` (never return).
- Source: <https://rigtorp.se/virtual-memory/>

**Aeron (Martin Thompson et al.)** (grade B): log buffers are mmap'd files of three equal pre-allocated terms (default ~16 MiB each, 64 KiB–1 GiB range) written round-robin; the transport path is "Allocation-Free: the send and receive buffer is pre-allocated so the transport path is free from the allocation or reclamation of memory." This is the market-data formulation of the same invariant: size the ring at startup, never allocate in the loop.
- Sources: <https://aeron.io/docs/aeron/log-buffers-images>, <https://theaeronfiles.com/aeron-transport/log-buffers>

## 5. Varnish / nginx / Redis / jemalloc — allocation hygiene (grade A unless noted)

**Varnish.** jemalloc is the default stevedore allocator because it is "fast, efficient, and very stable" and "does a better job than most in fighting fragmentation" — bounded worst-case ~20%. Sizing rule: ~5% fixed overhead + worst-case 20% fragmentation → "only 75% of available memory should be allocated for actual malloc storage" (raised to ~90% with `lg_dirty_mult:8,lg_chunk:18`, at an ~8% perf cost). Defragmenting is rejected as "computationally expensive"; the design accepts bounded waste instead. High-but-stable RSS is the intended steady state, not a leak.
- Source: <https://info.varnish-software.com/blog/understanding-varnish-cache-memory-usage>

**Why arenas don't hand memory back** — LinkedIn Venice post-mortem: glibc malloc freed memory only when the *top* of the address range emptied — "as objects were deleted, memory would not go back down unless all objects created at the top of the address range were also removed, exposing the stack-like behavior of the glibc allocator"; "the total amount of unused memory could go as high as 30%". Nodes OOM'd across clusters "in batches" after weeks. Switching to jemalloc (size-classed bins, ≤20% bound, time-based decay return) removed the growth entirely. Operationally telling: "the weekly release cadence had inadvertently masked the issue" — restart cadence had been the accidental mitigation.
- Source: <https://www.linkedin.com/blog/engineering/infrastructure/taming-memory-fragmentation-in-venice-with-jemalloc>

**jemalloc decay knobs** (grade B): `dirty_decay_ms`/`muzzy_decay_ms` control gradual return of clean/dirty pages; `-1` disables purging entirely (GitHub jemalloc #1816, #1398). Varnish-class systems tune the decay to hold working sets resident; they never free-to-OS on the hot path.

**Redis.** Official latency doc: "Transparent huge pages must be disabled from your kernel" — with THP, after fork: "In a busy instance, a few event loops runs will cause commands to target a few thousand of pages, causing the copy on write of almost the whole process memory. This will result in big latency and big memory usage." Swapping is the other cardinal sin: paged-out pages stall clients on random-I/O swap-in; diagnosis is `/proc/<pid>/smaps` grep `Swap:`.
- Source: <https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency/>

**nginx** (grade B): shared-memory zones are fixed-size at config time and carved by `ngx_slab_alloc` size-classes; zone exhaustion is a hard `no memory` failure, not growth (<https://nginx.org/en/docs/dev/development_guide.html>, <https://blog.openresty.com/en/nginx-lua-shdict-fragmentation>). Same shape: fixed arena, bounded waste, no runtime resize.

## 6. Post-mortems: "degraded over uptime, reboot fixed it"

| Incident | Symptom | Root cause | Adopted fix | Grade |
|---|---|---|---|---|
| Splunk (current release notes) | ≥30% indexing/search degradation + latency on THP systems | THP "too aggressive at coalescing memory pages"; "can prevent the jemalloc memory allocation implementation from releasing memory back to the operating system"; I/O regressions swapping huge pages | Disable THP **and** direct compaction/defrag: "Either combination of having THP or direct memory compaction turned on can severely degrade performance" | A |
| Volt Active Data (2015) | Idle RSS climbed 21→50 GB; "page fault count... did not change" | khugepaged coalescing; "the remapping implicitly pages them in because of the zero-filling" — 512× RSS inflation; 25-line C repro, buddyinfo correlation | Disable THP or `madvise`-only | A |
| MongoDB ≤7.0 | Startup warnings; latency spikes from khugepaged contention | THP background compaction | Init-script disable before mongod starts. **8.0 flipped**: per-CPU-cache TCMalloc works with `THP=always + defrag=defer+madvise + khugepaged/max_ptes_none=0` — note the knob set that makes THP tolerable | A |
| PostgreSQL on Linux 6.8→7.0 (AWS, 2026) | pgbench throughput ~0.51x baseline | Kernel page-management change; massive TLB/page-fault cost without huge pages; process-per-connection amplifies | Enable huge pages; fix "may not be easy" (Phoronix) | B |
| LinkedIn Venice | Weeks-scale RSS growth → OOM killer, "nodes go offline in batches" | glibc allocator address-range stack behavior, ≤30% unusable holes | jemalloc with decay; restart cadence had been masking it | A |
| YugabyteDB TA-26440 (2025) | THP-driven RSS inflation, instability | THP | Disable THP | B |

The recurring root causes are exactly the four candidates named in our own incident: **THP/khugepaged interference, compaction stalls, page scatter (TLB), and swap**. Reboot "fixes" all four simultaneously, which is why reboots look curative.

## 7. Operational question: online re-compact/recycle instead of reboot?

**What the kernel offers (grade A, kernel docs):**
- `echo 1 > /proc/sys/vm/compact_memory` — synchronous compaction of all zones so "free memory is available in contiguous blocks where possible." A manual, blocking, all-zone stall.
- `vm.compaction_proactiveness` (0–100, default 20) — background kcompactd driven by a per-node fragmentation score; docs warn compaction moves pages "belonging to different processes" and "could also lead to latency spikes"; ">80" values risk "excessive background compaction activity".
- `vm.extfrag_threshold` (default 500) — compaction is skipped when the fragmentation index is at/below it; debugfs `/sys/kernel/debug/extfrag/extfrag_index` exposes per-order state.
- `vm.watermark_boost_factor` (default 15000 = 150% of high watermark) — reclaim on pageblock-mobility mixing "so that compaction has less work later".
- Sources: <https://docs.kernel.org/admin-guide/sysctl/vm.html>, <https://docs.kernel.org/admin-guide/mm/hugetlbpage.html>

**What vendors actually schedule:** only Alibaba cloud guidance surfaced recommending off-peak `echo 1 > /proc/sys/vm/compact_memory` on a timer, optionally with cache drops (grade C, single search source). No DB, DPDK, or HFT operator publishes a hugetlb-pool "defrag schedule." Oracle/PG/DPDK/HRT all pre-empt the need by boot-time reservation.

**The documented limit of online re-arming.** PostgreSQL's doc: runtime reservation may need "to repeat the command or to reboot." Kernel doc: boot cmdline is "the most reliable method"; runtime success "depends on the amount of physically contiguous memory that is present in system at the time of the allocation attempt." Once a box is fragmented, `compact_memory` is a gamble with a stall attached — not a guarantee.

**The established recycle is therefore: drain → restart process → re-arm, against a kernel-owned pool.** Freed hugetlb pages return to the pool on process exit (kernel doc: freed persistent pages "will be returned to the huge page pool"); the next process maps a pristine, contiguous, pinned pool in milliseconds. Oracle instances and PostgreSQL servers bounce daily under this model without any kernel reboot. **The performance-reboot is only necessary when the pool lives in reclaimable anonymous/tmpfs memory** — then every recycle re-rolls physical layout from a fragmented buddy allocator, and month-scale uptime degrades the layout under the resident set.

**Watchdog material** (documented counters, grade A): `HugePages_Free/Rsvd/Surp` (pool health), `/proc/vmstat compact_stall*`, `compact_fail`, `compact_success` (compaction pressure — our box: 1.66M attempts, 71% fail), `pgmigrate_success/fail` (migration churn), `thp_fault_alloc`/`thp_collapse_alloc`, `dTLB-load-misses` via perf, `tlb:tlb_flush` for shootdowns.

---

## 8. Synthesis — the extracted invariant

**One sentence:** every system that streams gigabytes per second for months without a performance reboot treats hot-path memory as a **boot-reserved, kernel-pinned, huge-page-backed pool** carved by its **own bounded-waste arena**, with **zero allocation and zero page-table mutation on the hot path**, and **THP/swap/compaction excluded by construction**.

**The five sub-invariants (all five, or the guarantee degrades):**
1. **Reserved, not allocated.** The pool is carved at boot (cmdline) or at earliest init, before fragmentation exists. Runtime reservation is treated as best-effort recovery, never the design. (PG boot cmdline; Oracle MOS 361468.1; DPDK sys_reqs; kernel doc "most reliable method".)
2. **Pinned and exempt.** Pool pages cannot swap, cannot migrate, cannot be reclaimed for other users. The kernel's normal memory lifecycle does not apply to them. (hugetlb: "cannot be swapped out"; DPDK pinned; Oracle LOCK_SGA.)
3. **Arena-carved.** The process owns an allocator over the pool: size-class/slab, bounded waste (jemalloc ≤20%), freed objects return to the arena — never to the general allocator, never to the OS mid-flight. (Varnish stevedores, Redis jemalloc, nginx slab, DPDK mempool, MongoDB TCMalloc.)
4. **Huge-page-backed.** The TLB must cover the streaming working set; 4K backing of the hot pool is a defect, not a fallback (Oracle's 2M→4K entry math; HRT 8 MiB vs 4 GiB coverage; PG `huge_pages=on` fail-fast).
5. **Deterministic page tables.** All mapping work happens at startup (mmap + populate + lock); afterwards nothing mutates page tables — no THP, no KSM, no NUMA balancing, no compaction-driven migration, no malloc/free internals on the hot path (rigtorp's shootdown list).

**Boot vs runtime, by system:**

| System | Reserved at boot | Armed at runtime | Pool owner |
|---|---|---|---|
| PostgreSQL | `hugepagesz=2M hugepages=N` or persisted `vm.nr_hugepages` | `postmaster -C shared_memory_size_in_huge_pages` computes N; `huge_pages=on` enforces | kernel hugetlb pool; freed back on shutdown |
| Oracle | `vm.nr_hugepages` (MOS 401749.1 sizing script); AIX `lgpg_regions` + `v_pinshm` | instance start; `LOCK_SGA=TRUE` triggers large-page attempt; alert log prints coverage % | kernel pool; SGA 100%-in-large-pages is the pass condition |
| DPDK | `hugepagesz=1G hugepages=N` preferred (contiguity) | EAL mmaps hugetlbfs at init; `dpdk-hugepages.py` runtime fallback | kernel pool + EAL heap on top; mempools carved once |
| HFT | hugetlbfs pool pre-sized (min pre-allocated, max hard) | startup mmap + mlock + pre-fault; allocators reserve via tunables | kernel pool; process arenas on top |
| Varnish/Redis/nginx | nothing kernel-side | fixed arenas sized at config/start; jemalloc decay tuned to hold, not return | process heap (bounded-waste allocator is the guarantee) |
| MongoDB 8.0 | init service sets THP `always` + `defer+madvise` + `max_ptes_none=0` **before** mongod | TCMalloc per-CPU caches (`rseq=0`) | THP pool — the one licensed exception, with the chaos knobs pinned |

**The recurring sizing/arming recipe:**
1. Size the pool to max steady-state working set + 10–20% headroom (PG: +1 page round-up and `min_dynamic_shared_memory`; Oracle: script + headroom; Varnish: 75–90% rule; DPDK: mempool worst case).
2. Reserve at boot cmdline; persist the sysctl as the second line of defense.
3. Enforce, don't hope: `huge_pages=on` / 100%-in-large-pages check / DPDK fatal at init. A silent 4K fallback is the bug that ships.
4. Leave the rest of RAM alone — the pool is banked, everything else keeps normal kernel service.
5. Verify at arm time: `HugePages_Rsvd` == pool size mapped, alert-log/`pmap` evidence.
6. Monitor in flight: pool free counts, `compact_*` vmstat counters, TLB miss rate, `Swap:` per mapping.
7. Recycle = process restart, never pool release and never kernel reboot. The pool outlives the process; that is the entire trick.

**Application to this box (mapping only, not an implementation plan):** the engine already satisfies the recycle half (several restarts/day). The missing half is kernel ownership: ~80–130 GiB anon expert-weight pool plus ~107 GiB tmpfs placement store are reclaimable, 4K-backed, compaction-eligible, and swap-capturable (zram full at day 6). The precedent-mapped shape is a boot-reserved hugetlb pool (2 MiB or 1 GiB) covering the steady-state working set with headroom, the placement store on a hugetlbfs mount (shmem THP stays off, so tmpfs will never self-promote), `huge_pages`-style fail-fast arming at engine start, and a watchdog on `HugePages_Free` + `compact_stall*` + `dTLB-load-misses`.

---

## Sources

Primary (fetched and quoted):
- PostgreSQL: <https://www.postgresql.org/docs/current/runtime-config-resource.html>, <https://www.postgresql.org/docs/current/kernel-resources.html>, <https://www.cybertec-postgresql.com/en/huge-pages-postgresql/>
- Oracle: <https://www.oracle.com/technical-resources/articles/it-infrastructure/dev-hugepages.html>, <https://docs.oracle.com/en/database/oracle/oracle-database/18/unxar/tuning-oracle-database.html>
- Kernel: <https://docs.kernel.org/admin-guide/mm/hugetlbpage.html>, <https://docs.kernel.org/admin-guide/sysctl/vm.html>
- DPDK/SPDK: <https://doc.dpdk.org/guides/linux_gsg/sys_reqs.html>, <https://www.dpdk.org/memory-in-dpdk-part-1-general-concepts/>
- HFT: <https://www.hudsonrivertrading.com/hrtbeat/low-latency-optimization-part-1>, <https://www.hudsonrivertrading.com/hrtbeat/low-latency-optimization-part-2>, <https://rigtorp.se/virtual-memory/>, <https://aeron.io/docs/aeron/log-buffers-images>, <https://theaeronfiles.com/aeron-transport/log-buffers>
- Allocators/long-uptime: <https://info.varnish-software.com/blog/understanding-varnish-cache-memory-usage>, <https://www.linkedin.com/blog/engineering/infrastructure/taming-memory-fragmentation-in-venice-with-jemalloc>, <https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency/>, <https://nginx.org/en/docs/dev/development_guide.html>
- Post-mortems: <https://help.splunk.com/en/splunk-enterprise/release-notes-and-updates/release-notes/9.4/known-issues-for-this-release/transparent-huge-memory-pages-and-splunk-performance>, <https://www.voltactivedata.com/blog/2015/07/getting-around-transparent-huge-pages-trap-linux/>, <https://www.mongodb.com/docs/manual/tutorial/transparent-huge-pages/>

Secondary (search-verified, not page-fetched):
- <https://www.phoronix.com/news/Linux-7.0-AWS-PostgreSQL-Drop> (AWS/Linux 7.0 Postgres regression, 2026-04)
- MOS notes 1557478.1, 361323.1, 361468.1, 749851.1, 1134002.1, 401749.1 (titles via Oracle article)
- <https://blog.openresty.com/en/nginx-lua-shdict-fragmentation>, <https://github.com/jemalloc/jemalloc/issues/1816>, <https://github.com/jemalloc/jemalloc/issues/1398>
- <https://www.percona.com/blog/settling-the-myth-of-transparent-hugepages-for-databases>, <https://access.redhat.com/solutions/2728441>
- Alibaba off-peak `compact_memory` scheduling and Oracle blog "Linux Memory Compaction: Internals and Debugging" (grade C — not fetched)
