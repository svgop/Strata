---
type: research-digest
id: 2026-11-09-regression-memory-state
date: 2026-11-09
scope: system-memory side effects of the reserved 88 GiB hugepage pool on the GSQ IQ3_S serving path; what holding 88 GiB back changes, priced per mechanism
subject: /proc state (buddy order-9 supply, watermarks, compact deltas), page-cache residency of serve-time reads (dense r--s mapping, PLE mmap stream, file tier), KV host shadow THP era table, zram, pool-cost verdict
method: live /proc + smaps evidence during operator engine cycles (no engine booted by this probe; a 256 MiB MADV_HUGEPAGE probe and 60 s vmstat deltas were the only perturbations), log era-classification over logs/custom-iq3_s.log (669k+ lines, dated "engine started" markers), source reads (hugepage.hpp, layer.cpp, shadow_experts.cpp, expert_source.cpp, generate.cpp)
---

# Regression memory state — what the 88 GiB reservation does to the serving path

## Verdict (read this first)

**The pool's RAM cost is real but it does not land on the pinned decode path — it lands on everything around that path.** Four measured mechanisms, in priced order:

1. **Cache ceiling halved (the big one).** Nov-6: ~130 GiB free-ish; the whole 77.9 GiB pack (51.1 + 26.8 GiB GGUF shards) plus PLE sat in page cache — zero serve-time disk reads. Pool era: `Cached` 44-65 GiB with the engine up, `MemAvailable` 53.5 GiB settled (14.4-20 GiB in today's squeeze cycles). Every still-file-backed serve path now competes for roughly half the cache: the dense-weight mapping (51.1 GiB mapped, 31.3 GiB hot resident, `Referenced` 30 GiB — actively read), the PLE n-gram stream (`--ple-io mmap`: 26.8 GiB mapped r--p, **0.1-1.0 GiB resident** — streamed from cache/disk on demand), and the `--mmap-experts` file tier whose own boot line gates decode read-ahead "only while the page cache has room for it and it pays" (`STRATA_IO_PF_GATE=1`) — the engine itself prices this debit.
2. **THP supply for small arenas collapses while the pool is drawn, and heals at serve.** With the pool consumed, Normal-zone buddy order-9/10 = 0/0 blocks. KV host shadow (0.26 GiB × 91 windows): 58-84 of 91 windows `(rejected)` = 4 KiB during Nov-7-19:xx and today's load starts; 91/91 `(HUGETLB POOL)` on the brief STRATA_POOL_SMALL_ARENAS-era runs; **88-91/91 `(THP accepted)` at settled serve today**. A 256 MiB `MADV_HUGEPAGE` probe on the aged box mid-load: 200-207 of 256 MiB landed on 2 MiB pages (76-79 %), control 0 — supply is degraded, not dead.
3. **Load-time tax per engine start.** 60 s window spanning a teardown+load: kswapd stole **23.0 M pages (87.7 GiB)**, `allocstall_movable` +16,236, `compact_stall` +1,910 with 1,732 failures (91 %), `pswpout` +14,386 pages. Boot-to-date (22.4 h, all post-pool): `pgsteal_kswapd` 608 M pages (2.32 TiB, ~105 GiB/h recycled file cache), `compact_fail` 155,666 — vs the ≤1,000/cycle mechanism gate in `2026-11-07-preboot-aging-proof.md` (H3). Seven-plus engine cycles today each paid this.
4. **Draw-order all-or-nothing fallback (observed live, 11:05 today).** When the complement takes the pool first, `huge_pool_covers()` fails for the 31.64 GiB store and it lands on the **swappable 4 KiB tmpfs tier** (`STRATA_SHADOW_EXPERTS=/dev/shm/strata-shadow-q2_0`, no "2 MiB pool pages" note, log:667335) — while adjacent runs got the memfd pool store (populated 4.5-16.8 s). In that squeezed state zram hit **7.69/8 GiB (94 %)** and the engine itself held 840 MiB swapped.

**Exonerations (with numbers):** NUMA — single node0, `numa_miss 0`; no placement mechanism exists. DMA coherency — no mechanism; pool pages are ordinary cache-coherent RAM, and the 2 MiB backing *improves* the GPU DMA walk (~9 M PTEs → ~18 k, priced +5-15 % on the staging path at `include/strata/platform/hugepage.hpp:2-4`). Complement/store reads on pool are strictly better than the Nov-6 anon-pinned 4 KiB originals, and complement pin time dropped from 87-104 s (Nov-7 pre-pool, fragmented-buddy struggles) to 4.5-18 s (pool era).

**Attribution boundary:** the Nov-6 → now delta (229.1/224.8/303.5 → 175-195 solo) spans the GPU clock-governance v2 incident (a measured **uniform ~30 % lane loss** on Nov-7, `2026-11-07-gpu-clock-governance`), engine config churn, and the pool. Memory evidence above cannot price the pool's share alone; the discriminating A/B (same binary+config, pool armed vs `nr_hugepages=0`, drill exists in `campaign/HOSTMEM-NO-REBOOT.md`) was **not run** — engines were booting under the operator all session.

---

## 0. Session state discipline

The box never returned to the assumed "engines down" state: the operator launched/cycled engines 7+ times today (10:41, 10:46, 10:49, 10:52, 11:02, 11:04, 11:0x — dated `engine started:` markers at log lines 655652…666469). This digest captures **two** states: settled-serve (10:46, PID 512634, `--resident-experts` pinned tier — the task's described architecture) and squeeze (11:07). All reads were passive `/proc`; the only actives were a 60 s vmstat delta and the 256 MiB THP probe. Kernel `7.2.12-300.fc45.x86_64`, boot 2026-11-08 12:23 — **every vmstat counter below is pool-era only** (pre-pool counters died with the Nov-8 reboot).

Boundary: `/proc/pagetypeinfo` is root-only on this kernel (`cat` → Permission denied; `sudo -n` needs a password). Buddyinfo + zoneinfo carry the substitute evidence. Repo has no `.docs/log.txt`/`index.md`/`technical_summary.md` (named gap; logger-owned, not agent-edited).

## 1. Current state

**Settled serve (10:46, engine up, pinned tier, Private_Hugetlb 69.5 GiB = complement 37.86 + store 31.64):**

| field | value |
|---|---|
| MemTotal / MemFree | 184.2 GiB / 5.5 GiB |
| MemAvailable | **53.5 GiB** (Nov-6 era: ~130 GiB) |
| Cached | 51.0 GiB (incl. dense mapping's 31.3 GiB resident) |
| AnonPages / AnonHugePages | 34.0 GiB / 14.6 GiB (THP=madvise; shmem_enabled=never confirmed) |
| HugePages_Total / Free | 45076 (88.03 GiB) / **9493** → 69.5 GiB drawn (35,583 pages = complement + store, exact) |
| Unevictable | 260 MiB (complement is on hugetlb, not mlock — pool pages leave Unevictable) |
| engine Swap | 0 |
| zram | 3.9 GiB used; holders = desktop (Steam 419 MB, gnome-shell 317 MB, Xwayland 228 MB, chrome, dnf5daemon) |
| buddy Normal (order9/10) | **0 / 0** blocks (sub-2 MiB free ≈ 4.8 GiB — the whole Normal free list is fragmented below order-9) |
| buddy DMA32 (order9/10) | 38 / 53 blocks (DMA32 untouched by the pool: managed 1.86 GiB) |
| zone Normal watermarks | min 65 MiB / low 252 MiB / high 438 MiB; managed 182.3 GiB incl. pool pages |
| DirectMap | 1 G: 75 GiB, 2 M: 62.7 GiB, 4 k: 50.6 GiB (1 G mappings intact — the pool did not shatter direct maps) |

**Squeeze (11:07, next engine loading + tmpfs store present):** MemAvailable **14.4 GiB**, SwapFree 506 MiB (zram 94 % full, orig 7.18 GiB compressed to 4.50 GiB), Shmem 37.4 GiB (`/dev/shm/strata-shadow-q2_0`, the F-367 tmpfs tier), loading engine: Pss_Anon 33.3 GiB, Private_Hugetlb 40 GiB (complement mid-pin), **Swap 840 MiB on the engine itself**. Buddy Normal order-9 0, order-10 0; order-0 1.2 GiB.

**Compact deltas:** 22.4 h totals — `compact_stall` 221,672 / `compact_fail` 155,666 / `compact_success` 66,006; `thp_fault_alloc` 61,621 vs `thp_fault_fallback` 35,203 (36 % fallback rate); `thp_scan_exceed_swap_pte` 89,935 (khugepaged refuses collapse over swap ptes — the rejection-healing path is swap-pressure-sensitive). Per-cycle: +1,732 fail in one load minute — 1.7× over the aging-proof ≤1,000/cycle gate **during load cycles only**; no clean steady-window sample existed all session.

## 2. Which serving reads are still file-backed at steady state

Everything below measured from PID 512634 `/proc/smaps` at settled serve (10:46):

| mapping | flags | Size | Rss | Referenced | role |
|---|---|---|---|---|---|
| shard 00001 (`--native`, dense+pack) | `r--s` MAP_SHARED | 51.1 GiB | **31.3 GiB** | 30.0 GiB | dense hot set lives in page cache; Ref≈Rss = actively read at serve |
| shard 00002 (`--ple-gguf`, PLE) | `r--p` MAP_PRIVATE | 26.8 GiB | **0.1-1.0 GiB** | ~equal Rss | `--ple-io mmap` **streams** it: every miss is a cache hit or a disk read |
| memfd `strata-shadow-q2_0` | r--s, MFD_HUGETLB | 31.64 GiB | (Private_Hugetlb) | — | pool-backed, unevictable, populated by 256 MiB memcpy in 4.5-16.8 s |
| expert complement | (huge_alloc) | 37.86 GiB | (Private_Hugetlb) | — | pool-backed copy filled from the shard mapping; `lock_resident` only on the non-pool path |

- `STRATA_ARENA_MMAP` default is **off** (`src/core/expert_source.cpp:5613` — env opt-in for small-RAM boxes; mapped arena is "not locked, not pinned"). The default arena is the pinned pool/anon complement, not a file mapping — so the *arena* is cache-independent; the *shards* are not.
- PLE: `--ple-io` default is `direct` (unbuffered SSD, `src/program/generate.cpp:558, 896`); this engine runs `mmap` → `PleIo::Mmap` (`generate.cpp:4131`) → **still file-backed at serve time**, now competing for a halved cache.
- Dense weights are copied out to private buffers at load (`src/core/native_dense.cpp` has no file reads; allocs granule-rounded); the r--s mapping stays resident because the file tier + first-touch + metadata keep reading it. Under the `--mmap-experts` tier (newest runs), experts are computed "from the mapping (page-cached)" with decode read-ahead gated on cache room (`STRATA_IO_PF_GATE=1`) — that tier is **directly exposed** to the cache debit; the pinned tier (`--resident-experts`, the 10:46 run) is not.
- `MemAvailable` does not count the pool (`hugepage.hpp:33-49` states it and ships `huge_pool_free_bytes()` as the correction; the ds-pool-sizing digest quantified the trap: MemAvailable 78.02 GB → copy budget `can` 69.43 GB < `all` → DS copy silently off). Any budget math on this box must add `HugePages_Free × 2 MiB`.

## 3. KV host shadow — did the line read differently before the pool?

Yes, and the log era-classifies it cleanly (91 report lines per run ≈ per-window allocations; semantics from `hugepage.hpp:182-196`: `HUGETLB POOL` = VmFlags `ht`, `THP accepted` = THP-backed > 0, `rejected` = 0 huge bytes):

| era | runs | shadow outcome per run (of 91) |
|---|---|---|
| Nov-6 (pre-pool, pre-line) | 607294-614508 | line did not exist yet (code added Nov-7) — no baseline |
| Nov-7 19:xx, pool arming day, THP only | 635511, 638405 | 33/91 and 10/91 accepted, max 0.09 GiB — **mostly 4 KiB** |
| Nov-7 20:xx, pool armed, small arenas allowed | 641714, 642375, 643662 | **91/91 `(HUGETLB POOL)` 0.26/0.26** — perfect |
| Nov-7 20:xx, pool busy (complement+store first) | 643017, 644300, 644940, 647796 | 7-34 accepted, 57-84 rejected — the starvation that wrote the ≥8 GiB gate |
| Nov-8 13:xx (fresh post-reboot buddy) | 651214, 652671, 654179 | **91/91 accepted**, 0 rejected |
| Nov-9 10:xx (aged buddy, churn) | 655652-662039 | 30-88 accepted; rejections cluster at load start |
| Nov-9 settled/late runs | 662751, 664608 | **91/91 accepted** 0.26 |

So: the "0.00 GiB of 0.26 GiB" lines are real but they are the **load-transient and pool-exhausted** states, not steady serve; steady serve today heals to full acceptance. Per-window cost when rejected: the 0.26 GiB window runs 4 KiB pages → the KV staging DMA/dTLB walk pays the tax `hugepage.hpp:2-4` priced at +5-15 % *on the staging path* — and that path is small here: the warmup report reads "KV streaming: 97.92 % of 13,824 block reads hit VRAM, 1.2 MiB read from RAM". Residual risk: healing goes through khugepaged/compaction, which `thp_scan_exceed_swap_pte` (89,935) shows is blocked under swap pressure — exactly the squeeze state.

Gate context (`hugepage.hpp:88-97`): only ≥ 8 GiB organs draw the pool by default; the 2026-11-07 measurement ("~90 per-layer KV shadows drew 23.4 GiB in the first seconds of boot and starved the 37.6 GiB complement's cover check") is why the perfect `HUGETLB POOL` runs no longer occur. `STRATA_POOL_DIET=1` in the engine env is the CPU **thread**-pool park flag (`src/kernels/cpu/pool.cpp:729`) — unrelated to the memory pool.

## 4. zram / swap pressure

Settled serve: 3.9 GiB in zram, holders all desktop, **engine Swap 0** — the pool did not push the serving path into swap. Pool era since boot: `pswpout` 7.57 M pages (29.5 GiB) / `pswpin` 5.14 M — desktop anon evicted by the halved cache, a kswapd work-amp but not a lane cost at decode. Squeeze state is different: zram 94 % full, engine anon swapped (840 MiB), tmpfs store pages now swap-eligible — the concurrent-plan state the pool-sizing digest warned about (`91-strata-hugepages.conf` declares 54,272 pages; **45,076 armed, 9,196 short** — the runtime-raise boundary).

## 5. Mechanism ledger

| # | mechanism | direction | evidence |
|---|---|---|---|
| M1 | cache ceiling halved → file-backed serve reads (dense mapping first-touch, PLE mmap stream, file-tier experts) can miss to disk | **debit, priced** | §2 table; `Cached` 51 GiB vs 130 GiB era; `STRATA_IO_PF_GATE` exists to gate on it |
| M2 | kswapd file-cache recycling | debit at load/restart; background at serve | +87.7 GiB pgsteal in one load minute; 105 GiB/h boot-to-date; pgscan_file = 89 % of scan effort |
| M3 | compaction during serve | debit confined to load windows | compact_fail 1,732/min during load vs ≤1,000/cycle gate; Normal order-9 = 0 while pool drawn |
| M4 | THP quality of small arenas (KV shadow windows) | debit mid-load, healed at serve; worst under swap pressure | §3 era table; thp_fault_fallback 35,203; probe 76-79 % coverage |
| M5 | draw-order store fallback to swappable tmpfs | debit, conditional | log:667335 vs :666471/:669174; `huge_pool_covers` all-or-nothing (`shadow_experts.cpp:159-176`) |
| M6 | zram saturation in squeeze | debit, conditional (multi-tenant) | §4; engine Swap 0 when settled |
| M7 | complement/store read path | **credit** | 2 MiB unevictable pool pages vs Nov-6's 4 KiB anon; pin 87-104 s → 4.5-18 s |
| M8 | NUMA placement of pool pages | exonerated | 1 socket / 1 node0, `numa_miss 0` |
| M9 | DMA coherency | exonerated | no mechanism; pool improves PTE walk 9 M→18 k (`hugepage.hpp:2-4`) |
| M10 | watermark/kswapd at steady serve | exonerated at settled state | Normal free 470 MiB-3.2 GiB > low 252 MiB when settled; dips only during load |

**Bottom line:** in the pinned tier (`--resident-experts` + memfd store), the pool makes the decode path *better* than Nov-6 and the 175-195 vs 229-303 gap cannot be charged to pool RAM from memory evidence — the decode working set (complement 37.9 + store 31.6 + dense hot set 31.3 resident) fits the box with room. The debit is real but adjacent: it prices in whenever serving relies on still-file-backed paths (PLE stream, file tier) or whenever the box runs load/squeeze cycles (M2/M3/M5/M6). The confound ledger (GPU clock v2's measured ~30 % lane loss, config churn) must be cleared before any residual is attributed to memory. Discriminating experiment, not run here: same binary+config, pool-armed vs pool-drained (`campaign/HOSTMEM-NO-REBOOT.md` drill), same warm battery, watch `majflt`/minor-fault rate on the engine and `Cached` floor.

## 6. Boundary notes

- `/proc/pagetypeinfo`: root-only on kernel 7.2.12 (sudo needs password) — buddyinfo/zoneinfo substituted; the pageblock-fragmentation question is answered at block granularity: Normal order-9/10 = 0 while the pool is drawn, order-9 ≤ 15 blocks after teardown, and pageblock-level detail needs a root pass (next probe: `sudo cat /proc/pagetypeinfo` under the operator's hand).
- Pre-pool vmstat totals are unrecoverable (reboot Nov-8 12:23 wiped counters); Nov-6-era comparisons rely on log-derived behavior and the 2026-11-07 digests' recorded figures (MemAvailable 78.02 GB, HugePages_Total 24,369 mid-arm).
- Engines cycled under the operator all session; no steady-state idle snapshot exists. Pin-time and KV-shadow statistics are per-run complete (91 windows), state samples are point-in-time.
