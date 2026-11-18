---
type: research
id: system-state-differential
date: 2026-11-18
lane: qwen (GSQ-RCO iq3_s, port 8081, engine pid 142199, cgroup strata-engine-142019.scope)
scope: read-only system-state differential, GSQ lane regression ~20% vs 2026-11-10 bank era; zram / dead shadow store / page-cache squeeze / NTFS transport / mlock doctrine / bank-era records
related: 2026-11-18-config-diff-since-bank.md (row 1 = the disarm, row 2 = ntfs3 transport), [[strata-gsq-full-recovery-2026-11-09]], [[strata-estate-consolidation-2026-11-15]]
---

# System-state differential vs the bank era — qwen lane, 2026-11-18 18:22–18:37 SAST

Study method: read-only. Two live snapshots bracket the live lane's t66/t67 benches
(17:51–18:08) and the current engine boot (18:12:48). Engine start banner in
`logs/custom-iq3_s.log` line 131639 confirms the disarmed configuration
(no STRATA_SHADOW_EXPERTS, no store mapping); the last armed banner is line 124808
(pre-rollback boots).

## 1. Headline state, first snapshot (18:22:30) vs bank era (2026-11-10)

| metric | 18:22 today | bank era witness (Nov 10 boot audit, [[strata-gsq-full-recovery-2026-11-09]]) |
|---|---|---|
| zram used | 8,380,624 / 8,388,604 kB — **100% FULL** | **4 kB** of 8 G (post-reboot/pre-GSQ audit); 37.7 MB with DS live |
| /dev/shm | 37.55 GiB = dead strata-shadow-q2_0 (Oct 13, unmapped, lsof empty) | 38 GiB shadow store **ARMED** — mapped, guard-mlocked, productive |
| MemFree / MemAvailable | 1.23 GiB / 62.8 GiB | fresh boot, empty zram; recorded comparators: t26 (Oct 21) 109.9 GiB, vram-deficit boot (Nov 9) 48.9 GiB |
| engine VmSwap / cgroup swap.current | **0 / 0** (guard live) | 0 / 0 (C7 witness) |
| hugepages pool | 0 (doctrine: unarmed-by-doctrine OK) | 0 (landmine removed) |
| GPU clocks | 2940/13801 pinned through t66/t67 + metrics.jsonl (46–48 C, 95–97 W) | 2962/13801 held |

Clocks, pool, and swap-guard all match the bank posture. The differential is: dead
store, full zram, starved page cache.

## 2. zram: who is swapped, when it filled, who pays

Live VmSwap census (per `/proc/*/status`, both snapshots):

- 120–121 pids hold swap, total **2,342,852 kB (~2.23 GiB)** — stable across the study.
- Top holders (kB): chrome zygote 435,320; dnf5daemon-server 136,056; gnome-software
  131,912; chrome main 99,004; chrome crashpad 84,152; chrome gpu 74,712; zcode-cli
  67,240 + 5 more instances 52–64 MB each; gnome-shell 65,996; ibus-typing-booster 41,620.
- **Engine pid 142199: VmSwap 0. Cgroup `memory.swap.current`: 0.** `memory.events`:
  oom 0, oom_kill 0. The lane pays zero swap tax.

Swap accounting closure (first snapshot): SwapUsed 8,380,624 kB − live VmSwap
2,342,852 kB = **5.76 GiB orphaned slots**. Shmem reconciliation: meminfo Shmem
99.1 GB − engine Pss_Shmem 65.75 GB (its own shared-anon pools, mapped) ≈ 31.8 GiB
unmapped tmpfs ≈ /dev/shm store resident portion. 31.8 GiB resident + 5.76 GiB
swapped = 37.56 GiB = the store's exact size. **The dead store had ~15% of its pages
swapped out into zram (dirty tmpfs must swap, not drop), and those slots held zram
full.** zram fill was gradual (since-boot avg so=826 kB/s over the 3 h boot; pswpout
2,190,220 pages ≈ 8.34 GiB), driven by the armed-engine teardowns (17:22 rollback
unmapped the store, its pages became swappable) plus desktop anon.

Kernel journal since 2026-11-10: **no OOM kills, no pswperr, no swap warnings** —
only boot-time zram setup lines. No zram incident is logged; the fullness is silent.

**Mid-study event (named boundary):** between 18:22:30 and 18:32, SwapFree jumped
7,980 kB → 6,259,224 kB (**5.97 GiB freed**) with pswpin +661 pages only (no readback),
no tmpfs removal (store still present, df unchanged), no device reset (mm_stat
mem_used_max preserved at 6.62 GB). One-time orphan-slot release; actor unattributed
(desktop active — Steam dir appears 18:35). State stable since: 2 GiB used / 6.1 GiB
free, census unchanged. Verdict numbers below cite the study-start state, which is
the state the t66/t67 benches ran under.

## 3. The dead 39.3 GiB shadow store: the squeeze quantified

Composition (`du`, mtimes Oct 13): experts.bin 33.97 GB + dense.bin 5.84 GB +
embd.bin 0.48 GB + meta 1.6 MB. Engine references: **zero** — no /dev/shm VMA in
`/proc/142199/maps` (full map walk), lsof empty, and the current engine's own log
serves misses from the file tier ("blob reads from the file … the GGUF in place").

Cost while dead:

- **~31.8 GiB RAM permanently resident** (unmapped dirty tmpfs; not reclaimable while
  zram is full — tmpfs dirty pages only exit RAM via swap).
- **5.76 GiB of its pages already in zram** (study start), holding zram full.
- Removal (live-safe, engine holds no references) returns ~31.8 GiB RAM immediately
  and frees the 5.8 GiB of slots (now already freed by the mid-study event), i.e.
  **~36 GiB of physical headroom** that becomes page cache.
- Bank-era comparison: same RAM class, opposite sign. Bank era the store was armed —
  mapped by the engine, guarded by the ea56db09 store-mlock (measured then: 299 MiB
  of it was swap-resident without the lock), and **productive: it served decode CPU
  misses at RAM speed**. Today it serves nothing and taxes everything.

## 4. The 53 storage major-faults/round: what faults, what it costs

- Engine cgroup `memory.stat.pgmajfault`: 1085 at first read, 1251 ~12 min later
  (engine up 16 min, ~20+ rounds) — **≈ 53/round, matching the operator's number**.
  The fault source is the file tier's pread misses (log: "(0 major faults)" on
  cache-hit requests, "(28 major faults)" on a 403 MB-read request) — the PLE/gguf
  mmaps themselves are resident (shard1 Rss 28.7/52.3 GiB, shard2 18.3/27.5 GiB) and
  the pool is anon.
- **Engine steady-state NTFS traffic is real but small**: read_bytes grew 51.78 →
  54.28 GiB across ~10 min (~3.5 MB/s avg; 104 MB in one 15 s window = 7 MB/s, exactly
  matching nvme2n1 read_sectors delta). Majors ≈ 0.7/s.
- **Lane-visible file-tier telemetry (per request)**: "the OS read 25–67 MB from
  storage … for 220–270 MB of expert reads; page cache held 195–249 MB" early in the
  lane; degraded to "**OS read 222–403 MB from storage** … page cache held 444 of
  536 MB / 769 of 892 MB / 897 of 1023 MB" under bigger (conc-class) requests, with
  "**decode waited 15.0 → 74.2 → 124.2 ms** in 113–241 fetches". This wait is the
  system-level tax on every miss that bank era served from the RAM store at ~75 us.
- Startup bulk is one-time: 46 GiB page-in of the mapped ggufs (read_bytes 48.2 GiB
  ≈ resident file pages), server.py preload 72 GiB rchar. nvme2n1 since boot: 1.43 TB
  read total, writes negligible.
- ntfs3 mount: `rw,nosuid,nodev,relatime,uid=1000,gid=1000,acl,iocharset=utf8,prealloc`
  on nvme2n1p2. The ntfs3-vs-ext4 transport delta remains UNISOLATED (config-diff
  row 2, UNKNOWN) — no A/B exists. NTFS is not an independent driver; its cost exists
  because the disarm moved misses onto it.

## 5. Pool/mlock doctrine check (ea56db09)

- `smaps_rollup`: Rss 126.4 GiB = Pss_Shmem 65.7 (shared-anon pools: 37.73 GiB
  page-locked expert resident pool + KV staging "3.09 GiB of pinned RAM" + cache) +
  Pss_File 46.3 (the two gguf mmaps + libs) + Pss_Anon 12.1. VmLck/VmPin = 0.
- VmLck 0 is **doctrine-consistent, not a guard failure**: the store-mlock ships in
  the shadow-store mapping path (`shadow_experts.cpp`); the store is disarmed, so
  there is nothing to mlock. The pool is pinned by `memory.swap.max=0` on the scope
  (swap.current 0 across the study despite 24.9M direct-reclaim scans) — the M4 move
  working as designed. Driver-owned CUDA pins (cvec, KV, DMA staging) are not
  process-mlocked by design ([[strata-gsq-full-recovery-2026-11-09]] warning).
- Cgroup limits max/max, sock_throttled 3, zero pressure events.

## 6. Bank-era records cross-check

- `campaign/results/fork-ud-first-bank.md` (Nov 6–9): the Nov-7 regression hunt's
  zram precedent — "ZRAM SWAP 100% FULL, live paging (si=452/s) — cleared by the
  operator. Partial fix"; Nov-9 store slow-mode: **24.4 ms/window CPU experts (10x)
  on the zram-full box** when the store lost its RAM tier. Today's zram-full state
  with a disarmed store is the same pathology class, one tier further out.
- Nov-10 boot audit: zram 4 kB, pool 0, store armed 38 GiB, engine swap.max=0,
  isolation cmdline active, warm bench 227.6/210.1/293.9 (record class).
- t66-warm-recheck (17:51–17:57) + t67-stock-headtohead (18:04–18:08), from
  `metrics.jsonl`: clocks 2940 pinned, ITL p50 ~12 ms, acceptance 61–65.6%, golden
  drift line present — benches ran on the zram-full, store-dead box.
- MEMORY.md bank: "engine binary exonerated (bank-era build), shadow rollback = the
  ~20% cost (cpu-exp 2x), dead 39.3 GiB /dev/shm store squatting RAM, swap guard
  verified working" — this study quantifies the last two terms and adds the
  page-cache mechanism.

## 7. Verdict — ranked system-level contributors to the ~20%

Primary cause (engine-level, not system): the STRATA_SHADOW_EXPERTS disarm
(config-diff row 1) — cpu-exp 2x, banked. The system-level stack beneath it:

| rank | contributor | evidence | size / lane-visible cost | fixable without stopping the lane |
|---|---|---|---|---|
| 1 | **Dead 37.5 GiB shadow store squat** | maps+lsof: zero references; 31.8 GiB resident + 5.76 GiB swapped (study start); arithmetic closes to the byte | starves the page cache the file tier depends on; converts RAM-speed misses into NTFS reads | **YES** — `rm -rf /dev/shm/strata-shadow-q2_0` (launcher rebuilds at next Qwen boot); returns ~36 GiB physical headroom |
| 2 | **Page-cache squeeze / reclaim churn** | MemFree 1.2 GiB, MemAvailable 62.8→52.4 GiB across the study; pgscan_direct 24.9M, compact_stall 22,784, compact_fail 10,518 in 3 h; lane cgroup workingset_refault_file 11.3M pages (~43 GiB); file tier "page cache held X of Y" gaps + decode waits 15–124 ms/request | the measurable per-request tax: storage reads 25→403 MB/request, 0.7 majors/s, wait term inside the window budget | partial — shrinks as a consequence of (1); full fix needs re-arm (restart-gated, blocked on latent-bug audit) |
| 3 | **zram state** | 100% full at study start (5.76 GiB store orphans + 2.23 GiB desktop; engine 0); Nov 10 bank era: 4 kB; mid-study one-time 5.97 GiB release restored 6.1 GiB headroom (actor unattributed, stable since) | direct lane tax: **zero** (guard verified); indirect: full zram blocked tmpfs reclaim, freezing the 31.8 GiB | already largely self-resolved this session; recycle not needed mid-lane |
| 4 | **NTFS transport (ntfs3)** | 3.5–7 MB/s sustained lane reads, majors negligible, nvme2 healthy, no fs errors in journal; ntfs3-vs-ext4 delta UNISOLATED | not a standalone driver — its cost exists only via the disarmed file tier | NO — any transport A/B or re-arm requires a restart; not justified by evidence |
| — | Exonerated | engine binary (bank-build interleaves), clocks (2940 pinned through benches), swap-in stalls in lane (engine VmSwap 0, cgroup swap.current 0, si=0 steady), hugepage pool (0, doctrine), OOM (journal clean since Nov 10, memory.events clean) | — | — |

Falsifier: after a live `rm` of the store, watch the lane's own file-tier telemetry —
"OS read MB from storage" per request should collapse toward cache-only levels and
"decode waited" should shrink to the sub-10 ms class, while cpu-exp stays ~2x (the
disarm's own cost). If cpu-exp also drops, this ranking under-attributed the squat;
if neither moves, the squeeze ranking is wrong. Expect partial recovery, not the full
bank class — full recovery is the re-arm (engine restart, blocked on the latent-bug
audit per MEMORY.md).
