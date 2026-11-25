# F-453 working notes (2026-11-25, night window) — dated findings, evidence-first

This file is the round's running record; the F-entry in SVG-CUSTOM.md is its final form. Every number
below was read from the named file or command on this date.

## 01:51–02:00 — the box, the boot, the placement

- `/proc/cmdline`: no `isolcpus` / `nohz_full` / `rcu_nocbs`. `grubby --info=ALL`: NONE of the four boot
  entries (7.2.15, 7.2.12, 7.2.10, rescue) carries them, and `/etc/default/grub` does not either. The recipe in
  `docs/PERFORMANCE-SETUP.md:38` is `grubby --update-kernel=DEFAULT --args=...` — it edits one entry and never the
  default file, so a kernel install drops it (the incident the same doc §3 records). Restoring it is a boot-config
  decision for the operator; it is not an engine change.
- **"Unpinned" was never the engine's state.** Boot 01:58:51, engine pid 1001126, banner: `15 pool workers on logical
  processors 1..15, the host thread on 0 (draining too) (--host-core first)`; aux plan `CPUs 16 (off at start)`.
  Per-thread affinity (`taskset -pc` over `/proc/<pid>/task/*`): `strata-w0..w14` on CPUs 1..15; `strata-stdin`,
  `strata-io` and three `strata` threads on CPU 0; 11 driver threads on 0-31. F-451 II.5's "the engine ran
  unpinned all day" is wrong: it ran the engine's fallback plan (host on the housekeeping core, workers across both
  CCDs). `src/program/generate.cpp:1935` is NOT a start-up affinity reset (it widens one job-team thread).
- **The pin interleave F-451 proposed is a banked negative, three times over** (law 8):
  whole-process taskset `-20%` (F-285b era, cited at SVG-CUSTOM:10001); `STRATA_POOL_CORES="8-15,24-31"` F-295
  M1 -13.0 / pair -13.9 / conc -20.3% ("the pool wants the WHOLE topology", SVG-CUSTOM:9991-10010);
  `STRATA_AUX_CPUS=auto` F-363 M1 -7.7 / conc -8.2% (SVG-CUSTOM:11694). F-363's own unpinned control boots read
  conc 315.2 and 307.5. DROPPED. `docs/PERFORMANCE-SETUP.md:15` and `:40` recommend the falsified pin - stale.

## 02:01–02:04 — standing re-bank on that boot (device launch on; `stand-leg{0..3}.txt`)

| leg | M1 | pair | 6-conc | 1-min load at start |
|---|---|---|---|---|
| 0 (discard) | 206.5 | 185.6 | 271.0 | 2.57 |
| 1 | 232.1 | 199.8 | 266.8 | 5.04 |
| 2 | 232.6 | 190.0 | 285.3 | 5.33 |
| 3 | 232.8 | 205.1 | 255.3 | 4.09 |
| mean 1-3 | **232.5** | 198.3 | **269.1** | |

(The load average is inflated by the engine's own spinning threads.) M1 is 0.3% spread across three legs and
+3.1% over the 225.5 bank. Conc: 255-285 inside one boot, 11% spread. Conc history in `lanes.csv`: no point between
11-09 and 11-24; on 11-09 alone 237.3 / 276.1 / 290.1 / 291.0 / 307.6; F-285b's five-boot pooled conc 278.3
(sigma 12.3); SVG-CUSTOM:9345 "pair/conc single boots swing +-5-8% on this box independent of binary".

## What the leg file says about the two lanes (`stand-leg2.txt`)

- M1 is TWO native generations: `prompt 1176 ... 1489 generated in 6123 ms (243.2 tok/s)`, then
  `prompt 2682 tokens = 2664 reused + 18 read in 69 ms ... 494 generated in 2321 ms (212.8 tok/s)`. The continuation
  is the phase gate's hand-over (serve/server.py:5663: "a reasoning-budget continuation is another native
  generation"); it re-reads 18 tokens. Cost ~1.5% of the leg, paid by every bank. Not a defect.
- M1 request accounting: 12.2 ms/window, 2.67 tokens/window, hit 97.7%, accept 76.3%.
- **Conc is host-bound.** `rows 4.90 | window 13.74 ms | gpu-wait 0.40 | cpu-exp 2.88 | hit 95.9% | accept 65.3%
  | 279.8 rows/s` -> 3.574 ms per row, 17.5 ms per cycle, of which the host waits on the GPU 0.40 ms (solo: 8.96).
  In conc every host-millisecond saved is conc throughput.

## 03:15 — THE RECALIBRATED SOLO MAP: the window is LAUNCH-BOUND, not bandwidth-bound (stage bandwidth-floor audit)

`campaign/results/f453-2026-11-25/stage-bandwidth-audit.md` (157 lines; bytes from the pack manifest and the GGUF header
via the owning readers, times from the F-452 node trace, 293 warm windows).

- A warm window reads **4.36 GB of fixed weights** (GDN projections 1.557, hyper-connections 1.283, LM head 0.521, QSA
  projections 0.477, shared expert 0.146, router 0.126, PLE 0.066 GB) plus 1.2-2.0 GB of routed experts. A T=2 window
  moves 5.77-6.51 GB in 8.73 ms of wall: **660-746 GB/s average** (T=3: 629-738). The box's demonstrated rate for these
  matvec kernels is **1.64 TB/s** (the LM head: 521.47 MB in 317 us). The window runs at about 45% of it.
- Headroom at 1.3 TB/s per T=2 window, ms: QSA + projections 0.90, GDN + projections 0.88, shared expert 0.73,
  router/plan 0.58, hyper-connections 0.57. Total 3.66 of 8.73 ms. Routed experts and the LM head have none.
- **None of the three largest is HBM-bound**: GDN 796 GB/s, QSA 393, shared expert 177. The time is in too few blocks
  (`gdn_ab_multi_kernel` 48 blocks, `decode_attn_i8_split_kernel` 80, `resolve_kernel` 1; the shared expert's six
  serial launches per layer for 3 MB), chains of 0.7-5 us launches, and one stall: the GDN attention gate matvec starts
  a median **10.9 us after its input is ready** (2,932 T=2 instances).
- The hyper-connection contention hypothesis is REFUTED: 0 of 56,834 `gr_*_v3_kernel` instances overlap a side-stream
  kernel; the read runs on an idle GPU at 830 GB/s because it is 160-164 blocks on 170 SMs with a serial per-token dot.
  (The traced boot ran the v3 read; the standing boot now reports the staged HC read - different kernels: these HC
  rows describe the traced recipe only.)
- Correction to my figure: a routed expert is 2,046,400 B mean (1,510,400-2,662,400), not 1.4 MB.

**The solo lever this opens (law 6, structure before squeezing): launch shape and order, in bitwise pieces.**
GDN: launch the gate matvec right after its producer (0.11-0.17 ms per T=2 window; the cheapest falsifier is one read
of the capture order); give `gdn_ab_multi_kernel` its 10 loads in flight (up to 0.22 ms). Shared expert: one launch for
gate + up (the PAIR form exists), the gate scalar as a 513th row of the router GEMM, the sigmoid scale inside the down
matvec (up to 0.33 ms). All bitwise. Sum ~0.7 ms of 8.73 = **+8% M1 without touching an arithmetic**; with the QSA
split/fusion work (parity harness) 1.3-1.7 ms = **+15-20%**. Priced, not built: each is an engine change behind an env.

## 03:05 — THE FIRST SERVE-MODE ANATOMY OF CONC (nsys, graph level) - two populations, not a round boundary

Instruments (new, both reusable): `campaign/nsys-serve-anatomy.sh <label> [conc|m1]` boots the standing server inside an
nsys session with collection OFF, warms it by the battery protocol un-instrumented, collects a lane-only run, exports;
`campaign/nsys-serve-summary.py <sqlite>` reads it.  Evidence: `campaign/results/f453-anatomy/anatomy.{nsys-rep,sqlite}`,
reading in `campaign/results/f453-2026-11-25/anatomy-conc-summary.txt`.  Capture 26.26 s = two 6-conc runs (276.0 /
273.2 t/s under injection - a diagnostic, not a point).

Three things cost an attempt each and are written into the scripts: the session model (`launch` takes session and trace
options; `--output`, `--export`, `--sample` belong to `start`); `systemd-run --user --scope` cannot open its bus under
nsys's injected library, so the anatomy boots with STRATA_SWAP_GUARD=0 (placement unchanged on this box: the scope is
the swap cap only); GRAPH_TRACE keeps the session's pre-capture history on the same clock, so every GPU table is
windowed to the API span.  And nsys's CPU sampler and scheduler trace attach only to processes started AFTER
collection begins: zero samples and zero SCHED rows for an already-running engine.

**GPU busy 82.0%** of the capture (21,534 ms), idle 4,727 ms:
- gaps >= 3 ms: **255 gaps, 3,440 ms = 72.8% of all idle** (mean 13.5 ms) - one every 2.6 rounds;
- 1-3 ms: 334 gaps, 583 ms; < 1 ms: ~36,000 gaps, 703 ms.

**Steady rounds (573 of 671, keyed on cohort A's window graph): idle p50 0.74 ms per round = 5.0% of their time.**
The round: cohort B's window (`g317`, 20-22 ms) and cohort A's (`g230`, 19-22 ms) run CONCURRENTLY, A launched ~0.5 ms
after B; then a tail of two commit graphs (1.21 and 1.42 ms) and six drafter graphs (~0.5 ms each); then ~0.7-1.3 ms
idle before B's next launch.  Of the steady idle, 41% is the serve thread inside `cudaGraphLaunch` - the window
graph's host submission, **0.557 ms per launch (p50)**, of which only the FIRST cohort's is exposed (the second is
hidden under the first window).

**Long rounds (98, median period 36.4 ms): 2,937 ms of idle.**  Their signature: the GPU activity before and after the
stall is memcpy -> memcpy (1,342 ms) - stretches where the only GPU traffic is sporadic H2D copies of a few MB
(0.05-0.14 ms each) 0.3-3 ms apart; **93.5% of the stall time the serve thread is outside any CUDA call** (host code,
not a driver wait); the stall ends when the host issues the next `cudaMemcpyAsync`.

**Decisions.**
- Device-launch twin for the batch window graphs: priced +1.2% conc (0.30 s of 26 s) - NOT BUILT.  The audit's
  falsifier (< ~0.5 ms submission kills it) lands at 0.557 with only one of two launches exposed.
- NEW NUMBER: the two commit graphs cost 2.6 ms of GPU time per round, serialized after the windows (a tenth of the
  round).  Unexamined as a lever; recorded for the queue.
- The conc lever is the long round, and it is host-side: next instrument is `campaign/perf-serve-anatomy.sh`.

**Correction of my 02:30 arithmetic:** "the GPU is idle ~60% of the conc cycle" was wrong - it assumed twelve rows at
65% acceptance every round; the rounds average ~9 rows and the steady round idles 5%.

## 02:40 — RETRACTION of the 02:25 section: the batch-cache "thrash" was a misread log (law 8)

**The footgun.** `logs/custom-iq3_s.log` is APPENDED ACROSS BOOTS - it is not truncated at engine start. It holds 40
`engine started:` lines (2026-11-22 15:59:10 .. 2026-11-25 01:59:05). This boot begins at log line 551517. Every
number in the 02:25 section came from `tail -n +41` of the FIRST boot's banner line: whole-file totals over 40
processes, read as one boot. Every analysis of this log must start at the LAST `engine started:` line.

**What the live boot actually did** (`campaign/results/f453-2026-11-25/batch-graph-cache-audit.md`, part 0):
30 captures over 28 distinct layouts; 1,528 cohort rounds (1,466 of shape A.S=6 B.S=6); the 64-layout cap was
never reached; no VRAM-eviction line; no doorbell-variant flip. The cache is NOT thrashing.

**What one capture costs** (from the log's own `cohorts: windows done in X ms` lines; a capture round against the
mean of its capture-free neighbours of the same shape): median **+11.2 ms** (p25 9.3, p75 13.6, min 5.4, n = 88),
other shapes +8.6 (n = 246) / +9.4 (n = 46); two captures in a round +12.1 / +16.0. Confound: the capture round is
also the first round after the active set changed. Twelve captures per boot = about 0.12 s against 39.6 s of cohort
window time: <= 0.3%. An admission-time latency tax, not a throughput lever.

**Consequences, banked:** the canonical row order (`STRATA_COHORT_ROW_ORDER=slot`), the cap knob and a capture
telemetry build are all NOT RUN - each priced under 0.3% of conc. The idea stays on the shelf with its re-open
clause: it re-opens only if a workload shows the same live layout re-captured inside one engine process (a long
lived multi-tenant serve with rotating slots), which no probe on this record does.

**The ghost client.** The "probe-shaped traffic during the gate" (A.S=6 B.S=6 rounds, paired 2,000-token
generations, `gen 1` probes) was earlier boots' batteries in the same file. A 70 s sample of client-side sockets to
:8081 with process attribution found ONE client: the gate (pid 1010347). No timer, no cron, no second server, no
remote session. About 20 minutes were spent on it. The first `ss` filter also selected the wrong end of the socket
(`sport` = the server's side); `dport = :8081` is the client's.

## 02:25 — THE BATCH GRAPH CACHE THRASHES ON THE CONC LANE — WITHDRAWN, see 02:40 (the numbers below are whole-file totals over 40 boots, not one boot)

`logs/custom-iq3_s.log`, this boot only (4 probe legs with a 6-conc phase each):

- `strata verify: captured the batch window over slots <a,b,...>`: **1,254 captures** on the serve thread.
- **95 distinct layouts** (`/tmp/f453-batch-captures.txt`); the layout cache is LRU-capped at 64 with batch-mtp
  (verify.hpp:400, generate.cpp:9539/9620/9691/9713) -> the working set exceeds the cache.
- The same layouts re-captured dozens of times: `1,1,2,2,0,0` 55x, `1,1,2,2` 54x, `3,3` 51x, `1,1,0,0` 47x,
  `4,4,3,3` 41x, `5,5` 39x, `2,2,0,0` 39x, `4,4,5,5,3,3` 38x, `2,2` 38x.
- Rows per capture are only even counts: 2 rows 234, 4 rows 538, 6 rows 482 - every slot contributes exactly
  two rows (bonus + one draft), the window caps at 8, so six requests are served through subsets of 1-4 slots and
  which subset lands in a window depends on timing. The key is the per-row slot sequence; timing variety becomes
  layout variety.
- Each capture is capture + instantiate + upload + `cudaStreamSynchronize` on the serve path
  (verify.cpp:3716-3720). The lane is host-bound, so this time is the lane's. At ~313 captures per conc leg and
  X ms per capture the leg pays 0.31*X s of its 12.6 s: X = 3 -> 7%, X = 6 -> 15%, X = 10 -> 25%. X is NOT
  MEASURED: the serve-mode API trace is the next instrument.
- Also on this path: every batch window graph is host-launched through the helper thread
  (STRATA_VERIFY_ASYNC_LAUNCH=1, ~2,660 nodes per 6-row graph): no batch exec has a device twin (audit, part F).

## 02:20 — the drafter launch path (campaign/results/f453-2026-11-25/drafter-launch-path-audit.md)

Inline on the calling thread; no helper thread to remove. Round graph: 0.31 ms of occupancy around 0.10-0.14 ms
of kernels; step graphs 0.20-0.21 ms each, overlap nothing. No host DATA dependency between steps (the host holds
control only). Zero-code discriminator for the round graph's unexplained 0.17-0.22 ms: one boot with
`STRATA_COMMIT_SYNC=1 STRATA_COMMIT_TIMING=1 STRATA_DRAFT_TIMING=1` (the async commit graph shares the GPU).


`campaign/mmlu-pro-gate.py --suite N` (F-453): fixed stratified sample of the MMLU-Pro test split (12,032 q,
14 categories; proportional allocation by largest remainder; `random.Random(20261125)` consumed in
sorted-category order; asked in dataset order; ids written to the result). `--suite 1500 --list`: biology 89,
business 98, chemistry 141, computer science 51, economics 105, engineering 121, health 102, history 48, law 137,
math 169, other 115, philosophy 62, physics 162, psychology 100; first ids 74, 84, 92, 93, 103; last 12252.
Two draws compared: identical, 1500 unique. The 70-question path is the same measurement as before.
The 70q gate took 3 min 50 s on 11-24 (20:33:08 -> 20:36:57): 3.3 s per question.
