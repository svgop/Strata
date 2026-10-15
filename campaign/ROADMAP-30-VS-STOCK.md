# ROADMAP: +30% vs upstream stock (the night-and-day target)

**Baseline (F-198, window 2, same-hour):** stock v0.1.44 = M1 147.1 / pair 149.1 / 6-conc 187.7.
(F-208: upstream tag v0.1.44.1 = pure hotfix, answers byte-identical, no CUDA-decode delta —
baseline STANDS, no re-baselining.)

**Warm-class standing (F-203..F-207):** M1 179.9-187.8 / pair 171.0-175.0 / 6-conc 210.1-219.1
(**+22-27 / +16 / +13%**).

**The target in absolute terms (stock-row × 1.30):** M1 191.2 · pair 193.8 · 6-conc 244.0.
**Still needed ON TOP of warm-class:** M1 +2-6% · pair +11-13% · 6-conc +11-16%.

---

## 0. F-208 RECON RECALIBRATION — the master queue (supersedes ordering in §2-§4; four briefs in research/F-208-*)

Four recon agents (campaign audit, engine hostile review, external SOTA, measurement lanes) +
main-loop review, 2026-10-15. Full findings: `research/F-208-{campaign-audit,engine-overhaul,
external-sota,measurement-lanes}-agent.md`.

**LANE A — window-budget code rocks (conc/M1/pair), EV order:**
1. **T1.1 partial device plan** (4.90 ms tail; plan in research/T1.1-partial-plan-impl-plan.md;
   publish-side pre-verified F-208: `doorbell_publish_res` already miss-aware, host already
   res-aware — GPU-side plan/wait/dst edits only).
2. **T1.2a PR#540 folded attention kernel ON sm_120** — the 5090 runs the OLDEST variant
   (STRATA_ATTN_PRE75_BUILT excludes sm≥75, qsa_decode_attn.cu:291; runtime gate cc<75 :609;
   kernel documented bit-for-bit). Port path: upstream PR #2075 does the decode+drafter hookup
   with a bit-exact merge (cherry-pick first); fallback = -DSTRATA_EXPERIMENTAL_SM60=ON + one-line
   env-force. Est −1.5-2.5 ms of the 4.40 ms line. THEN T1.2b row-fold (R19c) on top; Fireworks
   MiniMax-M3 kernel (KV-outer scheduling, portable to sm_120) is the reference.
3. **T1.4 hc wide-grid reshape** — 1-warp-per-row, 324 blocks, lane chunk order preserved =
   bitwise identical (overhaul A3); CHECK PR#646/#783's `hc_write_multi` single-warp kernel
   first (never merged, our tree lacks it). Est −1.5-3.0 ms of the 4.2 ms line.
4. **A2 plan-thread decoupling** — the half of the doorbell tail T1.1 cannot remove (single
   service thread serializes layer l+1's flags behind layer l's drain, verify.cpp:3080-3130).
   Est −0.5-1.5 ms. This is the one-week-rewrite pick; do it incrementally.
5. **A6 batch group-split G=2** (`!batch_rec_` forces G=1, verify.cpp:882 — the overlap machinery
   is compiled out for the 6-conc path). Est −1.0-2.5 ms tail-hiding. Audit stage_batch rows[]
   + per-group stateful kernels first (overhaul premortem #3).
6. **Doorbell posted-write inversion — REPRICED after T1.1** (the 1.8-3.6 ms estimate was
   pre-T1.1 graph arithmetic; poll population shrinks). Includes A4 (q8_1 activation publish,
   4× copy cut) as its publish-side half.
7. **R28 GDN one-pass fused identity + log-space decay** (execution-validated exact, 1.5-2×;
   ~3.6 ms of GDN lines) — CHECK #646's `gdn_conv_multi` first.
8. Small class, bundle after the big lines: A8 draft-tail event (mtp.cpp:1826 sync bound),
   A9 FUSE_HITS A/B, T1.8 exact two-phase head (0.4-0.7 ms, no-MMLU-risk byte cut), #2514
   top-k dispatch audit (decode may land on the reference select kernel, 1.7-2.4× slower).

**LANE B — acceptance/depth (the pair hedge):**
1. Stair DIAGNOSTIC boot first (STRATA_SPEC_DEPTH=1 → per-depth histogram; depths 4-6 were
   never OFFERED — chain starves at floor 0.5, generate.cpp:12796-12798; spec-8 re-test is
   meaningless without this).
2. T1.3a stair adoption (offer-side only) → spec-6/8 WITH stair.
3. T1.3b deeper slot chains (pair +4-8%, F-187 pricing).
4. Block verification (lossless drop-in, +3-8% class).
5. DT 0.80/0.90 re-sweep (2 boots).

**LANE C — measurement catch-up (~2-2.5 h battery, makes all 7 lanes scorable; spec in
research/F-208-measurement-lanes-agent.md):** prefill row (client `timings.prompt_per_second`
= fresh-only rate — NO new engine surface), first stock acceptance+mimic rows (stock DOES print
acceptance — F-198 note 3 was a probe --log misconfig), M5 mimic port (4 diffs; tool found at
/mnt/DATA/LLM/Engine/llama-custom/bench/measure/mimicbench.py, stdlib-only), owed RR-0.75
adherence check + one MMLU gate on the true stack, stock MMLU row.

**LANE D — env re-arms on the FIXED wrapper (zero code; F-200 non-adopted arms kept stale
verdicts by omission — void-class):** STRATA_ATTN_LANECELL=1 FIRST (read +3.6/+5.4/+1.9 on a
BARE base — full-stack effect plausibly larger; note #2268 reports it neutral — re-measure),
then GR_V3 (MMLU-gated), L2 72, HC_SPLIT=1, MMQ_KQ-on-prefill, DOORBELL_STORE (only after
re-deriving per-wait-site value semantics per F-203's own order), #2326 PIPED_CONCURRENT
cherry-pick (cold-prompt reads beside decoding; 11.9s→0.8s pause class).

**Stacked arithmetic (overhaul §C):** 4.40→~1.5 (T1.2a+b) · 4.2→~1.5 (T1.4) · 4.90→~0.5
(T1.1+A2+A4+A6) · emit ~1.0 (R33+A8) ⇒ window ~31.4→21-24 ms ⇒ 6-conc clears the 244 bar
(needs ≤24.2 ms) with margin; pair via Lane B hedge; M1 is nearly closed already.

**Hygiene (F-208):** phase-table re-measure post-R33/F-196 BEFORE committing Lane A order
(all estimates priced against the pre-R33 mix); 49 commits unpushed on one SSD (power-cut
precedent) — bundle/remote backup needed; §4 spacing-class items are POST-F-206 DEVALUED
(node savings ≠ wall time) except where serialized handoff time is the claim (A9);
#2268 warns "bitwise" fused-stack claims drifted 2/12 greedy probes — MMLU-gate any stack
adoption.

---

## 1. Where the time actually goes (measured, F-192/F-194)

The 6-concurrency decode window is **31.41 ms** (525-window live mean, 6.64 rows):

| stage | ms | share | what it is |
|---|---|---|---|
| GPU-reach wait | 21.68 | 69% | pure graph execution — the host spins while ~1,800 nodes run at ~9.5 µs spacing (bytes need only ~2.3 µs/node) |
| — QSA attention | 4.40 | #1 line | latency-starved: 64 serial dependent pool reads/thread, NO row sharing across query rows (width growth 1.6→4.4 ms is pure re-gather), 12 warp-shuffles/cell CUDA tax |
| — hc reads | 4.20 | #2 line | parallelism-starved: geometry gives the down-projection only 41 blocks (157 MB/window q8); Little's-law ceiling ~80 GB/s of 1,792 |
| — experts+rest | ~13 | — | MoE grouped reads + dense projections + head (521 MB/window) |
| CPU-expert doorbell | 4.90 | 16% | host runs the 2% CPU-routed experts while the GPU stalls at those layers — all-or-nothing fallback (verify_kernels.cu:1141-1155) |
| emit | 1.87 | 6% | per-slot MTP draft launches + bookkeeping on host (generate.cpp ~10300-10326) |
| commit | 1.22 | 4% | **RECOVERED by F-196** (parity double buffer) |
| adapt | 0.18 | — | healthy |

Round-9's law: the engine is **execution-bound, not bandwidth-bound** (24-26% of GDDR7 peak) —
the honest ceiling without byte cuts is ~2.9×. A 30% cut is well inside it.

**The recoverable pool in one 6-conc window: doorbell 4.9 + attention ~2.2 + hc ~1.8 + emit ~1 ≈
10 ms of 31.4 (32%)** — more than the 6.6 ms the conc target needs; the tiers below harvest it in
risk order. M1's 17 ms single window has the same attention/hc lines plus the draft/acceptance
class (rescue already banked at +20.6 pts accept / +8.2% M1; chain math: banked ≈ (1−a⁷)/(1−a)).

---

## 2. TIER 0 — the env battery (one boot each, ABAB, zero code)

All exist in-tree now; F-195's revised order. Each: fresh boot → 3+3 probes, means, Lock Law.

| # | arm | attacks | expected | gate |
|---|---|---|---|---|
| T0.1 | `STRATA_TYPICAL_DEVICE=1` | rescue moved device-side (parity-proven) | +2-5% sampled M1 | none (parity-proven) |
| T0.2 | `STRATA_VERIFY_ASYNC_LAUNCH=1` | the 69% GPU-reach line — if the launch CALL is the wait, async launch attacks it directly (v0.1.44 HostGraphLaunch) | unknown, possibly large | none |
| T0.3 | `STRATA_ATTN_LANECELL=1` | kills the 12 warp-shuffles/cell in attention | claimed bit-identical; move the 4.40 ms line | phase table self-check |
| T0.4 | `STRATA_HC_SPLIT=0` then `=1` | the 41-block starvation (down = 41 blocks) | move the 4.2 ms line | none |
| T0.5 | `STRATA_GR_V3=1` | split-K hc read | ~1-2% | MMLU (sum-order change) |
| T0.6 | `ROUTE_RESIDENT 0.75` | the CPU-expert tail (2% routed) | ~1-3% | MMLU (routing change) |
| T0.7 | `L2_SETASIDE 72` | GDN state in L2 | ~1% single-stream | none |
| T0.8 | `STRATA_MMQ_KQ=1` | prefill K-quant kernels (prefill metric) | upstream ×1.3-1.7 claim on prefill | none |

A knob that moves its phase-table line but not wall time tells us the line was hidden — equally
valuable. Cheapest first, always same-window toggles (the F-189 law: no comparison across 20+ min).

---

## 3. TIER 1 — surgical code (the ceiling-breakers, measured-value order)

### T1.1 — Partial device plan (THE #1 code item: the 4.90 ms doorbell tail, 9-16% of window)
Mechanism: when a layer-group has ANY missing expert, the resident plan aborts all-or-nothing
(`s_bad`, verify_kernels.cu:1141-1155) and the whole group takes the host round-trip — for a 2%
CPU-routed tail. The per-region machinery ALREADY EXISTS (counts[2]=0 / start2[0]=n layout,
pre-doorbell miss publish verify.cpp:1519-1526; the E-6 plan covers resident regions only).
Rewrite: publish a partial plan per region; only genuinely-missing regions wait on the host.
Exact arithmetic (routing math unchanged) → no MMLU risk class. One quiet-window reboot to land.

### T1.2 — R19c: fold the query rows into the attention block (the 6-conc ceiling)
The window's rows re-gather the SAME context KV independently — width growth 1.6→4.4 ms is pure
re-gather. One block loading the KV cell ONCE for all rows of the slot group: 6 rows × 12 KB sq
fits 228 KB shared. Halves the #1 GPU line (~2.2 ms/window at width 6, more at 8-12).
This is also what makes window width 8→12 (T2.5) affordable later.

### T1.3 — Deeper slot chains + the floor stair (the pair/M1 acceptance class)
Pair: slots verify ONE proposal today (`set_max_drafts(1)` generate.cpp:4736, `draft(2,…)`);
chain to 3 with the matched-prefix accept loop (~:10223). At 63-68% slot acceptance, a second
proposal banks ~0.6 extra rows/window nearly free.
M1: the serial chain stops at depth 3 under the flat min-p 0.5 floor while depth acceptance runs
96/93/90% (F-159) — stair 0.5/0.5/0.30 is offer-side only (serial verify = true rejection
sampling → distribution-neutral). Banked ≈ (1−a⁷)/(1−a): a=0.80→0.87 is +21% banked rows.

### T1.4 — hc read parallelism (the 4.2 ms line)
After T0.4/T0.5 pick the best block layout, the code form: fuse the hc down/up read into one
pass over the row group (QFUSE done properly — the F-187 rejection was the naive form), or split-K
across the 41-block starvation. Target ~1.5-2 ms.

### T1.5 — Emit on the device (1.87 ms)
Per-slot draft launches are host-serialized. A device-driven draft chain (the drafter's round
graphs launched inside/behind the window graph, gated by the commit event) removes the host from
the round trip. Composes with T1.3.

### T1.6 — Fenced admission (the churn hitches, 54.8-58.3 ms waves)
~14-30 cudaDeviceSynchronize per 6-slot admit wave (copy_to_slot/from_slot,
generate.cpp:10049-10140). Stream-ordered events + one batched save/restore → ~15 ms. Not in the
steady-state window, but it is the pair/conc jitter under real traffic.

### T1.7 — Graph pre-capture / sentinel slots (the layout churn)
176 re-captures per 5 boots from BATCH_MTP slot rotation; interim = pre-capture canonical
rotation subsets inside copy_to_slot; full fix = sentinel-slot bucketing (fixed slot count,
zero-length placeholders, ONE graph shape). Also the enabler for cheap window width.

### T1.8 — Exact two-phase head (521 MB/window lm-head read)
4-bit shortlist + full-scan only on disagreement (EXACT, unlike tqllm's shipped form).
~1-2% of window; small prize, cheap code.

---

## 4. TIER 2 — formulation/pathing rewrites

- **T2.1 Layered prefill** — layer-group interleaving across chunks kills the per-chunk
  expert-reload tax; up to 70% TTFT claims in the literature; prefill is priority metric #1.
- **T2.2 ECHO per-row verify budget** — stop drafting rows whose recent acceptance is cold;
  frees window rows for hot slots (concurrent metric).
- **T2.3 DuetServe look-ahead launch + launch-completion events + QKV-multistream-in-graph** —
  the 9.5 µs node-spacing is launch-path latency; overlapping the next window's launch hides it.
  (PDL alone measured no-op F-191 — this is the fuller form.)
- **T2.4 Conditional graph nodes** — mapped spin gates + E-6 plans already scaffold it; NO
  production engine ships conditional nodes today — genuine green field, kills doorbell waits
  inside the graph.
- **T2.5 Window width 8→12** — 1.9 GiB of 21.7 free; only after T1.2 makes width cheap.

## 5. TIER 3 — rebuild/training class

- **TQ3 rotation quants** (~11% more experts/GiB — softens every cache trade; rebuild).
- **~3-bit/trellis KV** (deep-context lane; 12→~4.5 KB/token).
- **DFlash2 port from tqllm** (gated on a sampled-mode tree-acceptance measurement FIRST —
  greedy-only claims don't survive temp 0.7).
- **FastMTP + KV-injection** (training tier, upstream pos-2 0.476→0.616 for our family).

## 6. Cross-cutting (do alongside, not instead)

- **RUN-2 STALL hunt** (one ~97-100 t/s collapsed run per boot; worth several % of every battery
  mean; suspects: expert-profile-save fsync, adapt refill, clock transitions; newest data point:
  it hit the OLDEST engine's FIRST run — F-197).
- **MMLU gates** where marked; single-stream protection law (concurrency-for-M1 trades are shelf
  variants, F-183 precedent).
- **User-side hardware blocks still open**: iGPU cable move (frees ~0.3-0.5 GB VRAM + removes
  compositor contention — top hardware task per F-193), GPU lock state check (F-190 verdict:
  locked-2835 mildly beneficial).

---

## 7. The stacked arithmetic (honest, with overlap)

| metric | banked vs stock | T0 battery | T1 stack | realistic landing |
|---|---|---|---|---|
| M1 | +12.0% | +3-7% (T0.1/T0.3/T0.7) | +8-15% (T1.3 stair, T1.1, T1.2/T1.4 window cuts) | **+25-35%** ✅ |
| pair | +7.6% | +2-4% | +12-20% (T1.3 chains, T1.6, T1.1, T1.2) | **+22-32%** ✅ |
| 6-conc | +2.7% | +2-5% | +15-25% (T1.1 4.9ms, T1.2 2.2ms, T1.4 1.8ms, T1.5 1ms, T1.7) | **+20-33%** (the stretch goal) |

Overlaps are real (cutting one stage shrinks the window the others are measured against); the
recoverable-pool arithmetic in §1 says the pool is deep enough IF the Tier-1 items land. Order of
execution: T0 battery first (days), then T1.1 → T1.2 → T1.3 (the three that carry the target),
each under the Lock Law: fresh-boot ABAB, 3+3 same-hour probes, means, MMLU where marked.
