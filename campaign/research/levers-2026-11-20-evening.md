# LEVERS 2026-11-20 evening — the far-higher-t/s investigation: the wall as one equation, the coverage correction, the priced queue

Method: read-only investigation from the F-411→F-418 recalibrated map + targeted code reads
(this session's sites cited). No boots, no builds. Priced per law 15 (the needle named before
the leg runs); falsifier-first per law 4; scoping per law 4b; capability envelope per law 16.

## 0. The wall (the map every lever prices against)

Solo M1 window ~13.6 ms at load 2-3: GPU-reach idle **8.79 ms at 3.10 rows** vs **0.51 ms at
4.88 rows** (same boot, F-414); CPU experts 2.85-3.27 ms; adapt wait 0.76 ms quiet / **1.19 ms
under load** (live anatomy line); emit 0.35-0.61. The kernels are at 95-98% of the measured
stream anchor (F-411) — they are not the lever. **The wall is one equation: host-stage
exposure × (1 − rows in flight).** Every high-value lever either adds rows (depth, acceptance),
shrinks a host stage (pool granularity, swap traffic), or moves a host stage under the GPU
window (lookahead). The lanes are the same wall at different row counts: the batch anatomy
(4.68 rows) still shows reach idle 1.33 + CPU experts 2.98 + adapt 1.19 — the conc lane is
gated by the same host stages.

## 1. Tier 1 — the +10-25% class (rows × host stages)

**A. Draft depth-2 re-test (armed, mechanism live).** STRATA_BATCH_DRAFT_DEPTH=2: round 1
+6.8% M1 (legs peaked 239.4), M3b canary QUIET (F-416). The ladder is named: quiet window,
five-boot means, STRATA_SPEC_DEPTH published on both arms so position-1 acceptance prices D=3
in the same battery, MMLU ×2 on adoption. The ceiling is the wall itself: rows 2.4→3.0+
converts the 8.79 ms idle. Cheapest big lever on the board — env only, no build.

**B. Lookahead overlap of CPU-miss experts (ceiling ~+21%).** RouterLookahead already runs
host-side per layer and its enable_stats/stats_score already score every prediction against
what the next layer really routes (recall bands at top-k, k+4, k+10). The falsifier is one
boot: STRATA_LOOKAHEAD_STATS=1 + probe leg. High recall → the third consumer (predicted-miss
jobs handed to ExpertPool as speculative work; epoch-claim expires wrong predictions naturally)
moves up to the whole 2.9 ms term under the GPU window. Low recall kills it free.

**C. Pool-granularity surgery (new pricing this session).** The mechanism, read at
src/kernels/cpu/pool.cpp:1040-1051: `run_split` = phase1 → **serial `act_quant_q8_1` barrier
on the service thread** (line 1050; every worker parked between phases) → phase2, ~48
per-layer calls/window, ~1 miss each: the two-phase machinery plus the barrier dominates the
GEMV — 65 µs/miss vs the ~26-30 µs DRAM floor (2.2-2.5×). Arm: fold the activation quant into
the phase-1 task's epilogue (the worker that wrote `ff` quantizes its own `a2`; the barrier
dissolves) or batch the quant loop across the pool. Halves the miss path → ~1.35 ms/window
exposed (F-414's waitCPU) → ~+10% class. **Composes with B**: C halves the term, B hides the
remainder.

**D. Context n-gram draft source (never attempted; T47 §3.7).** Prompt-lookup-style second
proposer spliced into the coupled chain; token-substitution form, lossless by construction.
The falsifier is zero-boot: a hash-lookup hit-rate probe over harvest/operator traces. +5-25%
on copy-heavy real traffic, 0% on open-ended probes — the scoping law names the arm-condition
(traffic class), not a standing trade.

## 2. Tier 2 — the coverage axis: a correction that re-prices F-418's fit

**The correction (boot-line + code evidence):** the resident cache slots are the model's own
pack form — `ExpertCache::slot_stride(lay.blob_bytes(l))` per layer (generate.cpp:6445,
expert_cache.cpp:571), IQ3_S, max blob 2.66 MB. The boot line: **9900 slots = 18.78 GiB =
1.94 MiB/slot average** (the prompt path's borrow line agrees: 5189 slots = 9.80 GiB).
F-418's fit arithmetic used the Q2_0 canonical blob (1.3824 MiB) and priced 16k slots at
22.1 GiB; the true fit is **16k × 1.94 ≈ 30.3 GiB** — the card is 32.6, the engine holds
31.1, free ~1.5. The geometry guard will refuse (the F-409 refusal class); the 16k falsifier
boot is still worth one boot to make the guard name the cap exactly.

**The re-price:** the fit on this card is ~11k slots (+2.1 GiB from reserve + slack) → offline
~92.0% (+1.6pp) → live ~+1% by F-412's translation. The 99% goal (~20.8k slots) needs ~40
GiB — off the envelope on 32 GB; the goal is either a different card or a different form.
The live operator-traffic lines already read 94.9-97.0% lookup hit — the 99% ask is a
coverage ask, and coverage on this card is capped by bytes/slot.

**The real coverage lever: cache form per expert.** The Q2_0 canonical form is 1.3824 MiB
(−29% bytes/slot): the same 18.78 GiB holds ~13.6k slots → offline ~94% (+3.6pp). The GPU
Q2_0 walk exists in the tree (q2_native_kernels); the miss path already computes Q2_0 —
the model's effective quality is already mixed (96% IQ3_S + 4% Q2_0). A pure Q2_0-form cache
moves hits onto Q2_0 — a quality-axis trade (law 16, MMLU-gate class). The bounded shape is
**frequency-tiered requant** (T47 §3.8, MxMoE/MoPEQ class): hot experts IQ3_S, cold Q2_0,
mixed-form cache — coverage per byte up, quality price bounded and measured. This is the
coverage lever that survives the card's ceiling; it is a design, not an arm.

## 3. Tier 3 — the drafter's quality (multiplies Tier 1)

- **M3b era-gap hunt:** 77-78% now vs the 87-92% record era — the slot-depth telemetry
  (F-415) is armed; the hunt is which leg of the draft chain pays (depth-0 acceptance
  62.5%; the MTP chain dp counters read 0/0 — the chain path does not run on the standing).
- **Retrain:** the deployed drafter is Oct 13 (dense.bin 116 MB); the trainer's checkpoint
  home and the harvest corpus are GONE after the estate move (`/home/svg/LLM/Custom-data`
  empty) — the retrain cycle starts with a re-harvest (the standing trainer is
  campaign/retrain/train_cycle.py). The drafter is ours: a deeper/better drafter shrinks
  waste and raises the depth-2/3 payoff (F-229's re-open clause).
- **spec-min-p 0.70:** upstream measured +4% on their engine (F-418); env-ladder arm for
  ours, M3b canary + MMLU ×2 (distribution-adjacent knob).

## 4. Tier 4 — the conc lane (325 → 450 aggregate)

- **C2 admission/tail overlap** (T47-conc-450's top named lever): ~40-50% of the round wall
  is NOT the verify window — finish tails, admission stops, adapt join. The field is
  unanimous that host work must hide under device work at our window lengths.
- **C4 second draft row = the depth-2 arm** (batch-lane corroboration: F-416 conc +2.5
  round 1); verify the kVerifyMaxT headroom at 9-wide before pricing.
- **swaps=80 (1/0.97/80):** same offline share, half the swap traffic; the target metric is
  the adapt wait under load (1.19 vs 0.76 quiet). config-ladder 3v3, cheap, F-418's curve
  already justifies it.
- **Host placement:** F-201's CCD split + IRQ steering + renice of ambient burn (naive
  whole-process taskset falsified, −20%); banks the record under load, quiet-box gain near
  zero — the load band is the operator's norm, so it is worth the leg.
- **Cohorts-3** needs batch≥9 + the b9 lane fix; elastic already delivers 9-wide with zero
  solo tax (F-273), so the capacity is ready — the window machinery is the gate.

## 5. Stability program

1. **F-271 class is structural in the ladders** (re-pin + md5 at DONE, F-417). The remaining
   exposure: the serving boot's own learned profile — the convergence signature is live in
   the log (exchanges 4164→11891 climbing under operator traffic). Boot-invariant candidate:
   an anatomy watchdog — exchange/file-read rate above 2× the live band → LOUD alert line
   (never auto-re-pin; the operator decides).
2. **NaN census root-cause** (the shadow arm under host-memory pressure; ActQ/run_split_multi
   audit) — open defect class with a live bite; canonical-pack-primary is the mitigation, not
   the cure.
3. **Elastic spare pool:** "0 chunks left in the spare pool" at admission → materialize pays
   the release path (6.3 ms) instead of the background promote (2.2-8 ms); price
   STRATA_ELASTIC_WATERMARK=2 — a TTFT lever, small but free.
4. **Estate data home:** Custom-data is empty post-move — re-harvest is a prerequisite for
   the retrain lever, and the workspace law wants the home rebuilt before the first artifact.
5. **Upstream watch:** ADAPT_FETCH re-prices if coverage/depth changes the miss economics
   (F-412's re-open clause); SLOT_READ_WIDE/AHEAD named for the multi-turn TTFT lane;
   POOL_DEP not ported (measured −2.9% window on a 5090 upstream).

## 6. The priced order (law 4/15: falsifier cost, then needle)

| # | Arm | Kind | Cost | Priced needle |
|---|---|---|---|---|
| 1 | Lookahead stats falsifier | boot | 1 boot | decides the +21% class (B) |
| 2 | Depth-2 quiet re-test (+stair, D=3 pricing) | ladder | 5 boots | rows 2.4→3.0: +10-25% (A) |
| 3 | swaps=80 + spec-min-p 0.7 | ladders | 2 ladders | adapt wait 1.19→~0.7; +1-4% |
| 4 | Pool-granularity arm (fold/parallel quant) | build | ~3 files | ~1.35 ms exposed → ~+10% (C) |
| 5 | n-gram draft source | falsifier | offline | +5-25% copy-heavy traffic (D) |
| 6 | 16k-slot falsifier boot | boot | 1 boot | names the coverage cap (expect REFUSE) |
| 7 | M3b era-gap hunt | telemetry | 1 boot | the acceptance leg (§3) |
| 8 | Re-harvest + retrain cycle | training | GPU-exclusive | drafter quality: multiplies 2/7 |
| 9 | C2 admission overlap | build | large | the conc round wall (450) |
| 10 | Mixed-form cache (tiered requant) | design | large | coverage/byte +29% (§2) |

1-3 are behind the card and already armed (the runbook + the two ladders). 4-7 are new or
re-priced this session. The composition A+B+C attacks the same wall from three sides and is
the credible path to solo 250-270 in the current envelope; 8 raises the ceiling on top;
9-10 are the lane/coverage plays beyond.
