# T50 GOAL-V5 — THE POST-RESEARCH QUEUE (six-lane dossier synthesis, 2026-10-25)

Sources: T50-r1 (specdec) / r2 (host) / r3 (MoE) / r4 (linear-attention) / r5 (serving) / r6 (kernels) + prior T50-new-levers.md. Method: six researchers, 3-6 papers + >=2 production sources each, every proposal costed against OUR measured window anatomy. This file is the execution queue; dossiers hold the derivations.

## THE CONVERGENCE (four independent lanes, one diagnosis)

1. **The window period is not GPU-bound.** R1's reconciliation: record 223 t/s x tau 4.8 = 21.52 ms = CPU(6) = 4.72*6-6.8 EXACTLY. The GPU walk (9.0-10.4 ms, flat in T) is fully hidden under CPU+replay. Every structural lever below converts the period toward the GPU walk; tau multiplies on top.
2. **The GDN commit replay is a 2024 anti-pattern.** 2.47 ms/row x 2-4 accepted rows = 5-10 ms/window re-scanning what verify already touched. 2026 SOTA replaces it wholesale: KVBuffer (arXiv:2605.19049, verify 2.78x, states never materialized), SpecLA (2607.16673, factor-recovery), and SGLang PRODUCTION (PyTorch blog 2025-12: per-draft-slot states, "promote the last accepted slot to become the new main state"). Our scoped-but-unstarted slot idea is now a shipped feature elsewhere.
3. **The graph shape carries ~70-75% structure overhead.** R6: critical path ~950-1,100 nodes at 9.5 us spacing carrying 2.3 us of work; a healthy captured graph moves node-to-node in ~1.3 us (Hazy Research measured). The window is not expensive; its SHAPE is. F-204's node inventory already prices the fusions (-900..-1,000 nodes available).
4. **The CPU tier is a scheduling term, not a bandwidth term.** R2: ~353 MB/window in 13.56 ms = 26 GB/s against 80-100 GB/s capacity; the pool re-forks 48x/window in a ring->pool->release chain (session.cpp:949-984). Every external hybrid system (APEX +84-96%, HybriMoE 1.70x decode, KTransformers production, vLLM V1 1.7x at small batch) converged on the same fix: whole-window sync, no per-layer barriers.

Production precedent table: per-slot states (SGLang), async scheduling (vLLM V1 default), device-side scheduling (Blink, up to 2.1x), glue-node deletion (TRT-LLM DeepSeek 5.5x attribution), chunked commit (KVBuffer/SpecLA), admission fencing (TRT-LLM CapacityScheduler), draft throttling (SmartSpec/TurboSpec, SGLang MTP blog's own tuning rule).

## THE LADDER (stacked; overlap accounted — L1/L3 share the CPU-period prize, priced once)

**L1 — COMMIT KILL (days, THE pillar).** Chunked-suffix commit of accepted rows (one WY chunk, n_keep<=4, ~0.3-0.8 ms total) + CUDA-graph capture of verify+commit (per-row 2.47 -> 0.3-0.5 ms class). Evidence: KVBuffer 2.78x verify; SGLang ships the slot form. Expected: window -5..-10 ms => solo 223 -> 300-450 band (+35-100%); floor +25% under the worst reading. Phase 2: per-slot states + pointer promotion (SGLang exact form), commit kernel deleted entirely.

**L2 — GRAPH SURGERY (days, R6 tranche 1).** F-204 fusions behind existing gates: LFUSE 6->3 (-144), fused wait+copy doorbell (-144), PLE postops snap (-56); tranche 2: quantize-into-producer (-180), router 4->2 (-96), hc read 3->2 (-96). Bitwise precedent per node class; falsifier = gate flip + bit-check + node-count assert (one afternoon). Expected: -2.3 ms (tranche 1) -> +15-25%; both tranches -> +35-45% on the window term.

**L3 — WINDOW-FUSED CPU TIER (1-2 wk).** Persistent polling workers on a per-window miss queue (kill the 48x fork/join); then device-plan lookahead + combine-last (the unbuilt CPU-TIER-OVERLAP remainder). Byte floor 3.5-4.5 ms vs 13.5 today; with L2's walk at ~7.4 ms, per-layer hiding becomes arithmetically possible (it was doomed at the old ratio). Lands the banked 276-class; stacked with L1+L2: solo 300-380 realistic.

**L4 — TAU (training + policy, parallel track).** (a) Drafter-v2: per-depth sequential conditioning (Hydra++/JetSpec lesson: conditioning, not replay, holds the tail) on the LIVE DAgger loop; (b) entropy-gated early stop (AdaEDL class, training-free, per-window — NOT the falsified static gate; offline AUROC falsifier); (c) ReaDy committed-state reads into the drafter (+8.5% acceptance on our model family); (d) SPEC_COST port rides here. tau 4.8 -> 5.5-6.0 => +15-25% multiplicative on whatever period L1-L3 leave.

**L5 — SERVING/STABILITY (1-2 wk).** (a) Acceptance-aware draft throttle T per slot (SmartSpec/TETRIS; T 6->3 under load = +12-19% verified-token efficiency; one-constant falsifier probe); (b) cohort-3 after b9 fix — ALONE only ~354, but x throttle = 402-464 (the 450 path); (c) admission fencing TRT-LLM-style (reserve-to-completion, pause-not-evict, warm-affinity) — the stability half of the mandate, 0.5-1 day; (d) green-context solo-lane isolation, gated on the 5090 power-wall smoke test.

**L6 — MoE DENSITY (1 wk, user-gated for quality).** Frequency-tail mixed-bit re-quant (IQ3_S hot / IQ2_XS-class tail ~2.7-2.9 avg bpw): slots 10,623 -> ~13.4-14k (+27-32%), every tier's coverage multiplied => +2.3-11% solo, +9-16% conc (histogram falsifier first); then ProMoE-class depth-1 prefetch for the last hit points; then pool retirement.

**L7 — MEGAKERNEL-LITE (2-6 wk, the endgame).** Per-GDN-layer kernel 39->~4 nodes; per-QSA-layer 52->~8 (counter-based readiness, chunked handoff; Hazy's 60 ns barriers / 500 ns instruction latency vs our 9.5 us edges). Window ~1,800 -> ~280 nodes => 2.3-2.7 ms windows => the 500+ t/s regime. Prerequisite: device-launch loop (tail-launch self-relaunching graph, WHILE conditional) whose half-day falsifier (extend host_graph_launch_test.cu) must run FIRST — it decides the loop architecture for L7 and kills the host from the window boundary (+10-20% alone, Blink shape).

## THE STACK MATH (honest)

Period today 21.5 ms (CPU-bound) x tau 4.8 = 223. After L1+L3: period -> GPU walk ~9.7 => ceiling ~495 at tau 4.8; realistic-with-losses solo 300-380 (+35-70%). L2 tranches take the walk to ~6.4-7.4 => ceiling 650-750; realistic 380-450. L4 tau 5.5-6.0 multiplies ~+15-25% on top: solo 430-550. L7 is the beyond-50% endgame (walk 2.3-2.7 => 800+ ceiling). Conc: L5 throttle+cohort-3 = 402-464, compounding with every window shrink. **+30-50% is the FLOOR of L1-L3, not the stretch.** Stability: fencing + T-controller + (power-gated) green contexts target ms/window p95 and TTFT p99 alongside.

## THIS WEEK'S FALSIFIER QUEUE (cheapest first; none touch the engine)

1. nsys the verify+commit region: >1 ms/row launch/sync gaps => L1-GO (hours) [r4]
2. Microbench existing chunked WY kernel at n_keep=4 + commit: <=1 ms & prefill-chunk tolerance => L1-GO (half day) [r4]
3. host_graph_launch_test.cu extension: device tail-launch vs host relaunch; spacing <0.7x => L7 loop architecture GO (half day) [r6/r2]
4. Flip L2 gates (LFUSE, fused-wait, PLE postops) + bit-check + assert nodes_ -344 (afternoon) [r6]
5. Instrumented boot: dm-wait-stats/pool decomposition; host-plan-dominated => L3-GO (one boot) [r2]
6. Offline tau replay from draft-prob logs: chain vs DP-tree vs 2-way fanout at fixed 6 nodes; <5% => kill L4-trees before any C++ [r1]
7. Entropy AUROC vs realized per-depth acceptance from existing logs; <0.65 => kill the entropy gate [r1]
8. Expert-activation histogram from learned profile; tail mass <1.5% beyond coverage => halve L6 [r3]
9. Conc probes at fixed T=4 / T=3, batch 6, cohorts 2: aggregate >= +8% => L5-throttle GO [r5]

## BANKED NEGATIVES / CLOSED LANES (this round, do not re-purchase)

- tcgen05/TMEM weight-stationary: sm_100-only, closed for sm_120 (Blackwell microbench arXiv:2512.02189; FA4/FlashInfer route 5090 to cuDNN).
- Fiddler-class CPU expert compute: re-dead externally at our hit rate.
- SP-MoE draft-coupled prefetch: downgraded to the cutoff-layer policy only (its 1.07-3.5x is a stalling-system number; our ceiling is N x 2.65 pts).
- PDL as a primary lever: +3% on TRT-LLM's healthy graph, neutral on ours, rejected as too coarse by Hazy; revisit only inside L7 kernels.
- cuDNN attention swap: cannot absorb our indexer/top-k/doorbell glue nodes.
- InfoCal: does not exist under that name (citation-hygiene note).
- Sub-1-bit experts (QMoE class): no place above ~2.5 bpw at our MMLU floor.
- Static depth gates remain dead; the entropy gate is per-window and must A/B against both no-gate and the static form.

## DOCTRINE NOTE (why this is not "playing safe")

Every ladder item is a STRUCTURAL change with a shipped production precedent (SGLang slots, vLLM async, Blink device-side, TRT-LLM glue deletion, KVBuffer commits) — we are not inventing risk, we are ending our isolation from the 2026 frontier. The campaign law still binds each leg: cheapest falsifier first, interleave verdicts, MMLU/M3b gates on anything token-touching, one-line rollbacks, F-entry + commit + push. The difference is the bar we aim at: window-period halving, not +4% increments.

---

# V5.1 ADDENDUM (2026-10-25) — EFFORT RE-PRICED FOR AI EXECUTION

The V5 effort column was human-hours. Executor is an AI agent with the campaign law as binding protocol. What changes: code reading, patching, ports, harnesses, and offline analysis are CHEAP (a "session" = one focused agent work block, ~30-90 min wall including compile). What does NOT shrink: (a) the measurement discipline itself — every adoption still owes a 3v3 interleave (6 boots x ~12 min = ~75 min wall, unattended), bit-check + MMLU gates (~15 min), (b) GPU serialization — one 5090, live server; training chunks, re-quant passes, and clean nsys captures need server-DOWN windows, (c) debug loops on risky kernels — cost scales with bench iterations, not lines written.

Units: **AS** = agent session (~30-90 min, includes its own builds/tests). **Leg** = boot+prewarm+3 probes (~12 min wall). **EXW** = GPU-exclusive window (server down; user-coordinated).

## The ladder, re-priced

| Item | Human (V5) | AI-realistic | Wall-clock floor | EXW? |
|---|---|---|---|---|
| Offline falsifier batch (tau replay, entropy AUROC, expert histogram, draft-log analysis) | days | **<1 AS total** | ~30 min (logs already on disk) | no |
| L2 tranche-1 gate flips + bit-check + node assert | days | **1 AS** | 1 leg (~12 min) | no |
| nsys commit-region capture + analysis | hours | **<1 AS** | ~20 min | prefer 1 EXW |
| L1a chunked-suffix commit (patch + parity + bench) | days | **1-2 AS** | 1-2 legs | no |
| L1b graph-capture of verify+commit | 1 day | **1 AS** | 1 leg | no |
| L1c per-slot states + pointer promotion (SGLang form) | M-H | **2-3 AS** (allocator + slot threading; the surface AI handles well) | 2 legs | no |
| SPEC_COST port (F-330) | half-day | **1 AS** (header+hook, self-contained; parity harness included) | 1 leg | no |
| L5a draft-throttle controller (server.py policy + T plumbing) | 1-2 days | **1 AS** | 2 legs | no |
| L5c admission fencing (server.py policy only) | 0.5-1 day | **<1 AS** | 1 leg | no |
| L5b cohort-3 / b9 lane fix | medium | **1-3 AS** (bug surface unknown = the variance) | 2 legs | no |
| L4b entropy gate (engine) | 1 day | **1 AS** (after the offline AUROC) | 1 leg | no |
| L4a drafter-v2 (per-depth conditioning) code side | 1-2 days | **1-2 AS** on the standing trainer | training chunks: 25-30 min each, checkpointed | YES |
| L4c ReaDy committed-state drafter reads | M-H | **2-3 AS** | 1 leg | no |
| L3 window-fused CPU tier (session.cpp rework + fences) | 1-2 wk | **3-5 AS** | 2-3 legs | no |
| L2 tranche-2 kernels (quant-into-producer, router 4->2, hc 3->2) | days | **1-2 AS** | 1-2 legs | no |
| L6 re-quant tooling (MxMoE-style allocator + IQ2 tail packer) | 1 wk | **2-3 AS** | compute pass ~1-2 h GPU + 2 gates | YES |
| L7 device-launch loop (tail-launch/WHILE graph) | 1-2 wk | **2-3 AS** after the existing half-day falsifier | 1-2 legs | no |
| L7 megakernel-lite, GDN layer class | 2-6 wk | **3-6 AS** (fusing pieces that exist bitwise-proven; bench loop per session) | parity + 2 legs | no |
| L7 megakernel-lite, QSA layer class | (in above) | **4-8 AS** (split-cell attn + indexer fusion = the hard surface) | parity + 2 legs | no |
| L5d green-context lane isolation | M-L | **1-2 AS** + power-wall smoke first | smoke ~10 min | no |

## Re-ranked execution plan (falsifier-ordered, AI-priced)

The V5 ordering survives (it was falsifier-gated, not effort-gated), but two promotions: **L1c (per-slot states) and L7 (megakernel-lite) move from "later campaigns" into the main plan** — their human-effort price was the only thing keeping them out.

- **Phase 0 — no server impact, <1 AS:** the entire offline falsifier batch. Decides L4-trees, L4b, halves-or-not L6, re-prices L1.
- **Phase 1 — next quiet boot window, ~3-4 AS total:** L2 tranche-1 (gate flips) + nsys commit capture + L1a+L1b (one build, separate env gates, armed one per leg per the one-variable law). SPEC_COST rides the same build behind its own gate.
- **Phase 2 — ~2-3 AS:** L5a throttle + L5c fencing (server-side only, no engine build). L5b cohort-3 if the b9 fix reads clean.
- **Phase 3 — EXW-coordinated with the user:** L4a training chunks (the DAgger loop is already live; 25-30 min chunks) and the L6 re-quant compute pass. These are the only items that NEED the server down for real stretches.
- **Phase 4 — the structural stack, ~8-15 AS across sessions:** L1c per-slot states, L3 CPU tier, L2 tranche-2, L4b/c, L7 device-launch loop then GDN megakernel then QSA megakernel.

**Bottom line:** the full V5 ladder is ~25-40 agent sessions of engineering plus its measurement legs — call it 2-4 calendar weeks of part-time agent work with the GPU mostly serving, instead of a multi-month human project. The binding constraints are (1) EXW windows for training/re-quant (user-coordinated), (2) interleave-battery wall-clock (~75 min per adoption verdict, unattended), (3) megakernel debug-loop variance — the only item whose session count I cannot bound tightly in advance, which is exactly why its half-day falsifier runs first.

---

# V5.2 ADDENDUM (2026-10-25) — THE PICK ORDER: reward-per-session ranking

Ratio = central expected gain / (agent sessions + 0.2 per measurement leg), risk-adjusted. Dependencies noted where they gate the payoff (the period is CPU-bound today: walk-shrinkers pay fully only after L1 un-hides the GPU walk — priced in).

| # | Item | Cost | Expected (central) | Ratio note |
|---|---|---|---|---|
| 0 | Offline falsifier batch | <1 AS, 0 legs | re-prices 4 items, kills dead ones | free information; always first |
| 1 | **L1a+L1b commit kill** (chunked suffix + graph capture) | 2-3 AS, 1-2 legs | **+25-50% solo** | ~15-20%/AS AND the biggest absolute; unlocks L2/L3 payoffs |
| 2 | **L5a draft throttle** (server.py policy, no engine build) | ~1.5 AS-equiv, 2 legs | **+14% conc**, solo held | zero build risk, no parity surface; the cheapest real % on the board |
| 3 | **L2-t1 gate flips** (LFUSE, fused-wait, PLE postops) | 1 AS, 1 leg | +8-12% (full value post-L1) | gates already exist; bit-check does the verifying |
| 4 | **L5c admission fencing** (server.py policy) | <1 AS, 1 leg | +2-4% + the p95/p99 stability win | cheapest item in the whole plan |
| 5 | **SPEC_COST port** (rides any build, own env gate) | 1 AS, 1 leg | +1-3% | self-contained; amortizes an existing build session |
| 6 | **L4b entropy gate** (after AUROC passes) | 1 AS, 1 leg | +5-10% solo | offline AUROC may kill it free first |
| 7 | **L1c per-slot states** (SGLang form) | 2-3 AS, 2 legs | +10-30% beyond/instead of L1a | second bite of the same prize |
| 8 | **L6 re-quant tail** (histogram-gated) | 2-3 AS + EXW 1-2 h + gates | +7% solo, +10-16% conc | needs your quality gate + a server-down window |
| 9 | **L5b cohort-3 / b9 fix** | 1-3 AS, 2 legs | +4.6% alone; x L5a = the 450 path | variance is the bug surface, not the code |
| 10 | **L4a drafter-v2** (code) + training chunks | 1-2 AS + EXW chunks | +15-25% via tau | the DAgger loop is live; chunks are 25-30 min each |
| 11 | **L3 window-fused CPU tier** | 3-5 AS, 2-3 legs | +10-15% standalone; completes the 276-class | biggest non-megakernel surface left |
| 12 | **L2-t2 kernels** (quant-into-producer, router, hc) | 1-2 AS, 1-2 legs | +10-20% of the walk, post-L1 | cheap, sequenced after the walk is exposed |
| 13 | **L7 device-launch loop** (after its falsifier) | 2-3 AS, 1-2 legs | +10-20% | also the architecture decision for 14-15 |
| 14 | **ReaDy committed-state drafter reads** | 2-3 AS, 1 leg | +8% (acceptance) | clean mechanism, moderate surface |
| 15 | **L7 GDN megakernel** | 3-6 AS, parity + 2 legs | window 3.5-4.5x class | huge reward, unbounded debug variance — falsifier-gated |
| 16 | **L7 QSA megakernel** | 4-8 AS, parity + 2 legs | same class, harder surface | after 15 proves the pattern |
| 17 | L5d green contexts (power-wall smoke first) | 1-2 AS | stability tails, ~0 aggregate | stability specialist only |
| 18 | L6-P2 prefetch / P3 pool retirement (riders) | <1 AS each | +1-4% each, post-L6-P1 | endgame polish |

Execution order = 0,1 first batch; 2,3,4,5 ride the same quiet windows; 6-10 as gates pass; 11+ the structural stack. Items 1-7 (~8-11 AS total) carry a realistic **+40-70% combined** — the goal's upper band — before any megakernel work.

## V5.2 note — "why not a faster language than Python?" (priced, not adopted)

serve/server.py (8.3k lines) is a request/API/tokenize/streaming front end over stdin/stdout; the C++/CUDA engine owns every per-window cost (decode, cohorts, tiers, graphs, pools). Measured Python-adjacent surface = boundary/emit tails 1.1-2.2 ms/window (engine printf + Python readline, shared with engine-side D2D/commit work) — already priced hideable by R2-P5 (+4-8%) and removable by L7's device-resident loop. Decode throughput is proven through the full Python stack (338.5 t/s @ 6-conc; engine's own window accounting matches the same period — no inter-layer tax visible). A Rust/C++ rewrite (~10-20 AS, API/MCP/streaming regression risk) buys low-single-digit % at best = does not make the pick order. Post-L7 revisit clause: when windows hit ~2.5 ms, move the emit protocol off stdout onto mapped-memory rings (engine-side work, still not a language swap); tokenize-in-engine only if TTFT profiling indicts strata_tokenizer on huge prompts (measure first: 88k-token prompt's prefill is 12.5 s engine-side — tokenization is noise next to it today).

---

# V5.3 ADDENDUM (2026-10-25 evening) — the measured reality + two new lanes + re-rank

**Session data folded in:** F-331 (int8-MMA pair adopted, pair +6.5%), F-332 (adapt-early adopted, adapt wait 1.03→0.3-0.6 ms/window), F-333 (pin kept; EMIT_LATE_FLUSH honestly rejected inside band), F-334 (cleaned-box 10-leg bank: M1 219.2 mean/232.5 best, conc 305.9/327.2 — at the old record band with the new arms aboard), F-335 (bg-load stable baseline: 216.2/203.1/304.4/77.1 with user load ramping 0.95→6.0). Standing recipe = F-331+F-332+pin.

**The binding constraint is now MEASUREMENT, not ideas:** identical recipe across boots spans M1 means 189.8–219.2 (~15%); R9's power arithmetic: at σw=5% a +3% arm needs 44 pairs — unmeasurable; with clock/power pinning (σw→3%) it needs 16 pairs at 80% power, one boot, interleaved ABBA with paired-by-adjacency CI. **The measurement law v2 (T51-r9-method.md) is therefore inserted at the TOP of the pick order** — it unblocks every rider and every future small arm.

## Updated pick order (v5.3)

| # | Item | Why here now |
|---|---|---|
| 0 | **Measurement law v2** — sudo pinning (persistenced, -pl 575, -lgc 2700, CPU perf governor + X3D-CCD pin), interleaved-paired protocol, per-leg telemetry + reject gates | gates everything; a calibration form + known-positive reproduction before it becomes law |
| 0b | **Latency P-A** (short-read token budget + decode-first packing) | smallest effort in the whole plan; ITL p99 19.8→~12-14 ms |
| 1 | L1 commit kill (unchanged — the biggest single lever) | falsifiers are half a day, no engine change |
| 2 | L5a draft throttle (conc) + L5c admission fencing (stability) | server.py policy; the fencing is also latency P-C's first half |
| 3 | L2-t1 gate flips; L6 re-quant (histogram-gated, user-gated quality) | one-afternoon + EXW compute |
| 4 | **Latency P-B/P-D** (interleaved bounded prefill; per-cohort adaptive T) | the TTFT p99 <1.2s + ITL p99 <10ms path |
| 5 | **Cold/warm P1** (page-cache pin, PSI-gated) + **P2** (warm-artifact persistence) | makes the 219-234 band durable; P2's falsifier first (boot-phase instrumentation) |
| 6 | L3 CPU-tier fusion; L2-t2; SPEC_COST; cohort-3 | the structural stack |
| 7 | Cold/warm P3 (cross-restart parking); L7 device-launch + megakernels | endgame items |

New goal rows added to the mission: **TTFT p99 <1.2s, ITL p99 <10ms** (T51-r8), **restart-to-first-quality-token halved + warm band durable under pressure** (T51-r7). Throughput targets unchanged (chase resumes from 219/306 standing).

---

# V5.4 DOCTRINE (user order, 2026-10-26): SHARED-COMPUTE IS THE STANDARD TEST CONDITION

All baselines and A/B forms from now on run WITH the user's background load live. Comparisons are
fair only within the same shared-compute condition; the cleaned-box banks (F-334/F-339) become a
SEPARATE ledger re-visited only at a later quiet-device stage. The shared-compute anchor mean (today's
battery) is the number to beat; the quiet bests (232.5 solo / 327.2 conc) are the GOAL UNDER LOAD —
beating them shared-compute implies the quiet-device best lifts proportionally (user projection:
quiet solo 300+, quiet conc 400+; to be measured once, later, as confirmation — not as a gate).
