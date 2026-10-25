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
