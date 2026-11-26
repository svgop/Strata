# Deep-research sweep, master index (2026-11-26) — six lanes, digests beside this file
Lanes: steering-sota, quant-stability, sampling-stability, kv-cache, mtp-specdec, engine-performance (each with per-claim URLs in the digest; provenance split production-proven / measured / paper-only inside each).

## Cross-lane conclusions that survive tonight's own batteries
1. THE MMLU SWING IS SERVING, NOT THE QUANT. The pack is public ISTA-DASLab GSQ-RCO, IQ3_S at BF16 parity per its card, with a .rco-allocation.txt audit file shipped. The 88.6-97.1 spread = "a different function per boot": quant-mix by residency (F-450 corroborated) + batch-size-dependent kernel reduction order (Thinking Machines: 1000 T0 completions -> 80 unique, 1 with batch-invariant kernels). Securing the ceiling = pin quant per tensor, pin eval numerics, then avg@k in-boot (vendor practice: R1, K2 avg@64) and the 1500-question suite for decisions (70q = 1.43 pts/question = canary only).
2. THE LOOP FIX IS THE VENDOR'S OWN PRESCRIPTION. "temp 1.0, top_k 20, presence 1.5" is the Qwen3.5 THINKING preset (2026); 0.6 was the old card; every 2025-26 vendor (K2-T, M1, V3.2) moved thinking UP to 1.0 and handles repetition with presence 0-2. Our 10/24 -> 0/24 matches. Second guard: DRY (structure-safe, llama.cpp default chain). Never blanket repeat_penalty; XTC is for fiction.
3. STRUCTURE STILL BEATS SQUEEZING (law 6 confirmed by 2024-26 practice): every large published win is a window, an overlap, a removed transfer, or removed host faults - not a faster kernel.
4. THE FRONTIER FOR MTP IS TRAINING-SIDE: FastMTP-class recursive head-only retrain (+12-25 t/s band, trainer live, windows.npy owed first); everything else in the walk is lossless tuning (draft-temp 0.4-0.65 sweep, block verification +3-6% "never worse", depth gate).
5. VECTORS: no accuracy-raising vector exists (5 families, 29 configs, all ~0) - the table's value is style/behavior; the accuracy-relevant items are mechanical: per-layer read-out gain compensation (4-stream sum makes constant-norm adds layer-varying), multi-direction refusal with independence tests, per-layer dose schedules, smooth phase ramps, and a possible third thinking-phase overthinking-control table (-15-30% thinking tokens at flat accuracy).

## Top actions (ranked; free falsifiers first)
A. [FREE, code read] GDN/DeltaNet recurrence precision: llama.cpp issue 29167 - fp16 recurrent state on Qwen3.8 hybrids produced "token soup" on >2K prompts; check ours is FP32.
B. [FREE, logging] Per-tensor effective-quant audit across 5 boots (residency-dependent quant = the standing suspect for the MMLU swing).
C. [1 boot] Presence 1.5 as the standing thinking default + the 1500q gate pair (loops 0/24, accuracy-neutral 64/64 at 70q).
D. [1 battery] Draft-temp sweep 0.85 -> {0.65, 0.5, 0.4}: +3-5 pts acceptance by collision-math, byte-stable output under the coupled contract.
E. [1 battery each, free premises first] cohort-ITL-during-prefill (Sarathi), depth-stair read (adaptive budget), pool-combine code read, THP boot (GLIBC_TUNABLES zero-code).
F. [days] FastMTP retrain cycle (windows.npy owed), block verification walk, block-granular prefix reuse, FP8 KV for QSA layers (shelf-variant labeling).
