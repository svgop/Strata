# T50-r5 Serving Dossier - Concurrency, Scheduling, Stability (single RTX 5090, cohort-overlapped windows)

Researcher R5, 2026-10-25. Scope: arxiv systems work (2024-2026) + production engine docs/blogs on LLM serving concurrency, scheduling, and stability, mapped onto our measured box (338.5 t/s @ 6-conc, goal 450; solo 223/234; pair ~222; cohort-2 jump 245.9->306.8; ms/window drift 14->10; TTFT p50 0.3 s / p95 0.9 s; pool theorem: CPU pool waits only hidden by another window in flight).

## 1. Per-source findings

### 1.1 Sarathi-Serve (OSDI'24) - chunked prefill + stall-free batching - GRADE A
arXiv:2403.02310; usenix.org/conference/osdi24/presentation/agrawal. (a) chunked prefills; (b) stall-free batching - mix prefill chunks with decodes each step; (c) from Sarathi (2308.16369): decode-maximal batching (piggyback decodes onto prefill chunks). 2.6-7.75x goodput at matched SLOs vs FT/Orca-class; TBT near-constant under prefill load. Relevance: in-window TTFT stability - a prefill chunk co-scheduled inside a decode window must be width-bounded so the decode cohort's ms/window does not inflate; their measured TBT inflation from mixed steps is exactly our ms/window drift channel.

### 1.2 SmartSpec / TurboSpec (arXiv:2406.14066) - goodput-driven draft-length control - GRADE A (full text v1)
Per-step speculation length k maximizes predicted goodput = tokens/step; accepted-length model l(a,k) = (1-a^(k+1))/(1-a); linear forward-time model with offline-profiled coefficients; moving-average acceptance a per request. Key finding (directly about us): small batches want k>4; as batch grows, rejected proposals waste compute; at very high load goodput is maximized by **disabling speculation entirely** - above request-rate 12, k=5 gave no benefit; above 16, k=3 degraded. Never worse than no-spec (dials k->0 under load). 1.2-3.2x latency reduction at low rates, no degradation at high. Our effective token-batch is 6 slots x 7 positions x 2 cohorts ~ 84/step - squarely in their "stops paying" regime unless a is very high.

### 1.3 TETRIS (ACL'25, arXiv:2502.15197) - optimal draft-token selection under batch saturation - GRADE B+
Per-request selection of WHICH draft tokens to verify (not just uniform length), maximizing accepted length under a compute-saturation constraint coupling batch size to optimal depth. Justifies heterogeneous T per slot: hot slots keep T=6, cold drop to T=2-3 - draft budget per-slot, not per-window.

### 1.4 SpecServe -> AdaSpec (arXiv:2503.05096, SoCC'25) - SLO-aware adaptive speculation - GRADE B
Theoretical efficiency model under real-time load; controller toggles speculation per request against TTFT/TBT deadlines; up to 66% speedup vs SOTA SD systems. Template for wiring our ms/window + TTFT p95 telemetry into a T-feedback controller. Follow-ons: AdaServe (2501.12162, multi-SLO), FASER (2604.20503), WISP (2601.11652), Nightjar (2512.22420).

### 1.5 Nexus (arXiv:2507.06608) - proactive intra-GPU PD disaggregation - GRADE B
Anticipatory (not feedback) partitioning of one GPU's compute/memory/bandwidth between prefill and decode; treats memory-BW contention as the bottleneck. Up to 2.2x throughput vs vLLM, 20x lower TTFT, 2.5x lower TBT. Principle: size lane/arena widths for predicted load, don't react after TTFT breaks.

### 1.6 Mooncake (FAST'25, arXiv:2407.00079) - KVCache-centric serving - GRADE B-
Global KV pool (DRAM/SSD/CPU) + admission by cache affinity; ~75% of chatbot requests hit the cache at peak (Kimi production); pool-aware scheduling +75% request handling / +125% throughput in simulation. Validates our conversation parking (49 GiB, 9-15 ms restore); the lesson: affinity-aware admission is a proven lever, not a nicety.

### 1.7 DistServe (OSDI'24, arXiv:2401.09670) - GRADE B+
Up to 4.48x goodput / 7.2x request rate at same SLO vs co-located. Single-GPU lesson: don't let a prefill ride a decode window (cohorts already approximate this structurally). Defines goodput under simultaneous TTFT+TPOT SLOs - the right objective for the 450 goal + p99 targets.

### 1.8 SGLang v0.4 zero-overhead batch scheduler (LMSYS blog 2024-12-04) - GRADE A
CPU scheduler runs one batch ahead; future-token indirection + CUDA events resolve cross-batch deps; Nsight-verified zero GPU idle across 5 consecutive decode batches. 1.1x vs v0.3, 1.3x vs baselines; cites MLSys: unoptimized engines spend **up to half of wall time on CPU overhead**. Cache-aware router: 1.9x throughput, 75% vs 20% prefix hit (8xA100). Direct corroboration of our pool theorem; vLLM's analogue = Dual Batch Overlap.

### 1.9 vLLM V1 (blog 2025-01-27) - GRADE A
Async scheduling - CPU never blocks on GPU sync in the step loop; CPU overhead at batch ~32 fell 40% -> 12% (~70% reduction); up to **1.7x** vs V0, largest at SMALL batch (1.7x at batch 8, H100); zero-overhead prefix caching by default (free at 0% hit). Proves CPU/GPU overlap yields ~1.7x at small batch - our batch-6/2-cohort regime; our parking lookup should be unconditional-path cheap.

### 1.10 PegaInfer green-contexts writeup (pegainfer.org/blog/green-ctx) - GRADE A-
CUDA Green Contexts (12.4+, runtime API 13.1) split SMs into isolated domains; ~7-8 lines; graphs capturable. Memory-bound decode stops scaling past ~16 SMs on a 5070 Ti (>80% of theoretical BW); concurrent-kernel bench 1173 ms vs 1860 ms single-stream (~60% faster) on 5070 Ti, 34% on H200; isolation: 50 us kernel unblocked (~95 us e2e) behind a 10 ms kernel after 16/112 split. Caveats: L2 pollution from decode weight flush (mitigate with PTX .cs/evict_first - needs Blackwell, WE HAVE IT); **on a 5090 green contexts often hit the power wall**; heavy-prefill regressed; cross-stream alloc lifetime is the hard bug. Also: ETH SoCC'25 interference study - P90 TBT 14.2 ms for co-located Llama-3.1-8B without partitioning (matches our 14 ms early-window); DuetServe (2511.04791) = adaptive follow-on.

### 1.11 TensorRT-LLM PyTorch scheduler docs - GRADE A
Two-phase scheduler. CapacityScheduler = admission control: reserve blocks-to-completion for in-flight generations; admit CONTEXT_INIT only if the ENTIRE need-to-completion fits; pause rather than evict. MicroBatchScheduler splits context vs generation. The canonical production admission-fencing design - maps 1:1 to our promote/demote arena thrash.

## 2. Derivation baseline (our numbers)

- Solo 223-234; pair ~222 -> **intra-window batch scaling nearly flat past width 1-2**: a second slot in the same window buys ~0. Per-step time = fixed costs (weight read / launch / pool wait), not per-slot compute. Corollary: aggregate concurrency gains must come from overlapping windows, not window width. Cohort-2's +24.8% is the measured proof.
- Idle model: cohort-2 speedup 1.248 = 1/(1-i) -> residual non-hidden idle i ~ 20%. If residual scales ~1/cohort-depth, cohort-3 predicts ~306.8 x 1/(1-0.133) ~ 354 t/s (+~4.6% over 338.5). Cohort-3 ALONE does not reach 450 - must pair with a GPU-efficiency lever (draft throttle).
- Spec efficiency at fixed T=6: l(a,6)/7 per slot-position. a=0.8 -> 0.61; a=0.7 -> 0.51; a=0.6 -> 0.43. At T=3: 0.685/0.583/0.51. **Dropping T 6->3 raises verified-token yield per forward ~12-19%** (a=0.6-0.8) and frees per-step headroom that cohort overlap converts into window rate.

## 3. PROPOSALS (ranked by expected value)

### P1 - Acceptance-aware draft throttle (SmartSpec/TETRIS class): adaptive MTP T per slot/window
Track moving-average acceptance a per slot; per window choose T = argmax l(a,T)/(T+1) under current effective token-batch; clamp T in [2,6], hysteresis +-1; optionally per-slot (TETRIS) phase 2; under load >= target floor T=3; solo lane keeps T=6.
Gain: +12-19% verified-token efficiency x conversion fraction (cohort overlap usually converts) => 338.5 x 1.10-1.19 ~ **372-403 t/s**; SmartSpec's measured no-downside bounds risk near zero.
Risk: mis-estimated a on bursts; T changing mid-window vs cohort bookkeeping.
Cheapest falsifier: one probe set at fixed T=4 and T=3 at batch 6 / cohorts 2 (one-constant change in generate.cpp). Ship-adaptive only if aggregate >= +8% and solo unchanged. Falsified if aggregate drops at T=4 (means a ~ 0.85+ already).
Port: S-M. generate.cpp T constant + acceptance counters per slot; serve/server.py controller. 1-2 days.

### P2 - Cohort-3 enable (b9 lane fix + batch>=9 gate), staged with admission fencing
Land the b9 lane fix, batch 9 / STRATA_COHORTS=3; third verifier overlaps the other two.
Gain: ~354 standalone (+4.6%); **multiplicative with P1: 372-403 x 1.08-1.15 ~ 402-464** - the only derivation-backed path to 450.
Risk: pool (CPU) depth must feed 3 windows; TTFT p95 may rise with 9 in flight; 5090 power wall can cap.
Falsifier: canary probe batch 9 / cohorts 3, T fixed. Predicted >= 355 aggregate, TTFT p95 <= 1.1 s; falsified if < 345 (no hidden wait left - redeploy to P1).
Port: the known gate - b9 lane fix (slot-arena width assumptions, verifier scheduling depth 3) + config plumbing. Medium; already scoped.

### P3 - Admission fencing a la TRT-LLM CapacityScheduler (STABILITY lever)
Never admit a conversation to an active slot unless its FULL KV/slot-arena need-to-completion fits arena headroom; else park (49 GiB at 9-15 ms restore). Reserve-to-completion for active slots; demote, never evict. Affinity-biased admission: prefer warm parked state (Mooncake).
Gain: variance, not throughput: cut promote/demote events (2-15 ms each), tighten ms/window p95 (the 14 ms tail = interference-polluted early-window; ETH's 14.2 ms P90 analogue), TTFT p99 < ~1.2 s. Indirect +2-4%.
Risk: over-conservative admission under fragmentation; needs a floor.
Falsifier: instrument promote/demote count + ms/window p95 one probe set with the fence on; falsified if neither moves while aggregate drops >3%.
Port: S. serve/server.py admission path only. 0.5-1 day. Highest stability-per-line in this dossier.

### P4 - Green-context / MPS lane isolation for the solo lane (TTFT p99 + solo-record protection)
Partition SMs (~16/112-class split: decode saturates BW near ~16 SMs on this class) into a solo/latency lane and a cohort/throughput lane; PTX .cs/evict_first cache hints (Blackwell) to blunt L2 cross-talk.
Gain: aggregate-neutral to -5%; the point is tails: solo lane keeps ~full solo t/s during concurrent probes; isolated-lane TTFT p99 approaches p50 (PegaInfer: 50 us kernel ~95 us e2e behind a 10 ms kernel).
Risk: highest operational - cross-stream allocation lifetime bugs (Xid), 5090 power wall may erase the benefit, static splits idle SMs (Nexus/DuetServe argue adaptive).
Falsifier: 10-line green-context smoke test OUTSIDE the engine: solo probe while batch-6 probe runs, with/without SM split; watch nvidia-smi power. Abort if power-capped.
Port: M-L. Stream/GPU-RAII restructuring + serve/server.py lane routing. After P1-P3.

### P5 - Window-width modulation by queue depth (Sarathi class)
Width is nearly free real estate (pair ~ solo): modulate width 2-6 by queue depth; bound co-scheduled prefill chunks so decode ms/window stays <= ~12 ms (stall-free rule); narrow windows at low occupancy (lower per-step cost -> TTFT p50).
Gain: +3-8% aggregate at the right times + TTFT p50; Sarathi's 2.6-7.75x goodput band is the upper-bound analogy, not a prediction.
Risk: width oscillation = arena churn (interacts with P3); wrong bounds recreate the 14 ms tail.
Falsifier: offline replay of probe traces through a width-2 vs width-6 window simulator on recorded ms/window histograms; predicted delta < 3% => skip.
Port: M. serve/server.py width policy + generate.cpp arena re-sizing hooks (VMM-elastic arenas already at 2-15 ms).

### Ranking rationale
P1 first (largest derivation-backed gain, cheapest falsifier, low blast radius). P2 only with/after P1 (cohort-3 alone ~354, short of goal; combined 402-464 is the 450 path). P3 immediately after P1 regardless (cheap, targets the stability half of the mandate). P4 = solo-lane/p99 specialist gated on the power-wall smoke test. P5 last (simulator first).

## 4. Cross-cutting notes
- Pool theorem independently corroborated: SGLang (~50% wall time CPU-side; overlap = 1.1x) + vLLM V1 (40%->12% = ~1.7x at small batch). Residual CPU segments (parking restore, arena promote/demote, verifier prep) are the remaining tax - P2/P3 attack exactly those.
- Renamed-paper trap: 2406.14066 = SmartSpec->TurboSpec; 2503.05096 = SpecServe->AdaSpec. Cite both names when searching.
- Not covered: Preble/CacheGen/LMCache prefix-transfer (parking covers the hot path); SGLang TBO (multi-GPU); learned admission (if P3's static fence proves too coarse).

Sources: [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) | [2406.14066 v1](https://arxiv.org/html/2406.14066v1) | [TETRIS](https://aclanthology.org/2025.acl-long.1598) | [AdaSpec](https://arxiv.org/abs/2503.05096) | [Nexus](https://arxiv.org/abs/2507.06608) | [Mooncake](https://arxiv.org/abs/2407.00079) | [DistServe](https://haoailab.com/blogs/distserve) | [DuetServe](https://arxiv.org/html/2511.04791v2) | [ETH SoCC'25](https://anakli.inf.ethz.ch/papers/gpu_interference_socc25.pdf) | [SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4) | [vLLM V1](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) | [vLLM DBO](https://docs.vllm.ai/en/v0.27.0/design/dbo) | [PegaInfer green-ctx](https://pegainfer.org/blog/green-ctx) | [TRT-LLM scheduler](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/torch/scheduler.md)
