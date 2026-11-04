---
type: research-digest
date: 2026-11-04
topic: Serving-layer latency, ITL smoothness, admission/reliability engineering, speculative-decoding serving integration, prefix-cache advances (2024-2026 literature + production stacks)
machine-context: AM5 single-socket 16C/32T, dual-channel DDR5 (49.5 GB/s measured userland ceiling), RTX 5070 12 GB (sm_120), 184 GB RAM, NVMe, Fedora. One Python ThreadingHTTPServer fronts a hybrid MoE engine (decode reads experts from host RAM at 93-100% of DRAM ceiling; spec decoding ~88-94% acceptance; conversation-cache prefix reuse 93-94%; lazy CUDA graphs per cohort batch shape; parallel=6 slots). Second MoE engine (748B-A16B class, RAM-copied experts, own drafter) alternates on the same port. In-tree: prefill-cadence throttle (vLLM PR #44558 shape), max-queue + queue-timeout, idle unload/reload, restart with 503 "starting", slot drain/restore, batch telemetry. Measured: TTFT = 539 ms fixed + 6.3K tok/s marginal; live traffic 150-175K ctx at 191-210 t/s; decode windows 9-16 ms; serial attn-to-moe lane per sequence.
status: verified
sources: arXiv HTML full texts (fetched), OSDI/MLSys/EuroSys/FAST/NSDI papers, lmsys.org/vllm.ai/cohere.com engineering blogs (fetched), vLLM/SGLang docs + GitHub issues (fetched/searched), Spheron ops article (fetched) — every claim carries its source inline; search-level-only claims are flagged.
---

# Serving smoothness + reliability digest (2026-11-04)

Falsified-boundary respected throughout: cluster / ≥2-GPU mechanisms (DistServe, Splitwise, PD disaggregation) are recorded only for their single-box-transferable takeaway. SMT-on-CPU-lane and multi-GPU fusion are not re-proposed.

## 1. ITL smoothness / decode-prefill interference

### 1.1 Sarathi-Serve stall-free batching — the canonical mechanism (OSDI'24)
[USENIX](https://www.usenix.org/conference/osdi24/presentation/agrawal) | [PDF](https://www.usenix.org/system/files/osdi24-agrawal.pdf) | [arXiv:2403.02310](https://arxiv.org/html/2403.02310v3) (fetched, v3).

- Mechanism: split every prefill into near-equal compute-sized chunks; per iteration (a) compute a token budget from the TBT SLO via one-time profiling, (b) pack ALL ongoing decodes first (1 token each), (c) fill the remaining budget with one prefill chunk, (d) admit new requests only into leftover budget. Iteration latency becomes bounded and prompt-length-independent.
- Chunk-size guidance (fetched): strict-SLO budget **512 tokens**, relaxed **2048** (LLaMA2-70B relaxed: 1536). Power-of-two alignment matters — chunk of 257 vs 256 can add **32% prefill time** (tile quantization). Chunking overhead ≤~25% at 512, "almost negligible" at 2048 (Yi-34B).
- Numbers (fetched): naive hybrid batching = up to **28.3× TBT increase** vs decode-only; vLLM stalls "over several seconds"; capacity under strict TBT SLO: **2.6×** vs vLLM (Mistral-7B, 1×A100), **3.7×** (Yi-34B), **4.3×** (LLaMA2-70B), up to **5.6×** (Falcon-180B). Yi-34B ablation at budget 1024: hybrid-only P99 TBT 0.68-1.38 s; chunked-only 0.17-0.20 s; combined 0.14-0.17 s. Stall-free batching alone (no chunking) yields only ~half the latency gain.
- Key physics (fetched, Fig. 9 discussion): the marginal ITL cost of piggybacked prefill tokens **shrinks** as decode batch size and context grow (more arithmetic-intensity slack in memory-bound decode iterations); iteration time is T = max(T_math, T_mem) — the budget is "max tokens packable without breaking the TBT SLO."
- Infra: none special — scheduler-level, single GPU fine.

### 1.2 vLLM chunked-prefill scheduler details (production descendant)
[Performance & tuning docs](https://docs.vllm.ai/en/v0.9.1/configuration/optimization) | [v0.4.2 docs](https://docs.vllm.ai/en/v0.4.2/models/performance.html) | [halfrost scheduler walkthrough](https://github.com/halfrost/LLM-ML-Repo-Notes) (search-level).
- `max_num_batched_tokens` = per-iteration token budget; with chunked prefill on, the scheduler **prioritizes all pending decodes first**, then spends the leftover budget on prefill chunks — this is why decode ITL stays stable (Sarathi Algorithm 3 shape, shipped).
- Same budget is also the preemption/memory-planning knob — raising it raises TTFT throughput but risks ITL spikes.

### 1.3 TWP token-window pacing — closest match to our box (arXiv 2610.17689)
[arXiv HTML](https://arxiv.org/html/2610.17689v1) (fetched).
- Mechanism: closed-loop token-granularity admission controller in the serving layer, no engine changes. Per decode step: smooth ITL with an exponential filter; update a dynamic window H_t = soft cap on **total remaining output tokens in flight** (not request count) from normalized slowdown σ_t = ℓ̂_t/ℓ_min; admit a pending request only if (token window has room) AND (KV headroom > δ_kv); starved requests force-admit past T_max; **prefill staged in chunks of C=64 tokens, one chunk per decode step** to bound the noisy-neighbor memory-bus penalty. Whole feedback loop fits inside one decode-step boundary.
- Numbers (fetched, single Tesla V100 32 GB, Llama-3.2-3B FP16, Poisson arrivals): throughput **0.197 req/s vs 0.136** (ServerlessLLM) = **+45%**; TTFT P99 **66 ms** vs 78 ms; ITL tight at 43.5-43.8 ms vs baseline tails 61-62 ms; **lowest ITL coefficient of variation (0.0144 vs 0.0203 RR)**; P99 latency under burst 10.3 s vs 17-18 s baselines.
- Infra: pure scheduling layer between queue and backend; 3 tunables from one profiling sweep (H₀ = B₀·L̄ = 400 tokens, gain γ=0.15, C=64).
- Why this matters here: this is a single-GPU, scheduling-layer-only result — the exact insertion point of our Python front server.

### 1.4 SGLang v0.4 zero-overhead overlap scheduler
[LMSYS blog 2024-12-04](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4) (fetched) | [server args (`--disable-overlap-schedule`)](https://docs.sglang.ai/advanced_features/server_arguments.html) | [overlap-failure case, GitHub issue #19347](https://github.com/sgl-project/sglang/issues/19347).
- Mechanism: scheduler runs **one batch ahead** — prepares next batch metadata (scheduling, memory allocation, prefix matching, radix-cache ops) while the GPU computes the current batch; the token dependency is resolved with "future tokens" + carefully placed CUDA events. Idea credited to NanoFlow ([arXiv:2408.12757](https://arxiv.org/abs/2408.12757)).
- Numbers (fetched): **1.1× throughput vs SGLang v0.3, 1.3× vs other baselines** (Llama-3.2-3B, 500 prompts, 4096-in/2048-out); largest gains on small models + large TP. No per-token latency claim.
- Caveat: overlap silently fails in some configurations (issue #19347) — needs Nsight-style verification that GPU idle gaps actually closed.
- Infra: single GPU, on by default in SGLang v0.4+.

### 1.5 PDMux / MuxWise — intra-GPU prefill-decode multiplexing (kernel-level)
[LMSYS blog 2025-09-28](https://www.lmsys.org/blog/2025-09-28-pdmux) | [arXiv:2504.14489](https://arxiv.org/html/2504.14489v3).
- Mechanism: overlap compute-bound prefill and memory-bound decode **on the same GPU** via fine-grained SM/communication scheduling so phases don't block each other; claims high-goodput without disaggregation.
- Infra: GPU kernel + SM-partitioner control. **Priced: DEAD for our engine** (we don't own attention-kernel scheduling at that level). Transferable takeaway only: P and D stress disjoint resources, so the interference to arbitrate is the shared resource — on our box that shared resource is DRAM bandwidth, already arbitrated by the in-tree cadence throttle.
- Sibling: DuetServe ([arXiv:2511.04791](https://arxiv.org/html/2511.04791v1)) — SM-level spatial partitioning on a single GPU; same DEAD verdict.

### 1.6 DualPace — CPU as an independently scheduled decode engine (our architecture, independently validated)
[arXiv:2610.33501](https://arxiv.org/html/2610.33501) (fetched).
- Mechanism: two engines under one LLMEngine — GPU engine (owns all prefills + GPU-resident decodes) and CPU engine (owns CPU-resident decodes with **its own queue, scheduling clock, KV manager**). "**There is no global iteration barrier** requiring the two batches to start or finish together." GPU helps CPU via "Dense Assist" stage-boundary calls that never enter the GPU scheduler queue. Global host-KV safety constraint: B_used + B_planned + B_growth + B_migration ≤ B_capacity; swap-in only if the request's finish-peak KV footprint fits through completion (anti-oscillation admission rule).
- Numbers (fetched): decode-oriented sustainable rate **+6.9-43.2% over NEO**, **+11.5-90.6% over vLLM** at equal latency; at 1.5 s P99 ITL target **+7.3-50.5% request rates vs NEO**; CPU "TP=2" adds 15-30%; host-KV pool 120 GiB beats 30/60 by up to 28.7%.
- Infra: Xeon with **AMX** for the CPU dense lane (Sapphire Rapids) — we lack AMX on AM5, so the Dense-Assist speedups don't transfer. The **scheduling-authority split does transfer**: per-slot independent clocks, no global barrier, admission gated on finish-peak footprint. (Note: cluster-free, single GPU + host DRAM.)

### 1.7 ConServe (agentic, conversation-level) + ConServe (GPU harvesting) — name collision warning
- Conversation-level disaggregation: [arXiv:2606.01839](https://arxiv.org/abs/2606.01839) (search-level). Schedules at the **conversation** level, transfers KV at most once per conversation, **pins the conversation to one decoder instance** across turns, acts on observed cache hits rather than predictions. Robust vs AMPD under prediction error. Cluster-shaped (prefiller/decoder tiers) but the pinning idea is single-box.
- Fine-Grained GPU Harvesting ConServe: [arXiv:2410.01228](https://arxiv.org/abs/2410.01228) (ICML'26, search-level) — runs online + offline jobs on the same GPU with preemption; **P99 online TTFT reduced 76.5%** while offline throughput +1.04× (revised: P99 latency 2.9× lower avg, offline throughput +2.2×). Directly relevant if we ever run offline/batch jobs on this box: harvest idle GPU cycles but preempt instantly on online arrivals.

## 2. Admission control & queueing

### 2.1 vLLM preemption: RECOMPUTE vs SWAP
[vLLM optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization) | [nm-vllm performance docs](https://nm-vllm.readthedocs.io/en/0.5.0/models/performance.html) | [preemption discussion](https://discuss.vllm.ai/t/request-preemption-option/1672) | [source-level walkthrough](https://labhub.hopto.org/blog/ai-platform/2026-08-12-vllm-scheduler-and-preemption?lang=en) (search-level).
- V1 default = **RECOMPUTE**: victim's KV blocks are discarded; request rescheduled later and prefill recomputed. SWAP offloads KV to host and swaps back — preferable for long-context victims, at the cost of host-side KV management. Scheduler preempts the **last-arrived** request(s) under KV pressure and tries to avoid preemption entirely via conservative scheduling.
- Economics note for pricing: recompute cost = full prefill re-read. At our 6.3K tok/s marginal and 150-175K live contexts, one preempt-recompute ≈ **24-28 s of pure penalty** — an order of magnitude worse than vLLM's 4K-context regime.

### 2.2 Backpressure / shedding playbook (Spheron ops article)
[LLM Serving Backpressure: Queuing, Load Shedding, and Admission Control](https://www.spheron.network/blog/llm-inference-overload-queuing-load-shedding-and-admission-c) (fetched).
- Unbounded default queue = everything times out together. Fix: cap the queue, reject fast with 503 (vLLM `--max-waiting-queue-length` proposal → `SchedulerWaitingQueueFullError` → HTTP 503; LiteLLM `max_queued_requests_per_worker` + `admission_queue_timeout_seconds`). Measured example: 150-request cap → rejections in **<15 ms** instead of 8 s timeouts.
- Priority tiers with split queue budget (150 total): health checks 5 reserved (never shed), interactive 120, background 25 (shed first).
- Admission signal: gate on **in-flight count + queue depth**, not RPS (per-request cost varies 100×). Anthropic-style triple limit: requests + input tokens + output tokens per minute.
- Semantics: **429** = per-caller quota (client should slow down); **503/529-class** = system capacity (fast reject, retryable); always send `Retry-After`; client backoff must be jittered (unjittered retries synchronize into waves; measured +35% retry amplification; Google SRE adaptive throttling at requests:accepted ratio K=2).
- Watch **queue depth as leading indicator** — it moved at minute one while mean TTFT stayed flat (190→210 ms) and errors lagged to minute three.

### 2.3 Cancellation hygiene (production bug history)
[vLLM #10087 — client disconnect not aborting](https://github.com/vllm-project/vllm/issues/10087) | [#4240 abort streaming](https://github.com/vllm-project/vllm/issues/4240) | [PR #11190 polling-free cancellation](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/11190/overview) | [#54947 keeps generating after close](https://github.com/vllm-project/vllm/issues/54947) | [vLLM API abort pattern](https://docs.vllm.ai/en/v0.9.0/api/vllm/index.html) | [vllm-omni backpressure RFC #6646](https://github.com/vllm-project/vllm-omni/issues/6646).
- Canonical pattern: on `request.is_disconnected()` → `engine.abort(request_id)`; PR #11190 made cancellation transparent and polling-free. Known failure modes: Starlette swallowing disconnects (#10087), backend cancellation not fully propagating → wasted generation, abort counter staying 0 after hard `stream.close()` (#51921). TCP window backpressure from a slow client can stall the output path ([SSE/backpressure explainer](https://dev.to/kuldeep_paul/streaming-llm-responses-end-to-end-sse-backpressure-and-mid-generation-cancellation-4d60)).
- Lesson: disconnect detection must live at the socket-write boundary, and abort must free the slot + draft work immediately.

### 2.4 Fairness (multi-tenant)
["Fairness in Serving Large Language Models" OSDI'24 (VTC)](https://www.usenix.org/conference/osdi24/presentation/sheng) (search-level) | [Equinox holistic fair scheduling, arXiv:2508.16646](https://arxiv.org/abs/2508.16646) | [Locality-aware fair scheduling, arXiv:2501.14312](https://arxiv.org/abs/2501.14312).
- Virtual-Token-Counter fairness: cost function f(input, output tokens); schedulers allocate by virtual token debt; reduces unfairness **~40×** vs vLLM at near-zero slowdown. Equinox (2025) argues single-metric fairness breaks under prefill/decode bifurcation.

## 3. SLO-aware scheduling

### 3.1 SOLA — state-aware scheduling (MLSys 2025)
[OpenReview](https://openreview.net/forum?id=ubIvpetAd6) | [slides](https://mlsys.org/media/mlsys-2025/Slides/3231.pdf).
- Mechanism (search-level; OpenReview was bot-gated on fetch): records last iteration's scheduling state and optimizes the current iteration jointly over the **TTFT-vs-TPOT and across-request tradeoffs**. Reported: SLO attainment improved **from 45.5% to near-100%** in evaluated scenarios vs vLLM default.
- Infra: scheduler-level, vLLM-based, single-instance.

### 3.2 SLOs-Serve — soft admission + guaranteed tier (arXiv 2504.08784)
[arXiv HTML](https://arxiv.org/html/2504.08784v1) (fetched).
- Mechanism: per request a prefill deadline (pDDL) + TPOT choice; scheduler admits only the subset whose cumulative token demand fits the roofline throughput line — **admitted = SLO-guaranteed until completion; the rest fall to a best-effort tier**, preemptible by SLO-guaranteed arrivals, and preemption **discards KV but keeps generated tokens**. SLO-adaptive speculative decoding lengths per request; burst requests degrade to OpenAI-batch-API-style best effort.
- Numbers (fetched): serving capacity at 90% SLO attainment = **2.2× geo-mean** over best of Sarathi/vLLM, 2.4× over DistServe; ablations: SLO-adaptive spec decoding **1.66×**, burst-resilient tiering **1.34×**, routing 1.19×; scheduler overhead **<10 ms, mostly <2 ms**.
- Infra: vLLM backend, ~10K LOC Python; offline roofline profiling per hardware. Multi-replica routing is cluster — not needed for the tiering mechanism itself.

### 3.3 Others
- [AdaServe, EuroSys'26 (CMU)](https://www.cs.cmu.edu/~zhihaoj2/papers/AdaServe_EuroSys26.pdf): multi-SLO serving via SLO-customized decoding; predicts speculative-token accuracy from draft logits to pick per-request speculation — the scheduling-optimal version of §5.
- [JITServe, NSDI'26](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf): SLO attainment with imprecise request info; up to 83.2% resource savings vs SOTA (search-level).
- [SLO-aware scheduling via simulated annealing, arXiv:2504.14966](https://arxiv.org/html/2504.14966v2): up to 5× SLO-attainment gain, −31.6% avg latency (search-level).
- [llm-d predicted-latency scheduling](https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms): production-style admission from live TTFT/TPOT predictions (search-level; cluster-oriented).

## 4. Reliability engineering for inference servers

### 4.1 Watchdog + health endpoints (SGLang pattern)
[Server args (`--watchdog-timeout`)](https://sgl-project.github.io/advanced_features/server_arguments.html) | [watchdog fire on hierarchical cache, issue #12756](https://github.com/sgl-project/sglang/issues/12756) | [scheduler-death detection lag, issue #18421](https://github.com/sgl-project/sglang/issues/18421) (fetched/searched).
- Mechanism: a watchdog thread checks forward-batch progress; if a batch exceeds `--watchdog-timeout` (default **300 s**) the server **crashes itself deliberately** rather than hang forever. HTTP layer: `/health` (process alive) and `/health_generate` (engine actually completes a generation probe). Known gap: if the scheduler subprocess dies, health checks still wait `HEALTH_CHECK_TIMEOUT` (default **20 s**) before failing.
- Design lesson: two-tier health — cheap liveness vs engine-can-decode probe; and the watchdog must be set from your own latency scale (ours: decode windows 9-16 ms, so a 1-2 s no-progress trip is 100× headroom, not 300 s).

### 4.2 Mooncake production lessons (FAST'25)
[USENIX PDF](https://www.usenix.org/system/files/fast25-qin.pdf) | [arXiv:2407.00079](https://arxiv.org/html/2407.00079v4) | [MooncakeStore + vLLM docs](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/kv-cache-storage.html) | [repo](https://github.com/kvcache-ai/Mooncake) (fetched/searched).
- Single-instance crash is treated as **tolerable**: connections drop, instances operate independently; the transfer layer does health checks + path failover (cluster part N/A here).
- Eviction economics: evict KV by **expected recompute cost vs memory pressure** — too-aggressive eviction turns into recomputation storms; hoarding starves active SLOs.
- Overload behavior: fall back to a smaller model rather than dropping requests (Kimi routes overflow to a LightLLM-class fallback).
- Lesson matching our tree: our engine-restart-with-503 + slot drain/restore already implements "instance crash tolerable, state recoverable"; the missing piece is automatic trip (watchdog) and the fallback tier.

### 4.3 Graceful degradation / brownout patterns
[Spheron backpressure article](https://www.spheron.network/blog/llm-inference-overload-queuing-load-shedding-and-admission-c) (fetched) | [GraceInfer, ResearchGate](https://www.researchgate.net/publication/414626269_Semantic-Aware_Graceful_Degradation_in_Multi-_Tenant_AI_Inference_Backends) | [LiteLLM fallbacks docs](https://docs.litellm.ai/docs/proxy/reliability) | [GMI fallback routing](https://www.gmicloud.ai/en/blog/primary-and-fallback-models-how-auto-routing-improves-reliability-in-ai-applications) | [circuit-breaker state machine explainer](https://blog.levelupcoding.com/p/circuit-breakers-clearly-explained) (search-level).
- Standard degradation ladder: shed background → truncate context → route to smaller/faster model → 503 with Retry-After. Circuit breaker = CLOSED/OPEN/HALF-OPEN per dependency with failure-count trip.
- On our box the "smaller model" tier already exists structurally: the second engine on the same port. A brownout mode = cap parallel slots, force short draft length, allow one prefill chunk per round, route interactive traffic to the fast engine and defer everything else.

### 4.4 Engine reliability research
[Reliability of LLM Inference Engines from a Static Perspective (MDPI)](https://www.mdpi.com/2504-2289/10/2/60): automated root-cause analysis over vLLM-class issue corpora; failure taxonomies concentrated in scheduler/detokenizer/output-path code — the same places the vLLM stall issues above live ([#51921 permanent stall](https://github.com/vllm-project/vllm/issues/51921), [#52319 silent hang](https://github.com/vllm-project/vllm/issues/52319)).

## 5. Speculative-decoding serving integration

### 5.1 Cohere DSD — hardware-aware dynamic draft length (the priced one)
[Cohere blog](https://cohere.com/blog/hardware-aware-dynamic-speculative-decoding) (fetched).
- Mechanism: goodput = AL/ITL (AL = acceptance length, decays exponentially per draft position). Extend draft count K until marginal AL gain < marginal ITL cost; offline-profile AL and ITL into a lookup table. K moves **up when memory-bandwidth-bound** (weight reads dominate, compute idle), **down when compute-bound**. **MoE is non-monotonic**: low K at low batch (each extra draft token loads additional experts), higher K at mid batch (experts already resident), low K again at high batch.
- Numbers (fetched): up to **23% faster than fixed-K SD** (EAGLE K=3 baseline) at batch 64-128 (dense Command A); 7.5% faster than no-SD vanilla at BS 128; low BS: matches fixed-K. MoE model: little gain (attributed to EAGLE single-timestep training).
- Deployment (fetched): landed in vLLM via 2 PRs — async-scheduling bookkeeping (draft/verify counts no longer 1:1 per step) and CUDA-graph capture extended to all valid K per batch size so runtime K changes still hit a captured graph.
- Infra: single GPU, no extra hardware.

### 5.2 SpecServe — SLO-aware adaptive speculation (arXiv 2503.05096)
[arXiv HTML](https://arxiv.org/html/2503.05096v1) (fetched).
- Mechanism: (a) **adaptive drafter** — predict-execute-correct loop; continue drafting while estimated efficiency (expected accepted tokens ÷ modeled quadratic time cost) improves, else stop; raises draft length under low load/high acceptance, shortens under high load. (b) **confidence-prior verifier** — pruning likely-rejected draft tokens in ascending acceptance order before verify (verification can be up to **90% of SD time**); (c) SLO estimator blocks drafting when the step would exceed TPOT SLO; **auto-fallback to autoregressive under bursts**.
- Numbers (fetched): **1.14-14.3×** vs SOTA speculative systems, 1.1-1.8× vs autoregressive across 15 setups; SLO attainment lead **5-38%**; adaptive drafter alone +6-55% speedup; fixed-SL policies fall BELOW 1× speedup in 7% (SL=1) / 13% (SL=3) / 40% (SL=5) of setups; ~1 in 10 pruned tokens would have been accepted (pruning is safe).
- Infra: vLLM + ~2K LOC; offline linear-regression fit of forward-pass timings after any model/platform change; draft+target KV split in halves.

### 5.3 DISCO/SLi — per-iteration speculation-lookahead classifier (arXiv 2405.04304)
[arXiv HTML](https://arxiv.org/html/2405.04304v4) (search-level with exact figures quoted from abstract/paper snippets).
- Mechanism: a lightweight classifier chooses the speculation lookahead per iteration.
- Numbers: **+10.3% average speedup over the best static SL**, **+31.4% over prior dynamic heuristics**, identical outputs.

### 5.4 DSDE — KLD-stability control (arXiv 2509.01083)
[arXiv abs](https://arxiv.org/abs/2509.01083) | [HTML](https://arxiv.org/html/2509.01083v1) (fetched).
- Mechanism: post-hoc diagnostic — KL divergence between draft and target logits per verify step; exponentially-weighted variance ratio over short (N=10) vs long (N=30) windows (WVIR; >1 = growing instability); next SL = (1 − SF·WVIR)·(SL_max − SL_min) + SL_min, collapsing to SL_min=2 under instability; per-batch cap = arithmetic mean of per-sequence predictions (fixes the straggler effect); SL_max calibrated at generation start from a few preliminary steps.
- Numbers (fetched, 8×A100): matches static-opt within 1-7% at temp 0 **without static-opt's ~22 h of per-dataset profiling**; in low-acceptance regimes avoids competitor blowups (180% vs 234-335% latency increase); batch 1→64 scaling: 12.16×/13.01× with the cap vs 11.21×/11.92× naive per-sequence.
- Caveat (fetched): runs in vLLM eager mode, no CUDA graphs — on our box the lazy per-cohort CUDA graph capture must treat K as a capture dimension (same fix as Cohere's vLLM PR).

### 5.5 Production knobs
[SGLang adaptive speculative decoding docs](https://lmsysorg.mintlify.app/docs/advanced_features/adaptive_speculative_decoding) (search-level): runtime adjustment of `speculative_num_steps` / `speculative_num_draft_tokens` is a shipped feature — precedent that dynamic K is a serving-layer knob, not exotic.

## 6. Prefix / conversation cache advances since RadixAttention

### 6.1 Baseline and TTL
[vLLM APC design](https://docs.vllm.ai/en/stable/design/prefix_caching): block-hash-keyed reuse, **LRU eviction, no TTL option** ([forum confirmation](https://discuss.vllm.ai/t/how-to-adjust-prefix-caching-timeout/1663)). [LMCache config](https://docs.lmcache.ai/mp/configuration.html): `--l1-write-ttl-seconds` **default 600 s**, `--l1-read-ttl-seconds`, session TTL; tiered LRU GPU→CPU→disk ([LMCache repo](https://github.com/lmcache/lmcache), [tech report](https://lmcache.ai/wp-content/uploads/2026/07/tech_report.pdf)). AWS guidance: TTL-based invalidation for KV tiering ([Redis + LMCache blog](https://redis.io/blog/get-faster-llm-inference-and-cheaper-responses-with-lmcache-and-redis)).

### 6.2 KVFlow — workflow-aware eviction (NeurIPS 2025)
[arXiv HTML](https://arxiv.org/html/2507.07400v1) | [NeurIPS page](https://neurips.cc/virtual/2025/poster/119883) | [OpenReview PDF](https://openreview.net/pdf?id=5Iw1nDtYmT) | [SGLang RFC #24656](https://github.com/sgl-project/sglang/issues/24656) (search-level).
- Mechanism: model agent execution as an Agent Workflow Graph; evict KV of agents with the **largest steps-to-execution** first — keep what will be reused soonest. Up to **1.5× speedup** from eliminating cache misses for imminent steps.

### 6.3 CacheTTL — TTL pinning after tool calls (arXiv 2511.02230)
[arXiv HTML](https://arxiv.org/html/2511.02230v4) (fetched).
- Mechanism: when a turn containing a tool call finishes, pin its KV with TTL τ* chosen to maximize expected net benefit = CacheMissCost (prefill cost × footprint) + OutofOrderCost − memory-blocked cost, with P(τ, f) an empirical CDF of that tool's historical duration; victims unpinned by latest arrival on deadlock; priority tuple: preempted > TTL-pinned > arrival order.
- Numbers (fetched): delay reduced **1.12-3.66×** across setups, up to **8.18×** on real SWE-agent workloads; vs plain vLLM up to **2× avg response-time reduction** (Llama-3.1-8B); RL rollout 144.9 vs 93.4 steps/min (vLLM); overhead **~1 ms**; ~1K LOC on vLLM 0.10.2.
- Infra: vLLM + optional LMCache CPU offload; client sends program_id; <10 min profiling.

### 6.4 CacheGen — compressed/streamed KV (SIGCOMM'24) + CacheBlend — non-prefix reuse (EuroSys'25 best paper)
CacheGen: [arXiv:2310.07240](https://arxiv.org/abs/2310.07240) | [ACM](https://dl.acm.org/doi/10.1145/3651890.3672274) | [LMCache integration post](https://blog.lmcache.ai/2025/20250715-cachegen/) (search-level) — custom tensor encoder, **bandwidth-adaptive bit-widths (8→6→4)**, ~3× smaller than plain quantization at preserved quality; targets network/disk fetch of long contexts.
CacheBlend: [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) | [EuroSys PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/09/eurosys25-final999.pdf) (search-level) — reuse precomputed KV for **non-prefix (RAG) chunks** by selectively recomputing KV for a small token subset (cross-chunk attention correction); ~**100% KV hit rate, ~3× RAG speedup**, 15× TTFT reduction claims in paper. Follow-ups: CacheClip ([arXiv:2510.10129](https://arxiv.org/abs/2510.10129)), InfoBlend (OpenReview).
- Infra: both implemented in LMCache; CacheBlend needs engine-level selective recompute.

### 6.5 Marconi (hybrid SSM+attention prefix caching, MLSys'25)
[MLSys page](https://mlsys.org/virtual/2025/poster/3360) | [vLLM pluggable-eviction RFC #64150](https://github.com/vllm-project/vllm/issues/64150) (search-level). Admission/eviction by actual KV **footprint** rather than token count. Only relevant if we ever host SSM-hybrid models — noted for the footprint-based-admission principle.

---

## Priced for our box

Ranked by (expected ITL/TTFT gain × implementation cheapness ÷ risk to standing decode speed). Serving-layer Python changes are cheap; anything needing an engine bench protocol is marked. DEAD items named at the end.

### Tier 1 — do first (serve-layer, near-zero risk to decode)

**1. Token-budgeted prefill chunking + decode-first packing (Sarathi §1.1 + TWP §1.3) — complement the in-tree cadence throttle with a token budget.**
Price: our cadence throttle bounds *how often* a prompt read happens but not *how big* each read is; a 6.3K-tok/s marginal means a round that reads C prompt tokens adds ≈ C/6300 s to that round (C=2048 → ~325 ms = ~20-36 decode windows; C=256 → ~41 ms ≈ 2-4 windows; C=64 → ~10 ms ≈ 1 window — invisible). Add per-round prompt-token budget B (power-of-two, start 256-512), pack all active-slot decodes first, spend leftover budget on one prefill chunk — exactly Sarathi Algorithm 3 with our measured constants. TTFT cost: a 150K prompt takes ~24 s regardless; chunking only re-shapes its interference footprint, which is the point. Cheap (front-server + engine admission hook), no new kernels, no risk to the 93-100% DRAM steady state — the wall is respected because chunk reads ride the same expert path.

**2. Two-tier admission + overload semantics (SLOs-Serve §3.2 + Spheron §2.2).**
Price: cheap, pure front-server. Guaranteed tier = requests whose predicted TTFT (539 ms + est-prefill-cs/6.3K + queue) fits budget; everything else → best-effort slot or the second engine as the overflow/brownout tier (SLOs-Serve showed the tiering alone is worth **1.34×** capacity; Cohere/Mooncake use small-engine fallback under overload). Add: 429 vs 503 distinction, `Retry-After`, queue-depth-first telemetry (leading indicator), reserved health slots. Existing max-queue + queue-timeout already implements half of this — this prices as finishing the pattern.

**3. Decode-window watchdog + /health_generate + engine circuit breaker (SGLang §4.1 + Mooncake §4.2).**
Price: cheap. Our decode windows are 9-16 ms; trip a watchdog if no round completes in ~1-2 s (100× headroom) → drive the in-tree restart-with-503 path automatically; `/health_generate` = one probe token through whichever engine is resident; per-engine CLOSED/OPEN/HALF-OPEN breaker so a sick 748B engine can't blackhole the shared port — route to the healthy one and mark the sick one "starting". This converts the existing manual restart machinery into self-healing.

**4. Cancellation hygiene + slow-client backpressure (§2.3).**
Price: cheap. Detect closed sockets at the write boundary, abort immediately (vLLM's bug history shows every variant of this failing in production), free the slot + discard draft work, count aborted-but-generated tokens in telemetry. With parallel=6, one zombie client is 17% of capacity.

**5. Conversation→slot/engine pinning + TTL + time-to-next-turn eviction (ConServe §1.7 + CacheTTL §6.3 + KVFlow §6.2 + LMCache TTL §6.1).**
Price: cheap-to-moderate front-server + cache metadata. We already hold 93-94% prefix hit and slot drain/restore — add: (a) sticky conversation→slot affinity across turns (ConServe's "observation, not prediction"), (b) TTL default 600 s (LMCache default) on idle conversation prefixes so abandoned 150K+ prefixes stop squatting on DRAM/VRAM, (c) evict-by-time-to-next-turn (KVFlow's steps-to-execution analog) instead of pure LRU. CacheTTL measured **1.12-3.66×** delay reduction and ~1 ms overhead from exactly this class of policy.

**6. TWP token-window pacing as the slot-admission gate (§1.3).**
Price: cheap. Soft cap H_t on total remaining output tokens across the 6 slots (est. from per-request max_tokens minus emitted), gate new admissions on H_t + KV headroom; expand/contract from smoothed ITL. Keeps the batch shape inside the DRAM wall instead of overfilling slots and letting every window stretch. TWP: +45% throughput, ITL CV 0.0144, all scheduler-layer on one V100 — the strongest direct evidence for a single-box Python front.

### Tier 2 — engine bench protocol required, high expected payoff

**7. Dynamic draft length K per cohort/batch shape (Cohere DSD §5.1 + SpecServe §5.2).**
Price: engine change with a clean bench protocol. Our decode lane is DRAM-bound at low slot counts — Cohere's rule says K goes UP when memory-bandwidth-bound (verify tokens ride nearly free under the weight-read stream) and DOWN as slots fill toward the wall; their **MoE non-monotonicity warning is exactly us** (extra draft tokens pull extra experts per round at low occupancy). Protocol: sweep K ∈ {2..6} × cohort batch shape × measured acceptance (88-94% is the high-acceptance regime where SpecServe's adaptive drafter gains +6-55% live); acceptance telemetry already in-tree makes the controller cheap. CUDA-graph dimension: capture K as a graph dimension like Cohere's vLLM PR (our lazy per-cohort capture extends naturally). Expected: mid-teens % decode-speed class at busy, plus protection against SL-fixed regression (fixed-SL fell below 1× in up to 40% of setups). Risk: bounded — cap K at high occupancy, fallback to K=1 under burst (SpecServe's AR-fallback).

**8. SLO-adaptive speculation + verify pruning (SpecServe confidence-prior verifier §5.2).**
Price: engine bench. Verify is where wasted work lives (up to 90% of SD time per SpecServe); pruning draft tokens with low posterior mass before verify trims wasted expert reads — directly relevant when a round's expert set is the scarce resource. Pairs with #7 in the same bench harness.

### Tier 3 — context-dependent / capacity-bound

**9. CacheGen-class compression of idle prefixes (§6.4).** Price: engine change (KV encode/decode in engine memory format). Only pays if VRAM KV capacity (12 GB, 6 slots × 150-175K ctx) is the binding constraint — it likely is; 8-bit idle-prefix residency would ~double effective cache capacity. Bench: hit-rate vs quality vs (de)compress latency; our prefix reads must stay off the decode critical path.

**10. CacheBlend selective recompute for edited/branched conversations (§6.4).** Price: engine change (cross-chunk attention correction). Unlocks TTFT wins on mid-context edits/regenerations (non-prefix reuse) — value depends on how much live traffic edits mid-context; instrument first, bench second.

**11. DSDE KLD/WVIR dynamic SL controller (§5.4).** Price: engine-level; adopt only if #7's lookup-table approach proves too static across cohorts — DSDE's value is avoiding per-cohort profiling, which we may not need given our telemetry.

### DEAD here (respecting falsified list + box)
- **PDMux/MuxWise, DuetServe** (§1.5) — SM-partitioning/kernel-scheduling infra we don't own; also redundant: our P and D already run on disjoint resources (GPU compute vs CPU DRAM lane), with DRAM arbitration owned by the cadence throttle.
- **DistServe/Splitwise PD-disaggregation, Mooncake transfer-tier, llm-d routing** — ≥2 GPUs / cluster.
- **DualPace Dense-Assist numerics** (§1.6) — needs AMX (Sapphire Rapids); AM5 lacks it. The scheduling-authority split (no global iteration barrier, finish-peak footprint admission, independent per-engine clocks) transfers cheaply and is worth taking: our two engines and 6 slots should not share one global barrier.
- **Marconi** (§6.5) — SSM-hybrid models only.
- **SWAP-based preemption** (§2.1) — with 150-175K contexts, recompute ≈ 24-28 s either way; the correct policy on this box is *avoidance* (token-window + two-tier admission) and, if forced, preempt-newest-smallest-prefix, never the largest context.
- **Quantization-fallback under load** — quality risk with no literature support found; context trim + smaller-engine routing are the supported degradation ladder (§4.3).
