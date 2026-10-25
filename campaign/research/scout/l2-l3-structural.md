# SCOUT DIGEST — L2 graph surgery + L3 window-fused CPU tier
Row: L2/L3 structural | Date: 2026-10-25 | Sources: actual code/PRs/docs, papers only where noted.

## 1. vLLM v1 full CUDA-graph mode (PIECEWISE → FULL_AND_PIECEWISE)

**Code:** `vllm/config/cuda.py` (`CUDAGraphMode`: NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE), `vllm/v1/cudagraph_dispatcher.py` (dispatch = the "only source of truth" for graph keys), `vllm/compilation/cuda_graph.py` (`CUDAGraphWrapper`), design doc `docs/design/cuda_graphs.md`. Core PR: vllm-project/vllm#20059 ("Allow full cudagraph with separate attention"); FlashInfer follow-up #21367.

**Structural trick:** Nested wrappers — a FULL wrapper wraps the *whole model*, PIECEWISE wrappers wrap each compiled piece *inside*. Dispatch key = `BatchDescriptor(num_tokens, num_reqs, uniform, has_lora)`; dispatch priority FULL > PIECEWISE > eager. Per-backend support enum `AttentionCGSupport`: ALWAYS > UNIFORM_BATCH > UNIFORM_SINGLE_TOKEN_DECODE > NEVER; hybrid models auto-downgrade to the *minimum* capability instead of disabling graphs entirely. Uniform-decode batches reuse the non-uniform graph of the same batch size. Their first full-graph attempt (flattened FX graph, everything fused into one runnable) cost ~56 ms CPU launch vs ~28 ms piecewise — killed; wrapper nesting preserves eager fallback paths and captures outside the compiled fx graph.

**Measured:** A100/40G, Qwen2.5-7B-GPTQ-Int4: full-graph +5% output tput (1898→1992 tok/s), TPOT −2.9% vs piecewise. Triton attention: ITL ~−38% for small LLMs in FULL_AND_PIECEWISE vs PIECEWISE.

**Failure modes:** capture-under-flight handled by warming up in NONE mode with attention explicitly executed before capture; FA2 capture bug (zero seq_lens tensor bypassed kernels during capture — the op was never "incompatible", it was a capture bug); mixed-batch FULL capture was pathological ("494 blocks per sm" vs ~4.6 eager) so mixed batches fall back to piecewise; duplicate graph sets cost memory + startup; backends not cudagraph-safe force mode downgrade.

**Grade: A.** The dispatcher/gate architecture is the direct template for arming multiple fusion gates as one coherent capture mode.

## 2. llama.cpp CUDA graphs (#6766 + 2026 follow-ups)

**Code:** `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_backend_cuda_graph_compute` (+ `cpy.cu` helper). Issue #6763, PR #6766 (merged 2024-05); follow-ups #7302, #6933, #19521/#19375/#19566 (qwen3-next, 2026), #27721 (Pascal re-enable, Aug 2026). Build guard `GGML_CUDA_USE_GRAPHS`; kill-switch `GGML_CUDA_DISABLE_GRAPHS=1`.

**Structural trick:** No fusion at all. Capture once, then per token patch params in place (`cudaGraphKernelNodeGetParams`/`SetParams`, `cudaGraphExecUpdate`) and relaunch — never recapture unless node properties (shapes, strides, src pointers) change (`can_reuse` compares them). Gain is purely GPU-side inter-kernel gap removal: "CPU-side launch overhead was already well hidden." Benchmark log shows **graph nodes = 1030** for a 7B dense model — essentially our ~1,000-node window; 2026 PRs cite ~700 nodes for dense models.

**Measured (consumer GPUs, tg128 b=1):** RTX 3090 llama-2-7B q4_0 +10% (116.3→128.3); RTX 4090 +6% (llama-3-8B); f16 only +2%; H100 +14%; graph exec itself ~40% faster. #27721: re-enabled on Pascal sm_61, ~+40% tg on a MoE model, +6–7% dense on GTX 1080, bit-identical outputs. #19521 (2026, qwen3-next MoE): ppl pass 4.09 s vs 5.61 s with/without graphs, identical PPL 8.3729.

**Failure modes:** multi-GPU `cudaGraphInstantiate` "operation not permitted" (6×4090), illegal memory access on 3×P40 → graphs disabled for multi-GPU/batch>1; Pascal/Volta regressions (until #27721); ROCm neutral, excluded; fused-ADD mutation bug (`node->src[2]->data` mismatch, spun off to #19566); #7302 — consecutive-update counter needlessly disabled graphs on long prompts; the batch>1 disable heuristic is an acknowledged band-aid; ggml graph regeneration per token remains a real cost. Slaren's stated endgame: `graph_plan` + `graph_plan_update` in ggml-backend. Related: #6933 moved CPU memsets off the critical path (+3%).

**Grade: A.** Same code lineage as ours; the exec-update-in-place pattern and the 1030-node datapoint calibrate exactly what fusion must remove.

## 3. TensorRT-LLM graph capture + PDL

**Code/docs:** blog `docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes...` (GB200, TRT-LLM v1.3.0rc8); PDL via `cudaLaunchAttributeProgrammaticStreamSerialization`, legal *inside* graphs since CUDA 12.3 (graph edge data with PDL semantics); FlashInfer's TRT-LLM-derived MoE kernels (`trtllm_fp8/fp4_block_scale_moe`) expose `enable_pdl`, default auto-ON for SM90+, default True in `flashinfer.fused_moe`.

**Structural trick:** two layers of surgery. (a) Fixed capture-size ladder with padding: default x2 = 23 sizes (1–128 fine, then 256/512/1024/2048); new +64 default = 49 graphs — captures more sizes instead of paying pad waste ("up to 50% padding waste in high-concurrency"). (b) PDL edges let kernel n+1's prologue overlap kernel n's epilogue — this attacks exactly the residual inter-node spacing that plain graphs leave behind.

**Measured:** +64 config: up to 1.3× TPS aggregated, 1.5× DeepSeek-R1 disaggregated at concurrency ~500–600. Memory: ~10 MB per graph (~260 MB for 49); +8 config (259 graphs) needs 2.6–4.9 GB → cuts DeepSeek-V3.2 KV cache 17.2% → net −5% regression. Kimi-K2 gained least (graph memory evicted KV). Capture-phase startup 2.4×.

**Condition nodes:** no production TRT-LLM condition-node usage found in public code; their PDL-in-graph is the shipped trick. CUDA 12.4 added IF/WHILE conditional nodes, 12.8 added IF/ELSE + SWITCH (Blackwell) — see §6.

**Grade: A−.** PDL is the highest-leverage trick for our 9.5 µs → 1.3 µs spacing goal; the capture-size memory-budget failure mode is directly transferable.

## 4. KTransformers CPU-GPU async overlap (CPUInfer)

**Code:** current: `kt-kernel/` (binary-ext + `kt_kernel_ext.CPUInfer(n)`); readable lineage in `archive/csrc/ktransformers_ext/cpu_backend/{cpuinfer.h,task_queue.h,backend.cpp}` and `archive/csrc/ktransformers_ext/operators/{amx/llamafile}/moe.*`; Python hooks in `ktransformers/operators/cpuinfer.py`.

**Structural trick (how they avoid per-layer host barriers):** (a) `submit_with_cuda_stream` uses `cudaLaunchHostFunc(stream, fn, args)` so CPU expert work is *stream-ordered* after the GPU kernels that produce its inputs — no host polling of events, no blocking join per layer; `sync_with_cuda_stream` enqueues the drain callback onto the stream, so sync is also stream-ordered. (b) Inside one MoE layer, `moe.cpp` partitions into `nth × k` tasks (`forward_one`/`forward_many` split by `group_min_len`) pushed to a work-stealing backend (`do_work_stealing_job`) — the fork/join stays *inside* the CPU pool, never round-trips to the scheduler. (c) `--kt-max-deferred-experts-per-token` (0–7, 0=synchronous): explicitly defers cold-expert results to overlap CPU batch n with GPU batch n+1 (5–7 risks accuracy). (d) NUMA: `--kt-threadpool-count` = number of NUMA nodes, separate pools per domain; `Backend(thread_num-1)` reserves the calling thread; workers = physical cores only (HT hurts); weights `numa_alloc_onnode`-replicated per gate projection.

**Honest caveat vs our plan:** their `TaskQueue` is mutex+condvar with a single consumer thread and `std::atomic<bool> sync_flag` — wake-latency on every drain, not a persistent spin-poller. Our "persistent polling workers + whole-window sync" is one step beyond KTransformers: keep workers parked spinning on an atomic task index, sync once per window instead of condvar wake per layer.

**Measured:** 3–28× vs llama.cpp CPU-offload on 24 GB GPU + DRAM; DeepSeek-R1 FP8 8×L20: 227.85 tok/s total / 87.58 output @ 8-way; Qwen3-Next-30B-A3B 8+ it/s on one 4090.

**Grade: B+.** Right submission mechanics (stream-ordered host launch, in-pool work stealing, deferred experts); their sync hygiene (condvar) is the documented gap we exploit.

## 5. HybriMoE (DAC'25, PKU-SEC-Lab) + APEX

**Code:** github.com/PKU-SEC-Lab/HybriMoE — a ktransformers fork (same `cpu_backend/{CPUInfer,TaskQueue,backend}` + work-stealing MoE operator); scheduler-level additions: `--cache_size` (expert-score kernel-level cache) and `--prefetch_size` (impact-driven prefetch of next-layer experts on idle cores). Paper: arXiv:2504.05897.

**Structural trick:** dynamic CPU-GPU task scheduling so neither device idles during the other's phase; prefetch prioritized by *simulated impact* of preloading experts from subsequent layers, not by naive next-layer order; expert-score caching to minimize misses. Delayed-result-sync: results from prefetched/deferred experts are consumed one layer later — the fork/join is amortized across layers.

**Honest caveat:** public repo is thin (6 commits, no explicit prefetch stream/lookahead code visible in the operator; mechanics live in the paper). No separate public "APEX" repo found. **Grade: C+.** Design validation only; do not copy code from here — KTransformers' archive is the real reference implementation.

## 6. Device-side conditional graph launches (2025–2026 production)

**Primitive:** CUDA 12.4 IF/WHILE nodes (`cudaGraphConditionalHandleCreate`, device-side `cudaGraphSetConditional`), CUDA 12.8 adds IF/ELSE and SWITCH (Blackwell); body graphs populated by capture via `cudaStreamBeginCaptureToGraph` (12.3). NVIDIA sample `graphConditionalNodes.cu`; tutorial repo Thespica/conditional-cuda-graph.

**Production reality check:** No engine ships MoE routing as conditional nodes yet. Closest production-grade device-side pattern is vLLM RFC #39701 (Apr 2026, "routing replay CUDA-graph-compatible"): persistent per-layer device buffers attached as module attributes (torch.compile captures by reference), `cudagraph_mark_tensor_static` to stop snapshot/restore clobbering, async D2H via events + pinned memory — <5% overhead on GB200 across 9 configs, GSM8K unchanged. SGLang's Breakable CUDA Graphs (#19102 Feb 2026, #22218, #27988; runner/backend refactor #23906; library `meta-pytorch/breakable-cuda-graphs`) is the 2026 default for capture-with-holes: `@eager_on_graph` ops break capture, output copied into a *persistent boundary buffer* so the next segment replays against a fixed address; one shared graph mempool per shape; weak refs keep eager outputs alive (technique from vLLM #9724). Measured: prefill gpt-oss-120b TP4/GB300 — full 1.93×, BCG 1.70×, TC-piecewise 1.45× vs eager; GLM-5.2 only BCG captures (1.60×); BCG build 3.8–5.2× faster than compiler piecewise; 42 shapes on a 78-layer MoE ≈ 2.4 GB graph memory, but capturing through `chunked_prefill_size` cuts the eager activation peak so total lands *below* no-graph baseline. Failure modes: batches exceeding slot count fall back eager; sentinel metadata must be rewritten every replay; BCG gains shrink when compute-bound. "Blink-style": no project of that name found in this space — treat the reference as unverified; the mechanisms above are the real shipped equivalents.

**Grade: B (BCG/RFC #39701: B+).** BCG's break+boundary-buffer is exactly our "CPU-expert gate as a graph seam" pattern.

## TRICK SHEET (trick → our in-tree analog site)

| # | Trick (source) | Our analog site / use |
|---|---|---|
| T1 | Capture once + param patch in place, node-property `can_reuse` (llama.cpp #6766) | Decode-window graph: patch per-window params, never rebuild; ban recapture on hot path |
| T2 | FULL/PIECEWISE dual-mode dispatch, auto-downgrade by backend support (vLLM #20059) | Gate dispatcher: arm shared-expert-fuse + fused-wait + router/hc-fold as ONE capture mode, per-gate fallback to piecewise seam, never all-or-nothing |
| T3 | PDL edges inside graph, auto-on SM90+ (TRT-LLM/FlashInfer `enable_pdl`) | Adjacent-kernel edges in decode window: 9.5 µs spacing → ~1.3 µs; gate on cc≥9.0, capture the attribute into the graph |
| T4 | Breakable graph: eager break + persistent boundary buffer + shared mempool (SGLang #19102) | CPU-expert fork/join sites become graph seams with fixed boundary buffers — removes 48 host round-trips |
| T5 | Stream-ordered host submission `cudaLaunchHostFunc` + in-pool work stealing + deferred experts (KTransformers) | Replace per-layer fork/join with stream-ordered submit + `max_deferred`-style overlap; whole-window single sync |
| T6 | Persistent spin-poll workers (our L3 design; KTransformers uses condvar — the gap) | Worker pool: parked spin on atomic index, per-NUMA pools, physical cores only |
| T7 | `mark_tensor_static` + attribute-by-reference + event-driven D2H (vLLM RFC #39701) | Router output + expert-ID buffers: static-marked, one window-level D2H, no per-layer copies |
| T8 | Capture-size ladder memory budget: 10 MB/graph; over-capture → KV eviction → net regression (TRT-LLM blog20) | Budget graph pool vs KV cache before adding window shapes; cap captured window lengths |
| T9 | Move CPU memsets/prep off critical path (+3%, llama.cpp #6933) | Host-side prep outside window; warm-up runs eager-with-attention before capture (vLLM capture-under-flight rule) |
| T10 | Conditional IF/ELSE/SWITCH nodes (CUDA 12.4/12.8) | Future: route dispatch inside window graph; requires CUDA≥12.8 + Blackwell for SWITCH; NOT in first tranche |

## OUR NEXT STEP — gate-flip tranche (exact plan)

**Phase 0 (instrument):** nsys the current window: confirm ~1,000 nodes, measure per-edge gap (expect ~9.5 µs where healthy is ~1.3 µs). llama.cpp's 7B graph was also 1,030 nodes — our count is consistent with "zero structural fusion, one node per op."

**Phase 1 (L2 fusion-arm, gates together):** flip the three gates simultaneously under a single capture mode (vLLM-style dispatcher, T2): (a) shared-expert fusion — shared expert GEMM+act merged into the routed path's launch cluster; (b) fused waits — merge the wait/flag-check nodes; (c) router/hc fold — router softmax + top-k + hc fold into one kernel. Per-layer critical path ≈ 15 nodes today; these folds remove ~5–7 nodes/layer → **~1,000 → ~600–650 nodes**. Bit-check: token-identical + max-ulp logits vs eager path (llama.cpp #19521 standard: identical PPL to 8.3729-level precision; vLLM standard: gsm8k unchanged).

**Phase 2 (spacing):** enable PDL on every intra-window kernel edge on SM90+ (T3). Target spacing 9.5 µs → ≤1.5 µs; window time should drop near-linearly since we are launch/gap-bound (llama.cpp: graph exec alone −40%; TRT-LLM: PDL default-on for MoE kernels). Keep GGML_CUDA_DISABLE_GRAPHS-style escape env.

**Phase 3 (L3 window-fused CPU tier):** replace 48 fork/joins with (a) T4 seams: fixed boundary buffers at the 4–6 genuine CPU-expert gates; (b) T5 stream-ordered submission; (c) T6 persistent polling workers (per-NUMA pool, physical cores); (d) one whole-window sync. Expected node delta: the 48 fork/join node clusters (~5–8 nodes each ≈ 250–350 nodes) collapse to ~6 seams → **~600 → ~350–400 critical-path nodes**. Bandwidth target: 26 GB/s → 60–80 GB/s of the 80–100 available (persistent workers remove wake latency; window sync removes per-layer drains; HybriMoE impact-driven prefetch orders copies into the idle phase).

**Phase 4 (budget + guards):** graph-pool memory audit (T8): with 6–8 window shapes expect ~60–100 MB — verify no KV eviction (TRT-LLM's −5% lesson); capture-under-flight rule: warm-up in eager with attention run once before capture; multi-GPU and batch>1 keep the llama.cpp disable path until validated.

**Expected end state:** ~350–400 nodes at ~1.3–1.5 µs spacing + one sync per window; i.e., ~3–4× window-time reduction vs 1,000 × 9.5 µs, minus remaining dense GEMM time.

**Key sources:** [vLLM CUDA graphs design](https://docs.vllm.ai/en/stable/design/cuda_graphs), [vLLM PR #20059](https://github.com/vllm-project/vllm/pull/20059), [llama.cpp #6766](https://github.com/ggml-org/llama.cpp/pull/6766), [#6763](https://github.com/ggml-org/llama.cpp/pull/6763), [#19521](https://github.com/ggml-org/llama.cpp/pull/19521), [#27721](https://github.com/ggml-org/llama.cpp/pull/27721), [TRT-LLM blog20](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.md), [FlashInfer fused_moe PDL](https://docs.flashinfer.ai/api/fused_moe.html), [KTransformers](https://github.com/kvcache-ai/ktransformers) (`archive/csrc/ktransformers_ext/cpu_backend/`), [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE), [NVIDIA conditional nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes), [LMSYS advanced CUDA graphs](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph), [vLLM RFC #39701](https://github.com/vllm-project/vllm/issues/39701).
