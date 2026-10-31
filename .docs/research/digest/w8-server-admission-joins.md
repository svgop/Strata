---
type: digest
id: w8-server-scout
created: 2026-10-31
status: evidence
scope: W8 server lane — how leading serving stacks admit a new request into a running decode batch
method: primary sources (paper full text, PR thread, scheduler code, official tuning guides); contradictions re-fetched neutral. Unproductive: "PR #12176 = overlap scheduler" (unverified, dropped); arXiv 2206.01834 is a physics paper, not ORCA.
---

# W8 — Server admission & joins: competitor evidence

## vLLM v1 (confidence: HIGH)
- **Join mechanism:** PagedAttention KV — every request holds its own page table into one shared GPU block pool; admission = allocate blocks + append scheduler state. No park, no copy, no STOP. V1 blog: scheduler "removes the traditional distinction between 'prefill' and 'decode' phases"; per-step state is `{request_id: num_tokens}`.
- **Chunk policy** (`vllm/v1/core/sched/scheduler.py`): step budget `max_num_batched_tokens` (default 8192 per V1 docs/discuss); `self.running` scheduled first, then waiting; chunk = `min(num_new_tokens, token_budget, input_budget)`; `long_prefill_token_threshold` caps a chunk only when >1 request is eligible (anti-starvation — sole request takes whole budget); multiple chunked prefills per step; FCFS deliberately non-strict (`continue` not `break`).
- **PR #44558 "Add prefill step cadence for better non-PD DP balancing"** (njhill, merged 2026-06-16, commit d8d9599): admission throttle to every Nth step (N=2 optimal) + parameter-free capacity-bound saturation guard (throttle suppressed when waiting queue not drained). DeepSeek-R1-0528 FP4, DP=8, 2xGB200, 24 req/s, 1k in/200 out: **P50 ITL 274.9→32.1 ms (−88%)**, mean TTFT 813→502, P99 TTFT 2476→1475, throughput flat (22.43→22.59). Sweep: N=3 gives ITL 26.7 ms but P99 TTFT 630→3175 ms (U-curve; N=2 is the sweet spot). 16k-prefill-heavy workload: no ITL win, TTFT regresses (3034→3233 ms) — win is decode-bound sub-saturation only. Revert PR #45882 closed; still in main; HPU scheduler broke on the new `throttle_prefills` arg (fixed #45897).
- **Async scheduling:** default since v0.14.0 ("Async scheduling is now the default. Disable with `--no-async-scheduling`"); "helps avoid gaps in GPU utilization".
- Sources: [V1 blog](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) · [PR #44558](https://github.com/vllm-project/vllm/pull/44558) · [scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) · [engine args](https://docs.vllm.ai/en/stable/configuration/engine_args)

## Sarathi-Serve (OSDI'24) (confidence: HIGH)
- **Join:** stall-free batching (Alg. 3) — per iteration: decode tokens first, then next chunk of in-flight prefills, then new admits with "the maximum chunk size that can be accommodated within the leftover token budget"; "new requests join a running batch without pausing ongoing decodes".
- **Chunk policy:** token budget τ = max tokens per batch without violating TBT SLO (one-time profiled); τ=512 strict / 2048 relaxed (1536 for LLaMA2-70B); chunk should be a GPU-tile multiple — 257 vs 256 → +32% prefill time.
- **Math:** no closed-form ITL/TTFT formulas in the paper. Budget bounds per-iteration latency → TBT "nearly independent of the total length of input prompts"; TTFT ≈ ceil(prompt/τ_eff) windows × window time + queue. Unchunked hybrid join raises TBT "by up to 28.3×" vs decode-only.
- **Numbers:** up to 2.6x (Mistral-7B, 1x A100) and **6.9x (Falcon-180B, 8x A100)** vs Orca/vLLM; 3.2x vs Orca, 2.7x vs vLLM (Yi-34B, 2x A100). Row's "6.3x" is not in the paper.
- Sources: [arXiv 2403.02310](https://arxiv.org/html/2403.02310v1) · [USENIX OSDI'24](https://www.usenix.org/conference/osdi24/presentation/agrawal)

## SGLang (confidence: HIGH scheduler / MEDIUM join-timing detail)
- **Join:** radix-tree paged KV pool shared by all sequences; new request is prefix-matched, pages allocated, enters at the next batch preparation. Overlap scheduler runs CPU one batch ahead (batch N+1 metadata prepared during batch N's GPU compute) → an arrival waits ≤1 batch for its prep slot; no KV park.
- **Chunk policy:** `--chunked-prefill-size` default ≈4096 (version-dependent; −1 disables); `--enable-mixed-chunk` mixes decode tokens into prefill steps.
- **Numbers:** overlap scheduler "a 1.1x speedup against its previous version and a 1.3x speedup against other state-of-the-art baselines" (Llama-3.2-3B, 500 prompts, 4096/2048; Nsight: "no single idle time on the GPU"); cache-aware router: "up to 1.9x throughput increase and 3.8x hit rate improvement" (82,665→158,596 tok/s; hit 20%→75%; 8xA100).
- Sources: [LMSYS v0.4 blog](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4) · [server args](https://sgl-project.github.io/advanced_features/server_arguments.html) · [issue #19347](https://github.com/sgl-project/sglang/issues/19347)

## ORCA (OSDI'22) (confidence: MEDIUM-HIGH)
- **Join — the origin:** iteration-level scheduling admits/retires at any decode-iteration boundary (abstract attacks rigid batching: finished requests "cannot return to the client" until batch end). Selective batching: batch only GEMM-safe ops; attention per-request.
- **Join cost:** no chunking — prompt processed at admission in co-scheduled iterations; the unchunked-hybrid class was later measured (Sarathi) at up to 28.3x TBT blowup. Same cost class as W8's STOP+drain+park.
- **Numbers:** "36.9× throughput improvement at the same level of latency" vs NVIDIA FasterTransformer (GPT-3 175B).
- **Correction:** ORCA has no arXiv abs at 2206.01834 — that ID is a hep-ph physics paper (verified twice); cite USENIX pp.521-538.
- Source: [USENIX OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu)

## TensorRT-LLM (confidence: HIGH)
- **Join:** in-flight batching; generation-phase requests scheduled first ("TRT-LLM prioritizes scheduling requests in generation phase first"); a context request needs its full prompt to fit the remaining step budget unless chunked context is on ("context phase of a request to be broken up into pieces and processed over several execution iterations"). Paged KV; no park.
- **Chunk policy:** `max_num_tokens` (default 8192) = shared per-step token budget; `max_batch_size` (default 2048) = slot cap; chunked context "recommended that you always enable it"; tune powers of 2 ≥1024, grid-search both; case study: max_batch_size 64→512 ≈ +20% throughput.
- Sources: [tuning guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.md) · [NVIDIA blog](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill)

## Mooncake — prefix-cache-guided admission (confidence: HIGH, abstract-level)
- **Join:** request steered to the replica holding its longest cached KV prefix (disaggregated global KVCache over RDMA); admission minimizes recomputation.
- **Numbers:** "up to a 525% increase in throughput in certain simulated scenarios while adhering to SLOs"; Kimi handles "+75% more requests" (production); v5 current 2026-10.
- Source: [arXiv 2407.00079](https://arxiv.org/abs/2407.00079)

## Corrections to the row
- Sarathi "6.3x" → paper max is 6.9x (Falcon-180B); Mistral-7B single A100 = 2.6x.
- PR #44558 is real (merged 2026-06-16) but is a cadence *throttle* (prefill every Nth step, N=2, saturation guard), not plain interleave — interleave-every-step is exactly the baseline that leaves P50 ITL at ~275 ms.
- ORCA join cost rests on Sarathi's unchunked-hybrid measurement (28.3x), not an ORCA-paper formula.

## What this changes on the row
- Cadence ≠ plain interleave: shipped upstream (vLLM #44558) runs prefill reads every 2nd window with a saturation guard — interleave-every-step is the baseline that keeps P50 ITL at ~275 ms. Turn the flag on at N=2 and add the guard.
- Round-robin reads = vLLM v1 budget scheduler: decodes first, leftover budget to chunks, chunk = min(budget_left, long_prefill_threshold) where the cap applies only when >1 admission waits; multiple chunked prefills per step allowed.
- PROMOTE at window boundary without park/STOP is how all six stacks admit (paged KV, same pool): join cost = one window boundary, not 80-150 ms; full-KV park-to-host per join has no analog anywhere upstream.
- Budget the window: ITL ≈ window time; target TTFT ≈ ceil(prompt/chunk) windows + queue (Sarathi). Keep chunks on GPU-tile multiples — 257 vs 256 cost +32% prefill.
- Replace the 1 s poll with SGLang-style one-batch-ahead prep + async default (vLLM v0.14): build next window's admission metadata while the current window runs on GPU.
- Add prefix-cache-guided admission (Mooncake/SGLang router): route joins to cached prefixes — up to 1.9x throughput / 3.8x hit rate before any prefill work is spent.

## Source count
Code/PRs 3 · Docs/specs 6 · Literature 3 · Practitioner/community 2 — 14 sources, 4 categories. Residual unknown: vLLM v1 async-scheduling PR number (blog cites it; number unverified — does not affect conclusions).
