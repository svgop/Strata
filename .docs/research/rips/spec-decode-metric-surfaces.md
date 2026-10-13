# RIPS — verbatim metric / flag / API surfaces captured from comparables

Every line below is an identifier copied from the source named beside it, not a paraphrase. Captured
2026-10-12 (goal round 34) while researching the draft-yield / acceptance-metric decision. These are the
strings to grep for when re-verifying any claim in
[the digest](../../digest/speculative-draft-yield-and-acceptance-metrics.md).

## vLLM
- Counters: `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`, `vllm:spec_decode_num_accepted_tokens`, `vllm:spec_decode_num_accepted_tokens_per_pos{position="N"}`
- `mean_acceptance_length` = `1 + (num_accepted_tokens / num_drafts)`; documented range 1.0 … k+1
- `_acceptance_length_to_rates(length, n)` — "the minimum-variance schedule" (`vllm/config/speculative.py`)
- `num_speculative_tokens_per_batch_size` entries `(range_start, range_end, num_speculative_tokens)`
- `enable_adaptive_verification` (opt-in; `method="dspark"`), `enable_adaptive_k`, `disable_eagle_block_drop` ("for measuring the acceptance-rate impact")
- `--per-request-spec-decode-metrics detailed` → `acceptance_histogram`, per-step arrays
- PR #44336 (RTX 4050 + Qwen2.5-7B): fixed K=5 `75.2 tok/s @ length 1.88`; adaptive K `75.9 tok/s @ 1.86`; oracle K=12 `60.8 tok/s @ 2.27`
- `prompt_lookup_min` / `prompt_lookup_max`; suffix decoding `min_token_prob 0.1`

## SGLang
- `sglang:spec_accept_length` ("accepted drafts + bonus token per forward"), `sglang:spec_accept_rate` ("accepted drafts / proposed drafts in batch")
- `sglang:spec_cap_length`, `sglang:spec_block_accept_length`, `sglang:spec_num_steps`, `sglang:spec_num_draft_tokens`
- `--speculative-adaptive`, `AdaptiveSpeculativeParams` (EMA), `SpecRuntimeState`, `AdaptiveController`, `candidate_steps`
- `target_steps ≈ clamp(round(ema_accept_len) + 1, min, max)`

## NVIDIA TensorRT-LLM
- `trtllm-bench` "DECODING STATISTICS": `num_draft_tokens_percentiles [DT]`, `num_accepted_draft_tokens_percentiles [ADT]`, `draft_acceptance_rate_percentiles [DAR]`, `acceptance_length_percentiles [AL]` — each with min/max/avg/p50/p90/p95/p99
- `acceptance_length` = `sum(output_tokens) / total_decoding_iterations`
- `max_draft_len`, `max_total_draft_tokens`, `dynamic_tree_max_topK`, `num_nextn_predict_layers`, `max_matching_ngram_size`
- `draft_len_schedule = {100: 4, 200: 3, 300: 2}`; `spec_decode_algo=AUTO`
- `use_relaxed_acceptance_for_thinking`, `relaxed_topk`, `relaxed_delta` — "We remove tokens t for which log(P(top 1 token)) - log(P(t)) > relaxed_delta"
- Docs: "speed ups are only observable at low batch sizes"; "currently no way to dynamically disable speculation"
- Qwen3.8-Flash-Next deployment guide caps `max_draft_len` at **3**; GSM8K 95.11 (MTP3) vs 95.72 (BF16); "a rate measured on one workload is not a model constant"

## llama.cpp
- `--spec-draft-n-max` (default **3**), `--spec-draft-n-min` (default **0**), `--spec-draft-p-min` / `--draft-p-min` (default **0.00 = disabled**), `--spec-draft-conf-min` (DSpark, default 0)
- Console: `draft acceptance rate = 0.57576 (  171 accepted /   297 generated)`
- Prometheus: `llamacpp:spec_decode_num_draft_tokens_total`, `llamacpp:spec_decode_num_accepted_tokens_total`, `llamacpp:spec_decode_num_drafts_total`, `llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`
- Discussion #25198: recommends p-min 0.8 with n-max 16, accepting `acceptance 67.8% → 55.7%` for `+20.4% TPS` (avg draft len 3.71 → 5.20); a 2026-09-23 comment reports 0.70 was 17–21% **slower** on reasoning text; another landed on 0.6

## ExLlamaV2
- `speculative_prob_threshold = 0.25` **hard-coded** (`exllamav2/generator/streaming.py`)
- `num_speculative_tokens = 5` (streaming) vs `num_draft_tokens = 4` (dynamic — argmax, no floor)
- `get_sd_stats()` → (`efficiency` = accepted/total_tokens, `accuracy` = accepted/total_draft_tokens, …) — **our "acceptance rate" is their `accuracy`, our "tokens/window" is their `efficiency`**

## LMDeploy
- `lmdeploy:spec_decode_mean_accept_rate`, `lmdeploy:spec_decode_mean_accept_length` ("including the bonus token")
- `SpeculativeDecodingStats` prints draft_acceptance_rate %, mean_acceptance_length, per-position rates
- `--speculative-algorithm` (default None), `--speculative-num-draft-tokens` (default 1), `--speculative-dflash-block-size` (default None)

## MLC-LLM
- `EngineMetrics.spec_decode`: `draft_count[]`, `accept_count[]`, `accept_prob{step=i}`, `accept_rate{step=i}`, `accept_len{step=i}`
- `--speculative-mode` (default `disable`), `spec_draft_length` (default **0 = adaptive**), `spec_tree_width` (default 1)
- `auto_spec_decode.cc` comment: "Right now we use the fixed table … we will follow up to adopt powerful draft length selection" (table 4/3/2/0 by batch size)

## HuggingFace TGI
- The **entire** speculation telemetry: `tgi_request_skipped_tokens` — "Speculated tokens per request" (Histogram)
- `--speculate <N>` — "number of input_ids to speculate on"; Medusa auto-detected from the checkpoint, else n-gram

## Medusa / EAGLE / Hydra / Lookahead / REST (papers)
- Medusa co-threshold: `min(ε, δ · exp(−H(p)))`; head count "five heads are sufficient at most … sometimes three or four may be enough"; "we ascertain that it is typically unnecessary to match the distribution of the original model"
- EAGLE-2: "**Why is acceptance rate not included? The acceptance rate only reflects the performance of the draft model.**"
- EAGLE-1: acceptance rate "is less applicable for tree drafts due to multiple tokens sampled per location with only one accepted"
- EAGLE-1 definition of τ: "The average number of tokens accepted per forward pass of the target LLM"; EAGLE-3 reports per-position `0-α … 4-α`
- EAGLE-3 Table 1 ranks SpS, PLD, Medusa, Lookahead, Hydra, EAGLE, EAGLE-2, EAGLE-3 by speedup and τ with no acceptance-rate column
- Hydra: "average throughput, which is the number of tokens generated per second" + "average acceptance length, which is the number of tokens generated per decoding step"; posterior threshold sweep ε ∈ {0.05, 0.1, 0.15, 0.2, 0.25}
- Lookahead: "their speedups are bounded by the token acceptance rate"; per-model operating points 7B `W=15,N=5`, 13B `W=10,N=5`, 33B `W=7,N=5`
- REST: "Mean Token Time (ms/token)" + "Speedup (×)"; CodeLlama-7B HumanEval 27.89 → 11.82 ms/token = 2.36×

## Model-side MTP numbers
- DeepSeek-V3 report §2.2: second-token acceptance **85–90%** at depth 1 → **1.8× TPS**
- FastMTP (arXiv 2509.18362): vanilla head reuse **70% / 11% / 2%** at depths 1/2/3 → **81% / 56% / 36%** after multi-step fine-tuning, 2.03×
- vLLM on DeepSeek-V4-Pro-0813, 8×B300 TP=8: per-position survival **>70% at position 1, <10% at position 7** of a 7-token block
- ReDrafter (arXiv 2403.09919): H100 Vicuna 7B/13B/33B **2.80× / 2.80× / 2.61×** at **4.20 / 4.21 / 3.87 tokens per step**; M2 Ultra beam width 1→3 raises tokens/step 2.15→2.44 and speedup 1.43×→1.52×, but width 4 keeps 2.44 tokens/step at **0.75×**
- TRT-LLM nGram on Llama-4-Scout: accepted length 1.37/1.40 (turn 1), 1.66/1.77 (turn 2); batch 1–4 best `k=3,v=5`, batch 32 best `k=5,v=3`
