# Digest — draft yield, acceptance length, and the metric that should be defended

Topic: how the field defines and gates the speculative-decoding trade that MAGMA v2 faces, and what objective it
optimises. Written 2026-10-12 (goal round 34) for the re-authored survey. Full detail per system:
[`.docs/research/`](../research/) — `serving-engines-spec-metrics.md`, `local-runtimes-draft-gates.md`,
`draft-head-designs-accept-length.md`, `mtp-and-autotuning.md`; our own side in
[`engine-inventory.md`](../research/engine-inventory.md); verbatim identifiers in
[`rips/spec-decode-metric-surfaces.md`](../research/rips/spec-decode-metric-surfaces.md).

## The decision this informs

Lowering our MTP floor `--spec-min-p` 0.7 → 0.5 measured **+8–11% throughput**, **tokens/window 2.2 → 2.5**, and
**acceptance rate 91% → 87%**. Our six-metric law protects the acceptance rate, so the rate's fall vetoes the
change. Eighteen comparables were studied to find out whether that veto is how the field works.

## The answer, in one line

**No surveyed system treats acceptance rate as the objective or as a protected metric; the field's headline is
acceptance *length* — which is exactly our tokens/window — and the rate is published, if at all, as a diagnostic.**

## 1. What each comparable publishes as its objective

| system | headline metric | acceptance rate's role | source |
|---|---|---|---|
| vLLM | `mean_acceptance_length` = `1 + accepted_draft_tokens / num_spec_steps` (docs give the range 1.0 … k+1) | not computed in-engine; documented as a PromQL division | [vLLM acceptance metrics](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/) |
| SGLang | `sglang:spec_accept_length` = "accepted drafts + bonus token per forward" | published alongside (`spec_accept_rate`); steers depth via **length**, not rate | [SGLang adaptive spec](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding) |
| TensorRT-LLM | `acceptance_length` = `sum(output_tokens) / total_decoding_iterations`, as min/max/avg/p50/p90/p95/p99 | an *output* of `trtllm-bench`, never an input gate | [trtllm-bench reporting](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/bench/dataclasses/reporting.py) |
| LMDeploy | `lmdeploy:spec_decode_mean_accept_length` ("including the bonus token") | derived from the same counters, no side taken | [LMDeploy stats](https://raw.githubusercontent.com/InternLM/lmdeploy/main/lmdeploy/metrics/stats.py) |
| MLC-LLM | cumulative `accept_len{step=i}` | per-position `accept_rate{step=i}` is the *diagnostic* for where the chain dies | [MLC auto_spec_decode.cc](https://raw.githubusercontent.com/mlc-ai/mlc-llm/main/cpp/serve/engine_actions/auto_spec_decode.cc) |
| EAGLE-1/2/3 | walltime speedup + average acceptance length τ = "the average number of tokens accepted per forward pass" | **explicitly declined**: "Why is acceptance rate not included? The acceptance rate only reflects the performance of the draft model." | [EAGLE-2 §4](https://arxiv.org/html/2406.16858v2), [EAGLE-1](https://ar5iv.labs.arxiv.org/html/2401.15077) |
| Medusa / Hydra | wall-clock speedup; Hydra: "average throughput" + "average acceptance length" | Hydra sweeps the threshold and scores it on speculation **length and judged quality**, not on the ratio | [Medusa](https://arxiv.org/html/2401.10774v3), [Hydra](https://arxiv.org/html/2402.05109v2) |
| Lookahead / REST | wall-clock speedup, decoding steps / **ms-per-token on named hardware** | Lookahead treats it only as an upper bound on speedup | [Lookahead](https://arxiv.org/html/2402.02057v1), [REST](https://ar5iv.labs.arxiv.org/html/2311.08252) |
| HuggingFace TGI | one metric: `tgi_request_skipped_tokens` ("Speculated tokens per request") | **no acceptance metric exists at all** in a production engine that ships speculation | [TGI metrics](https://huggingface.co/docs/text-generation-inference/main/en/reference/metrics) |
| llama.cpp | console `draft acceptance rate = …`; Prometheus `llamacpp:spec_decode_num_accepted_tokens_total` etc. | published, and the floor that produces it is **disabled by default** (`--spec-draft-p-min 0.00`) | [llama.cpp speculative docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md) |
| ExLlamaV2 | prints tokens/second | names both sides: our rate is its **`accuracy`**, our tokens/window is its **`efficiency`** | [streaming.py](https://raw.githubusercontent.com/turboderp-org/exllamav2/master/exllamav2/generator/streaming.py) |

**EAGLE-3's Table 1 ranks eight methods** (SpS, PLD, Medusa, Lookahead, Hydra, EAGLE, EAGLE-2, EAGLE-3) by
speedup and τ with **no acceptance-rate column at all** ([EAGLE-3](https://arxiv.org/html/2503.01840v3)). A field
that used the rate as a regression gate could not rank its own published results.

## 2. The decisive counter-example — measured, same trade, someone else's hardware

vLLM PR #44336, RTX 4050 + Qwen2.5-7B (via [mtp-and-autotuning.md](../research/mtp-and-autotuning.md)):

| depth policy | throughput | acceptance length |
|---|---|---|
| fixed K=5 | 75.2 tok/s | 1.88 |
| adaptive K | **75.9 tok/s** | 1.86 (lower) |
| oracle K=12 | **60.8 tok/s** (−19%) | **2.27 (highest)** |

**The highest acceptance length is 19% slower than the middle one.** Our situation is the mirror image: our
length and our throughput both RISE together (2.2 → 2.5, +8–11%) and only the ratio falls. On these sources,
**the rate is not a throughput predictor at all.**

ReDrafter supplies the other direction: beam width 1 → 3 raises tokens/step 2.15 → 2.44 and speedup 1.43× → 1.52×,
but beam width 4 keeps tokens/step 2.44 and **collapses speedup to 0.75×** — depth past the operating point is
paid for in wall-clock, not in accuracy.

## 3. Acceptance decaying with depth is the designed shape

- DeepSeek-V3: 85–90% acceptance for the **second** token (depth 1), buying 1.8× TPS ([arXiv 2412.19437 §2.2](https://arxiv.org/abs/2412.19437)).
- FastMTP: 70% / 11% / 2% at depths 1/2/3 with a reused head; 81% / 56% / 36% after multi-step fine-tuning ([arXiv 2509.18362](https://arxiv.org/abs/2509.18362)).
- vLLM on DeepSeek-V4-Pro: per-position survival **>70% at position 1, <10% at position 7** of a 7-token block.

So a blended rate that falls when the chain deepens is arithmetic, not degradation. **Our numbers are consistent
with this and refine it:** our chain tests only ~1.4 drafts per window at the 0.7 floor (927 drafts over ~680
windows) and ~2.0 at 0.5 — the floor, not the head, is what limits our depth today.

**A cross-check that cuts against our M3 as written:** for a *fixed* depth-4 chain the expected banked tokens are
α + α² + α³ + α⁴ ≈ 3.18 at α = 0.91 and 2.86 at α = 0.87 — a **fall**. We measured a **rise** (2.2 → 2.5), which
means our logged rate is a blended ratio over a *variable-length* chain, not the chain's α. It is therefore not
comparable across floor settings, which is precisely the property a non-regression metric must have.

## 4. The floor itself: where 0.5 sits

| system | floor / threshold | default | note |
|---|---|---|---|
| MAGMA v2 | `--spec-min-p` | **0.7** (standing), 0.5 (candidate) | gates extending the verify window |
| llama.cpp | `--spec-draft-p-min` (alias `--draft-p-min`), with `--spec-draft-n-max 3`, `--spec-draft-n-min 0` | **0.00 = disabled** | community landing points: 0.6–0.8; one report has 0.70 as **17–21% slower** on reasoning text; [discussion #25198](https://github.com/ggml-org/llama.cpp/discussions/25198) recommends 0.8 with n-max 16, accepting 67.8% → 55.7% acceptance for **+20.4% TPS** |
| ExLlamaV2 | `speculative_prob_threshold = 0.25` hard-coded | 0.25 | the only floor above zero by default; its dynamic path uses **no floor** |
| Medusa | `min(ε, δ·exp(−H(p)))` — fixed threshold **and** an entropy-scaled term | ε swept | widens automatically on flat distributions |
| Hydra | posterior threshold ε ∈ {0.05, 0.1, 0.15, 0.2, 0.25} | swept | scored on speculation length **and judged quality** |
| vLLM / TRT-LLM / SGLang / TGI | **no probability floor exists** | — | depth is fixed, or keyed to batch size, or (SGLang) adaptively steered by accept length |

**Ours is the highest fixed floor in the survey, and the only one whose project treats its own output ratio as a
regression gate.** The nearest relative is TRT-LLM's `relaxed_topk`/`relaxed_delta` partial-acceptance rule, which
NVIDIA ships explicitly to stop deep chains wasting verify work.

## 5. Auto-tuning the trade — who closes the loop, and on what objective

| system | mechanism | objective |
|---|---|---|
| SGLang | `--speculative-adaptive`: EMA of accept length, per-batch-size trackers, hysteresis, `candidate_steps` ladder | accept-**length** EMA with batch-tier ceilings |
| vLLM | `num_speculative_tokens_per_batch_size` = `(range_start, range_end, num_speculative_tokens)`; `enable_adaptive_verification` | TPOT protection; expected accepted tokens per unit step time |
| TensorRT-LLM | `draft_len_schedule` = `{100:4, 200:3, 300:2}` | min-latency; docs state there is "currently no way to dynamically disable speculation" |
| MLC-LLM | `spec_draft_length 0 = adaptive` → fixed table 4/3/2/0 by batch size | throughput; comment admits the fixed table awaits "powerful draft length selection" |
| **MAGMA v2 (ours)** | `DraftPolicy` learns EMA tokens-per-window and ms-per-window per size and picks by **expected committed tokens per millisecond** — but only to choose MTP **vs lookup**, never the MTP depth itself | already the right objective, wrong scope |

**Every mechanism found optimises throughput or latency. Not one optimises acceptance rate.**

## 6. What we have not done that the field does

1. **Per-position acceptance.** vLLM `…_num_accepted_tokens_per_pos{position}`, llama.cpp `…_per_pos_total{position}`, MLC `accept_rate{step=i}`, EAGLE-3's `0-α … 4-α`. We have `STRATA_SPEC_DEPTH=1` (`generate.cpp:12659`) printing accepted/tested for five depths — **never armed**. Without it we cannot say whether 0.5 rescued position 1 or bought position 4.
2. **Invert a target length into a floor.** vLLM's `_acceptance_length_to_rates(length, n)` derives the per-position rates implied by a desired mean acceptance length. That is the calibration our floor has never had.
3. **Attribute metric moves to policy vs workload.** SGLang publishes the active tier (`spec_num_steps`, `spec_num_draft_tokens`). We cannot currently tell whether a number moved because the gate moved or because the content did.
4. **Batch-keyed depth.** Our pair/slot path runs **draftless** ("slot drafts accepted 0 of 0") while every surveyed engine keys depth to batch size. That is a whole lane's worth of unused speculation.
5. **Entropy-scaled acceptance.** Medusa's co-threshold widens on flat distributions; our typical-acceptance rescue is a bare constant (`STRATA_TYPICAL_ACCEPT=0.2`), and Medusa/Hydra sweep ε over 0.05–0.25 — an order below our 0.5–0.7 floor.
6. **Report length next to the rate.** Every system that reports both does so side by side; our scoreboard carries the rate alone.

## 7. Caveats the research itself raises

- **Typical acceptance is not lossless.** Medusa: "we ascertain that it is typically unnecessary to match the distribution of the original model"; EAGLE-2/3 exclude Medusa from their temperature=1 tables for that reason. Our `spec_min_p` gate is *not* in that class (every offered draft is still verified exactly), but `STRATA_TYPICAL_ACCEPT` **is**, and it is armed at 0.2 in the standing recipe.
- **A rate measured on one workload is not a model constant** — TensorRT-LLM states this outright, and caps **our exact model** (Qwen3.8-Flash-Next) at `max_draft_len` 3, reporting GSM8K 95.11 (MTP3) vs 95.72 (BF16).
- **Two gaps were left unfilled rather than invented:** REST's verbatim metrics sentence and Medusa's numeric threshold ablation both fell past the retrieval cut, and are marked NOT VERIFIED in their files.
