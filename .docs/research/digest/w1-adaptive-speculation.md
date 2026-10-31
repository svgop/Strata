---
type: digest
id: w1-adaptive-speculation-scout
created: 2026-10-31
status: evidence
topic: W1 adaptive speculation — draft-length control + verification shaping in leading implementations
scope: read-only research; no code changes
---

# W1 — Adaptive speculation: competitive digest

Baseline under test: fixed T=8 (`--spec 6` + rescue), 30-41% row efficiency (2.4-3.6 accepted of 8-9 rows), window wall = 8.4 + 1.57×rows ms. Fields per competitor: draft-length policy, feedback signals, per-request adaptation, tree vs chain, measured numbers, confidence grade.

## Leviathan et al. — length-choice theory (arXiv:2211.17192)
- Mechanism: closed-form expected window value under i.i.d. acceptance; T chosen once from measured α and cost ratio.
- Policy: E[accepted](T) = (1−α^(T+1))/(1−α); walltime gain W(α,T,c) = E[accepted]/((1−α)(T·c+1)), c = t_draft/t_target; T* = argmax_T W, integer, "found numerically". Corollary 3.9: gain exists iff α > c; floor (1+α)/(1+c).
- Signals: offline α per (draft, target, task). Oracle per-step T adds ~60% over best fixed — the paper's own argument for adaptivity.
- Numbers: T5-XXL on TPU-v4: α 0.53-0.82, T* 3-7, 1.4-3.4×. Measured c ∈ 0.02-0.11.
- Grade: verified-from-source. https://arxiv.org/abs/2211.17192

## vLLM (mainline + Ascend dynamic SD)
- Mechanism: mainline = fixed `num_speculative_tokens` per step (ngram, eagle/eagle3, mtp, suffix, draft_model); `suffix_decoding_max_spec_factor` caps depth as a multiple of prefix-match length (dynamic, per-request). Ascend = two dynamic controllers.
- Policy (Ascend DSpark/DFlash): draft K tokens; per-request confidence = DSpark head sigmoid or `max(softmax(logits))` of argmax draft (DFlash, head-free); every `budget_update_interval=50` steps recompute shared per-request verify budget; each request keeps floor `min_verify_tokens` (example `initial_verify_budget_per_req: 5`, `budget_threshold: 0.7`).
- Policy (Ascend batch-size path): shared K per concurrency band `[start_bs, end_bs, optimal_K]` — K=3 for bs 1-64, K=1 for 65-128, K=0 (spec off) 129-512.
- Signals: draft confidence per position; concurrency. Per-request: yes (DSpark/DFlash budget); batch-K is global.
- Numbers: none published for the dynamic paths; related draft-KV-window work reports +381% E2E at 32k ctx. Mainline docs table: dynamic SD "higher than base SD" at high QPS.
- Grade: verified-from-source (docs; no measured-number source). https://docs.vllm.ai/en/latest/features/speculative_decoding.html ; https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html

## TensorRT-LLM
- Mechanism: draft-target chain with engine-build-time fixed `max_draft_len` — docs state "a single sequence of draft tokens with length max_draft_len is created for every request" and "no way to dynamically disable speculation". No DTR/dynamic-draft-len feature exists.
- Tree: EAGLE3 `use_dynamic_tree` + `dynamic_tree_max_topK`; budget `max_total_draft_tokens` with `max_draft_len <= max_total_draft_tokens <= dynamic_tree_max_topK * max_draft_len` — tree expands at runtime inside a fixed row budget. Medusa/ReDrafter: tree config is a runtime parameter; ReDrafter runs beam search + acceptance inside the engine.
- Lossy acceptance: MTP `relaxed_topk`/`relaxed_delta` — accept draft token in top-K if `log(P(top1)) − log(P(t)) <= relaxed_delta`.
- Numbers: none published on the feature page; acceptance bound example 1-5 tokens/step for a 4-head Medusa tree.
- Grade: verified-from-source. https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html ; https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md

## SGLang EAGLE-3
- Mechanism: tree drafting — expand `speculative-num-steps` levels with `speculative-eagle-topk` branching, rerank nodes to top `speculative-num-draft-tokens` final nodes, verify the tree in one pass. Auto defaults: steps=5, topk=4, draft_tokens=8 (Llama/Grok); if topk=1 then draft_tokens = steps+1 (chain).
- Policy: static per server launch; no runtime length adaptation. Tuning is offline sweep via `bench_speculative.py` (accept-length-adjusted throughput). AutoSpec RFC for runtime tuning is open, not merged.
- Signals: none online (offline acceptance-length benchmark only). Per-request: no. `--speculative-accept-threshold-single/-acc` default 1.0 (inactive).
- Numbers (LLaMA-3.1-8B, MT-bench, 1×H100): baseline 158.34 tok/s → EAGLE-2 244.10 → EAGLE-3 373.25 (2.36×).
- Grade: verified-from-source. https://docs.sglang.io/advanced_features/speculative_decoding.html

## AdaEAGLE (arXiv:2412.18910)
- Mechanism: Lightweight Draft Length Predictor — 3-layer MLP + residual; input = [last-validated-token embedding; target final hidden state]; output scalar → k̄ = Round(MLP), k = min(k_max, max(0, k̄)); predicted before each draft round. Trained on (state, longest-accepted-prefix) pairs with asymmetric L1 (penalty λ>1 for under-prediction; draft pass ~20× cheaper than target).
- Variant DDD: exit draft when steps > predicted k AND sequence log-prob < −0.6 (insensitive −0.2..−1.0).
- Signals: target hidden state (context-based), not acceptance history. Per-request: yes, per iteration. Chain only, greedy only.
- Numbers: Vicuna-7B+EAGLE, A800: 66.35 tok/s, τ=3.03 vs best fixed EAGLE 64.66, τ=2.71; oracle-lengths τ=3.41, 83.46 tok/s → ~42% headroom over fixed; waste tokens grow 12k→50k as fixed length 2→6.
- Grade: measured-in-paper. https://arxiv.org/abs/2412.18910

## DISCO — dynamic speculation lookahead (arXiv:2405.04304)
- Mechanism: 2-layer FFN classifier after each draft token i decides "draft another" vs "verify"; input = [Top-10 draft probs, draft entropy, position i]; trained with cross-entropy against draft/target agreement labels (Total-Variance target). Cap SL_max for fixed tensor shapes; threshold τ tuned per dataset for latency.
- Signals: draft distribution only (no online acceptance EMA). Per-request: yes, per iteration; classifier transfers across same-category tasks (MBPP→HumanEval).
- Numbers (greedy, 1×A100): +11.2%/+11.4%/+13.6%/+4.9% vs validation-tuned static SL (avg +10.3%); +31.4% avg vs Gante entropy heuristic; E2E 1.84-2.15× over target-only; oracle SL 2.22-2.48×, oracle variance huge (18.0±46.2 on MBPP) — static SL provably mis-calibrated per step.
- Grade: measured-in-paper. https://arxiv.org/abs/2405.04304

## DSBD — dynamic-width speculative beam decoding (arXiv:2409.16560, AAAI'25)
- Mechanism: draft forest of W_L trees; before verifying layer l, compute expected accepted-beam count from the joint acceptance distribution P_{p,q}^(l)(m,k) built from draft/target probs (speculative-sampling recursion), then set width W̃_L^(l)(t) = max{K : 1 − Σ_{k<K} P(m,k) ≥ t}; W = max(W_min, W̃); t ∈ {0.7,0.9}. Rejected parents auto-reject children; surplus draft beams discarded once W_L accepted.
- Signals: draft + target probabilities per row (statistical, pre-verification). Per-request: yes, recomputed per layer per step.
- Numbers: Llama-2-13B/68M, Llama-3.1-8B/1B: 1.5-1.9× vs beam sampling, EM 86 vs 74-75 (SpecInfer) at equal speed.
- Grade: measured-in-paper. https://arxiv.org/abs/2409.16560
- Relevance: this is the published math for "runner-up tokens as branch rows at fixed row budget" — width from expected acceptance, not fixed top-k.

## Medusa — typical acceptance (arXiv:2401.10774)
- Mechanism: lossy acceptance — accept candidate x if p(x) > min(ϵ, δ·exp(−H(p))) (hard floor + entropy-scaled threshold); ablation uses α = √ϵ, ϵ swept 0.01-0.25. Distribution not preserved (MT-Bench quality Δ −0.14..+0.05); gains grow with temperature.
- Tree: 5 heads, base = per-head top-k Cartesian product; optimized tree greedily adds nodes maximizing Π a_j^(i_j) (calibrated head accuracies) to a 64-node budget — beats a 256-node dense tree.
- Numbers (batch 1): Vicuna-7B 2.83× (3.47 acc tokens/step), 13B 2.83× (3.51), 33B 2.35×; speedup peaks near 64 candidates, degrades beyond batch ~32.
- Grade: measured-in-paper. https://arxiv.org/abs/2401.10774

## What this changes on the row
- Our controller = Leviathan W(α,T,c) with measured cost_ms(T) instead of the (γc+1) denominator — theory-endorsed; the α > c corollary says the T=1 min-gain guard is exactly the speedup-existence condition. Keep.
- No leading engine adapts T online from acceptance alone: production adaptivity is (a) confidence-budgeted per-request verify length (vLLM-Ascend DSpark/DFlash, with a `min_verify_tokens` floor), (b) K per concurrency band (vLLM-Ascend), (c) offline sweep (SGLang). Our per-request/per-slot depth plan should copy shape (a): EMA keep-count per slot → shared budget → per-request floor.
- Cheap adaptive-trim signals beat acceptance-EMA lag: DISCO (draft top-k probs + entropy + position, +10.3% over tuned static) and AdaEAGLE (target hidden state, ~42% oracle headroom) adapt before/inside the draft pass. An entropy gate on our existing cut-only trim is the lowest-risk first move.
- Runner-up tree is validated by three engines: SGLang reranks tree to a fixed `num_draft_tokens` row budget; TRT-LLM EAGLE3 grows a dynamic tree inside `max_total_draft_tokens`; Medusa's optimized 64-node tree beats 256 dense nodes. DSBD supplies the exact selection rule: rank branch rows by expected accepted count from the p/q recursion, keep rows until cumulative acceptance-prob mass ≥ t.
- TRT-LLM confirms a real product need: "no way to dynamically disable speculation" is listed as a limitation there; our controller must include the concurrency dimension (batch × rows is what inflates verify cost) — our linear row-wall cost model already captures this; wire batch size in as a controller input.
- Lossy acceptance (Medusa typical, TRT-LLM relaxed top-k/δ) buys ~10-15% extra accepted tokens and helps most at high temperature — keep as a flagged-off option behind the lossless default.
