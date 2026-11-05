---
type: research-digest
date: 2026-11-05
topic: Runtime decode control v2 — draft on/off, depth, mid-flight abort, verify-row batching, for the qwen slot-drafter with fast chain decay
supersedes-adds-to: 2026-11-04-decode-control.md (AdaEDL/block-verify/BanditSpec/vLLM-AV banked there, not re-derived), w1-adaptive-speculation.md, w2-ungated-draft-chain.md
in-tree-anchor: src/spec/controller.cpp (W1 cost-model controller, adopted ee435ae5) — read for this digest
status: verified
sources: arXiv HTML full texts fetched unless flagged; vLLM + SGLang production source files fetched via gh/curl (paths inline); every claim graded measured-public / design-inference / speculation.
---

# Decode control v2 — the controller for THIS rig (2026-11-05)

## 1. Question

What is the best production formulation for RUNTIME decode control — when to draft at all, how deep, when to abort a draft mid-flight, how to batch verification — given our measured acceptance shape: per-token acceptance high (88.3% overall, 96% edit-phase, slot-draft 62.9%) but chains decay before block 5 (the shape that killed DSpark at depth 12), verify lane DRAM-byte-bound at 93–100% of 49.5 GB/s, 8-row window cap, W1 cost-model controller already adopted.

**ID correction to the mandate:** AdaEDL is arXiv **2410.18351** (Agrawal, Jeon, Lee — Qualcomm, ICLR 2025; re-verified today at https://arxiv.org/abs/2410.18351). The mandate's 2410.18398 is "You Only Look Around" (low-light object detection, NeurIPS 2024) — unrelated; do not cite it.

**What the in-tree controller already has (read from `src/spec/controller.cpp`, 73 lines):**
- Survival-product acceptance math: `Controller::expected()` — `run *= p[i]; e += run` — is exactly the EAGLE-2/D-cut survival form.
- Per-position online tables: `mtp_p_[i]`, `lookup_q_[bucket]`, EMA-updated from verify labels **censored at first rejection** (`observe()` — labels only positions ≤ accepted+1; identical labeling rule to vLLM's shipped estimator).
- The off-arm: `Source::None` + `min_gain_` guard vs 1/step_ms(0) AR baseline.
- The MoE drafting tax in the cost: `CostModel::step_ms` prices CPU misses via distinct-expert growth `u = distinct_ratio[n-1]` and `extra_use_cost` — rows are NOT free on this box, the controller already knows.
- Known weakness in `observe()`: the extrapolation law pulls all unobserved deep positions toward the deepest observed rate — a cruder, optimistic version of vLLM's damping (it can over-extend the chain after one lucky deep window).

So the deliverable below is a **delta spec on the adopted controller**, not a replacement.

## 2. Findings

### 2.1 The production controller, in code — vLLM adaptive verification (SHIPPED)
Provenance: `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` (646 lines) and `acceptance_estimator.py` (515 lines), fetched from main today; [docs](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification) (prior digest); [blog 2026-10-14](https://raw.githubusercontent.com/vllm-project/vllm-project.github.io/main/_posts/2026-10-14-adaptive-verification-online-estimator.md) fetched. Grade: **measured-public** (code read + blog AUC table).
- Feature: **logit(q_max)** of the temperature-scaled draft distribution — `m* − log(Σ_{z≠m*} e^(z−m*))`, clamped ±40 (`_MAX_LOG_ODDS = 40.0`). It is a function of the **prefix alone**: "Scoring the drawn token instead would select on the proposal and bias the emitted distribution toward the drafter's confident modes" (verbatim code comment — the losslessness argument; also the D-cut Appendix B caveat).
- Model: per-position logistic `p = σ(ω·f + β_k)` — ONE shared slope ω, K per-position intercepts. K+1 parameters total. Cold start ω=0.5, β=−0.2 (medians across five production model pairs: DSV4-Flash, Muse-Glimmer-30B, MiMo-V2.5-Pro, Inkling, Kimi-K2.5).
- Recalibration: IRLS (Newton) on the Bernoulli log-likelihood, arrowhead normal equations solved in one Triton kernel. `REFIT_INTERVAL = 100` steps, ridge `L2 = 1e-3`, `DAMPING_OBSERVATIONS = 50.0` — "a data-poor position keeps learning, just slowly, instead of freezing until it crosses a threshold." Slope damped by total evidence `n/(n+50)`, each intercept by its own count.
- Budget: `survival = cumprod(confidence_probs)`, **global sort** over all (request, step) slots, `draft_budget = argmax((num_sampling_requests + cumsum(sorted_survival)) / costs)`; costs = runtime-measured draft+verify cost tables (step functions below the CUDA-graph capture limit, interpolated above, median of 5 profiled replays). Survival is monotone non-increasing along a request → winners are always contiguous prefixes.
- Measured feature table (~4.5M labeled drafts): logit(q_max) AUC **0.8336**, ECE 0.0100; raw q_max AUC 0.8259, ECE 0.0736 (7× worse calibration); trained confidence head AUC 0.8191 — the one-feature online logistic **matches or beats a trained head at zero training cost**. Slope stable across 5 workloads: ω 0.51–0.55.
- The decay shape, measured in production: [DSpark blog](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification) — "the last drafted token of a 7-token block survives less than 10% of the time, against more than 70% for the first." Same shape we measured. Their budget algorithm handles it by construction, with no hysteresis at all — the coarse discretization (budget argmax) plus monotone survival curves are the stabilizer.
- **Mechanism:** slot competition by survival product replaces per-slot fixed allocation; the estimator replaces offline tables. Both map one-to-one onto our `mtp_p_`/`lookup_q_` and `Choice` selection.

### 2.2 The production controller, in code — SGLang adaptive speculative decoding (SHIPPED)
Provenance: `python/sglang/srt/speculative/adaptive_spec_params.py` (350 lines, fetched from main today) + `adaptive_runtime_state.py` (state-swap scaffolding) + PR #37274 (policy interface, merged 2026-09-02). Grade: **measured-public** (code read; no published benchmark numbers — named boundary).
- State: per-batch-size-slot `AdaptiveStepSlot` — EMA of accepted draft length (`ema_alpha = 0.2`), current steps, batch count. `DEFAULT_ADAPTIVE_CONFIG` ladders: BS 1 → `[1,3,5,7]`, BS 8 → `[0,1,3]`, BS 32 → `[0,1]`, BS 64 → `[0]` (spec fully off).
- Law: `target_steps = clamp(round(ema_accept_len) + 1, min, max)` — "probes one step beyond observed acceptance." EMA updates every batch; the LADDER moves at most one rung every `update_interval = 5` batches after `warmup_batches = 10`.
- **Hysteresis (the shipped constants):** down at `ema ≤ prev_step − 0.5 + down_hysteresis` with `down_hysteresis = −0.25` (BS 1 only); up at `ema > current_step − 0.5 + up_hysteresis` with `up_hysteresis = 0.0`. Asymmetric: dropping a rung needs more evidence than climbing — the oscillation damper at the decay knee.
- Spec-off arm is a first-class rung: after a steps=0 interval the controller re-probes the smallest positive rung ("Probe the smallest positive step after a zero-step nospec interval") with the EMA re-seeded neutral — exactly BanditSpec's γ=0 arm, shipped.
- The `ceiling_coeff` knob caps depth downward only, "never blocks step-ups, so the system can explore higher steps and let the EMA catch up" — a one-sided trust-region against EMA under-estimation.
- Constraints (from `adaptive_unsupported_reason()`): EAGLE/EAGLE3 only, topk=1 (chain) only, no attention-DP, no two-batch-overlap; runtime step switching = atomic swap of pre-built `SpecRuntimeState` (attention backends + graph runners per step count) — CUDA-graph shapes are capture-time facts, the policy only swaps pointers.

### 2.3 Confidence-survival row pruning — D-cut (batched verify-depth pruning)
[arXiv:2607.14647](https://arxiv.org/html/2607.14647) fetched. Grade: **measured-public**.
- Signal: drafter's per-token confidence → prefix product `s_{i,k} = Π c_{i,t}` (bonus position s=1 guaranteed — the K ≥ B floor "ensures every request retains at least its bonus token"). All B×(D+1) slots ranked **globally**; top-K admitted under the row budget. Monotonicity → "only suffixes are pruned, never interior positions" — verify reads contiguous KV per request.
- Budget from a **discrete bucket set** ℛ = {0.25, 0.50, 0.75, 1.00} of the scheduled total, chosen by argmax of summed survival over profiled cost C(B,ρ), profiled once at startup. Discretization is their stability mechanism (no hysteresis, no EMA).
- Prediction quality of the signal: AUROC **0.957/0.962/0.972** (GSM8K/HumanEval/MT-Bench) for block acceptance; Spearman 0.748/0.770/0.579.
- Numbers: 8×H20/H800, DFlash D=16 baseline, concurrency 16–64: +1.26→1.65× over the fixed-depth drafter; up to 3.0× over AR on MoE. At T=1.0 the fixed-depth drafter collapses to 0.68× vs AR while D-cut sustains 1.15× — the survival budget IS the temperature robustness. Selector overhead 0.55–0.58 ms/step (2.2–3.4%).
- Caveats (verbatim-graded): naive confidence scoring is biased under rejection sampling — their Appendix B shifted-score construction is required for sampling correctness (same conclusion as vLLM's prefix-only feature); "early stopping may miss a later optimum when the utility-cost curve is not unimodal."

### 2.4 AdaEDL — the entropy abort, and what entropy is actually good for
[arXiv:2410.18351](https://arxiv.org/html/2410.18351) re-fetched today. Grade: **measured-public**.
- Rule: stop drafting when `1 − γ·H_DM(x) < λ`; H = entropy of the draft logits, γ = 0.2, λ tuned online (EMA α=0.9 of realized acceptance, ±ε steps). Justified as an approximate lower bound on acceptance via Pinsker/TVD — a ONE-SIDED bound, valid at every temperature ("at T=1.7 static spec decode drops 57% below autoregressive while AdaEDL still provides an 8% boost"). +10–57% over static draft length (A100).
- What entropy is NOT: DSDE's measured correlation table ([arXiv:2509.01083](https://arxiv.org/html/2509.01083v1), banked 2026-11-04) — draft entropy vs per-token acceptance r = −0.339 (T=0), −0.235 (T=1.0). Entropy is a weak INSTANCE predictor.
- The key decomposition (vLLM estimator blog, measured): intercept β₀ correlates with draft entropy at **r = −0.89** (entropy sets the REGIME/level), while the slope ω is nearly orthogonal to entropy at **r = −0.08** (the per-token signal is not entropy). **Two signals, two timescales: entropy gates the regime, q_max-derived features gate the token.** This is the cheapest-per-bit allocation on our box: entropy for abort/ladder decisions (1 pass, no training), logit(q_max)-logistic for depth/survival (2 reductions, K+1 online params).
- Successors checked: EASD ([arXiv:2512.23765](https://arxiv.org/html/2512.23765v1), fetched) — entropy + top-5-overlap trigger zeroes the draft top token OUT of the target distribution = **lossy** (also slows GPQA to 0.93× when it over-fires) → SKIP for our min_p lanes. EDSD (ACL 2026, search-level) is a training-time design change, not a runtime controller. llama.cpp entropy gate ([issue #29875](https://github.com/ggml-org/llama.cpp/issues/29875), fetched) — OPEN/stale, never merged; CPU-only bench (Qwen2.5-3B/0.5B, K=8): entropy 1.5 bits beats their existing p_min 0.6 gate by +11.3% (16.50 vs 14.83 t/s, acceptance 88.5% vs 79.0%) — evidence entropy beats naive top-1 thresholds even before counting that our loop already reads logits (no added sync). Their reviewer's Gini impurity (1 − Σp²) suggestion is the cheapest variant (no log) — grade: design-inference, untested publicly at scale.

### 2.5 EAGLE-2/3 tree policies — what transfers to a slot-drafter
EAGLE-2: [arXiv:2406.16858](https://arxiv.org/html/2406.16858v2) fetched. EAGLE-3: [arXiv:2503.01840](https://arxiv.org/html/2503.01840) fetched. Grade: **measured-public**.
- EAGLE-2's math is the survival product again: node value `V_i = Π_{t_j ∈ Path(root,t_i)} p_j ≈ Π c_j`; expansion = top-k per layer by value (k=10); final rerank to the token budget with ties broken toward shallower nodes (keeps the selected set a connected tree); depth fixed at 6, budgets 60/50/48 nodes. Calibration measured: draft confidence <0.05 → acceptance ≈0.04; >0.95 → ≈0.98. 3.05–4.26×, +20–40% over EAGLE-1.
- The context-adaptive claim: "the confidence score (probability) of the draft model is a good approximation of the acceptance rate of draft tokens" — the whole dynamic-tree edifice rests on per-token confidence as acceptance proxy, exactly the feature vLLM ships and D-cut scores (AUROC 0.95+).
- EAGLE-3 inherits the tree ("EAGLE-3 also adopts the context-aware dynamic draft tree proposed in EAGLE-2"), deepens 6→8 at constant node budget, and — decisive for us — its own serving benchmarks **disabled the tree** (chain length 3 on SGLang, 2 on vLLM): at real serving shapes the tree's marginal value over a well-sized chain is small.
- Transfer verdict: a slot-drafter with per-position confidences gets the EAGLE-2 benefit through **survival ranking over (slot, position)** — the tree is the multi-branch generalization of the same ranking; SpecPilot's best-first builder (banked, hfpapers-specverify §2b) is the tree form when the snapshot-free GDN verify lands. Until then, chains + survival budget capture most of the value at zero row-budget change.

### 2.6 Block verification, beyond 2403.10444 — SpecTr-GBV
[arXiv:2604.25925](https://arxiv.org/html/2604.25925v1) fetched (EMNLP 2026). Grade: **measured-public**.
- Unifies multi-draft generation with greedy block verification via optimal transport over draft/target token blocks. K i.i.d. drafts of length L; acceptance iterates over draft SEQUENCES (not positions), each sub-block accepted with probability h_ik carrying a `(1 − min{p/q, 1})^K` structure so marginal acceptance grows with K; rejected sub-blocks recorded in a set H and skipped on repeat; correction token from the residual. **K=1 reduces exactly to GBV (2403.10444)** — strict generalization, "the optimal expected number of accepted tokens ... increases as the number of draft sequences grows."
- Compute: a rule change on the same verify logits — "no extra target-model passes"; per-position acceptance computed at O(|Ω|), cheaper than SpecTr's O(|Ω| log K); measured verify overhead −50%+ vs SpecTr (−74.8% HumanEval).
- Numbers: DeepSeek-33B/1.3B: BE 8.59, SR 2.12 = +29.3% vs static SD, +8.2% vs SpecTr, +27.0% vs single-draft GBV; temperature-insensitive ("minimal variation across temperatures" — unlike token-verify whose gains grow with T; this is a different beast, gains persist at low T).
- Caveats: claims optimality **within i.i.d. draft generation** (not over trees); needs a cross-iteration distribution modification (next L−τ−1 tokens drawn from q_new — implementation care); margin over baselines widens 0.78%→2.75% for K=1→7 — the win needs multiple drafts to bite.
- For us: we banked GBV (Theorem 2 dominance, +2–8% sampled lanes, zero gain greedy). SpecTr-GBV's K>1 form is unlocked only when two draft sequences per position exist (lookup + MTP continuation are natural candidates but are not i.i.d. — they share the prompt and often the tokens, which shrinks the K-gain; the routing-overlap counterpoint in hfpapers-specverify §3 applies). Grade for the K>1 step on our rig: **speculation**; bank GBV K=1 first (already queued), revisit K=2 only after GBV lands and the two drafters' candidate-overlap is measured.

### 2.7 Cost-aware trees and profiled controllers — CAST, FASER (context for the cost model)
CAST: [arXiv:2510.26577](https://arxiv.org/html/2510.26577) fetched. FASER: [arXiv:2604.20503](https://arxiv.org/html/2604.20503) fetched. Grade: **measured-public**.
- CAST proves EAGLE-2/3's confidence-only trees are special cases of utility-vs-cost maximization (Theorem 4.1: linear cost recovers EAGLE); keeps nodes only while marginal utility/cost ≥ threshold; +5–20% over SOTA, stays >2× at batch 32 where EAGLE degrades toward 1×. It prices verify with hardware-profiled `f(B,c,n)` tables — the same shape as our W1 cost model, no MoE/lane-bytes term (named boundary: nothing in CAST, D-cut, or FASER prices expert-union growth; that term remains our novelty, already in `step_ms`).
- FASER: "a higher acceptance rate does not necessarily imply lower latency" — optimal length 3, not length 1, at batch 32/len 6, because short lengths pay unamortized verification rounds. Objective J = T̂/(s·â+ε) — latency per committed token, not acceptance. Length per request via GP-LCB; drift handled by "a sliding window of recent batches rather than the full history" + 2-hour background profile refresh. Up to 53% throughput / 1.92× latency vs SOTA.
- Takeaway: our W1 controller's `min_gain_` guard and tokens-per-ms objective are the right objective family (FASER/CAST validate); the missing piece is drift handling (sliding window, not lifetime EMA — our `observe()` EMA has no window and never forgets a stale regime; vLLM's refit+damping is the middle ground).

### 2.8 The decay-survival controller — does anything published survive chain decay?
The combination "chain decay awareness + hysteresis" has **no single named paper** (searched; named boundary — closest are D-cut and vLLM AV, which handle decay implicitly via survival monotonicity, and SGLang, which handles oscillation via asymmetric hysteresis but not decay explicitly). The synthesis in §3 composes the two published halves; grade: **design-inference** anchored on measured-public parts. Evidence the composition is sound:
- vLLM measured the same decay in production (<10% tail survival) and their controller survives it via monotone survival + budget argmax — no oscillation reported.
- SGLang measured oscillation risk exists where signals are coarse (accepted-length EMA) and shipped asymmetric hysteresis as the fix.
- Our DSpark depth-12 failure is explained by exactly their split: depth was sized from an average-acceptance signal that could not see the within-chain decay, and depth changes had no hysteresis — both failure halves have shipped fixes (§3).

## 3. The single best controller for THIS rig — "feature-modulated survival control"

Delta spec on `src/spec/controller.cpp`. Four additions, in adoption order; each independently gated.

### 3.1 State variables (all per-session unless noted)
- `β_k` (K_MAX+1 floats): per-position logistic intercepts — SEEDED from current `mtp_p_[k]`/`lookup_q_[bucket]` via `β_k = logit(p_k)` (the tables become the prior, not a parallel system).
- `ω` (1 float): shared slope, init 0.5 (vLLM cold-start median across five model pairs).
- IRLS accumulators: `info[k,3]`, `grad[k,2]`, `counts[k]` — O(K) floats, cleared per refit (vLLM layout, arrowhead normal equations).
- `ema_accept_len` per concurrency band {solo, pair, conc} (3 floats), init current depth − 1 (SGLang law).
- Last window's per-position entropy path `H_1..H_k` and `f_k = logit(q_max)` (device-side reductions, K_MAX floats parked in pinned staging we already own).
- Optional: `λ` (entropy-gate threshold), EMA-tuned ±ε around realized acceptance (AdaEDL law).

### 3.2 Inputs and compute cost
- `f = logit(q_max) = m* − log(Σ_{z≠m*} e^(z−m*))` over the temperature-scaled draft logits: one max-reduce + one logsumexp-reduce over V (~152K) on logits the MTP head already materialized — µs on the 5070, no new memory traffic beyond the logits themselves. Lossless-safe: prefix function of the draft distribution (vLLM verbatim argument; trimming never conditions on the drawn token).
- `H = −Σ q log q` on the same logits: one fused pass (or Gini `1 − Σp²` if the log shows up — llama.cpp reviewer suggestion, cheaper, unproven).
- Labels: FREE — `observe()` already receives `accepted` per window; the censoring rule (label positions ≤ accepted+1) is already implemented. Add: fold `(f_k, label)` into the IRLS accumulators in the same pass.
- Cost tables: `step_ms` unchanged — it already prices dense ratio, distinct-expert growth (the MoE tax), sync, and draft steps. Add draft-abort savings by stopping at the aborted k (the cost model is already linear in k).

### 3.3 Decision rules
1. **Draft on/off** (the off-arm, hardened): keep `Source::None` but gate it by the per-band EMA with SGLang's constants — off when `ema_accept_len ≤ 0.5 + down_hysteresis` (down_hysteresis = −0.25), re-probe smallest positive depth after ≥5 no-spec windows with EMA re-seeded neutral. Today's `min_gain_` guard stays as the per-window economic check on top.
2. **Depth**: `k* = argmax_{k ≤ K_MAX} [1 + Σ_{i≤k} ŝ_i] / step_ms(k)` — unchanged form, but `ŝ_i = σ(ω·f_i + β_i)` replaces the bare table: the current window's logits modulate the prior. One ladder-rung change per decision vs the previous window (SGLang one-rung law; their TODO names overshoot as the known failure of unbounded moves).
3. **Mid-flight abort** (new, host-side, free in our architecture): after each drafted position, stop the chain if `1 − γ·H_i < λ` (γ=0.2) OR `ŝ_i < s_min` (s_min = 0.10, the survival floor implied by the production decay tail). W2 digest establishes our draft loop is host-driven and already reads logits per step — the abort costs zero sync (the RFC #36657 `.item()` problem is vLLM-specific). **Phase coupling:** if/when the single-graph ungated draft chain lands (W2's plan), the true mid-flight abort must retire to a stale-signal PRE-LAUNCH sizing — budget from the previous window's H/ŝ path, host-decided, graph replayed at the chosen k, post-hoc trim as today (vLLM's double-buffered stale-confidence pattern is the shipped proof this loses little). Do not attempt device-side early exit inside a captured graph — zero precedent anywhere (W2), and conditional-graph machinery is already the named next pick elsewhere.
4. **Verify-row batching at conc** (new): under the 8-row cap, admit rows by **global top-K over (slot, position) survival products** instead of per-slot cost-model shares. Monotone → contiguous prefixes per slot (D-cut), so KV reads stay contiguous and the verify pass shape is unchanged; the decision is 6×K_MAX floats sorted on host — no doorbell, no graph reshape. The W1 cost model still sets the TOTAL row budget; survival only allocates it across slots. At solo this rule is a no-op (one slot).

### 3.4 Adaptation law with hysteresis
- Estimator refit every **100 windows** (IRLS, ridge 1e-3); updates damped `n/(n+50)` per position — replaces the risky deep-position extrapolation in `observe()` (which can chain a lucky window into an over-extended table).
- Depth governor: EMA α=0.2 per concurrency band, ladder moves 1 rung / 5 windows after 10 warmup; **asymmetric hysteresis down −0.25 / up 0.0** — the measured-production constants (SGLang).
- Entropy gate λ: EMA-tuned ±1 step around realized acceptance (AdaEDL), clamped to [0.3, 0.9]; never below the survival floor.
- Never move depth and off/on in the same decision (SGLang's zero-step re-probe law); never let `ceiling_coeff`-style caps block step-ups (one-sided caps only).

### 3.5 Predicted effect — arithmetic on our numbers
Calibration: a decay path per our falsification ("chain decays before block 5"), per-position conditional p = [0.70, 0.60, 0.50, 0.35, 0.15, 0.10, 0.08, 0.06] gives survival cumprod [0.70, 0.42, 0.21, 0.074, 0.011, 0.001, ...] → E[accepted | k=8] ≈ 1 + 1.42 ≈ **2.42 tokens on 9 rows ≈ 27% row efficiency** — inside our measured 30–41% band (2.4–3.6 of 8–9 rows, w1 digest). Model calibrated.
- **Rows 5–8 are worth Σ ≈ 0.012 tokens** and cost 4 × 1.57 ms window wall + 4 draft steps + U(8)/U(4) extra expert union on a lane at 93–100% ceiling. The abort/dept-cap at the knee saves ~6 ms/window at −0.01 tokens in decayed windows. W1 already prices this in `step_ms` for the AVERAGE case; the abort covers the per-window tail the EMA misses (hard windows drafted to depth before the EMA moves).
- **Conc survival allocation:** measured conc = 4.81 rows/window, 271.5 rows/s, T ≈ 3.4 + bonus at 315.4 tok/s (banks). Live traffic mixes phases (96% edit-phase vs 88.3% overall vs 62.9% slot-draft): edit-phase slots' survival at position 3 ≈ 0.89, decayed generative slots' ≈ 0.21. Global top-K shifts ~1–1.5 rows/window from ~0.1-survival tails to ~0.85-survival prefixes → +0.4–0.8 accepted/window on ~4.4 → **+9–18% accepted tokens at identical row count and window count**. Class check: D-cut measured +26–65% over fixed-depth at conc 16–64 (bigger batches, worse baseline drafter); vLLM AV sits on the throughput-interactivity frontier at 5 of 7 concurrencies. We claim the conservative fraction: **conc 315 → 340–360 t/s**.
- **Entropy abort in deep/generative phases:** AdaEDL +10–57% over static (A100), SVIP +17–22% long-context (banked); our draft chain is MTP+lookup inside 9–16 ms windows — expected **+5–10% window throughput at 150–261K depth**, ~0 on edit-phase (gate never fires at 96% acceptance — free insurance).
- **Composition:** block verification (banked #4, +2–8% sampled lanes, zero compute, Theorem 2 dominance) is orthogonal — different axis (accept rule, not control).
- **Aggregate vs the ≥+4% campaign bar:** solo +1–4% (near-optimal already; edit-phase dominates the battery), pair +3–8%, conc +8–15%, deep-lane +5–10% → battery average ~+4–9%. Clears the bar on the lanes with headroom; solo near-neutral is acceptable only because the average clears.

### 3.6 Falsifier (which measured signal kills it)
- **Estimator kill:** log AUC of `f = logit(q_max)` vs the observed first-rejection position on OUR drafter over ≥2000 windows. If AUC < 0.65 (vLLM: 0.8336; DSDE warns entropy-class signals sit at r ≈ −0.24..−0.34), the feature term adds nothing on our drafter → revert to bare tables + slow EMA; the arm dies. This is the first measurement to run — cheap, no engine change (telemetry-only).
- **Conc-kill:** if per-slot survival at conc is homogeneous (all slots in the same phase → top-K ≈ current allocation), the reallocation gain collapses <1% → the allocation arm dies, keep W1.
- **Abort-kill:** if aborted windows show no window-time delta at equal accepted tokens vs matched non-aborted windows (i.e., draft rows are cheaper than `step_ms` says), the MoE-tax premise is wrong at the margin → abort arm dies.
- **Campaign law:** after hysteresis tuning, < +4% average across the battery → falsified, never retried blind; record which sub-arm carried the loss.

## 4. What NOT to do

1. **DSpark-class depth from average acceptance (the measured failure).** Depth 12 died because the sizing signal was an average that could not see within-chain decay, and depth moves had no damping: rows past the decay knee accept at <10–20% (our falsification; vLLM measured the identical <10% tail in production) while each row costs 1.57 ms window wall + serial draft step + expert-union growth on a saturated lane. Both halves have shipped fixes — per-position survival signals (vLLM/D-cut) and asymmetric hysteresis (SGLang) — and neither existed in the falsified arm. Depth beyond the knee is not "more speculation", it is paid-for rejection.
2. **Confidence thresholds per request (RFC #36657 shape).** Dominated by budget ranking (D-cut/vLLM shipped form); τ is model-pair/dataset dependent (0.4–0.6 manual range); closed stale upstream. Mean-of-batch gating also lets one confident request starve another (vLLM RFC's own failure note).
3. **q_sampled-derived features anywhere in a trim decision.** Breaks losslessness — conditioning the trim on the realized draw biases the emitted distribution (vLLM footnote's two-token counterexample; D-cut App. B). Only prefix functions of the draft distribution (q_max logit, entropy) are safe.
4. **Trained acceptance heads (SpecDec++/OmniDraft/AGSD).** Training tax; heads underestimate acceptance → wrongful early exit (OmniDraft's own lesson); and the shipped evidence says a K+1-parameter online logistic matches them (AUC 0.8191 head vs 0.8336 feature, vLLM). Import the lessons free: calibrate optimistic on decay, expect per-domain thresholds to differ ~2×.
5. **Per-token host sync for gating (llama.cpp p_min shape).** Our loop already reads logits per step (W2); adding a round trip is the documented anti-pattern. The abort rides existing reads or retires to pre-launch sizing under graphs.
6. **DSDE KLD-length control.** Trails static-opt at T=0 and AdaEDL at T=1.0; its value was the signal-correlation table, which we import as the estimator gate — not as a controller.
7. **Lossy verification/penalty (Judge family, Medusa typical, EASD).** Our min_p sampled lanes need exactness; measured OOD regression (HumanEval 86.6→80.4) and EASD's 0.93× GPQA regression when the penalty over-fires.
8. **SV companion model.** Third forward pass = lane bytes we do not have on a DRAM-saturated box; its win regime is compute-bound batches we do not serve.
9. **CAST-style static profile tables as the primary controller.** Profile staleness under phase shifts (edit↔generative↔depth); FASER's 2-hour refresh exists because static profiles drift. Profile once to SEED our cost model (W1 does), decide online from damped online fits.
10. **Device-side early exit inside a captured graph.** Zero precedent (W2 sweep); the graph-conditional machinery belongs to the one-doorbell work. Under graphs, the production-proven shape is stale-signal pre-launch sizing + post-hoc trim.
11. **SpecTr-GBV K>1 now.** Requires ≥2 genuinely diverse draft sequences; our lookup+MTP share tokens heavily (62.9% slot-draft overlap argues the diversity is limited); measure candidate overlap after GBV K=1 lands before spending anything on K=2.

## Sources

Fetched full text: AdaEDL https://arxiv.org/abs/2410.18351 (+html) | EAGLE-2 https://arxiv.org/html/2406.16858v2 | EAGLE-3 https://arxiv.org/html/2503.01840 | SpecTr-GBV https://arxiv.org/html/2604.25925v1 | CAST https://arxiv.org/html/2510.26577 | D-cut https://arxiv.org/html/2607.14647 | SV https://arxiv.org/html/2509.24328 | FASER https://arxiv.org/html/2604.20503 | EASD https://arxiv.org/html/2512.23765v1
Fetched production code (today): vLLM `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`, `acceptance_estimator.py` (main); SGLang `python/sglang/srt/speculative/adaptive_spec_params.py`, `adaptive_runtime_state.py` (main); PR https://github.com/sgl-project/sglang/pull/37274
Fetched production blogs/issues: https://raw.githubusercontent.com/vllm-project/vllm-project.github.io/main/_posts/2026-10-14-adaptive-verification-online-estimator.md | https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification | https://github.com/ggml-org/llama.cpp/issues/29875 | https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification
Banked (not re-derived): block verification 2403.10444 (Theorem 2), BanditSpec 2505.15141, DSDE 2509.01083 signal table, SpecServe 2503.05096, SVIP 2411.18462, SpecPilot 2611.02501 best-first/GDN/tree — all in 2026-11-04-*.md digests.
In-tree: `src/spec/controller.cpp` (W1 controller, read for this digest; anchors in §1 and §3).
