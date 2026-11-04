---
type: research-digest
date: 2026-11-04
topic: Decode-control intelligence (2025-2026): online acceptance prediction, sampling x speculation quality, verify-window economics at fixed row budget, training-free draft upgrades, production dynamic-depth controllers
machine-context: AM5 single-socket 16C/32T, dual-channel DDR5 (49.5 GB/s userland ceiling), RTX 5070 12 GB (sm_120), 184 GB RAM. Hybrid MoE engine: decode reads experts from host RAM at 93-100% of DRAM ceiling; MTP draft layer + lookup drafter; measured acceptance 88-94% live (96% edit-phase), verify window T 3.1-3.9 at 6-way concurrency, 8-row verify cap (12 proven, blocked on 8-column activation family); in-tree cost-model controller sizes the serial window; per-position acceptance tables fitted on own trace; min_p sampling, temperature 0.7/1.0; windows 9-16 ms; CPU expert lane is DRAM-bound; extra draft rows load extra experts (MoE drafting tax).
status: verified
sources: arXiv HTML full texts (fetched unless flagged search-level), ICLR/ICML/MLSys/NeurIPS/EMNLP/SoCC proceedings, vLLM/SGLang docs + GitHub RFCs (fetched), engineering blogs (fetched where noted) — every claim carries its source inline; search-level-only claims are flagged.
---

# Decode control intelligence digest (2026-11-04)

Falsified boundary respected throughout: our verify lane is DRAM-byte-bound, so only mechanisms that cut bytes (fewer wasted rows) or raise accepted-tokens-per-row transfer directly; GPU-compute-bound tricks (parallel drafting, SM partitioning) are recorded for their control-policy takeaway only. Queued items (Cohere DSD occupancy-gated K, SpecPilot best-first fixed-T accept + certified-argmax screen + token-forcing gate, EAGLE-3, DLite, dynamic tree verify scheduling) are not re-proposed.

## 1. Acceptance-prediction frontier

### 1.1 vLLM adaptive verification — online acceptance estimator + survival-product budget (SHIPPED)
[docs](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification) (fetched) | [vLLM blog 2026-10-14](https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator) (search-level).
- Mechanism: per-step draft-length control. Each (request, position) slot scored by **survival probability** — "the running product of that request's per-position acceptance estimates"; slots compete globally ("position 5 of a confident request can outrank position 1 of a doubtful one") until a token **budget** is spent. Budget source: "a cost model profiled at startup... picks the token count that maximizes expected accepted tokens per second."
- For drafters without a confidence head: "a small logistic model predicts acceptance from draft logits and is **recalibrated during serving** against the target's verification results." This is the online replacement for offline-fitted tables.
- Constraints (fetched): full CUDA graphs required (`--enforce-eager` rejected at startup); no LoRA; no pipeline parallelism; `use_local_argmax_reduction` incompatible without a confidence head; attention backend must tolerate device-decided query lengths. Supported: EAGLE, MTP, DFlash, DSpark, standalone drafters. Flag: `enable_adaptive_verification`, env `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`. No benchmark numbers published on the docs page.
- **Verdict: TRANSFERS.** This is our exact shape (per-position tables + cost-model controller + row budget) with one upgrade we lack: online recalibration against observed verification outcomes. Term moved: **per-position acceptance tables → drift-proof online estimator**; fewer wasted rows = fewer DRAM expert loads.

### 1.2 BanditSpec — per-step bandit over draft length with stopping-time regret bounds
[arXiv:2505.15141](https://arxiv.org/html/2505.15141) (fetched) | [ICML 2025 poster](https://icml.cc/virtual/2025/poster/44460) | [code](https://github.com/sail-sg/BanditSpec).
- Mechanism: each hyperparameter config (draft length γ∈{0..4}, drafter, tree) is a bandit arm, selected **per decoding round**; reward = accepted-token length per call. UCBSpec maintains only empirical means + UCB indexes; EXP3Spec variant for adversarial settings. Under stationary conditional means (no independence required — explicitly handles prompt-dependent acceptance), UCBSpec regret is `O(H·L²·log E[len])`; lower bound shows it is "optimal up to constants and L" for truncated-geometric acceptance. Proposition 3.1: output distribution matches canonical decoding (lossless).
- Numbers (fetched, single A100, batch 11): LLaMA3-8B-Instruct 105.7–138.3 tok/s vs EAGLE-2's 98.2–128.8 (+13% class); Qwen2-7B up to 161.4 vs 138.6 tok/s (+19% on Debug Bench). In the length experiment (EAGLE-1 drafter, L=4, batch 1–50), UCBSpec "closely approaches the (varying) oracle best" and **switches speculation off (γ=0) when inferior to canonical decoding** — the no-spec arm is the principled version of our battery-neutral controller.
- **Verdict: TRANSFERS.** Term moved: **cost-model window sizing → bandit over window sizes with an explicit off-arm**. Our per-position tables become arm means; state is a handful of floats per slot.

### 1.3 vLLM RFC #36657 — DSL confidence-threshold early exit (measured, not merged)
[RFC issue](https://github.com/vllm-project/vllm/issues/36657) (fetched) | [linked PR #35301](https://github.com/vllm-project/vllm/pull/35301). Authors include Intel's jmamou (DISCO).
- Mechanism: after each draft step, batch-mean of the drafter's top-token softmax probability; if below threshold τ, stop drafting early, zero-pad to full width (padded positions rejected by verifier). Mean not min, so "a single uncertain request in a mixed batch does not short-circuit drafting for the whole batch."
- Numbers (fetched, 1×A100 80GB): Llama-3.1-8B+1B mt-bench c=1 **1.77× vs target / 1.44× vs static SD(k=5)**; OPT-6.7B+125M 3.10×/1.64×; random-8K c=8 3.60×/1.52×; mt-bench c=4 1.32×/1.63×.
- Failure modes (fetched): per-step `.item()` GPU→CPU sync per draft step (design doc replaced it with a 1-byte async D→H copy on a dedicated stream); τ manual, 0.4–0.6 recommended starting range, model-pair/dataset dependent. **Closed as not planned (stale) Jul 2026** — the idea lives on in adaptive verification (1.1).
- **Verdict: TRANSFERS (mechanism only).** On our box the draft loop is ours, so the sync problem vanishes — we read draft logits every step already. Term moved: **serial draft depth → confidence-gated early exit**.

### 1.4 DSDE — KLD-stability length control (weak-signal evidence)
[arXiv:2509.01083](https://arxiv.org/html/2509.01083v1) (fetched).
- Mechanism: training-free, per-sequence per-iteration SL from post-verification signals: weighted variance intensity ratio (WVIR) of draft-target KLD over short (N=10) vs long (N=30) windows; predicted SL = (1 − SF·WVIR)·(SL_max − SL_min) + SL_min, SL_min=2; SL_max calibrated online from a short preamble; adaptive batch cap = arithmetic mean of per-sequence predictions.
- Numbers (fetched, 8×A100, LLaMA-3.1-70B+1B): temp 0.0: DSDE 2.75× vs **static-opt 2.86×** vs AdaEDL 2.78× (all trail static-opt). Temp 1.0: DSDE 2.00× vs AdaEDL 2.17×. Robustness win: in the low-acceptance regime (Gemma pair) AdaEDL degrades to 234–335% of static-opt latency while DSDE stays near static-opt.
- Failure evidence (fetched): "All evaluated signals exhibit a weak correlation with token-level acceptance" — draft entropy r = −0.339 (T=0), −0.235 (T=1.0); WVIR only 0.128 / −0.031. "Sampling noise disrupts lagging regional diagnostics" at T=1.0.
- **Verdict: WATCH.** Mechanism is dominated (trails static-opt at T=0, trails AdaEDL at T=1.0), but its **signal-correlation table is the gate for our own controller**: fit per-position acceptance per temperature and treat any entropy/KLD-based controller as a lagging indicator, not a predictor.

### 1.5 Trained acceptance heads — the training-tax family
- SpecDec++ (foundational): acceptance head on the drafter + halting criterion. [OpenReview](https://openreview.net/forum?id=NnExMNiTHw) (search-level). Head training ≈16.7% of total compute.
- OmniDraft (NVIDIA, 2507.02659): [arXiv HTML](https://arxiv.org/html/2507.02659) (fetched). Adds acceptance head à la SpecDec++ with online distillation; documents two drift lessons: joint head+distill updates give higher acceptance, interleaved updates give better speedup because joint "might be underestimating the acceptance probability, leading to wrongful early exit"; head thresholds γ=0.3 (text) vs 0.7 (code) — "relaxing a strict threshold later improves speedup." Vicuna-7B: 2.15× (GSM8K) vs 1.44× static (A100). Core product (one drafter serving cross-vocabulary targets via n-gram cache) is not our problem.
- AGSD / Acceptance-Guided Adaptive SD: online candidate-length selection via acceptance head. [IEEE](https://ieeexplore.ieee.org/abstract/document/11463423) | [PDF](https://zhoufeng6288.github.io/pub/Acceptance-Guided_Adaptive_Speculative_Decoding_for_Efficient_Large_Language_Model_Inference.pdf) (search-level).
- **Verdict: SKIP (new weights + training loop), but IMPORT the lessons free:** heads underestimate acceptance → calibrate our tables/estimator toward optimistic-decay; per-domain thresholds differ ~2×.

### 1.6 Remaining frontier, brief
- DISCO (Intel, NeurIPS 2024): per-iteration classifier decides continue-draft vs hand-off, provably output-identical; ~10% avg over best static SL. [arXiv:2405.04304](https://arxiv.org/html/2405.04304v4) (search-level). The RFC in 1.3 is its production descendant.
- Learning to Draft (ICLR 2026): two co-adaptive RL policies optimizing per-step throughput directly. [OpenReview](https://openreview.net/forum?id=IK9cbzzXLt) | [arXiv:2603.01639](https://arxiv.org/abs/2603.01639) | [code](https://github.com/zhzihao/Learning-to-Draft) (search-level). **SKIP** for us (RL training loop); the framing (optimize accepted-tok/s objective directly, not acceptance rate) is the right objective for our bandit arms.
- PEARL (ICLR 2025): overlaps drafting with verification for adaptive length at negligible draft cost. [ICLR](https://iclr.cc/virtual/2025/poster/29693) (search-level). GPU-compute-bound trick; our draft lane (MTP head + lookup) is not the bottleneck — our verify lane is. WATCH only for the overlap-scheduling idea if the MTP head ever becomes serial-critical.
- Nightjar (Dec 2025): MAB planner selects length per batch size, proactively disables speculation, CPU-offloads drafter under memory pressure; +14.76% throughput / −20.18% latency vs static SD. [arXiv:2512.22420](https://arxiv.org/abs/2512.22420) (abstract fetched; numbers search-level).

## 2. Sampling × speculation interaction

### 2.1 Theory floor — what "lossless" actually covers
- Standard rejection-sampling verification preserves the target distribution **as sampled**, not just greedy (Leviathan et al. [arXiv:2302.01318](https://www.alphaxiv.org/abs/2302.01318)). Yin et al., NeurIPS 2024, gives the general sampling-theoretic treatment. [Proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/e7349e785900b93d8b4971a3f2c1cefe-Paper-Conference.pdf) (search-level). BanditSpec Prop 3.1 re-proves it for adaptive length ([fetched](https://arxiv.org/html/2505.15141)).
- Greedy caveat: strict greedy-output equality holds only "up to numerical precision" — kernel nondeterminism flips argmaxes. The MLSys 2026 audit measured real divergence: output-length distributions differ across batch sizes even at temperature 0 ("baseline 737±1048 at batch 1 vs 674±784 at batch 32"), differences "not consistently longer or shorter." [arXiv:2601.11580](https://arxiv.org/html/2601.11580v1) (fetched).
- **min_p boundary (named, no paper found):** no 2025-26 paper studies min_p × speculation interaction specifically. Theory verdict: the accept test is exact for whatever target distribution the verify pass computes — our verify rows already produce full target logits per position, so min_p truncation applied identically at draft-accept and residual-resample preserves the min_p-transformed distribution exactly. Verify-order marginals are a non-issue because we never reconstruct the marginal from verify order; we hold per-row logits. Residual risk: none identified; this is a verified-theory boundary, not a measured one.

### 2.2 Block verification — the lossless acceptance rule that gains WITH temperature
[arXiv:2403.10444](https://arxiv.org/html/2403.10444v3) (fetched) | ICLR 2025, Google (Sun, Mendlovic, Leviathan, Aharoni et al.) | [OpenReview](https://openreview.net/forum?id=frsg32u0rO).
- Mechanism: verify the entire drafted block jointly — running acceptance probability per prefix `p_i = min{p_{i-1}·M_b(X_i)/M_s(X_i), 1}`, per-sub-block thresholds, commit the **longest** accepted sub-block (token verification commits to first rejection), correction token sampled from the block-residual distribution. Needs only the per-row target/draft probabilities the verify pass already computes: "a plug-and-play replacement" with "no additional computation or code complexity costs."
- Theory (fetched): Theorem 1 exact distribution preservation; **Theorem 2: for any valid verifier, E[BlockVerify accepted length] ≥ E[Verify]** — dominance, not parity. Caveat: at temperature 0 it "degenerates to token verification" (zero gain, zero loss).
- Numbers (fetched, batch 1): PALM-2-S + PALM-2-XXS, γ=8: block-efficiency +7.00–10.06% (avg 8.30%), wall-clock +5.36–8.14% (avg 6.49%). Vicuna-7B + 68M, H100, γ=8: up to 8.7% block-efficiency / 6.7% wall-clock. **Gains rise with temperature** (T=1.0: 6.07% wall-clock vs T=0.2: 1.66%) and with draft length γ; "independent of the batch size."
- **Verdict: TRANSFERS — top pick at fixed row budget.** Our cap is 8 rows; block verification raises accepted-tokens-per-window **without touching the row budget or the expert-load schedule**. Expect the +5–8% class on sampled lanes (temp 0.7/1.0), concentrated where acceptance dips (live deep traffic); zero gain on edit-phase greedy lanes. Term moved: **verify acceptance rule**.

### 2.3 Temperature is the dominant acceptance covariate
[Temperature-Centric Investigation of Speculative Decoding](https://arxiv.org/html/2410.10141v1) (fetched).
- Speedup declines monotonically T 0→1: "relative difference of around 30%" (2.23× → 1.72×, Alpaca); out-of-domain (GSM8K) gap reaches **42%**; draft-target temperature mismatch costs up to 24%.
- Consequence for us: our per-position acceptance tables must be **keyed by temperature** (we run 0.7/1.0); a table fitted at one temperature silently mispredicts at the other. Our live 88–94% at T=0.7/1.0 is consistent with this paper's regime. Also: their speedup formula is the same family our cost model uses — `E(speedup) = (1−α^(k+1)) / ((1−α)(kc+1))` with α acceptance, k length, c draft/target cost ratio ([Illusion paper, fetched](https://arxiv.org/html/2601.11580v1)).

### 2.4 Lossy verification family — quality cost, for the gate
- Judge Decoding: 16.4k-param linear judge on target embeddings accepts "contextually correct" mismatches; H100 batch 1, 8-bit: 8B/405B 3.9× (129.3 tok/s), acceptance length 19.7 vs 6.3. **Quality cost measured:** in-distribution parity on GSM8K/HumanEval/MMLU, but OOD (judge trained without code): HumanEval **86.6% → 80.4%**; "the mathematical guarantee to maintain target quality is lost." Evaluated greedy-only; sampling interaction "unexplored." [arXiv:2501.19309](https://arxiv.org/html/2501.19309v1) (fetched). Follow-ups: [AutoJudge 2504.20039](https://arxiv.org/html/2504.20039v1), [SelfJudge 2510.02329](https://arxiv.org/abs/2510.02329), [Training-Free Per-Step Lossy SD 2609.02897](https://arxiv.org/html/2609.02897v2) (search-level).
- **Verdict: SKIP.** Our min_p sampled lanes require exactness; the one measured regression (−6.2 pts OOD) is the exact failure mode we gate against. If we ever want a lossy lane, the gate is: train per-domain, measure OOD, cap blast radius to non-code lanes.

## 3. Verify-window economics at fixed row budget

### 3.1 Block verification — dominance proof (see 2.2)
The only scheme in this literature with a **proved dominance over token-wise verification** at identical rows and compute. Theorem 2 is the license: at our 8-row cap, switching the accept rule is pure upside on sampled lanes.

### 3.2 SpecServe/AdaSpec — confidence-prior row pruning + load-aware reversion
[arXiv:2503.05096](https://arxiv.org/html/2503.05096v1) (fetched) | SoCC 2025 as AdaSpec ([ACM](https://dl.acm.org/doi/10.1145/3772052.3772239), search-level).
- Mechanism: two-level controller. (a) Drafter-level: acceptance modeled as product of per-token draft confidences; time model quadratic in SL — "there exists a unique optimal speculative length" maximizing throughput; step-level "predict-execute-correct." (b) Request-level **Confidence Prior Verifier**: "greedily eliminates low-acceptance draft tokens while estimated throughput improves" — prune doomed rows before verify. SLO gate: if estimated step time exceeds the TPOT limit, throughput is "set to −1 to prevent the adaptive drafter from proceeding" → revert to autoregressive.
- Numbers (fetched, L40/A800/8×A800): 1.14–14.3× vs other speculative systems, 1.1–1.8× vs autoregressive; SLO attainment +5–38%. Ablation: 3.17 avg draft tokens vs 5.67 (threshold-based), acceptance 0.40 vs 0.20, **speedup 1.38× vs 0.85×** — pruning wastes fewer rows AND lifts per-row acceptance.
- Collapse theory (fetched): throughput vs SL declines when `g′(0)c − b < 0` — smaller acceptance, larger batch, longer context, larger draft-relative cost. Vanilla fixed SL=1/3/5 fails to beat autoregressive in 7%/13%/40% of setups; SpecServe never below 1.0× (reverts to AR). "Verification can be up to 90% of the total time."
- **Verdict: TRANSFERS.** Term moved: **rows-per-verify at fixed budget** — drop low-confidence draft tokens before they load experts. Our per-position tables already provide the confidence product; the missing piece is the per-request prune + the explicit revert-to-no-spec arm under load (see BanditSpec γ=0, 1.2).

### 3.3 AdaEDL — entropy-gated early stop of the draft chain
[arXiv:2410.18351](https://arxiv.org/html/2410.18351) (fetched) | ICLR 2025, Qualcomm.
- Mechanism: stop drafting when `1 − γ·H_DM(x) < λ` (H = draft logits entropy, γ=0.2, λ tuned online: EMA of acceptance vs target α=0.9, ±ε steps, smoothed). Justified as "an approximate lower bound on the expected acceptance rate" via Pinsker/TVD. "Training and parameter-free"; cost O(V) on logits we already materialize.
- Numbers (fetched, single A100 80GB): +10–57% over static draft length; Pythia-6.9B: 46.4 TPS (56% speedup) vs 32.2 TPS static-best; at T=1.7 static spec decode "drops 57%" below autoregressive while AdaEDL "still provides an 8% boost."
- **Verdict: TRANSFERS — the direct answer to the MoE drafting tax.** Every draft row we don't emit is an expert load we don't pay. Entropy is computed on MTP logits already in registers. Term moved: **serial draft depth per step**.
- Corroboration: SVIP (EMNLP 2025) — draft-entropy spike detection, training-free: +17% on MT-Bench 8K context, +22% on QwQ long-form vs fixed lengths. [arXiv:2411.18462](https://arxiv.org/abs/2411.18462) | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.844) | [code](https://github.com/geralt-targaryen/svip) (abstract fetched; hardware not stated).

### 3.4 Sequoia — row-budget topology, for the day the 12-row family lands
[arXiv:2402.12374](https://arxiv.org/abs/2402.12374) | [HTML](https://arxiv.org/html/2402.12374v3) | [Together blog](https://www.together.ai/blog/sequoia) (search-level).
- Mechanism: DP over tree topologies at a fixed token budget + hardware-aware size/depth optimizer (verification is parallel; spend budget on shape, not length). +5–30% over prior tree methods **with a fixed token budget** on A100; up to 4.04× (Llama2-7B).
- **Verdict: WATCH.** Distinct from queued "dynamic tree verify scheduling" (offline topology choice, not runtime scheduling), but it lands in the same blocked place: our 8-column activation family. Revisit with the 12-row headroom work.

## 4. Draft-model quality without new weights

### 4.1 Entropy-gated early stop (see 3.3) — the highest-leverage "draft quality" move is not drafting doomed rows at all.

### 4.2 SwitchSD — copy-intent probe routes lookup vs neural drafter per step
[arXiv:2609.20186](https://arxiv.org/html/2609.20186v1) (fetched).
- Mechanism: a strictly linear probe on target hidden states predicts **copy-intent** ("p_ℓ(j)=σ(w_ℓ^⊤ H_j^(ℓ))", post-attention layers best, peak at intermediate layer, "AUC >0.99"); when probe > τ (F1-optimal ≈0.4) AND a κ=5-gram repetition exists, switch to copy-based drafting (W=10 candidates × D=20, block-triangular parallel verify); else neural drafting (EAGLE-3). Adjusts drafter per step, not just length.
- Numbers (fetched, H100): up to 2.58× on HumanEval vs vanilla (98.63 tok/s); +15% class over EAGLE-3; CNN/DM 1.83×, Math500 2.03×; still 1.54× at T=1.0; "preserving exact generation quality" (lossless — verify unchanged).
- **Verdict: TRANSFERS (with a training asterisk).** We already own both drafters (lookup + MTP). A per-step router between them is the mechanism; the probe needs a small supervised fit (linear; Judge Decoding's judge was "16.4k parameters on just 30k tokens in less than 1.5 hours" — same cost class). Term moved: **drafter selection per step**. Even a feature-free heuristic (n-gram hit depth ≥ k → lookup-only rows) captures part of it with zero training.

### 4.3 Position-level drafter oracle — the headroom number
[arXiv:2601.11580](https://arxiv.org/html/2601.11580v1) (fetched, MLSys 2026 oral): Oracle_Combine picks, per token position, the longer-accepted span between EAGLE and nn-gram — "achieving up to a 2.2× further speedup" over the best fixed strategy; the adaptive-combination headline is 4.9× end-to-end vs no-SD (simulated, perfect predictor). Figure 10 shows per-position advantages **alternate** between copy and neural sources. This is the upper bound that justifies 4.2's router.

### 4.4 KV-cache reuse between draft and target
[arXiv:2604.26412](https://arxiv.org/html/2604.26412v1) (search-level): "KV-Reuse Hypothesis" — draft reusing target KV gives richer long-horizon signals; reusing target hidden states alone is "biased context compression." Our MTP drafter is already target-conditioned; the unexplored variant is feeding **target KV** (not just last hidden state) into the MTP head at long context. WATCH — costs bytes on the DRAM-bound lane; gate on measured long-context acceptance decay.
- Production adjacent: Arctic Inference suffix decoding + shift parallelism in vLLM: 4× on SWE-Bench agent traffic, 2.8× decode, 1.7–1.8× on gpt-oss reasoning. [Snowflake blog](https://www.snowflake.com/en/blog/engineering/fast-speculative-decoding-vllm-arctic) (search-level) | [paper 2507.11830](https://arxiv.org/html/2507.11830v1) | [repo](https://github.com/snowflakedb/ArcticInference).

### 4.5 Grammar/schema-aware drafting
- FactorSpec (Nov 2026): compiles JSON Schema + relations into a factor graph; "an exact constraint is itself a drafter" — determined tokens get a single-token mask, "the target accepts it with probability one"; up to 10.5 accepted tokens/verify, 1.43× faster than per-load-tuned speculation, 13.9× valid records/GPU-second. Cost: "about eight host cores per GPU" for the constraint service. [arXiv:2611.02297](https://arxiv.org/html/2611.02297v1) (fetched).
- **Verdict: WATCH.** On our box 8 host cores compete with the expert-staging lane; a cheaper variant (mask the lookup drafter's candidates by the active FSA state, no relational solver) is buildable from our existing constrained-decode path. No paper surfaced doing exactly that cheap variant — named gap.
- Context: [Draft-Conditioned Constrained Decoding 2603.03305](https://arxiv.org/pdf/2603.03305) (search-level), [Grammar-Aligned Decoding / ASAp](https://openreview.net/forum?id=L6CYAzpO1k) (search-level) — note ASAp warns naive constrained decoding distorts target semantics; any grammar-drafting lane must reuse the verify pass, not mask-and-hope.

### 4.6 Prompt-conditioned draft biasing — boundary named
No 2025-26 paper surfaced that biases a **learned** drafter's logits with prompt-retrieval evidence (the exact "prompt-conditioned draft biasing" ask). What exists: prompt-lookup/n-gram drafting as a separate drafter ([vLLM docs](https://docs.vllm.ai/en/v0.5.4/models/spec_decode.html), search-level) and the router/oracle evidence in 4.2–4.3 that says the right architecture is per-position **routing**, not logit mixing. Term stays: drafter routing, not biasing.

## 5. Production controllers — what ships today

### 5.1 vLLM
- Shipped: adaptive verification (online estimator + survival budget, [docs](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification), fetched — see 1.1); `speculative_disable_by_batch_size` cutoff flag ("disables speculative decoding for new incoming requests if the number of enqueued requests exceeds this value", [docs](https://docs.vllm.ai/en/latest/features/speculative_decoding), search-level).
- Measured collapse points (Squeezebits, vLLM v0.6.3, Llama-3.1-70B TP4 on 4×A100): benefit held to concurrency ≈32 at 1K input, dropping to ≈16 at 2K input — "the concurrency threshold at which speculative decoding remained beneficial decreased from 32 to 16 as the input length increased from 1K to 2K"; draft-token sweet spot 4, "when number of draft tokens exceeds 5, throughput begins to degrade"; acceptance fell 54.0% → 50.9% from 1K→2K input. [blog](https://blog.squeezebits.com/vllm-vs-tensorrtllm-11-speculative-decoding-37301) (fetched).
- Not shipped: RFC #36657 DSL (closed stale — see 1.3).

### 5.2 SGLang
- Shipped: [Adaptive Speculative Decoding](https://lmsysorg.mintlify.app/docs/advanced_features/adaptive_speculative_decoding) (fetched): EMA of accepted-draft-length per batch-size slot ("observations at small BS don't pollute the large BS signal"), tier ladder over pre-captured CUDA-graph shapes ("runtime switching is a reference swap, not an online graph recapture"), hysteresis (down −0.25 / up 0.0), warmup 10, update interval 5; policy: "target_steps ≈ clamp(round(ema_accept_len) + 1, min, max)". Presets: conservative BS 8–31 → steps [1,3], BS≥32 locked to [1]; aggressive ladders [1,3,7] with `ceiling_coeff`. Caveats: "at high BS, narrow ladders ([1,2] or [1]) often beat wide ones"; GPU memory scales with candidate steps; stable workloads gain nothing.
- RFC AutoSpec (auto-tuning at runtime): [issue #15319](https://github.com/sgl-project/sglang/issues/15319) (search-level); roadmap: [issue #23705](https://github.com/sgl-project/sglang/issues/23705) (search-level).

### 5.3 TensorRT-LLM
- v0.14 measured: "speculative decoding achieves higher throughput than vanilla decoding only when the maximum concurrency is set to 1" — flat spec throughput at concurrency 2/4/8 while vanilla scales. [Squeezebits](https://blog.squeezebits.com/vllm-vs-tensorrtllm-11-speculative-decoding-37301) (fetched). Docs: [speculative-decoding.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md) (search-level). ReDrafter (beam + dynamic tree attention, up to 3.5×): [Apple](https://machinelearning.apple.com/research/redrafter-nvidia-tensorrt-llm) (search-level).
- Baseten describes dynamic per-batch draft sizing on acceptance rate ([blog](https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm)) — page is JS-walled, could not verify text (named boundary).

### 5.4 Collapse synthesis vs our box
GPU stacks collapse under concurrency because verify becomes compute-bound: EAGLE GSM8K speedup 1.73× → 1.21× as batch 1→128; 70B loses 14% vs 8B's 4.3% growing batch; "verification stage generally takes the largest execution time, ranging from 42% to 95%" ([Illusion, fetched](https://arxiv.org/html/2601.11580v1)). SGLang's own preset locks depth to 1 at BS≥32. Our measured regime — T 3.1–3.9 at 6-way with +35% class sustained — sits far below those collapse points because our verify is DRAM-byte-bound, not SM-bound: our collapse term is **extra rows × expert loads**, which is exactly why row-pruning (3.2), early-stop (3.3), and fixed-row accept-rule changes (2.2) are the levers that transfer, and batch-parallelism tricks are not.

## Ranked verdicts for our box (expected accepted-tok/s at measured baseline: α 0.88–0.94, T 3.1–3.9, 8-row cap, temp 0.7/1.0)

| # | Mechanism | Term moved | Expected effect | Evidence |
|---|---|---|---|---|
| 1 | Entropy-gated early stop of draft chain (AdaEDL/SVIP class) | serial draft depth; MoE drafting tax | +10–20% window-time class at equal accepted tokens in deep/low-accept phases; free on logits we already hold | [AdaEDL](https://arxiv.org/html/2410.18351) fetched; [SVIP](https://arxiv.org/abs/2411.18462) |
| 2 | Online logistic acceptance estimator on draft logits (vLLM-shipped shape) + survival-product row budget | per-position tables → drift-proof online estimator; rows-per-verify | replaces table refits; RFC-DSL-class gains 1.44–1.64× vs static SD measured on A100 | [vLLM docs](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification) fetched; [RFC #36657](https://github.com/vllm-project/vllm/issues/36657) fetched |
| 3 | UCBSpec bandit over window sizes with γ=0 off-arm | cost-model controller → drift-adaptive bandit + principled no-spec arm under load | +13–19% class vs tuned static at batch 11 (A100); regret-bounded | [BanditSpec](https://arxiv.org/html/2505.15141) fetched |
| 4 | Block verification accept rule on sampled lanes | verify acceptance rule at FIXED row budget | +2–8% accepted-per-window, provably ≥ token-verify (Theorem 2), zero extra compute, zero gain on greedy edit lanes | [Block Verification](https://arxiv.org/html/2403.10444v3) fetched |
| 5 | Copy-intent router between lookup and MTP drafter (SwitchSD shape) | drafter selection per step | +15% class over tuned static drafter (H100); headroom 2.2× further per oracle; needs tiny linear-probe fit | [SwitchSD](https://arxiv.org/html/2609.20186v1) fetched; [Oracle_Combine](https://arxiv.org/html/2601.11580v1) fetched |

SKIP list with reasons: Judge-family lossy verification (measured OOD regression 86.6→80.4; our sampled lanes need exactness); trained acceptance heads (weights+training tax; import the underestimation/threshold lessons instead); DSDE KLD control (trails static-opt at T=0 and AdaEDL at T=1.0; keep only its signal-correlation gate); OmniDraft cross-vocab distillation (not our problem); Learning-to-Draft RL (training loop; keep the accepted-tok/s objective framing); Sequoia/FactorSpec (blocked on 8-column activation family / 8 host cores, revisit respectively).

Boundaries named: min_p × speculation has no dedicated paper (theory says exact; we never reconstruct marginals from verify order — full per-row logits are held); prompt-conditioned logit biasing of a learned drafter did not surface as a mechanism — the literature converged on per-position routing instead; Baseten's TRT-LLM dynamic-sizing blog is JS-walled and unverified.
