# External audit — persona-control alternatives + evaluation methodology (subagent 58d60ff6, 2026-11-23)

Provenance: read-only web-research subagent. Verified against primary sources unless marked
UNVERIFIED. Banked verbatim; numbers are the agent's.

## 1. Verified citations

- 2408.05147 Gemma Scope (Lieberum et al.): JumpReLU SAE suite, all layers/sublayers Gemma-2 2B/9B
  + select 27B layers. Claims coverage/quality, NOT persona-steering results.
- 2406.04093 (OpenAI, Gao et al.): top-k/k-sparse SAEs, scaling laws, 16M-latent SAE on GPT-4 over
  40B tokens; downstream-effect sparsity metric. Abstract reports no steering numbers.
- **2506.19823 (OpenAI/Berkeley, "Persona Features Control Emergent Misalignment")**: SAE model
  diffing pre/post finetune on GPT-4o; "misaligned persona" features; a toxic-persona feature most
  strongly causally controls emergent misalignment; few hundred benign samples reverse it.
  Strongest verified evidence that sparse persona features exist and steer persona.
- 2502.03714 U-SAE: cross-model dictionary — demonstrated on vision models, NOT LLM persona
  steering (premise corrected by the agent).
- 2503.17547 Matryoshka SAEs; 2503.05613 SAE survey. MoE-specific SAE work: small workshop reports
  only; no Gemma-Scope-equivalent suite for any MoE.
- **2507.21509 + Anthropic blog "Persona vectors"**: automated trait-description → contrastive
  prompts → diff-in-means vector (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct). Uses: trait
  monitoring (projection predicts the trait BEFORE the response); finetune trait shifts correlate
  with vector shifts; data flagging (LMSYS-Chat-1M, catches samples an LLM judge missed); post-hoc
  steering reverses a trait BUT "makes the model less intelligent" (their words, MMLU drop);
  **preventative steering** — generate finetuning data while steering TOWARD the trait, train on
  those completions without steering — prevents trait acquisition with little-to-no MMLU loss. So
  vector → generated training data IS supported; it is a training-data intervention.
- Anthropic blog "Evaluating feature steering" (no arXiv): 29 SAE features on Claude 3 Sonnet,
  factor −20..20: shared sweet spot ≈(−5,5) with no capability damage; beyond it sharp collapse
  ("sometimes unusable"). Off-target effects persist. **Appendix 4: prompt engineering showed
  sensitivity/specificity comparable to feature steering (preliminary)**. They note SAE steering
  requires significant compute to train SAEs and edit the residual stream.
- Decoding-time: 2210.15097 Contrastive Decoding; 2309.03883 DoLa; 2105.03023 DExperts (ACL 2021);
  2104.05218 FUDGE; 2312.06681 CAA. **No published result shows persona control at zero capability
  cost at decode time** (gap).
- 2407.18416 PersonaGym: 200 personas/10k questions; GPT-4.1 scored same as LLaMA-3-8B — scale
  doesn't buy adherence. Review: 2406.17624.
- Judge methodology: 2306.05685 MT-Bench (position/verbosity/self-enhancement bias; swap protocol);
  2406.07791 — **swap-balancing alone does not fully remove position bias, even in strong judges**.
- "SteerEval": arXiv 2601.21105 exists but is recommender-domain. De facto standard in steering
  papers: MMLU-style retention gate + targeted behavioral eval + dose sweep (the Anthropic design).
  UNVERIFIED: any benchmark purpose-built for "trait moved, capability intact".

## 2. Verdicts per alternative (fit: no full FT, custom engine, 1×24GB + RAM MoE)

- **SAE steering: POOR fit now.** No SAEs for this custom model; suite-scale training is
  infrastructure-grade; MoE adds per-expert dictionaries; 24GB cannot train a useful suite.
  **Sparse does NOT avoid the capability cliff** — Anthropic's own eval shows the identical
  dose→capability cliff for SAE features; sparse buys attribution, not immunity. If any SAE work:
  ONE small SAE on one mid layer as a MONITOR (projection canary), not a steering mechanism.
- **Decoding-time**: mechanism engine-compatible (logit arithmetic); prompt-pair contrast doubles
  decode FLOPs unless shared-prefix KV is exploited (feasible). Best-of-n + rubric reranking is
  zero-capability-cost by construction at n× generation cost. Persona-specific published evidence:
  none (UNVERIFIED). Cheap to try, shallow ceiling.
- **System prompt/template — the most underrated lever**: Anthropic preliminary: prompting ≈
  feature steering in sensitivity/specificity; persona-vectors blog: system prompts reliably
  activate trait vectors; PersonaGym: absolute adherence stays low even at frontier scale (expect
  register-level wins, not deep-behavior wins). **Concrete upgrade: use the trait-vector projection
  as the objective for a systematic prompt search** (enumerate/evolve system-prompt variants; score
  by vector projection + judge). Components verified; the search method UNVERIFIED as published.
- **LoRA on MoE: conditional GO.** LoRA forgets less than full FT (2405.09673); MoE-LoRA: 2312.09979
  LoRAMoE (adapters + load balancing), 2404.13628 MoLA (layer-wise expert allocation). Router-only
  for persona: UNVERIFIED. Data: few hundred samples suffice (2506.19823 precedent); persona-vector
  data-flagging (2507.21509) screens training data for trait leakage/side effects BEFORE training.
  Shape: QLoRA on attention + shared experts (per-routed-expert adapters awkward when experts swap
  to RAM); trait/anti-trait contrast pairs; vector-projection monitoring during training. Gate
  behind prompt+decoding tiers failing.
- **Mechanism insight**: capability-damage-first is canonical for dense vectors AND sparse features;
  the escape is not a better vector family but a channel that does not perturb the forward pass
  (prompt/template, decode-time, screened small finetune).

## 3. Eval protocol upgrades (ranked, cheap, weak-judge-proof)

1. **Length confound control** (free): judges are verbosity-biased; log length delta per pair;
   length-matched subset; trait effect vanishing at matched length = cosmetics.
2. **Identical-pair anchors as formal gate**: per session, judge K identical pairs → flip rate IS
   the position-bias noise floor; accept persona claims only above the anchor band by a pre-set
   margin (equivalence-test style). 2406.07791: swap-balancing under-corrects — the anchor band,
   not the swap, must carry the decision.
3. **Binary rubric items** (6–10 yes/no criteria per trait), not holistic 1–10; weak judges agree
   far better on binary decisions; score verdict tokens by likelihood if logprobs available.
4. **Early capability tripwires** (by sensitivity): (a) truncation/finish-reason telemetry +
   mean-length drift per dose (free from logs — truncation observed first on this rig);
   (b) 20-item greedy arithmetic/counting micro-suite inside the same interleave window (first
   casualty observed); (c) teacher-forced KL/ΔNLL on ~50 fixed neutral canary prompts vs baseline
   (distribution damage appears before task behavior; we own the logits); (d) trait-vector
   projection on the same canaries (2507.21509: predicts trait behavior in advance) — tripwire,
   not gate; (e) 10-prompt refusal-rate battery.
5. **Retention gate**: keep MMLU(-Pro) formal, run only after tripwires pass; two gates if
   borderline.
6. **Dose-response sweep**: ≥5 doses under ABAB interleave; fit trait-effect vs capability-cost
   curves; report the knee and the damage-first crossover. Anthropic found the sweet spot shared
   across 29 features — measure this model's cliff once, price every future vector against it.
7. **Judge validation as instrument**: one known-positive and one known-negative control pair per
   session; a judge failing a control voids the session's reads (wrapper-validation law).

## 4. Unverified (do not cite as established)

Decoding-time persona control at zero capability cost; router-only LoRA for persona; prompt-search
optimized against an internal persona-vector monitor (components verified, method unpublished);
purpose-built "trait moved, capability intact" benchmark; SAE suites for any MoE at Gemma-Scope
scale.

Key URLs: https://arxiv.org/abs/2408.05147 · https://arxiv.org/abs/2406.04093 ·
https://arxiv.org/abs/2506.19823 · https://arxiv.org/abs/2502.03714 · https://arxiv.org/abs/2503.17547 ·
https://arxiv.org/abs/2507.21509 · https://www.anthropic.com/research/persona-vectors ·
https://www.anthropic.com/research/evaluating-feature-steering · https://arxiv.org/abs/2210.15097 ·
https://arxiv.org/abs/2309.03883 · https://arxiv.org/abs/2105.03023 · https://arxiv.org/abs/2104.05218 ·
https://arxiv.org/abs/2312.06681 · https://arxiv.org/abs/2407.18416 · https://arxiv.org/abs/2306.05685 ·
https://arxiv.org/abs/2406.07791 · https://arxiv.org/abs/2312.09979 · https://arxiv.org/abs/2404.13628 ·
https://arxiv.org/abs/2405.09673 · https://arxiv.org/abs/2601.21105
