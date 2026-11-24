# External audit — activation-steering methods (subagent 4aa6065f, 2026-11-23)

Provenance: read-only web-research subagent, commissioned with the cvec geometry (one GGUF table
per boot, per-layer row `direction.<l>` + scalar, same vector into all 4 hyper-connection streams).
All IDs fetched and verified against abstracts by the agent; corrections it issued to my own priors
are marked. Banked verbatim; numbers below are the agent's, not re-measured locally.

## 1. Verified citation table

| ID | Method (one-line mechanism) | Reported effect |
|---|---|---|
| 2312.06681 CAA (Panickssery et al.) | Mean-difference of residual activations over hundreds of same-prompt contrastive pairs; add α·v at all post-prompt tokens, one layer | Full steering of 7 behaviors on Llama-2 7B/13B at multiplier ±1; optimal layer 13 (7B), 14–15 (13B); larger multipliers degrade text quality; MMLU ~unchanged at ±1 (0.63→0.61/0.62); beats system prompts; vector gen <5 min/1 GPU |
| 2406.11717 Arditi et al., refusal direction | Difference-in-means per (layer, position); ONE vector selected by held-out validation (ablate→bypass, add→induce, minimal side change); add at a single layer | Refusal is 1-D across 13 chat models ≤72B; 128 train/32 val pairs suffice |
| 2308.10248 ActAdd (Turner et al.) | Single prompt-pair difference, one position, fixed multiplier | SOTA sentiment-shift/detox on OPT/LLaMA; CAA found it non-robust across behaviors/prompts |
| 2306.03341 ITI (Li et al.) — **correction: not 2303.00867** | Per-head truthful directions from probes; shift activations along top-k head directions with tuned strength (+bias correction) | Alpaca TruthfulQA 32.5%→65.1%; explicit truthfulness/helpfulness tradeoff |
| 2310.15213 Function vectors (Todd et al.) | Attention heads transport a compact task vector; extracted by causal mediation, added zero-shot | Strong causal effects in MIDDLE layers; FVs compose by summation |
| 2507.21509 Persona vectors (Chen, Arditi, Sleight, Evans, Lindsey — verified real) | Automated DiffMean directions for personality traits; deployment-time monitoring; post-hoc and PREVENTATIVE steering; data flagging | Both intended and unintended post-finetune personality shifts correlate with movement along persona vectors; preventative steering avoids shifts |
| 2310.01405 RepE (Zou et al.) | Population-level reading vectors (PCA/mean-difference pairs) | Wide safety-relevant control |
| 2410.04962 Activation Scaling (Stoehr et al.) — **correction: not 2408.08214** | Gradient-learned SCALARS on existing activation vectors; scalar a learnable function of the vector itself | Effectiveness+faithfulness ≈ steering vectors with more minimality; norm/context-dependent scaling |
| 2406.15518 "Steering Without Side Effects" (Stickland et al.) | Fine-tune to minimize KL(steered‖unsteered) on benign inputs, then steer | Prevents 44% of jailbreaks at MT-Bench ≈ parity |
| 2411.02193 SAE-TS (Chalnev et al.) | Measure a steering vector's causal effect via SAE features; build vectors targeting wanted features | Better effect-vs-coherence balance than CAA and raw SAE-latent steering |
| 2501.17148 AxBench (Wu et al.) | Large steering/detection benchmark, Gemma-2-2B/9B | **Prompting outperforms ALL representation-steering methods**; DiffMean best at detection; SAEs not competitive; ReFT-r1 competitive+interpretable |
| 2311.06668 In-Context Vectors — **correction: not 2311.06685** | One vector from demonstration-pair latent shifts; tunable magnitude | Beats ICL and fine-tuning on safety/style/role-play; vectors compose arithmetically |
| 2306.03819 LEACE (Belrose et al.) | Closed-form whitening-based linear concept erasure, every layer | Provably blocks all linear classifiers on a concept, minimal-norm |
| 2611.01496 Sweep Less, Steer More | d′ as cheap layer selector, incl. open-ended generation | d′ identifies performant layers without sweeps; MCQA conclusions transfer inconsistently to open-ended |
| 2611.01493 TopoSteer | Layer selection from displacement-cloud geometry (MST/H0), not separability | Separability and displacement-geometry criteria can rank layers OPPOSITELY; no held-out validation needed |
| 2610.41661 When Steering Shifts Discourse | Side effects beyond perplexity: formality, evidence, ambiguity | Counter-vector targeting the affected dim restores ~78% of unintended changes, keeps ~93% of intended |
| 2611.21655 Depth-aware persona steering | Distribute persona vector across layers with learned depth weights | Cross-layer distribution beats single-layer; effect monotone in window 0.3–0.7 then collapse (617→306→25 chars); Qwen3-family tested |
| 2608.16650 PAS | Decompose residual into PCA trait subspace + orthogonal rest; distinct scaling exponents | Monotonic bidirectional control (ρ=+0.92), 15.4% vs 8.7% baselines; decomposition + layer selection each essential |
| 2611.18274 Right Token, Right Time (CFP/CLABS) | Per-token steering budget allocation; closed-loop gating | Steering at EVERY token displaces from trained distribution; allocation of "when" is decisive |
| 2611.03518 VIAS | Value-isolating contrast (same scenario, one value omitted) | Naive multi-value contrastive vectors are nearly identical — a general compliance signal, not the value |
| 2610.08559 Latent bias directions = confidence | What a contrastive direction encodes | "Success" can be reduced confidence/abstention, not the trait — cautionary for single-direction claims |
| OpenReview fUCPq5RvmH MERA (ICML 2025) | Calibrate WHEN and HOW MUCH to steer; abstain when unconfident | Non-degrading error correction; composes over other steering |
| OpenReview OpuPBNcQwe DIRECTER (ICLR 2026) | Dynamic strength via plausibility-guided decoding (KV-cache scaling) | Up to +6.5% IFEval without quality tradeoffs |
| OpenReview 2mSyg4k2r7 Orthogonal/Parametrized Multi-Attribute Steering | Rescale to each vector's steerable interval; Gram-orthogonalize multi-trait vectors | Orthogonalization decouples traits, prevents interference/oversteering |

## 2. Fit verdicts vs our engine

FITS UNCHANGED: CAA (α·v per layer is exactly our row format), Arditi add-mode (their Eq. 3 is
x+r at one layer = s=1, v=raw DiffMean), ActAdd/RepE/ICV, persona vectors, function vectors, ITI
approximately, d′/TopoSteer (row selection only), PAS/Activation-Scaling construction-side,
whitened/PCA directions (whitened v̂ is just another row).

NEEDS ENGINE CHANGE: (a) directional ablation x−v̂v̂ᵀx / LEACE scrubbing (projection op, not
addition); (b) per-token gating/masks; (c) context-dependent scalar s(x,t); (d) per-head
interventions; (e) KL-then-steer and preventative steering (training-time).

GEOMETRY HAZARD (literature gap — no paper covers 4-stream broadcast): effective read-out dose may
be a multiple of the literature's assumption; price rho in effective-residual units. (Follow-up
audit 2026-11-23-lowlevel: the read-out is the row-wise SUM of streams → ~4× at equal α.)

## 3. Ranked next moves (agent's ranking, condensed)

1. **Causal layer selection before multi-layer arming** (Arditi protocol; TopoSteer proves
   separability rankings can be opposite to causal effectiveness). Cheapest falsifier: d′ ranking
   from existing dumps vs one causal-validation ranking on a single trait.
2. **Effective-dose audit of the 4-stream addition** — compute per-layer read/sum weights, restate
   standing rho as effective-residual fraction. Zero boots. (Executed 2026-11-23; see
   `../results/vec-2026-11-23/02-r03-capture/REPORT.md`.)
3. **Percentile dosing + mid-depth ramp** (CAA mid-band 0.36–0.41 depth; 2611.21655 monotone
   window 0.3–0.7, collapse beyond). Falsifier: offline KL grid on existing dumps.
4. **Persona-vector suite** (2507.21509 recipe) for the target traits; falsifier: |cos| against
   refusal/capability vectors from existing dumps — entanglement kills the arm pre-boot.
5. **Institutionalized side-effect battery**: discourse dims (2610.41661), pairwise-vector cos
   confound checks (VIAS), MMLU+coherence gates; counter-vector rows are expressible in our table.

## 4. Unverified (agent's explicit list — do not cite)

"Oversteering" (Panfilov, ICLR 2025) not found under that title; "SteerEval" as an LLM
persona-intervention benchmark does not exist (2601.21105 is a recommender benchmark); "MultiICL"
is unattributed shorthand for CAA's many-pairs design; covariance-whitening transfer benefit is
hypothesis-only (nearest: LEACE, PAS); exact ITI hyperparameters not verified.
