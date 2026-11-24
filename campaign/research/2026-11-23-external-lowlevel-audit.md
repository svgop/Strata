# External audit — low-level residual / hyper-connection engineering (subagent 027ab2f8, 2026-11-23)

Provenance: read-only web-research subagent. All IDs fetched and verified from arXiv by the agent;
unverified items listed at the end. Banked verbatim; numbers are the agent's.

## Citation integrity first

**arXiv 2409.03756 is NOT hyper-connections** (it is a hypergraph-spectral math.PR paper).
Hyper-Connections is **arXiv 2409.19606** (Zhu et al., ByteDance Seed, ICLR 2025). Correct every
downstream reference.

## Verified citations

- **2409.19606 Hyper-Connections**: residual stream → n parallel streams; per layer HC matrix
  [[0,B],[A_m,A_r]] — block input h0 = A_mᵀH (weighted READ mix), block output broadcast to ALL
  streams via B, streams cross-mixed via A_r; **final output = row-wise SUM of streams**. DHC:
  A_m, A_r, B = static init + s∘tanh(norm(H)W), input-dependent, tanh-bounded. Init: B = 1_{1×n}
  (broadcast write), A_m = cyclic one-hot e_{k mod n} (stream k mod n is the primary carrier read
  by block k), A_r = I. OLMoE-1B-7B DHC×4: 1.8× faster convergence, ~+6 pts ARC-Challenge at 500B
  tokens. Our 4-stream model is the DHC×4 config family.
- 2503.14125 Frac-Connections: partition hidden into parts instead of n× width.
- 2512.24880 mHC (DeepSeek team): unconstrained HC "compromises the identity mapping property" →
  instability; constrains A_r to doubly stochastic (Sinkhorn).
- 2610.06653 (Birkhoff geometry): doubly stochastic mixer = MEAN channel (exact residual net) +
  DIFFERENCE channel contracted by σ2 — "extra width is a fading memory, horizon 1/(1−σ2)".
- 2609.02672 oHC: unconstrained mixing rescales streams by an unbounded, layer-compounding factor.
- 2611.23026 TPHC: each sublayer writes a single output to all streams; >4 streams helps little.
- 2611.05984 SimpleHC: learned read/write GATING alone ≈ full HC; gating and mixing are the two
  effective interaction groups.
- 2611.24567: mHC streams specialize (read/write gates diverge late); benefit survives A_r=I →
  read/write ROUTING, not mixing, is the functional axis.
- Heimersheim & Turner 2023 (no arXiv): residual ‖h‖ grows ~×1.045/layer (gpt-2-xl); LayerNorm
  makes cancellation hard, overshadowing easy.
- 2308.10248 ActAdd; 2312.06681 CAA; 2306.03341 ITI; 2406.11717 Arditi; 2507.21509 persona vectors;
  2406.15518 KTS — as in the methods audit.
- **2509.09660 SteerMoE (ICLR 2026)**: behavior-linked experts found by activation-frequency
  contrast on paired inputs; force (de)activate at inference; safety +20%, faithfulness +27%, no
  finetuning.
- **2606.04160 Decomposing Refusal Steering in MoE**: residual steering transfers to MoE unimpeded
  by routing; a SINGLE expert at its own chosen location recovers 78% of full steering effect; only
  54% when pinned to the full-layer location.
- **2608.21236 RARE (COLM'26 interpretability workshop)** — THE key MoE result for this rig:
  routing is more sensitive to SEMANTIC than behavioral content; projecting the perturbation onto
  the NULL SPACE of the router matrix substantially recovers steering: harmfulness ASR 53.3% while
  retaining 67.8% MMLU; TruthfulQA MC1 41.0→58.6; CounterFact efficacy 16.8→96.3%.
- 2202.05262 ROME; 2210.07229 MEMIT — weight-edit lane, needs weight surgery (violates no-retraining
  constraint).

## Geometry verdicts

- **V1**: "λ-residual vs gates" framing is wrong. Streams have no fixed roles; the functional roles
  are the matrices: B (write broadcast), A_m (read mix — this IS the gate), A_r (cross-mixing).
- **V2**: One-shared-direction-to-4-streams is defensible only as mean-channel injection and is
  **structurally wrong on dose**: block l+1's input dose = (Σ_j A_m_j)·v ≈ 1×v (weights ~convex);
  **final read-out (row-sum of streams) sees ~4×v** — four times a dense model's read-out dose at
  equal α; the vector is then re-mixed by A_r every layer (unconstrained compounding risk, oHC/mHC
  critiques). Keep the SAME direction in all streams — **per-stream differential vectors are
  architecturally damped (mHC family) or chaotically mixed (DHC): wrong tool** — but renormalize
  dose per the read/sum structure (divide α by ~4 for read-out-matched dosing).
- **V3**: "add after block l" ⇒ first affected computation is block l+1's A_m read; literature
  "steer at layer L" = post-block L = our direction.[L] = dump slot L+1. Our dump-slot-k ←
  direction.[k−1] convention matches the literature exactly.
- **V4**: Facts-fail-first has a named mechanism (RARE): router gates read semantic content off the
  residual, so router-visible steering components corrupt expert selection BEFORE persona
  saturates; ×1.045/layer norm growth mis-calibrates any fixed raw dose across depth.
- **V5**: MoE steering is not an empty lane (SteerMoE, Decomposing-Refusal, RARE). Router-logit-bias
  steering: no published work found.

## Ranked engine tweaks (no retraining; reuse existing machinery)

- **T1 — Read-out-matched dose renormalization**: scale every direction row by 1/4 (or per-layer
  A_m rows if present); add mode; loader-only. Expected up to ~4× dose headroom. Risk ~zero.
  Falsifier: one interleave, broadcast α vs α/4, persona eval + MMLU-Pro gate on both curves.
  (Lab form: no engine change — the ladder dose axis is the multiplier.)
- **T2 — Router-null-space projection (RARE port)**: loader applies P = I − Rᵀ(RRᵀ)⁻¹R per MoE
  layer to each cvec row at load (one GEMM/row, once); kernel untouched. Falsifier (free):
  ‖R·v‖/‖v‖ per layer on existing tables; then one boot with router-logit KL on ~50 fixed prompts.
- **T3 — Norm-calibrated dose (RMS prior)**: per-layer per-stream RMS calibration constant; flattens
  dose-response across depth. Falsifier: ‖h_l‖ histograms from existing dumps.
- **T4 — Mid-band restriction + single-layer sweep**: zero rows outside [l0,l1]; concentrates dose
  where persona directions act; cuts early routing corruption. Falsifier: three single-layer boots
  (⅓, ½, ⅔ depth) at fixed α.
- **T5 — Expert-mask steering channel (SteerMoE port)**: offline contrastive activation-frequency
  diff → per-layer top-k persona experts; engine masks them per request on the existing
  expert-profile machinery. ZERO residual-norm perturbation — lowest capability risk of all lanes.
  Falsifier: contrastive diff on 100 paired pairs with existing profile counters; clean top-k
  separation = GO.

## Unverified (do not quote as fact)

Per-stream norm magnitudes of trained HC (2409.19606 App. D visualization-only); Arditi's exact
layer band; ActAdd's injection-layer number; router-logit-BIAS steering (absence verified at
title/abstract level only); no paper isolating pre-attention vs post-MLP injection points; NO
published work on steering hyper-connection streams (all 37 "hyper-connections" arXiv entries
checked) — this rig's results are novel evidence.
