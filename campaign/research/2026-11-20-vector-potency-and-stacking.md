# Vector potency v2 and perfect vector stacking — research + design

Date: 2026-11-20. Status: research + design; no engine/config changes; standing recipe untouched.
Related: `campaign/research/2026-11-20-abliteration-softness-vector-analysis.md`,
`campaign/arms/persona-war-r01/README.md`, `campaign/results/abl-2026-11-20/`.

The engine can today hold **one steering operation per layer** (`project` or `add`, one layer range,
one request-wide on/off). Potency is limited by that single slot and by the fact that both armed vectors
are **offline proxies** (a recovered rank-one weight direction; a logit-lens token contrast). The
research below says the gains are in three places: a **measured multi-direction subspace** instead of a
proxy, a **norm-preserving** projection instead of a plain subtraction, and **composition in a joint
subspace** instead of one axis. Two of the three need the stacking change; one does not.

---

## A. What the literature says

Reliability labels: **PR** = peer-reviewed, **PP** = preprint, **CR** = community report.

### A.1 Refusal is a low-dimensional subspace, not a single dimension
- **Gabliteration** (PP, [arXiv 2512.18901](https://arxiv.org/abs/2512.18901)): single-direction
  abliteration leaves residual refusal along secondary directions; extract **k directions by SVD of the
  paired harmful/harmless difference matrix** `D = H_h − H_n` (k typically 1–3, ≥90 % of the variance),
  project out with a **ridge-regularized projector** `P = R(RᵀR + λI_k)⁻¹Rᵀ`, and select layers by a
  separability metric. Their extraction beats mean-difference at equal refusal reduction; logistic
  probes underperform; the pairing is stable across 3–5 shuffles.
- Multi-direction claims are supported independently: Wollschläger et al. (PR, ICML 2025,
  [2502.17420](https://arxiv.org/abs/2502.17420)) find multiple mechanistically independent refusal
  directions; orthogonal non-dominant safety directions are causally relevant
  ([2502.09674](https://arxiv.org/abs/2502.09674), PR); general-context refusal needs ≥5 dimensions
  ([2610.04245](https://arxiv.org/abs/2610.04245), PP).
- Refusal and harmfulness are separately encoded (PR, Zhao et al., NeurIPS 2025,
  [poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/115056)): abliteration removes the
  action, not the recognition — this is why the model still hedges after it stops refusing.

### A.2 Norm-preserving and projected abliteration: the quality lever
- **Norm-Preserving Biprojected Abliteration** (CR, grimjim,
  [blog](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)): conventional
  abliteration is "mathematically unprincipled" because the unit direction carries a magnitude component
  and disturbs weight geometry. Their MPOA: decompose each weight row into magnitude · unit direction,
  ablate only the **directional** component, re-normalise, recombine with the original row magnitudes
  (rank-1, so cheap). Measured on Gemma3-12B: UGI 32.61 / NatInt **21.33** vs baseline 19.58 / 18.72
  and standard abliteration 32.08 / 18.64 — refusal removal **and better reasoning than baseline**.
  Caveats they state: layer choice was heuristic; biprojection returned some refusals (hydra effect);
  magnitude sparsification at 0.995 was needed to separate the direction from outlier activations;
  intermediate math in fp32.
- **Projected abliteration** (CR, [blog](https://huggingface.co/blog/grimjim/projected-abliteration)):
  removing only the mechanistically relevant component (orthogonal to the harmless mean) reduces
  collateral; the harmless direction is per-layer, refusal is near-universal (their "stop sign vs
  driving" argument).
- **Layer selection**: quality = `SNR × (1 − cosine(harmful_mean, harmless_mean))`; a measurement from a
  strong source layer applied to a **chunk** of nearby intervention layers (N-to-M mapping) beat strict
  per-layer application in their experiments. More intervention layers → less self-repair → more
  compliance (hydra effect, [McGrath et al. 2023](https://arxiv.org/abs/2307.09458) cited by them); the
  number of layers is therefore the compliance/capability dial.
- **Heretic** (CR/PR, [repo](https://github.com/p-e-w/heretic)): optimises per-component, per-layer
  weights (`max_weight`, `max_weight_position`, `max_weight_distance`, `min_weight`) against a refusal
  objective + KL; the same refusal rate at far lower KL than flat abliteration. A community Qwen3.6
  v2 report ([card](https://www.toolify.ai/ai-model/wangzhang-qwen3-6-27b-abliterated-v2)) shows the
  per-component split was a *null result* on that model — per-layer weights are the durable part.

### A.3 Composing multiple steering directions
- **ORBIT** (PP, [arXiv 2606.22357](https://arxiv.org/abs/2606.22357)): naive summation of attribute
  vectors suffers **norm imbalance and directional cancellation**; they build a **joint subspace** from
  the per-attribute planes via SVD, then apply a **single norm-preserving rotation** toward the combined
  target, with per-token gating and an optional additive boost. Stronger, more balanced steering than
  CAA/K-Steering with better coherence. Angular steering subsumes CAA (small angle) and directional
  ablation (θ=π).
- **DySCo** (PR, IEEE TDSC 2026,
  [paper](https://www.computer.org/csdl/journal/tq/5555/01/11735080/2kB9WgiTXzO)): each attribute as a
  low-rank subspace; "compose multiple attributes by direct vector addition can collapse context-
  dependent attribute variations and introduce cross-attribute interference".
- **Interference-Aware Composition of Steering Vectors** (PR, Findings EMNLP 2026,
  [PDF](https://aclanthology.org/2026.findings-emnlp.1024.pdf)): composition must account for the
  geometry of the directions, not just sum them.
- **LEACE** (PR, [arXiv 2306.03819](https://arxiv.org/abs/2306.03819)): closed-form least-squares
  concept erasure with minimal collateral; "concept scrubbing" across all layers is practical; needs
  covariance statistics of activations (streaming). The principled erasure alternative to projection.
- Format precedents: [GLP](https://raw.githubusercontent.com/msuiche/weightless/refs/heads/main/spec/GLP.md)
  pins `direction.N` → layer N (no offset, layer 0 inexpressible) — confirms this engine's mapping;
  [llm_steer](https://rdrr.io/cran/relm/man/llm_steer.html) documents "interventions compose".

### A.4 Persona steering
- CAA (PP, [2312.06681](https://arxiv.org/abs/2312.06681)): contrastive mean difference over ~200
  paired examples, added at the best layer (~40 % depth), coefficient ±1 default; large coefficients
  degrade open-ended quality.
- Persona vectors (PP, [2507.21509](https://arxiv.org/abs/2507.21509)): 5 contrastive system prompts ×
  40 questions, activations averaged over **response tokens**, one best layer, coefficient sweep;
  trait expression is monotone in coefficient and MMLU degrades at large coefficients.
- Assistant-axis / persona-drift (PP, [2601.10387](https://arxiv.org/abs/2601.10387)): moving off the
  Assistant persona produces theatrical register; supports "register/identity" steering but warns about
  capability cost.
- Practical consequence for our persona file: the current vector is a **logit-lens token contrast** on
  the output head (register, not behaviour). A measured activation-difference vector at 4–6 layers is
  the v2, and for three traits the composition should be a **joint-subspace rotation**, not a weighted
  sum of three contrasts.

---

## B. Refusal vector v2 — the measured pipeline

### B.1 The instrument already exists
`--dump-layers PATH` writes **`n_layers * hc * n_embd` floats per position, appended in position order**
(`generate.cpp:570-574`, `session.cpp:671,766`): the residual after every layer, every prompt position.
It needs the captured path (refused with `--no-capture`), and it disables the fused gr path for
correctness (`generate.cpp:4073`). `--dump-halves` adds both halves' block outputs + injections per
layer; `--dump-routing` adds expert ids/weights. **No llama.cpp build and no new engine code are needed
to measure per-layer activations** — one quiet-window `strata generate` run per prompt set.

### B.2 Pipeline
1. **Corpus**: a public refusal benchmark (JailbreakBench / HarmBench prompts) and an equal-size harmless
   control set, rendered through the same chat template, thinking off and on. No custom harmful prompts
   are authored for this; use the published sets. Held-out split for scoring.
2. **Capture**: one engine run per prompt with `--dump-layers`; keep the post-instruction positions (all
   positions: the Cudecnik author measured that a last-token-only direction does **not** abliterate under
   projection even though it is nearly orthogonal to the all-position one).
3. **Stream reduction**: the dump is `hc × n_embd` (4 hyper-connection streams). The engine applies the
   **same 2560-d v to every stream** (`cvec.cu` grid `(hc, T)`, per-stream dot), so the extractable
   direction is the shared one: reduce the 4×2560 difference to one 2560 vector (mean over streams, and
   validate the alternative first-PC reduction).
4. **Magnitude sparsification**: clip the top 0.5 % of activations (grimjim's 0.995) before means — this
   model's activations have strong outliers (we saw this in the output head: dot std 0.012 with isolated
   extremes).
5. **Directions**: per layer, `u_l = mean_h − mean_n` (Winsorized); build the paired-difference matrix `D`
   and take its top-k right singular vectors (k = 1..3, keep ≥90 % variance); optionally average over 3–5
   prompt shuffles for stability.
6. **Projected abliteration**: remove the harmless-mean component from each direction,
   `u_l ← u_l − (u_l·ĥ_l)ĥ_l`, where `ĥ_l` is the harmless mean direction.
7. **Layer selection**: `quality_l = SNR_l × (1 − cos(harmful_mean_l, harmless_mean_l))`; take the strong
   source layers and apply each to a chunk of nearby target layers (N-to-M), per grimjim.
8. **Scoring**: held-out 50 harmful / 50 harmless, thinking off and on (a truncated thinking trace reads
   as compliance — count it separately); KL to stock on neutral text; MMLU and an agentic task set as
   capability gates. Adopt on means, interleaved boots.

### B.3 Engine modes needed for v2
| Mode | Math | Why |
|---|---|---|
| `project` (exists) | `h ← h − s(h·v)v` | rank-1 axis removal |
| `project-norm` (new, ~20 lines) | `h ← (h − s(h·v)v) · |h|/|h'|` | grimjim MPOA: same removal, no magnitude damage — the highest-value single change that needs **no stacking** |
| `project-subspace` / k project slots | `h ← h − Σ_i (h·u_i)u_i` on an orthonormalised basis | Gabliteration multi-direction removal (needs stacking) |
| Biprojection | use a source layer's direction on a target chunk, with the target's harmless direction removed | grimjim; a file-construction choice, no kernel change |

### B.4 What can be built before the measurement job
A rank-2 **interim** candidate is buildable offline from the two existing vectors: the per-layer
activation-difference vector `u_l` and the recovered Huihui direction `r`. They are −0.42…−0.77 aligned;
`u_l` alone removes only `(u_l·r)²` of `r` (≈0.58 at the peak, 0.18 at layer 47). The orthogonal
complement `w_l = normalize(r − (r·u_l)u_l)` gives a second, orthogonal axis, and the pair
`{u_l, w_l}` spans `{u_l, r}`. Built:
- `qwen3.8-flash-next-refusal-v2-primary.gguf` — sha256 `ef0724c5…` (identical to the bundled per-layer vector)
- `qwen3.8-flash-next-refusal-v2-orth.gguf` — sha256 `04fb3d4e…`, `w_l` per layer, unit, exactly ⟂ `u_l`
- builder: `tools/refusal_subspace.py`
The pair is **inert until stacking exists** (one projection slot today removes one axis, not two).

---

## C. Persona vector v2

1. **Measured extraction** (needs only `--dump-layers`): contrastive pairs that differ only in the trait
   — direct/committed/imperative vs hedged/playful/deferential — answers averaged over **response
   tokens** (persona-vectors method), per-layer, 4–6 candidate layers, best layer by effect size.
2. **Three traits, one rotation**: assertive, war-identity, resourceful should compose as an
   **ORBIT-style joint subspace + single norm-preserving rotation**, not three summed contrasts. Naive
   weighting is exactly the norm-imbalance/cancellation failure ORBIT documents; our current file's
   cluster weights are a deliberate compromise, not the optimum.
3. **Orthogonalise against the refusal axes at load** (already done for the current file: cos 0.01).
4. **Norm-preserving angular steering**: rotate `h` toward the target within `span{h, t}` by angle θ;
   θ=π is directional ablation, small θ is CAA. This is a third kernel mode and composes with ORBIT.
5. **Calibrate** with a held-out register battery (hedge rate, joke markers, commitment, imperative
   voice, CTF reframing, permission-seeking) + KL + MMLU; the published safe range is small.
6. **Behaviour is not a vector**: "deep web research" needs the client tool surface and a template
   directive to use it. The vector only makes the register consistent with that directive.

---

## D. Perfect stacking — what it takes

### D.1 Target semantics (the contract)
Per layer, in this exact order:
1. **All `project` slots** define one subspace; Gram-Schmidt their directions into an orthonormal basis
   `U = [u₁…u_k]` and apply the **true subspace projector**: `h ← h − U Uᵀ h` (or `project-norm` with
   the rescale). Sequential rank-1 projections of *non-orthogonal* axes are **not** the subspace
   projector; orthogonalising at load is what makes k-direction removal equal to Gabliteration's
   `P = R(RᵀR+λI)⁻¹Rᵀ`.
2. **All `add` slots** are orthogonalised against `span(U)` and against each other, then applied
   additively after the projections (an addition before the projection would be partly erased).
   Optional `rotate` slot: one joint-subspace rotation (ORBIT) instead of additive.
3. Slot order is canonical — projections, then additions/rotations — regardless of config order.
   (Projections onto orthogonal axes commute; additions commute; the cross-mode order matters.)

### D.2 Data model and code changes
| Layer | Change |
|---|---|
| `include/strata/kernels/cvec.hpp` | `struct Cvec` → `struct CvecSlot {dir, s, mode, first, last, steered, name, digest}`; `cvec()` returns a `std::vector<CvecSlot>`; `covers(l)` = any slot covers; `cvec_tables()` → a span of per-slot tables |
| `src/kernels/cuda/cvec.cu` | `g_cvec` → `std::vector<CvecSlot>`; `cvec_upload` takes a vector of slots; `cvec_apply` loops the ordered slots; keep the single-slot path bitwise identical when `slots.size()==1` |
| `src/prefill/kernels.cu` (`gr_write_cvec_norm_rs_kernel`) | extend to `nslots` (two block reductions per project slot) **or** fall back to the unfused path when `nslots>1`; the fused path is an opt-in perf arm, so the fallback is acceptable at first and both must be priced |
| `src/program/generate.cpp::load_control_vectors` | parse per-slot specs; validate each (F32/width/layer range/steered>0/angle in [0,π]); build and orthogonalise; compute per-slot and overall digests; summary gains a slot list |
| config | new repeatable `--cvec-slot name=…,file=FILE:SCALE,mode=project\|add\|rotate\|project-norm,layers=A-B[,angle=θ]`; the legacy `--control-vector-scaled` + `--cvec-mode` path stays as the single-slot default (byte-identical) |
| `serve/handshake.py` | IDENT v2: `"cvecs":[{name,mode,first,last,steered,scale,…},…]`; gate checks each slot against the asked spec; keep the v1 parser and the legaacy INFO string (`project:4-44+rotate:16-32`) |
| `serve/server.py` | `projection_key` → per-slot mask; `experimental_speed_projection` (bool) = all slots; add `experimental_speed_projection_mask` (int); include the full slot digest in the conversation/prompt-cache key (`generate.cpp:14166,14586` already key on the single cvec state) |
| telemetry | `/metrics` `cvec` becomes the slot list + per-slot steered counts; `/props` unchanged |

### D.3 The batch-isolation hazard (fix it in the same change)
The armed flag is **one per-device int** (`cvec.cu:14-19,137-150`); a new admission flips it while other
batch slots decode. Mixed `experimental_speed_projection` across concurrent requests is therefore not
isolated today (the live probe found exactly this; only ON-only traffic was safe). With slots it gets
worse. Perfect fix: replace the global `on` int with a **per-slot, per-batch-row mask** (a small int
array indexed by row) or **admission-group requests by profile** so a window never mixes profiles. The
first is the correct one; the second is the bounded fallback.

### D.4 Tests (must be executable, not conventions)
1. **Parity**: for one slot, stacked == the current single-vector path bitwise (digest identical).
2. **Subspace**: `h − UUᵀh` == sequential projections onto the Gram-Schmidt basis (bitwise/1e-6).
3. **Order**: an `add` slot after a `project` survives; the reverse order is refused/documented.
4. **Orthogonalisation**: an add axis with a known component along the project axis loses exactly that
   component at load (assert the cosine is ~0 in the uploaded tables).
5. **Mixed-profile batch**: two concurrent requests with different masks each match their solo
   reference token-for-token.
6. **Handshake v2**: parse, gate, refuse malformed; legacy v1 still serves.
7. **Digest/cache**: changing any slot invalidates the conversation-cache key; no stale steering.
8. **Config validation**: per-slot range past the model, empty slot, duplicate file, angle out of range,
   negative scale.
9. **Perf**: probe-lane M1/conc for 1/2/3 slots, fused vs unfused; the cvec cost is ≤3 % today — expect
   +0.2–0.5 % per extra slot; the priority-lane rule applies (no regression beyond noise).

### D.5 Phased plan (each phase lands and can be gated)
- **P0 — `project-norm` mode** (no stacking): new cvec mode + kernel branch; expect the grimjim
  capability gain; measure refusal/KL/MMLU against the current projection on the same prompts.
- **P1 — slot refactor, one slot**: C++ data model + config + handshake v2 + digest + telemetry; the
  single-slot path must be bit-identical to today (parity test), default config unchanged.
- **P2 — multi-slot math and masks**: ordered application, Gram-Schmidt, subspace projection, add/rotate
  slots, per-row masks, fused-path extension or fallback; tests 1–9.
- **P3 — measured v2**: the `--dump-layers` extraction job (B.2), build the k-direction refusal vector
  and the ORBIT-composed persona vector, calibrate, then interleaved A/B under the campaign law.

### D.6 Risks to record now
- **Stream reduction**: the dump is 4 hyper-connection streams; the shared-axis reduction choice must be
  validated (mean vs first-PC) or the extracted direction is a mixture.
- **PLE regeneration**: the per-layer embedding path recreates residual content nonlinearly after the
  weight writers (the Cudecnik finding: weight-space abliteration plateaus at ~60 % refusal on this
  architecture). Runtime projection after the write is mandatory; this is why we project at runtime.
- **Hydra effect**: fewer intervention layers self-repair; more layers increase compliance but also
  collateral — the layer count is a dial to measure, not a constant.
- **Quantization**: extraction on the served IQ3_S vs BF16 changes the geometry; measure on what is
  served, then confirm the direction transfers.
- **Safety**: every change here removes safety behaviour; the refusal and persona vectors are both
  uncensoring tools. Gate them, label them, and keep the stock recipe one config line away.
