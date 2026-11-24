# Fresh-eyes review: control-vector index mapping, dump timing, capture harness, method — 2026-11-23

Scope: read-only review of the 2026-11-20/26 vector round (persona r01/r02, the r03 capture
harness) against the engine source and the primary literature. No engine run, no GPU, no git.
Static analysis is marked as such; everything quoted from source carries file:line at the
reviewed tree. Literature was pulled from arxiv.org HTML (`curl`) and reduced locally; quotes are
verbatim from the paper body.

Summary verdict: the measured persona-vector path (r03) is sound in principle and **not runnable
as written**. The capture harness cannot boot on the standing pack (native IQ pack gate), its
index mapping is one block late, and the dossier's composition plan mixes two steering modes in
one boot, which the loader cannot express. All three are fixable before any compute is spent.

---

## 1. Verdicts on questions 1–3

### Q1. Index mapping — `direction.<l>` acts on dump slot l+1

**Rule: dump slot k → `direction.<k−1>`** for k in 2..48. Slot 1 (after block 0) is not
steerable; slot 0 is the input embedding.

Evidence chain, all read in this review:

| fact | source |
|---|---|
| loader: `data[l] += scale·src` for `1 ≤ l < L`; `direction.0` and `≥ L` dropped | `src/program/generate.cpp:1971-1972, 1978` |
| loader: range defaults `first = 1`, `last = L−1`; a 0 first is clamped to 1 | `generate.cpp:1988-1989` |
| loader: project → `s[l]=‖d‖`, `dir[l]=d/‖d‖`; add → `s[l]=1`, `dir[l]=d` | `generate.cpp:2001-2011` |
| kernel reads `s_l[layer]` and `dir + layer·n` by the RAW `layer` argument | `src/kernels/cuda/cvec.cu:53, 56` |
| `cvec_apply` passes `layer` through to the kernel unchanged | `cvec.cu:162-170` |
| `block_layer_post(…, layer, …)` applies `cvec_apply(bb.R, layer, …)` after block `layer`'s FFN write | `src/core/layer.cpp:1639-1650` |
| captured graphs call `block_layer_post(tables, g, l, …)` with `l` the 0-based block loop variable | `src/core/session.cpp:231, 255` |
| `covers(l)` = `l ≥ first && l ≤ last && steered[l]` on the same index | `include/strata/kernels/cvec.hpp:35` |
| prefill (prompt) path uses the same `l` | `src/prefill/prefill.cpp:4770, 4795-4797, 4816-4818, 4828` |
| serve decode window uses the same `l` | `src/core/verify.cpp:2371-2378` |
| dump: slot 0 = `s.R` before `pre[0]`; slot `l+1` = `s.R` enqueued after `post[l]` | `session.cpp:666-678, 762-774` |
| dump file layout: `(n_layers+1)·hc·n_embd` f32 per position, written per loop iteration | `generate.cpp:7384, 17867` |

(a) Block b's residual write (0-based) is followed by `direction.<b>`, b in 1..47. Block 0 is
never steered (`direction.0` dropped; range clamp). `direction.47` acts on the final pre-head
residual. This is what `tools/cvec_table.py:12-13`, the stacking dossier line 80 and the softness
dossier line 24 already state; the two capture tools contradict it.

(b) The state acted on by `direction.<l>` is the post-write residual of block l = dump slot l+1.
Hence slot k ↔ `direction.<k−1>`. The harness/builder assume "slot k → `direction.<k>`":
`tools/persona_vector.py:277-281` builds `v_layers[l]` at slot index l, `:322-327` selects
`band` over the same indices, `:338` stores `rows_out[l]`, `:371` writes `direction.{l}`. Every
measured direction is therefore applied **one block later than measured**, and the Gram–Schmidt
at `:329` orthogonalises a slot-k measurement against the refusal axis `u_per[k]`, which the
engine applies at slot k+1 (`u_per` is read as `direction.N`).

The report's band logic (`tools/persona_capture.py:312-319`) — "layer 0 excluded (block 0 is
not steerable)" — is wrong on two counts: the non-steerable slot is **1**, and slot indices
must be shifted by −1 before they become `direction` indices. The correct exclusion is
`1 < slot ≤ N_LAYERS` (slot 48 → `direction.47` is legal; slot 1 has no direction).

Literature cross-check on indexing: Persona Vectors indexes layers from 1 and defines "layer 20
activation" as the **output** of the 20th block (`2507.21509v1`, App. B.4: "we refer to layers
starting at index 1 … 'layer 20 activation' refers to the output of the 20th layer"). Under this
engine that state is dump slot 20 → `direction.19`.

### Q2. Dump timing — after the write, after the steer (when one is armed)

`--dump-layers` forces the unfused GR path: `layer_set_fused_gr(o.gr_native_mmvf && … &&
o.dump_layers.empty() && …)` (`generate.cpp:4073-4074`). `block_layer_post` then takes the
`gr_write` + optional `cvec_apply(…, write=false)` branch (`layer.cpp:1641-1643`), so the copy
enqueued at `session.cpp:766-769` sees the residual **after** the block's residual write and
**after** the steering of `direction.<l>` when `covers(l)`.

With no vector loaded (`cvec().loaded()` false → `covers` false everywhere) the slot is the
clean, unsteered post-write state — exactly the kernel's steering input `x = r + w·bo`
(`cvec.cu:57-66`). It does not matter for extraction **provided the capture boot loads no
vector**; the harness passes no `--control-vector*` flag and the loader has no other source
(`generate.cpp:2833, 2846` are the only writers of `cvec_files`). A vector armed on the capture
boot would silently yield post-steer slots for every covered layer.

Numerics note: the standing config does not pass `--gr-native-mmvf`, so the standing recipe
already runs the unfused GR kernels; the dump path changes nothing about which kernels produce
the states.

### Q3. Harness validity (`tools/persona_capture.py`, 348 lines) — NO-GO, see §4

Checked against the parser in `generate.cpp`:

| harness flag | parser | verdict |
|---|---|---|
| `--pack PATH` | `:2565` | exists |
| `--native SHARD1` | `:2640` → `native_preset`; `:3363-3367` resolves all shards via `gguf_split_paths` | exists; the standing config passes the same shard path |
| `--ple-gguf SHARD2` | `:2613` | exists |
| `--tokens-file` | `:2572-2583`; list grammar `:1183-1201` (whitespace/comma, non-negative int32) | exists; the harness writes space-separated ids |
| `--max-new` | `:2584` (default 16, `:498`) | exists |
| `--seed` | `:2596` (sets `greedy=false`) | exists |
| `--temperature` | `:2599` (default 1.0; `top_k=20`, `top_p=0.95` defaults `:502-503`) | exists |
| `--max-context` | `:2585` (default 4096) | exists |
| `--dump-layers` | `:2610`; refused with `--no-capture` `:7275-7280` | exists |

GPU residency of a bare boot: `expert_cache` defaults to 0 → auto (`:638, 4569`); a bare boot is
GPU-resident for experts, not a CPU crawl. Not a concern.

`parse_produced()` assumption: `output  :` is printed once to stdout at the end
(`generate.cpp:18586-18588`) with every produced id. Correct for a successful run.

Row count `rows = P + M − 1`: correct for the pure one-token loop (`generate.cpp:7356-7369`,
`:17976-17977`) and holds with an EOS stop too (the EOS id is pushed before the break,
`:17976-17979`). It requires `spec = 0` (default `:736`) and `prefill_chunk = 0` (default
`:701`) — exactly what a **native pack forbids** (D1).

`extract_means` windows (`:189-190`): rows `P..P+M−2` for `response`, `P−1..P+M−2` for
`pre-emission` — consistent with the loop (row r = position r; the M-th response token is never
fed). Correct once D1 and D4 are fixed.

Welford (`:252-256`): vectorised single-sample Welford, correct.

`report()` d′ (`:302-304`): pooled per-dimension variance over both poles, then
`‖diff‖ / mean(σ_dim)`. It scales as √n_embd (a per-dimension effect of 0.1 reads ≈ 5); it is a
ranking statistic only. The function executes (synthetic run on numpy 2.5.3, CPU) and in the
flat-profile case returns the whole stack as the band (D5).

Position windows after an EOS: without `--stop-eos` the run continues past `<|im_end|>` (D4).

---

## 2. Ranked defect list (with fixes)

**D1 — BLOCKER. The harness cannot produce a single dump row on the standing pack.**
The pack `packs/iq3_s` is native (`native_experts.txt` present → `expert_layout().native`,
`src/kernels/cpu/expert_layout.cpp:392-403`). A native pack is refused without `--spec ≥ 2` and
`--prefill CHUNK` (`generate.cpp:3890-3896`); the harness passes neither
(`persona_capture.py:156-163`) → every boot exits 2. With those flags it is still dead: the
batched prefill consumes positions `0..n_batched−1` without dumping (`:17599-17746`), and on a
native pack the one-token loop exits before its first iteration (`if (native_pack) { spec_pos =
pos; break; }`, `:17769-17770`); the speculative loop (`:17998+`) never writes a row. The only
`fwrite` of the staging buffer is `:17867`. The harness treats a non-zero return as "BOOT
FAILED … continue" (`:243-249`), then writes an **empty** `means.npz` and `report()` returns 0
(`:268-276, 279-328`): 288 failed boots, exit status 0, nothing measured.
Fix (engine, correct): refuse `--dump-layers` on a native pack at option validation with a
message naming the verify-path limitation (law 14: fail loudly, not with an empty file and a
"layers dumped" line at `:18563-18567`), and add the per-layer row write to the verify window
(`verify.cpp`, after the layer's `cvec_apply`/`gr_write` at `:2371-2379`: copy `Rt(t)` per layer
into `layer_stage`). Fix (harness, zero engine change): retarget to the non-native pack
`packs/q2_0` (manifest: Qwen3.8-Flash-Next GSQ-RCO Q2_0, 48 layers; `tools/pack_index.py`),
where `spec=0`/`prefill_chunk=0` defaults hold and the row math is exact — and name the
confound (Q2_0 states, IQ3_S deployment). Fix (harness, mandatory either way): abort on the
first failed boot; assert `len(stats) == 2·len(CONTRASTS)` and every count > 0 before `savez`;
non-zero exit on any failure.

**D2 — CRITICAL. Index off by one in `--method dump`.** `persona_vector.py:277-281, 322-327,
338, 371` write slot k's row to `direction.<k>`; the engine applies `direction.<k>` to slot k+1
(§Q1). Fix: `rows_out[slot − 1] = v_slot` for slot ≥ 2; orthogonalisation axes at slot k are
`u_per[k−1]` (`r` is layer-independent); skip slot 1; `report()` band filter
`1 < r["layer"] <= N_LAYERS` and report in slot units with the shift stated.

**D3 — CRITICAL (plan). The composition A/B in the dossier is not expressible.**
`2026-11-26-vector-v2-research-and-fix.md` §D.1 step 2 plans one boot with refusal-r + r02 under
`--cvec-mode add`. `load_control_vectors` sums all files into `data` (`:1978`) and then applies
the **single** `o.cvec_mode` to the sum (`:1997-2011`). Measured on the files: every row in all
six vectors is unit-norm; refusal-r is one direction replicated at 47 layers. In add mode the
refusal rows would be **added** at scale 1.0 at 47 boundaries — Arditi: "adding [the direction]
to the model's activations induces refusal" (`2406.11717`, §3.2) — with the file's sign unrecorded
(project mode is sign-blind). In project mode the summed bisector is projected, which the
dossier's own §A.2 rules out. Two modes cannot share a boot before the slot refactor (potency
dossier §D). Fix: strike step 2; test orthogonality by the refusal battery on the persona-only
boot vs the refusal-only boot (same hour, interleaved) and defer true composition to P1/P2.

**D4 — HIGH. No EOS stop.** `stop_eos` defaults to false (`generate.cpp:783`); `--stop-eos`
(`:2771`) arms the default ids `{248044, 248046}` (`:784`; `<|im_end|>` = 248046). The harness
passes neither, so a thinking-off answer is followed by ~90 post-`<|im_end|>` positions that are
averaged into the "response". Fix: add `--stop-eos` to `engine_argv`; truncate `produced` at the
first EOS id in `parse_produced` as a belt; exclude the EOS row from the mean (optional).

**D5 — HIGH (method). Band selection is unsupervised and degenerate.** `report()` picks
`d′ ≥ 0.8·max` (`persona_capture.py:311-313`). On a flat profile — the common case for residual
streams — this returns the entire stack: a synthetic run (no planted structure) returned
`[1..48]` for all three traits and "best layers" 31/22/8 (noise). The literature selects the
layer by **causal steering effect at a single layer with a fixed coefficient**: CAA "sweep over
all layers … assessing the effect size on the held-out test questions" (`2312.06681v3` §4.1;
optimum L13/32 for 7B, 14–15/40 for 13B); Persona Vectors "apply steering individually at each
layer using the same coefficient, and select the layer that elicits the highest trait expression
score" (App. B.4; Qwen2.5-7B layer 20/28, Llama-3.1-8B 16/32); Arditi selects "the single most
effective vector … by evaluating each candidate vector over validation sets" (§2.3). Fix: the
report ranks ≤ 3 candidate slots (d′ peak prominence > 1.25× the stack median, head-bias
concentration below threshold); the choice is made by the validation ladder (§5); band width
≤ 5 layers; drop the 0.8 rule.

**D6 — MEDIUM. No outlier clipping before means.** Raw states are averaged
(`persona_capture.py:190`). The potency dossier §B.2 step 4 and grimjim's practice call for
Winsorisation at 0.995 before means (outlier dimensions align the two class means). Fix: per run,
per (layer, dim), clip the response rows at that run's 99.5th percentile before the mean (the
dumps are deleted per run, so the pool cannot be global).

**D7 — MEDIUM. Dose is not norm-calibrated.** Rows are unit, "dose via FILE:SCALE", start 0.08
(`persona_vector.py:363`); the dossier's "0.08 × 9 layers ≈ 0.72 total" is a sum of head-bias
units, not a residual-norm fraction. Per literature (§3c) the dose must be a fraction of the
residual norm at the target slot. Fix: scale each written row by `ρ · mean‖h_slot‖` (the
capture yields `‖μ_l‖` per layer for free), sweep `ρ ∈ {0.02, 0.05, 0.1, 0.2}`; keep the
head-bias diagnostic per candidate ρ.

**D8 — MEDIUM (engine UX). `--dump-layers` on a native pack yields an empty file with exit 0.**
Consequence of `:17769-17770` plus the unconditional "layers dumped" line `:18563-18567`. Same
fix as D1's engine half.

**D9 — LOW (phase gating).** `--control-vector-phase` exists only in the `--serve` request path
(`generate.cpp:14591-14597, 16969-16985`); standalone `generate` never calls `cvec_set_enabled`
and never warns — a loaded vector is ON from upload for prompt and generation
(`cvec.hpp:42` "Starts ON"). The on/off flag is one device int (`cvec.cu:18, 137-150`): under
`--parallel 6` every request start flips it for all slots (`:14591, :14597`) and every arm/disarm
flips every in-flight request (`:16974, :16981`). Fix: validate phase mode with single-request
traffic only; warn at boot when a phase-armed vector meets `--parallel > 1`; per-row mask at P2.

**D10 — LOW (workspace law).** Default workdir is inside the repo
(`persona_capture.py:195`); `layers.bin` is ~0.7 GB per boot in `campaign/results/` and
`--keep-dumps` would park many GB there. Fix: raw dumps under
`/home/svg/LLM/Custom-data/mtp/fixtures/persona-capture/`; only `runs.jsonl`, `means.npz`,
`report.json` in `campaign/results/`.

**D11 — LOW (record drift).** Citation IDs in the brief/dossier: CAA is `2312.06681`
(`2403.19434` is a WASP-76b spectroscopy paper); `2505.24158` is an MLLM long-video paper, not an
orthogonality paper — no arXiv title matching "Optimizing Orthogonality" exists (API query,
2026-11-23). The dossier's "persona-vectors' mid-stack single layers" is wrong for its own
example: Qwen2.5-7B layer 20/28 = 0.71 depth. Potency dossier line 103 says the dump is
`n_layers·hc·n_embd` per position; it is `(n_layers+1)·hc·n_embd` (`generate.cpp:7384`).

---

## 3. Method recommendations (with citations)

**(a) Lexical confound under contrastive system prompts — yes, it is the r01 trap in hidden
form, and the defense is a concentration cap plus a depth cap, not zero head alignment.**
Under two system prompts the poles generate different words; at late layers the response-token
difference converges on the output-lexical direction (the unembedding rows of the pole
vocabularies), which is what r01 was by construction. Billa (`2604.15557v1`) shows the steering
effect is governed by "the steering direction projected through the unembedding" and that this
alignment *predicts* success — so head alignment cannot be zeroed; the r01 failure was
*concentration* (construction vocabulary at the 99.2nd promotion percentile, dossier §A.1).
Defense, in order: (1) restrict candidates to ≤ 0.6 depth (CAA's optima 0.36–0.41; Arditi and
Persona validate per layer, Persona's Qwen pick is the exception at 0.71); (2) run the existing
head-bias diagnostic (`persona_vector.py:373-395`) **per candidate layer** and require
construction-vocab promotion percentile ≤ 75 and energy fraction ≤ 0.1 %; (3) Winsorise before
means (D6); (4) optional ablation arm: project out the prompt-last contrast direction (the
system-prompt encoding, Persona App. A.3's "prompt last") from the response-token direction —
expected to weaken the vector (the prompt state is the trait's carrier), so measure, do not
assume.

**(b) Embedding-contrast (r02) at layers 18–26 — no literature support for a layer-0 → mid-stack
transfer.** The direction is a static lexical axis (unit embedding-row contrast, one vector
replicated at 9 layers, measured adjacent-cosine 1.000). CAA §8.2 finds steering vectors from
different layers dissimilar; ActAdd notes only that "transplanting activations from layer 2 to
layer 20 sometimes increases anger" (`2308.10248v4`, App. G) — "sometimes", with coefficients
whose norm was "nearly ten times the norm of the underlying forward pass". grimjim's
biprojection moves a *strong source layer's* direction to a *chunk of neighbours*, never from
layer 0. Quantitative prediction for r02 at :0.08: per-layer relative perturbation ≈ 0.08/‖h‖
(‖h‖ at mid-stack is tens to hundreds → ≤ 0.5 %/layer), total head-bias ≈ 0.073 × 0.72 ≈ 0.05
logits. r02 at :0.08 is predicted to be a no-op; one boot settles it (§5, B-r02).

**(c) Magnitude.** What the papers do: CAA normalises vectors **across behaviours** but "not
over the layers to preserve a 'natural norm' given the sampled activations", and sweeps
multipliers on held-out data (`2312.06681v3`, §3 and "Vector normalization choices"). ActAdd
multiplies the raw difference by an injection coefficient and reports the relative norm
decreasing with depth — i.e. no per-layer norm matching (`2308.10248v4`, App. G). Persona
Vectors "normalize each direction and apply steering with a fixed norm magnitude" (App. A.3),
steering "at each decoding step" with a scalar coefficient (§3.2), taking the "largest steering
coefficient for which the model's coherence remains above 80" (App. J.3), with MMLU degrading at
large coefficients (§5.1). None angularly calibrates. Recommendation: additive dose as a
fraction ρ of the measured mean residual norm at the target slot (D7); angular calibration
(`s_l = tan θ · mean‖h_l‖`) is the same quantity reparametrised for small θ and is not what the
additive literature validates — use ρ, report θ = atan ρ for readability. Norm-preserving
rotation (`2510.26243`) is a kernel mode, out of scope here.

**(d) Orthogonalisation.** To the refusal axes: correct and already built (two Gram–Schmidt
passes, cos < 5e-3). Add: orthogonalise the trait vectors against **each other** when composing
(`2610.37517v2`: "naively combining vectors causes them to interfere … we orthogonalize the
vectors"); remember "orthogonality alone does not imply intervention-independence"
(`2502.17420`). To the unembedding rows of the trait vocabulary: **no** as a hard constraint —
it removes part of the mechanism (Billa's output-alignment principle); use it as the
concentration cap in (a). The paper the brief names for this (Lyu et al., 2505.24158) could not
be located (D11).

**(e) Phase gating (off during reasoning, on for the answer).** No paper found that steers a
reasoning model only post-`</think>`; the nearest is Hua et al. `2510.20487` (steering
evaluation-awareness, whole request). Engine-wise it is the only mode that matches CAA's
application semantics — "added at all token positions after the user's prompt"
(`2312.06681`) — because the default steers the prompt read as well (`prefill.cpp:4827`). Risks:
D9 (serve-only; global flag under `--parallel 6`); the reasoning trace stays in the un-steered
register, so the answer inherits hedged conclusions — a register lever, not a reasoning lever;
and the trait must be scored on the answer only. Validate with single-request traffic.

**(f) Hyper-connections.** No steering literature on multi-stream residual architectures was
found. A single shared direction across the 4 streams is the uniform direction of stream space;
the engine's `gr_read` mixes streams by learned `hc_*` weights, so a stream-mixed difference
(e.g. `(+v, −v, 0, 0)`) is invisible to the shared-axis reduction. The harness's
`stream_pair_cos_min` (`persona_capture.py:295-298`) is the right gate; require ≥ 0.8 at the
candidate slots. MoE: `2509.09660v2` steers MoE models through router logits (expert
(de)activation), not the residual stream; no paper measures the router-mediated effect of
residual steering. Here the router reads the normed mix of R (`layer.cpp:1583-1595`), so a
steer at block l changes block l+1's routing. Cheap instrument: `--dump-routing` (`:579-583`,
one record of layer/k/ids/weights per position) on a steered vs un-steered boot — no new code.

Robustness caveats for the validation design: steerability is "highly variable across different
inputs" and "brittle to reasonable changes in the prompt" for several concepts
(`2407.12404v8`) — score the trait on held-out prompt shapes, not the capture questions.

---

## 4. GO / NO-GO on `tools/persona_capture.py` as written

**NO-GO.** Do not spend the quiet window on it. Before any boot:

1. D1: pick the route — (i) engine: loud refusal + verify-path row write (correct, ~1 day), or
   (ii) retarget to `packs/q2_0` with the quantisation confound on the record. Either way:
   abort on first failed boot, assert coverage, non-zero exit.
2. D2: `direction.<slot−1>` in `persona_vector.py --method dump`; axes `u_per[slot−1]`.
3. D4: `--stop-eos` in `engine_argv`; EOS truncation in `parse_produced`.
4. D5: replace the 0.8·max band with a ranked candidate list; the ladder chooses.
5. D6/D7: Winsorise per run; write `ρ·‖μ‖`-scaled rows.
6. D10: raw dumps out of the repo tree.
7. A geometry probe before the battery (§5, P0).

Everything else in the harness (flags, list grammar, `output  :` line, row math on a non-native
pack, Welford, windows) is correct as read.

---

## 5. Minimal validation design (fewest server boots)

**P0 — two engine falsifiers, no server (≈ 2 one-shot boots, minutes each).**
P0a: `--dump-layers` on the chosen pack with a 4-token prompt, `--max-new 2`: assert
`rows == 5` and slot 0 equals the `token_embd` row of token 0 (pack `dense.bin`). Kills or
confirms the geometry and the slot-0 claim. On the native pack expect the loud refusal (D1/D8).
P0b: same prompt, refusal-r armed at `--control-vector-layer-range 22 22`, `--max-new 1`,
vs unarmed: assert slot 22 bitwise identical and slot 23 different. Proves
"direction.22 acts on slot 23" empirically — the cheapest possible falsifier of D2.

**P1 — capture, no server (12 boots).** Two traits × two poles × three held-in questions × one
rollout, thinking off, `--stop-eos`, no vector loaded. Report per slot: d′, stream-pair cos,
`‖μ_slot‖`, head-bias concentration of `v̂_slot` (W_out rows from the pack). Output: ≤ 3
candidate slots ranked, all ≤ 0.6 depth, concentration under cap, stream cos ≥ 0.8. This is
the capture the dossier planned at 288 boots; 12 decide whether any slot is worth a server
boot. (CAA extracts from hundreds of pairs but at a single answer position; Persona from 10
rollouts per question — the 12-boot profile is a screen, not the final vector; widen to the
full set only for the slot that survives P2.)

**P2 — server, interleaved, means only (6 boots for 3 candidates + control).**
B0 control (standing recipe, refusal-r only) → B1 slot A at ρ=0.1 → B0 → B2 slot B → B0 →
B3 slot C. Each boot: persona battery (POSTURE, B1/B4 correctness, JOKE reframing, hedge
rate; fixed seeds; thinking off and on; phase mode under **single-request** traffic, then
repeated under `--parallel` to expose D9), refusal battery (6 harm prompts, 1 token: expect
0/12 "I"), MMLU gate ≥ 93.3, KL ≤ 0.2 nats on neutral text. Kill: no trait move beyond run
noise at ρ=0.1 with capability intact → the measured direction is inert at that slot; bank the
negative with the d′/concentration numbers.
**P3 — dose (2 boots).** Winner slot at ρ ∈ {0.05, 0.2}, interleaved with B0. Adopt the largest
ρ with all gates green (the CAA-frontier discipline: capability budget first).

**Optional control (1 boot): r02 at :0.08** on the same battery — predicted no-op (§3b); if it
moves the register, the head-bias model of r01 is incomplete and the capture report must say so.

Total: 2 probe boots + 12 capture boots (no server) + 8 server boots.

---

## Sources

- CAA: Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*, arXiv 2312.06681v3 — https://arxiv.org/abs/2312.06681
- ActAdd: Turner et al., *Activation Addition: Steering Language Models Without Optimization*, arXiv 2308.10248v4 — https://arxiv.org/abs/2308.10248
- Persona Vectors: Chua et al., arXiv 2507.21509v1 — https://arxiv.org/abs/2507.21509
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, arXiv 2406.11717 — https://arxiv.org/abs/2406.11717
- Billa, *Predicting Where Steering Vectors Succeed*, arXiv 2604.15557v1 — https://arxiv.org/abs/2604.15557
- Chanin et al., *Analysing the Generalisation and Reliability of Steering Vectors*, arXiv 2407.12404v8 — https://arxiv.org/abs/2407.12404
- *Orthogonal and Parametrized Multi-Attribute Steering of LLMs*, arXiv 2610.37517v2 — https://arxiv.org/abs/2610.37517
- *Steering MoE LLMs via Expert (De)Activation*, arXiv 2509.09660v2 — https://arxiv.org/abs/2509.09660
- Hua et al., *Steering Evaluation-Aware Language Models to Act Like They Are Deployed*, arXiv 2510.20487 — https://arxiv.org/abs/2510.20487
- Angular steering 2510.26243, concept cones 2502.17420: as cited in `2026-11-26-vector-v2-research-and-fix.md` (not re-read here).
- Not located: "Improving Steering Vectors in LLMs by Optimizing Orthogonality" (the brief's 2505.24158 is *Threading Keyframe with Narratives*, an MLLM paper).
