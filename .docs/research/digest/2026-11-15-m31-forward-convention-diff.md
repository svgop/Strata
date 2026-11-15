---
type: research
date: 2026-11-15
title: MiniMax-M3.1-Flash forward convention diff — DwarfStar (IMPL-A) vs strata mm3 (IMPL-B)
impl_a: /home/svg/LLM/data/references/ds4-dfm-rs
impl_b: /home/svg/LLM/strata-custom-2/src/mm3
maya_ref: /home/svg/LLM/data/references/project-maya/src/mm3
verdict: all eleven forward stages match in semantic convention; divergences are precision-class plus four real convention gaps, none of which alone explains word-salad loops
---

# M3.1 forward convention diff: IMPL-A (ds4-dfm-rs) vs IMPL-B (strata mm3)

Scope: the M3.1-Flash forward walked in order through both trees, one convention per
stage. IMPL-A = DwarfStar's proven path (`minimax_vq` Rust validation + C/CUDA
runtime): `ds4_minimax_graph.inc` (forward), `ds4_minimax_gpu.cuh` +
`cuda/minimax_{decode,q8,rows,mma8w}.cuh` (kernels), `ds4_minimax_bind.inc`
(geometry), `ds4_minimax_kv.inc` (KV tiers), `ds4_minimax_draft.inc` (DSpark).
IMPL-B = strata's port (`mm3_model.cu` forward, `mm3_kernels.cu` own kernels,
`mm3_gguf.cu`/`vendor/ds4_iq` trunk GEMVs, `mm3_vq.cu` GVQ experts,
`mm3_dspark.cu`, `mm3_main.cpp` serve loop, `mm3_sample.hpp`).

Reference arithmetic contract both cite: transformers `MiniMaxM3VLDecoderLayer`
(A: `ds4_minimax.h:22-23`; B: `mm3_model.cu:5-12`).

## Stage-by-stage diff

### 1. Embedding lookup — MATCH

- A: host gathers BF16 rows from the mapped embedding part, device widens,
  no scale, no floor: `ds4_minimax_graph.inc:521-531` (host memcpy), kernel
  `mm_embed_kernel` `ds4_minimax_gpu.cuh:3096-3111` (`__bfloat162float`, cast to
  row format; prefill keeps BF16 rows via `DS4_GVQ_ROWS_BF16`).
- B: host widens BF16 → F32 by `u = bits << 16`, no scale:
  `mm3_model.cu:3047-3069` (`embed`). GGUF/Q8 types via `gg::host_row`.
- Divergence: none. (B keeps the residual F32 for prefill; A prefill keeps BF16
  rows — precision only.)

### 2. RMSNorm convention — MATCH (bake law now aligned)

- A: Gemma runtime form, never baked: `out = round((1 + w) * (x * inv))`,
  `inv = rsqrtf(mean(x^2) + eps)`, eps 1e-6, weight BF16 read to F32:
  `mm_row_rms` `cuda/minimax_rows.cuh:30-37` (eps `ds4_minimax_gpu.cuh:38`).
  Final norm identical via `ds4_gpu_minimax_rms` `ds4_minimax_gpu.cuh:1096-1111`.
- B: `w1 = 1 + w` baked at upload — `norm_add()` returns 1.0 for vq packs:
  `mm3_model.cu:368-377, 1330-1335, 1398`; kernel `rms_kernel`
  `mm3_kernels.cu:138-175` (`v = x * r * w1[i]`).
  The raw-vs-baked norm law (the previous word-suffix-loop bug class) is applied.
- Per-head q/k norms: A `mm_qk_head` `ds4_minimax_gpu.cuh:1230-1250` (mean over
  the 128 head dims, then (1+w), one BF16 round); B `norm_rope`
  `mm3_kernels.cu:180-197` (same shape; w1 pre-baked). MATCH.
- Divergence (precision): A rounds x to the BF16 grid before the sum-of-squares
  (`glmvq_round(glmvq_value(...))`) and rounds each output once
  (`glmvq_round` = `__float2bfloat16_rn`, `cuda/glm_vq_ops.cuh:18-20`); B norms
  raw F32 and never rounds activations (B keeps F32 rows; only KV and index
  tensors carry model-dtype rounding). Systematic but sub-BF16-ulp.

### 3. QKV projection + q/k norms + RoPE — MATCH

- Partial rotary 0.5: both rotate ONLY dims [0, 64), as two 32-halves — pairs
  (d, d+32) for d < 32; dims [64,128) pass through. NeoX form
  (x1*c − x2*s, x2*c + x1*s on the split halves), NOT GPT-J interleaved:
  A `mm_qk_head` tail `ds4_minimax_gpu.cuh:1241-1249`; B `norm_rope`
  `mm3_kernels.cu:187-194` (`d < n_rot`, `half = n_rot/2`, `sh[d+half]`).
- theta 1e7: A evaluates `pos * 1e7^(-2j/64)` and cos/sin in double, rounds to
  F32 once, a per-position table (`mm_rope_kernel`
  `ds4_minimax_gpu.cuh:1211-1222`); B precomputes inv_freq in double → F32
  (`rope_inv_freq` `mm3_kernels.cu:2157-2160`) and does `cosf/sinf(pos*inv)` in
  F32 per element (`mm3_kernels.cu:191-193`).
- Divergence (precision, context-length dependent): F32 angle + `cosf/sinf` vs
  double. At pos 32k the F32 angle ulp is ~0.004 rad → rope phase error up to
  ~0.002 rad; A is exact at any position. grows with context.
- QKV projections themselves: A Q4_K decode with the input row rounded to BF16
  (`mm_decode_gu` `cuda/minimax_decode.cuh:81`, graph `ds4_gvq_q4p_prepared`);
  B GGUF-typed trunk GEMV (token path `x3::mv` with Q8-quantized activations,
  `mm3_gguf.cu:364-439`; prompt path FP16 x FP16 GEMMs from `d.xn16`,
  `mm3_model.cu:2836-2839`). Same math, different activation precision.

### 4. The indexer — MATCH (all sub-conventions)

- Projections index_q [512 = 4x128], index_k [128] from Q8_0 raw: A
  `mm_q8_gemv` `ds4_minimax_gpu.cuh:637-677` (value = f16 scale * int8, exact in
  F32, fixed-order xor tree); B `rd_w<Q8>` in `rows_dot_kernel`
  `mm3_kernels.cu:1274-1282` (same exact value), prompt path dequantizes to F32
  then cuBLAS (`mm3_model.cu:2846-2851`, `q8_0_f32`
  `mm3_kernels.cu:1323-1328`). MATCH.
- Per-head norms + rope + BF16 rounding of the queries: A
  `ds4_gpu_minimax_tail` blocks 72-76 `ds4_minimax_gpu.cuh:1313-1362`; B
  `index_kernel` `mm3_kernels.cu:383-394` — iq rounded through BF16
  (`bf2f(f2bf(...))`), ik stored BF16 at row `pos`. MATCH
  (A's index keys are NVFP4 72-byte rows instead — see KV layout below).
- Block score = **max** over the block's visible keys (key pos <= query pos) of
  the dot: A `mm_score_kernel` `ds4_minimax_gpu.cuh:1447-1489` /
  `mm_score_few_kernel:1502-1562`; B `block_scores_kernel`
  `mm3_kernels.cu:403-448` (token kernel `:454-500`, wide kernel `:512-586`).
  MATCH.
- Top-k selection incl. local/init rules: own block always; when the own block
  index < 16 take every visible block 0..own, else own + the 15 best earlier
  blocks; ties to the lower block; -1 padding: A `mm_topk_kernel`
  `ds4_minimax_gpu.cuh:1579-1640` (`mm_topk_before:1568`), "at most 16 visible →
  all" gate `:1691-1693,1590`; B `select_kernel` `mm3_kernels.cu:596-663`
  (`qb + 1 <= topk` early-out `:611`, `better()` `:593`).
  `sparse_init_block` rejected when nonzero (B `mm3_geo.hpp` "init blocks").
  MATCH.
- Ties: torch.topk semantics both (`v > w || (v == w && i < j)`).
- `sparse_attention_freq`: every layer sparse in both (B refuses a full-attention
  layer, `mm3_geo.hpp` `for (int s : sparse) if (!s) return no(...)`).
  `disable_index_value`: neither implements an indexer value path. MATCH.

### 5. Sparse attention — MATCH

- GQA grouping: KV head = head/16; the group's 16 q heads share the KV group's
  kept list: A `mm_attn_kernel` `ds4_minimax_gpu.cuh:1747-1843`
  (`kv = head/16`, qs loaded from `kv * MM_GROUP`); B `sparse_attn_kernel`
  `mm3_kernels.cu:770-902` (`g * kG + h`). Index head h ↔ KV group h
  (kept `[Hkv][T][topk]`). MATCH.
- Softmax scale 128^-0.5 in F32: A `0.08838834764831845f`
  `ds4_minimax_gpu.cuh:1762`; B `1.0f / std::sqrt(g.hd)` folded into q before
  the dot (`mm3_model.cu:2871`, kernel `:788`) — Hadamard-linear, commutes.
  MATCH (fold order differs numerically).
- Online softmax split: A F32 online over tiles + block partials
  (`mm_attn_combine:2573-2592`); B F32 online over 64-key tiles + splits
  (`attn_combine_kernel:1061-1105`, max-of-maxima then rescale — same math).
  MATCH.
- Causality: inside the own block keys <= p; earlier blocks fully visible;
  kept blocks only <= own. Both. MATCH.

### 6. Attention output + residual — MATCH (precision)

- A: heads rounded once (BF16 grid), o_proj Q4_K, residual add
  `cur = round(cur + round(branch))` (`mm_residual_kernel`
  `cuda/minimax_rows.cuh:67-97`).
- B: token path o_proj GEMV from F32 `d.att`; prompt path rounds to FP16
  (`to_f16` `mm3_model.cu:2874`); residual `k::add` F32
  (`mm3_model.cu:2563`, `mm3_kernels.cu:1249-1252`). Divergence: rounding grid
  (BF16 vs FP16/none) — precision only.

### 7. Router — MATCH

- Sigmoid + correction bias, top-4 by `s + bias`, ties to lower expert, weights
  = unbiased sigmoid renormalized over the chosen 4, then routed_scaling 2.0:
  A `mm_route_row` `ds4_minimax_gpu.cuh:512-546` (sigmoid in double, weights
  `__fdiv_rn(s,sum) * 2.0f`); B `route_kernel` `mm3_kernels.cu:1108-1153`
  (sigmoid F32, `w /= sum`), scale applied at the plan/combine
  (`mm3_model.cu:2659` `pw[i] = w[i] * g.scale`, `combine_kernel:1243`).
  Multiplication order differs; value equal.
- Logits never TF32: A doc `ds4_minimax.h:75`; B `CUBLAS_PEDANTIC_MATH`
  `mm3_model.cu:1212`. MATCH.
- Divergence (precision): A's sigmoid in double-then-round; B `expf`.
- Top-k tie/selection order: A selection ranks by bias-adjusted score, then the
  expert sum runs in **ascending expert id**; B's expert sum runs in **selection
  rank order** (`combine_kernel:1242`, `vq combine_kernel` `mm3_vq.cu:444`).
  F32 accumulation order only.

### 8. MoE expert compute — MATCH

- swigluoai: `g = min(gate, 7)` (upper clamp only), `u = clamp(up, ±7)`,
  `s = 1/(1 + exp(-1.702 g))`, `round(g*s*(u+1))` — clamp BEFORE the sigmoid
  multiply, +1 inside: A `glmvq_swiglu_value<GLMVQ_CLAMP_OAI>`
  `cuda/glm_vq_ops.cuh:141-146` (limit 7 `cuda/minimax_decode.cuh:13`); B
  `swoai` `mm3_kernels.cu:1211-1215` and `oai` `mm3_vq.cu:310-314`
  (alpha `x3::kSwigluAlpha = 1.702f`, `mm3_exl3.hpp:43`). MATCH.
- Down projection and gain: both multiply the row gain into the F32 dot, A once
  then BF16-round (`mm_down_dot` `cuda/minimax_decode.cuh:57`), B
  `__fmul_rn(g, acc)` F32 (`mm3_vq.cu:433`). Precision only.
- Activation precision: A decode keeps expert mid rows BF16; B F32/FP16 —
  precision only. Expert decode itself is differential-proven (f16_rows bitwise,
  moe rel 1.6e-6) and GVR1 residual magic is verified on device before every
  served blob (`mm3_vq.cu:453-501`).

### 9. MoE combine — MATCH

- A: `branch = round(sum_experts(weight * row, ascending id) + round(shared))`,
  then `cur = round(cur + round(branch))` (`mm_decode_reduce`
  `cuda/minimax_decode.cuh:138-154`, `mm_mma_reduce:62-73`); shared expert
  UNSCALED, inside the branch, routed weights carry the 2.0.
- B: `out = sum(w_i * y_i) * scale + shared` (routed only scaled), residual add
  separately (`combine_kernel` `mm3_kernels.cu:1237-1247`; token path
  `x3::moe_down` → `combine_kernel` `mm3_vq.cu:437-447` + `k::add`
  `mm3_model.cu:2628`). Same algebra. MATCH (order/rounding precision only).

### 10. DSpark drafter — MATCH

- On only with the drafter part + request (A `minimax_vq.rs:93-100` draft > 1;
  B `Options::dspark`, `mm3_main.cpp:558`). Off by default in both.
- Taps: residual AFTER layers {3,17,31,45,59} (A default tap_mode=0,
  `ds4_minimax_graph.inc:576-605`, `ds4_minimax_draft.inc:43`; B copies `d.h`
  after layer li when `tap_of[li] >= 0`, `mm3_model.cu:3278-3286`).
- fc → hidden_norm → per-layer k_norm+RoPE into a ring (window 4096) → block
  [anchor, mask x 6] → final_norm → target head → row i + `markov_w2 ·
  markov_w1[row i-1]` → argmax; row i proposes position p+i+1: A
  `ds4_minimax_draft.inc:5-37`; B `mm3_dspark.cu:346-467` (`draft_block` pos[t]
  = P+t, `mm3_kernels.cu:2405-2415`), Markov bias `markov_greedy`/
  `markov_pick` `mm3_kernels.cu:1729-1905`. MATCH.
- Verification: the target forwards anchor+proposals once (A MINIMAX_HEAD_ALL
  writes every row's logits, `ds4_minimax_graph.inc:20,612-617`; B the window
  forward `T = k+1` returning `k+1` logit rows, `mm3_main.cpp:685-686`); commit
  keeps the accepted prefix only — greedy keep iff draft == target argmax
  (`mm3_main.cpp:707-709`); stale KV rows above the commit point are rewritten
  before any read in both (A `ds4_minimax_draft.inc:34-37`; B writes at
  `pos = state.size()` each round, `mm3_main.cpp:607`). MATCH.

### 11. Sampling — MATCH

- Greedy argmax, ties to the lower id: B `dist_c` `mm3_sample.hpp:74-79`; A
  device argmax route (greedy = temperature <= 0, `ds4_ds41_sample.inc:23`).
- Temperature: logits / T then softmax (B `mm3_sample.hpp:80-108`); A same
  route. MATCH.

## KV cache layouts, sink handling

- A: per (layer, 128-token block, KV head) NVFP4 pages — K data 64 B + 8 E4M3
  scales, then V, per token row 72 B; encoder `scale = E4M3(max(amax/6, 2^-9))`,
  E2M1 round-to-nearest-even, global scale 1 (`ds4_minimax.h:38-44`,
  `mm_fp4_encode` `ds4_minimax_gpu.cuh:1278-1297`); index keys 72 B/token
  NVFP4; SSD tier + pinned pool with a slot table (`ds4_minimax_kv.inc:1-46`).
  No Hadamard. No attention-sink handling anywhere in A.
- B: BF16 caches by default config, or INT8 (per-row `max|v|/127`), FP8 E4M3
  (power-of-two scale per 32), FP4 E2M1 (E4M3 scale per 16, NVFP4-style) —
  `mm3_kernels.cuh:27-51`, writer `put_row` `mm3_kernels.cu:296-345`; optional
  128-point Hadamard rotation of K and/or V rows before rounding, queries and
  output rotated back (`had128w/had_block` `mm3_kernels.cu:200-243`, applied in
  `sparse_attn_kernel:785-800,884-901`; **all rows rotate**, per `--kv-rot` bit
  0 = K, bit 1 = V, defaults kv_k=kv_v=1 INT8, rot=3). Index keys BF16.
  No sink handling in B either. — Design-level difference (B's own KV scheme),
  not a convention error; B's KV is finer-grained than A's NVFP4.

## Divergences found (complete list)

1. **Pad-logit masking missing in B.** A writes -1e30 over head rows
   >= MINIMAX_TOKENS (200061 of 200064 — "the head's last 3 of 200,064 rows are
   padding", `ds4_minimax_bind.inc:47-48`; kernel `mm_logits_kernel`
   `ds4_minimax_gpu.cuh:3037-3050`, called `ds4_minimax_graph.inc:542`). B
   samples over all 200064 rows with no mask (`mm3_model.cu:3339-3351`,
   `mm3_sample.hpp:112-116`). A sick model can emit ids 200061-200063.
2. **Rope phase precision.** A double-exact cos/sin table
   (`ds4_minimax_gpu.cuh:1207-1217`); B F32 `cosf/sinf(pos*inv)`
   (`mm3_kernels.cu:191-193`) — error grows with position; A stays exact at 1M.
3. **BF16-rounding of activations.** A rounds every activation to the BF16 grid
   at every reference rounding point (`glmvq_round` everywhere); B keeps F32
   rows and rounds only where a tensor is stored (KV, index keys, FP16 GEMM
   inputs). Systematic sub-ulp drift through 60 layers.
4. **Fast transcendentals in B's expert path.** `__expf` in `oai`
   (`mm3_vq.cu:312`) and `expf` router sigmoid vs A's double sigmoid
   (`ds4_minimax_gpu.cuh:517`). ~2^-21 relative.
5. **MoE accumulation order.** A sums experts in ascending id
   (`mm_decode_route` rank placement, `ds4_minimax_gpu.cuh:47-57`); B sums in
   top-k selection-rank order (`combine_kernel:1242`, `mm3_vq.cu:444`).
6. **Head-input grid.** A o_proj input BF16-rounded; B F32 (token) / FP16
   (prompt) (`mm3_model.cu:2874`, `:2557`).
7. **Residual adds unrounded.** A rounds each branch and each residual sum to
   BF16 (`mm_residual_kernel:87`); B plain F32 `+=` (`add_kernel:1249`).

## THE DIVERGENCE LIST — ranked by word-salad likelihood

None of these is a wrong-context or wrong-weight semantic error; the eleven
stages above MATCH. Ranked by plausible contribution to "in-distribution
tokens, wrong context use, loops on word-suffix tokens":

1. **Pad-logit masking (divergence 1)** — only true convention gap in the
   token path. Fix: after the head GEMV/GEMM in `chunk_`, zero the last 3
   columns (rows 200061..200063) of every logits row, or filter them in
   `dist`/`dist_c`/`draw` (`mm3_main.cpp`), matching A's `mm_logits_kernel`
   `ds4_minimax_gpu.cuh:3037-3050`.
2. **Rope precision at long context (divergence 2)** — wrong *relative*
   phases degrade attention selectivity as context grows; loops on
   word-suffix tokens under greedy is the classic presentation. Fix: build A's
   per-position F32 cos/sin tables in double (`mm_rope_kernel` semantics) and
   read them in `norm_rope`, or compute `sincos((double)pos * inv)`.
3. **The 60-layer BF16-grid drift (divergences 3, 4, 5, 6, 7)** — individually
   sub-ulp, jointly a KL-scale deviation from the proven artifact. Fix (one
   line each): `glmvq_round`-equivalent (`__float2bfloat16_rn` + widen) at A's
   rounding points in `rms_kernel`, `norm_rope`, `swoai/oai`, `combine` /
   `add`; ascending-expert sort in the plan (or sum by expert id) in
   `combine_kernel`/`mm3_vq.cu combine_kernel`.
4. **Not in the diffed stages (named boundary).** Every stage above matching
   means the salad driver is not a forward convention: the next executable
   probes are (a) bit-compare per-layer residual streams — B already has
   `MM3_DUMP_HS` (`mm3_model.cu:3287-3296`) and A `DS4_MINIMAX_TRACE`
   (`ds4_minimax_graph.inc:179-195`); the first diverging layer names the
   stage (trunk GEMV, KV tier, or a runtime race in the stream forks
   `fork/join`, `mm3_model.cu:993-1003`); (b) run B with `--kv-cache bf16
   --kv-rot off` and STRATA_MM3_MSA_OLD=1 to isolate the KV/Hadamard/MSA-span
   paths; (c) `--spec 0` to exclude the draft window path.

Also verified green in this pass: TF32 off (`CUBLAS_PEDANTIC_MATH`), norm +1
bake present for every norm including per-head q/k/index and the final norm,
GVR1 residual magic check on-device, routes_ok guard, greedy spec-verify
correctness, position/KV alignment across rejected drafts.
