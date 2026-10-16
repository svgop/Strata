# SUB-WARP GROUPED EXPERT KERNELS — port plan (eddo 674bf0ca, T10 #1; plan written 2026-10-16)

**Prize**: prof221's biggest stable kernel line = VRAM expert hits 1.87/3.26 ms per window
(light/heavy). Eddo's measured deltas on OUR model+quant class (Qwen3.8-Flash-Next IQ3_S,
RTX 3090): experts 36.7→32.7 µs/layer @1 tok, 75.3→56.1 @4, 133.9→94.3 @8; e2e IQ3_S decode
+5.9% code / +5.7% think / +6.4% after-16K. Bitwise by construction; expert_grouped_parity.

**The trick** (from their kernel, read in full):
- A row's dot no longer takes a whole warp: a G-lane sub-group (G=16 for gate/up "gus80",
  G=8 for IQ4_NL down, G=4 for Q2_0 down) owns the row. Lane t of the G-group stands for the
  warp's lanes t + G*j — it sums those virtual lanes' calls in the ORIGINAL row_dot order and
  folds the J=32/G partial sums pairwise (register adds in warp_sum's xor order), then a
  G-wide shfl_xor — every float is the old kernel's bitwise value.
- The weight block is DECODED ONCE for NE entries (the multi-token window's activations
  x[0..NE)), and an activation block is FETCHED ONCE for R rows (R=2: gate+up share the
  decode). AMORTIZATION: R×NE reuse of each weight decode = the whole win (weights dominate;
  activations are tiny at 1-8 tokens).
- gus80: n_embd 2560 = 80 calls (2560/32); a half-warp per h-column, 2 blocks/SM, 64 regs,
  codebook (iq lut) in shared memory. down: n_ff 640, 8/4 lanes per row, ≤4 entries a pass.

**Why it is NOT a cherry-pick**: their kernels build on `iq_dot.cuh` — a load/fetch/apply
split of EVERY i-quant dot (Split<TY>::load/fetch/apply with the format's codebook behind a
pointer) — the output of a ~4k-line refactor of their iq_kernels.cu. OUR iq_kernels.cu kept
the original monolithic kernel shapes. Port = re-implement the sub-warp trick against OUR
kernel bodies (keep our layouts/summation orders EXACTLY, only change the thread-to-work
mapping per the lane-substitution rule above).

**OUR formats** (lay.fmt per layer, IQ3_S pack): gate/up = IQ3_S-family i-quants (the
gus80 kernel's supported set: IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS, Q2_0 — covers us), down =
IQ4_NL + Q2_0. Our grouped launch: `native_expert_grouped` (verify.cpp grouped()) — the
wiring point is inside our native grouped kernels, called with (L, gp, gs, gn, p_dst, p_tok,
cap, xq, scratch, dst). OUR parity harnesses: expert_layout_test / gguf parity + the
window-hash discipline.

**Implementation steps** (next session):
1. Extract their iq_dot.cuh + the two kernels (git show 674bf0ca) as REFERENCE ONLY.
2. In our iq_kernels.cu (or a new file beside it, included by native_moe.cu's grouped path):
   write `rows_sub<TY,G,NS,R,NE>` with OUR dot bodies spliced into the load/apply split for
   the formats our pack actually uses (IQ3_S gate/up first — the most common; then Q2_0/IQ4_NL
   down). The lane-substitution + pairwise-fold order rule is the ONLY thing that must match
   their trick; the arithmetic inside load/apply must match OUR current kernels.
3. Wire: native_gus80-style launch in native_expert_grouped's gate/up stage when
   n_embd==2560 && entries>=1 (env STRATA_EXPERT_SUBWARP=1, default off), same for down.
4. Parity: per-format expert_grouped_parity (theirs ships one; adapt to our harness) +
   window-hash boot vs off.
5. ABAB 3v3 (arm = STRATA_EXPERT_SUBWARP=1, ctl = standing). Expect +3-6% M1/pair/conc if
   their transfer holds; the VRAM-hits line in the next profile map should shrink visibly.

**Also banked while pricing this**:
- T10 #2 draft-proxy RE-PRICED (found during integration design): our standing draft mode is
  the F-184 COUPLED sampler (coupled_draft_sample over the FULL logit array, target Philox +
  penalties); the proxy's clean top-1+prob output only fits the non-coupled greedy path.
  Faithful port = extend rescore_kernel to emit top-K exact (id, logit) + softmax normalizer,
  restrict coupled sampling to the K set with a penalties/min-p fallback to the full head.
  A day-class surgery for ~+2-3% (head line 0.43-0.78 ms/window halved) — worth it only AFTER
  the sub-warp kernels.
- T10 #6 MMQ dense projections at 7-8-token windows applies to OUR M1 SERIAL WINDOWS (our
  boots capture 6/7/8-token windows — confirmed in today's logs): +5.7% IQ3_S at 8-token
  windows on their rig. Dense projections (qkv/gate/out-proj) through MMQ tensor cores in the
  big serial windows = a real M1 lane; medium port (MMQ ctx into the captured graph; we
  already build moe_mmq).

## APPENDIX — the MMQ-at-7-8-token-windows lane (eddo 3443d955, recon 2026-10-16)

Their diff is only +90 lines of WIRING (a proj() lambda routing the window's big dense
projections — GDN qkv/gate/out, QSA q/k/v/out — through quantize-to-q8_1 + a per-stream MMQ
context's run_dense when n>=7 and n_in/n_out>=n_embd and type!=Q8_0; pre-capture warmup of
every shape because the context's stream-k scratch pools may not allocate inside a capture —
our F-91 law's cousin). Measured +5.7% IQ3_S at 8-token windows (their rig); MMQ 1.1-1.65x
dp4a at these shapes (scratch/aq_bench.cu).

OUR SIDE: include/strata/prefill/moe_mmq.hpp exists with quantize/supported/fits/q8_bytes
and the expert-grouped Product API — but NOT their per-stream dense run_dense context.
Port path: host the dense product on OUR Product API (one virtual "expert", identity bounds
{0,T}, no expert stride) per stream (main + shared-expert branch), or add a thin dense entry
beside Product. Wire at our verify.cpp projection sites behind STRATA_WINDOW_MMQ=1.

**VALIDATION IS THE STORY**: MMQ (int8 tensor-core mma) is NOT bitwise our dp4a path — the
first non-bit-exact kernel swap in the campaign's verify window. Gates: M3b in band + MMLU
battery + mimic + ABAB perf; the exact-match verify CONTRACT is untouched (draft vs target
consistency is within-window), but acceptance may shift a fraction of a point — measure it.
