# Frontier numerics/formulations audit (2026-11-13) — below-kernel math vs our formulations

Batch 2 final of the first twelve. Verdict up front: the engine is overwhelmingly reference-faithful — most
sites are pinned bit-exact transcriptions, no ordering bug found. The findings are divergences, gaps, and
the FOURTH built-but-unarmed instance of the wave.

## Ranked findings
1. **Same weights, two codebooks** — decode/expert uses the int8-requantized TQ trellis codebook
   (iq_kernels.cu:668-676, tq_lut_i8.cuh; RMS 0.85% of codebook std, per-expert L1 bound 3e-2) while
   prefill uses fp16 (tq_lut_f16) — the same expert computes different dots in one forward pass; the file
   itself says "agree within tolerance, not bitwise". Align prefill to int8 (frontier MMQ also standardized
   on int8). MED; parity rerun.
2. **Router softmax runs in FP64 on the CUDA path — 512 software-emulated double exps per token per layer,
   one-thread serial double sum, double shared arrays (2x smem) — while the PROVEN bit-identical fast
   kernel (router_top10_fast_kernel with the bracket test) is __HIPCC__-gated ONLY** (router_top10.cu:
   105-115 slow path; 283-383 fast kernel; 416-419 variant returns false on CUDA). FP64 exp on GeForce is
   ~1/32 rate. Port the fast kernel's launch path + parity harness to CUDA. MED; ids/weights bit-identical
   required; the bracket test is the ready-made argument. THE fourth built-but-unarmed find.
3. Mixed expf/__expf for identical formulations across paths (GDN decay gate: native_gdn.cu:64 expf vs
   verify_kernels.cu:389,521 __expf; SwiGLU ds4 expf vs moe_fused __expf) — verify-mode vs native-mode drift
   in last bits; each consistent only with its own harness. Pick one per formulation, re-bless both. LOW.
4. **Sampler chain lacks typical_p, xtc, top_n_sigma — silently ignored if requested** (sampler.cu:3-21:
   penalties -> top_k -> top_p -> min_p -> temperature only; grep-verified). llama.cpp's 2026 canonical
   chain interleaves all three. Each is a short walk over the existing sel list. MED.
5. STRATA_ROUTE_RESIDENT swap path drops the 2^-14 renorm clamp (native_router.cu:304-309; the clamp is
   documented load-bearing in router_top10.cu:15-17) — mathematically unreachable for real top-10
   distributions; one-line fmax for contract hygiene. Trivial.
6. compress_kernel recomputes expf(score-m) three times per element (ds4_kernels.cu:609-617) — hoist to
   registers. Trivial.
7. Sampler distribution math deliberately MORE precise than reference (double sums; 1/t multiply vs /t) —
   boundary tokens can differ vs llama.cpp chain; document as intended. Zero cost.
8. FP64 shared arrays halve occupancy for no accuracy need — folds into 2.
9. GDN prescale (MSInfer) implemented, default-off, correctly prioritized (bandwidth saving structurally
   absent with register-resident state). No action.

## Formulations already right (negatives, banked)
Online softmax w/ sink-in-denominator (StreamingLLM form); flash attn fp32 accum + FMA-folded rescale;
RMSNorm rsqrtf(mean+eps) everywhere; norm+quant fusion (gdn_q8_1_store; SwiGLU+amax+int8 fused — the
vLLM/sgl-kernel frontier); MoE combine BITWISE-pinned to ggml incl. the ZADD fold; router exact lowest-index
tie-breaking in every variant + fp32 GEMV over bf16 weights read directly; full YaRN (freq_scale, ext_factor,
corr_dims, mscale) with verified blend direction; GDN exp-decay not RoPE; TQ/IQ exact int32 dp4a with
documented overflow bounds; E4M3/E2M1 hardware-exact decode w/ OCP-FN saturation; KV q8 scale-vs-stored-f16;
temp-0 dedicated greedy kernels; radix top-k deterministic tie order; s26_tsum bitwise-proven packed sums;
DeepSeek FP8 discipline (fp32 where the frontier promotes); no Kahan needed at these lengths.

Sources: UW flash-attn notes; FLASH-D; FA4; StreamingLLM; DeepSeek-V3 TR; Colfax FP8; NVIDIA FP8 intro; HF
MoE handbook; Rank-Aware PTQ; Megatron Core MoE; YaRN; vLLM context-extension + quantization docs; sgl-kernel;
QTIP/TCQ discussion; 2026 quant-format survey; optimi Kahan; low-precision survey; ggml rope.cu.
