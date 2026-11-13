# Frontier quant-formats-and-kernels audit (2026-11-13) — GGUF zoo / FP4-Blackwell / Marlin lineage vs our tree

Batch 1 of the 18-agent frontier wave.

## Hardware boundaries pinned (gate every claim)
sm_120 (5090): 5th-gen TCs with FP8 e4m3/e5m2 + FP4 e2m1+microscaling at 4:1 FP4:FP8, via the LEGACY mma
pipeline. tcgen05/TMEM is sm_100/103 datacenter-only — DeepGEMM-class blockscaled/nvfp4-tcgen05 ideas are
boundaries, not port targets. The FP4 path on sm_120 = m16n8k64 mma (confirmed in llama.cpp PR #17906 text).

## Ranked findings
1. **Few-row MMQ (BS 2-24, PR #31533)** is upstream's answer to the band F-379 blocked — but the community
   35% is upstream-MMQ-vs-upstream-MMVQ; OUR MMVQ already decodes weights once and applies via IL rows
   (theirs re-decodes per column). Honest transfer: 5-15% on multi-token windows, ~0 at T=1. Action:
   **MMVQ-12 (F-379's named project) first**; revisit MMQ only past T=12.
2. **FP4 tensor-core path unexploited** — PR #17906 MXFP4 activations + m16n8k64 FP4 mma: 1.26-1.31x PREFILL
   on 5090-class, PPL unchanged. Our MXFP4 rows are dp4a/WMMA-F16 class (ds4_gguf.cu:274,767; NVFP4 rows
   iq_kernels.cu:583-614 dp4a). Prefill-only gain; parity-harness-gated (nvcc 13.x mma = the miscompile law's
   neighborhood). MED port.
3. **GGUF types we cannot load** (live enum verified): Q1_0=41 (new, trivial Q2_0-clone rows), F8_E4M3=43
   (new, dense rows), ik's IQ*_K lattice (port medium-high, only if an ik GGUF arrives). RECORD CORRECTION:
   the circulating "IQ*_K merged Feb 2026" claim is FALSE — PR #19726 closed unmerged (verified).
4. Mixed-input GEMM lineage (Marlin/Machete) dead on sm_120 (Hopper-only wgmma; vLLM sm_120 failures) — our
   decode lane is the correct form; CUTLASS SM120 blockscaled grouped GEMM relevant only behind finding 2's
   re-pack. Boundary documented.
5. Activation quant current (q8_1 + IL rows; fp32-scale smem staging at moe_fused_iq.cu:128-150 is
   DeepGEMM-class); FP4 activations ride finding 2. No action.
6. **AVX-512 on the 9950X3D: full 512-bit + VNNI (Zen 5)** — our IQ AVX-512 rows are decode-once multi-token
   (ahead of ggml-cpu's re-decode-per-token single-token AVX2); fold VNNI into the IQ rows the way q2_avx2
   already does (we ship what upstream PR #26348 still has open). Small, compile-gated.
7. Scale formulations: nothing to change; our documented departures are parity-proven correct (do not
   "modernize" Q5/Q4 offset-on-quantized-sums, trellis int8 codebook, NVFP4 UE4M3 decode).
8. NVIDIA's vendor +50-90% tg on 5090 decomposes into PRs we mirror or can cheaply arm: PDL is IN-TREE
   (dense_tc.cu pdl_trigger) — the cheap action is **arm+A/B PDL on the expert decode chain** (env-gated).
9. Already-ahead list (do not regress): decode-once multi-column MMVQ with bitwise parity; IL tables + IL-Q8
   (sm_120 rows a conservative Q6_K mirror with a marked emit slot awaiting measurement — the priced ILQ8
   package); EXPDEC smem windows; NVFP4 rows; Q2_0-at-42 + shadow store + VNNI rows; prefill staged
   decode-once kernels; GSQ-RCO runtime dispatch; nvcc-law hygiene (13.2 miscompile, 12.8 spill; X2 TF32
   stays falsified).

## Ranked actions
1. MMVQ-12 (unlocks T 9-12; F-379 costed)  2. PDL arm/A/B on expert chain  3. FP4 mma prefill path
4. Q1_0 + F8_E4M3 loaders  5. VNNI in IQ AVX-512 rows  6. measure the IL-Q8 emit slot.

## Boundary
Every decode-lane gap shrinks toward zero under the byte-bound law at 96% VRAM hit; the gains that matter are
prefill (FP4 1.26-1.31x) and batch-window (MMVQ-12). Vendor/community numbers on unspecified hardware are not
sm_120 facts until benched here.

Sources: llama.cpp PRs #17906/#31533/#19726(closed)/#30715/#26348/#30278; NVIDIA Blackwell whitepaper; NVIDIA
forums (FP4 ratio; no TMEM on consumer); lna-lab blackwell-geforce-nvfp4-gemm; GSQ arXiv:2604.18556; RCO
arXiv:2605.00649; jangwook.net (false-merge correction); Red Hat Machete; vLLM #16131; DeepSeek-V3 report;
NVIDIA TE FP8 blockwise; OpenBenchmarking 9950X3D; Phoronix EPYC AVX-512.
