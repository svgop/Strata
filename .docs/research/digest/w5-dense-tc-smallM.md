---
type: digest
id: w5-dense-tc-scout
created: 2026-10-31
status: evidence
tracker: W5 — tensor-core arms for wide windows
scope: small-M (1-8 rows) tensor-core GEMM/GEMV dispatch in weight-only-quant kernels; target HW RTX 5090 (sm_120)
---

# W5 — dense TC small-M competitor digest

## 1. llama.cpp CUDA MMQ (int8 MMA, exact integer dots)

- **Mechanism.** MMQ quantizes activations (src1) once to q8_1 multi-column tiles, dequantizes K-quant weights to int8 in-kernel, and runs exact int8 `mma` on Turing+ (`turing_mma_available(cc)`); pre-Turing falls to dp4a MMQ. Tile = one block computes J output columns (J ∈ {32,64,128}); host picks best J via `ggml_cuda_mmq_get_config`; 8 warps; hard floor: 48 KiB smem/block else BLAS fallback.
- **M-crossover.** Source-verified dispatch (`ggml/src/ggml-cuda/mmq.cu` L435): `if (turing_mma_available(cc)) return true;` — MMQ (TC) used for ALL M on CC >= 7.5. MMVQ (dp4a GEMV) capped at `MMVQ_MAX_BATCH_SIZE = 8` with per-quant-type per-arch crossovers: Q3_K 4 (old)/5 (Turing+), Q5_K 5, Q6_K 4, IQ4_XS 5 (mmvq.cu L150-188). Pre-Turing: dp4a MMQ only while `ne11 < MMQ_DP4A_MAX_BATCH_SIZE (=64)`, else dequant+cuBLAS.
- **The PRs.** Int8 TC MMQ for our exact K-quants: #7860 q4_K/q5_K/q6_K (merged 2024-06-11), #7921 q2_K/q3_K (2024-06-14), #8062 TC perf tuning, #8100 writeback fix, #8075 "use MMQ instead of cuBLAS by default" (2024-06-24). Per-type MMVQ→MMQ crossover table: #26079 (merged 2026-08-20); retunes #28285 (sm_87), #28912 (sm_70).
- **A-operand/scale handling.** `block_q8_1_mmq`: activations grouped into 128-value blocks, transposed into smem-contiguous layout, 16-byte pad prevents bank conflicts; pad bytes carry per-32 scales (`d4` fp32 / `ds4` half2 scale+partial-sum / `d2s6`). K-quant super-block scales stay W-side, applied fp32 per super-block after integer dots.
- **Staging.** Activation tiles staged via load-tile kernels (`mmq-load-tiles.cuh`); unmerged PR #28572 pipelines NVFP4 tile loads with cp.async + TMA on Blackwell: +14% pp.
- **Blackwell sm_120 notes.** Issue #24399: out-of-range smem store in `mul_mat_q<Q8_0,128>` MMA write-back epilogue on RTX 5090 (fixed); PR #28784: IQ1_S/IQ2_S/IQ3_S garbage output on sm_120; #19662: MXFP4 "mma with block scale" unsupported; mmq.cu L127: NVFP4/MXFP4 can take native W4A4 `blackwell_mma_available` path; GB10 (DGX Spark) has its own MMVQ param set.
- Numbers/URLs: https://github.com/ggml-org/llama.cpp/pull/7860, /pull/26079, /pull/28572, /issues/24399; https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh
- **Confidence: high** (source read today, lines cited).

## 2. Marlin (W4A16, IST-DASLab)

- **Mechanism.** FP16 tensor cores on pre-shuffled W4: weights packed offline into 16x64 tiles laid out for ideal loads, INT4 pairs interleaved (pattern 64207531) so dequant = one `lop3` (extract 4 bits + set exponent) + fp16 subtract with offset-8 folded in; `mma.sync` 16x16x8 fp16. Sync-free: dequant interleaved with TC issue so both pipelines saturate.
- **M-crossover.** Design point bs 16-64. Roofline: A10 FLOPs/byte ~200 → memory-bound until b_opt ~50, so TC tiles pay off only then. ~3.9x vs FP16 up to bs 16-32, decaying toward 1.5x at bs 128. Kernel runs any M, but dedicated GEMV (AWQ-style) stays near-optimal (3.87x) at bs 1 — GEMV wins bs 1, Marlin wins 4-64 (SqueezeBits confirms on H100).
- **A-operand/scale handling.** A stays fp16 (never quantized); group-128 scales in fp16 applied in registers post-dequant before TC operand.
- **Staging.** cp.async global→smem, pipeline depth P=4, double buffering, `evict_first` hint; activations served from L2 and reused across registers.
- Numbers: ~3.9x peak (0.125-bit scale overhead accounted), 2.8x e2e vLLM at bs16. https://arxiv.org/abs/2408.11743, https://github.com/IST-DASLab/marlin
- **Confidence: high** (paper + README verified).

## 3. Machete (sm90 W4A16, Neural Magic)

- **Mechanism.** Hopper wgmma + TMA + warp specialization (producer/consumer warps). Weights prepacked offline (`machete_prepack_B`, CUTLASS CuTe layout algebra) into interleaved layout enabling 128-bit smem loads; dequant upconverts 4-bit to fp16 **in registers**, deliberately avoiding smem round-trip. Weights placed as wgmma A operand via Y^T = W^T X^T; plain `mma` would forfeit ~37% of Hopper peak.
- **M-crossover.** Large-batch kernel by design: "outperforms other mixed-input kernels for batch/prefill seq len 128+"; sub-32 batch explicitly listed as future work. vLLM gates it to exactly CC 90 (`is_device_capability(90)`) — dead on sm_120.
- **A-operand/scale handling.** A fp16; group scales (gs 64/128) + optional zeropoints applied post-scale so FMA order holds.
- **Staging.** TMA global→smem subtensors; multiple generated ScheduleConfigs (tile shapes, scheduler) with per-problem-shape default heuristic (`generate.py`), never one config for all shapes.
- Numbers: e2e Llama-3.1-70B 1xH100 +29% input tok/s, +32% output; 405B 4xH100 +42%. https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel, https://github.com/vllm-project/vllm/blob/main/csrc/quantization/machete/Readme.md
- **Confidence: high** (blog + vLLM source verified). Note: task premise "smem-staged dequant" is inverted — staging is smem, dequant is registers.

## 4. TensorRT-LLM weight-only kernels

- **Mechanism.** Two-path split: CUTLASS mixed-input `fpA_intB` persistent GEMM (sm90 + sm100 builders, incl. sm100 UMMA weight-only collectives) for prefill/context; `weightOnlyBatchedGemv` family (per-type-pair dispatchers, `InterleavedForHopper` weight layouts, `cudaCoreGemm` tile configs) for decode. PyTorch backend W4A16/W8A16 linear calls `torch.ops.trtllm.weight_only_quant_gemm` (linear.py L2978).
- **M-crossover.** Official rationale: "batch size <= 4 → memory-bound → weight-only quant preferred" (quantization blog). Exact gemv→cutlass M threshold not located in source today — named gap. New `low_m_gemm` backend (2026): row-specialized GEMM for M <= 32, exact-M row tiles for M <= 8, autotuner buckets {1..8,16,32}, `TRTLLM_LOW_M_GEMM_BACKEND` — validates per-row-count kernel specialization instead of padding.
- **A-operand/scale handling.** A fp16/bf16 (or fp8/int8 for W4A8 variants); scales per-channel or groupwise, `scaleonly` vs `scalebias` forms; Hopper layout interleaving moves scale handling into TC-friendly layout.
- **Staging.** CUTLASS persistent kernels, cp.async pipelines; sm100 UMMA builders for Blackwell datacenter; consumer sm_120 not addressed in weight-only builders reviewed.
- Numbers: weight-only ~2x at bs 1 both frameworks; TRT-LLM beats vLLM 1.18x/1.15x at bs 256 (prefill/decode-heavy). https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md, https://blog.squeezebits.com/vllm-vs-tensorrtllm-6-weightonly-quantization-33728, https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv, https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/low_m_gemm.py
- **Confidence: medium-high** (paths + gates verified; gemv→GEMM threshold unverified).

## 5. vLLM GPTQ/AWQ marlin paths + Blackwell tuning notes

- **Dispatch.** Marlin `get_min_capability() = 75` (Turing+, inline PTX, CUDA-only); Machete locked to CC 90. 2026 main moved kernels to `vllm/model_executor/kernels/linear/mixed_precision/{marlin,machete}.py`; marlin variants also serve mxfp4/nvfp4 (incl. Blackwell mxint4 MoE test `test_marlin_vs_trtllm_mxint4.py`).
- **Measured crossover (SqueezeBits, H100).** bs 1: ExLlamaV2 GEMV beats Marlin/Machete. bs 4-16: Marlin ~2x ExLlamaV2, beats TRT-LLM best. bs 64-256: Machete best of vLLM, TRT-LLM best overall. Weight-only ~2x over FP16 at bs 1 for both frameworks.
- **Blackwell/RTX 5090 (llama.cpp).** sm_120 mma-MMQ shipped with real bugs: smem OOB in MMA writeback epilogue (#24399), IQ_K garbage output (PR #28784), MXFP4 block-scale mma not compilable (#19662). Field trend: cp.async/TMA tile-load pipelining is the active Blackwell lever (+14% pp, PR #28572, unmerged). https://github.com/ggml-org/llama.cpp/issues/24399
- **Confidence: high** (capability gates read from source; benchmark numbers from SqueezeBits lab data).

## What this changes on the row

- Unmask wider: MMQ-MMA-any-M is field-proven on Turing+ (llama.cpp mmq.cu L435); our [4,8] env gate is more conservative than the ecosystem — but first fix the T=1-3 regression, not force the unmask.
- The 0.72-0.9x at T=1 is a tile-config gap, not a TC-vs-dp4a law: llama.cpp solves it with per-type J-width (32/64/128) + ntiles_x selection; add a narrow-J tile variant for T <= 4 and per-type crossover tuning (Q3_K 4/5, Q5_K 5, Q6_K 4, IQ4_XS 5 is their calibrated answer).
- Keep exact int8 dots and copy `block_q8_1_mmq` activation staging (transposed q8_1, 16-byte pad carrying per-32 scales) — the fp16-dequant route (Marlin/Machete lop3) is a different design point that discards our K-quant mixture advantage.
- Dry-run the MMA write-back epilogue for sm_120 smem bounds before any unmask: RTX 5090 already produced an OOB crash in exactly that code path upstream (#24399) plus IQ_K garbage (#28784).
- Keep the cp.async ring; evaluate TMA for the Blackwell weight tile loads — upstream's only unmerged +14% pp Blackwell win is cp.async/TMA tile-load pipelining (#28572).
- Grouped TC expert GEMM for T >= 4: port Marlin-style fp16-dequant grouped GEMM (vLLM `marlin_moe`, min CC 75) rather than Machete — Machete is hard-gated to CC 90 and dead on sm_120; TRT-LLM's exact-M<=8 row tiles endorse our T-granular dispatch.
