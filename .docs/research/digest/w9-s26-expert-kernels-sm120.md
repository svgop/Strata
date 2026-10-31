---
type: digest
id: w9-s26-scout
created: 2026-10-31
status: evidence
topic: W9 - S26 expert kernels on sm_120 (RTX 5090): IQ3_S/IQ4_NL LUT handling and quantized-GEMV tuning in reference implementations
scope: read-only research; no repo files modified besides this digest
---

## 1. llama.cpp vecdotq.cuh - IQ3_S / IQ4_NL LUT addressing
- LUT mechanism (IQ3_S): direct gathers `iq3s_grid[qs[l0] | ((qh << (8-l0)) & 0x100)]`; the 9th grid-index bit is folded from `qh` by shift+mask. Signs branchless: `__vcmpne4(...,0)` then `__vsub4(g ^ s, s)`. No `__ldg`, no smem, no `__byte_perm` on this path.
- IQ4_NL/XS: no memory LUT at all - the 16-entry `kvalues_iq4nl` is emulated in registers by chained `__byte_perm`: 4 uniform `table32[0..3]` loads + selector `(0x32103210 | ((q4 & 0x88888888) >> 1))`, loop i in {0,1} with shift 16*i, final reorder `__byte_perm(tmp[0],tmp[1], 0x6420/0x7531)`. Our `ds4_iq.cu` `tab16()` is this exact form.
- LUT storage: `GGML_TABLE_BEGIN` (ggml-common.h:493) = `static const __device__` - device GLOBAL (L1/L2 cached), not `__constant__`. iq3s_grid = 512 x u32 = 2 KB; kvalues_iq4nl = 16 bytes.
- Staging: none anywhere in upstream IQ/K-quant vecdot paths. Confidence: High (raw source read, master 2026-10-31).
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/vecdotq.cuh

## 2. llama.cpp mmvq.cu - multi-column kernels and per-arch tuning
- Kernel: `<type, ncols_dst 1-8, has_fusion, small_k, halve_iters>`; block = (warp_size, nwarps), `__launch_bounds__(nwarps*warp_size, 1)`. Weight bytes are re-read once and reused across ncols_dst activations per thread (the multi-row reuse trick our engine ported).
- smem: partial sums only (`tmp_shared[nwarps-1][ncols_dst][rows_per_block][warp_size]`) - upstream stages NO activations and NO LUT in smem.
- Per-arch `calc_nwarps` tables: GENERIC 4 (nrc 1-4) / 2 (nrc 5-8) / 1; TURING: 2 for Q2_K..Q6_K at nrc=1 else 4; RDNA4: nrc=1 -> 8 warps for IQ4_NL/IQ4_XS; GB10/DGX Spark: 2x generic + `prefetch.global.L2` asm + halved K loop for IQ4_NL/IQ4_XS/Q4_K/Q5_K/Q6_K/IQ3_S... (halve_iters only on GB10).
- Blackwell crossover (tuned on RTX 5090): `ggml_cuda_should_use_mmvq` uses MMVQ while `ne11 <= 5` (Q2_K/Q3_K/Q4_K), `<= 6` (Q5_K), `<= 7` (Q6_K); past that MMQ/mma. Per-type `get_mmvq_mmid_max_batch(ggml_type, cc)` tables exist for MoE MUL_MAT_ID.
- Confidence: High (raw source read). https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mmvq.cu

## 3. llama.cpp sm_120 / RTX 5090 incidents (the decision-relevant layer)
- nvcc 13.2 sm_120 miscompile: byte extraction via `uint8_t*` from a packed u32 word drops the byte mask -> OOB iq3s_grid index -> IQ3_S garbage. Affects BOTH MMQ and dp4a vecdot (`GGML_CUDA_FORCE_CUBLAS=1` does not help). Discriminator: presence of IQ3_S tensors, not model arch. CUDA 13.3 CLOSES it; 13.1 unaffected (RTX PRO 6000 green on 13.1/13.3). Issue #28581 still OPEN; fix PR #28784 (convert to explicit `__byte_perm`) closed unmerged. https://github.com/ggml-org/llama.cpp/issues/28581
- nvcc 12.8 sm_120a local-memory spill: `int8_t*` read of a dequantized word kept in local memory -> 128-336 B stack frame -> FA 2.1x slower; RTX 5090 decode 100.5 -> 171.9 t/s at 16k ctx (1.71x), 2.07x at 32k. Fix (PR #30077, open): read with shifts, never byte-pointer, on `__CUDA_ARCH__ >= 1000`. https://github.com/ggml-org/llama.cpp/pull/30077
- Maintainer (JohannesGaessler) rejected the Blackwell cuBLAS fallback for IQ quants (PR #28823): "This is almost certainly the wrong fix" -> upstream arms dp4a IQ kernels on Blackwell consumer. https://github.com/ggml-org/llama.cpp/pull/28823
- SM120 IQ MMVQ multi-column is NOT batch-invariant at n=5 and n=8 vs n single-column calls (issue #28111, open): per-ncols specialization changes accumulation order; lowering the MMVQ/MMQ threshold makes it worse. https://github.com/ggml-org/llama.cpp/issues/28111
- Measured decode on 5090 (issue #28196): GDDR7 roofline 1792 GB/s; dense Q4_K_M 27B decodes at 28% of roofline on Windows path vs 86% on 4090, while qwen3moe 30B.A3B hits 134 t/s (control) - MoE decode path itself is healthy. https://github.com/ggml-org/llama.cpp/issues/28196
- FA stream-k bug: 5090 has 170 SMs (prime count) -> block-count divisibility cost 10-13% prefill; +37-40% when fixed (issue #30784). https://github.com/ggml-org/llama.cpp/issues/30784

## 4. Marlin (IST-DASLab) - W4A16 on tensor cores, no LUT at all
- Dequant is pure register arithmetic, credited to FasterTransformer cutlass `interleaved_numeric_conversion.h`: `lop3.b32` with imm `(0xf0 & 0xcc) | 0xaa` (0xEA), masks `LO=0x000f000f` / `HI=0x00f000f0`, magic `EX=0x64006400` (fp16 exponent injection), then `__hsub2(0x64086408)` / `__hfma2(0x2c002c00, 0xd480d480)` -> 4 fp16 per u32, 1 shift total. Comment: explicit lop3 because "the compiler does not seem to automatically recognize it".
- Fragments feed `ldmatrix.sync.aligned.m8n8.x4.shared.b16` int8-mma path; dequant overlapped with matmul ("m dimension inner loop"). GPTQ group scales applied per fragment.
- Relevance: this is the alternative mechanism to LUT gathers for NIBBLE quants (IQ4_NL-like); IQ3_S's 512-entry grid has no lop3 equivalent (9-bit index, codebook values not nibble-affine).
- Confidence: High (source read). https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu

## 5. AWQ kernels (mit-han-lab) - scalar GEMV contrast + same lop3 idiom
- `dequantize_s4_to_fp16x2` (dequantize.cuh, "Modified from NVIDIA FasterTransformer"): same lop3 0xEA / 0x64006400 / fma trick, 1 shift, then sub/fma.f16x2.
- GEMV (`gemv_cuda.cu` g64/g128): grid.z batches, blockDim.y covers output channels, scalar `scaling_factor * (w - z) * act` per element - no dp4a, no LUT, no smem; slow reference class vs llama.cpp-style dp4a GEMV.
- Confidence: High (source read). https://github.com/mit-han-lab/llm-awq/tree/main/awq/kernels/csrc/quantization

## 6. NVIDIA dp4a / Blackwell consumer (sm_120) guidance
- dp4a (`__dp4a`) requires sm_61+ and carries to sm_120; working MMVQ dp4a paths on 5090 in current llama.cpp confirm empirically. INT8 tensor cores via `mma.sync` also available on sm_120 (5th-gen TC); no native INT4 TC mode (NVIDIA dev forum). Decode GEMV is bandwidth-bound: consensus (llama.cpp practice + community analyses) is dp4a for small-M decode, mma for prefill; tensor cores cannot help when bandwidth-limited, and dp4a avoids activation requantization. Confidence: Medium (forum + secondary analyses; no NVIDIA PR doc fetched).
- RTX 5090: 170 SMs / 21,760 cores, 32 GB GDDR7 512-bit, 1792 GB/s, 96 MB L2 (GB202 whitepaper; 128 MB only on full die / RTX PRO 6000). Confidence: Medium-High (whitepaper figure via secondary summary; SM count high, from issue #30784 + spec sheets).
  https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf

## What this changes on the row
- Tracker premise is stale: in-tree, S26 is NOT gated to gfx1151 - `arch_defaults.cpp:51` defaults `STRATA_EXPERT_V2=1` and the dispatch at `iq_kernels.cu:4782` checks only env + geometry (gu 21 / d 20 / 2560 / 640), with LT=LX=true, RPW=4, on every backend; only the nontemporal loads are HIP-only (`iq_kernels.cu:3133`). The row's work is measure-and-tune on sm_120, not "arm".
- Arm-or-not risk is low and upstream-corroborated: dp4a IQ GEMV is the default Blackwell decode mechanism in llama.cpp and the cuBLAS fallback was rejected by the maintainer; keep S26 armed, measure vs native.
- Real sm_120 risk is toolchain, not kernel design: our `S26Split<21>`/`S26IQ3S` use the exact `uint8_t*`-from-u32 pattern miscompiled by nvcc 13.2 (#28581). Gate: build with CUDA >= 13.3 (this box: /usr/local/cuda-13.4, unaffected); for belt-and-braces, convert those byte reads to `__byte_perm`/shift form and keep the memcmp-vs-native parity harness per chunk.
- Expected mechanism of gain on sm_120: 2 KB `iq3s_grid` becomes L1-resident after first touch and the dependent-gather latency is hidden by the RPW=4 row interleave; LDS staging (LT) buys little on 128 B-line Blackwell and costs occupancy. Expected delta small (a few %); the IQ4_NL side has near-zero LUT traffic already (__byte_perm emulation).
- The per-block smem LUT fallback already exists (`ds4_iq.cu` `load_tab`, staged by 4 threads, consumed by `tab16`) - no new fallback needs building; only a dispatch toggle if S26-LDS measures slower than native+`load_tab`.
- Porting caution from #28111: multi-column/interleaved accumulation can be non-bit-stable per ncols on sm_120 - keep bitwise parity assertions per GRP_NC/RPW configuration, not just one shape.
