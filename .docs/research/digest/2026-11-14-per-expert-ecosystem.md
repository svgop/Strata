---
type: research
title: Per-expert MoE artifact ecosystem — NVFP4/MXFP4, Baekpica VQ family, convertible checkpoints, adoption targets
date: 2026-11-14
engine-context: strata (RTX 5090 32 GB VRAM sm_120, 184 GiB RAM, ~150 GiB expert-blob pool, per-layer/per-expert loader, fused ffn_*_exps rejected)
method: HF API (author listings, blob inventories, safetensors header + index.json probes via HTTP range), GitHub API/PR search (gh), web search
labels: [measured] = directly verified against the artifact; [inference] = derived, not directly measured; [boundary] = named capability gap
---

# Per-expert MoE artifact ecosystem digest — 2026-11-14

## 1. NVFP4 / MXFP4 landscape

### 1.1 Who serves NVFP4/MXFP4 MoE weights today

| Engine | NVFP4 | MXFP4 | Evidence |
|---|---|---|---|
| vLLM | Yes, native | via formats | Kernels at `csrc/quantization/fp4/nvfp4_quant_kernels.cu`, `nvfp4_quant_entry.cu`, `nvfp4_scaled_mm_entry.cu`; `vllm/models/flint/gguf_nvfp4.py` (GGUF NVFP4 reader); per-vendor MoE paths `vllm/models/minimax_m3/{nvidia,amd}/indexer_nvfp4.py` [measured, code search] |
| SGLang | Yes | — | NVIDIA NVFP4 model cards state "ready for inference with SGLang and vLLM" (e.g. nvidia/GLM-5.2-NVFP4, Oct 2026) [measured, search result] |
| TensorRT-LLM | Yes | Yes | `cpp/tensorrt_llm/kernels/marlin/marlin_nvfp4_gemm.cu` + `marlin_nvfp4.h`, fused-MoE paths `tensorrt_llm/_torch/moe/fused_moe/{cutlass,marlin}/nvfp4.py`, `w4a16_nvfp4.py` [measured, code search] |
| llama.cpp | Landing (merged CUDA/Vulkan kernels; converters merged; model wiring partially open) | Yes in mainline (gpt-oss class), Blackwell work ongoing | See 1.2 [measured] |
| KTransformers | No hits | No hits | `gh search code "nvfp4 OR mxfp4" --repo kvcache-ai/KTransformers` returned zero files [boundary: code-search only, may miss renamed symbols] |
| ISTA-DASLab MoESQ | NVFP4 W4A4 + paired-4:8 sparsity | — | Requires **patched vLLM v0.30.0** via MoESQ `integrations/vllm/install.sh` with CUTLASS grouped-GEMM for SM100/SM120; explicitly "does not load in upstream vLLM, TensorRT-LLM, or llama.cpp" [measured, HF card]. GitHub repo `ISTA-DASLab/MoESQ` returns 404 as of 2026-11-14 [boundary: tooling referenced but not retrievable; patched vLLM unverified] |

### 1.2 llama.cpp PR state (all verified via gh API)

Merged — NVFP4 kernels and conversion (port sources inside our already-vendored lineage):
- `#29857` CUDA: optimize accumulation in MMQ for NVFP4 (2026-10-05)
- `#29173` CUDA: handle compute type for NVFP4 on cublas path (2026-10-01)
- `#26048` CUDA: fuse w_s scale in epilogue MMQ for nvfp4 checkpoints (2026-11-12)
- `#31296` CUDA: fuse NVFP4 gate/up matmuls and GLU in MMQ (2026-11-05)
- `#31079` Vulkan: fix OOB NVFP4 UE4M3 scale lookup (2026-11-03); `dequant_nvfp4.comp` and `dequant_mxfp4.comp` exist in `ggml-vulkan/vulkan-shaders/` [measured, tree]
- `#28636` Convert: support compressed-tensors mixed-precision NVFP4 checkpoint → GGUF (merged 2026-10-09; validated on dense unsloth/Qwen3.8-27B-NVFP4; touches `conversion/base.py` only)
- `#31427` conversion: use original tensor names in `_flush_nvfp4_experts` (2026-11-03) and `#31611` conversion: resolve quantized_layers from source tensor names (2026-11-05) — both in `conversion/base.py` [measured]
- Model-side NVFP4 GGUFs merged: `#31608` LFM3, `#31657` qwen2 NVFP4 scales, `#31449` Granite NVFP4 scales, `#31097` minimax-m3 forward quant scales (2026-10-24)
- `#31954` Hexagon: IQ2_XXS and NVFP4 (2026-11-14)

Merged — MXFP4 kernels (Blackwell-relevant): `#31075` Metal faster MXFP4 mat-vec, `#29770` Metal bf16 math mxfp4 mul-mat, `#29809` SYCL MXFP4 MoE arithmetic decoding + weight reordering, `#30691` OpenCL MXFP4 MoE (Adreno), `#29483` WebGPU MMVQ incl. MXFP4, `#31776` SpacemiT MXFP4 repack. CUDA MXFP4 instances exist: `ggml/src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu` and `mmq-instance-nvfp4.cu` [measured, tree].

Open — watch list:
- `#31192` **model: add MiniMax-M3.1** — converter for the mixed MXFP8+NVFP4 checkpoint keeps routed experts **native NVFP4 with a per-expert `1/global_scale` tensor** (or exact dequant to BF16/F16/F32/Q8_*). This is the first mainline-GGUF path with per-expert-indexable expert tensors. OPEN as of 2026-11-14 [measured, PR body].
- `#26869` quantization: complete MXFP4 and NVFP4; `#25153` imatrix-aware NVFP4 quantization (scale search) + wire NVFP4 ftype; `#26704` CUDA experimental SM120 CUTLASS MoE prefill for MXFP4+NVFP4; `#26311` CUDA linear weight repack + MMVQ for NVFP4; `#26159` Blackwell NVFP4 MoE work scheduling (+10–15% prefill); `#31790`/`#30788` NVFP4 MMVQ work; `#31236` MXFP4/NVFP4 rounding ties-to-even; `#27736` loader fix for MTP NVFP4 scales.
- `#32187` model: add Inkling 2 (stacked on Inkling 1 `#25731`) — banded flash attention with per-token sinks; useful reference for an Inkling-2 port [measured, PR body].
- `#31236`-adjacent Reddit: "llama.cpp experimental native MXFP4 support for Blackwell" discussion (PTQ-based) — https://www.reddit.com/r/LocalLLaMA/comments/1pwybg6/llamacpp_experimental_native_mxfp4_support_for

### 1.3 Open kernel implementations we could port

- **llama.cpp (MIT, already our vendor lineage)**: `ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu`, `mmq-instance-mxfp4.cu`; `ggml-vulkan/vulkan-shaders/dequant_nvfp4.comp`, `dequant_mxfp4.comp`; OpenCL MoE MXFP4 kernels (`gemm_moe_mxfp4_*.cl`, `gemv_moe_mxfp4_f32.cl`, `mul_mv_id_mxfp4_*.cl`) [measured, HEAD tree].
- **vLLM (Apache-2.0)**: `csrc/quantization/fp4/nvfp4_quant_kernels.cu` (quant), `nvfp4_scaled_mm_entry.cu` (CUTLASS-backed scaled MM), benchmarks under `benchmarks/kernels/` [measured].
- **TRT-LLM (Apache-2.0)**: Marlin dequant-GEMM `cpp/tensorrt_llm/kernels/marlin/marlin_nvfp4_gemm.cu` — Marlin-style weight-only dequant GEMM is the closest match to a per-expert GEMV streaming loader; CUTLASS fused-MoE variants under `tensorrt_llm/_torch/moe/fused_moe/` [measured].
- Format math is small: NVFP4 = E2M1 values + FP8-E4M3 scale per 16 elements + FP32 per-tensor scale; MXFP4 = E2M1 + E8M0 scale per 32 [measured, NVIDIA developer blog + HF MoESQ card].

### 1.4 Quality vs bpw

- NVIDIA recipe claims: NVFP4 matches BF16-class accuracy at 2.3x throughput (recipe blog + whitepaper). NVFP4-QAD report: KL-divergence distillation recovers NVFP4 MoE accuracy, especially MoE models [measured claims by NVIDIA, not independent].
- NVFP4 (16-elem blocks, FP8 scales) vs MXFP4 (32-elem, E8M0 scales): NVFP4 finer scale granularity, reported better accuracy at same calibration; MXFP4 closes most of the gap. Counterpoint: an independent Blackwell benchmark found INT4-AWQ beat NVFP4 on speed in 3/4 models [search-level evidence, no primary URL].
- **P48NVFP4-MoESQ (ISTA-DASLab, Qwen3.8-Flash-Next)**: routed experts only, NVFP4 W4A4 + paired-4:8 sparsity = **2.75 bpw effective incl. mask/scales** (2.0 bpw kept values). Receipts [measured, HF card]: OpenLLM v1 79.03 vs BF16 82.09 (96.3% recovery); LiveCodeBench v6 93.9% recovery at 131k tokens; **SWE-bench Verified 84.0% recovery** — agentic/code capability degrades hardest. Experts 225.0 GiB BF16 → 38.7 GiB.
- **ISTA-DASLab GSQ-RCO (DeepSeek-V4.1-Flash-1017)** [measured, HF card]: IQ1_M 1.75 bpw → 98.7% task-avg recovery; **IQ2_XXS 2.00 bpw → 100.0%** (AIME25 100.00, GPQA-D 90.91, LCB v6 89.71); IQ2_XS 2.25 bpw → 99.8%. No KL/PPL published. Dense tensors Q5_K–Q8_0; experts IQ1_S–IQ3_S; Engram tables IQ4_NL (110.6 GiB, separate shard).
- **2-bit VQ class receipts** (see §2): MiniMax VQ KL 0.150/88.2% top-1; Beam KL 0.144/87.7%; MiMo KL 0.088/92.2%; GLM KL 0.288/81.4%. NVFP4 at 4.0 bpw carries ~2x the bits of these packs — quality-per-byte comparison favors VQ/GSQ packs at equal pool budget [inference from cited receipts].
- MXFP4 vs Q4_K_M GGUF: community reports MXFP4 lower perplexity (r/LocalLLaMA, gpt-oss era) [search-level].

### 1.5 Per-expert NVFP4/MXFP4 checkpoints — do they exist?

Yes, and the important ones are official. Verified by safetensors header/index probes [all measured]:

- **MiniMaxAI/MiniMax-M3.1-Flash** (official, 48 shards, 236 GB class): 89,632 tensors in index; **87,894 expert tensors, ALL per-expert-indexed** — `language_model.model.layers.{L}.block_sparse_moe.experts.{0..127}.w1|w2|w3.{weight_packed,weight_scale,input_global_scale,weight_global_scale}`; **zero fused `_exps` tensors**. Shared experts stored as `shared_experts.{gate,up,down}_proj.weight(+weight_scale_inv)` (MXFP8-class block scales). README/Baekpica card: "routed experts NVFP4, attention/dense/shared MXFP8". Per-expert indexable: YES.
- **moonshotai/Kimi-K3** (official, 96 shards): 494,592 routed per-expert tensors (`block_sparse_moe.experts.{0..895}.w1|w2|w3.weight_packed/weight_scale`), 896 experts/layer, NVFP4-style packing. Per-expert indexable: YES. (Model far exceeds our pool at any format — excluded on size.)
- **thinkingmachines/Inkling-2-Small** (official, MXFP8): stacked `model.llm.layers.N.mlp.experts.w13_weight*` / `w2_weight*` (+`_amax`, `_outer_scale_denominator`; dtypes F8_E4M3/U8/F32). Stacked along expert axis — sliceable dim-0, no per-expert tensor names.
- **Qwen/Qwen3.8-Flash-Next** (official, 131 shards): fused-stack `model.language_model.layers.N.mlp.experts.gate_up_proj` / `down_proj` — same stacked class, sliceable dim-0.
- Converters: llama.cpp `#28636` (merged) converts compressed-tensors NVFP4 checkpoints (dense-validated); `#31192` (open) is the per-expert NVFP4 GGUF writer for MiniMax. The YoungAi/DS4 toolchain (§4) already converts the MiniMax NVFP4 official checkpoint into per-layer VQ blobs — proven by the shipped MiniMax-M3.1-Flash-VQ-GGUF we serve today [measured].

## 2. Baekpica family inventory

Author listing: 58 public repos (2023 vicunas + 2026 MoE line). Two product lines: **VQ-GGUF** (routed experts vector-quantized, per-layer blobs or single-GGUF `vqblob`) and **Mixed-Quant-GGUF** (conventional GGUF mixes) plus **REAP/REAM expert-pruned** BF16/NVFP4 safetensors packs (K-EXAONE, Solar-Open2, DeepSeek-V4-Flash-0731-120B-REAM-104E).

Runtime: **ds4-dfm-rs** (github.com/Baekpica/ds4-dfm-rs, MIT, Rust host + CUDA `.cuh` kernel families, pushed 2026-11-14, 19 stars) — continuation of antirez/ds4 "DwarfStar". Mainline llama.cpp/Ollama/Transformers do NOT read the VQ formats [stated on every card, measured]. Quantizer: **fodelf/YoungAi** gguf-tools (`v41_quantize`, `v41_to_gguf`; MIT incl. GGML notice; author Wenzhou Wu/fodelf; repo described "DeepSeek 41 Flash local inference engine for Metal and CUDA", pushed 2026-11-02) [measured].

VQ format (from `crates/ds4-core/tests/fixtures/vq/vq_fmt.h`, v2.2/v3) [measured, source read]:
- Per-layer blob `dql_vq_L%02d.bin` or GGUF overlay `blk.L.ffn_exps_vq.blob`: `[u32 'DQVL'][u32 ver][u32 L][u32 nexp][nexp x 3 u64 payload offsets]` — **per-expert per-matrix (w1/w3/w2) slot table inside the blob header** (`ds4vq_slot(blob, e, which)`), nexp read from header (384 for V4.1, 256 for older).
- Per-matrix payload `DQVQ`/`DQV3`: `[dim:u16][nc:u16][rows:u32][cols:u32][codebook nc x dim f16][row gains rows x f16][index bitstream]`; v3 = one E4M3 codebook per layer hoisted to blob header + 13-bit indices split as 12-bit stream + 1-bit plane. Version whitelist — unknown versions hard-stop at load.
- Zero-dependency C dequant (`ds4vq_dequant_f32`/`f16`) in the same header — **directly portable; codebook-gather VQ is exactly our loader's admissible class** [inference: matches our kernel constraints trivially].

### VQ pack table [measured: sizes via HF blob API; receipts quoted from cards]

| Repo | Base (license) | Total | Expert layout | Recipe / bpw (routed experts) | Fidelity receipts | Runtime |
|---|---|---|---|---|---|---|
| MiniMax-M3.1-Flash-VQ-GGUF | MiniMaxAI/MiniMax-M3.1-Flash (MiniMax community, non-commercial free) | 115.6 GB / 61 parts | `layer-03..59-experts.gguf` (57) + body-nonexpert + embed/vision BF16 + dspark-q8_0 | vq8 r13x57+r8x20: 13-bit K8192 all 57 layers, 8-bit residual on 20; 1.63/2.63 bpw, **1.98 avg**; trunk Q4_K/Q8_0 | **KL 0.150, 88.2% top-1** (assistant turns); CharXiv RQ 77.5 vs 82.5 | ds4-dfm-rs PRs #110/#126 |
| GLM-5.3-Flash-VQ-GGUF | zai-org/GLM-5.3-Flash (MIT) | 94.6 GB / 56 parts | `GLM-5.3-Flash-VQ-layer-03..44.gguf` (42; NOTE: different name pattern, same per-layer class) + native-01..09 + vision + MTP 4 shards | GVQ1 container v2; R13 (13-bit K8192, 1.625 bpw) on 23 layers; R13+8 (2.625 bpw) on 19; **~2.1 bpw avg**; 316 Q4_K matrices | PPL/KL/top-1 3.789 / **0.288 / 81.4%** (this rev) vs original 3.508; all-R13-class ref 0.151; per-case KL table published | ds4-dfm-rs PR #96 (GVQ1 v2) |
| GLM-5.3-Flash-Uncensored-VQ-GGUF | orcarouter/GLM-5.3-Flash-Uncensored-FP8 (MIT) | same as above | same | same | not re-published | same |
| DeepSeek-V4.1-Flash-1017-VQ-GGUF | deepseek-ai/DeepSeek-V4.1-Flash-1017 (MIT) | 317.6 GB total: backbone 113.6 GB **single GGUF** (40 byte-range parts, 1,000 tensors, 43 `vqblob` type-42 tensors = 107.75 GB) + vision 0.97 GB + **Engram 2 x 101.5 GB official safetensors** | vqblob class: NOT per-layer blobs; slot tables give per-expert offsets inside each vqblob | YoungAi `vq8sh14-q4k-mtpnative`: 8-dim VQ, E4M3 shared codebook/layer, F16 row gains; layers 0-13 K8192 (13-bit), 14-39 K4096 (12-bit); **1.625/1.5 index bits**; trunk q4_K; MTP 3 DSpark towers 12-bit (+2.56 GB) | weight cosine 0.9253 (L0-13) / 0.9107 (L14-39); reference V4.1 receipts top-1 69-79% by domain, KL 0.42-0.69; **1017 fidelity not yet re-published** [boundary] | ds4-dfm-rs PRs #80/#83 |
| DeepSeek-V4.1-Flash-1017-iVQ-GGUF | same (MIT) | same class | same | iVQ = codebooks+gains refit on 3.65M calib tokens + imatrix; same bits | KL 0.369 → **0.350** (-4.9%), per-layer expert error ~-7% | same |
| DeepSeek-V4.1-Flash-1017-hVQ-GGUF | same (MIT) | same shapes | same | hVQ = GPTQ/GPTVQ-style error feedback (H = D^1/2 R D^1/2) | shares tensors/kernels with VQ release; used for the speed graph | same |
| DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF | dealignai FP8 uncensored (MIT) | same | same | same as VQ | n/a | same |
| Beam-1.0-501B-A23B-VQ-GGUF | reflection/Beam-1.0-501B-A23B (**Apache-2.0**, 501B-A23B, BF16 1.0 TB) | 99.9 GB / 52 parts (93.0 GiB) | `layer-03..51-experts.gguf` (49; L3-10 13-bit 2.05 GB, L11-51 12-bit 1.89 GB) + body-nonexpert Q4_K + embed BF16 + DFlash2-Q8_0 drafter | `beam1-vq8-mix8r13-41r12-q4k-embssd-v1`; 8-dim VQ, E4M3 shared book/layer, F16 gains, calibration-weighted Lloyd on 3.91M tokens; 1.625/1.5 index bits | **KL 0.144, 87.7% top-1** (5,491 tok; first build 0.295/82.9%); GB10 benchmarks vs BF16: GPQA-D 97.8%, AIME26 96.0%, IFBench 98.0%, AA-LCR 98.5%, LCB 95.1%, SciCode 88.0% | ds4-dfm-rs PRs #105/#106; FP8 KV required on 9 full-attn layers |
| A.X-K2-VQ-GGUF | skt/A.X-K2 (**Apache-2.0**, 688B/33B active, arXiv 2608.30181) | 107.7 GB / 62 parts | `layer-01..60-experts.gguf` (60) + body-nonexpert + embed BF16 | `axk2-vq8-r10x7-r9x53-wlloyd-q4k-q8idx-embssd-v1`: 10-bit K1024 x7 layers, 9-bit K512 x53; **1.144 bpw avg** | KL 0.754/0.727, top-1 71.6%, PPL 3.10→5.42; **card warns "severe quality loss"** | ds4-dfm-rs PRs #114/#125 |
| MiMo-V2.6-Flash-RL-VQ-GGUF | XiaomiMiMo/MiMo-V2.6-Flash-RL (MIT, 309B/15B active omni) | 98.6 GB / 53 parts (91.7 GiB) | `layer-NN-experts.gguf` (47) + body-nonexpert + audio/vision/dflash/mtp-q8_0 + embed BF16 | `mimo26f-vq8-r13x47-r8x32-q8-fp8kv-embssd-v1`: 13-bit all 47, 8-bit residual x32; **2.31 bpw avg**; Q8_0 trunk; FP8 KV | **KL 0.088, 92.2% top-1** (assistant turns; floor 0.020/97.8%); best receipts in family | ds4-dfm-rs PRs #127/#128 |
| MiMo-V2.6-Flash-RL-UNCENSORED / -MOPD-VQ | dealignai uncensored / MOPD variant (MIT) | same | same | same | n/a | same |
| Ling-3.1-flash-VQ-GGUF | inclusionAI/Ling-3.1-flash (MIT) | 116.1 GB | layer-expert blobs + body-nonexpert + mtp-native | v2 lineage (superseded by 1019) | n/a | ds4-dfm-rs |
| Ling-3.1-flash-1019-VQ-GGUF | inclusionAI/Ling-3.1-flash-1019 (MIT, 572B) | 115.4 GB / 72 parts (107.3 GiB GGUF) | `layer-04..72-experts.gguf` (69) + body-nonexpert + mtp-native + word-embeddings BF16 | `ling31-1019-ivq8-all12-q4k-embssd-v1`: 12-bit K4096 all MoE+MTP, 1.5 index bits; Q4_K trunk w/ column importance | KL 1.0877, 73.79% top-1 vs original (3.7M-token calibration); **card documents batch-sensitivity**: original-vs-itself split-batch KL 0.4014 [weak receipts vs family] | ds4-dfm-rs PRs #123/#124 |
| Inkling-2-Small-VQ-GGUF | thinkingmachines/Inkling-2-Small (**Apache-2.0**, 594B/17B active, 384E top-8, MXFP8 source) | 118.2 GB / 52 parts (110.1 GiB) | `layer-00..47-experts.gguf` (48; 102.85 GiB device-resident) + body-nonexpert + mtp-q8_0 + vision/embed BF16 | VQ dim-8, 12-bit K4096 E4M3, 1.5 bpw; trunk Q4_K importance-weighted; **NVFP4 K/V** (E2M1+E4M3/16); hVQ error feedback, 3.6M-token calibration | generated KL **0.163, 88.1% top-1**; images KL 0.088/95.3%; hVQ vs iVQ same codebooks: KL 0.249→0.156 (-37.4%); NVFP4 KV alone KL 0.0028 | ds4-dfm-rs PRs #151-#153 |

Also in the family (conventional GGUF, fused layouts): Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF (836 dl), GLM-5.3-Flash-Uncensored-Mixed-Quant, K2-Horizon-375B, Step-3.7-Flash, Darwin-180B-RSI, IQuest-Q1, Naive-N0.5, dots3-note-prev, Solar-Open2, Motif-3, Inkling-Small — plus REAP/REAM pruned NVFP4 safetensors (K-EXAONE 152B/145B, Solar-Open2 148B/120B, DeepSeek-V4-Flash-0731-120B-REAM-104E-NVFP4, compressed-tensors tagged).

Comparison vs what we serve today [inference from receipts]:
- **GLM-5.3-Flash VQ vs nothing**: we don't serve GLM; at 88 GiB it is the smallest pack in the family.
- **DeepSeek-V4.1-Flash VQ vs our GSQ-RCO IQ2_XS**: VQ backbone is ~106 GiB (fits pool) at ~1.5-1.6 index bpw but published 1017 fidelity is absent and V4.1-class receipts are top-1 69-79% / KL 0.42-0.69; GSQ-RCO publishes 99.8-100% task recovery at 2.0-2.25 bpw but totals 248.8-265.8 GB (backbone 138-155 GB + Engram 110.6 GB) — **exceeds our 150 GiB pool**; our in-tree GSQ-RCO IQ2_XS remains the only DS-V4.1 option that fits [inference]. VQ's Engram (203 GB) is SSD-row-streamed by ds4-dfm-rs; replicating that is a new mechanism for us [boundary].
- **MiMo/Beam/Inkling**: strictly new capability — no current strata equivalent.

## 3. Other per-expert publishers

- **ISTA-DASLab** (github.com/IST-DASLab/GSQ; papers arXiv:2604.18556 GSQ, arXiv:2605.00649 RCO, arXiv:2610.02241 MoESQ):
  - GSQ-RCO GGUFs — **fused layouts, mainline llama.cpp b11870+ compatible**; card states types are assigned "per layer (not per expert, which GGUF cannot express)" [measured]. New since our last sweep: DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF (2026-10-23, 3.76M dl) + -Coder- profile (2026-11-02); Qwen3.8-Flash-Next-GSQ-RCO-GGUF (2026-09-07) + -Coder- (2026-09-26). **No move off fused layouts** [measured]. Note: their card lists Strata (github.com/Niko1221/Strata) as a supported runtime for these builds.
  - P48NVFP4-MoESQ series (compressed-tensors, patched vLLM only): Kimi-K3 (978 GiB — out of class), Kimi-K2.5 (347.6 GB — out), Qwen3.5-397B-A17B (**155.3 GB / 144.6 GiB — fits pool**), Qwen3-30B-A3B (13.1 GB), Qwen3.8-Flash-Next (149.0 GiB disk; 95.4 GiB BF16 n-gram tables CPU-served). Storage schema per routed-expert linear: `weight_sparse_packed` u8 [out, K/4], `weight_sparse_mask` u8 [out, K/16], `weight_scale` f8e4m3 [out, K/32] + FP32 global scales — per-expert-indexable in principle, but no per-expert tensor names published and no portable kernel outside the patched vLLM [boundary].
  - FPQuant QAT NVFP4/MXFP4 small-model series (Oct 2025) — QAT receipts exist for dense 1.7B-32B models, not MoE.
- **staccs** — DeepSeek-V4.1-Flash-1017 VQ builds: `lecore-vq10sh12` single GGUF **99.12 GB** (12-bit VQ L0-13, 10-bit L14-39, MTP 10-bit, zero calibration; command `v41_quantize --skel q4k --mtp-vq --vq-nc 1024 --vq-nc-layers 0:14=4096 --vq-shared-cb --vq-cb-fp8`). Receipts [measured]: WikiText-2 PPL 3.615 vs 1.838 (ratio 1.967), KL mean 0.797 (median 0.195), top-1 74.46%, prob-mass 0.692 (below YoungAi's >=0.90 target); explicit comparison "Baekpica VQ 113.6 GB: 78.5% top-1, KL 0.694 — this build trades ~4 points of agreement for 15 GB less". Runtime: ds4-dfm-rs + uploader's 10/11-bit patch (PR in prep). Plus 4 abliterated lecore variants. Engram read from official shards at runtime.
- **shad0wsh00ck** — DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF (repack class).
- **Synchopate** — VQ3-64 CPU class with **llama-modchip** patched runtime (github.com/synchopate/llama-modchip): DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF (60.6 GB, 3-shard; LUT-VQ3-64 custom GGUF type **43** (24 B/64 W = 3.00 bpw, AVX-512 VBMI VPERMI2B decode), MLA in "Fractal-64" **repurposed GGML_TYPE_Q2_0 = type 42**, 3.75 bpw; routed experts transcoded from unsloth UD-IQ1_S donor; KLD 0.8409/0.4397 nats, 75.2% top-1, WT2 PPL 12.23); NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64-CPU (14.4 GB) + abliterated variant. **GGUF type-ID namespace collision: 42/43 mean different things in DS4-vqblob vs Synchopate files — validate by magic, never by type id** [measured, both cards; inference: loader hazard].
- **Jlfn** (ds4-dfm-rs DS-V4.1 port author) — gemma-4-26b-a4b-vq-gguf (10.7 GB), spark-x2.6-4b-vq-gguf: VQ class extends to small MoE.
- **HelloSun** — openPangu-VL-7B-VQ3-gguf.
- **Boundary**: no publisher found shipping per-expert/per-layer blobs loadable by **mainline** llama.cpp; the per-layer-blob class is the ds4-dfm-rs family (Baekpica, staccs, Jlfn, Synchopate-adjacent). r/LocalLLaMA announcement threads for Baekpica packs did not surface via search (vendor communicates via HF cards + ds4-dfm-rs PRs) [boundary: search recall, not absence].

## 4. Convertible checkpoints + tooling

### 4.1 Official checkpoint expert layouts (verified via index.json/header probes)

| Model | License | Routed-expert layout | Per-expert indexable? | Repo size |
|---|---|---|---|---|
| MiniMaxAI/MiniMax-M3.1-Flash | MiniMax community (non-commercial free) | NVFP4 packed, `block_sparse_moe.experts.E.w1|w2|w3.{weight_packed,weight_scale,*global_scale}`, 128 E | **YES, by tensor name** | 236 GB class |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | MIT | FP8, `layers.N.ffn.experts.E.w1|w3.weight(+.scale)`, 384 E (94,464 per-expert tensors = 41 MoE layers x 384 x 3 x 2) | **YES, by tensor name** | ~700 GB class (96,085 tensors in index) |
| zai-org/GLM-5.3-Flash | MIT | FP8, `mlp.experts.E.{down,gate,up}_proj.weight(+weight_scale_inv)` | **YES, by tensor name** | 328 GB |
| inclusionAI/Ling-3.1-flash (+1019) | MIT | FP8, `mlp.experts.E.{gate,up}_proj.weight(+scale_inv)`, 512 E | **YES, by tensor name** | 574 GB |
| moonshotai/Kimi-K3 | other | NVFP4-packed, `block_sparse_moe.experts.E.w1|w2|w3.{weight_packed,weight_scale}`, 896 E | **YES, by tensor name** | 96 shards (out of pool class) |
| moonshotai/Kimi-K2.5 | other | n/a (not probed; 1.03T params — out of class) | — | 595 GB |
| Qwen/Qwen3.8-Flash-Next | Qwen community | fused-stack `mlp.experts.gate_up_proj`/`down_proj` [E, ...] | Sliceable dim-0, no per-expert names | 131 shards |
| thinkingmachines/Inkling-2-Small | Apache-2.0 | MXFP8 stacked `mlp.experts.w13_weight*`/`w2_weight*` (+amax/outer scales) | Sliceable dim-0 | 393 GB |

### 4.2 Tooling that preserves the per-expert split

- **fodelf/YoungAi gguf-tools** (MIT incl. GGML notice): `v41_quantize` + `v41_to_gguf` — the only open converter chain producing per-layer expert blobs from official per-expert checkpoints (proven: DS-V4.1-VQ, MiniMax-VQ, and family all credit it) [measured].
- **Baekpica/ds4-dfm-rs** `vq_fmt.h`: zero-dependency C parse/dequant of the blob format + Rust per-family layout modules (`minimax_vq`, `glm_vq`, `ling31_vq`, `mimo2_vq`, `beam_vq`, `axk2_vq`, `inkling2_vq`, `gemma4_vq`) — MIT; the spec + decoder are the port seed for strata [measured].
- **llama.cpp mainline**: converts to FUSED `ffn_*_exps` for MoE; NVFP4 path (`conversion/base.py` `_flush_nvfp4_experts`, `#31427`/`#31611` merged) writes quantized expert tensors, and open PR `#31192` explicitly keeps MiniMax experts native NVFP4 with per-expert `1/global_scale` tensors — track this PR: if merged, mainline GGUF gains per-expert-indexable NVFP4 experts [measured PR body].
- **synchopate/llama-modchip**: patched runtime for VQ3-64/Fractal-64 types (CPU AVX-512); `scripts/build_dsv4_vq3.py` converter [measured, card].
- **Boundary**: no tool found that converts official per-expert checkpoints into our IQ2/IQ3/GSQ-RCO per-expert class directly; our in-house GSQ/RCO pack pipeline remains the owner of that step. KTransformers: no FP4 code found [boundary].

## 5. Ranked adoption targets (≤150 GiB pool, per-expert layouts)

Ranking = quality receipts per byte, license, loader fit, stream cost. Stream ceilings are [inference] from measured bytes/token over our ~50 GB/s DRAM lane.

1. **Baekpica/MiMo-V2.6-Flash-RL-VQ-GGUF** — MIT; 98.5 GB (91.7 GiB pool + small VRAM trunk); 309B/15B active = lightest per-token stream in family; **best measured fidelity in class (KL 0.088 / 92.2% top-1)**; omnimodal (vision+audio), MTP + dflash drafters included; 47 layers x 256 experts, same `layer-NN-experts.gguf` class we already serve. Port cost: `mimo2_vq` layout module + trunk Q8_0 (trivial). Risk: FP8 KV cache path required (GA layers).
2. **Baekpica/Beam-1.0-501B-A23B-VQ-GGUF** — Apache-2.0 (cleanest license); 99.9 GB (93.0 GiB); strongest published benchmark receipts (GPQA-D 97.8%, AIME 96.0%, IFBench 98.0% recovery vs BF16; KL 0.144/87.7%); DFlash2 drafter included. Risk: 23B active → measured 7.24 GB/token read ⇒ ~6.5-7 tok/s ceiling on our DRAM lane; FP8 KV mandatory on 9 full-attention layers (BF16 KV = 134.4 GiB).
3. **Baekpica/Inkling-2-Small-VQ-GGUF** — Apache-2.0; 110.1 GiB; thinkingmachines lineage, 594B/17B active; hVQ receipts KL 0.163/88.1% with hVQ-vs-iVQ ablation (-37.4% KL); NVFP4 K/V receipts measured. Risk: banded attention + per-token attention sinks = new arch kernels (llama.cpp `#32187` is an open reference implementation).
4. (Reserve) **Baekpica/GLM-5.3-Flash-VQ-GGUF** — MIT; smallest pool (88.0 GiB); vision + MTP; but current-rev fidelity (KL 0.288/81.4%) trails 1-3, and ~7.98 GB/token stream.

Explicit non-targets: A.X-K2-VQ (card warns severe quality loss at 1.144 bpw), Ling-1019-VQ (weak, batch-sensitive receipts), DS-V4.1 VQ/iVQ/hVQ (Engram 203 GB defeats the pool; single-GGUF vqblob class needs SSD-row Engram streaming — new mechanism; our GSQ-RCO IQ2_XS serves DS-V4.1 at higher measured fidelity), P48NVFP4-MoESQ (patched-vLLM-only, MoESQ repo 404), Kimi-K3/K2.5 (size), NVFP4-direct official MiniMax (only via PR #31192 class of loaders — watch, don't adopt yet).

**Plan-changing findings**: (a) the official MiniMax M3.1-Flash checkpoint is fully per-expert NVFP4 — a strata-native NVFP4 dequant/GEMV port (sources: llama.cpp mmq-instance-nvfp4.cu + vLLM fp4 kernels + TRT-LLM marlin_nvfp4_gemm.cu) gives us a second, higher-bpw serving tier (4.0 bpw experts) for the model we already serve at 1.98 bpw VQ; (b) ds4-dfm-rs `vq_fmt.h` is MIT and exactly our format class — the port is a header, not a project; (c) GGUF type-ID 42/43 collisions across publishers — validate by magic bytes; (d) if llama.cpp `#31192` merges, per-expert NVFP4 GGUF becomes a mainline artifact class.

## 6. Sources

### Hugging Face — Baekpica family
- https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF
- https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF
- https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF
- https://huggingface.co/Baekpica/Ling-3.1-flash-VQ-GGUF
- https://huggingface.co/Baekpica/Ling-3.1-flash-1019-VQ-GGUF
- https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF
- https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF
- https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF
- https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF
- https://huggingface.co/Baekpica/Beam-1.0-501B-A23B-VQ-GGUF
- https://huggingface.co/Baekpica/A.X-K2-VQ-GGUF
- https://huggingface.co/Baekpica/MiMo-V2.6-Flash-RL-VQ-GGUF
- https://huggingface.co/Baekpica/MiMo-V2.6-Flash-RL-UNCENSORED-VQ-GGUF
- https://huggingface.co/Baekpica/MiMo-V2.6-Flash-MOPD-VQ-GGUF
- https://huggingface.co/Baekpica/Inkling-2-Small-VQ-GGUF
- https://huggingface.co/Baekpica/DeepSeek-V4-Flash-0731-120B-REAM-104E-NVFP4 (REAM pruned NVFP4 line)
- https://huggingface.co/Baekpica/K-EXAONE-152B-A23B-REAP-80E-NVFP4 (REAP line)

### Hugging Face — official checkpoints (probed)
- https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
- https://huggingface.co/zai-org/GLM-5.3-Flash
- https://huggingface.co/inclusionAI/Ling-3.1-flash
- https://huggingface.co/inclusionAI/Ling-3.1-flash-1019
- https://huggingface.co/moonshotai/Kimi-K3
- https://huggingface.co/moonshotai/Kimi-K2.5
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://huggingface.co/thinkingmachines/Inkling-2-Small
- https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL
- https://huggingface.co/reflection/Beam-1.0-501B-A23B
- https://huggingface.co/skt/A.X-K2

### Hugging Face — ISTA-DASLab and other publishers
- https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
- https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF
- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
- https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-P48NVFP4-MoESQ
- https://huggingface.co/ISTA-DASLab/Kimi-K3-P48NVFP4-MoESQ
- https://huggingface.co/ISTA-DASLab/Kimi-K2.5-P48NVFP4-MoESQ
- https://huggingface.co/ISTA-DASLab/Qwen3.5-397B-A17B-P48NVFP4-MoESQ
- https://huggingface.co/ISTA-DASLab/Qwen3-30B-A3B-P48NVFP4-MoESQ
- https://huggingface.co/ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4 (FPQuant QAT series)
- https://huggingface.co/staccs/DeepSeek-V4.1-Flash-1017-lecore-vq10sh12-GGUF
- https://huggingface.co/shad0wsh00ck/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF
- https://huggingface.co/Synchopate/DeepSeek-V4-Flash-0731-VQ3-64GB-CPU-GGUF
- https://huggingface.co/Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64-CPU-GGUF
- https://huggingface.co/Jlfn/gemma-4-26b-a4b-vq-gguf
- https://huggingface.co/HelloSun/openPangu-VL-7B-VQ3-gguf
- https://huggingface.co/nvidia/GLM-5.2-NVFP4 (NVFP4 QAT-class release served by vLLM/SGLang)

### GitHub
- https://github.com/Baekpica/ds4-dfm-rs — MIT; format spec `crates/ds4-core/tests/fixtures/vq/vq_fmt.h`; kernel families `cuda/*.cuh`
- https://github.com/fodelf/YoungAi — MIT quantizer (`gguf-tools/`, `v41_quantize`, `v41_to_gguf`)
- https://github.com/IST-DASLab/GSQ
- https://github.com/Niko1221/Strata (MIT, public engine repo; listed as supported runtime on ISTA-DASLab GSQ-RCO cards)
- https://github.com/dedene/DeepSeek-V4.1-Flash-1017-DGX-Spark (ds4-dfm-rs serving example)
- llama.cpp PRs (verified merged/open states via gh API, Nov 2026): #31192 (MiniMax-M3.1, per-expert NVFP4 GGUF — open), #28636 (compressed-tensors NVFP4 convert — merged), #31427, #31611, #31097, #31608, #31657, #31449, #29857, #29173, #26048, #31296, #31079, #31954, #28413 (OCP FP8), #28392 (FP8 KV), #26869, #25153, #26704 (SM120 CUTLASS MoE), #26311, #26159, #31790, #30788, #31236, #22112, #24738, #27736, #20456, #32187 (Inkling 2 — open), #25731 (Inkling 1), #31075, #29770, #29809, #30691, #29483, #31776, #30643 — all under https://github.com/ggml-org/llama.cpp/pull/N
- vLLM NVFP4 kernel paths: https://github.com/vllm-project/vllm/tree/main/csrc/quantization/fp4 (`nvfp4_quant_kernels.cu`, `nvfp4_scaled_mm_entry.cu`, `nvfp4_quant_entry.cu`), `vllm/models/flint/gguf_nvfp4.py`
- TRT-LLM NVFP4 kernels: https://github.com/NVIDIA/TensorRT-LLM — `cpp/tensorrt_llm/kernels/marlin/marlin_nvfp4_gemm.cu`, `tensorrt_llm/_torch/moe/fused_moe/{cutlass,marlin}/nvfp4.py`
- MoESQ tooling: referenced as `ISTA-DASLab/MoESQ` on HF cards — GitHub 404 as of 2026-11-14 [boundary]

### Papers and blogs
- NVFP4 format/recipe: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-training/
- NVFP4 recipe throughput: https://developer.nvidia.com/blog/nvfp4-recipe-same-accuracy-with-23x-higher-throughput/
- NVIDIA Research 4-bit: https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit
- NVFP4 whitepaper: https://research.nvidia.com/publication/nvfp4-efficient-and-accurate-low-precision-training
- NVFP4 QAD report (KL distillation for MoE): https://research.nvidia.com/labs/nemotron (NVFP4-QAD-Report.pdf)
- GSQ paper: arXiv:2604.18556; RCO paper: arXiv:2605.00649; MoESQ paper: arXiv:2610.02241; REAM: arXiv:2604.04356; REAP: arXiv:2510.13999; A.X-K2 tech report: arXiv:2608.30181
- r/LocalLLaMA MXFP4 Blackwell discussion: https://www.reddit.com/r/LocalLLaMA/comments/1pwybg6/llamacpp_experimental_native_mxfp4_support_for
- GSQ-RCO Qwen3.8-Flash-Next coverage: https://richland.au/blog/a-1769b-moe-in-296-gb-of-ram-ista-daslab-ships-gsq-rco-ggufs-for-qwen38-flash-next
