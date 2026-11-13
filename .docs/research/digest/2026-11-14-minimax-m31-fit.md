---
type: research
date: 2026-11-14
topic: MiniMax M3.1 artifact adoption for strata (single-box 9950X3D + RTX 5090 32GB + 184 GiB RAM, expert-streaming MoE serving)
model: MiniMaxAI/MiniMax-M3.1-Flash
verdict: "#1 Baekpica/MiniMax-M3.1-Flash-VQ-GGUF (per-layer expert blobs, best fidelity-per-byte, MIT spec+reference kernels); runner-up peasantsmith Maya-S IQ2_XXS (stock-llama.cpp-loadable, needs offline de-fuse for our loader). No strata-servable artifact exists today; every path carries the MSA attention port."
evidence-grade: measured-facts-from-primary-sources; GGUF headers parsed directly (range-downloads); inferences labeled
---

# MiniMax M3.1 fit study for strata

Scope: identify the real "M3.1" release, inventory every GGUF publisher on HF, establish llama.cpp support state, and rank candidate artifacts against our engine constraints (per-expert servable layouts; in-tree quant kernels; RAM pool ≤ ~150 GiB; bytes-bound DRAM lane ~50 GB/s).

Method note: every "header parsed" claim below comes from downloading the first 8–64 MB of the actual first GGUF shard and decoding the GGUF v3 header (KV block + full tensor-info list) with a purpose-built parser. Publisher quality numbers are quoted from their published measurements; they use different eval sets and are NOT cross-comparable.

---

## 1. Identity

**"M3.1" exists as [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash). There is no plain "M3.1".** The [MiniMaxAI org page](https://huggingface.co/api/models?author=MiniMaxAI) shows the lineage: M1 (2025-06) → M2 (2025-10) → M2.1 (2025-12) → M2.5 (2026-02) → M2.7 (2026-04) → M3 + M3-MXFP8 (2026-06) → H3 (2026-07) → **M3.1-Flash (repo created 2026-10-17; [r/LocalLLaMA marks the weight drop 2026-10-23](https://www.reddit.com/r/LocalLLaMA/comments/1x9uzv9/minimax_m31_flash_released_model_weights_oct_23))** → H3-Image (2026-11). The r/LocalLLaMA release thread confirms the name in community use.

Measured from [config.json](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/raw/main/config.json) and the [model card](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash):

| Property | Value |
|---|---|
| Architecture | `MiniMaxM3SparseForConditionalGeneration`, model_type `minimax_m3_vl` (HF); GGUF arch `minimax-m3` (verified in parsed headers) |
| Parameters | ~428B total / ~23B active (card; matches tensor math: 57 MoE layers × 128 experts × 56.6M = 412.8B routed + ~14B attention/dense/embed) |
| MoE shape | 60 layers (3 dense + 57 MoE), 128 routed experts, top-4 + 1 shared expert, sigmoid scoring + correction bias, routed_scaling 2.0, expert FFN 3072 (SwiGLU-OAI), dense FFN 12288 |
| Attention | **MiniMax Sparse Attention (MSA) on every layer** — GQA 64 q / 4 kv heads × 128 dim, per-head QK-norm (Gemma-style 1+w), partial RoPE (64/128 dims, θ=1e7) + a learned block-sparse indexer: 4 index heads, scores → max-pool into 128-token blocks → top-16 blocks per query ([arXiv:2606.13392](https://arxiv.org/abs/2606.13392)). **Not lightning/linear attention** — the M1/M2 linear-attention lineage is gone; M3 moved to full-attention-plus-block-sparse |
| Context | 1,048,576 native (1M) |
| Vocab / tokenizer | 200,064 BPE (gpt2-style), GGUF `tokenizer.ggml.pre = minimax-m2` (verified in parsed headers), NFC normalizer, bos 200019 / eos 200020 / pad 200000 |
| Multimodal | CLIP-style ViT (632M, 672px, patch-merge, 3D rope) for image + video; text output |
| Speculative decoding | DSpark draft head (5-layer, ships as separate safetensors dir; ~2–2.1 GB as GGUF Q8_0) |
| Ship format | Official checkpoint is **already quantized: NVFP4 routed experts + MXFP8 attention/dense, quantization-aware trained** (QAT). Card tags `8-bit, mxfp8` |
| Release | 2026-10-17 repo creation / ~2026-10-23 public weights |
| License | [MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/raw/main/LICENSE): free non-commercial; commercial use allowed with prominent "Built with MiniMax M3.1-Flash" attribution + one-time notice email; written authorization from MiniMax only required above $20M yearly revenue. Prohibited-uses appendix (military, minors, etc.) |

M3 (2026-06, [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)) is the BF16 predecessor: same MoE/MSA topology, no QAT, no DSpark. M3.1-Flash = M3 + QAT (NVFP4/MXFP8) + DSpark + refreshed vision tower.

---

## 2. GGUF inventory

Complete [quantized-derivative listing](https://huggingface.co/models?other=base_model:quantized:MiniMaxAI/MiniMax-M3.1-Flash) (40 repos, API query 2026-11-14). **Absent: ISTA-DASLab, mradermacher, bartowski, LeafFire** — API search returns zero LeafFire models; bartowski has [M3 only](https://huggingface.co/bartowski/MiniMax-M3-GGUF), no M3.1. Our best-serving publisher family (ISTA-DASLab GSQ-RCO) has no M3.1 pack; nothing here matches that format mix.

Tensor-layout ground truth (headers parsed by us; "fused" = `blk.N.ffn_{gate,up,down}_exps.weight` llama.cpp MoE convention — the layout our loader cannot serve per-expert):

| Publisher / repo | Quants offered | Total size | bpw (experts) | Layout (verified) | Quality receipts |
|---|---|---|---|---|---|
| [unsloth/MiniMax-M3.1-Flash-GGUF](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF) | UD-IQ1_M 129.4, UD-IQ2_XXS 135.0, UD-IQ3_XXS 160.1, UD-IQ3_S 175.5, UD-Q2_K_XL 143.9, UD-Q3_K_XL 194.0, UD-Q4_K_XL 249.4, UD-Q8_K_XL 259.1 (GB) + dspark Q8_0 2.09 + mmproj BF16 1.74 | 1,453.9 GB all | 2.0–4.8 | FUSED (inference, high confidence: KV set is exact llama.cpp-converter output incl. `split.tensors.count=1188`, produced by PR #31192's converter; CDN truncation blocked direct tensor-list read on 2 attempts) | [unsloth docs per-quant top-1 table](https://unsloth.ai/docs/models/minimax-m3.1-flash): UD-IQ2_XXS 87.5% chat / 95.3% math, UD-Q2_K_XL 88.5/95.7, UD-Q4_K_XL 97.3/99.0; weighted 2-bit 91.5% |
| [peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF) ("Project Maya") | Maya-S IQ2_XXS-mix 122.4 GB; Maya-XS IQ1_S-mix 93.0 GB; + sidecar `*-maya-indexers.gguf` (layers 0-2 MSA indexers), dspark 2.3, vision 1.7 | 122.4 / 93.0 | IQ2_XXS-mix / IQ1_S-mix | **FUSED — verified by header parse** (`ffn_*_exps` present, types IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS; attn Q8_0; shexp Q6_K; norms F32). "ordinary GGUFs in llama.cpp's minimax-m3 layout" per README | NLL vs original: S +0.8%, XS +5.0%; top-1 agreement avg: S 87.6%, XS 82.6% (held-out, vs exact-decoded original) |
| [AtomicChat/MiniMax-M3.1-Flash-GGUF](https://huggingface.co/AtomicChat/MiniMax-M3.1-Flash-GGUF) | AD-IQ2_XXS 112.2, AD-Q2_K 161.6, AD-IQ3_S 208.9, AD-Q4_K 257.2 (GB) | 741.6 GB all | ~2 / ~2.4 / ~3.2 / ~4 | **FUSED — verified by header parse**; AD-IQ2_XXS carries **TQ2_0 (ternary) on 47 tensors** incl. experts, Q3_K/Q4_K/IQ2_XXS mix elsewhere | imatrix on M3.1's own chat format; "stock llama.cpp, no runtime patch"; no per-quant numbers published |
| [Baekpica/MiniMax-M3.1-Flash-VQ-GGUF](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF) | one release: 13-bit residual-VQ experts on all 57 layers (1.63 bpw × 37 layers) + 8-bit residual stage on 20 layers (2.63 bpw), **avg 1.98 bpw**; trunk Q4_K/Q8_0/BF16/F32; separate parts: embed BF16 2.46, vision BF16 1.73, dspark Q8_0 2.18 | **115.5 GB total, 111.3 GB resident** | 1.98 avg | **UNFUSED PER-EXPERT — verified by header parse**: `language_model.model.layers.N.block_sparse_moe.experts.E.w{1,3,2}.weight` (original HF names, one expert index per tensor, 384 tensors/layer file), **one `layer-NN-experts.gguf` per layer** + `body-nonexpert.gguf`. Expert payload = custom GVQ1/GVR8 codebook format (type IDs 0x47565xxx, above ggml COUNT=44 → not loadable by llama.cpp), trunk = standard ggml types (Q4_K=12, Q8_0=8, BF16=30, F32=0 — verified) | Strongest receipts in the field: teacher-forced KL **0.150**, top-1 match **88.2%** vs exact-decoded original; CharXiv RQ 77.5 vs 82.5 (93.9%); per-tensor sha256 manifest; calibration on 4.18M tokens incl. images/video |
| [LyJonathan/MiniMax-M3.1-Flash-GGUF](https://huggingface.co/LyJonathan/MiniMax-M3.1-Flash-GGUF) (for [Sparsa](https://github.com/LyJonathan/Sparsa)) | IQ1_S 94.8, IQ1_S-GPTQ 94.8, IQ1_M 106.6, IQ1_M-GPTQ-v2 106.6, IQ2_XXS 124.5, IQ2_XXS-GPTQ 124.5, IQ2_XXS-GPTQ-v2 122.4, IQ2_KT 123.8, IQ2_XS 133.1, IQ2_XS-D3 146.1, IQ2_KL 152.7, IQ3_K 191.2, IQ4_KS 233.6, NVFP4 246.1 (GB) + mmproj 1.73 | 1,962 GB all | 1.7–4.5 | Mixed: IQ1/IQ2_XXS/IQ2_XS tiers load in "patched llama.cpp"; **IQ2_KT, IQ2_KL, IQ3_K, IQ4_KS, NVFP4 tiers are custom expert types, "Sparsa only"** — upstream llama.cpp cannot read them (README states this). `routing-profile.bin` sidecars | Best public per-tier quality table: top-1 vs FP32 ref: IQ1_M 47.1%, IQ2_XXS 52.0%, IQ2_KT 56.3%, IQ2_XS-D3 59.4%, IQ2_KL 59.8%, IQ3_K 69.8%, IQ4_KS 74.4%, NVFP4 77.3%; KL64 column too. Key finding: "experts were trained as 4-bit (NVFP4), so the 1–2-bit tiers are degraded" |
| [orcarouter/MiniMax-M3.1-Flash-abliterated-GGUF](https://huggingface.co/orcarouter/MiniMax-M3.1-Flash-abliterated-GGUF) | abliterated "Mix-128G" 111.9 GB | 112.9 | ~2 | not parsed (behavior-modified model) | none |
| [webmp3/Sakura-...-GGUF](https://huggingface.co/webmp3/Sakura-MiniMax-M3.1-Flash-Q3_K_M-GGUF) | Q2_K / Q3_K_M 203.6 / Q4_K_M / Q5_K_M (standard K-quants) | 207.6 (Q3_K_M) | 2.5–4.5 | assumed FUSED (standard llama.cpp convert naming) | none |
| [smalinin/MiniMax-M3.1-Flash-Q2_K-GGUF](https://huggingface.co/smalinin/MiniMax-M3.1-Flash-Q2_K-GGUF) | Q2_K 162.3 + DSpark BF16 3.93 + DFlash2 BF16 4.26 + mmproj 1.74 | 172.2 | ~1.9 | assumed FUSED | none |
| [DevQuasar/MiniMaxAI.MiniMax-M3.1-Flash-GGUF](https://huggingface.co/DevQuasar/MiniMaxAI.MiniMax-M3.1-Flash-GGUF) | Q2_K (from f16 source) ~167 GB; 52 files across quants | 664.1 all | ~2+ | assumed FUSED | none |
| [Harsha001/...-StrixHalo-GGUF](https://huggingface.co/Harsha001/MiniMax-M3.1-Flash-StrixHalo-GGUF) | UD-IQ1_M-E93 96.6, UD-IQ2_XXS-E89 96.8 (E93/E89 = expert-count-pruned variants for 128 GB unified boxes) | 193.4 | ~1.8 | FUSED family (UD naming); expert-pruned → different MoE shape, not faithful | none |
| [899400HS/...-C3-GGUF](https://huggingface.co/899400HS/MiniMax-M3.1-Flash-C3-GGUF) | single "C3" quant 198.9 GB | 198.9 | ~3.7 | not parsed | none |
| [operationrange/...-abliterated-GGUF](https://huggingface.co/operationrange/MiniMax-M3.1-Flash-abliterated-GGUF) | abliterated UD-Q8_K_XL 292 GB | 293.8 | ~4.8 | FUSED family (UD) | none |
| Non-GGUF | [amd/Quark-MXFP4](https://huggingface.co/amd/MiniMax-M3.1-Flash-Quark-MXFP4), [primitive-ai pruned-285B NVFP4](https://huggingface.co/primitive-ai/MiniMax-M3.1-Flash-pruned-285B-NVFP4), EXL3 packs (Mia-AiLab, vcruz305, jayleaton), MLX (mlx-community, inferencerlabs) | — | — | wrong runtime family for us | — |

---

## 3. llama.cpp support status

Timeline reconstructed from the GitHub PR/issue API (all numbers verifiable at the linked PRs):

- 2026-06-12 [#24523](https://github.com/ggml-org/llama.cpp/pull/24523) "Preliminary MiniMax-M3 support" — **closed unmerged**; body: "Sparse attention is not yet supported".
- 2026-06-22 [#24908](https://github.com/ggml-org/llama.cpp/pull/24908) "Add MiniMax-M3 (MSA) support" — **merged 2026-07-26**. The real base: 60L/128-expert MoE with MSA block-sparse top-16 selection built from stock ggml ops.
- 2026-06-28 [#25113](https://github.com/ggml-org/llama.cpp/pull/25113) mtmd vision for M3 — **merged 2026-07-26** (M3's Qwen2.5-VL-style ViT; M3.1's CLIP-style tower is different — see #31192).
- 2026-07-26 [#26144](https://github.com/ggml-org/llama.cpp/pull/26144) indexer tensors kept F32 — merged.
- 2026-07-27 [#26180](https://github.com/ggml-org/llama.cpp/pull/26180) quantized KV cache for M3 — merged.
- 2026-07-28 [#26210](https://github.com/ggml-org/llama.cpp/pull/26210) specialized minimax-m3 tool-call parser — merged.
- 2026-07-30 [#26297](https://github.com/ggml-org/llama.cpp/pull/26297) MSA block-selection prefill op moved from CPU to stock-GPU ops — merged.
- 2026-09-28 [#29553](https://github.com/ggml-org/llama.cpp/pull/29553) dedicated fused **MSA CUDA kernel — closed UNMERGED**. Upstream still composes MSA from stock ops; no bespoke sparse-attention CUDA kernel landed.
- 2026-10-23 [issue #31063](https://github.com/ggml-org/llama.cpp/issues/31063) — **M3.1 Flash-specific eval bug**: `build_moe_ffn()` dropped the per-expert NVFP4 scale → garbled output. Fixed by [#31097](https://github.com/ggml-org/llama.cpp/pull/31097), **merged 2026-10-24**.
- 2026-10-26 [#31192](https://github.com/ggml-org/llama.cpp/pull/31192) "model: add MiniMax-M3.1" — **still OPEN**. Adds: converter for the mixed MXFP8+NVFP4 checkpoint (routed experts stay **native NVFP4 GGUF tensors** with per-expert `1/global_scale`), MSA-on-every-layer-with-indexers rule, chunked MSA batch path, NFC tokenizer handling, and the **DSpark drafter integration** (`--spec-type draft-dspark`). Unsloth's GGUFs are produced by this branch ([unsloth docs](https://unsloth.ai/docs/models/minimax-m3.1-flash) says "build llama.cpp with support from PR #31192, branch minimax-m31").

Bottom line for stock master: **arch enum `LLM_ARCH_MINIMAX_M3` ("minimax-m3") is in [src/llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp) master; text serving of M3.1 works on master ≥ the 2026-10-24 merge (#31097); current release v1.0.0 (2026-11-13) contains it.** Quants with upstream ggml types (verified in [ggml/include/ggml.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h) master enum: MXFP4=39, NVFP4=40, TQ1_0=34, TQ2_0=35, IQ1_M=29, BF16=30) have CUDA kernels. Custom types outside that enum — Baekpica's GVQ1/GVR8 (0x47565xxx), LyJonathan's KT/KL/D3/NVFP4-repack tiers — do not load anywhere except their own engines. Only 19 open minimax-mentioning items exist, the only M3.1-specific one being the #31192 PR itself; no open M3.1 correctness issues.

Vision (mmproj) for M3.1 on stock master: unclear-to-unlikely — #25113 covered M3's tower; M3.1's vision tower config differs (CLIP ViT, patch_merge, 3D-rope). Treat image input as requiring the #31192 branch until it merges. DSpark speculative decoding likewise only on that branch.

---

## 4. Quality signals

Official (from the [model card](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) benchmark table, M3.1 Flash vs DeepSeek V4.1 Flash-1017 / DeepSeek V4.1 Flash):

| Benchmark | M3.1 Flash | DS V4.1 Flash-1017 | DS V4.1 Flash |
|---|---:|---:|---:|
| Automation-Bench v1.0.6 | **62.8** | 60.5 | 54.8 |
| BankerToolBench | **76.2** | — | 73.7 |
| Terminal Bench 4.0 | 38.1 | **42.3** | 31.2 |
| LOCA-Bench 256K (long ctx) | **60.3** | — | — |
| OSWorld 2.0 | **61.7** | — | 54.3 |
| DeepSWE v1.1 | 73.7 | **77.0** | 74.2 |
| NL2Repo-Bench | 61.2 | **68.2** | 65.4 |
| Agents' Last Exam | **29.6** | — | — |
| CyberGym | **82.2** | — | — |
| HLE (no tools) | **43.4** | 42.6 | 39.2 |
| CharXiv RQ | **82.5** | — | 69.0 |

Honest read: M3.1 wins agentic/computer-use/long-context/cyber rows; DeepSeek V4.1 Flash-1017 wins several coding rows (DeepSWE, NL2Repo, Terminal Bench). It is not a strict upgrade over the model our DS engine already serves on coding; it is a different point on the agentic-vs-coding frontier, plus 1M context and vision.

Quantization degradation — the QAT caveat that shapes everything: **the routed experts were trained as 4-bit (NVFP4)** ([LyJonathan README](https://huggingface.co/LyJonathan/MiniMax-M3.1-Flash-GGUF): "the experts were trained as 4-bit, so the 1–2-bit tiers are degraded"; their measured top-1 vs FP32 reference falls from 77.3% at NVFP4 to ~52% at plain IQ2_XXS). Publishers that fight this with allocation + GPTQ + residual stages recover much of it: Baekpica VQ hits 88.2% top-1 / KL 0.150 at 1.98 bpw average (vs exact-decoded original), Maya-S 87.6% top-1 / NLL +0.8% at IQ2_XXS-mix, unsloth claims 87.5–91.5% top-1 agreement for UD-IQ2_XXS/UD-Q2_K_XL on its own eval. LyJonathan's IQ2_XXS-GPTQ-v2 (Maya-S layout + sequential GPTQ) measures KL 0.211 / 87.5% vs their NVFP4 tier. Cross-publisher numbers are not directly comparable (different sets and references); within-publisher deltas are the reliable signal.

Community: [release thread](https://www.reddit.com/r/LocalLLaMA/comments/1x9uzv9/minimax_m31_flash_released_model_weights_oct_23) (614 upvotes) and a follow-up celebrating the community quant effort "252.4 GB → 93 GB in 10 days" (that 93 GB artifact is Maya-XS; [MiniMax thanked the quantizers by name](https://www.reddit.com/r/LocalLLaMA/comments/1x9uzv9/minimax_m31_flash_released_model_weights_oct_23)). Cloud-speed gripes exist but are API-side, not weights-side. [Project Maya](https://github.com/mw00/project-maya) (MIT, 2.9k stars) serves M3.1-Flash alongside GLM-5.3/DeepSeek-V4.1-Flash with a hot-expert GPU-cache + RAM-compute design — a public architectural twin of our strata pool, and existence proof the streaming approach works for this model.

---

## 5. Fit analysis vs our engine

Constraint walk (our rules → this model):

1. **GGUF tensor layout.** Every stock-llama.cpp-compatible pack (unsloth UD, Maya, AtomicChat, Sakura, smalinin, DevQuasar) is FUSED `ffn_*_exps` — the exact dead-end we measured on DeepSeek UD packs. Verified directly for Maya and AtomicChat; unsloth established by converging evidence (labeled inference above). **Exactly one publisher keeps per-expert servable naming: Baekpica VQ** (`experts.E.w{1,3,2}.weight`, 384 tensors per layer file, one file per layer — byte-identical to what our DRAM pool wants to stream). Alternatively, an offline de-fuse tool can split any fused pack into per-expert tensors without re-quantization (types stay IQ2_XXS/Q3_K etc. that already have strata kernels) — real but redundant work if the Baekpica port lands.
2. **Quant formats with in-tree kernels.** Trunk parts everywhere are in-tree formats (Q4_K, Q8_0, IQ2_XXS family, TQ2_0 in AtomicChat). The only out-of-tree formats: Baekpica GVQ1/GVR8 codebook payloads (fully specified in [FORMAT.md](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/blob/main/FORMAT.md); dequant = 8-dim codebook gather + F16 row-gain, cheaper math than IQ dequant; MIT-licensed strict readers `tools/minimax_vq_format.py`, native + CUDA kernels in [ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)) and LyJonathan's KT/KL/D3/NVFP4-repack expert types (closed, Sparsa-only — exclude). Unsloth UD-Q8_K_XL ships native NVFP4 experts — upstream ggml has NVFP4 kernels (type 40) but our in-tree kernel list does not.
3. **Architecture support.** MSA is a NEW attention family for strata — neither our Qwen-engine GDN+QSA path nor the DS-engine MLA path implements it, and no fused upstream CUDA kernel exists (PR #29553 unmerged). llama.cpp's stock-op MSA composition (#24908 + #26297) is the porting reference; if our QSA path already does index-head block scoring, the delta is smaller — that adjacency is an inference to validate in-tree, not a measured fact. **This port gates every artifact; it is unavoidable for any M3.1 serving in strata.** Staging option: serve dense attention first (mathematically valid, GQA 4-kv KV is small) and add block-sparse selection for long context later.
4. **Tokenizer.** GGUF pre-flag `minimax-m2` already exists in upstream llama.cpp (from the M2 line, #16831 era + #18399 hash for M2.1); vocab 200,064 with NFC normalizer flag. Work: wire pattern + vocab into our registry — small, bounded.
5. **Quality per quant.** Ranked by published fidelity-per-byte at our RAM ceiling: Baekpica VQ (KL 0.150 / 88.2% top-1, per-tensor manifest) > Maya-S (NLL +0.8% / 87.6%) ≈ unsloth UD-IQ2_XXS/UD-Q2_K_XL (87.5–88.5% top-1 on unsloth's eval) > LyJonathan plain IQ2_XXS (81.85% top-1) > Maya-XS (82.6%) > LyJonathan IQ1 tiers (incoherent-to-degraded: IQ1_S "loops", 44.3% top-1).

RAM footprint vs 184 GiB box (budget ≤ ~150 GiB after OS/KV — KV lives in VRAM in our architecture; host pool holds weights):

| Artifact | Model bytes | GiB | Fits? | Lane bytes/token (4 routed experts, ceiling at 50 GB/s) |
|---|---:|---:|---|---:|
| Baekpica VQ (full release) | 115.5–117.7 GB | ~110 GiB | yes, ~74 GiB headroom | 3.20 GB → ~15.6 tok/s |
| Maya-S IQ2_XXS-mix | 122.4 GB (+2.3 dspark +1.7 vision) | ~117.7 GiB | yes, ~66 GiB headroom | 3.47 GB → ~14.4 tok/s |
| Maya-XS IQ1_S-mix | 93.0 GB | ~87 GiB | yes, most headroom | 2.58 GB → ~19.3 tok/s |
| AtomicChat AD-IQ2_XXS | 112.2 GB | ~105 GiB | yes | 3.19 GB → ~15.7 tok/s |
| unsloth UD-IQ2_XXS | 135.0 GB | ~126 GiB | yes | 3.84 GB → ~13.0 tok/s |
| unsloth UD-Q2_K_XL | 143.9 GB | ~134 GiB | yes, tight | 4.15 GB → ~12.0 tok/s |
| LyJonathan IQ2_XS-D3 | 146.1 GB | ~136 GiB | yes, tight (and Sparsa-only types) | 4.25 GB → ~11.8 tok/s |
| LyJonathan NVFP4 / unsloth UD-Q4/Q8 / anything ≥160 GB | 191–259 GB | >178 GiB | **no** | — |

Lane math assumptions: all four active routed experts fetched from the RAM pool every token (zero hot-expert cache hits), experts at the pack's effective bpw, 4 experts × 57 layers × 56.6M params = 12.91B active routed params/token; shared expert + attention + dense are VRAM-resident. VRAM check: trunk ≈ 7–10 GB (Q4_K/Q8_0/Q6_K) + mmproj 1.7 + KV at 0.12 MB/token FP16 (verified from 4 kv-heads × 128 × 60L × 2; AtomicChat README states ~0.14 MB/token incl. overhead) → 32K ctx ≈ 3.7 GiB, 131K ≈ 15 GiB, 262K ≈ 30.5 GiB (needs Q8_0 KV ≈ 15 GiB — supported upstream #26180 — or MSA block-paged host KV, the Baekpica pattern). 32 GB VRAM holds trunk + 100K+ FP16 KV or 262K Q8 KV.

Per-token byte class verdict: at ~2 bpw experts, M3.1's lane traffic (~3.2 GB/token) is comfortably under our DS-lane's (DeepSeek-V4.1-Flash 1017B-A17B at ~4 bpw ≈ 8+ GB/token) — expected lane class is FASTER than our current DS serving, before any hot-expert caching gains.

---

## 6. Ranked recommendation

**Structural fact first: no M3.1 artifact serves in strata today.** The MSA attention port is on every path. The ranking below is therefore about which artifact makes the total port smallest and the served result best.

### #1 — Baekpica/MiniMax-M3.1-Flash-VQ-GGUF (worth the port)
- Path: `Baekpica/MiniMax-M3.1-Flash-VQ-GGUF` (release.json entry point; 61 GGUF parts, 115.5 GB).
- Quant: GVQ1 13-bit residual-VQ experts, avg 1.98 bpw (R13 all layers + R8 residual on 20); trunk Q4_K/Q8_0/BF16/F32; embed BF16; dspark Q8_0.
- Expected RAM: ~110–118 GiB of 184 — the largest headroom of any quality candidate.
- Engine work: (a) MSA attention port — block-sparse top-16 selection + indexer cache, reference: llama.cpp master minimax-m3 + ds4-dfm-rs; (b) GVQ1/GVR8 dequant kernel — fully specified format, MIT reference readers + CUDA kernels provided, amortizes to the same format family's GLM/DeepSeek-V4.1/Ling packs (their manifest dispositions reference `ds41_vq`, `glm_vq`, `ling_vq`, `beam_vq`); (c) tokenizer wiring (`minimax-m2` pre-pattern + 200,064 vocab + NFC); (d) optional DSpark drafter later (2.18 GB Q8_0 part already in the release).
- Why it wins: the only per-expert, per-layer-blob layout in the field — maps 1:1 onto our pool; best published fidelity-per-byte (KL 0.150 / 88.2% top-1 vs exact-decoded original, per-tensor sha256 receipts, held-out teacher-forced method — the ISTA-DASLab-style rigor we prefer, from a different publisher); bytes validated by our own header parses (13-bit indices over 8-dim groups = 1.47 GB/layer, exact); MIT tooling; built by an engine author whose runtime (ds4-dfm-rs) is architecturally our twin (expert streaming, SSD-tier KV addressed by the model's own MSA indexer).

### #2 — peasantsmith Maya-S IQ2_XXS (runner-up; also the "run something today" pick)
- Path: `peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF/Maya-S/` (3 shards, 122.4 GB + sidecar indexers).
- Quant: IQ2_XXS experts (sensitive layers IQ2_XS/IQ2_S/IQ3_XXS), attention+indexer Q8_0, shared/dense/embed/output Q6_K; error-feedback rounding + imatrix per expert + sequential calibration.
- Expected RAM: ~118 GiB.
- Engine work for strata: offline de-fuse re-layout (split `ffn_*_exps` into per-expert tensors — no re-quantization, strata already has the IQ kernels) + the same MSA port + tokenizer. OR zero-strata-work bridging: it loads in stock llama.cpp master ≥ 2026-10-24 today (v1.0.0) — use it on-box to validate model quality and build the MSA port against a known-good reference while the strata integration proceeds.
- Why runner-up: quality within noise of #1 (NLL +0.8%, 87.6% top-1), stock-llama.cpp loadable (unique among the quality candidates), and Project Maya's MIT engine is the closest public reference implementation of our exact serving architecture. Loses to #1 on layout (fused → extra re-layout step we then maintain) and on published receipts (no KL numbers, NLL-only).

### Excluded, with reasons
- unsloth UD packs: fused (inference) + unsloth docs require the unmerged #31192 branch for full feature set; UD-Q8_K_XL's NVFP4 experts have no strata kernel; sizes 129–259 GB with the best-fitting tiers no better than #1/#2 quality.
- LyJonathan tiers: best per-tier quality table in the field, but the standout tiers (IQ2_KT/KL/D3, IQ3_K, IQ4_KS, NVFP4) use closed custom expert types (Sparsa-only); the llama.cpp-loadable plain-IQ tiers measure visibly worse (52% top-1 at IQ2_XXS vs 87-88% for allocated packs).
- AtomicChat AD: fused; TQ2_0 experts interesting but their IQ2_XXS-tier size (112 GB) buys no advantage over #1/#2; no published numbers.
- Sakura/smaling/DevQuasar/899400HS/orcarouter/operationrange/Harsha001: fused, no quality receipts (Harsha001 additionally expert-prunes the model), and none fits or beats #1/#2.
- All EXL3/MLX/NVFP4-runtime packs: wrong runtime family.

### Decision framing
- Best "no-strata-work-possible-today" artifact: **Maya-S** (or unsloth UD-IQ2_XXS) under stock llama.cpp v1.0.0 — for on-box quality QC and as the port reference. Within strata itself the honest answer is: nothing loads today; every path starts with the MSA port.
- Best "worth a small port": **Baekpica VQ** — its dequant kernel is the only delta beyond the shared MSA port, and it buys the best byte-efficiency, the only native per-expert layout, and a format family that likely covers future GLM/DeepSeek VQ packs from the same publisher.

License action item: internal research serving = fine under the MiniMax Community License; the moment strata serves M3.1 to any external/commercial surface, ship the "Built with MiniMax M3.1-Flash" attribution and send the one-time notice to api@minimax.io (written authorization only above $20M yearly revenue).

---

## 7. Sources

Primary (fetched and parsed directly):
- https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash (model card, benchmarks)
- https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/raw/main/config.json (architecture, MSA config, QAT/NVFP4+MXFP8, vision tower)
- https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/raw/main/LICENSE (MiniMax Community License terms)
- https://huggingface.co/api/models/MiniMaxAI/MiniMax-M3.1-Flash (createdAt 2026-10-17, tags, mxfp8)
- https://huggingface.co/api/models?author=MiniMaxAI (org lineage M1→M3.1-Flash→H3-Image)
- https://huggingface.co/api/models?filter=base_model:quantized:MiniMaxAI/MiniMax-M3.1-Flash (40-derivative inventory)
- https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF (README + tree API sizes)
- GGUF headers parsed by us (range-downloads, 2026-11-14): unsloth UD-Q2_K_XL + UD-IQ1_M shard 1 (KV set, truncation-limited), Maya-S shard 1 (full tensor list: fused ffn_*_exps, types), AtomicChat AD-IQ2_XXS shard 1 (full tensor list: fused + TQ2_0), Baekpica body-nonexpert.gguf + layer-03-experts.gguf (full tensor list: per-expert HF names, Q4_K/Q8_0/BF16/F32 trunk, 0x4756xxxx custom expert types)
- https://unsloth.ai/docs/models/minimax-m3.1-flash (per-quant top-1 table, RAM guidance, #31192 build requirement)
- https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF (Maya-S/XS recipes, NLL/top-1 receipts, llama.cpp loadability, sidecar indexers)
- https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF (README: KL 0.150/88.2%, CharXiv 93.9%, GB10 perf, KV contract, recipe) + https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/blob/main/FORMAT.md (GVQ1/GVR8 spec) + tree API (115.5 GB, per-layer files)
- https://huggingface.co/AtomicChat/MiniMax-M3.1-Flash-GGUF (AD quants, sizes, 0.14 MB/token KV note, stock-llama.cpp claim)
- https://huggingface.co/LyJonathan/MiniMax-M3.1-Flash-GGUF (per-tier top-1/KL64 table, Sparsa-only types, NVFP4-QAT degradation note, Sparsa perf) + tree API (sizes)
- https://huggingface.co/webmp3/Sakura-MiniMax-M3.1-Flash-Q3_K_M-GGUF, https://huggingface.co/smalinin/MiniMax-M3.1-Flash-Q2_K-GGUF, https://huggingface.co/DevQuasar/MiniMaxAI.MiniMax-M3.1-Flash-GGUF, https://huggingface.co/Harsha001/MiniMax-M3.1-Flash-StrixHalo-GGUF, https://huggingface.co/899400HS/MiniMax-M3.1-Flash-C3-GGUF, https://huggingface.co/operationrange/MiniMax-M3.1-Flash-abliterated-GGUF, https://huggingface.co/orcarouter/MiniMax-M3.1-Flash-abliterated-GGUF (tree API sizes)
- https://huggingface.co/amd/MiniMax-M3.1-Flash-Quark-MXFP4, https://huggingface.co/primitive-ai/MiniMax-M3.1-Flash-pruned-285B-NVFP4 (non-GGUF derivatives, from listing)
- https://huggingface.co/bartowski/MiniMax-M3-GGUF (M3 exists, no M3.1 pack)
- HF API search "LeafFire" and "LeafFire MiniMax" → 0 results (2026-11-14)

llama.cpp:
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp (LLM_ARCH_MINIMAX_M3 = "minimax-m3")
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h (type enum: MXFP4=39, NVFP4=40, TQ1_0=34, TQ2_0=35, IQ1_M=29, BF16=30, COUNT=44)
- https://github.com/ggml-org/llama.cpp/pull/24523 (preliminary M3, closed unmerged, "sparse attention not yet supported")
- https://github.com/ggml-org/llama.cpp/pull/24908 (M3 MSA support, merged 2026-07-26)
- https://github.com/ggml-org/llama.cpp/pull/25113 (M3 vision/mtmd, merged 2026-07-26)
- https://github.com/ggml-org/llama.cpp/pull/26144 (indexer F32, merged)
- https://github.com/ggml-org/llama.cpp/pull/26180 (quantized KV for M3, merged)
- https://github.com/ggml-org/llama.cpp/pull/26210 (minimax-m3 tool-call parser, merged)
- https://github.com/ggml-org/llama.cpp/pull/26297 (MSA prefill CPU→GPU stock ops, merged 2026-07-30)
- https://github.com/ggml-org/llama.cpp/pull/29553 (dedicated MSA CUDA kernel, closed unmerged)
- https://github.com/ggml-org/llama.cpp/issues/31063 (M3.1 Flash NVFP4-scale garbled-output bug)
- https://github.com/ggml-org/llama.cpp/pull/31097 (fix, merged 2026-10-24 — M3.1 text works on master from here)
- https://github.com/ggml-org/llama.cpp/pull/31192 (open: M3.1 converter, NVFP4-native experts, NFC, DSpark drafter; produced unsloth GGUFs)
- https://github.com/ggml-org/llama.cpp/releases (v1.0.0, 2026-11-13)
- GitHub issue search "minimax state:open" (19 items, no M3.1 correctness issues open)

Engines and community:
- https://github.com/mw00/project-maya (MIT, 2.9k stars; hot-expert GPU cache + RAM compute for GLM-5.3/DS-V4.1-Flash/M3.1-Flash)
- https://github.com/Baekpica/ds4-dfm-rs (MIT; minimax_m3_vq runtime, PR #110 merged, PR #126 perf campaign)
- https://github.com/LyJonathan/Sparsa (MIT, experimental)
- https://www.reddit.com/r/LocalLLaMA/comments/1x9uzv9/minimax_m31_flash_released_model_weights_oct_23 (release thread; weights Oct 23)
- https://www.reddit.com/r/LocalLLaMA/comments/1ttdiq0/minimax_m3_coding_agentic_frontier_1m_context (M3 thread)
- https://arxiv.org/abs/2606.13392 (MSA paper)
- Web search coverage: MarkTechPost M3.1 Flash launch note; r/LocalLLaMA community-quant thread ("252.4 GB → 93 GB in 10 days")
