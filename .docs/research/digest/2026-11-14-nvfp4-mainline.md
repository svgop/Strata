---
type: research
title: NVFP4-as-an-artifact-class — PR #31192 state, GGUF type-ID collision census, sm_120 kernel readiness, fit math, mainline momentum
date: 2026-11-14
engine-context: strata-custom-2 fork (RTX 5090 32 GB sm_120, 184 GiB RAM, ~50 GB/s DRAM lane, per-expert layouts streamed from host RAM, fused ffn_*_exps rejected)
method: gh API (PR/issue/review bodies + diffs), HF API + GGUF header range-parses (unsloth UD-Q8_K_XL shards 1-2 parsed byte-level), raw.githubusercontent fetches (upstream ggml.h, llama-modchip sources), local tree reads (vendored ggml.h/ggml.c/ggml-common.h/mmq*, ds4-dfm-rs tensors.rs + vq_fmt.h, FORMAT.md + memory-budget.json)
labels: "[measured] = read from primary artifact; [inference] = derived; [boundary] = named capability gap"
supersedes-complements: 2026-11-14-per-expert-ecosystem.md (ecosystem sweep), 2026-11-14-minimax-m31-fit.md (M3.1 fit study)
---

# NVFP4 mainline digest — 2026-11-14

## 1. PR #31192 — "model: add MiniMax-M3.1"

**State [measured, gh API 2026-11-14 12:35 UTC]: OPEN, not merged.** Branch `minimax-m31-upstream`, 7 changed files, last push/update **2026-10-26** — three weeks idle. One maintainer comment (CISC, 2026-10-26):

> "Please drop the dequant. … it does not belong to model-specific conversion, please follow up with a generic conversion option instead (though this is low priority)."

The requested change targets the `--outtype` dequant path (BF16/F16/F32/Q8_0 expert dequant) only. **The per-expert native NVFP4 writer is not contested** and is the part we care about. Merge has no explicit blocker listed; it waits on that cleanup. Related merged work in its lineage: #31427 + #31611 (`conversion/base.py` `_flush_nvfp4_experts`, tensor naming + quantized_layers resolution), #31097 (M3.1 per-expert scale forward fix, merged 2026-10-24).

### 1.1 What it produces [measured, PR body + conversion/minimax.py diff]

- **Tensor naming**: routed experts keep per-expert names via `format_tensor_name(MODEL_TENSOR.FFN_{GATE,DOWN,UP}_EXP, bid)` → `blk.L.ffn_{gate,up,down}.exp.E.weight`, written with `raw_dtype=GGMLQuantizationType.NVFP4` (type 40). Gate mapping from source checkpoint: `w1→FFN_GATE_EXP`, `w3→FFN_UP_EXP`, `w2→FFN_DOWN_EXP`; source `moe_quant_format == "nvfp4-pack-quantized"`, `group_size == 16` asserted.
- **Scale layout**: per (layer, proj) one F32 tensor `…exp.E…→ new_name.replace(".weight",".scale")` carrying `inv_global = 1.0 / weight_global_scale[E]` for all experts of that layer ("applied by build_moe_ffn"). The checkpoint's `input_global_scale` (activation scale) is **dropped** ("activation quantization is not part of the converted model"). Per-expert completeness is enforced: a layer with fewer than all 128 experts present hard-fails.
- **Type IDs used**: 40 (NVFP4) for expert data, 0 (F32) for the `.scale` vectors, standard ggml types for MXFP8-decoded body tensors (exact decode `e4m3 · 2^(e8m0 − 127)`); `--fp8-as-q8` for Q8_0 dense. No custom type IDs — this is the first mainline-GGUF path that is per-expert-indexable without squatting anything.
- Validation receipts in the PR body: float64 fixtures vs exact dequant, mean KL 5.8e-7 (BF16 src) / 5.6e-7 (NVFP4+MXFP8 src); on real weights, native NVFP4 GGUF agrees with a reference GPU implementation on 95.5–99.1% of greedy top-1 tokens.

### 1.2 Follow-ups and quality/kernel issues [measured]

| Issue | State | Content |
|---|---|---|
| [#31063](https://github.com/ggml-org/llama.cpp/issues/31063) | closed completed 2026-10-24 | `build_moe_ffn()` dropped the **per-expert NVFP4 scale** → garbled M3.1 output; fixed by #31097. The wiring hazard is real and already bit once. |
| [#31448](https://github.com/ggml-org/llama.cpp/issues/31448) | closed completed 2026-11-09 | granite dense: `build_ffn()` passed **NULL for NVFP4 weight scales** → garbage output. Same class, fixed. |
| [#32101](https://github.com/ggml-org/llama.cpp/issues/32101) | **OPEN** (2026-11-11) | qwen2vl graph ignores NVFP4 FFN scales → NULL `ffn_*_s`, NaN / PPL ~1e6. Reporter runs RTX 5070 Ti Laptop **SM120**; reproduces on CPU too — hardware-independent graph bug. |
| [#31079](https://github.com/ggml-org/llama.cpp/pull/31079) | merged 2026-11-03 | Vulkan **OOB NVFP4 UE4M3 scale lookup** fixed (`dequant_nvfp4.comp` exists in our tree). |
| [#27329](https://github.com/ggml-org/llama.cpp/issues/27329) | closed **not_planned** 2026-11-02 | RTX 5090D Blackwell + NVFP4 Qwen3.8-27B decode hang, CPU spin. 7 comments, never root-caused, closed not_planned. [boundary: unresolved sm_12x NVFP4 hang report] |
| [#31012](https://github.com/ggml-org/llama.cpp/issues/31012) (+#31236) | open | FP4 quantization rounds ties away from even — quantizer-quality nit, not correctness. |

Issue-search sweep "nvfp4" sorted by update (2026-11-14): no open OOB-read or memory-safety bug against the CUDA NVFP4 path; the open failure mode is **scale-tensor wiring dropped by a graph path** (the #32101 class), not kernel math.

### 1.3 Has anyone published a converted NVFP4 GGUF? [measured, HF API + header parse]

Yes — **unsloth/MiniMax-M3.1-Flash-GGUF** (59 likes, 28,794 downloads, updated 2026-10-26, linked from the PR body). I parsed its GGUF headers byte-level (range-downloads, `/tmp/mm31_q8xl_head.bin`, `/tmp/mm31_q8xl_s2.bin`):

- Tiers: UD-IQ1_M 129.4 GB … UD-Q4_K_XL 249.4 GB, **UD-Q8_K_XL 259.1 GB**; the fit study found UD-Q8_K_XL ships native NVFP4 experts.
- **Shard-2 header (measured)**: 72 tensors — `blk.N.ffn_gate_exps.weight` type **40 (NVFP4)** dims `(6144, 3072, 128)`, `blk.N.ffn_down_exps.weight` type 40 dims `(3072, 6144, 128)`, same for `ffn_up_exps`, each with `blk.N.ffn_*_exps.scale` type 0 (F32) dims `(128,)`. Shard 1 carries all 54 KV (incl. tokenizer, ~8.2 MB) and **zero tensors** (`split.tensors.count=1359`).
- Verdict: what is published today is the **fused/stacked class** (expert dim stacked into one tensor + one F32 scale vector per layer per proj), NOT the per-expert class PR #31192's converter writes. A true per-expert NVFP4 GGUF (≈21,888 expert tensors + 171 scale tensors for M3.1) **has no published conversion yet** [inference: 1359 ≪ 21,888; consistent with the merged stacked `_flush_nvfp4_experts` path, plus fit-study CDN-limited read of unsloth pack as fused].
- Second publisher exists in a different family: LyJonathan's NVFP4 tier (246.1 GB) is **Sparsa-only custom types** — not mainline GGUF at all (fit study, §2).
- Implication for strata: the unsloth UD-Q8_K_XL de-fuses losslessly — slice the 128-expert dim into per-expert tensors, carry each expert's F32 scale from the `.scale` vector. Zero requantization. But 259 GB total means it is a kernel-validation artifact, not a residency candidate.

## 2. GGUF type-ID hazard census — 42/43 and the neighbors

### 2.1 The registries [measured]

**Upstream mainline** ([ggml/include/ggml.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h) master, fetched 2026-11-14):

```
GGML_TYPE_MXFP4   = 39
GGML_TYPE_NVFP4   = 40   // 4 sub-blocks of 16, UE4M3 scales
GGML_TYPE_Q1_0    = 41
GGML_TYPE_Q2_0    = 42
GGML_TYPE_F8_E4M3 = 43   // NEW vs our vendored snapshot
GGML_TYPE_COUNT   = 44
```

**Our vendored tree** (`/home/svg/LLM/strata-custom-2/third_party/llama.cpp/ggml/include/ggml.h:429-433`): `MXFP4=39, NVFP4=40, Q1_0=41, Q2_0=42, COUNT=43` — a pre-F8_E4M3 snapshot. **Fork drift hazard of our own**: our tree does not know type 43 = F8_E4M3; upstream files using 43 will read as unknown here.

**ds4-dfm-rs host table** (`/home/svg/LLM/data/references/ds4-dfm-rs/crates/ds4-core/src/tensors.rs:110-124`, mirrored in the antirez-ds4 C engine `src/core/core_gguf.c:89-104`): standard 0-30 table + `PQ2_0=142` (Prism ternary, 128/34) + `DS41_TYPES`: `(40,"go1b",256,34), (41,"go2b",256,68), (42,"vqblob",1,1), (43,"fp4x32",32,17), (44,"fp8_32x32",1024,1025)`.

**Synchopate llama-modchip** ([ARCHITECTURE.md §3.13](https://github.com/synchopate/llama-modchip/blob/main/ARCHITECTURE.md), README, `vq3/vq3_enc_lib.c` fetched 2026-11-14): `Fractal-64` **repurposes `GGML_TYPE_Q2_0 = 42`** (block: `{f16 d; u8 scales[4]; u8 qs[16]; u8 qh[8]}` = 30 B/64 W = 3.75 bpw); `LUT-VQ3-64` **claims `GGML_TYPE_VQ3_0 = 43`** (block `{f16 d; u8 qs[22]}` = 24 B/64 W = 3.00 bpw). Converter writes raw type IDs (`dst_type == 42 / == 43` in `scripts/dsv4_singlepass.c:368-370`).

**Baekpica minimax-m3-vq8** (`/home/svg/LLM/data/models/VQ/MiniMax-M3.1-Flash/FORMAT.md:68-76`): expert payloads use huge u32 IDs — `0x47565138` (GVQ8 matrix), `0x47565238` (GVR8 residual), `0x47564342` (GVCB codebook) — above any ggml type, so they cannot collide; trunk uses standard 0/1/8/12/30.

### 2.2 Conflict table

| Type ID | Upstream ggml.h (master) | Our vendored fork | DS4 / vqblob class (YoungAi·Baekpica single-GGUF) | Synchopate llama-modchip |
|---|---|---|---|---|
| 40 | NVFP4 | NVFP4 | `go1b` (256 vals / 34 B) | — |
| 41 | Q1_0 | Q1_0 | `go2b` (256/68) | — |
| **42** | **Q2_0** (official legacy scalar quant) | Q2_0 | **`vqblob`** (opaque `DQVL` container, 1/1) | **Fractal-64** (30 B/64 W, in-place repurpose of Q2_0) |
| **43** | **F8_E4M3** (official, added upstream after our vendored snapshot) | *(COUNT sentinel — unknown)* | **`fp4x32`** (32 vals / 17 B) | **`VQ3_0` LUT-VQ3-64** (24 B/64 W) |
| 44 | *(unused)* | — | `fp8_32x32` (1024/1025) | — |
| 142 | — | — | `pq2_0` Prism ternary (128/34) | — |

Three-way collision on **42** (Q2_0 vs vqblob vs Fractal-64) and on **43** (F8_E4M3 vs fp4x32 vs LUT-VQ3-64). The nastiest pair is upstream-F8_E4M3 vs fp4x32: both are byte-plausible for a naive reader (1 B/elem vs 17 B/32 elem) — a size-mismatch error at best, silent misread at worst for tools that trust the ID.

### 2.3 What each class validates at payload start — reader design

| Class | Payload magic / self-identification | Source |
|---|---|---|
| Baekpica GVQ1/GVR8 (per-layer blob + minimax-m3-vq8) | 128-byte header, first 4 bytes `GVQ1` (residual: `GVR1`), version u16, then rows/cols/K/precision/codebook-ID/flags, two SHA-256 digests (codebook, body); offsets must be canonical or the strict reader rejects | `/home/svg/LLM/data/models/VQ/MiniMax-M3.1-Flash/FORMAT.md:95-115,133` |
| DS4 vqblob (type 42 in DS41 files) | `DQVL` blob magic checked before parse: `if (mg != DS4VQ_BLOB_MAGIC) return 0;`; per-matrix payloads `DQVQ` (`0x51565144`) / `DQV3` (`0x33565144`) each magic-checked; unknown versions hard-stop | `/home/svg/LLM/data/references/ds4-dfm-rs/crates/ds4-core/tests/fixtures/vq/vq_fmt.h:13-18,44,58` |
| Synchopate Fractal-64 / LUT-VQ3-64 | **No payload magic** — blocks start immediately (`block_vq3_0`, `block_q2_0` structs); the only validation is the patched runtime itself; converter keys on raw type ID alone | [llama-modchip](https://github.com/synchopate/llama-modchip) `vq3/vq3_enc_lib.c`, `scripts/dsv4_singlepass.c:368-370` |
| Mainline NVFP4 (type 40) | No magic; identity = ggml type ID + block math (`block_nvfp4` = 4×UE4M3 scale bytes + 32 B packed E2M1 = 36 B/64) | `/home/svg/LLM/strata-custom-2/third_party/llama.cpp/ggml/src/ggml-common.h:220-227` |

**Mitigation design — confirmed and refined**: a strata reader must dispatch on `(publisher family → magic bytes at payload start)` and never on type ID alone. Concretely: (1) treat GGUF type IDs 40-44 as *claims*, (2) probe the first bytes of the tensor payload for the family magic (`GVQ1`/`GVR1`, `DQVL`/`DQVQ`/`DQV3`), (3) fall back to block-math consistency checks (payload bytes vs element count) before accepting a standard ggml type, (4) hard-reject type 42/43 payloads that carry no recognized magic *and* no publisher metadata — that is the Synchopate class, and for it the only correct strata behavior is explicit refusal (their files are CPU-runtime-bound anyway). The DS4 and Baekpica families already meet this contract from their side; the Synchopate class structurally cannot, which is exactly why refusal-by-default is the right posture. Our own fork drift (vendored 43=COUNT vs upstream 43=F8_E4M3) adds rule zero: **re-sync the vendored ggml.h type enum before adding any new type**.

## 3. NVFP4 kernel readiness on sm_120 (our use: row-parallel GEMV/dequant feeding host-streamed experts)

### 3.1 What the vendored tree actually implements [measured, tree]

- `ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu` is one line: `DECL_MMQ_CASE(GGML_TYPE_NVFP4);` — instantiation only. The real machinery:
- **MMVQ (few-row GEMV) supports NVFP4**: `mmvq.cu:50` dispatches `vec_dot_nvfp4_q8_1`; `vecdotq.cuh:331-368` — DP4A integer dot with `kvalues_mxfp4` E2M1 LUT and per-16 `ue4m3_to_fp32` scale times the q8_1 ds. VDR: MMVQ=4, MMQ=8.
- **MoE path wired**: `get_mmvq_mmid_max_batch_turing_plus()`: `GGML_TYPE_NVFP4 → 8` — NVFP4 has the **highest MUL_MAT_ID MMVQ batch threshold of any type** (mmvq.cu:177-187), i.e. few-row per-expert GEMV is kept up to 8 tokens by design. A dedicated multi-token MoE kernel `mul_mat_vec_q_moe` is templated on type in the same file.
- **Fused GLU + per-expert scale epilogue**: the mmvq fusion path applies `x_scales = x_scale[channel]` / `gate_scales` **only for NVFP4** (mmvq.cu:648, 789-795, 826) — this is the merged #26048 (fuse w_s scale in epilogue) + #31296 (fuse gate/up matmuls + GLU) lineage: gate·up fusion with the per-expert F32 `.scale` consumed in-epilogue. Directly matches the PR #31192 scale layout.
- **MMQ (batch) on Blackwell**: `mmq-config-blackwell.cuh` — `CASE(GGML_TYPE_NVFP4, nthreads=256, occupancy=1, I=128, J∈{8,16,24,32,40,48,64,80,96,112,128}, SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, stream_k=true)` with both `fallback` variants. So batch-MMQ is not batch-only: dedicated tiles start at **J=8** (8 tokens), overlapping the MMVQ cutoff. Answer to "batch-MMQ only or also few-row": **both**, with a handoff at 8 tokens and dedicated Blackwell MMA tiles (SRAM layout `GGML_CUDA_MMQ_SRAM_LAYOUT_FP4 // MXFP4 and NVFP4 on Blackwell`, mmq.cuh:128).
- Activation quantization exists (`quantize_row_nvfp4_ref`, ggml.c type traits) — W4A4 serving is possible in-tree, though our engine runs weight-only.
- Also in tree: Vulkan `dequant_nvfp4.comp` (post-#31079).

**Assessment for our loader**: the vendored kernel set is already the right shape — per-expert GEMV with fused scale/GLU epilogue at ≤8-token batches, Blackwell MMA MMQ above. What is missing for strata is not kernels but our plumbing: (a) per-expert tensor residency + `.scale` F32 vector routing into the fusion args (the #31063/#32101 hazard lives exactly here), (b) `--outtype auto`-class GGUFs to test against (see §1.3 de-fuse path).

### 3.2 Alternates [measured, GitHub API]

- **vLLM** `csrc/quantization/fp4/`: `nvfp4_scaled_mm_sm120_kernels.cu` — **SM120-specific scaled-MM kernels exist** (consumer-Blackwell targeted), plus `nvfp4_blockwise_moe_kernel.cu`, `nvfp4_experts_quant.cu`, `nvfp4_quant_kernels.cu`, `nvfp4_scaled_mm_entry.cu`.
- **TRT-LLM** `cpp/tensorrt_llm/kernels/marlin/`: `marlin_nvfp4_gemm.cu` **and `marlin_nvfp4_moe_gemm.cu`** (+ template/dispatch headers). The MoE Marlin variant is the closest published match to weight-only per-expert dequant-GEMM streaming.
- Port order if we outgrow the vendored path: vendored MMVQ (already ours) → TRT-LLM MoE Marlin (weight-only, per-expert-friendly) → vLLM CUTLASS SM120 scaled-MM (W4A4 prefill class).

### 3.3 Known sm_120/consumer-Blackwell issues

- #27329: 5090D + NVFP4 decode hang, closed not_planned without root cause [boundary].
- #32101 (open): NVFP4 scale tensors ignored by one graph path → NaN; hardware-independent, also SM120 laptop.
- #31079 (merged): Vulkan OOB scale lookup — fixed; no open CUDA OOB.
- #26704: "CUDA: experimental SM120 CUTLASS MoE prefill for MXFP4 and NVFP4" — **still OPEN** (updated 2026-11-06); mainline prefill-MoE NVFP4 on sm_120 is not landed. Decode/GEMV path (our use) is landed; prefill path is the open front.

## 4. Size/fit math — verified, not headlined

### 4.1 The claim [measured arithmetic]

MiniMax-M3.1-Flash routed experts: 57 layers × 128 experts × (2×3072×6144 + 6144×3072) = 57 × 7.248 B = **413.1 B routed params** (matches fit-study's 412.8B within rounding). GGUF `block_nvfp4` = 36 B per 64 values = **4.5 bpw incl. UE4M3/16 scales** (ggml-common.h:220-227):

- Experts at NVFP4: 413.1e9 × 4.5/8 = **232.4 GB decimal = 216.5 GiB**, + F32 global-scale vectors (negligible: 57×3×128×4 B ≈ 87 KB).
- Box: 184 GiB = 197.6 GB total RAM; our pool discipline is ≤ ~150 GiB. **232 GB > 197.6 GB: does NOT fit as a full RAM pool. Confirmed.** (The "~220 GB" figure sits between the GB and GiB readings; magnitude and conclusion hold either way.)
- Cross-check: official checkpoint is 236 GB class (fit study) = 413.1B × 0.5625 B/param experts (232 GB) + MXFP8 attention/dense/shared + embed/head/F32 router ≈ consistent.
- Lane cost per token if it DID fit: 4 active experts × 57 layers × 31.84 MB ≈ 7.26 GB/token → ~6.9 tok/s ceiling at 50 GB/s. Even ignoring residency, NVFP4-for-this-model is a lane regression vs our served 1.98 bpw VQ pack (3.20 GB/token → ~15.6 tok/s; fit study §5).

### 4.2 Path (a) — tier-selective mixed residency (MoE-APEX pattern)

Viable as a design, and it is the only honest way to serve this model at 4 bpw on this box [inference from measured constants]:

- VRAM ledger: 32 GB − trunk (VQ body 6.45 GiB measured in `memory-budget.json`; native-NVFP4 serving body is Q8_0/BF16 class ≈ 7-14 GB) − KV (FP16 0.12 MB/token: 32K ≈ 3.7 GiB; MSA index keys NVFP4 = 4.22 GiB at 1M ctx, measured) − workspace ≈ **~17-20 GB for hot experts** → 31.84 MB per (layer,expert) pair → **~550-600 hot pairs ≈ 8% of the 7,296-pair space**.
- Cold tier must live in RAM in a lower-bpw format — the served VQ pack (103 GiB expert pool, exact part bytes in `memory-budget.json`: 78.33 + 16.88 GiB) is that cold tier. Dual-format artifact (NVFP4 hot + VQ cold, both from the same NVFP4 source) costs ~115 GB SSD, zero extra RAM.
- Requirements: per-expert residency manager, hit-rate telemetry, and the format-mismatch accept rule (hot = higher fidelity). All three are strata-shaped; none exist yet [boundary: no tier-selective residency code in tree].

### 4.3 Path (b) — models that DO fit: 100-160B-class per-expert NVFP4

At 4.5 bpw, ~110-150B-routed models produce 62-85 GB expert pools — the sweet spot. Verified candidates [measured, HF API 2026-11-14]:

| Checkpoint | Class | License | Fit | Per-expert? |
|---|---|---|---|---|
| [Baekpica/DeepSeek-V4-Flash-0731-120B-REAM-104E-NVFP4](https://huggingface.co/Baekpica/DeepSeek-V4-Flash-0731-120B-REAM-104E-NVFP4) | 119.8B / 43L × 104E top-6, 13.8B active; NVFP4A16 (E2M1 gs-16, FP8 scales + global), compressed-tensors `nvfp4-pack-quantized`, 8 shards ≤ 9.4 GB | **MIT** | experts ≈ 60-65 GB → fits easily | per-expert vLLM modules; gate/up fused within expert (README); exact naming unprobed [boundary] |
| [Baekpica/K-EXAONE-152B-A23B-REAP-80E-NVFP4](https://huggingface.co/Baekpica/K-EXAONE-152B-A23B-REAP-80E-NVFP4) | 152B class, 80E REAP-pruned | other (LG) | ~68 GB experts → fits | same family |
| DrRos/Qwen3.5-122B-A10B-NVFP4-GGUF (GGUF, not safetensors) | 122B MoE NVFP4 GGUF | — | ~65-75 GB total → fits | **fused class** (Qwen3.5 converter lineage #20505) — de-fuse before strata |
| jdziat/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4-GGUF | 120B-A12B NVFP4 GGUF | — | fits | fused class likely |
| ISTA-DASLab/Qwen3.5-397B-A17B-P48NVFP4-MoESQ | 155.3 GB (144.6 GiB) | — | fits pool at 2.75 bpw effective | no per-expert names; **patched-vLLM-only**, MoESQ repo 404 — excluded (ecosystem digest §3) |

Note: NVFP4 **MoE GGUF conversion has been mainline since #20505 (merged 2026-03-26, "Fix Qwen3.5/Qwen3.5 Moe NVFP4 Conversions")** — the converter chain for path (b) is proven mainline.

### 4.4 Path (c) — NVFP4 as requant SOURCE

- **For M3.1 there is no BF16 source.** The official checkpoint IS QAT NVFP4 experts + MXFP8 body; exact-dequant NVFP4 is the reference weights (PR #31192 validates to KL 5.6e-7 against exact dequant in float64). "Lossy-once-removed vs BF16" is undefined for this model — the honest statement: NVFP4 is the trained distribution.
- **Sub-4.5-bpw requant from NVFP4 works and is receipted**: our own served pack converts NVFP4 `weight_packed → GVQ1` (dispositions `consumed_vq`, `FORMAT.md:36-37`) and lands **KL 0.150 / 88.2% top-1** — inside the BF16-source family band (Beam 0.144/87.7%, MiMo 0.088/92.2%, ecosystem digest §1.4/§2). Calibration-based requantizers absorb the NVFP4 noise floor at these target bpw.
- **At-or-above-source-bpw requant gains nothing**: the QAT ceiling is the NVFP4 weights themselves (LyJonathan measured the NVFP4 tier at 77.3% top-1 vs FP32-exact-dequant reference; fit study §4). Unsloth's UD-Q8_K_XL keeping experts at NVFP4 while upconverting everything else is the ecosystem acting on this [inference, strong]. MXFP8 body → Q4_K/Q8_0 is the clean case: MXFP8 dequant is exact (FORMAT.md:153-154), so requant loss is the target format's alone.
- Published independent study of NVFP4→other-GGUF requant quality: none found [boundary]. The receipts above are publisher-measured, not lab-grade.

## 5. Mainline momentum

- **Type + kernels are mainline**: `GGML_TYPE_NVFP4 = 40` added 2026-03-11 (#19769, merged); MoE NVFP4 GGUF conversion fixed/merged 2026-03-26 (#20505); MMQ/MMVQ CUDA kernels merged through #29857/#26048/#31296; `nvfp4` is a documented type in ggml.c with to_float/from_float traits (vendored ggml.c measured). So NVFP4 weight support is ~8 months old upstream; activations (W4A4 CUTLASS scaled-MM) are the vLLM/TRT-LLM class of it.
- **Publisher traction on HF** [measured, HF search "nvfp4 gguf" 2026-11-14]: 40+ repos shipping NVFP4 GGUFs — mostly dense (Qwen3.8-27B class: cdiamond iMatrix variant at 3.3M downloads; gemma-4-31B; LFM3; Nemotron), MoE examples at DrRos Qwen3.5-122B/397B, Nemotron-3-Super-120B. **unsloth ships NVFP4-expert GGUFs** (§1.3). **bartowski/mradermacher: zero NVFP4 GGUFs found** (bartowski has M3 only, fit study §2). **ISTA-DASLab stays fused + compressed-tensors** (no move to per-expert GGUF, ecosystem digest §3).
- **Per-expert NVFP4 GGUF as a standard**: not yet — one open PR (#31192) away. The fused NVFP4-expert class is already de-facto published (unsloth); the per-expert naming is the delta. Baekpica is the only publisher shipping per-expert layouts today, and they are VQ/custom-typed, not type-40.
- Momentum read: the artifact class converging on mainline is **fused NVFP4 experts + per-layer F32 scale vector**; per-expert NVFP4 is one merge behind it.

## 6. Verdict + recommended action order

**Verdict**: NVFP4-as-an-artifact-class is real, mainline-adjacent, and kernel-ready on sm_120 for exactly our use (few-row per-expert GEMV with fused scale/GLU epilogue, MMQ above 8 tokens). But for THIS flagship model it fails residency arithmetic (232 GB experts > 197.6 GB box), so its value to strata today is: (1) a second serving tier via path (a) tier-selective mixed residency, (2) full-NVFP4 serving for the 100-160B class (path b — the REAM-104E MIT checkpoint is the cleanest entry), (3) the best available requant source for the M3.1 family (path c — already exploited by our VQ pack). The type-ID census confirms the reader rule (magic bytes, never type ID) and adds a new obligation: sync our vendored ggml.h enum to upstream (43=F8_E4M3, COUNT=44) before any type work.

**Action order**:
1. **Re-sync vendored `ggml.h` type enum with upstream master** (Q1_0=41, Q2_0=42, F8_E4M3=43, COUNT=44) and add the magic-byte dispatch rule (§2.3) to the strata reader spec — small, prevents a silent-misread class, unblocks every NVFP4/GVQ/vqblob path. Owners: `third_party/llama.cpp/ggml/include/ggml.h`, strata reader spec.
2. **Wire the NVFP4 de-fuse**: offline tool splitting fused `ffn_*_exps.weight` (type 40) + `.scale` F32 [n_experts] into per-expert tensors + per-expert scale scalars; validate against unsloth UD-Q8_K_XL shard-2 layout (§1.3) on the vendored MMVQ path. This exercises kernels without any new residency commitment.
3. **Evaluate path (b) seriously with the REAM-104E NVFP4 MIT checkpoint**: probe its safetensors index for exact expert tensor naming, size the expert pool exactly, and treat it as the first full-NVFP4 strata serving target (fits with 4x the headroom M3.1 lacks).
4. Keep PR #31192 on watch (dequant-drop is the only requested change); if merged with per-expert naming intact, path (a)'s hot-tier artifact becomes a mainline GGUF instead of a de-fuse.
5. Defer path (a) implementation until (2)/(3) land a working NVFP4 GEMV end-to-end; the tier manager is the large item and its value depends on measured hot-expert hit rates we do not yet have [boundary].

## 7. Sources

### Measured local artifacts
- `/home/svg/LLM/strata-custom-2/third_party/llama.cpp/ggml/include/ggml.h` (vendored type enum: MXFP4=39, NVFP4=40, Q1_0=41, Q2_0=42, COUNT=43)
- `/home/svg/LLM/strata-custom-2/third_party/llama.cpp/ggml/src/ggml-common.h:220-227` (`block_nvfp4`: QK 64, sub-block 16, UE4M3 scale bytes + 32 B packed E2M1 = 36 B/64)
- `/home/svg/LLM/strata-custom-2/third_party/llama.cpp/ggml/src/ggml.c` (NVFP4 type traits, quantize/dequant refs)
- `/home/svg/LLM/strata-custom-2/third_party/llama.cpp/ggml/src/ggml-cuda/mmq-instance-nvfp4.cu` (DECL_MMQ_CASE only), `mmq.cuh` (SRAM layouts incl. FP4-on-Blackwell, line 128), `mmq-config-blackwell.cuh` (NVFP4 tiles: 256 threads, I=128, J=8..128, stream-k), `mmvq.cu` (NVFP4 vec_dot dispatch :50, MMVQ-MoE max batch NVFP4=8 :177-187, fused NVFP4 scale/GLU epilogue :648/:789-795/:826, `mul_mat_vec_q_moe`), `vecdotq.cuh:331-368` (`vec_dot_nvfp4_q8_1`, VDR 4/8)
- `/home/svg/LLM/strata-custom-2/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/dequant_nvfp4.comp`
- `/home/svg/LLM/strata-custom-2/third_party/llama.cpp/conversion/base.py:780-886` (`_nvfp4_pack`, `_flush_nvfp4_experts` — stacked expert write + `.scale`/`.input_scale` tensors, merged #31427/#31611 lineage)
- `/home/svg/LLM/data/references/ds4-dfm-rs/crates/ds4-core/src/tensors.rs:73-135` (host type table: standard 0-30, PQ2_0=142, DS41 40-44 incl. 42=vqblob, 43=fp4x32)
- `/home/svg/LLM/data/references/ds4-dfm-rs/crates/ds4-core/tests/fixtures/vq/vq_fmt.h` (DQVL/DQVQ/DQV3 magics + hard-stop validation)
- `/home/svg/LLM/data/models/VQ/MiniMax-M3.1-Flash/FORMAT.md` (GVQ1/GVR1 payload magics, 0x47565xxx type IDs, MXFP8 exact dequant, dispositions `consumed_vq`)
- `/home/svg/LLM/data/models/VQ/MiniMax-M3.1-Flash/memory-budget.json` (VQ expert pool 78.33 + 16.88 GiB exact; body 6.45 GiB; MSA index keys 4.22 GiB/1M)
- `/tmp/mm31_q8xl_head.bin`, `/tmp/mm31_q8xl_s2.bin` — byte-parsed GGUF headers of unsloth UD-Q8_K_XL shards 1-2 (KV set, split.tensors.count=1359, fused NVFP4 tensor + F32 [128] scale layout)

### Web (fetched 2026-11-14)
- https://github.com/ggml-org/llama.cpp/pull/31192 (state, body, files, CISC comment)
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h (upstream enum incl. F8_E4M3=43, COUNT=44)
- https://github.com/ggml-org/llama.cpp/issues/31063, /issues/31448, /issues/32101, /issues/27329, /issues/31012, /pull/31079, /pull/26704, /pull/19769 (merged 2026-03-11), /pull/20505 (merged 2026-03-26)
- https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF (repo API: 28,794 dl, 47 files, 8 UD tiers)
- https://huggingface.co/Baekpica/DeepSeek-V4-Flash-0731-120B-REAM-104E-NVFP4 (README: 119.8B/43L/104E/top-6, NVFP4A16 compressed-tensors, 8 shards; MIT) and https://huggingface.co/Baekpica/K-EXAONE-152B-A23B-REAP-80E-NVFP4 (HF API: nvfp4, license:other)
- https://huggingface.co/api/models?search=nvfp4+gguf (40+ repos; DrRos Qwen3.5-122B/397B, cdiamond 3.3M-dl dense, jdziat Nemotron-120B)
- https://github.com/synchopate/llama-modchip (README, ARCHITECTURE.md §3.13: Fractal-64 = Q2_0=42, LUT-VQ3-64 = VQ3_0=43; `vq3/vq3_enc_lib.c` block structs; `scripts/dsv4_singlepass.c:368-370` raw-ID writes; no payload magic)
- https://github.com/vllm-project/vllm/tree/main/csrc/quantization/fp4 (incl. `nvfp4_scaled_mm_sm120_kernels.cu`, `nvfp4_blockwise_moe_kernel.cu`)
- https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/marlin (`marlin_nvfp4_gemm.cu`, `marlin_nvfp4_moe_gemm.cu`, templates/dispatch)

### Companion digests (local, same day)
- `.docs/research/digest/2026-11-14-per-expert-ecosystem.md` (publisher census, kernel PR ledger, quality receipts)
- `.docs/research/digest/2026-11-14-minimax-m31-fit.md` (M3.1 identity, GGUF inventory, lane math, LyJonathan NVFP4-tier receipts)
