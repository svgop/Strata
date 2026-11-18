---
type: research-study
date: 2026-11-18
subject: NVFP4 routed-expert support in upstream Strata (origin/main), pack availability, and go/no-go for RTX 5090 32 GB
repo: /home/svg/LLM/strata (read-only study; origin/main c77572806, local HEAD 9f7d977d2, 1261 commits behind)
status: complete
---

# Upstream NVFP4 study (origin/main @ c77572806, 2026-11-16)

## 0. Verdict first

| Question | Answer |
| --- | --- |
| Does upstream support NVFP4 Qwen3.8-Flash-Next packs? | Yes — first-class feature since 2026-09-30 (e51b4379d). Routed experts only. |
| Where does the rumored pack come from? | `Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata` (HF) — sergqwer's `strata-nvfp4` fork pack, upstreamed. ~128 GB. The rumor's "sergqwer ~128.8 GB" pack is this one. |
| Is NVFP4 resident-on-GPU or streamable? | Streamable MoE format, same expert-cache machinery as IQ native packs: VRAM cache slots + host (RAM) expert pool + PCIe fetch of misses (share 0.25 for NVFP4 vs 0.55 x16). Experts never fully reside on GPU except when VRAM is huge (72 GB card test). |
| Go/no-go on our 5090 32 GB | GO for a quality A/B. NO-GO as a speed upgrade. Condition: host RAM must hold the 63.29 GiB `experts.bin`; otherwise setup's RAM-budget mode costs ~30-40% decode and ~6x prompt latency (fork measurement). |

## 1. Provenance and commit lineage

| Commit | Date | Claim |
| --- | --- | --- |
| `e51b4379d` | 2026-09-30 | NVFP4 routed experts: converter, pack, decode, prompt path (W4A8/W4A4/FP16), AVX-512 CPU rows, docs/NVFP4.md. 24 files, +2139. Cherry-picked from `1f8a12858` of **sergqwer/strata-nvfp4** (MIT) — the rumored third-party fork IS the upstream source. |
| `9063d81bd` | — | UE4M3 scale decode through a table (ldexp version was 5x slower than ggml's). |
| `af8b47d45` | — | Int3 fix: `mmq_nvfp4_parity` F16C intrinsics x86-only (aarch64 CUDA builds). |
| `3cd33d504` | — | CMake: CUDA build with no arch option defaults to sm_120; `native` builds the W4A4 unit for sm_120a. |
| `525c0be13`, `39c387727` | — | Vendored gguf-py + llama.cpp HF converter (`third_party/strata-gguf-py`); NVFP4-local tools take the vendored converter. |
| `b7bcbcde9`, `82d3a0345` | — | NVFP4-QAD sized by its 76.8 GB pack, not the 102.2 GB checkpoint. |
| `cf9d86fba` | 2026-11-13 | Batch 19: fused prompt experts — NVFP4 gate/up and down decoded into the fused int8 tile (`moe_fused_iq.cu`), taken when `STRATA_PF_FUSED=1` and `STRATA_PREFILL_NVFP4` unset/`w4a8`. |
| `82aeb32b2`, `b933674ed`, `52d2d3ee2` | — | NVFP4 layers take the staged fused group entry below the stream floor; startup line counts NVFP4 layers. |
| `2f4e1dcae` | — | Expert-paths bench (`tests/cuda/prefill_expert_paths_bench.cpp`) gains the NVFP4 / NVFP4 row. |
| `a171bba3f` | 2026-11-14 | NVFP4 fused-kernel instances compiled for every CUDA arch including sm_75 (spill-baseline list). |
| `71395ec9c`, `4ebfcda43`, `9a779fb40` | — | Setup pins for NVFP4 families; Turing fused i-quants merged onto NVFP4 base; RAM/arena compared in one unit. |

## 2. Format and tensor types

- GGML type: `GGML_TYPE_NVFP4 = 40` in the vendored ggml (`third_party/strata-llama/ggml/include/ggml.h` L429-431; `GGML_FTYPE_MOSTLY_NVFP4 = 26`). MXFP4 = 39 sits beside it.
- Block (`ggml-common.h` L219-227): `block_nvfp4` = 4x UE4M3 sub-block scales (`d[4]`, one per 16 values) + 32 B packed E2M1 codes (`qs[32]`) = **36 B per 64 values**. Matches NVIDIA ModelOpt storage (group 16).
- Engine role lists all include type 40: `STRATA_GU_FMTS` (gate/up), `STRATA_D_FMTS` (down), `STRATA_MMVQ_FMTS` (`src/kernels/cuda/iq_kernels.cu` L851-861).
- Expert blobs in `experts.bin`: `[gate | up | down]` + 16-byte FP32 tail `{s_gate, s_up, s_down, 0}` (the per-expert `weight_scale_2`). Blob = **2.76 MB** (67,948,118,016 B / 24,576 experts) vs IQ2_XS 1.51 MB. The GGUF has no room for the tails, so **NVFP4 experts are read from `experts.bin` only** (`tools/iq_pack.py` `expert_layout()`, e51b4379d). Pack folder carries `native_experts.txt` (native pack).
- Coverage is **weights-only NVFP4 for routed experts**. Everything else stays in existing formats (tools/nvfp4_convert.py policy, docs/NVFP4.md):
  - Q8_0: attention, `ssm_out`, shared experts, output head, token embedding.
  - BF16: routers, SSM gates, indexer, PLE and hyper-connection projections.
  - PLE n-gram table: dropped from the GGUF; pack ships an FP8 table shard (`ple-fp8.gguf`, 51.2 GB) read via `--ple-gguf`; a GSQ-RCO IQ4_NL PLE shard substitutes with no quality loss (92.15% vs 92.09% top-1).
- Activations: decode multiplies NVFP4 weights against **q8_1** activations — UE4M3/E2M1 decoded in software (`ue4m3_to_f32_half`, NaN->0; `vec_dot_nvfp4_q8_1`, iq_kernels.cu L587-616), bit-identical to ggml. W4A4 (FP4 activations on Blackwell MMA) is opt-in only; `w4a4x2` exists only in bench/docs history, not in the release mode switch.

## 3. Kernel choice (decode / prompt / CPU)

| Path | Mechanism | File |
| --- | --- | --- |
| Decode experts | Own q8_1 kernels, software UE4M3/E2M1 decode; per-expert tail scales applied on FP32 outputs (s_gate/s_up before SwiGLU, s_down on down output; folding s_down into up rejected — FP16 subnormals) | `src/kernels/cuda/iq_kernels.cu` (L1955, 2038, 2222); `nvfp4_expert_gpu_parity` |
| Prefill, default `w4a8` | llama.cpp MMQ int8 tensor cores, Blackwell FP4 MMA hidden from device code and tile config — identical on every arch | `src/prefill/mmq_nvfp4_w4a8.cu` |
| Prefill, opt-in `w4a4` | FP4 x FP4 MMA; the one unit CMake builds 12x -> 12xa (`strata_mmq_w4a4`, `cmake/backends/cuda.cmake` L149-170); runtime check: sm_12x card in a 12x build or falls back to w4a8 with a note (moe_mmq.cu L291-298) | `src/prefill/mmq_nvfp4_w4a4.cu` |
| Prefill `fp16` | dequantize + cuBLAS FP16 GEMM (reference) | prefill.cpp |
| Prefill, fused (no-MMQ builds) | NVFP4 decoded into the fused int8 tile when `STRATA_PF_FUSED=1` and mode unset/w4a8 | `src/prefill/moe_fused_iq.cu` (cf9d86fba) |
| CPU pool rows | AVX-512 `vpdpbusd` lanes (1.8-3.7x ggml-cpu); AVX2 lanes bit-equal to ggml-cpu for non-AVX-512 CPUs; `STRATA_NO_NVFP4_512=1` / `STRATA_NO_NVFP4_256=1` fall back to ggml-cpu | `src/kernels/cpu/nvfp4_avx512.cpp`, `nvfp4_avx2.cpp` |

Enabling switch: `STRATA_PREFILL_NVFP4 = w4a8` (default) `| w4a4 | fp16` (`src/prefill/moe_mmq.cu` L199-205). HIP: NVFP4 MMQ and W4A4 are CUDA-only (`GGML_USE_HIP` guards); setup never offers NVFP4 packs on AMD/Intel.

## 4. Hardware requirements

- Decode: no CUDA-only intrinsics; measured on Tesla P100, RTX 3060, RTX 5070, RTX 5090. Default prefill (w4a8, int8 MMQ) needs sm_75+. (docs/NVFP4.md "Other GPUs")
- W4A4: Blackwell FP4 tensor cores only — sm_120a/12xa image + sm_12x card. sm_121 (DGX Spark) turns it on by default (`src/core/nv_defaults.cpp` L80; 1.106x prompt at 4K, 1.229x at 16K vs w4a8). sm_120 stays opt-in.
- Engine built for sm_75/86/89 as PTX ran on an RTX 5090 with parity results identical to sm_120 (docs/NVFP4.md).
- RTX 50 cards have FP4 cores; other NVIDIA cards run NVFP4 at 8-bit precision (docs/MODELS.md).

## 5. Residency model (the VRAM/RAM budget question)

- NVFP4 is a **streamable MoE format**: VRAM expert cache + host pool + PCIe fetch of misses. `pcie_start_frac()`: NVFP4 packs 0.25, other native packs 0.55 (`include/strata/core/defaults_026.hpp` L33-37). Fork data: PCIe share 0.25 -> 117-124 tok/s vs 0.55 -> 103-105 tok/s at 262K ctx.
- Setup runs NVFP4 packs in RAM-budget mode (`--resident-budget-gib`, RAM minus 24 GB) when RAM is below the ~68 GB of experts plus room (docs/NVFP4.md quick start). MODELS.md: "All three need about 80 GB of RAM to keep every expert in RAM".
- The 16-GiB-beside rule is sergqwer's fork law (`setup.py` L3623): on 128 GB + RTX 5090, all-experts-in-RAM 169 tok/s and 4.9 s 32K prompt, budget mode 118 tok/s and 30 s. **Host RAM is the throttle, not the GPU.**

## 6. Pack tooling (tools/)

| Tool | Role |
| --- | --- |
| `tools/nvfp4_convert.py` | ModelOpt checkpoint -> GGUF via vendored llama.cpp converter (`third_party/strata-gguf-py`), Strata type policy (BF16 projections, Q8_0 native tensors, no PLE, MTP dropped). |
| `tools/iq_pack.py` | `expert_layout()` writes NVFP4 blobs + 16-B scale tail into `experts.bin`. |
| `tools/nvfp4_verify.py`, `tools/nvfp4_verify_gguf.py` | Bit-for-bit conversion proof against the checkpoint. |
| `tools/nvfp4_local_pack.py` (+ `test_setup_nvfp4.py`, `requirements-nvfp4-local.txt`) | NVIDIA FP4-Local (QAD) prepare on the user's PC: CSF scale decode (`nvfp4_csf_fixed`/`_exceptions` -> plain E4M3), MXFP8 -> Q8_0 (+0.6% RMS), PLE as NVFP4x16 90-B rows (`ngram.cpp` format `nvfp4x16`), MTP -> Q2_0 draft (88.7% acceptance). ~12 min, ~110 GB scratch. |
| Parity/bench | `nvfp4_expert_gpu_parity`, `nvfp4_avx512_parity`, `mmq_nvfp4_parity`, `nvfp4_decode_bench`, `mmq_expert_bench`, `tests/quant/nvfp4_decode_model_test.cpp`, NVFP4 row in `prefill_expert_paths_bench` (2f4e1dcae). |
| `setup.py` | `NVFP4_REPOS`: every file pinned by revision, size, SHA-256 (checked against Hub LFS pointers 2026-10-12). |

## 7. Pack availability (exact HF repos)

| Setup size | `--model` | HF repo @ revision | Size |
| --- | --- | --- | --- |
| NVFP4 (Strata's version) | `NVFP4` | `Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata` @ `640a1db9` | ~128 GB total: `pack/experts.bin` 67,948,118,016 B (63.29 GiB), `pack/dense.bin` 1.54 GB, `qwen-nvfp4-gptq-dense.gguf` 5.99 GB, `ple-fp8.gguf` 51.2 GB, `token-embd-bf16.gguf` 1.27 GB |
| NVFP4-NVIDIA | `NVFP4-NVIDIA` | `Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata` @ `cca8f7fe` (conversion of `nvidia/Qwen3.8-Flash-Next-NVFP4`) | same sizes |
| NVFP4-QAD | `NVFP4-QAD` | `nvidia/Qwen3.8-Flash-Next-FP4-Local` @ `b9ce18a6` — 102 GB checkpoint prepared on-PC into a 76.8 GB pack | 102 GB (+110 GB scratch) |
| (conversion test source) | — | `jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4` | — |
| (upstream port source) | — | `sergqwer/strata-nvfp4` (MIT fork; engine path ported, packs mirrored by Maximilian228) | — |

The rumored "sergqwer Qwen3.8-FN-NVFP4 ~128.8 GB" = the NVFP4 (Strata's version) pack above (63.29 GiB + 1.5 + 6.0 + 51.2 + 1.3 GB ≈ 128 GB).

Launch: `strata launch --family nvfp4 --model NVFP4` (docs/NVFP4.md, docs/MODELS.md#nvfp4).

## 8. Performance and quality claims (with sources)

| Claim | Numbers | Source |
| --- | --- | --- |
| Fork author, RTX 5090 32 GB + 9950X3D + 128 GB DDR5-5600, 64K ctx, int8 KV, all experts in RAM (63.3 GiB, ~8,350 GPU cache slots) | NVFP4 decode 114-116 tok/s, 32K prompt 3,900 tok/s; IQ2_XS same engine 132-139 tok/s, 5,600 tok/s | docs/NVFP4.md "History / older measurements" (fork engine) |
| Quality, teacher-forced vs Unsloth Q8_0 (Strata 0.1.44, `quant_quality.py`, 4,995 positions) | NVFP4 GPTQ experts-on-CPU: top-1 92.09%, KL 0.070. **GSQ-RCO IQ3_S: top-1 87.69%, KL 0.144.** UD-Q5_K_XL 94.55%/0.038 | docs/NVFP4.md |
| Prefill precision/speed (4-8K prompts) | w4a8 2,503 tok/s (KL 0.0018), w4a4 2,904 (KL 0.0080), fp16 1,905 — w4a8 is 4.5x closer to fp16 at 86% of w4a4 speed | docs/NVFP4.md |
| RTX 20-40 prefill penalty | NVFP4 prompt products 1.35-1.6x slower than IQ3_S / IQ4_NL at 32K (FP32 scale per int8 step; only Blackwell FP4 MMA avoids it) | docs/NVFP4.md |
| NVFP4-QAD (FP4-Local) on RTX PRO 5000 72 GB, all experts in VRAM | decode 159 tok/s (137-216), prefill 3,981/4,813/4,916 tok/s at 4K/16K/32K; pplx 3.95 vs Q8_0 4.05 | docs/NVFP4.md |
| Decode kernel bandwidth | RTX 5070 0.294 ms / 64 experts at 1 token (89% DRAM BW) | docs/NVFP4.md (`nvfp4_decode_bench`) |
| `STRATA_ROUTE_TAIL_SKIP=7` | +7-25% decode (3060), +47-55% (5070 64 GB); quality cost: top-1 88.59% vs 91.79%. Off by default for NVFP4 | docs/NVFP4.md |
| All-RAM vs budget mode (fork rule origin) | 169 vs 118 tok/s decode; 4.9 s vs 30 s 32K prompt | setup.py L3623 (sergqwer 0.1.42-nvfp4.2) |

No commit message carries t/s claims beyond the docs tables above; the numbers live in docs/NVFP4.md and its history section.

## 9. Go/no-go: NVFP4 Qwen3.8-Flash pack on our 5090 32 GB

| Factor | Finding |
| --- | --- |
| Engine | Our working tree (HEAD `9f7d977d2`, 2026-11-05) already contains the full NVFP4 engine path (`src/kernels/cpu/nvfp4_*`, `src/prefill/moe_mmq.*`, `defaults_026.hpp` 0.25 PCIe share, docs/NVFP4.md). origin/main adds batch-19 fused-prompt NVFP4, QAD prepare, per-SM default tables. No checkout needed for a first test; update needed for the fused-prompt path. |
| GPU fit (32 GB) | Dense GGUF ~6 GB (Q8_0) + token embd 1.27 GB + KV + expert cache slots — fits like IQ3_S does today. NVFP4 is NOT resident-on-GPU: cache slots halve (2.76 MB vs ~1.9-2.3 MB IQ3_S-class blobs), PCIe miss share 0.25. |
| Host fit | `experts.bin` = 63.29 GiB vs our IQ3_S arena ~38 GB (19.4 GB expert cache + arena) -> **+~25-44 GB host RAM demand**. On a 128 GB box: all-experts-in-RAM, fork-class speed (114-116 tok/s decode on 5090). Under ~90 GB free: setup writes `--resident-budget-gib` (RAM - 24) and the fork numbers say decode drops ~30% and 32K prompts go 4.9 s -> 30 s. PLE (51.2 GB FP8 shard) stays mmap/SSD; GSQ-RCO IQ4_NL PLE shard substitutes at zero quality cost. |
| Quality | Confirmed better than our IQ3_S: top-1 92.09% vs 87.69%, KL 0.070 vs 0.144 (vs Q8_0 teacher). Below UD-Q5_K_XL (94.55%). |
| Speed | "Much more powerful" is FALSE for throughput on 32 GB: NVFP4 measured slower than IQ2_XS decode on the same 5090-class engine (114-116 vs 132-139 tok/s), and IQ3_S blobs are smaller still. Prefill on RTX 20-40 is 1.35-1.6x slower than IQ3_S/IQ4_NL; Blackwell only claws this back via opt-in w4a4 (2,904 vs 2,503 tok/s at 4-8K). |
| Risk | Conversion of the ready pack is pre-verified (SHA-256 pinned in setup.py, bit-for-bit verify tools); no conversion step on our side for the GPTQ/NVIDIA packs. |

**GO** — for a quality-tier A/B against the GSQ-RCO IQ3_S pack, with `strata launch --family nvfp4 --model NVFP4`, default `w4a8`, host RAM >= ~90 GB free for the 63.29 GiB arena. **NO-GO** as a latency upgrade and NO-GO if free host RAM cannot hold the expert arena (budget mode erases the quality win with a 30 s 32K prompt). Re-test `STRATA_ROUTE_TAIL_SKIP=7` separately — it buys +7-55% decode at a measured quality cost.
