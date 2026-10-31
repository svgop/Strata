---
type: digest
id: w7-prefill-scout
created: 2026-10-31
status: evidence
topic: W7 prefill — device-side MoE grouping, m-grouped GEMM, GEMM autotune, KV staging
scope: vLLM, DeepGEMM, SGLang, TensorRT-LLM, FlashInfer; main branches @ 2026-10-31
---

# W7 scout: prefill MoE grouping / grouped GEMM / autotune / staging

Summary: all five stacks group token-rows by expert with device kernels only — host allocates static upper-bound buffers, an align kernel writes order/bounds/slot tables, the GEMM grid launches on the bound and threads early-exit on a device counter. Zero host sync per layer anywhere. Autotune is offline-table (vLLM/SGLang Triton JSONs), heuristic-JIT (DeepGEMM), or warmup-context tactic profiling (TRT-LLM `_torch` AutoTuner, FlashInfer). H2D staging uses pinned double-buffer rings with per-slot CUDA events (TRT-LLM `host_staging.py`).

## 1. vLLM — fused MoE Triton + CUDA align kernel (HIGH, source read)
- Grouping: `moe_align_block_size` CUDA op (`csrc/moe/moe_align_sum_kernels.cu`: `moe_align_block_size_kernel`, `count_and_sort_expert_tokens_kernel`, plus `_moe_align_block_size_small_batch_expert` device path) emits `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded` as device tensors; both Triton kernels (incl. GPTQ/AWQ WNA16 variant = closest MMQ analog) `tl.load` all three. No `.item()/.cpu()/synchronize` in `fused_moe.py`/`layer.py` (grepped 2026-10-31).
- Sync elimination: static bound `topk_ids.numel() + num_experts*(block_size-1)`; grid on `EM = sorted_token_ids.size(0)`; per-block early-exit on `tl.load(num_tokens_post_padded_ptr)` (fused_moe.py:157-158, 716-719). Small-M: `EM = min(size, M*top_k*BLOCK_M)` when `M < BLOCK_SIZE_M`. Extreme sparsity (`tokens*top_k*4 <= num_experts`): alignment skipped entirely (`_prepare_expert_assignment`, fused_moe.py:1614-1646). EP: `expert_map` marks non-local experts -1, blocks skipped on device.
- Staging: KV host staging via NIXL connector pinned host buffers (`benchmarks/kernels/benchmark_host_buffer_kv_copy.py`, per-group coalescing).
- Autotune: offline Triton JSONs per `(E,N,device,dtype[,block_shape])` shipped in `.../fused_moe/configs/`; fallback heuristic `get_default_config` (fused_moe.py:1503-1515). Old tuner script removed from tree; SGLang keeps it (#3). Numbers: none in-repo.
- URL: https://github.com/vllm-project/vllm (docs/design/fused_moe_modular_kernel.md)

## 2. DeepGEMM — m-grouped FP8/FP4 GEMM, contiguous vs masked (HIGH)
- Mechanism: M-axis-only grouping, N/K fixed per expert family. Prefill = contiguous layout: experts' tokens concatenated; device `m_indices` (int32) names the expert per M-block; segments aligned to `get_mk_alignment_for_contiguous_layout()` (128 on SM90; SM100 gran_k 32/128 with k_alignment %128==0, `csrc/apis/layout.hpp:93-101`). Decode/CUDA-graph = masked layout: fixed per-expert M + device `mask`; "kernel computes only the valid portions", built for "CPU is unaware of the number of tokens each expert receives" (README). No host round trip in either layout.
- Autotune: NO runtime benchmark — deterministic heuristics enumerate block_m/n/cluster candidates, `get_best_config()` picks (`csrc/jit_kernels/heuristics/common.hpp:18`). SM100 MGroupedContiguous = exactly one candidate: `swap_ab=true, block_m=128, block_n=128, cluster_n=2 if N/128 even and SMs even` (sm100.hpp:30-44). Small-m: block_m 16/32 candidates for m<=16/32 ("smaller block M avoids TMA L2 OOB bound") — SM90 normal GEMM only.
- Staging: Mega MoE mega-kernel overlaps EP dispatch + GEMM1 + SwiGLU + GEMM2 + combine over symmetric buffers; weights pre-transformed and localized into locality domains (weight prefetch pattern, PR #462).
- Numbers: up to 1550 TFLOPS FP8 dense H800 (README, 2025-04); Mega MoE benched on DeepSeek-V4-Flash/Pro under EP8 across per-rank batch sizes (PR #316).
- URL: https://github.com/deepseek-ai/DeepGEMM (PRs #304, #316)

## 3. SGLang — fused MoE triton runner + DeepGEMM runner split (HIGH)
- Grouping: runner input carries precomputed `sorted_token_ids/expert_ids/num_tokens_post_padded` from `sgl_kernel` CUDA align op; small-batch op `moe_align_small_numel` gated by `SMALL_NUMEL_LIMIT` with shared-mem ceiling `_CUDA_SMALL_BATCH_MAX_BUCKETS = 64` documented inline (`moe_runner/triton_utils/moe_align_block_size.py:24-30`). Same bound-grid + device-counter pattern; no host reads in runner paths (grepped).
- DeepGEMM runner (`moe_runner/deep_gemm.py`): masked "standard layout" for CUDA graph (decode), memory-budget auto-policy for prefill; fused masked activation `silu_and_mul_masked_post_quant` (group 128 inline).
- Autotune: canonical offline tuner `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` sweeps BLOCK_M/N/K, GROUP_M, num_warps, num_stages per (E,N,dtype,device,block_shape); writes vLLM-compatible JSONs.
- GDN lead: `benchmark/bench_linear_attention/bench_gdn_prefill.py` + `layers/attention/hybrid_linear_attn_backend.py` — chunked-scan GDN prefill exists; internals not read. Numbers: none in-repo.
- URL: https://github.com/sgl-project/sglang

## 4. TensorRT-LLM — `_torch` AutoTuner + cuBLASLt practice
- Mechanism (HIGH, source read): `tensorrt_llm/_torch/autotuner.py` (2258 ln). Kernels implement `TunableRunner.get_valid_tactics()`; `with autotune(cache_path=...)` profiles each tactic per M-bucket `OptimizationProfile`, picks min-time, caches to a rank-specific file, merges tactics across ranks in one collective (`post_tune_merge_tactics`). Inference = cache hit, zero profiling; miss = fallback tactic + warning.
- cuBLASLt (MEDIUM): NVIDIA official sample = request N heuristics (`cublasLtMatmulAlgoGetHeuristic`), CUDA-event-time each algo (median), pick best for the fixed problem (`LtSgemmSimpleAutoTuning`, https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt). Docs have "Heuristics Cache" §3.1.2 + `...ForStream` §3.4.24 (https://docs.nvidia.com/cuda/cublas/). Grouped GEMM exists as EXPERIMENTAL cuBLASLt feature (FP8/MXFP8/NVFP4 `Lt*gemmGroupedSimple` samples) — viable alternative to a custom grouped launcher, unstable API.
- Blackwell narrow-M (LOW-MEDIUM): cuBLAS "nvjet" kernel family serves SM100 matmuls, not separately documented; heuristic top-K benchmark is the only sanctioned narrow-M method; no first-party skinny-GEMM doc found.
- Note: "engine builder timing cache" maps to TensorRT-core; nothing greppable in current TRT-LLM docs tree (no perf guide on main) — the `_torch` AutoTuner is the verifiable practice.
- Staging (HIGH): `tensorrt_llm/_torch/host_staging.py` — pinned staging ring of 2 buffers per key, CUDA event per slot, buffer rewritten only after prior copy completes; "host may run ahead of the device (overlap scheduling, chunked prefill)" — direct precedent for our KV staging double-buffer. Numbers: none in-repo.
- URL: https://github.com/NVIDIA/TensorRT-LLM

## 5. FlashInfer — autotuner + trtllm-gen MoE + shipped tables (HIGH)
- Mechanism: `flashinfer.autotune()` context; per `(operation, backend, shape)` the autotuner benchmarks all runners × tactics on first call, then reuses cache (docs/autotuning.rst). Timing warmup=3, repeat=10, CUDA-graph capture (`autotuner/autotuner.py:1725-1736`, `_graph_timing.py`); cache keyed by hash + pinned kernel-version string.
- Grouping: trtllm-gen backend (`fused_moe/backends/trtllm/sm100_runner.py`) takes packed topk ids; trtllm-gen kernel does routing/gather internally on device; no host sync in the runner.
- Shipped tables: `flashinfer/tuning_configs/v0_1_trtllm_fused_moe_NVIDIA_{B200,GB200}.py` pin tactic indices for M∈{1,16,128,1024,...} on DeepSeek-V3 shapes (256,7168) — offline-tuned, version-pinned, checked in.
- Frost low-latency MoE: bounded tactic sets ("selected from 15 valid configurations; warm and cold cases can prefer different groups/tiles") with measured latency JSONs shipped (`docs/frost_low_latency_h100.json`, `_b300*.json`): H100 FP16/FP8, tokens=1, H=2048, I=512, E=512, top-k=10 — the only in-repo measured MoE-latency tables across the five stacks.
- URL: https://github.com/flashinfer-ai/flashinfer

## Cross-cutting + open questions
- Universal no-sync trick: static upper-bound allocation + device-written counter + grid-on-bound + per-block early-exit. Two small-chunk shortcuts: block-aligned align kernel with small-batch specialization (vLLM/SGLang); skip alignment under extreme sparsity. Decode-vs-prefill layout split (contiguous/masked) is the industry pattern; masked = CUDA-graph-safe, host-blind. No stack re-uploads bounds/slot tables from host per layer.
- Unverified (fetch truncated / out of scope): cuBLASLt §3.1.2 full text; legacy TRT builder timing cache; GDN prep/scan pipelining internals (only benchmarks/backends located — separate scout needed); cuBLASLt experimental grouped GEMM has no weight-only-quant (MMQ-like) support shown.

## What this changes on the row
- Device grouping kernel = the vLLM/SGLang align pattern: static bound `rows*topk + E*(bs-1)`, grid on bound, per-block early-exit on device counter — deletes the host path and all per-layer cudaStreamSynchronize without dynamic sizing risk.
- Keep the host MMQ path only as a host-resident-weights fallback; adopt the skip-alignment shortcut for `tokens*topk*4 <= E`.
- vLLM `fused_moe_kernel_gptq_awq` (WNA16) is the reference for dequant fused into the GEMM via per-block `expert_ids`; its tuned-JSON key `(E,N,device,dtype,block_shape)` is the natural shape for our table.
- cuBLASLt: adopt NVIDIA's official recipe — heuristic top-N, CUDA-event median per algo at boot for fixed shapes, disk cache keyed by shape+arch+driver; evaluate experimental grouped GEMM before building a custom grouped launcher.
- KV staging double-buffer: copy TRT-LLM `host_staging.py` design (pinned ring=2, per-slot event, grow-on-demand, non_blocking).
- Autotune: prefer warmup-context tactic profiling (FlashInfer/TRT-LLM style) over pure heuristics; ship version-pinned tables like FlashInfer.
