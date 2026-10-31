---
type: digest
id: w4-rendezvous-scout
created: 2026-10-31
status: evidence
---

# W4 — MoE per-layer machinery: how leading stacks erase host rendezvous

Question: where do leading MoE implementations run routing/planning (device vs host), how do they overlap expert misses, do any prefetch next-layer experts, and how many launches survive per layer?
Field verdict: no production GPU-resident stack keeps a per-layer host plan pass. Routing, align/sort, dispatch, combine, and even grid sizing all run on device with fixed/upper-bound launches; CUDA-graph capture is the forcing constraint. Next-layer expert prefetch exists in two forms: NVIDIA's DWDP (VMM slot swap after layer completion) and Eliseev/Mazur's speculative load (guess from current layer's hidden state, issued mid-compute).

## vLLM fused_moe (Triton)
- Mechanism: 5-7 stream-ordered launches per layer: routing (topk_softmax, outside module) -> `moe_kernel_quantize_input` -> `moe_align_block_size` -> GEMM1 (w13) -> activation (silu·mul) -> re-quantize -> GEMM2 (w2) -> `ops.moe_sum`.
- Plan location: device. Source-verified: no `.item()`/`.cpu()`/`.tolist()`/synchronize in `fused_experts_impl`. Grid sized from host-known upper bound `EM = min(sorted_token_ids.size(0), A.size(0)*top_k*BLOCK_SIZE_M)`; kernel early-exits on the device count `num_tokens_post_padded` loaded via `tl.load` — oversized grid replaces the doorbell.
- Sparse-case shortcut: `_prepare_expert_assignment` skips align/sort entirely when `num_tokens*top_k*4 <= num_experts` (bs=1 top-10 on 512 experts: 40 < 512, qualifies) and passes a host-computed constant padded count.
- Combine fusion: router weight fused into GEMM2 epilogue (`MUL_ROUTED_WEIGHT`, fp32 multiply before cast). Miss handling: none — all experts resident; decode runs under CUDA graphs, so sync-freeness holds by construction (capture would fail on a sync).
- Measured: no per-layer overhead published in-module. TritonMoE (2026-03) measures an equivalent fused dispatch path at 5 launches vs 24+ for the CUDA baseline, 89-131% of CUDA perf at bs <= 512.
- Sources: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/fused_moe ; https://github.com/vllm-project/vllm/issues/16294
- Confidence: high — primary source read.

## SGLang fused MoE
- Mechanism: Triton fused MoE with an align-and-sort preprocessing step; the 2025-03-25 design post replaces explicit sorting: `sorted_token_ids` pads each expert group to BLOCK_M with invalid token indices and the GEMM skips them — "alignment without a costly separate sort" (blog since moved/404; design corroborated by two secondary sources and kernel docs).
- Router fusion: multiple fused gate/topk kernels (sigmoid+bias+softmax+topk+renorm, e.g. `moe_fused_gate` for DeepSeek) run as one device kernel per model family; issue #26771 (2026) consolidates them — evidence router+topk are never split host-side.
- Plan location: device; decode runs under CUDA graphs. DeepGEMM masked grouped GEMM integrated via `deep_gemm_wrapper` for unknown per-expert loads; DeepEP low-latency for EP dispatch.
- Launch fusion: Breakable CUDA Graph is the default capture path since v0.5.15 ("reducing per-step kernel-launch overhead"); PDL enabled across DeepSeek-V3.2/GLM-5 kernels (v0.5.12). DWDP prefill mode prefetches peer expert weights over NVLink P2P (v0.5.17, #29778).
- Measured: release-notes deltas only (TP LMHead allgather->a2a cut 320us->169us on DeepSeek-V4-Pro decode; DSA prefill topk v2 kernel 1.3-1.8x on B200). No per-MoE-layer handshake numbers.
- Sources: https://lmsys.org/blog/2025-03-25-moe-align-sort (404, archived design) ; https://github.com/sgl-project/sglang/issues/26771 ; https://sglang.ai (release notes v0.5.12-v0.5.17)
- Confidence: medium-high — design corroborated; primary blog text unreachable.

## TensorRT-LLM MoE (PyTorch backend, MOE_DEVELOPER_GUIDE)
- Mechanism: pipeline routing -> quantize/dispatch -> FC1 -> act -> FC2 -> finalize/combine per chunk. Routing runs on device: scheduler consumes "the original contiguous CUDA int32 route tensor" and the doc orders "do not restore route staging, tail fills, or a host-generated copy plan".
- MegaMoE (FUSED_COMM): one fused kernel = "dispatch+GEMM+act+GEMM+combine" (DeepGEMM `fp8_fp4_mega_moe`) or "dispatch+FC1+act+FC2+combine" (CuteDSL NVFP4); "no host comm; lockstep chunk launches". `fused_fc12` keeps the FC1->FC2 intermediate on-chip in one persistent kernel. Fused finalize mandatory on SM107 (disabling raises `FINALIZE_FUSION_REQUIRED`); v0.21.0 added fused finalize+allreduce (Qwen-MoE).
- Spin-wait is inside the mega-kernel: comm kernels "wait for their peers inside the kernel" with device-side timeouts; zero-token chunks still launch so peers cross the in-kernel NVLink barrier. Not a separate spin kernel per layer.
- Prefetch: DWDP — "after a layer finishes, prefetch the next MoE layer in the cyclic sequence whose slot is now free (the last layers of a pass prefetch the first layers of the next pass)". CUDA VMM composite VA + page pool on a copy stream; cached peer-interleaved `cudaMemcpyBatchAsync` plan with `contention_opt=True`; `invalidate_prefetch()` on control actions. Shared-expert work enqueues ahead via `MoE.load_balancer_overlap`.
- Measured: none published in the guide (timeouts only).
- Sources: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md ; https://github.com/NVIDIA/TensorRT-LLM/discussions/6606
- Confidence: high — official developer guide, direct quotes.

## DeepGEMM (DeepSeek)
- m-grouped contiguous (prefill/training): groups only the M axis, N/K fixed; "each expert segment must be aligned to the GEMM M block size (`get_mk_alignment_for_contiguous_layout()`)".
- m-grouped masked (decode): exists exactly "when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives" — pass a mask tensor, "the kernel computes only the valid portions". Canonical input: DeepEP low-latency kernel output. This is the shipping answer to host-unplanned decode.
- Mega MoE (2026-04, PRs #304/#316): "fuses and overlaps EP dispatch, linear 1 and linear 2 (FP8xFP4 or FP8xFP8), SwiGLU, and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation". PDL supported (`set_pdl`). JIT-compiled at runtime; SM90/SM100.
- HyperConnection: the README ships fused HC kernels ("HyperConnection (HC)", Mega mHC in 2026-09 PR #432) — direct precedent for fusing a hyper-connection Mix/Combine node adjacent to MoE.
- Measured: up to 1550 TFLOPS FP8 on H800 (2025-04, PR #74); "matches or exceeds expert-tuned libraries".
- Sources: https://github.com/deepseek-ai/DeepGEMM ; https://github.com/deepseek-ai/DeepGEMM/pull/304 ; https://github.com/deepseek-ai/DeepGEMM/pull/316
- Confidence: high — README read in full.

## KTransformers (SOSP25) + PowerInfer-2 — CPU/GPU hybrid placement
- Delegation: "Heterogeneous expert placement (hot experts on GPU, cold experts on CPU)"; attention and shared ops on GPU, routed FFN on CPU with Intel AMX/AVX512 INT4/INT8 kernels; operator substitution via YAML match/replace injection rules. GPU placement updated from routing statistics gathered during prefill (hot-expert caching).
- Overlap: CUDA Graphs "extended to MoE models" (SOSP25 paper); CUDA Graph in `prefill_and_generate`. Multi-stream CPU/GPU overlap and next-layer prefetch appear in secondary summaries but were NOT verifiable in primary docs — treat as unconfirmed.
- Measured (v0.3 README): DeepSeek-R1/V3 671B in "14GB VRAM and 382GB DRAM"; prefill 54.21 tok/s (32 cores) -> 255.26 tok/s (AMX MoE kernel) = "up to 27.79x speedup" vs llama.cpp; decode 8.73 -> 13.69 tok/s ("selectively using 6 experts") = 3.03x. SOSP25: prefill speedups 4.62-19.74x over existing systems.
- PowerInfer-2 (arXiv 2406.06282): neuron-cluster pipelining overlaps I/O with compute; prefetches expert parameters from storage while computing current clusters; SMT/Z3 cost model plans the CPU/GPU/NPU/flash split. 47B MoE on a smartphone, up to 29.2x speedup, 11.68 tok/s.
- Sources: https://github.com/kvcache-ai/ktransformers (README, v0.3 tag) ; https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf ; https://arxiv.org/abs/2406.06282
- Confidence: high on placement/AMX/numbers (primary); low on stream-overlap/prefetch claims (secondary only); medium on PowerInfer-2 (abstract + summaries).

## Mixtral offloading (Eliseev & Mazur, arXiv 2312.17238)
- Invariant stated: "the system cannot pre-fetch the next layer until it learns which experts should be loaded" — dense-model layer-ahead scheduling breaks for MoE. Their fix is a next-layer guess, not a plan change.
- Lookahead: apply NEXT layer's gating function to CURRENT layer's input hidden states (residual streams make current hidden states a good estimate of next layer's). Trigger: "immediately after the system finished loading all experts for the current layer"; fetch 1-2 most likely next experts into b=4 async prefetch buffers shared across all layers, never displacing the LRU cache.
- Cache: LRU with k=2 (12GB GPU) / k=4 (16GB) experts per layer; experts live in one contiguous pinned host buffer each so a move is a single H2D copy.
- Measured (Table 2, 2-bit experts, tok/s): RTX 3060: full 2.278; without pre-loading 2.051; without LRU+pre-load 1.547; naive accelerate offloading 0.919 — speculative loading ~+11% over LRU-only, full scheme ~2.5x over naive. Interactive 2-4 tok/s on T4/3060/3080M at bs=1.
- Sources: https://arxiv.org/html/2312.17238v1 ; https://github.com/dvmazur/mixtral-offloading
- Confidence: high — full text read.

## What this changes on the row
- Kill the doorbell, keep the grid: every leader sizes launches from host upper bounds and gates work on device counters (vLLM `tl.load(num_tokens_post_padded)` early-exit; TRT-LLM bans "a host-generated copy plan"). The fused router+plan+doorbell node must keep the plan tensor on device and never return it.
- Our GPU spin-wait kernel is the pre-fusion shape of TRT-LLM MegaMoE's in-kernel barrier: the industry endpoint is one mega-kernel (dispatch+GEMM+act+GEMM+combine), not a spin kernel waiting on a host pass. DeepGEMM Mega MoE is the shipping reference; it also ships fused HyperConnection kernels (Mega mHC), so the unfused Mix/Combine node on this row is fusable today.
- Lookahead staging has two precedents with different triggers: DWDP prefetches the next layer's experts AFTER a layer finishes (copy stream, VMM slots); Eliseev/Mazur issue the prefetch MID-layer from a gate-on-hidden-state guess. Our device-driven next-layer staging should trigger on current-layer plan availability to hide the CPU miss-GEMV window, and fall back to DWDP-style completion triggering.
- Decode guard worth stealing: vLLM skips align/sort when tokens*topk*4 <= num_experts — bs=1 top-10 on 512 experts qualifies; mirror this sparse path in the CPU miss planner (host constants, no device sort dependency).
- CUDA-graph capture is the constraint that forces all of this: DeepGEMM masked grouped GEMM exists precisely for "CPU unaware of per-expert token counts". The double-buffered deferred combine must be shape-fixed and mask-validity-driven to stay capturable.
- No competitor reports our 48-96 handshakes/window because none runs a per-layer host plan in decode; treat 25-40us/layer as self-incurred debt. Launch-count budget for the fused node: leaders land 1-7 launches/layer; fusion should aim to hold that band even with CPU misses.
