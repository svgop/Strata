# GPU inference optimization dossier — external intel re-pricing (2026-11-20)

Scope: single-GPU consumer-Blackwell (sm_120, RTX 5090-class, CUDA 13.x) MoE decode engine;
bandwidth-bound on IQ3_S expert GEMV (~80% of window time); known wait terms: GPU-reach ~1.4 ms,
adapt-tier ~2.3 ms, CPU-expert spill ~2.8 ms @96% hit. Each finding: mechanism → exact citation →
class ((a) production-proven in a major engine, (b) paper-only, (c) community wisdom).
Priority given to 2025–2026 primary sources. Items that could not be verified exactly are flagged.

## Area 1 — Bandwidth-bound MoE expert GEMV/GEMM: dequant-then-GEMV paths

1. **Marlin W4A16: fused in-register dequant + mma.sync + cp.async multistage pipeline.**
   The kernel dequantizes 4-bit weights in-register (dequant.h) and multiplies on bf16 tensor cores
   (`mma.sync.aligned.m16n8k16`, `cp.async` LDGSTS); its HBM→L2→SMEM→TCU pipeline keeps weight
   loads dominant up to batch ≈ 50; ~4× speedup vs prior 4-bit kernels at batch ≤ 32 on A10, ~3×
   end-to-end in vLLM. Critical for us: it compiles to **native sm_120/121 SASS (HMMA + LDGSTS)
   with nvcc 13 `-arch=sm_121a`** — no TMA/wgmma dependency, so it runs on consumer Blackwell.
   Citation: arXiv 2408.11743 (Frantar et al., "MARLIN"); repo IST-DASLab/marlin; vLLM vendored at
   `csrc/.../quantization/marlin/` (copyright "Elias Frantar / Neural Magic"); MoE grouped variant
   landed in vLLM PR #14447 "[Kernel] moe wna16 marlin kernel" (`moe/marlin_moe_wna16/ops.cu`,
   entry `moe_wna16_marlin_gemm`); generator explicitly names "SM120 RTX 5090, SM121 DGX Spark GB10"
   as targets (`moe/marlin_moe_wna16/generate_kernels.py:16-18`). (a)
2. **Machete: CUTLASS 3.x collective with TMA + weight pre-shuffling for Hopper.**
   Pre-packs weights into an interleaved layout so TMA loads feed wgmma directly; supports w4a16/w8a16/GPTQ;
   in vLLM since v0.6.2 (PR #7174 "[Kernel] (1/N) Machete - Hopper Optimized Mixed Precision Linear
   Kernel"); author's talk (Lucas Wilkinson, Neural Magic) at MIT CSAIL Scale ML, Dec 4 2024 details the
   pre-shuffle. SM90-targeted — not the direct port for sm_120 (see #1/#3 instead). Citation: vLLM PR #7174;
   `csrc/quantization/machete/Readme.md` ("spiritual successor to Marlin, based on Cutlass");
   https://www.csail.mit.edu/event/scale-ml-lucas-wilkinson-machete-cutting-edge-mixed-input-gemm-gpu-kernel-targeting-nvidia. (a)
3. **SM120-specific TMA warp-specialized MoE grouped GEMM exists upstream in CUTLASS/vLLM.**
   vLLM PR #43814 "[Bugfix][SM120] Enable CUTLASS grouped GEMM (MoE) for SM_120/SM_121 consumer
   Blackwell" unblocks grouped MoE on our exact arch; it requires CUTLASS PR #3280 "[SM120] Add
   ptr-array TMA collective for tensor/token-scaled FP8 grouped GEMM" (merged 2026-06-13; headers
   `sm120_mma_array_tma*.hpp`, `MainloopSm120ArrayTmaWarpSpecialized`, CUTLASS pin v4.4.2).
   NVIDIA's tensorrt-edge-llm ships a consumer-Blackwell FP16 MoE grouped GEMM:
   `f16_moe_blackwell_geforce_grouped_fp16` (SMs 120/121, "TMA, LdMatrix, warp MMA" — no tcgen05,
   matching what GeForce Blackwell exposes) and `nvfp4_a16_blackwell_moe` (Thor SM110 W4A16,
   weights dequantized straight into TMEM via tcgen05 kind::f16). Citations: vLLM PR #43814;
   NVIDIA/cutlass PR #3280; nvidia/tensorrt-edge-llm @ 358de11c
   (`kernelSrcs/f16_moe_cutedsl/README.md`, `kernelSrcs/nvfp4_a16_blackwell_moe/README.md`). (a)
4. **cp.async.bulk.prefetch.L2 (TMA-class prefetch, no destination) is now a production tool — with
   measured long-scoreboard evidence on RTX 5090.** llama.cpp PR #26705 "CUDA: branchless Q4_K/Q5_K
   unpack to speed up mmvq, L2 prefetch on DGX Spark" issues L2 prefetches inside the MMVQ kernel:
   ncu shows long_scoreboard 41.2 → 7.9 on the 5090 going from ne11=1→8 without prefetch
   context, but prefetch only pays where latency-bound — on the 5090 it *regressed* at ne11=8
   (7.90 → 8.16 stalls; net −1.57% 5090, −1.39% 4090, +4.81% Spark). Direct pricing lesson for our
   GEMV: prefetch requests are per-loop-iteration costs; they must be issued only while the kernel
   is latency-bound (small ne11 ≈ small token count). Citations: llama.cpp PR #26705; LLVM PR #123226
   "[NVPTX] Add Bulk Copy Prefetch Intrinsics" (compiler surface for cp.async.bulk.prefetch.L2.*,
   PTX ISA ≥ 7.0, CC ≥ 9.0); AAAI-25 paper "Accelerating LLM Inference Throughput via Asynchronous
   KV Cache Prefetching" (ojs.aaai.org article 39224) uses the same instruction. (a)/(b)
5. **Side-stream L2 weight prefetch at fixed layer points is in vLLM production (kimi_k3 path).**
   `vllm/models/kimi_k3/nvidia/specialized_decode/runtime/prefetch.py` launches a small prefetch
   kernel on a side stream at named fork points (e.g., after o-projection → prefetch the MoE front's
   packed router + latent-down rows) so bandwidth-idle kernels (attention core, all-reduces) warm L2
   for the next weight readers; A/B-tuned per TP width (skipped at B8/TP16 where it lost).
   Citation: https://docs.vllm.ai/en/latest/api/vllm/models/kimi_k3/nvidia/specialized_decode/runtime/prefetch/ . (a)
6. **Layer-ahead weight prefetch for VRAM-overflow models: llama.cpp `-pw`.** The backend scheduler
   overlaps layer-N compute with async-copy-engine transfer of layer-N+1 weights; dense models get
   clean overlap (C_n vs T_n+1), but MoE *cannot* know next-layer expert selection, so the win needs
   large ubatch (more experts touched per layer) — mirrors our adaptive expert cache's prediction
   problem exactly. Selective expert copy itself is llama.cpp PR #15346. Citations: llama.cpp `-pw`
   flag (llama-bench); PR #15346; analysis https://am17an.bearblog.dev/prefetching-weights-in-llamacpp/ . (a)
7. **cudaAccessPolicyWindow / persisting L2**: API is documented (CUDA C Programming Guide §L2
   persistence, `cudaAccessPropertyPersisting` + hitRatio; PTX `.level::cache_hint`), and vLLM's
   kimi_k3 prefetch module is the production-precedent alternative (explicit prefetch instead of
   policy windows). No major engine was found shipping `cudaAccessPolicyWindow` on the decode path
   in 2025-26 — treat as unproven for MoE streaming (paper-only / community). (b)/(c)
8. *Unverified:* "moores" — Neural Magic's SM100 Blackwell mixed-input GEMM successor to machete —
   exists in the vLLM tree, but its introducing PR number could not be pinned this pass (GitHub code
   search requires auth). Treat as (a) existence-confirmed only via tree presence; verify PR before citing.

## Area 2 — Warp-specialized producer-consumer decode kernels

1. **TRT-LLM MoE GEMM is TMA warp-specialized in production**: `moe_kernels.cu` calls
   `setupTmaWarpSpecializedInputs(...)` and runs fc1/fc2 through TMA-WS CUTLASS kernels, with
   opportunistic fusion of FC2 prequant scaling into the FC1 `doActivation` kernel.
   Citation: NVIDIA/TensorRT-LLM `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`
   (@ 7a8bd87f and current main). (a)
2. **CUTLASS example 77 (Blackwell FMHA) is the reference warp-role decode kernel**: producer warp
   (TMA load) / MMA / correct / epilogue roles selected by `warp_idx_to_role`, pipelines
   `consumer_wait/producer_commit` around tcgen05 MMA; files
   `sm100_fmha_mla_tma_warpspecialized.hpp`, `sm100_fmha_gen_mainloop_warpspecialized.hpp`.
   Citation: NVIDIA/cutlass `examples/77_blackwell_fmha/` (@ e6e2cc29, f74fea9). (a) (library exemplar)
3. **CUTLASS 3.x/4.x collective mainloop = the memory-bound GEMV recipe**: SM90/SM100
   `CollectiveMma` with producer TMA warps + consumer Math warps coordinated by pipeline barriers;
   array (ptr-array) variants extend this to grouped/MoE GEMM; SM120 array collective added by
   CUTLASS PR #3280 (see Area 1 #3). Citation: NVIDIA/cutlass `include/cutlass/gemm/collective/`
   (sm90/sm100/sm120 `*_array_tma_warpspecialized*`); overview: DeepWiki "Collective Mainloop and
   Pipeline Management" (NVIDIA/cutlass §5.1). (a)
4. **DeepGEMM (DeepSeek): SM90/SM100 FP8 GEMM via JIT, CUTLASS-concepts-without-CUTLASS-templates;
   grouped/masked variants are the MoE path** (`m_grouped_fp8_gemm_nt_contiguous` for prefill,
   `_masked` for decode); up to 1550 TFLOPS on H800 (PRs #74/#78/#81/#86, commit 340d988).
   Citation: deepseek-ai/DeepGEMM. (a)
5. **MegaMoE trend**: vLLM v0.20.0 added `--moe-backend deep_gemm_mega_moe` (DeepSeek-written mega
   kernel, Blackwell only) — single fused MoE kernel to kill inter-kernel gaps at decode.
   Citation: vLLM v0.20.0 release notes / @vllm_project 2026-04-27 post. (a) Research line:
   Mirage MPK (GPU MODE lecture 79), RaMP (arXiv 2604.26039, runtime-polymorphic MoE megakernel,
   TMA weight streaming dominant), GC-MoE (arXiv 2611.04222, CUDA green contexts overlapping EP
   dispatch with FP8/FP4 grouped GEMM — includes a warp-specialized-single-kernel baseline they
   reject because roles cannot repartition SMs without draining). (b)

## Area 3 — KV/prefix prefetching & paged-KV for 128K–262K context

1. **vLLM: eager KV prefetch for WAITING queue requests** — scheduler lifts future requests' KV
   pages while others prefill. Citation: vLLM PR #44731 "fix(scheduler): eager KV cache prefetch for
   waiting queue requests". Related: PR #7753 "Enable chunked prefill and prefix caching together"
   (canonical interplay of chunking + prefix hits). (a)
2. **Prefix-tree scheduling works and is measured**: prefix-tree sorting raises block-cache hit
   rate +73.6%, bubble-fill +65.2%, offline blocks completed +77.4% in a SOSP '26 production system.
   Citation: SOSP '26 paper, dl.acm.org doi 10.1145/3830418 (heuristics-for-prefill section). (a)/(b)
3. **SGLang hierarchical radix cache for 262K-context serving**: HiRadixCache (L1 GPU + L2 host +
   storage tiers, DMA prefetch policies, write-through, `hicache-storage-prefetch-policy`) shipped
   for P/D-disaggregated decode; the Unified Radix Cache blog demonstrates DeepSeek-V4-Flash-FP8 at
   `--context-length 262144 --page-size 64` with Mooncake 500 GiB tier. Citations: sgl-project/sglang
   PR #26560 "[P/D disagg] Add HiRadixCache (hierarchical KV cache) for decode disaggregation";
   lmsys.org blog 2026-08-11 "Unified Radix Cache: One Tree for Hybrid Model Prefix Caching". (a)
4. **KV compaction (closed-form/analytic) that survives long context**: SnapKV — pooled observation
   window selects KV positions (arXiv 2404.14461, NeurIPS 2024; validated to 380K NiaH);
   MiniCache — cross-layer KV merging via SLERP on adjacent-layer similarity, O(1)-style compaction
   per layer pair (arXiv 2405.14366, NeurIPS 2024); FlagRot — rotation before 2.5–3-bit KV pack,
   served in vLLM with paged INT2 + sink/recent windows (arXiv 2611.03992). All (b) with vLLM-based
   prototypes; none default-on in a major engine for exact-match-sensitive speculative decoding.
5. **FlashInfer page-scheduling**: paged attention with page-table indirection is FlashInfer's core
   decode API (batch_decode_with_kv_cache over kv_indptr/kv_indices; TRTLLM-gen paged kernels
   auto-selected on SM100+); spec-decode (multi-query verify) needs the non-causal decode variants
   still being generalized — vLLM tracking issue #46105 (DFlash bring-up) lists
   flashinfer-ai/flashinfer#3570 (non-causal TRTLLM-gen) and XQA as the optimized spec-decode KV
   kernels. Citations: flashinfer docs (attention module); vllm #46105; flashinfer #3570. (a)

## Area 4 — FP8/NVFP4 weight+KV quantization without quality collapse

1. **Production precision splits (TRT-LLM reference matrix)**: `tests/integration/defs/accuracy/
   references/mmlu.yaml` records per-model quant_algo + kv_cache_quant_algo (FP8 weights + FP8 KV
   for Llama-3.x/Qwen3-30B-A3B/Mixtral; NVFP4 for MiniMax-M3-NVFP4 etc.) with MMLU references —
   the de-facto "recipes" file. Citation: NVIDIA/TensorRT-LLM @ b8a5159f or main (same path). (a)
2. **NVFP4 weight quant cost is real and bounded**: NVIDIA's DeepSeek-R1-NVFP4 model card reports
   MMLU(5-shot) FP8 90.8 → NVFP4 86.9. Citation: huggingface.co/nvidia/DeepSeek-R1-NVFP4
   (commit b39d5496c7effd9f104f8fde476d0866bf035028). Per-layer splits (dense FP8 vs expert FP4)
   come from modelopt recipes; gpt-oss ships MXFP4 experts + bf16 attention/dense parts. (a)
3. **FP8 KV cache: vLLM production numbers + the sliding-window caveat**: FP8 KV cuts ITL slope
   ~54% at conc 1 → +14.9% output tok/s at conc 8 (Llama-3.1-8B); only +4.8% on gpt-oss-20b because
   SW layers cap memory savings — "skip-SW" variant applies FP8 KV only where it pays.
   Citation: vllm.ai blog 2026-04-22 "The State of FP8 KV-Cache and Attention Quantization in vLLM". (a)
4. **Speculative decoding + mixed precision**: QSPEC drafts on W4A16 and verifies on W8A16, sharing
   weights, with a probability-scale-aware acceptance to absorb the precision mismatch — the
   published treatment of exactly our exact-match-verification concern (draft/verify numeric
   divergence changes acceptance, not just speed). Citation: ACL Anthology 2025.emnlp-main.240
   ("QSPEC"). Companion evidence that engine/batch/kernel drift alone moves outputs at a measurable
   fraction of 4-bit effects: arXiv 2611.15671 ("The Noise Floor of a Quantization Comparison",
   Cloudflare, GB10). Also vLLM PR #54955 "[DeepSeek-V4] Batch-invariant kernels for 0-diff RL" and
   PR #46592 (prefix-cache hits change effective shapes; batch-invariant mode needs canonical
   chunking) — same canary logic as our M3b law. (b)/(a)
5. **Consumer-Blackwell caveat**: SM120/121 W4A4 FP4-tensorcore paths are separate compile targets
   from SM100 (vLLM CMake: ENABLE_NVFP4_SM100 vs ENABLE_NVFP4_SM120; ENABLE_CUTLASS_MOE_SM100 vs
   _SM120) and kernel-selection bugs shipped (W4A16 fallback silently chosen on SM120/121 — fixed
   by vLLM PR #55170 reordering `_POSSIBLE_NVFP4_KERNELS`). Citation: vLLM PR #55170; PR #43814;
   vLLM FP8/low-precision DeepWiki §7.2. (a)
6. **Trend line for 2025-26**: MXFP4/NVFP4 near-FP8 quality (≤1%) claimed for gpt-oss/DeepSeek-V4
   class models; TRT-LLM FP4 MoE fixes ongoing (e.g. PR #22108 MXFP4/FP8 MoE SwiGLU scale
   correctness). Citations: vLLM PR #37463 (MXFP4 W4A4 CUTLASS MoE SM100, E8M0 block-32 vs NVFP4
   E4M3 block-16 + global scale); TensorRT-LLM PR #22108. (a)

## Area 5 — CUDA graph capture across dynamic shapes/batch regimes

1. **vLLM dual-mode graphs**: `CUDAGraphWrapper`/`CudagraphDispatcher` with modes PIECEWISE (split
   at attention, capture each Inductor-compiled FX subgraph per shape, attention eager — handles
   varlen) and FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE (padding buckets for decode), with automatic
   downgrade when the attention backend can't do FULL. Citation: vLLM CUDA Graphs design doc
   (vllm.website.cncfstack.com/design/cuda_graphs/; source `vllm/config/compilation.py`
   CUDAGraphMode). (a)
2. **TRT-LLM piecewise capture coverage bug class**: with `enable_piecewise_cuda_graph=True`,
   num_tokens above the largest captured candidate silently fell back to eager; fix "Broader
   capture of piecewise cudagraph" — capture-set must enumerate reachable padded num_tokens.
   Citation: TensorRT-LLM PR #13574 (nvbug 5615248, commit 9c1869b3c0ab), documented in
   `.claude/skills/perf-regression-cookbook/references/cuda-graph-and-compile/piecewise-cudagraph-capture-coverage.md`. (a)
3. **SGLang: one graph per captured batch size** — `get_batch_sizes_to_capture()` derives the list
   from max_bs; `capture_one_batch_size(bs,...)` per bucket; memory pool shared across graphs;
   `--disable-cuda-graph-padding` gates padding-dependent replay; `--enable-profile-cuda-graph`
   instruments capture. Citation: sgl-project/sglang `python/sglang/srt/model_executor/
   cuda_graph_runner.py` (@ 7cb4669a) + docs/server_arguments.mdx. (a)
4. **Replay-overhead numbers**: NVIDIA's canonical example — a 2.9 µs kernel costing 9.6 µs/step
   with synchronous per-kernel launches drops to 3.4 µs as a graph (relayed by
   morphllm.com/cuda-graphs; original in NVIDIA CUDA graphs material) — the launch-tax scale our
   per-window graph replay already banks. (c) (secondary relay of vendor number)
5. **Graph capture vs background work (our dual-cohort hazard)**: default
   `capture_error_mode="global"` invalidates capture if ANY host thread issues CUDA work during
   capture — breaks side-stream KV-offload daemons / async weight prefetchers; fix uses
   `capture_error_mode="thread_local"`. Citation: vLLM PR #48053. Also vLLM PR #45623 (graph memory
   offload under sleep mode via cumem allocator — VMA-preserving restore, no re-capture). (a)
6. **Adaptive speculative depth × graphs**: PolyGraph — plan-buffered CUDA graphs where Triton
   tiles read active extents from a device plan, avoiding both shape-cache misses and MaxPad waste
   when draft depth d < d_max. Citation: arXiv 2610.23861. (b) (paper-only; directly our MTP
   depth-dynamism shape)

## Area 6 — Host-device sync elimination in decode loops

1. **PDL is default-on in TRT-LLM on SM≥90**, with a per-kernel accuracy-WAR ledger: commits
   8462cf6c96f1 (TRTLLM-9578, PR #9695 — default on), 84d2f1281857 (PR #7977
   `launchWithPdlWhenEnabled`), 21a93fbf9d10 (PR #10043 CuteDSL + MoeOutputMemset overlap); PDL
   disabled per-kernel for correctness at 9e7b50aefb (PR #10285 quant), 6ee8dbfe0b/ba25b6afae08
   (TinyGEMM), bf7142f8d1 (PR #11994 trtllm-gen routing NaN), 1c4dacb19a52 (PR #9799 dsv3 MoE),
   34e2fa5c96 (PR #9530 Qwen3-235B ATP). Mechanism: dependent kernel runs its preamble while the
   producer's tail drains (`cudaLaunchAttributeProgrammaticStreamSerialization`; PTX
   griddepcontrol.wait / launch_dependents). Citation: TensorRT-LLM perf-optimization-casebook
   `references/runtime-execution/pdl.md` (fetched 2026-11-20). (a)
2. **vLLM adopted PDL for decode kernel chains**: PR #50230 "[Perf][CUDA] Programmatic dependent
   launch for the DSA decode kernels" — `USE_PDL` constexpr with `gdc_wait()/gdc_launch_dependents()`
   on fused norm/rope + fused-q Triton kernels, `cudaLaunchKernelEx` PSS for NVFP4 quant kernels,
   runtime-gated `is_arch_support_pdl()`. (a)
3. **FlashInfer: PDL across the kernel surface + one documented race**: `enable_pdl` params are
   default-on where supported (e.g., softmax auto-enables PDL on CC≥9.0; rmsnorm defaults True);
   PR #3835 "disable PDL for finalize kernel to prevent GEMM2-write TMA race" is the canonical
   async-proxy-vs-generic-load visibility bug (needs `fence.proxy.async` ordering, not
   `fence.proxy.alias`) — our exact-match canary must watch this class. PR #7614 (cake MegaMoE)
   threads `use_pdl` through compact/persistent kernels. Citations: flashinfer-ai/flashinfer PRs
   #3835, #7614; docs.flashinfer.ai (enable_pdl). (a)
4. **Overlap schedulers (doorbell-class, no PDL)**: TRT-LLM Overlap Scheduler launches step n+1 GPU
   work before step-n CPU postprocessing completes (`_sample_async`); SGLang zero-overhead batch
   scheduler prepares batch N+1 on CPU while GPU runs N (default path; FutureMap negative-index
   placeholders resolved by a GPU kernel; per-batch `torch.cuda.Event` on the D2H copy); vLLM V1
   AsyncScheduler replaced an async-barrier fence with fresh pinned copies to kill the shared-buffer
   race. Citations: NVIDIA/TensorRT-LLM docs `features/overlap-scheduler.md` (@ e6caf6f0);
   sgl-project/sglang overlap scheduler (lmsys meetup slides, sgl-learning-materials; default since
   v0.4.x, current code `scheduler.py`/`future_map.py`); vLLM MRV2 async-scheduling notes
   (uw-syfi/vibe-serve async-scheduling.md). (a)
5. **Published PDL gain numbers**: TRT-LLM records direction-only ("measured Δ to be recorded from
   run" — its casebook `measured: []`); no vendor-published % was found for small-kernel chains —
   numbers live in per-PR discussion, so treat any PDL claim as requiring our own interleave.
   The mechanism class (producer-tail/consumer-preamble overlap) is production-proven per #1-#3. (a/c)

## Area 7 — GPU MODE / CUDA MODE community: talks & threads

1. **GPU MODE lecture 37: "Introduction to SASS & GPU Microarchitecture"** (Arun Demeure, NVIDIA) —
   the stall taxonomy source (long-scoreboard = waiting on global-memory return) for reading our
   GEMV ncu profiles. Citation: cuda-mode/lectures `lecture_037` (github.com/cuda-mode/lectures). (c)
2. **GPU MODE lecture 7: "Advanced Quantization"** (HDCharles) — CUDA-vs-Triton dequant kernel
   structures; and **lecture 34: "Low Bit Triton Kernels"** (Hicham Badri) — low-bit weight kernels
   incl. interleaving/dequant layout choices. Citations: cuda-mode/lectures lecture_007, lecture_034. (c)
3. **GPU MODE lecture 36 (Jay Shah, Colfax): "CUTLASS and Flash Attention 3"** + Colfax's CUTLASS
   tutorial series (parts 1-6, esp. part 4 "Hardware-supported Block-scaling with NVIDIA Blackwell
   GPUs") + lecture 103 "Fundamentals of CuTe Layout Algebra and Category-theoretic Interpretation"
   (Jack Carlisle & Jay Shah) — producer/consumer mainloop and swizzle-layout canon.
   Citations: cuda-mode/lectures lecture_036/lecture_103; research.colfax-intl.com (CUTLASS tutorial
   series; GPU Mode tagged posts). (c)
4. **GPU MODE lecture 23: "Tensor Cores"** (Vijay Thakkar & Pradeep Ramani) — the
   "Present and Future of CUTLASS" lineage; companion deck (SemiAnalysis-hosted PDF) documents
   Blackwell tcgen05/TMEM/2-SM MMA and cluster-scheduling changes. Citation: cuda-mode/lectures
   lecture_023; https://semianalysis.com/wp-content/uploads/2025/03/The-Present-and-Future-of-CUTLASS-Tensor-Core-Programming-Vijay-Thakkar.pdf . (c)
5. **Long-scoreboard mitigation with ncu numbers, quantized MMVQ, 1-4 col decode**: llama.cpp PR
   #26705 discussion (Area 1 #4) — the single best public thread tying long_scoreboard percentages
   to prefetch cost/benefit on 5090 vs Spark; QUICK paper (arXiv 2402.10076) formalizes
   conflict-free dequant interleaving (bank conflicts + locality). Citations: llama.cpp PR #26705;
   arXiv 2402.10076. (c)/(b)
6. **L2 partitioning on Blackwell**: GB202 (RTX 5090) die analysis shows L2 as two 32 MB tiers of
   2 MB blocks around the die center (locality matters for weight placement); datacenter B200 is
   dual-die with 4 physically partitioned L2 segments presented as one coherent cache — cross-die
   addresses pay NV-HBI latency. Citations: GB202 die-shot analyses (Tom's Hardware/Kurnala, Jan
   2025, via gamegpu/3dnews relays); community L2-coherence writeups citing Chips and Cheese
   microbenchmarks (blog.csdn.net/eloudy 166948854). Class (c) — community wisdom; original Chips
   and Cheese latency tables not fetched this pass.
7. Adjacent useful lectures: #22 "Hacker's Guide to Speculative Decoding in vLLM" (Cade Daniel),
   #40 "FlashInfer" (Zihao Ye), #57 "CuTe" (Cris Cecka), #72 long-context efficiency (Guangxuan
   Xiao), #79 "Mirage (MPK): compiling LLMs into mega kernels", #84 "Numerics and AI"
   (Paulius Micikevicius), Bonus "CUDA C++ llm.cpp" (Hemstad & Evtushenko). Citation:
   github.com/cuda-mode/lectures README (fetched 2026-11-20). (c)

## Transfer notes for our engine (re-pricing, not recipes)

- **IQ3_S GEMV on sm_120**: the credible port ladder is (i) llama.cpp few-row MMQ PR #31533
  (int8-MMA path for 2-24 src1 columns, IQ3_S supported, Ampere+) — matches our spec-verify shapes
  exactly; (ii) Marlin-style in-register dequant + mma.sync (proven native sm_120/121 SASS);
  TMA/wgmma collectives (machete-class) are SM90 tools — on SM120 use the ptr-array TMA +
  LdMatrix/warp-MMA pattern (CUTLASS #3280 / tensorrt-edge-llm "geforce" variant), not tcgen05.
- **Prefetch pricing**: L2 prefetch pays only while the GEMV is latency-bound (llama.cpp #26705
  5090 data); vLLM kimi_k3 shows side-stream prefetch at fixed layer fork points is the
  production shape; adapt-tier wait (2.3 ms) maps to their A/B-per-TP tuning (prefetch points are
  conditional, not universal).
- **CPU-expert spill (2.8 ms)**: llama.cpp `-pw` analysis prices the MoE prefetch problem —
  next-layer expert selection is unknown at issue time; the honest lever is larger effective
  windows (our cohorts) or predictive prefetch scored like kimi_k3's fork points.
- **Exact-match canary**: QSPEC + noise-floor (2611.15671) + FlashInfer PDL race (#3835) say
  numeric-path changes (PDL ordering, FP8 KV, kernel swaps) move acceptance before they move
  benchmarks — M3b stays the gate for every entry above.

*Compiled 2026-11-20 from live web sources; all URLs/IDs as cited. Items flagged unverified:
moores PR number; NVIDIA graph-launch 2.9/9.6/3.4 µs (secondary relay).*
