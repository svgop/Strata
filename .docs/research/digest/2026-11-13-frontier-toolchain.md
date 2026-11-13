# Frontier toolchain/compilation audit (2026-11-13) — CUDA 13.x/Blackwell features vs our build

Batch 2 of the 18-agent frontier wave. Headline: the tree is 2-3 toolchain generations past the plain-launch
era (clusters, PDL, conditional graphs, cudaMemcpyBatchAsync, L2 policy windows, 120a, cp.async pipelines,
hugetlb pinned arenas all in-tree). The gaps are ADOPTION (built + default-off), INSTRUMENTATION, one free
correctness guard, and two heavyweight idioms with bounded consumer-Blackwell value.

## Ranked findings
1. **CMake configure guard misses the poisoned nvcc 13.2.0/13.2.1** — the banked miscompile law (uint8_t-from-
   packed-u32, sm_120) lives in docs/digests but CMakeLists.txt:213-216 only checks <13.0; 13.2.x passes
   silently and builds a WRONG engine. ~6 lines, free, protects every community build. This box: 13.4 (safe).
2. **No -lineinfo in the CUDA release path** — zero-codegen-change flag that enables all NCU source-correlated
   profiling; we have ~100 bench/parity binaries to point ncu at. One line. (No first-party NCU session exists
   in the record — cites other projects' evidence only.)
3. **cudaMemcpyBatchAsync implemented but default OFF; CUDA 13.4's PreferOverlapWithCompute (mode 2) never
   exercised** on the 13.4 box. dma_batch.hpp full implementation + parity test green; the gap is a battery,
   not code. Small-medium on expert staging/upload rounds.
4. **Conditional-node window graph: machinery complete, the named mode/retry arm unclaimed** (verify.cpp:2552-
   2622; the comment names it). Low-medium; the falsifier bench exists.
5. **TMA/cp.async.bulk unused** — bounded: fa-5090 measured 94.4% of SOL on a 5090 with Ampere-style cp.async
   + 2-stage pipelining and NO TMA; we already have cp.async in all 9 hot tile kernels. TMA buys address-gen
   relief on the largest tiles only; subject to the 13.2 probe law. High port.
6. Warp specialization — FA3-class numbers are TMA/WGMMA/tcgen05 hardware; on sm_120 the room is the last ~5%.
   Prompt attention + GDN chunk kernels are the candidates. High port, small-medium gain.
7. LTO — low (helpers are .cuh headers textually included; cross-TU device-call density near zero); host CXX
   IPO single-digit at best; one measured try only.
8. **Stream priorities unused** for DMA/refill streams (all NonBlocking, no WithPriority) — upload-vs-verify
   arbitration on a one-copy-engine card. Trivial; hypothesis for the interleave battery.
9. 120a arch-specific target confined to the NVFP4 W4A4 unit — arm only when NVFP4 decode experts become a
   standing pack. CUTLASS grouped GEMM on sm_120: do NOT adopt (field-broken; our bitwise-validated kernels
   are the safer implementation).
10. No MPS/green contexts for the picture process co-residency — concurrency-only, operationally fragile.

## Already right (audited)
Conditional graphs fully built; cudaGraphUpload-after-instantiate fix; capture-mode tiers; graph hygiene
(zero-node guard, destroy ordering, registry by (layer,n_tokens)); device-graph-launch correctly EXCLUDED;
clusters in production kernels with parity; PDL engineered to expert depth (griddepcontrol split, per-arch
gating, Pascal regression guard); batched copy API; cp.async in 9 kernels; __launch_bounds__ in 38/48 files;
MaxDynamicSharedMemorySize set; L2 persistence windows; pinned/hugetlb/VMM memory paths; --use_fast_math
correctly fenced with parity; EAGER module loading with VRAM budgeting; serve/ has no runtime compile; the
miscompile law properly kept (only the configure guard missing).

Sources: fa-5090 (gau-nernst); cloudrift 5090 GEMM; NVIDIA profiler/compile docs; CUDA 11.2 device-LTO blog;
RAPIDS RMM #2509; CUTLASS 4.3.5/4.4.0 changelogs; vLLM #43507; PyTorch FA3 blog; arXiv:2407.08608; Lambda FA4.
