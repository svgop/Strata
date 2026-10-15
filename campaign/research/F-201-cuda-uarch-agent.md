# Deep brief 1: sm_120 microarchitecture (investigative-researcher, 2026-10-15) — 10 findings

**PREMISE CORRECTION: RTX 5090 L2 = 96 MB (not 128 — that's the full GB202 die). Our 60 MiB carve
= 62.5% of L2; the docs' formula allows ≤72 MB set-aside on a 96 MB part. Re-derive carve math;
query prop.l2CacheSize / persistingL2CacheMaxSize at runtime.**

1. **In-graph node spacing on Blackwell = 1.5-6 µs/node even on datacenter (LMSYS Ling-3.0-flash
Blackwell post, 2026-08-21) — CONDITIONAL, not a fixed floor**: the floor manifests only for tiny
serially-dependent kernels; long kernels let the front-end queue ahead (pegainfer: 47 µs kernels
show near-zero in-graph gaps). Their quantified non-floor contributors: blocking D2H sync
485 µs/step until removed; graph seams 5-15 µs/op dispatch host-bound; ~1 ms cudaGraphLaunch CPU
per step; PDL explains our no-op (launch_dependents releases only the LAUNCH; data-dependent
consumers still fence on producer retirement). 9.5→3 µs ≈ 11.7 ms of our window. PILOT: nsys
`--cuda-graph-trace=node` under locked clocks; A/B node-dense vs node-sparse same-bytes subgraphs.
2. **Consumer-Blackwell DVFS: headless/low-duty parts sit 180 MHz GPU / 405 MHz mem with no
auto-boost (vLLM SM120 thread, 6.5× throughput case!); power-cap drops ~2300→1935 MHz (−16%)**.
Our lock service covers this — but VERIFY it's actually holding during probes (the drift waves
may be governor ramp!). Ops-level, zero code.
3. **L2 access policy windows attach PER GRAPH KERNEL NODE (cudaGraphKernelNodeSetAttribute +
cudaKernelNodeAttributeAccessPolicyWindow) — phase-switch persisting/streaming INSIDE one graph.**
hitRatio is probabilistic; window > set-aside with hitRatio 1.0 measured 11% SLOWER (Lei Mao,
3090); set-aside formula = min(0.75×L2, cap). missProp=Streaming on expert/KV nodes so the ~36 MB
non-carved pool survives for hc/attention. ADOPT API shape, pilot magnitudes.
4. **Hot-vs-cold weight reads: 7 µs vs 11 µs (+57%) on a 2.6 MB gate GEMV when expert traffic
flushes L2 (LMSYS measured)**; GB202 latencies: L2 130 ns, GDDR7 329 ns (Chips and Cheese — both
WORSE than Ada's 107/254). Our hc 4.2 ms line is the analogous stream. If 30-50% of hc/attention
reads could hit L2: ~1.5-3 ms. CTA-tile swizzle alone: 87→94% L2 hit (+17-31% GEMM, LMSYS).
5. **TMA multicast does NOT exist on sm_120** (CCCL target list: SM_90a/100a/103a/107a/110a only —
120 absent); portable clusters ≤8 work but only DSMEM exchange (180-cycle SM-SM, 3.27 TB/s at
cluster=2, DEGRADING at 4). AVOID multicast plans; DSMEM pilot only for GDN state spill >99 KB.
6. **cp.async.bulk (1D unicast) IS on sm_120 — but a bare TMA swap REGRESSED (18 vs 20 TFLOPS on
consumer-Blackwell PTX post); pays only as warp-specialized producer pipeline (20→30 TFLOPS)**.
Paged KV can't use TMA anyway (contiguous descriptors can't express per-page addresses). Our
Marlin-class LDG.128+cp.async pipelines already near-right. Pilot only as producer-warp
double-buffer on the expert stream.
7. **Cache hints (__ldcs/__ldlu/__stwt): NO published Blackwell deltas anywhere — UNKNOWN, cheap
pilot**: mark read-once KV/expert tiles evict-first so they stop evicting the carved state; win is
protective (0-5% direct, up to ~1 ms indirect). One-line SASS change, A/B under nsys dram__bytes.
8. **bf16 router gate + lm_head ≈ +10% of the whole decode step (LMSYS, from fp32)** — small-op
precision beats the big GEMMs at batch-1; audit fp32 scalar/epilogue chains between our 35-44
kernels/layer (each µs × 48 layers ≈ 50 µs/window).
9. **Graph mechanics: cudaGraphUpload pre-warms; UseNodePriority no-op for a serial chain; DEVICE
graph launch removes ~1 ms/step host cudaGraphLaunch cost (LMSYS)** — 3-4%/window for us. ADOPT
upload + evaluate device launch.
10. **Driver/JIT = first-launch phenomena only, not steady-state spacing** (ETC megakernel paper:
5-10 µs eager; pegainfer 4.1 µs CUDA 13/driver 580). Don't chase driver versions; decompose with
finding-1's measurement. NOTE: GB202's L2/GDDR7 latencies are structurally worse than Ada's —
per-gap fixed work is more expensive on this part.

DEAD ENDS: arXiv 2512.02189 (B200-only), "WRENCH" paper doesn't exist, no published __ldcs
deltas, no consumer-Blackwell per-node floor (only our own nsys can answer).
