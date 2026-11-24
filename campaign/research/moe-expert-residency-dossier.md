# Dossier: Expert residency & offload for single-GPU MoE decode (batch-1) — v2 (trimmed; supersedes v1)

## 0. Calibration
Per token: 48 layers x top-10 x 0.62 MiB = 298 MiB; at 50-64 GB/s that is 4.7-5.9 ms => 169-215 tok/s at 0% cache hit. Your 227 tok/s (4.4 ms/token) implies the SLRU already yields ~10-30% effective hits or overlap beyond one lane. You sit near the bandwidth roofline: remaining wins come from traffic reduction and latency hiding, not DMA tuning. Cross-check: vLLM issue #54593 tabulates the same geometry class (top-10 of 512, 48 layers): 62.11 GiB experts on host, 1.22 GiB read per token — same ~1:51 ratio as yours (0.29 of 15.2 GiB). Reference on your exact hardware: llama.cpp b7951 on RTX 5090 + 9950X3D + 96 GB runs Qwen3-Coder-Next-80B Q4_K_S at ~26 tok/s with `-ot ".ffn_.*_exps.=CPU"`. Your engine is ~8.7x faster than the reference open-source config on identical silicon.

## 1. Fiddler — arXiv 2402.07033 (ICLR 2025)
Dense/attention on GPU; routed experts *executed on CPU*; only activations (KBs) cross PCIe, never weights; a cost model picks per-layer between CPU-execution and weight-transfer. Measured: 1.26x single-batch, 1.30x long prefill, 11.57x beam search on 22B and 90GB+ MoEs. Transfer to you: poor at batch 1. CPU must still read ~0.29 GiB/token of expert weights; dual-channel AM5 (~90-110 GB/s) floors that at ~3 ms with perfect pipelining — no better than your 4.4 ms — and reality is worse: llama.cpp CPU-only decodes Qwen3-Next-80B-A3B Q4_0 at 11.76 tok/s on a 48-core EPYC 9454P with 12-channel DDR5-4800 (PR #16095 comment). Fiddler's winning regime is beam search/batch, where activation traffic amortizes.

## 2. Pre-gated MoE — arXiv 2308.12066 (ISCA 2024)
A trained auxiliary pre-gating function predicts next-timestep expert selection from the current hidden state; the system prefetches token t+1's experts while computing t. Measured: -23% peak GPU memory vs GPU-only oracle at near-oracle latency. Transfer: temporal prefetch is your biggest unexploited lever, but the training-free descendant fits better — ProMoE (arXiv 2410.22134) trains only a small MLP predictor on intermediate-layer activations (no MoE retraining) and prefetches proactively on consumer GPUs. For you: predict layer L+1's top-k from layer L's router logits, launch DMA immediately; dual-lane concurrency then hides much of the 5-6 ms cold path.

## 3. Cache / offload systems
- MoE-Infinity — arXiv 2401.14361. Activation-trace-guided expert cache (expert-selection-rate drives replacement + prefetch). 3.1-16.7x per-token latency vs vLLM/Ollama/DeepSpeed/BrainStorm on DeepSeek/Mixtral, batch 1. Direct SLRU upgrade: weight recency by recent activation frequency.
- EdgeMoE — arXiv 2308.14352. Experts on external storage; two-stage gate preloading + expert-wise bitwidth adaptation. 1.19-2.77x vs dynamic-load baselines; 2.1-38.6x memory savings; 3.2-3.9x decode inflation without compute-I/O pipelining. Validates two-stage gating.
- AdapMoE — arXiv 2408.10284 (ASPLOS 2025). Sensitivity-based dynamic expert count + prefetch + cache. 25% fewer activated experts, 1.35x speedup, no accuracy loss. The only cited lever that directly cuts your 298 MiB/token: drop 2-3 marginal experts of your top-10 (validate perplexity).
- HoME — NOT FOUND as an offloading system. arXiv 2408.05430 "HoME" is Kuaishou multi-task learning (naming collision); 2410.19240 is unrelated (electroaerodynamic thrusters). If your notes meant hot/cold-expert work: HOBBIT (arXiv 2411.01433; mixed-precision expert offload, RTX 4090 evals) and "Taming Latency-Memory Trade-off in MoE-Based LLM Serving via Fine-Grained Expert Offloading" (arXiv 2502.05370) are the real citations.
- ExpertFlow — arXiv 2410.17954 (DAC 2026). Transformer routing-path predictor (one forward pass predicts all layers' usage), token scheduler grouping same-route tokens, predictive cache with runtime correction. -93.72% GPU memory, up to 10x vs offloading baselines. Predictor transfers; the token scheduler does not (serving-batch device).
- PowerInfer — arXiv 2312.12456 (EuroSys 2024); PowerInfer-2 — arXiv 2406.06282. Power-law activation: hot set preloaded on GPU, cold computed on CPU; P2 adds neuron-level pipelining + phone NVM storage — Mixtral 47B at 11.68 tok/s on a smartphone (22x SOTA). The hot/cold locality result is your SLRU's basis; neuron-granularity splitting is not worth it at 0.62 MiB experts.
- KTransformers — GitHub kvcache-ai/ktransformers; paper ACM 10.1145/3731569.3764843. Dense on GPU, routed experts in DRAM via AVX/AMX-optimized CPU kernels with overlap: DeepSeek-R1/V3 671B in 24 GB VRAM, up to 16 tok/s decode (README), prefill ~286 tok/s (project-reported). SGLang integration: issue #11425, PR #20516. Same verdict as Fiddler: needs server-class memory channels; AM5 cannot feed it.

## 4. llama.cpp --override-tensor on RTX 5090/4090-class (Qwen3-Next 80B-A3B)
- Model support: PR #16095, merged 2025-10-19, commit 2fdbf16eb (release b7186). -ot introduced by PR #11397.
- Measured, your exact rig (5090 + 9950X3D + 96 GB, b7951, Q4_K_S, `-ot ".ffn_.*_exps.=CPU"`, q8_0 KV): ~26 tok/s; the same flag hurts smaller MoEs.
- Full-VRAM ceiling, same model+GPU: Q2_K_L (27.23 GiB, Vulkan) tg128 74.65 → 94.36 tok/s after PR #17996; 3x3090 Q6_K 43.33 tok/s.
- CPU-only: 11.76 tok/s on EPYC 9454P — Qwen3-Next runs ~5x slower than Qwen3-30B-A3B (63.10 tok/s) there; the 512-expert gather is the tax.
- Discussion #20528 requests expert-ID-granular splitting and reports it blocked: all experts of a layer are fused in one tensor, so -ot cannot implement your fine-grained residency.
- Nov-2026 upstream convergence toward your design: PR #27861 (GPU-resident LRU for host-offloaded experts); PR #30594 (hybrid expert cache, profile-prefill, "zero-upload inference" — 0 PCIe bytes during generation at 100% profile hits); b12191 / PR #30966 (missed experts on CPU, --moe-cache-cpu-share); PR #31469 (--moe-direct-io: O_DIRECT streaming, VRAM slot pool, RAM victim cache). Self-reported outlier: prabanta-dev/janas claims 37 tok/s Qwen3-Next-80B on a 32 GB laptop vs 2.8 for llama.cpp (CPU repacking + speculative decoding).

## 5. vLLM / SGLang on expert offload
- vLLM #54593: --cpu-offload-gb evicts in declaration order — offloads the hottest 4.80 GiB (dense, fully read every token), keeps 62 GiB experts on host. Framework offload is placement-blind.
- vLLM #38256 (RFC ExpertWeightProvider, PR #37190): pinned host pool, fixed-size GPU cache, LFRU eviction, cross-layer prediction; production datapoint: RTX PRO 2000 8 GB, GPT-OSS-20B MXFP4: 30 tok/s stable single-stream decode, 97-100% cache hit from temporal prediction alone, 160x vs HF device_map=auto. Prior art: #29941 (merged), #33869/#31938 (closed, too large).
- vLLM PR #49321 (experimental): W4A16 cold-expert CPU offload via in-graph HIP gather.
- SGLang #49334: --cpu-offload-gb crashes FlashInfer moe_route_warp when the 512-expert fp32 correction_bias lands on CPU (sigmoid routing needs it CUDA-resident) — SGLang's offload is not expert-aware; KTransformers route (#11425, PR #20516) is the active effort.

## 6. Quantization vs the expert working set
You are already at ~4.1-4.25 effective bpw. llama.cpp PR #19769 measured Qwen3-4B: NVFP4 5.0 bpw, PPL 15.25 (+8.0% vs FP16) vs Q4_1 5.15 bpw, PPL 15.81 (+12.0%), KLD 0.110 vs 0.112, tg128 15.2 vs 14.7 tok/s on a 5090 — NVFP4 is already on the Q4 size/quality Pareto front; PR #28572 pipelines NVFP4 MMQ tile loads with cp.async+TMA on Blackwell (+14% prefill). Below 4 bits with quality: PagedMoE (arXiv 2611.03173) stores each expert once as a prefix-compatible bitplane tree — a cached 2-bit expert extends to 4-bit by fetching only missing bitplanes; 2.5-bit average routed precision matches GPTQ-INT4 on Qwen3.5-397B-A17B; 8.89 tok/s SSD-resident (1.33x/1.56x over independent-precision configs). HOBBIT (2411.01433) picks expert bitwidth dynamically at offload time.

## 7. SSD-resident experts (2025-2026)
- SSD-LLaMA — arXiv 2609.18110, implemented in llama.cpp. Three tiers (SSD/RAM/VRAM); persistent io_uring worker submits bounded batches of independent reads (timeout-flush when a batch doesn't fill); each completion immediately issues async H2D from a pinned-buffer pool into a pre-reserved VRAM slot; balanced CPU-GPU hybrid; every selected expert executed, none pruned. Measured: prefill 1.52-4.19x, decode 2.10-15.58x over baselines; >1 tok/s decode on >1T-param Kimi-K2.7-Code with a single RTX 5090 and <=32 GB RAM. Key for you: on a 5090 + fast SSD, giving experts to the CPU was a measured pessimization (513.95 -> 529.52-573.73 ms) — weights-to-GPU beats compute-on-CPU in your machine class.
- GPU-initiated reads: BaM — arXiv 2203.04910 / SIGMOD 2023 — NVMe queues mapped to the GPU, SSD → VRAM over PCIe P2P, no host bounce buffer; GPUDirect Storage (cuFile) is the vendor equivalent. No 2025-26 MoE system measured here uses GDS for experts yet; io_uring + pinning is the proven datapath. llama.cpp PR #31469 adds O_DIRECT expert streaming with a VRAM slot pool and RAM victim cache.

## Ranked actions for your engine
1. Temporal prefetch: ProMoE-style next-layer prediction from router logits (no retrain); DMA layer L+1's experts during layer L compute. Evidence: 97-100% hit rates from temporal prediction alone (vLLM #38256); Pre-gated MoE is the trained variant.
2. Replacement policy: activation-frequency weighting in the SLRU (MoE-Infinity) plus pinning the measured hot set in spare VRAM (PowerInfer; llama.cpp #30594 zero-upload decode).
3. Cut traffic: AdapMoE-style sensitivity pruning (-25% experts/token, no measured accuracy loss) and/or PagedMoE prefix-precision (2.5-bit ≈ INT4 quality).
4. Do not move expert compute to CPU (Fiddler/KTransformers): AM5 bandwidth, the 11.76 tok/s EPYC datapoint, and SSD-LLaMA's pessimization measurement all say weights-to-GPU is correct for 227+ tok/s.
5. NVMe tier only if you outgrow 96 GB: io_uring bounded-batch worker + pinned pool + slot-mapped async H2D (SSD-LLaMA), or BaM/GDS for GPU-initiated reads.

## Source table
| Source | ID | What | Key number |
|---|---|---|---|
| Fiddler | arXiv 2402.07033 (ICLR'25) | CPU expert exec | 1.26x/1.30x/11.57x |
| Pre-gated MoE | arXiv 2308.12066 (ISCA'24) | trained pre-gating | -23% peak VRAM |
| ProMoE | arXiv 2410.22134 | predictor prefetch | consumer GPUs |
| MoE-Infinity | arXiv 2401.14361 | trace-guided cache | 3.1-16.7x latency |
| EdgeMoE | arXiv 2308.14352 | 2-stage gate, bitwidth | 1.19-2.77x |
| AdapMoE | arXiv 2408.10284 (ASPLOS'25) | expert pruning | -25% experts, 1.35x |
| HOBBIT | arXiv 2411.01433 | mixed-precision offload | 4090 evals |
| ExpertFlow | arXiv 2410.17954 (DAC'26) | routing-path predictor | -93.72% VRAM, <=10x |
| PowerInfer / -2 | arXiv 2312.12456 / 2406.06282 | hot/cold locality | 47B @ 11.68 t/s phone |
| KTransformers | GH kvcache-ai; ACM 10.1145/3731569.3764843 | CPU-expert kernels | 671B @ <=16 t/s, 24 GB |
| llama.cpp Qwen3-Next | PR #16095, commit 2fdbf16eb | model support | merged 2025-10-19 |
| llama.cpp -ot | PR #11397 | tensor→buffer override | — |
| -ot measured | Reddit 1qx2teh (b7951) | your exact rig | ~26 t/s |
| Full-VRAM ref | PR #17996 | Q2_K_L Vulkan 5090 | 74.65→94.36 t/s |
| CPU-only ref | PR #16095 comment | EPYC 9454P | 11.76 t/s |
| Expert-split limit | Discussion #20528 | fused expert tensor | blocks -ot split |
| Expert-cache lineage | PRs #27861, #30594, #30966, #31469† | GPU LRU / zero-upload / O_DIRECT | 0 B PCIe @ 100% hits |
| vLLM eviction bug | Issue #54593 | declaration-order eviction | 1.22 GiB/token |
| vLLM RFC | Issue #38256, PR #37190 | ExpertWeightProvider, LFRU | 30 t/s, 97-100% hit |
| vLLM cold-expert | PR #49321† | W4A16 HIP gather | experimental |
| SGLang crash | Issue #49334 | bias must be CUDA-resident | 512-expert routing |
| SGLang + KT | #11425, PR #20516† | KT integration | — |
| NVFP4 (llama.cpp) | PRs #19769, #28572 | quant + MMQ pipelining | +8% vs +12% PPL; +14% pp |
| SSD-LLaMA | arXiv 2609.18110 | io_uring 3-tier | >1 t/s @ 1T params |
| PagedMoE | arXiv 2611.03173 | prefix-bitplane precision | 2.5-bit ≈ INT4 |
| BaM | arXiv 2203.04910; SIGMOD'23 | GPU-initiated NVMe | — |
| HoME clarification | arXiv 2408.05430 | naming collision, NOT offload | — |

Verification: all arXiv IDs fetched and abstracts read directly; GitHub #11397, #16095, #17996, #19769, #20528, #54593, #38256, #49334 fetched directly; † = title/snippet-verified only; Reddit/janas are self-reported. Saved at research/moe-expert-residency-dossier.md.