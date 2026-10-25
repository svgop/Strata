# T50-R3 MoE Offloading/Tiering/Quantization Dossier

**Researcher:** R3 | **Date:** 2026-10-25 | **Scope:** MoE expert offloading/tiering/prefetching + MoE quantization, priced against our T50 box (RTX 5090 32GB, 9950X3D, 184GB RAM, PCIe 5.0 x16, Qwen3.8-Flash-Next IQ3_S three-tier expert store: 10,623 VRAM slots / 20.6 GiB, 36 GiB pinned RAM, hit 96.6-97.1%, GPU window 9-10.4 ms flat, solo 223/234 t/s, 6-conc 338.5 t/s).

## 0. Our window anatomy - derived constants (used by every proposal below)

| Constant | Value | Derivation |
|---|---|---|
| Slot size | **~2.08 MB** | 20.6 GiB / 10,623 slots |
| Total expert population | ~20.7-24.0k experts | 43-50 GB / 2.08 MB - consistent with Qwen3-Next-80B family geometry (48 MoE layers x 512 routed experts = 24,576); 1:50 activation ratio + native MTP (vLLM blog, S2) |
| Routed slot-touches | **~144/token, ~288/window** | top-3 x 48 layers; window carries ~2 tokens |
| Tokens/window | **~2.16** | 1000/223 = 4.484 ms/token; 9.7 ms avg window / 4.484 |
| Miss traffic/window | **~18.9 MB** | 288 x 3.15% = 9.1 slots x 2.08 MB |
| Per-1% miss bytes | **6.0 MB** | 288 x 0.01 = 2.88 slots x 2.08 MB |
| **N = ms per 1% hit-rate per window** | **0.1-0.4 ms (central ~0.25 ms ~ 2.6% of window)** | PCIe path 0.095 ms + CPU-pool path ~0.08 ms; top-down: mean->best window spread 8.97->8.55 ms over 0.5-pt hit band => <=0.84 ms/pt upper bound |
| Gain from hit 96.85->99.5 | **+0.27-1.06 ms/window -> +2.8-11% solo (central +7% ~ 223->239); x4-6 at 6-conc -> +10-18% (338.5->372-400)** | 2.65 pts x N |
| Expert HBM reads | ~300 MB/token ~ 0.17 ms ~ 3.5-4% of window | 144 x 2.08 MB at ~1.79 TB/s => **bandwidth is a second-order lever on this box; slot count (hit-rate) dominates by ~5-10x** |

## 1. Per-source findings

### Papers

**1. Pre-gated MoE** - arXiv:2308.12066, ISCA'24. Auxiliary pre-gate predicts layer i+1 routing from layer i input; one-hot gates + JIT prefetch; needs training. 1.7x avg MoE-block time. **Grade A.** We need the training-free analog (ProMoE/Fate).

**2. AdapMoE** - arXiv:2408.10284, ICCAD'24. Adaptive sensitivity gating (dynamic top-k), gating-score-aware prefetch, edge. 1.35x latency vs top-1. **Grade A.** Dynamic top-k is quality-coupled; our router credit/budget arms occupy this lane.

**3. Fiddler** - arXiv:2402.07033, ICLR'25. Compute cache-miss experts on CPU; 1.26x single-batch vs Mixtral-offloading/DeepSpeed. **Grade A.** Already falsified here (net-negative at 2% CPU traffic); externally re-confirmed dead for our hit rate.

**4. Mixtral-offloading (CacheMoE)** - arXiv:2312.17238. Per-layer LRU + speculative expert loading by applying next layer's gate to current hidden states (residual-stream approximation, up to 10 layers ahead); HQQ 2-bit experts = 2.6 actual bpw; contiguous pinned host buffers + async prefetch. 2-2.5x vs naive offload. **Grade A.** Two gems: (a) 2.6-bpw datum anchors our tail re-quant sizing; (b) hidden-state gate-lookahead is training-free, bit-exact.

**5. MoE-Infinity** - arXiv:2401.14361, OSDI'24. Sequence-level activation tracing (GAV) -> prefetch + replacement. 3.1-16.7x over DeepSpeed-class. **Grade B.** GAV locality ~ what our learned profile + pins encode statically.

**6. ProMoE** - arXiv:2410.22134. Training-free proactive caching: per-layer ~2M-param MLP predictor memorizes hidden-state -> gate correlation, predicts layer i+1 experts (~200 us CPU, hidden); chunked prefetch, early preemption, reordered inference; model-equivalent, bit-exact. Predictor top-k accuracy **84.7%** (vs 58.3% token-ID lookup, 66.9% skip-predictor). RTX 4090 + PCIe 4.0 x8: decode **2.07x avg (up to 5.02x)** vs open baselines, 1.34x vs static/LRU; decode critical-path loading 29.2% vs LRU 50.9%. **Grade A.** Closest blueprint for P2.

**7. SP-MoE** - arXiv:2510.10302 (queued in our tree). SD-aware offloading: speculative expert prefetch via draft->target correspondence, cutoff-layer policy (analytical latency model caps prefetch depth), async threads + batched I/O. 1.07-3.5x TPOT vs SOTA. **Grade B.** Those gains are for stalling systems; at our 3.15% miss + fast DMA the ceiling is N x 2.65 pts = same as cheaper mechanisms => **adopt only the cutoff-layer (depth-1) policy; skip draft coupling.**

**8. HOBBIT** - arXiv:2411.01433. Mixed-precision offloading: dynamically serve cache-miss experts from low-precision copies; layer-replacement preloading (mixture output insensitive to individual expert versions); llama.cpp-based, code unreleased. Up to 9.93x vs stalled baselines. **Grade A mechanism / B details.** Blueprint for P1/P4.

**9. MoQE** - arXiv:2310.02410. Weight-only quant of expert layers down to 2-bit, attention high precision; experts robust; 79.6% size cut, 1.24x A100. **Grade B.** Foundational license for expert-only down-quant.

**10. QMoE** - arXiv:2310.16795, MLSys'24. Sub-1-bit MoE compression, 0.8 bpw, 20x. **Grade A.** Practical floor for our quality gate is ~2.5-2.8 bpw tail, not sub-1.

**11. MxMoE** - arXiv:2505.05799, ICML'25. Automated linear-block mixed-precision bit allocation jointly optimizing accuracy + measured latency. 2.4x lower PPL than GPTQ at 2.25-bit avg; up to 3.4x speedup. **Grade A.** The recipe for our re-quant pass.

**12. MoE-SpeQ** - arXiv:2511.14102. Quantized draft model proactively prefetches experts the target will verify with; <=2.34x vs SOTA offloading, >90% acceptance. **Grade B.** Same pricing verdict as SP-MoE.

**13. Fate** - arXiv:2502.12224. Cross-layer expert-activation prediction (training-free stats). Up to 4.5x / 1.9x vs LoD/EAP. **Grade B.** Alternative predictor for P2.

**14. Speculating Experts** - arXiv:2603.19289 (Mar 2026). Speculatively execute next-layer experts during transfer overlap. Up to 14% TPOT reduction. **Grade C.** External sanity check matching our N x 2.65 central band.

**15. MoE-Beyond** (Aug 2025). Learned activation prediction at edge; hit 17%->72% at 10% budget. **Grade C.** Steep curves at small budgets; we are on the flat tail at 97% => histogram falsifier governs.

Verified/excluded: SE-MoE (2205.10034) training system, N/A; ExpertFlow (2410.17954) redundant with our adaptive tier; OD-MoE (2512.03927) opposite regime; KMoE not found under that name; DyMoE (2603.19172) consistent with P1, no independent numbers.

### Production sources

- **P-A. vLLM Qwen3-Next blog** (2025-09-11). GDN hybrid, 1:50 activation, 3B active of 80B, native MTP, full CUDA graphs, FLA Triton kernels + hybrid KV manager. **Grade A.** Confirms family geometry + ~2.16 tokens/window anatomy.
- **P-B. NVIDIA NVFP4 blog** (Jun 2025). E2M1 + per-16 E4M3 fractional scales + FP32 per-tensor (~4.5 bits); <=1% degradation vs FP8 PTQ on DeepSeek-R1 across 7 evals; shipped TRT-LLM/vLLM. **Grade A.** Anchors the up-quant quality leg of P1b.
- **P-C. vLLM FP8 docs**: production default FP8_DYNAMIC RTN for all Linears. **Grade A.**
- **P-D. llama.cpp discussion #21419** (Apr 2026): HOBBIT-style dynamic expert load proposed, no code, KTransformers = working reference. **Grade A.** Our implementation is ours alone; no upstream to reuse.
- **P-E. Ecosystem**: llama.cpp --n-cpu-moe mainstream; ik_llama.cpp fine-grained offload; exllamav3 CPU-offload reports beating llama.cpp. **Grade B.**

## 2. Proposals, costed (ranked by expected value)

### P1 - Frequency-tail mixed-bit re-quant: slot multiplication - RANK 1
Keep hottest experts at IQ3_S (learned-profile ordering); re-quant tail to IQ2_XS-class (~2.5-2.8 effective bpw; Mixtral-offloading's "2-bit" = 2.6 actual). Average ~2.7-2.9 bpw vs 3.44 => slot 2.08 -> ~1.55-1.65 MB => **VRAM tier 10,623 -> ~13,400-14,000 slots (+27-32%)**; pinned RAM tier same +30% coverage; miss bytes shrink 22%.
Gain: hit 96.85 -> 99.0-99.6 => 2.15-2.75 pts x N = **+2.3-11% solo (central ~+7% => ~237-239); 6-conc +9-16% (369-393)**. Bandwidth leg honest: 3.44->2.7 bpw saves ~0.06 ms/token = +1.5% only (bandwidth second-order on 5090). Up-quant leg (nvfp4/W4A4 ~4.5 bpw) = quality variant, -2.5-8% perf, NOT perf lever.
Quality tax: not bit-exact; concentrated on cold experts (MoQE/MxMoE/HOBBIT all support); MMLU floor 93.3 re-validated. Matches the tree-priced +3-4% re-quant; multiplicative with every tier.
Cheapest falsifier: sorted expert-activation histogram from the learned profile; probability mass beyond current coverage <1.5% => gains halve, kill before any re-quant. Then 4 layers only -> MMLU delta + window.
Port: expert_source.cpp (second quant-class tier + per-class slot table), native_moe.cu (IQ2 MMQ path - ggml kernels exist), routing untouched.

### P2 - Router/hidden-state look-ahead prefetch with depth-1 cutoff (ProMoE + Fate + SP-MoE cutoff) - RANK 2
Per-layer ~2M-param MLP predictor (84.7% top-k, ~200 us, hidden) or hidden-state gate eval predicts layer i+1 experts during layer i; predicted misses prefetched RAM->VRAM via the existing --pcie-frac DMA queue; cutoff depth 1 (deeper would evict the learned tier). Bit-exact, timing only.
Gain: ceiling = N x 2.65 pts = **+2.8-11% solo central +7% (~239)**, x4-6 ms at 6-conc. If P1 lands first, residual misses 0.4-1.0% => marginal +1-4% => sequence AFTER P1.
Risk: predictors must train on post-adoption router outputs (resident bias 0.75 + credit arms), else accuracy collapses; PCIe contention with adaptive swaps.
Cheapest falsifier: predictor top-k recall on 20k live tokens; >=80% GO, <70% kill.
Port: verify routing tap + expert_source.cpp prefetch-queue priority; no native_moe.cu change. Cheapest big lever.

### P3 - CPU-pool retirement + PCIe-share retune - RANK 3
Once hits >=99%, pool traffic <1% (already net-negative at 2%). Retire 15 workers; repurpose as async DMA/prefetch engines; raise --pcie-frac to 0.6-0.8 during swap bursts. Bit-exact, ~free.
Gain: bounded by mean->best spread: **223->228-234 (+2-5%)**, similar conc; derivation: 0.2-0.4 ms of the 0.42 ms spread is miss-path jitter.
Falsifier: A/B pool disabled at today's hit rate; window p99 must tighten.
Port: expert_source.cpp miss-dispatch only. Highest certainty, lowest ceiling.

### P4 - HOBBIT-style 2-bit shadow tier for misses - RANK 4
Pin ~2.0 bpw shadow of the full store (~26 GB; fits ~107 GB of 184) in RAM; miss -> DMA 1.15 MB shadow slot (-45% bytes); adaptive swap replaces with true IQ3_S.
Gain: miss bytes halve => ~0.05-0.2 ms/window => **+1-4% solo**; ->0 if N at 0.1 lower bound.
Quality: non-bit-exact for <=3% of touches until replacement.
Falsifier: P1's histogram doubles here; pin N first via miss-rate sweep (VRAM tier -5/10/15% -> window vs miss rate); N < 0.15 ms/pt => kill.
Port: expert_source.cpp shadow tier + native_moe.cu IQ2 dequant matvec.

### P5 - SP-MoE (queued) pricing verdict - RANK 5
1.07-3.5x TPOT claims apply to stalling systems. Our ceiling from any draft-gated prefetch = N x 2.65 = +2.8-11%, captured more cheaply by P1/P2. **Adopt only the cutoff-layer depth-1 policy inside P2; drop the draft-coupling queue item.**

## 3. Bottom line for the campaign lane

Lane ceiling: hit 97->99.5%+ worth **+3-22%** (central +7-10% solo, +10-18% 6-conc); re-quant slot-multiplication (P1) is the only single change multiplying every tier's coverage AND already priced (+3-4%). P1+P2+P3 central: **223 -> ~245-255 solo (+10-14%), 338.5 -> ~385-400 6-conc** - real but not sufficient alone; other lanes carry the rest.
Kill-fast order: P1 histogram (free) -> P3 A/B (free) -> P2 recall log (cheap) -> P1 4-layer re-quant -> P4 only if N >= 0.15 ms/pt.
Honest negatives: expert-bandwidth reduction second-order on 5090 (~4% of window); Fiddler-class CPU compute stays dead; sub-1-bit has no place above ~2.5 bpw at our MMLU floor.

Sources: [Pre-gated MoE](https://arxiv.org/abs/2308.12066) | [AdapMoE](https://arxiv.org/abs/2408.10284) | [Fiddler](https://arxiv.org/abs/2402.07033) | [Mixtral-offloading](https://huggingface.co/papers/2312.17238) | [MoE-Infinity](https://ar5iv.labs.arxiv.org/html/2401.14361) | [ProMoE](https://arxiv.org/html/2410.22134v3) | [SP-MoE](https://arxiv.org/abs/2510.10302) | [HOBBIT](https://arxiv.org/abs/2411.01433) | [MoQE](https://arxiv.org/abs/2310.02410) | [QMoE](https://arxiv.org/abs/2310.16795) | [MxMoE](https://arxiv.org/abs/2505.05799) | [MoE-SpeQ](https://arxiv.org/abs/2511.14102) | [Fate](https://huggingface.co/papers/2502.12224) | [Speculating Experts](https://arxiv.org/html/2603.19289v1) | [vLLM Qwen3-Next](https://vllm.ai/blog/2025-09-11-qwen3-next) | [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) | [vLLM FP8 docs](https://docs.vllm.ai) | [llama.cpp #21419](https://github.com/ggml-org/llama.cpp/discussions/21419) | [Unsloth Qwen3-Next](https://unsloth.ai/docs/models/tutorials/qwen3-next) | [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
