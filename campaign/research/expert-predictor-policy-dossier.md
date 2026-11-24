# Dossier: expert-routing algorithms & predictive tiering intelligence (qwen4exp: 512 experts, top-10)

Scope: the 5 focus questions for Magma's expert pool (5,138 NVFP4 slots × 2.637 MiB; ~480 slots ≈ 1.24 GiB/token; current 3-horizon target h ≥ 0.876). Research run 2026-11-20; extends, not repeats, the two prior dossiers. (M) measured in fetched source, (D) derived from context's measurements, (E) estimate.

## 1. Deterministic top-10-of-512: measured alternatives to single-thread insertion sort

**Baseline pathology (M).** A one-thread-per-slice top-6-of-64 router select on H200 (DeepSeek-V2-Lite int4 QMoE, eager decode) costs **81.7 µs/layer = 28.4% of all decode GPU time**; a block-parallel rewrite drops it to **9.73 µs (8.4×)**, byte-identical to a CPU oracle via integer-keyed compare + lowest-index tie-break (onnx-genai commit 1660cff658393545cabb2b53714dbe5b4a6fb257, PR #1317, 2026-08-18; byte-identical lock test vs CPU oracle). At n ≤ 512 the cost is launch/serialization dominated: the indictment of insertion sort is measured.

**Alternatives at our scale:**
- **Radix select** (llama.cpp PR #28671, open): RTX 3090 / CUDA 12 / CUB 2.0.1, k=16: at n=4096 a segmented radix *sort* still wins, **24.1 µs vs 38.9 µs** for radix-select; radix-select wins from n=8192 (worst 1.42×, median 2.25×, up to 11× at 16 rows; e2e 5.1→0.25 ms/token, +13–18% decode). Threshold measured, not guessed; at n=512 radix is strictly worse; CCCL ≥ 3.2 ships DeviceTopK instead.
- **Warp/block bitonic** (AMD MI300X, ROCm 7.0, M): radix carries fixed cost — histogram + LDS atomics — independent of k; bitonic scales with k. Measured crossover heuristic n + k·log2(k) ≥ 3n → for n=512, k=10: 545 < 1536 ⇒ bitonic territory; AdaptiveTopK beats radix at small k and matches it at k=2048 (rocm.blogs.amd.com adaptive-topk).
- **RadiK** (ACM DOI 10.1145/3650200.3656596; arXiv 2501.14336, M): hierarchical radix; ≤2.5× non-batch / 4.8× batch over prior art, advantage grows at k>2048 and huge n. Wrong tool at n=512, k=10.
- **Bucket select**: the onnx-genai block kernel (per-warp partial lists → block merge) is this family; 9.73 µs/layer incl. dispatch is its measured reference.

**Recommendation (E).** Fuse selection into the router kernel: 16 warps × 32 lanes, per-thread private top-10 registers → warp bitonic merge → 160-candidate block merge, lowest-index tie-break, one CTA, zero extra launches. Marginal ~1–2 µs after the 2560×512 router GEMV (5.24 MB ≈ 3 µs at 1.79 TB/s) (D/E); keep the CPU-oracle byte-identical CI test (both references do, incl. 112 tie cases).

## 2. Expert temporal locality: measured reuse distributions

- **Power-law exists at neuron level** (M): PowerInfer (arXiv 2312.12456): a small hot-neuron subset drives **>80% of activations**; online predictors **>93% runtime accuracy**. Neurons, not experts — an upper bound on skew.
- **Decoder-only MoEs are NOT power-law** (M): ProMoE (arXiv 2410.22134, Fig. 5): encoder–decoder MoEs (Switch, NLLB) show power-law expert frequency and high LRU hits; **modern decoder-only MoEs are near-uniform**, attributed to load-balancing router design. Consequence: static PowerInfer-style pinning underperforms on qwen4exp; frequency must be windowed.
- **Temporal repetition within a generation** (M): MoE-Infinity (arXiv 2401.14361): at small batch, "only a limited number of experts are repetitively activated" — trace-driven selection-rate caching gives 3.1–16.7× latency wins (prior dossier, verified). Strongest in-geometry datapoint: vLLM RFC #38256 + tinyserve production: **97–100% cache hit from temporal prediction alone** at 238 GPU slots (GPT-OSS-20B, 30 t/s @ RTX PRO 2000 8 GB); LFRU score = freq/age "preserves hub experts".
- **Prediction + correction** (M): ExpertFlow (arXiv 2410.17954): cache hit **up to 91.96%, +27.65% avg over LRU**; §5.2: inter-layer routing-pattern reuse (Lina-style, prev-layer route → next-layer prediction) **fails above 32 experts/layer** — do not build fixed cross-layer pattern tables at 512.
- **Hot-set persistence across requests** (M): llama.cpp PR #30594 (open): offline-profiled routing distributions pre-populate the GPU cache (--moe-cache-profile); hybrid mode computes misses on CPU in parallel instead of PCIe-stalling: RTX 4060 Laptop 8 GB / Qwen3.5-35B-A3B Q6_K / 4 GB cache 22.5→37.7 t/s (1.7×); L4 5 GB cache 33.5→45.9 (1.4×); when the cache already covers most of a generation, plain -fit wins (51.6 vs 50.5) — residency/hot-set trade auditable via --moe-cache-audit JSONL.

**Synthesis.** No reuse-distance histogram exists for 512-expert decoder-only models — a literature gap. Working model: per-generation hubs recur at 0.92–1.0 (97–100% and 91.96% brackets); global skew is low, hubs drift between generations; the pool holds ~10.7 token-working-sets, so capacity is not the constraint — ordering is.

## 3. Predictor ceilings

| System | Input | Model | Horizon | Measured |
|---|---|---|---|---|
| Pre-gated MoE 2308.12066 (ISCA'24) | current hidden state | trained pre-gate replacing the router | L+1 | 4.2× perf; −23% peak VRAM vs GPU-only; comparable accuracy |
| ProMoE 2410.22134 | intermediate hidden states (adjacent-layer cosine ≈90%) | small learned predictor, no MoE retrain | L+1, stride prefetch | GoodPred metric (accuracy×efficiency); 2.20× prefill (max 3.21×), 2.07× decode (max 5.02×) |
| ExpertFlow 2410.17954 (DAC'26) | input tokens directly | transformer classifier, active/idle per expert, all layers in one pass | full generation path + runtime corrector | hit ≤91.96%, +27.65% vs LRU; −75.4% VRAM avg; 2–10× |
| vLLM #38256 / tinyserve | recent routing history | LFRU + cross-layer prediction | within generation | 97–100% hit; 30 t/s @ 8 GB GPU |
| Mira 2609.38090 (2026) | intermediate results | learned predictor | L+2 | 5.71× avg throughput, 11.71× TTFT, 3.84× beam, memory-constrained GPU |

**Ceiling synthesis.** Magma has a zero-training predictor the literature skips — their pipelines cannot reorder compute: **evaluate layer L+1's real router the moment layer L's output hidden exists** (≈3 µs GEMV vs ~0.5–1 ms/layer) ⇒ exact top-10 for L+1, h = 1.0 by construction, with ~0.5–1 ms prefetch lead ≫ the 41–53 µs/slot RAM pull (D). Horizon-2 requires a learned predictor, expected below the temporal band; ProMoE's stride recipe is the validated template. Pre-gated MoE's trained pre-gate modifies model weights — rejected, model is fixed.

## 4. Cache policy for expert streams

- **LFRU** (vLLM #38256): score = freq/age; production-validated on exactly this object type.
- **SLRU**: Magma today; MoE-Infinity's upgrade = windowed selection-rate weighting.
- **TinyLFU/W-TinyLFU** (arXiv 1512.00727, fetched): frequency-gated admission for skewed access; W-TinyLFU "equal or better hit-ratios than other state-of-the-art … the only scheme to obtain such good results on all traces." Caveat: it assumes skew; ProMoE's uniform-decoder-only finding means admission must key on **prediction rank**, not frequency alone.
- **Belady-min approximations**: PARROT (arXiv 2006.16239, ISCA'21): imitation learning of Belady's MIN, **>20% hit-rate gains** over SOTA on memory-intensive workloads. GL-Cache (FAST'23): group-level learning, **+7% avg / +25% P90 hit ratio vs LRB, 228× throughput vs LRB** (snippet-verified). LeCaR (FAST'18, DOI 10.1145/3277332.3277335): LRU+LFU regret bandit. CACHEUS (FAST'21, fetched): **SR-LRU** (scan-resistant) + **CR-LFU** (churn-resistant) experts beside ARC/LIRS/LFU; 17,766 experiments over 329 workloads; the tested line that adapts best to mixed LRU-friendly/LFU-friendly/scan/churn primitives.
- **Verdict.** With exact horizon-1 knowledge Magma implements Belady *exactly* for one layer: keep(L+1 needs) ∪ (h=2 predicted) ∪ hubs, evict worst-scored elsewhere. Learned Belady approximations earn their complexity only beyond horizon-1. For prefill bursts, CACHEUS's churn-resistance rule (don't let a one-shot flood evict hubs) is the right fallback; blind LRU is measurably the wrong default.

## 5. Admission control: pull-from-RAM vs pull-from-NVMe

Costs (D): slot 2.637 MiB; RAM pull 41–53 µs (50–64 GB/s measured H2D); NVMe slot ≈200–400 µs + io_uring latency; RAM tier ~6–10× cheaper and avoids polluting PLE page cache.
- **Fiddler** (arXiv 2402.07033, fetched): per-expert decision argmin over max(Σ n_input·lat_cpu, Σ (1−on_gpu)·lat_transfer); transfer latency constant in token count, activation traffic negligible. Confirms: no CPU execution at batch 1; always prefer the cheapest mover — RAM over NVMe when the victim cache holds the slot.
- **HOBBIT** (arXiv 2411.01433, fetched): when loading is the bottleneck, fetch **lower precision** — an int4 copy loads up to **4× faster** than fp16; token-level dynamic loading + layer-level adaptive prefetch + sequence-level multidimensional cache; up to 9.93× decode. Mapped: under pool contention, admit an uncertain demand as low-precision rather than evicting a hub.
- **Admission rule.** Priority: (1) exact L+1 top-10 (free, h=1.0), (2) h=2 predicted, (3) protected hubs; everything else is served from the RAM victim cache without eviction. TinyLFU-gate cold one-shots (admit only if recent frequency ≥ victim's). NVMe only on total RAM absence AND predicted-reuse×saved-transfer > read cost; issue as grouped shard reads (io_uring pattern, prior dossier).

## 6. Recommended stack & expected hit rates

- **P1 early-router exact lookahead** (§3): fuse router-of-L+1 + top-10 + DMA issue; h = 1.0 within horizon, zero training.
- **P2 horizon-2 learned predictor** (ProMoE/Mira recipe): enable only after P1 lands; measure GoodPred-style accuracy before arming.
- **C1 policy**: LFRU (freq/age) + protected hub partition + TinyLFU-rank admission gate; hub partition seeded from a prefill-profile pass (PR #30594 pattern), drift-audited.
- **A1 admission**: RAM-first; NVMe cold-path only; HOBBIT low-precision fallback under contention.
- **Expected blended slot-traffic hit: projected 0.93–0.97** vs today's 0.876 target — bracketed by ExpertFlow's 91.96% (weaker predictor, no exact lookahead) and tinyserve's 97–100% (temporal-only). Validate by recording 48×512 one-hot bitmaps per token and fitting your own reuse-distance histogram — that replaces projection with measurement and fills the §2 gap.

## Transfer to Magma
1. Kill the single-thread insertion sort; ship the fused deterministic block top-10 (lowest-index ties) with a CPU-oracle byte-identical CI test (both 2026 references do exactly this). (§1)
2. Add early-router evaluation: issue L+1 router GEMV immediately after layer L; DMA missing L+1 slots during L+1 attention. Biggest unexploited lever; ~0.5–1 ms lead per layer. (§3, §6)
3. Evolve SLRU → LFRU + hub partition + TinyLFU rank gate; log victims and hub drift (audit JSONL). (§4)
4. Log per-token expert bitmaps; publish-grade your own reuse-distance stats — nothing exists for 512-expert decoder-only models. (§2)
5. NVMe stays cold-path-only; RAM victim cache is the default miss tier; low-precision slot fallback under contention. (§5)
6. Revisit horizon-2 learned predictor only after P1's measured h is in hand.

## Source table
| Source | ID | Key number | Verification |
|---|---|---|---|
| onnx-genai top-k | commit 1660cff6…, PR #1317 | 81.7→9.73 µs/layer, 8.4×; 28.4%→4.5% GPU | patch |
| llama.cpp radix-select | PR #28671 (open) | 24.1 vs 38.9 µs @4096 cols; ≥8192 radix wins ≤11× | API |
| RadiK | ACM 10.1145/3650200.3656596; arXiv 2501.14336 | ≤2.5× non-batch, 4.8× batch; wins at k>2048 | HTML |
| AMD AdaptiveTopK | rocm.blogs.amd.com adaptive-topk | n+k·log2k ≥ 3n crossover; MI300X, ROCm 7.0 | HTML |
| PowerInfer | arXiv 2312.12456 | >80% hot-neuron activations; predictors >93% | HTML |
| ProMoE | arXiv 2410.22134 (v3) | decoder-only uniform skew; cosine ≈90%; 2.20×/2.07× | HTML |
| ExpertFlow | arXiv 2410.17954 | ≤91.96% hit, +27.65% vs LRU; >32 experts breaks patterns | HTML |
| Pre-gated MoE | arXiv 2308.12066 (ISCA'24) | 4.2×; −23% peak VRAM | HTML |
| vLLM RFC / tinyserve | issue #38256, PR #37190 (open) | 97–100% hit; 30 t/s @ 8 GB; 238 slots | API |
| Mira | arXiv 2609.38090 | L+2 predictor; 5.71×/11.71×/3.84× | HTML |
| MoE-Infinity | arXiv 2401.14361 | temporal locality at small batch (3.1–16.7× prior dossier) | HTML |
| PR #30594 | llama.cpp (open) | 1.7×/1.4× hybrid; profile prefill; -fit crossover | API |
| Fiddler | arXiv 2402.07033 | cost model: transfer const vs CPU linear | HTML |
| HOBBIT | arXiv 2411.01433 | int4 4× faster load; 9.93× | HTML |
| TinyLFU/W-TinyLFU | arXiv 1512.00727 | W-TinyLFU best-or-equal on all traces | abs |
| PARROT | arXiv 2006.16239 (ISCA'21) | >20% hit-rate gains (Belady imitation) | snippet |
| GL-Cache | FAST'23 (junchengyang.com/pdl.cmu.edu) | +7% avg/+25% P90 vs LRB; 228× throughput | snippet |
| LeCaR | FAST'18, DOI 10.1145/3277332.3277335 | LRU+LFU regret minimization | snippet |
| CACHEUS | FAST'21 (usenix.org abstract) | SR-LRU + CR-LFU; 329 workloads/17,766 sims | abs |

Verification: "fetched" = retrieved and read this session (full or targeted excerpts); "snippet" = title/snippet-verified only. SonicMoE (2512.14080) Fig. 22 top-k table was not retrievable; its numbers are omitted. CACHEUS miss-reduction counts absent from the fetched abstract, so only its mechanism is cited. Some ACM/v6 HTML paths 404'd; cited versions fetched instead.
