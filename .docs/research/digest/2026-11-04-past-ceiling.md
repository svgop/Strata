---
type: digest
id: past-ceiling-2026-11-04
created: 2026-11-04
status: evidence
topic: THE CEILING QUESTION — decode-throughput formulations BEYOND "fewer bytes" for the bandwidth-bound CPU+GPU MoE box: tokens-per-byte (speculative verify amortization), batch composition theory, CPU-GPU split solvers, OS-level DRAM scheduling, phase formulations, and a beyond-bandwidth claim audit (2025-2026, verified against primary sources)
scope: read-only research. No repo files modified besides this digest. Every claim carries its URL. Sibling digests 2026-11-04-moe-constrained-hardware.md (cache/prefetch/kernels: KTransformers, Fiddler, MoE-Infinity, AdapMoE/HybriMoE, ExpertFlow, SiDA-MoE, llama.cpp RFC) and 2026-11-04-hfpapers-specverify.md (SpecPilot) already own those surfaces — this digest does not re-derive them.
---

# Past-ceiling: formulations beyond fewer-bytes for a 49.5 GB/s lane (2026-11-04)

The law under test: MoE decode whose expert-lane reads host DRAM at 93-100% of a measured 49.5 GB/s ceiling (dual-channel DDR5, Zen5 16C/32T, RTX 5070 12 GB, 184 GB RAM); serial attn→moe per sequence irreducible; GPU term 7.2 ms vs lane 7.3 ms; ~4 ms slack (3 host syncs); windows 9-16 ms; conc lane 260-315 t/s at 6-way vs 215 solo. Only (a) fewer bytes, (b) shorter serial steps, (c) more sequences in flight have ever moved it.

This digest searches for a FOURTH term and re-prices the existing three. Headline: the literature's one genuinely new term is **tokens-per-byte** (speculative-verify amortization of expert reads) — and its measured viability on OUR shape is conditional on fine-grained routing, with one adversarial counter-result at our exact shape.

---

## 1. The fourth term: tokens-per-byte (speculative verify amortization) — Area 4

### 1a. The production-verified theory (Cohere, Apr 2026)

URL: https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding (published 2026-04-21, validated on Cohere production MoE via vLLM, k=8/N=128, K=3 draft).

- Arithmetic intensity of a MoE layer over T tokens-per-pass: `AI = T(k+S)/(N+S)`; sparsity keeps AI low → the layer stays bandwidth-bound at batch sizes where a dense model is compute-bound. Three regimes for spec decode: low BS (verification adds unique expert loads), moderate BS sweet spot (decode and verify both load nearly all experts — verify tokens nearly free), high BS (compute-bound, gains vanish).
- **The batch-composition closed form** (answers "when does 2× concurrency cost <2× bandwidth" at the expert-read level): expected unique experts for B token-positions under uniform routing `= N[1-(1-k/N)^B]`; under popularity skew `= Σ_e[1-(1-p_e)^B]`; empirical adds temporal correlation on top.
- Measured routing structure: most-popular expert up to 10× uniform; adjacent-token expert overlap 38% at step 1 (3.0 of 8 experts shared vs 11.8% independence baseline), still 2-4× independence at step 4; mid-network layers strongest (~0.50 overlap), early ~0.32, final ~0.25. Verifying 4 tokens loads ~2.5× top-k (20.36 unique empirical vs 25.4 independence) — 20-31% below independence. Stable across 13 Spec-Bench categories and 7 languages (0.377-0.385 step-1 overlap everywhere) — structural, not workload-specific.
- **The Amdahl split that prices verify on any engine**: `T_t(K+1)/T_t(1) = f·unique(K+1)/k + (1-f)` where f = routed-expert fraction of forward time. Their model: f=0.30 → verify ratio 1.46×, AL=2.73 accepted → implied 1.87-2.18× speedup at BS=1; draft cost ~15% of one target step (measured 14.3%). At BS=1, fixed-overhead amortization (kernel launches 2-10 µs) gives extra gain beyond expert math — `T_t(1)/T_t(4)=0.80` measured.

**Term attacked:** new term (d) — tokens per byte read. Not fewer bytes: the window's byte count grows by unique(K+1)/k, but tokens out grow by AL. On our lane the window is bytes-bound, so expected solo-sequence gain ≈ `AL / unique-ratio` ≈ 2.73/1.46 ≈ **1.6-1.9×** if our f (expert-lane fraction of window) and routing correlation match theirs. Our engine already banks part of this (triage-t2 W3.1: conc 4.93 rows × 0.685 acceptance = 3.38 tok/window, p/q verify +24-34% queued) — the PORTION THAT IS NEW is the routing-correlation math as the sizing rule, and everything in 1c/1d below.

### 1b. THE CORRECTION for our routing class (arXiv 2610.38981, Oct 2026)

URL: https://arxiv.org/html/2610.38981v1 ("Why Speculative Decoding Works Better Than Predicted"). 10 models, 30B-1T, 8-512 experts, H100 SXM/B200, vLLM v1, K=3.

- The Cohere/MoESD concave speedup-vs-BS curve (peak at moderate BS) holds ONLY for coarse-routing large-expert models (Mixtral-8x7B, Qwen2-57B class). **Fine-grained sparse MoEs show non-concave curves that often peak at or near batch size 1.**
- Second mechanism beyond expert bytes: cache-resident expert shards. Qwen3-30B FP8: 4.5 MB experts fit 50 MB L2 → L2 hit rate 5.72% (AR) → 27.79% (SD); MoE kernels hit 45-49% of peak HBM BW while attention hits 22-25%.
- Temporal overlap 35.1% (step 1) → 26.4% (step 3); unique experts 21-34% below independence; their TE model `TE ≈ (C_dense + k·C_expert)/(C_dense + unique_sd·C_expert)` systematically underestimates measured target efficiency (gaps up to +0.37).
- **Adversarial read for us:** Engine B (DS-class 4-of-384, fine-grained) sits in the peak-at-BS≈1 class. Two consequences: (i) spec-verify gains are LARGEST at low concurrency — a per-sequence verify-depth push outperforms a concurrency push; (ii) our conc lane's sublinear-but-positive scaling (260-315 at 6-way vs 215 solo, i.e., only 1.21-1.47× aggregate on 6× the sequences) is consistent with fine-grained routing offering near-zero cross-sequence expert reuse: uniform-model union over 6 positions × 4-of-384 = 384·(1-(380/384)^6) ≈ 23.4 unique vs 24 selected — the formulas in 1a predict conc amortization is real only when k/N is large or popularity skew is extreme. Re-price any 12-way conc push against this before spending a build.

### 1c. MoE-Spec: expert budgeting at verification (arXiv 2602.16052, Feb 2026)

URLs: https://arxiv.org/abs/2602.16052, https://arxiv.org/html/2602.16052v1, OpenReview https://openreview.net/forum?id=BnqJcRBl9D. B. McDanel et al., cited-by-11; referenced by vLLM issue "Adaptive Speculation Depth for MoE Models" (https://github.com/vllm-project/vllm/issues/46295).

- Mechanism: aggregate router probabilities over all M draft-tree tokens per layer (`s_i = Σ_t g_i(h_t)`), keep top-B experts per layer, load ONLY those; out-of-budget tokens handled by truncation (missing experts contribute zero) or substitution (top-k restricted to shortlist). Training-free, verified against full routing. Decouples speculation depth from memory cost: verify expert loads grow with B (budget), not with M (depth).
- Measured: A100 80GB (Mixtral on 2 GPUs), FP16: 10-30% higher throughput than EAGLE-3 baselines; Mixtral MBPP 2.3× vs EAGLE's 1.7×; selection overhead 2-3% of generation time; acceptance loss only 1.4% average.
- Honest boundary: paper targets HBM-resident experts ("experts fit in GPU memory"), cites offloading as orthogonal. **The aggregation+budget trick itself ports to our lane**: on the CPU lane the budget caps the window's byte growth during multi-row verify — exactly the lever that keeps 1a's verify ratio near 1.1-1.2× instead of 1.46× at 6-way conc. Expected on top of 1a: their +10-30% over the spec baseline.

### 1d. SubSpec: draft entirely on the GPU, verify on the full engine (NeurIPS 2025)

URLs: https://arxiv.org/html/2509.18344v2; NeurIPS 2025 proceedings pp. 60515-60538 (https://proceedings.neurips.cc/paper_files/paper/2025/hash/4e32ca54ae8a3a0ea4c1fd3663f73b0f-Abstract-Conference.html); placement description in vLLM issue https://github.com/vllm-project/vllm/issues/39427.

- Mechanism: for single-GPU + offloaded-portion setups, build the DRAFT model by replacing each CPU-offloaded layer with a low-bit quantized substitute resident on the GPU; verify against the full offloaded model → output distribution provably unchanged (lossless), training-free (substitutes are quantizations of the very layers they replace).
- Why it is a formulation, not a quant: the bytes stay on the lane; what moves is WHERE DRAFTING RUNS. The lane never sees draft steps — it sees only K+1-position verify rows. On our box: the 12 GB GPU (GDN/attn + resident tier + KV) hosts the substitutes; the draft cost lands on the GPU term, whose 7.2 ms is currently latency/launch-bound at BS≈1 (triage M7: decode latency-bound), not bytes-bound.
- Verdict hook: if our current draft path (W3 family) spends any lane time drafting, this placement moves it off. Expected +15-25% solo-sequence on top of verify amortization; losslessness means no MMLU gate needed beyond the standard parity suite.

### 1e. The offload-spec family (swept, verified at abstract level)

- **Specter** (arXiv 2511.14102, ASPLOS'27 submission): quantized draft keeps "high-fidelity expert activation patterns" → schedules expert transfers AHEAD of verification (lookahead prefetch); 1.50× average, up to 1.88× TPOT over "state-of-the-art offloading system"; DeepSeek-V2-Lite artifacts on HF. https://huggingface.co/papers/2511.14102 — prefetch-side companion to 1a; the prefetch family's verdict for us is already owned by 2026-11-04-moe-constrained-hardware.md §3 (MoE-Infinity: TRANSFERS with conditions); Specter's delta is that the draft itself is the prefetch oracle.
- **SpecMoEOff** (arXiv 2508.21706): spec decode enlarges per-expert workload and re-schedules GPU-CPU transfer; up to 2.5× decode throughput vs SOTA MoE offloading. https://arxiv.org/abs/2508.21706, https://huggingface.co/papers/2508.21706.
- **SP-MoE** (arXiv 2510.10302): uses the draft stage's latency to prefetch the verify stage's experts — the spec+prefetch composition. https://arxiv.org/html/2510.10302v2.
- **SpecMoE** (ACM 2026, Bang et al.): self-assisted speculative decoding + sparse expert offloading. https://dl.acm.org/doi/pdf/10.1145/3770743.3803936. **S2-MoE** (arXiv 2608.15018): routing-aware self-speculative decoding for edge. https://arxiv.org/html/2608.15018v1. **"Unveil Speculative Decoding's Potential for Accelerating MoE Inference"** (arXiv 2505.19645, likely the "MoESD" prediction Cohere validates): https://arxiv.org/html/2505.19645v1.

### 1f. ADVERSARIAL COUNTER-RESULT — spec decode can HURT at our shape

Allen Kuo audit (Medium, 2026): https://allenkuo.medium.com/when-speculative-decoding-helps-local-llms-and-when-it-doesnt-5c41dd804e4b — RTX PRO 6000 Blackwell 96 GB (400 W cap) + 9950X3D + 192 GB RAM, Qwen 3.6 35B-A3B Q4_K_M, `--n-cpu-moe` sweep.

- Without spec decode: 28 → 201 tok/s across the n-cpu-moe sweep (6 GiB partial → full GPU residency); 28-45 tok/s below ~16 GiB free. Community-reported with drafter on a 6 GB card: **17 → 11 tok/s at 65% acceptance — WORSE than baseline.**
- His mechanism: each draft position routes to a different 8-of-256; if experts are offloaded, every position's misses fetch serially — "acceptance rate alone cannot pay for that." His own win formula's condition 4 ("base decode not too slow" i.e. bandwidth-bound) means "the MTP corner gets narrower the smaller your card is."
- **Why this does NOT kill 1a for our engine:** the counter-result is from llama.cpp-class serial PCIe fetch per position. Our lane reads DRAM on CPU cores with the union amortization of 1a (unique-ratio ≤1.5×, not per-position fetch), and the engine's verify already batches rows in one window (conc 4.93 rows/window — triage-t2). The falsifier to log per window: `bytes(window)/tokens(window)` — if verify grows it past ~1.5× solo, the Kuo regime has arrived; expert budgeting (1c) is the counter-lever.

**Area 4 verdict roll-up:** tokens-per-byte is REAL and production-verified, composes with the queued W3.1 row, but its fine-grained-MoE form is peak-at-low-BS. TRANSFERS (1.6-1.9× solo; less at 6-way unless budgeted).

---

## 2. Batch composition theory — Area 1

### 2a. The union law and its corollaries (see 1a formulas)

Cross-SEQUENCE overlap at the same step is popularity-skew-only (different users, no temporal correlation): independence baseline applies — `Σ_e[1-(1-p_e)^B]` with p_e from routing logs. Same-sequence adjacent positions add the 2-4× temporal correlation. **Batch composition rule for our conc lane:** aggregate window bytes ≈ `D + E·Σ_layers unique(B·k positions)`; the marginal bandwidth cost of sequence B+1 ≈ E·Σ_layers Σ_e p_e(1-p_e)^{Bk}-class term → strongly concave in B for small k/N (Engine B), fatter for coarse routing. This is the first formulation in the literature that answers "when does 2× concurrency cost <2× bandwidth" from ROUTING STATISTICS alone — and it is measurable on our own logs for free (we have per-token router outputs; Cohere used vLLM's `enable_return_routed_experts` API to do exactly this: https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding).

### 2b. HRM/MILP: the policy formulation (MoE-Lightning, ASPLOS'25)

URL: https://pschafhalter.com/papers/2025-asplos-moe-lightning.pdf (UC Berkeley — Cao, Liu, Grigg, Schafhalter, Liu, Sheng, Gonzalez, Zaharia, Stoica; DOI 10.1145/3669940.3707267).

- Hierarchical Roofline Model: per-level compute roofs + cross-level bandwidth roofs; per-layer latency `T = max(comm_cpu_to_gpu, T_cpu, T_gpu)`; balance point `B_i·I_ix = B_{j,i}·I_jx`. Policy = 6-tuple (batch N, micro-batch μ, attention-on-GPU, FFN-on-GPU, fraction weights on GPU r_w, fraction KV on GPU r_c); MILP solve <1 min offline, zero runtime overhead.
- Measured: up to 10.3× vs FlexGen (unpadded) / 3.5× (padded) for Mixtral 8x7B on T4 16 GB + 24-core Xeon 192 GB; CPU attention kernel (MKL GQA) 3-4× faster than KV-transfer over PCIe — the insight that moved attention softmax to CPU.
- **Term attacked:** (c) via optimal (N, μ) and (b) via overlap ordering (CGOPipe). For our box the balance point is where we already are (GPU 7.2 ms ≈ lane 7.3 ms), so direct t/s upside is small; the TRANSFER is the formulation itself: it re-solves our (batch rows, pcie-frac, VRAM-tier size) tuning automatically after every re-bake/quant change instead of manual sweeps. Their CGOPipe 2-micro-batch lookahead also externally corroborates the queued W4.1/W4.2 deferred-combine + foresight class (their A1/A2/B1/B2 schedule, Algorithm 1).

### 2c. MoE-Lens: the KV-capacity term binds concurrency (HPDC'26)

URL: https://dl.acm.org/doi/full/10.1145/3806645.3807594 ("Towards the Hardware Limit of High-Throughput MoE LLM Inference" — MoE-Lens).

- Claim verified: three-stage model (theoretical bound → workload/resource-aware predictor at 94% accuracy → system). Core insight: **CPU memory capacity for KV cache is a primary bottleneck, not just PCIe** — MoE-Lightning's HRM missed it. Hardware: dual Xeon Platinum 8380 (750 GB DDR4, ~150 GB/s/socket), A40 masked to 24-32 GB; validated on RTX 3090 + EPYC 7742 and A100 + EPYC 7542. Mixtral-8x7B/8x22B, batch 20-25k.
- Measured: avg 4.6× over MoE-Lightning (up to 25.5× RAG); ~90% of GPU compute capacity approached at large KV. Mechanisms: VSLPipe phase split (CPU-only/GPU-only phases), all decode attention on CPU with hand-optimized AVX512 (3.1× vs auto-vectorized), contiguous data mover with fine-grained packets, next-layer weight prefetch. NO quantization.
- **For us:** their CPU-attention move costs ~60 GB/s of host DRAM — impossible on our 49.5 GB/s lane (SKIP CPU attention). The porting piece is the CAPACITY term: our (c)-term ceiling is set by 12 GB GPU KV; past 6-way, KV pressure forces either eviction or host-RAM KV, and host KV either crosses PCIe (their D-mover) or moves attention onto the lane. Their Fig/Eq-3 formalizes the concurrency ceiling our conc lane will hit. WATCH — this is the pricing model for how far (c) can go before it self-defeats.

---

## 3. CPU-GPU co-execution / split solvers — Area 2

### 3a. Apex: per-layer EMA cost-model solver (arXiv 2610.23006, Oct 2026)

URL: https://www.alphaxiv.org/abs/2610.23006, https://arxiv.org/html/2610.23006v1 (built on KTransformers + SGLang).

- Formulation: per layer, missed experts sorted by token-load; transfer count p splits them GPU vs CPU; solve `p* = argmin_{0≤p≤m} max{T̂ᶜ(p), T̂ᵍ(p)}` — linear scan, EMA-updated service rates: `T̂ᶜ(p) = Σ_{e∈C(p)} n_{i,e}/vᶜ`, `T̂ᵍ(p) = p/vᵀ + Σ_{e∈(A∩R)∪G(p)} n_{i,e}/vᵍ`, `v^(s) = (1-β)v^(s-1) + β·w/t`.
- Measured: RTX A6000 48 GB + Xeon 8358P + 512 GB, PCIe 4.0 ×16: up to 39.5% decode gain, 51.9% prefill; beats strongest reproduced baseline (Fiddler-style static popularity placement) by 31.1% decode on average at every pool size. Models: Qwen3.5-35B-A3B, Qwen3-Next-80B-A3B, GLM-4.5-Air, BF16.
- **Adversarial calibration:** their static baseline was massively imbalanced (CPU path 3.66 ms vs GPU 0.45 ms) — 31% over THAT does not translate to our box, whose static --pcie-frac split is already at the 7.2-vs-7.3 ms balance. Residual upside = adaptation to per-window routing variance a constant cannot track. WATCH→small TRANSFER: expected +2-5% conc-lane; the EMA machinery is a day of work and composes with W4.2 foresight.

### 3b. APEX: asynchronous overlap with deferred sync (arXiv 2506.03296 v4, Jan 2026)

URL: https://arxiv.org/html/2506.03296v4 (dense models — NOT MoE; disambiguation vs 3a required, name collision in literature).

- Mechanism: unified GPU batch for all linear ops; CPU attention on offloaded requests runs concurrently; result sync DEFERRED one full GPU iteration window ("CPU attention of layer i syncs just before GPU needs it for layer i's post-attention in the next cycle"); GPU never stalls on CPU — retries next iteration; profiling-based inequality chooses strategy per iteration.
- Measured: T4 + 2×Xeon 6130 / A10 + 2×Xeon 6342, Llama-2-7B/3.1-8B: up to +72% over NEO; CPU attention 3031 µs vs GPU 170 µs at batch 4 — CPU is <10% of GPU throughput on dense, which is why overlap works there.
- **Verdict: SKIP as mechanism (dense; CPU attention would eat our lane's 49.5 GB/s), KEEP as corroborating evidence** for the queued W4.1 deferred-combine/W4.2 foresight class: the deferred-sync-to-next-window pattern is the same move, published independently, with the same premise — hide host syncs under the GPU term. Our law's falsification of naive overlap at throughput-bound balance is consistent: their overlap works because their GPU sits at 16.5% utilization in the baseline (MoE-Lens' measurement of MoE-Lightning decode), ours does not.

### 3c. Static splits (KTransformers, llama.cpp --n-cpu-moe, Fiddler)

Already owned by 2026-11-04-moe-constrained-hardware.md §§1,4,5. New facts this pass only: community sweep numbers — `--n-cpu-moe` 6 GiB→full-GPU on a 96 GB card moves Qwen 3.6 35B-A3B 28→201 tok/s (https://allenkuo.medium.com/when-speculative-decoding-helps-local-llms-and-when-it-doesnt-5c41dd804e4b); HF guide to the current flag semantics (--cpu-moe vs --n-cpu-moe N, 32-token default batch behavior): https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide (Jan 2026); flag mechanics explainer: https://partnerinai.com/blogs/llamacpp-n-cpu-moe-performance-why-speedups-happen (May 2026). None of these move our law — they are the same term-(a)/(c) economics our tiers already implement.

---

## 4. The lane at the OS level: DRAM scheduling — Area 3

### 4a. Evidence of absence (2025-2026)

Three targeted searches (queries on record: "DRAM row buffer aware scheduling software optimization inference GEMV 2025 2026 arXiv"; "page coloring DDR5 memory bandwidth thread channel affinity Zen5 2025 2026"; "DRAM row buffer aware software scheduling page coloring 2025 2026 paper bandwidth inference GEMV") found NO 2025-2026 result combining row-buffer-aware software scheduling or page coloring with CPU inference bandwidth. The live toolbox is 2010-2012: page-coloring bank partitioning (PACT'12, Liu et al.), page coloring + XOR bank mapping (NPC'10, Mi et al.), micro-pages (ASPLOS'10, Sudan et al.), row-buffer-aware scheduling (MICRO'11, Kaseridis), RBL-vs-BLP balance (HPCA'12, Jeong & Erez) — all targeting INTERFERENCE ISOLATION among co-running apps, not single-stream bandwidth. 2025-2026 hits are hardware-side only: RACAM in-DRAM reuse (https://arxiv.org/html/2512.09304v1), row-buffer-stable scheduling for DRAM-PIM (ResearchGate Oct 2026 listing), GEMV row-locality placement for accelerator/PIM hardware (arXiv 2403.20297). NUMA-aware bandwidth scaling (https://arxiv.org/html/2506.09198v2) needs multi-socket; we are one socket, channels already interleaved by the IMC.
- Zen5 DDR5 behavior base: TechPowerUp DDR5 scaling on Zen 5 (https://www.techpowerup.com/review/ddr5-memory-performance-scaling-with-amd-zen-5); Chips and Cheese 9950X analysis (https://chipsandcheese.com/p/amds-ryzen-9950x-zen-5-on-desktop). Multi-NUMA llama.cpp thread-affinity discussion (not our shape): https://github.com/ggml-org/llama.cpp/discussions/19102.

**Verdict: SKIP.** The lane runs at 93-100% of the measured ceiling — software DRAM scheduling's theoretical recoverable mass is ≤7%, and the classical toolbox does not even target that regime. Row-buffer hit-rate manipulation on DDR5/Zen5 has no published inference result to inherit. Named boundary: any future gain here would have to come from the IMC's own scheduling, unreachable from userspace.

### 4b. The X3D question (memory-side caching analogue)

Community consensus verified: X3D L3 helps llama.cpp when the ACTIVE footprint (KV + dense) fits the cache; once weights spill, the part "just streams from RAM" (owner thread: https://www.reddit.com/r/LocalLLaMA/comments/1sa6l7v/fellow_9950x3d_owners_how_do_you_get_the_most_out); Phoronix measured a llama.cpp win for small models on 9950X3D (https://www.phoronix.com/review/amd-ryzen-9-9950x3d-linux/9); OpenBenchmarking carries the 9950X3D llama.cpp row (https://openbenchmarking.org/performance/test/pts/llama-cpp/db257a7755adbe206b4709bcdd5c5fdb23ba90fa). GB-scale expert tensors do not fit 64 MB L3 → hardware analogue SKIP.

**BUT the software analogue has a verified mechanism** (from 1b, GPU-side): expert SHARDS that fit last-level cache get 5.72%→27.79% hit rates under verify (https://arxiv.org/html/2610.38981v1). On Zen5, the analogue is: repeated expert rows across a conc/verify window hitting the 2×32 MB L3 instead of DRAM. This does NOT violate the law — it removes DRAM bytes (sub-term of (a)) without requantizing. It is also FREE TO MEASURE: `perf stat -e l3_hit,l3_miss` on the expert-read loop during a 6-row conc window. If unique-expert overlap within a window is real (2a formulas + our routing logs say it is for popular experts), expected mass is single-digit % — but at zero build cost for the measurement. Falsifier: L3 hit-rate delta between solo and 6-way windows ≈ 0 → close the line.

---

## 5. Beyond-bandwidth claims audit — Area 5

| claim | verified arithmetic | verdict |
|---|---|---|
| LMCache "10× MoE inference" (Apr 2026) https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x | cross-GPU expert DISPATCH + KV sharing across nodes — the 10× is vs a no-expert-sharing distributed baseline; not a single-box result | NOT APPLICABLE (multi-GPU) |
| MoE-Lightning "10.3×" https://pschafhalter.com/papers/2025-asplos-moe-lightning.pdf | vs FlexGen/DeepSpeed-ZeroI on T4 — baseline-relative; their own HRM CONFIRMS bandwidth-boundedness (balance-point Eq. 11); mechanism is utilization recovery (16.5% GPU util in baseline), not roofline-breaking | real but NOT past-ceiling |
| MoE-Lens "4.6× avg, ~90% of hardware limit" https://dl.acm.org/doi/full/10.1145/3806645.3807594 | approach-to-bound claim, not beyond-bound; bound itself is min(capacity term, GPU compute); 150 GB/s/socket DRAM and dual-socket KV capacity do our box no good | real, wrong hardware shape; KV-capacity insight ports (2c) |
| DeaMoE "up to 50.9% fewer weights loaded/step; 1.02-1.33× TPOT e2e on A40" https://arxiv.org/html/2608.14385v1 | department-shared backbone matrices cut per-step expert bytes — this is FEWER BYTES (term a) via retraining-from-scratch, not a new term; microbench peaks 1.97-2.0× are per-kernel, e2e 1.02-1.33× | BYTE-CUTTER IN DISGUISE → re-bake queue family (staffed elsewhere); also requires retraining — not a rebake of existing weights |
| "bandwidth spec" class (Specter 1.50×, SpecMoEOff 2.5×) — §1e | both are tokens-per-byte (verify amortization) or prefetch scheduling measured against offloading baselines; consistent with roofline | real, correctly framed as amortization |
| roofline norm baseline | decode tok/s ≈ BW / bytes-per-token, e.g. https://turingpi.com/llm-tokens-per-second-memory-bandwidth, https://fergusfinn.com/blog/inference-arithmetic, https://intuitionlabs.ai/articles/gpu-inference-performance-prefill-decode-batching | no credible late-2026 single-box claim past tok/s-per-GB/s found. The law's "nothing beyond bandwidth-norm" stance survives this sweep. |

Also swept, one line: CommitMoE fallback-free offload (https://researchgate.net/publication/402629996, details unverified — GPU-memory-constraint focus); FineMoE (EuroSys'26, fine-grained offloading, unverified); PreScope prefetch pipeline modeling (https://arxiv.org/html/2509.23638v1) and PopFetcher (USENIX ATC'25, https://www.usenix.org/system/files/atc25-zhang-junyi.pdf) — prefetch family, verdict owned by the constrained-hardware digest §3; "Improving LLM Inferencing Under Speculative Decoding" (arXiv 2511.12031 — bandwidth management + spec co-tuning, https://arxiv.org/abs/2511.12031, WATCH as the co-tuning reference); adaptive spec depth adoption: vLLM issue 46295 (https://github.com/vllm-project/vllm/issues/46295); community MTP-on-8GB trap (https://github.com/ggml-org/llama.cpp/discussions/25357); spec-decode consumer-GPU measurements 1.72-2.03× dense (https://inventivehq.com/blog/llama-cpp-speculative-decoding-consumer-gpu); LM Studio 1.5-3× (https://lmstudio.ai/blog/lmstudio-v0.3.10); negative community result on MoE clusters (https://dev.to/defilan/i-tested-speculative-decoding-on-my-home-gpu-cluster-heres-why-it-didnt-help-3ej6); expert-prediction offload experiment with measured numbers (https://www.reddit.com/r/LocalLLaMA/comments/1uybm8y/).

---

## 6. Verdict table (mechanism → number+hardware → law term → verdict)

| # | mechanism | number + hardware | term | verdict |
|---|---|---|---|---|
| 1 | tokens-per-byte via multi-row verify (1a/1b) | 1.87-2.18× at BS=1, AL=2.73, verify ratio 1.46× (Cohere prod MoE, vLLM); unique-expert union formulas | NEW term (d) | **TRANSFERS** — 1.6-1.9× solo-seq; composes with W3.1; fine-grained caveat: peak at BS≈1 |
| 2 | verify-time expert budgeting (MoE-Spec, 1c) | +10-30% over EAGLE-3, acceptance -1.4% (A100 80GB, FP16) | (d) guard | **TRANSFERS** — holds verify window ≤~1.2× at 6-way; 2-3% overhead |
| 3 | GPU-resident quantized-substitute drafting (SubSpec, 1d) | lossless, training-free (NeurIPS 2025); targets exactly single-GPU + offloaded portion | (b)+(d) placement | **TRANSFERS** — draft off the lane; +15-25% if draft currently touches lane |
| 4 | fine-grained non-concave correction (2610.38981, 1b) | 10 models H100/B200; concave sweet spot only for coarse routing; L2 hits 5.7→27.8% when expert fits cache | (c)/(d) interaction | **WATCH** — redirects effort: verify depth > more sequences for Engine-B-class routing; falsify 12-way conc push first |
| 5 | HRM/MILP balance-point policy model (MoE-Lightning, 2b) | 10.3× vs FlexGen (T4+Xeon), MILP <1 min, 94%-class accuracy sibling (MoE-Lens) | (b)/(c) pricing | **TRANSFERS (small direct)** — +0-5% direct; kills manual re-tuning after each re-bake |
| 6 | per-layer EMA split solver (Apex, 3a) | +39.5% decode, +31.1% over static (A6000+Xeon 8358P) but vs imbalanced baseline | (b) | WATCH→small — +2-5% over tuned static pcie-frac; composes with W4.2 |
| 7 | L3-resident expert shards under verify/conc (4b, mechanism from 1b) | GPU-L2 analogue 5.72→27.79% (Qwen3-30B FP8, H100) | (a)-adjacent (DRAM bytes removed, no requant) | WATCH — free measurement first: perf counters on expert loop, solo vs 6-way |
| 8 | KV-capacity concurrency ceiling (MoE-Lens, 2c) | 4.6× avg over MoE-Lightning (dual Xeon 8380 + A40); KV 70-210 GB | (c) ceiling | WATCH — pricing model for how far 6-way can scale |
| 9 | software DRAM scheduling / page coloring / row-buffer on Zen5 DDR5 (4a) | no 2025-26 inference result exists; toolbox is 2010-2012 interference isolation | none | **SKIP** — ≤7% theoretical mass, wrong regime, no inherable result |
| 10 | X3D-class L3 for expert weights (4b) | community: helps only when active set fits cache (Reddit/Phoronix) | none | **SKIP** for weights; superseded by #7 measurement |
| 11 | async CPU-attention overlap (APEX 2506.03296, 3b) | dense-only; +72% over NEO (T4); CPU attn 3031 µs vs GPU 170 µs | (b) | **SKIP as mechanism** (eats lane BW); corroborates W4.1/W4.2 already queued |
| 12 | DeaMoE department backbones (5) | e2e 1.02-1.33× TPOT A40; -50.9% bytes/step; retrain from scratch | (a) | **SKIP here** — byte-cutter, re-bake queue family |
| 13 | LMCache 10× / MoE-Lightning 10.3× / MoE-Lens 4.6× as past-ceiling claims (5) | see §5 table | — | all baseline-relative or multi-GPU — the bandwidth-norm norm SURVIVES |

## 7. Composition note and falsifiers

The three TRANSFERS compose on one axis: verify depth D with expert budget B (1c), drafted on GPU (1d), sized by the union formulas from our own routing logs (1a). Expected combined solo-sequence: window grows ≤1.2× (budgeted verify) + draft on GPU (~free) → tokens/window ≈ AL ≈ 2-3 → **~1.7-2.2× t/s on a lane whose bytes did not fall** — the first formulation family in this sweep that moves the box past term (a). The law is not broken: the window's byte count still sets the clock; the term being attacked is tokens-per-clock. Falsifiers, cheapest first: (i) log bytes(window)/tokens(window) per verify run — past ~1.5× solo means the Kuo regime (1f) arrived; (ii) perf-counter L3 hit-rate delta solo vs 6-way on the expert loop (#7); (iii) unique-expert count per window from router logs vs the Σ_e[1-(1-p_e)^B] prediction before sizing any budget B.
