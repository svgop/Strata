---
type: research-digest
date: 2026-11-04
topic: Quantization formulation for mixed-bpw MoE stores — post-training mixed-precision bit allocation with error bounds, CPU-lane format co-design, expert-heterogeneity allocation, shared/dense component precision, sub-4-bit KV formulations for MLA (2024H2-2026 literature + quant-kernel code)
machine-context: AM5 single-socket 16C/32T, dual-channel DDR5 (49.5 GB/s measured userland ceiling), RTX 5070 12 GB (sm_120), 184 GB RAM. Store A: Qwen-class 512-expert top-10, IQ3_S ~3.x bpw pack (78 GB), CPU lane reads at 93-100% of DRAM ceiling — bytes are THE constraint; gates MMLU-Pro 95.0 + mimic bench. Store B: DeepSeek-class 748B-A16B at mixed 1.75-2.31 bpw, 4-of-384 routing, flat gate mass, MLA attention. Calibration assets: 1,615-window live trace + expert-frequency telemetry.
status: verified
sources: arXiv HTML full texts (fetched), ICML/PMLR + ACL anthology pages, GitHub quant-kernel discussions (fetched) — every claim carries its source inline; search-level-only claims are flagged.
---

# Quantization formulation digest (2026-11-04)

Falsified-boundary respected throughout: REAP, AQLM/VQ, PuzzleMoE, channel-selection (2610.37386), NatScale, replacement-operator law (2610.41366), GPTQ/AWQ classics, QAT-class are known/queued and not re-surfaced. Everything here is post-training (no training loop we lack), CPU-relevant first.

## 1. Calibration frontier: post-training mixed precision on MoE with allocation math (not heuristics)

### 1.1 GEMQ — global expert-level BLP with FIM importance (closest to our trace-driven use case)
[GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](https://arxiv.org/html/2605.23078) (fetched) | [PMLR v306](https://proceedings.mlr.press/v306/deng26t.html).
- Mechanism: binary LP per expert over the whole model — min Σ ΔL̃_ij·x_ij s.t. Σ j·x_ij ≤ B (bit budget), Σ_j x_ij = 1. Importance = Taylor expansion of expected task-loss increase with **diagonal FIM**, evaluated on the perturbation of the **aggregated MoE layer output** (routing-score-weighted) — i.e., router mass is inside the loss term, exactly what our 1,615-window trace can supply. Solves "within seconds". Candidate bits {1,2,3}; router stored FP16 (quantizing a fine-tuned router to 4-bit costs ~nothing: WT2 5.03 vs 5.04).
- Numbers (Mixtral-8×7B, experts group-GPTQ g128, attention fixed 4-bit, 3×H100 quant / 1×H100 infer): 2.5 bpe WT2 5.03 / C4 9.02 / 7-task 65.13 vs uniform 6.10 / 10.35 / 65.49; **5-shot MMLU 2.5 bpe: 63.21 vs uniform 50.51** (FP 70.37). Headline: 87 GB → 16 GB (82% cut) at MMLU −7.2. Qwen1.5-MoE 2.0 bpe: WT2 8.79 vs PMQ 10.47 — gains larger on fine-grained MoE than Mixtral. Uniform 2.0 bpe collapses (WT2 6.29 / avg 55.72) while GEMQ 2.0 holds 6.03 / 59.98.
- Calibration-set trap: PMQ-style allocation **overfits calibration distribution** — Code-calibrated PMQ collapses on Conala (37.42) vs GEMQ-family 66.68 ([EAC-MoE Table 9](https://arxiv.org/html/2508.01625v1), fetched). Our trace is live traffic, not a static corpus — structurally immune to this failure mode.
- Attention ablation at 2.5 bpe: 8-bit attn WT2 4.81 / 17.00 GB; 4-bit 4.97 / 16.37 GB; **2-bit attention collapses** (WT2 31.14).

### 1.2 AlphaQ — calibration-free knapsack allocation on HTSR spectral signal
[AlphaQ (ICLR 2026)](https://arxiv.org/abs/2606.04980) | [arXiv HTML](https://arxiv.org/html/2606.04980v1) (fetched) | [code](https://github.com/Superone77/AlphaQ).
- Mechanism: multiple-choice knapsack ILP per layer-expert: min Σ x_{l,b}·η_{l,b}, η = (α̃/α_l)^γ·Var(W_l)·2^{-2b}, budget Σ N_l·b ≤ B_tot; candidate bits {1,2,3,4}; solves in seconds on CPU (PuLP). Sensitivity signal = **PL_Alpha_Hill** (Hill-estimated power-law exponent of the W^T W eigenvalue-spectrum tail; smaller α = heavier tail = better-trained = more bits) — **no calibration data at all**. Non-expert layers (attn, router) uniform 4-bit; experts group-GPTQ g128; fused Triton unpack+dequant+GEMM kernels; A40 GPU.
- Numbers: **Qwen1.5-MoE 3.5 expert bits ≈ BF16 parity** (avg acc 68.04 vs 67.93; PPL 7.38 vs 7.22) at ~7.6 GB vs 28.94 GB BF16 (>4× compression). Qwen3-30B-A3B 2.5-bit: 69.56 / WT2 9.27 vs PMQ 65.88 / 10.94 vs uniform 68.76 / 10.29. Mixtral 2.5-bit: 70.76 / 5.17 vs uniform 67.86 / 6.16. Layer-wise beats expert-wise (Mixtral 3-bit PPL 4.37 vs 4.72). Speedup vs BF16: 1.69× @2-bit, 1.50× @2.5-bit, 1.31× @3-bit.
- Cost: PL_Alpha_Hill offline pass = 57 ms/module (Qwen3-30B: 549 s total; OLMoE 180 s). Zero GPU requirement for the allocator itself.
- Ablation worth stealing: **global budget > block-wise budgeting** — matches our whole-store single-budget framing.

### 1.3 Theoretical-guarantee allocation: router-norm change → closed-form bit fractions
[Efficient Quantization of MoE with Theoretical Guarantees](https://arxiv.org/html/2604.06515v1) (fetched).
- Mechanism: two-layer MoE model with expert-choice routing; **Theorem 4.4** gives zero-misclassification preservation with high probability under per-column bit conditions b_h ≥ log₂(1+Ω(d√(log(kmd²)/l²·log l))) for task-aligned experts and b_l ≥ log₂(1+(α/(1−α))·...) for rare-feature experts; headline **gap law: rare-feature experts need log₂((1−α)/α) more bits than hot experts**. Practical ranking: router-norm change Λ = ‖w^(T)‖ − ‖w^(0)‖ (for pretrained models, final router norm substitutes; Spearman ρ≈0.999) + MaxVar reordering of ~4% of experts. Allocation itself is sorting + closed-form fractions κ = (b_avg − b_l)/(b_h − b_l) — "requires no GPU and negligible computation".
- Numbers (Mixtral-8×7B, experts GPTQ g128, non-MoE 3-bit, A5000): 8-task avg 70.01 @ **2.75 bits** (uniform 3-bit = 70.85 → **~8% fewer bytes at parity**), 69.04 @ 2.625, 66.72 @ 2.375, 64.26 @ 2.125. Beats PMQ above 2.0 bits; MMLU @2.5: 58.23 vs PMQ 54.05. Mixtral-8×22B: 60.10 @ 2.5, wins vs PMQ at ≤2.25 bits. PMQ's assignment overhead for comparison: 110 GB GPU / 2227 s (Mixtral-8x7B) — this method is free.
- Flag: this theory says **cold-but-critical experts deserve MORE bits** — the direct opposite of the frequency heuristic (§3). Resolution: it keys on router-mass *geometry*, not raw hit count; α (rare-token rate) is small so the gap is real. For a flat-gate store (ours, Store B) the frequency signal is near-uniform anyway and this norm-based ranking is the usable one.

### 1.4 MxMoE — linear-block ILP, activation-frequency-aware, GPU-optimal W4A16-class
[MxMoE (ICML 2025)](https://arxiv.org/html/2505.05799v1) (fetched) | [code](https://github.com/cat538/MxMoE) | [PMLR](https://proceedings.mlr.press/v267/duanmu25a.html).
- Mechanism: sensitivity Δ = ‖Ô−O‖₂ per linear block (gate/up/down), 128-sample offline calibration; objective min L^r·T^(1−r) as ILP under memory budget; expert activation frequencies (vary >10×) feed the runtime-cost model (memory- vs compute-bound regime per expert). Granularity ablation: **linear-block beats expert-level** (DS-V2-Lite PPL 6.11 vs 6.32).
- Numbers: DS-V2-Lite 2.25-bit weight avg: PPL 7.01 vs GPTQ 8.49 (2.4× headline); 3.25-bit: 6.08 vs 6.18 (thin). Mixtral 2.25-bit: 5.63 vs 5.69. All latency on RTX 4090; **no CPU results**; the A<83 arithmetic-intensity switch rule is an RTX-4090 number.
- Verdict signal for us: the per-linear-block granularity result (down_proj/up_proj/gate differ) transfers; the activation-quant co-design does not (we are weight-only, DRAM-bound).

### 1.5 Supporting cast (one line each, fetched unless flagged)
- [MoEQuant (ICML 2025)](https://arxiv.org/abs/2505.03804): expert-balanced calibration sampling (EBSS) + channel-affinity reordering/regrouping (AGQ) — fixes inter-expert variance poisoning of group-wise quants; near-FP at W4 on DeepSeek-V2/V3-class; [code](https://github.com/chenzx921020/MoEQuant). Mechanism composable with any of the allocators above.
- [EAQuant](https://arxiv.org/html/2506.13329v3): expert-aware smoothing aggregation + routing consistency for MoE PTQ (search-level detail only).
- [QuantMoE-Bench / Examining PTQ for MoE](https://arxiv.org/html/2406.08155v1) (fetched): the evidence base for §3/§4; first systematic component-sensitivity benchmark ([code](https://github.com/UNITES-Lab/moe-quantization)).
- [Automated Fine-Grained MoE Quantization (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.1386.pdf): sensitivity-aware per-FFN mixed 2/4-bit; assigns higher precision to down-projection (search-level; PDF fetch unsupported).
- [Mixed-Precision via Global Information Flow](https://arxiv.org/html/2508.04753v1): shows Hessian-based local sensitivity misses cascading error — argues for global criteria (search-level; relevant caveat to §1.1/§1.4).

## 2. Format co-design for a DRAM-bound CPU lane (AVX-2/512, no AMX on AM5)

### 2.1 The governing law, stated by the format author himself
[ik_llama.cpp discussion #8 (IQ_K family)](https://github.com/ikawrakow/ik_llama.cpp/discussions/8) (fetched): "**Token generation is memory bound, so speed is entirely determined by bpw**; packing details and lookup tables are irrelevant to decode speed." On our lane this is measured truth (93-100% of 49.5 GB/s) — so the format question reduces to: **minimum quantization error per stored byte, with dequant cost that hides under DRAM latency.** Complex entropy-coding is free if it fits the issue slots; large LUT walks are not.
- IQ_K formats (fetched): polynomial-fit non-linear scale mapping rounded to INT8, stored in tiny LUTs (4-64 entries) instead of 256-2048-entry codebooks; blocks of 16 in super-blocks of 256 (IQ2_K/IQ3_K/IQ4_K) or 32 (KS variants). **IQ2_KS = 2.1875 bpw, IQ2_K = 2.375, IQ3_K = 3.4375, IQ4_KS = 4.25**; an extra bit selects one of two LUTs (how IQ4_KS beats IQ4_XS at identical 4.25 bpw).
- CPU numbers (AVX2 Ryzen 7950X, fetched): IQ3_S pp512 28.04 t/s (mainline) → 180.77 t/s (iqk_avx2) = **6.45×**, tg128 6.80 → 16.10 t/s; IQ2_XS tg 10.88 → 21.46. Quality (QError = PPL ratio −1, LLaMA-3.1-70B): IQ3_K accuracy "very similar" to IQ3_S at same bpw; 16-weight blocks are 20-25% slower at pp than 32-weight blocks (decode-neutral).
- Mainline caveat: [llama.cpp feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix) notes **IQ3_S and IQ1_S are slow on CPU** ([issue ref #5886](https://github.com/ggml-org/llama.cpp/issues/5886)) — the i-quant codebook walk costs decode headroom; the IQ_K polynomial-LUT family was built to remove exactly that ([discussion #8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)).

### 2.2 LUT-GEMM line: what CPUs can and cannot win at low bits
- [T-MAC](https://arxiv.org/html/2407.00088v2) (fetched): LUT mpGEMV, g=4 → 16-entry tables; bit-serial additive decomposition; "up to 4× throughput, 70% energy" vs llama.cpp b2794; x86 tested device is AVX2-only (i5-1035G7) — **no AVX-512 tables in the fetched text**; multi-thread mpGEMV "primarily constrained by memory bandwidth" (confirms our regime). Kernel NMSE parity with llama.cpp at W4; their "fast aggregation" trades 2.5× NMSE. Quality is format-carried (quantizer under it is GPTQ/BitDistiller-class), not kernel-carried. [code](https://github.com/microsoft/T-MAC).
- [Vec-LUT](https://arxiv.org/html/2512.06443v1) (fetched): replaces per-token scalar lookups with one shared 1→N table, plain contiguous vector adds — **no PSHUF/TBL at all**; 1.8-3.2× over T-MAC scalar LUT, 3-4× prefill vs llama.cpp on AVX2/NEON; ternary-only (1.58-2.0 bpw, lossless packing); misaligned LUT layouts cost up to 12×. Prefill-side lesson for our 6.3K tok/s marginal lane; decode-side irrelevant below ~3 bpw on mixed formats.
- [LUT-GEMM (NeurIPS)](https://openreview.net/forum?id=gLARhFLE0F): the foundational kernel formulation (search-level).
- Take for Store A: bytes are the currency; the win is **same-or-better accuracy at lower bpw with dequant that fits the free issue slots**. IQ_K family (polynomial LUT, 32-weight blocks) is the measured best-fit on AVX2, with AVX-512 ports pending in-repo. VNNI (vpdpbusd) is the wrong tool at 2-3 bpw weight-only: it accelerates INT8×INT8 MACs, which is a prefill/activation-quant story, not a decode-bytes story.

## 3. Expert heterogeneity: routing statistics → per-expert bits

### 3.1 Measured: frequency heuristic works only when routing is skewed
[QuantMoE-Bench](https://arxiv.org/html/2406.08155v1) (fetched, Table 1 + Fig 1):
- DeepSeek-MoE-16B (skewed routing): giving 4-bit to the most-frequent experts at fixed extra-byte budget: **top-25 experts MMLU 41.06 vs random 36.54±1.55** (top-10: 62.99 vs 62.86 avg — the win concentrates on MMLU); frequency beats random by ~1-2 pts avg consistently.
- Mixtral (balanced routing): Frequent-2 = 54.20 vs Random-2 = 49.10±7.73 (win), but Frequent-4 = 63.82 vs Random-4 = 63.70±0.49 — **Δ≈0.1: frequency allocation is worth nothing on a balanced router**. Caveat carries directly: Store B's flat gate mass ⇒ frequency-based allocation is dead there; Store A's 512-expert top-10 router is exactly the skewed case where it pays.
- Frequency-vs-random detail: 2→4-bit upgrades of top-k frequent experts are real; 4→8-bit gains are minimal (top-10: 63.10 → 63.31).

### 3.2 Dynamic hot/cold precision tiering
[DynaExq — Dynamic Expert Quantization](https://arxiv.org/html/2511.15015v1) (fetched): EMA hotness score per expert Sᵢ(t) = αSᵢ(t−1) + (1−α)gᵢ(x_t); period-T sort, top experts promoted to INT4, rest demoted to INT2; pre-built copies swapped SSD→DRAM→HBM on async streams. Qwen3-MoE-80B (512-expert class, same shape family as Store A): static INT4 77.74 / INT2 72.65 vs DynaExq 76.68 (A6000) — **up to +4.03 pts over the static low-precision baseline**. Costs: dual precision copies resident (SSD-side), controller on CPU, ≥85% of INT2 throughput retained.
- Transfer shape for us: we do NOT need the dynamic machinery — we hold a 1,615-window live trace, so the **EMA hotness → static bit map** is the transferable core; the dual-copy swap layer is GPU-HBM economics and stays DEAD on a DRAM-bound lane (two copies = more bytes, the opposite of our constraint).

### 3.3 Router-calibration PTQ (protecting routing decisions at no byte cost)
[EAC-MoE (ACL 2025)](https://arxiv.org/html/2508.01625v1) (fetched): despite the name, not mixed-precision — uniform expert bits (2/2.5/3) + **router calibration with TopK-MSE on top-K logits only** (~1.3-2.8% of GPTQ time) + selection-count pruning at prefill. Mixtral 2.06-bit: WT2 5.09 / 8-task 66.31 vs GPTQ 5.51 / 62.56; combined with pruning 71.68 @ 18.98 GB (1.68×). Also documents PMQ calibration overfitting (Table 9, above).
- [Rank-Aware PTQ for MoE](https://openreview.net/forum?id=kPgLp47bJf): router ranking instability under quantization is the main degradation channel (search-level). Same thesis as the queued 2610.41366 replacement-operator law — do not double-count.

## 4. Shared/dense components: is the quality-per-byte gradient steeper there?

Yes — measured, not folklore ([QuantMoE-Bench](https://arxiv.org/html/2406.08155v1), fetched):
- Shared experts (DeepSeek-MoE-16B, Fig 3): 4/8-bit on the 2 shared experts vs the same bytes on 2 random routed experts **consistently wins** — shared experts run on every token, so their quantization error is injected once per token, never amortized across routing diversity.
- Attention (Fig 2): higher bits on attention vs random FFN experts wins **>5% at each bit budget** — attention runs every token on every layer.
- Early layers (Table 2, Mixtral): first-4 blocks @4-bit 51.12 vs last-4 45.54 (Δ 5.58); first-8 58.16 vs last-8 47.36 (Δ 10.80). DeepSeek deltas thin (≈1.0) — early-layer steepness is a Mixtral-size phenomenon; verify per-store.
- Corroboration: AlphaQ keeps non-expert (attention+router) at uniform 4-bit while experts range {1,2,3,4} ([fetched](https://arxiv.org/html/2606.04980v1)); GEMQ fixes attention at 4-bit and shows 2-bit attention collapse (WT2 31.14) ([fetched](https://arxiv.org/html/2605.23078)).
- Verdict for Store B's 24 MB/layer shared expert at Q5_K: the gradient is steeper in the shared expert than in the routed mass — **raising it (Q6_K/IQ5_K-class) while harvesting bytes from cold routed experts is the right trade direction**; cutting it is the wrong lever. It runs on 100% of tokens: its per-byte marginal quality is bounded below by the routed-expert curve.

## 5. KV quantization beyond FP8/INT4 (MLA-style for Store B; GQA-side for Store A)

### 5.1 RAQ-MLA — role-aware 2.47-BPE MLA cache (the headline for Store B)
[RAQ-MLA](https://arxiv.org/html/2610.33625v1) (fetched).
- Mechanism: split the MLA cache into its two roles — content latent c_kv (d=512) and decoupled RoPE key k_R (d=64). Error-propagation analysis: the **positional key carries 92.5% of attention-logit attribution** (its outlier channels: 63.8% from a tenth of 64 channels); the latent behaves as the value path (12.3% cache error → 1.7% at output via content-key, 8.3% via value). Allocation: latent per-token asymmetric INT2 (g128, clip 0.94) with a **value-aware invertible transform** fitted against attention-exposure-weighted value error; RoPE key per-channel INT4 with a **score-aware transform** fitted against positional-logit perturbation; alternating minimization, no gradients, transforms fused at load (A_c⁻¹ folds into W^UK/W^UV; A_R⁻ᵀ applied to the never-cached positional query).
- Numbers: **2.47 BPE effective** (2.25 latent + 4.25 RoPE) vs 16.00 BF16 = 6.5× capacity; near-zero mean deltas (GLM-4.7 −0.08, GLM-5.2 0.00, Moonlight +0.05, DS-V2-Lite −0.14 over 5 tasks); RULER-NIAH 93.66 @ 2.66 BPE vs OSCAR 93.60 @ 3.78 BPE (**29.6% smaller cache at parity**); decode kernel 12.6 vs FlashMLA 17.4 μs/layer (1.38×) on H20 TP2; TPOT 5.97 vs 6.56 ms. Baselines with BF16 RoPE key floor at 3.78 BPE — the fitted per-channel RoPE transform is what buys the last 1.3 BPE.
- Fit check: Store B is MLA with the same 512/64 split. Our FP8-KV (8 BPE) → 2.47 BPE ≈ **69% KV bytes cut**; vs a naive INT4 cache ≈ **45% cut at parity**.

### 5.2 QuantMLA — dual-path transforms + C2R4 with an error bound
[Function-Aligned Dual-Path Quantization for Low-Bit MLA (QuantMLA)](https://arxiv.org/html/2609.36760v2) (fetched).
- Mechanism: content path gets RMSNorm-commuting rotation+scales trained on attention-output reconstruction; RoPE path restricted to per-frequency-pair SO(2) rotations + paired scales (must commute with RoPE), trained on positional QK reconstruction; **Proposition 2: ‖ΔY_P‖²_F ≤ (τ²/4)(Σ‖V̄_h‖²)·Σ‖Ω⊙ΔZ_P,h‖²_F**. RoPE-vs-content distortion response ratio 1.94-6.86× across models — the RoPE path is the fragile one, again. Native FlashMLA decode-path kernel with fused unpack/dequant; BF16 sink-4/recent-128 protection.
- Numbers: C4R4 DS-V2-Lite 66.54 vs BF16 66.80 (−0.26); **C2R4** (INT2 content + INT4 RoPE): DS-V2-Lite 64.42 (+7.20 over best baseline), Moonlight 72.70 (+10.84); 3.22× physical KV compression @4K (320 B/token/layer incl. protection), near-BF16 attention latency to 1M ctx; vLLM R1-0528 8×GPU: 128K-context preemption 32 → 0 events, job throughput 5.17× (capacity-driven, workload-specific).
- For Store B at 150-175K ctx: C4R4 as the floor config, C2R4 as the capacity mode — with our own gate deciding.

### 5.3 Per-layer sensitivity maps (non-MLA / GQA side, for Store A's KV)
- [KVTuner (ICML 2025)](https://arxiv.org/html/2502.04420v2) (fetched): offline multi-objective search (Optuna+MOEA/D with Pareto+DBSCAN pruning, 3.4×10³⁰ → 15,625 configs) over per-layer (K-bit, V-bit) pairs from {2,4,8}; layer sensitivity is **prompt-independent** (a model property). **Nearly lossless 3.25-bit avg** on Llama-3.1-8B (GSM8K 0.7925 vs BF16 0.8038; uniform KIVI-2 collapses to 0.6327); sensitive models floor at 4.0-bit (Qwen2.5-7B uniform KV4 = 0.0078 → collapse; KVTuner-C4.00 = 0.7559). K>V asymmetry: key needs per-channel + higher bits (attention-shift error e_a grows 13.9× K8→K4); K8V2 (5-bit) beats K4V8 (6-bit) with 12.5% less memory. L40 GPU, KIVI kernels, flash-attention-compatible (coarse per-layer granularity). 26.7-38.3% throughput over KV8 at C6.
- [KVmix (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/download/40422/44383): gradient-based per-layer importance, 3-bit floor kernels (search-level).
- [MixKVQ](https://arxiv.org/html/2512.19206v1) (fetched): extreme low-bit — key channels triaged BF16/UINT4/UINT2 by **query-magnitude × scale salience** (query/key scale correlation only 0.16 — scale alone mispredicts), values uniform per-token 2-bit; lazy update every 128 tokens (2.17% of layer time). R1-Distill-Qwen-32B @ **2.3-bit avg: 66.04 vs BF16 67.84** (KVTuner-C2.91: 61.11, KIVI-KV2: 58.89); LongBench Mistral-7B C2.7 53.68 vs BF16 53.84. A800; not yet serving-integrated; **MLA explicitly excluded**.

## 6. Ranked verdicts for the two stores (quality-parity byte savings)

| # | Mechanism | Store | Evidence anchor | Expected delta at parity |
|---|---|---|---|---|
| 1 | RAQ-MLA role-aware MLA KV (INT2 latent + fitted INT4 RoPE transform) | B | §5.1 | **~45% vs INT4, ~69% vs FP8 KV bytes**; 175K-ctx capacity ↑6.5× vs BF16 |
| 2 | GEMQ BLP expert allocation driven by our trace (FIM on routed output) | A (+B ranking) | §1.1 | GEMQ 2.5 bpe ≈ uniform ~3.0 bpe ⇒ **~15-17% weight bytes**; MMLU +12.7 pts at fixed bytes (Mixtral) |
| 3 | Router-norm closed-form split + MaxVar (theory-bounded) | B (flat gate) | §1.3 | parity vs uniform 3-bit @ **2.75 bits ⇒ ~8% bytes**; free to compute |
| 4 | IQ_K format swap on Store A (IQ2_KS 2.1875 / IQ3_K 3.4375, polynomial LUT) | A | §2.1 | IQ2_KS ⇒ **~34% fewer bytes than 3.3 bpw** at near-i-quant accuracy (gate decides); decode-neutral |
| 5 | Hot-expert ±1-bit split from EMA hotness (static, trace-driven) | A only | §3.1/§3.2 | +4.5 MMLU pts at fixed bytes on skewed routers (DS-MoE); **zero bytes** — reallocation |
| 6 | AlphaQ knapsack on PL_Alpha_Hill (calibration-free) | A+B sanity layer | §1.2 | Qwen1.5-MoE 3.5-bit = BF16 parity (>4×); allocator runs in seconds on CPU |
| 7 | Shared-expert Q5_K→higher / harvest routed cold bytes | B | §4 | gradient steeper in shared expert; zero-byte trade, direction only |
| 8 | KVTuner per-layer (K,V) map @ ~3.25-bit | A KV | §5.3 | ~19% KV bytes vs uniform 4-bit at lossless gates |
| 9 | EAC-MoE TopK-MSE router calibration | A+B | §3.3 | zero bytes; ~1-4 pts avg at 2-2.5 expert bits |
| 10 | QuantMLA C2R4 capacity mode | B KV | §5.2 | 3.2-3.6× physical KV pack at −0.3 to −2.4 pts |

SKIP list: DynaExq dynamic dual-copy tiering (GPU-HBM economics; dual copies = anti-constraint on a bytes-bound lane — steal only the EMA statistic); MxMoE W4A16/W8A8 activation co-design (compute-bound GPU regime; our lane is weight-only DRAM-bound); Vec-LUT/T-MAC decode kernels for Store A at 3+ bpw (they target ternary and prefill; decode is bpw-determined per §2.1); MixKVQ for Store B (MLA excluded by the paper); frequency heuristic on Store B (flat gate mass = Mixtral-style Δ≈0.1 null result).

## 7. Cross-cutting laws worth carrying

- **Bytes are the only decode currency on a saturated lane** — every mechanism above should be scored as (quality parity, bpw delta, dequant issue-slot cost), in that order ([discussion #8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8); [T-MAC](https://arxiv.org/html/2407.00088v2) bandwidth note).
- **Allocation signals, ranked by calibration cost**: router-norm/HTSR spectral (zero data) → FIM-on-routed-output (needs activations, seconds of CPU) → live-trace EMA hotness (we own it) → offline-corpus frequency (overfits; [EAC-MoE Table 9](https://arxiv.org/html/2508.01625v1)).
- **MoE components are not one curve**: attention > shared experts > first-blocks (sometimes) > routed experts; 2-bit attention is a collapse threshold, not a curve point ([QuantMoE-Bench](https://arxiv.org/html/2406.08155v1), [GEMQ](https://arxiv.org/html/2605.23078)).
- **MLA cache roles are not symmetric**: RoPE path ≈ 92.5% of logit attribution, amplifies error 1.9-6.9×; content latent is the compressible mass ([RAQ-MLA](https://arxiv.org/html/2610.33625v1), [QuantMLA](https://arxiv.org/html/2609.36760v2)).
- **Router calibration is free quality**: TopK-MSE on top-K logits costs ~1-3% of quant time and is orthogonal to every allocator ([EAC-MoE](https://arxiv.org/html/2508.01625v1)).
