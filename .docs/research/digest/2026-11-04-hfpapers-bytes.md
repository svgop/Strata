---
type: digest
id: hfpapers-bytes
created: 2026-11-04
status: evidence
topic: HuggingFace daily-papers sweep (2026-10-21 → 2026-11-04) + arXiv full texts — newest work that cuts WEIGHT BYTES on the CPU lane or SHORTENS the serial decode step for hybrid single-GPU+DRAM MoE inference
scope: read-only web research. Method: HF daily_papers API (478 papers, 15 calendar days), 15 targeted HF paper-search queries, full-text (arxiv.org/html) verification of 16 papers — mechanisms and numbers below are from paper bodies, not abstracts. Abstract-level entries are marked [abstract-level]. Every claim carries its HF/arXiv URL. Priced only against the measured law: lane = 49.5 GB/s DRAM ceiling, Qwen CPU lane at 93-100% of ceiling (BINDING), DS GPU term 7.2 ms vs lane 7.3 ms (GPU wins pay now), 262K context, 12 GB VRAM.
---

# HF papers digest — bytes-cutting and step-shortening, late Oct / early Nov 2026

Question: from the current feed, what NEW mechanisms cut expert bytes on the DRAM lane, cut KV/attention bytes in the GPU term, or shorten the serial attn→moe chain — beyond the queued set (REAP-class re-bake, AQLM/VQ, FP8 KV, NVFP4/FlashInfer SM120, conditional CUDA graphs, EAGLE-3, REST, draft-K gating, MoE-Infinity prefetch, KTransformers) and without re-proposing falsified items (sparsity gating, SMT, THP, TF32, flat-routing prefetch guesses)?

## A. WEIGHT-BYTES family

### A1. Channel Selection for MoE Decoding Needs No Calibration Data — 2610.37386 — RANK 1, TRANSFER
- Source: https://huggingface.co/papers/2610.37386 ; https://arxiv.org/abs/2610.37386 (full text verified)
- Mechanism: per-expert CHANNEL selection (channel = one intermediate dim owning a gate/up row + down column). A selector = score + keep rule; top-k keep needs zero calibration data. Kept channels' weights are read at decode; skipped channels are reconstructed from (a) a rotated TERNARY SKETCH of gate/up rows — randomized Hadamard rotation in 128-coord blocks, codes {-1,0,1} + FP16 scale, 256 B → ~30 B per block (~7% resident-memory overhead) — and (b) a rank-32 low-rank map of down-projection columns (~0.5% overhead). Byte cost is charged in 32-B sectors: their reader totals 12.8% and 17.8% of routed-expert reads at the two lowest budgets.
- Measured: Nsight-verified DRAM reads 18.3% of full expert reads at batch 1 (12 Qwen3-Thinking layers, cold cache); MoE-block speedup 1.90x (Qwen3-30B-A3B-Thinking), 1.60-1.71x others; whole-model 1.15-1.36x batch 1, up to 1.88x at batch 8-16; serving 3.3-4.3x more tokens at 100 tok/s/user. HW: RTX PRO 6000 Blackwell (vLLM 0.29.0), A6000, 4x A5000/3090 — no cluster.
- Quality: at 17.8% reads, 98.1% of full-model accuracy (mean, 9 model-benchmark pairs: GPQA-Diamond/IFEval/MMLU-Pro), worst pair 95.0%; at 12.8% reads, mean 94.5%, worst 86.9%. Beats calibrated-threshold selectors (Prox 68.5% mean, TEAL 0%, WINA 1.1% at their budgets); thresholds also drift below budget (216/216 runs) and cause repetition-loop collapse.
- Lane verdict: TRANSFER — this is the largest lane-byte lever found. Expert reads at ~18% = ~5x fewer bytes on the 49.5 GB/s lane. Port path for the Qwen engine: re-bake each IQ3 expert with SELECTED CHANNELS PACKED FIRST in the store (contiguous per-expert layout), read only the selected byte range per routed expert — sector-strided gather stays coalesced on CPU. Open risk (named): selection + ternary-sketch recovery was proven on BF16 stacks; on top of 3-bpw quantized stores the stack quality is unproven — needs one A/B re-bake + eval before committing. Sketch is stored, not read, per token (7% VRAM/RAM cost once).
- Composition: multiplies with A4 (fewer experts) and runs under any quant allocator (A5).

### A2. Specter: Quantization-Enabled Multi-Step Speculative Offloading (ASPLOS '27) — 2511.14102 — RANK 2, TRANSFER
- Source: https://huggingface.co/papers/2511.14102 ; https://arxiv.org/abs/2511.14102 (full text verified); code https://github.com/wwwwwf/specter_asplos (Zenodo DOI 10.5281/zenodo.22937478)
- Mechanism: the INT4-quantized copy of the TARGET MoE itself is the draft (zero training; shares routing lineage — measured top-4 routing match 90.8% mean at INT4-GPTQ, routers/attention/shared experts kept FP16 because routing is quant-sensitive). Draft K tokens (K=8-16 tuned offline, acceptance ᾱ=96.63%, E[A]=28.7); the draft's ROUTER OUTPUTS feed an Incremental Routing Pattern Analyzer → per-layer expert working-set prediction (Expert Lookahead Buffer); a Layer-Aware Prefetch Planner splits drafting into chunks sized so compute ≈ transfer and issues async PCIe H2D of the top-M predicted experts on a separate stream; target FP16 verifies all K tokens in one masked pass (lossless by construction).
- Measured: single-request TPOT 1.50x avg / 1.88x max over spec-less MoE offload baseline; DSv2-Lite 72.2 ms/tok vs 344.9 (HF Transformers), Qwen1.5-MoE 35.3 ms vs 66.4; batch-32 throughput 130.3 tok/s vs 40.2 (3.24x). 55.5-60.5% of H2D load time hidden behind draft compute (22.4% on the 41.9B model). Cache hit 96.25-99.85%. HW: ONE A100-40GB + 24-core Xeon + 256 GB RAM, PCIe 4.0 (~31.5 GB/s) — i.e., a WEAKER memory hierarchy than ours.
- Quality: zero by construction (lossless spec sampling); misprediction costs latency only (75% wrong prefetch → 63.0→84.2 ms).
- Lane verdict: TRANSFER — and it cleanly EVades our falsified "prefetch-guessing on flat routing": the guess here is the draft's OWN router running on the exact next K tokens, not a statistical guess on flat history. On the Qwen engine (spec decode already running, VRAM-tier + CPU pool): draft step produces per-layer activation histograms for the verify batch; DRAM→VRAM staging of predicted experts overlaps the draft window instead of serializing after it. Expected effect: hide a large fraction of the verify-side lane wait behind already-paid draft compute — same magnitude class as their 55-60% I/O overlap; TPOT cut bounded by draft-window length vs lane time. K and chunk sizing are the tuning surface; their offline TPOT profiling maps to our draft-K gating work.

### A3. PuzzleMoE: Sparse Expert Merging + Bit-packed Inference — 2511.04805 — RANK 3, TRANSFER
- Source: https://huggingface.co/papers/2511.04805 ; https://arxiv.org/abs/2511.04805 (full text verified)
- Mechanism: pairwise 2→1 expert merging with two masks — similarity mask (symmetric percent difference of |W| ≤ 0.4 → magnitude-average) + saliency mask (Wanda |W|·||X||₂ keeps the salient expert's value at divergent positions); per-expert SIGNS stored and re-applied at inference; masks+signs packed INTO the BF16 exponent field (MoE exponents span ~5 bits, 3 bits free) so one memory access serves both merged experts. Custom Triton decode-GEMM decodes on the fly (~2x cuBLAS on the kernel).
- Measured: 25%/50% expert reduction; with 3-bit quant → 3.35 avg bpw ≈ 4.8x total compression; end-to-end up to 1.80x throughput (Mixtral 8x7B, 50%). HW: 8x RTX 5090 (serving), RTX 4090 (kernel benches) — CONSUMER Blackwell numbers exist.
- Quality: Mixtral PPL 3.84→4.10 (25%) / 4.36 (50%); MMLU 67.9→66.8/65.7; Qwen3-MoE-30B avg 72.6→71.9/71.2 (MMLU 77.8→75.1 at 50%); at 75% sparsity still beats prior merging (HC-SMoE 35.7 vs 54.1 avg on DeepSeek-MoE); +3-bit costs only 1.0-1.7% avg.
- Lane verdict: TRANSFER as a RE-BAKE TARGET FORMAT for the CPU pool: 512→256-expert pool halves distinct experts addressed per layer; top-k over the merged pool reads one packed weight where two experts stood (sign/mask decode on CPU is XOR-grade, no entropy codec). Expected lane-byte effect at 50% merge ≈ −40-50% on the expert lane; quality cost is the knob (start at 25%). The exponent-bit packing trick is BF16-specific — on our IQ3 store the equivalent is a packed sidecar-free layout; masks must live in the re-bake. Composes with A1 (channel select inside merged experts) and A5 (per-merged-expert bpw allocation).
- Caution from B3 (below): merge/fill must respect the router's trained gate semantics (renorm vs keep-scale), or gains evaporate.

### A4. Training-Free MoE Expert Pruning — Replacement Operators and Routing Semantics — 2610.41366 — TRANSFER (upgrade to queued REAP work)
- Source: https://huggingface.co/papers/2610.41366 ; https://arxiv.org/abs/2610.41366 (full text verified)
- Mechanism: isolates the REPLACEMENT decision in training-free pruning (selection held fixed). Seven operators in three families: fill-from-pruned (constant μ vs their "tilted novice" = rank-32 ridge linear map of the pruned expert's input, closed-form, +1.2-1.8% params), fill-from-survivor (MoRA proxy / slot remap / COMPEL Fisher compensation), reroute (keep-scale vs renormalize). KEY LAW: the better gate scale at 50% pruning is the one the router was TRAINED with — renormalizing on routers trained without renorm (OLMoE, DSv2-Lite) costs 0.8-8.5 pts; remapping loses everywhere.
- Measured: 25/50/75% pruning on OLMoE, DSv2-Lite, Moonlight, Granite. Tilt beats constant fill by +0.5 to +3.45 pts at 50%; OLMoE 75%: PPL 1434 (constant) → 172 (tilt). Prune+4bit vs uniform 3-bit at matched ~7 GB: tilt wins GSM8K +4.25 on DSv2-Lite; arms keep 98-99% dense GSM8K at 23-24% of FP16 size. Calibration DOMAIN moves masks 4-9x more than resampling (pick in-domain data). HW: not central; one-shot, no gradients.
- Lane verdict: TRANSFER — direct upgrade spec for the queued REAP-class re-bake: (1) use tilted-novice fill instead of drop/zero, (2) keep the router's native gate scale instead of renormalizing over survivors, (3) prune+4bit beats uniform-3bit at matched bytes — i.e., pruning and re-quant are COMPLEMENTS, not alternatives, on the byte budget. Effect: at 25% pruning with tilt, expert lane bytes −25% at ~98-99% in-domain task retention (their numbers, small models).

### A5. NatScale: Contrast-Calibrated Mixed-Precision MoE Quantization — 2610.20353 — TRANSFER (secondary; allocator for any re-bake)
- Source: https://huggingface.co/papers/2610.20353 ; https://arxiv.org/abs/2610.20353 (full text verified)
- Mechanism: post-training, frozen checkpoint. Three stages: (1) gradient-based expert sensitivities → rank-preserving POWER TRANSFORM w_i = g_i^β solved so normalized squared weights hit a Jeffreys-divergence target τ (default 1 nat; Weibull-MLE identity β₁=k̂/4) — rankings intact, CONTRAST controls how hard experts compete for bytes; (2) JOINT gate/up/down error estimation on fixed replay inputs + AlphaQ-style spectral prior; (3) LP-relaxation allocation over expert cost-payload hulls, deterministic rounding under byte budget B. Retuning τ/budget needs NO new forward/backward passes (stats cached).
- Measured: models include Qwen3.8-Flash-Next 125B/6B with 512 EXPERTS TOP-10 — same shape family as our Qwen engine. At routed 2.0/2.5/3.0 bpw: lowest KLD in 9/9, lowest PPL in 7/9 vs AlphaQ etc. (e.g. 35B @2.5 bpw: PPL 7.15 / Top1 87.94 vs AlphaQ 7.31 / 85.70). Concrete: 125B whole-model 74.53-81.95 GB tiers, "−1.31 GB at −11.04% KLD" vs GSQ-RCO IQ3_XXS. GGUF-native baselines (UD, IQ) compared directly. HW: not reported; no tok/s (allocation method).
- Lane verdict: TRANSFER as the ALLOCATOR for the next mixed-bpw re-bake: same bytes → lower KLD (quality headroom), or same quality → ~1-3% fewer bytes. Small direct win, zero-risk composition with A1/A3/A4. Its 512-expert/top-10 eval shape is the closest published proxy for our Qwen store.

### A6. QuantMLA: Function-Aligned Low-Bit MLA KV (INT4/INT2) — 2609.36760 — TRANSFER (DS engine, GPU term + VRAM headroom)
- Source: https://huggingface.co/papers/2609.36760 ; https://arxiv.org/abs/2609.36760 (full text verified)
- Mechanism: MLA KV = content latent C (dim d_c) + decoupled RoPE key K_P. C4R4 = INT4/INT4; C2R4 pushes content to INT2. Per-token asymmetric affine, group 64. Function-aligned transforms: content path = orthogonal rotation + per-channel scale fused via RMSNorm equivariance, trained on attention-OUTPUT reconstruction vs frozen BF16 teacher; RoPE path = block-diagonal SO(2) rotations + per-frequency-pair scales that COMMUTE with RoPE, trained on positional-QK reconstruction (finite distortion bound, Prop. 2). Protects 4 sink + latest 128 tokens in BF16. Native FlashMLA-based decode kernel fuses unpack/dequant into attention.
- Measured: KV compression 3.22x @4K → 3.59x @128K vs BF16 incl. metadata (320 vs 1152 B/token/layer); attention latency −7.4% to +4.3% (1K-4K), within ±0.5% from 64K up; serving under cache pressure 2.61→13.51 tok/s (5.17x, BF16 preempted 32x vs 0). Models: DSv2-Lite, Moonlight, LongCat-Flash-Lite, GLM-4.7-Flash; error study up to Kimi-K2/DSV3 scale. HW: 8-GPU vLLM serving (kernel is single-GPU native).
- Quality: C4R4 −0.13 to −0.26 pts avg vs BF16; S-NIAH-3 87.20 (best baseline +7.40); C2R4 INT2-content beats adapted-SmoothQuant/QuaRot by +6.2 to +10.8 pts; ablation: RTN 51.90 → +transforms+BF16-protect 57.95 ≈ BF16 57.90 MMLU.
- Lane verdict: TRANSFER for DS: vs the queued FP8-KV this is another ~1.8x KV-byte cut; at 262K that converts directly to VRAM headroom (12 GB box, ~1.2 GB free) and to fewer bytes in the attention kernel inside the 7.2 ms GPU term — and if any KV spills across the lane at 262K, it cuts lane bytes too. Work: port the RoPE-commuting transform + dequant-fused decode to FlashInfer sm_120 kernels. C2R4 is the stretch target; C4R4 is the safe re-bake.

### A7. TaSQ: 1-Bit KV via Tailored VQ Space — 2610.03027 — WATCH
- Source: https://huggingface.co/papers/2610.03027 ; https://arxiv.org/abs/2610.03027 (full text verified)
- Mechanism: VQ on PRE-RoPE keys (35% lower reconstruction error); query-guided DIAGONAL channel weighting (dense inverse rejected: +36→103% per-key overhead at 1k→16k); cross-head shared FP16 scale; covariance-aware 8-channel grouping via RoPE-pair-constrained matching (cost↔MSE r=0.994). K=1024 Fisher-weighted centroids. Values: standard contiguous VQ. Effective 1.27/1.25 bits K/V. RoPE applied on the fly in permuted coordinates.
- Measured: KV pool 235K → 2.94M tokens (12.5x); throughput 220.8→412.6 tok/s (1.87x vs BF16 KV); encode overhead +10-14% TTFT at 8k-32k prompts. HW: single RTX 6000 Ada, SGLang Triton kernels.
- Quality: WikiText-2 PPL 6.24→8.46 (real cost); general avg 59.21 vs 63.94 BF16; 64k NIAH 93.17 vs 98.17; long-CoT reasoning degrades least among 1-bit baselines (60.55 vs 67.18 BF16, NSNQuant 46.67).
- Verdict: WATCH — 1-bit KV frees VRAM aggressively, but the −4.7 avg / −5 NIAH cost is too steep for a 262K agentic lane where retrieval is load-bearing; revisit only if A6's INT4 still leaves KV as the binding VRAM term. Dequant-on-fly codebook lookups also add GPU-term work.

### A8. EntroPack: Entropy-Coded Weights at Arbitrary Bitrates — 2609.34185 — SKIP (named)
- Source: https://huggingface.co/papers/2609.34185 ; https://arxiv.org/abs/2609.34185 (full text verified)
- Mechanism: E8-lattice + rANS entropy coding, tile-level random access (32 states/tile), GPU decode fused into matmul; arbitrary bpp via scale bisection (rate error 0.8-2.1%); storage rate decoupled from compute dtype.
- Measured: 4.03 bpp → PPL 6.06 vs NF4 6.08 @4.50 bpp (better error at fewer bytes); decode overhead at OUR shape (batch-8-token autoregressive verify): +76% (272.8 vs 155.1 ms); per-layer CUDA decode 0.23-0.41 ms on H20; good only ≥4k-token batches.
- Verdict: SKIP for the CPU lane: rANS decode throughput (CPU, even tiled) sits far below the 49.5 GB/s lane — entropy decode becomes the new serial bound, exactly the "kernel speedup that doesn't cut effective bytes" trap; on the GPU tier it loses at M≤8 where our verify runs. The 4.03 bpp floor is also above IQ3 territory at matched quality.

### A9. [abstract-level] Weight-bytes family context
- GEMQ — Global Expert-Level Mixed-Precision for MoE — 2605.23078 (https://huggingface.co/papers/2605.23078): mixed-bpw allocation that explicitly models ROUTER SHIFTS induced by quantization; May 2026; superseded for us by NatScale (A5) but the router-shift term is a re-bake eval check worth stealing.
- FlexMoE — Nested Intra-Expert Pruning — 2606.27866 (https://huggingface.co/papers/2606.27866): ONE artifact serves multiple byte budgets via nested rank truncation per expert; useful if we ever serve tiered budgets from one re-bake.
- AIMER — Calibration-Free Task-Agnostic MoE Pruning — 2603.18492 (https://huggingface.co/papers/2603.18492): absolute-importance scoring without calibration sets; alternative selector to A4's, weaker evidence.
- WUSH-KV — 2609.38121 (https://huggingface.co/papers/2609.38121): data-adaptive second-order transforms for low-bit KV, value transform folded into weights, key transform post-RoPE; same goal as A6 with different transform algebra — second candidate if the RoPE-commuting port stalls.
- HyQuant — 2608.27875 (https://huggingface.co/papers/2608.27875): hybrid bit allocation across attention states (most states low-bit, outliers high); KV family.
- LEMS/KFAC-SVD — 2610.32878 (https://huggingface.co/papers/2610.32878, on feed 11-03): layer-wise error-model search fixes Fisher rank collapse in SVD compression; WATCH as a low-rank+quant re-bake option — on the byte-bound lane two GEMVs replace one, FLOPs are free, but rank quality at 3-bpw-equivalent budgets is unproven.

## B. STEP-SHORTENING family

### B1. JSSA: Joint Subspace Sparse Attention — 2610.19026 — TRANSFER (DS GPU term at 262K)
- Source: https://huggingface.co/papers/2610.19026 ; https://arxiv.org/abs/2610.19026 (full text verified)
- Mechanism: training-free, token-level, per-KV-head top-B selection at decode. Auxiliary FP8 PROJECTED-KEY cache (≤6.25% of BF16 KV; 128-dim for MLA) + mean-query projection through a JOINT low-rank basis M (same basis for q and k, chosen to minimize bilinear score error; closed-form spectral solution = top-r eigenvectors of combined query+key second moments; calibration = one eigendecomposition, 129-365 s on B200). Attention then runs on ORIGINAL full-dim K/V of the selected B positions via a gather-free indexed Triton kernel (per-KV-head distinct selections).
- Measured: decode throughput at 128K, B=2048: 1.51x (Qwen3-8B), 1.88x (Gemma4-26B), 6.43x (GLM-4.7-Flash, MLA); DeepSeek-V3.1 ~700B: up to 3.49x over full-KV at 128K batch 8 (96-99% of native DSA throughput); 700B AIME identical to native (87.50). HW: single B200 (vLLM, CUDA graphs, FP8 scoring cache); 8xB200 for the 700B runs.
- Quality: LongBench within 0.67 pts of full-KV; RULER 128K 75.38 vs 76.97 (budget 1.56%); PG-19 PPL gap ≤1.11; reasoning AIME within ±3.3 with generation-length inflation only 1.02-1.17x.
- Verdict: TRANSFER for the DS engine at 262K: attention KV reads inside the 7.2 ms GPU term drop by ~large factor at budget 2-4K of 262K; MLA path is explicitly supported (GLM-4.7-Flash is the best case). Work: sm_120 port of the indexed-attention kernel, FP8 KV first (queued) to fund the projected-key cache within ~1.2 GB free (MLA 128-dim keeps it small). Calibration (second moments + eigh) is a one-time local job. This is the strongest 262K attention lever on the feed.

### B2. SSA: Stratified Sparse Attention (heavy + sampled + importance-corrected) — 2610.28441 — TRANSFER-adjacent (backup to B1)
- Source: https://huggingface.co/papers/2610.28441 ; https://arxiv.org/abs/2610.28441 (full text verified)
- Mechanism: training-free. Split budget k into deterministic heavy set (λ-solved from proxy-score probabilities) + multinomial SAMPLE from the residual with 1/π_i importance correction — recovers attention mass that pure top-k discards; (ε,δ) error bound. Works on any proxy (PQCache 16-B codes/key = 60x smaller than 1-KiB fp16 pair).
- Measured: single H200; decode attention at 1M ctx vs FlashInfer: 50x sparsity → 1.79x (batch 1) to 2.96x (batch 16); at batch 1 SSA only beats FlashInfer BEYOND 256K context — exactly our regime. Quality: RULER-HARD 50x = 94%/85% of dense (32K/128K); AIME 59.8 at 50x vs oracle top-k 48.3, PQ 36.7.
- Verdict: keep as the QUALITY-AT-EXTREME-SPARSITY upgrade if B1's selection misses tail mass at 262K; its sampler adds 0.30-1.18 ms at 1M — must be re-priced at 262K. Code released.

### B3. Training-free pruning routing semantics (2610.41366) — see A4: the reroute/gate-scale law is ALSO the step-shortening guardrail for A3/A4 re-bakes (rerouting at batch 32 keeps only 5-7% of MoNE's savings; batch 1 is our shape).

### B4. Specter (A2) doubles as step-shortening: the ELB prefetch shortens the effective serial step by overlapping the lane wait with draft compute; K-chunk planner is the new draft-K gating surface.

### B5. SpecPilot: Agentic Optimization of Speculative Decoding — 2611.02501 — WATCH (ideas transportable; loop not)
- Source: https://huggingface.co/papers/2611.02501 ; https://arxiv.org/abs/2611.02501 (full text verified)
- Mechanism: autonomous loop (cost-model ceiling → gap-targeted patches → sealed validation with token-forcing anti-reward-hacking check) optimizing EXISTING spec systems (DFlash2/DSpark/EAGLE-3 on SGLang). Discovery highlights: best-first draft trees (verify N=32 nodes, depth 15), W12 lossless 12-bit weight format (verify-pass reads 44.67→33.58 GB on Qwen3.8-27B), TWO-LEVEL verification (cheap Q pass + exact pass; 6.16→11.67 tokens/exact pass), batching mechanisms (shared node budget, per-request drafter block).
- Measured: single-request speedups over AR 8.86-10.87x (stock systems 3.95-4.69x) on ONE H100 and ONE RTX PRO 6000 Blackwell — single-GPU, no cluster; batching 1.51-2.09x stock → 2.09-3.93x. Absolute 416-435 tok/s single-request on H100.
- Verdict: WATCH. All weights GPU-RESIDENT in their evals — W12 targets HBM bandwidth we don't have on the lane; CPU-offload never tested. Transportable pieces: best-first draft-tree shape and two-level verify for the DS GPU term (7.2 ms/round pays now), draft-tree shaping extends queued EAGLE-3 work. The loop itself needs paired-GPU validation and SGLang; our DS has no graphs yet — sequenced behind conditional-CUDA-graphs.

### B6. Soft-MTP: Multi-Token Prediction Done Right — 2610.38331 — SKIP (training-time; queued EAGLE-3 covers the slot)
- Source: https://huggingface.co/papers/2610.38331 ; https://arxiv.org/abs/2610.38331 (full text verified)
- Mechanism: one-line straight-through gradient routing (`h0[:,k:] += hk.detach() - hk`) removes the low-SNR residual (~4x lower SNR, ~80% of gradient mass) from MTP heads, leaving pure KL self-distillation toward the NTP verifier distribution; no extra params/FLOPs.
- Measured: acceptance length +1.8-2.2%, self-spec speedup +1.4-1.5% at 167M-1.2B scales; also improves base PPL.
- Verdict: SKIP as a lane item (we don't train DS; EAGLE-3/NEXTN heads are the queued draft). Keep as a SELECTION criterion when choosing released MTP checkpoints: Soft-MTP-trained heads accept measurably better.

### B7. LSA: Latent Sparse Attention (subquadratic, 10M ctx) — 2610.42147 — WATCH (infra exceeds ours)
- Source: https://huggingface.co/papers/2610.42147 ; https://arxiv.org/abs/2610.42147 (full text verified)
- Mechanism: keys L2-normalized → pooled 4x → Ward-clustered into ~8-key buckets → query scores bucket CENTROIDS (64-dim index space), top-p buckets expanded best-first to k long-range keys + sink + 512 window; flat or tree (O(log T)) search; one shared index list per query. D-LSA = distill only W_K(idx), W_Q(idx), per-layer temperature on a FROZEN model (forward KL); N-LSA = native training; training-free variant uses leading principal directions.
- Measured: 10M ctx on H200/B300 TP8/CP8; TPOT 2.10x (GLM-5.3-Flash) / 2.87x (Qwen-3.8-27B) at 10M; RULER within 5 pts to 1M, EXCEEDS full attention at 10M (63.9 vs 59.9). HW: H200/B300 clusters, CuTe/Triton kernels, SGLang.
- Verdict: WATCH — built for 1M-10M, we sit at 262K where B1/B2 already pay; distillation infra for D-LSA projections is a local-feasible but heavy job; cluster kernels don't port to sm_120 cheaply. Revisit if context target grows past ~512K.

### B8. [abstract-level] Step-shortening family context
- ProVe — 2610.27815 (https://huggingface.co/papers/2610.27815): role-reversal spec — fine-tune the LARGE model into a block-proposer, verify with small/self (ProVe-lossless). Needs large-model fine-tuning: SKIP for 748B; WATCH for future releases shipping proposer checkpoints.
- Substitute Speculative Decoding — 2509.18344 (https://huggingface.co/papers/2509.18344): training-free, lossless spec for OFFLOADED LLMs using a smaller/quantized substitute as draft; Oct 2025 (older than the window, surfaced by search). Our EAGLE-3 queue covers the head path; keep as fallback when no draft head exists for a target.
- P-EAGLE — 2602.01469 (https://huggingface.co/papers/2602.01469): parallel-drafting EAGLE with scalable training; Feb 2026; extends the queued EAGLE-3 item rather than replacing it.
- On the SPOT (2610.22543), D-Loop (2610.06011), Speculative-Diffusion-Two-Tradeoffs (2610.31819) — DLM-side spec variants; see C2 verdict for why DLM doesn't clear our lane law.
- Cache-Aware Joint Router Adaptation — 2609.04895 (https://huggingface.co/papers/2609.04895, full text verified): auxiliary temporal/spatio routers (12.6M params) anticipating expert reuse for the cache; hit 73-91% vs ~62-64% baselines, load 1353→974 MB/token (Qwen3-30B, 128-expert top-8). BUT: gains require FULL backbone+router joint post-training on 8x140 GB accelerators; frozen-backbone variant yields only "modest coverage gains"; eval is trace-simulated, no wall-clock. WATCH (training infra we lack). Same bucket: MaskCoFT 2609.34077 (https://huggingface.co/papers/2609.34077) — co-adaptive router+expert FT for offloaded MoE, fixes router-only adaptation's token-mismatch — WATCH; ReMoE 2605.27081 (https://huggingface.co/papers/2605.27081) — router-only FT predecessor.
- DAMI — 2610.41994 (https://huggingface.co/papers/2610.41994): dual-axis memory INITIALIZATION for gated linear attention long-context retrieval — training-time init for GLA layers; relevant to DS's 36 GDN layers only if a V4.x checkpoint trained with it appears; WATCH.
- You Don't Need Long Context — 2611.05306 (https://huggingface.co/papers/2611.05306, full text verified, feed 11-04): frequent context compaction (EoCC/every-action SAC) holds task accuracy (SWE-bench-V 70.2 vs 70.0; Terminal-Bench +4.6) while cutting peak context 130-200K → 15-37K; quantifies THE COST OF OUR 262K WINDOW: decode ~2x slower at long ctx on DGX Spark (109 GB unified, llama.cpp — our regime), 3.6-7x throughput 16K vs 256K on H100. TRANSFER as an OPERATIONAL lever: a compaction harness cuts the attention/KV term of the serial step without touching weights; caveat: one model collapsed 53.0→3.6 under aggressive compaction — per-model validation required.

## C. New-mechanism checks for single-GPU+CPU-RAM

### C1. Meta Expert: Cross-Layer Expert Sharing — 2610.27598 — SKIP (pretrain-time)
- Source: https://huggingface.co/papers/2610.27598 ; https://arxiv.org/abs/2610.27598 (full text verified)
- Mechanism: ONE router + ONE expert pool shared by ALL layers (trained from scratch); stores 70-80% of vanilla experts at equal-or-better loss (MMLU 0.4175 vs 0.4087 on 6B-A1.6B); shared experts act as O(1) cross-layer residual paths. Storage −20-30% experts / −15-25% params.
- Verdict: SKIP — "trained from scratch," no post-hoc conversion for pretrained MoE. Conceptually the deepest storage cut on the feed; note for any future pretrain effort.

### C2. BYOD: AR→Diffusion LM conversion (LoRA-only) — 2610.41574 — SKIP (lane law)
- Source: https://huggingface.co/papers/2610.41574 (abstract-level; https://arxiv.org/abs/2610.41574)
- Mechanism: LoRA-only recipe converts dense AR models (Gemma-2-9B, Llama-3.1-8B, Qwen2.5-7B, Ministral-8B) to diffusion LMs — parallel token updates per denoise round.
- Verdict: SKIP by our law: a DLM round routes a WHOLE BLOCK of tokens through the MoE → more distinct experts per round → MORE lane bytes per emitted token; rounds are fewer but the lane is byte-bound, not step-count-bound at batch>1 shapes. DLM conversion pays only where weights are GPU-resident (HBM-bound with idle FLOPs), not on our 49.5 GB/s lane.

### C3. [abstract-level] Other feed items checked and cleared
- FLAMIA (2611.03344) — microscaling flash attention for Blackwell TRAINING; causal-safety training focus, not decode bytes.
- Deep Fusion Memory (2610.25100), Cache Engineering for Agents (2610.28577) — agent-memory, not weight/KV bytes.
- Building AI Research Teams for Autonomous Discovery of Linear Attention (2610.28887) — meta-research process paper.
- Stratified Sparse Attention covered at B2; AB-Attention (2610.39663) — video-diffusion blocks; Specter covered at A2.

## D. Ranking by expected decode gain on THIS box

| # | Candidate | Engine | Family | Expected effect | Risk |
|---|-----------|--------|--------|-----------------|------|
| 1 | Channel selection, calibration-free (2610.37386) | Qwen + DS | bytes | expert-lane reads → ~18% (verified 18.3% DRAM at b1) → 3-5x lane cut; quality 94.5-98.1% retention | stack quality on IQ3 re-bake unproven — one A/B gates it |
| 2 | Specter INT4-self-draft + routing-lookahead prefetch (2511.14102) | Qwen (spec already on) | bytes+step | hides 40-60% of verify-side lane wait behind draft compute; lossless | K/chunk tuning; needs quantized self-copy on GPU tier |
| 3 | PuzzleMoE 2→1 merge + sign packing (2511.04805) | Qwen + DS pool | bytes | 25-50% fewer distinct experts → lane −25-50% at PPL +0.26-0.8; consumer-GPU numbers exist | merge at 512→256 shifts router semantics (apply B3 law) |
| 4 | QuantMLA C4R4 (2609.36760) | DS | bytes (KV) | KV ÷3.59 vs BF16 (≈1.8x beyond queued FP8), VRAM headroom at 262K, attention reads down; −0.13-0.26 pts | sm_120 kernel port (dequant-fused MLA decode) |
| 5 | JSSA (2610.19026) | DS | step (attn @262K) | attention KV reads inside 7.2 ms GPU term ÷5-30 at budget 2-4K; MLA-proven at DeepSeek scale | projected-key cache vs 1.2 GB free; kernel port |
| 6 | NatScale allocator (2610.20353) | both | bytes | −1-3% bytes at equal quality or quality headroom at fixed bytes; 512e/top-10 shape proven | none material (PTQ, cached stats) |
| 7 | Replacement-operator upgrade to REAP (2610.41366) | both | bytes | +tilt fill and keep-native-gate-scale on the queued prune+requant re-bake; prune+4bit > uniform 3bit at matched bytes | calibration domain sensitivity |
| 8 | Context compaction harness (2611.05306) | both | step (operational) | effective 262K→32-64K cuts attn/KV term 4-8x where task allows; accuracy held on agent benchmarks | per-model validation; one known collapse case |
| 9 | SpecPilot ideas: draft trees + two-level verify (2611.02501) | DS | step | verify-side GPU-term trims; sequences behind conditional CUDA graphs | infra (SGLang, paired-GPU validation) |
| 10 | SSA sampler (2610.28441) | DS | step | quality-at-extreme-sparsity backup if JSSA selection loses tail mass at 262K | sampler latency at 262K |

## E. Explicit non-surfacing notes (known/queued/falsified hygiene)
- Specter's prefetch is NOT the falsified "prefetch-guessing on flat routing": the signal is the draft model's own router outputs on the exact next K tokens (measured 90.8% top-4 match), not statistical reuse on flat histories.
- NatScale is NOT a REAP duplicate: REAP prunes experts; NatScale allocates BIT-WIDTHS across surviving experts; they compose (NatScale as the allocator pass of the same re-bake).
- EAGLE-3 / REST / draft-K gating / FP8 KV / CUDA graphs / MoE-Infinity / KTransformers items in the queued set were not re-surfaced; new-family members that extend them are marked as such (B5, B6, B8-substitute).
- Falsified items (sparsity gating, SMT, THP, TF32, flat-routing prefetch) do not appear in any verified candidate's mechanism.
