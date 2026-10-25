# T50 / R1 — Speculative-Decoding Research Dossier (2024–2026)

Researcher R1, 2026-10-25. Scope: spec-dec advances applicable to our RTX 5090 32GB / sm_120 box running Qwen3.8-Flash-Next (hybrid GDN/QSA linear attention, IQ3_S, MTP-style chain drafting, window T=6).

Our measured laws used throughout:
- `t/s = tau(T)/V(T)`; verify rows T=6; per-row acceptance 76–79% (M3b); tokens/window tau ≈ 2.3–4.8 depending on drafter.
- `GPU(T) ≈ 9.0–10.4 ms`, FLAT in T. `CPU(T) ≈ 4.72·T − 6.8 ms`. Host 0.2–0.5 ms. Drafter 3.5–4.6 ms/row.
- Standing records: solo 223 t/s (234 best), 6-conc 338 t/s, pair ~222. DAgger-retrained drafter: tau 1.65 vs stock 2.30 with depth-4/5 tail break.

## 0. Cost-law reconciliation (do this first — it prices everything)

- `CPU(6) = 4.72·6 − 6.8 = 21.52 ms`. Solo record period: `tau/V = 1000/223 → V = 4.48 ms per token`; at tau = 4.8 → window period = `4.8/223·1000 = 21.52 ms` — **exactly CPU(6)**. The solo 223 t/s record is therefore CPU-bound with GPU(6) ≈ 9.7 ms fully hidden under the CPU/drafter stream. Consequences: (a) t/s ≈ 46.5·tau at T=6 under the stock law; (b) tau raises and c_row cuts are near-additive levers; (c) GPU-flat means T growth is only priced in CPU rows.
- Law validation: `--spec 8` measured −22%. Law predicts `CPU(8) = 30.96 ms`, tau(8) ≈ 5.5–5.9 (tail acceptance decays) → 5.7/30.96·1000 ≈ 184 t/s = −17.5% vs 223. Matches the falsification within drafter-profile noise. The law is real; the intercept (−6.8) is per-window fixed cost, the 4.72 is irreducible per verify row *as currently hosted*.
- Open item for campaign: 6-conc 338 t/s implies ~85 ms per 6-request group at tau 4.8 vs serial-CPU 6·21.5 = 129 ms → ~34% of CPU already overlapped at concurrency; pair ~222 implies almost none. A one-boot probe (dm-wait-stats at conc 1/2/6) would pin the exposed-CPU fraction and re-rank P3/P5 precisely.
- tau from acceptance: chain tau(T) = (1−a^T)/(1−a). tau 2.30 → a ≈ 0.585; tau 4.8 at T=6 needs depth-1 acceptance ≈ 0.93 decaying down the tail — confirming the measured tail collapse is the tau binder.

---

## 1. Per-source findings

### 1.1 TreeWY: Speculative Verification for Gated DeltaNet Hybrids — arXiv:2608.20961 (Aug 2026) — CONFIDENCE: verified-from-paper (full-text fetched)
Ghantasala (Thomson Reuters). The only found paper on spec-dec verification for our exact architecture class (3:1 GDN:softmax hybrids, Qwen3.5-family targets).
- **Mechanism.** GDN recurrent states "cannot be truncated or partially rolled back the way a KV cache can," so chain/tree verify today snapshots k+1 full states per layer (4× committed state; 120 MiB/seq at 35B). TreeWY rewrites the gated delta rule as decayed additive attention with pseudo-values `ṽₜ = βₜ(vₜ − αₜSₜ₋₁kₜ)`, making all intermediate states recomputable as decay-weighted ancestor sums. Verification of a whole draft tree becomes ONE strictly lower-triangular linear system `(I + diag(β)G)Ṽ = R` solved by forward substitution (DFS pre-order node layout keeps G lower-triangular for any tree); on commit, only the accepted node's state is reconstructed. No snapshots ever taken.
- **Measured.** B200 (178 GiB), 35B TP1 / 397B TP8, vLLM fork, depth-3 MTP chain baseline. Memory: 2–3× lower peak KV at same load; 128× smaller per-head verify object. Throughput: up to 1.49× and ~40× lower p99 TTFT where memory binds (128-conc TTFT 683 ms vs 27,489 ms); 0.97–0.99× where memory is slack. Acceptance identical (mean |Δ|=0.039). Tree width: acceptance 3.24 (chain) → 3.58 at (3,3,3) — "trees enabled and correct, not a speedup." Store-all cost grows 4→40 state blocks with width; TreeWY stays at 1.
- **Relevance to us:** we already have the store-all approach (`STRATA_GDN_SLOTS` snapshot arena in verify.cpp). TreeWY is the drop-in theory for branch-safe GDN state commit if we ever branch drafts — and, more importantly for us, the snapshot arena we already run is the per-row machinery to reuse for 2-child fanout via snapshot/restore.
- Also cites (unverified leads): STree (Mamba2 WY transform), ReplaySSM (deferred state writes), Bole (Aug 2026, SGLang, independent concurrent derivation).

### 1.2 JetSpec: Breaking the Scaling Ceiling of Speculative Decoding — arXiv:2606.18394 (Jun 2026, Hao AI Lab) — CONFIDENCE: verified-from-paper (full-text fetched)
- **Mechanism.** A causal parallel tree drafter: one forward pass emits logits for many future positions under a tree-causal mask (each node sees prefix + its ancestors only), so branches are "branch-wise causal" — unlike DFlash's block-diffusion marginals which compose "individually plausible yet mutually inconsistent" branches. Best-first heap expansion by cumulative draft log-prob (entropy-only scoring collapses: 4.76× vs 8.15×), target verifies with tree attention, lossless. Drafter = 5-layer Qwen3-style decoder (32/8 heads, d=128) on fused hidden states from target layers {1,9,17,25,33}, forward-KL distillation on ~800K target-regenerated continuations (reverse KL was 36–46% worse).
- **Key framing (this is the paper's whole thesis):** speedup ≈ (1−α^(N+1))/((1−α)(N·c+1)) — you need BOTH high acceptance α and low per-row draft cost c. Drafting amortized to 0.85% of verify at budget 16, 0.054% at 256.
- **Measured.** H100/B200; Qwen3-8B dense AND Qwen3-30B-A3B (MoE). Budget 256: 9.64× (τ=10.76, MATH-500), 7.8–8.8× on math/code, 4.58× MT-Bench; MoE 9.45×. vLLM serving H100: 6.75× at batch 1 (968 vs 127.8 TPS) fading to 2.85× at batch 32 — budgets must shrink with load.
- **Relevance:** the (α, c) product law is exactly our situation (α tail-break plus c = 3.5–4.6 ms/row drafter + 4.72 ms/row verify-host). JetSpec says: fix the drafter's *branch-conditioning* and per-depth calibration, not just DAgger-replay it.

### 1.3 Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding — arXiv:2402.05109 (2024, MosaicML/Databricks) — CONFIDENCE: verified-from-abstract + secondary (HTML snippets)
- **Mechanism.** Per-depth draft heads where head d conditions on the tokens/embeddings proposed at depths 1..d−1 (sequentially dependent), replacing independent Medusa heads. Directly targets the deep-position acceptance collapse: independent heads' per-position acceptance "drops sharply at depths ~4–5"; sequential conditioning lifts it.
- **Measured.** Average acceptance length +0.46 tokens over Medusa-identical budget; Hydra++ (self-distillation) +1.31× further throughput. Hardware not quoted in abstract (A100-family in paper).
- **Relevance:** our depth-4/5 tail break is the same pathology; our drafter is already an autoregressive chain (mtp_draft loops steps), so the untried axis is *training-side*: per-depth loss weighting + explicitly conditioning on accepted-prefix embeddings à la Hydra++ rather than plain DAgger.

### 1.4 EAGLE-3 (and EAGLE-2 context) — arXiv:2503.01840 (Mar 2025) — CONFIDENCE: mechanism verified-from-abstract/HTML; numbers blog-corroborated
- **Mechanism.** Draft = one extra decoder layer trained on multi-layer target features (low/mid/high fused) predicting next *feature*, then the target's own LM head maps to tokens; EAGLE-2 adds dynamic draft trees sorted by draft-model confidence. 5–6.5× over AR on benchmarks.
- **Relevance:** (a) EAGLE-2's confidence-sorted tree reshaping at constant node budget is the cheapest tree win and needs no new drafter; (b) multi-layer feature fusion is a known +acceptance lever if we ever retrain our drafter head; (c) P-EAGLE (AWS blog, unverified) reports 1.69× over vanilla EAGLE-3 by *removing sequential drafting* — independent evidence that the sequential draft loop is the thing to attack.

### 1.5 Sequoia — arXiv:2402.12374 (NeurIPS 2024) and SpecExec — arXiv:2406.02532 (NeurIPS 2024) — CONFIDENCE: abstract-verified
- **Sequoia:** DP algorithm computes the *optimal draft tree for a given hardware accept-rate profile*; hardware-aware optimizer picks tree size/shape per consumer GPU (70B on one RTX 4090). Decouples tree topology from drafter.
- **SpecExec:** massively parallel verify of a large (hundreds-of-nodes) token tree on consumer devices; deterministic-verify variant with quantized-target tolerance.
- **Relevance:** their core result — at FIXED node budget, DP-shaped tree strictly dominates chain when per-node draft probabilities are known. On our box the budget is 6 rows (CPU-priced), so the move is reshape-not-extend (extending is already falsified: `--spec 8` −22%, lookup-chain killed by 4.72 ms/row).

### 1.6 Dynamic-length family: AdaEDL (arXiv:2410.18351), SVIP (EMNLP 2025, Geralt-Targaryen/SVIP), SpecDec++ (ICML 2024), DySpec (arXiv:2410.11744) — CONFIDENCE: abstract-verified (AdaEDL full-text fetch timed out)
- **AdaEDL:** training-free early draft stopping via an entropy-derived *lower bound on next-token acceptance probability*; stop extending the draft when the bound falls under a threshold. 10–57% over static draft length; up to +10% over other training-free stoppers.
- **SVIP:** draft model self-verifies its own continuation to decide length; training-free, plug-and-play, aimed at long-context where fixed length wastes either rejects or potential.
- **SpecDec++:** trained stopping head predicting whether to continue drafting (needs training, quality-taxed drafter).
- **DySpec:** hypernetwork-style dynamic token-tree construction conditioned on context (tree topology per step, not per window).
- **Relevance:** all four are *per-window dynamic* policies, categorically different from our falsified STATIC depth gates (global, context-blind). The dead static gate does not falsify entropy-conditioned stopping; but the A/B must be run explicitly.

### 1.7 TriForce — arXiv:2404.11912 (2024) — CONFIDENCE: abstract-verified
Hierarchical speculation for long context: retrieval-sparse KV as level-1 drafter, full model verifies. Not directly applicable at our context lengths; kept for the pattern "cheap statistical drafter under an expensive verifier" (maps to our ngram.hpp drafter as a second-tier proposer).
**InfoCal / "calibration-free acceptance estimation": NOT FOUND** on arXib or the web under that name. Nearest real things: Calibrated Speculative Decoding (arXiv:2604.13634, frequency-guided training-free token recovery, unverified-lead), AdaSpec (confidence-based acceptance estimators, unverified-lead), vLLM "Adaptive Verification" blog line (see 1.9). Grade: unverified — treat any InfoCal citation as suspect.

### 1.8 MoA: Mixture of Attentions for Speculative Decoding — arXiv:2410.03804 (Zimmer et al., UCL/Huawei) — CONFIDENCE: abstract-verified (note: often miscited as 2502.13340)
Grounded mixture-of-attentions drafter (multiple receptive fields) fixing draft partial-observability; also addresses low arithmetic intensity of small-batch inference. Ideas for drafter v2 conditioning; no hardware numbers usable for our costing.

### 1.9 Production sources
- **SGLang / LMSYS MTP blog (2025-07-17, lmsys.org/blog/2025-07-17-mtp)** — CONFIDENCE: verified-from-blog (full-text fetched). DeepSeek V3 on H200 decode nodes: baseline 51.0 tok/s/rank → +20.4% with overlap scheduling alone → MTP 3-token window 81.5 (+59.8%) → 4-token window 82.0 (+60.8%); acceptance length 2.44 @4-token vs 2.18 @3-token (topk=1). **At 128 concurrent requests/rank the MTP gain collapses to +14.2%** with draft_token_num=2. Tuning rule published: "if average acceptance stays above 2, try longer drafts; if it drops, reduce." MTP did NOT compose with overlap scheduling at blog date. Direct lessons: (a) our 76–79% per-row acceptance is already in the regime where draft-token growth pays less than acceptance quality; (b) concurrency hides spec-dec gains — matches our pair/6-conc compression.
- **Red Hat vLLM engineering article (developers.redhat.com, Apr 2026, GPT-OSS)** — CONFIDENCE: blog-claim with concrete numbers. Per-depth acceptance decay measured in production: 45.4% marginal acceptance at 2 draft tokens → 28.3% at 4; verify overhead explicitly named as the reason deeper chains underdeliver. Same tail-decay shape as our depth-4/5 break; independent evidence the tail is where the money is.
- **Snowflake Arctic "Fastest Speculative Decoding in vLLM" (May 2025)** — CONFIDENCE: blog-claim. Dynamic speculative decoding = acceptance-rate-aware per-load tuning of draft length (their 33–84% gains come from retuning per traffic class). Supports P4/P5 framing: length policy should be load-aware, not global.
- **vLLM spec-dec docs + "Adaptive Verification / DSpark confidence-scheduled verification" (vllm.ai blog index, 2025–2026)** — CONFIDENCE: blog-claim (not fetched in full). Runtime acceptance-rate estimation driving verification schedule; plus vLLM AMD blog (Aug 2026) standardizing AR/MAL metrics.
- **DeepSeek MTP production line (V3 tech report + deployments)** — CONFIDENCE: blog-claim. ~1.8× decode TPS from MTP speculation; SGLang/AMD ROCm and DigitalOcean Blackwell deployments; single MTP layer, 2-token draft; documented training–inference discrepancy tax on acceptance (GLM-5 paper note) — the exact DAgger-style failure we hit.
- **CPU/GPU Heterogeneous Speculative Decoding (arXiv:2412.18934, EMNLP 2025)** — CONFIDENCE: title/abstract-lead only (unverified). Pipelines CPU drafter against GPU verifier — the same hide-the-CPU trick our pipeline-windows plays; worth a full read before P3 implementation.
- **llama.cpp**: no merged first-class tree spec-dec; prompt-lookup/n-gram remain the production path (unverified-lead; not load-bearing for any proposal).

---

## 2. PROPOSALS (ranked by expected value; all lossless unless stated)

Cost frame: T=6, t/s ≈ 46.5·tau under stock CPU law; V(6) = CPU 21.52 + GPU 9.7 (hidden) + host 0.35. Marginal verify row = 4.72 ms; marginal drafter row = 3.5–4.6 ms; FastMTP (queued) moves c_row 4.72 → 3.56 and is treated as banked/orthogonal — every proposal below is priced WITHOUT FastMTP and again with it.

### P1 (rank 1) — Tail-Causal Drafter v2: per-depth sequential heads + branch-causal conditioning + depth-balanced loss (Hydra++ × JetSpec, training-side)
- **What physically changes in a window:** nothing structural — same T=6 chain, same rows, same V(T). Only the drafter's depth-4/5 proposals get good, so tau rises. Depths 4–5 stop breaking early; fewer windows terminate at depth 2–3.
- **Derivation:** DAgger frame: tau 1.65 → 2.30+ is +39% t/s (76.7 → 107+). Best-drafter frame: 76–79% mean acceptance with tail collapse → per-depth heads + depth-balanced CE (weight ∝ 1/P(reach d)) historically buy +3–5 acceptance points concentrated in the tail → tau 4.8 → 5.4–5.6 → t/s 251–260 (+13–17%). With FastMTP banked (period ≈ 14.6–16.3 ms): 5.5·1000/15.5 ≈ 355 t/s. JetSpec's evidence: conditioning quality, not replay alone, is what keeps α high as depth grows (entropy-only or branch-agnostic objectives plateau or invert).
- **Risk / quality tax:** none at output level (verification is lossless; drafter only changes speed). Real risk = a second 1.65-style regression; gate the swap on a stock-drafter tau A/B before promoting. Training cost: tools/mtp_train.py exists; one retrain run.
- **Cheapest falsifier (one probe + one code read):** parse the existing per-depth acceptance logs — if depth-1 acceptance > 0.85 while depth-4/5 < 0.3, tail training is the binder. Code read: mtp.cpp `mtp_draft` early-exit — determine whether tails die by min_p/EOS cutoff vs draft-prob collapse (if it's a cutoff bug, the fix is 5 lines, and the retrain is unnecessary).
- **Port effort:** tools/mtp_train.py loss + feature-conditioning change only; engine untouched (mtp.cpp interface unchanged). ~1–2 days. Touch points: `/home/svg/LLM/strata-custom-2/tools/mtp_train.py`, optionally `tools/mtp_bridge.py` for feature taps.

### P2 (rank 2) — Budget-neutral 2-way tree at the depth-2 node (EAGLE-2 confidence sort; GDN commit via existing slot arena)
- **What physically changes:** same 6 verify rows, same CPU(6) = 21.52 ms. Rows 3–6 become two contesting continuations of the depth-2 token instead of one chain; verify picks the best accepted path. GPU stays flat (rows unchanged; QSA rows need a 2-branch mask; GDN rows run child A then child B from the parent's `STRATA_GDN_SLOTS` snapshot — restore is a device memcpy per layer, off the CPU law).
- **Derivation:** at fixed node budget, shaped trees beat chains (Sequoia DP; EAGLE-2). Analogous measured tree-vs-chain acceptance gain: TreeWY 3.24 → 3.58 = +10.5% at width 3. Conservative: tau 4.8 → 5.2–5.35 (+8–11%) → t/s 241–249 (+8–11%). With FastMTP: ≈ 330–345. Note this is NOT the falsified lookup-chain extension: row count is constant; only topology changes. And it is NOT free-T: every extra node past 6 is priced at 4.72 ms and stays dead per our own law.
- **Risk / quality tax:** none (lossless). Engineering risk: GDN slot restore parity and QSA window-ring behavior under branching; tree-mask in verify batch kernels. Medium-high complexity — this is the largest code slice.
- **Cheapest falsifier (no boot, no C++):** offline replay — from existing draft-prob logs compute tau(chain, 6) vs tau(DP-optimal 6-node tree, then tau(2-way fanout at depth 2)). If offline tau gain < 5%, kill before writing C++. One python probe.
- **Port effort:** `/home/svg/LLM/strata-custom-2/src/core/verify.cpp` (branch mask + slot restore path; kVerifyMaxT=8 headroom already OK), `src/kernels/cuda/sampler.cu` + `src/kernels/verify_batch_parity.cpp` (tree-aware expansion/parity), `src/core/mtp.cpp` (draft fanout). 3–5 days.

### P3 (rank 3) — Host-free inner window: move per-row accept/commit/resample into the GPU (kill the 4.72 coefficient at its root)
- **What physically changes:** the window's inner loop stops round-tripping to the host per row: acceptance compare, commit selection, and bonus-row resample execute as a chained GPU section (sampler.cu), host touches only window entry/exit and the per-window commit vector. The CPU law's slope, not just its intercept, collapses.
- **Derivation:** c_row 4.72 → 0.6–1.2 ms ⇒ CPU(6) 21.52 → 3.6–7.2 ms + host. Period becomes GPU-bound: tau 4.8 / (9.7 + 0.35 + residual) → 449–475 t/s ceiling; realistic 380–450 (+70–100%). Post-FastMTP the same move stacks (3.56 is a better CPU constant; ~0.9 is no CPU constant at all). This is the only proposal whose ceiling clears the +30–50% goal alone; FastMTP at 3.56 ms/row still leaves CPU(6) ≈ 14.6–16.3 ms — banked but insufficient.
- **Risk / quality tax:** none at output parity if the accept math is bit-matched (parity harnesses exist). Highest engineering risk: graph capture across MoE CPU-tier expert dispatch (`remote_expert_opt.cu`) — if part of the 4.72 ms is actually CPU-tier expert GEMV on the 9950X3D rather than host overhead, this proposal mis-attributes and only the host share is winnable.
- **Cheapest falsifier (one boot + one code read):** boot with `STRATA_DM_WAIT_STATS=1` and read the waitCPU/waitB decomposition in verify.cpp (lines ~438–468) — it already separates "host plan" vs "CPU-rows" wait per layer. If waitCPU is dominated by host-plan microseconds rather than expert compute, the coefficient is host-side and movable. Code read: the per-row `for (t < T)` dispatch loops in mtp.cpp (~lines 979–1243) to inventory what actually serializes.
- **Port effort:** `src/kernels/cuda/sampler.cu` (GPU accept/resample chain), `src/core/verify.cpp` (handshake removal; dm-wait-stats counters give before/after for free), `src/core/mtp.cpp` (capture). 4–6 days. Pairs naturally with the queued FastMTP work — same capture scaffolding.

### P4 (rank 4) — AdaEDL/SVIP entropy-gated early stop (dynamic, training-free; explicitly NOT the falsified static gate)
- **What physically changes:** per window, the drafter keeps a running entropy of its own next-token distribution; when the entropy-derived lower bound on acceptance falls below θ at depth d, drafting stops and the verify window is built with T = d (variable T is already legal: verify.cpp enforces 2..kVerifyMaxT=8). Doomed rows are never drafted (−3.5–4.6 ms each) nor verified (−4.72 ms each).
- **Derivation:** fires only in bad windows. Weak-drafter frame (tau 1.65, tail dead): gate fires at depth 3–4 in a large fraction of windows → skip ~2–3 rows → period 21.52 → ~12–14 ms, tau 1.65 → ~1.5 → 76.7 → 110–125 t/s (+45–60% on that drafter). Best-drafter frame: fires rarely; +5–10%. Break-even rule for keeping row d: expected chain tokens at d ≥ 4.72 ms / period-per-token — under the stock law that is Δτ > 2.30·4.72/21.52 ≈ 0.50 tokens per added row at T=6.
- **Risk / quality tax:** none (skipped rows would have been rejected anyway — lossless by construction). Risks: θ mis-tuned → over-truncation (tau loss > row savings); must A/B against the dead static gate to prove dynamism is what pays, and against no-gate at both drafter frames.
- **Cheapest falsifier (offline, one probe):** from existing logs, correlation/AUROC of drafter entropy at depth d vs realized acceptance at depth d. AUROC < 0.65 → kill without touching the engine.
- **Port effort:** `src/core/mtp.cpp` draft loop + window build (T variable). ~1 day.

### P5 (rank 5, compounder) — Concurrency-domain window batching (verify_batch path), unlocked by P3
- **What physically changes:** 2–6 requests share one verify window (kVerifyMaxT=8 caps composition; e.g., 2×4 or 3+3+2 rows). GPU is flat in rows, so request 2..6's rows are near-free on GPU; the CPU law does NOT amortize (4.72 is per row) — so this only pays after P3 cuts the slope, or as conc-domain scheduling with P4 truncation.
- **Derivation:** today pair ≈ 222 t/s ≈ solo → no GPU parallelism is being exploited; 6-conc 338 t/s implies only ~34% CPU overlap (Sec. 0). With P3's c_row ≈ 0.9: 6-conc CPU(36 rows) ≈ 32 ms per group of ~6·tau tokens → GPU-bound ceiling 6·4.8/10.4 ms·1000 ≈ 2,770 t/s theoretical, realistically 500–700 t/s (stream serialization, KV ring capacity ×6, TTFT fairness). Under the current un-P3'd law, batching predicts NO gain — that asymmetry is itself a clean experiment.
- **Risk / quality tax:** none per-token; latency-tail and ring-capacity risks at 6-conc.
- **Cheapest falsifier (one boot):** run the 6-conc bench at T=4 vs T=6. If t/s is ~flat in T, the GPU binds (batch now); if it scales with T, CPU binds (do P3 first).
- **Port effort:** scheduler/window builder in `src/core/verify.cpp` + `tests/core/dma_batch_parity.cu`/`verify_batch_parity.cpp`. After P3, 2–3 days.

### Ranking rationale and composite
1. **P1** — biggest tau move per unit effort; robust under every accounting regime; engine untouched.
2. **P2** — cheap to falsify (offline), +8–11% band with measured analogues; largest code slice though.
3. **P3** — the only single-lever path past +50%; gated on the dm-wait-stats decomposition probe.
4. **P4** — training-free insurance, strongest on the weak drafter; pairs with P1's retrain cycle.
5. **P5** — compounder on P3; decides whether the goal is met in solo or served t/s.

Composite (independent of queued FastMTP, which stacks on top): P1 + P4 → 223 → 280–320 t/s (+26–43%); P1 + P4 + P2 → 300–345 (+35–55%). With FastMTP banked beneath: 355–420 t/s. The +30–50% campaign goal is reachable in this lane alone via tau recovery (P1) plus row-discipline (P4), with P3 as the stretch multiplier and P2 as the offline-cheap kicker.

## 3. Explicitly not re-proposed
Static depth gates (falsified — P4 differs: per-window entropy bound, must A/B the two), `--spec 8` (law-consistent −22%), lookup-chain extension (4.72 ms/row kills any row-count growth), draft temp 0.85, min(1,p/q) acceptance, typical-acceptance rescue (adopted), SPEC_COST port / FastMTP / ECHO budget (queued — priced around, not duplicated). InfoCal does not exist under that name; do not chase citations to it.
