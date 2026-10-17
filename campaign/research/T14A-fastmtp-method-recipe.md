# T14A — FastMTP-class MTP drafter retrain: the method literature and a concrete retrain recipe for OUR single-layer block

Date: 2026-10-16 (evening round; supersedes nothing — extends T11.R3's GO verdict in `SVG-CUSTOM-TODO.md` and the T7.B.1 entry).
Scope: (A) the 2024-2026 MTP/drafter-training literature, distilled to the TRAINING-RECIPE variables that move offered chain depth and per-depth acceptance; (B) mapping onto our exact drafter (Qwen3.8-Flash-Next's own MTP layer, 31 tensors / 5.214 GB BF16, sub-vocab bound head, coupled sampling, `--spec 6`, engine = strata v0.1.44.1); (C) a costed retrain recipe for one RTX 5090, ranked by (expected acceptance gain)/(training cost), with risks. READ-ONLY on the repo; live server untouched; /mnt/DATA untouched.

Verified-rig facts this file builds on (all confirmed by code read this session):
- Drafter = the model's own MTP layer: `mtp.fc_embedding` (2560x2560) + `mtp.fc_hidden` (2560x2560), `pre_fc_norm_{embedding,hidden}`, one self-attn block (with light indexer) + 512-expert MoE + shared expert + 3 hyper-connection mixers — exactly 31 tensors, 5.214 GB BF16 in 28 shards (`/home/svg/LLM/Custom-data/mtp/mtp-inventory.md`). NO own head, NO own embedding table: the token embedding is gathered from the target's table and the draft head is the TARGET head's 40,525-row sub-vocab subset, bound at load (`mtp.cpp` `bind()`, `tools/draft_vocab.py` subset, Q4_0 under `--mtp-q4`, full-head diagnostic `STRATA_MTP_FULL_HEAD=1`).
- Chain: draft j feeds back its own sampled token's embedding + its own output R state; verified window rows judge drafts by exact match against the target's Philox pick (coupled, `coupled_draft.hpp`) or by min(1,p/q) rejection (prob mode, `spec_prob.hpp`); the chain gate re-rolls under the drafter's own chain (same sampler chain, temp x STRATA_SPEC_DRAFT_TEMP=0.85) and a further draft is offered only while the drawn token survives `--spec-min-p 0.5` under q (`spec_prob.hpp` header; mtp.cpp:1697 per T13A).
- Measured (our rig, F-228/T13A): ~75-78% serial per-proposal acceptance at offered depths; offered chain only ~2.2 of 6 (the min-p gate under the drafter's own confidence is what trims it — "the min_p stair, not T=6"); marginal acceptance at depths 3/4/5 = 85/68/75% — depth is profitable at the margin; the offered DEPTH is confidence-limited. Window economics (T13A): ~11.5 ms window, ~2.47-2.9 committed tokens, each extra accepted row costs ~0.55 ms and returns a full token.
- Trainable-vs-frozen split: the 512 routed experts + shared expert are ~5.05 GB of the 5.214 GB (~2.55B params); the DENSE part (fc projections, mixers, attention, indexer, router) is only ~196 MB (~98M params) — this is the cheap first training surface.

---

## 1. Summary

FastMTP (arXiv 2509.18362) and its production form (vLLM-project `speculators`, Red Hat article 2026-09) show that the depth collapse of a vanilla MTP drafter is a TRAINING-OBJECTIVE artifact, not an architecture limit: a vanilla single MTP head is trained only for depth 1 (predict t+2 from the target's hidden at t) and never sees its own recursion, so its depth-2/3 acceptance collapses (70% -> 11% -> 2% on MiMo-7B). Fine-tuning ONLY the MTP layer, with a recursive teacher-forced objective over K depths, on data drawn from the TARGET MODEL'S OWN output distribution, recovers depth: 70/11/2 -> 81/56/36 per-step acceptance, mean speedup 1.21x -> 2.03x (+82% relative), with the base model frozen. The Red Hat productionization did this on the Qwen3-Next/Qwen3.5 head class — OUR head class — with 5-8K samples in ~443 s on 2x H200, moving per-position acceptance 0.897->0.912 / 0.719->0.776 / 0.476->0.616.

No architecture change is implied for us: our block already IS a single shared-weight recursive head served with a chain (`--spec 6`). Everything that matters is training-side: (1) recursive multi-depth CE with exponential depth weights, (2) K >= serve depth, (3) target-distribution data (self-distilled), (4) train against the SERVING stack's hiddens (ours = the quantized IQ3_S target via our own engine, a distribution advantage RH/FastMTP did not have — they trained against bf16 vLLM hiddens), (5) keep the frozen bound sub-vocab head and the target frozen. The literature's other levers — direct acceptance-overlap losses (LK, Feb 2026; +3.5-8.2% tau over KL), selective hard-token filtering (AdaSPEC, up to +15% acceptance), distillation temperature matching (0.7 > 0.5), EAGLE-3-style training-time self-rollout — are all bolt-on training-objective/data variants, not engine changes. The genuinely out-of-reach items are EAGLE-3 multi-layer feature fusion (needs intermediate-layer taps in the engine) and GLM-5-style N-layer MTP (engine supports exactly one MTP layer; but GLM-5's SHARED-parameters trick shows our single recursively-applied layer is the same family, just deeper-chained).

Calibrated expectation for our rig (R3 decision math, consistent with the literature deltas): RH-scale retrain ~+12-22 t/s (205-216 -> ~227); FastMTP-scale data/compute -> ~237; depth re-unlock (offered chain 2.2 -> 3.2-4) -> 247-255 t/s = the 250 path. Total cost: ~1-2 days engineering + a weekend of compute on the 5090.

---

## 2. Findings by confidence

### HIGH (primary source, read in full, uncontradicted)

**F1. FastMTP method and full recipe** [arXiv 2509.18362v1, full HTML read, 2025-09-16, single version].
- Loss: `L = sum_{k=1..K} alpha_k * CE` over ALL positions (`1+k..T+1`), `alpha_k = beta^(k-1) / sum_j beta^(j-1)`; used beta=0.6, K=3. The head runs RECURSIVELY: step k consumes the head's own step-(k-1) hidden + the ground-truth token embedding shifted by k (teacher-forced recursion — the hidden path is the head's own computation, the token path is ground truth).
- Base model FROZEN; only the MTP head trained (210.8M params, <3% of the 7,833M backbone, MiMo-7B-RL dense).
- Data: SELF-DISTILLED — all responses generated by the target itself (temp 0.6, top-k 20, top-p 0.95, max 4096 tok, SGLang) from instruction prompts; MinHash dedup + filtering; 389.4K samples; mix ~42% general / 18% math-reasoning / 13% code / 27% Chinese.
- Hyperparams: 3 epochs, cosine, peak LR 5e-5, warmup 0.05, AdamW(0.9, 0.95), global batch 64, ms-swift, under 1 day on one H20 server.
- Results: per-step acceptance 70->81 / 11->56 / 2->36 (vanilla collapses beyond step 1); speed peaks at K=3 (tau still rising to ~3.2 at K=3); self-distilled beats fixed (prompt,response) pairs 1.81x vs 1.67x; 2.03x mean vs 1.21x vanilla (+82%).
- Ablations: depth-K optimum K=3; language-aware vocab compression adds ~16% throughput (16-32K vocab optimal; applied ONLY at drafting, full vocab at verify -> lossless).
- Architecture: single shared-weight head applied recursively (vs DeepSeek-V3's cascaded independent modules); otherwise DeepSeek-V3 MTP architecture; EAGLE-style recursive drafting + parallel verify, provably lossless.

**F2. Red Hat / vLLM speculators production form on OUR head class** [developers.redhat.com article 2026-09-08; docs.vllm.ai speculators MTP page; github.com/vllm-project/speculators README].
- Verified models: Qwen3-Next and Qwen3.5 families including MoE — the same family as our target's drafter. Requires the verifier to ship native `mtp.*` weights (ours does).
- Method: recursive teacher-forced loop, depth = serve steps; per-step loss weights = normalized exponential decay, default beta=0.6 over K=3 -> [0.51, 0.31, 0.18]; `embed_tokens` and `lm_head` FROZEN and shared with the verifier (matches our bound head exactly); only `mtp.*` tensors train.
- Data: "MTP requires training data that the target model itself generates"; hidden states extracted from a live serving stack (vLLM), saved offline; the draft head is small (~100-400M) so training runs on pre-extracted hiddens WITHOUT loading the full verifier. Explicit warning in the article: their example was domain-narrow (GSM8K ~8K samples) — "longer training might yield a better speculator."
- Recipe example (Qwen3.5-9B): --max-samples 5000, --seq-length 8192, 3 epochs, LR 1e-4, K=3, beta=0.6; ~443 s on 2x H200.
- Measured (Qwen3-Next-80B-A3B-Instruct, GuideLLM): pos0 0.897->0.912, pos1 0.719->0.776, pos2 0.476->0.616 — biggest gain at the deepest position, exactly the deficit our rig has.

**F3. DeepSeek-V3 MTP design (the architecture our block descends from)** [arXiv 2412.19437v2 full HTML].
- D=1: one module, shared embedding + shared output head, input = M[RMSNorm(h); RMSNorm(Emb(t+1))] -> one Transformer block -> shared head. (Our block: same shape with two separate projections fc_hidden/fc_embedding and the Qwen3-Next 4-stream hyper-connection R state, HC=10240.)
- Loss: per-depth CE averaged over depths, weight lambda=0.3 for the first 10T tokens then 0.1 for the remaining 4.8T; trained JOINTLY with the backbone over 14.8T tokens, seq 4K, batch ramp 3072->15360. (We cannot redo pretraining; our lever is post-hoc head retraining, which FastMTP/speculators show is sufficient.)
- Inference: MTP speculative decoding reported at ~1.8x decode speedup; acceptance 85-90% widely quoted for the single extra token (the 85-90% figure is SECONDARY — see contradictions).
- Ablation (Table 4, module dropped at inference): consistent eval gains, largest on code (HumanEval 20.7->26.8 small, 44.5->53.7 large) — MTP pretraining helps code most; relevant to our domain weighting.

**F4. Gloeckle et al. 2024 (original MTP)** [ar5iv full text of arXiv 2404.19737].
- Loss: CE at ALL n future positions simultaneously, n independent output heads on a shared trunk, shared unembedding; applied at every training position. Parallel, NOT recursive — that is why it cannot be chained naively at depth.
- Scale: 0.3B-13B, 91-209.7B tokens (scaling study), 7B on 200B-1T tokens.
- Self-speculative decoding: greedy blockwise parallel decode, ~2.5 of 3 suggestions accepted on code, ~3.0x on code / 2.7x text at batch 42, speedup ~constant across batch sizes.
- Negative result that matters for us: multi-token loss on an ALREADY-PRETRAINED model (Llama 2 finetune) "did not yield significant improvements" — abrupt loss changes on a frozen trunk don't transfer. The FastMTP/RH answer is that retraining the HEAD (not the trunk) against the trunk's actual outputs does; the trunk here is the frozen 125B, the head is the MTP layer — no trunk loss change involved.

**F5. EAGLE-3 (the strongest external drafter, and what its recipe implies)** [arXiv 2503.01840v2 full HTML].
- Central diagnosis: the drafter's test-time inputs are its own estimates -> error accumulation; vanilla EAGLE's per-depth acceptance decays 1-alpha..5-alpha, EAGLE-3's stays flat. Fixes: (a) "training-time test" — feed the draft model's OWN output estimates back as inputs during training with modified attention masks (self-rollout simulation); (b) drop the feature-prediction loss, train token-level only; (c) fuse LOW+MID+HIGH layer features (3k -> FC -> k) because top-layer features only encode the next token.
- Data: ShareGPT (~68K entries) + UltraChat-200K (~464K), target-generated; ~8x EAGLE's data. LR 5e-5, AdamW, grad clip 0.5.
- Results: tau 6.62 (Vicuna-13B, T=0) vs 4.83 EAGLE-2; trees at inference (chains only for the throughput tests).
- Mapping: (a) and (b) are training-data/objective work we can adopt; (c) needs intermediate-layer taps in OUR engine — deferred (see B).

**F6. Medusa (the head-based baseline; recipe reference points)** [arXiv 2401.10774v3 full HTML].
- K heads (5 recommended), single-FFN heads, W2 initialized from the LM head, W1 zero; loss = weighted CE with lambda_k = 0.8^k (the exponential depth weighting predates FastMTP); Medusa-2 adds KL distillation from the frozen teacher via LoRA; head LR 4x backbone LR; ~5 h on one A100 for 7B. ShareGPT 60K; self-distillation ~100K model-generated samples for 33B/Zephyr.
- 2.3-2.8x speedup class. Dominated by MTP-style drafters for our purposes (T7 verdict stands) but it is the origin of the depth-weighted CE and the "teacher-KL optional, plain CE fine" finding.

**F7. LK Losses: direct acceptance-rate optimization** [arXiv 2602.23881v1, Feb 2026, full HTML read — the current loss-form frontier].
- Theory: acceptance alpha = sum_x min(p,q) = 1 - TV(p,q); KL and alpha share the global optimum q=p but capacity-limited drafts converge to points where lower KL coexists with LOWER acceptance (toy: fwd-KL alpha 50.2%, reverse-KL 50.8%, TV-optimal 60.2%). Pure TV is untrainable from scratch (vanishing gradient ~O(sqrt(k)/V), sign-only) -> two practical losses: hybrid `L = lambda*KL + (1-lambda)*TV` with adaptive lambda = exp(-eta*sg[alpha]) (KL early -> TV late), and `L = -log alpha`.
- Data: 660K prompts from Infinity-Instruct, responses generated by EACH target; batch 64, LR 4e-4, cosine, 100 warmup, 10 epochs (1 for the MTP arm), T=1.
- Results vs KL-trained drafts at T=1: tau +3.5-8.2% across 6 targets; DeepSeek-V3 MTP finetune +5.6%; largest for low-capacity drafts and high mismatch — our regime. LK losses also handle truncated draft vocabularies naturally (off-vocab contributes 0 to alpha; KL blows up) — directly our 40,525-row sub-vocab head.
- Evaluation hygiene they prove: greedy draft sampling systematically UNDERESTIMATES acceptance at T>0 — our coupled sampling sidesteps this by construction.

**F8. AdaSPEC (NeurIPS 2025; the cheapest distillation win)** [arXiv 2510.19779v1 full HTML].
- Two-stage selective KD: a reference draft (same size) is distilled to convergence first; per-token loss GAP identifies learnable-but-unlearned tokens; the real draft trains ONLY on the top-k fraction (k=0.4) — hard-to-fit tokens are filtered out. Forward KL retained (reverse-KL and TVD ablate worse here, in tension with F7 — see contradictions).
- Results: acceptance up to +15% over DistillSpec (49.88->65.12% MBPP best case), 10-20% wall-clock, LESS total training than the KD baseline; gains grow with draft-target gap. Cost adder: one reference-model training pass.

**F9. GLM-5: multi-depth MTP with shared parameters** [arXiv 2602.15763v1 full HTML, Feb 2026].
- Statement of our exact problem, from a frontier lab: for DeepSeek-V3's single module, "the training-inference discrepancy reduces the acceptance rate of the second token." GLM-5 trains 3 MTP layers with SHARED parameters — accept length 2.76 vs DeepSeek-V3.2's 2.55 at the same 4 spec steps. The sharing trick keeps memory flat while making depth-2/3 training real. Our single layer applied recursively is the parameter-shared degenerate case of this design.

**F10. Distillation lineage** [DistillSpec arXiv 2310.08461 (ICLR 2024); Online Speculative Decoding arXiv 2310.07177 (ICML 2024)].
- DistillSpec: logit-level KD of draft from target, divergence function matched to the sampling regime, 10-45% speedup over SD baseline. Establishes "teacher-signal choice is a first-class recipe variable."
- OSD: static-corpus drafts suffer query-distribution mismatch; continuously adapting the draft on target outputs yields +0.1-0.65 acceptance, 1.22-3.06x latency wins. Establishes "data must track the SERVED query distribution" — for us, our own prompt-replay corpus.

### MEDIUM (multiple independent credible sources)

**F11. Vanilla-MTP depth collapse is universal, and depth-2 serving is the shipped default** — FastMTP measures 70/11/2 (MiMo-7B); GLM-5 states the mechanism for DeepSeek-V3; the vLLM official Qwen3-Next recipe ships `num_speculative_tokens: 2` (docs.vllm.ai recipe + `qwen3_next_mtp` method, `--no-enable-chunked-prefill`) — i.e., the stock Qwen3-Next ecosystem does NOT trust depth >2 on the vanilla head. Our T7.B.1 note (GLM-4.5-Air head 0.92/0.68/0.38) is consistent in shape but NOT primary-verified (see contradictions).

**F12. Qwen3-Next MTP design points** — vLLM blog (2025-09-11): native MTP support, full CUDA-graph decode; Qwen blog: "high acceptance rate" MTP module, ~2-3x decoding speedup claim (secondary, blog-level). Design follows the DeepSeek-V3 lineage: separate draft layer predicting future tokens with auxiliary loss during pretraining (vLLM blog + NVIDIA Megatron-Bridge framing + Raschka's architecture gallery as corroborating secondary sources). No standalone Qwen3-Next technical report with MTP training detail exists as of today (searched; the Qwen3-Coder-Next report arXiv 2603.00729 deliberately omits architecture internals — verified by full-text fetch).

**F13. Our expected-gain calibration** — R3 decision math in `SVG-CUSTOM-TODO.md` (our own measurements + the literature ratios): RH-scale acceptance deltas -> ~227 t/s; FastMTP-scale -> ~237; depth re-unlock -> 247-255. Independent cross-check from T13A window economics: 250 t/s needs ~+17% tokens/window at fixed window time (2.47->2.9), and each extra OFFERED+accepted row nets a full token for 0.55 ms — a retrained drafter that re-lengthens the offered chain to 3.2-4 clears the bar arithmetically. These agree within ~10%.

### LOW (single source, or derived — flagged)

**F14. beta=0.8 with K=6-8 for OUR run** (R3 adaptation, derived): FastMTP's beta=0.6 over K=3 puts only [0.41, 0.246, 0.148, 0.089, 0.053, 0.032] on depths 1-6 — too bottom-light to retrain depths 4-6, which is precisely where our marginal acceptance (68-75%) and offered-chain deficit live. beta=0.8 over K=6 normalizes to [0.271, 0.217, 0.173, 0.139, 0.111, 0.089] — depth 6 keeps ~9% of the loss. No published run uses this; it is an extrapolation of FastMTP's own weighting rule. Arm it (beta 0.6 vs 0.8) rather than assume.

**F15. Coupled-sampling-specific objective** (derived, no source): our coupled exact-match acceptance per row is the collision probability sum_x p(x)*q(x) (both sides draw from the SAME uniform; identical picks iff same argmax over the CDF — the agreement probability is the L1 overlap of the two post-chain distributions' picks; to first order the collision term sum p*q dominates). F7's TV objective targets sum min(p,q) (rejection-sampling alpha). Both are "overlap" objectives, but the coupled form is not published. Practical consequence: an LK-style arm should include BOTH forms as variants; the plumbing (target top-64 p-lists logged from the verifier, which already computes them in coupled mode) is identical.

**F16. Quantized-backbone training hiddens** (R3 bonus, partially derived): FastMTP/RH train against the serving stack's hiddens (bf16 vLLM). Ours is the IQ3_S-quantized target as served by strata — training against ITS hidden states is distribution-matched by construction (bounded by the quant deltas, not above them). Nobody publishes this comparison; the principle "train against the serving stack" IS published (RH's online vLLM extraction).

### UNKNOWN (searched, not found)

- FastMTP OpenReview reviews/decision: openreview.net is behind a Cloudflare interstitial; review scrutiny unavailable. (Forum id J7xDwZSyI4 — retry from a different network if reviews matter.)
- DeepSeek-V3's own 85-90% acceptance figure: not located in the report full text I could fetch (truncated at that section); widely repeated in secondary analyses (alphaxiv summary, tekblueprint). Treat as plausible-but-secondary.
- GLM-4.5-Air per-depth acceptance 0.92/0.68/0.38 (cited in our T7.B.1): NOT in the GLM-4.5 report (verified — Section 2.4 gives only the lambda schedule) and NOT in the LMSYS blog (verified). Origin in our repo's T7 dossier untraced; keep as internal-only supporting color, cite FastMTP's 70/11/2 instead.
- Any published training run of a drafter against a QUANTIZED (sub-4-bit-MoE) frozen target: not found. Our F16 is the first plan I can find.
- Kimi K2 MTP: ships ZERO MTP layers per GLM-4.5's Table 1; no K2 drafter-training recipe located (low priority; not needed).

---

## 3. (B) Mapping the recipe onto OUR single-layer same-family drafter

**No architecture change is required.** FastMTP's entire method is training-side for exactly the block we already serve:
- Single shared-weight head applied recursively = our drafter chained to depth 6 (`--spec 6`; draft j feeds back its own sampled token's embedding via the target's embed table + its own output R). GLM-5's multi-layer design is the same family with N physical layers; the SHARED-parameters result (F9) says the recursive single layer is the memory-flat version of it — we keep ours.
- Frozen embed + frozen bound head = exactly RH speculators' setup ("embed_tokens and lm_head frozen and shared"). Our head is even closer to serving reality: the 40,525-row sub-vocab is what SERVES, so training q over the same sub-vocab has zero head-distribution shift. FastMTP's FR-Spec finding (16-32K vocab, drafting-only compression, lossless) retroactively validates our sub-vocab head rather than threatening it; run the existing `STRATA_MTP_FULL_HEAD=1` diagnostic once to bound what the subset costs today, before training, so post-retrain deltas are attributable.
- The teacher-forced recursion (F1) trains our served chain's exact failure point: depth k's input hidden is the head's own depth-(k-1) output — today the block has never been optimized for that path (it only ever saw depth-1 CE in pretraining).
- Engine changes needed: NONE for the method itself. Two small, gated, optional additions: (1) a prefill hidden-state dump lane (extend the existing `STRATA_MTP_TAP` pattern — today it dumps ONE window's R/sample/logits — to per-position R + token ids over a corpus replay; ~20 KB/token bf16; 8-16M tokens = 160-320 GB, fits the 735 GB free on /home; NOT /mnt/DATA); (2) optional verifier p-list logging (the coupled verifier already materializes the target's post-chain top-64 list per row — dump it to enable the F7/F15 overlap-loss arms). Stitching retrained tensors back is manifest-driven (`mtp-manifest.json`).

What the literature implies that we should NOT do:
- EAGLE-3 multi-layer feature fusion (low/mid/high): needs intermediate target-layer taps at verify time — real engine surgery; EAGLE-3's own gains come mostly from training-time test + token-level loss (F5), which we can take without it. Revisit only if depth-3+ conditional acceptance stays flat after arms 1-3.
- Medusa heads / extra heads: dominated, and would need new engine tree machinery (T7 verdict, unchanged).
- Trunk-side MTP loss (DeepSeek/Gloeckle form): impossible — target frozen; also explicitly the variant that fails on frozen trunks (F4's Llama-2 finetune negative).
- FR-Spec vocab re-derivation: orthogonal; ours ships and FastMTP says compression is fine if frequency-chosen.

---

## 4. (C) THE RETRAIN RECIPE, ranked by (expected acceptance gain)/(training cost)

Standing config this recipe assumes: `--spec 6`, coupled sampling, `--spec-min-p 0.5`, `STRATA_SPEC_DRAFT_TEMP 0.85`, `--mtp-q4`. Everything below trains ONLY the 31 mtp.* tensors; target, embed table, and the bound head stay frozen. Gates (from R3, kept): dry-run on a small sibling first; v1 gate = pos-2/3 acceptance +>=8pp each AND pos-0 regression <=1pp AND e2e >=+8%; kill-rule = extend served depth only while marginal acceptance > 6-9% breakeven; never boot-adjacent to the live server.

**ARM 0 — measurement prerequisites (hours, free of training risk).**
1. Replay ~2-5M tokens of the standing prompt corpus with `STRATA_MTP_FULL_HEAD=1` vs sub-vocab head; record per-depth offered/acceptance. Attributes the sub-vocab head's share of the deficit.
2. Check whether the mimic corpus is already target-generated text (if the corpus was produced by this model, it IS self-distilled by construction and Arm 2's data cost collapses to the dump pass). If not, generate 5-20M tokens of target output at temp ~0.6/top-p 0.95 (FastMTP's sampler) in the idle lane at ~200 t/s = 7-28 h, or accept corpus text and forfeit the +8%-relative self-distill delta (F1 ablation).
3. Build the prefill dump lane (extend `STRATA_MTP_TAP`): R (10240 bf16) + token id per position, 8-16M tokens -> 160-320 GB on /home (735 GB free). ~1-2 days eng. This dump is quantized-backbone-faithful (F16) — the single biggest distributional advantage over the published recipes.

**ARM 1 — FastMTP/RH-core retrain (THE bet; cost S; expected: offered chain 2.2 -> ~3-3.5, +12-25 t/s, RH/FastMTP-scale up to ~237).**
- Objective: recursive teacher-forced CE at all positions, depths k=1..K, weights alpha_k = beta^(k-1)/sum, K=8 (train deeper than serve=6; FastMTP K=3 was serve-matched, ours must cover 6), beta: arm 0.6 vs 0.8 (F14; 0.8 keeps ~9% of loss on depth 6 vs 3.2%).
- Trainable set, two stages: (a) dense-only ~98M params (fc_embedding/fc_hidden, pre_fc_norms, 3 hc mixers, self-attn + indexer, router, shared expert) — ~196 MB bf16 weights; AdamW fp32 states ~0.8 GB; trivially fits, ~1-2 h on the 5090; (b) all 31 tensors (2.55B routed-expert params added): bf16 weights 5.2 GB + bf16 grads 5.2 GB + 8-bit-AdamW ~5.3 GB (fp32 states 21 GB does NOT fit with activations) + activations with grad checkpointing (seq 8192, micro-batch 1-2, grad-accum to global 64) ~4-8 GB -> ~16-24 GB total, fits 32 GB; ~3-6 h.
- Frozen experts at train time: load the routed experts at their SERVE quant (Q2_0-equivalent) in stage (b) if VRAM-bound, else bf16; if bf16, re-quantize after and re-measure (Risk R2).
- Optimizer: AdamW(0.9, 0.95), cosine, peak LR 5e-5 (FastMTP) with the RH example's 1e-4 as the alternate arm; warmup 3-5%; 3 epochs; seq 8192; global batch 64.
- Data: the Arm-0 dump (8-16M tokens minimum; 30-60M if generation was cheap) — self-distilled, distribution-matched, quantized-backbone hiddens. Per-position pairs (hidden at t, tokens t+1..t+K+1).
- Expected (anchored): RH pos-deltas (pos2 +0.14) and FastMTP depth-2/3 recovery (11->56, 2->36) imply our offered chain re-lengthens because the min-p gate sees a sharper, recursion-trained q; R3's calibrated 227-237 t/s band is consistent with T13A window arithmetic (each re-offered accepted row: +1 token per +0.55 ms).

**ARM 2 — direct acceptance-overlap loss on logged verifier p-lists (cost S+; expected +3.5-8% tau, biggest at depth 4-6 and under coupled sampling; LK's DeepSeek-MTP arm measured +5.6%).**
- Add F7's hybrid `lambda*KL + (1-lambda)*TV` (adaptive lambda=exp(-3*sg[alpha])) or `-log alpha` as a SECOND loss term over the union of the target's logged top-64 p-list support and q's support; for coupled mode also arm the collision form sum p*q (F15, derived). Warm-start from Arm 1; 1-2 epochs, LR 2e-5.
- Requires the optional verifier p-list dump (Arm 0 extension, ~half a day). LK's theory says this is where capacity-limited drafters have the most headroom left after CE, and their truncated-vocab result says the sub-vocab head is safe under it (off-vocab mass = 0 in alpha, no KL blowup).

**ARM 3 — AdaSPEC-style selective filtering + KD-temperature match (cost S; expected up to +10-15% acceptance relative, untested on MTP-class heads).**
- The Arm-1 checkpoint IS the reference model AdaSPEC needs: compute per-token loss gaps on held-out dump, keep top-40% of tokens by gap, retrain from Arm-1 init on the filtered set (their ablation: training on the BOTTOM 40% actively hurts). Distillation temperature ~0.7 for the soft term (F-source: Temperature-Centric Investigation, arXiv 2410.10141 — KD temp 0.7 beat 0.5), and evaluate under the SERVE chain (0.85 draft-temp scale + min-p 0.5) so the objective is measured where the gate acts.

**ARM 4 — EAGLE-3 training-time test (cost M; expected: flattens the depth-3+ conditional curve; only if Arm 1-3 leave depth>=4 marginal acceptance < breakeven).**
- During recursion, at depths >=2 replace the ground-truth token embedding with the drafter's OWN sample (drawn from q at the serve temp scale) with probability rising over training (curriculum); attention masks unchanged (single layer, no cross-step attention to fix). This directly trains the served failure mode (own-token input) that teacher-forced recursion still never sees. No published MTP-head result; EAGLE-3's drafter-class evidence (F5 Fig. 7) is the basis. Data/objective otherwise identical to Arm 1.

**Ranked summary (gain/cost):** Arm 1 (dense-first stage (a), then (b)) >> Arm 0 prerequisites (they gate attribution, near-zero cost) > Arm 2 > Arm 3 > Arm 4. Do Arm 0 + Arm 1 stage (a) in one day; stage (b) the same weekend; Arms 2-4 only against the measured residual deficit.

**RISKS.**
- R1 — Coupled-sampling compatibility: LOW correctness risk, HIGH measurement discipline. In coupled exact-match mode the OUTPUT is a function of the seed alone for ANY q (the drafts only decide speed, `coupled_draft.hpp` header contract), so a retrained drafter cannot corrupt the output distribution — the loss mode is acceptance regression, not quality loss. The real coupling risk: the min-p 0.5 gate and 0.85 temp scale act on q at SERVE; a drafter trained only at T=1 CE can still be gate-trimmed. Mitigation: after each arm, re-sweep STRATA_SPEC_DRAFT_TEMP {0.7, 0.85, 1.0} x min-p {0.5, 0.35} on the SAME battery (cheap; T7.B.5 says this ground is unexplored) and gate on the serve-chain configuration, not the training configuration.
- R2 — Quantization sensitivity at Q4/Q2: the retrained bf16 tensors get DIRECT-CAST (projections+head rows Q4_0 under --mtp-q4; routed experts Q2_0) — never GPTQ (F-201 law). An LR-5e-5 finetune can grow per-tensor dynamic range and cost post-cast acceptance. Mitigation: per-tensor AMAX re-calibration before cast; ALWAYS measure the battery twice (bf16-serve vs q4-serve of the same checkpoint); if the cast costs >2pp acceptance, add a weight-clip (per-tensor scale freeze) or 1-epoch quantization-aware finetune at LR 5e-6.
- R3 — Domain overfit / "forgetting": the drafter cannot damage output quality (lossless by verification), so "catastrophic forgetting" manifests only as acceptance loss OFF the training distribution. RH's own warning: their GSM8K-narrow example may underperform a mixed diet. Mitigation: train on our standing prompt-replay mix (the corpus we actually serve), gate pos-0 regression <=1pp on the held-out battery, keep the pre-retrain checkpoint as a 2-minute rollback (28 shards, manifest-driven).
- R4 — Offered-chain gate may re-cap the gain: if retrained q at depths 4-6 still fails the min-p 0.5 gate, acceptance gains bank only at depths 1-3 (still the +12-22 t/s RH-scale band; the 250-t/s depth bonus needs the gate). This is the kill-rule domain: extend served depth only while marginal acceptance > 6-9% breakeven.
- R5 — First training run engineering: hidden-state dump volume, sequence packing, and the stitcher are all new code. De-risk on a small sibling model first (R3's 15-30-min dry-run plan), keep every stage env-gated and reversible per house rules.

---

## 5. Contradictions and their resolution

1. **AdaSPEC: forward-KL best (reverse-KL/TVD ablate worse) vs LK: TV- hybrid beats KL.** Resolution: different regimes — AdaSPEC's drafts are tiny independent models (31M-500M) trained from scratch on full vocab; LK's setting includes large truncated-vocab drafter heads finetuned from strong inits, and LK itself shows pure TV fails from scratch while the ADAPTIVE KL->TV schedule wins; AdaSPEC never tests an adaptive schedule. Both agree on the deeper point: plain KL is a sub-optimal proxy at capacity limits. Our plan: CE/KL first (Arm 1), adaptive-overlap second (Arm 2) — the published order of maturity.
2. **"MTP needs joint pretraining" (DeepSeek/Gloeckle lambda schedules) vs "post-hoc head-only retraining suffices" (FastMTP/RH).** Resolution: not in conflict — the joint-loss results are about BACKBONE quality (Table-4 eval gains, dropped at inference), while FastMTP/RH retrain only the DRAFTER against a frozen trunk and measure acceptance, not benchmarks. Gloeckle's negative (MTP loss on a finetuned Llama-2 trunk doesn't stick) concerns trunk loss changes, which no one proposes here. For our purpose the FastMTP/RH evidence is the applicable one, and it is direct.
3. **FastMTP K=3 optimal vs our K=6-8 plan.** Resolution: FastMTP's K=3 was measured with beta=0.6 on a 7B dense model whose speed peak fell at K=3 — tau was STILL RISING at K=3 (their own figure), and their serve chain was depth-3. We serve depth 6 and our deficit is depths 2-6, so we train K>=serve depth with a flatter beta. No contradiction, but F14 stays an armed extrapolation.
4. **DeepSeek-V3 "85-90% acceptance"**: could not be verified in the primary text I fetched; kept out of all arithmetic. Our expectations rest on FastMTP's and RH's measured per-depth tables instead.
5. **EAGLE-3 "68B tokens" folklore**: the paper states conversation counts (ShareGPT ~68K + UltraChat ~464K entries, ~8x EAGLE), not 68B tokens. Corrected here so the data-scale arm is not mispriced.

---

## 6. Open questions

- Does the mimic corpus consist of this model's own generations? (Determines Arm 2 data cost; one grep of the corpus pipeline answers it.)
- Is the prefill dump lane's R state (HC=10240) sufficient, or does training also want the pre-mixer `mixed_`/`sample_` taps the existing TAP already dumps? (The drafter's served input IS R via the final mixer; training on R matches serving — but a side-arm on `sample_` would let us ablate input representations cheaply.)
- What does the full-vocab head diagnostic actually show on our mix (Arm 0 item 1)? Decides whether a future engine change (dynamic head width) is ever worth it — expected NO.
- Coupled collision-probability loss (F15): does optimizing sum p*q instead of sum min(p,q) measurably differ under exact-match coupled verify? No published answer; Arm 2 arms both.
- Post-retrain, does the STRATA_SPEC_PROB_DT sweep optimum move off 0.85? (Almost certainly yes if q sharpens; the sweep is nearly free.)

---

## 7. Implications

- The 250 t/s single-stream goal has exactly one priced lever in this dossier: the retrain. Arithmetic (T13A) and the literature (F1/F2) agree that a recursion-trained drafter re-lengthens the offered chain, and the window economics convert offered depth to tokens at a fixed 0.55 ms/row cost. Engine micro-optimizations alone touch 250 only in good boots (T13A); the retrain clears it structurally.
- The retrain is ALSO the prerequisite that makes T13 queue items 3 (draft un-gating) and the depth-lane items pay: an un-gated chain only helps if deep drafts are confident.
- No engine surgery is required for the method; the two optional dumps (hidden states, verifier p-lists) are small, env-gated, and enable Arms 2-4 later. The retrain ships as a data artifact (28 shard replacements + manifest), i.e., fully reversible, boot-isolated from the live server.
- Quality exposure is structurally zero (verification is exact/probabilistically exact for any q); ALL risk concentrates in acceptance regression and quantization after-cast — both cheaply measurable, both gated.

---

## 8. Source list

Local / code (read this session):
1. `/home/svg/LLM/Custom-data/mtp/mtp-inventory.md` — 31 tensors, 5.214 GB BF16, shapes.
2. `/home/svg/LLM/strata-custom-2/src/core/mtp.cpp` — load/bind, sub-vocab head, coupled draft sampling, graph capture, TAP.
3. `/home/svg/LLM/strata-custom-2/include/strata/core/spec_prob.hpp` — q-chain, min(1,p/q), min-p gate, draft temp scale, counter spaces.
4. `/home/svg/LLM/strata-custom-2/include/strata/core/coupled_draft.hpp` — exact-match coupled contract, output-is-function-of-seed guarantee, counter arithmetic.
5. `/home/svg/LLM/strata-custom-2/SVG-CUSTOM-TODO.md` — T7.B.1, T11.R3 GO dossier + recipe sketch, T13 master queue, F-228/T13A measurements.
6. `/home/svg/LLM/strata-custom-2/campaign/research/T13A-decode-critical-path.md` — window economics (11.5 ms / 2.47 tokens / 0.55 ms per row; min_p stair).

Papers (full text unless noted):
7. FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction — arXiv 2509.18362v1 (abs + full HTML), Cai et al., 2025-09-16.
8. Better & Faster Large Language Models via Multi-token Prediction — arXiv 2404.19737 (ar5iv full text), Gloeckle et al., 2024.
9. DeepSeek-V3 Technical Report — arXiv 2412.19437v2 (full HTML), 2024-12/2025-02.
10. EAGLE-3: Scaling Inference Acceleration via Training-Time Test and Multi-Level Features — arXiv 2503.01840v2 (full HTML), 2025-03.
11. Medusa: Multiple LLM Heads for Faster Parallel Decoding — arXiv 2401.10774v3 (full HTML), 2024.
12. GLM-5 — arXiv 2602.15763v1 (full HTML), Feb 2026 (shared-parameter multi-layer MTP, accept length 2.76 vs 2.55).
13. GLM-4.5: Agentic, Reasoning, and Coding (ARC) — arXiv 2508.06471v1 (HTML; MTP section), Aug 2025.
14. LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding — arXiv 2602.23881v1 (full HTML), Feb 2026.
15. AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders — arXiv 2510.19779v1 (full HTML), NeurIPS 2025.
16. Online Speculative Decoding — arXiv 2310.07177 (ICML 2024; search-corroborated, abstract-level).
17. DistillSpec — arXiv 2310.08461 (ICLR 2024; search-corroborated).
18. Temperature-Centric Investigation of Speculative Decoding — arXiv 2410.10141 (search result: KD temp 0.7 > 0.5; not full-read).
19. Qwen3-Coder-Next Technical Report — arXiv 2603.00729 (HTML; negative result — no MTP internals).

Docs / practitioner:
20. Red Hat Developers: "Optimize vLLM speculative decoding with FastMTP heads" — developers.redhat.com, 2026-09-08 (recipe + acceptance table).
21. vLLM Speculators docs, MTP algorithm page — docs.vllm.ai/projects/speculators (method, beta=0.6 K=3, convert/train/stitch).
22. vllm-project/speculators GitHub README (offline hidden-state data generation, ~100-400M head params, train without loading verifier).
23. vLLM Qwen3-Next recipe — docs.vllm.ai recipes (num_speculative_tokens: 2, qwen3_next_mtp, --no-enable-chunked-prefill).
24. vLLM blog "vLLM Now Supports Qwen3-Next" — vllm.ai, 2025-09-11; Qwen3-Next blog (via search; JS-rendered, thin) — the ~2-3x MTP decode claim is blog-level only.

Counts: code/local 6, literature 13, practitioner/docs 5 (some overlap) — 24 sources, well past the 12/4 bar.

## 9. Methods note (including unproductive searches)

Productive: full-text fetches of FastMTP, Gloeckle (ar5iv — arxiv.org/html/2404.19737v2 is 404; v1 path untested, use ar5iv), DeepSeek-V3 v2, EAGLE-3 v2, Medusa v3, GLM-5, GLM-4.5, LK Losses, AdaSPEC; the Red Hat article, speculators docs+GitHub, vLLM recipe/blog.
Unproductive / blocked: arxiv.org/abs pages give abstracts only (two fetches wasted — go straight to /html or ar5iv); openreview.net FastMTP forum blocked by Cloudflare interstitial (reviews unavailable); qwen.ai blog is JS-rendered (content = "Qwen"); alphaxiv landing pages lack full text; Qwen3-Coder-Next report contains NO MTP content (confirmed); no standalone Qwen3-Next technical report exists (searched twice); "Decoding Speculative Decoding" as remembered does not exist under that name — the usable paper is arXiv 2410.10141; Kimi K2 drafter recipe not pursued (0 MTP layers per GLM-4.5 Table 1). Application/pdf content type is unsupported by the fetch tool — use /html renditions.
Self-corrections logged: (a) nearly cited "68B tokens" for EAGLE-3 from memory — the paper says conversation counts; (b) nearly accepted the GLM-4.5-Air 0.92/0.68/0.38 figures from our own T7 dossier as primary — they are not traceable to the report or LMSYS blog; demoted; (c) FastMTP's "self-distillation" is document-level (target-generated responses) while the recursion stays teacher-forced at the token level — the two are easy to conflate; EAGLE-3's training-time test is the variant that feeds back own TOKENS, and it is a separate, later arm here.
