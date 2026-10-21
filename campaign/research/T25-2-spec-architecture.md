# T25-2 — Speculative-decoding architecture study: every drafting/verification mechanism in the reference library, priced for our coupled exact-match chain at spec 6

Date: 2026-10-20/21. Lane: M1 single-stream primary (conc riders noted). Scope: read-only study of the five
reference clones (`/home/svg/LLM/data/references/{basalt, Xolotl, project-maya, DeepInfer, DFlash2}`, pins per
MANIFEST.md) against our contract — chain-only, position-keyed, one-pick-per-row coupled exact-match verify,
`--spec 6`, flat `spec_min_p 0.5`, typical-rescue in the batch lane. Nothing built, booted, or modified; only
this file written. Contract authority: T17C (`campaign/research/T17C-spec-algo-leaps.md` §2); closures respected
(tree verify CLOSED by arithmetic + T17C §5.5; STRATA_SPEC_PROB CLOSED WITH DATA, F-235: M1 −4.2 % / pair −5.1 % /
conc −7.0 %; draft-verify overlap demoted; Controller estimator measured 0–3 % ceiling).

---

## Summary

All five references implement the same three doors T17C mapped — **drafter (training/input side), offer shaping
(how many rows), acceptance rule** — and none of them breaks the contract. The family's live mechanisms are:
basalt's **cumulative keep rule** (`draft_cum.hpp`, host+device-exact, placement-scaled threshold) and its
**device-conditional draft graph** (CUDA conditional nodes end the draft chain on-device); Maya's **running DSpark
implementation** (confidence head + per-draft floor + cumulative product cut + drafter cool-down, measured +6.5 %
decode token-identical on an APU) and its measured **SPEC_SAMPLE / SPEC_CANCEL** pair; DeepInfer's **DSpark
confidence-scheduled varlen verification** (production: −30 % verified tokens at ~equal accepted length, +4–7 %
at their operating point, rising to +11 % at 96 seq/rank but **−1.5 % at 16**); Xolotl's **confidence-gated verify**
on a tiered-expert engine (huge there because a rejected row costs ~25 ms of NVMe/PCIe — our marginal row is
2.47 ms, so the same rule's headroom here is ~10× smaller), its **point-mass exact sampler**, and a
**split-MTP parity harness** pattern; DFlash2's **five-deep-tap + hc-mean drafter input** (capture patch proves the
SGLang-supported convention for our backbone) — the T14C+ multi-tap lever, with DSpark's 3-tap evidence
(77.1 % first-draft accuracy) as the existence proof.

**The one decision this study adds:** the offer-side door on our engine should converge on basalt's
joint-probability keep rule (recalibrated on OUR acceptance logs) as the principled form of the post-retrain
re-stair — not a per-depth min-p schedule — and its threshold should key on OUR c_row (or expert bytes), never a
flat constant. Everything else is either already ours (DraftPolicy chain arbitration — we are the donor), closed
with data (rejection sampling, trees), a T14B/T14C training-side rider (multi-tap input, confidence/VAT head), a
conc-lane rider priced ~0–2 % at our 6-slot scale (varlen budget), or tooling (parity harness, routing-trace
instrument). Standing calibration for every price below: acceptance 74.9 %, accepted ≈ 2.47 tokens/window at
offered 3.1–3.2 drafts/window, window V(T) = 9.7 + 2.47·(T−1) ms (so 2.47 is BOTH our accepted tokens/window and
our ms/marginal-row — same measured number, two meanings), W(1) = 8–9 ms, c_row 2.2–2.6 ms post-POOL_DIET
(3.56 s35 / 4.62 conc calibrations), depths 4–6 starve at the flat 0.5 floor, T14C elasticity +0.80 %/pp M1 per
per-proposal acceptance point, marginal-row break-even ≈ 0.72 expected tokens (→ ~0.56 at c_row 2.8).

---

## Ranked table

Magnitude = M1 lane unless stated; effort S ≤ 1 day, M multi-day, L project. "Class" = port shape
(idea / algorithm / code). Falsifier = cheapest first experiment, no build unless said.

| # | Mechanism (source) | Their evidence | Expected on OUR contract | Effort | Class | Conflict surface | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | **Cumulative keep rule + placement-scaled threshold** (basalt `draft_cum.hpp`; the same joint form in Maya DSpark `cum_min` and DeepInfer's cumprod) | Default-on in every basalt number since 5ec431db; the 324.6 tok/s / spec-7 config runs it; Maya DSpark cum rule measured **+6.5 % decode, token-identical** (12.04→12.82 tok/s, Radeon 8065S, greedy) where it replaced a FIXED window | Replaces the flat 0.5 gate as the W2.5 re-stair's principled form: **+0–1 % standalone** (the gate is already near the SpecDec++ 0.42 threshold), **+1–3 % post-retrain** when deep marginals rise and the joint rule re-lengthens offers 2.2→3.5 without re-opening flat-gate negatives (F-228) | S (rule ≈ 50 lines; bins recalibrated offline first) | algorithm (+idea: placement scaling) | Offer-side only ⇒ outputs p-exact by the coupled contract (`coupled_draft.hpp:9-12`); width mix changes ⇒ coupled-acceptance battery per F-230; bins encode THEIR drafter's calibration (argmax drafts) — recalibrate on ours | **Offline, no boot:** replay `draft_keep` on our logged (dprob, accepted) pairs; read offered-width mix and E[tokens]/E[rows] at th ∈ {0.30, 0.35, 0.40} vs flat 0.5, price at c_row 2.2–2.6 |
| 2 | **Multi-tap drafter input** (DFlash2: 5 completed-layer hc-means; DSpark/Xolotl: 3 entering-layer hc-means, 77.1 % first-draft) | DFlash2 card: AL +0.18..+0.42 over stock MTP; Xolotl fixed harness: DSpark 3-tap mean accepted 2.583/step, 3.58 tok/pass, first-draft==truth 77.1 %, conf Q1..Q4 12/40/78/99 (separable!) | The named T14C+ lever if T14B stalls (T23-dflash2 ACTION 2). Rider on T14B infra: harvest captures k deep hc-means (prefill-only, free); trainer input 2×2560 → (2+k)×2560; serve needs verify-side tap capture + pack contract v2. **+2–4 % M1 class** if first/deep marginals move like theirs, ON TOP of T14B | M on T14B infra; serve capture M/L | algorithm + code | Breaks the 31-tensor mtp/rt SHA contract (F-51/52/87/100) — pack version bump; T14B gates G0–G5 unchanged; do NOT start before T14B Arm-1 verdict | In the T14A dump extension: log per-layer hc-mean R at candidate taps (5/15/25/35/45 and 37/38/39-analogues), train a probe head offline, read Δ acceptance on the G0.2 stair before any engine surgery |
| 3 | **Device-conditional draft graph** (basalt `cudaGraphSetConditional` chain) | Default-on sole path (c400b2fd) of their 6.675 ms window; issue #16 shows the failure class (begun-word poll) | Deletes our per-step host min-p spin (~50–100 µs/step × up to 6 steps, `mtp.cpp:1697-1715`) and per-step graph launches: **+0.5–1.5 % M1** (host-tail class, T17A R2 band), zero acceptance change (same graphs, same math, device-side) | M/L (CUDA 12.4+ conditional nodes inside our whole-window capture) | code | Draft-loop restructure (our un-gated twin `mtp.cpp:1631-1665` is the natural host); graph-capture interplay; F-229's ungated twin measured −2.73 % = launch-shape-only, this is the device-side completion | **STRATA_DRAFT_TIMING g_dt.gap_ms capture boot** — instrument built, never captured (F-235 inconclusive); prices the per-step gap before any port |
| 4 | **DSpark confidence head** (DeepInfer `ConfidenceHead`; Maya running impl; = T17C's VAT head slot) | DeepInfer: ROC-AUC 0.81–0.90, ECE 3–8 % → ~1 % after STS; production −30 % verified at equal AL; Maya: the +6.5 % cum-rule result rides it; Xolotl: conf quartiles 12/40/78/99 | Same architectural slot as VAT's survival head (arXiv 2608.30135) — one T14B objective rider, two training signals: BCE against c* = 1 − ½‖p−q‖₁ (DSpark) vs simulated accept/reject (VAT). Value to us EXISTS only if the draft's own dprob is a loose acceptance predictor — else candidate #1's bin table suffices at zero cost | S (rider on T14B) | algorithm | T14B trainer scope; a serve-side head adds a tensor to the pack (contract v2, shared with #2) | **Offline:** from our logs, fit basalt-style acc|bin table on OUR dprob; if within-bin spread is tight (calibration table self-consistent, like basalt's), the head buys nothing — record and skip |
| 5 | **DSpark varlen verify budget, batch form** (DeepInfer `schedule_mtp_varlen` + `profile_decode_perf`) | Production table: **−1.5 % @16 seq/rank → +11.0 % @96**; verified 5.00→3.24–4.44, accepted 3.3→3.12–3.26 | Conc rider only. Our scale (6 slots, COHORTS=2, ~6-row windows) sits LEFT of their crossover: prior **0 to +2 % conc**, 0 M1 (degenerates to the offer gate single-stream). Correctly shaped for us as: per-request T in the cohort windows + a budget over rows, keyed to a measured c_row(T) curve | M | algorithm (idea for the shape, code is vLLM-shaped) | Batch finish/keep[] path (`generate.cpp:10310-10338`); F-231 stale-slot law; COHORTS untouched structurally | **Routing-trace instrument boot:** measure c_row as a function of window width T (mayà's U(n)/U(1) = {1.0, 1.70, 2.31, 2.88, …} suggests width-dependent expert reads). Flat c_row(T) ⇒ varlen has nothing to arbitrate — record and close |
| 6 | **Point-mass exact sampler + early draft cancel** (Xolotl `spec_accept`; Maya SPEC_SAMPLE/SPEC_CANCEL) | Xolotl: unit-tested 400k draws/case; Maya: SPEC_SAMPLE drafts-kept 68.6→74.7 %, decode +1.6–2.7 % (+5.3 % with cancel); cancel saves 5.4–6.5 ms/rejected draft on a split | **Sampling: CLOSED for us** (F-235 SPEC_PROB negative — their positives are on argmax baselines, ours is coupled at DT 0.85; see Contradictions). Cancel: our verify is ONE batched forward — no mid-forward cancel site exists; the analog is offer-side truncation = candidate #1. Conc cool-down (skip a slot's drafter after an all-below-floor round, Maya `cool`): **+0–1 % conc**, S | S (cool-down) | idea | F-235 closure respected; batch drafter split (F-206 draft_launch/finish) is where a cool-down lives | Batch-lane telemetry (per-slot accepted-rate EMA already implicit in keep[]); one COHORTS boot with a per-slot cool flag |
| 7 | **Split-MTP parity harness + acceptance-band gate** (Xolotl `split_mtp_batch_parity.py`) | Their harness: 5 fixed greedy prompts + 25k-token prompt, bit-exact text compare, draft-KV deliberately NOT bit-identical (MMQ vs mmvq), target tokens must match, "drafts accepted X of Y" within ±3 pts | Adopt as the G1/G2 gate pattern for any future draft-path precision change (T14B rollout lanes, T19 mmvq_il draft KV, STRATA_SPLIT_MTP_BATCH question). Zero M1 — pure tooling | S | code (HTTP-driven, engine-agnostic) | None (test-only) | n/a — adopt into T14B rollout spec (P5) |
| 8 | **Measured-cost arbitration** (Maya/basalt `DraftPolicy` cost EMA + shape prior; dormant `Controller` + `CostModel` priors) | Maya header: defaults from 23-Sep measurements (mmvq-multi, spec-economics routing trace, cpu-expert-multi); their `kShape` = RTX 5070 window-cost curve; U(n)/U(1) distinct-expert curve is real measured data | We are the DONOR on this line (our `DraftPolicy::chain()/chain_rate()/stale()` and `propose_after` exist only on our side). The one adoptable artifact is the **U(n) distinct-expert curve** as a measured input to candidate #5's budget rule | S (instrument only) | idea | None | Same routing-trace boot as #5 (shared falsifier) |

Not re-opened, with reasons: **tree verify** (T17C §5.5 arithmetic; basalt independently converged on flat windows —
their only "tree" is a retired diagnostic hook; the family's investment went to exactness proofs + keep rules);
**rejection sampling / STRATA_SPEC_PROB** (F-235 closed WITH DATA; Maya/Xolotl positives are argmax-baseline
results, not evidence against our closure); **draft-verify overlap** (draft = 6–9 % of our window; demoted in
T17C); **the Controller** (dormant in BOTH sibling trees — only `controller_test.cpp` references it; the framing
correction is recorded below).

---

## Per-mechanism sections

### M1 — basalt: drafter arbitration + the cumulative keep rule + device-conditional drafting

**Arbitration shape (what actually runs).** Three sources, two live. (1) The MTP chain with the **keep rule**
replacing our flat min-p gate; (2) a **suffix-lookup drafter** (trigram open-addressing table, WAYS=4 end positions,
min_match 3, backward extension to 32 — `include/basalt/spec/suffix_drafter.hpp:8-24`) consulted only when its
first token AGREES with the MTP's first draft (`generate.cpp:8408-8417`: `if (k > 0 && sbuf[0] == drafts[0])`);
(3) **DraftPolicy** choosing MTP-vs-lookup per round by expected tokens/ms with measured per-window-size round
costs (EMA α 0.1 over round ms, α 0.05 over MTP tokens, decayed lookup counts, β-priors {0.75, 0.88, 0.93, 0.96}
worth 4 observations, 3-probe escape so a guessed cost can't permanently veto measuring a size —
`src/spec/draft_policy.cpp:12-21,36-48,55-82`). The window is then trimmed by the keep rule:
`while (T < S_mtp && draft_keep(dprob, T-1, min_p, cum_adapt.th())) ++T` (`generate.cpp:9291`, siblings :6717, :8402).
A fourth file — `src/spec/controller.cpp`, the none/lookup/MTP arbitrator maximizing E[tokens]/T(step) with
K_MAX = 8 — is **dormant in basalt AND maya** (grep: referenced only by `controller_test.cpp` in both trees; all
published basalt behavior is flat-window + DraftPolicy + keep rule). See Contradiction 6.

**The keep rule, exactly** (`include/basalt/core/draft_cum.hpp`, whole file, ~78 lines):
- Binned acceptance model: `draft_cum_acc(p)` maps the draft's own probability to an observed-acceptance rate via
  8 bins: p<0.30→0.12, <0.50→0.26, <0.70→0.43, <0.80→0.51, <0.90→0.59, <0.95→0.66, <0.99→0.79, else 0.96 (:25-28).
- Keep test: draft j stays while `∏_{i≤j} acc(p_i) ≥ th`; with th = 0 the rule degenerates to our flat
  `p_j ≥ min_p` (:31-36). Base `kDraftCumBase0 = 0.35` (:39).
- Placement scaling: threshold ×{1.0→1.5} as mean CPU-served expert bytes/slot goes 1.1→3.5 MB
  (`draft_cum_scale()`, :44-50), set once at tier finalization (`generate.cpp:4779`:
  `(b_ram+b_top)/n_layers/n_expert/1e6`). "Fixed by the pack and its placement, never by timing" — a c_row proxy
  without a clock read.
- Per-request adaptation (`DraftCumAdapt`, :56-76): per window with n offers / a accepts, expected tokens
  e = Σ_j ∏_{i≤j} acc(p_i); EMAs obs ← 0.97·obs + a, exp ← 0.97·exp + e; threshold ramps linearly from base
  toward 0.25·scale as obs/exp crosses [0.9, 1.1]. Window threshold fixed before drafts, shared host/device.
- Device-conditional drafting: per-step draft graphs captured with CUDA conditional nodes (CUDA 12.4+). A 1-thread
  kernel at step end recomputes the identical product "the same products, in the same order, as draft_keep on the
  host" and sets/clears the handle (`verify_kernels.cu:816-847` `mtp_draft_cond`, :888-929 `mtp_step_end`);
  the host loop launches ONE conditional graph and polls a −1 sentinel (`mtp.cpp:1536-1588`, gate at :1563) —
  no per-step host decision, no host spin. History probs reach the device through a `hist` buffer.

**Their evidence.** No isolated A/B anywhere (deliberately folded, same as their PDL). Provenance by git:
`241b5b4c` 2026-10-05 (rule, host+device), `be7147f3` (device rule reads earlier steps' probs), `32e5522f`
(--exp spk_th 0.35 default, 0 = min-p rule), `47f6192f` (per-request adaptive threshold, then `5ec431db`
default-on), `7af4a2c6` 2026-10-18 (CPU-bytes scaling), `1ec9b8b5` (batch windows draft per slot with "solo's
draft-length rule"). Every published number (6.675 ms / 324.6 tok/s, spec 7, `docs/unified-kernel.md:32`) runs it;
their accepted-drafts 70.1 % at spec 7 vs our 74.9 % at spec 6 — the rule shapes offers, it does not raise
acceptance. `docs/precision/index.md:56` carries the disclosure that window width differs by pack and that verify
kernels are not batch-invariant across widths (near-tie flips possible) — our M3b covers the same class.

**Applicability / price / port / conflicts / falsifier.** As ranked (#1 + #3): this is the offer-side door's
principled form. It is exact under our contract for the same reason basalt's header states — "Only what is
proposed changes: acceptance and outputs stay exact" (and in OUR coupled form the output is a function of the
seed alone whatever the drafts are, `coupled_draft.hpp:9-12` both trees). The bins, however, encode THEIR drafter
(argmax drafts, spec 7) — ours sample at DT 0.85 coupled, so the acc|p table must be refit from our logs before
the threshold means anything (that refit IS the offline falsifier, and doubles as candidate #4's go/no-go).
Expected +0–1 % standalone (flat 0.5 is already within 0.08 of the SpecDec++ threshold 0.42 — T17C #4), +1–3 %
post-retrain when deep marginals rise: the joint rule re-lengthens offers toward [1, .85, .70, .50, .30] without
re-arming the flat 0.35 that F-228 measured negative twice. Port shape: ~50-line header + the two gate call sites
(`generate.cpp:14361-14364`, `mtp.cpp:1725` area) — algorithm port; the conditional graph is a separate M/L code
port gated on the DRAFT_TIMING capture. Conflict surface: F-230 width-mix law (battery, not M3b); our un-gated
twin and step-graph layout; COHORTS untouched.

### M2 — project-maya: the spec controller, the running DSpark, SPEC_SAMPLE / SPEC_CANCEL

**The controller is inherited, measured-priors code — and dormant.** `include/strata/spec/controller.hpp` is
byte-identical to basalt's modulo namespace, with one extra thing: provenance comments naming the measurements
behind the CostModel defaults — "The defaults are the 23 Sep measurements: dense ratio per token count from native
MMVQ (bench/results/2026-09-23-mmvq-multi), distinct experts per window from the routing trace
(2026-09-23-spec-economics), extra-token CPU cost 0.2-0.35 (2026-09-23-cpu-expert-multi)" (:12-15), and the
acceptance priors "from flyweight's measurement on this model (0.86 per draft); lookup by match length from the
offline replay (2026-09-23-suffix-lookup)" (`controller.cpp:21-22`). Their `draft_policy.cpp` kShape comment adds
"Between the RTX 5070's measured curves: +10 ms per token with every missed expert on the CPU
(bench/results/2026-09-27-spec/window-cost)". The CostModel: T(n) = dense_ms·dense_ratio[n] + (1−hit)·miss_ms·U(n)·
(1+extra_use·(n/U(n)−1)) + sync + k·draft_ms, with dense_ratio {1.0…3.3} and **U(n)/U(1) = {1.0, 1.70, 2.31, 2.88,
3.40, 3.89, 4.35, 4.80, 5.2}** — the distinct-expert growth curve, i.e., measured evidence that verify rows get
more expert-hungry with width. In Maya too the Controller is unwired (only `controller_test.cpp`); the live
arbitration is DraftPolicy + a flat per-request `spec_min_p` (default 0.9 for the DFlash decode path,
`generate.cpp:284-285, 2044-2045`). **Transferable here: the U(n) curve as a measurement target, not code.**

**The running DSpark (`src/ds4/`) — the most complete sibling implementation.**
- Confidence head: `out[t] = sigmoid(Σ w[i]·(i<dim ? x_t[i] : bf16(Embed[tok_t][i−dim])))` — one linear over
  [drafter hidden (dim=5120) ; previous-token embedding (R=d_rank)], a single dot + sigmoid
  (`ds4_kernels.cu:1256-1266`); weight = `confidence_head.proj.weight`, shape dim+R (`ds4_model.cu:3761`).
- Markov logit bias: `markov_w1` Embedding(V,256) of the previous token, `markov_w2` 256→V added per slot
  left-to-right (Xolotl audit (d); Maya mirrors DeepSeek's reference).
- Keep rule (both gates): `while (keep < n && keep < room−1 && conf[keep] ≥ conf_min && (cum *= conf[keep]) ≥ cum_min) ++keep`
  (`ds4_main.cpp:186-191`) — per-draft floor (default 0.5) AND cumulative product (default 0.5, env
  `MAYA_DS4_DSPARK_CUM` / request key `dspark_cum`).
- Cool-down: a round whose drafts all fell below the floor skips the drafter next round (`ds4_main.cpp:587-596`,
  `cool = try_spec && off == 0`).
- Verify: Leviathan/Chen with drafts drawn from q under `DS4_DSPARK_SAMPLE=1` (`ds4_model.hpp:110-121`).
- **Measured evidence (their comment, :161-165):** "0.5 by default: a Radeon 8065S, Maya-S, greedy over code /
  chat / a list, two runs: decode 12.04 → 12.82 tok/s (+6.5 %), the same tokens." That +6.5 % is the cumulative
  rule REPLACING a fixed verify width — i.e., it prices the rule where no offer gate existed. It does NOT price
  the rule against our flat 0.5 gate, which already harvests most of the same mass (Contradiction 1).

**SPEC_SAMPLE / SPEC_CANCEL (their SETTINGS.md, measured rows).** SPEC_SAMPLE (speculative sampling on the
two-GPU pipelined split): "drafts kept 68.6 → 74.7 %, decode +1.6–2.7 % alone, +5.3 % with SPEC_CANCEL (24.30 →
25.57 tok/s)" (SETTINGS.md :91). SPEC_CANCEL (head-group stops its speculative pass at the next layer once the
tail's token shows the draft wrong): "5.4–6.5 ms of the first GPU's time saved per rejected draft … decode +0.9 to
+2.6 %" (:90). Both are split-GPU-pipelining results — our verify is one batched forward with no mid-forward
cancel site; the cancel idea's single-GPU analog is offer-side truncation (M1). The +6.1 pp drafts-kept jump from
SPEC_SAMPLE is consistent with OUR F-235 negative, not contradicted by it (Contradiction 2).

### M3 — DeepInfer: DSpark confidence head + varlen scheduler (the algorithm, verified in code)

**In-repo surface.** Flags: `DEEPINFER_USE_NEXTN_DECODE = 5` (draft length, `environ.py:129`;
`config.py:224` sizes the batch as `max_num_seqs·(1+nextn)`), `DEEPINFER_ENABLE_MTP_VARLEN` (bool,
`environ.py:131`), `DEEPINFER_MAX_BACKTRACKING_TOKENS ≥ nextn+1` (:132). The drafter's graphs are FIXED-width
per captured batch size (`dflash_cuda_graph_runner.py`: bisect batch select, pad to `max_num_seqs`, replay) —
varlen lives entirely on the VERIFY side. Per-slot proposal storage with a confidence lane
(`mtp_info_pool.py:36-43`: `confidence` float32 buffer, init 1.0). `ConfidenceHead`:
`sigmoid((w·[hidden ; markov_emb]) / temperature[step_i])` with a PER-STEP temperature tuple
(`dflash.py:46-79`; env `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE`) — that per-step temperature is the
deployment form of the paper's Sequential Temperature Scaling; "HF DFlash 固定包含 confidence head" (always on,
:159). The Markov features: `markov_w1 = Embedding(V, 256)`, `markov_w2 = Linear(256, V)` logit bias
(`heads.py:26-34`); `AcceptRatePredictor(cfg.hidden_size + markov_rank)` on the last block (:105).

**The scheduler, exactly** (`tilelang_ops/schedule_mtp_varlen.py:35-98`, reference impl with a kernel twin):
per request r, survival a_{r,j} = cumprod(conf_{r,1..j}) (fresh requests remap to 0.0); flatten all (r, j);
**each sequence gets 1 base token + the count of its (r,j) positions inside the global top-(total_q − num_seqs)**
by survival probability (stable descending argsort — this is the paper's greedy, optimal because a_{r,j} is
non-increasing in j). Runs ON GPU from the previous step's confidence rates with a fused remap scatter
(`worker/base.py:280`; TileLang kernel for NPU, CUDA twin), pinned-buffer write-back — the production
"two-steps-prior" causal barrier from the paper §5.2. The cost curve SPS(B) is profiled once at init:
`profile_decode_perf()` (`model_runner.py:562+`; warmup 1 / measure 5 / trim 1 replays per captured batch size,
cached on disk). Verify seqlens_k follow from `num_accepted_tokens_gpu` (`base.py:285-291`).

**Their evidence (production, EP64/DP64, `doc/inference_deployment_qa.md:243-262`).** "Dynamic-length
verification cuts the number of draft tokens verified per step by about 30 % (5.00 → 3.2–4.4) while leaving the
average accepted length essentially unchanged (3.29 vs 3.20) … +4–7 % throughput at the production operating point
(~60 concurrent sequences per rank)." Table: conc 16 → **−1.5 %**; 32 → +1.6 %; 48 → +3.6 %; 60 → +5.4 %; 72 →
+6.7 %; 96 → +11.0 % (accepted length 3.32→3.24 at 16; 3.29→3.12 at 72). The 60–85 % figures are the FULL DSpark
system (drafter + scheduler) vs static MTP-1 per-user latency at matched throughput — not the scheduler's
marginal (T23-deepinfer already split these; the fetch confirms). Paper details (arXiv:2607.05147v1, fetched):
c* = 1 − ½‖p−q‖₁ as soft label + 0.9-weighted TV loss; head ROC-AUC 0.81–0.90; STS drops ECE to ~1 %;
lossless = non-anticipating admission (retrospective search leaks x_k into step k's admission — their Appendix A
counterexample); production γ = 5, offline γ = 7 (gains GROW with γ: +16–18 % → +22–30 % at γ = 15);
latency overhead of long blocks 0.2–1.3 %.

**Applicability / price.** The mechanism's gain is a function of (batch size) × (marginal-row cost curve). We are
at 6 slots / COHORTS 2 with ~6-row windows — left of their 16-seq crossover where the measured effect is NEGATIVE.
Single-stream it degenerates to the offer gate (M1). Hence 0–2 % conc, gated on the c_row(T) measurement: our
c_row 2.2–4.62 ms averages over widths; if the marginal row is flat in T, there is nothing to arbitrate; if it
rises (Maya's U(n) curve says expert reads grow ~5× by n = 8 in their offload regime — ours is 87–92 % VRAM-hit,
so much flatter), the budget rule has teeth. Do NOT port the two-step-deferred form: our COHORTS windows are
staged while idle and a one-window-old confidence is available without the CUDA-graph constraint they solve for.

### M4 — Xolotl: DSpark exact sampling, confidence-gated verify, split-MTP parity harness

**Exact sampling, point-mass form** (`tools/v41/engine/ds41_generate.cpp:365-382`): DSpark drafts by argmax
⇒ q is a point mass ⇒ Leviathan/Chen degenerates to "accept d with probability p(d); on rejection sample from
p without d" — `spec_accept()` with a rounding-safe residual walk; "the output distribution is the model's at any
temperature; unit-tested over 400k draws per case" (`docs/DEEPSEEK_V41.md:27-28`). Served as
`STRATA_SERVE_DSPARK` (on by default in their ds41 engine script).

**Confidence-gated verify** (`ds41_generate.cpp:3168-3186`): `while (kk < k && sig(sp_conf[kk]) ≥ conf_tau) ++kk;`
with `STRATA_DSPARK_CONF` 0.6 in serve — truncate the verify window at the first low-confidence draft.
"Every emitted token is still the trunk's own (lossless: only the verify width changes)" — the same
output-neutrality our coupled contract gives. `STRATA_DSPARK_CONFLOG` writes per-pass "k acc verify_ms p0..p4" —
an offline threshold-screen telemetry worth copying verbatim into any offer-gate work of ours.
**Their evidence** (RTX 4090 + 90 GB, tiered experts, 256-token decode): prose 9.5 → 17.3, code 18.8 → 25.5,
docs 29.9 → 30.3, python 26.6 → 29.1 tok/s — enormous BECAUSE "a rejected draft still costs its experts'
PCIe/NVMe bytes (~25 ms each)". Our marginal row is 2.47 ms — the same rule's headroom scales by that ratio.
Greedy DSpark itself: 1.32–1.53× over plain decode at depth 4 (:46-58).

**The offline DSpark audit** (`docs/ds4/V41_DSPARK_AUDIT.md`) is required reading for T14B's offline evaluator:
their harness reproduced the production convention exactly (verified against DeepSeek model.py, vLLM, and a ds4-ref
C implementation — conventions (a)–(d): main_hidden = hc-mean of the residual stream ENTERING target layers
37‖38‖39 → main_proj 15360→5120 + norm; block slot 0 = bonus, slots 1–4 = noise id, every slot predicts
sample_pos = query_pos+1; drafter-own window KV, non-causal; markov + confidence heads) and STILL measured 0.437
accepted/step until the real bug surfaced: **the checkpoint's `attn.wo_a` is FP8-with-E8M0-block-scales while the
reference module is bf16, and their loader dropped the scale (weights ~1000× off; every draft garbage)**. Fixed:
2.583 accepted/step, 3.58 tok/pass, first-draft==truth 77.1 %, conf quartiles 12/40/78/99. Lesson banked: a parity
harness that agrees with three references can still be wrong through WEIGHT LOADING — our G0 tensor-level
bit-gates exist precisely for this; keep them ahead of any end-to-end offline acceptance number.

**Split-MTP parity harness** (`tools/split_mtp_batch_parity.py:1-12`): dump/compare CLI over a running server —
5 fixed greedy prompts + one ~25k-token prompt × 256 tokens, exit-code compare; because the draft's K/V is
deliberately NOT bit-identical under the split (Q8_1×Q8_0 MMQ vs mmvq), only TARGET tokens must match bit-exact,
and the draft side is gated as "drafts accepted X of Y" within **±3 points** between arms. That two-tier gate
(target-bits strict, draft-acceptance statistical) is the right gate shape for OUR T14B rollout (P5) and any
future draft-KV precision change (T19 mmvq_il draft path, the STRATA_SPLIT_MTP_BATCH open question).

### M5 — DFlash2: five deep taps + hc-mean input (the drafter-input design)

**Exact capture contract** (`patches/qwen38-dflash2-capture.patch`, SGLang `qwen4_exp.py`): at zero-based
completed layers [5, 15, 25, 35, 45], AFTER the layer computes,
`hidden_states.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)` → aux_hidden_states.append(...) — i.e. the
MEAN OF THE FOUR HYPER-CONNECTION STREAMS of the COMPLETED layer output, 5 × 2560 = **12,800 features**; asserts
TP1/CP1; `set_dflash_layers_to_capture` hard-asserts the id list. Config: `block_size 8`, mask 248077,
selector_rank 256 / top-16, two-tap dynamic depthwise conv (kernel 2, group 16), 5 draft layers, full 248,320
vocab. Note the convention CONTRAST with DSpark (Xolotl audit (a)): DSpark taps the hc-mean of the stream
ENTERING target layers (inputs, `main_hiddens.append(h.mean(dim=2))` BEFORE the layer); DFlash2 taps COMPLETED
outputs. Neither is "correct" — each drafter is trained to its own convention — but for OUR backbone the
SGLang-supported capture path is completed-outputs, and our own engine's verify pass computes every layer of the
committed prefix, so taps at ANY layer set are available decode-side for the NEXT window's drafter steps
(the DSpark/Maya seed path proves the plumbing pattern: "seed() turns a window's taps (the streams' mean entering
layers 37..39) into it, through main_proj / main_norm and each block's wkv/kv_norm/rope/FP8" —
maya `ds4_model.cu:6901-6905`).

**Evidence and price for us.** DFlash2's claimed AL delta over stock MTP: +0.18/+0.42/+0.33 (their card;
below our T14B target +0.55–0.7 — T23-dflash2's verdict stands, this is NOT a T14B skip). DSpark's 3-tap
existence proof: 77.1 % first-draft accuracy and cleanly SEPARABLE confidences (quartiles 12/40/78/99) — the
separability is what makes confidence-gated offer shaping work at all. For our single-layer recursive head, the
port is a training-side input widening: harvest-time capture of k deep hc-means (the T14A exec-A dump extension
already walks prefill; taps add k buffer reads per position, prefill-only and free), trainer input 2×2560 →
(2+k)×2560, export adds k hc-mean tensors to the pack → **contract v2** (the 31-tensor SHA contract breaks
deliberately, versioned). Decode-side drafter step cost grows by k·2560-dot per step (~microseconds against a
~0.05 ms/step drafter). Band: +2–4 % M1 class ON TOP of T14B IF first/deep marginals move like the evidence
suggests; position-dependence caveat — our deep-position deficit is positional (F-235: 53.1 % deep vs 93.6 %
shallow at 121k), and deeper TAPS are not the same as deeper POSITIONS; taps buy conditioning signal, LongSpec
position-offset augmentation (T17F #3) buys the position axis. The two compose.

---

## Contradictions and their resolution

1. **Maya's +6.5 % cumulative-rule gain vs T17C's "+0–1 % standalone" for the same door.** Different baselines:
   Maya's rule replaced a FIXED verify width (no offer gate at all — pure "verify 5 always"); our flat 0.5 gate
   already implements a stopping rule within 0.08 of the SpecDec++ threshold. Both true. The rule's value to us is
   (a) the JOINT form (protects against a lucky shallow + doomed deep chain that flat per-draft floors admit), and
   (b) the post-retrain re-stair — priced +1–3 % there, +0–1 % standalone.
2. **Maya SPEC_SAMPLE +6.1 pp / +1.6–2.7 % vs our F-235 STRATA_SPEC_PROB −4.2 % M1.** Both correct: Maya's
   baseline drafts were the drafter's ARGMAX (q = point mass; sampling fixes a weak coupling), ours were already
   coupled samples at DT 0.85 (Σmin(p,q) ≤ coupled acceptance at sharp q — the ERSD Lemma 3.1 regime T17C §5.3
   documented). Resolution: rejection sampling pays where the draft RULE is weak; our closure stands and is now
   externally corroborated from both directions (Maya's gain exists only off the argmax baseline; our loss exists
   only off the coupled one).
3. **"+4–7 %" (Q&A) vs "60–85 %" (paper) for DSpark** — resolved in T23-deepinfer and confirmed by this pass:
   scheduler marginal (−1.5 %…+11 % by concurrency) vs full-system per-user latency at matched throughput. For us
   only the marginal band is honest, and only above ~16-way concurrency.
4. **T23-deepinfer attributed "verified tokens/step 5.0 → 3.2–4.4" to the paper; it is from the repo Q&A**
   (`inference_deployment_qa.md:247`, EP64/DP64 benchmark data). The paper's own budget statement is different
   (static-2 → ~4–6 under load). Corrected provenance; conclusion unchanged.
5. **DFlash2 taps completed-layer OUTPUTS; DSpark/Xolotl taps layer-INPUT hc-means.** Not a conflict — different
   drafters, each trained to its convention (the SGLang patch asserts DFlash2's; vLLM's code comment asserts
   DSpark's). For our port the choice is free but must match training; completed-outputs is the path our
   backbone's ecosystem already supports and our verify pass exposes naturally.
6. **Task framing: "maya's spec controller (E[tokens]/T(step), EMA acceptance, K_MAX=8)".** True as code, but the
   Controller is UNWIRED in maya AND basalt (only `controller_test.cpp` references it in both trees; both
   engines' published behavior is flat-window + DraftPolicy + keep rule). It is inherited Strata plan-v0.3-P6
   code whose value here is its MEASURED priors (U(n) curve, 0.86/position acceptance, RTX-5070 window-cost
   shape) — inputs to our budget-rule pricing, not a portable subsystem. The live arbitration in the family is
   DraftPolicy, where WE hold the more advanced variant.

---

## Open questions

- Our acc|dprob calibration: is a basalt-style bin table self-consistent on OUR coupled drafter (tight within-bin
  acceptance), or is the spread wide enough that a trained confidence head (M4) adds offer-shaping signal? The
  offline refit on existing logs answers this before either candidate costs a boot.
- c_row(T) on OUR stack: flat or rising with window width? One routing-trace boot (or COHORTS per-width telemetry)
  decides whether any varlen/budget mechanism has anything to arbitrate at 6 slots.
- Would the post-retrain drafter's deep marginals actually re-open widths 4–6 (F-228's s-curve), and does the
  joint rule beat the flat re-stair at the same offered mass? One G0.2 stair arm comparison on the SAME dump —
  the T17C falsifier, unchanged.
- Do DFlash2-style deep taps help POSITIONAL acceptance (our 53.1 % deep-position deficit) or only content
  conditioning? The offline probe-head training on harvested taps answers before any engine surgery.
- Does Maya's per-slot drafter cool-down matter at our 6-slot COHORTS shape (drafts cost at 6 slots, F-231)? One
  boot with per-slot telemetry; no instrument exists yet.
- Basalt's coupled default: their published numbers run `--coupled-draft` OFF (argmax drafts) — their 70.1 %
  accepted-drafts and the bin table are argmax-calibrated. If we ever diff acceptance against them directly,
  that asymmetry must be priced first.

## Implications

1. **Queue order unchanged, one refinement.** T14B retrain stays the king lever; the keep-rule offline refit
   (candidate #1's falsifier) should ride the SAME logs the T14B evaluator consumes — zero extra collection — and
   its boot (if the refit pencils) becomes the principled W2.5 re-stair rather than a flat schedule.
2. **The offer-side door is the only decode-loop door left, and it is exact by contract.** Every reference
   converged on shaping WHAT IS OFFERED (basalt joint rule, Maya floor+cum, Xolotl conf floor, DeepInfer global
   budget) and none on changing the acceptance rule — matching our F-235 closure from five independent directions.
   Any future proposal touching the verdict rule should be checked against this pattern before spending a boot.
3. **T14C+ multi-tap input is now specified end-to-end** (capture convention, evidence, contract cost, falsifier)
   — it enters the T14B decision tree as the named Arm-2-class rider if Arm-1 stalls, per T23-dflash2 ACTION 2,
   with DSpark's separability numbers as the gate expectation.
4. **Instrument debt, two items:** STRATA_DRAFT_TIMING capture (prices candidate #3 and the host-tail generally)
   and the routing-trace c_row(T) boot (prices candidates #5/#8 and the U(n) question). Both are one-boot;
   both have stalled once before (F-235 inconclusive/log-rotated) — re-run deliberately, log to files.
5. **Tooling adoptions:** the CONFLOG-style per-pass telemetry line and the split-parity two-tier gate
   (target-bits strict / draft-acceptance ±3 pts) go into the T14B rollout spec (P5) as written.

---

## Sources

**Code/history (primary, read this session; paths under `/home/svg/LLM/data/references/`):**
1. basalt `include/basalt/core/draft_cum.hpp` (whole file: bins :25-28, keep :31-36, base :39, scale :44-50, adapt :56-76).
2. basalt `src/spec/draft_policy.cpp` (:12-21 priors/shape, :36-48 cost EMA, :55-82 choose), `include/basalt/spec/draft_policy.hpp` (header rationale), `include/basalt/spec/suffix_drafter.hpp` (:8-24).
3. basalt `src/spec/controller.cpp` + `include/basalt/spec/controller.hpp` (K_MAX=8, CostModel priors) — dormant (grep: only controller_test.cpp references).
4. basalt `src/core/mtp.cpp:1495-1608` (draft loops; conditional-graph path :1536-1588; gate :1563), `src/kernels/cuda/verify_kernels.cu:816-847, 888-929` (mtp_draft_cond / mtp_step_end), `src/program/generate.cpp:4779, 6384, 8341-8417, 9262-9300` (wiring), `include/basalt/core/coupled_draft.hpp` (contract, `--coupled-draft` off by default).
5. basalt git: `241b5b4c` (2026-10-05 rule), `be7147f3`, `32e5522f`, `47f6192f`, `5ec431db` (adaptive default-on), `7af4a2c6` (2026-10-18 placement scaling), `1ec9b8b5`; `docs/precision/index.md:56` (width/precision disclosure); `docs/unified-kernel.md:32` (6.675 ms / 324.6 tok/s, spec 7).
6. project-maya `include/strata/spec/controller.hpp` (:12-15 measured priors), `src/spec/controller.cpp:21-22` (0.86 prior provenance), `src/spec/draft_policy.cpp:10-11` (RTX 5070 kShape provenance) — Controller dormant (grep).
7. project-maya `src/ds4/ds4_main.cpp:150-207, 580-600` (keep rule, cum default +6.5 % evidence, cool-down), `src/ds4/ds4_model.hpp:35-36, 110-121` (contract, SPEC_SAMPLE), `src/ds4/ds4_kernels.cu:1256-1266` (confidence kernel), `src/ds4/ds4_model.cu:3761, 6901-6905` (head weight; seed taps), `docs/SETTINGS.md:84-91` (MTP_KEEP 82-83 % vs 76-81 %; SPEC_CANCEL; SPEC_SAMPLE measured rows).
8. project-maya `src/program/generate.cpp:284-285, 2044-2045, 6340, 6866, 7580, 7603` (flat spec_min_p; DraftPolicy wiring; no draft_cum anywhere in tree).
9. DeepInfer `deepinfer/environ.py:129-132`, `config.py:13, 224`, `worker/dflash_cuda_graph_runner.py` (fixed-width drafter graphs), `model_executor/layers/mtp_info_pool.py:14-52` (confidence buffer), `model_executor/models/mtp/dflash.py:40-79, 159-195, 276-338` (ConfidenceHead, per-step temperature), `model_executor/models/v4_modules/heads.py:26-34, 70-108` (MarkovGPTHead, AcceptRatePredictor).
10. DeepInfer `model_executor/layers/tilelang_ops/schedule_mtp_varlen.py:11-98` (cumprod + global top-K scheduler, ref + kernel + tests), `worker/base.py:270-291` (seqlens by confidence), `worker/worker.py:120-135`, `worker/model_runner.py:562-600` (profile_decode_perf).
11. DeepInfer `doc/inference_deployment_qa.md:243-262` (production table: −1.5 %…+11 %; verified 5.00→3.2–4.4; accepted 3.29 vs 3.20; async scheduling description).
12. Xolotl `tools/v41/engine/ds41_generate.cpp:360-382` (spec_accept), `:3160-3200` (conf-gated verify + CONFLOG + DUMP), `tools/v41/engine/ds41_dspark.{hpp,cpp}` (block contract, expert-keep), `docs/DEEPSEEK_V41.md:20-75` (numbers), `docs/ds4/V41_DSPARK_AUDIT.md` (conventions a–d, wo_a scale bug, fixed numbers), `tools/split_mtp_batch_parity.py:1-12` (harness contract).
13. DFlash2 `config.json` (dflash_config), `patches/qwen38-dflash2-capture.patch` (completed-layer hc-mean capture, layers [5,15,25,35,45], features=12800).

**Campaign (our contract and constants):**
14. `campaign/research/T17C-spec-algo-leaps.md` §2 (contract map), §3 (coupling identity), §4-5 (ranked doors, VAT/LK/AdaSPEC, depth schedule), §6 (contradictions); `SVG-CUSTOM.md` F-233 scoreboard (:7998-8013: 74.9 % M3b, means), F-235 (:8074-8109: SPEC_PROB −4.2 % closed with data; depth-lane 53.1 % deep vs 93.6 % shallow; DRAFT_TIMING inconclusive), :1608/:3330 (V(T) = 9.7+2.47(T−1)), :1955 (drafts/window 3.1-3.2), :6166 (depths starve at floor 0.5), :8087 (SPEC_PROB priced out at 74.9 %).
15. `campaign/research/T23-basalt-portstudy.md` (calibration: their 8-bit-dense arm ≈ our 212.5; accepted/window 2.17 vs our ~2.47; candidate #6 analysis; coupled default off; dead-end scorecard).
16. `campaign/research/T23-{deepinfer,maya,xolotl,dflash2}.md` (recon verdicts banked 2026-10-20/21; T14C+ multi-tap ACTION; SPEC-related open questions); `campaign/research/T14C-fastmtp-acceptance-math-gates.md` (c_row, 0.72 rule, elasticities); `campaign/research/T14B-exec-*.md` + `campaign/T14B-specs/` (dump-lane, parity G0, rollout P5 — the riders' hosts).

**Literature:**
17. DSpark: Confidence-Scheduled Verification for Speculative Decoding — arXiv:2607.05147v1 (fetched HTML this session: c* = 1 − ½‖p−q‖₁ Eq.8, BCE + w_k = exp(−(k−1)/γ), α_tv = 0.9; STS → ECE ~1 %; Algorithm 1 greedy = optimal on non-increasing a_{r,j}; production two-step-prior barrier; non-anticipating lossless argument; γ=5 prod / 7 offline; gains grow with γ; ROC-AUC 0.81–0.90). Abstract page cross-checked.

Count: 17 numbered anchors (15 code/docs/history + 1 campaign-set + 1 literature), drawn from all five reference
repos, the campaign journal/specs, and arXiv — categories: code/history 13, documentation 6 (basalt precision,
maya SETTINGS, Xolotl DEEPSEEK_V41/audit, DeepInfer Q&A, DFlash2 card/patch), literature 1, campaign records 3.

## Methods note

All five repos read directly on disk at the MANIFEST pins (no fetches except the two arXiv pages). Wiring claims
("Controller is dormant") established by grep over each full tree, not by reading the test files alone. The DSpark
scheduler semantics were taken from DeepInfer's own reference implementation + its kernel twin + tests
(self-consistent triple), then cross-checked against the paper's Algorithm 1 description — they agree. Basalt
keep-rule provenance is git-first (commits read, dates and default-on flip identified) because no docs section
prices it. Unproductive / flagged: (a) arXiv abstract page lacks all math — use the HTML v1 for formulations;
(b) the "5.0 → 3.2–4.4" figure is Q&A-only, not in the paper (corrected above); (c) maya `draft_policy.cpp`
comment "+10 ms per token with every missed expert" reads like a units slip in their comment (their kShape values
are 0.0–3.6 relative units) — recorded as-is, not load-bearing; (d) Xolotl has NO engine-side `dspark` symbols in
`src/` — their DSpark lives in `tools/v41/engine/` (the ds41 sidecar engine), found via docs grep, not src grep;
(e) Maya/basalt `controller.cpp` diff is comment/namespace-only — verified by diff, so no behavioral divergence is
missed. Not done (out of scope): building anything, running any binary, reading DFlash2 weight binaries (LFS
pointers only), fetching SGLang/vLLM upstream sources beyond what the audit and patch quote.
