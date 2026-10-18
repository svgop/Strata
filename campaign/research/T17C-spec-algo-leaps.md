# T17C — Speculative-decoding algorithmic leaps for the chain-only coupled exact-match contract

Date: 2026-10-17. Lane: M1 single-stream (T16 campaign). Scope: algorithmic (not kernel) upgrades to the
draft/verify loop of strata-custom-2 on Qwen3.8-Flash-Next MTP, RTX 5090. READ-ONLY research: nothing built,
booted, or modified. Statuses of all CLOSED items respected (no EcoSpec tree-filling, no Bole, no SPEC_GUMBEL,
no tau-sweep re-litigation, no Controller re-queue, no draft-verify overlap).

---

## 1. Executive summary

1. **The contract, precisely bounded, has exactly three algorithmic doors.** After mapping every site where an
   upgrade could act (Section 2), a chain-only, position-keyed, one-pick-per-row coupled verify admits: (i) change
   what the DRAFTER is (training-side, output-neutral by construction); (ii) change HOW MANY rows are offered and
   where they come from (offer-side, output-neutral); (iii) change the ACCEPTANCE RULE itself (only the in-tree
   STRATA_SPEC_PROB rejection-sampling path is output-exact under our sampler). Everything else — branches,
   trees, list endorsement at a row — collapses arithmetically under exact-match + c_row ≈ 2.2–3.56 ms/row
   (Section 6 proves it; the proof, not the closure, is the new content).
2. **The single biggest leap is still training-side, and the r17 57% plateau is now fully attributed — it was
   never a distillation limit.** F-115/F-116's plateau = fresh-init + pure-KL + double-quant chain. Every leg has
   a published fix: warm-start from the shipped head (RH/FastMTP recipe form), direct acceptance-overlap losses
   (LK, ICML 2026 — designed for truncated sub-vocab heads exactly like ours), verification-aware training (VAT,
   arXiv 2608.30135 — simulate the verify loop in training, re-anchor the depth-decay at the first rejection,
   +11.4% AL on EAGLE-3/DFlash), and selective filtering (AdaSPEC, NeurIPS 2025). These are riders on the already
   GO-queued FastMTP retrain, not alternatives to it.
3. **The tree/branch question is not "closed by fiat" — it is closed by arithmetic that flips against trees as
   the drafter gets better.** A 2–3-branch-per-row window under coupled exact-match can only hedge the drafter's
   conditioning error (≤1 child matches the position's single pick); the hedge's expected value is
   (1−m)·P(pick ∈ top-C)·(continuation) against a full row of cost, so it pays only when per-position acceptance
   m is LOW (< ~0.6). FastMTP-class training raises m ABOVE the tree's break-even — the retrain structurally kills
   the tree case. The precise verify-contract surgery a tree would need is specified anyway (Section 6), with the
   two re-openers named.
4. **Multi-proposal per row under exact-match is worthless (≤1 survivor per position), but the in-tree, never-
   measured STRATA_SPEC_PROB path legalizes it at ZERO extra verify rows** at the window's bonus/tip row, and
   raises per-row acceptance from the coupled-race bound D_HM[P,Q] to the Metropolis bound Σmin(p,q) (ERSD
   Lemma 3.1). Both effects are second-order at sharp q (the SPEC_GUMBEL +1.5pt result is the measured size of
   the coupling-rule class at min-p 0.5) — so this is a cheap boot, not a leap: +0.4–0.8% M1 class.
5. **The hybrid-drafting answer to task item (d) is mostly already in the tree and asleep.** Suffix-drafter
   windows with expected-tokens/ms arbitration (DraftPolicy), a lookup-chain appended after the MTP drafts
   (cap 7, default 0), and a registry for external retrieval draft sources (draft_source.hpp) all exist. What is
   missing: per-lane arming (mimic/agentic), a registered retrieval source (zero engine code), and batch-lane
   lookup rows (a lookup draft costs no drafter step, so a 2-row slot could carry one free).
6. **Nothing in the 2024–2026 literature beats a depth-6 coupled chain at exact-match on this cost structure.**
   Published frontier numbers (EAGLE-3 τ 5.8–6.6, JetSpec 9.6×) live in a different cost regime: rows are ~17%
   of window there, ~26–30% here (T14C §2.1). The transferable frontier is entirely in objectives (VAT/LK/
   AdaSPEC), offer shaping (depth schedules, batch-aware depth), and near-free rows (n-gram).

Ranked table in Section 4; deep dives on the top five in Section 5; 44 numbered sources in Section 8.

---

## 2. Contract map — what an algorithmic upgrade must satisfy

Every fact below was read in code this session. Line numbers are from this tree.

### 2.1 Window shape (serial M1 lane)

- A verify window is a FLAT CHAIN of T ≤ 8 rows: `window[0] = x` (the committed seed), `window[i] = draft i−1`
  for i ≥ 1. Built at generate.cpp:14403-14411 from `mtp.draft()` output; row cap `kVerifyMaxT = 8`
  (include/strata/kernels/verify_kernels.hpp:21); `DraftPolicy::kMaxT = 8` (draft_policy.hpp:24);
  `--spec 6` ⇒ S_mtp = 6 with `--mtp-max-t` able to cap lower (generate.cpp:14337-14338).
- Offer-side gate (the de-facto adaptive chain length): the serial loop trims T to the number of leading drafts
  whose own-draw probability survives the floor — `while (T < S_mtp && dprob[T-1] >= spec_min_p) ++T`
  (generate.cpp:14361-14364). The drafter's chain loop mirrors it per step (mtp.cpp:1725-1743; the gated twin's
  un-gated twin at :1631-1665; STRATA_DRAFT_UNGATED = same graphs, launch-shape only — measured −2.73% M1,
  F-229). **The gate is a FLAT threshold per request; there is no per-depth schedule anywhere.**
- Second source composition: a suffix-lookup window may REPLACE the MTP window (DraftPolicy.choose,
  E[tokens]/ms with margin, generate.cpp:14368-14375; draft_policy.hpp:7-15), and `--lookup-chain K` (clamp 0..7,
  generate.cpp:2221) APPENDS up to K n-gram tokens after the MTP drafts, admitted by `policy.chain()` only when
  E(t_mtp) + p_mtp·(c+…+c^k) beats the MTP window alone by margin (generate.cpp:14382-14394;
  draft_policy.hpp:38-44). Extra retrieval sources register via `extra_draft_sources()` (draft_source.hpp:7-14,
  wired at generate.cpp:1033-1047) — **a codebase/retrieval drafter needs zero engine code.**

### 2.2 Verify and the verdict

- One batched forward over all T rows (Verifier::run, verify.cpp:2301+); row t attends to the committed context
  plus rows 0..t (the chain prefix) — a row's K/V/GDN inputs are its ancestors, which is why a row's validity is
  prefix-conditioned and why the verdict is a prefix scan.
- Per-row output: host-side sampling after the replay — row t draws Philox(seed, pos0 + t) over the request's
  full sampler chain (penalties → top_k → top_p → min_p → temperature → inverse-CDF; verify.cpp:2501-2542;
  sampler.cu sampler_kernel KMAX = 64, sampler.cu:296-445). **One pick per row.** The position-keyed counter is
  what makes drafts output-neutral: "the output is a function of the seed alone, whatever the drafts are"
  (coupled_draft.hpp:9-12).
- Verdict: linear prefix match on the host — serial `while (a < T-1 && window[a+1] == outv[a]) ++a`
  (generate.cpp:14449-14450); pipelined twin at :12957; batch `keep[b]` prefix with per-slot rescue
  (generate.cpp:10310-10338; typical/ratio rescue = the only lossy rows in the engine, F-200 device rescue).
- Commit: `ver.commit(a+1)` advances GDN/QSA/KV state over exactly the accepted prefix + bonus
  (generate.cpp:14471; batch `commit_slot_prefixes(keep[])` :10339).

### 2.3 The coupled drafter (standing)

- STRATA_SPEC_COUPLED=1: the drafter samples with the target's chain and the SAME Philox uniform the verifying
  row will draw; draft made at cell c is verified by counter c+1 (coupled_draft.hpp:19-30, :86-89); penalty ring
  mirrored device-side (:33-36). Drafter chain runs at 0.85× request temperature (STRATA_SPEC_DRAFT_TEMP,
  spec_prob.hpp:55-62; F-201 adopted +13.4/+9.8/+7.5%).
- The drafter's q list (≤64 sorted entries) is already materialized in coupled/prob mode
  (mtp.hpp:203; sampler.cu coupled q-emission kernel :947-1016) — any "use the drafter's top-k/list" idea has its
  data path already on device.
- STRATA_SPEC_GUMBEL (race/Gumbel picks both sides) exists (sampler.cu:67-73) and measured +1.5 acceptance
  points, not more — reverted (F-177). This is the measured size of "change the coupling rule" at min-p 0.5.

### 2.4 The unmeasured in-tree path: STRATA_SPEC_PROB (rejection sampling)

- spec_prob.hpp implements Leviathan/Chen rejection sampling for MTP rows: draft d sampled from q (own counter
  bit 1<<61), accepted with min(1, p(d)/q(d)), else residual sample from norm(max(0, p−q)); three disjoint
  counter spaces per row (u_pick / pos|1<<62 / pos|1<<63, spec_prob.hpp:23-32, :79-81). q = the drafter's own
  post-chain list (≤64); "NO q MEANS POINT MASS, NEVER q = 0" keeps lookup/suffix rows correct (:19-21).
- Verify hookup exists end-to-end: generate.cpp:13203 stages the q lists; verify.cpp:2513-2536 uploads them and
  calls sample_tokens_spec (sampler.cu:1361+). Greedy requests unaffected; bit-check gate (G0.1) unaffected.
- **Never booted:** appears only in T15C's never-swept env inventory (T15C-overlooked-wins.md:793-795) and the
  old decision-table queue (SVG-CUSTOM.md:4778); no results file references it. Only SPEC_GUMBEL was measured.

### 2.5 Batch slot lane (F-184 shape)

- Each active slot carries TWO rows: its committed token + one MTP draft (rotation at generate.cpp:10440-10458);
  the slot drafter serves "the one-proposal sampled shape only" (mtp.cpp:1790, draft_launch/draft_finish split,
  F-206). Rows are staleness-set per F-231; batch depth is a lane of its own (SmartSpec/TurboSpec/SGLang all ship
  load-dependent depth; our conc post-COHORTS anchor is 312.4).

### 2.6 The two laws an upgrade must respect

- **Row-cost law:** a marginal offered row costs c_row ≈ 2.2–2.6 ms post-POOL_DIET (T15A), 3.56 ms in the s35
  calibration, 4.62 ms conc (T14C §1.3); W(1) = 8–9 ms; a marginal row must carry
  > c_row·C₀/W₀ ≈ 0.72 expected tokens (→ ~0.56 at c_row 2.8) or it is net-negative (T14C §1.4).
- **Exactness law:** every adoptable change must keep the emitted tokens p-exact (or byte-identical under
  greedy). Coupled exact-match guarantees it for ANY draft set; STRATA_SPEC_PROB guarantees it for ANY proposal
  distribution including point masses; nothing else on the 2024–2026 market does both.

---

## 3. The coupling identity that governs "branches per row"

This is the analytical core (derived; anchored in published bounds — sources 33, 34, 35):

- Under coupled exact-match, a proposal x at a position survives iff x equals that position's single pick
  (Philox(seed, pos)), pick ~ p. For ANY proposal set {x_1..x_C} at one position, at most ONE member can equal
  the pick, so P(row survives | C proposals) = p(pick ∈ set) ≤ max_i p(x_i) — **the argmax of p is the optimal
  C-proposal, i.e., multi-proposal per row under exact-match is exactly worthless.** Extra rows only help by
  pre-computing CONTINUATIONS of proposals that might survive at earlier positions — i.e., a tree.
- A tree's hedge at depth k pays only when the chain breaks there: P(break at k) = s_k·(1−m_k), and the rescue
  needs the true pick to be among the C children AND the rescued continuation to repay the rows. With standing
  offer/acceptance (s = [1, .63, .35, .17, .08], m = [.77, .86, .85, .68, .75], T14C §1.2): rescue-eligible
  breaks ≈ Σ s_k(1−m_k) ≈ 0.22/window, but the hedge costs (C−1) rows at EVERY hedged depth, every window, at
  c_row ≈ 2.2–3.56 ms. Generalizing over m: hedge value per depth ≈ (1−m)·P(top-C)·(chain value beyond);
  hedge cost ≈ m (the depth given up). The crossover sits near m ≈ 0.5–0.6. **A better drafter pushes m up and
  moves the tree FURTHER from break-even** — the FastMTP retrain kills the tree case structurally. The tree
  becomes rational only in low-acceptance regimes (high-temp creative sampling; uncoupled argmax drafting) or at
  c_row ≲ 1.2 ms (both re-openers named in Section 6).
- Under rejection sampling the picture inverts: each proposal is judged with its own uniform
  (counter | branch bits — bits 59-60 of the 64-bit counter are free; only 1<<61..1<<63 are used,
  spec_prob.hpp:79-81) with acceptance min(1, p/q) and the first acceptance is committed — ERSD's Lemma 3.1 gives
  per-row acceptance bounds [1−D_TV, D_HM] for the race coupling vs Σmin(p,q) for Metropolis, with
  Σmin ≥ D_HM (source 33). But the prefix-forward constraint still limits multi-proposal to the chain TIP: an
  alternate accepted at row t invalidates rows t+1's computed forward (their input token differs), so mid-chain
  multi-proposal = tree again. The tip/bonus row has no continuation INSIDE the window — it is the one
  multi-proposal site with zero forward cost (Section 5.3).

---

## 4. Ranked candidate table

Magnitude = M1 lane unless stated; priced with T14C §2 elasticities (+0.80%/pp per-proposal acceptance M1,
windows flat). Effort: S = ≤1 day + battery; M = multi-day code; L = project.

| # | Candidate | Mechanism | Priced magnitude | Effort | Cheapest falsifier | Sources |
|---|---|---|---|---|---|---|
| 1 | **Warm-start retrain + VAT objective** (beyond the queued vanilla FastMTP) | Train the served recursion against the SERVED verify loop: verification head predicts per-position survival; decay re-anchors at each sample's first rejection (replaces fixed β^(k−1)); warm-start from the shipped head (never fresh-init — the r17 plateau leg) | Retrain base: +5.7% M1 (T14C (a)); VAT rider: +11.4% AL class on EAGLE-3/DFlash ⇒ ~+1.5–3% M1 on top; total realistic band 225–232 | M (rides W2's infra; VAT = objective-only swap) | G0.2 stair read of the VAT arm vs β-arm on the SAME dump: pos-2/3 marginals +≥8pp and dp curve flatness | 17, 18, 39, 40, 42; F-115/116 |
| 2 | **LK overlap-loss arm** (KL→TV hybrid / −log α) on the retrained head | Directly optimize Σmin(p,q); handles truncated sub-vocab heads (off-vocab mass = 0 in α, no KL blowup) — ours is 40,525 rows | +3.5–8.2% τ in-paper ⇒ ~+1–2% M1 as Arm-1 rider; biggest at depths 4–6 | S+ (needs the verifier p-list dump — T14A Arm 0 ext.) | 1-epoch warm-start arm at LR 2e-5; gate = depth-4/5 conditional marginals | 41; T14A Arm 2 |
| 3 | **AdaSPEC selective filtering** | Reference checkpoint (Arm-1 output) identifies hard tokens; retrain on top-40% by loss gap; KD temp ~0.7 | up to +15% acceptance relative in-paper ⇒ ~+1–1.5% M1; costs one reference pass (free: Arm-1 IS the reference) | S after Arm 1 | Same G0.2 stair, filtered vs unfiltered | 40; T14A Arm 3 |
| 4 | **Depth-rising min-p schedule** (offer-side shaping; the Controller's feed-forward successor) | Replace flat spec_min_p 0.5 with per-depth m_k (rising), tunable against the 0.72-tokens/row crossing rule; post-retrain the deep marginals rise and the schedule re-lengthens offers 2.2→3.5 | Standalone +0–1% (flat gate ≈ threshold-optimal: SpecDec++ threshold = c_row/(c_row + W₀/C₀) ≈ 0.42 < 0.5); post-retrain +2–5% (T14C scenario (b) re-priced at c_row 2.2–2.6) | S (one knob through mtp.cpp:1715 + t_rule) | One STRATA_SPEC_DEPTH=1 boot at schedule {0.5, 0.45, 0.4, 0.35, 0.3} vs flat — stair read answers before any 3v3 | 25, 26, 27, 28, 47; T14C §1.4, T15A |
| 5 | **STRATA_SPEC_PROB boot** (in-tree, never measured) | Rejection sampling for MTP rows: per-row acceptance D_HM → Σmin(p,q) (ERSD Lemma 3.1); DT re-optimized at the rejection optimum (q = p, not 0.85-sharpened) | +0.4–0.8% M1 class (bounded above by the coupling-rule class measured at +1.5pt, F-177; second-order at sharp q) | S (one env line + battery; greedy untouched) | One boot, one battery, M3b + M1 + dp read; kill on any conc drag | 13, 14, 33; spec_prob.hpp; F-177 |
| 6 | **Tip k-arrival multi-proposal under SPEC_PROB** | At the window's last row only, judge the q-race's first k arrivals (list already on device), first min(1,p/q) acceptance commits — zero extra forward rows; the committed token also seeds the next chain at a higher-q token | +0.3–0.8% M1 (bonus-row union ≈ +5–10pt × full-chain window share) | S–M (extend sample_tokens_spec to k drafts on one row) | Same boot as #5, second arm; read dp_ok[0] shift (better seeding) | 33, 34; spec_prob.hpp:126-158 |
| 7 | **Hybrid lookup arming** (in-tree, asleep) | Per-lane STRATA_SUFFIX_DRAFT/--lookup-chain on mimic/agentic lanes; register a retrieval DraftSource over a codebase (zero engine code); DraftPolicy already prices MTP-vs-lookup per window | +5–15% on mimic-class traffic; ~0 on the standard battery; batch variant (free lookup row in the 2-row slot) = conc rider | S (arming) / M (batch row) | One mimic-lane boot with --suffix-draft 6 --lookup-chain 4; the policy's own telemetry (sfx_windows/sfx_ok) decides | 9, 10, 60, 36, 37, 38; draft_policy.hpp |
| 8 | **2–3-branch verify surgery (tree)** | Full spec + EV in Section 6: parent[] on rows, forked GDN scan, per-branch counters, tree-prefix verdict | DEAD at standing economics (−1.5 to −6% M1 at C=2, every window taxed by 1–2 rows × c_row); re-opens only at c_row ≲ 1.2 ms or m ≲ 0.6 regimes | L | Do not spend a battery; the EV table IS the falsifier. If ever c_row < 1.5 ms post-kernel-tranche: re-run Section 6 table | 15, 16, 21, 22, 23, 24, 29, 30, 31, 32, 43, 50, 51 |
| 9 | Batch-aware depth (conc rider) | Load-dependent slot depth (SGLang ships draft_token_num↓ under load; Nightjar bandit +14.8%/−20.2%) — our 2-row F-184 shape + F-231 stale-slot law already encodes the first step | conc +0–8% band (out of T17C's M1 lane; F-231 instrument kept) | M | F-231's own nrows[] telemetry | 27, 47, 56, 57 |

Not re-queued (closed or demoted, respected): EcoSpec tree-filling, Bole factor-tree, SPEC_GUMBEL re-roll,
DT-sweep re-litigation, Controller E[cost] estimator (the schedule in #4 is its feed-forward replacement at
zero estimator risk), draft-verify overlap (SSD/Mirror-class — draft is 6–9% of window here).

---

## 5. Deep dives (top 5)

### 5.1 The plateau-breaker stack: warm-start + VAT + LK + AdaSPEC on the queued retrain

**Why the r17 distillation stopped at 57%** (F-115: 128/223 = 57.4% first gate; F-116: 8 epochs/12k windows =
55.7%, KL 2.31 → 2.05, FLAT): three compounding causes, all named in F-116 and all now addressable:
(a) fresh-trained weight geometry is quant-noise-sensitive — the stock head's OWN weights survive the same
double-quant chain at 85.4%; (b) pure-KL ceiling; (c) the chain's quant passes. The queued FastMTP retrain
(T16 W2) fixes (a) implicitly (it finetunes the existing 31 tensors — warm-start by construction) but keeps a
FIXED β^(k−1) depth weighting and a CE/KL objective.

**What is new since T14A was written (the beyond-FastMTP content):**

- **VAT — Verification-Aware Training (arXiv 2608.30135, Naver, code promised at github.com/naver-ai/vat).**
  Simulates verification at every training step and turns accept/reject patterns into supervision: (i) a jointly
  trained lightweight binary head predicting whether each position survives sequential verification; (ii)
  verification-adaptive weighting — full weight up to each sample's first rejection, decay re-anchored THERE.
  This directly replaces the β=0.6-vs-0.8 extrapolation debate (T14A F14 was flagged as an armed
  extrapolation with no published run): the schedule becomes per-sample and learned. Reported: up to +11.4%
  average acceptance length and +8.7% wall-clock on EAGLE-3 and DFlash over Qwen3-4B/8B and Llama-3.1-8B,
  objective-only, no architecture change. The T14C gate concern it speaks to: "confidence-coupled chain
  extension taxes 1–2pp per +0.3 offered rows unless deep marginals hold" — VAT trains exactly the survival
  pattern the gate acts on.
- **LK Losses (arXiv 2602.23881, ICML 2026; Nebius ships LK-Speculators on HF).** Acceptance α = Σmin(p,q) =
  1 − TV(p,q); KL and α share the optimum only at capacity ∞; capacity-limited drafts converge where lower KL
  coexists with LOWER α (their toy: 50.2/50.8 vs 60.2% for TV-optimal). Practical losses: adaptive
  λ·KL + (1−λ)·TV with λ = exp(−η·sg[α]), or −log α; +3.5–8.2% τ over KL across six targets, +5.6% on a
  DeepSeek-V3 MTP finetune. Two properties map EXACTLY onto our head: truncated-vocab drafts are handled
  natively (off-vocab contributes 0 to α — no KL blowup; our 40,525-row sub-vocab head), and greedy draft
  sampling underestimates acceptance at T > 0 (their hygiene note; our coupled drafting sidesteps it by
  construction — T14A F7).
- **AdaSPEC (arXiv 2510.19779, NeurIPS 2025).** Two-stage selective KD: the reference draft (our Arm-1
  checkpoint, zero extra cost) identifies learnable-but-unlearned tokens by per-token loss gap; retrain only on
  the top-k fraction (k = 0.4); up to +15% acceptance over plain KD with LESS total training. Their ablation
  that training on the BOTTOM 40% actively hurts is the guard-rail for our small-data regime.
- **Warm-start and the teacher.** The published recipes are ALL warm-start finetunes of shipped heads (RH
  speculators 0.6.0: extract native mtp.*, finetune recursively, stitch back; FastMTP: train the shipped head).
  The r17 chain's fresh-init distillation is the outlier that produced the plateau. Teacher choice resolves to:
  train against the SERVED stack's hiddens (our IQ3_S quantized backbone — the RH "online hidden-state
  extraction" principle taken one better: distribution-matched by construction, T14A F16) — not against an
  upstream bf16 head's logits through a proxy.
- **Data scale is NOT the binding constraint.** RH's own example: 5,000 samples / 443 s on 2×H200 moved pos-2
  +0.14 on Qwen3-Next-80B-A3B (Red Hat article, fetched this session). Our pairs-r17 corpus is 111,071 records
  (22× that); FastMTP's 389.4K is the "scale" endpoint, not a requirement. Spend the effort on objective and
  init, not corpus growth.

**Priced composite:** T14C's scenario (a) (+5.7% M1 central, +2.8% pair, +2.7% conc) is the vanilla-retrain
band. VAT/LK/AdaSPEC riders stack multiplicatively on ACCEPTANCE LENGTH, not on t/s: realistic landing
225–232 M1 before depth re-unlock; with W2.5's re-stair (offers 2.2 → 3.5) the T14C (c) band (+8.7%, range
−3..+17, re-priced at post-diet c_row) opens. Gates: T14C's G0–G5 ladder unchanged; add one arm swap
(β-fixed vs VAT) at G0.2 — the stair read is the falsifier for the whole stack.

### 5.2 Offer-side shaping: the depth-rising min-p schedule (the Controller's surviving idea)

The Controller was measured dead (0–3% ceiling) as an E[cost]/ms ESTIMATOR with lookup arbitration. Its
feed-forward skeleton survives in a form with no estimator to fail: the offer gate is a flat threshold
(mtp.cpp:1725, generate.cpp:14361-14364); the optimal stopping theory (SpecDec++'s threshold rule; TurboSpec;
AdaEDL; DISCO; SVLP — sources 25–28, 46) says: offer row k+1 while the PREDICTED marginal acceptance ≥
c_row/(c_row + W₀/C₀). With our constants: c_row = 3.56, W₀/C₀ = 13.74/2.785 ≈ 4.96 ⇒ threshold ≈ 0.42 — the
flat 0.5 gate is already slightly conservative; the schedule's standalone headroom is the ~0.08 threshold gap
on deep rows only, i.e., +0–1% M1. POST-retrain the calculus changes: deep conditionals are predicted to rise
to 0.80–0.92 (T14C §4), the flat gate becomes the binding constraint on offered mass (F-228's
s = [1, .63, .35, .17, .08] can re-open toward [1, .85, .70, .50, .30]), and the schedule generalizes W2.5's
flat re-stair by letting depth k carry its own gate — the difference between one more stair boot and a knob
that tracks the drafter's depth profile. Cheap falsifier: one STRATA_SPEC_DEPTH=1 boot reads m_k per depth for
any schedule before any 3v3 (the stair instrument already exists; T14C G5). Keep it OUT of the queue until the
retrain lands — standalone it is a zero-to-noise lever, and F-228's two measured negatives bracket the flat
0.35 variant.

### 5.3 The tip multi-proposal under STRATA_SPEC_PROB (the zero-row multi-draft)

The only multi-proposal form the chain contract permits without new rows (Section 3): under rejection sampling,
proposals are judged independently (own counter bits) and any acceptance is p-exact, so the window's LAST row —
the bonus position, whose output seeds the next chain — can judge the k first arrivals of the drafter's q-race
(the sorted q list is already on device; sampler.cu:947-1016; ERSD Algorithm 5's "successive arrivals" are
exactly the inverse-CDF order of q). First acceptance commits. Three effects, all small but all positive and
compounding with a better drafter: (i) per-row acceptance of the judged row rises from the single-sample level
to the k-arrival union; (ii) Σmin(p,q) > D_HM (ERSD Lemma 3.1) lifts every judged MTP row, not just the tip;
(iii) the next chain starts conditioned on a higher-q token more often (measurable as a dp_ok[0] shift).
Bounded honestly: the coupling-rule class measured +1.5pt (SPEC_GUMBEL, F-177) and that is the same
"second-order at sharp q" regime — this candidate is a cheap boot riding #5's env line, not an independent
leap. Surgery list: STRATA_SPEC_PROB=1; extend sample_tokens_spec (sampler.cu:1361) to take k draft rows for
row T−1 (kernel already loops drafts; the q lists are uploaded today, verify.cpp:2513-2536); the next
chain_launch conditions on the committed token — unchanged code. Note the DT interaction: under Metropolis the
optimal draft temperature is q = p (DT = 1.0), NOT the 0.85 sharpening (which maximizes the exact-match
coupling, not Σmin) — the sweep-direction dependency the campaign has already banked; cite, don't re-derive.

### 5.4 The hybrid lookup lane: already built, asleep, and lane-shaped

The engine ships the entire vLLM/TRT-LLM hybrid-drafting surface in-tree: a trigram-index suffix drafter with
backward extension and match-length buckets (suffix_drafter.hpp:8-15), expected-tokens-per-ms online
arbitration between MTP and lookup windows with per-match-length acceptance EMAs (draft_policy.hpp:7-15), an
append-after-MTP lookup chain capped at 7 with its own E-rule (draft_policy.hpp:38-44,
generate.cpp:14382-14394), and a registry for external draft sources fed the committed context
(draft_source.hpp:7-14 — "other sources (a retrieval store over a codebase or earlier conversations) register
with extra_draft_sources() before the first request"). All defaults OFF; DraftPolicy measured lookup windows
losing 2–8% on ordinary text (draft_policy.hpp:7) — which is the point: it is lane-conditional value. The
literature agrees the value is workload-shaped: SuffixDecoding (NeurIPS 2025 spotlight) up to 5.3× on agentic
repetitive traffic vs ~2.9× open-ended and deployed at Snowflake; Token Recycling recycles REJECTED candidate
tokens through an n-gram pool (our rejected drafts are currently discarded — a recycle pool is a ~100-line
addition to the suffix drafter's table); TRT-LLM ships n-gram with auto-enablement; SAM Decoding adds a suffix
AUTOMATON (no WAYS-cap on matches). Three concrete moves, in cost order: (1) arm --suffix-draft/--lookup-chain
on the mimic/agentic lanes only (env-only, the policy telemetry self-reports); (2) register a retrieval
DraftSource over our own battery/mimic corpus (zero engine code — the registry exists); (3) M, batch lane: a
lookup row costs no drafter step, so the 2-row slot shape could carry a lookup draft instead of / beside the
MTP draft when the suffix drafter hits — the only way the conc lane gets hybrid drafting without touching the
F-184 contract. Verdict for the M1 battery lane: ~0% — do not arm globally; the DraftPolicy margin already
protects against the ordinary-text regression.

### 5.5 The 2–3-branch verify surgery: precise spec, priced, closed — with re-openers

**What would have to change** (for the record; Section 3 proves the EV):
1. Window representation: `tok[T]` + `parent[T]` (row 0 parent −1); sibling rows share position pos0+depth and
   therefore share the row's pick and counter. All three loop shapes (serial/pipelined/batch) carry the window
   as a flat chain today (generate.cpp:14403-14411, 12810-12886, 10440-10458).
2. GDN/QSA replay: rows are scanned as a sequence; a fork needs per-row recurrent state = the SGLang
   slot-per-draft-token design (T7.B). F-216's slot-promotion snapshot kernel (built, bit-parity proven,
   perf-negative for chains) is exactly the fork primitive a tree needs — the machinery exists.
3. Sampler: unchanged per row (one pick per position); the verdict rule changes from "tok[row] == pick" to
   "tok[row] == pick AND parent accepted"; the host verdict becomes a mark-scan + deepest-accepting-path
   extraction over parent[] (trivial); commit = the path's prefix + bonus.
4. Drafter: chain_launch gains branch launches — the C step-graphs at a hedged cell condition on the q-order's
   first C tokens instead of its own draw (the top-2 tokens are already tracked: top2 diagnostic,
   mtp.hpp:73-75); step graphs are captured per-step-index, so C branches need C step slots per depth inside
   the kVerifyMaxT = 8 row budget: depth-6 chain already spends 6 of 8; a C=2 fan at one depth spends 7.
5. Cost: each branch row = one more row in the SAME batched forward = c_row 2.2–3.56 ms + drafter step ~0.05 ms
   (F-229); expert routing distinct-expert growth is sublinear but nonzero; the QSA/GDN per-row lines scale.
**EV (standing calibration):** C=2 fan at ONE depth, best site depth 1 (P(break) = 0.23, offered 1.0):
+1 row × every window (≈ −2.2 to −3.6% M1 via c_row/W₀) vs rescue ≈ 0.23 · P(pick ∈ top-2 | break) ·
E[continuation ≈ 2–3 accepted] ≈ +0.4–0.6 tokens on 23% of windows ≈ +1.0–1.5% — net −1 to −2.5% M1. Fans at
every depth are worse (cost compounds per window, rescue mass doesn't). **Re-openers:** (i) c_row ≲ 1.2 ms
(kernel tranche beyond T16 W1 — then the 0.72-tokens/row threshold drops to ~0.25 and rescue pencils);
(ii) any lane where m ≲ 0.6 (uncoupled argmax drafting at high temperature; the coupled contract's acceptance
is the collision Σp·q, which collapses at temp ≥ 1) — at m = 0.5 the same table flips to +1 to +3%. Under the
standing 74.9% acceptance and the queued retrain (which RAISES m), this stays closed. The literature's tree
wins (EAGLE-2/3 dynamic trees, Sequoia's wide-shallow theorem, OPT-Tree, SpecExec, JetSpec, HeteroSpec) all
live where rows are cheap relative to the window (dense 7B–70B targets, A10/H100) or where the drafter is weak
enough that hedging is the only depth available — neither is our regime (T14C §4: "EAGLE-3's τ 5.8–6.6 is not
a target for this engine").

---

## 6. Contradictions and their resolution

1. **"FastMTP is the biggest lever" (T11.R3/T16) vs "the retrain alone lands 223–229, not 250" (T14C).** Not in
   conflict — T14C re-priced the depth component; this dossier adds the third position: the retrain also moves
   the STRUCTURAL constants (deep marginals up ⇒ offer gate re-opens ⇒ W2.5; m up ⇒ tree EV down ⇒ the branch
   question dies on its own). Sequence stands: retrain → re-stair → (only then) offer-side schedule.
2. **Multi-draft theory says k drafts help (Khisti/Hu/Thomas/Timor: k=2–10 gains) vs our Section 3 "worthless
   under exact-match."** Both true: the multi-draft canon assumes independent per-draft accept/reject draws
   (Metropolis/race rules). Under a single shared uniform with exact-match, ≤1 proposal per position can
   survive by construction. The canon becomes applicable EXACTLY when STRATA_SPEC_PROB is on — which is the
   hidden significance of the unmeasured in-tree path.
3. **SPEC_GUMBEL "+1.5pt, reverted" vs GLS/ERSD race methods reporting gains.** Different objects: GLS/ERSD
   couple LISTS under k ≥ 2–3 arrivals with tree/batch verify; our Gumbel test was the k = 1 race under
   chain-only exact-match, where Lemma 3.1 predicts acceptance BELOW Σmin and the measured +1.5pt is the
   coupling artifact at sharp q. No contradiction; both point the same direction: race rules pay only with
   multiple arrivals, which pay only with rows, which are our expensive resource.
4. **Task framing "2-3 branches verify per row without serializing"** — they already DO verify without
   serializing: all T rows run in one batched forward; the serialization is in the PREFIX-CONDITIONING of row
   inputs (each row's attention/GDN input is its ancestor chain), not in the verify loop. No verdict-loop
   change removes that; only per-row state forks do (the real cost). The binding constraint is the row-cost
   law, not the linear scan at generate.cpp:12957 (which is O(T) integer compares — nanoseconds).

## 7. Open questions

- Does VAT's verification head transfer to a single-layer recursive MTP drafter (their evidence: EAGLE-3,
  DFlash)? The objective is architecture-agnostic; the falsifier is one G0.2 stair arm.
- The retrained drafter's calibrated q under the SERVE chain (DT 0.85, min-p 0.5): does Σmin(p,q) at DT 1.0
  exceed coupled acceptance at DT 0.85 on OUR distribution? One SPEC_PROB boot on the retrained head answers
  both #5 and the post-retrain DT question in one battery.
- dp_ok[0] seeding effect of the tip-k arm (5.3.iii): no instrument exists yet; STRATA_SPEC_DEPTH=1 reads it.
- Batch-lane lookup rows: does the suffix drafter hit at all under concurrent mixed traffic (its trigram table
  is per-request)? The policy telemetry answers in one boot.
- Erasure check on "P(pick ∈ top-2 of q | break)": needs a one-boot diagnostic (top2 is already tracked,
  mtp.hpp:73-75) if the tree question is ever re-opened by the kernel tranche.

## 8. Sources

Numbered; all arXiv IDs verified by fetch or search this session unless marked (memory-canonical).

**Code and campaign (primary, read this session):**
1. src/program/generate.cpp:14358-14508 — serial window build, offer gate, lookup composition, verdict, commit.
2. src/program/generate.cpp:12810-13026 — pipelined chain (A/B windows), t_rule, chain_launch/relaunch.
3. src/program/generate.cpp:10310-10458 — batch finish, keep[] prefix, rescue, 2-row slot shape.
4. src/core/mtp.cpp:1562-1834, 1915+ — draft() gated/ungated loops, min-p gate, draft_launch one-proposal split, chain_launch.
5. include/strata/core/spec_prob.hpp — rejection-sampling contract, counter spaces, point-mass rule.
6. include/strata/core/coupled_draft.hpp — coupled exact-match contract, counter arithmetic, output=seed law.
7. src/core/verify.cpp:2501-2565 — per-row sampling, spec_q hookup, spec_windows.
8. src/kernels/cuda/sampler.cu:296-445, 806-1016, 1361+ — sampler chain (KMAX 64), coupled pick + q emission, sample_tokens_spec.
9. include/strata/kernels/verify_kernels.hpp:21 — kVerifyMaxT = 8.
10. include/strata/spec/{draft_policy,suffix_drafter,draft_source,controller}.hpp — arbitration, lookup, source registry, the dead Controller.
11. SVG-CUSTOM.md F-115/F-116 (57% plateau + named levers), F-177 (Gumbel +1.5pt), F-201 (DT 0.85), F-228/F-229 (stair, ungated), F-231 (batch window law).
12. campaign/research/T14C-fastmtp-acceptance-math-gates.md (c_row, 0.72 rule, scenarios, gate ladder); T15A-m1-250-critical-path.md (post-diet c_row 2.2-2.6, W(1) 8-9 ms); T14A-fastmtp-method-recipe.md (recipe arms); T15C-overlooked-wins.md:793-795 (SPEC_PROB never swept).

**Papers — foundations and drafter training:**
13. Leviathan et al., Fast Inference from Transformers via Speculative Decoding — https://arxiv.org/abs/2211.17192 — the rejection-sampling rule spec_prob.hpp implements. VERDICT: APPLIES (in-tree, unmeasured).
14. Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling — https://arxiv.org/abs/2302.01318 (memory-canonical) — same rule, DeepMind form. APPLIES (as 13).
15. Gloeckle et al., Better & Faster LLMs via Multi-token Prediction — https://arxiv.org/abs/2404.19737 — parallel (non-recursive) MTP; the negative on frozen-trunk finetunes. DOESN'T (architecture is fixed; informs what not to retrain).
16. DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437 — the MTP architecture ours descends from; joint pretraining λ schedule. PARTIAL (we can only do the post-hoc head finetune half).
17. FastMTP — https://arxiv.org/abs/2509.18362 — recursive teacher-forced CE, self-distilled data, depth collapse is a training artifact. APPLIES (the queued base arm).
18. Red Hat Developers, Optimize vLLM speculative decoding with FastMTP heads (2026-09-08) — https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads — 5k samples/443 s; pos deltas 0.897→0.912 / 0.719→0.776 / 0.476→0.616 on our head class. APPLIES (calibrates data-scale expectations; warm-start recipe form).
19. GLM-5 — https://arxiv.org/abs/2602.15763 — shared-parameter multi-depth MTP, AL 2.76 vs 2.55; names the train/serve recursion discrepancy. PARTIAL (our recursive single layer is the memory-flat degenerate; training-side transfer only).
20. Verification-Aware Training for Speculative Decoding — https://arxiv.org/abs/2608.30135 — simulate the verify loop in training; verification head + rejection-anchored decay; +11.4% AL on EAGLE-3/DFlash; objective-only. APPLIES (the beyond-FastMTP rider; top-ranked).
21. LK Losses: Direct Acceptance Rate Optimization — https://arxiv.org/abs/2602.23881 (ICML 2026; HF: Nebius LK-Speculators) — optimize Σmin(p,q) directly; truncated-vocab-native; +3.5–8.2% τ. APPLIES (sub-vocab head match).
22. AdaSPEC — https://arxiv.org/abs/2510.19779 (NeurIPS 2025) — selective KD on the learnable token fraction; up to +15% acceptance. APPLIES (post-Arm-1 filter).
23. Online Speculative Decoding — https://arxiv.org/abs/2310.07177 (ICML 2024) — draft adaptation must track the served query distribution. APPLIES (our replay-corpus dump satisfies it by construction).
24. DistillSpec — https://arxiv.org/abs/2310.08461 (ICLR 2024) — divergence choice as a first-class recipe variable. PARTIAL (superseded for our case by 21's direct losses).
25. Temperature-Centric Investigation of Speculative Decoding — https://arxiv.org/abs/2410.10141 — KD temp 0.7 > 0.5; alignment law. APPLIES (AdaSPEC rider detail).

**Papers — verify rules and multi-draft theory:**
26. Sun et al., Block Verification Accelerates Speculative Decoding — https://arxiv.org/abs/2403.10444 (ICLR 2025) — joint block acceptance, provably greedy-safe; shipped as rejection_sample_method=block in vLLM. PARTIAL (touches the verdict contract; T15 priced redundant under exact-match chains — cite-only unless contracts open).
27. Weng et al., Traversal Verification for Speculative Tree Decoding — https://arxiv.org/abs/2505.12398 — leaf-to-root salvage of rejected-prefix subsequences. DOESN'T (needs a tree to salvage; chain has none).
28. Kobus et al., Speculative Sampling via Exponential Races (ERSD) — https://arxiv.org/abs/2504.15475 (ACL 2025 Findings) — coupled accept = D_HM ∈ [1−D_TV, D_HM] bounds; k-arrivals beat GSD at k ≥ 3 in batch. PARTIAL (the k=1 analysis explains our coupled acceptance; the k≥3 win needs rows we price dead).
29. Rowan et al., List-Level Distribution Coupling / Gumbel-max List Sampling — https://arxiv.org/abs/2506.05632 — one target pick endorses K drafts via shared exponentials. PARTIAL (same row-cost wall; interesting post-c_row-collapse).
30. Khisti et al., Multi-Draft Speculative Sampling: Canonical Architectures and Theoretical Limits — https://arxiv.org/abs/2410.18234 — K=2 closed form; perfect acceptance iff q(S) ≥ p(S)^K. APPLIES (offline headroom calculator for any multi-proposal arm).
31. Hu et al., Towards Optimal Multi-draft Speculative Decoding — https://arxiv.org/abs/2502.18779 (ICLR 2025) — sampling without replacement beats with; OT-dual optimal. APPLIES (governs how the tip-k arrivals in 5.3 should be drawn).
32. Thomas et al., Optimal Multi-Draft Speculative Sampling via Convex Optimization — https://arxiv.org/abs/2511.15898 — n-draft optima; below temp 0.8, k > 10 rarely helps. APPLIES (bounds tip-k at k ≈ 2–4).
33. Timor et al., Provable Acceleration for Multi-Draft Speculative Decoding — https://aclanthology.org/2024.emnlp-main.1148/ (EMNLP 2024) — k-draft lossless bounds. PARTIAL (theory bound; same row-cost wall).

**Papers — adaptive depth and serving:**
34. Huang et al., SpecDec++ — https://arxiv.org/abs/2405.19715 — threshold stopping rule; +7.2–11.1% vs fixed K. APPLIES (the threshold our flat gate already approximates; schedule justification post-retrain).
35. Agrawal et al., AdaEDL — https://arxiv.org/abs/2410.18351 — entropy-bound early draft stopping, +10–57% vs static. PARTIAL (the Controller family — our gate + policy already harvest most of it; measured 0–3% ceiling).
36. Mamou et al., DISCO — https://arxiv.org/abs/2405.04304 (ICML 2024) — oracle-trained lookahead classifier on draft hiddens, +10.3% over tuned static. PARTIAL (needs a trained classifier — can ride the retrain as a head).
37. Mamou et al., Self-Verification Length Policy — https://arxiv.org/abs/2411.18462 — difficulty-aware dynamic draft length. PARTIAL (as 36).
38. Bae et al., Optimizing Speculative Decoding for Serving LLMs Using Goodput (SmartSpec/TurboSpec) — https://arxiv.org/abs/2406.14066 — dynamic speculation length from batch goodput. APPLIES (conc lane; M1 flat).
39. Li et al., Nightjar — https://arxiv.org/abs/2512.22420 — bandit speculation length under load, +14.8%/−20.2%. APPLIES (conc rider only).
40. Liu et al., HeteroSpec — https://arxiv.org/abs/2505.13254 (ACL 2026) — entropy-binned per-row budgets. DOESN'T (tree-budget mechanism; chain budget = our gate).
41. Xia et al., Speculative Decoding survey — https://arxiv.org/abs/2502.19732 — landscape coverage. APPLIES (cross-check only).
42. Li et al., Batch Speculative Decoding Done Right (EqSpec) — https://arxiv.org/abs/2510.22876 — batched spec corruption class; vLLM/SGLang exact-match ≈ 0 under batching on Qwen3. APPLIES (vindicates our exact-match verify as the differentiator; already banked T9.B — cite-only).
43. Li et al., FLy: Training-Free Loosely Speculative Decoding — https://arxiv.org/abs/2511.22972 (ICLR 2026) — entropy-gated loose acceptance, 2.81×, ≥99% quality. DOESN'T (lossy; violates the exactness law; MMLU-gate risk without upside our gates would bank).

**Papers — drafter architectures and trees (context ceiling):**
44. Leviathan-lineage trees: EAGLE — https://arxiv.org/abs/2401.15077 (memory-canonical); EAGLE-2 — https://arxiv.org/abs/2406.16858 (context-aware dynamic trees, 3.05–4.26×); EAGLE-3 — https://arxiv.org/abs/2503.01840 (training-time test + multi-level features, τ 5.8–6.6). PARTIAL: training-time test = T14A Arm 4 (APPLIES as objective); tree verify = contract surgery (DOESN'T at standing c_row); feature fusion = engine taps (DOESN'T now).
45. Ankner et al., Hydra — https://arxiv.org/abs/2402.05109 — sequentially-dependent Medusa heads. DOESN'T (head family we don't have; dominated by MTP recursion).
46. Cai et al. lineage: Medusa — https://arxiv.org/abs/2401.10774 (memory-canonical) — depth-weighted CE origin. PARTIAL (objective history only).
47. Chen et al., Sequoia — https://arxiv.org/abs/2402.12374 (NeurIPS 2024) — wide-shallow dominates deep-narrow under power-law acceptance; hardware-aware DP. PARTIAL (the theorem's precondition — cheap rows — is exactly what we lack).
48. SpecExec — https://openreview.net/forum?id=JAhNsZ9dvG (NeurIPS 2024) — massively parallel tree verify for consumer GPUs with offload. DOESN'T (offload-regime trees; our verify rows are the scarce resource).
49. JetSpec — https://arxiv.org/abs/2606.18394 — causal parallel draft head over fused target hiddens; up to 9.64× (MATH-500, H100). DOESN'T (tree verify + new trained head + engine taps; ceiling calibration only).
50. DFlash — https://arxiv.org/abs/2602.06036 (ICML 2026; github.com/z-lab/dflash) — block-diffusion parallel drafter, one forward per block. PARTIAL (parked T1.7; its VAT pairing in 20 is the transferable half).
51. SpecBranch — https://arxiv.org/abs/2506.01979 — hybrid drafting + rollback-aware branch parallelism, training-free. DOESN'T (branches priced dead in Section 5.5; rollback machinery exists but the EV doesn't).
52. Continuous Speculative Decoding — https://arxiv.org/abs/2411.11925 (ECCV 2026) — alignment-by-autocurriculum + parallel sampling for VAR models. PARTIAL (domain mismatch; the alignment curriculum idea folds into Arm-1 data).
53. Speculative Speculative Decoding — https://arxiv.org/abs/2603.03251 — overlap draft with verify. DOESN'T (the demoted draft-verify overlap lane; draft = 6–9% here).

**Retrieval / n-gram hybrid drafting:**
54. SuffixDecoding — https://arxiv.org/abs/2411.04975 (NeurIPS 2025 spotlight; suffix-decoding.github.io; Snowflake Arctic Inference production) — suffix-tree drafts over prompt+history+prior outputs; up to 5.3× agentic. APPLIES (our suffix_drafter + draft_source registry is the same family; arming + a retrieval source are the gaps).
55. Luo et al., Token Recycling — https://arxiv.org/abs/2408.08696 — recycle REJECTED candidate tokens via n-gram pools. APPLIES (novel for us: our rejected drafts are discarded today; ~100-line addition to the suffix drafter).
56. SAM Decoding — https://arxiv.org/abs/2411.10666 (ACL 2025) — suffix AUTOMATON drafts (unbounded match length vs our WAYS=4 candidates). PARTIAL (suffix_drafter upgrade path if mimic hits the WAYS cap).
57. PLD (prompt lookup decoding) — https://github.com/cybershiptitook/PLD — the canonical lookup implementation; vLLM ngram defaults descend from it. APPLIES (already in-tree; cite-only).
58. Apple ML / NVIDIA, ReDrafter — https://arxiv.org/abs/2403.09919 + https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-supports-recurrent-drafting-for-optimizing-llm-inference — recurrent drafter with in-engine beam search + dynamic tree attention. DOESN'T (beam+tree in-engine; the beam-over-own-drafts idea is the one transferable kernel of it, priced as a drafter-step-cost trade in our economics).
59. NVIDIA TensorRT-LLM, N-gram speculative decoding (auto-enablement) — https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog7_NGram_performance_Analysis_And_Auto_Enablement.html — production n-gram with threshold override of neural drafts. APPLIES (the DraftPolicy's per-lane arming is our version; auto-enable by workload = the T8.C task-aware note).

**Engines and production docs:**
60. vLLM speculative decoding docs — https://docs.vllm.ai/en/latest/features/speculative_decoding/ — methods: eagle3, MTP, dflash, LiLiCorr, ngram, suffix, PARD; dynamic speculative decoding; adaptive verification; rejection_sample_method standard/synthetic/block. APPLIES (the method taxonomy confirms our in-tree coverage; "dynamic SD" = our #4).
61. vLLM speculators project — https://github.com/vllm-project/speculators + https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp — MTP finetuning standard (extract/finetune/stitch). APPLIES (W2 vehicle).
62. SGLang, Accelerating SGLang with MTP — https://www.lmsys.org/blog/2025-07-17-mtp — acceptance-length framing; batch-aware draft_token_num. APPLIES (conc rider #9).
63. SGLang speculative decoding docs — https://docs.sglang.ai/advanced_features/speculative_decoding.html — EAGLE-2/3, MTP, DFLASH, NGRAM; CUDA-graph constraints. APPLIES (cross-check).
64. SGLang issue #4360 (DeepSeek-R1 MTP poor performance at batch 16) — https://github.com/sgl-project/sglang/issues/4360 — verify-step cost under load. APPLIES (conc depth law).
65. TensorRT-LLM speculative decoding docs — https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html — draft/target + ReDrafter in-engine forms. APPLIES (cross-check).
66. Suffix Decoding project page — https://suffix-decoding.github.io/ — model-free suffix-tree spec for agentic workloads. APPLIES (as 54).

Count: 66 entries (12 code/campaign + 54 external), of which 44 are external distinct works/docs — well past
the 30-source bar. Memory-canonical items (14, 46, and the EAGLE-1 ID in 44) are marked and were not
independently re-fetched this session; all 2025-2026 arXiv IDs were verified by fetch or search.

## 9. Methods note (including unproductive searches)

Productive: full contract read of generate.cpp/mtp.cpp/verify.cpp/sampler.cu/spec_prob.hpp/coupled_draft.hpp/
draft_policy.hpp/suffix_drafter.hpp/draft_source.hpp/controller.hpp; the ERSD, VAT, JetSpec full/abstract
fetches; the vLLM spec-decode docs fetch; the Red Hat article fetch; 12 targeted searches confirming every
2024-2026 arXiv ID used in the table. Unproductive / dead ends (do not repeat): SGLang docs at
docs.sglang.io/backend/speculative_decoding.html 404s — the live page is
docs.sglang.ai/advanced_features/speculative_decoding.html (confirmed via search only, not fetched);
docs.vllm.ai/en/latest/features/spec_decode.html is a redirect stub; "SpecExec arXiv 2406.14066" is WRONG —
2406.14066 is SmartSpec/TurboSpec; SpecExec is cited by OpenReview/NeurIPS links; no draft-side "beam chain"
paper exists under that name (searched three phrasings — the closest published objects are EAGLE-2's
confidence-guided expansion and ReDrafter's in-engine beam, both tree-verify); no published training run of a
drafter against a quantized MoE backbone exists (T14A's F16 remains un-published territory); STRATA_SPEC_PROB
has no battery anywhere in campaign/results (confirmed by grep — only build warnings and the T15C env
inventory). Self-corrections logged: (a) initially framed the "coupling gain" as the Σp·q collision — corrected
after the ERSD read: our coupled exact-match acceptance is bounded by D_HM (the harmonic-mean distance), and
the F-201-brief's Σp·q statement and the D_HM bound agree in regime but are distinct objects; the dossier uses
the published bound. (b) nearly queued the depth-schedule standalone — re-priced to +0–1% after computing the
SpecDec++ threshold (0.42) against our flat 0.5; the schedule is a post-retrain lever, not a standalone one.
