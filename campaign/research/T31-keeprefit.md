# T31 — the draft_cum keep-rule refit on the banked spec trace (roadmap item 3, F-260)

Date: 2026-10-22. Zero-boot offline analysis; no engine, server, config, build, or commit was
touched. Inputs: `campaign/results/spec-trace.txt` (1,615 rows), the prior 16-line
`campaign/results/spec-trace-analysis.txt`, the instrument source
(`src/program/generate.cpp:15416-15429`), and the cost/calibration laws T14C + T26-2 + F-228.
Simulator: `campaign/t31-keeprefit.py` (re-runnable, stdlib + numpy from `.venv`).

## Summary

1. **Parse is clean and the trace decodes fully**: 1,615 windows, 0 malformed, d-row count
   always `min(T-1, 8)`, no printed d below the standing 0.5 gate. The file is NOT one solo
   boot: it is **~15 appended boot segments** (the cumulative `resc` counter resets 14 times;
   segment sizes 33-441 windows). T/a marginals reproduce the prior analysis exactly.
2. **Rescue accounting corrected**: 466 rescue events in **391/1,615 windows (24.2%)**, 75
   windows with 2 rescues. The `resc` field is a per-process CUMULATIVE counter (field>0 on
   94.1% of rows — meaningless per window). The prior "387 rescued-deltas" and F-260's "24%
   advance via rescue" are approximately right in magnitude (window count), but the analysis
   line conflates events (466) with windows (391).
3. **The prior decile table's "tested n=3593" is wrong**: 3,593 is the total OFFERED rows.
   Actually tested (evaluated by the exact-match walk): 3,181 rescue-inclusive, 1,955 in
   rescue-free windows. All printed rows carry d >= 0.5; rows after the stopping rejection
   were never evaluated and deflate any pooled rate computed over them.
4. **Calibration (exact-match, tested rows)**: acceptance rises with d at every position
   (corr 0.38-0.48) and falls with depth at fixed d. Port-lattice refit values (T26-2's
   edges, our values): [0.5,0.7)->0.57, [0.7,0.8)->0.63, [0.8,0.9)->0.70, [0.9,0.95)->0.76,
   [0.95,0.99)->0.77, >=0.99->0.93-0.99 (pooled; per-position table below). The <0.5 bins are
   UNFILLABLE from this trace — the flat gate censored them. Position-4 is conditionally the
   *second-best* position (0.866) — pure survivorship — while pos-3 is the weakest (0.718).
5. **NOW verdict — F-260's "post-retrain only" framing is REFUTED model-side, with a catch.**
   The calibrated cum rule at th*=0.50 predicts **+4.9% net** at central c_row=2.4 ms
   (break-even 0.51 tokens/row vs cut-mass carry 0.334), robust across all 15 segments
   (+3.4..+6.0%), across c_row 2.2-2.6 (+3.7..+6.2%), and across the window bootstrap
   (p5..p95 = +4.0..+6.3%, P(net>0)=1.00). Raw-d products remain inert (+1.0% at th 0.5 —
   F-260 confirmed and now explained). **The catch**: the entire edge is the deep-position
   acceptance deficit — strip the position effect (pooled bins only) and the same rule reads
   **+0.05%**. The lever monetizes F-235, nothing else. Given F-183's probe read the
   neighboring min-p-0.7 arm at −3.0% (against this model's +3.1% for the same direction),
   the model-side gain is UNPROVEN, not banked: the refit's output is a battery arm, not an
   adoption. It is a ~50-line, default-off, offers-only-exact port (zero quality risk).
6. **Retrain go/no-go — INVERTED**: no shift is needed for the rule to pay (delta*=0.00 on
   the additive-shift model). The rule's incremental value DECAYS with drafter quality:
   below +2% at shift delta ≈ +0.05-0.06, exactly 0 (degenerates to the gate) by delta ≈
   +0.30. T14B's anti-drift rider targets the very deficit the rule feeds on. Therefore:
   **battery the keep rule BEFORE/alongside T14B; do not gate it on the retrain, and expect
   it to retire after a successful one.**

## 1. Parse hygiene

| item | value |
|---|---|
| lines parsed / malformed | 1,615 / 0 |
| d-row count vs min(T-1,8) | 0 mismatches |
| printed d < 0.5 (standing gate) | 0 (gate `--spec-min-p 0.5` confirmed; bins <0.5 censored) |
| T dist | {1:374, 2:372, 3:219, 4:184, 5:99, 6:367} — mean T 3.2248, mean offered 2.2248 |
| a dist | {0:477, 1:418, 2:253, 3:187, 4:84, 5:196} — mean a 1.7344, mean C=a+1 2.7344 |
| a > drafts-offered (impossible) | 0 |
| rescue segments (counter resets) | 15 segments (14 resets; append-mode file, one counter per boot) |
| rescue events / windows / 2-rescue windows | 466 / 391 (24.2%) / 75 (max 2/window, the code's cap) |
| windows with resc field > 0 | 1,520 (94.1%) — cumulative counter, not per-window state |
| mean a: all / resc-free / exact (a−resc) | 1.7344 / 1.3521 / 1.4458 → mean C exact **2.4458** (≈ T14C's standing 2.47; the 2.734 headline was rescue-inflated) |
| accept-all (a=T-1, T>=2) | 861/1,241 drafting windows = **69.4%** (53.3% of all — the prior "53%" included the 374 zero-draft windows) |
| cap-hit (T=6 & a=5) | 196 = **53.4% of T=6 windows** (the "12%" was of all windows) |
| d >= 0.9 share of printed rows | 69.6% (F-260's "most d >= 0.9" confirmed) |
| per-segment homogeneity | acc(offered) 0.67-0.85, mean a 1.21-2.50 — real traffic spread; the verdict is re-checked per segment below |

Position count correction: at `--spec 6` a window carries at most **5 drafts, positions 0..4**
(the brief's "0..5" counted the seed row; `generate.cpp:15425` prints `i+1 < T` rows).

## 2. Per-position acceptance (tested rows only; exact-match = rescue-free windows)

Lattice A = T26-2's fixed port edges (keep the lattice, refit the values). Cells with n<8
marked `~`; the [0.30,0.50) column is empty by construction (gate censoring).

| pos | [0.50,0.70) | [0.70,0.80) | [0.80,0.90) | [0.90,0.95) | [0.95,0.99) | >=0.99 | ALL |
|---|---|---|---|---|---|---|---|
| 0 | 48/76=0.632 | 33/51=0.647 | 69/89=0.775 | 45/60=0.750 | 117/132=0.886 | 435/442=0.984 | 747/850=0.879 |
| 1 | 20/51=0.392 | 17/26=0.654 | 29/39=0.744 | 18/29=0.621 | 66/73=0.904 | 262/274=0.956 | 412/492=0.837 |
| 2 | 14/28=0.500 | 15/25=0.600 | 22/26=0.846 | 12/17=0.706 | 18/26=0.692 | 172/177=0.972 | 253/299=0.846 |
| 3 | 9/18=0.500 | 7/14=0.500 | 12/25=0.480 | 7/16=0.438 | 12/17=0.706 | 93/105=0.886 | 140/195=0.718 |
| 4 | ~2/3 | 6/10=0.600 | ~4/7 | 8/10=0.800 | 14/19=0.737 | 69/70=0.986 | 103/119=0.866 |

- **Monotonicity in d HOLDS** at every position (corr(d-bin, accept) = +0.38..+0.48; the
  pooled A-lattice values above: 0.57 → 0.63 → 0.70 → 0.76 → 0.77 → 0.93-0.99). Individual
  thin cells wobble (pos-3 [0.90,0.95) 0.438 < pos-3 [0.5,0.7) 0.500) — noise at n=14-18.
- **Depth structure**: at fixed d acceptance falls with depth (pos-1 < pos-0 at every bin,
  e.g. [0.5,0.7): 0.392 vs 0.632; pos-3 weakest overall at 0.718). Pos-4's high conditional
  rate (0.866) is survivorship — only healthy windows reach it.
- F-261's per-position numbers (91.7/82.9/71.8/60.1/53.4) are the UNCONDITIONAL
  accepted/offered rates — reproduced exactly here (1138/1241, 720/869, 467/650, 280/466,
  196/367). They are the wrong conditioner for a sequential rule: the rule consumes the
  CONDITIONAL marginals above (a row is only offered-and-tested if the prefix held).
- The prior decile table pooled all 3,593 PRINTED rows ("tested") — deflating rates by
  counting offered-but-never-evaluated rows as rejections.

## 3. Rescue-inclusive accounting (what the rule actually controls)

| quantity | value |
|---|---|
| acceptance, offered denominator (all rows) | 0.7796 |
| acceptance, tested denominator, rescue-inclusive `a` (what `DraftCumAdapt::observe` would eat) | 0.8805 |
| acceptance, tested denominator, exact-match (what the keep rule controls with rescue off) | **0.8465** |
| rescue-advanced windows / events | 391 (24.2%) / 466 |
| rescue-inflation of mean C | 2.7344 → 2.4458 exact (−0.289 tok/w, −10.6%) |
| mean d/window, resc vs non-resc | 0.873 vs 0.917 (rescued windows run lower-d chains, as expected) |

With `STRATA_BATCH_RESCUE=0` standing, the exact-match column is the operative one; the
rescue-inclusive number is the observe()-semantics audit (T26-2 §c.4). The banked "74.9%"
acceptance matches neither — it was a different era's offered-denominator figure (T14C).

## 4. Simulated keep rules

Rule emulated (T26-2 §c.1/c.2, offers-only-exact): draft j offered iff
`c_j = prod_{i<=j} acc_hat(d_i, i) >= th` (f32, ascending); `T = 1 + leading drafts passing`,
**capped at the observed profile** — this trace is right-censored at the flat gate, so every
simulated rule can only CUT (extension into d<0.5 has no calibration data; priced separately
below, clearly labeled). Deltas are model-internal (same acc_hat both sides), so m̂ level
bias cancels to first order; a Monte-Carlo walk (2,000 reps x affected windows) confirms the
plug-in arithmetic (dA −0.2007 vs −0.2020).

Cost model (T14C §1.3, T26-2 amendment): `W = W0 − c_row·|dT|`, **W0 = 11.51 ms**
(C0_exact 2.4458 at the banked 212.5 tok/s = 4.71 ms/token; T14C's standing band), c_row
2.2-2.6 ms post-POOL_DIET (3.56 pre-diet). Crossing rule: a marginal row pays iff its
conditional acceptance > `c_row·C0/W0` = 0.467/0.510/0.552/0.677 (c = 2.2/2.4/2.6/3.56).

**th* = 0.50 flat** wins at central c_row (+4.92%); the plateau th 0.40-0.60 all clear +4%;
depth-decay vectors (e.g. 0.40+0.10k) land +4.6-5.1% — no shape bonus worth naming.

| rule family | th=0.50 net% (c=2.2 / 2.4 / 2.6 / 3.56) |
|---|---|
| cum-CALIBRATED flat | +3.68 / **+4.92** / +6.18 / +12.71 |
| cum-calibrated depth-decay 0.40+0.10k | +3.54 / +4.86 / +6.22 / +13.26 |
| cum-RAW-d product (prior analysis's rule) | +0.65 / +1.01 / +1.37 / +3.14 — **inert, F-260 confirmed** |
| flat-gate retune minp 0.7 (no cum) | +2.07 / +3.11 / +4.16 / +9.54 |

The raw-d sweep reproduces the prior analysis byte-for-byte (th 0.30 keeps 2.194, th 0.50
keeps 2.028 — their lines 15-16), cross-validating the parser. Raw products barely cut
because 69.6% of printed d >= 0.9; the calibrated bins cut 0.60 rows/window.

**Where the edge lives (th=0.50 cut decomposition, exact-match world):** cut tails carry
0.2007 tokens/window over 0.60 cut rows = 0.334 tokens/row vs break-even 0.510. Cut mass by
first-failure d-bin: [0.5,0.67) 0.278 rows/w (cell n=152), [0.70,0.81) 0.082 (n=136),
[0.90,0.96) 0.098 (n=167), [0.96,0.99) 0.037 (n=192), >=0.99 0.016+0.015 (n=193+891). The
biggest single lever inside it: windows whose products die on a mid-d row early.

**Sensitivity that decides the verdict:**

| variation | th=0.50 net% (central c_row) |
|---|---|
| shrunk per-(position x bin) calibration (primary) | **+4.92** |
| bootstrap over windows (200 resamples, same calibration) | p5 +3.99 / p50 +5.07 / p95 +6.27, P(net>0)=1.00 |
| POOLED bins only — position effect removed | **+0.05** |
| per-segment (15 boots, primary calibration) | +3.4 .. +6.0 — positive in every segment |
| c_row 2.2 → 2.6 | +3.68 → +6.18; flip-to-negative needs c_row < ~1.7 ms (all measured anchors 2.2-4.6 sit above) |
| W0 11.0-12.9 ms | verdict unchanged (break-even 0.53-0.46 vs carry 0.33) |

The window bootstrap only captures sampling noise. The pooled-bin bound shows the ENTIRE
edge is the position effect — the deep-position acceptance deficit (F-235). That effect is
real in the marginals (pos-3 exact 0.718, Wilson [0.651,0.776], n=195) but the per-(pos x
bin) cells are thin, and the one measured A/B of the nearest real intervention (F-183's
calibrate probe: min-p 0.5 134.9 vs 0.7 130.8 = −3.0%) points AGAINST this model's +3.1%
for min-p-0.7 — on a probe-grade read whose own ranges overlap and which F-263's variance
law discredits. F-173 round 2 (min-p 0.7→0.5 at spec 4 = +7.9% for loosening) is likewise
era-mismatched (spec 4, old binary, ~1.3 offered rows/window, different head).

## 5. Verdict NOW

**Model-side: a calibrated keep rule PAYS at spec-6 economics with the current drafter —
F-260's "post-retrain only" expectation is refuted with numbers.** The raw-threshold
inertness F-260 recorded is real but was the wrong rule: raw products cannot see the
calibrated deficit. Central estimate **+4.9% M1** (band +3.7..+6.2 over c_row; +0.05% if the
position effect is an artifact). **Operational verdict: WORTH A QUIET-WINDOW FIVE-BOOT
BATTERY, NOT AN ADOPTION.** Per F-263's law no 3-5% model prediction ships on less; per
F-183 the one probe-grade A/B in this family read the opposite sign; the port is ~50 lines,
default-off, and offers-only-exact (outputs p-exact by the coupled contract — zero quality
gate owed). Suggested battery arms: cum-cal th {0.40, 0.50} + minp-0.7 + ctl, five-boot
quiet window, all three lanes (conc unmodeled here: batch economics price the widest window,
F-262 — a shorter AVERAGE width should help conc, but that is a hypothesis, not a number).

## 6. Post-retrain projection (shift model, explicit assumptions)

Shift: `d → min(1, d+delta)` on every draft row (the retrained drafter is more confident AND
better calibrated); acceptance re-read from the SAME calibration map at the shifted d
(stability-of-the-map assumption — T14C's G0/G1 gates exist precisely to test it); dark
T=1 windows (23.2%, d unobserved) get one imputed draft from uniform[0.2,0.5); rows beyond
the observed profile stay right-censored — **extension upside is NOT priced** (it belongs to
the drafter, not the keep rule). Baseline for "the lever's own value" = the flat-0.5 gate
evaluated at the SAME shift.

| delta | rule net vs TODAY (shift+rule) | rule net vs gate-at-same-shift |
|---|---|---|
| 0.00 | +5.36 | **+5.36** (th 0.50) |
| 0.04 | +2.69 | +2.61 |
| 0.06 | +2.05 | +1.94 |
| 0.10 | +1.21 | +0.88 |
| 0.20 | +1.40 | +0.28 |
| 0.30 | +3.10 | 0.00 (rule degenerates to the gate) |

- **delta* (rule clears +2% vs the same-shift gate) = 0.00 — the lever does not wait for the
  retrain.** The go/no-go signal inverts: the rule STOPS clearing +2% at additive shift
  ≈ +0.05-0.06 (low-d acceptance ≈ +2pp, products start clearing th) and is exactly neutral
  by +0.30.
- The +0.36% gap between the two delta-0.00 entries (+5.36 vs +4.92) is the dark-window
  extension: the rule re-opens imputed d0<0.5 drafts whose acceptance is a below-gate
  extrapolation with ZERO data behind it. Unproven; the cut-only +4.92% is the defensible
  headline.
- Sequencing consequence: T14B's EAGLE-3.1 anti-drift rider attacks the deep-position
  deficit — the very thing the rule monetizes. A successful retrain retires this lever.
  Battery it now (it rides any boot) rather than gating it on T14B; do not expect it to
  stack post-retrain.

## Assumptions (every one stated)

A1 acceptance is sequential and the calibrated marginals are CONDITIONAL on an unbroken
prefix (tested rows only) — the exact-match walk's structure; the product formula needs no
independence beyond these conditionals, and the MC validates the arithmetic.
A2 acc_hat(d, k) is the true conditional acceptance and the offer trim changes nothing else
(offers-only-exact bit-contract, T26-2 §c.3 — trimming a chain never alters another draft's
bits). The main exposure: per-(pos x bin) cells are thin; the pooled-bin bound (+0.05%) is
the honest floor of that exposure.
A3 window cost linear in rows: W = W0 − c_row·|dT|; c_row measured 2.2-2.6 ms post-diet
(T14C/T26-2), F-228's own A/B closed arithmetically on this form. The drafter's own chain
time is ignored (overlapped/second-order; F-229 priced the step tail at ~0.05 ms GPU/step).
A4 the trace's mixed 15-boot traffic represents the battery's probe mix — checked per
segment (verdict positive in all 15).
A5 rescue-off world (standing law); exact-match calibration; rescue-inclusive accounting
reported separately. Dark windows are censored, not modeled, except where labeled.

## Contradictions and their resolution

1. **F-260 "raw thresholds inert / refit expected post-retrain" vs this refit's "pays now".**
   Resolution: F-260's inertness was measured on RAW-d products (reproduced here: +1.0%) and
   its refit prescription (calibrated bins) had not been run. The calibrated rule's edge is
   real model-side but rests entirely on the position effect — carried as an unproven
   battery arm, not an adoption. F-260 stands amended: the lever is viable-now-but-unproven,
   not post-retrain-gated.
2. **F-183's calibrate probe (min-p 0.7 = −3.0% vs 0.5) vs this model (+3.1% for min-p-0.7).**
   Unresolved. The probe is a single-window decode-rate median with overlapping ranges
   (F-263's variance law applies to it directly). Carried as the strongest evidence against
   the model-side gain; the battery resolves it.
3. **Prior analysis "tested n=3593 / rescued-deltas=387 / 53% accept-all / 12% cap-hit" vs
   this refit.** All four were denominator or field-confusion errors: tested = 3,181/1,955
   (offered = 3,593); rescue = 391 windows / 466 events (the 387 was approximately the
   window count); accept-all = 69.4% of drafting windows; cap-hit = 53.4% of T=6 windows.
   The T/a marginals and the raw-th kept-means reproduce exactly — the errors were all in
   the derived tables, not the parse.
4. **F-261's per-position "acceptance" vs the rule's needs.** F-261 printed unconditional
   rates (correct for its depth-7 case); the keep rule consumes conditional marginals. Both
   are reported; no contradiction, a conditioner distinction the port must respect
   (`observe()` feeds windows, the bins must be fit on tested rows).

## Open questions

- The [0.30,0.50) bins are unfillable from this trace (gate-censored). If the port ships,
  boot once with min-p ~0.3 (F-229's ungated pattern) to fill them, or accept the model's
  extrapolation only for the dark-extension rider (not priced here).
- Pair/conc lanes are unmodeled. The batch lane pays the widest window (F-262); a shorter
  average width is plausibly conc-positive but needs the battery's conc legs.
- Does basalt's per-request EMA (th 0.35→0.25·scale) track better than the fixed th* found
  here? The grid held th fixed; the EMA can only be replayed per-request, which this
  per-window trace cannot do (no request ids). Battery the fixed th* first.
- Is the position effect stable across workload classes? Segment-level says yes across
  boots; a per-prompt-class split needs request ids the trace does not carry.

## Implications

- Roadmap item 3 is COMPLETE and its verdict is stronger than advertised: the keep rule is
  the only spec-lane lever that is (a) zero-quality-risk (offers-only-exact), (b) model-side
  positive NOW (+4.9% central), and (c) sequenced to DIE at T14B — so it slots immediately
  before the retrain, riding the same battery boots as the window tranche.
- The refit's bin table (lattice A above) is the deliverable the port needs; th* = 0.50
  (scale 1.0), flat, no depth decay, EMA optional.
- The deep-position deficit (F-235) is now PRICED: it is worth up to ~5% M1 via avoidance
  (this rule) and is the same surface T14B's anti-drift rider targets. The two levers
  overlap; do not double-count them in the roadmap's composed bands.
- The spec-trace instrument should log the per-window resc DELTA (or per-request ids) next
  boot — two of the four prior-analysis corrections trace to that gap.

## Sources

1. `src/program/generate.cpp:15416-15429` — STRATA_SPEC_TRACE instrument (row format, resc
   semantics, tested-row counters); :15160-15164 the standing flat gate; :15384-15413 the
   exact-match walk + rescue (a is rescue-inclusive, resc cumulative, 2/window cap);
   :15499-15502 drafter chain early-stop.
2. `custom-iq3_s-8081.json` — standing `--spec 6 --spec-min-p 0.5` (gate value pinned).
3. `include/strata/core/mtp.hpp:155-163` — chain early-stop at spec_min_p ("a draft below
   spec_min_p ends it").
4. `campaign/results/spec-trace-analysis.txt` — the prior 16-line analysis (reproduced and
   corrected; kept-mean lines cross-validated exactly).
5. `campaign/results/spec-trace.txt` — the 1,615-window dataset (primary data).
6. `campaign/research/T26-2-spec-lane.md` §a.1/§c.1-c.4 — basalt draft_cum form (8 bins,
   prod(acc)>=th, EMA 0.97, placement scale), the port spec, the refit contract, rescue
   observe() decision, c_row 2.2-2.6 post-diet, DeepInfer V(T)=9.7+2.47(T−1) flat-c_row note.
7. `campaign/research/T14C-fastmtp-acceptance-math-gates.md` §1 — c_row 3.56 M1 / 4.62 conc
   measured, crossing rule (marginal > c_row·C0/W0), banked baselines table, s35 arithmetic.
8. `SVG-CUSTOM.md` F-260 (F-264 block) — the 477 us/window wait-split anchor; F-260's trace
   first-read and raw-th sweep; F-261 per-position reads; F-262/F-263 the conc law and the
   quiet-window five-boot adoption law; F-264 the wait-split; F-265 the pruning ledger.
9. `SVG-CUSTOM.md` F-228 — the measured depth stair, min-p-0.35 negative (−1.75% 3v3),
   "marginal row costs the full graph's T-scaling".
10. `SVG-CUSTOM.md` F-173/F-174 — min-p 0.7→0.5 +7.9% at spec 4 (era-mismatched context),
    standing recipe pin (BATCH_RESCUE=0).
11. `SVG-CUSTOM.md` F-183 — the calibrate run: min-p 0.5 134.9 vs 0.7 130.8 (probe-grade
    counter-evidence).
12. `SVG-CUSTOM-TODO.md` — roadmap item 3 wording, F-265 prune, dead-end table (draft_cum
    row: "keep iff prod(acc)>=th, base 0.35, EMA(0.97); observe() MUST be rescue-inclusive").
13. `campaign/research/T25-2-spec-lane.md` — the +0-1% standalone / +1-3% post-retrain
    estimate this refit replaces with measured-traffic arithmetic.
14. `campaign/t31-keeprefit.py` — this refit's simulator (argparse, `--input`, numpy f32
    emulation, MC + bootstrap built in).

## Methods note (what did not work / what to avoid)

- The additive-logistic calibration (logit ~ d + hinge(0.9) + position) MISFITS the thin
  low-d cells by up to ±20pp (predicted 0.61 where raw pos-0 [0.5,0.67) reads 0.40) and
  inflated the headline by ~6pp. Replaced by shrunk per-(position x bin) empirical rates
  (k=20 toward the pooled bin rate). Do not refit a smooth model on cells this thin without
  a pooled fallback.
- First simulator draft had `E[accepted] = Σ P(survive-to-j)·m_j` (the stopping probability,
  not the count) — it made cutting look free and produced an absurd +60%. The MC cross-check
  caught it (plug-in −0.265 vs MC −0.910). Keep the MC in the loop for any counterfactual
  acceptance math.
- `binof` initially sent d=1.0 rows into a phantom bin past the last edge (891 rows silently
  dropped from the printed tables, then an IndexError). Right-closed bin contract fixed it.
- Counting resc windows via the FIELD (>0 = 94% of rows) or via the final counter (38) are
  both wrong; only segment-aware deltas give the 391/466. Any future trace should log the
  delta directly.
- Unproductive searches: `grep draft_cum src/` (the rule exists only as the TODO/T26-2 port
  spec — our tree's gate is the flat min-p branch, T26-2 line 57); V(T) lookup in
  T23-deepinfer.md (the measured form lives in T26-2 §(d)#5, not the T23 file).
