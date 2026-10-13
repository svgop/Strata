
### F-177 — R5 GUMBEL COUPLING TESTED TO THE FULL FACTORIAL: ~+1.5 acceptance points, not 9 — REVERTED, and the rescue quantified (2026-10-14 ~00:45, goal round 5)

**THE HYPOTHESIS (F-176 session's queue head).** `STRATA_SPEC_GUMBEL=1` replaces the inverse-CDF pick
on both the draft and the verify side with a Gumbel-max pick whose noise is keyed on
(seed, counter, TOKEN ID) — "support invariant", so list-composition differences between the drafter's
and the target's truncated candidate lists stop costing agreement (`sampler.cu:66-80`,
`coupled_draft.hpp:14-17`). Upstream (llama.cpp-lab PRs #26/#30) measured +9% acceptance, lossless.
Pre-registered adoption bar: M1 ≥ +4% over same-night baseline, no mean regression beyond ±4%,
MMLU ≥ 93.3.

**PLUMBING VERIFIED LIVE FIRST** (a mid-experiment scare, resolved): the flag is consumed at
`sampler.cu:1242` (target side, `sample_tokens`) and `:812` (draft side, `coupled_stage_kernel`), and
the pick itself runs in `sampled_tail_warp` (`:534-540`), the shared tail of the DEFAULT split path
(`:795`), the one-block fallback (`:631`), and the coupled drafter's pick (`:871`, with `sub_to_id`
mapping and the verifying row's counter). Not confined to the Old path. The A/B tested the real thing.

**LEG A (standing recipe, rescue ON): 6v6, back-to-back same-night boots.**

| arm | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| baseline (6) | 168.7 | 145.6 | 191.6 | 79.1% |
| +GUMBEL (6) | 171.8 (+1.9%) | 145.4 (−0.2%) | 187.2 (−2.3%) | 81.0% |

Below the bar on every count; M3b ticks +1.9 pts with consistent direction but no t/s conversion.

**LEG B (rescue OFF — the masking test): 3v3.** If the typical-acceptance rescue were absorbing
Gumbel's recoverable mass, Gumbel would fire on the raw path.

| arm | M1 | pair | 6-conc | M3b (raw exact-match) |
|---|---|---|---|---|
| no rescue (3) | 155.9 | 146.3 | 190.7 | **58.5%** |
| no rescue + GUMBEL (3) | 157.0 (+0.7%) | 142.8 (−2.4%) | 186.0 (−2.5%) | **59.9% (+1.4 pts)** |

**THE VERDICT: the mechanism fires but the mass is not there.** Gumbel adds ~+1.5-2 acceptance
points in BOTH regimes — the rescue does not mask it. The upstream +9% came from an unfiltered
regime; our `--spec-min-p 0.5` means the drafter only offers rows it is confident about, and on
confident rows the draft's top token already equals the target's sample — the coupling artifact
Gumbel removes is second-order on the offered set. The 58.5% raw agreement is genuine distributional
disagreement between the rt draft head and the target at depth 6, which no sampling mechanics can
recover. Reverted; standing recipe restored and verified (env + boot log).

**THE DISCOVERY THE EXPERIMENT BOUGHT: the rescue, quantified for the first time on this lane.**
`STRATA_TYPICAL_ACCEPT=0.2`: 58.5% → 79.1% acceptance (+20.6 pts) and 155.9 → 168.7 M1 (+8.2%),
pair/6-conc unchanged within noise. It is the single largest acceptance mechanism in the standing
recipe — larger than everything else measured this campaign. Windows measured 0.70 → 0.92 on their
spec-4 config (their journal's patch map); this lane's spec-6 numbers are now on record.

**WHAT THIS OPENS (the re-ranked acceptance queue).** (1) **R5b: tau sweep** — the rescue threshold
0.2 is a single unswept number sitting on the largest measured lever; 0.25/0.30 could buy points, but
looser tau emits draft tokens further from the target's pick → **MMLU gate mandatory** (the Windows
F-15 rescue bug — text/context divergence per rescue — is the cautionary tale; tau loosens the same
criterion). (2) The draft head itself is the remaining structural lever (the 57% r17 plateau says
distillation is not; an F16/quant-chain upgrade per their F-116 next-levers note is the path if ever
revisited). (3) Depth stays the wrong direction to trade (M1 loses more than acceptance gains).

Sources: `campaign/results/r5-base{1..6}.txt`, `r5-gum{1..6}.txt`, `r5-nr{1..3}.txt`,
`r5-nrg{1..3}.txt` (18 probe files, all same-night, back-to-back boots, placement off = standing).
No MMLU gate run: nothing adopted.
