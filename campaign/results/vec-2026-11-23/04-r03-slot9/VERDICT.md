# Slot-9 falsifier — base-context dose ladder verdict

Vector: `/mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-persona-r03-r03s9-s9-rho1.gguf`
(one row direction.8, F32, norm 1.0276 = ρ×mean‖h₉‖, refusal-orthogonalized, built from the r03
capture). Stock template, mechanism isolation. 36 AB pairs per dose, blind + swap-balanced.

| dose | verdict | assertive | serious | direct | facts off/on | words | judgeA |
|---|---|---|---|---|---|---|---|
| 0.03 | MIXED | 3/2 p=0.50 | 5/3 p=0.36 | 6/4 p=0.38 | 1.00/1.00 | 167→161 | 1.0 |
| 0.1 | MIXED | 3/4 p=0.77 | 7/2 p=0.090 | 7/5 p=0.39 | 1.00/1.00 | 168→166 | 1.0 |
| **0.3** | **WINDOW** | **10/3 p=0.046** | **16/0 p=2e-05** | 8/5 p=0.29 | **1.00/1.00** | 166→145 | 1.0 |
| 0.6 | WINDOW (knee) | 11/2 p=0.011 | 17/0 p=1e-05 | 6/6 p=0.61 | 1.00/**0.94** | 167→156 | 1.0 |

## Reading

- **The window exists.** One clean slot (direction.8) at read-out-matched dose moves the register
  massively (serious 16–17/18 pairs) with facts fully intact at 0.3. r02's damage-first dose
  response was its construction (9 diffuse slots, dose axis uncorrected for the 4× read-out), not
  this model's nature.
- **The knee is priced**: 0.3 = clean window; 0.6 = first factual casualty (1.00→0.94) while the
  trait still gains. The survivable bracket is [~0.1 suggestive, 0.3 clean, 0.6 first blood].
- **Length confound CLOSED** (all-orders judged pairs; t0-only counts above are the stricter read):

  | 0.3 cutoff | n | assertive | serious | direct | words |
  |---|---|---|---|---|---|
  | all | 35 | 19/7 p=0.014 | **34/0** | 18/9 p=0.061 | 184→162 |
  | \|ΔW\|≤40 | 28 | 16/6 p=0.026 | **27/0** | 13/8 p=0.19 | 194→185 |
  | \|ΔW\|≤25 | 25 | 15/5 p=0.021 | **24/0** | 13/5 p=0.048 | 191→186 |
  | \|ΔW\|≤12 | 14 | 6/5 p=0.50 (power) | **14/0** | 6/5 p=0.50 | 182→180 |

  Serious loses zero judged pairs at every length cutoff — the register move is content-deep,
  not verbosity cosplay. Assertive is real but softens where power collapses (n=14). Direct is
  unresolved (highest judge-inconsistency: 8–9x per dose). (0.6 mirrors: serious 35/0 all,
  11/0 at ≤12; assertive 23/3 all.)
- **Judge caveat**: judgeA = 1.0 — the stock-context judge has the banked first-position bias
  (see `../00-r02-verdicts.md`); criterion counts are swap-balanced but the deployed test carries
  the decision (judgeA 0.79 there).
- Router channel: clean at 0.3 (facts 1.00) despite 2.21× visibility — the router-visible component
  did not corrupt routing at this dose. Its contribution to the 0.6 casualty is unpriced; the P12
  projected arm at 0.6 is the contained experiment (built, armed only if run).

## Verdict rule of record

The deployed-template ON/OFF test (`../05-r03-deployed/`, dose 0.3) decides whether this ships.
Stock numbers are mechanism evidence only.
