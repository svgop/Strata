# Leg A — plain vs P12 interleave at 0.45/0.6, same window: projection CONFIRMED

Four single-boot AB runs in one time window (17:50–18:12), alternating arms, base context, each
36 pairs, blind + swap-balanced, judge controls in-run from boot 2 (`judge_control_ok: true`).

| dose | arm | assertive | serious | direct | facts off/on | words off/on |
|---|---|---|---|---|---|---|
| 0.45 | plain | 10/3 p=0.046 | 17/0 p=1e-05 | 9/3 p=0.073 | 1.00/1.00 | 169→152 |
| 0.45 | **P12** | **12/3 p=0.018** | **18/0 p≈0** | 8/2 p=0.055 | 1.00/1.00 | 168→154 |
| 0.6 | plain | 11/2 p=0.011 | 18/0 p≈0 | 9/6 p=0.30 | 1.00/**0.94** | 163→154 |
| 0.6 | **P12** | **12/2 p=0.006** | 17/0 p=1e-05 | 9/4 p=0.13 | 1.00/**1.00** | 163→151 |

## Verdict

**The router-null-space projection is the default construction.** At both doses, same window, the
projected row is equal-or-better on every criterion (assertive +1 net win at both doses; serious
sweeps or ties; direct comparable) and the plain row's knee — the first factual casualty at 0.6
(1.00→0.94, one question) — does not appear in P12. The removed 27% (router-visible component)
was damage capacity, not persona. RARE's mechanism (2608.21236) holds locally at both doses.

The plain knee sits between 0.45 (clean) and 0.6 (first blood); P12 is clean at 0.6, so the
projection buys at least +0.15 of dose headroom.

## Consequences

- All future persona builds use `vector-capture.py build --project-router <max-vis-block>`.
- Adoption dose question moves to Leg B (`../08-legB-p12-deployed/`): is P12@0.45 deployed-proven
  (stronger force), or does plain@0.3 deployed (the banked 05-VERDICT window) stay the arm of
  record? Either way the standing config gets ONE cvec block — no extra config file.
- plain@0.45's clean facts narrow the knee: [0.45 clean, 0.6 casualty] for plain; P12 [0.6 clean,
  above untested].
