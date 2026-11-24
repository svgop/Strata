# MMLU interleave verdict — the persona window is real and it is at dose 0.3

Instrument: `campaign/mmlu-pro-interleave.py` — the 70q gate with ON vs OFF interleaved
per-request in ONE boot (`experimental_speed_projection: false` = stock), 8000-token budget (the
3000-token thinking cap cannot fire), every row carrying finish reason and generated tokens.

| dose | ON | OFF | on-only | off-only | McNemar p | wraps | median gen tok on/off |
|---|---|---|---|---|---|---|---|
| 0.45 (adopted in 0bd36187e) | 63/70 | 67/70 | 1 | 5 | 0.219 | 0/0 | 237/181 |
| **0.3 (adopted now)** | **65/70** | **66/70** | **1** | **2** | **1.000** | 0/0 | 208/198 |

Rows wrong under BOTH arms (the model's own misses, constant): 18, 58.
OFF-only-right at 0.45: 25, 30, 37, 50, 66 → at 0.3: 37, 50. #30 and #66 recover as the dose steps down.

## Reading

- **The window is a dose-response, not a flat cost.** Same boot, same window, per-request switch:
  at 0.3 the arm is statistically indistinguishable from stock (1 gained, 2 lost, equal reasoning
  length); at 0.45 it cost 4 net with reasoning lengthened (+31%). This matches the measured
  mechanism (harmonic-mean effective dose 1.59× nominal at 0.45 → ~1.06× at 0.3) — the register
  effect survives at 0.3 (deployed AB: ass 12/4, ser 13/1, dir 12/3, facts 1.00, `05-r03-deployed/`)
  while the MC-reasoning cost drops into the noise.
- **The truncation artifact was real but secondary.** The first gate (90.0) had 3 of 7 fails
  force-closed at 3001 tokens; the interleave (no wraps) reproduced the 63 anyway — so the dose,
  not the cap, carried the loss. The cap is a separate hazard for long-reasoning gates.
- **Power statement**: 70 questions cannot prove equivalence; p=1.000 at 1-vs-2 discordant rules out
  a 0.45-sized effect (5-vs-1) at this dose, not a ≤2-question one. The register evidence is from
  the deployed AB; the MC evidence is this interleave.

## Standing

`config/custom-iq3_s-8081.json`: persona row `…r03s9rnsp-s9-rho1.gguf:0.3` (table 1, add, 8-8) +
refusal-r (table 2, project, 1-47). Fallback if a future gate reads hot: step to 0.2; the row's
register effect is monotone in dose (04-r03-slot9 ladder). Server STOPPED after each run; learned
profile backed up before each boot (`Custom-data/learned-profile-pre-*.bin`).

## Queued (from the 18-angle forensics)

- Harmonic-mean dose calibration in `build_vector` (define dose in the block's read domain).
- Per-table enable (`cvec_set_enabled2`) for answer-phase scoping of the persona.
- SYCL `cvec.dp.cpp` lacks the second table while SYCL core calls `covers()` — drift, fix before
  any SYCL two-table boot.
- Gate hygiene: store content/finish/tokens (the interleave does; `mmlu-pro-gate.py` does not);
  two-table cases in `tools/test_handshake.py`; `web/app.js` composed-string rendering.
