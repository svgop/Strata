
### F-184 — F-31 SLOT COUPLING: ADOPTED — pair +11.5% for one env line (2026-10-14 ~09:30, goal round 14)

**THE QUESTION (operator: "proceed with F-31 and prove its worth it").** Concurrent batch slots ran
DRAFTLESS since the Windows era (STRATA_BATCH_MTP=0, suspended under the No-Regression Law after
F-30b): uncoupled drafts rarely exact-match at temp>0, and the rescue variant that made slots pay
was quality-banned (MMLU 95.7 → 91.4-92.9). Coupled drafting — the drafter samples with the
target's chain + the same Philox draw — is quality-safe by construction (coupled_draft.hpp:5-6),
and F-30b's own note said "the pair win came from drafting, not rescue".

**THE DIAGNOSTIC FIRST (the journal's own prescription: "a diagnostic boot, not more static
reading").** Static re-reading of the MERGED tree found the acceptance site plain exact-match
(`outb[i] == tok[i+1]`, generate.cpp ~10226) with the verify pick drawn by
`Verifier::sample_rows` (counter = row position, verify.cpp ~3172) — and the counter law
(coupled_draft.hpp:19-30) algebraically COINCIDES with the position convention. Booted
STRATA_BATCH_MTP=1 on the merged binary: **slot drafts accepted 60-67% sustained** — the coupled
class, live. The "counter staging defect" of the old journal does not exist in the v0.1.44-merged
tree; the wiring is complete. **F-31 was never a code project here — it was one suspended env
line.**

**THE PROOF (back-to-back boots, same night, our battery):**

| arm | M1 | pair | 6-conc | M3b | MMLU |
|---|---|---|---|---|---|
| STRATA_BATCH_MTP=1 (F-31) | 175.1 | **162.8** | 193.7 | 79.8% | **97.14 ± 1.99** |
| control (=0, standing) | 179.1 | 146.0 | 192.8 | 80.3% | (95.71 anchor class) |
| Δ | −2.2% (noise; serial path untouched) | **+11.5%** | +0.5% | −0.5 pt | best gate of the campaign |

**WHY PAIR AND NOT 6-CONC:** at pair, 2 slots + 2 draft rows = a 4-row window — the ~63% chance of
banking a second row per slot comes nearly free. At 6-conc the window doubles 6→12 rows and the
MoE expert-union cost of the extra rows eats the gain (+0.5%). The serial path (M1) is mechanically
unaffected; its −2.2% is the night's spread.

**QUALITY:** exact-match accept only — the emitted stream is the target's own picks on both accept
and reject (the F-15 divergence lesson cannot arise; the F-30b rescue stays OFF by default).
MMLU gate 97.14 ± 1.99 = the highest single gate ever recorded on this lane. The 60-67% slot
acceptance also lifts nothing distributional: drafts only decide HOW MANY tokens a window banks.

**ADOPTED:** `STRATA_BATCH_MTP=0 → 1` in BOTH the standing 131k launcher and the user's real-use
262k launcher (concurrency is exactly the real-use case). Standing restarted on the adopted arm
and verified (env confirmed on the engine process; health green). Sources:
`campaign/results/f31-{diag-pair,bmtp2,bmtp3,ctrl1,ctrl2,ctrl3}.txt`, `mmlu-f31-bmtp-ml.json`.

**WHAT THIS OPENS:** the concurrent ceiling moved for low concurrency; the 6-conc ceiling is now
purely the expert-union cost of wider windows — the tier/cache arms of F-183 (which pay only under
concurrency) and deeper slot drafts (draft(2,...) offers ONE proposal per slot; a longer slot
chain, or the F-183 cache arms stacked ON the now-cheaper slot rows, are the follow-ons).
