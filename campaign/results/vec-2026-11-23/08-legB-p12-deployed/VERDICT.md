# Leg B — P12 deployed ON/OFF at 0.45: WINDOW (the adoption dose)

Your template (asserted at boot), max effort, thinking on, projected row direction.8 at 0.45,
36 pairs, judge_control_ok = true.

| criterion | on/off wins (p) |
|---|---|
| assertive | 13/3 (p=0.011) |
| serious | **13/1 (p=0.0009)** |
| direct | **11/3 (p=0.029)** |
| factual | 1.00 → 1.00 |
| words | 244 → 222 (thinking-on lengths; no truncation) |
| judgeA | 0.75 (deployed calibration range) |

Secondary-signal note: trait/100w read 0.367→0.0 — the lexicon counter disagrees with the judge
sweep here; criteria + factual gates carry decisions, the lexicon is advisory only (flagged for
the judge-controls work, not blocking).

## Adoption decision

P12@0.45 deployed-proven: all three criteria significant, facts intact, judge control passed.
**The standing config `config/custom-iq3_s-8081.json` carries exactly this arm** (persona-r03
rnsp:0.45, range 8 8, add) — replacing the refusal-r block (one table + one mode per boot; the
displacement is measured by the refusal battery, `../09-refusal-pair/`, not assumed). plain@0.3
deployed (`../05-r03-deployed/VERDICT.md`) remains the banked fallback: if 0.45 ever reads hot,
step down to the plain row at 0.3 — the row set and the recipe are on file.

## fences FAIL in this run's fences.txt — explained, verdict unaffected

The `before` snapshot (18:14:11) read the standing config's Nov-20 hash (e85ea8ea…); the `after`
(18:27) read the adoption edit's hash (794dd52d…) — the edit's writeback landed mid-run, ~2 min
after the snapshot, despite the edit completing before launch. The measurement is unaffected: the
lab boot's cvec block is REPLACED wholesale by derive_config (all four args), so the boot armed
P12@0.45 add:8-8 regardless of which standing version the snapshot saw — verified at boot
("armed add:8-8 steered 1/48"). Launcher, engine binary, and learned profile: byte-identical
before/after. The next run's fences take the new baseline; the verification boot closes adoption.
