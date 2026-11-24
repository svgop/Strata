# Slot-9 deployed-context decision test — VERDICT: WINDOW (ships as the arm of record)

Vector: `qwen3.8-flash-next-persona-r03-r03s9-s9-rho1.gguf` (direction.8, add), dose 0.3,
**the user's deployed template** (`gsq-iq3s/chat_template.jinja`, asserted at boot, max effort,
thinking on), 36 AB pairs (18 t0 + 18 seeded), blind, swap-balanced.

| criterion | on/off wins (p) |
|---|---|
| assertive | 12/4 (p=0.038) |
| serious | **13/1 (p=0.0009)** |
| direct | **12/3 (p=0.018)** |
| factual | 1.00 → 1.00 (McNemar p=1.0) |
| words | 205 → 215 (+5%, no truncation, no collapse) |
| trait/100w (lexicon) | 0.235 → 0.328 (+40%) |
| judge position bias A | 0.625 (deployed calibration range) |

## Why this is the decision

- The verdict rule of record (since r02): the deployed ON/OFF paired test decides. r02 deployed
  was null (ass 9/7 p=0.40, ser 4/5 p=0.75, dir 6/8 p=0.79 at dose 0.08). Slot 9 at 0.3 moves ALL
  THREE criteria significantly under the same template, judge, and power — with facts untouched.
- Capability held at every gate: factual suite 16/16 on both sides, no truncation, no length
  collapse (the vector actually lengthens answers slightly — register expansion, not damage).
- The stock-context dose table (`../04-r03-slot9/VERDICT.md`) brackets the knee: 0.3 clean,
  0.6 first factual casualty. 0.3 sits inside the window with margin on both sides.

## Status

This is an EVIDENCE-ARMED result, not a standing deployment: the standing config remains
untouched (fences PASS before/after, learned profile untouched, no engine binary change). Next
operational steps, in order: (1) P12 projected arm at the 0.6 knee prices the router channel
(`../06-r03-p12/`); (2) if adopted into the standing recipe, it goes through the campaign law's
full gauntlet (interleave parity on all priority lanes + MMLU gate) — NOT by editing the standing
config from this result alone.

## Known limitations, stated

- 18 t0 pairs per dose; consistency filter conservative; the result rules in a LARGE effect, and
  the r02 null rules nothing out at small effect sizes.
- Judge is weak: row-level judge failures occur (1/36 in the 04-0.3 dose); counts use judged
  pairs. Known-positive/negative judge controls are adopted as a future gate (eval audit #7), not
  yet implemented.
- Dose 0.3 is calibrated for THIS capture (mean‖h₉‖ = 1.0276); re-arming after any model or
  template change requires re-pricing, not assumption.
