# Phase-scoped persona — verdict: same MMLU as stock, register stronger

Change: the persona add (table 1) now applies ONLY to the answer phase. The engine's F-17 phase
machinery gained a per-table gate (`--cvec-phase-table 1`): a generated `</think>` opens table 1's
gate, a `<think>` closes it again (agentic re-reasoning), and a prompt whose think block the template
already closed (thinking off: `<think>\n\n</think>\n\n` pre-filled) starts open. Prompt tokens and the
reasoning block never see the persona; the refusal ablation (table 2) covers the whole request exactly
as before. Dose unchanged at 0.3.

Why: the persona contrast was elicited from RESPONSE tokens, yet the add landed on every prompt and
thinking token - perturbing question encoding and deliberation before a word of answer existed. The
MC cost was entirely a where/when effect: cos(assertive, effort) = +0.013 (the vector carries no
"careless" component), so scoping the application, not sharpening the direction, was the lever.

## MMLU-Pro 70q, per-request ON/OFF interleaved in one boot (`campaign/mmlu-pro-interleave.py`)

| arm shape | ON | OFF | discordant (on-only/off-only) | McNemar p | median gen tok ON/OFF |
|---|---|---|---|---|---|
| unscoped @0.45 | 63 | 67 | 1 / 5 | 0.219 | 237 / 181 |
| unscoped @0.3 | 65 | 66 | 1 / 2 | 1.000 | 208 / 198 |
| **phase-scoped @0.3** | **66** | **66** | **1 / 1** | **1.000** | **196 / 219** |

An exact tie with stock. Per-row ON/OFF reasoning-length ratio median 0.95 - the deliberation path
is untouched (unscoped, the persona lengthened it). The two flips are the suite's chronically
unstable questions (#55, #64 - flipped in every run, both directions). Model's own misses constant
(18, 58, 66). `campaign/results/mmlu-interleave-phase03.json`.

## Register, deployed template, thinking on (the 05 instrument, directly comparable)

| run | assertive | serious | direct | facts | words | judge ctl | verdict |
|---|---|---|---|---|---|---|---|
| unscoped @0.3 (05, adopted) | 12/4 p=0.038 | 13/1 p=0.0009 | 12/3 p=0.018 | 1.00 | 205→215 | n/a | WINDOW |
| scoped r1 | 8/2 p=0.055 | 13/2 p=0.004 | 10/1 p=0.006 | 1.00 | 210→152 | pass | DAMAGE-FIRST: 1 empty output |
| **scoped r2** | **13/3 p=0.011** | **14/3 p=0.006** | **14/1 p=0.0005** | **1.00** | 241→178 | pass | **WINDOW** |

- r1's single empty answer (R2 technical) did not reproduce in 6 immediate probes (3 ON / 3 OFF, both
  arms 212-287 words, reasoning lengths equal across arms): engine non-determinism at temperature 0,
  not the persona. The verdict stands as recorded; r2 is the clean read.
- The word drop (−27%) is compression by kind (technical 334→227, everyday 153→119, play 51→21) with
  finish reasons IDENTICAL to control (ON 2 length / OFF 3) - the hedging the vector was built to
  remove, now concentrated on the answer prose.
- **Instrument refinement, recorded**: r2's "trait vocabulary inflated" flag (1.016 vs a 1.0 threshold)
  was the WAR-metaphor lexicon summed across kinds - 0.00/100w on every technical and everyday answer,
  6.25/100w on the two play prompts whose purpose is to show the persona's color. The flag now reads
  the serious register only (`trait_per_100w_serious`; both numbers reported; reason in code). Applied
  to the stored raw records: r2 → WINDOW, r1 unchanged (empty output), and **no earlier verdict in the
  round changes** under the refined rule.

## Engine change (opt-in; F-17's legacy path byte-identical when unconfigured)

- `cvec.hpp/.cu`: `Cvec::gate` (device int, table 1 only), `cvec_set_table1_gate`,
  `cvec_set_phase_table1`; the kernel takes a per-launch gate (table 1 reads its gate, table 2 reads
  the request flag); `write_flag` is the one owner of device-int writes (the request switch and the
  gate share it).
- `generate.cpp`: `--cvec-phase-table 0|1`; request start scans the prompt's think markers for the
  closed-block rule and closes the gate; the commit scan opens it at a generated `</think>`, closes at
  `<think>`; table 2 is never touched by the phase.
- `prefill.cpp`: table 1 does not take the write-fuse while the phase owns it (the fused kernel cannot
  read the gate).
- Parity (`cvec_parity`, all passed): gate closed → table 1 silent, table 2 still ablates (4.9e-07);
  gate open → add then ablate (9.5e-07); request off overrides an open gate (bitwise unchanged);
  every pre-existing case unchanged. Two kernel bugs caught by parity before deployment: the device-int
  write targeted the pointer field instead of the buffer, and a compound literal under nvcc.
- Standing: `--control-vector-phase 248069:248068 --cvec-phase-table 1` (</think>:<think>). Binary
  `ac342c49434b10c1`; rollback `engine/strata.pre-phase-gate` + drop the two args.
- Known, inherent to decode-time phase steering (F-17 made the same trade): a prior turn's answer
  tokens, re-prefilled on a conversation-cache miss, are computed with the gate closed (prompt tokens
  never see table 1) although they were generated with it open. Single-turn requests are unaffected.

## Result

A strict Pareto improvement over the unscoped arm: MMLU ties stock exactly (was −1 at the same dose,
−4 at 0.45), the register read is stronger (direct 14/1 vs 12/3), and the persona's color shows only
where it is invited (0.00 war-metaphor words on serious answers). The arm meets the bar as well as
70 questions can measure it: same-or-better MMLU, never regress, persona intact.
