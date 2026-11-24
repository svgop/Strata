# Bench numbers, 2026-11-24 ~14:00-14:10 — the pre-investigation bench (recorded late)

Recorded 2026-11-24 17:40 from the session transcript. These numbers were printed by `campaign/probe-lane.py`
and reported in conversation but NEVER teed to a file (the `--log` flag was mistaken for a save-to-file).
Three hours later the same agent read the tracker instead of this bench and told the operator that
"241.8 is October's record, not today" - wrong. Law 6 (`probe-lane.py` does not save files - pipe through
tee`) was not followed; this file closes that gap. The lesson is structural, not procedural: an un-teed
probe result does not exist for the next reader, including the same reader three hours later.

## Boot

- Engine: the refactored two-table kernel (`ec999a3bc`-era binary, = today's `engine/strata.pre-phase-gate`
  lineage), boot 13:58:04, warm-in paid, leg 1 discarded (F-360).
- Config: standing as of 14:08 - persona @**0.45 UNSCOPED** (table 1, add, 8-8) + refusal-r (table 2,
  project, 1-47). No phase scoping yet; dose was still 0.45.
- Template: the deployed jinja (sha 87412506). Learned profile: the 13:57 state (backed up pre-bench;
  today's `data/expert-profile-learned.pre-persona-campaign.bin`).
- Slot dir: `/home/svg/LLM/data-rt/conversation-slots` (home; before the DATA re-path).

## Legs (probe-lane, full battery)

| leg | M1 | 6-conc | note |
|---|---|---|---|
| 1 | 195.7 | 262.6 | cold - first-touch (F-360), discarded |
| 2 | 231.1 | 274.7 | measured |
| 3 | **242.6** | 281.5 | measured |

**Warm mean: M1 236.9, 6-conc 278.1.** (F-408 quiet-window bank, 11-20, no persona: M1 220.6 mean / 228.7 best.)

## Why this is the reference for the 17:10 regression question

Same day, same box, same profile state, same template as the 17:08 and 17:16 boots - which read M1
210.5 and 208.1 (three measured legs each). Between them: (1) the phase gate (`ba13d1acb`, binary 16:02),
(2) dose 0.45 -> 0.3 (`3c524a5dd`), (3) the DATA re-path (slot dir). The config ladder
`campaign/results/m1-phase-2026-11-24/` isolates (1)+(2) with the profile pinned; (3) has no
steady-state decode mechanism (slots are touched at park/restore only) and is covered by the boot proof.
