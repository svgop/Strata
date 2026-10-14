
### F-188 — "HAS SINGLE-STREAM REGRESSED?" — THE FROZEN-MASTER A/B: no engine regression; the 190-200 range is the box's peak state (2026-10-14 ~20:00, goal round 17)

**THE QUESTION (operator: single feels below the 190-200 class; pull the frozen master from
/mnt/DATA, build it, run the same tests — improved or worse since then?).**

**THE SETUP.** /mnt/DATA/LLM/Engine/strata-custom-2 copied to `~/LLM/strata-frozen-baseline`
(read-only source, allowed); one source fix (the port's `<cmath>`, the only Linux delta);
`-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` (the frozen CMake defaults CUDA OFF);
built clean (138 steps, `strata 0.1.43.1`). Run directory wired to OUR venv, OUR data/ (symlink —
same learned profile), same Custom-data; three back-to-back boots in one drift window:
A = frozen @ anchor recipe (batch 4, BATCH_MTP=0); B = current 0.1.44 @ the SAME anchor recipe;
C = current @ current recipe (F-31 + batch 6).

**THE RESULTS** (clean-run means; each boot had ONE collapsed run ~97-100 t/s across ALL metrics —
a transient, see below — dropped, noted):

| boot | M1 | pair | 6-conc |
|---|---|---|---|
| A: frozen 0.1.43.1 @ anchor | 176.8 | 175.6 | 214.6 |
| B: current 0.1.44 @ anchor | **185.4** | 157.4 | 208.5 |
| C: current @ current recipe | 180.9 | 174.1 | 215.3 |

**VERDICT 1 — NO SINGLE-STREAM REGRESSION FROM OUR WORK. The current binary is the FASTEST M1 of
the three boots** (185.4 anchor-recipe / 180.9 current-recipe vs the frozen engine's 176.8 — same
hour, same recipe for A vs B). The user's 190-200 memory is the box's PEAK WINDOW: even the frozen
engine — the binary that measured the 190.3 anchor — reads **176.8 now, −7% below its own anchor
mean, in the same hour our current binary reads 185**. The drift is environmental and swings every
recipe and every binary together (session M1 record: 160-196 on identical recipes; the frozen
engine's own anchor-era 190.3 vs 176.8 tonight). The engine improved; the box drifts.

**VERDICT 2 — the recipe arc is net-positive.** Current@current vs frozen@anchor: M1 +2.3%, pair
−0.9%, 6-conc +0.3% — and vs each engine's OWN era the concurrent means are at session highs
(pair 174-177, conc 215-218 class in this window).

**THE ANOMALY TO WATCH (two findings):** (1) Boot B's pair mean (157.4) sits below both the frozen
(175.6) and current-recipe (174.1) readings at the same hour — either a real pair cost somewhere
in the merged binary at the anchor recipe (COMMIT_BATCH default-on is the candidate) or drift
asymmetry between boots; the 01:30 merge A/B measured pair-neutral (160.4 vs 161.3), so no
conclusion today — QUEUED for one dedicated back-to-back re-test. (2) **THE RUN-2 STALL: each boot
dropped exactly one run to ~97-100 t/s across ALL metrics** (frozen run 1, current-B run 2,
current-C run 2 partial) — a recurring transient, NOT memory pressure (PSI avg 0.00, 110 GB
available), NOT GPU throttle events (dmesg clean, 40 °C, 2827 MHz). Suspects queued: the
expert-profile save (every 30 rounds, fsync-class pause), the adapt tier's refill thread, GPU
clock transitions. This is now the #1 smoothness item — it is worth several % of every battery's
mean and it poisons 3-run statistics.

Standing = boot C (current recipe), healthy. The frozen baseline stays at
`~/LLM/strata-frozen-baseline` as the permanent reference engine (launcher `launch-frozen.sh`).
Sources: campaign/results/frozen-{1,2,3}.txt, curanchor-{1,2,3}.txt, curcurrent-{1,2,3}.txt.
