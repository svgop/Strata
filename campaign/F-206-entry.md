### F-206 — THE FUSION TRANCHE (clean negative: node savings ≠ wall time) + R33 EMIT DE-SERIALIZATION SHIPPED (6-conc 219.1 = highest concurrent mean ever) + a power cut survived mid-battery (2026-10-15 ~12:30, goal round 18 execution)

**THE FUSION TRANCHE (on the FIXED wrapper — all arms warm, M3b 75-78, controls 181.5-191.6 M1).**
LFUSE −3.0% M1 / FUSE_WAIT −1.3% / DF_PLE −1.6% / the three-stack neutral — the −344 nodes/window
do NOT translate to wall time: the spacing is latency-hidden behind other terms (the graphs
brief's predicted "the line was already hidden" class).  All three stay default-off; the
FUSE_WAIT un-gate remains dormant.  A cheap, clean negative — the node inventory's bigger items
(#1 producer-fused quantize, #5 hc norm-fold, #6 batch-QSA suite) remain the fusion prizes, but
now with calibrated expectations: node counts are not the binding constraint at our spacing.

**R33 — THE SPLIT DRAFT (draft_launch/draft_finish + the two-pass emit loop).**  The per-slot
draft() was launch→spin→sync, serializing independent drafter streams; now every slot's residual
copy + draft graph launches first, then the waits run in launch order (the graphs overlap
GPU-side).  General draft() untouched (solo path keeps it).  Smoke: single + 3-concurrent
coherent; acceptance 76.6% class (unchanged — no distribution change, no MMLU class).

**THE VERDICT (5 runs across two batteries — the first cut short by a POWER CUT mid-battery;
run 1 pre-cut 175.2 + 4 runs post-reboot 186.1/173.3/219.1 M3b 76.9):**
vs the adjacent control band (185.6/174.3/214.0): M1 +0.3%, pair −0.6%, **6-conc +2.4% — and
219.1 is the HIGHEST CONCURRENT MEAN EVER RECORDED** (previous best 216.0/215.5).  Below the +4%
win bar → **ADOPTED AS HYGIENE** (F-196 precedent): strictly-better-or-neutral on exactly the
metric the mechanism targets (concurrent drafting slots), M1 flat as the mechanism predicts
(single stream: nothing to overlap), and the two-pass structure is the enabler for the batched
draft-head GEMM (the ~0.3-0.5ms follow-up in the F-204 emit map).  Code stays; scorecard
unchanged.  Post-reboot caveat noted: the 4-run battery ran in a fresh-boot window (cold caches,
proper 2-min warm-up applied) — the +2.4% carries that caveat, the all-time-high stands.

**OPS NOTES.**  The power cut wiped /tmp (rollback copies + scripts — git + ~/LLM/backup
survived; the F-206 code was committed BEFORE the re-run, the lesson applied).  engine/strata =
the R33 build (deployed, healthy).  Standing server restored on it.

Sources: campaign/results/f206-*, /tmp/f206-r33-redo.log (transient).
