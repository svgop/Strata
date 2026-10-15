### F-205 — THE ARM-WRAPPER BUG FOUND AND REPAIRED: every wrapped arm since F-200 booted the BARE engine; DT re-validated on the true base (+4.0% M1, stays); F-203's rejections VOID; new method law (2026-10-15 ~13:30, goal round 18 execution)

**THE BUG.** The arm wrappers used `set -a; grep "^export" launcher | bash; set +a` — the pipe
runs the exports in a CHILD shell that dies with the pipe; `set -a` does not propagate child
exports.  EVERY armed boot since F-200 ran the BARE engine (no TYPICAL_DEVICE, no
VERIFY_ASYNC_LAUNCH, no ROUTE_RESIDENT 0.75, no DT, no battery at all) plus its one arm var.
The tell that unraveled it: tonight's warm windows split perfectly by boot TYPE — all
launcher-direct controls warm (M3b 74-79), all wrapper arms cold (M3b 59-65) — impossible for
independent mechanisms, trivially explained by a missing battery.  The cold/warm M3b bimodality
across the night = mostly BARE-vs-FULL battery state, not environment.

**THE FIX, VALIDATED.** `source <(grep "^export" launcher)` in the parent, override AFTER
source (env-prefix on the launcher cannot work — it unconditionally re-exports its own values).
Validation boot (fixed wrapper + STRATA_WRAPPER_NOOP=1): **182.2/172.9/213.5, M3b 77.3** ≈
launcher control 187.8/173.6/215.6, M3b 76.2 — the wrapper is now faithful.

**WHAT STANDS / WHAT'S VOID.**
- STANDS: the standing recipe itself (all its readings were launcher-direct boots): tonight's
  warm controls **179.9-187.8 M1 / 171.0-175.0 pair / 210.1-215.6 conc, M3b 74-79**.
- STANDS: the F-200 adoptions (TYPICAL_DEVICE, VERIFY_ASYNC_LAUNCH, RR 0.75) — their arm deltas
  were bare-based, but the stack-level no-regression evidence (the f200-adopt boot, tonight's
  controls) holds; magnitudes need re-attribution, adoptions need no revert.
- RE-VALIDATED TONIGHT ON THE TRUE BASE: **STRATA_SPEC_DRAFT_TEMP=0.85 = +4.0% M1**
  (187.8 vs 180.6 no-DT), pair +0.7, conc +0.7 — AT the adopt bar, KEEPS its adoption (the
  bare-base +13.4% was the bare engine's larger acceptance headroom).
- VOID: **F-203's two rejections (DOORBELL_STORE, spec-8)** — both were bare+arm vs full-battery
  comparisons.  The dbstore M3b-13pt "correctness signature" was the missing battery.  Both arms
  go back to UNTESTED; re-queue on the fixed wrapper.
- The F-205 LFUSE/DF_PLE interleave (also bare-based) = VOID; re-queue (the node-inventory
  savings estimates stand on their own).
- F-201's DT sweep curve (bare-based) remains valid as a bare-engine dose-response; the peak
  location may shift on the full base — optional re-sweep.

**NEW METHOD LAW (added to the battery protocol):** every arm wrapper must be validated by a
NO-OP boot (unknown env var) reading warm-class M3b ≈ the launcher before ANY of its arms count;
overrides only via source-in-parent; launcher re-exports make env-prefix overrides useless.
**M3b stays the state-canary** — it caught this bug.

Also tonight: F-204 = DVFS exonerated (dmon: SM clocks 2797-2827 MHz, mem 13801-14001, zero dips
through a full battery — the gpu-lock holds under probes); the two F-204 scan maps banked
(research/F-204-emit-batching-map.md = the R33 plan; the node inventory in
research/F-204-node-inventory.md = the fusion queue with ~-900 to -1,000 nodes/window available,
three fusions already implemented behind gates).  FUSE_WAIT un-gated (default off, built).

Standing restored + healthy.  Sources: campaign/results/f205-*, f205v-*, f204-ctl,
/tmp/dmon-clocks.log.
