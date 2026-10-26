# SCOUT-2 — 5090 clock-lock field data (2026-10-26)
**Central finding: the recipe's fear is real but aimed at the wrong knob.**
- **-lmc (memory lock) on desktop GB202 = POISON:** ogkm #1265 + LACT #1128 (RTX 5090, driver 610.43.03): memory lock at a value the card already hits cost −16.43% FPS, −118 MHz realized core, STRANDED 142.1 W; the lock flips the global arbiter (clock candidate 2944→2788). NO-GO.
- **-lgc max-only (`-lgc 0,<observed>`) = benign:** ogkm #1268 same author: 0,pmu_mhz forced a V/F re-evaluation: −0.13% FPS, −10.9 mV, −15.6 W (power SAVINGS, no perf loss). CONDITIONAL GO at observed≈2880-2900; never min=max; never max below ~2700 for full-power work.
- Counterpoint: 5080 Laptop -lmc fixed P-state flapping (#1248) — laptop regime, do not transfer.
- NVIDIA forum 355113: Blackwell Linux perf-limiter BITMASK reports wrong during throttling — validate via per-sample clocks/power telemetry, not limiter flags.
- CloudRift cuBLAS-on-5090 (Apr 2026): careful PAIRED methodology alone surfaced a 50-60% driver bug on an unlocked 5090 — the no-sudo path works.
**GO/NO-GO:** -lgc 0,<observed> conditional GO (one bounded A/B with -rgc rollback if paired CV ever breaches the gate); -lmc REMOVED from the recipe; CPU governor/EPP = lowest-risk sudo knob if ever granted.
**ALIGNMENT: DEFER sudo pinning; the interleaved-paired form is the verdict source** — F-340 already produced gate-passing verdicts without locks, paired interleave cancels the drift pinning would remove, and the field data is single-author/driver-610-specific. Row's sudo item DEMOTED to a bounded, conditional experiment.
