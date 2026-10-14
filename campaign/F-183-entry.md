
### F-183 — THE CALIBRATE RUN, BACKED UP AND TESTED: adapt tier +12.7% on its probe but a trade-off on ours; the spare 1.2 GB filled and measured (2026-10-14 ~07:30, goal round 13)

**THE PROTOCOL (operator-specified).** See what --calibrate touches; back up the main config to
~/LLM/backup; run it; also check whether it calibrates VRAM occupancy (1.2 GB unused observed).

**WHAT IT TOUCHES (recon before running).** `setup.sh --calibrate` → `calibrate_config()` →
`tools/calibrate.py`: sweeps --pcie-frac {0,.2,.35,.55,.75,.9,1} → --spec-min-p {.3,.5,.7} → a
confirm round vs defaults → --pool-workers {15,10,8,4} → adapt-tier candidates — each value behind
an engine RESTART, decode-rate probe with 3-round medians; keeps only >3% winners; writes winners
into the config's args (the Launch-Law clobber) and per-PC into ~/.config/strata/settings.json.
Does NOT touch VRAM occupancy. Ran via a direct `calibrate_config()` call (setup's interactive
install path demands prompts / risks an 84 GB download with --yes). Backup first:
`~/LLM/backup/2026-10-14-precalibrate/` (configs, launchers, merged binary, BUILD.json, sha256
MANIFEST); the 13 other root jsons parked during the run so it found exactly one model.

**ITS RESULTS (its own probe):** pcie 0.9 nominal best but under the 3% bar (ranges overlap ours
0.35: 122-132 vs 0.9: 131-135) → it STRIPPED our --pcie-frac 0.35 (restored from backup per F-126);
spec-min-p 0.5 CONFIRMED optimal (134.9 vs 117.6/130.8); workers 15 CONFIRMED (136.8 vs 129-135);
**adapt tier: default 120.8 → every-1/80-swaps/decay-0.97 = 136.1 (+12.7%)** — the one real winner
it found, saved to the per-PC settings file.

**OUR VERDICT ON ITS WINNER (back-to-back boots, our probe battery):**

| arm | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| calibrated adapt (1/80/0.97) | 180.0 | 147.8 | 192.0 | 79.6% |
| standing adapt (0.85) | 183.5 | 143.0 | 186.4 | 81.0% |
| Δ | −1.9% | +3.3% | +3.0% | −1.4 pts |

Mixed, nothing clears ±4%, and the directions run against M1 and acceptance (both on the priority
list). Its probe = one long generation (TIER_MAX_NEW) where swap cadence dominates; ours = many
short diverse requests where the learned profile + B6 pins already manage the tier. **NOT ADOPTED**;
standing config restored byte-identical from backup.

**THE SPARE 1.2 GB (--vram-reserve-mib 2000 → 800):** cache 10,190 → **10,703 experts (20.31 GiB)**,
VRAM 32.07/32.6 — the gap filled as asked, then measured:

| arm | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| vram 800 (3 probes) | 170.8 | 148.4 | **195.6** | 79.9% |
| standing 2000 (control) | 183.5 | 143.0 | 186.4 | 81.0% |
| Δ | **−6.9%** | +3.7% | **+4.9%** | −1.1 pts |

A clean CONCURRENCY-vs-SINGLE-STREAM trade: concurrent slots pull a wider expert union, so +513
slots raise their hit rate; single-stream was already ~99% and pays the bigger cache's overhead.
Under the no-regression law NOT adoptable into standing (M1 −6.9% breaches), but recorded as the
**concurrent-favoring shelf arm** — it compounds with the same-signed adapt-tier arm and, later,
with F-31 slot coupling. If real-use priorities ever favor concurrent aggregate over M1, the 262k
daily driver can carry vram-800 (+ the adapt winners) as a labeled variant.

Standing recipe verified restored (reserve 2000, decay 0.85, 10,190 experts, healthy). Sources:
`campaign/results/cal-adapt{1,2,3}.txt`, `cal-ctrl{1,2,3}.txt`, `vram800-{1,2,3}.txt`;
`/tmp/calibrate-run2.log` (the full sweep); backup manifest in `~/LLM/backup/2026-10-14-precalibrate/`.
