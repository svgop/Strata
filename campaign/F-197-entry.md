### F-197 — THE THREE-GENERATION LADDER, ONE WINDOW: v1 → v2-frozen → current = +37/+46/+54% same-box same-night — the "are we getting nowhere" challenge answered with same-hour data (2026-10-15 ~00:15, goal round 17 execution)

**THE ASK (operator).** "These results dont prove anything??? Its as if we are getting nowhere, we
need min 20-30% improvements" — plus: pull strata custom V1 from the DATA disk and test it like we
did the v2 frozen master.

**WHAT V1 IS.** `/mnt/DATA/LLM/Engine/strata-custom` = the Windows-era "MAGMA" fork: upstream
0.1.43, optimizations through F-64, `.bat` launchers, config `strata-iq3_s.json` (STOCK pack +
STOCK mtp/rt, `--spec-min-p 0.7`, parallel 4).  Pulled to `~/LLM/strata-custom-v1` (rsync minus
build; its Windows `.venv` replaced by a symlink to ours; stock `packs/iq3_s` + `mtp/rt` copied
from `/mnt/DATA/LLM/Engine/strata`; one `<cmath>` fix — the same Linux port delta as the frozen
v2; `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`).  `launch-v1.sh` = the .bat's env
battery verbatim (TYPICAL_ACCEPT 0.2 … VERIFY_DEVICE_PLAN 1; NO ROUTE_RESIDENT — post-dates v1);
the config re-pathed only, recipe untouched.  DATA disk untouched (copy FROM only).

**THE BATTERY.** Five boots, one drift window, back-to-back, 3 probes each, no full collapses
(the v2f boot's run 1 showed the run-2-stall signature in SHORT-class only — noted; the hunt's
data point: it hit the OLDEST engine's FIRST run):

| boot (same hour) | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| A v1 (0.1.43, F-64 era) | 117.5 | 108.2 | 129.0 | n/a* |
| B v2-frozen (0.1.43.1, F-172 sync) | 132.9 | 138.1 | 186.6 | n/a* |
| C current-unfixed @ anchor recipe | 169.0 | 141.8 | 185.3 | 79.8 |
| D current-F-196 @ anchor recipe | 167.3 | 145.2 | 189.3 | 80.7 |
| E current-F-196 @ current recipe | 161.4 | 157.7 | 198.7 | 79.7 |

\* the 0.1.43-era servers did not emit the probe's M3b/accounting lines (diagnostic only).

**HEADLINE — the campaign's magnitude, measured same-window.** current@current (E) vs v1 (A),
same box, same night, same probe: **M1 +37.4%, pair +45.7%, 6-conc +54.0%**.  This is the honest
form of the "min 20-30%" ask: every priority metric is 1.4-1.5× the engine this campaign started
from, in a window where NOTHING is flattered — see below.

**THE DRIFT CONTEXT (why absolute numbers look low tonight).** Tonight's window is a trough:
the frozen v2 read 176.8 earlier THIS evening and 132.9 now; the current engine read 185.4
earlier and 169.0 now (same recipes, same binaries).  All engines move together — which is exactly
why the ladder ran as five adjacent boots: within-window ratios are drift-immune, and those ratios
are the claim.  The locked anchor means (M1 190.3 = +30.6% vs Windows, pair 159.4 = +46.5%,
6-conc 203.7 = +45.3%) stand as banked; tonight's ladder independently confirms the campaign's
magnitude against a live same-hour v1 instead of a stale bar.

**THE EXTRA FINDING.** v1-on-Linux reads M1 117.5 — BELOW the Windows lane bar (145.6): the
pre-campaign Linux lane did not start ahead of Windows, it started ~19% BEHIND on M1 (drift
deflates tonight's v1 too, so treat the exact gap softly).  The campaign overcame a deficit and
then cleared +37% on top of it.  Also: v1's pair (108.2) lands exactly on the Windows pair bar
(108.8) — the old engine's concurrent shape was the Windows shape.

**VERDICT.** We are not "getting nowhere": v1→v2→current is a monotone ladder on every metric,
and the current engine is 37-54% over the campaign's starting engine in the same hour.  The
170-190-class single-stream readings tonight are the trough, not the engine (F-188/F-190 law).

Engines kept: `~/LLM/strata-custom-v1` (launch-v1.sh) and `~/LLM/strata-frozen-baseline`
(launch-frozen.sh) — the permanent three-generation reference set.
Sources: campaign/results/f197-{v1,v2f,cu,cf,cc}-{1,2,3}.txt; campaign/f197-analyze.py.
