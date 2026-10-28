# Lane-trend tracker

One file, append-only, the whole campaign's lane history: `lanes.csv`.

```
engine,date,commit,label,lane,tps,ctx,kind
qwen,2026-10-27,50684ddd,F-367 standing bank (shm shadow store; THE bank),solo,222.0,0,mean
```

- `engine` — `qwen` (custom engine, IQ3_S) or `deepseek` (dsv41).
- `lane` — `solo` (M1), `pair` (M2 pair), `conc` (M2 6-lane aggregate). Same battery both engines: `campaign/probe-lane.py --port 8081 --log <engine log> --label <tag>`.
- `ctx` — context depth in tokens the leg ran at (0 = the standard short battery).
- `kind` — `mean` (multi-boot/leg banked mean, the preferred bank), `best` (a record leg), `single` (one boot, un-banked — read with the boot-variance law).
- One point (same date+commit+label) may carry several lane rows; missing lanes simply have no row.

**Protocol:** every adoption, falsification battery, or standing re-bank appends its lane rows (commit first so the hash exists), then `python3 campaign/tracker/build-trend.py` regenerates `lanes.json` + the chart. The chart's contract: the line inclines. A lane that turns downward across two banks is a regression signal, not noise — five-boot means outrank single legs.

**The incline law (AGENTS.md, campaign law 12):** the standing line never steps down. Only the standing recipe earns points; falsified arms and failed candidates never append (ledger only); a read below the last bank triggers the response ladder (same-hour control → five-boot means → revert under the binary-ladder law, the bank stands); `kind=tradeoff` rows (operator-signed capability trades) are the only labeled exception, and they fork rather than redraw the line. After every rebuild, read the chart — if any lane declined across the last two standing points, the round is not closed.

**The only way out is through (law 13):** a reverted regression stays OPEN until it is a real net win — scoped, composed, or restructured. When a wall stalls the push: zoom out (restate the constraint), recalibrate (re-measure where the time goes; new instruments if the old cannot see it), go deeper. No round closes on a shelf.

**The next lever queue (so the next point always has a named source):**
1. DS4_CACHE_THREADS=4 — VRAM swap-prep threads default 1; upstream's code note says try more on a PC; our swap-heavy single card is the case (T53).
2. DS4_CACHE_MARGIN sweep — swap admission (1.25 default).
3. dspark-conf sweep (0.65 standing).
4. Boot-time hugepages (vm.nr_hugepages≈56320) for the 109.5 GB store — needs a reboot to reserve.
5. Qwen side: the W-queue (campaign/ROADMAP-perf-8moves.md) — W1 adaptive spec and W9 S26 expert envs are the unexecuted high-ROI rows.

Depth probes (131K etc.) are a different axis than this chart's short-lane trend — record them in `results/` and reference from the ledger; only short-battery lanes go in `lanes.csv`.

DeepSeek lane history is short: the engine entered the custom tree 2026-10-28 (F-370). Points before that are the upstream-package engine on the same rig.
