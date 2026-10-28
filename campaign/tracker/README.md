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

Depth probes (131K etc.) are a different axis than this chart's short-lane trend — record them in `results/` and reference from the ledger; only short-battery lanes go in `lanes.csv`.

DeepSeek lane history is short: the engine entered the custom tree 2026-10-28 (F-370). Points before that are the upstream-package engine on the same rig.
