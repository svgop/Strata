# F-466 evidence (2026-11-25) - the step back: the request path ranked by the production log

- `production-request-time.txt` - `python3 campaign/reqlog-census.py --top 12` on `logs/custom-iq3_s.log`
  (112 boots, 21,470 accounting lines).  The selected boot is 11-22 15:59:10, the operator's traffic.
  The log is appended across boots; re-running the census on a longer log moves every number.
- `routed-only-streaming-trace.txt` - `tools/trace_ideal_tracking.py --routed-windows` on
  `data/calibration/trace-conc.bin` (gitignored, svg-owned), both legs of the F-465 diagnostic boot.
- `f452-capture-copies.txt` - the F-452 nsys capture's copy timeline: one prompt read's walk is 32.65 GB in 658 ms.
- `../ctx-f466-branch-2026-11-25/` - the depth ladder (six depths, idle box) and, after each depth, the BRANCH
  shape.  Its ladder.txt says "turn" on those six lines: the probe then appended the tail without the reply, and the
  engine's lines in serve-lines.txt show what that is (0 reused below 32,768 tokens; whole 32,768-token
  checkpoints above).  The harness printed CTX FAIL for this run on a restart count that included the boot's own
  start line; checked by hand against the log: one start line, no watchdog line, no request in error.
- `../ctx-f466-turn-2026-11-25/` - three depths and the TURN shape (the reply sent back): the first run of the
  fixed harness, CTX DONE.
- A first boot at 16:26 was void: the probe crashed on its first request (`urllib.request.Request` dropped in a
  rewrite) and the harness still printed DONE with zero measured lines.  No reading was taken from it.

All three boots ran the champion `3865e27a5ac1a170` through the launcher with battery-lib's cleanup: engine down
afterwards, learned profile re-pinned and verified (== `expert-profile-gold.bin`), no tracker point.
