# P-A prefill cadence - design note (2026-10-26)

Implements campaign/research/scout2/latency-lanes.md (REVISED P-A) in the Python front end only.
Env-gated, default OFF: `STRATA_PREFILL_CADENCE=1` arms it; `STRATA_PREFILL_CADENCE_N` sets the
spacing (default 2, floored at 1).

## Files touched

- `serve/server.py` - 93 insertions, 0 deletions (the only file changed by this task).
- No new dependencies; no engine (C++) files touched; `prefill_chunk` / `prefill_auto_max` /
  `--prefill` stay C++-side and untouched.

## How prompts are submitted today (the surface the gate sits on)

- `StrataEngine.generate_batched` (batch mode) sends one prompt read at a time on the control
  lines: solo `GEN ...` when alone, `BGEN <slot> ...` into a batch slot otherwise. The engine
  interleaves that read's chunks with the other slots' decode windows; a 32k-token read makes
  many windows carry a prefill chunk, and those windows cannot replay the decode graph.
- `PP_CHUNK_MAX` (32768) is NOT a server-side chunker: it only sizes the stall watchdog's
  patience for long chunk reads. So per the task, the server enforces the cadence and leaves
  chunking to the engine.

## The tick

One decode round = one `BT <slot> <tok>` line, counted in `StrataEngine._pump` (the only place
batch slot lines already route) into `self.cadence_round`. Rationale: BT lines are the server's
existing per-window decode signal (WSTAT is opt-in telemetry, not always on; DONE lines are
per-request). With several slots decoding, one engine window emits several BT lines, so the
counter then counts tokens rather than windows - the gate then demands more decode work between
two prompt reads, which is the direction the throttle wants (vLLM's cadence is per step; per BT
line is the same bound at one slot, tighter at many). Unarmed, the counter never moves.

## Gate logic

`StrataEngine._cadence_open()` - a prompt read (a prefill turn) may start when either:

1. **Waiver:** no slot is busy (`not any(self.slot_busy)`, the same signal the existing `alone`
   check reads) - a lone request's read cannot slow anyone, so its TTFT wins; or
2. `cadence_round - cadence_prefill_round >= STRATA_PREFILL_CADENCE_N` - the last prefill turn
   is at least N decode rounds behind, so the windows in between decoded without a prompt chunk.

`cadence_prefill_round` is stamped at every prompt-read send in `generate_batched` (the solo
`GEN` and the slot `BGEN`), so a request arriving while a waiver read is in progress spaces its
own slot admission from that read's round, not from zero.

Placement: inside `generate_batched`'s slot-admission loop, before `pick_slot` / the `BGEN`
send, i.e. before the request claims a slot. While the gate is shut the request gives back the
control lines (`ctl.release()`, the loop's existing invariant: never wait holding them, or a
BYIELD-ed read deadlocks) and waits on `slot_cv` with a 0.25 s timeout - woken early when a slot
frees (the waiver may open the gate at once), polled for round ticks otherwise. Each wake
re-checks the gate, the queue deadline (`QueueWaitExpired`), engine liveness (`EngineDied`),
cancel, and emits the turn's usual 10 s heartbeat. Unarmed, `_cadence_open` returns True before
any of this runs, so the path is byte-identical to before (93 insertions, 0 deletions).

Debug: a request that waited carries `cadence_wait_s` in its `slot_live` record (extra dict key;
`slots_view` reads only its own keys, so nothing else changes). Arm-time log line in `main()`:
`strata serve: prefill-cadence armed (1 chunk / <N> rounds, waiver on zero active decodes)`.

## Not done on purpose

- No server-side prompt slicing: the engine owns chunk size (`--prefill`); the scout note's
  "<=512-token slices" would need engine cooperation or `pipe=` tuning, out of scope here.
- PP_CHUNK_MAX left as watchdog math only (it is not wired as a chunk cap).

## Rollback

Unset `STRATA_PREFILL_CADENCE` (default) - the code paths all short-circuit on that one module
constant; behavior is unchanged without redeploying. To remove the code: revert the single
commit/hunk set in `serve/server.py` (93 added lines, none removed).

## Risks found

1. The tick counts BT lines, not engine windows (see above) - intentional approximation; at one
   decoding slot it is exact.
2. A long read, once admitted, runs to completion (or BYIELD) - the server cannot pause a read
   mid-prompt, so the cadence spaces prefill TURNS, it cannot thin the chunks inside one turn.
   The full vLLM per-step throttle needs the engine to re-read from the slot mid-prompt.
3. While the gate is shut, queued admissions hand the control lines around in FIFO ticket order
   without sending; each handoff is cheap and the gate opens within N decode rounds (or the
   waiver) as long as the engine decodes. A decoding-stalled engine is covered by the existing
   ENGINE_STALL_S watchdog and the queue deadline, same as the free-slot wait next door.

## Validation

- `.venv/bin/python -m py_compile serve/server.py` - OK.
- Isolated logic test (stub engine, no live engine touched): unarmed gate always open, waiver on
  zero busy slots, shut/open at N-1/N rounds since the last prefill turn, waiver wins over round
  math, default N=2 and floor at 1 - all pass.
- Live server (port 8081) not restarted, per the task constraint.
