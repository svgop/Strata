---
type: note
id: pool-split-notes
status: built-not-armed
date: 2026-10-26
source: F-348 (Project Maya CPU_SPLIT transfer), builder task "pool split"
---

# STRATA_POOL_SPLIT — the pool's anti-straggler chunk count (built, default off)

## Verdict

PATCHED AND BUILT. The Maya CPU_SPLIT mechanism is already 90% in the tree — the pool's claim
protocol (`claim()`, one CAS on `head_` per task, `pool.cpp:856`) is dynamic work-stealing by
construction. What was missing is only the granularity: `phase_tasks()` cuts each multi phase into
`3 * threads` fixed row ranges, so a worker stalled INSIDE a task (cold blob page fault, OS
preemption, DRAM contention) pins up to a full thread-share of the phase behind it. The patch arms
an env-gated chunks-per-thread count. Nothing about the queue, the epoch protocol, or the kernels
moved.

## 1. Analysis — the work unit and where a straggler bites

Read `expert_pool_dispatch_multi` (src/core/expert_source.cpp:4532+) and its drain:

- The dispatch groups a verify window's CPU-miss entries into per-expert `ExpertJobMulti` jobs
  (one job per distinct expert, up to MAXT=8 token rows each) and calls
  `ExpertPool::run_split_multi` / `run_split_multi_native` (pool.cpp:1057/1096).
- The pool runs TWO phases per batch: gate/up rows over `n * FF` rows (mode 3/5), then the host
  quantizes intermediates sequentially, then down rows over `n * H` rows (mode 4/6). The phase is
  one batch: `mtasks_` equal global-row ranges, `g0 = mrows_*i/mtasks_`, `g1 = mrows_*(i+1)/mtasks_`
  (pool.cpp drain, modes 3..6). Workers AND the host claim tasks one at a time through the CAS
  counter — a slow thread takes fewer claims. There is no per-entry monopolization: a task may span
  two experts' rows, but it never holds a whole expert unless the task count is 1.
- The exposure is the chunk COUNT, not the unit: at 3 tasks/thread a stall inside one task holds
  1/(3*threads) of the phase for the duration of the stall; worst case one worker's three queued
  tasks serialize behind one page fault. The 88-150 us per-entry spread is exactly this tail.
- `phase_tasks()` (pool.cpp:1053) already accepts an explicit `--pool-tasks N` (0..4096, capped by
  rows) — the lever existed, the automatic default did not move.

## 2. The patch (3 files, +69/-1 lines)

- include/strata/kernels/cpu/pool.hpp: `int split_per_thread_ = 0;` + doctrine comment.
- src/kernels/cpu/pool.cpp:
  - ctor: after `n_` is final, read `STRATA_POOL_SPLIT` (nonzero arms) and
    `STRATA_POOL_SPLIT_CHUNKS` (clamped 2..64, default 4); one startup line when armed.
  - `phase_tasks()`: explicit `--pool-tasks` wins; else if armed, `min(rows, min(kMaxTasks,
    CHUNKS * threads))`; else the historical `3 * threads` expression untouched.
- src/kernels/cpu/pool_tasks_test.cpp: parity harness extension (below).

Modes 1/2 (`run_split`, the single-token path) and the oracle fallback are untouched.

## 3. Parity — bit-identical by construction AND by harness

Construction: a phase's tasks tile `[0, mrows_)` exactly (floor-division ranges, task i+1 starts
where task i ends). Every output row (expert e's row r) belongs to exactly one task; the row
kernels (`s2_expert_gu_rows_multi`, `s2_expert_down_rows_multi`, `native_gu_rows`, `q2_rows_any`,
`native_down_rows`) compute each row from the blob and activations with no cross-row state. The
intermediate quantization runs on the host BETWEEN phases, after all gate/up tasks join — the
gu -> quant -> down pipeline does not see the task count. No partial sums cross chunks. Any
`mtasks_ >= 1` writes the same bits; only the schedule moves.

Harness: `pool_tasks_test` already proved the invariant for explicit task counts (bitwise memcmp
against the default partition across tasks 1..4096, workers, host on/off, NaN-poisoned outputs so
an unwritten row fails). Extended it with the env path: chunks {2, 4, 64} x host {on, off} x
{automatic, explicit --pool-tasks 41 precedence} x 2 repeats x n in {1, 7, 97} jobs.
RESULT: `pool tasks: 240 bitwise comparisons passed` (168 pre-existing + 72 new). The NaN prefill
also proves every row is rewritten exactly under the finer split.

## 4. Build

`.venv/bin/cmake --build build --target strata -j32` — clean link (pre-existing generate.cpp
warnings only). `pool_tasks_test` rebuilt and run. NOT deployed to bin/, server NOT booted.

## 5. Risks (mechanism-priced, none disqualifying)

- Claim CAS contention: one CAS per task per phase; at chunks=64 x 6 threads = 384 tasks, ~64
  CAS/thread/phase — microseconds against a ~0.5-1 ms phase. At chunks=64 on a many-core box the
  kMaxTasks=4096 clamp binds first; `begin_batch`'s 0xffff abort stays unreachable.
- False sharing: adjacent tasks of the SAME expert write adjacent rows of `split_multi_[e].ff[t]`
  (the boundary shares 1-2 cache lines per token slice; ff row = FF*4 = 2560 B). Already true today
  whenever a range spans experts; finer chunks add boundaries, not a new mechanism. The blobs'
  DRAM stream (0.92 MB/expert gu pass) dwarfs it.
- Empty tasks: the OFF path keeps today's shape (3*threads even when rows < that — pre-existing);
  the ON path caps by rows, so it strictly reduces empty tasks.
- The watchdog (`kStall` 60 s) semantics unchanged; `wait_done` counts the same way.
- Startup line: generate.cpp:6061 still prints the historical "3 * threads (automatic)" when the
  env is armed (out of this task's file scope); the pool's own armed line prints adjacent. Fix the
  generate.cpp line when this graduates from opt-in.

## 6. How to arm (one-line rollback per the campaign law)

    STRATA_POOL_SPLIT=1                    # 4 chunks/thread (24 tasks on the 6-thread pool)
    STRATA_POOL_SPLIT=1 STRATA_POOL_SPLIT_CHUNKS=48   # the Maya shape
Rollback: unset STRATA_POOL_SPLIT (=0). Byte-identical path restored; no relink needed.

## 7. Acceptance test (the falsifier, next session)

The interleave settles it: 3v3 arm/control inside one warm window, STRATA_POOL_SPLIT=1 vs unset,
decisive counters = `pool phases` ms + the per-layer-group GPU wait + per-entry p99 (the spread IS
the target: 88-150 us should compress toward 88). Chunks ladder 4 -> 16 -> 48 -> 64; a chunks value
that wins the tail but loses the mean is the false-sharing signature, not a win. Kill criteria
per the law: any lane regression, or no spread compression.
