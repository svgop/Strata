# R2-S3 — TWO COHORT VERIFIERS on one GPU: the conc-300 implementation plan

2026-10-17, from T13B C1 (campaign/research/T13B-concurrency-scheduling.md) + F-223's structural
theorem + the F-232 re-price sweep (the smalls are dead; this is the remaining conc lever).

## 0. The measured case (why this, why now)

- 6-conc window = 22.32 ms: GPU serial 16.47 + pool 3.95 (ADDITIVE host bookkeeping) + tail 1.2;
  ONE window in flight; GPU idle 26% of the period (T13B 1.6/1.7).
- F-223: no intra-window reordering can hide the pool — ONLY another window in flight can.
  F-223 also killed batch G=2: two GROUPS shared one Verifier's doorbell/flag/staging singletons
  (verify.cpp:3089-3092 reset per stage_batch) and corrupted under the PCIe share. Two Verifier
  INSTANCES own their buffers separately — the ghost is structurally impossible, not merely fixed.
- Arithmetic (T13B C1): period -> max(GPU ~16.4+contention, host ~5-6) => 17.5-18.5 ms =>
  in-window 354-374 rows/s => 307-325 wall (+20-28%).  THE one unmeasured number: two-stream
  contention (two ~2k-node graphs sharing 170 SMs).  This plan's pilot exists to measure it.

## 1. Shape

- Cohorts: slots {0,1,2} -> Verifier A, {3,4,5} -> Verifier B (balanced 3+3 at 6-conc; at pair/M1
  only A is active and the loop degenerates to today's serial path — M1/pair lanes untouched).
- Verifier B: a second Verifier constructed exactly like ver (same graph capture per layout; its own
  cs2_ stream, its own h_seq_/h_flag_/h_flagA_/h_flagB_/h_x_/h_ymiss_ mappings, its own staging,
  its own graph-LRU).  Weights are SHARED (WeightRef views) — only activations/staging/graphs dup.
- Service loop (the heart): launch A's graph on cs_, launch B's graph on cs2_, then ONE reactor
  thread serves both doorbell rings round-robin: A's layer l ring -> pool(A,l) -> raise A's flags;
  B's layer l ring -> pool(B,l) -> raise B's.  B's pool work runs while A's GPU executes — the
  overlap F-223 said was impossible with one window.  Single service thread PRESERVED (no new races).
- Commit/emit/draft: per-cohort, after that cohort's window completes (the existing post-window code
  parameterized by verifier — batch_step's tail already loops slots; cohort splits the slot set).
- pump()/batch_launch/batch_poll (generate.cpp:10427-10568, verify.cpp:3327-3380) are the MULTI-GPU
  layer-split machinery (n_pipe = stages+1 = 1 here) — DO NOT try to reuse it; S3 is a new loop
  beside batch_step (gated), not a modification of it.

## 2. Implementation sequence (each step boots green before the next)

S3.1 (gate, no behavior change): env STRATA_COHORTS (default 1).  Restructure batch_step's tail into
  `finish_cohort(Verifier&, slots, ...)`; both cohorts call it with the same data at COHORTS=1.
  Gate: 3v3 vs standing = dead even (bit-identical by construction).
S3.2 (second instance, still serial): construct ver2 + cs2_ + capture its layouts at boot when
  COHORTS=2; run the EXISTING single-window loop alternating ver/ver2 per WINDOW (A's window, then
  B's window, serially).  Gate: outputs byte-equal to COHORTS=1 on the same seeds (the graphs are
  the same kernels; only buffer identity differs).  VRAM audit line here: expert_slots delta printed.
S3.3 (the reactor): both graphs in flight, round-robin doorbell service (the wait site is
  verify.cpp:3161-3178's spin, parameterized by verifier; pool call :3188 same).  Gate: 6-conc
  fresh-eyes boot, non-finite count zero, token streams per slot identical to S3.2's log (byte-exact
  per-cohort outputs — same per-layer service order WITHIN a cohort).
S3.4 (the measurement): 3v3 COHORTS=2 vs 1 on the full battery.  The contention number lands here.
  Adopt bar: conc +4% min, M1/pair no-regress, M3b in-band (serial lane untouched), MMLU only if
  any non-identical bits surface (none expected: same kernels, same order per cohort).

## 3. Budgets, risks, rollback

- VRAM: second verifier's activations+staging+graph execs.  Standing free 1,383 MiB — if the
  expert_slots drop >2%, reclaim from conversation-cache-mib (49,152, tunable) not from the cache
  policy.  Measure at S3.2, decide before S3.3.
- Adapt tier (generate.cpp:10340) and route-resident counters are GLOBAL: the adapt join already
  runs off-window; verify both cohorts' windows see consistent tier state (the tier is monotone
  between windows — same guarantee as today's sequential windows).
- Doorbell starvation: A's pool must not starve B's rings — the round-robin checks both rings each
  iteration (spin with _mm_pause, poll flag A then flag B).
- PCIe share: pcie_mode staging is per-verifier; the DMA copy stream stays shared — stage_batch
  serializes on it (acceptable: PCIe is not the bottleneck at 2.3% misses).
- Rollback: STRATA_COHORTS=1 (default) is exactly today's path; the launchers change only on
  adoption.

## 4. Predictions to beat

- Pilot (S3.3 first boot): window period < 20 ms at 6-conc (from 22.32), GPU busy > 85%.
- Adoption (S3.4): conc >= 270 (stretch 300+), pair/M1 within noise of standing.
- If contention eats the win (period >= 21 ms with both graphs in flight): the fallback insight is
  still valuable — it prices SM headroom for every future overlap idea; journal and close.

## 5. Do NOT do in S3

- No batch_groups/G2 revival (closed F-223), no layer-split pump reuse, no third cohort (rows cap 8
  makes 3 cohorts sub-window-width), no service-thread fan-out (the single thread is the safety).
