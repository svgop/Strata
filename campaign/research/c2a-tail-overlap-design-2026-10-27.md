# C2a cohort-tail overlap — pre-build design (2026-10-27)

Status: design only; no code changed. Falsifies the naive form of the T47-1 lever and
splits it into one safely-buildable slice and one that needs a drafter rework first.

## The lever, as specified

T47-1 (a): today the reactor loop waits for ALL cohort windows before ANY cohort tail runs
(generate.cpp:12217-12233); reorder so cohort A's `finish` runs the moment A's `batch_poll`
reports done, under B's (and C's) in-flight windows. Priced at conc +8-15% from the tails being
~1/3 of the round wall; combined T47 band +9-18% → 355-385.

## What the tail actually does (generate.cpp:12264-12396)

Per active slot, in order:
1. acceptance chain-compare over `outb` (host reads of the verifier's mapped batch output);
2. `v.commit_slot_prefixes(keep)` — writes the host prefix table, then `cudaGraphLaunch` of the
   parity commit graph on the verifier's own stream (`cs_`, verify.cpp:3995);
3. non-finite check (host reads), BT emit (`printf`), slot state update;
4. per slot with MTP: `cudaMemcpyAsync` of the final R rows (generate.cpp:12353), then
   `draft_launch` (graph launch on the slot's own stream) and `draft_finish`.

## The hazard, named

`draft_finish` (mtp.cpp:1922-1936) spins on the draft output and then calls
`cudaStreamSynchronize(cs_)` — a **blocking host wait on GPU progress**. Running it while another
cohort's window is in flight is the issue-#31 deadlock class: the GPU window spins on a doorbell
flag that only the host reactor thread can raise, and that thread is now parked in a driver wait.
The `cudaMemcpyAsync` at :12353 is the same class the code comment at verify.cpp:9437 already
warns about ("the host stuck in that cudaMemcpyAsync on a driver lock for good").

This is almost certainly the F-285a hang mechanism: any "C2a-lite" that relocates the whole tail
into the poll loop puts a blocking stream sync and a host D2D copy inside every other cohort's
window. The relocation was not the bug; the blocking calls in the overlapped region were.

T47 §S5c states the same rule ("keep every driver-locking call out of the overlapped region")
but does not name the two call sites. Both are named above.

## The split

**Slice 1 (safe, buildable now) — head only.** Overlap `finish_head(ver, wa)` = acceptance walk +
`commit_slot_prefixes` + emit + slot state, for cohort A, as soon as A's poll returns 0 while B
still runs. Keep the per-slot draft block (memcpy + launch + finish) at the round end, where it is
today. Requirements for the head: `keep[]` and the slot set must persist in `CohortWin` between
the two calls; `commit_slot_prefixes` must run before the draft uses the accepted picks
(it does: drafts are launched from the committed prefix). All head calls are host reads, printf,
and one async graph launch on A's own stream — no blocking driver waits.
Expected band: the head is roughly 0.3-1 ms of a 32.8 ms round; **honest estimate +1-3% conc**,
not the headline band. It is still the right first landing because it exercises the reorder
protocol at low risk.

**Slice 2 (the real band) — non-blocking draft completion.** The draft tail is the expensive part
and it is unsafe only because `draft_finish` blocks. The fix is structural: `draft_launch` returns
without waiting, and the reactor loop polls the slot's draft event alongside the cohort doorbells
(the "port" form T47-1(b) named), so the draft completes while other cohorts progress and the
blocking sync disappears from the overlapped region. That also lets the draft tail overlap across
cohorts. This is a MtpDrafter/engine change with its own parity surface (the trim rule reads the
same output buffer, so the readout is unchanged; only the waiting moves).
Expected band: with the draft hidden, the full tail class overlaps → the T47 +8-15% band becomes
reachable. Build order: slice 1 (protocol, measured), then slice 2 (drafter) as its own ladder arm.

## Landing protocol (mandatory, F-285a law)

1. Same binary for arm-off and arm-on (`STRATA_COHORT_TAIL=1`, default off).
2. Arm-off proof on that binary: full boot + prewarm2 + greedy bit-check + one conc leg at the
   champion level. The default path must be structurally preserved — slice 1 keeps the existing
   tail block as the only call site when the flag is off, and the early call is a single guarded
   `if`.
3. Arm-on: health, cohort rounds clean (no `ERR`, no non-finite), BT line accounting unchanged
   (tokens equal in equals out at temperature 0), then ABAB env-ladder (law 10/11).
4. Hang guard: the ladder's health gate + `stop_server` restores the lane; a hang counts as a
   negative with the last stderr line logged.
5. Rollback: `STRATA_COHORT_TAIL=0` (one launcher line) and the binary ladder's archived champion.

## Boundary

This document prices the lever's mechanism and hazards, not its speedup. Slice 1's band is a
prediction below the headline; slice 2 is unbuilt. No measurement ran for this doc; all costs are
the banked T47/decode-anatomy terms.
