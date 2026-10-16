# GDN slot-promotion implementation plan (F-216, green-lit 2026-10-16)

Goal: eliminate the commit-time GDN recurrence replay. Our `commit(n_keep)` re-runs the
accepted prefix through `gdn_step_norm_multi`'s commit mode (verify.hpp:14-19); F-71 measured
the window marginal at 2.47 ms/row, which is why depths 4-6 never pay. SGLang's design
(private state slot per draft column, promote by pointer/copy swap on acceptance) removes the
replay class entirely.

## Why bit-parity holds BY CONSTRUCTION

The verify window already runs the recurrence over all T columns from the untouched state
(`gdn_step_norm_multi(state, ..., n_keep=nullptr)` — "state is read and NOT written",
verify_kernels.hpp:41-45). The kernel keeps the evolving state in REGISTERS
(`s[RPG]` per thread, verify_kernels.cu:286-370) and discards it at window end; commit then
replays tokens [0,m) through the SAME kernel with `n_keep=m`. Per-token arithmetic is
identical in both modes (same h/gate/beta/z rows, same FMA sequence), and acceptance is a
PREFIX — so the register state after token t-1 in verify mode is bit-identical to the state
commit mode would produce for n_keep=t. Persisting it at token boundaries (snapshot t =
state after column t) therefore yields states bit-equal to the replay's, without the replay.

## The change

1. **Kernel** (src/kernels/cuda/verify_kernels.cu): `gdn_step_norm_multi_kernel` gains
   `float* snap` (+ `int64_t snap_tok_stride`). After each token's state update in verify
   mode, every thread writes its RPG floats to
   `snap + t*snap_tok_stride + ((rg*RPG)*h_v + head)*S + col`. Null snap (default) = zero
   behavior change; the write is one extra branch + RPG stores per token. Wrapper
   `gdn_step_norm_multi(...)` grows the same defaulted params (verify_kernels.hpp).
2. **Arena** (src/core/verify.cpp): when `STRATA_GDN_SLOTS=1|2` and the window is
   single-slot (NOT `batch_rec_`), allocate at INIT (F-91: no alloc during capture):
   `T_max(=kVerifyMaxT=8) × gdn_rec_floats × n_gdn_layers` per session slot. ~8×~2MB×36
   ≈ 576 MB single-stream. Batch/concurrent keeps the replay path (3 GB at 6 slots is not
   worth it now; conc has its own profile).
3. **Window path**: pass `snap = arena + window's session slot` on the verify-mode
   recurrence call (graph-captured — the snapshot stores become graph nodes; the arena
   pointer is capture-stable).
4. **Commit path** (verify.cpp:2060 area): when slots on, SKIP the recurrence part of
   `gdn_commit_layers`; issue `gdn_state_promote(state, snap, n_keep, stride, floats)` —
   a copy kernel that reads `*n_keep` (device int, graph-reread at replay per graph.hpp:18)
   and copies snapshot[m-1] → state. KEEP: conv commit (`gdn_conv_commit` — cheap shifts),
   indexer key-tail restore, PLE snapshot (unchanged semantics).
5. **Env knob**: `STRATA_GDN_SLOTS` — 0 = off (default, standing behavior byte-identical);
   1 = on; 2 = on + parity audit (first 64 windows: run the old replay commit anyway into
   the state, then compare snapshot[m-1] vs state with a max-abs counter, log; expect exact
   0.0 — any nonzero is a hard error class, retract and diagnose).

## Validation protocol (the campaign laws)

- Phase 1 (this commit): kernel + wrapper change only, snap=null everywhere; build green;
  noop boot with the flag unset = Launch-Law-identical behavior.
- Phase 2: wiring (arena + window pass + promote + parity kernel).
- Phase 3: boot with STRATA_GDN_SLOTS=2 → engine log must show parity 0.0 across windows +
  M3b canary in band (the state only affects FUTURE windows; token outputs in-window are
  unchanged, so M3b must be exactly in band).
- Phase 4: 3v3 ABAB vs standing ctl (means-only; adopt bar +4%; M3b must not sag).
- Only after adoption: revisit the depth axis (spec 7-8 at kVerifyMaxT=8) under the cheaper
  commit — the stair's economics were marginal-cost-bound (F-71: "the depth axis re-opens
  only with a better head or exactness-coupled dynamic-T"; this is the third door: cheaper
  rows).

## Risks / interactions on the record

- F-196 commit-parity double buffer: the promote kernel runs inside the per-parity commit
  graphs; snapshot arena is parity-agnostic (windows alternate parities — snapshot is
  overwritten each window before its commit reads it; the promote happens before the next
  window's recurrence reuses the arena — sequence is window→promote→next window, enforced by
  graph ordering on the same stream).
- The STRATA_GDN_SPLIT=1 variant kernel (verify_kernels.cu:375+) and the state-only commit
  split need the same snap treatment or an explicit incompatibility guard (guard first:
  slots mode requires the default kernel; refuse the boot if SPLIT is on).
- MT semantics: snapshots indexed by column t within the window [0, n); n ≤ 8.
- If the engine ever runs `batch_rec_` windows for a single request (batch slots coupled
  drafting — F-184), slots mode must detect and fall back to replay per-window (env check +
  per-window branch, not a boot-time assumption).
