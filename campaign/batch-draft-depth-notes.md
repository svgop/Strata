# Batch draft depth D - builder note (2026-10-26)

Lifts the BATCH lane's speculative depth from the pinned 1 proposal per slot to a fixed
configurable D. Falsifier-first: test fixed D=2 / D=3 against the D=1 baseline before any
adaptive policy. Build-only change; the live server was not touched, nothing deployed.

## Files touched

- `src/core/mtp.cpp` - draft_launch() / draft_finish() generalized (49 lines changed).
- `include/strata/core/mtp.hpp` - the split-path contract comment (6 lines).
- `src/program/generate.cpp` - env knob, set_max_drafts, both collect loops, both accept
  sites, both residual-copy + draft_launch tails, admission draft_first, help text
  (132 lines changed).
- Build: `cmake --build build --target strata -j32` - CLEAN. The 2 warnings on the build
  (fprintf arg-count at the batch summary line, unused `w` at the second summary site) are
  pre-existing: identical count on a stash-build of the base.

## Env knobs

- `STRATA_BATCH_DRAFT_DEPTH` (int, default 1, clamped 1..min(o.spec-1, 3)). Requires
  `--batch-mtp`. D=1 is byte-identical to today's binary. Boot log line when D != 1:
  `strata generate: --batch-mtp: batch draft depth D=<D> (STRATA_BATCH_DRAFT_DEPTH): each
  slot verifies up to <D> proposals, 1 + min(actual, D) rows per slot`.

## Mechanism (what actually changed)

1. **Drafter API (mtp.cpp).** `draft_launch` was hard-pinned to the one-proposal shape
   (`if (max_steps != 1 || min_p <= 0.0f) err`). Now: lazy `capture_step(j)` per extra
   proposal, the step cells staged exactly as draft() stages them
   (`put(max_t_ + j - 1, coupled_draft_cell(p, a, j))`), all `h_out_[j]` armed to -1, and
   every step graph enqueued **back-to-back after the round graph** - the STRATA_DRAFT_UNGATED
   twin's launch shape. Back-to-back is required, not chosen: the gated form spins between
   launches, and the split path exists so every slot's round launches before any of them is
   waited on (F-206); a spin here would re-serialize the slots. `draft_finish` keeps the
   exact draft() trim rule - draft j is kept iff prob[j-1] >= min_p, first proposal always -
   as a pure post-sync readout, and reports the kept count through `*n_drafts` (was hardcoded
   n=1). cp-aware step launches (slot drafters DO get set_draft_sampling at admission,
   generate.cpp:16508).
2. **Depth knob (generate.cpp).** `set_max_drafts(batch_draft_depth)` on every slot drafter
   (was 1). The solo drafter and the pipeline/serial lanes' `set_max_drafts(S_mtp-1)` are
   untouched.
3. **Collect loops (both: cohort ~:12078, single ~:12413).** Per-slot rows =
   `mtp_use && draft_ready ? 1 + min(n_draft, D) : 1`, where `n_draft` is the ACTUAL proposal
   count the last round returned (new BSlot field, filled by draft_launch/draft_first via
   `&sl.n_draft`). The loop's for-condition (today's conservative `S + 2 <= kVerifyMaxT`
   admission) is untouched; at D > 1 a deeper slot that would overflow the 8-row cap runs
   TRUNCATED (`rows_b = kVerifyMaxT - S`, never below 2 given the entry condition) - its
   extra proposals sit that window out. At D=1 the truncation branch is dead code.
4. **Accept sites (both: cohort finish lambda ~:12260, single ~:12497).** Chain compare:
   `acc = 1; while (acc < nrows && outb[i+acc-1] == tok[i+acc]) ++acc; keep[b] = acc;`.
   Room guard generalized (`produced + nrows <= max_new && p + nrows + 1 <= max_context` -
   identical numbers at nrows=2). The rescue stays depth-1: it fires only when acc == 1 and
   only re-rolls proposal 1 (dprob[0] is the only confidence the slot holds). Telemetry
   units unchanged: bt_slot_drafts counts slot-windows with a draft row, bt_slot_accepts
   counts those with acc >= 2; bt_rows / bt_accepted were already keep-generic.
5. **Tails (both).** Residual copy width `nrows[t] * stride` (was `2 * stride`; the
   slot_mtp_rows buffers are o.spec rows each, and 1 + D <= o.spec by the clamp, so they
   fit). `draft_launch(nrows[t], outb + first[t], pos[first[t]], keep[b] - 1, ..., &sl.n_draft)`
   - T is the slot's actual window rows (the old T=2 is the nrows=2 case of this), a =
   keep-1, the same call shape the serial lane has always run through draft(). The 1-row
   stale-draft refresh (draft_first) and the admission-path draft_first (:16509) now pass
   `&sl.n_draft` too.
6. **commit_slot_prefixes** needed nothing: it was already width-generic (verify.cpp groups
   rows by slot, validates n <= group size, one commit graph per layout).

## D=1 bit-identity argument

- `set_max_drafts(1)` unchanged in effect: max_steps = min(max_t_-1, 1, cells) = 1.
- draft_launch at max_steps=1: no capture_step calls, no step-cell puts, `h_out_[0]` init
  only, no step-graph launches, the extra launch_ms accumulation is `max_steps > 1`-guarded.
  Instruction-for-instruction the old path.
- draft_finish at max_steps=1: the trim loop body never runs (`n < pend_max_steps_` fails at
  n=1); n=1, zeroing from 1, `*pend_n_ = 1`, `g_dt.steps += 0`, `g_mp.collect(1)`. The PLE
  prefetch moved into a lambda but runs the same guard, same ngram_rows, same prev-shuffle,
  same single prefetch of drafts[0].
- Collect loops: rows_b = 1 + min(n_draft, 1) = 2 when draft_ready (n_draft >= 1 always on
  success), so the row-expansion loop emits exactly one draft row with tok = draft[0],
  pos = p+1 - the old `if (nrows == 2)` block verbatim. Admission set identical (the
  conservative for-condition is preserved; the truncation branch is D>1-only).
- Accept sites: acc = (match ? 2 : 1); rescue gated on acc == 1 = the old else-branch;
  keep[b] = acc = the old keep=2 in both arms; bt_slot_accepts increments exactly where the
  two old arms did.
- Tails: nrows[t] >= 2 selects the same branch as `== 2`; the copy is 2*stride; draft_launch
  gets T=2, same args, plus a write to `&sl.n_draft` (a BSlot field that nothing reads
  differently at D=1: collect computes 1 + min(n,1) = 2 either way).

So the D=1 boot takes no new branch, launches no new graph, and produces the same keep
decisions, emissions, and K/V commits. Remaining test: a capture-diff/bit-check run
(campaign/bit-check.py) on a fixed prompt at D=1 vs the pre-change binary to prove it, not
just argue it.

## Risks

- **Layout churn (bounded).** Depth multiplies the window row-vector classes
  ({3,3,2}-style truncation mixes on rotation). The existing guard is armed:
  `set_batch_graph_limit(64)` + LRU on every batch verifier when batch_mtp is on
  (generate.cpp:9296 and the cohort twins). First use of each new class pays a capture
  (lazy, mid-serving, on the verifier's own stream - the same precedent as today's rotating
  layouts). Drafter-side: first use of step graph j per slot captures it lazily inside
  draft_launch (precedented: the admission draft_first already captures inside the batch
  loop today).
- **8-row cap vs 6 slots x cohorts.** batch=6, cohorts=2 -> 3 slots per cohort window.
  D=2: 3 full-width slots want 9 rows > 8 -> the deepest slot truncates to 2 rows (window
  = 8 rows, all 3 slots admitted). D=3: {4,4} fills 8 and the third slot is not admitted
  past S=6 (today's admission rule preserved) - depth-3 needs either >= 10-row windows
  (kVerifyMaxT bump, kernel-side) or relaxed admission (a D>1-only follow-up). D=3's
  realistic cohort shape is 2 slots x 4 rows; the single-verifier lane with 6 slots runs
  {4,4} then rotates.
- **draft_finish trim semantics.** The trim now matters: proposals past the first sub-min_p
  prob are still EXECUTED (~50 us each) but not returned, so n_draft varies per round and
  the next window's width varies with it (by design - that is the actual-proposals shape).
  The chain accept ignores dprob; min_p only shapes what the drafter proposes.
- **Coupled sampling in slots.** Slot drafters get set_draft_sampling per request; the
  step graphs now launch cp-aware. The coupled chain's per-step sampling under back-to-back
  enqueue is the same graphs/stream order/Philox draws argument as the greedy path
  (STRATA_DRAFT_UNGATED's proof), but coupled+D>1 in SLOTS is a new combination - smoke it
  with STRATA_SPEC_COUPLED=1 before trusting its acceptance numbers.
- **Near-context-end.** The drafter's max_steps clamp (`st_.max_cells - (p+a)`) can now
  land mid-band; max_steps=0 stays the old fatal error (unchanged exposure). Window pos
  validation (`pos+1 > slot max_cells`) sees draft rows up to p+3; the accept guard
  `p + nrows + 1 <= o.max_context` bounds admission the same way today's does at nrows=2.

## Smoke plan (build only; nothing booted here)

1. **D=1 bit-check (gate):** boot the new binary with --batch-mtp (no depth env) and run
   campaign/bit-check.py on a fixed prompt set vs the pre-change binary. Expect identical
   BT streams and accept telemetry. Any diff = the change is wrong, stop.
2. **D=2 ramp:** STRATA_BATCH_DRAFT_DEPTH=2, 6 concurrent prompts. Capture lines:
   the boot `batch draft depth D=2` line; per-window `b b=... nrows=... keep=...` trace
   (STRATA_BATCH_TRACE if armed) expecting nrows in {1,2,3} and keep <= nrows; the window
   summary's `avg %.2f rows` (expect between 6.0 and 8.0 per cohort window vs today's 6.0);
   `slot drafts accepted %lld of %lld` (the slot-window accept RATE should stay near the
   D=1 baseline - chain acceptance shifts rows/token, not slot-window hit rate);
   `MTP proposals accepted %lld of %lld` (offered count rises ~1.5x at D=2).
3. **Accept-rate sanity:** token-level accept per OFFERED proposal should be >= the D=1
   rate (chain accept only keeps matched prefixes). If token-level accept drops, the trim
   or the chain compare is misaligned - stop and bisect.
4. **D=3:** expect the truncation path in cohort windows (some windows with a 2-row slot
   after two 4-row slots); confirm no `verify: accepted prefix is outside its slot group`
   and no graph-capture errors; watch batch-graph LRU evictions (VRAM reserve).
5. **Adversarial shapes:** single active slot (all depths), rapid slot churn (admission
   draft_first + n_draft freshness), prompts ending mid-window (eos inside a chain keep),
   STRATA_SPEC_COUPLED=1 + D=2, STRATA_BATCH_MTP_MAX_SLOTS=2 + D=2 (the mtp_use flip must
   zero n_draft paths via the stale-draft reset).
6. **Then the falsifier:** fixed D=2 vs D=1 ABAB on the standing pair/conc batteries
   (campaign/batteries), the F-351 harness. Up to 3 rows/slot at D=2 vs 2 today. Adoption
   bar: pair throughput up with pair/conc quality in-noise, per the standing ledger.

## Scope notes

- The expected drafter-API blocker (draft_launch's one-proposal guard + draft_finish's n=1)
  was real but small: ~50 lines in mtp.cpp adopting draft()'s existing step machinery under
  the split path. No scope-out needed; total diff 187 lines across 3 files.
- Not done (deliberate): adaptive depth policy (the falsifier-first discipline - fixed D
  first), kVerifyMaxT bump (kernel-side, follow-up), relaxed D>1 admission past S+2
  (follow-up if D=3 truncation leaves rows on the table).
