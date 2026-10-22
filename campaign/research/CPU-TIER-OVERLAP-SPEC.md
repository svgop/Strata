# THE CPU-TIER OVERLAP — implementation spec (the last standing lever)

Owner measurements: F-125 (the term), F-126 (DMA refuted), F-127 (slots refuted).
Target: hide the per-layer CPU-expert time behind the GPU's own work.
Prize, already measured: F-125's arithmetic — removing the term from the T 4.38 window leaves
15.6 ms for 4.31 tokens/window = **276 t/s** on the operator's real class (the campaign's record class).

## 1. THE SERIALIZATION, EXACTLY

`src/core/session.cpp:960-969`, the per-layer loop of the token graph:

```cpp
// ... the loop above spins until the GPU rings seq == want for this layer, timed into tg.ms_wait
const auto t1 = Clock::now();
progress_at("token: the CPU experts of layer", l);
if (pool != nullptr)
    pool(user, s.db->h_x_f, s.db->h_ids, s.db->h_weights, g.n_embd, s.k, y_miss_host);   // BLOCKING
std::atomic_thread_fence(std::memory_order_seq_cst);
_mm_sfence();
*flag = want;                       // releases the GPU past this layer's wait
const auto t2 = Clock::now();
tg.ms_wait += (t1 - t0);            // waiting for the GPU to ring
tg.ms_pool += (t2 - t1);            // INSIDE THE POOL - the CPU expert compute
```

Per layer the order is **ring(l) -> pool(l) -> release -> ring(l+1) -> pool(l+1) -> release ...**, and
`include/strata/core/session.hpp:222-227` states the consequence in the engine's own words: *"the GPU has
nothing queued and is IDLE for all of this unless `pre[l]`'s tail still covers it."* That idle time is
the `CPU` term in the decode-timing line. Measured on the operator's class: **3.0-4.7 ms/window at T~2.5,
7.1-12.3 ms/window at T~4.0, 13.56 ms of a 29.12 ms window at T 4.38** (~0.28 ms per layer, ~182 CPU
experts per window at ~75 us each: a 1.94 MB expert read from RAM).

## 2. WHAT TO BUILD

**Pipeline the CPU tier one layer ahead.** Replace the single synchronous `pool(l)` with an
issue/wait pair, and start `pool(l+1)` before the host waits for ring(l+1):

```cpp
pool_issue(l);                 // resolves the plan for layer l and hands the work to the worker(s)
...                            // the host now does something else: wait for ring(l+1), stage, launch
pool_wait(l);                  // joins, then the fence + *flag = want release as today
```

Three shapes, in increasing order of work, all preserving the existing per-layer plan semantics:

1. **Two-deep pipeline (cheapest).** Keep one pool call in flight; issue `pool(l+1)` immediately after
   ring(l) returns (its plan is known from the layer-l doorbell read that the host already performs), then
   `pool_wait(l)`, release, and so on. Hides pool(l) behind the GPU's layer-(l-1) tail and the host's
   staging for l+1.
2. **Worker-pool split.** Give the CPU tier its own thread(s) (the pool already has 15 workers; what is
   missing is *asynchrony at the call site*, not compute). The host thread never computes; it issues and
   waits only at the layer that consumes the result.
3. **Look-ahead admission (largest).** Use the router look-ahead (F-63's GO item; `STRATA_PREDICT_CREDIT`
   / `STRATA_PREDICT_BUDGET` already exist for predicted admissions) to issue `pool(l+1)` *before* ring(l)
   returns, converting the whole CPU tier into prefetch.

## 3. CORRECTNESS INVARIANTS (do not break these)

- The CPU expert results for layer l must be complete and fenced (`_mm_sfence()`, then the release store
  to `*flag`) **before** the GPU's layer-l combine can read `y_miss_host`. The issue/wait split must keep
  that ordering exactly; only the *start* of the work moves.
- Tokens must stay identical by construction for the same configuration (this is a scheduling change, not
  an arithmetic one). Gate with the fingerprint/parity harness the engine already carries.
- The plan for layer l must not be mutated by the pipelined call: copy the per-layer plan inputs
  (ids/weights slices) into the issue call's own storage if the doorbell buffer is reused per layer.
- No change to `pcie_frac` semantics (F-126), no change to routing (F-125), no VRAM spend (F-127).

## 4. THE GATES (Proof Law; both classes; same window; v1 anchor beside it)

1. **Instrument first, for free:** `ms_pool` must fall while `ms_wait` holds. Read it from the
   decode-timing line (`pool + plan`, and the `[plan actq jobs CPU]` breakdown) — the instrument is
   already in the shared launcher.
2. **M1:** the four-class draft-yield sweep (`campaign/draft-yield-sweep.ps1`) vs the same sweep on the
   standing recipe in the same window. Adoption bar: M1 up, no class down.
3. **M2:** pair + 6-conc (`campaign/probe-lane.ps1`); must be at or above 176 t/s aggregate (F-124).
4. **M3-M5:** acceptance, MMLU-Pro 70q, mimic/compliance in-class. A scheduling change should move none
   of them; a move means the overlap broke an ordering.
5. **Parity:** tokens identical for the same seed/config against the standing build.

## 5. PREDICTED RESULT AND THE HONEST RISK

Predicted: `ms_pool` (the CPU term) largely disappears into `ms_wait`; the window at T 4.0 falls from
~21-28 ms toward ~12-15 ms; M1 from 184 to the 230-280 class. **Risk:** if the GPU's layer-l tail is
*shorter* than the CPU work for layer l, the overlap hides less than hoped, and the real limit is the
per-layer CPU time itself — in that case the answer is fewer CPU experts (routing/residency, both
already refuted) or a faster CPU path (AVX-512/T-MAC work, F-1's parked item). The measurement that
separates these two: the ratio `ms_pool / (ms_wait / n_layers)` after the pipeline lands.

## 6. CORRECTION (2026-10-12 ~18:05, round 8) — WHERE THE WAIT ACTUALLY SITS

Shape (1) as first written was wrong, and the reason matters: `pool(l)` cannot start before the ring for
layer l, because BOTH its inputs arrive with that ring — the plan (ids/weights/which experts missed, read
from the doorbell) and the activations it computes on (`s.db->h_x_f`). So "pipeline pool(l+1) before
ring(l+1)" is not implementable, and prediction does not rescue it (predicting the ids still leaves the
activations unknown until the layer's own attention has run).

What the evidence *does* support is a reordering, and the engine's own stage list names the place. The
per-layer stages are: `... router -> shared+quant -> **waitA** -> VRAM hits -> ... -> copy+combine`
(`STRATA_VERIFY_PROFILE`'s list, and `waitA`/`waitB`/`waitCPU` are the priced doorbell waits). The
resident-expert grouped GEMM - **"VRAM hits", 20-38 per layer-window in the measurements** - runs AFTER
`waitA`. The resident GEMM does not need the CPU results: it reads the same layer activations and its own
resident weights. Only the **combine** needs both halves.

**So the corrected build, in one sentence:** move the spin (`wait_flag_ge_kernel`,
`src/kernels/cuda/verify_kernels.cu:1051`, launched as a `<<<1,1>>>` node) to sit *immediately before the
combine* instead of before the resident grouped kernels, so the GPU computes the layer's resident experts
while the host computes the layer's missed ones, and only the sum waits for both.

- Expected: `CPU(T)` (measured 4.72·T − 6.8 ms) disappears into the resident-GEMM time, which is already
  on the GPU's critical path. The prize stays F-125's: **276 t/s** at the current window shape.
- Binding constraint to check **before** building: the resident GEMM's duration per layer vs the CPU work
  per layer (measured ~0.28 ms/layer at T 4.38). If the resident GEMM is shorter, the overlap hides only
  part of it - the ratio after the change (`ms_pool` vs the layer's grouped-GEMM time) is the measurement.
- Everything else in sections 3-5 (invariants, gates, risk) stands unchanged; the invariants get stricter,
  not looser: the combine must still see CPU results fenced and complete, and tokens must stay identical.

## 7. THE BOUND, DERIVED FROM NUMBERS ALREADY IN HAND (2026-10-12 ~18:15, round 9)

The build decision needs one number the spec asked for: the per-layer resident GEMM time against the
per-layer CPU work. The logged decode-timing lines give the counts, and the arithmetic is decisive about
the *direction* even where it cannot yet give the exact split.

Measured per layer-window (decode-timing, `per layer-window:` field), at two window depths:

| window | CPU experts / layer | VRAM hits / layer-window | CPU term / window | => CPU term / layer |
|---|---|---|---|---|
| T 3.79 | 2.52 (2.96 entries) | 39.93 | 8.63 ms | 0.180 ms |
| T 4.31 | 3.79 (4.42 entries) | 37.86 | 13.56 ms | 0.283 ms |

So the host does ~2.5-3.8 CPU expert computations per layer (0.07-0.11 ms each, ~1.94 MB from RAM) while
the GPU does ~38-40 resident expert computations per layer in the same layer - ten routed experts per row
times the rows in the window, of which ~91-94% are resident.

**THE DIRECTION IS PROVABLE WITHOUT MEASURING.** Let `P` be the GPU's per-layer work before the grouped
kernels, `R` the resident grouped-GEMM time, `C` the CPU expert time.
- Today: `P + C + R + combine` (the spin sits before the resident GEMM: `router -> shared+quant -> waitA ->
  VRAM hits`, so the GPU idles through `C`).
- After moving the spin to just before the combine: `max(P + R, P + C) + combine`.
- If `C <= R` the CPU term is fully hidden and the window gains exactly `C`. If `C > R` the result is
  `P + C + combine`, which is still **strictly less than today's `P + C + R + combine`** by `R`.

**The reordering can therefore never make a window slower** - it is monotonically non-worse, and it is a
scheduling change only (no arithmetic, no routing, no VRAM), so tokens stay identical. That is the
strongest case any lever in this campaign has had, and it is why this one is worth the build even though
the exact hidden fraction is not yet known: the floor of the win is `R` per layer (the resident GEMM time),
the ceiling is `C` (the whole CPU term, 4.72·T - 6.8 ms per window = the measured 276 t/s at T 4.38).

Correctness is what can still go wrong, not speed: the combine must see the CPU results fenced and
complete, and the resident GEMM must not read anything the CPU half is supposed to have produced (it does
not - it reads the same layer activations and its own resident weights). Gate it exactly as section 4
says, and read `ms_pool` and the window time from the decode-timing line in the first boot.

## 8. SECOND CORRECTION (2026-10-12 ~18:25, round 10) — THE PREMISE OF SECTIONS 6-7 IS WRONG; DO NOT BUILD IT AS WRITTEN

Sections 6 and 7 assumed one monolithic spin sitting before the resident-expert GEMM. **The code says
otherwise.** The real call sites in `src/core/verify.cpp` are per-group and already staged:

```
1596 / 1602   wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs) / wait_flag_ge(m_flagA_, ring, cs)
              -> flag A: the pool published this GROUP's GPU plan (with a skip predicate)
1609 / 1610   wait_flag_ge_or(m_flagB_, ring, skip_ + grp, cs) / wait_flag_ge(m_flagB_, ring, cs)
              -> flag B: the PCIe share is in staging (DMA) or already mapped
1623 / 1628   wait_flag_ge_or(m_flag_, ring, skip_ + grp, cs) / if (!fuse_rows) wait_flag_ge(m_flag_, ring, cs)
              -> the main flag: the CPU's share is in the mapped rows - and NOTE the guard:
                 when rows are fused, THIS WAIT IS NOT ISSUED AT ALL
```

So the engine already separates the plan wait, the staging/PCIe wait and the CPU-rows wait per expert
group, already skips a wait when the data is known present (`_or` with `skip_ + grp`), and already has a
path that removes the CPU wait outright (`fuse_rows`). There is no single spin to relocate: the ordering
F-131/F-132 proposed is, in spirit, already implemented, and the measured CPU term is the residual this
design does not hide.

**WHAT THE NEXT SESSION MUST DO INSTEAD - read first, build second:**
1. Read `src/core/verify.cpp:1560-1660` in full and map the actual per-group sequence: which grouped
   kernels run between flag B's wait and flag's wait, and what the `_or` skip predicate tests.
2. Find what sets `fuse_rows` and whether it can be armed on this pack/config. It is the one place in the
   engine that already deletes the CPU wait - the cheapest possible win if it is reachable, and a measured
   question, not a guess.
3. Only then decide between: (a) moving the *combine* later so the resident GEMM overlaps the CPU rows,
   (b) widening the `_or` skip predicate's coverage, (c) arming `fuse_rows`.
The prize is unchanged (276 t/s at T 4.38 measured, the CPU term = 4.72·T - 6.8 ms/window), and the
risk-profile proof of section 7 no longer applies to a reordering that turns out to be unnecessary.

## 9. THE ANSWER TO SECTION 8'S QUESTION: `STRATA_FUSE_WAIT` (2026-10-12 ~18:35, round 11)

`fuse_rows` is armed by `fuse_wait_on()`, and that function reads an environment switch. From
`src/core/verify.cpp`:

```
227-232   // gfx1201 graph whatever it does; measured on gfx12 only, so it is on there by default.
          // STRATA_FUSE_WAIT=0 / 1 forces it
          bool fuse_wait_on() { ... const char* v = std::getenv("STRATA_FUSE_WAIT"); ... }
1599      if (fuse_wait_on() && !prof_on_) {   // 143-amdk: one launch for the wait and the copy
1627      const bool fuse_rows = fuse_wait_on() && !prof_on_ && !remote_opt_ && dec_batch;   // 143-amdk
1632      else if (fuse_rows)   // the wait folded into the copy's blocks
```

So the engine ALREADY CONTAINS the fix this spec was written to propose: with the switch on, the wait is
folded into the copy launch and the CPU wait stops being a separate spin - it is folded into the copy's
blocks. Upstream's comment states it was **measured on gfx12 (AMD RDNA4) only and is on by default
there**; this machine is CUDA sm_120, where it has never been measured. Nothing in the standing recipe
sets it.

**THE TEST (cheap, one boot, no build):** boot the standing recipe with `STRATA_FUSE_WAIT=1` and run the
four-class sweep plus the concurrent probe. Read `ms_pool` / the `CPU` term and the window time from the
decode-timing line - that is the whole measurement.
- Preconditions, all satisfied in the standing recipe: `!prof_on_` (only `STRATA_VERIFY_PROFILE` sets it,
  and the launcher sets SPLIT_TIMING/DECODE_TIMING instead), `!remote_opt_` (no `--remote-expert-opt`),
  `dec_batch` (`STRATA_DEC_BATCH` unset = on).
- If the CPU term falls as predicted, the prize is F-125's: 276 t/s at T 4.38.
- If it falls but the window does not improve, the term was already hidden and section 8's read stands.
- Correctness gate first (section 4): tokens identical, then M1/M2, then MMLU/mimic in-class.
**This is a hypothesis with an exact arm, not a measured result.** It is the first test of the next round.

**SECTION 9 VERDICT (F-135): MEASURED AND REJECTED.** STRATA_FUSE_WAIT=1 halves throughput on sm_120 (80-90 vs 141-184 t/s) and more than doubles the GPU-reach wait (8.4-10.4 -> 20.7-24.6 ms) while the CPU term does not fall. The gfx12-only default does not transfer. Remaining shapes: the per-group _or skip predicate and the wider reordering.

