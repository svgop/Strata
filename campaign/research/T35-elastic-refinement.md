# T35 — Elastic-session refinement: where the time lives, and the ranked refinements

Date: 2026-10-23. Scope: OUR elastic-session implementation (F-272, commit 4505e56a, +320/−29 in
generate.cpp; adopted F-273 with `STRATA_ELASTIC_SESSIONS=1` in the standing launcher). Task:
latency / smoothness / responsiveness / speed / reliability refinements — no new features; every ms
and sub-ms counted; the terms compound per window. READ-ONLY on code; no boots, no builds.

Evidence base (primary, read directly): the live tree at HEAD fd0780ff (no generate.cpp change
since 4505e56a — `git log 4505e56a..HEAD -- src/program/generate.cpp` is empty, so the F-272 diff
IS the live code); `custom-iq3_s.log` (8 elastic boots, 59 promotes, 48 demotes with per-event
timings); `campaign/results/t34-elastic.log` + the seven t34 leg outputs; vmm.cpp/vmm.hpp;
serve/server.py (pick_slot, BADM-0 park); the ConversationCache parking infrastructure
(conversation_cache.hpp). Secondary: the five T33 dossiers in this directory. External numbers
(vAttention Table 3, Llumnix) are inherited from those dossiers with attribution, not re-fetched.

---

## Summary

The landed implementation is sound and its measured costs are inside the spec's contract, but the
log shows the lifecycle running in its WORST possible mode: the spare pool is empty at every
promote and after every demote (59/59 promotes say "0 chunks left", 48/48 demotes say "0 chunks
spare"), so every promote runs the full eviction release-walk synchronously at the admission and
every demote pays the cache remap + static-profile refill. Meanwhile the demote scan is
line-arrival-driven, so the whole demote bill lands on the TTFT of the first request after an idle
gap (measured: 128 ms for 4 slots; ~200-310 ms class at 9-wide). Three refinements fix the mode,
not the feature: (1) run the idle scan on a timer so demotes happen in true idle (R2); (2) keep one
arena's chunks permanently in the spare (a watermark) so every promote is the backgrounded map and
the churn disappears (R1); (3) park the demoted slot's conversation into the ConversationCache that
already exists instead of dropping it (R3). Beneath everything sits one physical fact: at 2 MiB
granules, one 0.73 GiB arena move = 373 driver calls at a measured 29-51 µs each — coarser
super-chunks (R4) divide the whole lifecycle's API time by ~16. Zero-code items: the default-off
direct slot copy (R6) and `STRATA_ELASTIC_S_MIN=1` (R7).

---

## Measured pathway anatomy (today's numbers, all from custom-iq3_s.log unless noted)

Substrate: arena = 0.73 GiB = 373 VmmChunks of 2 MiB. vmm.cpp does one `cuMemMap` per chunk and
batches `cuMemSetAccess` once per run of newly mapped chunks (so the map is the only per-chunk
call). Measured per-chunk cost 29-51 µs — 25x vAttention's A100 "map 2 µs" (their Table 3 via
T33-priorart-materialization.md H3); this is GSP-driver reality on sm_120 and it re-prices every
"tens of ms" estimate in the T33 dossiers.

**Boot** (line 1576/14358 class): 6 arenas freed to the driver in 34-38 ms (batch 6); 9 arenas,
5.10 GiB, in 60 ms (batch 9); cache then auto-sizes into the freed VRAM: 8,226 slots vs 6,923
standing (+1,303). Host pins: 1.55 GiB total per process at BOTH batch 6 and batch 9 — flat, not
×batch (see contradiction C4; the F-272 journal's "bounded --batch x 1.55 GiB" overstates).

**PROMOTE** (arrival on unmapped slot b; code generate.cpp:11510-11551, 12614-12618, 16064-16070):
- Background start at BGEN receipt requires `es.spare.size() >= chunks_per`. In the observed logs
  this is NEVER true (spare is 0 at every promote), so every promote runs synchronously at the
  admission, after the request's GEN 1 token.
- a. `adapt_quiesce` (cudaDeviceSynchronize + apply_pending + adapt_tick): ~0-2 ms at the admission
  point (the tier just landed; not separately timed).
- b. `kvg_release_until` eviction walk: ~180 cache slots given up (kvg.lo walk; each resident expert
  D2D-moved into a colder loan slot or dropped), one device sync, ~373 cache granules unmapped into
  the spare (6.2-15.3 ms by the demote's own unmap measurement), residency table H2D (<1 ms).
- c. `map_range` 373 chunks + one setAccess run: 10.8-19.3 ms when measured standalone ("from the
  spare pool" cases). NOTE: the printed "map X ms" timer starts BEFORE the release block, so the
  logged promote lines CONFLATE b+c — and the source label always says "the spare pool" because
  `had` is sampled after the release has filled the spare (the "from the expert cache's tail" label
  is dead code in practice). Total promote lines measured 11.3-19.8 ms including the walk.
- d. `session_zero` on the side stream: 0.5-0.6 ms + sync.
- e. `copy_to_slot` (conversation into the slot): default host path = per-stage checkpoint
  save/restore + per-state KV save/restore with several device syncs — the 12.5 ms (76 tokens) to
  ~40-90 ms class (the code's own comment prices the host round trip at 40-90 ms on splits;
  `STRATA_SLOT_COPY_DIRECT=1` device-to-device path exists, default off).
- TTFT pays e plus any part of a-d that the GEN 1 prefill did not already cover.

**DEMOTE** (idle scan, generate.cpp:12565 + 11552-11578; fires on a line arrival when nothing
decodes, after STRATA_ELASTIC_DEMOTE_S=10 s of slot idleness, only for slots >= s_min, never
active/partial):
- a. `adapt_quiesce` (~0 at idle).
- b. unmap 373 chunks: 6.2-15.3 ms measured.
- c. `bs[b] = BSlot{}` — the slot conversation is DROPPED (re-read on return; the slot also stops
  being a `slot_source` candidate for returning chats).
- d. `kvg_cache_eat_spare`: remap 373 chunks into the cache's range (11-19 ms class) + refill from
  the BOOT-STATIC profile: H2D of ~180 experts ≈ 415 MiB (~15-17 ms) + one sync.
- Total measured: 24.8-44.2 ms, and it runs on the REQUEST THREAD of the arrival that triggered the
  scan — i.e., before that request's prompt read even starts. Observed bill: 4 demotes = 128 ms
  (log lines 2063-2066 immediately before the next request); the b9 leg shows the same 4-demote
  129 ms bill between its shorts and conc phases; worst case at 9-wide (7 demotes) ≈ 200-310 ms.

**TTFT p99, 2.0-2.4 s (t34 legs)**: conc-phase arrivals read 18-234 fresh prompt tokens each
(113-351 ms measured), serialized on one request thread; each admission adds promote (11-20 ms) +
copy_to_slot (~40-90 ms class) + first window. The p99 arrival is the last-served one: prefill
queueing + admission copies dominate; the elastic-specific share is ~25-60 ms of ~2,400 ms (<3%)
plus the demote bill when it fires (up to ~130-210 ms on one arrival). The 5865-token fresh-read
TTFT of 1,361 ms shows where the seconds live: full re-reads, which is exactly what demote-without-
park forces on returning chats.

**ITL p99, 35 ms @ 9-conc vs 20 ms @ 6-conc**: consistent with the row-linear window cost
(c_row ≈ 2.4 ms, F-270): +3 rows ≈ +7 ms mean, and the tail is widest-window wall time + doorbell
fetches. No elastic event (promote/demote) appears inside any conc phase in the logs — the 35 ms
is structural to 9-wide windows, not lifecycle machinery.

**Smoothness (the demote transient)**: the release walk strips residency from ~180 loan-region
experts; the refill places by BOOT-STATIC rank, not live heat. The request immediately after the
cycle shows it: "decode expert cache hit rate: 93.1% (38,655/41,539)" and "decode waited 56.8 ms in
332 fetches" (log line ~2053), vs 99%+ hit / 0.1-0.5 ms waits on its neighbours. The promote/demote
cycle is therefore a hit-rate saw-tooth, and the F1 solo bracketing's low end (211.5 vs OFF 234.3)
is its plausible signature — bracketed, not per-phase attributed (open question Q3).

**Spare-pool dynamics**: the spare oscillates 0 → 373 → 0 every cycle (maximal churn): every
promote pays the eviction walk; every demote pays the refill; a re-promote of the SAME slot pays
both directions plus the expert-heat loss. Starvation is guarded (cache floor 128 slots; BADM-0
retries bounded at 6 with a 2 s slot park, server.py:1600-1611; zero "could not be materialized"
lines in the log). `give_way` correctly refuses unmapped slots and partial reads are demote-guarded.

---

## Ranked refinements

| # | Refinement | Mechanism | Expected gain (arithmetic) | Risk | Cheapest falsifier | Code? |
|---|---|---|---|---|---|---|
| 1 | **Timed-wait idle scan** — demotes run in true idle, not at the arrival | `next_line`'s cv-wait gains a deadline = earliest `idle_since`+`demote_s`; on wake, re-check `!batch_on() && !pipe_inflight()`, run `es_idle_scan`, wait again (~20-30 lines) | Removes the measured 128 ms (4-slot) / ~200-310 ms (7-slot worst) demote bill from the post-idle arrival's TTFT; every demote event currently at 10 s after idle stops being a latency event | Lowest — the scan's own gates re-checked at wake; demote at true idle is the safest possible moment (this is exactly where the scan already runs, just attached to the wrong wakeup) | Log-only: the 4 demote lines at 2063-2066 sit between two requests' lines — the bill is already demonstrated; a timed-wait boot re-run of the F3 shorts+conc pattern shows TTFT p99 without it | code (small) |
| 2 | **Spare watermark of one arena** — boot pre-fills the spare with one arena's chunks; demote eats only the excess above `chunks_per` | Boot release loop routes ONE arena to `es.spare` instead of `vmm_chunk_free` (+5 lines); `es_demote` eats `spare − chunks_per` instead of all (~10 lines). Invariant: the spare always holds exactly one promote's worth | Every promote becomes the backgrounded map (11-19 ms, hidden under the arrival's GEN 1 prefill — any prompt ≥ ~20 ms of read covers it): the synchronous release walk + eviction leave the TTFT path for ALL promotes, not just post-demote ones. Demote in the common (single-slot) cycle drops to unmap-only: 28-40 ms → 6-10 ms. The boot placement survives the first promote (no eviction of freshly-placed experts) | One arena (0.73 GiB ≈ 320 of 8,226 cache slots ≈ 3.9%) held from the cache permanently while elastic is on. F-273 arithmetic: +1,300 slots → +4.0% conc, so −320 ≈ −1.0% conc UPPER bound; likely ~0 at the F-254 saturation point — but it is a real trade against the adopted +4.0% | Zero-code approximation first: `STRATA_ELASTIC_S_MIN=1` boot (equivalent VRAM reservation, one more mapped slot instead of a warm spare) + one F3 battery round — reads the cache-size cost directly. Then the ~15-line watermark behind an env, F3 again | code (small) |
| 3 | **`STRATA_SLOT_COPY_DIRECT=1` on the standing recipe** | copy_to_slot's device-to-device path (one sync pair) instead of the default per-stage checkpoint host path with its multiple syncs | −20-60 ms class off EVERY batch admission's TTFT (the default path's checkpoint+sync class is 12-63 ms measured in the park/restore lines; the code prices the host round trip at 40-90 ms). At the 9-conc ramp, admissions serialize behind windows, so the tail gains the most | It is default-off for a reason — unmeasured on this recipe; the `failed()` drain note hints at past flakiness. Byte-identity must be re-verified (golden hash) | One F3 leg with the env set: TTFT histograms + golden hash + the engine log's admission timing. No build | ZERO code |
| 4 | **32 MiB session super-chunks** | `vmm_chunk_new/granularity` parameterized per citizen: arenas reserve/map/unmap in 32 MiB physical chunks (23 calls/arena instead of 373); per-call cost is size-independent GSP RPC (measured 29-51 µs at 2 MiB; vAttention's table agrees per-call cost does not scale with size) | Promote map 11-19 ms → ~1-2 ms (×4 conservative if 32 MiB maps cost 2-4x a 2 MiB map); demote unmap 6-15 ms → ~0.5-1 ms; boot free 34-60 ms → ~5 ms; the eat's remap shrinks identically. Both lifecycle directions fall below the GEN 1 prefill floor even on the synchronous path | The cache can no longer eat super-chunks at granule granularity: the eat path must convert (free the super-chunk, create granules — same API cost, deferred to idle) or map 16-granule runs (eviction quantizes to ~14-slot steps; floor 128 still workable). Pool split by size class needs tag discipline | CUDA-API microbench off-line (no engine): time 23 × cuMemMap(32 MiB) vs 373 × cuMemMap(2 MiB) on this driver; if the per-call cost is flat, the win is confirmed before any tree change | code (medium) |
| 5 | **Park-on-demote through the existing ConversationCache** | `es_demote` saves the slot's state D2H (`conversation_checkpoint_save` + `conversation_kv_save` — the copy_from_slot machinery) into `conversations` (48 GiB budget, LRU, pinned) instead of `bs[b] = BSlot{}`; the return path restores H2D (ideally straight into the promoted slot) | Returning-chat TTFT after idle demotion: measured full re-read 1,361 ms @ 5,865 tokens vs park D2H 63 ms + restore H2D ~60 ms class → ~1.2 s off the p99 FOR CHAT-RETURN PATTERNS. The single biggest absolute item for real multi-chat serving; invisible to the t34 battery's pattern | Demote grows by the D2H park time at idle (fine after #1); RAM budget competition with the main path's parks; the server's `slot_held` goes stale on demote today (pick_slot prefers a prefix the slot no longer holds — correctness is safe, the engine re-reads, but the heuristic mis-fires) and should be cleared by the same change | Today's logs already carry both numbers (1,361 ms re-read vs 63 ms park); a two-turn chat probe with a >10 s gap between turns, demoted vs s_min-held, is one boot + one script | code (medium) |
| 6 | **`STRATA_ELASTIC_S_MIN=1`** | One fewer always-mapped arena → +0.73 GiB to the boot cache (+~320 slots ≈ +3.9% pool) | Conc pool arithmetic: ~+0.7-1.0% conc class upper bound (F-273: +1,300 slots → +4.0%), M1 neutral-to-tiny-positive; pair arrivals pay slot 1's promote (hidden under GEN 1 when the spare is full — without #2, the FIRST promote after boot still runs the walk) | The pair lane's second-slot admission latency is the watch item; the F-271 solo non-negotiable is untouched (slot 0 stays mapped) | One F3 battery round with the env set — the env already exists | ZERO code |
| 7 | **Live-heat refill on demote** (tranche-2 item, scoped) | `kvg_cache_eat_spare` fills from the boot-static `profile`; refill instead by the in-memory learned ranking (`rank_learned_profile(heat)` at demote time — what the just-served workload actually routed), keeping the resident-set check | Shrinks the demote-adjacent hit-rate dip (measured 93.1% vs 99%+; "decode waited 56.8 ms in 332 fetches") and shortens the M1 relearn transient; worth ~0.5-1% M1 in flip-flop-heavy phases, 0 in single-leg runs. NOTE: with #2 in place the common cycle never refills at all — this item's value DROPS to the floor-case and excess-eat paths | F-271 hygiene: the learned FILE poisons; this uses the running boot's in-memory heat, which is the safe form — but batteries must still control the profile (the law stands) | The per-request cache-hit lines around demotes in custom-iq3_s.log give the baseline dip for free; one env-gated boot re-running the F3 short pattern reads the delta | code (small) |
| 8 | **Honest promote telemetry** | Split `map_ms` into release-walk vs map (the timer starts before the release block today) and print the true source (the "the expert cache's tail" label is dead: `had` is sampled after the release fills the spare) — ~6 lines | Not a speed item: it makes every future tuning decision (including #2's before/after) readable from the log instead of inferred from an internal contradiction (this dossier had to reconstruct the promote anatomy from the demote's numbers) | None | None — it is a defect: all 59 promotes claim "from the spare pool" while every demote leaves "0 chunks spare" | code (trivial) |

---

## The three highest-value items, argued

**#1 Timed-wait idle scan.** This is pure removal: no VRAM trade, no semantics change, and it
deletes the single largest elastic-attributable latency event in the logs. The scan's placement is
a scheduling accident — it runs when a LINE ARRIVES because that is the only wakeup the main loop
had, and the code comments acknowledge the laziness ("a fully idle engine demotes lazily at its
next request"). The consequence is that a request pays for capacity it does not use: 128 ms
measured (4 demotes), scaling to ~200-310 ms at 9-wide, attached to the first arrival after every
≥10 s quiet gap — precisely the burst-onset request whose TTFT a human notices. The main loop
already blocks in a condition variable (`next_line`, generate.cpp:11133); a timed variant that
wakes at the earliest demote deadline turns every demote into a true-idle operation, which is also
the safest context the lifecycle has (no windows, no admission, quiesce trivially satisfied). It
composes with everything else: #2's background promote assumes the spare is warm, which demotes-at-
idle make reliable; #5's park D2H belongs in true idle by the same argument.

**#2 Spare watermark of one arena.** The log's most damning steady-state fact is that the spare is
EMPTY at every promote and after every demote — 59/59 and 48/48. The lifecycle was designed so the
common case is "spare full → background map"; the observed mode is "spare empty → synchronous
eviction walk at the admission", the exact opposite. One invariant fixes it: the spare always holds
`chunks_per` chunks. Boot routes one arena's chunks to the spare instead of the driver (the cache
grows into 2.18 GiB instead of 2.91: +880 slots instead of +1,303 — the trade is 3.9% of the pool);
demote eats only the excess above the watermark. The arithmetic per event: promote goes from
[quiesce + ~180-slot eviction walk + 373 unmaps + 373 maps ≈ 11-20 ms synchronous] to [map only,
11-19 ms, on `es_bg`, hidden under the arrival's GEN 1 prefill — net TTFT ≈ 0 for any real
prompt]; demote goes from [unmap + remap + ~415 MiB H2D refill = 28-40 ms] to [unmap = 6-10 ms].
And the churn — the eviction of experts the cache just placed, the static-rank refill, the 93.1%
hit-rate dip — simply stops happening in the common cycle, because the common cycle stops touching
the cache at all. The Llumnix framing from T33-serving-delta-notes.md is exact: this is the
state-unset-inside-band hysteresis, with the band floor = one arena of warm chunks; the floor-case
eviction walk remains as the pressure valve, exactly as their freeness band keeps a threshold path.
The risk is one number: −320 cache slots against the +1,303 that won F3's conc lane. The honest
expectation is a sub-1% conc giveback, possibly zero at saturation — but it must be measured, which
is why #6 (the same VRAM spent as a mapped slot instead) is the zero-code proxy for the trade.

**#5 Park-on-demote.** This is the only refinement that moves seconds. Everything else in this
dossier fights over tens of milliseconds; the demote's `bs[b] = BSlot{}` converts every returning
chat after an idle gap into a FULL re-read — 1,361 ms at 5,865 tokens measured, and the sky is the
limit (the 131k context exists). The engine already owns the complete alternative: the
ConversationCache parks main-path conversations to host RAM every request end (42-63 ms measured
D2H, budgeted at 48 GiB, LRU + pinned, prefix-matching on return) — the demote path just does not
use it. Parking a slot conversation is the same two calls `copy_from_slot` already makes, at the
safest moment (true idle, after #1), for the cost of the same tens of ms the main path already
pays every request. The p99 of a multi-chat workload with think-time gaps — which is what this
engine serves — is dominated by exactly these re-reads. The wiring caveats are real but bounded:
restore directly into the promoted slot (one H2D) rather than park→main→slot (two copies); clear
the server's `slot_held` for the demoted slot so `pick_slot`'s prefix heuristic stops preferring a
ghost (today it is merely slow, not wrong — the engine re-reads what the slot no longer holds).

---

## Contradictions with the T33 dossiers (and within the journal)

- **C1 — the vAttention per-call costs do not transfer to this driver.** T33-priorart-
  materialization.md H3 (vAttention Table 3, A100): 2 MiB map 2 µs, setAccess 38 µs, "0.78 GiB via
  fresh VMM mapping ≈ 20-30 ms". Measured here (sm_120, GSP driver, setAccess already batched per
  run): 29-51 µs PER MAP — the map, not the access, is the cost. The materialization dossier's
  amendment-4 arithmetic ("re-grow ≈ 25-30 ms per 0.78 GiB") landed in the right range by luck of
  cancellation; its per-call model is wrong on this platform by ~25x, which re-prices R4 upward
  (the call count, not the bytes, is the lever) and explains why spec v3's "~25-30 ms map" became
  11-19 ms only after batching setAccess.
- **C2 — spec v3's promotion budget assumed a migration that does not exist on this path.** Spec v3
  budgets "map (~25-30 ms) + the measured tail-flip migration (34-59 ms) ≈ 60-90 ms off-path". The
  landed implementation needs no migration at all: conversations ride `copy_to_slot` from the main
  sessions (the 34-59 ms figure described the older slot-to-slot give-back path, a different
  mechanism). The landed promote (11-20 ms) + copy (12-90 ms) is inside the budget, but the
  composition is different from the spec's — relevant to anyone tuning "the swap" expecting
  Llumnix-style pipelining: there is nothing to tail-flip on this path; the copy is the term.
- **C3 — the promote log line is self-contradicting** (see R8): all 59 promotes label their source
  "the spare pool" while the spare is provably empty at every one of them (the boot frees to the
  driver, demotes eat to zero). The `had >= chunks_per` source test samples AFTER the release walk
  has filled the spare. Any future analysis reading the promote lines at face value — including
  F-272's own "(map 12-17 from the spare pool)" — overstates how often the cheap path ran: it ran
  ZERO times in this log. F-272's headline numbers (11-19 ms) remain correct as totals; the
  attribution inside them does not.
- **C4 — the host-pin bound is flat, not ×batch.** The F-272 journal says pins are "bounded --batch
  x 1.55 GiB". The boot lines print 1.55 GiB total at batch 6 AND at batch 9 (the pinned KV-
  streaming shadow is a process-level pool; per-slot deltas after the first are ~0). The true bound
  is ~1.55 GiB per process regardless of batch — a contradiction in the favourable direction; the
  "demote must unpin or leak" concern from spec v3's lifecycle accounting is even smaller than
  feared, and unpinning is indeed impossible anyway (the pins are baked into captured graphs).
- **C5 — spec v1's static-9 −9.6% attribution was already corrected in F-272** (the pl_want gate is
  inert on single-GPU). T35 adds nothing new here except confirmation from the live gate: with
  `--batch`, the pipeline is not constructed, so `pl_active < 4` never binds on this box. The gate
  carries the right semantic for future multi-GPU elastic configs; it is dead code today.

---

## Honest negatives (considered and rejected)

- **Fault-ahead / demand mapping for arenas.** CUDA VMM has no demand page-in: accessing mapped-
  access VA without physical backing is a fault against the context, not a page request. Eager
  mapping is mandatory; "map on first touch" is not available. (The lazy-capture tax is a different,
  real thing — see below.)
- **Pre-capturing the wide batch graphs at boot** (to kill the ramp's lazy-capture tax, ~3x window
  cost for the 7/8/9-row shapes, once per boot). The VRAM for wide-shape graph buffers was exactly
  F-271's boot-killer ("9-row captures claimed at request time beat boot-time sizing"); reclaiming
  it at boot trades ~0.1-0.3 GiB of the pool that just won +4.0% conc to save a few tens of ms once
  per boot. Rejected on the battery's own arithmetic.
- **Deeper predictive pre-mapping (Llumnix virtual-usage beyond one step).** The watermark (R2) IS
  the 1-step prediction, and it is the right depth: arrivals in the measured workloads come at most
  one promote ahead of each other (the b9 ramp promoted 7 slots behind 7 serialized admissions,
  each hidden under its own GEN 1). Pre-mapping 2+ ahead holds 1.5-2.2 GiB out of the cache to
  save ms that GEN 1 already hides. Rejected.
- **Raising `STRATA_ELASTIC_DEMOTE_S` as the hysteresis tuning.** Looks like the same lever as R2
  but is its opposite: a longer demote delay keeps slots MAPPED through gaps, so the solo lane that
  follows a conc phase reads a smaller cache (−1,300 slots class) — it trades M1 (the no-regression
  lane) to save demotes. The watermark is the correct form of the same idea: demote promptly (the
  cache regains nearly everything), keep only one arena warm.
- **Hiding the release walk under an existing wait.** The walk's quiesce contract (device idle +
  tier swaps landed) has no concurrent wait to hide under during serving — by construction the walk
  may only run where nothing is in flight, which is the admission point it already occupies.
  Overlap is not available; prediction (R2) is.
- **Parking the whole 0.73 GiB arena on demote** (instead of conversation-scoped parking, R5).
  Copies rope tables, page tables and every unused cell; the checkpoint machinery already copies
  only the used prefix. Strictly worse than R5 in bytes and in restore time.
- **A cudaMemPool for the spare.** Already struck by spec v3; the VmmChunk spare is the substrate,
  and `cuMemPoolTrimTo` cannot move memory between citizens (T33-priorart-materialization.md,
  honest negative 5). Re-affirmed against this code: nothing in the lifecycle wants pool semantics
  beyond what the spare vector already does.
- **BADM-0 retry storms.** Bounded by design (6 refusals max, 2 s park per refusal,
  server.py:1600-1611) and unexercised (zero refusal lines in 59 promotes). The residual risk — a
  refusal discards that admission's GEN 1 prefill and retries on another slot — is real but only
  reachable at the cache floor (128 slots), which a healthy config never touches. Watched, not
  acted on.

---

## Findings by confidence

- HIGH (code + log, uncontradicted): the spare is empty at every promote and every demote (59/59,
  48/48); the demote bill lands on the arriving request's thread (code order at generate.cpp:12565;
  128 ms measured for 4 slots); the promote print conflates release+map and its source label is
  dead code (code, plus the internal log contradiction); demote = unmap 6-15 ms + remap/refill
  19-30 ms; promote totals 11-20 ms; per-chunk map cost 29-51 µs; pins 1.55 GiB flat; ITL p99 35 ms
  @ 9-conc contains no lifecycle events; the hit-rate dip after the cycle (93.1% vs 99%+).
- MEDIUM (multiple independent evidence, or inference from strong primaries): the M1-bracket low
  end (211.5) is the churn transient's signature (mechanism measured at the hit-rate level; the M1
  attribution is inference); the −1.0% conc upper bound for the watermark's 320-slot cost (linear
  scaling from F-273's +1,300 → +4.0%); copy_to_slot's 40-90 ms class on the default path (code
  comment + park/restore timing lines, not directly timed per admission).
- LOW: super-chunk per-call cost being flat in size (vAttention's table + one indirect measurement;
  needs the microbench); the exact split of the promote's 11-20 ms between release-walk and map
  (unresolvable from the current log — R8 resolves it).
- UNKNOWN (searched, not found): any promote failure or BADM-0 event in the logs (searched "could
  not be materialized", "BADM" in the engine log — zero hits; the retry path is code-verified
  only); any give_way refusal of an unmapped slot in the logs (no counter prints exist to grep).

## Open questions

- Q1: The promote's release-walk vs map split (R8 answers in one boot). Also: why the walk+map
  totals fit 11-20 ms when the demote's unmap alone measures 6-15 ms — warmer driver state, fewer
  unmaps than assumed, or both?
- Q2: Does the demote-eat refill actually copy ~180 experts (~415 MiB) every cycle, or only the
  non-resident subset (the `host_res[i] >= 0` skip makes it workload-dependent)? The eat's share of
  the 28-40 ms is inferred, not logged. One counter on `kvg.refilled` per demote line answers it.
- Q3: Is the F1 solo bracket's low end (211.5) the churn transient? Needs the five-boot pooled
  confirmation already owed from F-272/F-273, with per-phase hit-rate lines kept.
- Q4: `STRATA_SLOT_COPY_DIRECT=1` on THIS recipe: byte-identity (golden hash) and the admission
  copy's actual ms — the falsifier is one battery leg.
- Q5: Chat-return TTFT distribution on a real multi-chat trace (the t34 probe has no think-gap
  pattern) — the baseline that R5's gain should be measured against.

## Implications

- The adopted +4.0% conc / zero-solo-tax result does not depend on the churn mode; R1+R2 remove the
  churn without touching the capacity mechanism, and the battery lanes to re-verify are F3 (conc 3/3
  + M1) plus the five-boot pooled confirmation already owed.
- The two zero-code items (R3, R6) are one-battery-leg decisions; R6 touches every admission and
  should ride the next adoption batch's MMLU/mimic formality (it is timing-class-affecting until
  proven byte-identical).
- R4 is the structural follow-up if this box's GSP call costs stay dominant at higher churn rates
  (more slots, more cycles); at the current cycle rate the watermark makes call count mostly
  irrelevant to latency, so R4's priority is below the policy fixes.
- Reliability posture is good: gates complete with a per-process fallback, promote failure is
  rolled back and bounded (6 retries, 2 s parks), teardown order is scope-safe, pins flat and
  bounded. The one latent wart is the server-side `slot_held` staleness on demote (slow, not
  wrong) — fix alongside R5.

## Source list

Primary (this repo, read directly):
1. generate.cpp at HEAD (F-272 diff = live code; commit 4505e56a): ElasticSessions struct
   (~5378), boot release (~6085), KvGrow cache-side split (7928-8159), kvg_release_until
   (7964-8046), kvg_cache_eat_spare (8079-8142), es_promote/es_demote/es_idle_scan/EsBg
   (11510-11604), idle-scan call site (12565), BGEN background start (12614-12618), give_way gate
   (14105), pl_want active-count gate (14834-class), admission promote + copy (16064-16076),
   copy_to_slot/direct_slot_copy (11656-11733), slot_source (13222-13296), next_line (11133),
   kv-grow gate requiring batch==0 (5237-5255), static `profile` load (4455).
2. include/strata/core/vmm.hpp + src/core/vmm.cpp: 2 MiB granularity, one cuMemMap per chunk,
   setAccess batched per run, chunk create/release, rollback semantics.
3. include/strata/core/conversation_cache.hpp: ConversationCache / SavedConversation /
   ConversationKv (host-side parking, budgets, pins, LRU, borrows).
4. custom-iq3_s.log: 8 elastic boots; 59 promote lines (11.3-19.8 ms; map 10.8-19.3; zero
   0.5-0.6), 48 demote lines (24.8-44.2 ms; unmap 6.2-15.3), boot lines (2.91/5.10 GiB freed,
   8,226 vs 6,923 slots, pins 1.55 GiB flat), park/restore lines (42-63 ms parks, 12.5 ms restore),
   re-read line (5,865 tokens = 1,361 ms), hit-rate lines (93.1% dip vs 99%+ steady), zero
   materialization failures.
5. campaign/results/t34-elastic.log + t34-*.out: F3 metrics (TTFT p50/p99 306-344/2,020-2,420 ms;
   ITL p99 20 ms), F4 (388/2,395 ms; 35 ms), accept 74.5-77.98%; battery script (per-leg boot,
   prewarm, probe, kill).
6. serve/server.py: pick_slot held-prefix heuristic (2107-2150), BGEN_REFUSALS_MAX=6 +
   SLOT_PARK_S=2.0 park path (1600-1611), BADM drain (2057).
Secondary (this repo, inherited):
7. T33-dynamic-session-allocation.md (spec v1-v3), T33-ourtree-verification.md,
   T33-priorart-materialization.md (H1-H5, vAttention Table 3, failure catalogue),
   T33-priorart-serving.md (Llumnix, vLLM sleep mode, RFC #20098, MoE-Infinity, Sarathi,
   Clockwork), T33-serving-delta-notes.md (Llumnix policy layer D1-D2).
8. SVG-CUSTOM.md F-270..F-273: the falsification chain, the F-272 journal (deviations list), the
   F-273 adoption battery and the c_row 2.4 ms row-cost finding (F-270), the F-254/F-271 hygiene
   and saturation context.

## Methods note

Method: read the F-272 commit diff (git show 4505e56a) and verified no later generate.cpp change
exists; read every live call site the diff touches plus the machinery it shares (vmm, kvg walks,
copy paths, parking, server slot protocol); extracted all 59 promote / 48 demote lines and the
surrounding request lines from custom-iq3_s.log to reconstruct the actual lifecycle mode; read the
t34 battery artifacts. Cross-checked every claimed mechanism against code line numbers; the
promote-path attribution was reconstructed from the demote's numbers after the promote line proved
self-contradictory (C3) — flag it, do not trust it further without R8's timer split.
Unproductive / do-not-repeat: grepping the launcher for kv-grow flags (the gate is `o.batch == 0`
in generate.cpp:5248 — with batch slots, kv-grow is structurally off and only cache_side runs;
the "elastic K/V" boot line's absence confirmed it); searching the log for BADM refusals and
give_way refusals (no counter prints exist — absence of evidence there is not evidence of absence);
attempting to decompose the promote line's map_ms against the demote's unmap alone (internally
inconsistent by construction — C3 — stop re-deriving it).
