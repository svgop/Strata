# T33 — Prior-art dossier verified against our tree (main-loop study)

Date: 2026-10-23. Input: `T33-priorart-materialization.md` (the parallel session's survey). Method:
every claim that touches our code was checked against the tree (the F-254 law). Verdict up front:
**the dossier is right about the DESIGN it was asked to critique, and our STANDING engine is already
ahead of its assumptions in three of the four places that matter — the one real gap is exactly the
F-271 finding.**

## What our engine ALREADY has (the dossier's amendments, pre-empted)

1. **No per-session capture exists today — and never did.** The captured graphs are per VERIFY LANE
   per window shape T: `Verifier::capture` (verify.cpp:2513) fills `exec_[T]` / `exec_nr_[T]` on the
   lane objects (four capture families: 2522/2657/3554/3597). Sessions stage rows INTO lanes; the
   lane buffers are what the graphs bake. Amendment 1 ("capture once per lane at fixed shape, never
   per session") describes our standing engine — it correctly kills the T33 spec's PROPOSED
   per-session materialization, not anything shipped. Last night's F-271 "capture tax" was the
   restart cycle re-capturing ~8 shapes x doorbell families on first traffic — a once-per-boot cost,
   never per session.
2. **Graph-vs-residency decoupling exists — homegrown, different mechanism than #28248.** The
   doorbell duality at verify.cpp:2504-2511: `exec_[T]` = zero-doorbell graph (all experts resident),
   `exec_nr_[T]` = doorbell graph ("a prompt loan, a shrink or a swap: the doorbell graph runs the
   windows"). Residency changes swap GRAPH FAMILIES, never recapture. Equivalent intent to
   #28248's map-table; ours trades a second graph family for zero indirection in the hot path.
3. **Live KV <-> expert-cache re-partitioning via VMM is already in production in our tree.**
   generate.cpp:7947 `kvg_trim`: `qsa_kv_elastic_shrink` unmaps KV VmmChunks into a spare pool, then
   remaps spare chunks into the expert cache's VMM range (`xcache.vmm_range()`, granule-mapped —
   the 7955+ loop) — "the slots refill with the profile's hottest experts the GPU does not hold."
   This IS dossier amendments 3+4 (vAttention-class VMM repartition, spare pool, background
   semantics) — for two of the three subsystems. The expert cache sits in a stable VA range whose
   physical pages come and go; graphs reference the VA. Answering the dossier's open question:
   **no map-table indirection exists or is needed — stable-VA + doorbell families already delivers
   #28248's guarantee.**

## The one real gap (== F-271)

**Session buffers are plain `cudaMalloc`, eagerly, at boot** (generate.cpp:5441: `cudaMalloc(&buf,
bytes)` per slot, `session_init` into it, fixed count = `--batch`). No VA reservation, no pool, no
map-on-demand. This is why 9-wide cannot boot: 9 x 0.78 GiB is claimed unconditionally, before any
request exists, and the late claims (drafters, captures) then fight the pool sizing (F-271: engine
death + restart at every reserve).

## The design this implies (input to the T33 spec)

**Make sessions the third citizen of the existing VmmChunk elastic triad.** KV and the expert cache
already share the spare-chunk machinery; sessions should join them:
- Reserve VA for MAX sessions (9) x 0.78 GiB once at boot — zero physical commitment.
- Session start = map physical granules into the slot's VA (+ `session_init`/`session_zero` +
  conversation H2D) — dossier H3/VMM numbers: ~25-30 ms per session at 2 MB granules, run in the
  materialization background thread (the failure-mode catalogue's rule: never on the decode stream).
- Session end = unmap into the spare pool (~10 ms); the cache's kvg_trim-style refill can then eat
  the freed chunks for hot experts — memory the user is not using RIGHT NOW becomes cache, which is
  exactly "9 takes VRAM only when used" with the idle VRAM put to work instead of parked.
- Sizing floor (the F-271 rail, matching #28248's formula): free - mapped sessions - KV - rail,
  with the rail at the measured ~2 GiB class — the auto-raise behavior server.py already does
  becomes the floor's enforcement.
- Graph safety needs ONE verification the spec should carry: confirm no capture family bakes a
  SESSION address (expected: none — lanes own the baked buffers; sessions stage in). If true, VMM
  sessions are graph-safe by construction.

## The dossier's open questions, answered from the tree

- "Data-dependent grid sizes beyond row counts?" — the device-plan route (verify.cpp:1113
  `device_plan_ = !all_resident_ && ...`) already moves routing decisions device-side INSIDE the
  graphs; we are past padding-only designs on the MoE tail (this is the mechanism vLLM's closed
  spec-length RFC named as future work).
- "Is #28248's slot-map already in the sibling/upstream?" — No (`slot_map`/`map_table` greps empty);
  our stable-VA + doorbell approach makes the table unnecessary for us.
- L1 (our true capture cost) — boot/lane-level only; the honest number is measurable from one
  instrumented boot (the capture free-VRAM lines already timestamp-able); it is never a per-session
  term, so it does not gate the session design.
