# campaign/standing-line.md - F-450: the quality/perf standing line (operator directive, 2026-11-23)

"Better quality at the same performance, or better performance at the same quality."

## The rule

A candidate change lands only if, measured A/B against the same baseline build under both
batteries below:

    quality(delta) >= 0  AND  perf(delta) >= 0,   at least one strictly > 0.

A quality-for-speed trade (either axis negative) is not a candidate.  It is an operator
decision, taken only with the measured trade on the table and recorded as its own F-entry.

## The batteries (both required, same build pair, engine order reversed)

- Quality: fixed prompt suite (5-task coding class + short factual), run greedy AND
  sampled-with-fixed-seed; report exact-match rate and task score.  Greedy runs under a
  determinism-pinned config must be byte-identical, or the divergence is a defect first.
- Perf: TTFT at 4K and 32K prompts; median decode tok/s over full-length runs only;
  completion count (n/6 style); cold and warm labeled separately, never mixed in one median.
- Determinism canary: `shadow_misses` (include/strata/core/expert_source.hpp:661) logged per
  run.  Two runs of the same prompt with different `shadow_misses` carry different
  quantization mixes (native-hit vs Q2_0-miss compute, expert_source.cpp:5198-5200); their
  outputs are not comparable and neither is a valid quality measurement.

External benchmark evidence cited below (2026-11-23, anonymous rig, Strata 0.1.59 vs
PolyStrata 0.1.45, Qwen3.6-35B-A3B UD-Q4_K_XL, 16 GB 4090 laptop) is DIRECTIONAL ONLY.
Parity contracts remain A/B against our own builds.

## The ledger

### Tier A - both axes up, or quality up at zero perf cost

A1. RAISE EXPERT RESIDENCY on constrained-VRAM configs.  The external decode gap (3.21x,
    143.2 vs 44.6 tok/s) tracks the residency gap (3.67x, 11.3 vs 3.08 GiB) - decode is
    expert-byte-bound, so residency converts misses to hits at ~zero cost.  With the Q2_0
    shadow armed every converted miss is ALSO a quality upgrade (Q2_0-CPU -> native-GPU).
    Levers: pin budgets (STRATA_ARENA_PIN_GIB / STRATA_PIN_RESERVE_GIB /
    STRATA_PARTIAL_PIN_GIB), PCIe share, KV footprint (see B1).
    Falsifier: shadow_misses -> ~0 at steady state; ITL battery; quality battery.

A2. ROUTE TAIL SKIP STAYS 0, everywhere.  External: off was FASTER (135.6 vs 131 tok/s)
    AND higher quality (91.5 vs 88.4, 5 coding tasks x 3 reps; the delta sits in the
    hardest tasks); the upstream author ships 0.  Batteries and configs must never enable
    it; any future default flip is a Tier C decision.
    Falsifier: grep configs/launch envs for STRATA_ROUTE_TAIL_SKIP != 0.

A3. MTP DEPTH TUNING, behind exact-verify certification.  mtp.cpp has zero unit callers
    yet MTP depth 2 shipped as an external default - a coverage gap on a live decode-lane
    feature.  If greedy verify is exact (committed tokens == non-MTP greedy), depth is a
    free-speed knob; certify with an MTP-on vs MTP-off greedy exact-match canary (extend
    the F-419 M3b canary), then measure acceptance-rate-vs-depth.
    Falsifier: exact-match canary; DRAFT_TIMING per depth.

### Tier B - perf up at ~equal quality (the quality battery must certify "equal")

B1. KV q8_0 INSTEAD OF FP32 on long contexts.  4x KV bytes cut -> decode bandwidth and
    VRAM freed for A1.  q8_0 KV is near-lossless in principle; the battery decides, not
    the principle.  Existing infra: STRATA_HC_Q8 family.
B2. LONG-PROMPT PREFILL SCALING (the TTFT lane).  External: our prefill is flat
    1,940 -> 1,922 tok/s from 4K -> 32K while the competitor scales 1,852 -> 2,917; TTFT
    17.05 s vs 11.26 s at 32K.  Chunk overlap/batching headroom in Prefill::run_impl
    (src/prefill/prefill.cpp:2562).  Quality-neutral by construction (exact math, same KV
    writes) - certify with prefill parity fingerprints.
    Falsifier: input tok/s curve at 4K/16K/32K vs baseline build.
B3. ROUTING-AWARE COLD PREFETCH.  The external 65-70 s cold reads on synthetic prompts vs
    1.6 s on matched text = uniform routing over a cold expert store.  Prefill's router
    histogram predicts decode's expert working set; prefetch those blobs during prefill
    (STRATA_FILL_AHEAD / STRATA_IO_PF_* family).  Quality-positive under the shadow
    (fewer Q2_0 misses), quality-neutral without.
B4. SHADOW POPULATE OVERLAP AT BOOT.  Our own logs: memfd populate 5.3-17.8 s serial at
    boot.  Overlap with model load.  Perf-only, quality-neutral by construction.

### Tier C - quarantined quality-for-speed knobs (operator authorization required)

C1. STRATA_SHADOW_EXPERTS (Q2_0 miss compute, expert_source.cpp:5198).  BY DESIGN the
    misses' outputs differ by quantization; the hit/miss boundary moves with cache
    history, so armed-shadow runs are not output-stable across cache states.  May stay
    armed only where the quality battery passes WITH it armed, or where residency holds
    misses ~ 0 (cold-start insurance, not steady-state compute).
C2. STRATA_ROUTE_TAIL_SKIP != 0.  See A2.  Not a candidate.
C3. STRATA_SPEC_PROB probabilistic draft acceptance (mtp.cpp:653).  Sampled traffic only;
    certify distribution exactness before it counts as quality-neutral.

## Provenance

The 2026-11-23 external numbers are internally consistent (3.21x = 143.2/44.6;
32768/2917 = 11.23 s ~ 11.26; 32768/1922 = 17.05 s) but unverifiable - no artifacts.
Use as direction, falsify on this box.

## Corrections and additions (F-462/F-463, 2026-11-25) - the ledger re-read against the measured floor (F-461)

- II `STRATA_ADAPT_OVERLAP=1` is WITHDRAWN for this rig: removed by F-187, measured -5..-8 % at 6-conc (F-185),
  refused by the engine under DMA_BATCH.  Keep DMA_BATCH + ADAPT_LAG + DEVICE_PLAN.
- A1 residency: the placement family is the other session's priced frontier (F-453 ff., "gold" pinned for
  batteries); F-461 names the hit rate its movable variable 1.  New config items from the F-463 trace: +~600
  slots from idle VRAM (A14), the learned-vs-base profile as an operator question, not a finding.
  RESOLVED for gold (F-464, same-window ABAB): the standing boots on the 10-14 seed and never reads the learned
  bin; as the boot prior gold costs 6-conc 3.5-6.8 % (tail misplacement: 38 % more PCIe-routed misses) with M1/M3b
  parity - the standing keeps the seed.
  CLOSED by F-465 on a concurrent routing trace (`tools/trace_ideal_tracking.py`): allocation and capacity are
  spent on this rig (+411 slots +0.14-0.25 pt of hit rate, the trace's own cut +0.2-0.4); the live 6-slot hit rate
  (92.4-95.0 %) is what an ideal tracker earns with 40-75 rows of memory at 290-506 swaps per window, and the tier
  pays 5.3 - it is swap-rate-bound.  Blob copies run at 42 GB/s (measured; the engine's 12 GB/s is a default), so
  the lever is the swap's bytes: the copy-back is 41 % of each swap.  Priced, not built.  The F-463 "+~600 slots"
  item is retired with the capacity result.
- A2 tail-skip: production runs 7 (T63) while the determinism pin says 0 - a contract conflict; on this 32 GB
  rig at 96 % hits its perf value is ~nil (~2 skipped entries/window), so 7 -> 0 is a quality-side battery item.
- The quality battery exists: the other session's paired 534-question test (p = 0.345 for the F-452 state).
- NEW lane absent from the standing line: PREFILL (420 ms + 0.155 ms/token; 62 % of the log's wall time is
  prompt reads; per-request p50 58.5 t/s is diluted by other slots' reads).  See F-463 A1/A3/A4/A7-A10.
- Solo-decode "8-11 ms/token" and "prefill flat" in the original text were misreadings (MTP-amortized windows;
  laptop rig).  F-461's numbers govern: solo 235-248, conc 289-297 t/s.

## Corrections and additions (F-466, 2026-11-25) - the floor's regime; the request path ranked by the production log

- THE FLOOR'S REGIME: F-461's solo and concurrent numbers are measured at ~1.1k-token prompts with long generations.
  The production log (11-22 -> 11-24, 4,819 requests, 15,817 request-seconds) carries a 96k-token median context,
  two requests in three generating <= 64 tokens, and split its time 57.5 % decode / 42.5 % prompt read on the engine's accounting lines.  [CORRECTED by F-467: those lines are printed at a request's admission to the batch lane, so their decode is the solo lane's only; by the server's completion lines prompt reads are at most 8.3 % of request time and batch-lane decode about 80 %.]  (F-463's
  "62 % of wall" set request-seconds against cohort-round seconds; like with like it is 42.5 %.)
- Depth is not a decode cost on this engine: the idle ladder reads 292.6-308.4 t/s at 57k-168k tokens against
  222.0-251.7 at 2k (nine runs, drafts accepted 0.952-1.000).  F-181's -11 % at 244k is not reproduced to 168k.
- The request path, by the operator's request-seconds (`campaign/reqlog-census.py`): (1) cold re-reads of long
  sessions 21.7 % - the conversation cache is byte-bound at 8 GiB (three or four conversations of this size; 708
  evictions in the boot); (2) the per-turn read below 1,024 tokens 10.4 % - 0.8-1.2 ms per token, for bytes that
  need a quarter of that on the link (`prefill.cpp:135`, STAGE = 8); (3) the per-turn read of 1,024-8,192 tokens
  8.5 % - stream-all moves 30.0 GB where 26-31 % is routed (`prefill.cpp:2905`); (4) prefill at depth, 5.9-6.5k
  tok/s, sets every cold start's 17-28 s stall; (5) reads run on the window thread and stall every slot.  Decode -
  the adaptive tier, the CPU pool, the profile, the link share - is about 91 % of request time (80 % of it in the batch lane; F-467) and is at its floor for this kernel set.
- Instruments: `campaign/reqlog-census.py` and `campaign/ctx-lane.sh --turn | --branch` (reads, never tracker
  points).  F-463's A8 is withdrawn: the WY-chunked recurrence is in the standing behind STRATA_GDN_CHUNKED (F-224).

## Corrections and additions (F-467, 2026-11-25)

- **Standing config:** `--conversation-cache-mib` 8192 -> 32768 and `--conversation-cache-slots` 24 -> 96
  (operator's choice of size; the slot count is the cache's second cap and would have bound at ~6 GiB of the 32
  for short sessions).  `"prewarm": {"hold_s": 180}`: a request arriving during the warm-in waits for it instead of
  ending it.  Rollback: the two args to 8192 / 24, `"prewarm": {}`.
- **F-468 (19:45):** `"reasoning_loop_recovery": "recover"` and `"log_sampling": true`: a reply that repeats
  itself - in the thinking or the answer - is found within a few thousand tokens and never runs to its limit
  (three forced loops ended at 4,096-6,145 of 16,000 tokens); the evidence lands in `logs/reasoning-loop-*.log`.
  Rollback: remove the two keys.
- **Lanes, interleaved, 4 boots an arm (17:05-17:45):** M1 215.7 vs 218.3 (-1.2 %), 6-conc 257.2 vs 264.3
  (-2.7 %), M3b 78.1 vs 77.9 (+0.2 %): parity; the arms are one engine state while the probe runs (the cache never
  passed 4.25 GiB).  No tracker point: the standing's lanes did not move.
- **The cache's proof is a working set, not the probe:** 20 conversations of ~29k tokens, one turn each in order -
  8 GiB: 0 of 20 turns served (88.5 s of re-reads); 32 GiB: see F-467 section 3.
- **The census (F-466) corrected:** the engine's per-request line is an admission record under --batch.  Prompt
  reads are at most 8.3 % of the operator's request time; the concurrent decode lane about 80 %.
- **The warm-in, measured by shape** (`campaign/cold-cost-probe.py`): after the existing warm-in the only cold cost
  is the expert tier's fit to new content during its first request (solo first 143-175 against 190-230 t/s; the
  first tool call 138-163 against 207-219): 0.1-0.7 s a request.  Tool-call, slot-count and deep-read stages were
  built or priced and removed: no first-use cost to remove.  The warm-in now reports its stages and a settled
  reading on /v1/status and says NOT CONVERGED when its last short is below 90 % of its first.
- **RAM:** with a full 32 GiB cache about 18 GiB stay free (MemAvailable 88.7 GiB measured with 19 GiB parked).
