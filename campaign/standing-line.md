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
