# Sampler split stage: a one-pass warp top-k instead of k argmax rounds (spec, 2026-11-25)

Status: BUILT 2026-11-25 05:15, default off (`STRATA_SAMPLER_FASTK=1` arms it). Evidence:
`campaign/results/f453-2026-11-25/drafter-stepgraph-pricing.md`.
- Passed: `build-fresh/sampler_fastk_parity --selftest` - the one-pass top-k against the k argmax rounds, list for list,
  bitwise over 192 cases (vocabulary widths 4,096 / 8,209 / 40,960 / 151,936; k 1 / 20 / 40 / 64; -inf, NaN, +inf, -0.0,
  duplicates, a short candidate list in every fixture), and the comparison is shown to see a one-entry difference.
  `sampler_parity --selftest`: 0 failures unarmed and armed.
- Not run: the benchmark on a quiet card (`--bench`); the engine ladder (bit-identity, M3b).
- Two defects the list-level test caught before any engine run: the sort read list slots past k that were never written
  (uninitialised shared memory entered the sort), and the ties pass tested a lane's rank without testing that the lane was
  itself a match, so non-members below the first match overwrote slot 0.

## The term

`src/kernels/cuda/sampler.cu:696` `sampler_split_part_kernel`: per 4,096-logit block, each of 4 warps holds 1,024
penalised logits (32 per lane, registers) and runs `k` sequential argmax rounds; warp 0 merges the four lists.
Measured in the drafter's step graphs (1,826 instances): **69.7 us on a grid of 10 blocks** at `kpart` = 64 = 1.09 us
per round - 25% of a 275 us step, on 40 warps of a 170-SM card.  The drafter path (`spec_draft_sample`, :1386) must
assume the widest list (the request's top_k is a device value); the standing traffic runs top_k 20 (the engine's
default - neither probe nor gate sends one).  The main sampler's split runs 38 blocks at k = 20 for the same kernel.

## The order both must return

Candidates: finite values and +inf (the rounds' first round admits every value below +inf or equal to it; -inf never
passes `> b0` with `b0` initialised to -inf; NaN fails every comparison).  Order: value descending, id ascending (two
chains walked in ascending id with a strict `>` keep the smallest id among equals).  Fewer than k candidates: all of
them, then sentinels `(n_vocab, -inf bits)`.

## Algorithm (per warp; `s[32]` per lane, id = lo + 32 j + lane)

1. Key.  `k32(f)`: normalise -0.0 to +0.0, then `(bits >= 0) ? bits | 0x80000000 : ~bits` - order-preserving on
   finite floats and the two infinities.  Non-candidates (NaN, -inf): masked out of every count.
2. Threshold (32 steps).  The smallest key `T` with `#(key > T) < k`.  Each step: per-lane count over the 32 registers,
   one 5-shuffle warp reduction.  Then `#(key >= T) >= k`.
3. Collect, in id order (j outer, lane inner: ids ascend).  Pass A: every `key > T` (`m < k` entries).  Pass B: the ties
   `key == T` until k.  Position = running base + popcount of the ballot bits below the lane.
4. Sort the k entries by `(value desc, id asc)` = ascending 64-bit key `((uint64) ~k32 << 32) | id`: a warp bitonic
   sort of 64 keys, 2 per lane, 21 stages.  Write `wl[warp][r] = (id, value bits)`; sentinels past the count.

Both procedures return the first k candidates of one strict total order, so the lists are identical element for element.

Cost estimate (unmeasured): threshold ~2,000 instructions, collect ~64 ballots, sort ~250 instructions: 3-5 us against
69.7 us at k = 64, against ~22 us at k = 20.

## Proof owed

- `sampler_parity` (existing) plus a randomised equivalence test against the rounds kernel: k in {1, 20, 40, 64},
  vocabularies of 1,024-4,096 with -inf, NaN, +inf, -0.0 and duplicate values injected; memcmp of every list.
- The env ladder's bit-identity check (the change is a recompute of the same math) and the M3b canary.

## Arm and price

`STRATA_SAMPLER_FASTK=1`, a template flag on the kernel, default off; rollback: unset.
Drafter: -46 (k 20) to -65 (k 64) us per step x 1.56 steps per window; the main sampler's split: ~-17 us per call.
**+0.8-1.2% M1: a composition piece (law 15), not a promotion.**

## The cheaper alternative

k-aware step graphs (lazy capture keyed by the request's top_k; the default list = the engine's default): 64 -> 20
rounds, -46 us per step, no new kernel - but graph-key plumbing in mtp.cpp.  A clamp of the request's top_k to a
captured constant is refused (law 16: a silent capability cut).

## Result (2026-11-25, F-453 closeout)

Bench, quiet card, `sampler_fastk_parity --bench` (`campaign/results/f453-2026-11-25/sampler-fastk-bench.txt`):
- the drafter's shape (nv 40,960, k 64): **68.6 us by k argmax rounds -> 20.3 us by the one-pass top-k** (the serve trace had
  read 69.7 us on 64 rounds of 1.09 us);
- the main sampler's shape (nv 151,936, k 20): 20.4 -> 12.8 us.

Engine price: 48 us x ~2.0 steps per solo window = ~0.1 ms of the 0.57 ms step-graph time = **about +1.1% M1** - real at the
kernel, under the +4% bar alone.  Kept default off (`STRATA_SAMPLER_FASTK=1` in the tree candidate `7a4d3477b2975d6a`;
bitwise in both modes, 192 cases + the planted-difference self-check) as a COMPOSITION piece: with the `gdn_ab` launch
fold (+1.6-3%, F-453) the two clear ~3-4% together.  Not run in the engine alone (law 15).  The one-pass is 20.3 us rather
than the ~5 us first estimated: ten blocks, a 32-step threshold search per warp - a finer split (more blocks over the
vocabulary, a wider merge) is the next kernel step if the composition needs more.
