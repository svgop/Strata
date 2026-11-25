# F-453 — the drafter's step graph priced at node level (2026-11-25 04:10; read-only)

Source: `Custom-data/profiles/2026-11-25-f453/f452-dev.sqlite` (the 11-24 node-level capture, `--cuda-graph-trace=node`).
Scripts: `census-scripts/f453-stepgraph.py`, `census-scripts/f453-stepsper.py` (copies of /tmp/f453-*.py). Instances are cut
where consecutive kernels of one graphId are more than 150 us apart. Nothing was built or run.

## 1. The step graph is KERNEL-bound, not launch-bound

Step graphs (graphIds 13, 16, 19, 22, 25, 52, 71, 77, 80; median 150-350 us, 30-80 kernels): 1,826 instances.
Per instance: kernel time **275.1 us**, inter-kernel gaps **9.5 us** (median gap 0.19 us, p90 0.19, max 100.8), sum 284.6 us.

The earlier audit's "50 nodes x ~5 us = launch-bound" (drafter-launch-path-audit.md, ranked cause 1: plain serial edges,
UNCONFIRMED) is FALSIFIED: per-node cost is the kernels' own execution on small grids.  Programmatic dependent launch for the
drafter's graphs is worth at most the 9.5 us of gaps (3% of a step): DEAD.

| us per step | kernel | grid |
|---:|---|---|
| 69.7 | `sampler_split_part_kernel` | 10 blocks |
| 36.3 | `native_small_mmvq_kernel` (the head) | 40,525 |
| 22.7 | `gr_up_v3_kernel` x2.8 | 160 |
| 17.5 | `gr_down_v3_kernel` x2.8 | 41x4 |
| 12.6 | `decode_attn_i8_split_kernel` | 1x2x20 |
| 11.9 / 9.4 / 4.9 / 4.3 | `native_small_mmvq_kernel` x1 / x1.8 / `native_mmvq_multi_kernel` x0.9 / x2 | 12,288 / 2,560 / 1,280 / 640 |
| 8.8 / 5.6 | `gu_grouped_t_kernel` / `down_grouped_t_kernel` (the block's experts) | 40x10 |
| 8.1 | `spec_draft_merge_kernel` | 1 block |
| 4.1 / 4.0 | `mtp_select_kernel` / `coupled_stage_kernel` | 1 block |

## 2. The sampler: 64 argmax rounds on 10 of 170 SMs

`sampler.cu:696` `sampler_split_part_kernel`: each warp holds 1,024 penalised logits in registers and runs `k` sequential
argmax rounds.  The drafter path (`spec_draft_sample`, :1386-1403) launches it with `kpart = coupled_kpart(nv)` = the WIDEST
list (kSelMax = 64) - "since the request's top_k is only known on the device" - over `coupled_blocks(nv)` 4,096-logit blocks.
The traced grid is 10 blocks: the drafter samples a ~40k draft vocabulary (dvocab_), 4 warps per block.  69.7 us / 64 = 1.09 us
per round.  The standing traffic sends no top_k (probe-lane.py, mmlu-pro-gate.py): the engine default is 20
(generate.cpp:502, :14035).  So 44 of 64 rounds produce candidates the merge never reads.

Levers (none built):
- k-aware lists: capture the step graphs per `kpart` (lazy; the default list = the engine's default top_k, a wider list only
  when a request asks for it).  A clamp of the request's top_k to a captured constant is NOT acceptable (law 16: a silent
  capability cut).  Expected 64 -> 20 rounds: -46 us/step.
- a one-pass selection (warp bitonic top-k) instead of k scans: up to -60 us/step; parity against the reference sampler.
- a finer block split cuts the scan per round but not the dependent reduction chain: the smaller share.

## 3. Steps per solo window (re-prices the step cap)

709 solo steady intervals (consecutive instances of one window graph, inter-start < 25 ms, no other window between):

| window graph | n | steps/window mean | distribution 0..5 | step-graph GPU time | wall between windows |
|---|---:|---:|---|---:|---:|
| 38 | 317 | 2.41 | 78 / 10 / 108 / 25 / 17 / 79 | 0.661 ms | 1.23 ms |
| 4 | 300 | 0.73 | 82 / 218 / 0 / 0 / 0 / 0 | 0.191 ms | 0.73 ms |
| all | 709 | **1.56** | median 1 | **0.423 ms** | **0.980 ms** |

Drafts per window = 1 (the round graph's draft 0) + steps = 2.56; the standing emits ~1.97 tokens per 8.73 ms window, so
~0.7-0.8 drafts per window are never verified: 0.2-0.3 ms per window.  The step cap's +2% pricing stands; the census's
"step chain 1,364 us median" is wall (the round graph and host gaps included), not 3.5 steps.

With steps/window at 1.56, the sampler lever is 0.07-0.09 ms per window (+0.8-1.0% M1): a composition piece.
