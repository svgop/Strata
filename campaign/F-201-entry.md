### F-201 — R25 DRAFT-TEMPERATURE ADOPTED (+13.4/+9.8/+7.5%): the coupled drafter samples at 0.85×; the doorbell/fence tax quantified; L2 premise corrected to 96MB (2026-10-15 ~06:30, goal round 18 execution)

**R25 — THE CHANGE (sampler.cu, ~10 lines).** `STRATA_SPEC_DRAFT_TEMP=x` scales the DRAFT
chain's sampling temperature in `coupled_stage_kernel` (the staging kernel that copies the
request's SamplerParams to device — the same place gumbel is overridden).  The exact-match
verifier emits the TARGET's picks either way, so this is a PROPOSALS-ONLY change — the output
distribution is untouched (no MMLU gate class; the emitted stream = target's own samples).
Mechanism: a cooler draft sharpens q toward its own top pick, raising the draft/target overlap
(arXiv:2609.24150's temperature-scaled overlap result — lossless for any q under rejection).

**THE SWEEP (6 boots, 3 probes each, fresh boots, one window, no collapses):**

| arm (DT) | M1 | pair | 6-conc | vs ctl |
|---|---|---|---|---|
| control (1.0) | 132.1 | 128.9 | 192.2 | — |
| 0.45 | 136.5 | 136.7 | 201.8 | +3.3/+6.0/+5.0 |
| 0.65 | 137.7 | 133.8 | 193.9 | +4.2/+3.8/+0.9 |
| 0.80 | 147.4 | 138.7 | 201.1 | +11.6/+7.6/+4.6 |
| **0.85** | **149.8** | **141.5** | **206.8** | **+13.4/+9.8/+7.5 ADOPTED** |
| 0.90 | 135.8 | 134.0 | 195.2 | +2.8/+4.0/+1.6 |

Clean response curve peaking at 0.85 (0.90 collapses toward control — argmax-like drafts lose
coverage).  This window was again the low-acceptance state (accept lines ~30%, M3b 62-65), making
the lever's effect visible precisely where the rescue saturates less.

**R26 AUDIT (quantified by the PCIe deep brief — campaign/research/F-201-pcie-doorbell-agent.md):**
- Device-side polls of host-mapped flags = UNCACHED non-posted PCIe reads, ~1-2µs RTT each →
  est. **1.8-3.6ms of a 17.1ms window**.  Fix direction: INVERT host→GPU signaling (host posts
  doorbells via cuStreamWriteValue/BatchMemOp as posted writes; GPU polls DEVICE memory at
  L2-hit ~200-300ns).
- `__threadfence_system()` = MEMBAR.SC.SYS + ERRBAR + CCTL.IVALL (SASS-verified) = full L1+L2
  invalidate + PCIe drain — budget ONE per handoff, never per poll.  13 sites in elementwise.cu.
- Our 12.7GB/s DMA = per-op overhead floor (~5-10µs/op at 256KB chunks); batch to 2.8-10MB
  phase-aligned megablobs → ~3-4× of the DMA slice of the 4.9ms tail.  AM5 mixed-traffic cap
  ~15-25GB/s (single IO die, dual-channel DDR5 shared with the CPU pool).
- WC mappings for one-way staging only (never host-polled counters); THP+cudaHostRegister for
  the 40GB pool (PTE ÷512); NVMe: page-cache-prefetch the model once, QD≥4.
**Combined upper bound: ~4-7ms of the window + ~3-3.6ms of the tail.**

**R27 FINDING.** The GDN step kernel's block-shape retune is ALREADY BANKED in the fork (the
MAX_H=8→1 fix, documented in gdn.cu:17-32).  Sibling kernels share the starvation SHAPE but are
µs-scale; the real item is the step kernel's ILP rewrite (two serial i-passes, strided global
loads — the latency-bound pattern) = designed change, queued with the attention row-fold.

**DEEP BRIEF 1 (CUDA µarch — F-201-cuda-uarch-agent.md) highlights:**
- **PREMISE CORRECTION: RTX 5090 L2 = 96MB (not 128)** — our 60MiB carve is 62.5%; docs formula
  allows ≤72MB.  Re-derive carve math; query persistingL2CacheMaxSize.
- In-graph node spacing on Blackwell is CONDITIONAL (1.5-6µs even on datacenter; near-zero for
  long kernels) — our 9.5µs is not a floor; decompose with nsys --cuda-graph-trace=node.
- L2 accessPolicyWindow attaches PER GRAPH NODE — phase-switch persisting (KV/hc) vs streaming
  (experts) inside one graph; hitRatio discipline (window>carve @1.0 = 11% SLOWER, measured).
- Hot-vs-cold weight reads 7 vs 11µs (LMSYS measured, expert-traffic flushes-L2 class = our hc
  line); CTA-tile swizzle alone 87→94% L2 hit.
- TMA multicast ABSENT on sm_120 (clusters=DSMEM only); bare TMA swap regresses — producer-warp
  pipelines only; bf16 small-ops (router/head) ≈ +10% decode step (LMSYS); device graph launch
  removes ~1ms/step host cost; GB202 L2/GDDR7 latencies WORSE than Ada's (130/329ns).
- DVFS: headless consumer parts sit 180MHz GPU/405MHz mem with no auto-boost (vLLM 6.5× case) —
  verify the lock service holds clocks DURING probes (a drift-wave suspect!).

**STANDING NOW:** F-200 stack + STRATA_SPEC_DRAFT_TEMP=0.85.  Cumulative vs the night's start
(f200 ctl): M1 132→~150 class (+13%), pair 129→142 (+10%), conc 192→207 (+8%) — measured across
the two batteries in their own windows; next fresh-window battery re-baselines the stack.
Rollback: /tmp/strata-preF201-rollback (pre-DT binary; F-196 binary preserved at
~/LLM/backup/strata-preF196-rollback).
Sources: campaign/results/f201-*, research/F-201-{cuda-uarch,pcie-doorbell}-agent.md.
