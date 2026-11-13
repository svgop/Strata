# The frontier master roadmap — 18 audits folded into one attack order (2026-11-13)

Owner: this file. Sources: the eighteen digests at `.docs/research/digest/2026-11-13-frontier-*.md`
(serving-arch, moe-offload, quant-kernels, toolchain, spec-decode, product-surface, attention, quality-eval,
platform, long-context, llamacpp-delta, numerics, crosspoll-horizon, prediction-theory, rdma-doorbell,
db-lessons, weight-space, sysconf) + the earlier no-more-ceilings wave. Every item carries its digest anchor.
Gain classes judged against the measured walls: DRAM lane 49.5-50 GB/s (only BYTES/OVERLAP move it); the
doorbell tail 0.98 quiet / 12.27 busy; window 8.4+1.57x rows ms; conc 13.4 ms @ 4.81 rows.

THE WAVE'S VERDICT: the architecture validated everywhere (upstream shipped our hybrid; five of their graph
PRs target a layer we replaced; DB science says our policies are IRM-optimal). The gaps are EXECUTION
WIRING: proven-but-unarmed arms, measured-then-discarded inputs, and a handful of priced kernel ports.

## TIER Q — the quality gates, FIRST (they gate every other tier; near-zero engineering)
| # | item | anchor | cost |
|---|---|---|---|
| Q1 | Wire the KL yardstick into t20-gates with delta-vs-baseline bands + a committed per-revision reference TSV | quality #1/#8 | low |
| Q2 | 512q temp-0 seeded MMLU gate (the 70q gate has an 11pp MDE — past 1-3pp verdicts were noise) | quality #2 | low |
| Q3 | bit-check `compare` subcommand + 50-100 prompt bank, auto-run | quality #3 | low |
| Q4 | CI: CPU-runnable parity + tokenizer oracle in CI | quality #4 | moderate |
| Q5 | Quality rows in lanes.csv (kind=quality) + determinism envelope stated | quality #6/#7 | trivial |

## TIER 0 — built-but-unarmed: proven machinery sitting off (one boot cycle can carry several)
| # | item | evidence | port |
|---|---|---|---|
| 0.1 | Router FP64->FP32: ungate the bit-identical fast kernel on CUDA (FP64 soft-exp per token per layer today) | numerics #2 | S-M |
| 0.2 | Arm KV_PREFETCH (measured +5.3% @32K prefill; decode-miss overlap is the same arm) | long-ctx #3 | env |
| 0.3 | Session-loop cudaEventQuery throttle (the verify-loop form; the anti-pattern's defense predates the fence fix) | rdma #1 | trivial |
| 0.4 | UMONITOR/UMWAIT on doorbell spins (pool pattern verbatim, in-tree) | rdma #4 | S |
| 0.5 | STRATA_BATCH_G2: batch windows read the captured-but-ignored group split | serving #5 | S-M |
| 0.6 | STRATA_DMA_BATCH=1/2 battery (mode 2 = CUDA 13.4 overlap flag, never exercised) | toolchain #3 | battery |
| 0.7 | Conditional-graph mode/retry arm (machinery complete, named next step) | toolchain #4 | S-M |
| 0.8 | DT 0.85->1.0 re-sweep under SPEC_PROB (0.85 was the exact-match optimum; now wrong-shaped) | spec #4 | env |
| 0.9 | TC attention defaults off-launcher silently scalar — flip defaults with fallback ladder | attention #1 | S |
| 0.10 | PDL arm on the expert decode chain (in-tree on dense_tc) | quant #8 | env A/B |
| 0.11 | Suffix/lookup lane arming (built, asleep; a lookup draft is a free batch row) | spec #6 | S |
| 0.12 | MMQ stream-k OOB cherry-pick (correctness; vendored tree carries the fixed upstream bug) | llamacpp #8 | trivial |
| 0.13 | -lineinfo in the CUDA release path (enables all NCU work; then the first-party NCU pass) | toolchain #2 | trivial |
| 0.14 | CMake guard: refuse poisoned nvcc 13.2.0/13.2.1 | toolchain #1 | trivial |
| 0.15 | CLK usage-count CLOCK for KV residency + adaptive rebalance cap | db #1/#5 | trivial |
| 0.16 | Two-codebook alignment (prompt path -> int8 TQ codebook = decode) | numerics #1 | M |
| 0.17 | Proactive yield condition (give_way's `can`) + stream priorities + exp spin backoff + clamp hygiene | long-ctx #4, toolchain #8, rdma #8, numerics #5 | S |

## TIER D — device-level, zero engine code (one root command first: `sudo dmidecode -t memory`)
| # | item | expected |
|---|---|---|
| D1 | Verify EXPO; tune tRFC/tREFI (2DPC 4-DIMM at JEDEC-5600 today) | +2-5% lane |
| D2 | `iommu.pt=on` (fully-translated DMA-FQ today) | +1-3% DMA lanes |
| D3 | Pin the derate directly (2950/13800; Xid-79 class stability) | stability |
| D4 | L0 baseline logging in the battery header | hygiene |
| D5 | Strategic: 2x96GiB 1DPC or CUDIMM wave | +5-10%/+20-40% ceiling |

## TIER 1 — cheap code (days), the enablers
| # | item | anchor |
|---|---|---|
| 1.1 | Cost-model feedback loop (Spark-AQE lesson; the refit inputs arrive every window; unlocks D5 and every horizon item) | crosspoll #1 |
| 1.2 | Tier-aware admission in `_admit_next` (LMetric score x Bidaw cost x Agentix session key — four papers, one function) | sysconf (a) |
| 1.3 | Goodput instrumentation (paired-SLO metric; the admission family's objective) | sysconf #12 |
| 1.4 | Sensitivity SPREAD measurement -> RCO chooser re-bake at expert granularity (our traffic as the budget; tooling only; −7-13% blob class) | weight #5->#1 |
| 1.5 | ClassLRU/CRF runtime recency over the static profile (cheapest residency upgrade) | llamacpp #3 |
| 1.6 | Locality-ordered miss partition (fewer/larger DMAs; 12.7->15-25 GB/s per-op) | moe #6 |
| 1.7 | ADAPT_OVERLAP for rebalance (un-block the next forward) | moe #3 |
| 1.8 | Belady bound + conditional-entropy sweep + timeliness CDF (the instrumentation before any predictor work; one scripting day) | prediction F2+gaps |
| 1.9 | Persistent prelaunch worker (no std::thread per window on the hot path) | crosspoll #4 |
| 1.10 | Per-layer quota retuning from the measured-then-discarded miss counts | moe #2/db #4 |
| 1.11 | SLRU/TinyLFU admission gating + pf usage-count latch | moe #4/db #2 |
| 1.12 | dj<=2 io-prefetch retest (copies still capped; budget exists now) | moe #7 |

## TIER 2 — medium ports (weeks)
| # | item | anchor |
|---|---|---|
| 2.1 | Zero-copy host-mapped streaming for the PCIe share (kills the 2x DRAM round trip; pattern proven on the DS4 store) | moe #1 — the top BYTES item |
| 2.2 | Completion-gated stage reuse (re-enable pinned staging; the MR-lifetime fix) | rdma #6 |
| 2.3 | cuStream Write/WaitValue32 memops as graph nodes (frees spinning SMs; deletes driver-callback raises) | rdma #5 |
| 2.4 | Futex/eventfd dedicated poller (the structural tail fix) | rdma #2 |
| 2.5 | Frame-graph liveness linter + arena aliasing (VRAM to residency; catches the #267 class by construction) | crosspoll #2 |
| 2.6 | Graded shadow store (per-lane quant allocation; tooling only; MoPEQ inversion warning) | weight #4 |
| 2.7 | Batch-lane adaptive verify-depth controller (AdaSpec cost model + per-slot confidence; the LEGAL F-352 re-open) | spec #1/sysconf (c) |
| 2.8 | LK/VAT drafter objective on the retrain (F-353's own named condition; acceptance 0.59->0.65-0.70 = every lane) | spec #2 — the keystone |
| 2.9 | Block verification on the sampled lane (composes with SPEC_PROB) | spec #3 |
| 2.10 | Cross-slot checkpoint mount (the 80%-exists radix intermediate) | serving #3 |
| 2.11 | Structured outputs via the sampler's logit_bias-proven mask plumbing | product #1 |
| 2.12 | logprobs transport (the DSI1/GENI sidecar pattern) | product #2 |
| 2.13 | DS prefill CPU-combine (one row per token instead of K experts) | llamacpp #4 |
| 2.14 | CacheGen-class codec on the T66 disk tier | sysconf #4 |
| 2.15 | Live degeneration detectors on the content channel (metric-first) | quality #5 |

## TIER 3 — kernel ports (weeks-months; KL/bit gates per law)
| # | item | anchor |
|---|---|---|
| 3.1 | Few-row MMQ / MMVQ-12 for the verify window (same-GPU same-workload +20.6% upstream; honest transfer +5-12%) | llamacpp #1/quant #1 |
| 3.2 | Decode int8-MMA (the 3.1x engine is prefill-only; +3-9% conc) | attention #2 |
| 3.3 | DS4 MMA attention + ring->paged (FlashMLA runs our BYTE-IDENTICAL KV formats on TCs) | attention #3/#4 |
| 3.4 | GDN fused-SM120 (FlashInfer shipped the pattern; external validation) | attention #6 |
| 3.5 | FP4 mma prefill path (1.26-1.31x PR-measured on this GPU class) | quant #2 |
| 3.6 | Whole-window service + device plan (the named rewrite; folds the doorbell findings; prefill too) | serving #1/#9 + rdma family |
| 3.7 | MoE-APEX dual-precision tiers (the tier-selective TQ2_0; ONE arm with the KL gate) | sysconf #2 + weight |
| 3.8 | Batch G=2 + always-split cohorts + decoupled admission | serving #2/#5 |
| 3.9 | Green-context SM fencing for spin kernels (CUDA 13.1; cheap falsifier) | crosspoll #5 |

## KILL-LINES — banked with evidence; do not re-price without a named falsifier change
Expert merging (functional subspace collapse at LLM scale). Trellis own-bakes (we MEASURED Gyro: 17-21%
slower, 49/60 vs 57/60; ik's claim was vs their baseline). LoSparse/MoBE decomposition. Pack-to-pack
distillation (250x our scale). DENSEGEMM at T<=8. DCP on PCIe. tcgen05/FA4 on sm_120 (twice-verified).
MagicDec. Multi-instance classes (Llumnix/Parrot/StreamEP/SwiftSpec). Preemption-kill (BYIELD owns it).
DirectKV (C2C fabric). CUDA-CUTLASS grouped GEMM on sm_120 (field-broken).

## Standing phase-0 falsifiers still queued from the pre-wave order (fold into T0 boot cycles)
`parallel:8`+cohorts 4+4 exact-fit (config-only, conc target); the env ladder (QIL/DOORBELL_STORE/
FUSE_WAIT=2/PDL); the prefill-chunk ladder 4096/8192/16384; the DS CUDA MMA prompt-attention port (=3.5).

## The open one-boot sequence I recommend
1. Q1-Q5 (the gates) — before any adoption is judged.
2. One boot cycle carrying 0.1-0.17 + the phase-0 config falsifiers, each env-gated with its own witness.
3. D1's root command + D2 grubby line at the operator's convenience; then the warm bench re-baseline.
4. Then by value: 1.1/1.2/1.4 -> 2.1/2.8 -> 3.1/3.2/3.6.
