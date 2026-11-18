# Reliability research, pass 2 — mechanisms, numbers, and design deltas

Status: DEEP DIGEST (2026-11-19). Companion to `proactive-reliability-research.md` (pass 1,
principles) and `reliability-stabilization.md` (the plan). This pass goes one level down:
checkpoint/restore internals, anchor stability over time, the statistics of certification,
integrity machinery, postmortem lessons, and the true sources of GPU boot nondeterminism.
Every section ends in concrete deltas to R3/R4/R5 or new plan items.

## 1. Restart-consistency by checkpoint/restore — the strongest known form

**vLLM Sleep Mode** (https://docs.vllm.ai/en/latest/features/sleep_mode/,
https://vllm.ai/blog/2025-10-26-sleep-mode) is the closest production analogue to our problem,
and its numbers are the ceiling we should aim at: waking a sleeping model is **18-20× faster
than a fresh load** (~0.1-0.8 s small models, ~3-6 s large), because the engine never actually
disassembles its state. Three transferable mechanisms:

- **Level 1 vs Level 2 state taxonomy**: L1 keeps weights (backed up to CPU RAM) and discards
  KV; L2 keeps only load-time buffers (rope scaling tensors, pointer tables, closures) at their
  **same addresses** and discards weights. The lesson is the taxonomy itself: a boot's state
  falls into classes — parameters, derived tensors, scratch, KV, captured graphs — and each
  class needs an explicit keep/discard/recompute decision. Our boot path has never had this table.
- **CUDA graphs "replayed, not recaptured"** (`sleep_mode_offload_cudagraph`): graphs are
  captured into a cuMem pool, backed up, and restored in place. Graph determinism survives
  sleep. For us: window-graph replay state is exactly the class that must survive an anchor
  restore, or every restore pays recapture.
- **`cache_salt` / version matching**: KV blocks carry the weight version so stale blocks can
  never match after an update. This is our R2 fingerprint generalized: EVERY cached/anchored
  state must carry the fingerprint of the inputs that produced it (profile, pack hash, geometry,
  env) and refuse mismatches.

**NVIDIA cuda-checkpoint + CRIU** (https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu):
official support, driver ≥550, drains submitted work, copies device memory to host, releases
GPU. Limits: x64 only, single process, **no UVM, no IPC handles**. Verdict for us: viable as a
future "hard checkpoint" path on this box (sm_120, driver-new), but the IPC/UVM exclusions and
the 30 s stage deadlines make it a Phase 6+ option, not the R3 mechanism.

**Firecracker/MicroVM snapshots** (https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md,
"Restoring Uniqueness in MicroVM Snapshots", arXiv:2102.12892): restore-in-milliseconds at
scale, and the subtle law — restored state is NOT fresh state (PRNG entropy, timestamps, unique
IDs must be re-seeded at restore). For us: an anchored restore must re-seed exactly the things
that must differ (admission timestamps, session epoch) while keeping the things that must not
(resident set, graphs).

**Delta to the plan:** R3's idle re-anchor gets a stronger sibling — a **sleep-mode-class
"soft checkpoint"**: on idle, the engine writes the full anchored state (resident table,
graph pool descriptors, tier accounting) alongside the fingerprint; a same-config restart
RESTORES it instead of re-filling, collapsing restart variance to the fingerprint check alone.
Also: add the **state taxonomy table** (parameters / derived / scratch / KV / graphs / tier)
to the Boot Contract as a design deliverable of R3.

## 2. Does an anchor stay good? Routing stability over time

**"Not All Models Suit Expert Offloading" (arXiv:2505.16056)** measures what we need: local
routing consistency, via SRP (can a fixed expert group cover a token segment?) and SCH (oracle
cache hit rate at a size limit), across 20 MoE models. Findings that bear directly on R3:

- **The optimal cache size for most models is ≈2× the active experts** — the balance point
  between coverage and waste. (Our regime differs — we cache ~200/layer vs 8 active — but the
  *method* prices any candidate cache size against the model's SCH curve.)
- **A strong trade-off exists between local routing consistency and local load balance**: models
  trained to balance load locally route less consistently → their hot sets churn more → anchors
  decay faster. Qwen3.8's balance regime determines our anchor half-life; measurable with their
  published method (code: https://github.com/ljcleo/moe-lrc).
- **Domain-specialized experts drive consistency more than vocabulary-specialized ones** — an
  anchor curated toward domain-stable experts decays slower than one chasing raw frequency.
- SCH correlates strongly with real cache hit rates — it is the offline instrument for pricing
  anchor quality WITHOUT boots.

**CARE-MoE** (https://dl.acm.org/doi/full/10.1145/3832810.3832869) names the exact mechanism we
need for anchor refresh: a **drift ratio CHDS(e) = F_current(e)/F_offline(e)** — current
activation frequency over the offline (anchored) frequency — with iterative swap optimization
when drift-induced imbalance crosses a threshold. This is the principled version of our
"re-anchor on idle": the anchor is not rewritten on a timer but when measured drift exceeds a
priced bound.

**ELDR** (arXiv:2607.00466) treats **workload drift** as a first-class scheduling input in
disaggregated MoE serving; **domain-expert perturbation results** (arXiv:2601.10159) show
up/down-weighting domain experts moves accuracy by 5-10 points on domain tasks — the literature
confirmation of T18's mechanism (residency changes routing; routing changes quality), which is
why our anchor promotion carries an MMLU gate.

**Delta to the plan:** R3c (anchor promotion) gains a **drift metric** (CHDS-style, from the
engine's own heat counters) published in the R5 report, with a priced refresh trigger; anchor
curation prefers domain-stable experts; add "measure SCH/SRP for Qwen3.8" as a cheap offline
task that prices anchor half-life before any code lands.

## 3. The statistics of certification — R4's decode length, computed

Wald's **SPRT** (https://cran.r-project.org/web/packages/SPRT/index.html; truncated sequential
binomial planning: https://ieeexplore.ieee.org/document/5386017) gives the minimal-average-sample
decision rule between two hit-rate hypotheses, with both error rates chosen. For runtime drift
monitoring, **CUSUM** (https://www.sciencedirect.com/science/article/pii/S1110016824007105)
detects small sustained shifts that threshold alarms miss — Shewhart-style thresholds are
"abortive for small to medium changes."

Our numbers (computed from the engine's own routing density, ~346 routed entries/token):
- 32-token decode: n≈11k entries → 95% CI on hit = **±0.39 points**
- 64-token decode: n≈22k → **±0.27 points**
- 128-token decode: n≈44k → **±0.19 points**

The f382 spread was 91.9-96.2% (minimum actionable gap ≈0.7 points). **A 64-token greedy decode
resolves the certification decision with ~2.6× margin**; SPRT boundaries certify good boots in
even fewer tokens and only bad boots run long. Cost: 64 tokens ≈ 0.3 s at 200 t/s.

**Delta to the plan:** R4 is now fully specified: SPRT on the per-entry Bernoulli stream
(H0: hit ≥ 95.5% vs H1: hit ≤ 94.5%, α=β=0.05), truncation at 128 tokens; CUSUM (not
thresholds) on the running hit EWMA for in-boot drift alarms in the R5 report.

## 4. Integrity machinery — the doublewrite pattern

**InnoDB's doublewrite buffer** (https://dev.mysql.com/doc/refman/8.0/en/innodb-doublewrite-buffer.html):
pages are written to a sequential scratch area + single fsync BEFORE their final in-place
write, so a torn write is always recoverable; and it has a **DETECT_ONLY mode** — checksum
without recovery, for benchmarking. Two laws for us: (a) our pack/state writes should be
doublewrite-shaped (we already do tmp+rename for profiles; the shadow STORE's in-place
population does not have this); (b) a DETECT_ONLY checksum tier is the right first step for
pack verification — measure corruption incidence before paying recovery complexity.

**Postmortems of the exact failure class:** the Redis cascade (https://blog.stackademic.com/redis-cache-failures-dont-stay-in-the-cache-layer-f4eda5546589)
— hit rate 94→71→34% in 18 minutes while the on-call waited because "a temporary dip isn't
unusual" — is our F-271 story in production form: **hit rate is the page-first signal, and its
alarm must be statistical (§3), not eyeballed**. MongoDB WT-16718 (https://jira.mongodb.org/browse/WT-16718):
a background task (checkpoint) blocked eviction until application threads were recruited into
the maintenance work — the exact shape of our adapt tier stealing decode windows when it
falls behind. Law: background maintenance (adapt, compaction, checkpoint) must have a hard
budget on request-critical threads, which is R3a's churn budget by another name.

## 5. The true sources of GPU boot nondeterminism

The JAX GPU determinism note (https://raw.githubusercontent.com/vene/jax/refs/heads/gpu-determinism-note/docs/gpu_determinism.rst)
lists them precisely: **autotuning selects different cuBLAS/cuDNN kernels per run depending on
exogenous factors**, and **atomics make reduction order nondeterministic**. For our engine this
means: two boots with identical fills can still differ if any kernel-selection autotuning runs
at boot (heuristic plan choice, split-K selection) — the fingerprint must cover the SELECTED
PLANS, or plan choice must be seeded/pinned per fingerprint. Deterministic re-execution as a
verifiability primitive (arXiv:2602.00182) is the extreme form: byte-equality of a fixed
greedy decode across boots is the strongest possible certification canary — our R4 fixed
greedy decode can hash its output and add "token-identical to the anchor decode" to the
certificate.

**Delta to the plan:** R2's fingerprint gains the kernel-plan selections; R4's certificate
gains the greedy-decode output hash (byte-equality vs the anchor's recorded decode).

## Consolidated deltas

1. R3 + soft checkpoint (sleep-mode class): state taxonomy table, restore-in-place graphs,
   fingerprint-salted state (§1).
2. R3c + CHDS drift ratio with priced refresh trigger; domain-stable anchor curation; offline
   SCH/SRP measurement for Qwen3.8 as a zero-boot task (§2).
3. R4 = SPRT (64-128 tokens, error rates 0.05) + CUSUM runtime drift + greedy-decode hash
   byte-equality (§3, §5).
4. Phase 4 doublewrite-shaped store writes; DETECT_ONLY checksum tier first (§4).
5. R2 fingerprint extended to kernel-plan selections (§5).
