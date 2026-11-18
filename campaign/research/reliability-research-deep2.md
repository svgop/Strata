# Reliability research, pass 3 — the defect class named, drift statistics, kernel I/O variance, GPU library determinism

Status: DEEP DIGEST 2 (2026-11-19). Third layer under `proactive-reliability-research.md`
(principles) and `reliability-research-deep.md` (mechanisms, certification math). This pass:
the fail-slow taxonomy our defect belongs to, the drift-statistics toolchain production ML
already uses, the precise kernel I/O variance mechanisms, and the determinism rule for GPU
library kernel selection.

## 1. Our defect class has a name: fail-slow

**"Fail-Slow at Scale"** (Gunawi et al., FAST '18; slides:
https://people.cs.vt.edu/huaicheng/s/fast18-failslow-slides-a.pdf) studied 101 fail-slow
hardware cases across 12 institutions. The definitions and findings map onto the boot lottery
one-to-one:

- **Fail-slow = still running and functional, but significantly slower than expected.** Its
  relatives: fail-stutter (Arpaci-Dusseau, HotOS'11), gray failure (Huang et al., HotOS'17),
  limp mode (Do et al., SoCC'13). Our bad boot — serving, correct, at 129 t/s — is a textbook
  fail-slow.
- **Faults convert**: fail-stop, fail-transient, and fail-PARTIAL faults all convert into
  fail-slow at scale. Our F-382 anatomy: a fail-partial (a fill that misses 4 hit-points)
  converting into a persistent fail-slow.
- **Detection is the hard part**: 13% of cases detected in hours, 13% in days, 11% in weeks,
  17% in months, 50% unknown. The boot lottery lived for days before f382 quantified it.
- **Root causes are mostly not code**: device errors, firmware, temperature, power,
  environment, configuration — and cascades (vibration → slow disk). The paper's ask of
  vendors: **when error masking becomes frequent, throw explicit signals instead of masking.**

That last line is the Boot Contract's reason to exist, from the literature: R2's refused boot
and R4's certification ARE the "explicit signals" — the engine refuses to mask a degraded
state behind `/health: ok`.

## 2. Drift statistics — the anchor-refresh policy, from production ML

**PSI (Population Stability Index)** — the credit-scoring/ML-ops standard for "has the
distribution shifted": bin the variable, compare anchor vs current proportions,
`PSI = Σ (a−c)·ln(a/c)`. Industry thresholds (https://github.com/SuchanMadhikarmi/mlops-complete-guide/blob/master/production-engineering/04-drift-detection.md):
**<0.10 stable, 0.10-0.20 moderate (watch), ≥0.20 significant (act)**. Applied to our anchor:
PSI over per-expert activation-frequency deciles, anchor window vs trailing window — a cheap,
interpretable refresh trigger with decades of production precedent. Evidently-style weekly
feature-drift review (https://www.netguru.com/blog/how-to-make-an-ai-model) is the same law:
**watch input distributions, catch silent degradation before output metrics show it** — our
analogue is watching routing heat, not waiting for t/s complaints.

**BOCPD** (Bayesian Online Changepoint Detection; risk-calibrated form:
arXiv:2510.09619) maintains a run-length posterior over "has the process changed," and the
cited work derives alert thresholds from **SRE error budgets** — the exact coupling our R5
report wants: hit-rate budget → posterior odds → alarm. Heavier than CUSUM; the split that
fits us: **PSI for the slow anchor-refresh decision (per idle period, interpretable),
CUSUM/BOCPD for fast in-boot drift** (per window, streaming).

## 3. Kernel I/O variance — the exact mechanisms under our mmap reads

- **Cassandra CASSANDRA-17237** (https://issues.apache.org/jira/browse/CASSANDRA-17237):
  kernel readahead on mmap'd files interacts pathologically with random point reads —
  the kernel's readahead window (128 KiB default) amplifies a 4 KiB random read into 128 KiB
  of I/O, and the extra pages EVICT useful cache. Our GGUF mmap expert reads are this exact
  pattern; the engine's decode-gate readahead accounting ("read ahead 6.5 MB, 4.4 used") is
  the countermeasure already in place — the law it implies: **every mapping declares its
  access pattern** (MADV_RANDOM for point-read mappings, MADV_SEQUENTIAL for sweeps).
- **Readahead window growth** (LKML, https://lkml.org/lkml/2010/4/12/8): first touch reads
  512 B, readahead grows 4-page → 128 KiB across sequential faults — the mechanics of
  first-touch cost variance that warmup must absorb.
- **Mozilla's startup I/O discipline** (https://phabricator.services.mozilla.com/D288588):
  a production answer to "startup should be deterministic": `posix_fadvise(WILLNEED)`
  prefetch of startup files, **IOPRIO_CLASS_IDLE for non-critical startup I/O** (never
  compete with the critical path), `MADV_SEQUENTIAL` on sequentially-read mmaps,
  `MADV_COLD` on idle caches. This is a browser's prewarm law and it is ours: the prewarm
  sweep is fadvise/WILLNEED; the profile-save and store population should run at idle I/O
  priority; cold mappings the derivation cannot see still need `prewarm.extra_files` (F-360's
  15% first-leg tax, now with the kernel mechanism named).

## 4. GPU library determinism — the selection rule, solved

- **The rule** (bwiemz/NSL commit, https://github.com/bwiemz/NSL/commit/5f6bfebf92d77879f031baf5b344060bd99bb7d7):
  "a timing-based autotune winner is a function of machine state at first use — two runs
  could select different kernels and differ in bits ACROSS processes... **heuristic[0] is a
  pure function of (shapes, library, device, workspace cap)**, so determinism keeps the Lt
  path, just untimed." This is the whole decision in one sentence: untimed heuristic[0] is
  deterministic; timed autotuning is machine-state-dependent.
- **The perf side** (HorizonXP/voxtral.c#13, https://github.com/HorizonXP/voxtral.c/issues/13):
  production practice queries top-N heuristics, times them once with CUDA events, caches the
  winner. Compatible with determinism ONLY if the winner is persisted keyed by the boot
  fingerprint — then it is a pure function of recorded state.
- **Graph capture determinism** (ooples/AiDotNet.Tensors#705,
  https://github.com/ooples/AiDotNet.Tensors/pull/705/files/6cf937c6741f12d73b106ab6b03a8e49febf61d3):
  capture-determinism = a **deterministic scratch pool** (same device address every replay)
  and **no allocator bookkeeping inside the capture region** (it perturbs the graph;
  cuGraphLaunch fails). Our engine's window graphs obey this today by construction
  (pre-armed geometry); the law to keep is address stability of everything a graph touches,
  which the soft checkpoint (pass 2 §1) must preserve.

## 5. Serving frameworks — what single-instance lanes lack

The production pattern (https://scalablebook.icsgen-ai.org/part-5-distributed-inference/module-23-distributed-inference-systems/section-23.8.html,
https://medium.com/@abineshsiva1998/serving-machine-learning-models-17-patterns-and-5-tools-explained-306641ba9d24):
**keep minimum replicas warm**, health-gate traffic per replica, let the autoscaler absorb
restarts — a user never meets a cold replica. A single-GPU lane has no replica cushion, so
R4 certification is the substitute: a boot that would fail a readiness probe must never
receive traffic. Same law, one instance instead of a fleet.

## Consolidated deltas (beyond passes 1-2)

1. **Fail-slow checklist in the Boot Contract**: every masked error becomes an explicit
   signal; the certification is the refusal to mask (§1).
2. **Anchor-refresh = PSI policy**: PSI over expert-frequency deciles, anchor vs trailing;
   ≥0.20 forces re-anchor + journal, 0.10-0.20 watches. CUSUM/BOCPD stay the fast in-boot
   alarms, thresholds tied to the hit-rate error budget (§2).
3. **Mapping-class law**: MADV_RANDOM/SEQUENTIAL per mapping class; idle I/O priority for
   prewarm-adjacent and state-save I/O; prewarm.extra_files for every mapping the derivation
   cannot see (§3).
4. **cuBLASLt policy**: untimed heuristic[0] is the deterministic default; timed autotune
   winners only ever load from the fingerprint-keyed cache (§4).
5. **R4 is the single-instance readiness probe** — no traffic to an uncertified boot, ever (§5).
