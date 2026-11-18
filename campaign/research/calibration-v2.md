# Calibration v2 — a native, reliable expert-residency subsystem

Status: DESIGN (2026-11-19), grounded in F-382..F-389. Replaces the adaptive tier's open-loop
swap heuristic with a layered system: estimator -> policy -> predictor -> calibration protocol ->
persistence with champion/challenger -> certification. Sources: 56, cited inline.

## 1. The measured problem (what v2 must fix)

- The VRAM expert tier converges to a different local optimum on every boot (hit 90.2-96.4%,
  M1 128.9-224.6 on identical inputs; F-382/F-386). The geometry is now deterministic (F-386);
  the residual variance is the tier's open-loop control.
- The static profile seed is a 79% floor (static residency, F-387); the tier earns the other
  13-17 points. A profile saved from a certified 212.6 boot seeded a *worse* state (90.2%,
  F-389): the save records a heat ranking, not the converged residency.
- The switch is a hand-rolled scan: every 4 rounds, up to 96 swaps, victim = least decayed count,
  candidate = missed expert with count >= 0.5 or credit, admit only if gain >= 1.5. No admission
  filter, no window/ghost structure, no segment between global and local heat, no prediction, no
  calibration protocol. `--calibrate` (tools/calibrate.py) tunes only the *rate* knobs
  (--pcie-frac, --spec-min-p, --pool-workers, --adapt-every/swaps/decay, --prefill-pipe) per PC.
- The literature is unanimous that this class of policy loses badly to modern ones: TinyLFU's
  admission policy alone beats LRU by 30-40% miss ratio [1]; S3-FIFO reports up to 72% lower miss
  ratio than LRU [2]; MoE-Infinity's request-level activation matrix beats LRU substantially, and
  at large batches plain LRU collapses below 0.1% hit [3]; differential caching (globally hot /
  locally hot / cold) beats both [4]; prediction-based prefetching cuts TPOT 14% [5].

## 2. What the field has established (the design's foundations)

**Expert activation is skewed, temporally local, and partially predictable.**
- Recent selections predict the next step better than fused outputs; a residual-predictive
  structure exists in pretrained routers [6].
- PRED-MoE: "recently used experts are more likely to be reused in subsequent steps" [7].
- MoE-Infinity: a request-level expert-activation matrix makes offloading decisions [3].
- CECAR: hit rates are fundamentally bounded even under Belady; routing can be made
  cache-cooperative, but binary cache presence induces routing instability [8].

**Admission control beats eviction tuning.**
- TinyLFU: a frequency-based admission policy (candidate vs victim) layered on any eviction policy
  is a large, robust win [1]; W-TinyLFU adds a small recency window + a frequency sketch [9].
- CATinyLFU extends it with access-time variation [10].

**Modern eviction algorithms: scan resistance + low overhead.**
- ARC / CAR: self-tuning recency/frequency, scan-resistant [11][12].
- LIRS / CLOCK-Pro / CLOCK-Pro+: inter-reference recency, utility-driven adaptation [13][14][15].
- S3-FIFO: small/main/ghost FIFO queues, two-bit counters, ~72% lower miss than LRU at best [2].
- SIEVE: lazy promotion with a roving hand, simpler and 2-6x faster than LRU under concurrency [16].

**Segmentation: global vs local heat.**
- Diff-MoE: experts classified globally hot / locally hot / cold; priority-driven differential
  caching; locality-preserving replacement [4].
- Two-stage offloading: offline domain heatmaps + online spatial/temporal features; degrades
  gracefully to LRU out of distribution [17].
- HybriMoE: activation is unstable, fixed mappings fail; hybrid CPU-GPU scheduling + cache
  management [18].

**Prediction and prefetch.**
- SPICE: speculative prefetching with low-rank expert surrogates; overlaps CPU compute with
  transfers [19].
- "Speculating experts" [20]; Fiddler: popularity from calibration data is near-universal across
  domains [21]; a 2025 study reports ~82% accuracy predicting top-k experts and 95% of prefetched
  non-top-k experts still active [5].
- moe-l2: domain-predicted preloading into an mmap LRU cache [22].

**Serving-systems context (for scale).**
- Expert parallelism in vLLM/TensorRT-LLM/SGLang/DeepSpeed [23][24][25][26]; TensorRT-LLM's
  online EP load balancer (imbalance is stable across iterations but differs per request) [24] is
  the multi-GPU analogue of our single-GPU residency problem.
- Hot-expert replication for distributed imbalance [27]; inter-layer expert affinity (ExFlow) [28];
  parallelism-strategy comparisons [29]; MoEntwine [30].
- Streaming frequency estimation: Space-Saving, Count-Min, exponentially decayed aggregates
  [31][32][33]; learning-based eviction (LRB, GL-Cache, KVP-RL) improves over heuristics but needs
  training infrastructure [34][35][36] — a later tier, not the first.

## 3. The architecture

Six layers, each with one owner and one contract. The boot flow is: prewarm -> calibrate ->
certify -> save-if-better -> serve.

### 3.1 Estimator layer — `CalibrationStats`
- A **frequency sketch** per (layer, expert): 4-bit saturating counters (24,576 x 4 bit = 12 KiB),
  halved when the total access count reaches W = 10 x capacity (TinyLFU aging) [1][9].
  This replaces the float decay (`u *= 0.85` per round), whose half-life drifts with the traffic
  rate and whose per-boot phase differs.
- A **recency window** (the last N distinct experts per layer, N ~ 128) for the W-TinyLFU window.
- A **ghost list** of recently evicted (layer, expert) keys (no bytes) for admission decisions [2].
- One writer: the window submitter; one reader: the calibration thread. All updates are versioned
  (a monotonic `epoch`); the policy never reads a torn epoch.

### 3.2 Policy layer — `ResidencyPolicy`
- **Two segments** (Diff-MoE [4]): `global` (the calibrated profile's top-K, persistent) and
  `local` (per-request/conversation, evictable on idle). Sizes: global 70-80%, local 20-30%,
  configurable; the split is *measured*, not guessed.
- **Admission (TinyLFU)**: a candidate enters only if `freq(candidate) > freq(victim)` at the same
  epoch minus a margin; the ghost list gives the second-chance signal.
- **Eviction**: S3-FIFO-style small/main queues inside `local` (small ~10%, ghost 1x) [2];
  the global segment is evicted only by the calibration protocol (never by traffic).
- **Damping**: two-epoch confirmation for local swaps; a per-request churn budget
  (<= 1 cache capacity per boot, the invariant); `--adapt-every` remains the rate knob.

### 3.3 Predictor layer — `RoutePredictor` (opt-in, measured)
- A tiny probe on the router's hidden state (or the previous step's selected-expert provenance [6]):
  predicts the next window's top-k experts. Accuracy gate: >= 80% top-k [5], else off.
- Prefetch the predicted experts asynchronously into the free VRAM tier slots before the window
  (SPICE-style overlap [19]); a prefetched expert that misses costs the eviction of a cold one.
- The existing `STRATA_PREDICT_CREDIT/BUDGET` and the model's own lookahead are the v1 of this; v2
  promotes it from a swap tie-breaker to a prefetcher with a measured accuracy contract.

### 3.4 Calibration protocol — `--calibrate-tier` (the flag today's --calibrate should grow into)
- A fixed, deterministic workload: a corpus of representative prompts (the operator's real
  traffic, not the probe's), greedy decoding, a fixed seed, a fixed token budget (e.g., 8K tokens),
  one pass per calibration epoch.
- Run at boot after prewarm (or on demand): warm the cache, converge the local segment, then
  freeze it. Cost: ~60-120 s at 200 t/s.
- Output: `CalibrationResult { fingerprint, hit_rate, decode_rate, stats_digest, model_hash,
  corpus_id, epoch }` — the boot contract's evidence object.
- **Reliability**: the calibration workload is fixed and greedy, so the result is a function of the
  model + the policy state, not of sampling. This is what makes "calibrated at prewarm" real.

### 3.5 Persistence — champion/challenger with an automatic save gate
- The profile file becomes `calibration.bin` (v2 format: ranked pairs + frequencies + result
  metadata + a checksum), written atomically (tmp + rename), versioned.
- **Champion/challenger**: the incumbent is the champion. A new calibration is a challenger. It
  replaces the champion ONLY if it beats it on a held-out calibration split by a margin
  (e.g., >= +1.0 pp hit rate) with the same fingerprint rules [37][38].
- This is the "if any new calibration beats the old learned expert, it can save automatically"
  rule, made executable: no human step, no silent regression.
- The engine loads the champion at boot as the global segment seed; the local segment always
  starts empty (a clean, deterministic state).

### 3.6 Certification — the boot contract (already partly built)
- After calibration, one fixed greedy decode (the campaign's M1, 3 samples, max) must be >= the
  box's bar (200 t/s here); the result is published to /health as
  `boot: {state, hit, tps, fingerprint, calibration}`.
- A boot that fails certification does not advertise ready; the launcher/report says so (R4).

## 4. Invariants (executable, not prose)
1. `global segment size + local segment size + free == cache capacity` (a static_assert at init).
2. Every residency mutation increments `epoch`; the policy never reads across epochs.
3. The churn budget is enforced per boot (a counter; exceeding it logs and stops swaps).
4. The profile file is written atomically and carries its own checksum; a corrupt or
   checksum-mismatched champion is rejected at load (never used).
5. The calibration result is deterministic for a fixed (model, corpus, policy): two runs must
   produce the same fingerprint; a differing fingerprint fails the boot contract.
6. The certification bar is a config value; the boot may not claim ready below it.

## 5. Migration plan (strangler, no rewrite)
- **P1 (1-2 days): the estimator swap.** Replace `u *= decay` with the 4-bit sketch + aging;
  keep the existing scan/swap as the policy, behind `STRATA_CAL_V2=estimator`. Measure: hit-rate
  variance across 5 boots (the lottery) — the target is a std-dev under 1 pp.
- **P2 (2-3 days): admission + segments.** TinyLFU admission on the existing scan; split
  global/local; keep the churn budget. Measure: hit rate and the boot variance.
- **P3 (2 days): the calibration protocol + the champion/challenger save.** `--calibrate-tier`,
  the v2 profile format, the automatic save gate, the /health result. This is the "calibrated at
  boot" deliverable.
- **P4 (opt-in): the predictor.** Only if the accuracy gate (>= 80% top-k) is met on the corpus.
- Each phase lands behind an env flag, default off, with a one-line rollback, and is gated by the
  campaign's 3v3 boot interleave (law 1/2/11).

## 6. Expected effect (grounded in the measurements)
- The bad draws cost ~40-50% (135-160 vs 200-224). A deterministic calibration + admission removes
  the draw variance; the target is every boot in the 200-224 band.
- The literature's admission/segmentation gains (TinyLFU 30-40% miss reduction [1], S3-FIFO up to
  72% [2], differential caching [4]) are on top of the 79% floor; the v1 tier already recovers
  +13-17 pp, so the realistic v2 headroom is several more pp of hit rate at equal capacity.
- The fundamental bound is Belady/CECAR [8]: the ceiling is the hot working set, not the algorithm;
  v2's job is to hit it every boot, not to exceed it.

## 7. Sources
[1] TinyLFU — https://arxiv.org/html/1512.00727v1
[2] S3-FIFO — https://s3fifo.com/
[3] MoE-Infinity — https://arxiv.org/html/2401.14361v2
[4] Diff-MoE — https://dl.acm.org/doi/10.1145/3712285.3759903
[5] Expert prefetching study — https://export.arxiv.org/pdf/2508.18983
[6] Residual predictive structure in routing — https://export.arxiv.org/pdf/2609.17940
[7] PRED-MoE — https://dlnext.acm.org/doi/pdf/10.1145/3789240.3830285
[8] CECAR — https://neurips.cc/virtual/2026/loc/paris/poster/152287
[9] W-TinyLFU / Caffeine — https://dl.acm.org/doi/pdf/10.1145/3507920
[10] CATinyLFU — https://dl.acm.org/doi/pdf/10.1145/3578369
[11] ARC — https://dl.acm.org/doi/abs/10.5555/1090694.1090708
[12] CAR — https://dl.acm.org/doi/10.5555/1973374.1973388
[13] LIRS — https://dl.acm.org/doi/10.1145/511334.511340
[14] CLOCK-Pro — https://dl.acm.org/doi/proceedings/10.1145/3319647
[15] CLOCK-Pro+ — https://dl.acm.org/doi/abs/10.1145/3319647.3325838
[16] SIEVE — https://dl.acm.org/doi/abs/10.5555/3691825.3691893
[17] Two-stage expert offloading — https://ieeexplore.ieee.org/document/11397596/references
[18] HybriMoE — https://ieeexplore.ieee.org/document/11133274
[19] SPICE — https://browse-export.arxiv.org/pdf/2608.21240
[20] Speculating experts — https://export.arxiv.org/pdf/2603.19289
[21] Fiddler — https://arxiv.org/html/2402.07033v3
[22] moe-l2 — https://github.com/yalun753/moe-l2
[23] vLLM EP — https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
[24] TensorRT-LLM EP — https://nvidia.github.io/TensorRT-LLM/1.4.0rc4/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html
[25] SGLang EP — https://docs.sglang.io/docs/advanced_features/expert_parallelism
[26] DeepSpeed-MoE — https://ar5iv.labs.arxiv.org/html/2201.05596
[27] Hot expert replication — https://ieeexplore.ieee.org/document/11661192
[28] ExFlow — https://www.computer.org/csdl/proceedings-article/ipdps/2024/871100a915/1YpzY1D9GV2
[29] Parallelism strategies — https://ieeexplore.ieee.org/abstract/document/11540265
[30] MoEntwine — https://xplorestaging.ieee.org/document/11408594
[31] Space-Saving — https://ieeexplore.ieee.org/abstract/document/11112965
[32] Count-Min sketch — https://pkg.go.dev/github.com/v8fg/kit4go@v0.2.0/countmin
[33] Exponentially decayed aggregates — https://dimacs.rutgers.edu/~graham/pubs/slides/expdecay.pdf
[34] Learning-based eviction — https://dl.acm.org/doi/full/10.1145/3839232
[35] GL-Cache (CMU thesis) — http://reports-archive.adm.cs.cmu.edu/anon/anon/home/ftp/usr/ftp/usr0/ftp/home/anon/2024/CMU-CS-24-149.pdf
[36] KVP (KV eviction RL) — https://icml.cc/virtual/2026/poster/66783
[37] Champion/challenger (adaptive replacement, utility-driven) — https://dl.acm.org/doi/abs/10.1145/3319647.3325838
[38] LFU-based admission quality (TinyLFU's quality-of-result argument) — https://arxiv.org/html/1512.00727v1
