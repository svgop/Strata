# Proactive reliability research — engineering the boot lottery out of existence

Status: RESEARCH DIGEST (2026-11-19). Companion to `reliability-stabilization.md` (the plan) and
`boot-contract.md` (the design). Question asked: how do production systems make this class of
defect — silent, state-dependent performance variance at startup — structurally impossible
rather than merely detected? 40+ sources, six domains, each mapped to our engine's plan.

## A. Determinism by construction

**Reproducible builds** — the property "identical inputs → bit-identical outputs" is a decade-old
discipline with tooling and a manifesto (https://reproducible-builds.org; Lamby &
Zacchiroli, arXiv:2104.06020). The transferable law: no build/boot step may depend on ambient
state (time, free memory, file order) unless that dependency is itself hashed and pinned.
TensorRT-LLM's determinism work (https://github.com/NVIDIA/TensorRT-LLM/issues/2474) chases the
same property inside an inference engine. Our R1 (pinned geometry) + R2 (fill fingerprint) ARE
this law applied to boot; the discipline says go further: hash every input the boot reads
(profile, pack manifest, env, driver version) into the published fingerprint.

**Memory determinism** — databases disable THP because transparent promotion is a variance
source, not just a performance one (PingCAP: https://www.pingcap.com/blog/transparent-huge-pages-why-we-disable-it-for-databases;
Oracle Exadata: https://blogs.oracle.com/exadata/huge-pages-or-transparent-huge-pages-in-context-of-exadata).
Production latency work pins NUMA, C-states, IRQ affinity (https://www.shenglisoft.com/en/blog/low-latency-server-bios-tuning-isolcpus-numa-pinning-c-states-irq-affinity,
https://linuxgd.medium.com/mastering-numa-optimization-in-production-environments-2477fec5c96e).
CUDA's stream-ordered memory pools (https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
exist precisely to make GPU allocation deterministic and fragmentation-free.
→ Our estate already moved this way (boot-reserved hugepage pool, taskset CCD pin, v6 governor);
the law from the literature: EVERY boot-time allocation is a named, asserted budget line —
R1's allocation table is the audit surface, and any "free VRAM happened to be X" read is a defect.

**Startup as a measured, predictable pipeline** — the first systematic vLLM cold-start study
(arXiv:2606.07362, MLSys 2026) decomposes startup into six named steps, shows it is CPU-bound,
and builds an analytical model that PREDICTS startup latency per hardware. Triton ships model
warmup as configuration (https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).
→ R5's boot report object is the same instrument; the prediction model is the natural extension
(a boot that takes 2× its predicted step time is degraded BEFORE any decode runs).

## B. State anchoring — the R3 precedents

**Idle-time reconfiguration is published practice**: MoEpic (arXiv:2509.08342) explicitly
"when the device is idle, adjusts the expert cache in VRAM based on the updated cache
configuration." Our R3 idle re-anchor is the same move with a stronger contract (return to the
anchor, not a re-optimization).

**Cache state as a first-class, versioned citizen**: SGLang's RadixAttention
(https://www.lmsys.org/blog/2024-01-17-sglang) makes the KV prefix cache a tree that persists
across requests, and its cache-aware load balancer (https://www.lmsys.org/blog/2024-12-04-sglang-v0-4)
routes requests TO cache state — 3.8× hit-rate improvement, communication-free via an approximate
tree. NVIDIA Dynamo (https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models)
and Apple's tiered semantic cache (https://machinelearning.apple.com/research/semantic-caching)
make the same bet: state hierarchy explicitly owned and observable.
→ Residency must be an owned artifact: named, fingerprinted, restorable — not a side effect of
whatever the adapt tier last did.

**Checkpoint/restore as the strongest form**: CRIU (https://criu.org) and Firecracker
microVM snapshots (https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md,
https://www.pandastack.ai/blog/firecracker-memory-snapshots) checkpoint whole-process state and
restore in milliseconds — serverless platforms boot "cold" instances from snapshots so cold ≈ warm.
→ Future option beyond R3: snapshot the armed engine (resident set + memory layout) at a certified
state and restore it; restart then cannot differ because it is not a re-assembly at all.

## C. The MoE caching literature — hit rate is THE metric

- **MoE-Infinity** (arXiv:2401.14361, OSDI'24): activation sparsity on personal machines;
  trace-guided expert cache, 3.1-16.7× over vLLM/DeepSpeed-class baselines.
- **MoEpic** (arXiv:2509.08342): hit <50% at constrained caches is normal for LRU/LFU; their
  LCP priority μ·ρ^(ν/ω) (frequency × recency decay — our EWMA decay 0.85 is the same family);
  40% hit ⇒ 3.81× latency. Our knife edge (4.3 hit-points ⇒ 50% throughput) is this curve.
- **ExpertFlow** (arXiv:2410.17954), **Fiddler** (arXiv:2402.07033, ICLR'24), **Pre-gated MoE**
  (arXiv:2308.12066), **AdapMoE** (per-layer adaptive caching), **DAOP** (arXiv:2501.10375),
  **spatio-temporal prefetching** (arXiv:2606.15453), **domain-aware two-stage offloading**
  (IEEE 11397596): the whole field treats expert residency as a prediction+caching problem.
  None of them A/B a static anchor vs learned — our T18 result (learned fill pays a quality tax)
  is a datum the literature does not have.
- **Aerospike** (https://aerospike.com/blog/cache-hit-rate): hit rate as a metric misleads when
  tiers are conflated — instrument per-tier hits (VRAM/PCIe/RAM/CPU), which our window accounting
  already does; keep it in the R5 report.

→ Post-stability performance queue, literature-backed: next-layer expert prediction
(Pre-gated/AdapMoE class) to convert misses into prefetches; MoEpic's split-caching (cache the
top fraction of hot experts → more experts resident per GiB) as a VRAM-multiplier lever.

## D. Integrity and contracts — fail loudly, verify at boundaries

- **End-to-end checksums** (ZFS case study, FAST'10: https://research.cs.wisc.edu/wind/Publications/zfs-corruption-fast10.pdf;
  Google Cloud Storage: https://cloud.google.com/blog/products/storage-data-transfer/enabling-end-to-end-checksums-in-cloud-storage):
  verify data at EVERY boundary, because every layer silently corrupts eventually. Our pack NaN
  day is the exact case study. → Phase 4 checksums, with the ZFS rule: checksum at read time, not
  just write time.
- **Crash consistency** (https://0xkiire.com/crash-consistency-fsync-rename): tmp+rename+fsync
  is the only correct atomic-write pattern; our profile writer already does it — extend to every
  state file (fingerprints, anchors, certifications).
- **Design by contract** (https://en.wikipedia.org/wiki/Design_by_contract): pre/post-conditions
  executable at boundaries. The Boot Contract IS a contract; its exit criteria are assertions.
  Law 18 (invariants become structure) is this, already doctrine.
- **Kubernetes probes** (https://kubernetes.io/docs/concepts/workloads/pods/probes): the
  startup/readiness/liveness SPLIT is the industry answer to "health said ok but the service is
  useless." Our /health "ok" is liveness; R4's `certified` is readiness, and readiness must fail
  CLOSED. The **power-on self-test** (https://en.wikipedia.org/wiki/Power-on_self-test) is the
  40-year-old ancestor: boot proves itself before the OS is allowed to run.

## E. Variance discipline — measure the spread

- **The Tail at Scale** (Barroso & Dean, CACM: https://cacm.acm.org/research/the-tail-at-scale):
  variability, not mean, is what users experience; engineer against the tail explicitly.
  Our boot spread is a tail problem at boot granularity — the tracker's spread column (Phase 5)
  is the tail metric.
- **SRE golden signals + error budgets** (https://sre.google/sre-book/monitoring-distributed-systems,
  https://sre.google/workbook/alerting-on-slos): define a hit-rate budget (e.g. ≥95.5% per boot)
  and alert on burn rate, not threshold crossings.
- **GPU power/clock variance** (arXiv:2510.06902, OSTI characterization:
  https://www.osti.gov/servlets/purl/3005429) and **MLPerf's measurement rules**
  (https://mlcommons.org/benchmarks/inference-datacenter,
  https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc):
  clocks and power recorded with every number, warmup defined by rule. Our v6 governor + probe
  discipline already comply; the law is to keep them attached to every banked number.

## F. Chaos and crash-only — make recovery the common path

- **Crash-Only Software** (Candea & Fox, HotOS IX 2003:
  https://www.usenix.org/conference/hotos-ix/crash-only-software,
  http://dslab.epfl.ch/pubs/crashonly.pdf): one way down (crash), one way up (recovery);
  because recovery is the common path it is exercised constantly and therefore always works.
  For us: every engine stop should be crash-safe BY DESIGN, and every start is the recovery
  path — certified every time (R4), never trusted.
- **Chaos engineering** (Netflix practice: https://medium.com/devsecops-ai/how-netflix-uses-chaos-engineering-and-progressive-delivery-together-for-ultra-reliability-61cf07e658b5):
  inject the failure classes routinely, in production form. → Phase 5 soak becomes a chaos
  battery: corrupt-a-pack drill, kill-9 drill, pool-drain drill, profile-poisoning drill — each
  must produce a REFUSED boot or a certified self-heal (idle re-anchor), never silent slowness.

## What this changes in the plan

1. **R3 is confirmed, not novel** — idle reconfiguration (MoEpic) and state-anchored routing
   (SGLang) are published practice; our addition is the CONTRACT (anchor is fingerprinted,
   promotion is explicit, excursions are budgeted).
2. **R4 gets teeth from K8s**: readiness fails closed; `certified` is the readiness signal.
3. **Phase 4 gets the ZFS rule**: checksum at every read boundary, not just boot.
4. **Phase 5 becomes a chaos battery** with four named drills, not just a restart soak.
5. **New (Phase 6 candidate)**: snapshot-restore boot (CRIU/Firecracker class) as the eventual
   strongest form of "a restart cannot differ" — priced after R3 lands.
6. **Perf queue gains two literature-backed levers**: next-layer expert prediction
   (Pre-gated/AdapMoE class) and MoEpic-style split caching for hit-rate-per-GiB.
