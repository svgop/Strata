# T27-3 — SERVING, OPS & STABILITY: the external frontier (2025-2026), the roadmap audit, and the premium-feel gap

Date: 2026-10-21. Scope: ONLINE frontier research (arXiv 2025-2026, vLLM/SGLang/TensorRT-LLM/Dynamo/llm-d/
llama.cpp-ecosystem, NVIDIA dev blogs, release notes) turned into priced, falsifiable candidates for a
SINGLE RTX 5090 + 9950X3D + 184 GB RAM, Qwen3.8-Flash-Next MoE IQ3_S, COHORTS=2 two-cohort reactor,
scoreboard M1 212.5 / pair 202.8 / conc 315.7 at 6 clients, acceptance 74.9%. READ-ONLY session: no boots,
no builds, no GPU, no server contact; the only write is this dossier. Prior dossiers consumed and NOT
repeated: T25-3 (family serving study), T26-4 (serving-conc source map), T26-6 (memory/host/cold-start),
F-201-{scheduling,kv,nvme,pcie}, F-200-*, R2-S3, T17B, T21/F-249 cohort results. Anchors re-verified in our
tree this session (line numbers from the working tree at 382d84b1-era). Estimates are marked EST.

---

## (a) FRONTIER MAP — what is genuinely new, with sources and dates

### A1. vLLM V1: chunked prefill always-on, decode-priority token budget (Jan 2025 → current)
- Sources: [Red Hat: vLLM V1 alpha (2025-01-28)](https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture);
  [vLLM Optimization and Tuning docs](https://docs.vllm.ai/en/stable/configuration/optimization) ("In V1, chunked prefill is enabled by default... the scheduling policy prioritizes decode requests");
  [vLLM issue #18547 (2025-05): chunked_prefill cannot be turned off in V1](https://github.com/vllm-project/vllm/issues/18547);
  [Ubicloud: Life of an Inference Request, vLLM V1 (2025-06)](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1).
- Mechanism: one scheduler loop; a fixed per-step token budget; a long prompt enters as chunks interleaved
  with decode rows; decode wins the budget first. No phase switch exists anymore — the prefill/decode
  boundary is dissolved into a budget question.
- Relation to ours: this is the TOKEN-BUDGET form of exactly what our `STRATA_BATCH_DECODE_SHARE` does in
  TIME form (piecewise read; between chunks the slots decode for share × piece_ms,
  generate.cpp:13696-13704, 14185-14191). The V1 design is the strongest independent confirmation that
  "decode-priority under a bounded admission share" is the industry-converged shape for single-GPU serving.
  The row-mixing extreme (prompt rows inside decode windows; Xolotl `SERVE_STEP_MAX`, T25-3 §3.5) stays
  blocked by `kVerifyMaxT=8` accounting: 3 slots × 2 MTP rows = 6 rows/window at 6 clients, 2 rows
  headroom (verify_kernels.hpp:21; collect cap generate.cpp:11524) — headroom exists but F-201-scheduling
  finding 1 (shape-keyed graphs) is its prerequisite, so it remains second-order to the share raise.

### A2. Sarathi-Serve lineage: stall-free chunked prefill is the thing that moves p99 (OSDI'24 → 2026 reviews)
- Sources: [arXiv:2403.02310 (Sarathi-Serve)](https://arxiv.org/html/2403.02310v2);
  [OSDI'24 presentation](https://www.usenix.org/conference/osdi24/presentation/agrawal);
  [2026 technical review (zhongzhuzhou.org, 2026-05-20)](https://www.zhongzhuzhou.org/blog/2026-05-20-sarathiserve-technical-review-en)
  ("stall-free batching *without* chunking yields only ~half the latency gains" — the stall, not the batching, is the bottleneck).
- Mechanism/relation: chunk + co-schedule with decodes = our piecewise read. The review's point maps to us
  exactly: the win is in HOW the admission interleave is shaped, which is what the decode-share raise and
  (later) the PIPED port tune. CONFIRMS T26-4 §6's ladder order; the TTFT readout must accompany the
  conc readout (a share raise can hold conc flat and still improve p99 TTFT — that is a win by this
  literature's own metric).

### A3. TWP — closed-loop token-window pacing on ITL + queue-depth signals (arXiv, Oct 2026) — NEW
- Source: [arXiv:2610.17689 (2026-10)](https://arxiv.org/abs/2610.17689).
- Mechanism: admission is paced by a feedback loop on observed ITL, queue depth and workload signals —
  the controller holds or admits the next request so the decode regime stays stable; chunked prefill on
  the way in. Reported: stable TTFT, lower p99, less ITL variation, throughput up vs conventional
  schedulers (their serverless benches; magnitude is THEIR measurement, not transferable 1:1).
- Applicability: this is the closed-loop version of our open-loop fixed `decode_share=0.5`. Everything the
  controller needs already exists on our box as SIGNALS without new engine code: per-window ms and
  tokens/window log lines (probe-lane.py:137 parses "avg T ..., N tokens/window, M ms/window"), server
  live state `queued`/`waiting`/`admitting` (server.py:4247-4253 metrics()), and the admission point is
  one Python function (`_admit_next`, server.py:1719-1729). A pacing gate = hold the next admission while
  the ms/window EWMA exceeds a threshold (or while a `decode_share`-derived budget is exhausted).
  NEW candidate, Python-only (S-M). Stability-lane, not scoreboard (6 fixed clients have no queue; the
  controller's value shows at 7+ and in TTFT variance).

### A4. Fairness without preemption: VTC and its locality-aware successor — NEW for our lane
- Sources: [USENIX OSDI'24: Fairness in Serving LLMs (VTC)](https://www.usenix.org/conference/osdi24/presentation/sheng),
  [arXiv:2401.00588](https://arxiv.org/pdf/2401.00588);
  [arXiv:2501.14312 (2025-01): Locality-aware Fair Scheduling in LLM Serving](https://arxiv.org/abs/2501.14312)
  (VTC ignores prefix-cache locality; combines fair counters with cache-aware dispatch).
- Mechanism: a per-client virtual token counter (input+output token cost); the scheduler admits whoever has
  the LEAST accumulated service. Purely an ADMISSION-ORDER policy — no in-flight preemption.
- Applicability: this is the missing middle between our merged-but-dormant `STRATA_ADMIT_SHORT_FIRST`
  (server.py:310-317, order-only inside `_take_control`, T26-4 §3.4) and maya's fair-slice (needs a PAUSE
  verb we lack, generate.cpp:12258-12263 vs maya generate.cpp:1131-1191). VTC-lite = per-client counters +
  `_admit_next` reorder: ~40 lines of Python, zero engine change, works under the F-14 ticket law exactly
  the way short-first does (tickets stay in arrival order; only WHO takes the next control line changes).
  It cannot preempt a decoding answer (that remains maya-slice/PAUSE territory) — fairness granularity is
  "who is admitted next", not "who is paused". NEW candidate (S-M), 7+-client lane only.
- Their locality caveat is our pick_slot: affinity is already handled post-admission (server.py:2107
  held-prefix→empty→LRU), so the 2501.14312 half is already effectively ours.

### A5. QoE/phase-aware schedulers (Andes; Pascal HPCA'26; JITServe NSDI'26) — mostly confirmations
- Sources: [Andes, arXiv:2404.16283](https://arxiv.org/abs/2404.16283) (token-level preemptive scheduling
  for prompt TTFT + smooth ITL on constrained GPU);
  [Pascal, arXiv:2602.11530 (HPCA'26, 2026-02)](https://arxiv.org/abs/2602.11530) (phase-aware
  reasoning/answer scheduling, controlled preemption, token pacing; p99 TTFT −72% — multi-instance
  design, migration at phase boundaries);
  [JITServe (NSDI'26)](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf) (SLO goodput with imprecise request info).
- Applicability: Andes/Pascal's preemption core = maya's fair-slice shape (in-flight pause + resume from
  prompt+produced) — CONFIRMS that the PAUSE verb is the frontier mechanism for the 7-12-client QoE lane,
  and that nothing cheaper with the same power has appeared. Pascal's placement/migration half is DEAD for
  one box. JITServe's uncertainty-aware admission is conceptually already covered by our VRAM-sizing-at-boot
  + QueueWaitExpired (server.py:391) at our scale; nothing portable priced.

### A6. Speculative decoding × batching: settled complementary (ICLR'25 → 2025-2026 practice)
- Sources: [MagicDec (ICLR 2025)](https://infini-ai-lab.github.io/MagicDec), [arXiv:2408.11049v5 (2025-04)](https://arxiv.org/html/2408.11049v5);
  [Together AI blog: spec decoding for high-throughput long-context](https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference)
  (up to 2x on 8xA100, throughput AND latency).
- Mechanism: at high batch the verify pass turns KV reads into the bottleneck; spec's draft-verify
  structure REDUCES effective KV reads per accepted token, so spec pays at batch too, not just solo.
- Applicability: our conc lane keeps slot MTP drafts ON (74.9% acceptance at 6 clients) — the frontier
  says that is correct and the "MTP-off-under-cohorts" test (T17B candidate 8) is a boundary probe, not a
  promising direction. CONFIRMED, no boot needed; draft depth adaptation remains the T25-2 lane (draft_cum
  keep rule), untouched here.

### A7. SGLang 2025: overlap scheduler + piecewise/breakable CUDA graphs
- Sources: [LMSYS: SGLang v0.4 zero-overhead batch scheduler (2024-12, the base)](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4);
  [SGLang piecewise CUDA graph docs](https://sgl-project.github.io/advanced_features/piecewise_cuda_graph.html);
  [SGLang releases (v0.5.x 2025: BCG default prefill path, verify/extend graphs)](https://github.com/sgl-project/sglang/releases).
- Mechanism: CPU scheduling overlapped one batch ahead of GPU execution (no device syncs in the loop);
  piecewise/breakable graphs give prefill dynamic shapes without full re-capture.
- Applicability: the overlap principle is our PIPED-for-cohorts port (T26-4 §3.1: reader thread + doorbell
  thread + main thread launching rounds) — a second, independent, at-scale confirmation of the mechanism.
  The piecewise-graph half maps to the window-architecture project (T25-5/F-201: launch-map diet, PDL),
  not to serving. CONFIRMED both; no new candidate.

### A8. NVIDIA Dynamo 1.0 / llm-d: disaggregation, KV-aware routing, Prometheus observability (2025)
- Sources: [NVIDIA Dynamo announcement (2025-03)](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models);
  [Dynamo disaggregated serving docs](https://docs.nvidia.com/dynamo/knowledge-base/concepts/system-architecture/disaggregated-serving);
  [Red Hat: llm-d KV-cache-aware routing (2025-10-07)](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference);
  [llm-d blog: KV-Cache Wins You Can See (2025-09)](https://llm-d.ai/blog/kvcache-wins-you-can-see);
  [llm-d-kv-cache-manager](https://github.com/llm-d/llm-d-kv-cache-manager/blob/main/README.md).
- Applicability: P/D disaggregation is a ≥2-GPU answer; on ONE GPU the same literature's answer for
  prefill/decode interference is chunked prefill (A1/A2) — which we run. KV-aware routing = our pick_slot
  held-prefix policy at single-instance scale (server.py:2107). The transferable half is observability
  practice: queue depth, admission wait, cache-hit and saturation counters exported per request — see (d):
  our server exports vLLM-named TTFT/ITL/E2E histograms TODAY (serve/prometheus.py:16-22, 39-47, 96-98;
  observe call server.py:5142; /metrics render server.py:6000) and the probe never reads them.

### A9. Cold start / warm start: the 2025-2026 practice, mapped
- Sources: [arXiv:2606.07362: Breaking the Ice — vLLM cold-start anatomy (2026-06)](https://arxiv.org/pdf/2606.07362)
  (startup decomposes into weight load, compile, graph capture; page cache explicitly part of the warm story);
  [vLLM sleep mode / zero-reload model switching (2025-10-26)](https://vllm.ai/blog/2025-10-26-sleep-mode);
  [Run:ai model streamer (NVIDIA blog)](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer);
  [CRIU for vLLM cold start (forum)](https://discuss.vllm.ai/t/using-criu-to-reduce-cold-start-latency-for-llm-tasks/639);
  [route179: model streaming + compile caching (2026-06)](https://route179.dev/2026/06/30/optimizing-vllm-cold-start-with-model-streaming-and-compile-caching).
- Mapping: their four cost buckets are ours: weight load = the 46-60 GiB shard read (T26-6 §a.1.4),
  compile/capture = our lazy cohort batch-graph captures (T17B §1.4), page cache = the
  pin_depends_on_reclaim → drop_file_cache back-to-back-boot law (expert_source.cpp:100-102,
  3511-3513, 3474-3484), engine init = server ready sequence. `prewarm2.sh` addresses page cache +
  captures + pools in one package — the frontier confirms the shape and adds nothing cheaper for a
  dedicated box (sleep mode/CRIU target multi-model swap, not our case). CONFIRMED.

### A10. Reproducibility: batch invariance (Thinking Machines, 2025-09) and its critique (2026)
- Sources: [Defeating Nondeterminism in LLM Inference (Thinking Machines, 2025-09)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/);
  [batch_invariant_ops repo](https://github.com/thinking-machines-lab/batch_invariant_ops);
  [vLLM batch-invariance docs](https://docs.vllm.ai/en/latest/features/batch_invariance/);
  [arXiv:2601.17768 (2026): critique — batch invariance as a poor fit for real serving, alternatives](https://arxiv.org/html/2601.17768v1).
- Mechanism: temperature-0 outputs vary with batch composition because reduction order varies; batch-
  invariant kernels restore bit-exactness at a throughput cost. The 2026 critique argues the cost is not
  worth it for serving and proposes verified-alternatives instead.
- Applicability: the full mechanism is DEAD for us (custom int8-expert graphs; bit-perf contract). The
  cheap residue is a DETECTOR, not an invariant: our probe already issues a deterministic request every
  run (probe-lane.py:67 — 'Say OK.', 8 tokens, temperature **0**) and prints only the token count. Hashing
  that response text per boot gives a golden-drift alarm (engine binary, pack, sampler config, template)
  for ~3 lines of Python in the probe harness. NEW, trivial. Premium-feel: reproducibility made observable.

### A11. KV-offload admission policy: EfficientAgent (2026-09) — a negative for us, a citation for keeps
- Source: [arXiv:2609.33762 (2026-09-27)](https://arxiv.org/html/2609.33762v1): stack-distance model sizes
  the KV host tier and locates the capacity transition; agentic reuse measured at 98.1% of prompt tokens
  already in a task's prefix cache (their workloads, their hardware).
- Applicability: our conversation cache (49,152 MiB, LCP across parked+active, conversation_cache.hpp:187,
  254-259) does not thrash at 6 clients; `reused_tokens` is already totaled (server.py:816). The paper's
  policy machinery (write admission, stack-distance sizing) targets regimes ours isn't in. KEEP what we
  run; the transferable residue is one line in (d): trend reused/prompt ratio per lane so a future
  workload change that DOES thrash the pool is visible.

### A12. MoE-offload serving papers (2025-2026): the frontier arrives where we already live
- Sources: [PreScope, arXiv:2509.23638 (2025-09)](https://arxiv.org/html/2509.23638v1) (predictive prefetch
  of next-layer hot experts); [adaptive offloading, arXiv:2509.08342 (2025-09)](https://arxiv.org/html/2509.08342v1);
  FineMoE (EuroSys'26) and CommitMoE (AAAI'26) fine-grained/fallback-free offloading (surfaced via search,
  no preprint numbers verified this session); [llama.cpp discussion #28248: persistent expert slot pool](https://github.com/ggml-org/llama.cpp/discussions/28248).
- Applicability: predictive prefetch = ForesightSwap's class, family-measured 0.91-0.98x and left off
  (Xolotl DETAILS.md:1736-1738; T25-3 §3.4) — our 2.0-2.3% miss hidden under cohort windows makes it
  worse, not better. Fallback-free two-tier (VRAM + pinned RAM, no NVMe in the steady path) is what we
  already run (T26-6 §b.1). No port candidate found; see (e).

---

## (b) APPLICABILITY TABLE — candidate → our file:line → effect / cost / falsifier / probe metric

| # | candidate | our anchor | expected effect (marked) | cost | cheapest falsifier | probe metric that decides |
|---|---|---|---|---|---|---|
| B1 | decode-share raise 0.5→0.75 (staged in t26-a-slate.sh) | generate.cpp:13696-13704 (static, default 0.5), 14185-14191 (budget windows) | +1-4% conc (EST, T26-4 §6); plus TTFT-variance down per A2 | S (env) | one 3v3 ABAB boot, staged campaign/t26-a-slate.sh | conc tok/s AND per-request TTFT p99 from /metrics; M1/pair no-regress |
| B2 | PIPED-for-cohorts port (reader thread beside reactor) | generate.cpp:12024-12025 (piped gate), 13838-13850 (refusal + branch), 13913-13971 (doorbells); verify.cpp:3935-3985 (batch_serve) | +3-6% conc net (EST; gross +6-8% per T17B #4) | M | one 3v3 boot with the port + `--pcie-mode kernel` (gate requires it: batch_read.hpp:48 ← generate.cpp:13838) | conc tok/s; admission share of wall from the same boot's log; token-equality gate |
| B3 | TWP-style admission pacing (NEW) | `_admit_next` server.py:1719-1729; signals: ms/window log lines (probe-lane.py:137), live.queued/waiting/admitting server.py:4247-4253 | stable TTFT + lower p99; ~0 conc at 6 fixed clients (EST); value at 7+ and variance | S-M (Python only) | 7-client burst lane, pacing env on/off | TTFT p99 + ITL histogram deltas from /metrics (not means) |
| B4 | VTC-lite admission fairness (NEW) | F-14 tickets server.py:1141-1143; `_take_control` 1655-1690; short-first precedent 310-317, 1670-1688 | queued-TTFT fairness (max/median TTFT ratio down; EST, no number) | S-M (Python only, ~40 lines) | 7-client lane, two clients of different token weight | per-client TTFT spread from /metrics; conc 0 by construction |
| B5 | Probe-side metrics harness (scrape /metrics; golden hash) — see (d) | /metrics render server.py:6000; histograms prometheus.py:39-47 + observe server.py:5142; golden request already sent at temp 0, probe-lane.py:67 | zero scoreboard; makes every other row measurable | S (probe-script only, ~20 lines) | run it once on a standing boot; confirm histograms non-empty and hash stable across 2 boots | n/a — this is the instrument |
| B6 | Clock/temp/power telemetry per lane leg — see (d) | results files carry no clock context today; epochs 2835 vs 3135 proven post-hoc (F-253/254; transcript) | makes the quiet-window re-bank auditable | S (nvidia-smi scrape in battery scripts) | one rebank leg with the scrape; compare banked tok/s against recorded sm_clock | clock delta vs tok/s delta correlation |
| B7 | maya fair-slice (PAUSE verb) | control verbs generate.cpp:12258-12263 (BSTOP/BYIELD only); BYIELD give-way 13710-13713, 13801-13810; server side server.py:1645 (YIELDS_MAX=2), 1744-1747 | TTFT 60→10 s class (their #93, T25-3 §4.1); 0% conc | M-L | only if the 7-12-client lane is chartered; B3/B4 are its cheap precursors | queued-TTFT under a long answer + short arrival |
| B8 | elastic KV + parallel-8 (4+4) | kv-grow gate generate.cpp:5194-5205 (batch carve); kv-resident streaming live; kVerifyMaxT=8 verify_kernels.hpp:21 | standalone ~0-2% conc (EST, T26-4 §3.3); enabler of +15-25% class | M-L | unchanged: KV-depth instrumentation first; funding from vram-reserve, not the cache (F-183 slope) | conc at 8 clients vs 6; per-request KV depth log |
| B9 | prewarm2.sh fast-warm package | campaign/prewarm2.sh (staged, never run, AUDIT-2026-10-18.md:263); drop law expert_source.cpp:100-102, 3511-3513 | +3-5% five-boot means, +8-13% first leg (EST, T26-6 §a.2) | S (ops only) | one cold-vs-warm boot pair | first `file tier I/O` MB line ≈ 0; pool term ≤2.4 ms/window; captures before probe |
| B10 | aux_cpus=1 diagnostic | generate.cpp:5926-5943; server.py:1404-1409 | ≤0.1 ms/window-class (EST, T26-6 §c.4) | S | one diagnostic leg | ms/window distribution shift; no scoreboard expectation |
| B11 | golden-hash boot alarm (from A10) | probe-lane.py:67 (temp-0 request exists); serve/chat_golden.json is template-golden only (checked) | catches engine/pack/sampler drift per boot | S (~3 lines in probe) | hash across two standing boots | hash equality; any change is an investigation, not a regression |

---

## (c) ROADMAP AUDIT — verdict per item

| item | verdict | evidence |
|---|---|---|
| **STRATA_BATCH_DECODE_SHARE raise** (default 0.5 → 0.75 staged) | **CONFIRMED** | vLLM V1 made decode-priority-under-budget the permanent single-GPU shape (A1: docs + issue #18547); Sarathi lineage says the interleave shape is where p99 lives (A2). Our knob is the time-share form of the same law (generate.cpp:13696-13704). Falsifier already staged (t26-a-slate.sh). AMENDMENT to the audit protocol: read TTFT p99 alongside conc — a flat conc with better p99 is a frontier-endorsed win (A2), and the share raise trades read latency for decode share, so TTFT is the canary. |
| **Reader-thread PIPED port** (M cost, +3-6% net) | **CONFIRMED** (unchanged mechanism; second independent confirmation) | SGLang's overlap scheduler (A7) and vLLM V1's fused loop (A1) are the same principle at cluster scale; basalt ships it always-on (T26-4 §1.1). Our insertion spec (T26-4 §3.1: gate beside :13850, doorbells iterate {ver, ver_b}, reader = sp.run :13986-14001) stands. The dma→kernel requirement is unamended (batch_read.hpp:48 ← generate.cpp:13838; STRATA_DMA_BATCH=0 does not clear the gate). |
| **Elastic KV + parallel-8** (4+4 fits kVerifyMaxT=8) | **CONFIRMED** as-is; nothing online supersedes it | Frontier's P/D-disagg answer needs ≥2 GPUs (A8) — its single-GPU equivalent (chunked prefill) we already run; the KV-funding question is local (F-183 slope; kv-resident 97-99% VRAM hit, boot log :94/:103 via T26-4 §2). Dynamo/llm-d add nothing to a one-box KV plan. Priority order unchanged (enabler, not standalone). |
| **ADMIT_SHORT_FIRST** (TTFT-only at 7+) | **CONFIRMED**, and **AMENDED upward in value** as the precedent for B4 | The 2025-2026 fairness frontier (A4) says order-only fairness is a legitimate lane (VTC), not a consolation prize: short-first is the trivial member of the VTC family. Arming it remains env-only (server.py:310-317, 1670-1688). |
| **maya fair-slice** (PAUSE-verb work) | **CONFIRMED** (fairness-only; PAUSE gap real) with a **NEW cheaper sibling** | Pascal/Andes (A5) independently converge on in-flight preemption as the QoE mechanism — the PAUSE verb remains the only route to slicing a DECODE. But A4 (VTC) and A3 (TWP pacing) show the ADMISSION side of the same problem is solvable in Python without any engine verb (B3/B4). May the slice stay charter-gated on the 7-12-client lane; B3/B4 are its new, cheaper precursors. |
| **prewarm2.sh** (fast-warm package, +3-5% five-boot means EST) | **CONFIRMED** | vLLM cold-start anatomy (A9) decomposes startup the same way our T26-6 §a does (weight load / compile-capture / page cache); their remedies (persistent caches, capture caching) map onto page-cache re-warm + the capture-forcing burst that prewarm2 already contains. Sleep mode/CRIU/streaming add nothing for a dedicated single-model box. Run it as the staged same-session post-boot probe. |
| **aux_cpus=1 diagnostic** | **CONFIRMED** (diagnostic only) | No frontier signal found either way (no paper or release note prices host-core topology for CPU-offload MoE decode); it stays a cheap diagnostic with no scoreboard expectation (T26-6 §c.4). |
| **Quiet-window re-bank at 3135** | **CONFIRMED and AMENDED**: re-bank WITH clock telemetry | The 2835→3135 epoch drift is exactly the failure mode the observability literature warns about (banked absolutes silently re-based, A8/Dynamo metrics practice; our own F-253/254). The re-bank should record sm_clock/temp/power per leg (B6) so the next epoch is detected by the artifacts, not post-hoc. Zero-risk addition to the battery scripts. |

---

## (d) STABILITY / RELIABILITY: metrics we do not track today (and the cost of each)

The server-side machinery largely EXISTS; the measurement culture has not plugged into it. That is the
headline: **the premium-feel gap is a probe-side gap, not a server-side one.**

1. **TTFT distribution per request — exists server-side, unused by the probe.** Latencies histograms
   (TTFT/ITL/E2E, vLLM bucket edges) are observed per finished request (server.py:5142) and exported at
   /metrics in vLLM names (prometheus.py:39-47, 96-98; render server.py:6000). probe-lane.py records only
   whole-request wall time (probe-lane.py:28-33) and engine-log means (TREND ms/window is an early/late
   MEAN, :126-130). Cost: ~15 lines to scrape /metrics before/after each lane and print p50/p99. Decides
   B1's TTFT canary and any stability claim.
2. **ITL (inter-token latency) variance** — same histograms, same scrape. The premium feel of "smooth"
   streaming is exactly the ITL histogram; nothing in campaign/results contains it today.
3. **Queue depth / admission wait time series** — live counters exist (`queued`, `waiting`, `admitting`,
   server.py:4247-4253; QueueWaitExpired path server.py:391, 1663) but the probe's conc_run joins threads
   and reports one wall (probe-lane.py:35-45). Add per-request submit→first-byte timestamps in the probe
   client itself (zero server involvement): TTFT at the client, which is where premium feel lives.
4. **GPU clock/temp/power per lane leg** — nothing in campaign/results carries clock context; the
   2835→3135 epoch was discovered after the fact (F-253/254). One `nvidia-smi --query-gpu=clocks.sm,
   temperature.gpu,power.draw` scrape per probe leg (B6) makes every banked number self-describing.
5. **Per-boot golden hash** — the probe already fires a temperature-0, 8-token request every run
   (probe-lane.py:67) and throws the text away. Hash it (B11): a free determinism alarm for engine
   binary/pack/sampler/template drift, the cheap residue of the batch-invariance line (A10). Note:
   serve/chat_golden.json is a chat-TEMPLATE golden, not a generation golden (checked this session) — the
   gap is real.
6. **Admission-phase duration** — BGEN→first-BT per request is derivable from the engine log the probe
   already parses (probe-lane.py:101-119 stops at the batch-summary regex). This prices the admission
   share of wall on the CURRENT cohort recipe — T26-4 §7's top unknown, answerable with zero boots once
   parsed.
7. **ms/window p99 (not mean)** — the rows are already in the probe's parsed list; p99/max per request is
   a one-line change to the TREND block (:126-130). The degradation signal currently hides inside a mean.
8. **Stability incident counters** — restarts, watchdog trips, QueueWaitExpired, BGEN refusals (#1994
   retry counter server.py:1600-1604): all logged, none summarized. A boot-level tally line costs
   nothing and turns "did the leg have an event?" from archaeology into a field.

What we would gain: with 1-8 added (one session of probe-script + battery-script work, no engine or
server code), every A/B on the board acquires a stability readout (TTFT p99, ITL variance, clock context,
golden hash), the re-bank becomes auditable, and the cold-start law gets a per-boot receipt.

---

## (e) NEGATIVE CANDIDATES — look strong, die here (with reasons)

1. **Prefill/decode disaggregation (Dynamo-style P/D)** — needs ≥2 GPU instances; on one GPU the same
   literature prescribes chunked prefill, which we run (A8; our piecewise read, generate.cpp:14185-14191).
   No second GPU to send prefill to. DEAD.
2. **Batch-invariant kernels for bit-reproducibility** (A10) — requires rewriting every reduction to
   composition-invariant form; our graphs are hand-built around the int8 expert path and the bit-perf
   contract; the 2026 critique itself argues the serving cost is not worth it even at cluster scale
   (arXiv:2601.17768). We take the golden-hash detector instead. DEAD as kernels; B11 is the residue.
3. **Pascal-style multi-instance phase migration** (A5) — one box, one instance; the placement/migration
   half has no object to act on. The preemption half is maya-slice, already priced. DEAD as migration.
4. **PreScope-class predictive expert prefetch** (A12) — ForesightSwap's class, family-measured 0.91-0.98x
   and left off on three boxes; our 2.0-2.3% miss hides under cohort windows. New papers do not change the
   arithmetic. DEAD without a new miss-rate regime (would need >5-10% steady miss).
5. **EfficientAgent stack-distance sizing / write admission for the conversation cache** (A11) — our pool
   is 49 GiB against a 6-client working set with LCP across parked+active; no thrash signal exists
   (reused_tokens totals healthy). Sizing policy solves a problem we do not have. DEAD; keep the
   reused/prompt trend line as a tripwire.
6. **vLLM sleep mode / CRIU restore / weight streaming for boot** (A9) — all target multi-model or
   serverless swap-in-seconds; our boot is minutes on a dedicated box and the binding cost is the
   page-cache drop law, which prewarm2 already addresses operationally. DEAD for this box.
7. **Full TWP closed-loop controller at exactly 6 fixed clients** — no queue exists to pace; the loop
   degenerates to a constant. Keep B3 as a 7+-lane artifact, not a scoreboard candidate. DEAD at 6.
8. **SuperInfer (A5 list), KVCompass/cluster prefix-sharing planes (arXiv:2610.24179)** — superchip
   hardware, and multi-replica cache discovery respectively; neither object exists here. DEAD.
9. **More slots as a latency fix (llama.cpp folk wisdom "more slots = worse per-request latency")** —
   consistent with F-249 (COHORTS>3 flat at 6 clients); the frontier folk record CONFIRMS the measured
   negative. Stays dead at 6 clients.

---

## (f) CONFIDENCE GRADING

- HIGH: the observability gap itself (metrics exist server-side, probe never reads them — read at source
  this session: prometheus.py, server.py:5142/:6000, probe-lane.py whole); the decode-share verdict (frontier
  A1/A2 + in-tree mechanism); prewarm2 confirmation (T26-6's own code trace + A9's decomposition); the
  negative list items 1, 3, 4, 9 (mechanical or family-measured).
- MEDIUM: B3/B4 sizing and shape (mechanisms published, magnitudes theirs-not-ours; our Python port is new
  code); the PIPED port's +3-6% band (carried from T26-4 §7, admission share on the CURRENT recipe still
  unmeasured — item d6 answers it); prewarm2's +3-5% band (EST, never split after T18 killed the learned
  lever).
- LOW: any external throughput percentage transfer (cluster numbers on H100/B200 classes do not scale to a
  5090 + CPU expert pool; used here only for direction); FineMoE/CommitMoE specifics (no preprint numbers
  verified this session — cited as surfaced-by-search only); B6's exact variance attribution (clock vs
  other epoch effects) until the first instrumented re-bank leg.

## SOURCES (external; all fetched or searched this session)

A1: https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture ;
https://docs.vllm.ai/en/stable/configuration/optimization ; https://github.com/vllm-project/vllm/issues/18547 ;
https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1
A2: https://arxiv.org/html/2403.02310v2 ; https://www.usenix.org/conference/osdi24/presentation/agrawal ;
https://www.zhongzhuzhou.org/blog/2026-05-20-sarathiserve-technical-review-en
A3: https://arxiv.org/abs/2610.17689
A4: https://www.usenix.org/conference/osdi24/presentation/sheng ; https://arxiv.org/pdf/2401.00588 ;
https://arxiv.org/abs/2501.14312
A5: https://arxiv.org/abs/2404.16283 ; https://arxiv.org/abs/2602.11530 ;
https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf
A6: https://infini-ai-lab.github.io/MagicDec ; https://arxiv.org/html/2408.11049v5 ;
https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference
A7: https://www.lmsys.org/blog/2024-12-04-sglang-v0-4 ;
https://sgl-project.github.io/advanced_features/piecewise_cuda_graph.html ; https://github.com/sgl-project/sglang/releases
A8: https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models ;
https://docs.nvidia.com/dynamo/knowledge-base/concepts/system-architecture/disaggregated-serving ;
https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference ;
https://llm-d.ai/blog/kvcache-wins-you-can-see ; https://github.com/llm-d/llm-d-kv-cache-manager/blob/main/README.md
A9: https://arxiv.org/pdf/2606.07362 ; https://vllm.ai/blog/2025-10-26-sleep-mode ;
https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer ;
https://discuss.vllm.ai/t/using-criu-to-reduce-cold-start-latency-for-llm-tasks/639 ;
https://route179.dev/2026/06/30/optimizing-vllm-cold-start-with-model-streaming-and-compile-caching
A10: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ ;
https://github.com/thinking-machines-lab/batch_invariant_ops ;
https://docs.vllm.ai/en/latest/features/batch_invariance/ ; https://arxiv.org/html/2601.17768v1
A11: https://arxiv.org/html/2609.33762v1
A12: https://arxiv.org/html/2509.23638v1 ; https://arxiv.org/html/2509.08342v1 ;
https://github.com/ggml-org/llama.cpp/discussions/28248
(e)8: KVCompass https://arxiv.org/abs/2610.24179 (surfaced via arXiv listing fetch)

Internal anchors (all read this session, working tree): generate.cpp:12024-12025, 12258-12296,
13690-13720, 13830-13860; verify.cpp:3896-3990; serve/server.py:391, 421-427 area (:263 QUEUE_BEAT_S),
1132, 1141-1143, 1600-1604, 1655-1690, 1719-1729, 1941-1943, 1988, 2107, 3554-3555, 4229-4290 (metrics),
5142, 6000; serve/prometheus.py:1-98; serve/ready.py:1-30; serve/chat_golden.json (template golden only);
campaign/probe-lane.py:1-153; campaign/prewarm2.sh, campaign/t26-a-slate.sh (staged, never run);
expert_source.cpp:100-102, 3474-3484, 3511-3536 (per T26-6, re-anchored).
