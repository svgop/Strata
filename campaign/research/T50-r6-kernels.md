# T50 / R6 — Raw kernel & graph efficiency: shrinking the ~1,800-node verify window

**Researcher:** R6 (kernel/graph lane). **Date:** 2026-10-25.
**Box:** RTX 5090 (sm_120 consumer Blackwell, 32 GB, CUDA 13.4), Ryzen 9950X3D.
**Engine facts used for costing:** verify window = CUDA-graph-captured ~1,800 nodes; measured ~9.5 µs node-to-node spacing carrying ~2.3 µs of real GPU work; GPU-side window term 9.0–10.4 ms flat across depths; int8 MMA 732 vs fp16 233 (3.1×); PDL arm measured neutral; 60 MiB L2 set-aside over GDN state; solo 223 mean / 234 best t/s. Goal +30–50%.

---

## 0. Derivation preamble — the window arithmetic we must beat

1,800 nodes × 9.5 µs = 17.1 ms, but the measured window is 9.0–10.4 ms. Therefore the graph is NOT a pure serial chain: breadth (the per-layer shared-expert fork `sh_fork`, batched multi-row launches, per-row QSA loops — see F-204-node-inventory.md) hides roughly half the nodes. The consistent model is:

- **Critical path** N_crit ≈ 9.5 ms / 9.5 µs ≈ **950–1,100 nodes**.
- Overhead term: 9.5 µs − 2.3 µs ≈ **7.2 µs/node of pure structure cost** (graph node dispatch, launch-to-launch latency, flag-wait spins, copy/memcpy glue, fork/join edges) ≈ 7 ms ≈ **~70–75% of the window**.
- Sanity check against the literature: Hazy Research measured kernel→kernel at **~2.1 µs on a raw stream and ~1.3 µs inside a captured CUDA graph** (H100). A healthy captured graph moves kernel-to-kernel in ~1.3 µs; ours spends 9.5 µs per node-step. **Our window has ~7× the per-node structural overhead of a well-formed graph** — the 1,800-node window is not intrinsically expensive, its *shape* is.

So every proposal below is scored as: **Δwindow ≈ ΔN_crit × 9.5 µs**, plus any change to the per-node constant.

Repo anchors (all read-only refs): `src/core/graph.cpp` (GraphRegistry, 146 lines: capture → instantiate → launch → bounded spin), `src/core/verify.cpp` (`record_window`, doorbell/ring, per-layer `sh_fork`, GPU-side waits #267, separate commit graph), `src/kernels/cuda/verify_kernels.cu` (2,154 lines), `campaign/research/F-204-node-inventory.md` (per-layer node counts + top-10 fusion list), gates `STRATA_LFUSE` (verify.cpp:90), `STRATA_FUSE_WAIT`, `STRATA_QSA_SLOTS`, `STRATA_DF_PLE`; existing probes `tests/core/host_graph_launch_test.cu`, `src/kernels/pdl_parity.cpp`, `src/kernels/decode_cluster_parity.cpp`.

---

## 1. Per-source findings

### Arxiv papers

**A1. MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs — arXiv:2512.22219** (Dec 2025, v2 Jun 2026; CMU/Meta lineage — Mirage/OSDI'25 arXiv:2405.05751 follow-up). [Confidence: HIGH — top-group authors, open source (mirage-project/mirage)]
- **Mechanism:** lowers model inference into an *SM-level task graph* (dependencies at individual-SM granularity), then runs it in a single persistent megakernel with decentralized per-SM scheduling; cross-operator software pipelining and compute/communication overlap impossible under kernel-per-operator.
- **Numbers:** up to **1.7× lower end-to-end inference latency** vs kernel-per-operator serving systems (abstract); hardware not in abstract page (lineage targets H100-class). The launch-overhead-elimination mechanism is exactly our disease.
- **Why it matters here:** it is the existence proof that an entire multi-layer decode pass can be one resident kernel with dependency edges costed at ~ns (counters), not ~µs (graph nodes).

**A2. Hazy Research, "Look Ma, No Bubbles!" (May 27 2025) + Megakernels repo (HazyResearch/Megakernels)** — counted here as the empirical megakernel source. [Confidence: HIGH — full measured breakdown published, code released]
- **Mechanism:** whole Llama-3.2-1B bf16 bs=1 forward as one persistent kernel, on-GPU interpreter, per-SM pre-scheduled instruction streams, paged SMEM (13×16 KiB pages, explicit request/release so weight loads pipeline across instruction boundaries), **counter arrays in global memory for dependency sync, 4-chunk MLP pipelining** (down-proj starts as soon as its chunk is ready).
- **Numbers (the ones that matter to us):**
  - Stream launch ~**2.1 µs**; CUDA graph **~1.3 µs** kernel→kernel.
  - **PDL tried and rejected**: `cudaGridDependencySynchronize` is too coarse (must wait for all of QKV before any attention head). Our "PDL neutral on 13/1,860 nodes" result is the same physics at our scale.
  - B200 680 µs/fwd-pass decomposition: ~500 ns load/store latency per instruction; **async barrier ~60 ns even in pass state**; warp-sync overhead 40 µs of 600.
  - H100: 78% of memory bandwidth vs ≤50% for vLLM/SGLang; ~2.5× vs vLLM.
- **Directly transferable:** counter-array sync + chunked producer/consumer = the replacement for our flag-wait and doorbell glue nodes.

**A3. Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search — arXiv:2605.11581** (May 2026). [Confidence: MEDIUM-HIGH — abstract with numbers, commercial deployment claimed]
- **Mechanism:** under a fixed deployment config the optimal megakernel path is unique ⇒ hoist all runtime branching to compile time ("solidified" execution path); 3D SMEM constraint model + K-dim splitting (−50% peak SMEM); MLIR offline DAG search; hybrid engine (megakernel decode + TRT-LLM prefill).
- **Numbers:** decode **kernel launch overhead = 14.6% of end-to-end decode time** on their baseline; single-batch **+23.6% over vanilla TRT-LLM** and +50.2% over vLLM on an L20.
- **Relevance:** corroborates megakernel gains on a *small-datacenter* GPU (L20 ≈ Ada-class); "compile-time solidification" is the right doctrine for us too — our verify depths are fixed at capture time.

**A4. ForgeMegakernel: A General Framework for Efficient Auto-Regressive Model Decode Megakernels — arXiv:2609.12379** (Sep 2026). [Confidence: MEDIUM — abstract-level numbers only]
- **Mechanism:** agent-generated per-model megakernels; per-SM instruction streams, **dependency counters instead of global synchronization**, shared-memory buffer pool for load balance, mid-state test oracle (performance + error + precision checked mid-generation).
- **Numbers:** 50.5–85.9% memory-bandwidth utilization across 14 decode ops / 8 model families (0.6B–13B); geomean **1.21× vs SGLang 0.5.18, 1.54× vs a megakernel compiler**.
- **Relevance:** independent replication of A1/A2 mechanics (counters, per-SM streams) with a precision oracle — the same bit-exactness discipline our doctrine requires.

**A5. Microbenchmarking NVIDIA's Blackwell Architecture — arXiv:2512.02189** (v3). [Confidence: HIGH for B200 numbers; explicitly NOT sm_120]
- **Numbers:** B200 = 148 SMs, 4 L2 partitions (2× Hopper); tcgen05 MMA at m64n8k16: **INT8 3,928 TOPS**, FP8 1,913–3,851, FP4 up to 7,700; **FP32 accumulation halves FP16 tensor rate** (964.8→482.4) — accumulator datapath is the limiter; instruction latency flat 11.2–12.6 cycles across precisions; **TMEM = 256 KB/SM (512 cols × 128 lanes)**; STREAM Triad ~4.14 TB/s = 52% of peak.
- **Critical negative result for our box:** the whole study is sm_100; **nothing tcgen05/TMEM applies to sm_120** (confirmed by ecosystem: FA4 does not support consumer Blackwell; FlashInfer routes RTX 5090 attention to cuDNN open-source kernels). Our in-tree non-tcgen05 int8 mma.sync path (732 vs 233) is already the correct consumer-Blackwell play; **drop any tcgen05/TMEM weight-stationary MoE lane — it is datacenter-only**. Also: their "memory bandwidth + launch overhead dominate once tensor cores run at 96–99% of peak" is our thesis in one line.

**A6. Blink: CPU-Free LLM Inference — arXiv:2604.07609** (Apr 2026). [Confidence: MEDIUM — abstract numbers; mechanism from abstract + NVIDIA device-graph-launch lineage]
- **Mechanism:** host scheduling replaced by a **persistent GPU kernel** doing batching/scheduling/KV management; inputs land via RDMA from a SmartNIC. Follows NVIDIA's device-graph-launch/conditional-node line (CUDA 12.0+ device graph launch; vLLM's conditional-node warning path).
- **Numbers:** P99 TPOT up to 3.40×, decode throughput up to 2.1× vs TRT-LLM/vLLM/SGLang; stable under CPU interference where baselines degrade up to 100×.
- **Relevance:** the "no host in the decode loop" end-state. Our window still has host-flag GPU spins and a host-launched commit graph — Blink/MPK say those round-trips are deletable.

### Production sources

**P-a. NVIDIA TensorRT-LLM tech blog: "Pushing Latency Boundaries: DeepSeek-R1 on 8×B200".** [Confidence: HIGH — full progression table]
67→**368 TPS/user** (5.5×) at bs=1-ish min-latency. Piecewise attribution: CUDA graphs "necessary to overcome CPU overhead for small workloads"; **PDL: 168→173 TPS = +3%** (matches our neutral arm — PDL is a µs-band-aid, not structural); **multi-stream overlap** shared experts + RMSNorms (+6-7 each step); **sparse experts as plain GEMMs with output masking** (grouped-GEMM dispatch/index glue deleted — redundant compute is free because memory-bound); custom oneshot AllReduce **fused with RMSNorm+quant**; grouped RMSNorm; custom RouterGEMM for m≤30; bf16 attention because fp8 "doesn't help at this scale". Lesson: their biggest single wins were **glue-node deletion and stream reshape**, not per-FLOP work.

**P-b. NVIDIA CUDA docs/blogs: device graph launch (CUDA 12.0) + conditional nodes (12.4+) + capture-to-graph (12.8+) + CUDA 13.0/13.1.** [Confidence: HIGH]
Device-side graph launch latency is **>2× lower than host launch and constant regardless of graph topology**; tail-launch creates in-graph loops ("effectively a loop in the execution flow"); fire-and-forget vs tail modes; restriction: one pending launch + one self-relaunch per graph. Conditional (IF/WHILE) nodes embed data-dependent control flow inside a single graph — vLLM literally warns "conditional node support … CUDA graphs with while loops are disabled, decoding speed will be slower" when unavailable, i.e., production treats while-loop-in-graph as the decode-correct form. `cudaStreamBeginCaptureToGraph` (12.8+) lets us graft conditional/tail-launch nodes onto our existing stream-captured window. CUDA 13.1 exposes **green contexts** (SM partitioning) at runtime API level. We are on CUDA 13.4: all of this is available on our driver today.

**P-c. llama.cpp CUDA-graph merge + follow-ups (ggml-org #6763/#6766, NVIDIA blog Aug 2024, #27009 Aug 2026).** [Confidence: HIGH]
Plain stream-capture of the decode sequence gave **+5–18% decode** on consumer boxes — from launch overhead alone, before any fusion. #27009: shape-stable ubatches so graphs are reusable across slot counts (our GraphRegistry `(LayerType, n_tokens)` key already does this). Discussion #17621: fusion to cut memory traffic + launch time is the standing consumer-side lever.

**P-d. FlashInfer release notes / FA4 sm_120 exclusion / cuDNN 9.x on RTX 5090.** [Confidence: HIGH]
FlashInfer: attention on RTX 5090 runs via **cuDNN's open-source kernels**; FA4 (tcgen05/TMEM) is sm_100-only. cuDNN 9.x fused flash is the supported sm_120 attention path. For us: relevant only as a cross-check against our custom int8-MMA QSA attention (split cells + LSE merge); cuDNN cannot absorb our indexer/top-k/doorbell glue, so it is not a window-shrinking lever.

**Checked and closed:** "Halfmoon" (the brief's candidate) is a SOSP'23 fault-tolerant serverless paper — not an LLM kernel line; no 2026 LLM paper by that name found. "NVFuser/PDFS" — no arxiv match under megakernel; the actual 2026 megakernel follow-ups are A3/A4.

---

## 2. PROPOSALS — costed against OUR numbers, ranked by expected value

### P2 — "Cheap tranche" graph surgery: execute F-204 items #2+#3+#7, then #1+#4+#5 (RANK 1: best gain-per-risk-week)

- **Mechanism:** pure node deletion inside the existing captured window, no new execution model. (i) `STRATA_LFUSE` shared-expert chain 6→3 (**−144**, implemented + bitwise-checked, gated at verify.cpp:90); (ii) fused wait+copy doorbell forms (**−144**, `fuse_wait_on()` currently hard-false on CUDA — un-gate); (iii) `native_ple_postops_batch_snap` (**−56**, exists, bitwise-equal). Then the new-kernel tranche: q8_1 quantize folded into producers (**−180**, per-form re-test — F-187's QFUSE was a mirror trade), router chain 4→2 (**−96**), hc read 3→2 (**−96**).
- **Costing:** cheap tranche −344 window nodes; these are per-layer *serial-chain* nodes (doorbell copies, shared-expert join points), so take ≥70% on the critical path: ΔN_crit ≈ −240 ⇒ **Δwindow ≈ −240 × 9.5 µs ≈ −2.3 ms** ⇒ window 9.7→~7.4 ms ⇒ **+15–25% verify throughput**. With the second tranche (−372 more, ~60% serial ⇒ −220): cumulative **window ~6.4–6.9 ms ⇒ +35–45%** — the campaign goal, with zero exotic machinery.
- **Risk:** LOW for bitwise outputs (LFUSE/PLE already bitwise-checked; quantize-fold needs per-form A/B per the F-187 caution). Graph capture while the other cohort is in flight is unchanged — same capture path, fewer launches.
- **Cheapest falsifier:** flip the three gates, run `depth-lane` + `bit-check.py`; `CapturedGraph::end()` already returns `nodes_` (graph.cpp:59) — assert node count drops ~−344/window and the window term drops ≥1.5 ms on the depth-lane harness. One afternoon.
- **Port effort:** days. Sites: `src/core/verify.cpp` (gates, record_window), `src/kernels/cuda/verify_kernels.cu` for the two new folded kernels (quantize-into-producer, router-top10-in-epilogue).

### P1 — Megakernel-lite: one persistent fused kernel per layer class (RANK 2: ceiling; the double-digit play)

- **Mechanism:** MPK/Hazy/Forge mechanics at our scale, without the full interpreter: a per-GDN-layer persistent kernel (39→~4 nodes: fused projections+norm step [exists as `gdn_step_norm_multi`], fused conv+recurrence, MoE apply, one doorbell-publish node) and per-QSA-layer persistent kernel (52→~8: projections; split-cell attention + **LSE merge moved in-kernel**; indexer block-scores+radix top-k; combine; doorbell). Inside each kernel, dependencies become **atomic-counter readiness + chunked handoff** (Hazy: counter arrays, 4-chunk MLP pipelining; barrier cost ~60 ns; instruction-to-instruction ~500 ns) instead of graph edges at 9.5 µs. Ada-MK's doctrine applies: verify depths are fixed ⇒ solidify the execution path at capture time, no runtime branching.
- **Costing:** window ~1,800 → ~40 layers × ~6 + prologue/epilogue ≈ **~280 nodes**; N_crit ~1,000 → **~170–200**. Window ≈ 200×9.5 µs (remaining graph spine) + 0.4–0.8 ms in-kernel serial ≈ **2.3–2.7 ms** vs 9.0–10.4 ⇒ **3.5–4.5× on the window term**. Even if only half that survives acceptance-rate and host-term dilution at the token level, this clears +30–50% with headroom; at the optimistic end solo 223 → 350+ t/s.
- **Risk:** MEDIUM. Bit-exact packing doctrine: every fused stage must reproduce bit-identical outputs — precedent exists (LFUSE, native_moe_combine_multi_hits_added) but attention+indexer in one kernel is a real bitwise surface. Resident-kernel × cohort-in-flight: a persistent kernel must not spin-wait on host flags the way the current graph does (#267 GPU waits) — needs the P3 control-flow arm or counter-based readiness fed by mapped-memory writes (which the doorbell already does). Register-pressure cliff if we naively fuse GDN+MoE: keep it per-stage (per-layer), not whole-window, first.
- **Cheapest falsifier:** hand-fuse ONE GDN layer's 39-node suite (gates make 90% of it exist), measure the fused kernel vs the captured 39-node chain on the same inputs: budget 39×9.5 = 370 µs vs ~90 µs of raw work — the fused kernel wins unless warp serialization bites (GDN steps are row-parallel across 170 SMs; hard to lose). A 4-layer mini-window via `GraphRegistry::record` gives the end-to-end number in a day.
- **Port effort:** 1–2 weeks per layer class (GDN first — fewer nodes, all pieces bitwise-proven; QSA second, harder). Sites: `verify_kernels.cu` (all step kernels), `verify.cpp` record_window body.

### P3 — Device-initiated graph launch + conditional nodes: delete the host from the window/commit boundary (RANK 3)

- **Mechanism:** today the window graph ends in doorbell/flag nodes the host observes, then the host launches the commit graph and the next window (verify.cpp window_run/commit; GPU flag-waits #267; graph.cpp `launch`+`wait_ms` spin). Move to: single self-relaunching graph using **tail launch** (`cudaGraphInstantiateFlagDeviceLaunch`, CUDA 12.0+, present in 13.4) or **WHILE conditional node** (`cudaGraphConditionalHandle`, 12.4+) grafted onto the capture via `cudaStreamBeginCaptureToGraph` (12.8+). The verify loop ("while verify not done: window; commit") becomes one GPU-resident graph; the ~100–150 wait/copy/glue nodes per window that exist only to handshake with the host collapse into conditional edges. Blink/A6 is the systems proof; vLLM treats while-loop-in-graph as the decode-correct form.
- **Costing:** node deletion ≈ −100 to −150 (mostly serial doorbell/handoff) ⇒ −0.9 to −1.4 ms; plus removal of 2–4 host round-trips per window (doorbell observe → commit launch → next window launch) worth another ~0.3–1 ms of latency and, more importantly, kills the inter-window flat overhead that keeps the window term 9.0–10.4 ms *even at depth 0*. Also shrinks the PDL-neutral surface problem: with 1,800→~1,650 nodes of pure compute spine, a second PDL attempt has more room. Expect **+10–20%** standalone, and it is a **prerequisite** for P1's full win.
- **Risk:** MEDIUM. Capture-while-cohort-in-flight is exactly the transition-bug class this touches: device-launch graphs must not be destroyed/re-instantiated while the other cohort's stream references them (`CapturedGraph::reset()` already documents that exec destroy BLOCKS — graph.cpp:36-45). One-pending-launch-per-graph restriction caps how many windows can be in flight GPU-side; the bounded `wait_ms` protocol needs a device-visible completion flag instead of `cudaEventQuery`. MTP/spec arm interplay needs re-audit.
- **Cheapest falsifier:** extend `tests/core/host_graph_launch_test.cu`: two-node graph → device tail-launch of a second graph; measure launch-to-launch on our driver vs host relaunch (expect ≥2× per NVIDIA; if our box shows no advantage at graph granularity, kill the arm). Half a day.
- **Port effort:** 1–2 weeks. Sites: `graph.cpp` (instantiate flags, launch protocol), `verify.cpp` (window/commit boundary, #267 waits), new conditional-node construction in the capture path.

### P4 — Critical-path reshape: stream/counter overlap of independent per-layer chains (RANK 4, folds into P1)

- **Mechanism:** TRT-LLM's multi-stream overlap (shared experts on own stream, two RMSNorms overlapped) already exists here as per-layer `sh_fork`; extend it: router chain ∥ hc read ∥ K/V append are independent per F-204; and replace flag-spin waits with counter-array readiness so a consumer kernel starts when *its* producer partition lands (Hazy chunk pipelining) instead of when the whole prior node retires.
- **Costing:** targets the 7.2 µs/node overhead constant rather than node count: moving 25% of serial edges to overlapped streams ⇒ N_crit −25% ⇒ **−2.4 ms**; counter-readiness inside P1 kernels is what turns their boundaries from ~500 ns into ~60 ns. Standalone +10–20%; compounding with P1/P2.
- **Risk:** LOW-MEDIUM (stream forking inside capture is proven here via sh_fork; counter waits change the hang-detection story — keep #267's bounded-wait semantics host-side).
- **Cheapest falsifier:** capture a 2-layer window with router/hc-read on a second stream; `cudaGraphGetNodes` depth check + depth-lane timing. One day. Sites: `verify.cpp` record_window.

### P5 — CUDA 13.1 green contexts: SM-partition the two cohorts (RANK 5: orthogonal insurance)

- **Mechanism:** runtime-API green contexts (13.1) pin the verify window's graph to an SM subset and the other cohort to the rest, stabilizing capture/replay timing and removing cross-cohort SM eviction from the spacing measurement (some of the 7.2 µs/node may be cohort interference noise, which would make every node above cheaper than billed).
- **Costing:** unknown sign/magnitude until measured; plausibly tightens the 9.5 µs constant by 5–15%. No node-count change.
- **Risk:** LOW; complements (does not replace) anything else. Cheapest falsifier: measure the same 100-node captured chain alone vs with a synthetic cohort running, green-context vs not. Sites: `src/core/device.cu`, `graph.cpp` stream creation. Effort: 2–4 days.

---

## 3. Ranking summary and doctrine notes

| # | Proposal | Δwindow (ms) | Solo t/s (est.) | Effort | Risk |
|---|----------|-------------|------------------|--------|------|
| 1 | P2 F-204 graph surgery | −2.3 (tranche 1) → −3.3 (both) | 223→~265–300 | days | LOW |
| 2 | P1 megakernel-lite (per-layer persistent kernels) | −7 ms (9.7→~2.5) | 223→300–450+ | 2–6 wks | MED |
| 3 | P3 device-launch/conditional window loop | −1.2 to −2.4 + enables P1 | +10–20% | 1–2 wks | MED |
| 4 | P4 counter/stream reshape | −2.4 standalone, compounds | +10–20% | 1 wk | LOW-MED |
| 5 | P5 green contexts | −0.5 to −1.5 (constant-term only) | +3–10% | 2–4 d | LOW |

Sequencing: **P2 immediately** (its kernels are P1's building blocks); **P3 falsifier in parallel** (half-day test decides the window-loop architecture); **P1 on GDN first** (its 39-node suite is 90% bitwise-proven), QSA second; P4 lands inside P1 by construction. Closed lanes: tcgen05/TMEM (sm_100-only — A5 + FA4/FlashInfer sm_120 exclusions), cuDNN attention swap (cannot absorb our glue; only a perf cross-check), PDL as a primary lever (neutral here; +3% on TRT-LLM's much healthier graph — A2 rejected it for coarseness; revisit only after P1 shrinks the surface).

The one-line thesis the sources unanimously support: our window is **launch/structure-bound at 9.5 µs/node against a ~1.3 µs achievable** (A2), launch overhead is 14.6% of decode even in production stacks (A3), and the two structural cures are **fewer, bigger resident kernels with counter dependencies** (A1/A2/A4) and **a GPU-resident loop with no host in the boundary** (A6/P-b). Bit-exactness is a discipline the megakernel literature already practices (A4's mid-state oracle), not a blocker.
