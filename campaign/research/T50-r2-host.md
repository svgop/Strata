# T50-R2 — HOST-SIDE SCHEDULING & ORCHESTRATION: killing or hiding the 4.72·T CPU term and the 9.5 µs node spacing

Date: 2026-10-25 (wave T50, researcher R2). Scope: READ-ONLY on the tree; a live server runs (no processes touched).
Deliverable: external prior art + costed proposals against OUR measured numbers.
Grading: **[M]** = measured-ours (tasking facts or banked F/T entries, anchors re-read this session); **[A]** = external,
deep-fetched this session (mechanism + numbers extracted from primary source); **[B]** = external, abstract/summary-level
verified; **[C]** = search-verified only; **[H]** = hypothesis (arithmetic shown, constants marked).

---

## §0 THE ONE-SENTENCE FINDING

**The CPU term is not a bandwidth term — it is a *scheduling* term.** ~182 missed experts × 1.94 MB = ~353 MB per
window, delivered in 13.56 ms = **26 GB/s aggregate** [M: spec §7 counts; arithmetic this session], against a
9950X3D + DDR5 multi-thread stream capacity of ~80–100 GB/s [H: marked constant]. The 15-worker pool re-forks per
layer, 48 times per window, in a hard ring→pool→release chain (session.cpp:949-984); the bytes say the whole tier's
floor is **~3.5–4.5 ms**, not 13.5. Every serious external system that runs our exact shape (CPU experts + GPU
experts, one MoE layer at a time) reached the same conclusion the same way: stop paying per-layer fork/join, give
the CPU tier a whole-window compute window, and let the GPU poll non-blockingly (HybriMoE, APEX, KTransformers,
DeepSeek's overlap scheduler, vLLM's async scheduling, Blink).

---

## §1 PER-SOURCE FINDINGS

### 1.1 Sarathi-Serve — "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" — arXiv 2403.02310, OSDI'24 — [A]

- Mechanism: chunked-prefills split a prompt into near-equal chunks capped by a per-iteration **token budget τ**
  profiled once from the TBT SLO (512 strict / 2048 relaxed); **stall-free batching** forms each mixed iteration in a
  fixed order — all ongoing decode tokens first, then one prefill chunk, then new admissions — so decodes never
  pause. Aligns chunk sizes to tile sizes (a 257-vs-256 chunk cost ~32% extra prefill time: tile quantization).
- Numbers: Mistral-7B / 1×A100-80G up to **2.6×** capacity vs Orca/vLLM; Yi-34B / 2×A100 **2.8×**; LLaMA2-70B /
  8×A40; Falcon-180B / 8×A100 TP4-PP2 **6.9×**. TBT inflation without chunking up to **28.3×**.
- Relevance to us: the *ordering principle* — the batch's fixed-cost members (decodes) pack first, the variable one
  (prefill chunk) takes the remainder — is exactly the discipline our conc lane's admissions violate under F-14 FIFO.
  Marginal for solo; grade carried for T47-conc context.
- Confidence: A (full HTML fetched; the per-iteration CPU scheduling cost is NOT quantified in this version — noted).

### 1.2 DistServe — "Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving" — arXiv 2401.09670, OSDI'24 — [A−]

- Mechanism: put prefill (compute-bound) and decode (memory-bound) on different GPU pools, each scaled to its own
  SLO; optimizes **goodput** (throughput meeting both TTFT and TPOT).
- Numbers: up to **7.4×** goodput vs collocated. Under collocation a single large prefill inflates TPOT **2–30×**
  even with chunked prefill (retro blog).
- 18-month retrospective (haoailab.com/blogs/distserve-retro, Nov 2025) [A]: P/D disaggregation is now "the default
  playbook" (Dynamo, llm-d, SGLang, vLLM, Mooncake, DeepSeek production, Rubin CPX); adoption lagged 2024 because the
  architectural refactoring cost is high; the win emerges at fleet scale, not single GPU.
- Relevance: the transferable idea is not two GPUs — it is **phase separation by resource character**. Our CPU-expert
  tier IS a second, much slower device already in the pipeline; we simply run it *synchronously per layer* instead of
  as a pipelined disaggregated stage. (Also: their measured refactoring-cost objection matches our F-135 lesson —
  architecture changes need their own proof lane.)
- Confidence: A− (paper via search + USENIX; retro deep-fetched).

### 1.3 SGLang — "Efficiently Programming Large Language Models using SGLang" — arXiv 2312.07104 — [B+]

- Mechanism: RadixAttention (automatic KV reuse via radix tree, LRU) + an **overlap scheduler** that overlaps CPU
  scheduling with GPU compute so the CPU is never the gap.
- Numbers: 2–5× throughput where prompt overlap dominates; 1,600+ citations. A 2025 issue on sgl-project/sglang
  reports *disabling* the overlap scheduler moves H20 Mean TTFT by ~32% — the mechanism is still load-bearing in
  production three years on.
- Relevance: precedent that one-step CPU-ahead scheduling survives contact with production; our C2 conc-lane overlap
  is the same family (carried [C: T47-conc S1]). The solo lane's analogue is §3-P5 below (window-boundary tails),
  which our tree does NOT yet do.
- Confidence: B+ (paper summary + issue search-verified).

### 1.4 APEX — "Parallel CPU-GPU Execution for LLM Inference" — arXiv 2506.03296 — [A]

- The closest academic match to our architecture (dense variant). Mechanism: GPU runs shared linear layers for the
  whole batch; CPU computes the offloaded (KV-in-DRAM) attention; sync is **delayed result sync** — CPU results for
  layer *i* are only handed back just before the GPU needs them for post-attention linears of layer *i* **next
  iteration**, giving the CPU roughly the GPU's full iteration as its compute window. The GPU polls **non-blockingly**
  and re-checks next iteration if not ready. Two threads (pybind11, GIL released); no per-layer barrier.
- Numbers: T4 (16 GB) + 2×Xeon 6130: **+84–96%** throughput vs vLLM; A10: +11–89%; vs NEO (best hybrid baseline) up
  to +49%. Their measured inequality: pipelining only wins when CPU speed ≥ ~13% of GPU's; NEO's sub-batch
  "asymmetric pipelining" doubles GPU linear work and bubbles.
- Relevance: proves the *whole-window sync* shape (not per-layer) is the one that works on real hardware, and that a
  slower device tolerates being one full window behind. Our analog: hand the pool the whole window's miss set and let
  the in-graph combine wait on a device counter, not a host join.
- Confidence: A (full HTML fetched; dense-only, not MoE — the transfer to expert misses is our extrapolation [H]).

### 1.5 HybriMoE — "HybriMoE: Scheduling and Caching for Efficient Hybrid CPU-GPU MoE Inference" — arXiv 2504.05897, DAC'25 — [B]

- Mechanism: (i) dynamic intra-layer scheduling to balance expert work across CPU and GPU; (ii) **impact-driven
  inter-layer prefetching**; (iii) score-based caching against activation instability. Built on KTransformers.
- Numbers: average **1.33× prefill / 1.70× decode** vs the state-of-the-art hybrid baseline (KTransformers) on three
  MoE LLMs; code at PKU-SEC-Lab/HybriMoE.
- Relevance: the existence proof that *scheduling the CPU tier dynamically* (not fixed per-layer mapping) buys ~1.7×
  at decode on exactly our class of system. Supports P1/P2's premise that our 13.56 ms is mostly schedule, not bytes.
- Confidence: B (abstract verified; stall percentages live in the full text, not fetched this session).

### 1.6 Mirage MPK — "Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference" — arXiv 2512.22219, OSDI'26 — [A−]

- Mechanism: a compiler that lowers a model into an **SM-level task graph executed inside one persistent megakernel**
  — decentralized per-SM task scheduling, cross-operator software pipelining; the host launch loop ceases to exist.
- Numbers: up to **1.7× lower end-to-end latency** vs kernel-per-operator serving systems; code mirage-project/mirage.
- Relevance: the endgame form of "the graph walk is the overhead". Also evidence the field converged on the same
  diagnosis as our 9.5 µs-vs-2.3 µs measurement: node spacing, not node work, is the cost.
- Confidence: A− (abstract page fetched; per-config numbers live in the PDF).

### 1.7 Hazy Research — "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B" (blog, May 2025, Stanford) — [A]

- Mechanism: whole forward pass as one persistent kernel; static per-SM instruction schedules from a Python-level
  interpreter; SMEM page allocator; **producer-consumer counters in global memory** instead of stream dependencies
  (the MLP-down consumer waits in 4 counter-chunks, not one barrier). They explicitly found PDL's
  `cudaGridDependencySynchronize` **too coarse** and chunked it manually.
- Numbers (batch-1, Llama-3.2-1B): H100 forward <1 ms (~78% of 3.35 TB/s; >1.5× SGLang, ~2.5× vLLM); B200 **680 µs**
  (>3.5× vLLM, >1.5× SGLang). Their measured launch costs: **~2.1 µs** stream launch, **~1.3 µs** CUDA-graph launch
  on H100; an optimistic 5 µs/kernel stall estimate caps 16-layer×7-kernel Llama at ~770 fwd/s vs a 1350 fwd/s
  memory ceiling ⇒ **~40%+ of time lost to inter-kernel gaps** at batch-1. B200 600 µs breakdown: 250 µs activation
  latency, 200 µs matvec, 30 µs awaiting weights, **40 µs warp sync**, 80 µs setup/misc.
- Relevance: independent quantification of our 9.5 µs-vs-2.3 µs finding (graph launch ≈ 1.3 µs is the floor they
  beat by *leaving* graphs; we cannot leave graphs, but our wait-counter-in-global-memory pattern is exactly their
  mechanism — and our doorbell spin kernels already are that pattern).
- Confidence: A (blog deep-fetched).

### 1.8 Blink — "CPU-Free LLM Inference by Delegating the Serving Loop to the GPU" — arXiv 2604.07609 (Apr 2026) — [A]

- THE 2026 device-side-scheduling source the tasking asked about. Mechanism: a persistent 1-block 256-thread kernel
  IS the scheduler: scans a GPU-resident ring buffer for prompts (~1–5 µs per 4,096-slot scan), claims slots by
  atomic CAS, **device-side "fire-and-forget" launches of pre-captured CUDA graphs (~2 µs vs 11–17 µs host
  launch)**, polls completion, publishes tokens to the ring. Host exits the inference path entirely after capture
  (frontend moved to a BlueField-3 DPU). The CUDA cap of 120 outstanding device-launched graphs is evaded by
  **window-based tail-launch recovery** (after 120, one tail launch re-instantiates; <0.03 µs amortized/step).
- Numbers (H100-96G, fp16, vs TRT-LLM v1.1 / vLLM v0.13 / SGLang v0.5.8): P99 TTFT up to **8.47×**, P99 TPOT up to
  **3.40×**, decode throughput up to **2.1×**; on Qwen-3-30B-A3B (MoE) **+37% plateau throughput vs TRT-LLM**;
  under synthetic host CPU interference the baselines keep only 28–54% of isolated throughput while Blink stays at
  0.92–1.14×. Limitations: single GPU, model fully resident (CPU/DRAM offload unsupported — *our* shape), no spec
  decode yet, 120-launch workaround required.
- Relevance: the strongest published evidence that moving the *loop*, not just the work, onto the device pays in
  production-sized systems; and that the payoff grows with orchestration overhead share — which is what our
  window boundary is. Supports P4.
- Confidence: A (full HTML fetched).

### 1.9 KTransformers (SOSP'25) + SGLang integration — DOI 10.1145/3731569.3764843 (no arXiv) — [B+/C]

- Production CPU/GPU hybrid MoE (DeepSeek-671B-class on one GPU): hot experts + attention on GPU, cold experts on
  CPU (AMX), **asynchronous CPU-GPU compute overlap and expert-weight prefetch**; SGLang shipped it as a backend
  (LMSYS blog, Oct 22 2025). Reported decode from 4.68 t/s baseline to materially higher with overlap+placement;
  prefill up to 27.95× vs llama.cpp.
- Relevance: the production standard for our exact hybrid; their decode wins came from the same three moves as P1/P2
  (batch the CPU tier, prefetch by prediction, overlap instead of barrier).
- Confidence: B+/C (paper PDF surfaced via search; integration blog verified).

### 1.10 vLLM V1 → v0.14 async scheduling — vllm.ai blog 2025-01-27 (fetched) + v0.14.0 release coverage — [A−/C]

- Mechanism: V1's diagnosis — as GPU step times shrink (H100 small-model steps ~5 ms), **CPU-side scheduling/prep/
  detokenization becomes the bottleneck and leaves the GPU idle between steps**. V1: isolated EngineCore process,
  zero-overhead prefix caching, persistent-batch input prep by diffs, up to 1.7× throughput vs V0. The overlap
  scheduler (April 2025 PR) then runs the CPU scheduler **one step ahead** of the GPU; v0.14.0 made **async
  scheduling the default**, overlapping CPU scheduling with GPU execution to close GPU gaps.
- Numbers: 1.7× throughput (V1, ShareGPT, Llama-3.1-8B/3.3-70B); latency lower at all QPS; prefix cache <1% cost at
  0% hit. Per-step gap numbers: the AWS Neuron design doc and secondary coverage cite ~5–10 ms GPU dead time/step
  eliminated [C].
- Relevance: the one-step-delayed *boundary* mechanism is exactly P5's shape applied to our window tails.
- Confidence: A− (V1 blog fetched; v0.14 specifics via secondary coverage, Medium 403'd).

### 1.11 PDL: TensorRT-LLM casebook + llama.cpp PR #15479 — [A−/C]

- TRT-LLM (NVIDIA/TensorRT-LLM perf-optimization-casebook, `runtime-execution/pdl.md`): PDL lets a dependent kernel
  begin grid setup/preamble while the producer's tail drains (`cudaLaunchAttributeProgrammaticStreamSerialization`);
  gated **SM ≥ 90**; targeted exactly at "**chains of many small kernels at low concurrency**" (small-batch decode).
  No measured % in the doc; verification is qualitative (nsys gap check + accuracy parity vs `TRTLLM_ENABLE_PDL=0`).
  **Caveat, their own words: PDL "HAS produced real accuracy/NaN regressions"** in TinyGEMM, quant kernels, FMHA/
  routing, dsv3 MoE, Qwen3-235B ATP — per-kernel gating with rollback is mandatory.
- llama.cpp PR #15479 (Aug–Sep 2025, closed unmerged): systematic `cudaTriggerProgrammaticLaunchCompletion()` /
  `cudaGridDependencySynchronize()` insertion at every kernel entry/exit + PDL attributes on captured graph nodes;
  sync points placed so **no kernel logic overlapped and ordering was unchanged**; reviewer's verdict: "largest when
  running very small models on very fast GPUs" but "a lot of changes for a marginal speedup" — closed as
  benchmarking-scenario-only.
- Relevance: P3's exact prior art, including the honest negative. Our owned T26-3 band (−0.5..−1.5 ms, 12 files,
  nsys-gated) is *smaller and better scoped* than llama.cpp's blanket insertion — we already have only ~13/1,860
  nodes armed and a measured reason (F-279) the blanket default lost.
- Confidence: A− (casebook fetched) / A− (PR issue fetched; benchmark image not rendered — no t/s extracted).

### 1.12 DeepSeek-V3/R1 inference system disclosure (OpenInfra Week Day 6, Feb 2025) — [C]

- Mechanism: prefill/decode disaggregation with different parallelism per phase, large-scale EP, redundant experts,
  and an **overlap scheduler that hides all host/comms work under GPU execution** at fleet scale. Replay numbers
  (RTP-LLM): ~32.2K prefill / **~14.8K decode t/s per H800 node**.
- Relevance: doctrine, not mechanism — the world's largest MoE operator treats "no host work on the critical path"
  as an invariant. Our tier is 8 orders of magnitude smaller but the invariant is the same one P1/P2/P5 enforce.
- Confidence: C (disclosure verified via search + replay; no deep fetch this session).

### 1.13 CUDA primitives: conditional nodes + green contexts — [B/C]

- Conditional graph nodes (developer.nvidia.com/blog, May 2024): `cudaGraphConditionalHandle` gives device-side
  if/while *inside one graph launch* — the host never re-enters the loop. API surfaced 12.4+, docs-wide by 12.8; we
  are on CUDA 13.4. Device-launched-graph caps (120/255/1) bound device-launched graphs only, not conditional nodes
  inside a host-launched graph [C: T37 S11].
- Green contexts (developer.nvidia.com/blog, Oct 2026): explicit SM partitioning with strict isolation; primary-
  context grids still execute on all SMs (forum thread), so only a green context actually protects a decode lane's
  SMs. MPS = coarser percentage carve across processes.
- Relevance: conditional nodes are the P4 primitive (a device-side layer loop inside the standing window graph);
  green contexts are a conc-lane isolation tool, not a solo-lane lever (banked: T50 §1.3 priced conditional-node
  draft-shape control at +0.3–0.9% with its consumer rule dead — P4 is a *different* consumer: the layer loop, not
  the draft shape).

### 1.14 Corroboration: "Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Serving" — arXiv 2603.22774 (May 2026) — [C]

- Identifies host-side paths (scheduling, RPC, input prep) as the CPU-scarcity bottleneck in serving clusters —
  academic confirmation that the host path, not FLOPs, is the modern gap; motivation section of Blink.

---

## §2 OUR ARITHMETIC BASE (all [M] unless marked)

- tau = 4.31 tokens/window (accepted, record class). t/s = tau/V.
- Standing solo: 223 mean / 234 best ⇒ V_ee(standing) = 4.31/223 = **19.33 ms** end-to-end per-window equivalent.
- Record-point window (T 4.31–4.38): window **29.12 ms** with CPU term **13.56 ms** exposed (4.72·4.31−6.8 = 13.54 ✓
  measured 13.56) ⇒ in-window 148 t/s; CPU term removed ⇒ 15.6 ms ⇒ **276 t/s in-window** (banked F-125). The
  campaign convention (F-125/CPU-TIER-OVERLAP §1) books this as the +23.8% prize class (223→276).
- GPU(T) ≈ 9.0–10.4 ms flat over 48 layers ⇒ **0.19–0.22 ms/layer walk**. CPU term/layer at T 4.31 = **0.283 ms**.
  ⇒ *per-layer* CPU > *per-layer* GPU walk: the §6 "move the spin before the resident GEMM" shape was arithmetically
  doomed to hide at most R per layer (~0.04–0.07 ms/layer from T13A's 1.9–3.5 ms hits line). **Only a window-level
  shape reaches the 276 class.**
- Node spacing: tasking's fresh measurement ~9.5 µs/node spacing carrying ~2.3 µs work on dependency-bound
  stretches; banked window-average marginal 0.48–0.95 µs/node × ~1,860 nodes = 0.9–1.8 ms/window (T50 §1.2). The two
  reconcile only if the 9.5 µs stretches are a subset of the walk — the A1 census nsys rider (T50 §5.3) settles it;
  P3 is priced off the banked number and flagged.
- Boundary tails: commit 0.02–0.34 + emit 0.35–0.61 + adapt 0.57–0.68 + host stage/launch 0.2–0.5 ⇒ **1.1–2.2 ms**
  of host-side window-boundary work [M: T50 §1.1/§1.3].
- CPU tier byte check: ~182 misses × 1.94 MB = 353 MB / 13.56 ms = **26 GB/s** vs ~80–100 GB/s stream capacity [H]
  ⇒ scheduling-bound; floor ≈ 3.5–4.5 ms at 26→80+ GB/s.

---

## §3 PROPOSALS (costed; each names the stage that waits, on whom)

### P1 [STRUCTURAL] — Window-fused CPU tier: persistent polling workers + per-window miss queue (the "DPDK shape")

- **Stage that waits / on whom:** the GPU's in-graph flag spins (`wait_flag_ge*`, verify.cpp:2215–2280) wait on the
  HOST, because the host thread walks ring(l)→pool(l)→release 48× per window with a blocking, re-forked pool call
  (session.cpp:949–984, pool at :977). The 15 workers never run as a window-sized cohort.
- **What physically changes:** workers become persistent spinning consumers of a per-window mapped job queue (the
  doorbell buffer already carries `h_ids`/`h_w_`; add per-layer completion counters). The host thread stops
  computing entirely: it reads ring(l), publishes the plan (its existing flagA store), and moves on; the layer's
  release becomes "counter(l) == jobs(l)", checked by the existing in-graph spin kernel — same memory-ordering
  contract (fence + release before the combine reads `y_miss_host`). 353 MB then drains at the machine's stream
  floor ≈ 3.5–4.5 ms instead of 26 GB/s.
- **Expected gain [H, constants shown]:** CPU term 13.56 → 4.5–6.0 ms ⇒ record window 29.12 → 20.1–21.6 ms ⇒
  in-window 148 → 199–215 t/s; in campaign convention that recovers ~55–65% of the 223→276 gap ⇒ **223 → ~253–261**.
- **Prior art:** APEX's two-thread non-blocking shape [A]; HybriMoE's dynamic intra-layer CPU/GPU schedule (1.70×
  decode vs KTransformers) [B]; KTransformers/SGLang production overlap [B+].
- **Risk:** NUMA/CCD placement is 0-for-2 in this campaign (F-295) — pin workers to the LMD node of the pinned
  arena; 15 spinning threads burn cores (fine on 32t idle box); queue must be per-window reset without tearing.
- **Cheapest falsifier:** one instrumented boot. `tg.ms_pool` already exists (session.cpp:983) — add queue-drain
  timing; if `ms_pool` falls ≥40% with tokens bit-identical, GO. No graph re-capture (flag semantics unchanged).
- **Effort:** M. Sites: session.cpp:949–984 (loop), PoolFn impl (include/strata/core/session.hpp:222–227), pool
  internals in src/core/expert_source.cpp / peer_experts.cpp / pinned.cu.

### P2 [STRUCTURAL] — Device-plan lookahead + combine-last reordering (finish CPU-TIER-OVERLAP §8/§9's unbuilt remainder)

- **Stage that waits / on whom:** the resident grouped GEMM waits at flagA (verify.cpp:2232) for the host's *plan*
  publish; the combine waits for CPU rows (`wait_flag_ge_or`, verify.cpp:2265–2268). F-135 killed only the
  FUSE_WAIT folding; the per-group reordering with the *partial-plan* path was explicitly left open.
- **What physically changes:** (a) arm the existing planned path (`wait_flag_ge_planned`, verify.cpp:2242 — "fork
  T1.1: partial-plan aware") so resident kernels launch on the predicted plan; (b) per group: resident GEMM ∥ host
  pool, **combine strictly last** — widen the `_or` skip to cover a correctly predicted whole plan. With P1 landed,
  per-layer CPU residual is 0.09–0.12 ms < 0.20 ms/layer walk ⇒ the residual hides under the in-graph walk for the
  first time (today 0.283 > 0.20 — hiding was arithmetically impossible; **P1 changes the ratio that doomed §6**).
- **Expected gain [M→H]:** residual CPU hidden ⇒ window → the 15.6 ms class ⇒ **in-window 276**; campaign
  convention: the full 223→276 (+23.8%) prize. P1+P2 composed = the 276 class (no double-count: P1 compresses the
  term, P2 hides the compressed residual).
- **Prior art:** HybriMoE's inter-layer prefetch-by-impact (prediction feeds the schedule) [B]; DeepSeek's
  overlap-scheduler invariant [C]; the engine's own §7 monotonicity proof (reordering can never be slower).
- **Risk:** mispredicted plans cost a re-plan stall; ordering invariants (spec §3) get stricter; F-135's lesson that
  gfx12 defaults don't transfer to sm_120 — this variant is a *reordering*, not a transfer.
- **Cheapest falsifier:** nsys on one boot — per-layer grouped-GEMM duration vs per-layer pool residual (spec §7's
  binding check), plus tokens-identical parity vs standing.
- **Effort:** M–L. Sites: verify.cpp:2195–2290 (per-group waits), verify_kernels.cu spin kernels (the :1051-class
  `wait_flag_ge`), device_plan_ fork.

### P3 — PDL full chain + node diet (own the 9.5 µs spacing where the work is 2.3 µs)

- **Stage that waits / on whom:** the GPU front-end between ~1,860 nodes; banked marginal 0.48–0.95 µs/node
  (0.9–1.8 ms/window), fresh 9.5-vs-2.3 µs on the worst stretches.
- **What physically changes:** extend PDL from ~13 nodes to the decode-critical chain (TRT-LLM's exact target:
  "chains of many small kernels at low concurrency", SM ≥ 90; we are sm_120); the dependent node's grid setup
  overlaps the producer's tail, compressing spacing without reordering memory semantics. `STRATA_DF_PDL=0` is
  pinned in the launcher (:32) — the surface is already armed in code (verify_kernels.cu:2084).
- **Expected gain:** owned T26-3 band **−0.5..−1.5 ms** ⇒ +2–4% (223 → 228–234), stretch ×1.08 if the census proves
  the 9.5 µs stretches are ~30–40% of the walk. Alone it never reaches 290; composed it decides whether 300+ exists.
- **Prior art:** TRT-LLM casebook [A−]; llama.cpp #15479 — including their reviewer's "marginal" verdict, which is
  why this is scoped to the *critical chain only*, not blanket insertion.
- **Risk:** per-kernel NaN/accuracy regressions are documented at TRT-LLM (their casebook) — per-node gate + the
  standing PDL-race screen; F-279 measured the blanket default losing.
- **Cheapest falsifier:** nsys per-pair probe on ONE dense norm→GEMM pair before the 12-file chain (T26-3's entry
  gate). **Effort:** M (owned, already scoped).

### P4 [STRUCTURAL, aggressive] — Device-resident window loop (Blink shape, 2026): the host exits the per-window path

- **Stage that waits / on whom:** every window boundary — host stage/launch 0.2–0.5, in-graph gaps 0.4–0.8, commit/
  emit/adapt ~1.3 ms, and the draft chain's host-gated launches (mtp.cpp:1670–1766, ~50 µs/step × 2.2 steps).
- **What physically changes:** one persistent `<<<1,1>>>` scheduler block (the pattern already exists in-tree: the
  doorbell spin kernels) walks layers inside ONE while-conditional graph (CUDA 12.4+ conditional nodes; CUDA 13.4 in
  hand); window graphs are launched device-side, fire-and-forget (~2 µs vs 11–17 µs host, Blink [A]); the
  120-outstanding-launch cap is evaded by tail-launch re-instantiation — trivial for us because our graph families
  are fixed-shape (re-instantiate the same exec). Host keeps exactly two duties: feed P1's miss queue, drain tokens.
  NOTE honestly: T50 §1.3 priced conditional nodes at +0.3–0.9% — that priced the *draft-shape* consumer (dead
  rule). This is a different consumer: the layer loop and the launch path, which Blink measured at up to 2.1×
  decode throughput / 8.47× P99 TTFT vs host-driven loops [A].
- **Expected gain [H]:** removes ~1.5–2.5 ms of boundary/gap terms ⇒ **+8–15%** on top of P1/P2's 276-class (→
  ~298–317); alone +4–8% (223 → 232–241). Also derisks the conc lane: Blink's interference numbers show host-driven
  engines lose 28–54% of throughput under host load — our 6-client box lives there.
- **Risk:** HIGH — device-launch caps (120/255/1) and sm_120 maturity; debuggability; Blink required fixed-size MoE
  buffers (we have fixed-shape graphs, so OK) and a fully resident model (ours is, minus the CPU tier which P1
  serves through mapped memory the scheduler block never touches).
- **Cheapest falsifier (no engine change):** 30-line prototype in bench/ — replay the standing `exec_[T]` graph N
  times from a device-launched loop vs host `cudaGraphLaunch`; GO only if steady-state spacing < 0.7× host spacing
  on sm_120. **Effort:** L.

### P5 (cheap rider) — One-step-delayed window boundary: launch window n+1 before running window n's tails (vLLM async-scheduling shape)

- **Stage that waits / on whom:** the GPU idles at every window boundary while the host serially runs commit
  (0.02–0.34) + emit (0.35–0.61) + adapt (0.57–0.68) + stage/launch (0.2–0.5) ≈ **1.1–2.2 ms**; the next window's
  ~12 ms replay is a huge shadow to hide in.
- **What physically changes:** reorder the host pipeline: stage + `cudaGraphLaunch` of window n+1 (and enqueue the
  draft chain) FIRST, then run window n's emit/adapt/bookkeeping on double-buffered output slots. Scheduling-only,
  exactness-neutral. This is vLLM's one-step-delayed scheduler (§1.10) and Sarathi's ordering principle (§1.1)
  applied at our window boundary.
- **Expected gain [H]:** hide 60–80% of the boundary tail ⇒ 0.8–1.6 ms/window ⇒ **+4–8%** (223 → 232–241);
  composes with everything (disjoint from the CPU term).
- **Cheapest falsifier:** zero new boots to price — the decode-timing line already prints the tails; one reorder
  build, read the same line. **Effort:** S–M. Sites: generate.cpp bt_run/bt_commit/bt_emit (:11775, :12467–12551),
  rescue paths (:11814), HostGraphLaunch call (verify.cpp:2965).

---

## §4 COMPOSED LADDER (honest)

| rung | mechanism | Δ | cumulative (campaign convention) |
|---|---|---|---|
| B | standing | — | 223 |
| P5 | boundary tails under next window's replay | +4–8% | 232–241 |
| P1 | CPU tier to memory floor (scheduling, not bytes) | +13–17% of headline gap share → ~253–261 | ~253–261 |
| P2 | hide compressed residual under the walk | to the 276 class | **276–287** |
| P3 | PDL chain + diet on node spacing | +2–4% (stretch +8%) | 282–298 (stretch ~311) |
| P4 | device-resident loop (aggressive) | +8–15% of boundary/gaps | ~298–317 if it lands |
| τ | retrain v2 / requant (owned: T14B/T17F) | +6–16% tau-side | crosses 335 |

No double-counting: P5 (boundary), P1+P2 (the one CPU term), P3 (in-graph spacing), P4 (the launch/loop path) are
disjoint terms of V; P4 partially overlaps P5's hiding and should supersede it, not stack. **The +30% goal (290) is
reachable with P5+P1+P2+P3 landing in-band; the +50% goal (335) additionally requires tau growth (retrain/requant,
owned elsewhere) or P4 landing high.** Biggest single unknown: whether the 9.5 µs stretches are a subset (census
nsys) and whether the pool's 26 GB/s is fork/join overhead (P1's one-boot falsifier answers it for ~a boot's cost).

## §5 SOURCES (external, this session unless carried)

1. Sarathi-Serve — arXiv 2403.02310 — https://arxiv.org/html/2403.02310v1 (OSDI'24) [A]
2. DistServe — arXiv 2401.09670 — https://arxiv.org/abs/2401.09670; retro https://haoailab.com/blogs/distserve-retro [A−]
3. SGLang — arXiv 2312.07104 — https://arxiv.org/pdf/2312.07104; LMSYS blog https://lmsys.org/blog/2024-01-17-sglang [B+]
4. APEX — arXiv 2506.03296 — https://arxiv.org/html/2506.03296v2 [A]
5. HybriMoE — arXiv 2504.05897 — https://arxiv.org/abs/2504.05897 (DAC'25; code PKU-SEC-Lab/HybriMoE) [B]
6. Mirage MPK — arXiv 2512.22219 — https://arxiv.org/abs/2512.22219 (OSDI'26; mirage-project/mirage) [A−]
7. Hazy Research megakernel — https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles (github HazyResearch/Megakernels) [A]
8. Blink — arXiv 2604.07609 — https://arxiv.org/html/2604.07609v1 [A]
9. KTransformers — SOSP'25 DOI 10.1145/3731569.3764843; SGLang integration https://www.lmsys.org/blog/2025-10-22-KTransformers; docs https://ktransformers.net/en/docs/optimization-techniques/amx [B+/C]
10. vLLM V1 — https://vllm.ai/blog/2025-01-27-v1-alpha-release; v0.14 async-scheduling coverage; docs https://docs.vllm.ai/en/latest/api/vllm/config/scheduler; AWS Neuron design doc [A−/C]
11. TRT-LLM PDL casebook — https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/references/runtime-execution/pdl.md; CUDA PDL guide https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html; llama.cpp PR https://github.com/ggml-org/llama.cpp/issues/15479 [A−]
12. DeepSeek-V3/R1 inference system — https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md (replay: https://rtp-llm.ai/build/en/references/deepseek/reporter.html) [C]
13. CUDA conditional nodes — https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes; green contexts — https://developer.nvidia.com/blog/control-how-your-gpu-shares-work-with-green-contexts [B/C]
14. Corroboration — CPU-induced slowdowns — arXiv 2603.22774 [C]

Local anchors (read this session): session.cpp:931–991 (session_run_token: ring→pool→release loop; ms_pool timer
:983); session.cpp:900–928 (capture: doorbell_wait + copy_from_mapped + moe_hit_add nodes); verify.cpp:2215–2280
(per-group flagA/flagB/flag waits, planned/_or variants), :2232/:2242/:2245/:2265–2268, :2776/:2913/:2988/:3459/
:3738/:3853/:3995/:4058/:4431/:4596 (graph instantiate/launch sites); generate.cpp:16936 (session_run_token call);
banked: mtp.cpp:1670–1766, generate.cpp:9300–9305/:11775/:11814/:12467–12551, device.cu:375–431, launcher
launch-custom-iq3s-8081.sh (:32 DF_PDL=0), CPU-TIER-OVERLAP-SPEC.md (§6–§9 corrections, F-135 verdict), T50-new-levers.md
(§1 anatomy, §1.3 launch-cadence, A4 serving-family check).
