---
type: digest
id: condgraph-zero-host
created: 2026-11-05
status: evidence
topic: The zero-host-sync decode window — CUDA device-side conditional graph nodes (cudaGraphConditionalHandle, CUDA_GRAPH_NODE_TYPE_CONDITIONAL) on CUDA 13.x / sm_120, how production engines actually run per-step decode loops (TRT-LLM static-buffer graphs, vLLM capture-sizes + piecewise, SGLang overlap + BCG), the persistent-megakernel alternative at tiny batch (Hazy Research, Cohere North production megakernel, Mirage MPK, FlashDecoding++), and dynamic-shape handling (graph param update vs capture buckets vs padded dispatch)
scope: read-only web research + read-only code anchors; fresh fetches 2026-11-05 carry inline URLs; facts fetched 2026-11-04 in sibling digests (2026-11-04-latency-architecture.md, 2026-11-04-sm120-kernel-advances.md, 2026-11-04-past-ceiling.md, 2026-11-04-decode-control.md) are cited as carried, not re-derived. Confidence grades: measured-public / measured-ours (in-tree or this-rig) / design-inference / speculation. Falsified standing respected: PDL alone, full-graph-everything for DS (+10-15% not 3.75x), device-launch standalone, stream priorities, W12 GDN_SPLIT, SMT lanes, E1 partial combine, TC_QT scorer, DS4_AHEAD, STRATA_PREFILL_CADENCE, DSpark, GR_UP_WIDE, PF_FUSED, GR_V3, TF32 scorer. The campaign checkpoint names the conditional-WHILE one-doorbell graph as the next pick — this digest is its implementation brief.
---

# Conditional-node zero-host decode window — evidence digest (2026-11-05)

## 1. Question

What is the strongest production-proven technique to eliminate host round-trips inside a per-window decode loop on CUDA 13.x / sm_120, and exactly how does it land on (a) the qwen path's ~48 host-device doorbell round-trips per 9-16 ms verify window and (b) the dsv41 path's 3 host syncs per ~16 ms round? Constraint carried from the campaign: an arm adopts at >= +4% average across lanes with a named falsifier; the DRAM lane (49.5 GB/s ceiling, 7.3 ms DS lane term) is untouchable by any host-protocol mechanism.

## 2. Findings

### 2.1 Conditional-node API: exact semantics and limits, CUDA 12.4 -> 13.x — measured-public

Fresh fetches: NVIDIA blog https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes ; CUDA 13.5 Programming Guide s4.2 https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html (both 2026-11-05).

- Version line: IF/WHILE introduced CUDA 12.4; ELSE and SWITCH added in 12.8, which is also the release that "delivers NVIDIA Blackwell support" (https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support). The 13.x programming guide documents the full feature set as stable API. ROCm parity is still an open feature request (carried: https://github.com/ROCm/hip/issues/3905).
- Semantics, exactly: a conditional node reads its handle "when the node is executed and after each completion of the body graph." WHILE "will continue to execute their body graph until the condition value is zero." The condition evaluation is a device-side memory read of the handle word — the per-evaluation cost is negligible; the real per-iteration costs are body-node dispatch and any spin kernels (see 2.6 for the measured node cost).
- Who sets the condition: `cudaGraphSetConditional(handle, value)` is a DEVICE-only function, called from a kernel inside the body. There is NO host-side set API (confirmed both in the guide and the runtime-API surface). Per-window loop state therefore rides mapped pinned words or device scalars read by a "state kernel" at the body tail — never conditional-node params.
- Handle lifecycle: `cudaGraphConditionalHandleCreate(&h, graph, initVal, cudaGraphCondAssignDefault)`; "the handle must be associated with a single conditional node. Handles cannot be destroyed." The initializer is re-applied at each graph start, so the while-loop state RESETS FREE on every `cudaGraphLaunch` — no host reset call needed. Without the assign-default flag the value is undefined at start (correctness trap).
- Body-node whitelist (guide s4.2.4.2, verbatim class): kernel, empty, memcpy, memset, child graph, and nested conditional nodes. FORBIDDEN: event nodes, host-function nodes, CUDA Dynamic Parallelism, device graph launch inside the body. Cooperative launches are allowed "so long as MPS is not in use" (we do not run MPS). Memcpy/memset inside bodies must act on device memory or pinned device-mapped host memory — exactly our mapped staging ring; CUDA-array copies are not permitted. All kernels in any nesting level must be one context, one device.
- Build path: `cudaGraphAddNode(&node, graph, deps, nDeps, &cParams)` with `cParams = { cudaGraphNodeTypeConditional }`, fields `.conditional.handle`, `.conditional.type` (`cudaGraphCondTypeWhile`), `.conditional.size = 1`; the empty body graph is retrieved from `cParams.conditional.phGraph_out[0]` AFTER the add. The body can be populated with `cudaStreamBeginCaptureToGraph(stream, bodyGraph, ..., cudaStreamCaptureModeRelaxed)` (CUDA 12.3+) — stream capture inserts nodes into an existing graph, so our existing launch-code bodies capture without re-expression as raw AddNode calls. Full samples: NVIDIA/cuda-samples `graphConditionalNodes.cu`.
- Update law post-instantiate: "conditional nodes can't have parameters changed" — per-node exec-update does not apply to them; whole-graph `cudaGraphExecUpdate` requires topology identity and matching handle creation order. Practical consequence: T-bucket/shape families still need per-shape captures (2.5); the conditional wrapper itself never changes per window.
- Limits: nesting allowed, no stated depth limit; SWITCH runs no body when value >= n (clean off-arm); no published NVIDIA numbers on condition-read or body-dispatch overhead — device-side graph dispatch is priced instead by the carried driver study: 5.9 us host submission for a 2000-kernel graph under CUDA 13.0, ONE doorbell write (arxiv.org/html/2604.26889v1, fetched 11-04), i.e. per-node device dispatch is ns-scale.

### 2.2 sm_120 / consumer Blackwell status — measured-public, thin field history

- Support: CUDA 12.8+ on Blackwell is the NVIDIA-stated floor for the full IF/ELSE/SWITCH/WHILE set (source above). Our toolchain is CUDA 13.x (nvcc 13.2 on this rig) — inside the stable window.
- Field history is THIN and this is the honest risk: no public inference engine ships conditional nodes (carried: vLLM/SGLang/TRT-LLM all use torch-style whole-capture graphs; nothing in TRT-LLM release notes). Adopters are HPC/graphics plus one 2026 sensor-processing paper that roots a GPU-resident graph in a while-conditional node on Blackwell GPUs (arxiv.org/html/2610.13809v1 — existence proof, not a perf source). A targeted search for sm_120 / RTX 5090 conditional-node bugs (hangs, Xid errors) found NO direct report (2026-11-05 search on record); nearest neighbors are unrelated: vLLM #27057 is a quality bug with PIECEWISE_AND_FULL graphs on B200, and an RTX 5090 Laptop Xid-8 hang thread is graph-adjacent but not conditional-node specific.
- Closest vLLM-integrated device-predicate artifact remains xMIx (carried, arXiv 2607.22595): device-computed predicate gates nested execution at +1.3% mean ITL / +2.6% TTFT vs +510% for eager hooks — evidence that even the WORST form of device-side gating is cheap when compiled rather than hooked.
- Named boundary: first action on this rig is the in-tree microbench (3.5 falsifier a), not a literature answer.

### 2.3 How production engines actually run decode loops — none is zero-host per STEP; all are zero-host per LAYER — measured-public

Fresh fetch 2026-11-05: vLLM design doc https://docs.vllm.ai/en/stable/design/cuda_graphs ; TRT-LLM architecture overview https://nvidia.github.io/TensorRT-LLM/architecture/overview.html + carried blog20. SGLang carried from the 11-04 digests (fetched then).

The structural fact every engine shares: ONE captured graph replay per decode STEP, host orchestration between steps, zero host inside the step.

- TensorRT-LLM: pre-captures graphs "for a predetermined set of batch sizes" into per-size static (stable-address) buffers; per step the scheduler assembles the batch, copies inputs into the static buffers, launches that size's graph; "if an incoming batch's size doesn't match a captured graph, it's padded to the nearest larger supported size" (architecture overview). Host cost per step: input assembly + H2D copies + one graph launch + result readback. Carried metadata numbers (blog20, fetched 11-04): ~10 MB per captured graph, 49 graphs ~ 260 MB, 259 graphs ~ 2.6-4.9 GB.
- vLLM V1: capture on first `_dummy_run`; dispatch key is a `BatchDescriptor` whose `num_tokens` "can be the padded token length"; `cudagraph_capture_sizes` config; three modes — FULL (attention inside the graph, backend-permitting: `AttentionCGSupport` ranges ALWAYS > UNIFORM_BATCH > UNIFORM_SINGLE_TOKEN_DECODE > NEVER; FlashInfer upgraded to UNIFORM_BATCH with the TRTLLM backend on Blackwell), PIECEWISE (torch.compile splits at attention; "attention or other CUDA-Graphs-incompatible operations stay eager"; the boundary-buffer trick is carried vLLM PR #9724), and FULL_DECODE_ONLY. Host still dispatches per step (`cudagraphdispatcher.dispatch` + `set_forward_context`) — mode NONE runs eager for warmup.
- SGLang (carried, fetched 11-04): overlap scheduler runs the CPU scheduler one batch ahead inside GPU compute (no GPU idle over 5 consecutive decode steps, 1.1x end-to-end, lmsys.org/blog/2024-12-04-sglang-v0-4); decode gets whole-graph capture per bs; draft runners get their own graphs (lmsys.org/blog/2026-08-17-advanced-cuda-graph — and the draft-extend graph fix alone took Ling-3.0 NEXTN 288 -> 526 tok/s, the strongest single datapoint that the spec-draft path is where serialized launches hide); Breakable CUDA Graphs (PR #19102, default prefill strategy by v0.5.15; standalone library github.com/meta-pytorch/breakable-cuda-graphs) let eager segments sit INSIDE one logical graph — the exact shape of "graph runs GPU layers, host runs the lane between" without tearing the graph apart. BCG replay beat torch.compile piecewise by ~17% by removing per-segment dispatch.
- Mapping to our engines: our qwen verify window IS one step. The 48 doorbell round-trips are per-LAYER/phase events inside one step — the exact class every production engine already collapses into a single replay. No production engine eliminates the host per STEP, and ours must not either (admission, sampling settings, and the CPU expert lane are host work by construction). The target is: one `cudaGraphLaunch` per window, host spends the window 100% inside the ring-service loop with zero CUDA calls.

### 2.4 The persistent-kernel alternative at tiny batch — real, production-shipped once, and a ground-up rewrite — measured-public

Fresh fetches 2026-11-05: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles ; https://cohere.com/blog/megakernels. Carried: Mirage MPK (arXiv 2512.22219, OSDI '26, 1.2-2.1x, multi-GPU NVSHMEM), GPUOS (arXiv 2604.17861v1, RTX 5090-proven ring, <100 ns submission), FlashDecoding++ (arXiv 2311.01282, MLSys 2024: persistent "flat GEMM" via the HEAT method + async softmax with unified max — the persistent-GEMM-at-decode precedent).

- Launch-cost floor, measured on H100 dummy kernels: stream launch ~2.1 us; CUDA graph replay ~1.3 us (Hazy Research). Graphs alone cut launch cost ~2.1 -> 1.3 us and "leave a lot on the table": graph+compile baselines (vLLM/SGLang) reach "at most 50% of available GPU bandwidth" vs 78% for the megakernel — the residual is kernel-BOUNDARY bubbles, not launch APIs.
- In-kernel dependency cost: explicit counters in global memory, ~60 ns per already-passed wait, ~40 us total cross-warp sync per B200 forward ("CUDA's asynchronous barriers are relatively slow"). This is the number that prices spin-flag handshakes INSIDE one replay vs kernel boundaries between replays: ~60 ns vs us-scale.
- Production proof (the big one): Cohere North Mini Code decode megakernel — "we launch exactly one threadblock per SM, and it stays resident for the entire decode step"; controller warps read a host-prepared task list (32-int32 descriptors) from global memory; dependencies are atomicAdd arrival counters; 30B MoE / 3.3B active BF16 on one H100, batch 1: "292 tok/s ... 1.58x faster than vLLM" (vLLM 185 tok/s, 39% of SoL vs Cohere 62%); batch 8 end-to-end 1.25-1.41x across AIME/GPQA/MMLU-Pro/SciCode/LiveCodeBench; accuracy held. Host per step (C++ decode loop): update KV page tables, clear scratch/barriers, launch, sample, stream, retire. Known limits: decode-only, max batch 8, no mixed prefill/decode, batch changes require Python to PARK the C++ decode loop, mutate state, resume.
- Costs of the form: static per-model schedules (their waves are precompiled; ours would be too), smem paging discipline (Hazy: 13x16KiB pages on 213 kB), occupancy pinned at 1 block/SM for the whole step, and scheduling machinery that is an engine rewrite, not a patch. kog.ai reports the same form on MI300X (blog.kog.ai, single-kernel decode, 3000+ tok/s/GPU) — cross-vendor corroboration, not sm_120 evidence.
- Verdict vs conditional graphs for THIS rig: the megakernel wins where kernel-boundary bubbles dominate a compute-dense step (their 1B and 30B models run entirely in VRAM). Our qwen window is dominated by (i) the CPU expert lane's DRAM reads and (ii) host-protocol latency around it — the graph collapses (ii) without touching (i), at patch cost. The megakernel would ALSO have to absorb the lane handoff as in-kernel spins (it does, via counters) but buys nothing extra on (i). Keep it WATCH: Cohere's design (park/resume, max-batch-8, decode-only) maps to our 1-6 requests, and it is the named escalation IF conditional graphs stall on CUDA 13.2 — but it is a quarter-scale build, not an arm.

### 2.5 Dynamic shape handling (T=1..8 verify rows, 1-6 requests) — measured-public + design-inference

Four mechanisms exist in production; none uses recapture in steady state:

1. Padded dispatch into captured buckets — TRT-LLM pads to nearest captured size; vLLM's dispatch key carries the padded `num_tokens`. Cost: wasted rows are dead compute; at our tiny shapes (1-8 latency-bound rows) dead-row cost is small, and the in-tree MMVQ 8-column family is fixed-width by design.
2. Per-K/per-shape capture ladders — Cohere DSD captures "CUDA graphs for EVERY valid K" (carried); SGLang's adaptive ladder is "a reference swap, not an online graph recapture" (carried). In-tree: `set_batch_graph_limit` LRU on batch graph pairs already implements bounded capture families.
3. Whole-graph exec update — llama.cpp patches per-token params on ONE instantiated graph (carried, +5-18% token-gen merge). Guide law: updates require topology identity; memcpy size is baked at capture (patch the node via `cudaGraphExecMemcpyNodeSetParams` or capture max size and carry the true count in a device scalar). In-tree measured law (graph.hpp NOTE 2): "a graph re-reads its input buffers at replay, but NOT its kernel arguments — anything that changes per step must be DATA in a device buffer, never an argument."
4. Conditional SWITCH over shape — runs body[n] for value n, runs NOTHING for value >= n. On our box the honest use is window-level mode select (verify / retry / off), not per-row shape: T is already a capture key, and row-count-as-data would require kernel-side changes gated on measured dead-row cost.

Design-inference for us: keep per-(T, layout) captures with the existing LRU bound; the conditional wrapper is orthogonal to shape. A T-agnostic graph (row count read from a mapped word) is a follow-on arm only if dead-row cost measures nonzero at 6-way.

### 2.6 What the tree already has — measured-ours

Read-only anchors, 2026-11-05:

- `include/strata/core/graph.hpp` — P2.S5 GraphRegistry, capture-once/replay-many keyed by (layer_type, n_tokens) with fixed-address buffers. Three MEASURED facts baked into the interface (from `bench/micro/graph_capture.cu`): (1) the host wait polls an event (`wait_ms`, 0.021 ms) rather than memory alone — the comment attributes the memory-spin hazard to Windows command batching, a platform claim this Linux rig has NOT revalidated (validate-first item); (2) graphs re-read input buffers at replay but never kernel arguments — per-step values are data-in-device-buffer; (3) fixed addresses are the caller's obligation. Measured on this machine: "10-22 us PER CALL into the driver", "a layer with 8 submissions is ~0.1 ms of round trip."
- `include/strata/core/host_graph_launch.hpp` — the off-thread launch discipline: "A graph which waits for CPU doorbells must therefore launch outside its service thread," with a release-waits escape for early failure. The deadlock-shaped constraint for ANY in-graph wait already has engineering in the tree.
- `src/core/verify.cpp` — the mapped staging ring EXISTS: 17 mapped pinned buffers (m_tok_, m_step_, m_pos_, m_commit_, m_ple_, m_out_, m_ids_, m_w_, m_seq_, m_flag_, m_flagA_, m_flagB_, m_plan_err_, m_ymiss_, balance and plan staging). GPU-side wait kernels exist (`wait_flag_ge`, `wait_flag_ge_or`) and the FUSED wait+copy node family (wait_copy_i32_from_mapped / copy_rows_from_mapped_after — "thread 0 spins on *flag >= value, the block then copies", one kernel node fewer, "a dependent kernel costs ~2.4 us in a gfx1201 graph" — note: that node-cost figure is from the HIP/gfx build; the CUDA rig equivalent is the 10-22 us driver-call figure above). Host flags are raised by mapped-memory STORES (no CUDA call): "the words the spin kernels read are mapped host memory, so a store here is visible." `batch_poll` serves "the layers of this stage's window that have rung ... with no CUDA call" (and STRATA_PIPED_CONCURRENT moves it to its own thread to dodge the driver-lock wait, #31).
- The zero-doorbell proof exists: `ar_live()` / `ar_on()` (STRATA_ADMIT_MERGE_FIRST gate) runs the window graph with every expert resident — no doorbells. The conditional-window move GENERALIZES this: non-resident layers run inside the same single replay, with the existing spin/fused-wait nodes carrying the handshake.
- Delta to build, therefore: capture scope (per-stage graphs -> ONE whole-window body) + conditional-WHILE wrapper (window-level retry/mode) + off-thread launch (already exists) + host serve loop (already exists, unchanged). This is a capture-scope refactor of machinery that is already measured, not new protocol.

## 3. The single best intervention: the whole-window conditional graph, one doorbell

### 3.1 Graph topology (qwen path)

ONE host-launched graph per window family. NOT a per-layer loop — the layer sequence is straight-line nodes inside one body; the WHILE exists only at window level.

```
root graph:
  [child: draft chain — lookup + MTP head nodes, straight-line]        (folds today's serialized
        |                                                               draft launch/finish into
        v                                                               the same replay)
  [state kernel A]  reads: mapped control block (T, positions, token ids already staged),
                    sets: handle_hWin = (plan_err==0 && T>0) ? 1 : 0
        |
        v
  [CONDITIONAL: WHILE handle_hWin]  size=1, body = window_body
        body (window_body):
          for each layer L in {stage order}:                      # straight-line, captured
            [attention/GDN kernels for L]
            [fused wait+copy node: spins m_flagB_L >= seq, then reads expert rows]
            [post-MoE glue for L]
          [ring nodes: doorbell raises per phase — GPU store to mapped word]
          [state kernel B]  reads: m_plan_err_, m_commit_, window-done word
                            sets: handle_hWin = 0 (steady state: always, after one pass;
                                  retry state: leaves 1 to re-run the body once)
        |
        v
  [accept/head + sampling kernels]                                 (already device-side in the window)
  [completion node: kernel store-release to mapped done word]      (host reads once per window;
                                                                    optional single event poll)
```

- What the condition reads: ONLY device/mapped state written by body kernels (m_plan_err_, window-done word). Never host-written conditional params — none exist (2.1). The handle initializer (initVal=1, cudaGraphCondAssignDefault) re-arms the loop on every launch for free.
- Steady state: the body executes EXACTLY ONCE per replay (state kernel B zeroes the handle). Retry paths (plan_err) get one in-graph re-pass without host relaunch. If a later measurement shows the two state kernels add real cost, the straight-line graph with no conditional is the degenerate first step — the one-doorbell property comes from capture scope, not from the WHILE.
- Where host participates: write deltas into the mapped ring before launch (ids, positions, control), call cudaGraphLaunch OFF THE SERVICE THREAD (HostGraphLaunch discipline), then spend the window inside batch_poll serving m_flagA_/m_flagB_ rings with pure memory ops (already zero-CUDA). One completion read after.
- Per-layer GPU wait for the CPU lane: keep the in-tree fused wait+copy nodes (spin folded into the consuming kernel's head) — they are the measured-cheap form (~2.4 us/node class on gfx; spin-only nodes cost a node each). Never a bare spin kernel + separate consumer.

### 3.2 CUDA API skeleton, in call order (runtime API, CUDA >= 12.8 semantics; our 13.x)

```c
// ---- build (once per (T, layout) family; fixed addresses allocated BEFORE this) ----
cudaGraph_t g;            cudaGraphCreate(&g, 0);
cudaGraphConditionalHandle hWin;
cudaGraphConditionalHandleCreate(&hWin, g, /*initVal=*/1, cudaGraphCondAssignDefault);

// state kernel A node (cudaGraphNodeParams{cudaGraphNodeTypeKernel}, args = {&m_ctl_dev, &hWin})
cudaGraphAddNode(&nA, g, nullptr, 0, &kParamsA);

// conditional node
cudaGraphNodeParams c = { cudaGraphNodeTypeConditional };
c.conditional.handle = hWin;
c.conditional.type   = cudaGraphCondTypeWhile;
c.conditional.size   = 1;
cudaGraphAddNode(&nCond, g, &nA, 1, &c);
cudaGraph_t body = c.conditional.phGraph_out[0];          // empty body, populate next

// populate body WITHOUT re-writing launches as raw AddNode calls:
cudaStream_t cs;
cudaStreamBeginCaptureToGraph(cs, body, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed);
  ... existing per-layer launch code on cs, with wait_copy_i32_from_mapped(...) nodes
  ... state kernel B: cudaGraphSetConditional(hWin, /*keep=*/0);
cudaStreamEndCapture(cs, nullptr);

// accept/head/sampling + completion-store nodes on g, depending on nCond
cudaGraphAddNode(...);   // etc.

// ---- instantiate + lifecycle ----
cudaGraphExec_t exec;
cudaGraphInstantiate(&exec, g, cudaGraphInstantiateFlagUpload);   // upload once, off the hot path
// steady state per window:
//   host: store deltas into mapped ring (plain stores / store-release)
//   host: HostGraphLaunch(exec, stream)  — OFF the ring-service thread (in-tree law)
//   host: batch_poll(...) — zero CUDA calls, serves m_flag rings
//   host: read mapped completion word (or one cudaEventQuery, 0.021 ms class)
```

Lifecycle rules: never reallocate any buffer the body touched (fixed addresses are load-bearing; graph.hpp NOTE 3); shape change = pick another family's exec (LRU), never `cudaGraphExecUpdate` into a different topology; if a node must change per window, it is data in the mapped ring, not a kernel argument (graph.hpp NOTE 2).

### 3.3 Capture/replay lifecycle

1. Boot: allocate mapped staging (exists), capture per (T, layout) family lazily on first use (existing capture-condition flow), instantiate with UPLOAD.
2. Per window: stage deltas -> one off-thread launch -> ring service -> one completion read.
3. Retry (plan_err): in-graph second body pass, host untouched.
4. Eviction: batch_graph_limit LRU (exists) bounds families; each captured family ~ 10 MB class metadata (carried TRT-LLM blog20 figure) — 8 T-buckets x few layouts stays inside the ~1.2 GB free floor; monitor with the pin/exhausted hints already in verify.cpp.
5. Failure containment: HostGraphLaunch's release_waits escape already exists for early service failure; keep timeout semantics on the completion read (treat timeout as hang, never as fine — graph.hpp wait law).

### 3.4 Arithmetic vs the doorbell cost (explicit assumptions, per term provenance)

Qwen window, midpoint 12.5 ms (measured 9-16 ms band):

- Today, per window: ~48 doorbell RTs. Decomposition (measured-ours): a subset carry host driver calls at 10-22 us each (graph.hpp, this machine) — call it 24-48 calls = 0.24-1.06 ms; per-wait event polls at 0.021 ms (wait_ms) ~ 0.5 ms at 24 waits; PCIe flag visibility 48 x 1-2 us = 0.05-0.10 ms (carried, arxiv 2512.16056v2). Total protocol ~ 0.8-1.7 ms — consistent with the prior digest's 0.5-2 ms band, PLUS the serialized draft launch/finish (two more launch/wait cycles, the SGLang draft-extend class where one graph fix was worth 288->526 tok/s).
- After, per window: 1 x cudaGraphLaunch (5.9 us per 2000-node graph class, carried driver study; our body is ~200-600 nodes) + 48 mapped-flag exchanges at visibility cost only (~0.05-0.10 ms total) + 1 completion read (~0.02 ms). Residual ~ 0.10-0.15 ms.
- Recovery: ~0.65-1.55 ms off the window = +5-12% window-rate at the midpoint; +4-6% at the 16 ms conc end, +7-10% at the 9 ms serial end. On decode banks (232.2/222.3/315.4 tok/s) that is the +4-8% class across lanes — at/above the adoption bar, which is exactly why the campaign checkpoint names this pick. The draft-chain fold is the upside tail, not the base case.
- What this does NOT move: the CPU expert lane's DRAM bytes (49.5 GB/s ceiling) and the GPU compute term. If post-landing profiling shows the window dominated by lane waits, this architecture class is exhausted (prior digest boundary, unchanged).
- dsv41 landing (second, larger lift): DS has NO graph infra in-tree; the same topology (while-body = one round's phases; the 3 syncs become mapped-flag exchanges served by the existing lane threads) shrinks the ~4 ms slack toward its real-work floor. Carried re-pricing stands: +10-15% DS class, lane term untouched. Build only after the qwen landing validates the pattern end-to-end on CUDA 13.2 (cross-engine change protocol: bench DS -> bench Qwen -> change -> bench both).

### 3.5 Falsifier

Named, cheapest first — micro-falsifiers before any bank:

- (a) Mapped-flag visibility probe on CUDA 13.2 / sm_120 (spin-kernel `__ldcg` + `__nanosleep` poll of a host-toggled mapped word): must show <= ~3 us round trip. If visibility is worse on this driver, in-graph waits are too costly — STOP, the arm dies at the microbench. (Carried 11-04 measured 1-2 us on PCIe5-class parts; this is the sm_120/CUDA-13.2 revalidation.)
- (b) Whole-window STRAIGHT-LINE graph (no conditional): window wall-time vs today's stage launches must drop >= 0.4 ms. If not, the launch/wait decomposition is wrong — stop before adding the conditional wrapper.
- (c) Conditional wrapper added: assert (i) handle initializer re-arms across replays (no drift over 10k windows), (ii) plan_err retry pass produces byte-identical output to the two-launch recovery path, (iii) no node-count/memory creep in the LRU.
- Full falsifier (the bank): 3 legs x {solo, pair, conc} before/after (Qwen solo variance +/-15 t/s needs >= 3 legs — W-cycle law). KILL if mean delta < +4% across lanes or any lane regresses beyond noise. Sub-kill telemetry: host CUDA-call count per window must read ~1-2 (from ~24-48); if it reads high, staging leaks a sync.

## 4. What we explicitly should NOT do

- Do NOT device-launch the window graph (`cudaGraphInstantiateFlagDeviceLaunch`). Sourced hard constraint: conditional nodes are ABSENT from the device-launched-graph allowed node list (kernel/memcpy/memset/child only), one pending launch max, upload required per update (guide s4.2.6). The window stays host-launched — one doorbell per window is the design, not a defect. This also re-confirms the carried DEAD verdict on device-launch-standalone.
- Do NOT make the WHILE iterate per LAYER. A condition kernel per layer adds a device node dispatch per iteration for zero protocol win — the layer sequence is straight-line nodes; the WHILE is window-level only. (Design-inference, priced from 2.1's dispatch notes and 2.6's ~2.4 us/node class.)
- Do NOT reach for a host-side `cudaGraphSetConditional` — it does not exist (device-only, 2.1). Loop state rides mapped words + body state kernels.
- Do NOT patch conditional-node params between replays — "conditional nodes can't have parameters changed" (guide s4.2.3). Per-window variation is data in the ring; per-shape variation is another family capture.
- Do NOT grow the capture ladder to padding-bloat: 259-graph sets cost GBs of metadata (carried blog20) against a ~1.2 GB free floor. Keep the LRU bound; measure dead-row cost before any T-agnostic-graph kernel work.
- Do NOT put events or host functions in the body (forbidden in conditional bodies anyway — 2.1) and do NOT reintroduce `cudaLaunchHostFunc` anywhere on the window path (carried 12 ms tail measurement).
- Do NOT treat the graph.hpp Windows memory-spin caveat as settled Linux fact — the memory-only-spin hazard was measured against Windows display-driver batching. Revalidate on this Linux rig in falsifier (a) before trusting ANY memory-only host wait.
- Do NOT build the megakernel now. Cohere's production result is real and its limits (decode-only, max batch 8, park/resume on batch change) map to our shape, but it is an engine rewrite priced in quarters; its role is the named escalation IF (a) or (b) fails — and FlashDecoding++'s persistent-GEMM lesson stays with the queued GDN-stack kernel work, not the window loop.
- Do NOT expect lane recovery: no mechanism in this digest touches the 49.5 GB/s lane term or the GPU compute term (carried re-pricing: X3 +10-15% DS class stands). If the +4% bar fails on window-rate, the answer is the lane/quant queue, not a graph retry — falsified arms are never retried blind.

## Sources

Fresh (2026-11-05): https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes ; https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html ; https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support ; https://docs.vllm.ai/en/stable/design/cuda_graphs ; https://nvidia.github.io/TensorRT-LLM/architecture/overview.html ; https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles ; https://cohere.com/blog/megakernels ; https://arxiv.org/html/2604.23467v1 (hybrid JIT+CUDA-graph TRT-LLM baseline, H100/CUDA 12.4) ; https://arxiv.org/abs/2311.01282 (FlashDecoding++, MLSys 2024) ; https://arxiv.org/html/2610.13809v1 (while-rooted GPU-resident graph on Blackwell) ; search-on-record: sm_120/RTX 5090 conditional-node bug sweep (none found); cudaGraphSetConditional device-only confirmation.

Carried (fetched 2026-11-04, sibling digests): arxiv.org/html/2604.26889v1 (driver study: 5.9 us / 2000-node graph, one doorbell) ; arxiv.org/html/2512.16056v2 (1-2 us mapped-flag RT) ; arxiv.org/html/2604.17861v1 + github.com/Multi-V-VM/GPUOS (ring, RTX 5090) ; github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.md (10 MB/graph metadata) ; developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs (param patching) ; lmsys.org/blog/2026-08-17-advanced-cuda-graph (BCG, 17% vs piecewise; draft-extend 288->526) ; lmsys.org/blog/2024-12-04-sglang-v0-4 (overlap scheduler) ; github.com/sgl-project/sglang PR #19102 + github.com/meta-pytorch/breakable-cuda-graphs ; github.com/vllm-project/vllm PR #9724 (boundary buffers) ; arxiv.org/html/2607.22595v1 (xMIx) ; cohere.com/blog/hardware-aware-dynamic-speculative-decoding (DSD per-K capture) ; github.com/mirage-project/mirage + arxiv.org/abs/2512.22219 (MPK) ; developer.nvidia.com/blog/enabling-dynamic-control-flow-in-cuda-graphs-with-device-graph-launch ; ROCm/hip#3905.

In-tree (read-only, 2026-11-05): /home/svg/LLM/strata-custom-2/include/strata/core/graph.hpp ; /home/svg/LLM/strata-custom-2/include/strata/core/host_graph_launch.hpp ; /home/svg/LLM/strata-custom-2/include/strata/core/verify.hpp ; /home/svg/LLM/strata-custom-2/src/core/verify.cpp ; /home/svg/LLM/strata-custom-2/include/strata/kernels/verify_kernels.hpp ; /home/svg/LLM/strata-custom-2/include/strata/kernels/elementwise.hpp ; /home/svg/LLM/strata-custom-2/bench/micro/graph_capture.cu.
