# Scout 2 (L2/L3 structural): per-node spacing + capture mechanics — round 2

Date: 2026-10-26. Scope: production numbers on CUDA-graph per-node spacing and node-density reduction, plus MoE capture mechanics. Read-only pass on the strata-custom-2 tree; this file is the deliverable.

---

## Findings

### 1. vLLM FULL vs PIECEWISE vs eager — PR #20059 (the "+5% tput / -2.9% TPOT" claim in full context)

**Grade A** (primary source, author-posted benchmark tables).

- Setup: A100 40G, torch 2.6, CUDA 12.4; Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4; `benchmark_serving`, ShareGPT, 100 prompts, request rate 20.
- FA2: output tok/s 1898.67 → 1992.27 (+5%); mean TPOT 17.08 → 16.57 ms (-2.9%); ITL 13.45 → 12.83 ms; TTFT 76.37 → 78.69 ms (slight regression — the pathology below).
- FlashInfer: 1907.38 → 1946.35 tok/s (+2%); TPOT 16.85 → 16.34 ms (-3.1%).
- Per-step CPU launch context (same PR): eager execution of the flattened fx graph takes "roughly 56 ms to fully launch the model" plus ~5 ms safety-check latency; piecewise drops this to 28 ms. That is host-side launch cost, not GPU spacing, but it bounds what graphs recover.
- Strongest small-model number in the PR: Triton-attention ITL "reduced by 38% for small LLMs" FULL_AND_PIECEWISE vs piecewise. Per-node overhead dominates when kernels are small — exactly our regime.
- **494-blocks-per-SM pathology, primary text:** the captured *mixed-batch* full graph launched "494 blocks per sm for the attention kernel (FA2) ... while it only needs 4.6 blocks per sm when attention is running in eager." Cause: graph capture freezes the attention grid at the padded max-batch shape; one mixed prefill+decode step then runs a decode-sized problem on a prefill-sized grid. This is why vLLM shipped FULL_AND_PIECEWISE, not FULL — a structural argument for keeping attention out of the graph or shape-gating it.
- Corroborating independent measurement (grade B): tomasruizt.github.io/posts/09_full-cuda-graphs-spec-decode — Qwen3-32B + Qwen3-1.7B drafter, spec decode: target runtime 24.36 → 23.65 ms (piecewise→full, -2.91%); draft 2.50 → 2.27 ms (-9.10%); forecast TPOT at K=5: 10.43 → 9.91 ms (-4.9%). Draft (tiny kernels) gains 3x more than target — same "spacing dominates small kernels" signal.

### 2. llama.cpp #6763 / #6766 — per-GPU numbers

**Grade A** for the NVIDIA-authored issue, B for the merged-PR folklore.

- Issue #6763 (NVIDIA): llama-2-7b Q4_K_M, H100-PCIe, bs=1: **143.35 → 163.83 tok/s (+14%)**. "Execution of the graph itself is around 40% faster with CUDA graphs" — i.e. the inter-kernel gaps were ~40% of graph-wall time and graphs remove most of them; end-to-end gain is diluted by host-side graph prep. A100-PCIe: PR #6933 schedule-reset added +3% overall (llama2 7B, bs1). Overall claim: 10-15% on 7B-class.
- NVIDIA blog (grade B): up to **1.2x** for Llama 7B Q4 on H100, bs=1; an additional "~10%" expected from graph-prep/sampling CPU work. **No per-model node counts and no µs figures published** — the row's hope for published node counts from #6766 is not met; the PR and blog quantify speedup, not nodes. What they do document is mechanics: capture once, reuse aggressively ("capture and instantiation outweigh the benefits" otherwise); infrequent structural changes → full re-capture; `cudaGraphExecUpdate` for cheaper topology updates; **frequent minor changes (KV-cache pointers) patched by updating individual node params before replay**. That is the llama.cpp capture-once/param-patch pattern, from the horse's mouth.
- Capture-disable heuristics (grade B, from PR #7302 + commit #19754): graphs auto-disable after 4+ consecutive topology updates ("consecutive update counter" — data-dependent graph shape churn permanently disables capture); `GGML_CUDA_DISABLE_GRAPHS=1` escape hatch. This is the production answer to data-dependent routing: llama.cpp does NOT capture dynamic MoE routing; `mul_mat_id` paths that would need a stream sync are excluded from capture (TAG_MUL_MAT_ID_CUDA_GRAPHS fallback).
- Reddit folklore (grade C): +5-18% on merge; a report of `GGML_CUDA_GRAPH_OPT` *slowing* an RTX 3090 — graphs are not free on consumer parts when recapture churns.

### 3. TRT-LLM blog20 (Tuning CUDA Graph Batch Sizes)

**Grade A** (official tech blog, full tables).

- No per-launch µs published. Their lever is **node/batch-density via padding, not spacing**: default 23 graphs (1,2,4,...,128,256,...,2048); above bs=128 the gaps "can cause up to 50% padding waste."
- **+64 story:** 49 graphs, max padding per step 1024 → 63 tokens → up to **1.3x TPS at high concurrency** (aggregated), and up to **1.5x around concurrency 500-600** disaggregated (DeepSeek-R1). +8 (259 graphs) adds up to 1.15x more at conc 576 but costs 2.6 GB extra graph memory and cuts KV capacity (R1 -8.6%, V3.2 -17.2%). Not recommended.
- Graph metadata: **~10 MB per CUDA graph** (mem_get_info, DeepSeek-R1 TP=4). Capture time 23 graphs 10.77 s vs 49 graphs 25.90 s. Default flipped to +64.
- Takeaway for us: node-density reduction at the *batch* axis (fewer padded-waste nodes) bought 1.3-1.5x at load — but that is throughput shape-matching, not per-node launch spacing.

### 4. SGLang Breakable CUDA Graph (BCG)

**Grade A** (official blog numbers; secondary visual explainer).

- Prefill replay vs eager (gpt-oss-120b, TP4, 4xGB300, decode graphs disabled, fixed input len + 1 out token): **full graph 1.93x, BCG 1.70x, tc_piecewise 1.45x**. So the "break" costs the full→BCG delta: **12% of replay perf** (1.93 → 1.70). GLM-5.2 (78-layer MoE + DSA, untraceable by torch.compile): **BCG-only capture, 1.60x over eager** — this is the MoE-with-exotic-attention data point: BCG captures what neither full nor piecewise can.
- Toy forward launch counts: eager 12 launches, BCG 4, full 1. Build time: BCG 27.7 s vs tc_piecewise 106.6 s (Qwen3-235B-A22B); 35.2 vs 183.1 s (GLM-5.2).
- Regime evidence: prefill latency curves "flat across a 32x range in prompt length" — the launch-bound regime, where spacing is the whole story.
- No per-break µs published; the cost is expressed only as the 1.93→1.70 replay ratio.

### 5. Published nsys/ncu decode-window gap numbers (consumer GPUs)

**Grade B/C** — no one published a per-node µs table for vLLM/TRT-LLM decode on consumer GPUs that I could find this pass. What exists:

- NVIDIA llama.cpp blog (grade B): pre-graph A100 timeline shows decode kernels "tiny (microseconds)" with gaps dominating; post-graph gaps "very small" — never quantified.
- Practitioner guides (grade C): typical kernel-launch overhead quoted **~2-10 µs** per launch in the ungraphed decode regime on consumer RTX cards; nsys gap-hunting is the recommended method.
- **The best published decomposition is the fused MoE megakernel paper, arXiv 2609.04244 (grade A):** H200, Qwen3.5-35B-A3B-FP8, Table 1: at B=1 the step's ITL is 4.32 ms with CUDA graphs already on; **boundary (inter-kernel gap) entitlement E_boundary = 0.24 ms = 5.6% of ITL**, while intra-kernel bandwidth underutilization E_throughput = 1.14 ms. At B=1 the baseline also reports: expert GEMMs 23.0 µs total (gate+up 15.8, down 7.2), routing 3.4 µs, ~23 µs of low-bandwidth auxiliary work. Read against ~10^3 nodes/step, residual boundary cost is **~0.2-0.3 µs/node** on H200 with graphs. Mega-kernel fusion (node deletion to the limit) buys **18.7% TPOT** vs vLLM grouped GEMM at B=1 — that is the ceiling number for node-density reduction with graphs already present.

### 6. PDL production measurements

**Grade B** (mechanics well-sourced; standalone deltas weakly published — consistent with the row's "+3% class" prior).

- llama.cpp #15479/#15480 (grade A for the verdict): PDL launch attributes set at graph-capture time + `cudaTriggerProgrammaticLaunchCompletion`/`cudaGridDependencySynchronize` added at kernel entry/exit, conservatively (no logic overlap). Maintainer verdict: "a lot of changes for a marginal speedup ... largest when running very small models on very fast GPUs." Closed, not merged.
- FlashInfer: `enable_pdl` on attention/elementwise/trtllm-gen MoE kernels, **auto-enabled for >= sm90**; RFC #3605 documents it **silently no-ops below SM90**; #2913 fixed CUTLASS fused-MoE PDL crashes on SM12x via GDC. vLLM #59737 enables PDL for MoE top-k gating (no standalone % surfaced in release notes).
- TRT-LLM: `enable_pdl` spans trtllm-gen attention, FP8 GEMMs, quant, MoE; blog26 applies it to Blackwell attention. No single published all-kernel delta; the casebook skill (`TensorRT-LLM/.claude/skills/perf-optimization-casebook/references/runtime-execution/pdl.md`) frames it as hiding launch/dependency latency, i.e. sub-µs per boundary.
- **Consumer-GPU caveat:** on pre-Hopper consumer parts PDL is a no-op; even on Hopper the merged-where-adopted evidence clusters around "a few percent." PDL cannot explain or fix a 9.5 µs/node spacing — the boundary it attacks is ~0.3-0.5 µs wide.

---

## SPACING TABLE

| Engine / config | Nodes per step | Per-node cost | Step overhead | Source (grade) |
|---|---|---|---|---|
| llama.cpp pre-graph, 7B Q4, H100 | "hundreds" (not published) | gaps dominate; graph exec -40% after graphs | +14% end-to-end = 0.87 ms/token recovered | #6763 (A) |
| llama.cpp post-graph (implied residual) | ~10^3 (7B) | ~0.9 µs/node equivalent recovered; residual sub-µs | — | #6763 derived (B) |
| vLLM piecewise→FULL, A100, 7B GPTQ | attention region only (~28 layers x few kernels) | 0.51 ms/step recovered over ~60-90 launches ≈ **6-8 µs/launch** (eager, CPU-bound) | TPOT -2.9% | PR #20059 (A) |
| vLLM FULL_AND_PIECEWISE vs piecewise, small LLMs | — | spacing-dominated regime | ITL -38% | PR #20059 (A) |
| Megakernel baseline, H200, B=1, MoE, graphs ON | ~10^3 | **~0.2-0.3 µs/node residual boundary** (0.24 ms of 4.32 ms ITL) | 5.6% of ITL | arXiv 2609.04244 (A) |
| SGLang full vs BCG (breaks), GB300 prefill | 4 launches vs 1 (toy) | break cost = 12% replay perf | 1.93x → 1.70x | SGLang blog (A) |
| TRT-LLM blog20 | 23-259 graphs; ~10 MB/graph | not published (their lever is padding waste, not spacing) | up to 1.3-1.5x tput from +64 sizing | blog20 (A) |
| **Ours** | **~1,000** | **9.5 µs/node (2.3 µs work)** | **~9.5 ms window, ~7.2 ms gap** | internal |

---

## ALIGNMENT

**Our 9.5 µs/node is abnormal.** The published band for a healthy, host-decoupled graph is **~0.2-2 µs/node**: megakernel paper's residual boundary is 0.24 ms across a full MoE step on H200 (graphs on); llama.cpp recovered 0.87 ms/token by deleting gaps that were ~40% of graph-wall time pre-graph; vLLM's full-vs-piecewise delta (6-8 µs per *eager* launch, CPU-pipelined) is the worst-case published number and it applies only outside graphs. "Normal for our node mix" does not rescue it either: the H200 MoE baseline has the same mix class (routing 3.4 µs, GEMMs ~23 µs, ~23 µs auxiliary per layer) and still lands at 5.6% boundary share of ITL. Our gap share is ~7.2/9.5 = **76% of the window** — 10x the worst healthy published share.

Bug-class ranking (matches the 9.5 µs magnitude, which is *between* a graph-node dispatch ~0.5 µs and an eager launch ~3-8 µs — i.e. something is serializing the graph against the host every few nodes):

1. **Host-flag spin / doorbell poll inside the graph** (prime suspect): a tiny kernel polling mapped/pinned memory written by the host per layer or per segment burns the gap ON the SM and reproduces exactly this signature — constant per-node spacing independent of kernel work. llama.cpp's production rules exist precisely because host-coupled nodes (sync-requiring `mul_mat_id`, RPC-fresh pointers) break capture; a strata engine that lets any doorbell or semaphore-wait node into the replay gets this.
2. **Memcpy/memset/event nodes interleaved in the graph** — each adds µs-scale serialized stalls.
3. **Graph replayed in segments with host glue between replays** (accidental PIECEWISE): check the nsys lane — one graph-launch per step vs N.

Audit signature (one nsys pass answers it): (a) count graph launches per decode step; (b) look for H2D copies or `cudaMemsetAsync` inside the window; (c) find kernels whose *duration equals the inter-node gap* (poll kernels are visible as tiny, gap-length kernels, unlike true launch gaps where the GPU is idle).

**Single most rewarding structural move: the host-spin audit, before node deletion and before PDL.** The published ceilings cap the alternatives: node deletion buys at most the 2.3 µs work per removed node (and the 18.7% megakernel result shows full fusion-class deletion buys high-teens % when boundary is already only 5.6%); PDL buys ~3% class and is a **no-op below SM90** (FlashInfer RFC #3605) and "marginal" even on Hopper (llama.cpp #15480). If the audit confirms one host-coupled node per layer, deleting that coupling is worth ~7 µs x 1000 nodes ≈ 7 ms/step — an order of magnitude above everything else on the sheet, and it is a one-line capture-rule fix, not a kernel rewrite.

## Source ledger

- vLLM PR #20059 (github.com/vllm-project/vllm/pull/20059) — A
- vLLM design doc docs.vllm.ai/en/latest/design/cuda_graphs — A (qualitative; points to #20059 for numbers)
- tomasruizt.github.io/posts/09_full-cuda-graphs-spec-decode — B
- llama.cpp #6763, #6766, #6933, #7302, #15479/#15480 — A
- NVIDIA blog: developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs — B
- TRT-LLM blog20 (github.com/NVIDIA/TensorRT-LLM/.../blog20_Tuning_CUDA_Graph_Batch_Sizes...) — A
- SGLang BCG blog + yichizhang.dev/AI-Infra-Visualized/en/lessons/breakable-cuda-graph — A/B
- arXiv 2609.04244 (fused MoE mega-kernel) — A
- FlashInfer RFC #3605, PR #2913; vLLM PR #59737; TRT-LLM pdl.md casebook — B
