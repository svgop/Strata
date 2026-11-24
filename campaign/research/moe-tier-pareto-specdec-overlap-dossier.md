# Magma dossier — Tier Pareto × speculative decoding × captured-graph overlap (qwen4exp, RTX 5090 + 9950X3D)

Method: fresh fetches (arXiv HTML/abs, GitHub API, CUTLASS docs) on 2026-11-20; continuity from the two prior dossiers. All times are derived arithmetic on stated constants unless a citation is attached. Fetch status per source in the table.

## 1. Bandwidth arithmetic — when CPU-exec beats PCIe streaming

Constants: slot 2.637 MiB = 2.765 MB; FLOPs/token/slot = 2·3·D·F = 9.83 MFLOP (gate, up, down; D=2560, F=640); activations H2D+D2H = 10.24 KB/token/slot; PCIe 50–64 GB/s measured; AM5 ~90 GB/s; HBM 1.79 TB/s; CPU realistic 1.5–2.4 TFLOPS FP32.

Per-slot latency (serial, no overlap):
- PCIe transfer 43.2–55.3 µs, batch-invariant to b≈200 (activation RTT adds 3.2 µs @b=1, 9.0 µs @b=32).
- GPU exec after arrival: 1.54 µs (HBM-bound) + launch ≈ 5 µs/slot.
- CPU exec: memory-bound 30.7 µs (slot from DRAM) until the compute knee: arithmetic intensity 3.55·b FLOP/B vs machine balance 17–27 FLOP/B → knee b* ≈ 6–9. Compute term: b=8 → 33–52 µs; b=32 → 131–210 µs.

Crossover: **b=1**: CPU ~34 µs/slot vs 46–59 streaming → CPU 1.4–1.7× faster per slot. **b=8**: 38–57 vs 47–60 → wash. **b=32**: 140–219 vs 51–65 → streaming wins 2.1–3.4×. Aggregate b=1, all 480 slots cold-but-RAM-resident: CPU 14.3 ms/token vs PCIe 20.7–26.5 ms — but AM5's 90 GB/s is shared with NVMe staging fills (~12–14 GB/s Gen5) and the host PLE gather, so the theoretical margin evaporates in contention. Published regime check: Fiddler (2402.07033, fetched) wins 1.26× @bs=1, 1.30× long prefill, 11.57× beam search — CPU-exec pays only when activations amortize, i.e., the b*≈6–9 knee. SSD-LLaMA (2609.18110, fetched), your machine class, measured: "moving one and six experts to the CPU increases TPOT from 513.95 to 529.52 ms" (other split 573.73 ms); "No tested CPU–GPU split therefore improves over the all-GPU path." Verdict: CPU-exec ties at b=1, is Pareto-dominated from b≈8; keep a Fiddler-style per-layer arbiter, expect it dormant on AM5. Dense arena: 9.9 GiB/pass = 5.94 ms on HBM vs 193 ms PCIe, 118 ms CPU — VRAM-only. PLE rows are gather-only; CPU-exec n/a.

## 2. Speculative decoding as tier-latency hider

Anchors: DeepSeek-V3 (2412.19437, fetched): MTP second-token acceptance 85–90%, 1.8× TPS. TRT-LLM blog02 (snippet-verified, two raw-URL hits at commit a4349b2cd9a5): nextn=3 best at bs=1 → **2.16×**; relaxed acceptance → 2.33×; CUDA-graph min-latency path → 7.22×. EAGLE-3 (2503.01840, fetched): up to 6.5×, +1.4× over EAGLE-2, 1.38× throughput @bs=64 in SGLang.

Tier interplay is arithmetic: a verify pass reads the **union** of drafted tokens' experts once per slot. Measured next-token expert reuse: 44–48% (Qwen3-30B-A3B) and 0.444 in llama.cpp router traces (2610.34440, fetched). With u=0.44 dedup, verify-pass bytes ≈ 480·2.637 MiB·0.56 ≈ 0.72 GiB → 11–14 ms CE; amortized by 2.16× MTP → **~6.3 ms per emitted token ≈ the 5.9 ms dense-SM floor** — the point where the engine flips CE-bound → SM-bound. Acceptance vs expert-cache state: **no published work found conditioning draft proposals on expert residency**; closest is 2610.34440 — best draft length is a monotone staircase in acceptance rate, but acceptance is treated as exogenous to cache state. EAGLE-3 drafts feature-level (dense head), so acceptance is structurally insensitive to target-side expert residency; only the verify-pass union is. Magma steering option (unpublished): rank n-gram/PLE-derived candidates by the MTP-head-predicted routes' residency, gated by 2610.34440's staircase threshold (acceptance loss must buy bytes). Cache-admission precedent: MoE-Infinity 2401.14361 (continuity-verified) promotes experts by activation traces — promote on accepted-draft recurrence.

## 3. Expert-skip under cache pressure (AdapMoE)

AdapMoE (2408.10284, fetched, ASPLOS'25): per-expert perturbation metric with threshold T — "If the perturbation is less than the threshold T, the gating system can safely activate a smaller subset of experts." −25% activated experts, 1.35×, no accuracy degradation vs fixed top-k (heuristic probability cutoffs did cost accuracy — the bounded-perturbation framing is the mechanism). For Magma: under pool thrash, execute top-7/8 of 10 → −0.31–0.62 GiB/token PCIe. Quality risk concentrates where router score gaps at ranks 7–10 are small (exactly what T measures). KLD-gate per layer before enabling.

## 4. PagedMoE prefix-bitplane partial fetch

PagedMoE (2611.03173, fetched): each expert stored once as a prefix-compatible tree codebook + bitplane layout — lower-precision forms are valid prefixes; a cached 2-bit expert extends by fetching only missing bitplanes. Pre-I/O allocator picks per-expert bit width from current routing weights + offline per-expert error profiles under an average-bit budget; cached 1–8-bit prefixes execute directly, never materializing full precision. Numbers: 2.5-bit average routed precision ≈ GPTQ-INT4 (Qwen3.5-397B-A17B) / MXFP4 on most GLM5.2 tasks; 8.89 tok/s SSD-resident @24 GiB cache = 1.33× independent-precision objects, 1.56× uniform-4-bit / ik_llama.cpp. Bandwidth lever: cold slot at 2.5 effective bits ≈ 1.69 MB → 26–34 µs PCIe vs 43–55 µs (1.6× bytes); refinement bitplanes can ride a later pass — a cold expert becomes approximate-but-executable within one layer budget (~124 µs SM/layer). SSD-LLaMA's independent lever: rANS-compressed experts on SSD, GPU-side decompress into VRAM slots (fetched).

## 5. Captured-graph overlap recipe (CE + PDL)

Semantics (CUTLASS doc, fetched): PDL lets two same-stream kernels overlap on Hopper/Blackwell. Launch consumer with cudaLaunchAttributeProgrammaticStreamSerialization=1; producer calls cudaTriggerProgrammaticLaunchCompletion() (griddepcontrol.launch_dependents); consumer calls cudaGridDependencySynchronize() (griddepcontrol.wait) after output-independent prologue loads.

Measured caveat (FlashInfer PR #7885, fetched): B200-class, driver 615.71.10 — eager: 0/320 consumers start early; CUDA graph: 320/320 start early by 0.4–1.8 µs (all-SM producer; up to ~400 µs with 8-block producers). **PDL early-start exists only under graph capture on current drivers.** Accuracy hazard: TRT-LLM PR #8318 (snippet-verified) — misplaced PDL pre-exit signaled completion before consumers read results (nvbugs 5545522); validate bitwise after enabling.

Recipe: (1) capture each layer as fork/join DAG — SM stream (attention, dense, MoE GEMMs) + CE stream (H2D of next-layer expert union from pinned ring), events at capture so memcpy nodes are graph nodes on the CE lane; (2) PDL-chain same-stream kernels: router → permute → expert GEMM → unpermute → residual; (3) budget: SM busy 5.94 ms/pass vs 480 cold slots = 20.7–26.5 ms CE → **PCIe is the b=1 bottleneck**; CE hides only ~2.5 slots/layer (124 µs SM budget) → **VRAM hit floor ≈75% before SM-bound**; (4) PCIe is one shared lane — NVMe staging DMA and victim-cache fills contend; ~50–64 GB/s is ~3% of HBM so CE-vs-SM controller contention is negligible; CE count finite — read cudaDevAttrAsyncEngineCount, don't assume parallel lanes; (5) pinned staging mandatory — pageable H2D forces staging syncs that break CE overlap and capture (CUDA programming guide; cited, not fetched).

## Decision matrix (per slot unless noted; ⋖ = overlappable via §5)

| payload ↓ / tier → | VRAM | RAM | NVMe | CPU-exec |
|---|---|---|---|---|
| Dense 9.9 GiB/pass (F16) | **5.94 ms** SM-bound, b-invariant ✅ | 193 ms ❌ | worse ❌ | 118 ms ❌ |
| Hot expert (VRAM-resident) | ~5 µs all b (1.5 HBM + launch) ✅ | — | — | — |
| Warm/cold expert (RAM) | — | 46–59 / 47–60 / 51–65 µs @b=1/8/32 ⋖ | n/a | **34 / 38 / 140 µs** @b=1/8/32 (needs RAM-resident + quiet DRAM) |
| Warm/cold expert (NVMe) | — | staging hop | 190 µs @14 GB/s + PCIe leg ⋖ (pipelined ⇒ ≈ PCIe leg; rANS compression cuts bytes) | worse |
| PLE row 160 B (b-invariant) | L2/HBM, ns ✅ | 1.8 ns BW + ~80 ns latency ✅ | 20–100 µs latency-bound (4 KiB page = 0.29 µs BW); prefetch 16 rows ≈ 5 KiB ⋖ | n/a |

Aggregate @b=1, 480 slots/token: VRAM hits free; RAM stream 20.7–26.5 ms ⋖; CPU-exec 14.3 ms uncontended (tie after contention); AdapMoE skip −25% ⇒ −26% bytes; PagedMoE bits ⇒ −38% bytes.

## Transfer to Magma
1. CPU-exec: implement the Fiddler-style per-layer arbiter but expect it dormant (tie @b=1, dominated ≥b≈8); its niche is absorbing CE-lane bursts while DRAM is quiet. Do not spend core time here first.
2. Spec-dec is the cheapest tier lever: MTP nextn=3 @bs=1 (2.16× published) + dedup verify (reuse 0.44) converts the 20–26 ms CE floor to ~6.3 ms/emitted token ≈ SM floor. Sequence: dedup verify → MTP → then NVMe complexity.
3. Cache admission: promote experts recurring in accepted drafts (MoE-Infinity precedent); target ≥75% VRAM hit to flip CE-bound → SM-bound.
4. AdapMoE T-valve under thrash: −25% experts ≈ −25% PCIe; KLD-gate it.
5. PagedMoE bitplanes for NVMe-tier cold slots: 1.6× byte cut at GPTQ-INT4-adjacent quality; partial-fetch refinements ride later passes.
6. Graph: fork/join DAG with CE memcpy nodes + PDL chains; bitwise-validate (PR #8318 hazard); PDL early-start is graph-only on current drivers.
7. Measure first: cudaDevAttrAsyncEngineCount; single- vs multi-CE H2D on 5090; your own draft-union reuse (llama.cpp traces: 0.444).

## Source table

| # | Source | Access (2026-11-20) | Datapoint used |
|---|---|---|---|
| 1 | Fiddler, arXiv 2402.07033 | fetched (HTML v3) | 1.26×/1.30×/11.57× regimes; offline-prof hot placement |
| 2 | SSD-LLaMA, arXiv 2609.18110 | fetched (HTML) | CPU split 513.95→529.52/573.73 ms; io_uring batches; rANS GPU decompress; 2.10–15.58× decode |
| 3 | AdapMoE, arXiv 2408.10284 | fetched (HTML) | perturbation threshold T; −25% experts, 1.35×, no loss |
| 4 | PagedMoE, arXiv 2611.03173 | fetched (HTML v1) | prefix-bitplane tree; 2.5-bit ≈ GPTQ-INT4; 8.89 tok/s; 1.33×/1.56× |
| 5 | DeepSeek-V3, arXiv 2412.19437 | fetched (HTML) | MTP acceptance 85–90%; 1.8× TPS |
| 6 | EAGLE-3, arXiv 2503.01840 | fetched (abs v3) | ≤6.5×; +1.4× vs EAGLE-2; 1.38× @bs=64 |
| 7 | Spec-dec for MoE, arXiv 2610.34440 | fetched (abs + HTML v1) | dedup bytes model; reuse 44–48% / 0.444; draft-length staircase |
| 8 | TRT-LLM blog02, commit a4349b2cd9a5 | snippet-verified ×2 (direct fetch mis-rendered) | nextn=3 → 2.16× @bs=1; 2.33× relaxed; 7.22× graph |
| 9 | FlashInfer PR #7885 (+comments, issue #6955) | fetched (GitHub API) | PDL early-start: eager 0/320, graph 320/320, 0.4–1.8 µs |
| 10 | CUTLASS dependent_kernel_launch doc | fetched | PDL semantics, Hopper/Blackwell |
| 11 | TRT-LLM PR #8318 (nvbugs 5545522) | snippet-verified | PDL PREEXIT accuracy hazard |
| 12 | Prior dossiers (workspace) | read | PLE 160 B / 16-row ≈5 KiB shape; MoE-Infinity 2401.14361; llama.cpp #31469/#30966 |

CUDA programming guide + cudaDevAttrAsyncEngineCount: cited from docs.nvidia.com, not fetched.