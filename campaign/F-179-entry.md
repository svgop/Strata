
### F-179 — THE ROUND-9 DEEP STUDY: the engine is execution-bound, not bandwidth-bound — the budget quantified and the barrier named (2026-10-14 ~02:30, goal round 9)

Three study legs (two deep code agents + one external scan) + operator paper reads (megakernels/Hazy,
Mirage MPK, CUDA conditional nodes, SSD overlap, ReplaySSM, the empirical spec-decode anatomy).

## THE THESIS (all three legs converge, with numbers)

**One verify window = one CUDA graph of ~1,800 kernel nodes (48 layers x 35-44 nodes), 17.12 ms wall
at single-stream (12.73 ms at 6-conc), carrying 7.35 GB — ~9.5 µs effective node spacing vs ~2.3 µs
of bytes per node. The engine streams at 24-26% of GDDR7 peak (1792 GB/s).** The per-layer "fixed
cost" is ~70% node spacing + under-filled small kernels + doorbell handshakes, ~30% real traffic.
Independent confirmation: arXiv 2605.30571 measures H100 batch-1 decode at 27% of peak HBM with
~283 launches/step — our regime is the field's regime; graphs alone capped theirs at 1.26x.

**The honest ceiling without touching bytes: ~2.9x (176 → ~500 t/s class).** Every real cut must
remove nodes or waits, not FLOPs. (And at 24-26% effective BW, byte cuts are DEVALUED: 1 GB saved
≈ 2 ms now vs 2.4 ms at the floor — fix execution first.)

**The byte budget** (@T=3.6): routed experts 3.28 GB (45%), dense projections 1.78 GB, hc
(hyper-connection) reads 1.26 GB (bf16 [320x10240] 2x/layer!), head 0.52 GB (Q6_K 248320x2560),
GDN state 0.23, KV/idx 0.15, shared/router 0.25, activations ~0.2. **The dense skeleton is 55% of
bytes** — read every layer every window regardless of T; the "it's all expert traffic" MoE mythology
is wrong at this geometry. Draft chain is only 1.1-1.2 ms (~6-9% of a round); verify 87-90%; commit
1.10 ms (a single-buffer hazard on `h_commitb_`, not a data dependency — verify.cpp:2977).

**The worst trade in the window**: VRAM entry hit is 97.4-98.0%, but the doorbell/host path engages
unless ALL 20-36 entries of a layer-window are resident — ~half of layer-windows pay a host round
trip (pool 1.56-2.10 ms/window + interleaved GPU wait-nodes) for a 1.8-2.4% CPU-expert tail
(0.5-0.6 CPU entries per layer-window). Env-tunable today via ROUTE_RESIDENT margin/profile.

## THE STRENGTHENED QUEUE (merged round-8 + round-9; the delta vs F-178 marked NEW)

**Tier 0 — env-only, free:**
- R9: F-67 overlap unblock (unchanged; +2-6% conc, two-boot Windows evidence).
- NEW **STRATA_QFUSE=1**: hc-read writes the consumer's q8_1 inline (verify.cpp:1065-1069) — −48
  quantize nodes/window, exists, OFF, never tested here.
- R10: PDL (`STRATA_DF_PDL=1`) — now production-backed as THE field answer to node spacing
  (SGLang applies it across Gate-Up/activation/Down/top-k; TRT-LLM ships it globally on Blackwell;
  FlashInfer on-by-contract). Sustained-reasoning gate mandatory (F-107's −40% was WDDM).
- NEW **diagnostic boot**: `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1 STRATA_VERIFY_NODES=1`
  — replaces every estimate in the budget table with measurements. Zero risk; should precede all
  Tier-1 code work.
- NEW **ROUTE_RESIDENT margin sweep**: kill the CPU-expert doorbell tail (~2-3 ms/window potential;
  the mec RFC's +84% on our model family is the same mechanism in a harsher regime). MMLU-gated
  (round-1 note: ROUTE_RESIDENT changes the output).
- R5b: tau 0.15/0.10 + STRATA_SPEC_RATIO=1 (unchanged, MMLU-gated).
- R8: clocks + power cap + Xid-79 firmware check + iGPU cable (user steps; variance first).

**Tier 1 — code, days:**
- NEW **commit double-buffer**: `h_commitb_` hazard fix returns most of 1.10 ms/window.
- NEW stream-priority one-liner (verify > adapt; mtp.cpp:461 pattern exists).
- **Ungated resident share** (partial device plan): extend resident_plan_kernel to plan resident
  groups of partially-resident layers, gate only the miss share — removes the host from the GPU
  critical path ≈ 1.5-2.5 ms/window (9-15%). [1-2 weeks]
- **Fenced batched admission**: 30-56 ms → 10-20 ms (28 device syncs → event fences + coalesced KV
  images). Attacks the 184-vs-210 rows/s gap and tail latency.

**Tier 2 — research→code, weeks:**
- **F-31 slot coupling** (the concurrent ceiling breaker) + **device-driven draft chain**: CUDA
  conditional nodes / device graph launch are green field — NO production engine uses them
  (SGLang/TRT-LLM/vLLM all stopped at full/piecewise graphs; PyTorch doesn't expose them); the
  scaffolding exists here (mapped spin gates, E-6 device plans, skip words). sm_120 supports both.
- **Node-count fusion program**: router+route_resident fusion (−48 nodes), QSA epilogue fusions,
  BCG-style segment breaks for the CPU-expert path (SGLang's Breakable CUDA Graph pattern — the
  non-PyTorch idea is API-simple), TRT-LLM-style router-GEMM+topk single node.
- FastMTP (Tier 3 [training], unchanged): acceptance multiplies window amortization (rows/s ∝ T at
  fixed window cost) — orthogonal to and multiplicative with all of the above.

## DEMOTED BY ROUND-9 EVIDENCE

- **Draft-verify overlap / SSD-style pre-speculation**: draft is 6-9% of the round — overlap caps
  at ~6-9%. (SSD's 30% needs a second GPU.)
- **ReplaySSM as a direct lever**: GDN state is 0.23 GB/window; halving saves ~0.1 ms. Its
  checkpoint+rollback pattern stays relevant only as an F-31/overlap enabler.
- **Megakernel from scratch**: the window is ALREADY one graph launch; megakernel benefits
  decompose into node-count reduction + device-driven execution, reachable via cheaper paths. MPK
  is PyTorch-only (not drop-in for hand-written CUDA).
- **hc bf16→Q8 byte cut**: worth ~0.4 ms at current execution density — LOW until node spacing
  is fixed (then it becomes the top byte cut).
- L2 persisting windows for MoE: no published MoE-decode measurement exists anywhere (open slot);
  DSMEM/clusters: zero decode-engine adoption.

## KEY EXTERNAL SOURCES

Hazy Research megakernels (blog 2025-05-27; paged shmem + counter sync; batch-1 dense only);
Mirage Persistent Kernel (arXiv 2512.22219; 1.0-1.7x, PyTorch-only); SGLang HPC-Ops PDL
(lmsys 2026-08-07); SGLang Breakable CUDA Graphs (lmsys 2026-08-17; BCG 17% over piecewise);
TRT-LLM DeepSeek-V4 Blackwell blog26 (router 8→3 µs, top-k 384→112 µs, GVR routing-correlation
top-k 1.40-2.17x kernel-level, MegaMoE +15.3%); llama.cpp #28248 mec RFC (device-side expert-id
indirection; +84% Qwen3.8-Flash-Next all-CPU 4090); llama.cpp #27050 (5090: ~2.7 ms idle per graph
launch, sampling D2H costs); arXiv 2605.30571 (27%-of-peak batch-1, graphs cap 1.26x); arXiv
2607.17283 (K=6 optimal plateau; "the speedup is a systems property"); SSD (arXiv 2603.03251);
ReplaySSM (NVIDIA blog; GDN covered; no gain at batch 1); FastMTP (Red Hat, Sep 2026); conditional
nodes (NVIDIA blog 2024; IF 12.3, WHILE 12.4, ELSE/SWITCH 12.8, CUPTI bodies 13.1).

No experiments run this round (study-only per operator). Engine untouched; standing recipe serving.
