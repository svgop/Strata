# T17D — sm_120 KERNEL FRONTIER: what 2024-2026 Blackwell/CUDA 13.x features can still unlock on the RTX 5090 decode path

Date: 2026-10-17. Scope: kernel-level candidates for the consumer-Blackwell (sm_120, RTX 5090, CUDA
13.4) build only, priced against the measured landscape (execution-bound windows, node spacing ~6us,
hc line = #1 kernel line ALU/cvt-bound, VRAM-hits line at 87-100% roofline, QSA decode attention
4.40 ms/window width-growing, gr_down serial chain 3.8x off byte floor). READ-ONLY session: no
builds, boots, API calls, or git ops; the only file written is this dossier. Hard laws honored: PDL
is CLOSED by F-191 (no re-queue); T15A/T15C/T16 queued rungs are referenced only as sequencing
constraints, never re-priced as new; every external claim carries a numbered source; every internal
claim carries file:line.

---

## 1. Executive summary

1. **The platform facts are now pinned by primary sources.** On sm_120: TMA (`cp.async.bulk`,
   `.tensor`) EXISTS and works (empirically driven on a 5090 [S15]; CUTLASS SM12x kernels use it
   [S16]); thread-block clusters exist (portable max 8 [S1]; we already ship cluster kernels behind
   `STRATA_QSA_CLUSTER`, src/kernels/decode_cluster_parity.cpp:1-15); **tcgen05/TMEM/wgmma DO NOT
   exist** (measured on GB203: "tcgen05 is yet to be supported for the arch. sm_120a", wgmma
   unsupported [S24]; PTX forum [S11]; Colfax [S16]); Cluster Launch Control (CLC) DOES exist (CC
   10.0+ per the CUDA guide [S6], family targets sm_120f [S14]); conditional graph nodes (IF/ELSE/
   SWITCH/WHILE) exist (CUDA 12.4/12.8; CC 9.0+ [S4][S8][S9]).
2. **The one number that re-prices everything compute-side** (CUDA 13.4 Best Practices Guide,
   Table 5, CC 12.0/12.1 column [S2]): `add.f16x2` / bf16 packed FMA = **64 results/clk/SM** while
   `add.f32` = **128** — i.e. **there is NO packed-fp16/bf16 CUDA-core rate advantage on consumer
   Blackwell** (the Hopper 2x is gone), and the SIMD/video pipe halved (`vabsdiff4` 64→32). Two
   consequences: (a) the hc line's unpack8 kernel is capped by the 64/clk integer/shift/cvt pipe at
   ~half the FFMA pipe (deep-dive A), so a packed-dot rewrite is worth ~2x on THAT cap, not more;
   (b) the big compute leaps must come from the **tensor pipe** (mma.sync bf16/f16/s8, f32
   accumulate — the only fast path sm_120 has) or from **issue-count removal** (dp4a/TMA/256-bit
   loads).
3. **Ranked leaps** (table in section 3): (1) tensor-core hc read line via mma.sync m16n8k16 with
   the 4-stream batch on the N axis (architecturally full-tile even at T=1) — the only mechanism
   with a >10% M1 ceiling left in a single kernel line; (2) packed-f16x2 chunked-f32 hc dot — ~half
   the effort, ~half the ceiling of (1); (3) mma port of QSA decode attention (the prompt-attention
   mma engine already in-tree + the TRT-LLM XQA/FlashInfer SM120 existence proofs) — attacks the
   conc ceiling line; (4) CLC persistent layer-group kernel — the ~2x spacing-collapse gamble, now
   with a shipped-on-SM12x precedent and preemption semantics that fit our clock behavior; (5)
   conditional nodes around the flag-spin and empty-PCIe nodes — a bounded M-effort conc-favored
   cut. TMA alone is an enabler, not a win (transport is not our bottleneck [S15]); PDL stays
   closed; green contexts stay queued; FP4/tcgen05-class paths stay dead on this pack.
4. **CUDA 13.0-13.4 contains no new launch-latency magic for this engine** (13.0's relevant items
   are 256-bit vector types and dropped pre-Turing targets [S10]; 13.1 lands the green-context
   runtime API [S40][S41]; 13.4's release-note perf item is FP4 matmul on Blackwell Ultra [S9]).
   The usable 13.x-era mechanics are the 12.8/12.9 graph items (ELSE/SWITCH conditional nodes,
   `cuMemcpyBatchAsync`, family cubins sm_120f) [S8][S14].

---

## 2. Platform facts for sm_120 (what is real, with the source for each)

| Feature | On sm_120? | Evidence |
|---|---|---|
| TMA `cp.async.bulk` (+`.tensor`, mbarrier) | YES (no cluster multicast focus) | driven on a 5090: `cp.async.bulk.tensor.2d` double-buffered with mbarriers beat cuBLAS batched SGEMM 1.4-1.7x [S15]; "TMA: Available on SM12x" [S16]; PTX 8.7 added the sm_120 target [S3] |
| Thread-block clusters + DSM | YES, portable max 8 | Blackwell tuning guide: portable 8 (16 non-portable on B200 only) [S1]; our own cluster kernels run today (decode_cluster_parity.cpp:1-15) |
| tcgen05 / TMEM / wgmma / 2-CTA MMA | **NO** | "the wgmma instruction isn't supported in GB203 and the tcgen05 instruction hasn't been implemented yet for GB203"; "tcgen05 is yet to be supported for the arch. sm_120a" [S24]; sm_120 excluded from tcgen05 targets [S11]; "SM12x does not use tcgen05 or TMEM; instead, like with SM8x it uses warp-level mma.sync" [S16] |
| mma.sync m16n8k16 bf16/f16, m16n8k32 s8 (+blockscaled fp4/fp6/fp8 kinds) | YES | QMMA SASS observed for fp8/fp6/fp4, mma present on GB203 [S24]; NVFP4 blockscaled via `mma.sync.aligned.kind::mxf4nvf4...m16n8k64` on SM12x [S16]; in-tree: our sm_120 build runs mma.sync m16n8k8/k16 in qsa_prompt_attn.cu:49-66 today |
| Tensor-core dense rates (5090) | FP16/BF16 209.5 TFLOPS (419 sparse), FP8/INT8 2x that | whitepaper figures via NVIDIA forum clarification threads [S12][S13] |
| CUDA-core packed fp16/bf16 FMA | **64/clk/SM = HALF the FP32 pipe (128)** | CUDA 13.4 Best Practices Guide Table 5, CC "12.0/12.1" column [S2] |
| cvt f32.f16 / int shifts / AND | 64/clk/SM each | same table [S2] — the unpack8 cap |
| dp4a | same video-SIMD family, halved on CC 12.0 (`vabsdiff4` 64→32) | same table [S2]; exact int32 accumulate semantics unchanged |
| Cluster Launch Control (work stealing) | YES (CC 10.0+, family target) | CUDA guide: "a feature introduced in the NVIDIA Blackwell GPU architecture (compute capability 10.0)" [S6]; sm_100f/sm_120f family targets [S14] |
| Conditional graph nodes IF/ELSE/SWITCH/WHILE | YES (CC 9.0+) | guide: condition evaluated ON DEVICE, `cudaGraphSetConditional`, body graphs allow kernel/memcpy(pinned-mapped)/memset/child/conditional, NO device-graph-launch inside bodies [S4]; ELSE+SWITCH added CUDA 12.8 [S8]; CC 9.0 requirement [S44] |
| Device graph launch limits | 120 fire-and-forget total, 255 pending tail, 1 self-launch | [S4] |
| PDL in graphs | YES (CC 9.0+) — but F-191 measured it a no-op here | semantics: wait blocks until primaries "completed and flushed"; overlap = preamble + launch latency only [S5]; F-191 (SVG-CUSTOM.md:6444) |
| Green contexts | driver API 12.4+, runtime API CUDA 13.1 | [S7][S40][S41]; queued T9#3, unchanged |
| Occupancy limits | 48 warps/SM, 32 blocks/SM, 128 KB smem/SM, 99 KB/block | [S1]; matches the measured ~99 KB on GB203 [S24] |
| FA4 / tcgen05 attention kernels | **NO** | FA4 targets sm_100/sm_103; consumer cc 12.0 unsupported [S27][S28][S29] |
| XQA-style MMA decode attention | YES, shipping on SM120/121 | FlashInfer ships TRT-LLM's open-sourced XQA decode kernel on "Blackwell RTX and DGX Spark (SM120/121)" [S25]; XQA open-sourced via TRT-LLM PR #3762 [S30] |

---

## 3. Ranked candidate table

Pricing convention: Δms/window on the T13A anatomy (11.4-12.0 ms window, 2.47 tok/window standing
[T13A]); % M1 = Δ/(window−Δ) at fixed acceptance. Percentages, not absolute t/s — the T16 #1 clock
re-bank means the frozen five-boot means predate the 3135 lock (SVG-CUSTOM-TODO.md T16 #1).

| # | Candidate | Mechanism | Priced magnitude | Effort | Cheapest falsifier | Sources |
|---|---|---|---|---|---|---|
| 1 | **TC-hc**: mma.sync bf16 (f32-accum) hc read kernels, 4-stream batch on the N axis | Weights are bf16 in the pack — zero unpack; xn staged once; m16n8k16 tiles with N = HC(4)×T (full n8 tile even at T=1); kills the 64/clk int-pipe cap AND the serial 320-FMA chain | hc line 2.87 ms/window [T13A #2]; byte floor ~0.72 ms (26.2 MB/layer×48 at ~1.76 TB/s); realistic −0.9..−1.6 ms/window = **+9..17% M1**; ceiling +21% if the ALU-bound share holds | L (6 kernel classes in fused_gr.cu; staged/cp.async substrate exists at :855-925) | Offline `fused_gr_bench` (STRATA_BENCH_QMIXED hook, fused_gr_bench.cpp:225) prototype at (T=1..6, D=10240, LR=320): if the mma variant < 0.7x of the staged kernel per read, close for one binary; MUST bench on top of the queued T15A/T15C rungs to avoid double-count | [S2][S12][S16][S24] |
| 2 | **Packed-dot hc** (f16x2 chunked-f32 promotion; no unpack) | xn converted to f16x2 once per tile; weights consumed native-packed; `__hfma2` chains with f32 promotion every 16-32 k-steps (bounds the f16-accum error); halves chain length AND issue count (24 ops→4 per 8 MACs) at the 64/clk packed rate = ~2x the current int-capped 64 MACs/clk | −0.4..−0.8 ms/window = **+4..8% M1** | M (same files; numerics harness reused) | Same bench as #1, plus one KL spot-probe of the promotion interval sweep {8,16,32}; different bits → full gate per law | [S2] |
| 3 | **TC-QSA-decode**: port the prompt-attention mma engine into `attn_chunk_kernel` | Scores s[cell][qh] = K(cells×256)·Q(256×12→pad16): m16n8k16, M=cells (16-aligned, width-growing = free), N=G=12 q-heads, K=256 = 16 steps; PV identical shape; f16 pools exact-element, int8 pools via the in-tree exact-fp16 encoding (qsa_prompt_attn.cu:67); decode attention today is dequant+FFMA per cell (qsa_decode_attn.cu:34-70, 202, 494-507) | attention decode line 1.96-4.40 ms/window class; −0.3..−0.8 ms at conc-heavy = **+3..8% conc**, grows with context (the width-growing ceiling line) | M-L (engine exists in qsa_prompt_attn.cu:49-66 + ldmatrix in native_qsa_score.cu:42-47) | Offline parity + microbench at cells∈{2048, 8192, 32768}, T=1..8 vs attn_chunk_kernel (the decode_cluster_parity pattern); if <1.3x at cells=2048, the win is long-ctx-only | [S12][S24][S25][S30] |
| 4 | **CLC persistent layer-group kernel** (scoped upgrade of the banked G1 gamble) | One persistent kernel per layer-group walks the control chain with device flags; CLC (`clusterlaunchcontrol.try_cancel`, mbarrier 16B response) steals tiles with preemption safety and no SM-count knowledge — fixes static-persistent tail imbalance and is preemptable (fits our clock/power behavior, unlike hard cooperative pins) | The ~2x ceiling play: node spacing ~6us × ~2050 nodes ≈ the whole window [T12-GPU]; compressing toward the 4.3 ms light-class byte floor | XL | 20-line sm_120 CLC compile+run proto (PTXAS 13.4) + a 2-layer persistent microbench vs the captured graph slice; gate any adoption on F-223-class discriminators | [S6][S14][S16][S17][S18][S19][S25] |
| 5 | **Conditional nodes around spins + empty-PCIe** (advances T12-GPU #7's "green field" with 12.8 mechanics) | waitA/waitB/waitCPU spin kernels (verify_kernels.cu:1487 `wait_flag_ge <<<1,1>>>`) and the always-launched PCIe grouped node (verify.cpp:1709) become IF bodies around the real work; condition evaluated on device — a satisfied flag skips the spin node entirely, no host round-trip (contrast F-222's T1.1: host flagA RTT 0.26 ms) | −0.2..−0.5 ms/window = **+2..5% M1**, more at conc where flags wait longer | M (explicit graph construction or BeginCaptureToGraph surgery; handle-default resets per launch) | Standalone 20-node graph harness: replay with satisfied/unsatisfied device flags vs the spin-node baseline — no server needed | [S4][S8][S44] |
| 6 | **TMA as the substrate of #1/#3** (not a standalone win) | `cp.async.bulk` 1D for the 20 KB W rows / 2D tensor map for W tiles; mbarrier double-buffer replaces per-thread `cp.async16` chains (fused_gr.cu:817-837) — frees LSU/issue slots on an issue-bound kernel; zero standalone value when transport is not the bottleneck | 0 alone; folded into #1 | rides #1 | nsys on gr_down: if LSU pipe <20% busy, TMA-alone is pointless (expected) | [S15][S16] |
| 7 | **cuMemcpyBatchAsync** for the copy_rows/PCIe-slot chains | One batched API call replaces the per-row D2D chain (verify.cpp:1711-1730) and the ~80 KB doorbell publish path; fewer nodes AND fewer driver round-trips | +0..1% M1; node-count law (F-205/F-206) says wall time only moves if the copies are latency-bound — measure first | S | nsys node count + spacing on one window with/without | [S8] |
| 8 | **dp4a/IMMA q32 path** (parked behind QFUSE) | The QFUSE hc-q32 kernel (fused_gr.cu:1628-1740) dequantizes int8→float4 then FFMA; dp4a (exact int32) or mma.s8 removes 64/clk converts → ~2x that path | Conditional: q32 runs only under QFUSE (OFF standing, verify.cpp:176) — priced only if the QFUSE/HC_Q8 adjudication arms it | M | dp4a microbench at (LR=320) — one offline binary | [S2] |

**Do-not-build (source-priced):** FA4/tcgen05 anything [S27][S28][S29][S11]; FP4/MXFP4 tensor decode
matvec (bytes INCREASE at bs1-2 — banked T7.C; per-tile scale factors add reads [S16]); router GEMM
via mma (the router line's cost is top10+plan+bell, its GEMV share ~0.27 ms of a 3.39 ms AMD-era
measure — router_top10.cu:30-33; 2x on the GEMV ≈ 0.1 ms); TMA-multicast (cluster multicast is
sm_90a/sm_100a-optimized, not an sm_120 play [S3]); tensor-core decode EXPERT matvec (already
closed by T7.C byte arithmetic — unchanged).

---

## 4. Deep dives

### A. The hc line: why it is slow, and what the 13.4 table changes (candidates 1, 2, 8)

Anatomy (fused_gr.cu, compiled D = HC·N = 4×2560 = 10240, LR = 320; DOWN_BLOCKS = LR/8 = 40,
UP_BLOCKS = 80 — :36-40):
- `gr_down_kernel` (:81-137): one warp per output row; the row loop `for (j = lane; j < D/8; j +=
  32) acc += dot8(__ldg(w4+j), xn+j*8)` (:127-128) is a **single serial 320-FMA chain** (D/8/32 =
  40 chunks × 8 FMAs into ONE accumulator). Each dot8 first pays unpack8 (:51-58): per uint4, 4
  shifts + 4 ANDs (integer, 64/clk pipes [S2]) before 8 FFMAs (128/clk).
- **The cap arithmetic**: per 8 MACs the kernel issues 8 integer ops (64/clk) + 8 FFMAs (128/clk).
  The integer/cvt side delivers at most 64 MACs/clk/SM — **half the FP32 pipe**. T12-MEM's verdict
  ("hc reads are ALU/cvt-bound, not bandwidth-bound"; 3.93/1.25 ms vs 26.2 MB/layer,
  SVG-CUSTOM-TODO.md T12 #2) is exactly this cap. It is NOT primarily the serial chain's latency:
  41 blocks × 8 warps = 328 warps on 170 SMs (2 warps/SM) gives each chain the whole machine, but
  the issue mix itself is half-idle FFMA slots.
- **Candidate 2 (packed f16x2) re-pricing**: on CC 12.0 the packed pipe is 64 INSTRUCTIONS/clk × 2
  ways = 128 MACs/clk — 2x the current int-capped 64 — but NOT 4x, and not the Hopper-class 2x
  over FP32 that `f16x2=128 on CC 9.0` would suggest [S2]. The conversion of xn to f16x2 is
  amortized over LR rows (down) / 2560 rows (up) — negligible. Accumulation: pure f16/bf16
  accumulation over 320 terms is outside our numerics comfort; the chunked-f32 promotion (promote
  pair-accumulators every 16-32 k-steps) bounds the error and keeps most of the issue win.
  Different bits → full gate (F-230 law).
- **Candidate 1 (mma) is the bigger ceiling because it leaves the CUDA-core pipes entirely**: A =
  W rows (bf16, native — no conversion anywhere), B = [xn_s0..xn_s3] stacked with T tokens → the N
  axis = HC×T ≤ 32; at T=1 the n8 tile is half-used (4 of 8 columns) → ~256 tensor MACs/clk/SM
  effective vs 64 today = 4x, rising to ~8x at T≥2. The 5090 tensor ceiling: 209.5 TFLOPS dense
  bf16/f16 with f32 accumulate [S12] ≈ 512 MACs/clk/SM — far above anything the CUDA-core pipes
  can express on this shape. Accumulation stays f32 (mma `.f32`), so numerics land closer to today
  than candidate 2; the only bit change is summation order (mma's fixed hardware order vs the
  lane-strided warp_sum) → still different-bits, full gate, but the same gate class the GDN_CHUNKED
  arm already passed through (F-230 tax discipline).
- **Sequencing**: the queued T15A/T15C rungs (UPM_COLS 16→8, 82-block staged down 4-warp,
  norm_fast flip, TSUM/multi-acc/HC_Q8) attack occupancy and ILP of the SAME line. Candidates 1-2
  must be benched ON TOP of them (the rungs move the baseline; a 30→21 us line benched at 30 over
  prices the mma win). The staged kernel's cp.async double-buffer (:855-925) is the substrate both
  candidates reuse; candidate 1 additionally wants ldmatrix staging of W tiles (pattern already in
  native_qsa_score.cu:42-47).
- **smem/pipeline note for the chain itself**: with the staged kernel already double-buffering the
  xn tiles via cp.async groups (:884-913), the remaining serial structure is the accumulator chain
  and the final warp_sum butterfly. Candidate 2 halves the chain (160 hfma2 links); candidate 1
  replaces chain+warp_sum with per-warp mma accumulation (no butterfly). The queued "multi-acc"
  rung is the bitwise-ish intermediate; both candidates subsume it.

### B. QSA decode attention via mma (candidate 3)

Today: `attn_chunk_kernel` — grid (n_chunks, n_head_kv=2), 256 threads; each warp scores cells
warp+8k with each lane holding 8 of 256 dims (qsa_decode_attn.cu:202), dequantizing K/V per
`load8_{f16,q8,q4}` (:44-70) into floats, FMA-reducing; PV folds one cell's 12 weights into 12
accumulators (:494-507). No mma on the decode path — mma.sync m16n8k8/k16 f32-accum exists
in-tree and runs on our sm_120 build in qsa_prompt_attn.cu:49-66, including the exact int8→fp16
mantissa encoding ("1024 + (c+128) built in the mantissa, minus 1152", :67) for the IMMA-class
opt-in.
- The decode score shape IS a tensor shape: per KV head, cells×256 K against 12 q-heads → m16n8k16
  with M = cells (width-growing context = MORE utilization, the opposite of the usual decode
  problem), N = 12 (pad 16), K = 256 (16 k-steps). PV: 12 weights × V dims — same engine.
- Existence proofs that this pattern ships on consumer Blackwell: TRT-LLM open-sourced its
  decoding-phase XQA kernels (discussion #3987, PR #3762) [S30], and FlashInfer ships "the XQA
  decode kernel — used on Blackwell RTX and DGX Spark (SM120/121)" with speculative-decoding
  ragged-Q and sliding-window support [S25]. FlashInfer 0.7.0 also added "experimental fused GDN
  decode steps on SM120 that combine projection, convolution, gating, and recurrent state updates"
  (#4481/#4708) — external validation that OUR other decode line (GDN step) is fusible on this
  exact arch, worth a look when the F-201 R27 ILP rewrite is scheduled (queued; not re-priced here).
- Exactness: f16 KV pools are element-exact; the int8 pools ride the in-tree exact-fp16 encoding
  (products ≤ ~1.33M stay inside f32's 2^24 exact-integer window per mma k-step; the same
  exact-intent class upstream accepted for prompt attention). The gate: f16 path = M3b canary +
  acceptance monitor (order change only); int8 path = full gate.
- Width-growing context: the conc ceiling line (4.40 ms/window) grows with cells; the mma version's
  per-cell cost FALLS with utilization — this is the only candidate whose relative value grows with
  context depth (262k switch = T16 #6).

### C. CLC persistent layer-group kernel (candidate 4)

The gamble is banked (T7.A G1 / T8.A "MPK-lite"); what is NEW since:
- **CLC is real and reachable on this card**: CUDA guide documents work-stealing via CLC at "compute
  capability 10.0+" [S6]; family targets sm_120f carry it [S14]; Colfax's deep dive shows the
  programming pattern (one scheduler warp/cluster, `try_cancel.async` → 16 B smem response via
  mbarrier, `query_cancel.is_canceled/get_first_ctaid`, "submitting another request after a failed
  one is UB") and measures dynamic-vs-static persistent on imbalanced grouped GEMM (CLC clearly
  better; balanced: mixed, L2-hit-rate effects) [S17]. CUTLASS ships depth-3 CLC pipelines; the
  CuTeDSL `cute_arch` API exposes `try_cancel` [S23].
- **Shipped-on-SM12x precedent**: FlashInfer 0.6.17's SM12x fused-MoE families add "cooperative
  persistent launches, a tensor-core decode path for small batches, and shape-stable route packing"
  [S25]; SonicMoE extends to SM120 with ~500 LoC [S21]; Cohere runs a megakernel in production
  (1.58x over vLLM) [S19]; the Hazy megakernel hits 78% of H100 bandwidth at bs1 (vs ~50% engine
  typical) [S18]; ForgeMegakernel (arXiv 2609.12379) generalizes the compiler approach [S22].
- **Why CLC specifically fits OUR constraints** (this is the delta vs the banked gamble):
  1. Preemption: CLC failures "can signal preemption by a higher-priority kernel, after which CLC
     can resume" [S17] — a hard cooperative pin cannot; our box already shows clock/throttle
     sensitivity (T8.B, cloudrift's 84% small-kernel variance [S15]).
  2. No SM-count introspection: our grid shapes are fixed at capture; CLC scheduling doesn't need
     the 170-SM constant in the graph.
  3. Graph compatibility: a persistent layer-group kernel COLLAPSES hundreds of nodes into one —
     the one-graph-per-window structure survives (the graph simply has fewer nodes); conditional
     bodies and device-graph-launch limits [S4] are not needed for this form.
- **What it must absorb** (the honest risk ledger): the doorbell/flag protocol becomes device-side
  (the FUSE_WAIT adjudication and doorbell-collapse lane are the queued preconditions — this
  candidate consumes their outcome, does not preempt them); the F-223 G2 corruption class demands
  per-slot buffers; and the F-191 lesson applies in reverse — PDL was a no-op because spacing is
  dependency serialization, and a persistent kernel is precisely the mechanism that removes the
  serialization, not another attempt to hide it.

### D. Conditional nodes (candidate 5)

New mechanics vs the T12-GPU one-liner [S4][S8][S44]:
- IF (with optional ELSE body), WHILE, and (12.8) SWITCH; the condition value is read and evaluated
  **on the device** when the node's dependencies are met, set by device code via
  `cudaGraphSetConditional`; optional per-launch default reset at graph start.
- Body graphs allow kernel / empty / memcpy (device+pinned-mapped only) / memset / child /
  conditional nodes; nesting allowed; **no device-graph launch or CDP inside bodies**; cooperative
  launches OK outside MPS.
- Fit: our wait spins are 1-thread kernels charged 0.4-0.8 ms/window (3h line, T13A table) plus the
  always-launched empty PCIe node (verify.cpp:1709). An IF around {wait + dependent copy} makes the
  satisfied-flag case pay ~zero, and unlike the closed FUSE_WAIT/host-plan arms, the evaluation
  happens without any host round-trip (the exact gap F-222 measured at 0.26 ms). The doorbell
  publisher kernel writes the handle value directly — no new sync.
- Costs: explicit graph construction surgery (capture of conditional nodes needs
  `cudaStreamBeginCaptureToGraph` semantics [S4]); per-T instantiations already exist for the
  window graphs, so the handle/default plumbing rides the existing exec_ cache; instantiate cost is
  boot-time (our graphs are captured once — F-91 law).

### E. PDL: closed, and now docs-confirmed closed (no action; recorded so nobody re-opens it)

The PDL guide's own semantics close the loop on F-191: `cudaGridDependencySynchronize` "will block
until all primary kernels ... have completed and flushed results to global memory" [S5] — PDL only
overlaps the secondary's PREAMBLE and launch latency. Our kernels already do exactly the sanctioned
preamble (weight preloads before pdl_wait, kPdlPrefetch, pdl.hpp:40-52), and F-191 measured the
attribute a clean no-op because our node spacing is dependency serialization, not launch latency.
The only unexplored PDL corner is `ProgrammaticEvent` with `triggerAtBlockStart=1` →
`cudaGraphKernelNodePortLaunchCompletion` (fires when blocks LAUNCH, not complete) [S5] — a
micro-overlap with no candidate line attached; not worth a boot. RE-TEST ONLY inside candidate 4's
persistent redesign, per F-191's own note.

### F. CUDA 13.x launch-latency: the honest inventory

- 13.0 [S10]: 256-bit vector types (`double4_32a` etc., "leveraging Blackwell's 256-bit loads/
  stores") — a real micro for our issue-bound streaming kernels (xn staging, KV load8 paths are
  128-bit today, qsa_decode_attn.cu:44-47); drops pre-Turing targets (irrelevant); green contexts
  Jetson-only (13.1 generalizes [S40][S41]).
- 12.8/12.9 (in our 13.4 toolchain) [S8][S14]: ELSE+SWITCH conditional nodes; `cuMemcpyBatchAsync`
  (candidate 7); family cubins `sm_120f` (compile-once for the family — relevant only if we ever
  ship binaries).
- 13.4 release notes [S9]: green contexts support line + FP4 matmul +5% on Blackwell ULTRA
  (sm_103-class — not our card).
- Nothing in 13.x reduces per-node graph replay cost beyond what F-191/F-205/F-206 already
  measured. The launch-latency frontier for THIS engine is candidates 4/5, not toolkit flags.

---

## 5. Contradictions and their resolution

1. **"bf16-native dot path" as the untried hc lane (T12-GPU surviving-lever note) vs the 13.4
   throughput table.** The table kills the naive 2x-rate expectation (f16x2 = 64/clk on CC 12.0,
   HALF the f32 pipe) [S2]. Resolution: the lane survives re-priced as candidate 2 — its win is
   issue-COUNT (24→4 ops per 8 MACs) and chain-length halving at the packed pipe's own 128-MAC/clk
   rate, i.e. ~2x over the int-capped kernel, not over FFMA. The T15C #3 (HC_Q8) bimodal expectation
   stays the cheap empirical arbitrator of byte-vs-ALU binding.
2. **"TMA for decode GEMV usually a loss" (T9.A trap) vs cloudrift's TMA win on a 5090.** Both true:
   cloudrift's win is a large FP32 GEMM where LSU-instruction elimination and 89 KB smem tiles
   matter [S15]; our decode shapes are occupancy/issue-bound tiny grids where transport is not the
   bottleneck (their own Reddit-level caveat: "helps on paper (LSU drops), but the transport isn't
   the bottleneck"). Verdict: TMA = substrate (candidate 6), not a standalone arm.
3. **Persistent kernels vs cooperative-launch hazards (T8.A do-not-bother "full megakernel").** The
   banked objection was "no published data-dependent-MoE-with-offload instantiation". Unchanged for
   the FULL megakernel; candidate 4 is scoped to a layer-group walk and gains CLC preemption
   semantics [S17] + two shipped SM12x persistent precedents [S21][S25] that did not exist when T8.A
   wrote. The gamble stays a gamble — the falsifier order (CLC proto → 2-layer bench) is unchanged.
4. **FlashInfer NVFP4-on-SM120 breakage reports [S37][S38][S39] vs FlashInfer 0.7.x SM120 parity
   claims [S25].** Time-resolved: Feb 2026 breakage, fixed by the 0.6.15→0.7.2 series; the residual
   lesson for us is the dependency floor — "a PTXAS or CuTe DSL assembler from CUDA 13.4 or newer"
   is required for the SM120 Frost-class kernels [S25] — our box's 13.4 toolchain exactly meets it.

## 6. Open questions

1. What fraction of the hc line is issue-mix (int-pipe cap) vs latency vs occupancy? One nsys/ncu
   pass on gr_down/gr_up with `--clock-control none` (T9.A profiling law) splits candidates 1/2's
   pricing precisely — and prices the queued rungs' residual.
2. Does the mma hc kernel hold bit-tolerance at the f32-accum order change (KL spot-probe), or does
   the gate force the full M3b+MMLU battery on candidate 1? (Expected: full battery; the probe just
   sets expectations.)
3. Are conditional-node bodies capture-compatible with our recorder, or does candidate 5 require
   explicit-graph construction for the whole window? One prototype answers it.
4. CLC on GB202 specifically: the CUDA guide statement is family-wide (CC 10.0+) [S6]; no published
   5090 CLC benchmark exists (searched; only B200 measurements [S17]). The 20-line proto is the
   falsifier.
5. Does `dp4a` sit on the halved video pipe on CC 12.0 (the `vabsdiff4` 64→32 hint [S2])? One SASS/
   microbench check before candidate 8 is priced beyond "2x-the-converts".

## 7. Implications

- The **hc line is the only kernel line with a >10% single-mechanism ceiling left** on M1, and both
  of its mechanisms (mma, packed-dot) are now precisely characterized against primary-source
  throughput data. They sequence AFTER the queued T15A/T15C rungs and share one bench harness —
  the same fused_gr bench the rungs need anyway.
- The **conc ceiling line (QSA decode attention) has a shipping external pattern** (XQA on
  SM120/121) and an in-tree mma engine to port; it is the only candidate whose value GROWS with the
  262k switch.
- **PDL is permanently closed** with docs-level confirmation; nobody should spend a boot on it
  again short of the persistent redesign.
- The **persistent/conditional-node pair** (candidates 4/5) is the structural frontier; both now
  have primary-source mechanics, CC statements, and shipped precedents that did not exist at T8/T9
  time. Both are falsifiable without a server boot.
- Toolchain: stay on CUDA 13.4 PTXAS for any CuTe-DSL/CLC prototype (the FlashInfer floor) [S25];
  compile experiments under `-arch=sm_120f` where family features (CLC) are used [S14].

---

## SOURCES (all fetched or search-verified this session, 2026-10-17)

Format: URL — one-line relevance — verdict for sm_120 consumer decode graphs.

1. https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html — NVIDIA Blackwell Tuning Guide:
   CC 12.0 = 48 warps/SM, 32 blocks/SM, 128 KB smem/SM, 99 KB/block; clusters portable max 8; DSM
   guidance — APPLIES (the sm_120 occupancy/smem envelope; note the guide is otherwise B200-centric).
2. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html — CUDA 13.4 Best Practices
   Guide, Table 5 "Throughput of Native Arithmetic Instructions": CC 12.0/12.1 `add.f16x2`=64 (vs
   `add.f32`=128), shifts/AND/cvt=64, `vabsdiff4`=32, bf16 footnote "64 for __nv_bfloat16" at 8.0;
   table downloaded and parsed cell-by-cell this session — APPLIES (the decisive CUDA-core rate
   facts for every compute candidate).
3. https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html — PTX ISA 8.7:
   "Adds support for sm_120 target architecture"; `.multicast::cluster` optimized for
   sm_90a/sm_100a/sm_101a — APPLIES (sm_120 baseline; multicast not our play).
4. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html — CUDA
   Graphs chapter: conditional nodes IF/ELSE/WHILE/SWITCH, on-device condition evaluation,
   `cudaGraphSetConditional`, body-graph allowed node types and the no-device-graph-launch-inside
   rule, device launch limits (120 fire-and-forget, 255 pending tail), PDL edge data
   (`cudaGraphDependencyTypeProgrammatic`, ports) — APPLIES (candidate 5's mechanics).
5. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
   — PDL chapter: CC 9.0+, trigger/wait semantics, "block until all primary kernels ... completed
   and flushed", graphs ports (Programmatic vs LaunchCompletion) — APPLIES but confirms closure
   (F-191 no-op is semantics-consistent).
6. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html
   — Work Stealing with CLC: "introduced in the NVIDIA Blackwell GPU architecture (compute
   capability 10.0)", request/decode instructions, UB rules, comparison table (overhead +
   preemption + load balancing) — APPLIES (candidate 4's primitive).
7. https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GREEN__CONTEXTS.html —
   Green contexts driver API (cuGreenCtxCreate, SM-granularity resource descriptions) — PARTIAL
   (driver-API only here; queued T9#3 unchanged).
8. https://docs.nvidia.com/cuda/archive/12.8.2/cuda-features-archive/index.html — CUDA 12.8
   features: "ELSE graph support", "SWITCH node support", `cuMemcpyBatchAsync`/`cuMemcpyBatch3DAsync`,
   compiler support for SM_100/101/120 — APPLIES (candidates 5, 7).
9. https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html — CUDA Toolkit 13.4 release
   notes: green contexts support; FP4 matmul +5% geomean on Blackwell ULTRA — PARTIAL (the FP4 item
   is sm_103-class, not sm_120).
10. https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0 — CUDA 13.0
    "What's New": 256-bit vector types on Blackwell, green contexts Jetson, dropped pre-Turing
    offline compilation — PARTIAL (256-bit loads = the only micro relevant to us).
11. https://forums.developer.nvidia.com/t/does-b200-5090-support-tcgen05-ld-red/363672 — NVIDIA
    forum: tcgen05 limited to sm_100a/sm_101a(sm_110a) families; sm_120 NOT supported — DOESN'T
    (kills tcgen05-class designs on our card; confirms T7.A).
12. https://forums.developer.nvidia.com/t/rtx-5090-specs-fp16-tensor-tflops-is-ambiguous/351063 —
    NVIDIA forum: 5090 FP16/BF16 tensor 209.5 dense / 419 sparse with FP32 accumulate (whitepaper
    figures; the whitepaper PDF itself is not machine-readable in this session) — APPLIES (tensor
    ceiling used in candidate 1's arithmetic).
13. https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543 — NVIDIA forum:
    "FP32 accumulate means 16-bit operands multiplied into 32-bit results" (2x figure
    clarification) — APPLIES (same).
14. https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features
    — CUDA 12.9 family-specific targets (sm_100f/sm_120f aliases and family features incl. CLC-
    class instructions) — APPLIES (family-cubin guidance; CLC target).
15. https://www.cloudrift.ai/blog/beating-cublas-on-rtx-5090 — practitioner: TMA
    (`cp.async.bulk.tensor.2d` + mbarrier double-buffer) driven on RTX 5090 beating cuBLAS batched
    SGEMM 1.4-1.7x; sm_120 cuBLAS dispatch tables missing; 84% per-call clock variance on ~us
    kernels; dispatch_stall 44% — APPLIES (TMA real; variance law; NVIDIA-side "tables zero"
    parallel to our arch_defaults gap).
16. https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x
    — Colfax SM12x tutorial: "SM12x does not use tcgen05 or TMEM; instead, like with SM8x it uses
    warp-level mma.sync"; TMA available; setmaxnreg, CLC, PDL supported; kind::mxf4nvf4 m16n8k64;
    ~60% of peak observed; CTA tiles ≤128 — APPLIES (the canonical sm_120 kernel-feature map).
17. https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus
    — Colfax CLC deep dive: try_cancel/query_cancel pattern, mbarrier 16 B response, UB rules,
    dynamic beats static persistent on imbalanced grouped GEMM, preemption via CLC failure/resume,
    depth-3 pipelines — PARTIAL (measured on B200 only; mechanism transfers, per open question 4).
18. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles — Hazy megakernel: 78% of H100
    memory bandwidth at bs1 decode, 1.5x over existing systems — PARTIAL (H100; the persistent-
    step design reference).
19. https://cohere.com/blog/megakernels — Cohere North Mini Code megakernel in production, 1.58x
    over vLLM — PARTIAL (production precedent for persistent-step serving).
20. https://www.turintech.ai/blog/inside-llama-cpp-pr-26802 — llama.cpp PR 26802 on RTX 5090
    (CUDA 13.3): narrowing an over-broad graph-capture guard restored capture for bf16 MoE,
    1.13x at 16-way batch — PARTIAL (we already capture everything; cites as the 5090
    "capture-breadth" datapoint and the automated-search method).
21. https://tridao.me/blog/2026/sonicmoe-blackwell — SonicMoE: CLC scheduler ~10% over DeepGEMM
    sm100, tcgen05/TMEM/2-CTA design, "We extend to SM120 (Blackwell GeForce GPUs such as 5090) by
    simply adding a base GEMM class with ~500 LoC changes" — PARTIAL (training-shaped; SM120 port
    existence proof).
22. https://arxiv.org/html/2609.12379v1 — ForgeMegakernel: general megakernel framework; cites
    hand-written megakernels at 74-78% of H100 bandwidth — PARTIAL (framework context for
    candidate 4).
23. https://docs.nvidia.com/cutlass/4.7.0/media/docs/pythonDSL/cute_dsl_api/cute_arch.html —
    CUTLASS CuTeDSL `cute_arch` API: `try_cancel` ("atomically requests cancelling the launch of a
    cluster not yet running") — APPLIES (the highest-level API for the CLC proto).
24. https://arxiv.org/html/2507.10789v2 — "Dissecting the NVIDIA Blackwell Architecture" (GB203 =
    RTX 5080, measured): "the wgmma instruction isn't supported in GB203 and the tcgen05
    instruction hasn't been implemented yet for GB203"; "tcgen05 is yet to be supported for the
    arch. sm_120a"; mma present (QMMA SASS for fp8/fp4); smem ~99 KB measured; L2 latency ~358
    cycles; unified INT32/FP32 — APPLIES (the measured consumer-Blackwell uarch reference).
25. https://flashinfer.ai/releases — FlashInfer release history: XQA decode kernel on "Blackwell
    RTX and DGX Spark (SM120/121)"; "#4481/#4708 experimental fused GDN decode steps on SM120
    combining projection, convolution, gating, recurrent state updates"; 0.6.17 SM12x "cooperative
    persistent launches, a tensor-core decode path for small batches"; SM120 MegaMoE (#4387); W4A16
    1.36x vs Marlin at one token (SM121, Qwen3.8-Flash-Next); CUDA 13.4 PTXAS floor for Frost-class
    SM120 kernels — APPLIES (the densest sm_120 kernel-existence source; candidates 3/4).
26. https://github.com/flashinfer-ai/flashinfer/issues/3374 — sparse-MLA paged attention for SM120
    (RTX PRO 6000), decode/prefill dispatch by num_tokens — PARTIAL (MLA-shaped, not our QSA;
    SM120 attention momentum evidence).
27. https://github.com/Dao-AILab/flash-attention/issues/1763 — flash-attention: missing SM120
    kernels crash on RTX 5090 — DOESN'T (FA lineage does not cover sm_120 decode).
28. https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739 — "FA4 cannot run on RTX
    5090 (SM120)" — DOESN'T (FA4 = tcgen05-class targets only).
29. https://www.spheron.network/blog/flashattention-4-blackwell-gpu-cloud-guide — FA4 targets
    datacenter Blackwell sm_100/sm_103; consumer cc 12.0 unsupported — DOESN'T (same).
30. https://github.com/NVIDIA/TensorRT-LLM/discussions/3987 — TRT-LLM open-sourced decoding-phase
    XQA kernels (PR #3762) + fmha_v2 — APPLIES (candidate 3's reference codebase).
31. https://github.com/NVIDIA/TensorRT-LLM/issues/6158 — XQA = the speculative-decoding attention
    kernel family; Blackwell config gaps in TRTLLM-GEN — PARTIAL (XQA relevance + Blackwell
    maturity caveats).
32. https://github.com/NVIDIA/cutlass/issues/2186 — CUTLASS SM120 GEMM support request (3.8 shipped
    only to sm_100a) — APPLIES (sm120 support timeline).
33. https://docs.nvidia.com/cutlass/4.7.0/media/docs/cpp/blackwell_functionality.html — CUTLASS
    Blackwell functionality: tcgen05.mma for SM100 — PARTIAL (what sm_120 does NOT get).
34. https://gau-nernst.github.io/tcgen05 — tcgen05 explainer: SM120 uses the traditional mma.sync
    path — APPLIES (independent confirmation of [S16][S24]).
35. https://github.com/ggml-org/llama.cpp/issues/15479 — llama.cpp PDL tracking issue — PARTIAL
    (engine-adjacent PDL evidence; our F-191 verdict stands and supersedes for our stack).
36. https://ashvardanian.com/posts/longest-ptx-instruction — anatomy of the CLC instruction
    (clusterlaunchcontrol.try_cancel...mbarrier::complete_tx::bytes) — APPLIES (instruction-level
    reference for the CLC proto).
37. https://github.com/flashinfer-ai/flashinfer/issues/2577 — NVFP4 mm_fp4 GEMM broken on SM120
    (Feb 2026); cutlass/cudnn backends worked — PARTIAL (time-resolved; see contradiction 4).
38. https://github.com/vllm-project/vllm/issues/36835 — CUTLASS rejecting sm_120f binaries on
    sm_121 (tooling gap around family targets) — PARTIAL (family-cubin caveat).
39. https://github.com/sgl-project/sglang/issues/6160 + https://github.com/sgl-project/sglang-omni/issues/1196
    — SGLang RTX 5090 flashinfer/pytorch version friction; CUDA-graph capture hitting Hopper-only
    kernels ("no kernel image") — PARTIAL (ecosystem maturity context; the second URL documents the
    capture-vs-kernel-availability failure class our all-in-tree kernel strategy avoids).
40. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html —
    Green contexts programming-guide chapter (SM partitioning semantics) — PARTIAL (queued item,
    unchanged).
41. https://www.phoronix.com/news/NVIDIA-CUDA-13.1 — CUDA 13.1 lands green-context runtime API +
    CUDA Tile — PARTIAL (queued item, unchanged).
42. https://learn-kernels.com/chapters/hardware/nvidia-blackwell — persistent work-stealing via
    try_cancel, "every coordinate guaranteed exactly once" — APPLIES (secondary CLC reference).
43. https://www.paulwillchan.com/articles/outperforming-cublas-b200 — practical CLC usage in a GEMM
    that beats cuBLAS on B200 — PARTIAL (datacenter-only measurements).
44. https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes —
    NVIDIA's conditional-nodes announcement (IF 12.3 / WHILE 12.4; devices of compute capability
    9.0+, conditions read at execution time) — APPLIES (candidate 5's CC eligibility: sm_120 =
    CC 12.0).

Source counts by category: NVIDIA official docs/specs 1-10, 23, 33, 40 (13); NVIDIA forums/blog
11-14, 44 (5); practitioner/engineering blogs 15-21, 29, 36, 42-43 (11); code/issues 25-28,
30-32, 34-35, 37-39 (12); arXiv 22, 24 (2). Total distinct sources: 43 (44 numbered entries).

## Methods note (unproductive searches, not repeated)

- getpoppy.app "Writing PTX For Consumer Blackwell" (July 2026): DNS resolution failed from this
  box (two fetch attempts) — only the search snippet (cp.async.cg register-bypass doubling) was
  seen; NOT cited as evidence.
- medium.com "Beating vLLM and Llama.cpp on Gemma4-12B": HTTP 403 — snippet-only ("cp.async.bulk
  .tensor requires sm_90+; sm_120 has base TMA but not multicast"); consistent with [S15][S16][S3]
  which are cited instead.
- The RTX Blackwell whitepaper PDF (images.nvidia.com) is application/pdf — WebFetch unsupported;
  its tensor-rate figures are cited via the NVIDIA forum clarification threads [S12][S13] and
  marked as such.
- The CUDA 13.4 Best Practices/Programming Guide pages truncate in WebFetch; solved by curl +
  HTML table parsing (Table 5 extracted cell-exact).
- Search for the llama.cpp PDL PR number returned noise (university homepage); the tracking issue
  #15479 is cited instead; our own pdl.hpp/F-191 remain the controlling local record.
- No source found anywhere publishing per-kernel us/token decode numbers for the 5090 (confirms
  T7.A's note — our nsys timeline remains the only such source).
