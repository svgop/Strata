# Structural & Slowness Forensic Audit — 2026-11-20 (F-419)

Read-only audit across the whole repo. Method: four parallel lane dissections (prefill, decode,
orchestration/boot, external research) plus cross-cutting probes (sync primitives, getenv surface,
god-module density, include graph, backend parity coverage, tracked binaries). Every line number is
a code-read anchor; **no measurement was taken this pass** — all classifications are static analysis.
Two invariants govern everything that follows: (1) exact behavioral parity unless authorized;
(2) no lane regression (prefill TTFT, decode ITL, t/s at or above baseline).

---

## 1. Severity-ranked master suspect table

| # | Severity | Finding | Anchor (file:line) | Bound class | Blast radius |
|---|----------|---------|--------------------|-------------|--------------|
| A1 | Critical | `main()` is a ~16,280-line god-function (2459→18739) with 116 includes (one duplicated) and 68 top-level defs | `src/program/generate.cpp:2459` | structure | every boot, every lane |
| A2 | Critical | `Prefill::run_impl` is a 2473-line god-function (14 lambdas, 2 threads, 1 async in one scope) | `src/prefill/prefill.cpp:2562-5034` | structure | whole prefill lane |
| A3 | High | `expert_pool_dispatch_multi` is a ~570-line god-function (plan bookkeeping + PCIe share + cold-overlap + fetch-admit + pool invoke inlined) | `src/core/expert_source.cpp:4643-5210` | structure/latency | every decode window |
| A4 | High | `post(l,grp)` lambda is ~190 lines with a ~30-branch env-flag matrix, hand-written node order per combination | `src/core/verify.cpp:2195-2381` | structure/latency | every decode window, per layer |
| A5 | High | `Service._run` is ~534 lines mixing budget logic, loop-recovery, non-finite retry, stop-string cutting | `serve/server.py:5571-6105` | structure | serving layer |
| B1 | High | Serial `act_quant_q8_1` between the two parallel pool phases (gate/up → down) stalls the pool behind the last expert's quant — the F-414 "CPU-expert term" | `src/kernels/cpu/pool.cpp:1050, 1090-1092, 1119-1122` | latency-bound | every window with ≥1 CPU miss |
| B2 | High | Per-layer host↔GPU doorbell round trip (48 layers/window of serial `_mm_pause` spin + `wait_flag_ge`) | `src/core/verify.cpp:3154-3244` | latency-bound | all layers, all windows |
| B3 | High | Per-MoE-layer host round-trip: `cudaStreamSynchronize` + host grouping + H2D table re-upload on the default MMQ path | `src/prefill/prefill.cpp:3735→3909-3914` | latency-bound + PCIe | every MoE layer, default prefill |
| B4 | High | Quadratic indexer block-scoring O(T·pos/4) — the only genuinely quadratic term; in sparse-attention *selection*, not attention | `src/kernels/cuda/qsa_select.cu:201` (also :34/:355/:518) | compute-bound | every QSA layer, long prompts |
| B5 | Med | HBM weight streaming with reuse ≤ verify-window width (1-8 tokens); router is a 2.6 MB BF16 pass/layer with reuse 1 | `src/kernels/cuda/iq_kernels.cu:1820-1865`, `src/core/layer.cpp:406-418` | memory-bandwidth-bound (HBM) | all layers; worst at `--spec 1` |
| B6 | Med | Host-side activation quant serial loop (`act_quant_q8_1(H=2560)` per token, single-threaded) before the pool | `src/core/expert_source.cpp:5014-5024` | latency-bound | every window with CPU miss |
| B7 | Med | O(n²) plan bookkeeping live by default (the F-225 fast `plan_onepass` was reverted for −6.9% M1) | `src/core/expert_source.cpp:4770-4779, 4820-4826` | latency/compute | per layer, before flagA |
| B8 | Med | Un-fused silu/mix elementwise epilogues when `gr_upmix` fusion is refused (two extra full passes per half) | `src/prefill/prefill.cpp:3168, 3198-3204` | memory-bandwidth-bound | layers where fusion declines |
| B9 | Med | Per-window blocking sync pair (stream sync + copy-stream sync) at window tail; poll measured ~3% worse, so blocking is a deliberate local optimum | `src/core/verify.cpp:3286, 3298` | latency-bound | every window |
| B10 | Med | Drafter per-step serial launch + `_mm_pause` spin + full sync (`STRATA_DRAFT_UNGATED` off) | `src/core/mtp.cpp:1744-1772, 1842-1864` | latency-bound | speculative decode |
| C1 | High | Time-seeded sampling RNG: unseeded sampled requests seed Philox from `steady_clock::now()` → nondeterministic output per request | `src/program/generate.cpp:15610-15611` | determinism | every unseeded sampled request; threatens M3b canary |
| C2 | High | Per-token `json.dumps()` + blocking `wfile.flush()` SSE, three copy-pasted loops | `serve/server.py:7676-7677, 7909-7911, 7774-7790` | latency-bound | every streamed token |
| C3 | High | Per-token global `status_lock` grab in `_note()` | `serve/server.py:5522, 5810` | GIL/mutex-bound | concurrent requests (6-conc lane) |
| C4 | Med | Per-window `getenv("STRATA_LOOKUP_CHAIN_FIXED")` — environment walk inside the decode loop (same var also read `static` at :15898) | `src/program/generate.cpp:18326` | latency-bound | every window when lookup-chain active |
| C5 | Med | getenv truthiness 3-way inconsistency: `""`/`"0"`/unset resolve differently at `:184` vs `:336` vs `:368` | `src/program/generate.cpp:178-370` | determinism | any env-armed boot |
| C6 | Med | Full-prompt re-serialization `','.join(...)` per generation pass (O(262k) string build per thinking-wrap/retry) | `serve/server.py:2467, 2043, 2208` | latency-bound | scales with context envelope |
| C7 | Med | Two-thread ASCII-text protocol hop per token (UTF-8 decode + `int()` parse + 2 queue ops + 2 context switches) | `serve/server.py:1176, 1341-1382, 2490-2517` | latency-bound | every token |
| C8 | Med | `STRATA_RESIDENT_HEADROOM_GIB` read at 3 sites with different precedence reach | `src/program/generate.cpp:2898, 2909, 2946` | determinism | resident-mode boot |
| D1 | Med | Three duplicated decode loops (serial serve / batch-cohort / generate) with slight drift | `generate.cpp:16829-17015`, `12908+`, `14884+`, `18358+` | structure | drift risk on hottest path |
| D2 | Med | Duplicated token-read loop + three request handlers + error blocks in server.py | `serve/server.py:2451 vs 1644`, `7630/7882/7713` | structure | drift risk |
| D3 | Med | Backend drift: SYCL is a dpct mirror (56 `.dp.cpp`) with 40 parity files, Metal 16, vs CUDA 76 | `sycl/`, `metal/`, `src/` | parity | Intel/Metal targets |
| D4 | Med | 40 near-identical one-line env-helper lambdas at generate.cpp top, each with bespoke truthiness | `generate.cpp:178-370` | structure | root of C5 |
| D5 | Med | Unstable `std::sort` with tie-prone gain comparators → boot-to-boot perf (swap order) nondeterminism | `generate.cpp:8243, 10847, 16777, 18190` | determinism | perf only, not output |
| D6 | Low | 708 files with binary extensions tracked in git (dominated by `bench/results/*.out`) | `bench/results/` | hygiene | repo size / search |
| D7 | Low | Magic geometry constants duplicated without local `static_assert` (N=2560, FF=640, HC=4, …) | `kernels.cu:23-24`, `qsa.hpp:112`, `moe_fused_iq.cu:59` | correctness | silent corruption if geometry changes |

## 2. Roofline bottleneck classification (per lane)

| Lane | Dominant bound | Mechanism | Anchor evidence |
|------|----------------|-----------|-----------------|
| Decode (solo, M1) | **latency-bound** | CPU-expert serial quant (B1) + per-layer doorbell round trip (B2) dominate; GPU-reach idle 8.79 ms at 3.1 rows vs 0.51 ms at 4.9 rows (F-414) | `pool.cpp:1090`, `verify.cpp:3154` |
| Decode (concurrent) | **latency + HBM** | row-flat window (F-414): graph replay IS the critical path; weight streaming reuse grows with window width | `verify.cpp`, `iq_kernels.cu:1820` |
| Prefill (TTFT) | **latency + compute** | per-MoE-layer host round-trip (B3) + quadratic indexer scoring (B4); attention is already fused flash-style (linear in T, cap 2051 cells) | `prefill.cpp:3735`, `qsa_select.cu:201` |
| Serving host | **latency-bound (GIL/syscall)** | per-token SSE serialize+flush (C2) + status_lock (C3) + text-protocol hop (C7) | `server.py:7676, 5522, 2490` |

## 3. Boot-state lottery surface (synthesis)

Three distinct non-determinism classes, ranked by output impact:

1. **Output-changing** — time-seeded sampling RNG (C1); env-gated kernel/quant selection in the
   prefill lane (87 getenv sites; `STRATA_PF_FUSED`, `STRATA_WMMA_GEMM`, `STRATA_BF16_TC`, … each
   changes numerics); `STRATA_ROUTE_TAIL_SKIP` (deliberate owner default 7, output-changing, loud).
2. **Perf-changing** — unstable sort ties (D5); cached-static env read order.
3. **Benign** — `random_device` temp filenames (`conversation_file.cpp:94-104`).

The structural root is the ~40 bespoke env-helper lambdas (`generate.cpp:178-370`) each encoding a
different truthiness rule (`""` arms some, `"0"` arms others). Any env-armed boot is a lottery across
that matrix. The fix is one owned `env_flag()` helper with a single convention — the Mikado foundation.

## 4. External research dossier (citations)

| Mechanism | Best source (ID) | Verdict for this rig |
|-----------|------------------|----------------------|
| Async L2 KV prefetch (`cp.async.bulk.prefetch.L2`) | FlashInfer `4510df4`; AAAI'26 "Async KV Cache Prefetching" | Bank, don't queue — KV is not the binding term until experts are FP4'd |
| Warp-specialized persistent MoE decode | FA3 arXiv:2407.08608; FlashInfer PR #5558 (SM100/SM103 only) | Real shape; #5558 **fails closed on sm_120** — port gap |
| CUDA Graph dynamic shapes | vLLM PR #45084; TRT-LLM `cuda_graph_runner.py` | Already graphed; padding rule is the delta |
| NVFP4/FP8 disaggregated MoE | vLLM GB200 blog; FlashInfer PR #3738/#3858 | Highest lever; **sm_120 FP4 rate unproven** — measure first |
| Split-K + TMA MoE prefill | TRT-LLM PR #13833; CUTLASS Blackwell | tcgen05/TMEM is datacenter; sm_120 gap |
| CPU-offload MoE prefetch | vLLM RFC #33869; SPICE arXiv:2608.21240 | Overlaps in-tree `RouterLookahead` + ForesightSwap (F-418) |

**Convergence with banked intel:** the sweep re-prices rather than re-discovers — it independently
confirms T35/T42/F-252 (no fp32-accum unlock; fp8/fp4 sm_120 paths carry pack-change + SF-transform
hazards, vLLM #51884). The one new falsifier it names is the direct on-card FP4-vs-FP8-vs-BF16 GEMM
microbench (partially covered by `mma-accum-ratelock.cu`, F-252).

## 5. Safety-net gap analysis (Golden Master)

The tree already carries **76 CUDA parity files** (bit-exact kernel harnesses) — a strong foundation.
Gaps specific to the structural seams this audit targets:

| Gap | Seam | Needed test |
|-----|------|-------------|
| G1 | env-truthiness (C5/D4) | Golden Master pinning `""`/`"0"`/unset/garbage → on/off for all 40 helpers, before unification |
| G2 | sampling determinism (C1) | Characterization test: current time-seeded behavior; a fixed-seed arm proves the exact-match canary |
| G3 | decode window output | M3b exact-match already exists; extend to the 3 duplicated loops (D1) as a drift sentinel |
| G4 | boot geometry | `static_assert` the magic constants (D7) against `kCompiledGeometry` |

## 6. Mikado dependency tree (execution sequence)

Bottom-up, so the main branch stays green at every step. Safety nets (G*) are prerequisites, not
part of the dependency graph proper.

```
Layer 0 — leaf utilities (no deps)
  M1  unify the 40 env-helper lambdas → one owned env_flag(name, convention)   [foundation]
  M2  hoist per-window getenv to static (18326; also 18059)                    [zero-behavior]
  M3  named constants + static_assert for magic geometry (D7)                  [compile-time]
Layer 1 — safety nets (MUST land before any behavior-touching refactor)
  G1  env-truthiness Golden Master      G2  sampling-seed characterization
  G3  decode-loop drift sentinel        G4  geometry static_asserts
Layer 2 — god-module decomposition (depends on M1 + G*)
  D1  extract Options parsing from main()        D2  extract serve loop from main()
  D3  decompose Prefill::run_impl into stages    D4  decompose expert_pool_dispatch_multi
  D5  collapse post() env-matrix into owned, parity-pinned kernels
Layer 3 — consolidation (depends on Layer 2)
  C1  unify the 3 duplicated decode loops        C2  unify server.py request handlers
  C3  binary token framing (kill the ASCII-text hop)   [perf, gated]
Layer 4 — clutter purge (depends on Layer 3)
  P1  delete dead paths (native_small_mmvq ncols=1, dead default-off plan_onepass, #if GPU_METAL stubs)
  P2  triage 708 tracked binary-extension files
Layer 5 — verification gate
  V1  full test suite + type checker + linter + all 76 parity harnesses
  V2  interleaved same-hour battery ONLY for high-risk shifts (Layer 3+)
  V3  F-entry + commit + tracker append (law 14)
```

## 7. Non-findings (hunted, correctly absent — the audit's honesty register)

- **No naive quadratic attention, no un-fused softmax+mask+dropout.** `prompt_attn_kernel`
  (`qsa_prompt_attn.cu:116`) is fused block-sparse flash-style: online softmax, tensor-core MMA,
  cp.async prefetch, 2051-cell cap. Attention is O(T·2051), linear in T.
- **No `cudaDeviceSynchronize` in either hot loop.** Only one-time setup paths (verify.cpp:133,
  prefill drains). Synchronization is per-window `cudaStreamSynchronize` + mapped-flag spins.
- **No output-affecting `unordered_map` iteration.** Ordered `std::map` on the output paths.
- **No uninitialized-memory defect** in Options or the prefill buffers; T18's stale-`job_of` SIGSEGV
  is already fixed with a reset-at-call-start guard.
- **Deterministic sampling when greedy** (default) or explicitly seeded; Philox is counter-based,
  no time/thread seed on the greedy path.
- **No `.cpp`-including-`.cpp`** smell; only 3 TODO/FIXME markers; zero `#if 0` blocks.
- The `engine/strata.pre-*` files are deliberate binary-ladder rollback archives (law 10), not clutter.
