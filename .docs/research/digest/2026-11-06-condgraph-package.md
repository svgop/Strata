---
type: digest
id: condgraph-package
created: 2026-11-06
status: evidence
topic: The STRATA_WINDOW_GRAPH patch package — the conditional-node whole-window wrapper landed as a compiling default-off increment on the fork at 1465fa6b (salvaged from the crashed agent's scratch worktree), plus the cond_node falsifier microbench and the measured capture-legality probe that settles the brief's open conditional-body question on this rig
scope: implementation package record + compile proof + measured-ours probe results (RTX 5090, driver 615.71.09, CUDA 13.4 toolkit, sm_120). Sourced constraints carried from 2026-11-05-condgraph-zero-host.md are cited, not re-derived. No GPU benches were run (the GPU is serialized; the operator runs the bench legs). Artifact locations: scratch worktree /tmp/condgraph (git worktree of this repo at 1465fa6b), patch /tmp/condgraph/window-graph.patch, build /tmp/condgraph/build-pkg, probe /tmp/probe_cond/cond_probe.cu.
---

# Conditional-node window-graph patch package (2026-11-06)

## 1. Mission and state

Collapse the decode window's per-layer host round-trips into ONE host-launched replay per window, using conditional-node graph machinery where legal (brief: 2026-11-05-condgraph-zero-host.md). The predecessor crashed mid-build; this pass resumed the scratch at /tmp/condgraph and delivered the package.

Salvage survey verdict: **the predecessor's direction was correct and is kept 100%** — a conditional-WHILE wrapper in `Verifier::capture()` whose body is the straight-line layer sequence `record_window` already captures, plus the `win_cond_tail` device kernel and the `cond_node` microbench. Nothing needed a restart; the wrapper's scope is exactly the brief's minimal legal increment: capture scope (per-(T,layout) families already exist) + the conditional wrapper + in-graph waits only where the standing capture already has them. The failure was compile-class only — the target was never linked, and four host-compile errors were latent in the diff.

## 2. What landed (5 files, +395/-15)

Patch: `/tmp/condgraph/window-graph.patch` (495 lines; the bench file is intent-to-add, so it is inside the diff). Worktree: `/tmp/condgraph` (dirty against 1465fa6b; the only untracked file is the patch itself).

- `src/core/verify.cpp` — `g_window_graph` (env `STRATA_WINDOW_GRAPH=1`, opt-in, default OFF, read once); in `capture()`: when `wrap = g_window_graph && ar_on()`, the root graph is created empty, ONE `cudaGraphConditionalHandle` (initVal=1, `cudaGraphCondAssignDefault`) + ONE conditional node (`cudaGraphCondTypeWhile`, size=1) are added via `cudaGraphAddNode` (the 6-arg edge-data form the CUDA 13.4 headers declare), and the body graph from `p.conditional.phGraph_out[0]` is populated by `cudaStreamBeginCaptureToGraph(cs_, body, ...)` + the UNCHANGED `record_window(T, cs_, ...)`; `win_cond_tail` is launched as the body's last node; `cudaStreamEndCapture(cs_, nullptr)` closes the body; the ROOT instantiates into the existing per-(T,layout) family (`instantiate_evicting` — fresh `cudaGraphInstantiate`, LRU eviction on VRAM failure, so no exec-update path touches the conditional node, which the guide says cannot have parameters changed). With the switch off the code is the standing straight-line capture, branch-for-branch: default-OFF is bit-identical by construction.
- `src/kernels/cuda/verify_kernels.cu` — `win_cond_tail_kernel` (one thread, `cudaGraphSetConditional(handle, 0u)`, guarded `CUDART_VERSION >= 12040`, CUDA-only) + host launcher `win_cond_tail(unsigned long long handle, void* stream)`. The handle reaches the kernel as a captured argument; graph.hpp NOTE 2 (graphs re-read buffers, never arguments) is satisfied because handles are immortal and never change for the graph's lifetime.
- `include/strata/kernels/verify_kernels.hpp` — the `win_cond_tail` declaration (`unsigned long long` handle so the header stays free of CUDA includes; the static-assert wall at verify.hpp:579-581 is untouched).
- `CMakeLists.txt` — the `cond_node` microbench target, same pattern as `pinned_capture` (cudart only, `SKIP_RETURN_CODE 77` for the CUDA < 12.4 skip, `TIMEOUT 120`).
- `bench/micro/cond_node.cu` (new, 253 lines) — the brief's section 3.5 falsifier rung 1: (a) mapped-flag visibility round trip, host store -> resident spin kernel (`__nanosleep` poll) -> host read, gate p50 <= 3.0 us over 2000 rounds; (c-i) one pass per launch across 10k launches with zero host reset calls (re-arm proof); (c-ii) a 3-iteration live WHILE (terminates on the body's own counter write — the conditional is live, not silently stripped); (d) dispatch pricing, a 512-node straight-line body vs the same body wrapped one-pass, 2000 replays each, two passes reporting the cache-warm second.

Four compile-class fixes on top of the predecessor's diff (all found by the first real link attempt, none semantic):

1. `cudaGraphConditionalHandle` is `unsigned long long`, not a pointer: `h_win = nullptr` / three bench `h = nullptr` would not compile under nvcc 13.4 host checking -> `= 0`.
2. `win_cond_tail` was called unqualified inside `Verifier::capture()` (the `using namespace strata::kernels` lives in `record_window`, not there) -> qualified call.
3. The bench read `std::atomic_thread_fence` without `<atomic>` -> include added.
4. The three wrap blocks in `capture()` now carry `&& CUDART_VERSION >= 12040`, matching the .cu guard, so a pre-12.4 toolkit degrades to the standing straight-line capture instead of failing to compile.

## 3. Measured-ours probe: the brief's open legality question is settled for THIS rig

The brief's open risk was the body whitelist (kernel/empty/memcpy/memset/child/conditional only — no event nodes) against the standing capture's cross-stream fork: `sh_stream_on()` defaults ON on CUDA, so `record_window` records `ev_fork_`/`ev_join_` events around the shared expert every MoE layer, and only `cs_` is ever captured (`sh_cs_` never gets a BeginCapture). Probe: `/tmp/probe_cond/cond_probe.cu`, built with `/usr/local/cuda-13.4/bin/nvcc -O2 -arch=sm_120 -std=c++17`, run on RTX 5090 / driver 615.71.09 / sm_120:

- The engine's exact fork shape (event record on the capturing stream, wait from the non-capturing fork stream, kernels on the fork, record back, join wait) folds into a conditional-body `cudaStreamBeginCaptureToGraph` capture as **plain edges: the body holds exactly 3 kernel nodes, ZERO event record/wait nodes**. EndCapture succeeds; `cudaGraphInstantiate` accepts the wrapped graph.
- One pass per launch: counters read 64 = 32 threads x 2 kernels x 1 pass; the tail's `cudaGraphSetConditional(h, 0)` terminated the WHILE.
- Re-arm with no host call: an immediate second launch read 128 — the `cudaGraphCondAssignDefault` initializer resets the handle at every graph start, exactly as the brief's section 2.1 states.

Consequence: the default-ON shared-expert fork is LEGAL inside the conditional body on this driver/toolkit — the wrapped window body is the standing body's node set plus one tail kernel, and the standing STRATA_VERIFY_NODES listing (now pointed at the body when wrapped) can re-verify that node mix on any real capture. The other capture-time forks (`df_branch_` draft branches, `side_`/`side2_` STRATA_SIDE_MIXER) are opt-in and off by default; the same edge-folding mechanism covers them when armed, and the opt-in `STRATA_DF_BRANCH` #905 hazard note (window graphs vs NVML queries, Linux) is unchanged by this package.

## 4. Compile proof

Fresh build dir `/tmp/condgraph/build-pkg` (the scratch's existing `build/` had no CMakeCache — broken, left alone; the fork's `build-fresh/_deps/strata_llamacpp-src` was reused via `FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP` so no network clone was needed):

```
/home/svg/LLM/strata-custom-2/.venv/bin/cmake -S /tmp/condgraph -B /tmp/condgraph/build-pkg -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/home/svg/LLM/strata-custom-2/.venv/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=/home/svg/LLM/strata-custom-2/build-fresh/_deps/strata_llamacpp-src \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
/home/svg/LLM/strata-custom-2/.venv/bin/ninja -C /tmp/condgraph/build-pkg -j16 strata cond_node
```

Result: ninja exit 0; `strata` (91.6 MB, engine with the wrapper compiled in, switch off) and `cond_node` (53.8 KB) both link. Zero warnings from the touched files; the only build-log warnings are the pre-existing `native_mmvq.cu` "loop is not reachable" instantiation notes in untouched TUs. No engine run, no GPU bench (the GPU is serialized — the bench legs below are the operator's).

## 5. Boot/bench legs (operator runs; cheapest first)

1. **cond_node micro (the falsifier's rung 1):** `/tmp/condgraph/build-pkg/cond_node` on an idle GPU. Gates: (a) mapped-flag round trip p50 <= 3.0 us — if it fails, the in-graph-wait architecture dies at the microbench (brief 3.5a); (c-i) exactly 10000 body passes over 10000 launches; (c-ii) exactly 3 passes for one launch of the iterating WHILE; (d) prints the per-replay conditional delta — the price of one conditional dispatch + one tail kernel, expected us-scale-or-less against the carried ~60 ns in-graph wait and 5.9 us/graph submission figures. Exit 0 = every gate met; exit 1 names the failed gate; 77 = skip (toolkit < 12.4, impossible here).
2. **Window battery (rung b/c on the real path):** boot any banked quant with `STRATA_WINDOW_GRAPH=1` on an all-resident stage (the wrap arms only when `ar_on()` at capture time — an all-resident boot hits it immediately; a cold-cache boot captures straight-line and the switch stays inert for this session, which is the honest degrade). Read the boot banner: `STRATA_VERIFY_NODES=1` lists the BODY's node mix (3-kernel-class probe result generalizes: kernels + the tail, no event nodes). Then the standing window battery unchanged: 3 legs x {solo, pair, conc} vs the switch-off control for bit-identity (`ids`/logits A/B) and window wall-time; sub-kill telemetry is host CUDA-call count per window (expected ~1-2 with `STRATA_ADMIT_MERGE_FIRST` + the wrapper, same as the ar_on control — the wrapper adds dispatch cost, not host calls).

## 6. What remains to full one-doorbell

This package is the machinery increment, deliberately not the whole brief. Named remainder:

- **Window-level retry/mode arm:** state kernel B reading `m_plan_err_` to KEEP the handle at 1 for one in-graph re-pass (brief 3.1/3.3) — the wrapper's reason to exist; today's tail zeroes unconditionally, so behavior is one-pass-identical to the standing capture.
- **Draft-chain fold:** the serialized draft launch/finish rides outside the window graph today (`capture_commit` is its own graph); folding it as child/body nodes is the upside-tail item.
- **Exec_nr_ (doorbell-family) wrapping:** a stage that boots cold or flips `ar_off_` captures the doorbell variant straight-line (wrap requires `ar_on()` at capture). Wrapping the doorbell body too is legal by the same probe mechanism (its `wait_flag_ge`/fused wait+copy spins are kernel nodes) but is a separate increment.
- **The 3-leg bank** with the +4% across-lanes bar and the kill rule — untouched from the brief; cond_node (d)'s delta prices the overhead the bank must overcome.

## Sources

In-tree (this package): `/tmp/condgraph/window-graph.patch`; probe `/tmp/probe_cond/cond_probe.cu` + its console transcript (3 kernel nodes, 0 event nodes, 64/128 counters); build log `/tmp/condgraph/build-pkg/build.log` (ninja exit 0). Carried constraints: 2026-11-05-condgraph-zero-host.md sections 2.1 (whitelist, host-launched law, handle lifecycle, update law), 3.2 (API skeleton — the package follows it, with `cudaStreamCaptureModeThreadLocal` instead of Relaxed and the 6-arg `cudaGraphAddNode`/`cudaStreamBeginCaptureToGraph` forms the CUDA 13.4 headers declare), 3.5 (falsifier ladder — cond_node implements rung a/c mechanics). Measured on this machine: probe results above, driver 615.71.09, toolkit 13.4, sm_120.
