---
type: research-digest
id: 2026-11-08-calibration-engines
date: 2026-11-08
scope: How production LLM engines do per-device auto-calibration/auto-tuning — TensorRT-LLM, vLLM, SGLang, llama.cpp, MLC/TVM, ONNX Runtime, Triton, PyTorch TunableOp; search methodology (grid/adaptive/Bayesian); quant activation calibration; UX and runtime cost — and a recommended design for OUR `--calibrate`
subject: auto-calibration mechanics in LLM inference engines (kernel tactic search, memory-architecture sizing, knob search, activation scale fitting), cache design, and UX budget
authorization: operator-owned engine (strata-custom-2); research was READ-ONLY web research into public docs/source/issues. No engine state written by this pass.
---

# Calibration-engine digest — how the best engines auto-tune, and the design for OUR `--calibrate`

## Verdict (read this first)

1. **The gold standard is TensorRT-LLM's Python AutoTuner, and its full mechanism is now known at source level.** Per-op exhaustive grid over bucketed dynamic shapes, short-profile early exit (2 reps; escalate to 10 only if avg > 1 ms), CUDA-event or globaltimer timing, and a JSON disk cache keyed by `(custom_op, runner, runner.unique_id(), nearest-profile)` with `device_name` + `device_capability` + `lib_version` in the metadata, `fcntl` lock, tmp+fsync+rename atomic write, and rank-slot merge-not-overwrite. Copy this cache design almost verbatim (`tensorrt_llm/_torch/autotuner.py`).
2. **Device tuning and quant activation calibration are different artifacts and must never share a cache identity.** Device tuning cache is keyed by GPU/host fingerprint (TRT-LLM, TunableOp, ORT timing cache). Quant calibration is keyed by model + calibration dataset and travels with the model (llama.cpp imatrix GGUF, llm-compressor FP8 static scales, ModelOpt scales). One flag, two subcommands, two stores.
3. **No major engine auto-tunes host-memory architecture (hugepage pools, expert-cache splits, PCIe share).** vLLM/SGLang tune a single split knob (`gpu_memory_utilization` / `--mem-fraction-static`) with one memory-profile run per boot or manual 0.01-step increments; nobody sizes a hugetlb-backed memfd store from measurement. Our mechanic would be novel on exactly the axis we care about.
4. **The winning search method at single-device scale is: exhaustive-with-pruning + early-exit + persistent per-device cache — not Bayesian.** TRT-LLM (grid + short-profile), TunableOp (exhaustive capped at 30 ms/100 iters per op), Triton (prune_configs_by + top-k perf model). Bayesian/cost-model search (OtterTune GP, TVM XGBoost) pays off at fleet scale or when real measurements are extremely expensive — not on one box with cheap micro-benchmarks.
5. **Cache-miss behavior is a designed contract, not an error:** TRT-LLM falls back to tactic -1 with a loud warning; TunableOp falls back to "Default"; TVM's `MetaScheduleApplyDatabase` "Only replaces functions that have records". Our engine needs the same: a safe fallback config per knob plus a visible "untuned" marker.
6. **UX pattern that wins: implicit first-run tuning with hard time caps, then amortized cache hits.** TunableOp caps itself at 30 ms/100 iterations per op; TRT-LLM short-profiles; vLLM profiles once per boot in seconds and offers `--kv-cache-memory` to skip it on later boots. Budget: minutes for a full tune, sub-second cache-hit validation per boot.
7. **Trust boundaries every tuner enforces:** version/device validators that reject stale files (TunableOp "Validator" header lines; ORT "not portable" engines), cache-invalidating env vars hashed into Triton's disk key, and L2-cold measurement hygiene (TRT-LLM `use_cold_l2_cache`, TunableOp rotating buffer). Adopt all three.
8. **Quant calibration data volume is small and standardized:** 256-512 samples (llm-compressor examples use `ultrachat_200k[:512]`, AWQ 256); FP8 can skip calibration entirely via dynamic per-token scales; INT4-class schemes need it (GPTQ/AWQ/imatrix). Our quant arms can borrow imatrix's kurtosis/energy statistics to flag tensors needing higher precision.
9. **Recommended: rebuild `--calibrate` as a phased, capped, cache-backed mechanic** — phases: (0) device+model-set fingerprint, (1) host-memory micro-bench (hugepage pool, memfd, copy bandwidth), (2) interference/PCIe-share arms, (3) knob arms reusing the campaign battery, (4) commit with gate. Hard wall-clock caps per phase; atomic locked JSON; explicit `--verify`/`--dry-run`. Full sketch in §7.
10. **What it must never do:** never apply without the +4%/no-lane-worse gate; never cache without device+engine identity in the key; never tune on an unwarmed/aged host (preboot doctrine); never merge quant-calibration artifacts into the device cache; never auto-apply on boot without a verify pass.

Confidence scale used below (this pass is web research; no local measurement):

- **A** — primary source fetched and quoted today (official docs page, GitHub source file, or README read directly).
- **B** — primary source seen only via search-result snippets of the official page/paper; quote likely verbatim but not page-verified.
- **C** — secondary/community (blog, issue thread, third-party guide).
- **D** — inference from precedent; a hypothesis to validate, not a fact.

---

## 1. TensorRT-LLM: the Autotuner, `trtllm-bench`, and what is tunable vs fixed

### 1.1 The Python AutoTuner (torch backend) — mechanism at source level (grade A)

Source: [`tensorrt_llm/_torch/autotuner.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/autotuner.py), fetched raw 2026-11-08.

- **Trigger.** Tuning runs inside an `autotune(...)` context manager that sets `autotuner.is_tuning_mode`. Kernels call `choose_one()` at dispatch; on tuning-mode + cache-miss it profiles every `(runner, tactic)` pair. Search space: "a grid search, do cartesian product for all the dynamic axis" — dynamic dims are bucketed (`gen_tuning_buckets`, `map_to_tuning_buckets`), and `tune_max_num_tokens` saturates/clamps the token dimension so the biggest bucket caps real workload shapes.
- **Adaptive early exit (their trick to keep tuning cheap).** `profile_fewer_repeat = 2`: each candidate gets a 2-repeat short profile first; if its average exceeds `short_profile_threshold_ms = 1` (1 ms), that estimate is kept; only fast candidates get the full `repeat` run (default 10). Warmup default 2. `TLLM_AUTOTUNER_DISABLE_SHORT_PROFILE=1` forces full profiling.
- **Timing backends.** CUDA events (`start_evt.elapsed_time(end_evt)`) or a device global-timer kernel (ns, converted to ms). `TLLM_PROFILING_TIMER=globaltimer|cuda_event` overrides; unset auto-selects globaltimer under confidential computing (CUDA events unreliable there).
- **Measurement hygiene.** Optional CUDA-graph capture with a `delay_kernel` before replay to strip host launch overhead; `use_cold_l2_cache` clones input buffers sized against the device L2 cache so each candidate measures cold-cache.
- **Failure handling.** Failed profiles record time = inf; a single-candidate space skips timing entirely (`min_time = 0.0`).
- **Fallback contract.** Inference-time cache miss → warning "using the fallback tactic, due to cache miss" and runner index 0 / `tactic == -1`, which "has special meaning, means the fallback kernel which should be able to implement any shapes".
- **Cache.** `AutoTunerProfilingCache`, JSON on disk via `save_cache`/`load_cache`:
  - Key: `(custom_op, runner.__class__.__name__, str(runner.unique_id()), _find_nearest_profile(...))` — shape bucketed, constrained dims set to -1.
  - Metadata: `"device_name"` (`torch.cuda.get_device_name()`), `"device_capability"` (SM), `lib_version`, `creation_timestamp`. Device identity is in the file — a cache from another GPU does not silently win.
  - Concurrency: separate `.lock` file with `fcntl.lockf(LOCK_EX)`; atomic write via tmp + fsync + rename; rank-specific slots merged, never overwritten ("shared" key for shared entries).
  - `TuningConfig.exclude_from_cache` keeps entries memory-only.
- **Env vars.** `TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO`, `TLLM_PROFILING_TIMER`, `TLLM_AUTOTUNER_PRIME_CACHED_TACTICS` (default on: pre-compiles cached winners during a tuning pass; off skips that), `TLLM_AUTOTUNER_DISABLE_SHORT_PROFILE`.
- **Cache location.** `TLLM_AUTOTUNER_CACHE_PATH` env var; default resolves under the engine cache root to `autotuner/cache.json` (`tensorrt_llm/_bootstrap.py`, and the [LLM API env table](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html): `TLLM_AUTOTUNER_CACHE_PATH` → `autotuner/cache.json`) (grade A, via GitHub code search of the repo).
- **Cross-process discipline (their own docs demand it).** The layer-wise-benchmarks README: "The autotuner cache file should be generated in Step 1 and reused in Steps 2 and 3" — calibration output is a first-class artifact consumed by later runs ([examples/layer_wise_benchmarks/README.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/layer_wise_benchmarks/README.md)) (grade A).

### 1.2 Build-time tactic selection (TensorRT builder layer) (grade A)

TensorRT-LLM legacy builds compile engines via TensorRT, whose builder times tactics at engine build. The mechanism is exposed through the timing cache, which ONNX Runtime's TRT-EP docs describe precisely: it "will work across models as it simply stores kernel latencies for specific configurations and cubins (TRT 9.0+)", is reusable "across one GPU compute capability similar to an engine", and cut a Stable Diffusion UNet session build "from 384 seconds to 42 seconds" ([ORT TensorRT-ExecutionProvider docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)). So: tactic search happens twice in TRT-LLM's world — builder-time tactic timing (minutes, timing-cache amortized) and runtime AutoTuner lazy per-bucket tuning (JSON-cache amortized).

### 1.3 `trtllm-bench` (grade A)

Source: [`docs/source/commands/trtllm-bench.rst`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/commands/trtllm-bench.rst). "A comprehensive benchmarking tool for TensorRT LLM engines" with `throughput`, `latency`, and `prepare-dataset` subcommands; "Synthetic datasets with normal or uniform token distributions"; "Configurable random seeds for reproducible results". It is a measurement harness, not a tuner — the tuner is the runtime AutoTuner above.

### 1.4 Tunable vs fixed (grade A)

- **Tunable:** per-kernel tactics (GEMM, MoE, attention runners) auto-tuned + cached; MoE backend/comm/parallel-layout/CUDA-graph choices via `bench_moe --search full` (sweeps "backend, communication, parallel layout, and CUDA Graph" — [MoE perf guide](https://nvidia.github.io/TensorRT-LLM/developer-guide/moe-perf-guide.html)); max batch size / max num tokens; sharding; FP8 ([performance tuning guide](https://nvidia.github.io/TensorRT-LLM/legacy/performance/performance-tuning-guide/index.html): "While defaults are expected to provide solid performance, TensorRT-LLM has several configurable options that can improve performance for your particular workload").
- **Fixed/shipped:** the MoE backend decision tree itself — "The implementations differ in how their kernels are tuned, some picking parameters from the problem shape and some using a fixed configuration", and the recommended tree comes from "bench_moe sweeps on TensorRT-LLM v1.3.0rc26" with the caveat it "may not be optimal for a specific workload". CUDA graph on for decode, off for prefill is a fixed phase policy. Speculative-decoding depth: no per-device auto-tuner documented in this pass (grade D — absence verified only for the pages fetched).

---

## 2. vLLM: one automatic memory probe, no tactic tuner, a bench suite with a sweeper

### 2.1 Auto-config from a memory profile run (grade B, official API docs)

V1's worker implements `determine_available_memory()`, which "profiles the peak memory usage of the model to determine how much memory can be used for KV cache without OOMs" ([vLLM v1 gpu_worker API docs](https://docs.vllm.ai/en/v0.32.0/api/vllm/v1/worker/gpu_worker)). Mechanism: one dummy forward pass at boot measures peak activation memory; `available = total × gpu_memory_utilization − (weights + peak_activation)` becomes the pre-allocated KV cache. Known weakness: profiling noise and other processes' GPU memory polluting the measurement ([issue #12529](https://github.com/vllm-project/vllm/issues/12529)).

### 2.2 The calibration-reuse knob (grade A)

The optimization docs describe the exact "cache the profile" UX we want to copy: `--kv-cache-memory` makes vLLM log the value that reproduces the current allocation; passing it back "skips the memory-profiling measurement and the CUDA-graph memory estimation pass" — valid only on the same GPU with the same initial free memory; "conservative values cap throughput; optimistic ones OOM" ([vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html)).

### 2.3 What else is auto vs manual (grade A, same page)

- Auto: chunked prefill ("enabled by default whenever possible"), attention backend selection ("auto-selected by GPU arch/model/config"), CUDA-graph/compile under `-O2` default ("Fast optimization... PIECEWISE cudagraphs"; `-O3` "Currently equal to -O2"), NUMA binding ("`--numa-bind` auto-detects GPU-to-NUMA mapping"), compile cache under `VLLM_CACHE_ROOT` (default `~/.cache/vllm`), preemption mode RECOMPUTE.
- Manual: `gpu_memory_utilization`, `max_num_batched_tokens` (guidance: ~2048 for ITL, "> 8192" for throughput), TP/PP/EP/DP, `max_num_seqs`. No kernel-tactic autotuner exists in vLLM this pass; kernel choice comes from compiled libraries (FlashInfer/CUTLASS dispatch), not measurement.

### 2.4 `vllm bench` (grade A)

Seven subcommands: `serve`, `throughput`, `latency`, `forward`, `startup`, `mm-processor`, and `sweep` — "Benchmark for a parameter sweep" ([bench CLI docs](https://docs.vllm.ai/en/latest/cli/bench.html)). Sweep subcommands run the server "under multiple settings", explore "the latency-throughput tradeoff", and plot "Pareto frontier between tokens/s/user and tokens/s/GPU" ([bench sweep docs](https://docs.vllm.ai/en/latest/cli/bench/sweep.html)). This is the closest thing to our interleaved-arm battery in any mainstream engine — but it is an operator tool, not wired into engine startup.

### 2.5 FP8 calibration stance (grade B)

vLLM quantizes BF16/FP16 models to FP8 with "dynamic" per-token activation scales "without any calibration data"; static scales require a calibrated checkpoint (`activation_scheme="static"`) from AutoFP8/llm-compressor ([FP8 docs](https://docs.vllm.ai/en/v0.5.4/quantization/fp8.html)).

---

## 3. SGLang / llama.cpp / MLC-TVM / ONNX Runtime (plus Triton and TunableOp as kernel-level references)

### 3.1 SGLang: manual tuning guide, no auto-tuner (grade A)

The official [hyperparameter tuning page](https://docs.sglang.io/advanced_features/hyperparameter_tuning.html) is a manual recipe book, not a tuner:
- "Achieving a large batch size is the most important thing for attaining high throughput."
- `--mem-fraction-static`: "reserving 5-8 GB of memory for activations is typically sufficient", verify via the `available_gpu_mem` boot log; or "raise it in 0.01 steps until OOM appears for your workload".
- `--schedule-conservativeness`: drop to ~0.3 if KV under-used, raise to ~1.3 on frequent retractions.
- Health bands read from the decode log: "A healthy range for `#queue-req` is 100 - 2000", `token usage > 0.9`.
- Verdict on auto-tuning: "No built-in auto-tuner is described — defaults come from 'simple heuristics'".
- Tooling: `bench_one_batch` (now `sglang.benchmark.offline_throughput`) measures a fixed static batch at the ModelRunner level ([benchmark and profiling docs](https://docs.sglang.ai/references/benchmark_and_profiling.html), grade B).

### 3.2 llama.cpp: benchmark-only (`llama-bench`), zero autotune (grade A)

Source: [`tools/llama-bench/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md). Measures pp/tg/pg in t/s with std deviation, `-r 5` repetitions, optional `--no-warmup`; sweep syntax "Multiple values can be given for each parameter by separating them with ','" and ranges "first-last+step" or "first-last*mult", run "with all combinations of the specified options"; outputs `csv|json|jsonl|md|sql` — SQL inserts "suitable for importing into a SQLite database". Explicit finding: **no caching or autotuning mechanism exists** — the closest features are warmup runs and repetition averaging. Kernel selection is compile-time/heuristic (hand-tuned per arch), not measured per device. This is the gap our engine inherits from this lineage and should close.

### 3.3 MLC-LLM / TVM: the MetaSchedule heritage (grade A)

- Current MLC-LLM docs contain no autotuning page anymore (repo `docs/compilation/` holds only `compile_models.rst`, `configure_quantization.rst`, etc. — checked via GitHub contents API today). The `--opt=o0..o3` flag selects optimization tiers, not a tuning run ([compile models docs](https://llm.mlc.ai/docs/compilation/compile_models.html)). The autotune workflow moved into TVM itself.
- TVM MetaSchedule ([deep-dive tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)): "SearchStrategy (default: EvolutionarySearch): Explores the design space using an evolutionary algorithm guided by a cost model"; "CostModel (default: XGBModel): Predicts schedule performance using XGBoost, reducing the number of actual hardware measurements needed"; "Builder / Runner: Compile and execute candidates on real hardware".
- **Cache/reuse:** "Database (default: JSONDatabase): Persistently stores tuning records (schedule traces + measured run times)" in two newline-delimited JSON files under `work_dir` — `database_workload.json` ("One line per unique workload (structural hash + serialized IRModule)") and `database_tuning_record.json` ("workload index + schedule trace + measured run times"); "Records are appended incrementally". Budget allocation across kernels: "TaskScheduler (default: GradientBasedScheduler): Allocates tuning budget across multiple tasks based on their weights". Application: `MetaScheduleApplyDatabase` "Applies the best schedules from the tuning database. Only replaces functions that have records; the rest are left unchanged."
- **Keying caveat:** records key on workload structural hash, not explicitly on target/device in this doc — porting tuned records across GPUs is the user's responsibility (grade A text, device-keying gap graded as a design flaw to avoid).

### 3.4 ONNX Runtime: engine cache + timing cache (grade A)

From the [TensorRT-ExecutionProvider docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html):
- `trt_timing_cache_enable` / `trt_timing_cache_path`; the timing cache "simply stores kernel latencies for specific configurations and cubins", works "across models", is reusable "across one GPU compute capability", and shrank a UNet build "from 384 seconds to 42 seconds".
- `trt_engine_cache_enable` / `trt_engine_cache_path`: engines are "created for specific settings such as model path/name, precision, workspace, profiles etc, and specific GPUs and it's not portable"; cached profiles validate input shapes and "the engine is rebuilt" on mismatch; docs warn to "clean up any old engine and profile cache files" when ORT/TRT version or hardware changes.
- Failure mode documented: stale caches after version bump — explicit invalidation required, no auto-validation of TRT version inside old files.

### 3.5 Triton autotuner: the sharp edges to avoid (grade A)

Source: [`python/triton/runtime/autotuner.py`](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py).
- In-memory cache `self.cache: Dict[Tuple, Config]` keyed by the `key` arg values **plus dtype strings of every tensor arg** — "The key does not include device/GPU info." Multi-GPU processes share one dict; a config tuned on GPU 0 can be served to GPU 1.
- Disk cache is **opt-in** (`cache_results` knob, default off): without it, "tuning repeats every process launch". The disk key hashes `triton_key()`, backend hash, kernel source cache_key, `get_cache_invalidating_env_vars()`, and the config list; stored as `{fn.__name__}.autotune.json`; kernels with pre-hooks are excluded.
- Benchmarking via `do_bench` with `quantiles=(0.5, 0.2, 0.8)` (median + tail bands — cheap noise discipline); failing configs (OutOfResources etc.) return `float("inf")`.
- Pruning: `early_config_prune` ("It should return at least one config") and `perf_model` + `top_k` pre-ranking.

### 3.6 PyTorch TunableOp: the most complete calibration-file contract in the field (grade A)

Source: [`cuda.tunable` API docs](https://docs.pytorch.org/docs/2.15/cuda.tunable.html).
- What it tunes: GEMM backend selection (rocBLAS vs hipBLASLt on ROCm; cuBLASLt algo candidates on CUDA, default 8 candidates).
- Separation of concerns: `torch.cuda.tunable.enable()` is "the big on/off switch"; `tuning_enable()` gates the tuning phase — "Once a result exists for given inputs... no further tuning will take place even when the tuning setting is enabled."
- Results file: default `tunableop_results.csv` with the GPU device ordinal auto-inserted into the filename so multi-GPU runs do not clobber each other; written "automatically when the application terminates".
- **Validation header:** the file starts with "Validator" lines (PyTorch, ROCm, hipBLASLt, rocBLAS versions) — a version "mismatch causes the file to be rejected". This is the exact stale-artifact guard our cache needs.
- Row schema: op name with dtype/transpose (`GemmTunableOp_float_NT`), params (`nt_4096_4096_64`), solution name, avg time; solution "Default" = explicit fallback entry.
- Budget caps: `set_max_tuning_duration` / `set_max_tuning_iterations` — "the smaller of the two will be honored", defaults 30 ms / 100 iterations per op.
- Offline workflow: `record_untuned_enable()` collects encountered GEMMs into `tunableop_untuned0.csv` during production runs; a separate script tunes the file into `tunableop_results0.csv`. Multi-GPU helper `mgpu_tune_gemm_in_file`.
- Hygiene: rotating buffer sized to L2 for cold-cache measurement; optional numerical-check tolerances; wildcard fallback for missed signatures.

---

## 4. Calibration-search methodology: what beats a human, and what they measured

### 4.1 The families

| Family | Example | Search | Model of the world | Where it wins |
|---|---|---|---|---|
| Exhaustive + prune + early-exit + persistent cache | TRT-LLM AutoTuner, TunableOp, Triton, CUTLASS profiler | grid over bucketed shapes | none (pure measurement) | single device, cheap micro-benchmarks (grade A for all mechanisms above) |
| Evolutionary search + learned cost model | TVM MetaSchedule | evolutionary mutators guided by XGBoost | learned latency predictor cuts real measurements | many kernels × limited measurement budget (grade A) |
| GP/Bayesian config optimizer with cross-session reuse | OtterTune (VLDB'17) | Gaussian-process optimizer over knob space | global model from prior sessions + local model for this workload | knobs too expensive to grid; repeated tuning sessions to learn from (grade B) |
| Bandit ensembles of search techniques | OpenTuner (OSDI'14) | "ensembles of disparate search techniques simultaneously, techniques which perform well will receive larger testing budgets and techniques which perform poorly will be disabled" | none | unknown search-space geometry (grade A, [README](https://github.com/jansel/opentuner)) |
| Static profiles, no search | Linux `tuned` | none — "Daemon for monitoring and adaptive tuning of system devices" applying predefined profiles; profile recommendation uses "only static detection" | none | fleet consistency, not per-device optima (grade A, [tuned README](https://github.com/redhat-performance/tuned)) |

### 4.2 Better-than-human evidence

- **OtterTune** (Van Aken, Pavlo, Gordon, Zhang; [arXiv:1711.00006](https://arxiv.org/abs/1711.00006), [CMU PDF](https://db.cs.cmu.edu/papers/2017/p1009-van-aken.pdf)): "recommends configurations that are as good as or better than those generated by existing tuning tools or a human expert"; it "leverages machine learning informed by data gathered from previous tuning sessions", identifies the most impactful knobs, "map[s] unseen workloads to the most similar known workloads", and recommends via a Gaussian-process-based optimizer (grade B — abstract via search snippets of the primary paper).
- **TVM/MetaSchedule** lineage claims beats over vendor libraries in its tutorial series (grade C for magnitude claims; the mechanism is grade A).
- **TunableOp community evidence:** "PyTorch TunableOp is effectively mandatory for good performance on RDNA3" ([ROCm issue #5040](https://github.com/ROCm/legacy-rocm-build/issues/5040), grade C) — measured selection beats vendor defaults.
- **TensorRT timing cache** 384 s → 42 s build (grade A) — the payoff number for caching tactic latencies.

### 4.3 What they measured to get there

- Kernel-level: median of N timed reps after warmup, quantiles (0.5/0.2/0.8) for noise bands, cold-L2 buffer rotation, CUDA-event or globaltimer clocks, inf-on-failure, short-profile escalation (all grade A, §1-3).
- System-level: vLLM one memory-profile forward pass (peak activation), SGLang boot-log `available_gpu_mem` + queue/token-usage bands, `vllm bench sweep` Pareto frontiers tokens/s/user vs tokens/s/GPU (grade A/B).
- **Synthesis for our scale:** on one box with cheap micro-benchmarks, exhaustive-with-pruning + short-profile early-exit + persistent device-keyed cache is the proven production recipe. Bayesian methods earn their complexity only when (a) the space is combinatorially huge, (b) each evaluation is expensive, or (c) cross-session data reuse is real. We can still borrow OtterTune's *session-reuse prior*: seed arms from the previous calibration run's winners, re-verify, only search the deltas (grade D design judgment, built on grade A precedents).

---

## 5. "Calibration" sense 2: activation calibration for quantized kernels

### 5.1 The two senses are structurally different (grade A/D)

- **Device/kernel tuning** answers "which tactic/config is fastest here?" — artifact is keyed by device fingerprint, invalidated by driver/engine change, travels with the machine.
- **Activation calibration** answers "which scales does this model need for quantized kernels?" — artifact is keyed by model + calibration dataset, computed once offline, travels with the model file.
- Evidence: TRT-LLM autotuner cache carries `device_name`/`device_capability`; llama.cpp ships imatrix inside GGUF keyed to the model; TunableOp rejects on software-version mismatch while quant checkpoints ship pre-calibrated scales. Unify the *flag*, never the *store* (grade D conclusion on grade A evidence).

### 5.2 How the best engines run it

- **FP8 — often skip it.** vLLM: dynamic per-token activation scales computed at runtime, "without any calibration data" ([FP8 docs](https://docs.vllm.ai/en/v0.5.4/quantization/fp8.html), grade B); static scales only via pre-calibrated checkpoints (AutoFP8, `activation_scheme="static"`).
- **INT4/weighted schemes — small calibration set, offline, cached in artifact.** llm-compressor: "Quantization is applied by selecting an algorithm and calling the `oneshot` API" with recipes (`QuantizationModifier(scheme="FP8_BLOCK", ...)`); algorithms include "GPTQ" ("Weighted quantization with calibration"), "AWQ", "SmoothQuant", "AutoRound" ([README](https://github.com/vllm-project/llm-compressor), grade A). Example calibration volumes: `NUM_CALIBRATION_SAMPLES = 512` over `HuggingFaceH4/ultrachat_200k` (W4A16 + MoE examples), AWQ example 256 (grade A, repo code search). QADModifier "caches calibration inputs and the unquantized subgraph output as the teacher".
- **NVIDIA ModelOpt** (feeds TRT-LLM checkpoints): `mtq.quantize(model, cfg, forward_loop)` then `mtq.calibrate(model, forward_loop)` — calibration replays data through the model to fit scales; calibrators `max` (min-max) and `entropy` (KL) ([ModelOpt docs/API](https://docs.nvidia.com/modelopt/), grade B); scales export into the checkpoint quant config.
- **ONNX Runtime INT8:** MinMax, Entropy, Percentile calibrators with a calibration cache file (grade B, ORT quantization docs via search).
- **llama.cpp lineage (ours):** `llama-imatrix` — "Compute an importance matrix for a model and given text dataset... Can be used during quantization to enhance the quality of the quantized models" ([imatrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md), grade A). User supplies calibration text (example `wiki.train.raw`), matrix saved as GGUF and applied via `llama-quantize --imatrix`. Notable diagnostics: per-tensor kurtosis ("outlier features" needing higher precision), normalized entropy ("how well a prompt exercises the model"), energy ΣE[A²] (high-energy tensors' quant error "propagate[s] strongly"); `--in-file` merges existing imatrix runs; `--process-output` off by default ("better not to utilize the importance matrix when quantizing these tensors").
- **Per-device? No.** No engine conditions activation scales on the GPU — scales are dataset-driven. Device-dependent quant behavior (e.g. FP8 scale application paths) is a kernel-selection problem, i.e. sense-1 tuning (grade A/D).

---

## 6. UX: how calibration is presented and what it costs

| Engine | Presentation | First-run cost | Amortized cost | Cache hit validation |
|---|---|---|---|---|
| TensorRT-LLM | Implicit: first inference inside `autotune()` tunes bucket-by-bucket; explicit via shared `TLLM_AUTOTUNER_CACHE_PATH` in scripted flows | Seconds-to-minutes per bucket set, short-profile-capped | `autotuner/cache.json` load; near-zero | Keyed lookup; miss → fallback + warning (A) |
| vLLM | Implicit: memory profile run at every boot; explicit skip via `--kv-cache-memory` | Seconds (one forward pass) | None unless flag passed | Flag validity check: same GPU + same free memory (A) |
| SGLang / llama.cpp | None — manual bench tools (`bench_one_batch`, `llama-bench`) | operator's time | none | none (A) |
| MLC/TVM | Explicit separate tune step, budgeted by task weight | Hours for full nets (evolutionary + XGBoost), user-set budget | JSONL database apply | `MetaScheduleApplyDatabase` replaces only recorded functions (A) |
| ONNX Runtime (TRT EP) | Implicit with explicit knobs | Build 384 s → 42 s with timing cache; engine cache session ~9 s | timing + engine caches | Shape-profile validation; rebuild on mismatch (A) |
| TunableOp | Explicit env-gated tuning run (`PYTORCH_TUNABLEOP_TUNING=1`), then plain runs | Capped 30 ms / 100 iters per op | CSV load at startup | Validator header version check; reject on mismatch (A) |
| Linux tuned | Explicit profile apply + `tuned-adm verify` | Instant (static) | none | verify compares profile vs live settings (A) |

Cross-engine pattern (grade D synthesis): the best UX is **implicit-but-capped first run + explicit artifact + explicit reverify**. Nothing ships a "wizard"; everything ships a cache file with validators.

---

## 7. Recommended design sketch for OUR `--calibrate`

Targets: `/home/svg/LLM/strata-custom-2` engine (llama.cpp lineage, CUDA + CPU-expert-offload MoE, 2 MiB hugepage pool, hugetlb memfd store, GPU expert-cache splits, PCIe share fraction, batch/verify sizes, speculative depth, quant-sensitive arms).

### 7.1 Shape of the flag

```
--calibrate device   # sense 1: per-device tuning (default if no arg)
--calibrate quant    # sense 2: activation calibration (per-model artifact)
--calibrate all      # quant, then device (device arms exercise quant kernels)
--calibrate-status   # print cache identity, ages, per-knob winners + evidence
--calibrate-verify   # re-measure cached winners only; flag stale entries
--no-calibrate       # never auto-tune; load cache or fall back to defaults
--calibrate-caps <ms,#iters>   # hard budget override (TunableOp precedent)
```

Never auto-run quant inside device calibration by default; both may exist, separately invoked, separately cached.

### 7.2 Cache design (copy TRT-LLM + TunableOp, per §1.1/§3.6)

- One JSON store per sense: `~/.cache/strata/calibration/device/<fingerprint>.json` and per-model `models/<model>.calib.gguf|json`.
- Fingerprint (key + metadata, both — TRT-LLM precedent): GPU name + SM, VRAM, driver version, host CPU model + NUMA topology, hugepage availability, engine `lib_version`/build id, and a `validators` header (engine version, gguf-hash, driver) whose mismatch **rejects the file** (TunableOp precedent).
- Entry key: `(surface, knob_group, bucketed-shape-profile)`; shape buckets mirror our batch/verify/depth grids; token-dim saturated by `tune_max_num_tokens`-style clamp.
- Writes: tmp + fsync + rename under `fcntl` lock; append/merge per-run slots, never overwrite prior runs' evidence; `creation_timestamp` per entry.
- Miss contract: every knob group declares a safe fallback value; miss logs loudly ("using fallback, due to cache miss") and marks the boot `calibration=partial` in status output.
- Versioned invalidation: hash cache-invalidating env/engine defines into the file key (Triton precedent) instead of trusting users to delete files (ORT's documented failure mode).

### 7.3 Phases (each with a hard wall-clock cap; default full budget ≤ 15 min, incremental ≤ 2 min)

- **Phase 0 — fingerprint (seconds).** Hash device + model set; load cache; compute delta plan (which buckets lack winners). Cache-hit boot stops here (< 1 s target).
- **Phase 1 — host-memory micro-bench (minutes).** Measure: hugepage pool allocation/fault latency across candidate pool sizes (2 MiB granularity, sized to the model set's resident set); memfd-hugetlb store round-trip; host↔device copy bandwidth/latency for expert-cache split candidates; PCIe share fraction via copy-during-compute interference deltas. All solo measurements, cold-cache hygiene (rotate buffers, L2-sized; TRT-LLM/TunableOp precedent). Output: pool size + split fractions + PCIe share prior.
- **Phase 2 — engine arms (minutes, reuses the campaign battery).** Boot the engine per arm; run probe legs (solo/pair/concurrent generation, prefill TTFT); interleave A/B arms with kill lines. Arm space: batch/verify sizes, speculative depth, PCIe fraction, expert split — coordinate descent per knob group with bucket grid, **not** full cartesian (space too big for exhaustive; TRT-LLM's cartesian works because buckets are per-kernel, ours are system-level). Seed arms from previous calibration winners, re-verify first (OtterTune session-reuse prior, grade D).
- **Phase 3 — gate.** Keep a winner only on the campaign rule: clears **+4%** vs incumbent on its target metric **with no lane worse**; else keep incumbent. Re-run winner pair once more interleaved to reject noise (quantile check: median + 0.2/0.8 bands, Triton precedent).
- **Phase 4 — commit.** Atomic write of winners + per-arm evidence table (metric, reps, quantiles, kill-line status). Print `--calibrate-status` diff. Never mutate engine defaults files implicitly; the engine reads the cache unless `--no-calibrate`.

### 7.4 Quant sense (sense 2)

- `--calibrate quant`: run imatrix-style forward passes over a calibration corpus (512 samples, chat-style; llama.cpp `wiki.train.raw` precedent), emit importance-matrix artifact keyed to model hash; per-tensor stats (kurtosis, H-norm, energy) flag tensors for higher-precision arms; device-tuning arms then A/B those quant-sensitive kernel choices under sense-1 rules. Never cache quant scales by device.

### 7.5 What it must never do

1. Never apply a config that fails the +4%/no-lane-worse gate — calibration proposes, the gate disposes.
2. Never cache without device + engine identity in key and validators in file header.
3. Never tune on an unwarmed host — honor the preboot/aging doctrine (warmup legs, clocks settled per the clock-governance digest) before measuring.
4. Never let calibration write into model artifacts or mutate defaults silently — separate stores, explicit apply, `--calibrate-verify` for revalidation.
5. Never trust single-shot wall-clock comparisons — reps, warmups, quantiles, interleaved order.
6. Never auto-tune at every boot by default — one-time plus `--calibrate-verify`; vLLM's every-boot profile is the anti-pattern when boot latency matters.
7. Never exceed the budget caps — abort phase, keep incumbents, report partial (TunableOp 30 ms/100-iter precedent).

---

## Sources

Primary (fetched/quoted this pass):
- TensorRT-LLM AutoTuner source: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/autotuner.py
- TensorRT-LLM env table + cache path: https://nvidia.github.io/TensorRT-LLM/llm-api/index.html ; https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_bootstrap.py
- TensorRT-LLM MoE perf guide: https://nvidia.github.io/TensorRT-LLM/developer-guide/moe-perf-guide.html
- TensorRT-LLM perf tuning guide: https://nvidia.github.io/TensorRT-LLM/legacy/performance/performance-tuning-guide/index.html
- trtllm-bench docs: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/commands/trtllm-bench.rst
- TRT-LLM layer-wise benchmarks (cache reuse discipline): https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/layer_wise_benchmarks/README.md
- vLLM optimization/tuning: https://docs.vllm.ai/en/stable/configuration/optimization.html
- vLLM v1 gpu_worker (`determine_available_memory`): https://docs.vllm.ai/en/v0.32.0/api/vllm/v1/worker/gpu_worker
- vLLM bench + sweep: https://docs.vllm.ai/en/latest/cli/bench.html ; https://docs.vllm.ai/en/latest/cli/bench/sweep.html
- vLLM FP8: https://docs.vllm.ai/en/v0.5.4/quantization/fp8.html
- SGLang hyperparameter tuning: https://docs.sglang.io/advanced_features/hyperparameter_tuning.html
- llama.cpp llama-bench: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- llama.cpp imatrix: https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md
- TVM MetaSchedule deep dive: https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html
- MLC-LLM compile docs: https://llm.mlc.ai/docs/compilation/compile_models.html
- ONNX Runtime TRT EP caches: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- Triton autotuner source: https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
- PyTorch TunableOp: https://docs.pytorch.org/docs/2.15/cuda.tunable.html
- llm-compressor README: https://github.com/vllm-project/llm-compressor
- OpenTuner README: https://github.com/jansel/opentuner
- tuned README: https://github.com/redhat-performance/tuned

Secondary (search-verified snippets):
- OtterTune paper: https://arxiv.org/abs/1711.00006 ; https://db.cs.cmu.edu/papers/2017/p1009-van-aken.pdf
- vLLM V1 memory accounting issues: https://github.com/vllm-project/vllm/issues/12529
- TunableOp RDNA3 necessity report: https://github.com/ROCm/legacy-rocm-build/issues/5040
- ModelOpt calibration API: https://docs.nvidia.com/modelopt/
- SGLang benchmark/profiling reference: https://docs.sglang.ai/references/benchmark_and_profiling.html
