---
type: digest
id: w0-measure-scout
created: 2026-10-31
status: evidence
topic: W0 baseline & measurement discipline — how leading GPU/LLM benchmark suites handle drift
scope: read-only research; sources fetched from upstream main/master 2026-10-31
---

# W0 — Measurement discipline: competitor digest

Problem shape: strata gates optimizations on 3-leg same-hour A/B batteries (solo/pair/6-way), sees M1 180-222 t/s drift across same-boot windows, chronic ~6% SM clock deficit (always-on Blackwell "Reliability" flag), episodic 1785 MHz power-cap transients during concurrent cold-prefill. Fields per suite: warmup, statistics, monitoring, A/B protocol, variance numbers, grade.

## vLLM (`vllm bench serve`, ex-benchmark_serving.py)
- Warmup: `--num-warmups`, DEFAULT 0 — opt-in, sent through the same concurrency semaphore as the timed run, excluded from metrics. No cache flush; dataset state persists between runs.
- Statistics: mean/std/median + percentile set per metric; `--metric-percentiles` default "99"; `--percentile-metrics` default "ttft,tpot,itl" (e2el for non-streaming). Reports for each of TTFT/TPOT/ITL/E2EL: `mean_*, std_*, median_*, percentiles_*`.
- Sample size: `--num-prompts` default 1000; `--request-rate` default inf (open-loop controlled instead by `--max-concurrency`).
- Hygiene: `freeze_gc_heap()` before the run ("Avoid GC processing static data — reduce pause times"); single-token probe requests measured separately to see how the main workload stalls unrelated requests.
- Protocol: single timed pass per invocation; repeatability is the caller's job — no variance gate shipped.
- Grade: verified-from-source (main, 2026-10-31).
- https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py

## SGLang (`sglang.benchmark.serving` + steady_state_serving)
- Warmup: `--warmup-requests` default 1, "run warmup requests with short output first"; warmup launched at the same concurrency as the measured run — code comment: "Keep warmup at the same concurrency as the measured run. Otherwise, --warmup-requests also becomes an unbounded warmup burst." Warmup failure aborts the benchmark.
- Cache control: `--flush-cache` POSTs `/flush_cache` (sglang) or `/reset_prefix_cache` (vLLM) after warmup; vLLM path requires `VLLM_SERVER_DEV_MODE=1` — "the benchmark fails loudly rather than accidentally measuring warm-cache performance." Docstring: "Flush an engine's prefix cache after benchmark warmup."
- Statistics: fixed p50/p90/p95/p99 for TTFT/TPOT/ITL/E2E per request set.
- Steady-state mode (dedicated entry point): `--steady-state-concurrency-ratio` (required, (0,1]; example 0.8) — computes concurrency-over-time from per-request start/latency, threshold = ceil(peak*ratio), and reports throughput only over "the longest continuous interval above the concurrency threshold" — i.e. ramp-up/cold segments are excised from the headline number.
- Grade: verified-from-source (main, 2026-10-31).
- https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/steady_state.py ; https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx

## MLPerf Inference (rules v6-pre master + loadgen config + audit + ISCA'20 paper)
- Warmup: explicitly untimed — SUT ops include "compilation, cache warmup, and preprocessing"; timed phase starts only after. No prescribed warmup count; LoadGen semantics (min duration AND min query count must both be met) absorb ramp.
- Statistics: latency percentile fixed per scenario — Server = 99% ("no more than 1% of queries may exceed the latency bound"), SingleStream 90%-ile, MultiStream 99%-ile. Repeatability rule from the paper: "we require five runs for the server scenario, with the result being the minimum of these five. The other scenarios require one run."
- Stabilization: every scenario `min_duration = 600000` (600 s) in mlperf.conf; paper rationale verbatim: "The minimum run time ensures we measure the equilibrium behavior of power-management systems and systems that support dynamic voltage and frequency scaling (DVFS)." (Paper-era floor was 60 s; current conf = 10 min.)
- Variance/repeatability enforcement: audit guidelines — re-run results within "2% as the threshold for 'materially different' performance"; audit checklist item "how is the clock set and does it keep consistency throughout the inference?"
- Grade: verified-from-source (rules adoc, mlperf.conf, audit guidelines, arXiv:1911.02549).
- https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc ; https://github.com/mlcommons/inference/blob/master/loadgen/mlperf.conf ; https://arxiv.org/abs/1911.02549

## NVIDIA practice (TensorRT perf-benchmarking doc + Triton perf_analyzer)
- Monitoring: "While measuring performance, it is recommended that you record and monitor the GPU status in parallel to the inference workload" — `nvidia-smi -q` snapshot BEFORE (power cap, clock settings) plus `nvidia-smi dmon -s pcu -f <FILE> -c <COUNT>` tracing clocks/power/temperature/utilization DURING every benchmark.
- Clock doctrine: `-lgc` locks for "deterministic comparisons and consistent performance measurements"; floating clocks ok for peak but "running TensorRT workloads with floating clocks or with throttling taking place can lead to ... unstable performance measurements." Locked clocks still fall to power/thermal throttling — "when the throttling kicks in, the device behaves like the clock floats."
- Measured variance: "Two runs of the same engine on the same GPU can disagree by tens of percent if the clock floats during one run and locks during the other, or if thermal throttling kicks in halfway through." Power-throttle skew hits short windows: "unstable performance measurements if the measurements are taken within a short time, such as within 20ms"; gappy (sync-per-iteration) benchmarks overstate sustained throughput. Thermal throttle begins at `GPU Slowdown Temp` (`nvidia-smi -q -d TEMPERATURE`).
- Triton perf_analyzer: stability gate shipped in the tool — "The measurement is considered stable if the ratio of max/min from the recent 3 measurements is within (stability percentage)% in terms of both inferences per second and latency. Default is 10 (%)"; windows = 5000 ms (`--measurement-interval`) or ≥50 requests (`count_windows`); `--warmup-request-count` default 0; trtexec headline metric = median latency + min/mean/p90/p95/p99 summary.
- Grade: verified-from-source (docs.nvidia.com current pages, 2026-10-31).
- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/cli.html ; https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html

## Ray llmperf
- No warmup phase; RNG seeded (`random.seed(11111)`); per-request records; reports mean and stddev of TTFT, ITL, E2E latency, per-request output throughput across completed requests. Correctness companion script runs repeat rounds. Statistical handling is thin — the weakest of the surveyed suites.
- Grade: verified-from-source (main, 2026-10-31).
- https://github.com/ray-project/llmperf/blob/main/token_benchmark_ray.py

## Kernel-bench practice (llama-bench, Triton do_bench, torch.utils.benchmark, 2026 kernel papers)
- llama-bench: 1 warmup prompt run + 1 warmup gen run before timing; `-r/--repetitions` default 5; reports avg_ns/stddev_ns and avg_ts/stddev_ts columns; per-depth ladder with prefill state snapshot/restore so repeated reps don't re-pay prefill; memory cleared between reps.
- Triton `do_bench(fn, warmup=25, rep=100)`: time-budgeted (counts derived from a 5-call estimate), L2 cache cleared before EVERY timed iteration, CUDA events, returns mean/median/min/max/quantiles.
- torch.utils.benchmark: warmup built in; replicates mandatory — "run-to-run variation is a significant confounding factor. It is expected that all measurements should include replicates to quantify noise and allow median computation, which is more robust than mean"; `adaptive_autorange` repeats "until iqr/median is smaller than threshold" (default 0.1 = 10% noise gate); A/B via `env` tag + Compare so both arms merge cleanly; wall-time nondeterminism ("adaptive clock speeds") is why Callgrind instruction counts are offered as the reproducible complement. Recipe demo: first bmm call 2775.5 us vs 22.4 us warm (cuBLAS load) — cold-start skew >100x.
- 2026 kernel papers: FastKernels — "GPU clocks are locked at 1,593 MHz for reproducible timing" (H100), "Each measurement is averaged over 3 runs with 10 warmup iterations", speedups with 95% bootstrap percentile CIs (10k resamples) "to quantify variability under noisy serving conditions". NVIDIA SOL-ExecBench — clocks locked via nvidia-smi "1,500 MHz for B200", benchmarking serialized per GPU, "10 warmup iterations and 50 timed iterations per trial over 3 trials, with the reported runtime taken as the mean across trials", L2 cleared by zeroing a 256 MB buffer before each timed iteration, fresh input addresses per run.
- Grade: verified-from-source (code + papers, 2026-10-31).
- https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp ; https://github.com/triton-lang/triton/blob/main/python/triton/testing.py ; https://docs.pytorch.org/docs/stable/benchmark_utils.html ; https://arxiv.org/abs/2605.23215 ; https://arxiv.org/abs/2603.19173

## Cross-suite variance anchors
- Noise-floor shapes in the wild: 10% max/min over 3 windows (Triton default), 10% iqr/median (torch adaptive stop), 2% "materially different" (MLPerf audit), min-of-5 runs (MLPerf Server). None of the surveyed suites trust a single A/B pair — every one either replicates (MLPerf, llama-bench, SOL-ExecBench, FastKernels), windows-and-gates (Triton, torch), or excises unstable segments (SGLang steady-state).
- No suite publishes a same-boot window-drift number for LLM serving throughput; strata's 180-222 t/s band (~21% max/min) sits far above every published noise treatment, consistent with TensorRT's "tens of percent" clock/thermal explanation.

## What this changes on the row
- Adopt a two-tier noise floor: verdict gate = Triton's rule (max/min across >=3 same-boot windows <= 10%) for within-window health, and MLPerf audit's 2% as the minimum delta a build-vs-build verdict must exceed; deltas inside the band are recorded as "no change", never as wins.
- Make every battery leg replicate MLPerf Server: >=5 passes per leg, report the min (robust to which pass catches a power-cap transient); a single pass never produces a verdict.
- Fix warmup + cache hygiene per SGLang: warmup at the leg's exact concurrency, exclude from metrics, flush prefix cache after warmup, abort the run if warmup fails; then add vLLM's hygiene (freeze GC heap) — cold-prefill storms must be measured in a dedicated leg, not leak into steady legs.
- Trace clocks/power/temperature through every battery (TensorRT doctrine): `nvidia-smi -q` snapshot before, `nvidia-smi dmon -s pcu` (or DCGM) to file during; verdict windows invalid where SM clock deviates beyond a band around the leg median (catches the 1785 MHz transients and the ~6% flag deficit) — excluded windows trigger a re-run, not a silent mean.
- Enforce a per-run minimum duration to DVFS equilibrium (MLPerf: 600 s; paper rationale quoted above) plus boot/settle spacing — a fixed idle-or-sustain settle (e.g. >=1 min at leg load) before first timed token of each leg; interleave legs A-B-A-B-A-B instead of sequential blocks (torch `env`-tag rationale: both arms share the same clock/thermal weather).
- Stand up per-depth bands + depth-slope watch on the llama-bench shape: fixed context-depth ladder, prefill state snapshot/restore, 5 reps per depth, report mean+-stddev and p50/p99 TTFT/ITL per depth bucket; alert when the t/s-vs-depth slope changes more than the 2% verdict floor between builds.
