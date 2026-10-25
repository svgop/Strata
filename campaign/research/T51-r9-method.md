# T51 / R9 — Benchmark variance control on a single box: literature + protocol proposal

Date: 2026-10-25. Researcher: R9. Scope: how production labs + systems literature handle
single-box benchmark variance (GPU/CPU control, A/B statistics, engine guides, regression tooling),
and a concrete protocol proposal for OUR campaign (RTX 5090 + 9950X3D, measured boot sigma ~6-8%,
leg sigma ~5%, intraday drift ±8%, background-process cost 3-5%).

---

## PART 1 — PER-SOURCE FINDINGS (graded A = authoritative/quantitative, B = strong production practice, C = community/anecdote)

### (a) GPU/CPU variance control

**[A] NVIDIA CUTLASS — GEMM Performance Measurement Methodology Guidelines** (docs.nvidia.com/cutlass/4.7.0/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)
The single most load-bearing source found. NVIDIA's own doctrine:
- Two sanctioned modes: **fixed-frequency** (lock GPC clock with `nvidia-smi -lgc`, verify it is *sustained* throughout; if not, lower it) and **fixed-power** (default DVFS, but then clocks/power/temp MUST be monitored and reported).
- Warmup until power draw stabilizes: GPC clocks oscillate for roughly the **first ~3 s** of back-to-back GEMMs; example protocol uses **10,000 warmups + 4,000 timed iterations + 1 s cooldown between tests**; ">1000 iterations gives stable results" for large problems; smaller problems need MORE iterations.
- **Cooldown between tests** returns GPU to idle power/temp/clock so one test doesn't contaminate the next.
- **Buffer rotation** (footprint ≥ 2× L2, ≥ 2 buffers, rotated pointers) so repeated iterations don't get fake L2 hits — i.e., warm-state is *specified*, not accidental.
- Validate before trusting the number: stable GPC frequency during timed window, low per-iteration L2 hit rate, minimal inter-kernel gaps.

**[A-] NVIDIA Nsight Compute Profiling Guide** — documents `nvidia-smi --lock-gpu-clocks=<min>,<max>` (setting both = base/TDP frequency) and `--reset-gpu-clocks` as the official mechanism.

**[B+] NVIDIA SOL-ExecBench blog** (research.nvidia.com) — NVIDIA's speed-of-light kernel benchmark **locks GPU clocks** via nvidia-smi at a hardware-specific frequency (e.g., 1500 MHz on B200) explicitly "to improve reproducibility". Confirms clock-locking is current NVIDIA benchmark practice, not overclocker folklore.

**[A-] TensorRT for RTX — Performance Benchmarking guide** — thermal throttling begins when GPU hits a driver threshold (~85 °C typical; 4090-class cards commonly cited at 82 °C core), at which point clocks drop. Requires thermal headroom or your benchmark measures your cooler, not your change.

**[A-] TensorRT-LLM perf-benchmarking docs** (nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html) — the opposite, equally valid pole: recommends `sudo nvidia-smi -pm 1` (persistence), setting `-pl` to the max power limit, and **resetting clocks (`-rgc`) and letting DVFS run** — explicitly warning that locking at MAX frequency "can sometimes lead to thermal throttling". Note the reconciliation: both camps agree the sin is *uncontrolled* state; CUTLASS locks to a *sustainable* (sub-max) frequency, TRT-LLM leaves DVFS but pins power limit + persistence. `trtllm-bench` exists specifically to make published numbers reproducible; includes a warmup phase in its runs.

**[A-] NVIDIA driver persistence docs** + StackOverflow — without persistence mode the driver unloads when the last client exits; the next run pays GPU re-init (UVM harmonization etc.). "Kills GPU cold starts" — one of the mechanisms behind our fresh-vs-warm boot asymmetry.

**[C+/B-] Community (overclock.net, Reddit, MSI guide, Facebook group)** — reboot-to-reboot differences have plausible mechanisms: **DRAM retraining on every boot** (alters memory perf even at fixed timings), cold-boot clock ramp behavior, driver state. Typical modern run-to-run spread ~2% when healthy. For the 9950X3D: the maximal-determinism recipe is BIOS **disable CPB (Core Performance Boost) + PBO, fixed all-core ratio** — but runtime alternatives exist (below), so no BIOS trip needed.

**[B] Mytkowicz, Diwan et al., "Producing Wrong Data Without Doing Anything Obviously Wrong!" (ASPLOS 2009, ~588 citations)** — measurement bias from *innocuous* setup facts (UNIX environment size, link order) flips benchmark conclusions; remedy: **randomize setup across runs + causal checks**. The canonical citation for "your boot lottery is real and the fix is randomization + blocking, not more repeats of one boot".

### (b) Statistical methodology for A/B in systems eval

**[A] Kalibera & Jones, "Rigorous Benchmarking in Reasonable Time" (SIGMETRICS 2013)** + companion "Quantifying Performance Changes with Effect Size Confidence Intervals" (arXiv:2007.10899) — the standard factorial model of benchmark noise (layers: iterations < runs < ... ), how to *choose* repetitions to hit a target CI width, and effect-size CIs as the honest A/B verdict. Directly justifies: estimate σ first, then derive N — do not pick N by folklore.

**[B+] Google Benchmark — Random Interleaving** (google.github.io/benchmark/random_interleaving.html, issue #1051) — production implementation of interleaved repetitions to lower run-to-run variance *without increasing execution time*. This is the same idea as our ABAB interleave, shipped in the most-used C++ benchmark harness.

**[B] "Robust benchmarking in noisy environments" (arXiv:1608.04295)** — benchmark strategy robust to timer error/OS jitter/environment drift; pairs with Kalibera-Jones.

**[B] Interleaving literature from online experiments (Chapelle et al. 2012; Amazon Science; Airbnb, DoorDash engineering posts)** — paired/interleaved designs cancel shared environmental variance and yield large sensitivity gains (Airbnb: ~50x fewer samples than A/B; DoorDash: within-subject variance control). Caveat documented: position/carryover bias → randomize which arm goes first.

**[A] MLPerf Inference run discipline** (inference_rules.adoc; blog.lo0.es LoadGen walkthrough; MLCommons endpoints compliance checks; ISCA'20 paper):
- LoadGen enforces **warmup-present** as a compliance check; performance runs have **minimum durations (≥600 s)** and **minimum query counts derived from a 99%-confidence table** (e.g., 23,886 → rounded 24,576 samples at 90% latency CI; single-stream min 1,024 queries; offline min 24,576 samples).
- Mandatory unmodified LoadGen, one accuracy run per perf result, announced seeds, **audits** (random + committee-picked submissions), and Power rules explicitly forbid cherry-picking ("no combining best perf from one run with lowest power from another"). Translation for us: pre-registered exclusion rules only; never mix legs across runs to compose a number.

**[B] AIMS Lab (Stanford), "Quantifying Variance in Evaluation Benchmarks" (ICML 2025 oral)** — variance in evals is itself measurable and reportable; treat variance estimation as a first-class deliverable.

### (c) Production engine practice

**[A-] vLLM benchmarking docs** (docs.vllm.ai/en/latest/benchmarking/cli, `vllm bench throughput`, discussion #7181):
- Official warning: **repeating `vllm bench serve` against the same server reuses prefix-cache entries and inflates throughput** — this is a measured-state contaminant, exactly our page-cache/warm problem at KV level. `vllm bench throughput` has `--num-warmups`; community guide: always warm up and discard early requests; `vllm bench sweep` exists for config sweeps.

**[B+] llama.cpp `llama-bench`** — defaults encode the doctrine: **5 repetitions per test, warmup run on by default** (`--no-warmup` exists), md/csv/json output. The bar every local benchmark user already runs through.

**[B+] SGLang `bench_serving` guide + sglang-omni benchmarks README** — warmup flags (`--warmup`, `--warmup-meta`), multi-backend support; LMSYS "deterministic inference" post (Sep 2025) documents CUDA-graph determinism work (relevant to per-run state stability).

**[B] Mozilla AI cross-engine benchmarking (llama.cpp/llamafile/LM Studio/Ollama) + ewams.net llama-bench guide + TechnoLynx guide** — consistent community protocol: pin everything you can (quant, threads, batch, context), warm up, repeat, report medians; direct C++ counters may not represent end-user experience (NVIDIA-forum llama-benchy note).

**[B] 5sigmas.com LLM inference economics, ch. 6** — "one run does not describe variance"; report rep count, warmup policy, per-rep percentiles, dispersion, and **the variant order** (they flag thermal/background drift as the reason order matters); warmup is part of the experiment — cold vs warm measure different problems, so **declare and fix** the cache state; MLPerf Power no-cherry-picking rule cited.

### (d) 2025-2026 continuous perf-regression tooling

**[B] Bencher (bencher.dev)** — self-hostable continuous benchmarking; statistical thresholding, **bare-metal runner guidance**, hyperfine support ("same bare metal hardware"), OSS. The right shape for our campaign (local, wall-clock, statistical gates).
**[B-/C] CodSpeed (codspeed.io)** — solves CI noise by *instrumented, hardware-independent* measurement — not applicable to GPU wall-clock serving benchmarks, but conceptually the strongest anti-noise play in the space.
**[B-] Brendan Gregg, page-cache hit-ratio regression case (2014)** — a production perf regression *was* a page-cache hit-rate change; justifies logging cache state per leg.
**[C+] vLLM `bench sweep`, DCGM/dcgm-exporter, nvidia-smi dmon** — standard telemetry for per-leg validation logs.

**Bottom line of the literature:** everyone who publishes trustworthy single-box numbers (1) pins or bounds clocks/power state, (2) treats warmup + cache state as part of the experiment, (3) interleaves and pairs, (4) derives N from a measured σ, (5) logs enough telemetry to reject contaminated runs post-hoc. Our boot lottery is not exotic — it is the un-modeled sum of (DRAM retraining + driver re-init + DVFS history + cache state + ambient), and every one of those has a named control.

---

## PART 2 — PROTOCOL PROPOSAL ("campaign law if it survives our data")

### P0. Statistical model and power arithmetic (from OUR measured numbers)

Model a leg's throughput: `y = arm_effect + boot_effect + residual`, with our measurements:
- boot σ_b ≈ 6–8% (solo means 189.8–219.2 across boots),
- leg residual σ_w ≈ 5% within a boot,
- intraday drift ±8%, background processes 3–5%.

**Why single-boot sequential (non-interleaved) A/B is hopeless:** if arm B is measured in a different boot/hour than A, the comparison inherits `σ_between ≈ √(σ_b² + σ_b²) ≈ 9–11%`. To get a ±3% CI half-width you need `n ≥ (1.96·10%/3%)² ≈ 43` boots **per arm**. Infeasible. This is the arithmetic proof of our boot lottery.

**Why interleaved paired wins:** pair each B leg with the adjacent A leg inside the same boot (ABAB). The boot effect cancels in the difference. The paired delta has `σ_d ≈ √2·σ_w·√(1−ρ)` where ρ = correlation from shared recent thermal/cache state; conservative bound ρ=0.

Power formula (two-sided, α=0.05): `n_pairs ≥ ((z_{1−α/2} + z_{power})·σ_d / δ)²`, with (1.96+0.84)=2.80 for 80% power, (1.96+1.28)=3.24 for 90%.

Detecting **δ = +3%**, per σ_d:

| σ_d | legs σ_w | pairs @80% pow | pairs @90% pow |
|---|---|---|---|
| 7.1% (=√2·5%, unpinned, today) | 5% | 44 | 59 |
| 5.7% (=√2·4%, partially controlled) | 4% | 28 | 38 |
| **4.2% (=√2·3%, pinned + gated)** | **3%** | **16** | **21** |

Detecting the **+4% adopt bar**: 9 pairs (pinned) vs 25 pairs (unpinned) at 80% power.
CI half-width ±2% on a solo mean: 24 legs unpinned vs **9 legs pinned**.

**Design conclusion:** with pinning + preconditions driving σ_w to ~3%, **16 paired deltas** resolve +3% at 80% power (21 at 90%). Concretely: **8 blocks of 2v2 (ABBA/BAAB, randomized start) = 32 legs, 16 adjacent pairs**, or **6 blocks of 3v3 = 36 legs, 18 pairs** if per-leg fixed overhead favors longer blocks. All inside ONE boot; forms never straddle a reboot. Unpinned, the same resolution costs 88+ legs — this is why <4% arms became unmeasurable.

Secondary rules:
- **Adopt iff** pooled mean delta ≥ +4% AND paired-delta 95% t-CI excludes 0 (CI is the variance guard; Kalibera-Jones effect-size CI). Arms predicted +3% get a measured CI, not adoption.
- **Rolling σ audit:** re-estimate σ_d from every form's paired deltas; if σ_d > 4.5%, the pinning/preconditions have regressed — fix before running more arms. Adaptive N only by *appending a new full form in a new boot* (never peek-and-stop: fixed blocks per form, optional extension = new form; avoids α inflation).
- **Solo watchdog:** each form opens and closes with one reference-config leg. If (open vs close) drifts >4% or the close leg falls outside the rolling control chart (EWMA of solo legs), the form's time window is contaminated → reject the form. This directly weaponizes our ±8% drift observation.
- Report medians of paired deltas alongside means (bootstrap BCa on 20% trimmed mean as sanity at n≈16-20; t-CI stays primary per "means-only" law).

### P1. Clock/power pinning (sudo steps, ranked by variance-reduced-per-effort)

1. **Persistence mode** (~1 min, kills driver re-init variance / fresh-vs-warm asymmetry):
   `sudo systemctl enable --now nvidia-persistenced` (or `sudo nvidia-smi -pm 1`).
2. **Power limit at max** (removes "is the cap binding?" unknown): `nvidia-smi -q -d POWER` → `sudo nvidia-smi -pl 575`.
3. **Lock GPC clocks to a SUSTAINABLE frequency, not max** (CUTLASS fixed-frequency mode; biggest single leg-sigma reducer: targets σ_w 5%→~3%):
   - observed sm_clock 2955–2970 → lock ~2700–2760: `sudo nvidia-smi -lgc 2700,2700`
   - verify under a 60 s real-model load: `nvidia-smi dmon -s pcu -d 1 -o T` — sm_clock must be flat, temps < 75 °C, no throttle bits; if power/temp rises into the cap, lower 5% and re-verify. Do NOT lock at max (TRT-LLM warning).
   - leave memory clocks at default (mem-clock variance is negligible; locking mem costs real throughput).
   - after a campaign day: `sudo nvidia-smi -rgc`.
   - caveat (recorded in every result header): locked-clock numbers are for **relative A/B only**; never compare absolute t/s across lock settings (salykova/CUTLASS note).
4. **CPU governor + boost cap** (runtime, no BIOS): `sudo cpupower frequency-set -g performance`; cap boost below max via `/sys/.../scaling_max_freq` (e.g., 5.2 GHz vs 5.7 boost) or BIOS CPB-off if a maintenance reboot ever happens anyway; hold out of deep C-states during forms (`/dev/cpu_dma_latency` holder). Pin engine to the X3D CCD (`lscpu -e` → L3=0 mask, typically cores 0-7) with `taskset`/`numactl --cpunodebind=0 --membind=0`; bench client on the other CCD. `sudo turbostat --interval 1` logged per leg.
5. Optional tier (only if 3-4 don't get σ_d ≤ 4.2%): BIOS CPB/PBO off + fixed all-core ratio.

### P2. Warmup / thermal / state preconditions (per form and per leg)

- **Form = one boot.** Boot → persistence verified → model load → **warmup ≥60 s of the real workload shape** (vLLM `--num-warmups` / first-request CUDA-graph capture + allocator growth) → verify clocks settled (CUTLASS: ~3 s oscillation for kernels; engines need the full graph-capture warmup).
- **Thermal gate:** start each leg with GPU temp ≤ 70 °C; maintain identical A/B duty inside the interleave so both arms ride the same thermal ramp; abort a leg that exceeds 76 °C.
- **KV/prefix cache:** flush between legs (vLLM `POST /flush_cache` / `reset_prefix_cache`, SGLang `flush_cache`) or rotate prompt sets — vLLM docs prove repeats otherwise inflate throughput.
- **Page cache:** weights must be page-cache-warm and *identically* warm for every leg in a form (`fincore` ≥95% of model bytes); never mix a fresh-boot leg with warm-boot legs (fresh-vs-warm asymmetry is lane-dependent). If a cold-cache control is ever wanted, run it as its own declared scenario (5sigmas rule), via `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`.
- **Leg length:** ≥60 s or ≥3k generated tokens per leg, fixed synthetic lengths + greedy decoding so A/B decode the same token counts.
- **Order randomization:** alternate ABBA/BAAB starts per block (Mytkowicz: randomize the innocuous); optionally engine restart every K legs if allocator drift is suspected.

### P3. Per-leg telemetry (makes bad legs rejectable post-hoc)

1 Hz sidecar + start/end snapshots per leg (log to `legs/<leg_id>/`):
- GPU: `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,clocks_throttle_reasons.active,temperature.gpu,temperature.memory,power.draw,power.limit,utilization.gpu,fan.speed,pstate --format=csv -l 1` (DCGM equivalent OK); `nvidia-smi -q -d PERFORMANCE` at leg start/end.
- CPU: turbostat log; per-core cpufreq min/max during leg; snapshot of top consumers (`top -b -n1`).
- State: `/proc/sys/kernel/random/boot_id` + uptime; driver version; `-lgc`/`-pl` values + persistence flag; `/proc/meminfo` Cached/Available; `fincore` % of weights cached; engine log line confirming prefix-cache flush; leg index within boot.
- **Pre-registered reject gates** (leg marked invalid, never deleted; ≥1 invalid leg → rerun form as supersession):
  1. any sample sm_clock < locked − 1%, or unexpected clocks_throttle_reasons bits;
  2. start temp > 70 °C or any sample > 76 °C;
  3. non-campaign CPU use > 5% during leg (covers the measured 3–5% background cost);
  4. weights page-cache < 95% or prefix-flush not confirmed;
  5. solo-watchdog drift > 4% open-vs-close (rejects whole form);
  6. paired delta beyond 3×MAD of the form's pairs → rerun that form, flag, investigate telemetry (never silently drop).

### Ranking (variance reduced per unit effort)

1. **Never split an A/B across boots; ABAB/2v2 interleave + paired analysis** — eliminates the 6–8% boot sigma from comparisons outright; ~zero extra effort (already have the harness).
2. **Clock lock at sustainable freq + persistence + max PL** — expected σ_w 5%→~3%, halves required N; ~15 min of sudo + one verification run.
3. **Warmup + thermal + prefix/page-cache gating** — removes fresh/warm asymmetry and JIT/graph-capture contamination; cheap, scripted.
4. **Telemetry sidecar + pre-registered reject gates** — converts "weird legs" from noise into rejections; medium effort, prevents silent 3–5% contaminations.
5. **CPU governor/boost cap + X3D-CCD pinning** — matters most for CPU-influenced lanes (tokenize, sample, retrieval); low effort.
6. **Rolling σ_d audit + solo-watchdog EWMA (Bencher-style tracking)** — ongoing guard, catches drift before it eats a form.

### Validation plan for this protocol (before it becomes law)

Run ONE calibration form: pin everything per P1/P2, 8×2v2 blocks of the reference config vs itself. Checks: (i) measured σ_d of self-vs-self pairs ≤ 4.2% (if yes, N=16 pairs stands); (ii) solo watchdog delta ≤2%; (iii) no reject gates fire spuriously. Then one known-positive A/B (any previously adopted >4% arm) must reproduce with CI excluding 0 in a single form — that replaces today's 5-boot pooled form.

---

## SOURCE LIST

- CUTLASS GEMM measurement methodology: https://docs.nvidia.com/cutlass/4.7.0/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html
- Nsight Compute profiling guide (clock lock/reset): https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- NVIDIA SOL-ExecBench (clock-locked benchmark): https://research.nvidia.com/benchmarks/sol-execbench/blog/introducing-sol-execbench
- TensorRT for RTX benchmarking (thermal ~85°C): https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/performance/benchmarking.html
- TRT-LLM perf-benchmarking (persistence, -pl, -rgc, warmup): https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html
- NVIDIA driver persistence: https://docs.nvidia.com/deploy/driver-persistence/persistence-mode-legacy.html
- MLPerf Inference rules: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- MLPerf LoadGen comparability walkthrough: https://blog.lo0.es/en/posts/mlperf-inference-reading-loadgen-comparability/
- MLPerf compliance checks (warmup-present, min queries): https://docs.mlcommons.org/endpoints-submission-docs/reference/compliance-checks
- MLPerf paper: https://arxiv.org/abs/1911.02549
- vLLM benchmark CLI (prefix-cache warning): https://docs.vllm.ai/en/latest/benchmarking/cli ; throughput (--num-warmups): https://docs.vllm.ai/en/stable/cli/bench/throughput ; methodology discussion: https://github.com/vllm-project/vllm/discussions/7181
- llama-bench (5 reps, warmup default): https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- SGLang bench_serving: https://lmsysorg.mintlify.app/docs/developer_guide/bench_serving ; deterministic inference: https://www.lmsys.org/blog/2025-09-22-sglang-deterministic
- Kalibera & Jones 2013: https://kar.kent.ac.uk/33611/45/p63-kaliber.pdf ; effect-size CIs: https://arxiv.org/abs/2007.10899
- Mytkowicz et al. ASPLOS'09: https://dl.acm.org/doi/10.1145/1508284.1508275
- Google Benchmark random interleaving: https://google.github.io/benchmark/random_interleaving.html
- Robust benchmarking in noisy environments: https://arxiv.org/abs/1608.04295
- Airbnb interleaving (~50x sensitivity): https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e ; DoorDash: https://careersatdoordash.com/blog/doordash-experimentation-with-interleaving-designs
- Bencher: https://bencher.dev/docs/explanation/continuous-benchmarking , https://bencher.dev/docs/tutorial/bare-metal ; CodSpeed: https://codspeed.io
- Brendan Gregg page-cache hit ratio: https://www.brendangregg.com/blog/2014-12-31/linux-page-cache-hit-ratio.html
- 5sigmas benchmarking chapter (order, warmup-as-experiment, no cherry-picking): https://5sigmas.com/en/series/llm-inference-engineering-economics/06-benchmarking-inference-cost-task-throughput-latency-energy-hardware-constraints
- AIMS (Stanford) variance quantification: https://aimslab.stanford.edu/research
- Clock-lock in kernel work (relative-only caveat): https://salykova.github.io/sgemm-gpu ; power/frequency scaling study: https://arxiv.org/html/2510.06902v1
- 9950X3D determinism recipes (CPB/PBO off): https://www.msi.com/blog/how-to-boost-amd-ryzen-9-9950x3d-gaming-performance ; https://www.overclock.net/threads/issue-with-cpb-core-performance-boost-enabled.1792105
