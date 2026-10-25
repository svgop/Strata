# SCOUT DIGEST — Stable baseline + measurement law v2 (kill the boot lottery)
Box: RTX 5090 (GB202) + Ryzen 9 9950X3D. Symptom: identical recipe, solo t/s mean spans 189.8–219.2 across boots (~14% spread).
Date: 2026-10-25. Sources checked against live repos/docs (links inline).

## Per-implementation findings

### 1. NVIDIA official clock practice (nvidia-smi / Nsight) — GRADE A- (adopt)
- **Commands**: `sudo nvidia-smi -pm 1` (persistence); `nvidia-smi -q -d POWER` (query min/max pl); `sudo nvidia-smi -lgc <min>,<max>` / `-lmc <mhz>`; undo with `-rgc` / `-rmc`; cap power with `sudo nvidia-smi -pl <W>`. NVML equivalents: `nvmlDeviceSetGpuLockedClocks`, `nvmlDeviceSetPowerManagementLimit`. NVIDIA's SetStablePowerState blog (developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate) recommends query-supported-clocks → lock core AND memory → run → reset both, and warns `SetStablePowerState` (DX12) does NOT lock memory clocks, "results less comparable".
- **SOL-ExecBench** (NVIDIA, arXiv 2603.19173, NVIDIA/SOL-ExecBench): every run uses a single GPU with **SM clocks locked at 1,500 MHz via nvidia-smi** "to reduce frequency-scaling noise"; still reports **~5% run-to-run latency variance** remains. Precedent that locking alone does not kill variance below ~5%.
- **Nsight Compute**: by default ncu locks clocks to BASE (`--clock-control base`); Jan.ai/ML-eff (jan.ai/post/how-we-benchmark-kernels) found base-lock distorts latency (676.64→575 μs with `--clock-control none` on RTX PRO 6000 Blackwell) and that **locking to max "just sets the ceiling"** — the card still drops toward base (~2287 vs 2617 MHz) under power caps. Their fixes that we should copy: 10 warmup + 20 repeats, report **median**; L2 flush buffer (zero an L2-sized tensor before each rep); an untimed dummy matmul so short kernels don't outrun the CPU enqueuing the end event; reconcile cross-clock numbers via FLOPS ∝ SM clock.
- **5090/Blackwell quirks (GSP-RM)**: open-gpu-kernel-modules issue #1268 — on RTX 5090, after final GPC-clock arbitration **voltage is NOT re-evaluated**, so a memory-clock lock left ~142 W of board headroom unused and *reduced* RT performance; max-only feedback via `nvmlDeviceSetGpuLockedClocks(dev, 0, current_pmu_mhz)` re-triggers V/F evaluation (same FPS, −10.9 mV, −15.6 W). Practical consequences: (a) lock SM clocks to a **sustained-stable value, not max**; (b) be suspicious of `-lmc` on GB202 — validate before use; (c) 5090 min power-limit floor is high (community report r/nvidia 1myfsbi), so `-pl` is a coarse knob; clock-lock-at-knee beats power-capping.

### 2. MLPerf Inference harness + power rules — GRADE A (best public reject-gate precedent)
- **NVIDIA closed tuning guide** (inference_results_v3.1/closed/NVIDIA/documentation/performance_tuning_guide.md): `sudo nvidia-smi -lgc 1590,1590` "lock the GPU clocks at max frequency" (T4 example); thermal check via `nvidia-smi dmon -s pc`, temperature should **stabilize** (their example ~65C) before trusting runs; NUMA audit (`numactl --hardware`, `nvidia-smi topo -m`); THP `always` for server scenario.
- **Telemetry gates that actually reject runs** (tools/submission/power/power_checker.py, mlcommons/inference): testing-mode duration must be within **5%** of ranging-mode duration (reject above); average power during testing must not be **>5% below** ranging; loadgen begin/end timestamps must sit inside each mode's window. Power runs additionally require **locked GPU clocks** per MLPerf rules, with power measured by external PTDaemon analyzer (docs/power).
- Failure mode they guard against: comparing a cold ranging run to a hot testing run — exactly our boot-lottery shape. Adopt: **pair legs back-to-back (interleaved), gate on duration delta ≤5% per leg**, gate on temperature stabilization before leg 1.

### 3. airbench (KellerJordan/cifar10-airbench) — GRADE B- (minimal but honest)
- Exact discipline (airbench/__init__.py, utils.py): `run=-1` = dedicated warmup entry point (`warmup94(...)`); timing loop records CUDA events, `torch.cuda.synchronize()`, accumulates `starter.elapsed_time(ender)` per epoch; paper (arXiv 2404.00498): "The only purpose of the first run is to warmup, so we" exclude it — report mean over subsequent runs on one A100. No clock locks, no reject gates, no medians; variance controlled purely by warmup + repetition.
- Note: the task's "tinygrad airbench" is a mislabel — airbench is KellerJordan's; tinygrad's own material is docs.tinygrad.org/developer/speed (kernel grouping/memory-bound analysis, no clock law).
- Takeaway: single-warmup-then-mean is NOT sufficient for a 14% boot spread; it assumes same-boot stability.

### 4. llama-bench (ggml-org/llama.cpp tools/llama-bench; ik_llama.cpp inherits) — GRADE C (stats yes, discipline no)
- Defaults verified in source: `-r/--repetitions` **5**, warmup **enabled** (`--no-warmup` to skip), `--delay <s>` between tests, `--prio <-1..3>` (raise priority), `--numa <distribute|isolate|numactl>`. Reports `avg_ns` + `stddev_ns` (sample stdev, n−1) and `avg_ts`/`stddev_ts`. **No median, no outlier rejection, no clock/thermal gating** — garbage runs flow straight into the average. ik_llama.cpp's bench tooling is the same llama-bench lineage plus CPU offload flags (--cpu-moe, -ot), no extra variance law.
- Takeaway: llama-bench numbers with wide stddev are exactly our 189.8–219.2 signature; we need our own gates, its stddev is the alarm bell.

### 5. Linux CPU side on Zen5 X3D — GRADE B+ (fragmented sources, concrete knobs)
- Kernel doc (Documentation/admin-guide/pm/amd-pstate.rst): amd_pstate has **active mode** (`amd_pstate=active`, EPP hints to CPPC firmware which autonomously ramps clocks by power/thermal) vs **passive mode** (driver requests, governor executes). EPP writable per-CPU at `/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference` (named profiles or integer 0–255, lower = performance-biased); global amd_pstate controls live in `/sys/devices/system/cpu/amd_pstate/` (incl. `cpb_boost`, kernel ≥6.6, and per-CPU `boost`).
- Concrete practice for consistent benchmarking (Framework community thread on amd_pstate benchmarks; Phoronix 9950X3D methodology on OpenBenchmarking.org): `sudo cpupower -c all frequency-set -g performance`; EPP → `performance`; **disable core boost** (`echo 0 | sudo tee /sys/devices/system/cpu/amd_pstate/cpb_boost`, or generic `/sys/devices/system/cpu/cpufreq/boost`) — trades peak for flat, thermally-stable clocks; **pin to the V-cache CCD** (CCD0 on 9950X3D; find CPUs via `lscpu -e=CPU,CORE,CACHE,L3`) with `taskset -c` so the scheduler can't hop to the frequency-advantaged non-X3D CCD mid-run. acpi-cpufreq (legacy) gives cruder direct control; amd_pstate active + EPP performance is the modern equivalent and what distros ship on Zen5.
- Failure modes: boost re-enables after BIOS update; cpb_boost resets on module reload; per-CPU EPP drifts when tuned daemons (power-profiles-daemon/tlp) run.

### 6. Continuous single-box setups with per-run telemetry — GRADE B
- **Phoronix Test Suite `system_monitor` module** (docs PDF, features page): records sensors (CPU freq, CPU/GPU temp, power, fan) **per run into the result file**, viewable/plotted per run — the closest public template for "attach telemetry to every leg, keep or reject later". Enable via pts config modules/system_monitor (sensors list, `Monitor: All`).
- **GPU MODE KernelBot** (gpu-mode/kernelbot + reference-kernels): runs every submission on fixed cloud GPUs with warmup + repeated timed trials; per-run distributions published (GPUMODE/kernelbot-data on HF). Clock-gate rules are not public in the repos I could read (modal_runner has no public clock check) — grade reflects practice-at-scale, not copyable gates.
- No public single-box setup publishes a full "clock-sag reject rule" we can lift verbatim; MLPerf's 5% + NVIDIA's dmon temperature-stabilization + PTS per-run sensor logging are the three copyable pieces.

## READY-TO-RUN command block (sudo + verify + rollback)

```bash
# ---- PRE (sudo) : run before every measurement session, same order every boot
sudo nvidia-smi -pm 1                       # persistence daemon mode
nvidia-smi -q -d POWER | grep -A3 "Power Limit"   # read Min/Max/Default pl first
sudo nvidia-smi -pl 450                     # below default 575W: firm ceiling, less sag
# pick SM lock at the sustained knee, not max (Blackwell V/F quirk, issue #1268);
# 1500 MHz is NVIDIA's own SOL-ExecBench value; try 1500 and 1800 in calibration:
sudo nvidia-smi -lgc 1500,1500
# NOTE: do NOT -lmc on the 5090 until calibration proves it's neutral (issue #1268)

# CPU: flat clocks, no boost lottery
sudo cpupower -c all frequency-set -g performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
echo 0 | sudo tee /sys/devices/system/cpu/amd_pstate/cpb_boost   # disable core boost
# stop tuned/profile daemons during sessions: sudo systemctl stop power-profiles-daemon

# Pin benchmark to V-cache CCD (find the L3-big CPUs first):
lscpu -e=CPU,CORE,CACHE,L3                  # e.g. CCD0 = V-cache
taskset -c <v-cache-ccd-cpus> <your bench cmd>

# ---- VERIFY (no sudo)
nvidia-smi -q -d CLOCK,POWER -l 5           # SM clock pinned; "Clocks Event Reasons" idle-only
nvidia-smi dmon -s pc                       # temp must stabilize (~65C rule from MLPerf guide) before leg 1
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # performance
grep . /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | sort -u  # flat, no boost spikes

# ---- PER-LEG TELEMETRY + REJECT GATES (wrap each leg)
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,\
clocks_event_reasons.active --format=csv -l 1 > leg_telemetry.csv &
# REJECT the leg if any of:
#   SM clock deviates >0.5% from locked value for >2 consecutive samples
#   any throttle reason other than "Idle" active
#   leg t/s differs >5% from paired leg (MLPerf duration-delta rule)
#   temp failed to stabilize before first sample

# ---- ROLLBACK (post-session)
sudo nvidia-smi -rgc                        # unlock SM clocks
sudo nvidia-smi -pl 575                     # restore default pl (use value from -q -d POWER)
echo 1 | sudo tee /sys/devices/system/cpu/amd_pstate/cpb_boost
sudo cpupower -c all frequency-set -g schedutil   # or your distro default governor
# EPP back: echo balance_performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
```

## OUR NEXT STEP (calibration form to fill on the box)
For each candidate SM lock {1500, 1650, 1800, unlocked} × boost {off, on}, over 3 boots:
record per-leg mean t/s, stddev, min/max SM clock, temp-at-start/end, and board power.
Fill in:
1. `LOCK_MHZ = <argmin knee>` where power headroom stops binding (t/s plateaus, sag-seconds = 0)
2. boot-to-boot span of leg means under pinned settings (target: <<14%; SOL-ExecBench says ~5% floor is realistic, our paired-interleave should beat it)
3. reject thresholds from observed distributions: `THRESH_CLOCK=0.5%`, `THRESH_PAIR=5%`, `T_STAB=<observed temp-stabilization time>` — replace the placeholders above with measured values before freezing protocol v2.

## Sources
- https://docs.nvidia.com/deploy/nvidia-smi/index.html — official -lgc/-lmc/-pl semantics
- https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate — lock-both-clocks procedure
- https://arxiv.org/html/2603.19173v1 + github.com/NVIDIA/SOL-ExecBench — 1,500 MHz lock, ~5% residual variance
- https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1268 — 5090 GSP voltage re-evaluation quirk
- https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus — practical -lgc syntax
- https://raw.githubusercontent.com/mlcommons/inference_results_v3.1/main/closed/NVIDIA/documentation/performance_tuning_guide.md — `sudo nvidia-smi -lgc 1590,1590`, dmon temp-stabilization, NUMA
- https://github.com/mlcommons/inference tools/submission/power/power_checker.py + docs/power — 5% ranging/testing gates, PTDaemon
- https://github.com/KellerJordan/cifar10-airbench (airbench/__init__.py, utils.py) + https://arxiv.org/pdf/2404.00498 — run=-1 warmup, discard-first-run
- https://jan.ai/post/how-we-benchmark-kernels — Blackwell lock-to-max caveat, median-of-20, L2 flush
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/llama-bench/llama-bench.cpp — reps=5, warmup, avg±stddev
- https://raw.githubusercontent.com/torvalds/linux/master/Documentation/admin-guide/pm/amd-pstate.rst — EPP, boost, active/passive
- https://community.frame.work/t/responded-linux-amd-pstate-scaling-driver-benchmarks/45830, https://www.phoronix.com/review/amd-9950x3d2-windows-linux/5 — X3D methodology
- https://fossies.org/linux/phoronix-test-suite/documentation/phoronix-test-suite.pdf — system_monitor per-run sensors
- https://github.com/gpu-mode/kernelbot, https://huggingface.co/datasets/GPUMODE/kernelbot-data — leaderboard-scale repetition practice
