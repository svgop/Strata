---
type: research-digest
id: 2026-11-07-gpu-clock-governance
date: 2026-11-07
scope: RTX 5090 clock governance for the strata inference service — control surfaces, boost/ramp mechanics, detection, production practice, safety, and the validated v4 service design
subject: NVIDIA clock governance on driver 615.71.09 / GB202 (sm_120), GSP
authorization: operator-owned card; research was READ-ONLY (queries only). No GPU state was written by this research pass.
---

# GPU clock governance digest — RTX 5090 / driver 615.71.09 / strata-gpu-lock

## Verdict (read this first)

1. **The floor-range design is correct and already deployed.** The box runs `strata-gpu-lock` v3 (`/usr/libexec/strata-gpu-lock`): engine listening → `-lgc SM_FLOOR,SM_MAX` + `-lmc MEM_FLOOR,MEM_MAX`; no engine → `-rgc`/`-rmc`. The research mandate's "measured problem" (30 s quiet timer hard-pinning 1800 single-value, bursts paying the 1780 MHz ramp) was v2's failure; v3 already removes the service from the boost critical path. This digest validates v3's assumptions against driver-source evidence, corrects one wrong belief, and lists the deltas for a v4.
2. **Correction: 0x400 is NOT idle.** On this driver, clocks_event_reasons bit 0x400 is `nvmlClocksEventReasonReliability` — "the reliability policy is currently limiting the GPU clocks" (V/F curve margin trim). `GpuIdle` is bit 0x1. Read live twice today: quiet-with-floor shows exactly `0x400` via NVML and `Reliability: Active` in `nvidia-smi -q`.
3. **The floor holds at idle.** Live, with the engine resident and GPU quiet: P3, SM 2062 MHz (floor 1800, DVFS floats above it), mem pinned 7001, 53.6 W, 33 C. The card never fell to the lazy idle P-state. Idle cost of the floor: ~+6-18 W vs the operator's released idle (35-48 W).
4. **3135 is a ceiling, not a sustained state.** Live dmon during real decode: pclk 2805-2850 MHz, mclk 13801, 276-387 W, pviol=1 (SW power cap firing). Sustained-max decode sits ~2830 ± 30 MHz. Budget and bank against ~2850, not 3135.
5. **The single largest v4 delta is safety hardening, not clock logic**: idempotent `ExecStartPre` release, an independent dead-man timer, `-i 0` device scoping, and a runtime ban on `-q -d SUPPORTED_CLOCKS` (GSP crash vector on open-module 610.x).

Confidence scale used below:
- **A** — measured on this box today (telemetry, benchmarks, live NVML).
- **B** — driver-matched source: local `/usr/local/cuda/include/nvml.h` (CUDA 13.4, same train as 615.71.09) or this install's `nvidia-smi -h` text.
- **C** — public doc or GitHub issue naming the same hardware/driver family.
- **D** — community/secondary; treat as a hypothesis, not a fact.

---

## 1. System ground truth (measured 2026-11-07, grade A)

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090 (GB202, sm_120), bus 01:00.0, display attached | `nvidia-smi` |
| Kernel module | **Open** kernel module 615.71.09 | `/proc/driver/nvidia/version` |
| GSP firmware | 615.71.09 | `nvidia-smi -q` |
| CUDA UMD | 13.4 | `nvidia-smi -q` |
| Persistence | `nvidia-persistenced.service` active running, Persistence-M On | `systemctl`, `nvidia-smi` |
| Power limits | Default **575 W**; range 400-600 W | `nvidia-smi -q -d POWER` |
| Max clocks | SM/Graphics 3135 MHz, Memory 14001 MHz | `nvidia-smi -q -d CLOCK` |
| Supported graphics steps | 180 … 3090 MHz (~7-8 MHz grid) under mem 14001 | `nvidia-smi -q -d SUPPORTED_CLOCKS` |
| Thermal limits | Effective T.Limit 57 C (current); slowdown/shutdown spec offsets -2 / -5 C | `nvidia-smi -q -d TEMPERATURE` |
| Quiet + floor state | P3, SM 2062, mem 7001, 53.6 W avg, 33 C, reasons `0x400` | `-q`, NVML probe |
| Busy decode (live traffic) | pclk 2805/2850, mclk 13801, 387 W/276 W, SM 80/100 %, **pviol=1** | `nvidia-smi dmon -c 2` |
| Released idle (operator banks) | 35-48 W | operator measurement |

Poll cost benchmarks (grade A):

| Method | Cost/call | Notes |
|---|---|---|
| NVML `nvmlDeviceGetUtilizationRates` | **0.4 µs** | ctypes probe, 500 iterations |
| NVML `nvmlDeviceGetClockInfo` | 1.7 µs | same |
| NVML `nvmlDeviceGetPerformanceState` | 0.6 µs | same |
| `ss -tlnH sport = :8081` | 3.75 ms | the engine-presence probe |
| `nvidia-smi --query-gpu=...` (spawn) | 23 ms | ~6,000x the NVML path |
| `nvidia-smi dmon` | 1 Hz stream | works on this GSP card; carries pclk, mclk, pwr, pviol, tviol |

pynvml is **not installed** on the system python (`ModuleNotFoundError`). NVML is reachable today via ctypes (`libnvidia-ml.so.1`) or `pip install nvidia-ml-py` if the v4 port wants the µs path.

---

## 2. Clock control surfaces (mandate 1)

### 2.1 `-lgc` / `--lock-gpu-clocks` — range semantics (grade B, driver-matched header)

From the local nvml.h (`/usr/local/cuda/include/nvml.h`, `nvmlDeviceSetGpuLockedClocks`):

> "Sets the clocks that the device will be running at to the value in the **range** of minGpuClockMHz to maxGpuClockMHz. Can be used as a setting to request constant performance."

- A **pair** (`-lgc 1800,3135`) = floor + ceiling; DVFS moves inside it. A **single value** = min=max = hard pin (v2's failure mode: no headroom, boost blocked until the service re-wrote the lock).
- `nvidia-smi -h` (this install): "defines the range of desired locked GPU clock speed in MHz. Setting this will **supersede application clocks and take effect regardless if an app is running**."
- The symbolic mode table in nvml.h admits the floor is not absolute: `unlimited|tdp` → "Upper bound is TDP but **clock may drift below this**". So NVML promises the floor under load but never guarantees post-arbitration idle behavior (grade B). **Empirically on this box the floor held** (2062 MHz at quiet, grade A) — the drift-below-min case did not occur with a display attached and the engine resident.
- Requires root. **Volta+** for GPU clocks; **Ampere+** for `nvmlDeviceSetMemoryLockedClocks` (grade B).
- Values need not be on the supported grid; the driver clamps to it. Use grid values anyway (determinism).
- **Persistence:** device state in the driver. Header: "After system reboot or driver reload GPU clocks go back to their default value." Default = idle clocks. Locks are **not** tied to the writing process's lifetime — a hung governor leaves them armed (safety section 6).
- Verification is behavioral; there is no lock getter in NVML. Check via `-q -d CLOCK` (SM ≥ floor, mem = floor) and the reasons mask (`0x2 ApplicationsClocksSetting` fires when pinned; a pure range at quiet showed only `0x400` — grade A observation).

### 2.2 `-lmc` / memory clocks

Same range semantics ("For Ampere or newer"). Live: floor 7001 held at quiet; busy mclk 13801 (just under 14001 — DVFS trims, do not expect 14001 flat). `-lmcd` (deferred mem lock, applied at next module load) exists but is pointless for a userspace service.

### 2.3 `-rgc` / `-rmc`

"Resets the GPU/memory clocks to the default values." Header: default = idle clocks. Idempotent in practice (harmless when unlocked — v3 relies on this in `apply off`; grade A: the service runs it every release).

### 2.4 `-ac` / `--applications-clocks` — dead on this card

`nvidia-smi -q -d CLOCK` reports literally "Requested functionality has been deprecated" for Applications Clocks (grade A). nvml.h: "@deprecated Applications clocks are deprecated and **will be removed in CUDA 14.0**. Please use nvmlDeviceSetMemoryLockedClocks / nvmlDeviceSetGpuLockedClocks" (grade B). Do not build on `-ac`.

### 2.5 Persistence mode

`-pm 1` keeps the driver initialized so NVML calls do not pay device re-init latency. The unit already `Wants=nvidia-persistenced.service` and v3 re-asserts `-pm 1` on each floor apply. Locks themselves persist independent of persistence mode (they survive until release/reboot; grade B from header semantics). Keep persistenced enabled — it also stabilizes the NVML query path the service polls.

### 2.6 GSP firmware interactions

- On consumer Blackwell the **open kernel module + GSP is the shipping path**; this box runs open 615.71.09 + GSP 615.71.09 (grade A). `NVreg_EnableGpuFirmware=0` is not a viable escape hatch on this generation (grade C; the workaround circulates for older cards — AWS docs still recommend it for vGPU 14.x on G4dn/G5, which are pre-Blackwell).
- **Known crash vector:** open-gpu-kernel-modules issue #1239 — RTX 5070 Ti (GB203, sm_120), open module 610.43.02: a read-only `nvidia-smi -q -d SUPPORTED_CLOCKS` page-faults the GSP task (Xid 120), every RPC then times out (Xid 119), heartbeat stops, machine needs a hard power cycle. Reporter: "does not reproduce on pre-610 drivers" and the **proprietary** module of the same version is unaffected (grade C). This box ran `-q -d SUPPORTED_CLOCKS` twice today without incident (GB202 + 615.71.09-open), so the exact vector does not reproduce here — but the class is real: **never put SUPPORTED_CLOCKS queries in the service runtime loop**; freeze the grid values in `/etc/sysconfig/strata-gpu-lock` instead.
- Related GSP-instability class on 570-610 era: Xid 109 GSP context-switch timeout (#1243, 5060 Ti). Treat GSP as the failure domain to watch in `journalctl -k` (grade C).
- `-svfd` (VF-curve derate for deterministic perf) is "Only supported on Rubin and newer" — not available on GB202 (grade B, `nvidia-smi -h`).

### 2.7 Scope: per-device vs global

`-lgc`/`-lmc`/`-rgc`/`-rmc` take `-i <n>`. **The v3 script and the unit's ExecStopPost omit `-i` → they act on ALL GPUs.** Correct for today's single-GPU box; a footgun the day a second card lands. v4 pins `-i 0` everywhere.

---

## 3. Boost and ramp behavior (mandate 2)

### 3.1 What owned the 1780→3135 bounce (v2's measured failure)

- v2 hard-pinned 1800 = min AND max after 30 s quiet. A hard pin **blocks the boost**: the DVFS may not exceed the lock, so the first burst ran at ~1780 MHz under 94-98 % load until the 3 s poll observed load and re-wrote the lock to MAX. The ramp latency was **service poll latency + lock write latency**, not silicon latency (grade A — script history comment, banked clockwatch proof).
- v3's range lock hands the boost decision to the hardware: DVFS raises pclk inside [floor, ceiling] with no service in the loop. Live dmon during a real burst showed pclk already at 2805 within one 1 s sample window (grade A).

### 3.2 clocks_event_reasons — full bitmask (grade B, local header; grade A live cross-check)

| Bit | Constant | Meaning (header text) |
|---|---|---|
| 0x0000000000000001 | GpuIdle | GPU is idle |
| 0x0000000000000002 | ApplicationsClocksSetting | clocks set to application/user-defined values (fires on hard pins) |
| 0x0000000000000004 | SwPowerCap | software power cap activated |
| 0x0000000000000008 | HwSlowdown | HW slowdown (parent of the two below) |
| 0x0000000000000010 | SyncBoost | sync boost activated |
| 0x0000000000000020 | SwThermalSlowdown | software thermal slowdown |
| 0x0000000000000040 | HwThermalSlowdown | HW thermal slowdown |
| 0x0000000000000080 | HwPowerBrakeSlowdown | HW power brake |
| 0x0000000000000100 | DisplayClockSetting | display clock setting limited |
| 0x0000000000000200 | BoardLimit | board limit (operating) policy |
| **0x0000000000000400** | **Reliability** | **"the reliability policy is currently limiting the GPU clocks"** |

- **The bounce the operator attributed to "idle flags 0x400" was mislabeled.** 0x400 = Reliability = the V/F reliability curve trimming the top boost bins for voltage margin. It read `Active` while quiet at the floor (2062 MHz) and is expected to stay active whenever clocks sit in the elevated bins. Benign; not a fault; not idle (grade A + B).
- Which reason dominated the v2 bounce: during the stalled ramp the governors' active reason is **GpuIdle (0x1)** while the card decays toward idle, then **ApplicationsClocksSetting (0x2)** while the hard pin holds, then **SwPowerCap (0x4) + Reliability (0x400)** at the top of the boost. Cumulative SW Power Capping counter on the box: 3.67 s lifetime — small; power capping is not the dominant regime (grade A).
- Read reasons programmatically: `nvmlDeviceGetCurrentClocksEventReasons` (µs path; probe returned `rc=0 mask=0x400` live). In `nvidia-smi -q -d PERFORMANCE` the same data appears as named Active flags.

### 3.3 Sustained busy clocks are power-capped below the ceiling

Live decode: pclk 2805-2850 (not 3135), pviol=1, 276-387 W. The last ~280 MHz of the V/F curve is unreachable under a 575 W cap during real decode. Consequence: **floor choice matters more than ceiling choice**; the ceiling is decorative above ~2850 (grade A).

---

## 4. Busy/quiet detection (mandate 3)

Ranked by latency and cost (grades: A for measured costs, B for mechanism):

| Rank | Method | Cost | Granularity | Notes |
|---|---|---|---|---|
| 1 | **NVML** (ctypes/pynvml) | 0.4-1.7 µs | ~instant sample; utilization is driver-smoothed, expect ~1 s effective lag on the util number itself | The low-latency path. Not installed as pynvml; ctypes works today |
| 2 | **Engine port probe** (`ss` on :8080/:8081) | 3.75 ms | event-adjacent — the engine itself is the signal | Current v3 contract; immune to cmdline matches; zero GPU queries |
| 3 | `nvidia-smi --query-gpu=` | 23 ms | per-spawn | Fine at 3 s cadence; 0.8 % of one core |
| 4 | `nvidia-smi dmon -s pucvmet` | 1 Hz stream | 1 s | Best single view (pwr, sm/mem util, mclk/pclk, pviol, tviol); a feed, not a probe |
| 5 | DCGM | its own poller | configurable | Adds a service for nothing the port probe + NVML doesn't give on one card |

**There is no event-driven GPU-load hook in the stock stack** (grade B): NVML exposes no busy callback; systemd has no GPU load unit trigger; D-Bus exposes no NVIDIA utilization signal. "Event-driven" on this box means **the engine's own listen state** — which is exactly the v3 port contract, and it is the right contract: the boost path no longer depends on detection latency at all, so the poll cadence only gates release timing and the thermal guard. Keep 3 s (poll), 45 s (release grace). Sub-second detection buys nothing under this architecture.

Production cadence reference: benchmark harnesses lock clocks and poll at 1 Hz (dmon default); DCGM fleet default is 10 s; per-card latency-tuned governors that still poll sit at 0.2-1 s. All are slower than this box's boost path now that the hardware owns it (grade C for the external cadences, D for exact fleet defaults — unverifiable here).

---

## 5. Production practice + public 5090 data (mandate 4)

### 5.1 Clock pinning in production/benchmark stacks

- **NVIDIA/TensorRT-LLM** (in-repo troubleshooting reference): "**Lock GPU clocks: `nvidia-smi -lgc <freq>` for reproducible benchmarks**" — grade C, fetched 2026-11-07 (`.claude/skills/kernel-cute-writing/references/troubleshooting.md` on main).
- **arXiv 2505.11329** (GPU benchmarking methodology): "We also use fixed TDP settings to ensure stable performance measurements… The CLI tool nvidia-smi can be used to lock the GPU clocks" — grade C.
- **AWS EC2**: the EC2 GPU guidance page (docs.aws.amazon.com/AWSEC2/latest/UserGuide/optimize_gpu.html) directs users to set GPU clocks for consistent performance on accelerated instances (the page section retrieved today returned the GRID driver install flow; the clock-setting guidance is the page's stated purpose — grade C/Medium). AWS does **not** pre-pin clocks on your instance; NVIDIA datacenter instances expect the operator to pin (grade C).
- **vLLM and SGLang code search for `nvidia-smi -lgc`: zero hits** in both repos today (grade A — GitHub code search ran; negative result). Pinning is a benchmark/deployment-ops practice, not baked into those engines.
- Cloud posture in one line: nothing pins clocks for you; the governor-on-the-box pattern (this service) is the standard remedy.

### 5.2 Public 5090 numbers (grade C — Wikipedia GeForce 50 series table)

21760 CUDA cores; boost 2.41 GHz (NVIDIA spec — the driver's own clock table reaches 3135, sustained busy lands ~2850 under the cap); 32 GB GDDR7; 28 Gbps/pin; 512-bit; **1792 GB/s**; TDP **575 W**.

### 5.3 Decode is DRAM-bandwidth-bound — quantified

- Roofline: single-stream decode tok/s ≤ BW / bytes-per-token-weights. Examples at full 1792 GB/s: a 16 B dense model at Q8 (~16 GB/token) ceilings ~112 tok/s; a 30 B at Q8 ~60 tok/s. At the 7001 floor the memory domain offers ~896 GB/s — still above quiet demand, but half the burst ceiling (grade A arithmetic on C data).
- The v2 measurement (uniform ~30 % decode loss at SM 1780 under sustained-MAX baseline) decomposes as: SM clock cut 2850→1780 is 1.6x; a 30 % throughput loss implies roughly half of decode time is SM-sensitive (attention at context, sampler, launch gaps) and half mem-bound. Both domains lost in v2 (SM pinned low AND mem pinned at 7001). v3/v4 floors restore both instantly (grade B inference from A measurements).
- **Memory clocks matter more than SM clocks for pure decode** — but the floor-range design protects both, so the trade is moot; keep `-lmc 7001,14001`. GDDR7 P-state retiming latency on burst entry is not publicly documented (grade D); the mem floor sidesteps the question entirely — the observed quiet mem sits at 7001 and busy at 13801 with no observable retrain stall.

### 5.4 Thermal/reliability envelope

- Observed: 33 C at quiet, 50-51 C during 387 W decode bursts; effective T.Limit 57 C (current reading), slowdown/shutdown spec offsets -2/-5 C; huge headroom (grade A).
- Sustained max is **by design within envelope**: 575 W SW cap first (pviol counter proves it fires), then SW thermal slowdown, then HW thermal slowdown, then HW power brake — the driver's ladder of backstops (grade B header semantics + C NVML docs). No consumer-card sustained-max derating data specific to GB202 exists publicly; the Blackwell Pro 6000 "MHz degrading" thread is a PMIC sensor defect class, not a clock-lock issue (grade D). Watch Xid in `journalctl -k`, not folklore.

---

## 6. Safety: failure modes of the wrong design (mandate 5)

1. **Locked clocks + hung governor — who releases?** Locks are driver device state; they survive the writer's death and any process exit; only `-rgc`/`-rmc`, reboot, or driver reload clears them (grade B). Defense in depth (v4):
   - layer 1: `trap TERM INT → apply off` inside the daemon (v3 has it);
   - layer 2: `ExecStopPost=` `-rgc` + `-rmc` in the unit (v3 unit has it — fires on ANY service exit including `systemctl kill`);
   - layer 3 (v4 delta): `ExecStartPre=` release, so a crashed predecessor's stale locks are cleared at start;
   - layer 4 (v4 delta): independent dead-man `strata-gpu-lock-deadman.timer` — a 60 s timer whose service checks the daemon's heartbeat file mtime; stale > 90 s → run the release and `systemctl try-restart` the governor. This survives a daemon wedged in a way that skips traps.
2. **Thermal runaway via clock lock: impossible by construction.** The lock range never raises limits — 575 W SW cap, thermal ladder, and power brake are driver-side and upstream of any NVML lock (grade B). v3's shed at 83 C / resume at 78 C adds margin well before any HW backstop; keep the 5 C hysteresis to prevent flapping.
3. **The real residual risks** (all grade C/D, watched not feared): GSP firmware RPC hangs (section 2.6 — keep SUPPORTED_CLOCKS out of the runtime loop; alert on Xid 119/120/109); mem floor raising idle draw slightly (accepted, measured +6-18 W); a second GPU appearing and receiving global-scope locks (fixed by `-i 0` in v4).
4. **Do not** chase determinism with a single-value lock on a serving box — that is v2, measured, regressions banked. A full pin is for benchmarking only (TensorRT-LLM guidance context, grade C).

---

## 7. RECOMMENDED v4 design (implement from this)

The architecture stands (floor-range while the engine listens; released when it does not; hardware owns the boost). v4 is v3 plus scoping, idempotent release, dead-man backup, and a reasons guardrail.

### 7.1 Exact nvidia-smi calls

```bash
# FLOOR (engine listening) — idempotent, safe to re-apply every poll
nvidia-smi -i 0 -pm 1
nvidia-smi -i 0 -lgc 1800,3135      # SM floor+ceiling; DVFS moves inside; boost is HW-owned
nvidia-smi -i 0 -lmc 7001,14001     # mem floor: 896 GB/s instant on burst entry

# RELEASED (no engine for GRACE, or thermal shed)
nvidia-smi -i 0 -rgc
nvidia-smi -i 0 -rmc

# VERIFY (behavioral; no lock getter exists)
nvidia-smi -i 0 -q -d CLOCK,PERFORMANCE     # quiet: SM>=floor, mem=7001, reasons ~0x400
nvidia-smi -i 0 dmon -s pucvmet -c 5        # busy: pclk ~2805-2850, mclk ~13801, pviol<=1

# TELEMETRY per poll (one batched call, 23 ms, replaces the two separate calls in v3)
nvidia-smi -i 0 --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.active \
            --format=csv,noheader,nounits
```

Never in the runtime loop: `-q -d SUPPORTED_CLOCKS` (GSP crash vector, section 2.6), `-ac` (deprecated, removal CUDA 14.0), single-value `-lgc` (v2's failure), anything without `-i 0`.

### 7.2 Config (`/etc/sysconfig/strata-gpu-lock`)

```bash
SM_FLOOR=1800      # start here; A/B test 2100 (grade A: floor>1800 only buys burst-entry margin
                   # the range lock already covers — raise only if clockwatch shows SM below 2062 at burst start)
SM_MAX=3135        # decorative above ~2850 under the 575 W cap; keep as ceiling
MEM_FLOOR=7001
MEM_MAX=14001
TEMP_SHED=83
TEMP_RESUME=78
PORTS="8080 8081"
GRACE=45           # release hysteresis: no engine listening for this long
POLL=3             # gates release + thermal only — NOT the boost path
HEARTBEAT=/run/strata-gpu-lock.heartbeat
```

### 7.3 State machine

```
            engine listening on PORTS
  RELEASED ────────────────────────────> FLOOR
  (-rgc/-rmc)        apply floor            (-lgc 1800,3135 + -lmc 7001,14001)
     ^  │            within one poll           │
     │  │ no engine for GRACE=45s              │ temp >= 83 C
     │  └──────────────────────────────────────┘ (either state)
     │                        SHEDDED: release, resume when <= 78 C (5 C hysteresis)
     │
     └── any release: -rgc + -rmc + heartbeat unlink
```

- **RELEASED**: DVFS owns; deep idle ~35-48 W.
- **FLOOR**: quiet → P3, SM ~2062 (floor + DVFS margin), mem 7001, ~54 W; busy → pclk 2805-2850, mclk 13801, under the 575 W cap. Transition on engine-presence is immediate (idempotent re-apply); the boost itself is hardware-latency, service-independent.
- **SHEDDED** (thermal sub-state): release, resume below 78 C.
- Guardrail in the poll loop: log `clocks_event_reasons` on change. `0x400` at quiet and `0x4|0x400` under load are normal (grade A). `0x8/0x40/0x80` (HW slowdown family) → shed the floor and log loud — that is the driver telling you the silicon disagrees with your policy.

### 7.4 systemd shape

`Type=simple` is correct for the bash daemon (no `sd_notify` from bash; do not fake it). Keep the daemon; add the deltas:

```ini
# /usr/lib/systemd/system/strata-gpu-lock.service (v4 deltas marked)
[Unit]
Description=Strata GPU clock governor (floor-range while a Strata engine listens)
Documentation=man:nvidia-smi(1)
After=multi-user.target
Wants=nvidia-persistenced.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/nvidia-smi -i 0 -rgc        # v4: clear stale locks from a crashed predecessor
ExecStartPre=-/usr/bin/nvidia-smi -i 0 -rmc        # v4
ExecStart=/usr/libexec/strata-gpu-lock
ExecReload=/bin/kill -HUP $MAINPID
ExecStopPost=-/usr/bin/nvidia-smi -i 0 -rgc        # v4: add -i 0
ExecStopPost=-/usr/bin/nvidia-smi -i 0 -rmc        # v4: add -i 0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Independent dead-man (v4, new pair — covers a wedged daemon that skips its trap):

```ini
# /usr/lib/systemd/system/strata-gpu-lock-deadman.service
[Service]
Type=oneshot
ExecStart=/usr/libexec/strata-gpu-lock-deadman
# deadman script: if HEARTBEAT mtime older than 90s -> nvidia-smi -i 0 -rgc -rmc
#                 -> logger -t strata-gpu-lock "deadman released clocks"
#                 -> systemctl try-restart strata-gpu-lock

# /usr/lib/systemd/system/strata-gpu-lock-deadman.timer
[Timer]
OnUnitActiveSec=60s
[Install]
WantedBy=timers.target
```

The daemon touches `HEARTBEAT` every poll. Optional later upgrade: port the daemon to python + `nvidia-ml-py` + `sd_notify` for `Type=notify` and `WatchdogSec=30` — worth it only if the dead-man path ever fires in practice.

### 7.5 Acceptance protocol for the operator (all read-only or already-reversible)

1. Enable v4; quiet check: `-q -d CLOCK,PERFORMANCE` → P3, SM ≥ 1800, mem 7001, ~54 W, reasons `0x400`.
2. Busy check: fire a decode burst after a 60 s gap; `dmon -s pucvmet` in parallel → first sample already pclk ≥ 2600, mclk ≥ 13800; no 1780-style plateau. Bank the burst TTFT/tps against the v2 banks — expect the ~30 % inflation gone.
3. Idle A/B: `systemctl stop strata-gpu-lock` → wait 5 min → power draw (expect 35-48 W) → `systemctl start`. Delta is the floor's price.
4. Kill test: `systemctl kill -s SIGKILL strata-gpu-lock` → confirm locks cleared (unit ExecStopPost runs even on SIGKILL) → deadman timer armed → `systemctl start`.
5. `journalctl -k | grep -i xid` nightly for the GSP class (119/120/109).

### 7.6 Open questions (probes, not blockers)

- Floor 1800 → 2100: only if clockwatch shows burst-start SM under ~2062. The range lock already floats quiet clocks to ~2062; the floor mostly protects against arbiter edge cases.
- GSP RPC stability on 615.x-open GB202 across driver upgrades: re-run acceptance step 4 after every driver bump.
- pynvml/ctypes port: 0.4 µs telemetry is a luxury, not a need — the boost path no longer depends on it.

---

## Provenance index

- Local, driver-matched: `/usr/local/cuda/include/nvml.h` (clocks_event_reasons constants, Set/ResetGpuLockedClocks + SetMemoryLockedClocks doc comments, applications-clocks deprecation, utilization API); `nvidia-smi -h` (option semantics); `/proc/driver/nvidia/version`; `/usr/lib/systemd/system/strata-gpu-lock.service`; `/usr/libexec/strata-gpu-lock`.
- On-box telemetry 2026-11-07: `nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE,SUPPORTED_CLOCKS`, `nvidia-smi dmon -c 2`, ctypes NVML probe (latency + live reasons mask 0x400), poll-cost benchmarks.
- Web: open-gpu-kernel-modules issues 1239 (GB203 SUPPORTED_CLOCKS GSP crash) and 1243 (Xid 109) — github.com/NVIDIA/open-gpu-kernel-modules; Wikipedia "GeForce 50 series" (5090 specs: 1792 GB/s, 575 W); AWS EC2 GPU guidance — docs.aws.amazon.com/AWSEC2/latest/UserGuide/optimize_gpu.html; arXiv 2505.11329 (fixed TDP/clocks benchmark methodology); NVIDIA/TensorRT-LLM in-repo troubleshooting reference ("Lock GPU clocks: nvidia-smi -lgc"); negative code-search results in vllm-project/vllm and sgl-project/sglang.
