
### F-123 — THE DEGRADATION, BOUNDED: sustained traffic, concurrency and long prompts are all EXONERATED; the shared launcher now diagnoses the next episode itself (2026-10-12 ~17:40)

F-119 found the process degrading ~10x over ~30 minutes (10.3 t/s aged vs 101.6 fresh). This round tried
to reproduce it under instrumentation and FAILED to - which bounds it, and leaves the diagnosis in place.

**1. THE STATE COUNTERS ARE EXONERATED (the trap the Proof Law caught).** ms/window was correlated
against every counter that grows inside a boot, on the already-captured boots:

| counter | good boot 01:11 (17.8 ms/win) | degraded boot 16:29 (73.7 ms/win) | verdict |
|---|---|---|---|
| adaptive exchanges | **55,260** | 29,526 | the GOOD boot had ~2x more - not the cause |
| conversation-cache evictions | **349** | 111 | same |
| KV index lookups per request | **3.5-5.9M** | 0.4-1.5M | same |
| Windows commit free | 103-105 GiB | 103-105 GiB | no pressure either way |
| GPU throttle / CPU load | none / 13% total | none | not the machine |

Pearson r = 0.82 for exchanges looked like a cause and was **correlation with time, not mechanism**.

**2. THE REPRODUCTION ATTEMPT (instrumented, one fresh boot, 17:17:30).**

| what was run | result |
|---|---|
| 210 sustained sequential requests over 5.7 min | **no degradation** - it improved: wait 9.34 -> 8.96 ms/window, 145 -> 167 t/s |
| 1x 2-conc burst + 2x 6-conc bursts (2,400 tok each) | 6-conc **176.5 / 176.8 t/s aggregate**; single stream AFTER: 152-163 t/s (up, not down) |
| a 54,819-token fresh prompt | following windows 12.9-13.7 ms (vs 12.8-15.3 before) - no effect |
| 25,544 cumulative windows with mixed traffic | per window: **wait 9.100, pool + plan 1.224, host staging 0.201, commit 0.085 ms** - flat to the third decimal |

The per-request decode breakdown on that boot (a 200-token request, 5.7 min in):
`11.58 ms/window = verify 10.39 (GPU-reach wait 8.49 + per-layer host 1.03 [plan 0.05, CPU 0.87] + stage
0.02) + commit/emit 0.06 + draft 1.07; per layer-window: CPU experts 0.20, VRAM hits 18.69, PCIe 0.01.`

**3. WHAT THIS MEANS.** On a quiet machine the engine is stable: ~86% of each window is the GPU doing
its work (wait 9.1 of 10.6 ms) and nothing grows. Every internal candidate - the adaptive tier, the
conversation cache, the KV index, the draft chain, memory pressure, the host staging/commit path - is
now refuted with numbers. **What remains is outside the process:** F-119's own observation that the GPU
sat at ~10% during a slow read points at the GPU being time-shared with another consumer (this box runs
Chrome and two Electron agent GUIs; a WDDM time-slice stretches every kernel without changing a single
one of the engine's counters), or a request shape not yet exercised (thinking-mode/tool-call traffic,
cancellation, streaming).

**4. THE DIAGNOSIS IS NOW SELF-CAPTURING.** `launch-v2-battery-8081.bat` carries
`STRATA_SPLIT_TIMING=1` + `STRATA_DECODE_TIMING=1` (temporary; remove once the stage is named). Because
the instrument lives in the SHARED launcher, whoever boots the lane next captures its own diagnosis: if
a slow episode shows **wait for the GPU** up, the GPU is being shared; if **pool + plan** or **host
staging** up, it is the host path. One line read, no experiment.

**5. ALSO RECORDED THIS ROUND:** the matched-window A/B stands (standing recipe 145.8 vs v1-arms 122.3 t/s
M1); the fresh-boot health class measured repeatedly today is 136-167 t/s single-stream with 6-conc
176 t/s aggregate - the best concurrent number on this build since the campaign's 196-205 class.
The lane was rebooted three times by the parallel session during this round (17:15, 17:16, 17:19), which
is what killed the first watch at 4 samples - recorded so the lost samples are not mistaken for data.
