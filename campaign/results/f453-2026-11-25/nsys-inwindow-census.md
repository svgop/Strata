# F-453 — in-window census by nsys (2026-11-25)

Read-only analysis of the existing trace `f452-dev` (device graph launch ON, node-level graph trace). No GPU workload was run, no engine was built or started, nothing was deleted. The only SQL work is over `/tmp/f452-dev.sqlite`.

## Source, method, checks

- **Source.** `/tmp/f452-dev.sqlite`: nsys 2026.3.2 export of `/tmp/f452-dev.nsys-rep` (`CUDA_GRAPH_TRACE_OPTIONS:MODE = Node`, `TRACE_DEVICE_GRAPH_LAUNCH = true`), RTX 5090, 24.42 s capture, 1,579,267 kernel records, 32,408 memcpy records, 213 memsets.
- **Script.** `/tmp/f453-census.py` (python3 + sqlite3; opens the database with `mode=ro`, writes only `/tmp/f453/out.md`). `/tmp/f453-report.py` copies every table and every number of this file from that output.
- **Instance identification (method used: parent launch).** The trace holds 772 executions of the one-node parent graph (kernel `window_dev_launch_kernel`, median 2.8 us). Each is followed by exactly one execution of the window graph; every window kernel was attached to the latest parent launch at or before its start. Checks: 772 parent launches, 772 instances, 0 instances overlapping, 0 kernels of another graph inside a window interval (no drafter kernel ever falls inside a window graph).
- **Kernel records per instance.** Six window graphs exist, one per token count T (T is the `n_tok` template argument of the hyper-connection read `gr_down_v3_kernel<1,T,...>`):

| T | graphId | kernel records per instance | executions (all / warm) | share of warm windows |
|---|---|---|---|---|
| 1 | 4 | 1955 (1956 on its first launch only) | 296 / 33 | 11% |
| 2 | 38 | 1873 | 341 / 163 | 56% |
| 3 | 30 | 1885 | 64 / 44 | 15% |
| 4 | 46 | 1932 (1933 on its first launch only) | 13 / 8 | 3% |
| 5 | 65 | 1940 (1941 on its first launch only) | 19 / 9 | 3% |
| 6 | 57 | 1949 | 39 / 36 | 12% |

  The expected ~1949 is the T=6 graph alone; the warm mean is 1897.0 kernel records per window (min 1873, max 1955). In graphs 4, 46 and 65 one node per execution has no kernel record after the first launch (graph 4: the `gate` after `decode_attn_merge_kernel`; graph 46: the `gr_down_v3_kernel` after `cvec_kernel`; graph 65: a `native_mmvq_multi_kernel<Q4KTraits,...>` on a side stream). The node runs: its duration appears as a gap between its neighbours (graph 46: 7.46 us between `cvec_kernel` and `gr_up_v3_kernel`, against 7.3 us for the recorded first launch). Its time is therefore counted as idle in this report whenever no other stream covers it (0.0005 ms per window, section 5).
- **Warm windows.** Last 38% of the 772 executions = floor(0.38 x 772) = **293 windows** (indices 479..771). Warm T mix: T1=33, T2=163, T3=44, T4=8, T5=9, T6=36, i.e. 2.676 tokens per window.
- **Active / idle.** Active = union of [start,end) over all kernels on the 3 streams; idle = wall minus that union. An independent SQL window-function pass over the 293 warm windows reproduces the script exactly (union 9.978 ms, wall 10.203 ms, idle 225.0 us per window).
- **Summed versus union.** "Summed" adds the three streams and therefore counts overlapped time up to three times (1.821 ms per window). Every table states which it uses.
- **Memcpys** are copy-engine activity, not kernels; they are counted only where a line says so.

## Numbers first (293 warm windows, means; medians in parentheses)

| quantity | per window |
|---|---|
| window-to-window period (first kernel to first kernel) | **12.163 ms** (10.772 ms) |
| = window-graph wall (first to last kernel) | **10.203 ms** (8.929 ms) |
| — GPU idle inside the window graph (no kernel on any stream) | **0.225 ms** (0.186 ms) |
| — wall on which only handshake/spin-named kernels run | **1.895 ms** (0.944 ms) |
| + drafter phase (previous window end to parent launch start) | **1.956 ms** (1.711 ms) |
| — of which drafter kernels active (union) | 1.084 ms (1.199 ms) |
| — of which idle, memcpys counted as activity | 0.702 ms (0.371 ms) |
| boundary: last drafter kernel to first window kernel | **22.5 us median**, 414.3 us mean |
| throughput implied by the trace (T / period) | 220.0 tokens/s |

The "~12-13 ms window" of the brief is the window-to-window period here (12.163 ms mean, 10.772 ms median). The window graph alone is 8.929 ms median. Inside it kernels cover 97.6% of the wall; idle is 0.225 ms. The drafter's step graphs measure 258.7 us-262.1 us (steps 2-5) and 287.7 us-333.1 us (step 1), consistent with the ~265 us in the brief.

## 1. Segmentation

Medians and means per warm window; units as marked.

| quantity (per warm window) | median | mean | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| A window-graph duration, first to last kernel | 8.929 ms | 10.203 ms | 7.809 ms | 15.286 ms | 6.881 ms | 28.465 ms |
| B drafter phase, end of previous window graph to parent launch start | 1.711 ms | 1.956 ms | 0.774 ms | 3.391 ms | 0.369 ms | 7.203 ms |
| C window-to-window period | 10.772 ms | 12.163 ms | 8.994 ms | 17.310 ms | 7.307 ms | 35.675 ms |
| D gap, last drafter kernel end to first window kernel | 22.5 us | 414.3 us | 12.2 us | 1145.1 us | 7.7 us | 5179.8 us |

A + B = 12.163 ms mean, equal to C. Over all 771 periods (cold and warm) the period is 9.378 ms median, 10.417 ms mean.

**Boundary gap D decomposed (per warm window):** last drafter kernel to parent launch start: median 16.1 us, mean 407.9 us (p10 5.8 us, p90 1138.5 us); parent kernel: median 2.8 us, mean 2.8 us; parent kernel end to first window kernel (device-launch latency): median 3.5 us, mean 3.5 us, p90 3.6 us. The typical boundary is 16 us of host time + 2.8 us parent kernel + 3.5 us launch latency.

**Typical versus slow boundaries** (same 1.4 ms cut as nsys-anatomy.md): typical, D < 1.4 ms: n = 265, median 22.2 us. Slow, D >= 1.4 ms: n = 28, median 3559 us (min 1466 us, max 5180 us); in these 28 windows the drafter phase is 4.71 ms median, 4.74 ms mean. Over all 771 boundaries (cold + warm) the median is the same 22.5 us (mean 409.5 us); 728 of 771 are below 1.4 ms. The brief's ~31 us median is the same order; the definitions differ (here: last drafter *kernel*, which can be later than the last drafter *graph* end).

**Drafter phase B decomposed (median | mean | p10 | p90):**
- previous window graph end to first drafter kernel: 34.6 us | 120.2 us | 17.1 us | 253.5 us
- first to last drafter kernel: 1522.7 us | 1427.9 us | 600.6 us | 2405.6 us
- last drafter kernel to parent launch start: 16.1 us | 407.9 us | 5.8 us | 1138.5 us

The last device activity before the parent launch is `mtp_select_kernel` 147, `H2D memcpy` 108, `res_patch_kernel` 38 (count of warm windows).

**Per T:**

| T | graphId | kernels | warm n | A wall median / mean (ms) | B drafter phase median / mean (ms) | D boundary median / mean (us) | tokens/s = T / (wall + drafter phase) |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 1955 | 33 | 8.31 / 8.32 | 0.75 / 1.47 | 22.3 / 463.0 | 102.1 |
| 2 | 38 | 1873 | 163 | 8.40 / 8.73 | 1.74 / 1.84 | 22.1 / 302.9 | 189.1 |
| 3 | 30 | 1885 | 44 | 9.27 / 10.24 | 1.32 / 1.86 | 23.4 / 460.7 | 248.1 |
| 4 | 46 | 1932 | 8 | 12.21 / 12.79 | 1.57 / 1.84 | 27.4 / 236.8 | 273.4 |
| 5 | 65 | 1940 | 9 | 15.67 / 15.50 | 1.83 / 2.30 | 22.4 / 131.3 | 280.9 |
| 6 | 57 | 1949 | 36 | 16.27 / 16.65 | 1.99 / 2.97 | 23.6 / 927.2 | 305.8 |

## 2. Inside the window graph (union timeline, 3 streams)

| per warm window | median | mean | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| kernel records (count) | 1873 | 1897.0 | | | 1873 | 1955 |
| wall, first to last kernel | 8.929 ms | 10.203 ms | 7.809 ms | 15.286 ms | 6.881 ms | 28.465 ms |
| ACTIVE, union over 3 streams | 8.733 ms | **9.978 ms** | 7.607 ms | 15.067 ms | 6.413 ms | 28.254 ms |
| IDLE | 0.186 ms | **0.225 ms** | 0.180 ms | 0.381 ms | 0.175 ms | 0.691 ms |
| IDLE share of wall | 2.14% | 2.37% | | | | 9.28% |

Idle with memcpys counted as activity: median 0.185 ms, mean 0.224 ms (no measurable change). No memset falls inside a warm window (0 records).

**Per-stream active and overlap (ms per warm window):**

| | median | mean | p10 | p90 |
|---|---|---|---|---|
| main stream (1238 kernels/window) | 6.641 ms | 7.314 ms | 5.598 ms | 10.650 ms |
| side stream A (lower id; 329 kernels/window) | 1.939 ms | 2.266 ms | 1.578 ms | 3.233 ms |
| side stream B (higher id; 329 kernels/window) | 1.997 ms | 2.220 ms | 1.589 ms | 3.270 ms |
| sum of per-stream active | 10.493 ms | 11.799 ms | 9.338 ms | 17.329 ms |
| **overlap = sum minus union** | **1.745 ms** | **1.821 ms** | 1.541 ms | 2.222 ms |

The overlap is 19.4% of the union (mean). By concurrency: exactly 1 stream busy 8.226 ms mean (7.028 ms median), exactly 2 streams 1.683 ms (1.607 ms), all 3 streams 60 us (69 us).

**Main-stream view (additive):** wall 10.203 ms = main-stream kernels 7.314 ms + main idle while a side stream runs 2.664 ms (median 2.079 ms) + GPU idle 0.225 ms. The main stream is idle 2.889 ms in total (median 2.324 ms).

**Memcpys inside the window interval (copy engine, not kernels):** 5.33 per window, 12.85 MiB per window, 388.8 us of copy time per window. Of these, 5.22 per window are H2D of at least 1 MiB (median 2.08 MiB, median 61.0 us each, implied 35.5 GB/s, all on one stream, id 19); 33 are in-graph D2D nodes of 40 KiB (the T=1 graph).

**Per T (medians except idle % and counts):**

| T | graph | kernels | warm n | wall | active | idle | idle mean | idle % of wall (mean) | main active | side A | side B | overlap | 1 stream | 2 streams | 3 streams | main idle | of which side stream running |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 1955 | 33 | 8.31 | 7.93 | 0.22 | 0.27 | 3.3 | 5.17 | 2.09 | 2.05 | 1.24 | 6.76 | 1.13 | 56 | 3.30 | 3.08 |
| 2 | 38 | 1873 | 163 | 8.40 | 8.18 | 0.18 | 0.22 | 2.5 | 6.42 | 1.81 | 1.85 | 1.72 | 6.37 | 1.60 | 59 | 2.03 | 1.82 |
| 3 | 30 | 1885 | 44 | 9.27 | 9.06 | 0.18 | 0.21 | 2.1 | 6.88 | 2.10 | 2.09 | 1.76 | 7.20 | 1.60 | 80 | 2.41 | 2.23 |
| 4 | 46 | 1932 | 8 | 12.21 | 11.91 | 0.21 | 0.27 | 2.2 | 8.68 | 2.52 | 2.36 | 1.92 | 10.15 | 1.84 | 39 | 3.39 | 3.09 |
| 5 | 65 | 1940 | 9 | 15.67 | 15.47 | 0.20 | 0.20 | 1.4 | 10.99 | 3.17 | 2.79 | 1.87 | 13.65 | 1.78 | 45 | 4.27 | 4.07 |
| 6 | 57 | 1949 | 36 | 16.27 | 16.07 | 0.20 | 0.24 | 1.5 | 11.15 | 3.35 | 3.34 | 2.19 | 13.95 | 1.98 | 104 | 4.60 | 4.39 |

Column units: wall, active, idle, main active, side A, side B, overlap, 1 stream, 2 streams, main idle and its side-stream part in ms; "3 streams" in us; kernels and warm n are counts.

## 3. Kernels inside warm window graphs

138 distinct kernel variants (demangled name with template arguments) run in the warm windows; their summed duration over the 3 streams is 11.80 ms per window.

### Top 40 by summed duration

| # | kernel (shortened demangled name) | n/window | mean us | median us | p90 us | ms/window | % of summed kernel duration | stream |
|---|---|---|---|---|---|---|---|---|
| 1 | `dm_wait_combine_kernel` | 42.6 | 25.72 | 2.37 | 78.24 | 1.10 | 9.3 | main |
| 2 | `wait_flag_ge_planned_kernel` | 96.0 | 6.55 | 0.86 | 5.41 | 0.63 | 5.3 | sides |
| 3 | `native_down_multi_kernel<20,1>` | 78.0 | 7.24 | 5.66 | 14.43 | 0.57 | 4.8 | sides |
| 4 | `native_mmvq_il_kernel<IlQ6K,2,1>` | 15.6 | 35.45 | 21.15 | 23.14 | 0.55 | 4.7 | main 4% |
| 5 | `gr_up_v3_kernel<1,2,1,1>` | 54.0 | 9.07 | 8.32 | 8.74 | 0.49 | 4.1 | main |
| 6 | `native_mmvq_multi_kernel<Q6KTraits,2,4,2,0,0>` | 48.4 | 9.38 | 8.70 | 13.15 | 0.45 | 3.8 | main 76% |
| 7 | `native_gu_swiglu_kernel<22,1>` | 40.0 | 10.28 | 8.16 | 21.76 | 0.41 | 3.5 | sides |
| 8 | `gr_down_v3_kernel<1,2,1,1,1,0>` | 54.0 | 7.18 | 6.82 | 7.30 | 0.39 | 3.3 | main |
| 9 | `native_gu_swiglu_kernel<18,0>` | 34.0 | 11.24 | 9.60 | 24.70 | 0.38 | 3.2 | sides |
| 10 | `route_resident_k` | 48.0 | 6.28 | 6.94 | 8.70 | 0.30 | 2.6 | main |
| 11 | `native_gu_swiglu_kernel<21,0>` | 20.0 | 12.69 | 9.63 | 25.86 | 0.25 | 2.2 | sides |
| 12 | `bf16_f32_mmvf_multi_kernel<256,2,1>` | 67.9 | 3.54 | 3.26 | 4.03 | 0.24 | 2.0 | main |
| 13 | `native_mmvq_il_kernel<IlQ6K,6,2>` | 10.1 | 23.58 | 17.06 | 20.03 | 0.24 | 2.0 | main 50% |
| 14 | `block_topk_cluster_kernel` | 12.0 | 16.92 | 15.78 | 16.26 | 0.20 | 1.7 | main |
| 15 | `gdn_step_norm_multi_kernel<1,0,0>` | 31.9 | 5.92 | 4.80 | 8.42 | 0.19 | 1.6 | main |
| 16 | `gdn_ab_multi_kernel<2,1>` | 20.0 | 9.33 | 8.22 | 10.30 | 0.19 | 1.6 | main |
| 17 | `cvec_kernel` | 48.0 | 3.67 | 3.17 | 3.36 | 0.18 | 1.5 | main |
| 18 | `native_q6_k_mmvq_kernel<0>` | 14.5 | 12.10 | 9.66 | 15.71 | 0.18 | 1.5 | main 52% |
| 19 | `gr_up_v3_kernel<1,6,1,1>` | 11.9 | 14.27 | 12.96 | 13.44 | 0.17 | 1.4 | main |
| 20 | `decode_attn_i8_split_kernel` | 12.0 | 13.32 | 12.16 | 13.95 | 0.16 | 1.4 | main |
| 21 | `gr_up_v3_kernel<1,3,1,1>` | 14.6 | 10.12 | 9.57 | 9.98 | 0.15 | 1.2 | main |
| 22 | `copy_i32_unless_kernel` | 48.0 | 3.05 | 1.09 | 5.06 | 0.15 | 1.2 | sides |
| 23 | `doorbell_publish_res_kernel` | 48.0 | 3.00 | 1.82 | 5.76 | 0.14 | 1.2 | main |
| 24 | `gr_down_v3_kernel<1,6,1,2,0,1>` | 11.9 | 11.47 | 11.15 | 11.87 | 0.14 | 1.2 | main |
| 25 | `native_mmvq_il_kernel<IlQ5K,2,1>` | 9.5 | 13.82 | 13.63 | 15.46 | 0.13 | 1.1 | main 24% |
| 26 | `native_mmvq_multi_kernel<Q6KTraits,3,4,2,0,0>` | 13.1 | 9.70 | 7.52 | 13.12 | 0.13 | 1.1 | main 76% |
| 27 | `route_multi` | 42.6 | 2.96 | 2.69 | 2.98 | 0.13 | 1.1 | main |
| 28 | `native_mmvq_il_kernel<IlQ6K,2,2>` | 7.8 | 16.11 | 15.81 | 16.54 | 0.13 | 1.1 | sides |
| 29 | `gr_down_v3_kernel<1,3,1,1,1,0>` | 14.6 | 8.44 | 7.55 | 8.32 | 0.12 | 1.0 | main |
| 30 | `native_quantize_q8_1_kernel` | 145.0 | 0.78 | 0.67 | 0.83 | 0.11 | 1.0 | main 75% |
| 31 | `native_mmvq_multi_kernel<Q4KTraits,2,4,2,0,0>` | 18.9 | 5.71 | 4.51 | 7.65 | 0.11 | 0.9 | main |
| 32 | `native_down_multi_kernel<42,1>` | 18.0 | 5.70 | 4.14 | 11.17 | 0.10 | 0.9 | sides |
| 33 | `native_mmvq_il_kernel<IlQ6K,3,4>` | 3.0 | 33.31 | 15.90 | 22.91 | 0.10 | 0.8 | main 5% |
| 34 | `wait_flag_ge_or_kernel` | 5.4 | 17.44 | 0.86 | 60.10 | 0.09 | 0.8 | sides |
| 35 | `gr_up_v3_kernel<1,1,1,1>` | 10.9 | 8.60 | 7.62 | 8.06 | 0.09 | 0.8 | main |
| 36 | `native_mmvq_multi_kernel<SmallTraits<IQ4NLBlock,4>,2,4,2,0,0>` | 26.1 | 3.59 | 3.04 | 4.64 | 0.09 | 0.8 | main |
| 37 | `resident_plan_kernel` | 48.0 | 1.87 | 1.98 | 2.24 | 0.09 | 0.8 | main |
| 38 | `native_mmvq_il_kernel<IlIQ4XS,2,2>` | 7.2 | 12.40 | 11.49 | 12.83 | 0.09 | 0.8 | sides |
| 39 | `gdn_conv_l2_multi_kernel` | 36.0 | 2.25 | 1.95 | 3.01 | 0.08 | 0.7 | sides |
| 40 | `gr_down_v3_kernel<1,1,1,1,1,0>` | 10.9 | 7.26 | 6.14 | 6.59 | 0.08 | 0.7 | main |

"n/window" is averaged over all 293 warm windows, including the windows whose graph does not contain the variant. "stream": main = the stream with the most kernels in the window (the token-serial chain); sides = the two others (expert GEMMs and attention projections).

### Spin / handshake-named kernels

Selection: name matches `wait|spin|flag|handshake|bell|doorbell|copy_i32|mapped|publish` (case-insensitive). 8 of the 138 variants match.

| kernel | what the signature shows | n/window | mean us | median us | p90 us | max us | ms/window | % of summed kernel time | % of window wall | stream |
|---|---|---|---|---|---|---|---|---|---|---|
| `dm_wait_combine_kernel` | wait for host-written flags + combine the CPU-pool expert rows (volatile float4/unsigned int args) | 42.59 | 25.72 | 2.37 | 78.24 | 1192.7 | 1.096 | 9.3 | 10.7 | main |
| `wait_flag_ge_planned_kernel` | spin until a host-written (volatile) flag reaches the planned value | 96.00 | 6.55 | 0.86 | 5.41 | 3852.5 | 0.629 | 5.3 | 6.2 | sides |
| `copy_i32_unless_kernel` | copy ints from host-mapped (volatile) memory unless a skip flag is set | 48.00 | 3.05 | 1.09 | 5.06 | 459.2 | 0.146 | 1.2 | 1.4 | sides |
| `doorbell_publish_res_kernel` | publish the resident plan to the host (writes an int + flag) | 48.00 | 3.00 | 1.82 | 5.76 | 326.5 | 0.144 | 1.2 | 1.4 | main |
| `wait_flag_ge_or_kernel` | spin until either of two host-written (volatile) flags reaches a value | 5.41 | 17.44 | 0.86 | 60.10 | 325.4 | 0.094 | 0.8 | 0.9 | sides |
| `copy_from_mapped_kernel` | copy float4 rows from host-mapped (volatile) memory | 15.68 | 1.55 | 1.18 | 2.08 | 11.8 | 0.024 | 0.2 | 0.2 | main |
| `copy_rows_or_zero_from_mapped_kernel` | copy selected rows from host-mapped memory or zero them | 5.41 | 3.88 | 1.28 | 13.76 | 204.9 | 0.021 | 0.2 | 0.2 | sides |
| `copy_i32_from_mapped_kernel` | copy ints from host-mapped (volatile) memory | 3.00 | 2.72 | 1.09 | 6.34 | 10.3 | 0.008 | 0.1 | 0.1 | main |

- **Total for the 8: 2.163 ms per window** = 18.3% of the summed kernel duration = 21.2% of the window wall. The three wait kernels alone (`dm_wait_combine_kernel`, `wait_flag_ge_planned_kernel`, `wait_flag_ge_or_kernel`) are 1.819 ms.
- **Union view:** 1.895 ms mean (0.944 ms median, p10 0.400 ms, p90 4.533 ms) of the window wall has only such kernels running on the GPU; a further 0.268 ms has them running alongside other kernels.
- **Dependence on T, summed (T1 1.64; T2 1.31; T3 1.93; T4 3.74; T5 5.07; T6 5.72 ms per window); only such kernels running (T1 1.44; T2 1.10; T3 1.68; T4 3.33; T5 4.55; T6 5.18 ms per window).**
- `dm_wait_combine_kernel`: T2 median 2.3 mean 15.1 us; T6 median 45.5 mean 66.1 us; absent from the T=1 graph.
- On the main stream the 8 take 1.272 ms per window.

## 4. Duration distribution of all kernels in warm window graphs

Bucket labels are upper-inclusive (2-4 means more than 2 and at most 4 us).

| bucket us | kernels per window | % of kernels | ms per window (summed over 3 streams) | % of summed kernel duration |
|---|---|---|---|---|
| <=2 | 750.7 | 39.6 | 0.747 | 6.3 |
| 2-4 | 389.8 | 20.5 | 1.129 | 9.6 |
| 4-8 | 328.7 | 17.3 | 1.943 | 16.5 |
| 8-16 | 327.5 | 17.3 | 3.675 | 31.1 |
| 16-32 | 77.4 | 4.1 | 1.589 | 13.5 |
| 32-64 | 10.8 | 0.6 | 0.453 | 3.8 |
| 64-128 | 6.5 | 0.3 | 0.562 | 4.8 |
| 128-512 | 5.7 | 0.3 | 1.500 | 12.7 |
| >512 | 0.2 | 0.0 | 0.201 | 1.7 |

Single numbers (summed duration of the kernels below the cut, divided by the window wall, mean of the per-window ratios):

| cut | ms per window | fraction of window-graph wall | fraction of summed kernel duration |
|---|---|---|---|
| <=4 us | 1.876 ms | **20.2%** (median 21.9%) | 15.9% |
| <=8 us | 3.820 ms | **40.7%** (median 43.0%) | 32.4% |
| <=16 us | 7.495 ms | **78.3%** (median 82.4%) | 63.5% |
| longer than 16 us | 4.305 ms | 38.3% (median 35.4%) | 36.5% |

60.1% of all kernel records (1140.5 of 1897.0 per window) are <=4 us long; 100.4 per window are longer than 16 us. Per T the <=4 / <=8 / <=16 shares of the wall are: T1 24.8/54.7/84.5; T2 22.2/43.9/83.6; T3 18.8/35.5/73.6; T4 13.4/29.4/62.4; T5 10.6/22.5/50.2; T6 8.8/20.6/51.5.

**Union view (an additive partition of the 10.203 ms wall, means):** only kernels longer than 16 us running 3.546 ms (median 2.402 ms); such kernels together with kernels <=16 us 0.607 ms; only kernels <=16 us running 5.825 ms; idle 0.225 ms.

**Main stream only (ms per window):** <=2 0.402; 2-4 1.012; 4-8 1.638; 8-16 2.297; 16-32 0.203; 32-64 0.254; 64-128 0.371; 128-512 1.048; >512 0.090.

## 5. Idle-gap census inside the window graph (union timeline)

1049.8 gaps per window; total idle 0.225 ms per window.

| gap length us | gaps per window | gaps in 293 windows | % of gaps | idle ms per window | % of idle | mean gap us |
|---|---|---|---|---|---|---|
| <=1 | 1049.42 | 307,480 | 100.0 | 0.1899 | 84.5 | 0.18 |
| 1-2 | 0.23 | 66 | 0.0 | 0.0003 | 0.1 | 1.30 |
| 2-4 | 0.00 | 0 | 0.0 | 0.0000 | 0.0 | 0 |
| 4-8 | 0.03 | 8 | 0.0 | 0.0002 | 0.1 | 7.65 |
| 8-16 | 0.00 | 0 | 0.0 | 0.0000 | 0.0 | 0 |
| 16-64 | 0.00 | 0 | 0.0 | 0.0000 | 0.0 | 0 |
| >64 | 0.17 | 49 | 0.0 | 0.0344 | 15.3 | 205.97 |

The idle is 307,480 node-to-node gaps of 0.18 us (one between almost every pair of consecutive graph nodes) plus 49 gaps longer than 64 us (mean 205.97 us) in the whole warm set. There is no gap of 2-4, 8-16 or 16-64 us. The 8 gaps of 4-8 us are one per T=4 window and the 66 gaps of 1-2 us lie in the T=1 windows: they are the unrecorded node of graphs 46 and 4 (see Method), 0.0003 ms + 0.0002 ms per window in total. Per T, idle in gaps >64 us: T1 0.218/0.003/0.000/0.000/0.000/0.000/0.048; T2 0.183/0.000/0.000/0.000/0.000/0.000/0.032; T3 0.183/0.000/0.000/0.000/0.000/0.000/0.031; T4 0.195/0.000/0.000/0.008/0.000/0.000/0.068; T5 0.197/0.000/0.000/0.000/0.000/0.000/0.000; T6 0.199/0.000/0.000/0.000/0.000/0.000/0.037 (buckets <=1/1-2/2-4/4-8/8-16/16-64/>64, ms per window).

### Ten largest gap classes by idle time (all gap lengths)

| # | kernel ending the busy period -> kernel(s) starting the next one [stream] | occurrences per window | idle ms per window | % of idle | median gap us | p90 us | max us |
|---|---|---|---|---|---|---|---|
| 1 | `gr_down_v3_kernel[main]` -> `gr_up_v3_kernel[main]` | 96.97 | 0.0217 | 9.7 | 0.19 | 0.2 | 233.9 |
| 2 | `cvec_kernel[main]` -> `gr_down_v3_kernel[main]` | 46.97 | 0.0115 | 5.1 | 0.19 | 0.2 | 230.7 |
| 3 | `dm_wait_combine_kernel[main]` -> `cvec_kernel[main]` | 40.82 | 0.0106 | 4.7 | 0.19 | 0.2 | 273.9 |
| 4 | `resident_plan_kernel[main]` -> `doorbell_publish_res_kernel[main]` | 48.00 | 0.0105 | 4.7 | 0.19 | 0.2 | 267.9 |
| 5 | `native_gu_swiglu_kernel[side]` -> `native_down_multi_kernel[side]` | 42.86 | 0.0099 | 4.4 | 0.19 | 0.2 | 235.2 |
| 6 | `route_resident_k[main]` -> `resident_plan_kernel[main]` | 48.00 | 0.0090 | 4.0 | 0.19 | 0.2 | 0.3 |
| 7 | `route_multi[main]` -> `route_resident_k[main]` | 42.59 | 0.0088 | 3.9 | 0.19 | 0.2 | 199.9 |
| 8 | `native_down_multi_kernel[side]` -> `dm_wait_combine_kernel[main]` | 29.41 | 0.0088 | 3.9 | 0.19 | 0.2 | 230.8 |
| 9 | `gr_up_v3_kernel[main]` -> `bf16_f32_mmvf_multi_kernel[main]` | 42.59 | 0.0076 | 3.4 | 0.19 | 0.2 | 0.3 |
| 10 | `bf16_f32_mmvf_multi_kernel[main]` -> `route_multi[main]` | 42.59 | 0.0076 | 3.4 | 0.19 | 0.2 | 0.3 |

Stream role: main = token-serial stream, side = either of the two other streams. Every class in this table is the ordinary 0.19 us launch gap of a node pair that occurs many times per window; the max column shows that each of these pairs also contains rare long gaps.

### Ten largest classes among gaps longer than 1 us (those long occurrences only)

| # | kernel ending the busy period -> kernel(s) starting the next one [stream] | occurrences per window | idle ms per window | % of idle | median gap us | p90 us | max us |
|---|---|---|---|---|---|---|---|
| 1 | `gr_down_v3_kernel[main]` -> `gr_up_v3_kernel[main]` | 0.02 | 0.0045 | 2.0 | 173.02 | 233.9 | 233.9 |
| 2 | `native_down_multi_kernel[side]` -> `dm_wait_combine_kernel[main]` | 0.02 | 0.0031 | 1.4 | 176.03 | 230.8 | 230.8 |
| 3 | `dm_wait_combine_kernel[main]` -> `cvec_kernel[main]` | 0.01 | 0.0031 | 1.4 | 231.92 | 273.9 | 273.9 |
| 4 | `cvec_kernel[main]` -> `gr_down_v3_kernel[main]` | 0.01 | 0.0029 | 1.3 | 229.74 | 230.7 | 230.7 |
| 5 | `native_gu_swiglu_kernel[side]` -> `native_down_multi_kernel[side]` | 0.01 | 0.0020 | 0.9 | 184.48 | 235.2 | 235.2 |
| 6 | `resident_plan_kernel[main]` -> `doorbell_publish_res_kernel[main]` | 0.01 | 0.0017 | 0.8 | 248.81 | 267.9 | 267.9 |
| 7 | `native_quantize_q8_1_kernel[main]` -> `native_mmvq_multi_kernel[main]` | 0.01 | 0.0015 | 0.7 | 226.91 | 280.9 | 280.9 |
| 8 | `native_quantize_q8_1_kernel[main]` -> `bf16_f32_mmvf_multi_kernel[main]` | 0.01 | 0.0015 | 0.7 | 222.50 | 270.9 | 270.9 |
| 9 | `resolve_kernel[main]` -> `copy_kernel[main]` | 0.01 | 0.0015 | 0.7 | 218.43 | 263.2 | 263.2 |
| 10 | `copy_kernel[main]` -> `decode_attn_i8_split_kernel[main]` | 0.01 | 0.0014 | 0.6 | 202.96 | 227.7 | 227.7 |

Gaps > 1 us: 0.4 per window, 0.0349 ms per window. The long gaps are spread over 29 node pairs; the ten listed have medians of 173-249 us and none has a long gap in more than 2.4% of the warm windows; each of the ten ends in a main-stream kernel.

## 6. The drafter phase

Interval: end of the previous window graph to the start of the parent launch. 293 warm phases.

| per warm phase | median | mean | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| duration | 1.711 ms | 1.956 ms | 0.774 ms | 3.391 ms | 0.369 ms | 7.203 ms |
| kernel records (count) | 212 | 191.7 | | | 53 | 271 |
| ACTIVE, union of kernels | 1.199 ms | 1.084 ms | 0.447 ms | 1.465 ms | 0.287 ms | 1.543 ms |
| IDLE, kernels only | 0.416 ms | 0.872 ms | 0.111 ms | 2.146 ms | 0.053 ms | 5.746 ms |
| ACTIVE, memcpys counted | 1.384 ms | 1.254 ms | 0.536 ms | 1.991 ms | 0.288 ms | 3.135 ms |
| IDLE, memcpys counted | 0.371 ms | 0.702 ms | 0.106 ms | 1.393 ms | 0.052 ms | 5.741 ms |

Idle share of the phase (kernels only): mean 35.4%, median 30.8%.

**Memcpys in the phase:** 4.56 per phase: H2D 1.32 per phase (15.36 MiB per phase, 334.1 us of copy time), D2H 3.25 per phase (6.36 MiB, 146.6 us; 1.44-2.54 MiB each, median 44.8 us each). The H2D >= 1 MiB transfers occur in 73 of the 293 phases, median size 25.85 MiB, median 510.2 us each, all on one stream (id 24). In 35 phases a memcpy is still running when the parent launch starts.

**Largest idle intervals (memcpys counted as activity; 0.702 ms per phase in total, 143.8 gaps per phase):**

| # | before -> after the idle interval | occurrences per phase | idle ms per phase | % of drafter idle | median us | p90 us | max us |
|---|---|---|---|---|---|---|---|
| 1 | `mtp_select_kernel` -> `H2D memcpy 96 KiB` | 0.25 | 0.377 | 53.7 | 751.7 | 4385.1 | 5075 |
| 2 | `mtp_select_kernel` -> `copy_from_mapped_multi_kernel` | 2.32 | 0.074 | 10.5 | 12.9 | 23.5 | 389 |
| 3 | `[previous window graph end]` -> `sampler_split_part_kernel` | 0.59 | 0.064 | 9.1 | 27.6 | 247.8 | 270 |
| 4 | `[previous window graph end]` -> `H2D memcpy 4 B` | 0.14 | 0.018 | 2.6 | 174.2 | 249.5 | 256 |
| 5 | `[previous window graph end]` -> `H2D memcpy 20 B` | 0.12 | 0.017 | 2.5 | 210.2 | 253.0 | 266 |
| 6 | `mtp_select_kernel` -> `[parent launch kernel]` | 0.50 | 0.017 | 2.5 | 16.0 | 22.0 | 269 |
| 7 | `[previous window graph end]` -> `H2D memcpy 8 B` | 0.09 | 0.012 | 1.7 | 177.4 | 251.0 | 265 |
| 8 | `sampler_split_merge_kernel` -> `copy_i32_from_mapped_kernel` | 0.47 | 0.011 | 1.6 | 10.7 | 46.6 | 247 |
| 9 | `H2D memcpy 96 KiB` -> `[parent launch kernel]` | 0.25 | 0.010 | 1.5 | 16.2 | 88.3 | 103 |
| 10 | `spec_verify_merge_kernel` -> `copy_i32_from_mapped_kernel` | 0.41 | 0.009 | 1.3 | 12.5 | 47.6 | 232 |
| 11 | `mtp_select_kernel` -> `res_patch_kernel` | 0.10 | 0.007 | 1.1 | 16.6 | 339.3 | 388 |
| 12 | `sampler_split_merge_kernel` -> `coupled_stage_kernel` | 0.11 | 0.005 | 0.7 | 32.3 | 76.1 | 162 |
| 13 | `mtp_select_kernel` -> `D2H memcpy 1925 KiB` | 0.05 | 0.003 | 0.5 | 64.3 | 116.3 | 121 |
| 14 | `[previous window graph end]` -> `H2D memcpy 12 B` | 0.03 | 0.003 | 0.4 | 16.7 | 249.9 | 250 |

Reading of the table: one class dominates the drafter idle. In 73 of 293 phases the last drafter kernel `mtp_select_kernel` is followed by a host gap of median 751.7 us (p90 4385 us), then an H2D memcpy of 96 KiB, then the usual 16 us to the parent launch: 0.377 ms per phase, 53.7% of the drafter idle. These are the same 73 phases that carry the >= 1 MiB H2D. The second class is the launch gap between consecutive step graphs (`mtp_select_kernel` to the next step's `copy_from_mapped_multi_kernel`), 2.32 per phase at a median 12.9 us, 0.074 ms per phase. The classes that start at `[previous window graph end]` are the host time before the first drafter activity; 128 of 293 phases have more than 100 us there.

Other facts: 31 of 293 phases contain a single idle gap > 1 ms (their phase duration: median 4.60 ms, mean 4.52 ms). The trailing host gap (last drafter kernel to parent launch) is below 100 us in 210 phases (median 15.4 us), 100-500 us in 37, 500-1400 us in 18, and >= 1.4 ms in 28 (median 3553 us).

**Kernel content of the phase (per phase, from the script):**

- eager (non-graph) kernels: `sampler_split_part_kernel` 1.00/phase, median 22.0 us, 22.2 us/phase; `sampler_split_merge_kernel` 0.59/phase, median 8.0 us, 4.7 us/phase; `spec_verify_merge_kernel` 0.41/phase, median 9.5 us, 3.9 us/phase; `res_patch_kernel` 0.25/phase, median 2.5 us, 1.3 us/phase
- graph kernels by name (top 12 by time): `sampler_split_part_kernel` 3.6/phase, median 69.6 us, 251.0 us/phase; `native_small_mmvq_kernel` 30.3/phase, median 2.8 us, 241.7 us/phase; `gdn_step_commit_layers_kernel` 0.9/phase, median 110.5 us, 98.1 us/phase; `gr_up_v3_kernel` 10.8/phase, median 7.7 us, 88.5 us/phase; `gr_down_v3_kernel` 10.8/phase, median 6.3 us, 67.8 us/phase; `decode_attn_i8_split_kernel` 3.6/phase, median 14.3 us, 48.5 us/phase; `native_mmvq_multi_kernel` 6.0/phase, median 5.5 us, 32.2 us/phase; `gu_grouped_t_kernel` 3.6/phase, median 8.7 us, 31.8 us/phase; `spec_draft_merge_kernel` 3.6/phase, median 7.9 us, 29.1 us/phase; `down_grouped_t_kernel` 3.6/phase, median 5.3 us, 20.1 us/phase; `embed_rows_kernel` 3.6/phase, median 4.0 us, 18.0 us/phase; `native_quantize_q8_1_kernel` 22.6/phase, median 0.7 us, 16.9 us/phase
- Summed drafter kernel duration: 1.142 ms per phase, of which 1.100 ms in the drafter graphs.

**Step graphs (duration = first to last kernel of the graph instance):**

| graph | role | nodes | phases | GPU duration median us | mean us | p90 us | min us | max us | ms per phase |
|---|---|---|---|---|---|---|---|---|---|
| 10 | round | 5 | 260 | 131.2 | 131.4 | 138.0 | 119.9 | 148 | 0.117 |
| 13 | step 1 | 51 | 84 | 287.7 | 283.6 | 292.5 | 261.4 | 393 | 0.081 |
| 52 | step 1 | 55 | 149 | 301.8 | 304.8 | 309.0 | 297.6 | 410 | 0.155 |
| 71 | step 1 | 58 | 42 | 307.4 | 310.8 | 314.9 | 304.2 | 413 | 0.045 |
| 74 | step 1 | 62 | 2 | 325.5 | 325.5 | 326.3 | 324.8 | 326 | 0.002 |
| 77 | step 1 | 59 | 8 | 317.1 | 317.0 | 324.2 | 312.7 | 324 | 0.009 |
| 80 | step 1 | 63 | 8 | 333.1 | 332.2 | 338.8 | 323.6 | 339 | 0.009 |
| 16 | step 2 | 50 | 254 | 262.1 | 263.4 | 268.5 | 255.5 | 287 | 0.228 |
| 19 | step 3 | 50 | 211 | 258.7 | 260.9 | 268.9 | 256.0 | 283 | 0.188 |
| 22 | step 4 | 50 | 165 | 258.9 | 261.3 | 270.4 | 256.0 | 284 | 0.147 |
| 25 | step 5 | 50 | 134 | 258.7 | 260.5 | 265.5 | 255.5 | 282 | 0.119 |

- The round graph (5 nodes) runs in 260 of 293 phases and takes 131.2 us median; its `gdn_step_commit_layers_kernel` is 110.5 us of that.
- Steps 2-5 (50 nodes each) take 258.7 us-262.1 us median. Step 1 has 6 captured variants (51-63 nodes) of 287.7 us-333.1 us median because it also carries the coupled-stage and multi-token nodes.
- The step chain (first step-graph kernel to last step-graph kernel end) is 1364.4 us median, 1170.8 us mean.
- graph sequence census (graphIds in start order: occurrences): (10, 52, 16, 19, 22, 25): 78; (10, 52, 16): 20; (10, 13, 16, 19, 22, 25): 20; (10, 52, 16, 19, 22): 20; (10, 71, 16, 19, 22, 25): 20; (10, 52, 16, 19): 17; (10, 52): 14; (10, 13, 16, 19): 11; (10, 71, 16, 19): 10; (13, 16, 19, 22, 25): 10; (10, 13, 16): 9; (10, 13): 7; (13,): 6; (13, 16): 6; (10, 71, 16): 6; (13, 16, 19): 6; (10, 71): 5; (13, 16, 19, 22): 5; (10, 13, 16, 19, 22): 4; (10, 80, 16, 19, 22, 25): 4; (10, 77): 3; (10, 80): 3; (10, 77, 16, 19): 2; (10, 77, 16, 19, 22, 25): 2; (10, 77, 16): 1; (10, 74, 16): 1; (10, 71, 16, 19, 22): 1; (10, 74): 1; (10, 80, 16, 19, 22): 1
- Step count in the phase versus T of the window that follows: 1 steps -> T1=33, T2=5, T3=1; 2 steps -> T2=42, T3=1; 3 steps -> T2=16, T3=30; 4 steps -> T2=18, T3=5, T4=8; 5 steps -> T2=82, T3=7, T5=9, T6=36. In 82 of the 134 five-step phases the next window graph is the T=2 graph.

## 7. Stage mapping

Window wall summed over the warm windows: 2989.5 ms; summed kernel duration 3457.2 ms (three streams).

### 7a. By name only

| class | ms per window (3 streams summed) | % of window wall | % of summed kernel duration | of which on the main stream, ms | kernels per window |
|---|---|---|---|---|---|
| host-handshake/spin/copy-to-from-mapped | 2.142 | 21.0 | 18.2 | 1.272 | 258.7 |
| CPU-expert combine/copy (T=1 path: mapped rows, hits, k10 combine) | 0.032 | 0.3 | 0.3 | 0.005 | 16.2 |
| router/top-k/plan | 0.532 | 5.2 | 4.5 | 0.532 | 144.0 |
| MoE routed-expert compute (grouped gate/up + down) | 1.739 | 17.0 | 14.7 | 0.000 | 192.0 |
| attention: GDN (recurrence/conv/norm) | 0.645 | 6.3 | 5.5 | 0.564 | 110.7 |
| attention: QSA (sparse attn, indexer/select, KV gather/append, rope, out-gate) | 0.691 | 6.8 | 5.9 | 0.611 | 143.9 |
| hyper-connections (HC stream read/write) | 1.756 | 17.2 | 14.9 | 1.756 | 195.0 |
| shared expert (gate scale by name; matmuls resolved by position) | 0.060 | 0.6 | 0.5 | 0.060 | 48.0 |
| normalization/quant | 0.281 | 2.8 | 2.4 | 0.164 | 304.6 |
| head/argmax/sampling | 0.007 | 0.1 | 0.1 | 0.007 | 2.0 |
| control vector (cvec) | 0.176 | 1.7 | 1.5 | 0.176 | 48.0 |
| dense matmul, stage NOT identifiable by name (native mmvq / bf16 mmvf) | 3.722 | 36.5 | 31.5 | 2.150 | 423.0 |
| other (embedding, per-token tail block: gate/broadcast/history/post) | 0.017 | 0.2 | 0.1 | 0.017 | 11.0 |

Members of each class (ms per window, count per window):

- host-handshake/spin/copy-to-from-mapped: `dm_wait_combine_kernel` 1.096 ms (n 42.6); `wait_flag_ge_planned_kernel` 0.629 ms (n 96.0); `copy_i32_unless_kernel` 0.146 ms (n 48.0); `doorbell_publish_res_kernel` 0.144 ms (n 48.0); `wait_flag_ge_or_kernel` 0.094 ms (n 5.4); `copy_from_mapped_kernel` 0.024 ms (n 15.7); `copy_i32_from_mapped_kernel` 0.008 ms (n 3.0)
- CPU-expert combine/copy (T=1 path: mapped rows, hits, k10 combine): `copy_rows_or_zero_from_mapped_kernel` 0.021 ms (n 5.4); `add_hits_kernel` 0.006 ms (n 5.4); `combine_k10_vec4` 0.005 ms (n 5.4)
- router/top-k/plan: `route_resident_k` 0.301 ms (n 48.0); `route_multi` 0.126 ms (n 42.6); `resident_plan_kernel` 0.090 ms (n 48.0); `route` 0.015 ms (n 5.4)
- MoE routed-expert compute (grouped gate/up + down): `native_gu_swiglu_kernel` 1.071 ms (n 96.0); `native_down_multi_kernel` 0.668 ms (n 96.0)
- attention: GDN (recurrence/conv/norm): `gdn_ab_multi_kernel` 0.349 ms (n 36.0); `gdn_step_norm_multi_kernel` 0.212 ms (n 36.0); `gdn_conv_l2_multi_kernel` 0.081 ms (n 36.0); `conv_residual_kernel` 0.003 ms (n 2.7)
- attention: QSA (sparse attn, indexer/select, KV gather/append, rope, out-gate): `block_topk_cluster_kernel` 0.203 ms (n 12.0); `decode_attn_i8_split_kernel` 0.160 ms (n 12.0); `resolve_kernel` 0.077 ms (n 12.0); `norm_rope_kernel` 0.075 ms (n 36.0); `block_scores_multi_kernel` 0.048 ms (n 12.0); `decode_attn_merge_kernel` 0.034 ms (n 12.0); `append` 0.030 ms (n 12.0); `copy_kernel` 0.030 ms (n 12.0); `kv_append_q8_kernel` 0.025 ms (n 12.0); `gate` 0.009 ms (n 11.9)
- hyper-connections (HC stream read/write): `gr_up_v3_kernel` 0.971 ms (n 97.0); `gr_down_v3_kernel` 0.784 ms (n 97.0); `broadcast_streams_kernel` 0.001 ms (n 1.0)
- shared expert (gate scale by name; matmuls resolved by position): `sigmoid_scale_rows_vec4_kernel` 0.060 ms (n 48.0)
- normalization/quant: `native_quantize_q8_1_kernel` 0.113 ms (n 145.0); `native_swiglu_quantize_q8_1_kernel` 0.056 ms (n 48.0); `native_q8_1_interleave_kernel` 0.051 ms (n 55.6); `quantize_q8_1_kernel` 0.045 ms (n 48.0); `weighted_rms_norm` 0.016 ms (n 8.0)
- head/argmax/sampling: `sampler_greedy_cluster_kernel` 0.004 ms (n 1.0); `logits_nonfinite_kernel` 0.003 ms (n 1.0)
- control vector (cvec): `cvec_kernel` 0.176 ms (n 48.0)
- dense matmul, stage NOT identifiable by name (native mmvq / bf16 mmvf): `native_mmvq_il_kernel` 1.801 ms (n 87.5); `native_mmvq_multi_kernel` 1.173 ms (n 179.5); `bf16_f32_mmvf_multi_kernel` 0.418 ms (n 108.3); `native_q6_k_mmvq_kernel` 0.176 ms (n 14.5); `native_q5_k_mmvq_kernel` 0.038 ms (n 3.9); `bf16_f32_mmvf_kernel` 0.037 ms (n 13.7); `native_iq4_xs_mmvq_kernel` 0.030 ms (n 4.7); `native_q4_k_mmvq_kernel` 0.027 ms (n 5.3); `native_small_mmvq_kernel` 0.022 ms (n 5.4)
- other (embedding, per-token tail block: gate/broadcast/history/post): `embed_rows_kernel` 0.006 ms (n 1.0); `history_advance_kernel` 0.005 ms (n 2.7); `gate_kernel` 0.003 ms (n 2.7); `broadcast_kernel` 0.002 ms (n 2.7); `post` 0.002 ms (n 2.0)

How the names were read: `gr_down_v3_kernel` / `gr_up_v3_kernel` are the hyper-connection stream read (src/kernels/cuda/fused_gr.cu:2784, "the stream-split hyper-connection read"); `sigmoid_scale_rows_vec4_kernel` is the shared-expert gate scale (src/kernels/cuda/shared_expert.cu:181); `native_gu_swiglu_kernel` and `native_down_multi_kernel` take a `NativeExpertLayout` argument (routed experts). `dm_wait_combine_kernel` both waits and combines; its whole duration is booked under handshake.

**Not classifiable by name (3.722 ms per window, 36.5% of the window wall, 423 kernels per window):** the nine dense-matmul kernels of the last but one bullet (`native_q*_mmvq`, `native_small_mmvq`, `native_mmvq_multi`, `native_mmvq_il`, `bf16_f32_mmvf(_multi)`): they name the weight type and the row count, not the layer. Also not classifiable by name: `post`, `gate_kernel`, `broadcast_kernel`, `history_advance_kernel` (0.017 ms per window together with `embed_rows_kernel`).

### 7b. Dense matmuls resolved by graph position (inference, not name)

Rule: each dense matmul is placed by where it runs relative to the main stream's hyper-connection reads (two per layer, a 97th before the head) and by the kernel that follows it on its stream. Layer structure read from the execution order: HC read, attention block (GDN in 36 layers, QSA in 12), HC read, MoE block (router, routed experts on the side streams, shared expert, combine), `cvec_kernel`. Unresolved dense matmuls: 0.

| sub-class | ms per window | % of window wall | kernels per window | members (ms per window) |
|---|---|---|---|---|
| attention segment, GDN layers (projections) | 1.678 | 16.4 | 107.9 | `native_mmvq_il_kernel` 1.237; `native_mmvq_multi_kernel` 0.295; `native_q6_k_mmvq_kernel` 0.093; `native_q5_k_mmvq_kernel` 0.028; `native_q4_k_mmvq_kernel` 0.014; `native_iq4_xs_mmvq_kernel` 0.012 |
| MoE block, non-router matmuls (shared-expert position) | 0.810 | 7.9 | 192.0 | `native_mmvq_multi_kernel` 0.631; `bf16_f32_mmvf_multi_kernel` 0.110; `native_small_mmvq_kernel` 0.022; `native_iq4_xs_mmvq_kernel` 0.013; `bf16_f32_mmvf_kernel` 0.009; `native_q4_k_mmvq_kernel` 0.009; `native_q6_k_mmvq_kernel` 0.008; `native_q5_k_mmvq_kernel` 0.005; `native_mmvq_il_kernel` 0.002 |
| attention segment, QSA layers (QKV/out projections) | 0.637 | 6.2 | 72.1 | `native_mmvq_multi_kernel` 0.246; `native_mmvq_il_kernel` 0.240; `bf16_f32_mmvf_multi_kernel` 0.094; `native_q6_k_mmvq_kernel` 0.034; `bf16_f32_mmvf_kernel` 0.008; `native_q5_k_mmvq_kernel` 0.006; `native_q4_k_mmvq_kernel` 0.005; `native_iq4_xs_mmvq_kernel` 0.005 |
| LM head (matmul feeding the greedy sampler) | 0.363 | 3.6 | 1.0 | `native_mmvq_il_kernel` 0.323; `native_q6_k_mmvq_kernel` 0.040 |
| router GEMM (feeds route/route_multi) | 0.189 | 1.8 | 47.9 | `bf16_f32_mmvf_multi_kernel` 0.174; `bf16_f32_mmvf_kernel` 0.015 |
| per-token tail block (two bf16 matvecs after the layer-0 combine) | 0.045 | 0.4 | 2.0 | `bf16_f32_mmvf_multi_kernel` 0.040; `bf16_f32_mmvf_kernel` 0.005 |

Counts confirm the placement: 3 matmuls per GDN layer, 6 per QSA layer, 1 router GEMM and 4 other matmuls per layer, 1 LM head. The two bf16 matvecs of the "per-token tail block" sit after the layer-0 combine (34 and 10 us in the instances read); the names do not say what they are.

### 7c. Merged stages (name + position), ms per window summed over the 3 streams

| stage | all warm, 3 streams | of which main stream | T1 (n=33) | T2 (n=163) | T3 (n=44) | T4 (n=8) | T5 (n=9) | T6 (n=36) |
|---|---|---|---|---|---|---|---|---|
| attention: GDN (recurrence/conv/norm) | 2.323 | 1.042 | 1.794 | 2.265 | 2.441 | 2.271 | 2.994 | 2.772 |
| host-handshake/spin/copy-to-from-mapped | 2.142 | 1.272 | 1.459 | 1.307 | 1.933 | 3.740 | 5.068 | 5.715 |
| hyper-connections (HC stream read/write) | 1.756 | 1.756 | 1.539 | 1.577 | 1.801 | 2.003 | 2.384 | 2.498 |
| MoE routed-expert compute (grouped gate/up + down) | 1.739 | 0.000 | 1.139 | 1.501 | 1.851 | 2.402 | 2.412 | 2.914 |
| attention: QSA (sparse attn, indexer/select, KV gather/append, rope, out-gate) | 1.328 | 0.879 | 1.098 | 1.294 | 1.313 | 1.380 | 1.610 | 1.630 |
| shared expert (gate scale by name; matmuls resolved by position) | 0.870 | 0.868 | 0.669 | 0.847 | 0.909 | 0.961 | 1.074 | 1.037 |
| router/top-k/plan | 0.720 | 0.720 | 0.574 | 0.679 | 0.789 | 0.769 | 0.835 | 0.918 |
| head/argmax/sampling | 0.370 | 0.370 | 0.363 | 0.375 | 0.331 | 0.361 | 0.346 | 0.408 |
| normalization/quant | 0.281 | 0.164 | 0.194 | 0.265 | 0.283 | 0.343 | 0.364 | 0.396 |
| control vector (cvec) | 0.176 | 0.176 | 0.155 | 0.170 | 0.170 | 0.155 | 0.224 | 0.224 |
| per-token tail block (two bf16 matvecs after the layer-0 combine; stage not identifiable) | 0.045 | 0.045 | 0.042 | 0.044 | 0.044 | 0.045 | 0.047 | 0.050 |
| CPU-expert combine/copy (T=1 path: mapped rows, hits, k10 combine) | 0.032 | 0.005 | 0.286 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| other (embedding, per-token tail block: gate/broadcast/history/post) | 0.017 | 0.017 | 0.011 | 0.014 | 0.018 | 0.024 | 0.028 | 0.032 |
| summed kernel duration | 11.799 | 7.314 | 9.323 | 10.338 | 11.883 | 14.454 | 17.386 | 18.594 |
| window wall | 10.203 | 8.318 | 8.732 | 10.236 | 12.787 | 15.503 | 16.653 |
| GPU idle (union) | 0.225 | 0.269 | 0.216 | 0.214 | 0.271 | 0.197 | 0.236 |
| drafter phase | 1.956 | 1.472 | 1.844 | 1.856 | 1.844 | 2.299 | 2.970 |
| wall + preceding drafter phase | 12.163 | 13.438 | 10.843 | 12.136 | 12.870 | 12.797 | 16.685 |
| tokens per second (T / (wall + drafter phase)) | 220.0 | 102.1 | 189.1 | 248.1 | 273.4 | 280.9 | 305.8 |

Column "of which main stream" is the part of the stage that runs on the token-serial stream; the main stream's 7.314 ms add up exactly across that column (the main stream runs 1238 of the 1897.0 kernel records per window). Columns T give the mean for windows of that T. The five rows below "summed kernel duration" are means: wall, GPU idle, the drafter phase that preceded the window, their sum, and the tokens/s that sum implies.

Shares of the 10.203 ms window wall (summed durations, three streams): GDN with its projections 2.323 ms = 22.8%; handshake/spin 2.142 ms = 21.0%; hyper-connections 1.756 ms = 17.2%; routed-expert compute 1.739 ms = 17.0%; QSA with its projections 1.328 ms = 13.0%; shared expert 0.870 ms = 8.5%; router/plan 0.720 ms = 7.1%; LM head and sampling 0.370 ms = 3.6%; normalization/quant 0.281 ms = 2.8%; control vector 0.176 ms = 1.7%; tail-block matmuls 0.045 ms = 0.4%; CPU-expert combine path 0.032 ms = 0.3%; other 0.017 ms = 0.2%. The shares sum to 115.6% because of the stream overlap (11.80 ms summed over a 10.203 ms wall).

PCIe staging/fetch: no kernel in the window graphs is a PCIe staging or fetch kernel by name (0 ms). The transfers are DMA memcpys: 5.22 H2D of >= 1 MiB per window inside the window interval (12.85 MiB, 388.8 us of copy time) and 73 large H2D (median 25.85 MiB) in 73 of 293 drafter phases.

## 8. Conclusion

Basis: 293 warm windows, means; the "window" of 12.163 ms (median 10.772 ms) is the window-to-window period = 10.203 ms window-graph wall + 1.956 ms drafter phase.

1. **(a) Kernels longer than 16 us: 4.305 ms per window** of summed duration (36.5% of all kernel time; 100.4 kernels per window). With nothing but such kernels on the GPU: 3.546 ms of wall. On the main stream alone they are 1.97 ms.
2. **(b) Kernels <=4 us: 1.876 ms per window** summed (60.1% of all kernel records), 1.41 ms of it on the main stream. Kernels of 4-16 us, not asked for but needed to close the sum: 5.618 ms.
3. The three figures sum to 11.80 ms against a wall of 10.203 ms; the difference, 1.821 ms, is time during which two or three streams run kernels at once (sum of per-stream active minus union).
4. **(c) GPU idle inside the window graph: 0.225 ms per window** (median 0.186 ms; 2.37% of the window wall, 1.85% of the period). 84.5% of it is 307,480 node-to-node launch gaps of 0.18 us; 15.3% is 49 gaps >64 us in 293 windows.
5. **(d) Spin/handshake-named kernels: 2.163 ms per window** summed (1.272 ms on the main stream); on 1.895 ms of wall (median 0.944 ms) they are the only kernels running. They are part of (a) and (b), not additional. By T, summed: T1 1.64; T2 1.31; T3 1.93; T4 3.74; T5 5.07; T6 5.72 ms per window.
6. **(e) Drafter phase: 1.956 ms per window** (median 1.711 ms): 1.084 ms of kernels, 0.170 ms of memcpys with the GPU otherwise idle, 0.702 ms idle. 0.377 ms of the idle is one host gap before an H2D memcpy, present in 73 of 293 phases.
7. **(f) Boundary: 22.5 us median** (16 us host + 2.8 us parent kernel + 3.5 us device-launch latency), 414.3 us mean because 28 of 293 boundaries take >= 1.4 ms (median 3559 us). It lies inside (e).
8. By name and position, the 11.80 ms of kernel time are: GDN 2.323 ms, handshake/spin 2.142 ms, hyper-connection read/write 1.756 ms, routed-expert GEMMs 1.739 ms, QSA 1.328 ms, shared expert 0.870 ms, router/plan 0.720 ms, LM head 0.370 ms, norm/quant 0.281 ms, control vector 0.176 ms, tail-block matmuls, CPU-expert combine path and other 0.09 ms in total.
9. On the token-serial main stream (7.314 ms of kernels), in ms: hyper-connections 1.756, handshake/spin 1.272, GDN 1.042, QSA 0.879, shared expert 0.868, router/plan 0.720, LM head 0.370, norm/quant 0.164, control vector 0.176, tail-block matmuls and other 0.067 ms. The routed-expert GEMMs (1.739 ms) run entirely on the two side streams.
10. The kernel names account for the window wall as stated above; no kernel name in the window graph denotes PCIe staging, and the trace shows no other idle inside the window graph than the 0.225 ms of item 4.

## Limits of this reading

- Node-level graph tracing records every node; any per-node profiling overhead is inside the measured durations and cannot be separated from this export.
- One node per execution in graphs 4, 46 and 65 has no kernel record after the first launch (see Method); its time is counted as idle when no other stream covers it.
- Stage assignment of the dense matmuls (7b) rests on graph position and successor kernels, not on names; the two bf16 matvecs of the per-token tail block remain unidentified.
- Memcpy sizes and directions are reported as recorded; the trace does not say what a transfer contains.
- Warm windows are the last 38% of the capture; the T mix of that stretch (T2 = 56%) sets every mean. The per-T tables in sections 1 and 2 and the per-T columns in section 7 carry the T dependence.
