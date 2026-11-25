# F-458 - what limits the CPU-expert kernel at the concurrent lane's shape

Standalone, CPU only, synthetic experts. Nothing deployed, no GPU program run, the engine not started.
Evidence: `pool-bench.out` (30 units per point, median; quartiles in the file), `pool-bench-repeat.out` (4 min later), `notes.md`.
## Built and run
- `pool_bench` (CMakeLists.txt, after `expert_multi_test`), `src/kernels/cpu/pool_bench.cpp`. `taskset -c 16-31 nice -n 19 ninja -C build-fresh pool_bench`: 0 warnings, 0 errors.
- Times the engine's own calls, `ExpertPool::run_split_multi_native` / `run_split_multi` (expert_source.cpp:5147), over N experts x T rows: median of 30 units after 5 warm-up; phases from the pool's `ms_multi_gu / q / down`.
- Pool as the standing recipe builds it: 15 workers + the draining host, pinned, 48 tasks per phase, `STRATA_POOL_DIET=1`, `STRATA_POOL_SPIN_US=100`. Fixture: the standing pack's seven native layer types with their layer counts (packs/iq3_s, 2560 x 640) and the shadow Q2_0 blob; 320 synthetic experts per format.
- Quiet box: load 1.70 when the measured phase began, 3.19 when it ended (the bench's own 16 threads); a browser held ~15 % of one core. Repeat run: 2.05 -> 5.40.
## Three facts that shape the reading
1. **12 rows read every weight twice.** `ExpertJobMulti` holds `MAXT = 8` rows and the engine refuses more (expert_source.cpp:4791); the kernels stop at 8 too (`iq4nl256_down_rows_v` fills a `y[8]`, iq_avx2.cpp:539; `gu_rows_nt` has cases 1..8, iq_avx512.cpp:214). 12 rows over two cohorts is 6 + 6: 89.8 us against 46.8 for one 6-row call. A 12-row single pass is kernel work.
2. **DRAM read ceiling 48-57 GB/s**: this bench's read loop 47.8-48.0 (16 threads), a standalone loop 49-57, one thread alone 35-53; 2 x 48 GiB DDR5-5200 = 83 GB/s theoretical. One core pulls ten times the kernel's 4.6 GB/s.
3. **A hot set must fit the smaller CCD**: the 16 threads span two CCDs, L3 96 MiB (CPUs 0-7) and 32 MiB (CPUs 8-15), and every thread reads every expert. N = 8 (16 MB) is cache-resident; N = 16 and 32 only partly.
## Table 1 - DRAM against cache (T = 12 = 6 + 6, N = 8, 15 workers + host) - us per expert-call
| format | layers | cold | gu | q | down | GB/s | hot | gu | q | down | GB/s | cold/hot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IQ2_S/IQ4_NL | 15 | 86.3 | 44.6 | 1.5 | 39.4 | 45.7 | 60.6 | 34.9 | 1.6 | 27.9 | 65.1 | 1.43 |
| IQ3_XXS/IQ4_NL | 14 | 93.6 | 53.2 | 1.6 | 39.5 | 46.5 | 72.2 | 45.7 | 1.5 | 28.6 | 60.2 | 1.30 |
| IQ3_S/IQ4_NL | 9 | 100.4 | 60.9 | 1.5 | 39.3 | 46.4 | 75.5 | 50.7 | 1.6 | 28.6 | 61.7 | 1.33 |
| IQ2_S/Q2_0 | 5 | 66.1 | 42.6 | 2.1 | 22.4 | 45.7 | 57.0 | 35.9 | 2.3 | 20.6 | 53.0 | 1.16 |
| IQ3_XXS/Q2_0 | 3 | 83.7 | 58.2 | 2.2 | 23.5 | 41.0 | 63.0 | 44.0 | 2.3 | 19.6 | 54.5 | 1.33 |
| IQ4_XS/IQ4_NL | 1 | 132.3 | 90.2 | 1.6 | 42.0 | 40.2 | 92.0 | 67.4 | 1.5 | 27.1 | 57.9 | 1.44 |
| IQ3_S/Q2_0 | 1 | 85.9 | 60.7 | 2.1 | 23.0 | 43.5 | 67.9 | 47.7 | 2.2 | 20.1 | 55.0 | 1.26 |
| shadow Q2_0 (`run_split_multi`) | - | 52.1 | 31.4 | 2.0 | 19.1 | 53.0 | 32.2 | 19.3 | 2.2 | 13.0 | 86.0 | 1.62 |
| **native mix** | 48 | **89.8** | 52.1 | 1.6 | 36.4 | 45.6 | **67.4** | 42.6 | 1.7 | 26.8 | 60.8 | 1.33 |
gu = gate/up rows, q = host-serial quantization, down = down rows (means; us is the median). Repeat: mix cold 91.0, hot 69.5 (+1.3 %, +3.1 %).
## Table 2 - workers (T = 12 = 6 + 6, N = 8) - us per expert-call, cold \| hot (GB/s cold)
| format | 1 thread, kernels direct | 5 threads | 9 threads | 16 threads |
|---|---|---|---|---|
| native mix | 898.3 \| 899.9 (4.6) | 189.8 \| 186.9 (21.6) | 113.9 \| 106.8 (35.9) | 89.8 \| 67.4 (45.6) |
| IQ2_S/IQ4_NL | 822.6 \| 831.6 (4.8) | 175.1 \| 170.9 (22.5) | 106.1 \| 96.9 (37.2) | 86.3 \| 60.6 (45.7) |
| shadow Q2_0 | 414.1 \| 407.3 (6.7) | 89.1 \| 86.0 (31.0) | 62.1 \| 52.8 (44.5) | 52.1 \| 32.2 (53.0) |
Hot, per thread: 10.9 / 10.5 / 10.2 / 9.1 G weights/s - flat: each core runs its own compute. Cold: 4.6 / 21.6 / 35.9 / 45.6 GB/s: it stops near the DRAM ceiling, not at the cores. One thread alone: cold = hot.
## Table 3 - rows against experts (15 workers + host, cold) - us per expert-call, native mix
| T \ N | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 1 (ggml's single-token dot) | 54.8 | 45.2 | 44.9 | 41.7 | 39.8 |
| 4 | 50.7 | 45.9 | 44.6 | 43.3 | 41.6 |
| 6 | 56.6 | 48.1 | 46.8 | 45.7 | 44.7 |
| 8 | 60.8 | 52.7 | 49.8 | 49.7 | 49.2 |
| 12 = 6 + 6 | 97.2 | 90.1 | 89.8 | 89.3 | 88.5 |
| 16 = 8 + 8 | 115.1 | 100.9 | 101.5 | 99.3 | 97.4 |
Hot at N = 8, same T: 14.6 / 27.8 / 33.8 / 39.8 / 67.4 / 80.2. Fit us = a + b*T over T = 1, 4, 6, 8: cold a 27-50, b 0.3-2.6 (R² 0.2-0.9); hot a 1-17, b 2.4-7.3 us per row (R² 0.90-0.999).
## Verdict
- **Limiting resource: DRAM read throughput, as the call's size drives it.** Cold, the native formats run at 40.2-46.5 GB/s. Dequant speed is ruled out by T = 1, a leaner kernel whose cache-resident time (14.6 us) is a third of its DRAM-resident 44.9: it still streams only 45.6 GB/s at N = 8, where the multi-token kernel at T = 6 takes 46.8 us against its 44.9 (+4 %).
- Compute-light or not, a call of N = 8 experts gets ~45 GB/s: N = 1 gets 37.3, N = 32 gets 51.4. The lean kernel's lead grows with the call (+3 % at N = 1, +12 % at N = 32): at 6 rows the multi-token kernel sits at the compute/bandwidth crossover, and a 4x bigger call buys it 4 % (46.8 -> 44.7 us).
- Compute limits cache-resident experts only (hot 1.16-1.62x faster): there the per-row dot products are 64 % of the time at T = 6 for the mix (50-69 % across the IQ formats of 3+ layers); gate/up is 63 % of it, down 40 %. The intermediate quantization is 1.5-2.3 us, 2-3 %: not it. Rows are nearly free from DRAM (T = 1 -> 8: +11 %); the second read is what costs.
- The pool's own synchronization (T = 12, N = 8, cold, mix): 150 us of host idle before each call 94.8 (repeat 88.4) against 89.8 (91.0) with none - +5.6 % / -2.9 %, zero within the run-to-run spread; 2000 us: 100.4 / 101.0, +11-12 %; sleepers kept spinning: 89.9 / 89.7; diet off: 89.8 / 86.7. Not the limiter at the engine's cadence.
## Against the engine's 63-66 us per expert at T = 12
Not reproduced for DRAM-resident native experts: 89.8 here, and the engine's figure is 27-30 % below it. It matches cache-resident native experts (67.4; the engine's figure is 2-7 % below) and lies between the shadow store cold (52.1) and the native mix cold (89.8). The engine's phase split (gate/up ~37, down ~17) matches the Q2_0-down forms here (down 19-23 us), not the 38 IQ4_NL-down layers (39-42 us).
Unconfirmed, and it would settle the mix: that the engine's CPU experts are mostly shadow-store and Q2_0-down forms, the cheaper ones. Its 175-330 CPU experts a round over 48 layers and two cohorts would be N ~ 2-3 a call: read the N = 1 and 4 columns. The engine adds at most the synchronization term above.
## Headroom for a kernel 25 % faster
- DRAM-resident experts: <= 4 % of pool time at N = 8 (46.8 against 44.9 us for the compute-light kernel), 6 % at N = 4. At the premise's pool share (25 % = +6-8 % concurrent puts the pool at ~26-32 % of a round): <= +1.3 % concurrent at N = 8, +2.0 % at N = 4. Inside the +-4 % band: not worth starting as a dequantization optimization.
- Cache-resident experts: 67.4 -> 55.7 x 0.75 + 11.7 = 53.5 us (11.7 = what a call pays whatever its size: N = 1 against N = 32), <= 21 % of pool time, <= +5-7 % concurrent - and only for what a second call finds in the 32 MiB CCD's L3. The 6 + 6 unit shows how little is found: 1.92x one 6-row call, not 2x.
- Not bounded here: the gap from 45.6 to the 51-57 GB/s that N = 32 and the read loops reach. The instrument that would: DRAM read counters (UMC `cas_count_read`) over one cold call at N = 8 against N = 32.
- The lever the measurement does price is bytes, not the kernel: 12 rows in ONE call, 89.8 -> ~52 us, -42 %. It needs kernels past MAXT = 8 and rows that the two cohorts hold at different times (F-233's structure): a structure question, nothing adopted here.
