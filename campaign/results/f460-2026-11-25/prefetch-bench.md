# F-460 - does a prefetch of the next expert call hand the call the hot rate? (the benchmark falsifier)

Standalone, CPU only, synthetic experts. Nothing adopted, no engine change, no GPU program, the engine not started.
Evidence: `prefetch-30-ccda.out`, `prefetch-30-ccdb-1.out`, `prefetch-30-ccdb-2.out` (three complete runs, 30 units per point, medians:
P1 call us, P2 ratios and GB/s, P3 wakes; `POOL_BENCH_PF_TRACE=1` prints every unit), `prefetch-30-first.out` (a fourth run whose P1 lacks the
prefetched-call column, P2 complete), `notes.md`. Below, `a / b / c` are those three runs in that order.

## Built and run
- `pool_bench --prefetch` (src/kernels/cpu/pool_bench.cpp, +667 -3 lines, a new option; `POOL_BENCH_PF_TRACE`, `POOL_BENCH_PF_TRAFFIC_CCD=B`):
  `taskset -c 16-31 nice -n 19 ninja -C build-fresh pool_bench`: 0 warnings, 0 errors. The default run on one format: 85 lines before and after,
  the same lines with the numbers stripped (the F-458 binary kept for the diff).
- The unit: select the call's N experts (cold: the next N of the 320) -> the PASS reads every byte of those N experts through the bench's
  not-foldable pattern, checked against per-expert byte sums (53,760 passes a run, 0 mismatches: exactly the call's bytes, each once) -> the
  host idles G while ONE thread reads X MiB of unrelated memory -> the engine's own call, timed inside -> the traffic is waited for.
- Who reads: `host` (the host thread inline), `one` (a thread beside the host), `all` (16 beside the pool's 16, a contiguous 1/16 each),
  `both` (one per CCD, each everything). The pool claims tasks at run time (pool.cpp:863), so a pass places a line in a CCD, never in the
  core that will read it. The cold control of each carries the same gap and traffic behind a SHAM pass (same readers, same bytes, other memory).
- Layout (sysfs): host on logical 0 (CCD A, L3 96 MiB); workers on 1-15 (CCD B, L3 32 MiB); readers on the SMT siblings 16-31. Traffic: beside
  the host in run a, beside worker 7 in runs b and c. Idle states (acpi_idle): C1 1 us, C2 18 us, C3 exit 350 us.
- Quiet box (1-minute load at start -> end): a 2.11 -> 4.20; b 0.97 -> 5.17; c 4.30 -> 8.68 (the average still decaying from b; the run queue
  was empty); first 1.47 -> 6.02. The end load is the bench's own 33 threads. 102-109 s a run.

## Table - native mix (layer-weighted, mean blob 2.05 MB), T = 6, 15 workers + host. us per CALL of N experts
| N | G ms | MiB | pf call: one / all / both | cold (one's control) | hot, same G, X | pass one (GB/s) / all / both | call/cold: one / all / both |
|---|---|---|---|---|---|---|---|
| 2 | 0 | 0 | 90.6 / 85.2 / 93.7 ; 91.7 / 90.7 / 88.9 ; 97.1 / 93.6 / 93.6 | 107.9 ; 114.2 ; 111.3 | 72.3 ; 73.7 ; 85.0 | 106 (38.5) / 86 / 163 ; 109 (37.6) / 99 / 159 ; 114 (36.1) / 99 / 160 | 0.84 / 0.81 / 0.80 ; 0.80 / 0.84 / 0.76 ; 0.87 / 0.85 / 0.79 |
| 2 | 0.25 | 8 | 94.4 / 94.7 / 90.0 ; 101.3 / 99.6 / 95.0 ; 97.1 / 96.4 / 91.0 | 115.1 ; 116.7 ; 116.5 | 90.0 ; 92.7 ; 90.5 | 100 (40.8) / 87 / 159 ; 112 (36.7) / 87 / 159 ; 109 (37.7) / 89 / 159 | 0.82 / 0.84 / 0.78 ; 0.87 / 0.85 / 0.79 ; 0.83 / 0.83 / 0.79 |
| 2 | 0.5 | 32 | 121.3 / 108.4 / 106.9 ; 108.2 / 113.3 / 105.5 ; 107.6 / 108.2 / 101.1 | 128.6 ; 132.4 ; 136.0 | 101.3 ; 108.9 ; 107.8 | 96 / 88 / 157 ; 114 / 88 / 157 ; 111 / 86 / 153 | 0.94 / 0.84 / 0.82 ; 0.82 / 0.85 / 0.79 ; 0.79 / 0.83 / 0.77 |
| 3 | 0 | 0 | 132.9 / 134.2 / 130.0 ; 137.7 / 138.7 / 125.2 ; 135.7 / 137.3 / 133.0 | 161.3 ; 165.4 ; 163.7 | 110.5 ; 110.4 ; 111.7 | 148 (41.6) / 129 / 238 ; 171 (36.0) / 134 / 237 ; 160 (38.4) / 128 / 240 | 0.82 / 0.83 / 0.78 ; 0.83 / 0.85 / 0.74 ; 0.83 / 0.85 / 0.79 |
| 3 | 0.25 | 0 | 134.5 / 137.3 / 128.5 ; 135.0 / 138.2 / 128.9 ; 134.2 / 138.2 / 128.5 | 162.2 ; 162.7 ; 161.8 | 127.8 ; 129.8 ; 130.7 | 162 / 127 / 232 ; 169 / 127 / 236 ; 161 / 127 / 230 | 0.83 / 0.85 / 0.79 ; 0.83 / 0.85 / 0.79 ; 0.83 / 0.86 / 0.79 |
| 3 | **0.25** | **8** | 133.7 / 140.4 / 133.6 ; 139.7 / 140.6 / 131.6 ; 134.9 / 137.0 / 129.1 | 162.1 ; 167.3 ; 161.5 | 131.8 ; 130.2 ; 128.6 | 153 (40.1) / 136 / 243 ; 165 (37.2) / 134 / 237 ; 164 (37.4) / 129 / 234 | **0.82 / 0.87 / 0.78 ; 0.84 / 0.85 / 0.79 ; 0.84 / 0.85 / 0.79** |
| 3 | 0.5 | 32 | 160.9 / 154.3 / 143.6 ; 151.3 / 157.9 / 145.7 ; 145.3 / 158.2 / 144.9 | 188.8 ; 187.2 ; 185.7 | 151.3 ; 154.4 ; 154.4 | 155 / 132 / 236 ; 155 / 129 / 230 ; 149 / 129 / 230 | 0.85 / 0.80 / 0.76 ; 0.81 / 0.84 / 0.77 ; 0.78 / 0.85 / 0.78 |
Anchors, gap 0, no pass, no traffic - cold 110.5 ; 107.4 ; 107.9 (N = 2), 157.0 ; 160.1 ; 154.1 (N = 3); hot 80.4 ; 77.6 ; 76.5, 117.2 ; 108.6 ; 114.5.
`host` (the host thread reading) agrees with `one` within 0.04 in 20 of the 21 cells above (run a, N = 3, 0.5 ms, 32 MiB: 0.92 against 0.85), at a
2-7 % longer pass. N = 4 and 8 (run a): call/cold one / all / both 0.81 / 0.83 / 0.79 and 0.78 / 0.79 / 0.77 at (0.25, 8); passes 206 / 169 / 312 and
409 / 336 / 630 us. Per native format at (0.25, 8), run a (table S): one 0.71-0.90, both 0.71-0.86.
Shadow Q2_0 store (blob 1.38 MB), N = 3, (0.25, 8): pf one 80.8 ; 113.7 ; 84.3 against cold 112.6 ; 113.8 ; 113.0 = 0.72 ; 1.00 ; 0.75; both 0.68 ; 0.70 ;
0.71. Three traced repeats: one 0.77 / 0.82 / 0.88, both 0.72 / 0.70 / 0.72. Wakes at (0.25, 8), us: one 2.9-3.4, both 4.2-4.8, all 31-42 (N = 2), 48-74 (N = 3).

## Verdict
- **Against the stop rule: PASSED - the lever does not close.** At G = 0.25 ms with 8 MiB of traffic the prefetched call beats its cold control by
  13-18 % (N = 2) and 16-18 % (N = 3) with one reader, 13-17 % with sixteen, 21-22 % with a reader per CCD.
- **At G = 0 it does not reach F-458's hot rate: 1.13-1.27x** (N = 3: 132.9-137.7 against the gap-0 anchor 108.6-117.2). The pass (148-171 us) outlasts the
  workers' 100 us spin, so the call wakes parked workers. Against a hot call under the SAME gap it is +3-5 % with one reader (134.2-135.0 against
  127.8-130.7) and 0.98-1.01 with a reader per CCD: the pass delivers the hot rate a call can have after a gap.
- **No decay, so no constant to fit.** G 0 -> 0.5 ms: call/cold (one) 0.80-0.87 -> 0.79-0.94 (N = 2), 0.82-0.83 -> 0.78-0.85 (N = 3). 0 -> 32 MiB of traffic:
  the same band, with the traffic through the 96 MiB L3 (a) or the 32 MiB one (b, c). A 4-6 MB set survives 0.5 ms and 32 MiB of streaming. Traffic costs only
  when it overruns into the call (32 MiB takes 758-906 us): both arms slow ~15 %, the ratio holds.
- **What caps it is the gap, not the memory system.** A 0.25 ms gap costs a hot call +9-20 % (127.8-130.7 against 108.6-117.2) and a cold one +2-5 % (161.5-167.3
  against 154.1-160.1). The engine's every call follows the GPU's wait, so its hot floor is hot-under-gap, and the hypothesis's -25 % (against gap-0 hot)
  is -16 to -18 % against what is reachable.
- **CCD locality is worth 3-5 % of the call.** One reader leaves the other CCD's 8 threads reading across the fabric; sixteen contiguous readers are no better
  (half of every thread's reads remote) and pay a 28-74 us wake against 1-3 us for one.
- **One instability, mechanism unconfirmed:** the shadow store's N = 3 cell read 1.00 in run b (0.72-0.88 in the five others). Its 8 MiB of traffic takes
  230-330 us, straddling the 250 us gap; in that run more than half the units overran. Per CCD the same cell held (0.70).

## What the prefetch costs
- One reader: 36-42 GB/s = 25-28 us per MB; 55 us per expert at the mean blob. N = 2: 100-114 us; N = 3: 148-171 us; N = 4: 206-223 us; N = 8: 409-448 us.
- It fits the idle time on a quiet DRAM: N <= 4 in 250 us (solo), N <= 5 in ~320 us (concurrent). Two concurrent one-thread readers (the `both` pass)
  share 51.5 GB/s, 26 each = 0.70x their alone rate: under a staging copy of the same rate the pass takes 1.4x - N = 3: 215-231 us.
- A reader per CCD reads every line twice at the DRAM ceiling: 234-243 us at N = 3, 310-325 at N = 4 - N = 3 only, with nothing else on DRAM.
- The pass holds 70-80 % of the DRAM read ceiling for its duration; whatever the host stages in that window pays that (not measured here).

## Price, as arithmetic (upper bounds: every CPU call prefetched, perfect routes, the pass free)
- (cold - pf) / cold x 0.46 x 0.715, one reader, native mix: N = 3: (1 - 0.82..0.84) = 16-18 % -> **+5.3-5.9 %** of the concurrent window;
  N = 2: 13-18 % -> +4.3-5.9 %. A reader per CCD: 21-22 % -> +6.9-7.2 %.
- Shadow Q2_0 calls (run a, per format): 0.65-0.74 with one reader -> +8.6-11.7 %; 0.64-0.68 per CCD -> +10.5-11.9 %.
- Weakest native forms with one reader (N = 3, run a): IQ3_XXS/Q2_0 0.90 (+3.4 %), IQ2_S/Q2_0 0.86 (+4.6 %); strongest IQ2_S/IQ4_NL 0.81 (+6.3 %).
- Against the priced +7-9 %: the mix lands at +4.3-5.9 % with the likelier build. Inside or at the edge of the +-4 % band before the lookahead's misses.

## Not told here; the instrument that would
1. Do the other CCD's threads hit the first CCD's L3 across the fabric, or go to DRAM, when one reader prefetches (the 3-5 %)? DRAM read bytes over one
   call: UMC `cas_count_read` (perf; `amd_umc_0/1`, `amd_df`, `amd_l3` are registered on this box) on a `one` call against a cold and a `both` call.
2. The router lookahead's hit rate. Assumed perfect; a miss is a wasted pass and a cold call.
3. The pass beside the engine's real staging traffic: measured only as two equal readers (0.70x each), with the reader and the traffic never overlapping.
4. What the pass's DRAM share costs the GPU's staging in the same window.
