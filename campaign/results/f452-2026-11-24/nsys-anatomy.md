# F-452 — the window boundary by nsys (2026-11-24)

Standalone engine, standing argv minus the profile save, harvest-probe.ids (1,663 tokens). Captures in /tmp (not committed):
`f452-node` (flag off, node-level graph trace), `f452-cpu` (flag off, CPU sampling, dwarf backtraces), `f452-dev` (flag on,
node-level). Warm = the last 38% of graph launches. The boundary gap = last drafter graph end -> first kernel of the next window graph.

## FLAG OFF (node trace)
- window graphs: [(2, 1956, 174), (23, 1873, 105), (29, 1885, 78), (32, 1933, 58), (35, 1941, 40), (41, 1949, 199)]
- boundary gap (us): n=218 median 1137 mean 1505 p10 582 p90 3471
- typical (<1.4 ms): n=142 median 875; slow (>=1.4 ms): n=76 median 2171
## FLAG ON  (node trace)
- window graphs: [(4, 1956, 296), (30, 1885, 64), (38, 1873, 341), (46, 1933, 13), (57, 1949, 39), (65, 1941, 19)]
- boundary gap (us): n=210 median 31 mean 614 p10 21 p90 2165
- device launch latency, parent kernel end -> first window kernel (us): median 3.5 p90 3.6
- typical (<1.4 ms): n=178 median 24; slow (>=1.4 ms): n=32 median 2928

## Flag off — the typical gap's host timeline (one median window, API calls by thread)
  +0.3 us   main thread: cudaStreamSynchronize (the draft's final sync), 1.2 us
  +0..+593  main thread on-CPU, no CUDA call: post-draft host path, stage_inputs, HostGraphLaunch thread creation
  +593      main thread scheduled out (finish() joins the launch helper)
  +609..+1215  helper thread (a fresh std::thread): cudaGraphLaunch = 606 us of host time
  +1078     GPU: first kernel of the window graph
## Flag off — all typical gaps, by T (medians, us): gap = pre-launch host + launch call; call-start -> first kernel
  T3 n=62: gap 1297 = pre-launch 857 + call 552; first kernel 468 after the call start
  T4 n=20: gap 2141 = 1725 + 557; 470
  T6 n=20: gap 1463 = 961 + 568; 536
  T8 n=10: gap 2141 = 1624 + 584; 491
  drafter graphs' cudaGraphLaunch host time: median 11.7 us (50 nodes)
## Flag off — main-thread CPU samples inside the gaps (sampler ~0.7 kHz)
  typical: 15 of 16 in Verifier::run's layer-reach spin (verify.cpp:3154; one instruction, 0x4fc82f5)
  slow:    65 of 117 in __memcpy_avx512_unaligned_erms, 16 under FileExpertSource::commit_exchanges() (expert_source.cpp:4198, the copy path :4218)
## Graph census (flag off): window T=6 = 1949 kernel nodes on 3 captured streams; 0 host nodes; 0 memsets; 174 memcpy nodes in graphs overall
## Engine accounting of the standalone runs (not lanes): flag off 1200 tok 165.3 t/s; flag on 1200 tok 148.9 t/s with the W1 controller cutting T (1.56 vs 2.12 tokens/round) - the sampled standalone is not a lane; the server ladder decides.

