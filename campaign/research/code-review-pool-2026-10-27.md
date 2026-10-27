---
type: research
status: code-review
date: 2026-10-27
lane: strata-custom-2
topic: the host-side CPU expert pool read end-to-end — miss path, tail, adaptive tier, levers, ranked
method: read-only code walk (server untouched, pid 20013), anchored on decode-anatomy-2026-10-27.md
---

# Code review — the CPU miss path, the tail, and what moves them (2026-10-27)

Every claim below carries a file:line. Live-lane claims carry the boot log line. The measured baseline
is `campaign/research/decode-anatomy-2026-10-27.md`: window 14.15 ms / 2.74 tok; per-layer host
2.69 ms [plan .11 + actq .05 + jobs .13 + CPU 2.24]; 0.64 CPU entries/layer-window (~30/window,
~74 µs each); `dm_wait_combine` 1391 ms and `wait_flag_ge_planned` 598 ms over 1272 windows, p50
2.2/1.2 µs, p99 169/103 µs, max 9.8/1.26 ms.

## 1. One CPU-miss entry, end to end

**Ring → dispatch.** The host spins `while (*seq < want)` on the mapped layer doorbell
(`src/core/verify.cpp:3049`), then calls the pool callback for layer L
(`verify.cpp:3097-3099`) = `expert_pool_dispatch_multi` (`src/core/expert_source.cpp:4533`).

**Routing (CPU vs VRAM), per entry.** `kind[i]` over the window's `n_tok*k <= 128` entries
(`expert_source.cpp:4599-4600`): kind 0 (VRAM) when `d.host_res[layer*512+e] >= 0` — the GPU tier's
residency table (ExpertCache, `src/core/expert_cache.cpp:154`); 10,313 of 24,576 experts were
resident at the live boot (log: "token graph hit path: 10313 resident experts, decided on the
device"). Peer/helper holders → kind 2 (`:4759-4764`, unused here). `ForesightSwap::take` would also
make a kind 0 (`:4763-4764`) — off on this lane (see §3). The rest are misses: the last
`pcie_num/256` share (`PcieModel.pick`, `:4734`; `--pcie-frac 0.35`) goes to kind 1 **only if the
blob is pinned** (`:4768`), i.e. it comes from the 36.95 GiB pinned RAM complement (log:
"mapped pinned cache complement ready: resident 36.95 GiB"); the remainder is kind -1 (CPU).
`STRATA_ROUTE_TAIL_SKIP` (default 7 on CUDA serve, `:4521-4531`) first zeroes every group every
token routes only at rank >= R (`:4694-4726`, kind -2). `STRATA_ROUTE_RESIDENT=0.75` biases the
router itself toward resident experts ranks 6-9 within a 0.75-logit margin
(`src/kernels/cuda/native_router.cu:253-324`).

**Plan publish (flagA) — before the CPU works.** The distinct-expert groups are written to the
mapped plan ring through `GpuPlanSink` (counts/start/dst/tok/ptr; `include/strata/core/expert_source.hpp:364-387`),
then `std::atomic_thread_fence(seq_cst)` + `P.publish` (`expert_source.cpp:4808-4810`) =
`Verifier::publish_plan` (`verify.cpp:3344-3348`): `_mm_sfence(); *h_flagA_ = cur_layer_ + 1`.
The GPU's `wait_flag_ge_planned_kernel` (`src/kernels/cuda/verify_kernels.cu:1647-1653`) releases
the VRAM groups at that store — the GPU starts computing while the host quantizes. PCIe groups wait
for flagB: `P.fetch` = `Verifier::fetch_dma` (`verify.cpp:3316-3342`) issues
`cudaMemcpyAsync` per pinned blob into staging slots and raises flagB from a `cudaLaunchHostFunc`
when the DMA lands.

**actq → jobs → pool.** Host quantizes each token's activation once per layer
(`act_quant_q8_1` / `native_quant_act`, `expert_source.cpp:4859-4869`; 0.05 ms/window). The miss
list is deduped (`:4882-4886`) and — the load-bearing branch — **with the shadow store armed the
misses skip all source machinery and prefetch**: `if (shadow) { /* nothing */ }` (`:4887-4889`),
and each job's blob is `strata::core::shadow::blob(layer, e)` (`:4954`), a raw pointer into the
31.64 GiB mapping: `base + layer*512*BLOB + e*BLOB` (`src/core/shadow_experts.cpp:160-164`). Jobs
group the window's tokens per distinct expert (`ExpertJobMulti`, `expert_source.cpp:4963-4994`).
Then one call: `run_split_multi` (shadow/canonical) or `run_split_multi_native` (IQ3_S)
(`:5001-5002`, `:5046-5047`).

**The pool's two phases.** `ExpertPool::run_split_multi` (`src/kernels/cpu/pool.cpp:1064-1101`):
phase 3 = gate/up rows (`mrows_ = n*FF`), host quantizes the intermediates, phase 4 = down rows
(`n*H`). Each `run_phase` (`pool.cpp:1021-1035`): `wait_parked` (all 15 workers spinning,
`:895-914`) → `begin_batch` (packs `epoch|njobs|index` into `head_`, bumps epoch, wakes sleepers,
`:881-893`, `:790-801`) → the host claims jobs too (`host_works_`) → `wait_done` →
`STRATA_POOL_DIET=1` skips the trailing repark (`:1028-1032`; adopted F-227). Workers claim via a
CAS on `head_` (`:870-879`) and compute an equal row range of the phase
(`drain`, `:939-1018`): canonical Q2_0 row kernels (`s2_expert_gu_rows_multi` /
`s2_expert_down_rows_multi`) or, for the native pack, the Q2_0/ggml-cpu row kernels
(`q2_rows_any`, `native_gu_rows`, `:958-996`). Tasks per phase = `phase_tasks` = 16/thread x 16
threads (log: "STRATA_POOL_SPLIT=1: 16 tasks per phase thread (15 workers +1 host…)";
`pool.cpp:1053-1062`).

**The pool's half of the doorbell (flagC / CPU rows).** Outputs go into mapped `h_ymiss_`. After
the dispatch returns, the host fences and raises `*flag = want` (`verify.cpp:3105-3106`, `:3135`).
The GPU's `dm_wait_combine_kernel` (`verify_kernels.cu:1700-1759`) is that wait fused with the
combine: thread 0 spins `*flag >= value` behind the device-plan skip guard (`:1734-1739`), then all
blocks read the pool's rows from mapped host memory (zero-copy over PCIe, `:1744-1747`) and fold
them with the VRAM parts and the shared row. **This is why the pool's slowness appears inside
`dm_wait_combine`'s histogram**: its duration = (time from the GPU reaching the combine node) −
(time the host needed to finish actq + jobs + the pool).

**Where the ~75 µs actually goes.** The entry is byte-bound. `BLOB = 1,382,400` B
(`include/strata/kernels/cpu/expert.hpp:49`; H=2560, FF=640, `:36-37`), so 75 µs = **18.4 GB/s
effective**, against the pool's measured aggregate of ~44 GB/s on the older 6-core target
(`pool.hpp:134-136` comment) and far under a 9950X3D's DDR5 capability. The gap is the shape of the
read, not the arithmetic (the kernel is AVX-512 VNNI `vpdpbusd` + VBMI `vpmultishiftqb`,
`expert.hpp:8-17`; the row accumulator stays float and reduces once, `:17-20`): at ~1 distinct
expert per layer (0.64/layer-window), 1.38 MB is split into ~57-86 KB per thread of **strided**
720-B gate/up rows and 180-B down rows across 16 threads, over 4-KB pages (338 pages/blob → ~21
minor faults per blob even fully cached, filemap fault-around is 16 pages), fronted by two park/
barrier cycles whose workers have almost always gone to sleep (`STRATA_POOL_SPIN_US=100`,
launcher; sleep at `pool.cpp:845-854`). The bytes dominate: the F-360 flip halved bytes
(IQ3_S ~2.05 MB → Q2_0 1.38 MB) and halved the mean (~150 → ~75 µs). The residual non-byte term
(faults + wake) is bounded by what the Q2_0 flip did NOT remove — it is the same term lever (c)
attacks. The pool's own phase counters (`ms_multi_gu/q/down`, `pool.hpp:214-216`) plus a
`/proc/vmstat` pgfault delta would split it exactly; do that before tuning.

## 2. The tail: what a 9.8 ms single-layer stall can be

Ranked by fit to the code:

1. **Major page faults on the 31.64 GiB shadow mapping — the prime suspect.** The store is a
   `MAP_SHARED` file mmap (`shadow_experts.cpp:139-142`; log: "…x 1382400 B = 31.64 GiB"). Nothing
   ever warms it: the shadow branch skips the miss prefetch (`expert_source.cpp:4887-4889`), and
   the live RouterLookahead warms the **native pack's** pages instead (io workers pread
   FileExpertSource spans, `:971-1045`; log: "routing-aware prefetch on (the next layer's router,
   file tier)"). An evicted blob faults back from NVMe in 64-KB fault-around steps: 0.3-1.5 ms per
   blob, and a layer that misses 2-4 cold blobs stalls milliseconds — the 9.8 ms `dm_wait_combine`
   max. Eviction is plausible, not hypothetical: the box holds the 31.64 GiB shadow + the ~47 GiB
   native pack mapping + KV/PLE mappings in one page cache (165 GiB cached at review time,
   `free -g`), and serving churn evicts.
2. **Worker wake after park.** With SPIN_US=100 the workers are asleep in nearly every layer
   (`pool.cpp:845-854`); each CPU layer pays `publish()`'s mutex+`notify_all` (`:797-800`) and
   15 futex wakes collected by `wait_parked` (`:895-914`). Typical cost ~10-50 µs — this is p99
   (169 µs), not the 9.8 ms, unless a worker's core was preempted (aux threads are pushed to
   CPU 16+ per the boot line "aux cpus: … go to CPUs 16", desktop noise otherwise).
3. **Cohort sharing of one pool.** `STRATA_COHORTS=2` clones the dispatch, not the pool:
   `coh_b_drive = drive` (`src/program/generate.cpp:12130`) keeps `drive.d.pool = &pool`
   (`:6909`) — two cohort windows serialize on one batch barrier. A conc-lane tail source; not the
   solo trace's 9.8 ms.
4. **Host-core contention.** The host pins to logical 0 (`--host-core first`, boot log), the
   default IRQ target; every GPU interrupt delays the spin/dispatch there. `HostCore::Last/
   Sibling` exists for exactly this (`pool.hpp:107-113`). Cheap A/B, sub-100 µs scale.
5. **NUMA: ruled out.** 9950X3D is one package; the multi-socket ordering in
   `pool.cpp:88-135` never fires on this box.
6. **Locks: clean.** The only token-path mutex is `sleep_mu_` (`pool.cpp:798`), held
   microseconds; `head_`/`done_`/`epoch_` are each on their own cache line (`pool.hpp:313-320`).
   mmap_lock serializes first-touch fault storms only per-page.

Read of the histograms: ~93% of `dm_wait_combine`'s 1391 ms sits above the p50 floor
(47768 x 2.2 µs = 105 ms) — **~1.0 ms/window is tail**, plus ~0.4 ms/window in
`wait_flag_ge_planned`'s tail. That ~1.4 ms/window, together with the spin kernels' overlap of the
pool's 2.24 ms host stage, is the ceiling for every lever below.

## 3. The adaptive tier and the prediction hooks

**Residency.** `ExpertCache` owns slots + the `residency_` table (`expert_cache.hpp:91-100`,
`:154`, `:256`); `host_res[]` mirrors it into the dispatch. Decisions run per adapt round at
request granularity, on a thread under the window (`STRATA_ADAPT_EARLY=1`, F-332), with victim
protection: `STRATA_ADAPT_PINCAP=48` (per-layer top-48 profile ranks pinned),
`STRATA_ADAPT_MIN_AGE=2`, `STRATA_ADAPT_DORMANT=64` (`generate.cpp:10259-10278`). Live churn:
"adaptive tier: 139 rounds, 4635 experts swapped in" — it chases routing drift between requests,
it does not chase within a window.

**The expert profile.** `--expert-profile data/expert-profile.bin`: 24,576 ranked (layer, expert)
pairs by long-run routing frequency (`expert_cache.hpp:42-55`, tools/make_profile.py format). It
preloads the VRAM tier at boot, feeds PINCAP's pin order, and is the `prior` for the learned heat
saved by `--expert-profile-save` (#477). `STRATA_PREDICT_CREDIT=1.0` mixes predictor credit into
the same ranking (`generate.cpp:10603`).

**How many misses are "expected".** 0.64-0.72 entries/layer-window on the live lane (log lines:
"CPU experts 0.67…0.72", "routed entries per window: VRAM 97-98%, CPU 1.9-2.7%, PCIe 0.2-0.5%").
This is the residual after two existing prunings: residency-biased routing (ROUTE_RESIDENT=0.75)
and tail-skip rank >= 7.

**Is there a prediction hook? Yes — three, one unplugged.**
- `RouterLookahead` is live: an aux thread recomputes layer+1's router from this layer's MoE input
  (`expert_source.cpp:2922-3053`, submitted per layer at `:4548`, armed at `generate.cpp:6916-6951`)
  and feeds (i) page warming with confidence+budget admission (`warm_scored`, `:851-882`) and
  (ii) adapt-tier credits. **But it warms the native pack, which the misses no longer read** — the
  shadow store has no warming path at all. The prediction exists; its consumer was switched out
  under it by F-360.
- `ForesightSwap` is the designed miss-killer: predicted (or repeated) misses are DMA'd into
  per-layer VRAM swap slots by a filler thread (`:6077-6101`, `:6171-6221`), and `take()` at plan
  time turns them into kind-0 GPU groups (`:4763-4764`, `:6103-6138`). It is **off twice over** on
  this lane: `STRATA_FS_SLOTS` unset (`:6238-6240`) and hard-refused beside
  `STRATA_VERIFY_DEVICE_PLAN=1` (`:6241-6245` — the device plans resident layers from the
  residency table and would not know a foresight slot).
- Mapping the whole IQ3_S pack into system RAM for O(1) access: **already done** — the 36.95 GiB
  pinned complement holds every expert the GPU tier does not (`--resident-experts`). It did not
  remove the misses and cannot: RAM residency fixes *latency*, the pool is bound by *bytes*, and
  the misses now deliberately read the smaller Q2_0 store (`:4954`). More RAM is not the lever;
  fewer bytes and fewer faults are.

Could the 0.64 go to ~0? Only by prediction (ForesightSwap) or by routing away more rank — the
profile+adapt cannot hold the last ~2% (full residency needs ~98 GiB of IQ3_S experts against a
32 GiB card). A predicted miss served from VRAM costs ~1.38-2.05 MB over PCIe x16 (~30-40 µs of
copy-engine time, off the critical path) against ~75 µs of pool time plus its share of the wait
kernel.

## 4. The levers, what each actually changes

- **STRATA_POOL_DIET=1** (`pool.hpp:269-272`, `pool.cpp:1028-1032`): skips the trailing repark in
  `run`/`run_phase`; the epoch-carrying claim makes it latency-shaping, not correctness. Adopted
  (F-227, +4.06% M1). Nothing left to take here.
- **STRATA_POOL_SPIN_US=100** (`pool.hpp:236-243`, `pool.cpp:727-728`): parked workers spin 100 µs
  then sleep. Current value keeps cores quiet between layers at the price of a wake per CPU layer.
  F-346's =2000 retest is unpriced on this lane; the clean fix is the WAITPKG arm — but
  `STRATA_POOL_UMWAIT` is MSVC-only (`pool.cpp:41-63`); a Linux UMONITOR/UMWAIT port is a real
  (small) tail lever.
- **STRATA_POOL_SPLIT=1** (16 tasks/thread; `pool.hpp:272-284`): the Maya CPU_SPLIT pattern —
  shrinks what one stalled worker pins. Armed on the live lane. With byte-bound streams all
  workers finish together; little left.
- **STRATA_ADAPT_*** (`generate.cpp:1582-1623`, `:10259-10278`, `verify.hpp:43`, `:465`): NOWAIT=1
  (window starts before adapt copies land), LAG=2 (decision lag), OVERLAP (off — refused beside
  `STRATA_DMA_BATCH`, `generate.cpp:10316`), EVICT_SYNC, CHECKRES (stale-residency probe,
  `expert_source.cpp:4848`), JOB_CPU (adapt thread pin), CALLBACK_WARMUP, PINCAP/MIN_AGE/DORMANT
  (victim protection), EARLY (scan under the window). Tier sync policy, already tuned by F-332.
- **peer_experts.cpp / remote_experts.cpp**: both are second-device paths, both unused on this
  single-5090 lane. `PeerExperts` = `--peer-device` expert tier on a second card writing rows into
  mapped host memory (`include/strata/core/peer_experts.hpp:1-33`). `RemoteExperts` = static
  profile-filled helper-GPU caches (`--remote-expert-*`, `include/strata/core/remote_experts.hpp`;
  measured on 2x RX 6900 XT per `expert_source.cpp:4628-4631`). Neither appears in the boot argv.
- **Project Maya**: not vendored — identified as a public fork of this engine and banked as intel
  (`SVG-CUSTOM.md:11248`, `:11258`). Transferables: CPU_SPLIT **built** as STRATA_POOL_SPLIT
  (parity-proven, `campaign/pool-split-notes.md:6,13,93`; `pool.hpp:275`); PCIE_SHARE ≈ the plan
  DMA lane already in-tree (`GpuPlanSink.ptr2` + `fetch_dma`, `verify.cpp:3316`); row-reuse
  (CPU_MT) already ours; AVX-512 compiled. Open Maya item: thread-count recalibration (15 →
  12/10 workers; SMT-sibling fighting, `SVG-CUSTOM.md:11265`), retest-cheap.
- **STRATA_ROUTE_TAIL_SKIP / STRATA_ROUTE_RESIDENT**: the routing-side miss pruners (§1), already
  armed; raising either changes tokens and needs the quality gate.

## 5. Ranked, in ms/window against the measured window

Ceiling first: the GPU's spin kernels sit ~1.4-1.6 ms/window above their launch floor, and the
pool's 2.24 ms host stage is what that wait is made of (partially hidden under VRAM expert work).
No host-side lever can win more than ~1.5 ms/window solo (~+10-11% t/s at 2.74 tok/window);
levers (a) and (c) draw on the same budget and partly overlap.

**(a) Kill the tail — make the shadow store fault-free. Expected 0.6-1.0 ms/window.**
Nothing warms or pins the 31.64 GiB mapping (§2.1); every evicted blob is an NVMe round trip
inside a layer. Two variants, both no-output-change:
  - ops-only: copy `q2_0/experts.bin` to `/dev/shm` (93 GiB tmpfs, 71 MB used — fits) at boot and
    point `STRATA_SHADOW_EXPERTS` there; shmem pages do not migrate to disk under this swap
    posture (7 GiB swap, swappiness 10), and tmpfs is THP-eligible.
  - in-tree (~20 lines in `shadow_experts.cpp::arm`): `mmap` + `MAP_POPULATE` (+ `madvise(MADV_HUGEPAGE)`
    where eligible), or `madvise(MADV_WILLNEED)` over the file at boot — but note `ulimit -l` is
    8 GiB, so `mlock` of 31.64 GiB needs a limit raise; WILLNEED alone re-warms only until the next
    eviction pressure.
  Cheapest true fix for the recurring case: wire the **existing** RouterLookahead want-list to the
  shadow mapping — `madvise(MADV_WILLNEED)` on `[shadow::blob(l,e), +BLOB)` per predicted expert;
  the mincore-based skip already exists (`resident_sampled_n`, `expert_source.cpp:1035`).
  Evidence for the size: ~1.0 ms/window of `dm_wait_combine` above floor, of which the fault-cold
  layers are the bulk of the mass above p99.

**(b) Kill the misses — re-plug ForesightSwap. Expected 0.4-0.8 ms/window, the largest single
ceiling but the highest cost.** The mechanism is complete end-to-end (§3): predict → filler
thread → `cudaMemcpyAsync` into per-layer slots → `take()` makes the plan call it a VRAM group.
At budget 32 copies/window (~44 MB, ~1 ms of copy-engine time, off-path) and a plausible 50-70%
recall on 0.64 entries/layer-window, most of the pool's wait contribution disappears and the
`actq`/`jobs` host stage shrinks with it. Blockers, in order: the hard guard at
`expert_source.cpp:6241-6245` (device-plan integration — the device planner must learn to treat a
foresight slot as resident, or this lane runs the host-planned graph for non-fully-resident
layers), then slot sizing (48 layers x 2 slots x 2.05 MB = ~197 MB, trivial against the 2 GiB
`--vram-reserve-mib`). This is a build-plus-boot-pair, not a flag.

**(c) Make per-entry compute faster. Expected 0.3-0.6 ms/window, mostly by removing fault/TLB/
wake overhead rather than arithmetic.**
  - AVX-512: already there (VNNI+VBMI, `expert.hpp:8-17`) — no head to claim.
  - Q2_0 vs IQ3_S decode: already on the cheaper one; the 2.0x-byte halving is banked (F-360).
  - Row reuse: dead end on this workload — 2.74 tok/window over ~31 distinct entries = nt ~ 0.9;
    tokens almost never share an expert per layer, so `run_split_multi`'s sharing is already
    idle.
  - What remains is exactly the non-byte term from §1: 4-KB-page minor faults (~21/blob), strided
    720-B row streams, and the park/wake barrier x2 per layer. The (a) RAM-resident+THP variant
    removes the fault term; the Linux UMWAIT port (or the F-346 SPIN_US=2000 retest) trims the
    barrier term; `--pool-workers` down to ~12 (Maya's SMT observation) and `--host-core last`
    (`pool.hpp:107-113`) are two cheap A/Bs in the same neighborhood.

**One explicit non-lever:** the PCIe share. `PcieModel`'s own fit already recorded that on this
box "the pool is DRAM-bound (moving experts to PCIe left the pool's time where it was, d ~ c: few
move)" (`include/strata/core/expert_source.hpp:389-394`) — `--pcie-frac 0.35` mostly moves
0.1 entries/layer-window of pinned misses; do not expect window time from re-tuning it.

## 6. Probes to close the residual unknowns (no benchmark run needed to start)

1. Split the 75 µs: log `ms_multi_gu/q/down` (`pool.hpp:214-216`) per window beside a
   `/proc/vmstat` `pgfault`/`pgmajfault` delta — one request's worth, read-only.
2. Confirm the tail's fault signature: STRATA_IO_STATS-style mincore sampling on the shadow blobs
   of layers whose `dm_wait_combine` exceeded 1 ms (the per-layer wait histogram already exists in
   the dm stats path, `verify_kernels.hpp:110-124`).
3. `STRATA_LOOKAHEAD_STATS=1` on a measurement boot would print the predictor's recall/precision
   and "covers X of the misses" (`expert_source.cpp:2895-2918`) — the number that prices lever (b)
   before any code is written.
