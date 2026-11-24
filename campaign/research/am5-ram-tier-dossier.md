# Research dossier: AM5 host RAM tier — bandwidth, CCD pinning, hugepages, hot-set sizing, allocator (Magma qwen4exp, 9950X3D)
Method legend: **[M]** measured locally on the target machine this session; **[F]** source fetched and read directly; **[S]** title/snippet verified only; **[C]** curl-fetched. Extends the two prior dossiers into the host tier.

## 0. Platform ground truth [M] (differs from the scenario spec)
dmidecode (passwordless sudo works) shows **4×48 GiB DDR5-5200 MT/s**, MemTotal 184 GiB — the box has **192 GB, not 96**. MPG X870E Carbon, glibc 2.44, governor=performance, **NUMA node(s): 1** (UMA; numactl not installed). V-cache CCD = **CPUs 0-7 (+SMT 16-23), L3 98304K**; plain CCD = **CPUs 8-15 (+24-31), L3 32768K** (per-CPU /sys cache index3). amd_x3d_vcache driver present (AMDI0101:00). THP: enabled=[madvise], defrag=[madvise], khugepaged on (10 s scan). **HugePages_Total: 0**; ulimit -l = unlimited both levels; CapEff 0x1fffffffff includes CAP_IPC_LOCK. A 64 GiB qemu VM + java were live during benchmarks → MT numbers below are depressed; single-core numbers are stable. Math below covers both the 96 GiB spec box and the measured 192 GB box.

## 1. DDR5 bandwidth: single-thread vs multi-thread
**Measured here [M]** (OpenMP STREAM-class triad, 100M×3 doubles, best-of-N; STREAM's own reference table: TRIAD moves 24 B/elem [C, cs.virginia.edu/stream/ref.html]):
- 1 core CCD0 (cpu4): triad **25.9 GB/s**, copy 22.9; 1 core CCD1 (cpu12): triad 25.4, copy 22.3. Hugepages: no effect at 1T (25.1).
- 4T per CCD: identical 31.3 both CCDs. 8T/CCD interleaved ×3: **CCD1 36.0-37.1 vs CCD0 30.0-31.2** — reproducible ~15% penalty on the V-cache CCD at saturation.
- 16T (0-15, no SMT): 33.5-36.9; 32T with SMT: 32.3 — **SMT hurts ~10% at DRAM saturation**. THP vs 4K at 16T: no consistent win under load.
**Published MT [F]**: ixbt 9950X3D review (AIDA64): DDR5-6000 CL36 → read **77,693** / write 77,459 / copy 67,513 MB/s; DDR5-5200 → 65,398/68,876/59,825; RAM latency 79.9 ns @6000, 88.9 @5200; L3 latency 11.9-12.0 ns (X3D) vs 10.6 ns (9950X); verdict: AMD stays below 70 GB/s vs theoretically >80 at 5200. OC ceiling [F]: Silicon Index Labs (SPD-Z) 9950X3D DDR5-8400: **read 101.8, write 69.7 GB/s**, idle latency 59.9 ns, loaded +57 ns at 30 background threads. **Resolution of the 60-110 GB/s claim**: stock AM5 = 65-78 GB/s MT read (kit-dependent), triad below that (write path); >100 needs OC kits. Single-thread ≈ **23-26 GB/s** — one helper thread cannot feed 1.24 GiB/token (needs ~19-22 GB/s minimum, i.e. one saturated core). This box's theoretical peak: 5200 MT/s ×2ch ×8 B = 83.2 GB/s [M].
**Fabric asymmetry [F]** (HardwareLuxx Zen-5 deep-dive): each CCD reads 32 B/cycle from the IOD but writes only 16 B/cycle; same-CCD core-to-core 25-30 ns; **Granite-Ridge inter-CCD ~250 ns** (vs ~80 ns on 7950X — AMD declined to explain). Triad-style traffic self-throttles; keep producer and consumer of a buffer on one CCD.

## 2. CCD topology, DMA/engram placement, pinning recipe
Map (measured): **0-7/16-23 = 96 MB L3 CCD; 8-15/24-31 = 32 MB CCD.** Kernel 6.13 ships the amd_x3d_vcache optimizer [F, kernel.org ABI doc]: amd_x3d_mode="cache" re-ranks the larger-L3 CCD's cores as preferred; "frequency" re-ranks the faster CCD (AMDI0101:00 present on this box [M]). It only steers the scheduler — pin explicitly.
- **Engram directory probe thread → cpu0** (sibling cpu16 left unassigned); PLE page-cache warm/steal thread → cpu2. Directory stays resident in 96 MB L3; L3 probe 12.0 ns vs ~80 ns DRAM [F ixbt] — a 24,576-slot × 32 B directory (~0.77 MiB) plus hot rows effectively never leave L3 under single-thread polling.
- **Bulk movers → CCD1 physical cores**: copy-engine pump + io_uring NVMe shard workers on cpu8-11, fill workers on cpu12-15. CCD1 measured faster at saturation (§1), and DMA completion/doorbell traffic cannot then evict the directory across the ~250 ns fabric hop [F hardwareluxx]. SMT siblings 24-31 left to the OS.
- Device side: documented zero-copy path is cudaHostAlloc(Mapped) / **cudaHostRegister + cudaHostGetDevicePointer** — "Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister" [F, CUDA runtime API]. Device polling of host RAM pays full PCIe latency per probe; poll a device-mirrored doorbell word, not the directory.
- Pinning mechanics: AM5 is UMA — numactl/--membind placement rules **do not apply** (1 NUMA node [M]); taskset -c per thread is the whole story. SMT-sibling avoidance is measured, not folklore: 32T SMT triad < 16T physical (§1). Independent trap warning: ACM DOI 10.1145/3695794 documents results invalidated by pinning to the wrong X3D CCD [S-snippet, PDF at memsys.io].
Recipe: taskset -c 0 → dir-probe; taskset -c 2 → ple-warm; taskset -c 8-11 → nvme+copypump; taskset -c 12-15 → fill; then: echo frequency > /sys/bus/platform/drivers/amd_x3d_vcache/AMDI0101:00/amd_x3d_mode (unpinned system work drifts to the faster CCD, off the 96 MB L3).

## 3. Hugepages & pinning for 20+ GiB of host pools
State [M]: THP madvise-only, empty hugetlb pool, unlimited memlock + CAP_IPC_LOCK. Kernel semantics [F, docs.kernel.org transhuge/hugetlbpage + mlock.2]: THP defrag "always" stalls in direct compaction; "defer" wakes kswapd/kcompactd in background; **"defer+madvise"** is the recommended compromise for allocation latency. mlock.2: the RLIMIT_MEMLOCK cap "is not enforced if the process is privileged (CAP_IPC_LOCK)"; verify via VmLck in /proc/self/status. hugetlb pool: /proc/sys/vm/nr_hugepages = persistent pages; runtime reservation **works here [M]** (16 pages instant; 1024×2 MiB reserved then reverted mid-load), but 20+ GiB should be boot-reserved to dodge fragmentation (kcompactd churn observed [M]). pdpe1gb in cpuinfo [M] → 1 GiB pages available.
Fault-in cost is not the differentiator [M]: 2 GiB touched+locked = 0.125 s via anon+MADV_HUGEPAGE vs 0.122-0.143 s via hugetlbfs (16 GiB/s under VM load). hugetlbfs buys **determinism**: no collapse failures, no khugepaged timing, no reclaim-splitting of a resident pool — matters for the DMA staging ring and directory, not the elastic victim tier. mlock of resident pages is instant [M, 16 GiB].
CUDA wiring [F]: mmap the pool (hugetlbfs MAP_SHARED, or anon+MADV_HUGEPAGE), touch, mlock, then **cudaHostRegister(ptr,size,flags)** "Registers an existing host memory range for use by CUDA". cudaHostAlloc alone never gives hugepage backing.
Config: boot cmdline default_hugepagesz=1G hugepagesz=1G hugepages=8 (staging+directory; 1 G pages = 8 PTEs). Victim tier: single anon mmap, MADV_HUGEPAGE at setup, pre-fault, mlock, cudaHostRegister once. THP defrag → defer+madvise. systemd unit: LimitMEMLOCK=infinity (this box's shells already unlimited [M]).

## 4. Hot-set arithmetic
Slot = 2.637 MiB. **96 GiB spec box**: OS 4 + driver/staging 3 + PLE page-cache budget 24 + **victim tier 48 GiB = 18,627 slots** + headroom 17 = 96. Alternative: caching the whole 47.75 GiB PLE table → victim ~20 GiB = 7,776 slots. Cap the PLE cache explicitly — it is reclaimable page cache; cgroup-v2 **memory.max is the knob** ("Memory usage hard limit... the main mechanism" [F, cgroup-v2 doc]; page cache charges against it). **192 GB measured box**: 62.1 GiB = the entire 24,576-slot set fits in the victim tier alone, plus ≥50 GiB PLE cache — NVMe becomes cold-start-only. **Inconsistency flagged**: scenario says "full set ~15.2 GiB" (≈5,776 slots) but 512×48 layers×2.637 MiB = **62.06 GiB**; if 15.2 GiB is right (~12 expert layers?), a 24 GiB victim holds the whole set with 1.6× slack. **NUMA-less note**: UMA kills the classic rules — no membind, no per-node page-cache skew, no zone_reclaim_mode tuning; only CCD-level placement survives (§2).

## 5. Allocator noise for fill workers
glibc "creates additional arenas if mutex contention is detected. Each arena is a large region... managed with its own mutexes" [F, malloc.3]; the arena hard limit is "usually calculated as a multiple of the number of available [cores]" via M_ARENA_TEST [C/F, mallopt.3 verified on-box (glibc 2.44) and via man7 curl]. 23 free-running fill threads drift toward dozens of arenas: RSS bloat, TLB pressure, trim/compact jitter. Recipe: **one static pool, zero per-slot malloc** — slots at fixed 2.637 MiB stride (2 MiB-aligned so a slot never straddles a hugepage), per-thread private scratch slabs inside the same reservation; libc traffic capped with mallopt(M_ARENA_MAX, 4) at startup. If libc malloc must serve the hot path, set MALLOC_TRIM_THRESHOLD=-1 so the arena does not madvise memory you just prefaulted.

## Transfer to Magma
1. Pin bulk movers (NVMe io_uring, copy pump, fills) to **cpu8-15**; directory probe on **cpu0**, PLE warm on cpu2; leave 16-31 to OS; set amd_x3d_mode=frequency. Never place producer/consumer pairs on different CCDs (~250 ns cross-CCD).
2. Victim tier: **48 GiB (96 GiB box) / whole-set 62.1 GiB (192 GB box)** via anon+MADV_HUGEPAGE+mlock+cudaHostRegister, 2 MiB-aligned slot stride; staging ring + directory on boot-reserved 1 GiB hugetlbfs; THP defrag=defer+madvise; cap PLE page cache with cgroup memory.max (~24 GiB) instead of trusting LRU.
3. Bandwidth reality: one host core ≈ 26 GB/s cannot feed 1.24 GiB/token victim-to-staging (~19 GB/s floor + overhead) — keep ≥2-3 CCD1 cores on copy duty and overlap with temporal prefetch (prior dossier's ProMoE / vLLM #38256 levers). Measured host-copy ceiling 30-37 GB/s MT under load; PCIe H2D (50-64) stays binding, so RAM bandwidth is not the wall — contention is (measured 36→31 GB/s swings under load).
4. mlock everything resident; this box already unlimited + CAP_IPC_LOCK [M]; production units need LimitMEMLOCK=infinity.
5. Resolve the 15.2-vs-62.1 GiB expert-set discrepancy before freezing tier sizes; the answers differ 4×.

## Source table
| Source | ID/URL | Status | Key fact |
|---|---|---|---|
| Local machine probes | lscpu, dmidecode, /sys, /proc | [M] | 4×48 DDR5-5200; UMA; CCD map; THP/hugetlb state |
| Local STREAM-class runs | /tmp/stream2.c, faultcost.c | [M] | 1T 25.9; 8T CCD0 30-31 vs CCD1 36-37; SMT −10%; 0.12 s/2 GiB fault-in |
| STREAM reference | cs.virginia.edu/stream/ref.html | [C] | Triad = 24 B/elem |
| ixbt 9950X3D review | ixbt.com/platform/amd-ryzen-9-9950x3d-zen5-review.html#3 | [F] | AIDA 77.7/77.5/67.5 @6000; 79.9 ns; L3 12.0 ns |
| Silicon Index Labs SPD-Z | siliconindexlabs.com/benchmarks/md7d9k55px17st2kxe5jmeh65x8f8pcc | [F] | 101.8 read @8400; +57 ns loaded @30T |
| HardwareLuxx Zen 5 | hardwareluxx.ru .../55956-testy-zen-5...html | [F] | IF 32R/16W B/cyc; inter-CCD ~250 ns |
| Kernel THP doc | docs.kernel.org/admin-guide/mm/transhuge.html | [F] | defer+madvise semantics |
| Kernel hugetlb doc | docs.kernel.org/admin-guide/mm/hugetlbpage.html | [F] | nr_hugepages persistent pool |
| cgroup-v2 doc | docs.kernel.org/admin-guide/cgroup-v2.html | [F] | memory.max hard limit |
| mlock(2) | man7.org/linux/man-pages/man2/mlock.2.html | [F] | CAP_IPC_LOCK bypass; VmLck |
| malloc(3) | man7.org/linux/man-pages/man3/malloc.3.html | [F] | arenas on contention |
| mallopt(3) | man7.org .../man3/mallopt.3.html + local man | [C/F] | M_ARENA_MAX/TEST; limit ∝ ncores |
| CUDA Runtime API | docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html | [F] | cudaHostRegister / GetDevicePointer |
| amd_x3d_vcache ABI | kernel.org/doc/Documentation/ABI/testing/sysfs-bus-platform-drivers-amd_x3d_vcache | [F] | cache/frequency modes; 6.13 |
| Large-cache latency paper | DOI 10.1145/3695794 (memsys.io PDF) | [S] | X3D wrong-CCD pinning trap |
