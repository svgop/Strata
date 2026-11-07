---
type: digest
id: preboot-qwen-next
created: 2026-11-07
status: evidence
topic: What to optimize NEXT after the 2 MiB hugepage move (registration speed, conc -4.4%, KV shadows, IQ4_XS 94 GiB pool)
scope: read-only research; no repo files modified besides this digest. Mixed local source read + web research, same day.
---

## 0. Drift found first: this boot's pool is 47.6 GiB, not the armed 88 GiB

`/proc/sys/vm/nr_hugepages` = **24369** pages (47.6 GiB), `HugePages_Total=Free=24369` (engine down, store copy absent from the mount). The arming file `campaign/91-strata-hugepages.conf` (= `/etc/sysctl.d/91-strata-hugepages.conf`) says **45056 (88 GiB)**. `/proc/cmdline` carries no hugepages args, so the reservation is systemd-sysctl's job — and it landed **silently partial** this boot (the exact failure the file's own comment names for runtime re-arms, `HOSTMEM-NO-REBOOT.md:94-95`).

Consequences for the same-window numbers quoted in the task:

- 47.6 GiB covers **one** big organ (GSQ complement 37.6 or UD arena 39.4) but NOT both together (77 GiB) and not organ+store (75.6+). Any pair/conc leg where a second organ arrived after the pool drained fell to the THP/4K fallback — a **mixed-class confound** for the "pair flat, conc −4.4%" verdict. Those runs are void until pool coverage per engine is proven per arm (§4).
- GSQ/UD solo gains (+7.2% / +4.9%) are plausible on a short pool (single organ fits); the conc verdict is not.

**Action R0 (free, before anything else):** every future measurement boot asserts `HugePages_Total` == armed value and per-engine `HugePages_Free` deltas + boot-log ARENA PAGES lines (`pinned.cu:424-439`) + `hugepage_report` lines (`expert_source.cpp:3726`, `layer.cpp:603`). Repair procedure if short at runtime is already written (`HOSTMEM-NO-REBOOT.md:94-95`): engine down → `compact_memory` → raise in steps → read back. Law-17 witness: the log line itself.

---

## 1. WEB: registering tens of GiB — what vLLM/SGLang/TRT-LLM/NCCL/UCX actually do

**Nobody eagerly registers tens of GiB at boot.** Every mature system amortizes or avoids the call; the documented speeds are consistent with ours being ~1-2 GB/s single-threaded:

- **vLLM**: pinned host pools come from PyTorch's `CachingHostAllocator` (`cudaHostAlloc`-class), hidden at init; Sleep Mode L1 keeps weights in pinned host RAM and wake-up is a plain H2D replay of pinned memory (pinned transfers saturate PCIe, ~52-60 GB/s effective on Gen5 per the May-2026 host-GPU bottleneck study). Sleep-mode docs: https://docs.vllm.ai/en/latest/features/sleep_mode ; Zero-Reload blog: https://vllm.ai/blog/2025-10-26-sleep-mode . Known wart at scale: `--cpu-offload-gb` uses ~1.9x the requested host RAM. Snowflake's semi-persistence work claims 5.6-19.9x faster swaps than vLLM L2 — evidence that first-pin cost is accepted and never re-paid: https://www.snowflake.com/en/blog/engineering/semi-persistence-gpu-model-swapping
- **SGLang**: wrote its own pinned allocator (avoids power-of-two rounding) and exposes hugepage backing via `SGLANG_HUGEPAGE_SIZE` (hugetlb/THP-aligned prefault) — the same "reserve pool, prefault, then pin" shape we built (recorded with citations in `.docs/research/digest/2026-11-07-hostmem-*.md`, `HOSTMEM-NO-REBOOT.md:35-37`). SGLang hit `cudaHostRegister` "invalid argument" fixed by a kernel upgrade: https://github.com/sgl-project/sglang/issues/6285
- **TensorRT-LLM**: allocates once at boot, never churns pools per load (`HOSTMEM-NO-REBOOT.md:44-46`).
- **NCCL/UCX**: both built **registration caches (rcache)** precisely because registration is expensive — register once per buffer, cache the handle, never re-register. UCX's `uct_cuda_copy_mem_reg` wraps `cudaHostRegister` and had its own invalid-argument class (https://github.com/openucx/ucx/issues/4393). UCX GPU-support docs describe the rcache as the mitigation for repeated expensive registration.
- **Multi-threaded registration: a documented dead end.** PyTorch issue #139: `cudaHostRegister` = page-locking syscalls **plus** registration in CUDA's own structures under a driver-internal lock, so it serializes (https://github.com/pytorch/pytorch/issues/139). NVIDIA forum, "strange behaviour of cudaHostRegister on multiple threads" (Jun 2024): https://forums.developer.nvidia.com/t/strange-behaviour-of-cudahostregister/294999 . There is a Nov-2025 feature request "Reduce the cost of cudaHostRegister" — still a request, not a fix.
- **CUDA 13.x**: released Aug 2025; release notes contain **no** `cudaHostRegister` performance entry (checked via search of the official notes; the only pinned-memory mention is best-practice guidance). No Blackwell-specific registration improvement surfaced. https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- **ODP** (on-demand paging): a GPUDirect-RDMA/datacenter-NIC + platform feature; not applicable to an RTX 5090 PCIe pinned path. Named boundary, no probe needed.
- **Speed variance**: NVIDIA forum thread "Why cudaHostRegister can have a huge difference in performance" (Sep 2024, https://forums.developer.nvidia.com/t/why-cudahostregister-can-have-a-huge-difference-in-performance/308105) attributes large per-call variance to memory block/alignment class — consistent with our THP-faulted-mapping stall (minutes) vs pool-faulted (tens of s). A May-2024 thread measured ~13% r/w asymmetry of `cudaHostRegister`ed vs `cudaHostAlloc`ed memory — the same class our zero-copy organs live on.

**Verdict for us**: the research-endorsed shape is exactly what we have — one pool, fault-first, ONE registration call, register once and never churn. What the literature adds: parallel registration buys nothing (driver-serialized), chunk granularity buys almost nothing (cost is per-page, not per-call), and there is no driver/toolkit update to wait for.

**Priced/falsifiable probe P1 (S)**: time `cudaHostRegister` alone over the faulted 37.6 GiB pool complement (wrap the call at `expert_source.cpp:3657` in a steady_clock print — log-only, no behavior change; `STRATA_PIN_GUARD=2` forces the path for a test boot). Expected 19-38 s at the documented 1-2 GB/s class. If it lands < 8 s, the +48 s GSQ boot cost lives elsewhere (store copy, faults) and §3's optimization target changes.

---

## 2. WEB: cudaHostRegister over MAP_HUGETLB — known failure modes vs our 5/5 evidence

- **The loudest public bug does NOT cover us**: open-gpu-kernel-modules #732 — `cudaHostRegister` returns "invalid argument" on memory mapped with `remap_pfn_range` since driver 550.54.14 (physical-continuity check added in `get_io_ptes`; open, no NVIDIA reply): https://github.com/NVIDIA/open-gpu-kernel-modules/issues/732 . That is **device I/O memory**, not anonymous RAM. Our path is `MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB` (`pinned.cu:256-257`, `hugepage.hpp:86-87`) — anonymous reserved-pool RAM, a class the driver has accepted on all public evidence and we accept 5/5 boots at 37-39 GiB.
- **File-backed mappings are the historically fragile class**: a 2011-era NVIDIA forum answer states registration only works on anonymous/`cudaHostAlloc`-style memory, not arbitrary `mmap` (https://forums.developer.nvidia.com/t/cudahostregister-crash-or-poor-performance-unknow-error-30-in-kernel-for-64bit-host-operating-syst/26392). We never register the hugetlbfs **store** mapping (miss path is CPU-side only; `shadow_experts.cpp` only had `MADV_COLLAPSE`, retired with the shmem route). Keep it that way, or test that leg separately.
- **Driver↔kernel version sensitivity**: sglang #6285 (fixed by kernel upgrade) and UCX #4393 show registration failures that move with kernel/driver pairs. Our 5/5 evidence binds to **kernel 7.2.12-300.fc45 + the installed driver** — record the pair in the boot record and re-test one boot after any driver upgrade (named boundary, one-line witness).
- **Registration splitting hugepages**: not a hugetlb failure mode — pool pages are reserved, unswappable, unmovable; GUP cannot split them. The split risk class is **THP** (pinning a THP-backed region), which is exactly the open risk HOSTMEM-NO-REBOOT M3 names ("witness: AnonHugePages before/after register", `HOSTMEM-NO-REBOOT.md:74-76`). Our pool route sidesteps it by construction.
- **IOMMU/DMA at 2 MiB**: F-201-zen5 #6 grounds the accounting (`campaign/research/F-201-zen5-host-agent.md:8`): 4 KB = ~10.5M IOMMU PTEs (~40 MB tables) vs 20,480 at 2 MiB; IOTLB miss ~330 ns (Farshin, PeerJ CS 2023; up to 20% loss at 4 KB, ~5% recovered at 2 MB). No forum-documented consumer-board IOMMU failure at 2 MiB registration found. The measured solo gains (+4.9% UD, +7.2% GSQ) are the proof on this board. Free witnesses to add: `dmesg | grep -i amd_iommu` page-fault/event counters after a bench boot; and the launcher's systemd scope must not cap hugetlb (`cgroup2 ... memory_hugetlb_accounting` is mounted on this box — check the scope's `MemoryHugetlbMax` is unset). One RLIMIT note: the Linux NVIDIA driver pins **past** RLIMIT_MEMLOCK (verified here with a 4 GiB register under `ulimit -l 1 GiB`, `pin_budget.hpp:16-17`), so memlock is not a failure mode for us.
- **Legacy thread caution**: cudaHostRegister crash/poor-performance reports on 64-bit Windows/classic kernels exist (forum 26392 above) — irrelevant to our Linux bare-metal pair, recorded for the archive.

---

## 3. LOCAL: what the paced registration loop does today, and what to change when backing == hugetlb

**Current mechanics** (all read this session):

- `pin_register` = thin wrapper over `cudaHostRegister`, no chunking, flags always `cudaHostRegisterPortable|cudaHostRegisterMapped` (`include/strata/core/pin_budget.hpp:101-109`; call sites `pinned.cu:486,506,615,672`, `expert_source.cpp:3657,3696`).
- Paced loop (`src/core/expert_source.cpp:3628-3683`): steps of **kStep = 1 GiB** (`:3635`); per step: `available_memory_bytes` + `pin_step_fits` gate against `pace_reserve` (default max(headroom, 2 GiB), `:3505-3509`), then a **4 KiB-stride touch** (`for (at = got; at < got+step; at += kPage)`, `:3645`) that faults the pages; after the loop the prefix is cut at an expert-blob boundary (`:3650-3655`) and pinned in **ONE** `cudaHostRegister` over the faulted range (`:3657`). The pool route **forces** the paced path when `huge_pool_covers(bytes)` (`:3542-3547`) and backs it with `huge_alloc` → MAP_HUGETLB (`:3584-3591`).
- The UD arena takes the opposite discipline: **register before any page is touched** (`pinned.cu:457-459` comment), i.e. the driver's GUP does the faulting of the unfaulted hugetlb mapping. This contradicts the stated contract in `include/strata/platform/hugepage.hpp:10` ("must be faulted 2 MiB BEFORE cudaHostRegister, which both routes do").

**What the research (§1) says to change, ONLY when `huge_map.hugetlb == true`** (pool pages were carved out at boot: no reclaim class, no #1250 OOM class, MAP_HUGETLB fails instantly when short):

1. **Drop the per-GiB availability gate** (`:3641`) on the hugetlb arm — it guards a failure class the pool cannot produce. Keep it for the THP fallback arm.
2. **Fault at 2 MiB, not 4 KiB**: replace the 4 KiB-stride walk with one `MADV_POPULATE_WRITE` over the whole mapping (kernel ≥ 5.14; this box runs 7.2.12). Today's walk writes ~9.86M cache lines and faults 19,248 huge pages for 37.6 GiB; MADV_POPULATE_WRITE does only the faults, in-kernel, and the kernel's huge-page zeroing is the irreducible part (~37.6 GiB of zeroing ≈ 3-6 s single-threaded).
3. **Keep ONE registration call** — §1 shows chunking/parallelism buy nothing.
4. **Unify PinnedArena on fault-first** (`pinned.cu:457-459`): `MAP_POPULATE` at reserve, or the same MADV_POPULATE_WRITE pass, so both organs honor the `hugepage.hpp:10` contract and the driver's GUP never faults internally.

**Priced boot win** (GSQ 45 s → 93 s, +48 s; decomposition estimates, P1 decides the split):

- Fault/touch + gate overhead: ~6-12 s buyable (items 1-2).
- The single 37.6 GiB register: ~19-38 s at the documented class — **irreducible** without giving up PORTABLE|Mapped pinning (and `cudaHostAlloc` would reintroduce 4 KB-by-construction, the degradation this campaign exists to remove).
- Expected: 93 → ~78-85 s; the remaining cost is the price of pinning, not overhead. Law-17 test: one boot pair, same pack, wall-clock boot lines + P1's register-only timing; acceptance ≥ 8 s saved with lanes unchanged.

---

## 4. The conc −4.4%: decisive measurement design, then candidate mechanisms

**Design (same run, same boot, no reboot — the STRATA_NO_LARGEPAGES switch is per-process):**

1. One boot, pool asserted full (R0). Launch both engines 3× per arm, interleaved: A(pool) → B(`STRATA_NO_LARGEPAGES=1`, routes pool+THP out in-process, honored at `pinned.cu:250`, `hugepage.hpp:85`) → A → B → A → B. Restart engines between arms; restart is a perfect pool re-arm (`HOSTMEM-NO-REBOOT.md:96-97`).
2. Leg shape: the exact 2+2 config that showed −4.4%; fixed prompt (`campaign/conc-probe.py` CONC prompt, `:9-10`), **temperature 0** (the harness's 0.7 default is unseeded noise; pass 0 via the request), `--streams` and `--tok` frozen at the banked config, `--rounds 6` per arm (~90 s/leg per the harness doc, `:4`).
3. Per-arm witnesses (cheap, decisive): boot log ARENA PAGES line per engine; `hugepage_report` complement line per engine; `HugePages_Free` delta per engine start; one `grep AnonHugePages\|VmFlags` smaps read of each complement. An arm with partial `ht` coverage is void, not scored.
4. Endpoint: aggregate tok/s per round; paired medians + IQR across the 3 interleaved A-B pairs; call it real only if |B−A| ≥ 2% with non-overlapping IQRs. Escalate to 3+3 legs/deeper context only after a reproduced, witnessed gap.

**Candidate mechanisms if pool-backed conc is genuinely ~4% slower, each with a falsifiable probe:**

- **M-a. Pool-coverage confound (top suspect, free)**: on the 47.6 GiB boot, the second engine in pair/conc fell off the pool — the "pool" arm was mixed-class. Probe: re-read the measured runs' boot logs for the second engine's ARENA PAGES line; if 4K/THP, the −4.4% is void and the protocol above re-decides it.
- **M-b. Physical interleave/striping**: 4 KB anon scatters at fine granularity; consecutive boot-buddy hugetlb pages are physically contiguous runs. DDR5 address hashing should stripe within a 2 MiB page, but bank/row distribution changes. Probe 1 (CPU-side): dump `/proc/self/pagemap` frame numbers for 64 consecutive pool pages — check the stripe pattern; then run `stream_bandwidth()` (exists, `pinned.cu:992-1023`) on 8 GiB windows, pool vs `cudaHostAlloc` backing, same boot, one leg. If pool GB/s ≥ 4K GB/s, mechanism dead. Probe 2 (end-to-end): the same stream bench under the conc leg count.
- **M-c. NUMA**: single-socket Zen5 is one NUMA node — verify once with `numactl --hardware` (if one node, permanently dead). CCD split is a separately priced item (T35 row 5, `campaign/research/T35-memory-hierarchy.md:107`) and is page-size-independent; not a conc mechanism.
- **M-d. Driver DMA-mapping class**: PORTABLE|Mapped registered memory vs `cudaHostAlloc` memory may take different GPU-side PTE granularities/caching classes. Probe: per-expert H2D latency from the existing batch telemetry (ms_pool/wait class lines) on one stream, A vs B; equal latency → dead. Escalate to nsys only if the telemetry line can't split it.
- **M-e. dTLB at concurrency** (only if a-d die): 2 MiB pages *shrink* the working set of TLB entries at 4+ legs — should help, not hurt; probe `perf stat -e dTLB_load_misses,l1_dtlb_load_misses` per arm once.

---

## 5. KV host shadows (23.4 GiB aggregate): keep gated; fix the fault-order bug; build the combined arena for a bigger-pool day

**What they are** (`src/core/layer.cpp:927-961`): per-layer **authoritative** K/V in host RAM, pinned + device-mapped; the layer's state cursor is built on the **device alias** (`Cursor hc{d}`, `:944`, from `cudaHostGetDevicePointer`, `:935`) — this is zero-copy KV streaming: kernels read AND write host RAM over PCIe per token (`docs/kv-streaming-design.md`; `hugepage.hpp:3-5` prices the class: ~9M PTEs → ~18k at 2 MiB, +5-15% staging path per F-201-zen5 #6). These are the hottest per-byte DMA organ, not a CPU-side staging area.

**Three findings:**

1. **Blind spot**: `kv_host_alloc` only reports page class when the pool route succeeds (`layer.cpp:603-609`); the THP/4K fallback prints nothing — today nobody knows what the shadows are backed with. Add the `hugepage_report` call on the fallback path (one line, loudness parity with `announce_page_class`, `pinned.cu:424-439`).
2. **Contract violation (bug, S)**: `kv_host_alloc` registers **without faulting first** (`layer.cpp:600-602` — `huge_alloc` then immediate `cudaHostRegister`), contradicting `hugepage.hpp:10` and re-introducing the T35 C2 unfaulted-registration stall class on the THP branch (small scale: 0.26 GiB each) and driver-GUP faulting on the hugetlb branch. Fix: MAP_POPULATE / MADV_POPULATE_WRITE before the register, mirroring §3 item 4.
3. **Pool arithmetic says: keep gated.** Standing-leg slack after organ+store: GSQ 88−75.6 = 12.4 GiB; UD-Q2 88−77.4 = 10.6; IQ4 at 96 GiB: 96−94 = 2. None fits 23.4 GiB. `STRATA_POOL_SMALL_ARENAS=1` re-creates the starvation the ≥8 GiB gate exists to prevent (`pinned.cu:238-243`, `hugepage.hpp:71-75`).

**Recommendation**: keep gated; fix (2), add (1); build the **combined arena** (one mmap + one registration for all ~90 layers, 23.4 GiB ≥ 8 GiB gate → pool-eligible; per-layer offsets inside) as ready machinery, armed only on a boot with a ≥ 130 GiB pool (94 + 23.4 + slack) — a dedicated bench boot on this 184 GiB box, not a standing config. **Priced**: the KV DMA path is per-token latency-sensitive; the IOTLB-class win bounds the end-to-end gain at ~1-3% (the class recovered ~5% at 2 MiB per Farshin via F-201 #6, on a path that is a fraction of the window). Law-17 test when armed: lane banks + smaps ht == 23.4 GiB + per-layer miss µs (the ~75 µs/entry store class is the store, not the shadows — track the KV-stream miss path separately).

---

## 6. IQ4_XS at 94 GiB pool need: raise the reservation to 49152 pages (96 GiB)

**Arithmetic on the 184 GiB box** (armed today: 45056 = 88 GiB):

| option | pool math | verdict |
|---|---|---|
| raise to 49152 (96 GiB) | IQ4 56+38 = 94 ≤ 96 (slack 2); GSQ 75.6 (slack 20.4); UD-Q2 77.4 (slack 18.6); ~88 GiB left for OS/desktop/page cache on 184 | **adopt** — one line, sanctioned by the sizing table ("raise-to-49152 at boot", `HOSTMEM-NO-REBOOT.md:97-98`) |
| arena-only + store on /dev/shm | store on tmpfs = the exact organ the root probe killed on this kernel: shmem PMD folios never materialize (`strata-huge-store.service` header, `HOSTMEM-NO-REBOOT.md:55-59`) → 4 KB store, miss path pays the TLB tax again | reject |
| `STRATA_POOL_SMALL_ARENAS=1` on IQ4 boots | 96−94 = 2 GiB slack < 23.4 GiB of small arenas → starves the 56 GiB arena the way the gate's 2026-11-07 measurement showed | reject |
| per-leg pool switching (45056 default, 49152 for IQ4 ladder legs) | two sysctl values to keep honest; only IQ4 needs the extra 8 GiB | fallback if the standing 96 GiB reservation ever pressures the desktop (watch `HugePages_Total` and MemAvailable after R0) |

Boot reservation of 96 GiB is buddy carving at boot — near-instant, unfaulted, unzeroed; no boot-time cost beyond the pristine-buddy requirement (already required for 45056). The hugetlbfs store mount (`size=40G`) covers the 38 GiB store unchanged.

**Law-17 test**: IQ4_XS leg boot pair (88 GiB pool → organ 2 must fall back vs 96 GiB → both organs ht-covered by smaps); acceptance: IQ4 solo bank moves by the +5-8% class measured on the other two packs (UD +4.9%, GSQ +7.2% same-window), GSQ/UD banks unchanged (law 13 — the raise only adds slack, it cannot cost the small packs), and `HugePages_Total == 49152` asserted in the boot log.

---

## Ranked actions (each priced, each with its witness)

| # | action | cost | expected | witness / law-17 test |
|---|---|---|---|---|
| R0 | assert pool coverage per boot (HugePages_Total, per-engine Free delta, ARENA PAGES lines) | free | un-confounds everything | log lines (§0) |
| R1 | raise `vm.nr_hugepages` to 49152 | 1 sysctl line | IQ4_XS lane +5-8% class | §6 boot pair |
| R2 | settle conc with interleaved STRATA_NO_LARGEPAGES A/B (temp 0, 6 rounds/arm, per-arm smaps) | ~1 h box time | verdict on −4.4% | §4 protocol |
| R3 | drop pacing + MADV_POPULATE_WRITE when backing==hugetlb; unify PinnedArena fault-first | S code | GSQ boot −6-12 s | §3 boot pair + P1 timing |
| R4 | P1 microbench: time the single register alone | log-only print | decides if ~30 s is irreducible | §1 |
| R5 | fix `kv_host_alloc` fault-before-register + fallback page-class report line | S code | removes stall class + blind spot | smaps line per boot (§5) |
| R6 | combined KV-shadow arena machinery, armed at ≥130 GiB pool (bench boot only) | M code | 1-3% e2e class | §5 law-17 leg |

Sources (web, this session): [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode), [vLLM zero-reload blog](https://vllm.ai/blog/2025-10-26-sleep-mode), [Snowflake semi-persistence](https://www.snowflake.com/en/blog/engineering/semi-persistence-gpu-model-swapping), [OGKM #732](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/732), [UCX #4393](https://github.com/openucx/ucx/issues/4393), [sglang #6285](https://github.com/sgl-project/sglang/issues/6285), [PyTorch #139](https://github.com/pytorch/pytorch/issues/139), [PyTorch #16280](https://github.com/pytorch/pytorch/issues/16280), [NVIDIA forum: multi-threaded cudaHostRegister](https://forums.developer.nvidia.com/t/strange-behaviour-of-cudahostregister/294999), [NVIDIA forum: registration performance variance](https://forums.developer.nvidia.com/t/why-cudahostregister-can-have-a-huge-difference-in-performance/308105), [NVIDIA forum: mmap'd memory registration](https://forums.developer.nvidia.com/t/cudahostregister-crash-or-poor-performance-unknow-error-30-in-kernel-for-64bit-host-operating-syst/26392), [CUDA toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html), [CUDA 13.0 announcement](https://developer.nvidia.com/blog/tag/cuda-13/), [UCX programming guide (rcache)](https://docs.nvidia.com/doca/archive/doca-v2.2.0/ucx-programming-guide/index.html), [H2D/D2H bandwidth study](https://github.com/ak811/cuda-h2d-d2h-bandwidth-study), [host-GPU bandwidth in LLM serving (May 2026)](https://www.google.com/goto?url=CAESYAHEIkXXJRl_khV47fkxqcwEQu8IYluaShYP4QAtGfqAs9poJ_PfGuym1NmuhCXNolrKOsJ0Mbt8KIj3PU-UXi7qP6xtNBK_e5TjBBOl-g7itFu_YzcDgQsbPI_iYCAKlQ), [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
