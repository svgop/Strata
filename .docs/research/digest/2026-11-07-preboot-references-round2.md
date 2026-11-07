---
type: research-digest
id: 2026-11-07-preboot-references-round2
date: 2026-11-07
topic: round-2 reference recon for the preboot host-memory pipeline — multi-model pool sizing, cudaHostRegister speed, hugetlb+register interactions, KV host-spill backing, hugetlbfs as file store, recent upstream changes
method: primary-source fetches 2026-11-07 (vLLM source + PRs/issues #60130 #51081 #42632 #58852 #60909 #61859 #63289, SGLang source environ.py/mmap_allocator.py/pool_host/common.py/paged_experts/store.py, FlexKV hugepage doc, LMCache docs, TRT-LLM docs+release notes, kernel source fs/hugetlbfs/inode.c, NVIDIA forum, RMM #2054) PLUS controlled local experiments on the target box (RTX 5090 32GB, driver 615.71.09, CUDA 13.4 nvcc, 2 MiB hugetlb pool 24369 pages, /mnt/strata-huge hugetlbfs mount). Every claim carries source + confidence grade.
box-context: custom llama.cpp-lineage CUDA engine; expert arenas 37-56 GiB from MAP_HUGETLB pool + cudaHostRegisterPortable (5/5 boots, 37-39 GiB each); 38 GiB read-mostly store moving to hugetlbfs; DeepSeek engine ~103 GiB CPU-lane experts; paced step-registration costs ~48 s per 37.6 GiB at boot; three model packs (81/42/94 GiB) share the box across reboots.
---

# Digest 2026-11-07 — Preboot reference recon, round 2

## Q1. Multi-model-one-box pool sizing

**The reference world has no cross-model shared host pool. Every stack sizes per-process, statically, from an explicit byte budget, allocated once at startup.** Nobody implements dynamic resize of a pinned pool; nobody implements a shared-pool-across-models scheme. Confidence: **verified-in-source** (negative result across six codebases/docs).

- **vLLM `SimpleCPUOffloadConnector`** (merged today, PR #60130): each rank gets **one host mapping sized to the resolved KV blocks**, budget from `cpu_bytes_to_use_per_rank` (server-wide `cpu_bytes_to_use` is divided by world_size; per-rank key overrides with a warning). `num_cpu_blocks` overrides pool-block count for ranks with different KV layouts (e.g. PP stages). Sources: `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` (lines 104-131), `vllm/v1/simple_kv_offload/host_buffer.py` ("One host mapping backing a rank's CPU KV caches"), docs/features/kv_offloading_usage.md. https://github.com/vllm-project/vllm/pull/60130
- **vLLM pool-reuse across models**: torch CachingHostAllocator caches pinned blocks in-process; pools live for process life (round-1 digest). Multi-model = one process per model, each with its own pool. **verified-in-source**
- **TRT-LLM**: host KV offload buffer is a single static byte config — `kv_cache_host_memory_bytes` (Triton) / `hostCacheSize` (Executor API) / `--kv_host_cache_bytes` (bench). Docs: "will create a 45 GiB offloading buffer in host memory" as a **one-time cost**: "allocating a lot of pinned memory on x86 machines can take a substantial amount of time (10s of seconds)". https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html — **docs-claimed**
- **LMCache**: `LMCACHE_MAX_LOCAL_CPU_SIZE` (GB) — "the amount of page-locked (for fast GPU transfer) CPU memory that LMCache will reserve", LRU-evicted so "there is no danger of running out of pinned CPU RAM". Static, per-process. https://docs.lmcache.ai/legacy/kv_cache/storage_backends/cpu_ram.html — **docs-claimed**
- **FlexKV**: `cpu_cache_gb` (example 32); pool allocated once by the TransferManager, workers attach via fd reopen. https://github.com/taco-project/FlexKV — **docs-claimed**
- **KTransformers**: no pinned/hugepage host-pool concept at all; expert weights stay **file-backed mmap** (round-1) and the new kt-kernel toolchain bakes **NUMA placement into the file layout**: "Each expert's weights are split across NUMA nodes for optimal memory access" (`blk.{layer}.ffn_{proj}_exps.{expert}.numa.{numa_idx}.weight`). https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/scripts/README.md — **verified-in-source**
- **SGLang**: HiCache host pools allocated once at boot per engine; **no multi-model-one-box guidance exists anywhere in docs or source**. Multi-model = separate engine processes. Confidence: **named boundary** (absence verified by code search `SGLANG_HUGEPAGE_SIZE|host pool` + docs, 2026-11-07).

**Implication for us**: static-per-largest reserved pool + per-process arenas is *ahead of* the reference stacks; their only extra trick is that vLLM's newest design makes the pool **one contiguous mapping per rank** with KV views into it (single registration, single NUMA policy), which maps 1:1 onto our arena layout.

## Q2. Registration speed: state of the art for a 40-90 GiB pin

**Answer: seconds, not a minute, comes from (a) prefault before register, (b) large backing pages, (c) chunked registration calls. Multi-threading registration does NOT help.**

Local experiment on this box (driver 615.71.09, RTX 5090, CUDA 13.4, 2 MiB hugetlb pool, GPU idle; C program, `cudaHostRegisterPortable`, wall-clock `CLOCK_MONOTONIC`; **verified-experiment**, scripts `/tmp/exp/pin_bench{,2,3}.c`):

| test | result |
|---|---|
| CUDA context init (`cudaFree(0)`) | 0.21 s (one-time; pollutes naive microbenchmarks) |
| 32 GiB anon MAP_HUGETLB, prefaulted, **single call, 1 thread** | **0.24 s = 133.8 GiB/s** |
| 32 GiB anon MAP_HUGETLB, 8 threads x 4 GiB chunks | 0.23 s = 137.4 GiB/s (no speedup — driver serializes) |
| 32 GiB anon MAP_HUGETLB, 16 threads x 2 GiB chunks | no speedup |
| 32 GiB, register **lazy** (pages untouched) | 1.53 s = 21.0 GiB/s (driver faults internally) |
| 32 GiB, `MADV_POPULATE_WRITE` then register | 1.29 s populate + 0.24 s register = same 1.53 s total |
| pinned H2D `cudaMemcpy` proof | 49.25 GiB/s (PCIe Gen5 x16, real) |
| 4 GiB hugetlbfs file MAP_SHARED register | success, fast (see Q3) |
| 4 GiB registration at base+4096 (2 MiB-misaligned base) | **success** |

Reference-world numbers, ranked:

1. **vLLM PR #60130** (2026-11-07): *"With 1 GiB pages, pinning 128 GiB takes **0.19 s** instead of **2.5 s** on ordinary pages."* — 1 GiB pages pin ~674 GiB/s, ordinary pages ~51 GiB/s. Also engine init TP=2 64 GiB/rank: 19.0 s vs 20.2 s. **verified-in-source (PR measurements)**
2. **vLLM #51081** (canonical registration-size bug): ordinary-page regions pin at *"~0.1 s per GB"* on 4xH200. Our production pace (0.78 GiB/s) is ~13x below the ordinary-page floor — the 48 s is pacing, not the driver. **verified-in-source (measured table in issue)**
3. **NVIDIA forum** (8 GiB blocks): initialized pages pin in 259 ms (32 GiB/s), uninitialized in 1678 ms (4.9 GiB/s) — same ~6.4x prefault factor as our local 21-vs-134 GiB/s. https://forums.developer.nvidia.com/t/why-cudahostregister-can-have-a-huge-difference-in-performance/308105 — **forum-inferred, matches local experiment**
4. **SGLang**: chunks registration calls at `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB` (**default 256 GB per call**) — "Avoid oversized cudaHostRegister calls on large host pools" — and prefaults host pools in parallel *allocation* threads (`SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS`, ">0: prefault host pools with a private THP mapping using this many threads"), chunks "THP-aligned so no huge page straddles two threads". `python/sglang/srt/mem_cache/pool_host/common.py`, `python/sglang/srt/environ.py` — **verified-in-source**
5. **vLLM async registration** (#42632, open feature): at TB scale "the kernel pinning every page (NVIDIA driver lock + RDMA cgroup accounting + page-table walks)... single call running for many minutes"; proposal = serve the GPU-only path immediately, pin in a background thread, gate connector traffic until ranks report ready (`async_register_cache` flag). **verified-in-source (proposal, not merged)**
6. **SGLang prefault doctrine** (`mmap_allocator.py`): *"cudaHostRegister has to pin real, pre-faulted pages, so these mappings can never be handed back lazily. MAP_POPULATE and MADV_POPULATE_WRITE each give that guarantee on their own, but asking for both makes the kernel walk the whole mapping twice. Prefer the madvise."* **verified-in-source**
7. **vLLM 10 GiB/s prefault cap** (`shared_offload_region.py`): "Page-cache inserts are serialized per shm file, capping pre-faulting of one file near 10 GiB/s" → they split the shared region into **32 backing files** so workers prefault in parallel; minimum segment 1 GiB. **verified-in-source**
8. **CUDA 13.x**: no async `cudaHostRegister` exists; the new capability is stream-ordered **pinned-memory pools**: CUDA 13 added `cudaMemGetDefaultMemPool(..., cudaMemAllocationType)` returning a `cudaMemPool_t` "backed by managed or pinned memory", usable with `cudaMallocFromPoolAsync` (RMM #2054, https://github.com/rapidsai/rmm/issues/2054). CUDA 12.8 release notes add nothing for host pinning (Blackwell support focus). Confidence: **docs-claimed** (RMM issue + toolkit notes), no measured speedup claims anywhere.

**What makes 40-90 GiB take seconds**: prefault once (`MADV_POPULATE_WRITE`, ~25 GiB/s on 2 MiB pages, 10 GiB/s/file on page cache → parallel files/threads), then one register call per ≥2 MiB-backed region (~130-670 GiB/s). 37.6 GiB prefaulted+registered = ~2.0 s; on 1 GiB pages vLLM measured 128 GiB in 0.19 s. Our 48 s is ~24x the whole-job cost of the reference recipe.

## Q3. cudaHostRegister over hugetlb-backed mappings

**No known failure mode in the size range we run.** The documented ceiling and all real reports are about page COUNT, not hugetlb backing:

- **Hard ceiling: a single `cudaHostRegister` call of ≥512 GiB fails with `cudaErrorMemoryAllocation` (code=2) regardless of free RAM** — vLLM #51081 measured the boundary on 4xH200 with 1.3 TB free: 511.96 GiB success (55.4 s), 512.15 GiB fails. Fix: register in ≤64 GiB chunks "that end on block-row boundaries... so the driver's page rounding never makes two registrations overlap and no block transfer spans a boundary"; all-or-nothing rollback; drain pending errors through one CudaRTLibrary instance (a failed registration leaves its error pending and torch reports it against an unrelated later call). https://github.com/vllm-project/vllm/issues/51081 — **verified-in-source**
- **At 450 GiB on 4 KiB pages** (~118M pages), double pinning (cudaHostRegister + NIXL RDMA MR) "wedge[d] K8s nodes during startup" — vLLM #58852. With 1 GiB pages "450 GiB becomes 450 pages, so pinning and MR translation tables become trivial". **verified-in-source**
- **Confidential Computing**: `cudaHostRegister` is not allowed in CC mode; every vLLM offload path depends on it → crash-loop on TDX/H200 CC-On (#55428). Irrelevant to us (no CC), but a known portability wall. **verified-in-source**
- **Partial-pin race**: registration failure on one TP rank can leave "mixed pinned/unpinned" ranks (#51762); unpin-at-exit is also expensive and must be done once per physical pool, not per rank (#61859: TP8 256 GiB shutdown stall 86-90 s because "every TP rank unpins the full shared host pool"). **verified-in-source**
- **ulimit/cgroup**: the interaction is RLIMIT_MEMLOCK for register and the hugetlb pool/cgroup limit for mapping: vLLM fails fast with "The container's memory or hugetlb limit is too small" (`host_buffer.py`); FlexKV requires "a large enough memlock limit" and reserves pages per NUMA node via `/sys/devices/system/node/node*/hugepages/.../nr_hugepages`. **verified-in-source**
- **IOMMU/DMA granularity**: no reports found of registration splitting huge pages or IOMMU rejects at 2 MiB/1 GiB granularity in vLLM/SGLang/DeepSpeed trackers (searched 2026-11-07). SGLang's HiCache and FlexKV's transfer workers register hugetlb/memfd-backed pools in production. **named boundary: negative search result**
- **Local experiment (verified-experiment)**: `cudaHostRegisterPortable` over (a) anon MAP_HUGETLB and (b) hugetlbfs file MAP_SHARED, both **32 GiB, success**, single call; also success at a 4 KiB-misaligned base (base+4096) — the driver does not require 2 MiB-aligned registration bases. Register+unregister+munmap returns all pages to the pool (HugePages_Free restored 24369/24369 after cleanup).

**Watch list for 56-103 GiB** (extrapolation from the graded evidence): stay under **512 GiB per single call** — we are 5-9x under it, fine; keep chunk boundaries aligned to copy granularity ("Splitting an unknown page-first layout at an arbitrary byte offset can make one cudaMemcpyBatchAsync span two registrations" — SGLang common.py); expect unpin cost at exit to mirror pin cost (register-once/unregister-once per boot); memlock ulimit must cover the full registered set; the real scaling risk at 103 GiB is prefault throughput (10 GiB/s per file), not registration.

## Q4. KV-cache host spill backing

**Every reference stack backs CPU KV spill with PINNED memory; huge pages are the 2026 opt-in upgrade. Nobody leaves host KV on pageable memory by default.** Confidence: **verified-in-source** across five stacks.

| Stack | Backing | Pinned? | Huge pages? | Config |
|---|---|---|---|---|
| vLLM SimpleCPUOffloadConnector | one host mapping per rank (PR #60130), KV blocks as views | yes, registered in ≤256 GiB chunks | opt-in `"2MB"`/`"1GB"`, **no fallback** ("startup fails; there is no fallback to ordinary pages") | `cpu_bytes_to_use_per_rank`, `cpu_hugepage_block_size`; Linux ≥5.14 for population |
| vLLM OffloadingConnector (shared) | `/dev/shm` mmap in 32 files, registered post-hoc | yes ("Pinning is a pure bandwidth optimization: when registration is unavailable the transfers still run correctly, just as pageable (slower) copies" — `host_register.py`) | no (hugetlbfs dir proposed in #58852/#45693, draft) | `num_cpu_blocks` |
| vLLM FlexKV connector (in mainline ≥v0.17.2) | `memfd_create(MFD_HUGETLB)` pool, workers mmap via `/proc/<pid>/fd` | yes, workers register | yes, `hugepage_size_bytes` 2 MiB/1 GiB, "best-effort with automatic fallback" | `cpu_cache_gb` |
| SGLang HiCache L2 | host pool via mmap/shm allocators | yes: register + **mlock** after | `SGLANG_HUGEPAGE_SIZE=2MB/1GB` for mmap tier; THP path with parallel prefault threads | pool sized at boot |
| LMCache | CPU RAM pool | yes ("page-locked... that LMCache will reserve") | opt-in `local_cpu_use_hugepages: true` + `vm.nr_hugepages`; "reduce TLB pressure and improve memory access performance" | `LMCACHE_MAX_LOCAL_CPU_SIZE` |
| TRT-LLM | single host offload buffer | yes ("is pinned memory"; "10s of seconds" warning) | none in source | `kv_cache_host_memory_bytes` |

Key docs lines (vLLM `docs/features/kv_offloading_usage.md`): *"Reserve enough pages on the host for every rank, rounding each rank's cache size up to a whole page. Containers also need a matching hugetlb allowance... If the pool or the limit is too small, startup fails; there is no fallback to ordinary pages."* And without the key: *"the caches use ordinary pages, and `THP_MEM_ALLOC_ENABLE=1` requests transparent huge pages. In both cases, the memory follows the worker's NUMA policy."*

**Our 23.4 GiB of per-layer 0.26 GiB KV host shadows**: the reference world would (a) fold them into the registered arena (vLLM's "one host mapping, views" move — registration is cheap enough now that gating them out of the pool by size is a solution to a problem that no longer exists), or (b) on 1 GiB pages make 0.26 GiB shadows sub-page views of a single 1 GiB-backed mapping. TRT-LLM's model (single `kv_cache_host_memory_bytes` pool containing all host KV) is the same shape.

## Q5. hugetlbfs as a file-store backing

**Direct precedent exists upstream**: vLLM #58852 mounts hugetlbfs as the host-pool directory (`host_pool_dir: /dev/hugepages-1Gi`, K8s `emptyDir: {medium: HugePages-1Gi}`), FlexKV dropped hugetlbfs-mount support in favor of `memfd_create(MFD_HUGETLB)` ("No hugetlbfs mount is needed and no file is created anywhere"). Nobody ships model SHARDS on hugetlbfs; read-only quant stores on hugetlbfs appear nowhere upstream (searched 2026-11-07). **named boundary + verified-in-source for the pool-file precedent.**

**Local experiment matrix on this box** (mount `/mnt/strata-huge`, pagesize=2M, uid 1000; kernel source cross-checked; **verified-experiment**):

| operation | result |
|---|---|
| `write(2)` (shell `>`, cp, dd, rsync) | **all fail EINVAL**; leave 0-byte files. rsync 3.5.1: "write failed... Invalid argument (22)" |
| `cp --reflink=always` | EXDEV (no reflink on hugetlbfs) |
| `copy_file_range()` into hugetlbfs | **EXDEV** (Invalid cross-device link) |
| `sendfile()` into hugetlbfs | **EINVAL** |
| `open(O_DIRECT)` | **EINVAL** — no direct I/O on hugetlbfs |
| `ftruncate` to 2 MiB-aligned size | OK |
| `ftruncate` to unaligned size (100 MiB+7) | **EINVAL** — kernel: `if (newsize & ~huge_page_mask(h)) return -EINVAL;` (fs/hugetlbfs/inode.c, `hugetlbfs_setattr`) |
| `fallocate(16 MiB)` | OK (preallocation works, blocks charged) |
| mmap MAP_SHARED read/write | OK; single-threaded memcpy page-cache→hugetlbfs **15.16 GiB/s** |
| `read()` from hugetlbfs file | OK (`.read_iter = hugetlbfs_read_iter` exists; `.write_iter` does NOT — write(2) unsupported is by design) |
| sparse behavior | ftruncate'd file = 0 blocks; touching 1 byte charges a full 2 MiB page (blocks=4096) |
| atime | mount is `relatime` — negligible |
| mount quota | `size=` enforced by statfs; note "hugetlbfs statfs reports 0 free without a size= limit" (vLLM #58852) |

**Consequence for the 38 GiB store**: stock `rsync` onto hugetlbfs is impossible by design; population is `ftruncate(2 MiB-aligned)` → `mmap MAP_SHARED` → `memcpy` (~2.5 s single-threaded at measured 15.16 GiB/s, multi-thread would improve). FlexKV's alternative removes the filesystem entirely: `memfd_create(MFD_HUGETLB)` + `ftruncate` — same pages, no mount, no file-store gotchas, nameable via `/proc/<pid>/fd/N` for cross-process sharing. The kernel-side contract is identical (reserved-pool accounting, no swap, no write(2)).

## Q6. Recent (6-12 months) upstream host-memory changes worth stealing

All verified-in-source, newest first:

1. **vLLM PR #60130** (merged 2026-11-07, today): one host buffer per rank for SimpleCPUOffload; opt-in HugeTLB pages; 256 GiB registration chunks; `MADV_DONTFORK` "so child processes can't make pinned pages copy-on-write"; NUMA-policy inheritance; fail-fast on pool shortfall. https://github.com/vllm-project/vllm/pull/60130
2. **vLLM #51081** (2026): 512 GiB single-registration ceiling + chunked-registration pattern (64 GiB chunks on block-row boundaries, rollback, error-drain through one cudart handle). Related #50094 (single shared region registered by every rank).
3. **vLLM #60909** (2026): `empty_pinned()` — allocate long-lived pinned buffers at **exact size** via `cudaHostAlloc` because torch's caching host allocator "rounds every block up to the next power of two" and "a buffer just over a power of two pins up to twice its size". (4.25 GiB saved on a 4.75 GiB tier in their repro.)
4. **vLLM #63289** (2026, draft): replace per-rank registration of a shared pool with **one `cuMemCreate` + fd handoff** — "Shutdown no longer scales with TP × pool size... TP8 at 256 GiB: 30s vs 86-90s"; blocked on NIXL/UCX misclassifying the memory (`ibv_reg_mr ... Bad address`). Alternative to #62166 (timeout raising).
5. **FlexKV hugepage host cache** (June 2026, in vLLM mainline ≥v0.17.2 as PR #34328): `memfd_create(MFD_HUGETLB)` pool, per-NUMA spread (`FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD=1`), workers attach by fd, verify via `ls -l /proc/<pid>/fd | grep memfd`. https://github.com/taco-project/FlexKV/blob/main/docs/hugepage/README_en.md
6. **vLLM tiered KV offloading blog** (2026-09-10) + KV offloading connector blog (2026-01-08): contiguous per-block KV layout (PR #27743) raised effective block size "by a factor of 2*num_layers" → offload throughput "by an order of magnitude"; bidirectional DMA on H100 peaked 83.4 GB/s with 2 MiB blocks. https://vllm.ai/blog/2026-09-10-tiered-kv-offloading
7. **SGLang**: `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB` (256 GB/call), `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS`, `SGLANG_HUGEPAGE_SIZE`; paged-experts host store (`layers/moe/paged_experts/store.py`): host copy of **all experts** in GPU layout, page-locked via memfd + `cudaHostRegister` in place, gathered by a device-side kernel reading "the pinned store directly, through its UVA address" — CUDA-graph capturable expert paging.
8. **TRT-LLM v1.3.0rc25** (2026): "Synchronize KV cache manager V2 host-tier quotas and fallback decisions across ranks"; "Prevent KV-cache estimation probes from pinning the complete host tier" (avoid pinning during size estimation).
9. **llama.cpp**: still no hugepage support — #12552 (llama-map 2 MiB/1 GiB support), #22022 (MADV_HUGEPAGE hint), #7420 (Direct I/O + THP) all **open**; our MAP_HUGETLB arena design is ahead of upstream.
10. **CUDA 13**: stream-ordered pools for pinned/managed host memory (`cudaMemGetDefaultMemPool` + allocation type; RMM #2054 tracks adoption). No async pin API in 12.8 or 13.x.

## Steal these (ranked)

1. **Prefault-then-register in one pass per boot; drop the pacing.** Measured floor: `MADV_POPULATE_WRITE` 1.29 s + register 0.24 s for 32 GiB on THIS box (133.8 GiB/s prefaulted register). Projected: 37.6 GiB ≈ 2 s instead of 48 s. SGLang/vLLM both mandate prefaulted pages before register; never let the driver fault lazily (6.4x penalty, forum + local both agree).
2. **1 GiB pages for the big arenas.** vLLM measured 128 GiB pinned in 0.19 s (~674 GiB/s) and #58852 shows page-count is the real scaling wall (118M pages at 450 GiB wedges nodes). We already run a 2 MiB pool; add a `hugepagesz=1G` reservation for the 38-103 GiB stores. 2 MiB is fine at our sizes; 1 GiB is the margin.
3. **Chunk registration at 256 GiB max per call, boundaries aligned to copy granularity, rollback on failure, drain errors through one cudart handle.** The 512 GiB single-call ceiling (vLLM #51081) is far, but the chunk+rollback+error-drain discipline costs nothing and prevents the "pending error reported against an unrelated call" failure class.
4. **Fold the 23.4 GiB KV host shadows into the registered arena** (vLLM #60130's one-mapping-per-rank shape): registration is now ~free relative to arena size; size-gating them out is obsolete. On 1 GiB pages they become sub-page views.
5. **Do NOT rsync onto hugetlbfs — it cannot work** (write(2) is unsupported by design; measured EINVAL on cp/rsync/dd/copy_file_range/sendfile/O_DIRECT). Population must be ftruncate(2 MiB-aligned)+mmap+memcpy (~2.5 s at measured 15.16 GiB/s), or skip the mount with FlexKV's `memfd_create(MFD_HUGETLB)` pattern (no file, no mount, fd-shareable across processes).
6. **Pin exactly-sized long-lived buffers** (vLLM #60909): avoid torch-style power-of-two rounding for arena slabs; we already size arenas exactly — keep it; the reference world had to build `empty_pinned()` to fix this.
7. **Register/unregister once per physical pool**, not per consumer (vLLM #63289/#61859): at 103 GiB DeepSeek, per-slab unpin at exit would stall shutdown exactly like their TP8 86-90 s case. Our boot-time-only registration is already the right shape; keep exit symmetric (one unregister per arena).
8. **Watch the deep lane**: async background registration with serve-ready gating (vLLM #42632) and CUDA 13 stream-ordered pinned pools (RMM #2054) are the direction of travel; neither is needed for our boot pipeline today.
9. **Add `MADV_DONTFORK` to every arena mapping** (vLLM #60130: pinned pages must not become COW in forked children) — cheap insurance for engine process trees.
10. **NUMA: bake placement into layout, not runtime policy** (KTransformers per-NUMA expert file shards; FlexKV per-node `nr_hugepages` reservation) — directly applicable to the 184 GiB box if it is multi-node, and already our file-layout philosophy.

---

## Source index

- vLLM PR #60130 https://github.com/vllm-project/vllm/pull/60130 (merged 2026-11-07)
- vLLM #51081 https://github.com/vllm-project/vllm/issues/51081 — 512 GiB ceiling, chunked registration
- vLLM #42632 https://github.com/vllm-project/vllm/issues/42632 — async registration proposal
- vLLM #58852 https://github.com/vllm-project/vllm/issues/58852 — hugetlbfs host_pool_dir
- vLLM #60909 https://github.com/vllm-project/vllm/issues/60909 — exact-size pinning
- vLLM #61859 https://github.com/vllm-project/vllm/issues/61859 — per-rank unpin shutdown stall
- vLLM #63289 https://github.com/vllm-project/vllm/issues/63289 — cuMemCreate shared pool once
- vLLM #55428 — CC mode forbids cudaHostRegister; #51762 — mixed pinned/unpinned ranks
- vLLM source: `vllm/v1/simple_kv_offload/host_buffer.py`, `vllm/v1/kv_offload/cpu/{shared_offload_region,host_register}.py`, `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`, `docs/features/kv_offloading_usage.md` (all fetched 2026-11-07, main branch)
- SGLang source: `python/sglang/srt/environ.py`, `python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py`, `python/sglang/srt/mem_cache/pool_host/common.py`, `python/sglang/srt/layers/moe/paged_experts/store.py`
- FlexKV: https://github.com/taco-project/FlexKV + docs/hugepage/README_en.md
- LMCache: https://docs.lmcache.ai/legacy/kv_cache/storage_backends/cpu_ram.html
- TRT-LLM: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html; releases v1.3.0rc25/v1.4.0rc*
- vLLM blogs: https://vllm.ai/blog/2026-01-08-kv-offloading-connector, https://vllm.ai/blog/2026-09-10-tiered-kv-offloading
- Kernel: fs/hugetlbfs/inode.c (torvalds/master, fetched 2026-11-07) — `hugetlbfs_setattr` alignment check, `hugetlbfs_file_operations`
- NVIDIA forum: https://forums.developer.nvidia.com/t/why-cudahostregister-can-have-a-huge-difference-in-performance/308105
- RMM #2054: https://github.com/rapidsai/rmm/issues/2054 — CUDA 13 pinned/managed pools
- KTransformers: https://kvcache-ai/ktransformers kt-kernel/scripts/README.md; https://ktransformers.net/en/docs/hardware-platforms/cpu-gpu-requirements
- llama.cpp issues #12552, #22022, #7420, #2251 (hugepage, all open or user-closed)
- Local experiments on this box (2026-11-07, GPU idle, pool restored to 24369/24369 after): pin_bench/pin_bench2/pin_bench3 (registration timings), q5 fs behavior matrix, memcpy bandwidth. Raw scripts in /tmp/exp/ (ephemeral); results tabulated above.
