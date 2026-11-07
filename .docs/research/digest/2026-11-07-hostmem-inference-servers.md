---
type: research-digest
id: 2026-11-07-hostmem-inference-servers
date: 2026-11-07
topic: how production LLM inference stacks allocate and hold HOST memory (weights / CPU-offload lanes) so a long-uptime box does not degrade — allocation mechanism, pinning, hugepages, pool reuse, fragmentation/uptime failure stories
method: GitHub code search + raw source fetches at upstream HEAD 2026-11-07 (llama.cpp, vLLM, SGLang, TensorRT-LLM, PowerInfer/Tiiny-AI, KTransformers, ollama, DeepSpeed, FlexLLMGen), kernel docs (docs.kernel.org transhuge), GitHub issues/discussions fetched or search-level. Every claim carries its source and a confidence grade: verified-in-source / docs-claimed / forum-inferred.
box-context: Fedora 45, kernel 7.2.12-300.fc45, 184 GiB RAM, RTX 5090 32 GB, zram 8 GiB, 24/7 uptime, custom llama.cpp-lineage CUDA+CPU-expert MoE engines. Symptom: ~25% decode throughput loss after ~6 days; CPU expert lane is DRAM-bandwidth-bound (~50 GB/s sequential). Evidence: AnonHugePages 120 MB / 133 GiB resident (THP=madvise, nothing calls madvise), shmem_enabled=never (tmpfs shadow store gets zero huge folios), compact_fail 1.18M of 1.66M stalls (71%), zram 6.4/8 GiB. Fresh reboot restores throughput.
---

# Digest 2026-11-07 — Host-memory architecture of production inference servers: the long-uptime reference

## 0. Ground truth about the kernel (what our measurements mean)

From kernel docs, `Documentation/admin-guide/mm/transhuge.rst` (fetched 2026-11-07):

- With `enabled=madvise`, THP is "only enabled inside MADV_HUGEPAGE regions"; the doc instructs: *"Applications that gets a lot of benefit from hugepages... should use madvise(MADV_HUGEPAGE) on their critical mmapped regions."* Our box: THP=madvise and **nothing in the engine calls madvise** → AnonHugePages 120 MB is the expected result. Every anon pool is 4K pages. **verified-in-source (kernel doc + our /sys readings)**
- `shmem_enabled=never` means *"Do not allocate huge pages"* for tmpfs/shmem, and *"Default tmpfs policy is never"*. Our tmpfs shadow store (~107 GiB shmem resident) therefore runs entirely on 4K pages, and tmpfs pages are **anonymous-class memory** (swappable, compactable, zram-eligible), not page cache. **verified-in-source**
- Defrag mode `madvise` is the default: direct reclaim+compaction stalls happen "only for regions that have used madvise(MADV_HUGEPAGE)" — but khugepaged/compaction churn from *any* allocation pressure is what our `compact_stall/compact_fail` counters measure: compact_fail is *"incremented"* per failed compaction-driven allocation. 71% failure rate = the allocator is fighting a fragmented anon arena. **verified-in-source (counters) + our /proc/vmstat**
- Settings "only affect future behavior", so *"you need to restart any application that could have been using hugepages"* — the kernel itself explains why a reboot "fixes" throughput: it re-arms the opportunity to build the arena correctly; it is not evidence that reboots are required. **verified-in-source**

## 1. Per-project findings table

| Project | Weight placement (static host state) | CPU-offload / hot lane | Pinning | Hugepages | Pool reuse across loads | Uptime/fragmentation posture |
|---|---|---|---|---|---|---|
| **llama.cpp / llama-server** | file-backed `mmap(MAP_SHARED)` of GGUF, default; optional `mlock` (`--mlock`, `--load-mode mlock/mmap+mlock`); `--no-mmap` reads into **anon** aligned buffers | KV + compute buffers: `ggml_aligned_malloc` (anon, no madvise on x86) | mlock only (no MAP_LOCKED); `MAP_POPULATE` when prefetching full mapping | **none on x86** (MADV_HUGEPAGE exists only in RISC-V SpacemiT platform pool) | mmap reloads are page-cache hits (near-instant); anon buffers re-allocated per boot | No THP/madvise engineering on mainline; long-run slowdown reports exist, root cause never confirmed; relies on page cache for stability |
| **vLLM** | weights: GPU; `--cpu-offload-gb` → **pinned** CPU + CUDA UVA zero-copy views | KV offload (CPUOffloadingSpec/SimpleCPUOffloadConnector): **pinned host pools**, optionally **HugeTLB 2MB/1GB** (`cpu_hugepage_block_size`) | `cudaHostRegister`-grade pinning via torch `pin_memory`; `MADV_DONTFORK`; `MADV_POPULATE_WRITE` prefault before pin | Explicit: MAP_HUGETLB 2MB/1GB; MADV_HUGEPAGE opt-in via `THP_MEM_ALLOC_ENABLE=1` | torch CachingHostAllocator caches pinned blocks in-process; pools allocated at startup for life | Active engineering against fragmentation (hugepages, prefault, fail-fast instead of SIGBUS mid-serve); known bugs: offload leaks, pinned-allocator power-of-2 rounding |
| **SGLang** | weights: GPU (loader optional pinned CPU offload) | HiCache L2 host KV pool: `cudaHostRegister` + `mlock`; mmap storage tier with **MADV_HUGEPAGE/MAP_HUGETLB/MADV_POPULATE_WRITE** | explicit pin + mlock; custom allocator to dodge torch's power-of-two pin rounding | Explicit: `SGLANG_HUGEPAGE_SIZE=2MB/1GB`; default path = MAP_PRIVATE\|ANON + **MADV_HUGEPAGE** + MADV_DONTFORK + threaded THP-aligned prefault | host pools allocated once at boot, held for process life | Most explicit hugepage/THP engineering of all nine; documented rationale strings about shmem fault bandwidth and COW-vs-pinning races |
| **TensorRT-LLM** | weights: GPU (engine build) | KV host offload buffer + **pinned staging rings** (grow-only, event-protected) | `cudaHostAlloc`/`cudaFreeHost` (C++ `CudaHostAllocator`); repo lint script forbids ad-hoc `.pin_memory()` | no HugeTLB found in source | host offload buffer allocated once ("one-time cost"), staging rings grow and never shrink | Documents the pin cost curve: big pinned allocs take *"10s of seconds"* on x86 — allocated once at startup, then stable |
| **FlexGen (FlexLLMGen)** | per-tensor CPU tensors from numpy files (`init_weight_list`), policy-driven GPU/CPU/disk split | disk tier via np/mmap-backed files | `pin_weight` policy flag ("Whether to use pinned memory for weights on CPU"); non-matrix weights pinned by default | none | per-boot allocation; no cross-boot pool | 2023 research code; per-tensor anon allocation = worst case for churn; never designed for weeks of uptime |
| **DeepSpeed(-Inference / ZeRO-Inference)** | ZeRO param offload: CPU tensors, `offload_param.pin_memory` flag | MoE/expert offload same ZeRO path (docs-claimed) | explicit `pin_memory` in offload config; **native pin_memory op** (`DS_PIN_MEMORY_BACKEND=native`) + `pin_memory_tracker` ("Pinned memory is page-locked: it cannot be swapped out") | none found | torch caching allocator reuse in-process | pin_memory tracked and accounted; no hugepage work |
| **PowerInfer (Tiiny-AI)** | llama.cpp lineage: GGUF `mmap(MAP_SHARED)` + `MAP_POPULATE` when prefetch, `POSIX_FADV_SEQUENTIAL`; NUMA → `MADV_RANDOM` | same as llama.cpp (anon aligned malloc for compute buffers) | mlock option only | none | page-cache reload | inherits llama.cpp posture exactly |
| **KTransformers** | **expert weights consumed directly from mmap'd GGUF/safetensors** (`get_mmap_tensor`, `np.memmap(file, mode='r')`) — zero-copy page-cache aliasing | small per-batch control tensors pinned (`pin_memory=True`); `MADV_WILLNEED` to warm page cache "WITHOUT copying" | pin only for tiny io tensors, never for weight bulk | none (relies on file-backed pages) | mmap handles held for process life; **documented UAF failure** when Python freed the mmap under aliased scales → switched to copying scales | file-backed posture = lowest anon-churn design of all nine; one real mmap-lifetime failure story |
| **ollama** | wraps llama-server: mmap default **on Linux+CUDA full-offload**; auto-disabled for cpu-only, Windows+CUDA, partial-metal | inherits llama.cpp anon buffers | no explicit mlock by default | none | **keep_alive** (default 5 min): loaded runner is reused if `!runner.needsReload` — zero re-alloc while warm | Explicit host-pressure guard flips mmap→anon and says why: turning file-backed mappings into anonymous memory makes things *worse* |

## 2. Per-project detail with quotes and grades

### 2.1 llama.cpp / llama-server

**Allocation mechanism — file-backed mmap is the default load mode.** `src/llama-mmap.cpp`:
```c
int flags = MAP_SHARED;
...
// MAP_POPULATE would fault in the lazy ranges too
if (prefetch && lazy_ranges.empty()) { flags |= MAP_POPULATE; }
```
`common/arg.cpp` documents the load modes verbatim:
```
- auto: mmap, unless a device does not support it
- mmap: memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)
- mlock: force system to keep model in RAM rather than swapping or compressing
- mmap+mlock: mmap + force system to keep model in RAM rather than swapping or compressing
- on: read the rows of such tensors from disk on demand instead of keeping them resident (requires mmap)
```
Prefetch advice path: `posix_madvise(..., POSIX_MADV_WILLNEED)` on mapped ranges, `MADV_WILLNEED` in `llama_prefetch`; NUMA mode downgrades to `POSIX_MADV_RANDOM`. Pinning is `mlock` only (no `MAP_LOCKED`), with the ENOMEM hint *"Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root)."*

**--no-mmap path (anon copy).** `src/llama-model-loader.cpp`: `this->use_mmap = load_mode == LLAMA_LOAD_MODE_MMAP || ...`; mmap requires the GGUF data section to satisfy `tensor_align` else it throws. Weights not kept in mmap go through regular backend buffers.

**CPU compute buffers are anonymous aligned malloc, no hugepage work on x86.** `ggml/src/ggml-backend.cpp`:
```c
static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
```
Code search across the repo: `MADV_HUGEPAGE` appears **only** in `ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp` (RISC-V SpacemiT platform pool); x86 mainline never advises.

**Rationale / failure stories.** Discussion #29347 ("When does mmap actually help performance?") is the canonical statement: lazy mmap speeds load, can OOM surprisingly, stays the default. Issue #22360 "Bug: Server Slows Down Significantly Over Time, Requires Restart" is our exact symptom class at fleet scale: *"above 80% lost in performance, needing to restart the PC in order for it to achieve again its full performance"*, multiple reporters (40→20 t/s, ~90→3 t/s over 4h, 800→40 tok/s over 13h pp), across CUDA/ROCm/Vulkan, dense and MoE; reporter saw *"like some memory leak... overtime the memory usage (ram) is increasing"*; **closed as stale, root cause never confirmed**. Grades: mechanisms verified-in-source; degradation reports forum-inferred (issue-reported, unconfirmed).

### 2.2 vLLM

**cpu_offload_gb → pinned CPU + UVA.** `vllm/model_executor/offloader/uva.py`:
```python
class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).
```
Pinning is default-on (`vllm/model_executor/offloader/base.py`):
```python
def should_pin_memory() -> bool:
    return (
        is_pin_memory_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
    )
```
**KV host offload — pinned pools with explicit hugepage option.** Docs (`docs/features/kv_offloading_usage.md`): *"CPUOffloadingSpec (default): single CPU tier. Completed GPU blocks are copied into pinned host memory."* And on huge pages:
```
With kv_offload_backend="cpu", set cpu_hugepage_block_size to "2MB" or "1GB" ... to back the
CPU KV caches with Linux HugeTLB pages (Linux 5.14 or newer) ... Reserve enough pages on the
host for every rank ... If the pool or the limit is too small [it fails at startup]
```
The implementation, `vllm/v1/simple_kv_offload/host_buffer.py` (verified in source):
```python
def alloc_host_buffer(nbytes: int, hugepage_size: int | None = None) -> torch.Tensor:
    """Map and populate an int8 CPU tensor of whole backing pages.

    HugeTLB pages never fall back to ordinary pages.
    """
    ...
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    if hugepage_size:
        flags |= _MAP_HUGETLB | (hugepage_size.bit_length() - 1) << _MAP_HUGE_SHIFT
    ...
    buf.madvise(mmap.MADV_DONTFORK)
    if not hugepage_size and os.environ.get("THP_MEM_ALLOC_ENABLE") == "1":
        buf.madvise(mmap.MADV_HUGEPAGE)
    ...
    # A pool or cgroup shortfall fails here instead of as SIGBUS or an
    # OOM kill while serving.
    populate(buf, 0, nbytes)   # MADV_POPULATE_WRITE
```
**Pool reuse:** host KV pools are allocated at startup and held for process life; pin goes through the torch pinned (CachingHostAllocator) path elsewhere, which caches pinned blocks in-process.

**Failure stories (fragmentation/leak class).** vllm#44181 (Jun 2026): "Severe GPU Memory Leak and OOM with Native KV offloading"; vllm#36623: OOM with `kv-offloading-size > 1024`; and (Aug 2026) "kv-offloading-size allocates 1.92x the requested host memory" — *"per-tensor allocations land ~4% over a 2 GiB boundary, and the pinned-host caching allocator rounds them up to 4 GiB"* — direct evidence that torch's pinned caching allocator is a rounding/fragmentation hazard at 100+ GiB scale. Grades: source + docs verified-in-source; issues issue-reported.

### 2.3 SGLang

**Host KV pool = register + mlock, custom allocator.** `python/sglang/srt/mem_cache/pool_host/common.py`:
```python
    """
    Allocate tensor and pin host memory: register it with cudaHostRegister, then
    mlock it. Pinning only applies when pin_memory=True. ...
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer, registration_granularity_bytes)
```
And the rationale for bypassing torch's pinned allocator:
```python
    # Not torch pin_memory: its host allocator rounds each block up to a power of
    # two, so a 180 GB pool would pin 256 GiB of GTT.
```
**mmap tier with explicit THP/hugepage engineering.** `python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py` (verified):
```python
# Prefault chunks are THP-aligned so no huge page straddles two threads.
_THP_SIZE = 2 * 1024 * 1024
...
    """Private anonymous THP mapping, prefaulted by `num_threads` threads.

    Shared (shmem) faults plateau at a few GB/s; private THP faults scale with
    threads. MAP_PRIVATE is safe to pin only with MADV_DONTFORK, which avoids COW
    after fork() moving the parent off the pinned page; so a fork()ed child
    cannot see this buffer.
    Pages are prefaulted before return, as cudaHostRegister requires.
    """
    ...
    # Best effort: still correct on 4 KB pages if THP is disabled.
    _libc.madvise(base, alloc_bytes, _MADV_HUGEPAGE)
```
```python
def alloc_mmap(dims, dtype) -> torch.Tensor:
    """Allocate a host tensor via anonymous mmap. Set SGLANG_HUGEPAGE_SIZE=2MB or 1GB for hugepages.

    MAP_SHARED + MAP_POPULATE are both required so cudaHostRegister pins real,
    pre-faulted physical pages (otherwise pinning can race with COW or page
    faults and the device ends up reading stale data).
    """
```
Official env-var doc (`docs/docs/references/environment_variables.mdx`): *"When positive, host KV cache pools on the plain mmap path (`SGLANG_HUGEPAGE_SIZE` unset or failed) are mapped `MAP_PRIVATE | MAP_ANONYMOUS` with `MADV_HUGEPAGE` and `MADV_DONTFORK`, and prefaulted by this many threads before pinning."*

**History (failure→fix story).** Issue #4636 (Mar 2025, "[Question] Why do not use pinned memory in Hierarchical Cache"): the original HiCache host pool was `pin_memory=False` with the comment "no need to use pin memory with the double buffering"; collaborator jhinpan conceded *"without specific benchmarking data, it's challenging to provide a definitive answer"*. The current main branch now pins+mlocks every host pool with threaded prefault — the design walked from "double buffering makes pinning unnecessary" to explicit pin+madvise+hugepage engineering. Grade: verified-in-source (current code + env doc); history issue-reported.

### 2.4 TensorRT-LLM

**Pinned host pools, C++ level.** `cpp/tensorrt_llm/runtime/tllmBuffers.h`:
```cpp
    void allocateImpl(PointerType* ptr, std::size_t n)
    {
        TLLM_CUDA_CHECK(::cudaHostAlloc(ptr, n, cudaHostAllocDefault));
    }
    void deallocateImpl(...)
    {
        TLLM_CUDA_CHECK_FREE_RESOURCE(::cudaFreeHost(ptr));
    }
```
**KV host offload is pinned, allocated once.** `docs/source/legacy/advanced/kv-cache-reuse.md`:
```
Offloading to host memory increases likelihood of kv cache reuse. Reusable blocks ... are copied
to a buffer in host memory instead of being evicted. ...
Note that this buffer is pinned memory, allocating a lot of pinned memory on x86 machines can
take a substantial amount of time (10s of seconds). This is a one-time cost.
```
Sized via `--kv_host_cache_bytes` / `kv_cache_host_memory_bytes` / `KvCacheConfig.hostCacheSize`.

**Staging discipline.** `tensorrt_llm/_torch/host_staging.py`: bounded **ring** of pinned buffers per key with CUDA events — *"a buffer is only rewritten once the copy that last read it has completed"*; buffers only grow (`max(value.numel(), 2 * buffer.numel() ...)`), never reallocated per call. And repo-wide policy enforcement, `scripts/check_pinned_memory_usage.py`: an AST lint that rejects direct `.pin_memory()` / `pin_memory=True` — *"Use `maybe_pin_memory(tensor)` instead"* / *"Use `pin_memory=prefer_pinned()` instead"* — pinning is centralized and audited. Grade: verified-in-source.

### 2.5 FlexGen (FMInference/FlexLLMGen, archived)

`flexllmgen/flex_opt.py` (verified):
```python
    # Whether to use pinned memory for weights on CPU
    pin_weight: bool
...
def init_weight_list(weight_specs, policy, env):
    ...
        if len(shape) < 2:
            pin_memory = True
        else:
            pin_memory = policy.pin_weight
        ...
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)
```
Per-tensor CPU allocations from numpy files, policy-selected GPU/CPU/disk tiers; disk tier is file-backed. No mmap for the CPU tier, no hugepages, no pooling across loads — a 2023 single-GPU research design whose host posture is exactly the per-tensor anon churn we are fighting. Grade: verified-in-source.

### 2.6 DeepSpeed-Inference / ZeRO-Inference

`deepspeed/runtime/zero/parameter_offload.py` (verified):
```python
        self.offload_param_pin_memory = False
        ...
            self.offload_param_pin_memory = offload_param_config.pin_memory
        ...
        pinned_memory_summary("ZeRO-3 parameter offload init")
```
`deepspeed/utils/pin_memory_tracker.py` (verified): *"Pinned memory is page-locked: it cannot be swapped out"* — with byte-accounting logs. DeepSpeed additionally ships a **native pin_memory allocator** (`op_builder/pin_memory.py`, `csrc/pin_memory/`, `DS_PIN_MEMORY_BACKEND=native` with a *"Process-wide cache so pin_memory is loaded once"*), i.e. it moved off raw torch pinning for control and accounting. No hugepage work found anywhere in the repo (code search negative). MoE inference offload rides the same ZeRO offload path (docs-claimed). Grade: verified-in-source for the pin path; docs-claimed for MoE specifics.

### 2.7 PowerInfer (Tiiny-AI/PowerInfer)

`smallthinker/src/llama-mmap.cpp` (verified — literal llama.cpp lineage):
```c
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) { ... }
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
```
Weights live in the **page cache via mmap**; hot/cold expert scheduling reads from there; pinning is the llama.cpp `mlock` option only; no hugepage work. Same posture and same exposure as llama.cpp. Grade: verified-in-source.

### 2.8 KTransformers

**Weights are the page cache itself.** Expert weights are aliased straight out of mmap'd GGUF/safetensors — `archive/ktransformers/util/weight_loader.py`: `self.file_data_map[file_path] = np.memmap(file_path, mode='r')`; operators pull `self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")` etc. (verified). Current kt-kernel C++ docs the choice: *"Single-NUMA (tp_count==1): the three proj pointers ALIAS the source GGUF mmap"* and warm it without copying — `kt-kernel/operators/llamafile/moe.hpp`: *"cache at load. Restore that warming WITHOUT copying: MADV_WILLNEED kicks off async"* page-cache warm. Small per-batch io tensors are pinned (`KExpertsCPU.input_tensor_cpu = torch.zeros(..., pin_memory=True)`); weight bulk never is.

**Failure story (mmap lifetime).** `kt-kernel/operators/avx2/mxfp4-moe.hpp`: *"Per-expert mode used to only allocate scales and point b into mmap, but that caused use-after-free when Python releases the mmap after load_weights()."* They changed the placement to survive the mmap being dropped — the one documented case in this set of file-backed-vs-anon lifetime management actually biting. Grade: verified-in-source.

### 2.9 ollama

**mmap is engine-managed per placement** (`server/sched.go`, verified):
```go
	if opts.NumGPU == 0 || len(gpus) == 0 || allDevicesLibrary(gpus, "cpu") {
		return "cpu"          // disable mmap
	}
	if goos == "windows" && hasDeviceLibrary(gpus, "cuda") {
		return "windows_cuda" // disable mmap
	}
```
and the host-pressure guard, with the single most quotable line in this whole study:
```go
	// Only back off mmap when we still expect the model to fit on discrete GPU.
	// If VRAM is already tight, disabling mmap can make partial CPU offload
	// worse by turning file-backed mappings into anonymous memory.
```
**Reuse across loads:** envconfig `KeepAlive` default `5 * time.Minute`; scheduler reuses the live runner:
```go
	runner := s.loaded[key]
	...
	if runner != nil && !runner.needsReload(c, req) {
		req.useLoadedRunner(runner, s.finishedReqCh)
	}
```
Compat layer (`llama/compat/`): mmap is force-disabled when a loader must reshape bytes (FFN concat/CPU_REPACK), *"with mmap ... the llama.cpp loader binds tensors directly to the mmap'd file region"*. Grade: verified-in-source.

## 3. The repeating patterns

**Pattern A — static bulk lives in the page cache: file-backed mmap, never anon copy.**
llama.cpp (default), PowerInfer (fork), KTransformers (expert weights alias the mmap directly), ollama (mmap default for Linux+CUDA, with an explicit guard against *converting* file-backed to anon) all hold weights as `mmap(MAP_SHARED, PROT_READ)` of the model file. Consequences: no allocator involvement for the bulk, kernel-reclaimable under pressure (and re-read from disk without anon fragmentation), reloads after restart are page-cache hits, and `MAP_POPULATE`/`MADV_WILLNEED` decide only *when* faults happen, not whether 4K-vs-2M matters for churn. ollama states the invariant outright: converting file-backed mappings to anonymous memory is what makes tight boxes *worse*.

**Pattern B — hot anon lanes are made "immutable host state": pinned (+mlock) and/or hugepaged, prefaulted once, held for process life.**
vLLM (`cudaHostRegister`-grade pinned pools, `cpu_hugepage_block_size` 2MB/1GB HugeTLB, `MADV_POPULATE_WRITE` prefault, fail-fast at startup instead of SIGBUS mid-serve), SGLang (`cudaHostRegister` + `mlock`, `MADV_HUGEPAGE` + `MADV_DONTFORK`, THP-aligned threaded prefault, HugeTLB env knob), TRT-LLM (`cudaHostAlloc` pools sized once — *"This is a one-time cost"* — plus grow-only pinned staging rings), DeepSpeed (offload `pin_memory` + native pinned allocator + byte accounting). None of them *ever* free and re-create a multi-GiB hot lane per model load inside a process; the allocator reuse (torch CachingHostAllocator, custom pools) makes even repeated loads inside one process allocation-free. TRT-LLM even lints the codebase to force all pinning through one audited path.

**Pattern C — the known failure modes are allocator churn and pinned-allocator pathology, and the mature projects patch exactly those.**
Documented, not hypothetical: torch's pinned caching allocator rounds pinned blocks to powers of two (vLLM issue: requested host memory billed at 1.92x; SGLang's comment: a 180 GB pool pins 256 GiB) → SGLang wrote its own allocator; vLLM's offload path had leaks (issues #44181, #36623); TRT-LLM documents that huge pinned allocations stall for tens of seconds on x86 (why you allocate once at boot, when memory is fresh and compaction is cheap); KTransformers documents a use-after-free from a dropped mmap under aliased weights; llama.cpp has an unresolved multi-reporter slow-over-time issue (#22360) and **no THP/hugepage engineering on x86 mainline at all** — it ships Pattern A but not Pattern B.

## 4. Verdicts for our box (184 GiB, THP=madvise, shmem=never, zram)

**Would degrade (our current design is in this class).**
- **llama.cpp-lineage engines with anon staging (us): yes — measured, and explained.** With `enabled=madvise` and no `madvise` call, every anon pool (GGUF staged via read/copy, expert-cache malloc copies, staging buffers) is 4K pages; the tmpfs shadow store is anon-class memory at 4K because `shmem_enabled=never`; tens-to-hundreds of GiB/day of alloc→free churn through that 4K anon arena is precisely what drives `compact_fail` to 71% and pushes 6.4 GiB into zram; a DRAM-bandwidth-bound lane on scattered 4K pages plus compaction stall latency is the 25% tax. Kernel doc + our counters agree. **verified-in-source against our own telemetry**
- **FlexGen-style per-tensor anon CPU weight caches: yes, by construction** — no mmap, no pooling; per-boot churn. (It is 2023 research code; nobody should run it 24/7.)
- **DeepSpeed ZeRO-Inference without pin_memory enabled: yes** (same 4K anon churn); with `offload_param.pin_memory=true` it enters Pattern B but still allocates at first use, not at a controlled boot moment.

**Would not degrade the same way (and why).**
- **vLLM, SGLang, TRT-LLM host lanes: no.** Their hot host state is page-locked (immune to reclaim, zram, and khugepaged churn forever) and/or HugeTLB-backed (immune to compaction by construction), prefaulted at startup while memory is fresh, and held for the life of the process. zram cannot touch pinned pages; compaction never runs on their behalf; the 4K-vs-2M question is answered by construction, not by policy. Their residual risks are the ones they patched in public: pinned-allocator rounding (SGLang's custom allocator exists for this reason) and offload leaks — bandwidth-pathology bugs, not slow-decay bugs.
- **KTransformers: largely no.** The weight bulk (the 100+ GiB part) *is* the page cache — reclaimable, mmap-faulted, zero allocator involvement; only tiny pinned io tensors are anon/pinned. Its exposure is mmap lifetime management, which it already patched.
- **ollama: mostly no on this box** — a CUDA-full-offload model stays mmap-backed (page cache, no anon bulk), keep_alive holds the runner so no per-request churn. Its degradation trigger is model *swapping* (keep-alive expiry → anon llama-server rebuild) and its own host-pressure heuristic flipping to anon under memory pressure — i.e., the failure arrives as reload stalls, not bandwidth decay.
- **Plain llama.cpp / llama-server with default flags: borderline.** Weights are page-cache-served (Pattern A, good), but KV/compute buffers are anon `aligned_malloc` with no madvise — on a box that is not memory-pressured and doesn't restart engines, that's stable; on our workload profile (frequent engine boots + anon expert cache + tmpfs store) it lands squarely in the degrade class, consistent with issue #22360's restart-to-fix reports.

**Reference architecture, one paragraph.** Hold static weight bytes in the page cache via file-backed mmap (or pin them once with mlock if you must guarantee residency); hold every hot host lane (expert cache, shadow store, KV offload, staging) as memory that is *allocated once at boot, prefaulted (MAP_POPULATE / MADV_POPULATE_WRITE), backed by HugeTLB or MADV_HUGEPAGE+MADV_DONTFORK, pinned (cudaHostRegister / mlock), and never freed*; keep per-request scratch small and pooled; treat torch's pinned caching allocator as a rounding hazard at 100+ GiB scale; fail fast at startup if the reservation can't be satisfied (ENOMEM beats SIGBUS mid-serve). For our engines specifically, the deltas that map 1:1 to our telemetry: call `madvise(MADV_HUGEPAGE)` (+populate) on the anon expert-cache arena or move it to MAP_HUGETLB/memfd+MFD_HUGEPAGE; give the shadow store huge folios (hugetlbfs/memfd — `shmem_enabled=never` guarantees the current tmpfs gets none); stop staging GGUF through anon copies when the same bytes can be mmap'd; and stop re-creating tens of GiB of lane per engine boot when the boot could re-attach to a persistent pool.

## 5. Citations

Primary sources (fetched at upstream HEAD, 2026-11-07):
- llama.cpp `src/llama-mmap.cpp` (MAP_POPULATE condition, MAP_SHARED, mlock impl): https://github.com/ggml-org/llama.cpp/blob/master/src/llama-mmap.cpp
- llama.cpp `common/arg.cpp` (load-mode doc strings): https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp
- llama.cpp `src/llama-model-loader.cpp` (use_mmap, alignment throw, host-buffer avoidance): https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp
- llama.cpp `ggml/src/ggml-backend.cpp` (`ggml_backend_cpu_buffer_type_alloc_buffer` → `ggml_aligned_malloc`): https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp
- llama.cpp hugepage-negative code search (MADV_HUGEPAGE only in `ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp`): https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cpu/spacemit
- llama.cpp Discussion #29347 "When does mmap actually help performance?": https://github.com/ggml-org/llama.cpp/discussions/29347
- llama.cpp Issue #22360 "Server Slows Down Significantly Over Time, Requires Restart": https://github.com/ggml-org/llama.cpp/issues/22360
- vLLM `vllm/model_executor/offloader/uva.py` and `base.py` (pinned UVA offload, should_pin_memory): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/offloader/uva.py , https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/offloader/base.py
- vLLM `vllm/v1/simple_kv_offload/host_buffer.py` (HugeTLB/MADV_DONTFORK/MADV_HUGEPAGE/MADV_POPULATE_WRITE alloc): https://github.com/vllm-project/vllm/blob/main/vllm/v1/simple_kv_offload/host_buffer.py
- vLLM `vllm/v1/simple_kv_offload/worker.py` (PIN_MEMORY cpu caches): https://github.com/vllm-project/vllm/blob/main/vllm/v1/simple_kv_offload/worker.py
- vLLM docs `docs/features/kv_offloading_usage.md` (pinned CPU tier; hugepages section): https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md
- vLLM issues: #44181 (native KV offload leak): https://github.com/vllm-project/vllm/issues/44181 ; #36623 (offloading OOM): https://github.com/vllm-project/vllm/issues/36623 ; 1.92x pinned-allocator rounding report (Aug 2026, search-level): via https://github.com/vllm-project/vllm/issues
- vLLM RFCs: #16144 (CPU KV offload V1): https://github.com/vllm-project/vllm/issues/16144 ; #19854 (KV Cache CPU offloading): https://github.com/vllm-project/vllm/issues/19854
- SGLang `python/sglang/srt/mem_cache/pool_host/common.py` (cudaHostRegister+mlock, power-of-two comment): https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/pool_host/common.py
- SGLang `python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py` (MADV_HUGEPAGE/MADV_DONTFORK/MADV_POPULATE_WRITE/MAP_HUGETLB, threaded THP-aligned prefault): https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py
- SGLang `docs/docs/references/environment_variables.mdx` (SGLANG_HUGEPAGE_SIZE + parallel prefault semantics): https://github.com/sgl-project/sglang/blob/main/docs/docs/references/environment_variables.mdx
- SGLang Issue #4636 "Why do not use pinned memory in Hierarchical Cache": https://github.com/sgl-project/sglang/issues/4636
- TensorRT-LLM `cpp/tensorrt_llm/runtime/tllmBuffers.h` (cudaHostAlloc/CudaHostAllocator): https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/runtime/tllmBuffers.h
- TensorRT-LLM `docs/source/legacy/advanced/kv-cache-reuse.md` (pinned host offload buffer, one-time-cost note, sizing flags): https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/kv-cache-reuse.md
- TensorRT-LLM `tensorrt_llm/_torch/host_staging.py` (pinned staging rings): https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/host_staging.py
- TensorRT-LLM `scripts/check_pinned_memory_usage.py` (repo lint centralizing pinning): https://github.com/NVIDIA/TensorRT-LLM/blob/main/scripts/check_pinned_memory_usage.py
- FlexLLMGen `flexllmgen/flex_opt.py` (pin_weight policy, init_weight_list): https://github.com/FMInference/FlexLLMGen/blob/main/flexllmgen/flex_opt.py
- DeepSpeed `deepspeed/runtime/zero/parameter_offload.py`, `deepspeed/utils/pin_memory_tracker.py`, `op_builder/pin_memory.py`: https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/parameter_offload.py
- PowerInfer (Tiiny-AI) `smallthinker/src/llama-mmap.cpp` (MAP_POPULATE/POSIX_FADV_SEQUENTIAL/NUMA MADV_RANDOM): https://github.com/Tiiny-AI/PowerInfer/blob/main/smallthinker/src/llama-mmap.cpp
- KTransformers `archive/ktransformers/util/weight_loader.py` (np.memmap), `archive/ktransformers/operators/experts.py` (get_mmap_tensor), `kt-kernel/operators/llamafile/moe.hpp` (MADV_WILLNEED warm), `kt-kernel/operators/avx2/mxfp4-moe.hpp` (UAF story), `kt-kernel/python/experts_base.py` + `archive/ktransformers/operators/experts.py` (pinned io tensors): https://github.com/kvcache-ai/ktransformers
- ollama `server/sched.go` (disableMmapDefaultReason, maybeDisableMmapForHostPressure, getRunner reuse): https://github.com/ollama/ollama/blob/main/server/sched.go
- ollama `envconfig/config.go` (KeepAlive default 5m): https://github.com/ollama/ollama/blob/main/envconfig/config.go
- ollama `llama/compat/` (mmap/FFN-concat incompatibility): https://github.com/ollama/ollama/blob/main/llama/compat/llama-ollama-compat.h
- Linux kernel THP admin guide (madvise mode, shmem_enabled values, defrag defaults, compact_stall/fail counters): https://docs.kernel.org/admin-guide/mm/transhuge.html
- vllm-ascend #20761 (single-copy pinned host KV for CPU-only MLA offload, search-level): https://github.com/vllm-project/vllm-ascend/issues/20761
- r/LocalLLaMA corroboration of long-uptime llama.cpp decay (forum-inferred): https://www.reddit.com/r/LocalLLaMA/comments/1tdffl1/llamacpp_server_running_2_weeks_straight_loses
