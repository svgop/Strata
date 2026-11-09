---
type: research-digest
date: 2026-11-09
topic: llama.cpp-lineage upstream performance regressions — causes, fixes, and mapped symptoms
case: strata-custom-2 (MoE + CPU-offloaded experts, RTX 5090, 184 GiB host)
method: GitHub issue/PR mining via authenticated gh API (ggml-org/llama.cpp, ggml-org/ggml, sgl-project/sglang) + web search; every claim carries its issue number and resolution state
sources-primary: ggml-org/llama.cpp issues/PRs, ggml-org/ggml, sgl-project/sglang, NVIDIA forums, NVIDIA/open-gpu-kernel-modules
---

# Upstream regression digest: decay, serial GPU-reach waits, host-memory policy, 12 ms stalls

Research question: what does the llama.cpp lineage (ggml, llama.cpp, ollama, vLLM, SGLang) have on
record that looks like (1) week-long decode decay cured by reboot/engine-restart, (2) a ~12 ms serial
per-window device→host flag wait while batched lanes pay 0.5 ms, (3) host-memory-policy sensitivity
(hugepage pool worse, tmpfs shadow, cudaHostRegister'd pinned buffers), (4) disproportionate serial-lane
loss under host memory pressure.

---

## 1. The decay-restart class ("slower over time, restart fixes it")

### 1.1 The anchor case: ggml-org/llama.cpp #22360 (closed/stale — but root-caused in comments)
- URL: https://github.com/ggml-org/llama.cpp/issues/22360 — "Bug: Server Slows Down Significantly Over
  Time, Requires Frequent Reboots". Vulkan (7900 XTX), then cross-backend reports. Closed stale,
  `bug-unconfirmed` at the tracker level. The task brief's summary ("root cause unconfirmed") is true of
  the issue body, but **the comment thread contains a root-cause analysis** that upstream never promoted
  to the issue title:
- Commenter `domainsquatter` (two long comments, backed by a stdlib-Python reproducer against the
  official `ghcr.io/ggml-org/llama.cpp:server-cuda13` image, 75 measurement rows, three independently
  built binaries on GB10/DGX Spark) decomposed the symptom into **two separate faults**:
  1. **RSS growth is a separate fault** from the throughput loss: "the server served 178.9 tok/s at 8
     clients while holding 5.8 GB, and 97.5 while holding ~2.6 GB" — memory growth does not track decay.
  2. **The slowdown follows the prompt, not the server.** After one 16-wide burst over prompt set A, on
     the same process, same minute: re-serving prompt set A = **8.5 tok/s per request**; serving
     unseen prompts = **21.2 tok/s per request**. Verbatim: *"'the server degrades and needs a restart'
     is the wrong model... A restart does fix it, but only because the cache goes with the process;
     sending a different prompt fixes it just as well."*
  3. Bonus mechanism: a **one-client occupancy cliff** — decode per slot drops 59–67% when in-flight
     goes 16→15 (16.8–17.4 → ~10.2 tok/s per slot), and 16 clients → 262 tok/s vs 15 → 146. Batch-shape
     collapse, not aging.
  - Consequence for us: any "week-long decay cured by reboot" that is really "long-lived server whose
    prompt-cache/slot state degraded" is upstream-documented as **per-prompt KV/slot state**, and engine
    restart and reboot are observationally equivalent because both drop the cache.
- Corroborating reports in the same thread (decay + restart-cures): `dannycycheng` (9070 XT: 40→20 t/s
  over hours), `Denkisen` (ROCm: 90→3 t/s over ~4 h), `ray73864` (Strix Halo Vulkan: pp 800→40 t/s over
  13 h), `EcceGratum` (R9700 ROCm: "Restarting llamacpp docker container fixes it"), `Ezzz-dev`
  (RAM grows "no matter the situation... fixes after restarting llama.cpp/LM Studio").

### 1.2 Confirmed per-request/per-token decay mechanisms with code-level owners
- **Per-request pinned-buffer teardown — ggml-org/llama.cpp #28872 (open draft PR: "llama : keep the
  backend scheduler on re-reserve instead of recreating it") + its fixed-input reproduction #31840
  (open: "backend sampling repeatedly reallocates pinned host compute buffers as requests finish")**.
  Mechanism: `llama_context::sched_reserve()` recreates the whole `ggml_backend_sched` whenever
  `sched_need_reserve` is set; `llama-server` calls `llama_set_sampler()` **for every request** (backend
  sampler attach/detach), so every request: (1) frees and reallocates all compute buffers *and the
  pinned host input buffer* — measured with CUPTI at `-ub 1024`, 262K ctx: **`cudaFreeHost` 115 ms +
  `cudaMallocHost` 424 ms per request** on a ~512 MB pinned buffer; (2) page-faults on the fresh buffers
  during first decode steps ("per-request warm-up"). Effect of the fix on a 4× GPU layer-split
  Qwen3.8-Flash-Next deployment: TTFT 0.8–1.1 s → 0.25–0.4 s, identical outputs. This is the closest
  documented mechanism for **traffic-correlated degradation that a restart clears** — it scales with
  request count, not uptime. Not merged upstream as of 2026-11-09.
- **CUDA/HIP graph cache growth — #25082** (closed stale without a merged fix): "system RAM grows
  unbounded with parallel slots due to CUDA graph cache never being evicted". Community PRs #25101 and
  #25114 ("trim CUDA/HIP graph cache when all slots are idle") were **closed unmerged**. Upstream later
  shipped eviction machinery in the other direction: **#30662 (merged 2026-11-03) "cuda : prevent
  premature eviction of graphs about to be used"** — i.e. the graph cache can both grow unbounded
  (#25082) and thrash (evict a graph that is about to be reused, forcing re-capture stalls). Both are
  decode-traffic-correlated decay/shape mechanisms.
- **Per-token host leak — #25740** (closed): "~0.15 MB/generated token, backend-independent — reproduces
  on CPU, RADV, and lavapipe". Host-RAM growth per token is a real upstream bug class; closing
  resolution PR not retrieved (issue predates the b10xxx era).
- Other growth reports: #22925 (closed, "System Memory leak 9010"), #26143 (open, RPC node cache grows
  without bound), #31127 (open, 2×V100 accumulates ~93 GB shmem-rss → kernel OOM under numactl),
  #27845/#23486/#28008 (SYCL/Vulkan leaks, closed).
- Historical: #832 (2023, "Performance degrading over time", closed) — the oldest decay report;
  #5856 (open) is quantization-pipeline decay, not runtime.

---

## 2. Serial-lane GPU-reach waits: the router-readback hard sync and the blocking `tensor_get`

### 2.1 Origin of the per-layer host sync: PR #15346 (merged 2025-08-20)
- URL: https://github.com/ggml-org/llama.cpp/pull/15346 — "sched : copy only the used experts when
  offloading prompt processing". Introduced **selective expert copying**, which requires the routing ids
  **on the host** to know which experts to upload. Per the #25859 analysis: "each layer does a D2H
  readback of the ids before the copies start. That readback is a hard sync point: compute has to finish
  producing the ids, then the pipeline stalls while they come back, then uploads begin." This is
  structurally identical to our serial lane spinning on a device-written flag per layer-group.

### 2.2 #25859 (open): "Offloaded-MoE prefill leaves the GPU idle waiting on serial expert H2D copies (single-GPU -ncmoe)"
- URL: https://github.com/ggml-org/llama.cpp/issues/25859. RTX 3060/PCIe4, Qwen3.6-35B-A3B `-ncmoe 26`:
  GPU idle **~42%** of prefill; copies run serially with the consuming matmul ("the single-GPU offload
  path always gets `n_copies = 1`").
- Measured fixes (token-identical at temp 0):
  1. **Page-lock the mmap-backed CPU expert weights** so H2D DMAs directly instead of the pageable
     bounce buffer: **+21% alone** (prefill 1143 → ~1400 t/s class).
  2. **Overlap uploads with compute on a second stream with staging slots**: combined **+64%**
     (1143 → 1880 t/s); GPU idle 42% → a few percent.
- Decode-side independent datapoint (`yalun753`, RTX 4090, Qwen3.6-35B IQ2_M, experts in host RAM):
  cudaHostRegister whole-tensor merged registration: **46.8 → 50.2 t/s decode**; overlap on second
  stream at bs=1: **no gain** (single expert ~1 MB = ~60 µs copy vs <10 µs kernel); selective top-K
  pinning with on-demand fallback, **~0.43 ms per pin on first touch**; on DeepSeek-V4-Flash (43×256
  experts, scattered routing) staged copy was *slower* — "instead of overlapping the copy we remove it".
- `mobilinkd` datapoint: upstream #21067 with `n_copies=1` "rarely keeps the pipeline from stalling";
  the fork's tunable ring `GGML_SCHED_PREFETCH_EXPERTS=4` gives **17–20% prefill** at a 2.4 GB cost.
- Also in thread: ubatch is a 2.0× prefill lever on the offload path (ub512→ub4096: 77→156 t/s), because
  at large batch the selected-expert union ≈ all experts — "you pay the sync to learn 'upload everything'".

### 2.3 #21067 (open PoC PR): "ggml: allow prefetching tensor overrides"
- URL: https://github.com/ggml-org/llama.cpp/pull/21067. `--prefetch-weights` flag (CUDA only);
  prefetch layer N+1 weights during layer N compute. Key invariant stated in the PR: **"`--no-mmap` is
  necessary for this work, otherwise the operations are implicitly serialized due to the weights not
  being pinned."** — upstream's own statement that mmap-backed (pageable) expert weights serialize
  against compute. Still open; never merged.

### 2.4 The blocking-`tensor_get` stall and its merged fix chain (#30252 → #30881 → #31391)
- **#30252** (closed): "server : saving idle slots to the prompt cache stalls all requests for seconds
  with interleaved KV cells". With unified KV + `--cache-idle-slots`, saving a 323-token sequence took
  **23,256 blocking `ggml_backend_tensor_get` calls** (one range per token × per layer × K/V), each a
  full device→host synchronization; each idle-slot save blocked 0.7–0.9 s while all other requests wait.
  This is the canonical upstream statement that **many small blocking D2H reads are catastrophic**
  versus batched reads.
- **#30881 (merged 2026-10-29)**: "ggml : add batched tensor copy API" — `ggml_backend_tensor_get_n /
  set_n / copy_n` with parallel arrays; backend-specific batching for mixed-buffer batches; generic
  fallback decomposes to per-tensor calls. URL: https://github.com/ggml-org/llama.cpp/pull/30881
- **#31391 (merged 2026-11-03)**: "cuda : gather batched tensor reads into mapped host memory (PR to
  master)" (successor of closed #30986). Mechanism: temporarily **registers and maps the CPU
  destination** (cudaHostRegister under RAII), uploads a descriptor array for the GPU source ranges,
  then **one kernel** copies all ranges directly into CPU memory; fallback keeps the ordered-copy path.
  URL: https://github.com/ggml-org/llama.cpp/pull/31391
  - RTX 5090, 64 streaming requests (Llama 3.1-8B Q4_K_M, np8): aggregate t/s **592.75 → 774.11
    (+30.6%)**; mean TTFT **385 → 151 ms (−60.9%)**; mean inter-token latency **12.12 → 9.30 ms
    (−23.3%)**; longest content gap **822 → 164 ms (−80.1%)**; snapshot-save interval **266.5 → 24.8 ms
    (−90.7%)**.
- **#30694 (closed unmerged, superseded by #31391)**: "llama : merge nearby KV ranges when reading state
  into host memory" — merge consecutive sub-64 KiB ranges of the same device tensor into one read.
  Useful as a design reference for range-merging.

### 2.5 #29796 (open): "cuda: skip redundant stream synchronization"
- URL: https://github.com/ggml-org/llama.cpp/pull/29796. "Repeated synchronization calls can enter the
  CUDA runtime even when no stream work has been submitted since the previous wait." Tracks stream-use
  state; a sync returns immediately if the stream was not re-acquired. Directly applicable to a host
  lane that spins on a device-written flag: the *first* wait pays, redundant waits should be free.
  Not merged as of 2026-11-09.

---

## 3. Host-memory policies: pinning, splitting, over-pinning, hugepages, direct I/O

### 3.1 The pinned-buffer 4 GB split chain: #30449 → #30793 → #30926
- **#30449 (merged 2026-10-16)** "Limit max CUDA pinned host buffer size to 4 GB". Problem: with
  `--load-mode none`, CPU-resident MoE weights go into **one giant `cudaMallocHost` buffer** (56 GiB for
  Qwen3.8-Flash-Next `-cmoe`); allocation fails (Windows first) and ggml **silently falls back to
  pageable memory** — "without pinned memory allocation, streaming of experts to GPU memory is much
  slower". Fix: cap pinned host buffers at 4 GiB so the allocator splits into several pinned buffers.
  RTX 5090 Windows result (`-cmoe`, `--load-mode none`, MTP): pp **73.28 → 110.66 t/s (+51%)**, pred
  **37.76 → 62.73 t/s (+66%)**. URL: https://github.com/ggml-org/llama.cpp/pull/30449
- **Collateral: #30793 (closed)** "Regression: first-inference TTFT 10s → 60s+ with large CPU-resident
  MoE weights after #30449" — split buffers have **no single base pointer, so `--load-mode mlock`
  skipped locking them**; an unpinned chunk could be paged out after load → first request 60 s+.
  Reported on EPYC Turin / 384 GB / 2×4080, GLM-5.3-Flash Q8_0 ~341 GB, `--ncmoe 44 --load-mode mlock`.
- **Fix: #30926 (merged 2026-10-22)** "Fix #30793" — `load_tensors` now locks **every distinct buffer**
  (each chunk of a split buffer), not just whole buffers. URL: https://github.com/ggml-org/llama.cpp/pull/30926
- Lessons: (a) silent pageable fallback is the disaster mode; (b) pinning and mlock are different
  mechanisms and both must cover split buffers; (c) a 4 GiB chunk cap is upstream's accepted shape for
  huge pinned pools.

### 3.2 Over-pinning fix: #30405 (merged 2026-10-13) "llama : do not map foreign ranges into a backend buffer"
- A host-pointer backend buffer **pins everything it wraps**. The loader splits tensors across several
  `ggml_context`s (per backend buffer type + lazy tensors), but a context's host-pointer buffer spanned
  first→last tensor in the file, wrapping **foreign ranges owned by other contexts** — e.g. pinning the
  entire 27.5 GiB `per_layer_token_embd` lazy tensor into the GPU buffer. Fix: `get_mapping_ranges()`
  subtracts other contexts' ranges; one host-pointer buffer per needed range; macOS `llama_prefetch()`
  enabled for on-demand ranges. Measured: mapped bytes 97.3 GB → 69.2 GB; cold PP 356 → 788 t/s with
  prefetch. URL: https://github.com/ggml-org/llama.cpp/pull/30405
- Direct relevance: our cudaHostRegister footprint must be range-exact; wrapping extra ranges pins host
  memory that inflates page pressure and worsens exactly the "batched vs serial under host memory
  pressure" split we observe.

### 3.3 load-mode migration (mmap/mlock/direct-io semantics)
- #20834 (merged 2026-07-23): refactor mlock/mmap/directio into `--load-mode`; #26934 (merged
  2026-08-15): migrate deprecated `--mmap/--no-mmap` to `--load-mode`; #26135 (merged 2026-07-27): add
  `-lm mlock` = mlock **without** mmap (weights private, never disk-backed, no page-cache eviction);
  #28334 (merged 2026-09-09): remove the old flags; #31321 (merged 2026-10-29): reuse the tensor staging
  buffer and skip the direct-io bounce buffer; #22877 (merged 2026-08-18): quantize evicts weights after
  each layer. A `-lm mlock` (mlock-no-mmap) mode is the upstream-endorsed shape for "tmpfs-like"
  staging without tmpfs: private, pinned, page-fault-free.
- #29130 (closed): "Slow prompt processing when MoE experts are read from disk via mmap — an
  optimization using O_DIRECT" — documents the failure mode where mmap-backed experts fall out of page
  cache and prefill drops off a cliff (disk-bound). Related: #27562 (closed, JIT expert streaming from
  storage), #31452 (open RFC, zero-OOM mmap alternative), #17546 (open, cpu-moe failures).

### 3.4 Hugepages / THP: experiments exist, nothing merged
- #2251 (2023 feature request, closed), #12444 (closed; claims up to 10× load-speedup potential),
  **#21821 (closed unmerged)** `--hugepages` HugeTLB-backed weight loading (env `LLAMA_ARG_HUGEPAGES`);
  **#12552 (open)** `llama-map` hugetlbfs 2M/1G via `-DGGML_USING_HUGE_PAGE_1G=1` (requires model file
  on hugetlbfs mount + kernel boot params); **#22022 (open)** `MADV_HUGEPAGE` hint on the weights mmap
  (THP `madvise` mode; ~1M 4K pages → ~2K 2M pages; "reduces re-fault latency spikes under memory
  pressure"; neutral on unloaded machines); **#7420 (open)** `--direct-io` + THP (bypass page cache:
  3–6× faster uncached load, no page-cache pollution).
- Verdict: **upstream never merged a hugepage pool**; the only still-alive proposals are *advisory*
  THP hints and O_DIRECT. Our finding that a hugepages pool made things worse has no upstream
  counter-example; the upstream-preferred alternatives are pinned buffers (#30449/#25859) and
  mlock-no-mmap (#26135).

### 3.5 Other pinned-memory items
- #31825 (open): improve large MoE loading time with offloading — discusses cudaHostRegister cost at
  load; #26448 (open): run MoE experts from host RAM via PCIe DMA (no H2D copy; UVA-direct mmap);
  ggml-org/ggml **#1495 (open)**: `ggml_backend_cuda_buffer_from_ptr` for UVA-direct mmap inference;
  #27038 (closed) SYCL host-pinned high CPU utilization for large allocations → fixed by #27559
  (merged 2026-09-01, cap host-pinned alloc at 2 GB); #29604 (merged 2026-09-30) SYCL reduce allreduce
  sync with pinned host buffers; #29608 (merged 2026-10-08) SYCL pinned ring buffer for bulk uploads;
  #30468 (merged 2026-10-16) Vulkan gpu-visible host memory fixes; #31538 (open) iGPU CPU-resident
  layers consume device memory via host buffer.

---

## 4. CUDA-graph lineage (multi-layer windows, cache thrash, recapture)
- **#30804 (merged 2026-10-23)** "CUDA: chunk graphs to amortize graph launch" — graphs are captured in
  chunks; follow-up build fix #31053 (merged 2026-10-23); reservation fix #31344 (merged 2026-10-29);
  debug env + env-read-once #30327 (merged 2026-10-13).
- **#30662 (merged 2026-11-03)** "cuda : prevent premature eviction of graphs about to be used" —
  graph-cache eviction was discarding graphs shortly before reuse → re-capture stalls (decode jitter).
- **#26771 (merged 2026-10-20)** "CUDA: Recapture graph after memory pool flush" — stale graph pointers
  after a pool flush; a graph replayed against flushed memory is a correctness/perf hazard.
- Growth side: #25082 (closed stale) unbounded graph-cache RAM under parallel slots; PRs #25101/#25114
  closed unmerged — no upstream fix for cache *trimming* shipped; only *retention* (#30662).
- #28725 (merged 2026-10-23) SYCL graph record/replay; #28549 (merged 2026-09-16) CUDA graph for MTP
  draft.

---

## 5. RTX 5090 stalls, watchdog kills, and the "12 ms" quantization

### 5.1 Watchdog/GSP class (seconds-scale, RTX 5090-specific)
- **#27102 (open)** "CUDA kernel stall during model execution, killed by watchdog": RTX Pro 6000
  Blackwell + two independent RTX 5090 reproductions (driver 610.57.04, CUDA 13.2/13.3): sustained
  generation → locked GPU channel → **XID 8** RC watchdog kill. Non-deterministic. Workaround (verified
  by a reporter on the exact repro): `options nvidia NVreg_RegistryDwords="RmWatchDogTimeout=60"` in
  `/etc/modprobe.d/nvidia.conf`; ggerganov confirms seeing it on RTX 5090 "at random times under heavy
  load" and points to the same advice in **#20146** (closed, "CUDA crash on tensor copy", 2×5070 Ti).
  URL: https://github.com/ggml-org/llama.cpp/issues/27102
- **#27910 (closed)** "CUDA inference on Qwen3.8-27B Q6_K causes RTX 5090 display loss and NVIDIA
  GSP/full-chip reset on Linux".
- GSP firmware as stall source (web): NVIDIA forum Xid 79 GSP crash under sustained CUDA load (RTX
  5090): https://forums.developer.nvidia.com/t/bug-report-fix-rtx-5090-xid-79-gsp-firmware-crash-under-sustained-cuda-load/369440;
  Xid 62 GSP watchdog on RTX 5080 during LLM inference: https://forums.developer.nvidia.com/t/bug-xid-62-gsp-watchdog-timeout-on-rtx-5080-during-high-context-llm-inference/371272;
  GSP halt on sm_120: https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1111. `NVreg_EnableGpuFirmware=0`
  is pre-Turing-only. No upstream llama.cpp fix; module-option workarounds only.
- Related pressure-shaped stalls: #30726 (open) Metal multi-minute stalls — "compute time normal, wall
  time balloons 10-50x"; #31261 (open) Vulkan decode "degrades over repetitions" −35%.

### 5.2 The ~12 ms figure: what upstream actually documents
- The only 12 ms-scale number in a merged-fix benchmark is the **#31391 baseline**: master's mean
  inter-token latency 12.12 ms vs 9.30 ms after gathering — an *aggregate of many sync round trips*, not
  a fixed driver quantum. No upstream source documents a Linux ~10-15 ms fixed stall from driver timer
  batching (that is a WDDM-class phenomenon; Linux paths show µs-scale sync round trips that become
  100 ms–seconds scale only when multiplied, per #30252's 23,256 blocking reads → 0.7–0.9 s per save).
- `CUDA_DEVICE_MAX_CONNECTIONS` appears upstream only in **#19219** (closed, MoE decode hang on Jetson
  Orin since b7309). No stream-priority work merged in llama.cpp CUDA. #2134 (closed, 2024)
  `sched_yield()` overhead +25-40% — the CPU-side spin/yield analog, fixed by the ggml thread pool
  (ggml #400).

---

## 6. vLLM / SGLang / ollama host-memory findings
- vLLM `--cpu-offload-gb`: community-documented ~3.5× throughput tax once weights spill to host
  (r/Vllm "CPU Offloading is a Trap for Production", late 2025); slow even when VRAM suffices;
  #36279 (Windows assertion); UVM offload shows anomalous GPU UVM utilization on GH200 (NVIDIA forum,
  Apr 2025). Production guidance: pinned-pool offload only for KV/prefix-cache tiers
  (OffloadingConnector + LMCache), not weight spill. Docs: https://docs.vllm.ai/en/latest/features/kv_offloading_usage
- SGLang HiCache (pinned host KV pool): design doc https://docs.sglang.ai/advanced_features/hicache_design.html;
  LMSYS blog 2025-09-10 (tier-move latency beyond plain cudaMemcpyAsync): https://www.lmsys.org/blog/2025-09-10-sglang-hicache;
  **#21880** HiCache pathologically slow in containers (storage→CPU prefetch, write-bandwidth bound):
  https://github.com/sgl-project/sglang/issues/21880; discussion **#23242** (Apr 2026): RSS growth over
  time traced to pinned host memory in the serving process — the pinned-pool-grows class again:
  https://github.com/sgl-project/sglang/discussions/23242
- ollama: inherits the llama.cpp page-cache/spill behavior; popular writeups converge on `--mlock` /
  no-swap as the decay fix; no distinct merged ollama-side fix surfaced.

---

## 7. Mapping to our four symptoms

| Our symptom | Closest documented upstream match | Owner | State (2026-11-09) |
|---|---|---|---|
| (1) week-long decode decay; reboot cures; restart also cures in some reports | #22360 — per-prompt KV/slot-state fault ("slowdown follows the prompt"; restart = cache drop) + separate RSS-growth fault; traffic-correlated per-request churn #28872/#31840; graph-cache growth #25082 | #22360 comments (unpromoted); #28872 draft | decay: root-cause comment only; churn: open PR, unmerged; growth: stale-closed, unmerged |
| (2) serial lane ~12 ms/window GPU-reach flag wait; batched lanes 0.5 ms | router-id D2H readback per layer from selective expert copy (#15346); blocking `tensor_get` fan-out #30252; fix = batched copy API + one-kernel gather (#30881, #31391); redundant-sync skip #29796 | #15346 merged (origin); #30881/#31391 merged (fix); #29796 open | fix exists upstream to steal |
| (3) hugepage pool worse; tmpfs shadow; cudaHostRegister'd buffers | pinned-split chain #30449 (+51-66% when experts truly pinned), mlock regression #30793 → #30926; over-pinning #30405; hugepages never merged (#21821 closed, #12552/#22022/#7420 open advisory-only) | #30449/#30926/#30405 merged; hugepages open | upstream prefers pinned-split + mlock-no-mmap over hugepages/tmpfs |
| (4) batched vs serial disproportionate loss under host pressure | pageable-vs-pinned DMA: pinning = +21% prefill / +7% decode (#25859); silent pageable fallback = the disaster mode (#30449 body); bounce-buffer elimination #31321; "weights not pinned → implicit serialization" (#21067 body) | #25859 open; #30449/#31321 merged | pinning invariant is upstream-endorsed |

---

## 8. Merged fixes worth stealing (complete list)

1. **#31391** (merged 2026-11-03) — CUDA gather kernel for batched tensor reads: temporary
   cudaHostRegister of destination + descriptor upload + one kernel; −90.7% snapshot-save interval,
   −80% longest gap on RTX 5090. https://github.com/ggml-org/llama.cpp/pull/31391
2. **#30881** (merged 2026-10-29) — `ggml_backend_tensor_get_n/set_n/copy_n` batched copy API (the API
   surface the gather hangs off). https://github.com/ggml-org/llama.cpp/pull/30881
3. **#30449** (merged 2026-10-16) — 4 GiB cap on CUDA pinned host buffers → allocator splits instead of
   silently falling back to pageable. https://github.com/ggml-org/llama.cpp/pull/30449
4. **#30926** (merged 2026-10-22) — mlock must lock every split-buffer chunk (per-chunk lock on
   `t->buffer`). https://github.com/ggml-org/llama.cpp/pull/30926
5. **#30405** (merged 2026-10-13) — do not pin foreign ranges; one host-pointer buffer per needed
   range. https://github.com/ggml-org/llama.cpp/pull/30405
6. **#30804** (merged 2026-10-23) — chunk CUDA graphs to amortize launch. https://github.com/ggml-org/llama.cpp/pull/30804
7. **#30662** (merged 2026-11-03) — prevent premature eviction of graphs about to be used. https://github.com/ggml-org/llama.cpp/pull/30662
8. **#26771** (merged 2026-10-20) — recapture graphs after memory-pool flush. https://github.com/ggml-org/llama.cpp/pull/26771
9. **#31321** (merged 2026-10-29) — reuse tensor staging buffer; skip direct-io bounce buffer. https://github.com/ggml-org/llama.cpp/pull/31321
10. **#29604/#29608/#27559** (merged, SYCL) — pinned-host sync reduction, pinned ring buffer for bulk
    uploads, pinned-alloc cap. https://github.com/ggml-org/llama.cpp/pull/29604

## 9. Unmerged PRs worth harvesting into strata (code exists, upstream hasn't taken it)
1. **#28872** — keep the backend scheduler on re-reserve (kills per-request `cudaFreeHost`+`cudaMallocHost`
   of the ~512 MB pinned input buffer; TTFT 0.8-1.1 s → 0.25-0.4 s). https://github.com/ggml-org/llama.cpp/pull/28872
2. **#25859 branch** (`thecodacus/llama.cpp` `fable5/prefetch-experts`) — expert-weight page-locking +
   tunable prefetch ring (`GGML_SCHED_PREFETCH_EXPERTS`), +64% prefill, +7% decode on the offload path. https://github.com/ggml-org/llama.cpp/issues/25859
3. **#21067** — `--prefetch-weights` tensor-override prefetch with the pinned-weights invariant. https://github.com/ggml-org/llama.cpp/pull/21067
4. **#29796** — skip redundant stream synchronizations. https://github.com/ggml-org/llama.cpp/pull/29796
5. **#30694** — merge nearby KV ranges (<64 KiB) when reading state to host. https://github.com/ggml-org/llama.cpp/pull/30694
6. **#12552 / #22022 / #7420** — hugetlbfs map, MADV_HUGEPAGE hint, direct-io (advisory THP only; we
   rejected a hugepages pool empirically — these stay reference-only). https://github.com/ggml-org/llama.cpp/pull/22022

## 10. Watchdog/driver workarounds (config, not code)
- `NVreg_RegistryDwords="RmWatchDogTimeout=60"` for #27102-class RTX 5090 stalls (verified by a
  reporter; ggerganov-endorsed thread). Check `dmesg` Xid 8/62/79 correlation during our stall windows.
- `#22360` decomposition rule: before blaming host/kernel aging, test whether the slowdown **follows the
  prompt** (KV/slot state) — different prompt on the same process is the free control experiment.

## Sources
- ggml-org/llama.cpp issues: #22360, #30793, #25859, #30252, #25082, #25740, #29281, #26918, #27137,
  #30342, #27038, #31840, #31825, #29130, #27562, #26448, #29949, #5856, #832, #2134, #27102, #20146,
  #27910, #30726, #31261, #19219, #26024, #31127, #26143, #22925, #31452, #17546, #2251, #12444, #31510
- ggml-org/llama.cpp PRs: #15346, #21067, #28872, #29796, #30881, #31391, #30986, #30694, #30449,
  #30926, #30405, #31321, #30804, #30662, #26771, #25101, #25114, #21821, #12552, #22022, #7420,
  #20834, #26934, #26135, #28334, #22877, #29604, #29608, #27559, #28725, #28549, #30327, #31053,
  #31344, #30468
- ggml-org/ggml: #400 (thread pool), #1495 (UVA-direct mmap buffer-from-ptr)
- sgl-project/sglang: #21880, discussion #23242; vLLM docs KV offloading; NVIDIA forums Xid 79 / Xid 62
  threads; NVIDIA/open-gpu-kernel-modules #1111
