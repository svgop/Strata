# Dossier: PCIe 5.0 H2D physics for expert-slot fills — Magma on RTX 5090 (single GPU, AM5)

Scope: host→device movement of NVFP4 expert slots (2.637 MiB each; ~480 slots ≈ 1.24 GiB per decode token) over PCIe 5.0 x16. Extends, not repeats, the residency dossier (cache policy) and the GDN/TMA/PLE dossier. Verification legend: **fetched** = I pulled the page this session and quote its text; **snippet** = title/snippet verified only.

## 1. Link physics and the small-transfer knee

Raw PCIe 5.0 x16 = 63 GB/s per direction (32 GT/s × 16 lanes, 128b/130b). Measured sustained H2D: RTX 5090 pinned 54 GB/s H2D / 53 D2H (gamegpu news, snippet); Gen5 x16 P2P writes on 5090/RTX-6000-class rigs 53–56 GB/s same-NUMA, 39–51 cross-NUMA (rtx6kpro repo, fetched — cross-NUMA loss is real; single-socket AM5 is immune); RTX 4090 Gen4 x16 back-to-back cudaMemcpyAsync 24–26 GB/s of 31.5 raw (gigagpu, fetched, same ~80% efficiency class). So ~52–55 GB/s is the realistic Gen5 ceiling; your measured 50–64 brackets it.

Per-operation CPU cost is the "knee": a PMBS22 HPC fit shows a ~2 µs per-transfer y-intercept (snippet); cudaMemcpyAsync per-call overhead is ~5–7 µs for small chunks (emergentmind breakdown, snippet; the classic SPIE 7705 Wong et al. study established the same µs-scale per-call floor, title-verified). Pinned-vs-pageable crossover historically sits below ~300 KB (StackOverflow bandwidthTest curve, fetched; dated Tesla C2050 but the shape holds): below a few hundred KB, call overhead dominates; by 1–2 MB you are link-limited. Arithmetic for your payload: 2.637 MiB at 50 GB/s = 55 µs link time, so one un-pipelined call adds ~5 µs (+9%); batch 8 slots (21.1 MiB ≈ 440 µs) and overhead falls below 1.2%. Batching is a single-digit-percent win; the ring (§3) is what hides the rest.

Sync path: cudaDeviceScheduleSpin makes CUDA "actively spin when waiting for results... can decrease latency," at CPU-thread cost; cudaDeviceScheduleBlockingSync blocks the host thread (Runtime API docs, fetched). Spin + cuStreamWaitValue mailboxes is right for the decode loop; blocking sync only on background NVMe threads.

## 2. Copy engines vs SM copies; concurrent CE throughput

Semantics (CUDA Runtime API doc text, archived copies, snippet-verified): asyncEngineCount = 1 → device can copy H2D **or** D2H concurrently with kernels; = 2 → both directions concurrently. NVIDIA's own training deck: "2 Copy Engine queues – one for H2D and one for D2H" (snippet). A PSI HPC note adds H2D and D2H are "one FIFO engine each" (snippet).

GeForce engine count: community deviceQuery outputs report 1 on consumer parts, but I found no authoritative fetched confirmation — one repo even claims the 5090 has 2 (unverified, likely confusing queues with engines). Action: read asyncEngineCount in Magma's device probe. The design below is insensitive to 1 vs 2 because every H2D lane shares one 63 GB/s link.

Can two H2D lanes reach 2×? No, on three levels: (a) CE copies on one engine serialize (FIFO); (b) SM-assisted copies (nvbandwright implements both initiators: CE via cuMemcpyAsync, SM via kernels that "can sometimes achieve higher bandwidth" and leave SMs free — README + DeepWiki, fetched) can overlap the CE, but they share the same PCIe link; (c) the link saturates at ~52–55 GB/s unidirectionally. SM copies buy latency-hiding, not bandwidth — skip them at batch 1; CE keeps all 170 SMs on decode.

## 3. Staging topology for 2.637 MiB slots (recommended)

- **Pool**: one ~96 MiB host arena = 4 rings × 8 slots × 2.637 MiB. mmap 2 MiB-aligned + madvise(MADV_HUGEPAGE), then cudaHostRegister. Precedent: Strata PR #650 backs a ~72 GiB expert arena this way because naive pinning locks ~18.8 M 4 KiB pages (snippet); nvbandwright ships --useHugePages for host allocations (fetched) — the reference practice for DMA hosts.
- **Depth 4**: two rings absorbing NVMe→RAM fill, one in DMA flight, one decode-margin. Cold-token reality: 1.24 GiB from one Gen5 NVMe (~12 GB/s) is ~110 ms vs ~26 ms on PCIe5 — the ring hides NVMe jitter, not PCIe.
- **Batching**: one cudaMemcpyAsync per 8-slot block (21.1 MiB, ~440 µs at 50 GB/s): 60 calls/token, ~5 µs CPU each ≈ 0.3 ms/token (~1%). One descriptor per slot costs 480 × 5 µs ≈ 2.4 ms/token of pure CPU plus small-TLP gaps. Batch unless a layer's experts are ragged.
- **Layout**: layer-major contiguous slots so a token's gather collapses to ≤48 descriptors; pad block boundaries to 2 MiB (<0.5% slack). DMA itself needs no alignment — GDS docs: transfers "do not need to be aligned to anything other than a byte boundary" (fetched); 2 MiB is for THP and CPU TLB, not the CE. For cold NVMe-only fills, GPUDirect Storage skips the CPU hop entirely but requires pinned cudaMalloc GPU buffers and 4 KB storage alignment (GDS design guide + PyTorch tutorial, both fetched).
- **CUDA graphs**: launch cost amortizes to ~2.5 µs + ~1 ns/node for ≥10-node graphs on Ampere-class parts (NVIDIA devblog, fetched); the original graphs blog measured 9.6 µs/launch-with-sync → 3.4 µs as a graph on V100 (fetched). CUDA 12.0 added scheduling heuristics that "avoid unintended serialization of memory copy operations" (fetched), so H2D memcpy nodes can sit in the layer graph beside compute. Memcpy nodes execute on the same engine path as stream copies (engineering inference, flagged). Update per-token addresses via cudaGraphExecMemcpyNodeSetParams; source pointers must stay pinned for the graph's lifetime — registered pools satisfy this.

## 4. WC / pinned / BAR placement for repeated fills

- **WC**: cudaHostAllocWriteCombined — "WC memory can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be read efficiently by most CPUs... a good option for buffers that will be written by the CPU and read by the device" (CUDA doc text via manpage mirror, fetched). Use WC only for bulk-sequential repack buffers the CPU never reads; keep the slot ring default-cacheable pinned, since NVMe/decompress writes it in scattered chunks and WC only rewards full-line sequential stores. A/B it; pinned-vs-pageable itself is the big win (×1.32 H2D in the PSI note, snippet).
- **BAR/ReBAR**: BARs are address windows for CPU/peer-initiated access (GPUDirect RDMA docs, fetched; BAR1 default-class 256 MB, large-BAR Tesla 16 GB). cudaMemcpyAsync DMA never traverses BAR1, so ReBAR (256 MiB → full-framebuffer aperture; netdata guide, snippet) is irrelevant to expert fills. It matters only if you add SM/CPU zero-copy reads from VRAM (e.g., logits peeks), where BAR1 exhaustion breaks mappings. Keep Above-4G Decoding + ReBAR enabled anyway (standard BIOS pairing; rtx6kpro BIOS list, fetched).

## 5. AM5 platform factors

- **Lanes**: GPU x16 Gen5 and M.2_1 are CPU-direct; the chipset's ~92 Gen4 lanes share one Gen4 x8 uplink (evezone, fetched). Put the NVMe shard drive on the CPU-attached M.2, never chipset.
- **IOMMU/AMD-Vi**: iommu=pt sets DMA passthrough by default (= iommu.passthrough=1) (kernel docs, fetched). One repo measurement: amd_iommu=off gave 2–6% and iommu=pt did NOT recover it (osmantic/ods, snippet); P2P rigs standardize on amd_iommu=on iommu=pt (cmp170hx wiki, snippet). Magma: run iommu=pt, A/B against off — 480 DMA fills/token amplifies even 1% translation cost.
- **Dual-CCD 9950X3D**: single IOD → symmetric PCIe. Pin the NVMe-fill thread off the V-cache CCD (save it for router/n-gram PLE work); do this for L3 contention, not DMA.
- **Above-4G/ReBAR**: enable, verify with nvidia-smi -q, move on — not on the DMA path.

## Transfer to Magma

1. Host staging: ~96 MiB pinned ring = 4 × 8 × 2.637 MiB, 2 MiB-aligned mmap + MADV_HUGEPAGE + cudaHostRegister; A/B a WC variant for bulk repack only.
2. Fill unit: 8-slot blocks (~440 µs each) → 60 cudaMemcpyAsync/token, or 48 memcpy nodes in a per-layer CUDA graph with cudaGraphExecMemcpyNodeSetParams; spin schedule + cuStreamWaitValue32 mailbox per ring; CPU overhead ≤1% vs ~9% un-batched.
3. Do not build: SM-assisted copy path, ReBAR-dependent zero-copy, or a "second CE lane" — the ~52–55 GB/s link cap makes all three dead ends; ~26 ms/token all-cold is the floor (1.24 GiB / 52 GB/s ≈ 24 ms; 1.24 GiB / 50 GB/s ≈ 26.6 ms).
4. Probe once: asyncEngineCount, and PCIe link state (nvidia-smi -q, lspci) to catch a board silently training Gen4 or x8.
5. Platform: NVMe on CPU M.2; iommu=pt (A/B off); fill thread off the V-cache CCD.
6. Planning: every 10% SLRU hit (residency dossier) returns ~2.4–2.6 ms of the cold-token floor.

## Source table

| # | Source | Locator | Status | Key number/claim |
|---|---|---|---|---|
| 1 | CUDA Runtime API docs, device flags | docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html | fetched | spin ↓ latency, blocking = host block |
| 2 | PMBS22 GPU paper (LBL) | crd.lbl.gov/assets/PMBS22_GPU_final.pdf | snippet | ~2 µs per-transfer intercept |
| 3 | cudaMemcpyAsync overhead breakdown | emergentmind.com/open-problems/breakdown-of-cudamemcpyasync-overhead | snippet | 5–7 µs/call, small chunks |
| 4 | Wong et al., SPIE 7705 | spiedigitallibrary 10.1117/12.852632 | title | µs-scale per-call floor |
| 5 | StackOverflow pinned curve | stackoverflow.com/questions/6933385 | fetched | crossover <300 KB |
| 6 | gigagpu 4090 PCIe page | gigagpu.com/rtx-4090-24gb-pcie-gen4-x16/ | fetched | Gen4 H2D 24–26 GB/s; Gen5 eff. ~52 |
| 7 | gamegpu 5090 news | en.gamegpu.com/news/zhelezo?start=98 | snippet | 54 / 53 GB/s H2D/D2H |
| 8 | rtx6kpro PCIe/BAR doc | github.com/local-inference-lab/rtx6kpro, hardware/pcie-bandwidth.md | fetched | Gen5 P2P 53–56 GB/s same-NUMA; BIOS ReBAR+Above-4G |
| 9 | asyncEngineCount semantics (archived Runtime API text) | docs.nvidia.com/cuda/archive/11.2.1 Runtime API PDF (via mirrors) | snippet | 1 = copy∥kernel; 2 = both directions |
| 10 | NVIDIA Streams & Concurrency webinar | developer.download.nvidia.com .../StreamsAndConcurrencyWebinar.pdf | snippet | CE queues: one H2D + one D2H |
| 11 | PSI CUDA performance note | gitea.psi.ch .../cf_cuda_performance.pdf | snippet | per-direction FIFO CE; pinned ×1.32 |
| 12 | nvbandwidth README + DeepWiki copy-initiators | github.com/NVIDIA/nvbandwidth; deepwiki.com/NVIDIA/nvbandwidth/3.2-copy-initiators | fetched | CE vs SM cells; --useHugePages |
| 13 | CUDA graphs devblogs (both) | developer.nvidia.com/blog/constant-time-launch...; /blog/cuda-graphs/ | fetched | 2.5 µs+~1 ns/node; 9.6→3.4 µs |
| 14 | GDS design guide + PyTorch GDS tutorial | docs.nvidia.com/gpudirect-storage/design-guide/application-suitability.html; docs.pytorch.org/tutorials/unstable/gpu_direct_storage.html | fetched | pinned cudaMalloc buffers; byte-boundary API / 4 KB storage |
| 15 | WC flag doc (manpage mirror of CUDA docs) | manpages.ubuntu.com/manpages/bionic/en/man3/cudaHostAlloc.3.html | fetched | WC: faster PCIe writes, poor CPU reads |
| 16 | GPUDirect RDMA docs | docs.nvidia.com/cuda/gpudirect-rdma/ | fetched | BARs = CPU/peer windows; BAR1 256 MB |
| 17 | netdata BAR1 guide | netdata.cloud/guides/nvidia-gpu/bar1-memory-exhaustion | snippet | ReBAR grows 256 MiB → framebuffer |
| 18 | Linux kernel-parameters docs | docs.kernel.org/admin-guide/kernel-parameters.html | fetched | iommu=pt = iommu.passthrough=1 |
| 19 | osmantic/ods system tuning | github.com/osmantic/ods .../system-tuning/README.md | snippet | amd_iommu=off 2–6%; pt did not recover |
| 20 | Strata PR #650 (THP expert arena) | github.com/Niko1221/Strata/pull/650 | snippet | 72 GiB THP+cudaHostRegister; 18.8M pages |
| 21 | evezone X670E lane guide | evezone.evetech.co.za/quick-bytes/x670e-pcie-lane-sharing... | fetched | chipset uplink Gen4 x8; GPU/M.2_1 CPU-direct |
| 22 | REJECTED: "5090 GDS results" | github.com/anthony-chaudhary/fak ...QWEN38-NVIDIA-5090-GPUDIRECT-RESULTS | excluded | self-labeled SIMULATED |
| 23 | "5090 has 2 async copy engines" claim | github.com/ForceInjection/AI-fundamentals .../07_cuda_streams_concurrency.md | snippet, unverified | conflicts with deviceQuery reports; probe on target |
