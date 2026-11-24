# Dossier — The NVMe tier: streaming expert slots & PLE rows for Magma (v1, 2026-11-20)

**Scope.** (1) io_uring mechanics for 2.637 MiB random slot reads; (2) measured Gen5/Optane numbers; (3) read amplification; (4) GPUDirect Storage (GDS)/BaM vs io_uring+pinned+cudaMemcpyAsync; (5) mmap+MADV_RANDOM vs explicit io_uring for the 47.75 GiB PLE table; placement + prefetch rule. Hardware: RTX 5090, 9950X3D, PCIe 5.0 x16, Gen5 NVMe, 96 GB DDR5.

**Method.** FETCHED = document body retrieved and quoted: SSD-LLaMA (arXiv HTML), six man/kernel-doc pages, NVIDIA GDS overview, BaM abs page, KitGuru T705 pp.4+10, Computerbase T705, POMACS abstract via Semantic Scholar API. SNIPPET = full text not retrievable (ACM PDF fetch blocked 404/402; ACM DL bot-walled): verified from title + search-index excerpts of the ACM PDFs; two forum/vendor pages snippet-verified. Derived arithmetic is marked.

## 1. io_uring for 2.637 MiB slot reads

**POMACS 2026 (DOI 10.1145/3788106), verified.** "Towards Scalable Storage Architectures for GPU Clusters Running Large Language Models" (Sedaghatgoo, Salkhordeh, Brinkmann, Asadi; Proc. ACM Meas. Anal. Comput. Syst. 2026): phase-aware evaluation of libaio vs io_uring vs SPDK vs GDS over SATA/NVMe/Optane-NVMe/Optane-PMem, >25,000 configurations. Abstract (fetched): "for inference, io_uring achieves the lowest latency and competitive IOPS for small random I/O on NVMe." PDF text (snippets): "For inference workloads, interrupt-driven io_uring achieves the best balance between tail latency, throughput, and CPU efficiency, outperforming all other datapaths"; libaio→interrupt io_uring gives "+6% to +11%" bandwidth/IOPS on NVMe/Optane (5–20% on streaming reads with better GB/s-per-core); §3.2.5 "SQPOLL: High CPU Cost, Minimal Benefit"; polling (SPDK, io_uring CQ-poll) advantageous "only when small, random I/O dominates and…" (snippet truncates); deployable rule: "GDS for pre-training and fine-tuning, and interrupt-driven io_uring for inference." The extended text (DOI 10.1145/3801489.3806840) characterizes decode I/O as "latency-sensitive small random reads … often in the 4–16 KiB range". **Premise confirmed: interrupt mode wins; SQPOLL/IOPOLL are CPU-costly exceptions.**

**Supported configuration** (man pages fetched):
- O_DIRECT required for polling [S4]; bypasses page cache "when applications do their own caching" [S5] — we do (SLRU), so O_DIRECT semantics are correct.
- Registered buffers: IORING_REGISTER_BUFFERS pins pages once (charged to RLIMIT_MEMLOCK; ≤1 GiB/buffer; anonymous memory; huge pages pinned whole); use IORING_OP_READ_FIXED [S3]. Ring of 32 × 2.637 MiB = 84.4 MiB pinned; raise RLIMIT_MEMLOCK.
- Fixed files: IORING_REGISTER_FILES + IOSQE_FIXED_FILE avoids per-op fd refcounting [S3].
- Queue depth: match the consumer — exactly 10 slots/layer (top-10/512). Run QD 10–16, per-slot completions firing async H2D immediately (SSD-LLaMA pattern [S2]). KitGuru's T705 4K curve plateaus at QD16 (176,819 IOPS @QD16 → 177,274 @QD32) [S10]: beyond 16 buys nothing.
- Completion: interrupt default. IOPOLL busy-waits, requires O_DIRECT + nvme poll_queues, "lower latency, but may consume more CPU" [S4]; POMACS escalates to polling only under SLA pressure [S1]. Batch style: persistent worker, bounded batches, timeout-flush [S2].
- Alignment: buffer, file offset, and length must each respect the logical block size; query with statx(STATX_DIOALIGN), kernel ≥6.1 [S5]. Pack slots 4 KiB-aligned.

**O_DIRECT at 79 GB shards + 96 GB RAM (derived).** Page-caching everything = 79 GB shards + 47.75 GiB PLE ≈ 127 GB demand > 96 GB → permanent MGLU/kswapd reclaim on the decode path (kernel doc: reclaim "directly impacts the kswapd CPU usage and RAM efficiency" [S7]; POMACS prices CPU efficiency [S1]). O_DIRECT slots + page cache reserved for PLE + bounded SLRU victims → steady resident ≈ 47.75 + 24–32 + 0.08 GiB ≈ 72–80 GB. **O_DIRECT justified; mmap-ing shard files disqualified.**

## 2. Measured drive numbers

| Drive | QD1 4K rand read | QD-high 4K | Large-block |
|---|---|---|---|
| Crucial T705 2TB (Gen5) | 26,861 IOPS = 110.0 MB/s (KitGuru measured [S9]); 76–107 MB/s on other platforms [S10,S11] → derived 37–54 µs/op | 176,819 IOPS = 724 MB/s @QD16 1T; CDM peak 1,575,736 IOPS (both measured [S9]) | 14.5 GB/s seq spec (Computerbase fetched [S8]); SEQ1M QD1 ≈ 4–5 GB/s class (platform-dependent [S12]) |
| Samsung 9100 Pro (Gen5) | 109.5–110.4 MB/s (Hardwareluxx measured [S13]) | — | spec 14,700 MB/s (KitGuru [S14]) |
| Samsung 990 Pro (Gen4) | spec 22K IOPS [S15]; SEQ1M Q1T1 4,431.6 MB/s "<236.51 µs>" (forum CDM readout [S16]) | 961K IOPS @ 131.5 µs peak (StorageReview [S15]) | spec 7,450 MB/s [S15] |
| Optane P5800X | 74,875 IOPS (KitGuru [S17]) → derived ~13.4 µs; Intel QoS spec 4KRR QD1 99% <6 µs, mixed <25 µs [S18] | ~1.5M IOPS class | — |

Derived for Magma: one 2.637 MiB slot at QD1 ≈ 0.6–0.8 ms (236.5 µs/1 MiB scaled + latency [S16]); 10 concurrent slots = 26.4 MiB/layer at QD 10–16 sustain ~50–70% of peak → **2.7–3.8 ms/layer (Gen5), 4–6 ms (Gen4-class)**. Optane's 3× QD1 latency win evaporates at 2.6 MiB granularity — transfer-dominated, lower bandwidth class.

## 3. Read amplification

Derived: slots 4 KiB-aligned in the shard file, read length ceil(payload/4096)·4096 → RA ≤ 1 + 4095/2,765,145 ≈ **1.0015** (≤4 KiB padding/slot; ≤ ~23 MiB over the 15.2 GiB set). Unaligned DIO is not "slow", it is rejected [S5]; SSD-LLaMA's expert pack exists precisely to make "one manifest lookup … one aligned O_DIRECT request" per expert [S2]. **Alignment to 4 KiB erases RA: confirmed.**

## 4. GDS/BaM vs io_uring + pinned + cudaMemcpyAsync

GDS docs (fetched): cuFile silently falls back to a CPU-staged compatibility path when offsets/buffers aren't cufile-block-size aligned, when I/O is "too small" for RDMA, or nvidia-fs.ko is missing; without GDS the path is "an extra copy through a bounce buffer in the CPU" [S6]. NVIDIA claims GDS latency gains are "most apparent with small transfers" [S6] — but POMACS's phase-aware measurement contradicts that generality: interrupt io_uring wins inference; GDS is the recommendation for large sequential pretrain/fine-tune transfers [S1]. BaM (arXiv 2203.04910, fetched): GPU-initiated NVMe queues + GPU-resident software cache that coalesces requests "while minimizing I/O traffic amplification" — full title: "GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture" — it pays in VRAM (cache + queue buffers), driver init, raw offsets; heavyweight for ~10 reads/layer at batch 1.

**SSD-LLaMA (arXiv 2609.18110, fetched): io_uring confirmed; reasons, from its own pipeline.** Only cache-missing experts become "O_DIRECT requests"; a persistent io_uring worker submits "multiple independent reads in bounded batches" (timeout-flush) into "a fixed pool of reusable pinned host buffers"; each completion immediately fires async H2D into a reserved VRAM slot, published after a CUDA event; bounded pools give backpressure [S2]. The paper never mentions GDS/BaM; visible rationale: compressed blocks must reach host RAM anyway (CPU-executed experts), one path serves CPU+GPU execution, and rANS compression (5.93× SSD-traffic cut; 8.89 GB/token on a 1T MoE) shrinks the PCIe hop GDS would bypass [S2]. Measured: 77.7% of peak SSD sequential bandwidth (9 GiB/s testbed) vs 43.2% best baseline (llama.cpp mmap / KTransformers DRAM+CPU); prefill 1.90–3.61× vs llama.cpp; decode 2.10–15.58×; 1+ tok/s on a 1T-param MoE with RTX 5090 + ≤32 GB RAM [S2]. **Verdict: io_uring+pinned+H2D is the practical datapath at 2.6 MiB / QD≈10; GDS/BaM buy nothing measurable here and add setup risk (consumer board, ext4/xfs, kernel modules).**

## 5. PLE table: mmap+MADV_RANDOM vs explicit io_uring

Keep **mmap + MADV_RANDOM**. man madvise.2 (fetched): MADV_RANDOM — "read ahead may be less useful" (kills readahead amplification for 160 B rows); MADV_WILLNEED reads ahead [S6man]. The page cache then *is* the LRU: kernel MGLU ranks pages into generations by access age and proactively reclaims cold ones [S7k]. Capacity fits (§1), so faults settle to warm hits; worst cold traffic ≈16 rows/token ≈ 64 KiB/token — trivial. Explicit io_uring would re-implement an LRU at 160 B granularity to beat the kernel's own; SSD-LLaMA's anti-mmap critique ("page faults, fragmented accesses, or serialized reads") targets MiB-scale expert blocks with end-to-end-exposed delivery latency, not embedding rows [S2]. WILLNEED batching one ubatch ahead (prior dossier) stands.

## 6. Placement rule and prefetch timing

**Rule — tier by payload, one mechanism per tier:**
1. **VRAM:** dense arena (9.9 GiB), KV (6 GiB), hot slots ≤5,138 of 5,895 (13.5 GiB) — SLRU by activation frequency.
2. **RAM explicit:** victim SLRU 24–32 GiB; 32-slot registered io_uring ring (84.4 MiB); H2D staging.
3. **RAM page cache:** PLE table only (mmap+MADV_RANDOM; WILLNEED one ubatch ahead).
4. **NVMe O_DIRECT io_uring** (IRQ, fixed files, registered buffers, QD 10–16): full slot set + shards, 4 KiB-aligned packs, never page-cached.
5. **CPU execution:** none (prior dossier stands); the io_uring path would serve it for free if ever needed [S2].

**When to prefetch:** expert slots, **one layer ahead, never one token ahead** — at batch 1 next-token routing doesn't exist. Submit layer L+1's 10 reads when L's hidden state is available (ProMoE proactive prediction from intermediate states [S19]; Pre-gated MoE = trained variant; vLLM temporal hits 97–100% per prior dossier). Derived steady state: per-layer IO 2.7–3.8 ms vs compute ~1.5–2.5 ms → IO-bound; per-slot early H2D (26.4 MiB ≈ 0.48 ms at your ~55 GB/s, separate stream, double-buffered) hides what compute can't [S2]. Hedge window = two layers only if measured predictor hit-rate <95%; reads are cancel-on-visit.

**Ceilings (derived):** cold cache (all 480 slots from NVMe): 1.236 GiB/token ÷ (50–70% × 9–14.5 GB/s) ≈ **4–7.5 tok/s** (SSD-LLaMA's measured 77.7% utilization ⇒ ~5.5 on a 9 GiB/s drive). Warm steady state (SLRU misses ~15–25% of lookups given Zipf skew): 190–330 MiB/token ≈ **15–50 tok/s IO floor**. NVMe is the capacity tier; latency comes from VRAM/RAM residency.

## Transfer to Magma

1. Shard format: expert packs — 4 KiB-aligned slot records + manifest (slot id → offset/len), one O_DIRECT read per slot [S2].
2. One io_uring ring: registered files (one fd/shard) + 32×2.637 MiB registered buffer ring (raise RLIMIT_MEMLOCK), READ_FIXED, IRQ completion, no SQPOLL/IOPOLL [S1,S3,S4man]. Bounded batch=16, timeout-flush ~200 µs [S2].
3. Query STATX_DIOALIGN at mount; assert 4 KiB [S5].
4. PLE: mmap+MADV_RANDOM at load; MADV_WILLNEED one ubatch ahead [S6man].
5. Prefetch L+1 at L's hidden state; per-slot completion → cudaMemcpyAsync into VRAM slot; publish on CUDA event [S2].
6. Never mmap shards (thrash math §1); adopt GDS/BaM only behind a measured regression gate [S1,S6gds].
7. Instrument NVMe bytes/token + SLRU miss rate; the cold→warm transition is the KPI.

## Source table

| # | Source | Establishes | Verification |
|---|---|---|---|
| S1 | Sedaghatgoo, Salkhordeh, Brinkmann, Asadi, "Towards Scalable Storage Architectures for GPU Clusters Running LLMs", POMACS 2026, DOI 10.1145/3788106; ext. text DOI 10.1145/3801489.3806840 | interrupt io_uring best for inference; +6–11% vs libaio; SQPOLL costly; GDS→pretrain/fine-tune; >25k configs | Abstract fetched (S2 API); quotes from ACM-PDF search snippets; PDF fetch blocked |
| S2 | SSD-LLaMA, arXiv 2609.18110 | io_uring pipeline, 77.7% BW util, compression, mmap critique, 5090/16 GB testbed | HTML fetched |
| S3 | man7 io_uring_register.2 | buffer/file registration mechanics+limits | Fetched |
| S4man | man7 io_uring_setup.2 | IOPOLL/SQPOLL, nvme poll_queues | Fetched |
| S5 | man7 open.2 | O_DIRECT semantics, alignment, STATX_DIOALIGN | Fetched |
| S6man | man7 madvise.2 | MADV_RANDOM/WILLNEED | Fetched |
| S7k | docs.kernel.org MGLU | page-cache LRU generations, proactive reclaim | Fetched |
| S6gds | NVIDIA GDS overview-guide | fallback/bounce conditions, latency claim | Fetched |
| S8 | Computerbase T705 test | T705 specs 14.5 GB/s, 1.55M IOPS QD512 | Fetched |
| S9 | KitGuru T705 2TB review pp.4/10 | measured 4K QD1=26,861 IOPS; QD16 plateau; 1.575M peak | Fetched |
| S10 | Tweaktown T705 review | 107+ MB/s Q1T1 4K | Snippet |
| S11 | mydrivers news (T705) | CDM 4K QD1 76 MB/s platform variance | Snippet |
| S12 | ascii.jp SSD review | QD1 sequential platform-dependence | Snippet |
| S13 | Hardwareluxx 9100 Pro test | 4K Q1T1 109.5–110.4 MB/s | Snippet |
| S14 | KitGuru 9100 Pro review | fastest consumer 4K QD1; 14,700 MB/s | Snippet |
| S15 | Computerbase + StorageReview 990 Pro | spec 7,450 MB/s, 22K IOPS QD1; 961K @131.5 µs | Snippet |
| S16 | level1techs forum (Optane thread) | 990 Pro SEQ1M Q1T1 4,431.6 MB/s <236.51 µs> | Snippet (user measurement) |
| S17 | KitGuru P5800X review | QD1 4K 74,875 IOPS | Snippet |
| S18 | theSSDreview P5800X | Intel QoS 4KRR QD1 <6 µs | Snippet |
| S19 | ProMoE, arXiv 2410.22134 | proactive expert prefetch from intermediate states | Snippet (title verified) |
| S20 | Prior dossiers (gdn-tma-ple; moe-residency) | continuity: PLE math, 97–100% temporal hits, BaM lineage | Local files |
