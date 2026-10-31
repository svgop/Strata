---
type: digest
id: w15-depth-scout
created: 2026-10-31
status: evidence
topic: W15 depth persistence — competitor KV residency, sparse selection, prefix matching
scope: read-only research; sources fetched 2026-10-31; no code touched
---

# W15 — Depth persistence: competitive digest

Baseline: decode 209-222 t/s @0-60K → 173-180 @134K → 114-166 @228-262K; 6-lane 341→179-210. Mechanisms: (a) static 32K-cell resident tier + page-claim kernel past it; (b) QSA score scan O(ctx/4)+single-block top-k; (c) prefix-reuse scan 73→313 ms (9K→99K). Fields per competitor: mechanism, selection-scan scaling, tier/prefetch, prefix match, numbers, grade.

## vLLM — paged KV, chained-hash prefix match, KV offload
- Mechanism: fixed-size blocks (16 tok); prefix match = hash chain — block hash = sha256(parent hash + block tokens + extra keys); match is a dict hit per block, no tree walk. LRU free queue, O(1) mid-queue moves; freed blocks re-enqueued tail in REVERSE order (later blocks less likely reused); only full blocks cached; sha256 default since v0.11 after collision-leak concern.
- Selection scan: n/a (full attention; density handled only via offload).
- Tier/prefetch: v1 `kv_offload` = OffloadingManager (lookup returns HIT/MISS/HIT_PENDING/RETRY; `touch` = LRU mark; `prepare_load` pins against eviction) + OffloadingWorker async submit_store/submit_load; CPU + storage media; speculative tokens excluded from accounting. Tier size static; no growth policy and no prefetch in the base API. Roadmap RFC #33689: pluggable out-of-tree backends, CPU↔GPU offload, tiering (fs/obj) still maturing.
- Prefix match: O(1) dict lookup per block ⇒ matching a 99K-token prompt ≈ 6.2K chained hash evaluations, independent of the number of cached entries.
- Grade: verified-from-source (design doc + API doc fetched 2026-10-31).
- https://docs.vllm.ai/en/stable/design/prefix_caching.html ; https://docs.vllm.ai/en/latest/api/vllm/v1/kv_offload/base.html ; https://github.com/vllm-project/vllm/issues/33689

## SGLang — RadixAttention + HiRadixCache
- Mechanism: radix tree over tokens; node = KV span; match longest shared prefix; LRU evicts leaves first, shared parents persist; node split + refcount for concurrent reuse. HiCache extends each node with tier masks (L1 GPU / L2 host DRAM / L3 storage) + exact local addresses; L3 existence queried live (metadata not synced).
- Selection scan: n/a.
- Tier/prefetch: write policies `write_through` / `write_through_selective` (writes next tier only past an access-count threshold — hot filter) / `write_back` (on upper-tier eviction). Prefetch fires when L3 hit >256 tokens; termination `best_effort` / `wait_complete` / `timeout` (2 s base + 0.1 s per 1K tokens, 30 s cap); layer N+1 host→GPU load overlaps layer N compute; GPU-assisted copy kernels up to 3x faster than cudaMemcpyAsync. Host pool = `--hicache-ratio` (multiple of device pool) or `--hicache-size` GB per rank — static, boot-sized.
- Prefix match: search/insert/evict O(prefix length); page-granular with node splits on partial match; C++ radix/HiRadix rewrite is a 2025 Q3 roadmap item to cut scheduler overhead (issue #7736, PR #7194).
- Numbers: design doc publishes no hit-rate/latency benchmarks (links blog); none reproduced here.
- Grade: verified-from-source (design doc); roadmap search-level.
- https://docs.sglang.io/docs/advanced_features/hicache_design.md ; https://arxiv.org/abs/2310.07204 ; https://github.com/sgl-project/sglang/issues/7736

## Quest (arXiv:2406.10774) — query-aware page selection
- Mechanism: 16-token pages; per-page metadata = elementwise min & max of Keys per channel (2·d floats per page), updated incrementally on append. Score = Σ_i max(q_i·m_i, q_i·M_i) upper-bounds any attention score in the page; top-K pages (2048-token budget) get exact attention. Direct analog of strata's pooled indexer + score scan.
- Scan scaling: score pass reads only metadata = 1/page-size of full-attention traffic; with fixed budget, approximate attention is ~constant in ctx; top-k via RAFT batched kernel costs 5-10 µs below 128K. Load fraction 1/S + K·S/L. First 2 layers kept full (low sparsity).
- Tier/prefetch: none — pages stay in HBM; selection replaces tiering.
- Numbers: 7.03x self-attention vs FlashInfer @32K (2048 budget, Llama2-7B); end-to-end decode 1.74x FP16 / 2.23x 4-bit @32K; up to 3.82x/4.54x vs eviction baselines.
- Grade: measured-in-paper.
- https://arxiv.org/abs/2406.10774

## DeepSeek NSA (arXiv:2502.11089) — hierarchical selection, tensor-core kernel
- NOTE: task brief cited "arXiv:2402.02710 area"; actual NSA id is 2502.11089 (verified on arXiv 2026-10-31). ACL 2025 best paper.
- Mechanism: three branches — compressed (block 32, stride 16, learned MLP φ), selected (block 64, n=16 blocks incl. 1 initial + 2 local), sliding window 512; learned per-branch gates.
- Scan scaling: selection importance is INDUCED FREE from the compression branch's softmax scores (Eq 8-9: sum-pool compressed scores over each selection block's span; cross-head SUM within the GQA group for shared selection) — no separate min-max scan, no standalone top-k pass; top-n runs over ≤(s-l)/d compressed scores. Decode touches ⌊(s-l)/d⌋ compressed + n·l' selected + w window tokens — bounded as s grows.
- Kernel: group-centric loading (all GQA-group queries for a position in SRAM), shared KV fetch of contiguous selected blocks, grid outer loop — tensor-core aligned.
- Numbers: decode up to 11.6x vs Triton FlashAttention-2 @64K (memory-access-bound, ~linear in tokens touched); 9.0x forward / 6.0x backward.
- Caveats for strata: §2.1 — per-head independent selection (Quest-style) under GQA pays the UNION of selections in KV traffic; NSA sums scores group-wide to force shared blocks. §6.1 — heuristic min-max selection showed low recall in native training; compression-derived scoring won.
- Grade: measured-in-paper (full HTML read).
- https://arxiv.org/abs/2502.11089

## LMCache + AttentionStore + Mooncake — KV tiering & prefetch
- LMCache (arXiv:2510.09665): tiers GPU→CPU DRAM→local disk→remote→object store; 256-token hash-keyed chunks; batched movement, zero-copy refcount, delayed decode storing; prefetch during the idle window between admission and execution; compute/IO pipelining cuts e2e delay 1.46x. CPU DRAM load 400 Gbps (~50 GB/s) vs 88 Gbps for native vLLM CPU offload (chunk vs page granularity). TTFT 1.9-8.1x smaller @10-20K inputs, QPS=1; up to 15x throughput with vLLM. Crossover: at 32 Gbps remote link, offload beats recompute only >256K inputs; 64-128 Gbps wins at all lengths. Production prefix hit ~50%; context truncation halves hit ratio (85→45%).
- AttentionStore (arXiv:2403.19708): GPU/DRAM/SSD hierarchy; scheduler-aware policy uses job-queue lookahead to PROMOTE caches before demand and evict never-accessed ones; layer-wise preload + async save; positional-encoding decoupling keeps caches valid past window overflow. TTFT -87%, prefill throughput +7.8x, cost -70%.
- Mooncake (arXiv:2407.00079): KVCache-centric disaggregation; router balances by prefix affinity over a CPU-DRAM/SSD pool; +525% throughput (simulated), +75% requests handled (Kimi traces).
- Grade: measured-in-paper (LMCache tables via full text; other two abstract-verified).
- https://arxiv.org/abs/2510.09665 ; https://arxiv.org/abs/2403.19708 ; https://arxiv.org/abs/2407.00079

## MInference (arXiv:2407.02490) — sparse prefill (optional pillar)
- Per-head offline pattern assignment (A-shape, vertical-slash, block-sparse), dynamic sparse-index build at inference, optimized kernels; prefill-only (decode untouched). Up to 10x prefill latency cut on A100; 1M-token prefill baseline = 30 min (8B, 1x A100).
- Grade: measured-in-paper (abstract).
- https://arxiv.org/abs/2407.02490

## What this changes on the row
- Prefix scan (c): replace linear token compare with vLLM-style chained block hash — hash = sha256(parent hash, cell tokens); O(1) per block, independent of entry count. Direct kill for the 73→313 ms growth; smallest diff, highest certainty.
- QSA scan (b): validated as designed — Quest's tensor-core min-max metadata scan holds top-k to 5-10 µs at 128K, and NSA induces selection scores from a coarser branch for free; build the two-level pooled-of-pooled prune on tensor cores, but prefer NSA-style sum-pooled scores (no recall loss) over a second full scan.
- GQA/union trap: sum scores across heads sharing KV (NSA Eq 10) before top-k, or per-step KV traffic = union of per-head picks — this can silently erase the win on 6-lane concurrency.
- Resident tier (a): no competitor grows the hot tier with active depth — all use static boot-sized L2 (hicache-ratio/size, LMCache tiers, vLLM offload) + LRU + prefetch. Elastic resident tier remains novel; keep the cap + expert-cache funding as planned.
- Miss latency is hidden by overlap, not reduced: HiCache prefetch at L3-hit>256 tokens with timeout termination; LMCache prefetches in the admission queue; vLLM submit_load is async. Adopt prefetch-on-selected-window with best_effort/timeout termination.
- Target invariant: NSA's speedup GROWS with depth (11.6x @64K) because per-step touched tokens are constant (n·l'+w+compressed). Set the row's success metric to per-step touched bytes O(K), so 6-lane throughput tracks the selection budget, not ctx length.
