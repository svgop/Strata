---
type: digest
id: w6-qsa-retune-scout
created: 2026-10-31
status: evidence
---

# W6 — split-KV / quant-KV decode attention: competitor scout (QSA retune)

Scope: how leading decode kernels size KV splits, guard empty splits, fuse the softmax merge, and run quantized-KV MMA. Sources read as code (main branch or pinned tags) unless marked paper-only.

## 1. FlashInfer (split-KV decode planner)

- Mechanism: plan stage (host) enumerates exact work items `request_indices` + `kv_tile_indices` + `o_indptr`; kernel does one chunk per block. Zero empty splits by construction.
- Single-request sizing (`include/flashinfer/attention/decode.cuh` L706-731): skip split if `seq_len <= 256`; else probe `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `max_grid = blocks_per_sm * num_sm`, `kv_chunk_size = max(ceil_div(seq_len, max_grid/num_kv_heads), 256)`, `num_chunks = ceil_div(seq_len, kv_chunk_size)` — capacity-driven, 256-token chunk floor bounds merge cost.
- Batch paged plan (`include/flashinfer/attention/scheduler.cuh` L289-364): if `batch*gdy >= max_grid` no split; else `PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid, gdy, pages, max(128/page_size,1))` — binary-search pages-per-chunk so total chunks <= GPU capacity.
- Guard: `block_valid_mask` check is the FIRST kernel statement (`decode.cuh` L431: `if (block_valid_mask && !block_valid_mask[bx]) return;`), used only in CUDA-graph mode where the grid is padded to a fixed size. Non-graph launches size the grid exactly — no mask needed.
- Merge: separate `MergeStates` (LSE-weighted) or `AttentionSum` launch; CUDAGraph keeps padded grid + mask (paper arXiv 2501.01005: "load-balanced scheduling ... compatible with CUDAGraph which requires static configuration").
- Measured: formulas above are code-read; no published split-waste % (exact planning makes it structurally zero). Confidence: high (source-read).

## 2. FlashAttention-2 (split-KV decode heuristic)

- Formula (`csrc/flash_attn/flash_api.cpp` L282-315): early-out 1 split if `batch*heads*m_blocks >= 0.8*num_SMs`; cap `max_splits = min(128, num_SMs*2, n_blocks)`; for each candidate `s`: skip if `ceil_div(n_blocks, s) == ceil_div(n_blocks, s-1)` (ineligible: no width change), wave-efficiency `eff = (mblocks*s/SMs) / ceil(mblocks*s/SMs)`; pick smallest `s` with `eff >= 0.85 * max_eff`. Called with `num_sm*2` because 128-thread blocks.
- Equal-width splits: `n_blocks_per_split = ceil_div(n_blocks_total, num_splits)`; tail waste is bounded to one partial block per split, not 20% of blocks.
- Empty-split guard (`src/flash_fwd_kernel.h` L544-586): `if (n_block_min >= n_block_max)` — placed after address setup, BEFORE any q/KV smem load — and the block WRITES IDENTITY partials (0 to O-accum, -inf to LSE-accum) then returns, keeping the merge uniform.
- Merge: separate `flash_fwd_splitkv_combine_kernel` launch (grid over output rows, `Log_max_splits` template dispatch, max 128 splits).
- Measured: heuristic targets >=85% of best wave efficiency; no split-waste numbers published (design keeps them near zero). Confidence: high (source-read).

## 3. FlashAttention-3

- Same `num_splits_heuristic` verbatim as FA2 (`vllm-project/flash-attention` @ b273a026, `csrc/flash_attn/flash_api.cpp` L307-340; vendored by vLLM).
- Known defect, directly relevant: the heuristic disables splitting in low-head-count decode and underutilizes SMs; a sequence-aware split policy recovers 21-24% decoder kernel efficiency (arXiv 2604.00028, "Sequence-Aware Split Heuristic ... FlashAttention-3", 2026).
- Quant-KV: FP8 e4m3 QK^T on Hopper tensor cores with block quantization + incoherent processing (per-block scales, Hadamard rotate); up to ~1.2 PFLOPs FP8, 2.6x lower error than naive FP8 attention (arXiv 2407.08608). No int8 multi-byte split — native low-precision MMA only.
- Confidence: high for heuristic (source-read), paper-level for FP8 numbers.

## 4. vLLM paged attention v1/v2 (int8/fp8 KV)

- Mechanism: fixed `PARTITION_SIZE=512` tokens per split; `max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, 512)`; grid (heads, seqs, partitions) at v0.8.5 (`csrc/attention/paged_attention_v2.cu` L51, L91-100).
- Guard: FIRST statement of the kernel — `if (partition_idx * PARTITION_SIZE >= seq_len) return;` (`attention_kernels.cuh` L116) — before q load. Tail waste = one partial partition per sequence max (<=511 tokens), regardless of width.
- Merge: separate `paged_attention_v2_reduce_kernel`, grid (heads, seqs): per-partition `max_logits`/`exp_sums` in smem -> global max -> `global_exp_sum` -> `inv_global_exp_sum`-weighted accumulate of `tmp_out` (L567-652). Classic LSE merge.
- Quant-KV: K and V dequantized tile-by-tile to fp16 BEFORE MMA — `fp8::scaled_convert<K_vec, ...>(*k_scale)` (L287), same for V (L417) — per-layer scalar `k_scale`/`v_scale`. No quantized MMA.
- Note: these CUDA kernels are gone from vLLM main (csrc tree has no `paged_attention*`); decode moved to vendored FA/FlashInfer backends — the fixed-partition design is legacy but still the clearest guard-placement reference. Confidence: high (source-read at v0.8.5).

## 5. TensorRT-LLM XQA (`cpp/kernels/xqa/mha.cu`)

- Split sizing (host, L2851-2866): `nbSubSeqPerSeq = min(max(1, SMs/(batch*heads)), ceil(maxSeqLen/ctaTile.x))`; env override `XQA_NB_SUB_SEQ`; grid `{splits, heads, batch}` (L2936). Same occupancy shape as FlashInfer: fill the machine, cap by sequence length.
- Guard/fusion: NO separate merge launch — each CTA writes partials to `scratch` + arrives on a `semaphores` counter; the atomic-last CTA (`mbsmem.isLastCta`, L2637-2721, `writesOutput = isLastCta`) loads siblings' partials, LSE-merges in smem, writes final output. One launch total.
- Correctness edges: attention sinks added by exactly one partial CTA (L2548); fully-masked partial rows carry zero weight "and must contribute zero when merged" (L2556-2560).
- Quant-KV: templated `KElemType/VElemType ∈ {int8_t, __nv_fp8_e4m3}` (L1065-1250); K scale FOLDED into the logit scale — `qkScale = qScale * (isKVCacheQuantized ? kvCacheScale[0] : 1)` (L1777) — so the QK dot runs in low precision and one scalar multiply fixes it after; V dequant via `voScale` at output accumulate (L2544). Per-tensor scalar scale, not per-64-dim.
- Confidence: high (source-read).

## 6. Sparse top-K decode: Quest + NSA

- Quest (arXiv 2406.10774; mit-han-lab/quest): per-page metadata = elementwise min/max of K; each decode step scores every page with q-vs-min/max estimates, takes top-K pages, runs exact attention ONLY on gathered pages. Measured: up to 7.03x self-attention speedup, 2.23x end-to-end latency cut; 1.7x vs full-cache FlashInfer with 2048-token budget at 32K ctx (LongChat-7B, README benchmark). Selection cost = one small GEMV over page metadata + topk over #pages, per step, per head.
- NSA / DeepSeek (arXiv 2502.11089): hierarchical — coarse compressed attention scores ARE the top-k selection signal (selection folded into a useful compute pass, no dedicated selection kernel); gathers 64-token blocks; hardware-aligned kernel claims up to 11.6x decode speedup at 64k ctx (Table 4, memory-access volume).
- Takeaway: gather-only-KV decode wins are large and the selection overhead is small when metadata is cheap and selection reuses existing scores. Confidence: high (abstracts + README), kernel internals paper-level.

## Quant-MMA precedent (named)

- No surveyed kernel splits int8 operands into multiple 8-bit planes for 24-bit-effective accumulation — vLLM dequants pre-MMA, TRT-LLM runs native int8/fp8 MMA with scalar post-scale, FA3 uses native FP8 MMA. Nearest precedent for a multi-pass split-accumulate MMA is QServe/QoQ progressive quantization (W4 split into high/low components executed on INT8 tensor cores; dequant overhead 20-90% without it, arXiv 2405.04532). The planned 3-way 8-bit i8v3 split appears novel in attention-KV context.

## What this changes on the row

- Size `nsp` from device capacity, not a fixed constant: all three production planners derive splits from `SMs/(batch*heads)` occupancy (TRT-LLM formula `min(max(1, SMs/(B*H)), ceil(maxSeq/tile))`, FlashInfer `blocks_per_sm*SMs` probe with chunk floor, FA2 wave-efficiency sweep with 0.85 threshold) — at 65 chunks/row the FA2-style sweep or TRT-LLM formula lands near 1-3 splits, not 20, and eliminates the ~20% empty blocks structurally.
- Hoist the empty-exit to the FIRST kernel statement (vLLM L116 pattern); if the grid stays padded for CUDA-graph capture, follow FA2 and write identity partials (0, -inf) before returning so the merge stays uniform, or adopt FlashInfer's `block_valid_mask`.
- Hoist q prescale/pack to a plan stage — competitors either fold scales into one scalar applied at the epilogue (TRT-LLM `qkScale = qScale * kvCacheScale`) or do per-block work only once per chunk; recomputing q prescale in all 40 blocks per row has no precedent in any surveyed kernel.
- Merge+gate: two launches is the legacy shape (FA2 combine, vLLM reduce, FlashInfer MergeStates); TRT-LLM XQA proves the single-launch shape — semaphore counter, atomic-last CTA loads partials and LSE-merges in smem. For a 40-block-per-row grid this is directly portable and removes one launch.
- Page-claim resolve as ONE block is foreign to every surveyed kernel: all of them resolve page indices per-CTA from the block table (or pre-compute `kv_tile_indices` in the plan stage, FlashInfer) — multi-CTA resolve should mirror per-CTA index derivation rather than a centralized claim.
- i8v3 3-way 8-bit split INT8 MMA: no direct precedent in surveyed kernels (they dequant pre-MMA or run native low-precision MMA); QServe progressive quantization is the named technique family; proceed as novel, validate against int8-KV dequant kernels (vLLM style) as the accuracy baseline.
