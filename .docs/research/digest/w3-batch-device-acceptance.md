---
type: digest
id: w3-batch-accept-scout
created: 2026-10-31
status: evidence
topic: Batched on-device speculative verification + sampling (rows <= 8) — vLLM, TRT-LLM, SGLang, FlashInfer, Medusa
---

## Summary

Every leading engine runs Leviathan/Chen acceptance AND the residual resample for a whole batch on
device: no host logits D2H, no per-sample host sync. Two shapes: (a) one kernel per REQUEST walking
its window serially, grid = batch (TRT-LLM, SGLang, FlashInfer); (b) vLLM's split into batched
vocab-pass kernels over a CSR layout of per-request logit rows. None bakes a host Philox counter:
all draw from (per-request seed, position/offset) device buffers, so graph replay is safe by
construction. Block verification (arXiv:2403.10444, in vLLM + SGLang) upgrades token-wise rejection
further, losslessly — it directly monetizes our coupled-draw disagreement window.

## vLLM — MRV2 probabilistic rejection sampler (Triton, all-device)
- Mechanism: `rejection_sample()` (vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py):
  4-5 Triton launches — per-(row, vocab-block) stats (BLOCK 8192); optional joint-prob/residual-mass
  for block verification; `_rejection_kernel` (grid=num_reqs, 1 warp) walks each request's window
  doing the log-space ratio test `target_logprob > log(u) + draft_logprob`; `_resample_kernel`
  computes residual `lp + log1p(-exp(lq-lp))` = log(max(p-q,0)) and samples it via seeded
  Gumbel-max over vocab blocks; `_insert_resampled_kernel` writes bonus/replacement. Draft probs
  come from a staged `DraftLogitsCache` [max_reqs, steps, V] — same shape as our per-slot q rows.
  Batch shape: CSR `cu_num_logits`, rows = sum(reqs*(k+1)), variable k per request (adaptive
  verification compacts on device), chunked at 1 GB fp32 logits (#48630 OOM fix).
- RNG-in-graph: counter-free. `seeds` = UvaBackedTensor (pinned host-mapped, per request, staged at
  schedule time); accept coin `tl.rand(seed ^ (step<<32), pos)`; Gumbel noise
  `murmur3(seed, pos, vocab_offset)` with draft salt 1<<30. Pure function of device buffers ->
  replay-safe, nothing to bump. Greedy stores target argmax, resample skipped.
- Numbers: fp64 Gumbel ~1/32-1/64x fp32 throughput on H100/Ada/Blackwell (code comment); fp32
  logsumexp caused an acceptance-rate gap (#40651), fixed by fp64 promotion (#64242, 2026-10-28) —
  precision bugs masquerade as acceptance drops. PRs: #35461 (2026-03-11, Triton sampler), #38496
  (fused), #46781 (2026-06-30, block verification), #50183 (NaN argmax fix).
- Confidence: verified-from-source (main @ 2026-10-31).

## TensorRT-LLM — chainSpeculativeSampling + sampledAccept (one CUDA kernel per window)
- Mechanism: `chainSpeculativeSamplingKernel` (kernels/speculativeDecoding/): grid = batchSize, one
  1024-thread CTA per sequence. Thread 0 does the accept loop `curand_uniform * pDraft < pTarget`
  on precomputed probs; on first rejection the CTA computes residual `max(target-draft, 0)`
  (float4-vectorized) and inverse-CDF samples it: block-reduce tile sums to the threshold tile,
  BlockScan inside it, atomicMin winner; falls back to target when residual is empty. One launch =
  accept + resample + emit (ids + per-row numTokens, -1 padding). Exactly our "one
  spec_verify_merge launch". Tree path: dynamicTreeKernels; MiniMax-M3 `sampledAccept.cu` does
  TP-sharded top-p rejection sampling across ranks without a full-vocab all-gather (rows <= 48;
  8 CTAs/row for rows<=32, cross-rank 4-word record multicast, Gumbel-max over CTA candidates).
- RNG-in-graph: `curand_init(seed[row], kPerRowSubsequenceBase | offset[row], 0)` — seed/offset are
  int64 device tensors, one entry per request, staged by the caller per step (thop signature
  `Tensor seed, Tensor offset`; kernels read const). Host writes the buffers before replay =
  mapped staging, no sync. Philox4x32-10 reimplemented inline in sampledAccept.
- Numbers: sampledAccept header — "first candidate lands in the nucleus with probability >= top_p,
  so most rows decide on their second exchange"; docs: rejection sampling preserves the target
  distribution but lowers acceptance vs greedy-tied draft chains.
- Confidence: verified-from-source (main @ 2026-10-31); runtime writer of seed/offset: inferred.

## SGLang — chain_speculative_sampling_triton (+ seeded coins)
- Mechanism: kernels/ops/speculative/reject_sampling.py: grid = batch, one program per request;
  scalar window walk, vectorized vocab loops (BLOCK_V 4096); classic coin*q < p or block
  verification (prefix_prob recurrence in registers; accept h = residual_mass / (residual_mass +
  1 - prefix_prob)); residual sampled in-kernel from max(p-q,0) with a final coin. Target probs
  pre-processed by full-vocab passes: temperature softmax, then top_k_renorm_prob /
  top_p_renorm_prob (renorm-then-plain-kernel, vs vLLM fusing). Greedy = `verify_tree_greedy`
  Triton kernel; trees via FlashInfer-derived `TreeSpeculativeSamplingTargetOnly`; batch = bs x
  draft_token_num candidates + tree CSR (retrieve_index/next_token/sibling).
- RNG-in-graph: coins are precomputed device tensors: `murmur_hash32(seed, seq_lens, col)` clamped
  to [0, 1-2^-24); draft proposals use the same murmur-keyed exponential race (`fast_sample`) with
  a salt + per-step column offset, "reproducible under batch reordering and CUDA graph replay";
  torch.rand only in the non-seeded fallback. E2E determinism tests exist.
- Numbers: plumbing comment — missing per-row top_k handoff "quietly drops draft accept length
  about 20%"; verify captured under CUDA graph (target_verify graph accounting in worker v2).
- Confidence: verified-from-source (main @ 2026-10-31).

## FlashInfer — sampling kernels + ChainSpeculativeSampling
- Mechanism: include/flashinfer/sampling.cuh `ChainSpeculativeSampling`: one CTA per sequence;
  thread 0 accept loop; block residual relu(target-draft) sampled via DeviceSamplingFromProb (CUB
  scan/reduce, early exit). Same family as the sorting-free top-k/top-p dual-pivot rejection
  kernels, O(log(1/eps)) rounds guaranteed (v0.2.3+). API takes seed/offset as int scalars OR
  device tensors. RNG-in-graph: kernel reads `seed_arr[bx*stride]`/`offset_arr[bx*stride]`,
  `curand_init(seed, bx, offset)`; replay safety is the CALLER's job — the default torch.Generator
  path bakes capture-time seed/offset into the graph, NOT replay-varying.
- Numbers: sorting-free kernels cut overall sampling time >50% vs sort-based (vLLM, 1xH100, 3
  models) — flashinfer.ai/2025/03/10/sampling.html; pitfall: non-monotonic fp prefix sums break
  inverse-CDF search (same blog).
- Confidence: verified-from-source (code); measured-in-paper (blog benchmarks).

## Medusa — typical acceptance (lossy baseline)

Typical acceptance: accept iff p_original(x|x_<) > min(eps, delta * exp(-H(p_original))) (arXiv:2401.10774 S2.3.1) — drops distribution-exactness to raise temperature acceptance; tree verify stays one fused pass; Medusa-2 e2e 2.3-3.6x. Not for our lossless row. measured-in-paper.

## RNG-in-graph, cross-engine

vLLM: no mutable state, keyed hash/Philox, seeds pinned-UVA per req + pos on device — safe, staged
once per request. TRT-LLM: curand Philox, per-row device seed/offset staged per step — safe, host
writes buffer. SGLang: murmur3 coins f(seed, seq_lens, col) on device — safe. FlashInfer: curand
Philox from device arrays — safe only if the caller stages them.

## What this changes on the row
- "One spec_verify_merge launch" validated: TRT-LLM and SGLang fuse accept + resample + emit in one
  per-request kernel (vLLM needs 4-5). Keep the fused plan.
- "Philox counter via mapped staging" = exactly TRT-LLM's pattern; the stronger vLLM/SGLang variant
  is counter-FREE keyed draws (seed, pos): zero per-step staging, deterministic under reordering,
  e2e-tested. Worth adopting over counter staging.
- Residual on device: second uniform + in-kernel inverse-CDF (TRT-LLM/SGLang/FlashInfer) or
  Gumbel-max over log(max(p-q,0)) (vLLM, skips a softmax pass); either way guard fp32 logsumexp —
  fp32 drift shifted vLLM acceptance rates (#40651, #64242).
- Add block verification (arXiv:2403.10444) instead of token-wise rejection: provably optimal
  lossless accepted length; same prefix_prob recurrence shipped in vLLM (#46781) and SGLang;
  directly monetizes the ~30-40% coupled-draw disagreement window.
- top-k/top-p: pre-renormalize target probs (SGLang), fuse pivot rejection (FlashInfer), or Gumbel
  with masked logits (vLLM); at rows<=8 renorm passes are affordable, fusing avoids extra sweeps.
- Copy the numerical-safety set: NaN->-inf before argmax (#50183), open-interval uniform clamps,
  last-valid-index fallback when fp prefix sums undershoot (FlashInfer, TRT-LLM).

## Sources
- vLLM main @ 2026-10-31: vllm/v1/worker/gpu/spec_decode/rejection_sampler{,_utils}.py, vllm/v1/worker/gpu/sample/{gumbel,states}.py; PRs #35461 #38496 #46781 #48630 #40651 #64242 #50183
- TRT-LLM main: kernels/speculativeDecoding/chainSpeculativeSamplingKernels.cu, kernels/minimaxM3/sampledAccept.cu, kernels/fusedSampling/fusedSamplingKernelsCommon.cuh, thop/chainSpeculativeSamplingOp.cpp, docs/source/features/speculative-decoding.md
- SGLang main: srt/speculative/{eagle_utils,spec_utils}.py, kernels/ops/speculative/reject_sampling.py, kernels/jit/csrc/speculative/sampling.cuh
- FlashInfer main: include/flashinfer/sampling.cuh, flashinfer/sampling.py, flashinfer.ai/2025/03/10/sampling.html
- Papers: arXiv:2211.17192 (Leviathan/Chen); arXiv:2403.10444 (block verification, ICML'24); arXiv:2401.10774 (Medusa)

## Methods note

Repos read at main, 2026-10-31, via raw.githubusercontent + GitHub API (files cached in /tmp/w3src).
Unproductive: web search for the Triton rejection-sampler PR — file commit history was the
provenance path instead. FlashInfer chain-kernel csrc not found in tree (kernel lives in
sampling.cuh; sufficient). TRT-LLM runtime writer of seed/offset not located; thop signature +
const kernel reads confirm the staging contract (reader verified, writer inferred). Repo-local
constraint honored: only this digest file written.
