# Frontier attention-kernel audit (2026-11-13) — FA3/FA4/FlashMLA/XQA/FlashInfer vs our attention

Batch 1 of the 18-agent frontier wave. Workload frame: attention = 5.4 ms of the 13.4 ms conc window; QSA
prefill = the TTFT wall. sm_120 platform facts pinned (TMA yes; tcgen05/TMEM/wgmma NO — twice-verified).

## Ranked findings
1. **Decode/drafter TC paths are env-gated; the code default silently falls to scalar FFMA** — any path that
   skips the launcher recipe (tests, bare starts, forks) pays scalar. Dispatch gap, not kernel gap. S port:
   flip default, keep the fallback ladder (preserves measured +2.8/+3.5/+2.5%).
2. **The decode split-TC kernel runs int8 KV at FP16-MMA rate; the measured 3.1x INT8-MMA engine (732 vs 233
   TOPS) is prefill-only** — port the imma16832 engine into the decode split kernel (shape analysis already
   in T17D §4B). +3-9% conc, growing with context. M-L port, int32-accum numerics class (exactness argument
   already written at qsa_prompt_attn.cu:1086).
3. **DS4 attention fully scalar under CUDA while FlashMLA runs the BYTE-IDENTICAL FP4/FP8 KV format on tensor
   cores** (FlashMLA FP8 = 512×e4m3+e8m0/32 = our kKvFp8 ring; FP4 = our kKvFp4). 1024 TFLOPS sparse decode
   on B200 at hd=512 = exactly ours. Not a library pull (no sm_120 support) — a build. DS decode per-query
   125 us → sub-50 plausible. L port. (Matches our own earlier WMMA-port finding; FlashMLA is the reference.)
4. **Rings cost a dequant→requant→backup/restore pass per step; paged KV deletes the conversion class** —
   the Qwen side already reads paged KV in-kernel; ds4 ring_to_ext/ext_to_ring/ring_restore is a pure
   bandwidth tax the frontier never pays (rollback = table decrement). M-L; a simplification.
5. Indexer/selection: QSA TC scorer matches frontier; DS4 scalar; fused indexer+topK shipped by cuDNN DSA
   (SM100) and DeepSelect (topk=512, 2-20x over torch.topk) — radix pattern exists in-tree at :717.
   Tie-rule exactness contract governs any select change.
6. **GDN recurrence scalar FP32 end-to-end; FlashInfer 0.7.0 SHIPS fused TC GDN decode for SM120**
   (projection+conv+gating+recurrent-state combined) — external validation that our decode line is fusible
   on our arch. L port; delta-rule numerics; F-224/GDN_CHUNKED gates exist.
7. sm_120 mechanics Ampere-era in every attention kernel (no TMA/mbarrier; ≤65 KB smem; clusters unused) —
   TMA is substrate not a standalone win (T17D verdict stands); fold into 2/3/6; the mbarrier pattern is
   proven in-tree by headport's attn_chain_kernel.cuh.
8. KV-split heuristic env-gated + capped 20; frontier makes split+GQA-packing default (XQA NB_SUB_SEQ).
   S-M; short-context latency.

## Already matching/beating
Paged KV (Qwen family) = FlashMLA/XQA parity; online softmax + flash-decoding split-K merge done right;
in-kernel dequant (exact int8→half2 mantissa trick, 3-term int8 split — never pre-dequantizes on the hot
path); Hadamard incoherent processing = FA3's accuracy mechanism ALREADY in-tree; cp.async double-buffered
pipelines everywhere; GQA packing in the prompt MMA engines; the INT8-MMA engine itself is frontier-class
(only wiring lags); hd-256 = the FA sweet spot; tcgen05/FA4/WGMMA correctly absent (not sm_120); selection
tie-rule discipline stricter than frontier defaults.

## Bottom line
Biggest honest gaps: (2) int8-MMA decode — the only lever attacking the width-growing 4.4 ms conc line
directly; (3)+(4) DS4 scalar attention + ring conversions vs FlashMLA's byte-identical TC-native formats;
(6) GDN fused-SM120 (FlashInfer proved it). QSA's TC prompt path/softmax/paged-KV/selection are at or above
frontier practice; 1 and 8 are cheap hardening.

Sources: FA4 blog + arXiv:2603.05451; FA3 arXiv:2407.08608; FlashMLA (deepseek-ai); cuDNN DSA; DeepSelect;
TRT-LLM XQA #3762; FlashInfer releases (0.6.14 CuTe-DSL GDN, 0.7.0 SM120 fused GDN); vLLM attention-backend
docs; NSA arXiv:2502.11089; SGLang DSA #15025; vLLM DeepSeek-V3.2 blog; in-tree T17D/T26-3/T39 dossiers.
