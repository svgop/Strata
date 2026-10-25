# T50-R4 - Linear-Attention / GDN / SSM Kernel Engineering Dossier

**Researcher:** R4 | **Date:** 2026-10-25 | **Box:** RTX 5090 32GB (sm_120, int8 MMA 732 vs fp16 233 measured), 9950X3D, 184GB RAM
**Engine:** Qwen3.8-Flash-Next (hybrid MoE: GDN + QSA + PLE), solo 223 mean/234 best t/s, prefill ~6,100 t/s @32K
**Target:** +30-50%. **Dominant known cost:** GDN `commit(n_keep)` replays accepted rows at **2.47 ms/row** (verify.hpp:14-19) against a **9.0-10.4 ms** decode window; window flat across T=2..8.

## Part 1 - Per-source findings

### A. Algorithm papers

**A1. "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"** - arXiv:2406.06484 (NeurIPS 2024) - **Grade A**. Chunkwise WY representation of the delta-rule update; intra-chunk = batched tensor-core matmuls; inter-chunk = one boundary update. The family our adopted prefill chunking (kernels.cu, 1.68x @32K) implements. Key transferable fact: **the chunked form is exact for any n-row suffix, including n_keep = 2-4 rows** - a 4-row commit is one chunk, one serial state step, not four serial scans.

**A2. "Gated Delta Networks"** - arXiv:2412.06464 (NVIDIA, ICLR 2025; ~619 cites) - **Grade A**. Decay-gated delta rule; hardware-efficient chunked algorithm on tensor cores; the layer our GDN implements. Code: NVlabs/GatedDeltaNet. Follow-up: **Gated DeltaNet-2** - arXiv:2605.22791 (2026) decouples key-side erase from value-side write gates; architecture-level, flags where the family is heading.

**A3. "Transformers are SSMs" (Mamba-2 SSD)** - arXiv:2405.21060 - **Grade A**. State-space duality; blocks of 64-256 tokens; intra-chunk attention-like matmul, inter-chunk scalar-times-state recurrence. **Up to 8x vs Mamba-1 selective scan** (A100). Same structural lesson: never step row-serially when the update admits a chunk-parallel form. Our commit-replay is the exact anti-pattern.

**A4. "KVBuffer: IO-aware Serving for Linear Attention"** - arXiv:2605.19049 (May 2026) - **Grade A-** (SGLang v0.5.10 + Triton, Qwen3-Next-80B, 4x L40S TP). **The most directly load-bearing paper for our commit path.** Linear-attention decode reads/writes the full state every step - "2 MB in a Qwen3-Next GDN layer" (fp32 128x128 x 32 v-heads - identical shape to ours); "verifying 44 draft tokens occupies an additional 384 MB per-request." Three tricks: (1) buffer recent per-token K/V (paged, optimal ~ 2*sqrt(d) ~ 22.6 for d=128), update state in batch; (2) **parallel speculative verification: buffer draft KVs, compute outputs chunkwise, update state with only accepted tokens - no temporary states, no replay**; (3) KV-only decode for L < d. Measured: decode latency **-45.17%** (buffer 32); spec verify **2.78x at 8 drafts** (~3x predicted); e2e up to **1.46x**; max served requests **5x**.

**A5. "SpecLA: Efficient Speculative Decoding for Linear-Attention Models"** - arXiv:2607.16673 (Jul 2026) - **Grade B+**. Stores compact factors produced during verification to recover accepted states (checkpoint-plus-recovery, no replay); topology-aware verify kernels; acceptance updates only the accepted trajectory. **Up to 1.70x e2e** on H100 with GDN-1.3B. Confirms 2026 SOTA commits by recovering verify-time factors, never re-scanning.

**A6. "ReaDy: Recurrent States as Dynamic Context for Speculative Decoding"** - arXiv:2610.20503 (Oct 2026) - **Grade B+**. Drafter reads the target's *committed* GDN states (rank-8 state reader in the final draft layer); verification standard; "subsequent rounds read only states associated with committed tokens" - no rollback because the drafter never writes states. Evaluated on a **Qwen3.8-27B hybrid (48 GDN + 16 full-attn) - our model's family**. B200: **302.78 TPS vs 280.42 (DSpark) vs 95.86 (AR) = 3.16x**; acceptance length **4.53 vs 4.17 (+8.5%)**; state-gather kernel 16.39 -> 4.62 us.

**A7. Supporting:** **TFLA** arXiv:2503.14376 (NeurIPS 2025; tiled linear-RNN kernels beating FA/LA; tile-scheduling patterns for 128x128 states) - B. **MiniMax-01** arXiv:2501.08313 (1:7 hybrid, hand-written CUDA lightning attention **>75% MFU**) - A-. **Kimi Linear/KDA** arXiv:2510.26692 (per-channel gated delta; **up to 6x decode at 1M ctx**; TRT-LLM ships KDA kernels) - B+. **RWKV-7 Goose** arXiv:2503.14456 - B. **Quamba** arXiv:2410.13229 (+Quamba2, ICML 2025): **int8 PTQ of the recurrent hidden state with closed-form clipping**, 1.72x gen latency on Mamba-2.8B edge - the exactness analysis our parked state byte-cut lacked - B. **ThunderKittens** arXiv:2410.20399 (ICLR 2025; up to **14x on linear-attention variants**; Blackwell TC tensor-memory + warp specialization) - B+. **Wu et al. PMLR v262**: earliest commit-problem taxonomy (Joint Attainment and Advancement; Activation Replay; replay = the naive baseline everyone replaced) - B.

### B. Production sources

**B1. SGLang hybrid serving - PyTorch blog "Hybrid Models Meet SGLang" (2025-12-03)** - **Grade A**. THE production proof of our scoped-but-unstarted per-slot idea, verbatim: "one independent Mamba cache slot per draft token... When a sequence of draft tokens is accepted, simply promote the last accepted slot to become the new main state." For trees: per-drafted-token `S_new = S_parent + v k^T` rooted at the correct parent - no replay, no rollback. MambaRadixCache: request-level Mamba pool + token-level KV pool; prefix caching without kernel changes. Measured (H200, Qwen3-Next-80B-A3B-FP8, bs=1): T=2 -> 257.20 t/s (acc 2.709); T=3 -> 306.94 (3.413); T=4/topk4 -> 324.57 (4.231).

**B2. vLLM Qwen3-Next blog (2025-09-11)** - **Grade A-**. FLA Triton GDN kernels; hybrid KV manager equalizes physical memory; **full CUDA-graph mode by default explicitly to kill launch overhead** (relevant to our 2.47 ms/row); roadmap names further GDN kernel work.

**B3. FlashInfer GDN decode kernel** - **Grade B+**. `flashinfer.gdn.gated_delta_rule_decode` bench workload from Qwen3-Next layers (k-last state layout): the kernel natively takes (reads, writes) state tensors **by slot index** - production GDN decode kernels already expose the per-slot interface; commit = an index write, not a scan.

**B4. cuLA (inclusionAI/Ant)** - **Grade B**. CuTe DSL/CUTLASS chunk kernels for Blackwell SM100: prefill up to 2.08x, varlen avg 1.47x across 126 configs. Reference for sm_120 chunk kernels beyond Triton.

**B5. TensorRT-LLM** - **Grade B**. Qwen3-Next tracking (#7694), GatedDeltaNet in AutoDeploy (#9639), KDA kernels shipped; #15120: dedicated speculative-only Mamba/conv state buffers - the memory-manager pattern.

**B6. FLA library** - **Grade A-**. Triton reference (vLLM/SGLang build on it): `chunk_gated_delta_rule` (prefill), `fused_recurrent_gated_delta_rule` (decode: one kernel, one state read+write). Open Blackwell issues exist (#618 GDN hang on B200) - prefer CUDA/CuTe (FlashInfer, cuLA) for sm_120.

## Part 2 - Proposals, costed against our numbers

Anchors: commit replay = 2.47 ms/row x n_keep; n_keep ~ 2-4 => **C = 4.94-9.88 ms/window**. Window W = 9.0-10.4 ms; accepted tokens/window lambda ~ 2-4; 223 t/s. Reading 1 (commit inside the window timer - most consistent with flat W across T): residual R = W - C ~ 0.5-5.5 ms (central ~2.3). Reading 2 (window excludes commit): step = W + C ~ 14-20 ms, commit 35-49% of step. Either way the replay dominates.

### P1. Kill the commit replay: chunked batch commit of the accepted suffix (KVBuffer/SpecLA-style) - RANK 1
**Mechanism:** stop calling the row-serial scan n_keep times. During verify, write per-draft-row GDN inputs (k_hat, v_hat, gate, beta) to a small side buffer (a few KB/layer - KVBuffer shows temp-state materialization is unnecessary). On accept of n_keep rows, issue **one** chunked WY pass over those rows - identical math to our adopted 32K prefill chunk, n_keep <= 4: intra-chunk parallel on tensor cores, one serial boundary update.
**Costing:** one launch + one chunk ~ 0.3-0.8 ms regardless of n_keep (vs 2.47 x n_keep). W' ~ R + 0.6 ~ **~3 ms** central. t/s = lambda/W: **223 x (9.7/3.0) ~ 2.5-3x ceiling; conservatively +60-150% (~360-560 t/s); floor +25-50% even under Reading 2.** Clears the goal under every reading.
**Risk:** low-moderate; numerics differ only in fp accumulation order - the same class our prefill accepted when WY chunking was adopted. Gate/decay at the chunk boundary must mirror kernels.cu's gated form exactly.
**Cheapest falsifier (~half a day, no engine changes):** microbench the existing chunked WY kernel on a 4-row suffix + state commit for all GDN+QSA layers; total <= 1 ms and error within prefill-chunk tolerance = validated.
**Port:** M. verify.hpp:14-19 (replace per-row loop with one chunked call); kernels.cu (expose chunked scan at n_keep granularity - already handles short prefill tails); verify_kernels.cu (per-row side-buffer writes during verify); fused_gr.cu unchanged initially.

### P2. Per-slot verify: fold state update into the verify graph, promote by pointer swap (SGLang-style) - RANK 2
**Mechanism:** SGLang's production design (B1): each draft column owns a private state slot; verify writes slot j from slot parent(j); commit = set the request's state pointer to the last accepted slot - O(1), zero post-hoc kernel. FlashInfer k-last layout (B3) proves the API shape; TRT-LLM spec-only buffers (B5) the memory-manager pattern.
**Costing:** strictly less GPU work than P1: W' ~ R ~ **2.3-3 ms** central => similar-to-higher ceiling (**up to ~3x, ~500-700 t/s**). Cost shifts to memory: T slots x 2 MB/layer x 48 layers => T=4 ~ 384 MB, T=8 ~ 768 MB (KVBuffer's own figure); fp16 slots halve it.
**Risk:** moderate - invasive memory management (slot allocation, parent-index plumbing); numerics arguably cleaner than chunked (exact per-token recurrence).
**Falsifier:** extend P1's microbench: T fused-recurrent updates writing T distinct slots from parents, commit = index swap; verify vs replay in a T=4 loop.
**Port:** M-H. verify_kernels.cu (slot index through every state read/write), verify.hpp (allocator + promotion), fused_gr.cu (slot-addressed fused recurrent kernel - likely already takes state pointers; mostly an allocator change). After P1 ships; P1 is the subset with 80% of the win.

### P3. CUDA-graph capture of the verify+commit region (vLLM-style) - RANK 3
**Mechanism:** vLLM runs full CUDA-graph mode by default because custom decode paths drown in launch overhead (B2). A 2.47 ms/row row-serial commit smells like latency+launch+sync per row, not FLOPs (per-row math ~ us). Capture the T-row verify + commit loop as one graph with static slot/buffer addresses.
**Costing:** if graphing alone cuts per-row commit to fused-recurrent-class ~0.3-0.5 ms: savings ~ (2.0 ms x n_keep) ~ **4-8 ms/window** - comparable to P1 for a day of work.
**Risk:** low (no math change); needs static shapes/bucketing across T and n_keep.
**Falsifier:** nsys the window; >1 ms per-row launch/sync gaps => graphing wins. Hours.
**Port:** S. verify.hpp + verify_kernels.cu launch paths only.

### P4. KVBuffer-style buffered chunkwise decode (defer state writes; KV-only for short context) - RANK 4
Beyond the spec-verify trick: defer all decode state updates into a ~2*sqrt(d) ~ 23-32-token K/V ring, amortizing the 2 MB/layer state read+write; L < d skips state entirely. Their numbers: decode -45%, verify 2.78x at 8 drafts. For us largely subsumed by P1/P2 at solo batch; incremental value is cheap T=8+ windows (their 2.78x at 8 drafts) => **+10-30% on top of P1 via deeper windows**. Risk moderate (ring bookkeeping; QSA's fp32+FWHT path fits less cleanly). Falsifier: microbench GDN layer bs=1 with state deferred over 32 tokens vs per-token. Port H (kernels.cu + fused_gr.cu + verify.hpp lifecycle). Only if P1+P3 land and we want T>6.

### P5. Recurrent-state dtype cut (fp32 -> bf16 slots / int8 with Quamba-style clipping) - RANK 5
Quamba quantizes the recurrent state to int8 with closed-form clipping + error feedback at 1.72x gen latency on edge Mamba - the quality analysis the parked bit-change lacked; but their wins were bandwidth-bound edges. On a 5090 post-P1: slot-write traffic ~ T x state x layers ~ 768 MB (fp32, T=8) -> 0.43 ms; int8 saves ~0.32 ms/window (**~+5-10% at W'~3 ms**). bf16 slots also halve P2's slot memory. Falsifier: eval suite with bf16 round-tripped states (zero kernel work - cast in the commit path). Port S-M. After P1/P2; value = memory headroom more than latency.

### Bonus (drafter-side, orthogonal): ReaDy-style committed-state reads into the drafter
Rank-8 reader over committed GDN states: **+8.5% acceptance length** (4.53 vs 4.17) on a Qwen3.8-27B hybrid at 3.16x decode (B200). If P1 lands, lambda is the remaining lever; +8% lambda ~ +8% t/s. Effort M-H (drafter graph change). Park behind P1.

### Recommended sequence
1. **P3 falsifier** (hours): nsys; graph the verify+commit loop => up to -4-8 ms/window.
2. **P1** (days): chunked suffix commit - the single highest-EV change; the 2.47 ms/row replay is pure anti-pattern per A1/A3/A4 and sits on the critical path of every accepted token. Landing zone **+60-150%, worst case +25-50%**.
3. **P2** if slot memory fits and tree drafts matter; **P5** for slot memory; **P4/bonus** next campaign.

## Sources
[arXiv:2406.06484](https://arxiv.org/abs/2406.06484) | [arXiv:2412.06464 + NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) | [arXiv:2605.22791 GDN-2](https://arxiv.org/abs/2605.22791) | [arXiv:2405.21060 SSD](https://arxiv.org/abs/2405.21060) + [chunked-scan blog](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm) | [arXiv:2605.19049 KVBuffer](https://arxiv.org/html/2605.19049v1) | [arXiv:2607.16673 SpecLA](https://arxiv.org/abs/2607.16673) | [arXiv:2610.20503 ReaDy](https://arxiv.org/html/2610.20503) | [arXiv:2503.14376 TFLA](https://arxiv.org/abs/2503.14376) | [arXiv:2501.08313 MiniMax-01](https://arxiv.org/abs/2501.08313) | [arXiv:2510.26692 Kimi KDA](https://arxiv.org/abs/2510.26692) | [arXiv:2503.14456 RWKV-7](https://arxiv.org/abs/2503.14456) | [arXiv:2410.13229 Quamba](https://arxiv.org/abs/2410.13229) | [arXiv:2410.20399 ThunderKittens](https://arxiv.org/html/2410.20399v1) | [PyTorch/SGLang hybrid blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention) | [vLLM Qwen3-Next](https://vllm.ai/blog/2025-09-11-qwen3-next) | [FlashInfer GDN bench](https://bench.flashinfer.ai/kernels/gdn_decode_qk16_v32_d128_k_last?p=0.95) | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | [inclusionAI/cuLA](https://github.com/inclusionAI/cuLA) | [TRT-LLM #7694](https://github.com/NVIDIA/TensorRT-LLM/issues/7694) [#9639](https://github.com/NVIDIA/TensorRT-LLM/issues/9639) [#15120](https://github.com/NVIDIA/TensorRT-LLM/issues/15120) | [Wu et al. PMLR v262](https://raw.githubusercontent.com/mlresearch/v262/main/assets/wu24a/wu24a.pdf)
