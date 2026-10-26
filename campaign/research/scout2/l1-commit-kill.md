# SCOUT2 — L1 commit/kill: kernel internals for the GDN commit falsifier

Row: GDN commit replays accepted spec tokens at 2.47 ms/row. Round: 2 (kernel-level). Date: 2026-10-26. Method: fetched issue/PR/paper primaries + vLLM API docs; all numbers below quoted from those sources. Grades: A = primary source w/ code or measurements; B = maintainer statement, no code; C = inference from geometry.

## F1. FlashInfer gdn_decode — how per-token checkpoints are written (Q1)

**Answer: FULL per-token state, written in-kernel during verify. Not compact.**

- Buffer: `intermediate_states_buffer: [pool, T_cache, HV, V, K]`. Kernel "writes state_after_token_t to `intermediate_states_buffer[pool_idx, t, :, :, :]`" — one full HV×V×K recurrent state per draft position per pool slot ([RFC #2687](https://github.com/flashinfer-ai/flashinfer/issues/2687), A).
- API: `gated_delta_rule_decode(q, k, v, state, A_log, a, dt_bias, b, state_layout="VK", state_indices=None, scale=None, output=None, intermediate_states_buffer=None, disable_state_update=False, use_qk_l2norm=True)`; q/k `[B,T,H,K]`, v `[B,T,HV,V]`, state `[pool,HV,V,K]`. BF16 kernel requires `T ∈ {1..4}, K=V=128`; a merged PR "add BF16 state kernel with MTP support beyond T>4 with intermediate caching" lifts T>4. State checkpointing on the SM90 prefill side: `kEnableCheckpointing` in `flat_collective_tma_warpspecialized_delta_rule.hpp` (PR #2908); `kInitStateFromInput` loads `ptr_input_state` instead of zeroing. Prefill writeback to a *paged* state pool is still an open gap (5.5/5.6) (A).
- Dtype: shipped MTP path stores intermediates in **BF16** ("2× smaller than FP32 buffer, allows longer rollback windows"). Our row's fp32×T assumption is the upper bound; the production form is bf16×T (A).
- Perf note: PR #2842 "eliminates ilp=1 fallback that caused ~33% slowdown at T=3→4 boundary" — dispatch step-functions at T boundaries are real in this kernel family (A).
- **Round-1 correction:** "b12x" is not a GDN verify kernel. FlashInfer #5561: "the b12x direct-micro kernel is gated to intermediate ≤ 512" — that is the **SM12x NVFP4 MoE decode** gate ([#5561](https://github.com/flashinfer-ai/flashinfer/issues/5561), A). GDN on our sm_120 box goes through the CuTe DSL decode family ([PR #2679 summary](https://github.com/mit-han-lab/KernelWiki/blob/master/sources/prs/flashinfer/PR-2679.md): BF16 hidden-state storage, T=1 + MTP variants; release notes: "BF16 state recovery/decode kernel that writes SSM state into preallocated space", B).
- **No published numbers:** [bench.flashinfer.ai gdn_decode_qk16_v32_d128_k_last](https://bench.flashinfer.ai/kernels/gdn_decode_qk16_v32_d128_k_last?p=0.95) exists (GVA 16q/16k/32v, d=128, k-last, seq_len 1, captured from Qwen3-Next linear layers, "verified") but has **zero solutions/latencies**. Do not cite FlashInfer bench numbers; generate our own (A).

## F2. SGLang ReplaySSM ring — record sizes, savings math, commit parallelization (Q2)

- **Record contents** ([#28695 ring spec-verify](https://github.com/sgl-project/sglang/issues/28695), A): per committed token per head: `d/k` (activation dtype, output-reconstruction only), `rawv/rawk` (activation dtype, lossless exact-fold replay), `g/β` (fp32). Plus frozen checkpoint S0 flushed every L committed tokens. For exact-fold only: rawv 256 B + rawk 256 B + g 4 B + β 4 B = **520 B/head/token** (bf16) vs full state 16 KB/head (bf16) / 32 KB (fp32) — ~30-60× smaller per head.
- **Paper math** ([arXiv:2610.13778](https://arxiv.org/html/2610.13778), A): per head, window L: `M_conv = 8L·d_v·d_k` vs `M_replay = 4(L+1)·d_v·d_k + ((L−1)(L+2)/2)·r`; checkpoint ratio → (L+1)/(2L) ≈ 0.5+, i.e. state traffic roughly halves. GDN record `r = 2d_v + 2d_k + 4` bytes. Nemotron-3-Super (L=16): 0.58 ratio = 1.7× traffic cut.
- **SGLang measured savings:** ring 1.81 GB vs `intermediate_ssm` 11.48 GB (Qwen3.5-35B-A3B TP1) — "6.4× smaller", pool avail 17.48 → 28.96 GB; 27B rank: ring 9.5 MB vs 226 MB for 8 draft steps (bf16, ring 16) (A). Sibling compact-cache PR [#27658](https://github.com/sgl-project/sglang/pull/27658) (A): stores `(k, Δv = vᵀ − kᵀDiag(α)S_{t-1}, decay α)` = `[3,128]` fp32 per head per token; Qwen3-Next-80B TP4/H20: `intermediate_ssm_state_cache` **2.32 GB → 0.06 GB** (~38×); freed 2.26 GB went to KV (3.39M → 3.58M tokens). Flag `--enable-linear-compact-spec-cache` (open, approved, CI red).
- **Commit mechanics:** accept length a → per-slot cursor advance `h ← h+a` once per verify step; rejected drafts are a pointer move, no state write-back. Flush replays only committed records from frozen S0 ("every committed token is replayed exactly once"); invariant `write_pos + spec_len ≤ ring_len` (ring ≥ 2× draft len, power-of-two). Reconstruction is a small GEMM on tensor cores; GDN draft corrections via one T×T triangular solve (UT transform, `(I+A)⁻¹`), output-only — the outer product `dⱼkⱼᵀ` never hits HBM on non-flush steps. Parallelization: the exact-fold kernel `gdn_replayssm_exact_fold_kernel` is **sequential per replay**, `num_warps=1`, `[BK,BV=32]` tile, written as a bitwise clone of `fused_sigmoid_gating_delta_rule_update_kernel` (division-form L2 norm, eps inside sqrt, decay→delta→rank-1 order) → accuracy max_abs ~1e-7. Requires fp32 SSM checkpoint (`--mamba-ssm-dtype float32` auto-set); verify output dots default tf32 (~5e-4, under the ~4e-3 bf16 cast floor). Grid over heads/layers/batch not published (B on grid, A on rest).
- **Known bug class:** decode-ring `_replay_metadata` advanced shared `write_pos` +1 on verify replays — "7938 foreign +1 writes across 2902 verify commits at bs=8" → repetition loops; fixed in b53102c. Our falsifier must check this class (A).
- Flag: `--enable-gdn-replayssm-spec`, `--linear-replayssm-cache-len` (default 16); GDN-only, topk ≤ 1; mutually exclusive with `--enable-linear-replayssm`.

## F3. Published commit-cost numbers (Q3)

| Path | Measurement | Source/grade |
|---|---|---|
| SGLang **scatter** commit (cache-intermediates, D=4, Qwen3.5-397B TP4, c256) | **≈948 µs/forward** (the kernel our 2.47 ms row mirrors) | [#27725](https://github.com/sgl-project/sglang/issues/27725), A |
| #27725 e2e ceilings (same config) | D1 (pool-direct slots, no scatter) ≈ +5%; fused recovery+output (2.4) +7.1/+11.3%; multi-stream/next-forward fusion (2.5) **≈ +14.8%**; plain Triton recompute −2.5% thru | A |
| #26520 none-mode (final-state recompute, closed unmerged) | TPOT 19.64 vs 18.77 ms (+4.6%), thru −2.5% at equal accept (2.9); `intermediate_ssm_state_cache: SKIPPED` | A |
| ReplaySSM paper, spec kernel T=4 B=256 (B300) | vLLM snapshots 831-1043 µs; STree 1085-2079 µs; Mamba-in-the-Llama 923-2261 µs; ReplaySSM verify = 0.89-1.21× a decode step | A |
| vLLM #27658 compact replay (H20) | TPOT b1 2.02 vs 2.03 ms, b32 9.36 vs 9.55 ms; GSM8K thru +2.7%, acc parity 0.951/0.952 | A |
| TRT-LLM #14003 (PR, merged) | nsys: `numGenReq × (1 + max_draft_len)` cudaMemcpyAsync + cudaStreamSynchronize **pairs per iteration**; replaced by pitched `cudaMemcpy2DAsync` + deferred `flush_state_transfers()`; sync off critical path. No µs given | A |
| TRT-LLM #15120 | Allocation bug only (spec-only SSM buffers leaked when spec off; OOM on big hybrids); fixed PR #15020. No timing | A (context) |
| FlashInfer bench_gdn | No published outputs | A (absence) |

#27725 memory anchor (256 slots, TP4, D=4): full-cache GDN state 29.2 GB vs recompute 6.6 GB per GPU; per-request 114 → 26 MB (−77%); intermediate cache = T × persistent.

## F4. Qwen3-Next runtime — `num_accepted_tokens_gpu` (Q4)

**Confirmed, no host sync** (vLLM `mamba_utils` docs, A):

- `num_accepted_tokens_gpu: torch.Tensor[num_reqs]` — per-request accepted counts, **device-resident**. Passed to `MambaSpecDecodeGPUContext.run_fused_postprocess(...)`, whose fused kernel "computes decisions and performs mamba state copies **entirely on GPU**, eliminating the CPU-GPU sync that was previously needed."
- Kernel writes `num_accepted_tokens_out` (pre-initialized from the input; overwrites entries to 1 when src_block == dest_block), with an in-kernel **snapshot copy** so the reader never races the in-place update ([#56736](https://github.com/vllm-project/vllm/issues/56736): snapshot is a same-stream copy ordered before the reader kernel — Xid 31 context, A).
- CPU sees counts only via `non_blocking=True` async `copy_` back in `postprocess_mamba_align_gpu`, consumed next iteration. Copy-spec internals: `_copy_mamba_state_block` mirrors V1 `get_conv_copy_spec`/`get_temporal_copy_spec`; temporal states full-block copy with `TEMPORAL_TILES` tiling; `_memcpy_u64_tiled` needs sub-8B alignment match (GB200 misalign trap).
- SGLang equivalent: `replayssm_write_pos` per-slot cursor advanced once per forward, CUDA-graph-safe (A).

## F5. Chunked-WY-suffix commit — anyone in production? (Q5)

**Attempted, measured, not shipped as default.** The math is deployed only inside ReplaySSM (UT-transform verify + sequential exact-fold flush), flag-gated default-off. As a *standalone* chunked-suffix commit it is exactly SGLang's "none mode" / Direction 2: PR [#26520](https://github.com/sgl-project/sglang/pull/26520) — stash post-conv GDN inputs during verify, recovery-only recurrence kernel writes the accepted final state; **closed unmerged Aug 7 2026**; follow-up #29451 (FlashInfer recovery integration). #27725's ladder treats it as the high-ceiling path (2.4 fused +7.1/+11.3%, 2.5 +14.8%) vs Direction 1's ~+5% ceiling. FlashInfer/SGLang-full/vLLM/TRT-LLM defaults all remain per-slot full-state checkpoints or copy-spec. So: **our P1a design has one production-track attempt, measured as a memory win but a TPOT regression until fused — nobody has shipped the fused form.**

## MEMORY MATH (our geometry: 36-48 layers, HV=32, K=V=128, fp32 state 2 MiB/layer, T=6, batch 6; box = RTX 5090, 32 GB GDDR7, ~28-30 GB resident)

State/layer/slot: 32×128×128×4 B = 2 MiB exactly.

| Design | Extra VRAM (batch 6) | Verify-path state writes/forward | Commit cost | Fits? |
|---|---|---|---|---|
| A. Per-slot full-state, fp32 (row's assumption) | 12 MiB/layer/req → 2.53-3.38 **GiB** | 2.53-3.38 GiB writes + commit promote 2 MiB/layer (576 MB w) or scatter 4 MiB/layer | scatter (our 2.47 ms class; SGLang measured 948 µs) or free-list update if pool-reserved | **NO** at fp32 — 2-3.4 GiB next to 28-30 GB residency |
| A'. Same, **bf16** (shipped FlashInfer form) | 1.26-1.69 GiB | 1.26-1.69 GiB | index update if T+1 pool slots reserved; else scatter | Marginal — only with KV cut |
| B. Compact ring (rawv/rawk/g/β) | ~520 B/head/token → 29-58 **MB** | tiny appends | cursor advance + rare exact-fold flush (sequential, 1e-7 exact) | **YES**, trivially |
| B'. `(k,Δv,α)` fp32 `[3,128]` (#27658 form) | 48 KB/token/layer → ~84 MB | tiny | replay kernel re-applies deltas | **YES** |
| C. Chunked suffix (our P1a) | stash ~17 KB/token/layer → ~29-58 MB, 0 extra state | **none** (output-only verify) | one chunked pass: reads stash, writes final state 2 MiB/layer → 432-576 MB w/forward | **YES** |

Bandwidth reality check: ReplaySSM ring measurements show achieved HBM bw 6.7% (b1) → 60.8% (b16) → 74.7% (b64) on H20. **At batch 6 we are latency-bound, not bandwidth-bound** — traffic-based projections (12 MiB×T×layers) overestimate A's step cost; A's real cost is extra kernel writes at low achieved bandwidth plus the commit kernel's launch/serialization. The 2.47 ms commit is a latency/launch problem first, a bytes problem second.

## FALSIFIER SPEC v2 (corrections to the planned microbench)

1. **T-sweep 1..8, not T=6 alone.** Expect dispatch step-functions at T=4 (BF16 kernel gate `T∈{1..4}`) and the historical ilp=1 fallback at T=3→4 (~33%). Do not fit one linear model across the boundary.
2. **Per-slot arm uses the FlashInfer layout**: `intermediate_states_buffer [pool, T_cache, HV, V, K]` + `state_indices` pool indirection; run **bf16 and fp32** variants. Our fp32×T=12 MB/layer figure is the ceiling; the shipped form is bf16 → halve.
3. **Report launch count and kernel time, not projected bytes.** At b≤8 the path is latency-bound (6.7% bw at b1); the commit win shows as fewer launches/less serialization, not bandwidth relief.
4. **Add the scatter-free arm**: reserve T+1 contiguous pool slots per request (FLA-style indices), checkpoints written in-kernel, commit = free-list update. Falsifier: does commit collapse from 2.47 ms to <50 µs? (SGLang D1.2 predicts yes; their scatter was 948 µs.)
5. **Exact-fold arm must clone the recurrent kernel bit-for-bit** (division-form L2 norm, eps inside sqrt, decay→delta→rank-1 order, num_warps=1) and gate at max_abs ≤ 1e-7 vs the recurrent update; the UT/tf32 variant gates at the bf16 cast floor (~4e-3). Tolerance ladder: fp32 atol 1e-4, bf16 atol 2e-2 (#28511).
6. **Stash post-conv k/v/a/b** for the suffix arm (#26520 lesson), not just g/β. Ring ≥ 2×T → 16, power-of-two; invariant `write_pos + spec_len ≤ ring_len`.
7. **Cursor-hygiene check**: verify-path commits must not advance the decode ring's `write_pos` (b53102c bug → repetition loops at bs=8). Add an assertion counter of foreign writes.
8. **Batched-copy arm**: pitched `cudaMemcpy2DAsync` over `{numLayers, numBlocks, kvFactor, blockSize}` vs per-layer loop; nsys-count memcpy+synchronize pairs per iteration (TRT-LLM baseline: `numGenReq×(1+max_draft_len)` pairs) — target zero syncs in the commit path.
9. **Acceptance stays device-side**: `num_accepted_tokens_gpu` pattern (`[num_reqs]`, fused kernel reads it, in-kernel snapshot, non_blocking copy-back). Any host sync in our harness invalidates the comparison.
10. **VRAM gate**: fp32-A at batch 6 cannot fit beside 28-30 GB residency; run A-fp32 only at batch ≤ 2 or accept an OOM boundary as a finding. Log pool-avail delta (#28695 style: 17.48 → 28.96 GB).
11. **No external numbers to lean on**: FlashInfer's public bench has zero gdn results; every us figure in our report must be self-generated on the 5090.

## ALIGNMENT — does FlashInfer's in-kernel-checkpoint form re-rank P1a vs P1c?

**Yes — it changes the first port, not the ceiling.**

- **P1c (per-slot promote) is now upstream-shaped**: FlashInfer ships exactly this (in-kernel per-token checkpoints + pool indices), and SGLang D1.2 shows the commit kernel itself can be deleted (pool-reserved T+1 slots → free-list update; their 948 µs scatter → ~0). Our 2.47 ms replay is the *naive* P1c (separate commit pass over copy-spec); the falsifier should first test whether adopting the FlashInfer form kills it with near-zero new code. Ceiling ~+5% e2e (D1-class).
- **P1a (chunked suffix) keeps the higher ceiling** (+7.1 to +14.8% e2e per #27725; ~3× D1) and is the only design that fits our VRAM headroom at batch 6 fp32 — but its un-fused production attempt (#26520) measured **+4.6% TPOT**; the win requires the fused recovery+output stage (2.4) and next-forward fusion (2.5b). Nobody has shipped that; first-mover upside is real.
- **Decision rule for the row**: (1) run FALSIFIER v2 arm 4 (pool-reserved FlashInfer-form P1c) — cheapest kill of the 2.47 ms lever, reuses merged upstream kernels; (2) in parallel, prototype P1a's recovery kernel for the bf16-stash geometry — if fused recovery lands under the P1c commit+verify-write total, promote P1a. (3) If VRAM headroom < ~1.7 GiB (bf16-A impossible), invert the order: compact-ring (B, ~30-60 MB) or P1a are the only fits, and ReplaySSM's flag-gated ring is the port target (`--enable-gdn-replayssm-spec` semantics, exact-fold clone).
- "b12x" from round 1 should be relabeled: SM12x NVFP4 **MoE** direct-micro gate (#5561), not a GDN verify kernel.

### Sources
[FlashInfer #2687](https://github.com/flashinfer-ai/flashinfer/issues/2687) · [FlashInfer #5561](https://github.com/flashinfer-ai/flashinfer/issues/5561) · [KernelWiki PR-2679](https://github.com/mit-han-lab/KernelWiki/blob/master/sources/prs/flashinfer/PR-2679.md) · [bench.flashinfer.ai gdn_decode](https://bench.flashinfer.ai/kernels/gdn_decode_qk16_v32_d128_k_last?p=0.95) · [SGLang #28511](https://github.com/sgl-project/sglang/issues/28511) · [SGLang #28695](https://github.com/sgl-project/sglang/issues/28695) · [SGLang PR #27658](https://github.com/sgl-project/sglang/pull/27658) · [SGLang #28730](https://github.com/sgl-project/sglang/issues/28730) · [SGLang #27725](https://github.com/sgl-project/sglang/issues/27725) · [SGLang PR #26520](https://github.com/sgl-project/sglang/pull/26520) · [arXiv:2610.13778](https://arxiv.org/html/2610.13778) · [TRT-LLM PR #14003](https://github.com/NVIDIA/TensorRT-LLM/issues/14003) · [TRT-LLM #15120](https://github.com/NVIDIA/TensorRT-LLM/issues/15120) · [vLLM mamba_utils docs](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/mamba_utils/) · [vLLM #56736](https://github.com/vllm-project/vllm/issues/56736) · [vLLM gpu_model_runner v0.18.0](https://docs.vllm.ai/en/v0.18.0/api/vllm/v1/worker/gpu_model_runner/)
