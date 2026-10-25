# L1 COMMIT KILL — scout digest (code-level blueprints)
Row: GDN commit path replays accepted spec-dec tokens through a row-serial scan (2.47 ms/row). Candidate fixes: (A) chunked-suffix commit (one WY chunk), (B) per-draft-slot states + pointer promotion. Sources studied at the CODE level: SGLang (both designs, shipped), fla-org kernels, FlashInfer b12x gdn_decode, vLLM Qwen3-Next hybrid, TensorRT-LLM MTP SSM cache, KVBuffer/SpecLA. Date: 2026-10-25.

---

## 1. SGLang — TWO shipped designs (best comparable; both generations in tree)

**Legacy per-draft-slot promotion** (`python/sglang/srt/mem_cache/memory_pool.py`, class `MambaPool`, ~line 504):
- Persistent state: `MambaPool.State` = `conv: List[torch.Tensor]` + `temporal: torch.Tensor` `[num_layers, size, HV, V, K]` fp32/bf16, slot = 1 per request (plus +1 spec slot). Allocator: `python/sglang/srt/mem_cache/allocator/mamba.py` — `MambaSlotAllocator`, a plain free-list, "fixed-size slots (1 per request) rather than paged token KV indices".
- Per-draft-token scratch: `SpeculativeState.intermediate_ssm` shape **`[num_mamba_layers, spec_state_size + 1, speculative_num_draft_tokens, HV, V, K]`** — a FULL state snapshot per draft step; `intermediate_conv_window` `[layers, size+1, draft, dim, K-1]` (dedup'd overlapping `as_strided` view on CUDA: D*(K-1) -> D+K-2 columns, halves conv footprint).
- The commit = **`scatter_mamba_states_after_mtp_verify`** (`python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py:781`): one fused Triton kernel `fused_mamba_state_scatter_with_mask` copies `intermediate_state_cache[req, last_correct_step_indices[req]] -> ssm_states[state_indices_tensor[req]]` for ALL layers at once; conv windows via `fused_conv_window_scatter_with_mask` (layer-batched `_conv_multi` variant). No recurrence, no host loop. That is our `gdn_state_promote` generalized to (layers × slots) in a single launch.
- Measured/quoted: memory_pool.py comment — the intermediate_ssm buffer is "the dominant spec scratch, ~46x the conv state"; "ReplaySSM skips this dominant scratch (~9GB @ K3 dspark γ=7)". Issue [#28730](https://github.com/sgl-project/sglang/issues/28730): ~36 linear/GDN layers per draft multiply this.

**New: ReplaySSM circular ring** (`--enable-linear-replayssm-spec`; kernels `python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py`, backend `python/sglang/srt/layers/attention/linear/gdn_backend.py`):
- Per-slot ring buffers instead of per-draft full states: `State.replayssm_d` `[L, slots, HV, Lcache, V]`, `replayssm_k` `[L, slots, H, Lcache, K]`, `replayssm_g` `[L, slots, HV, Lcache]` fp32 (compact per-token records, NOT state snapshots). Verify only writes ring records + per-request fp32 `checkpoint_state` `[num_slots, HV, V, K]`.
- Commit #1 (accept-only, near-free): `commit_gdn_replayssm_spec` -> `_advance_gdn_spec_cursors_kernel[(1,)]` — ONE tiny kernel advances `write_pos`/`cache_base` cursors per row using device-side `num_accepted` (includes bonus token). "Advance request-keyed cursors once per verify step (device-only)." Commit cost ≈ cursor arithmetic; promotion is implicit (checkpoint stays, history ring grows).
- Commit #2 (checkpoint fold = our chunked-suffix commit): `commit_gdn_replayssm_circular` -> `gdn_replayssm_compact_commit_kernel` grid `(cdiv(V,BV), B, HV*num_layers)`: each program loads `h0` (checkpoint), the whole accepted ring window (`o_t = tl.arange(0, MAX_CACHE_LEN)` over the ring, `phys = (base + o_t) & (MAX_CACHE_LEN - 1)`), and **materializes the new checkpoint directly from compact D/K/G in one parallel pass** — parallel over v-tiles/heads/layers, never serial over tokens across programs. Optional "track rows" snapshot any interior accepted step (for EAGLE resume / MTP draft state) without disturbing the active checkpoint.
- Measured (PyTorch blog "Hybrid Models Meet SGLang", Qwen3-Next-80B-A3B-FP8, H200, bs=1): 257.2 tok/s (MTP 2), 306.94 (MTP 3), 324.57 (MTP 4, topk 4, 8 draft tokens); accept lengths 2.709/3.413/4.231. Blog: "when a sequence of draft tokens is accepted, simply promote the last accepted slot to become the new main state."

**Grade: A.** Both of our candidate designs exist in production code with exact shapes; ReplaySSM additionally solves the memory blowup.

## 2. flash-linear-attention (fla-org) — the reusable math

- `fla/ops/gated_delta_rule/fused_recurrent.py`: `fused_recurrent_gated_delta_rule(q[B,T,H,K], k, v[B,T,HV,V], g, beta, initial_state[N,HV,K,V], output_final_state, state_v_first, cu_seqlens, use_gate_in_kernel, A_log, dt_bias, ...)` — the Triton kernel `fused_recurrent_gated_delta_rule_fwd_kernel` is **already varlen**: `pid` over `(n_seq, hv, v-tile)`, `for _ in tl.range(0, T)` serial inside one program, state `b_h` in registers, `h0`/`ht` pools indexed by `i_nh*K*V`. Commit use: feed ONLY the accepted suffix (T = n_keep, cu_seqlens = per-row offsets) with `initial_state` = committed pool, `output_final_state` = true. The row-serial chain becomes one launch parallel over rows/heads/tiles — never a host loop over rows.
- `fla/ops/gated_delta_rule/chunk.py` -> `chunk_gated_delta_rule_fwd`: `chunk_local_cumsum` -> `chunk_gated_delta_rule_fwd_intra` (fused kkt + solve_tril + recompute_w_u = the WY step) -> `chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state, output_final_state, state_v_first, chunk_size=64)` -> `chunk_fwd_o`. The one-WY-chunk state fold is exactly `fwd_h`: `W_out = exp(sigma) W_in + sum_s k_s exp(sigma-c_s) delta_s^T` (see our own `include/strata/prefill/gdn_chunk.hpp:32-35` — same formula). For n_keep <= 64 the accepted suffix IS one chunk, so chunked-suffix commit = 3 small launches (cumsum, intra, fwd_h) per layer-group, or fuse to 1.
- State layout: `[N, HV, K, V]` (k-major) with `state_v_first=True` flag for `[N, HV, V, K]` — mirrors FlashInfer's k-last option.

**Grade: B+ (library, not a serving commit path).** Direct port target for option A; our sm_70 `gdn_chunk.cu` already implements the same math.

## 3. FlashInfer — b12x `gdn_decode` (state-pointer decode + implicit promote)

Path: `flashinfer/experimental/b12x/sequence/gdn_decode/` (`api.py`, `_impl.py`, `_kernels.py`, `_preparation.py`, `_commit.py`, `reference.py`).
- `Binding` fields: `recurrent_state` (the pool), `query_start_loc`, **`num_accepted_tokens`** (device), **`state_indices: [sequence_capacity, column_capacity]`** (2-D: per request, per draft column), `output`. `run()` docstring: "Its initial recurrent checkpoint comes from `state_indices[r, num_accepted_tokens[r] - 1]`. Each token executes sequentially and writes its post-token checkpoint to the corresponding state-index column. A one-column plan with one token per request is ordinary decode." I.e. **the verify kernel itself maintains per-draft-slot states and the commit is pointer selection (`accepted-1`) — no copy at all on the hot path.**
- Kernel addressing (`_kernels.py`): `source_idx * stride_state_slot + value_head * stride_state_head + value_rows * stride_state_v` — state as `[slots, HV, V, K]` with `stride_indices_request/column` 2-D index table; `reference.py` confirms `state = recurrent_state[source_slot, value_head]` then per token `recurrent_state[destination_slot, value_head].copy_(state)`.
- `_commit.py` (`KdaCommitBinding` / `bind_kda_commit` / `run_kda_commit` "Recover accepted FP32 checkpoints without re-evaluating the model") is the KDA variant: takes `state_base_addrs`, `state_block_strides` (**raw byte-address tables**), `commit_lens`, `final_state_indices`, `boundary_state_indices`, `boundary_recovery_lens` — a device-side recovery kernel that reconstructs accepted checkpoints from per-step caches with zero host syncs. Note their aliasing contract: "If boundary and final destinations alias, the final state takes precedence."
- Benches: `benchmarks/bench_gdn_decode.py`, `bench_gdn_fused_decode.py`, `bench_gdn_prefix_materialize.py` (materialize = fold-commit benchmark), `bench_gdn_ucache_flush.py`.

**Grade: A-.** The cleanest "state pointers, not state copies" design; commit is an index-tensor update.

## 4. vLLM — Qwen3-Next hybrid (copy-spec promote, now + ReplaySSM ring)

- Shapes: `vllm/model_executor/layers/mamba/mamba_utils.py` — `MambaStateShapeCalculator.gated_delta_net_state_shape`: conv `[(conv_dim)/tp, kernel-1 + num_spec]` (**spec extends the conv window by num_spec columns**), temporal `(HV/tp, V, K)`. New `append_replayssm_ring` adds ring shapes `(nheads, logical_window [+1+num_spec for FlashInfer backend], head_dim)` etc. — vLLM is adopting the ReplaySSM design too.
- The commit: `get_temporal_copy_spec(state, block_ids, cur_block_idx, num_accepted_tokens)` -> `src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]`; returns `MambaCopySpec{start_addr=data_ptr(), num_elements}` — **a raw-pointer async memcpy of the winning per-draft-slot state into the request's base slot**. Conv: `get_conv_copy_spec` slices `state[src_block_id, offset:]` (SD layout) or defers to a fused postprocess kernel (DS layout, offset>0). Wired per model via `MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func() -> (get_conv_copy_spec, get_temporal_copy_spec)`; invoked from `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` (`MambaHybridModelState.postprocess_state`, with `num_accepted_tokens_gpu` kept device-side and small `_scatter_num_accepted_kernel`/`_fill_num_accepted_kernel` fixups; `num_accepted_tokens=1` treated as neutral non-spec).
- Slot sizing bug class: issue [#10693](https://github.com/vllm-project/vllm/issues/10693) (manager runs out of free slots after multistep scheduling — must reserve draft-token slots), [#38898](https://github.com/vllm-project/vllm/issues/38898) (SD vs DS conv layout), [#41884](https://github.com/vllm-project/vllm/issues/41884) (IndexError in `get_temporal_copy_spec`).

**Grade: B+.** Simplest production commit (pointer + memcpy), good evidence of the layout pitfalls.

## 5. TensorRT-LLM — shipped CUDA commit kernel + a perf war story

- Issue [#15120](https://github.com/NVIDIA/TensorRT-LLM/issues/15120) (fixed by PR #15020): AutoDeploy `MambaHybridCacheManager` binds "intermediate SSM/conv state and the replay SSM kernel buffers (Replay* handlers)" — spec-only buffers must be dropped (None sentinel) when spec is off; `ssm_replay=True` without spec was a silent no-op. Confirms TRT-LLM ships BOTH: per-step intermediate caches AND a replay-commit path.
- The shipped CUDA op: `cpp/tensorrt_llm/kernels/mamba2MTPSSMCache/` (`mamba2MTPSSMCache.cu/.h`, `mamba2MTPSSMCacheVec4/8/16.cu`, python binding `thop/mamba2MTPSSMCacheOp.cpp`). `Mamba2MTPSSMCacheParams`: `ssm [batch, nheads, head_dim, ssm_dim]`, `intermediate_states [batch, cache_steps, nheads, head_dim, ssm_dim]`, `intermediate_states_indices [bs]`, **`retrieve_parent_token [bs, cache_steps]`** — verify writes per-step states; the cache op folds/commits via parent pointers (EAGLE-tree safe), with vectorized 4/8/16-wide variants. Tests: `tests/unittest/_torch/modules/mamba/test_mamba2_mtp_ssm_cache.py`.
- Measured (their internal perf-cookbook case `mamba-hybrid-recurrent-state-d2h-syncs.md`, PR #14003): a regression where prep did per-element reads of a CUDA `state_indices` tensor -> `numGenReq × (1 + max_draft_len)` `cudaMemcpyAsync` + `cudaStreamSynchronize` pairs PER ITERATION; fix = alias the CUDA tensor (assert `data_ptr()` stability for graph replay), single pitched `cudaMemcpy2DAsync` across layers, defer `refresh_blocks()` until a transfer is actually scheduled. Lesson for us: **the commit must be device-only and CUDA-graph-capturable; any host read of accepted-length in commit reintroduces a per-row sync.**

**Grade: A- (native CUDA, closest to our C++/CUDA engine).**

## 6. KVBuffer / SpecLA

KVBuffer ("KVBuffer: IO-aware Serving for Linear Attention", [arXiv 2605.19049](https://arxiv.org/html/2605.19049v1)): linear-attention+spec-dec shared KV buffer; claims up to 45.17% linear-attn decode latency reduction and 5x more served requests under spec-dec. **No public author repo found** (searched GitHub); design-only. SpecLA likewise: no released code surfaced. Grade: C (paper claims, no code to port).

---

## PORT MAP (our sites -> their analog)

| Our site | What it does | Closest analog |
|---|---|---|
| `src/core/verify.cpp:1615-1618` (F-216 snapshot: writes `gdn_snap_` per column, layer-major `[gdn_layers × max_t × gdn_rec_floats_]` fp32) | per-draft-token full-state snapshot during window | SGLang `MambaPool.SpeculativeState.intermediate_ssm` `[layers, size+1, draft, HV, V, K]` (memory_pool.py:882) — same idea, but they allocate it in the POOL and skip it entirely under ReplaySSM |
| `src/core/verify.cpp:2664-2674` `gdn_commit_layers(...)` (STRATA_COMMIT_BATCH batched replay) | batched row-serial replay of accepted tokens, all layers 2 launches | TRT-LLM `Mamba2MTPSSMCache` kernels — replay/fold done inside one CUDA op with `retrieve_parent_token` pointers, not serial re-walk; and SGLang `gdn_replayssm_compact_commit_kernel` which folds from compact D/K/G records in a parallel (V-tile × B × HV×L) grid |
| `src/core/verify.cpp:2732-2735` `gdn_state_promote(state, snap + (gdn_index-ord0)*max_t*gdn_rec_floats_, ...)` per-layer host loop | promote snapshot[m-1] instead of replay | SGLang `scatter_mamba_states_after_mtp_verify` -> `fused_mamba_state_scatter_with_mask` (ONE launch, all layers+slots, step selected by `last_correct_step_indices`); vLLM `get_temporal_copy_spec` (pointer + memcpy); FlashInfer: no copy at all — `state_indices[r, accepted-1]` pointer selection |
| `src/core/verify.cpp:2741-2744` `gdn_snap_audit(...)` | bit-parity audit of promote vs replay | SGLang `fold_every_commit` / exact-fold path: "sequentially replay the committed ring window ... bit-identical to the recurrent baseline (closed loop)" |
| `include/strata/core/verify.hpp:380-391` `gdn_snap_` arena, `gdn_rec_floats_ = ssm_state_size² × v_heads` | snapshot arena bytes (= layers × max_t × ~2 MiB for K=128, HV=32) | SGLang replayssm ring `[L, slots, HV, Lcache, V/K]` — records O(token×V) not O(V×K), the ~9 GB -> small win from issue #28730 |
| `include/strata/kernels/verify_kernels.hpp:66-70` + `src/kernels/cuda/verify_kernels.cu:1278+` | `gdn_state_promote` / `gdn_snap_audit` kernels | TRT-LLM `mamba2MTPSSMCacheVec16.cu` vectorized fold; their d2h-syncs case file for graph-safety invariants |
| `include/strata/prefill/gdn_chunk.hpp` + `src/prefill/gdn_chunk.cu` (cumsum -> kkt -> fwd, 64-token chunks; `W_out = exp(sigma) W_in + Σ k_s exp(sigma-c_s) delta_s^T` at lines 32-35) | chunked GDN for prefill | fla `chunk_gated_delta_rule_fwd_h` — same formula; commit reuse = feed suffix (1 chunk) with `initial_state=state, output_final_state=true`; SGLang `gdn_replayssm_compact_commit_kernel` is its spec-commit incarnation |
| `h_commit_` host-mapped commit tensor (verify.cpp:923, `copy_i32_from_mapped(commit_, ...)`) | host reads n_keep before launching commit | FlashInfer `commit_lens`/`num_accepted_tokens` device metadata ("no CPU synchronization is performed"); TRT-LLM PR #14003 lesson — this is our sync hazard |

## OUR NEXT STEP — the microbench falsifier

Build `campaign/gdn-commit-falsifier.cu` (pattern of `campaign/mma-accum-ratelock.cu`), model geometry: GDN layers L ∈ {36, 48}, HV=32, K=V=128 (state 2 MiB/layer fp32), conv C=8192×4B. Four commits of the SAME accepted prefix, n ∈ {1..8} tokens, B ∈ {1, 4, 8} rows, timed by cudaEvents on `cs_`, 200 reps after 20 warmup, all launched from a captured CUDA graph (commit must replay):

- (S0) baseline: current batched replay `gdn_commit_layers` (the 2.47 ms/row path).
- (S1) promote: `gdn_state_promote` per layer (current F-216) — measures pure-copy floor (should be ~2·arena_bytes / ~3 TB/s ≈ µs-scale; if S1 is also slow, the scan was never the bottleneck).
- (S2) fused scatter-promote: ONE kernel grid (L × HV × V-tiles × B), reading `gdn_snap_[l, n_keep-1]` with `n_keep` read from device `commit_` (no host sync) — SGLang `fused_mamba_state_scatter_with_mask` shape.
- (S3) chunked-suffix commit: reuse `gdn_chunk.cu` on the n-token suffix with `W_in` = current state, out-of-place `W_out`, then copy-back (or in-place if we accept the clobber-on-accept semantics with a pre-copy of W_in for rejection reuse); parity vs S0 at fp32 tolerance (chunk math is FP32-level, not bitwise — verify against `gdn_snap_audit` harness).

Falsifier conditions (record with the run):
1. If S1 ≥ ~50 µs at B=1 (i.e. not bandwidth-explained), reject "commit is copy-bound" — profile first; the hypothesis space collapses to launch-count/host-sync overhead.
2. If S2 ≤ 10 µs and ≥20x faster than S0 at every (n, B), promote wins and the chunk path is only justified if it ALSO kills the `gdn_snap_` arena (max_t × 2 MiB × L ≈ 768 MiB at T=8, L=48). Acceptance: S3 must hold S3_time < S0_time/10 AND arena_bytes == 0 to displace S2 as default; otherwise S2 ships (STRATA_GDN_SLOTS=1 stays, batched fuse via S2).
3. Graph-capture gate: any variant that forces a host read of n_keep or adds per-row cudaMemcpyAsync+sync (TRT-LLM #14003 signature) is auto-rejected regardless of isolated time.
4. Memory gate (SGLang #28730): per-draft-slot pools scale slot_count × 72 MiB (L=36 fp32); with slots ≥ 8 the pointer-ring design (ReplaySSM-style compact records + fold) is the only memory-viable route — record arena/pool bytes for all variants.

Deliverable of the bench: a table time(n, B, variant) + graph-node count + bytes; the winning variant's kernel is then wired into `commit()`/`commit_slots()` behind `STRATA_GDN_SLOTS=3` (fused promote) or `=4` (chunk fold).
