---
type: research
date: 2026-11-14
topic: basetenlabs/deepseek-ultrafast — Rust batch-1 inference engine for DeepSeek-V4.1-Flash (the exact model class we serve); kernel, quant, spec-decode, and KV-compression techniques priced against our single-GPU + host-RAM MoE offload engine (strata-custom-2)
machine-context: AMD 9950X3D, RTX 5090 32 GB (sm_120), 184 GiB RAM, ~50 GB/s DRAM lane. Router+dense+KV in VRAM, expert blobs streamed per-expert from host RAM. Vendor llama.cpp kernels. DSpark + MTP drafter machinery. Serves Qwen3.8-Flash-Next 748B-A16B and DeepSeek-V4.1-Flash 1017B-A17B. Their engine: 4-8x datacenter Blackwell (sm_100a/sm_103a), 0.7-1.4 TB aggregate VRAM, single-request latency design.
status: verified — repo shallow-cloned and grepped locally (4 commits, /tmp clone); every claim carries a repo path or source URL; hardware-compat claims cross-checked against PTX ISA docs and FlashInfer issue tracker
sources: github.com/basetenlabs/deepseek-ultrafast (repo + API metadata, fetched), baseten.co optimization blog (fetched), PTX ISA docs + FlashInfer #2649 (search-verified for sm_120 compat)
---

# deepseek-ultrafast — value assessment for strata-custom-2 (2026-11-14)

Framing: this engine serves **DeepSeek-V4.1-Flash 1017B-A17B — our served model** — at batch-1 latency extremes on hardware we will never own. The transfer question is not "can we run it" (we cannot: sm_100a/sm_103a only) but "which of its mechanisms upgrade our decode lane." Three of our queued items (FP8 KV, conditional CUDA graphs, dspark drafter tuning) have shipped reference implementations here.

## 1. Identity

- **What.** Rust inference engine for DeepSeek-V4.1-Flash, `sm_100a`/`sm_103a` only, 1M-token context, OpenAI-compatible server, multi-server KV sharing. [README](https://github.com/basetenlabs/deepseek-ultrafast)
- **Why.** Baseten's hosted "DeepSeek-V4.1-Flash-Fast" product; kernels were "optimized by coding agents that could read the source of FlashInfer and SGLang" ([README acknowledgments](https://github.com/basetenlabs/deepseek-ultrafast); [optimization blog](https://www.baseten.co/blog/how-to-make-flash-models-even-faster-optimizing-deepseek-v41-flash-1017-fast)).
- **License.** Elastic License 2.0 (`LICENSE` at repo root; GitHub API reports `NOASSERTION`/Other). Internal use, copying, derivative works allowed; **cannot be offered as a hosted/managed service**. Fine for fork-and-study; not for resale.
- **Activity.** Created 2026-10-27, last push 2026-11-14, **4 commits total**, 172 stars, 5 forks, 0 open issues ([GitHub API](https://api.github.com/repos/basetenlabs/deepseek-ultrafast)). Young, single-drop release, no community process yet — treat as a snapshot, not a moving upstream.
- **Layout.** Rust workspace: `crates/{cuda,kernels,model,engine,server,kv-transfer,logging}`; 62 `.rs` files, 40 `.cu/.cuh`; kernels compiled to cubins via `crates/kernels/build.rs` and embedded with `include_bytes!`. `variant_*.cu` files compile **only when `DSUF_SM=sm_103a`** (build.rs lines 44-46) — B200 (`sm_100a`) builds use the base kernels.
- **Measured performance** ([README](https://github.com/basetenlabs/deepseek-ultrafast), repro script included):
  - Decode 1,740 tok/s on 4x GB300 TP4 / **2,060 tok/s on 8x B300 TP8**; step time 3.30 / 2.71 ms.
  - **Acceptance length 5.6 tokens/step** (ranges 5.1-6.1 and 5.3-6.1 over 30 runs).
  - Prefill 1M tokens in 13.2 s / 11.7 s (76k / 85k tok/s).
  - Blog: **991 tok/s** on Artificial Analysis with lowest time-per-task (Pareto frontier); fused MoE router kernel cut decode by **2.7 ms**; mHC kernel **+30%** at batch size 1 ([blog](https://www.baseten.co/blog/how-to-make-flash-models-even-faster-optimizing-deepseek-v41-flash-1017-fast)).
- **Serving shape.** One request at a time per server; multi-turn requires a session ID ([docs/SERVER.md](https://github.com/basetenlabs/deepseek-ultrafast/blob/main/docs/SERVER.md)). This is a batch-1 latency engine, our exact regime — not a throughput batcher.

## 2. Techniques inventory (file-anchored)

### 2.1 Attention — MLA + sparse indexer, bf16 ldmatrix core

- `crates/kernels/cu/attn.cu` (1,463 lines): MLA decode attention in **bf16 with `ldmatrix`** shared-memory loads (`.x4/.x2/.trans`, attn.cu:10-24) — sm_75+ instructions, **no tcgen05 dependency**. Struct `Meta { pos, m, tokens[16] }` (attn.cu:6-10) — decode steps carry up to 16 draft tokens.
- Full **DSA sparse-attention stack**: `idx_score` (lightning-indexer scores over quantized K candidates, attn.cu:44), `block_max` (attn.cu:130), `topk_select` (attn.cu:241), `compress_pool` (attn.cu:254). Candidate bitmap mask (`cand`/`ld_cand`/`cand_block`, attn.cu:39-40, applied at :122). Tensor-core indexer variants in `tcgemm.cu:540` (`idx_score_tc4b`) and `:696` (`idx_score_tc_dec`).
- PDL (`griddepcontrol.wait`/`launch_dependents`) declared in `common.cuh:14-15` and used across the attention/sampling chain (`attn.cu:46-47`, `sample.cu:822-823`, `tcgemv.cu:35-36,88-89`).

### 2.2 MoE — batch-1 fused slot-plan kernels (the portable gem) + tcgen05 variants

- `crates/kernels/cu/route.cuh:5-57`: single-warp top-k routing — `sqrt_softplus` scores + bias, warp-argmax loop with `__reduce_max_sync` (sm_80+), normalized weights x `route_scale`. Zero arch-specific hardware; ~57 lines.
- `crates/kernels/cu/gemv.cu` (1,907 lines, the master file): `MoePlan` struct (gemv.cu:14-21) — device-side dispatch plan of expert slots with token lists, token-expert pair indices, and routing weights. Fused batch-1 kernels: `moe_fp4_fused_n1` (gemv.cu:1693), `moe_fp4_fused_gate_n1` (:1696, fuses the router gate), `moe_w2_n1/n2` (:1904-1905). Inner loops are **`mma.sync.m16n8k16` bf16/f16 + ldmatrix** (gemv.cu:364-421, 932-947) — portable instruction set, no tcgen05.
- `crates/engine/src/ops.rs:946` (`moe_fused`): the fused chain routed→gather→w13→SwiGLU→w2→scatter as one launch with PDL flag and fused gate, invoked at `crates/engine/src/model.rs:3591` (timed section "moe fused"). Blog prices this fusion class at **−2.7 ms/decode step**.
- Prefill MoE: tile-list grouped GEMM — `PrefillMoe` with `TM_MAX_TILES` cap (model.rs:859), tcgen05-layout execution `tc_moe_w13_rm2`/`tc_moe_w2_rm` (model.rs:1109), `tcmoe.cu:19,86,585-586` (`moe_perm`, `moe_gather_quant`, `tc_moe_w13/w2`).
- Weight GEMM for prefill: `tc_gemm_fp8` / `tc_gemm_fp8p` persistent (tcgemm.cu:20,141), TMA tensor maps created **outside graph capture** (model.rs:1349,1441,1455,1493) — an operational gotcha they encode as hard errors.
- Dual-path dispatch is explicit in error strings (model.rs:2455-2460): tcgen05 decode GEMVs serve ≤8 rows; larger steps fall back to mma.sync GEMVs with different numerics. The repo therefore ships **portable mma.sync paths for everything** alongside the datacenter-only paths.

### 2.3 Quantization — FP8 block weights, FP4 KV offload

- `crates/model/src/formats.rs`: reference Rust FP8 quantizer for checkpoint conversion — `f32_to_e4m3_fast` RTNE bit-trick (formats.rs:42-60), **pow2 scales with floor 1/448, 32x32 blocks** (`quant_fp8_b32`, formats.rs:63+). DeepSeek FP8 block format, exactly the scheme our vendored kernels expect to consume.
- `crates/kernels/cu/quantblocks.cuh:5-58`: device-side FP8 block quant fused into GEMV prologues — hardware `f32x2_to_e4m3x2`/`e4m3x2_to_f16x2` converts, `round_scale_pow2`, scale applied at dequant via `dqmul`.
- `crates/engine/src/fp4.rs` + `crates/kernels/cu/kv_fp4.cu`: **KV cache FP4 pack/unpack** ("offload_pack", kv_fp4.cu:4-43). Two encodings ([`Encoding` enum, kv-transfer/src/encoding.rs:7-21](https://github.com/basetenlabs/deepseek-ultrafast/blob/main/crates/kv-transfer/src/encoding.rs)):
  - `Fp4E4m3`, group=16: scale = amax/6 rounded into an **E4M3 byte**; values E2M1 via `cvt.rn.satfinite.e2m1x2.f32` (kv_fp4.cu:33).
  - `Fp4E8m0`, group=32: **pow2 E8M0 scale byte** (kv_fp4.cu:38-39).
  - Stored size: `decoded_bytes/4 + decoded_bytes/(2*group)` (encoding.rs `stored_bytes`) ≈ **4.06 bits/value** (vs 16 for bf16), plus a validity bitmask (`atomicAnd`, kv_fp4.cu:41) so a corrupt quant fails the snapshot instead of poisoning the cache.
- `crates/engine/src/kvstore.rs`: layer-segmented KV snapshots with per-layer `compress_ratios`, export slot with pinned staging (8 MiB chunks) + async stream + event (kvstore.rs:22-33), budgeted by `crates/server/src/host_budget.rs` ("no available host memory for offload"). This is their host-RAM KV residency layer — same wall as ours, different ratio (their host copy is for replica fetch, ours is for capacity).

### 2.4 Speculative decoding — DSpark-class Markov drafter + MTP head, all inside one CUDA graph

- `crates/kernels/cu/draft_markov.cu` + `ops.cu:318` (`markov_rounds`/`markov_rounds_ld`): **rank-256 Markov drafter** over a compressed-vocab state (bf16, k=256 enforced at model.rs:3852), hash-conditioned (`hcond`), multi-round with LSE merges (`mk_lse`, draft_markov.cu:27-31) and confidence `theta`. Config knobs `dspark_markov_rank`, `dspark_block_size`, `dspark_noise_token_id` (model.rs:3952-3966) — **they call their drafter DSpark**, same lineage as ours.
- `crates/kernels/cu/draft_head8.cu`: 8-row MTP-style draft head — producer/consumer warp specialization with **TMA 1D ring buffer + mbarrier full/empty phases** (draft_head8.cu:25-58); "sgl mtp" wiring at model.rs:3818.
- Verify: `spec_draft_meta`/`spec_verify_meta`/`spec_accept` (ops.cu:572-614); draft block capped at 7 (8 verify rows incl. the live token) on tensor-core GEMV paths (model.rs:2455-2460); committed-vs-emitted bookkeeping in `crates/engine/src/decode_cache.rs`.
- **Adaptive spec CUDA graph**: `capture_spec` (model.rs:3936) captures a graph with N top-level nodes and per-block-length verify bodies ("captured adaptive spec graph: {n} top-level nodes, {blk} verify bodies", model.rs:4042); profile-tuned block sizes 6-7 (`crates/cuda/src/profile.rs` `dspark_block()`).
- **Measured: 5.6 tokens/step acceptance** (README) — a real number for a DSpark-class drafter on V4.1-Flash, directly comparable to our DSpark telemetry.

### 2.5 CUDA graphs & the host loop

- Whole decode step = **one captured graph**, launched repeatedly (server/engine.rs:247 `capture_spec`, :333 `graph.launch`).
- **Double-buffered device-side snapshots**: `spec_snapshot(launched % 2)` then reading step N-1's commit count while step N runs (server/engine.rs:333-346) — zero per-step host sync. Our engine B currently pays 3 host syncs/round (2026-11-04 digest, machine-context); this is the reference elimination pattern.
- Host scheduler is pure Rust (cudarc driver API, dynamic-loading — `Cargo.toml` workspace deps). Blog names the Python scheduler in SGLang as the small-batch bottleneck they removed; also credits NVIDIA's expanded PDL for part of the 2.8x Blackwell gain they cite ([blog](https://www.baseten.co/blog/how-to-make-flash-models-even-faster-optimizing-deepseek-v41-flash-1017-fast)).
- Prefill: chunked (default `--chunk 8192`), and **"CED bounded SWA replay prefill"** always on — "skips full-prompt decoder work and reconstructs the decoder's 128-row window" ([docs/SERVER.md](https://github.com/basetenlabs/deepseek-ultrafast/blob/main/docs/SERVER.md)). Window-replay is the same class of trick as prefix-cache hits for sliding-window layers.
- Kernel-selection layer: `crates/engine/src/ops.rs` is a capability-gated dispatch table (`hw()`, `variant_cc()`, `tmoe_on()`, `sgl_moe_on()`, `attn_sds_on()`, per-kernel smem fitters like `topk_cand_smem` against the 160 KiB limit at model.rs:1404) backed by `crates/cuda/src/profile.rs` hardware profiles. Clean pattern for our vendor-kernel dispatch.
- Profiling aids: `tools/nsys_breakdown.py`, `tools/nsys_layer_chain.py`.

### 2.6 V4.1 architecture extras (context, not ports)

- mHC (manifold-constrained hyper-connections): `hcpipe.cu`, `hcepi.cuh` — residual-stream mixing fused into the TP-comm epilogue with PDL weight prefetch; blog: **+30% at bs1**.
- Engram (hashed n-gram memory, a V4.1 arch feature): `crates/engine/src/engram.rs` (compressed vocab map, bucket primes, hash multipliers), `ops.cu:77,125,168,244` (`engram_gather/apply/peer`) — needed only to reproduce V4.1 numerics.
- Sampling: `sample.cu` (1,188 lines) with draft-aware verify sampling; decode top-k keeps a 128K-sized smem cache and streams larger rows (SERVER.md).

### 2.7 Multi-server KV sharing (does not transfer; interior does)

- `crates/kv-transfer/`: RDMA via [Velo](https://github.com/ai-dynamo/velo), Redis Lua-script ownership directory, session fingerprints. Every server keeps recent conversations' KV in **host memory**; a turn landing on another replica fetches the FP4-compressed snapshot host-to-host over RDMA per request — "Thanks to DeepSeek's extreme KV cache compression, the transfers are very fast" ([README](https://github.com/basetenlabs/deepseek-ultrafast)). The FP4 snapshot format is the transferable interior; the RDMA/Redis/kubernetes shell is not.

## 3. What transfers to single-GPU + host-RAM offload

1. **FP4 KV pack/unpack + snapshot segmentation** (`kv_fp4.cu`, `fp4.rs`, `kvstore.rs`, `encoding.rs`). The `cvt.rn.satfinite.e2m1x2.f32` instruction **is supported on sm_120** (RTX 5090) per PTX ISA docs; caveat: some sm_100-era PTX is missing on sm_120 and libraries needed `sm_120f` compile targets — documented in [FlashInfer #2649](https://github.com/flashinfer-ai/flashinfer/issues/2649) and the fp4_sm120 patches — so expect a compile-target fix, not a hardware gap. Effect for us: KV host-RAM footprint ~4x down (16-bit → ~4.06 bits/value), same for any PCIe-resident KV tier. This is our queued "FP8 KV (B)" item one generation further compressed.
2. **Batch-1 fused MoE GEMV family** (`gemv.cu` `MoePlan` + `moe_fp4_fused_n1/gate_n1` + `moe_w2_n1/n2`, `route.cuh`, `ops.rs::moe_fused`). mma.sync + ldmatrix core runs on sm_120 after retarget. Study value: their per-slot expert plan (expert → token list → weights in device struct) is the cleanest published analog to our per-expert-streamed decode; compare their persistent-CTA scheduling and fused SwiGLU-in-GEMV against our vendored llama.cpp kernels. Blog-priced fusion win: −2.7 ms/step class.
3. **Spec-decode plumbing around our existing drafters** (`ops.cu:572-614`, `model.rs:3927-4042`, `decode_cache.rs`, `draft_markov.cu`): adaptive per-block-length captured graph, ≤7-token blocks, double-buffered readback with no per-step host sync, and a 5.6 tok/step acceptance datapoint for a DSpark-class drafter on V4.1-Flash. Maps 1:1 onto our DSpark + MTP machinery; replaces our "3 host syncs/round" host loop.
4. **DSA sparse-attention candidate stack** (`attn.cu`: `idx_score`, `block_max`, `topk_select`, `compress_pool`): bf16/ldmatrix, portable; our long-context decode attention gains the indexer-gated candidate path without any datacenter dependency.
5. **FP8 block quant reference** (`formats.rs::quant_fp8_b32`, `quantblocks.cuh`): RTNE e4m3 + pow2 scales + 32x32 blocks, CPU reference plus fused device quant — vendor-grade input for our llama.cpp quant path.
6. **PDL chaining** (`common.cuh:14-15` and call sites): `griddepcontrol` is sm_90+; sm_120 supports it. Cheap launch-overlap win across our attn→moe→sample chain, and a prerequisite if we adopt their graph shape.
7. **SWA window-replay prefill** (concept from SERVER.md): skip full-prompt decoder work, replay only the 128-row window — direct prefill win for any SWA layers in our served models.
8. **Engineering patterns**: compile-time variant gating (`build.rs`), capability-dispatched kernel selection (`ops.rs`/`profile.rs`), tensor-maps-outside-capture discipline (model.rs errors), nsys chain scripts (`tools/`).

## 4. What does NOT transfer (skip; time sink otherwise)

- **All tcgen05/TMEM paths**: `tcgemv.cu`, `tcgemv2.cu`, `tcmoe.cu` decode MoE, `variant_zzzzzzaa.cu` (`tcgen05.mma kind::mxf8f6f4 block_scale`), `variant_zzzzzzzzzlp.cu`. These compile only for sm_103a (`build.rs` variant gate); `sm_100a` cubins do not run on sm_120a ([PTX ISA family-specific rules](https://docs.nvidia.com/cuda/parallel-thread-execution/)). tcgen05 exists on sm_120a in **limited, different form** (FP4 mma requires sm_120a) — so treat these files as concept reading, not port sources.
- **Cluster features** (`variant_zzzzzzaa.cu` `%cluster_ctarank`, `barrier.cluster.*`, `variant_zzzzzzaud.cu`): distributed shared memory / cluster sync is a datacenter-Blackwell pattern; sm_120 cluster support differs. Skip.
- **NVLS / NVLink SHARP** (`setup_nvls`, `comm_nvls_ucpush` in `comm.cu:710`): multicast allreduce needs NVSwitch-class fabrics. We are single-GPU; all of `comm.cu` (custom TP allreduce, 3-stage Lamport epochs, fused sum+hc-post epilogues) is inapplicable — our analog is "no comm at all," which is where our wins already come from.
- **RDMA KV transfer + Redis directory + k8s deploy** (`crates/kv-transfer/{rdma,address,directory}.rs`, `deploy/`): multi-replica infra. Only the FP4 snapshot format inside it transfers (item 3.1).
- **1M-context capacity math**: their 4-8 GPU aggregate VRAM (~0.7-1.4 TB) makes "keep everything resident" the right call; our 32 GB VRAM + streamed experts inverts the constraint. Their kernel choices for full-resident decode still teach, but their memory budgets do not.
- **Engram peer kernels** (`engram_gather_peer`, `engram_apply_peer`): cross-GPU engram propagation. N/A.

## 5. Ranked verdict

**Verdict: HIGH VALUE.** This is the closest published relative of strata-custom-2 we have found: batch-1, single-request latency design, DSpark-lineage drafter, FP4-compressed KV in host RAM, serving *our exact model*. It will not run on our hardware (Elastic 2.0 aside, sm_100a/sm_103a only), but roughly half its mechanism inventory is mma.sync-portable or pure pattern. Ranked port candidates:

| # | Candidate | Source paths | Effort | Expected gain class |
|---|---|---|---|---|
| 1 | FP4 KV pack/unpack + segment snapshots (E4M3-g16 / E8M0-g32 scales, validity bitmask) | `crates/kernels/cu/kv_fp4.cu` (72 lines), `crates/engine/src/fp4.rs`, `kvstore.rs`, `kv-transfer/src/encoding.rs` | **Trivial-small** (kernel is 72 lines; Rust plumbing modest; needs sm_120f cvt fix per FlashInfer #2649) | KV bytes ÷4 in host RAM and any PCIe KV tier — lane-compression on our primary wall |
| 2 | Batch-1 fused MoE GEMV family: MoePlan slot-plan + fused gate/SwiGLU, persistent CTAs | `crates/kernels/cu/gemv.cu` (MoePlan :14-21, moe_fp4_fused_n1 :1693, moe_w2 :1904), `route.cuh`, `ops.rs::moe_fused` | **Small-medium** (retarget formats to our quant scheme; the plan/fusion structure is the import) | Decode TPOT at batch-1; their fusion class priced at −2.7 ms/step |
| 3 | Spec-decode graph plumbing: adaptive per-block graph, ≤7-token blocks, double-buffered no-sync readback | `crates/engine/src/model.rs:3927-4042`, `ops.cu:572-614`, `decode_cache.rs`, `server/src/engine.rs:247-346` | **Small** (wiring onto existing DSpark+MTP; graph capture code is the import) | tok/s × acceptance; their 5.6 tok/step is the benchmark to beat; kills our 3-syncs/round |
| 4 | DSA sparse-attention candidate stack (indexer → block-max → top-k select) | `crates/kernels/cu/attn.cu` (idx_score :44, block_max :130, topk_select :241, compress_pool :254) | **Medium** | Long-context decode attention speedup; portable bf16/ldmatrix |
| 5 | FP8 32x32-block quant reference (RTNE e4m3 + pow2 1/448-floor scales) | `crates/model/src/formats.rs`, `crates/kernels/cu/quantblocks.cuh` | **Trivial** | Quant-pipeline correctness reference for vendored kernels |
| 6 | PDL chaining across attn→moe→sample | `common.cuh:14-15` + call sites | **Trivial** | Small TPOT cut from launch overlap on sm_120 |

Runners-up: SWA window-replay prefill concept (SERVER.md, medium); ops.rs/profile.rs dispatch-layer pattern (trivial, engineering hygiene). Do **not** budget time on tcgen05/tcmoe/comm.cu/kv-transfer-rdma — §4.

Bottom line: items 1+3 alone justify the study; item 2 is the decode-kernel comparison partner for our vendor path.

## 6. Sources

- Repo (cloned 2026-11-14, 4 commits @ 76766a9): [github.com/basetenlabs/deepseek-ultrafast](https://github.com/basetenlabs/deepseek-ultrafast) — all `crates/...` path anchors above
- Repo metadata (stars/forks/dates/license): [api.github.com/repos/basetenlabs/deepseek-ultrafast](https://api.github.com/repos/basetenlabs/deepseek-ultrafast)
- Baseten optimization blog (fused MoE −2.7 ms, mHC +30% bs1, PDL, CUDA graphs, 991 tok/s AA): [baseten.co/blog/how-to-make-flash-models-even-faster-optimizing-deepseek-v41-flash-1017-fast](https://www.baseten.co/blog/how-to-make-flash-models-even-faster-optimizing-deepseek-v41-flash-1017-fast)
- README perf table + acceptance ranges + KV sharing: [github.com/basetenlabs/deepseek-ultrafast#performance](https://github.com/basetenlabs/deepseek-ultrafast#performance)
- Server behavior (single-request shape, CED SWA replay, smem top-k): [docs/SERVER.md](https://github.com/basetenlabs/deepseek-ultrafast/blob/main/docs/SERVER.md)
- sm_120 FP4/e2m1 support + sm_100-era gaps: [PTX ISA docs](https://docs.nvidia.com/cuda/parallel-thread-execution/), [FlashInfer issue #2649](https://github.com/flashinfer-ai/flashinfer/issues/2649)
- tcgen05 family-specific targets (sm_100a/103a vs sm_120a limits): [PTX ISA docs](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- Velo RDMA library (third-party dep): [github.com/ai-dynamo/velo](https://github.com/ai-dynamo/velo)
