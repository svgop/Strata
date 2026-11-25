# F-453 — stage bandwidth-floor audit of the decode window (read-only, 2026-11-25)

Question: how far each stage of the warm window sits above the time its bytes need at 1.3 TB/s. No build, no GPU work, no engine run, no existing file modified; the only SQL is read-only over `/tmp/f452-dev.sqlite`.
Sources of every number: **trace** = queries over that sqlite for this file (293 warm windows = parent launches 479..771; T from graphId 4/38/30/46/65/57 = 1..6; stage times are summed kernel durations on the 3 streams). **census** = `nsys-inwindow-census.md` (section, line). **GGUF** = header parse of the `--native` shard with `tools/gguf_reader.py`. **pack** = `packs/iq3_s/index.txt` + `native_experts.txt`. **log** = `logs/custom-iq3_s.log`. Derived from code, not measured: marked UNCONFIRMED STATIC ANALYSIS.

## 0. Findings

1. Fixed weights per window are 4.36 GB: GDN projections 1.557 GB, hyper-connections (HC) 1.283 GB, LM head 0.521 GB, QSA projections 0.477 GB, shared expert 0.146 GB, router 0.126 GB, PLE 0.066 GB. Routed experts add 1.2-2.0 GB at T=2 (count not observable, section 2). A T=2 window moves 5.77-6.51 GB in 8.732 ms of wall: 660-746 GB/s average. T=3: 6.44-7.56 GB in 10.236 ms: 629-738 GB/s.
2. Headroom at 1.3 TB/s, T=2 / T=3: QSA + projections 0.90 / 0.91 ms, GDN + projections 0.88 / 1.05 ms, shared expert 0.73 / 0.79 ms, router/plan 0.58 / 0.69 ms, HC 0.57 / 0.78 ms. The LM head has none: it reads 521.47 MB in 317 us = 1.64 TB/s (trace, medians T=2 and T=3), the box's demonstrated rate for a Q6_K matvec; 1.3 TB/s is a conservative floor by 26% for kernels with enough blocks.
3. The HC contention hypothesis is refuted by the trace: 0 of 56,834 `gr_down_v3_kernel` / `gr_up_v3_kernel` instances in the 293 warm windows overlap any side-stream kernel. The HC read runs on an otherwise idle GPU at 830 GB/s (T=2).
4. The routed-expert byte figure in the brief does not reproduce: an expert is 2,046,400 B mean (1,510,400-2,662,400), not 1.4 MB, and 34 distinct experts per layer exceeds the 10 x 2.676 = 26.8 entries per layer the warm mix allows. The stage's effective rate is between 0.82 and 1.32 TB/s at T=2 depending on the count (section 2).
5. None of the three largest-headroom stages is HBM-bound: GDN 796 GB/s, QSA 393 GB/s, shared expert 177 GB/s (T=2). The time is in too few blocks (48, 16, 1), chains of 0.7-5 us launches, and one stall: the attention gate matvec starts a median 10.9 us after its input is ready (trace, 2,932 T=2 instances).

## 1. Tensor table of the served model

Pack `index.txt` rows have 19 fields `name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits ...` (src/core/weights.cpp:198-204); kind 4 = BF16 copied raw, kind 2 = F32 (weights.cpp:214-217). All BF16/F32 tensors sit in `dense.bin` (pool 1,538,035,200 B); the quantized projections have `src_bytes 0` (native GGUF, argv `--native`, log:551517); the 144 expert tensors are mapped from the GGUF by `native_experts.txt` (header: 512 experts, 50,292,326,400 B). The traced boot read HC weights as bf16 ("hc read weights bf16 rows", log:551582). The brief's names `attn_hyper_connection.*`, `hyper_connection_mixer.*`, `mlp.gate` do not exist in this GGUF (`general.architecture = qwen4exp`); the equivalents are the `hc_*` tensors and `ffn_gate_inp`. GGUF shapes are `ne0 x ne1` (ne0 contiguous).

| role (per layer; GGUF name) | layers | type (layer count) | shape | bytes (mean per layer) | bits/weight |
|---|---|---|---|---|---|
| HC read, attn: `hc_attn_down.weight`, `hc_attn_up.weight` | 48 | BF16 | 10240x320, 320x10240 | 6,553,600 each | 16 |
| HC read, ffn: `hc_ffn_down.weight`, `hc_ffn_up.weight` | 48 | BF16 | 10240x320, 320x10240 | 6,553,600 each | 16 |
| HC inject `hc_attn_inject`, `hc_ffn_inject` / norm `hc_attn_norm`, `hc_ffn_norm` | 48 | BF16 / F32 | 10240x4 / 10240 | 81,920 each / 40,960 each | 16 / 32 |
| HC before the head: `output_hc_down`, `output_hc_up` (BF16 320x10240), `output_hc_norm` (F32) | 1 | BF16, F32 | | 13,148,160 once | 16, 32 |
| GDN `attn_qkv.weight` | 36 | Q6_K (22), Q5_K (10), Q4_K (2), IQ4_XS (2) | 2560x10240 | 19,740,444 | 4.5-6.56 |
| GDN `attn_gate.weight` | 36 | Q6_K (14), Q4_K (10), IQ4_XS (8), Q5_K (4) | 2560x6144 | 10,533,547 | 4.5-6.56 |
| GDN `ssm_out.weight` | 36 | Q6_K (29), Q5_K (4), Q4_K (3) | 6144x2560 | 12,332,373 | 5.5-6.56 |
| GDN `ssm_alpha.weight`, `ssm_beta.weight` (BF16 2560x48, 245,760 each); `ssm_conv1d.weight` (F32 4x10240, 163,840); `ssm_a`, `ssm_dt.bias`, `ssm_norm.weight` (F32, 896) | 36 | BF16, F32 | | 656,256 | 16, 32 |
| QSA `attn_q.weight` | 12 | Q6_K (5), Q5_K (3), IQ4_XS (3), Q4_K (1) | 2560x12288 | 21,811,200 | 4.5-6.56 |
| QSA `attn_k.weight`, `attn_v.weight` | 12 | Q6_K (11 / 10), IQ4_XS (1 / 0), Q5_K (0 / 2) | 2560x512 | 1,043,627 / 1,046,187 | 5.5-6.56 |
| QSA `attn_output.weight` | 12 | Q6_K (11), Q4_K (1) | 6144x2560 | 12,564,480 | 4.5-6.56 |
| QSA indexer `q_proj`, `k_proj` (BF16); q/k norms and `attn_q_norm`, `attn_k_norm` (F32) | 12 | BF16, F32 | 2560x512, 2560x128 | 3,279,872 | 16, 32 |
| shared expert `ffn_gate_shexp.weight` | 48 | Q4_K (16), IQ4_XS (15), Q6_K (13), Q5_K (4) | 2560x640 | 1,037,067 | 4.5-6.56 |
| shared expert `ffn_up_shexp.weight` | 48 | IQ4_XS (13), Q4_K (14), Q6_K (13), Q5_K (8) | 2560x640 | 1,056,267 | 4.25-6.56 |
| shared expert `ffn_down_shexp.weight`, `ffn_gate_inp_shexp.weight` | 48 | IQ4_NL (47), Q8_0 (1); BF16 | 640x2560; 2560 | 938,667; 5,120 | 4.5-8.5; 16 |
| router `ffn_gate_inp.weight` | 48 | BF16 | 2560x512 | 2,621,440 | 16 |
| routed `ffn_gate_exps`, `ffn_up_exps` (512 experts each) | 48 | IQ3_XXS (17), IQ2_S (20), IQ3_S (10), IQ4_XS (1) | 2560x640x512 | 605,600 per expert | 2.56-4.25 |
| routed `ffn_down_exps` (512 experts) | 48 | IQ4_NL (39), Q2_0 (9) | 640x2560x512 | 835,200 per expert | 2.25-4.5 |
| -> one routed expert, gate + up + down | | | | 2,046,400 mean (1,510,400-2,662,400) | |
| LM head `output.weight` | 1 | Q6_K | 2560x248320 | 521,472,000 | 6.56 |
| PLE `ple_key.weight`, `ple_value.weight` (BF16 2560x10240, 2560x2560), `ple_conv1d.weight` (F16), `ple_norm_*` (F32) | 1 | BF16, F16, F32 | | 65,774,720 once | 16, 32 |
| `token_embd.weight` IQ4_XS (337,715,200 B): T rows per window; `per_layer_token_embd.weight` IQ4_NL 28.8 GB (`--ple-gguf`, mmap): n-gram rows only | | | | not a per-window cost | |

## 2. Bytes per window (weights once per window; every projection kernel reads its weights once for all T rows: `native_mmvq_il_kernel<I, NC=T, R>` native_mmvq.cu:1698-1711, `native_mmvq_multi_kernel<F, NCOLS=T, ..>` :1176-1189)

| stage | weights, B per window (arithmetic) | activations, B (T=2 / T=3) | total MB T=2 | total MB T=3 |
|---|---|---|---|---|
| HC | 96 x (2 x 320 x 10240 x 2 + 4 x 10240 x 2 + 10240 x 4) + 13,148,160 = 1,283,235,840 | per read per token <= R 40,960 read + R_out 40,960 write + mixed 10,240 write + bo_prev 40,960 read: 25.8 / 38.7 MB | 1,309.1 | 1,322.0 |
| GDN + proj | 36 x (19,740,444 + 10,533,547 + 12,332,373 + 656,256) = 1,557,454,336 | state 48 x 128 x 128 x 4 B read and written once per window = 6,291,456 per layer (verify_kernels.cu:374-377, :307) + ~281 KB per token per layer: 246.8 / 256.8 MB | 1,804.3 | 1,814.3 |
| QSA + proj | 12 x (21,811,200 + 1,043,627 + 1,046,187 + 12,564,480 + 3,279,872) = 476,944,384 | int8 K+V of the top-2048 tokens, 1.18 MB per token per layer, + ~100 KB other: 30.7 / 46.1 MB (upper bound; context length not in the trace) | 507.7 | 523.0 |
| shared expert | 48 x (1,037,067 + 1,056,267 + 938,667 + 5,120) = 145,781,760 | ~39 KB per token per layer: 3.8 / 5.6 MB | 149.6 | 151.4 |
| router / plan | 48 x 2560 x 512 x 2 = 125,829,120 | logits and plan arrays < 0.2 MB | 126.0 | 126.0 |
| routed experts | 48 x N x 2,046,400; N = distinct GPU-computed experts per layer: N <= 20 (T=2), <= 30 (T=3) | hq 656 B + out 10,240 B per entry: 10.6 / 15.9 MB | 1,229 (N=12.4) .. 1,975 (N=20) | 1,843 (N=18.6) .. 2,963 (N=30) |
| LM head | 248,320 x 2560 / 256 x 210 = 521,472,000 | logits 248,320 x 4 x T: 2.0 / 3.0 MB | 523.5 | 524.5 |
| PLE key / value | 52,428,800 + 13,107,200 = 65,536,000 | ~1 MB | 66.5 | 66.5 |
| cvec, quant, embedding, misc | none | ~50 / ~65 MB (UNCONFIRMED STATIC ANALYSIS) | 50 | 65 |
| window total | 4.36 GB fixed + routed | | 5,766 .. 6,512 | 6,436 .. 7,556 |

Routed-expert count N: the VRAM call is launched at the cap (grid (20, 20) at T=2, (20, 30) at T=3; iq_kernels.cu:3920, :4892, :1813-1818), so the trace does not show N. The PCIe call's grid is (20, 4) and runs 0.48 us (empty) in most layers; the copy engine brings 5.22 transfers >= 1 MiB, 12.85 MiB per window (census:104), about 6 experts. The engine's own per-window counters exist only for another run (log:545171: 718 windows of 4.85 rows, 30.08 VRAM + 0.24 PCIe + 1.05 CPU experts per layer-window = 65% of the 48.5 entries distinct; hit rates 84.5-95.7% at log:544805, :544822, :544851). N = 12.4 (T=2) and 18.6 (T=3) below apply that 62% VRAM distinct ratio and are UNCONFIRMED.

## 3. Headroom at 1.3 TB/s (floor = bytes / 1.3e12; measured = census 7c per-T means, lines 401-411, for every stage; no stage needed the overall mean)

| rank | stage | bytes MB T=2 / T=3 | measured ms T=2 / T=3 | effective GB/s T=2 / T=3 | floor ms T=2 / T=3 | headroom ms T=2 / T=3 |
|---|---|---|---|---|---|---|
| 1 | QSA + projections | 507.7 / 523.0 | 1.294 / 1.313 | 393 / 398 | 0.391 / 0.402 | 0.903 / 0.911 |
| 2 | GDN + projections | 1,804.3 / 1,814.3 | 2.265 / 2.441 | 796 / 743 | 1.388 / 1.396 | 0.877 / 1.045 |
| 3 | shared expert | 149.6 / 151.4 | 0.847 / 0.909 | 177 / 167 | 0.115 / 0.116 | 0.732 / 0.793 |
| 4 | router / plan | 126.0 / 126.0 | 0.679 / 0.789 | 186 / 160 | 0.097 / 0.097 | 0.582 / 0.692 |
| 5 | hyper-connections | 1,309.1 / 1,322.0 | 1.577 / 1.801 | 830 / 734 | 1.007 / 1.017 | 0.570 / 0.784 |
| 6 | routed experts | 1,229..1,975 / 1,843..2,963 | 1.501 / 1.851 | 819..1,316 / 996..1,601 | 0.945..1,519 / 1.418..2,279 | 0.556..-0.018 / 0.433..-0.428 |
| 7 | LM head | 523.5 / 524.5 | 0.375 / 0.331 | 1,396 / 1,585 | 0.403 / 0.403 | -0.028 / -0.072 |
| 8 | PLE key / value | 66.5 / 66.5 | 0.044 / 0.044 | 1,511 / 1,511 | 0.051 / 0.051 | -0.007 / -0.007 |
| - | host handshake / spin (no bytes) | 0 | 1.307 / 1.933 | - | 0 | not a bandwidth term |
| | window | 5,766..6,512 / 6,436..7,556 | wall 8.732 / 10.236 | 660..746 / 629..738 | 4.44..5.01 / 4.95..5.81 | wall minus floor 4.30..3.72 / 5.28..4.42 |

All-warm means (census 7c) for reference, ms: GDN 2.323, HC 1.756, routed 1.739, QSA 1.328, shared 0.870, router 0.720, head 0.370, handshake 2.142. The census's unidentified "per-token tail block" (census:411) is the PLE key / value pair: `bf16_f32_mmvf_multi_kernel<256,2,1>` at grid 10,240 (52.4 MB, 33.9 us median = 1.55 TB/s) and grid 2,560 (13.1 MB, 10.1 us = 1.30 TB/s), trace. The T=3 routed stage at N = 30 would need 1.60 TB/s, the LM head's rate, with 400-1,600-block kernels sharing HBM with the main stream: N is below the cap, and the stage is near or above 1.3 TB/s either way.

## 4. The three largest-headroom stages: kernels, launch shapes, mechanism

### 4a. GDN + projections (36 layers; T=2 window; trace)
| kernel (role) | n/win | mean / median us | T=3 median us | grid | block | regs/thread | shared B |
|---|---|---|---|---|---|---|---|
| `native_mmvq_il_kernel<IlQ6K,2,1>` (attn_qkv, Q6_K, 21.5 MB) | 22 | 23.66 / 21.28 | 19.84 (`<IlQ6K,3,2>`, grid 1280) | 2560 | 128 | 40 | 0 |
| `native_mmvq_il_kernel<IlQ5K,2,1>` / `<IlQ4K,2,1>` / `<IlIQ4XS,2,2>` (attn_qkv, 10 / 2 / 2 layers) | 10 / 2 / 2 | 15.02 / 13.79; 15.35 / 13.25; 11.55 / 11.52 | 13.95; 13.46; 12.24 | 2560, 2560, 1280 | 128 | 56, 64, - | 0 |
| `native_mmvq_il_kernel<IlQ6K,2,2>` / `<IlQ4K,2,2>` / `<IlIQ4XS,2,2>` / `<IlQ5K,2,2>` (attn_gate, 10.5 MB, side stream) | 14 / 10 / 8 / 4 | 16.11 / 15.81; 11.15 / 9.76; 11.08 / 10.75; 14.69 / 12.19 | 15.49; 13.63; 14.30; 15.58 (all `<..,3,4>`, grid 384) | 768 | 128 | 64, 64, 72, 64 | 0 |
| `native_mmvq_multi_kernel<Q6KTraits,2,4,2,0,0>` (ssm_out Q6_K, 12.9 MB; class also holds 11 QSA attn_output) | 29 | 13.45 / 12.86 | 12.83 (`<Q6KTraits,3,4,2,0,0>`) | 1280 | 32x4 | 72 | 1536 |
| `native_mmvq_multi_kernel<Q4KTraits,2,4,2,0,0>` / `native_mmvq_il_kernel<IlQ5K,2,1>` (ssm_out Q4_K / Q5_K) | 3 / 4 | 10.17 / 7.62; 9.17 / 8.29 | 9.98; 10.83 (`<IlQ5K,3,1>`) | 1280; 640 | 32x4; 128 | 64; 56 | 1536; 0 |
| `gdn_ab_multi_kernel<2,1>` (ssm_alpha, ssm_beta: 0.49 MB) | 36 | 9.33 / 8.22 | 9.15 (`<3,1>`) | 48 | 64 | 40 | 0 |
| `gdn_step_norm_multi_kernel<1,0,0>` (state 6.29 MB) | 36 | 5.02 / 4.58 | 5.54 | 48 | 128x4 | 100 | 6160 |
| `gdn_conv_l2_multi_kernel` (side stream) | 36 | 2.18 / 1.76 | 2.40 | 80x2 | 128 | 30 | 16 |
Sum 2.26 ms = census 2.265.

Mechanism (trace + code):
- The three projections stream at 0.66-1.0 TB/s (attn_qkv 21.5 MB in 21.3 us median; attn_gate 10.5 MB in 15.8 us; ssm_out 12.9 MB in 12.9 us) with full-occupancy grids (2560 x 128 threads). The same kernel at the LM head's grid of 62,080 reaches 1.64 TB/s. A line through ssm_out and the head, t = 5.5 us + bytes / 1.67 TB/s (UNCONFIRMED model, two points), predicts attn_qkv at 18.3 us against 21.3 measured. In one window (649) the same Q5_K qkv matvec (18.0 MB) takes 13.7, 21.7 and 12.7 us in layers 1, 2 and 26 (at 242.8, 368.3, 3877.2 us) while sharing HBM with the gate matvec for 5.5-11 us of each.
- `gdn_ab_multi_kernel`: 48 blocks x 64 threads = 3,072 threads on 170 SMs; one warp per row, 10 dependent 16-B loads per lane (verify_kernels.cu:116-151). 0.49 MB in 8.2 us = 60 GB/s: latency on the main stream, ahead of everything else in the layer.
- Stall: the gate matvec starts a median 4.67 us after `gdn_ab_multi_kernel` ends and 10.91 us after the `native_q8_1_interleave_kernel` that produces its input ends (T=2, n = 2,932; T=3: 4.40 / 11.07). The main stream's next kernel (`gdn_step_norm_multi_kernel`) starts 0.19 us after the gate matvec ends: 26.08 us from `gdn_ab` start to the state step, of which `gdn_ab` is 8.26 us. Why the launch waits is not in the export (the sqlite has node times, not graph edges).
- `gdn_step_norm_multi_kernel` is at the floor: 6.29 MB in 4.58 us = 1.37 TB/s. No redundant pass: weights are read once per window for all rows (NC = T).

### 4b. QSA + projections (12 layers; T=2 window; trace)
| kernel (role) | n/win | mean / median us | T=3 median us | grid | block | regs | shared B (static + dynamic) |
|---|---|---|---|---|---|---|---|
| `native_mmvq_il_kernel<IlQ6K,2,1>` / `<IlQ5K,2,1>` / `<IlQ4K,2,1>` / `<IlIQ4XS,2,2>` (attn_q, 21.8 MB) | 5 / 3 / 1 / 3 | 20.61 / 18.88; 16.02 / 15.42; 13.10 / 13.06; 16.46 / 12.74 | 22.50; 21.31; 13.10; 13.44 (T=3: `<IlQ6K,3,4>` grid 768; `<IlQ5K,3,4>` 768; `<IlQ4K,3,2>` 1536; `<IlIQ4XS,3,4>` 384) | 3072, 3072, 3072, 1536 | 128 | 56, 56, 64, - | 0 |
| `native_mmvq_multi_kernel<Q6KTraits,2,4,2,0,0>` / `<Q5KTraits,..>` / `<IQ4XSTraits,..>` (attn_k, attn_v, ~1.05 MB, side stream) | 21 / 2 / 1 | 6.49 / 5.28; 5.65 / 5.02; 4.69 / 4.58 | 5.15; 5.33; 6.46 (`<..,3,4,2,0,0>`) | 256 | 32x4 | 72, 40, 74 | 1536 |
| `bf16_f32_mmvf_multi_kernel<256,2,1>` (indexer k_proj / q_proj, bf16) | 12 / 12 | 4.14 / 3.74; 3.73 / 3.36 (class of 60 incl. router) | 4.06; 3.68 | 128; 512 | 256 | 42 | 256 |
| `block_topk_cluster_kernel` | 12 | 17.39 / 15.71 | 15.87 | 8x2, cluster 8 | 1024 | 64 | 18,768 + 32,772 |
| `decode_attn_i8_split_kernel` | 12 | 12.82 / 12.03 | 12.34 | 2x2x20 | 128 | 165 | 0 + 45,584 |
| `resolve_kernel` | 12 | 5.48 / 4.27 | 5.54 | 1 | 1024 | 46 | 140 |
| `norm_rope_kernel<0>` (3 per layer) | 36 | 2.21 / 1.73 | 1.76 | 48, 8, 4 | 256 | 40 | 128 |
| `block_scores_multi_kernel` | 12 | 3.66 / 3.23 | 3.78 | 256 | 256 | 48 | 16,512 |
| `decode_attn_merge_kernel`, `copy_kernel`, `kv_append_q8_kernel<1>`, `append<0>`, `gate` | 12 each | 2.83, 2.35, 2.20, 2.05, 0.80 (means) | 2.66, 0.90, 1.89, 2.69, 0.79 | 2x24; 96; 2x4x4; 1; 48 | 256; 128; 64; 256; 256 | 40, 27, 34, 35, 20 | 0, 0, 8, 640, 0 |
Sum 1.29 ms = census 1.294 (projections 0.67, attention kernels 0.62).

Mechanism: the six projections move 39.7 MB per layer in ~50 us summed (0.75-0.8 TB/s): the same kernels and per-launch cost as 4a, three of them at 0.66-1.05 MB. The attention chain is 16 dependent launches per layer (14 attention kernels incl. `copy_from_mapped_kernel`, 2 quantizers), 51.9 us in the instance at 488-548 us of window 649, for at most 2.4 MB of K/V: `block_topk_cluster_kernel` is 16 CTAs (8 per query) doing a 4-pass radix select across a cluster (qsa_select.cu:1404-1444) and takes 15.7 us at T=2, 15.9 at T=3: fixed latency on 9% of the GPU's SMs; `decode_attn_i8_split_kernel` is 80 CTAs of 128 threads (qsa_prompt_attn.cu:800-817), fewer blocks than SMs; `resolve_kernel` is one block. Per launch ~3.2 us where the byte floor of the whole chain is ~2 us.

### 4c. Shared expert (48 layers; T=2 window; trace)
| kernel (role) | n/win | mean / median us | T=3 median us | grid | block | regs | shared B |
|---|---|---|---|---|---|---|---|
| `native_mmvq_multi_kernel` gate and up (~1.04 MB each): `<Q4KTraits,2,4,2,0,0>` / `<IQ4XSTraits,..>` / `<Q6KTraits,..>` / `<Q5KTraits,..>` | 30 / 28 / 26 / 12 (= gate + up layers: 16+14, 15+13, 13+13, 4+8) | 5.11 / 4.29; 4.90 / 3.68; 5.47 / 5.31; 5.90 / 4.35 | 4.45; 3.62; 5.39; 4.29 (`<..,3,4,2,0,0>`) | 320 | 32x4 | 64, 74, 72, 40 | 1536 |
| `native_mmvq_multi_kernel<SmallTraits<IQ4NLBlock,4>,2,4,2,0,0>` / `<SmallTraits<Q80Block,8>,..>` (down, 0.92 MB) | 47 / 1 | 3.59 / 3.04; 5.03 / 3.01 | 4.06 | 1280 | 32x4 | 62 | 1536 |
| `bf16_f32_mmvf_multi_kernel<256,2,1>` (the `ffn_gate_inp_shexp` scalar, 5 KB) | 48 | 2.38 / 1.92 | 2.02 | 1 | 256 | 42 | 256 |
| `sigmoid_scale_rows_vec4_kernel` (shared_expert.cu:181-199) | 48 | 1.16 / 0.96 | 0.96 | 5x2 | 128 | 16 | 0 |
| `native_swiglu_quantize_q8_1_kernel` (booked under norm/quant by the census) | 48 | 1.17 / 0.80 | 0.90 | 5 | 256 | 20 | 0 |
Sum 0.90 ms with the quantizer, 0.847 without = census 0.847.

Mechanism: six serial launches per layer on the main stream for 3.04 MB of weights, 17.6 us summed, 177 GB/s. Three are ~1 MB matvecs of 3.0-5.9 us (0.2-0.3 TB/s): at the LM head's rate their bytes need 0.6 us each, the rest is the per-launch cost of 4a. Two are latency only: the shared-expert gate scalar is one 256-thread block for a 2560-long dot (1.9 us for 5 KB), and the sigmoid scale re-reads and re-writes the 2,560-float row (1.0 us). Node-to-node gaps add 0.19 us each (census:228).

### 4d. Hyper-connections: the contention test
| kernel | n/win | T=2 mean / median us | T=3 mean / median us | grid | block | regs | shared B |
|---|---|---|---|---|---|---|---|
| `gr_down_v3_kernel<1,T,1,1,1,0>` | 97 | 7.18 / 6.82 | 8.44 / 7.55 | 41x4 = 164 (fused_gr.cu:2861) | 256 | 76 (T3: 100) | 64 + 20,480 (T3: 96 + 30,720) |
| `gr_up_v3_kernel<1,T,1,1>` | 97 | 9.07 / 8.32 | 10.12 / 9.57 | 160 (UPM_BLOCKS, :387, :2936) | 256 | 80 (T3: 87) | 3,104 (T3: 4,656) |

Test: for every instance of both kernels in the 293 warm windows (56,834; T=2: 31,622, T=3: 8,536), the time overlap with kernels of the two side streams: **overlapped by a side-stream kernel: n = 0, 0 us**. Not overlapped: all 56,834; medians (down / up): 6.82 / 8.32 us at T=2, 7.55 / 9.57 at T=3. A copy-engine memcpy covers 699 instances (1.2%; T=2: 174) for their whole duration with no change: pooled median 7.65 us with it, 7.65 us for the 31,448 T=2 instances with nothing else running. The side streams start their first kernel of each layer within 0.5 us after `gr_up_v3_kernel` ends (window 649, five layers read: e.g. 240.2 -> 240.4). Contention with the expert streams is absent; the hypothesis is refuted for this trace.

Why 830 GB/s: too few blocks (164 and 160 blocks on 170 SMs, one block per SM, no second block to hide latency behind), a prologue that stages R, bo_prev and w_norm into shared memory and reduces before the first weight load (fused_gr.cu:852-943), and in `gr_up_v3_kernel` 1.25 16-B weight loads per lane per row followed by a 5-shuffle warp reduction per token (fused_gr.cu:1142-1166): low bytes in flight. The duration grows faster than the bytes with T (down: 6.14 / 6.82 / 7.55 us at T = 1 / 2 / 3 for 6.74 / 6.81 / 6.88 MB): the per-token dot loop is serial work, not bandwidth. `gr_down` moves 6.81 MB in 6.82 us = 998 GB/s at the median, `gr_up` 6.90 MB in 8.32 us = 830 GB/s.

## 5. Smallest change that recovers the headroom (no implementation)

| stage | change | output |
|---|---|---|
| GDN + proj | Order the gate matvec after its producer (the q8_1 interleave), not after `gdn_ab`: it is ready 10.9 us earlier; the state step would start 3-5 us sooner per layer (the qkv matvec + conv chain, ~22.5 us, then sets the path): 0.11-0.17 ms per T=2 window. | Same kernels, same per-stream order: bitwise. The dependency is in the capture order (UNCONFIRMED: graph edges not exported). |
| GDN + proj | Give `gdn_ab_multi_kernel` its 10 loads in flight (constant trip count, the `c_gdn_ab_fast` path verify_kernels.cu:124-128, or 2 warps per row with a final add): up to 6 us x 36 = 0.22 ms, on the critical path. | Constant trip count: bitwise (same FMAs, same order). Two warps per row: new summation order, parity harness. |
| QSA + proj | More CTAs where there are fewer than SMs: `decode_attn_i8_split_kernel` gridDim.z 20 -> 40+, `block_topk_cluster_kernel` cluster 8 -> 16, `resolve_kernel` 1 -> 8 blocks with a second-level pass; and fold the three `norm_rope_kernel` launches and `resolve_kernel` + `copy_kernel` into one launch each (saves ~5 launches x 0.19 us gap + their fixed cost). | Attention splits: the merge sums a different grouping, parity harness. Top-k: exact selection, identical ids if tie-breaking is unchanged: bitwise, confirm with the diff harness. Fusions of elementwise passes: bitwise where each element's arithmetic is unchanged. |
| shared expert | One launch for gate and up: the kernel already has the PAIR form (native_mmvq.cu:1174-1188); append `ffn_gate_inp_shexp` as a 513th row of the router GEMM instead of a 1-block launch; apply the sigmoid scale where the down matvec writes its row (one multiply, same operands). Prices the three removed launches at their medians + the 0.19 us gap: sigmoid 0.055 ms, gate scalar 0.10 ms, up matvec's fixed cost ~0.17 ms: up to 0.33 of the 0.73 ms. | All three bitwise: each output keeps its own row code and operation order (PAIR: "each output's code is the single matrix's"). |
| HC (5th) | Not a faster kernel first: the read occupies an idle GPU, so prefetch the next layer's qkv / gate weights (32 MB) into L2 from a side stream during the 16 us read; then S = 2 for `gr_down` (328 blocks, exists: fused_gr.cu:3038) or Q8_0 HC weights (13.2 -> 7.0 MB per read). | Prefetch: bitwise. S = 2 and Q8: new summation order / new precision: parity harness and the MMLU gate. |

Every item is env-gated, default off, with the ABAB interleave and the all-lane parity the campaign law requires; the priced term is given per line.

## 6. Not determined

- N, the distinct GPU-computed experts per layer and window: the VRAM call is cap-sized and no counter is in the trace; the log's per-window tier counters (log:545171) belong to another run. The routed-expert rows are a bracket, not a point.
- The CPU-pool expert share of the routed stage: `dm_wait_combine_kernel` mixes wait and combine (census:378).
- Why the gate matvec launches late (4a) and why 15% of T=2 LM-head instances take 655 us instead of 317 (p90 654.94 us, n = 163, GPU otherwise idle, no memcpy): neither is visible in kernel timings.
- Context length of the capture, hence the QSA indexer bytes; the 1.18 MB per token per layer is the top-k bound.
- Whether `c_gdn_ab_fast` was on in the traced binary, and whether STRATA_HC_Q8 was set: the kernel names do not carry either. The traced boot ran the v3 read; the log's current boot reports the staged read (log:551581), a different kernel set: the HC numbers here describe the traced boot.
- DRAM traffic as opposed to bytes derived from shapes and code: no counter data. All activation bytes and the linear t(bytes) model are UNCONFIRMED STATIC ANALYSIS.
- Clock and power state during the capture; per-node tracing overhead inside every duration (census:444).
- L2 capacity and hit behaviour: in the layer-0 instance of window 649 the gate matvec (10.5 MB, 9.5 us) runs entirely inside the qkv matvec (21.5 MB, 21.3 us), which needs about 2.0 TB/s if both stream uniformly, above the 1.79 TB/s peak: part of it is L2 hits or ramp effects, not resolvable from timings.
