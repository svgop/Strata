# T14B-exec-B — the P2 torch parity module: execution-ready spec

Date: 2026-10-18. Author: Researcher (read-only session; nothing was booted, built, or modified on the lane; /mnt/DATA untouched).
Engine: `/home/svg/LLM/strata-custom-2` at git `315919bc`, built binary `engine/strata` (2026-10-17 11:15).
Sibling specs: `campaign/research/T14B-fastmtp-training-infra.md` (training infrastructure), `campaign/research/T14B-exec-C-trainer-export-spec.md` (trainer/export), `campaign/research/T14C-fastmtp-acceptance-math-gates.md` (G1+ gates). This file is the P2 gate: the module math and its proof harness.

---

## 0. Read-me

P2 = one PyTorch module that reproduces the ENGINE's drafter forward (one MTP layer + final mixer + sub-vocab head) exactly enough that the existing `STRATA_MTP_TAP` stage-diff harness adjudicates it. Three laws govern every line of the implementation; each was paid for once in the R17 arc (`SVG-CUSTOM.md` F-77..F-118):

1. **Weights come from the RT directory (`/home/svg/LLM/Custom-data/mtp/rt`), never from the BF16 sidecar.** The engine serves the packed rt dialect; a sidecar-sourced module can never close parity even when the math is right (F-94, F-95).
2. **The Gemma "+1" is ALREADY folded into every f32 `*norm.weight` in dense.bin** (`tools/mtp_rt.py:85`: F32 norm tensors are stored as `w + 1`). The torch module loads the stored gamma as-is. Re-folding is the F-109 bug (every norm shifted by +1; 0% acceptance).
3. **`--mtp-hnorm stream` is the lane's mode** (`campaign/adaptmtp-iq3_s-8081.json`): the hidden-input norm is PER-STREAM grouped RMS (mtp.cpp:941-942), not one pooled RMS (F-83 A).

Everything below cites `file:line` in the repo. Engine facts were read from source on 2026-10-18; nothing here is from memory or secondary summaries.

---

## 1. Versions and scope in force

- Engine: strata-custom-2 @ `315919bc`; journal state F-236 (F-65..F-121 R17 arc complete in `SVG-CUSTOM.md` lines 2443-3278).
- Drafter weights: `/home/svg/LLM/Custom-data/mtp/rt` = `dense.bin` 116,099,072 B + `dense.txt` (29 tensors) + `experts.bin` 707,788,800 B (512 x 1,382,400 B) + `draft_vocab.bin` 162,100 B = **40,525 token ids** (the "en" subset; matches `data/draft_vocab_en.bin`).
- Lane boot (the config the parity boot must copy): `campaign/adaptmtp-iq3_s-8081.json` — `--spec 6`, `--kv int8`, `--kv-resident 32768`, `--mtp-q4 all`, `--mtp-hnorm stream`, `--mtp /home/svg/LLM/Custom-data/mtp/rt`, `--native` GSQ-RCO-IQ3_S shard 1+2.
- Geometry (`include/strata/core/geometry.hpp:24-64`, confirmed against the GGUF metadata of `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001`): N=2560, HC=4, HC_LR=320, NH=24, NKV=2, HD=256, n_rot=64, rope freq_base **1e7** (`rope.freq_base`), sections [11,11,10,0] (text positions: one position per row), 512 experts top-10, n_ff=640 (routed and shared), n_vocab=248,320, rms_eps=1e-6 (`EPS`, mtp.cpp:91).
- Draft window: `kVerifyMaxT = 8` (`verify_kernels.hpp:21`); the lane runs `--spec 6` so `max_t_ = 6` and one draft round produces `max_steps = min(max_t_-1, max_drafts_) = 5` drafts (mtp.cpp:1588) = 1 round application + 4 step applications (chain outputs j=0..4). Verify window = accepted row + drafts = 6 tokens. **K=6 in the training plan = the 6 supervised positions of one round** (window row + 5 chain outputs).
- torch: 2.11 (cu130) verified present in the F-78 era environment; gguf-py local at `third_party/llama.cpp/gguf-py` (`tools/_paths.py`).

---

## 2. THE FORWARD, tensor by tensor

### 2.1 Weight provenance — what the module loads

From `dense.txt` (the rt manifest; every offset 256-byte aligned; kinds `q8_0 | bf16 | f32`):

| # | rt tensor (`mtp.` prefix stripped) | rt kind | rows x cols | bytes |
|---|---|---|---|---|
| 1 | fc_embedding.weight | q8_0 | 2560x2560 | 6,963,200 |
| 2 | fc_hidden.weight | q8_0 | 2560x2560 | 6,963,200 |
| 3 | hyper_connection_mixer.hc_norm.weight | **f32 (1+w folded)** | 10240 | 40,960 |
| 4 | hyper_connection_mixer.input_mix_weight_down.weight | bf16 | 320x10240 | 6,553,600 |
| 5 | hyper_connection_mixer.input_mix_weight_up.weight | bf16 | 10240x320 | 6,553,600 |
| 6 | attn_hyper_connection.block_inject_weight.weight | bf16 | 4x10240 | 81,920 |
| 7 | attn_hyper_connection.hc_norm.weight | **f32 (1+w folded)** | 10240 | 40,960 |
| 8 | attn_hyper_connection.input_mix_weight_down.weight | bf16 | 320x10240 | 6,553,600 |
| 9 | attn_hyper_connection.input_mix_weight_up.weight | bf16 | 10240x320 | 6,553,600 |
| 10 | mlp.gate.weight (router) | bf16 | 512x2560 | 2,621,440 |
| 11 | mlp.shared_expert.down_proj.weight | q8_0 | 2560x640 | 1,740,800 |
| 12 | mlp.shared_expert.gate_proj.weight | q8_0 | 640x2560 | 1,740,800 |
| 13 | mlp.shared_expert.up_proj.weight | q8_0 | 640x2560 | 1,740,800 |
| 14 | mlp.shared_expert_gate.weight | bf16 | 1x2560 | 5,120 |
| 15 | mlp_hyper_connection.block_inject_weight.weight | bf16 | 4x10240 | 81,920 |
| 16 | mlp_hyper_connection.hc_norm.weight | **f32 (1+w folded)** | 10240 | 40,960 |
| 17 | mlp_hyper_connection.input_mix_weight_down.weight | bf16 | 320x10240 | 6,553,600 |
| 18 | mlp_hyper_connection.input_mix_weight_up.weight | bf16 | 10240x320 | 6,553,600 |
| 19 | self_attn.indexer.index_qk_proj.weight | q8_0 (UNUSED at serve) | 640x2560 | 1,740,800 |
| 20 | self_attn.indexer.k_layernorm.weight | f32 (UNUSED at serve) | 128 | 512 |
| 21 | self_attn.indexer.q_layernorm.weight | f32 (UNUSED at serve) | 128 | 512 |
| 22 | self_attn.k_norm.weight | **f32 (1+w folded)** | 256 | 1,024 |
| 23 | self_attn.k_proj.weight | q8_0 | 512x2560 | 1,392,640 |
| 24 | self_attn.o_proj.weight | q8_0 | 2560x6144 | 16,711,680 |
| 25 | self_attn.q_norm.weight | **f32 (1+w folded)** | 256 | 1,024 |
| 26 | self_attn.q_proj.weight | q8_0 | 12288x2560 | 33,423,360 |
| 27 | self_attn.v_proj.weight | q8_0 | 512x2560 | 1,392,640 |
| 28 | pre_fc_norm_embedding.weight | **f32 (1+w folded)** | 2560 | 10,240 |
| 29 | pre_fc_norm_hidden.weight | **f32 (1+w folded)** | 10240 | 40,960 |

Plus:
- **experts.bin**: 512 blobs of 1,382,400 B (`BLOB = 3*(2560*640*18/64)`, `cpu/expert.hpp:52`). Per blob (`tools/mtp_rt.py:52-66`): Q2_0 codes plane of the gate/up rows first (`O_GU_CODES=0`, rows INTERLEAVED: blob row 2r = gate row r, 2r+1 = up row r — the GGUF tensor is gate-major (gate rows 0..639, up 640..1279) and the pack interleaves), then down codes (`O_D_CODES = 2*640*640 = 819,200`), then the fp16 scales planes. Q2_0 block: QK=64 weights per fp16 scale, 16 code bytes (2 bits/weight, 4 elements per byte, LSB-first), 18 B/block; gate/up row = 2560 w = 40 blocks = 720 B; down row = 640 w = 10 blocks = 180 B. **F-114 law: dequantize the STOCK `experts.bin` blob bytes, never a re-quant.**
- **draft head**: 40,525 rows gathered from the main native head (`output.weight`, Q6_K in the GSQ GGUF) at the ids in `draft_vocab.bin` (mtp.cpp:726-778), then **re-quantized Q4_0** because the lane runs `--mtp-q4 all` (`make_q4_head`, mtp.cpp:246-269: dequant Q6_K -> ggml reference Q4_0 per row, in place). Torch mirror: read the Q6_K rows for the 40,525 ids via gguf-py `quants.dequantize`, requant Q4_0 with gguf-py's reference quantizer, and use the Q4_0 **bytes** (not the Q6_K originals) for the G0-exact head.
- **token embedding**: gathered per token from the main model's `token_embd` GGUF-form table (`native_embed` path, mtp.cpp:925-936). Torch: dequantize the same table once with gguf-py and index rows; keep fp32 for the gathered row.

Q8_0 reference quantize (what dense.bin's q8_0 tensors contain; `mtp_rt.py:34-44`): per 32-block `d = amax/127` stored fp16; codes `sign(v)*floor(|v|+0.5)` (round half away from zero). Q4_0 (ggml reference): per 32-block, `d = amax/(-8)` convention of ggml's `quantize_row_q4_0_ref` — use gguf-py's implementation, do not hand-roll (the F-79 lesson: the hand-rolled dequant cost a session).

### 2.2 The engine's op semantics — per-stage with torch equivalents

Serve always runs the full layer with **T=1** (record_front prefetches K/V for the window rows; `record_rest` runs the body once for the draft row; mtp.cpp:902-908, 1341-1350). The module models ONE application: inputs `(token:int32, Rin:[4,2560] f32, pos:int32, cell:int32)`.

**(B1) Embedding branch.** `emb = dequant(token_embd)[token]` (f32, mtp.cpp:925-936) -> `en = rms(emb) * pre_fc_norm_embedding` (native_qsa.cu:45-68: `scale = rsqrtf(mean(x^2) + 1e-6)`; gamma multiplied AFTER the scale; fp32) -> `e2 = en @ fc_embedding^T` where fc_embedding is consumed as its **Q8_0/Q4_0 bytes against a Q8_1-quantized activation** (mtp.cpp:938-939).
- Torch (G0-exact): `q81(en)` per 32-block: `d = clamp_f32(amax/127)`, `q = roundf(x/d)` clamp [-127,127] (`q8_1_finite.hpp:18-27`); dot = fp32 sum over blocks of `d_act * d_w(fp16->f32) * sum(q_act * w_code)`. With `--mtp-q4 all` the weight side is the Q4_0 bytes: per 32-block `d_w` fp16 -> f32 and 4-bit codes (low nibble = even index). Weight bytes are bit-exact inputs; the only freedom is fp32 summation order.

**(B2) Residual branch + fusion.** Per stream c in 0..3: `hn[c] = rms(Rin[c]) * pre_fc_norm_hidden[c*2560:(c+1)*2560]` — STREAM mode, one RMS over the 2560-slice, each scaled by ITS slice of the 10240 gamma (mtp.cpp:941-942; kernel `norm_grouped`, native_qsa.cu:71-93: row r reads gamma slice `r % groups`) -> `h2[c] = hn[c] @ fc_hidden^T` -> `R[c] = h2[c] + e2` (`add_streams_broadcast`, mtp.cpp:950). Pooled mode exists (mtp.cpp:944) but the lane does not use it — assert the boot arg.

**(B3) HC read (shared formula, used 3x).** The exact math of `fused_gr_read_multi` (`fused_gr.cu`; the norm step :215-253, the down step :266-318, the up step :377-429; the T=1 multi kernels are bitwise the single-token kernels per the file's own parity law, :258-271, and gr_parity):
```
gw[c] = 2*sigmoid(inj_prev[c]/4)                    (only when apply=True; fused_gr.cu:88,223)
R'[c] = R[c] + bo_prev * gw[c]                      (apply=True only; :226-232)
rs[c] = rsqrtf( mean_over_2560(R'[c]^2) + 1e-6 )    (on the RAW R', BEFORE gamma; written to a.rs :248-249)
xn[c] = R'[c] * w_norm[c*2560:(c+1)*2560] * rs[c]   (:237, :252)
lo    = silu( (w_down @ xn) / 4 )                   ([320]; x = dot/HC then x/(1+exp(-x)); :312-316)
inj_new[c] = w_inject[c] @ xn                       ([4]; only when w_inject != nullptr; :123, :131-132)
up_i   = w_up[i,:] @ lo                             (i over 10240; :386-418)
rv_i   = R'[i]                                      (= R'[c*N+d]; R_out written when apply)
mixed[i] = sum_c( rv_i * w_norm[i] * rs[c] * sigmoid(up_i) ) / 4   (:411-426)
```
Sites: attn read mtp.cpp:952-965 (apply=False, w_inject set, outputs `lo_, rs_, inj_, mixed_, xn_`); mlp read mtp.cpp:1071-1085 (**apply=True with bo_prev = the attention output `bo_`, inj_prev = `inj_` — the attention write is folded into the MLP read**, and it writes `R_out = R'` back into `R_` in-place, fused_gr.cu:412-415); final mixer mtp.cpp:1204-1214 (`head_mix_multi_on` is the CUDA default, mtp.cpp:1100-1108; `STRATA_FUSE_HEAD_GR` default off :1096-1099) — apply=False, w_inject=None, output `sample_`.
- Torch: exact fp32 transcription; bf16 weights upcast to f32; `sigmoid`/`exp` in fp32; the `/4` on the down product and the final `/4` mean are part of the contract (a missing `/4` is a 4x error, exactly the class F-82 hunted).

**(B4) Attention block.**
- `kcur = q81(mixed) @ k_proj^T` ([2,256]); `vcur = q81(mixed) @ v_proj^T` (mtp.cpp:967-970).
- K: per-head RMS over HD with `k_norm` (f32, folded), then **partial rope**: pairs `i < 32` of each 256-row rotate `(x[i], x[i+32])` by `theta = pos * 1e7^(-2i/64)`, no scaling (artifact ships no rope.scaling; `native_rope.cu:46-59, 205`: `out[i] = a*c - b*s; out[i+32] = a*s + b*c`; pairs >= 32 pass through). Note the rotate-half is INSIDE the 64 rotated dims (`x[i]` with `x[i + n_rot/2]`), not across the full 256.
- Append (K,V) for the draft row at `cell` into the layer's own KV state — INT8 on this lane: `scale = fp16(max|x|/127)` per 64 values, `code = clamp(rint(x/scale), -127, 127)`, storage dequant `fp16(code*scale)` (`kv_q8.hpp:5-12`; append at mtp.cpp:997-1013). **The G0-exact module quantizes its K/V with this exact rule before attention.**
- Q: `qfull = q81(mixed) @ q_proj^T` ([24, 512]) — per head `[q(256) | gate(256)]` (F-86 verified; mtp.cpp:1047-1059). q = the FIRST half; `rms(q) * q_norm` (over HD) then the same partial rope at `pos`. The gate half is NOT normed, NOT roped.
- Scores: GQA 24 query heads over 2 KV heads (12:1), `s = (q . k_dequant) / sqrt(256)` (qsa_decode_attn.cu:764,897), softmax **online (flash split-K: chunk max/sum partials, log-sum-exp merge, fp32)** over the causal cell set `[0, cell]` (the draft row's own K/V was just appended; ring/window masking via `window_ids`+`ident_`, mtp.cpp:1062-1066 — with `--kv-resident 32768` a cell is visible iff `0 <= c <= cell` and `c > cell - 32768`; at short positions that is plain causal). `attn = sum p_dequant * v_dequant` in fp32.
- Gate: `attn32 = attn * sigmoid(gate_half)` elementwise per head/channel (`native_qsa.cu:94-102` — plain sigmoid of the RAW gate logit, no scaling, no norm).
- `bo = attn32 @ o_proj^T` ([2560]; o_proj rows are out=2560 x in=6144) (mtp.cpp:1067-1069).

**(B5) MLP block.** HC read #2 (B3, apply=True, inputs `R_=R'` after the attention fold, `bo_prev=bo_`, `inj_prev=inj_`): yields `R_` (attention written in), `mixed_mlp`, `inj2`.
- Router: `logits = mixed_mlp @ mlp.gate.weight^T` — bf16 weights x fp32 activations, fp32 accumulate (`bf16_gemv_fp32_mmvf`, mtp.cpp:1145-1146) -> `p = softmax(logits)` over ALL 512 in fp32 -> `ids` = stable argsort descending, ties by ascending index -> `w = p[ids]`, `s = clamp(sum(w), min=2^-14)`, `w = w/s` (`router_top10.cu:9-21` — the ggml gather-form semantics; the clamp is part of the renorm). **This answers F-84's question: the engine renormalizes the top-10 weights exactly as the torch `top_w/top_w.sum()` did, with the 2^-14 floor.**
- Routed experts (mtp.cpp:1157-1161): activation `mixed_mlp` quantized `quantize_q8_0_scaled` (32-blocks, fp16 d in bytes + **fp32 scale sidecar** — R4.2h, the GPU hit path's multiplier); for each of the 10 selected blobs: `gate/up` rows from the interleaved Q2_0 codes (dequant `x = d * code_value`, code_value = 2-bit code per ggml Q2_0 reference — use gguf-py `dequantize` on the reconstructed (1280,720)-bytes-per-row view), `h = silu(gate) * up` (s2_expert_grouped.cu:169-174), `h` re-quantized Q8_0, `down` projection -> `parts[k]` ([2560]).
- Shared expert (mtp.cpp:1089-1134): `h_s = silu(x @ gate_proj^T) * (x @ up_proj.T)` then `@ down_proj^T` (Q8_0/Q4_0 MMVQ under `--mtp-q4 all`), `g = sigmoid(dot(bf16(mixed_mlp), shared_expert_gate_bf16))` — ONE scalar per token, both operands BF16 (the contract, `shared_expert.hpp:5-16`, `.cu:96-104`), `shared = h_s * g`.
- Combine: `y = sum_k w[k] * parts[k] + shared` (native_moe.cu:38-62, plain add of the pre-gated shared; mtp.cpp:1173-1184).
- HC write: `R_ = R_ + y * 2*sigmoid(inj2/4)` per stream (`gr_write_kernel`, gr.cu:283-297; called mtp.cpp:1180-1188 — every stream adds the SAME block output, only the per-stream weight differs).

**(B6) Final mixer + head.** HC read #3 (B3, apply=False, no inject) -> `sample` ([2560]) -> `q81(sample)` -> `head_logits = q81 @ head_bytes^T` over the 40,525-row Q4_0 subset (mtp.cpp:1223-1227) -> argmax (greedy requests; `argmax_rows` mtp.cpp:1252-1253) mapped through `dvocab_` to real token ids (mtp.cpp:1262). The TAP logs `head_logits_` pre-map.
- Indexer: **the drafter does NOT run the indexer at serve.** `self_attn.indexer.*` is loaded in dense.bin but the layer's own K/V is read directly: "the layer's own K/V (dense attention: no indexer state is read)" (mtp.cpp:368) and no indexer kernel appears in record_front/record_rest. The MQA 4/1x128 top-2048 machinery belongs to the trunk's QSA layers. The splice tool (`mtp_indexqk_splice_r17.py`) exists only so the trained GGUF carries the tensors the loader requires.

### 2.3 Precision contract

- Storage dtypes are the CONTRACT: f32 norms, bf16 mixers/router/gate, q8_0/q4_0 projections, q2_0 expert blobs, int8 KV. The module stores weights as the engine's BYTES and dequantizes at load to fp32; the G0-exact forward re-applies the engine's activation quantizers (q8_1 everywhere a projection runs, q8_0 at the expert boundary, int8 on K/V) so the arithmetic sees the same integers.
- All accumulations fp32. Reduction-order freedom gives <= ~2e-6 relative on 2560-10240-term dots; that is the noise floor every threshold below is set against.
- The TRAINING forward must NOT use the rounding quantizers (roundf kills gradients): see section 5.4 for the two-mode design and its stated risk.

---

## 3. THE G0 HARNESS PROTOCOL

### 3.1 Boot (copied from the standing lane; the only additions are env)

Never touch the production lane. One diagnostic boot on a free slot, same build, same args as `campaign/adaptmtp-iq3_s-8081.json` plus:

```
STRATA_MTP_TAP=/home/svg/LLM/Custom-data/mtp/parity/tap     # the only new env
```

The tap is production-inert without it (mtp.cpp:77, 288-294, 1232-1239). Do NOT add `--pipeline-windows 2` (it would force max_t=8 + capture chains — harmless to the tap but changes the boot shape; keep the lane shape).

### 3.2 What the tap dumps

- At load, when `STRATA_MTP_TAP` is set: pinned host buffers are allocated OUTSIDE capture, `g_tap_nv = 248,320` (mtp.cpp:286-294).
- At record_rest (T=1, every draft application), three `cudaMemcpyAsync` D2H copies are issued on the capture stream so they become GRAPH NODES firing on every replay (mtp.cpp:1234-1238): `R_` first 4*2560 floats (the draft row's post-layer residual — the final mixer's input), `sample_` 2560 floats (post mixer), `head_logits_` nv floats (nv = 40,525 with the en sub-vocab).
- Once per process, at the first request's per-request tail, `mtp_tap_dump_once()` writes `<prefix>_R.f32` (40,960 B), `<prefix>_sample.f32` (10,240 B), `<prefix>_logits.f32` (**993,280 B = 248,320 floats, of which ONLY the first 40,525 are valid; the tail is stale allocation** — mtp.cpp:75-88, esp. the comment at :86) (generate.cpp:13618-13620).
- The three copies are issued in ONE record_rest call, so the three files are mutually consistent snapshots of one application, whichever round survived. Which round it is (token/pos) is NOT recorded — the T1/T2 diffs below do not need it.

### 3.3 Probe protocol (the five capture laws baked in — they are already fixed in THIS build)

The Windows-era tap laws (F-87..F-91) are all satisfied by the current implementation (pinned buffers at load, graph-node copies, once-flag host dump). The protocol inherits only the operational rules:
1. Boot warmup runs zero-value windows (F-87 law 1): after boot, issue ONE short real request; then **check `std(R_) > 0` before diffing**; if zeros, issue another request and re-check (the dump is once-per-process — if the once-flag fired on a warmup window, re-boot; in practice the flag fires at the first REQUEST tail, which follows real traffic).
2. One request is enough; short prompt keeps the KV reconstruction in section T3 small.
3. Record the boot log line `strata mtp: draft head over 40525 tokens` and the `--mtp-q4: draft head ... as Q4_0` line to pin `dhead_type_`.

### 3.4 The comparison script (new, torch-side; ~150 lines)

`tools/mtp_parity_diff.py` (proposed path):
- Inputs: the three `.f32` files, `dense.bin`+`dense.txt`, `experts.bin`, `draft_vocab.bin`, the GSQ GGUF (for `token_embd` + `output.weight` rows), `--n-vocab-sub 40525`.
- T1 mixer test: `mixed_pred = hc_read(R, hc_norm_mixer, down, up, inject=None, apply=False)`; report `max_abs, max_rel, cos` vs `sample.f32` and the argmax-through-head agreement.
- T2 head test: `logits_pred = mmvq(q81(sample_pred_or_tap), q4_0_head_bytes)`; compare the first 40,525 floats of `logits.f32` ONLY.
- T3 full-layer test (needs a dumped-window probe; see 3.5): reconstruct the whole window in torch and diff `R_pred` vs `R.f32`.
- Output: a per-stage table (the F-82 "std ladder" columns) + PASS/FAIL against the thresholds below.

Thresholds (fp32-rounding-consistent):
| stage | metric | gate |
|---|---|---|
| q81 codes, weight bytes, router ids/top-10 set, argmax | bit-exact | required |
| rs values (every RMS) | max rel | <= 1e-6 |
| B3 read outputs (lo/inj/mixed) at T1 | max rel vs tap | <= 1e-6 (R17 measured **rel-err 0.000000, cos 1.000000**, F-95) |
| head logits | max rel | <= 5e-5 (2560-term Q4_0 dot; fp32 order only) |
| full layer T3 vs tap R_ | max rel | <= 1e-2 and cos >= 0.9999 (the F16 dump-input noise floor dominates; see 3.5) |

R17 precedent numbers: the tap diff went rel 0.58/cos 0.816 (sidecar weights) -> 0.46/0.918 (read realigned) -> **0.000000/1.000000** (rt weights + source-exact read) (F-92, F-95). Any residual ~1e-2 with cos ~0.999 at T1 today means a WEIGHT PROVENANCE bug, not a math bug (F-94's exact signature).

### 3.5 Scoping law (F-96) — what G0 can and cannot prove

The engine's drafter attends over the ACCUMULATED session KV; a torch forward that builds KV from the window's own rows can never match a served window's `R_` end-to-end **by construction** (F-96; the 0% argmax mystery of the R17 harness was exactly this). Therefore:
- **T1+T2 are the G0 gate** (binary, ~one boot, minutes): they need only the tap files and close bit-tight. F-95 already proved the method once.
- **T3 is a controlled extra**: boot the tap on a probe whose prompt IS the dump corpus's first chunk (tokens + the dump's per-position h rows = the same residuals the engine's prefill consumed). Torch rebuilds prefill K/V (int8-quantized with the B4 rule) + the draft round, then diffs `R_pred` vs `R.f32` at the ~1e-2/cos 0.9999 band (the dump rows are F16-quantized per T14A's lane design — if the R17 dump stored f32, tighten to 1e-3). T3 validates B1-B5 end to end; a T3 miss localizes via the section 7 ladder, never by eyeball.

---

## 4. THE R17 BUG LIST — every fix baked in as a checkbox (SVG-CUSTOM.md F-77..F-118)

Module math:
- [ ] **hnorm STREAM = per-stream grouped RMS**, each 2560-slice scaled by ITS slice of the 10240 gamma — not a pooled RMS, not `rms(h)*gamma(h)` twice (F-83 A; engine mtp.cpp:941-942; fixed in F-84: "fuse 0.44, hc_attn 1.00").
- [ ] **HC read geometry = the multi-kernel's**: xn from raw-R' per-stream RMS x gamma; `lo = silu(w_down@xn/4)`; inject = `w_inject@xn`; gate at the UP stage: `mixed = mean_c(rv * w_norm * rs * sigmoid(w_up@lo))` with `rv = R' + bo*2*sigmoid(inj/4)` when apply (F-83 B, F-92 — plain-rms down input, gamma at the up stage, cos 0.816 -> 0.918; F-93 — the multi kernel is NOT the single-row variant's geometry as originally assumed).
- [ ] **No double RMS anywhere in the read** (the torch `xn * rsqrt * norm(xn)` pattern was wrong; F-84 item 2).
- [ ] **`2*sigmoid(inj/HC)` UNDIVIDED is wrong; the engine divides by HC=4** in BOTH the read gate and gr_write (F-83: suspect 2 exonerated by fused_gr.cu:86; gr.cu:288).
- [ ] **Fuse dialect = split-and-add** (`fc_embedding` on the normed embedding + `fc_hidden` on the normed residual, ADDED per stream) — never a single `eh_proj(2560,5120)` on `[e;h]` concat (F-80 bug 2, F-81 fix 2).
- [ ] **Shared-expert gate = sigmoid(dot(bf16 x, bf16 w)) = ONE scalar per token**, multiply AFTER down (F-81 fix 4; shared_expert.hpp:5-16).
- [ ] **bf16 softmax cast** in any attention reference path (F-81 fix 3) — ours is fp32 flash; keep it fp32.
- [ ] **Router: softmax over ALL 512, gather top-10, `w / clamp(sum, 2^-14)`** (F-84 item 1 resolved by router_top10.cu:9-21).
- [ ] **Off-by-one: the loader yields W+chain+1 tokens; derive widths from the passed T, not the raw record** (F-80 bug 1, F-81 fix 1 — trainer-side but it burned a smoke run).

Weight dialect:
- [ ] **All weights from the RT (dense.bin/dense.txt/experts.bin), not the sidecar** (F-92 isolate, F-94 proof: mixer down maxdiff 2.65/mean 0.028, up 14.3/0.176 vs sidecar; F-95 exact only after rt sourcing).
- [ ] **Gemma +1: dense.bin's f32 norms ALREADY carry `1+w`; load as-is** (F-109 — the double fold shifted every norm by +1; F-111: fixing it alone was NOT sufficient; F-112: indexer layernorms proven `1.2305 vs 0.2305`, patched stock-exact).
- [ ] **Under `--mtp-q4 all`, the engine's projections and head are the Q4_0 re-quants of dense.bin's Q8_0 (and the head of Q6_K); mirror the requant bytes** (mtp.cpp:211-269; the lane flag, adaptmtp json).
- [ ] **experts.bin = the stock blob bytes; the pack's Q8_0-sidecar -> q2_0 re-quant path was the final corruption (0/462 -> 287/336 = 85.4% on the swap test)** (F-113, F-114).
- [ ] **draft_vocab.bin must be present in the rt dir (4 files, not 3)** — a missing map mis-routes every draft token with perfect weights (F-112).
- [ ] **Pack dtype law: only `*norm.weight` tensors take f32; hc mixers/router/shared_expert_gate stay bf16; watch rt alignment for the bf16 gemv tensors** (F-100..F-103: hc_norm bf16-vs-f32 flips regressed QSA; router alignment "null or misaligned pointer"; the (out,in) transpose convention — write (out,in) as-is, the chain flips itself, F-100).

Harness laws (already satisfied in-tree; do not regress):
- [ ] Tap buffers pinned at load, outside capture (F-87 law 2).
- [ ] No memcpy on the capturing stream; no stream creation during capture; copies as GRAPH NODES, host dump once per process at the request tail (F-88, F-89, F-91; implemented mtp.cpp:1234-1239, generate.cpp:13620).
- [ ] Zero-value warmup windows exist; gate diffs on `std(R_) > 0` (F-87 law 1).

Chain/regime laws (they bound what parity MEANS):
- [ ] **Full-window parity vs a SERVED session is impossible from window rows alone** (session-KV accumulation; F-96). G0 gates on T1/T2; T3 uses a controlled reconstruction.
- [ ] The stock round-trip floor through the quantized chain is ~85.4%; a trained head first reads ~57% pure-KL (F-115) with the plateau attributed to double-quant + fresh-init + pure-KL (T17C) — these are G1 concerns, NOT module bugs; do not "fix" the module over them.
- [ ] Vocab layout: the target's vocab is the GSQ/strata layout; any emb/output row sourced outside the GSQ GGUF is permuted (F-85 a).

---

## 5. THE DATA IN — a training window as module inputs

### 5.1 Record -> tensors

One dump record position = `(token:int32, h:[4,2560] f32 (the target's HC residual — the drafter's Rin), pos:int32)`. A window of W positions (256..1024) yields:
- `Rin[t] = h[t]` for t in 0..W-1 (the B2 input),
- `tokens[t]` (the B1 input and the CE teacher tokens),
- positions `pos[t]` (rope; real serving-range positions — F-105's POS_MAX lesson: spread over the realistic 0..8192+ band, not origin-only, not 0..250k),
- the GT chain: for K=6 supervised outputs, targets are `tokens[t+1..t+6]`.

### 5.2 The K=6 recursive chain (matches the engine's draft chain)

The engine (mtp.cpp:1359-1387 + `mtp_select`, verify_kernels.cu:656-670): after each application, `Rin <- R_` (the module's own updated 4x2560 residual) and `token <- argmax(head_logits)`; cell advances by 1; rope uses the running position. The trainer reproduces this with `--spec 6`'s shape — **K=6 = 1 window application + 5 recursive applications** (max_steps = max_t_-1 = 5 with max_t_=6), each supervised by CE against the GT next token:
- step 0: `(token=t_i, Rin=h_i)` -> CE vs `t_{i+1}` (window CE);
- step j (1..5): `(token = t_{i+j}` [teacher-forced] `or argmax` [recursive], `Rin = module's R_` from step j-1`) -> CE vs `t_{i+j+1}`, decay-weighted (R17: chain CE learned fastest, 1.8-3.0; window CE ~8.6, F-97).
Loss = window CE + sum_j decay^j * chain CE (the FastMTP recursive-aligned recipe; the KV is the mixed real/recursive construction below). Sample the teacher-forced vs self-fed token choice per batch (start teacher-forced: the paper's stable variant; the engine's self-fed geometry is approached as K grows).

### 5.3 KV construction (the stated approximation)

The trainer builds the drafter's KV from the WINDOW ROWS themselves (each window position's K/V appended at its real position) — the "mixed real/recursive KV" of the v5 design. **The engine drafts against the accumulated session KV; this is an approximation BY DESIGN** (F-96), and the R17 failure autopsies (F-104: trained head 0/428 while the stock round-trip read 27.8%) are the standing warning that conditioning drift, not module math, is what kills acceptance. Mitigations already queued elsewhere: distillation through the SAME window-KV forward (the KV error cancels between student and teacher, F-109/Q3 insight, proven 0 -> 57.4%, F-115), on-policy rejection replay (T17F), and the T14A prefill-dump lane. P2's job ends at G0; this risk is inherited, not owned.

### 5.4 PLE/ngram: NOT modeled — stated choice

The PLE path (`prefetch_ple`, mtp.cpp:1619-1628, 1827-1833) only warms reads: `ngram_rows -> ple.table->prefetch_rows` — no draft logit, token, or probability passes through it. **Choice: the trainer does not model PLE. Risk: none for parity or acceptance; it affects expert-cache warmth (throughput) only.**

### 5.5 Two-mode forward (stated choice + risk)

- `quantize_mirror=True` (G0): every projection sees the engine's q8_1 activation image and the byte-dequantized weights; exact enough for section 3 thresholds.
- `quantize_mirror=False` (training): smooth bf16/fp32 matmuls over the same dequantized weights; no roundf. **Risk (named): the head trains against the smooth forward but is served under the quantized forward — the ~1e-2 per-projection quant noise the engine itself carries.** Bounded precedent: the stock weights survive the same noise at 85.4% (F-114); the distill arm trains both sides under one geometry; the F16-preservation chain upgrade (T14B/T17C) attacks the residual gap. Do NOT try to backprop through roundf.

---

## 6. VRAM / SPEED BUDGET (forward+backward)

Counts from section 2 shapes; trainable set = dense projections + mixers + router + shared expert + norms (~82.5M params) + head (40525x2560 = 103.7M) ~= **186M params**. Frozen: token_embd table (248,320x2560; bf16 1.27 GB or the GGUF's IQ4_XS ~0.64 GB) + experts (frozen by law, F-114).

| item | bytes |
|---|---|
| trainable bf16 (module 165 MB + head 207 MB) | ~0.39 GB |
| AdamW fp32 m+v (8 B/param) | 1.49 GB |
| fp32 master weights | 0.75 GB |
| frozen emb (bf16) | 1.27 GB |
| frozen experts — bf16 dequant / q2_0 bytes + on-the-fly | 5.03 GB / 0.71 GB |
| **weights+optimizer subtotal** | **~8.9 GB (q2_0 experts) .. 13.2 GB** |

Activations (fp32 saved for backward, per position per application): layer body ~55 KB (R/xn 10 KB each, qfull 12 KB, attn 6 KB, mixed 3x, parts 10 KB, misc) + head logits+softmax ~324 KB = **~380 KB**. Chain K=6 (serial): ~2.3 MB/position.

| batch (B x W) | activations (est.) | total VRAM | verdict on 32 GB (5090) |
|---|---|---|---|
| 2 x 256 | ~1.2 GB | ~11-15 GB | easy |
| 2 x 1024 | ~4.7 GB (+autograd softmax copies ~2x logits -> ~7-9 GB) | ~17-22 GB | fits; chunked CE already assumed (F-81) |
| 4 x 1024 | ~9.5-18 GB | ~19-31 GB | tight: chain-step checkpointing (recompute steps 1..5 in backward) or grad accum 2x |

FLOPs: ~242 MMAC/position/application (head 103.7M = 43%, experts top-10 49.2M, qkvo 50M, fc 13.1M, mixers 19.7M, shared 4.9M, router 1.3M); fwd+bwd+recompute ~2.0-2.5 GFLOP/position-application; K=6 serial per window row.
- B2x256: ~6 TFLOP/step -> **0.1-0.3 s/step**; B4x1024: ~48 TFLOP -> **1-2 s/step** (30-50% MFU; small-GEMM + serial-chain overhead).
- R17 measured anchor: 1500 windows, batch 2, 3 epochs = **112 s** of 5090 time, ~7 steps/s (F-97) — iteration is cheap; the budget above is consistent with it at smaller W.

---

## 7. THE TEST LADDER (cheapest first; nothing below needs a boot until E5)

- **E0 bytes**: load dense.bin via dense.txt; assert offsets/sizes/rows/cols against section 2.1's table; assert all f32 `*norm.weight` tensors; assert `draft_vocab.bin` = 40,525 ids; assert experts.bin size = 512*1,382,400. (Catches the F-112 four-file class and any pack drift.)
- **E1 dequant spot checks**: 16 random rows per tensor vs gguf-py `dequantize` of the packed GGUF sidecar — q8_0 tensors match to the fp16-scale round trip (<= 1e-7 rel); Q2_0 expert rows reconstruct bit-exact from blob bytes; Q6_K head rows dequant, requant Q4_0, compare against the boot's own `--mtp-q4` bytes if available.
- **E2 unit ops vs kernel semantics** (torch vs the source formulas, fixed vectors): rms (eps placement: `rsqrt(mean+eps)`, gamma after), rms_grouped (slice gamma), rope (pairs inside n_rot, base 1e7), q81 (roundf-half-away, clamps), router (gather-form + 2^-14 clamp + stable ties), gr_write gate, gate_apply (raw sigmoid). Parity fixtures already exist in-tree to copy vector shapes from (`src/kernels/gr_parity.cpp`, `rope_parity.cpp`, `router_top10_parity.cpp`, `elementwise_parity.cpp`).
- **E3 std ladder** (the F-82 recipe, no engine): run the module on one dumped window; per-stage output std must be ~0.5-3.0 band (fuse 0.44, hc_attn 1.00, attn 2.10, hc_comb 2.19, moe ~1, flat/head_in ~1). A 3-10x hot stage names the bug class without a boot.
- **E4 mixer+head closed-form**: hc_read on random R -> known-formula cross-check in float64 (rel <= 1e-6), then head mmvq vs a naive fp32 loop (<= 5e-5).
- **E5 G0 TAP T1/T2** (section 3.4): the binary gate. Then optional E6 T3 full-layer (3.5).
- **E7 chain sanity** (no boot): K=6 teacher-forced roll on a dumped window; loss ~uniform (12.42 nats over 40,525? — no: CE over the SUB-VOCAB at init ~10.5; the stock module should read window CE ~ the R17 stock ladder's ~1-3 band on in-distribution data, F-82's "its true CE should read ~1-3"). Only a coarse tripwire — G0 is the gate.

---

## 8. Contradictions and resolutions

- **"The multi kernel may differ from the single-row math" (F-93) vs "gr_multi_parity checks bitwise"**: resolved by the in-tree contract comments — all multi variants compute each output with the plain read's operations in its order and are bitwise-checked (`fused_gr.cu:691-711`, F-217 note at :323-328); F-93's doubt was about the TORCH transcription, resolved by F-92/F-95's realignment + exact result. The spec transcribes the multi path.
- **Draft chain length "K=6" vs "7 drafts"**: `kVerifyMaxT=8` is the ceiling used by `--pipeline-windows 2`; the lane runs `--spec 6` -> max_t_=6, max_steps=5 drafts. The spec's K=6 = the 6 supervised positions (window + 5). If a future boot changes `--spec`, K follows `max_steps + 1`.
- **Task premise "Q4_0 sub-vocab head"**: true ONLY under `--mtp-q4 all/head` (the lane's setting, adaptmtp json). Without it the bound head keeps the native head's format (Q6_K for GSQ). The parity boot pins and logs `dhead_type_`.
- **Task premise "indexer MQA 4/1x128 top-2048"**: those tensors exist (inventory #19-21) but the drafter never runs the indexer at serve (mtp.cpp:368). The module excludes it; the splice keeps the tensors for the loader only.

## 9. Open questions

1. The R17 dump record layout (buun `_nextn__` magic) is not in this repo — the trainer/export spec (T14B-exec-C) owns the reader; P2 assumes `(token, h[4,2560], pos)` per record and f16 vs f32 precision is unconfirmed (sets the T3 band).
2. Whether `mtp_tap_dump_once`'s once-flag can land on a warmup round of the FIRST request's cache fill (needs one live boot to confirm; mitigation in 3.3.1).
3. Shared expert activation under `--mtp-q4 all`: whether the native Q4_0 projections consume the shared `xq_` q8_1 image or a fresh quantize of `mixed_` (mtp.cpp:1093 reuses `xq_`; the exact aliasing is E2-verifiable but does not change the math contract).

## 10. Implications

- P2 is buildable entirely from in-tree artifacts; no engine change, no boot is required to WRITE it, and one (1) diagnostic boot closes G0 via T1/T2. F-95 already proved the T1 method end-to-end once.
- The module is also the trainer's forward (two-mode flag), so every byte-level decision here propagates into T14B-exec-C's export/graft: same weight sources, same fold convention, same K.
- The remaining acceptance risk after G0 is the named regime risk (5.3) and the chain double-quant cost (F-116) — both owned by later gates (G1+), not by P2.

## 11. Sources

Engine (all read 2026-10-18, repo @ 315919bc): `src/core/mtp.cpp` (TAP 68-88, 286-294, 1229-1239; forward 902-1269; chain 1359-1387, 1562-1775; head bind 690-781; q4 210-269); `src/kernels/cuda/fused_gr.cu` (read math 81-429, variant contract 691-711, 1045-1091); `src/kernels/cuda/gr.cu` (norm 126-161, down 163-230, gate 232-257, inject 259-281, write 283-297); `src/kernels/cuda/native_qsa.cu` (norm 44-68, grouped 70-93, gate 94-102); `src/kernels/cuda/router_top10.cu` (semantics 9-45); `src/kernels/cuda/native_moe.cu` (combine 34-72); `src/kernels/cuda/shared_expert.cu` + `include/strata/kernels/shared_expert.hpp` (contract 1-16); `src/kernels/cuda/native_rope.cu` (44-59, 205, 231); `src/kernels/cuda/native_mmvq.cu` (155-170); `src/kernels/cuda/qsa_decode_attn.cu` (scale 764, 897) + `include/strata/kernels/qsa_decode_attn.hpp` (14-28); `src/kernels/cuda/verify_kernels.cu` (mtp_select 656-670) + `verify_kernels.hpp:21`; `include/strata/core/geometry.hpp` (24-64); `include/strata/kernels/q8_1_finite.hpp` (16-31); `include/strata/kernels/kv_q8.hpp` (5-26); `include/strata/kernels/cpu/expert.hpp` (13-60); `include/strata/core/mtp.hpp` (61-234); `src/program/generate.cpp` (917, 2223-2229, 2660-2667, 4710-4736, 13618-13620); `tools/mtp_rt.py` (1-96); `tools/_paths.py`; `src/kernels/*_parity.cpp` (fixture inventory).
Artifacts: `/home/svg/LLM/Custom-data/mtp/mtp-inventory.md` + `.json` (31 tensors); `/home/svg/LLM/Custom-data/mtp/rt/{dense.bin,dense.txt,experts.bin,draft_vocab.bin}` (sizes read); GSQ GGUF metadata (rope.freq_base 1e7, sections, indexer dims); `campaign/adaptmtp-iq3_s-8081.json` (the lane boot); `campaign/research/{INDEX.md, T14A, T14B, T14C}`.
Journal: `SVG-CUSTOM.md` F-77..F-121 (lines 2443-3278).

## 12. Methods note (what did NOT work / where search ended)

- No tap-diff python, `std_ladder.py`, `verify-parity.py`, or `train-fastmtp-v5.py` exists in THIS repo (Windows/buun-era artifacts): the comparison script of 3.4 is a fresh design, and the dump reader belongs to exec-C. Searched: repo-wide filename/content greps for `STRATA_MTP_TAP`, `_R.f32`, `std_ladder`, `dump-nextn`, `_nextn__` — only the engine C++ hits.
- `verify.cpp` contains no direct draft-consumption lines (the window assembly lives in the verify kernels/spec functions); the chain semantics were taken from mtp.cpp's draft()/capture_step/mtp_select instead.
- `coupled_draft_cell` was not opened; under greedy requests the coupled path is inactive and drafts are plain argmax (coupled_active_ requires non-greedy, mtp.cpp:638-643) — G0's greedy diff is unaffected.
