# T26-3 — PREFILL & THE WINDOW/LAUNCH ARCHITECTURE: the F-252 matmul inventory, the launch-map diet, the PDL chain, and the 6-7k plateau

2026-10-21. Read-only study (CPU-only: code, greps, banked logs; the machine was live on 8081 — no builds, no boots, no GPU work, no source edits; the only file written is this one).
Trees: ours `/home/svg/LLM/strata-custom-2` (standing boot = `launch-custom-iq3s-8081.sh` + `custom-iq3_s-8081.json`, IQ3_S native pack, 48-layer Qwen3.8-Flash-Next, `--kv int8 --kv-resident 32768 --prefill auto:32768`), basalt read-only at `/home/svg/LLM/data/references/basalt`.
Read-first dossiers not repeated: T25-5 (window anatomy), T25-4 (prefill/quant), T13A (decode critical path), F-200 agent briefs, F-204 (node inventory). F-252 (SVG-CUSTOM.md:8550) is the law under test: **fp16-accum mma = 476 TFLOPS, fp32-accum (fp16 or bf16 in) = 238, ratio 1.99/1.97, measured on our 5090** (`campaign/mma-accum-ratelock.cu`, contended-card ratios).
Geometry (prefill.cpp:131-132): N=2560, HC=4, D=10240, LR=320, K=10 experts, NE=512, C=10240, ZV=6144, HV=48; head 2560×248320.

---

## (a) THE MATMUL INVENTORY — every GEMM/WMMA/MMQ in the prefill and decode path, by accumulator

### (a.1) PREFILL (per 32,768-token chunk unless noted; the phase table is the banked 32k anatomy, campaign/results/knob-pft-engtail.txt:380-384 = F-239)

| # | op (shapes, per layer ×48) | file:line | pipe / accumulator | K | bit-identity contract? | fp16-accum convertible? |
|---|---|---|---|---|---|
| P1 | Native dense projections, Q8_0/K-quant weights dequant→fp16: attn_qkv 10240×2560 (GDN 24L), attn_gate 6144×2560, ssm_out 2560×6144, shexp gate/up 640×2560 ×2, shexp down 2560×640; QSA 24L: attn_q 6144×2560, attn_k/v 512×2560, attn_output 2560×6144, indexer k 128×2560 / q 512×2560 | prefill.cpp:2437 `native_proj` → gemm.cu:1251 `Gemm::native` → dequant_f16 + gemm.cu:1119-1141 `Gemm::f16` = cublasGemmEx CUDA_R_16F, CUDA_R_32F out, **CUBLAS_COMPUTE_32F** | fp16-in / **fp32-accum** → the **238 TFLOPS** lock | 2560 / 6144 | NO bitwise contract on this path: prefill already runs non-bitwise classes (PF_FUSED KL 2e-4, NVFP4 W4A4 first-token KL 0.0004-0.03 default, TC block scores "FP32-level not bitwise", GDN_CHUNKED sum-order). Only the *acceptance* contract (F-230) binds | **YES — the actionable core.** Inputs already fp16; flip COMPUTE_32F→COMPUTE_32F_FAST_16F (or custom kernel). Error: HMMA f16-accum rounds once per k=16 tile → 80-640 roundings; RMS rel error ≈ 2·10⁻⁴·√K/√16·√16… ≈ **0.5-1% at K=2560, ~1.6% at K=6144, ~2% at K=10240** (random-walk estimate, marked ESTIMATE); chunked fp16-accum (flush to fp32 every 256 k) cuts it ~3.2x. Same acceptance class as the already-default W4A4. Gated by the V1 prompt canary + F-230 tax (see (e)) |
| P2 | BF16 resident projections: hc_attn/ffn down 320×10240, up 320→10240 (N=320-side: wu ne0=LR), inject 4×10240, ×2 halves; ssm_alpha/beta 48×2560; router ffn_gate_inp 512×2560; shexp scalar gate 1×2560 | prefill.cpp:2445 `bf16_proj` → gemm.cu:1052-1056 = cublasGemmEx CUDA_R_16BF, CUDA_R_32F out, **CUBLAS_COMPUTE_32F**; (gfx11-only alternates: wmma_gemm.cu:492 `kernel_hcdown`, :587 `kernel_hcd_exact` — never run on sm_120, the STRATA_WMMA_GFX11 guard compiles them out) | bf16-in / **fp32-accum** → 238 lock | 10240 / 2560 | NO (same class as P1); hc outputs feed gr_mix into the residual — KL-gated | **YES.** bf16→fp16 is exact for every value in fp16 range — this exact conversion lane already exists in-tree (gemm.cu:667-670 comment, bf16_to_f16 :101, used below sm_80): "weights and normalized activations sit there". Then FAST_16F. Largest-K risk: hc up/down at K=10240 (~2% rel, chunked-accum mitigation). Range assert per tensor is one kernel |
| P3 | Routed experts MMQ: gu 1280×2560 + down 2560×640 per (token,10-of-512) rows, IQ3_S/IQ2_S/IQ3_XXS tabs | prefill.cpp:3571 `use_mmq = mmq_plan().any` (default on, :1163-1165) → moe_mmq.cu (+ third_party/llama.cpp mmq.cuh; NVFP4 W4A4 default sm_120 variant mmq_nvfp4_w4a4.cu) | **INT8/dp4a/IMMA, int32 accum** — the integer tensor pipe (~2× the fp32-accum fp16 pipe), NOT the F-252 pipe | 2560 | byte-level pack contract (COHORTS); FUSE_HITS ZADD bitwise | **NO — not affected by F-252 and not convertible** (already on the faster pipe). Measured effective ≈ **185 TOPS** (832 ms for ~154 TFLOP, derived — ESTIMATE) |
| P4 | QSA prompt attention, standing `--kv int8`: scores+PV int8 codes, per (query tile, KV head) | qsa_prompt_attn.cu:539 `prompt_attn_i8_kernel`, mma m16n8k32 **s32.s8.s8.s32** :1093, :1104 | **INT32 accum (IMMA)** — full int pipe | 256 | selection output contract only (same cells read) | NO (not the fp32 pipe) |
| P5 | QSA prompt attention, fp16-KV variants (non-standing) | qsa_prompt_attn.cu:57-64 m16n8k8/16 **f32.f16.f16.f32**; :1483 m8n8k4 f32.f16 | fp16-in / **fp32-accum** (238 lock) | 256 | not bitwise (header) | YES technically, but non-standing — skip |
| P6 | QSA block-selection scores (topk logits) | qsa_select.cu:194, native_qsa_score.cu:52 m16n8k8 **f32.tf32.tf32.f32** (raw F32 bits, "deliberately no cvt.rn.tf32") | tf32-in / fp32-accum — tf32 runs at ~half the fp16 rate → effective lock ≈ **119 TFLOPS**-class | 128 | selection changes = different cells = output-changing | **NO (recommended)** — truncating inputs to fp16 flips borderline selections; the win is small (select 139 ms = 3.2%) |
| P7 | GDN recurrence, standing CUDA (STRATA_GDN_CHUNKED=1): chunked WY | kernels.cu:1174 `gdn_rec_chunked`, prep :892 / scan :991 — **FP32 CUDA-core, no tensor cores** (the WMMA chunked variant kernels.cu:1935 `gdn_rec_wmma` is __HIPCC__/gfx11-only, :2616; gdn_chunk.cu is sm_70-only, header "no tensor cores yet") | CUDA-core fp32 FMA — **not on the tensor pipe at all** | 64-chunk dense sub-matmuls | F-224: different FP32 sum order, KL 3e-7-class, M3b in-band | Different lever: TC-BF16 GDN (basalt gdn_chunked.cu design, T25-4 #7) — bounded by their 1.5e-2 worst-element parity harness; GDN-perturbation-sensitive per SVG-CUSTOM-TODO.md:1034. **This is a separate candidate, not an accum flip** |
| P8 | Head (output.weight 248320×2560) + token_embd paths | Gemm::native (as P1), once per prompt | fp32-accum (238) | 2560 | none (greedy next-token in prefill) | YES, minor (1.27 TFLOP/prompt ≈ 0.5 ms) |
| P9 | glue: gr_norm/gr_silu/gr_mix, gdn conv+gates, rope, rms, swiglu, combine (moe combine multi + hits) | kernels.cu, fused_gr.cu, prefill.cpp:4698 kPfCombine region | CUDA-core fp32 | — | various | n/a (no mma) |

**Inventory verdict:** on the CUDA/5090 build, exactly two prefill families sit on the half-rate fp32-accum tensor pipe: **P1 (fp16 dequant-GEMM dense) and P2 (bf16 hc/router/gates)**. Together with P8 they are ~200-235 TFLOP of the ~390 TFLOP per 32k chunk (derived — ESTIMATE); the experts (P3) are already integer, the attention (P4) integer, the selection (P6) tf32, the recurrence (P7) not on the pipe. **F-252 consequence (2) is confirmed and quantified; consequence (3) (decode untouched) is confirmed in (a.2).**

### (a.2) DECODE (verify window; for completeness — F-252 says flat, code confirms)

| op | file:line | pipe/accum | fp16-accum relevant? |
|---|---|---|---|
| dense projections (T24 port) | dense_tc.cu:70/:75 m16n8k32 s32.s8.s8.s32 | int32 accum (IMMA); **STRATA_DENSE_TC default OFF** (dense_tc.cu:6,1218-1226; verify.cpp:1341) | NO — already the fast int pipe |
| hc/gr reads | native_bf16.cu `bf16_f32_mmvf_multi_kernel` (launch_pdl :322-328) | CUDA-core fp32 GEMV-class (M=1-8) | NO — bandwidth-shaped, tensor pipe irrelevant |
| head / dense mmvq | native_mmvq.cu | CUDA-core dequant+dot fp32 | NO — byte-bound (head at 63-68% roofline, T13A #6) |
| fused native experts | iq_kernels.cu:5107, :5299 `wmma::mma_sync` with **`fragment<accumulator,16,16,16,float>`** = fp16-in/**fp32-accum** (238 lock) | the ONE decode fp32-accum TC line | NO in practice: the expert-hit line is byte-floored at 87-92% GDDR7 roofline (T13A 3c) — the mma rate is not the binder |
| QSA decode scores | native_qsa_score.cu:52 tf32.f32 | tf32/fp32-accum | marginal (small share) |
| attention | native_flash_attn.cu, qsa_decode_attn.cu — **zero mma.sync** | CUDA-core fp32 | NO (TC-attention is a separate F-200-attention candidate) |

**Decode inventory verdict: nothing convertible, nothing that matters — F-252 (3) confirmed from code.**

### (a.3) The fp16-accum numerical argument (the F-252 lead, made concrete)

- Ceiling: P1+P2 ≈ 1.10-1.43 s of the 4.325 s 32k GPU timeline (dense phases 36.6% minus non-GEMM kernel share — ESTIMATE). Effective rate today ≈ 140-180 TFLOPS = 60-75% of the 238 lock (derived). At the same efficiency on a 476 pipe: **−0.55..−0.75 s per 32k-chunk-scale ≈ +13..17% prefill GPU**, i.e. ~6.4k → ~7.3-7.5k tok/s at 120k before any other change. Cross-check: Fateev's 3090 family measured +10-12% prefill for the same unlock (F-252), with GEMM ~40% of prefill.
- Error: HMMA f16-accum writes the accumulator at fp16 once per k=16 tile; roundings = K/16 (160 at K=2560). Random-walk RMS relative error ≈ (2⁻¹¹/√3)·√(Σ|S_k|²)/|Y| ≈ **~1%·(√(K/2560))** (ESTIMATE, uncorrelated inputs); chunked accumulation (fp32 flush every 256 k) → ~0.3%. That is: (i) ~30-100× smaller than the already-default NVFP4 W4A4 drift class (KL 0.0004-0.03, mmq_nvfp4_w4a8.cu:3-5), (ii) LARGER than GDN_CHUNKED's 3e-7, so **the F-230 acceptance tax applies** — see (e).
- Contract: no bitwise contract exists on the prefill dense path to break; the gates that exist are exactly the right ones (quant_quality.py prompt/pcompare V1 canary — in-tree, un-armed; M3b; 2× MMLU-Pro).

---

## (b) LAUNCH-MAP DIET SPEC (the decode window; basalt §2.2 map vs F-204 measured)

Standing: **GDN 39 + QSA 52 nodes/layer doorbell (28/41 all-resident), window ≈ 2,100** (F-204, measured); batch S=6 ≈ 2,900. Basalt: **9-11/GDN, 21-23/QSA, ≈640-750 incl. event nodes** (T25-5 §2.2, code-counted). The diet, item by item (our anchors from F-204/T13A, basalt counterpart from T25-5 §2.2):

| # | merge (our nodes deleted/window) | our anchor | basalt counterpart |
|---|---|---|---|
| 1 | q8_1 quantize folded into producers (−180) | standalone quantize nodes ×4-5/layer; hooks exist (gdn_step_norm_multi xq_out, fused_gr q8_mixed) | attn merge / route / rec / hc read all write xq |
| 2 | LFUSE shared-expert chain 6→3 (−144) | gate exists: STRATA_LFUSE, verify.cpp:90 — **arm, don't port** | shared_expert_multi side stream |
| 3 | Fused wait+copy doorbell forms (−144) | STRATA_FUSE_WAIT (F-205 un-gated); waitA/waitB verify.cpp:1685-1699 | dm_wait_kernel: ONE 1-block wait (dec_moe.cuh:314-329) |
| 4 | Router chain 4→2 (−96) | GEMV+top10+route_resident+plan+bell verify.cpp:1519-1590 | router GEMV + top10 INSIDE the hc read (:1871-1889) |
| 5 | hc read 3→2 (−96) | gr_norm fold into gr_down (bitwise-checkable) | hc_fused_read: norm+down+up+inject ONE launch (hc_fused.cu:968-1010) |
| 6 | Batch-window QSA per-row suite → batched (−330 at S=6; ~0 single-session) | per-row QSA launches | qsaf_wide fused chain (21-23 nodes) |
| 7 | PLE row loop → batched postops (−56) | STRATA_DF_PLE exists, bitwise-equal | PLE forked at window start (ple_s_) |
| 8 | QSA small-pairs merges (−48) | K+V mmvq one launch; attn chunk+merge; kv_resolve 2→1 | kv_post fusions + 3 branch forks |
| 9 | Expert-side: skip empty-PCIe grouped node when counts==0; rebase into fetch_blobs; gr_write into combine (−100..−140; FUSE_HITS half already ADOPTED F-226) | verify.cpp:1709 (always launched), :1736, :1743-1763 | device counts checks inside the wait kernel |
| 10 | Misc: expert-input quantize into hc-ffn tail, gate-apply writes q8_1, head mixer 3→2, prologue copies (−60..−90) | — | — |

**Arithmetic:** items 1-5,7-10 sum **−900..−1,000** single-session (F-204's own sum) → **~1,100-1,200 doorbell nodes**; adding the dm-style whole-graph doorbell switch (basalt exec_ vs exec_nr_ replaces our per-group spin nodes) → **~1,000-1,100**. Reaching basalt's 640-750 additionally needs mid_chain (3 launches→1 at T≤4) and the QSA branch forks on priority side-streams. Node term at the measured marginal (R1 §1: 0.48 µs/node late, 0.95 early): **−0.43..−0.95 ms/window ≈ −0.17..−0.38 ms/token at 2.47 tok/window = +4..+9% M1**; plus the fusion execution terms (router slice of 3d ~0.3-0.4 ms; quant slices of 3e/3f) → the T25-5 envelope **−0.8..−1.7 ms/window stands**. Prefill side-note: the prefill path is stream-launched (no graph, prefill.cpp), ~50-70 launches/layer × 48 ≈ 2.5-3.4k launches/chunk ≈ 5-13 ms host submit per 6,144-token chunk ≈ **<1.5% of chunk wall** (derived) — the diet is a decode-window lever, not the prefill lever.

---

## (c) PDL — our actual sites vs basalt's chain, and the band

**Our armed surface (all behind `pdl_supported()`, which requires `STRATA_DF_PDL` set — verify_kernels.cu:1709-1720 "opt-in until the A/B says otherwise"; the standing launcher does NOT set it → the standing boot runs with PDL entirely OFF). Recorder side exists: PdlScope in the window capture (verify.cpp:1104-1112, `G==1` only).**

| what | where | notes |
|---|---|---|
| launch_pdl sites: **21** | native_bf16.cu:322-328 (×7, hc-read bf16 mmv); native_mmvq.cu:1482,1497,1498,1503,1504 (×5 mmvq) + :2462 (quantize_q8_1); dense_tc.cu (×6); verify_kernels.cu:1674 (gpu_stamp) | vs basalt ~73-79 launch sites / 16 files |
| pdl_trigger: **13** | dense_tc.cu ×5 (last-work-unit + pair-start, the T24 port); native_mmvq.cu:159 (quantize→next projection), :1199 (after first-block weight staging); verify_kernels.cu:1655 | trigger-before-wait = basalt's start-trigger pattern, already correct where present |
| pdl_wait + pre-wait staging | native_bf16.cu:95-102 (PRE=8 weight rows staged before wait); native_mmvq.cu:1189-1200 (first block's weights staged, then trigger+wait); dense_tc ×5 | the "constant reads before pdl_wait" leg exists in 3 files |
| files with ZERO PDL that basalt's chain covers | fused_gr.cu (hc/gr reads), gdn.cu/fused_gdn.cu/native_gdn*.cu, native_flash_attn.cu, qsa_decode_attn.cu, qsa_select.cu, native_qsa_score.cu, native_moe.cu (combine), shared_expert.cu, native_router/router_top10.cu, quantize_act.cu, s2_gemv*.cu, kv_stream.cu | basalt has triggers in exactly these classes (dec_moe 6, qsa_dec_attn 4, qsa_select 3, gu2 3, dec_gdn_rec 2, hc_fused/mid_chain/shared_expert/native_moe/pk_moe/qsa_dec_fused/iq_kernels 1 each) |
| graph flag | our instantiate passes **0**: verify.cpp:3588, :3595, :2628 — `cudaGraphInstantiateFlagUseNodePriority` absent (basalt verify.cpp:2334, :3285). Priority streams DO exist (verify.cpp:929) | one-line arm |

**Band verdict on −0.5..−1.5 ms/window (T13A #9, T25-5 T-1): CONFIRMED as the full-chain envelope, REFUTED as a description of anything armed today.** Today's in-tree state = 4 files / 21 launches / 13 triggers, with STRATA_DF_PDL unset and STRATA_DENSE_TC off → measured value standing = **0**. The full band needs: (1) the env arm, (2) dense_tc adoption (T24), (3) triggers+staging in ~12 more kernel files, (4) chain discipline at the spin/copy boundaries (pdl.hpp already forbids programmatic edges after memcpy/event nodes). Sanity: ~1,800-2,000 kernel→kernel edges × (0.3-0.5 µs launch-tail + 0.2-0.4 µs weight-read overlap) ≈ −0.9..−1.8 ms — the claim is inside it once the chain is continuous; a partial chain (today's files ≈ 25-40 of ~2,050 nodes) prices at −0.01..−0.05 ms. The cheapest falsifier stays T25-5 T-1's: arm 2-3 pairs env-gated + nsys overlap probe.

---

## (d) PREFILL PLATEAU ATTRIBUTION — where 6.0-7.0k tok/s at 90-120k comes from

**The measured ladder** (live log custom-iq3_s.log:12229 ff., boot of 2026-10-20/21; identical shape in campaign/results/ctx-ladder-standing.txt):

| ctx | wall | tok/s | marginal segment |
|---|---|---|---|
| 4,086 | 1.22 s | 3,356 | — |
| 32,238 | 5.04 s | 6,391 | 32→64k: 4,662 ms / 32,230 tok = **145 µs/tok** |
| 64,468 | 9.71 s | 6,642 | |
| 96,663 | 14.57 s | 6,637 | 64→96k: 4,859 ms / 32,195 tok = **151 µs/tok** |
| 120,678 | 18.75 s | 6,435 | 96→120k: 4,188 ms / 24,015 tok = **174 µs/tok (+20%)** |

(Second boot, real text: 122,877 tok in 16,997 ms = 7,229 tok/s — depth-t17f-real120k.txt:6 — the band 6.4-7.2k.)

**Decomposition.** Fixed-per-token terms (banked 32k anatomy, knob-pft-engtail.txt:380, GPU 4,325 ms = 132 µs/tok):
- **MoE complex 41.4%** (combine 724 ms — charged-to-phase including host CPU-share joins, PfTimer comment prefill.cpp:2464-2466 + :4698-4710; gemm gu 523; gemm down 309; dequant 156; gather 55; host grouping 24). Experts measured ≈ **185 TOPS effective int8** (derived ESTIMATE) — near the practical MMQ ceiling; NOT unlockable by F-252.
- **Dense projections 36.6%** (hc read 479 incl. norm/mix kernels; gdn 437; qsa proj 304; gdn out 253; router+shared 110) — the P1+P2 families on the **238-lock pipe at ~60-75% of it**.
- GDN recurrence+conv 10.2% (FP32 CUDA-core, no TC — P7).
- QSA attention+select 10.9% at 32k average position — **this is the term that grows**: marginal +29 µs/tok at the 96→120k segment (174 vs 145), from qsa select (per-query scores rise with n_bid ∝ position — the code note "the grid reaches the batch's last query's n_bid (they rise with the position)", prefill.cpp:3330-3334) + qsa attn widths + KV staging of pre-32k cells (kv-resident 32768; "KV streaming: ... 344.8 MiB read from RAM" lines) . If this were O(P²) the 120k prompt would carry ~+7 s of attention — it carries ~+0.8-2.3 s over flat (derived from the ladder vs 132 µs/tok × tokens) → the selection is cap-limited (top-2048, T13A §4) as designed.

**Named dominant term: the flat per-token pipeline itself — MoE int8 experts+combine (~41%) then the dense fp16/bf16 GEMMs on the half-rate fp32-accum pipe (~35% gross, ~25-33% GEMM-only).** Attention is NOT the dominant term at 90-120k; it is a +15-20% marginal tax on the top segments. Consequence ranking for prefill: (1) fp16-accum P1/P2 (+10..17%), (2) TC-BF16 GDN recurrence (+4..6% ceiling), (3) dense dequant elimination (156 ms = 3.6%; STRATA_DENSE_MMQ=1 already measured ~neutral-to-negative at large T — prefillprobe-dmmq.txt/-dmmq2.txt and F-230's curve, crossover ~4k), (4) attention-family work only matters past ~120k.

---

## (e) PLAN IMPACT VERDICTS

| item | verdict | exact statement |
|---|---|---|
| **Window project — launch-map diet** (T25-5 T-2 / R1 S1+S2) | **CONFIRMED** | F-204's item list is the spec; −900..−1,000 nodes → ~1,100-1,200 doorbell (~1,000-1,100 with the dm doorbell switch); −0.43..−0.95 ms/window node term + 0.2-0.5 execution = the −0.8..−1.7 envelope. Sequence unchanged (F0b nsys trace → LFUSE arm first) |
| **Window project — PDL chain** (T-1) | **CONFIRMED, CONDITIONAL** | Band −0.5..−1.5 ms/window holds for the FULL chain (12 more files + env + dense_tc); today's armed surface prices at ~0 (env off, dense_tc off). Not a refutation of F-191; the F-191 closure stays (attribute-only). Cheapest falsifier unchanged (2-3 pairs + nsys) |
| **Window project — wait/combine restructure** (T-3) | **CONFIRMED** (band re-anchored) | −0.2..−0.5 ms/window band stands; its own falsifier (wait-kernel spin stats; if exposed tier wait < 0.2 ms the item prices near zero) still un-run |
| **Window project — cudaGraphInstantiateFlagUseNodePriority** | **CONFIRMED, NEW ARM** | Our instantiate passes 0 (verify.cpp:3588/:3595/:2628); basalt passes the flag (verify.cpp:2334). XS; falsifier = one capture A/B + nsys overlap of the shared-expert fork. Watch: priority inversion vs doorbell spin nodes |
| **NEW — fp16-accum prefill dense GEMMs** (P1+P2, the F-252 actionable core) | **NEW** — top prefill candidate, with a mandatory tax gate | Implementation XS: `CUBLAS_COMPUTE_32F_FAST_16F` in Gemm::f16 (gemm.cu:1119-1141) + route P2 through the existing exact bf16→fp16 lane (gemm.cu:101). Expected +10..17% prefill GPU (−0.55..−0.75 s/32k-scale). MUST price the **F-230 prefill-acceptance tax** (~−2.5% decode class at ~1e-4..1e-3 KL distance; dense-MMQ precedent lost −2.8% M1 for a +0.45-3.1% prefill win — here the prefill win is 3-5x larger so the net is positive, but the gate is not optional): V1 prompt canary (quant_quality.py prompt/pcompare) + M3b + 2× MMLU-Pro + ABAB on M1/pair/conc in the same battery. Mitigation if FAST_16F is too lossy: chunked fp16-accum (fp32 flush every 256 k, ~3.2x error cut) in a custom kernel; fall back to FFN-side-only GEMMs (gate/shexp/down don't write KV). Verify by nsys kernel name that FAST_16F selects an f16-accum HMMA on sm_120 |
| **fp16-accum for decode** | **KILLED** | Inventory (a.2): decode TC lines are int8 (dense_tc), byte-floored (iq_kernels wmma), or CUDA-core GEMV; nothing converts, nothing binds |
| **TC-BF16 GDN recurrence** (P7) | **CONFIRMED as queued** (T25-4 #7) — now with the plateau number attached | FP32 CUDA-core recurrence = 10.2% of prefill GPU; basalt's chunked-TC design + parity harness (1.5e-2 worst-element bound) is the template; GDN-perturbation sensitivity keeps it behind fp16-accum |
| **tools/calibrate.py falsifier** | **CONFIRMED** (scope corrected) | It measures per-box: `--pcie-frac` (7 values, standing config hand-sets **0.35** — never measured here), `--spec-min-p` (0.3/0.5/0.7; note min-p 0.35 already separately falsified at −1.75%, F-entry in T13A arc), `--pool-workers`, adapt-tier triple, STRATA_SH_STREAM (HIP), `--prefill-pipe` (**layer-split only** — n/a single GPU). Interleaved 3 rounds, kept only at >3% (MIN_GAIN) over default. How it changes window decisions: pcie-frac sizes the doorbell/PCIe/CPU split = segments 3f/3h/4 — the wait/combine (T-3) pricing and the dm whole-graph doorbell switch depend on it. It does NOT touch the 90-120k prefill plateau (decode knobs only). Zero-port; run before any T-3 port work |

**Sequencing recommendation (consistent with T25-5 §8, amended):** (0) calibrate.py pcie-frac (CPU-idle window, no code) and F0b nsys node trace → (1) UseNodePriority one-flag arm + fp16-accum FAST_16F env-gated A/B (the two XS arms, both falsifiable in one boot each) → (2) fp16-accum quality gate (V1 prompt canary → M3b → MMLU-Pro) → (3) LFUSE/FUSE_WAIT/quantize-fold diet tranche → (4) PDL chain extension file-class by file-class (fused_gr + native_moe first: they are the hc-read and combine chains) → (5) wait/combine restructure → (6) TC-GDN.

---

## (f) UNKNOWNS, confidence-graded

1. **Does cuBLAS COMPUTE_32F_FAST_16F select an fp16-accumulate HMMA kernel on sm_120 for our shapes?** — HIGH confidence it exists and is the intended lever (documented "allows 16-bit arithmetic internally"; our probe proves the 2x pipe); MEDIUM that cuBLAS picks the f16-accum kernel rather than a fp32-accum kernel with fp16 inputs at every shape — one nsys kernel-name read settles it. (Falsifier cost: minutes, inside the first A/B boot.)
2. **GEMM-only share of the dense phases** (phase charging includes norm/mix/dequant kernels) — MEDIUM; derived band 1.10-1.43 s/32k. One STRATA_PREFILL_TIMING boot with per-kernel names (or the queued nsys pass) resolves.
3. **fp16-accum KL distance for THIS pack** — UNKNOWN until the canary; the ~0.5-2% RMS-per-GEMM estimate (uncorrelated-input model) could be 2-3x worse on correlated activations; the chunked-accum mitigation is designed in if so.
4. **Per-chunk growth inside a 120k prompt** (4 chunk rows would decompose the +20% marginal precisely: attention vs KV staging vs select) — no banked row exists (timing env was off in every ladder boot); one boot answers.
5. **bf16→fp16 range-exactness for every P2 tensor** — HIGH for weights/normalized activations (code comment gemm.cu:667-670, measured on RDNA2: none leaves range); cheap per-tensor assert kernel if wanted.
6. **Basalt's isolated PDL size** — deliberately unpublished on their side (T25-5 §7); ours must be priced by the T-1 falsifier. Confidence in the −0.5..−1.5 envelope: MEDIUM (mechanism certain, magnitude derived).
7. **Wait/combine exposed tier-wait on the standing boot** — UNKNOWN (the --wait-stats instrument has never been banked here); decides whether T-3 is worth M/L effort.

## Sources
Code (this session): src/prefill/{prefill.cpp:131-132,2437-2530,2464-2466,3571,4698; gemm.cu:667-670,1052-1056,1119-1141,1251-1308; wmma_gemm.cu (gfx11-only guards :37,:272); kernels.cu:866-892,991,1174,1624-1652,1935,2595-2680; moe_mmq.cu:7,244; mmq_nvfp4_w4a4.cu:1-5; mmq_nvfp4_w4a8.cu:3-5; gdn_chunk.cu:1-8; moe_f16_kq.cu:1-3}, src/kernels/cuda/{qsa_prompt_attn.cu:40-64,539,1093,1104,1483; native_qsa_score.cu:51-52; qsa_select.cu:194; dense_tc.cu:6,70-75,1218-1226; iq_kernels.cu:5107,5299; native_bf16.cu:95-102,322-328; native_mmvq.cu:159-160,1189-1200,1482-1504,2462; verify_kernels.cu:1655-1674,1709-1720,1762}, src/core/verify.cpp:929,1104-1112,1341,2628,3588-3595, src/kernels/pdl_parity.cpp, include/strata/kernels/pdl.hpp, launch-custom-iq3s-8081.sh, custom-iq3_s-8081.json, /home/svg/LLM/data/packs/iq3_s/{index.txt,conversions.json}.
basalt (read-only): src/kernels/cuda/{hc_fused.cu (start trigger), dec_moe.cuh, dense_tc.cu, qsa_dec_attn.cu, qsa_select.cu, gu2.cuh, dec_gdn_rec.cu, mid_chain.cu, shared_expert.cu, native_moe.cu, pk_moe.cuh, qsa_dec_fused.cu — trigger counts grepped this session}; src/core/verify.cpp:2334 (UseNodePriority).
Banked evidence: SVG-CUSTOM.md:8550 (F-252), :7878-7912 (F-230 dense-MMQ negative + the prefill acceptance-tax law), campaign/results/{knob-pft-engtail.txt:380-384 (F-239 32k anatomy), ctx-ladder-standing.txt, depth-t17f-real120k.txt, prefillprobe-dmmq.txt, prefillprobe-dmmq2.txt}, custom-iq3_s.log:12229-12240 (live ladder), campaign/research/{T25-5,T25-4,T13A,F-204,R1 §1}, campaign/mma-accum-ratelock.cu.
Methods note. CPU-only; no process touched, no build, no GPU binary run. Estimates are marked ESTIMATE and derived from banked phase tables + index.txt geometry; no numbers were invented. Unproductive searches (do not repeat): wmma_gemm.cu/kernels.cu WMMA on the CUDA build (compiled out — gfx11 guards); mma.sync in native_flash_attn.cu (zero — CUDA-core); STRATA_DF_PDL in any campaign script (never set); a prefill-timing row for a >32k chunk (does not exist in any banked log).
