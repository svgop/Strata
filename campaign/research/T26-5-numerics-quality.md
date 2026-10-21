# T26-5 — NUMERICS & QUALITY MECHANICS: RECIPE-α at the math level, the KL yardstick, fp16-accumulate, and the gate protocol

- **Date:** 2026-10-21. Read-only study; CPU-only arithmetic; no builds, no boots, no GPU, no source edits. The machine is LIVE (8081 serving); every number below is either re-derived from on-disk artifacts (marked **derived**) or explicitly **[estimate]**.
- **Read first, not repeated:** T25-4-prefill-quant.md, T25-6-methodology.md, T14B-specs/requant-spec.md, T14B-specs/gates-spec.md, F-251 (SVG-CUSTOM.md:8482-8548, the dense K-quant-mixture premise correction), F-252 (SVG-CUSTOM.md:8550-8572, the measured 238/476 TFLOPS fp32/fp16-accum lock).
- **Executed this session (the only CPU-side runs):** re-parse of `/home/svg/LLM/data/packs/iq3_s/native_experts.txt` (all 48 rows, blob law verified per row); regeneration + diff of `data/quant_quality/v1/set.ids.json` against the live tokenizer (**MATCHES — the KL fixture is valid**); trap checks on the gate scripts.

---

## (a) RECIPE-α arithmetic and amendments

### a.1 What the spec actually quantizes (code-anchored)

Decision variable = 96 cells (48 layers × {gu, d}); RECIPE-α = "cold-12": layers **3, 15, 16, 23, 31, 39, 40, 43, 44, 45, 46, 47** (coldest in the learned profile, cross-checked static; requant-spec §3.1) → `(gu IQ2_S [type 22, 2.5625 bpw], d Q2_0-gs64 [type 42, 2.25 bpw])`. Everything else `=COPY`. Constraints held: gu must share one type per layer (iq_pack.py:452); every target type is in the shipped kernel set {16,17,18,21,22,23,42,20} (native_expert.cpp:40-48, expert_layout.cpp:355-368); the engine re-validates offsets/types per boot (`check_experts_gguf`, expert_source.cpp:4340-4391).

**Blob law (re-verified exactly on all 48 rows of the live file):**
```
blob(gu,d) = 409,600·bpw_gu + 204,800·bpw_d      [bytes/expert; 409,600 = 2×1,638,400 gate+up params,
                                                  204,800 = 1,638,400 down params; per-layer stride]
```
Current census (counted this session — matches requant-spec §1 exactly): (18,20)×14, (22,20)×15, (21,20)×9, (18,42)×3, (22,42)×5, (21,42)×1, (23,20)×1. per-expert params 3×1,638,400 = 4,915,200; total expert params 48×512×4,915,200 = 120.80 B.

### a.2 The bytes, re-derived (**all numbers reproduced by script from the live native_experts.txt**)

| quantity | current | RECIPE-α | arithmetic |
|---|---|---|---|
| experts total | 50,292,326,400 B = **46.84 GiB** | 46,137,344,000 B = **42.97 GiB** | −4,154,982,400 B = −3.87 GiB |
| Δ | — | **−8.26 %** | Σ per-layer deltas × 512 |
| effective bpw | **3.3307** | **3.0556** | bytes×8 / 120.80e9 |
| avg blob | 2,046,400 B (1.9519 MiB) | 1,877,333 B (**1.7904 MiB**) | total / 24,576 |

Per-layer deltas (blob 2,176,000→1,510,400 etc.): L3,15,23,39 **−460,800** (down-only, gu stays IQ2_S); L16,31,43,46 **−665,600** (gu IQ3_XXS→IQ2_S + down); L40,44,45 **−819,200** (gu IQ3_S→IQ2_S + down); L47 **−1,152,000** (gu IQ4_XS→IQ2_S + down). Σ = 8,115,200 B/layer-set. All **derived**, all matching requant-spec §3.2.

**Slots (the +792).** Charge = ceil256(blob) per slot; bank = 17.71 GiB = 19,014,098,534 B; ranking layer-uniform (138-240 of 512 pairs/layer in the top-9,337). Both the mean-blob and per-layer-ceil256 models give **exactly 10,129** for α (vs 9,292 model for today's pack; banked boot ledger 9,337). **Banked delta 9,337 → 10,129 = +792**; model delta 9,292 → 10,129 = +837; the 45-slot gap is boot-to-boot free-VRAM variance (requant-spec §1). Variants: α+R2 (gu 21→18 on L17,25,27,28,35, −153,600×5×512 = −393,216,000 B) → 42.60 GiB, 3.030 bpw, **10,216 slots (+879)**; β (gu-only, down untouched) → 45.61 GiB, 3.2431 bpw, **9,544 model ≈ 9,590 banked (+253)**. All **derived**; spec's table confirmed to the slot.

**The M1 claim.** Hits line 3.26 ms of the 14.2 ms window (T12-MEM, 2,495 hits × 2.05 MB, 87-92 % of GDDR7 roofline). −8.26 % of hit bytes = −0.269 ms → **+1.90 % M1 direct** (derived). Miss-pool/PCIe/staging terms scale with blob → +0.5-1 % more; conc gains more (wider unions, F-183 direction). **"+2-4 % M1" is the honest band** — the T17F "+6-10 %" assumed unsaturated slot credit. Slot credit: +792/512 × 2.1 pt = **+3.25 pt nominal hit rate, saturating** at the standing 97.7 % → honest marginal +1.5-2 pt at 131k; the unsquandered term is the 262k absorption: 10,129 × (8,630/9,337) = **9,362 ≥ 9,337** (derived; spec §6.1 said 9,369 from its rounded 10,129×(8630/9337) — same conclusion, +2 B-chain difference is rounding).

**Correction found (minor, execution-affecting):** requant-spec §0/§4.4 say "**36 of 144** expert tensors requantized, ~5 GB in". From the live file, α as spec'd requantizes **28 tensors**: 12 down_exps (every cold layer's d is IQ4_NL today) + 8 layers × {gate_exps, up_exps} (L16,31,43,46 from IQ3_XXS; L40,44,45 from IQ3_S; L47 from IQ4_XS). The 4 down-only layers keep gu at IQ2_S **by COPY** (that is what −460,800 means; re-quantizing same-type gu would add an error generation for zero bytes). Input volume of the 28 ≈ **11.3 GB** (12×471.9 MB downs at IQ4_NL + 8×642 MB + 3×721 MB + 2×891 MB gu half-pairs) → output ≈ **7.1 GB** (downs at Q2_0 235.9 MB, gu at IQ2_S 268.7 MB); Δ ≈ −4.15 GB ✓ self-consistent. **[derived]** Amend §0/§4.4: 28 tensors, ~11 GB in / ~7 GB out (imatrix pass reads the source bytes regardless; time estimate unchanged).

### a.3 What the family evidence supports

- **maya** (35bb2b2): engine-level KL(FP8‖quant) target ≤ 0.05, zero-shot retention S24 97.7 / S 97.9 / M 97.9 / L 99.2 %; **GPTQ-on-down measured WORSE than imatrix rounding** (quantize.py:14-16) — direct external validation of α's imatrix Q2_0-down cuts; gu-GPTQ with a cold-expert Hessian prior is the pre-validated retreat (T25-4 Ranked #2).
- **basalt** (fb0944c7): dense Q8_0→Q6_K re-quant disclosed **KLD +0.0063** (features.md:26-44) — one K-quant format step on the dense side costs ~0.006 KLD; α's per-tensor change is the same class, confined to 36→28 cold-cell tensors.
- **Field row:** full-pack Q8_0→IQ3_S = mean KL 0.144 / p99 1.975 (docs/QUANTS.md:73-81). α must add a *fraction* of one format step, not a step (V1 thresholds below).
- **GEMQ hazard:** >40 % reroute at 1.5-bit; mitigated by R0-hot-protect + the V1b routing-agreement gate (≥97 % per-layer top-10, no layer < 90 %).

### a.4 Where the spec must be amended (exact)

1. **`=COPY` per-tensor override: now REQUIRED and BANKED.** F-251: the pinned fork only knew whole-file COPY; the T24-S1 patch is in-tree at third_party/llama.cpp `tools/quantize/quantize.cpp:339-345` ("name=COPY" → GGML_TYPE_COUNT sentinel) + `src/llama-quant.cpp:683-695` (`tensor_force_copies` returns the tensor's own type). The spec's §0 "every other tensor byte-identical" sentence is now TRUE; at write time it was aspirational. Verified at fork HEAD 61068748.
2. **F-251 premise propagates:** the dense projections are a **K-quant mixture** (qkv/attn_qkv 22 Q6_K / 10 Q5_K / 2 Q4_K / 2 IQ4_XS; ssm_out 29/4/3; attn_q 5/3/3/1; attn_output 11/1) — NOT Q8_0. The re-quant's dense plan is unaffected (explicit-everything =COPY), but (i) the 1,224-line tensor-types.txt must carry the mixed dense names verbatim; (ii) any tooling that assumed Q8_0 dense (T23's dense_tc requant rationale) is dead; (iii) **contradiction flag:** T25-1:95 says "our GDN qkv/z/out are Q8_0" (blocking mid_chain) while F-251's header census says K-quant — F-251 is the later, header-derived record and wins; re-derive mid_chain's blocker from the headers before citing it again (see §f).
3. **Tensor count / byte volume:** §0+§4.4 "36 tensors, ~5 GB in" → **28 tensors, ~11 GB in / ~7 GB out** (a.2).
4. **V1 canary legs (adopt T25-4 #3):** `quant_quality.py prompt` + `pcompare` (both in-tree, un-armed; cmd_prompt :177, cmd_pcompare :239) become MANDATORY V1 rows — the re-quant changes prefill-path bits and V1's run/compare scores decode windows only.
5. **Calib corpus + reporting (T25-4 #11):** add a 16-32k slice to the imatrix corpus and run V1 `compare --segments` (per-domain read) — the 262k switch is where the thin-calib hazard lives.
6. **R2 dial:** gate α→α+R2 through measured per-cell error screening (maya `test --experts 32` + Xolotl v41 λ-alloc; T25-4 Ranked #1), not rank bands alone.
7. **PATH DRIFT (new this session — every T14B spec is affected):** `/home/svg/LLM/Custom-data/` is now **EMPTY** (contents migrated to `/home/svg/LLM/data/`; standing config custom-iq3_s-8081.json:3-7 points at `/home/svg/LLM/data/models|packs`). Re-base requant-spec §4.0 (`RQ=`, `RQ_PACK=`), harvest-spec :259 (dump path), gates-spec :47,:67-72,:119-121,:421 (rt2/rt-g0) onto `/home/svg/LLM/data/...` before execution. **Production drafter moved too:** config :56-57 serves `--mtp /home/svg/LLM/data/mtp-en/rt` (`"draft_vocab": "en"`), where only `draft_vocab.bin` is local (162,100 B, EN subset) and `dense.bin`/`dense.txt`/`experts.bin` are **symlinks** into `mtp/rt` (425,196 B full-vocab draft_vocab). Gates-spec G0's byte-copy (:120-121) would bind the FULL vocab where production serves the EN subset — copy `draft_vocab.bin` from **mtp-en/rt** and train against the same subset (off-subset masking already spec'd, trainer-spec). The re-quant itself is safe: token_embd/output/head are `=COPY`, so the draft_vocab binding stays valid across the pack swap.

---

## (b) The KL yardstick — concrete spec from our existing hooks

**Everything needed is in-tree; nothing was ported tonight; one CPU-only validation was executed.**

| component | status | anchor |
|---|---|---|
| Token fixture `data/quant_quality/v1/set.ids.json` | EXISTS, committed; **re-generated vs live tokenizer and byte-matched this session** (5 segments × 1,000 ids: code, docs, english, multilingual, agent) | tool cmd_ids (quant_quality.py:49-70) |
| Engine writer | `STRATA_LOGPOS=<path>` + `STRATA_LOGPOS_TOPK=64` (clamp 0..256): one line per forced position — pos, target id, logp(target), argmax, logp(argmax), hit, logp(im_end), logp(target w/o im_end), then 64 `id:logp` (natural log) | verify.cpp:3232-3262 (`window_logprobs`), verify.hpp:330-333 |
| Force-through-decode | `--short-read N` (default 64) reads fresh text through verify windows instead of the batched prompt path; the tool sets it to seg_tokens+64 and FAILS if < half the positions scored (batched-path leak guard) | generate.cpp:999, :2696; quant_quality.py:155-157 |
| Placement pinning | FIXED args `--expert-cache 1 --adapt-every 100000 --pcie-frac 0 --suffix-draft 0 --prompt-cache 0 --kv fp16` + env `STRATA_ROUTE_TAIL_SKIP=0, STRATA_PREFILL_CPU_SHARE=0, STRATA_PCIE_FRAC_DEFAULT=old` | quant_quality.py:38-45 |
| Scorer | `compare`: per segment + overall top-1 agreement, mean KL(REF‖RUN) over REF's top-64 renormalized (unlisted tokens get RUN's 64th logp — the p99 threshold carries this truncation bias), KL p99, both PPLs; asserts same text per position | quant_quality.py:284-320 |
| Prompt-path twin | `prompt` (20 batched prompts = 5 segments × lens 256/512/768/1000, full-vocab fp32 logits via `STRATA_DUMP_FIRST_LOGITS`) + `pcompare` (full-vocab KL + top-1, median/max) | quant_quality.py:174-264 |

**Protocol (the requant-spec V1 row, made exact):**

1. **Reference = the CURRENT pack, same engine, same session** (no Q8_0 of this checkpoint exists on the box — requant-spec §4.6). REF run and arm run are both launched by the tool from config JSONs that differ ONLY in pack/shard paths.
2. **Self-noise calibration (one extra pair, once):** run the standing config twice (`ctl-a.tsv`, `ctl-b.tsv`, different boots) and `compare ctl-a ctl-b`. Expected ≤ 0.01 mean KL (placement is pinned; basalt's 0.01-0.02 reference-noise lesson says *measure, don't assume* — T25-6 §1.2/open-q1). If the floor reads higher, every threshold below moves up by the floor: **the gate's resolution is its own noise.**
3. **Pass thresholds (decode path, per segment AND overall):** top-1 ≥ **97 %**, mean KL ≤ **0.05**, KL p99 ≤ **1.0**, ΔPPL = PPL/PPL_ref ≤ 1.005 (informational). Calibrated against the field row 0.144/1.975 = one full format step; the cold-12 recut must price as a fraction of a step. **Prompt path:** top-1 ≥ 19/20, KL mean ≤ 0.05, median ≤ 0.01, max ≤ 1.0 **[estimate — Xolotl's own bar was "every quant picks the reference's top token after all 20 prompts"; our first arm sets the precedent]**.
4. **Production-matching leg (amendment):** one extra run pair with `--gpu-experts auto`. The default `--expert-cache 1` pins experts to the CPU pool (determinism); the standing serving path is GPU-resident experts + int8 MMQ decode — a bit-level rounding difference between CPU IQ kernels and GPU MMQ on the same new codes is invisible to the default legs. +2 launches, only if the CPU-tier verdict is not already clean-pass/clean-fail.
5. **Distribution stats:** bank the per-position KL histogram offline (read the two TSVs with `read_run`; ~20 lines of throwaway python — no tool edit needed) and report median/p90/p99/max per segment; a fat tail with a clean mean is the GEMQ-routing-drift signature.
6. **V1b routing gate (unchanged):** one `--dump-routing` boot per pack over the same 100k probe; per-layer top-10 agreement ≥ 97 % mean, no layer < 90 % (parser = make_profile.py:46-58).
7. **Tonight's CPU-only prep — DONE and remaining:** ids fixture verified (executed); still to do before the first boot: write `campaign/quant-quality-ctl.json` / `-rq1.json` (copies of the standing config; the tool strips serving-only args itself, DROP_VALUE/DROP_FLAG :42-45; keep `--expert-profile` and the control-vector args — both sides identical); nothing else requires the GPU.

**Cost:** REF+arm = 2 launches ≈ 40-60 min; +prompt pair ≈ 30 min; +self-noise pair once ≈ 40-60 min; +gpu-experts pair optional. First substantive read available for the price of two boots.

---

## (c) fp16-accumulate: error analysis for the prefill GEMMs (F-252 follow-up)

**Baseline (verified):** every dense prefill GEMM runs `cublasGemmEx … CUBLAS_COMPUTE_32F` on FP16/BF16 inputs (gemm.cu:1052-1056, :1119-1141, :1230-1233; parity harness same, gemm_bf16_parity.cu:104-106). The standing launcher arms no `STRATA_DENSE_MMQ`, so the prefill expert GEMMs also run the default dequant→FP16-GEMM path (gemm.cu:4-9; down-proj call shape `m.gemm.f16(…, ne, N, 640)` prefill.cpp:4627). Decode is untouched (int8 MMQ, int32 accum; F-252 consequence 3). F-252 measured the pipe: fp16-accum 476 vs fp32-accum 238 TFLOPS (ratio 1.99/1.97, contended absolute, stable ratio).

**Per-op inventory (K = reduction dim), from index.txt + prefill.cpp:131 (`N=2560, HC=4, D=10240, LR=320`), :464-465 (the two K-6144 projections), :4627:**

| op | K | lines (32k chunk, F-239) | fp16-accum RMS rel. err **[model]** | fp32-accum same term |
|---|---|---|---|---|
| attn_qkv / q / kv / QSA proj (GDN in-proj incl.) | 2560 | 304 ms | **5.0e-3** | 6.1e-7 |
| expert gate/up (dequant+f16 GEMM) | 2560 | 523 ms (in 832 ms gemms) | **5.0e-3** | 6.1e-7 |
| hc up-proj / mixer (D=10240 stream) | 2560 | 479 ms (hc) | **5.0e-3** | 6.1e-7 |
| ssm_out (36 GDN layers) | 6144 | 253 ms (out proj) | **7.8e-3** | 9.5e-7 |
| attn_output (12 full-attn layers) | 6144 | in attn 332 ms | **7.8e-3** | 9.5e-7 |
| expert down | 640 | 309 ms (in 832 ms gemms) | **2.5e-3** | 3.0e-7 |
| output_hc_down (LM-head hc mix) | 10240 | decode-side | **1.0e-2** | 1.2e-6 |
| attention QK^T (per head) | 128-256 | in attn 332 ms | 1.1-1.6e-3 | ~1e-8 |
| attention PV (P≥0, monotone sums) | T | in attn 332 ms | **1.8e-2 @T=4096; 5.1e-2 @T=32k** [model] | ~1e-7 |

**The model (stated so it can be contested):** fp16 unit roundoff u₁₆ = 2⁻¹¹ = 4.88e-4. Random-sign partial sums (zero-mean dot products): RMS relative error ≈ u₁₆·√(K/24); all-nonneg sums (PV after softmax): ≈ u₁₆·√(K/3); worst case K·u₁₆ (irrelevant statistically). fp32 reference: u₃₂·√(K/24) with u₃₂ = 5.96e-8. **Every number in the table's error column is this model, i.e. [estimate]** — the *ranking* and the PV term are robust to the model's constant.

**Where it becomes untenable:**
1. **Attention PV — fatal by two independent mechanisms.** (i) Monotone accumulation error ≈ u₁₆·√(T/3) → 1.8 % at T=4096, 5.1 % at T=32k. (ii) Absolute-ULP stagnation: fp16 spacing at a running sum ~1.0 is 9.77e-4, while per-term increments are |v|/T ≈ 3e-5 at T=32k — **every increment rounds away; the sum stops growing** (partial retention begins only below T ≈ 1-2k). PV must stay fp32-accum at any serving context; fp16-accum PV is not a candidate at any depth we serve (131k/262k lanes).
2. **K ≥ 6144 GEMMs (ssm_out, attn_output) breach the family-anchored noise class 3x.** Anchors: Q6_K weight quant dot-error ≈ 1.1e-3; the hc8 requant measured q8-vs-bf16 rel-rms **2.3-2.4e-3** (F-251 S2, hc_q8_parity 18/18) — i.e. fp16-accum at K=6144 (7.8e-3) injects ~3.3x the error of a full hc requant whose *mechanism* (not numerics) already regressed M1 −7.2 %; K=2560 (5.0e-3) is 2.2x that class. Setting "half a Q6_K step" (5.5e-4) as the per-op budget gives K ≤ ~540 — only the down GEMM (640, 2.5e-3) is even in the argument zone, and it is 4.5x that budget.
3. **Overflow: not a risk** — post-RMSNorm activations keep partial sums O(10²) ≪ 65,504; the PV sum is convex-bounded by max|v|. (Checked, recorded so nobody re-raises it.)

**Model-level consequence [estimate, with the extrapolation shown]:** KL scales ~quadratically in independent zero-mean perturbations. basalt's dense Q8_0→Q6_K (per-GEMM ~1e-3 class over ~4 dense GEMMs/layer) measured KLD +0.0063; a blanket fp16-accum switch perturbs the same lines PLUS experts PLUS hc at 2.5e-3-1.0e-2 → expect **(5-8x)² ≈ 25-60x → KLD ≈ 0.15-0.40** — 3-7x over the V1 gate (0.05), and 10²-10³x the F-230 trigger (1.8e-4 KL of KV = −2.7 % acceptance = −2.8 % M1). The F-230 tax applies because prefill numerics shift the KV the coupled exact-match verifier judges. Directionally certain, magnitude model-extrapolated.

**Rate accounting:** GEMM-phase lines ≈ 304+253+832+479+332 ≈ 2.2 s of the ~4.3 s 32k chunk ≈ **50 % of prefill** [estimate; Fateev's family number "GEMM phases ~40 %" corroborates]. A perfect 2x on those lines ≈ −25 % prefill; realistic kernel-efficiency loss on COMPUTE_16F paths → **+10-12 % prefill** is the honest center — the same number Fateev's *unlock* delivers, and the unlock changes the RATE, NOT THE NUMERICS (fp32 accumulators at the fp16 rate: zero KL cost).

**KL-gate verdict (predicted):** blanket fp16-accum **fails** V1 (mean KL ≫ 0.05; acceptance-tax regime). The only survivable form is scoped: COMPUTE_16F on the **K=640 expert-down GEMMs only** (2.5e-3, statistically the hc8 parity class), KL-gated before any battery; upside ≈ 2x on 309 ms ≈ +3-4 % prefill **before** the F-230 tax, expected net ≈ 0. **R3 inventory entry: KILLED as a blanket candidate; AMENDED to "K=640-only, KL-yardstick-gated, low priority"; the free +10-12 % is the Blackwell unlock watch (F-252 already records it).**

---

## (d) Quality protocol for gating an adopted arm (least GPU time)

Order = cheapest kill first; every rung already has a banked instrument. Serial boots only (T16 law); arm/ctl same session (F-225); config copies single-variable (AGENTS.md rule 5).

| step | instrument | pass | cost |
|---|---|---|---|
| 0 (desk, no GPU) | ids fixture (DONE tonight); configs written; =COPY patch present (verified, quantize.cpp:339) | binary | 0 |
| V0 | pack build + one boot: `check_experts_gguf` silent-pass, `expert cache 10129 slots, 17.71 GiB` ledger line | binary (pack bug) | ~30 min |
| V1 | KL yardstick §b: REF+arm decode runs + prompt/pcompare pair; V1b routing dump pair | §b thresholds | ~1.5-2 h |
| V2 | t17-leg A/B same night; read M1/pair/conc; M3b **informational only** (bits change by design) | M1 ≥ −4 % floor; adoption NOT claimed from one boot (F-231) | ~1 h |
| V3 | 2x MMLU-Pro per side, `.venv/bin/python campaign/mmlu-pro-gate.py 0.933 <label>`, then mimicbench2 **on the same boots** | pooled arm ≥ pooled ctl − 1σ (σ 5.2/run, 3.7 pooled); anchor 95.0; any single < 93.3 or 2-mean < 92 escalates; mimic ≥ 87.9 anchor, ≥ 84.9 hard floor | ~1.5-2 h |
| V4 | five-boot protocol per lane vs 3135-re-banked baselines — only on an adoption claim | campaign +4 %/−4 % law | ~3-4 h |

**Traps checked in the scripts this session:**
- `mmlu-pro-gate.py` imports `datasets` at module top (:14) — **must run under `/home/svg/LLM/strata-custom-2/.venv/bin/python`**; VERIFIED the venv has it (F-230's fix landed). Floor arg 0.933 is per-request `spec_min_p` (:29-37); **arm/ctl matching is the sensitive variable, not the floor value** (gates-spec §6-5). Prints binomial σ (:59-61); warms with `ask("hi",20)` — do not pre-tear the boot.
- `mimicbench2.py` is the ONLY anchor instrument (87.9 = 58/66 weighted; the v1 mimicbench 95.3-98.4 band is a different scale — never mix). Writes an absolute results path; usage `mimicbench2.py [port] [label]`.
- `bit-check.py`: 3 prompts, temp 0, sha256. On divergence: **fresh same-binary ctl re-boot FIRST** (F-180 stateful-tier confound); only reproducible arm-vs-ctl divergence with stable ctl kills.
- `quant_quality.py run` launches its own engine (not t17-leg): respect its 16 GB `--min-free-gb` guard; it does NOT stop the 8081 server — schedule V1 away from the live server or fence it.
- Path drift (§a.4-7) and the **draft_vocab EN trap** (G0 copies from `mtp-en/rt`, not `mtp/rt`).
- Re-quant ordering law stands: pack first, harvest/trainer artifacts stamped against the NEW pack sha (harvest-spec §2/§8.1); a drafter harvested on old-pack hiddens eats the F-230 shift at r18.

---

## (e) PLAN IMPACT verdicts

| # | item | verdict |
|---|---|---|
| 1 | **Re-quant GO (RECIPE-α)** | **CONFIRMED, AMENDED (exact):** arithmetic re-derived and verified (−8.26 %, 3.0556 bpw, 10,129 slots / +792 banked, +1.90 % direct M1, +2-4 % band, 262k absorption 9,362 ≥ 9,337). Amendments: =COPY patch is mandatory and banked (quantize.cpp:339 + llama-quant.cpp:683); **28 tensors not 36, ~11 GB in not ~5** (§a.2); V1 gains the prompt/pcompare legs + per-segment reporting + long-context calib slice; R2 gated by measured per-cell screening; all `/home/svg/LLM/Custom-data/` paths re-based to `/home/svg/LLM/data/` (dir is empty — found this session). |
| 2 | **KL yardstick** | **NEW (as a standing instrument), built entirely from in-tree hooks.** Spec in §b: STRATA_LOGPOS(+TOPK) + --short-read + quant_quality.py run/compare/prompt/pcompare; reference = current pack same-session; self-noise calibration pair first; thresholds top-1 ≥ 97 % / mean KL ≤ 0.05 / p99 ≤ 1.0 against the 0.144/1.975 field step. ids fixture verified against the live tokenizer tonight. Feeds R3: law-3 MMLU becomes the second stage, spent only when KL says the distribution moved (T25-6 implication). |
| 3 | **BENCHMARK_PROTOCOL conformance** | **CONFIRMED** (upstream fixtures + runner in-tree since the v0.2.1 merge; one conforming run = config + 20-30 min). Caveats carried: protocol is upstream-draft status; conforming runs pin stock sampling T=1.0/top_p 0.95 — a separate comparable row, not a probe-lane replacement (T25-6 §2.1). |
| 4 | **Requant tooling upgrades** | **CONFIRMED** with the exact set: =COPY (in-tree, verified); prompt/pcompare in V1 (S); `--segments` + 16-32k calib slice (S); append-only shard recut (~150-line iq_pack.py variant, saves ~1-2 h + 75 GB) (M, optional); per-cell measured-error chooser as the R2 enabler (S/M); per-layer resume sentinels for the 1-2.5 h quantize step (S, insurance). |
| 5 | **fp16-accum candidate (F-252 feed to R3)** | **KILLED as blanket; AMENDED to scoped-only:** per-op table §c — PV untenable by stagnation (T ≥ ~2-4k), K ≥ 6144 GEMMs ~3x the hc8 noise class, predicted blanket KLD 0.15-0.40 ≫ 0.05 gate plus F-230 tax; only the K=640 expert-down line survives screening (2.5e-3) at ≤ +3-4 % prefill before tax. The free path is the Blackwell unlock (rate-not-numerics, +10-12 % family-measured) — keep F-252's watch. All error magnitudes [estimate], anchors (hc8 2.3e-3, basalt +0.0063, F-230 1.8e-4) measured. |
| 6 | **DFlash2 training-data compatibility** | **AMENDED:** the drafter stays KILLED as drop-in (T23: full 248,320 vocab, non-causal block diffusion, SGLang-only). The adoptable asset is the tap convention as the T14C+ rider: completed-layer output hc-means at layers [5,15,25,35,45] (5×2560 = 12,800 features) appended to the 2×2560 drafter input. Numerics contract: capture the taps on the ADOPTED pack (bits shift hiddens); average the 4 hc streams in f32, store bf16-RNE (the dump hook's form); serve-side capture must be the same prefill-path instance in the same form, and G0 TAP parity (cos > 0.9999/stage) must extend to tap tensors; 31-tensor mtp/rt SHA contract gets a version bump; `experts.bin` (q2_0 codes + fp16 scales) and `draft_vocab.bin` stay stock byte-identical (F-114). Sanity gate adopted from the DSpark audit: flat confidence-quartiles = broken input pipeline (Xolotl 0.437→2.583). DeepInfer's FP8 per-16-block scale lesson: never per-tensor scales when downcasting captures. |
| 7 | **NEW — spec-path/contract drift** | **NEW:** Custom-data→data migration + mtp→mtp-en production swap make four spec path sets stale and would bind the FULL draft_vocab where production serves the EN 162,100 B subset (§a.4-7). Fix is five lines of sed-class edits in requant/harvest/gates specs before anything executes. |

---

## (f) Unknowns, confidence-graded

1. **Our KL yardstick's self-noise floor** (ctl-vs-ctl mean KL; expected ≤ 0.01-0.02 by basalt's analogy) — HIGH confidence it is measurable in one cheap pair; value unknown until run. Sets every threshold's resolution.
2. **fp16-accum → KL mapping constant** (quadratic extrapolation from basalt +0.0063 @ ~1e-3 per-GEMM) — MEDIUM-LOW confidence on magnitude (§c table all [estimate]); direction and the PV-stagnation mechanism are HIGH (ULP arithmetic, not modeling).
3. **CPU-tier canary vs GPU-MMQ decode rounding** on the same recut codes — MEDIUM; the `--gpu-experts` leg exists precisely for this and is priced (+2 launches).
4. **T25-1 vs F-251 dense-type contradiction** ("GDN qkv/z/out Q8_0" vs header K-quant census) — MEDIUM-HIGH that F-251 is right (header-derived, later); re-derive mid_chain's type blocker from the GGUF headers before any T25-1 citation is spent.
5. **Realized slot yield 10,129** — HIGH for the model (two independent charge models agree exactly); ±45 boot variance is the known band; the boot ledger line is the arbiter.
6. **Attention q-head split (128 q + 128 gate per 256 slot)** — MEDIUM (trainer-spec :87 "48 heads × [q|gate] 256 interleaved"); cosmetic: QK^T error is 1.1-1.6e-3 either way and no verdict changes.
7. **Prompt-path thresholds (19/20, median ≤ 0.01)** — LOW until the first arm sets precedent (marked [estimate] in §b); the decode-path thresholds are calibration-anchored.
8. **Accepted mimic re-run variance** — unbanked (gates-spec §7); the ±1-task tolerance stays inference until the first paired run adds a second anchor reading.

## Sources (all read this session unless marked)

- Our tree: `data/packs/iq3_s/native_experts.txt` (48 rows parsed, blob law verified per row); `data/packs/iq3_s/index.txt` (tensor roles/shapes: attn_qkv 36, ssm_out 36, attn_output 12 @ K=6144, output_hc_down K=10240, ssm_conv1d [4,10240]); `custom-iq3_s-8081.json` (:3-7 data/ paths, :56-57 mtp-en/rt, draft_vocab en); `launch-custom-iq3s-8081.sh:8-36` (no STRATA_DENSE_MMQ armed); `tools/quant_quality.py` (whole); `src/core/verify.cpp:3232-3262`; `include/strata/core/verify.hpp:330-333`; `src/program/generate.cpp:999, :2696`; `src/prefill/gemm.cu:4-9, :1052-1056, :1119-1141, :1230-1233`; `src/prefill/prefill.cpp:131, :453, :464-465, :4627`; `campaign/mmlu-pro-gate.py` (whole); `campaign/mimicbench2.py:1-60`; `campaign/bit-check.py` (whole); `campaign/depth-lane-real.py:1-40`; `campaign/mma-accum-ratelock.cu:1-50`; `SVG-CUSTOM.md:8482-8572` (F-251/F-252); `AGENTS.md:187-193` (rule 10, F-230 law).
- Specs: `campaign/T14B-specs/{requant,gates,harvest,trainer}-spec.md` (whole; trainer-spec :116-117 record format, :258-279 export chain + q2_0/dense.bin/draft_vocab forms); `campaign/T14B-exec-A-harvest-spec.md` (:26-71 dump contract, bf16 RNE rows [n][10240]).
- Data: `/home/svg/LLM/data/mtp{,-en}/` (inventory: MTP block BF16 31 tensors; rt overlay: dense/experts symlinked, draft_vocab 162,100 vs 425,196 B); `data/quant_quality/v1/` (fixture + verified regeneration).
- Fork: `third_party/llama.cpp` HEAD 61068748 — `tools/quantize/quantize.cpp:73-74, :339-345, :592, :611`; `src/llama-quant.cpp:683-695`; `src/models/qwen4exp.cpp` present.
- Prior dossiers (read, not repeated): T25-4, T25-6, T25-1:95, T25-2 §M5 + item rows, T23-dflash2, requant/gates/harvest/trainer specs, F-251/F-252.

**Methods note.** Productive: scripting the census/arithmetic against the live native_experts.txt instead of trusting the spec's prose (found the 36→28 tensor and ~5→11 GB corrections); regenerating set.ids.json (turned "port the tokenizer" into "already valid"); reading the launcher env (confirmed the fp32-accum cuBLAS baseline). Empty searches recorded: no BF16/Q8_0 source GGUF on the box (re-confirmed via requant-spec §4.6's find sweep; not repeated); no campaign/kl-gate.py yet; no prior T26-* research rows in INDEX.md. Not done (constraints): no GPU, no builds, no boots, no edits outside this dossier; the INDEX.md row is owed at banking time.
