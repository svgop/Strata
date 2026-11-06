---
type: digest
id: ud-slowdown-kernels-2026-11-06
created: 2026-11-06
status: evidence
topic: The UD-IQ2_XS conc slowdown, named kernel by kernel at engine 0.1.57 (commit af6ba4c6's dispatch-counter taxonomy): the exact per-layer, per-window kernel path each pack takes, the optimization state of every form UD gets that IQ3_S misses, the honest ceiling arithmetic, and the ranked fix design
scope: read-only study of /home/svg/LLM/strata (upstream at 9f7d977d, engine 0.1.57) and /home/svg/LLM/strata-custom-2; packs at /home/svg/LLM/data/packs/{iq3_s,unsloth-ud-iq2_xs}; source GGUFs at /home/svg/LLM/data/models/{GSQ-RCO,UD3.1}; the fork's own STRATA_DISPATCH_REPORT captures at campaign/results/dispatch-{ud-iq2xs,iq3s-gsq}.json. No builds, no benchmarks run. Siblings: 2026-11-04-past-ceiling.md (tokens-per-byte formulation — this digest is its kernel-level proof), campaign/results/ud-slowdown-dispatch-evidence.md (the earlier note; its "eager" reading is corrected below).
---

# UD slowdown, kernel by kernel (2026-11-06)

Verdict: **the engine is innocent; the config split is guilty.** At identical engine 0.1.57 on one sm_120, the two runs are not the same experiment. The UD run drafts with the **full 106,299-token draft vocabulary**; the GSQ run drafts with the shipped **"en" subset, 40,525 tokens** (`--draft-vocab en` baked into `strata-iq3s-stock.json` via `--mtp /home/svg/LLM/data/mtp-en/rt`, whose `draft_vocab.bin` is 162,100 B = 40,525 u32 ids; `mtp/rt/draft_vocab.bin` is 425,196 B = 106,299). Everything else in the draft model is symlinked — the weights are byte-identical; the vocabulary is the only draft difference. The fork's own dispatch capture already measured the consequence: **UD's verify windows collapsed to T ≈ 2.0** (2,262 window graphs: T1 999 / T2 686 / T3 272 / T4 147 / T5 71 / T6 64 / T8 12 / T12 11 — 87% at T ≤ 3), while GSQ's sample ran T4+. Spec depth 4 vs 6 showed no change because acceptance, not depth, sets T: `T = 1 + accepted drafts`, and a full-vocab draft misses at min-p 0.5 roughly every other slot. A window re-reads the dense + HC weights (the non-expert fixed traffic) once per window; at T = 2 that weight traffic rides 2.3x more bytes per token than at T = 4.5. That is the 45%.

One correction to `ud-slowdown-dispatch-evidence.md`: the 2,262-vs-13 delta in `launches` is **window count**, not eager launches. `graph_launch(T, kind)` fires only on the captured-graph launch paths (`verify.cpp:3208-3211` prelaunch, `verify.cpp:3332/3368` run, `verify.cpp:5239` pipelined — all guard on a non-null `exec_`, and `capture()` at `verify.cpp:2754` captures on demand before first launch). Every UD window in that capture was a kind-1 (doorbell) CUDA graph, same as GSQ's. The histogram's message is not "eager vs captured" — it is **acceptance collapse**, and it is a bigger, cheaper finding than the note claimed.

---

## 1. The exact per-layer types each pack serves (measured, not assumed)

From `native_experts.txt` of both packs (ggml type ids: 16 IQ2_XXS, 18 IQ3_XXS, 20 IQ4_NL, 21 IQ3_S, 22 IQ2_S, 23 IQ4_XS, 42 Q2_0, 6 Q5_0; per-expert blob = 2 gate/up rows of [640 x 2560] + down rows of [2560 x 640], n_embd 2560 / n_ff 640 / 512 experts / 48 layers):

| layer count | Pack A GSQ-RCO "IQ3_S" gu -> d | blob B | Pack B UD3.1 UD-IQ2_XS gu -> d | blob B |
|---|---|---|---|---|
| 21 | — | — | IQ2_S -> IQ4_NL | 1,971,200 |
| 18 | — | — | IQ2_S -> Q2_0 | 1,510,400 |
| 15 | IQ2_S -> IQ4_NL | 1,971,200 | — | — |
| 14 | IQ3_XXS -> IQ4_NL | 2,176,000 | — | — |
| 9 | IQ3_S -> IQ4_NL | 2,329,600 | — | — |
| 5 | IQ2_S -> Q2_0 | 1,510,400 | IQ2_XXS -> Q2_0 | 1,305,600 |
| 3 | IQ3_XXS -> Q2_0 | 1,715,200 | IQ2_XXS -> IQ4_NL | 1,766,400 |
| 1 | IQ4_XS -> IQ4_NL (**the 2.66 MB blob**) | 2,662,400 | IQ2_S -> Q5_0 (**the 2.18 MB blob**) | 2,176,000 |
| 1 | IQ3_S -> Q2_0 | 1,868,800 | — | — |

Both packs share the IQ2_S gu family for 20/40 layers. Pool: A 50.29 GB, B 42.28 GB (-16%).

**Dense side (served natively from the source GGUF, not dense.bin — `tools/iq_pack.py` header: "Quantized dense tensors ... are served natively from the GGUF by the engine"):**

| projection | Pack A (GSQ GGUF) | Pack B (UD GGUF) |
|---|---|---|
| attn_qkv [10240 x 2560] | Q6_K x~26 / Q5_K / Q4_K / IQ4_XS | **Q8_0 x48** |
| attn_gate [6144 x 2560] | Q4_K / Q6_K / IQ4_XS / Q5_K | Q6_K x48 |
| ssm_out [2560 x 6144] | Q6_K / Q4_K / Q5_K | **Q8_0 x48** |
| ffn_*_shexp | Q8_0 (index code_bits 8) | Q8_0 (same) |
| hc_* (attn/ffn up/down/inject, output_hc_*) | BF16 in GGUF and dense.bin (compat_bf16=false — GSQ stores them BF16 natively) | Q8_0 in GGUF, **BF16 in dense.bin** (compat-bf16 rounds them, doubling bytes), **plus a Q8_0 copy quantized back at load** (`STRATA_HC_Q8=1`, `native_dense.cpp:456-490`) |

Per-window fixed weight traffic (once per window regardless of T, up to the HC slice factor): A ≈ 2.26 GB dense + 1.25 GiB HC bf16 ≈ **3.5 GB**; B ≈ 2.90 GB dense + 0.65 GiB HC q8 ≈ **3.6 GB** — equal. Expert bytes per token-slot: A ≈ 8 x 2.056 MB x 48 ≈ 0.79 GB; B ≈ 8 x 1.720 MB x 48 ≈ 0.66 GB. Per token: A@T4.5 ≈ 1.57 GB, B@T2 ≈ **2.45 GB** (2.3 GB of it the same weights re-read every second token).

## 2. Kernel by kernel: what each form runs at 0.1.57 on sm_120 (stock env — `arch_default_env` returns an EMPTY table for CUDA builds, `arch_defaults.cpp:41-91`; every AMD default incl. `STRATA_EXPDEC`/`HC_Q8` auto is HIP-only)

**Grouped experts — both packs take the same family.** `native_expert_grouped` (`iq_kernels.cu:6277`): `exp_mode()` = 0 on sm_120 (the CUDA layout; mode 8 is Volta/P100/P40/GB10-only, `iq_kernels.cu:6189-6208`), so `gs_layout` = true and `gu_swiglu_on()` = true on cc 12.0 (`iq_kernels.cu:4246-4262`, "on for a compute capability 12.0 card") -> **gu_form 2**: one `native_gu_swiglu_kernel<TG, stage>` launch per group, TG staged-grid (g_stage_grid default on) for IQ2_S(22)/IQ2_XXS(16), unstaged for IQ3_XXS(18)/IQ3_S(21) (`stage_tables_on()` = opt-in `STRATA_IQ_STAGE_TABLES`, off) and IQ4_XS(23) (not `kStageIqGrid`). Down: `launch_down` -> `native_down_multi_kernel` (kSplit true) for IQ4_NL(20) and Q2_0(42); **the single Q5_0(6) down layer of pack B takes the full-warp generic `native_down_kernel`** (`kSplit<6>` has no specialization) — 1/48 layers, noise. EXPDEC never fires (off without env; its coverage lacks IQ4_XS gu and Q5_0 down anyway). NIBBLE never fires (sm_60-only build gate, `expert_nibble.hpp:20-37`). `s2_expert_grouped.cu` never fires (that is the estate Q2_0/S2 pack path, `verify.cpp:2387` `!lay.native` branch; both packs are native). Verdict: **the expert path cannot explain the deficit** — UD reads 16% fewer blob bytes at 99.0-99.5% hits through the same kernel family.

**HC read — the two packs genuinely diverge.** Pack A (no `STRATA_HC_Q8`, CUDA auto = off): bf16 rows, `hc_q8 == nullptr` -> `fused_gr_read_multi_slice` -> `v3 = gr_v3_default_on(sm_120, >13 GiB) = true` (`fused_gr.cu:2651`, `2847-2852`) -> the **V3 stream-split read**: `gr_down_v3_kernel` + `gr_up_v3_kernel` with exact-T instantiations T=1..8, `STRATA_GR_R2` rows-per-warp dispatch, PF/prefetch masks, dispatch counter **kHc var 20+split**. Pack B (`STRATA_HC_Q8=1` in `strata-unsloth-ud-iq2_xs.json`, the S23 experiment, `fused_gr.cu:1970`): `hc_q8 != nullptr` -> `launch_q8` (`fused_gr.cu:2838`) -> **`gr_down_q8_kernel` + `gr_up_q8_kernel`**, a 176-block fixed grid (10 row groups x 16 column chunks), no exact-T family, no cp.async staged variant, and it **declines `fused_gr_read_persist`** (`fused_gr.cu:3776`) and the staged down kernel (`fused_gr.cu:1617`). In the fork's own capture this is the `hc read: weights Q8_0 copies=1 ... x48 x2 halves` line vs GSQ's `copies=0`. The S23 form is documented **"Opt-in until decided"** with **no NVIDIA decode measurement** (`docs/QUANTS.md:164-168`; auto-on only gfx12/gfx1151 HIP, `native_dense.cpp:46-77`). It halves HC bytes (0.65 vs 1.25 GiB per pass) and may still lose the per-byte race at T<=2, where no kernel amortizes its weight reads.

**Dense projections — UD's qkv/ssm_out fall off the tuned path.** The decode window runs `native_mmvq_il` (STRATA_MMVQ_IL default on) — but `native_mmvq_il_supported` whitelists **only Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS** (`native_mmvq.cu:3956-3960`). **No Q8_0 IL kernel exists**: no rows-per-warp specialization, no `kIlRows_120` cell, no 5-8-token table entry. UD's attn_qkv + ssm_out (2 x 48 projections, 42.5 MB/layer = 2.0 GB/pass, the single largest dense block) run the plain multi-column `native_q8_0_mmvq` (**kDense var 0**, not var 10+r). GSQ's K-quant/IQ4_XS dense gets the sm_120 tables measured on an RTX PRO 5000 — e.g. the Q5_K head at 5 tokens **899 -> 491 us** (`docs/MMVQ_IL_TABLE.md:8-17`). This is a real per-window deficit that grows with T — invisible solo, fully exposed exactly when M1 is fixed and windows grow.

**Window/HC-side extras visible in the fork's own dispatch JSON:** UD `gdn layer: conv+ab fused=0 batch_rows=0` + `side_gates=1 side_proj=1` on every layer (forked side-stream gate/query projections — a latency form for SMALL T), vs GSQ's `batch_rows=1` (72 layers) and in-line gates. UD `head: final mixer q8=1` (the Q8 output mixer) vs GSQ `q8=0`. UD `experts layer ... pcie_mode=2` (DMA staging) vs GSQ `pcie_mode=0` — the miss-path form; at 99.0-99.5% hits, innocent, as the operator already established. UD `router layer: batched native router=0` at T1 (batching needs T>=2). None of these is 45%; all of them are the small-T tax.

## 3. The honest ceiling arithmetic

Let F = per-window fixed weight traffic (dense + HC + shared expert), E = expert bytes per token-slot, T = tokens per window. Per-token bytes = F/T + E. Measured conc: A 338 t/s, B 183 t/s.

| | A (T = 4.5, F = 3.51 GB, E = 0.79 GB) | B measured (T = 2.0, F = 3.58 GB, E = 0.66 GB) | B at A's T = 4.5 |
|---|---|---|---|
| bytes/token | 1.57 GB | 2.45 GB | **1.46 GB** |
| implied bytes/s at measured t/s | 530 GB/s | 448 GB/s | — |
| conc t/s at the same bytes/s | 338 | 183 | **~342-360** |

**Pack B at pack A's window size should MATCH or BEAT pack A** — fewer expert bytes per slot, equal fixed traffic, and its HC form reads 48% fewer HC bytes. The measured 183 t/s is the T = 2 window tax: 3.58 GB of dense+HC weights re-read every ~2 tokens instead of every ~4.5. The ~45% deficit decomposes to ~35-40 points of window collapse (M1) and ~5-10 points of form deficit at small T (M2/M4; M3 only bites once T recovers).

## 4. Ranked mechanisms

1. **Spec-window collapse from the full-vocab draft head** — root cause, config-owned, not a kernel. UD's own capture: avg T 1.96. Fix: point UD at `mtp-en/rt/draft_vocab.bin` (the shipped "en" 40,525-token subset; `mtp.cpp:727-760` documents the subsets and prices acceptance) and optionally `--mtp-q4 all` to match A's draft form. **Expected on the 6-lane: +65-85% (183 -> ~300-340).** Zero code.
2. **HC read form** — `launch_q8`/`gr_down_q8_kernel` (S23, never perf-measured on NVIDIA, declines staged/persist forms, 176-block grid) vs A's V3 exact-T stream-split. Fix A/B: one env flip `STRATA_HC_Q8=0` on UD (bf16 V3 reads 1.25 GiB but at T4.5 the V3 form's efficiency should win; at T2 neither wins). Long-term: port the q8 loads into the V3/staged kernel family so the byte win and the form win stack. **Expected: 0-15%** after M1; measure before porting.
3. **Dense Q8_0 without the IL path** — add `native_q8_0` rows-per-warp IL kernels + measured `kIlRows_120` cells (the dp4a sub-warp row dot already exists in the grouped-down kernels to copy). Only pays once T >= 4. **Expected: +10-20% on UD conc at T4.5** (2.0 GB/pass at ~1.8x per-byte gap closes).
4. **Small-T GDN/router forms** (`batch_rows=0`, forked side projections, per-token router at T1) — mostly M1's shadow; they heal as T grows. **Expected: +2-5%.**
5. **pcie_mode 2 vs 0** — innocent at 99%+ hits; leave unless the hit rate drops under multi-slot load.
6. **Refuted: a per-type expert-kernel deficit.** Same `native_expert_grouped` family, same gu_form 2, kSplit down kernels for every material type; the Q5_0-down layer and the EXPDEC/nib/s26 exclusions are 0-1 layers each. The expert cache and its 99.0-99.5% hit rate are clean.

## 5. The one experiment

**Swap the draft vocabulary on pack B and rerun the 6-lane conc round.** `cp`/point `--draft-vocab` to the shipped "en" subset (or copy `mtp/rt` to a new dir with `draft_vocab.bin` = `mtp-en/rt/draft_vocab.bin`, 40,525 ids), keep every other flag, run the same 1 warm + 1 conc round with `STRATA_DISPATCH_REPORT=<path>` set. Confirm in one read: the `launches` histogram moves from `T1 999 / T2 686 / T3 272 ...` to T4-T8 dominance, and the 6-lane aggregate lands near ~300-340. If it does, M1 is proven and the remaining gap is M2/M3, each then measurable by one more single-variable run (`STRATA_HC_Q8=0`; then the IL port when it is written). If T does NOT move, the acceptance collapse is intrinsic to the UD quant (draft-vs-target divergence, not vocabulary breadth) — then the next probe is `--mtp-q4 all` on B (A's exact draft form) before any kernel work.

## 6. File map (all under /home/svg/LLM/strata)

- Dispatch counters / taxonomy: `include/strata/core/dispatch_stats.hpp` (commit af6ba4c6), emitters in `src/kernels/cuda/native_mmvq.cu:3545,3946` (kDense var 0 / 10+r), `src/kernels/cuda/iq_kernels.cu:6252,6371` (kExpert var 0, gu_form 1/2/3), `src/kernels/cuda/fused_gr.cu:1619,2908` (kHc var 0/1/2 plain/split/staged, var 20+split V3), `src/core/verify.cpp:1393-1397` (win notes), `include/strata/kernels/pdl.hpp`.
- Grouped experts: `src/kernels/cuda/iq_kernels.cu` — `native_expert_grouped:6277`, `gu_swiglu_on:4246`, `launch_gu_swiglu:4273`, `launch_gu:4223`, `launch_down:4330`, kSplit table `:908-1260`, EXPDEC `:4081-4214` (CUDA-off), stage tables `:3423-3448`.
- HC read: `src/kernels/cuda/fused_gr.cu` — S23 block `:1970-2400`, `launch_q8:2582-2620`, V3 gate + dispatch `:2847-2937`, persist refusal `:3776`, `gr_v3_default_on:2651`.
- Dense IL whitelist: `src/kernels/cuda/native_mmvq.cu` — `native_mmvq_il_supported:3956` (types 11/12/13/14/23 only), `native_mmvq:3539` switch, table doc `docs/MMVQ_IL_TABLE.md`.
- HC_Q8 loading: `src/core/native_dense.cpp:46-95,429-496` (Q8_0 copies from GGUF, or quantized from the bf16 pack rows).
- Draft vocab: `src/core/mtp.cpp:549,727-760,795-830` (subsets "en" 40,525 / "cyrillic" 58,963; loader reads 4 bytes per id).
- Window graph launch counters: `src/core/verify.cpp:3208-3211,3332,3368,5239`; capture `:2754`.
- The fork's own captures: `/home/svg/LLM/strata-custom-2/campaign/results/dispatch-ud-iq2xs.json`, `dispatch-iq3s-gsq.json`, `ud-slowdown-dispatch-evidence.md`.
- Pack manifests: `/home/svg/LLM/data/packs/{iq3_s,unsloth-ud-iq2_xs}/{native_experts.txt,index.txt,conversions.json,compat-bf16.json}`; source GGUFs under `/home/svg/LLM/data/models/`.
- Run configs: `/home/svg/LLM/strata/strata-iq3s-stock.json` (`--mtp .../mtp-en/rt`, no HC_Q8 env), `strata-unsloth-ud-iq2_xs.json` (`--mtp .../mtp/rt`, `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`, no `--mtp-q4`).
