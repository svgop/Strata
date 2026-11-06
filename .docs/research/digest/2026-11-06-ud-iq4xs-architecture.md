---
type: digest
id: ud-iq4xs-architecture-2026-11-06
created: 2026-11-06
status: evidence
topic: The UD-IQ4_XS v31 set architecture (what Unsloth Dynamic 3.1 bought per layer), the fork engine's per-(gu,d) kernel path map with file:line proof, the same-hour window anatomy arithmetic, and the ranked plan that makes UD-IQ4_XS beat GSQ IQ3_S on this engine
scope: read-only study of /home/svg/LLM/strata-custom-2 (HEAD b7af39da, engine 0.1.46, sm_120 RTX 5090); packs /home/svg/LLM/data/packs/{unsloth-ud-iq4_xs-v31,unsloth-ud-iq2_xs,iq3_s}; source GGUFs /home/svg/LLM/data/models/UD3.1/Qwen3.8-Flash-Next/UD-IQ4_XS/ and /home/svg/LLM/data/models/GSQ-RCO/Qwen3.8-Flash-Next/; logs logs/custom-{ud-iq4xs,ud-iq2_xs,iq3_s}.log. No builds, no GPU runs. Siblings: 2026-11-06-ud-slowdown-kernels.md (the IQ2_XS study this extends), campaign/results/fork-ud-first-bank.md (the same-hour decomposition this digests into per-pair mechanisms), 2026-11-04-past-ceiling.md (tokens-per-byte formulation).
---

# UD-IQ4_XS architecture, engine path map, and the gap-closing plan (2026-11-06)

Verdict: the +17-24% deficit is **one kernel-form hole plus a residency tax**, not bytes alone. The pack's dominant pairs — {gu IQ3_XXS, d Q5_0} x20 and {gu IQ4_XS, d Q5_0} x13, 69% of all layers and 72% of pool bytes — are the only pairs in either pack whose **down projection has no kSplit specialization**: `kSplit<6>` does not exist, so 33 of 48 layers run `native_down_kernel<6>` (one warp per row, activation re-read from global per entry, one entry at a time), while every GSQ layer and every other UD layer takes the shared-staged `native_down_multi_kernel`. The gu side is already GSQ-class (fused swiglu, sm_120 default). Add a +20% bytes-per-hit handicap (4.01 vs 3.34 avg bpw — the quality crown's price) and the bank's "+21% GPU compute per hit" decomposes exactly. The plan: flip Q5_0 downs onto the existing multi kernel (a ~2-line change with parity and D_FMTS coverage already in-tree), then either repack d6->d20 (zero code, -4.65% pool) or build the S26-class pair kernels for {18,6}/{23,6} — the item that pushes per-byte past GSQ and satisfies the operator's win condition.

---

## 1. The set: what Unsloth Dynamic 3.1 (UD-IQ4_XS v31) actually is

Model: Qwen3.8-Flash-Next — 48 layers, n_embd 2560, n_ff 640 per expert, 512 experts/layer, hybrid GDN+QSA attention (boot lines: `ssm_*` tensors, QSA KV streaming), PLE table 320,001,536 rows. Per-expert weight block: gate+up 2 x [640 x 2560] + down [2560 x 640] = 4,915,200 elements.

### 1.1 Type matrix (native_experts.txt v4, 48 layers; ggml enum resolved from third_party/llama.cpp/ggml/include/ggml.h:390-433)

| pair (gu->d) | layers | blob B | per-type pool B | share | gu bpw | d bpw | eff. bpw |
|---|---|---|---|---|---|---|---|
| IQ3_XXS(18) -> Q5_0(6) | 20 | 2,380,800 | 24,379,392,000 | 40.4% | 3.0625 | 5.5 | 3.875 |
| IQ4_XS(23) -> Q5_0(6) | 13 | 2,867,200 | 19,084,083,200 | 31.6% | 4.25 | 5.5 | 4.667 |
| IQ3_XXS(18) -> IQ4_NL(20) | 8 | 2,176,000 | 8,912,896,000 | 14.8% | 3.0625 | 4.5 | 3.542 |
| IQ4_XS(23) -> IQ4_NL(20) | 4 | 2,662,400 | 5,452,595,200 | 9.0% | 4.25 | 4.5 | 4.333 |
| IQ2_S(22) -> Q2_0(42) | 2 | 1,510,400 | 1,546,649,600 | 2.6% | 2.5625 | 2.25 | 2.459 |
| IQ2_S(22) -> IQ4_NL(20) | 1 | 1,971,200 | 1,009,254,400 | 1.7% | 2.5625 | 4.5 | 3.209 |

Total pool 60,384,870,400 B = 56.24 GiB (boot: `loaded 56.24 GiB at 4.74 GiB/s`); avg blob 2.457 MB; avg bpw 4.006. GSQ: 50,292,326,400 B, avg blob 2.046 MB, avg bpw 3.337 (matrix: 15x{22,20}, 14x{18,20}, 9x{21,20}, 5x{22,42}, 3x{18,42}, 1x{21,42}, 1x{23,20}). **Every IQ4_XS layer blob is byte-arithmetic-consistent with its (gu,d) pair** — no decoy types, the manifest is the truth.

### 1.2 Depth pattern (which layers got the heavy types)

- **Heavy gu23 (IQ4_XS 4.25 bpw)**: layers 4-5, 15-18, 27-35 (mixed with d20 at 30/32/34/35), 37, 47 — the FFN core of both mid bands, plus first-attention-pair and the last layer.
- **Lightest gu22 (IQ2_S 2.5625 bpw)**: layers 8, 9, 10 — the early GDN/hybrid zone (the deepest early block, lowest sensitivity).
- **Q5_0 downs (5.5 bpw)**: 33 layers — everything except {x,20} x13 and {22,42} x2. This is Unsloth spending bits on the down projection (the SwiGLU-output side), and it is exactly the third of each layer that our engine serves on the untuned form.
- Layers 42-44 (late): {18,20}; layer 47 (last): {23,6} — heaviest at the output boundary, per the usual UD sensitivity ordering.

### 1.3 Shard layout (Qwen3.8-Flash-Next-UD-IQ4_XS-0000X-of-00004.gguf)

- S1 10.9 MB: header + metadata KVs (tokenizer, architecture).
- S2 49.60 GB: **all 48 layers' dense tensors** (1186 tensors: Q6_K 213, Q8_0 284, F32 557, BF16 24 + experts L0-35, 108 tensors). Tensor census per role: attn_qkv/attn_gate Q6_K x36 + Q8_0 x1 each, ssm_out Q6_K, ffn_*_shexp Q8_0, ffn_gate_inp F32, hc_* Q8_0 in GGUF (-> bf16 rows at load, below).
- S3 15.26 GB: experts L36-47 (36 tensors) + layer 35's up projection (manifest row 35: gate S2 / up S3 @2880 / down S2 — the only cross-shard expert).
- S4 28.80 GB: one IQ4_NL tensor — the PLE table (320M rows).

Structural facts the engine exploits or could: (a) the expert arena maps the whole pool once (56.24 GiB host-pinned, `cudaHostRegister PORTABLE`), so shard boundaries cost nothing at tier time; (b) `data/expert-profile.bin` (24576 ranked pairs) is shared with GSQ — same model, same routing distribution, so tier placement compares fairly; (c) blob-size spread 1.51-2.87 MB means an adaptive swap that trades one {23,6} blob for two {22,42} blobs changes hot-set cardinality — the adaptive tier's replacement heuristic is not byte-aware today (P7 below).

### 1.4 Dense side (served natively from the GGUF; pack carries 1.48 GB dense.bin + compat-bf16.json + conversions.json)

- `STRATA_UD_BF16_DENSE=1` (boot): 1.17 GiB of Q8_0 HC projections dequantized to bf16 rows at load — HC read + final mixer take the copies=0 form (+0.555 GiB VRAM vs Q8 copies). Same form GSQ runs (`hc read weights bf16 rows copies=0`, both logs).
- Head: `experimental native Q5_K head, 521472000 bytes` — identical on both lanes. Token embedding IQ4_XS, 322 MiB host-mapped.
- Decode dense projections stay on native_mmvq (`STRATA_DENSE_TC unset -> mask 0`). Q6_K is inside the IL whitelist, so the dominant dense tensors take the sm_120 IL tables — the dense side is GSQ-class on this pack (it was IQ2_XS whose dense was Q8_0-heavy).
- KV: `--kv q4_0` (GSQ pins `--kv int8`) — 1.69 vs 3.09 GiB pinned, slot sessions 0.77 vs 0.95 GiB. UD is lighter here.

## 2. The engine's path map per (gu,d) pair (fork b7af39da, engine 0.1.46, sm_120 cc 12.0)

Dispatch spine: `native_expert_grouped` (src/kernels/cuda/iq_kernels.cu:4772) -> exp_mode()==0 on sm_120 -> gs_layout -> `gu_swiglu_on()` (iq_kernels.cu:3839, default on for cc 12.0) -> `launch_gu_swiglu` per gu type; then `launch_down` per d type. S26 fires only for {21,20}+`STRATA_EXPERT_V2` (iq_kernels.cu:4786, not set on either launcher — parked on both lanes); V2K only gu 12/13 (:4818); EXPDEC covers gu {16,17,18,21,22} (:3694-3696 — **23 excluded**) and d {20,42} (:3698-3700 — **6 excluded**), env-gated off anyway.

| pair | gu kernel (file:line) | down kernel | form class vs GSQ |
|---|---|---|---|
| {18,6} x20 | `native_gu_swiglu_kernel<18,false>` iq_kernels.cu:3877 (stage tables off, :3822) | **`native_down_kernel<6>` generic** :3887 (`kSplit<6>` absent, table :814-1367) | gu = GSQ-class; **down = the hole** |
| {23,6} x13 | `native_gu_swiglu_kernel<23,false>` :3877 (23 not in `kStageIqGrid`, :820) | **`native_down_kernel<6>` generic** :3887 | same |
| {18,20} x8 | as {18,6} | `native_down_multi_kernel<20,true>` :3890 — 8-lane rows, shared-staged activations (:1981-2069, hb==20 path) | = GSQ's tuned form |
| {23,20} x4 | as {23,6} | as {18,20} | = GSQ-class |
| {22,42} x2 | `native_gu_swiglu_kernel<22,true>` staged :3874 (`g_stage_grid` default on, :3071) | `native_down_multi_kernel<42,true>` :3890 | staged gu — better than GSQ's unstaged 22 |
| {22,20} x1 | as {22,42} | as {18,20} | GSQ-class |

The two down forms (iq_kernels.cu:1937 vs :1970): generic = one warp per output row, `row_dot<6>` over 20 q8_1 blocks, **activation blocks re-read from global per (row, entry), one entry at a time**; multi = GRP_NC entries staged cooperatively into `s_hq_buf` shared memory, weight row decoded once per entry batch (`row_dot_multi`, :2080-2111 — the generic-TD branch handles hb==20, which is our n_ff 640). The EXPDEC note (:2131-2139) quantifies why this matters: pre-staging, each entry's global activation re-read costs ~60% of a whole expert's time (~1.6 us/entry vs 2.6 us/expert on the 2080 Ti study). The multi kernel **is already instantiated for type 6** (STRATA_D_FMTS includes X(6), :763, instantiated at :4985) and already parity-tested (`native_grouped_parity.cpp:360-361` — "6 = Q5_0: UD-IQ4_XS"); only the `kSplit<6>` constexpr routes around it.

Dense/HC/head/drafter (both lanes unless noted): Q6_K dense -> `native_mmvq_il` (whitelist {23,12,13,14,11,8}, src/kernels/cuda/native_mmvq.cu:3276-3282); Q8_0 dense -> plain `native_q8_0_mmvq` — the IL-Q8 form exists (IlIQ4XS-class kernels, native_mmvq.cu:1536+) but is **parked off** (`il_q8_on()` :3142 requires `STRATA_MMVQ_IL_Q8=1`; the night A/B read -1.0/+3.7/+1.6%, fork-ud-first-bank.md). HC read: bf16 rows, staged form (`STRATA_HC_SPLIT`-class cp.async staging, boot: "the hyper-connection read runs as staged"). Drafter: identical both lanes — mtp Q8_0 859 MiB VRAM, `--mtp-q4 all`, 40525-token Q4_0 head (55.7 MiB/step), `STRATA_ROUTE_RESIDENT_MTP` bias. Not a differentiator.

### 2.1 Window anatomy (the logs, serial-class same-hour lines)

| metric | GSQ IQ3_S | UD-IQ4_XS |
|---|---|---|
| windows / avg rows | 4.82 | 4.69-4.76 |
| ms/window | 12.49-13.83 (bank's pair: 13.59) | 15.22-15.89 |
| run split | CUDA0 wait 0.37-0.58 + CPU experts 2.00-2.68 | CUDA0 wait 0.46-0.97 + CPU experts 1.81-2.15 |
| VRAM-hit us/layer-window | 29.8-31.1 | 28.5-29.6 |
| routed: VRAM % | 94.9-96.5 | 93.5-95.1 |
| routed: PCIe % | 0.5-1.0 (pcie_frac 0.35 + dma pinned) | 1.9-2.7 (default 0.55 probe, no dma arg) |
| routed: CPU % | 2.9-4.1 | 3.0-3.8 |
| serve hit rate | 96.0-100% | 92.2% aggregate |
| cache slots / GiB | 10079 / 19.12 (41.0% resident) | 8299 / 18.95 (33.8% resident) |
| rows/s (line) | 262.9-300.8 | 230.1-245.9 |

Pooled all-class: GSQ 643,011 windows avg 14.67 ms @ 4.15 rows vs IQ4_XS 5,406 windows avg 15.33 ms @ 4.06 rows. The bank's same-hour conc-class pair (4.76 vs 4.70 rows): 15.89 vs 13.59 ms with **identical routing union (28.55 vs 28.81 hits/layer-window), identical CPU term, identical GPU-reach wait** — the whole +17% window is GPU compute per hit (+21%).

### 2.2 The arithmetic that closes the loop

- Bytes per VRAM hit: 1369 hits x 2.457 MB avg = 3.36 GB/window (UD) vs 1383 x 2.046 = 2.83 GB/window (GSQ) = **+19-20%** at identical hit counts.
- Measured compute-per-hit: **+21%**. So per-byte efficiency is at parity overall — which means the generic-down hole on 69% of layers is being **offset** by the gu side's efficiency (fused swiglu + staged 22 + lighter early layers), and fixing the down form does not just close the hole, it creates the surplus the win condition needs.
- Fixed per-window traffic (dense+HC+head, re-read once per window): UD ~4.2-4.3 GB vs GSQ ~3.9 GB — at T=4.7 that is ~+0.06-0.09 ms/row, secondary. Expert bytes and form carry the deficit; residency (33.8% vs 41.0% slots, 92.2% vs 96%+ serve hits) carries the miss-path tax.

## 3. The gap-closing plan (ranked by expected lane delta, each with its falsifier)

Baseline: solo 176.8 / pair 196.5 / conc 238.5 vs GSQ 233.1 / 224.8 / 303.5 (lanes.csv legs 164.3-176.9 / 183.1-188.5 / 227.7-246.2 at 3b56b58c; bank means match). Serial window gap +2.7-3.2 ms (+22-26%); conc gap -22%; pair gap -13%.

**P1. Route Q5_0 downs onto the multi kernel — flip `kSplit<6>` to true (iq_kernels.cu:814).** The instantiation (STRATA_D_FMTS X(6) -> native_down_multi_kernel<6>), the parity case (native_grouped_parity 6, "IQ4_XS: n_ff % 256"), and the geometry branch (hb==20 shared-staging path, :2080) are all already in-tree; this is the ~2-line kernel-form change. Affected: 33/48 layers, 72% of pool bytes. Expected: the multi form removes the per-(row,entry) global activation re-read and batches entries — on GSQ's down types this is the shipped tuned form; if it saves >=15% per byte on the down third, affected layers shed 5-10% -> window -3.5-7% -> solo ~+4-7% (184-189), pair/conc similar. **Falsifier (law 17)**: 3+3 interleaved same-boot, loads recorded: arm serial-class windows must drop >=0.4 ms (~3%) with per-layer-window VRAM-hit us falling on the {x,6} layers; native_grouped_parity bitwise 6-case must pass (it already tests this exact pair). If the window moves <2%, the generic down was not binding at this geometry — revert and let P3 carry.

**P2. Repack variant: d 6 -> d 20 on the 33 Q5_0-down layers (pack-side, zero code).** {18,6}->{18,20} -8.6%/blob, {23,6}->{23,20} -7.1%/blob: pool -2.81 GB (-4.65%), cache slots ~8690 at the same VRAM, and 33 layers land on the tuned kSplit<20> sub8 staged form **today**. Expected: bytes -4.65% + form -> serial +5-9%, conc +6-10%. Quality gate: down 5.5 -> 4.5 bpw on 33 layers — the 2x MMLU floor law decides (v31's crown is 95.71; the pack builder's d20 downs are the IQ2_XS pack's proven default). Run it as a parallel pack, not a replacement. **Falsifier**: same-boot 3+3 vs v31 + MMLU-Pro 2x >= 93.3; if lanes gain <3%, bytes were not binding — keep v31 and lean on P1/P3.

**P3. S26-class pair kernels for {18,6} and {23,6} (engine, the win condition).** `s26_launch_l` is geometry-exact for our shapes (n_embd 2560, n_ff 640/hb 20, 80-dot gu rows) and already templated over (gu, d) — extend the guard at iq_kernels.cu:4786 from `gu==21 && d==20` to include `gu==18||gu==23, d==6`, instantiate `s26_gu_l_kernel<18/23>`, `s26_down_l_kernel<6>`. Precedent: 1.09-1.12x bitwise vs the *tuned* GSQ pair (s26 comment, :4789-4791). On UD's pairs (currently one tuned third + one generic third pre-P1) the relative win is larger. Expected stacked on P1: +5-10% window further; this is the item that pushes per-byte below GSQ's 0.83x and makes the +20% bytes handicap winnable. **Falsifier**: bitwise harness (s26 class is bitwise by construction, :2140-2142) + 3+3: must clear +3% over the P1 state on serial windows, else park.

**P4. Tier and miss-path parity (config, one boot, zero code).** Add to the UD config what GSQ pins: `--pcie-frac 0.35 --pcie-mode dma` (UD's PCIe share is 3-5x GSQ's), `--resident-experts`, `--adapt-decay 0.85`, and `--vram-reserve-mib 2000 -> 1024` (+~0.95 GiB -> ~+390 slots, 33.8% -> 35.4% resident). Expected: serve hit rate 92.2% -> 94-96%, miss-path terms shrink -> +1-3% every lane. **Falsifier**: single-boot same-hour A/B: routed VRAM% >= +2pp AND rows/s >= +1%; hit rate up with flat rows/s means the miss path was not binding — revert the extras, keep only what pays.

**P5. gu-side micro-arms for IQ4_XS (engine small, env-first).** (a) add 23 to `kStageIqGrid` (:820) — staged codebook, precedent 16/17/22; (b) widen `expdec_gu_ok` (:3695) to include 23 and `expdec_down_ok` to include 6 post-P1. Test (a) with `STRATA_IQ_STAGE_GRID`/`STRATA_EXPDEC` env A/B before any code lands. Expected +1-3%. **Falsifier**: 2-leg env A/B neutral -> drop; code only after the env arm clears +2%.

**P6. IL-Q8 emitted-table revisit (parked, low priority for THIS pack).** The parked arm (-1.0/+3.7/+1.6) was priced on IQ2_XS's Q8_0-heavy dense; UD-IQ4_XS's dense is Q6_K-dominant (IL already on) and its HC is bf16 rows. Residual Q8_0 dense: shexp + 1 attn layer. Expected 0-2%. Revisit only via the tuned emitted table (the micro's real cell wins), never the bare flip. **Falsifier**: the parked protocol; same +4% bar.

**P7. Byte-aware adaptive tier (structural, park behind P4).** The adaptive tier swaps whole blobs without weight: evicting one 2.87 MB {23,6} admits two 1.44 MB {22,42} blobs. Normalize swap value by bytes and add per-layer slot budgets (profile rank is global today; heavy layers consume tier slots faster). Expected +0.5-1.5pp hit rate. **Falsifier**: dispatch-report swap log before/after; adopt only if hit rate rises without hot-layer regression.

Win condition arithmetic (honest): at P1+P4 the window lands ~14.3-14.9 ms (parity-minus — the +20% bytes/hit remains). Clearing GSQ needs P3 (+5-10%) plus P2's byte cut or P5 stacking: 13.59 x 1.20 x (form factor <=0.86) ~= 14.0 ms — i.e. **the plan must land P1, P3 and (P2 or P5+P4) together** for solo/conc to cross 233/303. The quality crown (95.71 MMLU, +0.67 bpw) is what buys the right to carry those bytes; every item above keeps it gated by the 2x MMLU law.

## 4. Evidence file map

- Manifests: /home/svg/LLM/data/packs/unsloth-ud-iq4_xs-v31/{native_experts.txt,index.txt,conversions.json,compat-bf16.json}; GSQ /home/svg/LLM/data/packs/iq3_s/native_experts.txt (v3, total 50292326400).
- GGUF shards: /home/svg/LLM/data/models/UD3.1/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3,4}-of-00004.gguf (census in this digest, section 1.3).
- Kernels: src/kernels/cuda/iq_kernels.cu — kSplit table :814-1367, kStageIqGrid :820, Fmt<6> :740, native_down_kernel :1937, native_down_multi_kernel :1970 (generic branch :2072-2126), native_gu_swiglu_kernel :2556, launch_gu_swiglu :3859, launch_down :3883, gu_swiglu_on :3839, g_stage_grid :3071, expdec gates :3694-3700, S26 :4786, V2K :4802, native_expert_grouped :4772, STRATA_{GU,D}_FMTS :755-764, is_iq :3029, iq_row_bytes :3902.
- Dense: src/kernels/cuda/native_mmvq.cu — native_mmvq_il_supported :3276, il_q8_on :3142.
- Configs/launchers: config/custom-ud-iq4xs-8081.json, config/custom-iq3_s-8081.json, launch-custom-{iq3s,ud-iq4xs}-8081.sh (standing env byte-identical; both `--mtp-en/rt` + draft_vocab en).
- Logs: logs/custom-ud-iq4xs.log (boot 14:31 onward is the valid pack; the 14:06 boot predates the v31 shard rename), logs/custom-iq3_s.log, logs/custom-ud-iq2_xs.log.
- Banks: campaign/results/fork-ud-first-bank.md (the same-hour decomposition + SMALL_UNR attribution + IL-Q8 verdict), campaign/tracker/lanes.csv, campaign/results/dispatch-{iq3s-gsq,ud-iq2xs}.json (upstream-era captures; a ud-iq4xs dispatch capture does not exist yet — capture one during the P1 battery).
- Sibling doctrine: .docs/research/digest/2026-11-06-ud-slowdown-kernels.md (kernel taxonomy), 2026-11-04-past-ceiling.md (bytes formulation).
