# T14B — TRAINER EXECUTION spec: the FastMTP retrain pipeline (r18)

Date: 2026-10-18. Owner lane: T16 W2 / T17 #3 "RETRAIN v2". This is the execution spec for the TRAINER only —
it consumes the T14B harvest and emits an rt-dir-compatible MTP head; the gates are T14C's ladder and are only
summarized here as the handoff contract (Section 10).

Banked inputs (no new research performed; every external claim below cites these):
- `SVG-CUSTOM-TODO.md` T11.R3 (GO recipe, :793-806), T17 QUEUE (:1096-1236), STEP 0 RESULTS (:1238-1269).
- `campaign/research/T14A-fastmtp-method-recipe.md` (arms 0-4, risks R1-R5).
- `campaign/research/T14B-fastmtp-training-infra.md` (infra audit: dump formats, export chain, VRAM budget, F-77..F-118 laws).
- `campaign/research/T14C-fastmtp-acceptance-math-gates.md` (G0.1-G5 ladder, v1 gate, kill-rule).
- `campaign/research/T17C-spec-algo-leaps.md` §5.1 (VAT / LK / AdaSPEC objective swaps, warm-start discipline).
- `campaign/research/T17F-training-quant-context.md` §3.1 (Draft-OPD rejection replay, LongSpec position-offset aug).
- STEP 0 #5 (2026-10-18 corrected read): position is the PRIMARY deep-acceptance deficit (real-corpus deep 53.1%
  vs shallow 93.6% = -40.5pt), content secondary (-14.9pt). Position-oversampling CONFIRMED as a T14B input;
  content-matched draft data additionally promoted; MIMIC corpus ABSENT from Custom-data (repo prose used).

Hard constraints honored: one RTX 5090 (32 GB, exclusive during training), 9950X3D, 184 GB RAM, no cloud compute.
No engine boots, builds, training runs, or server calls were performed while writing this spec.

---

## 1. Summary

Custom PyTorch trainer, dense-only, warm-start, on a separate venv: train the drafter's ~90.5 M dense params
(everything except the two `mlp.experts.*` tensors) with recursive teacher-forced CE (FastMTP, arXiv 2509.18362)
against the harvest of the SERVED quantized backbone's hidden states; ship via the in-tree export chain
(`mtp_pack.py` -> `mtp_rt.py`) into a new `rt-r18` dir beside `rt-r17`, stock `experts.bin` + stock
`draft_vocab.bin` byte-identical (the F-114 validated config); the runtime's own `--mtp-q4 all` path does the
Q4_0 direct-cast at load — never GPTQ (F-201 law). Three objective swaps (VAT / LK / AdaSPEC) and the beta x K
matrix run as config-selectable arms on the SAME trainer, <= 6 arms in the first sweep, each 2-6 h and 11-16 GB.
Rollback is structural: the standing config still points at `mtp/rt` and is never written.

---

## 2. Trainer architecture and framework choice

### 2.1 Decision: custom PyTorch trainer. Speculators 0.6.0 is the recipe, not the base.

- **Custom torch trainer = GO (primary).** The block we serve is `qwen4exp`-family with components no external
  trainer implements: the hyper-connection mixer (`hyper_connection_mixer.input_mix_weight_{down,up}` 10240<->320,
  `attn/mlp_hyper_connection.*` — `mtp/rt/dense.txt`), the QSA indexer attention, and the split-add
  fc_embedding/fc_hidden pair. Red Hat Speculators 0.6.0 supports Qwen3-Next and Qwen3.5 incl. MoE — **not**
  `qwen4_exp`/Flash-Next (no Gated-Residual block, no indexer-MTP wiring; T14B §2.5 "NO-GO as-is, GO as recipe").
  Its loss/schedule/export SHAPE is what we implement (per-step weight alpha_k = beta^(k-1)/sum, frozen
  embed_tokens + lm_head, target-generated data).
- **torch on sm_120 = settled.** Native since 2.7.0-cu128 stable; current 2.9-2.11 ships cu130 wheels (T14B §2.5).
  CUDA 13.4 libs at `/usr/local/cuda-13.4` (custom-iq3_s-8081.json `lib_dirs`).
- **The three objective swaps are riders, not forks.** VAT / LK / AdaSPEC differ from the base arm ONLY in the
  loss term and (for AdaSPEC) the data filter — same module, same harvest, same export (Section 4).

### 2.2 Environment — new trainer venv; the repo `.venv` stays untouched

The task offered "PyTorch + the existing `.venv`". Resolved AGAINST that option, on the banked law (T14B P0:
"do NOT touch the repo's `.venv`") plus one new fact: the repo `.venv` is python 3.15.0 and carries the ENGINE
TOOLING's pins (numpy 2.5.3, gguf 0.19.0, huggingface_hub 2.2.0, datasets 5.1.0 — probed this session). A torch
install churns the numpy/gguf resolver and risks the campaign scripts (`mimicbench2.py`, `probe-lane.py`,
`mtp_rt.py`) mid-weekend. Serving tooling must not share an env with a pip experiment.

```
P0  python3.14 -m venv ~/venvs/mtp-train        # see wheel note below
    ~/venvs/mtp-train/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130 \
        numpy safetensors
    # gguf-py: use the vendored third_party/llama.cpp/gguf-py (head dequant for the bound-row teacher)
    # optional, G0 only: pip install speculators==0.6.0   (pulls transformers; isolated in THIS venv)
    verify: torch.cuda.get_device_capability() == (12, 0)
```

Wheel risk, with decision rule: the repo `.venv` is cp315; cu130 torch wheels for cp315 may not exist. If the
cp315 wheel resolves, either python works; if not, pin the trainer venv to the newest python with a published
cu130 wheel (python3.13/3.14 from Fedora 45 repos). Do NOT upgrade the repo `.venv`'s python to chase it.

### 2.3 The module (`mtp_head_qwen4exp.py`) and its forward conventions

Mirror of the 31-tensor block, initialized from the SHIPPED tensors (warm-start law, Section 3.4):
`tensors/<name>.bin` BF16 + `mtp-manifest.json` (the `mtp_fetch.py` layout; stock copy already at
`/home/svg/LLM/Custom-data/mtp/tensors/`, sha256-verified).

Forward conventions — ALL previously convicted bug sites, cite-checked in the current tree:
1. hnorm STREAM mode (production arm `--mtp-hnorm stream`): per-stream RMS over the hc slices of the 10240 input,
   each scaled by its slice of `pre_fc_norm_hidden` (T14B §2.2, mtp.cpp:941).
2. fc_embedding + fc_hidden are a SPLIT-AND-ADD pair over concat(e, h) (F-81 fix 2; the phaseb adapter's
   `eh_proj` semantics, tools/mtp_phaseb_adapter.py header).
3. Norms: stored RAW in tensors/, forward uses the Gemma fold `w -> 1+w` (F-109/F-110 double-fold law;
   the fold normally happens at rt build, tools/mtp_rt.py:85 — the trainer forwards folded, exports raw).
4. hc-mixer read: hc_norm (f32, folded) + down 320x10240 -> up 10240x320, the fused_gr read semantics.
5. QSA self-attn: q_proj 12288x2560 = 48 heads x [q|gate] 256 interleaved, q_norm/k_norm f32, indexer MQA
   4 heads / key 128 with the top-2048 block budget (degenerate-safe on short teacher-forced windows; T14B §2.2).
6. MoE: router bf16 512x2560, top-10, 512 routed experts FROZEN, shared expert with ONE scalar sigmoid gate
   (F-81 fix 4; router top-10 at mtp.cpp:1141-1148).
7. Output mixer + head: the head is the 40,525-row subset of the main 248,320x2560 Q6_K `output.weight`,
   GATHERED and FROZEN — exactly what serve-time `bind()` does (mtp.cpp:690-781; subset file = the 40,525
   int32 ids in `rt/draft_vocab.bin`). The trainer never trains these rows ("frozen embed + BOUND lm_head",
   T11.R3). Dequantize the subset rows once at trainer init (vendored gguf-py `to_float`).
8. Teacher-forced recursion: depth k fuses the frozen embedding of the GROUND-TRUTH token t+k with the head's own
   depth-(k-1) output R; CE vs token t+k+1; output fed back (FastMTP arXiv 2509.18362; T14A §4 Arm 1).

SDPA attention only; NO GDN anywhere in the trainer (the drafter's self_attn is QSA; the target's GDN layers ran
inside the engine during harvest).

### 2.4 G0 parity gate (F-80 law — hard, before ANY training step)

Boot the lane binary with `STRATA_MTP_TAP=/tmp/tap` (mtp.cpp:77, alloc :288-293, dump block :1229-1240: writes
the drafter's `R_`, `sample_`, `head_logits_` for the first window, pinned, capture-legal, inert without the env).
Feed one real window; run the torch module on the same inputs; diff STAGE BY STAGE (cos > 0.9999, argmax match).
The tap exists in the shipping binary today; the r17-era Windows trainer proved this exact gate catches the
0.167-repeats class (F-80). Without G0 pass: no training.

---

## 3. Data: the T14B harvest contract

### 3.1 Base records — `--dump-final-r` (in-tree, production-inert)

- Flag: `--dump-final-r PATH` (arg parse generate.cpp:2158, member :638, handle :13768-13769).
- **Record format (code-pinned this session, CORRECTS the T14B dossier's int32 note):** per position,
  `int64 pos, int64 tok` written as one `int64_t posrec[2]`, then `g.hc * g.n_embd` f32 (4x2560 = 10240 floats)
  of the session residual R — the drafter's exact serving input state (generate.cpp:13969-13973:
  `cudaMemcpy(final_r_host..., ss.R ...); fwrite(posrec); fwrite(final_r_host...)`).
  Record size = 16 + 40960 = **40,976 bytes**. Written from the token drive loop (one D2H of 160 KB + fwrite
  per token). Alignment: record i holds the state AFTER consuming token i; the drafter's serving input at i is
  `(R_i, emb(tok_i))` and its label is `tok_{i+1}` — pure offset bookkeeping; keep conversations contiguous,
  drop cross-seam pairs (F-77 law; T14B P1).
- Corpus: our own serving-traffic mix. STEP 0 #5 killed the "MIMIC corpus" assumption (absent from Custom-data);
  the confirmed substitute = repo prose (45 files, the depth-lane-real.py set) + standing battery prompt mix.
  Content-matched draft data is PROMOTED (STEP 0 #5), and the corpus is target-replay text so it is
  self-distilled by construction (T14A Arm 0 item 2).
- **Position discipline (STEP 0 #5, the confirmed lever):** harvest MUST span deep positions (the probe showed
  deep 53.1% vs shallow 93.6% acceptance is primarily POSITION-driven). Include prompts long enough to cover
  >= 64k-class positions at full weight (oversample them ~2x relative to traffic share), and keep `pos` exact so
  training windows cut at real positions (F-105 law: cover what you serve).

### 3.2 Extensions riding the same pass (the "rejection/pick pairs")

1. **Verifier p-list dump** (enables LK; T14A Arm 0 ext., ~half a day): the coupled verifier already materializes
   the target's post-chain top-64 per row (sampler.cu coupled q-emission :947-1016 class; T14A F7/F15 plumbing).
   Record: `(pos, count, then count x {int32 id, f32 logp})`.
2. **On-policy rejection replay tuples** (Draft-OPD class, arXiv 2605.29343; T17F #2): during the same replay,
   run the DRAFTER as served (coupled, draft-temp 0.85, min-p 0.5) and record per offered row
   `(pos, depth, draft_tok i32, accepted u8, target_pick i32)`. The exact-match verifier emits this for free
   (T17F §3.1). Upweight rejected positions 2-4x in the CE (the config's `replay_weight`).
3. **LongSpec position-offset augmentation** (arXiv 2502.17421; T17F #3): training-time only — draw random large
   position offsets for a fraction of windows so 240k-class positions are seen on short data. Zero harvest cost;
   `pos_aug` config flag. Kept per STEP 0 #5 ("LongSpec aug keeps its slot").

### 3.3 Throughput probe FIRST (30 min, before building anything)

Dump 100k tokens and measure the rate. The hook is per-token; if it runs decode-class (~200 tok/s) the 16M-token
corpus costs ~22 h — still survivable, but the batched-prefill extension (~50 lines after each prefill chunk,
plumbing = `ver.final_R_all()` / layer_stage host buffer, generate.cpp:14146/14320-14330 per T14B P1) turns it
into ~45 min at the measured 6,291-6,972 tok/s prefill. Probe decides which arm of P1 we build.

### 3.4 Scale and disk

16M positions target (T14A/T14B sizing; RH's 5k-sample precedent says scale is NOT the binding constraint —
T17C §5.1). f32 harvest = ~650 GB; bf16 sidecar conversion at write = ~330 GB. `/home` has **705 GB free**
(df this session; was 735 at T14B time — do not let the f32 variant plus the pack coexist). Budget:
bf16 records preferred; if f32 kept, cap ~12M positions and delete after export. Add: torch venv ~4 GB,
export artifacts ~7 GB, G0 sibling ~20 GB. Peak fits with ~350 GB slack in the bf16 plan.

---

## 4. Loss implementation plan — one trainer, config-selectable objectives

Single entry point (`train.py --config <json>`); the config JSON fully determines the arm:

```json
{ "objective": "ce | ce+lk | vat | adaspec",
  "beta": 0.8, "K": 6, "lr": 5e-4, "epochs": 3, "seq": 1024, "windows_per_batch": 4,
  "replay_weight": 2.0, "pos_aug": true,
  "lk":  {"form": "hybrid | logalpha", "eta": 3.0, "coupled_variant": true},
  "vat": {"survival_head": true, "decay": "rejection_anchored"},
  "adaspec": {"reference_ckpt": "runs/a1/final", "keep_top": 0.4, "kd_temp": 0.7},
  "data": {"records": "...", "plists": "...", "replay": "..."},
  "out": "runs/<arm-id>", "seed": 0 }
```

- **Base CE (objective `ce`)** — FastMTP recursive teacher-forced CE (arXiv 2509.18362), per-step weights
  `alpha_k = beta^(k-1) / sum_j beta^(j-1)`, CE over the 40,525-row subset vs GT next tokens. Target tokens
  OUTSIDE the subset are DROPPED from the loss (those positions can never be drafted at serve — draft_vocab
  handling, Section 9.3). beta and K are config fields (the arm matrix, Section 6).
- **LK (objective `ce+lk`, arXiv 2602.23881)** — second-stage rider on an `ce` checkpoint: loss =
  `lambda*KL + (1-lambda)*TV` with adaptive `lambda = exp(-eta*sg[alpha])`, or `-log alpha`, over the UNION of the
  logged p-list support and q's support; `alpha = sum min(p,q)` computed on the subset only (off-vocab mass = 0 —
  the truncated-sub-vocab-native property, T17C §5.1). `coupled_variant: true` adds the F15 collision form
  `sum p*q` as a logged alternative term (armed, not assumed; T14A F15). 1-2 epochs, LR 2e-5, warm-start (T14A Arm 2).
- **VAT (objective `vat`, arXiv 2608.30135)** — trains against the simulated verify loop: (i) a jointly trained
  lightweight binary head predicting per-position survival; (ii) verification-adaptive weighting — full weight to
  each sample's chain UP TO its first rejection, decay re-anchored THERE (replaces the fixed beta^(k-1) with a
  per-sample schedule; the rejection tuples of 3.2.2 ARE the supervision). +11.4% AL class published on
  EAGLE-3/DFlash; objective-only (T17C §5.1, source 20).
- **AdaSPEC (objective `adaspec`, arXiv 2510.19779)** — two-stage selective KD. The A1 checkpoint IS the
  reference (zero extra cost, T17C table #3): compute per-token loss gaps on held-out dump, keep top-40% by gap,
  retrain from the A1 init on the filtered set, forward-KL with KD temp 0.7 (arXiv 2410.10141; T14A Arm 3).
  Their ablation that bottom-40% training HURTS is the guard-rail in our small-data regime.
- **Optional additive stock-head KL** (`ce` config, `stock_kl: true`): KL toward the frozen stock head under the
  SAME window forward — the error-cancels conditioning insight (F-109/Q3); ~15 lines; only if a gate stalls
  (the proven 0->57 fallback, F-115/F-116; T14B P3).
- **Optional last pass if the serving gate shows cast loss:** ~200 steps fake-quant (Q4_0-projection / q8_0-dense)
  aware finetune at LR 5e-6 (DWQ-class; T14A risk R2 mitigation; T14B P3 "fake-quant" rider).

### 4.1 Training-state discipline (the r17 post-mortem is a hard law here)

The r17 57% plateau = fresh-init + pure-KL + double-quant, FULLY ATTRIBUTED (F-115/F-116; T17C §1 item 2) —
NOT a distillation limit. Therefore, for EVERY arm: warm-start from the SHIPPED dense weights (the stock rt
dense dequantized — never fresh-init); CE family objective primary (never pure-KL); the shipped config keeps
stock experts so the chain's quant passes touch only the dense (one q8_0/q4_0 cast at serve).

---

## 5. VRAM and time budget per arm (dense-only config)

| Item | Bytes | Basis |
|---|---:|---|
| Frozen routed experts (bf16, fwd only) | 5.04 GB | T14B §4 |
| Frozen dense fwd copy (bf16) | ~0.18 GB | 31-tensor block minus experts |
| Frozen bound head rows 40,525x2560 bf16 | ~0.21 GB | + tokenizer embed gather on CPU |
| Trainable ~90.5 M dense: bf16 w + grad + AdamW fp32 m,v + fp32 master | ~1.45 GB | T14B §4 |
| Activations (4x1024-pos windows, expert checkpointing) | 2-6 GB | T14B §4 |
| CUDA context + workspace | 1.5-2 GB | measured class |
| **Total** | **~11-16 GB of 32 GB** | matches T11.R3 "8-14 GB" |

Wall clock per arm: **2-4 h** for 3 epochs x 16M tokens (40-80 effective TFLOPS with MoE gather + 40k-row CE;
RH reference: 443 s on 2xH200 for the 9B head — T14B §4), inside T11.R3's 2-6 h/arm band. VAT adds the survival
head (~2 M params, negligible) and one extra forward over the depth axis (~+30%). LK adds one epoch-class pass
(~1-2 h). AdaSPEC costs one reference scoring pass (~30 min) + a shortened retrain (~2 h, filtered data).
Full-expert FT stays NO-GO for v1 (~21-26 GB with 8-bit Adam + routing autograd nobody has written here;
contradicts F-114; T14B §4/§5). LoRA NO-GO (trainable surface already 90.5 M; no merge path in the rt chain).

Batteries and training run SERIAL (own-traffic box law, T16 preamble): never train while a battery boots.

---

## 6. The arm matrix (first sweep: 6 arms, beta x K x objective)

All arms warm-start from the shipped dense (Section 4.1); all ship through the same export; all are ranked by the
SAME gate (G0.2 stair read, Section 10). Selection = best arm by pos-2/3 marginals at pos-0-neutral.

| # | Arm id | objective | beta | K | Why it exists | Cost |
|---|---|---|---|---|---|---|
| 1 | a1-b8-k6 | ce | 0.8 | 6 | PRIMARY. Flattened weighting funds the pos-2-6 deficit (T11.R3; F14: 0.8 keeps ~9% of loss on depth 6 vs 3.2% at 0.6) | 2-4 h |
| 2 | a2-b6-k6 | ce | 0.6 | 6 | The PUBLISHED FastMTP/RH weighting — F14 is an armed extrapolation; this arm falsifies it | 2-4 h |
| 3 | a3-b8-k8 | ce | 0.8 | 8 | Train deeper than serve=6 (T14A Arm 1: "K>=serve depth"); tests whether depth-7/8 supervision lifts 4-6 | 3-5 h |
| 4 | a4-lk | ce+lk | 0.8 | 6 | Direct sum min(p,q) optimization on a1; truncated-sub-vocab-native (T17C #2; LK +5.6% on a DeepSeek MTP finetune) | +1-2 h |
| 5 | a5-vat | vat | n/a (learned) | 6 | Rejection-anchored decay replaces the fixed schedule; simulates the served verify loop (T17C #1; +11.4% AL class) | +30% |
| 6 | a6-adaspec | adaspec | 0.8 | 6 | Selective top-40% refilter of a1's residuals; reference = a1 (free) (T17C #3; up to +15% rel.) | ~2-2.5 h |

Sequence: G0 dry-run (Section 10) -> a1 first, a2/a3 same weekend (cheap, same data) -> a4/a5/a6 SECOND pass only
against the measured residual deficit (T14A ranked summary: arms beyond the core "only against the measured
residual"). Expected band (T17C): retrain base +5.7% M1, riders stack multiplicatively on acceptance length ->
realistic 225-232 M1 BEFORE depth re-unlock (T17C §5.1 priced composite).

Kill/bank discipline: every arm's artifacts live in its own run dir; no arm's export ever overwrites another's;
the best arm graduates to `rt-r18` (Section 9) — the others stay checkpoints.

---

## 7. Export chain and the Q4 direct-cast step (format-compatible with the mtp-q4 loader)

### 7.1 Chain (in-tree, validated end-to-end in the r17-era; T14B §2.3/§4 P4)

```
trainer -> <out>/tensors/<name>.bin (BF16 LE, norms RAW w, NOT 1+w — F-109 law)
        -> <out>/mtp-manifest.json (name, dtype "BF16", shape, file, sha256 of the TRAINED tensor)
        -> python tools/mtp_pack.py --src <out> --experts q2_0 --out mtp-r18-q2_0.gguf
        -> python tools/mtp_rt.py --gguf mtp-r18-q2_0.gguf --out /home/svg/LLM/Custom-data/mtp-iq3s/rt-r18
        -> copy rt/draft_vocab.bin and rt/experts.bin BYTE-IDENTICAL into rt-r18   (F-114 config)
        -> value audit: every UNtrained tensor byte/quant-equal vs rt (one script; the F-111..F-113 audit)
```

`mtp_rt.py` writes: `experts.bin` (512 blobs, `cpu::BLOB` = 3*(2560*640*18//64) = 1,382,400 B each — gate/up rows
interleaved 2r=gate/2r+1=up then down, Q2_0 codes + fp16 scales planes; tools/mtp_rt.py header), `dense.bin`
(q8_0 for the big projections, bf16 for hc mixers/router, f32 norms with the +1 fold), `dense.txt`
(`name kind rows cols offset bytes`, kind in {q8_0, bf16, f32}). The loader parses exactly this
(mtp.cpp:313-323) and REQUIRES q8_0 for 9 named tensors (mtp.cpp:363-366) — the trainer must not perturb the
kind map (tools/mtp_rt.py `Q8` set). The two shipped dense.txt files differ only in line endings (rt = LF,
rt-r17 = CRLF); the tokenizer-style parser tolerates both, but emit LF.

### 7.2 The direct-cast at load — what "ship bf16, serve Q4" means concretely

The runtime's `--mtp-q4 all|proj|head` (parse generate.cpp:2231-2234; wiring `set_q4` generate.cpp:4712,
mtp.hpp:69) performs the Q4 cast at LOAD with ggml's reference quantizer — there is no quantized training export:

- **Projections:** `make_q4_dense` (src/core/mtp.cpp:210-243, called from `load()` at mtp.cpp:337) takes EVERY
  `q8_0` dense tensor with `cols % 32 == 0` (mtp.cpp:216), dequantizes each row (Q8_0 row = cols/32*34 B) and
  requantizes `GGML_TYPE_Q4_0` (row = cols/32*18 B, mtp.cpp:225-229) into one device buffer `dense4_`; at forward
  `wq()` hands out the Q4_0 copy (mtp.cpp:202-208). Standing config arms it: `"--mtp-q4", "all"`.
- **Draft head:** `make_q4_head` (mtp.cpp:245-269) reads the BOUND 40,525-row subset (gathered from the main Q6_K
  `output.weight` at bind, mtp.cpp:766), dequantizes each row via the head type's `to_float`, requantizes Q4_0,
  replaces the rows in place and sets `dhead_type_ = GGML_Q4_0` (mtp.cpp:259-263; called from `bind()`
  mtp.cpp:773). 40,525 rows: 81.2 MiB Q6_K -> 58.4 MiB Q4_0 read/step.
- **Law:** direct-cast ONLY, never GPTQ (F-201). The trainer therefore does NOT emit quantized weights; it emits
  bf16 and the gate measures the AS-SERVED cast (`--mtp-q4 all --mtp-hnorm stream` identical in both battery
  arms — train-serve arm parity, T14B P6). If the cast costs > 2pp acceptance at the gate: per-tensor AMAX
  recalibration pass / weight-clip, or the fake-quant finetune of Section 4 (T14A risk R2).

---

## 8. Trainer-side artifacts (runs, harvest, checkpoints)

```
/home/svg/LLM/Custom-data/mtp-iq3s/r18-train/
  records/            harvest: records.f32|f16 (dump-final-r), plists.bin, replay.bin, PROBE.md (rate + sizes)
  runs/<arm>/         train_config.json, ckpt-last.pt, ckpt-best.pt, loss.jsonl,
                      tensors/ + mtp-manifest.json (the exportable snapshot), audit-vs-stock.txt
  export/             mtp-r18-q2_0.gguf, rt-r18 staging, gguf report json
```

Resume law: harvest append-only and resumable; each run dir is self-describing (config + seed + data sha256)
so any arm can be reproduced or abandoned without touching the others. Delete the harvest after the winning
arm's export passes the audit (disk, Section 3.4).

---

## 9. Runtime layout, rollout, rollback

### 9.1 The rt dir contract (what the loader consumes)

Four files (F-112): `dense.bin` + `dense.txt` (mtp.cpp:313-325), `experts.bin` (mtp.cpp:344-359, 512 x BLOB),
`draft_vocab.bin` (40,525 int32 token ids = the en subset; 162,100 bytes). Boot-log verification lines: the
drafter's files_s MiB print (mtp.cpp:528 region) and `bind()`'s "draft head over 40525 tokens" (mtp.cpp:770).

### 9.2 Directory + flip convention (rt-r18 beside rt-r17)

`/home/svg/LLM/Custom-data/mtp-iq3s/` is the retrain dir: `mtp-r17-q2_0.gguf`, `rt-r17/`, `r17-fetch/` live
there today (ls this session; rt-r17's `dense.bin` DIFFERS from stock while `experts.bin`/`draft_vocab.bin` are
byte-identical — the r17 precedent of "new dense, stock rest", sha256-checked this session). r18 follows it:

- Production/standing config `custom-iq3_s-8081.json` keeps `"--mtp", "/home/svg/LLM/Custom-data/mtp/rt"` —
  NEVER written.
- A/B config = COPY `custom-iq3s-rt18-8081.json` with only `--mtp .../mtp-iq3s/rt-r18`, the log path, and the
  port changed; A/B launcher = copy of `launch-custom-iq3s-8081.sh` pointing at that config (Launch Law: never
  setup.py, start via launchers). Rollback = boot the standing launcher; adoption = flip the standing config's
  `--mtp` string after the gates pass (one line, one command to revert).

### 9.3 draft_vocab "en" handling

The subset file in an rt dir is ALWAYS named `draft_vocab.bin`; the CHOICE lives in the config key
`"draft_vocab"` (setup.py:5014-5015 `DRAFT_VOCABS = {"cjk": "draft_vocab.bin", "en": "draft_vocab_en.bin", ...}`;
standing config carries `"draft_vocab": "en"`). `refresh_draft_vocab` (setup.py:5282-5301) copies the shipped
subset over the rt dir's copy only when the existing bytes match a SHIPPED subset and differ from the chosen
one; a stock-hash copy in rt-r18 is left untouched. The trainer consumes `rt/draft_vocab.bin` directly
(mtp.hpp:224 `vocab_file()` = `rt_dir + "/draft_vocab.bin"`, overridable via `set_draft_vocab` mtp.hpp:72):
CE over exactly these 40,525 rows; off-subset targets dropped (Section 4); the head rows the trainer freezes
are the SAME rows `bind()` gathers (mtp.cpp:726-777) — zero head-distribution shift (T14A §3).

---

## 10. Handoff into the gates spec

### 10.1 G0 dry-run on the 9B sibling — concrete form

- **Which sibling:** Qwen3.5-9B — the exact example model of the Red Hat Speculators 0.6.0 article ("a
  Qwen3.5-9B head trained in about 443 seconds on 2xH200"; T14B §2.5, T11.R3 :798). It is a SUPPORTED speculators
  verifier class (Qwen3.5 incl. MoE), which is the point: it exercises the recipe machinery we cannot point at
  `qwen4_exp`.
- **Where the weights come from:** HF hub under a PINNED revision (the `mtp_fetch.py` discipline — fixed sha,
  206-content-range-checked range reads, #327), via `huggingface_hub` (already in the repo `.venv`, 2.2.0) or a
  speculators/transformers download in the trainer venv. ~18-20 GB bf16; disk fits (Section 3.4). This is a
  weights FETCH, not cloud compute — the no-cloud law binds compute, and the box already uses this exact pattern
  for the BF16 MTP-tensor fetches (tools/mtp_fetch.py header).
- **What runs (one pass, 15-30 min budget):** speculators 0.6.0 `prepare_data` on a small sample (e.g. 64
  samples x 4k ctx) -> one MTP-head finetune for a handful of steps -> checkpoint save/reload. Success =
  torch sm_120 device capability (12,0) confirmed; one full data->train->export cycle completes; loss decreases
  over the steps; no OOM at the venv/python/driver stack we will use for the real arms.
- **What it de-risks:** env (wheel/python/driver), the recipe's data-record shape (a reference for our record
  alignment), and the schedule constants (beta normalization). It is NOT a quality gate and its numbers do not
  transfer (different model, different head class).
- **Fallback if the hub is unreachable:** the free stock ROUND-TRIP test instead — run the STOCK tensors through
  our export chain into rt-r18 untrained and require stock-class acceptance on one battery (the F-114 test;
  T14B §5 dry-run row). This must happen before a1 regardless, as the export-chain gate.

### 10.2 The gate ladder the winning arm enters (T14C, summarized for the handoff)

1. **G0.1** greedy bit-exact mimic, `campaign/bit-check.py` (minutes; binary) — only sanity-checks the boot;
   a drafter swap is different-bits BY DESIGN, so the real gates are acceptance + MMLU, not bit-identity
   (T16 #9 law).
2. **G0.2 M3d stair read** (`STRATA_SPEC_DEPTH=1`, T14C :182-190): the trainer's v1 gate — pos-2/3 acceptance
   +>= 8 pp each, pos-0 regression <= 1 pp, offered/window reported. This read SELECTS among the 6 arms.
3. **G1** one config-boot battery, M1 pre-screen +4% (T14C :190-196). **G2** ABAB 3v3 adoption bar (+4% M1,
   abab-run.sh pattern; T16 anchors to re-bank at clock 3135). **G3** MMLU-Pro 2x + mimic band (quality floor
   MMLU 95.0 / mimic 87.9; different-bits law). **G4** F-229 ungated re-boot once offered >= ~4. **G5** min-p
   re-sweep / depth schedule — the kill-rule domain: extend served depth only while marginal acceptance >
   6-9% breakeven (T11.R3; T14C :221-230).
- Battery hygiene: arms run `--mtp-q4 all --mtp-hnorm stream --spec 6 --spec-min-p 0.5` IDENTICAL to production
  in both battery legs; batteries serial; five-boot means per T16; quality floor absolute (No-Regression Law).
- Deliverables the gates spec receives from this pipeline: `rt-r18` (the winning arm), its
  `runs/<arm>/train_config.json` + `audit-vs-stock.txt`, the G0.2 stair expectation sheet (which positions
  should move), and the rollback string (Section 9.2).

---

## 11. Contradictions and their resolutions

1. **"PyTorch + the existing .venv" (task) vs T14B P0 "do NOT touch the repo's .venv".** Resolved for the
   research law with a new code fact: the .venv is python 3.15 and pins the engine tooling's numpy/gguf; the
   trainer gets its own venv (Section 2.2), which also isolates the speculators install for G0.
2. **T14B dossier's record format (`{int32 pos, int32 tok}`) vs the code.** The code writes `int64_t posrec[2]`
   (generate.cpp:13971-13972) — 16-byte header, 40,976-byte records. The spec follows the CODE; the dossier's
   line is corrected here.
3. **"rt2 in mtp/" (T14B P4 example) vs "rt-r18 beside rt-r17" (task).** The task's convention matches the
   on-disk reality (rt-r17 lives in mtp-iq3s/); rt-r18 goes to `/home/svg/LLM/Custom-data/mtp-iq3s/rt-r18`.
   The T14B P5 rollout mechanics (config copy + launcher copy, rt never written) are unchanged.
4. **r17 "head" naming.** r17 retrained the DRAFTER DENSE (rt-r17 dense.bin differs; experts/draft_vocab
   identical) — there is no head file in any rt dir because the draft head is bound from the backbone at load
   (mtp.cpp:726-777). "Head-only retrain" (T11.R3) means the 31-tensor MTP block with experts frozen; the
   lm_head rows stay bound and frozen. This spec's arms implement exactly that.
5. **beta=0.8 as "OUR adaptation" vs F14's "arm it, don't assume it".** Both honored: beta 0.8 is the primary
   arm a1 AND the published 0.6 runs as a2 (Section 6).

## 12. Open questions (owned, with their cheap resolvers)

- Dump throughput in the prompt path: the 30-min probe (Section 3.3) decides per-token vs batched extension.
- Whether harvested real positions suffice or explicit offset-augmented windows are needed for 131k-class
  coverage: the harvest's own pos histogram answers it (STEP 0 #5 says position coverage is where the deficit
  lives — check the histogram BEFORE training, not after).
- cp315 torch wheel availability: P0's decision rule (Section 2.2).
- Whether LK's coupled variant `sum p*q` (F15) beats `sum min(p,q)` under our exact-match verify: a4 arms both
  as config flags; one extra G0.2 stair read prices it.
- a5-vat's survival-head transfer to a single-layer recursive drafter: open in T17C §7; the G0.2 flatness of the
  dp curve is the falsifier.

## 13. Implications

- The trainer is the last NEW code on the retrain path: dump flag, export chain, quantizers, head bind, fold
  conventions, parity tap, and the validated shipping config all exist in-tree and in-journal (T14B §10).
- Everything the trainer emits is additive and reversible: rt-r18 beside rt-r17, standing config untouched,
  rollback = boot the standing launcher (Section 9). The gates spend starts only after the audit passes.
- Sequencing constraints from the campaign ledger respected: the T17F pack re-quant (if GO) lands BEFORE the
  harvest so the drafter trains on the serving distribution (T17F §6; TODO :1224); the harvest is THIS lane's
  first compute act, and the probe prices it before anything is built.

## 14. Sources

Code (read this session, engine = /home/svg/LLM/strata-custom-2):
- src/core/mtp.cpp: tap env :77/:288-293/:1229-1240; q4 projections :202-243 (make_q4_dense :210, call :337);
  q4 head :245-269 (make_q4_head; dhead_type_ :261; call from bind :773); load/dense.txt :313-325; required
  q8_0 tensors :363-366; experts.bin :344-359; full_head_env :539-546; bind :690-781 (gather :766, hint :655);
  hnorm :941; router top-10 :1141-1148; head mmvq :1223-1262.
- include/strata/core/mtp.hpp:69 (set_q4), :72/:224 (draft vocab path), :219-228 (q4 state).
- src/program/generate.cpp: --mtp-q4 parse :2231-2234, set_q4 :4712; --dump-final-r :638/:2158/:13768-13769;
  record write :13969-13973 (int64 posrec[2] + hc*n_embd f32); final_R_all handoff :8046-8051/:10385/:11943.
- include/strata/kernels/cpu/expert_layout.hpp:3-37 (BLOB layout); tools/mtp_rt.py (header: blob relayout,
  q8_0/bf16/f32 kinds, +1 fold :85; BLOB = 3*(2560*640*18//64)); tools/mtp_pack.py (header: q2_0 grid,
  supports_mtp_export=False note); tools/mtp_fetch.py (pinned-rev range fetch, #327); tools/mtp_phaseb_adapter.py
  (eh_proj split semantics); tools/draft_vocab.py (subset semantics); setup.py:5014-5015/:5282-5301
  (DRAFT_VOCABS, refresh_draft_vocab keep-rule); custom-iq3_s-8081.json (--mtp mtp/rt, --mtp-q4 all,
  --mtp-hnorm stream, draft_vocab en); launch-custom-iq3s-8081.sh (launcher template).
- Runtime dirs: /home/svg/LLM/Custom-data/mtp/rt (dense.txt formats, 116,099,072 B dense; 40,525-id
  draft_vocab.bin = 162,100 B); /home/svg/LLM/Custom-data/mtp-iq3s/ (rt-r17 + mtp-r17-q2_0.gguf + r17-fetch;
  sha256 compare: dense DIFF, experts/draft_vocab SAME). Disk: /home 705 GB free (df, this session).
- Repo .venv probe: python 3.15.0; numpy 2.5.3, gguf 0.19.0, huggingface_hub 2.2.0, datasets 5.1.0 present;
  torch/transformers ABSENT (import-checked).

Banked research and journal (all claims about measurements, laws, and external recipes):
- SVG-CUSTOM-TODO.md: T11.R3 :793-806; T16 W2 :1066-1071; T17 QUEUE :1096-1236; STEP 0 :1238-1269.
- campaign/research/T14A-fastmtp-method-recipe.md (F1-F16, arms 0-4, risks R1-R5, §5 contradictions).
- campaign/research/T14B-fastmtp-training-infra.md (§2 inventory, §3 pipeline, §4 budgets, §5 verdicts,
  F-77..F-118 laws, §12 methods).
- campaign/research/T14C-fastmtp-acceptance-math-gates.md (gate ladder :37-41/:176-230; c_row and 0.72 rule).
- campaign/research/T17C-spec-algo-leaps.md (§5.1 plateau-breaker stack + priced composite; sources 17-25).
- campaign/research/T17F-training-quant-context.md (§3.1 Draft-OPD replay; §6 sequencing law).
- arXiv IDs (banked, verified in the dossiers): FastMTP 2509.18362; VAT 2608.30135; LK losses 2602.23881;
  AdaSPEC 2510.19779; KD-temp 2410.10141; Draft-OPD 2605.29343; LongSpec 2502.17421; EAGLE-3 2503.01840;
  Red Hat Speculators 0.6.0 article (developers.redhat.com, 2026-09-08, Qwen3.5-9B/443 s example).

## 15. Methods note

Read-only session: no builds, boots, training, server calls, or git operations; only this spec file written
(plus its parent directory). Two dossier-vs-code discrepancies were caught and resolved in favor of the code
(record int64 header, Section 11.2) and the on-disk layout (rt-r18 location, Section 11.3) — the reason every
format claim above carries a file:line. Unproductive: no search needed for the sibling model ON DISK — the
box carries no 9B sibling (models/ listing), so G0's sibling is defined by the RH article's example class and
fetched per Section 10.1; the T14B dossier's earlier "no sibling on box -> use the round-trip test" remains the
fallback, not the primary, per this task's explicit G0 framing.
