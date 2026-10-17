# T14B — FastMTP retrain: the training-INFRASTRUCTURE spec (read-only audit)

Date: 2026-10-16.
Scope: the cheapest viable pipeline to retrain the MTP draft layer (31 tensors, 5.214 GB BF16, rt/ at
`/home/svg/LLM/Custom-data/mtp/rt`) against the FROZEN 125B target (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, arch
`qwen4exp`), on this rig alone (RTX 5090 32 GB sm_120, Ryzen 9950X3D, 184 GB RAM, Fedora). READ-ONLY audit:
nothing was built, booted, written or re-quantized; `/mnt/DATA` only listed.
Method: line-level read of the engine's drafter/load/dump paths (`src/core/mtp.cpp`, `src/program/generate.cpp`,
`src/core/weights.cpp`), the MTP tooling (`tools/mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py`, `mtp_bridge.py`,
`mtp_phaseb_adapter.py`, `draft_vocab.py`), the prior training-campaign journal (F-77..F-118 in `SVG-CUSTOM.md`,
the Windows-era R17 arc), T11.R3's GO dossier (`SVG-CUSTOM-TODO.md:793`), GGUF metadata of the target (parsed
read-only with the vendored gguf-py), plus web verification of the external recipe (FastMTP arXiv 2509.18362,
vLLM/Red Hat Speculators 0.6.0, transformers' MTP gap, torch sm_120).

---

## 1. Summary

**GO.** Every component the pipeline needs already exists in-tree or is measured; the pipeline is
harvest (engine dump) → train (dense-only, experts frozen) → export (pack→rt2) → A/B (parallel rt dir, one
config string). The two genuinely NEW pieces of code are (a) a corpus-scale hidden-state dump and (b) a torch
trainer for the engine's exact 31-tensor block — and both have in-repo precedents with named bugs already fixed
in the journal (F-80/F-81 bug list, F-104 mixed-KV lesson, F-105 position-locality lesson, F-109/F-110 fold and
name-map laws, F-112..F-114 the validated shipping config). The Windows-era attempt PROVED the chain (stock
round-trip 85.4 % acceptance) and proved distillation transfers (0→57 %) but plateaued BELOW stock; the named
ceilings (conditioning mismatch, chain quant on fresh weights, position spread) are all addressed in the spec
below. T11.R3's verdict stands; this dossier is its infra leg.

**The single most important configuration decision (from F-114, hard-won): train the DENSE ~90.5 M params
only, freeze the 512 routed experts, and ship trained dense + STOCK `experts.bin` + STOCK `draft_vocab.bin`.**
That is the validated configuration; it also drops trainer VRAM to ~11–16 GB and deletes the expert-requant
risk class entirely (the F-113 guilty component).

---

## 2. What exists today (verified inventory)

### 2.1 The rig and the files

| Item | Value | Source |
|---|---|---|
| GPU | RTX 5090, 32 607 MiB, driver 615.71.09 | `nvidia-smi` |
| Host RAM | 184 GB total, 116 GB available | `free -g` |
| Disk | `/home` 952 GB, **735 GB free**; `/mnt/DATA` 3.7 T (frozen, untouched) | `df -h` |
| Target | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` 2 shards, 83.6 GB total. **The BF16 checkpoint (360 GB, 131 shards) is NOT local** — only the 31 extracted MTP tensors are (`mtp/tensors/*.bin` + manifest, sha256-verified) | `ls`, `mtp-manifest.json`, `tools/mtp_fetch.py` (source = HF `Qwen/Qwen3.8-Flash-Next`, pinned rev `de4b8e4d`) |
| Drafter rt/ | `dense.bin` 116 MB (q8_0/bf16/f32) + `dense.txt` + `experts.bin` 707 MB (q2_0) + `draft_vocab.bin` 162 100 B = **40 525 int32 token ids** (sub-vocab head, Q4_0-cast under `--mtp-q4`) | `mtp/rt/`, `mtp.cpp:725-769`, T13A table |
| Production boot | `--mtp /home/svg/LLM/Custom-data/mtp/rt`, arms `--mtp-q4 all --mtp-hnorm stream --spec 6 --spec-min-p 0.5`, draft_vocab "en" | `custom-iq3_s-8081.json` |
| Target arch (GGUF) | `qwen4exp`: 48 blocks, full_attention_interval 4 → **36 GDN + 12 QSA**; hidden 2560; 512 experts top-10, FF 640; hc count 4, low-rank 320; indexer 4 heads/key 128/top-2048; ctx 262 144; full vocab 248 320 | GGUF metadata (parsed), matches HF card tag `qwen4_exp` |
| Measured prefill | **6291–6972 tok/s** (32K–128K ctx; 3754 t/s at 4K) | `campaign/results/*.txt` |
| Torch | **NOT in the Linux venv** (numpy 2.5.3, gguf 0.19.0 only). The F-78 "torch 2.11.0+cu130 verified" was the Windows box. CUDA 13.4 libs at `/usr/local/cuda-13.4`; driver supports cu130 wheels; sm_120 supported since torch 2.7-cu128 stable (Apr 2025) | venv probe, `custom-iq3_s-8081.json` lib_dirs |

### 2.2 Engine teacher/tap infrastructure (all env-gated, production-inert)

- **`--dump-final-r PATH`** (generate.cpp:13482, 13683-13688): per position, writes `{int32 pos, int32 tok}` +
  final R (hc×n_embd = **4×2560 f32**, 41 KB/pos). This is EXACTLY the drafter's input state + token — the
  teacher record. Draft consumes it via `ver.final_R_all()` → `cudaMemcpyAsync` → `draft_launch`
  (generate.cpp:10320-10332).
- **`--dump-layers PATH`** (needs `--no-capture`): R after EVERY layer + input, per position (the C1 ladder) —
  overkill for MTP (head reads only the last R) but useful for debugging.
- **`--dump-logits PATH`**: one line of raw logits per position — only needed if KL-vs-target distillation is
  added later (the spec below needs CE only).
- **`STRATA_MTP_TAP=<prefix>`** (mtp.cpp:68-93, 286-293, 1229-1240): dumps the DRAFTER's R_/sample_/head_logits_
  for the first window, pinned buffers, capture-legal (F-87/F-88 lessons are already IN the code). Purpose: G0
  value-parity of the torch module against the engine BEFORE training. Deployed and inert without the env var.
- Serving-geometry facts the trainer must mirror: `--mtp-hnorm stream` (per-stream RMS, mtp.cpp:941) is the
  PRODUCTION arm (config), fc_embedding+fc_hidden are a split-and-add pair (F-81 fix 2), shared-expert gate is
  ONE scalar sigmoid per token (F-81 fix 4), norms are Gemma-style (rt build folds `w → 1+w`,
  mtp_rt.py:85 — the F-109 double-fold law), drafter MoE = 512 experts top-10 (mtp.cpp:1141-1148),
  head = 40 525-row subset of the main 248 320×2560 Q6_K head, bound at load (mtp.cpp:690-769) — **the head is
  never in rt/ and retraining never touches it**.

### 2.3 The export chain (in-tree, validated end-to-end on Windows)

`tensors/<name>.bin` BF16 + `mtp-manifest.json` (mtp_fetch.py layout) → **`mtp_pack.py --src <dir> --experts
q2_0 --out mtp.gguf`** (dense BF16/F32, experts q2_0/q4_0/q8_0, streamed per-expert, per-tensor error report;
the *norm-only tensors go F32 raw — the `+1` fold happens LATER, at rt build) → **`mtp_rt.py --gguf <mtp.gguf>
--out <rt2>`** (experts.bin blob relayout + dense.bin q8_0/bf16 + dense.txt). llama.cpp's own converter is a
dead end (`qwen4exp ... supports_mtp_export = False`, mtp_pack.py header). A trained head can skip the
Windows-era graft/adapter/splice stages ENTIRELY by emitting the mtp_fetch.py layout directly — those stages
existed only to translate the external sidecar's naming.

### 2.4 The prior campaign's hard-won laws (F-77..F-118, `SVG-CUSTOM.md`) — the bug list is the spec

1. **G0 parity before ANY training step** (F-80 law; "without G0, v1 0.167 repeats"): torch module vs engine via
   STRATA_MTP_TAP, stage by stage.
2. **Conditioning** (F-104 named root): the trainer built window-local KV while the engine drafts from session
   KV → the trained head read 0/428 despite descending loss. Fixes: short windows as continuations at REAL
   positions; or teacher-student same-forward (the KV error cancels, F-109/Q3).
3. **Position locality** (F-105): the head drafts perfectly ONLY at trained positions (6/6 at origin, 0/377
   elsewhere); pos-spread must cover the REAL serving range — harvesting from own traffic gives this by
   construction (the dump records carry pos).
4. **Shipping config** (F-112/F-113/F-114): rt needs all FOUR files; the expert re-quant path corrupted the
   blob once; final validation = trained dense through-chain + STOCK experts.bin + stock draft_vocab.bin →
   stock-class 85.4 % acceptance. 
5. **Norm fold** (F-109/F-110): norms are stored raw in tensors/ (mtp_pack writes F32 raw), folded `+1` only at
   rt build; the trainer must EXPORT raw (w), FORWARD folded (1+w).
6. **Distillation plateau** (F-115/F-116): pure-KL transfer 0→57.4 % in 2 epochs, flat at 8 epochs; ceiling was
   the chain's double-quant on fresh weights + KL ceiling. Named levers: fewer quant passes, top-k KL /
   smoothing, mixed CE+KL. FastMTP's recursive teacher-forced CE (the R3 recipe) is a DIFFERENT objective —
   trained on GT tokens against target states, not KL toward the stock head — and is the primary here, with
   stock-head KL as an optional additive term.

### 2.5 External recipe (verified)

- **FastMTP** (arXiv 2509.18362, Cai et al., Sep 2025): single shared MTP head trained RECURSIVELY
  (teacher-forced: step k fuses the current hidden with emb(tok[t+k+1]), CE vs tok[t+k+2], output fed back),
  fixes the train/serve recursion mismatch; 2.03× e2e.
- **Speculators 0.6.0** (vllm-project/speculators; Red Hat AI article Sep 2026) is the production form of
  exactly this: MTPDraftModel = one layer fusing verifier last hidden + token embedding, shares embed_tokens +
  lm_head; per-step loss α_k = β^(k-1)/Σβ^(j-1), β=0.6, 3 steps → [0.51, 0.31, 0.18]; data generated by the
  target itself (`prepare_data.py`, e.g. 5 000 samples × 8 192 ctx); hidden states extracted ONLINE from a live
  server; Qwen3.5-9B head trained "in about 443 seconds on 2×H200"; measured acceptance 0.897/0.719/0.476 →
  0.912/0.776/0.616. **Supported verifiers: Qwen3-Next and Qwen3.5 incl. MoE — NOT our qwen4_exp/Flash-Next**
  (no hyper-connection/Gated-Residual block, no indexer-MTP wiring). So Speculators is the RECIPE and the loss/
  schedule reference, not a drop-in trainer.
- **transformers**: Qwen3-Next arch exists but "MTP is not generally available in HF Transformers" (model cards;
  mtp_weights unused, issue #40968). Flash-Next (tag `qwen4_exp`, 131 shards ≈ 360 GB, card says "125B/6B
  active, 51B n-gram, 4B MTP") postdates it; no public support. HF path dead.
- **torch on 5090**: sm_120 native since 2.7.0 cu128 stable; current 2.9–2.11 ships cu128/cu130 wheels — no
  source builds needed.

---

## 3. The pipeline (numbered)

**P0 — Trainer env (0.5 h).** New venv (do NOT touch the repo's `.venv`):
`python -m venv ~/venvs/mtp-train && pip install torch --index-url https://download.pytorch.org/whl/cu130
numpy safetensors` (+ gguf-py from `third_party/llama.cpp/gguf-py` for head dequant). Verify
`torch.cuda.get_device_capability() == (12, 0)`.

**P1 — Teacher-data harvest (engine dump; 0.5–1 d eng incl. probe, then 0.7–22 h compute).**
- Corpus: our own serving-traffic corpus (the mimic lane's prompts; RH used 5 000 samples × 8 192; R3 sized
  8–16 M positions ≈ 160–320 GB bf16 — fits 735 GB free). Distribution, temperature and POSITION spread match
  serving by construction (fixes F-105 for free).
- Boot the engine OFFLINE (lane down or second config), replay the corpus with `--dump-final-r records.f32`
  (+ `--no-capture` NOT required for final_r). Record = `{pos i32, tok i32, R 4×2560 f32}`.
- **Probe first** (30 min): dump 100 k tokens and measure the rate. The hook lives in the per-token prompt loop;
  if it runs at ~200 tok/s instead of the measured 6.3–7 k tok/s prefill rate, extend the BATCHED prefill path
  (~50 lines: after each prefill chunk, copy the chunk's final-R rows — the plumbing they read through,
  `ver.final_R_all()` and the layer_stage host buffer `layer_dump` already demonstrates, generate.cpp:14146,
  14320-14330). Numbers below budget both arms.
- Alignment for the trainer: record i holds (R_i, tok_i); the drafter's serving input at position i is
  (R_i, emb(tok_i)) and its label is tok_{i+1} — pure offset bookkeeping over the stream; keep conversations
  contiguous, drop cross-seam pairs (the chunk-seam law, F-77).
- Optional (only if stock-head KL is added later): `--dump-logits` on the same pass; +248 320 f32/pos ≈ 1 MB/pos
  → do NOT bank it by default.

**P2 — Torch module + G0 parity gate (1–1.5 d; THE gate).**
- Write `mtp_head_qwen4exp.py`: embedding gather (frozen, from the GGUF/pack or a BF16 range-fetch of
  token_embd, 1.27 GB via mtp_fetch.py machinery) → pre_fc_norm_embedding → fc_embedding; hnorm STREAM mode
  (per-stream RMS × its slice of pre_fc_norm_hidden, mtp.cpp:941) → fc_hidden; add; the hc-mixer read
  (hc_norm + down/up 10240↔320, the fused_gr_read semantics); the QSA layer (q 12288 = 48×256 with per-head
  [q|gate] interleave split, q_norm; k/v 2×256; o 24×256; indexer MQA 4/1 × 128, top-2048 block budget —
  teacher-forced windows are short, the indexer degenerates gracefully); MoE top-10 + shared expert + scalar
  sigmoid gate; the output hc-mixer; head over the 40 525-row subset (rows dequantized from the main GGUF's
  Q6_K `output.weight`, frozen).
- The forward conventions are all in the journal (F-80/F-81 fixes 1–4, F-95 exact module, the F-106..F-108
  structural exoneration list: output path, h semantics, vocab, fusion alignment). Use SDPA attention; no
  custom kernels needed (NO GDN in the trainer — the drafter's self_attn is QSA; the target's GDN layers run
  only inside the engine during harvest).
- **G0**: boot the lane binary with `STRATA_MTP_TAP=/tmp/tap`, feed one real window, diff torch-vs-engine R →
  sample → logits stage by stage (cos > 0.9999 per stage, argmax match on names). The tap dumps exactly these
  three tensors. Without G0: NO training (the F-80 law).

**P3 — Trainer (1 d on top of P2).**
- Loop = Speculators' FastMTP form: teacher-forced recursion to depth K=6 (serve depth), loss
  α_k = β^(k-1)/Σβ^(j-1) with β=0.8 (R3's adaptation for our pos-2..6 deficit), CE over the 40 525-row subset
  vs GT tokens; windows of W=256–1024 positions cut from the harvest stream at their REAL pos ids; batch 2–4
  windows (2×256×… was the v5 design point for this card, F-78).
- Trainable: the ~90.5 M dense params ONLY (everything except the two `mlp.experts.*` tensors). Frozen bf16:
  experts (5.04 GB), embedding (CPU-gathered), head rows. AdamW lr 1e-4, 2–3 epochs, cosine, bf16 autocast,
  grad-checkpoint the expert MLP only if activation-tight. Optional additive term (only if the gate stalls):
  KL toward the STOCK head's logits under the SAME window forward (the F-109/Q3 cancellation insight) — ~15
  lines, teacher = the frozen stock weights we already have.
- Optional last pass if the serving gate shows quant loss: ~200 steps of fake-quant (Q4_0 projections /
  q8_0 dense)-aware fine-tune (DWQ-style, the ecosystem-validated lever, F-117a).

**P4 — Export (0.5 h, one command each).**
Trainer writes `<out>/tensors/<name>.bin` (BF16, little-endian u16 view) + `<out>/mtp-manifest.json` (name,
dtype "BF16", shape, file, sha256 — sha256 of the TRAINED tensors; pack only uses them for its source string)
with norms written RAW (w, not 1+w — mtp_rt.py folds at build; the F-109 law) →
`python tools/mtp_pack.py --src <out> --experts q2_0 --out mtp-trained-q2_0.gguf` →
`python tools/mtp_rt.py --gguf mtp-trained-q2_0.gguf --out /home/svg/LLM/Custom-data/mtp/rt2` →
**copy `rt/draft_vocab.bin` and `rt/experts.bin` byte-identical into rt2** (stock experts = the F-114 validated
config; the pack's fresh q2_0 experts are computed but NOT shipped in v1). Value-audit rt2 vs rt: every
UNtrained tensor must be byte/quant-equal to stock (the F-111..F-113 audit, now one script).

**P5 — Rollout/rollback (0 h eng — pure config, no engine code).**
No `--mtp-rt2` flag is needed: `--mtp` takes a DIRECTORY, and the parallel-directory pattern already exists
(`mtp-iq3s/rt-r17` sits beside production `mtp/rt`). Production `rt/` is never written. A/B via a COPY of the
config (`custom-iq3s-rt2-8081.json` with only `--mtp …/rt2` changed) + a copy of the launcher (Launch Law:
never setup.py, start via launchers). Rollback = boot the standing config. Verify with /health + the boot log's
drafter line (rt2's MiB printed at bind, mtp.cpp:528).

**P6 — Gates (the campaign law, ~1–2 h).**
3× `campaign/probe-lane.py` before/after (same hour); **M3b acceptance** (standing 66.6 % → R3 target ~75 %;
per-depth histograms: the gate is pos-2/3 +≥8 pp each, pos-0 ≤1 pp regression, e2e ≥ +8 %); 2× MMLU-Pro gate
`campaign/mmlu-pro-gate.py 0.933 <label>` (drafter weights are output-affecting when acceptance changes the
resample path — treat as token-stream-affecting); keep `--mtp-q4 all --mtp-hnorm stream` IDENTICAL to
production in both arms (train-serve arm parity). Kill-rule: depth extension only while marginal acceptance
> 6–9 % (R3).

---

## 4. Budgets

### VRAM (trainer, P3)

| Item | Bytes |
|---|---:|
| Frozen routed experts (bf16 fwd-only) | 5.04 GB |
| Frozen dense fwd copy (bf16) | 0.18 GB |
| Frozen draft-head rows 40 525×2560 bf16 | 0.21 GB |
| Trainable ~90.5 M dense: bf16 param + grad + AdamW fp32 m,v + fp32 master | ≈ 1.45 GB |
| Activations (2×512-pos windows, checkpointed experts) | 2–6 GB |
| CUDA context + workspace | 1.5–2 GB |
| **Total** | **≈ 11–16 GB of 32 GB** (matches R3 "8–14 GB") |

Variants: full-expert FT (2.6 B params) needs +10 GB with 8-bit Adam and no fp32 master → ~21–26 GB, plus
routing-scatter autograd nobody has written here → **not v1** (F-114 says stock experts IS the config). LoRA:
strictly worse than dense-only here (the trainable surface is already only 90.5 M; LoRA adds merge complexity
into a runtime that has no MTP-LoRA path). AdamW fp32 states on ALL 2.6 B params = ~36 GB → does not fit; the
task's "21 GB tight but feasible" assumed bf16 optim states — unvalidated, skip.

### Disk (735 GB free)

| Item | Size |
|---|---:|
| Harvest f32, 16 M pos × 41 KB | 640 GB (fits, tight) — **write bf16 (20 KB/pos, 320 GB) or cap f32 at ~12 M pos** |
| Trainer env (torch cu130) | ~4 GB |
| tensors/ export copy + packed GGUF + rt2 | ~5.2 + 0.9 + 0.83 GB |
| **Peak** | **≈ 330–650 GB — fits; delete harvest after P4** |

### Wall clock

| Stage | Estimate | Basis |
|---|---|---|
| P0 env | 0.5 h | wheel install |
| P1 probe | 0.5 h | 100 k tokens |
| P1 harvest 16 M pos | **45 min** (batched-dump ext) / **22 h** (per-token fallback) | 6.3–7 k tok/s measured vs ~200 tok/s decode-class |
| P2+P3 eng | 2–2.5 d | module + trainer + gates; journal halves it if the buun v5 trainer is retrievable (it is NOT on this box — searched) |
| P3 train (3 ep × 16 M tok ≈ 48 M tok, ~9–10 GFLOP/tok) | **2–4 h** | 40–80 TFLOPS effective on 5090 with MoE gather + 40 k-row CE; RH reference: 443 s on 2×H200 for 9 B-scale data |
| P4 export + audit | 0.5 h | two commands + diff script |
| P6 gates | 1–2 h | campaign law |
| **End-to-end** | **2–4 d eng + one overnight-to-weekend compute window** | matches T11.R3 "1–2 days eng + a weekend" |

---

## 5. Go/no-go per sub-path

| Sub-path | Verdict | Why |
|---|---|---|
| (1a) HF transformers as teacher | **NO-GO** | MTP "not generally available in transformers" (Qwen3-Next cards; issue #40968); Flash-Next `qwen4_exp` (hyper-connections, indexer) unsupported; BF16 weights not local (360 GB) |
| (1b) Engine as teacher dumper | **GO** | `--dump-final-r` exists and writes exactly (pos, tok, R=drafter input); 6.5 k tok/s prefill measured; quantized-backbone hiddens = distribution-matched (R3 bonus); needs a throughput probe + maybe a ~50-line batched-path extension |
| (1c) GT-tokens only (no teacher states) | **NO-GO** (as primary) | the drafter's INPUT at inference is the target's last R — without teacher states the input path cannot be conditioned (EAGLE-class input); the MTP literature trains the head WITH hidden states; recursion depth k≥2 still consumes R_0 |
| Trainer: Speculators 0.6.0 as-is | **NO-GO as-is, GO as recipe** | supports Qwen3-Next/Qwen3.5, not qwen4_exp/Flash-Next (no Gated-Residual block, no indexer-MTP); its loss/schedule/export shape is the reference |
| Trainer: custom torch, dense-only + frozen stock experts | **GO (primary)** | 11–16 GB VRAM; every forward convention documented in-journal; F-114-validated shipping config |
| Trainer: full 2.6 B-param FT | **NO-GO v1** | ~21–26 GB with 8-bit Adam + custom routing autograd; unvalidated; contradicts F-114 |
| Trainer: LoRA on the MTP block | **NO-GO** | dense surface is already 90.5 M params — LoRA saves nothing that matters and adds a merge stage the rt chain lacks |
| Export: trainer → tensors/ → mtp_pack → mtp_rt → rt2 | **GO** | in-tree, validated (85.4 % round-trip class after F-109..F-114 fixes); skips graft/adapter/splice entirely |
| Export: llama.cpp GGUF conversion of the MTP block | **NO-GO** | `qwen4exp supports_mtp_export = False` (mtp_pack.py header) |
| Rollback: parallel rt2 dir + config copy | **GO** | `--mtp` is a directory flag; the pattern exists (`rt-r17`); zero engine code |
| Dry-run on a "9B sibling" first (R3 gate) | **CONDITIONAL** | no sibling model is on this box (searched); de-risk instead with the free round-trip test: run the STOCK tensors through the NEW export chain → gate must reproduce stock-class acceptance BEFORE training (this is the F-114 test, cheap and decisive) |

---

## 6. The 3 biggest infra risks

1. **Train/serve geometry mismatch (the F-104/F-105 class — the thing that produced 0/428 last time).**
   Wrong hnorm mode (config runs `stream`), wrong fc split-and-add, window-local vs session conditioning,
   absolute-position handling, the Gemma fold. Mitigation: G0 tap parity as a HARD pre-training gate
   (STRATA_MTP_TAP is deployed); harvest from own traffic so positions/distribution match by construction;
   train windows as continuations at real pos ids; value-audit rt2 tensor-by-tensor vs rt before any boot.
2. **Chain quantization on fresh weights (the F-113/F-116 ceiling — the 57 %-vs-85 % gap).** Stock weights
   survive bf16→q2_0/q8_0/q4 direct-cast; fresh weights were quant-noise-sensitive. Mitigation: v1 ships
   trained DENSE only + STOCK experts.bin + STOCK draft_vocab.bin (validated config); gates run under the
   production arms (`--mtp-q4 all --mtp-hnorm stream`); if the gate shows quant loss, add the fake-quant
   (DWQ-style) fine-tune pass and/or fewer quant passes rather than shipping blind.
3. **Teacher-dump throughput and disk at corpus scale (unmeasured).** The dump hook lives in the per-token
   prompt path; if it runs decode-class (~200 tok/s), 16 M positions = ~22 h and f32 records = 640 GB. 
   Mitigation: 30-minute probe BEFORE building anything; the batched-path extension is small and the fast
   numbers (45 min) make even the bad arm survivable; write bf16 or cap the corpus; harvest is resumable
   (append-only records).

---

## 7. Findings by confidence

**High (primary source: code/file read, or measured, this session)**
- Drafter input = target final residual R (4×2560); `--dump-final-r` writes (pos, tok, R); handoff via
  `final_R_all()` (generate.cpp:10320-10332, 13683-13688; mtp.cpp bind signature).
- rt/ format and the full export chain mtp_pack.py→mtp_rt.py; norms fold +1 at rt build; 4 required files
  (tools/*.py; F-109/F-112/F-114).
- Draft head = 40 525-row subset bound at load from the main Q6_K head; never stored in rt/ (mtp.cpp:690-769;
  T13A table: 58.3 MiB/step).
- Arch `qwen4exp`: 48 blocks, 36 GDN + 12 QSA (full_attention_interval 4), 512 experts top-10, hc 4×rank-320,
  indexer top-2048, vocab 248 320 (GGUF metadata parsed; T13A).
- Rig: 184 GB RAM, 735 GB free, prefill 6291–6972 tok/s measured (free/df/nvidia-smi; campaign/results).
- No BF16 checkpoint local; only 31 extracted tensors + IQ3_S quant; source = HF pinned revision
  (models/ listing; mtp-manifest.json; mtp_fetch.py).
- Production arms `--mtp-q4 all --mtp-hnorm stream --spec 6`; `--mtp` points at the rt DIRECTORY
  (custom-iq3_s-8081.json).
- STRATA_MTP_TAP dumps drafter R/sample/logits, pinned, capture-legal, inert without env (mtp.cpp:68-93,
  286-293, 1229-1240).
- Dense-trainable ≈ 90.5 M params (tensor math; cross-check F-81 "88.9 M dense-trainable").
- The Windows-era chain was validated end-to-end at stock-class 85.4 % and distillation transferred 0→57 %
  then plateaued (F-114/F-115/F-116).
- torch NOT in the Linux venv (import fails).

**Medium (multiple independent credible sources)**
- FastMTP recipe and Speculators 0.6.0 form (arXiv page + Red Hat article + vLLM blog; the journal's R3 docket
  agrees on β-decay, recursion, data scale).
- transformers' MTP gap for Qwen3-Next (official model cards + GitHub issue #40968).
- sm_120 supported in stable torch since 2.7-cu128 (PyTorch forums/Salad/NVIDIA forum concur).
- Trainer VRAM 11–16 GB is comfortable on 32 GB (R3 estimate + F-78 v5 design point + this budget).

**Low (single source or inference — flagged)**
- "4B MTP" on the HF card vs our 2.6 B count (card rounding or includes fc/n-gram pieces — irrelevant to the
  pipeline; we train what the GGUF ships).
- The batched-path dump extension being ~50 lines (inference from the layer_dump/final_R_all plumbing; not
  attempted — READ-ONLY).
- Harvest 16 M positions sufficing (R3's sizing + RH's 5 k-sample precedent; no first-party measurement for
  THIS 2.6 B-param MoE head — the P6 gates exist to catch under-training).

**Unknown (searched, not found)**
- The buun-sidecar trainer files (`train-fastmtp-v5.py`, `mtp_head_flashnext.py`, `verify-parity.py`,
  `std_ladder.py`, `graft_v2.py`, `head-init-*.pt`, `pairs-r17-*.bin`) — NOT on this Linux box (searched
  /home/svg to depth 5; they live in the Windows-era "buun" repo "carried by patch 0001"). If the user can
  copy them over, P2+P3 drop from ~2.5 d to ~0.5–1 d.
- Public HF support for `qwen4_exp` MTP export (none found; llama.cpp explicitly refuses).

---

## 8. Contradictions and resolutions

- **"21 GB full-FT is feasible" (task premise) vs F-114's shipping config.** Resolved: fp32 AdamW on all
  2.6 B params = ~36 GB (does not fit); bf16-optim variants are unvalidated here and need routing autograd;
  the campaign's own validated answer is dense-only + stock experts. Full-FT stays a no-go for v1 on evidence,
  not on vibes.
- **Windows-era "27.8 % stock round-trip" vs the rt chain's real quality.** Resolved by F-109..F-114: the 27.8 %
  was polluted by the double Gemma fold, a 13/28 name map, a missing draft_vocab.bin and a corrupt expert
  requant; the corrected round-trip reads 85.4 % (stock class). The chain is NOT the bottleneck once the five
  fixes are in — and all five are documented in-journal.
- **T11.R3 "5 k conversations, 2–6 h harvest" vs the measured prefill.** Resolved: 6.5 k tok/s makes the
  token-generation trivially cheap; the open variable is the dump path's own throughput (probe first). R3's
  2–6 h remains a safe upper bound.
- **Distillation (Windows Q3) vs FastMTP CE (R3).** Not actually contradictory: Q3's KL was the fallback that
  works under conditioning mismatch (error cancels); R3's CE is the primary objective with harvest-from-engine
  removing the mismatch. Spec keeps CE primary, KL additive-optional.

---

## 9. Open questions

1. `--dump-final-r` throughput in the prompt path (per-token vs batched) — the P1 probe decides; if slow, the
   batched extension lands in generate.cpp's prefill chunk loop.
2. Where the trainer sources the frozen `token_embd` (packs/q2_0/embd.bin is 476 MB — format unverified; else
   BF16 range-fetch from HF via mtp_fetch.py machinery, 1.27 GB) — either works; embd accuracy only matters to
   bf16.
3. Whether the serving-position distribution (up to 131 k ctx in the config) needs long-position windows or
   whether harvested real positions suffice — the dump answers it empirically; F-105's law is "cover what you
   serve".
4. Whether the record corpus should include the r17-era `pairs-r17-*.bin` (Windows artifacts, not local) —
   no: harvest fresh from THIS lane's traffic; old-corpus conditioning is what F-104 convicted.
5. Retrieval of the buun trainer/module from the Windows side (pure speedup, not a dependency).

---

## 10. Implications

- The retrain is NOT blocked by infrastructure: everything hard (dump format, rt layout, quantizers, head bind,
  fold conventions, parity tap, validated shipping config) exists and is journaled. Engineering is the trainer
  module + the harvest probe; compute is an overnight.
- The acceptance lever compounds with the whole T13A/T11 queue (draft-temp 0.85, typical-accept, batch-MTP are
  acceptance-agnostic; a better head lifts every lane). R3's math: RH-scale deltas ≈ 227 t/s class, FastMTP-scale
  ≈ 237, depth re-unlock → 247–255 — the 250 path runs THROUGH this pipeline.
- If the gate fails, the fallback ladder is cheap and pre-named: (a) additive stock-head KL (proven 0→57),
  (b) fake-quant fine-tune (proven direction, F-117a), (c) more epochs/data (RH says "longer training might
  yield a better speculator"). None of them require new infrastructure.
- Production safety is structural: rt/ is never written; rt2 is additive; rollback is a config string.

---

## 11. Sources

In-tree (read this session): `src/core/mtp.cpp` (tap 68-93/286-293/1229-1240; load 314-364; bind 690-769;
hnorm 941; router 1141-1156; sample head 1223-1262); `src/program/generate.cpp` (dump options 489-497/859-862/
2023-2025; final_r 13482/13683-13688; handoff 10320-10332; layer dump 14146/14320-14330; --mtp arg 2174);
`tools/mtp_fetch.py` (HF source, pinned rev, range machinery); `tools/mtp_pack.py` (layout, quantizers,
`supports_mtp_export = False` note); `tools/mtp_rt.py` (rt layout, q8_0/bf16/f32 kinds, +1 fold line 85);
`tools/mtp_bridge.py`, `tools/mtp_phaseb_adapter.py` (sidecar naming, 31-tensor equivalence);
`tools/draft_vocab.py`; `mtp/mtp-inventory.{md,json}`, `mtp/mtp-manifest.json`, `mtp/rt/dense.txt`,
`mtp/mtp-q2_0.gguf.report.json`; `custom-iq3_s-8081.json`; `launch-custom-iq3s-8081.sh`;
`SVG-CUSTOM.md` F-77..F-119 (lines 2440-3240); `SVG-CUSTOM-TODO.md` (T7.B.1 line 319, T11.R3 lines 793-806,
queue lines 874/998); `campaign/research/T13A-decode-critical-path.md` (budget table, 40 525-row head,
214.7 t/s standing); `campaign/research/F-201-mtp-agent.md` (10 findings; RH 8 k-sample/443 s precedent;
FR-Spec vocab retention); `campaign/results/*.txt` (prefill 6291-6972 t/s); AGENTS.md (campaign law);
GGUF header of `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (parsed read-only); `free -g`, `df -h`,
`nvidia-smi`.

Web: [FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction (arXiv 2509.18362)](https://arxiv.org/abs/2509.18362);
[Red Hat Developers: Optimize vLLM speculative decoding with FastMTP heads (Speculators 0.6.0, Sep 2026)](https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads);
[vllm-project/speculators (GitHub)](https://github.com/vllm-project/speculators);
[vLLM blog: Speculators v0.3.0 training support](https://vllm.ai/blog/2025-12-13-speculators-v030);
[Red Hat: Standardized, production-ready speculative decoding](https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding);
[Qwen3-Next-80B-A3B-Instruct model card (MTP not available in transformers)](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct);
[transformers issue #40968 (mtp_weights unused)](https://github.com/huggingface/transformers/issues/40968);
[HF Qwen/Qwen3.8-Flash-Next (qwen4_exp, 131 shards)](https://huggingface.co/Qwen/Qwen3.8-Flash-Next);
[PyTorch forums: sm_120 support](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099);
[NVIDIA forum: RTX 5090 sm_120 wheels](https://forums.developer.nvidia.com/t/rtx-5090-not-working-with-pytorch-and-stable-diffusion-sm-120-unsupported/338015).

## 12. Methods note (what did NOT work, so it is not repeated)

- A hand-rolled GGUF header parser died mid-KV (MemoryError on a string field) — use the vendored gguf-py with
  the repo venv; system python3 lacks numpy.
- `grep torch` in tools/ found only 4 files, none of them the trainer — the trainer is NOT in this repo; the
  trail goes through the F-77..F-118 journal to the Windows-era "buun" sidecar repo. Searches for
  `train-fastmtp*`, `mtp_head_flashnext*`, `verify-parity*`, `head-init-flashnext*`, `pairs-r17*`, `*buun*`
  under /home/svg (depth 5): zero hits.
- Web searches for "Qwen3.8-Flash-Next transformers support" return the HF card but no library support —
  treat the card's own numbers (125B/6B/51B/4B) as descriptive only; the GGUF and the in-tree inventory are
  authoritative for geometry.
- The HF card fetch flagged its content as possibly synthetic; every geometry claim used in this dossier was
  cross-checked against the LOCAL GGUF metadata and the local tensor inventory before use.
