# T14B-exec-C — MTP drafter retrain: the TRAINER (P3), EXPORT/ROLLOUT (P4–P5), GATES (P6), and the optional H200 sweep — execution-ready spec

Date: 2026-10-18. Scope: P3–P6 of the T14B pipeline as re-stated in T14B-fastmtp-training-infra.md (the infra
audit, GO verdict) + T14A (method recipe) + T14C (acceptance arithmetic and gate ladder). This file is leg C of
the relaunched trio (A = harvest, B = module/G0 parity — both killed at zero output by the 10-17 ISP crash,
F-235); where A/B interfaces are needed they are PINNED HERE so the three legs interlock without a meeting.
READ-ONLY research: nothing was built, booted, or written outside this file; `/mnt/DATA` untouched; live server
untouched. Every path in this file is absolute and was verified to exist this session unless marked `(new)`.

Established inputs this spec builds on (do not re-derive): dense-only training (~90.5 M trainable of 2.64 B;
F-114 validated shipping config), FastMTP recursive teacher-forced CE, K=6, β=0.8 (R3), CE over the 40,525-row
sub-vocab, windows at REAL pos ids, batch 2–4, 2–4 h per 3-epoch×16M-token run on the 5090; export = trainer
writes `tensors/*.bin` BF16 + `mtp-manifest.json` (norms RAW, F-109) → `mtp_pack.py --experts q2_0` →
`mtp_rt.py` → `rt2/` with stock `experts.bin` + `draft_vocab.bin` byte-identical (F-114); rollout = config copy
pointing `--mtp` at `rt2/`; gates per T14C ladder. NEW since T14B: **F-235** measured deep-position acceptance
53.1 % vs shallow 93.6 % and (second entry) attributed the decay MOSTLY to content, demoting LongSpec
position-aug and promoting content-matched data; **F-228** measured the depth stair (offered ≈ 2.20, marginals
85/68/75 % at depths 3/4/5) and killed gate-tuning; **F-229** closed ungated chains conditionally on a deeper
chain; **T17C/T17F** banked the retrain-upgrade stack (VAT / LK / AdaSPEC / Draft-OPD / LongSpec) with the
r17 plateau fully attributed (fresh-init + pure-KL + double-quant, F-115/F-116).

---

## 1. Summary

P3 is a ~600-line torch trainer around exec-B's G0-verified module (`mtp_head_qwen4exp.MTPHead`); its three
non-obvious decisions are (1) the loss assembly carries a **deep-window loss weight `w_deep = 1.0` (2× on
windows with pos ≥ 16,384) as the v1 default, armed to 0 in the sweep** — this is the F-235 decision, resolved
in §2.4; (2) **peak LR 1e-4 (RH form) with 5e-5 (FastMTP form) as the first sweep arm** — justified in §2.3;
(3) labels outside the 40,525-row sub-vocab are **masked out of the CE** (they can never be accepted at serve —
LK's off-vocab-mass-0 argument), with the masked fraction logged as a harvest-quality stat. The offline
acceptance evaluator (§3) is the sweep's scoring function and the trainer's live dashboard: greedy per-chain-
step match on held-out windows, position-bucketed, with the STOCK weights run through the identical evaluator
as the baseline — the pre-boot analogue of T14C's G0.2 stair read. The sweep (§4) prunes the 16-cell grid to
6 runs; on the 5090 only S1+S4 run (12–18 h of GPU), the full grid is the H200 session's payload (6 × 20–40
min). Export/rollout (§5–§6) are two commands plus a config copy; the audit (§5.4) has a free zero-training
identity run that byte-validates the entire chain before any trained byte exists. Gates (§7) follow T14C's
ladder with the campaign's 3× probe-lane / 2× MMLU law; one honest flag is raised: **T14C re-priced the
retrain's e2e central to +5.7 % M1 at fixed shape, so the R3-era "+8 % e2e" PASS line is recorded as the
full-win target, with the adoption bar at the standing +4 % + M3b +8 pp — the operator owns that call (OD4).**

---

## 2. THE TRAINER (P3) — code-level

New files (all `(new)`): `tools/mtp_train.py` (trainer + evaluator CLI), `tools/mtp_eval.py` (standalone
evaluator), `tools/mtp_audit_rt.py` (the §5.4 audit). Torch venv per T14B P0: `~/venvs/mtp-train`
(`torch --index-url https://download.pytorch.org/whl/cu130`, numpy, gguf-py from
`third_party/llama.cpp/gguf-py`); verify `torch.cuda.get_device_capability() == (12, 0)`. Repo `.venv` is never
touched.

### 2.1 Data pipeline: windows from the harvest shards

**Input contract (pinned for leg A).** Harvest records live at `/home/svg/LLM/Custom-data/harvest/` as
`records-<shard>.f32` or `.bf16`, headerless little-endian streams of fixed records
`{int32 pos, int32 tok, R float32[4][2560]}` = 40,968 B (f32) or `{int32 pos, int32 tok, R bf16[4][2560]}` =
20,488 B (bf16) — the exact `--dump-final-r` layout (generate.cpp:13683–13688; T14B §2.2). Sidecar
`seams-<shard>.json`: sorted int64 record indices that START a conversation (cross-seam pairs are dropped —
the chunk-seam law, F-77). Sidecar `manifest.json`: record count, dtype, engine build hash, corpus provenance
(prompts files + sha256), masked-label fraction (filled by the trainer at first pass).

**Window indexing (one pass, cached).** Scan records; a window is a maximal run of W=512 consecutive records
(pos strictly +1 within the run, no seam crossing; runs shorter than W+K are dropped). Cache as an int64 array
`(start_record, pos_min, pos_max)` in `harvest/windows.npy`. Windows keep their REAL pos ids; they are fed to
the module as continuations (pos ids absolute, never re-based — the F-104/F-105 conditioning laws).

**Sampling.** Per step draw B=4 windows (batch 2–4 allowed; 4×512 is the v5-class design point, F-78).
Sampling weights: `w_win = 1 + w_deep · 1[pos_max ≥ 16384]`, `w_deep` CLI flag, default 1 (§2.4). A window's
loss weight is the SAME `w_win` (loss weighting, not duplication — no double visits under a cosine schedule).
Hold out ~2 % of conversations (fixed seed 1234) as eval — never trained.

**Anchors and labels.** For window records i=0..W−1 with tokens t_i and teacher states R_i: anchor a=0..W−1−K
runs the K-step recursion (labels t_{a+1}..t_{a+K}; embeddings emb(t_{a+k−1}) from the frozen table). Label
mapping: `draft_vocab.bin` = 40,525 int32 ids into the 248,320 vocab; build `label_id → sub_row` LUT once;
labels not in the LUT are MASKED from the CE at that (anchor, k) cell and counted (`masked_frac` — expected
<1 % since the subset is frequency-chosen; if harvest shows >2 %, STOP and re-derive the subset with
`tools/draft_vocab.py` before training).

### 2.2 Loop structure (K=6 recursive teacher forcing)

```
for step in range(max_steps):
    windows = sampler.draw(4)                       # w_win-weighted, epochless shuffle
    R0      = gather_states(windows)                # B×W×10240 bf16 → GPU
    toks    = gather_tokens(windows)                # B×W int32
    h = module.encode_input(R0, emb[toks])          # pre_fc_norm_* + fc split-and-add (F-81 fix 2)
    loss = 0
    chain_h = h[:, :W-K]                            # anchors
    for k in 1..K:
        logits_k = module.step_forward(chain_h)     # QSA+self-attn(k-length causal KV)+MoE(top-10
                                                    #  +shared, scalar sigmoid gate)+output mixer
        ce_k     = CE(logits_k, labels_k, mask)     # over 40,525 sub-rows, fp32
        loss    += alpha[k] * w_win.dot(ce_k) / w_win.sum()
        chain_h  = module.next_state(chain_h)       # output hc-mixer → next-step R (the recursion)
        chain_h  = module.fuse_token(chain_h, emb[toks[:, k:W-K+k]])   # teacher-forced token path
    loss.backward(); clip_grad_norm_(1.0); adamw.step(); sched.step()
```

The module internals are exec-B's deliverable (G0 tap parity, cos > 0.9999 per stage vs `STRATA_MTP_TAP`);
the trainer must NOT re-derive any convention. The hnorm mode is STREAM (production arm, mtp.cpp:941), norms
are forwarded folded (`1+w`) and exported raw (F-109), the head is the frozen 40,525-row Q6_K subset.
SDPA attention only; no custom kernels; grad-checkpoint the expert MLP only if activations tighten (they
should not at 4×512).

### 2.3 Optimizer, schedule, dtype

- **Trainable:** all 29 dense tensors (~90.5 M params — fc 13.1 M, q_proj 31.5 M, o_proj 15.7 M, k/v 2.6 M,
  indexer 1.6 M, shared expert 4.9 M, hc mixers 19.7 M, router 1.3 M, norms ~0.02 M; tensor-math check against
  dense.txt, cross-check F-81's 88.9 M). **Frozen:** the two `mlp.experts.*` tensors (2.55 B params, bf16
  forward-only), the embedding table, the head rows. This is the F-114 shipping config; full-expert FT and
  LoRA stay no-go (T14B §5).
- **Optimizer:** AdamW(β1=0.9, β2=0.95, eps=1e-8), weight decay 0.01 (FastMTP used ms-swift AdamW(0.9, 0.95);
  WD is a non-variable in every published recipe — 0.01, not swept), grad clip 1.0.
- **Schedule:** 3 epochs, cosine to 10 % of peak, warmup 3 % (~700 steps at 16 M×3, batch 4×512 ≈ 2 k tokens/
  step → ~23 k steps).
- **Peak LR — decision: 1e-4 primary (RH), 5e-5 first sweep arm (FastMTP).** Justification for 90.5 M:
  both published anchors are warm-start finetunes of a shipped head on the same head class — FastMTP
  (arXiv 2509.18362) used 5e-5 on a 210.8 M head; Red Hat speculators 0.6.0 (Sep 2026 article) used 1e-4 on a
  ~100–400 M head (Qwen3.5-9B example, 443 s on 2×H200). 90.5 M sits inside RH's bracket and BELOW FastMTP's
  count; the per-param risk of 1e-4 is bounded by (a) warm-start from the shipped weights (never fresh-init —
  the r17 plateau leg, F-115/F-116), (b) clip 1.0, (c) loss-spike abort (§2.5). FastMTP's own τ was still
  rising at its LR, and our trainable set is 1/29th of the block (only dense), so 1e-4 is the RH-demonstrated
  value at our scale, not an extrapolation. Arm, don't assume.
- **Dtype policy:** compute bf16 autocast; trainable params hold fp32 master + fp32 AdamW m,v (≈ 1.45 GB
  total, T14B §4 table); frozen experts/embedding/head bf16; CE in fp32; optimizer applied to fp32 master,
  cast to bf16 for forward. No 8-bit optimizers (unvalidated here; unnecessary at 1.45 GB).
- **Abort/health:** loss NaN or > 3× rolling median for 20 steps → abort, checkpoint rolled back; grad-norm
  and per-α_k loss logged every 50 steps (JSONL at `<out>/train-log.jsonl`).

### 2.4 The deep-position decision (F-235), resolved

F-235 banked two readings: deep-position acceptance 53.1 % vs shallow 93.6 % (depth-t17f-real120k), and the
closeout's decomposition — filler@121 k 68.0 % vs real-corpus@123 k 53.1 %, "decay is MOSTLY CONTENT, not
position; content-matched data PROMOTED, LongSpec position-aug DEMOTED." The two entries disagree on which
factor dominates; they agree on the action. **Decision: the corpus is content-matched by construction
(harvest = own-traffic replay at real pos ids — leg A), so the only remaining training-side lever is mass on
deep windows: v1 default `w_deep = 1.0` (2× loss weight on windows with pos_max ≥ 16,384, the 32 k/131 k
serving range), with `w_deep = 0` as sweep arm S6's falsifier.** Rationale: the M3b gate tests deep CHAIN
marginals and F-228 showed the depth lever routes through deep marginal acceptance (breakeven +5 pp,
T14C §1.4); the weighting costs one line and one sweep cell; it is validated by the evaluator's position
buckets (§3) and killed if any shallow bucket regresses >0.5 pp offline. Position-OFFSET augmentation
(LongSpec) stays demoted per F-235 — do not implement it in v1.

### 2.5 Checkpointing, logging, and the live acceptance proxy

- **Checkpoints:** `<out>/ckpt/step-<n>.pt` at each epoch end + best-by-evaluator + every 30 min
  (trainable fp32 master + AdamW + sampler/RNG state ≈ 1.3 GB each; keep last 2 + best). Resume-safe.
- **Logging:** every 50 steps — total loss, per-α_k CE_k, grad norm, LR, tok/s, `masked_frac` running.
  Every 2,000 steps AND at every epoch end — the **offline acceptance proxy** (§3 evaluator on 2 k held-out
  windows): per-chain-step greedy match m̂_1..m̂_6, position-bucketed, plus the chain-sim
  Ĉ = 1 + Σ_k s_k·Π_{j≤k} m̂_j with F-228's offer shape s = [1, .63, .35, .17, .08, ~0]. This is the number
  the operator watches during the 2–4 h run; it is the only training-visible estimate of what M3b will say.
- **The frozen pieces at init (decided, closing T14B open question 2):** token_embd via one
  `tools/mtp_fetch.py` BF16 range-fetch from HF rev `de4b8e4d` → `/home/svg/LLM/Custom-data/mtp/token_embd.bf16`
  (1.27 GB, sha256 recorded; offline fallback: `packs/q2_0/embd.bin` 476 MB after a format check). Head rows:
  dequantize the 40,525-row Q6_K subset of `output.weight` from the local main GGUF once at init (minutes;
  0.21 GB bf16 on GPU). Cache both under `/home/svg/LLM/Custom-data/mtp/train-cache/` `(new)`.

### 2.6 VRAM/time budget (5090, unchanged from T14B §4)

11–16 GB of 32 GB (experts 5.04 + frozen dense/head 0.39 + trainable states 1.45 + activations 2–6 + context
1.5–2). Throughput ≈ 9–10 GFLOP/tok effective-inclusive (6 recursive forwards + backward); 48 M tokens →
2–4 h on the 5090 at 40–80 effective TFLOPS; 8 M-token arm S4 ≈ 1–1.5 h.

### 2.7 Optional loss arms from T17 (banked, cited — include as FLAGS, not v1 defaults)

| Flag | Arm | Citation / evidence | When to arm |
|---|---|---|---|
| `--loss lk` | LK overlap loss λ·KL+(1−λ)·TV with adaptive λ=exp(−3·sg[α]), or −log α; native to the 40,525-row sub-vocab (off-vocab mass = 0 in α, no KL blowup) | arXiv 2602.23881 (ICML 2026): +3.5–8.2 % τ over KL; +5.6 % on a DeepSeek-V3 MTP finetune; T17C: ~+1–2 % M1 as rider | If offline m̂_4..6 lift < +4 pp after S1; needs the verifier p-list dump (leg A option) |
| `--vat` | Verification-aware training: verify-loop simulation, decay re-anchored at each sample's first rejection (replaces fixed β^(k−1)) | arXiv 2608.30135 (Naver, github.com/naver-ai/vat): +11.4 % AL on EAGLE-3/DFlash, objective-only; T17C: ~+1.5–3 % M1 on top | If offered/window stalls < 3 after S1 passes |
| `--adaspec` | Two-stage selective KD: S1 checkpoint IS the reference; retrain on top-40 % tokens by loss gap; KD temp 0.7 | arXiv 2510.19779 (NeurIPS 2025): up to +15 % acceptance relative | Post-S1 polish if e2e gate is borderline (OD4) |
| `--replay-rejection` | Draft-OPD-style on-policy rejection replay: upweight drafter-rejected positions 2–4× | arXiv 2605.29343: +23 % over EAGLE-3; T17F: +3–8 pp deep marginals → +2.4–6.4 % M1; rides the harvest at ~0.5 d instrumentation | v2; needs leg A to also record (pos, draft, accepted?, target pick) tuples during harvest |
| `--longspec-aug` | Position-offset augmentation | arXiv 2502.17421 (Anchor-Offset Indices, +14.7 % AL) | **DEMOTED by F-235** — do not arm without new evidence |
| `--fake-quant` | ~200 steps at LR 5e-6 with STE fake-quant (q8_0 projections, q2_0 experts) — DWQ-class | T14B P3 / T14A R2 | ONLY if the boot gate shows quant loss (offline m̂ vs M3b gap > 2 pp) |

Warm-start discipline is not a flag: every arm initializes from the SHIPPED rt dense tensors (dequantized) or
from S1 — never fresh-init (F-115/F-116).

---

## 3. THE OFFLINE ACCEPTANCE EVALUATOR (pre-boot proxy)

`tools/mtp_eval.py (new)`. Purpose: rank sweep arms and watch training WITHOUT boots; it is advisory — T14C's
G0.2 boot stair remains the binding gate.

- **Data:** the held-out ~2 % conversations (~300 k positions), fixed seed, identical across every arm and
  the baseline. Served from the same harvest shards, never augmented.
- **Metric:** greedy per-chain-step match rate m̂_k = mean[argmax logits_k == label_k] over valid anchors,
  k = 1..K, computed under the EXACT serving input conventions (stream hnorm, folded norms, sub-vocab head;
  drafter chain fed its own output state + GT token embeddings — teacher-forced, no sampling).
- **Slices:** overall per k; position buckets pos < 4 k / 4 k–16 k / 16 k–64 k / > 64 k (the F-235 axes);
  chain-sim Ĉ with F-228's offer shape; masked-label fraction.
- **Baseline (mandatory):** run the STOCK rt dense tensors (`/home/svg/LLM/Custom-data/mtp/rt/dense.bin`
  dequantized per dense.txt) through the SAME evaluator → m̂_k^stock. All gates compare Δm̂_k = m̂_k −
  m̂_k^stock. The offline gate mirroring G0.2: **Δm̂_3, Δm̂_4 ≥ +8 pp each, Δm̂_1 ≤ +1 pp regression** (chain
  steps 1-indexed; boot-side these are T14C's pos-2/3 = depth-2/3 conditional marginals — greedy-vs-coupled
  offsets mean offline and boot numbers will NOT match in level, only in sign and roughly in delta).
- **Hygiene (LK's own law, arXiv 2602.23881):** greedy match systematically UNDERESTIMATES acceptance at
  T>0; the bias is shared by all arms so RANKING is valid — never quote m̂ as a serving prediction.
- Runtime: ~300 k anchors × 6 steps ≈ minutes on the 5090; also runs CPU-only (slow) if the GPU is busy.

---

## 4. THE SWEEP (optional; grid → 6 runs)

Grid β {0.6, 0.8} × K {4, 6} × LR {5e-5, 1e-4} × data {8 M, 16 M} = 16 cells. Pruned to 6 by fixing
data=16 M and LR=1e-4 except where a cell exists to falsify exactly one variable (one-variable-at-a-time
around S1, the campaign's cheapest-falsifier law):

| Run | β | K | LR | data | w_deep | What it decides |
|---|---|---|---|---|---|---|
| **S1** | 0.8 | 6 | 1e-4 | 16 M | 1 | THE primary (R3 recipe + §2.4 decision) |
| S2 | 0.8 | 6 | 5e-5 | 16 M | 1 | LR fallback (FastMTP form) — arm if S1's m̂_1 regresses |
| S3 | 0.6 | 6 | 1e-4 | 16 M | 1 | β falsifier (published FastMTP β vs R3's 0.8) |
| S4 | 0.8 | 6 | 1e-4 | 8 M | 1 | data-scale falsifier; doubles as the 5090 v1 shakedown (1–1.5 h) |
| S5 | 0.8 | 4 | 1e-4 | 16 M | 1 | K falsifier (is depth-5/6 loss mass buying anything?) |
| S6 | 0.8 | 6 | 1e-4 | 16 M | 0 | w_deep falsifier (the §2.4 decision, armed both ways) |

- **Scoring:** the evaluator (§3). Winner = max (Δm̂_3 + Δm̂_4) subject to Δm̂_1 ≥ −1 pp and no bucket
  regressing > 0.5 pp. Boot-validate only the top 1–2 arms (gates are boots).
- **Per-run time:** 5090 — S4 ≈ 1–1.5 h, 16 M runs ≈ 2.5–3.5 h each (all six locally = 12–18 h; run S1+S4
  only, the rest only if H200 is declined). H200 — 4.6e17 FLOP at ~150–300 effective TFLOPS ≈ **20–40 min
  per 16 M run** (the task's number; my sanity band 25–50 min — this workload is MoE-gather + 40 k-row CE,
  not dense GEMM, so the 5090→H200 ratio is ~2.5–4×, not the 5–6× peak-float ratio).
- **Sequencing:** S1 + S4 on the 5090 Day 2 regardless (S1 is the ship candidate; S4 validates the data
  pipeline cheaply). H200 grid (S2/S3/S5/S6 + re-scored S1) only on operator go (OD3), or if S1 misses the
  offline gate.

### 4.1 H200 LOGISTICS runbook

**Providers ($/GPU-hr, checked 2026-10-18, web):** Lambda ≈ $3.99–4.29 (H200 SXM on-demand, 99.9 % SLA, no
egress fees); Vast.ai marketplace ≈ $4.01–4.34 typical, cheapest tracked offers materially lower, no SLA;
RunPod ≈ $6.35 (per-second billing, 99 % SLA). Plan at ~$4/hr; Vast can halve it, RunPod ~1.5× it.
Sources: [Lambda instances](https://lambda.ai/instances), [Vast.ai H200](https://vast.ai/pricing/gpu/H200),
[RunPod pricing](https://www.runpod.io/pricing) (via the 2026 pricing-comparison survey).

**Upload manifest (exact bytes, verified this session):**

| Item | Path | Bytes | Needed for |
|---|---|---|---|
| Model shard 1 | `/home/svg/LLM/Custom-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | harvest-on-H200 only |
| Model shard 2 | `/home/svg/LLM/Custom-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | harvest-on-H200 only |
| Pack | `/home/svg/LLM/Custom-data/packs/iq3_s/` (tokenizer incl.) | 1.5 GB | harvest-on-H200 only |
| Drafter rt | `/home/svg/LLM/Custom-data/mtp/rt/` (824 MB: dense.bin 116,099,072 + dense.txt 1,851 + experts.bin 707,788,800 + draft_vocab.bin 162,100) | 823,851,823 | warm-start + evaluator baseline |
| 31 tensors | `/home/svg/LLM/Custom-data/mtp/tensors/` (+ manifest + verified.json) | 5.24 GB | warm-start alt + export chain |
| token_embd | `mtp/train-cache/token_embd.bf16` | 1.27 GB | trainer |
| Head rows | `mtp/train-cache/head-subset.bf16` (extract locally first) | 0.21 GB | trainer |
| Harvest records | `/home/svg/LLM/Custom-data/harvest/` (16 M pos, bf16) | ~330 GB | trainer |
| Prompts corpus | leg A's prompts files + repo prose set | ≤ 100 MB | provenance/re-harvest |

Training-only session ≈ 337 GB upload (~45 min at 1 Gbps, ~5 min at 10 Gbps). Harvest session adds 85 GB.

**SECURITY FLAG (operator decision, blocking):** every upload arm exposes the user's PRIVATE artifacts — the
GSQ-RCO IQ3_S quant (own quantization of public Qwen weights), the pack, the refusal-r recipe, and the
drafter weights — to a third-party host. The training-only arm (337 GB incl. the drafter + records) and the
harvest arm (422 GB incl. the full private quant) differ in degree, not in kind. **No upload without explicit
operator consent (OD3).** Mitigations if consented: ephemeral instance, delete on egress, no snapshots, no
shared filesystems, provider with no-egress-fee + private-project posture (Lambda class).

**Engine build for sm_90:** `CMakeLists.txt:161–199` defaults `CMAKE_CUDA_ARCHITECTURES=120` and validates
each arch ≥ 75 (Pre-sm_75 refused; the QSA tf32 mma has a portable fp32-FMA fallback so non-sm_120 cards
build; the CUDA-13 warning applies only to arch ≥ 120). So: `cmake -B build-h200 -DCMAKE_CUDA_ARCHITECTURES=90`
is legal on any CUDA ≥ 12 image; CUDA 13 is NOT required for sm_90. 45 `.cu` kernels + ~209 C++/CUDA sources
→ estimate 15–40 min on a 16–32-core cloud x86 host. **Risk flag:** the engine is developed/measured on
sm_120 (Turing-port comment names 75/86/89/120; sm_90 untested upstream) — the H200 harvest arm must pass a
functional probe (prefill rate + dump-rate + a 100 k-token dump diffed for format sanity), not cross-arch
bit-identity.

**Harvest-on-H200 flow (contingency; primary harvest is the 5090's P1):** H200's 141 GB HBM holds the whole
83.6 GB quant + drafter + KV with `--resident-experts` — zero expert paging, so prefill should exceed the
5090's measured 6,291–6,972 tok/s (estimate 10–15 k tok/s; MEASURE, don't assume — same 30-min 100 k-token
dump probe first, T14B risk 3). Config = byte-copy of `custom-iq3_s-8081.json` with `--mtp` → the uploaded
rt, port 8082, plus `--dump-final-r`; replay the prompts corpus via the server API; identical record format
(§2.1). 16 M positions ≈ 20–45 min compute + dump overhead. Then train on the same box (records never leave)
and egress only §2.5's outputs.

**Egress:** `<out>/tensors/` (5.24 GB) + `mtp-manifest.json` + sha256 sums — minutes, free on Lambda-class
providers. Verify sha256 on arrival, then run P4 LOCALLY (pack/rt/audit stay on the trusted box).

**Cost estimate:** training-only sweep session ≈ upload 0.75 h + env 0.25 h + 6 runs ≈ 3 h + slack 1 h ≈
5–6 GPU-h ≈ **$20–25 at Lambda's $4** (RunPod ~$35; Vast ~$12–20 if a cheap node holds). Harvest-contingency
session adds build + probe + harvest ≈ +2–3 h ≈ +$10. Budget line for the ask: **≤ $40 all-in training-only;
≤ $70 with the harvest contingency.**

---

## 5. THE EXPORT (P4) — command by command

Working model of the chain (verified against the tools this session): trainer writes the mtp_fetch.py layout;
`mtp_pack.py --src <dir> --experts q2_0` reads `tensors/<name>.bin` as BF16 (`load_bf16`, little-endian u16),
keeps dense BF16 (F32 for 1-D norms, RAW — the `+1` fold happens ONLY in mtp_rt.py:85), quantizes the two
expert tensors q2_0 (deterministic 17-step grid search), writes one GGUF; `mtp_rt.py` relayouts experts into
`experts.bin` (gate/up row-interleave, codes/scales planes) and writes `dense.bin` + `dense.txt`
(q8_0 for the 10 projection names in its Q8 set, bf16 for hc/router, f32 norms with `+1` applied, 256-byte
alignment). `draft_vocab.bin` is never in the chain — it is copied.

### 5.0 The identity run (FREE, do this FIRST — T14B §5 dry-run substitute)

Zero training steps: run the STOCK `mtp/tensors/` through export steps 1–3 below into `rt2-id `(new)``
(trainer flag `--identity-export`), then run §5.4's audit expecting EVERY tensor byte-equal to
`/home/svg/LLM/Custom-data/mtp/rt/`. This validates pack+rt+audit end-to-end before a single trained byte
exists (the F-114 round-trip test, replayed on this chain). Any diff = chain bug; STOP.

### 5.1 Trainer → tensors (in `tools/mtp_train.py --export`)

Writes `/home/svg/LLM/Custom-data/mtp-trained/tensors/` `(new)`: all 31 `<name>.bin` BF16 little-endian u16
views (fp32 master cast down; experts COPIED byte-identical from `mtp/tensors/`, not re-saved) +
`mtp-manifest.json` in the exact stock-manifest schema (name, dtype "BF16", shape, file, sha256 — of the
TRAINED tensors). **Norms written RAW (w, never 1+w) — the F-109 law; mtp_rt.py folds at build.**

### 5.2 Pack and rt-build

```
cd /home/svg/LLM/strata-custom-2
~/venvs/mtp-train/bin/python tools/mtp_pack.py \
    --src /home/svg/LLM/Custom-data/mtp-trained/tensors \
    --experts q2_0 \
    --out  /home/svg/LLM/Custom-data/mtp-trained/mtp-trained-q2_0.gguf
~/venvs/mtp-train/bin/python tools/mtp_rt.py \
    --gguf /home/svg/LLM/Custom-data/mtp-trained/mtp-trained-q2_0.gguf \
    --out  /home/svg/LLM/Custom-data/mtp/rt2          # (new dir; rt/ is NEVER written)
```

(`.venv/bin/python` also works — both tools need only numpy + the vendored gguf-py via `tools/_paths.py`.)

### 5.3 The byte-identical copy list

`rt/` layout (verified): `dense.bin` 116,099,072 B · `dense.txt` 1,851 B · `draft_vocab.bin` 162,100 B ·
`experts.bin` 707,788,800 B. Copy into `rt2/`, overwriting mtp_rt.py's fresh outputs:

```
cp /home/svg/LLM/Custom-data/mtp/rt/draft_vocab.bin  /home/svg/LLM/Custom-data/mtp/rt2/
cp /home/svg/LLM/Custom-data/mtp/rt/experts.bin      /home/svg/LLM/Custom-data/mtp/rt2/
```

- `draft_vocab.bin` (162,100 B = 40,525 × int32) MUST be the copy — the head bind depends on it.
- `experts.bin` MUST be the stock copy (F-114 validated config; the pack's fresh q2_0 experts are computed
  but not shipped). **Extra invariant (free):** because the trainer copied the stock expert tensors and both
  quantizers are deterministic, mtp_rt.py's OWN fresh `experts.bin` should already byte-match stock — assert
  it before overwriting; a mismatch = chain bug, STOP (this is the identity run's second face).

### 5.4 The audit (`tools/mtp_audit_rt.py (new)`; precedent = `mtp-iq3s/rt-r17.report.json`)

Read both `dense.txt` files (29 lines: name kind rows cols offset bytes) and produce
`campaign/results/rt2-audit-<label>.json`:
1. `dense.txt` byte-equal between rt and rt2 (same layout/kinds/sizes) — else STOP.
2. Per line: sha256 of `dense.bin[offset : offset+bytes]` in rt vs rt2. Expected for a TRAINED export: ALL
   29 dense tensors DIFFER (dense-only training touches every dense tensor); report the diff list. For the
   §5.0 identity run: ALL equal.
3. `experts.bin` and `draft_vocab.bin` sha256 equal to stock (after §5.3).
4. Summary line vs the r17 report class (total bytes 892,203,008).
**Adoption rule: no boot until the audit passes.** Post-boot MMLU/M3b validate what the audit cannot.

---

## 6. ROLLOUT + ROLLBACK (P5) — pure config, zero engine code

```
cd /home/svg/LLM/strata-custom-2
cp custom-iq3_s-8081.json campaign/configs/custom-iq3s-rt2-8081.json   # (new dir ok)
# edit the copy: ONLY the --mtp value → /home/svg/LLM/Custom-data/mtp/rt2
#   (battery-config.sh reads the config's "log" key — point it at a fresh log,
#    e.g. /home/svg/LLM/strata-custom-2/custom-iq3s-rt2.log, so arms never share a log)
cp launch-custom-iq3s-8081.sh launch-custom-iq3s-rt2-8081.sh           # (new)
# edit the copy: --config → campaign/configs/custom-iq3s-rt2-8081.json  (Launch Law: never setup.py)
```

The parallel-directory pattern is proven (`mtp-iq3s/rt-r17` sits beside production `mtp/rt`); production
`rt/` is never written; rollback = boot the standing launcher (or `=0`-class: nothing to roll back — rt2 is
additive). Boot verification greps (arm boot, before ANY probe):

```
grep -E 'rt2|mtp' /home/svg/LLM/strata-custom-2/custom-iq3s-rt2.log | head   # drafter bind line prints
                                                                             # rt2's path + MiB (mtp.cpp:528)
curl -s http://127.0.0.1:8081/health                                          # ready
grep -iE 'draft|spec' /home/svg/LLM/strata-custom-2/custom-iq3s-rt2.log | tail # spec 6 / min-p 0.5 / q4 / hnorm stream lines
```

Serving arms IDENTICAL in both A/B arms: `--mtp-q4 all --mtp-hnorm stream --spec 6 --spec-min-p 0.5` +
the launcher's full standing env (F-214 lesson: `battery-config.sh` re-evals the launcher's export lines —
a direct server.py boot drops them and the M3b canary will catch it at ~62 %).

---

## 7. GATES (P6) — PASS/FAIL with abort criteria

Order is by cost; each gate has a cheap kill. Gates run on the arm config (§6) vs the standing config,
same hour, same box, arm/ctl env identical except `--mtp`.

| Gate | Instrument | PASS | FAIL/abort |
|---|---|---|---|
| G0.1 bit-identity | `campaign/bit-check.py rt2` (3 prompts, temp 0) vs a standing-boot run — verified-exact outputs mean text MUST be byte-identical | byte-identical | ANY divergence → engine/drafter leak bug; abort, do not pass G0.2 |
| G0.2 stair read | `STRATA_SPEC_DEPTH=1` boot + M1 probe; dp_ok/dp_tested marginals (T14C §1.1) | depth-2 AND depth-3 conditional marginals **+≥8 pp each**; depth-1 ≤ 1 pp regression; offered/window recorded for G5 | pos-2/3 < +8 pp or pos-0 regresses → abort; back to sweep (§4) / arms (§2.7) |
| G1 pre-screen | 3× `campaign/probe-lane.py --port 8081 --label <arm>` same-hour vs 3× standing ctl | M1 ≥ +4 % vs same-day ctl anchors; M3b in the 74–79 warm band | < +4 % → no 3v3 is spent; abort to sweep |
| G2 adoption 3v3 | ABAB 3× arm/ctl boots (`battery-config.sh` + `abab-run.sh` pattern) | M1 ≥ +4 % with arm ahead in most rounds; pair AND conc no-regression (watch F-231's pair flag — an M1-only win at pair-floor does not adopt); M3b ≥ 70 | any lane regression → NO-GO (Never-Regress law) |
| G3 quality | 2× `.venv/bin/python campaign/mmlu-pro-gate.py 0.933 <label>` + mimic lane band | pooled arm ≥ pooled ctl − 1σ (σ ≈ 5.2/run) AND ≥ 0.933 floor; mimic 95.3–98.4 band | floor miss → abort; something is wrong beyond the drafter |
| G4 (post-adoption) | F-229 ungated re-boot: `STRATA_DRAFT_UNGATED=1`, one boot | ≥ +1.5 % M1 at offered ≥ 4 | stays default-off (F-229's −2.73 % stands) |
| G5 (post-adoption) | F-228 min-p re-stair: one `STRATA_SPEC_DEPTH=1` boot at 0.35 (and 0.42) | crossing rule: ΔC/Δoffered > 0.72 tokens/row (T14C §1.4); then ONE confirming 3v3 | below rule → 0.35 stays closed (two measured negatives bracket 0.42) |

**The e2e number, stated honestly (OD4):** R3's v1 gate said e2e ≥ +8 %; T14C's fixed-shape arithmetic puts
the retrain's central at **+5.7 % M1** (per-proposal 78→85 %), +8.7 % only with depth re-unlock, and warns
G2's +4 % is the most likely failure. Recorded resolution: M3b +8 pp (a drafter property) is the hard
quality gate; +4 % e2e is the adoption bar (campaign law); **+8 % e2e is the full-win marker that
unlocks G4/G5 with a clear conscience**. If M3b passes at +8 pp but e2e lands +5–7 %, that IS the
T14C-predicted outcome: the operator chooses adopt-anyway (still the best lever; quality structurally safe —
verified-exact outputs) vs iterate (S2–S6, then LK/AdaSPEC riders). Do NOT silently re-define either number.

---

## 8. SCHEDULE — one implementer, 3 days (P0–P1 day 1 · P2 day 1–2 · P3 day 2 · P4–P6 day 2–3)

- **Day 1 — P0+P1:** trainer venv + torch sanity (0.5 h); window indexer + loader against leg A's harvest
  contract (§2.1; if leg A's dump-rate probe failed, the H200 harvest contingency is decided TODAY, OD3);
  `mtp_eval.py` + stock baseline (§3); **OD1**: operator signs the corpus composition (content-matched, deep
  mass, masked_frac < 2 %).
- **Day 1–2 — P2:** receive exec-B's G0-verified module; wire the trainer loop (§2.2–2.5); S4 shakedown run
  (8 M, 1–1.5 h) end-to-end INTO THE IDENTITY EXPORT (§5.0) — first live proof of loader+module+export+audit.
- **Day 2 — P3:** S1 primary (16 M, ~3 h) watched on the live m̂ dashboard; S2/S3/S5/S6 only if H200 declined
  (else they run there overnight); **OD2**: operator confirms S1 config + w_deep; offline gate read.
- **Day 2–3 — P4+P5:** export chain on the winner (§5, ~0.5 h incl. audit); config copy + boot verification
  (§6); G0.1/G0.2 (~1 h).
- **Day 3 — P6:** G1 (3× probe-lane same-hour, ~1 h) → G2 3v3 (~3–4 h wall) → G3 MMLU 2× + mimic (~1 h) →
  **OD4**: adopt or iterate; post-adoption G4/G5 single boots if offered-depth pre-computations cross
  (**OD5**). Standing restored on 8081 in every exit path.

Operator decision points: **OD1** corpus sign-off (Day 1) · **OD2** primary-run config incl. w_deep (Day 2)
· **OD3** H200 go/no-go + SECURITY consent for uploading private artifacts (Day 1, blocking for any upload)
· **OD4** adoption under the +8 pp/+4 %/+8 % gate set (Day 3) · **OD5** post-adoption re-tests (Day 3).

---

## 9. RISKS table

| Risk | Evidence class | Mitigation (in order) |
|---|---|---|
| Chain-quant loss on fresh weights (offline m̂ strong, M3b weak, gap > 2 pp) | F-113/F-116 (the 57-vs-85 gap); T14A R2 | v1 ships trained dense + STOCK experts.bin + draft_vocab.bin (F-114); gates run under production arms (`--mtp-q4 all --mtp-hnorm stream`); then `--fake-quant` pass (~200 steps, LR 5e-6) and/or fewer quant passes; never ship blind |
| Parity stall (G0 tap mismatch torch-vs-engine) | F-80/F-104/F-105 (the 0/428 class) | exec-B's G0 gate is a HARD pre-training gate; conventions all in-journal (stream hnorm, split-and-add, scalar gate, folded norms); harvest at real pos ids; the §5.0 identity run proves the export side independently |
| Offline acceptance < +8 pp on pos-2/3 | T14C: G2's +4 % is the likeliest failure; RH's own "longer training might yield a better speculator" | sweep iteration is pre-priced: S2–S6 (§4), then the §2.7 arms in T17 rank order (LK → VAT → AdaSPEC → rejection-replay v2); each costs hours, not days; abort bar per gate table |
| Deep windows underfit / shallow regression from w_deep | F-235's two readings (§2.4) | w_deep armed BOTH ways (S1 vs S6); evaluator position buckets kill any arm regressing a shallow bucket > 0.5 pp |
| Harvest too slow or malformed (leg A dependency) | T14B risk 3 (dump in per-token path ~200 tok/s worst case) | 30-min probe law; H200 harvest contingency (§4.1) with all-experts-in-VRAM; records contract pinned in §2.1 so A/C cannot silently diverge |
| e2e gate ambiguity kills a good drafter (or ships a mediocre one) | T14C re-pricing vs R3's +8 % | the OD4 framing in §7: M3b +8 pp hard, +4 % adoption bar, +8 % full-win marker; decision recorded, not silent |
| Post-adoption re-tunes re-open settled negatives | F-228 (min-p 0.35 −1.75 %), F-229 (ungated −2.73 %) | both are POST-ADOPTION single boots (G4/G5) with pre-computed crossing rules (offered ≥ 4; 0.72 tokens/row); never re-armed on the old drafter |
| H200 session leaks private artifacts | new | OD3 consent gate; ephemeral instance, no snapshots, delete-on-egress; training-only arm minimizes (no full quant) but does not eliminate exposure |

---

## 10. Findings by confidence

**High (code/file read or measured this session unless noted):**
- rt/ layout and exact byte sizes; dense.txt = 29 dense tensors; manifest = 31 entries; pack/rt tool
  semantics incl. the RAW-norm fold point (mtp_rt.py:85) and the Q8 name set (`tools/mtp_pack.py`,
  `tools/mtp_rt.py`, `ls`).
- CMake accepts `CMAKE_CUDA_ARCHITECTURES=90` (≥ 75 gate, portable QSA fallback, CUDA-13 warning only for
  ≥ 120) — `CMakeLists.txt:155–199`.
- Gate scripts and exact invocations: `bit-check.py <label>`, `probe-lane.py --port 8081 --label <l>`,
  `mmlu-pro-gate.py <floor> <label>` (needs `.venv` — datasets), `battery-config.sh <config> <label>`
  re-evals the launcher env (F-214), `abab-run.sh` binary-swap pattern.
- Config/launcher contents and the rollout diff surface (`custom-iq3_s-8081.json`,
  `launch-custom-iq3s-8081.sh`); the rt-r17 audit precedent (`mtp-iq3s/rt-r17.report.json`,
  total 892,203,008 B).
- Upload-manifest byte counts (models 54,817,524,224 + 28,800,138,432; tensors 5.24 GB; rt 823,851,823;
  pack iq3_s 1.5 GB); 705 GB free on /home.
- Trainable-param census ≈ 90.5 M from dense.txt shapes (cross-checks F-81/F-114).
- F-228/F-229/F-235 contents and the F-235 internal tension (both journal entries read in full).
- T17 upgrade citations and their evidence classes (T17C §5.1, T17F items 2/3: VAT 2608.30135, LK 2602.23881,
  AdaSPEC 2510.19779, Draft-OPD 2605.29343, LongSpec 2502.17421).

**Medium (multiple independent credible sources):**
- LR bracket 5e-5/1e-4 for a ~90–210 M warm-started MTP head (FastMTP paper + Red Hat speculators article;
  both warm-start).
- H200 pricing ($3.99–6.35/hr across Lambda/Vast/RunPod, 2026-10-18 web).
- H200 per-run 20–40 min (task input + FLOP arithmetic at 150–300 effective TFLOPS; band stated).
- Trainer VRAM 11–16 GB and 2–4 h per 16 M×3ep run on the 5090 (T14B budget + F-78 design point).

**Low (single source or inference — flagged):**
- 9–10 GFLOP/tok effective-inclusive (derived from tensor census; no published analogue for this block).
- H200 prefill 10–15 k tok/s with all experts resident (extrapolation from the 5090's measured
  6,291–6,972; the probe decides).
- sm_90 build cleanliness (the CMake text supports it; no sm_90 build of this engine is on record).
- Cloud build time 15–40 min (source census × typical nvcc throughput; not measured).

**Unknown (searched, not found):**
- Leg A/B outputs (harvest records, G0-verified module) — the trio was killed at zero output (F-235); the
  interfaces are pinned here so they cannot diverge.
- The mimic corpus — ABSENT from Custom-data (F-235 closeout); the corpus decision is OD1.
- Any sm_90 (H100/H200) run of this engine, public or private.

## 11. Contradictions and resolutions

1. **F-235's two entries** ("deficit is positional → oversample deep" vs "mostly CONTENT → position-aug
   demoted"). Resolved pragmatically in §2.4: content-matched harvest (both agree) + w_deep armed both ways
   (S1/S6) with bucket-guarded scoring; LongSpec aug stays demoted.
2. **R3's "+8 % e2e" gate vs T14C's +5.7 % central.** Resolved in §7/OD4: three distinct numbers (8 pp
   drafter-quality / 4 % adoption / 8 % full-win), operator owns the middle case; nothing silently redefined.
3. **Greedy offline match vs coupled sampled acceptance.** Different quantities (LK's own hygiene law);
   evaluator is a ranking instrument; boot G0.2 stays binding.
4. **Trainer LR references disagree (5e-5 vs 1e-4).** Resolved by arming both (S1/S2); primary = RH's 1e-4
   as the same-head-class, same-scale, warm-start precedent.

## 12. Open questions

1. Leg A's dump throughput on the 5090 (probe) — decides whether §4.1's H200 harvest contingency is needed.
2. token_embd source sanity: pack `embd.bin` format unverified — the HF range-fetch is primary until checked.
3. Whether `STRATA_MTP_FULL_HEAD=1` baseline should be banked pre-retrain (T14A Arm 0.1) to attribute the
   sub-vocab head's share — one replay pass; recommended if harvest is cheap.
4. Does the verifier p-list dump (LK/rejection-replay fuel) ride leg A's pass or wait for v2? (T17F says
   ~0.5 d; OD1 can bundle it.)
5. Exact M3b drift expectation under a deeper offered chain (T14C: the RATE can compress while C rises —
   G1 reads M3b jointly with offered/window, not alone).

## 13. Implications

- The 5090-only path is fully specified and costs $0: S1+S4 tonight-class runs, export/rollout/gates Day 2–3,
  expected landing per T14C = +5.7 % M1 central (+2.8 % pair, +2.7 % conc) with the M3b +8 pp drafter gate as
  the real prize and G4/G5 as free compounding options.
- The H200 sweep converts the sweep from a 12–18 h 5090 occupation into a ~$20–40 one-evening session and
  prices the harvest contingency; its only hard gate is the OD3 security consent.
- Every artifact is additive (rt2/, new configs, new tools); production `rt/` and the standing launcher are
  never modified; rollback is a boot choice.

## 14. Sources

In-tree (read this session): `tools/mtp_pack.py`, `tools/mtp_rt.py` (headers + quantizers + fold line),
`tools/mtp_fetch.py` (via T14B), `CMakeLists.txt:155–199`, `custom-iq3_s-8081.json`,
`launch-custom-iq3s-8081.sh`, `campaign/probe-lane.py`, `campaign/mmlu-pro-gate.py`, `campaign/bit-check.py`,
`campaign/battery-config.sh`, `campaign/battery-boot.sh`, `campaign/abab-run.sh`,
`SVG-CUSTOM.md` F-228/F-229/F-230/F-233/F-234a/F-235 (both entries), `SVG-CUSTOM-TODO.md` (T17 queue +
consolidated verdict + STEP-0 results), `campaign/research/T14A-fastmtp-method-recipe.md`,
`T14B-fastmtp-training-infra.md`, `T14C-fastmtp-acceptance-math-gates.md`, `T17C-spec-algo-leaps.md`,
`T17F-training-quant-context.md`, `INDEX.md`, `AGENTS.md` (campaign doctrine), `mtp/rt/` + `dense.txt` +
`mtp-manifest.json` + `tensors/` listing, `mtp-iq3s/rt-r17.report.json`, `models/` + `packs/` listings,
`df -h`. Web: [Lambda instances](https://lambda.ai/instances) ·
[Vast.ai H200 pricing](https://vast.ai/pricing/gpu/H200) ·
[RunPod pricing](https://www.runpod.io/pricing) (2026-10-18; H200 $3.99–6.35/hr band) · the T17-cited
papers by arXiv id only (2608.30135 VAT, 2602.23881 LK, 2510.19779 AdaSPEC, 2605.29343 Draft-OPD,
2502.17421 LongSpec, 2509.18362 FastMTP) — already full-text verified in T14A/T17C/T17F; not re-fetched.

## 15. Methods note

Unproductive searches this session: no harvest/records/dump corpus exists anywhere under Custom-data
(F-235 had already said MIMIC is absent — confirmed by find); no trainer/module code exists in-repo (T14B's
negative re-confirmed by the tools listing); no T14B-exec sibling files exist (the trio died in the crash —
this file pins the A/B interfaces itself). The mtp-iq3s/rt-r17 directory was read as the audit-format
precedent rather than re-deriving a layout. Web pricing was taken from a 2026-08 comparison survey plus the
three provider pricing pages as summarized in search results; per-provider quotes move hourly — re-check at
OD3 before booking.
