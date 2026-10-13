# THE MTP HEAD REFERENCE — the formulation, the chain, and the defects (F-122)

The reproducible record of the MAGMA v2 drafter work. Companion to F-84..F-121 (the dated
flight-log narrative); this file is the *standing reference* a stranger or a future session
needs to rebuild, extend, or share the work. All paths are absolute for this estate.

## 1. The drafter's architecture (what the engine runs)

The MTP drafter is the model's own **nextn layer 48** (Qwen3.8-Flash-Next family), not an
EAGLE/Medusa tree. Per verify window it drafts up to 4 chained tokens with:

- **the shared attention block** (`self_attn.*`) over the target's session KV, with the
  **QSA indexer** (`self_attn.indexer.{q,k}_layernorm`, `index_qk_proj`) selecting the
  attended key ranges — the "Qwen Sparse Attention" path;
- **hyper-connection mixers** (`hyper_connection_mixer.*`, and per sub-layer
  `attn/mlp.hyper_connection.*`) — the HC channel-mixing that gives this family its name;
- **a MoE FFN** (`mlp.gate`, `mlp.experts`, `mlp.shared_expert*`);
- **the head stack**: `nextn.enorm`, `nextn.hnorm`, `nextn.eh_proj`, `nextn.hc_head.*`;
- **a reduced 106,299-entry draft vocabulary** (`draft_vocab.bin` + the Q4_0 sub-vocab head)
  — the engine drafts over the subset, not the full 248k vocab.

Engine source anchors: `src/core/mtp.cpp` (load, bind, draft, fuse), `src/core/fused_gr.cu`
(the hyper-connection read kernels), `src/program/generate.cpp` (the verify windows, the
acceptance tests).

## 2. The formulations (verified against source, F-90..F-95)

**The hyper-connection read (fused_gr.cu; single-row and multi variants are identical):**
`xn = R × gamma × rs` at BOTH the down input and the gate stage; `lo = silu(down(xn) / HC)`;
the output `mixed = mean_c(...)` over the HC streams. The engine's norm here is a single
**grouped** norm (per-slice RMS × the gamma slice), NOT the triple-norm the paper's code
suggests — F-84's fix (commit aea3481) took the std-ladder from 12×-worse-than-uniform to
near-uniform by adopting the engine's form.

**The fuse (`mtp.cpp:709-719`):** the per-stream `[e; h]` concat of width 5120 splits into
`fc_embedding(e) + fc_hidden(hn)` and the two are **added**; equivalently the export merges
the two matrices as `eh_proj = [fc_emb | fc_hidden]` concatenated on dim 1 (torch (out,in)
convention). Getting the half-order or the split-and-add wrong breaks every draft.
Commit af740c6 records the merge; F-85/F-86 the confirmations.

**The hidden-norm (`--mtp-hnorm stream`, F-43):** the drafter's hidden-norm is per-stream
RMSNorm × the gamma slice — matching llama.cpp's reference graph, not a pooled form. This
one flag moved the depth-decode class past its goal (102/121/146 t/s).

**The sub-vocab head:** 106,299 entries (`dhead_`, `draft_vocab` subset, Q4_0 re-quantized,
`dvocab_host_` maps to full-vocab ids). `--mtp-q4` keeps it at Q4_0 (146 MB/step read vs
212.9).

## 3. The dense-weight format (`dense.txt` / `dense.bin` / `experts.bin`)

`dense.txt`: one line per tensor — `name kind rows cols offset bytes`; `kind ∈ {q8_0, bf16,
f32}`; offsets 256-byte aligned. `experts.bin`: the MoE blobs as GGUF Q2_0 blocks relaid.
`draft_vocab.bin`: the sub-vocab map. **All four files are required** — a missing
`draft_vocab.bin` silently mis-routes every draft (F-112).

**The norm conventions (the two that cost hours):**
- `*norm.weight` tensors are **f32** in the rt; the engine applies the **+1 Gemma fold** at
  rt build (`mtp_pack.py`/`mtp_rt.py` line ~85: `w -> 1 + w` for f32 norms).
- The rt-exported head-init therefore **already carries the fold**. Any graft that writes
  those values back through the pack must **UNFOLD (`w - 1`)** or the norms land at 2+w
  (F-109/F-110 — the double-fold that produced 0% acceptance for hours).
- `mlp.shared_expert_gate.weight` is **bf16** (1×2560), NOT f32 — folding it or writing it
  f32 breaks the router gemv (F-102's "null or misaligned pointer").

## 4. The training chain (the 10-minute loop)

1. **Data** — `--dump-nextn` pairs dump from the buun fork (`bench/lora-data/pairs-r17-gsq.bin`,
   111,071 records; x pairs at ctx 2048; the dump writes h rows + target tokens).
2. **Train** — `E:\LLM\Engine\buun-llama-custom\bench\fastmtp\train-fastmtp-v5.py` (CE) or
   `train-distill.py` (teacher-student KL; the teacher and student share the identical
   mixed-KV forward so the window-KV approximation cancels — F-116/Q3). 112 s per 1.5k
   windows on the 5090.
3. **Graft** — `graft_v2.py` (+ `graft_r17.py`'s MODULE_TO_GGUF map; `map_stock28.py` for
   the export→sidecar name transform): writes the 25 trained dense tensors into a stock
   BF16 copy of the sidecar.
4. **Adapter** — `E:\LLM\Engine\strata-custom\tools\mtp_phaseb_adapter.py` (GGUF → the
   per-tensor fetch layout, `r17-fetch/`).
5. **Splice** — `mtp_indexqk_splice_r17.py` (installs the standing `index_qk_proj`).
6. **Pack** — `mtp_pack.py` (fetch → a q2_0 GGUF; *norm-only f32 rule*).
7. **rt build** — `mtp_rt.py` (GGUF → `dense.bin`/`dense.txt`/`experts.bin`).
8. **Assemble** — copy the **stock** `experts.bin` + `draft_vocab.bin` in (the dense phase
   never trains the experts; the pack's own expert requant is broken — F-113/F-114).
9. **Boot** — `launch-v2-r17-8081.bat` + `strata-iq3_s-r17.json`.
10. **Gate G1** — read `drafts accepted N of M` from `strata-iq3_s.log`.

## 5. The six graft-chain defects (each proven in raw bytes)

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | every 2D tensor transposed | QSA "requires a stream, aligned spans" | write (out,in) AS-IS; no `.T` |
| 2 | `hc_norm` values bf16 in the rt | "fused_gr_read_multi invalid arguments" | pack writes `*norm.weight` f32 |
| 3 | the double +1 Gemma fold | raw bytes: 1.2305 vs stock 0.2305 on `indexer.k_layernorm` | UNFOLD in the graft |
| 4 | only 13/28 tensors name-mapped | graft reported 13 matched | `map_stock28.py` |
| 5 | `draft_vocab.bin` missing | 0/462 with perfect dense weights | copy the stock file |
| 6 | the pack's expert requant (Q8_0→q2_0) | whole-blob hash differs; stock swap → 85.4% | dense-through-chain + **stock experts.bin** |

## 6. Measured results (same-window, Proof Law)

- **The chain validated:** 85.4% draft acceptance (287/336) at 133.9 t/s on the corrected
  stock round-trip (F-114) — the stock class, through our own graft/pack/rt path.
- **The distillation transfer:** 57.4% first gate (F-115); 8-epoch plateau ~55.7% — the
  ceiling is the chain's double-quant on fresh weights, not the training (F-116).
- **Production (official head + arms):** 92.5% acceptance; single-stream warm 113-129 t/s;
  MMLU 95.7%; hit rate 95-96%.

## 7. The engine-side levers that produced the v1 lead

Recipe: `launch-v2-battery-8081.bat`. Adoptions, each with its measured margin in the log:
`ADAPT_OVERLAP`+`_RESIDENT` (F-67), `ROUTE_RESIDENT=0.5` (F-68), `ROUTE_RESIDENT_MTP=1`
(F-72; the drafter-router residency bias — an acceptance lever), `SPEC_COUPLED=1`,
`TYPICAL_ACCEPT=0.2` (the first-rejection rescue), `--mtp-hnorm stream` (F-43),
`EXCHANGE_ROTATE`, `L2_SETASIDE_MIB=60`, `PINCAP=48/MIN_AGE=2/DORMANT=64`. **Retracted:**
`DF_PDL=1` (F-107 — cost 40% on sustained reasoning) and `SPEC_RATIO=1` (F-118 — cost 21%).

## 8. The open items (queue, in order)

1. **The process degradation** (F-119): a long-lived process fell to 10.3 t/s where a fresh
   boot read 101.6; suspects: conversation cache growth, the adaptive tier's state after
   mixed traffic, KV/pipe growth, L2 set-aside after window recaptures. Biggest single
   lever found.
2. **The F16-preservation chain upgrade** — removes the double-quant that caps the
   distilled head at ~57%.
3. **RouterLookahead** (F-63 GO) — the doorbell elimination at depth.
4. **The sustained-context KV/prefetch study** — the real-workload class runs 1600+.
5. **The pack's expert requant fix** — re-enables full-phase (expert-training) heads.
