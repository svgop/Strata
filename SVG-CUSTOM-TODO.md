# SVG-CUSTOM-TODO — external harvest queue, tracked with citations (2026-10-16)

Companion to SVG-CUSTOM.md. Every externally-sourced adoption candidate lives here with its
source, mechanism, evidence, estimated gain for our box, gate class, and status — so nothing
harvested can be forgotten. Statuses: [QUEUED] in the campaign queue · [GATED] needs MMLU /
acceptance gate before adoption · [PARKED] deliberately deferred · [SKIP] rejected with reason.
Campaign journal (F-entries) stays in SVG-CUSTOM.md; this file is the intake ledger only.

Box context for all gain estimates: RTX 5090 32GB sm_120, 96MB L2, 9950X3D, 184GB DDR5,
IQ3_S pack, standing M1 ~180-192 / pair ~171-175 / conc ~210-219 vs targets 191.2/193.8/244.0.

---

## T1 — tqllm harvest (turbo-tan/tqllm)

Source: https://github.com/turbo-tan/tqllm (Apache-2.0, C++/CUDA, created 2026-10-02,
v1.0.0 2026-10-14; verified 2026-10-16 via GitHub API + raw file fetches). Independent engine
for the SAME model class we host (Qwen3.8-Flash-Next MoE: qsa indexer attention, PLE n-gram
layers, GDN recurrence). NOT our upstream (Niko1221/Strata) — a sibling implementation, which
is why technique transfer is direct. Head-to-head battery of the same day is recorded in T5.

- **T1.1 Shared expert fused into routed launches** — [QUEUED, T-queue after T1.4 hc wide-grid]
  Mechanism: shared expert as routed row e, gate `1/(1+e^-x[e])`, appended (k+1)-th slot with
  `cold_out=-1`, recombined in the same sum kernel. Source: `kernels/moe.cu` (router_k /
  extend_ids_k / sum_k), kill-switch `TQLLM_MOE_FUSE_SHARED=0`.
  Evidence: IQ3_S 207.8→220.1 tok/s (+5.9%), Q5_K_M 176.5→187.1 (+6.0%), identical output
  (their CHANGELOG 1.0.0, RTX 3090).
  Our gap: our path is `route → publish → shared expert → combine` with fp16 weights and own
  scratch (include/strata/core/layer.hpp:112-150) — separate launches, so the fusion applies.
  Est. +4-6% M1. Exact-output class (no MMLU gate needed beyond a canary boot).

- **T1.2 int8 prefill GEMM (cuBLASLt IMMA)** — [GATED]
  Mechanism: per-row dynamic symmetric quant (absmax/127 RTN) of activations per-token and
  weights per-channel, `CUBLAS_COMPUTE_32I` int32 tensor-core GEMM, dequant c·sw·sx.
  Source: `kernels/int8_gemm.cu`; `default_on()` = Ampere + sm_89 + sm_120 ("GeForce int8
  full-rate, FP16/FP8 with FP32 accum half-rate"); 8 cuBLASLt heuristics timed once per shape,
  cached, never timed inside graph capture.
  Evidence: 3090 20.7K prompt 1640→2123 tok/s (+29%); sm_120 5060 Ti 27.5K 1066→1200 (+11-13%,
  2026-10-09 comment in-file).
  LOSSY — MMLU + M3b gate class before adoption. Est. +10-15% prefill (priority metric with no
  big lever queued).

- **T1.3 Expert-offload cache upgrades** — [QUEUED, conc lane]
  Source: `src/mixed_expert_cache.cpp` (32KB) + `kernels/moe.cu` staging kernels.
  Mechanisms, each citable in-file:
  - decayed-frequency replacement `freq/(1+0.05·Δclock)`, recency tie-break, tuned by trace
    replay (kValueDecay=0.05);
  - prefetch admission for low-confidence ranks 4-6: skip if victim `(age<8 && val≥2) ||
    (freq≥4 && age<16)`;
  - `TQFN_PF_EVENTS`: one CUDA event per prefetched expert — compute waits only on slots it
    reads; in-flight unread slots are not demand victims;
  - split miss stream (separate upload stream, released/ready events; `wait_misses()`) — hits
    keep running while misses load;
  - zero-copy demand misses: kernel reads pinned host memory directly (`kZeroCopyParts=2`),
    skipping the DMA queue behind prefetches.
  Evidence: 16GB budget, MiniCPM-V 4.7 35B-A3B Q5_K_M: 71.9/112.5/54.9 → 96.2/146.3/85.9 tok/s
  (+34/+30/+56%; CHANGELOG), beats llama.cpp `-ncmoe 19` (62.8). Placement-only (output
  unchanged). Our 32GB box sees less thrash; est. +5-15% conc (our weakest gap: 219 vs 244).

- **T1.4 Matvec rule table + Split16-class kernels for IQ3_S** — [PARKED until 5090 re-bench]
  Source: `kernels/formats/matvec_rules.hpp` — ordered per-(format-bits, cc, k/32, rows) rules,
  each citing its microbenchmark; runtime override `TQLLM_MATVEC_RULE`; choice never timed at
  runtime. Bits 21 = IQ3 exists in their table.
  Evidence (3090): IQ3 short-k split16 73.3 vs legacy 80.3 µs (−8.7%); TQ3 long-k split16+pf1
  64.2 vs 78.9; Blackwell 5060 Ti: legacy wins for TQ3 (416 vs 476-543 µs) — crossovers are
  cc-dependent and our 96MB L2 moves them again. Adopt the mechanism, write the split16 variant
  for our shapes, re-bench on 5090 with their `tqkbench` methodology. Est. +3-8% decode IF a
  variant wins on our shapes.

- **T1.5 Prompt-lookup drafting** — [QUEUED, cheap; pairs with mimic lane]
  Source: tqllm `--lookup L` (6-gram prompt match, `--lookup-first K`); constraint
  `kGdnReplayCols = 8` in `src/tree_columns.hpp` ("lookup on top of a full 7-draft tree failed
  every request" — replay-column budget).
  Note: that 8-column GDN replay budget may also explain OUR stair diagnostic (depths 4-6 never
  offered) — check our equivalent cap when the stair lane runs.
  No weights, composes with MTP, helps repetitive text = our mimic/adherence lane. Est. +5-15%
  on mimic/acceptance-heavy lanes.

- **T1.6 Two-stage LM head (4-bit shortlist → exact int8 re-score)** — [GATED, verify-path care]
  Source: `src/qwen35.cpp` `build_head4`/`head8` (host code); 0.64 GB vs 1.27 GB head read per
  decode step (Qwen3.8). Their own README: "an optimization, not a certified full-vocab argmax"
  — verify path must keep the exact head (theirs does; ours must too).
  Clean subset: draft-only reduced head (gathered rows of output.weight for draft argmax only,
  verify keeps full head — exact by construction; `tools/tqfn_stage.cpp`
  `TQFN_DRAFT_VOCAB="llama-mtp-vocab-v1 V K"`). Est. +2-5% decode.

- **T1.7 DFlash2 block-diffusion drafter** — [PARKED, multi-week]
  Source: `kernels/dflash.cu` + `src/qwen35.cpp`; paper arXiv 2602.06036 (z-lab, ICML 2026);
  ecosystem: vLLM speculators guide, NeMo recipe. 0.79GB drafter, 65,536-token draft vocab,
  device-side best-first tree (256-node pool, depth 16), retrieval window (2048 pos, ≤128
  far-context chunks re-rotated into it; `dfl_chunk_score_k`/`dfl_chunk_top_k`/`dfl_gather_k`).
  Evidence: 3090 131 vs 51 tok/s (32K summary), 155 vs 55.5 (2K), bit-exact vs greedy.
  Their MoE MTP is "not wired yet" — our native MTP already carries us. Est. +10-30% long-ctx
  decode IF it out-drafts native MTP. Parked behind current queue.

- **T1.8 KV compaction via attention matching** — [PARKED, 262k/1m lane]
  Source: `kernels/kv_compact.cu` (23KB); paper arXiv 2602.16284 (MIT, Feb 2026), code
  github.com/adamzweiger/compaction, llama.cpp issue #20037 requests the same. Idle-time GPU
  compaction: NNLS mass-preserving stand-in weights (multiplicative updates), ridge least
  squares for V stand-ins (cuSOLVER Cholesky), per-row log-weight bias, middle layers
  [A/4,5A/8) at half ratio (measured: 8x error +29-49% there vs +3-16% at ends).
  ~0% on the standard battery; capability for long-context configs. LOSSY → own gate.

- **T1.9 CUDA-graph hygiene** — [QUEUED, reliability]
  Source: `src/graph_lru.hpp` + `src/graph_capture.hpp` + `src/batch_graph_key.hpp`.
  GraphLru: ~12MB/graph accounting, LRU cap 256, OOM → quiesce → clear all → retry once,
  evict only after the stream passed. Capture RAII: a throwing body still ends the capture
  ("one failed request into a dead server"). Key hashing mixes mode before layout fields.
  0% perf; adopt wherever our instantiate_evicting pool lacks these semantics.

- **T1.10 VMM-aliased KV tiers** — [PARKED]
  Source: `src/kv_store.hpp` (tiered policy: one reserved VMM range, device+host pages under
  one pointer; two staging buffers by attention-layer parity; never call cudaMemGetInfo —
  count every byte). Builds on our existing `include/strata/core/vmm.hpp` (--kv-grow arena).

- **T1.x Rejected from tqllm (with reasons)** — [SKIP]
  - FP8 prefill GEMM (`kernels/fp8_gemm.cu`): their own gate rejected it — GB10 +40% prefill
    but KLD 0.010-0.014, PPL +2.7%, "a wrong 16K answer"; default_off everywhere. Our gates
    would reject it too.
  - CPU expert miss pool (`src/cpu_expert_matvec.cpp`, `TQFN_CPU_MISS`): CPU numerics
    explicitly not bit-identical to the GPU path; IQ4_NL only in their AVX2 code. Conflicts
    with our canaries; PCIe DMA (not host compute) is our offload bottleneck.
  - Grouped prefill GEMM lane: pre-gate in their own docs/MOE_STATUS.md ("do not cite
    performance until gates pass").
  - TurboQuant TQ3/TQ4 rotation (arXiv 2504.19874, types 300/301): a re-quantization project
    (new basis mechanism + re-quant + KLD gate), not a flag. Long-horizon quality-per-bit play;
    revisit if IQ3_S quality ever becomes the binding constraint.

## T2 — unbug tuning note (X/Twitter, 2026-10)

Source: https://x.com/unbug/status/2110794862467641554 — "+72% prefill short ctx, +27% long,
VRAM hit rate pinned 100%" on Qwen3.8-Flash Q6-KXL via `--pcie-frac`, `layer_split`,
`--trim-stage-weights`; "MTP still isn't supported with parallel request."

Our status, verified in-tree 2026-10-16:
- `--pcie-frac`: we run it daily — standing config `--pcie-frac 0.35 --pcie-mode dma`
  (custom-iq3_s-8081.json; readers at include/strata/core/expert_source.hpp:468,
  verify.hpp:317/:541).
- `--layer-split` + `--trim-stage-weights`: exist in our tree (src/program/generate.cpp:653/
  896/2162; :3165 notes PR #559 = STRATA_STAGE_TRIM=1 of PR #639). They are MULTI-GPU
  layer-split knobs ("each GPU loads only its own layers' dense weights") — unbug's +72% is a
  multi-GPU recipe. Single-5090 box: not applicable; nothing to do.
- "MTP with parallel requests — solved?" **YES — our F-184** (SVG-CUSTOM.md:5984,
  2026-10-14): coupled slot drafting — drafter samples with the target's chain + the same
  Philox draw, quality-safe by construction (the uncoupled and rescue variants were
  quality-banned). Standing env `STRATA_BATCH_MTP=1 STRATA_SPEC_COUPLED=1`; pair +11.5%,
  MMLU 97.14 PASS. Upstream stock still ships draftless concurrent slots. This is also
  PR-value item P3 for Niko (campaign/research/PR-UPSTREAM-plan.md).

## T3 — llama.cpp v0.7.0 harvest

Source: https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0
Status: research agent dispatched 2026-10-16 (background). FINDINGS PENDING — fill in on
return.

## T4 — Star Elastic harvest

Source: https://arxiv.org/html/2605.07182v1 ("Star Elastic: Many-in-One Reasoning LLMs with
Efficient Budget Control"; used by NVIDIA to compress Nemotron 3.5 Super VL 120B-A12B →
Nemotron 3.5 Bolt 74B-A9B 24GB).
Status: research agent dispatched 2026-10-16 (background). FINDINGS PENDING — fill in on
return. Prior: likely training-side (useless to a fixed-weight engine) — agent instructed to
separate "requires training" from "inference-time policy".

## T5 — Head-to-head battery: our Strata vs tqllm, same model (2026-10-16)

Method: our engine rebooted fresh per round via campaign/battery-boot.sh (F-209 driver),
probe-lane.py lanes, means over 3 rounds. tqllm: cloned to ~/LLM/tqllm @c7f9852, built with
CUDA 13.4/sm_120, run via its standalone tqfn-stage --decode path (fresh process per round,
prompt x2 per process, warm pass = measurement; their serve engine could not be used — see
F-213: MTP manifest port = architecture reconciliation, declined). One 2-line patch to their
src/tokenizer.cpp (GsqRco fallback for type-42 shards). Full record: F-213 in SVG-CUSTOM.md +
campaign/results/tqllm-r{1,2,3}.txt.

| Lane | Our Strata (standing, spec 6 + MTP) | tqllm (no drafting possible) |
|---|---|---|
| M1 single-stream | **180.9 t/s** | 47.5 t/s (hit 93.0%) |
| M2 pair | **171.7 t/s** | not runnable (serve engine requires the MTP port) |
| M2 6-conc | **213.9 t/s** | not runnable (same) |
| M3b acceptance | 76.4% | n/a (no drafter) |
| Prefill (1783-tok prompt) | not probed this battery (separate lane) | 54.2 t/s warm / 36.8 cold (chunk 8, their serve default) |

Verdict: tqllm is 3.8x behind on M1; the no-drafting handicap explains at most ~1.9x, so
plain-vs-plain stays >=2x. Their MoE prefill lane is pre-gate by their own docs and caps at
chunk<=51 (chunk*topk <= slots <= 512 experts) — chunk 48 ran >20 min without finishing on our
box. Engineering note: their end-to-end loss does NOT devalue the T1 harvest — the techniques
(shared-expert fusion, expert-cache policies, matvec rule tables) are measured at kernel level
in their tree and transfer independently. Their GsqRco dialect support (gguf.cpp:232) confirms
the shared artifact family; our no-spec parity boot was invalid because our native pack
contract requires --spec >= 2.
