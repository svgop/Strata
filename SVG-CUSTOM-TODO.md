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
  MMLU 97.14 PASS. **2026-10-16 correction, byte-level verification: this is NOT a PR
  candidate — the whole mechanism (coupled_draft.hpp + --batch-mtp flag + docs + SYCL copy)
  already lives in upstream/main, and our generate.cpp BATCH_MTP site is byte-identical to
  theirs. Our delta = one env line + the evidence. Per operator rule ("if it's just a flip
  there is no PR"), the vehicle is an ISSUE with the measured case for flipping the default
  (draft banked in the session record); PR-UPSTREAM-plan.md P3 reclassified into the
  evidence-backed-default-flips class (was P6).**

## T3 — llama.cpp v0.7.0 harvest (research agent report, 2026-10-16)

Source: https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0 (v0.6.0→v0.7.0 = 240
commits; ggml v0.27.0). Ranked by value to us; every item cites its PR.

- **T3.1 GDN/recurrent state-snapshot fusion** — [#29807] write recurrent-state snapshots
  directly into the recurrent cache inside the scan kernel (both K>1 speculative and K==1),
  no separate cpy pass. Measured: Nemotron-3.5-Lightning-30B-A3B Q4_K_M, RTX PRO 6000, WITH
  MTP: server decode 432.58→447.63 tok/s (**+3.5%, faster on 80/80 prompts, acceptance
  unchanged 0.777**). Direct fit: our GDN snapshot writes during MTP verification.
  [QUEUED → T6]
- **T3.2 MTP catch-up crop pattern** — [#29928] zero-output forwards (MTP catch-up,
  draft-context prefill) gather rows so FFN/head run over zero rows: 4-token catch-up
  6.9→2.9 ms of kernels (**−58%**); `n_outputs` in the graph hash, "no reallocation".
  Caution: sparse-attention graph reallocation trap documented (#29867) — our qsa+graphs
  risk area. GLM-5.3-Flash reporter: MTP doubles decode at short ctx (97% acceptance),
  +52% at 32k. [QUEUED → T6]
- **T3.3 GPU LRU expert cache for host experts** — [#29887] `--moe-cache-mib`: host-resident
  experts computed on GPU against an LRU cache; only misses cross PCIe; batch ≤32. RTX 5090,
  Qwen3.8-Flash-Next Q4_0 93.7GB: 30.8→54.5 t/s (1.77x, 77% hit) → `-cmoe` + 19GB cache
  **67.8 t/s (2.20x, 89% hit)**. Lessons: bank slots ≥ per-token expert fan-in (GLM-5.3-Flash
  sizing bug → 0.7% hit); PCIe 3.0 loses. The unmerged follow-ups matter more to us than the
  base (we already sit at 93% hit): async/overlapped cache updates, LFU-with-decay +
  admission (91% hit at 36% cache, ~4 uploads/token vs ~30 for LRU). [feeds T1.3 / T7.C]
- **T3.4 CUDA top-k shape dispatch** — [#28713] bitonic (short rows) / chunked radix (many
  long rows) / CCCL DeviceTopK (1-2 rows; CCCL ≥3.4.3 — ships with our CUDA 13.4). RTX 5090
  micro geomean 31x; e2e Qwen3.8-Flash-Next prefill 1.625x on GB300; TG ~1.0. → our qsa
  cell-selection top-k. Exact-class; prefill-side win. [QUEUED]
- **T3.5 Chunked GDN prefill (BF16-MMA tensor-core scan)** — [#29353 + #30087] prefill-only,
  batch >2048, **K==1 only** ("prefill with speculation would not use this kernel").
  RTX 5090: +16.4-17.9% pp2048 (Qwen3.5-35B-A3B); wikitext ppl 5.7773→5.7818, 98.8% top-token
  agreement. → our GDN prefill lane. [GATED: mimic/adherence A/B]
- **T3.6 Correctness audits from contract changes**: ggml `mul_mat_id` with duplicate expert
  ids per token is now UB by contract (ggml v0.27.0) — audit our router; #29994 k-pool
  scatter data race on shared sequences — audit our qsa selection; #30014 ggml symbol
  visibility hidden by default — check vendored linkage at next sync; #30514/#30574
  draft-model KV checkpoint bloat fixes (37GB class) if we ever persist sessions.
- **T3.7 Graph discipline**: #30223 reserve and decode must build through the identical code
  path with identical sampler chain counts (graph-stability law); #30329 realloc callback +
  perf line (cheap observability for our no-realloc budget); #30042 keep ONE sparse decode
  path, let attention skip masked rows via an n_kv_max-style bound — never fork graph
  topology per ubatch (dense-vs-FA gap ~3x on one backend — validate our path); #30480 build
  the selection tensor in the consumer's layout (byte-identical strides, fewer reshape
  nodes).
- **T3.8 Pinned staging ring for bulk uploads** — [#29608; SYCL-only code, pattern
  transfers]: 4-slot pinned ring, side stream, event-gated (load time 96.8→47.8s on 3x Arc).
  Companion to our expert-miss upload path.
- Skip list: all Vulkan/Metal/SYCL/OpenCL backend work, CI/docs, non-our-class archs,
  tokenizers. Note for next vendored sync: server default port now 9931 and chat output is
  an array of parts (tooling expectations).

## T4 — Star Elastic harvest (research agent report, 2026-10-16)

Source: https://arxiv.org/abs/2605.07182 (+ Nemotron 3.5 Bolt model card/config.json, HF
artifacts; LiLiCorr arXiv 2608.20530; QAD arXiv 2601.20088). Verdict up front: the core
method — nested trained submodels, zero-shot tensor slicing, phase-switching (small model
thinks / large answers; up to +16% acc, 1.9x lower latency) — is ALL training-side and
unusable on our fixed GGUF. Layer-skip/early-exit on non-elastic checkpoints is explicitly
unsupported (depth was the worst axis even WITH training; dropped from final models).
Usable edges only:

- **T4.1 Prompt-level reasoning-budget control** (pure serve-side logic): hard thinking-token
  budget + forced `</think>`; `enable_thinking` / `reasoning_effort` are plain chat-template
  string injections (our model family already ships `reasoning_effort`). Cheap.
- **T4.2 Strip prior-turn thinking from KV history** (`preserve_thinking=false` pattern /
  Bolt's `truncate_history_thinking`): multi-turn agent loops carry dead `<think>` blocks in
  KV — removing them is a context/KV win independent of any training. Cheap.
- **T4.3 REAP expert-importance for placement** — S_j = mean over active tokens of (router
  gate × expert output norm), computable OFFLINE on our frozen weights from a calibration
  corpus; drives VRAM-resident vs host placement + prefetch priority, with PER-LAYER budgets
  (Bolt's config proves heterogeneity: 240-496 experts/layer kept, widths 1536-2560).
  Honest caveat: REAP→cache-value is an untested inference (low confidence, worth one A/B
  against our decayed-frequency policy). [feeds T1.3]
- **T4.4 Drafter path validation**: NVIDIA's Bolt recipe = separately-trained block drafter
  (DFlash) + LiLiCorr lattice reranker (+9-19% acceptance length, +4-13% throughput over
  vanilla DFlash) — target weights untouched. Already parked as T1.7/T7.B.
- **T4.5 Quality-per-bit datapoint**: Bolt ships IQ2-XXS at 2.0625 bpw recovered by
  quantization-aware distillation — ~2 bpw is viable ONLY with a training budget we don't
  have; our floor stays IQ3_S-class ~3.4 bpw. No action.

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
contract requires --spec >= 2. Clone deleted 2026-10-16 per operator directive (harvest and
battery records remain banked here and in campaign/results/).

## T7 — Deep-dive recon: through-the-stack swap research (2026-10-16)

Three researcher agents dispatched with a full-rewrite mandate (any kernel/format/system may be
swapped if proven better or higher-ceiling). Distinct domains, no overlap with T3/T4:
- T7.A decode/attention kernels + Blackwell compute (qsa attention swaps, GDN scan SOTA,
  MoE low-bit GEMM vs per-expert matvec, sm_120 tcgen05/FP4/FP8 ceilings, L2 persistence).
- T7.B speculative decoding SOTA (EAGLE-3/Medusa/PLD acceptance-depth frontier, tree vs chain
  with exact-match verify, DFlash2+LiLiCorr, draft-temp theory, GDN/hybrid state-replay costs).
- T7.C quantization + memory/offload systems (IQ3_S vs MXFP4/NVFP4 per-bit on 5090, expert
  offload policies incl. llama.cpp #29887, KV format frontier, pinned-ring/VMM/L2 windows,
  single-stream ceiling estimate at 3.4-4.5 bpw on 32GB+184GB).

ALL THREE REPORTED 2026-10-16. Condensed findings:

### T7.A — kernels/Blackwell (FlashInfer 0.6.14-0.7.2, cuDNN 9.7.1-9.27, TRT-LLM, SGLang, ~36 sources)

**HEADLINE VERDICT: we are overhead-bound, not bandwidth-bound.** Roofline from our own numbers:
weight-streaming floor ≈ 2.5x current 180.9 (~400-500 tok/s); the surplus is launch gaps, legacy
grids, GDN serialization, doorbell round-trips. Kernel-level 2x wins ship as +5-8% e2e in every
engine measured — the 2x lives in step STRUCTURE, not single kernels.

Sure wins (measured on our model class / hardware class):
- **T7.A.1 IndexShare-MTP indexer reuse** [SGLang day-0 for Qwen3.8-Flash-Next, S10]: capture
  the QSA top-k selection at draft-extend; all N draft steps reuse the frozen selection.
  Shipped for this EXACT architecture; "accept length unchanged"; indexer invocations N→1.
  Output-safe by construction. Small migration (metadata + 1 kernel).
- **T7.A.2 Wide-grid occupancy fix + occupancy pass over ALL small step kernels** — confirms
  T6 items 1-2. Kernel-level 1.3-2.4x precedents (KDA 1.33x, HC ~2x, MSA 1.9-2.4x).
- **T7.A.3 Hyper-connection Mix/Combine fusion into split-K kernels** [SGLang, our model
  class]: Mix 12.36→6.03 µs (2.05x), e2e +7.6%; Combine 4.17→2.13 µs (1.96x), e2e +5.49%.
  Same math fused — no quality class.
- **T7.A.4 Fused top-10 expert gather-matvec** (one launch, precomputed routing) —
  MegaMoE-class backends with CUDA-graph replay exist on SM120 [FlashInfer 0.7.2].
- **T7.A.5 cuDNN ≥9.27 for the MTP VERIFY pass only** (2-4 query tokens; FlashInfer 0.7.2
  auto-routes cuDNN precisely to cut verification latency). Note: sm_120 SDPA needs 9.27
  (NVRTC-only engines before); head_dim 256 on sm_120 unverified — test.

High-ceiling gambles: **G1 persistent warp-specialized step megakernel** (attention+GDN+MoE
under one launch/layer-group; doorbells become device-side flags; Fireworks KV-outer design
1.9-2.4x kernel-level is the reference) — THE post-200 play; **G2** port FlashInfer MSA/VSA
SM12x block-sparse kernels as the QSA decoder (only shipped native family; hostile contract:
plan() outside graphs, 64-block pages); **G4** L2 persistence window over the 36 GDN states
(~37MB — fits the 96MB L2; cheap, untested anywhere, derived not sourced).

Do-not-bother (hard evidence): **SM12x has NO tcgen05/TMEM/2-CTA MMA** (warp-level mma.sync
only — Colfax + FlashInfer corroborated); Machete (Hopper-only); DeepGEMM sm_120 (unmerged);
EXL3 decode (119 vs 155 tok/s at matched size on 5090 — prefill-only win); Marlin repack at
bs1-2 (+25% bytes, no benefit); cuDNN NSA components (SM100+-oriented, block-size mismatch).
**IQ3_S is byte-optimal at bs1-2**: every tensor-core format costs MORE bytes (NVFP4 4.25bpw
= 1.24x, Marlin 4.0-4.5bpw = 1.2-1.3x); on RTX PRO 6000 native NVFP4 grouped GEMM measured
39 tok/s vs Marlin W4A16 46-49. No published µs/token decode-attention numbers for the 5090
exist anywhere — our nsys timeline is the only source.

### T7.B — speculative decoding SOTA (~40 sources)

Sure wins:
- **T7.B.1 FastMTP-style MTP head retrain** [arXiv 2509.18362] — THE depth lever: vanilla MTP
  per-step acceptance collapses ~70%→~10% with depth (GLM-4.5-Air head: 0.92/0.68/0.38);
  inference-matched retraining recovers it (+82% over vanilla, 2.03x, lossless). Draft-only —
  output untouched. Strongest published support that our depth-4-6 cap is the HEAD, not theory.
  Requires a training run (our first).
- **T7.B.2 PLE/n-gram as a concurrent drafter lane** [TRT-LLM production]: +96% bs=1
  multi-turn, AL to 4.73 on translation; beats EAGLE outright when prompt-output BLEU>0.6.
  We already own the PLE tables; exact-match verify unaffected by draft source. Mimic lane.
- **T7.B.3 Adaptive draft depth** (AdaEDL entropy stopping, +10-57%; oracle +30%; TurboSpec
  goodput control). Graph-compatible via pad-to-max-depth + mask (SGLang #23330 bug class).
- **T7.B.4 Top-k tree on the MTP head at bs=1**: +5.7% measured on Qwen3-Next (chain-3 →
  steps4/topk4/draft8; τ 3.413→4.231). Lossless under exact-match.
- **T7.B.5 Draft-sharpening sweep 0.7-1.0** — NO published benchmark of coupled-RNG
  exact-match + sharpened draft exists; theory says our semantics escape the q=p optimum
  tradeoff. Our 0.85 is unexplored ground; sweep + entropy-conditioned sharpening.

**The stair-diagnostic answer: SGLang's slot-per-draft-token GDN state design** — draft tokens
write private state slots; acceptance promotes the last accepted slot by POINTER SWAP, no
replay; tree topk>1 applies each draft's update from its parent slot. Our ~8 replay-column
budget is a self-imposed tradeoff the published engines did not need. Design-against: vLLM's
bug class = state-slot bookkeeping under graph replay (repeated/skipped tokens; fix = sync
accepted-count copies before row moves).

Gambles: TTS online distillation of the drafter from verifier logits (+34-41% AL, proven on a
hybrid MoE; graph-replay sync barrier needed); DFlash block-diffusion drafter (2.4-3.5x vs
native MTP 1.7x on the SAME models; needs drafter training + KV-injection); LiLiCorr rerank
(+7-19% AL at 2.8% overhead, dense-only evidence); SpecExec mega-trees (offload regime).
Do-not-bother: LayerSkip retrofit (needs training-time early-exit), relaxed/judge acceptance
(violates MMLU gate), BlockVerify rule swap (+1.5-2.8% AL, redundant under exact-match),
Medusa retrofit (dominated). Paper ceiling: τ 4.2 (tree on native MTP) to τ 6-9 (DFlash-class)
→ ~270-400 tok/s single-stream.

### T7.C — quantization + memory/offload (~37 sources)

**HEADLINE VERDICT: nothing strictly dominates IQ3_S per-bit at batch-1 decode on sm_120.**
Our 180.9 t/s @93% hit is ~2.7x public SOTA on this exact model+GPU (llama.cpp #29887's 67.8
t/s best). Decode matvec at M=1 is bandwidth-bound; XFP (arXiv 2605.14844, RTX PRO 6000,
122B-A10B): ~3.97 effective bits beat Marlin INT4 by +49% purely on bytes-read at M=1.
5090 dense rates: INT8 = FP8 = 838 TOPS; only FP4 doubles again and only for GEMM shapes.

Sure wins:
- **T7.C.1 Admission control on cache fills + ONE packed H2D per expert** (gate+up+down
  contiguous): removes the novel-content cliff [llama.cpp #24528: ADMIT_AFTER=64]; packing
  two H2Ds into one removed 40,596 ops in traces. Persistent slot table + bookkeeping beats
  staging ("staging buffers are a no-op when made correct", #23170 post-mortem).
- **T7.C.2 Graph-captured DMA: fixed slot table + memcpy nodes + side-stream next-layer
  prefetch** [DuoServe arXiv 2509.07379: next-layer expert predictor 54-67% exact top-k,
  90-95% ≥1-hit; 1.78-7.04x decode on weaker HW]. Graphs +5-18% decode, reuse +13%.
- **T7.C.3 Hit rate 93→95-96%**: each +1pt ≈ −43% miss bytes at 7% miss. Our decayed-frequency
  policy IS the community's hypothesized winner; add explicit admission gating.
- **T7.C.4 ffn_down bit protection (quality insurance, NOT speed)**: super-weight sensitivity
  concentrates in ffn_down [arXiv 2505.02390: static Q2 −8.9% AIME, Q3 −1.8%, Q4 −0.7%].
  Mixed-bit ffn_down→IQ4_XS costs ~+5% expert bytes (−2-3% decode) and buys MMLU margin.
  Only if quality headroom is ever needed.
Gamble: **MXFP4 expert copy for PREFILL ONLY** (dual-format store in spare host RAM; PP
+26-31% on sm_120 via llama.cpp PR #17906 mxf4nvf4 path; tg untouched; NMSE fails strict
gates — PPL/AIME passed, gate carefully).
Do-not-bother: full 2-bit ternary experts (static Q2 −8.9%; incompatible with our MMLU gate
without QAT); FP4 tensor-core decode matvec (PR #17906 tg unchanged; XFP underperforms
Marlin on SM120); re-quantizing all experts to 4bpw+ (+25-30% bytes cuts the decode roofline
~20-25% — the opposite of 200 t/s); GDS (NVMe-only, disk tier −70% throughput); L2
persistence over EXPERTS (working set ≫96MB, ≤2-3% theoretical — reconcile with T7.A G4:
states-in-L2 is the plausible variant, experts-in-L2 is not); porting vLLM/SGLang offload
ideas (nothing merged; RFC-only).
Ceiling: bytes/token ≈2.9-3.4GB → roofline 450-520, kernel-efficiency-adjusted hard ceiling
~300-400, realistic 220-300 via 95-96% hit + full DMA overlap + MTP.

### T6 RE-RANK AFTER T7 (the unified stack for M1 > 200 stable)

The three agents converge: overhead-bound, formats already optimal, 200+ needs no quality
gates. Revised order (old T6 numbers in brackets):
1. **IndexShare-MTP indexer reuse** (T7.A.1) — NEW, cheap, shipped for our architecture.
2. **Wide-grid + occupancy pass** (T6.1/T6.2 = T7.A.2) — confirmed.
3. **GDN slot-promotion redesign** (T7.B) — replaces the replay-optimization direction of the
   stair diagnostic (T6.5): slot-per-draft-token + promote-on-accept; unlock depths 4-6.
4. **Hyper-connection kernel fusion** (T7.A.3) — +7.6%/+5.5% e2e measured on our model class.
5. **LANECELL re-arm tranche** (T6.3) — unchanged, cheapest measured %.
6. **Fused expert gather-matvec + packed single-H2D + graph-captured DMA slots** (T7.A.4 +
   T7.C.1/2) — the launch/overhead attack; composes with hit-rate push 93→95-96 (T7.C.3).
7. **Shared-expert fusion** (T6.4) — unchanged.
8. **Adaptive depth + tree topk + sharpening sweep** (T7.B.3/4/5) — after 3 lands.
9. **FastMTP head retrain** (T7.B.1) — biggest single lever but our first training run;
   scope it as its own project decision.
10. **Boot-variance control** (T6.9) — unchanged, holds the floor.
Post-200 ceiling play: **persistent step megakernel** (T7.A G1) — the ~2x; our doorbell/ring
is exactly the machinery it would replace with device-side scheduling.
Quality-insurance sidelines (not speed): ffn_down bit protection (T7.C.4), reasoning-budget
steering + think-strip KV (T4.1/4.2), MXFP4-prefill-only (T7.C gamble).

## T6 — M1 > 200 t/s stable: ranked move stack (2026-10-16, recon review)

Gap math: today's 3-round mean 180.9 → 200 needs +10.6%; standing band top 191.6 → +4.4%.
"Stable" = floor above 200, so the target mean is ~205-210. Ranked by value x certainty / effort;

all evidence citations inline (our F-entries or T1.x above).

1. **T1.2a pre75-on-sm120 + row-fold** — F-208: sm_120 runs the OLDEST attention kernel
   lineage (PR#540 gated cc<75 at qsa_decode_attn.cu:291/:609); upstream PR #2075 is the
   cherry-pick path; our qsa kernel is byte-identical to upstream so the port is low-friction.
   The single hottest decode path. Test their branch, measure, adopt, comment our numbers on
   the PR. Est +5-15%.
2. **T1.4 hc wide-grid** — designed bit-identical (lane chunk order preserved), 324 blocks vs
   41, ~30 lines, pure occupancy fix on the same decode window. Bit-identity class (canary
   boot only). Est +3-8%, stacks with 1 on the same path.
3. **Re-arm tranche, LANECELL first** — F-200 void boundary corrected; original tranche reads
   +3.6/+5.4/+1.9 (LANECELL), then GR_V3 / L2-72 / HC_SPLIT / MMQ_KQ on bare base. Env-flip
   only, one ABAB battery each, OFF one command away. Cheapest measured % on the board.
   Est +2-4% net after overlap.
4. **Shared-expert fusion (T1.1 above)** — +5.9% measured on IQ3_S (their tree), identical
   output; our path provably launches separately (layer.hpp:150). Est +4-6%.
5. **Stair diagnostic → deeper spec** — depths 4-6 never offered (Lane B law: precedes any
   spec-8 re-test). tqllm lead: their kGdnReplayCols=8 caps combined draft+lookup depth on
   hybrid GDN models (src/tree_columns.hpp) — audit our replay-column cap. If depths 5-6
   become offerable at >=70% acceptance, spec-6 -> spec-8 is the only lever that multiplies
   tokens/window instead of shaving ms. Est +8-15% IF unlocked; diagnostic is cheap.
6. **T1.1 doorbell partial plan** — publish-side pre-verified; impl plan written
   (campaign/research/T1.1-partial-plan-impl-plan.md, bit-identical required). Est +1-3% M1;
   also groundwork for batch G=2 / service-thread rewrite (conc's big lever).
7. **Draft-only reduced head (T1.6 clean subset)** — exact by construction. Est +2-5%.
8. **SPEC_DRAFT_TEMP micro-sweep** — standing 0.85 = peak of the 0.45-0.90 sweep; try
   0.825/0.875, one battery. Est +0-1%.
9. **Boot-variance control** — today's boots: 187.8/183.4/171.6 (boot 3 sagged ~9%).
   Ensure boots load the learned expert profile (profile-save-every 30 writes it), add a
   fixed pre-warm sequence to the battery driver. This is what converts "peaks near 200"
   into "floor above 200".

Stack math: (1+2) +8-14% combined, x (3) +3%, x (4) +5% => mean ~211-222; item 5 pushes
further; item 9 holds the floor. Sequence: 1 -> 2 (one code lane), 3 in parallel (env-only,
tonight-class), then 4; diagnostic 5 anytime.

Recon additions (2026-10-, from T3 llama.cpp agent — both measured on our model class, both
slot into the same MTP window as items 1-2):
- **T6.10 GDN snapshot-write fusion (T3.1, #29807)**: fuse our GDN recurrent-state snapshot
  writes into the scan kernel for the speculative path — measured +3.5% decode-under-MTP,
  acceptance unchanged, composes with CUDA graphs. Small kernel change; pairs with the
  attention lane.
- **T6.11 MTP catch-up crop (T3.2, #29928)**: gather rows so catch-up/zero-output forwards
  skip FFN+head — measured −58% catch-up kernel time; `n_outputs` joins the graph key.
  Attacks the same tail our doorbell work targets, at a fraction of the code.


## T8 — GOAL ROUND "NEW CEILING" ledger (2026-10-16, operator directive)

Standing goal: significant, simultaneous, stable improvement on M1/pair/conc + smoothness +
quality; evaluate capped logic and build new ceilings. Session results:
- F-214: mtp-window + IndexShare adaptation CLOSED (mechanisms banked); LANECELL re-arm
  confirmed conc-favored. Battery-env-inheritance law.
- F-215: LANECELL conc profile LOCKED (launch-custom-iq3s-8081-conc.sh). Scoreboard: all three
  targets touched (M1 208.3 / pair 194.4 / conc 244.8).
- F-216: GDN slot-promotion BUILT + bit-parity PROVEN (instrument kept); replay is NOT the
  depth gate — F-71 attribution corrected.
- F-217: hc wide-grid -9.2% (SMEM tile = load-bearing xn reuse), occupancy probe neutral;
  F-208 cost model falsified -> profile-first law.
- F-218: FRESH COST MAP (STRATA_VERIFY_PROFILE): light 13.35ms (VRAM hits 1.93 / hc 2.39 /
  copy+combine 1.08 / attn 0.93) vs heavy 24.31ms (**attention 4.09** / VRAM 3.46 / hc 4.00).

PRIORITY ORDER per F-218 (supersedes the T6 re-rank):
1. T1.2a pre75-on-sm120 via upstream #2075 + row-fold — lands on the 4.09ms heavy-class
   attention line. Est -1.5-2.5ms heavy windows.
2. Fused expert gather-matvec + packed single-H2D + admission (T3.3/T7.C) — the 1.9-3.5ms
   VRAM-hits line, biggest stable across classes.
3. HC Mix/Combine kernel FUSION (SGLang-class, +7.6/+5.5% e2e on our model class) — the
   2.4-4.0ms hc line; grid/occupancy already falsified (F-217).
4. T1.1 doorbell partial (publish-side plan) — the 1.1-1.4ms copy+combine neighborhood.
5. Variance/smoothness control (boot spread 171-208; agent T8.B researching knobs).
Background: 3 research agents out (T8.A megakernel/launch-overhead frontier, T8.B serving
smoothness/jitter science, T8.C sibling-fork + discovery mining) — fold on return.

### T8.C — discovery-mining report folded (agent, 2026-10-16; upstream tracker wave + fork census)

UPSTREAM: main FROZEN at 0.1.44.1 (296b59d) — but 12 PRs/~9 issues landed in the tracker
2026-10-15/16. The ones we need are UNMERGED: we port them ourselves.
- **#2758 CORRECTNESS — PORTED THIS SESSION (F-219)**: record_window's self-commit carries
  !g_qfuse() but Verifier::commit's one-token skip didn't → under STRATA_QFUSE=1 GDN state
  advanced nowhere on one-token rounds (silent wrong tokens). Our tree HAD it (verify.cpp:884
  vs :2705); one-line fix applied, boot-validated 186.2/77.1.
- **#2612** (5090 D calib: 140-162 tok/s IQ3_S 500K): vram-reserve 12288→1500 MiB tripled the
  auto expert cache → +30% same-prompt decode. We already run --vram-reserve-mib 2000 (the
  good regime) — validates our setting; our 180-208 = top of public record for 5090+IQ3_S.
- **#2756** STRATA_PF_ATTN_BATCH 32→128 QSA prefill attention (SYCL, bitwise via FP32-hash
  parity, ~9% QSA prompt-attn time) — port candidate for our prefill lane.
- **#2746** Q2_0 packed-word decode (Battlemage: 41-45% kernel, all 65,536 words
  byte-identical, +4.2-4.5% TPOT) — adjacent to our type-42 Q2_0 64/18 dialect; adapt.
- **#2443** conversation-cache eviction redesign (revisit prefill 1778→90 tokens, sm_120-
  validated) — if we hit parking refusals.
- **#2458/#2434** LRFU beats CLOCK 3-4% of misses (real traces) + alloc-inside-capture
  poisons-the-stream crash pattern + --dump-kv-select tool.
- **#1242** capture-lifetime bugs (image-position table overwritten under a live captured
  graph; draft-head type -1 crash on first batch-MTP admission) — audit patterns.
- **#2745** NVFP4+layer-split peer-prefill tail scales → non-finite logits (landmine only if
  we adopt NVFP4 packs + splits; STRATA_PREFILL_HELP=0 guard).
- Task-aware spec: code +91% / prose −29% at draft-6 (r/unsloth); SGLang accept ~3.3 —
  our --spec defaults should be task-aware for the mimic/prose lanes.
FORKS: **eddoursul/Strata (custom branch, 539 ahead — the densest portable source; measured
1.5-1.7x vs upstream on 3090+5070Ti)**: k8v4 hybrid KV (INT8 K + Hadamard-rotated Q4_0 V,
qsa_prompt_attn mode 3), HC fusions F-1/F-2 (882bb6d/b046845 — REFERENCE for our #3 priority
lane), expert grouping tables via mapped pinned memory OFF the copy engine (fe609ce), sm_75
gate relaxation reference (aad5bb1). jmnargi/Strata-V100: sm_70 + MTP + WMMA (gate can go
lower than we thought), L2/L3 conversation parking. MistVVK/XeStrata: gemm_rows grid-dim-2
hazard (not in our tree — checked). lighttransport glm53f: GSQ/RCO/GyRot/trellis quant
research note (1abf442) + llvq ≈ IQ2_XS at 2.06bpw. tqllm fork xhinker/azllm active 10-16
(MTP experiments — later pass). MODEL CARD verify-list: indexer budget 512 blocks/2048 tokens,
rotary dim 64, n-gram table at layer 2, 24Q/2KV heads hd256 — audit our constants.
PROFILING recipes: nsys --cuda-graph-trace=node + nvtx-precapture (names phases inside
startup-captured graphs), vLLM phase-data attribution pattern, cudaProfilerStart capture
ranges; SGLang #5593: ~2ms idle bubble before first kernel of graph replays — look for it in
our traces. Not merged/watchlist: #2453 (Windows config-side regression — binary A/B blind),
#2010/#2038 as rebase regression tests.

### T8.B — smoothness/variance report folded (agent, 2026-10-16) — knob #1 EXECUTED

MECHANISM SETTLED: the morning-evening gap (181 vs 196-208) is HOST-SIDE page-cache warm-in of
the mmap'd ~85GB, NOT GPU thermal — a warm GPU reading faster contradicts throttling. The
5090's SM clock does bounce 260-460 MHz on sub-ms bursts (cloudrift measured 84% per-call
variance at 8us kernels; boost never engages for bursty decode), so clock-locking is the GPU-
side variance clip. Stats: at our ±5-8% boot CV a naive A/B needs ~49 boots/arm to see +4%;
our ABAB same-window interleave (paired, sigma_d 2-3%) needs 2-5 rounds — the existing law is
quantitatively vindicated; CUPED on a prior-round covariate would cut unpaired needs ~2.5x.
EXECUTED (no sudo needed): campaign/prewarm.sh — full sequential read of pack+shards into the
page cache before every battery boot (7s warm, ~15-30s post-reboot), wired into battery-boot
and battery-config. PENDING USER SUDO (paste block in F-220 journal): clock lock ~2700 MHz +
memory clocks at max + verify. The remaining knob table (pinning/mlock under a systemd
LimitMEMLOCK override only if pressure ever appears; CPU pinning to the V-Cache CCD cores 0-7
+ EPP performance; THP=skip; power limit=skip +0.1%) ranked in the agent report. Do-not-
bother: power-limit raise, GDDR7 thermal intervention, p99 from short rounds (2000-token
rounds give p99 CI ±44% — aggregate instead). Watch: GB202 Xid 79/GSP cluster under sustained
CUDA load — check `journalctl -k | grep -i xid`; PCIe ASPM off mitigates if we're affected.

### T8.A — megakernel/launch-overhead report folded (agent, 2026-10-16) — PDL is the next structural lane

CONVERGENT EVIDENCE: four independent measurements agree the 2x is structural (pegainfer 11.4%
launch gaps pre-graph; NVIDIA llama.cpp proposal 40% graph-time but +10-15% e2e; Baidu 14.6%
launch overhead; Hazy: engines use ~50% of achievable BW at bs1, megakernel 78%).
RANKED (the agent's swap table, adapted to our post-F-221 state):
1. **PDL (Programmatic Dependent Launch) inside the step graphs** — sm_90+ INCLUDING sm_120
   (llama.cpp ships it default-on, +5-22% tg on Blackwell-class; TRT-LLM default-on; works in
   captured graphs via cudaGraphDependencyTypeProgrammatic edges or post-capture attribute
   rewrite — llama.cpp's approach). DISCIPLINE: parity-gate every kernel vs PDL-off (TRT-LLM
   hit real NaN/accuracy regressions in quant kernels — our class); use the CUDA intrinsics
   (cudaGridDependencySynchronize / cudaTriggerProgrammaticLaunchCompletion), NEVER raw
   inline PTX (FlashInfer #2558 UB, 14-file migration); under exact-match verify a race shows
   as acceptance drift — the M3b canary is the tripwire. PDL cannot cross two graph launches
   (intra-graph edges only).
2. **Doorbell collapse** — publish a 64-128B line (indices + layer/step id) instead of ~80KB
   of mapped f32 per miss layer; bulk metadata in a device ring the host reads via gdrcopy or
   one DtoH per step. "The doorbell design, not the DMA, is the anomalous part vs SOTA" — no
   published hybrid-MoE engine puts per-layer host signaling on the critical path (Fiddler,
   Pre-gated, DuoServe, KTransformers all ship activations/prefetch/deferral instead).
   Directly attacks the conc lane's 19.74ms GPU-reach wait (batch accounting line).
3. **Prefetch across the MTP draft horizon** — mispredicted expert prefetch is SAFE under
   exact-match verify (costs a re-fetch, never correctness). DuoServe-class: next-layer
   predictor 54-67% exact top-k, 90-95% >=1-hit.
4. **Capture hygiene** — we hold the F-91 law; add cudaGraphUpload (first-launch upload
   penalty 28% class on a fresh graph) and alloc/free-per-step audits (pegainfer found
   292/step at 5.4us).
Gambles: conditional nodes (CUDA 12.4/12.8 IF/WHILE/ELSE) + device graph launch (>2x cheaper
than host launch, structure-independent — but VERIFY 13.4's device-graph memcpy restrictions
before building the expert-service sub-graph on it); MPK-lite persistent fusion of the
per-layer control chain. Do-not-bother: full megakernel (no published data-dependent-MoE-
with-offload instantiation — ETC needed a compiler), CDP (measured worse), green contexts
(5090 power wall reported).
METHODOLOGY WARNING (SGLang #5593): the ~2ms "bubble between graph replays" was profiler
inflation (NVTX/nsys) — re-profile with wall-minus-event accounting, NVTX disabled.

## T9 — third research round (2026-10-16, targets: single 250 STABLE / conc 300; all 3 agents folded same day)

Dispatch: A=Blackwell/CUDA runtime+kernels, B=arXiv/academic frontier, C=serving-systems/OS.
Stack facts given: post-F-221 band M1 ~192-200 / pair ~182-186 / conc ~226-233; prof221 map
(attention NO LONGER the top kernel line; GPU-reach wait 10.99/17.78ms = dominant line).

### T9.A — Blackwell/CUDA (9 items + 11 traps)
1. **PDL — CLOSED BY F-191 (agent finding CORRECTED)**: the agent reported PDL "half-wired"
   (launch attr only, no intrinsics) — WRONG on our tree: pdl.hpp carries the full discipline
   (pdl_wait/pdl_trigger via griddepcontrol PTX = the intrinsic semantics, STRATA_PDL_RESTRICT,
   kernel-only capture edges via pdl_launch_ok, parity harness) and F-73 (Oct 12) ADOPTED it,
   F-107 RETRACTED it on Windows (WDDM −40%), and F-191 (Oct 14, Linux, drift-immune
   immediate-toggle ABAB) measured it a CLEAN NO-OP (ctl2 ≈ pdl on every metric; launch spacing
   already hidden behind the host round-trip). Default-off = no live hazard. Re-test ONLY if the
   window structure changes (post-pipelining/G2). The T8.A "PDL rollout" queue item is hereby
   marked superseded by F-191.
2. **cp.async.bulk.prefetch.L2** (CC>=9.0, no SMEM/mbarrier/completion = graph-safe):
   vLLM-decode-attn paper arXiv 2504.06319 — L2 hit 0.06%->43-83%, kernel 1.84-2.15x; trap:
   regresses 2-5% at high GQA (check KV-head geometry); grows with context (top attention
   lever at 32k+); ALSO the cheapest mechanism for the banked draft-horizon expert prefetch
   (wrong guess costs only L2 capacity).
3. **Green contexts — runtime API in CUDA 13.1** (was T8 do-not-bother; new evidence
   outranks): SM/queue partition per stream; 5070Ti naive 2-stream +3% vs GC +60% (mixed
   compute+memory), H200 +34%, NVIDIA example 1.05ms->95us. Traps: cross-stream alloc Xids,
   clusters may exceed split, L2 NOT partitioned, static split idles SMs — keep a full-SM
   stream for pure phases; 5090 power-wall caution from T8 stands (cap-aware). Conc +5-15 +
   the stability lever for 300.
4. **L2 persistence retarget**: weight persistence dead (85GB vs 96MB); pin the RECYCLED
   set (KV tail, router logits, doorbell/commit flags, GR staging) + evict_first hints on
   weight streams (pegainfer ~10% pollution penalty recovered). hitRatio is probabilistic
   (window/set-aside); window must arm on the CONSUMING stream (PyTorch #101489 trap);
   measure via lts__t_sector hit-rate, not wall time.
5. Miss-path ping-pong pinned staging + copy-engine overlap of next-layer misses with
   current-layer expert compute (adjacent to packed single-H2D; gate on miss rate; small
   H2D lands ~24.5GB/s on 5090 — why packing matters).
6. Clocks/power recipe for STABLE: 575->480-500W cap ≈ negligible decode cost, kills boost
   oscillation; GDDR7 OC = SILENT CORRUPTION (link ECC gap — on-die ECC doesn't cover the
   link) — hard ban; -lgc for A/B honesty (matches staged F-220 sudo); validate the FE
   throttling defect once via clocks_event_reasons.
7-8. Conditional nodes + device graph launch (limits: <=120 fire-and-forget/parent, 1
   pending self-launch, body graphs can't device-launch, no host updates while
   device-launched — prototype-gated C); CUDA 13 device-updatable graph nodes (thin docs, C).
9. Profiling fixes: ncu default LOCKS CLOCKS + FLUSHES CACHES (--clock-control none + -lgc,
   --cache-control none for L2 work); CUPTI ~2-4%; ncu 2025.1+ required for sm_120.
TRAPS: TMA for decode GEMV usually a loss (m=1-2; TMA's decode relevance = the prefetch
form + 1D bulk staging at long ctx); clusters cap 8 portable (16 = B200 only); cuBLASLt
grouped GEMM skip (can't consume IQ3_S; revisit only dense/shared large-m); zero-copy
mapped expert reads DON'T (SM-load serialization over PCIe); B200/tcgen05/TMEM results do
NOT transfer to sm_120 (128KB smem, 48-warp cap).

### T9.B — arXiv/academic frontier (10 items + traps)
1. **EcoSpec expert-cost-aware tree filling** (arXiv 2607.12696): score nodes S=P/ΔCost
   where ΔCost = NEW experts forced into HBM, with a coverage buffer; 1.36x vs EAGLE-3's
   1.22x on Qwen3-235B from node selection ALONE, acceptance unchanged; pair with the
   marginal-cost stopping rule proven in "Bounding SD in MoE" (2609.22156), OPT-Tree
   (2406.17276)/Sequoia (2402.12374) topology, AdaEDL (2410.18351) entropy depth. Est single
   +12-20 (THE 250 lever), conc +8-12. Lands: draft_policy.cpp; our router gates are already
   materialized → predictor ~0ms. Exactness-safe (candidates only, verifier untouched).
2. Concurrency goodput governor: optimal horizon FALLS as GPU saturates (Su 2310.18813;
   TurboSpec 2406.14066; Nightjar 2512.22420 +14.76%/−20.18%; TETRIS ACL25). Conc +10-18.
3. Expert residency overhaul: MoEpic VERTICAL expert split (2509.08342 — cache the hot top
   segment of experts, ~2x experts per VRAM budget, 37.5-65.7% latency cut), FluxMoE
   budget-aware planner (2604.02715), ProMoE proactive prefetch (2410.22134, 2.07x). Conc
   +15-30 if miss rate >15%; single +5-12 + flatter tail.
4. QuantSpec dual-view KV (2502.10424 ICML25): low-bit VIEW for drafting, k8v4 verify
   untouched (bit-exact); >90% acceptance kept; single +5-10 @8-32k, +10-20 long-ctx.
5. Block verification (2403.10444 ICLR25, provably greedy-safe) + GBV multi-path (2602.16961,
   >30% block-efficiency, >15% walltime) — TOUCHES the verify contract → needs owner sign-off
   + differential test; single +8-15.
6. HOBBIT draft-only shadow experts (2411.01433): low-bit substitutes for DRAFT windows only;
   +3-8 + variance-flat; +~20GB host.
7. Fiddler (2402.07033) / KTransformers (SOSP25) deferred CPU experts: conc +10-25 when
   miss-limited; CPU-vs-GPU bits differ → draft/lookup windows ONLY (verify misses must stay
   GPU or the contract breaks).
8. FlashDecoding++ unified-max async softmax (2311.01282): +2-6% @32k+, ~0 short; keep
   online-softmax fallback with a deterministic trigger.
9. Long-ctx KV paging/compressed-draft (PiKV 2508.06526, LongSpec 2502.17421, MagicDec
   2408.11049, LMCache): 262k/1M tier; verify windows must stay resident.
10. Expert merge/distillation (UniMoMo 2608.08627: ~99.9-102% quality at 4/grp, 1.28-1.63x;
    Sub-MoE AAAI26): rewrite-tier ceiling (misses vanish if 512->256 fits 32GB) — sequence
    after #3 quantifies residual miss rate.
TRAPS: MoE-Spec budgeting is LOSSY (substitution + renormalized gates — off-contract; EcoSpec
is the lossless route); ProMoE's cache-aware ROUTING changes outputs (prediction only);
depth-16 @conc-32 collapsed 2486->607 (plateau 3-4 accepted); EqSpec 2510.22876: most batched
spec impls silently corrupt (position/KV misalignment) AND vLLM/SGLang exact-match ≈ 0 on
Qwen3 under batching — OUR exact-match verify is the differentiator, never adopt batch-expansion
verify; sub-4-bit KV drifts on verify (draft-view only); draft-free methods underdeliver vs
trained MTP (FastMTP retrain stays the right expensive bet; AdaSPEC NeurIPS25 = cheapest KD).

### T9.C — serving-systems/OS (10 items + traps)
1. Batch-aware spec depth (SGLang MTP blog: draft_token_num=2 under load; Reducto: MTP LOSES
   at 64-80 conc, target forward 3x vs draft 42%; plateau 3-4 accepted): conc +10-20. Same
   lever as T9.B#2 — merged there.
2. **One-step-ahead host pipelining** (SGLang v0.4 zero-overhead scheduler: future tokens +
   event scheduling, zero GPU idle, overlap alone +20.4%; vLLM V1 same pattern): the service
   thread processes window N-1's doorbell while the GPU runs N. Conc +10-20, single +5-12.
   The intermediate step between T1.1 and the multi-thread rewrite — keeps FIFO commit.
3. Device-resident continuation: conditional nodes + device graph launch as the
   beyond-doorbell endgame (control-flow round-trips gone): conc +10-25; C until a 20-line
   sm_120/CUDA-13.4 prototype verifies support + the body-graph limits.
4. Graph batch-size ladder science (TRT-LLM blog20, measured): default x2 ladder wastes up to
   ~50% padding mid-gap; +64 ladder up to 1.3x; +8 REGRESSED DeepSeek-V3.2 5% (graphs ate
   17.2% of KV); ~10MB metadata/graph; capture scales linearly. For us: G∈{1,2,3,4,6} sized
   to 2-4 lanes, measure padding waste first, watch expert-tier VRAM competition. A-.
5. CCD pinning (X3D CCD for service thread + driver submission; other CCD for IRQs): eunomia
   eBPF study — clean host 1.2% cost, contended 8.8-20.5%; taskset+nice recovered +7.6%/96%
   fewer switches. Variance lever for STABLE 250; near-zero mean gain on our quiet box.
6. Clock lock (-lgc/-lmc): 50-series multi-second spikes resolved by -lgc (NVIDIA forum) —
   the staged F-220 sudo; lock during A/Bs for honesty.
7. Pack off NTFS3 -> ext4/xfs (Phoronix roundups): load/warmup −20-80%, steady ~0-3; mlock/
   MAP_POPULATE the hot host tier (RLIMIT_MEMLOCK). THP=always is NOT a tg win (llama.cpp
   #2251); hugetlbfs "10x" is startup-only (#12444).
8. Prefix-sharing KV cache (RadixAttention 2312.07104, ~6x shared-prefix; block-hash APC):
   0-40% workload-dependent (battery prompts share prefixes); COW discipline + a verify-mode
   that recomputes one lane's prefix and diffs.
9. Same-length window composition (EXSpec 2510.22876: group same-length lanes, δ=0, avoids
   the ~40%-of-compute repad cycle at batch 8): conc +3-10; trap: P90/P99 1.5-7.7x worse if
   lanes wait for partners — cap the queue delay.
10. Draft/verify overlap on side streams: P-EAGLE (vLLM 0.16 PR#32887, 1.55-1.69x) needs a
    TRAINED mask head (C for our fixed MTP head — check checkpoint first); portable variant =
    df_branch_ overlap of window N+1's draft with N's verify (single +5-15; tree acceptance
    is batch-invariant per 2609.24698, throughput peaks near bs≈4 — matches our 2-4 lanes).

### T9 merged queue (updated master priorities)
1. T1.1 partial device plan — CLOSED NEUTRAL (F-222: flagA round-trip worth only 0.26ms;
   the tail = service-thread pool compute 4.5ms additive + GPU serial execution 16.4ms) →
   replaced by: T1.1-v2 device-side PCIe share for partial groups (kills the 4.5ms stage;
   plumbing landed, env-gated STRATA_PLAN_PARTIAL) + one-step-ahead pipelining (next).
2. One-step-ahead service-thread pipelining (T9.C#2) — doorbell successor, pairs with T1.1.
3. EcoSpec tree filling + marginal-cost rule (T9.B#1) — the single-250 lever.
4. Batch-aware depth governor (T9.B#2 = T9.C#1 merged).
5. cp.async.bulk.prefetch.L2 (T9.A#2) — 32k+ attention + expert-prefetch mechanism.
6. Green contexts (T9.A#3; re-test vs T8's power-wall caution) — conc stability.
7. Residency overhaul MoEpic/FluxMoE/ProMoE + miss-path overlap (T9.B#3 + T9.A#5).
8. Graph ladder G∈{1..6} (T9.C#4, TRT-LLM blog20 numbers).
9. Clock lock (user sudo staged) + CCD pinning (T9.A#6 + T9.C#5) — the STABLE half of 250.
10. QuantSpec dual-view KV (T9.B#4); block verification (T9.B#5, contract sign-off gate).

## T10 — eddoursel fork overlap audit (2026-10-16, background agent, full absence-checks run)

Base: eddo/custom vs our main (367-file delta). F-209 law applied — 30+ mechanisms verified
ALREADY SHARED (F-1/F-2, HC_Q8, base elastic K/V, commit batch, MMQ_IL, foresight, k8v4, rope
scaling, ...); N/A list documented (multi-GPU/Metal/SYCL/NVFP4/UD formats). OURS-ONLY list
noted (--prefill-head-alias, --vram-elastic, --pool-affinity, coupled-draft, KV_PREFETCH ...).
RANKED PORT CANDIDATES (their hash + measured value + port cost):
1. **674bf0ca sub-warp grouped expert kernels** (iq_dot.cuh load/fetch/apply + native_gus80
   half-warp-per-column gate/up for n_embd 2560 = OUR SHAPE + native_down_sub): IQ3_S decode
   +5.9% code/+5.7% think/+6.4% after-16K; experts 36.7->32.7us @1tok, 75.3->56.1 @4,
   133.9->94.3 @8 (3090). ATTACKS THE PROF221 VRAM-HITS LINE (biggest stable kernel line).
   Port: MEDIUM extract-and-adapt (their iq_kernels diverged ~4k lines; parity harness comes).
2. **2ae47d0f draft-head 2-bit proxy** (draft_proxy.cu +439, mtp.cpp hook, STRATA_DRAFT_PROXY):
   40,525-row head -> 32.4MB 2-bit proxy; 110->51us/draft; round 1.49->1.31ms; bitwise logits,
   13/4568 min-p flips (within 0.048). **RE-PRICED 2026-10-16 during integration design: our
   standing draft mode is the F-184 COUPLED sampler (full-vocab logits + target Philox +
   penalties) — the proxy's top-1+prob only fits the non-coupled greedy path; a faithful port
   = top-K exact rescore emission + penalty-aware fallback to the full head = day-class
   surgery for ~+2-3% (head line 0.43-0.78 ms/window halved). Queue AFTER the sub-warp
   kernels; kernel read complete, files extracted /tmp/dproxy; Q6_K type 14 + n_in 2560
   preconditions CONFIRMED for our dhead_.** Sub-warp port plan (T10 #1) + this note banked
   at campaign/research/SUBWARP-EXPERT-PORT-PLAN.md.
3. 50c83e0f+236c73ba prefill GDN mma chunk-walk (one fused kernel, WY-form, m16n8k16):
   recurrence 1.30->0.61ms per 2048-tok layer (2.1x, 3090); NOT bitwise (~4e-4) — prefill lane;
   port MEDIUM.
4. 78652d23+e6ed2a50 elastic-EXTENDED K/V (pooled keys + rope tables grow): +165 expert slots
   (+1.8% cache) at our 131072 context, more at 262k+; port MEDIUM (fit to our kvg_ensure).
5. c7544ef6+4a9669ff QSA top-k past 262,144 cells (dynamic smem by capacity; 1.47M cells on a
   3090, more on 5090): the 262k/1M enabler; selections bit-equal; EASY-MEDIUM.
6. 3443d955 MMQ dense projections at 7-8 token windows (+5.7% IQ3_S @8-tok): **APPLIES TO OUR
   M1 SERIAL WINDOWS — today's boots capture 6/7/8-token windows (confirmed in the logs);
   this is a real M1 lane, not just lookup-chain territory; MEDIUM.**
7. 240c8be7+06dfaa57 --feed-max (short prompt parts via verify windows): interactive-latency
   floor; MEDIUM (adapt to our serve loop); pairs with ac21812c preemption fix (45 lines).
8. 9488909a post-prompt tier refill seeded by PROMPT routing counts (batched lent-slot refill,
   1.3x-gap admission): prompt -2.8/3.5%, post-prompt answers +0.5-2.1%; the "prompt counts
   into usage" half ports as an IDEA first; skip their --adapt-decay (noise).
9. ffb5d6f7 --arena-place by measured RAM read rate: +5.4-6.7% UD only, IQ3_S +-0.1 on THEIR
   asymmetric DDR5 — measure ours first (STRATA_ARENA_PLACE_DRY); likely skip.
10. b2141a7c --expert-math exact (bit-exact GPU-vs-CPU-pool expert math incl shared expf):
    quality/measurement instrument; wide but shallow.
11. Load lane (LOAD_THREADS/PIN_AFTER_COPY/PLE_RAM_NOWAIT): startup-time only; low priority.

## T11 — fifth research round (2026-10-16 evening; 5 agents: R1 lineage, R2 pipelined-loop dossier, R3 FastMTP, R4 CPU pool, R5 quality gate)

### T11.R1 — lineage mine: ALL SOURCES CLEARED + 3 new portables (agent report folded verbatim-in-summary)
CLEARANCE: upstream/main FULLY merged (0 unmerged; BOOKKEEPING CORRECTION: d7a2453 was OUR
merge commit — upstream's real 0.1.44.1 tag = 296b59d2 and our HEAD already carries ~15
commits past it: #2324 AVX fixes, #2323 Metal, reasoning_effort etc. — we are AHEAD of
upstream main entirely); eddo/upstream == upstream/main; pr2075 fully ported (our side a
superset); rc/0.1.30 + 51 tags contained. Zero unmerged engine commits anywhere fetched.
NEW (verified absent, not in T10):
1. **a1444e33 prefill DENSE MMQ (run_dense)** — T10 mis-filed this as already-shared (we have
   the moe_mmq MODULE incl quantize_mmq_q8_1_cuda, NOT the prefill dense wiring; run_dense =
   0 hits our side). GDN qkv/gate/out + QSA q/out + shared-expert down through dense MMQ on
   GGUF blocks, q8_1 activations, small 512/640-row shapes stay FP16 GEMM; GOTCHA: zeroed
   4KiB tail for non-256-multiple rows (NaN×0 poisoned prompts). MEASURED: IQ3_S 2K prefill
   −27..−30%, 16K −16%, follow-up turns −6.5..−22.7%, KL +1.8e-4 noise, argmax agreement
   unchanged. PREFILL LANE #1 (our first-listed priority). Port MEDIUM (~130 lines + 5 sites
   + tail-zeroing).
2. **2d6b1db8 PLE key/value projections BESIDE LAYER 0** (from jesdga95/basalt) — today they
   run on the critical path at layer 1 (our verify.cpp `l == 1 && ple_on` ~:1032); the fix
   branches them at the window's start, waits the PLE flag, runs both beside layer 0, layer 1
   joins. IQ3_S code +4.1% (thinking −0.2). Decode lane, EASY-MEDIUM (BF16-key path only).
3. **95979cf2 profile-ordered bounded expert copy-ahead in prefill** (~1ms/32MiB pieces, ≤2
   queued, until the layer's routing is host-side): single-GPU 3090 IQ3_S 2K −8.9% / 16K
   −9.9%. MEDIUM-HIGH idea-level adaptation to our ring.
Plus idea-level: tier moves admitted up to 1.3× the measured window gap (0.49→0.12ms commit,
pairs with the pipelining lane); TIER LOST-WAKEUP BUG-CLASS audit for OUR expert_cache
retire path (their 2056b4b4 froze at 35% hits — discriminator audit, not a port);
instruments: STRATA_TIER_TRACE, per-request tier counters in /metrics, --deterministic-adapt.

### T11.R3 — FastMTP retrain feasibility: VERDICT GO (full dossier in agent report)
Recipe: FastMTP recursive teacher-forced CE (arXiv 2509.18362; Red Hat Speculators 0.6.0 =
production form, Qwen3-Next+3.5 MoE SUPPORTED = our head class). Loss weights β^(k-1)
(β=0.6 → [.51,.31,.18]; OUR adaptation: β≈0.8 for the pos-2-6 deficit, K=6-8 = serve depth).
Data: SELF-DISTILLED 5k conversations suffice (RH: 5k samples, 443s on 2×H200 for 9B; our
head-only ≈ 8-14GB VRAM, 5090 fits; ~2-6h/arm; frozen embed + BOUND lm_head — mtp.bind ties
the backbone head ✓). Harvest: replay our MIMIC CORPUS through Strata prefill-only with a
hidden-state dump at the MTP input (~2-6h, 160-320GB bf16 fits). Ship: bf16 → DIRECT-CAST Q4
(never GPTQ — F-201 law). DECISION MATH (calibrated to our 2.78 banked/13.2ms ≈ 205-210):
RH-scale deltas → ~227; FastMTP-scale → ~237; + depth re-unlock (pos-4-6 conditional
products lift) → 247-255 = THE 250 PATH; 250 needs AL ≈ 3.25-3.4 (banked +0.55-0.7/window).
Gates: dry-run on 9B sibling first (pipeline de-risk, 15-30min on 5090); v1 gate pos-2/3
+≥8pp each & pos-0 ≤1pp regression & e2e ≥+8%; kill-rule depth extension only while marginal
acceptance > 6-9% breakeven. COST: 1-2 days engineering + a weekend of compute. NO CLOUD.
BONUS ADVANTAGE: train on the QUANTIZED backbone's hiddens (distribution-matched by
construction — RH trains bf16-serves-bf16; bounded by their deltas, not above).

### T11.R2 — the pipelined service loop DESIGN DOSSIER (conc-300's roadmap; full dossier in agent report)
THE DECISIVE INSIGHT (matches F-223's falsifications): inside one verify window the residual
chain is strictly sequential — NO intra-window reordering can hide the pool. Every engine
that succeeded (SGLang v0.4 zero-overhead, vLLM V1 async, TRT-LLM overlap) keeps ONE STEP OF
OTHER WORK in flight — for us that means ANOTHER WINDOW: two cohort Verifiers (slots 0-3 /
4-7), two non-blocking streams (each Verifier already owns one), ONE service reactor — all
on the EXISTING batch_launch/batch_poll + pump() machinery. Per-verifier buffers/flags make
the window-N+1 overwrite hazard structurally impossible (the G2 ghost). F-196's commit
double buffer = the in-repo template (S1 fallback shape). Monotonic-across-windows flag
values ((window<<6)|step) replace per-window resets. Frozen cohort layouts + pad rows =>
graph re-keying becomes boot-time. MTP one-model overlap = the TRT-LLM-allowed spec mode.
STAGES (each env-gated + reversible, byte-exact mimic gate per stage): S0 bench+gates → S1
defer-raise + F-196-style payload parity (proves machinery, +0-1.5%) → S2 reactor-ize the
concurrent lane via batch_launch/poll (+3-8%) → S3 TWO COHORT VERIFIERS (+20-45% → 300-360;
the two-stream contention factor is the one number S3 exists to measure). Failure-mode
ledger from the wild: SGLang #18168/#6949 overlap×EAGLE crash class; vLLM #29134 spec still
forces a host sync; EqSpec ragged-batch corruption class => gates are BYTE-EXACT token
equality + position-contiguity asserts, not scores. Rejected: KTransformers deferred experts
(numerics-changing), multi-threaded pool (no win), conditional-node host-spin removal.

### T11.R4 — CPU expert pool lane (two decisive numbers + ranked moves)
(a) The pool is PARALLELISM-limited, NOT bandwidth-limited: 42-84 entries x 1.94MB =
81-163MB/window at 4.2-4.8ms = 17-38GB/s aggregate vs 64GB/s/CCD fabric + ~60-70GB/s DRAM
real caps; only ~3-5 unique experts/layer = 3-5 streams at the SINGLE-CORE roofline
(~26-28GB/s/core). (b) Hiding beats shrinking: full hide = +23-29% t/s; all kernel work caps
at -0.4..-1.0ms. RANKED: #1 the repo's OWN CPU-TIER-OVERLAP-SPEC.md wait-at-combine finish
(-4.2..-4.8ms IF staging delivers; NOTE the F-223 additive measurement vs R4's arithmetic
(C=0.12ms/layer < R=0.43ms/layer => should hide) implies the residual is INTER-LAYER SERVICE
LATENCY (wake + post-window work) = exactly R2-S2's reactor target — run R4's discriminator
ms_pool/(ms_wait/n_layers) FIRST); #2 split each expert's matvec across 2 pool threads
(3-5 streams => row-split to ~40-45us/entry, +7-11% pre-overlap); #3 THP madvise+hugify the
arena (NOT system always; 474 PTE walks/entry today; file-backed THP needs ext4/xfs —
NTFS3 untested => pairs with #4); #4 arena off NTFS3 (steady-state ~0, but unlocks #3's
fs-THP + fixes the fault path; one-time 85GB copy); #5 pool workers on the FREQUENCY CCD
(X3D L3 useless — zero reuse streaming; 5750 vs 5550 fmax; service thread gets the X3D CCD);
#6 vpermi2b/VBMI LUT + VNNI Q2_0 down (llamafile/llama.cpp precedents; only if #1 leaves C>R);
#7 phase-boundary prefetch priming; #8 UMWAIT port (power only). DO-NOT-BOTHER: arena-place
(symmetric DIMMs), NUMA (AM5 UMA), DDR5-6000+ chasing (4-DIMM 2DPC caps ~5200-5600; bandwidth
not binding anyway), mlock-as-throughput, T-MAC (activation quant changes tokens), SMT-off
globally (pinning suffices), routing/residency tail-shrinking (refuted F-125/F-127).

### T11.R5 — the QUALITY GATE design (replaces the two saturated instruments)
THE PROBLEM (measured): MMLU-70's CI is ±5.1pp vs a 4% adopt bar (F-200 saw 92.86 AND 97.14
on the SAME build); mimic is ceiling-pinned at 100%. 3-bit damage lands in code/math FIRST
(arXiv 2404.14047) — MMLU-only is structurally blind. DESIGN: (prereq ~1-2 days) an
/internal/loglikelihood surface (serve/ has NO logprobs today — stubbed); freeze golden
prompt sets + goldens under git. TIER-0 per token-stream-affecting adoption (~45-75min):
G1 spec-vs-nospec DIFFERENTIAL — battery-nospec.sh EXISTS (the "pack contract refuses --spec
0" belief was wrong for differential boots) — 300 fixed greedy prompts + pair/6-conc shapes,
0 divergent tokens single-stream, ANY conc divergence = investigate-before-adopt (EqSpec
class); G2 logit canary: 128k-token corpus NLL + Unsloth-class Divergence-300@32 KL vs
golden (the field's "lossless" standard = KL-within-noise, never "MMLU unchanged"); G3
MMLU-Pro n=300 stratified, CLOZE-LOGLIKELIHOOD grading (kills conditional variance, up to
2/3), paired vs golden (Miller arXiv 2411.00640); G4 steering cost same-boot cvec on/off.
TIER-1 weekly: generative MMLU-Pro n=300 with McNemar, code/math cell n=150 (3-bit damage
channel), mimic panel ΔNLL+top1+JS+MAUVE (replaces the 100% ceiling), GPQA-M monthly.
CONTROL VECTOR finding: scale 1.0 layers 4-24 is on the wrong side of both knobs (early
layers + high scale = collateral damage per the steering literature); sweep
{0.6,0.8,1.0}x{4-24,10-20,12-18} within ONE boot likely finds equal refusal at lower damage
— a possible free capability recovery. Batch-numerics law (Thinking Machines): greedy is
seed-free; comparisons must be boot-vs-boot, never seed-vs-seed.

### T11 MASTER QUEUE (supersedes T9's; 2026-10-16 evening)
1. CONC-300: R2's S0→S3 cohort pipeline (start S1+S2; S3 = the decisive stage).
2. SINGLE-250: T11.R3 FastMTP retrain (GO verdict; 1-2 days eng + weekend compute) — the
   acceptance lever everything else compounds on.
3. PREFILL: a1444e33 dense-MMQ prefill (-27..-30% IQ3_S 2K; T11.R1 #1) + 95979cf2 copy-ahead.
4. KERNELS: sub-warp experts (SUBWARP-EXPERT-PORT-PLAN.md) + MMQ-7-8tok windows (appendix).
5. CHEAP WINS: 2d6b1db8 PLE-beside-layer-0 (+4.1% code decode); R4#3+5 THP+CCD pinning;
   R5's /internal/loglikelihood + the cvec sweep (quality recovery).
6. QUALITY INFRA: R5's Tier-0 gate before the next numerics-touching adoption.
