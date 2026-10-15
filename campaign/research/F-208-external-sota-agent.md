# F-208 — EXTERNAL SOTA RECON: upstream, attention, spec-decode, MoE offload, prefill (external agent, 2026-10-15)

**Method:** web + GitHub-API reconnaissance only. No builds, no benchmarks, no server contact.
Repo reads were read-only (git ls-remote, API fetches, grep of the local tree at
`v0.1.44-48-g74bff7a` to verify which upstream code markers ARE/AREN'T in our base).
All "today" dates are 2026-10-15. Confidence: **High** = primary source (API/page fetched
directly, or observed in the tree); **Med** = multiple credible secondary sources or
self-reported-but-detailed practitioner numbers; **Low** = single secondary source / not read in full.

**Budget lines referenced (6-conc window 31.41 ms, F-208 overhaul):** QSA attention 4.40,
hc reads 4.2, doorbell tail 4.90, GDN ~3.6, emit ~1.87, GPU-reach 21.68, adapt 0.18.
Goal ×1.30 on M1 / pair / 6-conc vs stock v0.1.44 (M1 147.1 / pair 149.1 / 6-conc 187.7).

---

## 1. UPSTREAM WATCH — has stock moved past v0.1.44?

**U1. v0.1.44.1 exists; main == v0.1.44.1; it is a pure hotfix. NO RE-BASELINING NEEDED. — High**
Evidence chain: `git ls-remote` shows `refs/tags/v0.1.44.1` → commit 296b59d which is also
`refs/heads/main` (verified 2026-10-15, so nothing newer is even on main). The
compare `v0.1.44...v0.1.44.1` contains exactly 11 commits (Oct 13), all hotfix/servo:
- #2324: NVFP4 CPU lookup tables carried AVX static initialisers → silent SIGILL at start on
  pre-AVX2 CPUs (Xeon X5670-class). Not our CPU (9950X3D has AVX-512).
- #2327 (fixing #2322): CUDA 12.0/12.1 build fix — `__ushort_as_half` in host code replaced by
  manual bit-widening, verified over all 65,536 inputs. We build CUDA 13.4 — N/A.
- #2323: Apple M2 `gdn_step` launched with 448 threads where the kernel needs 512 → kernels
  silently never ran on M2 Max; Metal dispatches now validate thread limits. Not our platform,
  but note the *class* of bug: launch-geometry validation. Our hc-read reshape (queue #4) should
  add the same assert.
- Server crash reporting, setup preserving installed model size, reasoning_effort on Responses API.
Release notes state verbatim: "answers are byte-identical to 0.1.44". No perf content, no
kernel changes, no cherry-picks for us (optional: #2327 only if we ever build CUDA 12.x).
Sources: `git ls-remote https://github.com/Niko1221/Strata`; releases page
(https://github.com/Niko1221/Strata/releases); compare API v0.1.44...v0.1.44.1.

**U2. No merged perf PR after #540 that we lack. The highest-value upstream work for our
hardware is sitting UNMERGED in open/closed PRs. — High**
Verified individually via the API: #646 (closed, never merged as PR), #783 (closed, never
merged), #2075 (OPEN), #2514 (OPEN), #1339 (OPEN), #2326/#2050 (OPEN), #1912 (OPEN),
#2268/#2482 (OPEN issues). Detail per item below.

**U3. PR #540 lineage (our queue item #3's source kernel). — High**
https://github.com/Niko1221/Strata/pull/540 (sskver, merged Oct 4 → already inside our v0.1.44 base;
local tree has `attn_chunk_kernel_pre75` in `src/kernels/cuda/qsa_decode_attn.cu`).
As merged: `attn_chunk_kernel_pre75` compiled ONLY under `STRATA_EXPERIMENTAL_SM60`, launched
only below sm_75 (`STRATA_ATTN_PRE75`), because it uses ~2× registers (80 vs 38 on sm_70) and
also serves decode. Bit-exact by construction (same operands, same order). Numbers: V100 8K
prompt +22%, attention fix alone +7%; sm_75 +8-9% prefill. Extended upstream by:
- #655 (BF16 projections via FP16 TC, sm_75, +15-18% prefill on 2080 Ti; NOT bit-exact, KL 8.4e-3 — irrelevant for us, IQ3_S int8 path),
- #600 (TC prompt attention sm_70), #593 (superseded), #835/#849 (HIP ports, +6-12% prompt on RX 6900 XT),
- **#2075 (OPEN, imanu86, Oct 12): `STRATA_ATTN_DAILY=1` runs the QSA *decode* attention (verify
  window + MTP drafter, single query AND batch) through #540's pre75 chunk kernel on EVERY CUDA
  card, plus `attn_merge_v2_kernel` which skips chunks the chunk kernel marked empty. Bit-for-bit
  vs default kernels (reduce12 adds the same pairs as warp_sum's butterfly; parity ctest
  `qsa_merge_daily_parity_*` on four KV formats incl. non-resident pages). nsys RTX 3060:
  attn_chunk_kernel 99.8 µs → 46.4 µs per call. Same SMEM footprint (15,872 B, `spt` shares `sq`),
  68-78 regs.** Author cautions decode attention was only ~8% of per-row window cost at 131K on a
  3060 — on OUR box QSA attention is 14% of the 6-conc window and the base kernel is the older
  dependent-load-chain variant (F-208 overhaul A1), so the equivalent row-fold is worth
  proportionally more here. MAPS: QSA 4.40 ms line; queue #3 is upstream-validated; #2075 gives
  us (a) the empty-chunk skip merge and (b) the batched/drafter hookup pattern for free.
  Incremental. Bit-exact: YES per #2075's parity tests (stronger evidence than #2268's stack drift,
  which involved other switches — see U5).

**U4. PR #646 / #783 (stuchapin909): the big unmerged perf pile — PARTS already in our base,
parts NOT. Check item-by-item before re-implementing. — High on presence/absence (grepped the
local tree); Med on their numbers (self-reported, sm_86, unreviewed).**
https://github.com/Niko1221/Strata/pull/646 (+39-72% tok/s, dual 3090, IQ2_XS 256K, --spec 4,
temp 0; claims exact FP summation trees, bit-identical trajectories + acceptance counts, all
windows). https://github.com/Niko1221/Strata/pull/783 (+9-10% more on 0.1.39, 0-ULP parity over
25 GPU tests). Neither was merged via the PR mechanism (#783 says "#646 merged only the initial
commit"), and the upstream compare API 404s on that commit — but our local tree HAS:
- `row_dot_40_sub8` / `EXACT_N` in `iq_kernels.cu` → sub-warp row packing (4 rows/warp for IQ4_NL,
  2 for Q2_0) + exact-n∈{1,2} routed-group specialization landed. (Note: our model's TD may differ;
  the *technique* — kill idle lanes on the second pass — is what matters.)
- `all_resident_` zero-doorbell fully-resident verify graph in `verify.cpp` → exists in stock
  v0.1.44 but requires 100% residency; we run 75% — this is exactly why queue #1 (partial device
  plan) is the right generalization, and it has an in-tree precedent.
- `gdn_step_commit` + EXACT_T specializations in `verify_kernels.cu` / `fused_gr.cu`.
NOT in our tree (grepped): `gdn_conv_multi` (single-launch GDN conv across the window),
`hc_write_multi` (single-warp hyper-connection write), batched PLE key/value projections
(65.5 MB BF16 read once per window instead of per token), fused MTP q_proj/q_idx and gate+up
projections with NVMe/mmap PLE prefetch overlap, elimination of 48 redundant `f32_to_bf16_bulk`
launches/window, SMEM staging of IQ-codebook grids (`stage_iq_grid` — kills `__constant__` bank
conflicts on divergent lanes; directly relevant to IQ3_S). MAPS: hc 4.2 ms (hc_write_multi,
conv multi), GDN 3.6 ms (gdn_conv_multi; #783's mixer kernels), emit (PLE batch, MTP fused
projections). Incremental per item. Bit-exact: their claims are exact-tree-preserving; re-verify
with our parity harness per item.
URLs: /pull/646, /pull/783.

**U5. Issue #2268 (RTX PRO 6000 sm_120, all-resident, 0.1.43.1) — external corroboration of two
of our own conclusions. — High**
https://github.com/Niko1221/Strata/issues/2268
- `STRATA_DUMMY_NODES` probe (1,536 empty nodes injected): deltas within ±3% noise → node
  overhead ≤ ~0.5 µs, fusion ceiling ~2-3% decode. Independently confirms our F-206 "node-count
  fusion is wall-neutral" negative. Stop spending there.
- Fused-stack opt-ins (EXPERT_V2, TSUM, ATTN_LANECELL, PLE_BATCH, MMVF_ROWS together): neutral on
  a fully-resident sm_120 card; GDN_CHUNKED negative on sm_120 (RTX 5090 recurrence gains did not
  carry over). Caveat for our audit file's ATTN_LANECELL optimism: in this stack it contributed
  ~0 on resident sm_120 — but our card is 75%-resident with doorbell misses, a different pressure
  profile, and the audit measured it positive on our wrapper. Re-measure, don't assume either way.
- **Correctness warning: 2/12 greedy temp-0 probes drifted under the fused stack despite
  "bitwise equal" kernel claims.** Treat any multi-switch adoption as MMLU-gated, not just
  parity-tested.
- Bug flag: `STRATA_PLE_BATCH` unset counts as ON in prefill.cpp but OFF in verify.cpp — check our
  tree for the same asymmetry.

**U6. Issue #2482: `STRATA_ROUTE_TAIL_SKIP` (default ON = 7 since 0.1.42) vs 0: +6.2% warm decode
on RTX 5080 sm_120. — Med-High** https://github.com/Niko1221/Strata/issues/2482
The differing draft-acceptance totals between arms (221/307 vs 223/345 etc.) imply token streams
DIVERGE between skip=0 and skip=7 → tail-skip is NOT output-identical to skip=0. Both our trees
run the default (7), so baseline is safe; but treat any retuning of this knob as
distribution-changing (MMLU gate). No action for now; do not "fix" it blind.

**U7. PR #1339: experimental SM120 8-column Q8 QKV MMQ — measured NEGATIVE end-to-end by its own
author (isolated stage 20.5→14.3 µs, but decode 70.98 off vs 68.28 on, outputs diverge at index
16). Do NOT cherry-pick. Also a clean demonstration that changing reduction order both changes
outputs and can lose throughput. — High** https://github.com/Niko1221/Strata/pull/1339

**U8. PR #2326 (+ #2050): cold-prompt reads run beside decoding slot groups
(`STRATA_PIPED_CONCURRENT=1`, opt-in, experimental). 2×5090, 4 streams + 64K and 32K cold
prompts joining: longest stream pause 11.9 s → 0.8 s, same work 6.4% sooner; tokens SHA-256
equal when off. vLLM/SGLang chunked-prefill analog for Strata's split kernel paths — prompt and
decode kernels can't share a pass but CAN overlap on the same GPUs. MAPS: our pair/6-conc
metrics under prompt churn (the M1 lane is decode-only, so this moves the *concurrency* metrics,
not M1). Incremental (cherry-pick + validate on our slot topology). — High (directly measured on
2×5090).** https://github.com/Niko1221/Strata/pull/2326, /pull/2050

**U9. tqllm.com — Flash-Next ship has NOT landed. — High (site fetched today).**
https://tqllm.com — "Coming soon · in development, not in the image yet: 16% faster than Strata —
Qwen3.8-Flash-Next, 125B, on one RTX 3090... routed experts streamed from system RAM... same
session as Strata 0.1.44, with output identical to the exact reference." Still a 3090-lane
tuned-vs-stock tease (matches our F-207 read); nothing to re-baseline against. Their SHIPPED
techniques worth noting: `--moe-host-mb` keeps the LEAST-routed experts in host memory
(frequency-aware placement, see M1 below); TqCache oldest-KV-in-RAM exactly; DFlash2 drafter with
"tree, chain and MTP verify share one attention arithmetic with plain decoding, so greedy output
is identical token for token" (see S1); `--np` batched verify. Watch their Flash-Next ship; until
then they are not a competitor data point on our lane (5090/IQ3_S/75%-resident).

---

## 2. ATTENTION (QSA line, 4.40 ms)

**A1. The sm_120 hardware envelope is confirmed, and it shapes everything below. — High**
- RTX 5090 (CC 12.0) has NO thread-block clusters → no TMA multicast, no distributed SMEM
  (https://forums.developer.nvidia.com/t/thread-block-clustering-in-blackwell-gpus/320471;
  https://forums.developer.nvidia.com/t/different-ctas-accessing-the-same-shared-memory-address-on-rtx-5090-is-this-expected/352137).
- FlashAttention-4 cannot run on sm_120 — it needs sm_100-exclusive TMA/cluster/TMEM features
  (https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739); NVFP4 TMA-WS grouped-GEMM
  tactics crash with illegal instructions on "120a"
  (https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/discussions/9).
- What REMAINS available on sm_120: cp.async (sm_80+), plain (non-multicast) TMA, mma.sync-class
  tensor ops. FlashInfer ships and validates TMA warp-specialized configs on SM12x
  (flashinfer PR #4847). → Our QSA prefetch fix (cp.async, queue-adjacent) is squarely on the
  supported path; any Fireworks-style design must be ported minus tcgen05/TMEM/multicast.

**A2. Fireworks AI's MiniMax M3 sparse-attention kernel (top-16 of 128-token KV blocks per query,
exactly QSA-shaped): KV-outer scheduling beats Q-outer when queries share blocks. — Med-High
(single detailed practitioner source; B200/sm_100 only, no sm_120 data).**
https://fireworks.ai/blog/kernel-optimization-for-minimax-m3-on-nvidia-blackwell
- Q-outer (query-stationary, FlashInfer-decode-style): small MMA tiles, every selected KV block
  re-read per query → L2-bandwidth bound. KV-outer (KV-stationary): each selected block read ONCE,
  attend all queries that chose it, LSE-merge partials → HBM-bound but ~1.9-2.4× the Q-outer
  baseline; crossover rule nsb/N < 2.85 (production long-context always qualifies).
- Portability to us: at 6-conc our verify window has up to 6 queries whose selected cell sets
  likely overlap heavily (same context, adjacent positions/depths). If overlap is high, a
  KV-stationary gather for the shared cells converts our dependent random V-loads into
  block-streamed reads — the direct fix for "15× off bandwidth, no cp.async prefetch".
- Portable pieces: (a) 4-stage split (index build / scheduler / main / combine); (b) cp.async
  gathered loads staged directly into compute layout; (c) index gather overlapped with previous
  tile load (one-tile pipeline); (d) move scatter/partial writes OUT of the latency-sensitive
  kernel into a bandwidth-bound combine kernel; (e) persistent grid ≈ #SMs with atomic-free
  device-side work splitting; (f) fixed per-request shapes to kill D2H syncs. MAPS: QSA 4.40 ms
  (and hc 4.2 ms via (e)). Rewrite-scale for the gather kernel; bit-exactness requires preserving
  our accumulation order per query — feasible if each query's cells are still summed in selection
  order (the KV-outer loop visits blocks in one canonical order; keep per-query sub-sums ordered).
- Related: NVIDIA GTC session on NSA kernels with CUTLASS Python gather/scatter + MLP control
  (https://www.nvidia.com/en-us/on-demand/session/gtc26-s81470); PyTorch TLX block-sparse warp-
  specialized Blackwell kernel (https://pytorch.org/blog/tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention) — fixed-pattern only, tooling reference;
  mit-han-lab/Block-Sparse-Attention (B200). All sm_100-first; scheduling ideas portable.

**A3. GVR: exact top-k via temporal-correlation warm start — productionized in TRT-LLM and
FlashInfer. — High that it exists and is bit-exact; Med on transfer to our select sizes.**
https://arxiv.org/html/2604.22312v1 (B200/sm_100; TRT-LLM PR #12385, `enable_heuristic_topk`).
Uses the PREVIOUS step's top-k indices as a warm start, threshold search (secant, 1-2 iters),
ballot-free collection with cached counts, histogram+snap refine → **bit-exact top-k index sets**
vs torch.topk (ties nondeterministic same as baseline). 1.88× avg over radix-select, up to 2.42×;
graceful degradation (min 0.98×); TPOT −7.5% at 100K. FlashInfer v0.7.1 ships it as `gvr_2` with
"SM-count-aware register band" dispatch rungs (flashinfer PR #4986) and a graph-safe CUB fallback
for page-table transforms on SM120 (PR #5226).
MAPS: our QSA top-N select. Prereq check: PR #2514 (upstream, OPEN) reveals that decode-shaped
top-k calls can silently land on the one-workgroup REFERENCE kernel (one workgroup per query,
4 radix passes re-reading keys) instead of the register kernel — on HIP that fix alone was
1.7-2.4× kernel-level; **audit which kernel our CUDA decode select dispatches to** before porting
GVR (the cheap win may already be sitting there). Our QSA cell selections across MTP depths and
verify positions in one window should show strong temporal stability (same context) → high hit
ratio. Incremental. Bit-exact: YES (selection identity preserved).

**A4. FlashInfer v0.7.x (v0.7.0 Sep 22, v0.7.1 Oct 9, 2026) — sm_120 sparse-attention maintenance
is active and directly relevant. — High (official release notes).**
https://github.com/flashinfer-ai/flashinfer/releases (tag v0.7.1)
- DeepSeek-V4.1 sparse MLA runs on SM120/121 with runtime KV row stride + "NaN-safe masked
  gathers" (PR #5197, #5075) — the masked-gather pattern is the public equivalent of our QSA
  gather hardening.
- SM120 Sage block-sparse attention with QK-INT8/PV-FP8 and ALL-FP8 modes (PR #4691) — INT8-QK
  block-sparse on our exact arch; reference for a QSA score-phase shape.
- PRIMS "QToken-KvBlock-Sparse-Attention" (+ shapes/patterns PR #4996, #5229) and "skip-softmax on
  SM120 PRIMS" (PR #4859) — cell/block-selected attention is a first-class public primitive now.
- Low-latency "Frost" DSv4.1 sparse MLA decode (PR #5119); fixed page tables for block-sparse
  (PR #5144); paged FA2 equal-stride KV specialization (PR #4736).
- **W4A16 CuTe-DSL MoE on SM12x runs a Qwen3.8-Flash-Next MoE layer 1.36× faster than Marlin at
  one token (DGX Spark)** — same model family; evidence of ~36% GEMM headroom on sm_120 decode
  expert kernels (different weight format, but a bound worth knowing).
- "Reduce PCIe CE ring publication overhead on SM120" (PR #5169) — publication/doorbell cost over
  PCIe on our chip is a recognized problem class; read before finalizing the posted-write
  doorbell inversion (queue #2).

---

## 3. SPEC DECODE (MTP, exact-match verify)

**S1. Lossless tree verification is settled science and shipped — but it is a rewrite for us, and
our chain+exact-match design already preserves the guarantee. — High on the guarantee; the port
is Med-effort.**
Sequoia (https://arxiv.org/abs/2402.12374, NeurIPS 2024): hardware-aware draft TREES with a
verifier that provably preserves the target distribution; under greedy it deterministically
accepts the longest correct prefix → lossless. OPT-Tree (https://arxiv.org/abs/2406.17276,
TACL 2025): adaptive tree shapes per step. tqLLM ships "tree, chain and MTP verify share one
attention arithmetic... greedy output identical token for token" (https://tqllm.com).
MAPS to queue #5 (depth lane): tree width buys more accepted tokens per verify window without the
depth-4..6 latency penalty — but our MTP drafter proposes a single chain, and converting verify to
tree form means new verify kernels + acceptance walk + drafter branching (moderate-rewrite,
bit-exact under greedy). Do NOT take this before the cheaper depth-lane fix; revisit only if
acceptance at depth ≥4 stays the binding constraint after the floor fix.

**S2. Adaptive depth/lookahead beats static: validates and de-risks queue #5. — Med-High.**
DISCO (https://arxiv.org/abs/2405.04304, ICML 2024, Intel): per-iteration speculation-length
selection via a classifier on drafter hidden states, ~10% over the BEST static lookahead.
Dynamic Depth Decoding (EAGLE-2 dynamic depth); ACL 2025 "On-the-Fly Adaptation of Speculative
Decoding" (https://aclanthology.org/2025.acl-long.482.pdf); Together ATLAS adaptive-learning
speculator (https://www.together.ai/blog/adaptive-learning-speculator-system-atlas, Oct 2025).
Cheap lossless variant for us: acceptance-history-driven per-window depth cap/floor (deterministic
rule on measured acceptances). Bit-exactness note that matters for ALL of queue #5: under
exact-match greedy verify, changing the DRAFT distribution (temperature/floor/min-p) cannot change
produced tokens — candidates either equal the target or are rejected; only acceptance rate and
window time move. The draft side is a free-tuning zone; no MMLU exposure.
Corroboration for the mechanism: llama.cpp discussion — a badly chosen draft `min_p` LOWERS
acceptance and can make speculation net-negative
(https://github.com/ggml-org/llama.cpp/discussions/10466).
Training-side (LK Losses, https://arxiv.org/abs/2602.23881, Feb 2026, + Nebius blog): direct
acceptance-rate training objectives — not applicable (we don't retrain the MTP head); noted for
completeness.

**S3. Fused linear-attention/GDN work inside speculative verify is a proven win class. — Med.**
FlashInfer v0.7.1: fused Kimi K3 speculative verification — short conv + Kimi Delta Attention +
gated RMSNorm fused during verify, CUDA-graph capturable, ragged batches; +1.8-3.3% total
throughput in vLLM at 6 draft tokens, conc 1-32 (B300) (release notes; PR #4809).
Analog for us: fold our GDN/conv window steps (the ~3.6 ms line) across all draft+verify positions
into single launches — which is the same medicine as U4's unlanded `gdn_conv_multi` /
#783 mixer kernels. Two independent sources point at the same line.

---

## 4. MoE OFFLOAD (doorbell tail 4.90 ms; 75% resident)

**M1. Frequency-based expert placement is now a first-class, documented mechanism
(KTransformers). — Med-High.**
KTransformers (Tsinghua) paper: "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid
Inference for MoE Models" (ACM, 2025, https://dl.acm.org/doi/pdf/10.1145/3731569.3764843);
SGLang adopted it as a hybrid CPU-expert backend (https://www.lmsys.org/blog/2025-10-22-KTransformers,
Oct 2025); serving docs expose `--kt-num-gpu-experts` with placement strategies
**uniform | frequency | front-loading | random**, plus `--kt-enable-dynamic-expert-update`
(runtime re-placement from routing stats) and `--kt-max-deferred-experts-per-token` (deferred
expert execution for pipelining) (https://ktransformers.net/en/docs/optimization-techniques/expert-placement).
Question for our tree: is our 75% residency assigned uniformly/per-layer, or by routing frequency?
If uniform, frequency-aware placement (even a static profile from our own routing traces — upstream
#118 "routing traces exposed from the live server" is in-base tooling) directly shrinks the miss
rate that feeds the 4.90 ms doorbell tail. Outputs UNCHANGED by construction (every routed expert
is still computed; only where it runs moves) → bit-exact, no MMLU exposure. Incremental.
tqLLM's `--moe-host-mb` ("keeps the LEAST-routed experts in host memory") is the same idea shipped
on a 24 GB card.

**M2. Two-tier pinned-arena + hot-set caching is the converging design. — Med.**
llama.cpp issue: pinned-memory CPU region for ALL experts at load + GPU caching of frequently
used experts + non-blocking transfers
(https://github.com/ggml-org/llama.cpp/issues/20757, Mar 2026). Same shape as our T1.1 partial
plan — external validation of the design, worth a skim for failure modes (eviction policy,
fragmentation). Our kv-stream neighbors' selection-aware LRFU discussion (upstream #2415/#2458,
`--dump-kv-select` policy-bench trace) is the in-tree analogue if we ever need an eviction policy
for the expert arena.

**M3. GPU→host signaling: posted writes and copy-engine ring costs are the recognized cost
centers. — Med.**
FlashInfer #5169 (PCIe CE ring publication overhead on SM120, see A4); our queue #2
(cuStreamWriteValue posted-write doorbell inversion) has no public contradiction; GDRCopy-style
GPU-initiated host writes remain the canonical low-latency signaling primitive. Gap note: I did
not find a public write-up of the specific cuStreamWriteValue-as-doorbell pattern in an inference
engine — treat our inversion as novel-but-plausible; measure, don't cite.

**M4. Not followed up (recorded to avoid re-searching):** PowerInfer-2 lineage — searches this
session didn't surface 2026 advances beyond the above; arXiv 2609.14507 (Sep 2026, physically
partitioned KV cache for CPU-GPU offload + greedy expert-placement schedule critique) is a lead
only. We do not offload KV, so priority stays low.

---

## 5. PREFILL

**P1. Layered prefill (arXiv:2510.08055) — public code exists; follow-up named. — High on
existence; follow-ups unread.**
Paper: https://arxiv.org/abs/2510.08055 (SNU; layer groups as the scheduling unit; stall-free
decode; eliminates chunk-induced MoE weight reloads; ~6 citations). Code:
https://github.com/scale-snu/layered-prefill — having a reference implementation materially
de-risks queue #6. Named successors (NOT yet read; queue a follow-up): FineMoE (EuroSys'26,
fine-grained expert offloading), fMoE (arXiv:2502.05370), MoE-Prefill (arXiv:2605.02960,
prefill-only serving eliminating redundancies on the MoE prefill critical path).
MAPS: queue #6; also feeds the concurrency metrics via U8 overlap.

**P2. GDN prefill/decode kernel references. — Med.**
- FLA chunkwise-parallel prefill vs recurrent decode is the canonical split (Alibaba's Qwen3-Next
  claims ~7× prefill scaling: https://www.alibabacloud.com/blog/602580; vLLM native support incl.
  MTP: https://vllm.ai/blog/2025-09-11-qwen3-next).
- FlashInfer v0.7.1: **cuDNN's open-source fused linear-attention engine runs Gated DeltaNet
  prefill on Blackwell** — `chunk_gated_delta_rule(backend="cudnn")`, CUDA-12-compatible (PR
  #4968); default GDN prefill dispatch restored to CuTe (PR #5225); GDN CuTe-DSL decode kernels
  tuned per operand device (PR #4507); SM100 chunked-prefill kernel optimizations (PR #4917).
  Blackwell ragged prefill via backend="auto" up to 3.3× (PR #5133).
- CAUTION from inside Strata: `STRATA_GDN_CHUNKED=1` measured NEGATIVE on sm_120 (RTX PRO 6000,
  #2268) — the in-engine chunked path is not automatically the FLA story; our GDN ~3.6 ms decode
  line is better attacked via U4's single-launch multi kernels than by re-attempting chunking.

---

## 6. CONTRADICTIONS AND THEIR RESOLUTION

1. **"Fused kernels give 9-10%" (PR #783, sm_86) vs "fusions are neutral on sm_120" (#2268).**
   Resolution: #783's gains came from 2×3090s running a SPLIT layer with launch/SMEM pressure and
   doorbell round-trips; #2268's card was fully resident. Our box (75% resident, doorbell misses,
   170 SMs) sits between; per-item measurement required, no wholesale adoption. #2268's DUMMY_NODES
   ceiling (~0.5 µs/node) plus our own F-206 negative means pure node-count fusions stay dead.
2. **"#2075 bit-exact" vs "#2268 fused stack drifted on 2/12 probes."** Different switches: the
   drift came from EXPERT_V2/TSUM/PLE_BATCH/MMVF_ROWS, not the pre75 attention kernel; #2075 has
   memcmp parity tests on four KV formats. Adopt #2075-equivalent with its parity harness; gate
   the stack switches behind MMLU regardless.
3. **ATTN_LANECELL positive in our F-200 audit read vs neutral in #2268's sm_120 stack.**
   Unresolved directly; different baseline kernels (ours still runs the dependent-load-chain
   variant). Our queue #3 (row-fold) subsumes the mechanism; re-measure LANECELL only if row-fold
   leaves the 4.40 ms line above projection.
4. **tqllm "16% faster than Strata" vs our +22-27% M1.** Different lane (3090, 125B Flash-Next,
   tuned tqLLM vs stock Strata, output identical to reference). Not comparable to our 5090/IQ3_S
   tuned-vs-stock numbers; no re-baselining. Their Flash-Next is still unshipped (fetched today).

## 7. OPEN QUESTIONS

- Which kernel does our CUDA decode QSA top-N select dispatch to (ref vs register variant)?
  (A3 / PR #2514 evidence; one grep + one profile away.)
- What is the inter-query selected-cell overlap at 6-conc? Decides whether KV-outer (A2) pays.
- Is our 75% resident set uniform or frequency-aware? Decides M1's value on the doorbell tail.
- Does our tree share #2268's STRATA_PLE_BATCH prefill/verify default asymmetry?
- FineMoE / fMoE / MoE-Prefill contents (P1 follow-ups) — unread this session.
- PowerInfer-lineage 2026 work — not surfaced; low priority.

## 8. IMPLICATIONS FOR THE QUEUE

- Queue #1 (partial device plan): externally validated twice (in-tree all_resident_ precedent;
  llama.cpp #20757 converging design). Keep priority 1.
- Queue #2 (posted-write doorbell): plausible and uncontradicted; FlashInfer #5169 shows PCIe
  publication cost is a real sm_120 problem class. Measure against CE-ring alternative.
- Queue #3 (row-fold onto sm_120): upstream #2075 hands us the bit-exact decode hookup + empty-
  chunk skip merge — cherry-pick or port; bigger relative upside for us (14% of window vs their 8%).
- Queue #4 (hc-read reshape): #646's unlanded `hc_write_multi`/`gdn_conv_multi` are the same
  medicine; add launch-geometry validation (the M2 #2323 lesson).
- Queue #5 (depth lane): draft-side tuning is provably output-preserving under exact-match;
  acceptance-driven adaptive depth has ~10% precedent (DISCO). Tree verify only if this stalls.
- Queue #6 (layered prefill): reference implementation is public.
- NEW candidates: frequency-aware expert placement (M1, bit-exact, likely leverages the 4.90 ms
  line); #2326 PIPED_CONCURRENT cherry-pick for pair/6-conc under prompt churn; decode top-k
  dispatch audit (A3). AVOID: #1339 eight-column QKV; further node-count fusion.
- Baseline: stock = v0.1.44.1 now; byte-identical outputs, zero perf delta on our lane. Keep
  measuring against v0.1.44; add one v0.1.44.1 sanity row to the next stock boot for the record.

## 9. SOURCE LIST

Code/history (Niko1221/Strata): git ls-remote tags; /releases; compare v0.1.44...v0.1.44.1;
PRs #540, #646, #783, #2075, #2514, #1339, #2326, #2050, #1912, #109 (list page, low weight);
issues #2268, #2482; local tree grep at v0.1.44-48-g74bff7a (iq_kernels.cu, verify.cpp,
qsa_decode_attn.cu, fused_gr.cu, verify_kernels.cu).
Data/knowledge bases: GitHub API (Strata pulls/issues/search, compare, releases; FlashInfer
releases tag v0.7.1); llama.cpp issue #20757; HuggingFace NVFP4 discussion.
Specs/docs: FlashInfer v0.7.0/v0.7.1 release notes (flashinfer-ai/flashinfer/releases);
KTransformers expert-placement docs; vLLM spec-decode docs (docs.vllm.ai); NVIDIA developer
forums (cluster support on CC 12.0).
Literature: arXiv 2604.22312 (GVR); 2510.08055 (+ scale-snu/layered-prefill code);
2402.12374 (Sequoia); 2405.04304 (DISCO); 2406.17276 (OPT-Tree); 2602.23881 (LK Losses);
aclanthology.org/2025.acl-long.482.pdf; ACM 10.1145/3731569.3764843 (KTransformers).
Practitioner: fireworks.ai MiniMax M3 kernel blog; pytorch.org TLX block-sparse blog;
NVIDIA GTC S81470; lmsys.org KTransformers blog; together.ai ATLAS; nebius.com LK Losses;
llama.cpp discussion #10466; tqllm.com; gist (FA4-on-5090); alibabacloud.com Qwen3-Next blog;
vllm.ai Qwen3-Next blog; SGLang issue #18590.

## 10. METHODS NOTE (unproductive paths, do not repeat)

- GitHub PR-list HTML fetch (pulls?q=is:merged sort:created-desc) returned inconsistent dates and
  numbering — always use the REST API for PR/issue facts; HTML only for discussion color.
- WebSearch failed with "Model request failed" on ~6 multi-query batches; singles succeeded.
- "Sequoia lossless verify" needed an exact-paper query; generic "tree verification lossless"
  phrasing failed once.
- PowerInfer-2 2026 status: no productive hit; parked (M4).
- GTC S81470 session content not readable without registration — cited as a pointer only.
- tqllm "How it was measured" subpage not fetched this session; F-207 already characterized the
  lane (tuned-vs-stock, 3090); site fetch today confirmed the tease status only.
