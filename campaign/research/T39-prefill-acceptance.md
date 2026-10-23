# T39 — PREFILL PLATEAU & ACCEPTANCE MATH: the two tasking-named prefill candidates are already consumed or broken-premise at HEAD (record corrections), the short-prompt/chunk/KV reads are cheap and un-run, and the acceptance lane's real levers are one instrument, one env sweep, and one never-measured exact path

Date: 2026-10-23 (wave T39, researcher R3). Scope: read-only — code, journal, committed logs, git history,
web. No boots, no builds, no git ops, no INDEX edits; the only file written is this dossier. Repo
`/home/svg/LLM/strata-custom-2` at HEAD `3219b765` (F-278).
Grading: **[MEASURED-OURS]** = banked campaign record (journal F-entry + artifact) or read in code/output
this session; **[CORROBORATED]** = primary external source fetched or search-verified this session
(ID/URL given); **[CARRIED]** = primary external source verified by T37/T26 waves this month, re-checked
against this dossier's claims; **[HYPOTHESIS]** = my arithmetic from banked + corroborated numbers, marked
ESTIMATE where a constant is unmeasured. Every ms claim shows its arithmetic.
Builds past (not re-specified): T37-kernel-math §3.1 (TC-hc mma rungs), T37-scheduling-spec C1 (the
conditional-graph keep rule) — both are specced and owned there; T26-3 (matmul inventory + plateau
attribution), T26-5 (numerics/KL yardstick), T17C (spec algorithm leaps), T17F (training/quant/context),
T31 (keep-rule refit), T25-4/T25-3 (family studies).

---

## §0 SUMMARY

1. **Two record corrections that re-price the tasking's A-lens before any boot is spent.** (a) The T15
   "[PREFILL #1] MMQ quantize-dedup (~4-5%/chunk bit-exact)" is **already consumed at HEAD**: commit
   `c675f5b7` (2026-10-14) landed the scatter form — each token's MoE input quantized **once** and its
   q8_1 bytes written to its K expert rows (`quantize_scatter` default path, `prefill.cpp:4025-4040`;
   kernel `quantize_mmq_q8_1<scatter=true>`, third_party/llama.cpp `quantize.cu:606-640`;
   `STRATA_QUANT_GATHER=1` restores the old gather form). What remains of the item is only the 10× q8_1
   **write** amplification (~50 GB per 32k chunk ≈ ~30 ms ≈ 0.7% of the 4.3 s chunk wall at 1.7 TB/s,
   before L2 absorption) plus ~0.95 GB of Xq buffer VRAM — a decode-side slot prize, not a prefill-t/s
   one (§3.6). (b) The T37 §3.4 / T26-5 §c **K=640 fp16-accum scope is aimed at a call site that does
   not execute at the standing recipe**: F-239's "gemm down 309 ms" phase is the **MMQ int8 product**
   (H quantized to the down-expert's own type, `mmq::Product`, `prefill.cpp:4560-4595`), not the fp16
   dequant-GemmEx; the `Gemm::f16(..., N, 640)` call (`prefill.cpp:4626`) runs only on the
   fallback/FP16 expert path (`STRATA_PREFILL_MMQ=0` or non-fitting layers). T26-3 §a.1 P3 had it right;
   T26-5/T37 inherited the phase-name misreading. The proposed nsys falsifier would have returned "no
   cuBLAS kernel at that shape" — settled here for free (§1.2, §4.1).
2. **T3.1/#29807 GDN snapshot-write fusion is already the tree state.** F-216 phase 1 (2026-10-16) fused
   per-column recurrent-state snapshots INTO `gdn_step_norm_multi_kernel` (verify-mode writes of the
   register state after each column; verify_kernels.cu:337-357-class; arena at verify.cpp:791-816; the
   serial commit promotes snapshot[m-1], verify.cpp:1265). There is no separate cpy pass on the verify
   path to remove, and the prefill recurrence (`gdn_rec_chunked`) never had one. T6.10 is
   stale/superseded; F-216's final verdict (perf-negative for the promotion scheme in chains) already
   closed the family (§1.4).
3. **The prefill plateau's honest state at HEAD:** no un-priced in-tree lever of the ≥3% class remains.
   The MoE 41.4% line runs MMQ int8 at ~186 TFLOPS-equivalent on gu (523 ms) but only ~83.5 on the
   K=640 down product (309 ms) — the down product is the complex's soft line and its binder is
   shape/skinny-GEMM inefficiency, not the accumulator pipe (§2, §3.6). The dense 238-lock lines are
   T37 §3.1's territory. The cheapest real prefill moves are all reads: `tools/calibrate.py` (chunk
   b/a + pcie-frac — never run on this box), the KV_PREFETCH-at-131k decomposition, one ncu pass on the
   scatter-quantize writes, one PROMPT_DIRECT close-out leg (§3.4-3.7).
4. **The acceptance lane's ledger is older than the stack it stands on.** τ=0.2 is an R5-era value never
   re-swept post-DT-0.85/post-TYPICAL_DEVICE (T15C:68); the rescue stack's headline pricing (+20.6 pts
   accept / +8% M1 launcher comment; F-200's +9.0/+6.7/+6.4 vs *interpolated* control in a deep-trough
   window) predates COHORTS, elastic, dm, and the F-263 five-boot law. T31's refit on the banked
   1,615-window trace prices what the rescue stack actually does today: +0.289 tokens/window of
   rescue-inflation (mean C 2.446 exact → 2.734), 24.2% of windows rescued, offered-denominator
   acceptance 65.0% exact vs 77.96% rescue-inclusive (§1.5, §3.1).
5. **The 9-conc 77.98% "highest acceptance" is a denominator artifact until the batch lane gets a trace
   twin.** The serial M3b number is rescue-inclusive (and output-bending); the cohort lane runs no rescue
   (BATCH_RESCUE=0) and one draft/slot — its 77.98% is exact-match over position-0 tests. The
   comparison that would justify a second draft row needs instrument-first pricing (§3.2); T37 C4's EV
   arithmetic (+6-12% conc [HYPOTHESIS]) stands but its acceptance input is unmeasured.
6. **External 2025-26 verdicts:** the MoE prefill kernel frontier (FlashMoE/UniEP/MegaMOE-in-FlashInfer)
   is persistent-megakernel EP machinery — wrong problem class for a single-GPU offload box, and its
   transferable atom (one launch per expert group) is already ours. The entropy/adaptive-acceptance
   class (AdaSD 2512.11280) is lossy and fails the exactness law. The one live external datum for our
   short-prompt lane is llama.cpp #29130 (mmap expert reads slow prompt processing; O_DIRECT proposed) —
   it keeps PROMPT_DIRECT on the shelf for the cold tier while our own prior on this cache-warm box is
   negative-to-flat (§2, §3.7).

---

## §1 OUR-STATE VERIFICATION (every anchor opened at HEAD 3219b765 this session)

### 1.1 The MoE prefill expert path (the 41.4% line) — how it actually runs

- `mmq_plan()` (prefill.cpp:1159-1213): default **ON** (`env == nullptr || atoi != 0`), per-layer, keyed
  on `mmq::fits(gu_type, 1280)` / `mmq::fits(d_type, N)`; our IQ3_S pack's types all fit → every layer
  takes MMQ; `p.fallback` false. `STRATA_PREFILL_MMQ=0` is the kill to the FP16 path.
- The standing product chain per MMQ group (prefill.cpp:4560-4595): gather experts into group slots
  (`gather_native`, marked kPfDequant — a **misnomer**, it is the group-gather, no dequant) →
  `mmq::Product gu` (weights IQ2_S/IQ3_S/IQ3_XXS-class × Xq q8_1) → `mmq::swiglu` →
  `mmq::quantize(H → Hq at the DOWN type)` → `mmq::Product dn` (weights Q2_0/IQ4_NL × Hq) → combine.
  The fp16 dequant-GemmEx chain (kPfDequant = true dequant to `dq_gu/dq_d` rings, `Gemm::f16` calls at
  prefill.cpp:4614/:4626) is the `else` branch — executed only for fallback layers or `STRATA_PREFILL_MMQ=0`.
- Therefore F-239's standing anatomy (knob-pft-engtail.txt, 2026-10-19, post-dating the MMQ default):
  "gemm gu 523 / gemm down 309 / dequant 156" = **group-gather + int8 MMQ products**, matching T26-3's
  own ~186 TOPS-effective derivation (154.6 TFLOP / 0.832 s). The K=640 "GEMM" in that table is an int8
  MMQ run whose activations were quantized to the down-expert's own type — the fp16-accum question does
  not arise on it.
- Down-product efficiency arithmetic [ESTIMATE, from banked lines]: gu = T·K·1280·2560·2 FLOP = 103.1
  TFLOP per 32k chunk / 523 ms = **197 TFLOPS-equiv**; down = T·K·2560·640·2 = 25.8 TFLOP / 309 ms =
  **83.5 TFLOPS-equiv**. The down product is 2.4× less efficient per FLOP — K=640 skinny (tile
  quantization, corroborated: NVIDIA GEMM performance guide, §2/S8) plus q8_1 re-quantization of H per
  group. Any down-line lever competes with this number, not with 238.

### 1.2 The K=640 fp16-accum scope — premise broken at standing (record correction)

- T37-kernel-math §3.4 priced "COMPUTE_16F on the down-proj Gemm::f16 call (`prefill.cpp:4627` call
  shape)" at +3.0-3.7% prefill with a KL gate; T26-5 §c's per-op table carries the same 309 ms line as a
  cuBLAS fp16 GEMM. Both cite F-239. Verified this session: the 309 ms is the MMQ down product (§1.1);
  the Gemm::f16 K=640 call executes **only** under `STRATA_PREFILL_MMQ=0`/fallback — a path whose own
  record is that it is slower (it is the pre-MMQ form the MMQ path replaced; F-230's separate dense-MMQ
  arm lost −2.8% M1 on acceptance-tax grounds).
- Consequences: (i) T37 §3.4's falsifier step 1 (nsys kernel name at (M=32k, N=2560, K=640)) would read
  "no cuBLAS kernel" at standing — the question it was to settle (FAST_16F vs COMPUTE_16F kernel
  selection) is moot on this path; (ii) the T37 §6 open question 4 is answered negatively-by-construction;
  (iii) the fp16-accum family on the prefill lane is now **fully closed**: blanket dead (T26-5 KLD model
  0.15-0.40), K≥6144 dead, K=640 dead-at-standing (this correction), decode dead (T26-3 a.2). What
  survives as a shelf variant only: COMPUTE_16F on the fallback path's down call, for a world where
  someone runs STRATA_PREFILL_MMQ=0 — nobody does.
- The T37 §3.4 statement "the plateau's structural answer remains the T14B requant" is unaffected and
  stands (−8.26% expert bytes shrinks the whole 41.4% complex; user-GO pending).

### 1.3 The quantize-dedup item — consumed at HEAD (record correction)

- T15/TODO line 1027: "[PREFILL #1, S-M] dedup MMQ activation quantize T*K→T rows + q8_1 gather-copy
  (prefill.cpp:3457; 10× DRAM amplification, 3.36 GB vs 335 MB distinct/layer) ~1.6-2.2%/chunk
  BIT-EXACT … side effect ~0.5 GB VRAM freed. Best first prefill battery."
- HEAD: commit `c675f5b7` 2026-10-14 "prefill: quantize each token's MoE input once and scatter it to
  its k rows (the same bytes)" — files: `include/strata/prefill/moe_mmq.hpp:48-54` (the contract doc),
  `src/prefill/moe_mmq.cu:396-407` (`quantize_scatter`; `STRATA_QUANT_GATHER=1` = old form),
  `prefill.cpp:4020-4040` (the call: `mmq::quantize_scatter(xsrc, m.slot_dev, m.src_dev, m.Xq, …, T, K)`),
  kernel `quantize_mmq_q8_1<DS, scatter=true>` (third_party/llama.cpp quantize.cu:606-640, body
  :458-560: grid over **tokens**, "one physical row per token", write loop `nwrite = n_expert_used`
  storing the same quantized block to each of the token's K destination rows), parity test
  `tests/cuda/prefill_mmq_scatter_test.cpp` (150 lines).
- Netting against the T15 estimate: the **quantize-compute 10×** and the **fp32 read 10×** (3.36 GB →
  335 MB per layer-chunk) are removed — that was the bulk of the 1.6-2.2%/chunk estimate. Remaining
  measurable residue: (i) the K-way q8_1 **write** amplification — T·K rows × 3,200 B (2560/32 blocks ×
  40 B) ≈ 1.05 GB/layer ≈ 50.3 GB per 32k chunk ≈ ~30 ms at 1.7 TB/s ≈ **0.7% of the 4,325 ms wall**
  [ESTIMATE, upper bound — L2 may absorb part]; (ii) the Xq buffer is sized T·K rows ≈ 1.05 GB
  (dedup-to-T-rows frees ~0.95 GB ≈ +460 slots at the ~2 MiB slot charge ≈ +1.9 pt hit-rate — a
  **decode**-side prize, T17F's slot law). The TODO entry should be re-labeled "compute+read halves
  landed in c675f5b7; write/VRAM remainder priced §3.6" — recorded here so the next agent does not
  re-buy the battery T15 called "best first prefill battery".

### 1.4 T3.1/#29807 snapshot-write fusion — already the tree state (stale queue entry)

- F-216 phase 1 (SVG-CUSTOM.md:7516, landed 2026-10-16): `gdn_step_norm_multi_kernel` gains
  `snap`/`snap_tok_stride` — "verify-mode-only writes of the register state after each column"
  (verify_kernels.cu:337-357-class; wrapper guards at :1177). Arena: verify.cpp:791-816
  (`STRATA_GDN_SLOTS`, 36 layers × max_t × rec_floats); commit promotes snapshot[m-1]
  (verify.cpp:1265) with a bit-parity self-check (:1276, "zero mismatches = promote is bit-identical").
- F-216 phase 2 verdict (SVG-CUSTOM.md:7533): **BUILT, BIT-PARITY PROVEN, PERF-NEGATIVE** — "the replay
  is NOT the depth gate". So both halves of T6.10 are resolved: the fusion exists (no separate cpy pass
  on the verify path), and the snapshot-promotion scheme that would have consumed it lost. Prefill's GDN
  recurrence (kernels.cu `gdn_rec_chunked`, the F-224 adopted WY form) writes state per chunk by
  construction and has no snapshot pass. **T6.10 should be struck from the queue** — recorded here.

### 1.5 The acceptance stack at HEAD (anchors; the B-lens substrate)

| anchor | file:line at HEAD | status |
|---|---|---|
| `STRATA_TYPICAL_ACCEPT` threshold (default 0.0 if unset; launcher sets 0.2) | generate.cpp:354-360; launch-custom-iq3s-8081.sh:19 | ✓ |
| The typicality rule: accept draft iff `q > min(1, τ·exp(−H))`, H = full-vocab **plain-temperature** softmax entropy, no top-k/top-p replication ("slightly more permissive" than the buun original, in-code comment) | generate.cpp:405-412 (typical_accept), :368-403 (typical_stats) | ✓ |
| Device arm `STRATA_TYPICAL_DEVICE=1` → `row_typical_stats` kernel (2-float D2H) | verify.cpp:4311-4321; verify.hpp:108-110/:663 | ✓ |
| Rescue call shape: serial lane only, first rejection, **up to 2 rescues/window**, sampled traffic only, suffix windows excluded; after rescue the exact-match walk continues; RC1 commit-alignment fix present | generate.cpp:15939-15960 | ✓ |
| Batch-site rescue `STRATA_BATCH_RESCUE` **default OFF** (F-30b: MMLU 95.7 → 91.4-92.9 when armed) | generate.cpp:362-366; launcher :40 | ✓ |
| `STRATA_SPEC_RATIO` ratio_accept: `u < min(1, q/p)`, dedicated mt19937 stream; **in-code comment claims "Preserves the target distribution EXACTLY"** | generate.cpp:415-433 | ✓ — and the claim is FALSE, see below |
| `STRATA_SPEC_PROB` rejection-sampling path, hookup live (`set_spec_q` gated on `mtp.prob() && !from_sfx && !first_window && T_mtp > 1`); sub-knobs `_GATE`, `_DT`; never measured on this stack | verify.cpp:3211; generate.cpp:15910; spec_prob.hpp:43-62 | ✓ in-tree, unmeasured |
| Serial min-p flat gate + spec trace emission | generate.cpp:15695-15701 (gate), :15967-15971 (STRATA_SPEC_TRACE) | ✓ (T37 anchors hold) |
| T31 trace facts (banked): offered-denominator acceptance **0.7796** rescue-inclusive; mean C 2.7344 → **2.4458 exact** (−0.289 tok/w = −10.6% honesty gap); 391/1,615 windows (24.2%) carry ≥1 rescue; 53.4% of T=6 windows cap-hit; dprob monotone rising across the trace (57→84% class) | T31-keeprefit.md §1-3; campaign/results/spec-trace.txt | ✓ |
| Launcher comment: TYPICAL_ACCEPT "R5: worth +20.6pts accept / +8% M1"; TYPICAL_DEVICE "F-200 … +9.0/+6.7/+6.4 vs interpolated control"; KV_PREFETCH "+5.3%/4.1% prefill at 32K/64K" | launch-custom-iq3s-8081.sh:19/:21/:43 | ✓ — all three pre-COHORTS/pre-elastic/pre-F-263 pricing |

**The SPEC_RATIO exactness claim is false (new arithmetic, record it in-tree when next touched).** At a
rescue site the target pick y ~ p was already drawn (window counter); the draft x ~ q is independent
(coupled counters). P(emit v) = p(v)(1−q(v)) + p(v)q(v) + q(v)(1−p(v))·min(1, p(v)/q(v)) = **p(v) +
(1−p(v))·min(q(v), p(v))** — strictly over-emits the draft. It is *more* principled than the typicality
heuristic (bounded by the ratio, deterministic when q ≥ p) but it is output-bending, exactly as T15C's
inventory line ("output-changing") says and the in-code comment denies. The only distribution-exact
rescue-shaped lever in the tree is SPEC_PROB at the tip (residual draw included), per T17C §2.4/§5.3.

### 1.6 Remaining prefill-path anchors

- **Chunk policy**: `--prefill auto:32768` → `prefill_auto_max` clamped to {8192, 16384, 32768}
  (generate.cpp:2653-2670); auto-cap `host_limits::auto_chunk_cap` (#1630) may lower it
  (:2669-2670/:7755); per-session own chunk **2048** (:3152); `stream_all_min()` = 6,144 tokens is the
  routed-ring vs full-stream boundary (prefill.cpp:1242). 120k prompt = 4×32k chunks; ≤32k prompt =
  1 chunk. `tools/calibrate.py` (26.8 KB, present) measures chunk b/a (fixed cost per chunk) and
  `--pcie-frac` (7 values vs our hand-set 0.35) — **never run on this box** (T25-4 OQ2, still open).
- **PLE read path**: pinned, double-buffered, event-tracked (`ple_emb_host[2]`/`ple_copied[2]`,
  prefill.cpp:964-971, alloc :1487-1490; marks :3067-3111) — the next chunk's rows already read beside
  the current chunk's compute. The *concurrent-read-beside-windows* question is separate and refused at
  HEAD by config: `concurrent_read_off()` (include/strata/program/batch_read.hpp:45-57) returns
  **"--pcie-mode dma"** (our standing mode) among {STRATA_PREFILL_HELP, STRATA_SPLIT_MTP_BATCH,
  STRATA_FS_SLOTS, helper caches, staged source}. T37 C2b's contract (pcie-mode kernel/direct on the
  reader path) re-verified verbatim; T25-3's structural refusal stands at HEAD.
- **KV write path**: `STRATA_KV_PREFETCH=1` standing (launcher :21; adoption numbers +5.3%/4.1% at
  32K/64K are 32k-class only); mechanism prefill.cpp:2576-2594 + `kv_prefetch_after` :2832 (the next
  streamed layer's KV prefix staged on its own stream; `kv_drain` guard). Value at 131k **unmeasured**;
  the 96→120k marginal (+20%, 174 vs 145 µs/tok) bundles qsa-select growth + KV staging of pre-32k
  cells (T26-3 §d) and has never been decomposed per-chunk (T26-3 f.4 unknown — no STRATA_PREFILL_TIMING
  row exists for a >32k chunk).
- **W4A4/W4A8**: `fp4_activations(t)` requires `t == GGML_TYPE_NVFP4` and `nvfp4_mode() == W4A4`
  (moe_mmq.cu:350-360; dispatch :382/:481) — the activation quantizer and product path are keyed to the
  expert weight type being NVFP4. **Not armable for the IQ3_S pack**; they are pack-lane machinery
  (T25-4 #13 policy: MMQ default, W4A4 ≥1k chunks, recorded for the NVFP4 line). The tasking's "are they
  armable for prefill shapes?" — no, and no env can arm them without the NVFP4 pack.
- **PROMPT_DIRECT**: in-tree, opt-in, `STRATA_PROMPT_DIRECT=1` at the model-update read path
  (generate.cpp:8416-8423, `open_prompt_direct`); documented in docs/QUANTS.md:136 ("Linux, opt-in reads
  the prompt …"); never armed here (T15C:718). Xolotl's two-box record: 2× on SSD-bound, LOSES on a
  big-file-cache box (4K prompt −15%) — our 184 GB RAM holding the 85 GB pack warm is that case.
- **Standing scoreboard context** (records to beat, pooled): M1 215.2 / pair 206.4 / conc 323.0; M3b
  76-78 band; 9-conc acceptance 77.98% (F-273). ±5-8% single-boot noise on pair/conc (F-278 lesson b);
  the F-263 five-boot pooled law is the only judge for sub-4% claims.

---

## §2 EXTERNAL SWEEP (confidence-graded; primary source on every claim)

**[CORROBORATED] MoE prefill kernel frontier 2025-26 = persistent megakernels, EP-class.**
FlashMoE ("Fast Distributed MoE in a Single Kernel", NeurIPS 2025,
github.com/osayamenja/FlashMoE + OpenReview EZfDHprhZM): GPU-resident persistent kernel fusing dispatch,
expert compute, and combine with tile pipelining, up to 5×. UniEP (arXiv 2604.19241, Jul 2026): the whole
computation+communication graph as one MegaKernel. FlashInfer issue #3692 (Jun 22, 2026): MegaMOE
support proposal (NVFP4/MXFP8/FP8-BlockScale/BF16). FlashInfer v0.7.2 (Oct 2026, flashinfer.ai/releases):
MoELayer via cuDNN backend for Rubin. **Transfer verdict: none of these address our problem** — all
assume expert/data parallelism with an All2All or dispatch network to fuse; our single-GPU offload box's
"dispatch" is the host route + group gather and its "combine" is the hits fold (dm/fused). The
transferable atom — one launch per expert group — is already our GatherGroup + grouped `mmq::Product`
(prefill.cpp:4393-4595). Recorded so nobody ports a megakernel shell (law 12).

**[CORROBORATED] llama.cpp MoE prompt-processing state.** Issue #29130 (open, 2026): "slow prompt
processing when MoE experts are read from disk via mmap", O_DIRECT proposed — external validation that
the expert-read path binds short prompts in the upstream family, and that the O_DIRECT class (our
PROMPT_DIRECT) is the community's first tool for it. v0.7.0 release set (Oct 2026, releases page +
SourceForge mirror): a GPU cache for MoE experts kept in host memory (our MAGMA complement + resident
cache predates and goes further), LiliCorr speculative decoding, GLM5-Next MTP. Discussion #19316 (Feb
2026): hybrid quantization with ternary compute for offloaded MoE — an ideas-stage proposal; no measured
content. Nothing past T3's set is adoptable for us (law 12; our third_party tree is already ahead of
upstream on the MoE prefill path — the scatter-quantize in our tree is ours).

**[CORROBORATED] MTP × chunked-prefill interaction.** Spheron MTP deployment guide (Jun 2026) and vLLM
docs/bench pages: MTP + chunked prefill "same tokens/s, TTFT −70-80%" via prefill-decode overlap; vLLM's
Qwen3-Next recipe still ships `--no-enable-chunked-prefill` with MTP (T17F #47 carried). Our separated
prefill/decode window architecture has no such interaction to manage; the citation class is an
interaction warning for others, not a defect or an opportunity here.

**[CORROBORATED] Skinny-GEMM inefficiency is shape physics.** NVIDIA Deep Learning Performance Guide
(docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication): tile quantization —
small K (or M/N) leaves tensor-core tiles underfilled; efficiency falls with K. Corroborates the §1.1
arithmetic (down product 83.5 vs gu 197 TFLOPS-equiv at K=640 vs 2560): the down line's binder is shape,
not the accumulator pipe — which is one more reason the fp16-accum family was never going to move it.

**[CORROBORATED, NEGATIVE] Adaptive/entropy-conditioned acceptance rules are lossy.** AdaSD (arXiv
2512.11280, Dec 2025, abstract fetched): adapts both draft length and the acceptance criterion using
token entropy + Jensen-Shannon distance; up to 1.46× over vanilla SD **with admitted accuracy loss
(<1.8%)**. Same failure class as FLy (T17C #43). Under our exact-match contract with a quality floor
never traded, any acceptance bending is F-30b-priced (batch rescue: MMLU −3 to −4 points); these do not
transfer. The offers-side entropy work (AdaEDL/DISCO/SVLP class) is already harvested — our dprob gate +
keep rule consume the same signal for free (T17C §4).

**[CORROBORATED] RL-for-speculation is offers-side.** Learning-to-Draft / "Adaptive Speculative Decoding
with Reinforcement Learning" (arXiv 2603.01639, ICLR 2026; code github.com/zhzihao/Learning-to-Draft):
RL over the draft-and-verify cycle, two co-adaptive policies. Carried at T37 S13's verdict: a trained
policy pays only where per-draft confidence is not already free — ours is (dprob). ReSpec (MLSys 2026)
and Together's DAS (Apr 2026) are spec-for-RL-rollouts (lossless for rollouts) — different problem,
context only. AdaEAGLE (2412.18910) carried unchanged (trained draft-length predictor; the re-opener
remains T26-2 §c.3's within-bin-spread test).

**[CARRIED] The acceptance-multiplier framing is field-validated.** "Speculative Decoding: Performance
or Illusion?" (arXiv 2601.11580v2): verification dominates runtime 42-95% across production stacks —
accepted-tokens-per-window is the right multiplier to attack at fixed window cost (lens B's premise);
our captured-window determinism is the asset their batch-varying kernels lack. Nightjar (2512.22420),
Su et al. (2310.18813), Red Hat vLLM (2026-04-16), Medusa (2401.10774 §2.3.1 — the τ·exp(−H) form our
rescue implements), ERSD (2504.15475), SpecDec++ (2405.19715): all carried from T17C/T31/T37 with their
verdicts intact; nothing fetched this session re-prices them.

Source count this section: 12 external primary/engineering sources fetched or search-verified this
session + 11 carried from the T37/T31/T17C/T26 waves (re-checked against this dossier's claims), plus
~30 local code/record sources (§1, §7).

---

## §3 IMPLEMENTER-READY CANDIDATES (value-ranked; mechanism → band vs pooled records → cost-anatomy → falsifier-first → law-4b → rollback)

Value-ranking under the user's bar (past-the-noise wins, prefill the #1-stated metric): the honest
headline is that **no ≥4% prefill lever exists in-tree at HEAD**; the prefill items below are
reads/close-outs that either find the next lever or retire questions cheaply. The acceptance items carry
the larger priced bands but are M1-lane and mostly ≤1% exact — their value is re-pricing a stale stack
before the retrain (T31: several of these levers die at T14B; measuring them now is the only chance).

### 3.1 (Acceptance, rank 1 by band-at-stake) — THE RESCUE-STACK RE-PRICE: one τ sweep + the counters already in the tree

**Mechanism.** τ=0.2 was set by the R5/buun era ("chat acceptance 0.44→0.73 at threshold 0.2, temp 0.7"
— generate.cpp:347-353 comment) and has never been swept on this stack (T15C:68: "R5-era value; not
re-swept post-DT-0.85/post-TYPICAL_DEVICE"). Since then: DT 0.85 (sharper q → different rescue-site q),
TYPICAL_DEVICE (the rescue got cheaper per probe — up-to-2 rescues affordable), depth 6 standing,
rescue-inclusive acceptance 77.96% vs exact 65.0% (T31). Lower τ = looser = more rescue inflation
(F-178's direction law: lower tau = looser); higher τ = tighter = closer to exact-match. The sweep
prices the stack's CURRENT value and its sensitivity in one battery.

**Band vs records.** The stale claims: +20.6 pts accept / +8% M1 (launcher:19), F-200's +9.0/+6.7/+6.4
vs *interpolated* control in a deep-trough window — the weakest measurement class this campaign
produced, pre-F-263. T31 bounds the stack's current size: the rescue lifts mean C by +0.289 tokens/window
(2.446→2.734). At 4.71 ms/token that is ~1.36 ms/window ≈ **+11-13% M1 IF it were free** — it is not
free: it is output-bending on 24.2% of windows, and the only quality read of the bending class (F-30b,
batch site) cost −3 to −4 MMLU points. The realistic question the sweep answers: is τ=0.2 still the
right point on the bending/value curve at DT 0.85, or has the optimum moved toward 0.15 (more value,
more bending) or 0.30 (less bending, most of the value — T31's cut decomposition says rescue mass
concentrates in low-d windows)? Expected movement: flat-to-±2% M1 across {0.15, 0.25, 0.30} [ESTIMATE];
the deliverable is the re-priced curve, not a win.

**Cost anatomy (F-232).** Zero new code. Instruments already banked: `typical_probes`/`typical_rescues`
counters (generate.cpp:15943/:15958), the M3b probe, the trace instrument (resc field; T31's methods
note asks for the per-window resc DELTA next boot — add nothing, read it). The rescue path is
serial-lane only at standing → conc structurally unaffected (law-4b clean by construction).

**Falsifier-first.** The sweep IS the cheapest falsifier: three one-boot arms τ ∈ {0.15, 0.25, 0.30} vs
standing 0.20, same window, M1 + M3b + rescue counters + one STRATA_SPEC_TRACE leg for the resc-delta.
MMLU gate MANDATORY on any arm that moves M1 (the bending class is exactly F-30b's; law 10: the canary
is the contract). Verdict rules set in advance: τ* stays 0.2 unless an arm clears +2% pooled five-boot
AND passes 2× MMLU; otherwise the record gains the curve and the launcher comment's stale "+8% M1" is
corrected.

**Rollback.** `STRATA_TYPICAL_ACCEPT=0.2` (or unset for exact-match-only serial lanes — also a
legitimate arm: how much of the +0.289 tok/w survives at all under the F-263-era stack, at zero
bending).

### 3.2 (Conc/acceptance, rank 2) — THE BATCH-LANE TRACE TWIN (the acceptance instrument the conc lane lacks), then T37 C4's second-draft boot

**Mechanism.** T37 left it open: "the batch lane has no STRATA_SPEC_TRACE equivalent" (T37 open q.3).
The cohort tail already prints per-round (`"cohorts: windows done"`, generate.cpp:12085; per-cohort
finish :12104-12189 runs the single-draft exact match `outb[i] == w.tok[i+1]`); it does not emit
per-window (T, a) rows. One fprintf in the cohort finish — the C5 prescription applied to the spec
trace — gives the batch lane's acceptance denominator: T (=1 draft + seed per slot today), a, and
per-slot identity. Without it, "9-conc 77.98% = highest measured" (F-273) cannot be compared to the
serial lane's 77.96%-rescue-inclusive / 65.0%-exact (§1.5) — the numerators ride different rules.

**Band.** No direct t/s; it gates (a) T37 C4's second-draft-row boot (EV +6-12% conc arithmetic
[HYPOTHESIS]: pos-1 conditional 0.84 × 4.7 ms/token − c_row 2.4 ms = +1.55 ms/slot-window), whose
acceptance input is exactly what the twin measures — on the batch lane the pos-1 conditional is
**unmeasured** (the serial trace's 0.837 is prefix-conditioned on a different offer process); and (b)
every future "acceptance under load" claim (the F-273 class). This is C5-rank-0 logic for the spec lane.

**Cost anatomy.** One fprintf in the cohort finish tail next to the BT lines (:12143-class); file append
behind the same `STRATA_SPEC_TRACE` env (or a `STRATA_BATCH_TRACE` twin). The per-slot rows exist in
`w`/`outb` at the finish; zero new state.

**Falsifier-first.** The first 9-wide telemetry leg (already owed post-C5, F-278) reads both. If the
batch lane's exact acceptance at 1 draft/slot reads ≥ pos-0 serial (0.879), the "loaded lanes accept
better" story is a position-mix artifact and the second-row EV must use the twin's own conditional, not
the serial table — possibly killing C4 for one fprintf's price. If it reads low, the second-row boot
goes with a real prior.

**Law-4b / rollback.** Diagnostic, not an arm; default-off behind the env; one-line revert.

### 3.3 (Acceptance, rank 3) — STRATA_SPEC_PROB + tip-k: the only exact lever, still never measured (carried, sharpened)

T17C #5/#6 priced it: +0.4-0.8% M1 class, distribution-exact at the tip (ERSD Lemma 3.1:
Σmin(p,q) ≥ D_HM), in-tree end-to-end (§1.5 anchors), one env line + battery; DT re-optimization at the
rejection optimum (q = p ⇒ DT 1.0, not the coupled-optimal 0.85) rides the same boot. Sharpened this
session: **SPEC_RATIO is not the exact alternative** (§1.5 arithmetic — over-emits (1−p)·min(q,p) mass),
so if the rescue stack is ever to be replaced by something principled, SPEC_PROB is the only in-tree
form that is actually exact. Sequence: ride the 3.1 battery (same boots, second arm) — both levers die or
re-price at the same T14B milestone, so they should share boots now. Falsifier: the battery's own M1/dp
read; kill on any conc drag (T17C's rule). Rollback: unset (`STRATA_SPEC_PROB` off; `_GATE`/`_DT`
sub-knobs documented at spec_prob.hpp:43-62).

### 3.4 (Prefill, rank 1 by cost) — RUN tools/calibrate.py: the chunk-policy and pcie-frac read that has waited two waves

**Mechanism.** The chunk policy (`auto:32768`, clamps at generate.cpp:2653-2670; per-session 2048;
`stream_all_min` 6,144) has never been measured against this box's b/a (fixed-per-chunk cost), and
`--pcie-frac 0.35` is hand-set, never measured (T26-3 (e) verdict: "run it before any T-3 port work";
T25-4 OQ2). Post-elastic the VRAM constraint moved (2.91 GiB freed; the auto-cap may now allow a larger
chunk or a different buffer split), which is exactly the "still optimal post-elastic?" question.

**Band.** Unknown by definition — but it prices three queued items at once: the 96→120k +20% marginal
decomposition (with 3.7's boot), the staging-ring port decision (T25-4 #5), and the DMA/copy-stream
split. A better chunk size or pcie-frac is plausibly +0-3% prefill [ESTIMATE]; the read is the point.

**Cost anatomy.** Zero-port; the tool exists (26.8 KB), takes ~1 h, launches its own engine instances
(T26-5 trap note: respect the 16 GB guard, fence against the live server). Learned-profile hazard: back
up / pin `data/expert-profile-learned.bin` for both sides (F-271 law).

**Falsifier-first / rollback.** It IS the falsifier; adoption of any knob it moves goes through the
normal ±4%/five-boot law. Rollback = config unchanged (nothing standing depends on it).

### 3.5 (Prefill, rank 2 by cost) — KV_PREFETCH at 131k + the >32k chunk-timing row: decompose the +20% marginal

**Mechanism.** KV_PREFETCH's adoption numbers are 32K/64K (+5.3%/4.1%); at 131k the streamed KV term is
larger (kv-resident 32,768; everything past that stages from RAM — "KV streaming: 344.8 MiB read from
RAM" boot lines, T26-3 §d) and the 96→120k marginal grew +20% with the growth *attributed* to qsa-select
+ KV staging only by inference, never measured per-phase (T26-3 f.4: "no banked row exists — timing env
was off in every ladder boot").

**Band.** If kPfKvStage is ≥10% of the 120k marginal, the staging term is a real overlap target (and
KV_PREFETCH's 131k value prices itself); if it is small, the marginal is qsa-select growth — cap-limited
by design (top-2048) and T26-3's "attention-family work only matters past ~120k" gets its number.
Either way a plateau map edge. +0-2% prefill at 120k if the staging term is actionable [ESTIMATE].

**Cost anatomy.** One boot: standing + `STRATA_PREFILL_TIMING=1` + the ctx ladder to 120k, then the same
with `STRATA_KV_PREFETCH=0` (the paired read). The PfTimer rows (kPfKvStage, kPfQsa*, prefill.cpp:2478-
2483) decompose it; no new instrument.

**Falsifier-first.** The boot IS the falsifier (T26-3's unknown f.4 closed for one boot's price).
**Rollback**: unset the timing env; KV_PREFETCH stays standing either way (its 32k/64k record stands).

### 3.6 (Prefill/decode, rank 3) — the scatter-quantize residue: ncu the writes, then decide the ids-indirection port

**Mechanism.** §1.3's residue: 10× q8_1 write amplification (~50 GB/32k-chunk) + 1.05 GB Xq buffer. The
full form (quantize T rows; the MMQ product reads via ids indirection — upstream's original MoE read
shape) would cut both, worth ~0.7% prefill wall [ESTIMATE, upper bound] **plus ~0.95 GB VRAM → ~+460
slots → +1.9 pt hit-rate** (T17F slot law: +512 slots = +2.1 pt) — i.e., the decode-side prize is
probably larger than the prefill-side one. But the indirection lands in the MMQ product tile loaders —
the hot loop of the 197-TFLOPS-equiv gu line — which is exactly the register file the standing kernels
win with (T24's dense_tc lesson: a "cleaner" addressing form lost on this stack).

**Cost anatomy.** Sites all mapped: writer quantize.cu:458-560 (scatter loop `nwrite`), consumer
`mmq::Product` bounds/ids machinery (moe_mmq.hpp:57-75 — `ids`/`bounds` already parameterized; the
grouped launcher is the delta), Xq sizing prefill.cpp:4012-4013. Effort M, risk concentrated in the
product kernels.

**Falsifier-first (cheap).** One ncu memory-workload analysis on one layer's `quantize_mmq_q8_1<scatter>`
at a 32k chunk: if the K-fold writes mostly hit L2 (the 10 destinations of one token are written by the
SAME warp back-to-back — likely high L2 absorption), the DRAM-write term collapses and the prefill half
dies; the VRAM half can then be priced alone (keep T-sized Xq + indirection ONLY if the +460 slots clear
the hit-rate battery on their own). If ncu shows real DRAM writes, the port prices M with both prizes.

**Law-4b / rollback.** Would be env-gated (`STRATA_QUANT_GATHER` already reserves the env grammar);
rollback = standing scatter form.

### 3.7 (Prefill short-prompt, rank 4) — STRATA_PROMPT_DIRECT close-out: price it honestly for OUR box, then shelf it

**Mechanism.** T25-4 #6 carried the arm and the risk: Xolotl's own two-box record says O_DIRECT prompt
reads 2× on an SSD-bound box and LOSE on a big-file-cache box (4K prompt −15%). Our box: 184 GB RAM vs
the 85 GB pack — after any warmup the pack IS the page cache; and the 1,350-tok lane's measured 43.7%
wait_copy is the copy stream DMAing from the pinned complement (prefill.cpp:4406 "DMA straight from the
page-locked arena"), not NVMe latency. External llama.cpp #29130 (mmap expert reads slow prompts;
O_DIRECT proposed) keeps the cold-tier form credible — for boots before the cache warms, not for the
standing warm box.

**Band.** Prior: negative-to-flat on the standing warm box; the only upside class is first-boot/cold
(T37's PIN_GUARD=2 cold-start falsifier is the same territory — run them in the same window if at all).
0 to 2× on the short lane [their band], ≈0 expected here [ESTIMATE].

**Cost anatomy.** generate.cpp:8416-8423 (opt-in, logged); one t17-leg A/B at the 1,350-tok probe;
zero code.

**Falsifier-first.** One leg closes the two-wave-old question. **Rollback**: unset. **Disposition rule
proposed**: if the leg reads ≤0 on the warm box, the arm is recorded as cold-boot-only shelf (a labeled
variant, never standing) and T25-4 OQ on it closes.

---

## §4 NEGATIVES RESEARCHED-AND-REJECTED (with reasons; re-open conditions)

1. **K=640 fp16-accum at the standing recipe — dead as specified.** The call site does not execute
   (§1.2); the MMQ down product runs the int8 pipe with H quantized to the down type. The fp16-accum
   family on prefill is now closed end-to-end (blanket KLD 0.15-0.40 dead, T26-5; K≥6144 dead; K=640
   dead-at-standing here; decode dead, T26-3 a.2). Re-open ONLY as a shelf variant for a
   `STRATA_PREFILL_MMQ=0` world, which has no standing user.
2. **The T15 quantize-dedup battery — consumed.** Compute+read halves landed in c675f5b7 (2026-10-14),
   before the T15 estimate was banked. Do not re-run "the best first prefill battery"; the residue is
   re-scoped in §3.6 (write/VRAM, ncu-gated).
3. **T6.10 snapshot-write fusion — stale queue entry.** The fusion IS the tree (F-216 phase 1) and the
   promotion scheme that would have used it is banked PERF-NEGATIVE (F-216 final). Strike from the queue;
   the prefill recurrence never had a cpy pass.
4. **W4A4/W4A8 arming for the IQ3_S prefill — not armable.** Type-keyed to NVFP4 experts
   (moe_mmq.cu:350-360); pack-lane machinery (T25-4 #13). Re-open only if the NVFP4 pack line opens.
5. **MoE prefill megakernel imports (FlashMoE/UniEP/MegaMOE).** EP/distributed problem class; no
   All2All here; our grouped-launch form already occupies the transferable atom. Re-open if the campaign
   ever goes multi-GPU expert parallel (it is chartered not to).
6. **Entropy/JS-conditioned acceptance bending (AdaSD class) — lossy.** Admitted accuracy loss; fails
   the exactness law and the F-30b precedent prices the bending class in MMLU points. Same disposition
   for FLy (T17C #43). Re-open never under the current contract; the contract is a moat (EQSPEC
   evidence, T37 S14).
7. **STRATA_SPEC_RATIO as an "exact" rescue — the claim is false.** Over-emits (1−p)·min(q,p) mass per
   rescued site (§1.5). Not a candidate; needs a one-line comment correction when next touched. If a
   principled rescue is wanted: SPEC_PROB tip-only (§3.3) or nothing.
8. **RL draft-length policies as serve-side arms (LTD/AdaEAGLE/ReSpec).** Carried T37 S13: our per-draft
   dprob is free; re-opener unchanged (T26-2 §c.3 within-bin spread). ReSpec/DAS are rollout-side,
   different problem.
9. **Sarathi-class prefill-decode interference scheduling.** Our prefill/decode windows are separated by
   construction; the interference axis is owned by cohorts (T35 §5, T17F #46/#47 carried). Re-open only
   if a future merge co-batches them.
10. **Chunk-size change as a standing prefill arm before the read.** calibrate.py first (§3.4) — a
    chunk knob moved without the b/a measurement is exactly the "measure the premise before arming the
    lever" law-4 violation.
11. **Re-specifying TC-hc mma / the conditional-graph keep rule — BUILD PAST.** T37-kernel-math §3.1 and
    T37-scheduling-spec C1 own them; this dossier defers, and notes one interaction: TC-hc adoption cuts
    c_row, which lowers the second-draft-row break-even (0.72 → ~0.5 tokens/row class at c_row 1.7) —
    sequence C4's boot after any hc adoption verdict, not before (T17C §5.5's re-opener (i) fires at
    c_row ≲ 1.2 ms).

---

## OPEN QUESTIONS (ranked by what they gate)

1. The batch lane's exact acceptance and per-position conditionals (§3.2's twin) — gates C4's second-row
   boot and every "acceptance under load" claim.
2. τ curve on the current stack (§3.1) — gates whether the rescue stack keeps its standing value or
   re-tunes; and how much of the +0.289 tok/w is bending vs value (the τ=0 arm reads it).
3. kPfKvStage share of the 96→120k marginal (§3.5) — gates the 131k KV-staging lever vs closing the
   attention question at T26-3's cap-limited verdict.
4. L2 absorption of the scatter-quantize writes (§3.6 ncu) — gates the ids-indirection port and the
   +460-slot VRAM play.
5. Does the auto-cap (#1630) currently bind prefill_auto_max below 32768 on this box? One boot line
   ("auto" vs "auto:32768" banner) — if it binds, §3.4's chunk read has a prior.
6. Whether the depth-RISING min-p stair (T17C #4, +0-1% standalone / +2-5% post-retrain) has a cheap
   falsifier at HEAD: the F-228 stair instrument (STRATA_SPEC_DEPTH=1) still exists; one boot reads m_k
   under the F-263-era stack before any schedule is priced. Lower priority than §3.1-3.3 (the keep rule
   C1 subsumes it when it lands).

## IMPLICATIONS

- **The prefill plateau will not be moved by the two candidates this tasking named — both are already
  banked or broken.** That is the finding. The plateau's live doors are, in order: the T14B requant
  (structural, user-GO), the chunk/pcie read (§3.4), the 131k KV decomposition (§3.5), and the §3.6
  VRAM play whose bigger half is decode-side. Everything else on the dense-GEMM line belongs to T37 §3.1.
- **The acceptance lane is where the boot-cheap value sits this wave**: one battery (τ sweep + SPEC_PROB
  riding it) re-prices the oldest un-re-priced stack in the standing recipe and measures the only exact
  lever in the tree; one fprintf (batch trace twin) makes the conc-acceptance story measurable before
  the second-draft-row question spends its boot.
- **Three queue/journal hygiene items ride this dossier**: the T15 PREFILL-#1 re-label (consumed, c675f5b7),
  the T6.10 strike (superseded by F-216), and the T37 §3.4/T26-5 §c K=640 premise correction (phase-name
  misattribution). Each prevents a re-purchase.
- **Sequencing with the retrain (T14B)**: every §3.1/§3.3 lever decays at a successful anti-drift
  retrain (T31 §6: the keep-rule family is exactly 0 by delta ≈ +0.30). Measure now or credit the
  retrain with their value forever after.

## SOURCES

### Local — code (read this session at HEAD 3219b765)
src/prefill/prefill.cpp (:1159-1213 mmq_plan, :2478-2483 phase table, :2576-2594/:2832 KV prefetch,
:347-353 comment, :354-366 gates, :368-433 typical/ratio, :4020-4040 scatter call, :4393-4445 staging/
group gather, :4480-4595 MMQ product chain, :4600-4626 fp16 fallback chain); src/prefill/moe_mmq.cu
(:350-360 fp4_activations, :378-407 quantize/quantize_scatter); include/strata/prefill/moe_mmq.hpp
(:40-75 contracts); third_party/llama.cpp ggml/src/ggml-cuda/quantize.cu (:458-560 kernel body,
:575-640 launchers); src/program/generate.cpp (:699-706/:2653-2670/:3152 chunk policy, :8416-8423
PROMPT_DIRECT, :9340-9361 cohorts, :12085/:12104-12189 cohort tails, :15695-15701 gate, :15910 SPEC_PROB,
:15939-15960 rescue, :15967-15971 trace); src/core/verify.cpp (:791-816/:1265/:1276 snapshots,
:3211 set_spec_q, :4311-4321 row_typical_stats); src/kernels/cuda/verify_kernels.cu (:337-357-class
snap writes, :1177 guard); include/strata/program/batch_read.hpp:45-57 (concurrent_read_off);
include/strata/core/spec_prob.hpp:1-62; tools/calibrate.py; launch-custom-iq3s-8081.sh:19-45;
custom-iq3_s-8081.json; tests/cuda/prefill_mmq_scatter_test.cpp.

### Local — record (journal, dossiers, results, git)
SVG-CUSTOM.md: F-200 (:6903, device rescue +9.0/+6.7/+6.4 vs interpolated control), F-216 (:7516/:7533
snapshot fusion + perf-negative), F-228 (:7854, the depth stair + per-depth acceptance), F-230 (:7886,
dense-MMQ negative + the prefill acceptance-tax law), F-277/F-278 (:9240/:9310); SVG-CUSTOM-TODO.md
(:160/:451 T3.1-T6.10 queue entries, :1027 the T15 PREFILL-#1 item); T15C-overlooked-wins.md (:67-77,
:718, :793-798 the env inventory incl. TYPICAL_ACCEPT "not re-swept" and SPEC_RATIO "output-changing");
campaign/research/T37-kernel-math.md §3.4, T37-scheduling-spec.md (C2b/C4/C5, S10), T26-3 (a.1, d, e,
f.4), T26-5 (§b-§c), T25-4 (#5/#6/#13, OQ2), T25-3 §2.1, T17C (§2.4, §5.3, #4-#6, #43), T17F (slot law,
#46-#47), T31 (§1-§6 the trace refit); campaign/results/knob-pft-engtail.txt (F-239 anatomy, Oct 19);
campaign/results/spec-trace.txt (1,615 windows); git: c675f5b7 (2026-10-14 scatter-quantize), 848f73a9
(import), 61068748 (F-252).

### External (fetched or search-verified this session unless marked CARRIED)
1. FlashMoE — NeurIPS 2025, github.com/osayamenja/FlashMoE + openreview.net/pdf?id=EZfDHprhZM.
2. UniEP: Unified Expert-Parallel MoE MegaKernel — arXiv 2604.19241.
3. FlashInfer issue #3692 "Add support for MegaMOE" (Jun 2026) + flashinfer.ai/releases (v0.7.2).
4. vLLM Fused MoE modular kernel docs — docs.vllm.ai/en/stable/design/fused_moe_modular_kernel.
5. llama.cpp issue #29130 (slow MoE prompt processing via mmap; O_DIRECT proposed).
6. llama.cpp releases page + v0.7.0 set (host-MoE-expert GPU cache, LiliCorr, GLM5-Next MTP);
   discussion #19316 (ternary-compute MoE proposal).
7. Spheron, "Multi-Token Prediction on GPU Cloud" (Jun 2026) — MTP × chunked-prefill TTFT claims.
8. vLLM bench/MTP docs — docs.vllm.ai (E2E latency, over-admission guards).
9. NVIDIA Deep Learning Performance Guide (matrix multiplication; tile quantization).
10. AdaSD: Adaptive Speculative Decoding — arXiv 2512.11280 (abstract fetched; lossy).
11. Learning to Draft / Adaptive SD with RL — arXiv 2603.01639 (ICLR 2026) + code repo.
12. ReSpec (MLSys 2026, tianweiz07.github.io/Papers/26-mlsys.pdf) + Together DAS blog (Apr 2026).
13. CARRIED (verified in T37/T31/T17C/T26 waves, re-checked): Nightjar 2512.22420; Su et al. 2310.18813;
    Red Hat vLLM spec blog 2026-04-16; Medusa 2401.10774; ERSD 2504.15475; SpecDec++ 2405.19715;
    AdaEAGLE 2412.18910; EQSPEC/EXSPEC 2510.22876; Speculative Decoding: Performance or Illusion?
    2601.11580; NanoFlow 2408.12757; MagicDec 2408.11049; vLLM Qwen3-Next recipe (--no-enable-chunked-prefill).

Count: 12 fresh external + 12 carried external + ~30 local code/record sources.

## METHODS NOTE (including unproductive searches)

Productive: reading the scatter kernel BODY (not the call site) — the "quantized once, written K times"
pattern is only visible in `quantize_mmq_q8_1`'s write loop, and it is what split the T15 item into
consumed vs residue; `git log -S quantize_scatter_mmq_q8_1` dating the dedup to 2026-10-14 (before the
T15 estimate), which converted a candidate into a correction; walking prefill.cpp's phase MARKS
(kPfGemmGU/kPfGemmD/kPfDequant) through BOTH product branches to discover the F-239 names are
path-agnostic while the T26-5/T37 readings assumed the fp16 branch — the single highest-value read of
the session; the emit-distribution arithmetic on ratio_accept (three lines of probability that falsified
an in-code "EXACTLY" claim and contradicted nothing in the record — T15C had it right); reading
batch_read.hpp's refusal list directly instead of trusting the "structurally refused" summary (it is a
config refusal, not a code one — which is exactly why the falsifier is one launcher line).
Unproductive / dead ends (do not repeat): the ACL 2026 information-gain PDF (WebFetch cannot parse
application/pdf — abstract-level only; treat that entry as search-verified, not fetched); searching for
a consolidated llama.cpp changelog past v0.7.0 (does not exist; the compare view is the tool); "MegaMoE"
as an arXiv title (the object is FlashInfer-issue + FlashMoE/UniEP naming); campaign/results for any
STRATA_SPEC_RATIO or TYPICAL_ACCEPT sweep artifact (none exists — confirmed twice, by grep and by the
T15C inventory note).
Arithmetic honesty: every band in §3 marked ESTIMATE shows its constant; the two corrections in §1.2/
§1.3 are file:line-and-commit grounded, not inferred; the §1.5 over-emission derivation is stated
term-by-term so it can be checked (and, if wrong, corrected) in place.
