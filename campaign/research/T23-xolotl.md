# T23 — Xolotl (Indras-Mirror/Xolotl) recon for the Strata custom-2 campaign

Date: 2026-10-20 · Target: https://github.com/Indras-Mirror/Xolotl (main @ cd965e3c, 2026-10-21 UTC) · Read-only, no builds/boots.
Scope: assess against our setup — Strata custom-2 CUDA fork, Qwen3.8-Flash-Next MoE + MTP spec decode, RTX 5090 sm_120, single box (see docs/research/2026-10-16-strata-moe-quant-memory-offload.md).

**VERDICT: APPLIES** — overall. Xolotl is a sibling of our fork, not a competitor: it is upstream Strata (Niko1221/Strata) merged at 0.1.44.1 plus ~71k lines, and it already serves Qwen3.8-Flash-Next with MTP on sm_120-family hardware. Sub-verdicts: sm_120 CUDA decode APPLIES; MoE serving APPLIES (technique transfer); speculative decoding APPLIES; MTP head training (T14B) DOESN'T (no training tooling exists anywhere in this family — pack/runtime formats are the usable part); GGUF quantization PARTIAL (GSQ pipeline is DeepSeek-only; ecosystem quants + eval harness apply); model lineage APPLIES (documented topology + graft recipe).

## What it is

Standalone repo (not a GitHub fork; history re-rooted), MIT, C++, created 2026-10-06, pushed 2026-10-21, 257 stars, 33 forks, 6 open issues, **no releases/tags** — evaluation is commit-based. Self-description: DeepSeek-V4.1-Flash (552B MoE) at 26.8 tok/s on one RTX 4090 (24 GB) + 90 GB RAM, plus GLM-5.3-Flash, Qwen3.8-Flash-Next (127.8 tok/s) and Qwen3.8-27B dense (98.5 tok/s), one OpenAI/Anthropic server. "A heavily extended fork of Strata (merged at 0.1.44.1)".
Sources: https://github.com/Indras-Mirror/Xolotl (README.md), https://api.github.com/repos/Indras-Mirror/Xolotl, /releases, /commits?per_page=100.

**Lineage (measured, from MERGE_SCOPE.md, 2026-10-16):** exactly two source repos — the Indras fork (branches deepseek4/glm/mimo; root 01529aaa has tree identical to upstream `f2a08d4b`, 2026-09-24) and upstream Niko1221/Strata (29.5k stars; branches main + qwen + dense-eval). Histories are unrelated except via `git replace --graft 01529aaa f2a08d4b`; intra-fork merges have a single base fb58e0db; the cross-lineage merge to newest Qwen is a real 47-48-file merge (4 files conflict in both merges: `src/kernels/cuda/iq_kernels.cu`, `src/kernels/cpu/native_expert.cpp`, `src/prefill/moe_mmq.cu`, CMakeLists.txt). docs/UPSTREAM_SYNC.md documents their upstream-tracking discipline: merge-don't-rebase, ordered parity gates, and fork-isolation rules ("new files over in-place edits", fork kernels in `src/kernels/cuda/fork/*.cuh` hooked by one `#include`, default-no-op params inside upstream kernels, identical shared-file resolutions across forks; measured footprint +327/-87 lines inside upstream files).
Sources: https://github.com/Indras-Mirror/Xolotl/blob/main/MERGE_SCOPE.md, .../blob/main/docs/UPSTREAM_SYNC.md, https://api.github.com/repos/Niko1221/Strata.

## Findings by sub-question

**1. sm_120 CUDA decode — APPLIES (high confidence).**
- Upstream Strata is developed on sm_120: CMakeLists defaults `CMAKE_CUDA_ARCHITECTURES=120` with the comment "developed and measured on sm_120 (RTX 50)"; the tf32 mma in the QSA scorer has a portable fp32-FMA fallback so sm_75/86/89/120 all build; pre-sm_60 refused; CUDA 13 dropped sm_60/70. (https://github.com/Indras-Mirror/Xolotl/blob/main/CMakeLists.txt, lines ~164-190)
- Community RTX 5090 D v2 (sm_120, Windows) build report, issue #14 (2026-10-19): CUDA 12.8 **cannot** build sm_120 — ptxas rejects the MXFP4 fork kernel's block-scale mma (`.kind::mxf4 not supported on .target 'sm_120'`); fix = **CUDA 13.1 + `-DCMAKE_CUDA_ARCHITECTURES=120a`** (plain `120` fails the same way); VS "18" BuildTools rejected as nvcc host compiler — pin VS 2022 vcvars64. Load 152 GB in 49.7 s, 72 GiB arena, 14.38 GiB expert slots. (https://github.com/Indras-Mirror/Xolotl/issues/14, mirrored in docs/WINDOWS.md)
- README: "use CUDA 13.x for RTX 50; 12.8 builds had prompt-path crashes". Conversation-pool feature benchmarked on a 5090: three agents at 30-60K tokens, 47.5 -> 3.96 s/turn. (README.md)
- Consistent with our prior finding that Blackwell block-scale MMA needs the `a`-suffix target (docs/research/2026-10-16 file, item 3). Contradiction check: none — the two ptxas failures (compile-time, needs 120a; runtime crashes on 12.8) are complementary, both resolved by CUDA 13.1 + 120a.

**2. MoE serving — APPLIES (high confidence for technique transfer; our Qwen path itself is upstream Strata's).**
- Three-tier expert placement (VRAM LRU sized from free VRAM / pinned-RAM arena via cudaHostRegister + DMA / NVMe O_DIRECT ~5 GB/s, prefetched one layer ahead); conversation-adaptive arena rebalancing in NVMe idle time (file-tier reads -21..-30% per small agent turn, byte-checked slots); expert-cache admission fixes; `--remote-device` second GPU holding extra experts with route-bias; batch slots (#22) running N conversations per pass; conversation pool (#19/#20) resuming from pinned host RAM at turn boundaries; long-prompt OOM fixes past 512K (#21). (docs/xolotl/HOW_IT_WORKS.md; README.md "Recently added"; commits e56cfd43, 26aa0da7, 6d77e18d)
- For Qwen3.8-Flash-Next specifically Xolotl ships upstream Strata 0.1.44 behavior (MTP, int8 KV, vision); the tiering work targets the ds41/GLM engines — but the shared core (`src/{core,kernels,prefill,spec}`) is the same code our fork carries. (docs/xolotl/HOW_IT_WORKS.md; MERGE_SCOPE.md §2)
- Maps onto our prior recommendations (admission control, packed H2D, predictive prefetch, MTP+cache composition): Xolotl is a working reference implementation of most of that list, on the same engine base.

**3. Speculative decoding — APPLIES (high confidence).**
- Qwen models: MTP speculative decoding; dense Qwen3.8-27B uses draft 3. Flyweight MTP GGUF machinery: `tools/mtp_fetch.py` pulls the 31 `mtp.*` tensors from Qwen/Qwen3.8-Flash-Next BF16 (360 GB, 131 shards) via HTTP Range + per-tensor SHA256 at a pinned revision (HF mirrors that ignore Range are detected); `tools/mtp_pack.py` packs the draft block as Strata GGUF (dense BF16 ~0.18 GB, routed experts q2_0/q4_0/q8_0 — RTN, format chosen by **measured draft acceptance**); `tools/mtp_rt.py` writes the engine blob layout (experts.bin interleave, dense Q8_0, Gemma-style +1 RMSNorm); `tools/draft_vocab.py` builds draft-head token subsets (fixes #137: shipped EN+code subset held 27 of 55,328 Han tokens so Chinese drafted almost nothing; `--add cjk` / corpus-coverage mode); `tools/split_mtp_batch_parity.py` parity harness for STRATA_SPLIT_MTP_BATCH (draft KV via MMQ, target compared bit-exact, acceptance within ±3 pts). (tools/mtp_*.py, draft_vocab.py, split_mtp_batch_parity.py docstrings)
- DeepSeek side (design reference for our verify path): DSpark draft with **exact speculative sampling + confidence-gated verify**, cached verify-width graphs, draft parked off-card during long prompts. (README.md; docs/DEEPSEEK_V41.md referenced)
- Note: these MTP tools are **upstream Strata tools** (present in Niko1221/Strata tools/, 161 files), inherited by Xolotl at 0.1.44.1 — check whether our custom-2 base already has them before treating them as new. (https://api.github.com/repos/Niko1221/Strata/contents/tools)

**4. MTP head training (our T14B retrain) — DOESN'T (low-moderate confidence in the negative; high confidence nothing is there).**
- No training tooling anywhere in the tree. mtp_fetch/mtp_pack explicitly "Nothing here runs a model" / "No model runs here" — they consume the upstream-published MTP head. No PyTorch trainer, no LoRA/drafter training, no loss code. (tools/ tree listing; tool docstrings)
- Usable for T14B anyway: (a) the pack + rt blob formats are the runtime contract a retrained head must satisfy — and llama.cpp's qwen4exp converter drops the MTP head (`supports_mtp_export = False`), so we own the packer regardless; (b) pick draft-expert format by measured acceptance, not reconstruction error (their stated method); (c) draft-vocab restriction (draft_vocab.py) is a cheaper lever than retraining, with a known CJK failure mode; (d) their acceptance/parity harnesses (split_mtp_batch_parity.py, "drafts accepted X of Y" log) are the eval rig for a retrained head. (tools/mtp_pack.py, tools/draft_vocab.py)

**5. GGUF quantization — PARTIAL (medium-high confidence).**
- Xolotl's own GSQ pipeline (tools/v41/v41_calib.py, v41_quant.py: layer-streaming calibration through DeepSeek's model.py, per-expert GPTQ tier screening PTQ1_0-ternary/Q2_0/Q3_K on held-out sequences, sequential propagate, in-place GGUF write) is **DeepSeek-V4.1-specific**; porting to Qwen = writing a new model chain (real work). The transferable design: layer-streaming from safetensors, held-out screening, acceptance of the mixed map vs all-Q2_0 control. (tools/v41/v41_quant.py, v41_calib.py docstrings)
- Ecosystem already solves Flash-Next GGUF quants: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (6.1M downloads) + Coder/abliterated/Swift variants, unsloth UD k-quants, nvidia NVFP4 + FP4-Local, exl3, oQ4e-mtp. (https://huggingface.co/api/models?search=Qwen3.8-Flash-Next)
- docs/QUANTS.md: per-format kernel matrix for unsloth GGUFs (which formats get CUDA grouped-expert decode, MMQ prompt path via opt-in `STRATA_MMQ_KQ=1` + separate libstrata_kq.so, CPU AVX2) + `tools/quant_quality.py` teacher-forced scoring vs Q8_0 (top-1 agreement, KL, PPL; UD-Q6_K_XL 94.61% top-1 / KL 0.033 ... IQ3_XXS 84.46% / KL 0.202) — harness applies to any Strata-served GGUF, directly usable on sm_120. (docs/QUANTS.md)
- Unexamined lead: tools/nvfp4_convert.py + docs/NVFP4.md — potential complement to our "MXFP4-for-prefill" item.

**6. Model lineage — APPLIES (high confidence).**
- Qwen3.8-Flash-Next engine = upstream Strata (Niko1221; "Qwen3.8-Flash-Next on any consumer hardware", 12-24 GB cards); Qwen3.8-27B dense port is perronemirko's; DeepSeek-V4.1/GLM engines + GSQ quant are Indras Mirror's; quant fine-tunes by peasantsmith/ukisai/d0xin/gcsa-abliterix. (README.md Credits; https://api.github.com/repos/Niko1221/Strata)
- Practical fork hygiene intel for custom-2: UPSTREAM_SYNC.md's merge gates and isolation rules, and MERGE_SCOPE.md's graft recipe, are the documented map of this family's history — relevant if we ever pull Xolotl content into custom-2 (histories join only via the graft; criss-cross gotcha documented: do intra-fork merges first, graft last).

## Takeaways for our fork (specific)

1. **Build pin:** if we add any block-scale-MMA kernel (MXFP4/NVFP4 prefill path), sm_120 needs CUDA 13.1+ and the `120a` arch suffix; CUDA 12.8 fails ptxas and showed runtime prompt-path crashes in this codebase. (issue #14, README)
2. **Agent-loop serving:** conversation pool + batch slots + exact resumes are merged, benchmarked on a 5090 (47.5 -> 3.96 s/turn), and live in our engine family — prime cherry-pick/port candidates if our base predates them.
3. **T14B:** nobody in this family trains MTP heads; our retrain is novel. Borrow their runtime contract (mtp_pack/mtp_rt formats), acceptance-based format choice, and parity/acceptance harnesses as the eval rig.
4. **Quant:** don't port the GSQ pipeline for Flash-Next; use ecosystem GGUFs + their quant_quality.py harness for our sm_120 format decisions (agrees with our IQ3_S-no-dominant-format conclusion).
5. **Technique catalog:** DS4.1's adaptive arena, admission control, remote-expert card, and DSpark's exact-sampling/confidence-gated verify are documented designs on our shared core — read docs/DEEPSEEK_V41.md, docs/DISJOINT_EXPERT_CACHE.md, docs/BATCHED_DMA.md before designing equivalents for Qwen.

## Cheapest evaluation

Zero-build, ~1-2 h: (1) shallow sparse clone (`git clone --depth 1 --filter=blob:none --sparse`; sparse-checkout excludes `bench/` — 2,442 of 4,067 files are bench artifacts); (2) `git diff --stat <our-custom-2-base> xolotl/main -- src serve tools include CMakeLists.txt` to enumerate portable deltas since our merge point; (3) read docs/UPSTREAM_SYNC.md + issue #14 + tools/mtp_pack.py in full. First build gate (only if we port code): sm_120 build with CUDA 13.1 + `120a`, then their parity binaries (`src/kernels/*_parity`).

## Contradictions

- None material. Xolotl README's Qwen numbers (127.8 tok/s on 4090+90 GB DDR4) vs our measured 180.9 tok/s on 5090: different hardware, no conflict — our prior file's figures stand.
- "No MTP head in GSQ-RCO GGUF" (mtp_fetch) vs HF listing "oQ4e-mtp" / "NVFP4-GPTQ-Strata" repos: unverified whether those embed MTP; not load-bearing for us (we pack our own head regardless).

## Open questions

1. Which upstream Strata version/tag is custom-2's base? Determines whether MTP flyweight tools, draft_vocab, batch slots, conversation pool are already present (they exist upstream at 0.1.44.1+).
2. Does upstream main (pushed 2026-10-20) carry post-0.1.44.1 Qwen improvements we lack? (Upstream tracks ahead of Xolotl's merge point.)
3. Do docs/NVFP4.md + nvfp4_convert.py offer a better prefill path than our MXFP4-for-prefill item? (Unfetched this pass.)
4. Is STRATA_SPLIT_MTP_BATCH (MMQ draft KV) beneficial at sm_120 decode, and is it in our base?
5. Does ISTA-DASLab's GSQ-RCO method have a published spec we could reuse for Qwen expert quant without porting v41 tooling?

## Sources

Code/history: api.github.com/repos/Indras-Mirror/Xolotl (+ /commits, /git/trees, /releases, /tags, /issues/14); api.github.com/repos/Niko1221/Strata (+ /contents/tools); raw files CMakeLists.txt, tools/{mtp_fetch,mtp_pack,mtp_rt,dump_mtp,draft_vocab,split_mtp_batch_parity}.py, tools/v41/{v41_quant,v41_calib}.py. Docs/specs: README.md; MERGE_SCOPE.md; docs/UPSTREAM_SYNC.md; docs/xolotl/HOW_IT_WORKS.md; docs/WINDOWS.md; docs/QUANTS.md; README.strata.md; docs/MODELS.md. Data/knowledge bases: huggingface.co/api/models?author=IndrasMirror; huggingface.co/api/models?search=Qwen3.8-Flash-Next. Practitioner: issue #14 community RTX 5090 D v2 build report. Total 20+ sources, 4 categories.

## Methods note

All fetches 2026-10-20 via GitHub API / raw.githubusercontent / HF API; no clones, builds, or model boots. Unproductive: Xolotl releases/tags (zero — commit-based only); IndrasMirror HF profile contains no Flash-Next quant (GSQ-RCO Flash-Next quants are ISTA-DASLab's, not Xolotl's; the mtp_fetch "GSQ-RCO" reference is to the ecosystem). Not fetched (flagged as leads, not searched-and-failed): docs/DEEPSEEK_V41.md, docs/DISJOINT_EXPERT_CACHE.md, docs/BATCHED_DMA.md, docs/NVFP4.md, docs/ds4/STRATA_ARCH_MAP.md, docs/paper/Strata-Paper.pdf. Earlier repo name Strata-DSV4.1 redirects; repo is not a GitHub fork (standalone, re-rooted history).
