---
type: research
id: web-nvfp4-qwen-flash
date: 2026-11-18
query: Strata v0.2.12..v0.2.14.1 NVFP4 support; NVFP4 packs for Qwen3.8-Flash-Next; NVFP4 vs IQ3_S decode on RTX 5090; 128 GB pack on 32 GB VRAM streaming issues
status: complete
method: WebFetch (GitHub releases/issues/PR, Hugging Face model cards and trees) + WebSearch (r/LocalLLaMA, r/LocalLLM, r/BlackwellPerformance, llama.cpp/NVIDIA forums)
---

# Web research: NVFP4 for Qwen3.8-Flash-Next on Strata / RTX 5090

## Verdict summary

| Question | Verdict |
|---|---|
| Did v0.2.12..v0.2.14.1 add NVFP4? | **No.** NVFP4 support predates this window (tested in v0.2.7.1, 27 Oct; per sergqwer, upstream engine 0.1.44 NVFP4 packs were a port of the fork's work). The window's only NVFP4 text is one fit-label line in v0.2.13. |
| "sergqwer Qwen3.8-FN-NVFP4 (~128.8 GB)" HF pack | **Not verified.** `huggingface.co/sergqwer` returns 404. Real artifacts: `sergqwer/strata-nvfp4` GitHub fork (packs ~128–130 GB) and `jesdga/Qwen3.8-Flash-Next-Basalt` (NVFP4 files, 132 GB). |
| NVFP4 faster than IQ3_S for MoE decode on 5090? | **No, on published evidence.** IQ3_S decodes faster (254–320 t/s vs UD-Q4_K_XL 148–183 t/s; 1-bit integer beats NVFP4 packs 389 vs 307 t/s). NVFP4's wins are prefill (+24–40% on Blackwell) and batched throughput. Quality: NVFP4 ≈ Q4_K_XL-class, modestly above IQ3_S. |
| Model size | Task premise said 748B/16B-active. **Contradicted:** NVIDIA's card states 125B total / 6B active + 51B n-gram (PLE) + 4B MTP, 48 layers, 24,576 experts. |
| 128 GB pack on 32 GB VRAM + RAM streaming | Known pain point, documented. RAM-starved streaming costs ~30% decode / ~25% prefill (PR #5746); Windows pinned-memory failures; startup OOMs; mitigations exist (`--vram-reserve-mib`, cgroup auto-sizing, unbuffered NVMe reads). |

---

## 1. Strata releases v0.2.12 → v0.2.15

Repo live: [Niko1221/Strata](https://github.com/Niko1221/Strata) ("Qwen3.8-Flash-Next on any consumer"), [releases](https://github.com/Niko1221/Strata/releases). Latest = v0.2.15 (16 Nov). Versioning: app/Terminal version is separate from Control (CLI) and engine-package versions.

| Tag | Date | Headline | NVFP4? |
|---|---|---|---|
| v0.2.12 | 07 Nov | Tuned DFlash2 drafter ([incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), Apache-2.0; decode +2.4–5.0%); ~3x faster prompts on 12 GB cards (87→264 t/s at 2K, 361→981 t/s at 32K); Strix Halo prompts x1.24–2.14; Mac +10–14%; honest free-RAM accounting for 12 GB Linux cards; tool-call fixes (#5312, #4828, #5409, #5276, #5281, #4622, #4832). Engine 0.1.58. | **Absent** |
| v0.2.13 | 09 Nov | Per-card DFlash2 defaults; app main file 3.12→0.76 MB; Strix Halo Terminal launch fix; deterministic Arc inference; "Qwen4 groundwork" (signed feed, per-shape engine, `strata --geometry`; feed has no model list yet). Engine 0.1.59. | One fit-label line: "NVFP4-QAD is sized at 76.8 GB, the pack that runs" (DeepSeek fit labels; no benchmarks, no Qwen NVFP4 numbers) |
| v0.2.14 | 15 Nov | **Prefill moved to Strata's own kernels** on NVIDIA/AMD, replacing llama.cpp-derived code (fallback `-DSTRATA_MMQ=ON`); equal-size prompt chunks on single card; two-card split keeps per-card dense weights (#4522) + drafter-head reservation (#4549): 2x A6000 64.16→75.88 t/s; drafting persistence fixes. Engines 0.1.60 (Arc/DGX Spark 0.1.59). | **Absent** |
| v0.2.14.1 | ~15–16 Nov | Terminal/Control patch: Mac drafting-Off no longer downloads the 5.2 GB MTP layer (first launch 272 s → 13 s on M5 Max). | **Absent** |
| v0.2.15 (latest) | 16 Nov | App **no longer launches llama.cpp's llama-server** — all model serving through Strata's own engine; Qwen3.6 35B-A3B first-run fix; drafting-choice fixes. Engines unchanged (0.2.14 files, 0.1.60). | **Absent** |
| (context) v0.2.11 | 05 Nov | Unsloth "Dynamic v3.1" GGUF re-upload support: UD-IQ1_M 68.0 GB, UD-Q2_0 70.6 GB, UD-IQ2_XS 75.8 GB, UD-Q2_K_XL 79.6 GB, UD-IQ3_XXS 83.6 GB, UD-Q3_K_XL 89.5 GB, UD-IQ4_XS-v31 93.7 GB (28.8 GB shared expert table); Q5_0 expert format (Strata error 2.6–3.0% → 1.2–1.3% vs llama.cpp); 4x R9700 split engine +6.9% (1 user) / +32.7% (8 users); IQ3_S 4x R9700 104.2→186.8 t/s. Engine 0.1.57. | **Absent** |
| (context) v0.2.7.1 | 27 Oct | DGX Spark support; tested "Qwen3.8-Flash-Next (IQ3_S and the NVFP4 pack, 64K context)". | Present (earliest sighting in this survey) |

Note: the v0.2.7.1 tag page fetch was truncated and the GitHub API returned 404 for `releases/tags/v0.2.7.1`; the NVFP4 line comes from the releases index page. Flagged as partially sourced.

### RTX 5090 published numbers (v0.2.14 notes, build of 11-14; brackets = v0.2.13)

Strata's published 5090 table covers **IQ3_S and UD-Q4_K_XL only — no NVFP4 figures for 5090 in these releases.**

Prefill (tokens/s):

| Quant | 256 | 2,048 | 8,192 | 20,000 | 32,000 |
|---|---|---|---|---|---|
| IQ3_S | 1,347 (1,196) | 3,800 (3,730) | 6,901 (6,517) | — | 7,899 (7,178) |
| UD-Q4_K_XL | 988 (868) | — | 6,350 (5,945) | 7,710 (6,707) | 7,870 (6,851) |

Decode (tokens/s):

| Quant | chat | code | json |
|---|---|---|---|
| IQ3_S | 254.2 (274.0) | 304.8 (300.2) | 320.1 (316.0) |
| UD-Q4_K_XL | 148.6 (147.7) | 183.5 (183.9) | 160.5 (160.2) |

Reading: v0.2.14's own kernels lifted prefill (IQ3_S 32K +10.1%) at a small IQ3_S chat-decode cost (-7.2%); integer-quants decode speed ranks inversely with bits-per-weight. RTX 4090 prefill quoted in ms only (20K IQ3_S: 5,656 ms vs 6,110 ms).

### NVFP4 support provenance

- sergqwer README (fetched): upstream Strata engine **0.1.44** "NVFP4 packs" were "a port of this fork's work." Engine 0.1.44 predates engine 0.1.53 (v0.2.7.1, 27 Oct), 0.1.57 (v0.2.11), 0.1.58–0.1.60 (v0.2.12–v0.2.14).
- Task premise "upstream recently (v0.2.12..v0.2.14.1) added NVFP4 support" — **not supported by any release note in the window.** NVFP4 work is older; the Oct–Nov releases touched NVFP4 only via the v0.2.13 fit-label line and DGX Spark testing in v0.2.7.1.

---

## 2. NVFP4 pack availability (Hugging Face)

Reference model, per [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4): **125B total / 6B activated per token**, plus 51B n-gram (PLE) embeddings and 4B MTP; 48 layers; 24,576 routed experts; BF16 source ≈ 360 GB. The task's "748B-total/16B-active class" premise is contradicted by the owner's own card.

Verified NVFP4 artifacts:

| Repo | Size | Notes | HW target |
|---|---|---|---|
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | 132.7 GB (132,711,524,896 B) | Official. Model Optimizer v0.46.0; routed experts W4A4 NVFP4 (MSE-calibrated scales), attention + shared experts BF16, MTP experts 128x128 FP8, PLE per-tensor FP8; PTQ, no fine-tune; calibration: cnn_dailymail + Nemotron-Post-Training-Dataset-v2. Created 2026-09-02. 71.3k downloads/mo. | B200/B300, vLLM TP≥8 (per card); FP8-vs-NVFP4 evals near-flat (GPQA 92.0→91.5, HLE 34.7→35.4, MMMU-Pro 77.1→78.3, τ²-Bench 90.8→90.1). No MMLU/PPL on card. |
| [nvidia/Qwen3.8-Flash-Next-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD) | 76.8 GB (per Strata v0.2.13 fit label) | Experimental quantization-aware distillation checkpoint (Model Optimizer + NeMo, 100 optimizer updates). | Blackwell-class |
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | 135 GB (from 360 GB, ~2.7x) | Official NVFP4 checkpoint redistribution; agentic/chat/coding serving. | — |
| [MagneticLab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/MagneticLab/Qwen3.8-Flash-Next-NVFP4) | ~101 GB (via ModelScope) | — | — |
| [garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream](https://huggingface.co/garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream) | 48 GB lookup table streamed from SSD | Low-RAM variant. | — |
| [local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | — | QAD mixed-precision distilled; companion benchmarks at [local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/nvfp4-quantization-comparison.md). | — |
| [jesdga/Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) (file tree) | NVFP4-DQ6_K 132 GB, NVFP4-DQ8 132 GB (122.0 / 123.3 GiB per card table); full ladder UD-IQ1_S 70.4 GB → Q8_0 192 GB | **Basalt = Strata fork**, "fastest inference engine for Qwen3.8-Flash-Next on Blackwell" (MIT engine, sm_120). Benchmarks on RTX 5090 400 W + RTX 5060 Ti peer 180 W. | RTX 5090-class Blackwell |
| `sergqwer` HF pack "Qwen3.8-FN-NVFP4 ~128.8 GB" | — | **DOES NOT RESOLVE.** `huggingface.co/sergqwer` → HTTP 404; no indexed HF repo. Real sergqwer artifacts are GitHub-only: [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) — packs `qwen-nvfp4-gptq` (~130 GB disk = 63 GiB experts + 51 GB n-gram table), `qwen-nvfp4-mix` (GPTQ NVFP4 + Q8_0 in 38/48 layers, "128 GB PC"), `qwen-nvidia-nvfp4` (converted from NVIDIA ModelOpt), `qwen-q8` reference. Targets RTX 20/30/40/50 12 GB+ VRAM, 64 GB+ RAM, AVX-512/AVX2 CPU expert compute. 249 stars, MIT, tracks upstream 0.2.15. | Single consumer GPU + host RAM |

The "~128.8 GB" figure plausibly corresponds to sergqwer's `qwen-nvfp4-mix` ("128 GB PC") or the jesdga NVFP4 files (132 GB), but the exact claim — an HF repo named after sergqwer with that size — **could not be verified and should not be cited as fact.**

---

## 3. NVFP4 vs integer quants on Blackwell (MoE decode + quality)

### Direct, model-specific: sergqwer/strata-nvfp4 (Qwen3.8-Flash-Next, single GPU + RAM experts)

Top-token agreement vs Q8_0-all-experts reference (5,153 tokens of code/docs/chats); speed on their streaming setup (12 GB+ VRAM, 64 GB+ RAM):

| Pack | Agreement vs Q8_0 | Decode t/s |
|---|---|---|
| NVFP4+Q8 mix (Q8_0 in 38/48 layers) | 95.0% | 91 |
| Unsloth UD-Q4_K_XL | 94.1% | 169 |
| NVFP4 GPTQ (sergqwer) | 93.6% | 177 |
| NVIDIA NVFP4 (ModelOpt) | 92.9% | 167 |
| UD-Q3_K_XL | 90.7% | — |
| IQ3_S (ISTA-DASLab) | 90.0% | — |
| IQ2_XS | 85.0% | — |

KL to Q8_0: 0.051 (NVIDIA NVFP4 local) vs 0.038 (GPTQ); GPTQ error = 33.3% of plain rounding, 44.8% of NVIDIA's. No MMLU published on the fork.

Reading: **NVIDIA NVFP4 (92.9%) sits between UD-Q3_K_XL (90.7%) and UD-Q4_K_XL (94.1%) on quality, and does not decode faster than the integer quants in the same table.** NVFP4+Q8 mix buys +0.9 pt over UD-Q4_K_XL at roughly half the decode rate in that config.

### Direct, model-specific: jesdga/Basalt on RTX 5090 (400 W) + RTX 5060 Ti peer

Decode t/s (prose / structured / chat) and perplexity (BF16 reference PPL 3.273):

| Pack | Size (GiB) | Decode t/s | PPL / quality |
|---|---|---|---|
| NVFP4-DQ6_K | 122.0 | 307 / 601 / 335 | packs range 3.231–3.405; KL 0.0209–0.2676; top-1 agreement 82.76–95.34% |
| NVFP4-DQ8 | 123.3 | (see repo) | " |
| UD-IQ1_S | 65.6 | 389 / 763 / 457 | worst PPL 3.405 |
| UD-IQ4_XS | 90.9 | (see repo) | best PPL 3.231 |
| UD-Q4_K_XL | 106.7 | (see repo) | " |
| Q8_0 | 178.3 | (see repo) | " |

Reading: in the only setup with both formats on a 5090, **a 1-bit integer pack out-decodes the NVFP4 pack by ~27%** (389 vs 307 t/s prose). MoE decode on this card is bandwidth-bound; fewer bytes/token wins, tensor cores don't help single-stream decode.

### Generic Blackwell evidence (other models, cross-checks)

- llama.cpp: NVFP4 real inference since build **b8967** (Apr 2026; before that, fallback paths on 5090 / RTX PRO 6000 / B200). Native MXFP4 merged Dec 2025 ([NVIDIA forum](https://forums.developer.nvidia.com/t/llama-cpp-experimental-native-mxfp4-support-for-blackwell-pr/355639)).
- NVFP4 vs Q4_K_M GGUF: NVFP4 faster **only on Blackwell**, and the advantage is **prefill** — 24–40% faster prompt reading (16K prompt +40%); quality tradeoffs reported in long-context recall on MoE (Qwen3 35B-A3B). NVIDIA claims up to 1.9x throughput from its llama.cpp optimizations.
- Feb 2026 comparison article ("NVFP4 vs Q4_K_M vs GGUF Q6_K"): RTX 5090 **Q4_K_M GGUF 190 t/s decode vs NVFP4-via-vLLM 68 t/s** single-stream; Q4_K_M retains ~95% of BF16; NVFP4 ~97.5% of BF16 on MMLU but 79–82% on hard reasoning. URL not captured by the search surface — **attribution weak, treat as indicative.** Related artifacts: [witcheer/rtx-5090-benchmarks](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks) (HF dataset, 5090 32 GB), [r/LocalLLM vLLM recipe thread](https://www.reddit.com/r/LocalLLM/comments/1vu6w9u/my_rtx_5090_vllm_recipe_nvfp4_weights_nvfp4_kv) (NVFP4 strengths = batched throughput / TTFT, e.g. ~5,809 t/s prefill, not single-stream decode).
- Unsloth docs (search surface, URL not captured): GGUF decode **1.03–1.22x faster** than NVFP4 depending on model size.
- r/LocalLLaMA thread [NVFP4 – what does the quality usually equate to? (5090)](https://www.reddit.com/r/LocalLLaMA/comments/1wogz1a/nvfp4_what_does_the_quality_usually_equate_to_5090) — exists; content fetch blocked (Reddit anti-bot). Title-level signal only: users debate whether NVFP4 is a "no-brainer" for 5090 owners vs quality concerns. **Figures from this thread not extracted.**
- r/LocalLLM thread [Qwen3.8-Flash-Next Q6 vs NVFP4 on my RTX 5090](https://www.reddit.com/r/LocalLLM/comments/1xd8wsm/tested_qwen38flashnext_q6_vs_nvfp4_on_my_rtx_5090) — exists; UD-Q6_K_XL vs NVFP4-GPTQ with MMLU-Pro. Content fetch blocked. **Figures not extracted.**
- Strata PRs actively speeding up integer quants (counter-signal to "NVFP4 is the fast path"): [#6085](https://github.com/Niko1221/Strata/pull/6085) IQ3_XXS/IQ3_S/IQ2_S fast paths via in-register grid lookups (draft); [#6158](https://github.com/Niko1221/Strata/pull/6158) IQ2_S AVX-512 VBMI gate/up rows (draft).

### Answer to the key question

**Is NVFP4 meaningfully faster than IQ3_S-class integer quants for MoE decode on a 5090? No.** Every sourced comparison shows integer (even 1–3 bit) packs decoding equal or faster in single-stream MoE decode: 254–320 t/s IQ3_S vs 148–183 t/s UD-Q4_K_XL (Strata 5090 table), 389 t/s UD-IQ1_S vs 307 t/s NVFP4-DQ6_K (Basalt 5090), 177 t/s NVFP4-GPTQ vs faster UD-Q4_K_XL 169 t/s — within noise of each other — and 190 t/s GGUF Q4_K_M vs 68 t/s NVFP4-vLLM (third-party, weak attribution). NVFP4's measured advantages are **prefill** (+24–40% on Blackwell in llama.cpp) and **batched throughput/TTFT under concurrency** (vLLM/TRT-LLM serving), neither of which is the single-stream decode path this workspace runs.

**Quality tradeoff:** NVFP4 ≈ 92.9–95.0% top-token agreement vs Q8_0 (model-specific) — i.e. **Q4_K_XL-class, above IQ3_S (90.0%)**; generic numbers: ~97.5% of BF16 MMLU, 79–82% on hard reasoning, long-context-recall dips on MoE. NVFP4 costs ~122–135 GB vs ~90 GB for IQ3_S-class packs: on a 32 GB VRAM + host-RAM streaming setup, that extra ~35–45 GB lands in the RAM/SSD tier and increases expert-miss cost. **IQ3_S remains the right operating point for this 32 GB 5090 box unless prefill latency at long context or concurrent serving becomes the bottleneck.**

---

## 4. 128 GB packs on 32 GB VRAM + host-RAM streaming (known issues)

| Artifact | Hardware / pack | Symptom | Resolution / mitigation |
|---|---|---|---|
| [PR #5746](https://github.com/Niko1221/Strata/pull/5746) (draft, benchmark) | RTX PRO 4000 Blackwell 24 GB, NVFP4 GPTQ, v0.2.13.1, MTP spec decode, 131k ctx, int8 KV; 64 GB RAM simulated via cgroup `MemoryMax=56G` | RAM-capped vs 128 GB: short decode 73.1 vs 101.4 t/s; 4k 65.7 vs 97.3; 16k 68.7 vs 93.5 → **~30% decode / ~25% prefill cost**. Under cap: 49.31 GiB experts page-locked, 5,453 experts in GPU cache, ~14 GiB unbuffered NVMe reads, 1,299/1,327 prompt slots served from pack. Load 43 s capped vs 271 s uncapped (fragmented-RAM arena copy). | Engine reads cgroup limit and self-sizes; "full sanity scores" (12/12, 21/21 requests). Caveats: simulated cap, single machine, machine-learned expert profile. |
| Windows pinned-memory failure (search surface; issue number not captured) | 128 GB DDR5 + NVFP4 pack | `cudaHostAlloc` — "cannot pin 0.52 GiB of RAM for a layer's KV copy"; engine dies after ~2 layers. | Number/URL not recovered — **partially sourced.** |
| [Issue #1250](https://github.com/Niko1221/Strata/issues/1250) | RTX 3080 12 GB + 62 GB RAM, engine 0.1.40.1, Linux | Startup OOM: ~39 GB experts locked for GPU; expert set doesn't fit RAM+VRAM. | Open. |
| [Issue #779](https://github.com/Niko1221/Strata/issues/779) | Laptop | Crash after start; log shows 444 MiB free. | `--vram-reserve-mib 3072` (larger reserve shrinks expert slots). |
| [Issue #60](https://github.com/Niko1221/Strata/issues/60) | 16 GB VRAM + 64 GB RAM | OOM loading iq2_xs. | — |
| [Issue #5936](https://github.com/Niko1221/Strata/issues/5936) | GB10 (sm_121), Edge0-145B-NVFP4 pack (0.1.55) | Batched prompt attention hangs at context >= 64 tokens until watchdog kill. | Open — NVFP4 pack on GB10 broken. |
| [PR #5597](https://github.com/Niko1221/Strata/pull/5597) (draft) | RTX PRO 6000 Blackwell WS 96 GB, 350 W, engine 0.1.58, NVFP4 pack | Community benchmark (healthy case): all 24,576 experts fit in GPU cache (63.29 GiB) — zero RAM/SSD prompt waits. | Shows NVFP4 end-to-end only pays off when the expert set is VRAM-resident. |
| sergqwer/strata-nvfp4 README | <92 GiB RAM machines | Experts don't fit; fork ships low-RAM mode. | Unbuffered reads; Q8_0 expert subset; CPU computes missed experts (AVX-512/VNNI). |

**Operational conclusion for this workspace (5090 32 GB + host RAM):** 122–135 GB NVFP4 packs on 32 GB VRAM force the same tiered-expert path Strata already runs for IQ3_S (~90 GB), with ~35–45 GB more resident in RAM/SSD tiers and no decode upside per section 3. Watch `--vram-reserve-mib`, cgroup-aware sizing (works since ~0.1.4x), Windows pinned-RAM failures, and GB10 NVFP4 hang (#5936).

---

## 5. Unverified / contradicted claims (do not cite as fact)

1. **"sergqwer Qwen3.8-FN-NVFP4 (~128.8 GB)" HF pack** — no such HF repo; profile 404. Closest real artifacts: sergqwer GitHub fork packs (~128–130 GB), jesdga NVFP4 files (132 GB).
2. **"v0.2.12..v0.2.14.1 added NVFP4 support"** — contradicted by all fetched notes in the window; NVFP4 predates (engine 0.1.44 per sergqwer; tested in v0.2.7.1, 27 Oct).
3. **"748B-total/16B-active class"** — contradicted by NVIDIA's card: 125B total / 6B active + 51B PLE + 4B MTP.
4. **190 vs 68 t/s comparison** — search-surface only; article URL not captured.
5. **Unsloth "GGUF 1.03–1.22x faster than NVFP4"** — search-surface only; exact docs URL not captured.
6. **Reddit thread bodies** (r/LocalLLaMA 1wogz1a, r/LocalLLM 1xd8wsm) — fetch blocked; titles and search snippets only.
7. **v0.2.7.1 full notes** — page truncated + API 404; NVFP4 line from releases index only.
8. **RTX 5090 NVFP4 numbers published by Strata** — none exist in v0.2.12..v0.2.15 notes; the 5090 table is IQ3_S / UD-Q4_K_XL only.

## Sources

- https://github.com/niko1221/strata — repo root
- https://github.com/Niko1221/Strata/releases — releases index
- https://github.com/Niko1221/Strata/releases/tag/v0.2.12 — v0.2.12 notes
- https://github.com/Niko1221/Strata/releases/tag/v0.2.13 — v0.2.13 notes
- https://github.com/Niko1221/Strata/releases/tag/v0.2.14 — v0.2.14 + 0.2.14.1 notes, RTX 5090 table
- https://github.com/Niko1221/Strata/releases/tag/v0.2.11 — v0.2.11 notes (Unsloth v3.1 ladder)
- https://github.com/Niko1221/Strata/issues?q=nvfp4 — issue/PR index (#5936, #5746, #5597, #6085, #6158)
- https://github.com/Niko1221/Strata/pull/5746 — 64 vs 128 GB RAM NVFP4 benchmark
- https://github.com/Niko1221/Strata/issues/60 — OOM 16 GB VRAM
- https://github.com/Niko1221/Strata/issues/1250 — startup OOM 3080 12 GB
- https://github.com/Niko1221/Strata/issues/779 — vram-reserve mitigation
- https://github.com/sergqwer/strata-nvfp4 — fork: NVFP4 pack ladder, accuracy table, upstream 0.1.44 provenance claim
- https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4 — official pack (132.7 GB, 125B/6B, evals)
- https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD — QAD checkpoint
- https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4 — 135 GB redistribution
- https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt — Strata-fork engine + pack ladder + 5090 benchmarks
- https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/tree/main — file sizes
- https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks — 5090 quant benchmark dataset
- https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/nvfp4-quantization-comparison.md — NVFP4 accuracy tables
- https://www.reddit.com/r/LocalLLaMA/comments/1wogz1a/nvfp4_what_does_the_quality_usually_equate_to_5090 — thread (title-level only)
- https://www.reddit.com/r/LocalLLM/comments/1xd8wsm/tested_qwen38flashnext_q6_vs_nvfp4_on_my_rtx_5090 — thread (title-level only)
- https://www.reddit.com/r/LocalLLM/comments/1vu6w9u/my_rtx_5090_vllm_recipe_nvfp4_weights_nvfp4_kv — vLLM NVFP4 recipe (batch-vs-single-stream)
- https://www.reddit.com/r/BlackwellPerformance/ — RTX PRO 4500 32 GB + 64 GB DDR5, 188k warm ctx, 60–67 t/s NVFP4/Strata
- https://forums.developer.nvidia.com/t/llama-cpp-experimental-native-mxfp4-support-for-blackwell-pr/355639 — llama.cpp MXFP4/NVFP4 on Blackwell
