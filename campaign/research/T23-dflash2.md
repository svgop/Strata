# T23 — incoai/Qwen3.8-Flash-Next-DFlash2: ready-made drafter vs our T14B FastMTP retrain

Recon date 2026-10-20. Read-only; no builds/boots. Prompted by the T14B retrain plan
(T11.R3 GO in SVG-CUSTOM-TODO.md :793; spec trio in campaign/T14B-specs/) — question:
does this repo make the retrain redundant?

## VERDICT: DOESN'T (as a T14B skip / drop-in drafter). PARTIAL intel value — keep T14B.

It is a real, Apache-2.0, ready-trained drafter for exactly our backbone, but it is NOT an
MTP head: it is a 0.6B five-layer block-diffusion draft MODEL (DFlash 2), loadable by
neither our mtp/rt loader nor — per Inco's own card — upstream llama.cpp (missing
target-side capture of the mean of the four hyper-connection streams). SGLang-only today.
Its claimed acceptance-length delta over stock MTP (+0.18..+0.42 on their mix) is also
smaller than our T14B target (+0.55-0.7 over our banked 2.78). Servable-by-us = weeks of
new engine work, not a drafter swap. Three adoptable intel items listed under ACTION.

## Identity (HIGH confidence — primary sources: HF API record, config.json, raw READMEs)

- Repo: https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2 — created 2026-10-19
  (one day before this recon; 678 downloads, 48 likes). Files: model.safetensors (BF16,
  603,065,600 params ≈ 1.21 GB stored), config.json, README.md,
  patches/qwen38-dflash2-capture.patch (SGLang feature-capture patch), assets/figure.
- License: apache-2.0 (card YAML + API tags). base_model: Qwen/Qwen3.8-Flash-Next.
- Architecture (`DFlash2DraftModel`, model_type `qwen3`): 5 hidden layers, hidden 2560,
  32 Q / 8 KV heads (head_dim 128), sw 2048 sliding_attention on all layers,
  **is_causal: false, use_cache: false**, vocab **248,320 FULL head** (no draft_vocab
  subset), max_pos 262144. `dflash_config`: block_size 8, mask_token_id 248077,
  selector_rank 256 / selector_top_k 16, two-tap dynamic depthwise conv (kernel 2,
  group 16), num_target_layers 48 with **target_layer_ids [5, 15, 25, 35, 45]** — it
  consumes target hidden states from five deep taps plus the mean of the target's four
  hyper-connection streams (their GGUF card, quoted verbatim: "Qwen3.8-Flash-Next carries
  its hidden state as four hyper-connection streams, and the draft is trained on their
  mean."). Mechanism: drafts a whole block in one parallel pass, then a low-rank bilinear
  path selector walks one coherent path through each position's top-16 candidates.
- Quants: main repo BF16 safetensors only. Sibling
  https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF (updated 2026-10-19):
  Q4_K_M 362 MB / Q5_K_M 432 MB / Q6_K 506 MB / Q8_0 652 MB / BF16 1.22 GB.
  **No Q2/IQ quants** — our q2_0-class budget has no offering.
- Provenance: Inco AI (inco.ai; inference-stack startup; HF org activity under user
  zhijianliu; DFlash = ICML 2026 paper "Block Diffusion for Flash Speculative Decoding",
  Chen, Liang, Liu). **Training data/method NOT disclosed** anywhere (card, GGUF card,
  blog). The official research-lab org https://huggingface.co/z-lab has the Qwen3.8-27B
  and Muse-Glimmer-30B DFlash2 drafters + GGUFs but NOT the Flash-Next drafter — that
  exists only under `incoai`. Caveat: llama.cpp maintainer ngxson (PR #27342, Aug 28)
  flagged "many broken GGUFs are still being distributed from incoai account" (pre-Aug-27
  reconversions; vision breakage tracked in llama.cpp #27862, closed). This repo is
  Oct-19, post-fix — but unvetted by upstream, unlike the z-lab 27B artifacts.
- Sibling family (incoai org, 13 models): DFlash2 drafters for Qwen3.8-27B, GLM-5.3-Flash,
  Ornith-1.5-122B, Muse-Glimmer-30B; "Splash" = their Apple-silicon engine.

## Claimed performance (MEDIUM confidence — single self-reported source, GB300/SGLang, not our box)

From the main card (eval conditions as stated: GB300, TP1, SGLang, vs the target's
built-in MTP): acceptance length — GSM8K **3.26 vs 3.08**, HumanEval **2.94 vs 2.52**,
MT-Bench **2.83 vs 2.50**. Throughput vs autoregressive: up to **1.82x** conc-1 (MATH-500,
383.6 t/s), **1.73x** conc-32 (GSM8K, 2,907.3 t/s). Blog (https://inco.ai/blog/dflash2):
selector adds 2.0M params (+0.6% latency), two-tap conv adds 16.5M (+3% params, 0.7%
cycle latency); family-wide gains 16-25% over DFlash 1. Tension (unresolved): card eval
says block size 4, config says block_size 8.

## Format compatibility vs OUR loader — incompatible on every axis (HIGH confidence)

| Axis | Our mtp/rt drafter (journal F-51/F-52/F-87/F-100; exec-B spec) | DFlash2 | |
|---|---|---|---|
| Class | 1-layer causal nextn-style MTP head | 5-layer NON-causal block-diffusion model | incompatible |
| Tensors | 31-tensor canonical fetch (SHA oracle + dense.txt ground truth): fc_embedding/fc_hidden eh_proj split (2560x2560), fused index_qk q\|k, 8 hc-mixer tensors (hc_norm f32 1x10240 over the hc-mean R(10240)) | conv/selector/mask/5-layer set; zero tensor-name or shape overlap | loader fails at tensor 1 |
| Vocab | 106,299-entry draft_vocab SUBSET head, Q4_0, + draft_vocab.bin bind | full 248,320 head | bind_bytes contract absent |
| Target taps | ONE tap: hc-mean + embedding at the MTP input | FIVE deep taps (layers 5/15/25/35/45) + mean of 4 hc streams | needs invasive graph surgery in our fork |
| Drafting | coupled sequential chain, exact-match verifier, spec 4-6, DT 0.85 | parallel block draft + selector path walk, mask tokens | different verifier machinery |
| Fork check | `grep -ri dflash engine/ src/ tools/` = **zero hits** | upstream llama.cpp PR #27342 (merged 2026-08-27) is DFlash2-capable but Inco's card states it "**cannot drive this draft yet**" (hc-stream mean not computed in llama.cpp); their GGUF card: "Until it does, serve the safetensors checkpoint with SGLang." | unusable today, any lane |

Contradiction resolved: blog says "llama.cpp" is supported and PR #27342 is indeed merged
(from z-lab:dflash2; tested BF16/Q8_0/Q4_K_M on Qwen3.8-27B, 1.77-1.85x on M5 Pro; community
text-only runs of incoai GGUF drafts worked on RTX 3090). Both are true — engine support is
TARGET-dependent: works for the 27B (no hc capture needed), not for Flash-Next.

## What testing would take (recorded for completeness — verdict was DOESN'T, not APPLIES)

- One-boot falsifier in OUR engine: impossible. The rt loader's 31-tensor SHA/shape
  contract fails immediately; no dflash code exists in the fork. Download sizes if ever
  needed: safetensors 1.21 GB, GGUF Q4_K_M 362 MB (GGUF useless without llama.cpp-side
  hc capture).
- The real test lane (prices the INTEL, not a drop-in): SGLang + their bundled
  patches/qwen38-dflash2-capture.patch, ~1.2 GB download, Flash-Next target + drafter,
  acceptance measured on OUR mimic/depth lanes vs stock mtp/rt. Their GB300/TP1 numbers
  do not transfer to sm_120; no 5090 numbers exist anywhere.

## ACTION (adopted intel)

1. **No change to T14B.** Keep the FastMTP retrain (T11.R3 GO; spec trio, with the
   F-237 fixes: lr 1e-4, rt-r18 dir naming, G4/G5 relabel). Rationale: their claimed
   AL delta over stock MTP (+0.18/+0.42/+0.33) is below our retrain target
   (+0.55-0.7 over banked 2.78 -> 3.25-3.4 = the 250 path), and servability costs weeks.
2. **Bank the target-tap axis for T14C+.** DFlash2's five deep taps (layers 5/15/25/35/45)
   + hc-mean input is the EAGLE-3-fused-features class already flagged at journal :5636;
   our head taps only the MTP input. If T14B arms stall below gate, a multi-tap head is
   the named next lever. Their "trained on the mean of 4 hc streams" independently
   confirms our hc_mixer parity architecture (F-93/F-95 line of work).
3. **Reference anchors + watch item.** Stock-MTP AL on their mix (2.50-3.08) brackets our
   banked 2.78 — sanity anchor for the T14B offline evaluator. Watch incoai org and
   llama.cpp PR #27342: a llama.cpp-side Flash-Next hc capture would make the Q4_K_M GGUF
   bootable in a fork derivative (still a port into OUR heavily-modified engine, where
   custom beats stock cells by +17.8..+33.4%).

## Open questions

- Training data/method undisclosed — distribution match to our mimic/depth lanes unknown.
- Card eval block size 4 vs config block_size 8 — unresolved.
- Whether z-lab publishes an official Flash-Next drafter/GGUF (vetted lineage).
- sm_120 realizability of their gains — no consumer-GPU numbers exist.

## Sources

Primary: HF API model record (api/models/incoai/Qwen3.8-Flash-Next-DFlash2, 2026-10-19);
config.json (raw, quoted fields above); GGUF card raw README (hc-stream sentence verbatim);
llama.cpp PR #27342 (merged 2026-08-27, maintainer comments); incoai + z-lab org pages.
Secondary: inco.ai/blog/dflash2 (company blog — leads/corroboration only for numbers);
GGUF repo file listing. Local (our loader contract): SVG-CUSTOM.md F-51/F-52/F-87/F-100,
:5376-5384 (draft_vocab mechanics), :8046-8068 (stock mtp/rt config); SVG-CUSTOM-TODO.md
:793-809 (T11.R3 GO recipe); campaign/T14B-specs/ + T14A/T14C research files.
Source counts: 10 web (2 primary code/config, 3 primary registry/card, 2 org pages, 1 PR
thread, 2 blog/card) + 6 local.

## Methods note

Config.json and the GGUF raw README were fetched verbatim because they are load-bearing
(architecture fields; the hc-stream limitation). Card numbers accepted at MEDIUM only —
self-reported, GB300, no independent replication found (searched: no third-party evals of
the Flash-Next drafter; 27B drafter has community speed reports only). T11.R3 lives in
SVG-CUSTOM-TODO.md :793, not the journal — recorded so the next agent does not grep the
journal for it. No downloads/runs performed (read-only constraint). Provenance inference
"zhijianliu = Liu Zhijian (DFlash author)" is LOW confidence — name-match only.
