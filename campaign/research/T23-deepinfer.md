# T23 — DeepInfer (deepseek-ai/DeepInfer) recon for Strata custom-2

Date: 2026-10-20. Scope: repo + docs as of commit `9bd9cf2` (Oct 21, 2026); DSpark paper arXiv:2607.05147 (Jul 6, 2026). Read-only recon; no builds.

## Summary

DeepInfer is DeepSeek's production inference stack for **DeepSeek-V4.1-Flash only**, open-sourced Oct 20, 2026 (Apache-2.0, vLLM fork, offline inference). It targets **NVIDIA SM100 (CUDA 13) + Ascend 950 (CANN 9.2)** in a cluster shape (P/D disaggregation, Wide-EP decode on 64–128 GPUs); single-GPU is not a served mode. Nothing in it compiles for or targets **sm_120**. No GGUF/k-quant path (FP8/MXFP4/MXFP8 only, SM100 tensor-core formats). **One thing is worth taking, and it is an algorithm, not code: DSpark's confidence-scheduled verification length** for speculative decoding — directly applicable to our Qwen3.8-Flash-Next MTP loop, measured +4–7% throughput at DeepSeek's production operating point (+11% at 96 seq/rank), and it composes with (and relieves) our expert-cache verify cost.

## VERDICT: PARTIAL

Worth taking: **confidence-scheduled varlen MTP verification** (dynamic per-request draft-verify budget). Everything else — kernels (FlashMLA/DeepGEMM/DeepEP/DeepSelect/TileLang), MoE EP+EPLB, scheduler, quant — does not transfer to a single-box sm_120 llama.cpp-shaped engine.

## What it is

- "The high-performance inference framework that powers DeepSeek's large-scale production services"; built "primarily on vLLM", inspiration from SGLang/TRT-LLM; offline inference only. Apache-2.0. (https://github.com/deepseek-ai/DeepInfer — README)
- Brand new: 4 commits, initial `d585e65` Oct 20, 2026; **no releases/tags**; 16.2k stars; API surface is env-var driven (`DEEPINFER_*`). (https://github.com/deepseek-ai/DeepInfer/commits/main, /releases)
- Models: `deepseek-ai/DeepSeek-V4.1-Flash-1017` only. Hardware: "NVIDIA SM100 GPUs", "Huawei Ascend 950 NPUs"; CUDA 13, GCC ≥ 13, Python 3.12. (README)
- Shape: `csrc/` is thin pybind glue (`engine/`, `graph_timer/`, `utils/`); real kernels are git submodules — DeepEP, DeepGEMM, DeepSelect, FlashMLA (+ Ascend forks) under `vendor/`. (https://github.com/deepseek-ai/DeepInfer/tree/main/vendor, /tree/main/csrc)
- Explicitly **not included**: KV cache system, P/D disaggregation, cross-replica LB — this is the model/runtime layer of a much larger cluster stack. (README)

## Sub-question findings

**MoE expert serving — present, wrong shape (High).** MoE EP + EPLB, Wide-EP decode EP64–128, "Mega MoE" fused path ("regardless of EP size or batch size … significantly faster than unfused MoE"). All expert-parallel/multi-GPU; EPLB balancing via NVLink symm_mem. Zero single-GPU/host-offload content. (doc/inference_deployment_qa.md: https://raw.githubusercontent.com/deepseek-ai/DeepInfer/main/doc/inference_deployment_qa.md)

**Speculative decoding / MTP — the one relevant asset (High).** Two layers:
1. *In-repo*: DSpark + "confidence-scheduled verification"; env flags `DEEPINFER_ENABLE_MTP_VARLEN=1`, `DEEPINFER_USE_NEXTN_DECODE` (default draft length 5); `dflash_cuda_graph_runner.py` in `deepinfer/worker/`. (README; https://github.com/deepseek-ai/DeepInfer/tree/main/deepinfer/worker)
2. *Mechanism* (arXiv:2607.05147v1, PKU + DeepSeek-AI): lightweight confidence head (linear + sigmoid on drafter hidden state + Markov embedding of previous draft token) predicts P(token survives verification), trained against the analytical acceptance rate c* = 1 − ½‖p^d − p^t‖₁, calibrated by Sequential Temperature Scaling. A scheduler sorts cumulative survival probabilities across requests and greedily admits verification work against a profiled engine throughput curve SPS(B); lossless (non-anticipating); production version approximates capacity with confidences from two steps prior, reduced to a dynamic top-K. Effect: verified tokens/step cut ~30% (5.0 → 3.2–4.4) at ~unchanged accepted length. (https://arxiv.org/html/2607.05147v1)

**Decode kernels for sm_120 — absent (High).** NVIDIA path is SM100-only. Vendored deps are datacenter: DeepGEMM "NVIDIA SM90 or SM100 architecture GPU" (https://github.com/deepseek-ai/DeepGEMM — README); FlashMLA is MLA (our model is GDN/QSA hybrid — wrong attention); DeepSelect is DSA indexer/sampling top-k (https://github.com/deepseek-ai/DeepSelect); DeepEP is NVLink/RDMA EP transport. Repo-wide, no sm_120/consumer mention anywhere (README + Q&A searched).

**Batching/scheduling — present, Python cluster-side (High).** "Asynchronous scheduling (zero-overhead batch scheduler)", ZOS/CUDA-graph replay, batch-invariant kernels, deterministic sampling. Scheduler logic lives in the vLLM-fork Python layer; concept-level interest only for a C++ engine. (README; Q&A)

**Quantized (GGUF/k-quant) serving — absent (High).** Only low-precision KV/weights for SM100 tensor cores: main KV per-16 block FP8 (e4m3), indexer KV + routed-expert weights MXFP4, SWA KV MXFP8. No GGUF, no k-quants, no weight-only int formats. (Q&A; exact format wording read via fetch summarizer — re-verify lines before quoting in a paper trail.)

## Contradictions and resolution

- **"+4–7%" (repo Q&A) vs "60–85% faster per-user" (paper/press).** Not a real conflict: +4–7% is the *scheduler's marginal* throughput gain at a fixed production operating point (+11% at 96 seq/rank); 60–85% (Flash) / 57–78% (Pro) is the *full DSpark system* (DFlash drafter + confidence scheduling) vs static MTP baseline, per-user latency at matched aggregate throughput, on DeepSeek live traffic. For our fork the honest prior for the scheduler idea alone is the +4–7% band.
- **"DSpark = MTP?" — no.** The paper's drafter is DFlash (5-layer block-diffusion parallel drafter with target-layer KV injection), not a NextN/MTP module; production caps block γ=5. The confidence-head + scheduler is drafter-agnostic; that is precisely why it ports to an MTP drafter.
- Q&A quant quote mixes e4m3 with an NVFP4 comparison in the fetched summary; flagged above rather than resolved — verdict does not depend on it.

## Cheapest evaluation of the one takeaway

No build of DeepInfer needed. In the Strata fork (MTP verify loop + expert cache):
1. Instrument current fixed-γ MTP: log per-request accepted length τ, drafted-token confidences (proxy: draft token max-prob / entropy at each draft position; the paper's head is optional later), verify batch cost at B ∈ {1, 8, 32, 96}.
2. Offline-replay the policy: greedy per-request draft truncation by cumulative survival probability against a measured verify-cost curve; target the paper's signature — ~30% fewer verified tokens at ~equal τ.
3. Implementation is per-sequence draft-length choice in the verify batch (raggedness already handled by the batch dim); no new kernels. Decide adopt/reject on tokens/J at equal τ and p50/p99 latency at B=8/32.
Prior expectation: biggest win exactly where our cache hurts most (verify passes touching expert-cache misses) and as batch grows — the fixed-γ spec-decode trade-off degrades with batch size, which is the failure mode DSpark was built for.

## Open questions

- Does Qwen3.8-Flash-Next's released MTP head expose anything confidence-like, or must the head/proxy be trained on our acceptance logs? (Draft-logit entropy proxy needs no training; check its correlation offline first.)
- "Mega MoE" internals (fused permute-free path?) — SM100-targeted, but worth a design skim if Strata's MoE dispatch ever gets restructured. Not verified beyond the Q&A quote.
- Repo Q&A's SWA bounded replay (recompute last ~128 SWA tokens instead of storing SWA KV) — relevant only if our GDN/QSA hybrid has a sliding-window component; unverified for this model.

## Sources

1. README, raw — https://raw.githubusercontent.com/deepseek-ai/DeepInfer/main/README.md (what/scope/hw/features/env flags)
2. Commits — https://github.com/deepseek-ai/DeepInfer/commits/main (4 commits, dates, authors)
3. Releases — https://github.com/deepseek-ai/DeepInfer/releases (none)
4. Q&A deployment doc — https://raw.githubusercontent.com/deepseek-ai/DeepInfer/main/doc/inference_deployment_qa.md (DSpark production numbers, quant formats, Wide-EP/EPLB/Mega MoE, KV/storage)
5. setup.py, raw — https://raw.githubusercontent.com/deepseek-ai/DeepInfer/main/setup.py (thin CUDA ext; no arch flags)
6. csrc listing — https://github.com/deepseek-ai/DeepInfer/tree/main/csrc
7. deepinfer package listing — https://github.com/deepseek-ai/DeepInfer/tree/main/deepinfer
8. engine listing — https://github.com/deepseek-ai/DeepInfer/tree/main/deepinfer/engine
9. worker listing — https://github.com/deepseek-ai/DeepInfer/tree/main/deepinfer/worker (dflash_cuda_graph_runner.py)
10. vendor listing — https://github.com/deepseek-ai/DeepInfer/tree/main/vendor (submodule pins)
11. DeepSelect repo — https://github.com/deepseek-ai/DeepSelect (DSA top-k + sampling kernels; not MoE routing)
12. DeepGEMM README — https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md ("SM90 or SM100"; Mega MoE needs PyTorch ≥ 2.9)
13. DSpark paper — https://arxiv.org/html/2607.05147v1 (mechanism, results, γ=5, lossless property)

Leads (unverified, search-snippet only — do not cite): vLLM blog 2026-08-14 (dspark adaptive verification), LMSYS blog 2026-07-06 (dspark-sglang), Spheron/Fullstack explainers.

Categories: code/history 10, documentation 1 (repo Q&A), literature 1 (arXiv), practitioner 0 verified / 3 leads. 13 verified sources.

## Methods note

Fetched pages via URL→markdown reader (summarizer-mediated); load-bearing numbers cross-checked across README vs Q&A vs paper. `examples/offline_inference.py` fetch timed out — skipped (low value). GitHub code search for DSpark symbol locations not performed (auth-gated); located spec-decode surface via env flags + worker listing instead. The e4m3-vs-NVFP4 quant sentence and all format details are single-pass summarizer reads — flagged, not load-bearing. Did not search for DeepInfer forks or issue-tracker discussion of sm_120 (repo is 1 day old; PRs exist but not triaged).
