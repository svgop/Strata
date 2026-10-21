# T23 — Recon: mw00/project-maya

Date: 2026-10-20 (repo pushed 2026-10-21, v1.1.20). Evaluator: researcher, read-only, no builds/boots.
Target: https://github.com/mw00/project-maya

## Summary

Project Maya is a complete local-LLM stack (CUDA/HIP inference engine + OpenAI/Anthropic/Responses-compatible
server + web dashboard + quant tooling) that runs GLM-5.3-Flash (321B MoE) and DeepSeek-V4.1-Flash-1017 (748B MoE)
on 1–16 consumer GPUs. It is **not a GitHub fork** (`"fork": false`); its first commit (2026-10-07,
a813352e) declares: "Built on Strata (MIT, github.com/Niko1221/Strata)" — Strata's engine "rewritten for
GLM-5.3-Flash (the expert tiers across VRAM, RAM and SSD, the layer split, MTP decoding)".

**This makes Maya a sibling of our fork, not a stranger.** Upstream Niko1221/Strata is itself the
Qwen3.8-Flash-Next engine ("Qwen3.8-Flash-Next on any consumer hardware", 29.5k stars) — i.e. the same base our
custom-2 branch descends from. Maya kept 122 files under `include/strata/` and the `strata::` namespace, including
`include/strata/core/mtp.hpp` / `src/core/mtp.cpp`, whose comments describe the **Qwen-lineage MTP drafter**
("vLLM 0.30.0 `qwen4_exp` MTP", 512-expert top-10 router, QSA attention, acceptance 0.89/0.86/0.85 at draft steps
1–3, measured in `tools/mtp_probe.py`) with Strata plan references ("plan v0.3 P6", 2026-09-23 bench dates).

## Verdict: APPLIES (strong)

Closest living relative of our codebase, developed and measured on our exact GPU (sm_120/RTX 50). Four requested
dimensions:

| Area | Applies? | Evidence |
|---|---|---|
| sm_120 CUDA decode | Yes | `CMakeLists.txt` defaults `CMAKE_CUDA_ARCHITECTURES=120`, comment: "developed and measured on sm_120 (RTX 50)"; sm_75/86/89/120 + Pascal; "Blackwell 12.8 or newer" (CUDA). README speed table: RTX 5090 32 GB, 23.0 tok/s decode; "30+ on a single RTX 5090" |
| MoE serving | Yes | Expert tiers VRAM→RAM→SSD, LRU expert pool, CPU expert lane via ggml-cpu dot products (IQ types), threaded expert-cache swaps, layer split / expert / tensor parallel, `--parallel` batched decode (x1.4–2.0 aggregate). 56 .cu files incl. `native_moe.cu`, `s2_gemv*.cu`, `shared_expert.cu`, router kernels |
| Speculative decoding | Yes | `src/spec/` controller arbitrates none/suffix-lookup/MTP per step maximizing E[tokens committed]/T(step) with EMA per-position acceptance and a measured verify-cost model; `draft_policy.cpp`; ngram `suffix_drafter.cpp`; GLM NextN `glm_mtp.cu` (multi-GPU split, #290); **DFlash2 drafter** (+12–44% over the MTP block, via `third_party/beellama.cpp`); DeepSeek DSpark. Active branch: `mtp-rework` |
| Quant tooling | Yes (GGUF incl.) | `tools/maya_quant/` + `calibrate*.py` (error-feedback rounding, scale refinement, in-engine KL/NLL scoring `MAYA_GLM_SCORE`); Maya-S/S24/M/L GGUF quants from Z.ai FP8, measured 97.7–99.2% zero-shot retention (bench/results/MAYA-*.md); runs unsloth GGUF (Q6_K_XL/Q8_0, UD-Q4/Q8_K_XL) via ggml MMQ kernels; EXL3 incl. half-integer bpw; `gguf_reader/writer/fix_arch.py`, `mtp_rt.py` (runtime Q8_0 of MTP projections) |

## What does NOT transfer

- Model-specific kernels: `glm_kda.cu`, `glm_dsa.cu`, `glm_hc.cu`, `fused_gdn.cu`, `qsa*.cu` consumers, GLM chat
  template, whole `src/ds4/` DeepSeek engine (sparse attention, Engram tables, DSpark, vision tower).
- Scale mismatch: Maya's raison d'être is streaming 96–340 GB models from RAM/SSD; on our single 5090 with a
  Qwen-class MoE the tiering machinery is low priority (KV quant and `--parallel` batching still relevant).
- Divergence cost: 453 commits in 15 days, engine "rewritten" — blind cherry-picks will not apply cleanly; the
  `strata`-named files are the closest to our base.
- License: code MIT (clean); DFlash2 drafter weights are CC BY-NC-ND 4.0 — non-commercial, opt-in download
  (README credits); exllamav3 and beellama.cpp third_party code MIT.

## Takeaways for our fork

1. **Proven sm_120 baseline**: the engine is developed *on* RTX 50 hardware — their kernel choices (fused prompt
   experts, MMQ scratch accounting, tensor-core decode attention, quantized KV q4/q8/K8V4) are directly readable
   reference implementations for our sm_120 decode path.
2. **Spec-decode economics framework**: the `src/spec/` controller's cost model (dense/experts ratio per verify
   window size, expert hit-rate, sync cost, min-gain 5% gate) is a principled upgrade over a fixed MTP k — and its
   mtp.hpp/mtp.cpp is our model-lineage drafter, evolved. Diff against ours first.
3. **Quant methodology template**: QUANT-PLAN.md (calibration corpus in chat template, error-feedback rounding,
   KL<=0.05 target, in-engine scoring) is a ready recipe for quantizing Qwen3.8-Flash-Next releases.

## Cheapest evaluation (read-only, ~2–4 h)

1. Shallow clone (`git clone --depth 50 https://github.com/mw00/project-maya`) and diff `include/strata/`,
   `src/spec/`, `src/core/mtp.cpp` against our fork's equivalents.
2. Read `bench/results/MAYA-S.md`, `docs/QUANT-PLAN.md`, `docs/SETTINGS.md#drafting-with-dflash2`.
3. Watch `mtp-rework` branch + issues #290/#265 for MTP drafter economics data.

## Contradictions

- mtp.hpp's comment describes a 512-expert/top-10 Qwen-style MTP layer while Maya's GLM docs say 288-expert/top-8
  with a separate `glm_mtp.cu` NextN drafter. Resolution: mtp.cpp is inherited Strata (Qwen) code retained in the
  tree; whether it is reachable in Maya's GLM-only flow is unverified — treat its header comment as Strata-era
  documentation, and diff against our base rather than trusting it as current.

## Open questions

- Is `src/core/mtp.cpp` compiled/used in Maya's GLM config, or dormant Strata legacy? (Check CMake target list.)
- Does DFlash2's beellama.cpp drafter forward generalize beyond GLM's architecture?
- What exactly changed in `mtp-rework` (unmerged branch, aac3d492)?

## Sources

- https://github.com/mw00/project-maya (README: models, speed table incl. RTX 5090, requirements CUDA 12.x/12.8+
  Blackwell, credits/lineage) — fetched raw 2026-10-20
- https://api.github.com/repos/mw00/project-maya (metadata: fork=false, MIT, 1334 stars, created 2026-10-06,
  pushed 2026-10-21)
- https://api.github.com/repos/mw00/project-maya/commits (453 commits; first commit a813352e "Built on Strata")
- https://api.github.com/repos/mw00/project-maya/releases (v1.1.11–v1.1.20 notes: DFlash2, EXL3, GGUF, tiering)
- https://raw.githubusercontent.com/mw00/project-maya/main/CMakeLists.txt (arch 120 default, sm_120-dev remark)
- .../main/include/strata/spec/controller.hpp (cost model, K_MAX=8, EMA acceptance)
- .../main/include/strata/core/mtp.hpp (Qwen-lineage MTP drafter, acceptance 0.89/0.86/0.85)
- .../main/src/core/mtp.cpp (QSA/router_top10/s2_expert_grouped includes)
- .../main/docs/QUANT-PLAN.md (Maya-S/L quant pipeline from Z.ai FP8)
- .../main/CHANGELOG.md (v1.1.19/v1.1.20 details)
- https://api.github.com/repos/Niko1221/Strata (upstream: "Qwen3.8-Flash-Next on any consumer hardware")
- git tree (recursive, 1045 files): src/spec, src/kernels/cuda (56 .cu), tools/maya_quant, tools/mtp_*.py

## Methods note

Efficient: GitHub REST API for metadata/tree/commits/releases + raw file fetches; no search-engine detours needed.
Unproductive: none. Not done (deliberately): reading GLM kernel bodies, CONTRIBUTING.md, issues other than those
cited in release notes, and HF quant repos — out of scope for a verdict-level recon.
