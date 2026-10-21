# T23 — jesdga95/basalt: recon for the Strata custom-2 campaign

- **Date:** 2026-10-20 (repo last pushed 2026-10-20T19:36Z — same day)
- **Scope:** What is basalt; does it apply to our Strata fork (Qwen3.8-Flash-Next MoE + MTP, RTX 5090 sm_120, single box); takeaway + cheapest evaluation. Read-only recon; no builds/boots.
- **VERDICT: APPLIES** (strongest possible). Basalt is a public, MIT-licensed sibling Strata fork that has already built, measured and documented almost exactly what custom-2 targets.

## Summary

`jesdga95/basalt` is a single-model local inference engine for **Qwen3.8-Flash-Next** (MoE: n_embd 2560, 48 layers = 36 GDN + 12 QSA full-attention, **512 experts top-10**, MTP head), for **NVIDIA Blackwell sm_120 + CUDA 13 only**, on Linux/Windows x64, behind an OpenAI-/Anthropic-compatible API. It credits "[Strata](https://github.com/Niko1221/Strata): the project Basalt forks (MIT)" — our parent. `fork:false` in the GitHub API because it is not a GitHub-side fork: its git history **imports all of Strata's history** (root commit 2026-09-24 = Strata's repo creation), with divergence starting around 2026-10-04..06. Since then: **1723 commits** (repo created 2026-10-06), extremely heavy AI co-authorship (`Co-Authored-By: Claude Sonnet 5.5` throughout), **no releases, no tags**. Upstream Strata (29.5k stars, multi-vendor app: NVIDIA 20–50 series, AMD, Intel, Mac) is the consumer product; basalt is the sm_120 specialist engine.

## Coverage vs our axes (all verified in-tree)

| Axis | Status | Evidence |
|---|---|---|
| sm_120 CUDA decode | **Direct hit** | [CMakeLists.txt](https://github.com/jesdga95/basalt/blob/main/CMakeLists.txt) hard-fails any arch ≠ 120; FP4-expert target built `120a-real`; PDL launch wrapper ([pdl.hpp](https://github.com/jesdga95/basalt/blob/main/include/basalt/kernels/pdl.hpp), `__CUDA_ARCH__ >= 900`); TMA rules in [CONTRIBUTING.md](https://github.com/jesdga95/basalt/blob/main/CONTRIBUTING.md) ("never `cp.async.bulk.shared::cluster`"); 58 files in [src/kernels/cuda/](https://github.com/jesdga95/basalt/tree/main/src/kernels/cuda) incl. persistent `mid_chain` GDN kernel, `dense_tc` int8 tensor-core dense path, thread-block-cluster decode test. Measured on **RTX 5090 @ 400 W** ([docs/precision/index.md](https://github.com/jesdga95/basalt/blob/main/docs/precision/index.md)). |
| MoE serving | **Direct hit** | Expert tiers: VRAM → pinned system RAM → optional second GPU as "peer expert tier"; per-pack expert profile; resident experts; `expert_cache/peer_experts/remote_experts` in [src/core/](https://github.com/jesdga95/basalt/tree/main/src/core); int8 hyper-connection copies (hc8). Serving = Python [serve/](https://github.com/jesdga95/basalt/tree/main/serve) (OpenAI, Anthropic, Responses APIs, concurrency, chunked prompts, web UI, vision sidecar). |
| Speculative decoding (MTP) | **Direct hit** | Drafter shipped inside the model file (`mtp/*` sections; drafter experts Q2_0 resident in VRAM; optionally a "self-distilled" tuned dense layer); plus a **suffix/n-gram drafter** ([src/spec/suffix_drafter.cpp](https://github.com/jesdga95/basalt/blob/main/src/spec/suffix_drafter.cpp), min-match 3) coupled to MTP; draft length from a cumulative-acceptance threshold ([docs/precision/index.md](https://github.com/jesdga95/basalt/blob/main/docs/precision/index.md)). Bit-exactness claim: 275/275 greedy prompts identical and full-vocab fp32 logits bit-identical between `--spec 0` and `--spec 7`. |
| Quant tooling / GGUF | **Direct hit** | [tools/ggufq.py](https://github.com/jesdga95/basalt/blob/main/tools/ggufq.py) = gguf-py dequantizers embedded from llama.cpp @ `3cf0325` (pinned in [THIRD_PARTY_LICENSES](https://github.com/jesdga95/basalt/blob/main/THIRD_PARTY_LICENSES)); `tools/gguf_reader.py`/`gguf_writer.py`; [tools/convert/convert.py](https://github.com/jesdga95/basalt/blob/main/tools/convert/convert.py) builds single-file `.basalt` packs from GGUF shards **or from NVIDIA NVFP4 safetensors** (experts kept bit-exact, dense quantized once, Q6_K). Policy: "quantized once, never twice"; mixed per-layer expert types (IQ2/IQ3/IQ4/Q2_0 mixes documented per pack). Prefill MMQ instances for 14 GGUF types incl. NVFP4: [src/prefill/mmq/](https://github.com/jesdga95/basalt/tree/main/src/prefill/mmq). |
| Ready packs | **Yes** | [HF jesdga/Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt): NVFP4, UD-Q4/Q5/Q6_K_XL, Q8_0, GSQ-RCO IQ3_S/IQ3_XXS/Q2_0, + convert assets (MTP inventory, tuned dense, draft vocab, gguf sources). |

## Published baseline on our exact GPU class

Their test rig: **RTX 5090 @ 400 W** + RTX 5060 Ti peer + 128 GB RAM ([docs/precision/index.md](https://github.com/jesdga95/basalt/blob/main/docs/precision/index.md)). Reference number (agent A/B, IQ3_S pack, spec 7, separate-kernel build): **6.675 ms/decode window, 324.6 tok/s** ([docs/unified-kernel.md](https://github.com/jesdga95/basalt/blob/main/docs/unified-kernel.md)). Quality: their KLD vs Q8_0 matches llama.cpp's own `llama-perplexity` on the same GGUFs (IQ3_S 0.15264 vs 0.1526).

## Takeaways for custom-2

1. **Existence proof + calibration target.** Our exact plan (Strata → sm_120-only MoE+MTP fork on a 5090) already works elsewhere; their numbers bound what we should expect, and their failures (full-layer fused kernel was *slower*, retired with a root-cause writeup) are free lessons.
2. **Clean MIT code source.** Chain: Strata MIT → basalt MIT → embedded llama.cpp MIT with headers intact. Lifting subsystems (kernels, spec controller, expert tiers, pack tooling) is license-clean if we retain attribution headers / THIRD_PARTY_LICENSES entries.
3. **Methodology worth adopting wholesale:** every change ships (a) parity test vs previous kernel + double-precision CPU reference, (b) microbench line (µs + GB/s), (c) quality gate (greedy-identity, or PPL/KLD vs Q8_0 through the **decode** path), 400 W labeling, ABAB same-session A/B. See [CONTRIBUTING.md](https://github.com/jesdga95/basalt/blob/main/CONTRIBUTING.md) and [docs/precision/](https://github.com/jesdga95/basalt/tree/main/docs/precision).
4. **Concrete designs to study before we write our own:** `mid_chain` persistent GDN kernel; two-stage verify head (Q6_K head screened through a 2-bit copy on int8 tensor cores); PDL wrapper; expert tier placement profiles; `.basalt` 2 MiB-aligned mmap/O_DIRECT single-file format ([docs/model-file.md](https://github.com/jesdga95/basalt/blob/main/docs/model-file.md)).

## Caveats

- **Moving target:** daily pushes, no tags/releases — pin a commit SHA before diffing; note [issue #16](https://github.com/jesdga95/basalt/issues/16) reports 3 GPU-tier test failures on `c5f9fa24` (unmodified HEAD at the time).
- **Divergence is one-way:** no evidence of upstream Strata syncs after early Oct (recent merges are internal branches); their tree no longer matches Strata main.
- **Their rig ≠ ours exactly:** headline numbers use a 5060 Ti peer tier + 128 GB RAM; single-GPU placement changes expert tiers. Published numbers are maintainer-run; agent benches are labeled dev A/Bs.
- **Single-model scope** — only useful to us insofar as custom-2 targets the same geometry (it does).
- Windows quality never measured (we are Linux; irrelevant but noted).

## Cheapest evaluation

1. **Zero-build (recommended now):** `git remote add basalt https://github.com/jesdga95/basalt && git fetch basalt main` in our fork; pin a SHA; produce per-subsystem diffs (`src/kernels/cuda`, `src/spec`, `src/core/{mtp,expert_cache,peer_experts}.cpp`, `src/prefill`, `tools/convert`) and a cherry-pick/port candidate list. Docs already give us their numbers — no boot needed to start deciding what to port.
2. **Adopt docs:** copy their precision/quality-gate structure into our campaign standards (text reuse, MIT).
3. **Later, needs approval (not read-only):** build with CUDA 13/CMake 3.24+ and run `bin/basalt-bench` with the smallest pack (GSQ-RCO-Q2_0) on our box to calibrate against their published baselines — or skip and use their numbers as-is until our own engine exists.

## Contradictions

- GitHub API says `fork:false`; the project says "a fork of Strata." Resolved: history import, not a GitHub fork relationship (root commit == Strata root commit; both repos still pushed 2026-10-20).
- An early untracked summary in this session guessed the parent lived under a "strata-ai" org — wrong; raw README + THIRD_PARTY_LICENSES both name Niko1221/Strata. No other material contradictions found.

## Open questions

- Will basalt ever cut releases/tags or sync upstream Strata? (No signals; watch the repo.)
- Single-GPU (no peer tier) 5090 numbers are not published separately — unknown delta for our box.
- Redistribution: packs are Qwen Community License 1.0 (not MIT) — matters only if we republish packs, not code.
- Do their per-pack expert profiles transfer to a single-GPU tier layout, or need retuning?

## Sources

1. Repo metadata: https://api.github.com/repos/jesdga95/basalt (created 2026-10-06, 1723 commits, MIT, C++, pushed 2026-10-20)
2. README: https://raw.githubusercontent.com/jesdga95/basalt/main/README.md
3. AGENTS.md: https://raw.githubusercontent.com/jesdga95/basalt/main/AGENTS.md
4. CONTRIBUTING.md (scope + quality gates): https://raw.githubusercontent.com/jesdga95/basalt/main/CONTRIBUTING.md
5. THIRD_PARTY_LICENSES (llama.cpp pin `3cf0325`, Strata MIT): https://raw.githubusercontent.com/jesdga95/basalt/main/THIRD_PARTY_LICENSES
6. docs/platforms.md (sm_120 + CUDA 13 only): https://raw.githubusercontent.com/jesdga95/basalt/main/docs/platforms.md
7. docs/model-file.md (.basalt format, NVFP4 pack build): https://raw.githubusercontent.com/jesdga95/basalt/main/docs/model-file.md
8. docs/precision/index.md (per-tensor decode precision; RTX 5090 rig; KLD vs llama.cpp): https://raw.githubusercontent.com/jesdga95/basalt/main/docs/precision/index.md
9. docs/unified-kernel.md (mid_chain kernel; retired full-layer fusion + numbers): https://raw.githubusercontent.com/jesdga95/basalt/main/docs/unified-kernel.md
10. CMakeLists.txt (arch enforcement, `120a-real` FP4): https://raw.githubusercontent.com/jesdga95/basalt/main/CMakeLists.txt
11. Tree: https://api.github.com/repos/jesdga95/basalt/git/trees/main?recursive=1 (550 files; src/kernels/cuda=58, src/spec, src/prefill/mmq, tools/convert)
12. Speculative code: src/spec/suffix_drafter.cpp, src/core/mtp.cpp (raw, same repo)
13. Commits + merges: https://api.github.com/repos/jesdga95/basalt/commits (root commit 2026-09-24; recent 2026-10-19/20; internal merges only)
14. Issues: https://api.github.com/repos/jesdga95/basalt/issues?state=open (#13..#17)
15. Parent: https://api.github.com/repos/Niko1221/Strata + README (29,509 stars, multi-vendor, "Qwen3.8-Flash-Next … 125B")
16. HF packs: https://huggingface.co/api/models/jesdga/Qwen3.8-Flash-Next-Basalt (8 packs + convert assets)

## Methods note

Fetched directly via authenticated `gh api`, raw.githubusercontent.com, and HF API (primary). Unproductive: webReader against api.github.com failed twice with MCP network errors (switched to `gh api`); commit pagination by page number overshot (used `page=1723&per_page=1` for the root). Code search API confirmed sm_120 appears only in docs/CI (arch set by CMake), `__CUDA_ARCH__` guards in 11 kernel files. No builds or clones performed (read-only recon, per task).
