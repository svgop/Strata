---
type: research
id: upstream-engine-decoupling
date: 2026-11-18
repo: /home/svg/LLM/strata (operator-owned clone of Niko1221/Strata)
delta: 9f7d977d220e (v0.2.11-3, 2026-11-05) .. c775728069 (v0.2.15 == origin/main, 2026-11-16), 1261 commits
question: Did upstream "decouple from the standard engine process and now have their own engine process"?
---

# Upstream engine decoupling: verdict

## Verdict

The claim is true, in three separate landings inside the 1261-commit window (v0.2.11 -> v0.2.15):

1. **Per-model-shape engines (the decoupling the operator heard about).** A Qwen4 model whose geometry is not Flash-Next ships with **its own engine binary** as a separate signed component (`strata-q4-<shape>`), installed beside — never instead of — the standard `strata-engine`. Each such model runs its own engine process. Branch `f-0213-shape`, first commit `8a9feaedd` (2026-11-08 20:18), merged `d5d5b0837` (2026-11-08 22:20), first released in **v0.2.13** (tag 2026-11-09).
2. **DeepSeek's dedicated engine retired.** The only historical second engine (`strata-dsv41`) is removed; DeepSeek configs are refused outright (`strata_control/retired.py`, owner wording 2026-11-11, "DeepSeek out" 7-part series, e.g. `12355203d`). Released in **v0.2.14** (tag 2026-11-15).
3. **Host app decoupled from llama.cpp's llama-server.** The packaged app's fallback that spawned a llama-server from `STRATA_LLAMA_SERVER` or `PATH` for imported GGUFs is deleted; the app now loads **only** through the Strata engine. `b9b752d8e`, merged `f5ef26889` (`f-no-llama-server`, 2026-11-15). Released in **v0.2.15** (tag == origin/main == `c7757280`).

Not part of the change: no daemon, no socket/shared-memory IPC, no renamed engine entry point. The engine is still a resident child process `[exe, --serve]` talking stdin/stdout line protocol; `serve/server.py`'s spawn, watchdog, and restart model are unchanged.

## 1. Process architecture: 9f7d977d vs origin/main

Spawn site identical in shape at both refs:

- 9f7d977d `serve/server.py:1210`: `self.proc = popen("the Strata engine", [exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE, ...)` (role "engine", serve/lifeline.py kill-on-close child)
- origin/main `serve/server.py:1163`: same call, same arguments, same role.

Unchanged at both refs: `ENGINE_QUIT_WAIT_S/TERM/KILL` teardown ladder, `ENGINE_SILENCE_S` (300 s) and `ENGINE_STALL_S` (90 s) frozen-engine watchdogs, `ENGINE_READY_S` (900 s), `EngineDied` -> next request restarts the engine, `start_replicas` / `start_split`, and the separate on-demand `strata-vision` picture-reader process (`_start_reader`; main adds `_start_gpu_reader` retry logic for DGX Spark only). `serve/winjob.py` +88 lines: Windows `CREATE_SUSPENDED` job containment for MCP children — process hygiene, not decoupling.

IPC delta (string-literal diff of the protocol surface, 9f7d977d vs origin/main):

| Direction | Lines |
|---|---|
| Removed (DeepSeek/DS4 era) | `rid=` suffix on GEN/GENI; `STRATA_DS4_AUTO_PIN`, `STRATA_DS4_RESTORE_*`, `STRATA_DS4_SLOT_*`, `STRATA_QSA_SLOTS`, `STRATA_GDN_*`, `STRATA_BATCH_ATTN_ROWS`, `STRATA_BATCH_GDN_GROUPS`, `STRATA_BATCH_SELF_COMMIT`, `STRATA_CKPT_POOL*`, `STRATA_SLOT_*` |
| Added | `ERR ...`; `READY <n_embd>`; `READY <n_embd> <reader> [fallback]`; `GEN hold=N:ID` (reasoning-hold budget); `STRATA_DRAFT_COSTS_DIR`; `STRATA_UNDECLARED_TOOL_CALLS`; `STRATA_VIT_PROF`; `leg_keys()` rename of the GEN keys helper |

Same pipe protocol family (GEN / GENI / BGEN / BGENI / T / PP / READY / SWAIT / INFO), no new transport.

exe resolution: unchanged code path. `cfg["exe"]` -> `with_component_engine(cfg)` -> `strata_control/engine_component.expand` (REFERENCE_PREFIX `"component:"` already existed at 9f7d977d). The only behavioral change: `expand` now also resolves shape component ids through `strata_control/shape_engine.py`; an unknown component still raises `EngineMissing`.

## 2. The shape-engine mechanism (strata_control/shape_engine.py, new)

From the module docstring (origin/main):

- An engine is compiled for ONE model geometry (`-DSTRATA_GEOMETRY=...`, `cmake/strata_geometry.cmake`; geometries at `include/strata/core/geometries/{flash_next,q4_lite,q4_mid,q4_big,q4_deep}.hpp` — none of these exist at 9f7d977d) and refuses a GGUF of another geometry at load. The standard engine is built for Flash-Next and is untouched ("Flash-Next engine hash unchanged", merge `d5d5b0837`).
- Component id: `q4_lite` -> `strata-q4-lite`; feed family `qwen4exp_<shape>`; feed member `shape_engines`; older Controls ignore both, so they never offer the model.
- Install location: `<components>/shapes/<shape>/engine/<version>-<platform>-<backend>[-<variant>]/strata` (binary still named `strata`, Windows `strata.exe` — `shape_engine.exe_name()`), never inside the Strata engine's folder; manifest id is the shape's component and carries `"shape"`, so the Strata engine's scan never reads it.
- Use: the model's run config says `"exe": "component:strata-q4-lite"`; every start resolves it via `engine_component.expand` (full SHA-256 + signature check, as for the Strata engine).
- Fetch: setup (`strata launch`, app Download) pulls it through the signed feed when the model is chosen; `strata update` updates it in place; starting never downloads.
- Hardening: `8920bd493` (owner-approved wording, 11-08), `6dba26279` (titles over geometry names), `b2f268403` (2026-11-09: at install, Control runs `strata --geometry` on the candidate and refuses a binary that cannot name its shape / names a different one).

## 3. Top-level tree (question 2)

No new engine repo vendored; no `engine-nv/`, `runtime/`, `kern/` or sibling of `metal/` at origin/main.

| New tracked at origin/main | Removed vs 9f7d977d |
|---|---|
| `history/` (LINK-BASELINE.md, README.md, apps, docs, packaging, tools) | `chat.py` |
| `research/` (stage-reserve-proposal, tools) | `highlight.py` |
| `CMakePresets.json`, `requirements-dev.txt` | — |

`strata_control/` module delta: `shape_engine.py` and `retired.py` are the load-bearing additions; `engine_component.py` gains `shape` field, `read_manifest(folder, component_id)`, `read_manifest_any`, backend aliases (`amd`->hip etc.); `_installed_dsv41_version` and dsv41 update offers deleted. Local-only untracked dirs (`engine/`, `engine-dsv41/`, `engine-dsv41.work/`, `ds4-slots/`) are the operator's own builds; upstream never tracked them.

## 4. Commit evidence (delta log, 9f7d977d..origin/main)

| Hash | Date | What |
|---|---|---|
| `8a9feaedd` | 11-08 | Shape engines introduced: feed `shape_engines`, `package_engine.py --shape`, `strata_control/shape_engine.py` (314 lines), run config `component:strata-q4-lite`, `strata update` support; tests `tests/terminal/test_shape_engine.py` (611 lines), `tools/test_setup_shape.py` |
| `d5d5b0837` | 11-08 | Merge `f-0213-shape` (`61d883315`); owner sign-off 11-08 "separate components"; first in tag v0.2.13 |
| `b2f268403` | 11-09 | Install refuses an engine that cannot answer `strata --geometry` |
| `12355203d` | 11-11 | DeepSeek out (2/7): every DeepSeek entry path refused in plain words |
| `d6fd26186` | 11-14 | Version 0.2.14 / engine 0.1.60 (DeepSeek-free line) |
| `b9b752d8e` + `f5ef26889` | 11-15 | `f-no-llama-server`: `apps/host/server/supervisor.mjs` -172 lines, load route 409 in real-engine mode, `tests/no-llama-server.test.mjs`; first in tag v0.2.15 |

Tag timeline: v0.2.11 11-05, v0.2.12 11-07, **v0.2.13 11-09 (shape decoupling ships)**, v0.2.14 11-15 (DeepSeek out ships), v0.2.15 11-16 (llama-server removal ships; tag == origin/main).

## 5. Impact on `serve/server.py --engine strata`

- CLI unchanged: `ap.add_argument("--engine", choices=["mock", "strata"], default="mock")` is byte-identical at both refs; `--engine strata` still requires `--config`; config schema (exe, args, cwd, tokenizer, model_name) unchanged.
- A config naming `component:strata-engine` or a plain exe path resolves exactly as at v0.2.11 — the Flash-Next lane works unchanged. The old CLI is compatible.
- A shape model's config (`component:strata-q4-lite`) starts its own engine binary from `shapes/<shape>/engine/` through the same spawn and protocol; nothing in the launch command changes.
- **Break for the operator's DeepSeek lane:** v0.2.14+ refuses DeepSeek V4.1 configs before anything starts (`retired.py` D1: "{name} is a DeepSeek model. This version of Strata no longer runs DeepSeek models, so it was not started."), and `serve/server.py` no longer carries the DS4 surface (`STRATA_DS4_*`, `rid=`, `serve/test_dsv41.py`, `test_ds4_restore_points.py`, `test_ds_parallel.py` all deleted). Pulling main onto the working clone strands the local `engine-dsv41/`, `ds4-slots/`, `launch-deepseek-v41-8081.sh` and `strata-deepseek-*.json` flows; the engine binaries still run but server-side DS4 handling is gone.

## Boundary

- The mid-session `git pull --ff-only` (reflog 2026-11-18 17:53, 9f7d977d -> c7757280) was executed by another process during this investigation. All HEAD-vs-main diffs above were captured before the pull and are reproducible against `9f7d977d` explicitly.
- Engine-side (C++/CUDA) sources were read only through build config and test fixtures (`cmake/strata_geometry.cmake`, `include/strata/core/geometries/`, packaging tests); the `--serve` loop inside the engine binary itself was not audited line-by-line — the server-side contract is the evidence base here.
- `history/` and `research/` contents were listed, not read; they are documentation moves, not process changes (no build or serve file references them).
