---
type: research
id: upstream-delta-digest
created: 2026-11-18
repo: /home/svg/LLM/strata (https://github.com/Niko1221/Strata)
delta: 9f7d977d2 (v0.2.11+3) -> c77572806 (v0.2.15, origin/main)
commits: 1261 total, 1008 non-merge
---

# Upstream delta digest: v0.2.11 -> v0.2.15 (1261 commits)

Study of the fetched objects between local HEAD `9f7d977d2` (described `v0.2.11-3-g9f7d977d2`,
engine CMake version 0.1.57) and `origin/main` `c77572806` (exactly the `v0.2.15` tag, 0 commits
after; engine CMake version 0.1.61).

## Mid-study event: the checkout was already updated

The reflog records an external `git pull --ff-only` at **2026-11-18 17:53:19 +0200** that
fast-forwarded local `main` from `9f7d977d2` to `c77572806` while this study ran:

```
c77572806 HEAD@{2026-11-18 17:53:19 +0200}: pull --ff-only: Fast-forward
9f7d977d2 HEAD@{2026-11-06 01:30:21 +0200}: checkout: moving from ffa80d731 to main
```

State at study end: tracked tree clean (`git status --porcelain` shows only untracked operator
files: `bench/results/2026-10-28-community-rtx-5090-deepseek-v4-1-flash/`, `custom-arms.env`,
`ds4-slots/`, `data/expert-profile-stocktest-learned.bin`). All comparisons below were made
against explicit SHAs (`9f7d977d2`, `c77572806`), so the pull does not invalidate any claim.
`9f7d977d2` stays reachable for re-verification.

## (a) Major feature waves, in order

| Tag | Commits | Date | Engine | Content |
|---|---|---|---|---|
| v0.2.12 | 98 | Nov 5-7 | 0.1.58 | feed `models` member; `undeclared_tool_calls` config (75acfca32); tool-call parser picked from chat template (e7d484463); DFlash2 tuned drafter (9b02c62ab); config snapshot goldens (c10b74697); Mac prep-tool installer fix (146df3c46) |
| v0.2.13 | 212 | Nov 7-8 | 0.1.59 | **shape engines** (8a9feaedd, `strata_control/shape_engine.py`); geometry-compiled builds `STRATA_GEOMETRY` / `STRATA_GEOMETRY_FILE` (af7759b69); `strata --geometry` (febc0226f); ggml + llama.cpp + gguf-py vendored (`third_party/strata-ggml` 067b4dda5, `third_party/strata-llama` 463940282, `third_party/strata-gguf-py` 525c0be13); setup no longer downloads llama.cpp (6f273fc27); CMakeLists split per backend (8b1667701); `strata_gguf`/`strata_quants` quant layer (f9bbeeb44, 8a2b0f5f0); `CMakePresets.json` (e396ce821); settings register (6ea300dde); Intel Arc app setup (1130d824a) |
| v0.2.13.1 | 7 | Nov 9 | 0.1.59 | Control-only hotfix, Linux x86_64 (b08a069d6) |
| v0.2.14 | 895 | Nov 8-15 | 0.1.60 | the bulk wave, below |
| v0.2.14.1 | 6 | Nov 15 | 0.1.60 | Control-only drafter fixes (2ad883b61) |
| v0.2.15 | 46 | Nov 15-16 | 0.1.60 (packages carried by absolute URL, e39a52e33) | Drafting Off/Auto persistence (c77572806, 15c14f6c4, 1b4018870); `--backend amd` picks the AMD card's package on mixed-GPU PCs (fdfca1d9d); libatomic.so.1 bundled in Linux AMD packages (1a5c6dd19); Mac DFlash2 default drafter (3e8d3b294); serve boolean tool args (1aabfb1ce) |

The v0.2.14 wave (895 commits) decomposes into:

1. **DeepSeek removed** — 7-part series 5d5029be4 -> f1fce3286 (Nov 11): `src/dsv41/` deleted
   (e5806740a), serve code paths removed (5369576ff), configs refused at startup (12355203d),
   feed packages frozen ("Keep frozen", 4f66847b4).
2. **Builds without llama.cpp MMQ** — `-DSTRATA_MMQ=OFF` (e2bf2e451), then MMQ off by default
   (f40f8caca); Strata's own fused prompt kernels become the default across card classes
   (454305361, ef0fb17ab, df723eb5a, 9e3a36714, 3d302a600, 64e95049a).
3. **int8 expert kernel family** — `native_expert_i8` research kernel (fe67f7fbe) matures to
   default on sm_86 (7f8c92f07), gfx12 (5410b3bc0), RX 6000 (9783e09f9), integrated gfx11 Linux
   (69394994a); NVFP4 gate/up+down on the int8 kernels (cf9d86fba, 82aeb32b2).
4. **Image generation** — own DiT step, VAE decoder, noise generator for sd-cli
   (1705c8b9f, f7d8aa28e, a9f27d068); `src/image/` added; own picture reader becomes the default
   vision tower (59e075881).
5. **Drafting economics** — `--draft-costs` table learned once then fixed (4c999bcde), spec
   window learner family (4079483b6 .. fc66df986), DFlash2 per-device defaults incl. Blackwell
   cc 12.0 >= 48 GB conf 0.55 window 8 (6ec4d68ab, a04860fad).
6. **Memory/platform** — park pools on by default (01e21ef20), container memory-limit reads
   (81ac7c8ef), NUMA node pinning (716ebae14), expert-arena file-cache credit #5327 (28bd0185f),
   one network policy for install/update/launch (cc4f2dfe3), run manifest `STRATA_MANIFEST`
   (89f4fa18e), settings register regeneration batches.
7. **Blackwell / sm_120 specifics** — 128-row fused tiles on sm_120 buckets (79740c23c 80+ SMs,
   083b1ac82 100-149 SMs, 5b6396fda 40-79 SMs), two Q8_0-down pairs on sm_120 100-149 SMs
   (2248e6361), DFlash2 two parallel requests on one large Blackwell card (e41b329aa),
   `STRATA_IQ4_W4A8` on sm_120 documented (7bf5ec4eb, d735d66a4).

## (b) Breaking changes for config / CLI / build

**Config (serve run-config JSON):**
- Keys **added**: `"draft_costs": true|false` (4c999bcde; platform default on CUDA, HIP keeps
  live costs — server.py `draft_costs_default`) and `"undeclared_tool_calls": "call"|"text"`
  (75acfca32, env `STRATA_UNDECLARED_TOOL_CALLS`). Verified by key-set diff of server.py
  `cfg.get(...)` reads between `9f7d977d2` and `c77572806`: no key was removed.
- **DeepSeek configs refused**: `serve/server.py` main calls `strata_control.retired.config_reason(cfg)`
  and exits 2 before starting the engine (introduced 5d5029be4, wired 12355203d). A stored
  DeepSeek run config must be deleted, not migrated.
- **Launcher-owned env copies removed** (d1121b893): parallel stack, GDN stack, `STRATA_HC_Q8`,
  Metal fp16 windows, hipBLASLt table path are now engine-owned defaults; older configs carrying
  them are tolerated but the engine decides. Setup no longer writes `STRATA_HC_Q8` into
  Unsloth UD configs (741310713).
- Unchanged keys confirmed in both trees: `exe`, `args`, `user_args`, `cwd`, `tokenizer`,
  `model_name`, `host`, `port`, `api_key`, `gpu`, `layer_split`, `split_serve`, `parallel`,
  `vision`, `mtp`, `drafting`, `vram_elastic`, `vram_segment_mib`, `draft_vocab`, `images`, `env`.

**CLI (serve/server.py):**
- `--engine strata --config X --port N --host H` **still accepted verbatim**: at `c77572806`
  line 9954 `ap.add_argument("--engine", choices=["mock", "strata"], default="mock")`; `--config`
  (9955), `--host` (9957), `--port` (9970) unchanged. New flags: `--draft-costs` (4c999bcde),
  `--shutdown-timeout` semantics change — a stop without it refuses new requests while the engine
  closes (3cf22f5ef). `--split-serve PREFILL,DECODE` unchanged.
- Root entry points `chat.py` and `highlight.py` moved to `tools/` (ff6cd56cb); user entry points
  `setup.py`, `strata.py`, `START-HERE.bat`, `UPDATE.bat`, `BENCH.bat`, `bench.sh` stay.

**Build recipe:**
- CMakeLists.txt (2813 lines changed) split into `cmake/backends/{cpu,cuda,hip,sycl,metal,quant}.cmake`,
  `strata_ggml.cmake`, `strata_geometry.cmake`, `strata_gpu_engine.cmake` (8b1667701).
- ggml/llama.cpp are **vendored** (`third_party/strata-ggml`, `third_party/strata-llama`);
  `STRATA_GGML_DIR` is now override-only (e43ff9885); setup performs no llama.cpp download
  (6f273fc27); `third_party/stb` added (6f19a9662).
- `CMakePresets.json` added: `cpu-check`, `cpu-check-native`, `cuda-compile` presets (e396ce821).
- **Engine default build has no llama.cpp MMQ kernels** (f40f8caca); Strata's own fused/int8
  prompt kernels serve instead. This changes prompt-path arithmetic defaults vs 0.1.57 builds.
- **Geometry compilation**: a build targets exactly one model geometry via
  `STRATA_GEOMETRY` / `-DSTRATA_GEOMETRY_FILE` (af7759b69); packaging scripts take `SHAPE=<geometry>`
  (8b92e9ace); `strata --geometry` reports it (febc0226f); an engine refuses a GGUF of another
  geometry (motivating 8a9feaedd).
- Intel Arc source build requires oneAPI 2026.1 (fd5b3c455, 456b07cbe).
- Engine version: CMake `project(strata VERSION 0.1.57)` -> `0.1.61`; release packages shipped
  0.1.58 (7be42963e), 0.1.59 (eddbf228d), 0.1.60 (38f63bc59); v0.2.15 re-ships 0.1.60 packages
  (e39a52e33).
- New requirements file `requirements-dev.txt`; `history/` and `research/` top-level dirs added
  (8c743a697, 3d26434f2); retired env switches: `STRATA_DF2_GATE`, `STRATA_DF2_ROW_MS` (e62f3c1ff),
  `STRATA_GR_Q32_PL` (64fac3177), `STRATA_METAL_ROUTE`/`_PARTIAL`/`_WPF_MIB` (41b87a4bd),
  `STRATA_MMVQ_PF`/`_SG` (dff2c9114).

## (c) Engine-decoupling verdict

**Rumor "decoupled from the standard engine process and now have their own": FALSE as stated,
TRUE in a narrower sense.** There is no new engine *process* model. `serve/server.py` still
spawns exactly one engine binary from the config's `exe`:

```
self.proc = popen("the Strata engine", [exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE, ...)
```
(c77572806 line 1163; identical call at 9f7d977d2 line 1210). `--engine` choices remain
`["mock", "strata"]`.

What upstream actually decoupled is the engine *binary identity*:

- **Shape engines** — 8a9feaedd (Nov 8): a new-shape Qwen4 model ships its own engine component
  `strata-q4-<shape>`, packaged by `package_engine.py --shape`, installed into
  `<components>/shapes/<shape>/engine/` by `strata_control/shape_engine.py`, referenced in the
  run config as `component:strata-q4-lite` and resolved by `strata_control/engine_component.expand`.
  Model family `qwen4exp_<shape>` selects it. The standard `strata-engine` component (Flash-Next
  geometry) remains the default for every existing model.
- **Geometry-gated builds** — af7759b69, febc0226f, b2f268403: an engine binary is compiled for
  one geometry and refuses others; install runs `strata --geometry` against the signed manifest.
- **strata-split** (prefill/decode two-process mode, `serve/split.py`, `--split-serve`) already
  existed at 9f7d977d2 (server.py:1151-1152) — not new in this range.
- **DeepSeek's engine was removed**, not decoupled (e5806740a); its feed packages stay frozen
  (4f66847b4).

Consequence for this estate: a Qwen3.8-Flash-Next GSQ pack keeps running on the standard
`strata` engine; only `qwen4exp_*` family models pull a second engine binary.

## (d) Recommended update procedure

The checkout is **already at origin/main** (external ff-only pull, reflog above). Verification
and a from-scratch equivalent:

```bash
cd /home/svg/LLM/strata
git fetch --prune origin
git status -sb                       # expect: ## main...origin/main, no ahead/behind
git log --oneline -1                 # expect: c77572806 (v0.2.15)
git describe --tags HEAD             # expect: v0.2.15
# old baseline stays reachable for A/B:
git worktree add ../strata-0.2.11 9f7d977d2    # optional, keeps the old tree for head-to-head
```

Rebuild (no network needed for ggml — vendored):

```bash
cmake --preset cpu-check && cmake --build --preset cpu-check        # fast check build
# CUDA engine, Flash-Next geometry (RTX 5090 = sm_120, cc 12.0):
bash packaging/build_cuda_linux.sh    # or the Windows/MSVC equivalent; SHAPE= unset = flash_next
strata --geometry                     # verify: flash_next
```

Then serve head-to-head exactly as before:
`python serve/server.py --engine strata --config <cfg.json> --host 127.0.0.1 --port 8095`.
Config needs no migration unless it names DeepSeek (refused, exit 2) or carries the retired
launcher env copies (d1121b893); optional new keys `draft_costs`, `undeclared_tool_calls`.

Boundary: engine package version on releases is 0.1.60 while CMake says 0.1.61 — a self-built
engine reports 0.1.61 and may skip a feed's `min_engine` comparison in the wrong direction; pin
the built engine by `strata --geometry` plus explicit config `exe` path when mixing with feed
packages.

Companion studies: `2026-11-18-upstream-build-recipe.md`,
`2026-11-18-upstream-engine-decoupling.md`, `2026-11-18-upstream-nvfp4-study.md`.
