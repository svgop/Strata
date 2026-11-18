---
type: research
date: 2026-11-18
subject: Verified Linux CUDA engine build recipe for strata origin/main (product 0.2.15, engine 0.1.61) on this box
repo: /home/svg/LLM/strata (operator clone of github.com/Niko1221/Strata, READ-ONLY at time of research)
refs: HEAD 9f7d977d -> origin/main c7757280, merge-base == HEAD (ff-only pull verified possible)
---

# Upstream build recipe: Linux CUDA engine at origin/main

## Verified environment (evidence, not assumption)

| Item | Value | Evidence |
| --- | --- | --- |
| CUDA toolkit | 13.4 (V13.4.92) | `/usr/local/cuda -> /etc/alternatives/cuda` (cuda-13.4); `/usr/local/cuda/bin/nvcc --version` |
| GPU | RTX 5090, compute_cap 12.0 | `nvidia-smi --query-gpu=name,compute_cap` |
| Driver | 615.71.09 | `nvidia-smi` (cuda13 release floor: 580.65) |
| Host gcc | 16.2.1 (Fedora 45) | accepted by nvcc 13.4 — probe compile of C++20 for `sm_120a` ran clean |
| cmake / ninja | 4.4.3 / 1.13.2 in `.venv/bin` | matches `requirements-build.txt` pins exactly (`cmake==4.4.3`, `ninja==1.13.2`) |
| CPU / RAM / disk | 16 cores / 184 GB / 794 GB free | `nproc`, `free -g`, `df -h` |
| Git state | merge-base == HEAD, working tree untracked-only | `git merge-base HEAD origin/main` == `git rev-parse HEAD`; `git status --porcelain` |

## Git update path

`git pull --ff-only` works: HEAD (`9f7d977d`) is an ancestor of `origin/main` (`c7757280`), 1261 commits behind. Working tree has untracked files only (bench results, engine-dsv41 legacy dirs, configs) — the pull cannot touch them. Project-native alternative: `./update.sh` (SOURCE_INSTALL.md → Updating: pulls source changes, updates dependencies, refreshes installed configs without starting the model).

## What changed in the build system (HEAD → origin/main)

| Change | Detail |
| --- | --- |
| CMakeLists restructure | 2813-line diff; backend logic moved into `cmake/backends/{cpu,cuda,hip,metal,quant}.cmake`, plus `cmake/strata_geometry.cmake`, `cmake/strata_ggml.cmake`, `cmake/strata_gpu_engine.cmake` (1326 lines) |
| ggml default flipped | origin/main vendors `third_party/strata-ggml` in-tree (129 files, pulled by git — not a submodule; `.gitmodules` does not exist at origin/main). No `STRATA_GGML_DIR` needed anymore; the override "kept for two releases" per `packaging/build_cuda_linux.sh` |
| third_party churn | removed: `ggml-tq`, `antirez-ds4`, `project-maya`; added: `strata-ggml`, `strata-gguf-py`, `strata-llama`, `gufo` — all in-tree, no `git submodule update` |
| `STRATA_MMQ` default OFF (CMakeLists.txt:112) | llama.cpp MMQ prompt kernels now opt-in; no `libstrata_kq.so` in a default build. The existing local `engine/libstrata_kq.so` came from the old default |
| DSV41 build route deleted | `STRATA_BUILD_DSV41` option gone; `packaging/build_dsv41_cuda.sh` (and HIP/Windows/Mac variants) deleted. Local `engine-dsv41/`, `engine-dsv41.work/` are legacy artifacts |
| NVFP4 | **No flag exists** (`STRATA_NVFP4` is not a real option). W4A8 NVFP4 kernel ships in the default CUDA build; the W4A4 FP4-MMA unit (`strata_mmq_w4a4`, `src/prefill/mmq_nvfp4_w4a4.cu`) is always added and auto-rewrites `120 → 120a` (and `120-real → 120a-real`) for its one unit. Nothing to enable. See `docs/NVFP4.md` for model conversion, which is runtime-side |
| CUDA floor | sm_120 with CUDA < 13.0 warns: CUDA 12.8 engines crashed on long prompts (#220, #224); 13.0+ required. Installed 13.4 → clean. `STRATA_EXPERIMENTAL_SM60` still exists (release only uses it for the cuda12 variant) |
| CMakePresets.json | presets `cpu-check`, `cuda-compile` (release flags, `STRATA_PORTABLE=ON` — for cross-machine packages), `hip-compile`. For this box use the plain docs recipe, not `cuda-compile` (PORTABLE forces AVX2 baseline; a local build wants native CPU codegen) |
| Version mapping | CMakeLists `project(strata VERSION 0.1.61)` = engine internal version; product docs say 0.2.15. Existing local `engine/BUILD.json`: version 0.1.57, archs [120], src `e098a67d` — built from a much older tree with nvcc cuda-13.4 |

Recipe headline is unchanged between HEAD and origin/main (`docs/SOURCE_INSTALL.md` → "Build from git on Linux").

## Exact command sequence (after pull)

```bash
cd /home/svg/LLM/strata

# 1. Update source (ff-only verified possible; working tree is untracked-only)
git pull --ff-only

# 2. Toolchain on PATH: cmake 4.4.3 + ninja 1.13.2 live in the (untracked, surviving) .venv; nvcc in /usr/local/cuda/bin
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"

# 3. Configure in a FRESH build dir. Do NOT reuse build/: its cache carries removed
#    options (STRATA_BUILD_DSV41), a stale STRATA_GGML_DIR=third_party/llama.cpp
#    override, and pre-restructure target wiring.
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120

# 4. Build the engine (16 cores on this box; plain 120 emits sm_120 SASS + compute_120 PTX;
#    the W4A4 NVFP4 unit is auto-upgraded to 120a by the build system)
cmake --build build-cuda --target strata -j "$(nproc)"

# 5. Engine binary lands at: /home/svg/LLM/strata/build-cuda/strata
./build-cuda/strata --version   # expect engine version 0.1.61
```

Optional (only if serving picture inputs): the vision encoder is a separate project —
`cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_VISION_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build-vision --target strata-vision -j "$(nproc)"` (binary `build-vision/bin/strata-vision`; ggml from vendored `third_party/strata-llama`). The current local engine runs `"vision": "none"`.

Optional A/B (old MMQ prompt path, adds `libstrata_kq.so`): add `-DSTRATA_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON` to configure and build `--target strata strata_kq`.

## Release cross-check (what upstream itself runs)

`packaging/build_cuda_linux.sh` (VARIANT=cuda13): same configure core (`Release`, `STRATA_ENABLE_CUDA=ON`, `STRATA_BUILD_TESTS=OFF`, `STRATA_NATIVE_EXPERTS=ON`) plus `STRATA_PORTABLE=ON` and archs `75-real;86;89;120-real;120-virtual` for the multi-card feed package, then packages via `packaging/package_engine.py` with `collect_libs.py` pulling `libcublasLt` from `/usr/local/cuda/lib64`. CI (`.github/workflows/backends.yml`, job `cuda`) mirrors it with ccache on 4-core runners, `timeout-minutes: 120`.

## Expected cost

| Metric | Value | Basis |
| --- | --- | --- |
| Build time (cold, single arch) | roughly 10–30 min on 16 cores | CI ceiling is 120 min for a 3-arch build on a 4-core runner; previous single-arch local build exists (410 MB tree). No recorded wall time on this box |
| Incremental rebuild | minutes | ninja + warm build dir |
| Disk | ~400–600 MB build dir + 122 MB binary | existing `build/` = 410 MB; `engine/strata` = 122 MB; 794 GB free |

## Python side (serve/)

No action. `requirements.txt` diff HEAD→origin/main is comment-only (a doc pointer rename); `requirements-build.txt`, `requirements-monitor.txt`, `requirements-nvfp4-local.txt` unchanged; no `serve/pyproject.toml` or `serve/requirements.txt` diff. The `.venv` already pins the exact required cmake/ninja.

## Risks

| Risk | Verdict |
| --- | --- |
| CUDA floor vs installed toolkit | Cleared. Floor for sm_120 is 13.0 (12.8 crash history); installed 13.4 |
| gcc 16 unsupported by nvcc | Cleared by direct probe: nvcc 13.4 compiled and ran a C++20 `sm_120a` binary with system gcc 16.2.1 |
| Stale `build/` cache poisoning configure | Real. Removed options + stale `STRATA_GGML_DIR`. Mitigated: fresh `build-cuda` dir in the recipe |
| Missing cmake/ninja on PATH | Real but solved: they live in `.venv/bin` (PATH export in step 2). System cmake binary is absent (only cmake-filesystem installed) |
| Submodule drift after pull | None. No `.gitmodules` at origin/main; all third_party trees are in-tree |
| NVFP4 flag guess (`STRATA_NVFP4=ON`) | Wrong guess — no such option; kernels are in the default build. No action needed |
| Old local engine lineage | `engine/strata` (v0.1.57, src `e098a67d`) predates even local HEAD; the new build fully supersedes it. Local dsv41 scripts/engines have no origin/main counterpart (build route deleted upstream) |
