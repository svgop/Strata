---
type: research
date: 2026-11-14
topic: strict directory-architecture consolidation verdict — the mm3 lane landed (third_party/project-maya, src/mm3, build-mm3)
tree: /home/svg/LLM/strata-custom-2 @ main (acf270213)
method: full inventory of root, src/, include/, tests/, tools/, third_party/; diff-census of every cross-family duplicate pair; line-read of every debug instrument in src/mm3; CMakeCache comparison of build-fresh vs build-mm3; cross-checked against .docs/research/digest/2026-11-14-mm3-import-map.md (the settled import decision)
---

# Digest 2026-11-14 — Architecture consolidation verdict: the mm3 lane landed

The engine pattern is settled and correct: per-family `src/<family>/` + own CMake option + own executable + `tests/<family>/` + `tools/<family>/`, with family-private vendored kernels in `src/<family>/vendor/` cut into their own namespace and linked into nobody but that family. The tree does NOT need restructuring. It needs exactly four removals/tidies (build-mm3, the debug instruments, stale doc lines, untracked result records) and zero merges. Every verdict below is anchored to a path, a count, and a diff I ran on 2026-11-14.

## 0. Verdict summary (one line each)

1. **build-mm3 → DRIFT, delete today.** One build dir (`build-fresh`, Ninja/Release) carries both: `cmake -B build-fresh -DSTRATA_BUILD_MM3=ON`; the options are independent by design (CMakeLists.txt:2312, :2355).
2. **src/mm3/vendor/ vs src/dsv41/vendor/ → CANONICAL, keep separate forever.** The forks are genuinely diverged (exl3_cpu 1,135 diff lines; ds4_iq struct-incompatible: maya keeps `act/tu/pexp`, ours replaced them) and the `strata::mm3k` vs `strata::dsv41k` namespace cut is the never-link law enforced by the linker. Consolidating them is impossible without breaking bit-verified kernels; the correct spend is legibility (README in each vendor/, one stale NOTICE line fixed).
3. **Debug instruments → strip ~100 of ~110 lines, keep 3 mechanisms, land one `src/mm3/mm3_diag.hpp`.** Two instruments are currently UN-gated (mm3_model.cu:2542 fires per token per layer), and the CMake line 2366 bakes `MM3_VQ_KERNEL_DEBUG` device printfs into every production build — the chase scaffolding is now costing the hot path.
4. **Repetition census → zero cross-family merges.** Every near-identical pair (ds4_mx_kern.inc: 2 diff lines; ds4_gguf_cpu.cpp: 6) differs only in provenance header + namespace — fork-by-design, drift-free today. The only real duplication is the debug-instrument spray (§4).
5. **campaign/results/*.out (8 files) → tracked by policy, currently untracked → `git add`.** The .gitignore itself declares these "the record" (2026-11-09 policy).
6. **No directory in src/, include/, tests/, tools/, third_party/ needs to move.** One optional symmetry tidy deferred (include/strata/dsv41 → src/dsv41/vendor/) because it touches a bit-verified lane for zero behavior gain.

---

## 1. Inventory

### 1.1 Root

| Path | Purpose | Verdict |
|---|---|---|
| `AGENTS.md` (42 KB) | repo doctrine: campaign + architecture laws | CANONICAL |
| `CMakeLists.txt` (175,724 bytes) | single build spine; family blocks at :2309 (dsv41), :2350 (mm3) | CANONICAL (note: a 175 KB single file is the last structural debt worth watching, not acting on) |
| `build-fresh/` | standing build: Ninja, Release, `STRATA_BUILD_DSV41=ON`, `STRATA_BUILD_TESTS=ON` (CMakeCache.txt:963,977) | CANONICAL as THE one build dir |
| `build-mm3/` | scratch Makefile build, `STRATA_BUILD_MM3=ON`, `STRATA_BUILD_DSV41=OFF`, 1.2 GB | **DRIFT** — see §2 |
| `engine/` | runtime outputs (`strata`, `strata-vision`, `libstrata_kq.so`), gitignored (:50) | CANONICAL (ignored runtime home) |
| `src/`, `include/`, `tests/`, `tools/`, `third_party/` | the code spine | see §1.2–1.6 |
| `config/` (9 × `custom-*.json`) + root `launch-custom-*.sh` (6) | operator serve configs + launchers, one per model lane | CANONICAL (operator harness; `launch-custom-dsv41-iq2xs-8081.sh` + its config are untracked-new, pending add) |
| `campaign/` (2,319 tracked files) | measurement campaign: arms/, batteries/, results/ (1,867 tracked .out) | CANONICAL — except 8 new `.out` under `campaign/results/` are untracked while `.gitignore:104-105` declares result `.out` files "the record" → **CONSOLIDATE into git** |
| `data/` | packs (`data/packs/minimax-m3.1-flash-vq/` = the VQ release: mm3.txt + tensors.txt manifest, per `tools/mm3/vq_pack.py`), expert profiles, draft vocabs | CANONICAL |
| `docs/` (130 files) | user/ops docs | CANONICAL |
| `.docs/research/digest/` (115 files) | research digests — this file's home | CANONICAL |
| `bench/`, `bin/`, `cmake/`, `docker/`, `integrations/`, `metal/`, `packaging/`, `ref/`, `serve/` (98 py, the HTTP serve layer), `sycl/`, `strata_cli/`, `strata_control/`, `apps/` | platform/serve/packaging substrate | CANONICAL |
| `logs/`, `tools/__pycache__/`, `tools/{mm3,dsv41}/__pycache__/`, `.venv/`, `.refs/` (empty) | gitignored runtime scratch | CANONICAL (ignored); `.refs/` empty is a doctrine boundary, not a gap to fill |
| `SVG-CUSTOM.md` (892 KB), `SVG-CUSTOM-TODO.md` (166 KB), `SVG-CUSTOM-FEATURES.md` (16 KB) | operator log/ledger | CANONICAL (operator record; out of code scope) |
| `setup.py` (533 KB) | legacy packaging monolith | note only — no verdict (packaging lane, untested to touch) |

### 1.2 src/ (12 dirs, 390 tracked files)

| Dir | Files / lines | Purpose | Verdict |
|---|---|---|---|
| `src/core/` | 45 | the substrate every family links (host_mem.cpp is explicitly shared into dsv41 + mm3 executables, CMakeLists.txt:2324, :2383) | CANONICAL |
| `src/kernels/` | 169 | main engine kernels + CPU lattice (`cpu/` = expert/iq/avx/nvfp4 kernels, `strata_kernels_cpu`) | CANONICAL |
| `src/qwen35/` | 4 files, 5,206 lines | the main engine family (`q36_engine.cu` 213 KB) | CANONICAL |
| `src/dsv41/` | 40 files, 25,406 lines + `vendor/` 5 files, 4,631 lines | DeepSeek-V4.1 family; vendor = exl3.cu, exl3_legacy.cu, exl3_cpu.cpp, sampler.cu, moe_win.cu under `strata_dsv41_kernels` | CANONICAL |
| `src/mm3/` | 37 files, 21,095 lines + `vendor/` 42 files, 12,405 lines | MiniMax-M3.1 family (imported per the 2026-11-14 import map); vendor = maya's ds4_iq fork + CPU lattice + spec pair under `strata_mm3k` | CANONICAL — minus the §4 instruments |
| `src/spec/` | 7 | draft policy/controller (main engine's, kMaxT=8) | CANONICAL |
| `src/program/` | 6 | generate.cpp (15k+ lines, the decode program) | CANONICAL |
| `src/prefill/`, `src/artifact/`, `src/platform/`, `src/ngram/`, `src/plan/` | 30 / 3 / 3 / 2 / 1 | prefill substrate, GGUF reader, OS platform, PLE reader, planner main | CANONICAL |

### 1.3 include/strata/ (174 tracked files)

`artifact/ core/ dsv41/ hip_compat/ kernels/ ngram/ plan/ platform/ prefill/ program/ qwen35/ spec/` — mirrors src/ 1:1 with one exception: **no `include/strata/mm3/`**. mm3's headers are private under `src/mm3/` (targets add `-I src/mm3`, CMakeLists.txt:2392). That is the *stricter* form (family-private headers don't leak into the public include tree) and matches how the vendor headers are handled — not drift. The exception inside the exception: `include/strata/dsv41/` holds 6 public headers (exl3.hpp, exl3_cpu.hpp, sampler.hpp, moe_win.hpp, hip_or_cuda.h, hip_extra.h) consumed only by `src/dsv41/` (grep: zero cross-family consumers). Verdict: **CANONICAL, with one deferred symmetry tidy** (§6 step 5).

### 1.4 tests/ (166 files)

| Dir | Files | Verdict |
|---|---|---|
| `tests/dsv41/` | 11 | CANONICAL (the family pattern: unit + bench targets in CMake) |
| `tests/mm3/` | 4 files, 3,616 lines (`mm3_exl3_test.cu`, `mm3_gguf_test.cu`, `mm3_vq_test.cu`, `mm3_nvfp4_test.cu`) | CANONICAL — exactly the pattern: one parity suite per kernel class, each wired with `MAYA_DS4_IQ_ONLY=1` and CTest timeouts (CMakeLists.txt:2407–2484) |
| `tests/core|cuda|data|desktop|hip|platform|prefill|program|sycl|terminal|time_skip|tuning` | 20/7/6/5/19/3/2/3/1/68/4/13 | CANONICAL |
| (no `tests/qwen35/`) | — | not drift: the main engine's parity gates live in `tools/qwen35/` (q36_*_gate.py, parity.py) — a historical convention difference to document, not to migrate |

### 1.5 tools/ (233 tracked files)

Flat tool layer + per-family subdirs. The pack-tool convention: `tools/<family>/<quant>_pack.py` — `tools/dsv41/vision_pack.py` (the precedent), `tools/iq_pack.py`, `tools/mtp_pack.py` (flat, main-engine era), `tools/mm3/vq_pack.py` (the VQ pack: reads `data/packs/minimax-m3.1-flash-vq/mm3.txt`, releases in place). `vq_pack.py` matches the convention exactly. Verdict: **CANONICAL**. `tools/qwen35/` (25 files of q36 gates/dumps) CANONICAL. `tools/hip|image|vision|windows/` CANONICAL.

### 1.6 third_party/ (2,845 tracked files)

| Dir | Purpose | Verdict |
|---|---|---|
| `third_party/ggml/` | vendored upstream quant core | CANONICAL |
| `third_party/llama.cpp/` | vendored upstream reference | CANONICAL |
| `third_party/ggml-tq/` | the tq overlay (cmake/strata_ggml_tq_overlay.cmake) | CANONICAL |
| `third_party/basalt-fb0944c7/` | vendored basalt (include/ + src/, pinned by commit suffix in the dir name) | CANONICAL |
| `third_party/project-maya/` | **LICENSE + NOTICE.md only** — the license anchor for everything vendored from maya into `src/mm3/`, `src/mm3/vendor/`, `tools/mm3/` | CANONICAL — this is the correct shape: code lives in the family dir (the isolation law), the third_party entry carries only the legal record. One stale line to fix (§3) |

---

## 2. build-mm3 verdict: DRIFT, delete

Proof of drift:
- `build-mm3/CMakeCache.txt`: `STRATA_BUILD_MM3=ON`, `STRATA_BUILD_DSV41=OFF`, `STRATA_BUILD_TESTS=ON`, generator **Unix Makefiles**.
- `build-fresh/CMakeCache.txt`: `STRATA_BUILD_DSV41=ON`, `STRATA_BUILD_TESTS=ON`, generator **Ninja**, Release.
- The options are deliberately independent (CMakeLists.txt:2356 guards only `STRATA_BUILD_MM3`; :2313 guards only `STRATA_BUILD_DSV41`; neither touches the other's targets). A second build dir buys nothing — it duplicates 1.2 GB of artifacts (build-mm3 = 1.2 GB; build-fresh = 1.3 GB), splits the incremental state across two generators, and has already produced the exact failure mode build dirs exist to prevent: a stale parallel truth of what "the build" is.
- Both are gitignored (`build*/` is .gitignore line 12), so deletion is zero-git-risk.

Exact consolidation:
```
rm -rf /home/svg/LLM/strata-custom-2/build-mm3          # 1.2 GB, untracked
cmake -B /home/svg/LLM/strata-custom-2/build-fresh -DSTRATA_BUILD_MM3=ON
cmake --build /home/svg/LLM/strata-custom-2/build-fresh  # strata + strata-dsv41 + strata-mm3 + 4 mm3 tests
```
While in there: delete the hand-renamed binary snapshots in the standing dir (`build-fresh/strata-dsv41.pre-t61`, `.pre-t62`, `.pre-x1`) — campaign binary-ladder leftovers; the campaign's record home for binaries is `campaign/`, not the build tree.

Three-proof law: **geometry** — one build dir at root, zero `build-*` siblings; **behavior** — `strata-dsv41` and `strata-mm3` and the four `mm3_*_test` targets all present in `build-fresh/`; **cost** — 1.2 GB disk reclaimed, one CMakeCache, one compile_commands.json for tooling.

---

## 3. The vendor question: isolation law done right — keep, and make it legible

**Verdict: `src/mm3/vendor/` (42 files, 12,405 lines) is CANONICAL. No consolidation is possible without breaking the never-link law.** The evidence:

1. **The forks are genuinely diverged, not copies.** Diff census (2026-11-14, mm3/vendor vs the dsv41/mainline twins):
   - `exl3_cpu.cpp`: **1,135 diff lines** (1,502 vs 1,281 lines; `strata::mm3k` vs `strata::dsv41k::cpu`; maya's keeps the `strata::kernels::cpu`-lineage helpers inline).
   - `ds4_iq.cu` / `.cuh`: **struct-incompatible** — maya's `iq::Args` carries `act/tu/pexp` (the MiniMax activation semantics); ours replaced those fields with `xf16/light_*/hipk_*`. Linking both under one namespace collides; extending ours with `act/tu` would put MiniMax semantics inside dsv41's bit-verified kernels. This is the exact hazard the import map named risk #1 and the namespace cut answers.
   - `expert.cpp` family vs `src/kernels/cpu/`: 103–1,115 diff lines each, all cut under `strata::mm3k` (verified: `namespace strata::mm3k` in all 24 namespace-bearing vendor files).
2. **The near-identical pairs prove the discipline, not an opportunity.** `ds4_mx_kern.inc`: 251 vs 251 lines, **2 diff lines** (provenance header only: v1.1.63 vs v1.1.17). `ds4_gguf_cpu.cpp`: 418 vs 418, **6 diff lines** (header + one include path + one namespace qualifier). Today one could share these files — and today it would even work. It must not be done: a shared `.inc`/source re-couples the forks' evolution, so the next divergence in either fork silently edits the other's compiled bits (textual inclusion, no linker protection). The 250 duplicated lines cost nothing; the coupling costs bit-exactness. The whole point of the two-dir form is that each fork freezes at its own maya version — the version lines in the copyright headers are the provenance anchors and they must stay independent.
3. **The link graph already enforces the law.** `strata_mm3_kernels` and `strata_dsv41_kernels` are separate static libs; neither links the other; the only substrate both touch is `src/core/host_mem.cpp`. `MAYA_DS4_IQ_ONLY=1` (CMakeLists.txt:2365, and on all four mm3 test targets) keeps maya's ds4_iq compile-time-pinned.

What would make it legible (the actual gap — today the law lives in a CMake comment and the import map, not in the tree):

- **`src/mm3/vendor/README.md`** (~25 lines): source (project-maya v1.1.63, commit 6651b5b), the namespace cut (`strata::mm3k`), the never-link rule ("never link together with src/dsv41/ or src/kernels/ — ds4_iq is struct-incompatible; exl3_cpu is 1,135 lines diverged"), why the dir exists (fields dsv41's copy dropped: act/tu/pexp), and the compile pin (`MAYA_DS4_IQ_ONLY`).
- **`src/dsv41/vendor/README.md`** (~15 lines): same shape — maya v1.1.17 lineage, `strata::dsv41k` cut, "the mm3 twin is frozen at v1.1.63 and must never merge".
- **Fix one stale line in `third_party/project-maya/NOTICE.md`**: it says the vendored `exl3_cpu` "keeps `strata::kernels::cpu`" — the shipped code is `strata::mm3k` (src/mm3/vendor/exl3_cpu.hpp:17, exl3_cpu.cpp:54; the import map's risk-mitigation rename was applied, the NOTICE wasn't updated).

Cost: ~45 new lines, 1 line edited. Risk: zero. Proof: **geometry** — the law is readable at both vendor dirs without opening CMake; **behavior** — none (docs only); **cost** — the next import (the pattern will recur: maya families keep landing) copies the README template instead of re-deriving the law.

---

## 4. The debug-instrument audit

All instruments live in exactly two files (grep across src/mm3 + tests/mm3): `mm3_model.cu` (3,681 lines) and `mm3_vq.cu` (547 lines). ~110 instrument lines total. Note the gate semantics first: `mm3::env()` (mm3_env.hpp:12) tries `STRATA_<name>` then bare `<name>`, so `MM3_VQ_DEBUG` = `STRATA_MM3_VQ_DEBUG` or `MM3_VQ_DEBUG`. The choke point already exists — the clean form should use it, not add a second one.

### 4.1 The census (every block, classified)

| # | Site | Gate | What it does | Verdict |
|---|---|---|---|---|
| 1 | mm3_vq.cu:336–345, `moe_gu_kernel` | `#ifdef MM3_VQ_KERNEL_DEBUG` — **compiled in by default** (CMakeLists.txt:2366 puts `-DMM3_VQ_KERNEL_DEBUG` on `strata_mm3_kernels`) | per-block device printf of blob pointers whenever a residual layout is in play → thousands of device printfs per layer per token in every production build | **STRIP** as-is; re-land as the opt-in kernel trace in the diag header. The CMake line keeps `-lineinfo` only |
| 2 | mm3_vq.cu:426–439, `moe_gate_up` | **always-on** | D2H copy of the 8 plan entries + 256-alignment range-check; names the fault class ("runaway plan pointer") | **KEEP the mechanism, gate it.** The invariant (every plan entry is a 256-aligned blob base or 0) is permanent structural truth and has caught-real-bug value; but today it costs a cudaMemcpy device sync per layer per decode step. Under `STRATA_MM3_TRACE` it is free when off |
| 3 | mm3_vq.cu:415–417, :473–475, :491–493 (`moe_gate_up`, `moe_down`, `win_experts`) | `MM3_VQ_NO_RES` | zero the `vq_res[3]` offsets = run with residuals as absent — the ablation that exonerated residuals (commit acf270213: "residuals FULLY exonerated") | **KEEP** as a permanent documented A/B switch, folded into one helper (3 copies today) |
| 4 | mm3_vq.cu:425 + :440–456 + :457–465 | `MM3_VQ_DEBUG` (static counter `dbg_n=3`) | staged-out blob header dump + per-entry plan dump, 3 shots | **STRIP** — chase scaffolding, superseded by `mm3_vq_test` parity coverage |
| 5 | mm3_vq.cu:500–511, `win_experts` | `MM3_VQ_DEBUG` (once) | plan + layout dump | **STRIP** |
| 6 | mm3_model.cu:713–721 (VQ layer layout, load time) | `MM3_VQ_DEBUG` | per-layer offsets/src-bytes dump | **STRIP** |
| 7 | mm3_model.cu:1722–1728 (pool build) | `MM3_POOL_FORCE` | pin the whole expert set, bypass the heat-profile trim (with the mapped-page release + retry ladder as safety) | **KEEP** — a real override knob for boxes that hold the whole expert set; document it in the diag header + README, rename at will (`STRATA_MM3_POOL_FORCE` already works via mm3::env) |
| 8 | mm3_model.cu:1924–1929 (`stage`), :1939–1942 (`locate` res), :1965–1968 (`locate`) | `MM3_VQ_DEBUG` (static counters 500/400/4000) | staging/copy/locate traces | **STRIP** |
| 9 | mm3_model.cu:2071, :2094, :2496, :2520 | `MM3_VQ_DEBUG` | `vq swig …` pointer dumps with **baked stale line literals** (`@2062` sits at line 2071; `@2481` at 2496; `@2504` at 2520) — proof they were never meant to live | **STRIP** |
| 10 | mm3_model.cu:2541–2543 | **UNGATED** | `vq moe@…` per-token per-layer fprintf in the decode path | **STRIP** — pure leftover; hot-path cost today |
| 11 | mm3_model.cu:2719 | **UNGATED** | `vq set@…` per-pass fprintf — and its format string is `"…res %zu\\n"`, a literal backslash-n (no newline) | **STRIP** (broken and un-gated) |
| — | mm3_model.cu:3404 | `STRATA_MM3_TP_WINDOWS` | functional TP-mode switch, not an instrument | KEEP untouched (listed so the audit is complete) |

### 4.2 The clean form: one `src/mm3/mm3_diag.hpp`

~60 lines, beside `mm3_env.hpp` (which stays the env choke point):

```cpp
// src/mm3/mm3_diag.hpp — the one diagnostic gate for the mm3 family.
// STRATA_MM3_TRACE=1   : stage/locate/plan traces (off = zero cost, zero lines)
// STRATA_MM3_NO_RES=1  : the residual ablation (the exonerate switch)
// STRATA_MM3_POOL_FORCE=1 : pin the whole expert set past the heat-profile trim
#define MM3_TRACE_ON() (::mm3::env("STRATA_MM3_TRACE") != nullptr)
#define MM3_TRACE(...) do { if (MM3_TRACE_ON()) std::fprintf(stderr, __VA_ARGS__); } while (0)
inline bool mm3_no_res()  { static const bool v = ::mm3::env("STRATA_MM3_NO_RES") != nullptr; return v; }
inline bool mm3_pool_force() { static const bool v = ::mm3::env("STRATA_MM3_POOL_FORCE") != nullptr; return v; }
// MM3_CHECK_PLAN(a): the plan-entry 256-alignment invariant (audit #2), under MM3_TRACE_ON()
```
Backward-compat note: keep `MM3_VQ_DEBUG`/`MM3_VQ_NO_RES` working as aliases inside the header (one `||`) or accept the rename — the operator decides; the doc line in the header lists whichever survives.

Diff size: −~100 instrument lines across `mm3_model.cu` + `mm3_vq.cu`, +~60 in the new header, −1 token in CMakeLists.txt:2366 (`-DMM3_VQ_KERNEL_DEBUG` gone, `-lineinfo` stays). Risk: low — every removed block is a fprintf; the only behavioral carrier is the NO_RES gate (rewired 1:1) and the plan check (now env-gated). Proof: **geometry** — `grep -rn "MM3_VQ_DEBUG\|MM3_VQ_KERNEL_DEBUG" src/` returns only mm3_diag.hpp; **behavior** — `mm3_vq_test`, `mm3_nvfp4_test`, `mm3_gguf_test`, `mm3_exl3_test` green + one VQ boot parity warm run (`campaign/results/` gets the .out); **cost** — default-env decode shows zero vq fprintf lines, and the per-layer-per-token plan cudaMemcpy leaves the hot path.

---

## 5. Repetition census

Every candidate pair diffed on 2026-11-14. **Top repeats, verdict each:**

| Pair | Diff | Verdict |
|---|---|---|
| `src/mm3/vendor/ds4_mx_kern.inc` vs `src/dsv41/ds4_mx_kern.inc` (251/251 lines) | **2 lines** (provenance header) | fork-by-design — isolation law; do NOT share (see §3.2) |
| `src/mm3/vendor/ds4_gguf_cpu.cpp` vs `src/dsv41/ds4_gguf_cpu.cpp` (418/418) | **6 lines** (header + include + one namespace qualifier) | fork-by-design — same law |
| `src/mm3/vendor/suffix_drafter.*` vs `src/spec/suffix_drafter.*` (90/98 cpp) | 14 + 8 lines (ours has `propose_after`, maya's doesn't) | fork-by-design — capability divergence, NOTICE-documented |
| `src/mm3/vendor/draft_policy.*` vs `include/strata/spec/draft_policy.hpp` + `src/spec/draft_policy.cpp` | 80 hpp lines; kMaxT 8 vs 16 + `trust()`/bands | fork-by-design — the import map's named API gap; vendoring was the settled fix |
| `src/mm3/vendor/exl3_cpu.cpp` vs `src/dsv41/vendor/exl3_cpu.cpp` (1,502/1,281) | **1,135 lines** | fork-by-design — fully diverged optimization histories |
| `src/mm3/vendor/ds4_iq.cu/.cuh` vs `src/dsv41/ds4_iq.cu/.cuh` (2,445/1,630) | 815 + 131 lines; struct-incompatible (`act/tu/pexp`) | fork-by-design — THE reason for `strata::mm3k` |
| `src/mm3/vendor/{expert,expert_layout,iq_avx2,iq_avx512,iq_avx512_one,native_expert,q2_avx2,glm_lane_row4}.cpp` vs `src/kernels/cpu/` twins | 103–1,115 lines each | fork-by-design — maya's CPU lattice under `strata::mm3k` |
| `tools/mm3/vq_pack.py` vs `tools/dsv41/vision_pack.py` / `tools/iq_pack.py` | convention match (`tools/<family>/<quant>_pack.py`, manifest in `data/packs/`) | CANONICAL — no slop |
| the debug instruments (~110 lines, 2 files) | 3× duplicated NO_RES gate, 4× duplicated swig-trace line, 2× un-gated leftovers | **the only genuine slop in the census** → §4 consolidation into `mm3_diag.hpp` |
| `include/strata/dsv41/` (6 public headers) vs mm3's all-private header layout | structural asymmetry | not slop; dsv41's predates the private-vendor form. Deferred optional tidy (§6 step 5) |

Bottom line: the three families share exactly one file (`src/core/host_mem.cpp`) and one namespace-free substrate (`src/prefill`, `include/strata/core`). That is the design working. The consolidation budget belongs to the instruments, not the forks.

---

## 6. The ordered consolidation plan

Execution follows this verdict; nothing below is done. Each step: exact paths, expected diff, risk, proof.

### Step 1 — Kill build-mm3, one build dir carries both families
- Do: `rm -rf build-mm3` (1.2 GB, untracked, Makefile-based). `cmake -B build-fresh -DSTRATA_BUILD_MM3=ON` then full build.
- Also: `rm build-fresh/strata-dsv41.pre-t61 build-fresh/strata-dsv41.pre-t62 build-fresh/strata-dsv41.pre-x1` (stale hand-renamed binaries).
- Diff: 0 tracked lines. Risk: none (gitignored; options are independent by construction, CMakeLists.txt:2313/:2356).
- Proof: geometry — root has exactly `build-fresh/`; behavior — `strata`, `strata-dsv41`, `strata-mm3`, `mm3_{exl3,gguf,vq,nvfp4}_test` all present in one dir and ctest green; cost — 1.2 GB + one parallel cache gone.

### Step 2 — Land `src/mm3/mm3_diag.hpp`, strip the chase scaffolding
- Do: per §4.2. Touch exactly: `src/mm3/mm3_diag.hpp` (new, ~60 lines), `src/mm3/mm3_vq.cu` (−~46), `src/mm3/mm3_model.cu` (−~38, includes the two UNGATED sites :2541–2543, :2719), `CMakeLists.txt:2366` (drop `-DMM3_VQ_KERNEL_DEBUG`).
- Diff: ~+60/−90. Risk: low (the NO_RES rewire and the gated plan check are the only behavior carriers).
- Proof: geometry — single grep hit for the gates (the header); behavior — 4 mm3 tests green + one VQ boot warm parity run recorded in `campaign/results/`; cost — zero vq fprintf at default env, hot-path plan sync gone.

### Step 3 — Make the never-link law legible in the tree
- Do: new `src/mm3/vendor/README.md` (~25 lines: provenance v1.1.63/6651b5b, `strata::mm3k` cut, never-link rule, `MAYA_DS4_IQ_ONLY`, why the fork exists); new `src/dsv41/vendor/README.md` (~15 lines, same shape); edit `third_party/project-maya/NOTICE.md` line claiming exl3_cpu "keeps `strata::kernels::cpu`" → the shipped `strata::mm3k`.
- Diff: +~40/−1. Risk: zero.
- Proof: geometry — the law cites from both vendor dirs; behavior — n/a; cost — the next family import reuses the template.

### Step 4 — Record the pending records
- Do: `git add campaign/results/{ds-iq2-boot,ds-warm-1,iso-legA,iso-legB,iso-legC,postreboot-warm-1,refactor-warm-test,vq-m31-download}.out` (8 files, `.gitignore:104` policy: "result .out files under campaign/results are the record"); `git add config/custom-dsv41-iq2xs-8081.json launch-custom-dsv41-iq2xs-8081.sh` (the iq2xs lane's config+launcher pair, matching the other 8 lanes' committed pattern); commit `data/packs/minimax-m3.1-flash-vq/mm3.txt` modification.
- Diff: +11 files, 1 modified. Risk: zero.
- Proof: geometry — `git status` clean of untracked records; behavior — n/a; cost — the campaign record is complete in git, not on disk.

### Step 5 — DEFERRED (optional, only with a dsv41 parity gate): vendor-header symmetry
- Do (later): move `include/strata/dsv41/{exl3_cpu.hpp,sampler.hpp,moe_win.hpp,hip_or_cuda.h,hip_extra.h}` (keep `exl3.hpp` if external consumers appear) into `src/dsv41/vendor/`, re-point 5 includes in `src/dsv41/`. Grep today: zero consumers outside `src/dsv41/`.
- Why deferred: touches the bit-verified dsv41 lane for zero behavior gain — symmetry only. Do it the next time dsv41 is open for a kernel change anyway, with `ds4_gguf_test` + `iq_expert_test` green before and after.

### Final tree sketch (post step 4)
```
strata-custom-2/
  AGENTS.md  CMakeLists.txt  README.md  SVG-CUSTOM*.md        # doctrine + spine + operator record
  build-fresh/          # THE build dir: strata + strata-dsv41 + strata-mm3 + 4 mm3 tests
  campaign/  config/  data/  docs/  .docs/research/digest/    # record + configs + packs + research
  engine/  logs/  .venv  tools/__pycache__/                   # gitignored runtime
  src/
    core/ kernels/ prefill/ program/ spec/ artifact/ platform/ ngram/ plan/   # substrate (one owner each)
    qwen35/                        # family 1: main engine
    dsv41/  + vendor/ (5)          # family 2: strata-dsv41, dsv41k cut, vendor/README.md
    mm3/    + vendor/ (42)         # family 3: strata-mm3, mm3k cut, vendor/README.md, mm3_diag.hpp
  include/strata/                  # public substrate headers (families keep theirs private in src/)
  tests/{core,cuda,dsv41,mm3,...}  # family pattern: tests/<family>/
  tools/{mm3,dsv41,qwen35,...}     # family pattern: tools/<family>/
  third_party/{ggml,llama.cpp,ggml-tq,basalt-fb0944c7,project-maya/}   # upstreams; maya = LICENSE+NOTICE only
```
Net effect: −1.2 GB on disk, −1 build dir, −~90 instrument lines, +1 diagnostic owner, +2 law documents, −1 stale doc line, +11 recorded files. The directory architecture itself: already the correct form — three families, one substrate, zero cross-family symbol flow. The verdict is that the system design survived the mm3 import intact; the residue was the chase, and the chase now has one owner and one gate.
