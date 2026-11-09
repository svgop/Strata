---
type: digest
date: 2026-11-09
topic: repo-surface-hygiene (fresh-clone view + misfiled-artifact audit)
scope: root inventory, tools/, campaign/ non-results, config/ drift, .gitignore, docs set
method: read-only; sizes via du/stat; tracking via git ls-files / status --ignored; lint_tree.py executed
---

# Repo surface hygiene study — 2026-11-09

Post-prune state verified: commit 9bbdfb75 (2.6 GB estate prune) landed, `python3 tools/lint_tree.py` →
`0 finding(s)` (rerun today). Git tree is clean except `M campaign/results/stock-jury-boot.out`.
Largest tracked blobs today: `third_party/llama.cpp/models/ggml-vocab-gemma-4.gguf` 15.4 MB,
`docs/media/pagoda-preview.webp` 9.4 MB. No tracked file above 15.4 MB.

## 1. Repo root inventory (44 tracked top-level files)

**Canonical (upstream entry points, keep):** `CMakeLists.txt` (160 KB), `Dockerfile`, `.dockerignore`,
`setup.py` (521 KB install wizard — upstream-pattern monolith, its 60+ unit tests live co-located in
`tools/test_setup_*.py`), `setup.sh`/`SETUP.bat`, `update.sh`/`UPDATE.bat`, `bench.sh`/`BENCH.bat`,
`START-HERE.bat`, `flake.nix`/`flake.lock`, `.gitattributes`, `LICENSE`, `CONTRIBUTING.md`,
`SECURITY.md`, `conftest.py` (pytest fixture home), `chat.py` (terminal chat client), `strata.py`
(entry point), `bin/{strata,strata.cmd,strata.ps1}`, 4× `requirements*.txt`, `README.md` + 6
translations, `AGENTS.md` (41 KB project doctrine).

**Canonical by campaign law:** the 5 `launch-custom-*.sh` at root. `campaign/README.md:35` —
"Standing boot pair lives at REPO ROOT (do not move — LAUNCH LAW + F-249 .venv resolution)";
commit 91a1e5da made the taskset/swap-guard wrapper uniform across all five. Not misfiled.

**Canonical by README reference:** `SVG-CUSTOM.md` (871 KB journal), `SVG-CUSTOM-FEATURES.md` (16 KB),
`SVG-CUSTOM-TODO.md` (162 KB). `README.md:15-16` publicly links the first two as the differentiator
index; `campaign/README.md:4` calls the journal "the index of what ran and why";
`tools/kl_yardstick.py` cites the TODO. **NEEDS-DECISION:** 1.05 MB of campaign journal at repo root
is the heaviest text a fresh clone sees; a move to `campaign/` or `docs/` is a reference-rewrite
across README, campaign/README, launchers and campaign/research/*. Default: keep-in-place.

**Canonical data/binary homes (gitignored, correct):** `engine/` 102 MB (`strata` 91 MB,
`strata-vision`, `libstrata_kq.so`, `BUILD.json` — the only binary home), `logs/` 16 MB runtime,
`.venv/`, `build-fresh/` (untracked, covered by `build*/` `.gitignore:12`).

**Misfiled / debris:**
- `build-fresh/` is referenced as `exe` by 5 of 8 configs (§4) — a gitignored build dir is a config's
  binary source; `engine/` is the declared home. SAFE-NOW to repoint after the next install.
- `.refs/` — empty directory since Oct 12, zero content. Debris (scaffold stub that never filled).
- `.docs/` holds only `research/digest/` (40 digests, tracked). The required records `.docs/log.txt`,
  `.docs/index.md`, `.docs/technical_summary.md`, `.docs/workspace.json` are ABSENT and the project
  `AGENTS.md` nowhere owns `.docs` (grep: no mention) — contract boundary; name the gap, don't
  fabricate the records without an owner. NEEDS-DECISION.
- `ref/load.py` — P1.S4 lazy-dequant reference implementation. Canonical (its own docstring says so).

## 2. tools/ inventory (165 .py + lint_exceptions.json 16 KB + 2 .cu + 6 subdirs)

Convention is coherent: a tool and its offline `test_*` unit pair sit side by side; live-engine
regression scripts also live here; `tests/` owns engine-level suites (13 subsystem dirs). Liveness
witnesses: `lint_tree.py` + `lint_exceptions.json` (the tripwire, 0 findings), `strata-doctor.py`
(memory doctor + pool sizer, modified tonight 21:12 — alive), `calibrate.py`/`test_calibrate.py`.

One-liners (living instruments): `strata_bench.py` (one-command shareable bench), `strata_mcp.py`
(145 KB MCP install/start/check server), `strata_tokenizer.py` (reference tokenizer),
`strata_inspect.py` (GGUF header verdict), `strata_pack.py` (pack builder/verifier), `quantscope.py`
(remote quant discovery), `gguf_reader.py`/`gguf_writer.py` (minimal GGUF I/O), `serve_load.py`,
`needle_bench.py`, `ab_engine.py` (A/B two builds), `agent_turns_bench.py`, `research_run.py`,
`harvest_corpus.py`/`build_standard_corpus.py` (corpus), `check_isa_guard.py` (AVX2 reachability,
#795), `bench_eviction.py`/`replay_cache_policy.py` (expert-cache policy replay), `make_profile.py`/
`make_profile_mix.py`/`collect_profile_traces.py` (expert-profile pipeline), `hf_pin_fetch.py`.

Quant/MTP pipeline: `mtp_pack/fetch/rt/train/module` (plan v0.3 P6 + T14B trainer),
`nvfp4_convert/local_pack/verify/verify_gguf`, `iq_pack`/`iq_fixture`, `embd_bf16_pack`,
`ple_fp8_pack`, `q8_dense_gguf`, `pack_index`/`pack_layer`, `quant_formats`/`quant_bound`/
`quant_quality`, `kl_yardstick.py`, `draft_vocab.py`.

Subdirs: `hip/` (Windows HIP packagers + per-gfx rocblas/hipblaslt pin tables — canonical pins),
`vision/` (strata_vision sources), `image/` (setup + qimage), `dsv41/vision_pack.py`,
`windows/enable_large_pages.ps1`, `qwen35/` — 27-file `q36_*` vision-proto battery for a
concluded arch era; disposition candidate (NEEDS-DECISION, per F-267 below).

**One-off scripts masquerading as tools (flagged):**
- `mtp_indexqk_splice.py` AND `mtp_indexqk_splice_r17.py` — identical F-55 docstring; revision pair,
  one is dead weight. SAFE-NOW delete (git history archives).
- `mtp_bridge.py` (F-48), `mtp_phaseb_adapter.py` (F-52), `mtp_rt2_assemble.py` (F-325 stage 2) —
  one-shot retrain assemblers, era-pinned. NEEDS-DECISION (delete or move to campaign/retrain/).
- Issue-pinned live-engine scripts named `test_*` but not offline-runnable: `batch_groups_cache_test`
  (#1604), `cancel_park_test` (#1620), `early_close_test`, `parking_test`, `session_pressure_test`
  (#1572/3), `stop_resend_test` (#879/606), `split_mtp_batch_parity` — naming collision with the
  offline unit convention; they need a running engine. Note only; renaming is churn.

## 3. campaign/ non-results subdirs (du verified)

`results/` 40 MB, 1861 tracked files incl. 549 tracked `.out` — **the law-12 record, STAYS.**
Except one hole: `results/old-logs/` is 9.8 MB on disk, only 2 `.gz` tracked; **41 raw `.log` are
untracked-ignored** (`*.log` line 11 shadows them; the negation `!campaign/results/*.log` line 74
does not descend into `old-logs/`). Evidence invisible to git. NEEDS-DECISION: rotate to `.gz` +
track, or delete as superseded (only 2 of 41 have a `.gz` twin).

- `arms/` 160 KB, 22 tracked — parked arm configs+launchers, f33x–f35x era (Oct 26) + 262k pair.
  Doctrine-owned status "parked future work" (`campaign/README.md:27`). **Keep-in-place.**
- `shelf/` 12 KB, 1 tracked — `shared-hoist-F362.patch`, stashed. **Keep-in-place.**
- `retrain/` 136 KB, 5 tracked — T14B/F-325 trainer sources; era concluded, results recorded.
  **Keep-in-place** (sources only; the `*.npy` ignore rule matches nothing now).
- `l1-falsifier/` 1.4 MB, 4 tracked + untracked `build.log` — L1 commit falsifier, `results.txt`
  verdict recorded, era concluded. Keep sources+results. **SAFE-NOW:** ELF
  `l1-falsifier/l1_commit_falsifier` (1.3 MB) is TRACKED, contradicting `.gitignore:104`
  ("keep sources + results, never the ELF"); line 115 ignores `campaign/l1_commit_falsifier`, a path
  that does not exist. `git rm --cached` the ELF + fix the rule path.
- `shadow-bench/` 3.1 MB on disk, 2 tracked — shadow-store bench, concluded (`results.txt`).
  Keep-in-place; the 3.1 MB untracked ELF is regenerable debris, deletable. SAFE-NOW.
- `research/` 3.0 MB, 126 tracked, `INDEX.md` present (116 entries) — living dossier home.
  **Keep-in-place.**
- `upstream/` 1.2 MB, 26 tracked — 11 `.out`/`.json` boot/restore comparison evidence + 10
  `t50/*.patch` upstream-candidate patches + 2 test scripts. Evidence lives OUTSIDE the evidence
  home (results/). **NEEDS-DECISION:** move the 11 evidence files into `results/upstream-*`
  (paths are cited by research notes); keep the t50 patches (in-flight upstream work).
- `T14B-specs/` 160 KB, 4 tracked — retrain/re-quant execution specs, era record. **Keep-in-place.**
- `headport/` 1.7 MB, 152 tracked — head-port kernel sources (.cu/.cuh, basalt/, gu2/).
  Keep-in-place; its blob guards (`.gitignore:99-101`) now match nothing (blobs deleted in prune).
- `gpu-lock/` 12 KB, 2 tracked — v3 (4.3 KB binary) + README with install command; v3 is current
  (README dated 2026-11-07). **Misfile:** superseded `strata-gpu-lock-v4/-v5/-v6` + `v4.service`
  sit tracked at campaign ROOT (6.5–9.2 KB each), not in `gpu-lock/`. SAFE-NOW: move under
  `gpu-lock/` or delete (git history archives).
- `batteries/` 264 KB, 57 tracked — reusable harnesses per README. **Keep-in-place.**
- `tracker/` 120 KB, 7 tracked — living lane tracker (lanes.csv, trend HTML), README-linked.
  **Keep-in-place.**
- Loose at campaign root, NOT in README's living-instruments list: `t42-prefill-ladder.py`,
  `t43-smoke.py`, `t48-ngram-{analyze,probe}.py`, `t50-131k-probe.py`, `conc-probe.py`,
  `prefill-probe.py`, `template-probe.py`, `tpl-ux-probe.py`, `tpl-user-render-battery.py`,
  `mimic-compare.py`, `hostmem-probe.sh`, `shm-shadow.sh`. F-267 (`campaign/README.md:42`):
  "Run-once harnesses for closed questions get DELETED when the question closes." **NEEDS-DECISION:**
  operator ratifies the closed-question list, then delete; git history is the archive.
- `hostmem-anon-collapse` (13 KB ELF) tracked despite ignore line 114 — no-op on tracked files.
  Same untrack finding as l1-falsifier. SAFE-NOW.
- `campaign/__pycache__/` 164 KB — ignored, fine.

## 4. config/ drift (8 engine configs — what a config-lint would catch)

| finding | evidence | class |
|---|---|---|
| tokenizer dir ≠ pack dir | `custom-ud-iq4xs-8081.json`: pack `packs/unsloth-ud-iq4_xs-v31`, tokenizer `packs/unsloth-ud-iq2_xs/tokenizer` (copy-paste; latent break when the packs diverge) | lint rule |
| duplicate configs | `custom-dsv41-4e-8081.json` ≡ `custom-dsv41-ct32-8081.json` except log filename (verified diff) — ct32 lost its differentiating arg | lint rule |
| orphan configs | `custom-dsv41-{ct32,e5}-8081.json`, `custom-dsv41-t54.json` — zero references from any launcher, `serve/`, `tools/strata_mcp.py`, `bin/` (checked) | NEEDS-DECISION |
| exe outside binary home | 5 dsv41 configs → `build-fresh/strata-dsv41` (gitignored build dir); 3 qwen configs → `engine/strata` (canonical). Pruning build-fresh breaks 5 boots | lint rule |
| vram-reserve gap | `--vram-reserve-mib 2000` in qwen×3; absent from dsv41×5 (pool-doctrine uniformity vs `docs/PERFORMANCE-SETUP.md`) | lint rule |
| dead prewarm sections | `"prewarm": {}` EMPTY in dsv41-8081, iq3_s, iq2_xs, iq4xs; key absent in the other 4 dsv41 configs. Ladders "wait for the config's prewarm section" (`campaign/README.md:14`) — nothing to wait for, anywhere | lint rule |
| two channels for host | qwen configs set `"host": "0.0.0.0"`; dsv41 configs omit it, launchers pass `--host 0.0.0.0` on CLI | lint rule |
| field skew | `arch`/`engine_ready_s` only in dsv41; `parallel`/`draft_vocab`/`vision` only in qwen | lint rule (per-arch schema) |
| naming | 7 of 8 carry `-8081`; `custom-dsv41-t54.json` doesn't (and it is the only `--verbose` debug variant) | lint rule |
| consistent (no action) | `api_key` identical estate constant ×8 (ratified by 36f6098b: clients read STRATA_API_KEY, constant is fallback); `port 8081`, `gpu 0`, `gpus_asked true`, `lib_dirs cuda-13.4` ×8; note the key hex is committed in git — accepted local-estate design, boundary named | SAFE-NOW |

## 5. .gitignore audit (116 lines)

**Rules matching nothing now (post-prune fossils):** `ds4-slots/` (:106), `campaign/headport/
head_q6k.bin|vhead*|head_requant_probe` (:99-101), `campaign/retrain/*.npy` (:102),
`campaign/l1_commit_falsifier` (:115 — wrong path; real ELF is in `l1-falsifier/`),
`bench/micro/*.exe|obj|bin|dll|txt` (:6-10 — dir holds 3 sources only), `/release/artifacts/` (:84),
`.terminal-venv*/` (:80-81), `dist-hip/` (:77), `pack/` (:13; sibling `/packs/` :33 also absent —
a typo-fossil pair), `/build/` + `/build-cuda12/` (:30, :41 — fully shadowed by `build*/` :12),
`td.log` (:26 — shadowed by `*.log` :11), `/strata-*.log` (:38 — shadowed by `strata-*.log` :71).
Prophylactic rules for setup.py-generated paths on user machines (`engine-sycl/`, `engine-cuda12/`,
`.strata-mcp/`, `run-*.sh`) stay even though absent here.

**No-op rules:** `campaign/hostmem-anon-collapse` (:114) and the l1 ELF — targets are tracked;
ignore never applies to tracked files. `!campaign/results/*.txt` (:75) matches nothing — no global
`*.txt` exclude exists to negate. `!campaign/results/*.log` (:74) cannot reach `results/old-logs/`
(the 41 raw logs, §3). `!campaign/results/*.out` (:113) is needed only against deep `server-*.out`
(:67) matches — keep.

**Gaps that admit the next 500 MB blob:**
1. `campaign/results/` is tracked wholesale with no type gate — exactly where the pruned 2.6 GB
   lived. A future `results/**/*.bin|gguf|dmp` commits silently. SAFE-NOW: add explicit
   `campaign/results/*.bin`, `*.gguf`, `*.dmp` guards with a negation list if any are records.
2. No global `*.gguf` rule (only `bench/tiny-*|wide-*` :4-5). Precedent it already slipped through:
   vendored `third_party/llama.cpp/models/*.gguf` (15.4 MB max) and the tracked fixture
   `data/experimental-speed-projection/*.gguf` (472 KB). A dropped model anywhere else commits.
3. No `*.safetensors|*.pth|*.pt` rule at all.
4. No media size policy — `docs/media/pagoda-preview.webp` 9.4 MB is already the largest
   non-vendored blob.

## 6. Top-level docs set — coherent or accreted?

- Root paper set is coherent: `README.md` (31 KB, Nov 8) + `CONTRIBUTING/SECURITY/LICENSE/AGENTS.md`.
  **Translation drift:** the 6 README translations are frozen at Oct 20 and predate the
  "svg custom line" fork-differentiator section (README.md:14+) — the public face a fresh clone
  reads in English does not exist in zh/ja/de/fr/es/pt-BR. SAFE-NOW to note; translation pass is an
  operator call.
- `docs/` is a coherent product-doc tree by surface (hardware: AMD_HIP*, INTEL*, NVIDIA_V100,
  RADEON_780M, STRIX_HALO, MAC, OLDER_GPUS; feature docs: HUGEPAGES, EXPERTDEC, KV_PREFETCH,
  SHADOW_STORE, VRAM_ELASTIC, LOGIT_BIAS, PERFORMANCE-SETUP…; subdirs `benchmarks/` (3 community
  results), `monitoring/` (grafana/prometheus), `media/`, `paper/` (PDF 334 KB)).
  Accretion spots: six TERMINAL_* docs (TERMINAL, _LAUNCH, _QUALIFICATION, _QUICKSTART,
  _TELEMETRY_FIELDS, _IMPLEMENTATION_STATUS — the status/implementation pair could fold into the
  quickstart), and `docs/work/` (ACCEPTANCE-EVIDENCE, PHASE-A-RETURN-PACKET, QUALIFICATION-KIT,
  IMPLEMENTATION_STATUS) is campaign-era work-record material living in product docs — candidate
  for `campaign/research/` (NEEDS-DECISION).

## Boundary

Conclusions flip if: (a) the operator ratifies root placement of the SVG-CUSTOM journal (then §1's
move question closes); (b) any of the 41 `old-logs/*.log` or loose campaign probes are declared
living instruments rather than closed-question harnesses; (c) dsv41 configs intentionally target
`build-fresh/` as a frozen-binary home rather than a rebuild dir.
