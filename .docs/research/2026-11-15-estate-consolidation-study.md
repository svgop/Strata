---
type: research
date: 2026-11-15
subject: Estate consolidation study — /home/svg/LLM full architecture survey
method: Two independent read-only study agents (estate-wide + repo-internal), coordinator-verified load-bearing claims
companion: campaign/ROADMAP-estate-consolidation.md (the plan of record)
---

# Estate consolidation study, 2026-11-15

Read-only survey of `/home/svg/LLM` (~775 GB). Every claim below carries evidence
(path, size, sha256, or command output). Coordinator re-verified the deletions-riding
facts directly: ancestry (`git rev-list --count upstream/main..9f7d977d` = 0), the
refusal-vector twins (sha256 `59f9ae8b…` both sides), the dangling ist-das `model_dir`,
and the live engine (`strata-mm3`, PID 48587, 8081).

## 1. Verified estate shape

| Path | Size | Purpose | Verdict |
|---|---|---|---|
| `data/` | 655 GB | Canonical data home, governed by `data/AGENTS.md` | KEEP |
| `data/models/` | 602 GB | Weights by lineage: GSQ-RCO/ 335 GB, UD3.1/ 159 GB, VQ/MiniMax 108 GB | KEEP |
| `data/packs/` | 42 GB | 7 packs (AGENTS.md documents only 3 — stale) | KEEP, doc fix |
| `data/mtp/` | 6.5 GB | Stock drafter + 4.9 GB `tensors/` (now the ONLY local BF16 extraction — sources deleted 2026-11-14) + 889 MB `mtp-q2_0.gguf` (no estate referent) | KEEP tensors; q2_0 = delete candidate |
| `data/mtp-en/` | 196 KB | Serving home; 7 symlinks into `data/mtp` all resolve; every Qwen config loads it | KEEP (load-bearing) |
| `data/conversation-slots/` | 4.4 GB | Qwen T66 persistence, 14 slots | KEEP (runtime) |
| `data/references/` | 1.3 GB | 8 pinned study clones + MANIFEST.md; `Research/` subdir EMPTY | KEEP; rm empty dir |
| `data/vectors/` | 1.1 MB | 2 control vectors | KEEP |
| `data/ds4-slots/` | — | DOES NOT EXIST (doctrine lists it; launchers export it; engine re-creates on next DS boot) | pre-create in pass |
| `Custom-data/` | 113 GB | T14B training campaign home: `harvest/` 106 GB + `mtp/` 6.9 GB retrain workspace. Wired into committed tools | FOLD → `data/training/` |
| `strata/` | 2.4 GB | Upstream clone, HEAD `9f7d977d` ∈ sc2 `upstream/main` ancestry (count 0) — 100% redundant + 1.1 GB stale residue | DELETE after salvage |
| `strata-custom-2/` | 4.8 GB | ACTIVE fork (origin svgop/Strata, upstream Niko1221). 69 unpushed commits; live mm3 lane | KEEP |
| `.docs/` (estate) | 12 MB | recon bank + model-fit + zig feasibility + 11.8 MB regenerable bench binaries; `digest/` EMPTY since Oct 22 | FOLD into sc2, rm |
| `qwen_client.py` + `__pycache__/` | 9.4 KB | chat.qwen.ai scratch client, zero estate references | RELOCATE into sc2 tools/web/ |
| `.zcode/` | 72 KB | harness config (rich-tracking board) | LEAVE |

## 2. Parallel systems (the drift the consolidation kills)

1. **Two serving stacks.** sc2: `serve/server.py` + `build-fresh`/`engine` exes + 10 configs + 7 launchers (LIVE). `strata/`: its own `serve/server.py` + `engine-dsv41/` 0.1.52 bundle + ignored `strata-*.json` configs + `launch-deepseek-v41-8081.sh` (DEAD — last DS log Oct 28; its q2_0 config names bare `models/`-root paths that no longer exist). Everything server-side converges on sc2.
2. **Three strata sources.** sc2's `upstream` remote (fetched 2026-11-12, 474 commits ahead of the clone) + `data/references/Strata@v0.2.1.4` (pinned) + the redundant `strata/` clone. Two survive.
3. **MTP drafter copies.** `experts.bin` byte-identical in `data/mtp/rt`, `Custom-data/mtp/rt2`, `Custom-data/mtp/rt2-deploy` (sha256 `09398406…`, 707,788,800 B each). `dense.bin` three DISTINCT blobs; `rt2-deploy == export/dense_rt2.bin` (sha `88ccbbc7…`). The TRAINED dense (step 21430, best_mean 0.7932 vs stock 0.4049) is NOT what any config loads — serving still boots stock dense.
4. **Harvest generations.** `records/` (round 1) ⊂ `records-round2-aligned/`: 11/11 names+sizes equal, sampled shas equal (`records-dagger-0.f32` 4,099,894,656 B; `records-probe.f32` 68,143,088 B identical in all three homes). ~29 GB near-certain duplicate. Plus `quarantine/` 7.7 GB (rejected streams) and `records-d1/` 12 GB (earlier variant).
5. **Digest/doc trees.** sc2 `.docs/research/digest/` (118 live files, the operative stream) vs estate `.docs/digest/` (EMPTY — `.docs/research/60-docs.md:90-92` records the failed intent) vs estate `.docs/research/`. One research home wins.
6. **Refusal vector.** `strata/refusal-r.gguf` == `data/vectors/qwen3.8-flash-next-refusal-r.gguf` (sha256 verified). The strata copy violates doctrine twice (loose repo-root file; vector outside `vectors/`).
7. **Versioned-binary piles.** `strata/engine/`: current + `0.1.43.1-backup` + `v0.1.45-pre-v02` + `v0.2.1` (228 MB); `strata/engine-dsv41/` (665 MB, feed-reobtainable, BUILD.json `f-027-rel a42cc3d7`). All die with the clone.
8. **Launcher twins (repo-internal).** `launch-custom-ud-iq2xs-8081.sh` vs `-iq4xs-`: byte-identical except 2 lines (quant tag + config name). Standing-arms block (lines 16–39) duplicated across launchers and ≈ stale `strata/custom-arms.env`.

## 3. Repo-internal audit (strata-custom-2, 45 root entries)

Principled splits that STAY: `src/` vs `include/` vs `engine/` (sources / headers / build output); the five outer-layer homes (`serve` imports `strata_control`; `strata_cli` imports `strata_control`; `bin/` execs `strata.py`; `apps/host` references `integrations/`; `packaging` references `strata_cli`); `bench/` (hardware micro) vs `campaign/` (A/B system) vs `tools/` (dev tooling) vs `tests/` (pytest); `sycl/` (self-contained Intel line, zero filename overlap with `serve/`) vs `metal/` (parity tests) vs `docker/` vs `packaging/` vs `third_party/` (5 vendored deps).

Folds that LAND:
- `.refs/` EMPTY → rm. `ref/` (1 file, `ref/load.py` correctness oracle, imported by `tools/canonical_xcheck.py`, `serve/structured.py`, `tests/desktop/test_control_install.py`) → merge into tools/.
- `ask()` one-shot client reimplemented ≥9×: `tools/needle_bench.py:54`, `tools/research_run.py:45`, `tools/split_mtp_batch_parity.py:34`, `tools/qwen35/q36_serve_par.py:18`, `tools/qwen35/q36_serve_handoff.py:14`, `campaign/mmlu-pro-gate.py:30` (also embeds the API-key literal at :33), `campaign/template-probe.py:24`, `campaign/tpl-ux-probe.py:24`, `campaign/t42-prefill-ladder.py:30`. Owner: `tools/strata_bench.py:130 chat()`.
- `http_json` diverged twins: `tools/strata_bench.py:117` (json) vs `tools/strata_mcp.py:388` (status, json). One owner, two thin wrappers.
- `wait_ready` ×3: `tools/ab_engine.py:51`, `tools/strata_bench.py:708`, `tools/strata_mcp.py:2140`.
- API key duplicated: committed in `config/*.json` (the 36f6098b estate-constant ruling — env-overridable via STRATA_API_KEY) AND hardcoded again at `campaign/mmlu-pro-gate.py:33`. Kill the second literal only; the ruling stands.

Git state: branch main, 69 commits ahead of origin/main (unpushed), 0 behind. Uncommitted: 6 deleted README translations (`README.{de,es,fr,ja,pt-BR,zh-CN}.md`, worktree deletions — recoverable from history either way) + 2 untracked live-lane files (`config/custom-dsv41-iq2xs-8081.json`, `launch-custom-dsv41-iq2xs-8081.sh`). .gitignore coverage complete (build*/:12, logs/:14, __pycache__/:15, /.venv/:29, /engine/:50). Tracked binaries benign (Tauri icons + data seeds). **History bloat: ~1.04 GB of the 1.4 GB .git is pruned-from-HEAD blobs (850 MB ds4-slots/, 83 MB engine/snapshots/, 68 MB src2.tar, 39 MB custom-iq3_s.log) — NOT actionable under the no-history-rewrite law (shared remote svgop/Strata); operator-override item only.**

`build-fresh/` (1.4 GB) is LOAD-BEARING (exe for all 5 dsv41 configs + the live `strata-mm3`); `engine/` (102 MB) is load-bearing for the Qwen lanes (`campaign/battery-lib.sh:41`); `build-dbg/` (1,014 MB) has ZERO references — regenerable.

## 4. Live reference graph (the move-safety set)

All absolute-path holders that any estate move must update in the same pass:

- `config/*.json` — all 10; every one carries absolute exe/log/cwd/tokenizer/pack/model/mtp paths.
- Root launchers ×6 + the untracked iq2xs one — slot-dir exports (`STRATA_DS4_SLOT_DIR=/home/svg/LLM/data/ds4-slots`, `STRATA_CONV_SLOT_DIR=…/conversation-slots`). `launch-custom-mm3-8081.sh` carries no absolute path (delegates via `campaign/lib/launch-common.sh`) — the model shape.
- `campaign/battery-lib.sh:41,48` — kill-by-path matches on `$REPO/engine/` + `$REPO/build-fresh/`.
- Retrain/harvest tooling: `tools/mtp_train.py:6,110`, `tools/mtp_rt2_assemble.py:6-25` (STOCK_RT=data/mtp/rt), `tools/harvest_corpus.py:10,23`, `tools/harvest_sweep.sh:6-20`, `campaign/retrain/train_cycle.py:8-13,69-74` (RECORDS_GLOB, CKPT_DIR, EXPORT_DIR → Custom-data), `campaign/retrain/check_export_roundtrip.py:33-34`, `campaign/retrain/stair_probe.sh:23`.
- mm3 tools: `tools/mm3/vq_pack.py:78` (DEFAULT_OUT=data/packs/minimax-m3.1-flash-vq), `vq_real_diff.py`, `vq_tokenizer.py`; `tests/mm3/mm3_vq_test.cu`.
- Data-side: `data/packs/*/ds4.txt` model_dir keys (ist-das DANGLING → IQ2-XXS deleted 2026-11-14; iq2xs resolves); `data/packs/q2_0/manifest.json` cites `/home/svg/LLM/Strata-data/` (vanished third historical home — provenance only, blobs present; manifests are generated, never hand-edited); `data/mtp-en/*` 7 absolute symlinks (all resolve).
- Doctrine: `data/AGENTS.md` (zero absolute paths — move-safe; but stale: lists ds4-slots, documents 3 of 7 packs, names `../strata/` an engine home) and `strata-custom-2/AGENTS.md:291,310` (still declares `Custom-data/` THE data home — contradicts data/AGENTS.md).
- `serve/server.py` has NO absolute estate literals (derives slot dir from log-path parent, :3413/3449) — the server itself is path-free.
- Historical receipts (campaign/results 962 files, dated digests, SVG-CUSTOM journal) are records, NOT rewritten — update-on-touch only.

## 5. Contradictions and surprises

1. Three documents, three mutually exclusive stories about Custom-data: sc2 `AGENTS.md:291` ("Custom-data is the data home"), `data/AGENTS.md` (data/ is the home, Custom-data unmentioned), `.docs/research/70-roadmap.md:157` ("Custom-data does not exist" — false since Oct 24). Disk truth: Custom-data is the T14B campaign home, wired into 7+ committed tools.
2. The trained drafter never shipped: the whole T14B apparatus (4.4 GB ckpts, 106 GB harvest) produced `rt2-deploy`, sitting unreferenced next to a serving home that boots stock dense. Deploy-or-record is an open operator decision.
3. `data/mtp/tensors/` (4.9 GB) silently changed class from "derivable cache" to "only local copy" when its safetensors sources were deleted 2026-11-14. Nothing records the status change; rule-5 checks would now correctly refuse deletion.
4. `strata/spark/build_engine.sh` is TRACKED upstream content (blob c9d028e3 in upstream/main), not operator residue.
5. `fixtures/dflash2/model.safetensors` (1.2 GB real weights in Custom-data) while `data/references/DFlash2` deliberately holds only the LFS pointer — the reference spec honored in one home, bypassed in the other.
6. `q2_0/manifest.json` cites the vanished `Strata-data` home — the estate has had ≥3 data-home names in its history (Strata-data → Custom-data → data). The fold ends the pattern.
7. The estate-root stray is exactly one file (`qwen_client.py`) + pycache; `~/LLM/backup`, `~/LLM/tqllm`, `strata-custom-v1` etc. referenced in SVG-CUSTOM.md are already pruned — those references are historical only.
8. Live-config exposure: `config/custom-mm3-8081.json` carries the api_key plaintext (estate-constant ruling) and `--profile-out` INTO `data/packs/minimax-m3.1-flash-vq/` — the running engine writes its own expert-profile there (pack dir mtime Nov 15 15:36). Pack moves must respect open write handles.
9. `bench/results/2026-10-28-community-rtx-5090-deepseek-v4-1-flash/` (28 KB receipt) exists ONLY in the doomed clone — salvage before deletion.

## 6. Reclaim ledger

| Item | Size | Class |
|---|---|---|
| `strata/` clone (incl. engine-dsv41 665 MB, ds4-slots 413 MB, build 410 MB, old engine binaries 228 MB, .venv 223 MB) | 2.4 GB | redundant + stale |
| `build-dbg/` | 1,014 MB | regenerable, zero refs |
| `harvest/records/` (round 1 ⊂ round2-aligned) | 29 GB | duplicate (sha-gate all 11 before rm) |
| `quarantine/` | 7.7 GB | rejected streams — operator call |
| `records-d1/` | 12 GB | earlier variant — operator call |
| `data/mtp/mtp-q2_0.gguf` | 889 MB | no estate referent (regenerability gate first) |
| mtp experts.bin dedup (2 of 3 copies) | 823 MB | sha-proven duplicate |
| ist-das lane (pack 8.5 MB + 2 configs + 2 launchers) | ~50 KB | dangling since XXS deletion |
| estate `.docs/bench/zig-vs-cpp` binaries | 11.8 MB | regenerable from sources |
| **Committed total** | **~33.5 GB** | |
| Optional (operator signoff) | ~21 GB more | quarantine + d1 |
| .git history bloat | 1.04 GB | held by no-rewrite law |

## 7. Recovery records (king-deletion law — ledgered before any deletion)

- `strata/` clone: origin `https://github.com/Niko1221/Strata.git`, HEAD `9f7d977d` (∈ upstream/main, rev-list count 0). Re-clone: `git clone --depth 1 <origin>`. `spark/` = upstream blob c9d028e3 (in sc2 history). `engine-dsv41/` = packaged 0.1.52, feed-reobtainable (BUILD.json source `f-027-rel a42cc3d7`). `ds4-slots/` 413 MB = 17 conversation slots Oct 27–28 — **deletion loses those conversations; operator signoff required**. `refusal-r.gguf` deleted as sha-identical to `data/vectors/` copy.
- `build-dbg/`: regenerate with the repo's debug cmake preset (same recipe as build-fresh).
- `harvest/records/`: deleted only after all 11 sha pairs match `records-round2-aligned/`; the aligned set is the artifact of record.
- `mtp-q2_0.gguf`: deleted only if the owning quant tool in `tools/` regenerates it from `data/mtp/tensors/`; otherwise keep.
- Estate `.docs/bench` binaries: rebuild from the checked-in `.zig`/`.cpp` sources (regen command recorded in the roadmap).
- README translations: tracked, recoverable via `git checkout <sha> -- README.<lang>.md` at any time.
