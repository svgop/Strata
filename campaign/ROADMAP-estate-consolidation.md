---
type: roadmap
date: 2026-11-15
status: proposed (awaiting operator signoff on the flagged items)
objective: Full parity on Strata 2 — one engine system, one data home, no parallel stacks, no drift
evidence: .docs/research/2026-11-15-estate-consolidation-study.md
---

# ROADMAP — Estate consolidation (Strata 2 parity)

Kill every parallel system in `/home/svg/LLM`. End state: the estate root holds
exactly `data/`, `strata-custom-2/`, `.zcode/`; all data lives under one home
(`data/`); all research lives in the repo (`strata-custom-2/.docs/research/`);
the upstream clone is gone (absorbed by the fork's `upstream` remote + the pinned
`data/references/Strata`); every live reference resolves; both doctrine files tell
the same true story. Reclaim ~33.5 GB committed, ~54 GB with signoff items.

Laws that bind this roadmap: storage canon (move = update references in the same
pass; king-deletions ledger recovery first — study §7), data/AGENTS.md rules 1–10,
no-history-rewrite on the shared svgop/Strata remote, cross-engine change protocol
(bench → change → bench), engine-level-not-services (nothing lands in /tmp or as a
hand-edit; every fix ships wired-in), law 17 (better than found — every wave leaves
a resolving, doctor-clean estate).

## Target shape

```
/home/svg/LLM/
├── data/                      # ONE data home (weights, packs, vectors, slots, training)
│   ├── models/                # by quantization lineage (unchanged)
│   ├── packs/                 # 6 live packs (ist-das retired)
│   ├── vectors/
│   ├── mtp/                   # stock drafter weights of record + tensors/ (only local copy)
│   ├── mtp-en/                # serving drafter (symlinks)
│   ├── references/            # pinned study clones
│   ├── conversation-slots/    # Qwen runtime
│   ├── ds4-slots/             # DS runtime (pre-created in Wave 1)
│   └── training/              # ← ex-Custom-data
│       ├── harvest/           # corpus of record: records-round2-aligned/ + round2-only adds
│       └── mtp-retrain/       # T14B workspace (ckpts, fixtures, rt2, rt2-deploy, export)
├── strata-custom-2/           # the one engine system (Strata 2)
│   ├── .docs/research/        # ALL research: digests + recon bank + model-fit + zig feasibility
│   ├── tools/web/qwen_client.py  # the one relocated estate-root stray
│   └── (repo otherwise as audited: principled splits stay, folds land)
└── .zcode/                    # harness config
```

## Wave 0 — Git safety net (before anything moves)

| # | Action | Owner | Exit criterion |
|---|---|---|---|
| 0.1 | Commit the two untracked live-lane files: `config/custom-dsv41-iq2xs-8081.json`, `launch-custom-dsv41-iq2xs-8081.sh` | git | `git status --short` no longer lists them |
| 0.2 | Commit the 6 README-translation deletions (deliberate fork hygiene; recoverable from history) | git | worktree clean of ` D README.*` |
| 0.3 | Commit the consolidation study + this roadmap (`.docs/research/2026-11-15-estate-consolidation-study.md`, `campaign/ROADMAP-estate-consolidation.md`) | git | both tracked |
| 0.4 | `git push origin main` (69 + wave-0 commits) | git | `git rev-parse origin/main == HEAD` |

## Wave 1 — Upstream clone retirement (estate)

| # | Action | Owner | Exit criterion |
|---|---|---|---|
| 1.1 | Salvage the unique receipt: `cp -r strata/bench/results/2026-10-28-community-rtx-5090-deepseek-v4-1-flash strata-custom-2/bench/results/` + commit | bench/results | dir exists in fork; committed |
| 1.2 | SIGNOFF GATE: `strata/ds4-slots/` (413 MB, 17 slots, Oct 27–28 conversations) is lost on deletion — operator confirms | — | operator ack in writing |
| 1.3 | `rm -rf /home/svg/LLM/strata` (recovery record: study §7) | estate | `ls /home/svg/LLM` no longer shows `strata` |
| 1.4 | `mkdir -p data/ds4-slots` (the doctrine-listed home materializes; engine also self-creates) | data | dir exists, matches launchers' export |
| 1.5 | Update `data/AGENTS.md`: drop `../strata/` as an engine home; name `strata-custom-2` the only engine and its `upstream` remote + `data/references/Strata` the reference pair | data/AGENTS.md | `grep '../strata' data/AGENTS.md` = 0 |

## Wave 2 — Custom-data fold (the 113 GB rename; same filesystem, instant)

| # | Action | Owner | Exit criterion |
|---|---|---|---|
| 2.1 | Process gate: `lsof +D /home/svg/LLM/Custom-data` returns nothing (mm3 does not hold it; retrain idle) | — | empty output |
| 2.2 | `mv Custom-data data/training` then `mv data/training/mtp data/training/mtp-retrain` (kills the third "mtp" name collision) | estate | both paths exist; old names gone |
| 2.3 | Update every live reference, one commit: `tools/mtp_train.py`, `tools/mtp_rt2_assemble.py`, `tools/harvest_corpus.py`, `tools/harvest_sweep.sh`, `campaign/retrain/train_cycle.py`, `campaign/retrain/check_export_roundtrip.py`, `campaign/retrain/stair_probe.sh`, `strata-custom-2/AGENTS.md` (:291 data-home claim + :310 fixtures route) | listed files | `grep -rn "Custom-data" tools/ campaign/retrain/ campaign/lib/ config/ launch*.sh AGENTS.md data/AGENTS.md` = 0 hits |
| 2.4 | `data/AGENTS.md` gains the `training/` section (harvest corpus of record + mtp-retrain workspace + the fixtures/DFlash2 note) | data/AGENTS.md | section present, every path resolves |
| 2.5 | Proof: `python3 tools/harvest_corpus.py --help` + a dry path-check of `campaign/retrain/train_cycle.py` resolve under the new home | tools, campaign/retrain | both run clean |
| 2.6 | Dated records (campaign/results receipts, digest files, SVG-CUSTOM journal) are NOT rewritten — update-on-touch only | — | no mass rewrite in the diff |

## Wave 3 — Dead-weight removal (per-item gated)

| # | Action | Owner | Exit criterion |
|---|---|---|---|
| 3.1 | `rm -rf strata-custom-2/build-dbg` (1,014 MB; recovery = debug cmake preset; zero references proven) | repo | dir gone; `grep -rn build-dbg` across configs/launchers/campaign = 0 |
| 3.2 | Retire the dead ist-das DS lane (dangling since the 2026-11-14 XXS deletion, `model_dir` proven broken): `git rm` `config/custom-dsv41-8081.json`, `config/custom-dsv41-4e-8081.json`, `launch-custom-dsv41-8081.sh`, `launch-custom-dsv41-4e-8081.sh`; `rm -rf data/packs/deepseek-v4.1-flash-ist-das-2.00bpw` | config/, launchers, packs | no config/launcher references the ist-das pack; DS fleet = iq2xs lane only |
| 3.3 | `data/mtp/mtp-q2_0.gguf` (889 MB): verify the owning quant tool regenerates it from `data/mtp/tensors/` → delete + ledger; else keep | data/mtp | gate decided and ledgered either way |
| 3.4 | Harvest dedup: sha256 all 11 `records/` pairs vs `records-round2-aligned/`; all equal → `rm -rf harvest/records/` (the aligned set is the artifact of record) | data/training/harvest | sha ledger appended to study §6; 11/11 match proven before rm |
| 3.5 | SIGNOFF ITEMS: `quarantine/` (7.7 GB rejected streams) and `records-d1/` (12 GB earlier variant) — delete, archive, or keep | data/training/harvest | operator decision recorded |
| 3.6 | mtp experts dedup: rm the two redundant `experts.bin` copies in `mtp-retrain/rt2` + `rt2-deploy` (sha-proven identical to `data/mtp/rt`; `rt2_meta.json` records the sha) — or replace with hardlinks to keep workspace self-contained | data/training/mtp-retrain | sha256 re-verified in-pass; serving drafter untouched |
| 3.7 | rm empty dirs: `strata-custom-2/.refs/`, `strata-custom-2/logs/ds-pictures/`, `data/references/Research/` | repo, data | gone |
| 3.8 | Run `tools/strata-doctor.py` + df delta logged | repo | doctor passes; GB reclaimed recorded |

## Wave 4 — Research-home consolidation + estate-root cleanup

| # | Action | Owner | Exit criterion |
|---|---|---|---|
| 4.1 | `mv /home/svg/LLM/.docs/research/* strata-custom-2/.docs/research/` (recon bank, model-fit, zig feasibility join the digest stream — ONE research home, versioned) | repo .docs | all files present; estate `.docs/research` gone |
| 4.2 | Move `zig-vs-cpp` SOURCES (`.zig`/`.cpp`) into `strata-custom-2/.docs/bench/zig-vs-cpp/`; rm the 11.8 MB binaries (regen command ledgered in study §7) | repo .docs | sources present; regen command runs or is noted |
| 4.3 | `mv /home/svg/LLM/qwen_client.py strata-custom-2/tools/web/qwen_client.py` + commit; rm estate `__pycache__/` | tools/web | estate root clean |
| 4.4 | `rm -rf /home/svg/LLM/.docs` | estate | `ls -A /home/svg/LLM` = `data  strata-custom-2  .zcode` |

## Wave 5 — Repo-internal folds (code; A/A-gated)

| # | Action | Owner | Exit criterion |
|---|---|---|---|
| 5.1 | Fold the ud launcher twins (byte-identical except 2 lines) into one parameterized `launch-custom-ud-8081.sh <iq2xs\|iq4xs>`; launch law kept (stays at repo root) | launchers | `bash -n` clean; both configs boot; lanes in band (A/A vs the standing bank) |
| 5.2 | Fold lane-independent standing arms (launcher lines 16–39) into `campaign/arms/standing.env` sourced by `campaign/lib/launch-common.sh` | campaign | one owner; mm3 + one Qwen lane boot A/A in band |
| 5.3 | Fold the `ask()` ×9 implementations onto `tools/strata_bench.py chat()`; keep per-site wrappers one call thick | tools/, campaign/ | `grep -rn "def ask(" tools/ campaign/` = owner-only; touched scripts py_compile clean; one gate script (mmlu-pro-gate.py) runs end-to-end |
| 5.4 | `http_json` twins → one owner, two thin wrappers (`tools/strata_bench.py:117`, `tools/strata_mcp.py:388` keeps its tuple contract) | tools/ | both callers pass their tests |
| 5.5 | `wait_ready` ×3 → `tools/strata_bench.py:708` as owner | tools/ | ab_engine + strata_mcp import it; smoke passes |
| 5.6 | Kill the duplicate api-key literal at `campaign/mmlu-pro-gate.py:33` (read config/env per the 36f6098b estate-constant ruling; the ruling itself stands) | campaign/ | gate script runs with env var only |
| 5.7 | `ref/load.py` → `tools/` (update 3 importers: `tools/canonical_xcheck.py`, `serve/structured.py`, `tests/desktop/test_control_install.py`); rm `ref/` | tools/, serve/, tests/ | 3 importers' tests pass; dir gone |

## Wave 6 — Doctrine sync + executable invariant

| # | Action | Owner | Exit criterion |
|---|---|---|---|
| 6.1 | `data/AGENTS.md` full refresh: 7 packs → post-Wave-3 set, `tensors/` = only-local-copy status, `training/` section, ds4-slots pre-created, engine-home correction (Wave 1.5 folded here if not done) | data/AGENTS.md | every path in the file resolves on disk (script check) |
| 6.2 | `strata-custom-2/AGENTS.md`: Custom-data sections → `data/training`; estate-shape section matches the target tree | AGENTS.md | same script check |
| 6.3 | Extend `tools/strata-doctor.py` with an estate-path check (#9): every absolute path in `config/*.json` + root launchers + `data/packs/*/{ds4,manifest}.txt/json` resolves | tools/strata-doctor.py | doctor runs 9/9 clean; committed (ships wired-in, not a script in /tmp) |

## Optional items — operator decisions only

1. **Repo dir rename** `strata-custom-2` → `strata` (the `-custom-2` suffix references a dead sibling). One mv + 10 configs + 6 launchers + 2 doctrine files. Cost: muscle memory + historical absolute paths in dated records. Default: keep.
2. **Deploy the trained drafter**: `mtp-retrain/rt2-deploy/dense.bin` (best_mean 0.7932 vs stock 0.4049) sits unshipped while serving boots stock dense. Deploy via `tools/mtp_rt2_assemble.py` or record the hold.
3. **quarantine/ + records-d1/** (~19.7 GB): delete, archive off-device, or keep.
4. **git filter-repo** (~1.04 GB dead history): violates the no-rewrite law on the shared svgop/Strata remote — requires explicit operator override + coordinated force-push. Default: never.

## Proof harness (runs after every wave)

- `tools/strata-doctor.py` (9 checks after Wave 6.3 lands; 8 before).
- Path-resolution sweep: every absolute path in the 8 surviving configs + launchers + pack manifests resolves.
- Lane boot proof per touched lane: DS iq2xs, Qwen iq3_s / ud-iq2xs / ud-iq4xs, mm3 — each boots once post-change, banks to `lanes.csv` in band (cross-engine protocol; schedule around the live mm3 occupant of 8081).
- `git status` clean + pushed at each wave boundary.

## Execution timing

Waves 1–4 do not touch the live mm3 lane's read/write set (exe in `build-fresh/`, pack + `models/VQ/` in `data/`) — safe alongside PID 48587. Wave 3.2 and Wave 5 proof legs need 8081 free — run in the next lane-change window.
