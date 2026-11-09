---
type: research-digest
id: 2026-11-09-cleanup-repo-audit
date: 2026-11-09
scope: post-campaign artifact-hygiene audit of campaign/ (tracked inventory by class, duplicates, superseded gpu-lock versions, .gitignore policy, secret leakage) and the ordered tidy plan
subject: this digest only; READ-ONLY audit — no file under campaign/ moved, edited, or deleted; every action below is a proposal for the operator to run
method: git ls-files/stat blob census of campaign/ (2314 files), git ls-tree blob-hash duplicate scan, git grep secret sweep, reference check of every archive candidate against living code (launchers, batteries, docs/, tools/), .gitignore + campaign/README.md (F-267/F-268) policy read
---

# Repo cleanup audit — 2026-11-09

READ-ONLY pass. Nothing deleted or moved. The tidy is the command list at the end; run it top-to-bottom or not at all.

## 1. Inventory (tracked files under campaign/)

Total: **2314 tracked files, 34.1 MB**.

| Class | Count | Bytes | Note |
|---|---|---|---|
| results/ .out | 547 | 16.3 MB | boot logs — the raw evidence store |
| results/ .txt | 1049 | 4.5 MB | probe captures (`PROBE ... = N t/s` lines), mostly 1-3 KB |
| results/ .log | 128 | 2.6 MB | battery summaries |
| results/ .json | 115 | 0.97 MB | metrics/gates |
| results/ md+old-logs+t18-crashes+misc | 22 | 4.3 MB | old-logs/ = 2 rotated .gz (3.7 MB), t18-crashes/ = 5, metrics.jsonl, golden-response.sha256, 1 .bin (expert-profile-learned.poisoned-t33 — deliberate poisoned-state evidence, keep), 1 .diff, 1 .err, 1 .sh |
| batteries/ | 57 | 128 KB | 54 .sh runners + 3 summary .py |
| headport/ | 152 | — | C++/CUDA source + 11 json + 11 sh |
| research/ | 126 | — | dossiers, INDEX.md is the TOC |
| upstream/ | 26 | — | 2 arm json, 10 t50 patches, 2 test .sh, **12 .out boots** |
| arms/ | 22 | — | 11 launch .sh + 11 arm json (parked per README) |
| tracker/ | 7 | 110 KB | lanes.csv = chart data (law), builders, 2 html |
| campaign/ root | 48 | ~0.8 MB | living instruments + gpu-lock v4/v4.service/v5/v6 + 2 sysconfig .conf + probes + 2 ELFs + ROADMAPs |
| l1-falsifier/, shadow-bench/, gpu-lock/, retrain/, T14B-specs/ | 17 | 1.3 MB | includes **l1_commit_falsifier: 1.3 MB tracked ELF** |

Biggest 10 tracked files repo-wide (git ls-files | stat): `engine/snapshots/strata.pre-w1` **86.8 MB tracked engine binary** (outside campaign, pre-W1 snapshot), then third_party/llama.cpp vocab ggufs (15.8, 10.9, 7.8, 6.8, 5.9, 5.9, 4.8, 4.0 MB) and `third_party/.../miniaudio.h` 4.1 MB. Campaign-scoped biggest: old-logs .gz (2.8 MB, 0.9 MB — fine, compressed archives), l1_commit_falsifier ELF (1.3 MB), t18-{pre2,fixed,pre}-server.out (0.69/0.69/0.47 MB), f325-stair-*.out x4 (~0.45 MB each).

## 2. Duplicates

**Root vs batteries name collisions:** exactly one — `stock-jury.sh`. HEAD blobs differ (`campaign/stock-jury.sh` = 43768fe, `campaign/batteries/stock-jury.sh` = 0bd38a3): the batteries copy is the current instrument (writes `results/stock-jury.log` + `stock-jury-boot.out`), the root copy is already gone from the worktree (unstaged `D` — the migration happened, the deletion just was never staged). No other name collisions; no same-blob script pairs across root/batteries.

**.out whose .log also exists:** 6 stems — `t38-abab-b`, `t38-bincf`, `t38-sanity`, `t38-tiebreak`, `t40-verify`, `t41-prob-gates`. The other **541 .out have no .log sibling**: the .out IS the record. Mass .out deletion is off the table.

**Byte-identical .out groups (same git blob):**
- `f299-final-serve.out` = `f310-restore.out` = `f321-restore.out` = `f322-restore.out` (4x, 1382 B — the standard restore-boot banner, deterministic). Keep one canonical, drop 3.
- `t21-c3-arm-r1.out` = `r2` = `r3` (3x, 145 B — `launch-c3.sh: line 47: .venv/bin/python: No such file or directory`). Degenerate failure, no boot occurred, zero evidence. Drop all 3.

**Zero-byte tracked files (17):** f164-boot-err.txt, f165-err.txt, f166-err.txt, f167-err.txt, f206-r33-2.txt, f206-r33-3.txt, f362-on1-2.txt, final-err.txt, g2-arm-r3.txt, merged-boot-err.txt, pin3850-err.txt, pin5000-err.txt, reserve2000-err.txt, restore-2242-err.txt, restore-2243-err.txt, spec5-err.txt, t26-calibrate.err — all empty error captures.

**Superseded gpu-lock chain (who references what):**
- `campaign/gpu-lock/strata-gpu-lock-v3` + `gpu-lock/README.md` — **dead in the worktree**. Only reference is historical (`.docs/research/digest/2026-11-09-regression-clock-box.md:202` describes what that pass read). No living code path.
- `campaign/strata-gpu-lock-v4` + `v4.service` — **LIVE**: `docs/PERFORMANCE-SETUP.md:26` installs the .service; v5/v6 digests name v4 as the hard-rollback and design base.
- `campaign/strata-gpu-lock-v5` — **LIVE**: standing regime ("the standing regime stays v5-range until the falsifier fires", governor-v6 digest line 6); install/rollback source in gsq-recovery-matrix.
- `campaign/strata-gpu-lock-v6` — **LIVE**: install source at `docs/PERFORMANCE-SETUP.md:25`, checked by `tools/strata-doctor.py:190`.
- Verdict: keep v4/.service, v5, v6 exactly where they are (doc-referenced paths, README law: do not move); v3 pair is the only dead version.

## 3. Target layout

- **campaign/batteries/** stays the home of runnable instruments. **campaign/ root keeps the living instruments** (battery-lib.sh, probe-lane.py, probe-metrics.py, binary-ladder.sh, env-ladder.sh, mmlu-pro-gate.py, mimicbench2.py, bit-check.py, ctx-probe.py, depth-lane.py, depth-lane-real.py, mma-accum-ratelock.cu, t31-keeprefit.py — README law: referenced paths do not move), plus shm-shadow.sh (sourced by all three root launchers) and template-probe.py (sourced by batteries/t39-template-gates.sh + tpl-user-gates.sh).
- **campaign/archive-2026-11/** — single archive dir for concluded single-question instruments: hostmem-probe.sh, conc-probe.py, prefill-probe.py, t42-prefill-ladder.py, t43-smoke.py, t48-ngram-probe.py, t48-ngram-analyze.py, t50-131k-probe.py, tpl-ux-probe.py, tpl-user-render-battery.py, mimic-compare.py + gpu-lock/ (v3 pair). Carries a README mapping old→new path. *Conflict named:* campaign/README.md F-267 says run-once harnesses for closed questions get DELETED (git history is the archive). Archive is the reversible choice; deletion is the written law — operator picks. Exception: `prefill-probe.py` is quoted as `prefill-probe.sh` by `campaign/T14B-specs/harvest-spec.md` (a dangling reference — the .sh does not exist); fix the spec reference either way.
- **campaign/results/ stays FLAT.** Era dirs (results/<era>/) would churn 1859 files and break the exact paths digests and tracker note flows quote (e.g. `campaign/results/{era-interleave.log,gsq-warm-verdict.log}`, `campaign/results/gsq-close`). Storage canon: index and home are different things — add **`campaign/results/INDEX.md`** grouping by era instead: t15–t69 instrument era, f200–f367 fix era, hostmem/gsq/pin boot-storm era, bat/cf/qr/cr/rab/era/tpl/v02/ud arms, old-logs/, t18-crashes/.
- **Removal list (zero-value only):** the 17 zero-byte err files; t21-c3-arm-r{1,2,3}.out (degenerate identical); f310/f321/f322-restore.out (byte-identical to f299-final-serve.out, keep 1); ELFs `campaign/hostmem-anon-collapse` and `campaign/l1-falsifier/l1_commit_falsifier` untracked from git (sources `hostmem-anon-collapse.c` and `l1_commit_falsifier.cu` + `build.sh` remain; precedent: `campaign/shadow-bench/shadow_bench` is already ignored). Nothing else qualifies — every remaining .out/.txt/.log is the sole or paired record of a run.

## 4. .gitignore coverage and policy

Current state: `*.log` ignored globally with `!campaign/results/*.log` + `!campaign/results/*.txt` re-admitted (T20 fix — correct); `server-*.out` ignored; **plain `.out` is not ignored anywhere**, and F-267 says a battery's .out commits with its .log.

Policy recommendation (keep the law, close the gaps):
1. **Keep committing `campaign/results/*.out` as evidence.** The task's sketch ("commit .log, ignore .out") would destroy the record: 541/547 .out are the sole record of their run. .out committed with the battery commit; empty .out never committed.
2. Ignore boot .out outside the results store — `campaign/upstream/*.out` after the upstream boots are archived/deleted, and guard against new strays: `campaign/**/*.out` + `!campaign/results/*.out`.
3. Ignore the two ELFs (`campaign/hostmem-anon-collapse`, `campaign/l1-falsifier/l1_commit_falsifier`) — build products, never sources.
4. Ignore `.env` (the new secret home from section 5).
5. The untracked worktree evidence right now — `results/pin-ctl1.out`, `pin-ctl2.out`, `pin-pin1.out`, `pin-pin2.out`, `condring-parked.patch`, modified `results/stock-jury-boot.out` — is this session's record: commit it, do not ignore it.

## 5. Secret leakage

**Engine API key `3f0940df…e0b0df1b8` (full: 3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8) is hardcoded in 46 tracked files / 46 occurrences.** Zero occurrences inside campaign/results/ (the evidence store is clean). Files:lines:

- campaign/arms/*.json (api_key fields): custom-iq3_s-8081-b12.json:74, custom-iq3_s-8081-b9.json:74, custom-iq3s-262k.json:70, f337-pool14-iq3_s-8081.json:76, f345-ec9000.json:74, f345-reserve4500.json:74, f348-routing.json:76, f349-pw12.json:76, f354-b9c3.json:74, f355-kv16k.json:74, f355-kv8k.json:74
- campaign/batteries/: gsq-frac-sweep.sh:16, t22-custom-vs-stock.sh:47, t38-sanity.sh:33, t40-verify.sh:15, t40b-verify.sh:14, tpl-user-gates.sh:14
- campaign/*.py: bit-check.py:6, conc-probe.py:37, ctx-probe.py:8, depth-lane.py:16, depth-lane-real.py:23, mimicbench2.py:55, mmlu-pro-gate.py:35, prefill-probe.py:12, probe-lane.py:52, probe-metrics.py:99, t42-prefill-ladder.py:11, t43-smoke.py:9, t48-ngram-probe.py:14, t50-131k-probe.py:21, template-probe.py:12, tpl-ux-probe.py:13
- campaign/upstream/: s1-mtp.json:68, s2-dflash2.json:70, test-s1.sh:4, test-window.sh:7
- campaign/research/F-208-measurement-lanes-agent.md:110 (a dossier embedding the probe snippet)
- config/: custom-dsv41-4e-8081.json:29, custom-dsv41-8081.json:35, custom-dsv41-ct32-8081.json:29, custom-dsv41-e5-8081.json:29, custom-dsv41-t54.json:30

No foreign bearer tokens or sk- keys found (apps/host hits are `YOUR_API_KEY` placeholders; bench/results hits use `$KEY` variables). Informational: .out boot banners expose RFC1918 LAN IPs (192.168.0.22/.182) — expected for a LAN-bound engine, no action.

Remediation: move the key to `STRATA_API_KEY` (env or a git-ignored `.env`), make scripts read `${STRATA_API_KEY:?set STRATA_API_KEY}` with the arm-json `"api_key"` fields reading the same env at launch; rotate at the next engine restart (it is the local engine's auth token, so severity is moderate; rotation is one restart). History rewrite optional — decide by whether this repo ever leaves the machine.

## 6. Ordered tidy command list

Run from repo root `/home/svg/LLM/strata-custom-2`, in order. Nothing here loses evidence: every removal is zero-value or byte-identical, and git history retains every path.

```bash
# --- Phase 0: commit this session's pending evidence + finished stock-jury migration ----------
# SAFE
git add campaign/results/stock-jury-boot.out campaign/results/pin-ctl1.out \
        campaign/results/pin-ctl2.out campaign/results/pin-pin1.out campaign/results/pin-pin2.out \
        campaign/results/condring-parked.patch config/custom-ud-iq4xs-8081.json \
        docs/MMVQ_IL_TABLE.md tools/lint_exceptions.json
git add campaign/stock-jury.sh            # stages the already-deleted root copy (migration to batteries/ is done)
git commit -m "campaign: commit pin/condring evidence; finish stock-jury.sh move to batteries/"

# --- Phase 1: drop zero-value artifacts ------------------------------------------------------
# SAFE (empty captures; git history keeps them)
git rm -q campaign/results/f164-boot-err.txt campaign/results/f165-err.txt \
  campaign/results/f166-err.txt campaign/results/f167-err.txt campaign/results/f206-r33-2.txt \
  campaign/results/f206-r33-3.txt campaign/results/f362-on1-2.txt campaign/results/final-err.txt \
  campaign/results/g2-arm-r3.txt campaign/results/merged-boot-err.txt campaign/results/pin3850-err.txt \
  campaign/results/pin5000-err.txt campaign/results/reserve2000-err.txt campaign/results/restore-2242-err.txt \
  campaign/results/restore-2243-err.txt campaign/results/spec5-err.txt campaign/results/t26-calibrate.err
# SAFE (3 byte-identical degenerate failure outputs — no boot ever happened)
git rm -q campaign/results/t21-c3-arm-r1.out campaign/results/t21-c3-arm-r2.out campaign/results/t21-c3-arm-r3.out
# NEEDS-REVIEW (byte-identical to f299-final-serve.out; keep one canonical restore-boot record)
git rm -q campaign/results/f310-restore.out campaign/results/f321-restore.out campaign/results/f322-restore.out

# --- Phase 2: build products out of git (sources stay) ---------------------------------------
# SAFE (ELFs rebuild: cc campaign/hostmem-anon-collapse.c; bash campaign/l1-falsifier/build.sh)
git rm -q --cached campaign/hostmem-anon-collapse campaign/l1-falsifier/l1_commit_falsifier

# --- Phase 3: archive concluded instruments ---------------------------------------------------
# NEEDS-REVIEW (F-267 says delete closed-question harnesses; archive is the reversible choice)
mkdir -p campaign/archive-2026-11/gpu-lock
git mv campaign/hostmem-probe.sh campaign/conc-probe.py campaign/prefill-probe.py \
       campaign/t42-prefill-ladder.py campaign/t43-smoke.py campaign/t48-ngram-probe.py \
       campaign/t48-ngram-analyze.py campaign/t50-131k-probe.py campaign/tpl-ux-probe.py \
       campaign/tpl-user-render-battery.py campaign/mimic-compare.py campaign/archive-2026-11/
git mv campaign/gpu-lock/strata-gpu-lock-v3 campaign/gpu-lock/README.md campaign/archive-2026-11/gpu-lock/
# then write campaign/archive-2026-11/README.md (old→new path map) and fix the dangling
# prefill-probe.sh reference in campaign/T14B-specs/harvest-spec.md:34,260,262,270,385,438,526

# --- Phase 4: index the evidence store (home stays flat) -------------------------------------
# SAFE
# write campaign/results/INDEX.md grouped by era: t15-t69 / f200-f367 / hostmem+gsq+pin /
# bat-cf-qr-cr-rab-era-tpl-v02-ud / old-logs/ / t18-crashes/ (one line per file: what ran, verdict)

# --- Phase 5: .gitignore ----------------------------------------------------------------------
# SAFE — append:
cat >> .gitignore <<'EOF'

# campaign cleanup 2026-11-09: build products and stray boot logs never enter git
campaign/hostmem-anon-collapse
campaign/l1-falsifier/l1_commit_falsifier
campaign/**/*.out
!campaign/results/*.out
.env
EOF

# --- Phase 6: secret sweep ---------------------------------------------------------------------
# NEEDS-REVIEW — 46 files carry the engine key. Move to STRATA_API_KEY:
#   1. put STRATA_API_KEY=<key> in a git-ignored .env (or shell profile)
#   2. sed the 46 lines listed in section 5 to read "${STRATA_API_KEY:?set STRATA_API_KEY}"
#      (arm json api_key fields: set from env at launcher time, not stored)
#   3. rotate the key at the next engine restart
#   4. grep -r '3f0940df' --exclude-dir=.git  ->  must return zero
git commit -m "campaign: hygiene pass — zero-value artifacts out, ELFs untracked, concluded probes archived, results INDEX, key to env"
```

## 7. Boundary

- Audit ran against the current worktree (branch main, 9851 tracked files); the 5 untracked pin/condring files were counted as pending evidence, not clutter.
- `engine/snapshots/strata.pre-w1` (86.8 MB tracked binary) and the vendored third_party ggufs are outside this audit's scope but are the real repo-weight items if the operator ever cares about clone size (git-filter-repo territory — destructive, not proposed here).
- Drift noted: `campaign/T14B-specs/harvest-spec.md` references `campaign/prefill-probe.sh`, which does not exist (only `.py` is tracked) — fix during Phase 3.
