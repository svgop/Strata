---
type: digest
date: 2026-11-09
title: Chart law-12 repair — the missing rows since Nov-6, the re-baseline anchor block, renderer fix, and the recovery re-bank runbook
claim_under_test: "lanes.csv carries no rows after 2026-11-06 — the hostmem campaign's adopted changes (pool arming, three engine-bug fixes) have no chart rows, and the chart's baseline points at a retired composite"
verdict: "lanes.csv was NOT touched (append-only preserved). Prepared rows land in this digest: 3 kind=note rows (e8484c36 budget / f95455a6 single-owner / 919b4db3 release-thread), GSQ + UD + DS recovery-bank row templates, and a `#`-comment anchor block that pins leg-0 206.3/186.7/258.9 as the only legal re-baseline. Renderer breakage found and fixed: build-trend's chart JS was dead since the Nov-6 UD rows (TypeError on the UD engine keys — no panel drew at all); both builders now skip `#` comments and carry kind=note as annotations; lane-trend.html + pack-compare.html regenerated and verified."
evidence: campaign/tracker/lanes.csv (ends line 152, 2026-11-06); campaign/tracker/build-trend.py, build-packcompare.py, README.md; campaign/results/hostmem-leg0-verdict.md; campaign/results/gsq-restore-r1.out (witness lines); .docs/research/digest/2026-11-09-regression-bank-provenance.md; git e8484c36 f95455a6 919b4db3
---

# Chart law-12 repair (2026-11-09)

Law 12: every adopted change lands a row. Since Nov-6 the hostmem campaign adopted: the pool-armed GSQ architecture (df1a9c90), the single-owner store rule (f95455a6), the pool-blind budget fix (e8484c36), and the slot-release crash fix (919b4db3) — none rowed. The recovery banks that measure them are mid-flight (gsq-restore) or pending (ud-restore). This digest carries everything the tracker needs the moment they land. Nothing is appended to `lanes.csv` yet.

## 1. Prepared rows — append these when the recovery banks land

CSV field discipline: labels carry NO commas (the file never quotes fields; `;` separates clauses). `kind=note` rows use `lane=note`, `tps=0`, `ctx=0`; both builders keep them out of the charts and carry them into `lanes.json` per engine. `<...>` = fill from the verdict file.

### 1a. The three engine-bug fixes (kind=note — land these with the first recovery commit)

```
qwen,2026-11-09,e8484c36,NOTE engine fix ADOPTED: pool-blind complement budget - MemAvailable ignores the reserved pool so a pool-equipped box read RAM-starved and the complement CAPPED at 19.80 of 37.97 GiB (the rest through the file cache; solo 120-165 t/s = the REAL post-reboot GSQ regression; measured in hostmem-definitive 146.8/154.4/195.0); the budget now adds huge_pool_free_bytes() exactly as dsv41's copy budget always has (ds4_os.hpp); the complement sizes whole again and draws pool pages,note,0,0,note
qwen,2026-11-09,f95455a6,NOTE launcher rule ADOPTED: single-owner store - when the reserved pool covers the store (>=17000 free pages) the engine's memfd builds from the canonical pack and the /dev/shm fallback copy is skipped; the tmpfs tier was dead resident weight (31.6 GiB held twice; page-cache squeeze) and the gsq-clean legs ran pre-fix still double-holding it; pool-short boots keep the old ladder; rollback STRATA_SHADOW_MEMFD=0,note,0,0,note
qwen,2026-11-09,919b4db3,NOTE engine fix ADOPTED: slot-release thread crash live in every boot since the #1603 rename - _release_slot_when_done's wait() closure referenced bare 'releasing' (NameError after busy[slot]=False and before notify_all) so waiters slept their full timeout on every stop/disconnect release; the crash is in every battery .out since Nov-7 (clock-audit suspect list); affects ALL engines' boots since Nov-7 - treat pre-919b4db3 legs as carrying a release-latency tax; alias joins q/busy/held,note,0,0,note
```

Rationale for kind=note over in-label annotation: law 12 wants one row per adopted change and these three fixes have no lane measurement of their own — a measurement-shaped row (lane=solo) would plant a fake 0 tok/s point. The chart legend per engine now reads `lanes.json` → `engines[eng].notes`, so the annotations stay machine-readable.

### 1b. Recovery bank rows (kind=standing — one point = 3 lane rows; fill when the batteries bank)

**GSQ pool-armed** (gsq-restore; 5-boot battery, bank = last-3 mean; awaiting the running verdicts):

```
qwen,2026-11-09,<commit>,GSQ-RESTORE pool-armed ADOPTED standing re-bank (5 boots banked on last 3: <s1>/<p1>/<c1> + <s2>/<p2>/<c2> + <s3>/<p3>/<c3>; vs anchor leg-0 206.3/186.7/258.9 = <d1%>/<d2%>/<d3%>; witnesses: complement 37.85 of 37.85 GiB in 2 MiB pages (HUGETLB POOL); store memfd 31.64 GiB single-owner populated 7.4 s; spec window early 16.9 -> late 12.1 ms stable; rollback STRATA_HUGEPAGE_ARENAS=0 + STRATA_SHADOW_MEMFD=0),solo,<mean-solo>,0,standing
qwen,2026-11-09,<commit>,GSQ-RESTORE pool-armed ADOPTED standing re-bank (same point as the solo row),pair,<mean-pair>,0,standing
qwen,2026-11-09,<commit>,GSQ-RESTORE pool-armed ADOPTED standing re-bank (same point as the solo row),conc,<mean-conc>,0,standing
```

**UD packs** (ud-restore; UD-IQ2_XS reads against its own leg-0 anchor; UD-IQ4_XS has no leg-0 anchor — its label must say so):

```
qwen-ud-iq2xs,2026-11-09,<commit>,UD-RESTORE pool-armed UD-IQ2_XS ADOPTED standing re-bank (5 boots banked on last 3: <s1>/<p1>/<c1> + <s2>/<p2>/<c2> + <s3>/<p3>/<c3>; vs anchor leg-0 184.2/201.1/277.7 = <d1%>/<d2%>/<d3%>; witnesses: <arena GiB> of <arena GiB> in 2 MiB pages; store memfd single-owner; spec window <early> -> <late> ms; rollback STRATA_HUGEPAGE_ARENAS=0 + STRATA_SHADOW_MEMFD=0),solo,<mean-solo>,0,standing
qwen-ud-iq2xs,2026-11-09,<commit>,UD-RESTORE pool-armed UD-IQ2_XS ADOPTED standing re-bank (same point as the solo row),pair,<mean-pair>,0,standing
qwen-ud-iq2xs,2026-11-09,<commit>,UD-RESTORE pool-armed UD-IQ2_XS ADOPTED standing re-bank (same point as the solo row),conc,<mean-conc>,0,standing
qwen-ud-iq4xs,2026-11-09,<commit>,UD-RESTORE pool-armed UD-IQ4_XS ADOPTED standing re-bank (5 boots banked on last 3: <s1>/<p1>/<c1> + <s2>/<p2>/<c2> + <s3>/<p3>/<c3>; NO same-box leg-0 anchor for IQ4_XS - read vs the Nov-6 armed spot 176.8/196.5/238.5 (kind=mean 2-leg; a spot NOT an anchor) and vs the leg-0 GSQ anchor for cross-pack physics; witnesses: <arena line>; spec window <early> -> <late> ms),solo,<mean-solo>,0,standing
qwen-ud-iq4xs,2026-11-09,<commit>,UD-RESTORE pool-armed UD-IQ4_XS ADOPTED standing re-bank (same point as the solo row),pair,<mean-pair>,0,standing
qwen-ud-iq4xs,2026-11-09,<commit>,UD-RESTORE pool-armed UD-IQ4_XS ADOPTED standing re-bank (same point as the solo row),conc,<mean-conc>,0,standing
```

**DeepSeek** (no battery running; the template closes the gap that DS has no bank at all since the Nov-8 12:22 reboot):

```
deepseek,2026-11-09,<commit>,DS pool-armed ADOPTED standing re-bank (5 boots banked on last 3: <s1>/<p1>/<c1> + <s2>/<p2>/<c2> + <s3>/<p3>/<c3>; vs last standing T61-4e 59.3/57.9/65.3 Oct-30 and the Nov-2 T64 mean 65.3/62.05/70.45; DS has had no bank since the Nov-8 reboot - this row closes that gap; witnesses: <complement/arena line>; copy budget already pool-aware (ds4_os.hpp); spec window <early> -> <late> ms),solo,<mean-solo>,0,standing
deepseek,2026-11-09,<commit>,DS pool-armed ADOPTED standing re-bank (same point as the solo row),pair,<mean-pair>,0,standing
deepseek,2026-11-09,<commit>,DS pool-armed ADOPTED standing re-bank (same point as the solo row),conc,<mean-conc>,0,standing
```

`<mean-*>` = arithmetic mean of the banked boots' lane values, 1 decimal (the T53 convention: value = mean, label carries the per-boot trios). `<d%>` = `(mean / anchor − 1) × 100` per lane. `<commit>` = the newest engine commit ALL banked boots ran — check boot wall-clock against commit times (919b4db3 landed 11:05 mid-gsq-restore: either bank only post-fix boots or name the split in the label).

## 2. The anchor block — append once, as the first recovery append

The csv now supports comment lines: any line whose first non-blank character is `#` is a comment; both builders skip them anywhere in the file (README updated). Anchor blocks append at file END (append-only holds — never insert above the header).

```
# === RE-BASELINE ANCHOR (law 12, 2026-11-09; provenance: .docs/research/digest/2026-11-09-regression-bank-provenance.md) ===
# The honest same-box A/A anchor = hostmem leg-0 (2026-11-07, 3-boot means, aged box, zero env changes):
#   qwen          GSQ IQ3_S   206.3 / 186.7 / 258.9   (campaign/results/hostmem-leg0-verdict.md)
#   qwen-ud-iq2xs UD-IQ2_XS   184.2 / 201.1 / 277.7
# qwen-ud-iq4xs and deepseek have NO leg-0 anchor yet - their first recovery banks establish theirs.
# DO NOT compare any new boot against the 2026-11-06 trio 229.1/224.8/303.5: it is a cross-boot
# composite of two unbanked 2-leg kind=mean spots (csv lines 123 solo + 130-131 pair/conc; commits
# 1553a614 and HEAD-of-Nov-6 ~8 h apart; favorable quiet window; raw legs never preserved = unauditable).
# Every recovery bank lands as kind=standing on 5-boot means and reads against the anchor above:
# +-2% = anchor-parity (the quiet-window family law); -7..-17% = the aged-window band (T59 + leg-0 precedent).
```

## 3. Renderer rerun — breakage found, fixed, verified

Ran both builders on the CURRENT csv (no rows appended). Findings:

- **build-trend.py — chart dead since Nov-6.** The generated JS hardcoded `COLORS`/`ENAME`/`visible` keyed by `qwen`/`deepseek` only; the Nov-6 UD rows introduced `qwen-ud-iq2xs`/`qwen-ud-iq4xs`. Headline proof: `TypeError: Cannot read properties of undefined (reading 'solo')` thrown at `visible[p.eng][lane]` — the IIFE died on the first panel, so NO trend panel and NO prefill chart rendered, and the four UD legend buttons rendered `undefined`. The committed Nov-6 `lane-trend.html` carries the same defect.
- **Fix:** per-engine `short` + `css` now travel inside `lanes.json` (`ENGINES[eng].css/.short` replaces both hardcoded JS maps); `visible` is derived per engine key; empty-point guards added (`draw()` returns before building scales on an empty panel). Lane colors/legend/aria-labels now correct for all four packs.
- **Convention support (needed by sections 1–2):** both builders now skip `#` comment lines and blank lines, and route `kind=note` rows into `lanes.json` → `engines[eng].notes` instead of the chart points (a note row would otherwise plant a 0 tok/s point). `build-packcompare.py` had no runtime breakage; it got the same read filter.
- **Verification:** both scripts exit 0 and write their outputs; the inline chart JS passes `node --check` and a stubbed full execution of the IIFE completes clean (pre-fix it threw the TypeError above). Dry-run in `/tmp/tracker-dryrun`: injected `#` anchor lines + two note rows → points unchanged (qwen 26), notes carried (2), pack-compare still reads the last real mean (233.1), `updated` stays on the last MEASURED row.
- Regenerated: `campaign/tracker/lane-trend.html`, `campaign/tracker/pack-compare.html` (byte-identical — csv unchanged), `campaign/tracker/lanes.json`. `lanes.csv` NOT modified.

## 4. Re-bank runbook — the exact sequence per recovery bank

**Row grammar** (all four engine keys): `engine,date,commit,label,lane,tps,ctx,kind` — recovery rows are always `kind=standing`, `ctx=0`, one point = 3 rows (`solo`,`pair`,`conc`), labels comma-free, tps = mean of banked boots. Engine keys: `qwen` / `qwen-ud-iq2xs` / `qwen-ud-iq4xs` / `deepseek`.

1. **Bank in results first.** Battery closes → raw legs (`<battery>-r*.out`, `<battery>.log`) + verdict file saved under `campaign/results/`. No verdict, no rows.
2. **Commit code/battery artifacts before the csv** (README protocol: commit first so the hash exists): `git add campaign/batteries/<battery>.sh campaign/results/<battery>*` plus any engine/launcher files, then commit.
3. **Append to `campaign/tracker/lanes.csv`** in this order: (a) the anchor block of section 2 — first recovery append only; (b) the three `kind=note` rows of 1a; (c) the recovery bank rows of 1b with every `<...>` filled from the verdict.
4. **Regenerate:** `python3 campaign/tracker/build-trend.py && python3 campaign/tracker/build-packcompare.py`.
5. **Read the chart (the incline check).** Against the anchor block, never the retired composite: a lane declining across the last two standing points is a regression signal → law-12 response ladder (same-hour control → five-boot means → binary-ladder; rollbacks `STRATA_HUGEPAGE_ARENAS=0`, `STRATA_SHADOW_MEMFD=0`). A failure signal is the restore bank reading below ~190 solo with the complement witness green.
6. **Commit the tracker:** `git add campaign/tracker/lanes.csv campaign/tracker/lanes.json campaign/tracker/lane-trend.html campaign/tracker/pack-compare.html && git commit -m "law-12 recovery bank: <battery> rows + engine-fix notes; anchor block pins leg-0 206.3/186.7/258.9; Nov-6 composite retired"`.

**Witness note template** (the label clauses every recovery label must carry; values from the boot log of each banked boot):

- Complement: `complement <X> of <X> GiB in 2 MiB pages (HUGETLB POOL)` — from `strata: hugepages: the expert complement: 37.85 GiB of 37.85 GiB in 2 MiB pages (HUGETLB POOL; THP=always [madvise] never)` (gsq-restore-r1.out).
- Store: `store memfd <N> GiB single-owner populated <t> s` — from `strata: STRATA_SHADOW_EXPERTS=...: shadow ... store mapped - ... = 31.64 GiB; 2 MiB pool pages (memfd, populated in 7.4 s)`.
- Spec window: `spec window <early> -> <late> ms stable` — from `TREND ms/window: early 16.9 -> late 12.1 (stable)`.
- Plus: per-boot trios, anchor deltas, and the rollback switches. Any witness missing or red → the row does not land; name the gap in the verdict file.
