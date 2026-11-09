---
type: research-digest
date: 2026-11-09
subject: battery-fleet consolidation study — the 14 lint-owned kill-sweeps fold onto battery-lib, 12 of 14 are era-concluded
scope: campaign/batteries/ (60 entries), campaign/battery-lib.sh, tools/lint_tree.py, tools/lint_exceptions.json (READ-ONLY study; only this digest written)
method: full read of all 14 kill-owned scripts + battery-lib.sh + one lib-based template (t38-abab-b.sh); verdict evidence from results/*.log tails, git verdict commits, tracker/lanes.csv rows; lint scan-surface proof (RECORDS regex, KILL_SIG); warm-gate marker provenance traced into serve/server.py
verdict: the lib already owns the kill mechanism (cleanup_engine, battery-lib.sh:32-51) — what the 14 scripts carry that the lib lacks is the boot/warm-gate/probe loop, not the kill. 12 of 14 are era-concluded (the GSQ campaign closed today with the v6 record bank): archive unchanged to campaign/archive-2026-11/. 2 are still-current (ds-verify blocked on pool top-up, kernel-aa bound to a 7.2.12 reboot) and convert with zero new kill logic. Binding constraint: tools/lint_tree.py:71 RECORDS must learn campaign/archive-* before the 13 era JSON entries can retire — archived scripts under campaign/ are still lint-scanned today.
---

# 2026-11-09 battery-fleet consolidation study

## 1. Fleet inventory (campaign/batteries/, 60 entries)

### 1.1 STILL-CURRENT (a future bank could run it)

| script | role | evidence |
|---|---|---|
| t20-gates.sh | adoption gates (2x MMLU + mimic) for ANY adoption | header :2-3; README "batteries/ — reusable harnesses" |
| t22-custom-vs-stock.sh | fair-baseline A/B custom vs stock; dual-checkout, port-scoped by design | lint_tree.py:48 (named KILL_EXCEPTIONS); results/t22-custom-vs-stock.log |
| t31-rebank.sh + t31-summary.py | the quiet-window re-bank instrument (F-263 law) | README pairs them ("t31-rebank.sh + t31-summary.py"); the standing re-bank path |
| t44-champ-bank.sh | the champion's five-boot bank (F-285a rider) | header :2; re-runs at every champion change |
| ds-verify.sh | DS huge-arena + pin verification; NEVER run (no results/ds-verify-*); header :2 "SKELETON, do not run until the pool is armed", :14 "NOT RUN — the pool is 45076/54272 today", :32 pool guard refuses short pools | STILL-CURRENT-BLOCKED; raw kill → convert (§4) |
| kernel-aa.sh | the 7.2.12 kernel A/A falsifier; kernel 7.2.15 remains the DATED SUSPECT (commit eb2155ec "NEW dated suspect kernel 7.2.15 installed today vs record 7.2.12"); script aborts unless uname -r is 7.2.12 (:10); the 7.2.15 side was answered by results/kernel-aa-7215-quiet.log (18:46 today) — the 7.2.12 side never ran (no results/kernel-aa-7212.log) | STILL-CURRENT-CONDITIONAL (needs a 7.2.12 reboot); raw kill → convert (§4) |

### 1.2 ERA-CONCLUDED — the 14 lint-owned (archive UNCHANGED to campaign/archive-2026-11/)

Every one carries the JSON reason "campaign research battery from the 2026-11-07..09 regression campaign" (tools/lint_exceptions.json `kill`). Verdict evidence per script:

| script | ran? | verdict / why concluded |
|---|---|---|
| era-interleave.sh | DONE 15:36 Nov 9 (results/era-interleave.log) | THE BRANCH VERDICT, commit c4b0d2de: current >= era on all 3 lanes; the record was box-state, never code. Era binary gone: /home/svg/LLM/banked-era-4152/build-era/strata absent (estate prune 9bbdfb75) — cannot re-run |
| gsq-arms-screen.sh | DONE 16:53 (gsq-arms-screen.log) | both doorbell arms parked neutral/worse (eb2155ec, cea6a4c8; verify-arms-inventory digest) |
| gsq-condring-ab.sh | DONE 17:46 (gsq-condring-ab.log) | COND_RING PARKED by law 13 (6d31b476). Both binaries pruned: engine/strata.pre-condring absent (header :10-11 documents the regen recipe), /home/svg/LLM/condring-build/build/strata absent — cannot re-run |
| gsq-final-bisect.sh | INTERRUPTED (log ends mid-leg-a2 probe 1, no DONE) | discriminator answered by the corrected instrument chain (gsq-final/gsq-restore banks, lanes.csv 003d78d8/35653c90) + 2026-11-09-gsq-recovery-matrix.md |
| gsq-frac-sweep.sh | SWEEP DONE 13:33 (gsq-frac-sweep.log) | "BEST frac 0.5 median 156.2 vs 0.35 median 155.3 = +0.6%" — neutral; standing 0.35 stands; question closed |
| gsq-pinning-ab.sh | DONE 19:27 (gsq-pinning-ab.log; legs at load 2.7-4.5, disclosed) | pinning neutral = parked env-gated (lanes.csv v6 record row); launcher default now carries STRATA_ENGINE_CPUS (a01efdca) |
| gsq-quiet-record-hour.sh | DONE 18:07 (gsq-quiet-record-hour.log, 5 boots) | the proven +10-15% quiet-window lever (eb2155ec); banked as the v6 record-hour run (lanes.csv "RECORD-CLASS RESTORED"); future re-banks go through t31-rebank |
| gsq-residency-warmth.sh | DONE 16:37 (gsq-residency-warmth.log) | warmth curve FLAT, theory dead (eb2155ec "residency warmth curve (flat, theory dead)") |
| gsq-v6-verdict.sh | DONE 19:10 (gsq-v6-verdict.log) | v6 stands as the standing governor (0b8917d5); one-off with a hardcoded tonight-range baseline (:3) — not re-runnable as-is |
| gsq-warm-verdict.sh | DONE 14:51 (gsq-warm-verdict.log) | superseded same-day by the corrected instrument (gsq-final.sh/gsq-restore.sh) |
| kernel-aa.sh | never ran as written (kernel is 7.2.15) | see §1.1 — the ONE owned script reclassified current |
| stock-jury.sh | DONE 15:04 (stock-jury.log) | stock-tree control for the GSQ era concluded; the standing A/B instrument is t22-custom-vs-stock.sh. Note: its kill line is repo-blind (§3 Variant C) |
| ud-warm-verdict.sh | INTERRUPTED (ud-warm-iq2xs.log stops at pack-1 probe 1; ud-warm-iq4xs.log at probe 1; no DONE) | superseded by ud-restore.sh (lib-based, wait_prewarm per probe, 4 boots each) |

### 1.3 ERA-CONCLUDED — already lib-based (lint-clean; archive at leisure, no urgency)

- GSQ corrected-instrument era: gsq-bisect.sh, gsq-final.sh, gsq-restore.sh, ud-restore.sh (banks landed: lanes.csv 003d78d8, 35653c90, 9407b69a)
- Hostmem era (campaign concluded — results/hostmem-leg0-verdict.md, results/hostmem-day1-verdict.md): hostmem-leg0.sh, hostmem-leg1.sh, hostmem-leg2.sh, hostmem-definitive.sh, hostmem-gsq-clean.sh
- Tranche era (questions closed by their verdict commits): t32-tranche.sh, t32b-ctl-mimic.sh, t32b-dm-fiveboot.sh, t32b-dm-gates.sh, t32b-mimpair.sh, t34-elastic.sh, t36-battery.sh, t38-binary-counterfactual.sh, t38-sanity.sh, t38-tiebreak.sh, t38-abab-b.sh, t39-template-gates.sh, t39-tpl-mimpair.sh, t40-verify.sh, t40b-verify.sh, t41-battery.sh, t41-prob-gates.sh, t42-rebank.sh, t42b-counterfactual.sh, t49-ccdsplit.sh, t50-dt-sweep.sh, t63-battery.sh, t63-gate.sh (30a7ef7b: tail-skip rank 7 ADOPTED), t64-battery.sh (DS flat-routing priced out, 76110f4e), tpl-speed-ab.sh, tpl-user-gates.sh, tpl-user-mimpair.sh
- Summarizers of closed eras: t32-summary.py (Oct 22), t32b-summary.py (Oct 23); t31-summary.py STAYS (paired with t31-rebank per README)
- stale `__pycache__/` (Oct 22) — deletable at the next cleanup pass

### 1.4 DEAD

None outright. The two scripts whose preconditions vanished (era-interleave, gsq-condring-ab — binaries pruned) are era-concluded evidence, not dead: both archive with their verdict commits as the reason.

## 2. battery-lib.sh catalog (campaign/battery-lib.sh — the shared lib, sourced not executed)

| provides | site | contract |
|---|---|---|
| `port_pid` | :30 | pid of the $PORT listener via ss -ltnp |
| `cleanup_engine` | :32-51 | port-scoped TERM first (:38, the server closes its engine in order); GPU sweep scoped to THIS repo's engine binaries only — `*"$REPO/engine/"*|*"$REPO/build-fresh/"*` (:40-43), never a bare "strata" substring (the incident the lib was built from, header :6-8); 30 s port-free wait (:46-50) |
| `wait_health [tries x3s]` | :53-60 | any /health body = the API answers |
| `wait_prewarm [tries x5s]` | :62-73 | the law-11 gate: terminal states done/failed/yielded/absent (campaign law 11 "warmup is part of the bench", header :63) |
| `final_serve <launcher> <out>` | :75-78 | leave the standing server up for the operator |
| `STRATA_DROP_CACHE_ON_EXIT=1` | :23-28 | the 2026-11-09 bootstorm fix (kswapd +87.7 GiB pgsteal measured); exported only by lib-sourcing batteries |

Does NOT provide: a boot-leg helper (nohup + health + prewarm), a probe loop, witness greps, an engines-down gate, a binary-swap helper, a dry-run mode. The converted template already exists in-tree: t38-abab-b.sh (:10 source, :19-20 cleanup+port-busy guard, :33-34 wait_prewarm+probe, :37 cleanup, :42-47 restore + final serve).

## 3. The 14 owned kill-sweeps — exact patterns, scope, self-match

KILL_SIG = `pkill\b|nvidia-smi\s+--query-compute-apps` (tools/lint_tree.py:55), counted per LINE (:139-141). The 14 JSON baselines sum to 34 and match this census exactly.

### Variant A — exact-cmdline pkill + name-kill escalation (11 scripts, 32 sites)

```bash
pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
sleep 2; pkill -9 -x strata 2>/dev/null; sleep 5        # escalation deltas noted below
```

Sites (boot-fail line, stop line, escalation line):
- era-interleave.sh:19,27,28 · gsq-arms-screen.sh:16,23,24 · gsq-condring-ab.sh:21,29,30 · gsq-final-bisect.sh:16,24,25 · gsq-pinning-ab.sh:14,22,23 · gsq-quiet-record-hour.sh:15,22,23 · gsq-v6-verdict.sh:13,21,22 · kernel-aa.sh:16,24,25 (all `-9 -x` escalations)
- gsq-frac-sweep.sh:42,43 · gsq-warm-verdict.sh:21,22 (plain `-x`, no -9)
- gsq-residency-warmth.sh:21,22 (`-9 -x`, no trailing sleep)

**Scope verdict:** the -f pattern is port-scoped IN EFFECT — that exact argv is carried only by the 8081 IQ3_S server. The `-x strata` escalations are NOT port-scoped: name-exact match kills every process named `strata` box-wide — the OTHER lane's engine, a parity-bench strata binary, an interactive strata CLI; `-9` skips the TERM-first order cleanup_engine keeps (:33-38).

**Self-match class:** `-f` matches full cmdlines — any wrapping `bash -c '<the pattern>'`, `grep/tail <the pattern>`, or `vi +/<the pattern>` process self-matches and dies. As written (script files executed by name) no script's own argv contains the pattern, so none self-matches today; the class stays one wrapper away. `-x` cannot self-match but is cross-lane.

### Variant B — UD config argv (1 script, 2 sites)

```bash
pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-ud" 2>/dev/null
sleep 2; pkill -x strata 2>/dev/null; sleep 5
```
- ud-warm-verdict.sh:23,24 — same class as A with the custom-ud argv.

### Variant C — repo-blind / port-blind broad sweeps (2 scripts, 2 lines — the RISKIEST owned patterns)

- **ds-verify.sh:71**: `pkill -f 'serve/server\.py' 2>/dev/null; pkill -x strata 2>/dev/null; pkill -f '/build-fresh/strata' 2>/dev/null`
  - `'serve/server\.py'` matches ANY checkout's server (the standing 8081 custom server AND the stock tree's 8080 server — /home/svg/LLM/strata/serve/server.py exists) regardless of port or lane.
  - `'/build-fresh/strata'` is path-scoped but port-blind, and self-matches any operator tail/grep/less carrying that literal — the exact historical bug class.
- **stock-jury.sh:17**: `pkill -f "serve/server.py" 2>/dev/null; sleep 2; pkill -9 -x strata 2>/dev/null`
  - While aiming at the stock 8080 server it kills the standing 8081 custom server (any repo's server.py) and every strata-named process with SIGKILL, no TERM-first order.

### Mapping onto battery-lib

Every variant folds onto `cleanup_engine` (battery-lib.sh:32-51) with the already-parameterized PORT/REPO (battery-lib.sh:20-21):

- Variants A/B (and kernel-aa): lib defaults (PORT=8081, REPO=this checkout). The binary-swap scripts (era-interleave, gsq-condring-ab) copy their alt binaries to engine/strata before boot (:13, :15), so the :42 path sweep covers every binary they boot.
- ds-verify: lib defaults — the DS launcher (launch-custom-dsv41-8081.sh) lives in this repo, and :42 already matches `*"$REPO/build-fresh/"*`, the DS engine path its :71 third pattern was reaching for.
- stock-jury: `PORT=8080 REPO=/home/svg/LLM/strata` — the stock tree has no build-fresh/; /home/svg/LLM/strata/engine exists so :42 matches `*$REPO/engine/*`. Named residual: a stock-side GPU process whose cmdline carries neither repo path survives the sweep — the port TERM (:38) plus the 30 s port-free wait (:46-50) own the lane regardless.

**No new kill pattern is needed anywhere.** The lib already owns the mechanism; the 14 scripts duplicate only the boot/warm-gate/probe loop around it.

## 4. The consolidation proposal

### 4.1 CONVERT (2 scripts — the only still-current raw-kill members)

**ds-verify.sh** (NEEDS-BENCH for its trust leg; blocked on pool 45076/54272):
- source battery-lib (t38-abab-b.sh:10 idiom); replace :33's curl guard with `require_engines_down` (new, §4.3); replace :45-48's hand-rolled wait with `wait_health 300` (300 x 3 s = its HEALTH_TIMEOUT=900, :22); replace :71 with `cleanup_engine` (drops all three broad patterns); keep the pool guards and witness greps verbatim (:32, :54-68 — they are the experiment).

**kernel-aa.sh** (NEEDS-BENCH; one real leg requires the 7.2.12 reboot its :10 gate demands):
- source battery-lib; replace :14/:16's hand-rolled health loop with `wait_health`; replace the "page cache warm in" marker greps (:17) with `wait_prewarm` — the marker (serve/server.py:3843) fires at the files-sweep end BEFORE a replay terminal state (:3841-3845); on the standing config (config/custom-iq3_s-8081.json:83 `"prewarm": {}`) the two coincide, but wait_prewarm is the law-11 gate and the retired-marker no-op class is what battery-lib.sh:9-12 documents; replace :24-25 with `cleanup_engine`; probe loop → `probe3` (§4.3). The kernel gate (:10) stays verbatim.

Conversion side-effect, intended: both scripts' boots gain `STRATA_DROP_CACHE_ON_EXIT=1` (battery-lib.sh:28) — the 12 archived scripts ran their eras WITHOUT it (no lib source).

### 4.2 ARCHIVE unchanged → campaign/archive-2026-11/ (new dir)

- The 12 era-concluded owned scripts (§1.2), UNCHANGED — evidence integrity; refactoring dead instruments is migrate-on-touch spent on nothing. One-line README: era pointer to the lanes.csv rows and the verdict commits listed in §1.2.
- At leisure (no lint pressure — they are already clean): the §1.3 lib-based era scripts + t32/t32b summarizers, same archive with the same README pattern.
- Hygiene note before moving era outputs: results/stock-jury-boot.out is git-dirty with fresh (22:30 today) Chromium-viz stderr lines — confirm no live fd holder before archiving anything referencing it (results/ themselves are not part of this move).

### 4.3 battery-lib gains (~35 lines total; only 1+2+4 are load-bearing)

1. `require_engines_down` — the t38-abab-b.sh:20 guard (`[ -n "$(port_pid)" ] && abort`) as a lib gate; every converted script opens with it.
2. `boot_leg <tag> <launcher> <out>` — nohup boot + `wait_health` + `wait_prewarm`; returns ok. Retires the 100x3 s curl loops and the marker greps in one place.
3. `probe3 <label>` — the 3x `probe-lane.py --port $PORT --label <label>-$i --short-n 6` loop (the byte-identical block in all 14).
4. `STRATA_BATTERY_DRYRUN=1` — boot_leg/probe3/cleanup_engine print intent and no-op; the gates (require_engines_down, kernel gate, pool guards) still execute. This is what makes the dry-run protocol (§5) a defined mode instead of a manual trace.
5. Nice-to-have, not owed by this consolidation: `witness_boot <out> <log-offset>` (the complement/memfd/ARENA greps), `swap_binary <keep> <alt>` (the era-interleave/t38 pattern).

### 4.4 Lint coordination — the binding constraint (SAFE-NOW mechanics, NEEDS owner sign-off)

- `RECORDS` (tools/lint_tree.py:71) is `campaign/(results|research)/|SVG-CUSTOM|\.docs/|docs/|bench/results/` — `campaign/archive-2026-11/` is NOT skipped, so archived scripts keep firing the kill rule today. Extend to `campaign/(results|research|archive-*)/` — consistent with the never-scanned-records principle (lint_tree.py:26-28).
- Then retire the 13 era JSON entries (tools/lint_exceptions.json `kill`) in the SAME commit as the archive move (storage canon: reference updates land in the same commit). The file's own precedent is a `retired:*` section (`retired:prewarm2` exists); either delete outright or move with one reason line "era-concluded, archived unchanged to campaign/archive-2026-11/".
- Keep ds-verify/kernel-aa entries until their conversions land; then zero them (a conversion that keeps any pkill line is a failed conversion — the whole point is baseline 0).
- t22-custom-vs-stock.sh stays a named in-code exception (lint_tree.py:48) — untouched.

## 5. Validation protocol (per converted script, in order)

1. **Syntax gate** (SAFE-NOW): `bash -n campaign/batteries/<script>.sh`.
2. **Dry-run, engines down** (the default state — verified: no 8080/8081 listeners right now): with `STRATA_BATTERY_DRYRUN=1` — `require_engines_down` passes (port free), boot_leg/probe3 print intent, cleanup_engine no-ops (no listener to TERM; GPU sweep lists nothing); exits 0 with the full leg schedule in the log. ds-verify additionally proves its pool refusal fires — it dies at the :32 guard (pool 45076/54272 < 54272); THAT refusal is the gate firing, not a failure. kernel-aa's dry-run dies at its :10 kernel gate on 7.2.15 — same rule.
3. **One real leg before bank-trust** (NEEDS-BENCH, needs an engines-up window): run leg/ctl 1 only; verify `pgrep -x strata` is empty after the leg's stop, the port frees inside 30 s, probes and witnesses landed in OUT.
4. **Lint green** (SAFE-NOW once 4.4 lands): `python3 tools/lint_tree.py` exits 0 with the updated RECORDS and the retired JSON entries, same commit as the move.

Bank-trust rule: a converted script enters lanes.csv-eligible service only after step 3. Step 2 alone proves structure, never behavior.

## 6. SAFE-NOW vs NEEDS-BENCH

| item | status |
|---|---|
| Fleet inventory, kill census, scope/self-match analysis, lib mapping | SAFE-NOW (this digest; read-only evidence) |
| Archive dir + move of the 12 era-owned scripts + README | SAFE-NOW mechanically; NEEDS owner sign-off (touching campaign structure) |
| RECORDS regex extension + JSON retirement, same commit | SAFE-NOW mechanically; NEEDS owner sign-off (edits tools/lint_tree.py + lint_exceptions.json) |
| battery-lib gains 1, 2, 4 (require_engines_down, boot_leg, DRYRUN) | SAFE-NOW to write (lib is kill-owner, lint-clean by construction); NEEDS-BENCH for boot_leg's first real boot |
| ds-verify conversion | write SAFE-NOW; run NEEDS-BENCH (blocked on pool top-up — its own :32 guard refuses today) |
| kernel-aa conversion | write SAFE-NOW; run NEEDS-BENCH (blocked on the 7.2.12 reboot its :10 gate demands) |
| Step-3 trust legs for either conversion | NEEDS-BENCH |
