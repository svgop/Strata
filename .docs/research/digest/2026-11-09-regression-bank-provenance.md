---
type: digest
date: 2026-11-09
title: Regression-bank provenance — is 229.1/224.8/303.5 the right baseline for today's 175-195 solo?
claim_under_test: "GSQ IQ3_S solo/pair/conc = 229.1/224.8/303.5 (the Nov-6 standing bank) vs today's 175-195 solo = a regression"
verdict: NOT A VALID COMPARISON — the bank is a cross-row composite of two 2-leg spot means, never banked kind=standing; same-binary A/A re-banks in this CSV already land -7..-17% under their banks; today's boots run a different (adopted) memory architecture mid re-bank.
evidence: campaign/tracker/lanes.csv rows 123,129-131; commits 083a68a7, 4152f114, 0704fc1e, 9cedd332, e8484c36, f95455a6; campaign/results/hostmem-leg0-verdict.md, hostmem-day1-verdict.md, hostmem-gsq-clean.log, gsq-restore.log
---

# Provenance of the "Nov-6 standing bank" (229.1 / 224.8 / 303.5)

## 1. Row trace — the claimed bank is a composite of TWO different boots

The trio does not exist in any single row of `campaign/tracker/lanes.csv`:

| lane | value | CSV line | source row | commit | boot time | note says |
|---|---|---|---|---|---|---|
| solo | 229.1 | line 123 | "grv3+ud-ports spot (mask 133 on: **2-leg means**; off-control conc 283.6 = +6.3%; every lane at-or-above the prior means - GSQ not slower)" | 1553a614 | ~03:00 Nov-6 | kind=**mean**, "spot" |
| pair | 224.8 | line 130 | "smalls+conversion GSQ spot (mmvf router fast + grid-stride copy + the UD bf16-dense gate fix; quiet box; **2-leg means** vs the prior 229.1/220.7/301.6 = +1.7/+1.9/+0.6)" | HEAD (4152f114 era) | ~11:30 Nov-6 | kind=**mean**, "spot" |
| conc | 303.5 | line 131 | same smalls+conversion row | HEAD | ~11:30 Nov-6 | kind=**mean**, "spot" |

Facts established by the trace:

- **The trio mixes two boots run ~8 hours apart on two different binaries** (1553a614 vs HEAD-of-Nov-6, which added the smalls/conversion code). The honest same-boot trios are 229.1/**220.7/301.6** (grv3 spot) and **233.1**/224.8/303.5 (smalls spot). The claimed bank takes the lower solo from boot A and the higher pair/conc from boot B.
- **Neither row is a standing bank.** Both are kind=mean, labeled "spot", **2-leg means** (2 boots each). The last qwen point banked kind=standing is **Nov-2** (W1 adaptive-spec, 232.2/222.3/315.4, fcc130d5). The last row labeled "THE bank" is **Oct-27 F-367** (222.0/233.0/350.8, 50684ddd). "Nov-6 standing bank" is not a state the ledger ever recorded.
- **Leg recipe:** 2 boots x `campaign/probe-lane.py --short-n 6` (defaults: warmup "Say OK.", 6 A-shorts + fresh 5.8K-prompt + 6 B-shorts, M1 single-stream 2000 tok, M2 pair 2x900 tok aggregate, M2 6-conc 6x600 tok aggregate), ctx=0 (short battery). Not a 3-boot or 5-boot mean — the CSV's own protocol prefers five-boot means ("kind — mean (multi-boot/leg banked mean)"; the October banks were five-boot).
- **Raw leg files for both Nov-6 spots were never preserved.** The banking commits touch only tracker files (083a68a7: lanes.csv/json/html only; 4152f114: code + mmlu json + tracker). No file in `campaign/results/` carries a `=== PROBE` line dated Nov-6. Provenance for the numbers = commit messages + CSV rows only. Named gap: the Nov-6 spots are unauditable at leg level.
- The engine log read by the probe (`logs/custom-iq3_s.log`) is the same file today's probes read — the accounting side is shared. The probe invocation is identical (`--port 8081 --label <tag> --short-n 6`).

## 2. Instrument check — same probe, unchanged since Oct-22

`git log -- campaign/probe-lane.py` (followed through its Nov-3 rename): exactly three commits ever:

- 490079e4 Oct-13 (creation), ddcc0e93 Oct-22 (`--conc-n` added + methodology law), a60e6f29 Nov-3 (**pure rename**, numstat 1/1 — content identical).

So M1/pair/conc definitions, prompt sizes, warmup, A/B short classes, token budgets are **bit-identical across the entire Oct-17..Nov-9 table**. "M1 single-stream" never meant anything else. Today's numbers (gsq-clean r1-r3, `campaign/probe-lane.py --port 8081 --short-n 6`, fresh boot per leg via `launch-custom-iq3s-8081.sh`) are **the same instrument, same recipe** as every row. The difference is conditions, not instrument.

## 3. Day-by-day table — qwen GSQ IQ3_S, Oct-27..Nov-9 (all rows + the un-CSV'd re-banks)

| date | point (kind) | solo | pair | conc | conditions named in the note |
|---|---|---|---|---|---|
| Oct-27 | F-364 quiet-box post-reboot (mean) | 223.0 | 213.9 | 324.8 | quiet box |
| Oct-27 | F-367 "THE bank" (mean) | 222.0 | 233.0 | 350.8 | shm shadow store; pair> solo anomaly, never reproduced after |
| Oct-29 | T59 262K shadow-on (standing) | 207.4 | 198.4 | 294.1 | = A/A re-bank: **-6.6/-14.9/-16.2% vs F-367**, zero code change on lanes |
| Nov-2 | W1 matched control (mean) | 230.2 | 223.1 | 306.3 | champion binary + tmpfs shadow, 2-leg, quiet |
| Nov-2 | W1 adaptive-spec (standing) | 232.2 | 222.3 | 315.4 | battery-neutral vs ctl |
| Nov-2 | T63 control (mean) | 197.3 | 197.8 | 292.1 | "box 2955MHz + 3GB swap - **not bankable**" |
| Nov-2 | T63 tail-skip adopted (mean) | 217.5 | 206.5 | 290.5 | degraded box, "clean re-bank owed" |
| Nov-4 | serve-fix OLD (mean) | 207.5 | 206.7 | 277.7 | 2v2 same-hour |
| Nov-4 | serve-fix NEW (mean) | 215.7 | 201.5 | 259.3 | 2v2 same-hour |
| Nov-4 | crash-fix spot (mean) | 208.6 | 212.6 | 251.2 | |
| Nov-5 | geometry-gate spot (mean) | 205.2 | 197.8 | 260.6 | morning |
| Nov-5 | handshake spot, 3-leg (mean) | 210.5 | 192.7 | 250.9 | same-morning box |
| Nov-5 | bw-dense port, 2-leg (mean) | 224.0 | 216.7 | 295.3 | "+6.4/+12.4/+17.7 vs this mornings same-box pre-port" |
| Nov-6 | grv3+ud-ports spot, 2-leg (mean) | **229.1** | 220.7 | 301.6 | quiet box |
| Nov-6 | smalls+conversion spot, 2-leg (mean) | 233.1 | **224.8** | **303.5** | "quiet box" — the campaign's highest quiet-window trio |
| Nov-7 | hostmem leg-0 A/A re-bank, 3-boot (verdict file, not in CSV) | 206.3 | 186.7 | 258.9 | "aged box, 6 days uptime, zero env changes" = **-9.9/-16.9/-14.7% vs the Nov-6 composite** |
| Nov-7 | hostmem leg-1 GSQ control, 3-boot | 171.2 | 164.7 | 224.3 | partial-pool hold, "control tax -12..-17%" (day-1 verdict) |
| Nov-7 | hostmem leg-3 GSQ complement armed, 2+2 | 202.6 arm vs 189.0 ctl | 190.5/190.0 | 248.7/260.2 | **arm solo +7.2% same-window** — the architecture adopted |
| Nov-8 | hostmem definitive, 3-boot | 146.8 | 154.4 | 195.0 | **the budget bug** (e8484c36): complement CAPPED 19.80/37.97 GiB, "solo 120-165 t/s" — the REAL post-reboot regression, since fixed |
| Nov-9 | gsq-clean, 3-boot, 10:25-10:32 | 175.7/178.4/195.1 (mean **183.1**) | 156.1/182.8/177.0 (**172.0**) | 221.2/233.7/228.9 (**227.9**) | post-budget-fix, pre-single-owner-store (still double-held the 31.6 GiB tmpfs tier) |
| Nov-9 | gsq-restore, 5 boots banked on LAST 3 — **mid-flight at 10:43** | r1 75.3 (warm leg by design) | r1 165.0 | r1 239.0 | post-f95455a6 single-owner store; complement witness "37.85 GiB of 37.85 GiB in 2 MiB pages (HUGETLB POOL)" |

Shape of the curve: **Nov-6 is not an outlier peak** — it sits +1.3/+0.8/-0.9% on the Nov-2 quiet-window matched control (230.2/223.1/306.3). It is the top of a bimodal band: quiet-box days read 230-233 solo; degraded/aged-box days read 197-216 (Nov-2 T63, Nov-4, Nov-5); the two modes differ by ~10-15% and the CSV itself labels the low-mode days "not bankable". The Nov-6 bank is a **top-of-band, 2-leg, un-bankable-by-protocol spot** from the favorable mode.

## 4. A/A variance from the CSV's own history — the expected honest gap

Same-binary, no-lane-change re-banks against their own banks:

| A/A instance | bank vs re-bank | gap |
|---|---|---|
| T59 Oct-29 vs F-367 Oct-27 | 207.4/198.4/294.1 vs 222.0/233.0/350.8 | **-6.6 / -14.9 / -16.2%** |
| hostmem leg-0 Nov-7 vs Nov-6 composite | 206.3/186.7/258.9 vs 229.1/224.8/303.5 | **-9.9 / -16.9 / -14.7%** (`hostmem-leg0-verdict.md`: "-10% / -17% / -15%") |
| W1 ctl Nov-2 vs smalls spot Nov-6 | 230.2/223.1/306.3 vs 233.1/224.8/303.5 | -1.2 / -0.8 / +0.9% (quiet-to-quiet) |
| grv3 vs smalls spot, same day | 229.1/220.7/301.6 vs 233.1/224.8/303.5 | -1.7 / -1.8 / -0.6% |

Family law: quiet-window re-banks repeat within ~2%; **aged/degraded-window re-banks land -7..-17% under their banks**. The -8..-15% band is therefore **in-family**, measured twice on this exact CSV. The reply ladder in the tracker README (law 12) already prescribes this: a low read triggers same-hour control → five-boot means → binary-ladder, and the bank stands until that ladder runs.

## 5. Verdict

1. **Today's numbers ARE measured the same way** (same probe-lane.py, byte-identical since Oct-22; same launcher, same short battery, fresh boot per leg). The probe is not the difference.
2. **The baseline is not what the claim says.** "Nov-6 standing bank" is a composite: solo from a 2-leg spot on binary 1553a614, pair/conc from a different 2-leg spot ~8 h later on a different binary, both from the campaign's most favorable quiet window, kind=mean/spot (not standing), 2 legs (below the five-boot protocol), raw legs not preserved.
3. **-8..-15% is not out-of-family for same-binary re-banks** — two independent A/A instances in the record land -7..-17% (T59, hostmem leg-0). Against the in-family same-binary anchor (leg-0's 206.3/186.7/258.9), today's clean means 183.1/172.0/227.9 read **-11.3 / -7.9 / -12.0%** — inside the A/A band.
4. **Today's deeper vs-bank read (-20..-25%) has named, adopted causes, not a code regression:** (a) the comparison baseline is the inflated composite; (b) the box re-booted Nov-8 12:22 into the adopted hugepage architecture (45076-page 2 MiB pool armed now, complement on HUGETLB POOL) whose own same-window A/B priced the arena at **+7.2% solo**, so the architecture is not a -20% mechanism; (c) the genuinely real post-reboot drop (solo 120-165) was the pool-blind budget capping the complement at 19.80/37.97 GiB — found and fixed today (e8484c36, 10:22); (d) the gsq-clean legs ran before f95455a6 (10:38) still double-holding the 31.6 GiB tmpfs tier ("dead resident weight, page-cache squeeze" per the commit); the banked restore battery (5 boots, last-3) that removes it is mid-flight.
5. **Required before any regression call:** let gsq-restore bank its last 3 boots, then run the law-12 ladder (five-boot means against leg-0's 206.3/186.7/258.9 as the honest A/A anchor, not against the Nov-6 composite). A true failure signal is the restore bank also reading below ~190 solo with the complement witness green — that would implicate the new store rules, and the rollback switches are named per commit (STRATA_HUGEPAGE_ARENAS=0, STRATA_SHADOW_MEMFD=0).

Gaps named: (1) raw probe legs for both Nov-6 spots were never saved — the numbers are unauditable beyond commit messages; (2) `.docs/log.txt` (required project record) is absent from this repo; (3) the CSV carries no rows after Nov-6 — the Nov-7..9 re-banks live only in results files and verdicts, so the tracker's own incline-law check cannot run yet.
