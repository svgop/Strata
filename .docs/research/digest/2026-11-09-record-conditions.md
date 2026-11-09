---
type: research-digest
id: 2026-11-09-record-conditions
date: 2026-11-09
scope: every environmental difference between the Nov-6 morning/midday that produced GSQ 233.1/224.8/303.5 and the Nov-9 released-box reading 195.8/186.5/272.7 (era binary 4152f114 rebuilt = 193.7/185.4/269.7, era-interleave.log 15:20-15:36)
method: read-only audit — journalctl (strata-gpu-lock unit + kernel + boots), logs/custom-iq3_s.log boot-segment forensics (662k lines, 292 boots), git 4152f114..HEAD, /var/log/dnf5.log, rpm install epochs, live /proc + ps. No engine booted, no GPU state written.
evidence-anchors: campaign/results/era-interleave.log (verified leg-by-leg; era mean 193.7/185.4/269.7, current 195.8/186.5/272.7 — exact), logs/custom-iq3_s.log lines 574475/579766/585349 (bank boots) vs 716918-729603 (era legs)
---

# Record-conditions audit: what the 233.1 spot had that today's released box does not — 2026-11-09

## Verdict (read this first)

1. **The era A/A already isolates the question**: the 4152f114 binary and HEAD are equal today (193.7 vs 195.8 solo, inside noise). Code exonerated by experiment. Everything below is the environmental ledger.
2. **The residency premise of the hunt is FALSE.** The bank spot legs were NOT long-lived boots: the 233.1 trio came from boots 10:33:48 and 11:51:25 (log lines 579766, 585349) probed ~60-120 s after engine start — the same fresh-boot class as today (era1: start 15:19:44, probe 1 at 15:21:18). The 55 min / 4 h24 m residencies (13:11 and 14:40 boots) belong to the Nov-6 AFTERNOON — the day's worst windows (80.5-93 t/s at 18:44, then the 18:41:30 OOM kill). Residency warmth: **zero differential**.
3. **Warm-in curves are byte-for-byte the same class**: bank boot 1 hit ramp 93.5→97.5→97.7%, first-request 82.1 GiB storage read, decode waited 2346 ms; bank boot 2: 93.2→97.1→98.0→98.6%, 76.4 GiB, 1982 ms; era1 today: 93.4→97.4→97.8→98.4%, 58.3 GiB, 2131 ms. Both eras COLD at first request, steady by request 3.
4. **Clock regime: the bank ran under v2's hard pin (journal-proven per leg) and today's released legs run v4 range — sustained state equal, entry state BETTER today, and the only mem delta in measured windows is 13801-vs-14001 = ≤0.8% of a 12 ms GPU-reach.** The measured -30..-38% clock fault today was the v5 mem-only pin (SM held 2004 MHz), fixed at 13:32 (MEM_PIN=0), before the interleave. Pins are the hazard; the float is not.
5. **Two environmental variables are new today and NOT yet A/A'd against the bank era: (a) the governor regime v2→v3/v4 — cutover Nov-07 15:23, exactly the window the +2.0-2.2 ms/window in-run deficit appeared (Nov-7 12:17→15:30 boots, journal-proven), persisted across every reboot since because v4 is what boots; (b) kernel 7.2.12 → 7.2.15, installed today 14:39:31 SAST, rebooted in at ~14:41 — the entire released-box afternoon (incl. the era interleave) is the first bank-class measurement on it.**
6. **Everything else favors today or is a proven zero**: MemAvailable at guard 137.4 GiB (era legs) vs 113.0/122.3 (bank boots); fresh pristine buddy vs Nov-6's 4d17h degraded one; zram 3.8 GiB vs day-4.7 fill; store tier IDENTICAL (/dev/shm tmpfs both eras on these legs — the released box falls back to the Nov-6 ladder by the f95455a6 rule); engine command lines byte-identical; desktop journal activity LOWER today (chrome lines: 113 in the bank window vs 0 in the interleave hour) though today's desktop holds 0.35 GiB more VRAM (slots 9863-9980 vs 10020/10036).
7. **What remains, ranked: the quiet-window selection (the bank's own mode, +10-15% in-record), then the governor regime (timing-matched, unpriced), then the kernel (dated today, unpriced).** Reproduction matrix in §7.

Confidence: A for every quoted line (journal/log/git/rpm, this box); the ranking in §6 is inference from A-evidence plus the sibling digests (bank-provenance, clock-box, bench-flow, residency, memory-state).

---

## 1. Clock regime — Nov-6 bank window vs today (Task 1)

### 1.1 What the bank legs ran under (journal, `journalctl -t strata-gpu-lock`)

Every bank-window probe sat inside a v2 hard-pin span — `GPU processing — clocks max (3135/14001)`:

| bank event | v2 journal span | pin held through |
|---|---|---|
| 10:33:48 boot, probe 1 | processing 10:34:42 → quiet 10:35:27, re-pin 10:37:07 → 10:39:12 | M1/pair/conc of leg 1 |
| 11:51:25 boot, probe 2 | detected 11:52:04, processing 11:52:20 → gone 11:55:15 | the whole 3-probe block |
| night spot 03:05-03:09 (the 229.1 row) | same v2 state machine (03:0x spans, boot→probe ~60 s) | both mask-on legs |

Sustained state inside the pin: SM power-trimmed ~2830-2955 (575 W cap, pviol=1; the Nov-7 dmon proof), **mem hard 14001 through every burst**. Entry from gaps: v2 paid 3 s at 1800/7001 (the 1780-plateau defect, clockwatch-proven) — the bank carried that handicap and still posted 233.

### 1.2 What today's probes run under

- Released legs (era-interleave 15:20-15:36): v4 range — `engine listening — clock floor 1800-3135 sm / mem range 7001-14001` (journal 15:20:18, 15:24:41). Sustained: SM 2876 mean / 2962 peak at 211 W sampled live 13:32 post-fix ("the banked class"); mem 13801 under load (Nov-7 dmon grade A), 7001 only at quiet.
- **The v5 falsifier verdict (pre-registered rule, ≥+8%): REJECTED — and inverted.** The mem-only hard pin (`-lmc 14001,14001`, SM left as range) held **SM at ~2004 MHz during decode at 61 W — boost never engaged** (fork-ud-first-bank.md §incident-3). That is the v2-class -30% anatomy. `MEM_PIN=0` at 13:32:12 (journal restart line) restored 2876/2962 instantly. Operational law: **never pin mem without pinning SM to max in the same call** — v2's paired pin (3135/14001) sustained 2830-2955; the mem-only pin collapses SM on this driver/card.

### 1.3 Quantified: MEM 14001 vs 7001-13801 float on a 12 ms GPU-reach

Decode time splits ~half SM-sensitive / half mem-bound (the v2 -30% decomposition, governance digest §5.3; full BW 1792 GB/s @14001, 1766 @13801, 896 @7001).

| case | arithmetic | window effect | lane effect |
|---|---|---|---|
| Sustained float 13801 vs pin 14001 | mem half ×(14001/13801)=×1.0145 → +0.09 ms on 6 ms | 12.00 → 12.09 ms | **≤0.8%** — sub-noise (A/A repeats ±2%) |
| Burst entry at 7001, whole 12 ms window at floor | mem half ×2 → +6 ms | 12 → 18 ms | -33% on THAT window only |
| Same, ramp τ≈1 s (dmon 1 s samples already 13801 in-burst) inside a 10 s M1 | ~1/10 of leg at partial bins, ~-30% mem-half each | +0.09 ms amortized | **≈ -1%** |
| 220-tok shorts inside the first-second ramp | fully exposed | bound -33% each | **not observed** — shorts read 190-240 t/s today vs 200-240 bank era; the bound never materializes because legs enter on the warmup request, not cold |

Conclusion: v4's mem float prices **≤1.5% worst-case, ~0.8% typical** — it cannot carry -15..-25%. The direction is also wrong: v4 entry is better than v2's, which the bank overcame.

## 2. Residency warmth — the long-boot premise fails on the record (Task 2)

- Boot-marker forensics (log lines 574475-605117): night spot = boots 03:05:13 / 03:07:45 / 03:09:50 (2.5-min cadence, probe-per-boot); midday spot = boots 10:33:48 and 11:51:25 (the 11:37:30 boot crashed at token 0). **No bank leg exceeded ~4 min of engine residency at first probe.** The 55 min (13:11) and 4 h04 m (14:40) boots carried the afternoon traffic — the -65% window and the OOM kill — not the record.
- Warm-in curves (first 4 requests per boot): bank 93.5→97.5→97.7→98.x and 93.2→97.1→98.0→98.6; era1 today 93.4→97.4→97.8→98.4. First-request storage reads 82.1 / 76.4 (bank) vs 58.3 GiB (era1 — the F3 pre-boot sweep makes today warmer, not colder). Adaptive-tier exchange counts over the same requests: bank 149→259→334→336; era1 118→212→289→326 — same shape.
- Adaptive expert tier: rounds/swaps per run same class both eras (the "140-134 rounds, 5178-5982 swapped" figures in the hunt brief are the CURRENT session's sustained lines — today's log tail; bank-era legs show the same anatomy with 53-138 rounds, 3632-5982 swaps). Learned profile: written every 30 windows + exit, **never loaded by any of the 292 boots** (all carry `--expert-profile data/expert-profile.bin`) — drift is a save-only sink, structurally unable to move a boot (residency digest §3).
- Drafter/KV/prefix/slot elasticity: identical args both eras (`--spec 6 --kv int8 --kv-resident 32768 --mtp rt`, `STRATA_ELASTIC_SESSIONS` 6×0.97 GiB, 2 mapped); spec accept distributions overlap (A/A same-window). GPU cache slots 10020/10036 (bank) vs 9980 (era1) / 9863-9903 (cur) — the -0.35 GiB desktop-VRAM eat since the Nov-8 reboot (vram-deficit.md), priced ≤0.5% lane.
- **Named gap: no +5/+30/+60 probe-leg series exists on any long boot in the record** (Nov-2..6 results have no such cadence; the 14:40 boot's traffic was ad-hoc chat ending 62-112 t/s in the zram/OOM window). The warm-residency question is unanswerable from history — and moot for the bank comparison, which the markers prove was fresh-boot.

## 3. Config — nothing on the decode path changed (Task 3)

- Engine command lines (log markers): Nov-6 10:33:48 boot ≡ era1 (4152f114, today) ≡ cur1 (HEAD, today) — **byte-identical** (`diff` empty): `--max-context 262144 --batch 6 --vision --spec 6 --kv int8 --kv-resident 32768 --mtp rt --resident-experts --ple-io mmap --expert-profile-save-every 30 …`. Batch/spec/mtp/ctx identical.
- Launcher `launch-custom-iq3s-8081.sh`, 4152f114..HEAD (4 commits, all 2026-11-07/09): (1) `STRATA_HUGEPAGE_ARENAS=1` export (df1a9c90) — at pool=0 this takes the THP-madvise path, the same class Nov-6 ran; (2) single-owner store rule (f95455a6): `HugePages_Free >= 17000` → engine memfd from pack — **on the released box (Free=0) this branch is FALSE and the launcher falls back to the exact Nov-6 ladder** (tmpfs `/dev/shm/strata-shadow-q2_0`, confirmed in the era1 guard/shadow lines); (3) shm-shadow.sh call gated by the same condition (runs at pool=0); (4) tier-0 removal (5447b365). Net on the released box: **launcher behavior ≡ Nov-6**.
- `config/custom-iq3_s-8081.json`: **zero diff** 4152f114..HEAD. `src/program/generate.cpp` (+193 lines): only `STRATA_SPEC_LOGISTIC/ENTABORT/SURVTOPK` — env-gated, **all default OFF** (No-Regression Law), plus depth telemetry. `serve/server.py` (+140): pre-boot prewarm sweep (F3), bounded pool-settle wait, slot-release crash fix (919b4db3) — boot orchestration, not decode defaults. And the era interleave closes the whole question experimentally: none of it moves the lanes.

## 4. Page cache / buddy / boot lineage (Task 4)

| item | Nov-6 bank (10:33) | Nov-9 released legs (15:20) |
|---|---|---|
| OS boot | boot -3: Nov-01 17:45 → Nov-08 00:05 (**4d16h48m uptime at probe**) | boot 0: started ~14:41 today (**~40 min at probe**) |
| kernel | 7.2.12-300.fc45 (installed Oct-25) | **7.2.15-300.fc45 — installed TODAY 14:39:31 SAST** (rpm install epoch; two reboots 14:37-14:41, incl. a 2-min 7.2.12 boot) |
| MemAvailable at engine guard | 113.0 GiB (10:33 boot) / 122.3 (11:51) | 137.4 GiB (era1 memory-guard line) |
| buddy / zram | 4.7-day degrade mid-flight (HOSTMEM: order-9 18→9→0 GiB across the week); zram filled by evening | pristine buddy; zram 3.8-3.9 GiB |
| hugepage pool | none (`nr_hugepages=0`) | **0 now** (released; the morning's 45,076-page pool drained before 12:49) |
| store tier | /dev/shm tmpfs 31.64 GiB | /dev/shm tmpfs 31.64 GiB (identical line) |
| era end | global OOM kill 18:41:30 (anon 26 GB + shmem 95.7 GB) | — |

Boot lineage (journalctl --list-boots): daily boots Oct-26..Nov-01, then ONE 6.5-day boot carrying the entire bank era, the Nov-7 degradation onset AND the Nov-7 hugepage arming; Nov-08 14:22 → Nov-09 14:37 (the morning batteries, pool armed); today's two reboots at 14:37-14:41 (kernel swap). **The 7.2.15 variable is dated today and shared by every released-box leg.** Journal clock labels after the reboot run ~9 min fast vs `date` — relative orderings unaffected.

## 5. Desktop (Task 5)

| window | chrome journal lines | steam | gnome-shell | note |
|---|---|---|---|---|
| Nov-06 10:30-12:00 (bank window) | 113 | 0 | 0 | the row's own "quiet box" tag; boot -1 era carried 30,111 chrome lines total; a Chrome-at-load leg once read **92.6 t/s = -55%** (fork-ud-first-bank.md) |
| Nov-09 15:00-16:00 (interleave) | 0 | 0 | 0 | chrome processes exist now (34, main 0.5 GB) but silent; load 1.44 |

Direction: the bank window carried MORE desktop journal activity than the interleave hour — desktop load is not today's deficit; the only measurable desktop delta is the 0.35 GiB VRAM baseline (§2 slots row).

## 6. Ranked environmental variables (expected lane impact)

| # | variable | status | expected effect on GSQ lanes |
|---|---|---|---|
| 1 | **Window selection / quiet-mode** (the bank is a top-of-band 2-leg spot; the record's quiet mode = 230-233/224/303-306, low mode = 197-216; same-day Nov-6 swing 80.5-93 → 228-274 on one binary) | proven in-record (bank-provenance §3-4, bench-flow) | **+10-15%** low-mode → quiet-mode; today's 195.8 sits at the low mode's bottom edge |
| 2 | **Governor regime v2→v4** — the +2.0-2.2 ms/window in-run deficit appeared exactly at the Nov-07 15:23 cutover and survives every reboot (v4 boots itself) | timing-matched suspect; sustained SM equal; mem float priced ≤0.8%; falsifier REJECTED the mem-pin mechanism | **0 to +15% if the v2 sustained-pin state (not the float) is the active ingredient** — unpriced until R2 runs |
| 3 | **Kernel 7.2.12 → 7.2.15** — installed 14:39 today; every released-box number (incl. era A/A) is on 7.2.15; the bank is 7.2.12 | dated today, never A/A'd across | **unknown, 0 assumed until R3** |
| 4 | Prewarm race / measurement pattern (fresh-boot per leg, no wait_prewarm, 2-leg spot means) | bench-flow: M1/pair/conc clean in 6/7 boots; real on shorts | **0 on banked lanes**, real on short-class rows |
| 5 | Memory state (buddy/zram/MemAvailable/store tier) | every axis equal or favors today | **0** (not today's deficit) |
| 6 | Desktop load | lower today | **0** |
| 7 | VRAM slot deficit -0.35 GiB (10020→9863-9980) | dated Nov-8 reboot | **≤0.5%** |
| 8 | Residency warmth | bank legs were fresh-boot too | **0 — premise false** |
| 9 | v5 mem-only pin state | REJECTED; fixed 13:32 | -30..-38% **while active** — keep MEM_PIN=0 permanently |

## 7. Reproduction matrix — ordered conditions to reach 233/225/304 again

| # | condition | test | expected |
|---|---|---|---|
| R1 | Quiet-window discipline | desktop parked (chrome/steam off), no concurrent chat, 5-boot means, `wait_prewarm` gated, law-12 ladder vs the leg-0 anchor 206.3/186.7/258.9 | +2..+10%; the record's quiet mode must reappear here or it is gone |
| R2 | Governor A/B on the released box, same boot interleave: v4 range vs **v2 emulation** (`-lgc 3135,3135` + `-lmc 14001,14001` while processing; 1800/7001 after 30 s quiet — both pins together, never mem alone) | capture `dmon -s pucvmet` + pair-anatomy (`CUDA0 GPU-reach wait`, rows/s) per arm | discriminator: pair gpu_wait 11.4-12.0 → ~9.2 and rows/s 164-185 → 203-207 = the regime owns the +2 ms; if equal, close clocks permanently |
| R3 | Kernel A/A: one released-box battery on 7.2.12 (previous-kernel boot entry, still installed) vs 7.2.15 | same R1 recipe, interleaved days | closes variable #3; any delta here also explains why Nov-7 morning (7.2.12, v2) still read 232-266 |
| R4 | VRAM baseline: park the desktop holders of the extra 0.35 GiB | slots line returns 10020-10036-class | +0.1..0.5%, free |
| R5 | Only if R1-R3 leave a residual: forensics on the Nov-7 12:17→15:30 onset window (GSP state, module params, firmware loads) | journal sweep + `nvidia-smi -q` diff vs a 7.2.12 boot | last resort |

**Reachability statement:** the A/A experiment proves current code = era code, so 233/225/304 is reachable only if the environment still supports ~275-280 t/s engine decode. R1 is proven to exist in the record (+10-15%); R2 is the only timing-matched unknown that big; R3 is free to test. Do not chase the number with mem pins — the one measured pin experiment of today cost -30..-38% and its rejection is banked (35653c90).

## Provenance

- Journal: `journalctl -t strata-gpu-lock` (Nov-06 10:20-12:10 bank spans quoted verbatim; Nov-09 12:50-16:40 incl. the HARD-PIN→range restart at 13:32:12 and the 14:42 boot banner), `journalctl -k -b -3/-2/-1/0` (kernel versions, boot boundaries), `journalctl --list-boots`.
- Engine log: `logs/custom-iq3_s.log` lines 574475-579766-585349 (bank boots: guard 113.0/122.3 GiB, shadow tmpfs line, pre-fill, complement 22/25 s, warm-in curves, adaptive exchanges), 716918+ (era1: guard 137.4 GiB, 9980 slots, warm-in curve), 601297-605117 (the 4h afternoon boot, late-boot 62-112 t/s), log tail (current session 140-134 rounds, 5178-5982 swaps).
- Era test: `campaign/results/era-interleave.log` (all 12 probes read; means verified by hand) + `campaign/batteries/era-interleave.sh` (released-box protocol); era binary authenticity: worktree `/home/svg/LLM/banked-era-4152` detached at 4152f114 (clean), binary built 15:19 today.
- Box: `/proc/meminfo` (HugePages 0, MemAvailable 69.7 GiB with engine resident), `ps` (load 1.44, engine 130.4 GB RSS), `/var/log/dnf5.log` (no lane transactions Nov-06→09; metadata refresh only), `rpm -q --last kernel kernel-core` (7.2.15 installed 14:39:31 SAST today; 7.2.12 Oct-25).
- Siblings: 2026-11-09-regression-{bank-provenance,clock-box,bench-flow,code-audit,memory-state,residency}.md, 2026-11-09-v5-memclock-governor.md, 2026-11-07-gpu-clock-governance.md, campaign/results/fork-ud-first-bank.md (incident §3).
