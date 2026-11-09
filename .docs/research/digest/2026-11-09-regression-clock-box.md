---
type: research-digest
id: 2026-11-09-regression-clock-box
date: 2026-11-09
scope: non-engine box state across the Nov-6 bank era and the Nov-8/9 probe era — GPU clock regime, governor state transitions, host memory, desktop load, thermals, and the boot-cycle ramp question
subject: strata-gpu-lock v2→v4 regime reconstruction from journal evidence; verdict on the GSQ IQ3_S solo/pair/conc regression (233.1/224.8/303.5 → ~176-195/~156-183/~221-234)
method: read-only local study — journalctl (unit + kernel), systemctl, nvidia-smi + dmon snapshots, live leg capture, campaign/results probes from both eras. No code changed, no engine booted, no GPU state written by this pass.
---

# Box-state audit: is the clock/system regime the regression? — 2026-11-09

## Verdict (read this first)

**The GPU/system clock regime is EXONERATED as the cause of the regression.** The regime
did change between eras (v2 hard-pin governor → v4 floor-range governor), but every delta
points in the direction that should make Nov-8+ probes FASTER or equal, never slower:

1. **Sustained probe-load clocks are identical across eras (~2830-2955 MHz, mem ~13.8-14 Gbps).**
   v2 hard-pinned SM 3135/mem 14001 during processing and the 575 W cap trimmed it to
   ~2830-2955 (Nov-7 dmon proof). v4's range lock (1800,3135) lets the hardware boost to
   the same power-capped ~2830-2850. The ceiling is decorative above ~2850 in both designs
   (`.docs/research/digest/2026-11-07-gpu-clock-governance.md` §3.3, grade A).
2. **v4 removes v2's measured 30% burst defect.** The Nov-6 banks ran UNDER v2's defect:
   after 30 s quiet the card hard-pinned SM 1800 + mem 7001 (half bandwidth), and bursts
   entering from a gap ran seconds at 1777-1792 MHz at 94-98% util until the 3 s poll
   re-pinned max (the clockwatch proof, measured Nov-7 against the v2 era). The 229/225/303
   bank was achieved WITH that handicap. v4's floor-range makes burst entry a microsecond
   DVFS step — if the governor moved the score, it moved it UP.
3. **Live capture today (this pass):** quiet-with-floor reads SM 2107-2497, mclk 7001-13801
   floating, 56-58 W, 37-38 C, pviol 0 — never the 210 idle, never the 1780 plateau. The
   floor is applied RIGHT NOW (engine up since 10:42:25, journal "clock floor 1800-3135").
4. **No thermal shed, no guardrail event in any Nov-8/9 probe window.** Zero `thermal shed`,
   zero `cooled to`, zero `guardrail:` lines from strata-gpu-lock across both boots; GPU Xid
   count in both kernels is 0 (only r8169 ethernet XID strings). GSP clean.
5. **The -8..-25% sits in the measurement-pattern delta + engine deltas, not the box:**
   the Nov-6 banks ran on LONG-LIVED legs (55 min and 4 h24 m residencies, journal
   13:11→14:06 and 14:18→18:41) with prewarm completed before probing; the Nov-8/9 legs are
   2.5-minute fresh-boot cycles whose logs show `prewarm: page cache warm in 18 s` landing
   AFTER the first ~11 measured requests (gsq-clean-r1.out line 78), plus engine deltas
   (expert cache 17.46→18.73 GiB, ctx 131072→262144, shadow store disk→/dev/shm, live
   `NameError: releasing` thread crashes in every current engine log).

Confidence: A for every journal/live/file claim below (measured on this box); the
attribution argument is inference from A-evidence.

---

## 1. Per-era clock regime from the journal (strata-gpu-lock, Nov-06 → now)

### 1.1 Timeline of governor versions (unit events, `journalctl -u strata-gpu-lock`)

| When (journal label) | Event | Governor live |
|---|---|---|
| Nov-01 (boot -1) | unit started; ran **5d 23h 36min 19s**, consumed 57min CPU | **v2** |
| Nov-06 all day | 86 `GPU processing`, 72 `GPU quiet`, 21 `engine gone`, 14 `engine detected` | **v2** (all four messages are v2's hard-pin state machine) |
| Nov-07 15:23:17 | stopped + restarted (the "Consumed 5d23h" line closes the v2 era) | **v3** installed here — 2× `watching v3 (floor 1800-3135 …)` |
| Nov-07 16:04:46 | stopped + restarted → **10 consecutive start-operation timeouts** (45 s each, restart counter 1→10) — the Type=notify/READY storm while v4's unit was being fitted | v3→v4 churn, no governor most of 16:04-16:13 |
| Nov-07 16:13:37, 16:24:04 | v4 finally starts (13× `watching v4 … watchdog dead-man on`) | **v4** |
| Nov-08 ~12:22 real (journal labels 14:22 pre-NTP-step; box was off ~00:05→12:22) | reboot. v4 enabled at boot: `watching v4` at 12:24:02 | **v4** — uninterrupted since |
| Nov-08 12:58→13:20 | 4× `engine listening — clock floor` / 4× `engine gone 45s — clocks released` = the hostmem-definitive-iq3s legs (r1-r3 files 13:10-13:15) | v4 |
| Nov-09 10:25→now | 5× `listening` / 4× `gone` = gsq-clean r1-r3 + gsq-restore legs | v4 |

**Answer to "was the floor present during the Nov-6 banks?": NO — and no v2-style lock
either, in the v3/v4 sense.** Nov-6 ran v2: a three-tier HARD-PIN governor
(`/etc/sysconfig/strata-gpu-lock`, still the v2 tunables file today: `SM_CLOCK=3135`,
`SM_MID=1800`, `IDLE_GRACE=30`, `ACT_UTIL=10`, `ACT_POWER=150`):
engine resident + processing (util ≥10% or power ≥150 W) → hard pin SM 3135 / mem 14001
(single-value `-lgc`/`-lmc`: min=max, boost blocked above the pin, power cap trims to
~2830-2955); resident + quiet >30 s → hard pin SM 1800 / mem 7001; engine gone 45 s →
released. The v3/v4 FLOOR (range lock, hardware-owned boost) did not exist until Nov-07
15:23. The Nov-6 regime was strictly WORSE for bursty loads than what Nov-8+ runs.

### 1.2 Regime table per era

| Property | Nov-6 bank era (boot -1) | Nov-8/9 probe era (boot 0) |
|---|---|---|
| Governor | v2 hard-pin (since Nov-01) | v4 floor-range + watchdog (since Nov-08 12:24) |
| Sustained probe-load SM | ~2830-2955 (hard 3135 pin, pviol-trimmed; Nov-7 dmon proof) | ~2830-2850 (range ceiling, pviol-trimmed; same cap) |
| Sustained probe-load mem | 14001 hard-pinned | 13801 (DVFS under 7001,14001 range; digest §2.2 "do not expect 14001 flat") |
| Burst entry after >30 s gap | **seconds at 1777-1792 MHz + mem 7001** (measured 30% defect) | instant boost, first 1 s sample already 2805 (digest §3.1) |
| Burst entry after <30 s gap | enters at the still-pinned 3135/14001 (no decay) | enters at decayed ~2062-2557, ramps inside ~1 s |
| Between-leg state | released (DVFS ~210 idle, 35-48 W) | released (same) — release fires before every leg (below) |
| Thermal events in probe windows | none logged (v2 logged none) | none (no shed/guardrail lines Nov-08/09) |
| Floor applied right now (Nov-09 10:44+) | n/a | **YES** — engine listening since 10:42:25; live: SM 2107-2497 @ 56-58 W quiet, P1 |

Static conclusions possible without an engine running: the v4 state machine is intact in
the journal (floor on engine-listen, release on gone-45s, release-trap on TERM, watching
banner parsed `floor 1800-3135 / 7001-14001`); `/etc/sysconfig/strata-gpu-lock` still
carries v2-era variable names but v4 remaps them (`SM_MID`→floor 1800, `SM_CLOCK`→max
3135) so the effective floor/ceiling are exactly 1800-3135 / 7001-14001; persistence mode
On; driver 615.71.09 open + GSP unchanged across the reboot; kernel identical
(7.2.12-300.fc45) in boot -1 and boot 0.

## 2. Other box state per era

| Item | Nov-6 era (boot -1, Nov-01 17:45 → Nov-08 00:05) | Nov-8/9 era (boot 0, Nov-08 ~12:22 → now, 22 h) |
|---|---|---|
| Uptime at measurement | ~4.5-5 days | **~35 min (Nov-8 legs 12:58-13:20) / ~22 h (Nov-9 legs)** |
| Kernel / driver | 7.2.12-300.fc45 / NVIDIA 615.71.09 open | identical |
| GPU Xid / GSP faults | 0 | 0 |
| Hugepages | `vm.nr_hugepages=0` ("dead since forever", HOSTMEM-NO-REBOOT.md) — no pool | **pool armed**: sysctl drop-ins 90-strata-vm.conf (Nov-07: `compaction_proactiveness=40`) and 91-strata-hugepages.conf (Nov-08 13:02, config `nr_hugepages=54272` = 106 GiB); live pool **45076 pages = 88 GiB, 18.5 GiB still free** — 88 GiB of flexible RAM removed from the general allocator vs Nov-6 |
| zram | filled "within days" per HOSTMEM doc; day-5 state not sampled, but the era ended in a **global OOM kill Nov-06 18:41:30** (`strata`, anon 26 GB + shmem 95.7 GB) | 3.8 GiB / 8 GiB used at 22 h (mm_stat) — healthier |
| AnonHugePages | 120 MB of 133 GiB resident (HOSTMEM doc, era close) | 15.3 GB now (THP `madvise`) |
| Desktop/Chrome | heavy all era (30,111 chrome journal lines boot -1); documented Chrome-at-load-19 leg scored **92.6 t/s = fake -55%** (fork-ud-first-bank.md); bank trio itself marked "quiet box" | 135 lines / 33 processes now; load avg 3.88-3.90 with engine up |
| Temperature | 33 C quiet / 50-51 C busy (Nov-7 measurements); season start-of-summer | 37-38 C quiet / 42 C idle-ish now — +5-9 C vs Nov-7, irrelevant vs the 83 C shed and 57 C effective T.Limit history; zero shed events |
| Package state | — | no dnf install/upgrade transactions Nov-06→09 in /var/log/dnf5.log; engine built from the same tree (commit HEAD moved during Nov-6 day itself: 1553a614 → HEAD trio) |

Both directions of the host-memory ledger favor the NEW era: Nov-6 carried the aging
disease the HOSTMEM campaign convicted (free order-9 supply 18→16→9→0 GiB across that
week, zram refill, 4 KB expert arena) — and the Nov-6 trio still scored 233/225/303 on
that degraded, v2-handicapped box. The new box's only new cost is the 88 GiB reservation
(no GSQ benefit: the GSQ pinned pool is 4 KB by construction either way) and
kcompactd at proactiveness 40.

## 3. Ramp analysis — 2.5-minute boot cycles vs the bank era

**Both eras release clocks between legs; both re-arm within ≤3 s of the port binding; the
210 MHz released state never reaches a measured window in either era.**

Leg cadence from journal timestamps (release = `engine gone 45s`; listen = port bind at
`ready`):

- **Nov-6 morning battery legs (v2):** listen→stop ≈ 73-136 s, gap 48-1419 s. Every gap
  produced a release (`gone 45s`) before the next `detected` — legs started released, same
  as today.
- **Nov-6 bank trio window (v2):** LONG-LIVED server, not cycles. Residencies of 55 min
  (13:11:50→14:06:52) and 4 h24 m (14:18:00→18:41:30, ended by the OOM kill). The bank's
  2-leg means came from same-boot legs ("leg 1 of 2 same boot" is the tracker's own
  language for these banks). Under v2, once probing kept gaps <30 s, the card sat at the
  hard 3135/14001 pin for the whole bank; the 1780 plateau was paid once per leg at most.
- **Nov-9 legs (v4):** listen→stop ≈ 94 s, gaps 21-36 s (10:28:16 gone → 10:28:41 listen;
  10:30:51 → 10:31:12; 10:33:19 → 10:39:23). Release fired before every leg; each leg
  boots with clocks released, the floor lands ≤3 s after `ready`, and the probe block
  (shorts → M1 → M2 → M3) runs entirely in the floored regime with instant boost.
  Model-load time (60-180 s) runs BEFORE the port binds, so it never touches the measured
  windows either; under v4 the load itself even boosts (range), where v2 pinned it at
  1800/7001.

**The real measurement-pattern delta is not clocks — it is prewarm racing the probe.**
The Nov-6 bank legs had residency long enough that the page-cache/expert warm-in was
finished before the scored windows. The Nov-9 gsq-clean legs show
`prewarm: page cache warm in 18 s` (80.1 GiB fill) landing at gsq-clean-r1.out line 78 —
after ~11 scored requests — and `prewarm: replay done in 7 s` later in the same leg. An
80 GiB page-cache fill running through the shorts/M1/M2 windows competes for DRAM
bandwidth and CPU exactly where the CPU-expert lane is most sensitive. Leg spread agrees:
Nov-6 same-boot 2-leg banks sat within ±1% (229.1→233.1); Nov-9 fresh-boot legs spread
175.7→195.1 solo (±5.3%) with one fresh-boot leg at 75.3 solo while pair/conc stayed
normal (gsq-restore-r1) — a fresh-boot-only pathology shape.

## 4. Could the clock/system regime explain -8..-15% on all three lanes at once?

**No.** Mechanism-by-mechanism:

| Candidate | Direction vs Nov-6 | Verdict |
|---|---|---|
| v2→v4 sustained clocks | identical (~2830-2955 cap-trimmed; mem 13801 vs 14001 = -1.4%, digest says expect it) | cannot move any lane |
| v2→v4 burst entry | v4 strictly better (removes the measured 30% plateau defect Nov-6 paid) | would RAISE Nov-8 scores if anything |
| Between-leg ramp (210 idle → floor) | identical in both eras (release before every leg, re-arm ≤3 s after ready, load phase pre-bind) | non-delta |
| Thermal shed / guardrail / Xid | zero events in both eras' probe windows | non-delta |
| Hugepage reservation (new, 88 GiB) | costs GSQ nothing directly (pinned pool is 4 KB either way); shrinks page cache — and prewarm still completed inside each leg | pressure-side confounder at most, not a clock mechanism |
| zram / compaction | better than Nov-6 (3.8/8 GiB at 22 h vs day-5-refill + era-ending OOM) | would RAISE Nov-8 scores if anything |
| Desktop/Chrome | lighter now than the era average; the Nov-6 trio needed a manually-quiet box to post 233 | residual confounder, both eras |
| Temperature | +5-9 C, 45 C of shed margin left | non-delta |

What the -8..-25% IS plausibly made of (adjacent lanes, named for the next tier — all
engine/measurement-side, all visible in the same logs):

1. **Prewrite/prewarm race** on 2.5-min fresh-boot legs (§3) — depresses exactly solo/pair
   short windows, matches the wide leg spread and the 75.3 solo outlier with normal conc.
2. **Engine config deltas between the eras**: expert cache 9204/17.46 GiB (Nov-6 t22 log)
   → 9872/18.73 GiB (Nov-9 logs); ctx 131072 → 262144; shadow store `/home/svg/LLM/data/
   packs/q2_0` (Nov-8 r1, restore legs) → `/dev/shm/strata-shadow-q2_0` (Nov-9 clean legs).
3. **A live engine bug**: `NameError: name 'releasing' is not defined` at
   `serve/server.py:1728` crashes the slot-release watcher thread repeatedly in every
   Nov-8/9 engine log (Thread-39, 40, 47, 49, 73, 75, 96-98, 101-103…). Slot release
   failures on a small pool are a scheduling-path hazard the Nov-6 binary did not log.
4. **Law-17 baseline honesty**: the box's own measured cross-window drift is ±8%; Nov-6's
   own ordinary-conditions legs (post-writeback, load 4-7) read 191-224 — i.e. -4..-17%
   below the quiet-box trio ON THE SAME DAY. Today's 176-195 overlaps that band at its top.

**Recommended falsification, next battery, zero code changes:** rerun the exact battery
harness with (a) `wait_prewarm` enforced before the first scored request (the
battery-lib.sh gate exists), (b) one long-lived server leg (bank-era pattern) on this box.
If the long leg restores ~225+ solo, the regression is the boot-cycle/prewarm pattern +
engine deltas, and the box is acquitted with prejudice; the clock governor needs no
change. Capture `dmon -s pucvmet` during the M1 window of both patterns to bank the
sustained-clock proof for the v4 era (the one artifact this pass could not produce
statically — no scored window was live at audit time; the opportunistic quiet capture is
banked above).

## Provenance

- Journal: `journalctl -t strata-gpu-lock` (Nov-05→09 state transitions, watching banners,
  floor/release lines); `journalctl -u strata-gpu-lock.service` (unit stops, the Nov-07
  16:04-16:13 restart storm, v2's 5d23h36m invocation); `journalctl -k -b -1/-b 0` (kernel
  identity, HugeTLB registration, Xid sweep, Nov-06 18:41:30 OOM kill).
- Live this pass: `nvidia-smi --query-gpu=…` (floor state, 10:44), `nvidia-smi dmon -s
  pucvmet -c 5` (quiet-with-floor capture, 10:50), `ss` :8081 listener, `/proc/meminfo`
  (Hugetlb 45076 pages), `/sys/block/zram0/mm_stat`, `/proc/sys/vm/nr_hugepages`,
  `/sys/kernel/mm/transparent_hugepage/enabled`.
- Files: `/usr/libexec/strata-gpu-lock` (v4 script, v2 defect recorded in its header),
  `campaign/gpu-lock/strata-gpu-lock-v3` + README, `/etc/sysconfig/strata-gpu-lock` (v2
  tunables), `/etc/sysctl.d/90-strata-vm.conf` + `91-strata-hugepages.conf`,
  `campaign/tracker/lanes.csv` (Nov-6 rows 229.1/220.7/301.6 and 233.1/224.8/303.5),
  `campaign/results/fork-ud-first-bank.md` (Chrome -55% leg, law-17 legs 191-224),
  `campaign/results/gsq-clean-r{1,2,3}.out` + `hostmem-gsq-clean.log` (M1 175.7/178.4/195.1),
  `campaign/results/gsq-restore.log` (M1 75.3 leg), `campaign/results/hostmem-definitive-*
  (Nov-08)`, `campaign/results/t22-custom-r1.out` (Nov-6 engine log, 131072 ctx, 9204
  experts/17.46 GiB, no prewarm lines), `campaign/HOSTMEM-NO-REBOOT.md` (the aging
  conviction), `.docs/research/digest/2026-11-07-gpu-clock-governance.md` (sustained
  ~2830±30, 1780-plateau mechanism, floor semantics).
