---
type: research-digest
id: 2026-11-07-preboot-aging-proof
date: 2026-11-07
scope: the proof and drills that earn the "never needs a performance reboot again" claim on this box — the aging ratchet (cycle-1 == cycle-N), four operator drill cards (short-pool boot, kill-a-9, concurrent-pool pressure, STRATA_NO_LARGEPAGES round-trip), the daily watchdog hooks (diffs-to-write, not applied), long-soak precedents from kernel/DB/NIC practice, and the honest boundary of what the pool does NOT cover
subject: post-installation proof design for the boot-armed 2 MiB pool (HOSTMEM-NO-REBOOT.md M5, 45056 pages)
method: local design against the installed instruments (campaign/hostmem-anon-collapse.c, the ARENA PAGES banner in src/core/pinned.cu, campaign/shm-shadow.sh gates, battery-lib.sh, probe-lane.py, tracker/lanes.csv) + web research 2026-11-07 (kernel.org selftests/mm, gormanm/mmtests, UNH-IOL DPDK lab, MongoDB sys-perf change-point detection, soak-testing practice). No code changed in this pass; every hook below is a diff-to-write.
---

# The preboot-aging proof: ratchet, drills, watchdog

**Status:** design, 2026-11-07. Owner: this file; the drill cards become `campaign/batteries/` scripts at implementation time. Gates: law 17 (same-window interleave; the ±8% cross-window drift is measured, not assumed), law 13 (no lane worse), law 12 (chart inclines — the ratchet appends only standing-recipe points). The claim it earns: **"the 2026-11-07 reboot was the last one that ever changed performance."**

Everything below is sized to what already exists: the banner strings are read from `src/core/pinned.cu`, the store gates from `campaign/shm-shadow.sh`, the witness keys from `campaign/batteries/hostmem-leg1.sh` and `campaign/HOSTMEM-NO-REBOOT.md`. Day-1 numbers (`campaign/results/hostmem-day1-verdict.md`) calibrate every threshold.

## 0. The claim, stated falsifiably

"Never needs a performance reboot again" = **both** of these hold:

- **R1 (restart-invariance):** an engine stop/start cycle on an aged box re-arms to the same page class and the same lane numbers as cycle 1 — forever. Every restart is a perfect re-arm; nothing accumulates.
- **R2 (calendar-invariance):** the same holds across days of real box life (desktop, browser, zram churn) with **uptime only increasing** — the lane series at day N equals day 1 within noise, `/proc/uptime` strictly monotonic.

R1 without R2 is the old lie restated ("fresh boot wins by accident" — one restart is a fresh boot). R2 without R1 is unmonitored hope. The failure mode that started this campaign was a *week-long silent slide* (arena supply 18 → 16 → 9 → 0 GiB across days); the watchdog's job is to catch that class **in one cycle**.

---

## 1. The aging ratchet (the soak that proves R1 + R2)

### 1.1 Shape — two instruments, because two different decay laws exist

| instrument | proves | N | duration | cost |
|---|---|---|---|---|
| **Burst ratchet** | R1: per-restart decay | 10 boot/stop cycles, one evening window | ~60 min | 10 engine boots, solo-only probes |
| **Calendar ratchet** | R2: day-over-day decay under real aging | 14 daily batteries, no OS reboot | 14 days | rides the existing daily runner |

The burst alone cannot see day-scale decay (compact_fail accumulates over days, kswapd pressure builds over days). The calendar alone cannot cleanly see per-restart decay — its once-a-day batteries are confounded by the ±8% window drift the day-1 verdict measured. Together they cover both terms. This mirrors the kernel's own split: `run_vmtests.sh` is a bounded per-boot suite; mmtests runs N-iteration comparisons for the slow signals (§4).

### 1.2 Burst ratchet (R1) — exact schedule

Run in the daily runner's evening slot once, after the definitive post-boot battery of day-1 verdicts is banked. Engine-solo window: no desktop-heavy work during the hour (load avg recorded per cycle; law 17).

```
For cycle k = 1..10, on the UD-IQ2_XS pack (launch-custom-ud-iq2xs-8081.sh):
  cleanup_engine (battery-lib.sh)
  witness: gauge score  = campaign/hostmem-anon-collapse  (engine down — the state the boot faults into)
  witness: hp0          = HugePages_Total / HugePages_Free  (/proc/meminfo)
  witness: cf0          = compact_fail                      (/proc/vmstat)
  witness: up0          = /proc/uptime, load = /proc/loadavg
  boot launcher -> wait_health -> wait_prewarm
  witness: hp1 = HugePages_Free after the arena drew (draw = hp0.free - hp1)
  witness: banner class from the pack log:
      "strata: ARENA PAGES: 2 MiB (39.4 GiB arena, large-page backed)"      (pinned.cu:427)
      "strata generate: expert arena: cudaHostRegister PORTABLE ok; hugetlb 2 MB pages"
  witness: store tier line from the launcher stderr: "shm-shadow: tier ... (hugetlbfs)"
  probe:  python3 campaign/probe-lane.py --port 8081 --label ratchet-burst-c<k> \
            --short-n 6 --skip-conc        # solo lane only: M1 + shortsA/B, ~3 min
  witness: cf1 = compact_fail after the probe cycle (delta = cf1 - cf0)
  witness: zram fill = byte 1 of /sys/block/zram0/mm_stat
  witness: idle VRAM = nvidia-smi --query-gpu=memory.used --format=csv,noheader
stop engine, leave the standing lane up (final_serve)
```

Why UD-IQ2_XS solo-only for the burst: it is the organ with the largest single pool draw (39.4 GiB arena, MAP_HUGETLB all-or-nothing), its M1 lane is the most TLB-sensitive read in the campaign, and solo is the lane the 2 MiB class moved most (+4.9% day-1). The GSQ pack's paced-complement path is exercised daily by the calendar leg — no need to double-boot it 10 times.

**Why not long probes per cycle:** probe-lane full (pair + conc + 90-section fresh prompt) is ~8–10 min; ×10 cycles it eats the box and the window. Solo-only at 3 min keeps the whole ratchet inside one drift-stable hour. Law 11 holds — `wait_prewarm` before every cycle.

### 1.3 Calendar ratchet (R2) — exact schedule

14 days, engines restarting on the normal daily rhythm (the daily runner already boots per battery; every real-world restart between batteries is a free cycle — the watchdog hooks in §3 capture each). Rules:

1. **No OS reboot.** `/proc/uptime` is recorded in every witness row and must be strictly increasing across the 14 days. One reboot voids the ratchet and it restarts from day 1 (the claim is about THIS uptime arc).
2. **Daily battery** = the standing one from the day-1 verdict: 3 boots per pack (GSQ IQ3_S + UD-IQ2_XS), full probe-lane (solo/pair/conc), same launchers, zero env changes — exactly `hostmem-leg1.sh`'s shape with the definitive 3+3 sizing. Booked in the runner's evening window.
3. **Day 7 mid-check:** 2+2 interleave, same hour: two boots from the day-1 battery script vs two boots with the day-7 calendar state — no, the packs are identical; the interleave is *battery-as-day-1-schedule* vs *battery-run-right-now*, which is the same thing. The real mid-check is simpler: day 7's battery is compared to day 1's **same-hour** (book both at the same clock window), and the watchdog series carries the load/zram covariates that explain any read. No special legs.
4. **Day 14 verdict leg:** re-run the *identical* day-1 battery schedule (same packs, same order, same hour) on the aged box. This is the law-17 lens for the whole arc: same-window pairing of day-1-class vs day-14-class without ever rebooting to get the "day-1" side — the day-1 side is *already banked* and the same-hour pairing is day-1-schedule vs day-14-schedule run in one window (both sides boot fresh; the aged variable is the OS uptime, which is the thing under test).

### 1.4 Pass criteria (statistical, written before the run)

**Burst (R1), per lane solo t/s across cycles 1..10:**

- **Band gate:** every cycle's solo mean (the 6 shorts + M1 read averaged — probe-lane's SHORT-CLASS and M1 lines) within **±3% of the burst pooled mean** (all 10 cycles pooled — pooling avoids anchoring on a lucky cycle-1; a single boot's noise band is ~±4% per day-1, so a 3-boot-mean-per-cycle at ±3% is tight but the pooled-mean formulation tolerates one wobble).
- **Trend gate:** Theil–Sen slope of tps vs cycle k with bootstrap 95% CI must straddle 0; Spearman ρ over (k, tps) reported with p-value, reject only if ρ < 0 and p < 0.05. (n=10 needs |ρ| ≥ 0.648 for significance — the band gate carries the verdict; the trend gate exists to catch the slow slide the band could miss if it creeps from inside the band.)
- **Mechanism gate (hard zeros, no statistics):** banner class `2 MiB` in 10/10 cycles; pool draw within ±3% of cycle-1 draw every cycle; `compact_fail` delta ≤ 1000 per cycle (day-1 aged-box boots logged deltas in the thousands; on the armed pool the correct number is ~zero); gauge ≥ 95% promoted every cycle; `HugePages_Free` after teardown returns to the pre-boot free count ±2 pages.
- **Verdict:** all three gates green → R1 holds. Any red → the failure signature table (§2.5) names the drill.

**Calendar (R2), 14 days:**

- **Same-hour verdict gate (day 14):** per pack, per lane (solo/pair/conc), |day14 mean − day1 mean| ≤ **3%** on the same-hour re-run of the identical schedule. Three boots per side; the five-boot-means discipline applies if a lane straddles (tracker README protocol).
- **Trend gate:** Spearman ρ over the 14 daily solo means vs day index, reject only if ρ < 0 and p < 0.05; the daily means are also plotted with uptime as the x-covariate — the chart should show a *flat* line over rising uptime, which is the whole story (build-trend.py's contract is an incline for adoptions; for the ratchet the contract is **flatness over uptime** — the label carries it).
- **Watchdog gate:** zero RED watchdog rows across the 14 days (§3); AMBER rows must each carry an explanation row (e.g., a desktop zram spike with the engine out of swap by M4).
- **Verdict:** all gates green → R2 holds, and the campaign may append the chart row described in §1.6 and *only then* claim "never again."

**Why ±3% and not the campaign's ±4% adopt bar:** the ratchet measures *sameness*, not adoption; the day-1 verdict showed same-window 3+3 reads settle within ~±2% (UD pair/conc flat within 190.5/190.0) while single reads wobble ±4–8%. A 3% band on 3-boot means is above the measurement floor and below the degradation signal (the disease was −15% vs the standing bank at its mildest measured harm, and the 4K class cost −5.7% solo on UD day-1: 193.2→184.2 arm side vs the 4K class ~184).

### 1.5 Running alongside the daily runner (idle windows)

- The burst books the daily runner's own evening slot on day 0 — it *is* that day's battery (its pooled solo mean becomes the day's standing solo row if the pass holds; see §1.6).
- The calendar adds zero legs: it is the daily runner plus the watchdog hooks plus the day-14 re-run.
- Idle-window discipline: no battery launches while live serving is in progress (the runner already serializes on port 8081); the gauge binary runs only engine-down (it faults 256 MiB and collapses it — 2 s of work, but it belongs in the engine-down window by construction).
- The burst may coexist with desktop use but the row records loadavg; if load1 > 4 at any cycle, that cycle is voided and re-run (a cheap freshness guard, cheaper than disputing the verdict later).

### 1.6 Chart and tracker rows produced (law 12-conformant)

All rows: append to `campaign/tracker/lanes.csv`, then `python3 campaign/tracker/build-trend.py` **in the same commit** (tracker README protocol).

```csv
qwen-ud-iq2xs,2026-11-07,<commit>,HOSTMEM ratchet burst (10 cycles, uptime 6h, pool 45056, gauge 100%, cf+18),solo,<pooled mean>,0,mean
qwen,2026-11-21,<commit>,HOSTMEM ratchet d14 (uptime 336h, pool 45056, store hugetlbfs, cf-day +410),solo,<mean>,0,mean
qwen,2026-11-21,<commit>,HOSTMEM ratchet d14 (uptime 336h, pool 45056, store hugetlbfs, cf-day +410),pair,<mean>,0,mean
qwen,2026-11-21,<commit>,HOSTMEM ratchet d14 (uptime 336h, pool 45056, store hugetlbfs, cf-day +410),conc,<mean>,0,mean
qwen-ud-iq2xs,...  (same three lanes)
```

Rules: the label carries the mechanism witnesses (uptime, pool total, gauge, compact_fail) so the chart point is self-describing; only standing-recipe points append (the ratchet runs the standing launchers — this is satisfied); burst cycles 1..10 individually are diagnostic, living in `campaign/results/hostmem-ratchet-burst.log` and the F-entry, not the chart; the daily calendar batteries append `kind=mean` rows labeled `HOSTMEM ratchet d<k>` — 14 points across the arc, and the day-14 triple is the banked verdict point. A declining daily mean is a *question* under law 12 (same-hour control → five-boot means → binary-ladder), never an appended point.

**Engine column names:** `qwen` (GSQ IQ3_S lane), `qwen-ud-iq2xs` (UD lane) — matching the existing rows.

---

## 2. The drills (four operator cards)

Each card: purpose, exact commands, expected lines (verbatim from the sources named), failure signatures, restore. All four are destructive-by-design and run in a booked window with the operator present; none runs under live serving. Prereqs: root via sudo; engine down; `. campaign/battery-lib.sh` for `cleanup_engine`/`wait_health`/`final_serve`; `GAUGE=$REPO/campaign/hostmem-anon-collapse` built (`gcc -O2 -o ...` per leg1's guard).

Common helper snippets used by every card:

```bash
REPO=/home/svg/LLM/strata-custom-2; PORT=8081
hp_total()  { awk '/^HugePages_Total:/{print $2}' /proc/meminfo; }
hp_free()   { awk '/^HugePages_Free:/{print $2}'  /proc/meminfo; }
cf()        { awk '/^compact_fail/{print $2}' /proc/vmstat; }
zram_fill() { awk '{print $1/2**30 " GiB"}' /sys/block/zram0/mm_stat; }
```

### DRILL-A — short-pool boot (the degrade chain)

**Purpose:** prove the documented degrade chain actually fires on a pool shortfall: store → `/dev/shm` tier (pool-gate), arena → THP/4K fallback + banner, strict mode refuses for benches. This is the drill that keeps the fallback *loud* — the class of bug this campaign exists to kill is the silent 4K.

```bash
cd $REPO && . campaign/battery-lib.sh && cleanup_engine
# 1. shrink the pool at runtime (the same state a boot-armed shortfall leaves; engines are down so
#    every reserved page is free and the shrink is clean):
sudo sh -c 'echo 20000 > /proc/sys/vm/nr_hugepages'
echo "pool now: $(hp_total)/$(hp_free)"        # expect 20000/20000 (or less, never more)
# 2. serve-mode degrade (the default posture — must boot DEGRADED, never fail):
( cd $REPO && nohup ./launch-custom-ud-iq2xs-8081.sh >/tmp/drill-a-serve.out 2>&1 & )
wait_health && echo "serve-mode: booted (degraded posture, as documented)"
grep -aE "shm-shadow:|ARENA PAGES|expert arena" /tmp/drill-a-serve.out "$REPO/logs/custom-ud-iq2_xs.log" | tail -6
# 3. strict-mode refusal (bench posture — must refuse):
cleanup_engine
( cd $REPO && STRATA_ARENA_PAGES=strict nohup ./launch-custom-ud-iq2xs-8081.sh >/tmp/drill-a-strict.out 2>&1 & )
sleep 150   # the boot gets far enough to reserve the arena (paced step ~45-93 s class)
grep -aE "ARENA PAGES|strict" /tmp/drill-a-strict.out; wait_health || echo "strict: refused (no /health) - CORRECT"
# 4. GSQ variant while the short pool holds: the complement must NOT take the pool path:
cleanup_engine
( cd $REPO && nohup ./launch-custom-iq3s-8081.sh >/tmp/drill-a-gsq.out 2>&1 & )
wait_health; grep -aE "complement takes the paced|expert complement:" /tmp/drill-a-gsq.out "$REPO/logs/custom-iq3_s.log" | tail -3
# 5. restore (the documented runtime path from campaign/91-strata-hugepages.conf):
cleanup_engine
sudo sh -c 'echo 45056 > /proc/sys/vm/nr_hugepages'; echo "restored: $(hp_total)"
#   if short on the aged box: echo 1 > /proc/sys/vm/compact_memory once (engine down), then raise in steps.
[ "$(hp_total)" = 45056 ] || echo "RESTORE SHORT - run the compact-then-steps path before any bench"
```

**Expected lines:**

| where | expected (verbatim source) |
|---|---|
| launcher stderr | `shm-shadow: pool 20000 pages < 40000 (partial runtime reservation) - store stays on /dev/shm, the pool is the arena's; the boot-armed pool (sysctl.d 91-strata-hugepages) lifts this gate` (shm-shadow.sh:30) |
| launcher stderr | `launcher: shadow store /dev/shm/strata-shadow-q2_0` (tier-0 fell through to the tmpfs tier) |
| engine stderr (UD) | `strata: ARENA PAGES: 4K - DEGRADED CLASS (39.4 GiB arena): MAP_HUGETLB unavailable (needed 20164 2 MiB pages, vm.nr_hugepages=20000 ...)` (pinned.cu:273 note + :431 banner) |
| strict mode | `strata: STRATA_ARENA_PAGES=strict: refusing to serve a 4K arena - arm the 2 MiB pool (vm.nr_hugepages) or drop the strict flag` then process exit code **70** (pinned.cu:435-437); `/health` never answers |
| GSQ (short pool) | the paced line **absent**: `FileExpertSource: STRATA_HUGEPAGE_ARENAS=1: the complement takes the paced 2 MiB pool path...` must NOT appear (`huge_pool_covers()` false, expert_source.cpp:3541); the complement lands on the one-shot `cudaHostAlloc` path ("page-locked and mapped") — the documented 4K class |
| after restore | next standing boot: `strata: ARENA PAGES: 2 MiB (39.4 GiB arena, large-page backed)` + `cudaHostRegister PORTABLE ok; hugetlb 2 MB pages` + `shm-shadow: store on hugetlbfs - 2 MiB by construction (pool draw ~19000 pages, ...)` |

**Failure signatures:**

- Store went to hugetlbfs at pool 20000 → the pool gate in shm-shadow.sh did not see the real pool; the gate or the mount is lying (`stat -f -c %T /mnt/strata-huge`).
- Banner reads `2 MiB` while `HugePages_Total` = 20000 → impossible under all-or-nothing MAP_HUGETLB (pinned.cu:263 comment); if ever observed, stop — kernel regression or the banner reads the wrong backing.
- Strict mode served anyway (health answers, no refusal line) → `STRATA_ARENA_PAGES` not reaching the engine binary (launcher env plumbing), or the arena silently took a <1 GiB shape (banner suppressed by the `bytes < 1 GiB` guard, pinned.cu:425).
- Restore stuck short after compact-then-steps → the aged buddy cannot re-supply 45056 at runtime; this is the *named* runtime-raise boundary (kernel docs: silently partial) — the answer is a reboot, and the drill's own claim ("degrades loudly, boots anyway") is what makes that reboot safe instead of a performance gamble.

**PASS:** all six expected lines fire; strict exit 70 observed; restore reaches 45056; post-restore boot re-arms 2 MiB and solo is within ±3% of bank.

### DRILL-B — kill-a-9 (the crash case)

**Purpose:** SIGKILL the engine mid-serve and prove the crash loses nothing the pool owns: every arena page returns whole to `HugePages_Free`, the next boot re-arms perfectly. This is the drill that makes "engines restart several times daily" a feature with evidence.

```bash
cd $REPO && . campaign/battery-lib.sh
final_serve launch-custom-ud-iq2xs-8081.sh /tmp/drill-b-serve.out   # standing lane up, warmed
wait_prewarm
# open a real stream in the background (probe-lane's M1 leg mid-flight when the kill lands):
( cd $REPO && python3 campaign/probe-lane.py --port $PORT --label drill-b-victim --skip-conc >/tmp/drill-b-victim.log 2>&1 & )
sleep 20
HP0=$(hp_free); CF0=$(cf)
PID=$(ss -ltnp | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)      # server pid
EPID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1) # the ENGINE pid
kill -9 "$EPID"                                  # mid-serve, no cleanup, the crash case
sleep 5
HP1=$(hp_free); echo "pool returned: $((HP1-HP0)) pages (expect ~20164: the 39.4 GiB arena, all-or-nothing)"
cleanup_engine                                   # TERM the server, free the port
CF1=$(cf); echo "compact_fail delta across the crash: $((CF1-CF0)) (expect ~0: SIGKILL frees, never fragments)"
# re-arm (the next boot must be a perfect re-arm):
( cd $REPO && nohup ./launch-custom-ud-iq2xs-8081.sh >/tmp/drill-b-rearm.out 2>&1 & )
wait_health && wait_prewarm
grep -aE "ARENA PAGES|hugetlb 2 MB" /tmp/drill-b-rearm.out "$REPO/logs/custom-ud-iq2_xs.log" | tail -3
python3 campaign/probe-lane.py --port $PORT --label drill-b-rearm --short-n 6 --skip-conc | grep -E "M1|SHORT-CLASS"
```

**Expected lines / numbers:**

- `pool returned: ~20164 pages` (HugePages_Free climbs by the arena draw; the hugetlbfs store keeps its ~19000 held — that is expected, the store file still owns them).
- `compact_fail delta ≈ 0` — SIGKILL VMA teardown frees reserved pages without touching the buddy; if compaction ran, something migrated into the freed space and the "perfect re-arm" story has a hole.
- Re-arm boot: `strata: ARENA PAGES: 2 MiB (39.4 GiB arena, large-page backed)`; `cudaHostRegister PORTABLE ok; hugetlb 2 MB pages`.
- Re-arm solo within ±3% of the standing bank (226.5 class for UD day-1 legs).

**Failure signatures:**

- Free does not return the arena count → a second process still maps pool pages (check `ls /mnt/strata-huge` for accidental files; `grep -l hugetlb /proc/*/status`-style audit via `smaps_rollup` `HugetlbPages:`), or the driver pinned the pages out of the pool's accounting — escalate with the audit output.
- Re-arm banner reads 4K → the pool lost pages across the crash (the leak above) — DRILL-A's short state without a cause is a stop-and-investigate.
- Re-arm solo −5% with a 2 MiB banner → the degradation is NOT page class; look at the engine log's `ms/window` trend and the GPU witnesses — this exact ambiguity is why the banner exists (it fences the page-class hypothesis out in one line).

**PASS:** page return exact (±2 pages), cf delta ≤ 1000, re-arm banner 2 MiB, solo in-band, server cleaned and port released.

### DRILL-C — concurrent-pool pressure (booting into a partially-held pool)

**Purpose:** boot the engine while a foreign holder owns pool pages — the leg-1 "GSQ control tax" incident class (a partial-pool hold cost an un-migrated organ −12..−17%). The engine must degrade by the same loud chain as DRILL-A and must never half-map.

```bash
cd $REPO && . campaign/battery-lib.sh && cleanup_engine
ls /mnt/strata-huge/   # the hugetlbfs store mount (size=40G cap) is the honest holder shape
dd if=/dev/zero of=/mnt/strata-huge/pressure.bin bs=1M count=30720 status=none   # hold 30 GiB = 15360 pages
echo "held; free: $(hp_free)"   # expect ~45056 - store(~19000) - 15360 ≈ 10700 < arena need 20164
# boot UD into the squeezed pool:
( cd $REPO && nohup ./launch-custom-ud-iq2xs-8081.sh >/tmp/drill-c.out 2>&1 & )
wait_health || echo "boot failed - check: pool-priority should have degraded, not failed"
grep -aE "shm-shadow:|ARENA PAGES|MAP_HUGETLB" /tmp/drill-c.out "$REPO/logs/custom-ud-iq2_xs.log" | tail -6
# strict refuses under pressure too:
cleanup_engine
( cd $REPO && STRATA_ARENA_PAGES=strict nohup ./launch-custom-ud-iq2xs-8081.sh >/tmp/drill-c-strict.out 2>&1 & )
sleep 150; grep -aE "strict|refusing" /tmp/drill-c-strict.out; wait_health || echo "strict refused under pressure - CORRECT"
# release and verify zero residue:
cleanup_engine; rm -f /mnt/strata-huge/pressure.bin; sleep 2
echo "free after release: $(hp_free) (expect back to 45056 - store holdings)"
# re-arm clean:
( cd $REPO && nohup ./launch-custom-ud-iq2xs-8081.sh >/tmp/drill-c-clean.out 2>&1 & )
wait_health; grep -a "ARENA PAGES" /tmp/drill-c-clean.out "$REPO/logs/custom-ud-iq2_xs.log" | tail -2
```

**Expected:** the DRILL-A degrade chain fires (store deferral line, `4K - DEGRADED CLASS` banner with the MAP_HUGETLB note naming `needed 20164 ... vm.nr_hugepages=45056` — the note shows the *total* was fine and the *free* was not, which is exactly the competition signature); strict refusal + exit 70; after release, `HugePages_Free` returns to the pre-drill value ±2 pages; the clean re-boot re-arms 2 MiB.

**Failure signatures:**

- `dd` fails (ENOSPC) → the mount's `size=40G` cap is doing its job; hold less (e.g. 20 GiB) — the drill still lands free below the arena need with the store resident.
- Engine boots with a **partial** hugepage arena → impossible by construction (all-or-nothing, pinned.cu:263); if observed, the engine silently fell back without the banner — the loudness contract broke; stop and fix before any bench.
- Free does not return after `rm` → the hugetlbfs mount is holding freed pages (leak in the mount's accounting) — named kernel-level boundary, gather `ipcs -m` style evidence + `stat -f /mnt/strata-huge` before escalating.
- The un-migrated GSQ tax reappears on a *degraded* boot (this drill's serve-mode leg) → expected and documented; the number is a free AB data point for the pool-priority gates, record it in the drill log.

**PASS:** degrade chain loud under competition, no half-map, no residue, clean re-arm.

### DRILL-D — the A/B switch round-trip (`STRATA_NO_LARGEPAGES=1`)

**Purpose:** the one-boot A/B that fences the whole architecture: same binary, same window, large pages off → the 4K class and a zero pool draw; back on → the 2 MiB class and the full draw. This is also the standing evidence that the *pool* is what pays (law 1: the interleave decides; pinned.cu:122 documents the switch as "same run, same boot").

```bash
cd $REPO && . campaign/battery-lib.sh && cleanup_engine
# B leg (large pages off):
( cd $REPO && STRATA_NO_LARGEPAGES=1 nohup ./launch-custom-ud-iq2xs-8081.sh >/tmp/drill-d-b.out 2>&1 & )
wait_health && wait_prewarm
grep -aE "large pages skipped|ARENA PAGES" /tmp/drill-d-b.out "$REPO/logs/custom-ud-iq2_xs.log" | tail -3
HP0=$(hp_free); python3 campaign/probe-lane.py --port $PORT --label drill-d-B --short-n 6 | grep -E "M1|SHORT-CLASS"
HP1=$(hp_free); echo "B-leg pool draw: $((HP0-HP1)) (expect ~0: the flag skips the hugetlb try, pinned.cu:293)"
# A leg (standing):
cleanup_engine
( cd $REPO && nohup ./launch-custom-ud-iq2xs-8081.sh >/tmp/drill-d-a.out 2>&1 & )
wait_health && wait_prewarm
grep -aE "ARENA PAGES|hugetlb 2 MB" /tmp/drill-d-a.out "$REPO/logs/custom-ud-iq2_xs.log" | tail -3
python3 campaign/probe-lane.py --port $PORT --label drill-d-A --short-n 6 | grep -E "M1|SHORT-CLASS"
# round-trip = B then A in one window, twice (ABAB) if the first pass is the verdict leg:
```

**Expected:**

- B leg: `large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages` (pinned.cu:141/251 note) + `strata: ARENA PAGES: 4K - DEGRADED CLASS (39.4 GiB arena): ...`; pool draw ≈ 0; solo lands in the 4K class (day-1 measured the class gap: ~184 vs ~193 on UD, −5% class).
- A leg: the two 2 MiB lines; draw ≈ 20164 pages; solo back in bank class.
- ABAB interleaved (law 1): the A-B delta reproduces both rounds; that delta IS the live price of the architecture, recorded in the drill log.

**Failure signatures:**

- `2 MiB` banner with the flag set → env not reaching the binary (the switch is getenv-at-reserve; a wrapper swallowed it).
- A leg solo ≈ B leg solo → whichever side *thinks* it armed didn't: read the banners first (they exist exactly to fence this), then the pool draw numbers; a 2 MiB banner with a zero draw is a contradiction worth stopping for.
- B leg refuses to boot (strict) → expected in bench posture; in serve posture a no-boot is a defect (serve-degraded is the law-3 documented default).

**PASS:** both classes land where day-1 says they live, draws differ by the arena count, round-trip clean.

### 2.5 Drill → failure signature index (what each red means)

| symptom | drill that owns it | first probe |
|---|---|---|
| banner 4K on an armed boot | A (restore path) / B (re-arm) | `hp_total` + `hp_free` vs expected draw |
| store on /dev/shm without the deferral line | A | `stat -f -c %T /mnt/strata-huge` + pool gate read |
| strict served anyway | A/D | launcher env plumbing (`/proc/<pid>/environ` grep) |
| pool pages missing after stop/kill | B/C | free-count formula: total − store − holders |
| free does not return after release | C | mount accounting (`stat -f`), holder audit |
| solo −5% with a 2 MiB banner | B (first), then D | page class fenced out by banner; go to GPU/thermal witnesses |

---

## 3. The never-again watchdog (daily-runner hooks — diffs to write, not applied)

Design: one function appended to the daily runner's boot path (after `wait_prewarm`) + one verdict line per boot in the daily report. Every assertion is a grep or an awk on files that already exist — zero new instrumentation, ~40 lines total. The hooks fire on **every engine restart** (the runner boots several per battery), so a regression is caught **in one cycle** — the week-long slide that started this campaign cannot survive one.

```diff
--- a/campaign/batteries/daily-lane-runner.sh      (the runner's boot section)
+++ b/campaign/battersIES/../daily-lane-runner.sh
@@ after wait_prewarm, before the first probe leg:
+   # HOSTMEM watchdog (2026-11-07-preboot-aging-proof.md §3) — every boot asserts the re-arm.
+   . "$REPO/campaign/hostmem-watchdog.sh"     # to write; source-of-truth owner for all 6 hooks
+   hostmem_watchdog_pre   "$PACK_LOG"         # cf0, hp0, zram0, gauge score (engine-down already true here)
+   # ... existing boot + probe legs unchanged ...
+   hostmem_watchdog_post  "$PACK_LOG"         # banner class, store tier, draw, cf1, zram1, rss, vram
+   hostmem_watchdog_line  "$PACK" "$LABEL"    # the one daily-report line: GREEN/AMBER/RED + hook names
```

The hook body (to live in `campaign/hostmem-watchdog.sh`, single owner, battery-lib-grade — no copies):

| hook | assert | source of truth | red | amber |
|---|---|---|---|---|
| H1 banner class | UD: `strata: ARENA PAGES: 2 MiB (` in the pack log after boot; GSQ: `the expert complement: 37.59 GiB of 37.59 GiB in 2 MiB pages (HUGETLB POOL` | pinned.cu:427, day-1 verdict leg-3 lines | class line absent or `4K - DEGRADED CLASS` in serve mode | — (degraded serve is RED by definition here: with the pool armed, 4K is never the correct posture) |
| H1b strict (bench mode only) | bench boots carry `STRATA_ARENA_PAGES=strict`; a 4K class must end in the refusal + exit 70 | pinned.cu:435-437 | bench saw 4K without refusing | — |
| H2 pool draw | `draw = hp0.free − hp_free(after arena up)` within ±3% of the pack's expected count (UD 20164; GSQ complement 19246 + KV ~133; store +~19000 on a fresh-boot copy, 0 when fast-path) — and `HugePages_Free` never below floor `Total − (store + draw + 1000)` | /proc/meminfo; day-1 draw witnesses | draw off >3% or floor breached (foreign holder or silent shortfall) | draw 0 with a 2 MiB banner (flag armed? contradiction — stop) |
| H3 compact_fail | `cf1 − cf0 ≤ 1000` across boot + first probe cycle | /proc/vmstat; day-1 aged-box deltas in the thousands vs ~0 expected on the armed pool | > 1000 (the buddy is fighting again) | > 200 sustained across two cycles with lanes green |
| H4 zram fill | `/sys/block/zram0/mm_stat` byte 1 < 6.0 GiB (device 8G) | mm_stat; day-1: "zram refilling within days of being cleared" was an organ | > 7.0 GiB (full: reclaim storms next) | 6.0–7.0 with M4 proven (engine is MemorySwapMax=0 — the fill is desktop-owned, name it) |
| H5 store tier | `shm-shadow: tier /mnt/strata-huge (hugetlbfs)` and the `2 MiB by construction (pool draw` line present after any store (re)copy | shm-shadow.sh:20, :78 | tier line shows /dev/shm (the gate tripped: pool short at boot — run DRILL-A restore) | store fast-path skipped the copy (expected on warm boots — the draw witness H2 still must hold) |
| H6 gauge score | engine-down gauge ≥ 95% promoted (256 MiB collapse) | hostmem-anon-collapse.c; day-1 aged-box state was the ENOMEM/0% end | < 90% (the box lost PMD supply outside the pool — global THP dying again) | 90–95% once (re-check next cycle before reading a trend) |

Per-row record (one JSONL line per boot, `campaign/results/hostmem-watchdog.jsonl`, `type` field per OKF):

```json
{"type":"hostmem-watchdog-row","date":"2026-11-08","label":"daily-gsq-b1","pack":"gsq","verdict":"GREEN",
 "uptime_h":21.4,"hp_total":45056,"hp_free_after":5767,"draw":19246,"cf_delta":12,"zram_gib":5.1,
 "gauge_pct":100,"banner":"2 MiB","store":"hugetlbfs","rss_mib":512,"vram_idle_mib":1139,"load1":1.2}
```

**The daily-report line (the ONE line that catches the regression in a cycle):**

```
HOSTMEM WATCHDOG: GREEN  d2  uptime 45h  pool 45056 (draw gsq 19246 ud 20164)  cf+12  zram 5.1/8G  gauge 100%  store hugetlbfs
HOSTMEM WATCHDOG: RED    d5  uptime 117h [H1 banner=4K on ud-b2][H3 cf+48210] -> run DRILL-A (restore) + DRILL-B (return) before any bench
```

Rules that make one-cycle catching real:

1. The line prints for **every boot** (GREEN included) — silence is not evidence; a missing line is itself RED by the count check (`boots_seen == boots_logged`).
2. RED names the drill card inline — the response is pre-decided, not invented at 2 a.m.
3. Any RED voids banked reads from that boot window (law 17 hygiene: the box state was not the standing one) and books the re-run.
4. Weekly roll-up appended to `.docs/log.txt` by the logger (agents never hand-edit it): worst hook, trend of cf_delta and zram across the week — the same counters the ratchet's chart rows carry, so the watchdog series and the chart agree by construction.
5. After any kernel update or OS upgrade: re-run the full drill suite (A–D) once; the watchdog asserts the *installed* behavior, the drills re-prove it. The ratchet restarts its 14-day arc from the upgrade day (uptime claim resets — honesty over continuity).

---

## 4. Web precedents — how comparable programs prove memory-subsystem stability over weeks

**P1. Kernel `selftests/mm` (`run_vmtests.sh`) — the bounded per-boot suite.** The kernel's own mm regression harness is a single script that runs every mm selftest per boot, gating on kernel config (skips, not silent passes), with destructive tests deliberately separated after a fix ("fix destructive tests" thread) so a run cannot silently damage the host. *Stolen:* the split the ratchet uses — non-destructive hooks run on every boot (§3), destructive drills run deliberately in a booked window (§2); and the skip-vs-pass discipline: a hook that cannot run must print why, never pass quietly.

**P2. mmtests (Mel Gorman) — statistics over iterations, compare-not-eyeball.** Gorman's framework runs N iterations per kernel and compares kernel-vs-kernel with statistics (the project's "Higher or Lower" labels; openSUSE Hack Week 24 explicitly upgrades comparisons to confidence intervals). *Stolen:* the ratchet's verdict is a statistics read (pooled-mean band + Theil–Sen CI + Spearman), never "cycle 10 looked fine"; and mmtests' purpose — bisecting mm regressions — maps here to pinning a red watchdog day against the intervening commit/daily-change list, with the campaign's binary-ladder (law 10) as the bisect unit.

**P3. UNH-IOL DPDK Community Lab + Red Hat's `perf-regression-bisect` — automated bisect on a perf signal.** The lab runs automated per-patch functional + performance pipelines on bare-metal beds with perf-recheck robots across labs; Red Hat published a POC that auto-bisects a commit range to the regressing commit. *Stolen:* the bisect-on-fail posture — a RED watchdog row is not just an alert, it triggers the pre-named drill (§3 rule 2) and then the change-bisect; DPDK itself validates every stable/LTS branch by community test runs (the "validation" section of the stable policy) — the analogue is the 14-day ratchet re-armed after every kernel/OS change (§3 rule 5).

**P4. MongoDB `sys-perf` (Evergreen) — canary workloads + change-point detection filing tickets.** Every commit runs canary benchmark workloads; time-series results go through change-point detection (Daly 2020, arXiv:2003.00584) that filters noise and files a ticket on a step change (e.g., WiredTiger WT-17259: "1 regression ... across 1 test in 1 task"). *Stolen:* the 14-day lane series is our time series; the ±3% band + Spearman are the crude change-point detector, and the witness covariates in every row (uptime, load, zram, cf) are how a step gets *attributed* instead of just noticed — exactly the paper's noise-filtering problem on a box whose day-1 verdict measured ±8% window drift.

**P5. Soak-testing practice (RadView playbook, bugbug guide) — per-cycle counters at consistent intervals.** The standard method: sustained realistic load for hours-to-days, counter snapshots at consistent intervals (heap, connections, disk growth), compare end-state vs baseline per cycle, flag **resources not reclaimed between cycles** as the leak signal. *Stolen:* the six witnesses are per-cycle counters taken at fixed points (pre-boot engine-down, post-boot, post-probe); "not reclaimed between cycles" is DRILL-B's exact assertion (pool returns whole after SIGKILL — the crash case is a forced reclaim test).

**P6. PostgreSQL / Oracle / DPDK pool discipline (already graded A in the sibling digest `2026-11-07-hostmem-no-reboot-precedents.md`)** — PG fails fast on `huge_pages=on`, Oracle alerts on shortfall, DPDK fatal-at-init: the silent fallback is the bug that ships. *Stolen here:* the watchdog's RED-on-4K-in-serve-mode posture (H1) is that precedent, instrumented.

---

## 5. The honest boundary — what the pool does NOT cover, and the cheap tripwire for each

The claim is "the reboot bought nothing the pool now owns" — not "nothing on this box can ever degrade." Named residuals, each with a tripwire the watchdog already carries or one line adds:

| residual | why the pool can't cover it | cheap tripwire (owner: the watchdog row) |
|---|---|---|
| **Page cache for the 80–240 GiB pack files** (GGUFs stream from disk; the file tier reads the GGUF in place; the /dev/shm store tier is tmpfs RAM but the canonical copies are disk) | reserved pool ≠ page cache; cache evicts under desktop pressure regardless | the engine already prints per-request `file tier I/O ... (N major faults)`; watchdog asserts `major faults == 0` steady-state on warm boots (day-1 logs show 0 after warm-in); sustained N>0 = amber (cache thrash) + a `Cached` trend from /proc/meminfo per day |
| **GPU state** (VRAM fragmentation across restarts, GSP/driver-side allocations, ECC, clock policy) | the pool is host DRAM; the GPU side is the gpu-lock service's lane | per-cycle `nvidia-smi --query-gpu=memory.used,clocks.sm` at idle; idle VRAM growth >256 MiB across 10 cycles = amber; ECC query weekly (`ecc.errors.uncorrected.volatile.total` > 0 = RED, it is hardware) |
| **Driver/kernel leaks** (nvidia module pinned/allocated memory surviving engine restarts) | freed pool pages are kernel-accounted; driver-private allocations are not in `HugePages_*` | the same idle-VRAM witness is the net result; plus `cf_delta` trend (a driver fragmenting the buddy shows up in H3 within a cycle) |
| **Daemon growth** (server.py RSS, conversation-slot dir, logs) | userspace process growth is outside any pool | per-row `rss_mib` (server pid RSS) + slot-dir bytes weekly; idle RSS growing >1%/cycle monotonically 3+ cycles = amber (a leak the restarts should have reset — it didn't) |
| **zram refill for the desktop** (M4 opted the engine out; the desktop still swaps) | by design — zram serves non-engine | H4, with the M4 attribution note in the amber band |
| **Config drift** (`/etc/sysctl.d/91-strata-hugepages.conf` lost to an OS update or a setup script rewrite; the store unit unmounted) | the pool is only as real as the file that arms it | H2 asserts `HugePages_Total == 45056` every boot and H5 asserts the mount — a lost file is RED in one cycle, not one week |
| **Kernel behavior drift** (updates changing MADV_COLLAPSE, shmem, or hugetlb semantics — the root probe already proved docs can lie about this kernel) | no pool survives a semantics change | §3 rule 5: drill suite A–D re-runs after every kernel bump; H6's gauge is the canary (it directly measures collapse supply) |
| **Buddy supply OUTSIDE the pool** (the anon/THP side — the engine's small arenas, the desktop) | M2 raises proactiveness but the general zone still ages | H6 gauge score is exactly this measurement, every boot |
| **The IQ4_XS / DS configs** (94 GiB need > 88 GiB pool; DS boots have their own policy, undecided) | the standing rotation is sized for IQ2_XS + GSQ | the ratchet verdict is scoped to the standing packs; IQ4_XS/DS get their pool policy at their ladder legs (already named in HOSTMEM-NO-REBOOT.md sizing table) — the claim does not extend to them until their legs run |

**The boundary sentence for any public claim:** "on this box, for the standing Qwen rotation (IQ3_S, UD-IQ2_XS), engine restarts re-arm a boot-reserved 2 MiB pool with cycle-1 == cycle-N lanes over N cycles and 14 days of rising uptime, instrumented per cycle — page cache, GPU/driver state, and non-rotating configs remain separately watched."

---

## Sources

Local (read 2026-11-07): `campaign/HOSTMEM-NO-REBOOT.md`; `campaign/results/hostmem-day1-verdict.md`; `campaign/91-strata-hugepages.conf`; `campaign/strata-huge-store.service`; `campaign/shm-shadow.sh`; `campaign/hostmem-anon-collapse.c`; `campaign/battery-lib.sh`; `campaign/batteries/hostmem-leg1.sh`; `campaign/probe-lane.py`; `campaign/tracker/README.md` + `lanes.csv`; `src/core/pinned.cu` (banner + switch + pool-priority strings); `src/core/expert_source.cpp` (paced-complement gate); `launch-custom-ud-iq2xs-8081.sh` / `launch-custom-iq3s-8081.sh`; `AGENTS.md` laws 1/10/11/12/13/16/17; `.docs/research/digest/2026-11-07-hostmem-no-reboot-precedents.md`.

Web (searched 2026-11-07):

- Kernel selftests/mm and `run_vmtests.sh` unification: <https://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next.git/commit/?id=05f1edac80>; kselftest docs <https://www.kernel.org/doc/html/v6.5/dev-tools/kselftest.html>; destructive-tests fix thread: <https://lore.kernel.org/>; MADV contract tests: <https://lwn.net/Articles/1100803>; hugetlb admin doc: <https://docs.kernel.org/admin-guide/mm/hugetlbpage.html>
- mmtests: <https://github.com/gormanm/mmtests>; statistics upgrade (Hack Week 24): <https://hackweek.opensuse.org/24/projects/improve-mmtests-comparisons-and-reporting>
- UNH-IOL DPDK Community Lab: <https://www.iol.unh.edu/open-source-labs/dpdk>; deep dive: <https://www.iol.unh.edu/news/2023/10/18/enhancing-open-source-software-through-rigorous>; perf dashboard/rechecks: <https://dpdkdashboard.iol.unh.edu/results/dashboard/?page=1424&patchsets=all>; bisect POC: <https://github.com/RedHatPerf/perf-regression-bisect>; DPDK stable/LTS validation: <https://doc.dpdk.org/guides-26.07/contributing/stable.html>
- MongoDB change-point detection: <https://www.mongodb.com/company/blog/engineering/using-change-point-detection-find-performance-regressions>; paper: <https://arxiv.org/pdf/2003.00584>; example ticket: <https://jira.mongodb.org/browse/WT-17259>
- Soak-testing method (grade B/C — practice summaries, not primary sources): <https://www.radview.com/blog/soak-testing-software-playbook-memory-leak-detection-stability>; <https://bugbug.io/blog/software-testing/soak-testing>
