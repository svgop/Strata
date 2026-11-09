---
type: research-digest
id: 2026-11-09-gsq-recovery-matrix
date: 2026-11-09
scope: the closing matrix for GSQ full-speed recovery — evidence table (record vs today), the one-session ordered battery that resolves the governor-pin lever (a), the long-residency lever (b) and the off-by-default verify arms (c), the chart-row plan per the law-12 repair runbook, and the executable runbook with the decision tree
subject: from the verified 206.6/199.0/291.0 released-state bank toward the 233.1/224.8/303.5 record class — or the proof the record is condition-bounded
method: read-only planning pass — campaign/tracker/lanes.csv, campaign/results/{era-interleave.log,gsq-warm-verdict.log}, digests 2026-11-09-regression-clock-box / -v5-memclock-governor / -regression-residency / -regression-bench-flow / -regression-code-audit / -chart-law12-repair / -bootstorm-fixes, git log c4b0d2de..35653c90, campaign/battery-lib.sh, launch-custom-iq3s-8081.sh, campaign/strata-gpu-lock-v5, src/core/verify.cpp arm gates. No boot, no GPU state, no file outside this digest touched.
---

# GSQ recovery closing matrix — 2026-11-09

## Verdict (read this first)

Three levers can still move the released box toward the record class, and ONE session prices
all three: **(a) the v6 dual hard-pin** (SM min=max 3135 + MEM min=max 14001 while listening —
the exact v2 sustained state; v5's single-axis MEM pin is already falsified, commit 35653c90:
it held SM at 2004 MHz), **(b) residency to +50 min** (the record boots lived 55 min–4 h 24 m;
today's same-boot slope is already +9..+15 % from probe 1 to probe 3; the tail beyond 15 min
is unpriced), **(c) two unpriced off-by-default verify arms** (STRATA_DOORBELL_STORE=1 —
CUDA-wired at verify.cpp:2112, it directly attacks the 12.27 ms GPU-reach/doorbell serial
term; STRATA_VERIFY_QDEDUP=1). The binary is exonerated (c4b0d2de intraday interleave:
current ≥ era on all three lanes), host memory favors today (+0.3/+7.1/+10.6 % vs the anchor),
and a slice of the record gap is favorable-window selection that no lever recovers. Run A
first: if the pin adopts, B and C must run under it or their numbers die at install time.

**Order: A (governor, ~15 min, the only sudo) → B (residency ladder, ~65 min, no sudo) →
C (verify arms, ~15-22 min, no sudo) → close.** Total ≈ 2 h 00-20 m; 7-8 compound sudo
touchpoints, all in A and the close.

---

## 1. Evidence table — record vs today, and the named differences

### 1.1 Lane table

| point | date/instrument | solo | pair | conc | vs record (s/p/c) |
|---|---|---|---|---|---|
| **RECORD** Nov-6 HEAD spot (2-leg means; 10:33+11:51 boots; RETIRED as a *baseline*, still the record) | lanes.csv 129-131 | 233.1 | 224.8 | 303.5 | — |
| Record's sibling night spot (03:05-03:09, v2, 2-leg) | lanes.csv 123-125 | 229.1 | 220.7 | 301.6 | −1.7/−1.8/−0.6 % |
| **RELEASED-STATE bank** (35653c90, pool=0, warm probes 2-3, kernel 7.2.15) | lanes.csv 163-165 | 206.6 | 199.0 | 291.0 | **−11.4/−11.5/−4.1 %** |
| Intraday interleave, current binary (c4b0d2de; 2 boots × 3 probes, desktop-loaded afternoon) | era-interleave.log | 195.8 | 186.5 | 272.7 | −16.0/−17.0/−10.2 % |
| Intraday interleave, banked-era binary (4152f114 rebuilt exact) | era-interleave.log | 193.7 | 185.4 | 269.7 | −16.9/−17.5/−11.2 % |
| Best same-afternoon singles | era-cur1-2 / warm-v-3 / era-cur2-2 | 215.4 | 199.9 | 294.9 | −7.6/−11.1/−2.8 % |
| Legal anchor leg-0 (Nov-7, aged box) | hostmem-leg0-verdict | 206.3 | 186.7 | 258.9 | released bank reads +0.3/+7.1/+10.6 % |

### 1.2 Named differences and expected impact

| # | difference | record era | today | expected impact |
|---|---|---|---|---|
| 1 | **Governor while processing** | v2 hard-pin SM 3135 + MEM 14001 (single-value `-lgc`/`-lmc`) | v5 range SM 1800-3135, MEM 7001-14001 (`MEM_PIN=0` after the operator kill) | Sustained probe clocks are IDENTICAL both eras, ~2830-2955 MHz cap-trimmed (grade-A, Nov-7 dmon) — sustained is a non-delta; mem ceiling 13801 vs 14001 = −1.4 %. The open term is the burst-entry mem bin: v2 held 14001 through every burst, v4/v5 DVFS floats 7001-13801 and the ramp vs 13-14 ms decode bursts is unmeasured. v5's MEM-only pin FALSIFIED (SM held 2004; 35653c90). v6 dual-pin is untested → block A prices it; bar ≥ +8 % |
| 2 | **Engine residency** | boots lived 55 min and 4 h 24 m; prewarm complete before any scored window (journal 13:11→14:06, 14:18→18:41) | 2.5-6 min legs; gsq-clean showed `prewarm: page cache warm in 18 s` landing after ~11 scored requests | Measured same-boot slope today: solo 179.9→207.0 (+15.1 %), pair 181.4→199.9 (+10.2 %), conc 262.6→286.2 (+9.0 %) across probes 1→3 within 4 min (gsq-warm-verdict.log). The +20/+35/+50 min tail is unpriced → block B. If it keeps climbing, the "record" is residency-class; estimate +8-15 % solo-class, UNPROVEN |
| 3 | **Window selection** | 2-leg spot composite in a manually-quiet window; the SAME DAY's ordinary legs read 191-224 solo = −4..−17 % under the trio (fork-ud-first-bank law-17 legs; provenance digest) | 3-probe means, law-17 cross-window drift ±8 % | A slice of the 11-17 % gap is favorable-window selection, not physics — **unrecoverable by any lever**. Estimate 5-10 points. The matrix's honest target is record-class under a documented recipe, never the unauditable composite itself |
| 4 | Engine binary | 4152f114-era | current (c4b0d2de) | **EXONERATED**: interleaved E-C-E-C same afternoon, current ≥ era on all three lanes (195.8/186.5/272.7 vs 193.7/185.4/269.7, inside noise) |
| 5 | Host memory | day-5 boot: order-9 supply collapsing 18→0 GiB, zram near-full, global OOM 18:41 that evening | released pool=0, full 184 GiB general, fresh kernel 7.2.15, zram 3.8/8 | Favors today (released bank +0.3/+7.1/+10.6 % vs anchor). Non-lever |
| 6 | Desktop | manually quieted | load ~3-3.9 with engine up | Worst case measured −55 % (Chrome leg); ordinary desktop rides inside ±8 %. Schedule the decisive probes in a quiet window. Estimate, controlled by scheduling |
| 7 | **Serial-lane anatomy** | same | M1 window 13-14 ms carrying 12.27 ms GPU-reach (doorbell) wait on the serial path; pair/conc overlap windows and hide it | Bounds the solo lane. NO lever in {a,b,c} removes the doorbell term — if A/B/C all miss, (d) kVerifyMaxT 8→12 (T65 widening; T13 #5: +31-36 % conc arithmetic, L-cost rebuild) and (e) the conditional one-doorbell design (days) are the only structural paths |

---

## 2. The ONE-SESSION closing matrix

### 2.1 Blocks, order, cost

| block | resolves | wall | sudo | boots | probes |
|---|---|---|---|---|---|
| 0 pre-flight | standing-state witnesses | 5 min | 0 | 0 | 0 |
| **A governor v6 falsifier** | (a) | ~15 min | 7 compound ops | 1 | 1 throwaway + 6 banked, R-P-R-P-R-P interleave |
| **B residency ladder** | (b) | ~65 min | 0 | 1 | 8 (3 at +3, 1 each +10/+20/+35, 3 at +50) |
| **C verify arms** | (c) | ~15-22 min | 0 | 2 (+1 optional) | 3 warm each |
| close | adoption/restore, final_serve, rows | ~15 min | 0-1 | 1 (final_serve) | 0 |

Order law: **A first** — if the pin clears ≥ +8 %, B and C run under the pin (their numbers
must describe the state that would actually be deployed); a pin adopted after B/C would
invalidate them. B second — its +3 slot is the same-session fresh-residency control that C's
screening boots read against. C last — screening under the session's final posture.

### 2.2 Block A — the v6 dual-pin falsifier (same boot, interleaved phases)

- **Design:** one boot; 6 banked probes alternating RANGE (installed v5 behavior, `MEM_PIN=0`,
  `SM_PIN=0`) and PIN (`SM 3135,3135` + `MEM 14001,14001` while listening — exact v2 sustained
  state, but with v4's release-on-gone so the v2 1780-plateau defect is not reused). Phases
  flip via sysconfig + `systemctl reload` (the HUP trap re-sources and re-applies; v5 digest
  proves reload lands the flip without a restart). The interleave kills the warm-up drift that
  a block design (R-R-R then P-P-P) would confound; probe 1 of the boot is a throwaway (the
  known first-probe artifact: 179.9 vs 206/207 in warm-verdict).
- **The v6 script** = v5 + an `SM_PIN` knob, byte-identical everywhere else (same 6-hunk shape
  that v5 used for `MEM_PIN`). Step A1 below writes it; `bash -n` + a `DRY=1` smoke + `diff -u`
  review gate the install.
- **Witnesses per phase:** `journalctl -t strata-gpu-lock` line naming the mode (`mem range
  7001-14001` vs `mem HARD PIN 14001`, `watching v6`); one `nvidia-smi dmon -s pucvmet` capture
  per phase during an M1 window (banks the sustained-clock proof the clock-box audit could not
  take statically; kill-line below reads it).
- **Kill-lines:** health red → void boot, re-boot once, second failure aborts the block. dmon
  under PIN showing sustained SM < 2600 MHz → the pin collapsed again (v5's 2004 failure mode)
  → PIN phase void, **v6 rejected at the clock level** without spending the lane read. Any
  lane < 150 solo on a healthy boot → box-state red, abort the session.
- **Decision (pre-registered ≥ +8 % rule):** mean(PIN trio) ≥ 1.08 × mean(RANGE trio) on any
  lane AND no lane below −2 % → **v6 ADOPTED** (leave pins=1 standing, note row + mean rows).
  Otherwise **v6 REJECTED** (restore pins=0; with 35653c90 this closes the entire mem-pin
  lever class with prejudice — the clock governor is permanently non-explanatory on this box).

### 2.3 Block B — the residency ladder (one boot, engine stays resident)

- Slots at **+3 / +10 / +20 / +35 / +50 min** after prewarm terminal; **3 warm probes at +3 and
  +50** (the decision slots, law-17), 1 probe at each intermediate slot (the curve shape).
  Between slots: no traffic, desktop untouched — idle residency is the record condition being
  tested. The in-flight `gsq-residency-warmth.sh` run (launched 16:20, single probes at
  +3..+15) folds in as preliminary curve evidence only; the matrix boot is the decision.
- **Decision:** mean(+50 trio) ≥ 1.08 × mean(+3 trio) → **residency-class confirmed**: adopt
  the standing-runner posture (final_serve stays up; the record is documented as
  residency-class; future banks bank under the standing-runner recipe). Otherwise the warmth
  lever is bounded at the measured first-15-min slope and the record's long boots were not the
  mechanism (or are not transferable).
- Kill-lines: as §2.2 (health, prewarm terminal before every probe — law 11, witness lines).

### 2.4 Block C — off-by-default verify arms, one boot each

- **Witness only (no boot, 30 s):** the standing arms on the released box. The launcher already
  exports `STRATA_VERIFY_DEVICE_PLAN=1` (verify.cpp:1140, engine-default off; TODO 1838 marks
  the "arm it" falsifier VOID — it was never off) and `STRATA_VERIFY_ASYNC_LAUNCH=1` (F-200).
  Record the env witness line; no boot is spent on them.
- **C1 `STRATA_DOORBELL_STORE=1`** — the step's ring STORED by the GPU instead of
  read-modify-written over PCIe (verify.cpp:2112, `#649 A/B`); the only unpriced arm that
  targets the 12.27 ms GPU-reach/doorbell serial term directly. 1 boot, 3 warm probes.
  Changes the publish path → **golden/mimic bit-parity owed before any adoption** (law-13
  class); the boot-time hc selftest rides every boot.
- **C2 `STRATA_VERIFY_QDEDUP=1`** — verify.cpp:162/2148 gate, off by default, never priced on
  this tree. 1 boot, 3 warm probes.
- Optional C3 `STRATA_VERIFY_COHERENT=1` only if time remains (weak: HIP-flavored coherency
  posture, verify.cpp:226).
- **Screen bar (each arm):** any lane ≥ +8 % vs B's +3 trio AND no lane < −2 % → graduate to a
  3v3 confirm battery + parity/MMLU gate (its own later session). Otherwise park with the
  one-boot read banked in the verdict digest — no chart row.

### 2.5 The decision tree

| result | branch |
|---|---|
| A pin ≥ +8 % (any lane, no lane < −2 %) | **v6 install** (pins=1 standing; sysconfig carries `SM_PIN=1 MEM_PIN=1`); note row + 6 mean rows; B and C ran under pin; commit `campaign/strata-gpu-lock-v6`; rollback documented (`SM_PIN=0 MEM_PIN=0` + reload) |
| A pin < +8 % or dmon collapse | **v6 rejected**; restore pins=0 (1 sudo); with 35653c90 the mem-pin class is closed with prejudice; the record is NOT clock-recoverable |
| B +50 ≥ +8 % over +3 | **residency-class documented**: standing runner keeps the engine resident; note row; the record class is residency-bounded; future banks bank on long-resident boots |
| B +50 < +8 % | warmth bounded at the first-15-min slope; record's long boots not the transferable mechanism |
| neither A nor B fires | the remaining gap = doorbell serial term + window selection → **(d) kVerifyMaxT 8→12 import is the next session** (T65 widening; run the free MTP=0 batch-8 config falsifier first, T13 #5) and **(e) the conditional one-doorbell design is the only structural path** (days); the 233.1/224.8/303.5 record is documented **condition-bounded forever**: favorable window + v2 dual-pin + 55 min-4 h residency + quiet desktop, on a binary proven equal to today's |
| any C arm ≥ +8 % | graduate to confirm battery + golden/MMLU gate; else park, digest-only |

---

## 3. Chart plan (per 2026-11-09-chart-law12-repair.md §4)

The anchor block and the three engine-fix `kind=note` rows are ALREADY appended (lanes.csv
lines 153-159). The matrix appends only measurement rows and adoption notes — order: bank in
results → commit battery+results → append csv → regen → commit tracker.

| rows | kind | lane rows | label grammar (comma-free; `;` clauses; tps = trio mean, 1 decimal) |
|---|---|---|---|
| A RANGE phase | `mean` | 3 | `GSQ v6 falsifier RANGE phase (v5 sysconfig pins=0; same-boot R-P-R-P-R-P interleave; trio <s1>/<s2>/<s3>; dmon sustained SM <x> MEM <y>; vs anchor <d%>; witness mem range 7001-14001)` |
| A PIN phase | `mean` | 3 | `GSQ v6 falsifier PIN phase (SM min=max 3135 + MEM min=max 14001 while listening; same boot as the range phase; trio <...>; dmon sustained SM <x> MEM <y>; delta vs range <d1>/<d2>/<d3> pct; verdict by the >=+8 pct rule)` |
| A adoption | `note` | 1, only on ADOPT | `NOTE governor v6 ADOPTED: dual hard-pin SM min=max 3135 + MEM min=max 14001 while listening (falsifier <d1>/<d2>/<d3> pct same-boot interleave; sustained dmon <x>/<y>; rollback SM_PIN=0 + MEM_PIN=0 + reload)` — a REJECT lands digest-only, no row (the runbook: notes carry adopted changes) |
| B +50 trio | `mean` | 3 | `GSQ residency ladder +50 min spot (<governor mode>; trio <...> vs the same-boot +3 trio <...> = <d%>; slots +10/+20/+35 singles <...>; record boots were 55min-4h24m resident)` |
| B adoption | `note` | 1, only on residency-class | `NOTE residency-class documented: +50 min residency <d%> over +3 min on one boot; standing runner keeps the engine resident; the record class is residency-bounded` |
| C arms | none | 0 | parked reads live in the verdict digest; a graduated arm's confirm battery lands its own rows later |

Regen after append: `python3 campaign/tracker/build-trend.py && python3
campaign/tracker/build-packcompare.py`; read against the anchor block (±2 % = parity;
declining lanes = law-12 response ladder), never against the retired 233 composite.

---

## 4. RUNBOOK — execute step by step

All paths absolute-rooted at `/home/svg/LLM/strata-custom-2`; run from the repo root. Sudo
appears only in A and the close, one compound op each.

### Step 0 — pre-flight (5 min, no sudo)

```bash
cd /home/svg/LLM/strata-custom-2
git log --oneline -1                                    # expect c4b0d2de
cmp engine/strata /tmp/strata.current && echo BINARY=OK # current binary standing
cat /proc/sys/vm/nr_hugepages                           # expect 0 (released state)
journalctl -t strata-gpu-lock -n 2 --no-pager           # expect watching v5 ... mem range 7001-14001
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # expect empty
ss -ltn | grep 8081 || echo PORT=FREE
uptime                                                  # note load; decisive probes want a quiet window
mkdir -p campaign/results/gsq-close
```

Abort conditions: pool non-zero, governor banner not v5-range, port owned, compute apps up.

### Step A1 — build + install v6 (2 compound sudo ops)

```bash
cp campaign/strata-gpu-lock-v5 campaign/strata-gpu-lock-v6
python3 - campaign/strata-gpu-lock-v6 <<'EOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = s.replace("watching v5", "watching v6", 1)
s = s.replace("MEM_PIN=1        # 1 = hard-pin mem at MEM_MAX while listening (v5); 0 = v4 range behavior",
              "MEM_PIN=1        # 1 = hard-pin mem at MEM_MAX while listening (v5/v6); 0 = v4 range behavior\n"
              "SM_PIN=0         # v6: 1 = also hard-pin SM at SM_MAX while listening (min=max); 0 = v4/v5 range", 1)
s = s.replace("""mem_mode() {   # journal-facing description of the active mem regime
    if [ "$MEM_PIN" = 1 ]; then printf 'HARD PIN %s' "$MEM_MAX"
    else printf 'range %s-%s' "$MEM_FLOOR" "$MEM_MAX"; fi
}""",
"""mem_mode() {   # journal-facing description of the active mem regime
    if [ "$MEM_PIN" = 1 ]; then printf 'HARD PIN %s' "$MEM_MAX"
    else printf 'range %s-%s' "$MEM_FLOOR" "$MEM_MAX"; fi
}
sm_mode() {    # v6: journal-facing description of the active SM regime
    if [ "$SM_PIN" = 1 ]; then printf 'HARD PIN %s' "$SM_MAX"
    else printf 'range %s-%s' "$SM_FLOOR" "$SM_MAX"; fi
}""", 1)
s = s.replace('               "$NSMI" -i "$DEV" -lgc "$SM_FLOOR,$SM_MAX" >/dev/null 2>&1 || return 1',
              '               if [ "$SM_PIN" = 1 ]; then\n'
              '                   "$NSMI" -i "$DEV" -lgc "$SM_MAX,$SM_MAX" >/dev/null 2>&1 || return 1\n'
              '               else\n'
              '                   "$NSMI" -i "$DEV" -lgc "$SM_FLOOR,$SM_MAX" >/dev/null 2>&1 || return 1\n'
              '               fi', 1)
s = s.replace("      MEM_PIN=${MEM_PIN:-1}", "      MEM_PIN=${MEM_PIN:-1}\n      SM_PIN=${SM_PIN:-0}", 1)
s = s.replace('log "reload — floor re-applied (${SM_FLOOR}-${SM_MAX} sm, $(mem_mode) mem)"',
              'log "reload — floor re-applied ($(sm_mode) sm, $(mem_mode) mem)"', 1)
open(p, "w").write(s)
EOF
bash -n campaign/strata-gpu-lock-v6 && echo SYNTAX=OK
diff -u campaign/strata-gpu-lock-v5 campaign/strata-gpu-lock-v6 | head -60   # review: 6 hunks, nothing else
# DRY smoke both branches (no GPU state; fake listener on :18099, v5-digest pattern):
python3 -c 'import socket,time; s=socket.socket(); s.bind(("127.0.0.1",18099)); s.listen(1); time.sleep(4)' &
LISTEN_PID=$!
DRY=1 PORTS=18099 SM_PIN=1 MEM_PIN=1 timeout 5 bash campaign/strata-gpu-lock-v6 2>&1 | sed "s/^/[v6-smoke-pin] /"
wait $LISTEN_PID 2>/dev/null
sed 's/^SM_PIN=0/SM_PIN=1/' campaign/strata-gpu-lock-v6 > /tmp/v6-smoke-range   # NOT installed; syntax check only
bash -n /tmp/v6-smoke-range && echo SMOKE-SYNTAX=OK; rm -f /tmp/v6-smoke-range
sudo bash -c 'cp /home/svg/LLM/strata-custom-2/campaign/strata-gpu-lock-v6 /usr/libexec/strata-gpu-lock && systemctl restart strata-gpu-lock && sleep 1 && journalctl -t strata-gpu-lock -n 1 --no-pager'
# expect: watching v6 (range 1800-3135 sm / mem range 7001-14001 ...)  <- pins still 0 = RANGE control
```

### Step A2 — one boot, 1 throwaway + 6 interleaved probes (5 compound sudo flips)

```bash
OUT=campaign/results/gsq-close/gsq-close-governor.log; echo "GOV FALSIFIER start $(date)" > "$OUT"
( nohup ./launch-custom-iq3s-8081.sh > campaign/results/gsq-close/gov-boot.out 2>&1 & )
for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 3; done
curl -s -m 3 http://127.0.0.1:8081/health >/dev/null || { echo "BOOT FAILED" >> "$OUT"; exit 1; }
for t in $(seq 1 60); do grep -aq "page cache warm in" campaign/results/gsq-close/gov-boot.out && break; sleep 3; done
sleep 45
python3 campaign/probe-lane.py --port 8081 --label gov-warmup --short-n 6 >> "$OUT" 2>&1   # THROWAWAY
flip() {  # flip <0|1> — one compound sudo per phase change; witness line required
  sudo bash -c "printf 'SM_CLOCK=3135\nSM_MID=1800\nIDLE_GRACE=30\nACT_UTIL=10\nACT_POWER=150\nMEM_CLOCK=14001\nMEM_PIN=$1\nSM_PIN=$1\n' > /etc/sysconfig/strata-gpu-lock && systemctl reload strata-gpu-lock && sleep 1 && journalctl -t strata-gpu-lock -n 1 --no-pager" | tee -a "$OUT"
}
dmon() { nvidia-smi dmon -s pucvmet -c 60 -f "campaign/results/gsq-close/dmon-$1.txt" & }
for ph in R P R P R P; do
  [ "$ph" = P ] && flip 1 || flip 0
  sleep 12
  [ "$ph" = R ] && dmon R; [ "$ph" = P ] && dmon P
  echo "--- probe $ph $(date +%H:%M:%S) ---" >> "$OUT"
  python3 campaign/probe-lane.py --port 8081 --label "gov-$ph-$(date +%H%M%S)" --short-n 6 >> "$OUT" 2>&1
done
grep -aE "M1 single-stream|M2 pair|M2 6-conc" "$OUT"
pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s"; sleep 2; pkill -x strata
grep -aE "the expert complement|memfd, populated|page cache warm" campaign/results/gsq-close/gov-boot.out | head -3 >> "$OUT"
```

Read: `mean(P trio) vs mean(R trio)` per lane (M1/pair/conc lines). Apply §2.2 decision +
kill-lines (check `dmon-R.txt`/`dmon-P.txt` sustained SM: P < 2600 → reject at clock level).

### Step B — residency ladder (one boot, ~65 min, no sudo; start immediately after A)

```bash
cat > campaign/batteries/gsq-close-residency.sh <<'EOF'
#!/bin/bash
# one boot; 3 probes at +3, 1 at +10/+20/+35, 3 at +50 after prewarm terminal; idle between slots
set -u; cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-close/gsq-close-residency.log; echo "RESIDENCY LADDER start $(date)" > "$OUT"
( nohup ./launch-custom-iq3s-8081.sh > campaign/results/gsq-close/res-boot.out 2>&1 & )
for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 3; done
curl -s -m 3 http://127.0.0.1:8081/health >/dev/null || { echo "BOOT FAILED" >> "$OUT"; exit 1; }
T0=$(date +%s)
for t in $(seq 1 60); do grep -aq "page cache warm in" campaign/results/gsq-close/res-boot.out && break; sleep 3; done
sleep 60; T0=$(date +%s)   # T0 = prewarm terminal + settle
slot() { local spec=$1 n=$2 tag=$3 now target wait i
  now=$(date +%s); target=$((T0 + spec)); wait=$((target - now))
  [ $wait -gt 0 ] && sleep $wait
  for i in $(seq 1 "$n"); do
    echo "--- probe +$(( ($(date +%s) - T0) / 60 ))min $tag-$i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "res-$tag-$i" --short-n 6 >> "$OUT" 2>&1
  done; }
slot 180 3 early; slot 600 1 m10; slot 1200 1 m20; slot 2100 1 m35; slot 3000 3 late
grep -aE "the expert complement|memfd, populated|page cache warm" campaign/results/gsq-close/res-boot.out | head -3 >> "$OUT"
grep -a "TREND ms/window" "$OUT" | tail -2 >> "$OUT"
pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s"; sleep 2; pkill -x strata
echo "RESIDENCY LADDER DONE $(date)" >> "$OUT"; echo ALL DONE
EOF
chmod +x campaign/batteries/gsq-close-residency.sh
nohup bash campaign/batteries/gsq-close-residency.sh > /dev/null 2>&1 &
# returns ~65 min later; read mean(late trio) vs mean(early trio) per §2.3
```

### Step C — verify arms, one boot each (no sudo; run after B)

```bash
arm() { local tag=$1 envv=$2 OUT=campaign/results/gsq-close/gsq-close-arm-$tag.log
  echo "ARM $tag=$envv start $(date)" > "$OUT"
  ( nohup env "$envv" ./launch-custom-iq3s-8081.sh > campaign/results/gsq-close/arm-$tag-boot.out 2>&1 & )
  for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 3; done
  curl -s -m 3 http://127.0.0.1:8081/health >/dev/null || { echo "BOOT FAILED" >> "$OUT"; return; }
  for t in $(seq 1 60); do grep -aq "page cache warm in" campaign/results/gsq-close/arm-$tag-boot.out && break; sleep 3; done
  sleep 45
  for i in 1 2 3; do echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "arm-$tag-$i" --short-n 6 >> "$OUT" 2>&1; done
  grep -aE "the expert complement|memfd, populated" campaign/results/gsq-close/arm-$tag-boot.out | head -2 >> "$OUT"
  pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s"; sleep 2; pkill -x strata; sleep 5; }
# standing-arm witness (no boot): DEVICE_PLAN and ASYNC_LAUNCH are launcher-standing
grep -n "STRATA_VERIFY_DEVICE_PLAN\|STRATA_VERIFY_ASYNC_LAUNCH" launch-custom-iq3s-8081.sh | tee -a campaign/results/gsq-close/standing-arms-witness.txt
arm bellstore STRATA_DOORBELL_STORE=1
arm qdedup STRATA_VERIFY_QDEDUP=1
# optional third: arm coherent STRATA_VERIFY_COHERENT=1
```

Screen each vs B's early trio (§2.4 bar).

### Step D — close

```bash
# 1. governor end-state: ADOPTED -> nothing (pins=1 standing); REJECTED -> one sudo:
sudo bash -c 'printf "SM_CLOCK=3135\nSM_MID=1800\nIDLE_GRACE=30\nACT_UTIL=10\nACT_POWER=150\nMEM_CLOCK=14001\nMEM_PIN=0\nSM_PIN=0\n" > /etc/sysconfig/strata-gpu-lock && systemctl reload strata-gpu-lock && journalctl -t strata-gpu-lock -n 1 --no-pager'
# 2. standing server for the operator (keeps residency if B adopted the class):
( cd /home/svg/LLM/strata-custom-2 && nohup bash launch-custom-iq3s-8081.sh > campaign/results/gsq-close/final-serve.out 2>&1 & )
# 3. verdict digest + rows: write .docs/research/digest/2026-11-09-gsq-close-verdict.md,
#    commit battery+results FIRST, then append §3 rows to campaign/tracker/lanes.csv,
#    regen both builders, commit the tracker (chart-law12 §4 order).
git add campaign/batteries/gsq-close-residency.sh campaign/strata-gpu-lock-v6 campaign/results/gsq-close && git commit -m "gsq closing matrix: governor falsifier + residency ladder + verify-arm screens; harness committed"
# ... csv append, regen, tracker commit per law-12 §4 ...
```

### Boundaries named

- The falsifier flips governor state 5-7 times on the live box — every flip is journaled and
  witnessed; the sysconfig file is rewritten whole each flip (no append-drift).
- `dmon` sampling is 1 Hz; a 13-14 ms burst class is read through its sustained envelope, not
  per-burst. The kill-line reads sustained SM, which is exactly the envelope the v2-era proof
  used.
- Block B idle slots assume the desktop stays off the GPU; a Chrome-class load during a slot
  voids that slot (kill-line: any lane < 150).
- STRATA_VERIFY_QDEDUP/DOORBELL_STORE are single-boot screens under the ±8 % intraday band;
  a sub-bar read PARKS the arm, it never convicts it — only the ≥ +8 % graduate earns a
  confirm battery.
- The 262144-ctx and 18.7 GiB expert-cache config deltas vs the Nov-6 131072/19.0 GiB boots
  are standing engine config, priced in the anchor era; the matrix does not vary them.

## Provenance

- Read: `campaign/tracker/lanes.csv` (rows 123-165 + anchor block), `campaign/results/
  era-interleave.log`, `campaign/results/gsq-warm-verdict.log`, `campaign/results/
  gsq-residency-warmth.log` (in-flight, 16:20 launch), `.docs/research/digest/
  {2026-11-09-regression-clock-box,2026-11-09-v5-memclock-governor,2026-11-09-regression-residency,
  2026-11-09-regression-bench-flow,2026-11-09-regression-code-audit,2026-11-09-chart-law12-repair,
  2026-11-09-bootstorm-fixes}.md`, `campaign/battery-lib.sh`, `launch-custom-iq3s-8081.sh`,
  `campaign/batteries/{gsq-warm-verdict,gsq-residency-warmth,era-interleave,gsq-final,
  gsq-final-bisect}.sh`, `campaign/strata-gpu-lock-v5` (apply()/HUP/banner hunks),
  `src/core/verify.cpp` (arm gates at 162/226/228/1113/1140/2112/3081), `SVG-CUSTOM-TODO.md`
  (T13 #5, device-plan VOID note), git c4b0d2de / dc694709 / 35653c90 messages.
- Live checks: governor journal tail (v5 mem range live, 15:05-16:21), /tmp/clk-floor.txt and
  /tmp/clk-pin.txt absent (the single-probe falsifier never ran — this matrix supersedes it
  with the 3-probe interleave), engine/strata vs /tmp/strata.current.
- Wrote: this digest only.
