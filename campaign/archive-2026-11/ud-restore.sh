#!/bin/bash
# campaign/batteries/ud-restore.sh — UD pack restore on the corrected instrument (reserved
# 2 MiB pool, 45076 pages): iq2xs + iq4xs, 4 boots each, wait_prewarm BEFORE every probe
# (campaign law 11: the warm-in is part of the bench), witness grep per boot, LAST-3-boot
# means banked per pack per lane (solo/pair/conc), ANCHOR COMPARISON in the log tail.
#
# PORT COLLISION LAW (2026-11-09): the GSQ banking battery (campaign/batteries/gsq-final.sh,
# 5 boots) owns 8081 until a "^DONE" line lands in campaign/results/gsq-final.log.  hold_gate
# runs BEFORE every boot AND before every cleanup_engine — a cleanup landing on their live leg
# voids it, so the gate is checked FIRST, always.  The gate passes only when DONE exists, no
# gsq-* battery process is alive, the port is free, and the pool has its pages back.
#
# ANCHOR GATE (one-shot, iq2xs only — the only pack with a Nov-8 pool-armed read): any lane
# >5% under 193.2/201.2/274.0 dumps that boot's witness lines and STOPS the battery.  No fix
# iteration: the instrument is what is under test, not the arms.
# iq4xs has no Nov-8 pool-armed anchor (the Nov-8 definitive battery ran iq3s + iq2xs only);
# its Nov-6 spot (176.8/196.5/238.5) is compared report-only, vintage named in the block.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
RES=$REPO/campaign/results
GSQ_GATE=$RES/gsq-final.log
WITNESS_RE='hugetlb 2 MB pages|memfd, populated|cache complement ready|MAP_HUGETLB unavailable|transparent huge pages NOT requested|exceeds available'
NOV6_IQ2=(226.5 240.0 329.7)
NOV8_IQ2=(193.2 201.2 274.0)
NOV6_IQ4=(176.8 196.5 238.5)

pool_free() { awk '/^HugePages_Free/{print $2}' /proc/meminfo; }

lane_line() {  # lane_line <probe-output> -> "solo=X pair=Y conc=Z" (0 = lane missing/failed)
  local s p c
  s=$(sed -n 's/^M1 single-stream *: \([0-9.]*\) t\/s.*/\1/p' "$1" | head -1)
  p=$(sed -n 's/^M2 pair *: \([0-9.]*\) t\/s.*/\1/p' "$1" | head -1)
  c=$(sed -n 's/^M2 [0-9]*-conc *: \([0-9.]*\) t\/s.*/\1/p' "$1" | head -1)
  echo "solo=${s:-0} pair=${p:-0} conc=${c:-0}"
}

hold_gate() {  # hold_gate <log> — block until the GSQ battery is done and 8081 is truly free
  local waited=0
  while :; do
    if grep -aq "^DONE" "$GSQ_GATE" 2>/dev/null \
       && [ -z "$(port_pid)" ] \
       && ! pgrep -f "campaign/batteries/gsq" >/dev/null 2>&1; then
      return 0
    fi
    [ $((waited % 5)) -eq 0 ] && echo "HOLD: gsq-final owns 8081 (DONE absent | battery alive | port busy) — re-check in 60 s at $(date +%H:%M:%S)" >> "$1"
    sleep 60
    waited=$((waited + 1))
  done
}

witness() {  # witness <engine-log> <byte-offset> — per-boot arm evidence on stdout
  tail -c +"$(( $2 + 1 ))" "$1" | grep -aE "$WITNESS_RE" | head -8
}

gate_check() {  # gate_check <solo> <pair> <conc> <a_solo> <a_pair> <a_conc> — trip text, empty = pass
  awk -v s="$1" -v p="$2" -v c="$3" -v a1="$4" -v a2="$5" -v a3="$6" 'BEGIN {
    d = 0
    if (s + 0 < 0.95 * a1) { printf "  solo %s t/s = %+.1f%% vs anchor %.1f (5%% floor %.1f)\n", s, (s - a1) / a1 * 100, a1, 0.95 * a1; d = 1 }
    if (p + 0 < 0.95 * a2) { printf "  pair %s t/s = %+.1f%% vs anchor %.1f (5%% floor %.1f)\n", p, (p - a2) / a2 * 100, a2, 0.95 * a2; d = 1 }
    if (c + 0 < 0.95 * a3) { printf "  conc %s t/s = %+.1f%% vs anchor %.1f (5%% floor %.1f)\n", c, (c - a3) / a3 * 100, a3, 0.95 * a3; d = 1 }
    exit d
  }'
}

run_pack() {  # run_pack <pack> <engine-log> <result-log>
  local pack=$1 elog=$2 out=$3 r t ok off hf line s p c trip w
  {
    echo "=== UD RESTORE $pack start $(date) ==="
    echo "instrument: reserved 2 MiB pool $(pool_free)/$(awk '/^HugePages_Total/{print $2}' /proc/meminfo) pages free; corrected-instrument law: wait_prewarm BEFORE every probe"
    echo "anchors: Nov-6 spot ${NOV6_IQ2[*]} / Nov-8 pool-armed ${NOV8_IQ2[*]} (iq2xs gate); Nov-6 spot ${NOV6_IQ4[*]} (iq4xs, report-only)"
  } >> "$out"
  for r in 1 2 3 4; do
    hold_gate "$out"                     # collision gate FIRST — never cleanup onto a live GSQ leg
    cleanup_engine
    hf=0                                 # pool gate: the launcher's single-owner branch needs >=17000
    for t in $(seq 1 10); do
      hf=$(pool_free); [ "$hf" -ge 17000 ] && break
      sleep 2
    done
    if [ "$hf" -lt 17000 ]; then
      echo "=== POOL GATE STOP $pack r$r: only $hf/45076 2 MiB pages free after cleanup — refusing a pool-short boot ===" >> "$out"
      echo "POOL GATE STOP"; exit 2
    fi
    off=$(wc -c < "$elog" 2>/dev/null || echo 0)
    ( cd "$REPO" && nohup ./launch-custom-ud-"$pack"-8081.sh >"$RES/ud-restore-$pack-r$r.out" 2>&1 & )
    ok=0
    for t in $(seq 1 100); do
      [ -n "$(curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" ] && { ok=1; break; }
      sleep 3
    done
    echo "=== BOOT ud-$pack-r$r health=$ok pool_free_at_boot=$hf $(date +%H:%M:%S) ===" >> "$out"
    if [ "$ok" != 1 ]; then
      echo "BOOT FAILED ud-$pack-r$r — see $RES/ud-restore-$pack-r$r.out" >> "$out"
      cleanup_engine
      continue
    fi
    wait_prewarm                         # corrected-instrument law: warm-in terminal BEFORE the probe
    ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "ud-$pack-r$r" --short-n 6 --log "$elog" ) > "$RES/ud-restore-$pack-r$r.probe" 2>&1
    cat "$RES/ud-restore-$pack-r$r.probe" >> "$out"
    echo "--- WITNESS ud-$pack-r$r (engine log since boot: arena / memfd store / complement + any arm-failure line) ---" >> "$out"
    w=$(witness "$elog" "$off")
    if [ -n "$w" ]; then printf '%s\n' "$w" >> "$out"; else echo "(no witness lines matched)" >> "$out"; fi
    line=$(lane_line "$RES/ud-restore-$pack-r$r.probe")
    echo "LANES ud-$pack-r$r $line" >> "$out"
    s=${line#solo=}; s=${s%% *}
    p=${line#*pair=}; p=${p%% *}
    c=${line#*conc=}; c=${c%% *}
    echo "ud-restore: $pack r$r lanes solo=$s pair=$p conc=$c"
    if [ "$pack" = iq2xs ]; then
      trip=$(gate_check "$s" "$p" "$c" "${NOV8_IQ2[0]}" "${NOV8_IQ2[1]}" "${NOV8_IQ2[2]}")
      if [ -n "$trip" ]; then
        {
          echo "=== ANCHOR GATE TRIP ud-$pack-r$r — lane(s) >5% under the Nov-8 pool-armed anchor; STOPPING (no fix iteration) ==="
          printf '%s\n' "$trip"
          echo "--- witness lines for the tripped boot ---"
          witness "$elog" "$off"
        } >> "$out"
        cleanup_engine
        echo "ANCHOR GATE TRIP ud-$pack-r$r — battery stopped"; exit 1
      fi
    else
      trip=$(gate_check "$s" "$p" "$c" "${NOV6_IQ4[0]}" "${NOV6_IQ4[1]}" "${NOV6_IQ4[2]}")
      if [ -n "$trip" ]; then
        {
          echo "FLAG ud-$pack-r$r >5% under the Nov-6 spot anchor (report-only — no Nov-8 pool-armed anchor exists for this pack):"
          printf '%s\n' "$trip"
        } >> "$out"
      fi
    fi
    cleanup_engine
  done
}

bank_pack() {  # bank_pack <pack> <result-log> <nov6_s> <nov6_p> <nov6_c> [nov8_s nov8_p nov8_c]
  local pack=$1 out=$2
  python3 - "$out" "$pack" "${@:3}" <<'PYEOF' >> "$out"
import re, sys
out, pack = sys.argv[1], sys.argv[2]
nov6 = [float(x) for x in sys.argv[3:6]]
nov8 = [float(x) if x not in ("", "-") else None for x in sys.argv[6:9]] if len(sys.argv) > 8 else [None] * 3
lanes = {}
pat = re.compile(r"^LANES ud-%s-r(\d) solo=(\S+) pair=(\S+) conc=(\S+)" % re.escape(pack))
for l in open(out, errors="replace"):
    m = pat.match(l)
    if m:
        lanes[int(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
names = ("solo", "pair", "conc")
rows = [lanes[i] for i in (2, 3, 4) if i in lanes]
print()
print("=== BANKED SUMMARY ud-%s (LAST-3 means, boots r2-r4; %d of 3 bankable boots) ===" % (pack, len(rows)))
for i in sorted(lanes):
    print("  r%d: solo %.1f | pair %.1f | conc %.1f t/s" % (i,) + lanes[i])
if len(rows) < 3:
    print("  NOT BANKABLE: fewer than 3 healthy boots — means below are informational only")
    sys.exit(0)
mean = [sum(r[j] for r in rows) / len(rows) for j in range(3)]
for j, nm in enumerate(names):
    print("  %-4s : %6.1f t/s" % (nm, mean[j]))
def pct(a, b):
    return "%+6.1f%%" % ((a - b) / b * 100.0)
print()
print("=== ANCHOR COMPARISON ud-%s ===" % pack)
hdr = "  lane   banked   Nov-6-spot   delta"
if nov8[0] is not None:
    hdr += "    Nov-8-pool-armed   delta"
print(hdr)
for j, nm in enumerate(names):
    row = "  %-4s  %6.1f    %8.1f    %s" % (nm, mean[j], nov6[j], pct(mean[j], nov6[j]))
    if nov8[j] is not None:
        row += "      %8.1f         %s" % (nov8[j], pct(mean[j], nov8[j]))
    print(row)
if nov8[0] is None:
    print("  (no Nov-8 pool-armed anchor exists for this pack; the Nov-6 spot predates the")
    print("   reserved pool, so deltas vs it conflate instrument change with pack change)")
flags = [names[j] for j in range(3)
         if (nov8[j] is not None and mean[j] < 0.95 * nov8[j])
         or (nov8[j] is None and mean[j] < 0.95 * nov6[j])]
if flags:
    print("  FLAG: %s >5%% under its anchor (report-only)" % ", ".join(flags))
else:
    print("  no lane >5%% under its anchor mean")
PYEOF
}

echo "ud-restore: waiting on the gsq-final collision gate before the first boot"
run_pack iq2xs "$REPO/logs/custom-ud-iq2_xs.log" "$RES/ud-restore-iq2xs.log" || exit 1
bank_pack iq2xs "$RES/ud-restore-iq2xs.log" "${NOV6_IQ2[@]}" "${NOV8_IQ2[@]}"
run_pack iq4xs "$REPO/logs/custom-ud-iq4xs.log" "$RES/ud-restore-iq4xs.log" || exit 1
bank_pack iq4xs "$RES/ud-restore-iq4xs.log" "${NOV6_IQ4[@]}"
echo "UD RESTORE done $(date); engines left DOWN"
