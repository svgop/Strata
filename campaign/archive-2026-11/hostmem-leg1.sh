#!/bin/bash
# campaign/batteries/hostmem-leg1.sh - HOSTMEM LEG 1 (law 17 BEFORE): the aged-box baseline for the
# M1-M4 battery (campaign/HOSTMEM-NO-REBOOT.md).  3 boots x probe-lane per pack, standing launchers,
# zero env changes, plus per-boot page-class witnesses: the arena/guard line from the pack log
# (which backing the boot took), the compact_fail delta across the boot, and the anon-collapse
# gauge taken with the engine down (the state the next boot faults into).
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
RES=$REPO/campaign/results
GAUGE=$REPO/campaign/hostmem-anon-collapse
[ -x "$GAUGE" ] || gcc -O2 -o "$GAUGE" "$REPO/campaign/hostmem-anon-collapse.c" || exit 1

cf() { awk '/^compact_fail/{print $2}' /proc/vmstat; }

for spec in "iq3s:launch-custom-iq3s-8081.sh:logs/custom-iq3_s.log" \
            "iq2xs:launch-custom-ud-iq2xs-8081.sh:logs/custom-ud-iq2_xs.log"; do
  IFS=: read -r pack launch log <<<"$spec"
  OUT=$RES/hostmem-leg1-$pack.log
  echo "HOSTMEM LEG1 $pack start $(date)" > "$OUT"
  for r in 1 2 3; do
    cleanup_engine
    f0=$(cf)
    echo "--- r$r gauge(engine-down): $("$GAUGE" 2>/dev/null | tail -1) | compact_fail=$f0" >> "$OUT"
    ( cd "$REPO" && nohup "./$launch" >"$RES/hostmem-leg1-$pack-r$r.out" 2>&1 & )
    ok=0 t=
    for t in $(seq 1 100); do
      [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
      sleep 3
    done
    echo "=== LEG hm1-$pack-r$r health=$ok $(date +%H:%M:%S) compact_fail+$(( $(cf) - f0 )) ===" >> "$OUT"
    [ "$ok" = 1 ] || { cleanup_engine; continue; }
    ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "hm1-$pack-r$r" --short-n 6 >> "$OUT" 2>&1 )
    grep -aE "expert arena|transparent huge" "$REPO/$log" | tail -2 >> "$OUT" 2>/dev/null
    cleanup_engine
  done
  echo "LEG1 $pack done $(date)" >> "$OUT"
done
echo "HOSTMEM LEG1 ALL DONE $(date)"
