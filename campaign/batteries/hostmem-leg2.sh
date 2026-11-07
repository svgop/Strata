#!/bin/bash
# campaign/batteries/hostmem-leg2.sh - HOSTMEM LEG 2: GSQ's complement INTO the pool (T35 C2 arm,
# STRATA_HUGEPAGE_ARENAS=1) vs same-config control, ABAB interleaved 2+2 (the leg-1 control read
# -12..-17% vs leg-0: partial-pool squeeze vs reservation-churn window - this leg discriminates).
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/hostmem-leg2-gsq.log
echo "HOSTMEM LEG2 (GSQ T35C2 ABAB) start $(date)" > "$OUT"

boot_probe() {  # boot_probe <tag> <env-prefix|-> 
  local tag=$1 envp=$2
  cleanup_engine
  if [ "$envp" = "-" ]; then
    ( cd "$REPO" && nohup ./launch-custom-iq3s-8081.sh >"$RES/hostmem-leg2-$tag.out" 2>&1 & )
  else
    ( cd "$REPO" && nohup env "$envp" ./launch-custom-iq3s-8081.sh >"$RES/hostmem-leg2-$tag.out" 2>&1 & )
  fi
  local ok=0 t=
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG hm2-$tag health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "hm2-$tag" --short-n 6 >> "$OUT" 2>&1 )
  grep -aE "expert complement|expert arena|hugepage|hugetlb|transparent huge" "$REPO/logs/custom-iq3_s.log" | tail -3 >> "$OUT" 2>/dev/null
  grep -aE "HUGEPAGE|hugepage_arenas" "$RES/hostmem-leg2-$tag.out" | head -3 >> "$OUT" 2>/dev/null
  cleanup_engine
}

boot_probe arm-r1 "STRATA_HUGEPAGE_ARENAS=1"
boot_probe ctl-r1 "-"
boot_probe arm-r2 "STRATA_HUGEPAGE_ARENAS=1"
boot_probe ctl-r2 "-"
echo "LEG2 done $(date)" >> "$OUT"
echo "HOSTMEM LEG2 ALL DONE $(date)"
