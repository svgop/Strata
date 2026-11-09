#!/bin/bash
# campaign/batteries/hostmem-gsq-clean.sh - GSQ-only re-run after the pool-blind budget fix
# (e8484c36): 3 boots, witnesses with the NEW page-class patterns, law-17 vs leg-0's bank.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/hostmem-gsq-clean.log
echo "GSQ CLEAN (post budget fix) start $(date)" > "$OUT"
for r in 1 2 3; do
  cleanup_engine
  before=$(wc -l < "$REPO/logs/custom-iq3_s.log")
  ( cd "$REPO" && nohup ./launch-custom-iq3s-8081.sh >"$RES/gsq-clean-r$r.out" 2>&1 & )
  ok=0 t=
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG gsq-clean-r$r health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; continue; }
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "gsq-clean-r$r" --short-n 6 >> "$OUT" 2>&1 )
  tail -n +$before "$REPO/logs/custom-iq3_s.log" | grep -aE "hugepages: the|complement ready|memfd|ARENA PAGES|resident complement|exceeds available" | head -6 >> "$OUT"
  cleanup_engine
done
echo "GSQ CLEAN done $(date)" >> "$OUT"
echo "ALL DONE $(date)"
