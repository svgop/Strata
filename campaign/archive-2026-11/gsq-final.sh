#!/bin/bash
# campaign/batteries/gsq-final.sh - the corrected-instrument verdict: single-owner store + full
# complement + prewarm COMPLETE before every probe. 5 boots, last 3 banked (boot 1 warms).
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/gsq-final.log
echo "GSQ FINAL (corrected instrument) start $(date)" > "$OUT"
for r in 1 2 3 4 5; do
  cleanup_engine
  before=$(wc -l < "$REPO/logs/custom-iq3_s.log")
  ( cd "$REPO" && nohup ./launch-custom-iq3s-8081.sh >"$RES/gsq-final-r$r.out" 2>&1 & )
  ok=0 t=
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG gsq-final-r$r health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; continue; }
  wait_prewarm
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "gsq-final-r$r" --short-n 6 >> "$OUT" 2>&1 )
  tail -n +$before "$REPO/logs/custom-iq3_s.log" | grep -aE "hugepages: the expert complement|memfd, populated|resident complement|exceeds available" | head -3 >> "$OUT"
  cleanup_engine
done
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
