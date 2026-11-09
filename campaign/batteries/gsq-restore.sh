#!/bin/bash
# campaign/batteries/gsq-restore.sh - the restore battery: single-owner store + full complement,
# 5 boots, warm (boot 1 warms the pack), banked on the LAST 3.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/gsq-restore.log
echo "GSQ RESTORE start $(date)" > "$OUT"
for r in 1 2 3 4 5; do
  cleanup_engine
  before=$(wc -l < "$REPO/logs/custom-iq3_s.log")
  ( cd "$REPO" && nohup ./launch-custom-iq3s-8081.sh >"$RES/gsq-restore-r$r.out" 2>&1 & )
  ok=0 t=
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG gsq-restore-r$r health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; continue; }
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "gsq-restore-r$r" --short-n 6 >> "$OUT" 2>&1 )
  tail -n +$before "$REPO/logs/custom-iq3_s.log" | grep -aE "hugepages: the expert complement|memfd, populated|resident complement" | head -3 >> "$OUT"
  cleanup_engine
done
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
