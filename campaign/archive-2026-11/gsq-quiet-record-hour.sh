#!/bin/bash
# campaign/batteries/gsq-quiet-record-hour.sh - R1: the quiet-window reproduction at the record's own
# hour (the proven +10-15% lever: the record era's own ordinary legs read 191-224 vs its spots 233).
# 5 boots prewarm-gated, 3 probes each, released box, standing binary.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-quiet-record-hour.log
echo "QUIET RECORD-HOUR R1 start $(date)" > "$OUT"
for r in 1 2 3 4 5; do
  local_before=$(wc -l < logs/custom-iq3_s.log)
  ( nohup ./launch-custom-iq3s-8081.sh > "campaign/results/qr-$r.out" 2>&1 & )
  ok=0 t=
  for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
  echo "=== R1 boot $r health=$ok $(date +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) ===" >> "$OUT"
  [ "$ok" = 1 ] || { pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null; continue; }
  for t in $(seq 1 60); do grep -aq "page cache warm in" "campaign/results/qr-$r.out" && break; sleep 3; done
  sleep 40
  for i in 1 2 3; do
    echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "qr-$r-$i" --short-n 6 >> "$OUT" 2>&1
  done
  pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
  sleep 2; pkill -9 -x strata 2>/dev/null; sleep 5
done
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
