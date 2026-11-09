#!/bin/bash
# campaign/batteries/gsq-v6-verdict.sh - the v6 pinned-regime verdict: 3 boots, warm probes,
# vs tonight's range baseline (last-3 210.7/189.6/276.1).  Rule: >=+8% solo, no lane worse -> stands.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-v6-verdict.log
echo "V6 VERDICT start $(date)" > "$OUT"
for r in 1 2 3; do
  ( nohup ./launch-custom-iq3s-8081.sh > "campaign/results/v6-$r.out" 2>&1 & )
  ok=0 t=
  for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
  echo "=== V6 boot $r health=$ok $(date +%H:%M:%S) load=$(cut -d' ' -f1 /proc/loadavg) ===" >> "$OUT"
  [ "$ok" = 1 ] || { pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null; continue; }
  for t in $(seq 1 60); do grep -aq "page cache warm in" "campaign/results/v6-$r.out" && break; sleep 3; done
  sleep 40
  nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.mem --format=csv,noheader >> "$OUT"
  for i in 1 2 3; do
    echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "v6-$r-$i" --short-n 6 >> "$OUT" 2>&1
  done
  pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
  sleep 2; pkill -9 -x strata 2>/dev/null; sleep 5
done
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
