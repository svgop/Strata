#!/bin/bash
# campaign/stock-jury.sh - the stock tree's own GSQ runner (run-iq3s-stock.sh, port 8080), same
# probe instrument, warm (90 s settle after load; stock has no prewarm).  Port 8081 must be dark.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/stock-jury.log
echo "STOCK JURY start $(date)" > "$OUT"
( nohup /home/svg/LLM/strata/run-iq3s-stock.sh > campaign/results/stock-jury-boot.out 2>&1 & )
for t in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 3; done
curl -s -m 3 http://127.0.0.1:8080/health >/dev/null || { echo "STOCK BOOT FAILED" >> "$OUT"; exit 1; }
sleep 90
echo "== stock warm probes ==" >> "$OUT"
for i in 1 2 3; do
  echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
  python3 campaign/probe-lane.py --port 8080 --label stock-jury-$i --short-n 6 >> "$OUT" 2>&1
done
pkill -f "serve/server.py" 2>/dev/null; sleep 2; pkill -9 -x strata 2>/dev/null
echo "STOCK JURY DONE $(date)" >> "$OUT"
echo "ALL DONE"
