#!/bin/bash
# campaign/batteries/gsq-warm-verdict.sh - THE warm-residency verdict: one boot, prewarm-complete,
# 60 s settle, three back-to-back probes. Probe 3 = the bank-era-comparable number (long-lived warm
# server). Solo instrument: nothing else may own port 8081 while this runs.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-warm-verdict.log
echo "WARM VERDICT start $(date)" > "$OUT"
( nohup ./launch-custom-iq3s-8081.sh > campaign/results/gsq-warm-verdict-boot.out 2>&1 & )
for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 3; done
curl -s -m 3 http://127.0.0.1:8081/health >/dev/null || { echo "BOOT FAILED" >> "$OUT"; exit 1; }
T0=$(date +%s)
for t in $(seq 1 60); do grep -aq "page cache warm in" campaign/results/gsq-warm-verdict-boot.out && break; sleep 3; done
sleep 60
echo "== warm residency, boot+prewarm+settle $(( $(date +%s) - T0 ))s ==" >> "$OUT"
for i in 1 2 3; do
  echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
  python3 campaign/probe-lane.py --port 8081 --label warm-v-$i --short-n 6 >> "$OUT" 2>&1
done
grep -aE "hugepages: the expert complement|memfd, populated" campaign/results/gsq-warm-verdict-boot.out | head -2 >> "$OUT"
pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
sleep 2; pkill -x strata 2>/dev/null
echo "WARM VERDICT DONE $(date)" >> "$OUT"
echo "ALL DONE"
