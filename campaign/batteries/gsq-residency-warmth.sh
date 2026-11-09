#!/bin/bash
# campaign/batteries/gsq-residency-warmth.sh - does GSQ climb with continuous residency?
# One boot, probes at +3, +10, +20, +35 min after prewarm. The record's boots were 55min-4h24m
# resident; every battery today dies at 6-8 min. Released box, current binary, v5-range.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-residency-warmth.log
echo "RESIDENCY WARMTH start $(date)" > "$OUT"
( nohup ./launch-custom-iq3s-8081.sh > campaign/results/gsq-warmth-boot.out 2>&1 & )
for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 3; done
curl -s -m 3 http://127.0.0.1:8081/health >/dev/null || { echo "BOOT FAILED" >> "$OUT"; exit 1; }
T0=$(date +%s)
for t in $(seq 1 60); do grep -aq "page cache warm in" campaign/results/gsq-warmth-boot.out && break; sleep 3; done
for spec in 180 420 600 900; do
  now=$(date +%s); target=$((T0 + spec)); wait=$((target - now))
  [ $wait -gt 0 ] && sleep $wait
  echo "--- probe at +$(( ($(date +%s) - T0) / 60 )) min $(date +%H:%M:%S) ---" >> "$OUT"
  python3 campaign/probe-lane.py --port 8081 --label "warmth-$spec" --short-n 6 >> "$OUT" 2>&1
done
grep -a "strata batch:" logs/custom-iq3_s.log | tail -6 | grep -oE "avg [0-9.]+ rows|[0-9.]+ ms/window|GPU-reach wait [0-9.]+" >> "$OUT"
pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
sleep 2; pkill -9 -x strata 2>/dev/null
echo "RESIDENCY WARMTH DONE $(date)" >> "$OUT"
echo "ALL DONE"
