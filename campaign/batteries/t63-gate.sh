#!/bin/bash
# campaign/batteries/t63-gate.sh - the T63 adoption gate: 2x MMLU + mimic on a ts7 boot.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t63-gate.log
echo "T63 GATE start $(date)" > "$OUT"


cleanup_engine
( cd "$REPO" && env STRATA_ROUTE_TAIL_SKIP=7 nohup bash "$LAUNCH" >"$RES/t63-gate-boot.out" 2>&1 & )
ok=0
for t in $(seq 1 120); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT ts7 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  grep -m1 "TAIL_SKIP" "$REPO/logs/custom-iq3_s.log" >> "$OUT" 2>&1 || true
  wait_prewarm
  cd "$REPO"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t63-ts7-mmlu-a >> "$OUT" 2>&1 || echo "MMLU A failed" >> "$OUT"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t63-ts7-mmlu-b >> "$OUT" 2>&1 || echo "MMLU B failed" >> "$OUT"
  timeout 600 .venv/bin/python campaign/mimicbench2.py 8081 t63-ts7-mimic >> "$OUT" 2>&1 || echo "MIMIC failed" >> "$OUT"
fi
cleanup_engine

# standing daily driver back up (tail-skip OFF until adoption is signed)
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t63-post-gate-serve.out" 2>&1 & )
for t in $(seq 1 120); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && break
  sleep 3
done
echo "SERVER LEFT RUNNING $(date +%H:%M:%S)" >> "$OUT"
echo "T63 GATE end $(date)" >> "$OUT"
