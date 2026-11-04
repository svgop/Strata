#!/bin/bash
# campaign/batteries/t41-prob-gates.sh — QUALITY GATES for STRATA_SPEC_PROB=1 adoption:
# the battery read M1 +5.8% (2/2 legs above every ctl leg), conc +4.4%, M3b +2.0pp
# (mechanism-correlated). The arm changes the token stream on sampled traffic ->
# 2x MMLU (floor 0.933) + mimic (band 58-62/66) MANDATORY. One boot, env passthrough.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t41-prob-gates.log
echo "T41 PROB GATES start $(date)" > "$OUT"


cleanup_engine; sleep 2
[ -n "$(port_pid)" ] && { echo "ABORT: port busy" >> "$OUT"; exit 1; }
( cd "$REPO" && env STRATA_SPEC_PROB=1 nohup bash "$LAUNCH" >"$RES/t41-prob-gates.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT prob-gates health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  cd "$REPO"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t41-prob-mmlu-a >> "$OUT" 2>&1 || echo "MMLU A failed" >> "$OUT"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t41-prob-mmlu-b >> "$OUT" 2>&1 || echo "MMLU B failed" >> "$OUT"
  timeout 900 .venv/bin/python campaign/mimicbench2.py 8081 t41-prob-mimic >> "$OUT" 2>&1 || echo "MIMIC failed" >> "$OUT"
  cleanup_engine
fi
echo "T41 PROB GATES end $(date)" >> "$OUT"
echo "done" >> "$OUT"
