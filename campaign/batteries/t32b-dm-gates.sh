#!/bin/bash
# campaign/batteries/t32b-dm-gates.sh — T32 extension QUALITY GATES for STRATA_DM=1
# adoption (the owed 2x MMLU + mimic). One boot, engine env passes through the
# standing launcher (no launcher edit until adoption); no binary deploy.
# Judge: MMLU-Pro mean >= 0.933 twice (F-224 mean rule), mimic in its band.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t32b-dm-gates.log
echo "T32B DM GATES start $(date)" > "$OUT"


cleanup_engine; sleep 2
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy before start" >> "$OUT"; exit 1; }
( cd "$REPO" && env STRATA_DM=1 nohup bash "$LAUNCH" >"$RES/t32b-dmgates.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT dm-gates health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  grep -a "strata dm" "$REPO/logs/custom-iq3_s.log" | tail -1 >> "$OUT" 2>/dev/null
  cd "$REPO"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t32b-mmlu-a >> "$OUT" 2>&1 || echo "MMLU A failed" >> "$OUT"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t32b-mmlu-b >> "$OUT" 2>&1 || echo "MMLU B failed" >> "$OUT"
  timeout 900 .venv/bin/python campaign/mimicbench2.py 8081 t32b-mimic >> "$OUT" 2>&1 || echo "MIMIC failed" >> "$OUT"
  cleanup_engine
fi
echo "T32B DM GATES end $(date)" >> "$OUT"
echo "done" >> "$OUT"
