#!/bin/bash
# tqllm battery driver: one tqserve boot + one probe-lane run. Usage: battery-tqllm.sh <label> <pln|mtp>
#   pln = --np 6, all lanes (no MTP: their MTP is np-1 only)
#   mtp = --mtp --np 1, M1-only (--skip-conc)
# Frees the GPU by stopping our standing engine first (restored later via battery-boot.sh).
set -u
cd /home/svg/LLM/tqllm
LABEL="$1"; MODE="${2:-pln}"
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
MODEL=/home/svg/LLM/Custom-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-MERGED.gguf

NP=6; EXTRA=""; PROBEFLAGS=""
if [ "$MODE" = "mtp" ]; then NP=1; EXTRA="--mtp"; PROBEFLAGS="--skip-conc"; fi

echo "[$(date +%H:%M:%S)] freeing GPU (our engine + any tqserve)..."
pkill -f 'serve/serve[r].py' 2>/dev/null; pkill -f 'build/strat[a] ' 2>/dev/null; pkill -f 'build/strat[a]$' 2>/dev/null
pkill -f 'build/tqserv[e]' 2>/dev/null
sleep 3
pkill -9 -f 'serve/serve[r].py' 2>/dev/null; pkill -9 -f 'build/strat[a] ' 2>/dev/null; pkill -9 -f 'build/strat[a]$' 2>/dev/null
pkill -9 -f 'build/tqserv[e]' 2>/dev/null
sleep 2

echo "[$(date +%H:%M:%S)] booting tqserve (mode=$MODE np=$NP extra=$EXTRA)..."
nohup ./build/tqserve --model "$MODEL" --host 127.0.0.1 --port 8086 --api-key "$KEY" \
  --np $NP --kv q8 -ncmoe 48 --moe-cache-mb 16000 $EXTRA \
  > tqllm-server.out 2>&1 &
disown

READY=0
for i in $(seq 1 200); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
         --max-time 3 http://127.0.0.1:8086/v1/models 2>/dev/null)
  if [ "$CODE" = "200" ]; then READY=1; break; fi
  # bail early if the process died
  if ! pgrep -f 'build/tqserv[e]' > /dev/null; then
    echo "[$(date +%H:%M:%S)] TQSERVE DIED (tail of tqllm-server.out):"
    tail -30 tqllm-server.out
    exit 1
  fi
  sleep 3
done
if [ "$READY" != "1" ]; then
  echo "[$(date +%H:%M:%S)] SERVER NEVER READY (tail of tqllm-server.out):"
  tail -30 tqllm-server.out
  exit 1
fi
echo "[$(date +%H:%M:%S)] tqserve ready; settling 10s..."
sleep 10

echo "[$(date +%H:%M:%S)] probe run $LABEL"
/home/svg/LLM/strata-custom-2/.venv/bin/python /home/svg/LLM/strata-custom-2/campaign/probe-lane.py \
  --port 8086 --label "$LABEL" --log /dev/null $PROBEFLAGS \
  > "/home/svg/LLM/strata-custom-2/campaign/results/$LABEL.txt" 2>&1
RC=$?
grep -E 'M1 single-stream|M2 pair|M2 6-conc|warmup' "/home/svg/LLM/strata-custom-2/campaign/results/$LABEL.txt" || true
echo "[$(date +%H:%M:%S)] probe done rc=$RC"
exit $RC
