#!/bin/bash
# F-209 battery driver: one boot + one probe-lane run. Usage: battery-boot.sh <label>
# Stops whatever serves 8081, boots the standing launcher fresh, waits for readiness,
# runs campaign/probe-lane.py, banks campaign/results/<label>.txt.
set -u
cd /home/svg/LLM/strata-custom-2
LABEL="$1"
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"

echo "[$(date +%H:%M:%S)] stopping any server on 8081..."
pkill -f 'serve/serve[r].py' 2>/dev/null
pkill -f 'build/strat[a] ' 2>/dev/null
pkill -f 'build/strat[a]$' 2>/dev/null
sleep 3
pkill -9 -f 'serve/serve[r].py' 2>/dev/null
pkill -9 -f 'build/strat[a] ' 2>/dev/null
pkill -9 -f 'build/strat[a]$' 2>/dev/null
sleep 2

echo "[$(date +%H:%M:%S)] booting standing launcher..."
nohup ./launch-custom-iq3s-8081.sh > server-8081.out 2>&1 &
disown

READY=0
for i in $(seq 1 240); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
         --max-time 3 http://127.0.0.1:8081/v1/models 2>/dev/null)
  if [ "$CODE" = "200" ]; then READY=1; break; fi
  sleep 3
done
if [ "$READY" != "1" ]; then
  echo "[$(date +%H:%M:%S)] SERVER NEVER READY (tail of server-8081.out):"
  tail -20 server-8081.out
  exit 1
fi
echo "[$(date +%H:%M:%S)] server ready; settling 10s..."
sleep 10

echo "[$(date +%H:%M:%S)] probe run $LABEL"
.venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL" \
  > "campaign/results/$LABEL.txt" 2>&1
RC=$?
grep -E 'M1 single-stream|M2 pair|M2 6-conc|M3b rate|TREND' "campaign/results/$LABEL.txt"
echo "[$(date +%H:%M:%S)] probe done rc=$RC"
exit $RC
