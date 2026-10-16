#!/bin/bash
# Generic config-boot battery driver: boot serve/server.py with ANY config json, run the full
# probe-lane battery. Usage: battery-config.sh <config.json> <label>
set -u
cd /home/svg/LLM/strata-custom-2
CONFIG="$1"
LABEL="$2"
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
LOG=$(/home/svg/LLM/strata-custom-2/.venv/bin/python -c "import json;print(json.load(open('$CONFIG')).get('log','/home/svg/LLM/strata-custom-2/custom-iq3_s.log'))")

pkill -f 'serve/serve[r].py' 2>/dev/null; pkill -f 'build/strat[a] ' 2>/dev/null; pkill -f 'build/strat[a]$' 2>/dev/null
sleep 3
pkill -9 -f 'serve/serve[r].py' 2>/dev/null; pkill -9 -f 'build/strat[a] ' 2>/dev/null; pkill -9 -f 'build/strat[a]$' 2>/dev/null
sleep 2

# F-214 lesson: the standing env arms (SPEC_DRAFT_TEMP, BATCH_MTP, TYPICAL_ACCEPT, ...) live in
# launch-custom-iq3s-8081.sh's export block. Booting server.py directly DROPS them (both early
# "arms" ran near-stock and the M3b canary caught it at ~62%). Source the export lines in THIS
# shell so the nohup child inherits the full standing recipe; the config's own "env" dict then
# adds/overrides per-arm values on top.
eval "$(grep '^export ' launch-custom-iq3s-8081.sh)"
echo "[$(date +%H:%M:%S)] standing env loaded: SPEC_DRAFT_TEMP=$STRATA_SPEC_DRAFT_TEMP BATCH_MTP=$STRATA_BATCH_MTP TYPICAL_ACCEPT=$STRATA_TYPICAL_ACCEPT"

echo "[$(date +%H:%M:%S)] booting config=$CONFIG log=$LOG"
nohup .venv/bin/python serve/server.py --engine strata --config "$PWD/$CONFIG" --port 8081 --host 127.0.0.1 > server-cfgboot.out 2>&1 &
disown

READY=0
for i in $(seq 1 240); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
         --max-time 3 http://127.0.0.1:8081/v1/models 2>/dev/null)
  if [ "$CODE" = "200" ]; then READY=1; break; fi
  if ! pgrep -f 'serve/serve[r].py' > /dev/null; then
    echo "[$(date +%H:%M:%S)] SERVER DIED (tail):"; tail -15 server-cfgboot.out; exit 1
  fi
  sleep 3
done
[ "$READY" = "1" ] || { echo "NEVER READY"; tail -15 server-cfgboot.out; exit 1; }
echo "[$(date +%H:%M:%S)] ready; settling 10s"; sleep 10

.venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL" --log "$LOG" \
  > "campaign/results/$LABEL.txt" 2>&1
RC=$?
grep -E 'M1 single-stream|M2 pair|M2 6-conc|M3b rate|TREND' "campaign/results/$LABEL.txt"
echo "[$(date +%H:%M:%S)] probe done rc=$RC"
exit $RC
