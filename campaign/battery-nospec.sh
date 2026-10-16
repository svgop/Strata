#!/bin/bash
# No-spec parity battery: boots the standing engine with --spec 0 (campaign/nospec-iq3_s-8081.json),
# runs the M1-only probe (--skip-conc). Usage: battery-nospec.sh <label>
set -u
cd /home/svg/LLM/strata-custom-2
LABEL="$1"
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"

pkill -f 'serve/serve[r].py' 2>/dev/null; pkill -f 'build/strat[a] ' 2>/dev/null; pkill -f 'build/strat[a]$' 2>/dev/null
sleep 3
pkill -9 -f 'serve/serve[r].py' 2>/dev/null; pkill -9 -f 'build/strat[a] ' 2>/dev/null; pkill -9 -f 'build/strat[a]$' 2>/dev/null
sleep 2

echo "[$(date +%H:%M:%S)] booting no-spec engine..."
nohup .venv/bin/python serve/server.py --engine strata --config "$PWD/campaign/nospec-iq3_s-8081.json" --port 8081 --host 127.0.0.1 > server-nospec.out 2>&1 &
disown

READY=0
for i in $(seq 1 240); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
         --max-time 3 http://127.0.0.1:8081/v1/models 2>/dev/null)
  if [ "$CODE" = "200" ]; then READY=1; break; fi
  sleep 3
done
if [ "$READY" != "1" ]; then echo "NEVER READY"; tail -20 server-nospec.out; exit 1; fi
echo "[$(date +%H:%M:%S)] ready; settling 10s"; sleep 10

.venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL" \
  --log /home/svg/LLM/strata-custom-2/custom-iq3s-nospec.log --skip-conc \
  > "campaign/results/$LABEL.txt" 2>&1
grep -E 'M1 single-stream|fresh-long' "campaign/results/$LABEL.txt"
