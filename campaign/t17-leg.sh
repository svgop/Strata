#!/bin/bash
# T17 Step-0 leg driver (free-boot slate, 2026-10-17 night): stop whatever holds 8081,
# verify VRAM drained, boot serve/server.py with ANY config json on the STANDING env
# (F-214 lesson: eval the launcher's export lines; the config's own "env" dict overrides
# per-arm values on top - server.py:2811 applies it AFTER os.environ), run one probe-lane
# battery, bank campaign/results/<label>.txt.
# Usage: t17-leg.sh <config.json> <label> [post-boot-hook.sh]
# The optional post-boot hook runs between readiness and the probe (fast-warm package).
set -u
cd /home/svg/LLM/strata-custom-2
CONFIG="$1"
LABEL="$2"
HOOK="${3:-}"
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
LOG=$(/home/svg/LLM/strata-custom-2/.venv/bin/python -c "import json;print(json.load(open('$CONFIG')).get('log','/home/svg/LLM/strata-custom-2/custom-iq3_s.log'))")

echo "[$(date +%H:%M:%S)] leg $LABEL: stopping anything on 8081..."
pkill -f 'serve/serve[r].py' 2>/dev/null; pkill -f 'engine/strat[a] --serve' 2>/dev/null
sleep 4
pkill -9 -f 'serve/serve[r].py' 2>/dev/null; pkill -9 -f 'engine/strat[a] --serve' 2>/dev/null
# VRAM drain check (the law: one engine at a time, verify before booting the next)
for i in $(seq 1 40); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$USED" -lt 2000 ] && break
  sleep 3
done
echo "[$(date +%H:%M:%S)] VRAM after stop: ${USED} MiB"
if [ "${USED:-9999}" -ge 2000 ]; then
  echo "[$(date +%H:%M:%S)] ABORT: VRAM did not drain (something still holds the GPU)"; exit 1
fi

eval "$(grep '^export ' launch-custom-iq3s-8081.sh)"
echo "[$(date +%H:%M:%S)] standing env loaded (COHORTS=$STRATA_COHORTS BATCH_MTP=$STRATA_BATCH_MTP DRAFT_TEMP=$STRATA_SPEC_DRAFT_TEMP)"

bash campaign/prewarm.sh
echo "[$(date +%H:%M:%S)] booting config=$CONFIG log=$LOG"
nohup .venv/bin/python serve/server.py --engine strata --config "$PWD/$CONFIG" --port 8081 --host 127.0.0.1 > "server-t17leg-$LABEL.out" 2>&1 &
disown

READY=0
for i in $(seq 1 240); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
         --max-time 3 http://127.0.0.1:8081/v1/models 2>/dev/null)
  if [ "$CODE" = "200" ]; then READY=1; break; fi
  if ! pgrep -f 'serve/serve[r].py' > /dev/null; then
    echo "[$(date +%H:%M:%S)] SERVER DIED (tail):"; tail -15 "server-t17leg-$LABEL.out"; exit 1
  fi
  sleep 3
done
[ "$READY" = "1" ] || { echo "[$(date +%H:%M:%S)] NEVER READY"; tail -15 "server-t17leg-$LABEL.out"; exit 1; }
echo "[$(date +%H:%M:%S)] ready; settling 10s"; sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed "s/^/[$(date +%H:%M:%S)] VRAM in-boot: /"
# F-205 law: verify the ARM env reached the engine process (config env -> server.py:2811)
EPID=$(pgrep -f 'engine/strat[a] --serve' | head -1)
if [ -n "$EPID" ]; then
  tr '\0' '\n' < "/proc/$EPID/environ" 2>/dev/null | grep -E 'STRATA_(DUMMY_NODES|DRAFT_TIMING|SPEC_PROB|LOOKAHEAD_STATS|BATCH_MTP|COHORTS|SPEC_DRAFT_TEMP)' \
    | sed "s/^/[$(date +%H:%M:%S)] engine env: /"
fi

if [ -n "$HOOK" ]; then
  echo "[$(date +%H:%M:%S)] post-boot hook: $HOOK"
  bash "$HOOK" "$LOG" || echo "WARN: hook rc=$? (continuing)"
fi

echo "[$(date +%H:%M:%S)] probe run $LABEL"
.venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL" --log "$LOG" \
  > "campaign/results/$LABEL.txt" 2>&1
RC=$?
grep -E 'M1 single-stream|M2 pair|M2 6-conc|M3b rate|TREND' "campaign/results/$LABEL.txt"
echo "[$(date +%H:%M:%S)] leg $LABEL done rc=$RC"
exit $RC
