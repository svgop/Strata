#!/bin/bash
# campaign/rebank-3135.sh — five fresh standing boots at the 3135 lock, full probe each.
# The F-237 owed re-bank: M1/pair/conc means + acceptance at the locked clocks.
# Boots are serial; each leg is a fresh engine start. Port guard makes double-writer
# collisions fail loudly instead of silently corrupting a leg.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
OUT=$REPO/campaign/results/rebank3-mmvqil.log
mkdir -p "$REPO/campaign/results"
echo "REBANK3 start $(date)" > "$OUT"

port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }

cleanup_engine() {
  local p
  p=$(port_pid)
  [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  local pid c
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *strata*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
}

for i in 1 2 3 4 5; do
  cleanup_engine
  sleep 3
  if [ -n "$(port_pid)" ]; then
    echo "=== RUN $i ABORT: port $PORT occupied by foreign process ===" >> "$OUT"
    continue
  fi
  ( cd "$REPO" && nohup "$LAUNCH" >/dev/null 2>&1 & )
  ok=0
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== RUN $i health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  if [ "$ok" != 1 ]; then
    echo "RUN $i FAILED TO BOOT - skipping" >> "$OUT"
    continue
  fi
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "rebank3-r$i" --short-n 6 >> "$OUT" 2>&1
  cleanup_engine
done
echo "REBANK3 end $(date)" >> "$OUT"
echo "done" >> "$OUT"
