#!/bin/bash
# campaign/t21b-cohorts3.sh — COHORTS=3 battery, launcher copy at repo root (fixes t21's path bug).
set -u
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
OUT=$REPO/campaign/results/t21b-cohorts3.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T21B COHORTS3 start $(date)" > "$OUT"
port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }
cleanup_engine() {
  local p pid c
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *strata*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
}
boot() {
  cleanup_engine; sleep 2
  [ -n "$(port_pid)" ] && { echo "=== $2 ABORT port busy ===" >> "$OUT"; return 1; }
  ( cd "$REPO" && nohup "$1" >"$RES/t21b-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] && grep -i "cohorts" "$REPO/custom-iq3_s.log" | tail -1 >> "$OUT" 2>/dev/null
  [ "$ok" = 1 ]
}
leg() {
  boot "$1" "$2" || return
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t21b-$2" --short-n 6 >> "$OUT" 2>&1
  cleanup_engine
}
C3=$REPO/launch-custom-iq3s-8081-c3.sh
C2=$REPO/launch-custom-iq3s-8081.sh
for r in 1 2 3; do
  leg "$C3" "c3-arm-r$r"
  leg "$C2" "c2-ctl-r$r"
done
echo "T21B COHORTS3 end $(date)" >> "$OUT"
echo "done" >> "$OUT"
