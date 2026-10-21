#!/bin/bash
# campaign/t24-s1-battery.sh — T24-S1 dense_tc 3v3 ABAB (ctl = standing launcher, arm = STRATA_DENSE_TC=1).
# Same binary (engine/strata with the port default-off), same pack; the only delta is the env on the arm.
# Pattern: campaign/t21b-cohorts3.sh. Launcher copies at repo root (launch law).
set -u
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
OUT=$REPO/campaign/results/t24-s1-battery.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T24-S1 BATTERY start $(date)" > "$OUT"
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
  ( cd "$REPO" && nohup "$1" >"$RES/t24s1-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  # the arm banner AFTER this leg's probe (lazy flush) lands in the log with the leg's verdict below
  [ "$ok" = 1 ]
}
leg() {
  boot "$1" "$2" || return
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t24s1-$2" --short-n 6 >> "$OUT" 2>&1
  grep "strata dense_tc:" "$REPO/custom-iq3_s.log" | tail -1 >> "$OUT" 2>/dev/null
  cleanup_engine
}
CTL=$REPO/launch-custom-iq3s-8081.sh
ARM=$REPO/launch-custom-iq3s-8081-t24s1.sh
for r in 1 2 3; do
  leg "$CTL" "ctl-r$r"
  leg "$ARM" "arm-r$r"
done
echo "T24-S1 BATTERY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
