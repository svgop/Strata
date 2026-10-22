#!/bin/bash
# campaign/t30-discriminate.sh — 3-arm same-window battery after the proof battery failed to confirm:
#   NEW  = frac 0.55 + share 1.0 + spec 7 (current standing)
#   OLDS7= frac 0.35 + share 0.5 + spec 7 (is spec7's win real on the OLD base?)
#   OLD  = frac 0.35 + share 0.5 + spec 6 (pre-T28)
# 3 rounds x 3 arms, interleaved.
set -u
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
OUT=$REPO/campaign/results/t30-discriminate.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T30 DISCRIMINATE start $(date)" > "$OUT"
port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }
cleanup_engine() {
  local p pid c
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *strata*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
  sleep 2
}
boot() {  # $1 launcher $2 label
  cleanup_engine
  [ -n "$(port_pid)" ] && { echo "=== $2 ABORT port busy ===" >> "$OUT"; return 1; }
  ( cd "$REPO" && nohup "$1" >"$RES/t30-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 120); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ]
}
leg() {
  boot "$1" "$2" || return
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t30-$2" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t30-$2" 2>&1 | tee -a "$OUT" )
  cleanup_engine
}
for r in 1 2 3; do
  leg "$REPO/launch-olds7.sh" "olds7-r$r"
  leg "$REPO/launch-custom-iq3s-8081.sh" "new-r$r"
  leg "$REPO/launch-preT28.sh" "old-r$r"
done
echo "T30 DISCRIMINATE end $(date)" >> "$OUT"
echo "done" >> "$OUT"
