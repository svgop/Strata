#!/bin/bash
# campaign/t30-proof.sh — THE COMPOSED-GAIN PROOF: pre-T28 recipe (frac 0.35 + share 0.5 + spec 6)
# vs the new standing (frac 0.55 + share 1.0 + spec 7). 5v5 ABAB, same window, both probes.
# This is the no-regression proof the campaign owes after three adoptions.
set -u
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
OUT=$REPO/campaign/results/t30-proof.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T30 PROOF (preT28 vs T28-recipe) start $(date)" > "$OUT"
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
leg() {  # $1 launcher $2 label
  boot "$1" "$2" || return
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t30-$2" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t30-$2" 2>&1 | tee -a "$OUT" )
  cleanup_engine
}
OLD=$REPO/launch-preT28.sh
NEW=$REPO/launch-custom-iq3s-8081.sh
for r in 1 2 3 4 5; do
  leg "$NEW" "new-r$r"
  leg "$OLD" "old-r$r"
done
echo "T30 PROOF end $(date)" >> "$OUT"
echo "done" >> "$OUT"
