#!/bin/bash
# campaign/batteries/t49-ccdsplit.sh — F-295: the CCD thread-class split judged 3v3 ABAB.
#   ctl = standing (pool workers anywhere)    ccd = STRATA_POOL_CORES="8-15,24-31" (CCD1)
# The feed class gets a WHOLE CCD (16 logical) — isolation by abundance, not the falsified
# whole-tree scarcity pin.  Judge: M1/pair/conc pooled + the window-tax question (can we
# recover the measured +8.5-14.8%?).  Server restored after.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t49-ccdsplit.log
echo "T49 CCD SPLIT start $(date)" > "$OUT"


[ -n "$(port_pid)" ] && { echo "ABORT: port busy" >> "$OUT"; exit 1; }

run_leg() {  # $1 arm, $2 round
  local lbl="t49-$1-r$2"
  local env_pass=""
  [ "$1" = ccd ] && env_pass="STRATA_POOL_CORES=8-15,24-31"
  cleanup_engine
  ( cd "$REPO" && env $env_pass nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) env='$env_pass' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  wait_prewarm
  grep -m1 "STRATA_POOL_CORES restricts" "$RES/$lbl.out" >> "$OUT" 2>/dev/null
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -2 >> "$OUT" )
}

for r in 1 2 3; do
  run_leg ctl "$r"
  run_leg ccd "$r"
done

cleanup_engine
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t49-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { echo "SERVER UP $(date +%H:%M:%S)" >> "$OUT"; break; }
  sleep 3
done
  wait_prewarm
echo "T49 CCD SPLIT end $(date)" >> "$OUT"
echo "done" >> "$OUT"
