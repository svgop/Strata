#!/bin/bash
# campaign/batteries/t49-ccdsplit.sh — F-295: the CCD thread-class split judged 3v3 ABAB.
#   ctl = standing (pool workers anywhere)    ccd = STRATA_POOL_CORES="8-15,24-31" (CCD1)
# The feed class gets a WHOLE CCD (16 logical) — isolation by abundance, not the falsified
# whole-tree scarcity pin.  Judge: M1/pair/conc pooled + the window-tax question (can we
# recover the measured +8.5-14.8%?).  Server restored after.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t49-ccdsplit.log
echo "T49 CCD SPLIT start $(date)" > "$OUT"

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
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
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
( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
echo "T49 CCD SPLIT end $(date)" >> "$OUT"
echo "done" >> "$OUT"
