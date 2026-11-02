#!/bin/bash
# campaign/batteries/t64-battery.sh - T64: the A2 gate-mass adaptive-k arm (DeepSeek lane).
#
# DS4_ADAPTIVE_K=tau drops a decode token's tail pick when it carries < tau of the routed mass
# (route_publish publishes ids=-1/plan=0/w=0; the lane, the VRAM kernels and moe_win_group already
# skip those).  The lane is DRAM-saturated (probe-bandwidth.c: 49.5 GB/s achievable, lane 46-50),
# so only fewer bytes moves it.  Phase A (this script, no arg): 2 baseline legs with the arm off
# and the tail-share histogram on (DS4_ADAPTIVE_K_STATS=1; tau=0 -> identical behavior) - the legs
# double as the pricing run: grep "adaptive-k tail share" the .out files and pick tau.
# Phase B: TAU=<value> bash t64-battery.sh  -> 2 armed legs at that tau.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-dsv41-8081.sh
PORT=8081
RES=$REPO/campaign/results
TAU="${TAU:-}"
OUT=$RES/t64-battery.log
echo "T64 BATTERY start $(date) tau='${TAU:-off}'" >> "$OUT"

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

run_leg() {  # run_leg <tag> <round>
  local lbl="t64-$1-r$2"
  cleanup_engine
  ( cd "$REPO" && env $ENV_PASS nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 300); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) env='$ENV_PASS' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  ( cd "$REPO" && PORT=$PORT bash campaign/prewarm-ds4.sh "$RES/$lbl.out" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -2 >> "$OUT" )
  cleanup_engine
}

if [ -z "$TAU" ]; then
  for r in 1 2; do ENV_PASS="DS4_ADAPTIVE_K_STATS=1" ; run_leg ctl "$r"; done
else
  for r in 1 2; do ENV_PASS="DS4_ADAPTIVE_K=$TAU DS4_ADAPTIVE_K_STATS=1" ; run_leg "ak$TAU" "$r"; done
fi
echo "T64 BATTERY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
