#!/bin/bash
# campaign/batteries/t63-battery.sh - T63: the A2 gate-mass / tail-skip sweep (Qwen lane).
#
# Qwen half of A2: STRATA_ROUTE_TAIL_SKIP is OFF in the standing config (--batch 6 defeats the
# CUDA default's o.batch == 0 condition, generate.cpp:10234-10243).  This battery prices arming
# it at rank 7 (the default's rank) and rank 6, 2 legs each against 2 standing controls.
# The skip is CPU-tier-only (a missed expert every token of the window ranks >= the rank is
# neither copied nor computed); the draft-acceptance cost shows in M1/M3b, the byte saving in
# the conc lane's CPU-expert term.  Judge: +4% adopt bar with the MMLU gate owed if adopted.
#
# Legs: ctl x2, ts7 x2, ts6 x2 (env passthrough, all prewarmed, full probe + metrics).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t63-battery.log
echo "T63 BATTERY start $(date)" > "$OUT"

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

[ -n "$(port_pid)" ] || { echo "NOTE: port not busy at start (the live engine should have been up)" >> "$OUT"; }

run_leg() {  # run_leg <tag> <round>
  local lbl="t63-$1-r$2"
  cleanup_engine
  ( cd "$REPO" && env $ENV_PASS nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 120); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) env='$ENV_PASS' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -2 >> "$OUT" )
  cleanup_engine
}

for r in 1 2; do
  ENV_PASS=""                          ; run_leg ctl "$r"
  ENV_PASS="STRATA_ROUTE_TAIL_SKIP=7"  ; run_leg ts7 "$r"
  ENV_PASS="STRATA_ROUTE_TAIL_SKIP=6"  ; run_leg ts6 "$r"
done

# server back up for the user (the standing recipe)
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t63-final-serve.out" 2>&1 & )
for t in $(seq 1 120); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && break
  sleep 3
done
echo "SERVER LEFT RUNNING $(date +%H:%M:%S)" >> "$OUT"
echo "T63 BATTERY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
