#!/bin/bash
# campaign/batteries/t50-dt-sweep.sh — F-299: the global DRAFT_TEMP re-sweep POST-SPEC_PROB
# (F-201's 0.85 was swept pre-SPEC_PROB; the acceptance math changed).  Legs: 0.75 / 0.85
# (ctl = standing) / 0.95, one boot each, M1/pair/conc + acceptance.  If 0.85 still peaks
# globally, the per-position form (j-indexed dt in coupled_draft_sample) is the follow-up.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t50-dt-sweep.log
echo "T50 DT SWEEP start $(date)" > "$OUT"
[ -n "$(port_pid)" ] && { echo "ABORT: port busy" >> "$OUT"; exit 1; }
run_leg() {  # $1 dt value
  local lbl="t50-dt$1"
  cleanup_engine
  ( cd "$REPO" && env STRATA_SPEC_DRAFT_TEMP=$1 nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG dt=$1 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  wait_prewarm
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -2 >> "$OUT" )
}
run_leg 0.75
run_leg 0.85
run_leg 0.95
cleanup_engine
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t50-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { echo "SERVER UP $(date +%H:%M:%S)" >> "$OUT"; break; }
  sleep 3
done
  wait_prewarm
echo "T50 DT SWEEP end $(date)" >> "$OUT"
echo "done" >> "$OUT"
