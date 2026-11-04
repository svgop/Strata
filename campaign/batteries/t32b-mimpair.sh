#!/bin/bash
# campaign/batteries/t32b-mimpair.sh — the PAIRED MIMIC battery settling the dm-arm
# gate (same-window ctl read 60/66 vs dm 56/66 — a 4-item gap that follows the arm;
# never blame drift). 3v3 interleaved, mimic-only per boot. Judgment: dm mean within
# the paired ctl band (±2 weighted items) = bench-side tail draw -> gate PASSES;
# persistent >=3-item gap = numerics investigation (KL yardstick pair) before any
# adoption talk. Speed already settled separately (pooled 8-boot +2.5% M1).
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t32b-mimpair.log
echo "T32B MIMPAIR start $(date)" > "$OUT"


[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy before start" >> "$OUT"; exit 1; }

run_leg() {  # run_leg <tag> <round>
  local lbl="t32b-mimpair-$1-r$2"
  cleanup_engine
  ( cd "$REPO" && env $ENV_PASS nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) env='$ENV_PASS' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  ( cd "$REPO" && timeout 900 .venv/bin/python campaign/mimicbench2.py 8081 "$lbl" >> "$OUT" 2>&1 )
  cleanup_engine
}

for r in 1 2 3; do
  ENV_PASS=""                 ; run_leg ctl "$r"
  ENV_PASS="STRATA_DM=1"      ; run_leg dm  "$r"
done

echo "T32B MIMPAIR end $(date)" >> "$OUT"
echo "done" >> "$OUT"
