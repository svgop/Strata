#!/bin/bash
# campaign/batteries/t38-tiebreak.sh — the user-called tiebreaker: one more CLEAN boot of
# the deployed T38 binary (C5+V2) on the standing launcher (STRATA_DM=1), full probe.
# Decides the deploy question the two single boots left open:
#   new-binary legs: sanity 205.6/205.7/295.8   | rollback-binary leg: 207.6/215.3/314.2
# If this leg reads 207/215/314-class -> deploy clean (the 295.8 was boot noise);
# if pair/conc read low AGAIN vs the rollback leg -> real cohort-lane cost, investigate C5.
# SERVER LEFT RUNNING after the probe (user's monitor tab is live on :8081).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t38-tiebreak.log
echo "T38 TIEBREAK start $(date)" > "$OUT"

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

cleanup_engine; sleep 2
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy" >> "$OUT"; exit 1; }
cmp -s "$REPO/engine/strata" "$REPO/build/strata" || { echo "ABORT: engine != build" >> "$OUT"; exit 1; }

( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t38-tiebreak.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT t38-tiebreak (T38 binary, standing launcher) health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  grep -a "STRATA_DM armed" "$REPO/custom-iq3_s.log" | tail -1 >> "$OUT"
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t38-tiebreak" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t38-tiebreak" 2>&1 | tail -3 >> "$OUT" )
fi
echo "SERVER LEFT RUNNING (user monitor tab live) $(date +%H:%M:%S)" >> "$OUT"
echo "T38 TIEBREAK end $(date)" >> "$OUT"
echo "done" >> "$OUT"
