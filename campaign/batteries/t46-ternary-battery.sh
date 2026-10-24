#!/bin/bash
# campaign/batteries/t46-ternary-battery.sh — F-286: THE TERNARY JUDGMENT (the re-run that
# works).  Both arms ride the UNLOCKED b9 lane (STRATA_ELASTIC_S_MIN=9 — F-285's config
# fix; the admission-ramp hang never runs):
#   ctl9 = STRATA_COHORTS=2 (5+4, the rotation cap)   ter9 = STRATA_COHORTS=3 (3+3+3)
# 3v3 ABAB, 9-conc lane (probe-lane --conc-n 9), WSTAT + round shapes per leg.
# Judge: 9-conc aggregate pooled + every-slot-every-round fairness; the record target =
# the 323.0 conc record (T37 band 355-390 in a normal window; this window runs ~10% cool).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081-b9.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t46-ternary.log
echo "T46 TERNARY start $(date)" > "$OUT"
md5sum "$REPO/engine/strata" >> "$OUT"

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
  local lbl="t46-$1-r$2"
  local env_pass="STRATA_ELASTIC_S_MIN=9"
  [ "$1" = ter9 ] && env_pass="STRATA_ELASTIC_S_MIN=9 STRATA_COHORTS=3"
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
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 --conc-n 9 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -2 >> "$OUT" )
}

for r in 1 2 3; do
  run_leg ctl9 "$r"
  run_leg ter9 "$r"
done

# restore the standing daily server (b6 champion)
cleanup_engine
( cd "$REPO" && nohup bash launch-custom-iq3s-8081.sh >"$RES/t46-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { echo "SERVER UP (standing b6) $(date +%H:%M:%S)" >> "$OUT"; break; }
  sleep 3
done
( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
echo "T46 TERNARY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
