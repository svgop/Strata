#!/bin/bash
# campaign/batteries/t43-ternary-battery.sh — F-284: THE TERNARY REACTOR JUDGMENT.
# Engine: build/strata (the ver_c build, 01:19) deployed to engine/strata first.
# Legs (all --batch 9 via the b9 config/launcher copies at repo root):
#   ctl9 = STRATA_COHORTS unset (2 cohorts at 9-wide = the config-only control that
#          MEASURES the 5+4 rotation loss on the current line; F-273 read 9c~6-parity)
#   ter9 = STRATA_COHORTS=3 (3+3+3; every slot every round; band 355-390 per T37 C3)
# SAFETY SMOKE first: 12 concurrent known-answer requests, all-must-be-correct + zero ERR
# + the boot's "STRATA_COHORTS=3 on:" line + ternary round lines in the log.
# Judge: 9-conc aggregate (probe-lane --conc-n 9) 3v3 ABAB + WSTAT + round shapes.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081-b9.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t43-ternary.log
echo "T43 TERNARY start $(date)" > "$OUT"

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
cp "$REPO/build/strata" "$REPO/engine/strata" && echo "deployed build/strata ($(stat -c %y "$REPO/engine/strata" | cut -c1-16))" >> "$OUT"

run_leg() {  # $1 tag, $2 round, $3 env
  local lbl="t43-$1-r$2"
  cleanup_engine
  ( cd "$REPO" && env $3 nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) env='$3' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 --conc-n 9 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -2 >> "$OUT" )
}

# ---- safety smoke: boot ter9 and hit it with 12 concurrent known-answer requests
cleanup_engine
( cd "$REPO" && env STRATA_COHORTS=3 nohup bash "$LAUNCH" >"$RES/t43-smoke.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== SMOKE boot health=$ok ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/t43-smoke.py >> "$OUT" 2>&1 )
  echo "--- boot lines:" >> "$OUT"
  grep -E "COHORTS=3|COHORTS=2" "$RES/t43-smoke.out" | head -3 >> "$OUT"
fi

# ---- 3v3
for r in 1 2 3; do
  run_leg ctl9 "$r" ""
  run_leg ter9 "$r" "STRATA_COHORTS=3"
done

# restore: standing config (batch 6) + standing launcher, server UP
cleanup_engine
( cd "$REPO" && nohup bash launch-custom-iq3s-8081.sh >"$RES/t43-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { echo "SERVER UP (standing) $(date +%H:%M:%S)" >> "$OUT"; break; }
  sleep 3
done
( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
echo "T43 TERNARY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
