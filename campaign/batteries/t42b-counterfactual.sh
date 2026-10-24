#!/bin/bash
# campaign/batteries/t42b-counterfactual.sh — the T42 LOW-WINDOW COUNTERFACTUALS, ABAB x2.
# T42 pooled 198.4/189.8/283.6 vs dm pool 222.4/209.9/319.6 (-10% all lanes, GPU locked
# 2940/95%/362W throughout = not clocks).  The two standing-recipe changes since the pool
# that could touch speed, each isolated same-window:
#   A ctl  = standing (SPEC_PROB on, new template)
#   B prob0= STRATA_SPEC_PROB=0 (launcher now env-defaults it)
#   C tpl0 = old F-279 template (render-swap only)
# If neither B nor C recovers the band, the WINDOW is the story (records stand as banked;
# morning re-bank).  Server restored after.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t42b-counterfactual.log
echo "T42B COUNTERFACTUAL start $(date)" > "$OUT"

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

[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy before start" >> "$OUT"; exit 1; }

run_leg() {  # $1 arm, $2 round
  local lbl="t42b-$1-r$2"
  cleanup_engine
  if [ "$1" = tpl0 ]; then
    ( cd "$REPO" && git show fb00fe98^:serve/chat_template.jinja > serve/chat_template.jinja )
  else
    cp "/home/svg/Downloads/chat_template (4).jinja" "$REPO/serve/chat_template.jinja"
  fi
  local env_pass=""
  [ "$1" = prob0 ] && env_pass="STRATA_SPEC_PROB=0"
  ( cd "$REPO" && env $env_pass nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) tpl=$(md5sum "$REPO/serve/chat_template.jinja" | cut -c1-8) env='$env_pass' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -2 >> "$OUT" )
}

for r in 1 2; do
  run_leg ctl  "$r"
  run_leg prob0 "$r"
  run_leg tpl0  "$r"
done

# restore: new template + standing server
cleanup_engine
cp "/home/svg/Downloads/chat_template (4).jinja" "$REPO/serve/chat_template.jinja"
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t42b-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { echo "SERVER UP (standing, new template) $(date +%H:%M:%S)" >> "$OUT"; break; }
  sleep 3
done
( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
echo "T42B COUNTERFACTUAL end $(date)" >> "$OUT"
echo "done" >> "$OUT"
