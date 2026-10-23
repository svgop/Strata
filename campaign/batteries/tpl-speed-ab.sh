#!/bin/bash
# campaign/batteries/tpl-speed-ab.sh — the F-283 'feels slower' A/B: user's protocol
# template vs the F-279 standing one, judged on REAL-USAGE token economy + wall
# (campaign/tpl-ux-probe.py: 5 daily-driver prompts x2).  Leg A rides the LIVE warm
# server (new template); leg B = old template fresh boot; leg C = new template fresh
# boot + the 3-lane speed read (probe-lane.py) + server LEFT UP on the adopted state.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/tpl-speed-ab.log
echo "TPL SPEED A/B start $(date)" > "$OUT"

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

boot_tpl() {  # $1 = user|standing ; assumes port free
  if [ "$1" = user ]; then
    cp "/home/svg/Downloads/chat_template (4).jinja" "$REPO/serve/chat_template.jinja"
  else
    ( cd "$REPO" && git show fb00fe98^:serve/chat_template.jinja > "$REPO/serve/chat_template.jinja" )
  fi
  ( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/tpl-ab-$1.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "boot $1 health=$ok md5=$(md5sum "$REPO/serve/chat_template.jinja" | cut -c1-12)" >> "$OUT"
  [ "$ok" = 1 ] && ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  return 0
}

echo "=== leg A: LIVE warm server, user template ===" >> "$OUT"
( cd "$REPO" && python3 campaign/tpl-ux-probe.py 8081 tpl-ux-A-user 2 2>&1 | tee "$RES/tpl-ux-A.txt" >> "$OUT" )

echo "=== leg B: old template fresh boot ===" >> "$OUT"
cleanup_engine; sleep 2
boot_tpl standing
( cd "$REPO" && python3 campaign/tpl-ux-probe.py 8081 tpl-ux-B-standing 2 2>&1 | tee "$RES/tpl-ux-B.txt" >> "$OUT" )

echo "=== leg C: user template fresh boot + speed lanes ===" >> "$OUT"
cleanup_engine; sleep 2
boot_tpl user
( cd "$REPO" && python3 campaign/tpl-ux-probe.py 8081 tpl-ux-C-user 2 2>&1 | tee "$RES/tpl-ux-C.txt" >> "$OUT" )
echo "--- probe-lane (engine speed, current recipe):" >> "$OUT"
( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label tpl-ab-C-lane --short-n 6 >> "$OUT" 2>&1 )
( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label tpl-ab-C-lane 2>&1 | tail -2 >> "$OUT" )

echo "SERVER LEFT RUNNING (user template) $(date +%H:%M:%S)" >> "$OUT"
echo "TPL SPEED A/B end $(date)" >> "$OUT"
echo "done" >> "$OUT"
