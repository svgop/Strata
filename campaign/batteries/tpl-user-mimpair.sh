#!/bin/bash
# campaign/batteries/tpl-user-mimpair.sh — paired mimic interleave legs B/C/D after
# tpl-user-gates.sh leg 1 (user mimic A): B = standing, C = user, D = standing.
# A/B/A/B across boots settles template-vs-drift on the known style-defer risk lane
# (F-279 lesson: "clear natural prose" fights mimicry without a style-defer clause).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/tpl-user-mimpair.log
echo "TPL-USER MIMPAIR legs B-D start $(date)" > "$OUT"

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

boot_leg() {  # $1 = leg tag, $2 = user|standing
  cleanup_engine; sleep 2
  if [ -n "$(port_pid)" ]; then echo "ABORT: port busy before leg $1" >> "$OUT"; exit 1; fi
  if [ "$2" = user ]; then
    cp "/home/svg/Downloads/chat_template (4).jinja" "$REPO/serve/chat_template.jinja"
  else
    ( cd "$REPO" && git restore serve/chat_template.jinja )
  fi
  echo "=== leg $1 template=$2 md5=$(md5sum "$REPO/serve/chat_template.jinja" | cut -c1-12) $(date +%H:%M:%S) ===" >> "$OUT"
  ( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/tpl-mp-$1.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "boot $1 health=$ok" >> "$OUT"
  if [ "$ok" = 1 ]; then
    ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
    ( cd "$REPO" && timeout 900 .venv/bin/python campaign/mimicbench2.py 8081 tpl-mp-$1 >> "$OUT" 2>&1 ) \
      || echo "MIMIC $1 failed" >> "$OUT"
  fi
}

boot_leg B standing
boot_leg C user
boot_leg D standing
echo "TPL-USER MIMPAIR end $(date)" >> "$OUT"
echo "done" >> "$OUT"
