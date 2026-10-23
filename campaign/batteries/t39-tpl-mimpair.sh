#!/bin/bash
# campaign/batteries/t39-tpl-mimpair.sh — paired old-vs-new TEMPLATE mimic interleave (F-279).
# The template gate read mimic 56/66 (the known low mode) with MMLU 94.3x2 PASS; one real
# mechanism was the "clear natural prose" clause fighting style-mimicry tasks — now fixed
# with "honoring any style the request itself asks for". This battery settles tail-draw vs
# real cost: 2x old-template legs vs 2x fixed-new-template legs, ABAB, mimic only.
# Judge: new within the old band (±2 weighted items) = adopt; persistent gap = hold.
# Ends with the NEW files in place and the server LEFT RUNNING on them.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t39-tpl-mimpair.log
KEEP=/tmp/t39-tpl-new
echo "T39 TPL MIMPAIR start $(date)" > "$OUT"

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

# save the NEW (fixed) files once
mkdir -p "$KEEP"
cp "$REPO/serve/chat_template.jinja" "$REPO/serve/server.py" "$REPO/serve/chat_golden.json" "$KEEP/"

set_old() { ( cd "$REPO" && git checkout -- serve/chat_template.jinja serve/server.py serve/chat_golden.json ); echo "files: OLD template" >> "$OUT"; }
set_new() { cp "$KEEP/chat_template.jinja" "$KEEP/server.py" "$KEEP/chat_golden.json" "$REPO/serve/"; echo "files: NEW template (fixed)" >> "$OUT"; }

run_mimic() {  # run_mimic <label>
  cleanup_engine; sleep 2
  [ -n "$(port_pid)" ] && { echo "ABORT: port busy at $1" >> "$OUT"; return; }
  ( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/$1.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== BOOT $1 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  ( cd "$REPO" && timeout 900 .venv/bin/python campaign/mimicbench2.py 8081 "$1" >> "$OUT" 2>&1 )
  cleanup_engine
}

cleanup_engine
for r in 1 2; do
  set_old ; run_mimic "t39-tpl-old-r$r"
  set_new ; run_mimic "t39-tpl-new-r$r"
done

# final state: NEW files + server up on them
set_new
cmp -s "$REPO/serve/chat_template.jinja" "$KEEP/chat_template.jinja" && echo "final state: NEW files confirmed" >> "$OUT"
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t39-tpl-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && break
  sleep 3
done
echo "SERVER LEFT RUNNING (new template) $(date +%H:%M:%S)" >> "$OUT"
echo "T39 TPL MIMPAIR end $(date)" >> "$OUT"
echo "done" >> "$OUT"
