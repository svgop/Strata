#!/bin/bash
# campaign/batteries/t42-rebank.sh — THE OWED FIVE-BOOT POOLED RE-BANK of the standing
# recipe (F-282 line: dm + slots/min-tokens + PDL-pin + SPEC_PROB + the seven-arm stack;
# template swap is orthogonal).  F-263 law: records live as five-boot quiet-window pooled
# means.  Judge vs anchor 215.2/206.4/323.0 and dm pool 222.4/209.9/319.6.  Rounds 1 and 5
# add the PREFILL LADDER (t42-prefill-ladder.py) for the prefill-plateau proof.
# Server restored at the end (standing recipe = adopted state).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t42-rebank.log
echo "T42 RE-BANK start $(date)" > "$OUT"

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

for r in 1 2 3 4 5; do
  lbl="t42-ctl-r$r"
  cleanup_engine
  ( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  ok=0
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; continue; }
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -3 >> "$OUT" )
  if [ "$r" = 1 ] || [ "$r" = 5 ]; then
    ( cd "$REPO" && python3 campaign/t42-prefill-ladder.py 8081 "$lbl" >> "$OUT" 2>&1 )
  fi
done

# final: leave the standing server UP for the user
cleanup_engine
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t42-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { echo "SERVER UP $(date +%H:%M:%S)" >> "$OUT"; break; }
  sleep 3
done
( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
echo "T42 RE-BANK end $(date)" >> "$OUT"
echo "done" >> "$OUT"
