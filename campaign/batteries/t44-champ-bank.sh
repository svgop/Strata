#!/bin/bash
# campaign/batteries/t44-champ-bank.sh — THE RESTORED CHAMPION'S FIVE-BOOT BANK (F-285a
# rider): the full lane probe (M1 shorts A/B + 2000-tok long + pair + 6-conc + metrics)
# on 5 fresh prewarmed boots of the standing champion (engine/strata = build-fresh of
# HEAD + generate.cpp@9e5366c6).  F-263 law: the final number = the pooled mean.
# Server restored (champion, standing config) at the end.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t44-champ-bank.log
echo "T44 CHAMP BANK start $(date)" > "$OUT"
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

[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy before start" >> "$OUT"; exit 1; }

for r in 1 2 3 4 5; do
  lbl="t44-champ-r$r"
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
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -2 >> "$OUT" )
done

cleanup_engine
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t44-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { echo "SERVER UP (champion) $(date +%H:%M:%S)" >> "$OUT"; break; }
  sleep 3
done
( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
echo "T44 CHAMP BANK end $(date)" >> "$OUT"
echo "done" >> "$OUT"
