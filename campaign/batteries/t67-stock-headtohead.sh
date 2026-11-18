#!/bin/bash
# campaign/batteries/t67-stock-headtohead.sh — upstream v0.2.15 vs our fork, same pack, same
# window (2026-11-18).  The custom side is t66-warm-recheck (3 legs, minutes earlier, same
# box state); this battery runs the STOCK engine 3 legs on its native port 8080 against the
# identical IQ3_S pack on /mnt/DATA (config repointed, kv aligned to int8).  Judge: mean of 3
# vs T66's means — answers "has upstream gotten faster than us" without interleaved reboots.
set -u
CUST=/home/svg/LLM/strata-custom-2
STOCK=/home/svg/LLM/strata
RES=$CUST/campaign/results
OUT=$RES/t67-stock-headtohead.log
mkdir -p "$RES"
echo "T67 STOCK HEAD-TO-HEAD start $(date)" > "$OUT"

port_pid() { ss -ltnp 2>/dev/null | grep ":8080 " | grep -oP 'pid=\K[0-9]+' | head -1; }
cleanup() {
  local p pid c
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *"$STOCK/engine/"*|*"$STOCK/build-cuda/"*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
  local t
  for t in $(seq 1 30); do [ -z "$(port_pid)" ] && break; sleep 1; done
}

run_leg() {  # run_leg <round>
  local r=$1 lbl="t67-stock-r$1"
  cleanup
  ( cd "$STOCK" && nohup ./run-iq3s-stock.sh >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 150); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:8080/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup; return; }
  # stock has no server prewarm; one warm request settles the pack path (t22 law)
  curl -s -m 120 -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Authorization: Bearer 3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":200}' >/dev/null
  ( cd "$CUST" && python3 campaign/probe-lane.py --port 8080 --label "$lbl" --short-n 6 \
      --log "$STOCK/strata-iq3s-stock.log" >> "$OUT" 2>&1 )
  ( cd "$CUST" && .venv/bin/python campaign/probe-metrics.py --port 8080 --label "$lbl" 2>&1 | tail -3 >> "$OUT" )
  cleanup
}

for r in 1 2 3; do run_leg "$r"; done
echo "T67 end $(date)" >> "$OUT"
echo "done" >> "$OUT"
