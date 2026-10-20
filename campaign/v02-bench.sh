#!/bin/bash
# campaign/v02-bench.sh — stock v0.1.45 vs v0.2.1 A-B-A on both quants (F-234 method).
# Legs: iq3s [.45, .2.1, .45], q2_0 [.45, .2.1, .45]; fresh boot each, full probe.
set -u
STOCK=/home/svg/LLM/strata
REPO=/home/svg/LLM/strata-custom-2
PORT=8080
OUT=$REPO/campaign/results/v02-bench.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "V02 BENCH start $(date)" > "$OUT"

port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }
cleanup_engine() {
  local p pid c
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *strata*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
}
leg() {  # $1 binary  $2 config  $3 label
  local bin=$1 cfg=$2 label=$3
  cleanup_engine; sleep 2
  [ -n "$(port_pid)" ] && { echo "=== $label ABORT port busy ===" >> "$OUT"; return; }
  cp "$bin" "$STOCK/engine/strata"
  ( cd "$STOCK" && nohup .venv/bin/python serve/server.py --engine strata --config "$STOCK/$cfg" --port $PORT --host 0.0.0.0 >"$RES/v02-$label.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $label health=$ok $(date +%H:%M:%S) bin=$(basename $bin) ===" >> "$OUT"
  if [ "$ok" != 1 ]; then
    echo "LEG $label FAILED TO BOOT" >> "$OUT"
    tail -60 "$RES/v02-$label.out" >> "$OUT" 2>/dev/null
    tail -40 "$STOCK/strata-q2_0.log" "$STOCK/strata-iq3s-stock.log" 2>/dev/null >> "$OUT"
    return
  fi
  cd "$REPO" && python3 campaign/probe-lane.py --port $PORT --label "v02-$label" --short-n 6 >> "$OUT" 2>&1
  cleanup_engine
}

# save the new build as the v0.2.1 artifact first
cd "$STOCK" && cp build/strata engine/strata.v0.2.1 && echo "artifacts: $(ls engine/strata.v0.1.45-pre-v02 engine/strata.v0.2.1 2>/dev/null | wc -l)/2 present" >> "$OUT"

B45=$STOCK/engine/strata.v0.1.45-pre-v02
B2=$STOCK/engine/strata.v0.2.1

leg "$B45" strata-iq3s-stock.json iq3s-45-a1
leg "$B2"  strata-iq3s-stock.json iq3s-21-b
leg "$B45" strata-iq3s-stock.json iq3s-45-a2
leg "$B45" strata-q2_0.json       q2-45-a1
leg "$B2"  strata-q2_0.json       q2-21-b
leg "$B45" strata-q2_0.json       q2-45-a2

# leave the standing deploy = v0.2.1 (the requested update)
cp "$B2" "$STOCK/engine/strata"
echo "V02 BENCH end $(date)" >> "$OUT"
echo "done" >> "$OUT"
