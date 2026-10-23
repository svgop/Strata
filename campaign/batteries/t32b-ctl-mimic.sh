#!/bin/bash
# campaign/batteries/t32b-ctl-mimic.sh — the same-window CTL counterfactual for the
# dm-arm mimic read (84.8 vs anchor 87.9): one plain standing-recipe boot, mimic only.
# If ctl-same-window reads 56-58/66, dm's 56 is bench-side variance and the gate PASSES.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t32b-ctl-mimic.log
echo "T32B CTL MIMIC start $(date)" > "$OUT"

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

cleanup_engine; sleep 2
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy" >> "$OUT"; exit 1; }
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t32b-ctlmimic.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT ctl-mimic health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  cd "$REPO"
  timeout 900 .venv/bin/python campaign/mimicbench2.py 8081 t32b-ctl-mimic >> "$OUT" 2>&1 || echo "MIMIC failed" >> "$OUT"
  cleanup_engine
fi
echo "T32B CTL MIMIC end $(date)" >> "$OUT"
echo "done" >> "$OUT"
