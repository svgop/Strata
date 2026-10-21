#!/bin/bash
# campaign/t27-baseline.sh — the improvement-loop BASELINE (F-256 measurement epoch).
# 3 boots of the STANDING launcher (no env changes), each with the full lane probe
# (M1 single / pair / conc 6, prefill, decode) + probe-metrics stability capture
# (TTFT/ITL/E2E p50-p99, acceptance, clock/temp/power, golden-hash drift).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
OUT=$REPO/campaign/results/t27-baseline.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T27 BASELINE start $(date)" > "$OUT"

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

for r in 1 2 3; do
  cleanup_engine
  ( cd "$REPO" && nohup "$LAUNCH" >"$RES/t27-base-r$r.out" 2>&1 & )
  ok=0
  for t in $(seq 1 90); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG base-r$r health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || continue
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t27-base-r$r" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t27-base-r$r" 2>&1 | tee -a "$OUT" )
  grep -aE 'first token|window up to' "$REPO/custom-iq3_s.log" | tail -2 >> "$OUT" 2>/dev/null
  cleanup_engine
done

echo "T27 BASELINE end $(date)" >> "$OUT"
echo "done" >> "$OUT"
