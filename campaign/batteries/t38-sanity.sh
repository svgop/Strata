#!/bin/bash
# campaign/batteries/t38-sanity.sh — post-T38-deploy sanity: the standing binary changed
# (C5 instrument + V2 default-off kill-switch) and the launcher gained STRATA_DM=1 (F-277).
# One prewarmed standing boot + full lane probe; ALSO exercises C5 live (6-conc leg must
# leave a non-null window histogram in /metrics - cohort rounds now feed WSTAT).
# Judge: M1/pair/conc vs the prewarmed anchor band (215.2/206.4/323.0) and the t32b ctl
# pool (217.1/208.1/315.5); M3b in-band; no WSTAT null after a conc leg.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t38-sanity.log
echo "T38 SANITY start $(date)" > "$OUT"

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

cleanup_engine; sleep 2
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy" >> "$OUT"; exit 1; }
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t38-sanity.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT t38-sanity health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  grep -a "STRATA_DM armed\|engine started" "$REPO/custom-iq3_s.log" | tail -2 >> "$OUT"
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t38-sanity" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t38-sanity" 2>&1 | tail -4 >> "$OUT" )
  echo "--- /metrics window histogram after conc leg (C5 live check):" >> "$OUT"
  curl -s -m 5 -H "Authorization: Bearer 3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8" \
    "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -a "window" | head -6 >> "$OUT"
  cleanup_engine
fi
echo "T38 SANITY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
