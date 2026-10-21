#!/bin/bash
# campaign/t28-share10.sh — the decode-share curve's far end: STRATA_BATCH_DECODE_SHARE=1.0
# (decode gets the whole read-piece window; the 0.75 point read +1.1% conc / +2.0% M1 - the
# curve may not have ended). ABAB 3v3, env-only arm.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
OUT=$REPO/campaign/results/t28-share10.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T28 SHARE10 start $(date)" > "$OUT"

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
boot() {  # $1 extra env $2 label
  cleanup_engine
  [ -n "$(port_pid)" ] && { echo "=== $2 ABORT port busy ===" >> "$OUT"; return 1; }
  ( cd "$REPO" && nohup env $1 "$LAUNCH" >"$RES/t28-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) env=[$1] ===" >> "$OUT"
  [ "$ok" = 1 ]
}
leg() {  # $1 extra env $2 label
  boot "$1" "$2" || return
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t28-$2" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t28-$2" 2>&1 | tee -a "$OUT" )
  cleanup_engine
}
for r in 1 2 3; do
  leg "STRATA_BATCH_DECODE_SHARE=1.0" "share10-r$r"
  leg "" "share05-ctl-r$r"
done
echo "T28 SHARE10 end $(date)" >> "$OUT"
echo "done" >> "$OUT"
