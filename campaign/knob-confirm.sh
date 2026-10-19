#!/bin/bash
# campaign/knob-confirm.sh — confirmation 3v3 for TSUM and LP + MMLU gates for LP.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
ARMS=/home/svg/LLM/Strata/custom-arms.env
PORT=8081
OUT=$REPO/campaign/results/knob-confirm.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "KNOB CONFIRM start $(date)" > "$OUT"

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
boot() {  # $1 label $2 kind $3.. env
  local label=$1 kind=$2; shift 2
  cleanup_engine; sleep 2
  if [ -n "$(port_pid)" ]; then echo "=== $label ABORT port busy ===" >> "$OUT"; return 1; fi
  if [ "$kind" = "lp" ]; then
    ( cd "$REPO" && nohup env "$@" bash -c ". $ARMS; exec .venv/bin/python serve/server.py --engine strata --config $REPO/campaign/custom-iq3_s-lp.json --port $PORT --host 0.0.0.0" >"$RES/cf-$label.out" 2>&1 & )
  else
    ( cd "$REPO" && nohup env "$@" "$LAUNCH" >"$RES/cf-$label.out" 2>&1 & )
  fi
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $label health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ]
}
probe_leg() { cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$1" --short-n 6 >> "$OUT" 2>&1; }
battery() {  # $1 cand $2 kind $3.. env
  local cand=$1 kind=$2; shift 2
  local r
  for r in 1 2 3; do
    boot "$cand-arm-r$r" "$kind" "$@" && probe_leg "cf-$cand-arm-r$r"; cleanup_engine
    boot "$cand-ctl-r$r" launcher && probe_leg "cf-$cand-ctl-r$r"; cleanup_engine
  done
}

battery tsum launcher STRATA_TSUM=1
battery lp lp
# LP MMLU gates on a dedicated LP boot
if boot lp-mmlu lp; then
  cd "$REPO"
  timeout 540 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 confirm-lp-mmlu-a >> "$OUT" 2>&1 || echo "MMLU A failed" >> "$OUT"
  timeout 540 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 confirm-lp-mmlu-b >> "$OUT" 2>&1 || echo "MMLU B failed" >> "$OUT"
  cleanup_engine
fi
echo "KNOB CONFIRM end $(date)" >> "$OUT"
echo "done" >> "$OUT"
