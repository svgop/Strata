#!/bin/bash
# campaign/t15c-battery.sh — T15C hc-kernel tranche 3v3 ABAB batteries (knob-battery.sh pattern).
# Per stage: arm,ctl,arm,ctl,arm,ctl with fresh boots; drift is differenced out by alternation;
# the claim is the arm-vs-ctl mean of the interleaved legs. Per-leg .out captured; the arm env
# is verified per leg by its boot banner (F-231: a leg without its banner is void).
# Usage: t15c-battery.sh <stage-label> [--ctl-env VAR=1 ...] [arm-env assignments...]
# --ctl-env assigns envs to the CONTROL legs instead (for arms whose code default IS the flip:
# the control then carries the rollback env, e.g. STRATA_GR_NORM_FAST=0).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
STAGE=$1; shift
CTL_ENVS=()
while [ "${1:-}" = "--ctl-env" ]; do
  CTL_ENVS+=("$2"); shift 2
done
OUT=$REPO/campaign/results/t15c-$STAGE.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T15C $STAGE battery start $(date) arm-env: $* ctl-env: ${CTL_ENVS[*]:-none}" > "$OUT"

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

boot_probe() {  # $1 label  $2.. env for the arm leg (no env = standing control)
  local label=$1; shift
  local logf=$RES/t15c-$STAGE-$label.out
  cleanup_engine; sleep 2
  if [ -n "$(port_pid)" ]; then echo "=== $label ABORT port busy ===" >> "$OUT"; return 1; fi
  ( cd "$REPO" && nohup env "$@" "$LAUNCH" >"$logf" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 120); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $STAGE-$label health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  if [ "$ok" != 1 ]; then echo "LEG $STAGE-$label FAILED" >> "$OUT"; tail -40 "$logf" >> "$OUT"; return 1; fi
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t15c-$STAGE-$label" --short-n 6 >> "$OUT" 2>&1
  cleanup_engine
}

for r in 1 2 3; do
  boot_probe "arm-r$r" "$@"
  grep -E "strata hc: STRATA_[A-Z0-9_]+=1" "$RES/t15c-$STAGE-arm-r$r.out" >> "$OUT" 2>/dev/null
  if [ "${#CTL_ENVS[@]}" -gt 0 ]; then boot_probe "ctl-r$r" "${CTL_ENVS[@]}"; else boot_probe "ctl-r$r"; fi
done
echo "T15C $STAGE battery end $(date)" >> "$OUT"
echo "done" >> "$OUT"
