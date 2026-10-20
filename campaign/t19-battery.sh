#!/bin/bash
# campaign/t19-battery.sh — T19 v0.2.1-wave-port 3v3 ABAB batteries (t15c-battery.sh pattern).
# Per stage: arm,ctl,arm,ctl,arm,ctl with fresh boots; drift is differenced out by alternation;
# the claim is the arm-vs-ctl mean of the interleaved legs. Per-leg .out captured; the deployed
# binary is md5-verified per leg (F-231: a leg on the wrong binary is void); arm envs whose code
# prints a boot banner are grepped into the log.
# Usage: t19-battery.sh <stage-label> [--ctl-env VAR=0 ...] [arm-env assignments...]
# --ctl-env assigns envs to the CONTROL legs instead (for ports whose code default IS the arm:
# the control then carries the rollback env, e.g. STRATA_MMVQ_IL_WIDE=0).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
STAGE=$1; shift
CTL_ENVS=()
while [ "${1:-}" = "--ctl-env" ]; do
  CTL_ENVS+=("$2"); shift 2
done
OUT=$REPO/campaign/results/t19-$STAGE.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T19 $STAGE battery start $(date) arm-env: $* ctl-env: ${CTL_ENVS[*]:-none} binary: $(md5sum $REPO/engine/strata | cut -c1-8)" > "$OUT"

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
  local logf=$RES/t19-$STAGE-$label.out
  cleanup_engine; sleep 2
  if [ -n "$(port_pid)" ]; then echo "=== $label ABORT port busy ===" >> "$OUT"; return 1; fi
  ( cd "$REPO" && nohup env "$@" "$LAUNCH" >"$logf" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 120); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $STAGE-$label health=$ok bin=$(md5sum $REPO/engine/strata | cut -c1-8) t18=$(strings $REPO/engine/strata | grep -c t18) $(date +%H:%M:%S) ===" >> "$OUT"
  if [ "$ok" != 1 ]; then echo "LEG $STAGE-$label FAILED" >> "$OUT"; tail -40 "$logf" >> "$OUT"; return 1; fi
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t19-$STAGE-$label" --short-n 6 >> "$OUT" 2>&1
  cleanup_engine
}

for r in 1 2 3; do
  boot_probe "arm-r$r" "$@"
  grep -E "strata hc: STRATA_[A-Z0-9_]+=" "$RES/t19-$STAGE-arm-r$r.out" >> "$OUT" 2>/dev/null
  if [ "${#CTL_ENVS[@]}" -gt 0 ]; then boot_probe "ctl-r$r" "${CTL_ENVS[@]}"; else boot_probe "ctl-r$r"; fi
done
echo "T19 $STAGE battery end $(date)" >> "$OUT"
echo "done" >> "$OUT"
