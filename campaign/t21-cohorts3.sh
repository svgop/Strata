#!/bin/bash
# campaign/t21-cohorts3.sh — COHORTS=3 battery (3v3 ABAB vs standing COHORTS=2) + MMLU re-gate.
set -u
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
OUT=$REPO/campaign/results/t21-cohorts3.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T21 COHORTS3 start $(date)" > "$OUT"

# arm launcher = standing launcher with COHORTS=3 (launcher's own export wins, so sed a copy)
sed 's/^export STRATA_COHORTS=2/export STRATA_COHORTS=3/' "$REPO/launch-custom-iq3s-8081.sh" > "$REPO/campaign/launch-c3.sh"
chmod +x "$REPO/campaign/launch-c3.sh"
grep -q "STRATA_COHORTS=3" "$REPO/campaign/launch-c3.sh" || { echo "SED FAILED - aborting" >> "$OUT"; exit 1; }

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
boot() {  # $1 launcher $2 label
  cleanup_engine; sleep 2
  [ -n "$(port_pid)" ] && { echo "=== $2 ABORT port busy ===" >> "$OUT"; return 1; }
  ( cd "$REPO" && nohup "$1" >"$RES/t21-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  if [ "$ok" = 1 ]; then grep -i "cohorts" "$REPO/custom-iq3_s.log" | tail -1 >> "$OUT" 2>/dev/null; fi
  [ "$ok" = 1 ]
}
leg() {  # $1 launcher $2 label
  boot "$1" "$2" || return
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t21-$2" --short-n 6 >> "$OUT" 2>&1
  cleanup_engine
}

C3=$REPO/campaign/launch-c3.sh
C2=$REPO/launch-custom-iq3s-8081.sh
for r in 1 2 3; do
  leg "$C3" "c3-arm-r$r"
  leg "$C2" "c2-ctl-r$r"
done

# MMLU re-gate on the STANDING recipe (clears the merged-line watch flag)
if boot "$C2" "mmlu-regate"; then
  cd "$REPO"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t21-mmlu-regate-a >> "$OUT" 2>&1 || echo "MMLU A failed" >> "$OUT"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t21-mmlu-regate-b >> "$OUT" 2>&1 || echo "MMLU B failed" >> "$OUT"
  cleanup_engine
fi

echo "T21 COHORTS3 end $(date)" >> "$OUT"
echo "done" >> "$OUT"
