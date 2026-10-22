#!/bin/bash
# campaign/t20-gates.sh — the two law items for the merged binary: 2x MMLU (+mimic) and a
# five-boot re-bank. Runs the merged binary in the ORIGINAL harness (launcher unchanged).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
MERGED=/home/svg/LLM/strata-custom-2-merge/campaign/results/strata.t20-merged
PORT=8081
OUT=$REPO/campaign/results/t20-gates.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T20 GATES start $(date)" > "$OUT"

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
boot() {  # $1 label
  cleanup_engine; sleep 2
  [ -n "$(port_pid)" ] && { echo "=== $1 ABORT port busy ===" >> "$OUT"; return 1; }
  ( cd "$REPO" && nohup "$LAUNCH" >"$RES/t20-$1.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== BOOT $1 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ]
}

# deploy the merged binary into the original harness
cp "$MERGED" "$REPO/engine/strata" && echo "deployed merged: $(md5sum "$REPO/engine/strata" | cut -d' ' -f1)" >> "$OUT"

# --- gate: 2x MMLU + mimic on one boot
if boot mmlu; then
  cd "$REPO"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t20-mmlu-a >> "$OUT" 2>&1 || echo "MMLU A failed" >> "$OUT"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t20-mmlu-b >> "$OUT" 2>&1 || echo "MMLU B failed" >> "$OUT"
  timeout 600 .venv/bin/python campaign/mimicbench2.py 8081 t20-mimic >> "$OUT" 2>&1 || echo "MIMIC failed" >> "$OUT"
  cleanup_engine
fi

# --- five-boot re-bank with the merged binary deployed
for i in 1 2 3 4 5; do
  boot "rebank4-r$i" || continue
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "rebank4-r$i" --short-n 6 >> "$OUT" 2>&1
  cleanup_engine
done

echo "T20 GATES end $(date)" >> "$OUT"
echo "done" >> "$OUT"
