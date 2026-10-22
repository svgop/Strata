#!/bin/bash
# campaign/t31-rebank.sh — T31 QUIET-WINDOW RE-BANK (the F-263 instrument).
# Sub-4% candidates are only resolvable by five-boot quiet-window means pooled
# across boots, so the BASE recipe (standing launcher, zero env changes) gets
# its anchor FIRST, then prewarm2.sh is judged in the SAME window against that
# anchor (its +3-5% five-boot-mean claim, judged 3v5 back-to-back).
#   Phase A: 5 CTL boots  -> the anchor means (M1/pair/conc/M3b/accept).
#   Phase B: 3 prewarm2 boots (prewarm2.sh after health, before the probe).
# Every leg: probe-lane (M1/pair/conc/M3b/accept/ms-window trend) + probe-metrics
# (TTFT/ITL/E2E p50-p99, sm_clock/temp/power, golden-hash drift).
# Summarize with: .venv/bin/python campaign/t31-summary.py
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t31-rebank.log
echo "T31 REBANK start $(date)" > "$OUT"

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

run_round() {  # run_round <phase-tag> <round>
  local phase=$1 r=$2 lbl="t31-$1-r$2"
  cleanup_engine
  ( cd "$REPO" && nohup "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  if [ "$phase" = pre ]; then
    echo "--- prewarm2 $lbl $(date +%H:%M:%S) ---" >> "$OUT"
    ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  fi
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tee -a "$OUT" )
  grep -aE 'first token|window up to' "$REPO/custom-iq3_s.log" | tail -2 >> "$OUT" 2>/dev/null
  cleanup_engine
}

for r in 1 2 3 4 5; do run_round base "$r"; done
for r in 1 2 3; do run_round pre "$r"; done

echo "T31 REBANK end $(date)" >> "$OUT"
echo "done" >> "$OUT"
