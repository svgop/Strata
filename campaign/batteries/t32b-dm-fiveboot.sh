#!/bin/bash
# campaign/batteries/t32b-dm-fiveboot.sh — T32 EXTENSION: the owed dm/both five-boot
# pooled confirmation (F-263 law: sub-4% deltas resolve only as five-boot quiet-window
# means pooled across windows; T32 ran 3 rounds — these 5 interleave with them).
# Legs (env passes through the standing launcher; NO launcher copies):
#   ctl  = standing recipe, prewarmed posture (also feeds the pooled ctl set)
#   dm   = STRATA_DM=1                          (dm_wait_combine; T32 read +2.8% M1)
#   both = STRATA_SPEC_KEEP=0.5 STRATA_DM=1     (keep falsified solo -4.4% but the
#          both-arm pair +4.9% cleared the bar — this leg informs B3 conditional-graph
#          scoping; it is NOT an adoption candidate on solo numbers)
# 5 rounds ABAB-interleaved, every leg prewarmed (F-266 posture).
# Gates (2x MMLU + mimic) are decision-gated: main loop runs them ONLY if the pooled
# verdict adopts. Summarize by pooling labels t32-* + t32b-* through
# .venv/bin/python campaign/batteries/t32-summary.py.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t32b-dm-fiveboot.log
echo "T32B DM FIVE-BOOT start $(date)" > "$OUT"

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

run_leg() {  # run_leg <tag> <round>; env via $ENV_PASS
  local lbl="t32b-$1-r$2"
  cleanup_engine
  ( cd "$REPO" && env $ENV_PASS nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) env='$ENV_PASS' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tee -a "$OUT" )
  grep -a "spec-keep: armed\|strata dm" "$REPO/custom-iq3_s.log" | tail -1 >> "$OUT" 2>/dev/null
  cleanup_engine
}

for r in 1 2 3 4 5; do
  ENV_PASS=""                                     ; run_leg ctl  "$r"
  ENV_PASS="STRATA_DM=1"                          ; run_leg dm   "$r"
  ENV_PASS="STRATA_SPEC_KEEP=0.5 STRATA_DM=1"     ; run_leg both "$r"
done

echo "T32B DM FIVE-BOOT end $(date)" >> "$OUT"
echo "done" >> "$OUT"
