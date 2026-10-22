#!/bin/bash
# campaign/batteries/t32-tranche.sh — T32 WINDOW-TRANCHE battery, phase 1 (F-270).
# Arms (all default-off env gates, no launcher copies — outer env passes through the
# standing launcher, proven by the F-270 smokes):
#   ctl  = standing recipe (the prewarmed anchor posture)
#   keep = STRATA_SPEC_KEEP=0.5      (the F-269 calibrated draft_cum rule)
#   dm   = STRATA_DM=1               (the F-264-priced in-kernel group fold)
#   both = the two composed
# 3 interleaved rounds, every leg prewarmed (the adopted measurement posture, F-266)
# + probe-lane + probe-metrics. Judge per the F-263 law: means vs the battery's own
# CTL legs AND the banked prewarmed anchor 215.2/206.4/323.0; adoption of sub-4%
# deltas needs five-boot quiet-window means pooled across windows.
# Summarize with: .venv/bin/python campaign/batteries/t32-summary.py
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t32-tranche.log
echo "T32 TRANCHE start $(date)" > "$OUT"

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

run_leg() {  # run_leg <tag> <env-prefix...> -- handled by caller via ENV_PASS
  local tag=$1 lbl="t32-$1-r$2"
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

for r in 1 2 3; do
  ENV_PASS=""                                                ; run_leg ctl   "$r"
  ENV_PASS="STRATA_SPEC_KEEP=0.5"                            ; run_leg keep  "$r"
  ENV_PASS="STRATA_DM=1"                                     ; run_leg dm    "$r"
  ENV_PASS="STRATA_SPEC_KEEP=0.5 STRATA_DM=1"                ; run_leg both  "$r"
done

echo "T32 TRANCHE end $(date)" >> "$OUT"
echo "done" >> "$OUT"
