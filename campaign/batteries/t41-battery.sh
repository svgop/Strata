#!/bin/bash
# campaign/batteries/t41-battery.sh — the T39/T40 JUDGMENT BATTERY (quiet-window target).
# Legs (env passthrough, all prewarmed, full probe + metrics):
#   ctl    = standing recipe (now incl. E1 slots-24 + E6 min-tokens-200 in the config)
#   c2a    = STRATA_COHORT_OVERLAP=1   (finish-under-window; smoke proved safety; the +5-8%
#            conc band vs the thin-sample 27.3 p50 scare — 3v3 decides)
#   tau015 = STRATA_TYPICAL_ACCEPT=0.15 (LOOSER rescue per the F-178 direction law; MMLU-gated
#            if it wins anything)
#   prob   = STRATA_SPEC_PROB=1         (the never-measured probabilistic acceptance arm)
# 3 rounds ctl/c2a interleaved, then 2x tau015, 2x prob. Judge: five-boot pooled law; the
# c2a verdict reads the conc lane + WSTAT p50/p99; tau/prob read M1/acceptance + MMLU gate.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t41-battery.log
echo "T41 BATTERY start $(date)" > "$OUT"

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

[ -n "$(port_pid)" ] && { echo "ABORT: port busy before start" >> "$OUT"; exit 1; }

run_leg() {  # run_leg <tag> <round>
  local lbl="t41-$1-r$2"
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
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -2 >> "$OUT" )
  cleanup_engine
}

for r in 1 2 3; do
  ENV_PASS=""                          ; run_leg ctl   "$r"
  ENV_PASS="STRATA_COHORT_OVERLAP=1"  ; run_leg c2a   "$r"
done
ENV_PASS="STRATA_TYPICAL_ACCEPT=0.15" ; run_leg tau015 1; run_leg tau015 2
ENV_PASS="STRATA_SPEC_PROB=1"         ; run_leg prob   1; run_leg prob   2

# server back up for the user
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t41-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && break
  sleep 3
done
echo "SERVER LEFT RUNNING $(date +%H:%M:%S)" >> "$OUT"
echo "T41 BATTERY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
