#!/bin/bash
# campaign/t29-spec7.sh — depth 7 falsifier. Trace evidence (F-260): 53% of windows accept all
# offered drafts; 12% hit the T=6 cap with mean dprob 0.944; per-position acceptance 92->53%.
# Extending the cap adds rows only on confident continuations (~0.8 accept vs ~0.6 break-even).
# ABAB 3v3, config-copy arm (--spec 7 vs standing 6).
set -u
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
OUT=$REPO/campaign/results/t29-spec7.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T29 SPEC7 start $(date)" > "$OUT"
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
boot() {  # $1 launcher $2 label
  cleanup_engine
  [ -n "$(port_pid)" ] && { echo "=== $2 ABORT port busy ===" >> "$OUT"; return 1; }
  ( cd "$REPO" && nohup "$1" >"$RES/t29-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 120); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ]
}
leg() {  # $1 launcher $2 label
  boot "$1" "$2" || return
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t29-$2" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t29-$2" 2>&1 | tee -a "$OUT" )
  cleanup_engine
}
ARM=$REPO/launch-spec7.sh
CTL=$REPO/launch-custom-iq3s-8081.sh
for r in 1 2 3; do
  leg "$ARM" "spec7-r$r"
  leg "$CTL" "spec6-ctl-r$r"
done
echo "T29 SPEC7 end $(date)" >> "$OUT"
echo "done" >> "$OUT"
