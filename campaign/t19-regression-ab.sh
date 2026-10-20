#!/bin/bash
# campaign/t19-regression-ab.sh — same-window A/B: post-T19 binary vs pre-T19-S1 (norm_fast-only).
# Settles the rebank3 conc question (266 vs frozen 306): window shift or real regression.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
OUT=$REPO/campaign/results/t19-regression-ab.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T19 REGRESSION AB start $(date)" > "$OUT"

# preserve the post-T19 binary as an artifact
cp "$REPO/engine/strata" "$REPO/engine/strata.t19-final"
B_PRE=$REPO/engine/strata.pre-t19-s1
B_CUR=$REPO/engine/strata.t19-final
echo "markers: pre=$(strings $B_PRE | grep -c t19) cur=$(strings $B_CUR | grep -c t19) t18=$(strings $B_CUR | grep -c t18)" >> "$OUT"

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
leg() {  # $1 bin  $2 label
  cleanup_engine; sleep 2
  [ -n "$(port_pid)" ] && { echo "=== $2 ABORT port busy ===" >> "$OUT"; return; }
  cp "$1" "$REPO/engine/strata"
  ( cd "$REPO" && nohup "$LAUNCH" >"$RES/rab-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { echo "LEG $2 FAILED" >> "$OUT"; return; }
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "rab-$2" --short-n 6 >> "$OUT" 2>&1
  cleanup_engine
}

for r in 1 2 3; do
  leg "$B_PRE" "pre-r$r"
  leg "$B_CUR" "cur-r$r"
done

# restore the post-T19 binary as the standing engine
cp "$B_CUR" "$REPO/engine/strata"
echo "T19 REGRESSION AB end $(date)" >> "$OUT"
echo "done" >> "$OUT"
