#!/bin/bash
# campaign/t20-merge-ab.sh (in the MERGE COPY; operates on the ORIGINAL's launcher/engine slot) —
# T20 same-window A/B: Leg A = the v0.2.1-merged binary, Leg B = engine/strata.t19-final (control).
# Pattern of campaign/t19-regression-ab.sh: copy the chosen binary into the ORIGINAL's
# engine/strata, boot via the ORIGINAL's launch-custom-iq3s-8081.sh, probe --short-n 6, kill.
# One variable = the binary. 3v3 ABAB interleaved + one depth-lane-real 32k prefill run per binary.
# Restores engine/strata = strata.t19-final at the end (the standing engine stays the current one).
set -u
ORIG=/home/svg/LLM/strata-custom-2
LAUNCH=$ORIG/launch-custom-iq3s-8081.sh
PORT=8081
RES=/home/svg/LLM/strata-custom-2-merge/campaign/results
OUT=$RES/t20-merge-ab.log
B_MERGED=$RES/strata.t20-merged          # the copy's build, staged here by the boot-test step
B_T19=$ORIG/engine/strata.t19-final
mkdir -p "$RES"
echo "T20 MERGE AB start $(date)" > "$OUT"
echo "merged md5=$(md5sum "$B_MERGED" | cut -d' ' -f1)" >> "$OUT"
echo "t19    md5=$(md5sum "$B_T19" | cut -d' ' -f1)" >> "$OUT"

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
leg() {  # $1 bin  $2 label
  cleanup_engine; sleep 2
  [ -n "$(port_pid)" ] && { echo "=== $2 ABORT port busy ===" >> "$OUT"; return; }
  cp "$1" "$ORIG/engine/strata"
  ( cd "$ORIG" && nohup "$LAUNCH" >"$RES/t20-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { echo "LEG $2 FAILED" >> "$OUT"; cleanup_engine; return; }
  ( cd "$ORIG" && .venv/bin/python campaign/probe-lane.py --port "$PORT" --label "t20-$2" --short-n 6 ) >> "$OUT" 2>&1
  cleanup_engine
}
depth_leg() {  # $1 bin  $2 label  (one 32k real-corpus prefill run on a fresh boot)
  cleanup_engine; sleep 2
  [ -n "$(port_pid)" ] && { echo "=== depth-$2 ABORT port busy ===" >> "$OUT"; return; }
  cp "$1" "$ORIG/engine/strata"
  ( cd "$ORIG" && nohup "$LAUNCH" >"$RES/t20-depth-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== DEPTH LEG $2 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { echo "DEPTH LEG $2 FAILED" >> "$OUT"; cleanup_engine; return; }
  ( cd /home/svg/LLM/strata-custom-2-merge && .venv/bin/python campaign/depth-lane-real.py --port "$PORT" --label "t20-$2" --depth 32000 ) >> "$OUT" 2>&1
  cleanup_engine
}

for r in 1 2 3; do
  leg "$B_T19"    "t19-r$r"
  leg "$B_MERGED" "mrg-r$r"
done
depth_leg "$B_T19"    "t19-32k"
depth_leg "$B_MERGED" "mrg-32k"

# restore the standing engine (current binary stays until the user decides)
cleanup_engine
cp "$B_T19" "$ORIG/engine/strata"
echo "restored engine/strata md5=$(md5sum "$ORIG/engine/strata" | cut -d' ' -f1)" >> "$OUT"
echo "T20 MERGE AB end $(date)" >> "$OUT"
echo "done" >> "$OUT"
