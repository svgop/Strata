#!/bin/bash
# campaign/batteries/t38-binary-counterfactual.sh — same-window binary A/B for the T38
# deploy question: sanity read M1 205.6 on the new binary (C5+V2) + STRATA_DM=1; the t32b
# ctl window pool was 210.7-232.0. One boot of the ROLLBACK binary (strata.pre-t38) with
# the SAME launcher (dm armed) isolates the binary variable. Judge: both in the same
# band = window noise, deploy clean; rollback clearly higher = real regression.
# The engine is RESTORED to the new binary at the end no matter what (no early exit
# between the swap and the restore).
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t38-bincf.log
echo "T38 BINARY COUNTERFACTUAL start $(date)" > "$OUT"


cleanup_engine; sleep 2
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy" >> "$OUT"; exit 1; }

cp "$REPO/engine/strata" "$RES/t38-new-binary.keep" && cp "$REPO/engine/strata.pre-t38" "$REPO/engine/strata" \
  && echo "rollback binary deployed for leg A" >> "$OUT"

( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t38-bincf.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT t38-bincf (pre-t38 binary, dm launcher) health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  wait_prewarm
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t38-bincf" --short-n 6 >> "$OUT" 2>&1 )
fi
cleanup_engine

# restore the T38 binary unconditionally
cp "$RES/t38-new-binary.keep" "$REPO/engine/strata" && rm -f "$RES/t38-new-binary.keep" \
  && cmp -s "$REPO/engine/strata" "$REPO/build/strata" \
  && echo "T38 binary restored (== build/strata)" >> "$OUT"

echo "T38 BINARY COUNTERFACTUAL end $(date)" >> "$OUT"
echo "done" >> "$OUT"
