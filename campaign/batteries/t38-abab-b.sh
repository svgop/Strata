#!/bin/bash
# campaign/batteries/t38-abab-b.sh — the B leg completing A-B-A-B on the BINARY variable
# (F-203 interleave law). The pair/conc question the tiebreaker left open:
#   A new-binary:  pair 205.7 / conc 295.8 (busy)   and 198.1 / 292.6 (QUIET box)
#   B rollback:    pair 215.3 / conc 314.2 (busy)
# This leg = B (rollback binary) on the QUIET box. B high again -> real cohort-lane cost
# in the T38 binary (isolate C5); B low -> lane variance, deploy clean.
# Ends with: new binary restored and the server LEFT RUNNING on it (user's monitor).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t38-abab-b.log
echo "T38 ABAB-B start $(date)" > "$OUT"

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

cleanup_engine; sleep 2
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy" >> "$OUT"; exit 1; }

cp "$REPO/engine/strata" "$RES/t38-new-binary.keep" && cp "$REPO/engine/strata.pre-t38" "$REPO/engine/strata" \
  && echo "rollback binary deployed for leg B" >> "$OUT"

( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t38-abab-b.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT t38-abab-b (pre-t38 binary, standing launcher) health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t38-abab-b" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t38-abab-b" 2>&1 | tail -2 >> "$OUT" )
fi
cleanup_engine

# restore the T38 binary and bring the server back up on it (user's monitor)
cp "$RES/t38-new-binary.keep" "$REPO/engine/strata" && rm -f "$RES/t38-new-binary.keep"
cmp -s "$REPO/engine/strata" "$REPO/build/strata" && echo "T38 binary restored" >> "$OUT"
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t38-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && break
  sleep 3
done
echo "SERVER BACK UP on the T38 binary $(date +%H:%M:%S)" >> "$OUT"
echo "T38 ABAB-B end $(date)" >> "$OUT"
echo "done" >> "$OUT"
