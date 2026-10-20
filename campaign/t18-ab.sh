#!/bin/bash
# campaign/t18-ab.sh — T18 fix proof: 3v3 ABAB interleaved battery, fixed binary vs pre-fix binary.
# v2: copies come from STABLE source files (never engine/strata itself), and every leg verifies
# the marker string ("exceeded the verify window's token capacity", present only in the fixed
# build) on the DEPLOYED file before probing. Ends by restoring the standing (fixed) deploy.
# Arm leg = fixed build, ctl leg = pre-fix build; probe-lane.py (M1 / pair / 6-conc) as the
# instrument, fresh boot per leg, ABAB order; means differenced across the window.
set -u
REPO=/home/svg/LLM/strata-custom-2
RES=$REPO/campaign/results
PORT=8081
DEPLOY=$REPO/engine/strata
SRC_FIXED=$REPO/engine/strata.t18fixed      # stable copy of the fixed build
SRC_PREFIX=$REPO/engine/strata.pre-t18fix   # stable copy of the pre-fix build
MARK="exceeded the verify window's token capacity"
OUT=$RES/t18-ab.log
echo "T18 A/B battery v2 start $(date -Is)" > "$OUT"

port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }
engine_pids() { pgrep -f "$REPO/engine/strata --serve" 2>/dev/null; }
cleanup_all() {
  local p pid
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(engine_pids); do kill -TERM "$pid" 2>/dev/null; done
  sleep 3
  for pid in $(engine_pids); do kill -KILL "$pid" 2>/dev/null; done
  pkill -KILL -f "serve/server.py --engine strata" 2>/dev/null
  sleep 1
}
trap cleanup_all EXIT

is_fixed() { strings "$1" 2>/dev/null | grep -q "exceeded the verify window"; }

leg() {  # $1 label  $2 stable source
  local label=$1 src=$2
  local logf=$RES/t18-ab-$label.out
  cleanup_all; sleep 2
  cp "$src" "$DEPLOY"
  local want=0 got
  is_fixed "$src" && want=1
  is_fixed "$DEPLOY" && got=1 || got=0
  if [ "$got" != "$want" ]; then echo "LEG $label DEPLOY VERIFY FAILED (want=$want got=$got)" >> "$OUT"; return; fi
  ( cd "$REPO" && nohup ./launch-custom-iq3s-8081.sh > "$logf" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 120); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $label health=$ok $(date +%H:%M:%S) binary=$([ "$want" = 1 ] && echo fixed || echo prefix) ===" >> "$OUT"
  if [ "$ok" != 1 ]; then echo "LEG $label FAILED" >> "$OUT"; tail -20 "$logf" >> "$OUT"; return; fi
  ( cd "$REPO" && .venv/bin/python campaign/probe-lane.py --port "$PORT" --label "t18ab-$label" >> "$OUT" 2>&1 )
  grep -E "M1 |M2 " "$OUT" | tail -3 | sed "s/^/[$label] /" >> "$RES/t18-ab-lines.txt"
}

if ! is_fixed "$SRC_FIXED"; then echo "SRC_FIXED lacks the marker - aborting" >> "$OUT"; exit 1; fi
if is_fixed "$SRC_PREFIX"; then echo "SRC_PREFIX has the marker - aborting" >> "$OUT"; exit 1; fi
: > "$RES/t18-ab-lines.txt"
for r in 1 2 3; do
  leg "fix-r$r" "$SRC_FIXED"
  leg "ctl-r$r" "$SRC_PREFIX"
done
cleanup_all
# restore the standing deploy: the fixed binary
cp "$SRC_FIXED" "$DEPLOY"
{ is_fixed "$DEPLOY" && echo "standing deploy restored: fixed binary" || echo "RESTORE FAILED"; } >> "$OUT"
trap - EXIT
echo "T18 A/B battery v2 end $(date -Is)" >> "$OUT"
