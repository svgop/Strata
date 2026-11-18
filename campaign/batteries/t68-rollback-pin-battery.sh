#!/bin/bash
# campaign/batteries/t68-rollback-pin-battery.sh — the safe-config improvement check (2026-11-18).
#
# Config under test: shadow ROLLED BACK (the canonical pack is corrupt on disk - 96 non-finite
# Q2_0 scales in 36/48 random chunks, .docs/research/2026-11-18-corrupt-shadow-pack.md) + the
# new pin default (engine on the isolated CCD, the bank-era regime) + zram empty + the 37.5 GiB
# freed.  Judge: means vs T66 (169.1/169.0/225.2 - same recipe, unpinned) - the delta prices the
# pin + memory-headroom recovery with the miss path held at the rollback's native IQ3_S cost.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t68-rollback-pin.log
echo "T68 ROLLBACK+PIN start $(date)" > "$OUT"

run_round() {  # run_round <round>
  local r=$1 lbl="t68-rollback-r$1"
  cleanup_engine
  ( cd "$REPO" && STRATA_SHADOW_EXPERTS="" nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 120); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $lbl health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  wait_prewarm
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tail -3 >> "$OUT" )
  grep -aE 'first token|window up to' "$REPO/logs/custom-iq3_s.log" | tail -2 >> "$OUT" 2>/dev/null
  grep -ac "#879" "$REPO/logs/custom-iq3_s.log" >> "$OUT" 2>/dev/null
  cleanup_engine
}

for r in 1 2 3; do run_round "$r"; done
echo "T68 ROLLBACK+PIN end $(date)" >> "$OUT"
echo "done" >> "$OUT"
