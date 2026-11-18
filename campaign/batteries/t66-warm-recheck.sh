#!/bin/bash
# campaign/batteries/t66-warm-recheck.sh — the post-NaN-rollback regression check (2026-11-18).
#
# The bank anchors (219.6/210.4/290.3, warm multi-leg; record class 227.6/210.1/293.9) were
# built with the t31 BASE recipe: standing launcher, zero env changes, boot -> prewarm ->
# probe-lane -> cleanup per leg.  After the shadow-store rollback (385937a4d) and the estate
# move onto /mnt/DATA, this battery re-runs that exact recipe 3 legs to prove the shallow
# rate is intact.  Judge: means inside the bank band (solo >= 210, pair >= 200, conc >= 280
# or within the boot-to-boot variance the anchors themselves showed).
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t66-warm-recheck.log
echo "T66 WARM RECHECK start $(date)" > "$OUT"

[ -n "$(port_pid)" ] && { echo "NOTE: live engine up at start (idle confirmed) — leg 1 stops it" >> "$OUT"; }

run_round() {  # run_round <round>
  local r=$1 lbl="t66-warm-r$1"
  cleanup_engine
  ( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/$lbl.out" 2>&1 & )
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
  cleanup_engine
}

for r in 1 2 3; do run_round "$r"; done
echo "T66 WARM RECHECK end $(date)" >> "$OUT"
echo "done" >> "$OUT"
