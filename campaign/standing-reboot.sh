#!/bin/bash
# campaign/standing-reboot.sh — the sanctioned way to reboot the standing server (F-479).
#
# Refuses while the operator's requests flow (battery-lib's require_quiet_standing). With the
# port quiet it SIGTERMs the standing, boots the launcher as svg, waits for /health AND the
# prewarm terminal state, and prints the boot console path. The operator's word is still
# required BEFORE this runs — the gate checks traffic, not permission (journal law, F-478
# addendum). Manual reboots no longer happen as ad-hoc command lines: 2026-11-26's 33 reboots
# in 24 h each held live requests at the door for up to 180 s.
#
# Usage: campaign/standing-reboot.sh [label]   — label names logs/<label>-<HHMM>.out
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8081}
. campaign/battery-lib.sh
LABEL="${1:-standing-reboot}"
OUT="logs/${LABEL}-$(date +%H%M).out"

require_quiet_standing "$PORT" 120 "$REPO/logs/custom-iq3_s.log"

p=$(port_pid)
if [ -n "$p" ]; then
  echo "stopping the standing (pid $p)"; kill -TERM "$p" 2>/dev/null; sleep 8
fi
if [ -n "$(port_pid)" ]; then echo "port $PORT still bound after TERM; aborting"; exit 1; fi

echo "booting: $OUT"
nohup sudo -u svg -H env XDG_RUNTIME_DIR=/run/user/1000 STRATA_SWAP_GUARD=1 \
  ./launch-custom-iq3s-8081.sh > "$OUT" 2>&1 &
chown svg:svg "$OUT" 2>/dev/null || true
wait_health 120 || { echo "no /health in 6 min — console: $OUT"; exit 1; }
wait_prewarm 240 || echo "WARNING: prewarm not terminal after 20 min — check $OUT"
echo "standing up: $(curl -s --max-time 5 "http://127.0.0.1:$PORT/health" | head -c 200)"
echo "console: $OUT"
