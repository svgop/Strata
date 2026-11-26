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
# set -e + pipefail kills this script silently when `port_pid`'s grep finds no listener (grep
# exits 1 through the pipe) — the exact trap the second sanctioned-reboot run hit (F-483). Keep
# -u (typo guard) and pipefail (my own pipelines), handle failures explicitly instead of -e.
set -uo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8081}
. campaign/battery-lib.sh
LABEL="${1:-standing-reboot}"
OUT="logs/${LABEL}-$(date +%H%M).out"

# STRATA_PASS_ENV: named KEY=VALUE items armed on the engine process (the hit-99 stats leg:
# STRATA_PASS_ENV="STRATA_LOOKAHEAD_STATS=1"). The sudo boundary strips the caller's
# environment (verified 2026-11-26: env_reset drops STRATA_* before the launcher runs), so an
# export beside this script boots UNARMED and silently - the no-op-boot class F-205/F-483
# made law. Each item is validated (NAME=VALUE, NAME [A-Za-z_][A-Za-z0-9_]*) and passed as
# one argv element; nothing from the caller's environment reaches the engine unnamed.
PASS_ITEMS=()
set -f                                   # the split is a list, never a glob
for kv in ${STRATA_PASS_ENV:-}; do
  if [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    PASS_ITEMS+=("$kv")
  else
    echo "STRATA_PASS_ENV item is not NAME=VALUE: $kv - refusing"
    exit 1
  fi
done
set +f

# STRATA_REBOOT_DRY=1: print the boot command this run would execute and exit. Touches
# nothing (no gate, no stop, no boot) - the wrapper-validation law made cheap: verify the
# constructed command before spending a boot on it.
if [ "${STRATA_REBOOT_DRY:-0}" = 1 ]; then
  echo "dry-run, would run: nohup sudo -u svg -H env XDG_RUNTIME_DIR=/run/user/1000 STRATA_SWAP_GUARD=1" \
       "${PASS_ITEMS[@]+"${PASS_ITEMS[*]}"}" "./launch-custom-iq3s-8081.sh"
  exit 0
fi

require_quiet_standing "$PORT" 120 "$REPO/logs/custom-iq3_s.log"

p=$(port_pid || true)
if [ -n "$p" ]; then
  echo "stopping the standing (pid $p) — it closes its engine and parks its conversations in order"
  kill -TERM "$p" 2>/dev/null
  # the ordered shutdown can outlast a fixed sleep (parking 12+ conversation slots of ~250-300 MB
  # each): poll the port like cleanup_engine does, up to 60 s, and only then abort (a boot racing
  # the listener's release is the port-8081 race every long session has paid at least once).
  for t in $(seq 1 60); do
    [ -z "$(port_pid)" ] && { echo "port free after ${t}s"; break; }
    [ "$t" -eq 60 ] && { echo "port $PORT still bound after 60 s of ordered shutdown; aborting"; exit 1; }
    sleep 1
  done
fi

echo "booting: $OUT"
nohup sudo -u svg -H env XDG_RUNTIME_DIR=/run/user/1000 STRATA_SWAP_GUARD=1 \
  ${PASS_ITEMS[@]+"${PASS_ITEMS[@]}"} ./launch-custom-iq3s-8081.sh > "$OUT" 2>&1 &
if [ ${#PASS_ITEMS[@]} -gt 0 ]; then
  # the boot console carries what booted (law 14: the deployed process is the change)
  echo "[standing-reboot] armed env: ${PASS_ITEMS[*]}" >> "$OUT"
  echo "armed: ${PASS_ITEMS[*]}"
fi
chown svg:svg "$OUT" 2>/dev/null || true
wait_health 120 || { echo "no /health in 6 min — console: $OUT"; exit 1; }
wait_prewarm 240 || echo "WARNING: prewarm not terminal after 20 min — check $OUT"
echo "standing up: $(curl -s --max-time 5 "http://127.0.0.1:$PORT/health" | head -c 200)"
echo "console: $OUT"
