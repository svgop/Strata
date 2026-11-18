#!/bin/bash
# campaign/reset-state.sh — F-389: a restart leaves NO rubbish behind.
# The engine drains its parked conversations on exit (~5 GB, seconds AFTER the kill) and the next
# boot restores them; a clear that races the drain is undone.  This tool stops the engine, waits for
# the processes to exit AND the drain to finish (the slot dir must stop changing), clears the
# CAMPAIGN slot dir (the probe's synthetic conversations - never the operator's, unless asked),
# drops the page cache, and prints the memory state it leaves.  Then it optionally boots + certifies.
#   campaign/reset-state.sh                 # stop + clean
#   campaign/reset-state.sh --operator-slots # also clear the operator's synthetic leftovers (explicit)
#   campaign/reset-state.sh --boot [--store DIR] [--bar N]   # then boot (and certify)
set -u
. "$(dirname "$0")/battery-lib.sh"
REPO=${REPO:-/home/svg/LLM/strata-custom-2}
OP=0; BOOT=0; STORE=""; BAR=0
while [ $# -gt 0 ]; do case "$1" in
  --operator-slots) OP=1; shift;;
  --boot) BOOT=1; shift;;
  --store) STORE=$2; shift 2;;
  --bar) BAR=$2; shift 2;;
  *) echo "reset-state: unknown $1" >&2; exit 2;; esac; done
cleanup_engine
for i in $(seq 1 30); do
  pgrep -f 'serve/server.py --engine strata' >/dev/null 2>&1 || pgrep -f 'engine/strata' >/dev/null 2>&1 || break
  sleep 1
done
last=""; stable=0
for i in $(seq 1 30); do
  now=$(ls -la "$STRATA_CONV_SLOT_DIR" 2>/dev/null | md5sum | cut -d' ' -f1)
  [ "$now" = "$last" ] && stable=$((stable+1)) || stable=0
  last="$now"; [ "$stable" -ge 3 ] && break
  sleep 1
done
rm -rf "$STRATA_CONV_SLOT_DIR"; mkdir -p "$STRATA_CONV_SLOT_DIR"
[ "$OP" = 1 ] && rm -f /home/svg/LLM/data-rt/conversation-slots/slot-*.bin
sync
( echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null ) 2>/dev/null || true
echo "reset-state: $(date +%H:%M:%S) engine stopped, campaign slots cleared: $(ls "$STRATA_CONV_SLOT_DIR" 2>/dev/null | wc -l) files; operator slots: $(ls /home/svg/LLM/data-rt/conversation-slots/ 2>/dev/null | wc -l) files; page cache dropped"
grep -aE 'MemFree|MemAvailable' /proc/meminfo
[ "$BOOT" = 1 ] || exit 0
if [ "${BAR:-0}" -gt 0 ]; then
  exec "$REPO/campaign/boot-certify.sh" --bar "$BAR" --tries 3 ${STORE:+--store "$STORE"}
fi
( cd "$REPO" && env ${STORE:+STRATA_SHADOW_EXPERTS=$STORE} nohup bash "$REPO/launch-custom-iq3s-8081.sh" >"$REPO/logs/reset-boot.out" 2>&1 < /dev/null & )
wait_health && echo "reset-state: booted" || echo "reset-state: the boot did not answer /health"
