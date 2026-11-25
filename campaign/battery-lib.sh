#!/bin/bash
# campaign/battery-lib.sh — the shared battery infrastructure (source, never execute).
#
# The 31-battery fleet + the two ladders carried ~1,040 lines of this as copy-paste (26
# cleanup_engine() blocks were byte-identical), and the copies drifted three ways that bit us:
#   - the GPU kill sweep matched the SUBSTRING "strata" in any compute process's cmdline, which
#     killed whichever engine the OTHER lane had up (both launchers exec "server.py --engine
#     strata"; both engine binaries live in this repo) — 27 sites wide;
#   - the warm-in: 21 batteries still called campaign/prewarm2.sh and t64 called prewarm-ds4.sh
#     after those scripts were absorbed into the server itself (commit cc05319f) — a silent
#     no-op, so a battery could measure legs while the native warm-in was still replaying;
#   - the health-wait patience and the final-serve restore differed per copy for no reason.
#
# What belongs HERE is the machinery every boot pays; what stays in each battery is its
# experiment (arms, leg schedule, labels, probe flags).  Source it:
#     cd campaign && . ./battery-lib.sh                       # a ladder or a campaign/ script
#     . "$(dirname "$0")/../battery-lib.sh"                   # a battery in campaign/batteries/
# PORT must already be set by the caller (the fleet is 8081); REPO defaults to this checkout.

REPO=${REPO:-/home/svg/LLM/strata-custom-2}
PORT=${PORT:-8081}

# 2026-11-09 bootstorm: a battery's next leg boots while THIS leg's pack still sits in the page cache -
# the measured storm (kswapd +87.7 GiB pgsteal, +16k allocstalls in one load minute) is exactly that
# dead cache being reclaimed mid-load.  Drop it at every server stop instead (the drop runs after the
# engine process ends, where nothing latency-critical competes).  Off by default outside batteries: a
# same-model restart wants the cache warm.  An explicit value always wins.
export STRATA_DROP_CACHE_ON_EXIT=${STRATA_DROP_CACHE_ON_EXIT:-1}
# F-389 THE CAMPAIGN'S OWN SLOT DIR (2026-11-18, measured): a battery/probe run parks the
# conversations it creates (the probe's synthetic prompts, ~250-350 MB each) in the OPERATOR's
# conversation slot dir and every later boot restores and re-parks them - 15 root-owned synthetic
# slots (4.9 GB) were found sitting in /mnt/DATA/LLM/Models/GGUF/strata/data/conversation-slots, restored at every
# boot and re-parked.  The campaign's traffic is test rubbish until certified real traffic: it gets
# its own dir, cleaned at the start of every battery.  The operator's dir is never touched by a
# battery again.
export STRATA_CONV_SLOT_DIR=${STRATA_CONV_SLOT_DIR:-/tmp/strata-campaign-slots}
rm -rf "$STRATA_CONV_SLOT_DIR" 2>/dev/null || true
mkdir -p "$STRATA_CONV_SLOT_DIR" 2>/dev/null || true

port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }

# STRATA_BATTERY_DRYRUN=1 — a battery's pre-flight: its gates and guards fire (kernel checks,
# pool checks, engines-down refusals) but nothing boots, kills, or probes.  Every converted
# battery validates this way before it is trusted with a real leg.
battery_dryrun() { [ "${STRATA_BATTERY_DRYRUN:-0}" = "1" ]; }

cleanup_engine() {
  # Port-scoped TERM first — the server owns its engine process and closes it in order.  The GPU
  # sweep that follows is scoped to THIS repo's engine binaries only: never a bare "strata"
  # substring (it matched both engines' cmdlines and any in-repo parity bench), but the actual
  # paths this checkout launches (engine/ for the Qwen lane, build-fresh/ for the DS lane).
  local p pid c
  battery_dryrun && { echo "cleanup_engine: dry-run — would stop port $PORT (pid $(port_pid)) + this repo's GPU apps"; return 0; }
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *"$REPO/engine/"*|*"$REPO/build-fresh/"*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
  # then the port itself must go free — a boot racing the old listener's socket release is the
  # port-8081 race every long session has paid at least once (up to 30 s, then through anyway)
  local t
  for t in $(seq 1 30); do
    [ -z "$(port_pid)" ] && break
    sleep 1
  done
  pin_learned_profile
}

# F-406 THE BATTERY'S LAST DRAIN (2026-11-19): a battery's final exit drain — the parked synthetic
# conversations written as slot-<serial>.bin when its last engine ends — stayed in the campaign slot
# dir until the NEXT battery's start wipe.  /tmp is tmpfs: 3.0 GiB of dead test traffic held RAM
# between batteries (13 slots, root-owned, measured 2026-11-19 16:56).  cleanup_slots is the
# END-of-battery wipe.  It must NEVER run between a battery's boots: boot N+1 restores boot N's
# drained slots, which is the measured T66 restart axis — cleanup_engine stays slot-blind for
# exactly that reason.  The EXIT trap makes every shell that sources this lib clean on any exit
# path (abort included) without touching the kill discipline each battery owns; it wipes ONLY the
# campaign's declared dir — an operator-overridden STRATA_CONV_SLOT_DIR is never deleted.
cleanup_slots() {
  case "${STRATA_CONV_SLOT_DIR:-}" in
    /tmp/strata-campaign-slots) rm -rf "$STRATA_CONV_SLOT_DIR" 2>/dev/null || true ;;
  esac
}
trap 'cleanup_slots' EXIT

# F-271 as STRUCTURE, completed 2026-11-25 (F-453): the engine saves the learned profile when it EXITS, so a pin
# copied while an engine is up is overwritten by that engine's own exit - the 2026-11-20 incident, then again on
# 11-24, when the env ladder's restore boot undid the ladder's own just-verified re-pin.  The pin therefore belongs
# to the one function every battery runs once its last engine is down.  ONE pin file for every harness
# (STRATA_PROFILE_PIN; the default is the file three harnesses already named).  The standing recipe does not load
# the learned profile; a recipe that does would otherwise boot on the previous battery's synthetic traffic.
pin_learned_profile() {
  local pin="${STRATA_PROFILE_PIN:-$REPO/data/expert-profile-gold.bin}" dst="$REPO/data/expert-profile-learned.bin"
  battery_dryrun && { echo "pin_learned_profile: dry-run — would restore $dst from $pin"; return 0; }
  [ -f "$pin" ] || { echo "pin_learned_profile: no pin file at $pin - the learned profile stays as the engine saved it"; return 0; }
  [ -f "$dst" ] || { echo "pin_learned_profile: no learned profile at $dst - nothing to pin"; return 0; }
  local t pid busy
  for t in $(seq 1 30); do          # the exit save must be finished: wait until no engine of this repo is running
    busy=0
    for pid in $(pgrep -x strata); do
      case "$(readlink "/proc/$pid/exe" 2>/dev/null)" in "$REPO"/engine/*|"$REPO"/build-fresh/*) busy=1 ;; esac
    done
    [ "$busy" = 0 ] && break
    sleep 1
  done
  if [ "$busy" = 1 ]; then
    echo "pin_learned_profile: an engine of this repo is still running - the pin is deferred to its stop"
    return 1
  fi
  if cp -f "$pin" "$dst" && cmp -s "$pin" "$dst"; then
    echo "F-271 re-pin VERIFIED (learned == $(basename "$pin"), after the engine stop)"
  else
    echo "F-271 RE-PIN FAILED - pin it by hand before the next boot"
    return 1
  fi
}

wait_health() {  # wait_health [tries-of-3s] — any /health body means the API answers
  local t
  for t in $(seq 1 "${1:-120}"); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && return 0
    sleep 3
  done
  return 1
}

wait_prewarm() {  # wait_prewarm [tries-of-5s] — the server's own warm-in must reach a terminal
  # state before any leg runs (campaign law 11: warmup is part of the bench).  "done" it warmed;
  # "failed"/"yielded" it ended on its own terms (live traffic pre-empted it); "" the config has
  # no prewarm section — all three are terminal.  Non-terminal: idle/files/replay.
  local t s
  for t in $(seq 1 "${1:-240}"); do
    s=$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | sed -n 's/.*"prewarm": "\([a-z]*\)".*/\1/p')
    case "$s" in done|failed|yielded|"") return 0 ;; esac
    sleep 5
  done
  echo "wait_prewarm: no terminal warm state in 20 min (continuing)"
}

final_serve() {  # final_serve <launcher> <out-log> — leave the standing server up for the operator
  ( cd "$REPO" && nohup bash "$1" >"$2" 2>&1 & )
  wait_health || echo "final_serve: the server did not answer /health in 6 min"
}
