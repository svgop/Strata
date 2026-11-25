#!/usr/bin/env bash
# campaign/loop-arms.sh - boot the standing and single-change arms in turn and measure loop propensity in each boot
# (F-468 follow-up, 2026-11-25).
#
#   usage: bash campaign/loop-arms.sh <label> <arm> [<arm> ...]        example: bash campaign/loop-arms.sh loop-arms S L0 P0 R0
#
#     S   the standing
#     L0  the window graph launched from the HOST (STRATA_VERIFY_DEVICE_LAUNCH=0: F-452 off)
#     P0  the persona table removed   (campaign/results/f468-2026-11-25/config-p0.json)
#     R0  the refusal table removed   (campaign/results/f468-2026-11-25/config-r0.json)
#
# One boot an arm.  A boot is the operator's launcher with one line changed (the config path), started from the repo
# root as svg under the swap guard, on its own conversation-slot scratch under /tmp - never the operator's slot
# directory - then the server's warm-in, then campaign/loop-propensity.py with the request arm "standing" only (the
# boot IS the arm), then a clean stop (SIGTERM).  Results: campaign/results/<label>/<arm>-<time>/{server.log,
# replies.txt} and one line a boot in campaign/results/<label>/battery.log.  Refuses to start while an engine or the
# port is up.  Eight boots of six replies are about fifty minutes.  ARMS, REPEAT and PROMPTS (environment) change the
# request arms campaign/loop-propensity.py sends and how many replies: ARMS=standing,effort-max REPEAT=2.
set -u
cd "$(dirname "$0")/.."
LABEL=${1:?label}; shift
ROOT=campaign/results/$LABEL
mkdir -p "$ROOT"
SLOTS=/tmp/loop-arms-slots
STAMP=$(date +%H%M%S)
say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$ROOT/battery.log"; }
port_pid() { ss -ltnp 2>/dev/null | grep ':8081 ' | grep -oP 'pid=\K[0-9]+' | head -1; }
engine_up() { pgrep -f '[e]ngine/strata --serve' >/dev/null; }

if engine_up || [ -n "$(port_pid)" ]; then say "REFUSING: an engine or port 8081 is up"; exit 1; fi
for arm in "$@"; do
    case $arm in
        S)  CFG=config/custom-iq3_s-8081.json; EXTRA=() ;;
        L0) CFG=config/custom-iq3_s-8081.json; EXTRA=(STRATA_VERIFY_DEVICE_LAUNCH=0) ;;
        P0) CFG=campaign/results/f468-2026-11-25/config-p0.json; EXTRA=() ;;
        R0) CFG=campaign/results/f468-2026-11-25/config-r0.json; EXTRA=() ;;
        *) say "unknown arm: $arm"; exit 1 ;;
    esac
    OUT=$ROOT/$arm-$(date +%H%M%S)
    mkdir -p "$OUT"
    rm -rf "${SLOTS:?}" && mkdir -p "$SLOTS" && chown svg:svg "$SLOTS"
    LAUNCH="launch-arm-$STAMP.sh"
    sed "s#--config config/custom-iq3_s-8081.json#--config $CFG#" launch-custom-iq3s-8081.sh > "$LAUNCH"
    chmod 755 "$LAUNCH" && chown svg:svg "$LAUNCH"
    say "boot $arm: config $CFG ${EXTRA[*]:-}"
    sudo -u svg -H env XDG_RUNTIME_DIR=/run/user/1000 STRATA_SWAP_GUARD=1 STRATA_CONV_SLOT_DIR="$SLOTS" ${EXTRA[@]+"${EXTRA[@]}"} \
        setsid nohup bash "$LAUNCH" > "$OUT/server.log" 2>&1 < /dev/null &
    chown svg:svg "$OUT/server.log" 2>/dev/null
    T0=$(date +%s)
    for i in $(seq 1 100); do curl -s -m 2 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 3; done
    SW=none
    for i in $(seq 1 60); do
        SW=$(curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | sed -n 's/.*"prewarm": "\([a-z]*\)".*/\1/p')
        case "$SW" in done|failed|yielded) break ;; esac
        sleep 3
    done
    say "  warm-in $SW after $(( $(date +%s) - T0 )) s; $(grep -m1 -E '^\[strata\] control vector' "$OUT/server.log" | cut -c9-80)"
    python3 campaign/loop-propensity.py --prompts "${PROMPTS:-3}" --repeat "${REPEAT:-2}" --arms "${ARMS:-standing}" --log "$OUT/server.log" > "$OUT/replies.txt" 2>&1
    say "  loops: $(grep -cE '^prompt [0-9]+ standing +loop' "$OUT/replies.txt") of $(grep -cE '^prompt [0-9]+ standing' "$OUT/replies.txt") replies ($(grep -oE 'peak [0-9.]+' "$OUT/replies.txt" | cut -d' ' -f2 | sort -rn | head -1 | sed 's/^/worst peak /'))"
    P=$(port_pid)
    [ -n "$P" ] && sudo -u svg -H kill -TERM "$P"
    for i in $(seq 1 45); do [ -z "$(port_pid)" ] && ! engine_up && break; sleep 2; done
    say "  stopped; the boot took $(( $(date +%s) - T0 )) s"
    rm -f "${PWD:?}/${LAUNCH:?}"
done
rm -rf "${SLOTS:?}"
say "battery done"
