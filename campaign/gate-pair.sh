#!/usr/bin/env bash
# campaign/gate-pair.sh — the paired COLD-BOOT MMLU gate driver: one boot per arm, the gate run
# ONCE per boot.  This is the structural fix for the F-468/F-469 reuse confound.
#
#   NEVER TWO GATES ON ONE BOOT.  The second gate re-asks identical prompts and the conversation
#   cache serves run 1's parked KV back (F-450-class nondeterminism): the 2026-11-25 same-boot
#   pair moved 88.6 vs 97.1 and no one can say which arm moved — the boots were not separated
#   (campaign/results/f468-2026-11-25/mmlu-*.txt, journal F-468/F-469).  Here every arm gets its
#   own cold boot on a WIPED scratch slot dir, so no prompt of arm B ever meets a cache arm A
#   warmed.  ROUNDS=2 runs A,B then B,A — the order-reversed repeat that splits boot-order drift
#   evenly across the arms (campaign law 1, at boot granularity).
#
#   usage: bash campaign/gate-pair.sh <label> '<env A>' '<gate args A>' '<env B>' '<gate args B>'
#     <env X>     ONE quoted string of VAR=value pairs for the BOOT ('' = none), e.g.
#                 'STRATA_SPEC_DRAFT_TEMP=0.65' or 'STRATA_VERIFY_DEVICE_LAUNCH=0'.
#                 Presence is NOT an env — it is a gate flag: '--presence 1.5'.
#     <gate args> flags for campaign/mmlu-pro-gate.py AFTER its "<floor> <label>" positionals,
#                 e.g. '--suite 1500' or '--suite 1500 --presence 1.5 --repeats 3'.
#                 --repeats K (the gate's in-boot variance form) composes: the cold-boot pair
#                 fixes the BETWEEN-boot confound, the repeats measure the WITHIN-boot spread.
#   Per-arm gate labels are <label>-A / <label>-B (suffixed -r<k> when ROUNDS > 1), so the two
#   result JSONs campaign/results/mmlu-<label>-{A,B}.json never collide.
#
#   examples:
#     presence-vs-control, the 1500q pair with 3 in-boot repeats:
#       bash campaign/gate-pair.sh f470-pres '' '--suite 1500 --repeats 3' \
#                                       '' '--suite 1500 --presence 1.5 --repeats 3'
#     an env arm pair instead of a flag arm:
#       bash campaign/gate-pair.sh f470-dt '' '--suite 1500' 'STRATA_SPEC_DRAFT_TEMP=0.65' '--suite 1500'
#
#   Per arm: wipe the scratch slots (/tmp/gate-pair-slots), boot the operator's launcher with the
#   arm env (a derived launcher whose overrides are armed AFTER the standing exports —
#   launcher_with_env in battery-lib.sh; an env passed on the command line would be silently
#   overwritten by the launcher's hard exports, the F-284 class), wait for /health then the
#   "prewarm" word's terminal state, log the arm env AS THE ENGINE PROCESS SEES IT, run the gate
#   once, SIGTERM stop (cleanup_engine: port-scoped TERM + the repo-scoped GPU sweep + the F-271
#   learned-profile pin), wipe the scratch.  Refuses to start while an engine or port 8081 is up.
#   Leaves the port DOWN at the end.  GATE_FLOOR (default 0.933, the escalation floor) is the
#   gate's accuracy-floor positional; the SD over repeats is judged by the gate's own VERDICT line.
#   Logs: campaign/results/<label>/battery.log; per-arm boot log + full per-question gate stream:
#   campaign/results/<label>/<arm>-<time>/{server.log,gate.txt}.
#
#   Dry-run pre-flight (gates and guards fire; nothing boots, kills or probes — battery-lib's
#   STRATA_BATTERY_DRYRUN):
#     STRATA_BATTERY_DRYRUN=1 bash campaign/gate-pair.sh ...
set -u
cd "$(dirname "$0")/.."
. campaign/battery-lib.sh

LABEL=${1:?label}
# NOTE: an arm's env string may legitimately be EMPTY ('' = no boot-env override), so the args
# are checked by COUNT, not by :? — a :? on an empty string refuses the valid '' form.
[ $# -ge 5 ] || { echo "usage: bash campaign/gate-pair.sh <label> '<env A>' '<gate args A>' '<env B>' '<gate args B>'" >&2; exit 2; }
ENVA=$2; GATEA=$3; ENVB=$4; GATEB=$5
ROUNDS=${ROUNDS:-1}
GATE_FLOOR=${GATE_FLOOR:-0.933}
GP=.venv/bin/python; [ -x "$GP" ] || GP=python3
ROOT=campaign/results/$LABEL
mkdir -p "$ROOT"
SLOTS=/tmp/gate-pair-slots
STAMP=$(date +%H%M%S)
say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$ROOT/battery.log"; }
engine_up() { pgrep -f '[e]ngine/strata --serve' >/dev/null; }

if engine_up || [ -n "$(port_pid)" ]; then say "REFUSING: an engine or port $PORT is up"; exit 1; fi
# set only AFTER the refusal: an abort from here on may clean, an accidental run against a live
# server must never reach a kill
trap 'cleanup_engine 2>/dev/null; rm -rf "${SLOTS:?}"; chown -R svg:svg "$ROOT" 2>/dev/null || true' EXIT

armed_env_of() {  # armed_env_of <engine-pid> <env string> — what the ENGINE process actually carries
    local pid=$1 names="" v armed
    for v in $2; do names+="^${v%%=*}=|"; done
    names=${names%|}
    [ -z "$names" ] && return 0
    armed=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -E -- "$names" | tr '\n' ' ')
    printf '%s' "${armed% }"
}

run_arm() {  # $1 = A|B, $2 = round tag ('' for ROUNDS=1), $3 = env string, $4 = gate args
    local arm=$1 rnd=$2 armenv=$3 gateargs=$4
    local gl="$LABEL-$arm${rnd:+-$rnd}"
    local OUT=$ROOT/$arm${rnd:+-$rnd}-$(date +%H%M%S)
    mkdir -p "$OUT"
    local LAUNCH=launch-arm-$STAMP-$arm.sh
    # shellcheck disable=SC2086 — the env string is VAR=value pairs by contract
    launcher_with_env "$PWD/$LAUNCH" $armenv
    rm -rf "${SLOTS:?}" && mkdir -p "$SLOTS" && chown svg:svg "$SLOTS"
    say "boot arm $arm${rnd:+ (round $rnd)}: boot env '${armenv:-none}'; gate: $GP campaign/mmlu-pro-gate.py $GATE_FLOOR $gl $gateargs"
    if battery_dryrun; then
        say "  DRY-RUN: would boot $LAUNCH on scratch $SLOTS, wait /health + prewarm, run the gate once, SIGTERM stop"
        rm -f "${PWD:?}/${LAUNCH:?}"
        return 0
    fi
    sudo -u svg -H env XDG_RUNTIME_DIR=/run/user/1000 STRATA_SWAP_GUARD=1 STRATA_CONV_SLOT_DIR="$SLOTS" \
        setsid nohup bash "$LAUNCH" > "$OUT/server.log" 2>&1 < /dev/null &
    chown svg:svg "$OUT/server.log" 2>/dev/null
    local T0=$(date +%s) armed ep
    wait_health || { say "  ABORT arm $arm: no /health answer in 6 min (boot log $OUT/server.log); stopping whatever came up before arm B boots"; cleanup_engine; return 1; }
    wait_prewarm
    say "  warm-in terminal after $(( $(date +%s) - T0 )) s; $(grep -m1 -E '^\[strata\] control vector' "$OUT/server.log" | cut -c9-80)"
    ep=$(pgrep -x strata | head -1)
    armed=$(armed_env_of "$ep" "$armenv")
    say "  armed env as the engine process sees it: ${armed:-none}"
    ( cd "$REPO" && "$GP" campaign/mmlu-pro-gate.py "$GATE_FLOOR" "$gl" $gateargs ) > "$OUT/gate.txt" 2>&1
    local res; res=$(grep -hE '^=== MMLU-Pro|^VERDICT|^TRANSPORT FAILURES' "$OUT/gate.txt" | tr '\n' '|')
    say "  gate: ${res:-NO SUMMARY LINE — read $OUT/gate.txt}"
    cleanup_engine
    rm -f "${PWD:?}/${LAUNCH:?}"
    say "  stopped arm $arm; the boot took $(( $(date +%s) - T0 )) s"
}

for r in $(seq 1 "$ROUNDS"); do
    RND=$([ "$ROUNDS" -gt 1 ] && echo "r$r" || true)
    if [ $((r % 2)) -eq 1 ]; then
        run_arm A "$RND" "$ENVA" "$GATEA"; run_arm B "$RND" "$ENVB" "$GATEB"
    else
        run_arm B "$RND" "$ENVB" "$GATEB"; run_arm A "$RND" "$ENVA" "$GATEA"
    fi
done
say "battery done: $(( ROUNDS * 2 )) gated boots (one gate per boot — the pair is clean)"
