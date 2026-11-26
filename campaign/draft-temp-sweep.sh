#!/usr/bin/env bash
# campaign/draft-temp-sweep.sh — the STRATA_SPEC_DRAFT_TEMP sweep on the campaign's boot
# scaffolding.  The standing drafter samples at 0.85 (launch-custom-iq3s-8081.sh, F-201 ADOPTED;
# F-299's post-SPEC_PROB re-sweep closed 0.75/0.85/0.95 neutral).  This driver sweeps LOWER
# values — 0.65 / 0.5 / 0.4 by default — one COLD boot per value, M1 legs per campaign law 11.
#
#   usage: bash campaign/draft-temp-sweep.sh <label> [temp ...]
#     bash campaign/draft-temp-sweep.sh f470-dt                    # 0.65 / 0.5 / 0.4 (the asked sweep)
#     LEGS=3 bash campaign/draft-temp-sweep.sh f470-dt 0.85 0.65 0.5 0.4   # adds the control arm
#   LEGS (default 3) measured probe legs per arm AFTER one full discarded leg; SWEEP_TEMPS
#   ("0.65 0.5 0.4") is the env form of the arm list; ROUNDS>1 reverses the arm order on even
#   rounds (the boot-order balance — a 2-arm list becomes exactly ABAB; a 3+-arm list is
#   round-reversed, not a full Latin square — the honest limit of the simple form).
#   Campaign law 17's verdict needs a same-hour control: pass 0.85 as the first arm.
#
#   Per arm: a derived launcher with STRATA_SPEC_DRAFT_TEMP=<dt> armed AFTER the standing
#   exports (launcher_with_env in battery-lib.sh — the launcher's own hard export at 0.85 would
#   silently override a value passed on the `env` command line, the F-284 class), the boot waits
#   /health then the "prewarm" word's terminal state, the /proc environ check PROVES what armed
#   (a mismatch warns before any leg is measured), then the law-11 leg schedule:
#     leg 0  discarded (absorbs first-touch faults; campaign law 11 / F-360)
#     legs 1..LEGS  measured, each  campaign/probe-lane.py | tee campaign/results/<file>.txt
#   with <file> = <label>-dt<dt>[-r<k>]-{discard,leg<k>}.txt.  Means over measured legs only.
#   Between arms: SIGTERM stop (cleanup_engine: port-scoped TERM + repo-scoped GPU sweep + the
#   F-271 learned-profile pin) and a fresh scratch slot dir (/tmp/draft-temp-slots, wiped).
#   Refuses to start while an engine or port 8081 is up.  Leaves the port DOWN at the end.
#   Logs: campaign/results/<label>/battery.log (+ per-arm boot logs under it) and the summary
#   JSON campaign/results/<label>-dt-sweep.json.
#
#   THE M3b CANARY (the coupled contract).  Exact-match verify emits the TARGET's picks, so the
#   OUTPUT bytes are draft-temp-invariant — the contract the sweep must watch.  The probe's M3b
#   line is parsed per arm and a LOUD warning prints if the parsed values differ across arms.
#   Reading a mover: either the coupled contract broke (a BUG — the output can no longer be
#   assumed temp-invariant) or the acceptance RATE moved by its normal mechanism (the rate is
#   the speed dial F-201 tuned: proposal distribution x temp).  Tell them apart with the
#   output-byte checks (campaign/bit-check.py, probe-metrics.py golden hash) before calling it
#   a tradeoff.  The canary verdict is recorded in the summary JSON either way.
#
#   Dry-run pre-flight (nothing boots, kills or probes — battery-lib's STRATA_BATTERY_DRYRUN):
#     STRATA_BATTERY_DRYRUN=1 bash campaign/draft-temp-sweep.sh <label>
set -u
cd "$(dirname "$0")/.."
. campaign/battery-lib.sh

LABEL=${1:?label}; shift
TEMPS=("$@")
[ ${#TEMPS[@]} -eq 0 ] && IFS=' ' read -r -a TEMPS <<< "${SWEEP_TEMPS:-0.65 0.5 0.4}"
LEGS=${LEGS:-3}
ROUNDS=${ROUNDS:-1}
GP=.venv/bin/python; [ -x "$GP" ] || GP=python3
ROOT=campaign/results/$LABEL
mkdir -p "$ROOT"
SLOTS=/tmp/draft-temp-slots
STAMP=$(date +%H%M%S)
say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$ROOT/battery.log"; }
engine_up() { pgrep -f '[e]ngine/strata --serve' >/dev/null; }

if engine_up || [ -n "$(port_pid)" ]; then say "REFUSING: an engine or port $PORT is up"; exit 1; fi
# set only AFTER the refusal: an abort from here on may clean, an accidental run against a live
# server must never reach a kill
trap 'cleanup_engine 2>/dev/null; rm -rf "${SLOTS:?}"; chown -R svg:svg "$ROOT" 2>/dev/null || true' EXIT

run_arm() {  # $1 = dt value, $2 = round tag ('' for ROUNDS=1)
    local dt=$1 rnd=$2
    local tag="dt$dt${rnd:+-$rnd}"
    local OUT=$ROOT/$tag-$(date +%H%M%S)
    mkdir -p "$OUT"
    local LAUNCH=launch-arm-$STAMP-$tag.sh
    launcher_with_env "$PWD/$LAUNCH" "STRATA_SPEC_DRAFT_TEMP=$dt"
    rm -rf "${SLOTS:?}" && mkdir -p "$SLOTS" && chown svg:svg "$SLOTS"
    say "boot $tag: STRATA_SPEC_DRAFT_TEMP=$dt armed in derived launcher $LAUNCH (after the standing exports)"
    if battery_dryrun; then
        say "  DRY-RUN: would boot $LAUNCH on scratch $SLOTS, wait /health + prewarm, run 1 discarded + $LEGS measured legs of $GP campaign/probe-lane.py --port $PORT, SIGTERM stop"
        rm -f "${PWD:?}/${LAUNCH:?}"
        return 0
    fi
    sudo -u svg -H env XDG_RUNTIME_DIR=/run/user/1000 STRATA_SWAP_GUARD=1 STRATA_CONV_SLOT_DIR="$SLOTS" \
        setsid nohup bash "$LAUNCH" > "$OUT/server.log" 2>&1 < /dev/null &
    chown svg:svg "$OUT/server.log" 2>/dev/null
    local T0=$(date +%s) armed
    wait_health || { say "  ABORT $tag: no /health answer in 6 min (boot log $OUT/server.log); stopping whatever came up before the next arm boots"; cleanup_engine; return 1; }
    wait_prewarm
    armed=$(tr '\0' '\n' < "/proc/$(pgrep -x strata | head -1)/environ" 2>/dev/null \
            | sed -n 's/^STRATA_SPEC_DRAFT_TEMP=//p' | head -1)
    if [ "$armed" = "$dt" ]; then
        say "  warm-in terminal after $(( $(date +%s) - T0 )) s; engine environ STRATA_SPEC_DRAFT_TEMP=$armed VERIFIED"
    else
        say "  WARNING: engine environ STRATA_SPEC_DRAFT_TEMP='${armed:-MISSING}' != requested '$dt' — the F-284 class: the legs below would measure the WRONG arm; read no verdict from this boot"
    fi
    ( cd "$REPO" && "$GP" campaign/probe-lane.py --port "$PORT" --label "$LABEL-$tag-discard" ) 2>&1 \
        | tee "campaign/results/$LABEL-$tag-discard.txt" | grep -E '^M1 single-stream' | sed 's/^/[discard] /'
    local l
    for l in $(seq 1 "$LEGS"); do
        ( cd "$REPO" && "$GP" campaign/probe-lane.py --port "$PORT" --label "$LABEL-$tag-leg$l" ) 2>&1 \
            | tee "campaign/results/$LABEL-$tag-leg$l.txt" | grep -E '^M1 single-stream|^M3b rate'
    done
    cleanup_engine
    rm -f "${PWD:?}/${LAUNCH:?}"
    say "  stopped $tag; the boot took $(( $(date +%s) - T0 )) s"
}

for r in $(seq 1 "$ROUNDS"); do
    RND=$([ "$ROUNDS" -gt 1 ] && echo "r$r" || true)
    if [ $((r % 2)) -eq 1 ]; then
        for dt in "${TEMPS[@]}"; do run_arm "$dt" "$RND"; done
    else
        for dt in $(printf '%s\n' "${TEMPS[@]}" | tac); do run_arm "$dt" "$RND"; done
    fi
done

if battery_dryrun; then say "battery done (dry-run: $ROUNDS round(s) x ${#TEMPS[@]} arms, plan above)"; exit 0; fi

# THE SUMMARY: per-arm M1 means over the measured legs (never the discard), the M3b canary
# record, and the loud warning if the parsed M3b values differ across arms.
"$GP" - "$LABEL" "$ROUNDS" "$LEGS" "${TEMPS[@]}" <<'PY'
import glob, json, re, statistics as st, sys
label, rounds, LEGS, temps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
def legs(dt, r):
    tag = f"dt{dt}{f'-r{r}' if rounds > 1 else ''}"
    return [open(f).read() for f in sorted(glob.glob(f"campaign/results/{label}-{tag}-leg*.txt"))]
def vals(txt, pat):
    return [float(m.group(1)) for m in re.finditer(pat, txt)]
m1_pat, m3_pat = r"M1 single-stream : ([\d.]+) t/s", r"M3b rate         : ([\d.]+)%"
arms = {}
for dt in temps:
    m1, m3 = [], []
    for r in range(1, rounds + 1):
        for txt in legs(dt, r):
            m1 += vals(txt, m1_pat); m3 += vals(txt, m3_pat)
    arms[dt] = {"m1_legs": m1, "m3b_legs": m3,
                "m1_mean": st.mean(m1) if m1 else None, "m3b_mean": st.mean(m3) if m3 else None}
for dt, a in arms.items():
    m1 = " / ".join(f"{v:.1f}" for v in a["m1_legs"]) or "NO LEGS READ"
    m3 = " / ".join(f"{v:.1f}" for v in a["m3b_legs"]) or "-"
    print(f"dt{dt}: M1 legs {m1}  mean {a['m1_mean'] and round(a['m1_mean'], 1)} | M3b legs {m3}  mean {a['m3b_mean'] and round(a['m3b_mean'], 1)}")
read = {dt: a for dt, a in arms.items() if a["m3b_mean"] is not None and a["m1_mean"] is not None}
m3b = {dt: round(a["m3b_mean"], 1) for dt, a in read.items()}
if len(m3b) < 2:
    canary = None
    print("M3b canary: NOT COMPARABLE — fewer than two arms produced parseable M3b lines")
elif len(set(m3b.values())) == 1:
    canary = True
    print(f"M3b canary: IDENTICAL across arms ({list(m3b.values())[0]}%) — the coupled contract holds on the rate line")
else:
    canary = False
    detail = "  ".join(f"dt{dt}={v}%" for dt, v in m3b.items())
    print(f">>> M3B CANARY MOVED: {detail} — the parsed M3b values DIFFER across arms.  A mover is a BUG (the coupled")
    print(">>> contract broke; the output can no longer be assumed draft-temp-invariant) OR the acceptance rate moving")
    print(">>> by its normal mechanism (the rate is the speed dial F-201 tuned).  Separate them with the output-byte")
    print(">>> checks (campaign/bit-check.py, probe-metrics.py golden hash) before calling it a tradeoff.")
missing = [dt for dt, a in arms.items() if a["m1_mean"] is None]
if missing:
    print(f"NO LEGS READ for: {' '.join(missing)} — those boots produced no parseable M1 lines; read no verdict from them")
json.dump({"label": label, "arms": {dt: {**a, "m1_mean": a["m1_mean"] and round(a["m1_mean"], 2),
                                          "m3b_mean": a["m3b_mean"] and round(a["m3b_mean"], 2)} for dt, a in arms.items()},
           "legs_measured_per_arm": LEGS, "rounds": rounds,
           "m3b_canary_identical": canary, "m3b_by_arm": m3b},
          open(f"campaign/results/{label}-dt-sweep.json", "w"), indent=1)
PY
say "battery done: ${#TEMPS[@]} arm(s) x $ROUNDS round(s); summary campaign/results/$LABEL-dt-sweep.json"
