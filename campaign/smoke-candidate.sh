#!/bin/bash
# campaign/smoke-candidate.sh - the smoke leg every candidate binary pays BEFORE a ladder.
#
# One boot of whatever is in engine/strata (this script deploys nothing), the warm protocol, one full probe leg, and
# three checks a ladder assumes and never made:
#   1. the single-stream lane answers,
#   2. THE CONCURRENT LANE ANSWERS: tokens > 0 and no request of the leg ended in an error,
#   3. the engine was not restarted during the leg.
# F-453 (2026-11-26): a per-round telemetry edit inverted batch_poll's completion sense (0 = still running).  The
# candidate compiled, passed its kernel parities, and hung the batch lane with every new environment switch off; an env
# ladder then scored three zero-token concurrent legs as readings before the server's log was read.  A default-off
# switch does not make an ungated control-flow edit safe, and a kernel parity test does not see the serve loop.
#
# Usage: bash campaign/smoke-candidate.sh [VAR=value ...] <tag>
#   The VAR=value pairs are exported for the boot only (an armed diagnostic, an arm).  Exit 0: all three checks hold.
#   Exit 1: a check failed - the offending lines are printed; do NOT run a ladder on this binary.  Leaves the engine
#   DOWN and the learned profile pinned (battery-lib's cleanup_engine owns both).  Diagnostic: never a tracker point.
set -u
REPO=${REPO:-/home/svg/LLM/strata-custom-2}
PORT=${PORT:-8081}
cd "$REPO"
. "$REPO/campaign/battery-lib.sh"
[ $# -ge 1 ] || { echo "usage: smoke-candidate.sh [VAR=value ...] <tag>"; exit 2; }
TAG="${!#}"; ENVS=("${@:1:$#-1}")
OUT="$REPO/campaign/results/smoke-$TAG"; mkdir -p "$OUT"
cleanup_engine >/dev/null 2>&1
PREV=$(grep -n 'engine started:' logs/custom-iq3_s.log 2>/dev/null | tail -1 | cut -d: -f1); PREV=${PREV:-0}   # the log is appended across boots
env "${ENVS[@]}" nohup ./launch-custom-iq3s-8081.sh > "$OUT/boot.log" 2>&1 &
for i in $(seq 1 100); do curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status": "ok"' && break; sleep 5; done
# this boot's segment starts at ITS start line: the first one past the previous boot's (an anchor taken before the launch
# read the previous boot's lines once - the first version of this script reported a telemetry count from the boot before)
L0=$(tail -n +"$((PREV + 1))" logs/custom-iq3_s.log | grep -n 'engine started:' | head -1 | cut -d: -f1); L0=${L0:+$((L0 + PREV))}; L0=${L0:-$((PREV + 1))}
curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"status": "ok"' || { echo "SMOKE FAIL: no health ($TAG)"; tail -5 "$OUT/boot.log"; cleanup_engine | tail -1; exit 1; }
wait_prewarm
EP=$(pgrep -x strata | head -1)
echo "smoke $TAG: binary $(sha256sum engine/strata | cut -c1-16) | engine pid $EP | env: ${ENVS[*]:-none}"
.venv/bin/python campaign/probe-lane.py --port "$PORT" --label "smoke-$TAG" > "$OUT/leg.txt" 2>&1
grep -E '^M1|^M2|^prefill' "$OUT/leg.txt" | cut -c1-150
M1=$(grep -E '^M1 single-stream' "$OUT/leg.txt" | tail -1 | sed -E 's/.*: ([0-9.]+) t\/s.*/\1/')
CONC=$(grep -E '^M2 6-conc' "$OUT/leg.txt" | tail -1 | sed -E 's/.*: ([0-9.]+) t\/s.*/\1/')
ERRS=$(grep -c '(error' "$OUT/boot.log")
# starts AFTER this boot's own start line = restarts (the first version subtracted one more: an anchor fix made it -1)
RESTARTS=$(( $(grep -c 'engine started:' logs/custom-iq3_s.log) - $(head -n "$L0" logs/custom-iq3_s.log | grep -c 'engine started:') ))
STALLS=$(tail -n +"$L0" logs/custom-iq3_s.log | grep -c 'no progress for')
FAIL=0
ok(){ echo "  $1"; }; bad(){ echo "  *** $1"; FAIL=1; }
[ -n "$M1" ] && awk -v m="$M1" 'BEGIN{exit !(m>0)}' && ok "single-stream answers: $M1 t/s" || bad "single-stream lane did not answer"
[ -n "$CONC" ] && awk -v m="$CONC" 'BEGIN{exit !(m>0)}' && ok "concurrent lane answers: $CONC t/s" || bad "THE CONCURRENT LANE RETURNED NO TOKENS ($CONC): the batch lane is not served"
[ "$ERRS" = 0 ] && ok "no request of the leg ended in an error" || { bad "$ERRS request(s) ended in an error:"; grep '(error' "$OUT/boot.log" | head -3 | cut -c1-200 | sed 's/^/      /'; }
[ "$RESTARTS" = 0 ] && ok "the engine was not restarted" || bad "the engine restarted $RESTARTS time(s) during the leg"
[ "$STALLS" = 0 ] && ok "no stall was caught" || { bad "$STALLS stall(s) caught:"; tail -n +"$L0" logs/custom-iq3_s.log | grep -E 'no progress for|stall report' | head -2 | cut -c1-200 | sed 's/^/      /'; }
TEL=$(tail -n +"$L0" logs/custom-iq3_s.log | grep -c 'cohorts: tel'); [ "$TEL" -gt 0 ] && echo "  per-round telemetry lines: $TEL ($OUT/telemetry.txt)" && tail -n +"$L0" logs/custom-iq3_s.log | grep 'cohorts: tel' > "$OUT/telemetry.txt"
cleanup_engine | tail -1
[ "$FAIL" = 0 ] && echo "SMOKE PASS ($TAG)" || echo "SMOKE FAIL ($TAG) - do not ladder this binary"
exit "$FAIL"
