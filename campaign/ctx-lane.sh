#!/bin/bash
# campaign/ctx-lane.sh - the long-context lane read: ONE boot of whatever is in engine/strata (this script
# deploys nothing), the warm protocol, the ctx ladder, the engine down again.  A read, never a tracker point
# (campaign/tracker/README.md: depth is a different axis than the short-lane trend).
#
# Why it exists (F-466): every floor statement on this campaign was measured at a ~1.1k-token prompt, while
# the production log's median request carries a 96k-token context.  The ladder reads decode t/s against depth
# and the cold-read TTFT; --turn adds the TTFT of an agent turn (the reply sent back, then a ~300-token user
# message on a history the conversation cache holds) and --branch that of a history edit (see ctx-probe.py).
# The engine's per-request accounting lines of the boot are kept beside the ladder and echoed.
#
# Usage: bash campaign/ctx-lane.sh <label> [--turn | --branch] <tokens...>
#   Results: campaign/results/ctx-<label>/{boot-banner.txt,discard.txt,ladder.txt,serve-lines.txt}.
#   Exit 0 only with measured lines on disk, no request in error, no restart, no stall.  Leaves the engine DOWN
#   and the learned profile pinned (battery-lib's cleanup_engine owns both).  The probe's own offline self-test
#   runs BEFORE the boot: its first run here booted the standing for a probe that crashed on its first request
#   and still said DONE; its second counted the boot's own start line as a restart.
set -u
REPO=${REPO:-/home/svg/LLM/strata-custom-2}
PORT=${PORT:-8081}
cd "$REPO"
. "$REPO/campaign/battery-lib.sh"
[ $# -ge 2 ] || { echo "usage: ctx-lane.sh <label> [--delta] <tokens...>"; exit 2; }
LABEL=$1; shift
OUT="$REPO/campaign/results/ctx-$LABEL"; mkdir -p "$OUT"
.venv/bin/python campaign/ctx-probe.py --selftest || { echo "CTX FAIL: the probe failed its own self-test - nothing booted"; exit 1; }
cleanup_engine >/dev/null 2>&1
PREV=$(grep -n 'engine started:' logs/custom-iq3_s.log 2>/dev/null | tail -1 | cut -d: -f1); PREV=${PREV:-0}   # the log is appended across boots
nohup ./launch-custom-iq3s-8081.sh > "$OUT/boot.log" 2>&1 &
wait_health 100 || { echo "CTX FAIL: no health ($LABEL)"; tail -5 "$OUT/boot.log"; cleanup_engine | tail -1; exit 1; }
wait_prewarm
L0=$(tail -n +"$((PREV + 1))" logs/custom-iq3_s.log | grep -n 'engine started:' | head -1 | cut -d: -f1); L0=${L0:+$((L0 + PREV))}; L0=${L0:-$((PREV + 1))}
echo "ctx $LABEL: binary $(sha256sum engine/strata | cut -c1-16) | load $(cut -d' ' -f1-3 /proc/loadavg) | args: $*"
# campaign law 11: one discarded ladder request before the measured ones (a cold mapping's first-touch faults)
.venv/bin/python campaign/ctx-probe.py 4096 > "$OUT/discard.txt" 2>&1 || { echo "CTX FAIL: the discarded request failed ($LABEL)"; cat "$OUT/discard.txt"; cleanup_engine | tail -1; exit 1; }
cat "$OUT/discard.txt"
.venv/bin/python campaign/ctx-probe.py "$@" > "$OUT/ladder.txt" 2>&1; RC=$?
cat "$OUT/ladder.txt"
N=$(grep -c '^ctx .*| TTFT' "$OUT/ladder.txt")
tail -n +"$L0" logs/custom-iq3_s.log > "$OUT/boot-segment.log"
grep -E '^strata serve: prompt [0-9]+ tokens' "$OUT/boot-segment.log" > "$OUT/serve-lines.txt"
awk '/engine started:/{f=1} f' "$OUT/boot-segment.log" | grep -vE '^strata (serve|batch|KV prefetch|decode|elastic)|^\[strata\]' | head -400 > "$OUT/boot-banner.txt"
ERRS=$(grep -cE '\(error' "$OUT/boot.log")
# restarts are the start lines strictly AFTER this boot's own (L0 is that line: counting it is the off-by-one
# that failed a clean ladder on this instrument's first run)
RESTARTS=$(( $(tail -n +"$((L0 + 1))" logs/custom-iq3_s.log | grep -c 'engine started:') ))
STALLS=$(tail -n +"$L0" logs/custom-iq3_s.log | grep -c 'no progress for')
echo "engine accounting of the boot (reused + read tokens, read ms, drafts accepted):"; cut -c1-175 "$OUT/serve-lines.txt"
rm -f "$OUT/boot-segment.log" "$OUT/boot.log"
cleanup_engine | tail -1
FAIL=0
[ "$RC" = 0 ] && [ "$N" -gt 0 ] || { echo "  *** the probe exited $RC with $N measured lines"; FAIL=1; }
[ "$ERRS" = 0 ] || { echo "  *** $ERRS request(s) ended in error"; FAIL=1; }
[ "$RESTARTS" = 0 ] || { echo "  *** the engine restarted $RESTARTS time(s) during the read"; FAIL=1; }
[ "$STALLS" = 0 ] || { echo "  *** $STALLS stall(s) caught"; FAIL=1; }
if [ "$FAIL" = 0 ]; then echo "CTX DONE ($LABEL): $N measured lines in $OUT/ladder.txt, no request in error, no restart"
else echo "CTX FAIL ($LABEL) - do not read this ladder"; exit 1; fi
