#!/bin/bash
# campaign/nsys-serve-anatomy.sh — a serve-mode anatomy of ONE lane on the standing recipe, under nsys.
#
# F-452 resolved the solo window's boundary with a node-level trace of the STANDALONE engine.  This is the same
# instrument for the server path, where the lanes actually run.  It boots the standing server inside an nsys session
# with collection OFF, warms it by the battery protocol (health -> prewarm -> one full discarded probe leg, all
# un-instrumented), collects a lane-only run, and stops.  The trace's timeline is therefore the lane's own: no phase
# has to be cut out of a mixed timeline afterwards.
#
# A DIAGNOSTIC.  The CUDA injection and the CPU sampler cost a few percent: the t/s printed here are never a tracker
# point and never a verdict.
#
# Usage: bash campaign/nsys-serve-anatomy.sh <label> [conc|m1] [runs=2] [perf]
#   perf   also record the engine's serve thread with perf (DWARF call graphs, 999 Hz, CLOCK_REALTIME sample times)
#          during the collection: campaign/nsys-perf-stalls.py then reads the host call chains inside each GPU stall.
#          Sample times are perf's default CLOCK_MONOTONIC: --clockid realtime is refused on this box ("wrong
#          clockid (0)" - cpu/cycles/P is AMD precise sampling, monotonic only); nsys records the monotonic reading
#          of its own zero beside the UTC one, and the reader proves the alignment before attributing anything.
#   conc   the probe's own 6-concurrent run (six 600-token requests) - the host-bound lane
#   m1     the probe's own single-stream request (REAL, 2000 tokens)
# Leaves the readable record in campaign/results/<label>-anatomy/{boot.log,anatomy-summary.txt,stalls.txt,perf.log} and
# the captures (anatomy.nsys-rep, anatomy.sqlite, perf.serve.data) in $STRATA_PROFILE_HOME/<label>-<date>/, and prints the
# sqlite summary (campaign/nsys-serve-summary.py) when that script exists.  Leaves the engine DOWN and the learned
# profile pinned: battery-lib's cleanup_engine owns both.
set -u
REPO=${REPO:-/home/svg/LLM/strata-custom-2}
PORT=${PORT:-8081}
LABEL=${1:?label}
LANE=${2:-conc}
RUNS=${3:-2}
WITH_PERF=${4:-}
case "$LANE" in conc|m1) ;; *) echo "anatomy: lane must be conc or m1 (got '$LANE')" >&2; exit 2 ;; esac
case "$WITH_PERF" in ""|perf) ;; *) echo "anatomy: the 4th argument must be perf or empty (got '$WITH_PERF')" >&2; exit 2 ;; esac
cd "$REPO" || exit 1
. "$REPO/campaign/battery-lib.sh"
NSYS=${NSYS:-/usr/local/sbin/nsys}
SESSION="anat-$LABEL-$$"
O="$REPO/campaign/results/$LABEL-anatomy"                 # the readable record: boot log, summaries
# Profiler captures are bulk (tens to hundreds of MB): they live in the data estate, never in the repository; the
# summaries read from them stay with the record above.  STRATA_PROFILE_HOME relocates the home.
BIN="${STRATA_PROFILE_HOME:-/mnt/DATA/LLM/Models/GGUF/strata/Custom-data/profiles}/$LABEL-$(date +%F)"
mkdir -p "$O" "$BIN"
KEY=$(.venv/bin/python -c "import json;print(json.load(open('config/custom-iq3_s-8081.json'))['api_key'])")

cleanup_engine
echo "[$(date +%T)] launching the standing server inside nsys session $SESSION (collection off)"
# The session model is not `nsys profile`'s: launch takes the session and trace options (and waits on the primary
# process); output, export and the samplers belong to start.
# STRATA_SWAP_GUARD=0: `systemd-run --user --scope` cannot open its bus under nsys's injected library ("Failed to
# connect to user scope bus ... Operation not permitted", 2026-11-25), so the launcher's scope is forgone.  The scope
# only caps swap, which a short diagnostic never reaches; placement is unchanged - this box has no kernel pin, so
# the standing boot is the scope plus the engine's own self-pinning, and this boot is the self-pinning alone.
STRATA_SWAP_GUARD=0 nohup "$NSYS" launch --session="$SESSION" --wait=primary \
    --trace=cuda --cuda-graph-trace=graph --trace-fork-before-exec=true \
    -- ./launch-custom-iq3s-8081.sh > "$O/boot.log" 2>&1 &
NPID=$!
for i in $(seq 1 100); do
  curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status": "ok"' && break
  kill -0 "$NPID" 2>/dev/null || {   # a launch that died must not cost the loop's full patience
    echo "anatomy: ABORT - nsys launch exited early (boot log: $O/boot.log)"; tail -3 "$O/boot.log"; exit 1; }
  sleep 5
done
curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status": "ok"' || {
  echo "anatomy: ABORT - the engine did not answer /health in 500 s (boot log: $O/boot.log)"; exit 1; }
wait_prewarm
.venv/bin/python campaign/probe-lane.py --port "$PORT" --label "$LABEL-anatomy-discard" 2>&1 \
  | grep -E '^M1|^M2' | sed 's/^/[warm-discard] /'

echo "[$(date +%T)] collecting: $LANE x$RUNS${WITH_PERF:+ + perf on the serve thread}"
# nsys's own CPU sampler and scheduler trace attach only to processes started after collection begins (never an
# already-running engine), so host attribution is perf's job: wall-clock sample times, so a reader can drop each
# sample into the GPU timeline (TARGET_INFO_SESSION_START_TIME is the wall time of the capture's zero).
PERF_PID=""
if [ "$WITH_PERF" = perf ]; then
  SERVE_TID=$(pgrep -x strata | head -1)      # the engine's main thread is its serve thread (tid == pid)
  perf record -F 999 --call-graph dwarf,16384 -t "$SERVE_TID" -o "$BIN/perf.serve.data" \
      -- sleep $((RUNS * 15 + 4)) > "$O/perf.log" 2>&1 &
  PERF_PID=$!
  sleep 0.7
  kill -0 "$PERF_PID" 2>/dev/null || { echo "anatomy: ABORT - perf record exited at start:"; cat "$O/perf.log"; exit 1; }
fi
"$NSYS" start --session="$SESSION" --output="$BIN/anatomy" --force-overwrite=true --export=sqlite \
    --sample=none --backtrace=none
.venv/bin/python - "$LANE" "$RUNS" "$KEY" << 'PY'
import importlib.util, sys, time
lane, runs, key = sys.argv[1], int(sys.argv[2]), sys.argv[3]
spec = importlib.util.spec_from_file_location("probe_lane", "campaign/probe-lane.py")   # the lane's own request shapes
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
base = "http://127.0.0.1:8081/v1/chat/completions"
t0 = time.time()
for r in range(1, runs + 1):
    if lane == "conc":
        tok, wall = pl.conc_run(base, 6, 600, key)
    else:
        ok, tok, wall = pl.probe(base, pl.REAL, 2000, 0.7, key)
    print(f"[anatomy {lane} run {r}] {tok / wall:.1f} t/s ({tok} tok in {wall:.2f} s)  [lane elapsed {time.time() - t0:.1f} s]")
PY
"$NSYS" stop --session="$SESSION"
if [ -n "$PERF_PID" ]; then
  sleep 3; kill -INT "$PERF_PID" 2>/dev/null; wait "$PERF_PID" 2>/dev/null
  [ -s "$BIN/perf.serve.data" ] || echo "anatomy: perf wrote no data ($O/perf.log): $(tail -2 "$O/perf.log" | tr '\n' ' ')"
fi
"$NSYS" shutdown --session="$SESSION" --kill=none >/dev/null 2>&1 || true
cleanup_engine
for i in $(seq 1 90); do [ -s "$BIN/anatomy.nsys-rep" ] && [ -s "$BIN/anatomy.sqlite" ] && break; sleep 2; done
[ -s "$BIN/anatomy.nsys-rep" ] || { echo "anatomy: no report at $BIN/anatomy.nsys-rep"; exit 1; }
if [ ! -s "$BIN/anatomy.sqlite" ]; then     # start's --export normally writes it; the explicit export is the fallback
  "$NSYS" export --type sqlite --force-overwrite true --output "$BIN/anatomy.sqlite" "$BIN/anatomy.nsys-rep" >/dev/null 2>&1
fi
[ -s "$BIN/anatomy.sqlite" ] || { echo "anatomy: no sqlite export at $BIN/anatomy.sqlite"; exit 1; }
echo "[$(date +%T)] report $BIN/anatomy.nsys-rep ($(du -h "$BIN/anatomy.nsys-rep" | cut -f1)); sqlite $(du -h "$BIN/anatomy.sqlite" | cut -f1)"
if [ -f "$REPO/campaign/nsys-serve-summary.py" ]; then
  .venv/bin/python campaign/nsys-serve-summary.py "$BIN/anatomy.sqlite" | tee "$O/anatomy-summary.txt"
else
  echo "anatomy: campaign/nsys-serve-summary.py is absent - read the sqlite by hand"
fi
if [ -s "$BIN/perf.serve.data" ] && [ -f "$REPO/campaign/nsys-perf-stalls.py" ]; then
  echo
  .venv/bin/python campaign/nsys-perf-stalls.py "$BIN/anatomy.sqlite" "$BIN/perf.serve.data" | tee "$O/stalls.txt"
fi
