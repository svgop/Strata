#!/bin/bash
# campaign/perf-serve-anatomy.sh — where the engine's serve thread spends its HOST time in one lane (perf).
#
# The complement of nsys-serve-anatomy.sh: that instrument sees the GPU idle and the CUDA calls; this one sees the
# host code between them.  nsys's CPU sampler and scheduler trace attach only to processes started after collection
# begins, so they never cover an already-running engine (2026-11-25: zero samples, zero scheduling rows for it).
# Boots the standing server UN-instrumented, warms it by the battery protocol (health -> prewarm -> one full
# discarded probe leg), then records the engine's serve thread with perf (DWARF call graphs, 499 Hz) while the lane
# runs for the given seconds.  A DIAGNOSTIC: never a tracker point.
#
# Usage: bash campaign/perf-serve-anatomy.sh <label> [conc|m1] [seconds=14]
# Leaves campaign/results/<label>-perf/{perf.serve.txt,boot.log} and the capture (perf.serve.data) in
# $STRATA_PROFILE_HOME/<label>-<date>/, and prints the flat profile.
# Leaves the engine DOWN and the learned profile pinned: battery-lib's cleanup_engine owns both.
set -u
REPO=${REPO:-/home/svg/LLM/strata-custom-2}
PORT=${PORT:-8081}
LABEL=${1:?label}
LANE=${2:-conc}
SECONDS_TO_RUN=${3:-14}
case "$LANE" in conc|m1) ;; *) echo "perf-anatomy: lane must be conc or m1 (got '$LANE')" >&2; exit 2 ;; esac
cd "$REPO" || exit 1
. "$REPO/campaign/battery-lib.sh"
O="$REPO/campaign/results/$LABEL-perf"                       # the readable record
BIN="${STRATA_PROFILE_HOME:-/mnt/DATA/LLM/Models/GGUF/strata/Custom-data/profiles}/$LABEL-$(date +%F)"   # the capture (bulk)
mkdir -p "$O" "$BIN"
KEY=$(.venv/bin/python -c "import json;print(json.load(open('config/custom-iq3_s-8081.json'))['api_key'])")

cleanup_engine
echo "[$(date +%T)] launching the standing server (un-instrumented)"
nohup ./launch-custom-iq3s-8081.sh > "$O/boot.log" 2>&1 &
for i in $(seq 1 100); do
  curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status": "ok"' && break
  sleep 5
done
curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status": "ok"' || {
  echo "perf-anatomy: ABORT - the engine did not answer /health in 500 s (boot log: $O/boot.log)"; exit 1; }
wait_prewarm
.venv/bin/python campaign/probe-lane.py --port "$PORT" --label "$LABEL-perf-discard" 2>&1 \
  | grep -E '^M1|^M2' | sed 's/^/[warm-discard] /'

SERVE_TID=$(pgrep -x strata | head -1)      # the engine's main thread is its serve thread (tid == pid)
[ -n "$SERVE_TID" ] || { echo "perf-anatomy: no engine process"; exit 1; }
echo "[$(date +%T)] recording the serve thread (tid $SERVE_TID) for $SECONDS_TO_RUN s while $LANE runs"
perf record -F 499 --call-graph dwarf,16384 -t "$SERVE_TID" -o "$BIN/perf.serve.data" -- sleep "$SECONDS_TO_RUN" >/dev/null 2>&1 &
PERF_PID=$!
sleep 0.7
.venv/bin/python - "$LANE" "$SECONDS_TO_RUN" "$KEY" << 'PY'
import importlib.util, sys, time
lane, seconds, key = sys.argv[1], float(sys.argv[2]), sys.argv[3]
spec = importlib.util.spec_from_file_location("probe_lane", "campaign/probe-lane.py")   # the lane's own request shapes
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
base = "http://127.0.0.1:8081/v1/chat/completions"
t0 = time.time()
r = 0
while time.time() - t0 < seconds - 13.5 or r == 0:      # whole runs only; the conc run is ~13 s
    r += 1
    if lane == "conc":
        tok, wall = pl.conc_run(base, 6, 600, key)
    else:
        ok, tok, wall = pl.probe(base, pl.REAL, 2000, 0.7, key)
    print(f"[perf-anatomy {lane} run {r}] {tok / wall:.1f} t/s ({tok} tok in {wall:.2f} s)  [lane elapsed {time.time() - t0:.1f} s]")
PY
wait "$PERF_PID" || echo "perf-anatomy: perf record exited non-zero"
cleanup_engine
[ -s "$BIN/perf.serve.data" ] || { echo "perf-anatomy: no perf data at $BIN/perf.serve.data"; exit 1; }
perf report -i "$BIN/perf.serve.data" --stdio --no-children --percent-limit 1.0 --sort dso,sym > "$O/perf.serve.txt" 2>/dev/null
echo "[$(date +%T)] $(grep -m1 -E '^[0-9,]+ samples' "$O/perf.serve.txt" || echo '(sample count line not found)') | flat profile: $O/perf.serve.txt"
sed -n '1,45p' "$O/perf.serve.txt" | cut -c1-190
