#!/bin/bash
# campaign/batteries/t36-lifestic.sh — T34 F3+F4: the elastic-sessions decision battery.
# F3: standing (ctl) vs STRATA_ELASTIC_SESSIONS=1 STRATA_ELASTIC_PARK=1 STRATA_ELASTIC_WATERMARK=1 STRATA_ELASTIC_TIMED_SCAN=1 STRATA_VMM_CHUNK_MIB=32 STRATA_SERVE_TELEMETRY=1 (ela), 3 interleaved rounds, prewarmed,
#     full lane probe each — the no-regression law decides adoption-shape.
# F4: one parallel-9 elastic leg (config copy, root; deleted after) + 9-conc probe =
#     the F-271 question re-asked on the elastic substrate.
# Single-engine law on every boot (verified per leg: exactly one new engine-started line).
set -u
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t36-lifestic.log
echo "T36 BATTERY start $(date)" > "$OUT"
port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }
cleanup_engine() {
  local p pid c
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *strata*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
  sleep 2
}
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy" >> "$OUT"; exit 1; }

run_leg() {  # run_leg <tag> <round> <env-pass> [launcher] [conc-n] [config]
  local tag=$1 r=$2 envp=$3 launcher=${4:-$REPO/launch-custom-iq3s-8081.sh} conc=${5:-6} cfg=${6:-}
  local lbl="t34-$tag-r$2"
  [ -n "$cfg" ] && launcher=$REPO/launch-custom-iq3s-8081-b9.sh
  cleanup_engine
  local e0; e0=$(grep -ac "engine started" "$REPO/custom-iq3_s.log" 2>/dev/null || echo 0)
  ( cd "$REPO" && env $envp nohup bash "$launcher" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  sleep 20   # settle; the single-engine check follows
  local e1; e1=$(grep -ac "engine started" "$REPO/custom-iq3_s.log" 2>/dev/null || echo 0)
  echo "=== LEG $lbl health=$ok engines_delta=$((e1-e0)) $(date +%H:%M:%S) env='$envp' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 --conc-n "$conc" >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tee -a "$OUT" )
  grep -a "ELASTIC_SESSIONS on\|materialized\|demoted\|below 512" "$REPO/custom-iq3_s.log" | tail -2 >> "$OUT" 2>/dev/null
  cleanup_engine
}

# F3: 3 interleaved rounds
for r in 1 2 3; do
  run_leg ctl "$r" ""
  run_leg ela "$r" "STRATA_ELASTIC_SESSIONS=1 STRATA_ELASTIC_PARK=1 STRATA_ELASTIC_WATERMARK=1 STRATA_ELASTIC_TIMED_SCAN=1 STRATA_VMM_CHUNK_MIB=32 STRATA_SERVE_TELEMETRY=1"
done

# F4: parallel-9 elastic (config copy created here, deleted at the end)
python3 - <<'PY'
import json
cfg = json.load(open('/home/svg/LLM/strata-custom-2/custom-iq3_s-8081.json'))
cfg['parallel'] = 9
json.dump(cfg, open('/home/svg/LLM/strata-custom-2/custom-iq3s-b9.json', 'w'), indent=2)
PY
sed 's|custom-iq3_s-8081.json|custom-iq3s-b9.json|' "$REPO/launch-custom-iq3s-8081.sh" > "$REPO/launch-custom-iq3s-8081-b9.sh"
chmod +x "$REPO/launch-custom-iq3s-8081-b9.sh"
run_leg b9ela 1 "STRATA_ELASTIC_SESSIONS=1 STRATA_ELASTIC_PARK=1 STRATA_ELASTIC_WATERMARK=1 STRATA_ELASTIC_TIMED_SCAN=1 STRATA_VMM_CHUNK_MIB=32 STRATA_SERVE_TELEMETRY=1" "" 9 b9
rm -f "$REPO/custom-iq3s-b9.json" "$REPO/launch-custom-iq3s-8081-b9.sh"

echo "T36 BATTERY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
