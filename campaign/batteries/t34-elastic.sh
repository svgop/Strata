#!/bin/bash
# campaign/batteries/t34-elastic.sh — T34 F3+F4: the elastic-sessions decision battery.
# F3: standing (ctl) vs STRATA_ELASTIC_SESSIONS=1 (ela), 3 interleaved rounds, prewarmed,
#     full lane probe each — the no-regression law decides adoption-shape.
# F4: one parallel-9 elastic leg (config copy, root; deleted after) + 9-conc probe =
#     the F-271 question re-asked on the elastic substrate.
# Single-engine law on every boot (verified per leg: exactly one new engine-started line).
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t34-elastic.log
echo "T34 ELASTIC start $(date)" > "$OUT"
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy" >> "$OUT"; exit 1; }

run_leg() {  # run_leg <tag> <round> <env-pass> [launcher] [conc-n] [config]
  local tag=$1 r=$2 envp=$3 launcher=${4:-$REPO/launch-custom-iq3s-8081.sh} conc=${5:-6} cfg=${6:-}
  local lbl="t34-$tag-r$2"
  [ -n "$cfg" ] && launcher=$REPO/launch-custom-iq3s-8081-b9.sh
  cleanup_engine
  local e0; e0=$(grep -ac "engine started" "$REPO/logs/custom-iq3_s.log" 2>/dev/null || echo 0)
  ( cd "$REPO" && env $envp nohup bash "$launcher" >"$RES/$lbl.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  sleep 20   # settle; the single-engine check follows
  local e1; e1=$(grep -ac "engine started" "$REPO/logs/custom-iq3_s.log" 2>/dev/null || echo 0)
  echo "=== LEG $lbl health=$ok engines_delta=$((e1-e0)) $(date +%H:%M:%S) env='$envp' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  wait_prewarm
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "$lbl" --short-n 6 --conc-n "$conc" >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "$lbl" 2>&1 | tee -a "$OUT" )
  grep -a "ELASTIC_SESSIONS on\|materialized\|demoted\|below 512" "$REPO/logs/custom-iq3_s.log" | tail -2 >> "$OUT" 2>/dev/null
  cleanup_engine
}

# F3: 3 interleaved rounds
for r in 1 2 3; do
  run_leg ctl "$r" ""
  run_leg ela "$r" "STRATA_ELASTIC_SESSIONS=1"
done

# F4: parallel-9 elastic (config copy created here, deleted at the end)
python3 - <<'PY'
import json
cfg = json.load(open('/home/svg/LLM/strata-custom-2/config/custom-iq3_s-8081.json'))
cfg['parallel'] = 9
json.dump(cfg, open('/home/svg/LLM/strata-custom-2/config/custom-iq3s-b9.json', 'w'), indent=2)
PY
sed 's|custom-iq3_s-8081.json|custom-iq3s-b9.json|' "$REPO/launch-custom-iq3s-8081.sh" > "$REPO/launch-custom-iq3s-8081-b9.sh"
chmod +x "$REPO/launch-custom-iq3s-8081-b9.sh"
run_leg b9ela 1 "STRATA_ELASTIC_SESSIONS=1" "" 9 b9
rm -f "$REPO/custom-iq3s-b9.json" "$REPO/launch-custom-iq3s-8081-b9.sh"

echo "T34 ELASTIC end $(date)" >> "$OUT"
echo "done" >> "$OUT"
