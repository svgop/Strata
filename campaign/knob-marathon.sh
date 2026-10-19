#!/bin/bash
# campaign/knob-marathon.sh — seven single-variable legs vs the F-238 re-bank control.
# Control (same night): M1 193.7 / pair 199.3 / conc 303.4, M3b 76.2.
# Legs: pft (prefill-timing diag + 32k depth), lanecell, tsum, q6packed, q8packed,
#       lp (learned profile via config copy), q6psel (Q6P self-test diag).
# Every leg: fresh boot, full probe (--short-n 6), engine-log tail captured, port-guarded.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
ARMS=/home/svg/LLM/Strata/custom-arms.env
PORT=8081
OUT=$REPO/campaign/results/knob-marathon.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "KNOB MARATHON start $(date)" > "$OUT"

port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }
cleanup_engine() {
  local p pid c
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *strata*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
}

# learned-profile config copy (input = learned profile; save path unchanged)
python3 - "$REPO" << 'PYEOF'
import json, sys
repo = sys.argv[1]
cfg = json.load(open(repo + '/custom-iq3_s-8081.json'))
for i, a in enumerate(cfg['args']):
    if a == '--expert-profile':
        cfg['args'][i+1] = 'data/expert-profile-learned.bin'
json.dump(cfg, open(repo + '/campaign/custom-iq3_s-lp.json', 'w'), indent=1)
print('lp config written')
PYEOF

run_leg() {  # $1 label  $2 launch kind ("launcher"|"lp")  $3.. extra env assignments
  local label=$1 kind=$2; shift 2
  cleanup_engine; sleep 3
  if [ -n "$(port_pid)" ]; then
    echo "=== $label ABORT: port occupied ===" >> "$OUT"; return
  fi
  if [ "$kind" = "lp" ]; then
    ( cd "$REPO" && nohup env "$@" bash -c ". $ARMS; exec .venv/bin/python serve/server.py --engine strata --config $REPO/campaign/custom-iq3_s-lp.json --port $PORT --host 0.0.0.0" >/dev/null 2>&1 & )
  else
    ( cd "$REPO" && nohup env "$@" "$LAUNCH" >/dev/null 2>&1 & )
  fi
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $label health=$ok $(date +%H:%M:%S) env=[$*] ===" >> "$OUT"
  if [ "$ok" != 1 ]; then
    echo "LEG $label FAILED TO BOOT - skipping" >> "$OUT"
    tail -80 "$REPO/custom-iq3_s.log" >> "$OUT" 2>/dev/null
    return
  fi
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "knob-$label" --short-n 6 >> "$OUT" 2>&1
  if [ "$label" = "pft" ]; then
    timeout 300 .venv/bin/python campaign/depth-lane-real.py --depth 32000 >> "$OUT" 2>&1 || true
  fi
  tail -400 "$REPO/custom-iq3_s.log" > "$RES/knob-$label-engtail.txt" 2>/dev/null
  grep -iE "prefill timing|PREFILL_TIMING|lanecell| STRATA_TSUM|self-test|q6p" "$REPO/custom-iq3_s.log" | tail -40 >> "$OUT" 2>/dev/null
  cleanup_engine
}

run_leg pft       launcher STRATA_PREFILL_TIMING=1
run_leg lanecell  launcher STRATA_ATTN_LANECELL=1
run_leg tsum      launcher STRATA_TSUM=1
run_leg q6packed  launcher STRATA_Q6_PACKED=1
run_leg q8packed  launcher STRATA_Q8_PACKED=1
run_leg lp        lp
run_leg q6psel    launcher STRATA_Q6P_SELFTEST=1

echo "KNOB MARATHON end $(date)" >> "$OUT"
echo "done" >> "$OUT"
