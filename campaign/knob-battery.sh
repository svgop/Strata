#!/bin/bash
# campaign/knob-battery.sh — 3v3 ABAB interleaved batteries for the knob-marathon candidates.
# Per candidate: arm,ctl,arm,ctl,arm,ctl (fresh boot each). Drift is differenced out by
# alternation; the arm-vs-ctl mean is the claim. One extra diagnostic leg captures the
# Q6P self-test stderr that the first marathon lost to /dev/null.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
ARMS=/home/svg/LLM/strata/custom-arms.env
PORT=8081
OUT=$REPO/campaign/results/knob-battery.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "KNOB BATTERY start $(date)" > "$OUT"

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

boot_probe() {  # $1 label  $2 kind (launcher|lp)  $3.. env
  local label=$1 kind=$2; shift 2
  local logf=$RES/bat-$label.out
  cleanup_engine; sleep 2
  if [ -n "$(port_pid)" ]; then echo "=== $label ABORT port busy ===" >> "$OUT"; return; fi
  if [ "$kind" = "lp" ]; then
    ( cd "$REPO" && nohup env "$@" bash -c ". $ARMS; exec .venv/bin/python serve/server.py --engine strata --config $REPO/campaign/custom-iq3_s-lp.json --port $PORT --host 0.0.0.0" >"$logf" 2>&1 & )
  else
    ( cd "$REPO" && nohup env "$@" "$LAUNCH" >"$logf" 2>&1 & )
  fi
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $label health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  if [ "$ok" != 1 ]; then echo "LEG $label FAILED" >> "$OUT"; tail -40 "$logf" >> "$OUT"; return; fi
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "bat-$label" --short-n 6 >> "$OUT" 2>&1
  cleanup_engine
}

battery() {  # $1 candidate label  $2 kind  $3.. env for the arm leg
  local cand=$1 kind=$2; shift 2
  local r
  for r in 1 2 3; do
    boot_probe "$cand-arm-r$r" "$kind" "$@"
    boot_probe "$cand-ctl-r$r" launcher
  done
}

battery lanecell launcher STRATA_ATTN_LANECELL=1
battery tsum     launcher STRATA_TSUM=1
battery q6packed launcher STRATA_Q6_PACKED=1
battery q8packed launcher STRATA_Q8_PACKED=1
battery lp       lp

# Q6P self-test diagnostic with stderr captured
boot_probe q6psel-diag launcher STRATA_Q6P_SELFTEST=1
grep -iE "q6|selftest|self-test|bitwise" "$RES/bat-q6psel-diag.out" | head -30 >> "$OUT" 2>/dev/null

echo "KNOB BATTERY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
