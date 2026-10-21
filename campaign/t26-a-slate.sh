#!/bin/bash
# campaign/t26-a-slate.sh — T26 A-slate, corrected by F-254.
# The device-plan boot is VOID (STRATA_VERIFY_DEVICE_PLAN=1 is already exported by the standing launcher
# and T1.1 already priced the host-plan RTT it removes at ~0.26 ms, closed neutral). This slate runs the
# levers that are genuinely unarmed:
#   1. STRATA_BATCH_DECODE_SHARE raise (env-only; conc lane)   — ABAB 3v3, one variable
#   2. tools/calibrate.py (measures our pcie-frac; the config's 0.35 is hand-set, never measured)
#   3. campaign/prewarm2.sh (post-boot warm package; same-session capture, not an arm)
# Every leg: fresh boot, full probe (--short-n 6), engine-log tail captured, port-guarded.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
OUT=$REPO/campaign/results/t26-a-slate.log
RES=$REPO/campaign/results
SHARE=${SHARE:-0.75}          # arm value; standing default is 0.5 (generate.cpp:13698-13704)
mkdir -p "$RES"
echo "T26 A-SLATE start $(date)" > "$OUT"

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

preflight() {
  echo "--- preflight $(date) ---" >> "$OUT"
  git -C "$REPO" log -1 --oneline >> "$OUT" 2>&1
  git -C "$REPO" status --short >> "$OUT" 2>&1
  nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm --format=csv,noheader >> "$OUT" 2>&1
  [ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy before start" >> "$OUT"; exit 1; }
  return 0
}

boot() {  # $1 extra env (may be empty) $2 label
  cleanup_engine
  [ -n "$(port_pid)" ] && { echo "=== $2 ABORT port busy ===" >> "$OUT"; return 1; }
  ( cd "$REPO" && nohup env $1 "$LAUNCH" >"$RES/t26-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) env=[$1] ===" >> "$OUT"
  [ "$ok" = 1 ]
}

leg() {  # $1 extra env $2 label
  boot "$1" "$2" || return
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t26-$2" --short-n 6 >> "$OUT" 2>&1 )
  # T27-B stability capture, same leg: TTFT/ITL/E2E p50/p99 + acceptance + clock/temp/power + golden-hash drift
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t26-$2" 2>&1 | tee -a "$OUT" )
  grep -aE 'first token|window up to|device plan' "$REPO/custom-iq3_s.log" | tail -2 >> "$OUT" 2>/dev/null
  cleanup_engine
}

preflight

# ---- 1. STRATA_BATCH_DECODE_SHARE: ctl (0.5) vs arm ($SHARE), ABAB 3v3 ----
for r in 1 2 3; do
  leg "" "share-ctl-r$r"
  leg "STRATA_BATCH_DECODE_SHARE=$SHARE" "share${SHARE//./}-r$r"
done

# ---- 2. calibrate.py: one run, prints the measured settings (does NOT rewrite the config) ----
if boot "" "calibrate"; then
  ( cd "$REPO" && timeout 1800 .venv/bin/python tools/calibrate.py "$REPO/custom-iq3_s-8081.json" \
      > "$RES/t26-calibrate.json" 2>> "$OUT" ) || echo "calibrate failed" >> "$OUT"
  cleanup_engine
fi

# ---- 3. prewarm2: standing boot, warm package, then the probe (same-session capture) ----
if boot "" "prewarm2"; then
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t26-prewarm2" --short-n 6 >> "$OUT" 2>&1 )
  cleanup_engine
fi

echo "T26 A-SLATE end $(date)" >> "$OUT"
echo "done" >> "$OUT"
