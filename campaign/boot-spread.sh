#!/bin/bash
# campaign/boot-spread.sh — THE BOOT-SPREAD BATTERY (F-400's proof protocol).
#
# The boot-state lottery claim: the SAME recipe, the SAME inputs, M1 spreads 143/200/215 across boots.
# F-384 made the geometry deterministic (the fingerprint); F-400 made the tier's starting state
# deterministic (the calibration corpus at temperature 0 as the prewarm replay's first stage).  This
# battery is the proof: N sequential boots under ONE load band, three M1 samples each, and the
# spread read across the boots' maxima and their within-boot legs.  NO retry, NO certification gate -
# a boot that reads low is DATA (the old boot-certify re-booted until a boot passed, which is the
# lottery wearing a certificate; this records what the recipe actually does).
#
# Load discipline: the load average is recorded per leg; the summary DISCLOSES the band.  A spread
# verdict is only claimed within one stable band (the drift law).  This is not a bank instrument -
# banked numbers come from the quiet-window interleave.
#
# Usage: campaign/boot-spread.sh [--boots 5] [--samples 3] [--label f400]
set -u
. "$(dirname "$0")/battery-lib.sh"
REPO=${REPO:-/home/svg/LLM/strata-custom-2}
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
BOOTS=5; SAMPLES=3; LABEL=f400; MAXES=""
while [ $# -gt 0 ]; do case "$1" in --boots) BOOTS=$2; shift 2;; --samples) SAMPLES=$2; shift 2;;
                       --label) LABEL=$2; shift 2;; *) echo "boot-spread: unknown arg $1" >&2; exit 2;; esac; done
OUT=$REPO/campaign/results/spread-${LABEL}-$(date +%H%M%S).txt
KEY=3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8
PROMPT='Write a comprehensive technical analysis of modern GPU memory hierarchies and their impact on LLM inference performance. Cover HBM bandwidth, L2 cache behavior, PCIe transfer overhead, and pinned memory tradeoffs, with concrete numbers and examples throughout.'

echo "BOOT-SPREAD $(date) boots=$BOOTS samples=$SAMPLES label=$LABEL" | tee "$OUT"
m1() {  # one M1 leg: prints "<t/s> <load1>"
  python3 - "$KEY" "$PROMPT" <<'PY'
import json,sys,time,urllib.request,os
key,prompt=sys.argv[1],sys.argv[2]
body=json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":2000,"temperature":0.7}).encode()
req=urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",data=body,
    headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
t0=time.time()
with urllib.request.urlopen(req,timeout=600) as r: d=json.load(r)
dt=time.time()-t0
load=open("/proc/loadavg").read().split()[0]
print(f"{d['usage']['completion_tokens']/dt:.1f} {load}")
PY
}
for b in $(seq 1 "$BOOTS"); do
  cleanup_engine
  sleep 4
  rm -f "$REPO/logs/custom-iq3_s.log"
  ( cd "$REPO" && nohup bash "$LAUNCH" > "/tmp/spread-boot-$b.out" 2>&1 < /dev/null & )
  if ! wait_health; then echo "boot $b: NO /health — ABORT" | tee -a "$OUT"; exit 1; fi
  wait_prewarm
  FP=$(grep -o "boot fingerprint [0-9a-f]* slots=[0-9]*" "$REPO/logs/custom-iq3_s.log" | tail -1)
  BACKING=$(grep -o "backing: [a-z-]*" "$REPO/logs/custom-iq3_s.log" | tail -1)
  CREADY=$(grep "at READY" "/tmp/spread-boot-$b.out" | tail -1)
  CSET=$(grep "settled (post-warm-in)" "/tmp/spread-boot-$b.out" | tail -1)
  echo "boot $b: $FP $BACKING" | tee -a "$OUT"
  echo "  $CREADY" | tee -a "$OUT"
  echo "  $CSET" | tee -a "$OUT"
  BEST=0
  for s in $(seq 1 "$SAMPLES"); do
    read -r TPS LOAD <<< "$(m1)"
    echo "  boot $b leg $s: $TPS t/s (load $LOAD)" | tee -a "$OUT"
    awk -v a="$TPS" -v b="$BEST" 'BEGIN{exit !(a>b)}' && BEST=$TPS
  done
  echo "  boot $b MAX: $BEST" | tee -a "$OUT"
  MAXES="$MAXES $BEST"
done
cleanup_engine
cleanup_slots   # F-406: the last boot's exit drain (tmpfs RAM) leaves with the battery
echo | tee -a "$OUT"
echo "SUMMARY: maxima =$MAXES" | tee -a "$OUT"
python3 - $MAXES <<'PY' | tee -a "$OUT"
import sys,statistics
v=[float(x) for x in sys.argv[1:]]
if len(v)>=2:
    print(f"  boots={len(v)} max-mean={statistics.mean(v):.1f} spread={(max(v)-min(v))/statistics.mean(v)*100:.1f}%"
          f" min={min(v):.1f} max={max(v):.1f}")
    if len(v)>=3: print(f"  stdev={statistics.stdev(v):.1f}")
PY
echo "DISCLOSED: load band per leg above; a spread verdict is claimed only within one stable band." | tee -a "$OUT"
rm -f /tmp/spread-boot-*.out   # this battery's boot-console scratch (the evidence lives in $OUT)
