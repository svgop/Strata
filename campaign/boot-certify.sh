#!/bin/bash
# campaign/boot-certify.sh — THE BOOT CERTIFICATION (F-387, the Boot Contract R4).
# A boot is not "ready" because /health answers: it is ready when it has served ONE fixed
# greedy decode (2000 tokens, temperature 0, the campaign's M1 prompt) at or above the bar.
# A boot that does not certify is refused (re-booted, up to --tries); the certified boot stays up.
# Usage: campaign/boot-certify.sh [--bar 200] [--tries 3] [--store <validated pack dir>]
set -u
. "$(dirname "$0")/battery-lib.sh"
REPO=${REPO:-/home/svg/LLM/strata-custom-2}
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
BAR=200; TRIES=3; STORE=""
while [ $# -gt 0 ]; do case "$1" in --bar) BAR=$2; shift 2;; --tries) TRIES=$2; shift 2;; --store) STORE=$2; shift 2;; *) echo "boot-certify: unknown arg $1" >&2; exit 2;; esac; done
OUT=$REPO/campaign/results/boot-certify-$(date +%H%M%S).txt
TMP=/tmp/strata-boot-certify-$$.txt
KEY=3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8
PROMPT='Write a comprehensive technical analysis of modern GPU memory hierarchies and their impact on LLM inference performance. Cover HBM bandwidth, L2 cache behavior, PCIe transfer overhead, and pinned memory tradeoffs, with concrete numbers and examples throughout.'
echo "BOOT CERTIFICATION $(date) bar=$BAR tries=$TRIES store=${STORE:-<launcher default>}" | tee "$OUT"
# NOTE: the boot must NOT run inside a command substitution - the launcher's process tree would
# inherit the capture pipe and $(...) would block until the engine exits (F-387).
certify_once() {   # $1=try  $2=result file (the t/s)   detail -> $OUT
  local try=$1 res=$2
  cleanup_engine
  ( cd "$REPO" && env ${STORE:+STRATA_SHADOW_EXPERTS=$STORE} nohup bash "$LAUNCH" \
      >"${OUT%.txt}.boot$try.out" 2>&1 < /dev/null & )
  wait_health || { echo "try $try: no /health" >> "$OUT"; return 1; }
  wait_prewarm
  # THE CAMPAIGN'S M1 LANE: 2000 tokens at temperature 0.7, THREE samples, the MAX (the boot's
  # capability, not one draw - the sampled text moves a single leg by +-10%).
  local body t0 t1 resp tok dt tps best=0 sample
  body=$(python3 -c "import json,sys; print(json.dumps({'messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':2000,'temperature':0.7}))" "$PROMPT")
  for sample in 1 2 3; do
    t0=$(date +%s.%N)
    resp=$(curl -s -m 600 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d "$body" http://127.0.0.1:8081/v1/chat/completions)
    t1=$(date +%s.%N)
    tok=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['usage']['completion_tokens'])" "$resp" 2>/dev/null || echo 0)
    dt=$(python3 -c "print(f'{$t1-$t0:.2f}')")
    tps=$(python3 -c "print(f'{$tok/$dt:.1f}')")
    echo "try $try sample $sample: $tok tok in ${dt}s = $tps t/s" >> "$OUT"
    if python3 -c "import sys; sys.exit(0 if $tps > $best else 1)"; then best=$tps; fi
  done
  echo "try $try: best $best t/s (bar $BAR)" >> "$OUT"
  echo "$best" > "$res"
}
for t in $(seq 1 "$TRIES"); do
  rm -f "$TMP"
  certify_once "$t" "$TMP" || { echo "try $t: boot failed" | tee -a "$OUT"; continue; }
  tps=$(cat "$TMP" 2>/dev/null || echo 0)
  if python3 -c "import sys; sys.exit(0 if $tps >= $BAR else 1)"; then
    echo "CERTIFIED: $tps t/s on try $t (bar $BAR) - the boot stays up" | tee -a "$OUT"
    rm -f "$TMP"; exit 0
  fi
  echo "try $t: $tps t/s below the $BAR bar - re-booting" | tee -a "$OUT"
done
rm -f "$TMP"
echo "DEGRADED: $TRIES boots, none at $BAR t/s - the gate refuses to advertise this state (the last boot is up; see $OUT)" | tee -a "$OUT"
exit 1
