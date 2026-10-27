#!/bin/bash
# campaign/prewarm2.sh - T17B FAST-WARM PACKAGE, v2 (post-pin). Usage: prewarm2.sh <engine-log>
# The F-220 prewarm ran BEFORE boot and the boot's page-cache pin defeated it (T17B mechanism 1:
# FileExpertSource madvise(DONTNEED)/fadvise drop at the 41.63 GiB complement pin, #1250). v2 runs
# AFTER the engine is up:
#   1. assert the pin really completed (the log's complement-ready line),
#   2. re-populate the page cache over the pack + shards (dd pass, same set as prewarm.sh),
#   3. replay a few real-shaped /v1 requests to warm pools/graphs/adapt tier: two sequential
#      shorts, one ~2.5k-token prefill, one 6-way concurrent short burst (forces the lazy cohort
#      batch-graph captures).
# Prompts are deliberately DIFFERENT from probe-lane.py's constants so the probe's own shorts
# can never be served from this warm-in via conversation-cache prefixes.
set -u
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
LOG="${1:-/home/svg/LLM/strata-custom-2/custom-iq3_s.log}"
PORT="${PORT:-8081}"
T0=$(date +%s)

grep -q "mapped pinned cache complement ready" "$LOG" \
  && echo "[$(date +%H:%M:%S)] prewarm2: complement pin confirmed in $LOG" \
  || echo "[$(date +%H:%M:%S)] prewarm2: WARNING - pin line not found in $LOG (continuing)"

for f in /home/svg/LLM/data/packs/iq3_s/* \
         /home/svg/LLM/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
         /home/svg/LLM/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf; do
  [ -f "$f" ] || continue
  dd if="$f" of=/dev/null bs=8M status=none 2>/dev/null || cat "$f" > /dev/null 2>/dev/null || true
done
# campaign law 11: every mapping the run reads pays its first-touch faults in warmup.  The shadow
# store (F-360) is mmap'd but never touched by the replay, so without this pass its 31.64 GiB of
# page faults land in the first measured leg (~15% error, F-360 leg 1).
# The launcher exports STRATA_SHADOW_EXPERTS into the ENGINE's environment, not this shell's, so the
# path is read back from the running process (env-verify discipline: the process is the truth).
if [ -z "${STRATA_SHADOW_EXPERTS:-}" ]; then
  _ep=$(pgrep -x strata | head -1)
  if [ -n "${_ep:-}" ] && [ -r "/proc/$_ep/environ" ]; then
    STRATA_SHADOW_EXPERTS=$(tr '\0' '\n' < "/proc/$_ep/environ" | sed -n 's/^STRATA_SHADOW_EXPERTS=//p' | head -1)
    echo "[$(date +%H:%M:%S)] prewarm2: shadow path read from engine pid $_ep: ${STRATA_SHADOW_EXPERTS:-<unset>}"
  fi
fi
if [ -n "${STRATA_SHADOW_EXPERTS:-}" ] && [ -f "$STRATA_SHADOW_EXPERTS/experts.bin" ]; then
  case "$STRATA_SHADOW_EXPERTS" in
    /dev/shm/*)
      # F-367: tmpfs pages are RAM-resident — the first-touch fault pass below is a no-op read of
      # RAM; skip it (the copy itself is campaign/shm-shadow.sh's job after reboot).
      echo "[$(date +%H:%M:%S)] prewarm2: shadow on tmpfs ($STRATA_SHADOW_EXPERTS) — RAM-resident, no warm pass needed"
      ;;
    *)
      dd if="$STRATA_SHADOW_EXPERTS/experts.bin" of=/dev/null bs=8M status=none 2>/dev/null || true
      echo "[$(date +%H:%M:%S)] prewarm2: shadow store warmed ($STRATA_SHADOW_EXPERTS/experts.bin)"
      ;;
  esac
fi
echo "[$(date +%H:%M:%S)] prewarm2: page cache re-warmed in $(( $(date +%s) - T0 ))s"

ask() {  # ask <max_tokens> <prompt | @file> ; body built by python (no shell quoting hazards)
  python3 - "$1" "$2" <<'PY' > /tmp/prewarm2-body.json
import json, os, sys
a = sys.argv[2]
prompt = open(a[1:]).read() if a.startswith('@') else a
print(json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":int(sys.argv[1]),"temperature":0.7}))
PY
  curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
    -d @/tmp/prewarm2-body.json -o /dev/null -w '%{http_code}\n'
}
S1='Sketch the lifecycle of a container image from build to retirement, naming the layers, registries and failure points you would instrument along the way.'
S2='A search engine keeps returning stale results after reindexing. Walk through the debugging ladder you would climb, from cache headers to replication lag, with the tool you would use at each rung.'
python3 - <<'PY' > /tmp/prewarm2-mid.txt
para = ("Datacenter networking has shifted from oversubscribed fat trees toward spine-leaf fabrics "
        "with predictable hop counts, and the engineering questions moved with it. "
        "Congestion control now lives in NICs, telemetry is sampled per queue rather than per switch, "
        "and the unit of capacity planning became the rail rather than the port. ")
print(("Review note: " + para * 90)[:11000])
PY
MID=@/tmp/prewarm2-mid.txt
CONC='Two engineers disagree about whether to shard a database by tenant or by region. Draft the one-paragraph decision memo you would send them.'

ask 200 "$S1"; ask 200 "$S2"
ask 260 "$MID"
for i in 1 2 3 4 5 6; do ask 200 "$CONC (variant $i)" & done
wait
echo "[$(date +%H:%M:%S)] prewarm2: replay done (2 short + 1 mid prefill + 6-way burst) in $(( $(date +%s) - T0 ))s total"
