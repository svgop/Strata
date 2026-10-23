#!/bin/bash
# campaign/batteries/t40-verify.sh — verification boot for the main-loop T40 tranche-1
# instruments: C5b (cohort WSTAT now covers window+commit+emit), the batch-lane trace twin
# (STRATA_SPEC_TRACE "b " lines), plus the standing E1 slots-24 + PDL-pin + template.
# Checks: boot lines (DM armed, pdl line, conversation_cache slots), /metrics eviction
# counter (E1: expect 0 evictions at 24 slots on light traffic), WSTAT histogram populated
# after cohort traffic, trace twin writes "b " lines when armed. Server LEFT RUNNING.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t40-verify.log
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
echo "T40 VERIFY start $(date)" > "$OUT"

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

cleanup_engine; sleep 2
[ -n "$(port_pid)" ] && { echo "ABORT: port busy" >> "$OUT"; exit 1; }
cp "$REPO/engine/strata" "$REPO/engine/strata.pre-t40"
cp "$REPO/build/strata" "$REPO/engine/strata" && echo "deployed T40-instruments binary; rollback engine/strata.pre-t40" >> "$OUT"

( cd "$REPO" && env STRATA_SPEC_TRACE=/tmp/t40-btrace.txt nohup bash "$LAUNCH" >"$RES/t40-verify.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT t40-verify health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  grep -a "STRATA_DM armed\|strata pdl\|conversation.cache\|COHORTS" "$REPO/custom-iq3_s.log" | tail -4 >> "$OUT"
  # cohort-lane traffic: 6 concurrent short requests, two waves (rotate slots, park/restore)
  python3 - "$PORT" "$KEY" >> "$OUT" 2>&1 <<'PY'
import json, sys, time, urllib.request
port, key = sys.argv[1], sys.argv[2]
def ask(p, mt=120):
    body = json.dumps({"messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0.7}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r: json.loads(r.read())
t0 = time.time()
for wave in range(2):
    import threading
    ths = [threading.Thread(target=ask, args=(f"Explain in two sentences how a {x} heat pump works.", 120,)) for x in ("ground-source","air-source","hybrid","ductless","absorption","geothermal")]
    [t.start() for t in ths]; [t.join() for t in ths]
    time.sleep(4)
print(f"cohort traffic done: 12 requests in {time.time()-t0:.0f}s")
PY
  sleep 3
  echo "--- /metrics after cohort traffic:" >> "$OUT"
  curl -s -m 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
    | grep -ao '"elastic": {[^}]*}\|"conversation_cache": {[^}]*}\|"window": {[^}]*}' >> "$OUT"
  echo "--- trace twin (b lines):" >> "$OUT"
  head -3 /tmp/t40-btrace.txt >> "$OUT" 2>/dev/null; wc -l /tmp/t40-btrace.txt >> "$OUT" 2>/dev/null
  echo "--- WSTAT lines:" >> "$OUT"
  grep -a "^WSTAT" "$REPO/custom-iq3_s.log" | tail -2 >> "$OUT"
fi
echo "SERVER LEFT RUNNING $(date +%H:%M:%S)" >> "$OUT"
echo "T40 VERIFY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
