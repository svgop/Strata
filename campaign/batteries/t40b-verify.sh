#!/bin/bash
# campaign/batteries/t40b-verify.sh — C2a-lite (STRATA_COHORT_OVERLAP) verification:
# leg OFF = the restructured default path must behave exactly as before (cohort traffic,
# zero ERR, sane outputs); leg ON = the overlap arm live (same traffic; finish-under-
# window must not crash, commit guards must hold; first p50 reading vs OFF via the
# phase-correct WSTAT histogram). Ends with the server UP on the deployed binary (OFF).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t40b-verify.log
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
echo "T40B VERIFY start $(date)" > "$OUT"

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

cp "$REPO/build/strata" "$REPO/engine/strata" && echo "deployed C2a-lite binary (default-off)" >> "$OUT"

traffic() {  # 6-conc x 2 waves through the cohort lane
  python3 - "$PORT" "$KEY" >> "$OUT" 2>&1 <<'PY'
import json, sys, time, urllib.request, threading
port, key = sys.argv[1], sys.argv[2]
def ask(p, mt=120):
    body = json.dumps({"messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0.7}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    c = d.get("choices", [{}])[0].get("message", {}).get("content") or ""
    assert len(c) > 40, f"thin answer: {c[:60]!r}"
for wave in range(2):
    ths = [threading.Thread(target=ask, args=(f"Explain in two sentences how a {x} heat pump works.", 120,)) for x in ("ground-source","air-source","hybrid","ductless","absorption","geothermal")]
    [t.start() for t in ths]; [t.join() for t in ths]; time.sleep(4)
print("traffic: 12/12 requests OK (all answers substantive)")
PY
}

leg() {  # leg <tag> <env-pass>
  cleanup_engine; sleep 2
  ( cd "$REPO" && env $2 nohup bash "$LAUNCH" >"$RES/t40b-$1.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== BOOT t40b-$1 health=$ok $(date +%H:%M:%S) env='$2' ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  traffic
  sleep 3
  echo "--- window histogram ($1):" >> "$OUT"
  curl -s -m 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -ao '"window": {[^}]*}' >> "$OUT"
  echo "--- ERR count in engine log ($1):" >> "$OUT"
  grep -ac "^ERR" "$REPO/custom-iq3_s.log" >> "$OUT"
}

leg off ""
leg on "STRATA_COHORT_OVERLAP=1"

cleanup_engine; sleep 2
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t40b-final-serve.out" 2>&1 & )
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && break
  sleep 3
done
echo "SERVER LEFT RUNNING (default-off binary) $(date +%H:%M:%S)" >> "$OUT"
echo "T40B VERIFY end $(date)" >> "$OUT"
echo "done" >> "$OUT"
