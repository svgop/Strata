#!/bin/bash
# campaign/prewarm-ds4.sh - DS warm-up (the prewarm2.sh discipline, DeepSeek lane).
# Runs AFTER the dsv41 engine is up on 8081: re-populate the page cache over the pack and
# shards (the 109.5 GB store loads from them), then replay two shorts and one ~2.5k prefill
# so the dspark drafter, the expert cache and the slots folder are warm before any measured
# leg (first-light: cold leg 16.6 t/s vs 36.8 warm on the same boot).
# Usage: prewarm-ds4.sh [engine-log]
set -u
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
PORT="${PORT:-8081}"
T0=$(date +%s)

for f in /home/svg/LLM/data/packs/deepseek-v4.1-flash-ist-das-2.00bpw/* \
         /home/svg/LLM/data/models/deepseek-v4.1-flash-ist-das-2.00bpw*.gguf \
         /home/svg/LLM/data/models/deepseek-v4.1-flash-shared/*; do
  [ -f "$f" ] || continue
  dd if="$f" of=/dev/null bs=8M status=none 2>/dev/null || true
done
echo "[$(date +%H:%M:%S)] prewarm-ds4: page cache re-warmed in $(( $(date +%s) - T0 ))s"

ask() {
  python3 - "$1" "$2" <<'PY' > /tmp/prewarm-ds4-body.json
import json, sys
a = sys.argv[2]
prompt = open(a[1:]).read() if a.startswith('@') else a
print(json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":int(sys.argv[1]),"temperature":1.0}))
PY
  curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
    -d @/tmp/prewarm-ds4-body.json -o /dev/null -w '%{http_code}\n'
}

S1='Explain in two paragraphs how speculative drafting with a small drafter model changes the latency profile of a mixture-of-experts serving stack.'
S2='A vector database returns stale results after a merge. List the debugging ladder you would climb, one tool per rung.'
python3 - <<'PY' > /tmp/prewarm-ds4-mid.txt
para = ("The expert store holds thousands of small blocks and the router picks a handful per token; "
        "the hot set follows the conversation, the resident set follows the heat, and the PCIe link "
        "carries whatever neither tier holds. ")
print("Warm note: " + para * 80)[:9000]
PY

ask 200 "$S1"; ask 200 "$S2"
ask 200 "@/tmp/prewarm-ds4-mid.txt"
echo "[$(date +%H:%M:%S)] prewarm-ds4: replay done in $(( $(date +%s) - T0 ))s total"
