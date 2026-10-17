#!/bin/bash
# T12: paired prefill probe — boots two configs back to back, times N unique ~6k-token
# prompts (max_tokens=1 -> wall ~= prefill) on each, prints medians. Usage:
#   prefill-probe.sh <config-a.json> <config-b.json> [samples]
# GDN_CHUNKED costs ~0.18 us/token x 36 layers = ~39 ms on a 6k prompt: the sample size and
# the alternation are what make the pair readable.
set -u
cd /home/svg/LLM/strata-custom-2
A="$1"; B="$2"; N="${3:-4}"
WORDS="${4:-6400}"
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"

boot() {  # $1 config json (relative)
  pkill -f 'serve/serve[r].py' 2>/dev/null; pkill -f 'build/strat[a] ' 2>/dev/null; pkill -f 'build/strat[a]$' 2>/dev/null
  sleep 3
  pkill -9 -f 'serve/serve[r].py' 2>/dev/null; pkill -9 -f 'build/strat[a] ' 2>/dev/null; pkill -9 -f 'build/strat[a]$' 2>/dev/null
  sleep 2
  eval "$(grep '^export ' launch-custom-iq3s-8081.sh)"
  bash campaign/prewarm.sh
  nohup .venv/bin/python serve/server.py --engine strata --config "$PWD/$1" --port 8081 --host 127.0.0.1 > server-prefillprobe.out 2>&1 &
  disown
  for i in $(seq 1 240); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" --max-time 3 http://127.0.0.1:8081/v1/models 2>/dev/null)
    [ "$CODE" = "200" ] && { sleep 8; return 0; }
    pgrep -f 'serve/serve[r].py' > /dev/null || { echo "SERVER DIED booting $1"; tail -5 server-prefillprobe.out; return 1; }
    sleep 3
  done
  echo "NEVER READY $1"; return 1
}

samples() {  # $1 label
  for i in $(seq 1 "$N"); do
    .venv/bin/python - "$i" "$1" "$WORDS" << 'PYEOF'
import json, random, string, sys, time, urllib.request
i, label, W = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
random.seed(91000 + i * 7919)
words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 9))) for _ in range(W)]
prompt = 'P' + ''.join(random.choices(string.hexdigits.upper(), k=16)) + ' doc: ' + ' '.join(words) + ' Summarize in one word:'
body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 1}).encode()
req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", data=body,
                             headers={"Authorization": "Bearer 3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8",
                                      "Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=300) as r:
    json.load(r)
print(f"{label} {i} {time.perf_counter() - t0:.3f}s")
PYEOF
  done
}

boot "$A" || exit 1
samples A
boot "$B" || exit 1
samples B
echo PREFILL-PROBE-DONE
