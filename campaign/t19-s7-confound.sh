#!/bin/bash
# campaign/t19-s7-confound.sh — T19-S7: the F-244 confound diagnostic on the STOCK v0.2.1 tree.
# Question: how much of stock .2.1's +15.6% M1 wave (vs .45) is the sm_120 KERNEL PACKAGE
# (ba2c2d5f: SIDE_MIXER / ATTN_DAILY / GR_UP75 / GR_DOWN_TG, default-on on this card) vs the
# I/O-tier + other defaults? Method: 3v3 ABAB on the stock binary alone —
#   arm "pkg-on"  : stock defaults (the four switches unset = on for cc 12.0)
#   ctl "pkg-off" : STRATA_SIDE_MIXER=0 STRATA_ATTN_DAILY=0 STRATA_GR_UP75=0 STRATA_GR_DOWN_TG=0
# Read-only w.r.t. the stock tree: only engine/strata is re-copied from its own v0.2.1 artifact
# (F-244's own deploy step), and the final deploy restores it. Runs on port 8080, full probe
# --short-n 6, plus the S3 re-open rider: the fixed 4-prompt greedy identity set on every leg.
set -u
STOCK=/home/svg/LLM/strata
REPO=/home/svg/LLM/strata-custom-2
PORT=8080
KEY="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
OUT=$REPO/campaign/results/t19-s7.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T19 S7 confound start $(date) binary: $(md5sum $STOCK/engine/strata.v0.2.1 | cut -c1-8)" > "$OUT"

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

greedy_set() {  # the S3 re-open rider: fixed greedy prompts, hash + length per prompt
  local tag=$1
  for p in "The capital of France is? Answer with the city only." "List the first five prime numbers." \
           "Write the word strawberry." "Explain why the sky is blue in exactly two sentences."; do
    curl -s -m 120 http://127.0.0.1:$PORT/v1/chat/completions -H "Authorization: Bearer $KEY" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"qwen3.8-flash-next-iq3_s\", \"messages\": [{\"role\": \"user\", \"content\": \"$p\"}], \"max_tokens\": 150, \"temperature\": 0}" \
      | python3 -c "import json,sys,hashlib; d=json.load(sys.stdin)['choices'][0]['message']; t=str(d.get('reasoning_content'))+chr(0)+str(d.get('content')); print(hashlib.sha256(t.encode()).hexdigest()[:16], len(t))" 2>/dev/null
  done | sed "s/^/$tag /" >> "$OUT"
}

boot_probe() {  # $1 label, $2.. env
  local label=$1; shift
  local logf=$RES/t19-s7-$label.out
  cleanup_engine; sleep 2
  [ -n "$(port_pid)" ] && { echo "=== $label ABORT port busy ===" >> "$OUT"; return 1; }
  cp "$STOCK/engine/strata.v0.2.1" "$STOCK/engine/strata"
  ( cd "$STOCK" && nohup env "$@" .venv/bin/python serve/server.py --engine strata --config "$STOCK/strata-iq3s-stock.json" --port $PORT --host 0.0.0.0 >"$logf" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 120); do
    curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG s7-$label health=$ok bin=$(md5sum $STOCK/engine/strata | cut -c1-8) $(date +%H:%M:%S) ===" >> "$OUT"
  if [ "$ok" != 1 ]; then echo "LEG s7-$label FAILED" >> "$OUT"; tail -40 "$logf" >> "$OUT"; return 1; fi
  cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t19-s7-$label" --short-n 6 >> "$OUT" 2>&1
  greedy_set "greedy-$label"
  cleanup_engine
}

for r in 1 2 3; do
  boot_probe "on-r$r"    # stock defaults: the four ba2c2d5f switches on
  boot_probe "off-r$r" STRATA_SIDE_MIXER=0 STRATA_ATTN_DAILY=0 STRATA_GR_UP75=0 STRATA_GR_DOWN_TG=0
done
cp "$STOCK/engine/strata.v0.2.1" "$STOCK/engine/strata"
echo "T19 S7 confound end $(date); stock deploy restored to v0.2.1" >> "$OUT"
echo "done" >> "$OUT"
