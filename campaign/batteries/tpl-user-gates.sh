#!/bin/bash
# campaign/batteries/tpl-user-gates.sh — the user's candidate chat template
# (Downloads/chat_template (4).jinja) judged live, leg 1: swap in, boot, exact
# prompt-token overhead, think-probe x2 (baseline: standing F-279 = 608 completion
# tok mean / answers ~929 ch), MMLU x2 (floor 0.933), mimic leg A.
# Standing template is committed (F-279): git restore serve/chat_template.jinja = revert.
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/tpl-user-gates.log
KEY=3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8
echo "TPL-USER GATES leg1 start $(date)" > "$OUT"

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
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy" >> "$OUT"; exit 1; }

cp "/home/svg/Downloads/chat_template (4).jinja" "$REPO/serve/chat_template.jinja"
echo "swapped in user template: $(md5sum "$REPO/serve/chat_template.jinja" | cut -c1-12)" >> "$OUT"

( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/tpl-user-boot.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT user-template health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  ( cd "$REPO" && bash campaign/prewarm2.sh "$REPO/custom-iq3_s.log" >> "$OUT" 2>&1 )
  echo "--- exact prompt overhead (1-token request, usage.prompt_tokens):" >> "$OUT"
  curl -s -m 120 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
    -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":1}' \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('prompt_tokens:', d.get('usage',{}).get('prompt_tokens'))" >> "$OUT" 2>&1
  echo "--- think-probe (user template) x2:" >> "$OUT"
  ( cd "$REPO" && python3 campaign/template-probe.py 8081 user-template 2 2>&1 | tee "$RES/template-probe-user.txt" >> "$OUT" )
  cd "$REPO"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 tpl-user-mmlu-a >> "$OUT" 2>&1 || echo "MMLU A failed" >> "$OUT"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 tpl-user-mmlu-b >> "$OUT" 2>&1 || echo "MMLU B failed" >> "$OUT"
  timeout 900 .venv/bin/python campaign/mimicbench2.py 8081 tpl-user-mimic-a >> "$OUT" 2>&1 || echo "MIMIC A failed" >> "$OUT"
fi
echo "TPL-USER GATES leg1 end $(date)" >> "$OUT"
echo "done" >> "$OUT"
