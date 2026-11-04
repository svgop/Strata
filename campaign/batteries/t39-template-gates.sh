#!/bin/bash
# campaign/batteries/t39-template-gates.sh — F-279 template gate: the terse-dialect effort
# set (chat_template.jinja + EFFORT_TEXT mirror) judged by (1) the think-probe (12 fixed
# reasoning requests; baseline on the old template = mean 634 completion tok / 3.2s wall,
# 2 of 6 prompts burning the whole 900-token budget), (2) 2x MMLU-Pro (floor 0.933; banked
# band 95.7 x2), (3) mimic (banked class 60/66). Server LEFT RUNNING on the new template.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/t39-template-gates.log
echo "T39 TEMPLATE GATES start $(date)" > "$OUT"


cleanup_engine; sleep 2
[ -n "$(port_pid)" ] && { echo "ABORT: port $PORT busy" >> "$OUT"; exit 1; }
( cd "$REPO" && nohup bash "$LAUNCH" >"$RES/t39-tg.out" 2>&1 & )
ok=0
for t in $(seq 1 100); do
  [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
  sleep 3
done
echo "=== BOOT t39-template health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
if [ "$ok" = 1 ]; then
  wait_prewarm
  echo "--- think-probe (new template):" >> "$OUT"
  ( cd "$REPO" && python3 campaign/template-probe.py 8081 new-template 2 2>&1 | tee "$RES/template-probe-new.txt" >> "$OUT" )
  cd "$REPO"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t39-tpl-mmlu-a >> "$OUT" 2>&1 || echo "MMLU A failed" >> "$OUT"
  timeout 600 .venv/bin/python campaign/mmlu-pro-gate.py 0.933 t39-tpl-mmlu-b >> "$OUT" 2>&1 || echo "MMLU B failed" >> "$OUT"
  timeout 900 .venv/bin/python campaign/mimicbench2.py 8081 t39-tpl-mimic >> "$OUT" 2>&1 || echo "MIMIC failed" >> "$OUT"
fi
echo "SERVER LEFT RUNNING (new template) $(date +%H:%M:%S)" >> "$OUT"
echo "T39 TEMPLATE GATES end $(date)" >> "$OUT"
echo "done" >> "$OUT"
