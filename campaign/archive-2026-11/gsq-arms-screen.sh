#!/bin/bash
# campaign/batteries/gsq-arms-screen.sh - the two in-tree arms closest to the 12.27 ms doorbell term,
# one boot each on the released box: STRATA_DOORBELL_STORE=1 and STRATA_VERIFY_QDEDUP=1, each vs the
# standing launcher as control, warm probes, kill-line +8% solo or park.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-arms-screen.log
echo "ARMS SCREEN start $(date)" > "$OUT"
leg() {
  local tag=$1; shift
  local before; before=$(wc -l < logs/custom-iq3_s.log)
  ( nohup env "$@" ./launch-custom-iq3s-8081.sh > "campaign/results/arms-$tag.out" 2>&1 & )
  local ok=0 t=
  for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
  echo "=== LEG $tag health=$ok env:$* $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null; return; }
  for t in $(seq 1 60); do grep -aq "page cache warm in" "campaign/results/arms-$tag.out" && break; sleep 3; done
  sleep 40
  for i in 1 2 3; do
    echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "arm-$tag-$i" --short-n 6 >> "$OUT" 2>&1
  done
  pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
  sleep 2; pkill -9 -x strata 2>/dev/null; sleep 5
}
leg ctl1 DUMMY=0
leg dbs1 STRATA_DOORBELL_STORE=1
leg qdd1 STRATA_VERIFY_QDEDUP=1
leg ctl2 DUMMY=0
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
