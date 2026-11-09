#!/bin/bash
# campaign/batteries/era-interleave.sh - THE branch verdict: the banked-era binary (4152f114, the
# 233.1-spot commit) vs the current binary, both RELEASED (pool=0, full RAM), interleaved E-C-E-C,
# prewarm-gated, 3 probes each boot.  Witnesses: the era binary prints no ARENA PAGES banner
# (predates it) - the boot marker is the engine version line.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/era-interleave.log
echo "ERA INTERLEAVE (released box) start $(date)" > "$OUT"
cp engine/strata /tmp/strata.current
leg() {
  local tag=$1 bin=$2
  cp "$bin" engine/strata
  local before; before=$(wc -l < logs/custom-iq3_s.log)
  ( nohup ./launch-custom-iq3s-8081.sh > "campaign/results/era-$tag.out" 2>&1 & )
  local ok=0 t=
  for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
  echo "=== LEG $tag health=$ok bin=$bin $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null; return; }
  for t in $(seq 1 60); do grep -aq "page cache warm in" "campaign/results/era-$tag.out" && break; sleep 3; done
  sleep 45
  for i in 1 2 3; do
    echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "era-$tag-$i" --short-n 6 >> "$OUT" 2>&1
  done
  tail -n +$before logs/custom-iq3_s.log | grep -aE "ARENA PAGES|the expert complement|control vector mode" | head -2 >> "$OUT"
  pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
  sleep 2; pkill -9 -x strata 2>/dev/null; sleep 5
}
leg era1 /home/svg/LLM/banked-era-4152/build-era/strata
leg cur1 /tmp/strata.current
leg era2 /home/svg/LLM/banked-era-4152/build-era/strata
leg cur2 /tmp/strata.current
cp /tmp/strata.current engine/strata
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
