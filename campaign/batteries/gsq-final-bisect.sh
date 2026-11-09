#!/bin/bash
# campaign/batteries/gsq-final-bisect.sh - THE clean discriminator: A = full Nov-6 environment on
# today's binary (NO_LARGEPAGES=1 strips every pool draw; memfd off; tmpfs store) vs B = standing
# config.  2+2 interleaved, prewarm-gated probes, witnesses per boot.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-final-bisect.log
echo "FINAL BISECT start $(date)" > "$OUT"
leg() {
  local tag=$1; shift
  local before; before=$(wc -l < logs/custom-iq3_s.log)
  ( nohup env "$@" ./launch-custom-iq3s-8081.sh > "campaign/results/gsq-final-bisect-$tag.out" 2>&1 & )
  local ok=0 t=
  for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
  echo "=== LEG $tag health=$ok $(date +%H:%M:%S) env:$* ===" >> "$OUT"
  [ "$ok" = 1 ] || { pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null; return; }
  for t in $(seq 1 60); do grep -aq "page cache warm in" "campaign/results/gsq-final-bisect-$tag.out" && break; sleep 3; done
  sleep 45
  for i in 1 2 3; do
    echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "fx-$tag-$i" --short-n 6 >> "$OUT" 2>&1
  done
  tail -n +$before logs/custom-iq3_s.log | grep -aE "the expert complement:|memfd, populated|store mapped|ARENA PAGES" | head -3 >> "$OUT"
  pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
  sleep 2; pkill -x strata 2>/dev/null; sleep 5
}
leg a1 STRATA_NO_LARGEPAGES=1 STRATA_SHADOW_MEMFD=0 STRATA_SHADOW_EXPERTS=/dev/shm/strata-shadow-q2_0
leg b1 STRATA_SHADOW_EXPERTS=/home/svg/LLM/data/packs/q2_0
leg a2 STRATA_NO_LARGEPAGES=1 STRATA_SHADOW_MEMFD=0 STRATA_SHADOW_EXPERTS=/dev/shm/strata-shadow-q2_0
leg b2 STRATA_SHADOW_EXPERTS=/home/svg/LLM/data/packs/q2_0
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
