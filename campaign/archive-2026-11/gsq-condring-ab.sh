#!/bin/bash
# campaign/batteries/gsq-condring-ab.sh - law-17 ABAB: standing binary (ctl) vs the COND_RING binary
# (STRATA_COND_RING=1 STRATA_WINDOW_GRAPH=1), 3+3 interleaved boots, prewarm-gated 3 probes each.
# Keep line per the design: +3% solo with no lane worse. Watch the armed/refused boot line.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-condring-ab.log
echo "COND_RING ABAB start $(date)" > "$OUT"
CR=/home/svg/LLM/condring-build/build/strata
# ST binary removed 2026-11-09 cleanup (snapshot farm pruned): regen = apply
# campaign/results/condring-parked.patch to the tree, build with the banked cmake recipe, cp to engine/strata.
ST=/home/svg/LLM/strata-custom-2/engine/strata.pre-condring
leg() {
  local tag=$1 bin=$2; shift 2
  cp "$bin" engine/strata
  local before; before=$(wc -l < logs/custom-iq3_s.log)
  ( nohup env "$@" ./launch-custom-iq3s-8081.sh > "campaign/results/cr-$tag.out" 2>&1 & )
  local ok=0 t=
  for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
  echo "=== LEG $tag health=$ok bin=$(basename $bin) env:$* $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ] || { pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null; return; }
  for t in $(seq 1 60); do grep -aq "page cache warm in" "campaign/results/cr-$tag.out" && break; sleep 3; done
  sleep 40
  tail -n +$before logs/custom-iq3_s.log | grep -aE "STRATA_COND_RING|COND_RING armed|refused" | head -2 >> "$OUT"
  for i in 1 2 3; do
    echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "cr-$tag-$i" --short-n 6 >> "$OUT" 2>&1
  done
  pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
  sleep 2; pkill -9 -x strata 2>/dev/null; sleep 5
}
leg ctl1 "$ST"
leg cr1 "$CR" STRATA_COND_RING=1 STRATA_WINDOW_GRAPH=1
leg ctl2 "$ST"
leg cr2 "$CR" STRATA_COND_RING=1 STRATA_WINDOW_GRAPH=1
leg ctl3 "$ST"
leg cr3 "$CR" STRATA_COND_RING=1 STRATA_WINDOW_GRAPH=1
cp "$ST" engine/strata
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
