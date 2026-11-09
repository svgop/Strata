#!/bin/bash
# campaign/batteries/gsq-bisect.sh - THE bisection: A = the Nov-6 serving environment on today's
# binary (ARENAS=0: one-shot cudaHostAlloc complement; memfd off; tmpfs store tier) vs B = the
# current standing environment (pool complement + memfd store).  2+2 interleaved, prewarm-gated.
set -u
. "$(dirname "$0")/../battery-lib.sh"
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
RES=$REPO/campaign/results
OUT=$RES/gsq-bisect.log
echo "GSQ BISECT (A=Nov6-env vs B=current-env) start $(date)" > "$OUT"

leg() {  # leg <tag> <env...>
  local tag=$1; shift
  cleanup_engine
  before=$(wc -l < "$REPO/logs/custom-iq3_s.log")
  ( cd "$REPO" && nohup env "$@" ./launch-custom-iq3s-8081.sh >"$RES/gsq-bisect-$tag.out" 2>&1 & )
  ok=0 t=
  for t in $(seq 1 100); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG gsq-bisect-$tag health=$ok $(date +%H:%M:%S) env:$* ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; return; }
  wait_prewarm
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "gsq-bisect-$tag" --short-n 6 >> "$OUT" 2>&1 )
  tail -n +$before "$REPO/logs/custom-iq3_s.log" | grep -aE "hugepages: the expert complement|memfd, populated|store mapped|cudaHostAlloc" | head -2 >> "$OUT"
  cleanup_engine
}

leg a1 STRATA_HUGEPAGE_ARENAS=0 STRATA_SHADOW_MEMFD=0 STRATA_SHADOW_EXPERTS=/dev/shm/strata-shadow-q2_0
leg b1 STRATA_SHADOW_EXPERTS=/home/svg/LLM/data/packs/q2_0
leg a2 STRATA_HUGEPAGE_ARENAS=0 STRATA_SHADOW_MEMFD=0 STRATA_SHADOW_EXPERTS=/dev/shm/strata-shadow-q2_0
leg b2 STRATA_SHADOW_EXPERTS=/home/svg/LLM/data/packs/q2_0
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
