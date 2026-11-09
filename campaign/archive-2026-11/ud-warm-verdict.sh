#!/bin/bash
# campaign/batteries/ud-warm-verdict.sh - the UD packs' warm-residency banks, SERIAL (one launcher
# at a time; the parallel-battery era is over).  Per pack: one boot, prewarm-complete + 60 s settle,
# three probes, witnesses.  Probe 3 = the bank-era-comparable number.
set -u
cd /home/svg/LLM/strata-custom-2
run_pack() {
  local launch=$1 log=$2 out=$3 tag=$4
  local OUT=campaign/results/ud-warm-$tag.log
  echo "UD WARM $tag start $(date)" > "$OUT"
  ( nohup ./"$launch" > "campaign/results/$out" 2>&1 & )
  for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 3; done
  curl -s -m 3 http://127.0.0.1:8081/health >/dev/null || { echo "BOOT FAILED" >> "$OUT"; return; }
  local T0=$(date +%s)
  for t in $(seq 1 60); do grep -aq "page cache warm in" "campaign/results/$out" && break; sleep 3; done
  sleep 60
  echo "== warm residency, boot+prewarm+settle $(( $(date +%s) - T0 ))s ==" >> "$OUT"
  for i in 1 2 3; do
    echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "udw-$tag-$i" --short-n 6 >> "$OUT" 2>&1
  done
  tail -n 600 "$log" | grep -aE "hugetlb 2 MB pages|the expert complement:|memfd, populated" | head -3 >> "$OUT"
  pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-ud" 2>/dev/null
  sleep 2; pkill -x strata 2>/dev/null; sleep 5
  echo "UD WARM $tag DONE $(date)" >> "$OUT"
}
run_pack launch-custom-ud-iq2xs-8081.sh logs/custom-ud-iq2_xs.log ud-iq2xs-boot.out iq2xs
run_pack launch-custom-ud-iq4xs-8081.sh logs/custom-ud-iq4xs.log ud-iq4xs-boot.out iq4xs
echo "ALL DONE"
