#!/bin/bash
# gsq-pinning-ab.sh - A/B the engine CCD pinning (v6 clocks both legs): ctl = unpinned vs pin = the
# frequency CCD 8-15,24-31. 2+2 interleaved, prewarm-gated, 3 probes each.
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-pinning-ab.log
echo "PINNING AB start $(date)" > "$OUT"
leg() {
  local tag=$1; shift
  ( nohup env "$@" ./launch-custom-iq3s-8081.sh > "campaign/results/pin-$tag.out" 2>&1 & )
  local ok=0 t=
  for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
  echo "=== PIN-LEG $tag health=$ok $(date +%H:%M:%S) load=$(cut -d' ' -f1 /proc/loadavg) ===" >> "$OUT"
  [ "$ok" = 1 ] || { pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null; return; }
  for t in $(seq 1 60); do grep -aq "page cache warm in" "campaign/results/pin-$tag.out" && break; sleep 3; done
  sleep 40
  grep -aoE "GPU-reach wait [0-9.]+" logs/custom-iq3_s.log | tail -1 >> "$OUT"
  for i in 1 2 3; do
    echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "pin-$tag-$i" --short-n 6 >> "$OUT" 2>&1
  done
  pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
  sleep 2; pkill -9 -x strata 2>/dev/null; sleep 5
}
leg ctl1 STRATA_ENGINE_CPUS=
leg pin1 STRATA_ENGINE_CPUS=8-15,24-31
leg ctl2 STRATA_ENGINE_CPUS=
leg pin2 STRATA_ENGINE_CPUS=8-15,24-31
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
