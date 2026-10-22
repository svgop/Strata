#!/bin/bash
# campaign/t22-custom-vs-stock.sh — same-window 3v3: custom (merged v0.2.1 line) vs stock v0.2.1,
# both on the identical IQ3_S config/bytes. Stock legs on 8080 (stock harness), custom on 8081.
set -u
CUST=/home/svg/LLM/strata-custom-2
STOCK=/home/svg/LLM/strata
OUT=$CUST/campaign/results/t22-custom-vs-stock.log
RES=$CUST/campaign/results
mkdir -p "$RES"
echo "T22 CUSTOM vs STOCK start $(date)" > "$OUT"

pids() { ss -ltnp 2>/dev/null | grep -E ':(8080|8081) ' | grep -oP 'pid=\K[0-9]+'; }
cleanup() {
  local p pid c
  for p in $(pids); do kill -TERM "$p" 2>/dev/null; done
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *strata*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
}
boot() {  # $1 dir $2 script $3 port $4 label
  cleanup; sleep 2
  ( cd "$1" && nohup "$2" >"$RES/t22-$4.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    if [ -n "$(curl -s -m 3 http://127.0.0.1:$3/health 2>/dev/null)" ]; then ok=1; break; fi
    sleep 3
  done
  echo "=== LEG $4 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ]
}
leg() {  # $1 dir $2 script $3 port $4 label
  boot "$1" "$2" "$3" "$4" || return
  cd "$CUST" && python3 campaign/probe-lane.py --port "$3" --label "t22-$4" --short-n 6 >> "$OUT" 2>&1
  cleanup
}

for r in 1 2 3; do
  leg "$STOCK" "$STOCK/run-iq3s-stock.sh" 8080 "stock-r$r"
  leg "$CUST"  "$CUST/launch-custom-iq3s-8081.sh" 8081 "custom-r$r"
done
echo "T22 end $(date)" >> "$OUT"
echo "done" >> "$OUT"
