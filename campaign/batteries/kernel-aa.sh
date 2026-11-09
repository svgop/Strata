#!/bin/bash
# campaign/batteries/kernel-aa.sh - the kernel A/A: run AFTER the reboot into 7.2.12 (this script
# verifies it): 3 boots, warm probes, v6 pinned regime, released box.  Compared against the banked
# 7.2.15 anchor (191.9/182.1/257.5 at load 2-4, 18:44) and tonight's v6 7.2.15 bank (216.3/208.2/290.1).
set -u
cd /home/svg/LLM/strata-custom-2
. campaign/battery-lib.sh    # cleanup_engine / wait_health / wait_prewarm (port-scoped kills)
OUT=campaign/results/kernel-aa-7212.log
K=$(uname -r)
echo "KERNEL A/A start $(date) kernel=$K" > "$OUT"
case "$K" in 7.2.12*) ;; *) echo "NOT on 7.2.12 ($K) - aborting; the A/A needs the record kernel" >> "$OUT"; echo "ABORT: kernel $K"; exit 1;; esac
battery_dryrun && { echo "DRYRUN: kernel gate passed ($K) — 3 GSQ boots would run now"; exit 0; }
for r in 1 2 3; do
  ( nohup ./launch-custom-iq3s-8081.sh > "campaign/results/kaa12-$r.out" 2>&1 & )
  ok=1; wait_health 100 || ok=0
  echo "=== KAA boot $r health=$ok $(date +%H:%M:%S) load=$(cut -d' ' -f1 /proc/loadavg) ===" >> "$OUT"
  [ "$ok" = 1 ] || { cleanup_engine; continue; }
  wait_prewarm      # the server's own warm-in must reach a terminal state (replaces the
                    # "page cache warm in" grep — that marker fires before replay terminates)
  sleep 40
  nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.mem --format=csv,noheader >> "$OUT"
  for i in 1 2 3; do
    echo "--- probe $i $(date +%H:%M:%S) ---" >> "$OUT"
    python3 campaign/probe-lane.py --port 8081 --label "kaa12-$r-$i" --short-n 6 >> "$OUT" 2>&1
  done
  cleanup_engine
done
echo "DONE $(date)" >> "$OUT"
echo "ALL DONE"
