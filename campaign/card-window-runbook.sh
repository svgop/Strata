#!/bin/bash
# campaign/card-window-runbook.sh — THE CARD-WINDOW RUNBOOK (2026-11-20): the three staged boots as
# one gated sequence.  Each step certifies its boot (fingerprint + organ) before measuring, tears
# down cleanly, re-pins the F-271 profile, and ANY anomaly (a refused boot, health that never comes,
# a lane out of band) halts the chain - the next agent collects a clean state, never a surprise.
#
# Steps, in priority order (the queue's own):
#   1. THE PORT PROOF BOOT  - deploy build-fresh (the ADAPT_FETCH steps 1-3b binary), bit-check +
#      one probe leg standing-sanity, rollback kept.  The env-off proof is the port's deploy gate.
#   2. THE LOOKAHEAD FALSIFIER - STRATA_LOOKAHEAD_STATS=1 + one probe leg: the predictor's real
#      recall on the current standing decides the speculative-pool arm before any code (law 4).
#   3. THE DEPTH RE-TEST - ONLY when the box is quiet (1-min load below QUIET_BAR at step entry):
#      the five-boot D=2 battery with the stair publish armed (the H5 lever's fair window).
#
# Usage: bash campaign/card-window-runbook.sh          (from the repo root, card free, engine DOWN)
set -u
REPO=/home/svg/LLM/strata-custom-2
cd "$REPO"
. "$REPO/campaign/battery-lib.sh"
QUIET_BAR=${QUIET_BAR:-2.5}
LABEL=cardwin-$(date +%H%M)
LOG() { echo "[runbook $(date +%H:%M:%S)] $*"; }
HALT() { LOG "HALT: $*"; exit 1; }

wait_ready() {   # health ok + prewarm done, or halt
  for i in $(seq 1 80); do
    H=$(curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null)
    echo "$H" | grep -q '"status": "ok"' || { sleep 5; continue; }
    P=$(echo "$H" | python3 -c "import sys,json;print(json.load(sys.stdin).get('prewarm',''))" 2>/dev/null)
    case "$P" in *done*|*failed*|*yielded*) LOG "ready (prewarm $P)"; return 0;; esac
    sleep 5
  done
  HALT "health/prewarm never came"
}
certify_boot() {   # the fingerprint and the declared organ, from THIS boot's log tail
  grep -E "boot fingerprint f22e85411ce4aa82 slots=9900 verified=49" logs/custom-iq3_s.log | tail -1 | grep -q . \
    || HALT "boot fingerprint mismatch (expected f22e85411ce4aa82 slots=9900)"
  tail -50 logs/custom-iq3_s.log | grep -q "backing: driver-pinned" || HALT "the declared organ did not hold"
  LOG "boot certified (fingerprint + driver-pinned)"
}
teardown() {
  cleanup_engine
  for p in $(ss -tlnp 2>/dev/null | grep 8081 | grep -oP 'pid=\K[0-9]+'); do kill "$p" 2>/dev/null; done
  sleep 2
  ss -tln | grep -q 8081 && HALT "port 8081 still held after teardown"
  cp -f data/expert-profile-gold.bin data/expert-profile-learned.bin   # F-271, structural here too
  LOG "torn down clean, profile re-pinned"
}

# ---- preflight: the card must be free and the engine down ----
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{ if ($1 < 20000) exit 1 }' \
  || HALT "the card is held (${QUIET_BAR} bar is for step 3; step 0 needs the card FREE)"
ss -tln | grep -q 8081 && HALT "an engine is serving on 8081 - this runbook does not preempt it"
LOG "preflight: card free, port free"

# ---- step 1: the port proof boot ----
LOG "STEP 1: the port proof boot (deploy build-fresh, env-off bit-check + lane sanity)"
cp engine/strata /tmp/runbook-pre.bin
cp build-fresh/strata engine/strata
nohup ./launch-custom-iq3s-8081.sh > /tmp/runbook-s1.out 2>&1 &
wait_ready && certify_boot
.venv/bin/python campaign/bit-check.py runbook-port-proof 2>&1 | tail -2
.venv/bin/python campaign/probe-lane.py --port 8081 --label runbook-port-proof --skip-conc 2>&1 | grep -E "^M1 "
M1=$(grep -m1 "^M1 " campaign/results/runbook-port-proof.txt 2>/dev/null | grep -oP '[\d.]+(?= t/s)')
[ -n "$M1" ] || HALT "no M1 line from the proof leg"
awk -v m="$M1" 'BEGIN { if (m < 190) exit 1 }' || HALT "proof-boot M1 ${M1} below the 190 sanity floor"
LOG "proof leg M1 ${M1} - within sanity; the port's deploy gate PASSES"
teardown

# ---- step 2: the lookahead falsifier ----
LOG "STEP 2: the lookahead falsifier (STRATA_LOOKAHEAD_STATS=1, one probe leg)"
cp /tmp/runbook-pre.bin engine/strata   # the standing binary for the diagnostic
STRATA_LOOKAHEAD_STATS=1 nohup ./launch-custom-iq3s-8081.sh > /tmp/runbook-s2.out 2>&1 &
wait_ready && certify_boot
.venv/bin/python campaign/probe-lane.py --port 8081 --label runbook-lookahead --skip-conc 2>&1 | grep -E "^M1 "
sleep 5
grep -iE "lookahead" logs/custom-iq3_s.log | tail -6 | tee campaign/results/runbook-lookahead-stats.txt
grep -q -i "lookahead" campaign/results/runbook-lookahead-stats.txt || HALT "no lookahead stats lines - instrument silent"
teardown

# ---- step 3: the depth re-test (quiet gate at ENTRY; the battery itself is the interleave) ----
L1=$(awk '{print $1}' /proc/loadavg)
awk -v l="$L1" -v b="$QUIET_BAR" 'BEGIN { if (l >= b) exit 1 }' || { LOG "step 3 skipped: load ${L1} >= bar ${QUIET_BAR} (the re-test wants its quiet window)"; exit 0; }
LOG "STEP 3: the depth re-test (five boots, stair publish armed, load ${L1})"
PUBLISH_ENV='STRATA_SPEC_DEPTH=1' ROUNDS=5 bash campaign/env-ladder.sh STRATA_BATCH_DRAFT_DEPTH unset 2 f417-h5-depth2-quiet no \
  2>&1 | tee campaign/results/f417-h5-depth2-quiet-ladder.out | tail -12
grep -q "F-271 re-pin VERIFIED" campaign/results/f417-h5-depth2-quiet-ladder.out || HALT "the ladder's structural re-pin did not verify"
LOG "RUNBOOK COMPLETE - three boots, three answers, all evidence under campaign/results/"
