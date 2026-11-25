#!/bin/bash
# F-453 chain 3 (run once, 2026-11-25): after chain 2 -
#   (1) MECHANISM: the gdn_ab preload at node level, arm off against on, same hour, on the full candidate (two
#       serve-mode captures, m1 lane, STRATA_ANATOMY_GRAPH_TRACE=node).  The first ladder read +5.5% M1 against a priced
#       ceiling of about +2%: this says how much of it the kernel can explain.
#   (2) ATTRIBUTION: the quality deficit.  The champion with F-452's arm off (STRATA_VERIFY_DEVICE_LAUNCH=0) over the
#       first 534 questions of the 1,500-suite - the questions whose adopted-state rows are already in hand - paired.
# The champion is restored and the engine stopped on EVERY exit.  Nothing is decided here.
set -u
REPO=/home/svg/LLM/strata-custom-2; cd "$REPO"; O=campaign/results/f453-2026-11-25
. campaign/battery-lib.sh
PROF=${STRATA_PROFILE_HOME:-/mnt/DATA/LLM/Models/GGUF/strata/Custom-data/profiles}
log(){ echo "[$(date +%T)] $*" | tee -a $O/chain3.out; }
restore(){ [ -f engine/strata.pre-f453-gdnab ] && { cp engine/strata.pre-f453-gdnab engine/strata.new && mv engine/strata.new engine/strata && log "champion restored to engine/strata ($(sha256sum engine/strata | cut -c1-16))"; }; cleanup_engine | tail -1 | tee -a $O/chain3.out; }
trap restore EXIT
T0=$(date +%s); until grep -q 'chain 2 done' $O/chain2.out 2>/dev/null; do [ $(( $(date +%s) - T0 )) -gt 10800 ] && { log "ABORT: chain 2 did not finish in 3 h"; exit 1; }; sleep 30; done
for i in $(seq 1 120); do pgrep -x strata >/dev/null || break; sleep 5; done; sleep 15
pgrep -x strata >/dev/null && { log "ABORT: an engine is still up"; exit 1; }
[ "$(sha256sum build-fresh/strata | cut -c1-16)" = "$(head -1 $O/candidate-full-parity.txt | cut -c1-16)" ] || { log "ABORT: build-fresh/strata is not the candidate chain 2 proved"; exit 1; }
cp build-fresh/strata engine/strata; log "full candidate deployed for the mechanism captures ($(sha256sum engine/strata | cut -c1-16)); champion alongside"
for arm in off on; do
  v=3; [ "$arm" = on ] && v=7
  STRATA_ANATOMY_GRAPH_TRACE=node STRATA_SMALL_UNR=$v bash campaign/nsys-serve-anatomy.sh f453g-$arm m1 1 > $O/capture-gdnab-$arm.out 2>&1; log "capture $arm (STRATA_SMALL_UNR=$v, node-level) exit $?"
done
{ for arm in off on; do f=$(ls -t $PROF/f453g-$arm-*/anatomy.sqlite 2>/dev/null | head -1); echo "=== $arm: $f"; [ -n "$f" ] && .venv/bin/python $O/census-scripts/f453-gdnab-kernels.py "$f"; done; } | tee $O/gdnab-kernel-mechanism.txt
restore
cleanup_engine >/dev/null 2>&1
STRATA_VERIFY_DEVICE_LAUNCH=0 nohup ./launch-custom-iq3s-8081.sh > $O/attr-boot.log 2>&1 &
for i in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 5; done
curl -s -m 3 http://127.0.0.1:8081/health | grep -q '"status": "ok"' || { log "ABORT: no health (the attribution boot)"; exit 1; }
wait_prewarm; EP=$(pgrep -x strata | head -1)
log "control boot: binary $(sha256sum engine/strata | cut -c1-16) | STRATA_VERIFY_DEVICE_LAUNCH in the engine's env: $(tr '\0' '\n' < /proc/$EP/environ | grep '^STRATA_VERIFY_DEVICE_LAUNCH=' || echo unset)"
.venv/bin/python campaign/probe-lane.py --port 8081 --label f453-attr-discard 2>&1 | grep -E '^M1|^M2' | sed 's/^/[discard] /' | tee -a $O/chain3.out
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 f452-control --suite 1500 --first 534 > $O/mmlu-control-534.out 2>&1; log "control gate exit $?: $(grep -E '^=== MMLU' $O/mmlu-control-534.out | cut -c1-150)"
.venv/bin/python $O/census-scripts/f453-paired-gate.py campaign/results/mmlu-f452-control.json campaign/results/mmlu-f453-stand.json | tee $O/attribution-534.txt
log "chain 3 done"
