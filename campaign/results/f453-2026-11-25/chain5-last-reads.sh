#!/bin/bash
# F-453 chain 5 (run once, 2026-11-25 07:40): the last reads on the final candidate (1db35c0350ba40c6: the champion's
# kernels + the drafter step cap + the one-pass sampler + the per-round telemetry with the pool's phase split, every
# piece default off; gdn_parity bitwise).
#   1. the one-pass sampler's benchmark on the quiet card (before any boot),
#   2. a smoke leg (default env),
#   3. a telemetry boot with two measured legs -> the pool's gate/up, quantize and down phases per round,
#   4. a node-level capture with the step cap armed -> steps per window and the wall between windows, against chain 4's
#      uncaptured-arm baseline (f453g-off, same day).
# Restores the champion and stops the engine on EVERY exit.  Decides nothing.
set -u
REPO=/home/svg/LLM/strata-custom-2; cd "$REPO"; O=campaign/results/f453-2026-11-25
. campaign/battery-lib.sh
PROF=${STRATA_PROFILE_HOME:-/mnt/DATA/LLM/Models/GGUF/strata/Custom-data/profiles}
CHAMP=3865e27a5ac1a170; CAND=1db35c0350ba40c6
log(){ echo "[$(date +%T)] $*" | tee -a $O/chain5.out; }
restore(){ cp engine/strata.pre-f453-gdnab engine/strata.new && mv engine/strata.new engine/strata && log "champion restored to engine/strata ($(sha256sum engine/strata | cut -c1-16))"; cleanup_engine | tail -1 | tee -a $O/chain5.out; }
trap restore EXIT
cleanup_engine >/dev/null 2>&1
[ "$(sha256sum build-fresh/strata | cut -c1-16)" = "$CAND" ] || { log "ABORT: build-fresh/strata is not the final candidate"; exit 1; }
[ -s $O/sampler-fastk-bench.txt ] || build-fresh/sampler_fastk_parity --bench > $O/sampler-fastk-bench.txt 2>&1; log "1. sampler benchmark (quiet card): $(grep 'bench:' $O/sampler-fastk-bench.txt | sed -E 's/ +/ /g' | tr '\n' ';')"
cp build-fresh/strata engine/strata; log "final candidate deployed ($(sha256sum engine/strata | cut -c1-16)); champion alongside"
bash campaign/smoke-candidate.sh f453-final-off > $O/smoke-final-off.txt 2>&1; log "2. smoke (default env) exit $?: $(grep -E 'SMOKE (PASS|FAIL)|concurrent lane' $O/smoke-final-off.txt | tr '\n' ' ' | cut -c1-140)"
grep -q 'SMOKE PASS' $O/smoke-final-off.txt || { log "ABORT: the smoke leg failed"; exit 1; }
bash campaign/smoke-candidate.sh STRATA_COHORT_ROUND_TELEMETRY=1 f453-final-tel > $O/smoke-final-tel.txt 2>&1; log "3a. telemetry smoke exit $?: $(grep -E 'SMOKE (PASS|FAIL)' $O/smoke-final-tel.txt | tr '\n' ' ')"
grep -q 'SMOKE PASS' $O/smoke-final-tel.txt || { log "ABORT: the telemetry smoke leg failed"; exit 1; }
cleanup_engine >/dev/null 2>&1
PREV=$(grep -n 'engine started:' logs/custom-iq3_s.log | tail -1 | cut -d: -f1); PREV=${PREV:-0}
STRATA_COHORT_ROUND_TELEMETRY=1 nohup ./launch-custom-iq3s-8081.sh > $O/roundtel2-boot.log 2>&1 &
for i in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 5; done
curl -s -m 3 http://127.0.0.1:8081/health | grep -q '"status": "ok"' || { log "ABORT: no health (telemetry boot)"; exit 1; }
wait_prewarm; T0=$(tail -n +"$((PREV + 1))" logs/custom-iq3_s.log | grep -n 'engine started:' | head -1 | cut -d: -f1); T0=$(( ${T0:-1} + PREV ))
.venv/bin/python campaign/probe-lane.py --port 8081 --label f453-roundtel2-discard > $O/roundtel2-discard.txt 2>&1
for n in 1 2; do .venv/bin/python campaign/probe-lane.py --port 8081 --label f453-roundtel2-leg$n > $O/roundtel2-leg$n.txt 2>&1; log "3b. telemetry leg $n: $(grep -E '^M1|^M2 6-conc' $O/roundtel2-leg$n.txt | sed -E 's/ \(.*//' | tr '\n' ';' | cut -c1-120)"; done
tail -n +"$T0" logs/custom-iq3_s.log | grep 'cohorts: tel' > $O/round-telemetry-phases.txt; log "3b. telemetry lines: $(wc -l < $O/round-telemetry-phases.txt)"
cleanup_engine >/dev/null 2>&1
STRATA_ANATOMY_GRAPH_TRACE=node STRATA_DRAFT_STEP_CAP=1 bash campaign/nsys-serve-anatomy.sh f453g-cap m1 1 > $O/capture-stepcap.out 2>&1; log "4. capture with the step cap armed exit $?"
F=$(ls -t $PROF/f453g-cap-*/anatomy.sqlite 2>/dev/null | head -1); B=$(ls -t $PROF/f453g-off-*/anatomy.sqlite 2>/dev/null | head -1)
{ echo "=== baseline: step cap off ($B)"; [ -n "$B" ] && .venv/bin/python $O/census-scripts/f453-stepsper-file.py "$B"; echo "=== step cap on ($F)"; [ -n "$F" ] && .venv/bin/python $O/census-scripts/f453-stepsper-file.py "$F"; } | tee $O/stepcap-mechanism.txt
log "chain 5 done"
