#!/bin/bash
# F-453 chain 2 (run once, 2026-11-25): after the gdn_ab env ladder - relink the full candidate (gdn_ab preload +
# drafter step cap + per-round telemetry, all default off), prove parity on that exact binary, deploy it with the
# champion alongside, run the composed env ladder (STRATA_F453_COMPOSE 0 vs 1), a 70q MMLU gate on the arm, and one
# armed telemetry boot.  The champion is restored and the engine stopped on EVERY exit.  No promotion is decided here.
set -u
REPO=/home/svg/LLM/strata-custom-2; cd "$REPO"; O=campaign/results/f453-2026-11-25
. campaign/battery-lib.sh
log(){ echo "[$(date +%T)] $*" | tee -a $O/chain2.out; }
restore(){ cp engine/strata.pre-f453-full engine/strata.new 2>/dev/null && mv engine/strata.new engine/strata && log "champion restored to engine/strata ($(sha256sum engine/strata | cut -c1-16))"; cleanup_engine | tail -1 | tee -a $O/chain2.out; }
trap restore EXIT
T0=$(date +%s); until grep -q 'champion restored' $O/envladder-gdnab.out 2>/dev/null; do [ $(( $(date +%s) - T0 )) -gt 12600 ] && { log "ABORT: the first ladder did not finish in 3.5 h"; exit 1; }; sleep 20; done
for i in $(seq 1 120); do pgrep -x strata >/dev/null || break; sleep 5; done; sleep 15
pgrep -x strata >/dev/null && { log "ABORT: an engine is still up"; exit 1; }
[ "$(sha256sum engine/strata | cut -c1-16)" = "$(sha256sum engine/strata.pre-f453-gdnab | cut -c1-16)" ] || { log "ABORT: engine/strata is not the champion the first ladder restored"; exit 1; }
taskset -c 16-31 ninja -C build-fresh strata gdn_parity > $O/chain2-relink.log 2>&1 || { log "ABORT: the relink failed"; exit 1; }
{ sha256sum build-fresh/strata engine/strata; for v in 7 3; do STRATA_SMALL_UNR=$v build-fresh/gdn_parity --selftest 2>&1 | grep 'gdn_ab_multi' | sed "s/^/[STRATA_SMALL_UNR=$v] /"; done; } | tee $O/candidate-full-parity.txt
grep -q 'STRATA_SMALL_UNR=7.*bitwise' $O/candidate-full-parity.txt || { log "ABORT: parity on the full candidate"; exit 1; }
cp engine/strata engine/strata.pre-f453-full && cp build-fresh/strata engine/strata.new && mv engine/strata.new engine/strata && log "full candidate deployed (gdn_ab + step cap + telemetry, default off)"
ROUNDS=2 bash campaign/env-ladder.sh STRATA_F453_COMPOSE 0 1 f453-compose yes > $O/envladder-compose.out 2>&1; log "composed ladder exit $?"
boot(){ cleanup_engine >/dev/null 2>&1; env $1 nohup ./launch-custom-iq3s-8081.sh > $O/$2-boot.log 2>&1 & for i in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 5; done; curl -s -m 3 http://127.0.0.1:8081/health | grep -q '"status": "ok"' || { log "ABORT: no health ($2)"; exit 1; }; wait_prewarm; }
boot "STRATA_F453_COMPOSE=1" compose-gate; EP=$(pgrep -x strata | head -1)
log "arm gate boot: SMALL_UNR=7 in env: $(tr '\0' '\n' < /proc/$EP/environ | grep -c '^STRATA_SMALL_UNR=7$') | STEP_CAP=1 in env: $(tr '\0' '\n' < /proc/$EP/environ | grep -c '^STRATA_DRAFT_STEP_CAP=1$')"
.venv/bin/python campaign/probe-lane.py --port 8081 --label f453-compose-gate-discard 2>&1 | grep -E '^M1|^M2' | sed 's/^/[discard] /' | tee -a $O/chain2.out
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 f453-compose --suite 70 > $O/mmlu-compose-70.out 2>&1; log "70q gate on the arm exit $?: $(grep -i -E 'accuracy|floor' $O/mmlu-compose-70.out | tail -2 | tr '\n' ' ' | cut -c1-200)"
boot "STRATA_COHORT_ROUND_TELEMETRY=1" roundtel; L=$(grep -n 'engine started:' logs/custom-iq3_s.log | tail -1 | cut -d: -f1)
.venv/bin/python campaign/probe-lane.py --port 8081 --label f453-roundtel-discard 2>&1 | grep -E '^M1|^M2' | sed 's/^/[discard] /' | tee -a $O/chain2.out
for n in 1 2; do .venv/bin/python campaign/probe-lane.py --port 8081 --label f453-roundtel-leg$n 2>&1 | tee $O/roundtel-leg$n.txt | grep -E '^M1|^M2|^conc' | head -4 | sed "s/^/[leg $n] /" | tee -a $O/chain2.out; done
tail -n +$L logs/custom-iq3_s.log | grep -E 'cohorts: tel' > $O/round-telemetry.log; log "telemetry lines: $(wc -l < $O/round-telemetry.log)"
log "chain 2 done"
