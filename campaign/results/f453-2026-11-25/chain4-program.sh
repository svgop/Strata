#!/bin/bash
# F-453 chain 4 (run once, 2026-11-25 06:20): the whole remaining program on the fixed full candidate
# (87e4d95dcd365450: gdn_ab preload + drafter step cap + one-pass sampler + per-round telemetry, every piece default off;
# smoke legs passed: default env, telemetry armed, composed arm), in order of what each answer is worth -
#   A. ATTRIBUTION (RECOVER a): the champion with F-452's arm off (STRATA_VERIFY_DEVICE_LAUNCH=0) over the first 534
#      questions of the 1,500-suite, paired against the adopted state's rows.
#   B. MECHANISM: the gdn_ab preload at node level, arm off against on, same hour (ladder 1 read +5.5% M1 against a
#      priced ceiling of ~+2%).
#   C. The COMPOSED env ladder (STRATA_F453_COMPOSE 0 vs 1 = gdn_ab preload + drafter step cap), two rounds.
#   D. A 70-question gate on the composed arm (the step cap touches which rows are verified).
#   E. A warm-leg telemetry boot (the per-round split of the concurrent lane, measured legs only).
# The champion is restored and the engine stopped on EVERY exit.  Nothing is decided here.
set -u
REPO=/home/svg/LLM/strata-custom-2; cd "$REPO"; O=campaign/results/f453-2026-11-25
. campaign/battery-lib.sh
PROF=${STRATA_PROFILE_HOME:-/mnt/DATA/LLM/Models/GGUF/strata/Custom-data/profiles}
CHAMP=3865e27a5ac1a170; CAND=87e4d95dcd365450
log(){ echo "[$(date +%T)] $*" | tee -a $O/chain4.out; }
restore(){ cp engine/strata.pre-f453-gdnab engine/strata.new && mv engine/strata.new engine/strata && log "champion restored to engine/strata ($(sha256sum engine/strata | cut -c1-16))"; cleanup_engine | tail -1 | tee -a $O/chain4.out; }
trap restore EXIT
cleanup_engine >/dev/null 2>&1
[ "$(sha256sum engine/strata | cut -c1-16)" = "$CHAMP" ] || { log "ABORT: engine/strata is not the champion"; exit 1; }
[ "$(sha256sum build-fresh/strata | cut -c1-16)" = "$CAND" ] || { log "ABORT: build-fresh/strata is not the smoked candidate"; exit 1; }
boot(){ cleanup_engine >/dev/null 2>&1; L0=$(grep -c 'engine started:' logs/custom-iq3_s.log); env $1 nohup ./launch-custom-iq3s-8081.sh > $O/$2-boot.log 2>&1 & for i in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 5; done; curl -s -m 3 http://127.0.0.1:8081/health | grep -q '"status": "ok"' || { log "ABORT: no health ($2)"; exit 1; }; wait_prewarm; echo "boot $2: binary $(sha256sum engine/strata | cut -c1-16) log-from-line $L0" > $O/$2.anchor; }
leglines(){ grep -E '^M1|^M2|conc' "$1" | cut -c1-120; }

# ---- A. attribution
boot "STRATA_VERIFY_DEVICE_LAUNCH=0" attr; EP=$(pgrep -x strata | head -1); A0=$(cut -d' ' -f4 $O/attr.anchor)
log "A. control boot: binary $(sha256sum engine/strata | cut -c1-16) | device launch in the engine's env: $(tr '\0' '\n' < /proc/$EP/environ | grep '^STRATA_VERIFY_DEVICE_LAUNCH=' || echo unset)"
.venv/bin/python campaign/probe-lane.py --port 8081 --label f453-attr-discard > $O/attr-discard.txt 2>&1; leglines $O/attr-discard.txt | sed 's/^/[discard] /' | tee -a $O/chain4.out
grep -q 'error' $O/attr-boot.log && { log "ABORT: a request of the discard leg ended in an error"; exit 1; }
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 f452-control --suite 1500 --first 534 > $O/mmlu-control-534.out 2>&1; log "A. control gate exit $?: $(grep -E '^=== MMLU' $O/mmlu-control-534.out | cut -c1-150)"
.venv/bin/python $O/census-scripts/f453-paired-gate.py campaign/results/mmlu-f452-control.json campaign/results/mmlu-f453-stand.json | tee $O/attribution-534.txt

# ---- B. mechanism
cleanup_engine >/dev/null 2>&1; cp build-fresh/strata engine/strata; log "B. full candidate deployed ($(sha256sum engine/strata | cut -c1-16))"
for arm in off on; do v=3; [ "$arm" = on ] && v=7
  STRATA_ANATOMY_GRAPH_TRACE=node STRATA_SMALL_UNR=$v bash campaign/nsys-serve-anatomy.sh f453g-$arm m1 1 > $O/capture-gdnab-$arm.out 2>&1; log "B. capture $arm (STRATA_SMALL_UNR=$v, node-level) exit $?"; done
{ for arm in off on; do f=$(ls -t $PROF/f453g-$arm-*/anatomy.sqlite 2>/dev/null | head -1); echo "=== $arm: $f"; [ -n "$f" ] && .venv/bin/python $O/census-scripts/f453-gdnab-kernels.py "$f"; done; } | tee $O/gdnab-kernel-mechanism.txt

# ---- C. composed ladder
ROUNDS=2 bash campaign/env-ladder.sh STRATA_F453_COMPOSE 0 1 f453-compose yes > $O/envladder-compose.out 2>&1; log "C. composed ladder exit $?"

# ---- D. 70q gate on the arm
boot "STRATA_F453_COMPOSE=1" armgate; EP=$(pgrep -x strata | head -1)
log "D. arm boot: SMALL_UNR=7 in env: $(tr '\0' '\n' < /proc/$EP/environ | grep -c '^STRATA_SMALL_UNR=7$') | STEP_CAP=1 in env: $(tr '\0' '\n' < /proc/$EP/environ | grep -c '^STRATA_DRAFT_STEP_CAP=1$')"
.venv/bin/python campaign/probe-lane.py --port 8081 --label f453-armgate-discard > $O/armgate-discard.txt 2>&1; leglines $O/armgate-discard.txt | sed 's/^/[discard] /' | tee -a $O/chain4.out
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 f453-compose --suite 70 > $O/mmlu-compose-70.out 2>&1; log "D. 70q gate on the arm exit $?: $(grep -E '^=== MMLU' $O/mmlu-compose-70.out | cut -c1-150)"

# ---- E. warm-leg telemetry
boot "STRATA_COHORT_ROUND_TELEMETRY=1" roundtel; T0=$(cut -d' ' -f6 $O/roundtel.anchor)
.venv/bin/python campaign/probe-lane.py --port 8081 --label f453-roundtel-discard > $O/roundtel-discard.txt 2>&1
for n in 1 2; do .venv/bin/python campaign/probe-lane.py --port 8081 --label f453-roundtel-leg$n > $O/roundtel-leg$n.txt 2>&1; log "E. telemetry leg $n: $(leglines $O/roundtel-leg$n.txt | tr '\n' ';' | cut -c1-150)"; done
tail -n +"$T0" logs/custom-iq3_s.log | grep 'cohorts: tel' > $O/round-telemetry-warm.txt; log "E. telemetry lines (both legs and the discard): $(wc -l < $O/round-telemetry-warm.txt)"
log "chain 4 done"
