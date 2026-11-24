#!/bin/bash
# campaign/env-ladder.sh — the paired A/B for an ENV-GATED change on ONE binary.
#
# The binary ladder (campaign/binary-ladder.sh) compares two binaries. This compares two
# environments on the same binary, which is the honest test for a flag-gated optimization:
# nothing differs between the boots except the flag, so the pair deltas are the flag's effect.
#
# Law 11 (warmup is part of the bench): every boot pays health -> the server's own warm-in (the
# config's "prewarm" section) -> one full DISCARDED probe leg -> measured legs 1..3. Means are over
# the measured legs only.
# Optional bit identity: round 1 also runs campaign/bit-check.py on both arms; the shas must match
# (the change is a recompute of the same math, so a sha difference is a correctness failure).
#
# Usage: bash campaign/env-ladder.sh <VAR> <off-value> <on-value> <label> [bits]
#   off-value "unset" -> the variable is not exported for the control boot.
set -u
REPO=/home/svg/LLM/strata-custom-2
VAR=${1:?variable}
OFF=${2:?off value (or "unset")}
ON=${3:?on value}
LABEL=${4:?label}
BITS=${5:-yes}
cd "$REPO"
. "$REPO/campaign/battery-lib.sh"

cp engine/strata /tmp/env-ladder-pre-run.bin
echo "lane binary saved: $(md5sum engine/strata | cut -c1-8)"

boot_and_probe() {   # $1 = on|off, $2 = tag
  # PUBLISH_ENV (optional): space-separated PUBLISH-ONLY vars exported to BOTH arms - telemetry that
  # cannot change behavior (STRATA_SPEC_DEPTH=1's per-depth counters, DM_WAIT_STATS) so a battery can
  # capture its mechanism data in-window.  The arm variable itself stays the only A/B difference.
  if [ "$1" = on ]; then
    env ${PUBLISH_ENV:-} "$VAR=$ON" nohup ./launch-custom-iq3s-8081.sh > /tmp/envl-$2.out 2>&1 &
  elif [ "$OFF" = unset ]; then
    env ${PUBLISH_ENV:-} nohup ./launch-custom-iq3s-8081.sh > /tmp/envl-$2.out 2>&1 &
  else
    env ${PUBLISH_ENV:-} "$VAR=$OFF" nohup ./launch-custom-iq3s-8081.sh > /tmp/envl-$2.out 2>&1 &
  fi
  for i in $(seq 1 100); do
    curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break
    sleep 5
  done
  # F-452: a boot that never answered is a FAILED battery, not four zero-token legs (the 11-24 run: PUBLISH_ENV
  # unbound under `set -u` aborted the launch line, the loop above timed out in silence, every leg read 0 tok)
  curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' || {
    echo "env-ladder: ABORT - the engine did not answer /health in 500 s (boot log: /tmp/envl-$2.out)"; exit 1; }
  wait_prewarm
  # the arm's own banner/decline line, from the engine's environment as the process sees it
  local ep; ep=$(pgrep -x strata | head -1)
  echo "[$2] env $VAR=$(tr '\0' '\n' < /proc/$ep/environ | sed -n "s/^$VAR=//p" | head -1)"
  grep -E ".{0,60}STRATA_SHARED_HOIST.{0,40}" $REPO/logs/custom-iq3_s.log 2>/dev/null | tail -2
  uptime | awk -F'average:' '{print "[warm-discard] load:"$2}'
  .venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL-$2-warm" 2>&1 \
    | tee "campaign/results/$LABEL-$2-warm.txt" | grep -E "^M1|^M2" | sed 's/^/[warm-discard] /'
  for l in 1 2 3; do
    uptime | awk -F'average:' '{print "['"$LABEL-$2-$l"'] load:"$2}'
    .venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL-$2-$l" 2>&1 \
      | tee "campaign/results/$LABEL-$2-$l.txt" | grep -E "^M1|^M2"
  done
}

for round in $(seq 1 "${ROUNDS:-2}"); do
  echo "=== round $round: OFF ==="; cleanup_engine; boot_and_probe off "off$round"
  [ "$round" = 1 ] && [ "$BITS" = yes ] && .venv/bin/python campaign/bit-check.py "$LABEL-off"
  echo "=== round $round: ON ==="; cleanup_engine; boot_and_probe on "on$round"
  [ "$round" = 1 ] && [ "$BITS" = yes ] && .venv/bin/python campaign/bit-check.py "$LABEL-on"
done

cleanup_engine; cp /tmp/env-ladder-pre-run.bin engine/strata; nohup ./launch-custom-iq3s-8081.sh > /tmp/envl-restore.out 2>&1 &
for i in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 5; done
wait_prewarm

.venv/bin/python - "$LABEL" "${ROUNDS:-2}" << 'PY'
import re, statistics as st, sys
label, rounds = sys.argv[1], int(sys.argv[2])
def lane(tag, l, pat):
    t = open(f'campaign/results/{label}-{tag}-{l}.txt').read()
    return float(re.search(pat, t).group(1))
def boot(tag):
    return {k: st.mean([lane(tag, l, p) for l in (1, 2, 3)])
            for k, p in (('M1', r'M1 single-stream : ([\d.]+)'), ('pair', r'M2 pair          : ([\d.]+)'), ('conc', r'M2 6-conc        : ([\d.]+)'),
                         ('M3b', r'M3b rate         : ([\d.]+)'))}   # the canary (law 10): the means table carries it every battery
offs = [boot(f'off{r}') for r in range(1, rounds + 1)]
ons  = [boot(f'on{r}')  for r in range(1, rounds + 1)]
for r in range(rounds):
    print(f"off{r+1} {offs[r]}\non{r+1}  {ons[r]}")
ok = True
for lane in ('M1', 'pair', 'conc'):
    ds = [100 * (o[lane] - f[lane]) / f[lane] for f, o in zip(offs, ons)]
    print(f"{lane}: pairs {' / '.join(f'{d:+.1f}%' for d in ds)}  mean {st.mean(ds):+.1f}%")
    if st.mean(ds) < -4.0: ok = False
m3 = [100 * (o['M3b'] - f['M3b']) / f['M3b'] for f, o in zip(offs, ons)]
print(f"M3b canary: OFF {st.mean([b['M3b'] for b in offs]):.1f}% vs ON {st.mean([b['M3b'] for b in ons]):.1f}% (mean {st.mean(m3):+.1f}%)")
print("BAR: no lane below -4%:", "PASS" if ok else "FAIL", "- the targeted lane's +4% gain is judged on its mean above (law 2, both legs)")
PY
echo "CONFIG/ENV LADDER DONE — config restored, engine UP on the restored binary."
# F-271 as STRUCTURE (the 2026-11-20 incident: the OWED line was printed and missed - the battery's
# last boot saved a probe-traffic learned profile, the restore boot loaded it, the tier thrashed
# 25k exchanges with 10k file reads on real traffic): the re-pin happens HERE, verified, every run.
cp -f data/expert-profile-gold.bin data/expert-profile-learned.bin
md5sum data/expert-profile-gold.bin data/expert-profile-learned.bin | awk '{print $1}' | uniq -c | grep -q "^      2" \
  && echo "F-271 re-pin VERIFIED (learned == gold)" || echo "F-271 RE-PIN FAILED - pin it by hand before the next boot"
