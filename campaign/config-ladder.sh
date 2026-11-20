#!/bin/bash
# campaign/config-ladder.sh — the paired A/B for a CONFIG-ARGS change on ONE binary.
#
# env-ladder.sh compares two environments; this compares two engine ARG sets in the standing
# config, which is the honest test for a calibration/config-gated optimization: nothing differs
# between the boots except the args, so the pair deltas are the args' effect. The config is JSON
# the engine parses: it is mutated ONLY by a python json round-trip (the calibrate-tier.sh
# precedent, the owner law's "no textual edits of load-bearing formats"), restored by trap.
#
# Law 11: every boot pays health -> the server's warm-in -> one full DISCARDED probe leg ->
# measured legs 1..3; means over measured legs only. Law 17: load recorded before every leg.
# Bit identity is NOT expected for arms that move residency between compute sites (the CPU and
# s2 GPU kernels agree to float rounding, not to the bit - s2_expert_grouped.cu's own header):
# the gates for such arms are M3b in-band + the MMLU floor, so bits default to "no".
#
# Usage: bash campaign/config-ladder.sh <on-args> <label> [bits]
#   on-args: the arg tokens to apply on ON boots (space-separated), e.g.
#            "--adapt-every 1 --adapt-decay 0.97 --adapt-swaps 160"
#   Every flag named in on-args is first REMOVED (with its value) from the standing args, then
#   the on-args are appended; OFF boots run the pristine config byte-for-byte.
# ROUNDS (default 3) sets boots per arm - 3 = the doctrine 3v3.
set -u
REPO=/home/svg/LLM/strata-custom-2
ONARGS=${1:?on-args, e.g. "--adapt-every 1 --adapt-decay 0.97 --adapt-swaps 160"}
LABEL=${2:?label}
BITS=${3:-no}
ROUNDS=${ROUNDS:-3}
CFG=$REPO/config/custom-iq3_s-8081.json
cd "$REPO"
. "$REPO/campaign/battery-lib.sh"

cp "$CFG" /tmp/config-ladder-cfg.bak
restore() { cp /tmp/config-ladder-cfg.bak "$CFG" 2>/dev/null; }
trap restore EXIT

# F-271: pin one profile for both arms; the post-battery re-pin from gold stays owed.
cp -f data/expert-profile-gold.bin data/expert-profile-learned.bin

set_config_args() {   # $1 = "" (restore pristine) or the on-args string
  python3 - "$CFG" "$1" << 'PY'
import json, shlex, sys
path, on = sys.argv[1], sys.argv[2]
c = json.load(open(path))
a = c['args']
if on:
    toks = shlex.split(on)
    flags = {t for t in toks if t.startswith('--')}
    a = [t for i, t in enumerate(a) if not (t in flags or (i > 0 and a[i-1] in flags))]
    a += toks
else:
    a = json.load(open('/tmp/config-ladder-cfg.bak'))['args']
c['args'] = a
json.dump(c, open(path, 'w'), indent=2)
PY
}

boot_and_probe() {   # $1 = on|off, $2 = tag
  [ "$1" = on ] && set_config_args "$ONARGS" || set_config_args ""
  nohup ./launch-custom-iq3s-8081.sh > /tmp/cfgl-$2.out 2>&1 &
  for i in $(seq 1 100); do
    curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break
    sleep 5
  done
  wait_prewarm
  # the arm's own banner: the engine's REAL cmdline, read from the process (F-231: a deploy is
  # verified by what runs, not by what was asked for)
  local ep; ep=$(pgrep -x strata | head -1)
  echo "[$2] adapt args: $(tr '\0' ' ' < /proc/$ep/cmdline | grep -o '\-\-adapt-[a-z]* [0-9.]*' | tr '\n' ' ')"
  uptime | awk -F'average:' '{print "[warm-discard] load:"$2}'
  .venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL-$2-warm" 2>&1 \
    | tee "campaign/results/$LABEL-$2-warm.txt" | grep -E "^M1|^M2" | sed 's/^/[warm-discard] /'
  for l in 1 2 3; do
    uptime | awk -F'average:' '{print "['"$LABEL-$2-$l"'] load:"$2}'
    .venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL-$2-$l" 2>&1 \
      | tee "campaign/results/$LABEL-$2-$l.txt" | grep -E "^M1|^M2"
  done
}

for round in $(seq 1 "$ROUNDS"); do
  echo "=== round $round: OFF (pristine config) ==="; cleanup_engine; boot_and_probe off "off$round"
  [ "$round" = 1 ] && [ "$BITS" = yes ] && .venv/bin/python campaign/bit-check.py "$LABEL-off"
  echo "=== round $round: ON ($ONARGS) ==="; cleanup_engine; boot_and_probe on "on$round"
  [ "$round" = 1 ] && [ "$BITS" = yes ] && .venv/bin/python campaign/bit-check.py "$LABEL-on"
done

cleanup_engine
restore

.venv/bin/python - "$LABEL" "$ROUNDS" << 'PY'
import re, statistics as st, sys
label, rounds = sys.argv[1], int(sys.argv[2])
def lane(tag, l, pat):
    t = open(f'campaign/results/{label}-{tag}-{l}.txt').read()
    return float(re.search(pat, t).group(1))
def boot(tag):
    return {k: st.mean([lane(tag, l, p) for l in (1, 2, 3)])
            for k, p in (('M1', r'M1 single-stream : ([\d.]+)'), ('pair', r'M2 pair          : ([\d.]+)'), ('conc', r'M2 6-conc        : ([\d.]+)'))}
offs = [boot(f'off{r}') for r in range(1, rounds + 1)]
ons  = [boot(f'on{r}')  for r in range(1, rounds + 1)]
for r in range(rounds):
    print(f"off{r+1} {offs[r]}\non{r+1}  {ons[r]}")
for lane in ('M1', 'pair', 'conc'):
    ds = [100 * (o[lane] - f[lane]) / f[lane] for f, o in zip(offs, ons)]
    print(f"{lane}: pairs {' / '.join(f'{d:+.1f}%' for d in ds)}  mean {st.mean(ds):+.1f}%")
PY
echo "CONFIG LADDER DONE — config restored from /tmp/config-ladder-cfg.bak, engine DOWN."
echo "OWED: the F-271 re-pin (cp data/expert-profile-gold.bin data/expert-profile-learned.bin) and, on adoption, the MMLU gate."
