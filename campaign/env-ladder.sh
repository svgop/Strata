#!/bin/bash
# campaign/env-ladder.sh — the paired A/B for an ENV-GATED change on ONE binary.
#
# The binary ladder (campaign/binary-ladder.sh) compares two binaries. This compares two
# environments on the same binary, which is the honest test for a flag-gated optimization:
# nothing differs between the boots except the flag, so the pair deltas are the flag's effect.
#
# Law 11 (warmup is part of the bench): every boot pays health -> prewarm2 -> one full DISCARDED
# probe leg -> measured legs 1..3. Means are over the measured legs only.
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

cp engine/strata /tmp/env-ladder-pre-run.bin
echo "lane binary saved: $(md5sum engine/strata | cut -c1-8)"

stop_server() {
  local p; p=$(ps aux | grep "serve/server\.py --engine" | grep -v grep | awk '{print $2}' | head -1)
  [ -n "$p" ] && kill "$p"
  for i in $(seq 1 30); do
    [ "$(ps aux | grep -E 'serve/server\.py --engine|engine/strata --serve' | grep -v grep | wc -l)" -eq 0 ] && break
    sleep 2
  done
}

boot_and_probe() {   # $1 = on|off, $2 = tag
  if [ "$1" = on ]; then
    env "$VAR=$ON" nohup ./launch-custom-iq3s-8081.sh > /tmp/envl-$2.out 2>&1 &
  elif [ "$OFF" = unset ]; then
    nohup ./launch-custom-iq3s-8081.sh > /tmp/envl-$2.out 2>&1 &
  else
    env "$VAR=$OFF" nohup ./launch-custom-iq3s-8081.sh > /tmp/envl-$2.out 2>&1 &
  fi
  for i in $(seq 1 100); do
    curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break
    sleep 5
  done
  bash campaign/prewarm2.sh > /dev/null 2>&1
  # the arm's own banner/decline line, from the engine's environment as the process sees it
  local ep; ep=$(pgrep -x strata | head -1)
  echo "[$2] env $VAR=$(tr '\0' '\n' < /proc/$ep/environ | sed -n "s/^$VAR=//p" | head -1)"
  grep -h "STRATA_SHARED_HOIST" /tmp/envl-$2.out custom-iq3_s.log 2>/dev/null | tail -2
  .venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL-$2-warm" 2>&1 \
    | tee "campaign/results/$LABEL-$2-warm.txt" | grep -E "^M1|^M2" | sed 's/^/[warm-discard] /'
  for l in 1 2 3; do
    .venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL-$2-$l" 2>&1 \
      | tee "campaign/results/$LABEL-$2-$l.txt" | grep -E "^M1|^M2"
  done
}

for round in 1 2; do
  echo "=== round $round: OFF ==="; stop_server; boot_and_probe off "off$round"
  [ "$round" = 1 ] && [ "$BITS" = yes ] && .venv/bin/python campaign/bit-check.py "$LABEL-off"
  echo "=== round $round: ON ==="; stop_server; boot_and_probe on "on$round"
  [ "$round" = 1 ] && [ "$BITS" = yes ] && .venv/bin/python campaign/bit-check.py "$LABEL-on"
done

stop_server; cp /tmp/env-ladder-pre-run.bin engine/strata; nohup ./launch-custom-iq3s-8081.sh > /tmp/envl-restore.out 2>&1 &
for i in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 5; done
bash campaign/prewarm2.sh > /dev/null 2>&1

.venv/bin/python - "$LABEL" << 'PY'
import re, statistics as st, sys
label = sys.argv[1]
def lane(tag, l, pat):
    t = open(f'campaign/results/{label}-{tag}-{l}.txt').read()
    return float(re.search(pat, t).group(1))
def boot(tag):
    return {k: st.mean([lane(tag, l, p) for l in (1, 2, 3)])
            for k, p in (('M1', r'M1 single-stream : ([\d.]+)'), ('pair', r'M2 pair          : ([\d.]+)'), ('conc', r'M2 6-conc        : ([\d.]+)'))}
o1, n1, o2, n2 = boot('off1'), boot('on1'), boot('off2'), boot('on2')
print(f"\noff1 {o1}\non1  {n1}\noff2 {o2}\non2  {n2}")
for lane in ('M1', 'pair', 'conc'):
    d1 = 100 * (n1[lane] - o1[lane]) / o1[lane]
    d2 = 100 * (n2[lane] - o2[lane]) / o2[lane]
    print(f"{lane}: pairs {d1:+.1f}% / {d2:+.1f}%  mean {(d1 + d2) / 2:+.1f}%")
PY
echo "ENV LADDER DONE — lane binary restored from /tmp/env-ladder-pre-run.bin"
