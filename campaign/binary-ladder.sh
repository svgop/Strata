#!/bin/bash
# campaign/binary-ladder.sh — THE BINARY PROMOTION GAUNTLET (user law, 2026-10-26).
#
# Process: every optimization produces a CANDIDATE binary; the incumbent CHAMPION stays on
# disk. The gauntlet runs ABAB interleaved boots (champion/candidate/champion/candidate),
# 3 probe legs each, same hour, shared conditions — and prints the promotion verdict:
#   PROMOTE  = candidate is not worse beyond noise on any lane AND better beyond noise on >=1 lane
#              (promotion also requires every lane within the -4% regression bound)
#   HOLD     = candidate regresses any lane beyond the bound, or wins nothing (champion stays)
# Usage: bash campaign/binary-ladder.sh <champion-binary> <candidate-binary> <label>
#   (server must be RUNNING on the standing recipe from the root launcher; the script swaps
#    engine/strata itself: kill -> cp -> boot -> health -> prewarm -> 3 legs; ~30-40 min)
set -u
REPO=/home/svg/LLM/strata-custom-2
CHAMP=${1:?champion binary path}
CAND=${2:?candidate binary path}
LABEL=${3:?label}
cd "$REPO"

boot_and_probe() {   # $1 = binary, $2 = tag
  cp "$1" engine/strata
  nohup ./launch-custom-iq3s-8081.sh > /tmp/ladder-$2.out 2>&1 &
  for i in $(seq 1 100); do
    curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break
    sleep 5
  done
  bash campaign/prewarm2.sh > /dev/null 2>&1
  # campaign law 11: the first full leg is WARMUP, never measured (first-touch faults on any new
  # mapping - the shadow store cost leg 1 ~15% in F-360).  It is still written, tagged warm, for audit.
  .venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL-$2-warm" 2>&1 \
    | tee "campaign/results/$LABEL-$2-warm.txt" | grep -E "^M1|^M2" | sed 's/^/[warm-discard] /'
  for l in 1 2 3; do
    .venv/bin/python campaign/probe-lane.py --port 8081 --label "$LABEL-$2-$l" 2>&1 \
      | tee "campaign/results/$LABEL-$2-$l.txt" | grep -E "^M1|^M2"
  done
}

stop_server() {
  local p; p=$(ps aux | grep "serve/server\.py --engine" | grep -v grep | awk '{print $2}' | head -1)
  [ -n "$p" ] && kill "$p"
  for i in $(seq 1 30); do
    [ "$(ps aux | grep -E 'serve/server\.py --engine|engine/strata --serve' | grep -v grep | wc -l)" -eq 0 ] && break
    sleep 2
  done
}

for round in 1 2; do
  echo "=== round $round: champion ==="; stop_server; boot_and_probe "$CHAMP" "champ$round"
  echo "=== round $round: candidate ==="; stop_server; boot_and_probe "$CAND" "cand$round"
done
stop_server; cp "$CHAMP" engine/strata; nohup ./launch-custom-iq3s-8081.sh > /tmp/ladder-restore.out 2>&1 &
for i in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && break; sleep 5; done
bash campaign/prewarm2.sh > /dev/null 2>&1

.venv/bin/python - "$LABEL" << 'PY'
import re, statistics as st, sys
label = sys.argv[1]
def lane(tag, l, pat):
    t = open(f'campaign/results/{label}-{tag}-{l}.txt').read()
    return float(re.search(pat, t).group(1))
def boot(tag):
    return {k: st.mean([lane(tag, l, p) for l in (1,2,3)])
            for k, p in (('M1', r'M1 single-stream : ([\d.]+)'), ('pair', r'M2 pair          : ([\d.]+)'), ('conc', r'M2 6-conc        : ([\d.]+)'))}
ch1, cd1, ch2, cd2 = boot('champ1'), boot('cand1'), boot('champ2'), boot('cand2')
print(f"\nchamp1 {ch1}\ncand1  {cd1}\nchamp2 {ch2}\ncand2  {cd2}")
for lane in ('M1','pair','conc'):
    d1 = 100*(cd1[lane]-ch1[lane])/ch1[lane]; d2 = 100*(cd2[lane]-ch2[lane])/ch2[lane]
    m = (d1+d2)/2
    print(f"{lane}: pairs {d1:+.1f}% / {d2:+.1f}%  mean {m:+.1f}%")
PY
echo "GAUNTLET DONE — champion restored on engine/strata (the python table above is the verdict input)"
