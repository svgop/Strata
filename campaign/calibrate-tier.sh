#!/bin/bash
# campaign/calibrate-tier.sh — THE CALIBRATION PROTOCOL (F-391, Calibration v2 P0).
# "Calibrated as soon as prewarmed, reliably and consistently, and a better calibration saves itself."
#
# It works the way the upstream's own instruments work (tools/calibrate.py, replay_cache_policy.py),
# not by sampling live decode: it CAPTURES a routing trace from a fixed greedy corpus, then REPLAYS
# that trace through the adaptive tier offline.  Same trace + same profile = the same hit share, so
# the comparison is deterministic - the boot draw cannot decide the champion.
#
#   campaign/calibrate-tier.sh [--corpus N] [--tokens 600] [--bar 768] [--margin 0.5]
#   -> data/calibration/trace.bin, report.txt, champion.json; the best profile becomes
#      data/expert-profile.bin (atomic) only when it beats the incumbent by --margin pp.
#
# The knob sweep reports the best --adapt-every/--adapt-decay/--adapt-swaps for THIS box's trace;
# applying a knob set is an operator edit of the config (one variable per leg, campaign law 5).
set -u
. "$(dirname "$0")/battery-lib.sh"
REPO=${REPO:-/home/svg/LLM/strata-custom-2}
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
CFG=$REPO/config/custom-iq3_s-8081.json
KEY=3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8
CAL=$REPO/data/calibration; mkdir -p "$CAL"
TRACE=$CAL/trace.bin
CORPUS=${CORPUS:-3}; TOKENS=${TOKENS:-600}; BAR=${BAR:-768}; MARGIN=${MARGIN:-0.5}
OUT=$CAL/report-$(date +%H%M%S).txt
echo "CALIBRATION $(date) corpus=$CORPUS tokens=$TOKENS bar=${BAR}MiB margin=${MARGIN}pp" | tee "$OUT"

# 1. the headroom law: a calibration on a GPU at the free-VRAM cliff measures the cliff, not the tier
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "headroom: ${free_mib} MiB free (bar ${BAR})" | tee -a "$OUT"
[ "${free_mib:-0}" -ge "$BAR" ] || { echo "REFUSED: free VRAM below the headroom bar - close GPU programs first (the upstream cliff is 0 MiB at ~7x decode)" | tee -a "$OUT"; exit 1; }

# 2. capture: the standing recipe + --dump-routing on a fixed greedy corpus (config swap, restored by trap)
cp "$CFG" /tmp/calibrate-cfg.bak
restore() { cp /tmp/calibrate-cfg.bak "$CFG" 2>/dev/null; }
trap restore EXIT
rm -f "$TRACE"
python3 - "$CFG" "$TRACE" << 'PY'
import json,sys
p,t=sys.argv[1],sys.argv[2]; c=json.load(open(p)); a=c['args']
if '--dump-routing' not in a: a += ['--dump-routing', t]
json.dump(c,open(p,'w'),indent=2)
PY
cleanup_engine
( cd "$REPO" && nohup bash "$LAUNCH" > "$CAL/capture.out" 2>&1 < /dev/null & )
wait_health || { echo "FATAL: the boot did not answer /health" | tee -a "$OUT"; exit 1; }
wait_prewarm
# the three fixed prompts (M1, a code-shaped one, a short chat) at temperature 0: the corpus is the constant
python3 - "$KEY" "$TOKENS" "$CORPUS" << 'PY' | tee -a "$OUT"
import json,sys,time,urllib.request
key,tok,n=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
P=["Write a comprehensive technical analysis of modern GPU memory hierarchies and their impact on LLM inference performance. Cover HBM bandwidth, L2 cache behavior, PCIe transfer overhead, and pinned memory tradeoffs, with concrete numbers and examples throughout.",
   "Write a Python function that parses a JSONL file of inference requests, groups them by model, computes p50 and p99 latency with numpy, and returns a sorted summary. Include error handling and a docstring.",
   "Explain, in concrete terms, why a mixture-of-experts model offloads experts to CPU memory and how the GPU decides which experts to keep resident. Keep it under 300 words."]
tot=0
for i,p in enumerate(P[:n],1):
    body=json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":tok,"temperature":0}).encode()
    req=urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",data=body,headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=600) as r: d=json.load(r)
        dt=time.time()-t0; ct=d["usage"]["completion_tokens"]; tot+=ct
        print(f"  prompt {i}: {ct} tok in {dt:.2f}s = {ct/dt:.1f} t/s")
    except Exception as e:
        print(f"  prompt {i}: FAILED {e}")
print(f"  corpus tokens generated: {tot}")
PY
# 3. exit: the flush writes the trace; the engine saves its learned profile (the challenger)
cleanup_engine; sleep 4
restore; trap - EXIT
[ -s "$TRACE" ] || { echo "FATAL: the trace is empty (--dump-routing did not reach the engine)" | tee -a "$OUT"; exit 1; }
echo "trace: $(stat -c %s "$TRACE") bytes" | tee -a "$OUT"

# 4. replay: the SAME trace, offline, for the incumbent seed and the challenger; then a knob sweep
SEED=$REPO/data/expert-profile.bin
CHAL=$REPO/data/expert-profile-learned.bin
SLOTS=9900
# the tool prints "policy every E, swaps S, decay D: share 0.XXXX, swaps N ..." - the share is the answer
share_of() { sed -n 's/.*share \([0-9.][0-9.]*\).*/\1/p' | tail -1; }
replay() { .venv/bin/python "$REPO/tools/replay_cache_policy.py" "$TRACE" "$1" "$SLOTS" $2 2>/dev/null | share_of; }
echo "--- profile champion/challenger (hit share on the trace; deterministic) ---" | tee -a "$OUT"
seed_hit=$(replay "$SEED" ""); echo "  incumbent seed: $seed_hit" | tee -a "$OUT"
if [ -s "$CHAL" ]; then chal_hit=$(replay "$CHAL" ""); echo "  challenger learned: $chal_hit" | tee -a "$OUT"; else chal_hit=""; fi
echo "--- knob sweep (current: every=4 decay=0.85 swaps=96) ---" | tee -a "$OUT"
best=""; bestv=-1
for e in 1 2 4; do for d in 0.7 0.85 0.97; do for s in 32 96 160; do
  v=$(replay "$SEED" "--every $e --decay $d --swaps $s")
  echo "  every=$e decay=$d swaps=$s -> $v" >> "$OUT"
  val=${v:-0}
  if python3 -c "import sys; sys.exit(0 if float('$val') > float('$bestv') else 1)" 2>/dev/null; then best="every=$e decay=$d swaps=$s"; bestv=$val; fi
done; done; done
echo "BEST KNOBS: $best (share $bestv)" | tee -a "$OUT"
# 5. champion/challenger on the PROFILE: promote the challenger only when it beats the seed by the margin
python3 - "$seed_hit" "${chal_hit:-}" "$MARGIN" "$CHAL" "$SEED" "$CAL" "$OUT" << 'PY'
import sys,os,json,time,shutil
seed,chal,margin,chalp,seedp,cal,out=sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7]
def pct(s):
    try: return float(s.strip())
    except Exception: return None
a=pct(seed); b=pct(chal) if chal else None
meta=os.path.join(cal,'champion.json')
rec={'ts':time.strftime('%Y-%m-%dT%H:%M:%S'),'seed_hit':a,'challenger_hit':b,'margin_pp':margin}
if b is not None and a is not None and b >= a + margin / 100.0 and chalp.endswith('.bin') and os.path.getsize(chalp) > 0:
    tmp=seedp+'.new'; shutil.copyfile(chalp,tmp); os.replace(tmp,seedp)   # atomic promote
    rec['promoted']=True; rec['note']='challenger beat the seed by >= margin; the seed is now the challenger'
    print(f"PROMOTED: challenger {b}% >= seed {a}% + {margin}pp -> {seedp}", flush=True)
else:
    rec['promoted']=False; rec['note']='challenger did not beat the seed by the margin (or is absent); the incumbent stands'
    print(f"KEPT: seed {a}% / challenger {b if b is not None else 'absent'}% -> the incumbent stands", flush=True)
json.dump(rec,open(meta,'w'),indent=2)
open(out,'a').write(json.dumps(rec)+"\n")
PY
echo "CALIBRATION done $(date)" | tee -a "$OUT"
