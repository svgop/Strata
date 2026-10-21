#!/bin/bash
# campaign/t27-calibarm.sh <adapt|minp> — the calibrator's other two measured arms, one variable each.
#   adapt: --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97 (replaces standing decay 0.85; one
#          mechanism = the expert-tier policy, three sub-args changed together as calibrate.py's unit)
#   minp:  --spec-min-p 0.7 (replaces standing 0.5; depth-only, no token changes)
# ABAB 3v3 vs the standing config, config-copies only.
set -u
MODE="${1:-}"
case "$MODE" in adapt|minp) ;; *) echo "usage: $0 <adapt|minp>"; exit 1 ;; esac
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
OUT=$REPO/campaign/results/t27-calibarm-$MODE.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T27 CALIBARM/$MODE start $(date)" > "$OUT"

python3 - "$REPO" "$MODE" <<'PYEOF'
import json, sys
repo, mode = sys.argv[1], sys.argv[2]
cfg = json.load(open(f'{repo}/custom-iq3_s-8081.json', encoding='utf-8-sig'))
a = cfg['args']
def set_arg(flag, val):
    if flag in a:
        a[a.index(flag) + 1] = val
    else:
        # keep the args list shape: flag/value appended before the final --mtp path group
        a.extend([flag, val])
if mode == 'adapt':
    set_arg('--adapt-decay', '0.97')
    set_arg('--adapt-every', '1')
    set_arg('--adapt-swaps', '80')
else:
    set_arg('--spec-min-p', '0.7')
json.dump(cfg, open(f'{repo}/custom-iq3s-calib-{mode}.json', 'w'), indent=2)
print(f'calib-{mode} config written')
PYEOF
CFGNAME="custom-iq3s-calib-$MODE.json"
python3 -c "import json;c=json.load(open('$REPO/$CFGNAME'));a=c['args'];import sys;sys.exit(0 if ('$MODE'=='adapt' and a[a.index('--adapt-decay')+1]=='0.97' and a[a.index('--adapt-every')+1]=='1') or ('$MODE'=='minp' and a[a.index('--spec-min-p')+1]=='0.7') else 1)" \
  && echo "arm config verified" >> "$OUT" || { echo "ARM CONFIG BUILD FAILED" >> "$OUT"; exit 1; }
sed "s#custom-iq3_s-8081.json#$CFGNAME#" "$REPO/launch-custom-iq3s-8081.sh" > "$REPO/launch-calib-$MODE.sh"   # ROOT: F-249 lesson
chmod +x "$REPO/launch-calib-$MODE.sh"

port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }
cleanup_engine() {
  local p pid c
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *strata*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
  sleep 2
}
boot() {  # $1 launcher $2 label
  cleanup_engine
  [ -n "$(port_pid)" ] && { echo "=== $2 ABORT port busy ===" >> "$OUT"; return 1; }
  ( cd "$REPO" && nohup "$1" >"$RES/t27-$2.out" 2>&1 & )
  local ok=0 t
  for t in $(seq 1 90); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && { ok=1; break; }
    sleep 3
  done
  echo "=== LEG $2 health=$ok $(date +%H:%M:%S) ===" >> "$OUT"
  [ "$ok" = 1 ]
}
leg() {  # $1 launcher $2 label
  boot "$1" "$2" || return
  ( cd "$REPO" && python3 campaign/probe-lane.py --port "$PORT" --label "t27-$2" --short-n 6 >> "$OUT" 2>&1 )
  ( cd "$REPO" && .venv/bin/python campaign/probe-metrics.py --port "$PORT" --label "t27-$2" 2>&1 | tee -a "$OUT" )
  cleanup_engine
}
CTL=$REPO/launch-custom-iq3s-8081.sh
ARM=$REPO/launch-calib-$MODE.sh
for r in 1 2 3; do
  leg "$ARM" "$MODE-arm-r$r"
  leg "$CTL" "$MODE-ctl-r$r"
done
echo "T27 CALIBARM/$MODE end $(date)" >> "$OUT"
echo "done" >> "$OUT"
