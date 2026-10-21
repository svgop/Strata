#!/bin/bash
# campaign/t27-frac55.sh — pcie-frac falsifier born from calibrate.py's measurement:
# the PCIe probe says 54.5 GB/s -> pcie_frac 0.55; our standing config runs a hand-set 0.35.
# One variable: --pcie-frac 0.55 vs 0.35, ABAB 3v3, config-copy so the standing recipe is untouched.
set -u
REPO=/home/svg/LLM/strata-custom-2
PORT=8081
OUT=$REPO/campaign/results/t27-frac55.log
RES=$REPO/campaign/results
mkdir -p "$RES"
echo "T27 FRAC55 start $(date)" > "$OUT"

# arm config = standing config with --pcie-frac 0.55 (args list edit), arm launcher points at it
python3 - "$REPO" <<'PYEOF'
import json, sys
repo = sys.argv[1]
cfg = json.load(open(f'{repo}/custom-iq3_s-8081.json', encoding='utf-8-sig'))
a = cfg['args']
i = a.index('--pcie-frac')
a[i + 1] = '0.55'
json.dump(cfg, open(f'{repo}/custom-iq3s-frac55.json', 'w'), indent=2)
print('frac55 config written')
PYEOF
sed 's#custom-iq3_s-8081.json#custom-iq3s-frac55.json#' "$REPO/launch-custom-iq3s-8081.sh" > "$REPO/launch-frac55.sh"   # ROOT: launchers resolve .venv from their own dir (F-249 lesson)
chmod +x "$REPO/launch-frac55.sh"
grep -q "frac55" "$REPO/launch-frac55.sh" && grep -q '"0.55"' "$REPO/custom-iq3s-frac55.json" \
  && echo "arm artifacts verified" >> "$OUT" || { echo "ARTIFACT BUILD FAILED - aborting" >> "$OUT"; exit 1; }

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
ARM=$REPO/launch-frac55.sh
for r in 1 2 3; do
  leg "$ARM" "frac55-r$r"
  leg "$CTL" "frac35-ctl-r$r"
done

echo "T27 FRAC55 end $(date)" >> "$OUT"
echo "done" >> "$OUT"
