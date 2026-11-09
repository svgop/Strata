#!/bin/bash
# campaign/batteries/gsq-frac-sweep.sh - recalibrate --pcie-frac for the POOL ERA on the live engine
# (per-request strata_tune, rotating rounds, medians - the calibrate.py methodology; the standing
# 0.35 was calibrated before the hugepage architecture changed the miss-path costs).
set -u
cd /home/svg/LLM/strata-custom-2
OUT=campaign/results/gsq-frac-sweep.log
echo "PCIE-FRAC SWEEP (pool era) start $(date)" > "$OUT"
( nohup ./launch-custom-iq3s-8081.sh > campaign/results/gsq-frac-boot.out 2>&1 & )
for t in $(seq 1 100); do curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 3; done
curl -s -m 3 http://127.0.0.1:8081/health >/dev/null || { echo "BOOT FAILED" >> "$OUT"; exit 1; }
for t in $(seq 1 60); do grep -aq "page cache warm in" campaign/results/gsq-frac-boot.out && break; sleep 3; done
sleep 45
python3 - >> "$OUT" 2>&1 <<'PY'
import json, statistics, time, urllib.request
KEY = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
URL = "http://127.0.0.1:8081/v1/chat/completions"
def rate(frac, max_new=512):
    body = json.dumps({"messages": [{"role": "user", "content": "Explain how a modern GPU scheduler arbitrates between graphics and compute queues. Be detailed and technical."}],
                       "max_tokens": max_new, "temperature": 0,
                       "strata_tune": {"pcie_frac": frac}}).encode()
    req = urllib.request.Request(URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.monotonic()
    d = json.loads(urllib.request.urlopen(req, timeout=300).read())
    dt = time.monotonic() - t0
    n = d.get("usage", {}).get("completion_tokens", 0)
    return n / dt if n > 8 and dt > 0 else 0.0
values = [0.0, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0]
out = {v: [] for v in values}
rate(0.35, 64)   # warm
for r in range(3):
    for v in (values if r % 2 == 0 else values[::-1]):
        out[v].append(rate(v))
    print(f"round {r+1}/3: " + "  ".join(f"{v:g}:{out[v][-1]:.1f}" for v in values), flush=True)
med = {v: statistics.median(out[v]) for v in values}
for v in values:
    print(f"frac {v:g}: median {med[v]:.1f} ({min(out[v]):.1f}-{max(out[v]):.1f})", flush=True)
best = max(med, key=med.get)
print(f"BEST frac {best:g} median {med[best]:.1f} vs 0.35 median {med[0.35]:.1f} = {med[best]/med[0.35]-1.0:+.1%}", flush=True)
PY
pkill -f "serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/config/custom-iq3_s" 2>/dev/null
sleep 2; pkill -x strata 2>/dev/null
echo "SWEEP DONE $(date)" >> "$OUT"
echo "ALL DONE"
