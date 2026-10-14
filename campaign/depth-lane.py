#!/usr/bin/env python3
"""depth-lane.py - M6-class depth probe: one request, prompt filled to a target context
depth with deterministic synthetic filler, then a fixed decode. Reports prefill t/s
and decode t/s from the server's own timings, plus the depth actually reached.

Usage: python depth-lane.py --port 8081 --label d262k --depth 240000 --decode-tok 1500
Writes campaign/results/depth-<label>.txt and prints the headline lines.
"""
import argparse, json, random, time, urllib.request, os

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8081)
ap.add_argument('--label', default='depth')
ap.add_argument('--depth', type=int, default=240000, help='target prompt tokens')
ap.add_argument('--decode-tok', type=int, default=1500)
ap.add_argument('--key', default='3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8')
ap.add_argument('--temp', type=float, default=0.7)
a = ap.parse_args()

# Deterministic pseudo-text: varied sentences so KV/index/routing see realistic diversity
rng = random.Random(4242)
WORDS = ('system atlas vector module harbor quantum ledger freight meadow cipher lantern '
         'orchid tunnel granite beacon piston willow saffron relay cobalt theorem pasture '
         'velocity dossier crescent furnace lattice omens kettle prairie spline terrace '
         'umbra vellum wicket zephyr anvil marigold socket trellis basalt krill napkin '
         'onyx pivot quiver rune siphon tundra urn vortex wharf yield zenith').split()
SECTION = ('Record {n}: field agents surveyed the {w1} installation near the {w2} perimeter. '
           'The {w3} telemetry showed nominal drift across {n} calibrated spans; the {w4} '
           'ledger reconciled with the {w5} archive within tolerance. Follow-up: inspect the '
           '{w6} conduit at the next {w7} cycle and file the {w8} annex.\n')

def build_filler(target_tokens):
    parts, n = [], 0
    while n < target_tokens + 8192:
        n += 62
        parts.append(SECTION.format(n=n, **{f'w{i}': rng.choice(WORDS) for i in range(1, 9)}))
    return ' '.join(parts)

prompt = build_filler(a.depth)
body = {"model": "m", "messages": [{"role": "user", "content":
        prompt + "\n\nNow, ignoring the records above, write a detailed 700-word technical essay about tide-pool ecosystems, covering species, tides, and conservation, ending with a bulleted summary."}],
        "temperature": a.temp, "max_tokens": a.decode_tok, "stream": False}
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                             json.dumps(body).encode(),
                             {"Content-Type": "application/json", "Authorization": f"Bearer {a.key}"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=7200) as r:
    out = json.load(r)
wall = time.time() - t0

u = out.get("usage", {})
t = out.get("timings", {})
pt = u.get("prompt_tokens", 0); ct = u.get("completion_tokens", 0)
lines = [
    f"DEPTH-LANE {a.label}",
    f"depth reached     : {pt} prompt tokens (target {a.depth})",
    f"decoded           : {ct} tokens",
    f"wall total        : {wall:.1f}s",
    f"server timings    : {json.dumps(t)[:400]}",
]
# timings field names vary; extract any *prefill*/prompt rate and decode rate if present
for k, v in t.items():
    if 'prefill' in k.lower() or 'prompt' in k.lower(): lines.append(f"prefill metric {k} = {v}")
    if 'decode' in k.lower() or 'gen' in k.lower() or 'tok/s' in k.lower(): lines.append(f"decode metric {k} = {v}")
report = '\n'.join(lines)
print(report)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', f"depth-{a.label}.txt"), 'w') as f:
    f.write(report + '\n')
