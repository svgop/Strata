#!/usr/bin/env python3
"""depth-lane-real.py - T17F item-4 falsifier variant: the depth probe with REAL corpus text
instead of depth-lane.py's deterministic synthetic filler. The MIMIC corpus does NOT exist in
Custom-data (checked 2026-10-17: only models/, mtp/, packs/, fingerprints), so the real-corpus
source here is the repo's own technical prose (docs, README, AGENTS, campaign docs, journal) -
real, diverse, human/agent-written English, NOT template filler. Verdict scope: a position-vs-
content split on technical prose; the MIMIC-domain variant stays open.

Run BOTH variants on the SAME boot (filler via depth-lane.py, real via this) at the same target
depth with the same decode instruction: same position, different content class -> the acceptance
delta is the content-driven share of F-181's deep-acceptance decay.

Usage: python depth-lane-real.py --port 8081 --label real120k --depth 120000 --decode-tok 1500
Writes campaign/results/depth-<label>.txt and prints headline lines.
"""
import argparse, glob, json, os, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8081)
ap.add_argument('--label', default='real')
ap.add_argument('--depth', type=int, default=120000, help='target prompt tokens')
ap.add_argument('--decode-tok', type=int, default=1500)
ap.add_argument('--key', default='3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8')
ap.add_argument('--temp', type=float, default=0.7)
ap.add_argument('--log', default='/home/svg/LLM/strata-custom-2/custom-iq3_s.log')
a = ap.parse_args()

REPO = '/home/svg/LLM/strata-custom-2'
# real prose, deterministic order, ~4.1 chars/token -> depth tokens needs ~4.1*depth chars
SOURCES = [
    f'{REPO}/AGENTS.md', f'{REPO}/README.md',
    *sorted(glob.glob(f'{REPO}/docs/*.md')),
    f'{REPO}/campaign/GOAL-LINUX-SUPERIORITY.md',
    f'{REPO}/SVG-CUSTOM-TODO.md', f'{REPO}/SVG-CUSTOM.md',
]
target_chars = int(a.depth * 3.27) + 8192  # calibrated 2026-10-17: repo prose = 3.27 chars/token
parts = []
for p in SOURCES:
    if len(''.join(parts)) >= target_chars:
        break
    try:
        with open(p, 'r', errors='replace') as f:
            t = f.read()
    except OSError:
        continue
    parts.append(f"\n\n===== FILE: {os.path.basename(p)} =====\n\n" + t)
prompt = ''.join(parts)[:target_chars]

body = {"model": "m", "messages": [{"role": "user", "content":
        prompt + "\n\nNow, ignoring the records above, write a detailed 700-word technical essay about tide-pool ecosystems, covering species, tides, and conservation, ending with a bulleted summary."}],
        "temperature": a.temp, "max_tokens": a.decode_tok, "stream": False}
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                             json.dumps(body).encode(),
                             {"Content-Type": "application/json", "Authorization": f"Bearer {a.key}"})

off = 0
try:
    off = open(a.log, 'rb').seek(0, 2)
except OSError:
    pass
t0 = time.time()
with urllib.request.urlopen(req, timeout=7200) as r:
    out = json.load(r)
wall = time.time() - t0

u = out.get("usage", {})
pt = u.get("prompt_tokens", 0); ct = u.get("completion_tokens", 0)
lines = [
    f"DEPTH-LANE-REAL {a.label}",
    f"source corpus    : repo technical prose ({len(SOURCES)} files, real text; MIMIC corpus ABSENT from Custom-data)",
    f"depth reached     : {pt} prompt tokens (target {a.depth})",
    f"decoded           : {ct} tokens",
    f"wall total        : {wall:.1f}s",
]
time.sleep(2)
with open(a.log, 'r', errors='replace') as f:
    f.seek(off)
    tail = f.read().splitlines()
for l in tail:
    if re.search(r'prompt \d+ tokens.*?drafts accepted \d+ of \d+', l):
        lines.append(f"engine accept    : {l.strip()}")
    m = re.search(r'avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window', l)
    if m:
        lines.append(f"window summary   : T={m.group(1)}, {m.group(2)} tok/window, {m.group(3)} ms/window")
report = '\n'.join(lines)
print(report)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', f"depth-{a.label}.txt"), 'w') as f:
    f.write(report + '\n')
