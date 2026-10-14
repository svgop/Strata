#!/usr/bin/env python3
"""F-197 battery analyzer: clean-run means per boot (collapse discipline as F-188:
one run per boot may collapse to ~97-100 t/s across ALL metrics - environmental;
dropped and NOTED, never silently)."""
import glob, re, sys

def parse(path):
    r = {}
    for line in open(path, errors='replace'):
        m = re.match(r'M1 single-stream : ([\d.]+) t/s', line)
        if m: r['M1'] = float(m.group(1))
        m = re.match(r'M2 pair          : ([\d.]+) t/s', line)
        if m: r['pair'] = float(m.group(1))
        m = re.match(r'M2 6-conc        : ([\d.]+) t/s', line)
        if m: r['conc'] = float(m.group(1))
        m = re.match(r'M3b rate         : ([\d.]+)%', line)
        if m: r['M3b'] = float(m.group(1))
        m = re.match(r'fresh-long-prompt: (\d+) tok out in ([\d.]+)s', line)
        if m: r['long'] = float(m.group(2))
    return r

boots = {}
for label in sys.argv[1:]:
    runs = []
    for f in sorted(glob.glob(f'campaign/results/{label}-[0-9].txt')):
        p = parse(f)
        if p: runs.append(p)
    # collapse detection: a run whose M1 AND pair AND conc are all < 65% of the boot median
    if len(runs) >= 3:
        med = lambda k: sorted(r[k] for r in runs)[len(runs)//2]
        clean = []
        dropped = []
        for i, r in enumerate(runs):
            if all(r[k] < 0.65 * med(k) for k in ('M1','pair','conc') if k in r):
                dropped.append((i+1, r.get('M1')))
            else:
                clean.append(r)
        if dropped and len(clean) >= 2:
            print(f'  [{label}] dropped collapsed run(s): {dropped}')
        runs = clean
    mean = lambda k: sum(r[k] for r in runs if k in r) / max(1, sum(1 for r in runs if k in r))
    boots[label] = {k: round(mean(k), 1) for k in ('M1','pair','conc','M3b') if any(k in r for r in runs)}
    boots[label]['n'] = len(runs)

w = max(len(l) for l in boots) if boots else 4
print(f'{"boot":<{w}}  n    M1     pair   6-conc  M3b')
for l, b in boots.items():
    print(f'{l:<{w}}  {b.get("n","?"):<3}  {b.get("M1","-"):>6} {b.get("pair","-"):>6} {b.get("conc","-"):>6}  {b.get("M3b","-")}')
