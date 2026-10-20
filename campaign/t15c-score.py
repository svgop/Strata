#!/usr/bin/env python3
"""T15C battery scorer: interleaved arm/ctl means from a t15c-<stage>.log.
Legs are `=== LEG <stage>-arm-rN health=1` / `-ctl-rN` followed by a probe block
(probe-lane.py output captured into the same log). Prints per-lane arm/ctl means
and the arm delta %. Usage: t15c-score.py <stage> [log]"""
import re, sys, statistics

def parse(path):
    legs = {}   # label -> dict of metric -> value
    cur = None
    for line in open(path, errors='replace'):
        m = re.match(r'=== LEG ([\w-]+) health=(\d)', line)
        if m:
            cur = m.group(1) if m.group(2) == '1' else None
            continue
        if cur is None:
            continue
        for pat, key in ((r'PROBE END (\S+)', 'END'),
                         (r'M1 single-stream : ([\d.]+) t/s', 'M1'),
                         (r'M2 pair          : ([\d.]+) t/s', 'pair'),
                         (r'M2 6-conc        : ([\d.]+) t/s', 'conc'),
                         (r'M3b rate         : ([\d.]+)%', 'M3b')):
            m = re.search(pat, line)
            if m:
                if key == 'END':
                    if m.group(1).endswith(cur): legs.setdefault(cur, {})['done'] = 1
                else:
                    legs.setdefault(cur, {})[key] = float(m.group(1))
    return legs

stage = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else f'campaign/results/t15c-{stage}.log'
legs = parse(path)
lanes = ('M1', 'pair', 'conc', 'M3b')
arm = {k: [] for k in lanes}; ctl = {k: [] for k in lanes}
for name, vals in sorted(legs.items()):
    if not vals.get('done'):
        print(f'  (incomplete leg skipped: {name})'); continue
    side = arm if '-arm' in name else ctl
    for k in lanes:
        if k in vals: side[k].append(vals[k])
print(f'== {stage} ({len(arm["M1"])}v{len(ctl["M1"])} complete legs) ==')
for k in lanes:
    if not arm[k] or not ctl[k]:
        print(f'{k:5s}: missing legs'); continue
    ma, mc = statistics.mean(arm[k]), statistics.mean(ctl[k])
    d = 100 * (ma - mc) / mc
    print(f'{k:5s}: arm {ma:8.2f} ({" ".join(f"{v:.1f}" for v in arm[k])})  '
          f'ctl {mc:8.2f} ({" ".join(f"{v:.1f}" for v in ctl[k])})  delta {d:+.2f}%')
