#!/usr/bin/env python3
# campaign/batteries/t32-summary.py — T32 tranche summarizer (F-270). Parses
# campaign/results/t32-tranche.log legs, prints per-arm rows + means, and judges
# each arm vs the battery's own CTL legs (same window) and the banked prewarmed
# anchor (F-266). Decision rule printed with the table: sub-4% deltas need
# five-boot quiet-window means pooled across windows before adoption (F-263).
import re, sys
from statistics import mean

RES = '/home/svg/LLM/strata-custom-2/campaign/results'
LOG = f'{RES}/t32-tranche.log'
ANCHOR = {'m1': 215.2, 'pair': 206.4, 'conc': 323.0}   # F-266 prewarmed anchor
ARMS = ('ctl', 'keep', 'dm', 'both')

def parse_legs():
    legs, cur = [], None
    hdr = re.compile(r'^=== LEG (t32-\S+)-r(\d+) health=(\d)')
    pats = {
        'm1':   re.compile(r'M1 single-stream\s*:\s*([\d.]+)'),
        'pair': re.compile(r'M2 pair\s*:\s*([\d.]+)'),
        'conc': re.compile(r'M2 6-conc\s*:\s*([\d.]+)'),
        'm3b':  re.compile(r'M3b rate\s*:\s*([\d.]+)'),
    }
    for line in open(LOG, errors='replace'):
        m = hdr.match(line)
        if m:
            cur = {'label': m.group(1), 'health': m.group(3)}
            legs.append(cur)
            continue
        if cur is None:
            continue
        for k, p in pats.items():
            if k not in cur:
                mm = p.search(line)
                if mm:
                    cur[k] = float(mm.group(1))
    return legs

def main():
    legs = [l for l in parse_legs() if l.get('health') == '1']
    print(f'T32 legs with data: {len(legs)}\n')
    print(f"{'leg':<14}{'M1':>7}{'pair':>7}{'conc':>7}{'M3b':>7}")
    for l in legs:
        print(f"{l['label'].replace('t32-',''):<14}{l.get('m1',0):>7.1f}{l.get('pair',0):>7.1f}"
              f"{l.get('conc',0):>7.1f}{l.get('m3b',0):>7.1f}")

    means = {}
    for arm in ARMS:
        al = [l for l in legs if l['label'] == f't32-{arm}']
        if not al:
            continue
        means[arm] = {k: mean([l[k] for l in al if k in l]) for k in ('m1', 'pair', 'conc', 'm3b')}
        n = len(al)
        vals = ' '.join(f"{l.get('m1',0):.0f}" for l in al)
        print(f'\n{arm} (n={n})   M1 vals: {vals}')
        for k in ('m1', 'pair', 'conc', 'm3b'):
            print(f'  {k:<5} mean {means[arm][k]:7.1f}')

    if 'ctl' in means:
        print('\nvs battery CTL (same window):')
        for arm in ARMS:
            if arm == 'ctl' or arm not in means:
                continue
            parts = []
            for k in ('m1', 'pair', 'conc'):
                d = 100 * (means[arm][k] - means[arm and 'ctl'][k]) / means['ctl'][k]
                parts.append(f'{k} {d:+.1f}%')
            print(f'  {arm:<5} ' + '  '.join(parts))
        print('\nvs banked prewarmed anchor 215.2/206.4/323.0:')
        for arm in ARMS:
            if arm not in means:
                continue
            parts = []
            for k in ('m1', 'pair', 'conc'):
                d = 100 * (means[arm][k] - ANCHOR[k]) / ANCHOR[k]
                parts.append(f'{k} {d:+.1f}%')
            print(f'  {arm:<5} ' + '  '.join(parts))
    print('\nF-263 law: sub-4% deltas above need five-boot quiet-window means pooled across windows '
          'before adoption; quality guardrails: M3b band and acceptance must stay in-band.')

if __name__ == '__main__':
    sys.exit(main())
