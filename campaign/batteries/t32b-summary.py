#!/usr/bin/env python3
# campaign/batteries/t32b-summary.py — POOLED dm/both summarizer (T32 extension).
# Parses BOTH campaign/results/t32-tranche.log (3 rounds) and t32b-dm-fiveboot.log
# (5 rounds), pools arms across waves, prints per-wave + pooled means, and judges
# vs the pooled CTL and the F-266 prewarmed anchor. F-263 law: five-boot quiet-window
# means pooled across windows — this IS that instrument for the dm/both question.
import re
from statistics import mean

RES = '/home/svg/LLM/strata-custom-2/campaign/results'
LOGS = [f'{RES}/t32-tranche.log', f'{RES}/t32b-dm-fiveboot.log']
ANCHOR = {'m1': 215.2, 'pair': 206.4, 'conc': 323.0}
KEYS = ('m1', 'pair', 'conc', 'm3b')

def parse_legs():
    legs, cur = [], None
    hdr = re.compile(r'^=== LEG (t32\S*)-(ctl|keep|dm|both)-r(\d+) health=(\d)')
    pats = {
        'm1':   re.compile(r'M1 single-stream\s*:\s*([\d.]+)'),
        'pair': re.compile(r'M2 pair\s*:\s*([\d.]+)'),
        'conc': re.compile(r'M2 6-conc\s*:\s*([\d.]+)'),
        'm3b':  re.compile(r'M3b rate\s*:\s*([\d.]+)'),
    }
    for LOG in LOGS:
        try:
            fh = open(LOG, errors='replace')
        except FileNotFoundError:
            continue
        for line in fh:
            m = hdr.match(line)
            if m:
                cur = {'wave': m.group(1), 'arm': m.group(2), 'rnd': int(m.group(3)),
                       'health': m.group(4)}
                legs.append(cur)
                continue
            if cur is None:
                continue
            for k, p in pats.items():
                if k not in cur:
                    mm = p.search(line)
                    if mm:
                        cur[k] = float(mm.group(1))
        fh.close()
    return legs

def row(leg, label):
    print(f"{label:<26}" + ''.join(f"{leg.get(k, 0):>8.1f}" for k in KEYS))

def main():
    legs = [l for l in parse_legs() if l.get('health') == '1']
    print(f'pooled legs with data: {len(legs)}  (waves: ' +
          ', '.join(sorted({l["wave"] for l in legs})) + ')\n')
    print(f"{'leg':<26}" + ''.join(f'{k:>8}' for k in KEYS))
    for l in sorted(legs, key=lambda x: (x['arm'], x['wave'], x['rnd'])):
        row(l, f"{l['wave']}-{l['arm']}-r{l['rnd']}")

    print()
    pooled = {}
    for arm in ('ctl', 'keep', 'dm', 'both'):
        al = [l for l in legs if l['arm'] == arm and all(k in l for k in KEYS)]
        if not al:
            continue
        pooled[arm] = {k: mean([l[k] for l in al]) for k in KEYS}
        for wave in sorted({l['wave'] for l in al}):
            wl = [l for l in al if l['wave'] == wave]
            print(f"{arm} [{wave}] n={len(wl)}".ljust(26) +
                  ''.join(f"{mean([l[k] for l in wl]):>8.1f}" for k in KEYS))
        print(f"{arm} POOLED n={len(al)}".ljust(26) +
              ''.join(f"{pooled[arm][k]:>8.1f}" for k in KEYS))
        print()

    if 'ctl' in pooled:
        print('pooled arm vs pooled CTL:')
        for arm in ('keep', 'dm', 'both'):
            if arm not in pooled:
                continue
            parts = []
            for k in ('m1', 'pair', 'conc', 'm3b'):
                d = 100 * (pooled[arm][k] - pooled['ctl'][k]) / pooled['ctl'][k]
                parts.append(f'{k} {d:+.1f}%')
            print(f'  {arm:<5} ' + '  '.join(parts))
        print('\npooled arm vs banked prewarmed anchor 215.2/206.4/323.0:')
        for arm in ('ctl', 'keep', 'dm', 'both'):
            if arm not in pooled:
                continue
            parts = []
            for k in ('m1', 'pair', 'conc'):
                d = 100 * (pooled[arm][k] - ANCHOR[k]) / ANCHOR[k]
                parts.append(f'{k} {d:+.1f}%')
            print(f'  {arm:<5} ' + '  '.join(parts))
    print('\nF-263 law: adoption of sub-4% deltas rides five-boot pooled means (above). '
          'Quality guardrails: M3b band + acceptance in-band; keep-containing arms '
          'legitimately shift acceptance/golden-hash by row selection (cut-only rule).')

if __name__ == '__main__':
    main()
