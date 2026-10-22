#!/usr/bin/env python3
# campaign/t31-summary.py — T31 re-bank summarizer (F-266). Parses
# campaign/results/t31-rebank.log legs + campaign/results/metrics.jsonl records
# for t31-* labels, prints per-boot rows, phase means (the F-263 anchor),
# comparison vs the banked scoreboard, and the prewarm2 arm verdict inputs.
import json, re, sys
from statistics import mean, median

RES = '/home/svg/LLM/strata-custom-2/campaign/results'
LOG = f'{RES}/t31-rebank.log'
BANKED = {'m1': 212.5, 'pair': 202.8, 'conc': 315.7}  # T19 epoch five-boot means

def parse_legs():
    legs, cur = [], None
    hdr = re.compile(r'^=== LEG (t31-\S+) health=(\d)')
    pats = {
        'm1':    re.compile(r'M1 single-stream\s*:\s*([\d.]+)'),
        'pair':  re.compile(r'M2 pair\s*:\s*([\d.]+)'),
        'conc':  re.compile(r'M2 6-conc\s*:\s*([\d.]+)'),
        'm3b':   re.compile(r'M3b rate\s*:\s*([\d.]+)'),
        'trend': re.compile(r'TREND ms/window: early ([\d.]+) -> late ([\d.]+)'),
    }
    for line in open(LOG, errors='replace'):
        m = hdr.match(line)
        if m:
            cur = {'label': m.group(1), 'health': m.group(2)}
            legs.append(cur)
            continue
        if cur is None:
            continue
        for k, p in pats.items():
            if k not in cur or k == 'trend':
                mm = p.search(line)
                if mm:
                    cur[k] = (float(mm.group(1)), float(mm.group(2))) if k == 'trend' else float(mm.group(1))
    return legs

def parse_metrics():
    rows = []
    try:
        for line in open(f'{RES}/metrics.jsonl', errors='replace'):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(d.get('label', '')).startswith('t31-'):
                rows.append(d)
    except FileNotFoundError:
        pass
    return rows

def gget(rec, *keys, default=None):
    for k in keys:
        try:
            rec = rec[k]
        except (KeyError, TypeError):
            return default
    return rec

def main():
    legs = [l for l in parse_legs() if l.get('health') == '1']
    mets = parse_metrics()
    print(f'T31 legs with data: {len(legs)}   metrics records: {len(mets)}\n')

    print(f"{'leg':<14}{'M1':>7}{'pair':>7}{'conc':>7}{'M3b':>7}{'ITLp50':>9}{'ITLp99':>9}{'clk':>7}{'T':>4}{'W':>6}  golden")
    mbyl = {m.get('label'): m for m in mets}
    for l in legs:
        m = mbyl.get(l['label'], {})
        itl = gget(m, 'hist', 'vllm:inter_token_latency_seconds', default={})
        clk = gget(m, 'gpu', 'sm_clock_mhz', default=float('nan'))
        tmp = gget(m, 'gpu', 'temp_c', default=float('nan'))
        pw  = gget(m, 'gpu', 'power_w', default=float('nan'))
        gh  = str(gget(m, 'golden', 'hash', default='-'))[:8]
        print(f"{l['label']:<14}{l.get('m1',0):>7.1f}{l.get('pair',0):>7.1f}{l.get('conc',0):>7.1f}"
              f"{l.get('m3b',0):>7.1f}{itl.get('0.5',float('nan')):>9.4f}{itl.get('0.99',float('nan')):>9.4f}"
              f"{clk:>7.0f}{tmp:>4.0f}{pw:>6.0f}  {gh}")

    for phase, name in (('t31-base', 'ANCHOR (base, 5 boots)'), ('t31-pre', 'ARM (prewarm2, 3 boots)')):
        pl = [l for l in legs if l['label'].startswith(phase)]
        if not pl:
            continue
        print(f'\n{name}: n={len(pl)}')
        for k in ('m1', 'pair', 'conc', 'm3b'):
            vs = [l[k] for l in pl if k in l]
            if vs:
                print(f'  {k:<5} mean {mean(vs):7.1f}   min {min(vs):7.1f}   max {max(vs):7.1f}   '
                      f'vals {" ".join(f"{v:.1f}" for v in vs)}')
        tr = [l['trend'] for l in pl if 'trend' in l]
        if tr:
            print(f'  trend early->late mean: {mean(t[0] for t in tr):.1f} -> {mean(t[1] for t in tr):.1f} ms/window')

    base = [l for l in legs if l['label'].startswith('t31-base')]
    if base:
        print('\nvs BANKED five-boot means (T19 epoch):')
        for k in ('m1', 'pair', 'conc'):
            vs = [l[k] for l in base if k in l]
            if vs:
                d = 100 * (mean(vs) - BANKED[k]) / BANKED[k]
                print(f'  {k:<5} anchor {mean(vs):7.1f} vs banked {BANKED[k]:7.1f}  ({d:+.1f}%)')

    pre = [l for l in legs if l['label'].startswith('t31-pre')]
    if base and pre:
        print('\nprewarm2 arm vs anchor (same window, pooled means; adoption needs the arm mean clear of '
              'the anchor band, boots not straddling - F-263 law):')
        for k in ('m1', 'pair', 'conc'):
            b, p = [l[k] for l in base if k in l], [l[k] for l in pre if k in l]
            if b and p:
                print(f'  {k:<5} arm {mean(p):7.1f} vs anchor {mean(b):7.1f}  ({100*(mean(p)-mean(b))/mean(b):+.1f}%)'
                      f'   arm vals {" ".join(f"{v:.1f}" for v in p)}')
    hashes = {str(gget(m, 'golden', 'hash', default='-'))[:12] for m in mets}
    print(f'\ngolden hashes this run: {len(hashes)} distinct: {sorted(hashes)}')

if __name__ == '__main__':
    sys.exit(main())
