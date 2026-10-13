#!/usr/bin/env python3
# campaign/probe-lane.py — Linux port of campaign/probe-lane.ps1 (the Proof Law instrument).
# Same prompts, same sequence, same engine-log accounting. Usage:
#   .venv/bin/python campaign/probe-lane.py --port 8081 --label linux-standing
import argparse, json, re, threading, time, urllib.request
from datetime import datetime

REAL = ('Write a comprehensive technical analysis of modern GPU memory hierarchies and their impact on '
        'LLM inference performance. Cover HBM bandwidth, L2 cache behavior, PCIe transfer overhead, and '
        'pinned memory tradeoffs, with concrete numbers and examples throughout.')
CONC = ('Explain in detail how mixture-of-experts routing works in modern LLMs, including load balancing, '
        'capacity factors, and the tradeoffs versus dense models.')
SHORT = ('Summarize the tradeoffs between static expert residency and dynamic expert swapping in a '
         'memory-constrained mixture-of-experts inference engine. Be concrete.')
FRESH_TMPL = ('Section {i} of the incident review. At step {i} the operator observed a distinct change in '
              'behaviour and recorded it against the runbook entry number {i}. The measurement taken at '
              'that point was catalogued with identifier {i} and compared against the baseline that had '
              'been established before the change was introduced.')

def probe(base, prompt, max_tok, temp=0.7, key=None):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tok, "temperature": temp}).encode()
    req = urllib.request.Request(base, data=body, headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            d = json.load(r)
        dt = time.time() - t0
        return True, d["usage"]["completion_tokens"], dt
    except Exception:
        return False, 0, time.time() - t0

def conc_run(base, n, max_tok, key=None):
    out = []
    def work():
        out.append(probe(base, CONC, max_tok, 0.7, key))
    ths = [threading.Thread(target=work) for _ in range(n)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    agg_tok = sum(o[1] for o in out)
    return agg_tok, wall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--label', default='linux')
    ap.add_argument('--log', default='/home/svg/LLM/strata-custom-2/custom-iq3_s.log')
    ap.add_argument('--key', default='3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8')
    ap.add_argument('--short-n', type=int, default=6)
    ap.add_argument('--long-tok', type=int, default=2000)
    ap.add_argument('--fresh-sections', type=int, default=90)
    ap.add_argument('--skip-conc', action='store_true')
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}/v1/chat/completions"

    offset0 = 0
    try:
        offset0 = open(a.log, 'rb').seek(0, 2)
    except OSError:
        pass
    print(f"=== PROBE {a.label} | {datetime.now():%Y-%m-%d %H:%M:%S} | log={a.log}")

    ok, tok, sec = probe(base, 'Say OK.', 8, 0, a.key)
    print(f"    warmup: {tok} tok in {sec:.2f}s")

    shortsA = []
    for i in range(1, a.short_n + 1):
        ok, tok, sec = probe(base, SHORT, 220, 0.7, a.key)
        shortsA.append((tok, sec))
        print(f"    A.short[{i}]: {tok} tok in {sec:.2f}s = {tok/sec:.1f} t/s")

    fresh = '\n'.join(FRESH_TMPL.format(i=i) for i in range(1, a.fresh_sections + 1))
    ok, tok, sec = probe(base, fresh, 80, 0.7, a.key)
    print(f"    fresh-long-prompt: {tok} tok out in {sec:.2f}s")

    shortsB = []
    for i in range(1, a.short_n + 1):
        ok, tok, sec = probe(base, SHORT, 220, 0.7, a.key)
        shortsB.append((tok, sec))
        print(f"    B.short[{i}]: {tok} tok in {sec:.2f}s = {tok/sec:.1f} t/s")
    va = [t/s for t, s in shortsA if t > 20]
    vb = [t/s for t, s in shortsB if t > 20]
    if va and vb:
        ma, mb = sum(va)/len(va), sum(vb)/len(vb)
        print(f"SHORT-CLASS A/B   : before long prompt {ma:.1f} t/s -> after {mb:.1f} t/s ({100*(mb-ma)/ma:.0f}%)")

    ok, tok, sec = probe(base, REAL, a.long_tok, 0.7, a.key)
    print(f"M1 single-stream : {tok/sec:.1f} t/s ({tok} tok in {sec:.2f}s)")

    if not a.skip_conc:
        tok, wall = conc_run(base, 2, 900, a.key)
        print(f"M2 pair          : {tok/wall:.1f} t/s aggregate ({tok} tok in {wall:.2f}s)")
        tok, wall = conc_run(base, 6, 600, a.key)
        print(f"M2 6-conc        : {tok/wall:.1f} t/s aggregate ({tok} tok in {wall:.2f}s)")

    time.sleep(2)
    with open(a.log, 'r', errors='replace') as f:
        f.seek(offset0)
        lines = f.read().splitlines()

    rows, pending = [], None
    for l in lines:
        m = re.search(r'prompt (\d+) tokens.*?(\d+) generated in (\d+) ms.*?drafts accepted (\d+) of (\d+)', l)
        if m:
            if pending: rows.append(pending)
            g, ms, acc, tot = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            win = g - acc
            pending = dict(prompt=int(m.group(1)), gen=g, ms=ms, acc=acc, of=tot, win=win,
                           mswin=round(ms/win, 1) if win > 0 else None,
                           accept=round(100*acc/max(tot, 1), 1), hit=None)
        elif pending:
            h = re.search(r'decode expert cache hit rate: ([\d.]+)%', l)
            if h:
                pending['hit'] = float(h.group(1)); rows.append(pending); pending = None
    if pending: rows.append(pending)

    print("--- engine accounting, in order (ms/window is the degradation signal) ---")
    for k, r in enumerate(rows, 1):
        mw = f"{r['mswin']:7}" if r['mswin'] else '      -'
        hit = f"{r['hit']}" if r['hit'] is not None else '-'
        print(f"    [{k:2}] prompt {r['prompt']:4} gen {r['gen']:5} -> {mw} ms/window | accept {r['accept']:5}% ({r['acc']}/{r['of']}) | hit {hit}%")
    mws = [r['mswin'] for r in rows if r['mswin']]
    if len(mws) >= 4:
        half = len(mws)//2
        early, late = sum(mws[:half])/half, sum(mws[half:])/(len(mws)-half)
        print(f"TREND ms/window: early {early:.1f} -> late {late:.1f} ({'DEGRADING' if late > 1.3*early else 'stable'})")
    accs = [r['accept'] for r in rows if r['of'] >= 8]
    if accs:
        print(f"M3b rate         : {sum(accs)/len(accs):.1f}% mean over {len(accs)} requests (diagnostic)")

    dt = []
    for l in lines:
        m = re.search(r'avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window', l)
        if m:
            tw, mw = float(m.group(2)), float(m.group(3))
            if mw > 0:
                dt.append((tw, mw, 1000.0*(tw-1.0)/mw))
    if dt:
        atw = sum(x[0] for x in dt)/len(dt); amw = sum(x[1] for x in dt)/len(dt)
        aaps = sum(x[2] for x in dt)/len(dt)
        print(f"M3 accepted/s    : {aaps:.1f} accepted draft tokens per second (mean over {len(dt)} windows; {atw:.2f} banked tokens/window at {amw:.2f} ms/window)")
    pos = [l for l in lines if 'spec depth: accepted/tested' in l]
    for p in pos: print(f"M3d per-position : {p}")
    if not pos:
        print("M3d per-position : not printed (arm STRATA_SPEC_DEPTH=1 to publish accepted/tested by depth)")
    print(f"=== PROBE END {a.label}")

if __name__ == '__main__':
    main()
