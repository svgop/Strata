#!/usr/bin/env python3
"""campaign/tracker/build-packcompare.py - the second chart: prefill and decode, all packs, side by side.
Reads lanes.csv (the same data the lane-trend chart draws), takes each pack's LATEST same-day means
(decode solo/pair/conc + the newest prefill reading), and emits pack-compare.html (a self-contained
fragment beside lane-trend.html; regenerate with the tracker after every bank)."""
import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LANES = os.path.join(HERE, 'lanes.csv')

PACKS = [
    ('qwen',           'GSQ IQ3_S',              '81 GiB', 'var(--purple)', 1.0),
    ('qwen-ud-iq2xs',  'UD-IQ2_XS',              '42 GB',  'var(--green)',  1.0),
    ('qwen-ud-iq4xs',  'UD-IQ4_XS',              '94 GB',  'var(--green)',  0.55),
    ('deepseek',       'DeepSeek-V4.1',          '2 bpw',  'var(--blue)',   1.0),
]

def main():
    rows = list(csv.DictReader(open(LANES)))
    for r in rows:
        r['tps'] = float(r['tps'])
    lanes, prefill = defaultdict(dict), {}
    for r in rows:   # last write wins = the newest banked mean per (engine, lane)
        if r['lane'] in ('solo', 'pair', 'conc'):
            lanes[r['engine']][r['lane']] = r['tps']
        elif r['lane'] == 'prefill':
            prefill[r['engine']] = (r['tps'], int(r['ctx']))
    data = []
    for eng, name, size, css, op in PACKS:
        data.append({'name': name, 'size': size, 'css': css, 'opacity': op,
                     'solo': lanes[eng].get('solo'), 'pair': lanes[eng].get('pair'),
                     'conc': lanes[eng].get('conc'),
                     'prefill': prefill[eng][0] if eng in prefill else None,
                     'prefill_ctx': prefill[eng][1] if eng in prefill else None})
    frag = TEMPLATE.replace('__DATA__', json.dumps(data))
    out = os.path.join(HERE, 'pack-compare.html')
    with open(out, 'w') as f:
        f.write(frag)
    print(f'{out} written')

TEMPLATE = r'''<div id="pack-compare">
  <style>
    #pack-compare { font-size: var(--font-size-base); color: var(--foreground); }
    #pack-compare h3 { font-weight: 500; margin: 0 0 4px; }
    #pack-compare .pc-note { color: var(--muted-foreground); margin: 0 0 14px; }
    #pack-compare .pc-panel { margin: 12px 0 20px; }
    #pack-compare .pc-title { font-weight: 500; margin: 0 0 6px; }
    #pack-compare svg text { font-size: 12px; fill: var(--foreground); }
    #pack-compare .pc-bar { transition: width .4s ease; }
  </style>
  <h3>All packs — decode lanes and prefill, the latest banked means</h3>
  <div class="pc-note">One bar per pack, the same data the trend chart banks. Decode panel: tokens/s per lane. Prefill panel: cold-prompt tokens/s (the @24K probes for the Qwen packs; DeepSeek's newest banked reading, its own size).</div>
  <div class="pc-panel"><div class="pc-title">Decode (tokens/s)</div><svg id="pc-decode" width="100%" role="img" aria-label="Decode lanes per pack: solo, pair and concurrent tokens per second as horizontal bars"></svg></div>
  <div class="pc-panel"><div class="pc-title">Prefill (tokens/s)</div><svg id="pc-prefill" width="100%" role="img" aria-label="Prefill rate per pack as horizontal bars"></svg></div>
  <script>
  (function () {
    const data = __DATA__;
    const W = 720;
    function draw(svgId, rows, fmt) {
      const svg = document.getElementById(svgId);
      const rowH = 34, barH = 12, L = 108, R = 64;
      const H = rows.length * rowH + 8;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      const max = Math.max(...rows.map(r => r.v)) * 1.08;
      rows.forEach((r, i) => {
        const y = 6 + i * rowH;
        const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        lbl.setAttribute('x', L - 8); lbl.setAttribute('y', y + barH); lbl.setAttribute('text-anchor', 'end');
        lbl.setAttribute('fill', 'var(--muted-foreground)'); lbl.textContent = r.name;
        svg.appendChild(lbl);
        const track = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        track.setAttribute('x', L); track.setAttribute('y', y); track.setAttribute('width', W - L - R);
        track.setAttribute('height', barH); track.setAttribute('fill', 'var(--muted)'); track.setAttribute('opacity', '.35');
        svg.appendChild(track);
        const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bar.setAttribute('class', 'pc-bar'); bar.setAttribute('x', L); bar.setAttribute('y', y);
        bar.setAttribute('width', (W - L - R) * (r.v / max)); bar.setAttribute('height', barH);
        bar.setAttribute('fill', r.css); if (r.op < 1) bar.setAttribute('opacity', r.op);
        bar.setAttribute('rx', 2);
        bar.setAttribute('data-tooltip', r.tip);
        svg.appendChild(bar);
        const val = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        val.setAttribute('x', L + (W - L - R) * (r.v / max) + 6); val.setAttribute('y', y + barH);
        val.setAttribute('fill', 'var(--foreground)'); val.textContent = fmt(r.v);
        svg.appendChild(val);
      });
    }
    const dec = [];
    data.forEach(p => {
      if (p.solo != null) dec.push({name: p.name + ' solo', v: p.solo, css: p.css, op: p.opacity, tip: `${p.name} solo: ${p.solo} tok/s`});
      if (p.pair != null) dec.push({name: p.name + ' pair', v: p.pair, css: p.css, op: p.opacity, tip: `${p.name} pair: ${p.pair} tok/s aggregate`});
      if (p.conc != null) dec.push({name: p.name + ' conc', v: p.conc, css: p.css, op: p.opacity, tip: `${p.name} 6-concurrent: ${p.conc} tok/s aggregate`});
    });
    draw('pc-decode', dec, v => Math.round(v));
    const pf = data.filter(p => p.prefill != null)
                   .map(p => ({name: p.name, v: p.prefill, css: p.css, op: p.opacity,
                               tip: `${p.name} prefill: ${Math.round(p.prefill)} tok/s @${p.prefill_ctx.toLocaleString()} tokens`}));
    draw('pc-prefill', pf, v => Math.round(v).toLocaleString());
  })();
  </script>
</div>
'''

if __name__ == '__main__':
    main()
