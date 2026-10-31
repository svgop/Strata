#!/usr/bin/env python3
"""campaign/tracker/build-trend.py - the lane-trend tracker builder.

Reads lanes.csv (append-only: one row per measured lane point), writes
lanes.json (OKF) and regenerates the trend chart fragment with the data
inlined (no fetch in fragments). Rerun after every battery; the chart's
job is to show the incline and catch any lane that turns downward.

Usage: python3 build-trend.py [viz-dir]
  viz-dir defaults to this directory (writes lane-trend.html; copy/link it
  into a session visualization directory to render inline).
"""
import csv, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LANES = os.path.join(HERE, 'lanes.csv')

ENGINES = {
    'qwen': {'label': 'Qwen3.8 IQ3_S (custom engine)', 'css': 'var(--purple)'},
    'deepseek': {'label': 'DeepSeek-V4.1-Flash 2.00bpw (dsv41)', 'css': 'var(--blue)'},
}
LANE_ORDER = ['solo', 'pair', 'conc']

def d3_symbol(name):
    return {'circle': 'M0,3.2A3.2,3.2 0 1,1 0,-3.2A3.2,3.2 0 1,1 0,3.2Z',
            'square': 'M-3,-3H3V3H-3Z',
            'triangle': 'M0,-3.6L3.4,2.6L-3.4,2.6Z'}[name]

LANE_STYLE = {
    'solo': {'dash': 'none', 'marker': 'circle', 'symbol': d3_symbol('circle')},
    'pair': {'dash': '6 3', 'marker': 'square', 'symbol': d3_symbol('square')},
    'conc': {'dash': '2 3', 'marker': 'triangle', 'symbol': d3_symbol('triangle')},
}

def main():
    viz_dir = sys.argv[1] if len(sys.argv) > 1 else HERE
    rows = list(csv.DictReader(open(LANES)))
    for r in rows:
        r['tps'] = float(r['tps'])
    # group: engine -> ordered points -> lanes; prefill rows ride their own series (x = prompt size)
    pts, pf = {}, {eng: [] for eng in ENGINES}
    for eng in ENGINES:
        e = [r for r in rows if r['engine'] == eng]
        seen, order = {}, []
        for r in e:
            if r['lane'] == 'prefill':
                pf[eng].append({'date': r['date'], 'commit': r['commit'], 'label': r['label'],
                                'tps': r['tps'], 'ctx': int(r['ctx']), 'kind': r['kind']})
                continue
            key = (r['date'], r['commit'], r['label'])
            if key not in seen:
                seen[key] = {'date': r['date'], 'commit': r['commit'], 'label': r['label'],
                             'solo': None, 'pair': None, 'conc': None, 'kinds': {}}
                order.append(key)
            p = seen[key]
            if r['lane'] in LANE_ORDER:
                v = p[r['lane']]
                p[r['lane']] = r['tps'] if v is None else max(v, r['tps'])
                p['kinds'][r['lane']] = r['kind']
        pts[eng] = [seen[k] | {'kinds': seen[k]['kinds']} for k in order]
    data = {'type': 'strata-lane-trend', 'updated': rows[-1]['date'], 'unit': 'tok/s',
            'engines': {eng: {'label': ENGINES[eng]['label'], 'points': pts[eng], 'prefill': pf[eng]} for eng in ENGINES}}
    with open(os.path.join(HERE, 'lanes.json'), 'w') as f:
        json.dump(data, f, indent=1)
    frag = TEMPLATE.replace('__DATA__', json.dumps(data['engines'])) \
                   .replace('__LANES__', json.dumps(LANE_STYLE)) \
                   .replace('__UPDATED__', data['updated'])
    out = os.path.join(viz_dir, 'lane-trend.html')
    with open(out, 'w') as f:
        f.write(frag)
    n = {eng: len(pts[eng]) for eng in pts}
    print(f"lanes.json + {out} written | points: {n}")

TEMPLATE = r'''<div id="lane-trend-9f3" style="font:inherit">
<h2>Lane throughput trend &mdash; each point is an optimization / bank</h2>
<p class="text-small" style="margin:2px 0 10px">tok/s, client-measured standard battery (solo M1, pair M2, 6-lane conc M2) &middot; updated __UPDATED__ &middot; source: campaign/tracker/lanes.csv</p>
<div id="lt-legend-9f3" class="viz-row" style="gap:2px 10px;margin-bottom:8px"></div>
<div id="lt-panels-9f3"></div>
<h3 style="margin:14px 0 2px">Prefill &mdash; prompt tok/s by prompt size</h3>
<div id="lt-prefill-9f3"></div>
<div class="tooltip" id="lt-tip-9f3" role="tooltip" style="position:absolute;pointer-events:none;opacity:0"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(function () {
  const ENGINES = __DATA__;
  const LANES = __LANES__;
  const LANE_NAMES = { solo: 'solo', pair: 'pair', conc: '6-lane conc' };
  const COLORS = { qwen: 'var(--purple)', deepseek: 'var(--blue)' };
  const ENAME = { qwen: 'Qwen', deepseek: 'DeepSeek' };
  const root = document.getElementById('lane-trend-9f3');
  const tip = document.getElementById('lt-tip-9f3');
  const visible = { qwen: { solo: true, pair: true, conc: true },
                    deepseek: { solo: true, pair: true, conc: true } };

  // legend: one borderless toggle per engine x lane
  const leg = document.getElementById('lt-legend-9f3');
  const legBtns = [];
  for (const eng of Object.keys(ENGINES)) for (const lane of Object.keys(LANE_NAMES)) {
    const b = document.createElement('button');
    b.type = 'button'; b.setAttribute('aria-pressed', 'true');
    b.style.cssText = 'background:none;border:0;padding:2px 4px;font:inherit;cursor:pointer;display:inline-flex;align-items:center;gap:5px';
    const sw = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    sw.setAttribute('width', 22); sw.setAttribute('height', 10); sw.setAttribute('aria-hidden', 'true');
    const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    ln.setAttribute('x1', 0); ln.setAttribute('x2', 22); ln.setAttribute('y1', 5); ln.setAttribute('y2', 5);
    ln.setAttribute('stroke', COLORS[eng]); ln.setAttribute('stroke-width', 2);
    if (LANES[lane].dash !== 'none') ln.setAttribute('stroke-dasharray', LANES[lane].dash);
    sw.appendChild(ln);
    const tx = document.createElement('span'); tx.className = 'text-small';
    tx.textContent = ENAME[eng] + ' ' + LANE_NAMES[lane];
    b.append(sw, tx);
    b.addEventListener('click', () => {
      visible[eng][lane] = !visible[eng][lane];
      b.setAttribute('aria-pressed', String(visible[eng][lane]));
      b.style.opacity = visible[eng][lane] ? '1' : '.45';
      drawAll();
    });
    leg.appendChild(b); legBtns.push(b);
  }

  const panelsRoot = document.getElementById('lt-panels-9f3');
  const panels = [];
  for (const eng of Object.keys(ENGINES)) {
    const wrap = document.createElement('div');
    wrap.innerHTML = '<h3 style="margin:10px 0 2px">' + ENGINES[eng].label + '</h3>';
    const holder = document.createElement('div');
    wrap.appendChild(holder); panelsRoot.appendChild(wrap);
    panels.push({ eng, holder });
  }

  function fmt(v) { return v == null ? '\u2014' : (Math.round(v * 10) / 10).toFixed(1); }

  function draw(p) {
    const pts = ENGINES[p.eng].points;
    const W = Math.max(320, p.holder.clientWidth || 320);
    const H = Math.max(150, Math.min(260, W * 0.38));
    const M = { t: 14, r: 118, b: 34, l: 46 };
    d3.select(p.holder).select('svg').remove();
    const svg = d3.select(p.holder).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('display', 'block')
      .attr('role', 'img').attr('aria-label', ENAME[p.eng] + ' lane throughput trend by optimization');
    svg.append('rect').attr('data-chart-frame', true).attr('x', M.l).attr('y', M.t)
       .attr('width', Math.max(10, W - M.l - M.r)).attr('height', Math.max(10, H - M.t - M.b))
       .attr('fill', 'none').attr('stroke', 'var(--border)').attr('stroke-width', 1);
    const x = d3.scalePoint().domain(pts.map((_, i) => i)).range([M.l + 8, W - M.r - 8]).padding(0.5);
    const vals = pts.flatMap(d => ['solo', 'pair', 'conc'].map(l => d[l]).filter(v => v != null));
    const y = d3.scaleLinear().domain(d3.extent(vals).map((v, i, a) => i === 0 ? v * 0.92 : v * 1.05)).nice().range([H - M.b, M.t]);
    const ya = d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(6);
    const g = svg.append('g');
    g.append('g').attr('transform', `translate(${M.l},0)`).call(ya)
      .call(sel => sel.select('.domain').remove())
      .selectAll('text').attr('fill', 'var(--muted-foreground)').style('font-size', '12px').attr('class', 'tabular-nums');
    g.append('g').selectAll('line.grid').data(y.ticks(4)).join('line')
      .attr('x1', M.l).attr('x2', W - M.r).attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', 'var(--border)').attr('stroke-width', 0.5).attr('opacity', 0.55);
    const xa = d3.axisBottom(x).tickSize(0).tickPadding(8)
      .tickValues(pts.map((_, i) => i).filter(i => i % Math.max(1, Math.ceil(pts.length / 5)) === 0))
      .tickFormat(i => pts[i].commit.slice(0, 7));
    g.append('g').attr('transform', `translate(0,${H - M.b})`).call(xa)
      .call(sel => sel.select('.domain').attr('stroke', 'var(--border)'))
      .selectAll('text').attr('fill', 'var(--muted-foreground)').style('font-size', '11px');
    svg.append('text').attr('class', 'text-small').attr('x', M.l).attr('y', H - 6)
       .attr('fill', 'var(--muted-foreground)').style('font-size', '12px').text('commit / optimization');
    svg.append('text').attr('data-axis', 'y').attr('x', 12).attr('y', M.t + 2)
       .attr('fill', 'var(--muted-foreground)').style('font-size', '12px').text('tok/s');

    for (const lane of ['solo', 'pair', 'conc']) {
      const col = COLORS[p.eng];
      const all = pts.map((d, i) => ({ i, v: d[lane], d, k: d.kinds[lane] || 'mean' }));
      const isLine = q => q.v != null && (q.k === 'mean' || q.k === 'standing');
      const isDet = q => q.v != null && !isLine(q);
      const line = d3.line().defined(q => q.v != null).x(q => x(q.i)).y(q => y(q.v));
      if (visible[p.eng][lane]) {
        g.append('path').datum(all.filter(isLine)).attr('d', line)
          .attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2)
          .attr('stroke-dasharray', LANES[lane].dash === 'none' ? null : LANES[lane].dash);
        g.selectAll(null).data(all.filter(isLine)).join('path')
          .attr('d', LANES[lane].symbol).attr('transform', q => `translate(${x(q.i)},${y(q.v)})`)
          .attr('fill', col).attr('data-chart-hit', true)
          .append('title').text(q => `${q.d.label} (${q.d.commit}) ${q.d.date} · ${lane}: ${fmt(q.v)} tok/s`);
        // records / arm reads: detached diamonds (never a line segment - the incline law)
        g.selectAll(null).data(all.filter(isDet)).join('path')
          .attr('d', 'M0,-4.2L4.2,0L0,4.2L-4.2,0Z').attr('transform', q => `translate(${x(q.i)},${y(q.v)})`)
          .attr('fill', 'none').attr('stroke', col).attr('stroke-width', 1.6).attr('data-chart-hit', true)
          .append('title').text(q => `${q.d.label} (${q.d.commit}) ${q.d.date} · ${lane}: ${fmt(q.v)} tok/s [${q.k}, not standing]`);
        const last = [...all].reverse().find(q => q.v != null);
        if (last) g.append('text').attr('x', x(last.i) + 7).attr('y', y(last.v) + 4)
          .attr('fill', col).style('font-size', '12px').attr('class', 'tabular-nums')
          .text(`${LANE_NAMES[lane]} ${fmt(last.v)}`);
      }
    }
    // cross-series hover
    const overlay = g.append('rect').attr('data-chart-hover-overlay', 'cross-series')
      .attr('x', M.l).attr('y', M.t).attr('width', Math.max(10, W - M.l - M.r))
      .attr('height', Math.max(10, H - M.t - M.b)).attr('fill', 'transparent').attr('data-chart-hit', true);
    const guide = g.append('line').attr('data-chart-hover-guide', true)
      .attr('y1', M.t).attr('y2', H - M.b).attr('stroke', 'var(--border)').attr('stroke-width', 1).style('opacity', 0);
    overlay.on('pointermove', (ev) => {
      const [mx] = d3.pointer(ev);
      const i = Math.max(0, Math.min(pts.length - 1, Math.round(x.invert ? 0 : 0) || nearest(x, mx, pts.length)));
      hover(p, i, x(i), guide, svg);
    }).on('pointerleave', () => { guide.style('opacity', 0); tip.style.opacity = 0; });
  }
  function nearest(x, mx, n) {
    let best = 0, bd = 1e9;
    for (let i = 0; i < n; i++) { const d = Math.abs(x(i) - mx); if (d < bd) { bd = d; best = i; } }
    return best;
  }
  function hover(p, i, px, guide, svg) {
    const d = ENGINES[p.eng].points[i];
    guide.attr('x1', px).attr('x2', px).style('opacity', 1);
    tip.innerHTML = `<div style="font-weight:500">${d.label}</div>` +
      `<div class="text-small">${d.date} · ${d.commit}</div>` +
      ['solo', 'pair', 'conc'].map(l => `<div class="text-small tabular-nums">${LANE_NAMES[l]}: ${fmt(d[l])}</div>`).join('');
    tip.style.opacity = 1;
    const rr = root.getBoundingClientRect();
    tip.style.left = Math.min(rr.width - 190, Math.max(0, px + 12)) + 'px';
    tip.style.top = (svg.node().getBoundingClientRect().top - rr.top + 16) + 'px';
  }
  function drawPrefill() {
    const host = document.getElementById('lt-prefill-9f3');
    if (!host) return;
    host.textContent = '';
    const series = [];
    for (const eng of Object.keys(ENGINES)) {
      const pts = (ENGINES[eng].prefill || []).filter(p => p.tps != null && p.ctx > 0);
      if (pts.length) series.push({ eng, pts: [...pts].sort((a, b) => a.ctx - b.ctx) });
    }
    if (!series.length) return;
    const W = Math.max(320, host.clientWidth || 320);
    const H = Math.max(150, Math.min(230, W * 0.3));
    const M = { t: 12, r: 110, b: 32, l: 56 };
    const svg = d3.select(host).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('display', 'block')
      .attr('role', 'img').attr('aria-label', 'Prefill tok/s by prompt size');
    const all = series.flatMap(s => s.pts);
    const x = d3.scaleLog().domain(d3.extent(all, p => p.ctx)).range([M.l + 6, W - M.r - 6]).nice();
    const y = d3.scaleLinear().domain([0, d3.max(all, p => p.tps) * 1.12]).range([H - M.b, M.t]).nice();
    svg.append('rect').attr('data-chart-frame', true).attr('x', M.l).attr('y', M.t)
       .attr('width', Math.max(10, W - M.l - M.r)).attr('height', Math.max(10, H - M.t - M.b))
       .attr('fill', 'none').attr('stroke', 'var(--border)').attr('stroke-width', 1);
    const g = svg.append('g');
    g.append('g').attr('transform', `translate(${M.l},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(6))
      .call(sel => sel.select('.domain').remove())
      .selectAll('text').attr('fill', 'var(--muted-foreground)').style('font-size', '12px').attr('class', 'tabular-nums');
    g.append('g').selectAll('line.grid').data(y.ticks(4)).join('line')
      .attr('x1', M.l).attr('x2', W - M.r).attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', 'var(--border)').attr('stroke-width', 0.5).attr('opacity', 0.55);
    const kfmt = v => v >= 1e6 ? (v / 1e6).toFixed(v % 1e6 ? 1 : 0) + 'M' : Math.round(v / 1e3) + 'K';
    g.append('g').attr('transform', `translate(0,${H - M.b})`)
      .call(d3.axisBottom(x).tickValues([4e3, 1e4, 4e4, 1e5, 4e5, 1e6].filter(v => v >= x.domain()[0] * 0.9 && v <= x.domain()[1] * 1.1)).tickFormat(kfmt).tickSize(0).tickPadding(8))
      .call(sel => sel.select('.domain').attr('stroke', 'var(--border)'))
      .selectAll('text').attr('fill', 'var(--muted-foreground)').style('font-size', '11px');
    svg.append('text').attr('x', M.l).attr('y', H - 6).attr('fill', 'var(--muted-foreground)').style('font-size', '12px').text('prompt tokens (log)');
    svg.append('text').attr('data-axis', 'y').attr('x', 14).attr('y', M.t + 2).attr('fill', 'var(--muted-foreground)').style('font-size', '12px').text('tok/s');
    for (const s of series) {
      const col = COLORS[s.eng];
      const line = d3.line().x(p => x(p.ctx)).y(p => y(p.tps));
      g.append('path').datum(s.pts).attr('d', line)
        .attr('fill', 'none').attr('stroke', col).attr('stroke-width', 1.4).attr('stroke-dasharray', '4 3').attr('opacity', 0.7);
      g.selectAll(null).data(s.pts).join('circle')
        .attr('cx', p => x(p.ctx)).attr('cy', p => y(p.tps)).attr('r', 3.6).attr('fill', col)
        .attr('data-chart-hit', true)
        .append('title').text(p => `${p.label} (${p.commit}) ${p.date} · ${p.ctx.toLocaleString()} tok prompt · ${p.tps.toFixed(0)} tok/s [${p.kind}]`);
    }
    const key = svg.append('g').attr('transform', `translate(${W - M.r + 12},${M.t + 2})`);
    series.forEach((s, i) => {
      key.append('circle').attr('cx', 0).attr('cy', i * 15 + 3).attr('r', 3.6).attr('fill', COLORS[s.eng]);
      key.append('text').attr('x', 8).attr('y', i * 15 + 7).attr('fill', 'var(--muted-foreground)').style('font-size', '11px').text(ENAME[s.eng]);
    });
  }
  function drawAll() { panels.forEach(draw); drawPrefill(); }
  drawAll();
  if (window.ResizeObserver) new ResizeObserver(() => drawAll()).observe(panelsRoot);
})();
</script>
'''
if __name__ == '__main__':
    main()
