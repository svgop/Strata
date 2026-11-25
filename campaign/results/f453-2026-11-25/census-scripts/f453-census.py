#!/usr/bin/env python3
"""f453 in-window census -- read-only SQL over /tmp/f452-dev.sqlite (nsys 2026.3.2 SQLite export; node-level CUDA
graph trace, device graph launch ON).  CPU only.  Opens the database with mode=ro and writes nothing but
/tmp/f453/out.md (markdown fragments holding every number of the report).

Segments:  each `window_dev_launch_kernel` (one-node parent graph) is followed by exactly one execution of the
window graph; kernels are attached to the latest parent launch at or before their start.
"""
import bisect, collections, re, statistics as st, sqlite3

DB = 'file:/tmp/f452-dev.sqlite?mode=ro'
OUT = '/tmp/f453/out.md'
WARM_FRAC = 0.38
SLOW_PHASE_US = 1400          # same cut as nsys-anatomy.md (typical < 1.4 ms, slow >= 1.4 ms)

c = sqlite3.connect(DB, uri=True)
q = lambda s: c.execute(s).fetchall()
names = dict(q("SELECT id,value FROM StringIds"))
K = q("""SELECT start,end,streamId,graphId,graphNodeId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start""")
MC = q("SELECT start,end,streamId,graphNodeId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start")
MS = q("SELECT start,end,streamId,graphNodeId,bytes FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start")

PARENT = {7, 33, 41, 49, 60, 68}
WIN = {4, 30, 38, 46, 57, 65}
T_OF = {4: 1, 38: 2, 30: 3, 46: 4, 65: 5, 57: 6}
ROUND_G, STEP1 = 10, {13, 52, 71, 74, 77, 80}
STEP_ROLE = {10: 'round', 13: 'step 1', 52: 'step 1', 71: 'step 1', 74: 'step 1', 77: 'step 1', 80: 'step 1',
             16: 'step 2', 19: 'step 3', 22: 'step 4', 25: 'step 5'}
KIND = {1: 'H2D', 2: 'D2H', 8: 'D2D'}
SPIN = re.compile(r'wait|spin|flag|handshake|bell|doorbell|copy_i32|mapped|publish', re.I)


def short_name(v):
    v = re.sub(r'^void\s+', '', v)
    v = v.replace('strata::kernels::<unnamed>::', '').replace('strata::kernels::', '')
    v = re.sub(r'\((?!(?:int|bool)\)).*$', '', v, flags=re.S)      # parameter list (templates keep (int)/(bool))
    v = v.replace('(int)', '').replace('(bool)', '').replace('T1::F::', '').replace('T1::', '')
    return re.sub(r'\s+', '', v)


S, B = {}, {}
for k in K:
    d = k[5]
    if d not in S:
        S[d] = short_name(names[d]); B[d] = re.sub(r'<.*$', '', S[d])

# T of each window graph, read from the n_tok template argument of the hyper-connection read
for g in WIN:
    t = q(f"""SELECT k.demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE k.graphId={g}
              AND k.demangledName IN (SELECT id FROM StringIds WHERE value LIKE '%gr_down_v3_kernel%') LIMIT 1""")[0][0]
    m = re.search(r'<\(int\)1, \(int\)(\d+),', names[t])
    assert m and int(m.group(1)) == T_OF[g], (g, names[t])

# ------------------------------------------------------------------ instances
parents = [(k[0], k[1], k[3]) for k in K if k[3] in PARENT]
pstarts = [p[0] for p in parents]
kstarts = [k[0] for k in K]
mcstarts = [m[0] for m in MC]
inst = collections.defaultdict(list)
for k in K:
    if k[3] in WIN:
        inst[bisect.bisect_right(pstarts, k[0]) - 1].append(k)
N = len(parents)
I = []
for i in range(N):
    ks = inst[i]
    I.append(dict(i=i, ps=parents[i][0], pe=parents[i][1], ws=min(k[0] for k in ks), we=max(k[1] for k in ks), n=len(ks),
                  g=next(iter({k[3] for k in ks})), T=T_OF[next(iter({k[3] for k in ks}))]))
assert len(inst) == N and all(I[i]['we'] <= I[i + 1]['ps'] for i in range(N - 1))
n_warm = int(N * WARM_FRAC)
W0 = N - n_warm
WARM = list(range(W0, N))

out = []
P = lambda s='': out.append(s)
med, mean = (lambda xs: st.median(xs) if xs else float('nan')), (lambda xs: st.mean(xs) if xs else float('nan'))
def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')
def f(x, d=2): return f"{x:.{d}f}"
ms, us = (lambda ns: ns / 1e6), (lambda ns: ns / 1e3)
def stat(vals, unit='ms', d=3):
    v = [x / (1e6 if unit == 'ms' else 1e3) for x in vals]
    return f"median {f(med(v), d)} | mean {f(mean(v), d)} | p10 {f(pct(v, .1), d)} | p90 {f(pct(v, .9), d)} | min {f(min(v), d)} | max {f(max(v), d)} | {unit}"


def union(ivs):
    ivs = sorted(ivs); o = []
    for s, e in ivs:
        if o and s <= o[-1][1]:
            if e > o[-1][1]: o[-1][1] = e
        else:
            o.append([s, e])
    return o


def ulen(ivs): return sum(e - s for s, e in union(ivs))


def levels(by):
    """time with exactly 1 / 2 / 3 streams busy"""
    ev = []
    for ivs in by.values():
        for a, b in ivs: ev.append((a, 1)); ev.append((b, -1))
    ev.sort(); res = [0, 0, 0]; cur = 0; tp = None
    for t, d in ev:
        if tp is not None and cur: res[min(cur, 3) - 1] += t - tp
        cur += d; tp = t
    return res


def memcpys_in(a, b):
    return MC[bisect.bisect_left(mcstarts, a):bisect.bisect_right(mcstarts, b)]

# ================================================================== 1. segmentation
P(f"## S1")
P(f"total window-graph executions {N}; warm = last {int(WARM_FRAC*100)}% = {n_warm} (indices {W0}..{N-1}); cold {W0}")
cnt = collections.Counter((I[i]['g'], I[i]['n']) for i in range(N))
P("kernels per instance (graphId,T,kernels,executions): " + "; ".join(
    f"(g{g},T{T_OF[g]},{n},{c})" for (g, n), c in sorted(cnt.items(), key=lambda x: (T_OF[x[0][0]], x[0][1]))))
P(f"distinct nodes per graph: " + ", ".join(
    f"g{g}: {len({k[4] for k in inst[i]})}" for g in sorted(WIN, key=lambda g: T_OF[g]) for i in [[j for j in range(N) if I[j]['g'] == g][0]]))
P(f"instance overlap: {sum(1 for a, b in zip(I, I[1:]) if b['ws'] < a['we'])}; foreign kernels inside instances: " +
  str(sum(1 for i in range(N) for k in K[bisect.bisect_left(kstarts, I[i]['ws']):bisect.bisect_right(kstarts, I[i]['we'])] if k[3] != I[i]['g'])))
tw = collections.Counter(I[i]['T'] for i in WARM); ta = collections.Counter(I[i]['T'] for i in range(N))
P("warm T mix: " + ", ".join(f"T{t}={tw[t]}" for t in sorted(tw)) + f" (mean {f(sum(I[i]['T'] for i in WARM)/n_warm, 3)} tokens/window); all: " +
  ", ".join(f"T{t}={ta[t]}" for t in sorted(ta)))
P(f"parent kernel duration us: median {f(us(med([I[i]['pe']-I[i]['ps'] for i in WARM])),2)} mean {f(us(mean([I[i]['pe']-I[i]['ps'] for i in WARM])),2)}")
P(f"parent end -> first window kernel us: median {f(us(med([I[i]['ws']-I[i]['pe'] for i in WARM])),2)} mean {f(us(mean([I[i]['ws']-I[i]['pe'] for i in WARM])),2)} p90 {f(us(pct([I[i]['ws']-I[i]['pe'] for i in WARM], .9)),2)}")
wd = [I[i]['we'] - I[i]['ws'] for i in WARM]
dp = [I[i]['ps'] - I[i - 1]['we'] for i in WARM]
per = [I[i]['ws'] - I[i - 1]['ws'] for i in WARM]
P(f"A window-graph duration (first..last kernel): {stat(wd)}")
P(f"B drafter phase (prev window end -> parent launch start): {stat(dp)}")
P(f"C window-to-window period (first kernel -> first kernel): {stat(per)}")
lastk = {}; lead = []; body = []; trail = []; lastkind = collections.Counter()
for i in WARM:
    ks = K[bisect.bisect_right(kstarts, I[i - 1]['we']):bisect.bisect_left(kstarts, I[i]['ps'])]
    lastk[i] = max(k[1] for k in ks); lead.append(ks[0][0] - I[i - 1]['we']); body.append(lastk[i] - ks[0][0]); trail.append(I[i]['ps'] - lastk[i])
    ms_ = memcpys_in(I[i - 1]['we'], I[i]['ps'])
    e = max([(k[1], B[k[5]]) for k in ks[-1:]] + [(m[1], f"{KIND[m[5]]} memcpy {m[4]} B") for m in ms_])
    lastkind[e[1] if not e[1].startswith(('H2D', 'D2H')) else ('H2D memcpy' if e[1].startswith('H2D') else 'D2H memcpy')] += 1
gk = [I[i]['ws'] - lastk[i] for i in WARM]
P(f"D gap last drafter KERNEL end -> first window kernel: {stat(gk, 'us')}")
P(f"  D split: last drafter kernel -> parent launch start: {stat([I[i]['ps'] - lastk[i] for i in WARM], 'us')}")
P(f"A+B (mean) = {f(ms(mean(wd) + mean(dp)))} ms vs period mean {f(ms(mean(per)))} ms")
slow = [i for i in WARM if gk[i - W0] >= SLOW_PHASE_US * 1000]; typ = [i for i in WARM if gk[i - W0] < SLOW_PHASE_US * 1000]
P(f"boundary gap (D) typical (<{SLOW_PHASE_US/1000:.1f} ms): n={len(typ)} median {f(us(med([gk[i - W0] for i in typ])), 1)} us; slow (>={SLOW_PHASE_US/1000:.1f} ms): n={len(slow)}" +
  (f" median {f(us(med([gk[i - W0] for i in slow])), 0)} us, min {f(us(min([gk[i - W0] for i in slow])), 0)}, max {f(us(max([gk[i - W0] for i in slow])), 0)}" if slow else ""))
P(f"in the {len(slow)} slow-boundary windows the drafter phase is {stat([dp[i - W0] for i in slow]) if slow else 'n/a'}")
slow = [i for i in WARM if dp[i - W0] >= SLOW_PHASE_US * 1000]; typ = [i for i in WARM if dp[i - W0] < SLOW_PHASE_US * 1000]
P(f"drafter phase typical (<{SLOW_PHASE_US/1000:.1f} ms): n={len(typ)} phase {stat([I[i]['ps']-I[i-1]['we'] for i in typ])}; "
  f"last-kernel->first-window-kernel us: median {f(us(med([I[i]['ws']-lastk[i] for i in typ])),1)} mean {f(us(mean([I[i]['ws']-lastk[i] for i in typ])),1)}")
P(f"drafter phase slow (>={SLOW_PHASE_US/1000:.1f} ms): n={len(slow)} phase {stat([I[i]['ps']-I[i-1]['we'] for i in slow])}; "
  f"last-kernel->first-window-kernel us: median {f(us(med([I[i]['ws']-lastk[i] for i in slow])),1)} mean {f(us(mean([I[i]['ws']-lastk[i] for i in slow])),1)}")
P(f"drafter-phase decomposition us: window end -> first drafter kernel {stat(lead, 'us')}")
P(f"   first -> last drafter kernel {stat(body, 'us')}")
P(f"   last drafter kernel -> parent launch {stat(trail, 'us')}")
P("last device activity before the parent launch (warm): " + str(lastkind.most_common(6)))

# ================================================================== 2. inside the window graph
pw = {}
for i in WARM:
    ks = inst[i]; a, b = I[i]['ws'], I[i]['we']
    cnts = collections.Counter(k[2] for k in ks)
    main = max(cnts, key=lambda s: (cnts[s], -s)); sides = sorted(s for s in cnts if s != main)
    role = {main: 'main', sides[0]: 'side', sides[1]: 'side'}
    by = collections.defaultdict(list)
    for k in ks: by[role[k[2]]].append((k[0], k[1]))
    act = ulen([(k[0], k[1]) for k in ks]); wall = b - a
    l1, l2, l3 = levels(by)
    non = [(k[0], k[1]) for k in ks if not SPIN.search(S[k[5]])]
    spin = [(k[0], k[1]) for k in ks if SPIN.search(S[k[5]])]
    big = [(k[0], k[1]) for k in ks if k[1] - k[0] >= 16000]
    small = [(k[0], k[1]) for k in ks if k[1] - k[0] < 16000]
    mc = memcpys_in(a, b)
    U = ulen([(k[0], k[1]) for k in ks])
    pw[i] = dict(wall=wall, act=act, idle=wall - act, n=len(ks),
                 s={r: sum(e - s for s, e in v) for r, v in by.items()}, c={r: len(v) for r, v in by.items()}, l=[l1, l2, l3],
                 act_mc=ulen([(k[0], k[1]) for k in ks] + [(m[0], min(m[1], b)) for m in mc]),
                 only_spin=U - ulen(non), spin_plus=ulen(spin) - (U - ulen(non)), only_nonspin=U - ulen(spin),
                 only_big=U - ulen(small), big_plus=ulen(big) - (U - ulen(small)), only_small=U - ulen(big),
                 mc=mc, role=role)
P("## S2")
P(f"kernels per window: median {med([pw[i]['n'] for i in WARM])} mean {f(mean([pw[i]['n'] for i in WARM]), 1)} min {min(pw[i]['n'] for i in WARM)} max {max(pw[i]['n'] for i in WARM)}")
P(f"wall: {stat([pw[i]['wall'] for i in WARM])}")
P(f"active (union): {stat([pw[i]['act'] for i in WARM])}")
P(f"idle: {stat([pw[i]['idle'] for i in WARM])}")
P(f"idle share of wall: mean {f(100*mean([pw[i]['idle']/pw[i]['wall'] for i in WARM]), 2)}% median {f(100*med([pw[i]['idle']/pw[i]['wall'] for i in WARM]), 2)}% max {f(100*max(pw[i]['idle']/pw[i]['wall'] for i in WARM), 2)}%")
for r in ('main', 'side', 'side'):
    pass
P(f"main stream active: {stat([pw[i]['s']['main'] for i in WARM])}; kernels/window mean {f(mean([pw[i]['c']['main'] for i in WARM]), 1)}")
sa = []; sb = []
for i in WARM:
    v = sorted((s for s in pw[i]['role'] if pw[i]['role'][s] == 'side'))
    byst = collections.defaultdict(int)
    for k in inst[i]: byst[k[2]] += k[1] - k[0]
    sa.append(byst[v[0]]); sb.append(byst[v[1]])
P(f"side stream A (lower id) active: {stat(sa)}; kernels/window mean {f(sum(pw[i]['c']['side'] for i in WARM)/n_warm/2, 1)} each")
P(f"side stream B (higher id) active: {stat(sb)}")
ss = [sum(pw[i]['s'].values()) for i in WARM]
P(f"sum of per-stream active: {stat(ss)}")
P(f"overlap = sum per-stream - union: {stat([s - pw[i]['act'] for s, i in zip(ss, WARM)])}; as share of union: mean {f(100*mean([(s - pw[i]['act'])/pw[i]['act'] for s, i in zip(ss, WARM)]), 1)}%")
P(f"exactly 1 stream busy: {stat([pw[i]['l'][0] for i in WARM])}")
P(f"exactly 2 streams busy: {stat([pw[i]['l'][1] for i in WARM])}")
P(f"3 streams busy: {stat([pw[i]['l'][2] for i in WARM])}")
P(f"main-stream idle (wall - main active): {stat([pw[i]['wall'] - pw[i]['s']['main'] for i in WARM])}")
P(f"main-stream idle while a side stream runs (main idle - GPU idle): {stat([pw[i]['wall'] - pw[i]['s']['main'] - pw[i]['idle'] for i in WARM])}")
P(f"GPU idle with memcpy included: {stat([pw[i]['wall'] - pw[i]['act_mc'] for i in WARM])}")
mcw = [m for i in WARM for m in pw[i]['mc']]
P(f"memcpy inside the window interval (copy engine, not kernels): {f(len(mcw)/n_warm, 2)} per window, "
  f"{f(sum(m[4] for m in mcw)/n_warm/2**20, 2)} MiB per window, {f(sum(m[1]-m[0] for m in mcw)/n_warm/1e3, 1)} us of copy time per window; kinds " +
  str({KIND[k]: v for k, v in collections.Counter(m[5] for m in mcw).items()}) + "; in-graph memcpy nodes " + str(sum(1 for m in mcw if m[3] is not None)))
big_h2d = [m for m in mcw if m[5] == 1 and m[4] >= 2**20]
P(f"  H2D >= 1 MiB inside the window: {f(len(big_h2d)/n_warm, 2)} per window, median {f(med([m[4] for m in big_h2d])/2**20, 2)} MiB, median {f(us(med([m[1]-m[0] for m in big_h2d])), 1)} us each, "
  f"{f(sum(m[1]-m[0] for m in big_h2d)/n_warm/1e3, 1)} us per window, streams {sorted({m[2] for m in big_h2d})}, implied {f(med([m[4]/(m[1]-m[0]) for m in big_h2d]), 1)} GB/s")
P(f"memsets inside warm windows: {sum(1 for m in MS if any(I[i]['ws'] <= m[0] < I[i]['we'] for i in WARM))}")
P("### S2T per T (medians unless stated)")
P("| T | graph | kernels | warm n | wall | active | idle | idle mean | idle % of wall (mean) | main active | side A | side B | overlap | 1 stream | 2 streams | 3 streams | main idle | of which side stream running |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for g in sorted(WIN, key=lambda g: T_OF[g]):
    ws_ = [i for i in WARM if I[i]['g'] == g]
    if not ws_: continue
    p = [pw[i] for i in ws_]
    P(f"| {T_OF[g]} | {g} | {I[ws_[0]]['n']} | {len(ws_)} | {f(ms(med([x['wall'] for x in p])))} | {f(ms(med([x['act'] for x in p])))} | {f(ms(med([x['idle'] for x in p])))} | "
      f"{f(ms(mean([x['idle'] for x in p])))} | {f(100*mean([x['idle']/x['wall'] for x in p]), 1)} | {f(ms(med([x['s']['main'] for x in p])))} | "
      f"{f(ms(med([sa[WARM.index(i)] for i in ws_])))} | {f(ms(med([sb[WARM.index(i)] for i in ws_])))} | "
      f"{f(ms(med([sum(x['s'].values()) - x['act'] for x in p])))} | {f(ms(med([x['l'][0] for x in p])))} | {f(ms(med([x['l'][1] for x in p])))} | "
      f"{f(us(med([x['l'][2] for x in p])), 0)} | {f(ms(med([x['wall'] - x['s']['main'] for x in p])))} | {f(ms(med([x['wall'] - x['s']['main'] - x['idle'] for x in p])))} |")

# ================================================================== 3. top kernels + spin/wait
agg = collections.defaultdict(lambda: [0, 0, []]); aggr = collections.defaultdict(lambda: collections.Counter())
for i in WARM:
    for k in inst[i]:
        a = agg[k[5]]; a[0] += 1; a[1] += k[1] - k[0]; a[2].append(k[1] - k[0]); aggr[k[5]][pw[i]['role'][k[2]]] += 1
rows = sorted(agg.items(), key=lambda x: -x[1][1])
tot_all = sum(a[1] for _, a in rows)
wall_sum = sum(pw[i]['wall'] for i in WARM)
P("## S3")
P(f"distinct kernel variants (demangled, with template arguments) in warm windows: {len(rows)}; summed kernel duration over the 3 streams: {f(tot_all/n_warm/1e6)} ms per window")
P("| # | kernel | n/window | mean us | median us | p90 us | ms/window | % of summed kernel time | stream |")
P("|---|---|---|---|---|---|---|---|---|")
for r, (d, a) in enumerate(rows[:40], 1):
    sh = aggr[d]; where = 'main' if sh['main'] and not sh['side'] else ('sides' if sh['side'] and not sh['main'] else f"main {100*sh['main']/(sh['main']+sh['side']):.0f}%")
    P(f"| {r} | `{S[d]}` | {f(a[0]/n_warm, 1)} | {f(us(a[1]/a[0]))} | {f(us(med(a[2])))} | {f(us(pct(a[2], .9)))} | {f(a[1]/n_warm/1e6)} | {f(100*a[1]/tot_all, 1)} | {where} |")
P("### S3 spin/wait/handshake-named kernels (name matches wait|spin|flag|handshake|bell|doorbell|copy_i32|mapped|publish)")
P("| kernel | role by signature | n/window | mean us | median us | p90 us | max us | ms/window | % of summed kernel time | % of window wall | stream |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
ROLE = {'dm_wait_combine_kernel': 'wait for host-written flags + combine the CPU-pool expert rows (volatile float4/unsigned int args)',
        'wait_flag_ge_planned_kernel': 'spin until a host-written (volatile) flag reaches the planned value',
        'wait_flag_ge_or_kernel': 'spin until either of two host-written (volatile) flags reaches a value',
        'copy_i32_unless_kernel': 'copy ints from host-mapped (volatile) memory unless a skip flag is set',
        'doorbell_publish_res_kernel': 'publish the resident plan to the host (writes an int + flag)',
        'copy_from_mapped_kernel': 'copy float4 rows from host-mapped (volatile) memory',
        'copy_rows_or_zero_from_mapped_kernel': 'copy selected rows from host-mapped memory or zero them',
        'copy_i32_from_mapped_kernel': 'copy ints from host-mapped (volatile) memory'}
spin_tot = 0; spin_by_T = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, []]))
for i in WARM:
    for k in inst[i]:
        if SPIN.search(S[k[5]]):
            v = spin_by_T[I[i]['T']][B[k[5]]]; v[0] += 1; v[1] += k[1] - k[0]; v[2].append(k[1] - k[0])
for d, a in rows:
    if SPIN.search(S[d]):
        spin_tot += a[1]; sh = aggr[d]
        P(f"| `{S[d]}` | {ROLE.get(B[d], '')} | {f(a[0]/n_warm, 2)} | {f(us(a[1]/a[0]))} | {f(us(med(a[2])))} | {f(us(pct(a[2], .9)))} | {f(us(max(a[2])), 1)} | "
          f"{f(a[1]/n_warm/1e6, 3)} | {f(100*a[1]/tot_all, 1)} | {f(100*a[1]/wall_sum, 1)} | {'main' if sh['main'] and not sh['side'] else 'sides' if sh['side'] and not sh['main'] else 'both'} |")
pure = sum(a[1] for d, a in rows if B[d] in ('dm_wait_combine_kernel', 'wait_flag_ge_planned_kernel', 'wait_flag_ge_or_kernel'))
P(f"TOTAL spin/wait/handshake-named: {f(spin_tot/n_warm/1e6, 3)} ms per window = {f(100*spin_tot/tot_all, 1)}% of summed kernel duration = {f(100*spin_tot/wall_sum, 1)}% of window wall; "
  f"the three wait kernels alone: {f(pure/n_warm/1e6, 3)} ms")
P("per T (ms per window, then mean|median us): " + "; ".join(
    f"T{t}: total {f(sum(v[1] for v in spin_by_T[t].values())/tw[t]/1e6, 3)} [" + ", ".join(
        f"{b.replace('_kernel', '')} {f(v[1]/tw[t]/1e6, 3)} ({f(us(mean(v[2])), 1)}|{f(us(med(v[2])), 1)})"
        for b, v in sorted(spin_by_T[t].items(), key=lambda x: -x[1][1]) if v[1]/tw[t] > 2e5) + "]" for t in sorted(tw)))
P(f"spin/handshake-named time, union view: only such kernels running {stat([pw[i]['only_spin'] for i in WARM])}; "
  f"such kernels + other kernels running {stat([pw[i]['spin_plus'] for i in WARM])}")

# ================================================================== 4. duration distribution
edges = [0, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 512000, float('inf')]
lab = ['<=2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128', '128-512', '>512']
def bkt(d):   # upper edge inclusive: <=2, 2-4, 4-8, ... as labelled
    b = bisect.bisect_left(edges, d) - 1
    return min(max(b, 0), 8)
pb = {i: collections.defaultdict(lambda: [0, 0]) for i in WARM}; pbT = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for i in WARM:
    for k in inst[i]:
        d = k[1] - k[0]; b = bkt(d)
        pb[i][b][0] += 1; pb[i][b][1] += d; pbT[I[i]['T']][b][0] += 1; pbT[I[i]['T']][b][1] += d
ntot = sum(pb[i][b][0] for i in WARM for b in range(9))
P("## S4")
P("| bucket us | kernels per window | % of kernels | ms per window (summed over 3 streams) | % of summed kernel duration |")
P("|---|---|---|---|---|")
for b in range(9):
    n_ = sum(pb[i][b][0] for i in WARM) / n_warm; t_ = sum(pb[i][b][1] for i in WARM) / n_warm
    P(f"| {lab[b]} | {f(n_, 1)} | {f(100*n_/ntot*n_warm, 1)} | {f(ms(t_), 3)} | {f(100*t_*n_warm/tot_all, 1)} |")
def tsum(i, pred): return sum(pb[i][b][1] for b in range(9) if pred(b))
for thr, nm in ((4000, '<=4'), (8000, '<=8'), (16000, '<=16'), (16000, '>=16')):
    pred = (lambda b, thr=thr: edges[b + 1] <= thr) if nm.startswith('<=') else (lambda b: edges[b] >= 16000)
    fr = [tsum(i, pred) / pw[i]['wall'] for i in WARM]
    P(f"kernels {nm} us: summed {f(sum(tsum(i, pred) for i in WARM)/n_warm/1e6, 3)} ms per window = {f(100*mean(fr), 1)}% of window wall "
      f"(ratio of per-window sums; median {f(100*med(fr), 1)}%), = {f(100*sum(tsum(i, pred) for i in WARM)/tot_all, 1)}% of summed kernel duration")
P("per T, share of window wall taken by kernels <=4 / <=8 / <=16 us (summed durations / wall, means): " + "; ".join(
    f"T{t} {f(100*sum(pbT[t][b][1] for b in range(9) if edges[b+1] <= 4000)/sum(pw[i]['wall'] for i in WARM if I[i]['T']==t),1)}"
    f"/{f(100*sum(pbT[t][b][1] for b in range(9) if edges[b+1] <= 8000)/sum(pw[i]['wall'] for i in WARM if I[i]['T']==t),1)}"
    f"/{f(100*sum(pbT[t][b][1] for b in range(9) if edges[b+1] <= 16000)/sum(pw[i]['wall'] for i in WARM if I[i]['T']==t),1)}"
    for t in sorted(tw)))
P(f"union view of the window wall (additive): only kernels >=16 us running {stat([pw[i]['only_big'] for i in WARM])}; "
  f">=16 us and <16 us kernels running together {stat([pw[i]['big_plus'] for i in WARM])}; only kernels <16 us running {stat([pw[i]['only_small'] for i in WARM])}; idle {stat([pw[i]['idle'] for i in WARM])}")

# ================================================================== 5. idle-gap census
P_GAPCOUNT = collections.Counter()
gb = [0, 1000, 2000, 4000, 8000, 16000, 64000, float('inf')]
glab = ['<=1', '1-2', '2-4', '4-8', '8-16', '16-64', '>64']
def gbkt(L):  # upper edge inclusive: <=1, 1-2, 2-4, ...
    b = bisect.bisect_left(gb, L) - 1
    return min(b, 6)
gapb = {i: collections.defaultdict(lambda: [0, 0]) for i in WARM}; gapcls = collections.defaultdict(lambda: [0, 0, []])
longcls = collections.defaultdict(lambda: [0, 0, []]); gapb_cls = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, []]))
gapbT = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for i in WARM:
    ks = inst[i]; role = pw[i]['role']
    em = collections.defaultdict(list); sm = collections.defaultdict(list)
    for k in ks:
        em[k[1]].append(B[k[5]] + '[' + role[k[2]] + ']'); sm[k[0]].append(B[k[5]] + '[' + role[k[2]] + ']')
    u = union([(k[0], k[1]) for k in ks])
    for (a, b), (cc, d) in zip(u, u[1:]):
        L = cc - b; bb = gbkt(L)
        gapb[i][bb][0] += 1; gapb[i][bb][1] += L; gapbT[I[i]['T']][bb][0] += 1; gapbT[I[i]['T']][bb][1] += L
        key = ('+'.join(sorted(set(em.get(b, ['?'])))), '+'.join(sorted(set(sm.get(cc, ['?'])))))
        v = gapcls[key]; v[0] += 1; v[1] += L; v[2].append(L)
        w = gapb_cls[bb][key]; w[0] += 1; w[1] += L; w[2].append(L)
        if L > 1000:
            w = longcls[key]; w[0] += 1; w[1] += L; w[2].append(L)
tot_gap = sum(gapb[i][b][1] for i in WARM for b in range(7)); tot_gn = sum(gapb[i][b][0] for i in WARM for b in range(7))
P("## S5")
P(f"gaps on the union timeline per window: {f(tot_gn/n_warm, 1)}; idle {f(ms(tot_gap/n_warm), 3)} ms per window")
P("| gap length us | gaps per window | gaps in 293 windows | % of gaps | idle ms per window | % of idle | mean gap us |")
P("|---|---|---|---|---|---|---|")
for b in range(7):
    n_ = sum(gapb[i][b][0] for i in WARM) / n_warm; t_ = sum(gapb[i][b][1] for i in WARM) / n_warm
    P_GAPCOUNT[b] = sum(gapb[i][b][0] for i in WARM)
    P(f"| {glab[b]} | {f(n_, 2)} | {P_GAPCOUNT[b]:,} | {f(100*n_/tot_gn*n_warm, 1)} | {f(ms(t_), 4)} | {f(100*t_*n_warm/tot_gap, 1)} | {f(us(t_/n_), 2) if n_ else '0'} |")
P("per T, idle ms per window by gap bucket (" + "/".join(glab) + "): " + "; ".join(
    f"T{t} " + "/".join(f(ms(gapbT[t][b][1] / tw[t]), 3) for b in range(7)) for t in sorted(tw)))
def gap_table(items, title, tot):
    P(f"### {title}")
    P("| # | kernel ending the busy period -> kernel(s) starting the next one [stream] | occurrences per window | idle ms per window | % of idle | median gap us | p90 us | max us |")
    P("|---|---|---|---|---|---|---|---|")
    for r, (key, v) in enumerate(items, 1):
        P(f"| {r} | `{key[0]}` -> `{key[1]}` | {f(v[0]/n_warm, 2)} | {f(v[1]/n_warm/1e6, 4)} | {f(100*v[1]/tot, 1)} | {f(us(med(v[2])), 2)} | {f(us(pct(v[2], .9)), 1)} | {f(us(max(v[2])), 1)} |")
top = sorted(gapcls.items(), key=lambda x: -x[1][1])
gap_table(top[:10], "S5a ten largest gap classes by idle time (all gap lengths)", tot_gap)
big_cls = sorted(longcls.items(), key=lambda x: -x[1][1])
P(f"gaps > 1 us per window: {f(sum(gapb[i][b][0] for i in WARM for b in range(1, 7))/n_warm, 1)}; idle in them {f(ms(sum(gapb[i][b][1] for i in WARM for b in range(1, 7))/n_warm), 4)} ms per window; "
  f"gaps > 64 us per window: {f(sum(gapb[i][6][0] for i in WARM)/n_warm, 2)}; idle in them {f(ms(sum(gapb[i][6][1] for i in WARM)/n_warm), 4)} ms")
tot_long = sum(v[1] for v in longcls.values())
gap_table(big_cls[:10], "S5b ten largest classes among gaps longer than 1 us (only those long occurrences; % of all idle)", tot_gap)
P(f"classes of gaps > 1 us: {len(longcls)}; the ten listed cover {f(100*sum(v[1] for _, v in big_cls[:10])/tot_long, 1)}% of the >1 us idle")
P("by bucket, the kernel before -> after of the largest idle (each bucket: class, idle ms per window): " + " || ".join(
    f"{glab[b]}: " + "; ".join(f"`{k[0]}`->`{k[1]}` {f(v[1]/n_warm/1e6, 4)}" for k, v in sorted(
        ((k, v) for k, v in gapb_cls[b].items()), key=lambda x: -x[1][1])[:2]) for b in range(1, 7)))
P(f"distinct gap classes {len(gapcls)}; top-10 cover {f(100*sum(v[1] for _, v in top[:10])/tot_gap, 1)}% of the idle")

# ================================================================== 6. drafter phase
dh = {}
for i in WARM:
    a0, a1 = I[i - 1]['we'], I[i]['ps']
    ks = K[bisect.bisect_right(kstarts, a0):bisect.bisect_left(kstarts, a1)]
    mcl = memcpys_in(a0, a1)
    acts = [(k[0], k[1], 'K', B[k[5]], k[3]) for k in ks] + [(m[0], min(m[1], a1), 'C', f"{KIND[m[5]]} memcpy {m[4]} B" if m[4] < 1024 else f"{KIND[m[5]]} memcpy {m[4] // 1024} KiB", None) for m in mcl]
    acts.sort()
    dh[i] = dict(dur=a1 - a0, n=len(ks), act=ulen([(k[0], k[1]) for k in ks]), act_all=ulen([(x[0], x[1]) for x in acts]), acts=acts,
                 mc=mcl, cross=sum(1 for m in mcl if m[1] > a1))
P("## S6")
P(f"phase duration (prev window end -> parent launch): {stat([dh[i]['dur'] for i in WARM])}")
P(f"kernels per phase: median {med([dh[i]['n'] for i in WARM])} mean {f(mean([dh[i]['n'] for i in WARM]), 1)} min {min(dh[i]['n'] for i in WARM)} max {max(dh[i]['n'] for i in WARM)}")
P(f"active (union of kernels): {stat([dh[i]['act'] for i in WARM])}")
P(f"idle (kernels only): {stat([dh[i]['dur'] - dh[i]['act'] for i in WARM])}")
P(f"active (union incl. memcpys): {stat([dh[i]['act_all'] for i in WARM])}")
P(f"idle incl. memcpys: {stat([dh[i]['dur'] - dh[i]['act_all'] for i in WARM])}")
P(f"idle share of phase (kernels only): mean {f(100*mean([(dh[i]['dur']-dh[i]['act'])/dh[i]['dur'] for i in WARM]), 1)}% median {f(100*med([(dh[i]['dur']-dh[i]['act'])/dh[i]['dur'] for i in WARM]), 1)}%")
allmc = [m for i in WARM for m in dh[i]['mc']]
P(f"memcpy in the phase: {f(len(allmc)/n_warm, 2)} per phase; by kind " + str({KIND[k]: (f(v/n_warm, 2), f(sum(m[4] for m in allmc if m[5]==k)/n_warm/2**20, 2)) for k, v in collections.Counter(m[5] for m in allmc).items()}) +
  " (count per phase, MiB per phase); copy time per phase " + str({KIND[k]: f(sum(m[1]-m[0] for m in allmc if m[5]==k)/n_warm/1e3, 1) for k in {m[5] for m in allmc}}) +
  " us; memcpys still running at the parent launch: " + str(sum(dh[i]['cross'] for i in WARM)))
h2d = [m for m in allmc if m[5] == 1 and m[4] >= 2**20]; d2h = [m for m in allmc if m[5] == 2]
P(f"  H2D >= 1 MiB: {f(len(h2d)/n_warm, 2)} per phase, {f(sum(m[4] for m in h2d)/n_warm/2**20, 2)} MiB per phase, median {f(us(med([m[1]-m[0] for m in h2d])), 1)} us each, median size {f(med([m[4] for m in h2d])/2**20, 2)} MiB, streams {sorted({m[2] for m in h2d})}; "
  f"D2H: {f(len(d2h)/n_warm, 2)} per phase, {f(sum(m[4] for m in d2h)/n_warm/2**20, 2)} MiB per phase, median {f(us(med([m[1]-m[0] for m in d2h])), 1)} us each, sizes {sorted({m[4] for m in d2h})[:6]}")
dcls = collections.defaultdict(lambda: [0, 0, []]); maxgap = {}
for i in WARM:
    acts = dh[i]['acts']; a0 = I[i - 1]['we']; a1 = I[i]['ps']
    em = collections.defaultdict(list); sm = collections.defaultdict(list)
    for x in acts: em[x[1]].append(x[3]); sm[x[0]].append(x[3])
    u = union([(x[0], x[1]) for x in acts])
    if u and u[0][0] > a0:   # leading gap: previous window graph end -> first drafter activity
        v = dcls[('[previous window graph end]', '+'.join(sorted(set(sm.get(u[0][0], ['?'])))))]; v[0] += 1; v[1] += u[0][0] - a0; v[2].append(u[0][0] - a0)
    for (a, b), (cc, d) in zip(u, u[1:]):
        L = cc - b
        before = '[previous window graph end]' if b == a0 else '+'.join(sorted(set(em.get(b, ['?']))))
        after = '[parent launch kernel]' if cc == a1 else '+'.join(sorted(set(sm.get(cc, ['?']))))
        v = dcls[(before, after)]; v[0] += 1; v[1] += L; v[2].append(L)
    if u and u[-1][1] < a1:
        v = dcls[('+'.join(sorted(set(em.get(u[-1][1], ['?'])))), '[parent launch kernel]')]; v[0] += 1; v[1] += a1 - u[-1][1]; v[2].append(a1 - u[-1][1])
    maxgap[i] = max([cc - b for (a, b), (cc, d) in zip(u, u[1:])] + [a1 - u[-1][1] if u else 0])
tot_dg = sum(v[1] for v in dcls.values())
P(f"idle (incl. memcpys) in the phase {f(ms(tot_dg/n_warm), 3)} ms per phase; gaps per phase {f(sum(v[0] for v in dcls.values())/n_warm, 1)}")
P("| # | before -> after the idle interval | occurrences per phase | idle ms per phase | % of drafter idle | median us | p90 us | max us |")
P("|---|---|---|---|---|---|---|---|")
dtop = sorted(dcls.items(), key=lambda x: -x[1][1])
for r, (key, v) in enumerate(dtop[:14], 1):
    P(f"| {r} | `{key[0]}` -> `{key[1]}` | {f(v[0]/n_warm, 2)} | {f(v[1]/n_warm/1e6, 3)} | {f(100*v[1]/tot_dg, 1)} | {f(us(med(v[2])), 1)} | {f(us(pct(v[2], .9)), 1)} | {f(us(max(v[2])), 0)} |")
P(f"distinct idle classes {len(dcls)}; top-14 cover {f(100*sum(v[1] for _, v in dtop[:14])/tot_dg, 1)}%")
longg = [i for i in WARM if dh[i]['dur'] - dh[i]['act_all'] > SLOW_PHASE_US * 1000]
P(f"phases whose idle (incl. memcpy) exceeds {SLOW_PHASE_US} us: {len(longg)}; their idle {stat([dh[i]['dur'] - dh[i]['act_all'] for i in longg])}")
P(f"phases with an idle gap > 1 ms: {sum(1 for i in WARM if maxgap[i] > 1e6)} of {n_warm}; their phase duration {stat([dh[i]['dur'] for i in WARM if maxgap[i] > 1e6])}")
P(f"leading gap (window graph end -> first drafter kernel) > 100 us: {sum(1 for i in WARM if lead[i - W0] > 100e3)} phases")
tr_ = [I[i]['ps'] - lastk[i] for i in WARM]
for cut in (100e3, 500e3, 1400e3):
    a_ = [x for x in tr_ if x < cut]; b_ = [x for x in tr_ if x >= cut]
    P(f"trailing host gap (last drafter kernel -> parent launch) < {cut/1e3:.0f} us: n={len(a_)} median {f(us(med(a_)), 1)} mean {f(us(mean(a_)), 1)} us; >= {cut/1e3:.0f} us: n={len(b_)}" +
      (f" median {f(us(med(b_)), 0)} mean {f(us(mean(b_)), 0)} us" if b_ else ""))
eag = collections.defaultdict(lambda: [0, 0, []]); grp = collections.defaultdict(lambda: [0, 0, []])
for i in WARM:
    for x in dh[i]['acts']:
        if x[2] != 'K': continue
        v = (grp if x[4] is not None else eag)[x[3]]; v[0] += 1; v[1] += x[1] - x[0]; v[2].append(x[1] - x[0])
P("eager (non-graph) kernels: " + "; ".join(f"`{n}` {f(v[0]/n_warm, 2)}/phase, median {f(us(med(v[2])), 1)} us, {f(v[1]/n_warm/1e3, 1)} us/phase" for n, v in sorted(eag.items(), key=lambda x: -x[1][1])))
P("graph kernels by name (top 12 by time): " + "; ".join(f"`{n}` {f(v[0]/n_warm, 1)}/phase, median {f(us(med(v[2])), 1)} us, {f(v[1]/n_warm/1e3, 1)} us/phase" for n, v in sorted(grp.items(), key=lambda x: -x[1][1])[:12]))
gi = collections.defaultdict(lambda: [0, 0, [], 0]); seqs = collections.Counter(); tseq = collections.defaultdict(collections.Counter); chain = []; tg = collections.Counter()
for i in WARM:
    by = collections.defaultdict(list)
    for x in dh[i]['acts']:
        if x[2] == 'K' and x[4] is not None: by[x[4]].append(x)
    order = sorted(by, key=lambda g: min(x[0] for x in by[g])); seqs[tuple(order)] += 1
    tseq[sum(1 for g in order if g != ROUND_G)][I[i]['T']] += 1
    for g in order:
        d_ = max(x[1] for x in by[g]) - min(x[0] for x in by[g]); v = gi[g]; v[0] += 1; v[1] += d_; v[2].append(d_); v[3] += len(by[g])
    steps = [g for g in order if g != ROUND_G]
    chain.append((max(x[1] for g in steps for x in by[g]) - min(x[0] for g in steps for x in by[g])) if steps else 0)
    tg[(len(steps), I[i]['T'])] += 1
P("graph sequence census (graphIds in start order: occurrences): " + "; ".join(f"{seq}: {n}" for seq, n in seqs.most_common()))
P("| graph | role | nodes | phases | GPU duration median us | mean us | p90 us | min us | max us | ms per phase |")
P("|---|---|---|---|---|---|---|---|---|---|")
for g in sorted(gi, key=lambda g: (g != 10, STEP_ROLE.get(g, '?'), g)):
    v = gi[g]
    P(f"| {g} | {STEP_ROLE.get(g)} | {f(v[3]/v[0], 0)} | {v[0]} | {f(us(med(v[2])), 1)} | {f(us(mean(v[2])), 1)} | {f(us(pct(v[2], .9)), 1)} | {f(us(min(v[2])), 1)} | {f(us(max(v[2])), 0)} | {f(v[1]/n_warm/1e6, 3)} |")
P(f"step chain (first step-graph kernel -> last step-graph kernel end): {stat(chain, 'us')}")
P("step graphs in the phase (rows) vs T of the window that follows (columns): " + "; ".join(
    f"{k} steps: " + ", ".join(f"T{t}={k2}" for t, k2 in sorted(tseq[k].items())) for k in sorted(tseq)))
P(f"summed drafter graph-kernel duration: {f(sum(v[1] for v in gi.values())/n_warm/1e6, 3)} ms per phase; all drafter kernels summed: {f(sum(v[1] for v in list(grp.values()) + list(eag.values()))/n_warm/1e6, 3)} ms")

# ================================================================== 7. stage mapping
DENSE = {'native_q6_k_mmvq_kernel', 'native_q5_k_mmvq_kernel', 'native_q4_k_mmvq_kernel', 'native_iq4_xs_mmvq_kernel',
         'native_small_mmvq_kernel', 'native_mmvq_multi_kernel', 'native_mmvq_il_kernel', 'bf16_f32_mmvf_kernel', 'bf16_f32_mmvf_multi_kernel'}
CLS = [
    ('host-handshake/spin/copy-to-from-mapped', lambda b: b in {
        'dm_wait_combine_kernel', 'wait_flag_ge_planned_kernel', 'wait_flag_ge_or_kernel', 'copy_i32_unless_kernel', 'copy_i32_from_mapped_kernel',
        'doorbell_publish_res_kernel', 'copy_from_mapped_kernel'}),
    ('CPU-expert combine/copy (T=1 path: mapped rows, hits, k10 combine)', lambda b: b in {
        'copy_rows_or_zero_from_mapped_kernel', 'add_hits_kernel', 'combine_k10_vec4'}),
    ('router/top-k/plan', lambda b: b in {'route', 'route_multi', 'route_resident_k', 'resident_plan_kernel'}),
    ('MoE routed-expert compute (grouped gate/up + down)', lambda b: b in {'native_gu_swiglu_kernel', 'native_down_multi_kernel'}),
    ('attention: GDN (recurrence/conv/norm)', lambda b: b in {'gdn_ab_multi_kernel', 'gdn_conv_l2_multi_kernel', 'gdn_step_norm_multi_kernel', 'conv_residual_kernel'}),
    ('attention: QSA (sparse attn, indexer/select, KV gather/append, rope, out-gate)', lambda b: b in {
        'decode_attn_i8_split_kernel', 'decode_attn_merge_kernel', 'block_scores_multi_kernel', 'block_topk_cluster_kernel', 'resolve_kernel', 'copy_kernel',
        'kv_append_q8_kernel', 'append', 'norm_rope_kernel', 'gate'}),
    ('hyper-connections (HC stream read/write)', lambda b: b in {'gr_down_v3_kernel', 'gr_up_v3_kernel', 'broadcast_streams_kernel'}),
    ('shared expert (gate scale by name; matmuls resolved by position)', lambda b: b == 'sigmoid_scale_rows_vec4_kernel'),
    ('normalization/quant', lambda b: b in {'native_quantize_q8_1_kernel', 'quantize_q8_1_kernel', 'native_q8_1_interleave_kernel',
                                            'native_swiglu_quantize_q8_1_kernel', 'weighted_rms_norm'}),
    ('head/argmax/sampling', lambda b: b in {'sampler_greedy_cluster_kernel', 'logits_nonfinite_kernel'}),
    ('control vector (cvec)', lambda b: b == 'cvec_kernel'),
    ('dense matmul, stage NOT identifiable by name (native mmvq / bf16 mmvf)',
     lambda b: b in {'native_q6_k_mmvq_kernel', 'native_q5_k_mmvq_kernel', 'native_q4_k_mmvq_kernel', 'native_iq4_xs_mmvq_kernel',
                     'native_small_mmvq_kernel', 'native_mmvq_multi_kernel', 'native_mmvq_il_kernel', 'bf16_f32_mmvf_kernel',
                     'bf16_f32_mmvf_multi_kernel'}),
    ('other (embedding, per-token tail block: gate/broadcast/history/post)', lambda b: True),
]
def classify(b):
    for name, fn in CLS:
        if fn(b): return name

def dense_sub(ks, i):
    """position-informed sub-class for the dense-matmul family (graph position, not the name)"""
    cnts = collections.Counter(k[2] for k in ks); main = max(cnts, key=lambda s: (cnts[s], -s))
    maink = sorted((k for k in ks if k[2] == main))
    hc = [k for k in maink if B[k[5]] == 'gr_down_v3_kernel']
    hcs = [k[0] for k in hc]
    comb = [k for k in maink if B[k[5]] in ('dm_wait_combine_kernel', 'combine_k10_vec4')]
    cfm = [k for k in maink if B[k[5]] == 'copy_from_mapped_kernel' and comb and k[0] > comb[0][1]]
    tail = (cfm[0][0], hc[2][0]) if (cfm and len(hc) > 2) else (None, None)
    seg_gdn = collections.defaultdict(bool)
    for k in ks:
        j = bisect.bisect_right(hcs, k[0]) - 1
        if j >= 0 and B[k[5]].startswith('gdn_'): seg_gdn[j] = True
    per = collections.defaultdict(list)
    for k in ks: per[k[2]].append(k)
    succ = {a: b for l in per.values() for a, b in zip(l, l[1:])}
    res = {}
    for k in ks:
        if B[k[5]] not in DENSE: continue
        nb = B[succ[k][5]] if k in succ else None
        j = bisect.bisect_right(hcs, k[0]) - 1
        if tail[0] is not None and tail[0] <= k[0] < tail[1]:
            nm = 'per-token tail block (two bf16 matvecs after the layer-0 combine)'
        elif nb == 'sampler_greedy_cluster_kernel':
            nm = 'LM head (matmul feeding the greedy sampler)'
        elif j < 0:
            nm = 'unpositioned'
        elif j == len(hcs) - 1:
            nm = 'final segment before the LM head'
        elif j % 2 == 0:
            nm = 'attention segment, GDN layers (projections)' if seg_gdn[j] else 'attention segment, QSA layers (QKV/out projections)'
        else:
            nm = 'router GEMM (feeds route/route_multi)' if nb in ('route', 'route_multi') else 'MoE block, non-router matmuls (shared-expert position)'
        res[k] = nm
    return res

cls = collections.defaultdict(lambda: [0, 0, 0]); clsbn = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
sub = collections.defaultdict(lambda: [0, 0]); subn = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
merged = collections.defaultdict(lambda: [0, 0, 0]); mergedT = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
posbad = 0
for i in WARM:
    ks = inst[i]; ds = dense_sub(ks, i)
    for k in ks:
        d = k[1] - k[0]; b = B[k[5]]; nm = classify(b); main = pw[i]['role'][k[2]] == 'main'
        cls[nm][0] += 1; cls[nm][1] += d; cls[nm][2] += d if main else 0; clsbn[nm][b][0] += 1; clsbn[nm][b][1] += d
        if b in DENSE:
            s_ = ds[k]; sub[s_][0] += 1; sub[s_][1] += d; subn[s_][b][0] += 1; subn[s_][b][1] += d
            posbad += s_ == 'unpositioned'
            m = {'attention segment, GDN layers (projections)': 'attention: GDN (recurrence/conv/norm)',
                 'attention segment, QSA layers (QKV/out projections)': 'attention: QSA (sparse attn, indexer/select, KV gather/append, rope, out-gate)',
                 'router GEMM (feeds route/route_multi)': 'router/top-k/plan',
                 'MoE block, non-router matmuls (shared-expert position)': 'shared expert (gate scale by name; matmuls resolved by position)',
                 'LM head (matmul feeding the greedy sampler)': 'head/argmax/sampling'}.get(s_, nm)
        else:
            m = nm
        merged[m][0] += 1; merged[m][1] += d; merged[m][2] += d if main else 0
        mergedT[m][I[i]['T']][0] += 1; mergedT[m][I[i]['T']][1] += d
P("## S7")
P(f"window wall summed over warm windows {f(ms(wall_sum), 1)} ms; summed kernel duration {f(ms(tot_all), 1)} ms; unpositioned dense matmuls {posbad}")
P("### S7a by name only")
P("| class | ms per window (3 streams summed) | % of window wall | % of summed kernel duration | of which on the main stream, ms | kernels per window |")
P("|---|---|---|---|---|---|")
for name, _ in CLS:
    v = cls[name]
    P(f"| {name} | {f(v[1]/n_warm/1e6, 3)} | {f(100*v[1]/wall_sum, 1)} | {f(100*v[1]/tot_all, 1)} | {f(v[2]/n_warm/1e6, 3)} | {f(v[0]/n_warm, 1)} |")
P("PCIe staging/fetch: no kernel name in the window graphs denotes PCIe staging or fetch (0 ms); the expert-weight transfers are DMA memcpys (section 2).")
for name, _ in CLS:
    P(f"- {name}: " + "; ".join(f"`{b}` {f(v[1]/n_warm/1e6, 3)} ms (n {f(v[0]/n_warm, 1)})" for b, v in sorted(clsbn[name].items(), key=lambda x: -x[1][1])))
P("### S7b dense matmuls resolved by graph position (inference from the surrounding nodes, not from the name)")
P("| sub-class | ms per window | % of window wall | kernels per window | members (ms per window) |")
P("|---|---|---|---|---|")
for nm, v in sorted(sub.items(), key=lambda x: -x[1][1]):
    P(f"| {nm} | {f(v[1]/n_warm/1e6, 3)} | {f(100*v[1]/wall_sum, 1)} | {f(v[0]/n_warm, 1)} | " +
      "; ".join(f"`{b}` {f(vv[1]/n_warm/1e6, 3)}" for b, vv in sorted(subn[nm].items(), key=lambda x: -x[1][1])) + " |")
P("### S7c merged stages (name + position), ms per window summed over 3 streams")
P("| stage | all warm, 3 streams | of which main stream | " + " | ".join(f"T{t} (n={tw[t]})" for t in sorted(tw)) + " |")
P("|---|---|---|" + "---|" * len(tw))
tot_row = [0.0] * (len(tw) + 1)
for nm, v in sorted(merged.items(), key=lambda x: -x[1][1]):
    cells = [f(v[1]/n_warm/1e6, 3), f(v[2]/n_warm/1e6, 3)] + [f(mergedT[nm][t][1]/tw[t]/1e6, 3) for t in sorted(tw)]
    for j, x in enumerate([cells[0]] + cells[2:]): tot_row[j] += float(x)
    P(f"| {nm} | " + " | ".join(cells) + " |")
P("| summed kernel duration | " + " | ".join(f(x, 3) for x in tot_row) + " |")
P(f"main-stream kernels per window: {f(mean([pw[i]['c']['main'] for i in WARM]), 1)} of {f(mean([pw[i]['n'] for i in WARM]), 1)}; main-stream total {f(ms(mean([pw[i]['s']['main'] for i in WARM])), 3)} ms")
P("| window wall | " + " | ".join([f(ms(mean([pw[i]['wall'] for i in WARM])), 3)] + [f(ms(mean([pw[i]['wall'] for i in WARM if I[i]['T'] == t])), 3) for t in sorted(tw)]) + " |")
P("| GPU idle (union) | " + " | ".join([f(ms(mean([pw[i]['idle'] for i in WARM])), 3)] + [f(ms(mean([pw[i]['idle'] for i in WARM if I[i]['T'] == t])), 3) for t in sorted(tw)]) + " |")
P("| drafter phase | " + " | ".join([f(ms(mean([dh[i]['dur'] for i in WARM])), 3)] + [f(ms(mean([dh[i]['dur'] for i in WARM if I[i]['T'] == t])), 3) for t in sorted(tw)]) + " |")
P("| wall + preceding drafter phase | " + " | ".join([f(ms(mean(per)), 3)] + [f(ms(mean([I[i]['ws'] - I[i-1]['ws'] for i in WARM if I[i]['T'] == t])), 3) for t in sorted(tw)]) + " |")
P("| tokens per second (T / (wall + drafter phase)) | " + " | ".join([f(sum(I[i]['T'] for i in WARM)/sum(per)*1e9, 1)] + [f(t/(mean([pw[i]['wall'] + dh[i]['dur'] for i in WARM if I[i]['T'] == t]))*1e9, 1) for t in sorted(tw)]) + " |")

# ================================================================== 8. budget
P("## S8 means per warm window")
mwall, midle, mact = ms(mean([pw[i]['wall'] for i in WARM])), ms(mean([pw[i]['idle'] for i in WARM])), ms(mean([pw[i]['act'] for i in WARM]))
mmain = ms(mean([pw[i]['s']['main'] for i in WARM])); mside_only = ms(mean([pw[i]['wall'] - pw[i]['s']['main'] - pw[i]['idle'] for i in WARM]))
P(f"window graph wall {f(mwall, 3)} ms = active {f(mact, 3)} + idle {f(midle, 3)}; main-stream kernels {f(mmain, 3)}; side-stream-only {f(mside_only, 3)}")
big16 = ms(sum(tsum(i, lambda b: edges[b] >= 16000) for i in WARM) / n_warm); mid416 = ms(sum(tsum(i, lambda b: b >= 2 and b <= 3) for i in WARM) / n_warm)
lt4 = ms(sum(tsum(i, lambda b: b <= 1) for i in WARM) / n_warm)
P(f"summed kernel duration: >=16 us {f(big16, 3)}; 4-16 us {f(mid416, 3)}; <=4 us {f(lt4, 3)} ms")
P(f"union view: only >=16 us kernels {f(ms(mean([pw[i]['only_big'] for i in WARM])), 3)}; >=16 with <16 {f(ms(mean([pw[i]['big_plus'] for i in WARM])), 3)}; "
  f"only <16 us kernels {f(ms(mean([pw[i]['only_small'] for i in WARM])), 3)}; idle {f(midle, 3)}")
mb = collections.defaultdict(lambda: [0, 0, 0, 0])   # main-stream kernels by bucket: n, total, per-window sums
for i in WARM:
    for k in inst[i]:
        if pw[i]['role'][k[2]] != 'main': continue
        b = bkt(k[1] - k[0]); mb[b][0] += 1; mb[b][1] += k[1] - k[0]
P("main stream only, summed by bucket (ms per window): " + "; ".join(f"{lab[b]} {f(mb[b][1]/n_warm/1e6, 3)}" for b in range(9)))
msp = ms(sum(v[1] for nm, v in cls.items() if nm.startswith('host-handshake') for _ in [0]) * 0 + cls['host-handshake/spin/copy-to-from-mapped'][2] / n_warm)
P(f"handshake/spin class: {f(cls['host-handshake/spin/copy-to-from-mapped'][1]/n_warm/1e6, 3)} ms summed over streams, of which on the main stream {f(msp, 3)} ms; "
  f"regex-matched spin/wait total {f(spin_tot/n_warm/1e6, 3)} ms; union: only spin running {f(ms(mean([pw[i]['only_spin'] for i in WARM])), 3)}, spin+other {f(ms(mean([pw[i]['spin_plus'] for i in WARM])), 3)}")
P(f"drafter phase {f(ms(mean([dh[i]['dur'] for i in WARM])), 3)} ms = kernels active {f(ms(mean([dh[i]['act'] for i in WARM])), 3)} + idle {f(ms(mean([dh[i]['dur'] - dh[i]['act'] for i in WARM])), 3)} "
  f"(idle incl. memcpy {f(ms(mean([dh[i]['dur'] - dh[i]['act_all'] for i in WARM])), 3)}); median phase {f(ms(med([dh[i]['dur'] for i in WARM])), 3)}")
P(f"boundary: last drafter kernel -> first window kernel median {f(us(med(gk)), 1)} us, mean {f(us(mean(gk)), 1)} us (inside the drafter phase above)")
P(f"period {f(ms(mean(per)), 3)} ms (median {f(ms(med(per)), 3)}); tokens/s {f(sum(I[i]['T'] for i in WARM)/sum(per)*1e9, 1)}")
P("per T (means, ms per window): only handshake/spin kernels running; main-stream handshake/spin kernels; main-stream kernels >=16 us; main-stream kernels <=4 us; leading drafter gap (window end -> first drafter kernel): " + "; ".join(
    f"T{t}: {f(ms(mean([pw[i]['only_spin'] for i in WARM if I[i]['T']==t])), 3)}; {f(ms(mean([sum(k[1]-k[0] for k in inst[i] if pw[i]['role'][k[2]]=='main' and SPIN.search(S[k[5]])) for i in WARM if I[i]['T']==t])), 3)}; "
    f"{f(ms(mean([sum(k[1]-k[0] for k in inst[i] if pw[i]['role'][k[2]]=='main' and k[1]-k[0]>=16000) for i in WARM if I[i]['T']==t])), 3)}; "
    f"{f(ms(mean([sum(k[1]-k[0] for k in inst[i] if pw[i]['role'][k[2]]=='main' and k[1]-k[0]<=4000) for i in WARM if I[i]['T']==t])), 3)}; {f(us(mean([lead[i-W0] for i in WARM if I[i]['T']==t])), 1)} us"
    for t in sorted(tw)))
gk_all = []
for i in range(1, N):
    ks = K[bisect.bisect_right(kstarts, I[i - 1]['we']):bisect.bisect_left(kstarts, I[i]['ps'])]
    gk_all.append(I[i]['ws'] - max(k[1] for k in ks))
P(f"all 771 boundaries (cold + warm): last drafter kernel -> first window kernel median {f(us(med(gk_all)), 1)} us mean {f(us(mean(gk_all)), 1)} us; <1.4 ms: n={sum(1 for x in gk_all if x < 1400e3)} median {f(us(med([x for x in gk_all if x < 1400e3])), 1)} us")
P(f"warm phases preceded by an H2D >= 1 MiB in the drafter phase: {sum(1 for i in WARM if any(m[5]==1 and m[4] >= 2**20 for m in dh[i]['mc']))}; "
  f"warm phases whose last device activity before the parent launch is an H2D memcpy: {sum(1 for i in WARM if dh[i]['acts'] and dh[i]['acts'][-1][2]=='C' and dh[i]['acts'][-1][3].startswith('H2D'))}")

# ================================================================== KV block (consumed by /tmp/f453-report.py) and the per-T segmentation table
P("## S1T per T")
P("| T | graphId | kernels | warm n | A wall median / mean (ms) | B drafter phase median / mean (ms) | D boundary median / mean (us) | tokens/s = T / (wall + drafter phase) |")
P("|---|---|---|---|---|---|---|---|")
for g in sorted(WIN, key=lambda g: T_OF[g]):
    ws_ = [i for i in WARM if I[i]['g'] == g]
    if not ws_: continue
    w = [I[i]['we'] - I[i]['ws'] for i in ws_]; d_ = [I[i]['ps'] - I[i - 1]['we'] for i in ws_]; gg = [I[i]['ws'] - lastk[i] for i in ws_]
    P(f"| {T_OF[g]} | {g} | {I[ws_[0]]['n']} | {len(ws_)} | {f(ms(med(w)))} / {f(ms(mean(w)))} | {f(ms(med(d_)))} / {f(ms(mean(d_)))} | {f(us(med(gg)), 1)} / {f(us(mean(gg)), 1)} | "
      f"{f(T_OF[g]/ms(mean([pw[i]['wall'] + dh[i]['dur'] for i in ws_]))*1e3, 1)} |")

KV = {}
tot_k = sum(pb[i][b][0] for i in WARM for b in range(9))
def kv(k, v): KV[k] = v
def st6(vals, unit='ms', d=3):
    v = [x / (1e6 if unit == 'ms' else 1e3) for x in vals]
    return dict(med=f(med(v), d), mean=f(mean(v), d), p10=f(pct(v, .1), d), p90=f(pct(v, .9), d), mn=f(min(v), d), mx=f(max(v), d))
for nm, vals, unit, d in (('A', wd, 'ms', 3), ('B', dp, 'ms', 3), ('C', per, 'ms', 3), ('D', gk, 'us', 3)):
    for k2, v2 in st6(vals, unit, d).items(): kv(f'{nm}_{k2}', v2)
for nm, vals, unit, d in (('act', [pw[i]['act'] for i in WARM], 'ms', 3), ('idle', [pw[i]['idle'] for i in WARM], 'ms', 3),
                          ('wall', [pw[i]['wall'] for i in WARM], 'ms', 3),
                          ('main', [pw[i]['s']['main'] for i in WARM], 'ms', 3), ('sideA', sa, 'ms', 3), ('sideB', sb, 'ms', 3),
                          ('psum', ss, 'ms', 3), ('ovl', [x - pw[i]['act'] for x, i in zip(ss, WARM)], 'ms', 3),
                          ('b1', [pw[i]['l'][0] for i in WARM], 'ms', 3), ('b2', [pw[i]['l'][1] for i in WARM], 'ms', 3),
                          ('b3', [pw[i]['l'][2] for i in WARM], 'ms', 4),
                          ('midle', [pw[i]['wall'] - pw[i]['s']['main'] for i in WARM], 'ms', 3),
                          ('mside', [pw[i]['wall'] - pw[i]['s']['main'] - pw[i]['idle'] for i in WARM], 'ms', 3),
                          ('onlyspin', [pw[i]['only_spin'] for i in WARM], 'ms', 3), ('spinplus', [pw[i]['spin_plus'] for i in WARM], 'ms', 3),
                          ('onlybig', [pw[i]['only_big'] for i in WARM], 'ms', 3), ('bigplus', [pw[i]['big_plus'] for i in WARM], 'ms', 3),
                          ('onlysmall', [pw[i]['only_small'] for i in WARM], 'ms', 3),
                          ('dpdur', [dh[i]['dur'] for i in WARM], 'ms', 3), ('dpact', [dh[i]['act'] for i in WARM], 'ms', 3),
                          ('dpidlec', [dh[i]['dur'] - dh[i]['act_all'] for i in WARM], 'ms', 3), ('dpactmc', [dh[i]['act_all'] for i in WARM], 'ms', 3),
                          ('lead', lead, 'us', 1), ('bodyk', body, 'us', 1), ('trail', trail, 'us', 1),
                          ('hostg', [I[i]['ps'] - lastk[i] for i in WARM], 'us', 2), ('parent', [I[i]['pe'] - I[i]['ps'] for i in WARM], 'us', 2),
                          ('lat', [I[i]['ws'] - I[i]['pe'] for i in WARM], 'us', 2), ('chain', chain, 'us', 1)):
    for k2, v2 in st6(vals, unit, d).items(): kv(f'{nm}_{k2}', v2)
kv('N', N); kv('n_warm', n_warm); kv('W0', W0)
kv('tmix', ", ".join(f"T{t}={tw[t]}" for t in sorted(tw)))
kv('tpw', f(sum(I[i]['T'] for i in WARM)/n_warm, 3))
kv('kern_mean', f(mean([pw[i]['n'] for i in WARM]), 1)); kv('kern_med', f"{med([pw[i]['n'] for i in WARM]):.0f}")
kv('kern_min', min(pw[i]['n'] for i in WARM)); kv('kern_max', max(pw[i]['n'] for i in WARM))
kv('kern_main', f(mean([pw[i]['c']['main'] for i in WARM]), 0)); kv('kern_side', f(mean([pw[i]['c']['side'] for i in WARM])/2, 0))
kv('idle_sh_mean', f(100*mean([pw[i]['idle']/pw[i]['wall'] for i in WARM]), 2)); kv('idle_sh_med', f(100*med([pw[i]['idle']/pw[i]['wall'] for i in WARM]), 2))
kv('idle_sh_max', f(100*max(pw[i]['idle']/pw[i]['wall'] for i in WARM), 2))
kv('idle_sh_period', f(100*mean([pw[i]['idle'] for i in WARM])/mean(per), 2))
kv('cov_pct', f(100*mean([pw[i]['act']/pw[i]['wall'] for i in WARM]), 1))
kv('ovl_sh', f(100*mean([(x - pw[i]['act'])/pw[i]['act'] for x, i in zip(ss, WARM)]), 1))
kv('tok_s', f(sum(I[i]['T'] for i in WARM)/sum(per)*1e9, 1))
kv('variants', len(rows)); kv('summed_ms', f(tot_all/n_warm/1e6))
kv('spin_ms', f(spin_tot/n_warm/1e6, 3)); kv('spin_pct_sum', f(100*spin_tot/tot_all, 1)); kv('spin_pct_wall', f(100*spin_tot/wall_sum, 1))
kv('wait3_ms', f(pure/n_warm/1e6, 3)); kv('spin_main_ms', f(cls['host-handshake/spin/copy-to-from-mapped'][2]/n_warm/1e6, 3))
typD = [x for x in gk if x < SLOW_PHASE_US * 1000]; slowD = [x for x in gk if x >= SLOW_PHASE_US * 1000]
kv('typ_n', len(typD)); kv('typ_med', f(us(med(typD)), 1)); kv('slow_n', len(slowD))
kv('slow_med', f(us(med(slowD)), 0)); kv('slow_min', f(us(min(slowD)), 0)); kv('slow_max', f(us(max(slowD)), 0))
kv('slow_phase_med', f(ms(med([dp[i - W0] for i in WARM if gk[i - W0] >= SLOW_PHASE_US * 1000])))); kv('slow_phase_mean', f(ms(mean([dp[i - W0] for i in WARM if gk[i - W0] >= SLOW_PHASE_US * 1000]))))
kv('all_med', f(us(med(gk_all)), 1)); kv('all_mean', f(us(mean(gk_all)), 1)); kv('all_typ_n', sum(1 for x in gk_all if x < 1400e3))
kv('lastact', ", ".join(f"`{k}` {v}" for k, v in lastkind.most_common(3)))
def tk(thr, pred):
    t_ = sum(tsum(i, pred) for i in WARM) / n_warm; fr = [tsum(i, pred) / pw[i]['wall'] for i in WARM]
    return t_, f(100*mean(fr), 1), f(100*med(fr), 1), f(100*t_*n_warm/tot_all, 1)
le = lambda thr: (lambda b: edges[b + 1] <= thr)
for nm, thr in (('lt4', 4000), ('lt8', 8000), ('lt16', 16000)):
    t_, a, b_, c_ = tk(thr, le(thr)); kv(f'{nm}_ms', f(ms(t_), 3)); kv(f'{nm}_pct', a); kv(f'{nm}_pctmed', b_); kv(f'{nm}_pctsum', c_)
t_, a, b_, c_ = tk(16000, lambda b: edges[b] >= 16000); kv('ge16_ms', f(ms(t_), 3)); kv('ge16_pct', a); kv('ge16_pctmed', b_); kv('ge16_pctsum', c_)
kv('mid_ms', f(ms(sum(tsum(i, lambda b: b in (2, 3)) for i in WARM)/n_warm), 3))
kv('lt4_n', f(sum(pb[i][b][0] for i in WARM for b in (0, 1))/n_warm, 1)); kv('lt4_kpct', f(100*sum(pb[i][b][0] for i in WARM for b in (0, 1))/tot_k, 1))
kv('ge16_n', f(sum(pb[i][b][0] for i in WARM for b in range(4, 9))/n_warm, 1))
kv('main_ge16', f(sum(mb[b][1] for b in range(4, 9))/n_warm/1e6)); kv('main_lt4', f(sum(mb[b][1] for b in (0, 1))/n_warm/1e6))
kv('perT_small', "; ".join(
    f"T{t} " + "/".join(f(100*sum(pbT[t][b][1] for b in range(9) if edges[b+1] <= thr)/sum(pw[i]['wall'] for i in WARM if I[i]['T']==t), 1) for thr in (4000, 8000, 16000)) for t in sorted(tw)))
kv('mc_n', f(len(mcw)/n_warm, 2)); kv('mc_mib', f(sum(m[4] for m in mcw)/n_warm/2**20, 2)); kv('mc_us', f(sum(m[1]-m[0] for m in mcw)/n_warm/1e3, 1))
kv('mc_big_n', f(len(big_h2d)/n_warm, 2)); kv('mc_big_mib', f(med([m[4] for m in big_h2d])/2**20, 2)); kv('mc_big_us', f(us(med([m[1]-m[0] for m in big_h2d])), 1))
kv('mc_big_gbs', f(med([m[4]/(m[1]-m[0]) for m in big_h2d]), 1)); kv('mc_big_stream', ",".join(str(x) for x in sorted({m[2] for m in big_h2d})))
kv('mc_ingraph', sum(1 for m in mcw if m[3] is not None)); kv('mc_ingraph_kind', ",".join(KIND[k] for k in {m[5] for m in mcw if m[3] is not None}))
kv('memset_n', sum(1 for m in MS if any(I[i]['ws'] <= m[0] < I[i]['we'] for i in WARM)))
kv('idle_mc_med', f(ms(med([pw[i]['wall'] - pw[i]['act_mc'] for i in WARM])), 3)); kv('idle_mc_mean', f(ms(mean([pw[i]['wall'] - pw[i]['act_mc'] for i in WARM])), 3))
kv('gaps_n', f(tot_gn/n_warm, 1))
for b, nm in enumerate(('g1', 'g2', 'g4', 'g8', 'g16', 'g64', 'gBig')):
    kv(f'{nm}_n', f"{sum(gapb[i][b][0] for i in WARM)}"); kv(f'{nm}_pw', f(sum(gapb[i][b][0] for i in WARM)/n_warm, 2))
    kv(f'{nm}_ms', f(sum(gapb[i][b][1] for i in WARM)/n_warm/1e6, 4)); kv(f'{nm}_pct', f(100*sum(gapb[i][b][1] for i in WARM)/tot_gap, 1))
    kv(f'{nm}_mean', f(us(sum(gapb[i][b][1] for i in WARM)/max(1, sum(gapb[i][b][0] for i in WARM))), 2))
kv('gapT', "; ".join(f"T{t} " + "/".join(f(ms(gapbT[t][b][1]/tw[t]), 3) for b in range(7)) for t in sorted(tw)))
kv('long_classes', len(longcls)); kv('long_cover', f(100*sum(v[1] for _, v in big_cls[:10])/tot_long, 1))
kv('s5_med_min', f(us(min(med(v[2]) for _, v in big_cls[:10])), 0)); kv('s5_med_max', f(us(max(med(v[2]) for _, v in big_cls[:10])), 0))
kv('s5_occ_max', f(max(v[0] for _, v in big_cls[:10])/n_warm*100, 1))
kv('gap_gt1_n', f(sum(gapb[i][b][0] for i in WARM for b in range(1, 7))/n_warm, 1)); kv('gap_gt1_ms', f(ms(sum(gapb[i][b][1] for i in WARM for b in range(1, 7))/n_warm), 4))
kv('dp_n_med', f"{med([dh[i]['n'] for i in WARM]):.0f}"); kv('dp_n_mean', f(mean([dh[i]['n'] for i in WARM]), 1)); kv('dp_n_min', min(dh[i]['n'] for i in WARM)); kv('dp_n_max', max(dh[i]['n'] for i in WARM))
kv('dp_idlek_med', f(ms(med([dh[i]['dur'] - dh[i]['act'] for i in WARM])), 3)); kv('dp_idlek_mean', f(ms(mean([dh[i]['dur'] - dh[i]['act'] for i in WARM])), 3))
kv('dp_idlek_p10', f(ms(pct([dh[i]['dur'] - dh[i]['act'] for i in WARM], .1)), 3)); kv('dp_idlek_p90', f(ms(pct([dh[i]['dur'] - dh[i]['act'] for i in WARM], .9)), 3))
kv('dp_idlek_mn', f(ms(min(dh[i]['dur'] - dh[i]['act'] for i in WARM)), 3)); kv('dp_idlek_mx', f(ms(max(dh[i]['dur'] - dh[i]['act'] for i in WARM)), 3))
kv('dp_ish_mean', f(100*mean([(dh[i]['dur'] - dh[i]['act'])/dh[i]['dur'] for i in WARM]), 1)); kv('dp_ish_med', f(100*med([(dh[i]['dur'] - dh[i]['act'])/dh[i]['dur'] for i in WARM]), 1))
kv('dmc_n', f(len(allmc)/n_warm, 2)); kv('dmc_h2d_n', f(sum(1 for m in allmc if m[5]==1)/n_warm, 2)); kv('dmc_h2d_mib', f(sum(m[4] for m in allmc if m[5]==1)/n_warm/2**20, 2))
kv('dmc_h2d_us', f(sum(m[1]-m[0] for m in allmc if m[5]==1)/n_warm/1e3, 1)); kv('dmc_d2h_n', f(sum(1 for m in allmc if m[5]==2)/n_warm, 2))
kv('dmc_d2h_mib', f(sum(m[4] for m in allmc if m[5]==2)/n_warm/2**20, 2)); kv('dmc_d2h_us', f(sum(m[1]-m[0] for m in allmc if m[5]==2)/n_warm/1e3, 1))
kv('dmc_big_ph', sum(1 for i in WARM if any(m[5]==1 and m[4] >= 2**20 for m in dh[i]['mc'])))
kv('dmc_big_med_mib', f(med([m[4] for m in h2d])/2**20, 2)); kv('dmc_big_med_us', f(us(med([m[1]-m[0] for m in h2d])), 1)); kv('dmc_big_stream', ",".join(str(x) for x in sorted({x[2] for x in h2d})))
kv('dmc_d2h_med_us', f(us(med([m[1]-m[0] for m in d2h])), 1))
kv('dmc_cross', sum(dh[i]['cross'] for i in WARM))
kv('dmc_d2h_min_mib', f(min(m[4] for m in d2h)/2**20, 2)); kv('dmc_d2h_max_mib', f(max(m[4] for m in d2h)/2**20, 2))
c1 = dcls[('mtp_select_kernel', 'H2D memcpy 96 KiB')]
kv('dgc1_n', c1[0]); kv('dgc1_ms', f(c1[1]/n_warm/1e6, 3)); kv('dgc1_pct', f(100*c1[1]/tot_dg, 1)); kv('dgc1_med', f(us(med(c1[2])), 1)); kv('dgc1_p90', f(us(pct(c1[2], .9)), 0))
kv('dg_ms', f(tot_dg/n_warm/1e6, 3)); kv('dg_n', f(sum(v[0] for v in dcls.values())/n_warm, 1))
c2 = [v for k, v in dcls.items() if k[0] == 'mtp_select_kernel' and k[1] == 'copy_from_mapped_multi_kernel'][0]
kv('dgc2_n', f(c2[0]/n_warm, 2)); kv('dgc2_ms', f(c2[1]/n_warm/1e6, 3)); kv('dgc2_med', f(us(med(c2[2])), 1))
kv('dg_classes', len(dcls)); kv('dp_slowidle_n', len(longg))
kv('dp_longgap_n', sum(1 for i in WARM if maxgap[i] > 1e6))
kv('dp_longgap_med', f(ms(med([dh[i]['dur'] for i in WARM if maxgap[i] > 1e6])))); kv('dp_longgap_mean', f(ms(mean([dh[i]['dur'] for i in WARM if maxgap[i] > 1e6]))))
kv('dp_lead100_n', sum(1 for i in WARM if lead[i - W0] > 100e3))
trv = [I[i]['ps'] - lastk[i] for i in WARM]
kv('tr_a', sum(1 for x in trv if x < 100e3)); kv('tr_a_med', f(us(med([x for x in trv if x < 100e3])), 1)); kv('tr_b', sum(1 for x in trv if 100e3 <= x < 500e3))
kv('tr_c', sum(1 for x in trv if 500e3 <= x < 1400e3)); kv('tr_d', sum(1 for x in trv if x >= 1400e3)); kv('tr_d_med', f(us(med([x for x in trv if x >= 1400e3])), 0))
kv('dk_sum_ms', f(sum(v[1] for v in list(grp.values()) + list(eag.values()))/n_warm/1e6, 3)); kv('dg_sum_ms', f(sum(v[1] for v in gi.values())/n_warm/1e6, 3))
kv('steps_vs_T', "; ".join(f"{k} steps -> " + ", ".join(f"T{t}={k2}" for t, k2 in sorted(tseq[k].items())) for k in sorted(tseq)))
five = tseq[5]; kv('five_n', sum(five.values())); kv('five_t2', five.get(2, 0))
kv('round_n', gi[ROUND_G][0]); kv('round_med', f(us(med(gi[ROUND_G][2])), 1))
kv('round_commit_med', f(us(med(grp['gdn_step_commit_layers_kernel'][2])), 1))
kv('s2_min', f(us(min(med(gi[g][2]) for g in (16, 19, 22, 25))), 1)); kv('s2_max', f(us(max(med(gi[g][2]) for g in (16, 19, 22, 25))), 1))
kv('s1_min', f(us(min(med(gi[g][2]) for g in STEP1)), 1)); kv('s1_max', f(us(max(med(gi[g][2]) for g in STEP1)), 1))
kv('s1_nodes', f"{min(gi[g][3]//gi[g][0] for g in STEP1)}-{max(gi[g][3]//gi[g][0] for g in STEP1)}"); kv('s1_var', len(STEP1))
kv('wall_sum', f(ms(wall_sum), 1)); kv('tot_all', f(ms(tot_all), 1))
dens = cls['dense matmul, stage NOT identifiable by name (native mmvq / bf16 mmvf)']
kv('dens_ms', f(dens[1]/n_warm/1e6, 3)); kv('dens_pct', f(100*dens[1]/wall_sum, 1)); kv('dens_n', f(dens[0]/n_warm, 0))
kv('dens_unres', posbad)
shr = {nm: (f(v[1]/n_warm/1e6, 3), f(100*v[1]/wall_sum, 1), f(v[2]/n_warm/1e6, 3)) for nm, v in merged.items()}
SH = {'gdn': 'attention: GDN (recurrence/conv/norm)', 'hs': 'host-handshake/spin/copy-to-from-mapped', 'hc': 'hyper-connections (HC stream read/write)',
      'moe': 'MoE routed-expert compute (grouped gate/up + down)', 'qsa': 'attention: QSA (sparse attn, indexer/select, KV gather/append, rope, out-gate)',
      'sh': 'shared expert (gate scale by name; matmuls resolved by position)', 'rt': 'router/top-k/plan', 'hd': 'head/argmax/sampling', 'nq': 'normalization/quant',
      'cv': 'control vector (cvec)', 'dn': 'dense matmul, stage NOT identifiable by name (native mmvq / bf16 mmvf)',
      'cp': 'CPU-expert combine/copy (T=1 path: mapped rows, hits, k10 combine)', 'ot': 'other (embedding, per-token tail block: gate/broadcast/history/post)'}
for k2, nm in SH.items():
    kv(f'c_{k2}_ms', shr[nm][0]); kv(f'c_{k2}_pct', shr[nm][1]); kv(f'c_{k2}_main', shr[nm][2])
kv('c_sum_pct', f(sum(float(shr[nm][1]) for nm in merged), 1))
kv('hs_T', "; ".join(f"T{t} {f(sum(v[1] for k2, v in spin_by_T[t].items())/tw[t]/1e6, 2)}" for t in sorted(tw)))
kv('os_T', "; ".join(f"T{t} {f(ms(mean([pw[i]['only_spin'] for i in WARM if I[i]['T']==t])), 2)}" for t in sorted(tw)))
kv('nspin', sum(1 for d, a in rows if SPIN.search(S[d])))
dm = {t: v for t, v in spin_by_T.items()}
kv('dm_t2', f"T2 median {f(us(med(dm[2]['dm_wait_combine_kernel'][2])), 1)} mean {f(us(mean(dm[2]['dm_wait_combine_kernel'][2])), 1)} us; T6 median {f(us(med(dm[6]['dm_wait_combine_kernel'][2])), 1)} mean {f(us(mean(dm[6]['dm_wait_combine_kernel'][2])), 1)} us")
kv('mwall', f(mwall, 3)); kv('mact', f(mact, 3)); kv('midle', f(midle, 3)); kv('mmain', f(mmain, 3)); kv('mside_only', f(mside_only, 3))
kv('main_buckets', "; ".join(f"{lab[b]} {f(mb[b][1]/n_warm/1e6, 3)}" for b in range(9)))
kv('mb_other', f((merged['dense matmul, stage NOT identifiable by name (native mmvq / bf16 mmvf)'][2] + merged['other (embedding, per-token tail block: gate/broadcast/history/post)'][2] + merged['CPU-expert combine/copy (T=1 path: mapped rows, hits, k10 combine)'][2])/n_warm/1e6, 3))
kv('all_other', f((merged['dense matmul, stage NOT identifiable by name (native mmvq / bf16 mmvf)'][1] + merged['other (embedding, per-token tail block: gate/broadcast/history/post)'][1] + merged['CPU-expert combine/copy (T=1 path: mapped rows, hits, k10 combine)'][1])/n_warm/1e6, 2))
kv('spin_ge_lt', ", ".join(f"{lab[b]} {f(sum(a[1] for d, a in rows if SPIN.search(S[d]) for i in WARM for k in inst[i] if k[5]==d and bkt(k[1]-k[0])==b)/n_warm/1e6, 3)}" for b in (0, 1, 2, 3, 4, 5, 6, 7, 8)))
kv('cov', f(100*mean([pw[i]['act']/pw[i]['wall'] for i in WARM]), 1))
kv('per_all_med', f(ms(med([I[i]['ws'] - I[i-1]['ws'] for i in range(1, N)])), 3)); kv('per_all_mean', f(ms(mean([I[i]['ws'] - I[i-1]['ws'] for i in range(1, N)])), 3))
kv('exT', "; ".join(f"T{t} (g{g}) {f(100*tw[t]/n_warm, 0)}%" for g, t in sorted(T_OF.items(), key=lambda x: x[1])))
g162 = sum(1 for i in WARM for k in inst[i] if k[1] - k[0] >= 16000); kv('ge16_direct_n', f(g162/n_warm, 1))
kv('ge16_direct_ms', f(ms(sum(k[1] - k[0] for i in WARM for k in inst[i] if k[1] - k[0] >= 16000)/n_warm), 3))
le4d = sum(1 for i in WARM for k in inst[i] if k[1] - k[0] <= 4000); kv('lt4_direct_n', f(le4d/n_warm, 1))
P("## KV")
for k2, v2 in KV.items(): P(f"KV {k2} = {v2}")
open(OUT, 'w').write("\\n".join(out) + "\\n")
print("\n".join(out))
