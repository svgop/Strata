#!/usr/bin/env python3
"""F-457 (run once): the timeline neighbours of gdn_ab_multi_kernel inside the solo window graphs.
Usage: f457-neighbours.py <anatomy.sqlite> [--dump N]
Reads a node-level capture (STRATA_ANATOMY_GRAPH_TRACE=node).  For every gdn_ab_multi_kernel instance that sits
inside a window graph (graphId with >= 1500 kernels) it reports the kernel immediately before it and immediately
after it, both in GPU start order over all streams and on its own stream.  Read-only."""
import sqlite3, statistics as st, sys
from collections import Counter, defaultdict
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
q = lambda s, *a: c.execute(s, a).fetchall()
names = dict(q("select id, value from StringIds"))
K = q("select start, end, demangledName, gridX, blockX, streamId, graphId, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL order by start")
# window graphs: graphIds that own >= 1500 kernel instances
gc = Counter(k[6] for k in K if k[6] is not None)
win = {g for g, n in gc.items() if n >= 1500}
def short(nid):
    s = names.get(nid, "?")
    if "(" in s: s = s.split("(")[0]
    return s.replace("void ", "")
print(f"kernels {len(K)}; window graphIds {sorted(win)} (instances {[gc[g] for g in sorted(win)]})")
# window instances: runs of the window graph separated by > 150 us
inst = defaultdict(list)
for i, k in enumerate(K):
    if k[6] in win: inst[k[6]].append((k[0], k[1], i))
runs = []
for gid, ks in inst.items():
    cur = [ks[0]]
    for a in ks[1:]:
        if a[0] - cur[-1][1] > 150_000: runs.append(cur); cur = []
        cur.append(a)
    runs.append(cur)
print(f"window launches {len(runs)}")
# per graphId: launches, kernels per launch, gdn_ab per launch, gdn_ab instance count
per = defaultdict(lambda: [0, 0, 0])
for r in runs:
    per[r[0][0] and K[r[0][2]][6]][0] += 1; per[K[r[0][2]][6]][1] += len(r); per[K[r[0][2]][6]][2] += sum(1 for _, _, i in r if "gdn_ab_multi_kernel" in names.get(K[i][2], ""))
for g, (nl, nk, na) in sorted(per.items()):
    print(f"  graphId {g:>4}: launches {nl:>5}, kernels/launch {nk/nl:8.1f}, gdn_ab/launch {na/nl:6.2f}, gdn_ab instances {na}")
is_ab = lambda k: "gdn_ab_multi_kernel" in names.get(k[2], "")
def slot(k):
    return (short(k[2]), f"{k[3]}x{k[4]}")
prev_all, next_all, prev_str, next_str = (defaultdict(list) for _ in range(4))
gap_prev, gap_next = [], []
layer_pos = defaultdict(lambda: defaultdict(list))   # ordinal within window -> (prev, next, dur)
nab = 0
dump = int(sys.argv[sys.argv.index("--dump") + 1]) if "--dump" in sys.argv else 0
dumped = 0
for r in runs:
    for j, (s, e, i) in enumerate(r):
        k = K[i]
        if not is_ab(k): continue
        nab += 1
        p, n = K[i - 1], K[i + 1]
        prev_all[slot(p)].append((p[1] - p[0]) / 1e3); next_all[slot(n)].append((n[1] - n[0]) / 1e3)
        gap_prev.append((s - p[1]) / 1e3); gap_next.append((n[0] - e) / 1e3)
        # same stream
        m = i - 1
        while m >= 0 and K[m][5] != k[5]: m -= 1
        if m >= 0: prev_str[slot(K[m])].append((K[m][1] - K[m][0]) / 1e3); gap_prev.append(-1) if False else None
        m = i + 1
        while m < len(K) and K[m][5] != k[5]: m += 1
        if m < len(K): next_str[slot(K[m])].append((K[m][1] - K[m][0]) / 1e3)
        # ordinal of this gdn_ab in its window launch
        o = sum(1 for jj in range(j) if is_ab(K[r[jj][2]]))
        layer_pos[o]["dur"].append((e - s) / 1e3)
        if dumped < dump and o == dump:
            dumped += 1
            print(f"\n-- dump: window launch around gdn_ab ordinal {o} (stream {k[5]}, graphId {k[6]}, node {k[7]}) --")
            for jj in range(max(0, j - 12), min(len(r), j + 9)):
                kk = K[r[jj][2]]
                mark = ">>" if jj == j else "  "
                print(f"{mark} s{kk[5]:<3} +{(kk[0]-s)/1e3:8.2f} us  dur {(kk[1]-kk[0])/1e3:7.2f}  grid {kk[3]:>5}x{kk[4]:<4} {short(kk[2])}")
def show(title, d):
    print(f"\n{title}  (instances {sum(len(v) for v in d.values())})")
    for s, v in sorted(d.items(), key=lambda kv: -len(kv[1]))[:8]:
        v = sorted(v)
        print(f"  {len(v):>6}  {s[0][:62]:<62} {s[1]:>10}  median {v[len(v)//2]:6.2f} us  p10 {v[len(v)//10]:6.2f}  p90 {v[9*len(v)//10]:6.2f}")
print(f"\ngdn_ab_multi_kernel instances inside window graphs: {nab}")
show("BEFORE (GPU start order, all streams)", prev_all)
show("AFTER  (GPU start order, all streams)", next_all)
show("BEFORE (same stream)", prev_str)
show("AFTER  (same stream)", next_str)
for nm, g in (("gap from prev END to gdn_ab START", gap_prev), ("gap from gdn_ab END to next START", gap_next)):
    g = sorted(g); print(f"{nm}: median {g[len(g)//2]:.2f} us, p10 {g[len(g)//10]:.2f}, p90 {g[9*len(g)//10]:.2f}")
print("\nper ordinal (layer) gdn_ab duration: " + ", ".join(f"L{o}:{st.median(v['dur']):.2f}" for o, v in sorted(layer_pos.items())))
