#!/usr/bin/env python3
"""F-453 (run once): the gdn_ab preload's mechanism at node level.  Usage: f453-gdnab-kernels.py <anatomy.sqlite>
Reads a node-level capture (STRATA_ANATOMY_GRAPH_TRACE=node): every gdn_ab_multi_kernel instance's duration, and the
window graphs' (the > 1,500-kernel graphs) instance duration and kernel-time sum.  Read-only."""
import sqlite3, statistics as st, sys
from collections import defaultdict
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
q = lambda s, *a: c.execute(s, a).fetchall()
names = dict(q("select id, value from StringIds"))
K = q("select start, end, demangledName, gridX, graphId from CUPTI_ACTIVITY_KIND_KERNEL order by start")
ab = [(e - s) / 1e3 for s, e, n, g, gid in K if "gdn_ab_multi_kernel" in names.get(n, "")]
if ab:
    ab.sort()
    print(f"gdn_ab_multi_kernel: n={len(ab)}, median {ab[len(ab)//2]:.2f} us, mean {st.mean(ab):.2f} us, p10 {ab[len(ab)//10]:.2f}, p90 {ab[9*len(ab)//10]:.2f}   (the 11-24 baseline: 8.26 us mean)")
else:
    print("gdn_ab_multi_kernel: NO INSTANCES (is this a node-level export?)")
by = defaultdict(list)
for s, e, n, g, gid in K:
    if gid is not None: by[gid].append((s, e))
rows = []
for gid, ks in by.items():
    if len(ks) < 1500: continue
    ks.sort(); runs = []; cur = [ks[0]]
    for a, b in ks[1:]:
        if a - cur[-1][1] > 150_000: runs.append(cur); cur = []
        cur.append((a, b))
    runs.append(cur)
    d = [(r[-1][1] - r[0][0]) / 1e6 for r in runs]; k = [sum(b - a for a, b in r) / 1e6 for r in runs]
    rows.append((gid, len(runs), st.median(d), st.mean(k)))
for gid, n, d, k in sorted(rows, key=lambda r: -r[1])[:4]:
    print(f"window graphId {gid}: instances {n}, median occupancy {d:.3f} ms, mean kernel time {k:.3f} ms")
tot = sum(r[1] for r in rows); print(f"kernel rows {len(K)}; window instances {tot}")
