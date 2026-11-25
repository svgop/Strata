#!/usr/bin/env python3
"""F-457: the GDN front-end's chain per layer (qkv, the gate projection z, the conv, the step) from a node-level capture.
Usage: f457-gdn-chain.py <anatomy.sqlite> (read-only).  A layer is the span from one gdn_ab_multi_kernel start to the
next within 2 ms; times are medians over layers, printed in MICROSECONDS (the table's start/end are nanoseconds)."""
import bisect, sqlite3, statistics as st, sys
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
names = dict(c.execute("select id, value from StringIds"))
K = c.execute("select start,end,demangledName,graphId,gridX,blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
nm = lambda r: names.get(r[2], "?").split("(")[0]
ab = [r for r in K if "gdn_ab_multi" in nm(r)]; starts = [r[0] for r in K]
pairs = [(p, q) for p, q in zip(ab, ab[1:]) if 0 < q[0] - p[0] < 2_000_000]
T = []
for p, q in pairs:
    lay = K[bisect.bisect_left(starts, p[0]):bisect.bisect_left(starts, q[0])]
    mm = [r for r in lay if "mmvq_il" in nm(r)]; cv = [r for r in lay if "conv_l2" in nm(r)]; sp = [r for r in lay if "step_norm" in nm(r)]
    if len(mm) < 2 or not cv or not sp: continue
    a, b, v, s = mm[0], mm[1], cv[0], sp[0]           # qkv = the first projection, z = the second
    T.append((a[0] - p[0], a[1] - a[0], b[0] - a[0], b[1] - b[0], b[0] - a[1], v[0] - a[1], v[1] - v[0], s[0] - v[1], s[0] - a[0], a[4], b[4]))
if not T: sys.exit("no layer with two projections, a conv and a step was found")
med = lambda i: st.median(t[i] for t in T)
M = [med(i) / 1e3 for i in range(9)]; p = lambda i, f: sorted(t[i] for t in T)[f * len(T) // 10] / 1e3   # times in us; grids unscaled
print(f"layers {len(T)} | qkv: starts +{M[0]:.2f} us after gdn_ab, runs {M[1]:.2f} us, grid median {M[9]:.0f} blocks")
print(f"z: starts {M[2]:.2f} us after qkv's start ({M[4]:+.2f} us against qkv's END), runs {M[3]:.2f} us, grid median {M[10]:.0f} blocks")
print(f"conv: starts {M[5]:+.2f} us against qkv's END, runs {M[6]:.2f} us | step: {M[7]:.2f} us after the conv's end")
print(f"chain qkv START -> step START: median {M[8]:.2f} us (p10 {p(8, 1):.2f}, p90 {p(8, 9):.2f}) | x36 = {36 * M[8] / 1e3:.3f} ms per window")
