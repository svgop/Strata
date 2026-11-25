#!/usr/bin/env python3
"""F-457 (run once): is gdn_ab_multi_kernel on the path to its consumer?  Usage: f457-slack.py <anatomy.sqlite>
For every gdn_ab_multi_kernel instance in a window graph: the gap from its END to the START of the next
gdn_step_norm_multi_kernel in the same graph launch (its reader of gate/beta), the same for the conv, and how often
gdn_ab is the last of the step's producers to finish.  Read-only."""
import sqlite3, statistics as st, sys
from collections import Counter
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
names = dict(c.execute("select id, value from StringIds"))
K = c.execute("select start,end,demangledName,graphId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
isnm = lambda k, s: s in names.get(k[2], "")
gc = Counter(k[3] for k in K if k[3] is not None); win = {g for g, n in gc.items() if n >= 1500 and g < 100}
gaps, conv_gap, bind, last = [], [], 0, 0
for i, k in enumerate(K):
    if k[3] not in win or not isnm(k, "gdn_ab_multi_kernel"): continue
    j = i + 1
    while j < len(K) and not (K[j][3] == k[3] and isnm(K[j], "gdn_step_norm_multi_kernel")): j += 1
    m = j - 1
    while m > i and not isnm(K[m], "gdn_conv_l2_multi_kernel"): m -= 1
    if j >= len(K) or m <= i: continue
    g = (K[j][0] - k[1]) / 1e3; gaps.append(g); conv_gap.append((K[j][0] - K[m][1]) / 1e3)
    if g < 2.0: bind += 1
    if K[m][1] <= k[1]: last += 1        # the conv finished before gdn_ab did: gdn_ab is (one of) the last
pct = lambda v, f: v[int(len(v) * f)]
g = sorted(gaps); n = len(g); cg = sorted(conv_gap)
print(f"instances {n} (graphIds {sorted(win)})")
print(f"gdn_ab END -> gdn_step_norm START, us: median {pct(g,.5):.2f}  p10 {pct(g,.1):.2f}  p1 {pct(g,.01):.2f}  min {g[0]:.2f}  p90 {pct(g,.9):.2f}  mean {st.mean(g):.2f}")
print(f"conv END -> gdn_step_norm START, us:    median {pct(cg,.5):.2f}  p10 {pct(cg,.1):.2f}  p1 {pct(cg,.01):.2f}  min {cg[0]:.2f}  p90 {pct(cg,.9):.2f}  mean {st.mean(cg):.2f}")
print(f"gdn_ab among the last producers to finish (conv END <= gdn_ab END): {last} of {n} = {100*last/n:.2f}%")
print(f"gdn_ab END within 2 us of gdn_step_norm START or later (binding):   {bind} of {n} = {100*bind/n:.2f}%")
neg = sum(1 for x in gaps if x < 0); print(f"negative gaps (gdn_ab END after gdn_step_norm START): {neg}")
