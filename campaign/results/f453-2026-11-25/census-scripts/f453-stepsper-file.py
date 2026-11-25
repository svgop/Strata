#!/usr/bin/env python3
"""F-453: steps per solo window from a node-level capture.  Usage: f453-stepsper-file.py <anatomy.sqlite> (read-only).
A port of census-scripts/f453-stepsper.py to a path argument: consecutive instances of one window graph under 25 ms
apart are solo steady intervals; the step graphs between them are counted.  Also prints the wall between windows."""
import sqlite3, statistics as st
from collections import Counter, defaultdict
import sys
S=sys.argv[1]
c=sqlite3.connect(f"file:{S}?mode=ro",uri=True); q=lambda s,*a: c.execute(s,a).fetchall()
K=q("select graphId, start, end, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null order by graphId, start, graphNodeId")
per=defaultdict(list); cur=None; prev=None
for gid,s,e,nid in K:
    if gid!=cur: cur=gid; prev=None; run=[]
    if prev is not None and s-prev>150_000: per[cur].append(run); run=[]
    run.append((s,e)); prev=e
inst=[]
for gid,runs in per.items():
    for r in runs: inst.append((r[0][0], r[-1][1], gid, len(r)))
inst.sort()
big={gid for gid,runs in per.items() if len(runs)>=3 and st.median([r[-1][1]-r[0][1] for r in runs])>3_000_000}
step={gid for gid,runs in per.items() if len(runs)>=3 and 150_000<st.median([r[-1][1]-r[0][1] for r in runs])<350_000 and 30<=st.median([len(r) for r in runs])<=80}
print("window graphs:", sorted(big), "| step graphs:", sorted(step))
# solo steady state: consecutive window instances, same graph, inter-start < 25 ms, with no other window graph between
rows=[]; i=0
while i<len(inst):
    s,e,gid,n=inst[i]
    if gid in big:
        j=i+1
        while j<len(inst) and inst[j][2] not in big: j+=1
        if j<len(inst) and inst[j][2]==gid and inst[j][0]-s<25_000_000:
            between=inst[i+1:j]
            kinds=Counter(('step' if g in step else 'other') for _,_,g,_ in between)
            step_ms=sum(e2-s2 for s2,e2,g,_ in between if g in step)/1e6
            span=(inst[j][0]-e)/1e6
            rows.append((gid,kinds['step'],step_ms,span,sorted(g for _,_,g,_ in between if g not in step)))
    i+=1
print("solo steady intervals:", len(rows))
by=defaultdict(list)
for gid,ns,sm,span,o in rows: by[gid].append((ns,sm,span,tuple(o)))
for gid,v in sorted(by.items(), key=lambda x:-len(x[1]))[:6]:
    ns=[a for a,_,_,_ in v]; sm=[b for _,b,_,_ in v]; sp=[c for _,_,c,_ in v]
    print(f"  window graph {gid}: n={len(v)} steps/window mean {st.mean(ns):.2f} median {st.median(ns):.0f} dist {dict(sorted(Counter(ns).items()))} | step-graph time mean {st.mean(sm):.3f} ms | gap to next window (wall between windows) mean {st.mean(sp):.2f} ms")
alln=[a for v in by.values() for a,_,_,_ in v]; allsm=[b for v in by.values() for _,b,_,_ in v]; allsp=[c for v in by.values() for _,_,c,_ in v]
print(f"ALL: steps/window mean {st.mean(alln):.2f} (median {st.median(alln):.0f}); step-graph GPU time mean {st.mean(allsm):.3f} ms; wall between windows mean {st.mean(allsp):.3f} ms")
o=Counter(x for v in by.values() for *_,t in v for x in t); print("other graphs seen between windows (graphId: count):", dict(o.most_common(8)))
