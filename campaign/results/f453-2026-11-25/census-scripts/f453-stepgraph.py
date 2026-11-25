import sqlite3, statistics as st
from collections import defaultdict
S="/mnt/DATA/LLM/Models/GGUF/strata/Custom-data/profiles/2026-11-25-f453/f452-dev.sqlite"
c=sqlite3.connect(f"file:{S}?mode=ro",uri=True); q=lambda s,*a: c.execute(s,a).fetchall()
names=dict(q("select id, value from StringIds"))
K=q("select graphId, start, end, demangledName, gridX, gridY, gridZ, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null order by graphId, start, graphNodeId")
per=defaultdict(list); cur=None; prev=None
for k in K:
    if k[0]!=cur: cur=k[0]; prev=None; run=[]
    if prev is not None and k[1]-prev>150_000: per[cur].append(run); run=[]
    run.append(k); prev=k[2]
rows=[]
for gid,runs in per.items():
    if len(runs)<3: continue
    ds=[r[-1][2]-r[0][1] for r in runs]; nk=[len(r) for r in runs]
    rows.append((gid,len(runs),st.median(ds)/1e3,st.median(nk)))
print("graphIds with >=3 instances:", len(rows), "| the 9 longest by median duration:")
for r in sorted(rows,key=lambda r:-r[2])[:9]: print(f"  graphId {r[0]}: n={r[1]} median {r[2]:.1f} us, kernels {r[3]:.0f}")
step=[gid for gid,n,d,k in rows if 150<d<350 and 30<=k<=80]
print("step-graph candidates (median 150-350 us, 30-80 kernels):", step)
tot_k=tot_g=n_inst=0; gaps=[]; agg=defaultdict(lambda:[0,0.0]); nk=[]
for gid in step:
    for ks in per[gid]:
        nk.append(len(ks)); tot_k+=sum(k[2]-k[1] for k in ks)
        g=[max(0,ks[i+1][1]-ks[i][2]) for i in range(len(ks)-1)]; tot_g+=sum(g); n_inst+=1; gaps+=g
        for k in ks:
            a=agg[(names.get(k[3],'?'),k[4],k[5],k[6])]; a[0]+=1; a[1]+=(k[2]-k[1])/1e3
print(f"instances {n_inst}: kernels/instance median {st.median(nk):.0f}; kernel time {tot_k/n_inst/1e3:.1f} us, gaps {tot_g/n_inst/1e3:.1f} us, sum {(tot_k+tot_g)/n_inst/1e3:.1f} us")
gs=sorted(gaps); n=len(gs)
print(f"gaps: median {gs[n//2]/1e3:.2f} us, p10 {gs[n//10]/1e3:.2f}, p90 {gs[9*n//10]/1e3:.2f}, max {gs[-1]/1e3:.1f}; gaps >= 2 us hold {sum(g for g in gs if g>=2000)/n_inst/1e3:.1f} us/instance in {sum(1 for g in gs if g>=2000)/n_inst:.1f} gaps/instance")
print("kernels per step-graph instance (us, count, grid, name):")
for (nm,gx,gy,gz),(cnt,t) in sorted(agg.items(),key=lambda x:-x[1][1])[:14]:
    print(f"  {t/n_inst:6.1f} us x{cnt/n_inst:4.1f} grid {gx}x{gy}x{gz}  {nm[:74]}")
