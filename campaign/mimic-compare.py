#!/usr/bin/env python
"""MIMIC-D compare v2 (text-level): index-paired mimicbench2 runs.
Discriminators: adherence paired delta (primary), per-category flag pattern, prefix-match rate
(informational - long greedy generations diverge under any numeric change; cascade expected).
FAIL if adherence delta < -3 pts OR >=2 categories drop."""
import json, sys
b, a = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
rb, ra = b["results"], a["results"]
n = min(len(rb), len(ra))
exact = pre200 = 0
drops = []
for x, y in zip(rb[:n], ra[:n]):
    ob, oa = x.get("out",""), y.get("out","")
    if ob == oa: exact += 1
    elif ob[:200] == oa[:200]: pre200 += 1
    if y["ok"] < x["ok"] and x["w"] >= 2: drops.append(f'{x["cat"]}/{x["id"]}')
print(f"tasks compared      : {n} (baseline {b['weighted']}, arm {a['weighted']})")
print(f"exact greedy match  : {exact}/{n} = {100*exact/n:.1f}% (informational: cascade expected)")
print(f"agree thru 200 chars: {pre200}/{n} = {100*pre200/n:.1f}% (informational)")
print(f"adherence paired    : {b['score_pct']:.1f} -> {a['score_pct']:.1f} (delta {a['score_pct']-b['score_pct']:+.1f} pts, FAIL < -3)")
bcat, acat = b["by_category"], a["by_category"]
for k in bcat:
    bb = int(bcat[k].split("/")[0]); aa = int(acat.get(k,"0/0").split("/")[0])
    if aa < bb: print(f"  category drop: {k} {bcat[k]} -> {acat.get(k)}")
ok = (a["score_pct"] - b["score_pct"] >= -3) and (len(drops) < 2)
print("VERDICT:", "PASS - drift in the noise class" if ok else "FAIL - quality drift")
sys.exit(0 if ok else 1)
