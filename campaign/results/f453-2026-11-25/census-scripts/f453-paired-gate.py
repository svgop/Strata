#!/usr/bin/env python3
"""F-453 (run once): the quality ATTRIBUTION, paired.  Usage: f453-paired-gate.py <control.json> <adopted.json>
Both gates must have asked the same questions (the control is a --first N prefix of the adopted run's suite).  Per id:
control right/wrong against adopted right/wrong -> the four cells, the exact two-sided McNemar test on the discordant
pairs, and the per-category paired difference.  Read-only."""
import json, sys, math
from collections import defaultdict
ctl = {r["id"]: r for r in json.load(open(sys.argv[1]))["rows"] if "ok" in r}
ado = {r["id"]: r for r in json.load(open(sys.argv[2]))["rows"] if "ok" in r}
ids = [i for i in ctl if i in ado]
ok = lambda r: r["ok"] in (True, "True", 1, "1")
both = sum(ok(ctl[i]) and ok(ado[i]) for i in ids); neither = sum(not ok(ctl[i]) and not ok(ado[i]) for i in ids)
c_only = sum(ok(ctl[i]) and not ok(ado[i]) for i in ids); a_only = sum(not ok(ctl[i]) and ok(ado[i]) for i in ids)
n = len(ids); c = both + c_only; a = both + a_only
def two_sided_binom(b, c):
    m = b + c
    if m == 0: return 1.0
    k = min(b, c); p = sum(math.comb(m, j) for j in range(0, k + 1)) / 2 ** m
    return min(1.0, 2 * p)
print(f"paired over {n} questions (control {len(ctl)} rows, adopted {len(ado)} rows)")
print(f"  control {c}/{n} = {100*c/n:.1f}%   adopted {a}/{n} = {100*a/n:.1f}%   difference {100*(a-c)/n:+.1f} pts")
print(f"  both right {both} | both wrong {neither} | control only {c_only} | adopted only {a_only}")
print(f"  exact McNemar (two-sided) on the discordant pairs: p = {two_sided_binom(c_only, a_only):.3f}")
se = math.sqrt(max(1, c_only + a_only)) / n * 100
print(f"  standard error of the paired difference: {se:.2f} pts")
by = defaultdict(lambda: [0, 0, 0])
for i in ids:
    v = by[ctl[i]["cat"]]; v[0] += 1; v[1] += ok(ctl[i]); v[2] += ok(ado[i])
print("  per category (n, control, adopted):")
for cat, (m, cc, aa) in sorted(by.items(), key=lambda x: x[1][2] - x[1][1]):
    print(f"    {cat:17s} {m:3d}  {cc:3d}  {aa:3d}  {aa-cc:+d}")
