#!/usr/bin/env python3
"""F-459 - the both-cohorts CPU-miss share, read from `cohorts: tel` lines.

Usage: f459-shared.py <telemetry.txt> [<telemetry.txt> ...]

Each file is one boot's `cohorts: tel` lines (logs/custom-iq3_s.log, this boot's segment).  A line carries, when
STRATA_COHORT_SHARED_MISS and STRATA_COHORT_ROUND_TELEMETRY are both set, the round's record as
`shared +S of cpu +b (A +a)`: S = the (layer, expert) blobs cohort B computed on the CPU that cohort A computed too,
b and a the cohorts' distinct (layer, expert) counts.

Population (fixed before the data was read): rounds with >= 10 rows and since-prev < 100 ms (inside a run), and
b > 0 (a share needs a denominator).  The share is per round, S / b.  Fast round: window < 28 ms.

The price is arithmetic on the measured share, not a measurement: pool time removed = 0.42 x share x 0.46 (the
pool's share of a round) x 0.715 (the window's ms per ms of pool) = 0.1381 x share of the round.
"""
import re
import sys

NUM = r"[+-]?\d+(?:\.\d+)?"
LINE = re.compile(
    r"cohorts: tel rows (?P<ra>\d+)\+(?P<rb>\d+) "
    r"stage " + NUM + r"/" + NUM + r" go " + NUM + r"/" + NUM + r" poll " + NUM + r"/" + NUM + r" "
    r"window (?P<win>" + NUM + r") tail " + NUM + r" since-prev (?P<since>" + NUM + r") "
    r"\| wait " + NUM + r" pool (?P<pool>" + NUM + r") \[gate/up " + NUM + r" quantize " + NUM + r" down " + NUM + r"; call " + NUM + r"\] "
    r"\| hits " + NUM + r" cpu (?P<cpu>" + NUM + r") pcie " + NUM + r" swaps " + NUM +
    r"(?: shared \+(?P<sh>\d+) of cpu \+(?P<cb>\d+) \(A \+(?P<ca>\d+)\)(?: C \+(?P<shc>\d+) of \+(?P<cc>\d+))?)?\s*$"
)
MIN_ROWS, MAX_SINCE, FAST_WINDOW = 10, 100.0, 28.0
POOL_SHARE_OF_ROUND, WINDOW_MS_PER_POOL_MS, LEVER = 0.46, 0.715, 0.42


def pct(xs, p):
    """Linear interpolation between order statistics (numpy's default method)."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    r = (p / 100.0) * (n - 1)
    lo = int(r)
    hi = min(lo + 1, n - 1)
    return s[lo] + (r - lo) * (s[hi] - s[lo])


def stats(xs):
    """n, mean, median, p10, p90, min, max - one line, or `none`."""
    if not xs:
        return "none"
    return ("n %d  mean %.3f  median %.3f  p10 %.3f  p90 %.3f  min %.3f  max %.3f"
            % (len(xs), sum(xs) / len(xs), pct(xs, 50), pct(xs, 10), pct(xs, 90), min(xs), max(xs)))


def read(path):
    rows, unmatched, no_field = [], 0, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = LINE.match(raw.rstrip("\n"))
            if not m:
                if "cohorts: tel" in raw:
                    unmatched += 1
                    if "shared +" in raw:
                        no_field += 1
                continue
            d = m.groupdict()
            r = {k: (int(v) if v is not None and re.fullmatch(r"\d+", v) else float(v) if v is not None else None)
                 for k, v in d.items()}
            rows.append(r)
    return rows, unmatched


def report(path):
    rows, unmatched = read(path)
    print("=" * 100)
    print("F-459  both-cohorts CPU-miss share  -  %s" % path)
    print("=" * 100)
    tel = len(rows)
    with_field = sum(1 for r in rows if r["sh"] is not None)
    both = sum(1 for r in rows if r["ra"] > 0 and r["rb"] > 0)
    full_with_field = sum(1 for r in rows if r["ra"] > 0 and r["rb"] > 0 and r["sh"] is not None)
    print("tel lines %d | with the F-459 field %d | both cohorts populated %d, of which with the field %d | tel lines the reader could not parse %d"
          % (tel, with_field, both, full_with_field, unmatched))
    splits = {}
    for r in rows:
        k = "%d+%d" % (r["ra"], r["rb"])
        splits[k] = splits.get(k, 0) + 1
    print("row splits (A+B): " + ", ".join("%s x%d" % kv for kv in sorted(splits.items(), key=lambda kv: -kv[1])[:8]))

    armed = [r for r in rows if r["sh"] is not None]
    runs, cur = [], []
    for r in armed:
        if cur and r["since"] >= MAX_SINCE:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    inside = [r for r in armed if r["ra"] + r["rb"] >= MIN_ROWS and r["since"] < MAX_SINCE]
    kept = [r for r in inside if r["cb"] > 0]
    print("runs (split at since-prev >= %g ms) %d | rounds inside a run (rows >= %d, since-prev < %g) %d | of those with B CPU experts (b > 0) %d, excluded for b = 0: %d"
          % (MAX_SINCE, len(runs), MIN_ROWS, MAX_SINCE, len(inside), len(kept), len(inside) - len(kept)))

    # the instrument's own check: a + b is the same unit as the line's `cpu` delta (multi_misses)
    eq = sum(1 for r in kept if r["ca"] + r["cb"] == r["cpu"])
    lt = sum(1 for r in kept if r["ca"] + r["cb"] < r["cpu"])
    print("reconciliation, a + b against the line's own `cpu` (multi_misses) delta: equal %d of %d, smaller %d, larger %d"
          % (eq, len(kept), lt, len(kept) - eq - lt))
    if kept:
        pool_frac = sum(r["pool"] for r in kept if r["pool"] > 0) / sum(r["win"] for r in kept)
        print("this boot's own anatomy on the kept rounds: pool / window = %.3f (the price assumes 0.46 x 0.715 = %.3f of the round)"
              % (pool_frac, POOL_SHARE_OF_ROUND * WINDOW_MS_PER_POOL_MS))

    def share_block(tag, sel):
        print("[%s] rounds %d | rows A %.2f B %.2f | window ms mean %.2f median %.2f | CPU experts per round A %s | B %s"
              % (tag, len(sel), sum(r["ra"] for r in sel) / max(len(sel), 1), sum(r["rb"] for r in sel) / max(len(sel), 1),
                 sum(r["win"] for r in sel) / max(len(sel), 1), pct([r["win"] for r in sel], 50),
                 stats([r["ca"] for r in sel]), stats([r["cb"] for r in sel])))

    print("-" * 100)
    print("COHORT B: share = S / b per round")
    print("-" * 100)
    if not kept:
        print("no kept round: nothing to read")
    share_block("all kept", kept)
    sh = [r["sh"] / r["cb"] for r in kept]
    print("  share: " + stats(sh))
    print("  pooled (sum S / sum b): %.4f  (S %d, b %d)" % (sum(r["sh"] for r in kept) / max(sum(r["cb"] for r in kept), 1),
                                                           sum(r["sh"] for r in kept), sum(r["cb"] for r in kept)))
    sh_tot = [r["sh"] / (r["ca"] + r["cb"]) for r in kept if r["ca"] + r["cb"] > 0]
    print("  S over both cohorts' CPU experts, S / (a + b): " + stats(sh_tot))
    fast = [r for r in kept if r["win"] < FAST_WINDOW]
    slow = [r for r in kept if r["win"] >= FAST_WINDOW]
    for tag, sel in (("fast, window < %g ms" % FAST_WINDOW, fast), ("slow, window >= %g ms" % FAST_WINDOW, slow)):
        share_block(tag, sel)
        if sel:
            print("  share: " + stats([r["sh"] / r["cb"] for r in sel]))
    other = [r for r in armed if r not in inside]
    if other:
        share_block("not kept (short rows or a gap): context only, not the measurement", other)
        if any(r["cb"] > 0 for r in other):
            print("  share: " + stats([r["sh"] / r["cb"] for r in other if r["cb"] > 0]))

    print("-" * 100)
    print("COHORT C (ternary): against A, as B is")
    print("-" * 100)
    c = [r for r in kept if r["cc"] is not None]
    if c:
        share_block("C, all kept", c)
        print("  share: " + stats([r["shc"] / r["cc"] for r in c if r["cc"] > 0]))
    else:
        print("none: no ternary round in this boot")

    print("-" * 100)
    print("PRICE - arithmetic on the measured share, not a measurement")
    print("-" * 100)
    if kept:
        k = LEVER * POOL_SHARE_OF_ROUND * WINDOW_MS_PER_POOL_MS
        print("round shortens by 0.42 x 0.46 x 0.715 x share = %.4f x share (the brief's ~0.13); conc = x / (1 - x)" % k)
        for name, s in (("mean", sum(sh) / len(sh)), ("median", pct(sh, 50)), ("p10", pct(sh, 10)), ("p90", pct(sh, 90))):
            x = k * s
            print("  %-6s share %.4f (%.1f%%) -> round %+.2f%% -> conc %+.2f%%" % (name, s, 100 * s, -100 * x, 100 * x / (1 - x)))
        exact = []
        for r in kept:
            a, b, s_ = r["ca"], r["cb"], r["sh"]
            pool = 46.8 * (a + b - s_) + 89.8 * s_   # us: one six-row call per blob, two for a blob both cohorts computed
            exact.append((LEVER * 89.8 * s_) / pool if pool > 0 else 0.0)   # the share of BOTH cohorts' pool time removed
        m = sum(exact) / len(exact)
        print("  cross-check, the same per-blob prices with the measured a and b (0.42 x 89.8 x S / pool, the share of both cohorts' pool time): mean %.4f -> round %+.2f%% -> conc %+.2f%%"
              % (m, -100 * m * POOL_SHARE_OF_ROUND * WINDOW_MS_PER_POOL_MS,
                 100 * (m * POOL_SHARE_OF_ROUND * WINDOW_MS_PER_POOL_MS) / (1 - m * POOL_SHARE_OF_ROUND * WINDOW_MS_PER_POOL_MS)))
        ruling = "ABOVE the ~15% line: the structure question is named and handed on" if sum(sh) / len(sh) >= 0.15 else "UNDER the ~15% line: the lever closes"
        print("  ruling by the price, on the mean share: " + ruling)
    else:
        print("no kept round: no price")


def main(argv):
    if len(argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    for path in argv[1:]:
        report(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
