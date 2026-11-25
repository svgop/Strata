#!/usr/bin/env python3
"""campaign/nsys-perf-stalls.py — who owns the GPU's stalls: the serve thread's host call chains inside each idle gap.

    .venv/bin/python campaign/nsys-perf-stalls.py <anatomy.sqlite> <perf.serve.data> [--min-ms 3] [--pid N]

Reads the GPU timeline of an nsys-serve-anatomy.sh capture made with its `perf` argument: the engine's serve thread
was sampled by perf at a fixed frequency with DWARF call chains, and nsys records two clock readings of its own zero
(TARGET_INFO_SESSION_START_TIME: monotonic and UTC); the reader proves which one aligns with perf's sample times.  Every idle gap at least --min-ms long is listed, the
perf samples inside it are collected, and each sample is attributed to its ANCHOR: walking the call chain from the
root, the first frame that is neither the C runtime, `main`, nor one of main's anonymous lambdas (the lambda's number
is kept as context).  Samples exist only while the thread is on a CPU, so a stall's sample count is its host-CPU
time; the rest of the stall the thread spent blocked.  A diagnostic: never a tracker point.
"""
import argparse, bisect, re, sqlite3, subprocess, sys
from collections import Counter, defaultdict

PID_MASK = 0xFFFFFF
SKIP_ROOT = ("__libc_start_main", "_start", "__libc_start", "__GI___libc_start", "main")


def symbol_of(sym):
    return sym.split("+0x")[0]


def anchor_of(chain):
    """chain: engine-space frames root -> leaf (strings).  Returns (anchor, lambda context)."""
    ctx = ""
    for i in range(len(chain) - 1, -1, -1):
        s = chain[i]
        if s == "main" or s.startswith("main::{lambda"):
            if s.startswith("main::{lambda"):
                m = re.search(r"\}#(\d+)\}::operator", s)
                ctx = f"lambda#{m.group(1)}" if m else "lambda"
            continue
        if any(s.startswith(k) for k in SKIP_ROOT):
            continue
        return s, ctx
    return "(no frame above the root)", ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("perf_data")
    ap.add_argument("--min-ms", type=float, default=3.0)
    ap.add_argument("--pid", type=int, default=0)
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    q = lambda s, *args: c.execute(s, args).fetchall()

    if not a.pid:
        row = q("""select (r.globalTid >> 24) & ?, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r
                   where (r.globalTid >> 24) & ? in (select globalPid / 16777216 % 16777216 from PROCESSES where name = 'strata')
                   group by 1 order by 2 desc limit 1""", PID_MASK, PID_MASK)
        a.pid = row[0][0] if row else 0
    t0, t1 = q("select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME")[0]
    # The capture's zero in two clocks.  perf's sample times are CLOCK_MONOTONIC on this box; nsys records the
    # monotonic reading (systemClockNs) beside the UTC one.  Which of the two aligns is decided by the clock check.
    utc0, mono0 = q('select "utcEpochNs", "systemClockNs" from TARGET_INFO_SESSION_START_TIME')[0]
    zero = mono0
    iv = []
    for tbl, has_pid in (("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", True), ("CUPTI_ACTIVITY_KIND_KERNEL", True),
                         ("CUPTI_ACTIVITY_KIND_MEMCPY", True), ("CUPTI_ACTIVITY_KIND_MEMSET", True)):
        iv += q(f"select start, end from {tbl} where (globalPid >> 24) & ? = ? and start >= ? and end <= ?",
                PID_MASK, a.pid, t0, t1)
    iv.sort()
    busy = []
    for s, e in iv:
        if busy and s <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], e)
        else:
            busy.append([s, e])
    gaps = [(busy[i][0] - busy[i - 1][1], busy[i - 1][1], busy[i][0])
            for i in range(1, len(busy)) if busy[i][0] - busy[i - 1][1] >= a.min_ms * 1e6]

    # ---- perf samples: (wall ns, anchor, ctx, engine chain root -> leaf) --------------------------------------------------
    hdr = re.compile(r"^\S+\s+(\d+)\s+(\d+)\.(\d+):\s+\d+\s+(\S+):")
    frm = re.compile(r"^\s*([0-9a-f]+)\s+(.*?)\s+\((.*)\)\s*$")
    samples = []                                  # (wall ns, anchor, ctx, chain)
    chain, wall, ok = [], None, False
    out = subprocess.run(["perf", "script", "-i", a.perf_data], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = hdr.match(line)
        if m:
            if ok and chain:
                eng = [symbol_of(s) for _, s, d in chain if "kernel" not in d]
                anc, ctx = anchor_of(eng)
                samples.append((wall, anc, ctx, eng))
            chain, wall, ok = [], int(m.group(2)) * 10**9 + int(m.group(3)), m.group(4).startswith("cpu/cycles")
            continue
        f = frm.match(line)
        if f and ok:
            chain.append((f.group(1), f.group(2), f.group(3)))
    if ok and chain:
        eng = [symbol_of(s) for _, s, d in chain if "kernel" not in d]
        anc, ctx = anchor_of(eng)
        samples.append((wall, anc, ctx, eng))
    samples.sort()
    times = [s[0] for s in samples]
    inside = [s for s in samples if t0 + zero <= s[0] <= t1 + zero]
    rate = len(inside) / ((t1 - t0) / 1e9) if t1 > t0 else 0.0
    # ---- the clock check: a sample whose leaf is a CUDA API call must lie inside that call's own API interval -------
    api = q("""select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId
               where (r.globalTid >> 24) & ? = ? order by r.start""", PID_MASK, a.pid)
    api_starts = [x[0] for x in api]
    hit = test = 0
    for w, anc, ctx, eng in inside[::5]:
        leaf = eng[-1] if eng else ""
        if leaf.startswith("cuda"):
            i = max(0, bisect.bisect_right(api_starts, w - zero) - 2)
            test += 1
            for j in range(i, min(len(api), i + 5)):
                if api[j][0] - 50_000 <= w - zero <= api[j][1] + 50_000 and api[j][2].startswith(leaf):
                    hit += 1
                    break
    aligned = 100 * hit / test if test else 0.0
    if aligned < 80 and test:                     # the monotonic zero failed: try the UTC one before giving up
        alt_zero, hit2 = utc0, 0
        for w, anc, ctx, eng in inside[::5]:
            leaf = eng[-1] if eng else ""
            if leaf.startswith("cuda"):
                i = max(0, bisect.bisect_right(api_starts, w - alt_zero) - 2)
                for j in range(i, min(len(api), i + 5)):
                    if api[j][0] - 50_000 <= w - alt_zero <= api[j][1] + 50_000 and api[j][2].startswith(leaf):
                        hit2 += 1
                        break
        if 100 * hit2 / test > aligned:
            zero, aligned = alt_zero, 100 * hit2 / test
    print(f"clock alignment: {aligned:.0f}% of sampled CUDA-API leaves lie inside their own API interval "
          f"(test {test}; below 80% the two clocks do not agree and nothing below is attributable)")
    print(f"engine pid {a.pid} | capture {(t1 - t0) / 1e6:.0f} ms | perf samples in the capture: {len(inside)} "
          f"({rate:.0f}/s on the serve thread)")
    print(f"stalls >= {a.min_ms:.0f} ms: {len(gaps)}, {sum(g for g, _, _ in gaps) / 1e6:.0f} ms "
          f"(the serve thread's host-CPU time inside them is sample_count / {rate:.0f} per second)")

    def in_gap(gs, ge):
        lo, hi = bisect.bisect_left(times, zero + gs), bisect.bisect_right(times, zero + ge)
        return samples[lo:hi]

    classes = (("3-10 ms", 3, 10), ("10-30 ms", 10, 30), ("30+ ms", 30, 1e9))
    shown = set()
    for name, lo_ms, hi_ms in classes:
        sel = [(g, gs, ge) for g, gs, ge in gaps if lo_ms * 1e6 <= g < hi_ms * 1e6]
        if not sel:
            continue
        tot = sum(g for g, _, _ in sel)
        anchors, ctxs, chains = Counter(), Counter(), defaultdict(Counter)
        n = 0
        for g, gs, ge in sel:
            for w, anc, ctx, eng in in_gap(gs, ge):
                anchors[anc] += 1
                ctxs[(anc, ctx)] += 1
                n += 1
                chains[anc][tuple(x for x in eng if not x.startswith(("__", "pthread_", "pthread")))] += 1
        print(f"\n== stalls {name}: {len(sel)} stalls, {tot / 1e6:.0f} ms | serve-thread host-CPU time {n / rate / 1e3:.0f} ms "
              f"({100 * n / rate / tot if tot else 0:.0f}% of the stall time on a CPU, the rest blocked)")
        for anc, k in anchors.most_common(5):
            ctx = Counter({c: v for (an_, c), v in ctxs.items() if an_ == anc}).most_common(1)[0][0]
            print(f"   {100 * k / n if n else 0:5.1f}%  {k / rate / 1e3:6.1f} ms  {anc[:70]:70s} [{ctx}]")
            if anc not in shown:
                shown.add(anc)
                ch = chains[anc].most_common(1)[0][0]
                print("          chain (root -> leaf, engine frames): " + " <- ".join(x[:46] for x in reversed(ch[-7:])))


if __name__ == "__main__":
    main()
