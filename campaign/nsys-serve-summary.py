#!/usr/bin/env python3
"""campaign/nsys-serve-summary.py — read a serve-mode anatomy sqlite (campaign/nsys-serve-anatomy.sh) and print
where the GPU is idle and what the engine's serve thread does there.

    .venv/bin/python campaign/nsys-serve-summary.py campaign/results/<label>-anatomy/anatomy.sqlite [--pid N] [--gap-us 100]

The trace is graph-level (--cuda-graph-trace=graph): kernels inside a graph are one GRAPH_TRACE interval each, so
the GPU timeline is the union of the engine's graph executions, eager kernels, memcpys and memsets.  For every idle
gap at least --gap-us long the summary names the GPU activity before and after it, the CUDA API calls the serve
thread was inside while the GPU waited, and the fraction of the gap that thread spent on a CPU (SCHED_EVENTS).
A diagnostic reader: nothing it prints is a tracker point.
"""
import argparse, bisect, sqlite3, statistics as st, sys
from collections import Counter, defaultdict

PID_MASK = 0xFFFFFF
API_NAME = "cudaGraphLaunch"


def pid_of(g):
    return (g >> 24) & PID_MASK


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def union(intervals):
    """Merged (start, end) list of [(s, e), ...]; the caller sorts by start."""
    out = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--pid", type=int, default=0, help="engine pid (default: the 'strata' process with the most API rows)")
    ap.add_argument("--gap-us", type=float, default=100.0, help="idle gaps at least this long are attributed one by one")
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    q = lambda s, *args: c.execute(s, args).fetchall()

    # ---- scope: the engine process, its serve thread, the capture span ------------------------------------------------
    if not a.pid:
        row = q("""select (r.globalTid >> 24) & ?, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r
                   where (r.globalTid >> 24) & ? in (select globalPid / 16777216 % 16777216 from PROCESSES where name = 'strata')
                   group by 1 order by 2 desc limit 1""", PID_MASK, PID_MASK)
        a.pid = row[0][0] if row else 0
    thr = q("""select r.globalTid & ?, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r
               where (r.globalTid >> 24) & ? = ? group by 1 order by 2 desc""", PID_MASK, PID_MASK, a.pid)
    if not thr:
        sys.exit(f"no CUDA API rows for pid {a.pid}")
    serve_tid = thr[0][0]
    t0, t1 = q("select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME")[0]
    # GRAPH_TRACE shares this clock but keeps the session's pre-capture history (its first row is the warm-up);
    # every GPU table is therefore windowed to the API span, which is the collection.
    span = t1 - t0
    names = dict(q("select s.id, s.value from StringIds s"))
    tname = dict(q("select t.globalTid & ?, s.value from ThreadNames t join StringIds s on s.id = t.nameId", PID_MASK))
    print(f"engine pid {a.pid} | serve thread tid {serve_tid} ('{tname.get(serve_tid, '?')}') | capture {span / 1e6:.0f} ms")
    print("engine threads by API rows: " + ", ".join(f"{tid}('{tname.get(tid, '?')}'):{n}" for tid, n, _ in thr[:6]))

    # ---- serve-thread API table ------------------------------------------------------------------------------------
    api = q("""select r.start, r.end, names.value from CUPTI_ACTIVITY_KIND_RUNTIME r
               join StringIds names on names.id = r.nameId
               where (r.globalTid >> 24) & ? = ? and r.globalTid & ? = ? order by r.start""", PID_MASK, a.pid, PID_MASK, serve_tid)
    api_by_name = defaultdict(list)
    for s, e, n in api:
        api_by_name[n].append(e - s)
    print("\n== serve thread: CUDA API calls inside the capture (us = microseconds)")
    print(f"{'API':34s} {'calls':>7s} {'total ms':>9s} {'mean us':>8s} {'p50 us':>7s} {'p99 us':>7s} {'% span':>6s}")
    for n, d in sorted(api_by_name.items(), key=lambda kv: -sum(kv[1]))[:14]:
        print(f"{n[:34]:34s} {len(d):7d} {sum(d) / 1e6:9.1f} {st.mean(d) / 1e3:8.1f} {pct(d, .5) / 1e3:7.1f} "
              f"{pct(d, .99) / 1e3:7.1f} {100 * sum(d) / span:6.2f}")
    print(f"{'ALL':34s} {len(api):7d} {sum(e - s for s, e, _ in api) / 1e6:9.1f}")

    # ---- graph launches: GPU wall and the host submission that launched them ----------------------------------------
    gl = q("""select g.start, g.end, g.streamId, g.graphId, r.start, r.end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE g
              left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = g.correlationId and r.nameId in
                (select id from StringIds where value like ?)
              where (g.globalPid >> 24) & ? = ? and g.start >= ? and g.end <= ? order by g.start""",
              API_NAME + "%", PID_MASK, a.pid, t0, t1)
    per_graph = defaultdict(lambda: {"n": 0, "wall": [], "host": []})
    for gs, ge, stream, gid, rs, re_ in gl:
        d = per_graph[(gid, stream)]
        d["n"] += 1
        d["wall"].append(ge - gs)
        if rs is not None:
            d["host"].append(re_ - rs)
    print("\n== graph executions by graphId (GPU wall per execution; host = the cudaGraphLaunch call that submitted it)")
    print(f"{'graphId':>7s} {'stream':>6s} {'execs':>6s} {'GPU total ms':>12s} {'wall p50 us':>10s} {'wall mean us':>11s} {'host p50 us':>10s} {'host mean us':>11s}")
    for (gid, stream), d in sorted(per_graph.items(), key=lambda kv: -sum(kv[1]["wall"]))[:12]:
        print(f"{gid:7d} {stream:6d} {d['n']:6d} {sum(d['wall']) / 1e6:12.1f} {pct(d['wall'], .5):10.0f} "
              f"{st.mean(d['wall']):11.0f} {pct(d['host'], .5) if d['host'] else float('nan'):10.1f} "
              f"{st.mean(d['host']) if d['host'] else float('nan'):11.1f}")

    # ---- GPU busy / idle (timestamps are nanoseconds; everything prints in ms) ---------------------------------------
    gpu = []
    for s_, e_, gid in q("select start, end, graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where (globalPid >> 24) & ? = ? "
                         "and start >= ? and end <= ?", PID_MASK, a.pid, t0, t1):
        gpu.append((s_, e_, f"graph{gid}"))
    for tbl, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                      ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        for s_, e_ in q(f"select start, end from {tbl} where (globalPid >> 24) & ? = ? and start >= ? and end <= ?",
                        PID_MASK, a.pid, t0, t1):
            gpu.append((s_, e_, kind))
    gpu.sort()
    busy = union([[s_, e_] for s_, e_, _ in gpu])
    busy_ns = sum(e - s_ for s_, e in busy)
    idle_ns = span - busy_ns
    print(f"\n== GPU timeline (the engine's activities): busy {busy_ns / 1e6:.0f} ms of {span / 1e6:.0f} ms = "
          f"{100 * busy_ns / span:.1f}% | idle {idle_ns / 1e6:.0f} ms")
    gaps = []                                   # (length ns, start, end)
    for i in range(1, len(busy)):
        g = busy[i][0] - busy[i - 1][1]
        if g > 0:
            gaps.append((g, busy[i - 1][1], busy[i][0]))
    print(f"{'idle gap':>12s} {'count':>7s} {'total ms':>9s} {'% of idle':>9s}")
    for lo, hi in ((0, 1), (1, 5), (5, 20), (20, 50), (50, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 1e18)):
        ds = [g for g, _, _ in gaps if lo * 1e3 <= g < hi * 1e3]
        if ds:
            print(f"{f'{lo:.0f}-{hi:.0f} us':>12s} {len(ds):7d} {sum(ds) / 1e6:9.1f} {100 * sum(ds) / idle_ns:9.1f}")

    # ---- rounds: consecutive executions of the dominant long graph (a cohort's window graph) -----------------------------
    starts = defaultdict(list)
    for s_, e_, k in gpu:
        if k.startswith("graph") and (e_ - s_) >= 5_000_000:
            starts[k].append(s_)
    dom = max(starts, key=lambda k: len(starts[k]))
    ds_ = sorted(starts[dom])
    rounds = []                                 # (period ns, idle ns inside)
    gi = 0
    gaps_sorted = sorted(gaps, key=lambda g: g[1])
    for i in range(len(ds_) - 1):
        lo_, hi_ = ds_[i], ds_[i + 1]
        idle = sum(g for g, gs, ge in gaps_sorted if gs >= lo_ and ge <= hi_)
        rounds.append((hi_ - lo_, idle))
    per = [r[0] / 1e6 for r in rounds]
    med = st.median(per)
    print(f"\n== rounds keyed on {dom}: {len(rounds)} rounds | period ms p10 {pct(per, .1):.2f} p50 {med:.2f} "
          f"p90 {pct(per, .9):.2f} max {max(per):.2f}")
    thr = 1.25 * med * 1e6                       # periods are nanoseconds, med is printed in ms
    steady = [r for r in rounds if r[0] <= thr]
    longr = [r for r in rounds if r[0] > thr]
    idl = [r[1] / 1e6 for r in steady]
    print(f"steady rounds (period <= 1.25 x p50): {len(steady)} | idle per round ms p50 {st.median(idl):.2f} "
          f"p90 {pct(idl, .9):.2f} | {100 * sum(r[1] for r in steady) / sum(r[0] for r in steady):.1f}% of their time idle")
    if longr:
        print(f"long rounds: {len(longr)} | period ms p50 {st.median([r[0] / 1e6 for r in longr]):.2f} max {max(r[0] for r in longr) / 1e6:.2f} | "
              f"{sum(r[0] for r in longr) / 1e6:.0f} ms of the capture, {sum(r[1] for r in longr) / 1e6:.0f} ms of it idle "
              f"({100 * sum(r[1] for r in longr) / sum(r[0] for r in longr):.1f}%)")

    # ---- what the GPU waited for: the serve thread's API calls inside idle gaps, by round type ---------------------------
    api_starts = [s_ for s_, _, _ in api]

    def api_overlaps(gs, ge):
        """[(name, overlap ns)] for the serve thread's API calls intersecting [gs, ge)."""
        i = max(0, bisect.bisect_right(api_starts, gs) - 2)
        out = []
        while i < len(api) and api[i][0] < ge:
            s_, e_, n = api[i]
            o = min(ge, e_) - max(gs, s_)
            if o > 0:
                out.append((n, o))
            i += 1
        return out

    def api_active_at(t):
        """the API call the serve thread is inside at time t, or None (rows are sorted by start and barely overlap)."""
        i = bisect.bisect_right(api_starts, t) - 1
        for j in range(i, max(-1, i - 4), -1):
            if api[j][1] >= t:
                return api[j][2]
        return None

    def round_type(t):
        i = bisect.bisect_right(ds_, t) - 1
        if i < 0 or i + 1 >= len(ds_):
            return "edge"
        return "steady" if rounds[i][0] <= thr else "long"

    gpu_starts = [g[0] for g in gpu]

    def gpu_kind_at(t):
        """the GPU activity (kind) that starts at or ends at time t."""
        i = bisect.bisect_right(gpu_starts, t) - 1
        for j in range(max(0, i - 2), min(len(gpu), i + 3)):
            if abs(gpu[j][0] - t) < 50 or abs(gpu[j][1] - t) < 50:
                return gpu[j][2]
        return "?"

    big = [(g, gs, ge) for g, gs, ge in gaps if g >= a.gap_us * 1e3]
    print(f"\n== idle gaps >= {a.gap_us:.0f} us: {len(big)} gaps, {sum(g for g, _, _ in big) / 1e6:.0f} ms "
          f"({100 * sum(g for g, _, _ in big) / idle_ns:.1f}% of all idle)")
    for kind in ("steady", "long"):
        sel = [(g, gs, ge) for g, gs, ge in big if round_type((gs + ge) / 2) == kind]
        if not sel:
            continue
        api_time, ended, edges = Counter(), Counter(), Counter()
        for g, gs, ge in sel:
            for n, o in api_overlaps(gs, ge):
                api_time[n] += o
            ended[api_active_at(ge) or "(no API call on the serve thread: host code)"] += g
            edges[(gpu_kind_at(gs), gpu_kind_at(ge))] += g
        tot = sum(g for g, _, _ in sel)
        print(f"-- in {kind} rounds: {len(sel)} gaps, {tot / 1e6:.0f} ms")
        print("   GPU activity before -> after the gap (ms):")
        for (b_, aft), ns in edges.most_common(4):
            print(f"      {b_[:26]:26s} -> {aft[:26]:26s} {ns / 1e6:7.1f}")
        print("   serve-thread API time inside the gaps (ms; the rest is host code outside CUDA):")
        for n, ns in api_time.most_common(5):
            print(f"      {n[:44]:44s} {ns / 1e6:7.1f}  {100 * ns / tot:5.1f}%")
        print("   what was on the serve thread when the GPU resumed (ms of gap):")
        for n, ns in ended.most_common(4):
            print(f"      {n[:60]:60s} {ns / 1e6:7.1f}")

    print(f"\n== the 8 longest idle gaps")
    for g, gs, ge in sorted(big, reverse=True)[:8]:
        ov = sorted(api_overlaps(gs, ge), key=lambda kv: -kv[1])[:3]
        print(f"  {g / 1e6:7.2f} ms at +{(gs - t0) / 1e9:8.3f} s [{round_type((gs + ge) / 2)}] | "
              f"{gpu_kind_at(gs)[:16]} -> {gpu_kind_at(ge)[:16]} | "
              + ", ".join(f"{n[5:28]} {o / 1e6:.2f}" for n, o in ov))


if __name__ == "__main__":
    main()
