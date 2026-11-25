#!/usr/bin/env python3
"""campaign/reqlog-census.py - where the box's own request time goes, read from the engine log.

Two lines describe a request, and neither is the whole of it (F-467):

* The engine's accounting line (src/program/generate.cpp, "strata serve: prompt N tokens = R reused + X read in
  A ms ..., G generated in B ms ..., drafts accepted a of b, c checkpoints").  Its read term is complete: every
  request's prompt read is on it.  Its decode term is the SOLO lane's only.  Under --batch a request runs this
  path for one token and, when more are wanted, continues in the batch windows ("BADM <slot> 1",
  generate.cpp:17473): the line is printed there, at the ADMISSION, reading "1 generated ... drafts accepted 0
  of 0", and nothing in the log summarises the generation that follows.  Such a line is an admission record.
  F-466's census read those records as one-token generations and so undercounted decode (3,145 of 4,821 lines
  in the production boot).

* The server's completion line (serve/server.py:6055, "[strata] done: N tokens in X s (R tok/s) (stop,
  cancel=False)"): N is the request's true output, X its whole time from arrival to last token (prompt read
  included, to the second), R its rate since the first token.  It is printed for streamed requests that run to
  their end; a request without one is in no total here.

The log (logs/custom-iq3_s.log) is appended across boots, so every read is anchored at its boot.

This census answers the question the short probes cannot: the probes run ~1.1k-token prompts with long
generations, while the production log holds the traffic the operator actually sends.  It splits a boot's
request time into prompt reads (complete), solo-lane decode, and the batch lane's share (the server's total
less the other two, given as a bracket because the two lines carry no request id to pair them by); it reads
solo decode against context depth; and it reads the conversation cache's occupancy at every re-read.
Request-seconds are per request (concurrent requests overlap in wall time); the engine's prompt reads
run on the single window thread, so read seconds are close to wall seconds.

Usage:
  python3 campaign/reqlog-census.py                     # the boot with the most request-seconds
  python3 campaign/reqlog-census.py --boot "2026-11-22 15:59" [--log PATH] [--top 25]
Diagnostic: reads a log, touches nothing.
"""
import argparse
import bisect
import collections
import datetime as dt
import re
import statistics as st

BOOT_START = re.compile(r"^\[strata\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) engine started:")
# "N of M" is a read cancelled mid-way (#471): the tokens read are N.
PROMPT_LINE = re.compile(
    r"^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+)(?: of \d+)? read in (\d+) ms(?: \([\d.]+ tok/s\))?, "
    r"(\d+) generated in (\d+) ms(?: \([\d.]+ tok/s\))?, drafts accepted (\d+) of (\d+), (\d+) checkpoints")
DONE_LINE = re.compile(
    r"^\[strata\] done: ([\d,]+) tokens in ([\d,]+) s \(([\d.]+) tok/s\) \((\w+), cancel=(\w+)\)")
PARK_LINE = re.compile(
    r"^strata serve: conversation cache: parked (\d+) tokens.*? in (\d+(?:\.\d+)?) ms; parked=(\d+) bytes=(\d+) evictions=(\d+)")
RESTORE_LINE = re.compile(r"^strata serve: conversation cache: (restored|borrowed) (\d+) tokens.*? in (\d+(?:\.\d+)?) ms")
ELASTIC_LINE = re.compile(r"^strata elastic: slot \d+ (materialized|demoted) in (\d+(?:\.\d+)?) ms")

READ_BUCKETS = [(0, 0), (1, 64), (65, 256), (257, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 8192),
                (8193, 16384), (16385, 32768), (32769, 65536), (65537, 10**9)]
CTX_BUCKETS = [(0, 32768), (32769, 65536), (65537, 131072), (131073, 262144)]
KV_RESIDENT = 32768   # --kv-resident of the standing: cells of K/V resident in VRAM per QSA layer
BIG_READ = KV_RESIDENT    # a read past this re-creates K/V the VRAM window cannot hold


def bucket(value, edges):
    for lo, hi in edges:
        if lo <= value <= hi:
            return lo, hi
    return None


def label(lo, hi):
    return f"{lo:>6}-{hi if hi < 10**9 else 999999:<6}"


def read_log(path):
    """-> (boots [(line, datetime)], requests [dict], the cache state at each big read, parks, restores, elastic
    events, the server's completion lines [(line, tokens, seconds, rate, finish)])."""
    boots, reqs, dones = [], [], []
    cache = dict(slots=None, bytes=None, evictions=None)
    cache_at_big_read, parks, restores, elastic = [], [], [], collections.defaultdict(list)
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f, 1):
            m = BOOT_START.match(line)
            if m:
                boots.append((i, dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")))
                cache = dict(slots=None, bytes=None, evictions=None)
                continue
            m = DONE_LINE.match(line)
            if m:
                dones.append((i, int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")), float(m.group(3)), m.group(4)))
                continue
            m = PARK_LINE.match(line)
            if m:
                cache = dict(slots=int(m.group(3)), bytes=int(m.group(4)), evictions=int(m.group(5)))
                parks.append((int(m.group(1)), float(m.group(2)), i))
                continue
            m = RESTORE_LINE.match(line)
            if m:
                restores.append((m.group(1), int(m.group(2)), float(m.group(3)), i))
                continue
            m = ELASTIC_LINE.match(line)
            if m:
                elastic[m.group(1)].append((float(m.group(2)), i))
                continue
            m = PROMPT_LINE.match(line)
            if m:
                g = [int(x) for x in m.groups()]
                # an admission record: one token made in the solo lane and no drafts offered, the batch lane's work unseen
                r = dict(line=i, prompt=g[0], reused=g[1], read=g[2], read_ms=g[3], gen=g[4], gen_ms=g[5],
                         acc=g[6], drafts=g[7], admitted=(g[4] == 1 and g[7] == 0))
                reqs.append(r)
                if r["read"] > BIG_READ:
                    cache_at_big_read.append((r, dict(cache)))
    return boots, reqs, cache_at_big_read, parks, restores, elastic, dones


def boot_of(boots, starts, line):
    return bisect.bisect_right(starts, line) - 1


def size_table(title, rows, key_ms, key_tok, total_ms, extra=""):
    """rows: (lo, hi, n, sum_ms, sum_tok, median_ms)."""
    print(f"\n{title}{extra}")
    print("  tokens |     n |  total s | % of total |  mean ms | median ms | tok/s (sum tok / sum ms)")
    for lo, hi, n, ms, tok, med in rows:
        if n:
            print(f"  {label(lo, hi)} | {n:5d} | {ms / 1e3:8.1f} | {100 * ms / total_ms if total_ms else 0:9.1f}% |"
                  f" {ms / n:8.0f} | {med:9.0f} | {tok / ms * 1000 if ms else 0:10.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/custom-iq3_s.log")
    ap.add_argument("--boot", default=None, help='"YYYY-MM-DD HH:MM" of the boot\'s engine-start line; default: the boot with the most request-seconds')
    ap.add_argument("--top", type=int, default=12, help="boot rows to list")
    a = ap.parse_args()
    boots, reqs, big, parks, restores, elastic, dones = read_log(a.log)
    starts = [b[0] for b in boots]
    if not reqs:
        raise SystemExit("no per-request lines in the log")
    per = collections.defaultdict(list)
    for r in reqs:
        per[boot_of(boots, starts, r["line"])].append(r)
    served = collections.defaultdict(list)
    for d in dones:
        served[boot_of(boots, starts, d[0])].append(d)
    print(f"boots {len(boots)}, engine accounting lines {len(reqs)} ({sum(r['admitted'] for r in reqs):,} one-token records),"
          f" server completion lines {len(dones)}")
    print("\nPER BOOT: start | engine lines (one-token records) | read s | solo decode s | server lines | their request s | their tokens | read p50 | read max | acceptance (solo)")
    # seconds on both sides of the "or": the server's line counts seconds, the engine's milliseconds
    order = sorted(per, key=lambda b: -(sum(x[2] for x in served[b]) or sum(r["read_ms"] + r["gen_ms"] for r in per[b]) / 1e3))
    for b in order[:a.top]:
        R, reads, D = per[b], sorted(r["read"] for r in per[b]), served[b]
        dr = sum(r["drafts"] for r in R)
        print(f"{boots[b][1]:%m-%d %H:%M} | {len(R):5d} ({sum(r['admitted'] for r in R):5d}) | {sum(r['read_ms'] for r in R) / 1e3:7.0f} |"
              f" {sum(r['gen_ms'] for r in R) / 1e3:7.0f} | {len(D):5d} | {sum(x[2] for x in D):8,} | {sum(x[1] for x in D):10,} |"
              f" {st.median(reads):6.0f} | {reads[-1]:7,} | {dr and sum(r['acc'] for r in R) / dr or 0:.3f}")
    if a.boot:
        sel = [b for b in per if boots[b][1].strftime("%Y-%m-%d %H:%M") == a.boot]
        if not sel:
            raise SystemExit(f"no boot at {a.boot}")
        b = sel[0]
    else:
        b = order[0]
    R = per[b]
    D = served[b]
    RD = sum(r["read_ms"] for r in R)          # milliseconds, the unit every table below divides by
    GN = sum(r["gen_ms"] for r in R)
    T = RD + GN
    SD, ST = sum(x[2] for x in D), sum(x[1] for x in D)   # the server's line counts seconds
    print(f"\nSELECTED BOOT {boots[b][1]:%Y-%m-%d %H:%M:%S}: {len(R)} engine lines ({sum(r['admitted'] for r in R)} one-token records),"
          f" {len(D)} server completion lines")
    print("\nREQUEST TIME, TWO INSTRUMENTS (seconds)")
    print(f"  prompt reads (every engine line)                 {RD / 1e3:8,.0f}")
    print(f"  solo-lane decode (engine lines with a generation) {GN / 1e3:8,.0f}   {sum(1 for r in R if not r['admitted'])} requests,"
          f" {sum(r['gen'] for r in R):,} tokens")
    if D:
        print(f"  server-recorded request time, read included       {SD:8,.0f}   {len(D)} requests, {ST:,} tokens"
              f" (p50 {st.median(sorted(x[1] for x in D)):.0f} tokens a request, rate since the first token p50"
              f" {st.median(sorted(x[3] for x in D)):.0f} t/s)")
        lo = max(0.0, SD - (RD + GN) / 1e3)   # every read and every solo second inside the server's time is the batch lane's lower bound
        print(f"  batch-lane decode lies between {lo:,.0f} s (all reads and solo decode belong to requests with a server line) and"
              f" {SD:,.0f} s (none does); prompt reads are at most {100 * RD / 1e3 / SD if SD else 0:.1f} % of the server-recorded time")
    else:
        print("  no server completion lines in this boot: the batch lane's time is not on record")
    prom = sorted(r["prompt"] for r in R)
    print(f"  context per request: p50 {st.median(prom):,.0f}, p90 {prom[int(0.9 * (len(prom) - 1))]:,}, max {prom[-1]:,} tokens;"
          f" {sum(1 for r in R if r['prompt'] > KV_RESIDENT)} of {len(R)} requests run past the {KV_RESIDENT}-cell VRAM K/V window")
    cold = [r for r in R if r["read"] > BIG_READ]
    mid = [r for r in R if 8192 < r["read"] <= BIG_READ]
    small = [r for r in R if r["read"] <= 8192]
    print("\nPROMPT READS BY TERM (complete; shares of the engine-line seconds, whose decode is the solo lane's only)")
    for name, S in ((f"cold re-read (>{BIG_READ:,} tokens read)", cold), ("mid read (8,193-32,768)", mid),
                    ("per-turn delta read (<= 8,192 tokens)", small)):
        ms = sum(r["read_ms"] for r in S)
        print(f"  {name:38s} {ms / 1e3:7.0f} s  {100 * ms / T:5.1f}%   in {len(S):5d} requests"
              f"  ({sum(r['read'] for r in S) / ms * 1000 if ms else 0:,.0f} tok/s over them)")
    rows = []
    for lo, hi in READ_BUCKETS:
        S = [r for r in R if lo <= r["read"] <= hi]
        if S:
            ms = sum(r["read_ms"] for r in S)
            rows.append((lo, hi, len(S), ms, sum(r["read"] for r in S), st.median(r["read_ms"] for r in S)))
    size_table("PROMPT READS BY TOKENS READ", rows, "read_ms", "read", RD)
    rows = []
    for lo, hi in READ_BUCKETS:
        S = [r for r in R if lo <= r["gen"] <= hi and not r["admitted"]]
        if S:
            ms = sum(r["gen_ms"] for r in S)
            rows.append((lo, hi, len(S), ms, sum(r["gen"] for r in S), st.median(r["gen_ms"] for r in S)))
    size_table("SOLO-LANE DECODE BY TOKENS GENERATED (the batch lane's generations are on no engine line)", rows, "gen_ms", "gen", GN)
    print("\nSOLO-LANE DECODE RATE AGAINST CONTEXT DEPTH (requests generating >= 500 tokens): context | n | decode s | t/s (sum tok / sum ms) | p10 | p50 | p90 | acceptance")
    for lo, hi in CTX_BUCKETS:
        S = [r for r in R if lo <= r["prompt"] <= hi and r["gen"] >= 500 and r["gen_ms"] > 0]
        if S:
            v = sorted(r["gen"] / r["gen_ms"] * 1000 for r in S)
            dr = sum(r["drafts"] for r in S)
            ms = sum(r["gen_ms"] for r in S)
            print(f"  {label(lo, hi)} | {len(S):5d} | {ms / 1e3:7.0f} | {sum(r['gen'] for r in S) / ms * 1000:7.1f} |"
                  f" {v[int(0.1 * (len(v) - 1))]:6.1f} | {st.median(v):6.1f} | {v[int(0.9 * (len(v) - 1))]:6.1f} | {sum(r['acc'] for r in S) / dr if dr else 0:.3f}")
    print(f"\nCOLD RE-READS (>{BIG_READ:,} tokens): {len(cold)} requests, {sum(r['read'] for r in cold):,} tokens,"
          f" {sum(r['read_ms'] for r in cold) / 1e3:.0f} s; reused prefix p50 {st.median(sorted(r['reused'] for r in cold)) if cold else 0:,.0f} tokens")
    states = [c for r, c in big if boot_of(boots, starts, r["line"]) == b and c["bytes"] is not None]
    if states:
        sl = sorted(c["slots"] for c in states)
        by = sorted(c["bytes"] for c in states)
        print(f"  conversation cache at the park before each: slots used p50 {st.median(sl):.0f} (max {sl[-1]}),"
              f" bytes p50 {st.median(by) / 2**30:.2f} GiB (max {by[-1] / 2**30:.2f}); cumulative evictions by the last: {states[-1]['evictions']}")
    bp = [(p[0], p[1]) for p in parks if boot_of(boots, starts, p[2]) == b]
    br = [(x[1], x[2]) for x in restores if boot_of(boots, starts, x[3]) == b]
    for title, X in (("PARK (a finished turn's K/V, slot to the cache)", bp), ("RESTORE + BORROW (the next turn, cache to slot)", br)):
        rows = []
        for lo, hi in READ_BUCKETS:
            S = [x for x in X if lo <= x[0] <= hi]
            if S:
                ms = sum(x[1] for x in S)
                rows.append((lo, hi, len(S), ms, sum(x[0] for x in S), st.median(x[1] for x in S)))
        size_table(title, rows, "ms", "tok", sum(x[1] for x in X))
    print("\nELASTIC SLOT events of the selected boot: " + "; ".join(
        f"{k}: n={len([x for x in v if boot_of(boots, starts, x[1]) == b])},"
        f" {sum(x[0] for x in v if boot_of(boots, starts, x[1]) == b) / 1e3:.0f} s" for k, v in elastic.items()) or "none")


if __name__ == "__main__":
    main()
