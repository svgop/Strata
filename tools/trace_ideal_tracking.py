"""tools/trace_ideal_tracking.py - what a routing trace says the expert cache could hit, and what that would cost.

    python tools/trace_ideal_tracking.py TRACE.bin [--profile data/expert-profile.bin] [--rows A:B]
        [--horizons 10,20,40,80,160,320] [--eval-rows 2400] [--slots 9900] [--rows-per-window N]
        [--routed-windows 64,300,400,1000,2000] [--blob-bytes 2046400] [--link-gbps 49.6]

TRACE.bin is a `--dump-routing` file (the format tools/make_profile.py reads: int32 layer, int32 k, k ids, k weights).
Its records are taken per layer in file order, which is the pool's call order; with two cohorts the layers interleave,
so a record is never aligned with another layer's record.  None of this needs it to be: every statistic here is per
layer, and `--rows A:B` selects a leg by row index (counted from the trace's start for every layer).

For layer l the resident set at row t is the top n_l experts of layer l over the preceding X rows - ideal causal
tracking: the past known exactly, every swap instant and free.  n_l is the layer's share of the profile's first
`slots` pairs, which is what the engine fills (`--expert-cache-per-layer` off: one global counter in rank order, and
swaps stay inside a layer, so the cut fixes each layer's capacity for the boot).

Over the evaluated rows (the last `eval-rows` of the range) it prints:
  - the hit rate of ideal tracking at each horizon X, for the profile's cut, the trace's own cut at the same slot
    count, an equal split, and the profile's cut at larger slot counts (the capacity curve);
  - the experts per layer that 99 % of the entries need at the mid row of each horizon (the demand);
  - the swaps per window ideal tracking costs at each horizon: the entries into the set per row, summed over the
    layers, times `rows-per-window` (4.9 for a 6-slot window of this standing).

The X = whole-range limit of the same model is the static oracle.  A cut by a trace's own counts is its optimum:
covered routings are a sum of pair counts, so no per-layer allocation of the same slots covers more.  A real tier
differs from this one in two ways this tool prices separately - it swaps at a rate, and each swap moves bytes.

`--routed-windows W,...` prices a PROMPT READ of W tokens that streams only the experts its rows route: per window
of W consecutive rows, the distinct experts the GPU does not hold, their bytes and their link time, against what a
read that streams every non-resident expert moves.  The rows are decode rows, a stand-in for a prompt's tokens; the
arm it prices proves itself on a routing dump of real prompt chunks.
"""
import argparse
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NL, NE = 48, 512


def read_profile(path):
    """STRP v1 through the owner's reader (tools/make_profile.py): the ranked (layer, expert) pairs."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from make_profile import read_profile as owner
    return np.array(owner(str(path)), dtype=np.int64)


def read_trace(path):
    """per layer, the routed expert ids in call order: a list of (rows, k) arrays."""
    raw = np.fromfile(path, dtype=np.int32)
    k = int(raw[1])
    rec = 2 + 2 * k
    raw = raw[: (len(raw) // rec) * rec].reshape(-1, rec)
    if not (raw[:, 1] == k).all():
        raise SystemExit(f"{path}: k varies per record; this reader assumes one k")
    layer, ids = raw[:, 0], raw[:, 2:2 + k]
    return [ids[layer == l] for l in range(NL)], k


def capacities(pairs, slots):
    """the per-layer slot counts of the profile's first `slots` pairs"""
    return np.bincount(pairs[:slots, 0], minlength=NL)


def evaluate(E, lo, hi, eval_rows, caps, Xs):
    """E: per layer (rows, k) ids; rows [lo, hi) is the leg, its last `eval_rows` rows are evaluated."""
    hi = min(hi, min(e.shape[0] for e in E))
    t0 = max(lo, hi - eval_rows)
    k = E[0].shape[1]
    hit = np.zeros((len(Xs), len(caps)), np.int64)
    demand = np.zeros(NL, np.int64)
    tot = 0
    for l in range(NL):
        e = E[l]
        # cumulative per-expert counts: C[t] = counts of rows [0, t)
        C = np.vstack([np.zeros((1, NE), np.int64),
                       np.cumsum(np.stack([np.bincount(r, minlength=NE) for r in e]), axis=0)])
        mid = (t0 + hi) // 2
        xm = Xs[len(Xs) // 2]
        for t in range(t0, hi):
            et = e[t]
            tot += k
            for xi, xr in enumerate(Xs):
                c = C[t] - C[max(0, t - xr)]
                order = np.argsort(-c, kind="stable")
                rank = np.empty(NE, np.int32)
                rank[order] = np.arange(NE)
                r = rank[et]
                for ci, n in enumerate(caps):
                    hit[xi, ci] += int((r < n[l]).sum())
                if t == mid and xr == xm:   # the demand at the mid row of the mid horizon
                    demand[l] = int(np.searchsorted(np.cumsum(c[order]) / c.sum(), 0.99) + 1)
    return hit, tot, demand


def swap_rate(E, lo, hi, eval_rows, n_cap, Xs):
    """entries into the top-n set from one row to the next, summed over the layers: swaps per row at each horizon"""
    hi = min(hi, min(e.shape[0] for e in E))
    t0 = max(lo, hi - eval_rows)
    per_row = np.zeros(len(Xs), np.float64)
    for l in range(NL):
        e = E[l]
        C = np.vstack([np.zeros((1, NE), np.int64),
                       np.cumsum(np.stack([np.bincount(r, minlength=NE) for r in e]), axis=0)])
        prev = [None] * len(Xs)
        for t in range(t0, hi):
            for xi, xr in enumerate(Xs):
                c = C[t] - C[max(0, t - xr)]
                m = np.zeros(NE, bool)
                m[np.argpartition(-c, n_cap[l])[:n_cap[l]]] = True
                if prev[xi] is not None:
                    per_row[xi] += int((m & ~prev[xi]).sum())
                prev[xi] = m
    return per_row / max(1, hi - t0)


def resident_sets(pairs, slots):
    """the experts the GPU holds per layer: the profile's first `slots` pairs (the engine's fill order)"""
    res = [set() for _ in range(NL)]
    for l, e in pairs[:slots]:
        res[int(l)].add(int(e))
    return res


def routed_windows(E, lo, hi, resident, Ws):
    """windows of W consecutive rows -> per window, averaged over the layers, (distinct experts routed,
    distinct experts routed that the GPU does not hold): what a read of W tokens must stream if it streams
    only what its tokens route.  Per layer: rows are in call order, so a window is W consecutive rows."""
    hi = min(hi, min(e.shape[0] for e in E))
    out = []
    for W in Ws:
        per = np.zeros(((hi - lo) // W, 2))
        for w in range(per.shape[0]):
            for l in range(NL):
                ids = np.unique(E[l][lo + w * W: lo + (w + 1) * W])
                per[w] += (len(ids), len(set(ids.tolist()) - resident[l]))
        out.append((W, per / NL))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--profile", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--rows", default="", help="A:B row indices of the leg (default: the whole trace)")
    ap.add_argument("--horizons", default="10,20,40,80,160,320", help="rows of memory, comma separated")
    ap.add_argument("--eval-rows", type=int, default=2400)
    ap.add_argument("--slots", type=int, default=9900)
    ap.add_argument("--extra-slots", default="263,411,1000,2000", help="capacity points past --slots")
    ap.add_argument("--rows-per-window", type=float, default=0.0, help="converts swaps per row to swaps per window")
    ap.add_argument("--routed-windows", default="", help="comma separated window sizes in rows: prices a read that streams only the experts its rows route")
    ap.add_argument("--blob-bytes", type=int, default=2046400, help="mean bytes of one routed expert (F-453: 2,046,400)")
    ap.add_argument("--link-gbps", type=float, default=49.6, help="expert copies' rate on the link in GB/s (F-452 capture: 32.65 GB in 658 ms)")
    a = ap.parse_args()

    E, k = read_trace(a.trace)
    T = min(e.shape[0] for e in E)
    lo, hi = (int(x) for x in a.rows.split(":")) if a.rows else (0, T)
    Xs = [int(x) for x in a.horizons.split(",")]
    seed = read_profile(a.profile)
    leg_counts = np.stack([np.bincount(e[lo:hi].ravel(), minlength=NE) for e in E])
    order = np.argsort(-leg_counts.ravel(), kind="stable")
    fit = np.stack([order // NE, order % NE], axis=1)
    extras = [int(x) for x in a.extra_slots.split(",") if x]
    names = [f"cut{a.slots}", f"fit{a.slots}", "equal"] + [f"cut+{x}" for x in extras]
    caps = [capacities(seed, a.slots), capacities(fit, a.slots), np.full(NL, a.slots // NL)]
    caps += [capacities(seed, a.slots + x) for x in extras]
    if extras:   # the trace's own cut at the first extra capacity: allocation against capacity, apart
        names.append(f"fit+{extras[0]}")
        caps.append(capacities(fit, a.slots + extras[0]))

    hit, tot, demand = evaluate(E, lo, hi, a.eval_rows, caps, Xs)
    per_row = swap_rate(E, lo, hi, a.eval_rows, caps[0], Xs)
    print(f"trace {a.trace}: {T} rows per layer, k {k}; leg rows [{lo},{hi}) = {hi - lo}, evaluated {min(a.eval_rows, hi - lo)}")
    print("capacity ->   " + "  ".join(f"{n:>9s}" for n in names))
    for xi, xr in enumerate(Xs):
        sw = f" | {per_row[xi]:6.1f} swaps/row" + (f" = {per_row[xi] * a.rows_per_window:6.1f}/window" if a.rows_per_window else "")
        print(f"X={xr:4d} rows | " + "  ".join(f"{100 * hit[xi, ci] / tot:8.2f}%" for ci in range(len(caps))) + sw)
    print(f"experts per layer that 99 % of the entries need at the mid row (horizon {Xs[len(Xs) // 2]} rows): "
          f"median {int(np.median(demand))}, max {int(demand.max())}; the cut gives {int(caps[0].min())}-{int(caps[0].max())}")
    if a.routed_windows:
        Ws = [int(x) for x in a.routed_windows.split(",") if x]
        resident = resident_sets(seed, a.slots)
        nonres = sum(NE - len(r) for r in resident)
        print(f"\nexpert blobs one read moves (seed cut {a.slots}: {nonres} experts the GPU does not hold = "
              f"{nonres * a.blob_bytes / 1e9:.1f} GB = {nonres * a.blob_bytes / (a.link_gbps * 1e9):.2f} s at {a.link_gbps} GB/s)")
        print("window rows | windows | distinct experts routed/layer p50 | routed, not resident/layer p50 | of the non-resident | GB | s")
        for W, per in routed_windows(E, lo, hi, resident, Ws):
            if not len(per):
                continue
            dn = float(np.median(per[:, 1])) * NL
            print(f"{W:11d} | {len(per):7d} | {np.median(per[:, 0]):33.0f} | {np.median(per[:, 1]):34.0f} |"
                  f" {100 * dn / nonres:18.1f}% | {dn * a.blob_bytes / 1e9:5.1f} | {dn * a.blob_bytes / (a.link_gbps * 1e9):5.2f}")


if __name__ == "__main__":
    main()
