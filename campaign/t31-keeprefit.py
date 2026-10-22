#!/usr/bin/env python3
"""T31 - draft_cum keep-rule refit on the banked STRATA_SPEC_TRACE (roadmap item 3, F-260).

Zero-boot offline analysis. Reads campaign/results/spec-trace.txt rows written by the
STRATA_SPEC_TRACE instrument (src/program/generate.cpp:15418-15429):

    w T=<int> a=<int> resc=<int> d=<p0,p1,...>     (#d rows = T-1, capped 8; empty for T=1)

  T     = window rows offered (1 seed + T-1 drafts; standing S_mtp = --spec 6)
  a     = final accepted draft count, RESCUE-INCLUSIVE (the rescue path advances the
          exact-match walk, generate.cpp:15390-15413); max a = T-1
  resc  = CUMULATIVE typical/ratio rescue counter since process start (delta-detection:
          per-window rescues = resc[w] - resc[w-1]; a counter reset = a new boot segment,
          the file is opened in append mode)
  d_i   = the drafter's own probability for its drafted token at position i (the value the
          keep gate consumed). The standing gate is the FLAT rule at --spec-min-p 0.5
          (generate.cpp:15162-15164, custom-iq3_s-8081.json), so every printed d >= 0.5 and
          the first sub-gate draft is never printed (mtp chain early-stop, mtp.hpp:159).

Sections: (1) parse hygiene, (2) per-position x calibrated-bin acceptance (tested rows
only; exact-match primary, rescue-inclusive secondary) + monotonicity, (3) rescue-inclusive
accounting, (4) simulated keep rules vs the flat-0.5 gate (plug-in expected-token deltas,
measured-cost pricing), (5) NOW verdict, (6) post-retrain shift threshold for +2% net,
(7) Monte-Carlo cross-check of the plug-in math.

Cost model (T14C-fastmtp-acceptance-math-gates.md ss1.3, amended post-POOL_DIET by T26-2):
  W(offered) = W0 - c_row*(offered0 - offered), c_row in 2.2-2.6 ms/row post-diet
  (3.56 ms pre-diet central, 4.62 conc); one accepted token is worth W0/C0 ~ 4.7 ms.
  A marginal offered row pays iff its conditional acceptance m >
  c_row*C0/W0 (the crossing rule).tok/s = C/W, C = 1 + accepted/window; windows flat.

Rule form emulated (T26-2 ss c.1/c.2): draft j is offered iff
  c_j = prod_{i=0..j} acc_hat(d_i, i)  >=  th   (float32, ascending order)
T_new = 1 + (leading drafts passing), capped by the observed profile (this trace is
RIGHT-CENSORED at the flat-0.5 gate: rows the gate closed were never printed, so the
simulator can only CUT on observed rows; extension scenarios say so explicitly).

Assumptions (also printed by --assumptions):
  A1 positions independent given d (exact-match verify makes acceptance sequential;
     the calibrated marginals are CONDITIONAL on an unbroken prefix - tested rows only)
  A2 acc_hat(d, k) is the true conditional acceptance probability and is stable under
     the simulated policy change (offers-only-exact: trimming a chain never alters any
     other draft's bits, T26-2 ss c.3 bit-identity)
  A3 window cost W = F + c_row*(T-1), F absorbed; deltas priced at c_row only
  A4 windows stay distinct (no re-ordering/merging; chain_n=0 windows only in this trace)
  A5 rescue-off world for the NOW verdict (STRATA_BATCH_RESCUE=0 standing law); primary
     calibration excludes rescue-advanced windows, secondary includes them

Usage: .venv/bin/python campaign/t31-keeprefit.py [--input campaign/results/spec-trace.txt]
"""
import argparse
import math
import re
import sys

import numpy as np

F32 = np.float32

# ----------------------------------------------------------------------------- parsing
LINE_RE = re.compile(r"^w\s+T=(\d+)\s+a=(\d+)\s+resc=(-?\d+)\s+d=(.*)$")


def parse(path):
    wins, bad = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            s = raw.strip()
            if not s:
                continue
            m = LINE_RE.match(s)
            if not m:
                bad.append((ln, s[:80]))
                continue
            T, a, resc = int(m.group(1)), int(m.group(2)), int(m.group(3))
            dfield = m.group(4).strip()
            d = [F32(float(x)) for x in dfield.split(",") if x != ""] if dfield else []
            wins.append(dict(ln=ln, T=T, a=a, resc=resc, d=d))
    return wins, bad


def segments(wins):
    """Split on cumulative-resc counter resets (append-mode file, one counter per boot)."""
    segs, cur = [], [wins[0]] if wins else []
    for prev, w in zip(wins, wins[1:]):
        if w["resc"] < prev["resc"]:  # counter restarted: new boot segment
            segs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    for s in segs:
        run = 0
        for w in s:
            d = w["resc"] - run
            w["resc_delta"] = max(0, d)
            run = w["resc"]
    return segs


# ------------------------------------------------------------------------ calibration
# Lattice A: T26-2's fixed port bins (edges; keep the lattice, refit the values).
PORT_EDGES = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 1.01]
# Lattice B: equal-count deciles of the pooled tested-d pool, merged with fixed anchors.
ANCHORS = [0.3, 0.5, 0.7]


def decile_anchor_edges(dvals):
    q = np.quantile(dvals, [i / 10 for i in range(1, 10)])
    edges = sorted(set([round(float(x), 3) for x in q] + ANCHORS))
    return edges + [1.01]


def binof(d, edges):
    for i in range(len(edges) - 1):
        if d < edges[i + 1]:
            return i
    return len(edges) - 2


def collect_rows(wins, rescue_mode):
    """Tested rows only: position j is tested iff j <= a (inclusive: the stopping
    rejection); accepted iff j < a. rescue_mode 'exact' keeps windows with zero rescue
    deltas (a is then the pure exact-match count); 'incl' keeps everything (a counts
    rescued positions as accepted - what DraftCumAdapt::observe would eat)."""
    rows = []  # (win_idx, pos, d, acc)
    for wi, w in enumerate(wins):
        if rescue_mode == "exact" and w["resc_delta"] > 0:
            continue
        for j, dv in enumerate(w["d"]):
            if j <= w["a"]:
                rows.append((wi, j, float(dv), 1 if j < w["a"] else 0))
    return rows


def table(rows, edges):
    tab = {}
    for wi, j, d, acc in rows:
        b = binof(d, edges)
        tab.setdefault((j, b), [0, 0, 0.0])
        tab[(j, b)][0] += 1
        tab[(j, b)][1] += acc
    for k in tab:
        n, s, _ = tab[k]
        tab[k][2] = s / n if n else float("nan")
    return tab


def build_shrunk(rows, edges, k=20.0):
    """acc_hat(pos, d-bin) = (x_jb + k*p_b) / (n_jb + k): the raw per-(position x bin)
    rate shrunk toward the bin's pooled rate. Chosen over a parametric fit because the
    additive-logistic misfits the thin low-d cells by up to +/-20pp (see .md); the raw
    bins ARE the refit deliverable (T26-2 c.3 keeps basalt's lattice, refits values)."""
    bn, bx, jn, jx = {}, {}, {}, {}
    for wi, j, d, acc in rows:
        b = binof(d, edges)
        bn[b] = bn.get(b, 0) + 1
        bx[b] = bx.get(b, 0.0) + acc
        jn[(j, b)] = jn.get((j, b), 0) + 1
        jx[(j, b)] = jx.get((j, b), 0.0) + acc
    tab = {}
    for (j, b), n_jb in jn.items():
        p_b = bx[b] / bn[b] if bn.get(b) else 0.0
        tab[(j, b)] = (jx[(j, b)] + k * p_b) / (n_jb + k)
    return tab, bn, bx


def wilson(n, x, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def monotonicity(rows, edges):
    """Per position: pooled-bin acceptance slope vs bin index (and its sign)."""
    out = {}
    by_pos = {}
    for wi, j, d, acc in rows:
        by_pos.setdefault(j, []).append((binof(d, edges), acc))
    for j, v in by_pos.items():
        v.sort()
        xs = np.array([a for a, _ in v], dtype=float)
        ys = np.array([b for _, b in v], dtype=float)
        if len(v) < 2 or xs.std() == 0:
            out[j] = float("nan")
            continue
        out[j] = float(np.corrcoef(xs, ys)[0, 1])  # negative corr = monotone-down bins
    return out


def fit_logit(rows, l2=1e-2, iters=50):
    """logit(m) = s(d) + t(pos): bin-spline in d (linear tail beyond observed range is
    capped) + additive position effect. Used only to extrapolate below 0.5 (censored
    region) and to smooth thin cells; raw binned rates drive everything observable."""
    d = np.array([r[2] for r in rows], dtype=float)
    y = np.array([r[3] for r in rows], dtype=float)
    p = np.array([r[1] for r in rows], dtype=float)
    # basis: [1, d, max(d-0.9,0) (hinge for the >=0.9 mass), pos k=1..4 one-hot]
    X = np.column_stack([
        np.ones_like(d), d, np.maximum(d - 0.9, 0.0),
        *( (p == k).astype(float) for k in (1, 2, 3, 4) ),
    ])
    b = np.zeros(X.shape[1])
    b[0] = math.log(max(y.mean(), 1e-6) / (1 - min(y.mean(), 1 - 1e-6)))
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        g = X.T @ (y - mu) - l2 * b
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def mhat(b, d, pos):
    x = np.array([1.0, d, max(d - 0.9, 0.0)] + [1.0 if pos == k else 0.0 for k in (1, 2, 3, 4)])
    eta = np.clip(x @ b, -30, 30)
    return float(1 / (1 + math.exp(-eta)))


# ------------------------------------------------------------------------ rule sim
def kept_depth(w, thetas, accfn, allow_extend=False):
    """depths offered under the cum rule; returns K = number of drafts offered.
    th vector per depth (index j). Cuts only, unless allow_extend (NOT used on this
    trace: profiles are right-censored at the flat gate - see A2/extension note)."""
    c = F32(1.0)
    K = 0
    n = len(w["d"])
    lim = n if not allow_extend else 5
    for j in range(lim):
        if j >= n:
            break  # no observed d beyond the gate's cut: cannot simulate extension
        c = F32(c * F32(accfn(float(w["d"][j]), j)))
        if c < F32(thetas[min(j, len(thetas) - 1)]):
            break
        K += 1
    return K


def expected_accepted(w, K, accfn):
    # E[accepted | K offered] = sum_j P(all drafts 0..j accepted) - the exact-match walk
    # commits every row until the FIRST rejection, so position j contributes iff the
    # chain survived through j (inclusive).
    s, p = 0.0, 1.0
    for j in range(K):
        p *= accfn(float(w["d"][j]), j)
        s += p
    return s




def main():
    ap = argparse.ArgumentParser(description="T31 draft_cum keep-rule refit (zero-boot)")
    ap.add_argument("--input", default="campaign/results/spec-trace.txt")
    ap.add_argument("--w0-ms", type=float, default=11.51,
                    help="baseline window wall ms for the RESCUE-OFF baseline: "
                         "C0_exact=2.4458 tok/window at banked M1 212.5 tok/s "
                         "(= 4.71 ms/token; T14C standing band 11.4-12.0 ms)")
    ap.add_argument("--crow-ms", type=float, default=2.4,
                    help="marginal verify-row ms, central (post-POOL_DIET 2.2-2.6, T26-2)")
    ap.add_argument("--crow-ms-list", type=float, default=[2.2, 2.4, 2.6, 3.56])
    ap.add_argument("--target-net", type=float, default=2.0, help="retrain go/no-go net %%")
    ap.add_argument("--mc", type=int, default=2000, help="MC reps for the cross-check")
    ap.add_argument("--seed", type=int, default=260)
    args = ap.parse_args()

    wins, bad = parse(args.input)
    if not wins:
        sys.exit("no windows parsed")
    segs = segments(wins)
    n = len(wins)
    offered = np.array([len(w["d"]) for w in wins])
    T = np.array([w["T"] for w in wins])
    a = np.array([w["a"] for w in wins])
    rd = np.array([w["resc_delta"] for w in wins])
    print(f"== T31 keep-rule refit on {args.input}")
    print(f"lines-parsed={n} malformed={len(bad)} segments(boots)={len(segs)}")
    if bad[:5]:
        print("  malformed examples:", bad[:5])

    # ---------- 1. hygiene
    print("\n-- [1] parse hygiene")
    tdist = {t: int((T == t).sum()) for t in np.unique(T)}
    adist = {x: int((a == x).sum()) for x in np.unique(a)}
    print("T dist:", tdist, "sum", int(T.sum()), "mean T", round(float(T.mean()), 4))
    print("a dist:", adist, "mean a", round(float(a.mean()), 4), "mean C=a+1",
          round(float(a.mean() + 1), 4))
    print("offered drafts/window mean", round(float(offered.mean()), 4),
          "total offered rows", int(offered.sum()))
    lenbad = [(w["ln"], w["T"], len(w["d"])) for w in wins if len(w["d"]) != min(w["T"] - 1, 8)]
    print("d-row count != min(T-1,8):", len(lenbad), lenbad[:5])
    undergate = [float(x) for w in wins for x in w["d"] if x < F32(0.5)]
    print(f"printed d < 0.5 (standing gate): {len(undergate)} (expect 0)")
    print("resc field: final counter", wins[-1]["resc"], "| windows with field>0:",
          int((np.array([w["resc"] for w in wins]) > 0).sum()),
          "| windows with delta>0:", int((rd > 0).sum()),
          "| total rescue events:", int(rd.sum()),
          "| max deltas/window:", int(rd.max()))
    for si, s in enumerate(segs):
        st = [w["T"] for w in s]
        sa = [w["a"] for w in s]
        sr = sum(w["resc_delta"] for w in s)
        soff = sum(len(w["d"]) for w in s)
        dmin = min([float(x) for w in s for x in w["d"]], default=float("nan"))
        tmix = ",".join(f"{t}:{st.count(t)}" for t in sorted(set(st)))
        print(f"  seg{si}: n={len(s)} resc={sr} T[{tmix}] meanA={sum(sa)/len(s):.2f} "
              f"acc(off)={sum(sa)/max(1,soff):.3f} minD={dmin:.4f}")
    a_over = [(w["ln"], w["a"], len(w["d"])) for w in wins if w["a"] > len(w["d"])]
    print("a > drafts-offered (impossible):", len(a_over), a_over[:3])
    dge9 = [x for w in wins for x in w["d"] if x >= F32(0.9)]
    print(f"d >= 0.9 share of printed rows: {len(dge9)}/{int(offered.sum())} "
          f"= {len(dge9)/max(1,int(offered.sum()))*100:.1f}%")

    # ---------- 2. calibration (tested rows, exact-match primary)
    print("\n-- [2] per-position acceptance (tested rows = positions 0..a)")
    exact_rows = collect_rows(wins, "exact")
    incl_rows = collect_rows(wins, "incl")
    dpool = np.array([float(x) for w in wins for x in w["d"]])
    edges_b = decile_anchor_edges(dpool[np.linspace(0, len(dpool) - 1, 20000).astype(int)]) \
        if len(dpool) > 20000 else decile_anchor_edges(dpool)
    print("lattice B edges (pooled deciles + anchors):", edges_b[:-1])
    tested_offered = sum(min(w["a"] + 1, len(w["d"])) for w in wins)
    print(f"tested rows: {len(incl_rows)} (incl) / {len(exact_rows)} (exact, resc-free windows); "
          f"printed rows total {int(offered.sum())} - the prior analysis pooled ALL printed rows "
          f"as 'tested' (n=3593), which includes offered-but-never-evaluated rows after the "
          f"stopping rejection")
    bfit = fit_logit(exact_rows if exact_rows else incl_rows)  # kept only as the
    # below-0.5 extrapolator (censored region); PRIMARY calibration is the shrunk table
    acc_tab, bn_pool, bx_pool = build_shrunk(exact_rows if exact_rows else incl_rows, edges_b)
    lo_rate = bx_pool.get(1, 1.0) / bn_pool.get(1, 1.0) if bn_pool.get(1) else 0.5

    def acc_primary(d, k):
        b = binof(d, edges_b)
        v = acc_tab.get((k, b))
        if v is None:   # thin/absent cell: fall back to the bin's pooled rate
            v = bx_pool.get(b, 0.0) / bn_pool[b] if bn_pool.get(b) else lo_rate
        return v

    def acc_shift_table(delta, d, k):
        dd = min(1.0, d + delta)
        b = binof(dd, edges_b)
        if dd < edges_b[1]:          # below the gate's censor: proxy by the lowest live bin
            b = 1
        return acc_tab.get((k, b), lo_rate)

    for name, edges in (("A-port", PORT_EDGES), ("B-decile", edges_b)):
        tab = table(exact_rows, edges)
        print(f"\n  lattice {name} (exact-match, resc-free windows): "
              f"bins {['[' + f'{edges[i]:.2f}' + f',{edges[i+1]:.2f})' for i in range(len(edges)-1)]}")
        hdr = "pos | " + " | ".join(f"b{i}" for i in range(len(edges) - 1)) + " | ALL"
        print("  " + hdr)
        for j in range(0, 5):
            cells, tot_n, tot_x = [], 0, 0
            for bi in range(len(edges) - 1):
                nn, sx, _ = tab.get((j, bi), [0, 0, 0.0])
                tot_n += nn
                tot_x += sx
                if nn == 0:
                    cells.append(".")
                elif nn < 8:
                    cells.append(f"~{sx:.0f}/{nn}")
                else:
                    cells.append(f"{sx:.0f}/{nn}={sx/nn:.3f}")
            lo, hi = wilson(tot_n, tot_x)
            print(f"  {j}  | " + " | ".join(f"{c:>14s}" for c in cells) +
                  f" | {tot_x:.0f}/{tot_n}={tot_x/tot_n:.3f} [{lo:.3f},{hi:.3f}]")
        # monotonicity: acceptance must RISE with d (bin index) within a position
        mono = monotonicity(exact_rows, edges)
        print("  per-position corr(d-bin, accept):",
              {k: round(v, 3) for k, v in sorted(mono.items())}, "(positive = rises with d: GOOD)")
        print("  ('.'= censored bin, no printed d there; '~' = n<8, rate unreliable; "
              "b0 [0.30,0.50) is empty by construction - the flat-0.5 gate censors it)")
    # unconditional per-position rates (offered denominator) - reconciles F-261
    print("\n  per-position UNCONDITIONAL rates (accepted j<a / all offered rows at j) - F-261 form:")
    for j in range(5):
        offj = sum(1 for w in wins if len(w["d"]) > j)
        accj = sum(1 for w in wins if w["a"] > j)
        print(f"   pos{j}: {accj}/{offj} = {accj/offj:.3f}", end="")
    print()
    # depth marginals incl vs exact
    print("  per-position CONDITIONAL marginals (tested rows; what the sequential rule consumes):")
    for j in range(0, 5):
        ei = [(d, y) for wi, jj, d, y in incl_rows if jj == j]
        ee = [(d, y) for wi, jj, d, y in exact_rows if jj == j]
        ni, xi = len(ei), sum(y for _, y in ei)
        ne, xe = len(ee), sum(y for _, y in ee)
        print(f"   pos{j}: incl {xi:.0f}/{ni}={xi/ni:.3f}   exact {xe:.0f}/{ne}={xe/ne:.3f}"
              if ni and ne else f"   pos{j}: (no data)")
    print("  monotonicity across positions at fixed d is checked in the .md table "
          "(within-bin, acceptance falls with depth)")

    # ---------- 3. rescue-inclusive accounting
    print("\n-- [3] rescue-inclusive accounting")
    rescw = rd > 0
    print(f"windows advanced by rescue: {int(rescw.sum())}/{n} = {rescw.mean()*100:.1f}% "
          f"(prior analysis and F-260 counted resc-FIELD>0 = "
          f"{int((np.array([w['resc'] for w in wins])>0).sum())} = "
          f"{(np.array([w['resc'] for w in wins])>0).mean()*100:.1f}% - that is the cumulative "
          f"counter, not per-window events)")
    print(f"total rescue events {int(rd.sum())}; windows with 2 rescues: {int((rd==2).sum())}")
    print(f"mean a: all={a.mean():.4f}  resc-free={a[~rescw].mean():.4f}  "
          f"exact-match (a-rescue)={float((a-rd).mean()):.4f}")
    print(f"mean C=a+1: all={a.mean()+1:.4f}  exact={(a-rd).mean()+1:.4f}")
    acc_offered_all = a.sum() / offered.sum()
    tested_all = np.array([min(w['a'] + 1, len(w['d'])) for w in wins])
    acc_tested_all = a.sum() / tested_all.sum()
    acc_tested_exact = (a[~rescw].sum()) / tested_all[~rescw].sum()
    print(f"acceptance rates: offered-denom {acc_offered_all:.4f} | tested-denom incl "
          f"{acc_tested_all:.4f} | tested-denom exact {acc_tested_exact:.4f}")
    md_all = np.array([np.mean([float(x) for x in w["d"]]) if w["d"] else np.nan for w in wins])
    print(f"mean d/window (windows with drafts): resc {np.nanmean(md_all[rescw]):.4f} "
          f"vs non-resc {np.nanmean(md_all[~rescw]):.4f}")
    print(f"accept-all windows (a==T-1, T>=2): {int(((a == T - 1) & (T >= 2)).sum())}/{int((T>=2).sum())} "
          f"= {((a == T - 1) & (T >= 2)).sum()/max(1,(T>=2).sum())*100:.1f}% of drafting windows "
          f"({int(((a == T - 1) & (T >= 2)).sum())/n*100:.1f}% of all); cap-hit T=6&a=5: "
          f"{int(((T==6)&(a==5)).sum())} = {((T==6)&(a==5)).mean()/ (T==6).mean() *100 if (T==6).any() else 0:.1f}% of T=6")

    # ---------- 4+5. rule grid -> NOW verdict
    print("\n-- [4] simulated keep rules vs the standing flat-0.5 gate (plug-in, rescue-off world)")
    C0 = float((a - rd).mean() + 1)  # exact-match committed/window (rescue off going forward)
    off0 = float(offered.mean())
    W0 = args.w0_ms
    print(f"anchor: C0={C0:.4f} tok/window (exact-match), offered0={off0:.4f}, "
          f"W0={W0:.2f} ms, one token worth {W0/C0:.2f} ms; crossing rule: a marginal row pays "
          f"iff m > c_row*C0/W0")
    for cr in args.crow_ms_list:
        print(f"   c_row={cr:.2f} ms -> break-even m = {cr*C0/W0:.3f}")
    print(f"depth marginals vs break-even (exact, tested rows): "
          + ", ".join(f"m{j}={xe/ne:.3f}" for j in range(5)
                      for ee in [ [y for wi, jj, d, y in exact_rows if jj == j] ]
                      for ne in [len(ee)] for xe in [sum(ee)] if ne))

    def run_rule(thetas, accfn, label):
        dA, dT = 0.0, 0.0
        for w in wins:
            if len(w["d"]) == 0:
                continue
            K = kept_depth(w, thetas, accfn)
            if K == len(w["d"]):
                continue
            base = expected_accepted(w, len(w["d"]), accfn)
            new = expected_accepted(w, K, accfn)
            dA += (new - base) / n
            dT += (K - len(w["d"])) / n
        out = []
        for cr in args.crow_ms_list:
            Wn = W0 - cr * (-dT)
            net = ((C0 + dA) / Wn) / (C0 / W0) - 1
            out.append((cr, net * 100))
        best = max(out, key=lambda t: t[1])
        print(f"  {label}: dAccepted/w={dA:+.4f} dOffered/w={dT:+.4f} | net% " +
              " ".join(f"c={cr:.2f}:{v:+.2f}%" for cr, v in out))
        return dA, dT, out

    grid_flat = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
    results = {}
    for th in grid_flat:
        results[f"cum-cal th={th:.2f}"] = run_rule([th] * 5, acc_primary, f"cum-calibrated flat th={th:.2f}")
    for th0 in (0.2, 0.3, 0.4):
        for step in (0.05, 0.10):
            thv = [th0 + step * j for j in range(5)]
            results[f"cum-cal decay {thv[0]:.2f}+{step:.2f}k"] = run_rule(
                thv, acc_primary, f"cum-calibrated depth-decay th={['%.2f' % t for t in thv]}")
    # raw-d product (the prior 16-line analysis's rule): acc_hat = d itself
    for th in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70):
        results[f"cum-raw th={th:.2f}"] = run_rule([th] * 5, lambda d, k: d, f"cum-raw-d product th={th:.2f}")
    # per-row flat retune (F-228's closed stair): offer j iff d_j >= th
    def run_flat_retune(th):
        dA, dT = 0.0, 0.0
        for w in wins:
            K = 0
            for j, dv in enumerate(w["d"]):
                if float(dv) >= th:
                    K += 1
                else:
                    break
            if K == len(w["d"]):
                continue
            base = expected_accepted(w, len(w["d"]), acc_primary)
            new = expected_accepted(w, K, acc_primary)
            dA += (new - base) / n
            dT += (K - len(w["d"])) / n
        out = []
        for cr in args.crow_ms_list:
            Wn = W0 - cr * (-dT)
            net = ((C0 + dA) / Wn) / (C0 / W0) - 1
            out.append((cr, net * 100))
        print(f"  flat-retune minp={th:.2f}: dAccepted/w={dA:+.4f} dOffered/w={dT:+.4f} | net% " +
              " ".join(f"c={cr:.2f}:{v:+.2f}%" for cr, v in out))
        return dA, dT, out
    for th in (0.6, 0.7, 0.8, 0.9):
        results[f"flat-retune {th:.2f}"] = run_flat_retune(th)

    central = args.crow_ms
    best_label = max(results, key=lambda k: next(v for cr, v in results[k][2] if cr == central))
    best_net = next(v for cr, v in results[best_label][2] if cr == central)
    print(f"\n  BEST RULE NOW at central c_row={central:.2f} ms: {best_label} -> "
          f"{best_net:+.2f}% net")
    # calibration sanity: does the model reproduce the observed exact-match baseline?
    Cmodel = 1.0 + sum(expected_accepted(w, len(w["d"]), acc_primary) for w in wins
                       if len(w["d"]) > 0) / n
    print(f"  calibration sanity: model baseline C={Cmodel:.4f} vs observed exact-match "
          f"C0={C0:.4f} (rescue-free windows only contribute observed a; delta "
          f"{Cmodel - C0:+.4f})")
    # per-segment heterogeneity of the verdict for th*=0.5
    print("  per-segment verdict for cum-cal th=0.50 (dA, dT, net% at central c_row):")
    for si, sset in enumerate(segs):
        dA = dT = 0.0
        for w in sset:
            if len(w["d"]) == 0:
                continue
            K = kept_depth(w, [0.5] * 5, acc_primary)
            if K == len(w["d"]):
                continue
            dA += (expected_accepted(w, K, acc_primary) -
                   expected_accepted(w, len(w["d"]), acc_primary)) / len(sset)
            dT += (K - len(w["d"])) / len(sset)
        Wn = args.w0_ms - central * (-dT)
        net = ((C0 + dA) / Wn) / (C0 / args.w0_ms) - 1
        print(f"   seg{si}: n={len(sset)} dA={dA:+.4f} dT={dT:+.4f} net={net*100:+.2f}%")
    # decomposition of the th=0.5 delta: exactly WHICH rows get cut and what they carry
    print("  th=0.50 cut decomposition (exact-match world; cut tail attributed to the "
          "first cut position's d-bin):")
    print("   d-bin | rows-cut/window | tokens-carried/window | shrunk acc used | raw cell n")
    cutn, cuttok = {}, {}
    for w in wins:
        if len(w["d"]) == 0:
            continue
        K = kept_depth(w, [0.5] * 5, acc_primary)
        if K >= len(w["d"]):
            continue
        m = [acc_primary(float(x), j) for j, x in enumerate(w["d"])]
        base = expected_accepted(w, len(w["d"]), acc_primary)
        new = expected_accepted(w, K, acc_primary)
        b0 = binof(float(w["d"][K]), edges_b)
        cutn[b0] = cutn.get(b0, 0.0) + (len(w["d"]) - K) / n
        cuttok[b0] = cuttok.get(b0, 0.0) + (base - new) / n
    for b in sorted(set(cutn) | set(cuttok)):
        lo_b, hi_b = edges_b[b], edges_b[b + 1]
        cell = [(j, d, y) for wi, j, d, y in (exact_rows or incl_rows)
                if lo_b <= d < hi_b]
        ncell = len(cell)
        rate_used = acc_tab.get((0, b), float("nan"))
        print(f"   [{lo_b:.2f},{hi_b:.2f}) | {cutn.get(b,0):.4f} | {cuttok.get(b,0):+.4f} "
              f"| {rate_used:.3f} | {ncell}")

    # conservative bound: POOLED-bin rates only (no position effect) - if the verdict
    # survives this, it does not lean on the thin per-position cells
    pooled = {b: (bx_pool.get(b, 0.0) / bn_pool[b]) if bn_pool.get(b) else lo_rate
              for b in list(bn_pool) + [1]}
    acc_pooled = lambda d, k: pooled.get(binof(d, edges_b), lo_rate)
    dA = dT = 0.0
    for w in wins:
        if len(w["d"]) == 0:
            continue
        K = kept_depth(w, [0.5] * 5, acc_pooled)
        if K == len(w["d"]):
            continue
        dA += (expected_accepted(w, K, acc_pooled) -
               expected_accepted(w, len(w["d"]), acc_pooled)) / n
        dT += (K - len(w["d"])) / n
    Wn = args.w0_ms - central * (-dT)
    print(f"  conservative bound (pooled bins, no position effect), th=0.50: "
          f"dA={dA:+.4f} dT={dT:+.4f} net={((C0 + dA) / Wn) / (C0 / args.w0_ms) - 1:+.2f}% "
          f"at central c_row")

    # calibration-uncertainty bootstrap: resample WINDOWS with replacement, rebuild the
    # shrunk table, recompute the th=0.50 delta (cost anchor fixed -> acceptance-side band)
    print("  bootstrap band for cum-cal th=0.50 net% (central c_row, 200 window resamples):")
    rb = np.random.default_rng(args.seed + 2)
    boots = []
    thv = [0.5] * 5
    for rep in range(200):
        idx = rb.integers(0, n, n)
        wsr = [wins[i] for i in idx]
        rr = collect_rows(wsr, "exact") or collect_rows(wsr, "incl")
        tab_r, _, _ = build_shrunk(rr, edges_b)
        lo_r = (bx_pool.get(1, 1.0) / bn_pool.get(1, 1.0)) if bn_pool.get(1) else 0.5
        fn = lambda d, k: tab_r.get((k, binof(d, edges_b)), lo_r)
        dA = dT = 0.0
        for w in wsr:
            if len(w["d"]) == 0:
                continue
            K = kept_depth(w, thv, fn)
            if K == len(w["d"]):
                continue
            dA += (expected_accepted(w, K, fn) -
                   expected_accepted(w, len(w["d"]), fn)) / n
            dT += (K - len(w["d"])) / n
        Wn = args.w0_ms - central * (-dT)
        boots.append(((C0 + dA) / Wn) / (C0 / args.w0_ms) - 1)
    qb = np.percentile(boots, [5, 25, 50, 75, 95])
    print("   p5={:+.2f}% p25={:+.2f}% p50={:+.2f}% p75={:+.2f}% p95={:+.2f}%  "
          "P(net>0)={:.2f}  P(net>=+2%)={:.2f}".format(
              qb[0]*100, qb[1]*100, qb[2]*100, qb[3]*100, qb[4]*100,
              float(np.mean(np.array(boots) > 0)), float(np.mean(np.array(boots) >= 0.02))))

    # ---------- 6. post-retrain projection
    print("\n-- [5] post-retrain projection (shift d -> min(1, d+delta); acceptance re-read "
          "from the CURRENT calibration map; dark T=1 windows get one imputed draft)")
    rng = np.random.default_rng(args.seed)
    dark = int(sum(1 for w in wins if not w["d"]))
    print(f"dark windows (T=1, d unobserved, gate-closed at depth 0): {dark} = {dark/n*100:.1f}% "
          f"- their d0 is imputed uniform[0.2,0.5); rows beyond the observed profile are "
          f"RIGHT-CENSORED (the flat gate never printed them), so every K is capped at the "
          f"observed profile: extension upside is NOT priced (it belongs to the drafter, not "
          f"the keep rule; at delta>0 the censored mass grows - reported below)")
    wins_sim = []
    for w in wins:
        d = [float(x) for x in w["d"]]
        if not d:
            d = [rng.uniform(0.2, 0.4999)]
        wins_sim.append(dict(w, dd=d))

    acc_shift = acc_shift_table  # shrunk per-(pos,bin) table on the SHIFTED d
    # (d+delta lands inside the observed support for delta >= 0; below-gate proxies to
    # the lowest live bin - used only by the imputed dark windows at delta ~ 0)

    def sim(delta, thetas, impute_hi_check=True):
        """Returns (dA_rule_vs_gate, dT_rule_vs_gate, dA_shift_only, dT_shift_only,
        censored_extra). Baseline = flat-0.5 gate ON THE SHIFTED d (same-shift gate);
        rule = cum-product on the shifted d, capped at the observed profile.
        shift-only = gate-at-delta vs the MEASURED today baseline (delta=0 gate)."""
        rg = rw = sg = st_ = cens = 0.0
        for w in wins_sim:
            ds = [min(1.0, x + delta) for x in w["dd"]]
            cap = len(w["dd"])
            Kg = 0
            for dv in ds:
                if dv >= 0.5:
                    Kg += 1
                else:
                    break
            Kr = 0
            c = F32(1.0)
            for j, dv in enumerate(ds):
                c = F32(c * F32(acc_shift(delta, dv, j)))
                if c < F32(thetas[min(j, len(thetas) - 1)]):
                    break
                Kr += 1
            Kg_c, Kr_c = min(Kg, cap), min(Kr, cap)
            cens += max(0, Kg - cap) / n
            if Kr_c == Kg_c:
                continue
            accfn = lambda d, k: acc_shift(delta, d, k)
            rg += (expected_accepted({"d": [F32(x) for x in ds]}, Kr_c, accfn) -
                   expected_accepted({"d": [F32(x) for x in ds]}, Kg_c, accfn)) / n
            rw += (Kr_c - Kg_c) / n
        # shift-only vs measured today (gate at delta=0 keeps exactly the observed profile
        # for non-dark windows, 0 for dark ones)
        for w in wins_sim:
            ds = [min(1.0, x + delta) for x in w["dd"]]
            cap = len(w["dd"])
            Kg0 = cap if w["d"] else 0   # today's gate behavior (observed profile / dark)
            Kg = 0
            for dv in ds:
                if dv >= 0.5:
                    Kg += 1
                else:
                    break
            Kg = min(Kg, cap)
            if Kg == Kg0:
                continue
            accfn = lambda d, k: acc_shift(delta, d, k)
            sg += (expected_accepted({"d": [F32(x) for x in ds]}, Kg, accfn) -
                   expected_accepted({"d": [F32(x) for x in ds]}, Kg0, accfn)) / n
            st_ += (Kg - Kg0) / n
        return rg, rw, sg, st_, cens

    deltas = [round(0.02 * i, 2) for i in range(0, 26)]
    thgrid = [[t] * 5 for t in (0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)] + \
             [[0.1 + 0.1 * j for j in range(5)], [0.2 + 0.1 * j for j in range(5)],
              [0.3 + 0.1 * j for j in range(5)]]
    print(f"\ndelta | best rule net% vs TODAY (shift+rule) | best rule net% vs GATE-at-same-shift "
          f"(the lever's own value) | censored gate-rows/w at delta")
    thr_primary, thr_ctx = None, None
    for delta in deltas:
        b1 = b2 = -9.0
        l1 = l2_ = ""
        cens = 0.0
        for thv in thgrid:
            rg, rw, sg, st_, cens = sim(delta, thv)
            net_ctx = ((C0 + sg + rg) / (W0 - args.crow_ms * (-(st_ + rw)))) / (C0 / W0) - 1
            net_inc = ((C0 + sg + rg) / (W0 - args.crow_ms * (-(st_ + rw)))) / \
                      ((C0 + sg) / (W0 - args.crow_ms * (-st_))) - 1
            if net_ctx > b1:
                b1, l1 = net_ctx, str(thv[0])
            if net_inc > b2:
                b2, l2_ = net_inc, str(thv[0])
        if thr_ctx is None and b1 * 100 >= args.target_net:
            thr_ctx = delta
        if thr_primary is None and b2 * 100 >= args.target_net:
            thr_primary = delta
        m1 = " <== >=+2%" if delta == thr_ctx else ""
        m2 = " <== rule>=+2%" if delta == thr_primary else ""
        print(f"{delta:.2f} | {b1*100:+7.2f}% (th {l1}){m1} | {b2*100:+7.2f}% (th {l2_}){m2} | {cens:.3f}")
    print(f"\nRETRAIN GO/NO-GO for the keep-rule lever: best rule clears +{args.target_net:.0f}% "
          f"net vs the SAME-SHIFT gate at shift delta* = {thr_primary} "
          f"(clears +{args.target_net:.0f}% vs TODAY's gate at delta = {thr_ctx})")

    # ---------- 7. MC cross-check
    print("\n-- [6] Monte-Carlo cross-check of the plug-in (central c_row, best NOW rule)")
    thetas = [0.35] * 5
    lab = "cum-cal th=0.35"
    if best_label.startswith("cum-cal") and "th=" in best_label:
        try:
            tv = float(best_label.split("th=")[1])
            thetas, lab = [tv] * 5, best_label
        except Exception:
            pass
    r2 = np.random.default_rng(args.seed + 1)
    reps = args.mc
    plug_sum = mc_sum = 0.0
    n_aff = 0
    for w in wins:
        if len(w["d"]) == 0:
            continue
        K = kept_depth(w, thetas, acc_primary)
        if K == len(w["d"]):
            continue
        n_aff += 1
        plug_sum += expected_accepted(w, K, acc_primary) - expected_accepted(w, len(w["d"]), acc_primary)
        mb = np.array([acc_primary(float(x), j) for j, x in enumerate(w["d"])][:len(w["d"])])
        mr = mb[:K]
        ub = r2.random((reps, len(mb))) < mb
        ur = r2.random((reps, K)) < mr
        ab = ub.cumprod(axis=1).sum(axis=1)
        ar = ur.cumprod(axis=1).sum(axis=1)
        mc_sum += float((ar - ab).mean())
    print(f"{lab}: plug-in dA/w={plug_sum/n:+.5f}  MC dA/w={mc_sum/n:+.5f} "
          f"({n_aff} affected windows x {reps} reps)")


if __name__ == "__main__":
    main()
