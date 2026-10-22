#!/usr/bin/env python3
"""The KL yardstick (SVG-CUSTOM-TODO.md REVISED ORDER item 2): score a numerics-class candidate against a
reference by the divergence of their final-layer logits, BEFORE any GPU battery is spent on it.

The engine runs coupled MTP speculative decoding with EXACT-MATCH verification, so a small numerics change to
the logits measurably flips token acceptance: F-230 (SVG-CUSTOM.md:7884) paid -2.7% relative acceptance = -2.8%
M1 for a Q8-activation prefill change whose KL was ~1.8e-4 - KL at the 1e-4 scale is ALREADY acceptance-relevant,
and the disputed fp16-accum arms (R3's +10-17% perf vs R5's KL blowout) are decided by exactly this instrument.
The coarser pack-scale gates (top-1 >= 97%, mean KL <= 0.05) live in T26-5 section b / quant_quality.py; this tool
works one scale down, on raw logits captures, where numerics-class arms land.

    .venv/bin/python tools/kl_yardstick.py --ref REF.npy --cand CAND.npy            the human table
    .venv/bin/python tools/kl_yardstick.py --ref REF.npy --cand CAND.jsonl --json   machine-readable
    .venv/bin/python tools/kl_yardstick.py --selftest                              analytic fixtures

Inputs are .npy float arrays [N, V] (or one vocab row) or .jsonl files of {"logits": [...]} records, N rows of
vocab-sized logits.  The two sides are trimmed to the shorter N, rows with any non-finite logit are rejected
(counted, both sides), and everything is computed in float64 through a stable log-softmax (subtract logsumexp).
Metrics: per-row KL(ref||cand), KL(cand||ref) and the symmetric JS divergence (nats, bounded by ln 2), each as
mean / median / p95 / max over rows; a top-32-restricted KL over the union of each row's top-32 by ref and by
cand, both renormalized over that union (the acceptance-relevant head of the distribution, where exact-match
argmax decisions flip); the argmax agreement rate (the direct exact-match coupling signal); and mean / max
|logit diff| for sanity.

THE VERDICT BANDS ARE PROVISIONAL, anchored to the F-230 precedent and to be recalibrated by a ctl-vs-ctl
self-noise pair plus the first known-KL arm:
    exact            mean KL <  1e-7    numerically clean (GDN_CHUNKED's accepted ~3e-7 shift sits just above)
    benign           mean KL <  5e-5    ~3.6x under F-230; acceptance movement expected inside boot noise
    borderline       mean KL <  1e-3    F-230's 1.8e-4 sits HERE and cost -2.7% acceptance: the coupled
                                        acceptance battery (M3b + acceptance A/B) is mandatory before arming
    acceptance-risk  mean KL >= 1e-3    ~5.5x F-230; reject on numerics, no battery warranted
Both the full-KL and the top-32 verdicts are printed; a disagreement is flagged with the worse class as the gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TOPK = 32                      # the restricted-KL support size (the argmax neighborhood exact-match flips live in)
LN2 = math.log(2.0)
CHUNK_ROWS = 256               # rows per block, so a real capture's [N, V] never triples itself in temporaries
REL_TOL = 1e-10                # selftest: analytic fixtures must match to this relative tolerance
ULP_TOL = 1e-12                # selftest: order-of-summation effects only (permutation / V<=32 reduction checks)
# (upper bound, class) pairs; at or above the last bound the class is acceptance-risk
BANDS = ((1e-7, "exact"), (5e-5, "benign"), (1e-3, "borderline"))
RISK = "acceptance-risk"
CLASS_ORDER = ("exact", "benign", "borderline", RISK)
F230_REMINDER = ("F-230 anchor: a 1.8e-4 KL sat in this band and cost -2.7% relative acceptance = -2.8% M1 -\n"
                 "the coupled acceptance battery (M3b + acceptance A/B) is mandatory before arming this arm.")


# ------------------------------------------------------------------------------------------------ numerics
def log_softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax of a [n, V] float64 matrix (subtract the row logsumexp; never exponentiates an offset)."""
    m = x.max(axis=1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))


def _log_softmax_row(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - (m + math.log(float(np.exp(x - m).sum())))


def topk_kl_rows(a: np.ndarray, b: np.ndarray, k: int = TOPK) -> np.ndarray:
    """Per-row KL(ref||cand) restricted to the union of the row's top-k by ref and by cand.  The renormalization
    over the union is done in log space (the restricted log-softmax IS the renormalization), so extreme logits
    never underflow to a division by zero; with V <= k the union is the whole row and this reduces to the full KL."""
    n, v = a.shape
    kk = min(k, v)
    first = v - kk
    out = np.empty(n)
    for i in range(n):
        top_a = np.argpartition(a[i], first)[first:]      # k largest by ref (unordered; ties split deterministically)
        top_b = np.argpartition(b[i], first)[first:]      # k largest by cand
        u = np.union1d(top_a, top_b)                      # up to 2k columns, sorted
        lu = _log_softmax_row(a[i][u])
        lv = _log_softmax_row(b[i][u])
        out[i] = float((np.exp(lu) * (lu - lv)).sum())
    return out


def _stats(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "max": float(x.max())}


def evaluate(ref: np.ndarray, cand: np.ndarray, topk: int = TOPK) -> tuple[dict, dict]:
    """All metrics for one ref/cand pair (both [N, V] float64, finite-checked here) plus the verdict."""
    if ref.ndim != 2 or cand.ndim != 2:
        raise ValueError("logit matrices must be 2-D [N, V]")
    if ref.shape[1] != cand.shape[1]:
        raise ValueError(f"vocab mismatch: ref has V={ref.shape[1]}, cand has V={cand.shape[1]}")
    n = min(ref.shape[0], cand.shape[0])
    fin_r = np.isfinite(ref[:n]).all(axis=1)
    fin_c = np.isfinite(cand[:n]).all(axis=1)
    keep = fin_r & fin_c
    a = ref[:n][keep]
    b = cand[:n][keep]
    rows = {"ref": int(ref.shape[0]), "cand": int(cand.shape[0]), "trimmed_to": int(n),
            "nonfinite_ref": int((~fin_r).sum()), "nonfinite_cand": int((~fin_c).sum()),
            "rejected_union": int((~keep).sum()), "used": int(keep.sum())}
    if rows["used"] == 0:
        raise ValueError(f"no usable rows: {rows}")
    m = rows["used"]
    kl_rc = np.empty(m)            # KL(ref||cand)
    kl_cr = np.empty(m)            # KL(cand||ref)
    js = np.empty(m)               # symmetric JS, nats
    klt = np.empty(m)              # top-k restricted KL(ref||cand)
    agree = 0
    absd_sum = 0.0
    absd_n = 0
    absd_max = 0.0
    for s in range(0, m, CHUNK_ROWS):
        e = min(s + CHUNK_ROWS, m)
        x, y = a[s:e], b[s:e]
        lp, lq = log_softmax(x), log_softmax(y)
        kl_rc[s:e] = (np.exp(lp) * (lp - lq)).sum(axis=1)
        kl_cr[s:e] = (np.exp(lq) * (lq - lp)).sum(axis=1)
        lm = np.logaddexp(lp, lq) - LN2                # log of the mixture 0.5 p + 0.5 q, stably
        js[s:e] = 0.5 * (np.exp(lp) * (lp - lm)).sum(axis=1) + 0.5 * (np.exp(lq) * (lq - lm)).sum(axis=1)
        klt[s:e] = topk_kl_rows(x, y, topk)
        agree += int((x.argmax(axis=1) == y.argmax(axis=1)).sum())
        d = np.abs(x - y)
        absd_sum += float(d.sum())
        absd_n += int(d.size)
        absd_max = max(absd_max, float(d.max()))
    metrics = {"rows": rows, "vocab": int(a.shape[1]),
               "kl_ref_cand": _stats(kl_rc), "kl_cand_ref": _stats(kl_cr), "js_symmetric": _stats(js),
               "kl_ref_cand_top%d" % topk: _stats(klt),
               "argmax_agreement": agree / m, "argmax_agree_rows": agree,
               "logit_absdiff": {"mean": absd_sum / absd_n, "max": absd_max}}
    return metrics, _verdict(metrics, topk)


def classify(v: float) -> tuple[str, str]:
    """(class, band description) for one mean-KL value; thresholds PROVISIONAL, F-230-anchored."""
    lo = 0.0
    for bound, name in BANDS:
        if v < bound:
            return name, ("%.0e" % lo if lo else "0") + " <= x < %.0e" % bound
        lo = bound
    return RISK, "x >= %.0e" % BANDS[-1][0]


def _verdict(metrics: dict, topk: int) -> dict:
    full = metrics["kl_ref_cand"]["mean"]
    top = metrics["kl_ref_cand_top%d" % topk]["mean"]
    c_full, band_full = classify(full)
    c_top, band_top = classify(top)
    return {"thresholds": {"exact": 1e-7, "benign": 5e-5, "borderline": 1e-3, "provisional": True,
                           "anchor": "F-230: KL 1.8e-4 -> -2.7% acceptance = -2.8% M1"},
            "full_kl_mean": full, "full_kl_class": c_full, "full_kl_band": band_full,
            "top%d_kl_mean" % topk: top, "top%d_kl_class" % topk: c_top, "top%d_kl_band" % topk: band_top,
            "classes_agree": c_full == c_top,
            "gate_class": c_full if CLASS_ORDER.index(c_full) >= CLASS_ORDER.index(c_top) else c_top}


# ------------------------------------------------------------------------------------------------ input
NUMPY_MAGIC = b"\x93NUMPY"


def load_matrix(path: str) -> np.ndarray:
    """A .npy (by magic bytes, whatever the suffix) or a .jsonl of {"logits": [...]} records -> [N, V] float64.
    A bare 1-D .npy is one vocab row.  Non-finite values are kept here; evaluate() rejects and counts the rows."""
    p = Path(path)
    with open(p, "rb") as f:
        head = f.read(len(NUMPY_MAGIC))
    if head == NUMPY_MAGIC:
        arr = np.load(p, allow_pickle=False)
        if np.iscomplexobj(arr):
            raise ValueError(f"{path}: complex logits are not a distribution")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected a [N, V] array, got shape {arr.shape}")
        return arr.astype(np.float64, copy=False)
    rows = []
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)      # accepts NaN / Infinity literals; evaluate() rejects the row
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({e})")
            if not isinstance(rec, dict) or not isinstance(rec.get("logits"), list):
                raise ValueError(f"{path}:{lineno}: expected an object with a 'logits' list")
            rows.append(rec["logits"])
    if not rows:
        raise ValueError(f"{path}: no records")
    try:
        arr = np.array(rows, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: logits rows are not equal-length numbers ({e})")
    if arr.ndim != 2:
        raise ValueError(f"{path}: logits rows are not equal-length (shape {arr.shape})")
    return arr


# ------------------------------------------------------------------------------------------------ output
def print_human(ref_path: str, cand_path: str, metrics: dict, verdict: dict, topk: int) -> None:
    r = metrics["rows"]
    print(f"KL yardstick   ref={ref_path}   cand={cand_path}")
    trim = "" if r["ref"] == r["cand"] else f", trimmed to {r['trimmed_to']}"
    print(f"rows: used {r['used']} of ref {r['ref']} / cand {r['cand']}{trim}; "
          f"rejected {r['rejected_union']} nonfinite rows (ref {r['nonfinite_ref']}, cand {r['nonfinite_cand']}); "
          f"vocab V={metrics['vocab']}")
    topkey = "kl_ref_cand_top%d" % topk
    print(f"{'':26s}{'mean':>12s}{'median':>12s}{'p95':>12s}{'max':>12s}")
    for label, key in (("KL(ref||cand)", "kl_ref_cand"), ("KL(cand||ref)", "kl_cand_ref"),
                       ("JS symmetric (nats)", "js_symmetric"), (f"KL(ref||cand) top-{topk}", topkey)):
        s = metrics[key]
        print(f"{label:26s}{s['mean']:12.3e}{s['median']:12.3e}{s['p95']:12.3e}{s['max']:12.3e}")
    a = metrics["logit_absdiff"]
    print(f"argmax agreement         {metrics['argmax_agreement']:.4f}   "
          f"({metrics['argmax_agree_rows']}/{r['used']} rows)")
    print(f"|logit diff|             mean {a['mean']:.3e}   max {a['max']:.3e}")
    print(f"verdict  full-KL mean    {verdict['full_kl_mean']:.3e}   {verdict['full_kl_class'].upper():16s}"
          f"{verdict['full_kl_band']:20s}[PROVISIONAL, F-230-anchored]")
    print(f"         top-{topk}-KL mean  {verdict['top%d_kl_mean' % topk]:.3e}   "
          f"{verdict['top%d_kl_class' % topk].upper():16s}{verdict['top%d_kl_band' % topk]:20s}[PROVISIONAL, F-230-anchored]")
    if not verdict["classes_agree"]:
        print(f"         FLAG: verdicts disagree (full={verdict['full_kl_class']}, top-{topk}="
              f"{verdict['top%d_kl_class' % topk]}) - gate on the worse class ({verdict['gate_class']}) or investigate")
    if "borderline" in (verdict["full_kl_class"], verdict["top%d_kl_class" % topk]):
        print(F230_REMINDER)


def cmd_compare(a: argparse.Namespace) -> int:
    ref = load_matrix(a.ref)
    cand = load_matrix(a.cand)
    metrics, verdict = evaluate(ref, cand)
    if a.json:
        print(json.dumps({"metrics": metrics, "verdict": verdict}, indent=1))
    else:
        print_human(a.ref, a.cand, metrics, verdict, TOPK)
    return 0


# ------------------------------------------------------------------------------------------------ selftest
def cmd_selftest() -> int:
    """Fixtures with analytically known answers, asserted (not eyeballed); every failure is printed, exit 1."""
    failures = []

    def close(name: str, got: float, want: float, tol: float = REL_TOL) -> None:
        rel = abs(got - want) / max(abs(want), 1e-300)
        if rel > tol:
            failures.append(f"{name}: got {got!r}, want {want!r} (rel {rel:.1e} > {tol:.0e})")

    def exact_zero(name: str, got: float) -> None:
        if got != 0.0:
            failures.append(f"{name}: got {got!r}, want exactly 0.0")

    def ok(name: str, cond: bool, detail: str = "") -> None:
        if not cond:
            failures.append(f"{name}: {detail}")

    def run(ref, cand):
        return evaluate(np.asarray(ref, dtype=np.float64), np.asarray(cand, dtype=np.float64))

    # (a) closed-form 3-symbol pair, hand formula written out here; the +1000 offset proves the log-softmax
    #     subtracts the logsumexp (a naive exp/sum would overflow or lose the answer)
    p, q = (0.5, 0.25, 0.25), (0.25, 0.5, 0.25)
    big = 1000.0
    m, _ = run([[math.log(x) + big for x in p]], [[math.log(x) + big for x in q]])
    want_kl = 0.5 * math.log(0.5 / 0.25) + 0.25 * math.log(0.25 / 0.5) + 0.25 * math.log(0.25 / 0.25)  # = ln2/4
    want_js = 0.5 * math.log(0.5 / 0.375) + 0.25 * math.log(0.25 / 0.375) + 0.25 * math.log(0.25 / 0.25)
    close("(a) KL(P||Q) closed form", m["kl_ref_cand"]["mean"], want_kl)
    close("(a) KL(P||Q) = ln2/4", m["kl_ref_cand"]["mean"], math.log(2.0) / 4.0)
    close("(a) KL(Q||P) closed form", m["kl_cand_ref"]["mean"], want_kl)      # P,Q are column swaps: same KL
    close("(a) JS closed form (M = .375,.375,.25)", m["js_symmetric"]["mean"], want_js)
    close("(a) top-32 KL == full KL at V=3", m["kl_ref_cand_top32"]["mean"], want_kl)

    # (b) identity -> exactly 0.0 (float64 KL, both full and top-32; JS is only ulp-level: logaddexp rounds)
    rng = np.random.default_rng(20261022)
    ident = rng.normal(size=(4, 64))
    m, _ = run(ident, ident.copy())
    exact_zero("(b) KL(P||P) full", m["kl_ref_cand"]["mean"])
    exact_zero("(b) KL(P||P) full, max row", m["kl_ref_cand"]["max"])
    exact_zero("(b) KL(P||P) reverse", m["kl_cand_ref"]["mean"])
    exact_zero("(b) KL(P||P) top-32", m["kl_ref_cand_top32"]["mean"])
    ok("(b) JS(P,P) ~ 0", abs(m["js_symmetric"]["mean"]) <= ULP_TOL, f"got {m['js_symmetric']['mean']!r}")
    ok("(b) argmax agreement == 1", m["argmax_agreement"] == 1.0)
    exact_zero("(b) max |logit diff|", m["logit_absdiff"]["max"])

    # (c) two discretized normals vs a brute-force sum computed here with plain math.exp/log - independent of the
    #     tool's log-softmax/logsumexp path
    xs = [(-8.0 + 0.04 * i) for i in range(401)]
    pr = [math.exp(-(x - 0.0) ** 2 / (2 * 1.0 ** 2)) for x in xs]
    qr = [math.exp(-(x - 0.5) ** 2 / (2 * 1.2 ** 2)) for x in xs]
    zr, zq = sum(pr), sum(qr)
    pr = [v / zr for v in pr]
    qr = [v / zq for v in qr]
    want = sum(v * math.log(v / w) for v, w in zip(pr, qr) if v > 0)
    m, _ = run([[math.log(v) for v in pr]], [[math.log(v) for v in qr]])
    close("(c) discretized normals KL", m["kl_ref_cand"]["mean"], want)
    ok("(c) argmax agreement == 0 (the two normals peak at different x)",
       m["argmax_agreement"] == 0.0, f"got {m['argmax_agreement']!r}")

    # (d) permutation invariance: JS is symmetric by construction and nothing depends on vocabulary order
    a52 = rng.normal(size=(5, 50))
    b52 = rng.normal(size=(5, 50)) + 0.001
    m1, _ = run(a52, b52)
    m2, _ = run(b52, a52)                      # swap ref/cand: JS must not move
    close("(d) JS(P,Q) == JS(Q,P)", m2["js_symmetric"]["mean"], m1["js_symmetric"]["mean"], ULP_TOL)
    perm = rng.permutation(50)
    m3, _ = run(a52[:, perm], b52[:, perm])    # same distributions, relabeled vocabulary
    close("(d) KL invariant to vocab permutation", m3["kl_ref_cand"]["mean"], m1["kl_ref_cand"]["mean"], ULP_TOL)
    close("(d) JS invariant to vocab permutation", m3["js_symmetric"]["mean"], m1["js_symmetric"]["mean"], ULP_TOL)
    close("(d) top-32 KL invariant to vocab permutation", m3["kl_ref_cand_top32"]["mean"],
          m1["kl_ref_cand_top32"]["mean"], ULP_TOL)

    # top-32 KL reduces to the full KL when V <= 32 (the union is the whole row)
    a17 = rng.normal(size=(3, 17))
    b17 = rng.normal(size=(3, 17)) + 0.01
    m, _ = run(a17, b17)
    close("(top-k) V=17: top-32 KL == full KL", m["kl_ref_cand_top32"]["mean"], m["kl_ref_cand"]["mean"], ULP_TOL)

    # row plumbing: trim to min N, reject nonfinite rows with counts, refuse vocab mismatch / empty input
    base = rng.normal(size=(6, 10))
    ref6 = base.copy()
    ref6[3, 2] = float("nan")
    cand6 = rng.normal(size=(6, 10))
    cand6[1, 0] = float("inf")
    m, _ = run(ref6, cand6)
    r = m["rows"]
    ok("(rows) trim/report", (r["ref"], r["cand"], r["trimmed_to"], r["nonfinite_ref"], r["nonfinite_cand"],
                              r["rejected_union"], r["used"]) == (6, 6, 6, 1, 1, 2, 4), str(r))
    m, _ = run(base[:6], base[:4])
    ok("(rows) N trim to 4", m["rows"]["used"] == 4 and m["rows"]["trimmed_to"] == 4, str(m["rows"]))
    for name, ref_bad, cand_bad in (("(rows) vocab mismatch refused", np.zeros((2, 4)), np.zeros((2, 5))),
                                    ("(rows) all-nonfinite refused", np.full((2, 4), np.nan), np.zeros((2, 4)))):
        try:
            run(ref_bad, cand_bad)
            failures.append(f"{name}: no error raised")
        except ValueError:
            pass

    # loaders: .jsonl round-trips the same numbers as .npy (NaN literals parse, and the row is rejected later)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        npy = Path(td) / "t.npy"
        jl = Path(td) / "t.jsonl"
        np.save(npy, base[:3])
        jl.write_text("\n".join(json.dumps({"logits": [float(v) for v in row]}) for row in base[:2]) + "\n"
                      + json.dumps({"logits": [float("nan")] * 10}) + "\n")
        arr_npy = load_matrix(str(npy))
        arr_jl = load_matrix(str(jl))
        ok("(load) .npy shape", arr_npy.shape == (3, 10), str(arr_npy.shape))
        ok("(load) .jsonl values == .npy", np.array_equal(arr_jl[:2], arr_npy[:2]) and arr_jl.shape == (3, 10))
        m, _ = run(arr_npy[:2], arr_jl[:2])
        exact_zero("(load) .npy vs .jsonl identical rows", m["kl_ref_cand"]["mean"])

    if failures:
        print(f"SELFTEST FAIL ({len(failures)}):")
        for f in failures:
            print("  " + f)
        return 1
    print("selftest: all fixtures green (analytic KL/JS, exact identity, brute-force normals, permutation "
          "invariance, top-32 reduction, row rejection, loaders)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("    .venv"):])
    ap.add_argument("--ref", help="reference logits: .npy [N, V] or .jsonl of {\"logits\": [...]} records")
    ap.add_argument("--cand", help="candidate logits, same two formats")
    ap.add_argument("--json", action="store_true", help="machine-readable {metrics, verdict} instead of the table")
    ap.add_argument("--selftest", action="store_true", help="run the analytic fixtures and exit nonzero on failure")
    a = ap.parse_args(argv)
    if a.selftest:
        return cmd_selftest()
    if not a.ref or not a.cand:
        ap.error("--ref and --cand are required (or --selftest)")
    try:
        return cmd_compare(a)
    except (ValueError, OSError) as e:
        print(f"kl_yardstick: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
