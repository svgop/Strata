#!/usr/bin/env python3
"""tools/refusal_subspace.py — build the orthogonal-complement axis of a refusal subspace.

WHY THIS EXISTS
The armed engine has one projection axis per layer (`project`: h -= s (h.v) v). A rank-1 projection
removes one direction. The refusal signal is not guaranteed rank-1: two published directions for this
model family exist - the per-layer activation-difference vector (`u_l`, Cudecnik) and the model-level
direction recovered from the Huihui rank-one weight edit (`r`). They are ~0.77 aligned at the peak layer,
not identical. Removing both axes removes the full span{u_l, r} (a rank-2 ablation).

The second axis is the part of `r` that `u_l` does not already cover (Gram-Schmidt):
    w_l = normalize(r - (r . u_l) u_l)
so `u_l` and `w_l` are orthogonal and span{u_l, w_l} = span{u_l, r}. Applied as two `project` slots (an
engine change - one projection per layer today), the pair removes the whole span. With one slot the file
is inert: projecting w_l alone leaves u_l's component untouched.

WHAT THIS IS NOT
This is an OFFLINE interim: both inputs are existing vectors, no activation measurement. The stronger v2
is extracted from activations with `--dump-layers` (see
campaign/research/2026-11-20-vector-potency-and-stacking.md): per-layer diff-in-means over contrast
prompts, k>1 directions via SVD of the paired-difference matrix, the harmless-mean component projected
out, and per-layer weights selected by SNR x (1 - cosine). This tool's output is the cheap rank-2
candidate to measure while that measurement job is planned.

Usage:
  python tools/refusal_subspace.py \
    --primary data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf \
    --reference /mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-refusal-r.gguf \
    --out /mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-refusal-v2-orth.gguf
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from gguf_reader import GGUFFile          # noqa: E402  (the GGUF owner)
from gguf_writer import GGUFWriter        # noqa: E402  (the GGUF owner)


def read_vectors(path: str) -> dict[int, np.ndarray]:
    g = GGUFFile(path)
    raw = open(path, "rb").read()
    out: dict[int, np.ndarray] = {}
    for t in g.tensors:
        if not t.name.startswith("direction."):
            continue
        layer = int(t.name.split(".")[1])
        n = int(t.shape[0])
        off = g.data_start + t.offset
        out[layer] = np.frombuffer(raw[off:off + n * 4], dtype="<f4").astype(np.float64)
    return out


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--primary", required=True, help="per-layer activation-difference vector (u_l)")
    ap.add_argument("--reference", required=True, help="global/recovered reference direction (r)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-layer", type=int, default=None,
                    help="layer to read from --reference when it is per-layer; default = its first")
    ap.add_argument("--layers", type=int, default=47)
    args = ap.parse_args()

    u = read_vectors(args.primary)
    ref = read_vectors(args.reference)
    r = unit(ref[args.ref_layer if args.ref_layer is not None else min(ref)])

    rows, report = {}, {"per_layer": []}
    for l in range(1, args.layers + 1):
        if l not in u:
            continue
        ul = unit(u[l])
        wr = r - float(r @ ul) * ul
        nrm = float(np.linalg.norm(wr))
        if nrm < 1e-8:
            continue
        w = wr / nrm
        rows[l] = w
        report["per_layer"].append({
            "layer": l,
            "cos_u_r": round(float(ul @ r), 4),
            "r_removed_by_u": round(float(ul @ r) ** 2, 4),   # energy of r that u alone removes
            "residual_r_norm": round(nrm, 4),                  # energy of r left for w
            "cos_w_u": round(float(w @ ul), 12),
            "cos_w_r": round(float(w @ r), 4),
        })

    if not rows:
        raise SystemExit("no layers written")

    writer = GGUFWriter()
    writer.add("general.architecture", "controlvector")
    writer.add("controlvector.model_hint", "qwen4exp")
    writer.add("controlvector.layer_count", args.layers)
    writer.add("strata.refusal.kind", "orthogonal complement of the reference direction against the per-layer axis (rank-2 pair, slot 2)")
    writer.add("strata.refusal.primary", args.primary)
    writer.add("strata.refusal.reference", args.reference)
    for l in sorted(rows):
        writer.add_f32(f"direction.{l}", rows[l].astype(np.float32))
    path = writer.write(args.out)

    pl = report["per_layer"]
    print(json.dumps({
        "layers": len(rows),
        "cos_u_r_range": [min(x["cos_u_r"] for x in pl), max(x["cos_u_r"] for x in pl)],
        "mean_r_removed_by_u": round(float(np.mean([x["r_removed_by_u"] for x in pl])), 3),
        "mean_cos_w_r": round(float(np.mean([x["cos_w_r"] for x in pl])), 3),
        "max_abs_cos_w_u": max(abs(x["cos_w_u"]) for x in pl),
        "per_layer": pl,
    }, indent=1))
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
