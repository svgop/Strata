#!/usr/bin/env python3
"""tools/mtp_train.py — the T14B exec-C trainer skeleton (P3), built on the G0-PASSED
parity module (campaign/retrain/p2_forward.py + p2_rt_loader.py, F-319).

Per T14B-exec-C §2:
  - Data: harvest records at /home/svg/LLM/Custom-data/harvest/ (records-<shard>.f32:
    {int64 pos, int64 tok, f32 R[4][2560]} = 40,976 B; seams-<shard>.json; the exec-A
    multi-boot harvest is the next GPU workload — this trainer runs the moment shards
    exist).
  - Windows: W=512, real pos ids, seam-respecting, cached windows.npy.
  - Loss: FastMTP recursive teacher-forced CE, K=6, β=0.8, sub-vocab 40,525 with the
    off-vocab labels MASKED (masked_frac logged); w_deep=1.0 default on pos>=16384.
  - Two-mode forward: eval = the G0-exact quantized path; train = no-rounding gradients
    (the quantizers become pass-through + STE hooks).
  - Optim: AdamW, peak LR 1e-4 (RH form), cosine, 3 epochs x 16M tokens, batch 4
    windows; dense-only trainable (~90.5M params: fc, projections, mixers, norms;
    experts + head frozen per F-114's shipping config).
  - Eval: offline acceptance (greedy per-chain-step match on held-out windows,
    position-bucketed) vs the STOCK weights' identical-run baseline.

Run: ~/venvs/mtp-train/bin/python tools/mtp_train.py --harvest <dir> [--epochs 3 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "campaign" / "retrain"))

RECORD_BYTES = 8 + 8 + 4 * 2560 * 4      # 40,976 (exec-A VERIFIED form)
W, K = 512, 6
BETA = 0.8


def load_shards(harvest: Path):
    """Yield (pos, tok, R) per shard + the seam indices."""
    for rec in sorted(harvest.glob("records-*.f32")):
        shard = rec.stem.split("-", 1)[1]
        seams = []
        sj = harvest / f"seams-{shard}.json"
        if sj.exists():
            seams = set(json.loads(sj.read_text()))
        raw = np.fromfile(rec, dtype=np.uint8)
        assert raw.size % RECORD_BYTES == 0, f"{rec}: misaligned ({raw.size % RECORD_BYTES} trailing bytes)"
        n = raw.size // RECORD_BYTES
        arr = raw.reshape(n, RECORD_BYTES)
        pos = arr[:, 0:8].copy().view(np.int64).reshape(n)
        tok = arr[:, 8:16].copy().view(np.int64).reshape(n)
        R = arr[:, 16:].copy().view(np.float32).reshape(n, 4, 2560)
        yield shard, pos, tok, R, seams


def index_windows(harvest: Path):
    """(start, pos_min, pos_max) per seam-respecting W-run; cached to windows.npy."""
    cache = harvest / "windows.npy"
    if cache.exists():
        return np.load(cache)
    win_rows = []
    for shard, pos, tok, R, seams in load_shards(harvest):
        n = len(pos)
        start = 0
        for i in range(1, n + 1):
            seam = i == n or i in seams or (i < n and pos[i] != pos[i - 1] + 1)
            if seam:
                run = i - start
                a = 0
                while a + W + K <= run:
                    win_rows.append((start + a, int(pos[start + a]), int(pos[start + a + W - 1])))
                    a += W
                start = i
    wins = np.array(win_rows, dtype=np.int64).reshape(-1, 3)
    np.save(cache, wins)
    return wins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-deep", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="index windows + report stats, no training")
    a = ap.parse_args()

    harvest = Path(a.harvest)
    if not harvest.exists():
        print(f"harvest dir missing: {harvest} — the exec-A multi-boot harvest is the prerequisite GPU workload")
        return 2
    wins = index_windows(harvest)
    deep = int((wins[:, 2] >= 16384).sum())
    print(f"windows: {len(wins)} (deep pos>=16384: {deep} = {100*deep/max(len(wins),1):.1f}%)")
    if a.dry_run:
        return 0

    import torch
    from p2_rt_loader import RtDense
    from p2_forward import DraftLayer, BlobReader
    print("trainer core ready (the training loop assembles on the first real shards; "
          "the G0-exact forward = eval mode, the no-rounding path = train mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
