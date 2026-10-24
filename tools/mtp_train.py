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
    import torch.nn.functional as F
    from p2_rt_loader import RtDense
    from p2_forward import DraftLayer, BlobReader, silu, sigmoid_f32

    # ---- the trainable parameter view: dense tensors as torch Parameters (experts/head frozen, F-114)
    rt = RtDense()
    params = {k: torch.nn.Parameter(torch.from_numpy(v.copy()), requires_grad=True)
              for k, v in rt.tensors.items()}
    dv = np.fromfile(REPO / "/home/svg/LLM/data/mtp-en/rt/draft_vocab.bin", dtype=np.int32)
    in_sub = -np.ones(248320, dtype=np.int64); in_sub[dv] = np.arange(len(dv))
    hidden = (2 * len(dv) + 255) & ~255

    # ---- the acceptance evaluator (greedy per-chain-step match, position-bucketed) on torch windows
    def eval_acceptance(wins_idx, shards, n=32):
        # stock baseline = the current (untrained) params; returns match rates per depth 1..K
        hits = np.zeros(K, dtype=np.int64); tries = np.zeros(K, dtype=np.int64)
        rng = np.random.default_rng(99)
        sel = rng.choice(len(wins_idx), size=min(n, len(wins_idx)), replace=False)
        for wi in sel:
            start, pmin, _ = wins_idx[wi]
            for anchor in range(0, W - K, 64):
                r0 = anchor
                Rin = torch.from_numpy(shards[start + r0][2].copy()).float()
                labels = shards[start + r0 + 1:start + r0 + 1 + K, 1]
                # (the full recursion assembles on the G0 module's torch port; the first
                # cycle prices depth-1: the mixer + head on the true Rin)
                sample = Rin  # placeholder wiring: replaced by the module chain next
                for d in range(1):
                    sub = in_sub[labels[d]]
                    if sub < 0: continue
                    tries[d] += 1
                    # argmax over the sub-vocab head on `sample` (head = the main model's, frozen)
                    hits[d] += 0  # wired when the torch module chain lands
        return hits, tries

    # ---- assemble shards in memory (first-cycle scale fits)
    shards = []
    for shard, pos, tok, R, seams in load_shards(Path(a.harvest)):
        shards.append((shard, pos, tok, R))
    print(f"loaded {len(shards)} shards, {sum(len(s[1]) for s in shards)} records")
    wins = index_windows(Path(a.harvest))
    print(f"windows: {len(wins)} — the training loop + evaluator assemble on these; "
          f"first cycle = {a.epochs} epochs, batch {a.batch}, lr {a.lr}")
    # THE FIRST TRAINING CYCLE runs when the torch module chain (the G0 forward ported
    # to trainable tensors) lands — the two-mode design per exec-C §2.2-§2.4; the
    # evaluator's stock baseline is the params above, untrained.
    return 0


if __name__ == "__main__":
    sys.exit(main())
