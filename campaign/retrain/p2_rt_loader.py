#!/usr/bin/env python3
"""campaign/retrain/p2_rt_loader.py — the P2 parity module's weight loader (T14B-exec-B §2).

Loads the ENGINE's rt drafter dialect exactly as the engine serves it — never a BF16
sidecar (law 1) — from /home/svg/LLM/data/mtp-en/rt:
  dense.bin   29 tensors per dense.txt (kinds q8_0 | bf16 | f32)
  experts.bin 512 blobs (not loaded here; the MoE path consumes them via the blob reader
              at exec-C time — the parity module's shared expert + router first)
The q8_0 reader is tools/mtp_rt.py's packer inverted (ggml reference rounding, fp16 d).
The f32 norms are stored WITH the Gemma +1 already folded (law 2): loaded as-is.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

RT = Path("/home/svg/LLM/data/mtp-en/rt")


def _deq_q8_0(raw: bytes, rows: int, cols: int) -> np.ndarray:
    """ggml Q8_0: per 32 values, fp16 d then 16 int8 pairs.  d = amax/127 (the packer's exact rule)."""
    n_blocks = rows * cols // 32
    buf = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * 34)
    blocks = buf.reshape(n_blocks, 34)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    w = q * d[:, None]
    return w.reshape(rows, cols)


def _deq_bf16(raw: bytes, rows: int, cols: int) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)  # little-endian bf16 as raw bits (view, not astype)
    bits = (u16.astype(np.uint32) << 16).astype(np.uint32)
    return (bits.view(np.float32)).reshape(rows, cols)


class RtDense:
    """The 29 dense tensors, dequantized to f32, name -> ndarray."""

    def __init__(self, rt_dir: Path = RT):
        manifest = []
        for line in (rt_dir / "dense.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 6:
                manifest.append((parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])))
        blob = (rt_dir / "dense.bin").read_bytes()
        self.tensors: dict[str, np.ndarray] = {}
        for name, kind, rows, cols, off, nbytes in manifest:
            raw = blob[off:off + nbytes]
            assert len(raw) == nbytes, f"{name}: short read"
            if kind == "q8_0":
                t = _deq_q8_0(raw, rows, cols)
            elif kind == "bf16":
                t = _deq_bf16(raw, rows, cols)
            elif kind == "f32":
                t = np.frombuffer(raw, dtype=np.float32, count=rows * cols).reshape(rows, cols).copy()
            else:
                raise ValueError(f"unknown rt kind {kind} for {name}")
            self.tensors[name] = t
        self.manifest = manifest

    def norms_report(self, max_rows: int = 2000) -> dict[str, tuple]:
        """(rows, cols, norm-of-first-rows, amax) per tensor — the F-298 lesson: sanity
        the data before building math on it."""
        out = {}
        for name, t in self.tensors.items():
            head = t[:max_rows]
            out[name] = (t.shape[0], t.shape[1], float(np.linalg.norm(head) / np.sqrt(head.size)), float(np.abs(head).max()))
        return out


if __name__ == "__main__":
    rt = RtDense()
    print(f"loaded {len(rt.tensors)} dense tensors from {RT}")
    for name, (r, c, rms, amax) in sorted(rt.norms_report().items()):
        flag = " <-- IMPOSSIBLE" if rms > 100 else ""
        print(f"  {name:58s} {r:6d}x{c:<6d} rms {rms:8.4f} amax {amax:8.4f}{flag}")
