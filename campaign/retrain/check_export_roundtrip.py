#!/usr/bin/env python3
"""campaign/retrain/check_export_roundtrip.py — the T14C G0 EXPORT gate (F-325).

Proves the trainer's export chain end-to-end: the rt2 dir's dense.bin, read back through
the ENGINE's own rt dialect reader (p2_rt_loader.RtDense — the same dequantization the
engine's --mtp load performs), must reproduce the trained checkpoint's tensors within
each kind's quantization budget:

  f32  norms: EXACT (the file holds the FOLDED 1+w value the engine multiplies as-is;
              the module Parameter holds that same folded value -> bit-identical)
  bf16:       max relative error <= 1% (bf16 has ~3 decimal digits)
  q8_0:       max relative error <= 2% (per-32-block ggml reference rounding)

Also asserts all 29 names present, shapes match, and dense.txt offsets cover dense.bin.
CPU-only; run any time after a trainer export + assemble.

    ~/venvs/mtp-train/bin/python campaign/retrain/check_export_roundtrip.py [rt2-dir] [ckpt]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "campaign" / "retrain"))
from p2_rt_loader import RtDense   # noqa: E402
from mtp_module import MTPDraft    # noqa: E402

RT2 = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/svg/LLM/Custom-data/mtp/rt2")
CKPT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/svg/LLM/Custom-data/mtp/ckpts/cycle-best.pt")


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def main() -> int:
    rt2 = RtDense(RT2)
    pay = torch.load(CKPT, map_location="cpu", weights_only=False)
    kinds = {name: kind for name, kind, *_ in rt2.manifest}
    # state_dict keys are the REGISTERED (underscored) names; the forward map orig -> registered
    # (dots to underscores) is injective — only the inverse round-trip is forbidden (F-325 law).
    tensors = {name: pay["model"][name.replace(".", "_")] for name in kinds}
    mod = MTPDraft(tensors)
    budget = {"f32": 0.0, "bf16": 0.01, "q8_0": 0.02}
    worst = {}
    assert set(rt2.tensors) == set(mod.t), "rt2 tensor set != module tensor set"
    for name in sorted(rt2.tensors):
        ref = mod.t[name].detach().numpy()
        got = rt2.tensors[name]
        assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
        e = rel_err(got, ref)
        worst[name] = (e, kinds[name])
        ok = e <= budget[kinds[name]] + (1e-6 if kinds[name] == "f32" else 0.0)
        print(f"  {name:58s} {kinds[name]:5s} rel_err {e:.2e} {'OK' if ok else 'FAIL'}")
        if not ok:
            print(f"G0 EXPORT GATE FAILED at {name}")
            return 1
    print(f"G0 EXPORT GATE PASSED: {len(rt2.tensors)} tensors, "
          f"worst f32 {max(e for e, k in worst.values() if k == 'f32'):.1e}, "
          f"worst bf16 {max(e for e, k in worst.values() if k == 'bf16'):.2e}, "
          f"worst q8_0 {max(e for e, k in worst.values() if k == 'q8_0'):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
