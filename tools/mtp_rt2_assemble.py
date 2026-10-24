#!/usr/bin/env python3
"""tools/mtp_rt2_assemble.py — stage 2 of the FastMTP retrain pipeline (F-325).

Assembles the --mtp-ready rt2 drafter dir from the trainer's export:

    /home/svg/LLM/Custom-data/mtp/export/dense_rt2.{bin,txt}   (trained, F-325 trainer export)
  + experts.bin, draft_vocab.bin                                (FROZEN, byte-identical
                                                                 copies from the stock rt)
    -> /home/svg/LLM/Custom-data/mtp/rt2/{dense.bin,dense.txt,experts.bin,draft_vocab.bin}

The engine consumes it unchanged: `--mtp /home/svg/LLM/Custom-data/mtp/rt2` in a launcher
copy (LAUNCH LAW). Verifies: 29 dense.txt lines, sizes match the manifest's byte column,
sha256(experts/draft_vocab) == the stock source. Atomic: staged in rt2.tmp/ then renamed.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

EXPORT = Path("/home/svg/LLM/Custom-data/mtp/export")
STOCK_RT = Path("/home/svg/LLM/data/mtp/rt")          # the symlink-resolved true source
OUT = Path("/home/svg/LLM/Custom-data/mtp/rt2")
FROZEN = ("experts.bin", "draft_vocab.bin")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    dense_bin, dense_txt = EXPORT / "dense_rt2.bin", EXPORT / "dense_rt2.txt"
    for p in (dense_bin, dense_txt):
        if not p.exists():
            print(f"missing {p} — run campaign/retrain/train_cycle.py (export stage) first")
            return 2
    lines = [l.split() for l in dense_txt.read_text().splitlines() if l.strip()]
    assert len(lines) == 29, f"dense_rt2.txt has {len(lines)} tensors (expected 29)"
    total = int(lines[-1][4]) + int(lines[-1][5])
    assert dense_bin.stat().st_size >= total, "dense_rt2.bin shorter than its manifest"

    tmp = OUT.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copy2(dense_bin, tmp / "dense.bin")
    shutil.copy2(dense_txt, tmp / "dense.txt")
    meta = {"dense_bytes": dense_bin.stat().st_size, "dense_tensors": len(lines)}
    for name in FROZEN:
        src = STOCK_RT / name
        shutil.copy2(src, tmp / name)
        a, b = sha256(src), sha256(tmp / name)
        assert a == b, f"{name}: copy hash mismatch"
        meta[name + "_sha256"] = a
        meta[name + "_bytes"] = src.stat().st_size
    (tmp / "rt2_meta.json").write_text(__import__("json").dumps(meta, indent=1))
    if OUT.exists():
        shutil.rmtree(OUT)          # replace wholesale; the previous rt2 is regenerable
    os.rename(tmp, OUT)
    print(f"rt2 assembled: {meta['dense_tensors']} trained tensors + frozen experts/draft_vocab -> {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:18s} {p.stat().st_size / 1e6:9.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
