#!/usr/bin/env python
"""F-55: rebuild the adapter's index_qk from the STANDING rt's own q8_0 bytes (the
engine-graph-correct 2560-wide values), so the trained-head rt carries the stock indexer.
Usage: python tools\mtp_indexqk_splice.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

RT = Path(r"E:\LLM\Engine\strata\data\mtp\rt")
SRC = Path(r"E:\LLM\Engine\strata-custom-2\data\mtp\r17-fetch")

# locate the standing index_qk entry: name kind rows cols offset bytes
off = bytes_ = None
for ln in (RT / "dense.txt").read_text().splitlines():
    p = ln.split()
    if p[0] == "self_attn.indexer.index_qk_proj.weight":
        off, bytes_ = int(p[4]), int(p[5])
        break
assert off is not None
raw = np.frombuffer((RT / "dense.bin").read_bytes()[off:off + bytes_], dtype=np.uint8)
b = raw.reshape(-1, 34)
d = b[:, :2].copy().view(np.float16).astype(np.float32)
q = b[:, 2:].view(np.int8).astype(np.float32)
vals = (q * np.repeat(d, 32, axis=1)).reshape(-1)          # (640*2560,)
iqk = vals.reshape(640, 2560).astype(np.float32)
f = SRC / "layers_0_self_attn_indexer_index_qk_proj_weight.bin"
((iqk.view(np.uint32) >> 16).astype(np.uint16)).tofile(f)
import json, hashlib
mp = SRC / "mtp-manifest.json"
mani = json.loads(mp.read_text())
for t in mani:
    if t["name"].endswith("indexer.index_qk_proj.weight"):
        t["shape"] = [640, 2560]
        t["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
mp.write_text(json.dumps(mani, indent=1))
print("spliced standing index_qk ->", f, iqk.shape, "+ manifest updated")
