#!/usr/bin/env python
"""F-48 bridge: rename the buun-sidecar phaseB MTP head's tensors to the
strata mtp_rt.py naming (mtp.layers.0.*), matching by SHAPE against the
stock MTP GGUF's blk.48.* reference table. Writes a renamed GGUF.
Usage: python mtp_bridge.py <phaseB.gguf> <ref.gguf> <out.gguf>"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py
add_gguf_py()
import gguf

phaseb, ref, outp = sys.argv[1], sys.argv[2], sys.argv[3]

rp = gguf.GGUFReader(ref)
# reference: blk.48.<sub> -> shape; the strata name is mtp.layers.0.<sub>
ref_shapes = {}
for t in rp.tensors:
    if t.name.startswith("blk.48."):
        sub = t.name[len("blk.48."):]
        ref_shapes.setdefault(tuple(t.shape), []).append(sub)
print("reference tensors:", len(ref_shapes), "shapes")

r = gguf.GGUFReader(phaseb)
renames = {}
for t in r.tensors:
    sh = tuple(t.shape)
    subs = ref_shapes.get(sh)
    if subs is None:
        print(f"  NO SHAPE MATCH: {t.name} {sh}")
        continue
    if len(subs) == 1:
        renames[t.name] = "mtp.layers.0." + subs[0]
    else:
        # ambiguous: try suffix match on the tail of the name
        tail = t.name.split(".")[-2:]
        cand = [s for s in subs if s.split(".")[-2:] == tail]
        if len(cand) == 1:
            renames[t.name] = "mtp.layers.0." + cand[0]
        else:
            print(f"  AMBIGUOUS {t.name} {sh}: {subs} (tail {tail})")

print(f"mapped {len(renames)}/{len(list(r.tensors))} tensors")
# sanity: the two critical ones
for want, sh in [("mtp.layers.0.mlp.experts.gate_up_proj", "(512, 1280, 720)"),
                 ("mtp.layers.0.mlp.experts.down_proj", "(512, 2560, 180)")]:
    print(f"  {'OK ' if want in renames.values() else 'MISS'} {want}")

w = gguf.GGUFWriter(outp, "strata-mtp-bridge")
for t in r.tensors:
    if t.name in renames:
        data = np.asarray(t.data)
        w.add_tensor(renames[t.name], data, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("WROTE", outp)
