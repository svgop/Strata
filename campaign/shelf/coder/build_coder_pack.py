#!/usr/bin/env python3
"""Coder pack v2: the masked-router build.

The engine requires every layer's experts at a uniform stride in one file (ds4_model.cu:1828),
so sparse re-pointing cannot work.  But the Coder IS the base model with 192 experts kept and
its router rows taken from the base (kept-experts.json: 'ffn_gate_inp ... hold these experts in
this order'; every other tensor equals the base build).  So the exact functional equivalent is:

  - tensors.txt: the BASE manifest verbatim (all 384 experts, uniform strides, same files),
    with only layers.L.ffn.gate.bias redirected to coder-mask.bin and the mtp.* (dspark)
    tensors + the drafter file dropped (the Coder has no dspark layers)
  - coder-mask.bin: per layer, the base's 384 bias values with every NON-kept expert set to
    -1e30 - the biased top-k can then never pick a pruned expert, and the kept experts' scores
    and normalization are untouched, so the routing equals the Coder's exactly
  - ds4.txt: the base's, dspark keys dropped (n_routed_experts stays 384: that is the file)
"""
import json, os, shutil

MAIN = '/home/svg/LLM/data/packs/deepseek-v4.1-flash-ist-das-2.00bpw'
OUT = '/home/svg/LLM/data/packs/deepseek-v4.1-flash-coder-2.00bpw'
MASK = '/home/svg/LLM/data/models/GSQ-RCO/Deepseek-v4.1-Flash/IQ2-XXS/coder-mask.bin'
KEPT = '/tmp/kept.json'

kept = {int(k): set(v) for k, v in json.load(open(KEPT))['kept_experts'].items()}
NL, NE = 40, 384

files, tensors = {}, {}
for line in open(f'{MAIN}/tensors.txt'):
    t = line.split()
    if not t:
        continue
    if t[0] == 'file':
        files[int(t[1])] = t[2]
    elif t[0] == 't':
        tensors[t[1]] = (t[2], int(t[3]), int(t[4]), int(t[5]), [int(x) for x in t[6:]])

model_dir = next(l.split(' ', 1)[1].strip() for l in open(f'{MAIN}/ds4.txt') if l.startswith('model_dir '))

# ---- coder-mask.bin: per layer 384 F32 biases, non-kept = -1e30 ----
import struct
offsets = []
with open(MASK, 'wb') as out:
    for L in range(NL):
        gb = tensors[f'layers.{L}.ffn.gate.bias']            # F32 (384,)
        src = open(os.path.join(model_dir, files[gb[1]]), 'rb')
        src.seek(gb[2])
        vals = struct.unpack(f'<{NE}f', src.read(NE * 4))
        masked = [v if e in kept[L] else -1e30 for e, v in enumerate(vals)]
        offsets.append(out.tell())
        out.write(struct.pack(f'<{NE}f', *masked))
print('coder-mask.bin bytes:', os.path.getsize(MASK))

# ---- tensors.txt: base verbatim, gate.bias -> coder-mask.bin, no mtp ----
ofiles = {i: n for i, n in files.items() if i != 5}          # drop the drafter file
ofiles[5] = 'coder-mask.bin'
n = 0
with open(OUT + '/tensors.txt', 'w') as f:
    for i in sorted(ofiles):
        f.write(f'file {i} {ofiles[i]}\n')
    for name, (dtype, fi, off, nb, dims) in tensors.items():
        if name.startswith('mtp.'):
            continue
        if name.endswith('.ffn.gate.bias'):
            L = int(name.split('.')[1])
            f.write(f't layers.{L}.ffn.gate.bias F32 5 {offsets[L]} {NE * 4} {NE}\n')
        else:
            f.write(f"t {name} {dtype} {fi} {off} {nb} " + ' '.join(str(d) for d in dims) + '\n')
        n += 1
print('tensors.txt entries:', n)

# ---- ds4.txt: base's, no dspark ----
with open(OUT + '/ds4.txt', 'w') as f:
    for line in open(f'{MAIN}/ds4.txt'):
        if line.startswith('dspark_'):
            continue
        f.write(line)

shutil.copy2(f'{MAIN}/engram.bin', OUT + '/engram.bin')
shutil.copytree(f'{MAIN}/tokenizer', OUT + '/tokenizer', dirs_exist_ok=True)
print('done:', OUT)
