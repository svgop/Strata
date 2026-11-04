#!/usr/bin/env python3
"""tools/dsv41/vision_pack.py - DeepSeek-V4.1's vision tower for a GSQ-RCO pack, from the ISTA-DAS mmproj.

The engine reads the tower as pack tensors under the checkpoint's own naming (vision.blocks.N.*,
aligner.*, image_start/newline/end) and its geometry from ds4.txt's vision.<key> lines
(ds4_model.cu: load_vision).  The GSQ-RCO quants ship no vision tensors, but every build of the
model shares ISTA-DASLab's mmproj GGUF (the llama.cpp packaging of the same tower), which holds
every tensor the engine reads under v.blk.*/mm.*/v.* names - and GGUF stores weights
[ne0=in ... neN=out], the engine's own row-major [out][in], so the mapping below is a pure
rename: no transposition, no copies, the pack's tensors.txt points straight into the mmproj file.

    python tools/dsv41/vision_pack.py --mmproj <mmproj.gguf> --pack <pack dir> [--image-token-id N]

Appends to <pack>/tensors.txt a `file` line for the mmproj and the mapped tensor entries, and to
<pack>/ds4.txt the vision.<key> geometry lines and image_token_id (from the mmproj's clip.vision.*
metadata; the token id comes from the checkpoint's config.json - 129264 for V4.1-Flash-1017 -
because the mmproj does not carry it).  Idempotent: a prior integration is removed first.  Every
mapped tensor's shape is checked against the geometry, and the marker rows against the model's
width, before anything is written.

The mapping (mmproj -> the engine's name):
    v.patch_embd.weight           (P,P,3,D)   -> vision.patch_embed.proj.weight  [D,3,P,P] - the same
    v.patch_embd.bias             (D)         -> vision.patch_embed.proj.bias
    v.post_ln.weight              (D)         -> vision.norm.weight
    v.blk.N.attn_qkv.weight       (D,3D)      -> vision.blocks.N.attn.wqkv.weight
    v.blk.N.attn_qkv.bias         (3D)        -> vision.blocks.N.attn.wqkv.bias
    v.blk.N.attn_out.weight       (D,D)       -> vision.blocks.N.attn.wo.weight
    v.blk.N.attn_out.bias         (D)         -> vision.blocks.N.attn.wo.bias
    v.blk.N.ffn_gate.weight       (D,I)       -> vision.blocks.N.mlp.gate.weight   (w1's first half)
    v.blk.N.ffn_up.weight         (D,I)       -> vision.blocks.N.mlp.up.weight     (w1's second half)
    v.blk.N.ffn_down.weight       (I,D)       -> vision.blocks.N.mlp.w2.weight
    v.blk.N.ln1.weight            (D)         -> vision.blocks.N.norm1.weight
    v.blk.N.ln2.weight            (D)         -> vision.blocks.N.norm2.weight
    mm.1.weight                   (D*M*M,out) -> aligner.w1.weight
    mm.1.bias                     (out)       -> aligner.w1.bias
    mm.2.weight                   (out,out)   -> aligner.w2.weight
    mm.2.bias                     (out)       -> aligner.w2.bias
    v.token_embd.img_start        (width)     -> image_start
    v.image_newline               (width)     -> image_newline
    v.token_embd.img_end          (width)     -> image_end
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

# the engine's tensor names, by kind: (pack name, mmproj name, shape check as a function of D/I/P/M/layers)
def block_map(d, i):
    return [
        ("attn.wqkv.weight", "attn_qkv.weight", (d, 3 * d), "BF16"),
        ("attn.wqkv.bias", "attn_qkv.bias", (3 * d,), "F32"),
        ("attn.wo.weight", "attn_out.weight", (d, d), "BF16"),
        ("attn.wo.bias", "attn_out.bias", (d,), "F32"),
        ("mlp.gate.weight", "ffn_gate.weight", (d, i), "BF16"),
        ("mlp.up.weight", "ffn_up.weight", (d, i), "BF16"),
        ("mlp.w2.weight", "ffn_down.weight", (i, d), "BF16"),
        ("norm1.weight", "ln1.weight", (d,), "F32"),
        ("norm2.weight", "ln2.weight", (d,), "F32"),
    ]


def read_gguf(path: pathlib.Path):
    """(kv dict of scalars/strings, {tensor name: (dtype, shape, data offset)}, data base, alignment)."""
    buf = path.read_bytes()
    if buf[:4] != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF")
    p = 8
    nt, nk = struct.unpack_from("<QQ", buf, p)
    p += 16
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def value(t):
        nonlocal p
        if t == 8:
            n = struct.unpack_from("<Q", buf, p)[0]
            p += 8
            s = buf[p:p + n].decode("utf-8", "replace")
            p += n
            return s
        if t == 9:
            at = struct.unpack_from("<I", buf, p)[0]
            n = struct.unpack_from("<Q", buf, p + 4)[0]
            p += 12
            return [value(at) for _ in range(n)]
        v = struct.unpack_from({0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
                                7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t], buf, p)[0]
        p += sizes[t]
        return v

    kv = {}
    for _ in range(nk):
        n = struct.unpack_from("<Q", buf, p)[0]
        p += 8
        key = buf[p:p + n].decode("utf-8", "replace")
        p += n
        t = struct.unpack_from("<I", buf, p)[0]
        p += 4
        kv[key] = value(t)
    align = kv.get("general.alignment", 32)
    tensors = {}
    for _ in range(nt):
        n = struct.unpack_from("<Q", buf, p)[0]
        p += 8
        name = buf[p:p + n].decode("utf-8", "replace")
        p += n
        nd = struct.unpack_from("<I", buf, p)[0]
        p += 4
        dims = struct.unpack_from("<" + "Q" * nd, buf, p)
        p += 8 * nd
        ty = struct.unpack_from("<I", buf, p)[0]
        p += 4
        off = struct.unpack_from("<Q", buf, p)[0]
        p += 8
        tensors[name] = (ty, dims, off)
    base = (p + align - 1) & ~(align - 1)
    return kv, tensors, base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mmproj", required=True, type=pathlib.Path, help="the ISTA-DAS mmproj GGUF")
    ap.add_argument("--pack", required=True, type=pathlib.Path, help="the pack dir (tensors.txt + ds4.txt are updated)")
    ap.add_argument("--remove", action="store_true", help="take a prior integration out of the pack again")
    ap.add_argument("--image-token-id", type=int, default=129264,
                    help="the model's image token id (the checkpoint's config.json; the mmproj does not carry it)")
    a = ap.parse_args()

    tt, ds = a.pack / "tensors.txt", a.pack / "ds4.txt"
    if a.remove:
        lines = [ln for ln in tt.read_text().splitlines()
                 if not (ln.split() and ln.split()[0] == "file" and a.mmproj.name in ln.split()[2].lstrip("_"))
                 and not (ln.split() and ln.split()[0] == "t" and
                          (ln.split()[1].startswith(("vision.", "aligner.")) or
                           ln.split()[1] in ("image_start", "image_newline", "image_end")))]
        tt.write_text("\n".join(lines) + "\n")
        dsl = [l for l in ds.read_text().splitlines()
               if not (l.startswith("vision.") or l.startswith("image_token_id"))]
        ds.write_text("\n".join(dsl) + "\n")
        print("pack restored: the vision integration is out")
        return

    kv, tensors, base = read_gguf(a.mmproj)
    cv = {k.split("clip.vision.", 1)[1]: v for k, v in kv.items() if isinstance(v, (int, float)) and k.startswith("clip.vision.")}
    # the keys the mmproj always carries; rope_theta / max_image_tokens are not in its metadata (the
    # checkpoint's config.json has them) and default to the engine's own defaults (1e4, 1024)
    need = {"block_count": "num_hidden_layers", "embedding_length": "hidden_size",
            "attention.head_count": "num_attention_heads", "feed_forward_length": "intermediate_size",
            "patch_size": "patch_size"}
    missing = [k for k in need if k not in cv]
    if missing:
        raise SystemExit(f"{a.mmproj}: the mmproj's metadata lacks {missing} - not a DeepSeek vision mmproj?")
    layers, d = int(cv["block_count"]), int(cv["embedding_length"])
    inter, patch = int(cv["feed_forward_length"]), int(cv["patch_size"])
    merge = int(cv.get("projector.scale_factor", 3))
    max_tokens = int(cv.get("max_image_tokens", 1024))
    rope_theta = cv.get("rope_theta", 10000.0)
    print(f"tower: {layers} blocks, D={d}, I={inter}, patch={patch}, merge={merge}, max_tokens={max_tokens}")

    # the entries: (pack tensor name, dtype, shape, mmproj tensor name)
    entries = [("vision.patch_embed.proj.weight", "F32", (patch, patch, 3, d), "v.patch_embd.weight"),
               ("vision.patch_embed.proj.bias", "F32", (d,), "v.patch_embd.bias"),
               ("vision.norm.weight", "F32", (d,), "v.post_ln.weight")]
    for n in range(layers):
        for name, src, shape, dt in block_map(d, inter):
            entries.append((f"vision.blocks.{n}.{name}", dt, shape, f"v.blk.{n}.{src}"))
    out = cv["projection_dim"]
    entries += [("aligner.w1.weight", "BF16", (d * merge * merge, out), "mm.1.weight"),
                ("aligner.w1.bias", "F32", (out,), "mm.1.bias"),
                ("aligner.w2.weight", "BF16", (out, out), "mm.2.weight"),
                ("aligner.w2.bias", "F32", (out,), "mm.2.bias")]

    # verify every mapped tensor: present, the expected GGUF type (0 F32, 30 BF16) and shape
    tyname = {0: "F32", 30: "BF16"}
    resolved = []
    for name, dt, shape, src in entries:
        if src not in tensors:
            raise SystemExit(f"the mmproj lacks {src} (needed for {name})")
        ty, dims, off = tensors[src]
        if tyname.get(ty) != dt or tuple(dims) != tuple(shape):
            raise SystemExit(f"{src}: dtype/shape {tyname.get(ty)}{tuple(dims)}, expected {dt}{shape} - "
                             "the layout assumption broke, nothing written")
        resolved.append((name, dt, shape, off))
    for name, src in [("image_start", "v.token_embd.img_start"), ("image_newline", "v.image_newline"),
                      ("image_end", "v.token_embd.img_end")]:
        if src not in tensors:
            raise SystemExit(f"the mmproj lacks {src} (the marker row {name})")
        ty, dims, off = tensors[src]
        if ty != 0 or dims[0] != 5120:
            raise SystemExit(f"{src}: expected F32 [5120] (the model's width), got {tyname.get(ty)}{tuple(dims)}")
        resolved.append((name, "F32", (5120,), off))
    print(f"mapped {len(resolved)} tensors, all shapes verified")

    # the pack: a fresh file slot for the mmproj, the entries pointing at its payload offsets
    tt, ds = a.pack / "tensors.txt", a.pack / "ds4.txt"
    lines = tt.read_text().splitlines()
    vision_at = []
    for i, ln in enumerate(lines):
        w = ln.split()
        if w and w[0] == "file":
            if a.mmproj.name in w[2].lstrip("_"):   # a prior integration's file line goes too
                vision_at.append(i)
        elif w and w[0] == "t" and (w[1].startswith("vision.") or w[1].startswith("aligner.") or
                                    w[1] in ("image_start", "image_newline", "image_end")):
            vision_at.append(i)
    for i in reversed(vision_at):   # idempotent: a prior integration goes first
        del lines[i]
    files = {}   # the file table as it stands after the removal
    for ln in lines:
        w = ln.split()
        if w and w[0] == "file":
            files[int(w[1])] = w[2]
    slot = max(files, default=-1) + 1
    lines.append(f"file {slot} {a.mmproj.name}")
    for name_, dt, shape, off in resolved:
        flat = name_ == "vision.patch_embed.proj.weight"
        if flat:
            shape = (d, 3 * patch * patch)   # the checkpoint declares it a matrix [D][3PP]; the same bytes
        n_el = 1
        for s in shape:
            n_el *= s
        dims = shape if flat else tuple(reversed(shape))   # GGUF dims read inner->outer; the pack outer->inner
        lines.append(f"t {name_} {dt} {slot} {base + off} {n_el * (4 if dt == 'F32' else 2)} " +
                     " ".join(str(s) for s in dims))
    tt.write_text("\n".join(lines) + "\n")

    dsl = [l for l in ds.read_text().splitlines()
           if not (l.startswith("vision.") or l.startswith("image_token_id"))]
    dsl += [f"vision.num_hidden_layers {layers}", f"vision.hidden_size {d}",
            f"vision.num_attention_heads {int(cv['attention.head_count'])}", f"vision.intermediate_size {inter}",
            f"vision.patch_size {patch}", f"vision.rope_theta {rope_theta}",
            f"vision.downsample_ratio {merge}", f"vision.max_image_tokens {max_tokens}",
            f"image_token_id {a.image_token_id}"]
    ds.write_text("\n".join(dsl) + "\n")
    print(f"pack updated: {tt} (+{len(resolved) + 1} lines, file {slot} = {a.mmproj.name}), {ds} (vision geometry)")


if __name__ == "__main__":
    main()
