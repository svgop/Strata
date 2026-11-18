"""tools/mm3/vq_pack.py - the Baekpica MiniMax-M3.1-Flash VQ release into the mm3 engine's pack manifest form
(data/packs/minimax-m3.1-flash-vq/), beside the release's files (no weights copied).

    python tools/mm3/vq_pack.py --release ~/LLM/data/models/VQ/MiniMax-M3.1-Flash

writes the pack the engine's mm3 family reads (src/mm3/mm3_pack.hpp):
  mm3.txt       the geometry under config.json's text_config names - every key src/mm3/mm3_geo.hpp Geo::read
                consumes, the nested sparse_attention_config.* flattened with dotted keys, moe_layer_freq /
                sparse_attention_freq as comma lists, bos/eos from generation_config.json (else
                tokenizer_config.json), and model_dir = the release directory
  tensors.txt   `file <i> <path>` lines (paths relative to model_dir), then
                `t <name> <dtype> <file> <offset> <bytes> <dims...>` - dims outermost first (a matrix
                [out][in]), offset the GGUF-INTERNAL payload start (header + tensor-info table + alignment
                behind it: what Pack::load's whole-file maps index), bytes cross-checked against
                tensor-manifest.json's payload_bytes (refused on mismatch - the pack's own strict-reader law)

Names are the maya pack's convention (tools/mm3/pack_gguf.py); the release keeps the original HF names, which
are exactly what mm3_model.cu looks up (P = "language_model.model."), so trunk / embedding / vision rows are
identity.  Two exceptions:
  - the DSpark draft: mm3_dspark.cu reads under "dspark.language_model.model.dspark.", so every drafter row
    is `dspark.` + the release's name;
  - the experts' VQ containers: each expert's matrix is its own tensor
    (layers.N.block_sparse_moe.experts.E.w1/w3/w2.weight, GVQ1 payload) with its GVR8 residual beside it
    (`<matrix>.weight_residual`) and the per-layer codebooks
    (`...block_sparse_moe.vq_codebook`, `...vq_codebook_residual`) - the mm3_vq lane's rows.

The dtype vocabulary the engine's VQ format class claims (src/mm3/mm3_vq_host.hpp dispatches on the payload
magic, never on GGUF type ids - the ecosystem repurposes 42/43 three ways):
  VQ13    GVQ1 matrix,  0x47565138 - 13-bit codebook indices, [rows][cols] as stored
  VQ13R8  GVR8 residual, 0x47565238 - 8-bit residual stage of the R13+8 layers (release.json r13p8_layers)
  CB8192  E4M3FN codebook [8192, 8],  0x47564342, K = 8192 (the 13-bit stage's)
  CB256   E4M3FN codebook [256, 8],   0x47564342, K = 256 (the R8 residual stage's)
  Q4_K / Q8_0 / BF16 / F32 (+ F16) pass through under their ggml names (what gg::type_of reads today).

Every row is validated after the write (see validate()): bounds against the mapped files' real sizes, the
release's counts, the R13+8 split, the GVQ1/GVR8 headers' magic at payload start, and Geo::read's arithmetic.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import mmap
import os
import pathlib
import struct
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
_spec = importlib.util.spec_from_file_location("gguf_reader", HERE.parent / "gguf_reader.py")
ggr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ggr)   # (tools/gguf_reader.py: the GGUF v3 header reader, no ggml dependency)

FORMAT = "minimax-m3-vq8"
ARCH = "minimax_m3_vq"
GV_HDR = struct.Struct("<4sHBBIIIIIIQQQQ32s32s")   # (FORMAT.md's 128-byte GVQ1/GVR8 header)

# GGUF tensor-info type id -> (manifest `type` string, our dtype tag or None for the K-dependent codebooks)
TYPE_IDS = {
    0x47565138: ("GVQ1", "VQ13"),
    0x47565238: ("GVR8", "VQ13R8"),
    0x47564342: ("VQ_CODEBOOK_E4M3", None),          # CB8192 (K = 8192) / CB256 (K = 256) by entries
    0: ("F32", "F32"), 1: ("F16", "F16"), 8: ("Q8_0", "Q8_0"), 12: ("Q4_K", "Q4_K"), 30: ("BF16", "BF16"),
}
ALLOWED_DTYPES = {"VQ13", "VQ13R8", "CB8192", "CB256", "Q4_K", "Q8_0", "BF16", "F32", "F16"}

# every key Geo::read consumes (mm3_geo.hpp), in its order; lists Geo reads as comma lists
GEO_KEYS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "rotary_dim", "vocab_size", "num_local_experts", "num_experts_per_tok", "intermediate_size",
            "dense_intermediate_size", "shared_intermediate_size", "rope_theta", "rms_norm_eps", "swiglu_alpha",
            "swiglu_limit", "routed_scaling_factor", "moe_layer_freq",
            "sparse_attention_config.sparse_attention_freq", "sparse_attention_config.sparse_num_index_heads",
            "sparse_attention_config.sparse_index_dim", "sparse_attention_config.sparse_block_size",
            "sparse_attention_config.sparse_topk_blocks", "sparse_attention_config.sparse_local_block",
            "sparse_attention_config.sparse_init_block", "bos_token_id", "eos_token_id")
DEFAULT_OUT = pathlib.Path("/mnt/DATA/LLM/Models/GGUF/strata/data/packs/minimax-m3.1-flash-vq")   # the estate (storage canon: packs live in data/, not the repo)


def fail(why: str):
    raise SystemExit("vq_pack: " + why)


def dtype_of(type_id: int, stored_shape: list, manifest_type: str) -> str:
    """Our dtype tag for a stored tensor: the VQ containers by their ids, the codebooks by their entries."""
    want, tag = TYPE_IDS.get(type_id, (None, None))
    if want is None or want != manifest_type:
        fail("type id %#x (%s) the release's contract does not name" % (type_id, manifest_type))
    if tag is None:   # a codebook: K = 8192 -> CB8192, K = 256 -> CB256
        k = stored_shape[0] if stored_shape else 0
        tag = {8192: "CB8192", 256: "CB256"}.get(k)
        if tag is None:
            fail("codebook of K = %d: neither the 13-bit stage's 8192 nor the residual stage's 256" % k)
    return tag


def pack_name(name: str, role: str) -> str:
    """The pack's name for a release tensor: identity everywhere but the drafter (mm3_dspark.cu's
    DP = \"dspark.language_model.model.dspark.\" sits over the release's full names)."""
    return "dspark." + name if role == "drafter" else name


def geometry(text_config: dict, bos: int, eos: int) -> dict:
    """Geo::read's keys from config.json's text_config (the dotted flatten of sparse_attention_config)."""
    sc = text_config.get("sparse_attention_config") or {}
    g = {
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_attention_heads": text_config.get("num_attention_heads"),
        "num_key_value_heads": text_config.get("num_key_value_heads"),
        "head_dim": text_config.get("head_dim"),
        "rotary_dim": text_config.get("rotary_dim", round(text_config.get("head_dim", 0) *
                                                          text_config.get("partial_rotary_factor", 1.0))),
        "vocab_size": text_config.get("vocab_size"),
        "num_local_experts": text_config.get("num_local_experts"),
        "num_experts_per_tok": text_config.get("num_experts_per_tok"),
        "intermediate_size": text_config.get("intermediate_size"),
        "dense_intermediate_size": text_config.get("dense_intermediate_size"),
        "shared_intermediate_size": text_config.get("shared_intermediate_size"),
        "rope_theta": text_config.get("rope_theta"),
        "rms_norm_eps": text_config.get("rms_norm_eps"),
        "swiglu_alpha": text_config.get("swiglu_alpha"),
        "swiglu_limit": text_config.get("swiglu_limit"),
        "routed_scaling_factor": text_config.get("routed_scaling_factor"),
        "moe_layer_freq": text_config.get("moe_layer_freq"),
        "sparse_attention_config.sparse_attention_freq": sc.get("sparse_attention_freq"),
        "sparse_attention_config.sparse_num_index_heads": sc.get("sparse_num_index_heads"),
        "sparse_attention_config.sparse_index_dim": sc.get("sparse_index_dim"),
        "sparse_attention_config.sparse_block_size": sc.get("sparse_block_size"),
        "sparse_attention_config.sparse_topk_blocks": sc.get("sparse_topk_blocks"),
        "sparse_attention_config.sparse_local_block": sc.get("sparse_local_block"),
        "sparse_attention_config.sparse_init_block": sc.get("sparse_init_block", 0),
        "bos_token_id": bos,
        "eos_token_id": eos,
    }
    missing = [k for k, v in g.items() if v is None]
    if missing:
        fail("the release's config.json lacks %s" % ", ".join(missing))
    return g


def token_ids(release: pathlib.Path) -> tuple:
    """(bos, eos) from generation_config.json, else tokenizer_config.json."""
    for name in ("generation_config.json", "tokenizer_config.json"):
        p = release / name
        if not p.is_file():
            continue
        cfg = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(cfg.get("bos_token_id"), int) and isinstance(cfg.get("eos_token_id"), int):
            return int(cfg["bos_token_id"]), int(cfg["eos_token_id"])
    fail("neither generation_config.json nor tokenizer_config.json holds integer bos/eos token ids")


def check_part(part: dict, path: pathlib.Path, mrows: dict, gf: ggr.GGUFFile) -> list:
    """One part's rows ([name, dtype, offset, bytes, shape]) against its GGUF header and the manifest:
    names, shapes, the payload regions, the strict-reader laws.  Refuses on any mismatch."""
    role = part["role"]
    residual = bool(part.get("residual"))
    md = gf.metadata
    if md.get("general.architecture") != ARCH or md.get("minimax_m3_vq.variant") != FORMAT:
        fail("%s: not a %s part (architecture %r variant %r)" % (path.name, FORMAT, md.get("general.architecture"),
                                                                 md.get("minimax_m3_vq.variant")))
    if md.get("minimax_m3_vq.part_role") != role:
        fail("%s: part_role %r, release.json says %r" % (path.name, md.get("minimax_m3_vq.part_role"), role))
    if role == "trunk" and part.get("layer") is not None:
        if md.get("minimax_m3_vq.layer") != part["layer"]:
            fail("%s: layer %r, release.json says %r" % (path.name, md.get("minimax_m3_vq.layer"), part["layer"]))
        want_fv = 2 if residual else 1
        if md.get("minimax_m3_vq.format_version") != want_fv:
            fail("%s: format_version %r, the %s part carries %d" % (path.name, md.get("minimax_m3_vq.format_version"),
                                                                    "R13+8" if residual else "R13", want_fv))
    fsize = path.stat().st_size
    if "bytes" in part and part["bytes"] != fsize:
        fail("%s: %d bytes on disk, release.json says %d" % (path.name, fsize, part["bytes"]))
    if len(gf.tensors) != len(mrows):
        fail("%s: %d tensors in the GGUF, %d in tensor-manifest.json" % (path.name, len(gf.tensors), len(mrows)))
    out = []
    data_start, ts = gf.data_start, gf.tensors
    gaps = []   # [(file offset, byte count)] - the alignment padding the container puts behind small payloads
    for i, t in enumerate(ts):
        m = mrows.get(t.name)
        if m is None:
            fail("%s: %s is not in tensor-manifest.json" % (path.name, t.name))
        if list(reversed(t.shape)) != list(m["stored_shape"]):
            fail("%s: %s dims %s reversed are not the manifest's %s" % (path.name, t.name, t.shape, m["stored_shape"]))
        # the GGUF-internal payload: every start 128-aligned, an interior tensor's slack the alignment padding
        # behind a small payload (< 128 bytes, zero-filled), the last tensor at the file's end (FORMAT.md)
        off = data_start + t.offset
        end = data_start + ts[i + 1].offset if i + 1 < len(ts) else fsize
        if off % 128:
            fail("%s: %s payload at %d is not 128-aligned" % (path.name, t.name, off))
        region = end - off
        if region < m["payload_bytes"] or (i + 1 < len(ts) and region >= m["payload_bytes"] + 128) or \
                (i + 1 == len(ts) and region != m["payload_bytes"]):
            fail("%s: %s payload region %d bytes, manifest says %d (%s)" %
                 (path.name, t.name, region, m["payload_bytes"],
                  "overlap" if region < m["payload_bytes"] else "not at the file's end" if i + 1 == len(ts)
                  else "more than alignment padding"))
        if region > m["payload_bytes"]:
            gaps.append((off + m["payload_bytes"], region - m["payload_bytes"]))
        geom = ggr.BLOCK_GEOMETRY.get(ggr.GGML_TYPES.get(t.type_id, ""))
        if geom is not None:
            nel = 1
            for d in t.shape:
                nel *= d
            if nel % geom[0] or nel // geom[0] * geom[1] != m["payload_bytes"]:
                fail("%s: %s %d bytes are not whole %s blocks of the manifest's %d" %
                     (path.name, t.name, m["payload_bytes"], ggr.GGML_TYPES[t.type_id], m["payload_bytes"]))
        out.append([pack_name(t.name, role), dtype_of(t.type_id, m["stored_shape"], m["type"]),
                    off, m["payload_bytes"], list(m["stored_shape"])])
    if gaps:   # (the strict reader's law: zero padding)
        with path.open("rb") as f:
            for at, n in gaps:
                f.seek(at)
                if any(f.read(n)):
                    fail("%s: nonzero padding behind the payload at %d" % (path.name, at))
    return out


def build(release: pathlib.Path, out: pathlib.Path) -> dict:
    """The pack: mm3.txt + tensors.txt over the release's own files.  Returns the build's numbers."""
    rel = json.loads((release / "release.json").read_text(encoding="utf-8"))
    if rel.get("format") != FORMAT:
        fail("release.json: format %r, not %s" % (rel.get("format"), FORMAT))
    man = json.loads((release / "tensor-manifest.json").read_text(encoding="utf-8"))
    by_part = {}
    for t in man["tensors"]:
        by_part.setdefault(t["part"], {})[t["name"]] = t
    cfg = json.loads((release / "config.json").read_text(encoding="utf-8"))
    tc = cfg.get("text_config") or cfg
    bos, eos = token_ids(release)
    geo = geometry(tc, bos, eos)

    parts = sorted(rel["parts"], key=lambda p: (p["role"] != "trunk" or p.get("layer") is None,
                                                p.get("layer") if p.get("layer") is not None else 1 << 30,
                                                p["path"]))
    files, rows = [], []
    for part in parts:
        path = release / part["path"]
        if not path.is_file():
            fail("%s: not in the release" % path)
        gf = ggr.GGUFFile(path)
        files.append(part["path"])
        rows += [[name, dt, len(files) - 1, off, nb, shape]
                 for name, dt, off, nb, shape in check_part(part, path, by_part.get(part["path"], {}), gf)]
    empty = [p["path"] for p in parts if not by_part.get(p["path"])]
    if empty:
        fail("parts with no manifest rows: %s" % ", ".join(empty))
    unmapped = sorted(set(by_part) - {p["path"] for p in parts})
    if unmapped:
        fail("manifest parts the release does not list: %s" % ", ".join(unmapped))

    out.mkdir(parents=True, exist_ok=True)
    with (out / "mm3.txt").open("w", newline="\n") as f:
        for k in GEO_KEYS:
            v = geo[k]
            f.write("%s %s\n" % (k, ",".join(str(x) for x in v) if isinstance(v, list) else v))
        f.write("model_dir %s\n" % release)
        f.write("format vq\n")   # the engine's loader switch: the VQ expert path + the standard-type dense path
    with (out / "tensors.txt").open("w", newline="\n") as f:
        for i, name in enumerate(files):
            f.write("file %d %s\n" % (i, name))
        for name, dt, fi, off, nb, shape in rows:
            f.write("t %s %s %d %d %d %s\n" % (name, dt, fi, off, nb, " ".join(map(str, shape))))
    by_dt = {}
    for _, dt, _, _, nb, _ in rows:
        by_dt[dt] = by_dt.get(dt, 0) + nb
    return {"files": len(files), "rows": len(rows), "by_dtype": by_dt, "manifest": len(man["tensors"]),
            "r13p8": rel.get("r13p8_layers", []), "release": release}


# ---------------------------------------------------------------- validation

def read_pack(out: pathlib.Path) -> tuple:
    """The written pack back from disk, mm3_pack.hpp's parse: (kv dict, [(name, dtype, fi, off, bytes, shape)],
    [file names], error string or '')."""
    kv, tensors, files = {}, [], []
    for line in (out / "mm3.txt").read_text().splitlines():
        if " " in line:
            k, v = line.split(" ", 1)
            kv[k] = v
    for line in (out / "tensors.txt").read_text().splitlines():
        x = line.split()
        if not x:
            continue
        if x[0] == "file":
            files.append(x[2])
        elif x[0] == "t":
            tensors.append((x[1], x[2], int(x[3]), int(x[4]), int(x[5]), [int(d) for d in x[6:]]))
    return kv, tensors, files


def gv_header(path: pathlib.Path, off: int) -> tuple:
    """The 128-byte GVQ1/GVR8 header at a payload start, mapped: the parsed fields."""
    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return GV_HDR.unpack(mm[off:off + 128])
        finally:
            mm.close()


def validate(built: dict, out: pathlib.Path) -> list:
    """The five checks; each a (name, ok, detail) row for the report."""
    release, r13p8 = built["release"], set(built["r13p8"])
    kv, tensors, files = read_pack(out)
    rel = json.loads((release / "release.json").read_text(encoding="utf-8"))
    man = json.loads((release / "tensor-manifest.json").read_text(encoding="utf-8"))
    checks = []

    # 1. every row inside its file, a dtype the engine's classes claim, a shape
    bad = []
    sizes = [os.path.getsize(release / n) for n in files]
    seen = set()
    for name, dt, fi, off, nb, shape in tensors:
        if fi < 0 or fi >= len(files) or off + nb > sizes[fi]:
            bad.append("%s: outside its file" % name)
        if dt not in ALLOWED_DTYPES:
            bad.append("%s: dtype %s" % (name, dt))
        if not shape:
            bad.append("%s: empty shape" % name)
        if name in seen:
            bad.append("%s: duplicate row" % name)
        seen.add(name)
    checks.append(("rows: bounds + dtype + shape",
                   not bad, "%d/%d rows in bounds, %d dtypes in the allowed set, %d bad" %
                   (len(tensors) - len(bad), len(tensors), len(ALLOWED_DTYPES), len(bad))))

    # 2. the counts: 57 layer parts x (1 codebook + 384 matrices) + 20 x (1 + 1 + 384 + 384), body/embed/vision/dspark
    by_part = {}
    for name, dt, fi, off, nb, shape in tensors:
        by_part[files[fi]] = by_part.get(files[fi], 0) + 1
    man_by_part = {}
    for t in man["tensors"]:
        man_by_part[t["part"]] = man_by_part.get(t["part"], 0) + 1
    layer_parts = [p for p in rel["parts"] if p["role"] == "trunk" and p.get("layer") is not None]
    plain = [p for p in layer_parts if not p.get("residual")]
    res = [p for p in layer_parts if p.get("residual")]
    n_layer = sum(by_part.get(p["path"], 0) for p in layer_parts)
    n_other = sum(by_part.get(p["path"], 0) for p in rel["parts"] if p not in layer_parts)
    detail = ("37 plain layers x 385 = %d, 20 R13+8 layers x 770 = %d, body/embed/vision/dspark %d, "
              "total %d vs manifest %d" %
              (sum(by_part.get(p['path'], 0) for p in plain), sum(by_part.get(p['path'], 0) for p in res),
               n_other, len(tensors), built["manifest"]))
    ok = (len(layer_parts) == 57 and len(res) == 20 and len(plain) == 37 and
          all(by_part.get(p["path"]) == man_by_part.get(p["path"]) for p in rel["parts"]) and
          len(tensors) == built["manifest"] == man["tensor_count"])
    checks.append(("counts vs tensor-manifest.json", ok, detail))

    # 3. the R13+8 split: VQ13R8 on the 20, plain VQ13 on the other 37
    res_layers = {"layer-%02d-experts.gguf" % li for li in r13p8}
    n_r8 = sum(1 for name, dt, *_ in tensors if dt == "VQ13R8")
    n_v13 = sum(1 for name, dt, *_ in tensors if dt == "VQ13")
    wrong = []
    for p in layer_parts:
        dts = [dt for name, dt, fi, *_ in tensors if files[fi] == p["path"]]
        want_r8 = p["path"] in res_layers
        dtset = set(dts)
        if ("VQ13R8" in dtset) != want_r8 or "VQ13" not in dtset:
            wrong.append(p["path"])
        # every layer's 384 matrices are the 13-bit stage; the R13+8 layers carry their 8-bit residuals beside them
        if dts.count("VQ13") != 384 or dts.count("VQ13R8") != (384 if want_r8 else 0):
            wrong.append(p["path"])
        if dts.count("CB8192") != 1 or dts.count("CB256") != (1 if want_r8 else 0):
            wrong.append(p["path"])
    checks.append(("R13+8 split: VQ13R8 beside VQ13 on the 20, plain VQ13 on the 37",
                   not wrong and n_r8 == 20 * 384 and n_v13 == 57 * 384,
                   "%d VQ13 (57x384, every layer) + %d VQ13R8 (20x384, the R13+8 layers only), "
                   "%d CB8192 + %d CB256, %d misplaced" %
                   (n_v13, n_r8, sum(1 for _, dt, *_ in tensors if dt == "CB8192"),
                    sum(1 for _, dt, *_ in tensors if dt == "CB256"), len(wrong))))

    # 4. the GVQ1 header at the payload start, three sampled matrices (+ one GVR8)
    hdr_bad = []
    samples = [("layer-03-experts.gguf", "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight"),
               ("%s" % sorted(res_layers)[0], None), ("layer-59-experts.gguf",
                                                      "language_model.model.layers.59.block_sparse_moe.experts.127.w2.weight")]
    for fn, want_name in samples:
        rows_f = [t for t in tensors if files[t[2]] == fn and t[1] == "VQ13"]
        row = next((t for t in rows_f if want_name is None or t[0] == want_name), None)
        if row is None:
            hdr_bad.append("%s: no VQ13 sample" % fn)
            continue
        magic, ver, bits, vd, R, C, K, prec, cbid, flags, goff, ioff, hoff, total, cbs, bs = \
            gv_header(release / fn, row[3])
        if magic != b"GVQ1" or ver != 1 or bits != 13 or vd != 8 or K != 8192 or prec != 1 or goff != 128 or \
                total != row[4] or total > sizes[row[2]] - row[3]:
            hdr_bad.append("%s %s: magic %r ver %d bits %d K %d total %d vs row %d" %
                           (fn, row[0], magic, ver, bits, K, total, row[4]))
    r8row = next((t for t in tensors if t[1] == "VQ13R8" and files[t[2]] == sorted(res_layers)[0]), None)
    if r8row is not None:
        magic, ver, bits, vd, R, C, K, prec, cbid, flags, goff, ioff, hoff, total, cbs, bs = \
            gv_header(release / files[r8row[2]], r8row[3])
        if magic != b"GVR1" or ver != 1 or bits != 8 or vd != 8 or K != 256 or hoff != 0 or total != r8row[4]:
            hdr_bad.append("%s %s: GVR8 magic %r bits %d K %d total %d vs row %d" %
                           (files[r8row[2]], r8row[0], magic, bits, K, total, r8row[4]))
    checks.append(("GVQ1 headers at payload start",
                   not hdr_bad, "3 GVQ1 samples (layer-03 e0 w1, %s e0 w1, layer-59 e127 w2) + 1 GVR8: magic, "
                                "index_bits 13, entries 8192, total_bytes == payload_bytes; %d bad" %
                   (sorted(res_layers)[0], len(hdr_bad))))

    # 5. the geometry: every Geo key in mm3.txt, Geo::read's arithmetic against what it refuses
    missing = [k for k in GEO_KEYS if k not in kv]
    n = lambda k, d=0.0: float(kv.get(k, d))  # noqa: E731
    hd, nh, nkv, nrot = int(n("head_dim")), int(n("num_attention_heads")), int(n("num_key_value_heads")), int(n("rotary_dim"))
    nl = int(n("num_hidden_layers"))
    ixh, ixd, blk, topk = int(n("sparse_attention_config.sparse_num_index_heads")), int(n("sparse_attention_config.sparse_index_dim")), int(n("sparse_attention_config.sparse_block_size")), int(n("sparse_attention_config.sparse_topk_blocks"))
    local, init = int(n("sparse_attention_config.sparse_local_block")), int(n("sparse_attention_config.sparse_init_block"))
    moe = [int(x) for x in kv.get("moe_layer_freq", "").split(",") if x]
    sparse = [int(x) for x in kv.get("sparse_attention_config.sparse_attention_freq", "").split(",") if x]
    geo_bad = []
    if hd != 128 or nrot % 2 or nrot > hd: geo_bad.append("head_dim/rotary_dim")
    if nh != 16 * nkv: geo_bad.append("q heads per kv head")
    if ixh != nkv or ixh != 4 or ixd != 128: geo_bad.append("indexer heads/dim")
    if blk != 128 or local != 1 or not 1 <= topk <= 64 or init != 0: geo_bad.append("sparse blocks")
    if len(moe) != nl or len(sparse) != nl or any(not s for s in sparse): geo_bad.append("per-layer lists")
    if abs(n("swiglu_alpha") - 1.702) > 1e-6: geo_bad.append("swiglu_alpha")
    if int(n("num_experts_per_tok")) > 8 or int(n("num_local_experts")) > 256: geo_bad.append("experts")
    if any(int(n(k)) % 128 for k in ("hidden_size", "intermediate_size", "dense_intermediate_size",
                                     "shared_intermediate_size")): geo_bad.append("matrix sides")
    if kv.get("model_dir") != str(release): geo_bad.append("model_dir")
    checks.append(("geometry: every Geo key + Geo::read arithmetic",
                   not missing and not geo_bad,
                   "%d/%d keys present (bos %d eos %d), hd %d n_head %d = 16x%d n_kv, ix_heads %d ix_dim %d, "
                   "blk %d topk %d local %d, moe %d sparse %d of %d layers, dim/ff/ff_dense/ff_shared %d/%d/%d/%d "
                   "all %%128==0; %s" %
                   (len(GEO_KEYS) - len(missing), len(GEO_KEYS), int(n("bos_token_id")), int(n("eos_token_id")),
                    hd, nh, nkv, ixh, ixd, blk, topk, local, len(moe), len(sparse), nl,
                    int(n("hidden_size")), int(n("intermediate_size")), int(n("dense_intermediate_size")),
                    int(n("shared_intermediate_size")), ", ".join(geo_bad) or "no refusals")))
    return checks


def companions(release: pathlib.Path, out: pathlib.Path) -> str:
    """The DSpark drafter's pack wiring.  The draft's tensors are already indexed by build() (the release
    manifest carries the drafter part; its tensor names are the loader's own `dspark.` namespace) - what
    the pack lacks is the geometry keys in mm3.txt (the loader refuses --dspark without
    dspark.num_hidden_layers).  This step reads the drafter part's own dspark_config metadata and writes
    those keys.  Idempotent.  Returns a report line."""
    mm3 = out / "mm3.txt"
    have = mm3.read_text(encoding="utf-8")
    if "dspark.num_hidden_layers" in have:
        return "companions: the pack already carries the draft geometry"
    drafter = None
    for p in sorted(release.glob("*.gguf")):
        if not p.is_file():
            continue
        try:
            with p.open("rb") as f:
                magic, ver, nt, nkv = struct.unpack("<4sIQQ", f.read(24))
                if magic != b"GGUF":
                    continue
                def rds():
                    n = struct.unpack("<Q", f.read(8))[0]
                    return f.read(n).decode("utf-8", "replace")
                role = None
                for _ in range(nkv):
                    k = rds()
                    t = struct.unpack("<I", f.read(4))[0]
                    if t == 8:
                        v = rds()
                        if k == "minimax_m3_vq.part_role":
                            role = v
                    elif t == 9:
                        et = struct.unpack("<I", f.read(4))[0]
                        n = struct.unpack("<Q", f.read(8))[0]
                        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(et)
                        if sz is None:
                            for _ in range(n):
                                rds()
                        else:
                            f.seek(n * sz, 1)
                    else:
                        f.seek({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(t, 4), 1)
                if role == "drafter":
                    drafter = p
                    break
        except (OSError, struct.error):
            continue
    if drafter is None:
        return "companions: no drafter part in the release (DSpark stays off)"
    with drafter.open("rb") as f:
        magic, ver, nt, nkv = struct.unpack("<4sIQQ", f.read(24))
        def rds():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")
        SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        geo = {}
        for _ in range(nkv):
            k = rds()
            t = struct.unpack("<I", f.read(4))[0]
            if t == 8:
                v = rds()
                if k.endswith("dspark_config"):
                    c = json.loads(v)
                    geo = {"dspark.num_hidden_layers": c["num_hidden_layers"],
                           "dspark.dspark_ffn_hidden_size": c["dspark_ffn_hidden_size"],
                           "dspark.dspark_markov_rank": c["dspark_markov_rank"],
                           "dspark.dspark_block_size": c["dspark_block_size"],
                           "dspark.dspark_noise_token_id": c["dspark_noise_token_id"],
                           "dspark.dspark_target_layer_ids": ",".join(str(x) for x in c["dspark_target_layer_ids"]),
                           "dspark.dflash.sliding_window": c.get("dflash_config", {}).get("sliding_window", 4096)}
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                sz = SZ.get(et)
                if sz is None:
                    for _ in range(n):
                        rds()
                else:
                    f.seek(n * sz, 1)
            else:
                f.seek(SZ.get(t, 4), 1)
    if not geo:
        return "companions: %s carries no dspark config" % drafter.name
    with mm3.open("a", encoding="utf-8") as f:
        for k, v in geo.items():
            f.write("%s %s\n" % (k, v))
    return ("companions: the draft's geometry written from %s (%d layers, taps %s; its tensors were "
            "already indexed by the build)" % (drafter.name, geo["dspark.num_hidden_layers"],
                                               geo["dspark.dspark_target_layer_ids"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="the VQ release's directory (release.json + tensor-manifest.json)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="default: %s" % DEFAULT_OUT)
    a = ap.parse_args()
    release = pathlib.Path(a.release).expanduser().resolve()
    if not (release / "release.json").is_file():
        fail("%s: no release.json" % release)
    out = pathlib.Path(a.out).expanduser()
    if out.resolve() == release:
        fail("--out inside the release would write into it; the pack references the release in place")
    built = build(release, out)
    print(companions(release, out))
    print("%s -> %s: %d tensors in %d files (%s)" %
          (release.name, out, built["rows"], built["files"],
           ", ".join("%s %.2f GB" % (k, v / 1e9) for k, v in sorted(built["by_dtype"].items()))))
    ok = True
    print("\nvalidation (the pack's own strict-reader law):")
    for name, good, detail in validate(built, out):
        ok &= good
        print("  [%s] %s\n         %s" % ("PASS" if good else "FAIL", name, detail))
    print("\n%s" % ("all 5 checks pass" if ok else "VALIDATION FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
