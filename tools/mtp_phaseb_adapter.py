#!/usr/bin/env python
"""F-52 adapter: extract the trained phaseB MTP head (buun-sidecar GGUF) into the
mtp_fetch.py layout (BF16 .bin + manifest) for tools/mtp_pack.py.
Key semantics (series-192 0001): eh_proj = fc_embedding + fc_hidden fused over
concat(e, h) -> SPLIT: fc_embedding = eh_proj[:, :2560], fc_hidden = eh_proj[:, 2560:].
enorm -> pre_fc_norm_embedding; hnorm (10240) -> pre_fc_norm_hidden.
Usage: python mtp_phaseb_adapter.py <phaseB.gguf> <outdir>"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py
add_gguf_py()
import gguf

src, outp = sys.argv[1], Path(sys.argv[2])
outp.mkdir(parents=True, exist_ok=True)

# buun-sidecar name -> (canonical name, transform)
# transforms: None = as-is; ('split', a, b) = slice rows [a:b] of axis 0
M = {
    "nextn.eh_proj.weight":      ("fc_embedding.weight", ("split", 0, 2560)),
                                 # second half handled below via dup entry
    "nextn.enorm.weight":        ("pre_fc_norm_embedding.weight", None),
    "nextn.hnorm.weight":        ("pre_fc_norm_hidden.weight", None),
    "nextn.hc_head_norm.weight": ("hyper_connection_mixer.hc_norm.weight", None),
    "nextn.hc_head_up.weight":   ("hyper_connection_mixer.input_mix_weight_up.weight", None),
    "nextn.hc_head_down.weight": ("hyper_connection_mixer.input_mix_weight_down.weight", None),
    "attn_q.weight":     ("layers.0.self_attn.q_proj.weight", None),
    "attn_k.weight":     ("layers.0.self_attn.k_proj.weight", None),
    "attn_v.weight":     ("layers.0.self_attn.v_proj.weight", None),
    "attn_output.weight": ("layers.0.self_attn.o_proj.weight", None),
    "attn_q_norm.weight": ("layers.0.self_attn.q_norm.weight", None),
    "attn_k_norm.weight": ("layers.0.self_attn.k_norm.weight", None),
    "indexer.q_norm.weight": ("layers.0.self_attn.indexer.q_layernorm.weight", None),
    "indexer.k_norm.weight": ("layers.0.self_attn.indexer.k_layernorm.weight", None),
    "hc_attn_norm.weight":   ("layers.0.attn_hyper_connection.hc_norm.weight", None),
    "hc_attn_down.weight":   ("layers.0.attn_hyper_connection.input_mix_weight_down.weight", None),
    "hc_attn_up.weight":     ("layers.0.attn_hyper_connection.input_mix_weight_up.weight", None),
    "hc_attn_inject.weight": ("layers.0.attn_hyper_connection.block_inject_weight.weight", None),
    "hc_ffn_norm.weight":    ("layers.0.mlp_hyper_connection.hc_norm.weight", None),
    "hc_ffn_down.weight":    ("layers.0.mlp_hyper_connection.input_mix_weight_down.weight", None),
    "hc_ffn_up.weight":      ("layers.0.mlp_hyper_connection.input_mix_weight_up.weight", None),
    "hc_ffn_inject.weight":  ("layers.0.mlp_hyper_connection.block_inject_weight.weight", None),
    "ffn_gate_shexp.weight": ("layers.0.mlp.shared_expert.gate_proj.weight", None),
    "ffn_up_shexp.weight":   ("layers.0.mlp.shared_expert.up_proj.weight", None),
    "ffn_down_shexp.weight": ("layers.0.mlp.shared_expert.down_proj.weight", None),
    "ffn_gate_inp.weight":     ("layers.0.mlp.gate.weight", None),
    "ffn_gate_inp_shexp.weight": ("layers.0.mlp.shared_expert_gate.weight", None),
    "ffn_gate_exps.weight": ("layers.0.mlp.experts.gate_up_proj", ("fuse_gu",)),
    "ffn_up_exps.weight":   ("layers.0.mlp.experts.gate_up_proj", ("fuse_gu",)),
    "ffn_down_exps.weight": ("layers.0.mlp.experts.down_proj", None),
}
PREFIX = "mtp."

r = gguf.GGUFReader(src)
tensors = {t.name: np.asarray(t.data) for t in r.tensors}
# strip any dialect prefix the bridge added (we read the ORIGINAL phaseB names)
def find(name):
    for k in tensors:
        if k.endswith(name):
            return tensors[k]
    raise KeyError(name)

manifest = []
gate = find("ffn_gate_exps.weight").astype(np.float32)   # (2560,640,512) rows=in? -> transpose to out-major
up   = find("ffn_up_exps.weight").astype(np.float32)
down = find("ffn_down_exps.weight").astype(np.float32)   # (640,2560,512)
# canonical fused: gate_up (512, 1280, 2560): expert e rows [gate(640); up(640)] x in(2560)
def as_out_major(t, n_out):  # gguf (n_out?, ..., ne) -> we trust ggml row-major: shape (in, out, e) or (out, in, e)?
    # reference: ffn_gate_exps (2560,640,512): dim0=2560=in -> GGUF ne[0]=in. Per-expert matrix = (640_out, 2560_in) after T? ne order: ne0 fastest.
    # We need per-expert (out, in): take t[:,:,e].T when shape==(in,out,e).
    e_idx = t.shape[2]
    assert t.shape[0] in (2560,) and t.shape[1] in (640,), t.shape
    return np.ascontiguousarray(np.transpose(t[:, :, i]) for i in ()) if False else t  # placeholder

# Simpler: build fused per expert directly with explicit transpose
NE = 512
# phaseB tensors are EXPERT-MAJOR: (512, out, in). Handle both layouts defensively.
def per_expert(t, n_in):
    if t.ndim == 3 and t.shape[0] == NE and t.shape[2] == n_in:
        return t  # already (e, out, in)
    if t.ndim == 3 and t.shape[2] == NE:  # (in, out, e) gguf order
        return np.ascontiguousarray(np.transpose(t, (2, 1, 0)))
    raise ValueError(t.shape)
gate_p = per_expert(gate, 2560)   # (512, 640, 2560)
up_p   = per_expert(up, 2560)
down_p = per_expert(down, 640)    # (512, 2560, 640)
gu = np.empty((NE, 1280, 2560), dtype=np.float32)
gu[:, :640] = gate_p
gu[:, 640:] = up_p
dn = np.ascontiguousarray(down_p)
for name, arr in [("mtp.layers.0.mlp.experts.gate_up_proj", gu), ("mtp.layers.0.mlp.experts.down_proj", dn)]:
    f = outp / (name.replace("mtp.", "").replace(".", "_") + ".bin")
    (arr.astype(np.float32).view(np.uint32) << 0).astype(np.float32)  # noop guard
    bf = arr.astype(np.float32).astype(np.bfloat16) if hasattr(np, "bfloat16") else None
    if bf is None:
        u32 = (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)  # truncate-to-bf16
    else:
        u32 = bf.view(np.uint16)
    u32.tofile(f)
    manifest.append({"name": name, "shape": list(arr.shape), "file": f.name,
                     "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
# eh_proj split FIRST (fc_e/fc_h needed by the write loop below)
eh = find("nextn.eh_proj.weight").astype(np.float32)
print("eh_proj shape:", eh.shape)
# torch Linear(2*N_EMBD, N_EMBD): weight (2560 out, 5120 in); the 5120 axis is the INPUT axis.
if eh.shape[1] == 5120:
    fc_e, fc_h = eh[:, :2560], eh[:, 2560:]
elif eh.shape[0] == 5120:
    w = eh.T
    fc_e, fc_h = w[:, :2560], w[:, 2560:]
else:
    raise ValueError(f"eh_proj unexpected shape {eh.shape}")

# The standing rt's dense.txt is the ground truth for (rows, cols) per tensor:
STANDING = {}
for ln in Path(r"E:\LLM\Engine\strata\data\mtp\rt\dense.txt").read_text().splitlines():
    p = ln.split()
    if len(p) >= 4:
        STANDING[p[0]] = (int(p[2]), int(p[3]))

def write_dense(canon, a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    want = STANDING.get(canon)
    norm_like = (a.ndim == 1 or (a.ndim == 2 and a.shape[0] == 1)) and "shared_expert_gate" not in canon
    if "shared_expert_gate" in canon and a.ndim == 1:
        a = a.reshape(1, -1)      # the exception: rank-2 (1,n) manifest -> BF16 output
    if want and norm_like and want[0] == 1:
        a = a.reshape(-1)          # keep RANK 1 in the manifest: the pack maps rank-1 -> F32
    if want and not norm_like and a.ndim == 2 and tuple(a.shape) != want and tuple(a.shape[::-1]) == want:
        a = np.ascontiguousarray(a.T)
    if want:
        have = tuple(a.shape) if not norm_like else (1, a.shape[0])
        if tuple(have) != want:
            raise ValueError(f"{canon}: shape {have} != standing {want}")
    f = outp / (canon.replace(".", "_") + ".bin")
    ((a.view(np.uint32) >> 16).astype(np.uint16)).tofile(f)   # ALL files BF16 halves (the pack's contract)
    manifest.append({"name": PREFIX + canon, "shape": list(a.shape), "file": f.name,
                     "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})

for src_name, (canon, tf) in M.items():
    if "fuse_gu" in str(tf) or canon.startswith("fc_embedding") or "experts." in canon:
        continue
    write_dense(canon, find(src_name))
write_dense("fc_embedding.weight", fc_e)
write_dense("fc_hidden.weight", fc_h)
# The phaseB indexer was trained on the 5120-wide eh-concat input (architectural divergence from
# the stock graph) - the STOCK indexer projections (2560-wide) come from the reference GGUF instead.
REF = gguf.GGUFReader(r"E:\LLM\Models\GGUF\MTPs\mtp-Qwen3.8-Flash-Next-shared-BF16.gguf")
ref_t = {t.name: np.asarray(t.data).astype(np.float32) for t in REF.tensors}
iq = ref_t["blk.48.indexer.q_proj.weight"]
ik = ref_t["blk.48.indexer.k_proj.weight"]
print("indexer (STOCK ref) q:", iq.shape, "k:", ik.shape)
iqk = np.ascontiguousarray(np.concatenate([iq, ik], axis=0))
print("index_qk fused:", iqk.shape)
write_dense("layers.0.self_attn.indexer.index_qk_proj.weight", iqk)

(outp / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1))
print("WROTE", len(manifest), "tensors to", outp)
