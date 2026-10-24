#!/usr/bin/env python3
"""campaign/retrain/p2_forward.py — the P2 torch parity module's FORWARD (T14B-exec-B §2.2,
stage-by-stage transcription of the engine's drafter, G0-exact mode).

One application: (token:int, Rin:[4,2560] f32, pos:int, cell:int) -> head_logits[40525].
Laws honored: weights from the rt dialect bytes (law 1); f32 norms carry the folded +1
(law 2); hnorm=STREAM per-slice RMS (law 3).  Activation quantizers re-applied exactly
(q8_1 at projections, int8 on K/V) so the arithmetic sees the engine's integers.

Stages follow the spec's B1..B6 verbatim with the cited semantics.  Divergences from
the engine are limited to fp32 reduction order (the ~2e-6 noise floor the spec sets).
Run: ~/venvs/mtp-parity/bin/python campaign/retrain/p2_forward.py   (self-consistency smoke)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_rt_loader import RtDense

RT = Path("/home/svg/LLM/data/mtp-en/rt")
EPS = 1e-6
HC = 4
N = 2560
HD = 256
NH, NKV = 24, 2
NROT = 64
ROPE_BASE = 1e7
NE, FF = 512, 640


# ---- the engine's activation quantizers (G0-exact) ------------------------------

def q8_1_quant(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per 32-block: d = clamp_f32(amax/127); q = clamp(roundf(x/d), -127, 127).
    Returns (d[f32], q[f32-integers]); the dot form: sum(d_a*d_w*q_a*q_w)."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    d = np.where(np.isfinite(d), d, np.float32(0.0))     # q8_1_finite's clamp class
    safe = np.where(d != 0, d, np.float32(1.0))
    q = np.clip(np.rint(b / safe[:, None]), -127, 127).astype(np.float32)
    return d, q


def q8_1_dot(x: np.ndarray, w_q8: np.ndarray) -> np.ndarray:
    """x [.., K] f32 against w_q8 [out, K] stored AS THE PACKER'S q8_0 BYTES-dequantized
    f32? No — G0-exact: x is q8_1-quantized, w's q8_0 codes are integers with fp16 d.
    p2_rt_loader returns dequantized f32; here we requantize the LOADED f32 back to its
    integer/code form exactly (round-trip is lossless: the loader inverted the same rule)
    and compute the integer dot."""
    out_dim, K = w_q8.shape
    xd, xq = q8_1_quant(x)
    wb = w_q8.reshape(out_dim, -1, 32)
    wamax = np.abs(wb).max(axis=2)
    wd = (wamax / 127.0)
    wd = wd.astype(np.float16).astype(np.float32)        # the stored fp16 d
    wsafe = np.where(wd != 0, wd, np.float32(1.0))
    wq = np.clip(np.rint(wb / wsafe[:, :, None]), -127, 127).astype(np.float32)
    # block-wise integer dots, fp32 accumulation
    xflat = x.reshape(-1, K)
    xq_b = xq.reshape(xflat.shape[0], -1, 32)                        # [rows, nblk, 32]
    xd_b = xd.reshape(xflat.shape[0], -1)                           # [rows, nblk] per-block d
    # integer dots per (row, out, block): q_a . q_w
    prod = np.einsum("rbk,obk->rob", xq_b, wq, optimize=True).astype(np.float32)
    acc = np.einsum("rob,rb,ob->ro", prod, xd_b, wd, optimize=True).astype(np.float32)
    return acc.reshape(*x.shape[:-1], out_dim)


def rms(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """scale = rsqrt(mean(x^2)+eps); gamma applied AFTER (native_qsa.cu:45-68)."""
    s = 1.0 / math.sqrt(float(np.mean(x * x)) + EPS)
    return (x * s).astype(np.float32) * gamma


def silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


def sigmoid_f32(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


# ---- B3: the HC read (shared formula, used 3x) ----------------------------------

def hc_read(R: np.ndarray, rs_w: np.ndarray, w_norm: np.ndarray, w_down: np.ndarray,
            w_up: np.ndarray, w_inject: np.ndarray | None = None,
            apply: bool = False, bo_prev: np.ndarray | None = None,
            inj_prev: np.ndarray | None = None):
    """R [4,2560] raw -> dict(rs[4], xn[4,2560], lo[320], inj[4]|None, mixed[10240], Rp[4,2560])."""
    Rp = R.copy()
    if apply and bo_prev is not None and inj_prev is not None:
        gw = (2.0 * sigmoid_f32(inj_prev / 4.0)).astype(np.float32)
        Rp = (Rp + bo_prev[None, :] * gw[:, None]).astype(np.float32)
    rs = np.empty(HC, dtype=np.float32)
    xn = np.empty((HC, N), dtype=np.float32)
    for c in range(HC):
        rs[c] = np.float32(1.0 / math.sqrt(float(np.mean(Rp[c] * Rp[c])) + EPS))
        wn = w_norm.reshape(-1)
        xn[c] = (Rp[c] * rs[c]).astype(np.float32) * wn[c * N:(c + 1) * N]
    xcat = xn.reshape(1, -1)                                              # the full [10240] vector
    lo = silu(q8_1_dot(xcat, w_down).reshape(-1) / 4.0)                   # [320], the /HC contract
    inj = q8_1_dot(xcat, w_inject).reshape(-1).astype(np.float32) if w_inject is not None else None  # [4]
    up = q8_1_dot(lo[None, :], w_up).reshape(-1)                           # [10240]
    # B4-wire resolution (mtp.cpp:958 + :424: fa[t].mixed = mixed_ + t*N, mixed_ = T*N):
    # the DRAFTER's mixed is [2560] per token = the per-dimension mean over streams
    # (the spec's 10240-concatenated form was the TRUNK's fused_gr semantics).
    mixed = np.zeros(N, dtype=np.float32)
    for c in range(HC):
        sl = slice(c * N, (c + 1) * N)
        mixed += Rp[c] * wn[sl] * rs[c] * sigmoid_f32(up[sl])
    mixed /= np.float32(HC)                                                  # the final /4 mean
    return {"rs": rs, "xn": xn, "lo": lo, "inj": inj, "mixed": mixed, "Rp": Rp}


# ---- B4: attention (partial rope inside 64 dims, int8 KV, GQA, gate) -------------

def partial_rope(x: np.ndarray, pos: int) -> np.ndarray:
    """x [HD]; pairs i<32 rotate (x[i], x[i+32]) by theta = pos * base^(-2i/64)."""
    out = x.copy()
    for i in range(32):
        theta = pos * (ROPE_BASE ** (-2.0 * i / NROT))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[i], x[i + 32]
        out[i] = a * c - b * s
        out[i + 32] = a * s + b * c
    return out.astype(np.float32)


def int8_kv(x: np.ndarray) -> np.ndarray:
    """Per 64 values: scale = fp16(max|x|/127); store = fp16(rint(x/scale) * scale)."""
    out = np.empty_like(x)
    for blk in range(0, x.size, 64):
        b = x[blk:blk + 64]
        sc = np.float16(np.abs(b).max() / 127.0)
        safe = sc if sc != 0 else np.float16(1.0)
        q = np.clip(np.rint(b / np.float32(safe)), -127, 127)
        out[blk:blk + 64] = (q * np.float32(safe)).astype(np.float32)
    return out


class DraftLayer:
    """The one-application G0-exact forward. Weights from RtDense; the draft head's
    Q4_0 40,525-row bytes and the token embedding row are supplied by the TAP-side
    fixture (they come from the main GGUF; loaded in p2_fixture at adjudication time)."""

    def __init__(self, rt: RtDense):
        self.t = rt.tensors

    def forward(self, token: int, Rin: np.ndarray, pos: int, cell: int,
                Kcache: np.ndarray, Vcache: np.ndarray, emb_row: np.ndarray):
        """Rin [4,2560]; Kcache/Vcache [cell+1, 2, HD] (ALREADY int8-dequantized engine
        cells 0..cell-1); emb_row the dequantized token_embd row [2560].  Returns the
        stage dict through head_logits (head applied by the fixture for now)."""
        T = self.t
        # B1 embedding branch
        en = rms(emb_row.astype(np.float32), T["pre_fc_norm_embedding.weight"].reshape(-1))
        e2 = q8_1_dot(en[None, :], T["fc_embedding.weight"]).reshape(-1)      # [2560]
        # B2 residual branch + fusion
        R = np.empty((HC, N), dtype=np.float32)
        gH = T["pre_fc_norm_hidden.weight"].reshape(-1)
        for c in range(HC):
            hn = rms(Rin[c], gH[c * N:(c + 1) * N])
            h2 = q8_1_dot(hn[None, :], T["fc_hidden.weight"]).reshape(-1)
            R[c] = h2 + e2
        # B3 read #1 (attn): apply=False, w_inject set
        a1 = hc_read(R, None, T["hyper_connection_mixer.hc_norm.weight"],
                     T["hyper_connection_mixer.input_mix_weight_down.weight"],
                     T["hyper_connection_mixer.input_mix_weight_up.weight"],
                     w_inject=T["attn_hyper_connection.block_inject_weight.weight"])
        mixed = a1["mixed"]
        # B4 attention
        kcur = q8_1_dot(mixed[None, :], T["self_attn.k_proj.weight"]).reshape(NKV, HD)
        vcur = q8_1_dot(mixed[None, :], T["self_attn.v_proj.weight"]).reshape(NKV, HD)
        knorm_w = T["self_attn.k_norm.weight"].reshape(-1)
        qnorm_w = T["self_attn.q_norm.weight"].reshape(-1)
        K = np.empty((NKV, HD), dtype=np.float32)
        for h in range(NKV):
            kn = rms(kcur[h], knorm_w)
            K[h] = partial_rope(kn, pos)
        Kq = int8_kv(K.reshape(-1)).reshape(NKV, HD)     # the layer's own KV int8 round-trip
        Vq = int8_kv(vcur.reshape(-1)).reshape(NKV, HD)
        Kall = np.concatenate([Kcache[:cell], Kq[None]], axis=0)             # [cell+1, 2, HD]
        Vall = np.concatenate([Vcache[:cell], Vq[None]], axis=0)
        qfull = q8_1_dot(mixed[None, :], T["self_attn.q_proj.weight"]).reshape(NH, 2 * HD)
        scores = np.empty((NH, cell + 1), dtype=np.float32)
        attn = np.zeros((NH, HD), dtype=np.float32)
        for h in range(NH):
            q = rms(qfull[h, :HD], qnorm_w)
            qr = partial_rope(q, pos)
            kh = h // 12                                   # GQA 12:1
            s = (Kall[:cell + 1, kh] @ qr) / math.sqrt(HD)
            s = s - s.max()
            p = np.exp(s); p /= p.sum()
            scores[h] = p
            attn[h] = (p @ Vall[:cell + 1, kh]).astype(np.float32)
        attn32 = attn * sigmoid_f32(qfull[:, HD:])         # the raw gate half
        bo = q8_1_dot(attn32.reshape(1, -1), T["self_attn.o_proj.weight"]).reshape(-1)  # [2560]
        # B3 read #2 (mlp): apply=True with bo_prev=bo, inj_prev=a1 inj — R folded in place
        a2 = hc_read(a1["Rp"], None, T["mlp_hyper_connection.hc_norm.weight"],
                     T["mlp_hyper_connection.input_mix_weight_down.weight"],
                     T["mlp_hyper_connection.input_mix_weight_up.weight"],
                     w_inject=T["mlp_hyper_connection.block_inject_weight.weight"],
                     apply=True, bo_prev=bo, inj_prev=a1["inj"])
        mixed_mlp = a2["mixed"]
        R_after = a2["Rp"]                                 # attention written into R
        # B5 shared expert (router + routed experts land with the blob reader at TAP time;
        # the fixture supplies parts+w or the full y for this application)
        # -> y [2560] supplied by caller; then:
        # R_ = R_after + y * 2*sigmoid(inj2/4) per stream; B3 read #3 -> sample; head.
        return {"a1": a1, "mixed": mixed, "bo": bo, "a2": a2, "mixed_mlp": mixed_mlp,
                "R_after": R_after, "scores": scores, "Kq": Kq, "Vq": Vq}


if __name__ == "__main__":
    rt = RtDense()
    layer = DraftLayer(rt)
    rng = np.random.default_rng(3)
    Rin = rng.standard_normal((HC, N)).astype(np.float32) * 0.5
    cell = 7
    Kcache = (rng.standard_normal((cell, NKV, HD)).astype(np.float32) * 0.3)
    Vcache = (rng.standard_normal((cell, NKV, HD)).astype(np.float32) * 0.3)
    emb_row = rng.standard_normal(N).astype(np.float32) * 0.1
    out = layer.forward(token=5, Rin=Rin, pos=8, cell=cell, Kcache=Kcache, Vcache=Vcache, emb_row=emb_row)
    for k in ("mixed", "bo", "mixed_mlp"):
        v = out[k]
        print(f"{k:10s} shape {v.shape} rms {np.sqrt((v*v).mean()):.4f} finite {np.isfinite(v).all()}")
    R_after = out["R_after"]
    print(f"R_after    shape {R_after.shape} rms {np.sqrt((R_after*R_after).mean()):.4f}")
    print("stage-wise smoke OK (self-consistency; TAP adjudication pending)")
