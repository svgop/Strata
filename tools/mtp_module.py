#!/usr/bin/env python3
"""tools/mtp_module.py — the G0-proven drafter forward as TRAINABLE torch ops
(the T14B-exec-C two-mode module; transcription of campaign/retrain/p2_forward.py,
which passed the G0 gate at F-319: T1 std 4.87e-7, T2 max_rel 2.73e-6).

Two modes per exec-C §2.2-2.4:
  eval  = the engine's quantizers live (q8_1 at q8/q4-weight projections; the affine
          correction) — value-parity with the serve path;
  train = the quantizers become pass-through with straight-through hooks (no rounding
          in the graph; gradients flow through the raw weights).

One application: (Rin [4,2560] fp32, next_tokens [K]) -> sub-vocab logits per step.
The B4 wire = the per-dimension stream mean (F-303); bf16-class weights consume
UNQUANTIZED fp32 activations (F-305's law); the /HC and /4 contracts inline.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

HC, N = 4, 2560
EPS = 1e-6


class MTPDraft(torch.nn.Module):
    """Trainable view over the rt dense tensors (a dict of Parameters, same names as
    p2_rt_loader).  Experts + head frozen (the F-114 shipping config)."""

    def __init__(self, tensors: dict, mode: str = "train"):
        super().__init__()
        self.mode = mode
        for k, v in tensors.items():
            p = torch.nn.Parameter(v.detach().clone().float(), requires_grad=True)
            self.register_parameter(k.replace(".", "_"), p)
        self.t = {k: getattr(self, k.replace(".", "_")) for k in tensors}

    def _dot(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """A projection dot. eval-mode with q8/q4-class weight QUANTIZATION is only
        needed for value parity vs the engine; in BOTH modes here the weights are the
        dequantized f32 tensors and the dot is plain fp32 — the TRAIN graph.  (The
        eval-quantized path lives in p2_forward.py for G0; this module trains.)"""
        return x @ w.T

    def rms(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        s = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
        return x * s * gamma

    def hc_read(self, R, w_norm, w_down, w_up, w_inject=None, apply=False,
                bo_prev=None, inj_prev=None):
        wn = w_norm.reshape(-1)
        Rp = R
        if apply and bo_prev is not None and inj_prev is not None:
            gw = 2.0 * torch.sigmoid(inj_prev / 4.0)
            Rp = Rp + bo_prev.unsqueeze(0) * gw.unsqueeze(1)
        rs = torch.rsqrt(Rp.pow(2).mean(-1, keepdim=True) + EPS)          # [4,1]
        xn = Rp * rs * wn.reshape(HC, N)                                    # [4,2560]
        xcat = xn.reshape(1, -1)                                            # [1,10240]
        lo = F.silu(self._dot(xcat, w_down).reshape(-1) / 4.0)              # [320]
        inj = self._dot(xcat, w_inject).reshape(-1) if w_inject is not None else None
        up = self._dot(lo.unsqueeze(0), w_up).reshape(-1)                   # [10240]
        mixed = (Rp * wn.reshape(HC, N) * rs * torch.sigmoid(up.reshape(HC, N))).mean(0)  # [2560] (the /4)
        return {"rs": rs, "xn": xn, "lo": lo, "inj": inj, "mixed": mixed, "Rp": Rp}

    def forward(self, Rin: torch.Tensor) -> torch.Tensor:
        """Rin [4,2560] -> the final mixer's sample [2560] (B1..B6 minus attention's
        KV, which the first cycle prices through the mixer chain; the K=6 recursion
        consumes successive Rin rows from the harvest — teacher-forced, so no KV
        state is needed for the loss)."""
        t = self.t
        en = self.rms(Rin[0], t["pre_fc_norm_embedding.weight"].reshape(-1))
        e2 = self._dot(en.unsqueeze(0), t["fc_embedding.weight"]).reshape(-1)
        gH = t["pre_fc_norm_hidden.weight"].reshape(-1)
        R = torch.empty_like(Rin)
        for c in range(HC):
            hn = self.rms(Rin[c], gH[c * N:(c + 1) * N])
            R[c] = self._dot(hn.unsqueeze(0), t["fc_hidden.weight"]).reshape(-1) + e2
        a1 = self.hc_read(R, t["hyper_connection_mixer.hc_norm.weight"],
                          t["hyper_connection_mixer.input_mix_weight_down.weight"],
                          t["hyper_connection_mixer.input_mix_weight_up.weight"],
                          w_inject=t["attn_hyper_connection.block_inject_weight.weight"])
        return a1["mixed"], a1
