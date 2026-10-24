#!/usr/bin/env python3
"""campaign/retrain/train_cycle.py — the FastMTP retrain cycle with the CHECKPOINT SYSTEM (F-325).

The resumable K=6 teacher-forced CE cycle over the harvest corpus: every anchor contributes
six depth terms (R[a+d-1] -> tok[a+d], w_deep doubled at pos>=16384 per F-235), AdamW + cosine,
2% fixed eval split, per-depth top-1 acceptance vs the stock drafter.

Data home (the F-325 Custom-data layout):
  /home/svg/LLM/Custom-data/harvest/records/records-*.f32   {int64 pos, int64 tok, f32 R[4][2560]}
  /home/svg/LLM/Custom-data/harvest/records/windows.npy     seam-respecting window index (exec-A)
  /home/svg/LLM/Custom-data/mtp/fixtures/tap_head.q4        the Q4 head bytes (W_head build)
  /home/svg/LLM/Custom-data/mtp/ckpts/                      cycle-last.pt / cycle-best.pt / history.jsonl
  /home/svg/LLM/Custom-data/mtp/export/                     dense_rt2.bin / dense_rt2.txt (+ .tmp staging)

Checkpoint contract:
  cycle-last.pt  rolling full state {model, opt, sched, step, rng, eval_history, best_mean};
                 written ATOMICALLY (tmp + os.replace) every --save-every steps and at every
                 eval — a kill at any point loses at most one save interval.
  cycle-best.pt  same payload, snapshotted whenever the eval mean sets a new best.
  history.jsonl  one line per eval {step, wall_s, loss, per-depth rates, mean, delta}.
  Resume is automatic. A legacy bare state_dict (full_cycle_ckpt.pt) seeds the model only.

GPU-EXCLUSIVE: the 8081 server must be DOWN (single-engine law). Run under the training venv:
  ~/venvs/mtp-train/bin/python campaign/retrain/train_cycle.py --minutes 30
  ~/venvs/mtp-train/bin/python campaign/retrain/train_cycle.py --steps 30 --eval-every 20 --save-every 10   # smoke

Export: the trained dense tensors re-quantized to their STOCK rt kinds (q8_0 / bf16 / f32 raw
norms, the +1 un-fold per F-109), dense.txt dialect, atomically staged then renamed. Asserts
ALL 29 rt tensors are present — the F-325 lesson (a name round-trip bug exported 1 of 30
silently). The final --mtp-ready rt2 dir is assembled by tools/mtp_rt2_assemble.py, which
copies the frozen experts.bin + draft_vocab.bin byte-identical from the stock rt.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "campaign" / "retrain"))
from p2_rt_loader import RtDense          # noqa: E402
from mtp_module import MTPDraft           # noqa: E402


def pack_q8_0(x: np.ndarray) -> bytes:
    """ggml quantize_row_q8_0_ref — verbatim transcription of tools/mtp_rt.py's packer (the
    engine's own reference; kept local because importing mtp_rt drags in gguf/yaml, which the
    training venv does not carry). If mtp_rt's packer ever changes, this copy must follow."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()

RECORDS_GLOB = "/home/svg/LLM/Custom-data/harvest/records/records-*.f32"
WINDOWS_NPY = "/home/svg/LLM/Custom-data/harvest/records/windows.npy"
HEAD_Q4 = "/home/svg/LLM/Custom-data/mtp/fixtures/tap_head.q4"
DRAFT_VOCAB = "/home/svg/LLM/data/mtp-en/rt/draft_vocab.bin"
CKPT_DIR = Path("/home/svg/LLM/Custom-data/mtp/ckpts")   # module default; main() may rebind via --ckpt-dir
EXPORT_DIR = Path("/home/svg/LLM/Custom-data/mtp/export")
LEGACY_SEED = CKPT_DIR / "full_cycle_ckpt.pt"
REC = 40976
K = 6


def load_corpus():
    """All records concatenated + the seam-respecting anchor list from windows.npy.
    windows.npy starts are PER-SHARD record indices (they reset at each shard boundary) —
    each window is a 512-record tile of a seam-respecting run, so a global anchor a is
    valid iff a..a+K all fall inside one tile: a in [tile_start, tile_start + W - K - 1]."""
    W = 512
    pos_l, tok_l, R_l, offsets = [], [], [], [0]
    for f in sorted(glob.glob(RECORDS_GLOB)):
        raw = np.fromfile(f, dtype=np.uint8)
        n = raw.size // REC
        arr = raw.reshape(n, REC)
        pos_l.append(arr[:, 0:8].copy().view(np.int64).reshape(n))
        tok_l.append(arr[:, 8:16].copy().view(np.int64).reshape(n))
        R_l.append(arr[:, 16:].copy().view(np.float32).reshape(n, 4, 2560))
        offsets.append(offsets[-1] + n)
    pos = np.concatenate(pos_l); tok = np.concatenate(tok_l); R = np.concatenate(R_l)
    starts = list(map(int, np.load(WINDOWS_NPY)[:, 0]))
    anchors, shard, prev = [], 0, -1
    for st in starts:            # windows are in load order; a start reset marks the next shard
        if st < prev:
            shard += 1
        prev = st
        anchors.extend(range(offsets[shard] + st, offsets[shard] + st + W - K))
    return pos, tok, R, np.asarray(anchors, dtype=np.int64)


def build_head():
    """W_head [40525, 2560] from the engine's Q4 bytes: per-block SIGNED d, split-halves nibbles."""
    rows = np.fromfile(HEAD_Q4, dtype=np.uint8).reshape(40525, 80, 18)
    d_w = rows[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(40525, 80, 1)
    qs = rows[:, :, 2:]
    nib = np.empty((40525, 80, 32), dtype=np.float32)
    nib[:, :, :16] = qs & 0xF; nib[:, :, 16:] = qs >> 4
    return torch.from_numpy(((nib - 8.0) * d_w).reshape(40525, 2560).copy()).cuda()


def save_ckpt(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def export_dense(mod: MTPDraft, rt: RtDense, meta: dict):
    """The trained dense tensors -> export/dense_rt2.{bin,txt} in the rt dialect, atomically.
    Kind per tensor = its STOCK rt kind (rt.manifest is the authority); the name map is mod.t
    (original names), never a string round-trip. Hard-fails unless all rt tensors are covered."""
    kinds = {name: kind for name, kind, *_ in rt.manifest}
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines, blob = [], bytearray()
    off = 0
    for name, kind in sorted(kinds.items()):
        p = mod.t[name].detach()
        arr = p.cpu().numpy().astype(np.float32)
        rows, cols = arr.shape if arr.ndim == 2 else (1, arr.size)
        if kind == "q8_0":
            raw = pack_q8_0(arr.reshape(rows, cols))
        elif kind == "bf16":
            raw = ((arr.view(np.uint32) >> 16).astype(np.uint16)).tobytes()
        elif kind == "f32":
            raw = arr.astype(np.float32).tobytes()   # as-is: the param HOLDS the FOLDED (1+w)
            # value and the engine multiplies gamma directly (native_qsa norm kernel) — the
            # dense.bin f32 norm is the FOLDED value (mtp_rt.py packs data+1.0; G0 caught a
            # candidate un-fold at F-325: it would have shipped every norm off by exactly 1.0)
        else:
            raise ValueError(f"unknown rt kind {kind} for {name}")
        pad = (-off) % 256
        blob += b"\0" * pad; off += pad
        blob += raw
        lines.append(f"{name} {kind} {rows} {cols} {off} {len(raw)}")
        off += len(raw)
    assert len(lines) == len(rt.manifest), f"export covers {len(lines)}/{len(rt.manifest)} rt tensors"
    (EXPORT_DIR / "dense_rt2.bin.tmp").write_bytes(bytes(blob)); os.replace(EXPORT_DIR / "dense_rt2.bin.tmp", EXPORT_DIR / "dense_rt2.bin")
    (EXPORT_DIR / "dense_rt2.txt.tmp").write_text("\n".join(lines) + "\n"); os.replace(EXPORT_DIR / "dense_rt2.txt.tmp", EXPORT_DIR / "dense_rt2.txt")
    (EXPORT_DIR / "export_meta.json.tmp").write_text(json.dumps(meta, indent=1)); os.replace(EXPORT_DIR / "export_meta.json.tmp", EXPORT_DIR / "export_meta.json")
    return len(lines), off


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30.0, help="wall budget for this invocation")
    ap.add_argument("--steps", type=int, default=0, help="hard step cap for this invocation (0 = budget only)")
    ap.add_argument("--anchors-frac", type=float, default=0.25, help="fraction of train anchors visited this chunk")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--eval-n", type=int, default=200, help="eval anchors per mid-run eval (final uses 2x)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eval-only", action="store_true", help="no training: at-scale accept over the FULL eval split on cycle-best.pt")
    ap.add_argument("--ckpt-dir", default=str(globals()["CKPT_DIR"]), help="checkpoint dir (isolate experiments, e.g. ckpts-gentle)")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    CKPT_DIR = Path(a.ckpt_dir)
    LEGACY_SEED = CKPT_DIR / "full_cycle_ckpt.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pos, tok, R, anchors = load_corpus()
    print(f"corpus: {len(pos)} records, {len(anchors)} seam-respecting anchors", flush=True)

    rng = np.random.default_rng(a.seed)
    n_eval = max(len(anchors) // 50, 200)
    eval_set = set(rng.choice(anchors, size=n_eval, replace=False).tolist())
    train_pool = np.asarray([x for x in anchors.tolist() if x not in eval_set], dtype=np.int64)

    W_head = build_head()
    dv = np.fromfile(DRAFT_VOCAB, dtype=np.int32)
    in_sub = -np.ones(248320, dtype=np.int64); in_sub[dv] = np.arange(len(dv))

    rt = RtDense()
    mod = MTPDraft({k: torch.from_numpy(v.copy()) for k, v in rt.tensors.items()}).cuda()
    opt = torch.optim.AdamW([p for p in mod.parameters()], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)   # T_max fixed below, once chunk size is known
    last = CKPT_DIR / "cycle-last.pt"
    step0, hist, best_mean = 0, [], -1.0
    if last.exists():
        pay = torch.load(last, weights_only=False)
        mod.load_state_dict(pay["model"]); opt.load_state_dict(pay["opt"])
        rng.bit_generator.state = pay["rng"]
        step0, hist, best_mean = pay["step"], pay["eval_history"], pay["best_mean"]
        print(f"resumed cycle-last.pt at step {step0} (best {best_mean:.4f}, {len(hist)} evals)", flush=True)
    elif LEGACY_SEED.exists():
        mod.load_state_dict(torch.load(LEGACY_SEED, weights_only=True))
        print(f"legacy seed {LEGACY_SEED.name} loaded (model only, fresh optimizer)", flush=True)
    mod_stock = MTPDraft({k: torch.from_numpy(v.copy()) for k, v in rt.tensors.items()}).cuda()

    def depth_weight(p):
        return 2.0 if p >= 16384 else 1.0

    def batch_loss(batch):
        terms = []
        for anc in batch:
            wa = depth_weight(pos[anc])
            for d in range(1, K + 1):
                mixed, _ = mod(torch.from_numpy(R[anc + d - 1].copy()).cuda())
                ls = in_sub[tok[anc + d]]
                if ls < 0:
                    continue
                logits = mixed @ W_head.T
                terms.append(wa * F.cross_entropy(logits.unsqueeze(0), torch.tensor([int(ls)]).cuda()))
        return torch.stack(terms).mean()

    def accept(model, n):
        hits = np.zeros(K, dtype=np.int64); tries = np.zeros(K, dtype=np.int64)
        for anc in rng.choice(sorted(eval_set), size=min(n, len(eval_set)), replace=False):
            for d in range(1, K + 1):
                with torch.no_grad():
                    pred = int((model(torch.from_numpy(R[anc + d - 1].copy()).cuda())[0] @ W_head.T).argmax())
                ls = in_sub[tok[anc + d]]
                if ls < 0:
                    continue
                tries[d - 1] += 1; hits[d - 1] += int(pred == int(ls))
        return hits / np.maximum(tries, 1)

    if a.eval_only:
        best = CKPT_DIR / "cycle-best.pt"
        if best.exists():
            mod.load_state_dict(torch.load(best, weights_only=False)["model"])
            print("at-scale eval on cycle-best.pt", flush=True)
        rb = accept(mod_stock, len(eval_set))
        rt_ = accept(mod, len(eval_set))
        print(f"stock  (full split): {' '.join(f'{x:.4f}' for x in rb)} mean {rb.mean():.4f}", flush=True)
        print(f"trained(full split): {' '.join(f'{x:.4f}' for x in rt_)} mean {rt_.mean():.4f}", flush=True)
        print("deltas: " + " ".join(f"{(b - c) * 100:+.1f}pt" for b, c in zip(rt_, rb)), flush=True)
        deep = np.asarray([pos[x] >= 16384 for x in sorted(eval_set)])
        print(f"eval split: {len(eval_set)} anchors ({int(deep.sum())} deep >=16384)", flush=True)
        return 0

    r0 = accept(mod_stock, a.eval_n)
    stock_mean = float(r0.mean())
    print(f"stock eval: {' '.join(f'{x:.3f}' for x in r0)} mean {stock_mean:.4f}", flush=True)

    chunk = rng.choice(train_pool, size=max(int(len(train_pool) * a.anchors_frac), a.bs), replace=False)
    chunk_steps = (len(chunk) + a.bs - 1) // a.bs
    total_steps = a.steps if a.steps > 0 else chunk_steps
    print(f"chunk: {len(chunk)} anchors / {chunk_steps} steps (cap {total_steps}, budget {a.minutes:.0f} min)", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=chunk_steps)

    def run_eval(step, loss_avg):
        nonlocal best_mean
        r = accept(mod, a.eval_n)
        rec = {"step": step, "wall_s": round(time.time() - t0, 1), "loss": round(loss_avg, 4),
               "per_depth": [round(float(x), 4) for x in r], "mean": round(float(r.mean()), 4),
               "delta_pt": round(float((r.mean() - stock_mean) * 100), 2)}
        hist.append(rec)
        print(f"  eval @{step}: loss {rec['loss']:.4f} | {' '.join(f'{x:.3f}' for x in r)} | "
              f"mean {rec['mean']:.4f} ({rec['delta_pt']:+.1f}pt vs stock)", flush=True)
        with open(CKPT_DIR / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["mean"] > best_mean:
            best_mean = rec["mean"]
            save_ckpt(CKPT_DIR / "cycle-best.pt",
                      {"model": mod.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                       "step": step, "rng": rng.bit_generator.state, "eval_history": hist,
                       "best_mean": best_mean, "stock_mean": stock_mean})
        return rec

    step = step0
    ep_loss, ep_cnt = 0.0, 0
    order = rng.permutation(chunk)
    stop = False
    for s0 in range(0, len(order), a.bs):
        if (s0 // a.bs) >= total_steps or (time.time() - t0) / 60 > a.minutes:
            stop = True; break
        loss = batch_loss(order[s0:s0 + a.bs])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step(); sched.step()
        step += 1; ep_loss += float(loss.detach()); ep_cnt += 1
        if step % a.eval_every == 0:
            run_eval(step, ep_loss / max(ep_cnt, 1)); ep_loss, ep_cnt = 0.0, 0
        if step % a.save_every == 0:
            save_ckpt(last, {"model": mod.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                             "step": step, "rng": rng.bit_generator.state, "eval_history": hist,
                             "best_mean": best_mean, "stock_mean": stock_mean})

    final = run_eval(step, ep_loss / max(ep_cnt, 1))
    save_ckpt(last, {"model": mod.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "step": step, "rng": rng.bit_generator.state, "eval_history": hist,
                     "best_mean": best_mean, "stock_mean": stock_mean})
    rf = accept(mod, a.eval_n * 2)
    print(f"final eval (2x): {' '.join(f'{x:.3f}' for x in rf)} mean {rf.mean():.4f}", flush=True)
    print("deltas vs stock: " + " ".join(f"{(b - c) * 100:+.1f}pt" for b, c in zip(rf, r0)), flush=True)
    print(f"{'stopped on budget' if stop else 'chunk complete'} at step {step} "
          f"({(time.time() - t0) / 60:.1f} min); best mean {best_mean:.4f}", flush=True)

    if not a.no_export:
        n_t, n_b = export_dense(mod, rt, {"step": step, "best_mean": best_mean,
                                          "final_mean": float(rf.mean()), "stock_mean": stock_mean,
                                          "history_len": len(hist)})
        print(f"exported {n_t} tensors ({n_b / 1e6:.1f} MB) -> {EXPORT_DIR}/dense_rt2.{{bin,txt}}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
