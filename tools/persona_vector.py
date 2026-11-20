#!/usr/bin/env python3
"""tools/persona_vector.py — build a persona control vector from a token-contrast on the output head.

WHY THIS EXISTS
A control vector is a direction in the residual stream. The cleanest source of a *behavioural*
direction is activation contrast (CAA / persona vectors: mean activation difference between prompts
that elicit trait and anti-trait). This engine has no activation-dump path for arbitrary layers, so the
first-cut persona direction is built the other way round: the model's OUTPUT HEAD rows are vectors in
the same 2560-d residual space, so the difference of mean output rows between a trait vocabulary and an
anti-trait vocabulary is the direction that, added to the residual, raises the trait's tokens and lowers
the anti-trait's. That is a LOGIT-LENS persona vector: it biases register/identity, not a policy. Treat
it as a tone/identity knob; behaviour (tool use, research policy) belongs in the template + tool surface.

METHOD (preset `war-r01`)
- clusters, each a (positive, negative, weight): assert_core, war_identity, resourceful
- per cluster: d_c = mean(output rows of positive words) - mean(rows of negative words); d_c /= |d_c|
- v = sum_c w_c * d_c; v /= |v|
- optional orthogonalisation against the refusal axis the composition removes (so the two levers stay
  independent): v -= (v . r) r; v /= |v|
- write 47 identical F32 rows (`direction.1..47`) at unit norm; the CLI `FILE:SCALE` sets strength.

LIMITS, stated once: a token-level contrast cannot encode a behaviour it has no vocabulary for, and it
cannot add capability (there is no web tool in the engine). "Resourceful" here means the model's default
register leans to investigation/tooling/verification vocabulary instead of play/hedge/passivity.

Owner note: output.weight is read as Q6_K rows via the canonical ggml block layout
(dequantize_row_q6_K, third_party/llama.cpp/ggml/src/ggml-quants.c). If a second consumer needs
offline quantised-tensor rows, promote that reader to `tools/gguf_reader.py` (the GGUF owner).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from gguf_reader import GGUFFile          # noqa: E402  (the GGUF owner)
from gguf_writer import GGUFWriter        # noqa: E402  (the GGUF owner)

DEFAULT_VOCAB = "/mnt/DATA/LLM/Models/GGUF/strata/data/packs/iq3_s/tokenizer/vocab.json"
MODEL_SHARD = "/mnt/DATA/LLM/Models/GGUF/GSQ-RCO/qwen3.8-flash-next/gsq-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
REFUSAL_HUIHUI = "/mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-refusal-r.gguf"
REFUSAL_PERLAYER = REPO / "data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf"

# ---------------------------------------------------------------- presets
# Each word is resolved against the tokenizer; the leading-space form is preferred (it is what the
# model emits mid-sentence). Missing words are reported, never silently dropped from the count.
PRESETS = {
    # Extremely assertive, serious, non-playful, no lightness; a war-weapon identity; extremely
    # resourceful (investigate / research / tool / verify / adapt). Refusal-openers deliberately kept
    # OUT of the negative set: censorship removal is the abliteration vector's job, not this axis's.
    "war-r01": {
        "clusters": {
            "assert_core": (0.40, [
                "assertive", "assertively", "aggressive", "aggressively", "decisive", "decisively",
                "relentless", "uncompromising", "ruthless", "blunt", "firm", "direct", "directly",
                "explicit", "definite", "committed", "resolute", "determined", "authoritative",
                "dominant", "dominance", "commanding", "imperative", "serious", "seriously", "gravity",
                "grave", "solemn", "stern", "critical", "urgent", "precision", "precise", "disciplined",
                "focused", "resolve", "duty", "control", "command", "victory", "operational",
                "operations", "mission", "objective", "execute", "execution", "engage", "engagement",
                "protocol",
            ]),
            "war_identity": (0.25, [
                "weapon", "weapons", "arsenal", "lethal", "warfare", "combat", "tactical", "strategic",
                "strike", "neutralize", "target", "threat", "force", "defense", "offense", "battle",
                "soldier", "operative", "munition", "missile", "armament", "artillery", "war",
            ]),
            "resourceful": (0.35, [
                "resourceful", "resourcefulness", "ingenuity", "investigate", "investigation",
                "research", "search", "searching", "query", "queries", "verify", "verification",
                "cross-check", "reference", "references", "documentation", "docs", "manual", "tool",
                "tools", "tooling", "toolkit", "primitive", "primitives", "workaround", "fallback",
                "fallbacks", "adapt", "adaptive", "improvise", "iterate", "automation", "automate",
                "script", "scripts", "shell", "terminal", "pipeline", "harness", "orchestrate",
                "probe", "probing", "enumerate", "enumeration", "reconnaissance", "recon", "intel",
                "intelligence", "inventory", "acquire", "procure", "assemble", "construct",
                "prototype", "benchmark", "measure", "measurement", "evidence", "source", "sources",
                "debug", "debugging", "inspect", "inspection", "trace", "tracing",
            ]),
        },
        "negative": [
            "playful", "playfully", "joking", "joke", "funny", "humor", "humorous", "silly", "cute",
            "cheerful", "happy", "enjoy", "lighthearted", "light", "casual", "pretend", "imaginary",
            "hypothetical", "fiction", "fictional", "story", "imagine", "guessing", "guess", "maybe",
            "perhaps", "possibly", "hopefully", "sorry", "apolog", "apologize", "apologetic", "unable",
            "helpless", "stuck", "confused", "unsure", "unclear", "waiting", "wait", "defer",
            "deferential", "polite", "friendly", "kind", "gentle", "assistant", "helpful", "passively",
            "passive", "lazy", "idle", "hesitant",
        ],
    },
}


# ---------------------------------------------------------------- Q6_K row read
Q6_K_ROW_BLOCKS = 10          # n_embd 2560 / 256
Q6_K_BLOCK_BYTES = 210


def _dequant_q6_k_row(raw: np.ndarray) -> np.ndarray:
    """One row (blocks_per_row, 210) -> (elements,) float32. Canonical ggml layout."""
    blocks = raw.reshape(-1, Q6_K_BLOCK_BYTES).astype(np.uint8)
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32)[:, 0]
    ql, qh = blocks[:, :128], blocks[:, 128:192]
    sc = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)
    out = np.empty((blocks.shape[0], 256), dtype=np.float32)
    for base, qo, ho, so in ((0, 0, 0, 0), (128, 64, 32, 8)):
        a, b, hh = ql[:, qo:qo + 32], ql[:, qo + 32:qo + 64], qh[:, ho:ho + 32]
        q1 = ((a & 0xF).astype(np.int16) | (((hh >> 0) & 3) << 4).astype(np.int16)) - 32
        q2 = ((b & 0xF).astype(np.int16) | (((hh >> 2) & 3) << 4).astype(np.int16)) - 32
        q3 = ((a >> 4).astype(np.int16) | (((hh >> 4) & 3) << 4).astype(np.int16)) - 32
        q4 = ((b >> 4).astype(np.int16) | (((hh >> 6) & 3) << 4).astype(np.int16)) - 32
        for oo, si, q in ((0, 0, q1), (32, 2, q2), (64, 4, q3), (96, 6, q4)):
            ss = sc[:, so + np.arange(32) // 16 + si]
            out[:, base + oo:base + oo + 32] = d[:, None] * ss * q.astype(np.float32)
    return out.reshape(-1)


def read_output_rows(model_path: str, row_ids: list[int]) -> np.ndarray:
    g = GGUFFile(model_path)
    t = next((x for x in g.tensors if x.name == "output.weight"), None)
    if t is None or t.type_name != "Q6_K":
        raise SystemExit(f"{model_path}: output.weight not found as Q6_K")
    n_embd = int(t.shape[0])
    if n_embd % 256:
        raise SystemExit(f"output.weight width {n_embd} is not Q6_K-block aligned")
    blocks = n_embd // 256
    row_bytes = blocks * Q6_K_BLOCK_BYTES
    base = g.data_start + t.offset
    out = np.empty((len(row_ids), n_embd), dtype=np.float32)
    with open(model_path, "rb") as fh:
        for i, rid in enumerate(row_ids):
            fh.seek(base + rid * row_bytes)
            raw = np.frombuffer(fh.read(row_bytes), dtype=np.uint8)
            out[i] = _dequant_q6_k_row(raw)
    return out


# ---------------------------------------------------------------- token resolution
def resolve(vocab: dict[str, int], words: list[str]) -> tuple[list[tuple[str, int]], list[str]]:
    out, missing = [], []
    for w in words:
        for cand in ("Ġ" + w, w, "Ġ" + w.capitalize()):
            if cand in vocab:
                out.append((cand, vocab[cand]))
                break
        else:
            missing.append(w)
    return out, missing


# ---------------------------------------------------------------- refusal axis
def read_direction_row(path: str, layer: int) -> np.ndarray:
    g = GGUFFile(path)
    raw = open(path, "rb").read()
    for t in g.tensors:
        if t.name == f"direction.{layer}":
            off = g.data_start + t.offset
            v = np.frombuffer(raw[off:off + int(t.shape[0]) * 4], dtype="<f4").astype(np.float64)
            return v / np.linalg.norm(v)
    raise SystemExit(f"{path}: no direction.{layer}")


def mean_direction(path: str, layers: list[int]) -> np.ndarray:
    rows = [read_direction_row(path, l) for l in layers]
    m = np.mean(rows, axis=0)
    return m / np.linalg.norm(m)


# ---------------------------------------------------------------- build
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="war-r01", choices=sorted(PRESETS))
    ap.add_argument("--model", default=MODEL_SHARD)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=47, help="rows in the control-vector file")
    ap.add_argument("--orthogonal-to-perlayer", action="store_true",
                    help="remove the component along the per-layer refusal axis (the abliteration the "
                         "additive composition removes); keeps the two levers independent")
    ap.add_argument("--orthogonal-layers", default="16,32", help="A,B layers of the per-layer vector")
    ap.add_argument("--validate-cache", default="/tmp/qwen_out_W.npy",
                    help="optional cached float32 output head for full-vocab logit-lens validation")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    vocab = json.loads(pathlib.Path(args.vocab).read_text())

    neg_ids, neg_missing = resolve(vocab, preset["negative"])
    if len(neg_ids) < 10:
        raise SystemExit(f"negative set too small after resolution ({len(neg_ids)})")
    rows = {name: resolve(vocab, words) for name, (_, words) in preset["clusters"].items()}

    all_ids = sorted({i for _, i in neg_ids} | {i for name in rows for _, i in rows[name][0]})
    W = read_output_rows(args.model, all_ids)
    index = {rid: k for k, rid in enumerate(all_ids)}
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)

    u_neg = np.mean([Wn[index[i]] for _, i in neg_ids], axis=0)
    v = np.zeros_like(u_neg)
    weights, used = {}, {}
    for name, (w, _) in preset["clusters"].items():
        pos, _miss = rows[name]
        if len(pos) < 8:
            raise SystemExit(f"cluster {name} resolved to {len(pos)} tokens")
        u_pos = np.mean([Wn[index[i]] for _, i in pos], axis=0)
        d = u_pos - u_neg
        d /= np.linalg.norm(d)
        v += w * d
        weights[name], used[name] = w, len(pos)
    v /= np.linalg.norm(v)

    cos_raw = {
        "huihui_r": float(np.dot(v, read_direction_row(REFUSAL_HUIHUI, 1))),
    }
    a, b = (int(x) for x in args.orthogonal_layers.split(","))
    r_per = mean_direction(str(REFUSAL_PERLAYER), list(range(a, b + 1)))
    cos_raw["perlayer_mean_%d_%d" % (a, b)] = float(np.dot(v, r_per))
    if args.orthogonal_to_perlayer:
        v = v - np.dot(v, r_per) * r_per
        v /= np.linalg.norm(v)
        cos_raw["after_orthogonalisation"] = float(np.dot(v, r_per))

    report = {
        "preset": args.preset,
        "clusters": {k: {"weight": weights[k], "tokens": used[k]} for k in weights},
        "negative_tokens": len(neg_ids),
        "missing": {k: [w for w in rows[k][1]] for k in rows} | {"negative": neg_missing},
        "cos_to_refusal_before_orth": {k: round(x, 4) for k, x in cos_raw.items()},
        "norm": float(np.linalg.norm(v)),
        "sha256_source": None,
    }

    writer = GGUFWriter()
    writer.add("general.architecture", "controlvector")
    writer.add("controlvector.model_hint", "qwen4exp")
    writer.add("controlvector.layer_count", args.layers)
    writer.add("strata.persona.preset", args.preset)
    writer.add("strata.persona.kind", "output-head token contrast (logit-lens); register/identity, not policy")
    writer.add("strata.persona.clusters", json.dumps({k: weights[k] for k in weights}))
    writer.add("strata.persona.orthogonal_to_perlayer", bool(args.orthogonal_to_perlayer))
    for l in range(1, args.layers + 1):
        writer.add_f32(f"direction.{l}", v.astype(np.float32).reshape(-1))
    path = writer.write(args.out)

    # full-vocab validation, if a dequantised head is cached
    if args.validate_cache and pathlib.Path(args.validate_cache).exists():
        Wall = np.load(args.validate_cache, mmap_mode="r")
        s = np.asarray(Wall @ v.astype(np.float32))
        order = np.argsort(s)
        inv = {i: t for t, i in vocab.items()}
        def show(idx, label, n=20):
            print(f"\n{label}")
            for i in idx[:n]:
                tok = inv.get(int(i), "?")
                print("  %+0.4f  %r" % (s[i], tok.replace("Ġ", " ")))
        show(order, "MOST SUPPRESSED (v points away -> adding v lowers these)")
        show(order[::-1], "MOST PROMOTED (v points toward -> adding v raises these)")
        report["validation"] = {
            "dot_std": float(s.std()),
            "top_promoted": [inv.get(int(i), "?") for i in order[::-1][:15]],
            "top_suppressed": [inv.get(int(i), "?") for i in order[:15]],
        }

    print("\n" + json.dumps(report, indent=1))
    print(f"\nwrote {path}  ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
