#!/usr/bin/env python3
"""campaign/vector-capture.py - the measured persona direction: teacher-forced capture on the legacy
pack, candidate-slot ranking, and the norm-calibrated single-slot build.

METHOD (CAA, arXiv 2312.06681; Arditi 2406.11717): a difference of mean residual activations between
contrastive poles, taken at the answer positions of teacher-forced responses, at the slot where it
will be applied.  Responses are elicited on the DEPLOYED pack by `campaign/vector-lab.py elicit`;
each (prompt + response) is then fed to the legacy pack as a plain prompt and every position's
per-slot residual is dumped - no generation, so no sampling noise in the capture.

INDEX RULE (proven by boot, 2026-11-23: refusal-r armed at layer 22 alone left dump slots 0..22
bitwise identical and changed slot 23 in 86/86 positions): the engine applies `direction.<l>` to the
residual AFTER block l = dump slot l+1.  So dump slot k is written to direction.<k-1> (k in 2..48).
Slot 1 (after block 0) is not steerable; slot 0 is the input embedding.

WHY THE LEGACY PACK: the deployed native IQ pack cannot dump layers (generate.cpp:3891 requires
--spec >= 2; :7697 builds the residency table only when no layer dump is open; :18017 the spec loop
refuses without it; :17770 the sequential loop exits first thing on native packs).  packs/q2_0 is a
2-bit quant of the same model: a stated confound, paid down by validating the built vector on the
deployed pack.  That store is corrupt late - slots 45..48 are non-finite in every row - so slots
0..44 are used and asserted finite.

FAIL LOUD: a failed boot, a row count that disagrees with the token count, or a non-finite value in
slots 0..44 aborts with a non-zero exit and the sample id.  (The predecessor harness logged each
failure, wrote an empty result and exited 0.)

LAYER CHOICE IS NOT MADE HERE.  Separability alone degenerates - a flat profile ranks noise - so
this tool ranks at most 3 candidate slots per trait under explicit filters (prominence, depth, the
head-bias concentration that marks lexical forcing, stream coherence); the causal choice is made by
`campaign/vector-lab.py` at runtime.

Usage:
  .venv/bin/python campaign/vector-capture.py capture --elicit E.jsonl --out DIR --per-cell 12
  .venv/bin/python campaign/vector-capture.py report --out DIR
  .venv/bin/python campaign/vector-capture.py build --out DIR --slot K --trait T --rho 0.1
  .venv/bin/python campaign/vector-capture.py selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import persona_vector as PV      # noqa: E402  (the quantised-row reader: output-head dequant owner)
import cvec_table as CT          # noqa: E402  (the control-vector GGUF owner)
import strata_tokenizer as ST    # noqa: E402  (the tokenizer owner)
from serve.frontend import ChatTemplate  # noqa: E402  (the template owner)

ENGINE = REPO / "engine/strata"
PACK_LEGACY = "/mnt/DATA/LLM/Models/GGUF/strata/data/packs/q2_0"
PLE_GGUF = "/mnt/DATA/LLM/Models/GGUF/GSQ-RCO/qwen3.8-flash-next/gsq-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
PACK_NATIVE = "/mnt/DATA/LLM/Models/GGUF/strata/data/packs/iq3_s"
PLE_SHARD = ("/mnt/DATA/LLM/Models/GGUF/GSQ-RCO/qwen3.8-flash-next/gsq-iq3s/"
             "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")
VECTOR_DIR = pathlib.Path("/mnt/DATA/LLM/Models/GGUF/vectors")
DUMP_HOME = pathlib.Path("/mnt/DATA/LLM/Models/GGUF/strata/Custom-data/persona-vectors/capture")
SLOTS, HC, N = 49, 4, 2560
FINITE_SLOTS = 45                 # slots 0..44; 45..48 are non-finite on this store
ROW_BYTES = SLOTS * HC * N * 4    # 2,007,040
MAX_SLOT = 28                     # depth <= 0.6 (slot / 48): the mid-stack prior
CANDIDATES = 3

TRAIT_WORDS = {                   # each trait's construction vocabulary, for the head DIAGNOSTIC only
    "assertive": PV.PRESETS["war-r02"]["clusters"]["assert_core"][1],
    "resourceful": PV.PRESETS["war-r02"]["clusters"]["resourceful"][1],
    "effort": ["thorough", "thoroughly", "rigorous", "rigorously", "exhaustive", "complete", "completely",
               "meticulous", "careful", "carefully", "detailed", "verify", "verification", "full", "fully"],
}
MARTIAL_WORDS = PV.PRESETS["war-r02"]["clusters"]["war_identity"][1]


class CaptureError(RuntimeError):
    """Every way a capture can be wrong; each one aborts the run (fail loud)."""


# ---------------------------------------------------------------- owners reused, never re-implemented
def server_tokenizer():
    """serve/server.py:8853-8867, the same ST.Tokenizer the deployed server encodes with."""
    tpath = pathlib.Path(PACK_NATIVE) / "tokenizer"
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    stub = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8")) if (tpath / "tokenizer.json").exists() else {}
    return ST.Tokenizer(tokens, merges, types, pre=stub.get("pre", "qwen35"), pre_pattern=stub.get("pre_pattern"),
                        normalizer=stub.get("normalizer"))


def token_ids(sample, tok, tpl, max_response_tokens):
    """The ids the DEPLOYED engine saw: the stock template's render of the sample's messages, the
    response ids (the server's own when it returned logprobs, else its encode of the text), <|im_end|>."""
    if sample.get("chat_template", PACK_NATIVE + "/tokenizer/chat_template.jinja") != PACK_NATIVE + "/tokenizer/chat_template.jinja" \
            or sample.get("thinking"):
        raise CaptureError(f"sample {sample_key(sample)} was elicited under {sample.get('chat_template')!r} "
                           f"(thinking={sample.get('thinking')}): this capture renders the stock template, thinking off")
    prompt = tpl.render(sample["messages"], tools=None, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(prompt, parse_special=True)
    body = list(sample["tokens"]) if sample.get("tokens") else tok.encode(sample["content"], parse_special=False)
    body = body[:max_response_tokens]
    return ids + body + tok.encode("<|im_end|>", parse_special=True), len(ids), len(body)


def boot(tokens_file, dump_path):
    argv = [str(ENGINE), "--pack", PACK_LEGACY, "--ple-gguf", PLE_SHARD, "--gpu", "0",
            "--tokens-file", str(tokens_file), "--max-new", "1", "--seed", "1", "--temperature", "0",
            "--max-context", "4096", "--expert-cache", "1000", "--vram-reserve-mib", "8000",
            "--dump-layers", str(dump_path)]
    t0 = time.time()
    r = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=900)
    return r, round(time.time() - t0, 1)


def reduce_dump(dump_path, n_ids, n_prompt, n_body):
    """One run -> (winsorised mean (45,4,2560), per-stream norm (45,4), stream-reduced mean (45,2560))
    over the response rows.  rows = P-1+1 = len(ids) with --max-new 1; row i is the state after
    consuming token i, so the response rows are [n_prompt, n_prompt + n_body) and the <|im_end|> row
    is excluded."""
    size = pathlib.Path(dump_path).stat().st_size
    if size != n_ids * ROW_BYTES:
        raise CaptureError(f"{pathlib.Path(dump_path).name}: {size} bytes = {size / ROW_BYTES:.4f} rows; "
                           f"the token list has {n_ids} ids (rows = P-1+1)")
    mm = np.memmap(dump_path, dtype="<f4", mode="r", shape=(n_ids, SLOTS, HC, N))
    resp = np.array(mm[n_prompt:n_prompt + n_body, :FINITE_SLOTS])
    finite = np.isfinite(resp)
    if not finite.all():
        slot = int(np.flatnonzero(~finite.all(axis=(0, 2, 3)))[0])
        raise CaptureError(f"{pathlib.Path(dump_path).name}: {int((~finite).sum())} non-finite values, first in "
                           f"slot {slot} (slots 0..{FINITE_SLOTS - 1} must be finite)")
    norms = np.sqrt((resp.astype(np.float64) ** 2).sum(axis=3)).mean(axis=0)
    thr = float(np.percentile(np.abs(resp), 99.5))
    mean = np.clip(resp, -thr, thr).mean(axis=0, dtype=np.float64)
    return mean, norms, mean.mean(axis=1)


def accumulate(state, trait, pole, mean, norms, run_red):
    cell = state.setdefault(trait, {}).setdefault(pole, {"n": 0, "mean": np.zeros((FINITE_SLOTS, HC, N)),
                                                          "m2": np.zeros((FINITE_SLOTS, HC, N)),
                                                          "norm": np.zeros((FINITE_SLOTS, HC)), "runs": []})
    cell["n"] += 1
    delta = mean - cell["mean"]
    cell["mean"] += delta / cell["n"]                      # Welford over runs
    cell["m2"] += delta * (mean - cell["mean"])
    cell["norm"] += (norms - cell["norm"]) / cell["n"]
    cell["runs"].append(run_red.astype(np.float32))


def save_state(out, state):
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for trait, poles in state.items():
        for pole, c in poles.items():
            arrays[f"{trait}/{pole}/mean"] = c["mean"].astype(np.float32)
            arrays[f"{trait}/{pole}/m2"] = c["m2"].astype(np.float32)
            arrays[f"{trait}/{pole}/norm"] = c["norm"].astype(np.float32)
            arrays[f"{trait}/{pole}/count"] = np.int64(c["n"])
            if c["runs"]:
                arrays[f"{trait}/{pole}/runs"] = np.stack(c["runs"])
    np.savez(out / "means.npz", **arrays)


def load_state(out):
    """Resume from a saved state (counts, means, M2, norms; the per-run means are kept for split-half)."""
    path = out / "means.npz"
    if not path.exists():
        return {}
    z, state = np.load(path), {}
    for key in z.files:
        trait, pole, what = key.split("/")
        cell = state.setdefault(trait, {}).setdefault(pole, {"n": 0, "mean": np.zeros((FINITE_SLOTS, HC, N)),
                                                              "m2": np.zeros((FINITE_SLOTS, HC, N)),
                                                              "norm": np.zeros((FINITE_SLOTS, HC)), "runs": []})
        if what == "count":
            cell["n"] = int(z[key])
        elif what == "runs":
            cell["runs"] = [r for r in z[key]]
        else:
            cell[what] = z[key].astype(np.float64)
    return state


# ---------------------------------------------------------------- the diagnostic head
class HeadProbe:
    """The output head as a DIAGNOSTIC, never the objective: where a direction's per-unit logit bias
    lands.  Rows are read through tools/persona_vector.py (the dequant owner); the reference set is
    a fixed random vocabulary sample, so a percentile is an estimate (n = 8192, s.e. ~ 0.5%)."""

    def __init__(self, n_sample=8192, seed=20261123):
        self.vocab = json.loads(pathlib.Path(PV.DEFAULT_VOCAB).read_text())
        self.n_vocab = len(self.vocab)
        self.words = {t: PV.resolve(self.vocab, ws)[0] for t, ws in TRAIT_WORDS.items()}
        self.words["martial"] = PV.resolve(self.vocab, MARTIAL_WORDS)[0]
        taken = {i for pairs in self.words.values() for _, i in pairs}
        rng = np.random.default_rng(seed)
        pool = [int(i) for i in rng.choice(self.n_vocab, size=4 * n_sample, replace=False) if int(i) not in taken]
        self.ref_ids = pool[:n_sample]
        self.ids = sorted(taken | set(self.ref_ids))
        self.index = {i: k for k, i in enumerate(self.ids)}
        self.W = PV.read_output_rows(PV.MODEL_SHARD, self.ids)
        self.ref = None

    def diagnose(self, vhat):
        s = self.W @ vhat.astype(np.float32)
        if self.ref is None:
            self.ref = s[[self.index[i] for i in self.ref_ids]]
        out = {}
        for name, pairs in self.words.items():
            vals = np.array([s[self.index[i]] for _, i in pairs])
            out[name] = {"tokens": int(len(vals)),
                         "promotion_pct": round(float((self.ref[None, :] < vals[:, None]).mean()), 4),
                         "energy_fraction": round(float((vals ** 2).sum() / max(1e-30, (self.ref ** 2).mean() * self.n_vocab)), 6)}
        return out


def unit(v):
    return v / np.linalg.norm(v)


def splithalf(z, trait, k):
    """cos(even-run difference, odd-run difference) at slot k - the reliability of the direction."""
    halves = []
    for parity in (0, 1):
        d = None
        for pole, sign in (("+", 1.0), ("-", -1.0)):
            runs = z[f"{trait}/{pole}/runs"]
            if len(runs) < 2:
                return None
            part = runs[parity::2].astype(np.float64).mean(axis=0)[k] * sign
            d = part if d is None else d + part
        halves.append(d)
    return round(float(unit(halves[0]) @ unit(halves[1])), 4)


def make_report(out, head=None):
    z = np.load(out / "means.npz")
    u_per = CT.read_table(PV.REFUSAL_PERLAYER)
    r_tab = CT.read_table(PV.REFUSAL_HUIHUI)
    r = unit(r_tab[min(r_tab)])
    head = head or HeadProbe()
    traits = sorted({k.split("/")[0] for k in z.files if k.endswith("/mean")})
    report = {"index_rule": "dump slot k -> direction.<k-1>; slot 1 not steerable",
              "source_quant": PACK_LEGACY, "traits": {}}
    for trait in traits:
        pos, neg = z[f"{trait}/+/mean"].astype(np.float64), z[f"{trait}/-/mean"].astype(np.float64)
        m2 = (z[f"{trait}/+/m2"].astype(np.float64) + z[f"{trait}/-/m2"].astype(np.float64))
        dof = int(z[f"{trait}/+/count"]) + int(z[f"{trait}/-/count"]) - 2
        diff = pos - neg                                           # (45,4,2560)
        d = diff.mean(axis=1)                                      # stream-reduced (45,2560)
        sigma = np.sqrt((m2 / max(1, dof)).mean(axis=1)).mean(axis=1)   # per-element variance, pooled over streams
        dprime = np.linalg.norm(d, axis=1) / np.maximum(sigma, 1e-12)
        median = float(np.median(dprime[1:FINITE_SLOTS]))
        norm_slot = ((z[f"{trait}/+/norm"] + z[f"{trait}/-/norm"]).astype(np.float64) / 2).mean(axis=1)   # (45,)
        rows = []
        for k in range(1, FINITE_SLOTS):
            v = d[k]
            nv = float(np.linalg.norm(v))
            if nv <= 0:
                continue
            vhat = v / nv
            ps = diff[k] / np.maximum(np.linalg.norm(diff[k], axis=1, keepdims=True), 1e-12)
            diag = head.diagnose(vhat)
            mine = diag[trait]
            row = {"slot": k, "direction_index": k - 1, "depth": round(k / 48, 4),
                   "d_prime": round(float(dprime[k]), 3),
                   "prominence": round(float(dprime[k] / median), 3) if median > 0 else None,
                   "mean_norm": round(float(norm_slot[k]), 4),
                   "stream_pair_cos_min": round(min(float(ps[i] @ ps[j]) for i in range(HC) for j in range(i + 1, HC)), 4),
                   "cos_to_refusal_primary": round(float(vhat @ unit(u_per[k - 1])), 4) if (k - 1) in u_per else None,
                   "cos_to_refusal_huihui": round(float(vhat @ r), 4),
                   "head": {name: diag[name] for name in (trait, "martial")}}
            row["eligible"] = bool(row["prominence"] is not None and row["prominence"] > 1.25 and k <= MAX_SLOT
                                   and mine["promotion_pct"] <= 0.75 and mine["energy_fraction"] <= 0.10
                                   and row["stream_pair_cos_min"] >= 0.8)
            rows.append(row)
        eligible = sorted((x for x in rows if x["eligible"]), key=lambda x: -x["prominence"])[:CANDIDATES]
        for x in eligible:
            x["splithalf_cos"] = splithalf(z, trait, x["slot"])
        report["traits"][trait] = {
            "runs": {"+": int(z[f"{trait}/+/count"]), "-": int(z[f"{trait}/-/count"])},
            "median_d_prime": round(median, 3),
            "candidates": eligible,
            "no_candidate_note": None if eligible else "no slot passed the filters (prominence > 1.25, slot <= %d, "
                                  "promotion pct <= 0.75, energy <= 0.10, stream-pair cos >= 0.8)" % MAX_SLOT,
            "best_by_prominence_within_depth": [{k: x[k] for k in ("slot", "prominence", "d_prime", "stream_pair_cos_min")}
                                                for x in sorted((x for x in rows if x["slot"] <= MAX_SLOT),
                                                                key=lambda x: -(x["prominence"] or 0))[:6]],
            "per_slot": rows,
        }
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def print_report(report):
    for trait, t in report["traits"].items():
        print(f"== {trait}: runs +{t['runs']['+']} -{t['runs']['-']}, median d' {t['median_d_prime']}")
        if t["no_candidate_note"]:
            print("   NO CANDIDATE:", t["no_candidate_note"])
        for x in t["candidates"]:
            h = x["head"]
            print(f"   candidate slot {x['slot']:>2} -> direction.{x['direction_index']:<2} prominence {x['prominence']:>6}  "
                  f"d' {x['d_prime']:>7}  mean||h|| {x['mean_norm']:>7}  streams {x['stream_pair_cos_min']:.3f}  "
                  f"splithalf {x['splithalf_cos']}  head[{trait}] pct {h[trait]['promotion_pct']:.3f} energy {h[trait]['energy_fraction']}  "
                  f"cos_ref {x['cos_to_refusal_primary']}/{x['cos_to_refusal_huihui']}")
        print("   best by prominence within depth:", [(x["slot"], x["prominence"]) for x in t["best_by_prominence_within_depth"]])


# ---------------------------------------------------------------- build
def router_matrix(gguf_path, block):
    """The MoE router of one block (ffn_gate_inp, BF16 in the PLE GGUF) as a float64
    [n_experts, hidden] matrix.  Structure via the owner reader; tensor bytes via numpy."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
    from gguf_reader import GGUFFile
    G = GGUFFile(str(gguf_path))
    t = {x.name: x for x in G.tensors}.get(f"blk.{block}.ffn_gate_inp.weight")
    if t is None:
        raise SystemExit(f"router-audit: no blk.{block}.ffn_gate_inp.weight in {gguf_path}")
    with open(str(gguf_path), "rb") as fh:
        fh.seek(G.data_start + t.offset)
        raw = np.frombuffer(fh.read(int(np.prod(t.shape)) * 2), dtype="<u2")
    R = (raw.astype(np.uint32) << 16).view(np.float32).reshape(t.shape[1], t.shape[0]).astype(np.float64)
    if not np.isfinite(R).all():
        raise SystemExit(f"router-audit: non-finite entries in blk.{block} router")
    return R


def build_vector(out, slot, traits, rho, out_dir, tag, other_traits_orth=False, project_router=None,
                 router_gguf=PLE_GGUF):
    """ONE tensor, direction.<slot-1> (the index rule): rho * (mean ||h|| of the slot) * v-hat.
    v-hat is the (weighted) unit difference orthogonalised against the refusal axes that ACT at that
    index, so the persona lever and the abliteration lever stay independent."""
    if not 2 <= slot < FINITE_SLOTS:
        raise SystemExit(f"slot {slot} is outside 2..{FINITE_SLOTS - 1}")
    z = np.load(out / "means.npz")
    u_per = CT.read_table(PV.REFUSAL_PERLAYER)
    r_tab = CT.read_table(PV.REFUSAL_HUIHUI)
    units, acc = [], np.zeros(N)
    for trait, w in traits:
        d = (z[f"{trait}/+/mean"] - z[f"{trait}/-/mean"]).astype(np.float64)[slot].mean(axis=0)
        units.append(unit(d))
        acc += w * unit(d)
    cosines = {}
    if len(units) == 2:
        cosines["cos_between_traits"] = round(float(units[0] @ units[1]), 4)
        if cosines["cos_between_traits"] < 0.3:
            raise SystemExit(f"refusing to combine: the two unit directions have cos {cosines['cos_between_traits']} (< 0.3)")
    axes = []
    if (slot - 1) in u_per:
        axes.append(unit(u_per[slot - 1]))
    axes.append(unit(r_tab[min(r_tab)]))
    if other_traits_orth:
        for trait in sorted({t for t, _ in traits} ^ set(TRAIT_WORDS)):
            if f"{trait}/+/mean" in z.files:
                axes.append(unit((z[f"{trait}/+/mean"] - z[f"{trait}/-/mean"]).astype(np.float64)[slot].mean(axis=0)))
    v = acc.copy()
    for _ in range(2):                      # two passes: the refusal axes are only partly aligned
        v, _ = CT.orthogonalise(v, axes)
    v = unit(v)
    router_vis_before = None
    projection_note = "none"
    if project_router is not None:
        R = router_matrix(router_gguf, project_router)
        P = np.eye(N) - R.T @ np.linalg.solve(R @ R.T, R)
        router_vis_before = round(float(np.linalg.norm(R @ v) / (np.linalg.norm(R, "fro") / np.sqrt(N))), 4)
        kept = float(np.linalg.norm(P @ v))
        if kept <= 1e-9:
            raise SystemExit(f"project-router: v lies in blk.{project_router}'s row space (nothing survives)")
        v = P @ v / kept
        projection_note = f"I - R^T(RR^t)^-1 R vs blk.{project_router} router; norm kept {kept:.4f}"
    # The dose is defined in the BLOCK'S READ DOMAIN, not the stream-mean domain.  Every block
    # RMS-normalises each hyper-connection stream on its own before the gated mix, so stream scale is
    # discarded before the read and a fixed add's share of the block input scales with the HARMONIC
    # mean of the stream norms - not the arithmetic mean.  Calibrating on the arithmetic mean (1.028
    # at slot 9) under-priced the true dose by 1.59x at slot 9 (harmonic 0.647): the 2026-11-24 MMLU
    # interleave measured the consequence (4 net questions at the "0.45" row, none at "0.3").
    stream_norms = ((z[f"{traits[0][0]}/+/norm"] + z[f"{traits[0][0]}/-/norm"]).astype(np.float64) / 2)[slot]
    mean_norm_slot = float(stream_norms.mean())
    norm_h = float(len(stream_norms) / (1.0 / np.clip(stream_norms, 1e-9, None)).sum())   # harmonic
    per_stream_rel = rho * norm_h / stream_norms
    row = rho * norm_h * v
    name = f"qwen3.8-flash-next-persona-r03-{tag}-s{slot}-rho{rho:g}.gguf"
    path = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    provenance = {
        "method": "measured teacher-forced contrast (CAA 2312.06681; Arditi 2406.11717) on the legacy pack",
        "index_rule": "dump slot k -> direction.<k-1>", "slot": slot, "direction_index": slot - 1, "rho": rho,
        "harmonic_norm_slot": round(norm_h, 6), "arithmetic_norm_slot": round(mean_norm_slot, 6),
        "dose_domain": "harmonic mean of per-stream ||h|| (the block's RMS-per-stream read domain)",
        "per_stream_relative_dose": [round(float(x), 4) for x in per_stream_rel],
        "traits": ",".join(f"{t}:{w:g}" for t, w in traits),
        "source_quant": PACK_LEGACY, "orthogonalised_to": "refusal v2-primary per-layer + Huihui r at the acting index",
        "elicit_sha256": hashlib.sha256(pathlib.Path(out / "elicit.sha256").read_bytes()).hexdigest()[:16]
        if (out / "elicit.sha256").exists() else "n/a",
        "date": time.strftime("%Y-%m-%d"), "mode": "add", "dose": "FILE:SCALE multiplies the row",
    }
    if project_router is not None:
        provenance["router_projection"] = projection_note
        provenance["router_vis_before"] = router_vis_before
    CT.write_table(str(path), {slot - 1: row}, "measured persona contrast, one slot, norm-calibrated (add mode)", provenance)
    sanity = {"file": str(path), "tensor": f"direction.{slot - 1}", "row_norm": round(float(np.linalg.norm(row)), 6),
              "unit_norm": round(float(np.linalg.norm(v)), 6),
              "cos_to_refusal_primary": round(float(v @ unit(u_per[slot - 1])), 6) if (slot - 1) in u_per else None,
              "cos_to_refusal_huihui": round(float(v @ unit(r_tab[min(r_tab)])), 6), **cosines,
              "head_diagnostic": HeadProbe().diagnose(v)}
    (out / "build").mkdir(parents=True, exist_ok=True)
    (out / "build" / f"{name}.sanity.json").write_text(json.dumps(sanity, indent=1) + "\n")
    return path, sanity


# ---------------------------------------------------------------- commands
def load_samples(elicit, traits, per_cell):
    rows = [json.loads(x) for x in pathlib.Path(elicit).read_text().splitlines() if x.strip()]
    usable = [r for r in rows if not r.get("error") and (r.get("content") or "").strip() and r["trait"] in traits]
    picked = []
    for trait in traits:
        for pole in ("+", "-"):
            cell = sorted((r for r in usable if r["trait"] == trait and r["pole"] == pole), key=lambda r: (r["seed"], r["qi"]))
            picked += cell[:per_cell]
    return picked, len(rows)


def sample_key(s):
    return f"{s['trait']}-{s['pole']}-q{s['qi']}-s{s['seed']}"


def cmd_capture(a):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samples, total = load_samples(a.elicit, a.traits.split(","), a.per_cell)
    digest = hashlib.sha256(pathlib.Path(a.elicit).read_bytes()).hexdigest()
    (out / "elicit.sha256").write_text(digest + "\n")
    print(f"capture plan: {len(samples)} boots (of {total} elicited), response cap {a.max_response_tokens} ids", flush=True)
    tok, tpl = server_tokenizer(), ChatTemplate(PACK_NATIVE + "/tokenizer/chat_template.jinja")
    state = load_state(out)
    done = set()
    if (out / "runs.jsonl").exists():
        done = {rec["key"] for rec in (json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines() if x.strip())
                if rec.get("ok")}
    run_dir = DUMP_HOME / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    dump, tokens_file, failures, ran = run_dir / "layers.bin", run_dir / "tokens.txt", 0, 0
    with open(out / "runs.jsonl", "a") as log:
        for i, s in enumerate(samples):
            key = sample_key(s)
            if a.resume and key in done:
                continue
            ids, n_prompt, n_body = token_ids(s, tok, tpl, a.max_response_tokens)
            tokens_file.write_text(" ".join(map(str, ids)))
            if a.dry_run:
                print(f"[{i + 1}/{len(samples)}] {key}: {len(ids)} ids ({n_prompt} prompt + {n_body} response + im_end)")
                continue
            rec = {"key": key, "ids": len(ids), "prompt_ids": n_prompt}
            try:
                r, wall = boot(tokens_file, dump)
                if r.returncode != 0:
                    raise CaptureError(f"engine rc {r.returncode}: {r.stderr.strip()[-240:]}")
                mean, norms, red = reduce_dump(dump, len(ids), n_prompt, n_body)
                accumulate(state, s["trait"], s["pole"], mean, norms, red)
                ran += 1
                rec.update(ok=True, wall_s=wall)
                print(f"[{i + 1}/{len(samples)}] {key}: {len(ids)} ids, {wall}s", flush=True)
            except CaptureError as e:
                failures += 1
                rec.update(ok=False, error=str(e))
                print(f"FAILED {key}: {e}", flush=True)
            finally:
                if not a.keep_dumps:
                    dump.unlink(missing_ok=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if ran % 6 == 0:
                save_state(out, state)
            if failures > a.max_failures:
                save_state(out, state)
                sys.exit(f"capture aborted: {failures} failures exceed --max-failures {a.max_failures}; last: {rec.get('error')}")
    if a.dry_run:
        return 0
    save_state(out, state)
    print_report(make_report(out))
    return 0 if failures == 0 else 1


def cmd_report(a):
    print_report(make_report(pathlib.Path(a.out)))
    return 0


def cmd_build(a):
    traits = [(t.split(":")[0], float(t.split(":")[1])) for t in a.combine.split(",")] if a.combine else [(a.trait, 1.0)]
    tag = a.tag or (a.trait if not a.combine else "+".join(t for t, _ in traits))
    path, sanity = build_vector(pathlib.Path(a.out), a.slot, traits, a.rho, pathlib.Path(a.out_dir), tag,
                                a.other_traits_orth, project_router=a.project_router, router_gguf=a.router_gguf)
    print(json.dumps(sanity, indent=1))
    print("built", path)
    return 0


def cmd_selftest(a):
    """No engine: synthetic dumps with a planted difference prove the reduction, the index rule at the
    writer, and both fail-loud paths."""
    tmp = DUMP_HOME / ("selftest-" + time.strftime("%Y%m%d-%H%M%S"))
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.5, (FINITE_SLOTS, HC, N)).astype(np.float32)
    bump = rng.normal(0, 1, N).astype(np.float32)      # ONE direction shared by all 4 streams: what the
    bump /= np.linalg.norm(bump)                        # engine's kernel can express (cvec.cu adds one v to every stream)

    def write_sample(path, pole, n_ids=12, n_prompt=4, corrupt=None):
        x = np.empty((n_ids, SLOTS, HC, N), np.float32)
        x[:, :FINITE_SLOTS] = base + rng.normal(0, 0.05, (n_ids, FINITE_SLOTS, HC, N)).astype(np.float32)
        if pole == "+":
            x[:, 20] += 3.0 * bump
        x[:, FINITE_SLOTS:] = np.nan                       # the real store's late-slot defect
        if corrupt == "nan44":                           # row 5 is a RESPONSE row (rows 4..10): the ones read
            x[5, 44, 0, 0] = np.nan
        x.tofile(path)

    state = {}
    for pole in ("+", "-"):
        for j in range(6):
            p = tmp / f"{pole}{j}.bin"
            write_sample(p, pole)
            accumulate(state, "assertive", pole, *reduce_dump(p, 12, 4, 7))
    out = tmp / "out"
    save_state(out, state)
    report = make_report(out)
    print_report(report)
    cands = report["traits"]["assertive"]["candidates"]
    assert cands and cands[0]["slot"] == 20 and cands[0]["direction_index"] == 19, cands
    print("PROVED: the planted slot 20 ranks first and maps to direction.19 (dump slot k -> direction.<k-1>)")
    path, sanity = build_vector(out, 20, [("assertive", 1.0)], 0.1, tmp / "vec", "selftest")
    rows = CT.read_table(str(path))
    assert list(rows) == [19] and abs(np.linalg.norm(rows[19]) - sanity["row_norm"]) < 1e-5, list(rows)
    print(f"PROVED: the file carries exactly one tensor, direction.19 (row norm {sanity['row_norm']})")
    for corrupt, what in (("nan44", "non-finite value in slot 44"), ("rows", "row count that disagrees with the token count")):
        p = tmp / f"bad-{corrupt}.bin"
        write_sample(p, "+", corrupt=corrupt)
        try:
            reduce_dump(p, 12 if corrupt == "nan44" else 13, 4, 7)
        except CaptureError as e:
            print(f"PROVED: {what} aborts loudly: {e}")
        else:
            sys.exit(f"SELFTEST FAILED: {what} did not abort")
    try:
        build_vector(out, 20, [("assertive", 1.0)], 0.1, tmp / "vec", "selftest")
    except SystemExit as e:
        print(f"PROVED: an existing file is never overwritten: {e}")
    else:
        sys.exit("SELFTEST FAILED: overwrite was allowed")
    print("selftest OK ->", tmp)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--elicit", required=True)
    c.add_argument("--out", required=True, help="repo evidence dir (small files); raw dumps go to Custom-data")
    c.add_argument("--traits", default="assertive,resourceful,effort")
    c.add_argument("--per-cell", type=int, default=12, help="samples per (trait, pole)")
    c.add_argument("--max-response-tokens", type=int, default=200)
    c.add_argument("--max-failures", type=int, default=0)
    c.add_argument("--resume", action="store_true")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--keep-dumps", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--slot", type=int, required=True, help="the DUMP slot k (the file gets direction.<k-1>)")
    b.add_argument("--trait", default="assertive")
    b.add_argument("--combine", default=None, help='e.g. "assertive:0.6,resourceful:0.4"')
    b.add_argument("--rho", type=float, default=0.1, help="row norm as a fraction of the slot mean ||h||")
    b.add_argument("--other-traits-orth", action="store_true")
    b.add_argument("--project-router", type=int, default=None,
                   help="project v into blk.N's router null space (RARE 2608.21236): removes the router-visible "
                        "component that carries the facts-fail knee; norm kept is reported and recorded")
    b.add_argument("--router-gguf", default=PLE_GGUF, help="the GGUF carrying the router weights")
    b.add_argument("--tag", default=None)
    b.add_argument("--out-dir", default=str(VECTOR_DIR))
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"capture": cmd_capture, "report": cmd_report, "build": cmd_build, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
