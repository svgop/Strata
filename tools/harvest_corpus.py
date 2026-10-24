#!/usr/bin/env python3
"""tools/harvest_corpus.py — the exec-A seeded corpus generator (first-cycle scale).

Streams per the exec-A content mix, scaled for the FIRST training cycle:
  - deep ladder streams (~96-120k tokens): repo prose (AGENTS.md, SVG-CUSTOM.md, docs)
    chunk-shuffled with seeded wraps — the deep-position oversample;
  - conversation streams (~4k): mimic-lane-style rendered turns (offline task text,
    no HTTP), multiple per stream to fill ~40-60k.
Writes per-stream token-ID files (comma-separated) to <out>/stream-<n>.ids.
Usage: .venv/bin/python tools/harvest_corpus.py --out /home/svg/LLM/Custom-data/harvest/corpus --streams 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

PACK_TOK = Path("/home/svg/LLM/data/packs/iq3_s/tokenizer")


def build_tokenizer():
    import strata_tokenizer as ST
    vocab = json.loads((PACK_TOK / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    return ST.Tokenizer(tokens, (PACK_TOK / "merges.txt").read_text().split("\n"),
                        json.loads((PACK_TOK / "token_type.json").read_text()))


def repo_chunks(rng):
    """Chunked repo prose (the depth-lane-real SOURCES pattern)."""
    srcs = [REPO / "AGENTS.md", REPO / "SVG-CUSTOM.md", REPO / "GOAL.md"]
    srcs += sorted((REPO / "docs").glob("*.md")) if (REPO / "docs").exists() else []
    srcs += sorted(REPO.glob("campaign/research/T3*.md"))[:8]
    chunks = []
    for p in srcs:
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        step = 900
        for i in range(0, len(text), step):
            c = text[i:i + step].strip()
            if len(c) > 200:
                chunks.append(c)
    rng.shuffle(chunks)
    return chunks


def conversation(rng, idx):
    """A mimic-lane-style multi-turn exchange (offline builders, no HTTP)."""
    topics = [
        ("voice transfer", "Rewrite this technical note in the voice of a 19th-century lighthouse keeper's log."),
        ("doctrine absorption", "Absorb the naming and structure conventions of this API reference, then apply them to a new module."),
        ("architectural continuation", "Continue this service skeleton following its established layering and error conventions."),
        ("multi-turn correction", "The user corrects a detail; revise the earlier answer minimally, preserving voice."),
        ("style switch", "Re-render the same content first as release notes, then as an incident postmortem."),
        ("ui induction", "Infer the design system from these screens and spec one more screen in it."),
        ("internalize", "Summarize the house style in five rules, then follow them in a fresh example."),
        ("doc structure", "Reorganize this material into the reference's section grammar."),
    ]
    t, ask = topics[idx % len(topics)]
    turns = [f"Task ({t}): {ask}"]
    for k in range(rng.randint(3, 6)):
        turns.append(f"Revision {k}: apply the same {t} discipline to a variant; keep consistency with the prior turns.")
    return "\n\n".join(turns)


def render_with_template(tok, messages):
    """Render an assistant-turn conversation via the chat template, tokenized."""
    tmpl = (REPO / "serve" / "chat_template.jinja").read_text()
    # minimal render: concatenate user turns with the template's im_start framing
    body = ""
    for m in messages:
        body += f"<|im_start|>user\n{m}<|im_end|>\n"
    body += "<|im_start|>assistant\n<think>\n"
    return tok.encode(body, parse_special=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--deep-tokens", type=int, default=100_000)
    ap.add_argument("--conv-tokens", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    manifest = []
    for s in range(a.streams):
        rng = random.Random(a.seed + s)
        ids = []
        if s % 2 == 0:  # deep ladder stream
            chunks = repo_chunks(rng)
            i = 0
            while len(ids) < a.deep_tokens:
                c = chunks[i % len(chunks)]
                i += 1
                wrap = f"<|im_start|>user\nStudy this passage and continue its argument.\n\n{c}<|im_end|>\n"
                ids += tok.encode(wrap, parse_special=True)
            kind = f"deep-{a.deep_tokens}"
        else:           # conversation stream
            n = 0
            while len(ids) < a.conv_tokens:
                conv = render_with_template(tok, [conversation(rng, n), "Now do it again, tighter."])
                ids += conv
                n += 1
            kind = f"conv-{a.conv_tokens}"
        # cap the context: the lane's max-context is 131072
        ids = ids[:130_000]
        (out / f"stream-{s}.ids").write_text(",".join(map(str, ids)))
        manifest.append({"stream": s, "kind": kind, "tokens": len(ids)})
        print(f"stream-{s}: {kind} = {len(ids)} ids")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {a.streams} streams -> {out}")


if __name__ == "__main__":
    main()
