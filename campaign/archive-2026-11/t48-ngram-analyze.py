#!/usr/bin/env python3
"""campaign/t48-ngram-analyze.py — part 2 of the n-gram draft falsifier: the
prompt-lookup ceiling.  Tokenizes (prompt + completion) with the pack tokenizer, then
for every completion position i and n in 4..8 asks: does the n-gram ending at i occur
EARLIER in the sequence, and did its earlier continuation equal the actual token at i?
The per-request hit-rate = the fraction of positions an n-gram drafter would have
predicted exactly (the PLD metric).  Compare against the engine's MTP acceptance."""
import json, os, sys
from pathlib import Path

REPO = Path("/home/svg/LLM/strata-custom-2")
sys.path.insert(0, str(REPO / "tools"))
import strata_tokenizer as ST

t = Path("/home/svg/LLM/data/packs/iq3_s/tokenizer")
vocab = json.loads((t / "vocab.json").read_text())
tokens = [None] * len(vocab)
for s, i in vocab.items():
    tokens[i] = s
tok = ST.Tokenizer(tokens, (t / "merges.txt").read_text().split("\n"),
                   json.loads((t / "token_type.json").read_text()))

def ngram_ceiling(ids, lo, ns=range(4, 9)):
    """ids[lo:] = completion.  For each position i >= lo+1: was ids[i] the continuation
    of an earlier occurrence of the n-gram ids[i-n..i-1], for any n in ns?"""
    from collections import defaultdict
    hits = tried = 0
    table = defaultdict(list)          # n-gram tuple -> earlier continuations
    for i in range(1, len(ids)):
        if i >= lo + 1:
            ok = False
            for n in ns:
                if i - n < 0:
                    continue
                gram = tuple(ids[i - n:i])
                if ids[i] in table.get(gram, ()):
                    ok = True
                    break
            hits += ok
            tried += 1
        for n in ns:                   # index this position as an earlier occurrence
            if i - n + 1 >= 0 and i + 1 < len(ids) + 1 and i - n >= 0:
                table[tuple(ids[i - n:i])].append(ids[i]) if i >= lo or True else None
    return hits / max(tried, 1)

rows = json.loads((REPO / "campaign/results/t48-ngram-raw.json").read_text())
print(f"{'prompt':18s} {'MTP-accept':>10s} {'PLD-ceiling':>11s} {'headroom':>8s}")
for r in rows:
    import re
    m = re.search(r"(\d+) of (\d+) drafts accepted", r["engine_accept"])
    acc = int(m.group(1)) / int(m.group(2)) if m else 0.0
    prompt_ids = tok.encode(r["prompt"], parse_special=False)
    comp_ids = tok.encode(r["content"], parse_special=False)
    ids = prompt_ids + comp_ids
    ceil = ngram_ceiling(ids, len(prompt_ids))
    print(f"{r['kind']+'/'+r['name']:18s} {acc:10.1%} {ceil:11.1%} {max(ceil-acc,0):8.1%}")
