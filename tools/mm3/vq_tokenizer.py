#!/usr/bin/env python3
"""The VQ release's HF tokenizer.json -> the server's tokenizer stub (the same conversion every
family's pack carries: <pack>/tokenizer/{vocab.json, merges.txt, token_type.json, tokenizer.json}).

    python tools/mm3/vq_tokenizer.py --release /mnt/DATA/LLM/Models/GGUF/VQ/MiniMax-M3.1-Flash

The pre-tokenizer (the M2-lineage pattern) and the NFC normalizer are shipped in the stub so the
server runs what the family declared (tools/strata_tokenizer.py's PRE_PATTERNS/NORMALIZERS law).
The pattern here is transcribed from llama.cpp's MINIMAX_M2 case, which is byte-identical to the
release's tokenizer.json first Split - the stub ships the llama.cpp transcription of record.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from strata_tokenizer import PRE_PATTERNS, NORMALIZERS  # noqa: E402  (the transcription of record)

BOS, EOS = 200019, 200020  # generation_config.json (the mm3.txt geometry carries them too)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    a = ap.parse_args()
    rel = pathlib.Path(a.release)
    j = json.loads((rel / "tokenizer.json").read_text(encoding="utf-8"))

    vocab = j["model"]["vocab"]                      # token -> id, contiguous from 0
    n = len(vocab)
    tokens = [None] * n
    for s, i in vocab.items():
        tokens[i] = s
    types = [1] * n                                  # 1 = normal (the byte-unicode BPE pieces)
    for at in j.get("added_tokens", []):             # ids continue past the base vocabulary
        if at["id"] >= len(tokens):
            tokens += [None] * (at["id"] + 1 - len(tokens))
            types += [1] * (at["id"] + 1 - len(types))
        tokens[at["id"]] = at["content"]
        types[at["id"]] = 4                          # USER_DEFINED: matched with or without parse_special
    merges = ["%s %s" % (m[0], m[1]) if isinstance(m, list) else m for m in j["model"]["merges"]]

    out = rel / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({s: i for i, s in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    # the family's own chat template rides the stub (serve/server.py resolves tpath / "chat_template.jinja"
    # first): the release's <mm:think> tokens, its <effort> line, its tool spelling - the shared serve
    # template is the fallback for families without one, not a substitute for the model's own format
    src_tpl = rel / "chat_template.jinja"
    if src_tpl.exists():
        (out / "chat_template.jinja").write_text(src_tpl.read_text(encoding="utf-8"), encoding="utf-8")
    stub = {
        "model": "gpt2",
        "pre": "minimax-m3",
        "vocab_size": len(tokens),
        "n_merges": len(merges),
        "special_ids": {"bos": BOS, "eos": EOS},
        "add_bos_token": False,
        "pre_pattern": PRE_PATTERNS["minimax-m3"],
        "pre_pattern_source": "tools/strata_tokenizer.py MINIMAX_M3_PATTERN (llama.cpp L439-444; the "
                              "release tokenizer.json's first Split is identical)",
        "normalizer": NORMALIZERS.get("minimax-m3"),
    }
    (out / "tokenizer.json").write_text(json.dumps(stub, ensure_ascii=False, indent=1), encoding="utf-8")
    print("tokenizer stub: %s (%d tokens, %d merges, eos %d)" % (out, len(tokens), len(merges), EOS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
