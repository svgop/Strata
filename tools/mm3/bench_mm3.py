#!/usr/bin/env python3
"""The mm3 lane's standing battery: quality (the pack's own reference prompt) + decode + prefill,
one number each, so every engine change rides the same before/after measurement.

    python tools/mm3/bench_mm3.py [--config config/custom-mm3-8081.json] [--repeat N]

The probes (fixed, comparable across runs):
  quality  the pack reference prompt, one sentence: the content must contain the answer and no junk
  decode   a ~400-token generation: predicted tok/s (the server's own timing)
  prefill  a ~2255-token prompt: prompt tok/s
"""
import argparse
import json
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ESSAY = ("Write a detailed technical essay about how modern MoE inference engines stage mixture-of-experts "
         "weights between CPU RAM and GPU VRAM. Cover staging slots, LRU reuse, copy engines, speculative "
         "decoding, residency heuristics, and the tradeoffs of each. Aim for completeness.")
PARA = ("The mixture-of-experts architecture conditions each transformer layer on a learned router that "
        "selects a small subset of parameter matrices per token, trading memory footprint for compute "
        "sparsity; serving such models on a single GPU requires careful staging of expert blocks between "
        "pinned host memory and device memory, an LRU over staging slots, and routing-aware residency "
        "heuristics that keep the hottest experts resident while the long tail streams over PCIe. ")


def call(base, key, payload, timeout=600):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one(base, key):
    out = {}
    q = call(base, key, {"model": "minimax-m3.1-flash", "max_tokens": 120, "messages": [
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."}]})
    c = (q["choices"][0]["message"].get("content") or "")
    out["quality_ok"] = "Paris" in c and "capital of France is Paris." in c
    out["quality"] = c[:90]

    d = call(base, key, {"model": "minimax-m3.1-flash", "max_tokens": 400, "messages": [
        {"role": "user", "content": ESSAY}]})
    out["decode_tps"] = round((d.get("timings") or {}).get("predicted_per_second") or 0, 1)
    out["decode_tokens"] = d["usage"]["completion_tokens"]

    long_prompt = ("Summarize the following text in two sentences.\n\n" + PARA * 26)
    p = call(base, key, {"model": "minimax-m3.1-flash", "max_tokens": 8, "messages": [
        {"role": "user", "content": long_prompt}]})
    t = p.get("timings") or {}
    out["prefill_tps"] = round(t.get("prompt_per_second") or 0, 1)
    out["prompt_tokens"] = p["usage"]["prompt_tokens"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE.parents[1] / "config/custom-mm3-8081.json"))
    ap.add_argument("--host", default="http://127.0.0.1:8081")
    ap.add_argument("--repeat", type=int, default=1, help="runs after the first warm the slots/profile")
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.config).read_text())
    key = cfg["api_key"]
    rows = []
    for i in range(a.repeat):
        try:
            rows.append(one(a.host, key))
        except Exception as exc:  # noqa: BLE001 - a battery reports, it does not crash
            print("run %d FAILED: %s" % (i, str(exc)[:200]))
            return 1
    for i, r in enumerate(rows):
        tag = "warm " if i else "cold "
        print("%s quality=%s decode=%.1f t/s (%d tok)  prefill=%.1f t/s (%d tok)" %
              (tag, "PASS" if r["quality_ok"] else "FAIL", r["decode_tps"], r["decode_tokens"],
               r["prefill_tps"], r["prompt_tokens"]))
        if not r["quality_ok"]:
            print("   content: %r" % r["quality"])
    if a.repeat > 1:
        warm = rows[1:]
        print("WARM MEAN decode=%.1f t/s  prefill=%.1f t/s  quality=%s" %
              (sum(x["decode_tps"] for x in warm) / len(warm),
               sum(x["prefill_tps"] for x in warm) / len(warm),
               "PASS" if all(x["quality_ok"] for x in warm) else "FAIL"))
    return 0 if rows[-1]["quality_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
