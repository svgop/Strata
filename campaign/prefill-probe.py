#!/usr/bin/env python3
"""campaign/prefill-probe.py - the prefill-axis instrument (law 17's TTFT leg): N cold long prompts
(unique content, no KV reuse), wall time to first token at max_tokens=1.  The mean is the leg's number.
Usage: .venv/bin/python campaign/prefill-probe.py --port 8081 [--tok 24000] [--n 3]"""
import argparse
import json
import random
import string
import time
import urllib.request

KEY = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"


def long_prompt(tok_target):
    # ~1 token per 4 chars of varied technical prose; a nonce kills prefix reuse
    rng = random.Random()
    nonce = "".join(rng.choices(string.ascii_lowercase, k=12))
    base = ("The interconnect fabric arbitrates between cache-coherent agents and the memory controllers. "
            "Signaling overhead, lane allocation and power states each contribute to the effective bandwidth a "
            "workload observes, and the scheduler's reordering window interacts with all three. ") * (tok_target // 40)
    return base + f" Unique session {nonce}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--tok", type=int, default=24000)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    times, toks = [], []
    for i in range(a.n):
        body = json.dumps({"messages": [{"role": "user", "content": long_prompt(a.tok)}],
                           "max_tokens": 1, "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        t0 = time.monotonic()
        d = json.loads(urllib.request.urlopen(req, timeout=600).read())
        times.append(time.monotonic() - t0)
        toks.append(d.get("usage", {}).get("prompt_tokens", 0))
    m = sum(times) / len(times)
    real = sum(toks) / len(toks) if toks and toks[0] else a.tok
    print(f"PREFILL target {a.tok} -> actual {real:.0f} tok x{a.n}: mean TTFT {m:.2f}s "
          f"(legs {[round(t, 2) for t in times]}) = wall {real / m:.0f} tok/s | "
          f"engine rate: the log's 'read in X ms' line is the authoritative number (tokenize+admission excluded)")


if __name__ == "__main__":
    main()
