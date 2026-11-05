#!/usr/bin/env python3
"""campaign/conc-probe.py - the fast conc-only instrument for the upstream bisection: N
concurrent streams over the lane probe's CONC prompt, aggregate tokens/s, ~90 s a leg.
Usage: .venv/bin/python campaign/conc-probe.py --port 8080 [--streams 6] [--tok 600] [--key K]"""
import argparse
import json
import threading
import time
import urllib.request

CONC = ('Explain in detail how mixture-of-experts routing works in modern LLMs, including load balancing, '
        'capacity factors, and the tradeoffs versus dense models.')


def one(port, tok, key, out, i):
    body = json.dumps({"messages": [{"role": "user", "content": CONC}], "max_tokens": tok,
                       "temperature": 0.7, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        out[i] = d.get("usage", {}).get("completion_tokens", tok)
    except Exception as e:
        print(f"  stream {i}: {e}")
        out[i] = 0
    out[i + 8] = time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--tok", type=int, default=600)
    ap.add_argument("--key", default="3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    for r in range(a.rounds):
        out = {}
        th = [threading.Thread(target=one, args=(a.port, a.tok, a.key, out, i)) for i in range(a.streams)]
        t0 = time.monotonic()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.monotonic() - t0
        toks = sum(out.get(i, 0) for i in range(a.streams))
        print(f"round {r + 1}: {toks} tok in {wall:.2f}s = {toks / wall:.1f} t/s aggregate "
              f"(per-stream {[round(out.get(i, 0) / out.get(i + 8, 1), 1) for i in range(a.streams)]})")


if __name__ == "__main__":
    main()
