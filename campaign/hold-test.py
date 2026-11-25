#!/usr/bin/env python3
# campaign/hold-test.py - the warm-in's hold, from the client's side (F-467).
#
# A request sent the moment the port answers - while the server's warm-in still runs - and what it met: how long it
# waited for its first token, what /health said when it was sent (the warm-in's phase and the hold it advertised),
# and how fast it then decoded.  With the config's "prewarm.hold_s" the request waits at the door and is served by a
# warm engine; without it the request pre-empts the warm-in and runs on a half-warmed one.
#
# --no-wait sends the request with "X-Strata-Wait: 0", the header that bypasses the hold: the behaviour before
# F-467 (the request pre-empts the warm-in).  A second request of the same shape follows in both cases, so the
# first request's decode can be read against a warm one.
#
# Run it from campaign/ctx-lane.sh (it boots the server and starts this at /health's first answer):
#   bash campaign/ctx-lane.sh <label> --during .venv/bin/python campaign/hold-test.py -- .venv/bin/python campaign/ctx-probe.py ...
# The request helpers are campaign/ctx-probe.py's (the owner).
import importlib.util
import json
import pathlib
import sys
import time
import urllib.request

spec = importlib.util.spec_from_file_location("ctx_probe", pathlib.Path(__file__).with_name("ctx-probe.py"))
ctx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctx)

HEALTH = "http://127.0.0.1:8081/health"


def health():
    try:
        with urllib.request.urlopen(urllib.request.Request(HEALTH), timeout=3) as r:
            return json.loads(r.read() or b"{}")
    except Exception:
        return None


def body(seed):
    return {"model": ctx.MODEL, "stream": True, "max_tokens": 256,
            "messages": [{"role": "user", "content": ctx.gen_prompt(800, seed)}]}


def main():
    no_wait = "--no-wait" in sys.argv
    if no_wait:
        def post_nowait(b, stream):
            req = urllib.request.Request(ctx.URL, data=json.dumps(b).encode(),
                                         headers={"Authorization": "Bearer " + ctx.KEY, "Content-Type": "application/json",
                                                  "X-Strata-Wait": "0"}, method="POST")
            return urllib.request.urlopen(req, timeout=3600)
        ctx.post = post_nowait
    t_port = time.perf_counter()
    h = None
    while h is None:
        h = health()
        if h is None:
            time.sleep(0.05)
    t_port = time.perf_counter() - t_port
    ttft, dec, wall, up, uc, _, _ = ctx.stream_run(body(7703))
    h2 = health() or {}
    print(f"hold-test | sent at the port's first answer ({t_port:.2f} s after the probe began); /health then: "
          f"prewarm={h.get('prewarm')} hold_s={h.get('hold_s')}", flush=True)
    print(f"hold-test | {up or 0} prompt tokens | first token after {ttft if ttft else -1:6.2f} s | {uc or 0} tokens, "
          f"decode {uc / dec if dec else 0:5.1f} t/s | wall {wall:6.2f} s | /health after: prewarm={h2.get('prewarm')} "
          f"hold_s={h2.get('hold_s')}", flush=True)
    print(f"hold-test | TTFT {ttft if ttft else -1:.2f}s: the wait a live request met at the door"
          + (" (X-Strata-Wait: 0: no hold, the warm-in was pre-empted)" if no_wait else ""), flush=True)
    ttft2, dec2, wall2, _, uc2, _, _ = ctx.stream_run(body(8807))
    print(f"hold-test | again, a new session of the same shape | first token after {ttft2 if ttft2 else -1:5.2f} s | "
          f"{uc2 or 0} tokens, decode {uc2 / dec2 if dec2 else 0:5.1f} t/s | /health: prewarm={(health() or {}).get('prewarm')}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
