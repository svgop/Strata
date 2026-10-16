#!/usr/bin/env python3
# campaign/ctx-probe.py — long-context SINGLE-STREAM ladder: prefill t/s (TTFT of a streamed
# request) and steady decode t/s (token cadence after the first chunk) at increasing context
# depths.  Unique seeded random-word prompts (no n-gram/cache reuse across runs).  Usage:
#   python ctx-probe.py 4096 16384 32768 65536 98304 122880      (target prompt tokens)
import json, random, string, sys, time, urllib.request

KEY = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
URL = "http://127.0.0.1:8081/v1/chat/completions"
GEN = 512


def gen_prompt(n_words, seed):
    rng = random.Random(seed)
    words = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(n_words)]
    return ("X" + "".join(rng.choices(string.hexdigits.upper(), k=12)) + " document: " + " ".join(words)
            + " After reading it, write the numbers from 1 to 400, one per line.")


def post(body, stream):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def calibrate():
    # words -> tokens ratio, measured once (non-stream gives usage.prompt_tokens)
    with post({"model": "qwen3.8-flash-next-iq3_s",
               "messages": [{"role": "user", "content": gen_prompt(2000, 1)}],
               "max_tokens": 1}, stream=False) as r:
        u = json.load(r).get("usage", {})
    pt = u.get("prompt_tokens", 0)
    return pt / 2000.0 if pt else 1.4, pt


def ladder(target_tokens, ratio, seed0):
    n_words = int(target_tokens / ratio)
    body = {"model": "qwen3.8-flash-next-iq3_s",
            "messages": [{"role": "user", "content": gen_prompt(n_words, seed0)}],
            "max_tokens": GEN, "stream": True}
    t0 = time.perf_counter()
    ttft = chunks = None
    t_last = 0.0
    usage = {}
    with post(body, stream=True) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                continue
            try:
                obj = json.loads(p)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks = (chunks or 0) + 1
                    t_last = time.perf_counter() - t0
    total = time.perf_counter() - t0
    ptok = usage.get("prompt_tokens") or int(n_words * ratio)
    ctok = usage.get("completion_tokens") or chunks or GEN
    dec_s = t_last - ttft if ttft and t_last > ttft else float("nan")
    print(f"ctx {ptok:7d} | TTFT {ttft if ttft else -1:7.2f}s | prefill {ptok / ttft if ttft else 0:7.0f} t/s"
          f" | decode {ctok / dec_s if dec_s == dec_s and dec_s > 0 else 0:6.1f} t/s | gen {ctok} in {dec_s if dec_s == dec_s else 0:5.2f}s"
          f" | wall {total:7.2f}s", flush=True)


def main():
    sizes = [int(x) for x in sys.argv[1:]] or [4096, 16384, 32768, 65536, 98304, 122880]
    ratio, pt = calibrate()
    print(f"# calibration: 2000 words -> {pt} tokens (ratio {ratio:.4f}); gen {GEN}/run", flush=True)
    for i, t in enumerate(sizes):
        ladder(t, ratio, 100 + i * 7919)
    print("# done", flush=True)


if __name__ == "__main__":
    main()
