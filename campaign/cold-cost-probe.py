#!/usr/bin/env python3
# campaign/cold-cost-probe.py — what the FIRST request of each production shape costs after the server's
# warm-in, against the same shape run again at once (F-467).  Run it as the first thing after /health's
# "prewarm" word turns terminal (campaign/ctx-lane.sh <label> -- .venv/bin/python campaign/cold-cost-probe.py):
# anything sent before it warms what it measures.
#
# Why it exists: law 11 discards a battery's first leg because it is cold, which also throws away the only
# direct look at the cold cost the operator pays after every restart.  Across 20 boots in 6 batteries the
# discarded leg read 0 to -12.7 % below the measured ones on M1 (campaign/results/f467-2026-11-25/notes.md).
# This probe reads that cost BY SHAPE, first against again, on the consumer's own clock (time to first token,
# wall seconds, tokens a second) - not on the engine's accounting line: under --batch that line is printed
# at a request's ADMISSION and carries one generated token for everything the batch lane then generates
# (generate.cpp:17473; F-467).
#
# Shapes, each run twice ("first" is cold, "again" follows at once), in the order the warm-in should cover them:
#   solo     a new session, a ~3.8k-token prompt, 256 tokens            TTFT, decode t/s
#   tool     a new session, two tool schemas, a ~1.3k-token task, 160 tokens   TTFT, the tool calls' names
#   burstN   N concurrent new sessions, ~3.8k-token prompts, 200 tokens each (N = 2..6)   aggregate t/s, mean TTFT
#   deep     three ~97.5k-token cold reads (distinct documents; the production boot's median context), then two
#            agent turns on the last                                           TTFT each
# Sizes are in words, not tokens: this tokenizer cuts random letter-strings into 3.38 tokens a word (measured
# 2026-11-25: 800 words made a 3,832-token prompt, the ~1,110-token system prefix included), and the rows print the
# server's own prompt-token count.
# The request helpers are campaign/ctx-probe.py's (the owner): stream_run, gen_prompt, calibrate.
import importlib.util
import pathlib
import sys
import threading
import time

spec = importlib.util.spec_from_file_location("ctx_probe", pathlib.Path(__file__).with_name("ctx-probe.py"))
ctx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctx)

SOLO_WORDS = 800           # a ~3.8k-token prompt with the server's system prefix
DEEP_WORDS = 28571         # a ~97.5k-token prompt: past the 32,768-cell VRAM K/V window, the read that streams K/V
TURN_TAIL_WORDS = 76       # ~260 tokens: ctx-probe.py's agent-turn tail at its own measured ratio
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Today's weather in a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "get_time", "description": "The local time in a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
WORSE = {}                 # shape -> (first, again) for the summary


def row(shape, label, text):
    print(f"cold-cost | {shape:7s} | {label:5s} | {text}", flush=True)


def solo(seed):
    extra = {}
    ttft, dec, wall, up, uc, _, _ = ctx.stream_run({"model": ctx.MODEL, "stream": True, "max_tokens": 256,
                                                    "messages": [{"role": "user", "content": ctx.gen_prompt(SOLO_WORDS, seed)}]}, extra)
    return ttft, (uc / dec if dec else None), wall, uc, extra, up


def tool(seed):
    extra = {}
    body = {"model": ctx.MODEL, "stream": True, "max_tokens": 160, "tools": TOOLS, "tool_choice": "auto",
            "messages": [{"role": "system", "content": "You are an assistant that calls tools for facts you do not have."},
                         {"role": "user", "content": "Background, for context only: " + ctx.words(300, seed)
                          + "\n\nUse the tools to get today's weather in Cape Town and the local time in Nairobi, then tell me both in one sentence."}]}
    ttft, dec, wall, up, uc, _, _ = ctx.stream_run(body, extra)
    names = sorted({n for n, _ in extra["tool_calls"]})
    return ttft, wall, uc, extra["finish"], "+".join(names) or "-"


def burst(n, seed):
    out, t0 = [], time.perf_counter()

    def one(i):
        out.append(ctx.stream_run({"model": ctx.MODEL, "stream": True, "max_tokens": 200,
                                   "messages": [{"role": "user", "content": ctx.gen_prompt(SOLO_WORDS, seed + i)}]}))
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    ttfts = [r[0] for r in out if r[0] is not None]
    toks = sum(r[4] or 0 for r in out)
    return toks / wall if wall else None, (sum(ttfts) / len(ttfts) if ttfts else None), wall, toks


def deep(seed0):
    made = []
    for i in range(3):
        prompt = ctx.gen_prompt(DEEP_WORDS, seed0 + i)
        r = ctx.stream_run({"model": ctx.MODEL, "stream": True, "max_tokens": 8,
                            "messages": [{"role": "user", "content": prompt}]})
        made.append((prompt, r))
        row("deep", f"read{i + 1}", f"{r[3] or 0:6d} prompt tokens | TTFT {r[0] if r[0] else -1:6.2f}s")
        time.sleep(ctx.PARK_SETTLE_S)
    prompt, r = made[-1]
    turns = []
    for k in range(2):
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": r[5], **({"reasoning_content": r[6]} if r[6] else {})},
                    {"role": "user", "content": ctx.gen_tail(TURN_TAIL_WORDS * (k + 1), seed0 + 900 + k)}]
        t = ctx.stream_run({"model": ctx.MODEL, "stream": True, "max_tokens": 8, "messages": messages})
        turns.append(t[0])
        row("deep", f"turn{k + 1}", f"{t[3] or 0:6d} prompt tokens | TTFT {t[0] if t[0] else -1:6.2f}s")
        time.sleep(ctx.PARK_SETTLE_S)
    return turns


def pct(first, again):
    """first against again in percent, or n/a when either is missing."""
    return f"{100 * (first / again - 1):+.0f} %" if (first and again) else "n/a"


def main():
    # no calibration request: it would be the first request of all and warm the path this probe measures cold
    print(f"# cold-cost probe: {SOLO_WORDS} words a solo or burst prompt, {DEEP_WORDS} a deep one; "
          f"load {open('/proc/loadavg').read().split()[0]}", flush=True)
    for label, seed in (("first", 1103), ("again", 2207)):
        ttft, rate, wall, uc, _, up = solo(seed)
        row("solo", label, f"{up or 0:5d} prompt tokens | TTFT {ttft:5.2f}s | {uc or 0:3d} tokens, decode {rate:5.1f} t/s | wall {wall:5.2f}s")
        WORSE.setdefault("solo", {})[label] = (ttft, rate)
    for label, seed in (("first", 3301), ("again", 4409)):
        ttft, wall, uc, finish, names = tool(seed)
        row("tool", label, f"TTFT {ttft if ttft else -1:5.2f}s | finish {finish} | tools {names} | {uc or 0:3d} tokens | wall {wall:5.2f}s")
        WORSE.setdefault("tool", {})[label] = (ttft, wall)
    for n in (2, 3, 4, 5, 6):
        for label, seed in (("first", 5501 + 100 * n), ("again", 6607 + 100 * n)):
            agg, mttft, wall, toks = burst(n, seed)
            row(f"burst{n}", label, f"aggregate {agg:6.1f} t/s | mean TTFT {mttft:5.2f}s | {toks:4d} tokens in {wall:5.2f}s")
            WORSE.setdefault(f"burst{n}", {})[label] = (agg, mttft)
    turns = deep(7703)
    print("# summary - the cold cost left after the warm-in (first against again; a first more than 10 % worse is named):",
          flush=True)
    left = []
    for shape, d in WORSE.items():
        f, a = d["first"], d["again"]
        if shape.startswith("burst"):
            verdict = f"aggregate {f[0] or 0:.0f} vs {a[0] or 0:.0f} t/s ({pct(f[0], a[0])}), mean TTFT {f[1] or 0:.2f} vs {a[1] or 0:.2f} s"
            if a[0] and f[0] and f[0] < 0.9 * a[0]:
                left.append(shape)
        elif shape == "solo":
            verdict = f"TTFT {f[0] or 0:.2f} vs {a[0] or 0:.2f} s, decode {f[1] or 0:.0f} vs {a[1] or 0:.0f} t/s ({pct(f[1], a[1])})"
            if a[1] and f[1] and f[1] < 0.9 * a[1]:
                left.append(shape)
        else:
            verdict = f"TTFT {f[0] or 0:.2f} vs {a[0] or 0:.2f} s ({pct(f[0], a[0])})"
            if f[0] and a[0] and f[0] > 1.1 * a[0]:
                left.append(shape)
        print(f"#   {shape:7s} {verdict}", flush=True)
    print(f"# deep turns' TTFT {turns[0] or 0:.2f}s then {turns[1] or 0:.2f}s", flush=True)
    print(f"# cold after the warm-in: {', '.join(left) if left else 'nothing over 10 %'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
