#!/usr/bin/env python3
# campaign/ctx-probe.py — long-context SINGLE-STREAM ladder: prefill t/s (TTFT of a streamed
# request) and steady decode t/s (token cadence after the first chunk) at increasing context
# depths.  Unique seeded random-word prompts (no n-gram/cache reuse across runs).  Usage:
#   python ctx-probe.py 4096 16384 32768 65536 98304 122880      (target prompt tokens)
#   python ctx-probe.py --selftest          (offline: the request builder and the stream reader; no server)
#   python ctx-probe.py --turn  4096 32768 131072
#   python ctx-probe.py --branch 4096 32768 131072
#   python ctx-probe.py --rotate 20 33000
# --rotate N (F-467): the conversation cache's working set - N conversations of the one depth, each created cold,
# then one agent turn on each.  A cache that cannot hold them re-reads the oldest from the prompt.
# --turn (F-466): after each depth, one AGENT TURN - the ladder's reply is sent back (content and
# reasoning), then a ~300-token user message, max_tokens 8.  This is the production shape (the engine log
# shows 2,527 of the next prompts extending the previous prompt plus its generation), so the conversation
# cache serves the whole history and the engine reads only the new tail.  Its TTFT is the wait between two
# tool calls.
# --branch (F-466): the same depth with a ~300-token tail appended and NO reply - a sibling prompt (a history
# edit).  Measured 2026-11-25: below 32,768 tokens nothing is reused; above it the last whole 32,768-token
# checkpoint is, and the rest of the prompt is read again.
# The engine's own accounting of every request (tokens reused and read, read ms, drafts accepted) is the
# `strata serve: prompt` line in the boot's segment of the engine log; campaign/ctx-lane.sh prints them.
import os
import json, random, string, sys, time, urllib.request

KEY = os.environ.get("STRATA_API_KEY", "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8")
URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "qwen3.8-flash-next-iq3_s"
GEN = 512
INSTR = " After reading it, write the numbers from 1 to 400, one per line."
TAIL_TOKENS = 300        # a tool result's size; the production median of a warm read is 288 tokens (F-466)
PARK_SETTLE_S = 2.0      # a finished request's K/V is parked after it returns; the next request must find it parked


def words(n, seed):
    rng = random.Random(seed)
    return " ".join("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(n))


def gen_prompt(n_words, seed):
    # the draws keep their original order (words, then the header) so a seed still names the same prompt
    rng = random.Random(seed)
    w = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(n_words)]
    return "X" + "".join(rng.choices(string.hexdigits.upper(), k=12)) + " document: " + " ".join(w) + INSTR


def gen_tail(n_words, seed):
    return "New instruction, answer at once: " + words(n_words, seed)


def post(body, stream):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def stream_run(body, extra=None):
    """One streamed request -> (ttft s or None, first-to-last-token s or None, wall s, prompt tokens,
    completion tokens, the answer's content, its reasoning).  With a dict as `extra` (F-467) it also fills
    extra["finish"] (the stream's finish_reason) and extra["tool_calls"] (the tool calls' names and arguments
    as sent), which the seven-tuple has no room for."""
    t0 = time.perf_counter()
    ttft = None
    t_last = 0.0
    chunks = 0
    usage = {}
    text, reasoning, tools = [], [], []
    finish = None
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
                if ch[0].get("finish_reason"):
                    finish = ch[0]["finish_reason"]
                for tc in d.get("tool_calls") or ():
                    fn = tc.get("function") or {}
                    tools.append((fn.get("name") or "", fn.get("arguments") or ""))
                # a tool-call delta is a token too: a stream may open with one (no content, no reasoning)
                if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
                    t_last = time.perf_counter() - t0
                    if d.get("content"):
                        text.append(d["content"])
                    if d.get("reasoning_content"):
                        reasoning.append(d["reasoning_content"])
    total = time.perf_counter() - t0
    dec = (t_last - ttft) if ttft is not None and t_last > ttft else None
    if extra is not None:
        extra["finish"] = finish
        extra["tool_calls"] = tools
    return (ttft, dec, total, usage.get("prompt_tokens"), usage.get("completion_tokens"),
            "".join(text), "".join(reasoning))


def calibrate():
    # words -> tokens ratio, measured once (non-stream gives usage.prompt_tokens)
    with post({"model": MODEL, "messages": [{"role": "user", "content": gen_prompt(2000, 1)}],
               "max_tokens": 1}, stream=False) as r:
        u = json.load(r).get("usage", {})
    pt = u.get("prompt_tokens", 0)
    return pt / 2000.0 if pt else 1.4, pt


def ladder(target_tokens, ratio, seed0):
    """A cold read at one depth: TTFT, prefill t/s, steady decode t/s.  Returns (prompt tokens, reply)."""
    n_words = int(target_tokens / ratio)
    body = {"model": MODEL, "messages": [{"role": "user", "content": gen_prompt(n_words, seed0)}],
            "max_tokens": GEN, "stream": True}
    ttft, dec_s, total, up, uc, text, reasoning = stream_run(body)
    ptok = up or int(n_words * ratio)
    ctok = uc or GEN
    print(f"ctx {ptok:7d} | TTFT {ttft if ttft else -1:7.2f}s | prefill {ptok / ttft if ttft else 0:7.0f} t/s"
          f" | decode {ctok / dec_s if dec_s and dec_s > 0 else 0:6.1f} t/s | gen {ctok} in {dec_s if dec_s else 0:5.2f}s"
          f" | wall {total:7.2f}s", flush=True)
    return ptok, (text, reasoning)


def report(kind, ptok, ttft, total, ctok, tail_tokens):
    print(f"ctx {ptok if ptok else -1:7d} | {kind:6s} +{tail_tokens:4d} tokens | TTFT {ttft if ttft else -1:6.2f}s"
          f" | gen {ctok if ctok is not None else -1} | wall {total:5.2f}s", flush=True)


def turn_read(target_tokens, ratio, seed0, depth_ptok, reply):
    """An agent turn: the reply comes back (content and reasoning), then one short user message."""
    time.sleep(PARK_SETTLE_S)
    n_words = int(target_tokens / ratio)
    text, reasoning = reply
    messages = [{"role": "user", "content": gen_prompt(n_words, seed0)},
                {"role": "assistant", "content": text, **({"reasoning_content": reasoning} if reasoning else {})},
                {"role": "user", "content": gen_tail(int(TAIL_TOKENS / ratio), seed0 + 1)}]
    ttft, _, total, ptok, ctok, _, _ = stream_run({"model": MODEL, "messages": messages, "max_tokens": 8, "stream": True})
    report("turn", ptok, ttft, total, ctok, TAIL_TOKENS)


def branch_read(target_tokens, ratio, seed0, depth_ptok):
    """A sibling prompt: the same depth with a short tail appended, no reply."""
    time.sleep(PARK_SETTLE_S)
    n_words = int(target_tokens / ratio)
    prompt = gen_prompt(n_words, seed0) + " " + gen_tail(int(TAIL_TOKENS / ratio), seed0 + 1)
    ttft, _, total, ptok, ctok, _, _ = stream_run({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                                   "max_tokens": 8, "stream": True})
    report("branch", ptok, ttft, total, ctok, (ptok - depth_ptok) if ptok and depth_ptok else -1)


def rotate(target_tokens, ratio, seed0, n_sessions):
    """The conversation cache's working set (F-467): n_sessions conversations of one depth, each created cold,
    then one agent turn on each in the same order.  A cache that cannot hold them re-reads the oldest from the
    prompt: the operator's production failure (F-466: 201 re-reads of 17-27 s in one boot).  A turn counts as
    re-read when its TTFT is more than 40 % of its conversation's cold read; the engine's accounting lines in the
    boot's log say reused and read tokens exactly."""
    n_words = int(target_tokens / ratio)
    tail_words = int(TAIL_TOKENS / ratio)
    created = []
    for i in range(n_sessions):
        prompt = gen_prompt(n_words, seed0 + i)
        ttft, _, _, ptok, _, text, reasoning = stream_run({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                                           "max_tokens": 8, "stream": True})
        created.append((prompt, (text, reasoning), ttft))
        print(f"session {i:3d} | {ptok if ptok else -1:7d} tokens | created cold, TTFT {ttft if ttft else -1:6.2f}s", flush=True)
        time.sleep(PARK_SETTLE_S)
    served, reread, turn_ttfts = [], [], []
    for i, (prompt, reply, cold) in enumerate(created):
        time.sleep(PARK_SETTLE_S)
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": reply[0], **({"reasoning_content": reply[1]} if reply[1] else {})},
                    {"role": "user", "content": gen_tail(tail_words, seed0 + 5000 + i)}]
        ttft, _, _, _, _, _, _ = stream_run({"model": MODEL, "messages": messages, "max_tokens": 8, "stream": True})
        ttft = ttft if ttft else -1.0
        turn_ttfts.append(ttft)
        again = cold is not None and cold > 0 and ttft > 0.4 * cold
        (reread if again else served).append(i)
        print(f"session {i:3d} | turn | TTFT {ttft:6.2f}s {'RE-READ' if again else 'served'}", flush=True)
    ok = sorted(t for t in turn_ttfts if t >= 0)
    print(f"rotate {n_sessions} sessions of ~{target_tokens} tokens: {len(served)} turns served, {len(reread)} re-read"
          f"{' ' + str(tuple(reread)) if reread else ''}; turn TTFT {sum(ok):.1f} s in all, median {ok[len(ok) // 2]:.2f}s, "
          f"max {ok[-1]:.2f}s", flush=True)


def selftest():
    """Offline check of the request builder and the stream reader against a canned answer (no server)."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": "x"}], "max_tokens": 8, "stream": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    assert req.full_url == URL and json.loads(req.data)["stream"] is True

    class Fake:
        def __init__(self):
            self.lines = iter([b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n',
                               b'data: {"choices":[{"delta":{"content":"ok"}}],"usage":{"prompt_tokens":1234,"completion_tokens":2}}\n',
                               b'data: [DONE]\n'])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return self.lines

    real = urllib.request.urlopen
    urllib.request.urlopen = lambda request, timeout: Fake()
    try:
        ttft, dec, total, ptok, ctok, text, reasoning = stream_run(body)
    finally:
        urllib.request.urlopen = real
    assert ttft is not None and ttft >= 0 and ctok == 2 and ptok == 1234 and text == "ok" and reasoning == "think", \
        (ttft, dec, total, ptok, ctok, text, reasoning)
    print("ctx-probe selftest: ok")


def main():
    argv = sys.argv[1:]
    if "--selftest" in argv:
        return selftest()
    rotate_n = None
    if "--rotate" in argv:
        i = argv.index("--rotate")
        rotate_n = int(argv[i + 1])
        del argv[i:i + 2]
    after = [a for a in ("--turn", "--branch") if a in argv]
    args = [a for a in argv if not a.startswith("--")]
    sizes = [int(x) for x in args] or [4096, 16384, 32768, 65536, 98304, 122880]
    if rotate_n is not None and len(sizes) != 1:
        raise SystemExit("--rotate N takes exactly one depth: ctx-probe.py --rotate 20 33000")
    ratio, pt = calibrate()
    print(f"# calibration: 2000 words -> {pt} tokens (ratio {ratio:.4f}); gen {GEN}/run; after each depth: {after[0] if after else 'nothing'}",
          flush=True)
    for i, t in enumerate(sizes):
        seed0 = 100 + i * 7919
        if rotate_n is not None:
            rotate(t, ratio, seed0, rotate_n)
            continue
        ptok, reply = ladder(t, ratio, seed0)
        if after:
            if after[0] == "--turn":
                turn_read(t, ratio, seed0, ptok, reply)
            else:
                branch_read(t, ratio, seed0, ptok)
    print("# done", flush=True)


if __name__ == "__main__":
    main()
