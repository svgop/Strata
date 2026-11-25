#!/usr/bin/env python3
"""campaign/loop-propensity.py - how often does a reply fall into a loop, per arm?  (F-468 follow-up, 2026-11-25)

The loop guard is the seatbelt; this measures what makes the model loop.  Against a RUNNING standing server (it
boots nothing and stops nothing): `--prompts N` long-document review tasks, each sent under every arm in turn
(interleaved: prompt, arm, arm, ...), thinking on.  Every reply is scored:

    loop     a quarter or more of some 2,000-word window of the reply sits inside 12-word passages seen three
             times (the guard's rule, evaluated every 500 words: the reply's worst window)
    ok       anything else

Arms are request fields, so they need no reboot: "standing" (the config's own sampling), "cvec-off"
(`experimental_speed_projection: false`: both control vectors off for that request), "temp-0.6",
"presence-1.5", "effort-max", "effort-high", "effort-low" (the template's effort text for that level: every
request here sets `enable_thinking`, so it carries no effort unless the arm names one).  `--arms` picks a subset; `--extra arm='{json}'` adds one (e.g. '{"temperature": 0.6,
"presence_penalty": 1.5}').

    python campaign/loop-propensity.py [--prompts 3] [--max-tokens 12000] [--arms standing,cvec-off] [--repeat 2]

A reply's verdict comes from the text it produced alone (the guard's cut cannot hide a loop), not from a client's
impression: the most repeated line and the
12-word passage coverage of its last 2,000 words are printed with every reply.  Responses are saved under
logs/loop-propensity/<time>/<prompt>-<arm>-<r>.json.  One run of 3 prompts x 4 arms is about ten minutes.
"""
import argparse
import collections
import json
import pathlib
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "custom-iq3_s-8081.json"
SAVE_ROOT = ROOT / "logs" / "loop-propensity"
DOCS = ["src/core/expert_source.cpp", "src/program/generate.cpp", "serve/server.py", "src/prefill/prefill.cpp"]
DOC_CHARS = 60000
TASK = ("Here is part of a program. Go through it function by function, in order, and in your reasoning state one "
        "line for each of the form '- In `name`, if <its preconditions and state>? OK' or the issue you see. Be "
        "exhaustive: do not skip a function and do not summarise before the end. After the last one, name the "
        "three worst issues in three sentences.")
ARMS = {"standing": {}, "cvec-off": {"experimental_speed_projection": False}, "temp-0.6": {"temperature": 0.6},
        "presence-1.5": {"presence_penalty": 1.5}, "effort-max": {"reasoning_effort": "max"},
        "effort-high": {"reasoning_effort": "high"}, "effort-low": {"reasoning_effort": "low"}}
GUARD = ("repeated reasoning detected", "repeated answer detected", "repeated the same passages",
         "thinking repeated itself", "answer repeated itself", "low-effort")


def peak_coverage(text):
    """The server's own rule (serve/server.py reasoning_repeat_coverage: 12-word passages seen three times covering
    the last 2,000 words), evaluated at the end of every 500 words and at the very end: the reply's worst window.  A
    reply that ends in a loop scores high here whether or not the guard cut it."""
    words = re.findall(r"\w+|[^\w\s]", text.lower())[-30000:]
    if len(words) < 2000:
        return 0.0
    grams = [tuple(words[i:i + 12]) for i in range(len(words) - 11)]
    counts = collections.Counter(grams)
    best = 0.0
    for end in list(range(2000, len(words), 500)) + [len(words)]:
        start = end - 2000
        covered = set()
        for i in range(start, end - 11):
            if counts[grams[i]] >= 3:
                covered.update(range(i, i + 12))
        best = max(best, len(covered) / 2000)
    return best


def most_repeated(text):
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 25]
    return collections.Counter(lines).most_common(1)[0] if lines else ("", 0)


def post(url, key, messages, max_tokens, arm):
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": messages, "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": True}, **arm}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--arms", default="standing,cvec-off,temp-0.6,presence-1.5")
    ap.add_argument("--extra", action="append", default=[], help="name='{json}'")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--log", default=str(ROOT / "logs" / "standing-server.out"))
    a = ap.parse_args()
    arms = dict(ARMS)
    for spec in a.extra:
        name, _, js = spec.partition("=")
        arms[name] = json.loads(js)
    names = [n for n in a.arms.split(",") if n]
    cfg = json.loads(CONFIG.read_text())
    url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"
    log = pathlib.Path(a.log)
    save = SAVE_ROOT / time.strftime("%Y%m%d-%H%M%S")
    save.mkdir(parents=True, exist_ok=True)
    docs = [(ROOT / d).read_text(errors="replace")[:DOC_CHARS] for d in DOCS[:a.prompts]]
    tally = {n: [] for n in names}
    sizes = {n: [] for n in names}
    for p, doc in enumerate(docs):
        for r in range(a.repeat):
            for name in names:
                messages = [{"role": "user", "content": TASK + "\n\n```c++\n" + doc + "\n```"}]
                before = log.stat().st_size
                t0 = time.time()
                try:
                    obj = post(url, cfg["api_key"], messages, a.max_tokens, arms[name])
                except urllib.error.HTTPError as e:
                    print(f"prompt {p} {name:12s} HTTP {e.code}: {e.read()[:200]!r}")
                    tally[name].append("error")
                    continue
                ch = obj["choices"][0]
                msg = ch["message"]
                text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")
                (save / f"p{p}-{name}-{r}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1))
                with log.open("rb") as f:
                    f.seek(before)
                    said = [l for l in (raw.decode("utf-8", "replace") for raw in f.read().split(b"\n"))
                            if "[strata]" in l and any(k in l for k in GUARD)]
                n = obj["usage"]["completion_tokens"]
                cov = peak_coverage(text)
                guard = any("detected" in l or "repeated the same passages" in l for l in said)
                verdict = "loop" if cov >= 0.25 else "ok"     # from the text alone: the guard's own cut cannot hide it
                tally[name].append(verdict)
                sizes[name].append(n)
                line, times = most_repeated(text)
                print(f"prompt {p} {name:12s} {verdict:4s} finish {ch.get('finish_reason'):6s} {n:6d} tok "
                      f"{time.time() - t0:3.0f} s  peak {cov:.2f} guard {'yes' if guard else 'no '} most repeated x{times}: {line[:60]!r}", flush=True)
    print("--- loops / replies, and the median tokens of a reply:")
    for name in names:
        v = tally[name]
        med = f"{statistics.median(sizes[name]):.0f}" if sizes[name] else "-"
        print(f"  {name:12s} {v.count('loop')} / {len(v)} loops, median reply {med} tokens" + (f" ({v.count('error')} errors)" if "error" in v else ""))
    print(f"responses: {save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
