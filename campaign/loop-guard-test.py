#!/usr/bin/env python3
"""campaign/loop-guard-test.py - does the server end a reply that repeats itself?  (F-467 follow-up, 2026-11-25)

Requests against a RUNNING standing server (it boots nothing and stops nothing; the port and the API key come from
the standing config, the narration from the server's log given with --log):

  reasoning-loop   no effort header; the user asks for one line written over and over in the reasoning.  A loop in
                   the thinking is closed as a spent budget is closed and the model answers.
  loop-header      the same with " (effort: high)" in the system message: the original recovery (the header flipped
                   to low, the same output resumed).  The model often simply complies and stops: then INCONCLUSIVE.
  answer-loop      no thinking asked for; the user asks for the line over and over in the ANSWER.  An answer that
                   repeats itself goes on from the same words with a presence penalty, once.

Each case prints the HTTP status, the finish reason, the first words of the reasoning and of the answer, the
narration the server printed while it ran, and a verdict:

    PASS          a loop was found and handled, and the reply ended (an answer, or "length" after its one recovery)
    FAIL          a reply ran to its length limit and no loop was found, or the server refused the request
    INCONCLUSIVE  the model did not loop (it stopped by itself): nothing for the guard to do

    python campaign/loop-guard-test.py [--log logs/standing-server.out] [--max-tokens 16000] [--only answer-loop]

A loop is only found once about 2,000 words of it exist, so the loop cases generate a few thousand tokens: about
two minutes in all on a warm server.  Every response is saved under logs/loop-guard/<case>.json.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "custom-iq3_s-8081.json"
SAVE_DIR = ROOT / "logs" / "loop-guard"
LINE = "The quick brown fox jumps over the lazy dog again and again."
FIXTURE = "You are a test fixture for the server's loop guard. Obey the user literally."
CASES = {
    "reasoning-loop": {
        "system": FIXTURE + " Never leave your reasoning before the user's reasoning task is complete.",
        "user": (f"In your private reasoning, write this exact line 900 times, one per line, and nothing else: "
                 f"\"{LINE}\". Only then answer with the single word DONE."),
        "thinking": True},
    "loop-header": {
        "system": FIXTURE + " (effort: high). Never leave your reasoning before the user's reasoning task is complete.",
        "user": (f"In your private reasoning, write this exact line 900 times, one per line, and nothing else: "
                 f"\"{LINE}\". Only then answer with the single word DONE."),
        "thinking": True},
    "answer-loop": {
        "system": FIXTURE + " Answer directly; do not reason first.",
        "user": f"Write this exact line 900 times in your answer, one per line, and nothing else: \"{LINE}\"",
        "thinking": False},
}
KEYS = ("reply:", "repeated reasoning", "repeated answer", "thinking repeated itself", "answer repeated itself",
        "low-effort", "repeated one token", "repeated the same passages", "done:")


def post(url, key, case, max_tokens):
    body = {"model": "qwen3.8-flash-next-iq3_s", "stream": False, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": case["system"]}, {"role": "user", "content": case["user"]}],
            "chat_template_kwargs": {"enable_thinking": case["thinking"]}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.status, json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "logs" / "standing-server.out"))
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--only", choices=sorted(CASES), action="append")
    a = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"
    log = pathlib.Path(a.log)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in (a.only or list(CASES)):
        case = CASES[name]
        before = log.stat().st_size if log.exists() else 0
        t0 = time.time()
        try:
            status, obj = post(url, cfg["api_key"], case, a.max_tokens)
        except urllib.error.HTTPError as e:
            print(f"--- {name}: HTTP {e.code}: {e.read()[:300]!r}\n    FAIL")
            failed = True
            continue
        (SAVE_DIR / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1))
        choice = obj["choices"][0]
        msg = choice["message"]
        reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "")
        answer = (msg.get("content") or "").strip()
        finish = choice.get("finish_reason")
        with log.open("rb") as f:
            f.seek(before)
            said = [l for l in (raw.decode("utf-8", "replace").rstrip() for raw in f.read().split(b"\n"))
                    if "[strata]" in l and any(k in l for k in KEYS)]
        print(f"--- {name}: HTTP {status}, finish {finish}, {obj['usage']['completion_tokens']} tokens in {time.time() - t0:.0f} s")
        print(f"    reasoning: {len(reasoning.split())} words, starts {reasoning[:66]!r}")
        print(f"    answer:    {len(answer.split())} words, starts {answer[:96]!r}")
        for l in said[:9]:
            print("    " + l[:168])
        found = any(("repeated reasoning" in l or "repeated answer" in l) and "detected" in l for l in said)
        handled = any(k in l for l in said for k in ("thinking repeated itself", "answer repeated itself", "low-effort"))
        again = any("repeated the same passages" in l for l in said)
        if found and handled and (finish == "stop" or (finish == "length" and again)):
            verdict = "PASS"
        elif finish == "length" and not found:
            verdict = "FAIL"            # a reply ran to its limit and the guard saw nothing
        elif not found:
            verdict = "INCONCLUSIVE"    # the model did not loop
        else:
            verdict = "FAIL"
            failed = True
        failed = failed or verdict == "FAIL"
        print(f"    {verdict}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
