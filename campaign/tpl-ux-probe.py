#!/usr/bin/env python3
"""campaign/tpl-ux-probe.py — REAL-USAGE token-economy + wall probe for the template A/B
(F-283 'feels slower' question).  Unlike template-probe.py (puzzle-class reasoning), these
are daily-driver asks: explain, code review, capped writing, comparison, planning.  Measures
wall seconds, completion tokens (thinking rides inline), reasoning/content chars — the honest
'how long does an answer take' read.  max_tokens roomy on purpose: we want natural length.
"""
import json, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "ux"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
KEY = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"

PROMPTS = [
    "Explain how speculative decoding with a draft model speeds up LLM inference, and where its gains break down. For a senior engineer audience.",
    "Review this function and suggest improvements: def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "Draft a short product update email announcing that our inference engine hit a 30% speed improvement, for a technical but non-engineer audience. Keep it under 150 words.",
    "What are the main tradeoffs between PostgreSQL and SQLite for a small SaaS app?",
    "Help me plan a 3-day trip to Cape Town on a mid-range budget. Bullet points are fine.",
]

def ask(prompt, max_tokens=2500):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.7}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    msg = d.get("choices", [{}])[0].get("message", {})
    return {"dt": dt, "completion": d.get("usage", {}).get("completion_tokens"),
            "prompt": d.get("usage", {}).get("prompt_tokens"),
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "content_chars": len(msg.get("content") or "")}

def main():
    rows = []
    for i, p in enumerate(PROMPTS, 1):
        for r in range(1, RUNS + 1):
            try:
                res = ask(p)
            except Exception as e:
                print(f"[{i}.{r}] ERROR {e}")
                continue
            rows.append(res)
            print(f"[{i}.{r}] {res['dt']:5.1f}s  completion={res['completion']:5d}  "
                  f"reason={res['reasoning_chars']:5d}ch  content={res['content_chars']:5d}ch")
    if rows:
        n = len(rows)
        print(f"=== UX-PROBE {LABEL}: n={n}  mean wall={sum(r['dt'] for r in rows)/n:.1f}s  "
              f"mean completion={sum(r['completion'] for r in rows)/n:.0f} tok  "
              f"mean content={sum(r['content_chars'] for r in rows)/n:.0f} ch  "
              f"mean reason={sum(r['reasoning_chars'] for r in rows)/n:.0f} ch ===")

if __name__ == "__main__":
    main()
