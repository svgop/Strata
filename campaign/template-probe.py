#!/usr/bin/env python3
# campaign/template-probe.py — THINKING-STYLE probe for the terse-dialect effort set (F-279).
# Measures what the chat-template effort directives actually change: completion tokens
# (thinking rides inline), wall time, and visible-answer style. 6 fixed reasoning prompts
# x N runs each against :8081. Run BEFORE a template change (live old-template server) and
# AFTER (new-template boot); judge on completion_tokens + latency means and a style read.
import json, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "probe"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
KEY = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"

PROMPTS = [
    "A train leaves A at 60 km/h; another leaves B, 420 km away, toward it at 90 km/h. A bird flies between them at 120 km/h starting at A. How far has the bird flown when the trains meet? Show the key step.",
    "This Python function is supposed to dedupe a list preserving order but is subtly wrong for unhashable items: def dedupe(xs): return list(dict.fromkeys(xs)). Explain the failure mode and give a correct version.",
    "Choose between a 3-hour flight plus 2 hours of airports and a 9-hour train with a work table, for someone with a 6-hour writing deadline due tonight. List the real decision variables and recommend one.",
    "Three switches downstairs control one bulb upstairs. You may go upstairs once. How do you determine which switch works?",
    "Estimate how many piano tuners work in a city of 2 million. State your assumptions compactly.",
    "Explain why cache invalidation is called hard, in a way a strong junior engineer would retell correctly.",
]

def ask(prompt, max_tokens=900):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.7}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ch = d.get("choices", [{}])[0]
    msg = ch.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = d.get("usage", {})
    return {"dt": dt, "completion": usage.get("completion_tokens"),
            "reasoning_chars": len(reasoning), "content_chars": len(content),
            "head": content[:160].replace("\n", " ")}

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
            print(f"[{i}.{r}] {res['dt']:6.1f}s  completion={res['completion']}  "
                  f"reasoning_chars={res['reasoning_chars']}  content_chars={res['content_chars']}")
            print(f"      head: {res['head']}")
    if rows:
        n = len(rows)
        print(f"\n=== TEMPLATE-PROBE {LABEL}: n={n}  "
              f"mean completion={sum(r['completion'] for r in rows)/n:.0f} tok  "
              f"mean wall={sum(r['dt'] for r in rows)/n:.1f}s  "
              f"mean content={sum(r['content_chars'] for r in rows)/n:.0f} chars ===")

if __name__ == "__main__":
    main()
