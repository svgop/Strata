#!/usr/bin/env python3
"""campaign/t43-smoke.py — F-284 ternary SAFETY SMOKE: 12 concurrent deterministic
known-answer requests (9 slots + 3 queued), every answer must contain its expected
substring, zero server errors.  Prints PASS/FAIL."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT = "8081"
KEY = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"

QA = [
    ("What is 6 times 7? Answer with just the number.", "42"),
    ("What is the capital of France? One word.", "Paris"),
    ("What is 9 squared? Just the number.", "81"),
    ("How many days are in a leap year? Just the number.", "366"),
    ("What is 15 percent of 200? Just the number.", "30"),
    ("Name the chemical symbol for gold. Just the symbol.", "Au"),
    ("What is 3 to the power of 4? Just the number.", "81"),
    ("How many continents are there? Just the number.", "7"),
    ("What is the square root of 144? Just the number.", "12"),
    ("Name the largest planet in our solar system. One word.", "Jupiter"),
    ("What is 100 minus 37? Just the number.", "63"),
    ("What color do you get mixing red and white? One word.", "pink"),
]

def ask(i):
    q, want = QA[i]
    body = json.dumps({"messages": [{"role": "user", "content": q}],
                       "max_tokens": 300, "temperature": 0, "reasoning_effort": "low"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        content = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        ok = want.lower() in content.lower()
        return i, ok, want, content[:60], time.time() - t0
    except Exception as e:
        return i, False, want, f"ERROR {e}", time.time() - t0

with ThreadPoolExecutor(max_workers=12) as ex:
    results = list(ex.map(ask, range(len(QA))))

fails = [r for r in results if not r[1]]
for i, ok, want, got, dt in results:
    print(f"[{i:2d}] {'PASS' if ok else 'FAIL'} ({dt:5.1f}s) want={want!r} got={got!r}")
print(f"SMOKE: {len(results) - len(fails)}/{len(results)} correct, "
      f"mean wall {sum(r[4] for r in results)/len(results):.1f}s")
print("SMOKE PASS" if not fails else "SMOKE FAIL")
sys.exit(0 if not fails else 1)
