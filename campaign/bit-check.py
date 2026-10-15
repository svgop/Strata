#!/usr/bin/env python3
"""F-209 bit-identity check: fixed prompts at temperature 0, full text saved.
Usage: bit-check.py <label>  -> campaign/results/f209-bit-<label>.json"""
import hashlib, json, sys, time, urllib.request

KEY = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
API = "http://127.0.0.1:8081/v1/chat/completions"
PROMPTS = [
    ("p1", "Explain, step by step, why a wider grid with one warp per row can keep "
           "bitwise-identical output when lane chunk order is preserved. 250 words."),
    ("p2", "A train leaves at 13:37 travelling at 84.5 km/h. Ninety-one minutes later a "
           "second train leaves the same station at 102.3 km/h. When does it catch up? "
           "Show the arithmetic."),
    ("p3", "Write the first twelve lines of a Python function that merges two sorted "
           "lists in place, then explain its loop invariant."),
]

label = sys.argv[1] if len(sys.argv) > 1 else "x"
out = {}
for tag, p in PROMPTS:
    body = json.dumps({"messages": [{"role": "user", "content": p}],
                       "max_tokens": 320, "temperature": 0}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    blob = json.dumps(msg, sort_keys=True, ensure_ascii=False)
    txt = msg.get("content") if isinstance(msg.get("content"), str) else blob
    out[tag] = {"tok": d["usage"]["completion_tokens"], "sha": hashlib.sha256(txt.encode()).hexdigest(),
                "secs": round(time.time() - t0, 2)}
    print(f"{tag}: {out[tag]['tok']} tok, sha256 {out[tag]['sha'][:16]}..., {out[tag]['secs']}s")

path = f"/home/svg/LLM/strata-custom-2/campaign/results/f209-bit-{label}.json"
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("saved " + path)
