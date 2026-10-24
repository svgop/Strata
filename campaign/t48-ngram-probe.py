#!/usr/bin/env python3
"""campaign/t48-ngram-probe.py — the context-n-gram draft falsifier (T47-solo-300's
never-attempted candidate).  For each probe request: (a) the engine's OWN MTP acceptance
(the per-request 'drafts accepted X of Y' log line, matched by timestamp), and (b) the
PROMPT-LOOKUP CEILING: the fraction of the response's tokens an n-gram matcher (n=4..8)
over the preceding context would have predicted exactly (the PLD metric).  The verdict:
ceiling >> MTP-acceptance on copy-heavy traffic = the splice has addressable headroom;
no gap = the MTP drafter already covers copying and the candidate dies free.
Part 1 (this script): fire the prompts, save responses + the log offset map.
Part 2 (t48-ngram-analyze.py): tokenize and compute the ceiling."""
import json, subprocess, sys, time, urllib.request

PORT = "8081"
KEY = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
LOG = "/home/svg/LLM/strata-custom-2/custom-iq3_s.log"
OUT = "/home/svg/LLM/strata-custom-2/campaign/results/t48-ngram-raw.json"

COPY_HEAVY = [
    ("repeat-para", "Here is a paragraph I need preserved exactly in your reply. Copy it back VERBATIM first, word for word, then add one sentence of comment.\n\nThe quaternary cache governor retires superchunks only when the admission watermark concedes, a policy that trades residency for predictability and makes the demote path idempotent under contention. Every eviction carries a generation stamp so replay cannot resurrect a stale mapping.\n\nYour verbatim copy:"),
    ("code-reprint", "Reproduce this function exactly as written (same formatting), only renaming `xs` to `items`, then state what it does in one sentence:\n\ndef dedupe(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out"),
    ("list-restate", "Restate this exact list with the same numbering and wording, then add a fourth item in the same style:\n1. Prefill plateaus at the chunk boundary\n2. Decode rides the coupled drafter\n3. The verify window caps at eight rows"),
]
CONTROLS = [
    ("story", "Write a short original story (4-5 sentences) about a lighthouse keeper who discovers the light has been guiding something unexpected."),
    ("essay", "Argue in 3-4 sentences whether cache invalidation deserves its reputation as one of the two hard problems, taking a clear position."),
    ("analogy", "Explain speculative decoding using an original analogy a chef would appreciate. 3-4 sentences."),
]

def log_lines():
    return int(subprocess.run(["grep", "-c", "", LOG], capture_output=True, text=True).stdout.strip())

def ask(prompt, max_tokens=400):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.7}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d, time.time() - t0

def main():
    rows = []
    for kind, prompts in (("copy", COPY_HEAVY), ("control", CONTROLS)):
        for name, p in prompts:
            l0 = log_lines()
            d, dt = ask(p)
            time.sleep(1.0)                      # let the engine flush the stat line
            tail = subprocess.run(["tail", f"-{log_lines() - l0 + 40}", LOG],
                                  capture_output=True, text=True).stdout
            acc_lines = [l for l in tail.splitlines() if "drafts accepted" in l]
            rows.append({
                "kind": kind, "name": name, "prompt": p, "wall": dt,
                "content": d.get("choices", [{}])[0].get("message", {}).get("content") or "",
                "completion_tokens": d.get("usage", {}).get("completion_tokens"),
                "engine_accept": acc_lines[-1] if acc_lines else "(not found)",
            })
            print(f"[{kind}/{name}] {dt:.1f}s  {rows[-1]['engine_accept'][:80]}")
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"saved {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
