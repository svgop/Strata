#!/usr/bin/env python3
"""campaign/t42-prefill-ladder.py — PREFILL PLATEAU proof (F-284 re-bank rider): fires
fresh single-conversation prompts at growing lengths (max_tokens=1, pure prefill) and reads
the ENGINE's own per-request prefill line from custom-iq3_s.log ("prompt N tokens = 0
reused + N read in X ms (Y tok/s)") — the server-side rate, no client/HTTP/render noise.
The banked plateau was 6.0-7.0k tok/s to 120k context."""
import json, re, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "ladder"
KEY = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
LOG = "/home/svg/LLM/strata-custom-2/custom-iq3_s.log"

SIZES = [2000, 8000, 32000, 96000]          # target prompt tokens (approx; filler ~4 ch/tok)
pat = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)")

def filler(target_tok, salt):
    # deterministic unique filler: numbered sections with varied words (no cache dedup)
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
             "omicron pi rho sigma tau upsilon phi chi psi omega quaternion vector tensor "
             "matrix kernel latency bandwidth throughput speculative verify accept draft ").split()
    out, i = [], 0
    want = target_tok * 4                   # ~4 chars per token
    while sum(len(w) + 1 for w in out) < want:
        out.append(f"sec{i + salt}-{words[(i * 7 + salt) % len(words)]}")
        i += 1
    return " ".join(out)

def ask(prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return time.time() - t0, d.get("usage", {}).get("prompt_tokens")

def log_prefill(offset):
    fh = open(LOG, errors="replace")
    fh.seek(offset)
    best = None
    for line in fh:
        m = pat.search(line)
        if m:
            best = m
    fh.close()
    return best

def main():
    print(f"=== PREFILL LADDER {LABEL} | {time.strftime('%H:%M:%S')}")
    for i, size in enumerate(SIZES):
        p = filler(size, salt=i * 1000)
        wall, ptok = ask(p)
        fh = open(LOG, "rb"); off = fh.seek(0, 2); fh.close()
        time.sleep(0.6)                     # let the engine flush its stat line
        m = log_prefill(off) if False else None
        # the line may flush slightly later; poll a couple of times
        for _ in range(10):
            fh = open(LOG, errors="replace")
            fh.seek(off)
            tail = fh.read()
            fh.close()
            m = None
            for line in tail.splitlines():
                mm = pat.search(line)
                if mm:
                    m = mm
            if m:
                break
            time.sleep(0.5)
        if m:
            p_tok, ru, rr, rms, rts = m.groups()
            print(f"  ~{size//1000}k: engine {rr} tok in {rms} ms = {rts} tok/s  "
                  f"(client wall {wall:.1f}s, prompt_tokens={ptok})")
        else:
            print(f"  ~{size//1000}k: no engine line (client wall {wall:.1f}s, prompt_tokens={ptok})")

if __name__ == "__main__":
    main()
