#!/usr/bin/env python
"""MMLU-Pro (70q validation) with the DRAFT FLOOR carried PER REQUEST, so both floors can be
measured in ONE boot and ONE window - the matched-window form the ultimatum requires.

Why this exists: `mmlu-pro-strata2.py` posts a body with no `sampling.strata_tune`, so it can only
measure whatever floor the engine booted with. Comparing that against a number recorded on another
boot is the remembered-comparator error. Here the floor travels with each request through the same
per-request key the campaign's sweep instrument uses (serve/server.py:1005 -> generate.cpp:10277).

Usage: python mmlu-pro-gate.py <floor> <label>      e.g.  python mmlu-pro-gate.py 0.7 base07
Writes campaign/results/mmlu-<label>.json and prints the score.
"""
import json, os, re, sys, urllib.request
from datasets import load_dataset

API = "http://127.0.0.1:8081/v1/chat/completions"
FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
LABEL = sys.argv[2] if len(sys.argv) > 2 else f"floor{FLOOR}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"mmlu-{LABEL}.json")

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")


def fmt_options(opts):
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))


def ask(q, mt=4000):
    body = json.dumps({"messages": [{"role": "user", "content": q}],
                       "max_tokens": mt, "temperature": 0.1,
                       # the per-request tuning key: this request's draft floor only
                       "strata_tune": {"spec_min_p": FLOOR},
                       "sampling": {"strata_tune": {"spec_min_p": FLOOR}}}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json", "Authorization": "Bearer 3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"})
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return (r["choices"][0]["message"].get("content") or "").strip()


ask("hi", 20)

correct = 0
by_cat = {}
rows = []
for ex in ds:
    q = (f"{ex['question']}\n\n{fmt_options(ex['options'])}\n\n"
         "Answer with just the letter of the correct option.")
    out = ask(q)
    m = re.search(r"\b([A-J])\b", out)
    pred = m.group(1) if m else "?"
    ok = pred == ex["answer"]
    correct += ok
    by_cat.setdefault(ex["category"], []).append(ok)
    rows.append({"cat": ex["category"], "ok": bool(ok), "pred": pred, "gold": ex["answer"]})
    print(f"[{'PASS' if ok else 'FAIL'}] {ex['category']:18s} pred={pred} gold={ex['answer']}", flush=True)

n = len(ds)
pct = 100 * correct / n
# the binomial 1-sigma on this suite size: a drop smaller than this cannot be resolved by one run
se = 100 * ((pct / 100) * (1 - pct / 100) / n) ** 0.5
print(f"\n=== MMLU-Pro ({n} Qs) [floor {FLOOR}, {LABEL}]: {correct}/{n} = {pct:.1f}%  (1-sigma = {se:.2f} pts) ===")
for c, v in sorted(by_cat.items()):
    print(f"  {c:18s} {sum(v)}/{len(v)}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"label": LABEL, "floor": FLOOR, "suite": "MMLU-Pro validation (70q)",
               "score_pct": pct, "sigma_pts": se,
               "by_cat": {c: f"{sum(v)}/{len(v)}" for c, v in by_cat.items()}, "rows": rows},
              f, indent=1, ensure_ascii=False)
print("WROTE", OUT)
