#!/usr/bin/env python3
"""MMLU-Pro 70q gate, INTERLEAVED per-request arm: ON vs OFF in one boot, one window.

Why this exists: `mmlu-pro-gate.py` measures one arm per boot and stores only {cat, ok, pred, gold},
so a cross-boot delta cannot be split into arm effect vs boot noise, and a force-closed reasoning
block (the 2026-11-24 90.0 incident: 3 of 7 fails were wrapped at the 3000-token thinking cap)
is invisible in the JSON. Here both arms run under the same boot, alternating question by question
through the per-request key `experimental_speed_projection` (serve/server.py projection_key:
absent = on, false = off), at a budget the caller sets, and every row carries the evidence.

Usage: python mmlu-pro-interleave.py <label> [--max-tokens N] [--arm-key experimental_speed_projection]
Writes campaign/results/mmlu-interleave-<label>.json and prints both arms with the paired test.
"""
import argparse, json, os, re, sys, urllib.request
from datasets import load_dataset

API = f"http://127.0.0.1:{os.environ.get('STRATA_GATE_PORT', '8081')}/v1/chat/completions"
KEY = os.environ.get("STRATA_API_KEY", "3f0940df8efb9c36")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ask(q, mt, armed, key):
    body = {"messages": [{"role": "user", "content": q}], "max_tokens": mt, "temperature": 0.1}
    if not armed:
        body[key] = False                      # the per-request switch: false = the stock model
    req = urllib.request.Request(API, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    ch = r["choices"][0]
    return (ch["message"].get("content") or "").strip(), ch.get("finish_reason"), r.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--arm-key", default="experimental_speed_projection")
    a = ap.parse_args()
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    ask("hi", 20, True, a.arm_key)                         # the warm-up request, never scored
    rows = []
    for i, ex in enumerate(ds):
        letters = "ABCDEFGHIJ"
        q = (f"{ex['question']}\n\n" + "\n".join(f"{letters[k]}. {o}" for k, o in enumerate(ex["options"]))
             + "\n\nAnswer with just the letter of the correct option.")
        rec = {"idx": i, "cat": ex["category"], "gold": ex["answer"]}
        # alternate the order per question so drift cannot favour one arm
        for arm in (("on", "off") if i % 2 == 0 else ("off", "on")):
            out, fin, usage = ask(q, a.max_tokens, arm == "on", a.arm_key)
            m = re.search(r"\b([A-J])\b", out)
            pred = m.group(1) if m else "?"
            rec[arm] = {"pred": pred, "ok": pred == ex["answer"], "finish": fin,
                        "gen_tokens": usage.get("completion_tokens"), "content": out[-600:]}
        rec["flip"] = rec["on"]["ok"] != rec["off"]["ok"]
        rows.append(rec)
        print(f"[{i:2d}] {ex['category'][:14]:14s} on={'R' if rec['on']['ok'] else 'W'}"
              f"{'(' + str(rec['on']['finish'] or '')[:6] + ')' if rec['on']['finish'] != 'stop' else ''}"
              f" off={'R' if rec['off']['ok'] else 'W'}"
              f"{'(' + str(rec['off']['finish'] or '')[:6] + ')' if rec['off']['finish'] != 'stop' else ''}"
              f" tok={rec['on']['gen_tokens']}/{rec['off']['gen_tokens']}", flush=True)

    on = sum(r["on"]["ok"] for r in rows); off = sum(r["off"]["ok"] for r in rows)
    on_only = sum(r["on"]["ok"] and not r["off"]["ok"] for r in rows)
    off_only = sum(r["off"]["ok"] and not r["on"]["ok"] for r in rows)
    wraps = {arm: sum((r[arm]["finish"] == "length") for r in rows) for arm in ("on", "off")}
    fails_wrapped = {arm: sum((not r[arm]["ok"]) and r[arm]["finish"] == "length" for r in rows) for arm in ("on", "off")}
    gt = {arm: sorted(r[arm]["gen_tokens"] or 0 for r in rows) for arm in ("on", "off")}
    med = {arm: gt[arm][len(gt[arm]) // 2] for arm in gt}
    # exact McNemar on the discordant pairs
    n = on_only + off_only
    from math import comb
    p = min(1.0, 2 * sum(comb(n, k) for k in range(max(on_only, off_only), n + 1)) / 2 ** n) if n else 1.0
    s = {"label": a.label, "max_tokens": a.max_tokens, "n": len(rows),
         "on": on, "off": off, "on_only_right": on_only, "off_only_right": off_only, "mcnemar_p": round(p, 4),
         "finish_length": wraps, "fails_that_were_wrapped": fails_wrapped,
         "median_gen_tokens": med, "flips": sum(r["flip"] for r in rows)}
    print(f"\n=== MMLU-Pro interleave [{a.label}] max_tokens={a.max_tokens}: ON {on}/70  OFF {off}/70  "
          f"| on-only {on_only} off-only {off_only} McNemar p={p:.3f}\n    wrapped(finish=length) on {wraps['on']} off {wraps['off']} ; "
          f"fails-wrapped on {fails_wrapped['on']} off {fails_wrapped['off']} ; median gen tokens on {med['on']} off {med['off']}")
    with open(os.path.join(REPO, "campaign/results", f"mmlu-interleave-{a.label}.json"), "w") as f:
        json.dump({"summary": s, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
