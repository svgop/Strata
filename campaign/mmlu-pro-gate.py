#!/usr/bin/env python
"""MMLU-Pro gate with the DRAFT FLOOR carried PER REQUEST, so both floors can be measured in ONE
boot and ONE window - the matched-window form the ultimatum requires.

Why this exists: `mmlu-pro-strata2.py` posts a body with no `sampling.strata_tune`, so it can only
measure whatever floor the engine booted with. Comparing that against a number recorded on another
boot is the remembered-comparator error. Here the floor travels with each request through the same
per-request key the campaign's sweep instrument uses (serve/server.py:1005 -> generate.cpp:10277).

Usage: python mmlu-pro-gate.py <floor> <label> [--suite N] [--repeats K] [--list]
       e.g.  python mmlu-pro-gate.py 0.7 base07
             python mmlu-pro-gate.py 0.933 f453-stand --suite 1500
             python mmlu-pro-gate.py 0.933 f470-pres --suite 1500 --repeats 3
Writes campaign/results/mmlu-<label>.json and prints the score.

THE REPEATS FORM (F-468/F-469, 2026-11-25).  Two gates on ONE boot are CONFOUNDED: the second
gate re-asks identical prompts and the conversation cache serves run 1's parked KV back
(F-450-class nondeterminism; the F-469 pair moved 88.6 vs 97.1 within one boot).  The clean
between-boot comparison is campaign/gate-pair.sh (one cold boot per arm, gate run once).
--repeats K is the other half, the vendors' within-boot form (DeepSeek-R1: "conduct multiple
tests and average"; Kimi K2 avg@64): the SAME suite is asked K times against the ONE boot, so
the spread across repeats is the in-boot sampling variance at temperature 0.1 - report the
per-repeat scores, mean +/- sd, and the verdict vs the floor.  The repeats share the boot and
its cache state, so sd is the served-state spread, not an iid bound, and never a substitute
for the cold-boot pair.  --repeats 1 (the default) is byte-identical to the record form: no
repeat lines, no extra JSON keys.  Per-question transport failures retry ONCE and are then
excluded (the FAIL-TRANSPORT path - never scored wrong); the old form retried three times.

THE SUITE (F-453).  The default, --suite 70, is the validation split in dataset order - the campaign's
record suite; every MMLU number in the journal before F-453 is this suite, and its 1-sigma is 3.1
points at 93%.  --suite N (N > 70) draws a fixed stratified sample of N questions from the test split
(12,032 questions, 14 categories): the allocation per category is proportional (largest remainder),
the draw inside each category comes from one seeded generator consumed in sorted-category order, and
the questions are asked in category round-robin order, so every prefix of a run is category-balanced
- the set is the same on every run and every box, and its ids are written to the result.  (The first
run, 2026-11-25 label f453-stand, asked the same set in dataset order; its JSON carries no "order" key.)
At N = 1500 the 1-sigma is 0.60 points: a delta of one validation
question (1.43 points) reads as 2.4 sigma instead of 0.34.  The 70-question path is unchanged.
"""
import argparse, json, math, os, random, re, statistics, sys, time, urllib.request
from collections import defaultdict
from datasets import load_dataset

API = f"http://127.0.0.1:{os.environ.get('STRATA_GATE_PORT', '8081')}/v1/chat/completions"
DATASET = "TIGER-Lab/MMLU-Pro"
SUITE_SEED = 20261125   # the gate's fixed draw: change it and every --suite N result before it is another suite
ATTEMPTS = 2            # a transport failure is retried ONCE, then the question is excluded - never scored wrong


def suite(n):
    """The gate's question set: the validation split (n == 70) or the fixed stratified test sample (n > 70)."""
    if n == 70:
        return list(load_dataset(DATASET, split="validation")), "MMLU-Pro validation (70q)"
    ds = load_dataset(DATASET, split="test")
    if not 70 < n <= len(ds):
        sys.exit(f"--suite must be 70 or 71..{len(ds)}; got {n}")
    by_cat = defaultdict(list)
    for i, cat in enumerate(ds["category"]):
        by_cat[cat].append(i)
    exact = {c: n * len(ix) / len(ds) for c, ix in by_cat.items()}      # proportional quotas
    alloc = {c: math.floor(q) for c, q in exact.items()}
    seats = n - sum(alloc.values())                                     # the remainders' seats, largest first
    for c in sorted(exact, key=lambda c: (exact[c] - alloc[c], c), reverse=True)[:seats]:
        alloc[c] += 1
    rng = random.Random(SUITE_SEED)
    picked = []
    for c in sorted(by_cat):
        ix = sorted(by_cat[c])
        rng.shuffle(ix)
        picked += ix[: alloc[c]]
    # Asked in category round-robin order, so every PREFIX of a run is category-balanced: the test split is stored
    # by category, and a run stopped early in dataset order would be one category's score wearing a stratified label.
    lanes = defaultdict(list)
    for i in sorted(picked):
        lanes[ds[i]["category"]].append(i)
    order = []
    while any(lanes[c] for c in lanes):
        for c in sorted(lanes):
            if lanes[c]:
                order.append(lanes[c].pop(0))
    return [ds[i] for i in order], f"MMLU-Pro test ({n}q: stratified, seed {SUITE_SEED})"


def fmt_options(opts):
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))


def ask(q, mt, floor, presence):
    body = {"messages": [{"role": "user", "content": q}],
            "max_tokens": mt, "temperature": 0.1,
            # the per-request tuning key: this request's draft floor only
            "strata_tune": {"spec_min_p": floor},
            "sampling": {"strata_tune": {"spec_min_p": floor}}}
    if presence:
        # --presence F: a presence penalty on every request (the engine applies it over the last 64 tokens when a
        # penalty key arrives with no window; serve/server.py:1626).  The repetition knob the vendor names, so the
        # accuracy gate can be run on the penalty the standing is about to adopt.
        body["presence_penalty"] = presence
    body = json.dumps(body).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json",
                                             "Authorization": "Bearer " + os.environ.get(
                                                 "STRATA_API_KEY",
                                                 "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8")})
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return (r["choices"][0]["message"].get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("floor", type=float)
    ap.add_argument("label", nargs="?", default=None)
    ap.add_argument("--suite", type=int, default=70,
                    help="70 = the validation split (the record suite); N = the fixed stratified test sample")
    ap.add_argument("--list", action="store_true",
                    help="print the suite's composition and exit (no server contact)")
    ap.add_argument("--presence", type=float, default=0.0,
                    help="send presence_penalty F on every request (0 = off, the record form)")
    ap.add_argument("--first", type=int, default=0,
                    help="score only the first N questions of the suite (the order is category round-robin, so a prefix "
                         "is a balanced sample): the paired form of a comparison against a run that covered more")
    ap.add_argument("--repeats", type=int, default=1,
                    help="ask the SAME suite K times against this one boot and report per-repeat scores, mean +/- sd "
                         "and the verdict vs the floor (in-boot sampling variance at temperature 0.1, the DeepSeek-R1 "
                         "'multiple tests and average' form); 1 = the record form, byte-identical")
    a = ap.parse_args()
    if a.repeats < 1:
        sys.exit(f"--repeats must be >= 1; got {a.repeats}")
    examples, suite_name = suite(a.suite)
    if a.first > 0:
        if a.first > len(examples):
            sys.exit(f"--first {a.first} exceeds the suite's {len(examples)} questions")
        examples = examples[:a.first]
    if a.list:
        by_cat = defaultdict(int)
        for ex in examples:
            by_cat[ex["category"]] += 1
        print(json.dumps({"suite": suite_name, "n": len(examples), "by_cat": dict(sorted(by_cat.items())),
                          "first_ids": [ex["question_id"] for ex in examples[:5]],
                          "last_ids": [ex["question_id"] for ex in examples[-3:]]}, indent=1))
        return
    if not a.label:
        sys.exit("usage: mmlu-pro-gate.py <floor> <label> [--suite N] [--list]")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"mmlu-{a.label}.json")

    ask("hi", 20, a.floor, a.presence)

    K = a.repeats
    correct = 0                                   # pooled across repeats (feeds by_cat and the K == 1 headline)
    by_cat = {}
    rows, failed = [], []
    rep_scores = []                               # one percent per repeat, in repeat order
    t0 = time.time()
    for rep in range(1, K + 1):
        if K > 1:
            print(f"=== repeat {rep}/{K} ===", flush=True)
        rep_correct, rep_n = 0, 0
        for k, ex in enumerate(examples, 1):
            q = (f"{ex['question']}\n\n{fmt_options(ex['options'])}\n\n"
                 "Answer with just the letter of the correct option.")
            out_text, error = None, None
            for attempt in range(1, ATTEMPTS + 1):
                try:
                    out_text = ask(q, 4000, a.floor, a.presence)
                    break
                except Exception as e:                    # transport: retried once, then excluded - never scored wrong
                    error = repr(e)
                    print(f"[retry {attempt}/{ATTEMPTS}] request failed: {error}", flush=True)
                    time.sleep(5)
            if out_text is None:
                failed.append(ex["question_id"])
                row = {"id": ex["question_id"], "cat": ex["category"], "error": error}
                if K > 1:
                    row["repeat"] = rep
                rows.append(row)
                print(f"[FAIL-TRANSPORT] {ex['category']:18s} id={ex['question_id']} excluded", flush=True)
                continue
            m = re.search(r"\b([A-J])\b", out_text)
            pred = m.group(1) if m else "?"
            ok = pred == ex["answer"]
            correct += ok
            rep_correct += ok
            rep_n += 1
            by_cat.setdefault(ex["category"], []).append(ok)
            row = {"id": ex["question_id"], "cat": ex["category"], "ok": bool(ok), "pred": pred, "gold": ex["answer"]}
            if K > 1:
                row["repeat"] = rep
            rows.append(row)
            print(f"[{'PASS' if ok else 'FAIL'}] {ex['category']:18s} pred={pred} gold={ex['answer']}", flush=True)
            if k % 50 == 0 or k == len(examples):
                el = time.time() - t0
                print(f"[{k}/{len(examples)} running {100 * correct / len(rows):.1f}%] {el:.0f} s, {el / k:.2f} s/question", flush=True)
        rep_scores.append(100 * rep_correct / rep_n if rep_n else 0.0)
    seconds = time.time() - t0

    sd_pts, verdict = None, None
    if K == 1:
        n = len(examples) - len(failed)
        pct = 100 * correct / n
    else:
        # the mean over per-repeat scores is the vendor form (the repeat is the sampling unit);
        # sample sd (n-1) - with k >= 2 repeats it is the conservative spread estimate.
        n = len(examples)
        pct = sum(rep_scores) / len(rep_scores)
        sd_pts = statistics.stdev(rep_scores)
        verdict = "PASS" if pct >= a.floor * 100 else "FAIL"
    # the binomial 1-sigma on this suite size: a drop smaller than this cannot be resolved by one run.
    # Stratification only lowers it, so the simple-random form is the conservative reading.
    se = 100 * ((pct / 100) * (1 - pct / 100) / n) ** 0.5
    if K == 1:
        print(f"\n=== MMLU-Pro ({n} Qs) [floor {a.floor}, {a.label}]: {correct}/{n} = {pct:.1f}%  (1-sigma = {se:.2f} pts) ==="
              f"  [{suite_name}; 95% CI +/- {1.96 * se:.1f}; {seconds:.0f} s, {seconds / len(examples):.2f} s/question]")
    else:
        print(f"\n=== MMLU-Pro ({n} Qs x {K} repeats) [floor {a.floor}, {a.label}]: mean {pct:.2f} +/- {sd_pts:.2f} sd  (1-sigma = {se:.2f} pts) ==="
              f"  [{suite_name}; 95% CI +/- {1.96 * se:.1f}; {seconds:.0f} s total, {seconds / (len(examples) * K):.2f} s/question]")
        for i, s in enumerate(rep_scores, 1):
            print(f"  repeat {i}/{K}: {s:.1f}%")
        print(f"VERDICT vs floor {a.floor} ({a.floor * 100:.1f}%): {verdict} - mean {pct:.2f}, sd {sd_pts:.2f} over {K} in-boot repeats at temperature 0.1")
    for c, v in sorted(by_cat.items()):
        print(f"  {c:18s} {sum(v)}/{len(v)}")
    if failed:
        print(f"TRANSPORT FAILURES EXCLUDED: {len(failed)} -> {failed}")

    rec = {"label": a.label, "floor": a.floor, "presence": a.presence, "suite": suite_name, "n": n, "seed": SUITE_SEED,
           "order": "category round-robin" if a.suite != 70 else "dataset",
           "first": a.first,
           "score_pct": pct, "sigma_pts": se, "ci95_pts": 1.96 * se,
           "seconds": seconds, "excluded": failed,
           "by_cat": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
           "ids": [ex["question_id"] for ex in examples], "rows": rows}
    if K > 1:
        # the repeats form: per-repeat scores in order, their mean (score_pct) and sample sd, and the
        # verdict vs the floor the driver/consumer reads.  "excluded" pools the FAIL-TRANSPORT ids of
        # all repeats (each row carries its "repeat"); n above is the per-repeat suite size.
        rec["repeats"] = K
        rec["repeat_scores"] = [round(s, 3) for s in rep_scores]
        rec["sd_pts"] = sd_pts
        rec["verdict"] = verdict
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
