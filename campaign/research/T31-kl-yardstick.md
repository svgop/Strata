# T31 — THE KL YARDSTICK: `tools/kl_yardstick.py`, the numerics-class gate instrument (roadmap REVISED ORDER item 2)

- **Date:** 2026-10-22. **Mode:** CPU-only, zero boots — the 8081 battery is LIVE; nothing in this task touches
  `engine/`, the launchers, any `custom-iq3_s*.json`, builds, or git. Deliverables: the tool (pure python3 stdlib +
  numpy, run under `.venv/bin/python`) and this note. Reading `src/program/generate.cpp` / `serve/server.py` for the
  capture sketch was allowed; editing them was not, and was not done.
- **Read first, not repeated:** F-230 (SVG-CUSTOM.md:7884-7912 — the precedent this instrument prices),
  T26-5-numerics-quality.md section b (the engine-side KL protocol this complements) and section c (the fp16-accum
  error model this tool arbitrates), `tools/quant_quality.py` (the existing KL conventions: KL(REF||RUN), natural
  log), AGENTS.md laws 2/4/5 (never regress; cheapest falsifier first; read what exists).
- **Roadmap:** SVG-CUSTOM-TODO.md:1761 (item 2: "KL yardstick — gates fp16-accum and every future numerics-class
  arm") and :1737 / :1768 (item 8: scoped fp16-accum K=640/2560, R3's +10-17% vs R5's KL blowout — DISPUTED,
  this instrument decides).

---

## (a) What it is, and how to run it

Given reference logits P and candidate logits Q — each a `.npy` `[N, V]` float array or a `.jsonl` of
`{"logits": [...]}` records — it prints the divergence table and a verdict, or `--json` `{metrics, verdict}`:

```
.venv/bin/python tools/kl_yardstick.py --ref REF.npy --cand CAND.npy
.venv/bin/python tools/kl_yardstick.py --ref REF.npy --cand CAND.jsonl --json
.venv/bin/python tools/kl_yardstick.py --selftest
```

Rows are trimmed to the shorter N, rows with any non-finite logit are dropped and counted (both sides), everything
is float64 through a stable log-softmax. Metrics: KL(P||Q), KL(Q||P), symmetric JS (nats), each mean/median/p95/max;
top-32-restricted KL(P||Q) (union of each row's top-32 by P and by Q, renormalized); argmax agreement rate; mean/max
|logit diff|. Exit 0 for any completed measurement (the verdict is data, not a crash), 2 for unusable input
(vocab mismatch, malformed records, zero usable rows), 1 for a selftest failure. A representative run (synthetic
F-230-scale pair, from this session's verification):

```
KL(ref||cand)                2.147e-04   2.135e-04   3.039e-04   3.125e-04      (mean median p95 max)
KL(ref||cand) top-32         2.169e-04   ...
argmax agreement         1.0000   (200/200 rows)
verdict  full-KL mean    2.147e-04   BORDERLINE    5e-05 <= x < 1e-03  [PROVISIONAL, F-230-anchored]
F-230 anchor: a 1.8e-4 KL sat in this band and cost -2.7% relative acceptance = -2.8% M1 -
the coupled acceptance battery (M3b + acceptance A/B) is mandatory before arming this arm.
```

## (b) The math, written out

All in float64, natural log throughout (the repo's convention — STRATA_LOGPOS and quant_quality.py are nats).

- **Log-softmax:** `log p_v = z_v - logsumexp(z)`, `logsumexp(z) = m + log Σ exp(z_v - m)`, `m = max_v z_v`. The
  max-subtraction is why a +1000 logit offset (selftest fixture a) loses nothing.
- **Per-row KL:** `KL(P||Q) = Σ_v p_v (log p_v - log q_v)`, computed as `Σ_v exp(lp_v) · (lp_v - lq_v)` — never a
  `p/q` division, so no underflow-to-zero hazards; a `p_v` that underflows contributes exactly 0.
- **Reverse KL** `KL(Q||P)` same with roles swapped.
- **JS (nats, bounded by ln 2):** `M = ½P + ½Q` via `log m_v = logaddexp(lp_v, lq_v) - ln 2` (stable), then
  `JS = ½ KL(P||M) + ½ KL(Q||M)`. Symmetric by construction.
- **Top-32 restricted KL:** per row, `U = top32(P) ∪ top32(Q)` (≤ 64 columns — the acceptance-relevant head where
  exact-match argmax decisions flip); renormalization done IN LOG SPACE — the restricted log-softmax over U *is*
  the renormalization, so extreme logits never produce a divide-by-near-zero. At V ≤ 32, U is the whole row and
  this reduces to the full KL (asserted in the selftest).
- **Argmax agreement:** fraction of rows with `argmax P == argmax Q` (first-index tie convention) — the direct
  exact-match coupling signal.
- **Sanity:** mean/max |logit diff| over all elements.

## (c) The selftest fixture math (all asserted, exit 1 on any failure)

- **(a) 3-symbol closed form**, hand formula written in the test:
  `P = (½, ¼, ¼)`, `Q = (¼, ½, ¼)`, fed as logits `ln p + 1000` (offset proves the logsumexp path).
  `KL(P||Q) = ½ ln(½/¼) + ¼ ln(¼/½) + ¼ ln(¼/¼) = ½ ln2 − ¼ ln2 = ln2/4 ≈ 0.1732867951`.
  `KL(Q||P)` is the same value here (P, Q are column swaps of each other).
  `M = (⅜, ⅜, ¼)`; `KL(P||M) = ½ ln(4/3) + ¼ ln(2/3) ≈ 0.0424747592`, and by the pair's symmetry
  `JS = KL(P||M)` exactly. Asserted to ≤ 1e-10 relative, including the top-32 KL (= full KL at V = 3).
- **(b) identity → exactly 0.0** (not approximately): identical logits produce bitwise-identical log-softmax
  rows, every term is `p·(+0.0)`, and the sum of `+0.0` is `+0.0` — asserted `== 0.0` for full KL, reverse KL and
  top-32 KL. JS identity is asserted only ≤ 1e-12: `logaddexp(a,a) − ln2` can round one ulp off `a`, so JS of an
  identity is ulp-level, not exact, by construction.
- **(c) two discretized normals** (grid −8..8 step 0.04, 401 points; μ = 0, σ = 1 vs μ = 0.5, σ = 1.2, unnormalized
  `exp(−(x−μ)²/2σ²)` logits): expected KL computed in the test by a brute-force `Σ p ln(p/q)` over plain
  `math.exp`/`math.log` floats — a path independent of the tool's log-softmax. Asserted ≤ 1e-10. Also asserts
  argmax agreement == 0.0 here (the two normals peak at different x — the agreement metric demonstrably flips).
- **(d) permutation invariance of JS:** swapping P/Q roles gives `½A + ½B` vs `½B + ½A` (equal by IEEE
  commutativity), and relabeling the vocabulary (same distributions, permuted columns) leaves KL, JS and the
  top-32 KL unchanged up to summation order — asserted ≤ 1e-12.
- **Reduction:** at V = 17 (< 32), top-32 KL == full KL (≤ 1e-12, summation order only).
- **Plumbing:** trim to min N; nonfinite rows rejected with per-side counts (a NaN on one side drops the pair row);
  vocab mismatch and all-nonfinite inputs refused with errors; `.jsonl` round-trips `.npy` values exactly, NaN
  literals parse and are rejected downstream.

Session result: `selftest: all fixtures green (...)`, exit 0, under `.venv/bin/python` (3.15.0, numpy 2.5.3).

## (d) Threshold rationale — PROVISIONAL, F-230-anchored

F-230 (SVG-CUSTOM.md:7884): a Q8-activation prefill change moved the prompt KV by **~1.8e-4 KL** (upstream's own
number) and paid **−2.7% relative acceptance = −2.8% M1**, with slots/hit-rate exonerated — the mechanism is purely
exact-match decisions flipping. That single calibrated point anchors all four bands:

| band | mean KL(P||Q) | anchor argument |
|---|---|---|
| exact | < 1e-7 | 3 orders under F-230. GDN_CHUNKED's adopted FP32 sum-order shift (~3e-7, M3b in-band) sits just above this line, inside benign — consistent with its clean adoption. |
| benign | < 5e-5 | ~3.6x under F-230; F-230's law says the tax scales with the bits distance, so acceptance movement is expected well inside boot noise (the ±1% class the interleave protocol resolves). |
| borderline | 5e-5 .. 1e-3 | **F-230's own 1.8e-4 sits here** and cost −2.7% acceptance / −2.8% M1. The coupled acceptance battery (M3b + acceptance A/B) is MANDATORY before arming; the tool prints this reminder whenever either verdict lands here. |
| acceptance-risk | >= 1e-3 | ≥ 5.5x F-230 → projected ≥ −13% decode class; reject on numerics, no battery warranted. T26-5 §c's predicted blanket fp16-accum KLD 0.15–0.40 lands 150–400x into this band. |

Both the **full-KL mean** and the **top-32-KL mean** are classified; disagreement is FLAGGED and the gate class is
the worse of the two (a head-concentrated move can read benign on full KL and borderline restricted — the flag
catches exactly the acceptance-relevant case). Classification direction is KL(ref||cand), matching quant_quality.py's
KL(REF||RUN) convention (the reference is the true distribution; the question is how the candidate covers it).

**Two scales, two instruments — do not mix:** T26-5 §b's pack-scale gates (top-1 ≥ 97%, mean KL ≤ 0.05, p99 ≤ 1.0,
calibrated against the one-format-step 0.144 field row) referee whole-pack requants through the engine's
teacher-forced protocol. This tool works one scale down (1e-7..1e-3) on raw logits captures, where numerics-class
arms (GEMM accumulators, kernel swaps, fp16-accum scopes) land. A verdict from one scale never cites the other's
thresholds.

**Calibration owed before the thresholds lose the PROVISIONAL tag:** a ctl-vs-ctl self-noise pair (same bits, two
standalone boots). Expected to read exact/benign; if the floor reads higher, every band moves up by the floor —
"the gate's resolution is its own noise" (T26-5 §b protocol step 2).

## (e) Limitations

- **Logits-scale sensitivity:** the metrics are invariant to any per-row constant shift (softmax) but NOT to
  scaling — a temperature or logit-scale difference between the two captures reads as pure divergence. Captures
  must be raw final-layer logits, identical sampling configuration, no logit bias, same vocab packing (the EN
  162,100-token draft subset vs the full vocab is a hard incomparable — a real trap on this box, T26-5 §a.4-7).
- **Sampling-vs-greedy mismatch:** KL bounds the distortion of *sampled* continuations; the engine's coupling is
  *greedy exact-match*. Argmax agreement is the direct signal; a small mean KL can still flip near-ties (the
  top-32 KL localizes that), and conversely a rare fat-tail row can be KL-large and argmax-neutral. Read mean
  TOGETHER with p95/max: a fat tail over a clean mean is the routing-drift signature (T26-5 §b note 5).
- **float32 capture floor:** engine dumps are float32, so two identical-bits boots are not bit-identical inputs;
  in this session's synthetic verification, a float32-stored pair carrying ~2e-4-scale logit noise read ~1.7e-8 KL —
  the 'exact' band is reachable with real float32 captures, but only the real self-noise pair quantifies the floor
  on this engine.
- **Fixture coverage:** 8 prompts is a gate sample, not the acceptance distribution. The F-230 mechanism is
  prefill-bits → KV → verify-window logits, so captures must go through the batched prompt path (first-window
  logits), not only decode windows.
- **Ties at the top-32 boundary** split deterministically (argpartition); with real float logits this never binds,
  but synthetic fixtures with exact ties can.
- **p95 is linear-interpolated** (numpy default); medians over few rows are coarse. Memory: inputs live whole in
  RAM ([N, V] float64); derived math is chunked at 256 rows so temporaries are bounded, but a 100k-row full-vocab
  capture should be split.

## (f) DESIGN-ONLY sketch: how a real capture wires in later (STRATA_KL_CAPTURE; nothing implemented here)

The tree already owns every piece the capture needs (AGENTS.md law 5 — read what exists):

| existing hook | anchor | form |
|---|---|---|
| `STRATA_DUMP_FIRST_LOGITS` (serve path) | src/program/generate.cpp:15370-15386 | per request's FIRST verify window, `ver.copy_logits(row)` → one raw float32 full-vocab row per request to `<path>.<n>` — the first window's logits carry exactly what the prompt path did to the KV (the comment at :16829 says this in so many words) |
| `STRATA_DUMP_FIRST_LOGITS` (CLI generate) | src/program/generate.cpp:16833-16840 | same, single file, CLI requests |
| `--dump-logits [N]` / `--logits-stride` | src/program/generate.cpp:6959-6986, :949-950 | one raw float row per position, int32 `[vocab, rows]` header written with the first row (multi-position teacher-forced variant, zero new code) |
| `STRATA_LOGPOS(+TOPK)` | src/core/verify.cpp:3361+, include/strata/core/verify.hpp:330 | per-position target logprob + top-K `id:logp` (the T26-5 §b protocol's engine side) |
| fixture driver + consumer | tools/quant_quality.py `cmd_prompt` (:174-238) / `cmd_pcompare` (:239) | spawns one engine from a config JSON, sends batched prompts, reads `first.*` float32 files, computes KL |

The sketch, when a capture is actually needed:

1. **Engine side (~10 lines at the existing site):** `STRATA_KL_CAPTURE=<dir>` reuses the exact
   `STRATA_DUMP_FIRST_LOGITS` block in the serve request loop (generate.cpp:15373) — the first-window row is
   written per request exactly as today; nothing else in the window loop changes. The server itself
   (serve/server.py) never sees logits (the API returns tokens), so the dump must live engine-side; the server or a
   campaign driver only sequences the fixture.
2. **Fixture:** a committed 8-prompt file drawn from the verified `data/quant_quality/v1` set.ids.json segments
   (code/docs/english/multilingual/agent; short + long mix). A ~30-line campaign driver in the `cmd_prompt` shape
   (spawn engine from a config copy that differs ONLY in the arm's env, send the 8 prompts `GEN 1`, DONE-handshake)
   produces `<dir>/first.<0..7>`; a converter stamps them to `capture.npy [8, V]` float32 + `index.json`
   (prompt ids, config sha, engine build id) so any pair compared later is provably the same fixture.
3. **Scoring:** `tools/kl_yardstick.py --ref ctl/capture.npy --cand arm/capture.npy` — CPU, no engine, no port.
4. **Boot discipline (hard):** captures run on a STANDALONE boot, NEVER beside the live 8081 battery — the battery
   owns the GPU and its results; quant_quality.py's `--min-free-gb` guard and the campaign's serial-boots law (T16)
   both apply. Schedule in quiet windows; run the ctl-vs-ctl self-noise pair FIRST.
5. **Decode-side variant:** teacher-forced multi-position captures need no new code at all — CLI
   `strata generate --tokens-file <fixture> --max-new 1 --dump-logits out.bin` (generate.cpp:6959) then convert
   `out.bin` (int32 header + float rows) to `.npy`.

Out of scope by task constraint: any edit to `engine/`, `launch-custom-*.sh`, or `custom-iq3_s*.json` — this
section is the design record only.

## (g) Verification record (this session, CPU-only, nothing booted)

- `.venv/bin/python -m py_compile tools/kl_yardstick.py` — OK.
- `.venv/bin/python tools/kl_yardstick.py --selftest` — green, exit 0 (all fixture groups of section c).
- CLI exercised across all four verdict bands (synthetic [200, 1000] logits, float32 storage): 1.7e-8 → EXACT,
  3.8e-6 → BENIGN, 2.1e-4 → BORDERLINE (reminder printed), 2.9e-3 → ACCEPTANCE-RISK; `.jsonl` input with N=250 vs
  200 (trim) and 3 NaN rows (rejected, counted); disagreement case (head-only perturbation: full 4.8e-5 benign,
  top-32 2.8e-4 borderline) → FLAG printed, gate class = borderline; `--json` schema `{metrics, verdict}` validated;
  self-comparison reads 0.000e+00 across the table; missing file / empty jsonl / malformed record / ragged rows /
  no-args all exit 2 with a one-line error.
- Two defects found and fixed during that verification (kept as record): `|logit diff| mean` divided the
  all-element sum by row count only (read V× too large — mean exceeded max); the verdict class column was one
  character too narrow, gluing `ACCEPTANCE-RISK` into its band string.

## (h) Status and what is owed

- **DELIVERED:** `tools/kl_yardstick.py` (self-contained, stdlib + numpy, fixture-verified) and this note.
- **OWED at banking:** the INDEX.md row (per repo convention — T26-5 left the same owe); the ctl-vs-ctl self-noise
  pair (2 standalone boots, quiet window) to calibrate the floor; the first real arm through the gate = scoped
  fp16-accum K=640 (roadmap item 8's referee — R3 vs R5 resolved by a capture pair, not by argument).
- Thresholds remain PROVISIONAL until both calibrations land; the tool prints that tag on every verdict.

## Sources (read this session)

- `SVG-CUSTOM-TODO.md:1758-1768` (REVISED ORDER items 2 and 8), `:1737` (fp16-accum DISPUTED row), `:1035`, `:1092`.
- `SVG-CUSTOM.md:7884-7912` (F-230: 1.8e-4 → −2.7% acceptance → −2.8% M1; GDN_CHUNKED ~3e-7 in-band).
- `campaign/research/T26-5-numerics-quality.md` (§b protocol + thresholds, §c fp16-accum error model and KLD
  0.15-0.40 prediction, §f unknowns incl. the self-noise floor).
- `tools/quant_quality.py` (KL(REF||RUN) convention, natural log, cmd_prompt/pcompare patterns, DROP_VALUE handling).
- `tools/calibrate.py` (the header/register style this tool matches).
- `src/program/generate.cpp:15370-15386, 16825-16845, 6955-6986, 949-950` (the three logits-dump hooks + help text).
- `include/strata/core/verify.hpp:330-333` (STRATA_LOGPOS contract, `copy_logits`), `src/core/verify.cpp:3361+`.
- `serve/server.py` (endpoint map — POST /v1/chat/completions; engine I/O class: the server returns tokens, not
  logits — why the dump must hook engine-side).
- `AGENTS.md` (Campaign Doctrine laws 2/4/5; working-on-the-code pointers).
- `.venv`: python 3.15.0, numpy 2.5.3 (both exercised).
