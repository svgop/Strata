# F-208 — MEASUREMENT LANES: prefill battery, mimic port, acceptance denominator (design agent, 2026-10-15)

Closes the three unauditable goal lanes identified in `F-208-campaign-audit-agent.md` §(D):
**prefill**, **mimic adherence**, **acceptance-vs-stock**. Everything below was verified in source
in BOTH trees (`~/LLM/strata-custom-2`, `~/LLM/strata-upstream-0144`) — no builds, no server
contact. This file is the spec; implementation is a later step. Design agent wrote nothing else.

**Headline correction to F-198 note 3:** stock servers DO report per-request draft acceptance and
fresh-only prefill timing. F-198's "M3b/accept accounting lines did not print on the stock servers"
was a probe misconfiguration: `campaign/results/f198-up-1.txt` line 1 shows
`log=/home/svg/LLM/strata-custom-2/custom-iq3_s.log` (the probe default) while the stock boot wrote
its own log. That log — `/home/svg/LLM/strata-upstream-0144/strata-up.log` (Oct 14 23:04, the F-198
window) — contains **116 `... drafts accepted X of Y` lines** in the identical format. Lane 3 needs
no engine change and no new theory; it needs `--log` pointed at the boot under test.

---

## 0. What the servers actually expose (verified, both trees)

These are the primitive surfaces all three designs build on. Identical in both trees unless noted
(`serve/` is byte-identical except `server.py`; the only `server.py` diffs are fork F-14 admission
tickets and fork F-19 answer-space reservation — neither touches the surfaces below).

1. **Per-request engine serve line** (stderr → boot log), `src/program/generate.cpp` — custom
   :13292, upstream :12995, byte-identical format:
   ```
   strata serve: prompt <N> tokens = <R> reused + <F> read in <P> ms (<S> tok/s), <G> generated in <D> ms (<T> tok/s), drafts accepted <A> of <B>, <C> checkpoints
   ```
   `<S>` = `1000*F/P` — the **fresh-only cache-miss prefill rate** (the F-192 number). `<A>/<B>` =
   draft acceptance per request. This is what probe-lane.py's accounting section parses.
2. **Response `timings` block** (non-streaming `/v1/chat/completions`), `serve/server.py`
   `request_timings()` :4475, attached at :4436/:4753 — llama.cpp names, from the engine's own clock:
   ```json
   {"cache_n": 0, "prompt_n": 157195, "prompt_ms": 24057.8, "prompt_per_token_ms": 0.153,
    "prompt_per_second": 6534.1, "predicted_n": 266, "predicted_ms": 2279.3,
    "predicted_per_second": 116.7, "draft_n": 253, "draft_n_accepted": 158}
   ```
   (Real capture against the live 8081 server: `campaign/results/depth-262k-d130k.txt`.)
   `prompt_n` = prompt minus cache; `prompt_per_second` = fresh-only prefill rate; `draft_n`/
   `draft_n_accepted` = acceptance — **client-side, no logs needed, works on both engines**.
3. **`GET /metrics`** (Bearer): `history` = last 500 finished requests, each carrying `prompt_ms`,
   `decode_ms`, `drafts_offered`, `drafts_accepted`, `prompt_total`/`prompt_read`, `decode_tok_s`,
   `hit_rate`; plus `totals` (sums incl. drafts) and `live.{state,phase,tok_s,prefill_tok_s_mean}`.
   `prefill_tok_s_mean` is the LAST request's running mean (the "misleading mean" of F-192) — never
   score from it; use history rows or the response block.
4. **Window line** `strata decode timing: <W> windows, avg T <t>, <tok> tokens/window, <ms> ms/window ...`
   — `generate.cpp` custom :13058 / upstream :12767, gated by **`STRATA_DECODE_TIMING=1`** (custom
   :12127, upstream :11884) in BOTH trees. Not set on either standing boot (only
   `launch-diag-profile.sh` sets it; zero occurrences in the current custom log). Gives avg T →
   derived accepted/s `1000*(T-1)/ms_window` and banked tokens/window on either engine (equal footing
   for M3-class rows) from a diagnostic boot.
5. **PP protocol** (stdout): `PP <n> <total> <ms> <mean tok/s>` per prefill chunk; parsed at
   `server.py` :1155/:1838 into progress + `prefill_tok_s_mean`. Heartbeat only — not a scoring surface.

**Boot recipes (reference):** custom = `launch-custom-iq3s-8081.sh` (standing arms; config
`custom-iq3_s-8081.json`: `--spec 6 --spec-min-p 0.5 --max-context 131072 --prefill auto:32768
--parallel 6`, api_key set, log `custom-iq3_s.log`). Stock = `~/LLM/strata-upstream-0144/launch-stock.sh`
(config `stock-up-8081.json`: `--spec 4 --spec-min-p 0.7 --max-context 262144 --prefill auto:32768
--parallel 4`, NO api key, log `strata-up.log`). Same GGUFs/pack/mtp-rt/cvec/kv-int8/kv-resident
32768/pcie 0.35 dma/adapt-decay (F-198 recipe; pack+artifacts via tree-local symlinks/copies). Both
launchers `exec .venv/bin/python serve/server.py --engine strata --config ... --port 8081`.
**VRAM allows ONE engine at a time (~29-32 GB of 32)** — cross-engine windows are sequential boots
on the same port (planned downtime), exactly the F-198 pattern.

---

## 1. LANE: PREFILL BATTERY (new `campaign/probe-prefill.py`)

**Why this design:** probe-lane.py has no prefill metric (its `fresh-long-prompt` is ~3.6k tokens
and reports decode only). The F-192 correction says the honest number is the fresh cache-miss rate,
4,600-6,600 tok/s on 23.7k/44.4k/52.9k new tokens (~5,700 class). Surfaces 1+2 above give exactly
that, per request, on both engines.

### 1.1 Probe design

- **Freshness (cache-miss) is asserted, not assumed:** every request carries a unique nonce sentence
  at the FRONT of the prompt (prefix match dies on token 1). After each response, require
  `timings.cache_n == 0` and `timings.prompt_n >= 0.95 * target`; otherwise discard the rep and note.
- **Prompt body:** reuse depth-lane.py's deterministic filler verbatim (`random.Random(4242)`,
  `SECTION` template, ~62 tok/section estimate) so token counts match the M6a/depth lineage; actual
  count is `timings.prompt_n` (engine's own tokenizer — ground truth, cross-engine comparable because
  both serve the same GGUF/pack tokenizer).
- **Size classes (fresh prompt_n targets):** 4k, 16k, 32k, 64k, 96k. Rationale: 4k = probe-class
  context; 16k/32k straddle the kv-resident 32768 boundary; **64k is the M6a bar class** (goal
  register: Windows 1490.7 contended / 7590.6 best); 96k is the long-session class (F-192's 52.9k+
  traffic) and fits BOTH engines' max-context (custom 131072; stock 262144 — do not exceed ~120k or
  the custom boot's own limit is the binding constraint, which would make the row unmatched).
- **Per rep:** temp 0 (prefill measurement must not depend on sampling), `max_tokens` 24, one short
  forced question after the filler ("How many distinct record numbers appear? Answer in one short
  sentence."), non-streaming. Record: `prompt_n`, `prompt_ms`, `prompt_per_second`, `cache_n`,
  `predicted_per_second` (sanity), wall time, label, boot fingerprint.
- **Reps and scoring:** 3 reps per class per boot, same hour, **means only** (law 1). Per-class mean
  of `prompt_per_second`; the scoreboard headline is the **64k class mean** (bar alignment) plus the
  token-weighted mean over all classes. Collapsed reps (any class −30%+ vs boot median across ALL
  classes simultaneously) follow the f197-analyze collapse discipline: drop and NOTE, never silently.
- **Warm placement:** run AFTER one warmup request; prefill has no cold-cache story we want to
  measure here (expert cache/adapt tier warm state matches how the goal metrics are otherwise taken).
  Keep the order fixed: 4k→96k ascending, all reps of a class consecutive.

### 1.2 Script sketch (the whole instrument)

```python
#!/usr/bin/env python3
# campaign/probe-prefill.py — fresh-context (cache-miss) prefill battery. Both engines.
# Usage: .venv/bin/python campaign/probe-prefill.py --port 8081 --label cc-stand | tee campaign/results/prefill-cc-stand.txt
import argparse, json, random, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8081)
ap.add_argument('--label', default='prefill')
ap.add_argument('--key', default='3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--classes', default='4000,16000,32000,64000,96000')
a = ap.parse_args()
# filler: copy build_filler()/SECTION/WORDS verbatim from campaign/depth-lane.py (seed 4242)
def one(port, key, prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 24, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    return d["timings"], time.time() - t0
for target in [int(x) for x in a.classes.split(',')]:
    rates = []
    for r in range(a.reps):
        nonce = f"Run stamp {a.label}-{r}-{time.time_ns()}: ignore this sentence. "   # front nonce = no prefix reuse
        t, wall = one(a.port, a.key, nonce + build_filler(target) + QUESTION_TAIL)
        assert t["cache_n"] == 0, f"cache hit! {t}"
        rates.append(t["prompt_per_second"])
        print(f"  {a.label} {target:6d}[{r}] prompt_n={t['prompt_n']:6d} "
              f"prompt_ms={t['prompt_ms']:9.1f} prefill={t['prompt_per_second']:7.1f} t/s", flush=True)
    print(f"PREFILL {target:6d} : mean {sum(rates)/len(rates):7.1f} t/s over {len(rates)} fresh reps")
```

Equivalent single-shot curl form (ad-hoc checks):
```bash
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" -d '{"messages":[{"role":"user","content":"'"$NONCE $FILLER"' Q"}],"max_tokens":24,"temperature":0}' \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print(t["prompt_n"], t["prompt_ms"], t["prompt_per_second"])'
```

### 1.3 Running against BOTH engines (F-198 recipe reference)

- **Custom (standing):** against live 8081, no boot: `.venv/bin/python campaign/probe-prefill.py --port 8081 --label cc-<window> | tee campaign/results/prefill-cc-<window>.txt`
- **Stock:** planned-downtime boot per F-198: stop custom → `~/LLM/strata-upstream-0144/launch-stock.sh`
  (boots 8081; log lands in `strata-up.log` per its config `log` key) → warmup →
  `probe-prefill.py --port 8081 --key "" --label up-<window> | tee campaign/results/prefill-up-<window>.txt`
  (stock has no api key; sending the custom key is harmless but pass empty for cleanliness).
  Cross-check two reps against surface 1 by `grep "strata serve: prompt" strata-up.log`.
- **Same-hour law:** the stock boot and a custom re-boot must be in the same window, with the custom
  leg re-run immediately after (control-after-arm, law 1). See §4 for the exact sequence.
- **Also run inside every future battery** (both legs): ~3-4 min added per boot.

### 1.4 Scoreboard row format

```
| boot | window | 4k | 16k | 32k | 64k | 96k | tok-w mean | vs stock 64k |
|------|--------|----|----|----|----|----|------------|--------------|
| upstream stock v0.1.44 | 2026-10-XX w1 | ... | ... | ... | ... | ... | ... | — |
| current @ standing | 2026-10-XX w1 | ... | ... | ... | ... | ... | ... | +X.X% |
```
Goal line: 64k-class mean current ≥ 1.30 × stock 64k-class mean, same window, means of 3.
Retires the stale pre-campaign +13% figure and gives MMQ_KQ (T0.8) the lane it was built for.

---

## 2. LANE: MIMIC PORT (M5 adherence — `campaign/mimicbench.py`)

### 2.1 Source located (Windows-era, frozen disk — copy FROM is legal)

- **The M5 instrument:** `/mnt/DATA/LLM/Engine/llama-custom/bench/measure/mimicbench.py` — "mimicbench
  v1, 64 tasks + 60 drift sub-positions". Pure stdlib (`json, os, re, sys, time, urllib.request`); NO
  dependencies, NO LLM judge — every check is anchored regex/structural. Categories: lexicon 11,
  code-style 3, ui-mimicry 10, precedence 8, format-lock 8, drift ladder 12 (60 scored positions,
  first-drift index), distraction anchor 6, negative-instruction 6 = **64 tasks**. This is the "M5
  mimic 64" of the goal register (Windows bar 100%, ceiling; No-Regression Law binds M5).
- Era variants (context, do not port as-is): `mimicbench-mt4k.py` (max_tokens 4000),
  `mimicbench-strata-cvec0.py` (the strata-server adaptation: mt 4000 + `chat_template_kwargs.
  experimental_speed_projection: false` — this is the closest ancestor for our server),
  `mimicbench2.py` (a different, deeper 40-task suite — NOT the M5 bar; out of scope).
- Last measured band: 90.6-98.4 (best 98.4), pre-campaign (F-20/F-30 era). M5 open since F-173
  ("mimicbench not yet ported — the M5 tool must be located/ported from the Windows campaign",
  SVG-CUSTOM.md :5426-5427). **ROUTE_RESIDENT 0.75 (F-200 ADOPT, output-changing) has never been
  adherence-checked** — this port closes that hole first.

### 2.2 The port (four diffs, everything else byte-identical)

1. **API key:** the Windows tree sent no auth; our standing boot requires Bearer. Add
   `Authorization: Bearer <key>` (default the standing key, `--key` to override; stock accepts
   requests with or without).
2. **Output path:** `OUT` → `campaign/results/mimic-<label>.json` (same dir convention as
   mmlu-pro-gate.py). Pipe console through `tee campaign/results/mimic-<label>.txt` (law 6).
3. **Endpoint:** keep the default `/v1/chat/completions` ladder **byte-identical** (temp 0.1,
   mt 3000, same task corpus, same order) for ladder comparability. The optional `--drift-grammar`
   variant uses the engine-native `/completion` endpoint with a GBNF grammar — **that endpoint does
   not exist in the Linux server** (server.py routes only `/v1/chat/completions|/v1/messages|/v1/responses`;
   verified :5420/:5433/:6837). Drop the variant (or re-implement over structured output later);
   the v1 ladder never needed it (Windows-era strata runs used cvec0 = no grammar).
4. **Args:** `[port] [label] [--think off|low|default] [--key K]` — argparse-grade cleanup of the
   sys.argv indexing, same semantics. Keep `chat_template_kwargs` handling (enable_thinking /
   reasoning_effort — both supported by the Linux server, 7 `enable_thinking` refs in server.py).

### 2.3 Cross-engine comparability rule (the one real trap)

**Fork-only F-19 answer-space reservation:** our `server.py` (:4000-4007 of the diff vs upstream)
auto-reserves `max(384, max_new//4)` tokens of answer space when thinking is on and no budget is
given — added because the measured overrun failure mode was "4/70 MMLU-Pro, **3/64 mimicbench** at
default effort, EMPTY heads" on unreserved budgets (stock behavior). Upstream `server.py` has no
auto-reservation. So default-thinking mimic runs are NOT comparable across engines (fork rescues
empty answers, stock returns them → stock scores artificially low on format/drift tasks).

Rule:
- **Fork-side ladder anchor (vs Windows 100%/98.4 bar):** default thinking, mt 3000 — the fork's
  standing behavior, matches the strata-era instrument (with projection off; keep
  `experimental_speed_projection: false` in the body as cvec0 did, harmless and ladder-faithful).
- **Cross-engine rows:** `--think off` on BOTH engines (no reasoning budget in play at all), OR
  explicit `reasoning_budget_tokens` set equal on both (both server.py trees support the request
  key: fork :3121-3127, upstream has it too). Record the mode in the row. First catch-up session
  runs BOTH modes once on the fork to pin the mode offset, then cross-engine work uses --think off.

### 2.4 What it measures / acceptance bar / cadence

- **Measures:** instruction adherence given in-context rules — verbatim lexicon, convention mimicry,
  design-system discipline, rule precedence, byte-exact format locks, drift onset inside one growing
  context (first-drift index), distance survival (anchor), forbidden-word resistance. Score =
  task-weighted %, plus drift profile. Deterministic; reproducible at temp 0.1.
- **Bar:** goal register M5 = 100 (ceiling, hold); standing band from this stack re-pinned at the
  catch-up session. Adoption law (No-Regression + law 2/3): any output-changing arm whose arm-boot
  M5 mean drops below the standing band's lower edge (or below control same-window by > noise)
  = reject, same as an MMLU gate failure.
- **Cadence — REQUIRED for every output-changing arm** (anything that alters the token stream:
  sampler/rescue/typical changes (STRATA_TYPICAL_*, BATCH_RESCUE), drafter temp (SPEC_DRAFT_TEMP),
  routing/residency (ROUTE_RESIDENT*), floor/min-p, spec depth, cvec, draft vocab, engine merges,
  quant changes). Run 1x on the arm boot and 1x on the same-hour control boot, inside the battery
  window; escalate to 3x when first adopting or within 1.5 pts of the bar. Exempt: pure
  scheduling/env arms (POOL_SPIN, DMA_BATCH, PLE_BATCH, ASYNC_LAUNCH, L2_SETASIDE, PIN_BY_NEED,
  vram-reserve) — proven no-output-effect class.
- **Wall time:** 64 tasks, outputs mostly 50-400 tok at ~150-180 t/s; largest prompts ~8k (anchor-06)
  → **~5-10 min per run** against this engine.
- **Immediate debt:** first run on the standing stack (with RR 0.75 live) = the adherence check the
  RR 0.75 adoption owes (audit C5/E9).

---

## 3. LANE: ACCEPTANCE DENOMINATOR (equal footing, no fork-side stats)

### 3.1 What each engine reports (exact, from source)

| surface | custom | stock v0.1.44 |
|---|---|---|
| per-request log line `... drafts accepted A of B ...` | generate.cpp:13292 | **generate.cpp:12995 — IDENTICAL format**; proven live in F-198's own stock log (116 lines, strata-up.log) |
| response `timings.draft_n` / `draft_n_accepted` | server.py request_timings :4475 | same file, same code (server.py diffs are F-14/F-19 only) |
| `GET /metrics` history `drafts_offered/accepted` + totals | :4363/:2984 | same |
| window line `avg T, tokens/window, ms/window` | :13058, env `STRATA_DECODE_TIMING` :12127 | :12767, same env name :11884 |
| M3d per-position `accepted/tested` | fork-side `STRATA_SPEC_DEPTH` diagnostic | NOT in stock (this, and only this, is fork-side) |

**F-198 note 3 is thereby corrected:** the probe ran stock boots with `--log` defaulting to the
custom log (f198-up-1.txt:1), so its engine-accounting section was empty by construction.

### 3.2 Primary metric — same-probe-set means from the responses (zero new theory)

Extend probe-lane.py's `probe()` to also return the response `timings` block (it already has `d` in
hand — a three-line change), and print per-run:

```
M3b rate (responses) : 77.9% mean over 15 requests (drafts 12871/16522)     <- works on ANY engine
```

Standing recipe stays as-is (temp 0.7, same prompts, same order) → token streams remain comparable
(same request bytes, same template — `chat_template.jinja` byte-identical in both trees, same
tokenizer artifacts). Score = mean of per-request `draft_n_accepted/draft_n` over the run's
long-form requests (of >= 8 drafts, as today), 3 runs per boot, means only. The scoreboard's
acceptance row then has BOTH engines in the same window:

```
| boot | window | M3b % (resp) | drafts acc/of | banked/window (diag) | accepted/s (diag) |
| upstream stock | w1 | ... | ... | ... | ... |
| current        | w1 | ... | ... | ... | ... |
```

Known semantic caveat (record per row, don't hide): acceptance is engine-recipe-semantic — the fork
counts typical-rescue/DT-0.85-class banked tokens in its accepted figure; stock counts its own.
That is exactly what "acceptance vs stock" means for the goal (each engine's realized rows/window);
the derived cross-check below catches any accounting drift.

### 3.3 Cross-check — derived acceptance from throughput (diagnostic boots, optional but cheap)

One boot per engine with `STRATA_DECODE_TIMING=1` (diagnostic class, law 5 — never a perf row):
parse the `avg T` window line on BOTH (same format) → `banked tokens/window = avg T − 1`,
`accepted/s = 1000*(T−1)/ms_window` (probe-lane already computes this; it just never had stock
lines to read — now `--log` gets pointed correctly). Consistency check: fork M3b and derived
banked/window must move together vs stock; if M3b gains but banked/window does not, the offer
volume (drafts/window) changed and the "rate" gain is bookkeeping, not rows.

**Anti-method (rejected):** greedy-determinism signature counting (diff token streams between
engines at temp 0) — the fork's kernel/quant changes make logits differ at noise level, so streams
legitimately diverge; not an acceptance instrument here.

### 3.4 Probe-hygiene requirements

- `--log` MUST match the boot under test (custom: `custom-iq3_s.log`; stock: `strata-up.log` — the
  config `log` key of each tree). Add `--log` reminder line to the probe header print.
- Stock has no api_key; always send the Bearer header anyway (ignored) so one script serves both.
- Keep M3d/`STRATA_SPEC_DEPTH` clearly fork-side-diagnostic (as labeled today); never a stock row.

---

## 4. CATCH-UP SESSION (first measurement session; ~2 h wall, 2 planned boots)

**All seven lanes become scorable from this one window.** VRAM allows one engine at a time, so this
is the F-198 sequential-boot pattern; the two boots are one same-hour window (law 1's
control-after-arm satisfied by re-running current immediately after stock).

Phase 0 — prep, no boots, no downtime (~30-45 min, do it before the window):
1. Write `campaign/probe-prefill.py` (§1.2) and the probe-lane.py timings patch (§3.2; also add
   `--log` echo, prefill block invocation, and `--skip-prefill`).
2. Port `campaign/mimicbench.py` per §2.2; smoke-test 3 tasks against live 8081 (default + --think off).
3. `f197-analyze.py`: extend `parse()` for the prefill + response-M3b lines (means table gains
   `pref64` and `M3b*` columns).

Phase 1 — standing custom, LIVE boot, no downtime (~40 min):
4. `probe-lane.py --label cc-<w>` x3 (M1/pair/6-conc + response-M3b now inline) — refreshes the
   current leg and RR-0.75-era throughput controls.
5. `probe-prefill.py --label cc-<w>` x3 (inline in probe-lane or standalone; ~3-4 min each).
6. `mimicbench.py 8081 cc-<w>` x3 (~5-10 min each) → **closes the RR 0.75 adherence debt** and pins
   the standing M5 band (both think modes once if time allows).

Phase 2 — planned downtime, stock boot (~50 min):
7. Stop custom (record boot console per law), `~/LLM/strata-upstream-0144/launch-stock.sh`, wait
   ready (3-6 min load), warmup.
8. `probe-lane.py --port 8081 --log /home/svg/LLM/strata-upstream-0144/strata-up.log --label up-<w>` x3
   → M1/pair/6-conc **and the first-ever stock M3b rows**.
9. `probe-prefill.py --label up-<w>` x3 → **the stock prefill row** (retires the stale +13%).
10. `mimicbench.py 8081 up-<w> --think off` x3 → stock adherence denominator (optional; cheap).
11. Optional MMLU stock row while booted: `mmlu-pro-gate.py 0.7 up-<w>` (closes the last audit open
    question; note the gate's `strata_tune` per-request floor is a no-op-or-absent on stock — the
    stock floor is the boot's 0.7, pass 0.7 so the label is honest). ~10-15 min.

Phase 3 — back to standing custom, same window (~40 min):
12. Re-launch via `launch-custom-iq3s-8081.sh`, VERIFY env + /health (law 7 discipline), then:
    `probe-lane.py --label cc-<w>b` x3 + `probe-prefill.py --label cc-<w>b` x3 + `mimicbench.py 8081 cc-<w>b` x1.
    Current-vs-stock comparisons use cc-<w>b (adjacent) with cc-<w> as the within-day second point.

Phase 4 — record (~30 min, offline):
13. `f197-analyze.py cc-<w> cc-<w>b up-<w>`; write the scoreboard block (§1.4/§3.2 row formats);
    F-entry + results files (law 6). Journal the F-198 note-3 correction explicitly.
14. Optional later session (separate diagnostic boots, never perf rows):
    `STRATA_DECODE_TIMING=1` one boot per engine (custom: `launch-diag-profile.sh` pattern; stock:
    a launcher copy with the env) for banked/window + accepted/s equal-footing rows (§3.3).

Wall-time total: ~2.0-2.5 h including both boots — one evening window, matching how F-198 ran.

## 5. Law-compliance notes

- Same-hour 3+3, means-only: §1.1, §2.4, §3.2, §4 Phases 1-3.
- ABAB for arms: the control-after-arm re-run is Phase 3; future arm batteries run
  control(3) → arm(3) → control(3) with prefill+mimic inside each leg (mimic only for
  output-changing arms, §2.4).
- No downtime beyond planned battery boots: everything client-side except Phases 2-3 (one stock
  boot cycle), which is a planned battery boot by definition.
- Token streams comparable across engines: same request bytes, byte-identical chat template and
  tokenizer artifacts, explicit think-mode equalizer (§2.3), freshness asserted via `cache_n` (§1.1),
  no fork-only request keys (`strata_tune`) in cross-engine probes.
- Nothing here touches the engine, the launchers' standing arms, or the live server.

## 6. Sources (all verified in this session)

- `AGENTS.md` law; `campaign/GOAL-LINUX-SUPERIORITY.md` (bars, M5/M6/M3b rows); `campaign/F-198-entry.md`;
  `SVG-CUSTOM.md` F-173 (:5401, M5 open) / F-192 (:6464) / F-194 prefill correction (:6691);
  `campaign/research/F-208-campaign-audit-agent.md` §(A10)/(A11)/(D).
- Code: custom `src/program/generate.cpp` :12090 (prompt_ms), :13212 (DONE), :13292 (serve line),
  :13058 + :12127 (window line + env); upstream `generate.cpp` :12924, :12995, :12767 + :11884;
  upstream `src/qwen35/q36_engine.cu` :2488/:2509 (standalone q36 tool prints prompt ms + MTP accept —
  not the serve path). `serve/server.py` (both trees): :2984 history, :3476/:3577-3580 metrics extra,
  :4363 history row, :4436/:4753 timings in response, :4475 request_timings; fork-only diff = F-14
  tickets + F-19 reservation (:4000-4007 of diff). Configs: `custom-iq3_s-8081.json` vs
  `stock-up-8081.json` (full arg diff in §0).
- Frozen-tree tools: `/mnt/DATA/LLM/Engine/llama-custom/bench/measure/mimicbench.py` (+`-mt4k`,
  `-strata-cvec0` variants, `mimicbench2.py` distinct suite, `prefill-probe.py` ancestor, MEASUREMENT-RIG.md).
- Live proof artifacts: `campaign/results/depth-262k-d130k.txt` (timings block incl. draft_n),
  `campaign/results/f198-up-1.txt` (log misconfiguration, line 1),
  `/home/svg/LLM/strata-upstream-0144/strata-up.log` (116 acceptance lines from the stock boot),
  `campaign/results/depth-262k-d130k.txt` prefill 6,534 t/s @157k fresh.
