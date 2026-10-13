# THE NON-REGRESSION ULTIMATUM — binding operating law for this workspace

Issued by the operator on 2026-10-12. This file is the owner of the law. `AGENTS.md` carries the short
form that binds every session that opens this workspace; `SVG-CUSTOM.md` carries the evidence.

This law exists because of a specific, proven failure. On 2026-10-12 the campaign had reported v2 as
"ahead of v1 on every axis" on the strength of short explainer probes and concurrent batteries. The
operator then ran his real workload — one sustained reasoning stream — and read ~80 t/s against v1's
116. A 31% regression had been shipped, reported as a win, and argued for. It was real: one arm
(`STRATA_DF_PDL=1`) cost ~40% on the real class, and the adoption gate had never measured that class.
The doctrine below makes that failure mode structurally impossible to repeat.

Nothing in this file is advisory. A session that cannot satisfy it does not get to make a claim.

---

## 1. THE SIX METRICS (the standing scoreboard)

Every change is judged on these six, and on nothing else. They are measured on the live lane, on the
operator's real workload class, in a matched window.

| # | Metric | Exact probe | Never below |
|---|---|---|---|
| M1 | **Single-stream decode** | one request, sustained 2000-token default-reasoning generation (`six-metric.ps1` single-stream probe), t/s | its own best measured value |
| M2 | **Concurrent aggregate** | pair and 6-conc aggregate t/s on the same window (`six-metric.ps1` pair + 6-conc) | its own best measured value |
| M3 | **Accepted draft tokens per millisecond** (amended 2026-10-12 by operator decision; was "MTP draft acceptance") | accepted drafts per verify window ÷ that window's milliseconds, from the engine's `strata decode timing:` line. Reported BESIDE it, never instead of it: the acceptance rate (`drafts accepted A of B`, %), the banked tokens per window, and the per-position vector (`STRATA_SPEC_DEPTH=1`) | its own best measured value |
| M4 | **MMLU-Pro** | `mmlu-pro-strata2.py`, 70 questions, ≥2 runs | its own best measured value |
| M5 | **Mimic** | `mimicbench`, ≥2 runs, plus compliance and empty-answer count | its own best measured value |
| M6 | **Every other lane's t/s** | depth prefill and depth decode at the standing depth prompt | its own best measured value |

"Never below its own best" is the ultimatum. A change that raises M1 while lowering M4 is not an
improvement; it is a trade, and a trade may only ship when the trade is named, both numbers are
printed side by side, and the operator has accepted it explicitly.

**The ceiling that proves the bar is reachable:** M1 has been measured at 215.4 t/s on this hardware
(cached prefix, F-62) and 196 t/s aggregate at 6-conc (F-41). The bar is not theoretical.

## 2. THE BAR

- M1 sustained single-stream: **232+ t/s** (2x v1's measured 116) — the entry bar.
- M1 stretch, the stated target: **290+ t/s** (2x the best v2 number at the time the bar was set).
- All six metrics at or above v1 in the **same matched window**, and at or above their own best,
  for **three consecutive stable sessions**, with the chart trending up.

The bar does not move down to meet results. If a round cannot move the chart, the round reports the
raw numbers and exactly which item failed.

## 3. THE PROOF LAW — no claim without a probe

Every performance or quality claim carries all six of these, in the message where it is made:

1. the probe (command or script, and the prompt class it used),
2. the window (date, time, lane state — GPU-free MiB, boot class, external pressure),
3. the comparator measured **in the same window** (v1, or the previous config — never a remembered number),
4. the raw numbers for both sides,
5. the number of repeats and whether the effect exceeds the measurement floor,
6. the source artifact path (log, JSON, or measurement file) that a hostile auditor can open.

A claim missing any of the six is a hypothesis and must be labelled `HYPOTHESIS`.

## 4. THE NO-SIDESTEPPING LAW

A regression report from the operator outranks any measurement the agent has made. The response order
is fixed and immediate: **reproduce → matched-anchor → bisect → fix → record.** Arguing, resisting,
explaining the number away, blaming the lane without a matched measurement, or answering a regression
report with a different metric is a violation. The operator's perception of his own daily driver is
data; treat it as the highest-grade signal available.

## 5. THE ADOPTION GATE (what a change must pass before it is called an improvement)

1. Baseline six-metric run on the standing recipe, in-window.
2. Candidate six-metric run on the same window, immediately after (or interleaved A/B).
3. M1 and M2 both **at or above** baseline. No exceptions, no "the other class will carry it".
4. M3–M6 in-class (not below their own best; MMLU/mimic not below the v1 class).
5. Only then: adopt, commit the launcher change with the measured numbers inline and this file as the
   gate reference, and log the entry.

An arm that fails 3 or 4 is **rejected and recorded as rejected**, with its numbers, even if it wins
elsewhere. An arm adopted without step 1 and step 2 on the real class is not adopted — it is untested.

**The upstream-default rule.** Before arming any environment arm, read the upstream default and the
measured comment beside it in `src/`. If upstream turned that arm off for this platform, arming it
requires the operator-class probe to beat the default by more than the measurement floor. Upstream
measured it on real hardware; that measurement is evidence, and ignoring it is guessing.
(`STRATA_ADAPT_OVERLAP` on Windows is the standing example: `src/program/generate.cpp:1426`.)

## 6. THE TREND LAW

Progress is a chart that goes up, not a spike here and there. `PERF-CHART.md` is that chart. It holds
one row per session per metric, with lane state and source. Every session either adds a row that is
higher than the last on at least one metric with no metric lower, or it adds a row that documents
exactly why not, with raw numbers. A session that adds neither has produced nothing and says so.

## 7. THE ARCHAEOLOGY MANDATE

Every change made in the last 18 hours is re-audited against the six metrics. Anything that made any
metric worse is found, proven with raw numbers, and either fixed or retracted. Nothing stays in the
standing recipe that cannot show its own measurement on the real workload class. The standing recipe
is not a museum of adoptions; it is the set of levers that survived their own gate.

## 8. THE OUTSIDE-THE-BOX MANDATE

The environment-arm space is exhausted and it yielded single digits. The bar needs structural work:
kernel-boundary and launch overhead, window economics, acceptance (the trained head), residency and
routing, KV depth, prefill kernels, and first-principles re-derivations. When a limit feels reached,
the next lever is already on the board. The shallow surface is closed; go under it.

## 9. THE SAFETY INVARIANTS

- MAGMA v1 stays frozen at `8da8bf16`, triple-pushed, untouched, one launcher away.
- All work happens in `strata-custom-2`, branch `custom`, upstream-pullable (`rerere` primed,
  `STRATA_KEEP_MAPPING=1` carried by every launcher built on .43+).
- Test traffic never pollutes production artifacts: the learned expert profile and any learned state
  are restored before the lane is handed back.
- The lane is handed back serving the adopted recipe, never a test build.

## 10. THE REPORTING CONTRACT

When results come back, the report leads with the six metrics as measured, in a table, with the v1
anchor and the previous best beside them. Wins are stated as wins only where the probe produced them.
Negatives are stated first, not buried. Every claim traces to a row in `PERF-CHART.md`.

## 11. COMPLETION

The goal completes when M1–M6 are each at or above their best measured value **and** the bar is met in
three stable consecutive sessions with an upward chart, every adoption carrying both-classes proof and
quality in-class. Until then the standing order is: work the queue, gate on both classes, record every
number, and never report a win the probe did not produce.

## 12. WHY THIS IS PERSONAL, NOT ACADEMIC

This engine is the operator's daily driver and the thing he is judged on. Reporting "we are better"
while the machine got 31% slower is an integrity failure, not a measurement error. The law above is
the minimum standard for being allowed to touch the lane. Take it seriously.
