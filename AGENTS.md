# AGENTS.md

> **CUSTOM LAYER — BINDING LAW, READ FIRST.** This checkout is MAGMA v2 (`strata-custom-2`), the
> operator's daily driver, under a standing order to beat MAGMA v1 (frozen at `8da8bf16` in
> `../strata-custom`) on every metric with a monotonically upward trend. Before any change, any
> measurement, or any claim, read **[AGENTS.d/extractions/2026-10-12-non-regression-ultimatum.md](AGENTS.d/extractions/2026-10-12-non-regression-ultimatum.md)**
> and the standing chart **[PERF-CHART.md](PERF-CHART.md)**. The short form lives at the bottom of
> this file. Violating it is not a style issue; it has already cost 31% on the real workload once.

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

---

# CUSTOM LAYER — THE NON-REGRESSION ULTIMATUM (binding on every session in this checkout)

Owner: [AGENTS.d/extractions/2026-10-12-non-regression-ultimatum.md](AGENTS.d/extractions/2026-10-12-non-regression-ultimatum.md).
Chart: [PERF-CHART.md](PERF-CHART.md). Evidence: [SVG-CUSTOM.md](SVG-CUSTOM.md).

**THE SIX METRICS — NONE MAY EVER REGRESS.** (M1) single-stream decode on the real workload (sustained
2000-token default-reasoning generation); (M2) concurrent aggregate (pair and 6-conc); (M3) **accepted draft
tokens per millisecond of verify window** (amended 2026-10-12 by operator decision — the objective the
engine's own draft policy already optimises; the acceptance RATE, the banked tokens per window and the
per-position vector are reported beside it, never instead of it); (M4) MMLU-Pro; (M5) mimic, compliance, zero
empties; (M6) every other lane (depth prefill, depth decode). A change that lifts one and lowers another is a
trade, and a trade ships only when both numbers are printed side by side and the operator accepts it explicitly.

**WHY M3 WAS AMENDED (2026-10-12, evidence in `.docs/digest/speculative-draft-yield-and-acceptance-metrics.md`).**
The old wording — "MTP draft acceptance" — named a ratio whose denominator is the number of drafts *offered*,
so it falls whenever the chain is made to draft deeper even when throughput and banked tokens both rise. Eighteen
comparables were studied from primary sources and **not one treats that rate as its objective or as a
non-regression gate**: vLLM names `mean_acceptance_length`, SGLang `spec_accept_length`, TensorRT-LLM
`acceptance_length`, MLC-LLM cumulative `accept_len`, EAGLE's τ = tokens accepted per forward pass; EAGLE-2
declines to publish the rate at all ("it only reflects the performance of the draft model"); HuggingFace TGI
ships speculation with no acceptance metric whatsoever; and vLLM's own tuning PR measured the decisive case —
the highest acceptance length (2.27) ran **19% slower** than a lower one (1.86). The rate remains a reported
diagnostic; it is no longer the defended quantity.

**THE FIELD-PRACTICE CLAUSE (added 2026-10-12 by operator decision).** A decision that ships or rejects a trade
must cite the published practice of comparable engines alongside its own numbers — the flag, metric or
mechanism each one uses, with a source — because a veto reached on our numbers alone is how a defensible trade
gets rejected. The citation lives in `.docs/digest/`, its backing in `.docs/research/`, verbatim identifiers in
`.docs/research/rips/`. Research that is not written down did not happen and cannot be cited.


**THE BAR.** 232+ t/s sustained single-stream (2x v1's 116), target 290+. This is reachable: M1 has
measured 215.4 t/s on this card (F-62) and 196 t/s aggregate at 6-conc (F-41). The bar never moves down.

**THE PROOF LAW — NO GUESSING, ONLY PROVING.** Every claim carries: the probe, the window and lane
state, the comparator measured **in the same window** (never remembered), the raw numbers both sides,
the repeat count against the measurement floor, and the artifact path an auditor can open. Anything
less is labelled `HYPOTHESIS`.

**THE NO-SIDESTEPPING LAW.** An operator regression report outranks any measurement the agent has made.
The order is fixed: **reproduce → matched-anchor → bisect → fix → record.** Arguing, resisting,
explaining away, or answering with a different metric is a violation.

**THE ADOPTION GATE.** Baseline six-metric run, then candidate on the same window, then M1 and M2 both
at or above baseline and M3–M6 in-class — or the arm is rejected and recorded as rejected. Read the
upstream default and its measured comment in `src/` before arming any env var; if upstream turned it
off for this platform, beating the default by more than the noise floor is the price of arming it
(`STRATA_ADAPT_OVERLAP` on Windows is the standing example, `src/program/generate.cpp:1426`).

**THE TREND LAW.** Progress is a chart that goes up, not a spike here and there. Every session adds a
row to `PERF-CHART.md`: higher on at least one metric with none lower, or an honest entry saying
exactly which item failed with raw numbers.

**THE ARCHAEOLOGY MANDATE.** Every change from the last 18 hours is re-audited against the six metrics.
Anything that made a metric worse is proven with raw numbers and fixed or retracted. Nothing stays in
the standing recipe that cannot show its own measurement on the real workload class.

**SAFETY.** MAGMA v1 stays frozen at `8da8bf16`, untouched, one launcher away. All work in this
checkout, upstream-pullable (`rerere` primed, `STRATA_KEEP_MAPPING=1`). Test traffic never pollutes
production artifacts. The lane is handed back serving the adopted recipe, never a test build.
