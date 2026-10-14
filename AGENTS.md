# AGENTS.md

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

## The performance campaign law (svg's fork — ENFORCED on every optimization; supersedes nothing above, adds to it)

This repo is the Linux superiority campaign lane. Read `campaign/GOAL-LINUX-SUPERIORITY.md` (the
queue and bar table) and the last F-entries in `SVG-CUSTOM.md` before proposing changes. The
following are binding for ANY performance-affecting change — env arm, config, engine code, kernel
port, merge:

1. **BEFORE/AFTER, ALWAYS.** Before any change: 3× `campaign/probe-lane.py` on the standing recipe,
   same boot or back-to-back boots, SAME HOUR (this box drifts ±8% across hours — stale baselines
   are invalid; when in doubt, re-run the control immediately after the arm). After: 3× more,
   back-to-back with the control. Claims are made on MEANS only.
2. **Adopt bar.** The targeted metric gains ≥ +4% AND no priority metric regresses beyond −4%
   (run noise). Inside the band = not a win; record and revert.
3. **The priority metrics — ALL checked, every round:** M1 single-stream (probe "M1" line), M2 pair
   aggregate, M2 6-conc aggregate, M3b acceptance rate, prefill rate (probe prefill lines or
   `campaign/depth-lane.py`), MMLU gate (`campaign/mmlu-pro-gate.py 0.933 <label>`) whenever
   anything token-stream-affecting changes — 2 gates if the first is borderline; escalation floor
   93.3. Adherence (mimic) when the tool is ported.
4. **SINGLE-STREAM PROTECTION (explicit):** M1 must not regress beyond noise for ANY change,
   including changes that target concurrency. A concurrency-for-M1 trade is a SHELF VARIANT
   (labeled, in `~/LLM/backup/`), never a standing adoption — F-183 precedent, re-tested and
   confirmed twice.
5. **One variable per leg.** Diagnostic boots (telemetry env) are not arms. Sustained-reasoning
   probe is MANDATORY before adopting anything launch/overlap class (PDL precedent: a Windows −40%
   lived there).
6. **Record everything, wins AND negatives:** an F-entry in `SVG-CUSTOM.md` + a git commit with the
   full numbers + result files under `campaign/results/` (probe-lane.py does not save files — pipe
   through `tee campaign/results/<label>.txt`).
7. **Revert cleanly on failure:** restore from git or `~/LLM/backup/`, re-boot, and VERIFY the
   restored state (env on the engine process + /health) before declaring the round closed.
8. **The Launch Law:** never run `setup.py`/`setup.sh` on this lane (it rewrites config args);
   start via the launchers. Engine rebuilds deploy by `cp build/strata engine/strata` with a
   rollback copy kept alongside.
