# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## The Campaign Doctrine (how this fork thinks)

The standing results — M1 198.5 t/s (+35.0% vs stock), pair 197.6 (+32.5%), 6-conc 312.4 (+66.4%),
MMLU-Pro 95.0, 300+ t/s concurrent on one GPU — were produced by a method, not by luck. The method
is below, as law. Each law is universal: it states an engineering truth, not an engine fact, and each
was earned on this campaign's record at full price. The binding *procedure* for performance changes
lives in "The performance campaign law" at the bottom of this file; this section is the thinking that
procedure serves. Citations point into `SVG-CUSTOM.md` (F-entries) and `campaign/research/`.

1. **THE INTERLEAVE SETTLES IT.** No comparison is valid across boots, boxes, or warm states: the
   world moves more than most effects worth measuring. Alternate arm and control inside one time
   window (ABAB, 3v3), decide on means only, and never score a cold first boot against a warm third —
   the warm protocol alone is worth ~+10% on paper, and it belongs to nobody's arm. A single-boot
   number is a rumor; a same-window interleave is a verdict.
   *(Earned by: F-191 — arm and control within ±1.2% of each other while the box moved −5% under
   them; F-225 — the planner's "all-time peak" was the session warm trend, reversed by the formal
   3v3; F-231 — the falsifier's 267.5 conc was one lucky boot, the interleave read +1.81%.)*

2. **NEVER REGRESS.** An optimization that wins its lane and loses another is a regression wearing a
   trophy. Adoption requires parity on every priority lane plus the margin on the targeted one, and
   every change ships env-gated, default off, with a one-line rollback in the launcher. A trade
   someone genuinely wants is a labeled shelf variant, never a silent standing change.
   *(Earned by: F-226/F-227/F-233 — adopted only with all-lane parity plus bar, `=0`/`=1` rollback
   kept; F-183 — the concurrency-for-M1 trade shelved, re-tested twice; F-230 — a +3.1% prefill win
   rejected for its −2.8% M1 tax.)*

3. **MECHANISM OVER ACCOUNTING.** An unexplained millisecond is not a budget line to claim, it is a
   question to answer. Every candidate estimate must survive a read of what the segment physically
   spends — bytes moved, launches made, waits served — before it earns a battery. Accounting finds
   segments; only mechanism finds levers.
   *(Earned by: F-232 — all three T13B "wins" died on code reads: emit batching met pipe-buffered
   stdout and draft waits, the sync diet met an already-syncing copy and PCIe bytes; F-222 — the
   "4.90ms doorbell tail" was the service thread's own pool compute, additive with the GPU wait.)*

4. **THE CHEAPEST FALSIFIER FIRST.** Before any expensive test, run the cheapest one that could kill
   the idea: one code read, one counter that was never armed, one env boot. Measure the premise
   before arming the lever — a false premise is discovered free, a true one costs nothing extra.
   *(Earned by: F-228 — the depth stair measured before any depth arm; STRATA_SPEC_DEPTH had never
   once been on, and the data vindicated the acceptance premise and killed the gate-tune in one
   round; F-231's single-boot falsifier bought the window anatomy for one boot.)*

5. **READ WHAT EXISTS.** The biggest wins were wiring, not writing: the fast attention kernel, the
   chunked recurrence, the combine fold sat in the tree, gated and never armed; the cohort win was
   assembled from doorbell and batch machinery the tree already owned. Before designing anything,
   inventory the machinery — the tree, the upstream PRs, the unmerged branches. The scan costs
   minutes; the design costs weeks.
   *(Earned by: F-221 — PR #540's kernel one flag away, +6.0/+7.0/+5.3 all-lane; F-224 — the in-tree
   chunked recurrence, prefill recurrence 1.68x; F-226 — the HIP-gated fold, +4.35% M1 on its
   first-ever arm; F-233 — two cohorts on the existing service-loop bones.)*

6. **STRUCTURE BEATS SQUEEZING.** When a stage is additive, look for the pipeline shape that overlaps
   it, not for microseconds inside it. The concurrent-throughput goal fell to a second window in
   flight, not to shaving the one that existed; every intra-window idea had already been priced at
   single digits. Change the shape before tuning the parts.
   *(Earned by: F-223's theorem to F-233's escape — no intra-window reordering can hide the pool;
   STRATA_COHORTS=2 put another window in flight and conc went 245.9 → 306.8, +24.8%.)*

7. **STATE THE CONSTRAINT PRECISELY, THEN FIND ITS ONE ESCAPE.** A precise statement of what cannot
   be done is a design tool, not a surrender: "only another window in flight can hide the pool" is an
   instruction. Write the constraint as an equation and read off the single variable that can still
   move. Every ceiling so far was an assumption wearing a measurement's clothes — depths "never
   offered" were being offered all along; the pool that read 4.8 ms was mostly removable waits.
   *(Earned by: F-223 → F-233; F-228 — the "depths 4-6 never offered" note retired by the first
   per-depth data; F-227 — the trailing repark waits were pure serialization, +4.06% M1 when
   skipped.)*

8. **NEGATIVES ARE ASSETS.** A clean falsification with a mechanism is a permanent map edge: bank it,
   cite it, and define the condition that re-opens it. A wrong call is retracted on the record with
   its cause — the retraction is part of the result. What is not recorded will be re-purchased at
   full price by the next agent.
   *(Earned by: F-229 — the ungated-draft loss banked with mechanism and its re-open clause (a deeper
   mean drafter shrinks the waste); F-225 — the planner call reversed on the record, default flipped
   back; F-216 — a perf-negative instrument that proved the commit replay is not the depth gate.)*

9. **TRANSITIONS ARE WHERE IT BLEEDS.** The bugs live at boundaries: empty cohorts, pre-launched
   commits colliding with parity commits, graph captures while the other cohort is in flight,
   warm-vs-cold boots, wrapper processes matching their own kill patterns. Fence every transition
   explicitly, and the first time a footgun bites, write it down as law so it never bites twice.
   *(Earned by: F-233's three boot-gated bugs and the one-time device fence at the standing→cohort
   transition; F-231/F-216 — deploy and battery are never one command, and a deploy is verified by
   its banner before its probe is read.)*

10. **THE CANARY IS THE CONTRACT.** Keep one sensitive invariant under watch on every boot — here,
    acceptance (M3b) under exact-match verification — and treat any move in it as an output change
    whatever the mechanism claims. Distribution-adjacent changes pay a tax that speed metrics will
    not show until it is too late; the tax scales with bits distance.
    *(Earned by: F-230 — the prefill acceptance-tax law, ~1.8e-4 KL of KV change = −2.7% acceptance =
    −2.8% M1; F-205/F-214 — M3b exposed the bare-engine wrapper bug and the dropped env recipe in one
    round each.)*

11. **INSTRUMENT BEFORE YOU OPTIMIZE, AND INSTRUMENT THE INSTRUMENT.** Diff-harnesses prove
    equivalence, counters arm before questions are asked, per-depth data banks before levers pull.
    And validate the measuring device itself: a wrapper that silently boots the wrong engine voids
    every verdict it has produced. A plan that cannot be checked cheaply is not a plan.
    *(Earned by: F-224's STRATA_PLAN_DIFF boot — zero mismatches; F-205 — the wrapper bug: every
    wrapped arm since F-200 had booted the bare engine, F-203's rejections voided, the no-op-wrapper
    validation law written.)*

12. **EXTERNAL INTEL IS RE-PRICING, NOT RECIPES.** Audits of papers, forks, and production blogs
    de-risk and re-price the queue; they do not execute it. Verify overlap before porting — never
    port what the tree already has — and weigh a production precedent on your exact component class
    above a paper number on someone else's.
    *(Earned by: T14A/B/C — FastMTP turned from rumor into a costed GO with sample counts and a
    calibrated c_row = 3.56 ms/row; F-213 — tqllm measured 3.8x behind on M1, the transferable value
    was their discipline; F-214 — the IndexShare adaptation voided because the drafter already
    skipped the indexer; F-219 — upstream #2758's silent-corruption fix audited into the tree before
    it could fire.)*

13. **CHANGE IS CLOSURE, NOT DIFF.** Every file lives in a dependency closure — configs reference
    files, launchers reference configs, and a server that survives a change proves nothing about the
    next boot. Delete by closure, never by directory shape, and verify the restored state before
    declaring a round closed.
    *(Earned by: F-212 — refusal-r.gguf purged by shape, the standing config demanded it, and the
    break waited for the next reboot; the campaign law's revert-and-verify rule.)*

14. **THE RECORD IS THE ENGINE OF COMPOUNDING.** Journal every outcome with its mechanism, commit it,
    push it — un-pushed work does not exist, and the power cut proved the stakes. The corpus is what
    makes the next agent smarter than the last ceiling: every entry is written for the agent who
    arrives with no memory of this one.
    *(Earned by: F-206 — the code was committed before the re-run and the power cut took only /tmp;
    F-209 — 49 commits pushed with every journal-cited SHA preserved; F-197 — the three-generation
    ladder answered "are we getting nowhere" with same-hour data only the record made cheap.)*

### Operating tempo

One loop, one variable per leg, run over and over:

**recon** — read the code and the record: tree inventory, banked falsifications, current cost map.
→ **hypothesis with mechanism** — state what the segment physically spends and what changes it.
→ **cheapest falsifier** — kill the idea for free if it is wrong. → **instrument** — diff-harness,
counters armed, canary watched, wrapper no-op-validated. → **interleave** — ABAB or 3v3 in one
window, means only. → **verdict** — adopt with all-lane parity, env gate, one-line rollback; or
falsify with mechanism and bank the boundary. → **journal, commit, push.** The round closes when the
record carries it, not when the server is back up.

### On ceilings

Every number in this repo that reads "limit" is a priced constraint awaiting its escape. The 267.5
that was boot luck, the pool that was waits, the depths that were "never offered", the fastest kernel
in the tree behind one flag — each was a ceiling until someone stated it precisely and pulled the one
variable the statement left movable. Do not worship a ceiling and do not declare one: price it. That
is the whole doctrine.

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
