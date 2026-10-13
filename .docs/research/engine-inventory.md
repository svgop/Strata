# engine-inventory — what MAGMA v2 already has, for the draft-yield / acceptance-rate decision

Scope: the machinery this checkout already ships that bears on the question "should the speculative chain
draft deeper, and what is the objective?". Every claim carries a `file:line` handle. Written 2026-10-12
during goal round 34, for the re-authored survey.

## 1. The probability floor — the knob under decision

`--spec-min-p P` (config key; also settable **per request** as `spec_min_p`). The chain stops extending a
verify window when the draft's probability falls below `P`.

- Gate implemented at `src/program/generate.cpp:11522-11524`:
  `while (T < S_mtp && from + T - 1 < avail && pr[from + T - 1] >= (float) req_spec_min_p) ++T;`
- Per-request tuning keys parsed at `src/program/generate.cpp:10264-10279` (`pcie_frac`, `spec_min_p`,
  `aux_cpus`), forwarded by `serve/server.py:1005` from `sampling.strata_tune`.
- Standing value 0.7 (`strata-iq3_s.json`). Measured tonight: 0.5 and 0.35 raise throughput 8-11% and lower
  the acceptance *rate* ~4.5 points because the chain tests ~1010-1100 drafts instead of ~850.

## 2. DraftPolicy — the engine's own E/cost optimiser, ALREADY LIVE AND UNUSED FOR THIS

`include/strata/spec/draft_policy.hpp` (64 lines) + `src/spec/draft_policy.cpp`, constructed once per process
at `src/program/generate.cpp:9237` (`strata::spec::DraftPolicy policy(S)`) and fed every round:
`generate.cpp:11924`, `:12292` (`policy.observe`), `:12295`, `:13443-13444` (`policy.observe_chain`).

Its header states the objective in its own words:

- line 7: "the policy learns both sides online and **compares expected committed tokens per millisecond**".
- line 9-11: MTP side `E` = the mean tokens a window of that size has committed (EMA) at the measured cost of
  that size; lookup side `E(k) = 1 + q + q^2 + ... + q^k`, q = acceptance of lookup drafts whose match was
  about as long (4 buckets of match length, decayed counts), at the measured cost of k+1.
- line 13-14: costs are measured round times per window size (EMA; unseen sizes scaled by a prior shape), so
  it "adapts to the machine and the context length".
- line 15: "**It only chooses which drafts to verify: the output is unchanged.**"

State it maintains (`draft_policy.hpp:56-61`): `cost_[kMaxT+1]`, `mtp_tok_[kMaxT+1]`, `ok_/bad_` (lookup drafts
accepted / windows cut short, decayed, 4 match buckets), `cok_/cbad_` (the same for chained lookup drafts),
`rounds_`, `last_[]` (staleness). `kMaxT = 8`.

**What it decides today:** `Pick choose(int t_mtp, int lookup_k, int match)` — MTP window or a lookup window —
and `int chain(t_mtp, p_mtp, k_avail, match)` for `--lookup-chain`. **It does NOT decide the MTP chain length**:
`t_mtp` arrives from the `--spec-min-p` gate above. So the engine owns an E/cost table for MTP windows
(`mtp_tok_`, `cost_`) that is currently only used to *compare against lookup*, never to *choose the depth*.

Its doc comment also records the field practice it was built against: "llama.cpp's lookup decoding answers the
same problem with confidence thresholds on its n-gram statistics; here the policy learns both sides online".

## 3. Lookup / suffix drafting — implemented, one arm measured, currently OFF

- `--lookup-chain K` (opt-in, default **0 = off**, clamped 0-7) and `--lookup-chain-min M` (default 3):
  `generate.cpp:741-745`, `:2125-2126`, help at `:894-896`.
- `--suffix-draft N` default **3** (`generate.cpp:738`), so the suffix drafter runs at depth 3 by default;
  instance built at `:9236`.
- Arming the chain raises the verify cap: `o.spec = max(o.spec, min(o.mtp_max_t + o.lookup_chain, 8))`
  (`generate.cpp:2570`).
- **Measured once, rejected:** F-130 (`SVG-CUSTOM.md:3679-3697`) ran `--lookup-chain 4 --suffix-draft 8` and
  lost on all four classes (158.9→123.3, 184.1→125.4, 140.8→120.9, 152.7→133.5 t/s); T *fell* on three of
  four classes, so the arm did not lengthen the window at all. Recorded caveat: that arm's numbers were
  compared **across boots** against the 17:29 sweep, which is the comparison error this campaign keeps
  hitting.

## 4. The other acceptance machinery already present

- **Typical acceptance rescue**: `STRATA_TYPICAL_ACCEPT` (default threshold read at `generate.cpp:268`, launcher
  sets 0.2), device arm `STRATA_TYPICAL_DEVICE` (`:285-288`). Accepts a draft within a probability margin of
  the target's top choice instead of demanding an exact match.
- **Coupled draft sampling** `STRATA_SPEC_COUPLED=1`: drafts use the target's chain and the same Philox draw;
  acceptance ~ sum min(p,q). On in the standing recipe.
- **Probabilistic draft acceptance** `STRATA_SPEC_PROB=1` (`src/core/mtp.cpp:580`, `src/core/verify.cpp:2313`,
  `include/strata/core/spec_prob.hpp`): drafts judged by rejection sampling. **Not in the standing recipe.**
- **Gumbel-max coupling** `STRATA_SPEC_GUMBEL=1` (`src/kernels/cuda/sampler.cu:1112`), ported from
  llama.cpp-lab's `--spec-coupled`. **Not in the standing recipe.**
- **Ratio test** `STRATA_SPEC_RATIO` (batch path, `generate.cpp:9612-9622`).
- **`STRATA_SPEC_DEPTH=1`** prints per-depth accepted/tested for the first five depths
  (`generate.cpp:12659-12663`) — the instrument that would show *where* a chain dies, never yet read.
- Depth counters for top-2 rejections: `MtpDrafter::top2_env()` (`generate.cpp:12664`).

## 5. Self-tuning that exists vs self-tuning that is dormant

- `tools/calibrate.py` sweeps, **on a live engine without restarts**: `PCIE_FRACS = (0.0, 0.2, 0.35, 0.55,
  0.75, 0.9, 1.0)`, `SPEC_MIN_PS = (0.3, 0.5, 0.7)`, and pool-worker counts (restarts), reporting the best.
  Its min-p grid stops at 0.3 and was evidently never run against this lane's adopted config.
- `PcieAuto` (the engine's own refinement of the miss-path split) is **dormant twice over**:
  `generate.cpp:2829` gates the whole block on `o.pcie_frac < 0.0` and the adopted config passes
  `--pcie-frac 0.35` explicitly, so `probe_pcie_h2d_gbps` never runs; and `generate.cpp:8353` requires
  `o.batch == 0` while this lane runs the batch path (`parallel: 4`).

## 6. The measurement instruments this decision can use

- `campaign/pcie-split-sweep.ps1` — sweeps any per-request tuning key in ONE boot, ONE window, at
  `temperature 0` so every arm routes identically; carries M1, M2 (pair), M3 (acceptance) and the full
  decode-timing decomposition per arm. Built tonight.
- `campaign/probe-lane.ps1` — the six-metric probe (M1/M2/M3 + per-window trend).
- `campaign/gpu-state-during-decode.ps1`, `campaign/gpu-priority-ab.ps1`, `launch-v2-diag-vprof-8081.bat` —
  card state, scheduling-band A/B, and the 33-phase per-layer GPU profiler.
- Engine-side, per request: `strata decode timing:` (T, tokens/window, ms/window, GPU-reach wait, plan/actq/
  jobs/CPU, draft, misses/hits/PCIe per layer-window) and `drafts accepted N of M`.
