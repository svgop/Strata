# SCOUT DIGEST — Row: "Standing arms held" (2026-10-25)

Topic: how real projects run "adopted perf arm + confirmation/regression discipline" on constrained hardware; transfer to a single-box campaign with 6-15% boot variance.

## Per-implementation findings

**1. llama.cpp / llama-bench — B-.** 5 reps default + warmup + mean±stddev, raw samples in JSON/SQLite; but NO automated PR perf gate (community reports + maintainer judgment; Mozilla found build flags alone cause up to 63% variance). Transfer: 5-rep+warmup+raw-archival convention; pin builds. Not: community-report net. [README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)

**2. vLLM perf-eval nightly — A-.** Separate repo (vllm-project/perf-eval): every benchmark a YAML recipe per (model, hardware); every run tagged VLLM_COMMIT+IMAGE; raw runs + median persisted; dashboard with RC-vs-release paired Compare bisecting regressions to tens of commits. No numeric thresholds yet (alerting = roadmap). Transfer: recipe-as-code, commit-tagged results, paired RC compare. [CI blog](https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality) | [perf-eval](https://github.com/vllm-project/perf-eval)

**3. PyTorch benchmark / TorchBench — B+.** blocked_autorange (time-budgeted reps); Compare warns inside noise; TorchBench canonicalizes the box (machine_config asserts + logs into every result JSON). Transfer: assert-and-log machine state every form; median over bounded-duration runs. Not: Callgrind instruction counts (our arms are GPU/memory-bound e2e). [Tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)

**4. Bencher / CodSpeed — A-.** Seven threshold tests per metric: with n~5 use t-test prediction intervals (~25 samples) or IQR/delta-IQR — NOT z-score (needs >=30, our variance violates it); direction-aware lower-boundary gates for throughput; flaky metrics quarantined (_bencher_ignore: recorded, never gating); feature branches clone base thresholds. CodSpeed: instrumented determinism <1% variance (impossible for e2e serving — context only). [Thresholds](https://bencher.dev/docs/explanation/thresholds/)

**5. ik_llama.cpp / exllamav3 / TabbyAPI — B.** Best single-box discipline: every perf PR lands with a fixed PP/TG table at fixed depths; regressions get tracked issues; rolling performance-over-time curves. Manual/curated, one expert's box. Transfer: the fixed benchmark matrix per candidate + table-required-to-merge. [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

**6. MLPerf submission harness — A-.** Hard minimums sized from target margin of error; THE discipline = the audit: independent re-run must land within ~5% of the submitted number or fail; "submission != audit" split. Transfer: confirmation != discovery — an adopted arm closes only when the independent pooled form reproduces the A/B claim within a fixed band (5% is battle-tested). [Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)

## TRANSFER LIST (ranked)
1. **Interleaved paired boots (A-B-A-B) across the 5 boots of a pooled form; gate on per-boot deltas** (one-sample t / trimmed mean of deltas) — cancels 6-15% boot drift structurally; zero extra wall-clock. The single highest-leverage fix.
2. **Gate on t-test prediction intervals sized to n=5 (not z-scores, not fixed %)**; lower-boundary for throughput.
3. **MLPerf-style confirmation audit**: pooled re-run must reproduce the A/B claim within ±5%; miss → one re-run, then escalate/revert.
4. **Persist raw per-rep samples** (re-analyze/reject/recompute without burning another form).
5. **Assert+log machine state in every result** (clocks/governor/build hash; hard-fail the boot if unset).
6. **Tag every run commit+config; plot pooled means over time** (second, cheap regression net).
7. **Calibrate the noise floor first** (one A-vs-A form, 5 boots) — gate width from measured paired-delta distribution, not assumptions.
8. **Quarantine flaky metrics** from gating while still recording.

## OUR NEXT STEP
Rework the five-boot pooled form to interleaved paired boots; gate = mean(delta) > 2·sd(delta)/√5 AND ±5% adherence band vs the original A/B claim. Same boots, same hour — variance becomes the measurement.
