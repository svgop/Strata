# T25-6 — FORMULATIONS & METHODOLOGY: the family's harvest for custom-2

**Date:** 2026-10-20/21 study (pins: basalt `fb0944c7`, Xolotl `cd965e3c`, maya `35bb2b2` = v1.1.20,
Strata upstream `9de04cd4` = v0.2.1.4, DeepInfer `9bd9cf2`, DFlash2 `a22a174`; library per
`/home/svg/LLM/data/references/MANIFEST.md`, clones verified 2026-10-19..21).
**Read-only study.** No builds, no boots, no GPU. Our reference state: the merged v0.2.1 line
(F-248), MMLU-Pro anchor 95.0 (95.7+94.3 re-gate, F-249), mimic 87.9, M1 212.5 / pair 202.8 /
6-conc 315.7, journal F-172..F-250.

**Question:** which mathematical formulations, quality-gate methods, and engineering practices in
the five reference repos are worth adopting — and what would each change in our law or instruments?

---

## 0. Ranked adoption table

| # | Item | Source (verbatim anchor) | Beats our equivalent? | Adoption cost | Changes in our law/instruments |
|---|---|---|---|---|---|
| 1 | **Teacher-forced KL yardstick on a frozen token set** (per-position top-1 agreement + KL(REF\|\|RUN) over REF's top-64, through the decode path) | Xolotl `tools/quant_quality.py` (whole file); sets: `data/quant_quality/v1/set.ids.json`; engine hooks `STRATA_LOGPOS` / `STRATA_LOGPOS_TOPK` / `--short-read` — **all three already in our tree** (`src/core/verify.cpp:3246`, `include/strata/core/verify.hpp:330`, `src/program/generate.cpp:834/999/2696`) | **Yes, decisively.** Our only numeric quality gate is MMLU-Pro 70q (binomial 1σ ±2.8 pts at 95% — `mmlu-pro-gate.py:60`); it cannot resolve a 1-question regression, and each gate is a 70-request lottery. The KL yardstick is deterministic (no sampling), per-position (5,000 rows, not 70), and separates code/docs/multilingual/agent domains | **Low.** Engine work: zero. Python: port ~400 lines + build our own committed ids set (IQ3_S tokenizer). One engine boot per quant/build | New instrument `campaign/kl-gate.py` beside mmlu-pro-gate; law 3's "MMLU gate" gains a deterministic pre-gate so MMLU is spent only when KL says the distribution moved. Would have resolved F-180/F-248's "1-question delta" ambiguity for free |
| 2 | **The spec controller: EMA cost + acceptance model, expected-tokens argmax, min-gain gate** | Xolotl `src/spec/{controller.cpp,draft_policy.cpp}` == maya `src/spec/{controller.cpp,draft_policy.cpp}` (Xolotl's adds `kStaleRounds` re-probe + `chain()`); formulation verbatim in §3.1 below; unit-tested no-GPU (`controller_test.cpp` prints an acceptance→(source,k,tok/s) table) | **Yes as an instrument, GO-slow as a default.** Our F-228 proved the cost model truth the hard way ("the marginal row costs not ~0.55 ms of expert bytes but the FULL graph's T-scaling") — the controller encodes exactly that as a *measured* per-window-size cost curve `dense_ratio[n]`/`kShape[t]` instead of a per-row estimate. Our spec depth/floor is static config; theirs adapts per request with priors + online learning | **Medium.** Self-contained `src/spec/` (~700 lines + tests), no kernel work; but our verify path now has cohorts (F-233) and batch-MTP (F-231) — the cost model must be re-measured on OUR curve first (the EMA does this by construction). Gate it as an env arm per law 2 | Would turn the F-228 ledger ("depth pays via better drafter or cheaper rows") into a live per-request arbiter that banks the measured cost curve as it runs; feeds M3b-style per-depth canary automatically. Law 10's canary unchanged; law 4's falsifier = `controller_test` on measured constants |
| 3 | **Upstream standard benchmark protocol: frozen hashed fixtures + cache-reuse policing + engine-accounted decode rates + medians-with-exclusions** | Strata v0.2.1.4 `docs/BENCHMARK_PROTOCOL.md` (whole doc; protocol id `strata-qwen38-thinking-v1-draft.1`, corpus sha `e75eff8b…`), fixtures `bench/prompts/standard-v1/` (`build.json`, `MANIFEST.sha256`, per-request `max_shared_prefix_tokens`) | **Yes for comparability, no as a replacement.** Our probe-lane (4 text prompts, regex on engine log, means of 3) is a fine *internal* interleave instrument but its numbers are not comparable across the family: basalt publishes decode-only rates, Xolotl server-task rates, upstream engine-accounted rates with draft rejections excluded. Upstream's protocol is the family's common denominator | **Medium-low.** Our tree carries the fixtures + `strata bench run` already (v0.2.1 merged). Cost = one 20–30 min conforming run + recording discipline, not code | Adds a family-comparable row to every re-bank (F-248 class) without touching the probe-lane interleave law. Directly serves T25's "make OUR numbers comparable across the family" |
| 4 | **Upstream-sync discipline: merge-tree dry run, ordered parity gates, fork-isolation rules** | Xolotl `docs/UPSTREAM_SYNC.md` (rules 1–4 verbatim in §5.1); `MERGE_SCOPE.md` §3–§6 (graft by tree-hash equality, criss-cross hazard, `git merge-tree --write-tree` conflict census, resolve-per-kernel-function gated by parity tests) | **Yes — this is the missing half of our merge law.** Our T19/T20 merges validated by MMLU gate + re-bank (F-245/F-247/F-248) but had no *bit-level* gate and no conflict-surface management. Xolotl's order: CPU parity → mini-fixture bit-identity → real-model ppl band → per-model gates, all **before** committing the merge; plus the isolation rules that made their next merge 13 files instead of 51 | **Low process / medium practice.** The gates reuse item 1's yardstick + our bit-check.py. The isolation rules (FORK-list pattern, no-op-default params, footprint numstat) apply to our next kernel arm | Amends the campaign law's merge procedure (law 1/7 class): before the next upstream merge (v0.2.2+), run `git merge-tree` census, pin our in-tree edits (norm_fast, FUSE_HITS, POOL_DIET, COHORTS, T18, mmvq_il, GDN_CHUNKED) as a numbered footprint, and gate the merge on bit-parity before spending probe windows |
| 5 | **The acceptance-tax exchange rate: TV/token = 1−p(d) per kept draft; ≈ +1.5 % tokens per 0.01 TV/token; typical-acceptance lands on the same line** | basalt `docs/precision/features.md` "Relaxed drafting" (offline-replay method + tables); index.md summary | **Yes — it prices what our law 10 only bans.** Our F-230 measured an *involuntary* tax (1.8e-4 KL of KV change = −2.7 % acceptance = −2.8 % M1) and F-229 rejected ungated drafting; basalt gives the *voluntary* trade's exact price and shows Medusa-style typical acceptance is not a better trade than a plain threshold | **Low as knowledge; zero code** (we adopt no relaxed acceptance — the value is the pricing method: log real verify windows, replay the rule offline, report gain AND TV). ~100 lines of replay Python over M3d/LOGPOS data | Law 10 gains a quantitative clause: any distribution-adjacent arm must state its price in TV/token or KL, with basalt's ~1.5 %/0.01 TV as the market rate. Kills future "free acceptance" hypotheses at the falsifier stage |
| 6 | **Parity-test culture: every kernel lands with a double-precision reference + printed rel-RMS, and bitwise-independence claims** | basalt `docs/precision/index.md` activation tables (`dense_tc_parity` ~1e-7 RMS, `mid_chain_parity`/`dec_commit_parity` bitwise, NVFP4 expert kernels 5–8e-8 rel); maya `docs/QUANT-PLAN.md` "Engine work" (every type at ~1.5 % vs double = the q8_1 share); upstream `bench/micro/fusions_parity.cu`, `tests/cuda/prefill_mmq_kquant_test.cpp` | **Partially — we already do the proof, not the program.** Our F-216 proved bit-parity for the slot-promotion kernel; the family pattern is a *committed, re-runnable* parity binary per kernel with printed tolerances, so the proof survives the next merge | **Medium** (one parity program per touched kernel, synthetic data at model geometry; no GPU math beyond the kernel itself) | Law 11's "instrument the instrument" gains a standing rule: a kernel arm without a parity program in the tree is unadoptable, so merges re-prove old arms cheaply |
| 7 | **Compile-only CI pinned to the toolkit (CUDA 13.0.2, `CMAKE_CUDA_ARCHITECTURES=120`) with cpu-labelled tests** | basalt `.github/workflows/ci.yml` (`CUDA_VERSION: "13.0.2"`, `-DCMAKE_CUDA_ARCHITECTURES=120`, "No GPU is available: … only the `cpu`-labelled tests run"); note upstream's FP4 path is `sm_120a`-gated (`docs/NVFP4.md:140`), community boxes recorded as "CUDA 13.1" | **Yes.** Our tree has no engine CI at all (only inherited terminal workflows); basalt's repo "moves daily" and stays buildable by pinning the toolkit and compiling for sm_120 GPU-less. Our v0.2.1-merged line + next upstream merges would catch kernel/cmake bitrot for free | **Low** (one workflow, ~60 lines; fork of basalt's). Caveat: anything `120a`-gated compiles but cannot run in CI — mark those tests gpu-labelled | Supports law 14 (the record is the engine): a red CI is a journal-able negative that costs no boot. Pair with the Launch Law (builds deploy by `cp` — CI guards the tree, the launcher guards the deploy) |
| 8 | **Forced-continuation speed A/B (`MAYA_GLM_SPEC_FOLLOW`)** | maya `docs/SETTINGS.md:92` verbatim in §2.3 | **Yes, with a sharp caveat.** It removes the text-feedback confound (a near-tie flip changes the text → changes acceptance → changes speed) from speed A/Bs — the exact class of confusion in F-225 ("the session warm trend read as a win") and F-231 ("one lucky boot"). Caveat (theirs, verbatim): a text from another model or other settings *favours the drafts that made it* — same-model text only | **Low** (engine already decodes a tokens-file continuation; needs a follow-file mode + a `glm follow:`-class mismatch counter) | Would give the interleave law a third arm: same-window AND same-text. Most valuable for acceptance-affecting arms (draft temp, floors) where speed deltas ride on diverging outputs |
| 9 | **Bit-identity proof harness for spec decode: 275-prompt greedy sweep + FNV-1a full-vocab logit hash + forced-window KLD with identical rows.tsv md5** | basalt `docs/precision/features.md` "Speculative decoding does not change output" (62,271/62,271 rows bit-identical, spec 0 vs 7); `tools/greedy_identity.py` | **Method yes, tool no.** Our F-245 *invalidated* cross-boot greedy hashing on this engine — and basalt/Xolotl explain why: identity requires pinned expert tiers (`--adapt-swaps 0`; Xolotl: IQ models not byte-reproducible at T=0 unless `STRATA_IQ_MT_MIN=1`, CPU multi-token vs single-token kernel choice depends on drafts-in-window). Our adaptive cache + IQ3_S CPU tiers void naive greedy hashes | **Low-medium**: keep bit-check.py but add tier/env pinning + a logit-hash readout; the forced-window KLD identity (rows.tsv byte-identical across T=1/3/7) is the stronger, placement-free form (item 1) | Retires the greedy-hash instrument honestly (F-245 note already does); replaces it with the two valid forms: pinned-placement greedy identity (basalt) and forced-window KLD identity (item 1) |
| 10 | **Draft-expert precision chosen by acceptance, not by quality** | maya `docs/SETTINGS.md:84` (`MAYA_GLM_MTP_KEEP`: draft computes only its route's n best-scored missing experts — "a draft only has to be a good guess, but a rejected one costs a redo"; 82–83 % accepted vs 76–81 %, never slower) + `MODEL_CARD.md` (NextN experts Q2_K/Q3_K); Xolotl `docs/ds4/V41_RESUME_PROMPT.md:189` (lower-precision DSpark draft experts); basalt `docs/precision/index.md` weights table (drafter experts Q2_0, draft head 2-bit screen + exact rescoring of near-best rows, pick matches exact argmax 99.93 %) | **Yes — a lever we have not priced.** Our drafter runs at the pack's precision; the family uniformly runs the drafter cheaper *because verify, not the draft, decides the tokens*. Basalt's two-stage head is the refined version: approximate screen, exact rescoring within a bound | **Medium-high** on our engine (drafter weights are bound per pack; would need a requant + loader change). The cheap first falsifier: measure our drafter's own KL/acceptance at lower precision offline (T14B evaluator class, item 12) | A shelf-variant arm: "draft-side requant" — law 2 applies (all-lane parity + bar on M1; M3b canary must hold since drafts change which windows run, though not what verify accepts) |
| 11 | **Release gates for quants/behavioral changes: loop test + context-fill test + per-domain KL breakdown** | maya `tools/maya_quant/loop_test.py` (repeat-64gram share, unfinished runs, unclosed think blocks; T=1.0 and greedy), `ctx_fill_test.py`, `bench/results/MAYA-S.md` (KL per domain; wikitext flagged "largely memorized", KL 0.934 outlier) | **Yes for mimic-adjacent coverage.** Our mimic 87.9 anchor measures instruction-following; the loop test measures *degenerate sampling dynamics* (loops, never-ending runs) that neither MMLU nor KL on held-out text reliably catch. Per-domain KL breakdown catches "quant fine on chat, broken on tool calls" (maya tool-call KL 0.850 vs chat 0.163) | **Low** (one Python script; runs on the standing server) | A cheap second gate beside MMLU for anything token-stream-affecting: 5 long generations + 3 needle contexts, ~10 min. Would have given F-248's "one low MMLU run" a second independent signal the same night |
| 12 | **Offline acceptance evaluator with teacher states + confidence-calibration quartiles** | Xolotl `docs/ds4/V41_DSPARK_AUDIT.md` (method: drive the drafter exactly as production does against an FP8 teacher; the flat-confidence-quartile signature found the weight-loading bug: conf Q1..Q4 13/12/8/11 broken → 12/40/78/99 fixed; mean accepted 0.437 → 2.583) | **Yes — this is the T14B evaluator's missing spec.** Our banked stock-MTP anchor is 2.78 tokens/window (DFlash2 card: stock 2.50–3.08 on their mix); the audit defines how to build an evaluator that can catch a *broken* drafter (flat confidence quartiles) vs a merely weak one | **Medium** (offline Python harness + teacher logits; no engine boot) | T14B's GO already priced c_row=3.56 ms/row; this adds the evaluator design so retrain arms are judged on teacher-forced acceptance + calibration before any engine boot |
| 13 | **Batch-level confidence-budgeted varlen verification** | DeepInfer `deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py` (`compute_seqlens_q_by_confidence_rates`: cumprod confidence → stable argsort → top-(total_q−num_seqs) wins; ref impl in-file, tilelang kernel beside); README: `DEEPINFER_ENABLE_MTP_VARLEN=1`, +4–7 % throughput, lossless, drafter-agnostic (arXiv:2607.05147) | **Yes for the conc lane — our exact shape.** Our 6-conc lane runs fixed spec-6 windows per slot (F-233 cohorts batch them); DeepInfer allocates a *shared* verify-token budget across the batch by cumulative confidence instead of a fixed depth per sequence. At our 6×6 = 36-slot scale the waste from uniform depth is exactly what F-228's curve says it is | **High** (verify-window packing is engine core; cohorts complicate it further). Record as the Tier-2 lever it is; the falsifier is the item-2 cost model + per-slot confidence data we do not yet log | Tier-2 roadmap entry priced +4–7 % class on 6-conc; requires M3d-class per-position confidence logging first (a diagnostic boot per law 5) |
| 14 | **Window cost-shape constants + stale-cost re-probing** | Xolotl `src/spec/draft_policy.cpp:12` `kShape = {0.0,1.0,1.35,1.7,2.05,2.45,2.85,3.25,3.6}` (RTX 5070 measured, "+10 ms per token with every missed expert on the CPU… flatter with the default CPU/DMA split"), `kStaleRounds=300` re-probe, `kProbes=3` exploration | **Yes as a measurement plan.** We have no measured cost(T) curve for our verify windows — F-228's insight (cost scales with the full graph) deserves its own stair: one diagnostic boot, T=1..8, ms/window per T, on the standing recipe | **Near-zero** (one battery boot + the existing STRATA_VERIFY_PROFILE/DECODE_TIMING diagnostics) | Banks the constant `kShape_custom2[]`; feeds item 2's controller and re-prices every "deeper window" roadmap item with real numbers (law 4: cheapest falsifier first) |
| 15 | **Quant pipeline discipline: evidence table per decision, FP8-teacher fixtures, allocation proxy validated against realized error** | Xolotl `docs/ds4/V41_1017_QUANT_PLAN.md` (decision table with measured evidence: act-order ON 37/40 layers better, Hadamard OFF, adapters OFF; "the calibration and the quant must see the same weights the engine will run"; v1 realized/predicted per layer 0.90–1.20); maya `docs/QUANT-PLAN.md` (error-feedback rounding −25 % expert output error, KL ≤ 0.05 target, per-expert imatrix shrunk toward layer stats, fixed floors against loops: down never below gate/up) | **Yes for our re-quant lane.** Our IQ3_S pack is store-bought; F-248's ladder names "re-quant" as next. The family's pipeline turns a re-quant from an art into a gated measurement program with a per-decision evidence table | **High** (full pipeline; the V41 plan prices it in wall-clock). Adopt the *discipline* now, the pipeline when re-quant queues | The re-quant item in the ladder gets a spec: teacher fixtures, KL≤0.05 target on our own text mix (probe-lane transcripts!), floors (head/embeddings/router Q6_K+), loop test as release gate |

Skipped as not beating ours (checked, not adopted): basalt's `kShape`-only cost model (item 2's measured EMA supersedes it); maya's per-layer TRUNK_KEEP maps (we have no expert-trunk split on the 5090 lane); upstream's `strata_bench.py` quick check (explicitly non-comparable, "do not rename their output… to make it appear standard", BENCHMARK_PROTOCOL.md); DeepInfer's AIME/GPQA eval stack (reasoning-model eval, wrong instrument for a 70q gate); DFlash2 integration (already adjudicated DOESN'T-as-drop-in in T23); Xolotl's conversation-pool bench numbers (different model class, 552B on 4090+RAM — their *methods* transferred above, their numbers don't).

---

## 1. Precision / quality-gate methodology

### 1.1 Xolotl's quant quality yardstick (the top pick)

The harness (`tools/quant_quality.py`, docstring verbatim): "teacher-forced top-1 agreement and KL of a
quant against Q8_0, on one fixed text set (data/quant_quality/v1: code, docs, English, other languages, an
agent transcript)". Mechanics that matter:

- **Committed token ids**: `quant_quality.py ids` tokenizes the set once and commits
  `set.ids.json`, "so a box without the tokenizer scores the same positions".
- **Everything through the decode path**: `--short-read` forces the whole segment through verify
  windows; a guard fails the run if "the prompt took the batched path" (`if n1 - n0 < len(ids)//2`).
- **Fixed placement**: `FIXED = ["--expert-cache","1","--adapt-every","100000","--pcie-frac","0",
  "--suffix-draft","0","--prompt-cache","0","--kv","fp16"]` + `STRATA_ROUTE_TAIL_SKIP=0`,
  `STRATA_PREFILL_CPU_SHARE=0` — the same pinning discipline basalt documents (see 1.3).
- **The metric**: per position, top-1 agreement (same argmax as the Q8_0 reference run) and
  KL(REF‖RUN) over REF's 64 most likely tokens, both renormalized to that set, "a token RUN does
  not list gets RUN's 64th log-probability" (`compare()`, lines 284–320), reported per segment and
  overall with KL p99 and both PPLs.
- **Prompt-path twin**: `prompt`/`pcompare` dump full-vocabulary fp32 logits after 20 batched
  prompts and compare full-vocab KL + top-1 — the batched path gets its own gate.
- **Hygiene**: refuses to start an engine with less than `--min-free-gb` (default 16) available;
  asserts the same text at every position before comparing.

**Status of the hooks in our tree:** `STRATA_LOGPOS` ("after run(), write one line per row t of the
last window's head", `include/strata/core/verify.hpp:330`), `STRATA_LOGPOS_TOPK`
(`src/core/verify.cpp:3246`), and `--short-read` (`src/program/generate.cpp:999`) are all present in
the merged v0.2.1 line. The Xolotl harness should port nearly verbatim; the only new artifact is our
own committed ids set (tokenize probe-lane-class text with the IQ3_S pack tokenizer).

**Why it beats ours:** `mmlu-pro-gate.py` measures 70 samples with a printed 1-sigma of ±2.8 points —
its own comment says "a drop smaller than this cannot be resolved by one run". The yardstick
measures ~5,000 forced positions with zero sampling variance; a 0.01 KL move is visible. The two are
complements: KL says *whether* the distribution moved and where; MMLU says what it *costs on tasks*.

### 1.2 basalt's cross-engine KLD agreement and the reference-noise audit

`docs/precision/quality.md` establishes that an in-engine KLD gate can be *validated against*
`llama-perplexity`: "IQ3_S 0.15264 vs 0.1526, IQ3_XXS 0.2433 vs 0.2428 at 100 chunks" — agreement to
0.002 KLD and 0.2 points of same-top-1. Method verbatim: wikitext-2 in 512-token chunks, first half
prefilled, "rows 256 to 510 scored through the decode path", base = llama.cpp's
`--kl-divergence-base` Q8_0 file, "KLD accumulated in double from the fp32 logits"
(`docs/precision/index.md:108`), and `ppl_gate.py --diff A.tsv B.tsv` reports identical-row %, max
|dNLL|, max |dKLD| and the first differing chunk:pos.

The section "Why Q8_0 against Q8_0 is not zero" is the methodological jewel: basalt's own Q8_0 pack
scores KLD 0.0456 against the stock base — *the reference's own noise* (llama.cpp quantizes hc-read
activations; a numpy layer-0 reconstruction shows 1.06 % relative error vs fp64). With an F32-hc
patched base the same comparison reads 0.0233, "the level of two independent implementations".
Their ruling: "Reported KLD numbers therefore include about 0.01 to 0.02 of the reference's own
noise. They are kept against the stock base because anyone can regenerate it."

Lesson for us: any KL gate we adopt must (a) name its reference and its noise floor, (b) keep the
reference reproducible, and (c) never compare KL numbers across different references. Also their
scoring rule: "Quality is scored through decode" — prefill is a different computation (fp16/BF16
GEMMs, MMQ, FP4), so the gate teacher-forces through verify windows, never through the prompt path.

### 1.3 Forced-window KLD: the placement-free identity proof

`docs/precision/features.md` "Forced-window KLD": the gate is teacher-forced with window tokens =
reference text at forced widths T; IQ3_XXS at T=1/3/7 produces PPL/KLD/top-1 identical to the last
digit and **all 5,100 per-row lines byte-identical** (`ppl_gate.py --diff`: max |dKLD| 0.000000;
rows.tsv md5 equal across T). So "KLD with MTP on equals KLD with MTP off" and the gate is
insensitive to window width — exactly the invariance our acceptance canary (law 10) assumes but has
never been able to test directly.

Combined with the MTP bit-identity sweep (275 prompts, 62,271 full-vocab logit rows bit-identical
between `--spec 0` and `--spec 7`, with an FNV-1a logit-hash printed per accepted row), this is the
family's strongest statement of the *spec-does-not-change-output* invariant our M3b canary
approximates. Note their scope discipline: identity holds "with the expert tiers pinned… With
adaptation on, experts move between the CPU pool and the GPU, which round differently: a rounding
difference, not a quality difference."

### 1.4 maya's release gates: per-domain KL, loop test, context fill

`bench/results/MAYA-S.md` scores a quant with KL(FP8‖quant) over top-64-plus-one-bucket **per
domain** (chat 0.163–0.290, reasoning 0.061–0.422, code 0.256–0.450, tool calls 0.850, wikitext
0.934 with the annotation "largely memorized by FP8"). Two transferable rules: per-domain reporting
catches domain-specific damage a mean hides, and memorized benchmarks (wikitext) must be flagged —
they measure the reference's memorization, not your quant. Task battery: 400q/task zero-shot ARC /
HellaSwag / WinoGrande / PIQA scored lm-eval-harness-style with the FP8 teacher run in PyTorch and
the quant through the engine, reported as **recovery %** (97.9 % average). Release gates:
`loop_test.py` (long generations at T=1.0 and greedy; longest back-to-back 32-token-window repeat and
share of tokens inside repeated 64-grams — "a healthy answer: ~0"), and `ctx_fill_test.py` (needle
at 8K/16K/30K, asked at the end).

### 1.5 The acceptance-tax price (basalt relaxed drafting)

`docs/precision/features.md` "Relaxed drafting": keep a draft when p(d) ≥ P; "At a row where the
draft d has p(d) ≥ P, the output is d every time instead of a draw from p: that row is greedy. Its
total-variation distance from the model's distribution is exactly 1 − p(d), and 0 at every other
row." Offline replay over logged verify windows (~3,500 tokens/cell) measures **TV/token** and token
gain; the empirical law: "tokens gained per unit of TV is nearly the same at every threshold: about
+1.5 % per 0.01 TV/token." And: "Medusa-style typical acceptance… was replayed too and lands on the
same line; it is not a better trade, so the flag is the plain probability threshold." Their quality
gate honesty: "The teacher-forced KLD gate cannot see this change: it scores logits, and relaxed
drafting changes which token is emitted, not the logits" — so they add sampled-task pairs (gsm8k
300, MMLU-Pro 280, paired in one session).

For our law: F-229 (ungated draft −2.73 % M1) and F-230 (acceptance tax) get a market price. Any
future arm that trades distribution for speed must state its cost in TV/token, and any claimed
"lossless" acceptance relaxation must beat the plain threshold on the same replay.

---

## 2. Benchmark protocols

### 2.1 Upstream's standard protocol (comparability across the family)

`docs/BENCHMARK_PROTOCOL.md` (draft, implemented in `strata bench run` from 0.2.1 — our tree has it)
is the family's answer to "how do numbers compare across boxes". Key machinery, verbatim anchors:

- **Frozen fixtures**: one JSON per request with "its exact messages, tool schemas, seed, cap,
  expected output and frozen token count", `build.json` recording tokenizer/template sha256s,
  `MANIFEST.sha256`; "The runner checks every fixture against the bundle manifest before it sends
  anything, refuses to run if one byte differs". `.gitattributes` marks the bundle `-text`.
- **Cache policing**: each request stores `max_shared_prefix_tokens`; "A request's `cached_tokens`
  may be at most its `max_shared_prefix_tokens` … plus 8 tokens for merges at the boundary. Above
  that the user payload was read from the cache: the request is left out of the speed summaries, and
  a performance request makes the run `nonconforming`."
- **Honest accounting**: "Decode tokens/sec: Engine `decode_tokens / decode_s`. Include accepted
  reasoning and answer tokens; exclude rejected speculative drafts." Cached tokens never enter the
  prompt-rate numerator. "Retain API `output_tokens` separately from engine `decode_tokens`."
- **Statistics discipline**: "Compute medians, minima, maxima, and sample counts for each performance
  case separately… Three rounds are too few to advertise a stable p95." "Do not pool prompt sizes,
  quantizations, configurations, or concurrency levels."
- **Track separation**: "Use `configuration.track = "default"` for an unmodified release's
  setup-generated settings… `"custom"` for tuning, experimental options, or modified builds… Never
  merge these tracks into one headline result. A claimed improvement requires a separate,
  same-machine baseline."
- **No invisible retries**: status taxonomy `completed/truncated/failed/skipped`, `nonconforming`
  precedence, "A later run gets a new ID and preserves the earlier run".
- **Warm-up excluded**: "Include warm-up time in `required_duration_s`, but exclude warm-up requests
  from performance and quality summaries."

Our probe-lane already follows the interleave law (stronger than this protocol for *internal*
verdicts); what we lack is the *external* contract. Our tree ships the fixtures and the runner
(v0.2.1 merged), so a conforming run is configuration + 20–30 minutes, not code. One caution: the
protocol pins the stock model-card sampling (T=1.0, top_p 0.95, thinking on) — our lane runs T=0.7
custom arms, so a conforming run is a *separate* comparable number, not a replacement for probe-lane.

### 2.2 maya's spec-follow A/B (same-text speed comparisons)

`docs/SETTINGS.md:92`, verbatim: "`MAYA_GLM_SPEC_FOLLOW` | off | speed A/B (`--spec-follow PATH`):
a file of token ids… every request decodes that continuation instead of its own picks and ends with
it, and the drafting… keep exactly the drafts that match it: two builds or settings then process the
same text with the same windows, drafts and routes, where a near-tie flip would otherwise change the
text and with it the draft acceptance and the speed. The log's `glm follow:` line counts the tokens
where the model's own pick differed… **A text from another model, or from the same model with other
settings, favours the drafts that made it: compare a change on the same model and its own text**".

This isolates *mechanical* speed from *acceptance* speed. Our F-225/F-231 confusions (warm trend,
lucky boot) were window problems; the follow-mode problem is different and complementary — even a
perfect interleave cannot tell you whether an arm's speed delta came from changed text. Cost on our
side: our launcher/serve stack already accepts tokens-file prompts; the missing piece is the
follow-mode + mismatch counter in the engine.

### 2.3 Token-id fixtures everywhere

basalt `data/bench/{conv,struct,prose,prefill}.ids` (comma-separated token ids per line),
Xolotl's `set.ids.json`, upstream's hashed corpus: the family freezes *token ids*, not text, so
tokenizer drift across merges cannot silently change the workload. Our probe-lane prompts are
strings; a tokenizer change in a future merge would change our prompts' token counts and with them
every ms/window. Cheap hardening: freeze the ids of the four probe prompts once and post them
(`input_tokens_frozen`-style) so any drift is detected by the probe itself.

### 2.4 The community spec-sweep pattern (upstream bench/results)

`bench/results/2026-10-01-community-2x-rtx-pro-4500/spec-sweep/sweep.py`: per spec-N server restart,
3 runs per prompt, JSON rows `{spec, prompt, run, prompt_tokens, generated, tok_s, accepted,
drafted}` — the same shape as our F-228 depth stair, published as a results directory with its
harness. Ours is better instrumented (per-depth accepted/tested via STRATA_SPEC_DEPTH); theirs is
the publishing pattern to copy when banking the stair: harness + data + config in one directory.

---

## 3. Acceptance / cost formulations

### 3.1 The spec controller (maya → Xolotl; identical `controller.cpp`, Xolotl extends `draft_policy.cpp`)

The core arbitration, `src/spec/controller.cpp` (73 lines, unit-tested without GPU):

- **Cost model, decomposed** (`CostModel::step_ms`, verbatim):
  ```cpp
  const double dense = dense_ms * dense_ratio[n - 1];      // measured per-window-size curve
  const double u = distinct_ratio[n - 1];                  // distinct experts grow as U(n)
  const double uses_per_expert = (double) n / u;
  const double cpu = (1.0 - hit_rate) * cpu_all_miss_ms * u
                   * (1.0 + extra_use_cost * (uses_per_expert - 1.0));
  const double draft = k * (mtp ? mtp_draft_ms : lookup_draft_ms);
  return dense + cpu + sync_ms + draft;
  ```
  The coupon-collector term (distinct experts grow sublinearly in window size; each extra token on a
  missed expert costs only a fraction of a read) is the family's answer to per-row cost estimates.
- **Expected tokens** (`expected`): `E = 1 + Σ_{i=1..k} Π p`, with a `same` flag switching between
  iid per-position rate (lookup) and a per-position vector p_0..p_{k-1} (MTP).
- **Arbitration** (`choose`): consider Lookup k=1..available and MTP k=1..K_MAX, rate = E/cost,
  take argmax; then the min-gain gate: "if best.tokens_per_ms < baseline * (1.0 + min_gain_) → None".
- **Priors + online learning**: `mtp_p_ = 0.86` (from the flyweight's measurement on this model),
  `lookup_q_ = {0.35, 0.6, 0.8, 0.92}` by match-length bucket ("from the offline replay,
  2026-09-23-suffix-lookup"), both updated per position with one EMA: `p[i] += ema_*(hit − p[i])`.
- **The untested-position rule** (`observe`): when every drafted token was accepted, positions beyond
  the window "were never tested, so without this they keep their prior forever and the window can
  never grow. Pull them toward the deepest observed rate."
- **DraftPolicy (Xolotl's revision) adds**: EMA cost per window size (`kCostAlpha=0.1`) with a
  kShape prior for unmeasured sizes; EMA tokens per MTP window (`kTokAlpha=0.05`, prior
  `1 + 0.7*(t−1)`); a Beta-flavored lookup posterior `(ok + 4·q_prior)/(ok+bad+4)` with decay 0.97
  and priors `{0.75, 0.88, 0.93, 0.96}` ("the longer the match, the likelier its continuation —
  llama.cpp's lookup decoding gates on the same thing; worth 4 observations"); **exploration
  overrides**: a size is probed `kProbes=3` times before its guessed cost can veto it, and a
  `kStaleRounds=300` cost is re-measured "so one slow stretch cannot price it out for good (#1252)";
  and chained MTP+lookup: `r = (e0 + p_mtp·Σc^i)/cost(t_mtp+k)` with weaker chain priors
  `{0.45, 0.65, 0.8, 0.9}`.

`controller_test.cpp` demonstrates the behavior contract: persistent rejection turns speculation off;
a cheap wide verify takes the whole lookup window; a bare trigram match is not worth a window; full
acceptance is learned and the window widens; and it prints the acceptance→choice table at
p ∈ {0.5…0.95} under the default cost model.

**Relation to our record:** F-228 proved the cost truth (marginal row = full-graph T-scaling, so
"high marginal acceptance still nets negative" at our curve); F-229 banked the ungated negative with
its re-open clause ("a deeper mean drafter shrinks the waste"). The controller is the standing
mechanism that (a) measures OUR curve online instead of assuming one, (b) enforces a min-gain margin
per request — the per-request version of our adopt bar, and (c) makes the F-229 re-open clause
automatic: better drafter → learned p rises → windows widen only where the measured cost allows.
Adoption as a *default* would be a big distribution-adjacent change (law 2/10); adoption as an
*instrument* (a diagnostic boot that logs the controller's table without acting on it) is cheap and
bankable now.

### 3.2 Confidence-budgeted batch verification (DeepInfer)

`schedule_mtp_varlen.py::compute_seqlens_q_by_confidence_rates` — the algorithm, verbatim from the
reference implementation: per sequence, `cum_rates = cumprod(prev_confidence_rates)` (per-position
confidence from the previous MTP head; fresh seqs = 0); flatten all (seq,pos) rates, "Stable sort
descending by rate; for equal rates, original order (lower index first)"; each sequence gets
`1 + wins` where `wins = #(ranks < extra_budget)`, `extra_budget = total_q − num_seqs`. I.e., the
batch's verify-token budget is spent on the globally most-confident draft positions, not uniformly
per sequence. README: `DEEPINFER_ENABLE_MTP_VARLEN=1`, +4–7 % throughput, lossless, drafter-agnostic;
`DEEPINFER_USE_NEXTN_DECODE` caps draft length (default 5). Also of note: `DEEPINFER_BATCH_INVARIANT=1`
+ per-request `seq_seed` for deterministic sampling.

For us: the 6-conc lane (315.7 t/s) batches 6 slots × spec-6 into cohort reactors (F-233). Uniform
depth spends verify compute on low-confidence tails; the varlen rule reallocates it. This is a
Tier-2 engine change; the prerequisite instruments are per-position confidence logging (we log
accepted/tested by depth today, not offered-confidence) and item 3.1's cost curve.

### 3.3 Draft-side precision levers (acceptance-based format choice)

Three independent family precedents:
- maya `SETTINGS.md:84`: `MAYA_GLM_MTP_KEEP=2` — on a two-GPU split the draft block computes only
  its route's n best-scored missing experts: "a draft only has to be a good guess, but a rejected
  one costs the first GPU a redo. `2` drafts better than skipping all and was never slower
  (2x V100: 82–83 % accepted against 76–81 %)".
- maya `MODEL_CARD.md`: NextN draft experts Q2_K gate/up, Q3_K down while attention/shared/dense
  stay Q6_K — precision allocated by role, drafts cheapest.
- basalt `docs/precision/index.md` weights table: drafter experts Q2_0 all-VRAM; draft head rows
  gathered from the main head with a 2-bit screening copy — "on 1500 real draft-head inputs… the pick
  matches the exact argmax in 99.93 %", plus the scale correction `sqrt(‖w‖²/⟨w,ŵ⟩)` per group that
  halves the probability's mean relative error (0.028–0.043 → 0.010–0.020 on the parity head).
- Xolotl `V41_RESUME_PROMPT.md:189` queues "lower-precision DSpark draft experts (2.1 GiB of Q2_K on
  the card; K < 64…)" as an unmeasured candidate.

We have not priced draft-side requant at all. The verify-decides-tokens invariant (which our M3b
canary and basalt's bit-identity sweep both protect) is what makes this lever safe: drafts shape
*speed*, never *output* — unless a cheaper draft changes which windows run and thereby trips
placement-dependent rounding (the F-245/IQ_MT_MIN mechanism). That is exactly what the M3b canary +
KL gate (item 1) would police.

### 3.4 Draft length from placement, never from timing (basalt)

`docs/precision/index.md:56`: "The cumulative-acceptance threshold that ends a draft is the base
value times a factor of 1.0 to 1.5, set once from the mean CPU-served expert bytes per expert slot of
the pack's final tiers (1.0 at about 1.1 MB, 1.5 at 3.5 MB or more); it is a function of the pack and
its placement, never of timing." A closed-form, reproducible window-width rule — the same instinct as
basalt's `--prefill-cpu-share` being "a fixed 0.6… not a timing-driven value, so prefill is
reproducible bit for bit between runs". Contrast with the EMA controllers (3.1): Xolotl/maya adapt
online and accept history-dependence; basalt pins deterministic functions of state. Our campaign law
(interleave, means) tolerates the former; our bit-identity ambitions require the latter. Both are
self-consistent choices; do not mix them inside one instrument.

### 3.5 The measured verify-cost curve we owe ourselves

Xolotl's `kShape[t] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6}` is a *measured* RTX 5070
curve ("+10 ms per token with every missed expert on the CPU (bench/results/2026-09-27-spec/
window-cost), flatter with the default CPU/DMA split"). We have no equivalent for our
5090 + IQ3_S + cohorts path. F-228 gave us acceptance-by-depth but not ms/window-by-T. One
diagnostic battery (T = 1..8, standing recipe, STRATA_VERIFY_PROFILE class) banks
`kShape_custom2[]` and re-prices: adaptive depth (3.1), varlen batches (3.2), the spec-6 standing
choice itself, and any "cheaper rows" lever (hc norm-fold class). Cheapest-falsifier-first: this is
that falsifier for the whole depth lane.

---

## 4. Fork-maintenance practice (Xolotl)

### 4.1 UPSTREAM_SYNC.md — the standing sync procedure

Verbatim spine (from `docs/UPSTREAM_SYNC.md`):
- Dry run first: `git fetch upstream && git merge-tree --write-tree --name-only HEAD upstream/main`.
- "Do it in a worktree… never in the checkout a running job uses, and fast-forward the branch only
  after the gates pass. **Merge, never rebase**: both forks are public."
- Gates, in order (all before committing the merge): (1) CPU parity programs
  (`ptq1_0_test, pool_tasks_test, q8k_quant_parity, iq_avx2_parity, …` with `CUDA_VISIBLE_DEVICES=`),
  (2) "DS4 dense gate on the mini fixture, **bit-identical to the previous build**", (3) "Real model
  p600… ppl within the run-to-run band (10.4–10.65 …; F32 reference 10.4468)", (4) the other model's
  own gates.
- The one-time unrelated-histories fix, found by **tree-hash equality**: "our `main` (`6f32ec07`) and
  upstream were unrelated although `6f32ec07` has exactly upstream `a1641e9f`'s tree. The one-time fix
  was `git replace --graft 6f32ec07 a1641e9f && git merge upstream/main; git replace -d 6f32ec07`"
  (replace refs are repo-global — delete straight after).

### 4.2 The isolation rules (why their next merge is 13 files, not 51)

"Fork rules: keep our edits out of upstream's lines": (1) **New files over in-place edits** — fork
kernels live in `src/kernels/cuda/fork/*.cuh`, hooked by one `#include`; "Keep a helper next to the
code that uses it, so an upstream refactor cannot move it out from under us". (2) **Extend, don't
rewrite** — upstream's `STRATA_GU_FMTS`/`STRATA_D_FMTS` lines untouched; fork formats in
`STRATA_*_FMTS_FORK` "expanded on the line after each use". (3) **Defaults keep upstream's
behaviour** — "a fork feature that must live inside an upstream kernel takes a parameter whose
default is a no-op (the SwiGLU clamp `lim` defaults to +inf: bit-identical for every other model)".
(4) Identical shared files across their forks. Footprint is measured and published:
`git diff upstream/main --diff-filter=M --numstat` — "after the first merge 21 upstream files +727/−90;
after the isolation pass +327/−87", and "two generic improvements worth offering upstream as PRs".

### 4.3 MERGE_SCOPE.md — measuring a merge before doing it

- `git merge-tree --write-tree --name-only` gives the conflict census read-only (their table: intra-fork
  merge = 10 files/33 hunks; + mimo = 7/46; cross-lineage = 47–48 files, ~38 code).
- **Topology measured, not assumed**: ancestry checks, merge-base hashes, and the graft discovery —
  "The fork's root commit `01529aaa` has tree `4591e4cafcc8`, which is exactly the tree of upstream
  commit `f2a08d4b`" → `git replace --graft`.
- **The criss-cross trap, found the hard way**: after the graft, `merge-base --all` returns TWO bases;
  "Git then synthesises a virtual merge base… and that virtual base can itself contain conflict
  markers — verified: the stage-1 blob of `serve/server.py` literally contained `<<<<<<<<< Temporary
  merge branch 1`", producing 51 spurious conflicts where the real set is 13. Fix: intra-fork merges
  first on a single base; graft + cross-lineage merge last. (`-c core.useReplaceRefs=false` does not
  avoid it.)
- Conflict budgeting: "Raw diff size is misleading — glm's +1.6 M lines is one file
  (tokenizer.json)"; dropping the legacy engine "removes ~30 of the 46 mimo hunks".
- Resolution strategy for the hard part: "reconcile **per kernel function**, not per file; gate each
  on the existing parity tests — the harness is already there."
- Graft durability: "`git replace` refs are local and do not push… Decide before pushing."

**Application to us:** our v0.2.1-merged line (F-248) now carries in-tree edits on upstream lines:
norm_fast, FUSE_HITS, POOL_DIET, COHORTS, the T18 `job_of` fix, mmvq_il, GDN_CHUNKED, SIDE_MIXER-era
reverts, plus config/launcher divergence. Our next upstream merge will pay for every one of them in
conflict. The Xolotl practice adapted: (a) publish our footprint now (`git diff v0.2.1 --numstat` on
src/include), (b) gate the next merge on bit-parity (bit-check with pinned env) + the KL yardstick +
one MMLU gate, in that order, before any probe window, (c) prefer env-gated arms (we already do —
this is our `lim=+inf` equivalent, and it is why F-248's rollback is one cp) but for *code* that must
stay (T18 fix), consider the upstream-PR route so the fork edit disappears, and (d) when the v0.2.2
merge comes, run the merge-tree census first and resolve per kernel with the parity programs as the
arbiter.

---

## 5. Selftest / parity / CI patterns that compound

- **Parity programs with printed tolerances** (basalt): every numerics claim is a re-runnable binary
  plus a number — `dense_tc_parity` (~1e-7 RMS vs float64, "every column bitwise independent of T"),
  `mid_chain_parity`/`dec_commit_parity --finisher` (bitwise), NVFP4 grouped kernels vs per-entry
  (bitwise), CPU-tier Q6_K dot "worst error 1.3e-8 of the sum of absolute terms". maya: "Checked on
  synthetic experts against a double-precision reference: every type lands at the same ~1.5 % (expert
  kernels) and ~0.9 % (dense GEMV), which is the q8_1 activation's share" — i.e., they *explain the
  residual* rather than hide it. Upstream keeps `bench/micro/fusions_parity.cu` and
  `tests/cuda/prefill_mmq_kquant_test.cpp` with the same shape (MMQ vs double: 1.08e-2 vs FP16 path
  5e-4, printed).
- **Compiler-parity honesty** (basalt `docs/precision/builds.md`): MSVC vs GCC measured with the same
  parity programs ("rel 1.111e-06 vs 9.412e-07" — agreeing "to the third significant figure, which is
  the range FMA contraction and reduction order can move"), and the token-level consequence named:
  "the same pack, the same commit, temperature 0… `struct` is token-identical, `conv` diverges at
  token 23" — and the toolkit version excluded by rebuilding with CUDA 13.0. "Adaptive swaps must be
  **off** for any of this to mean anything."
- **Compile-only CI**: basalt's `ci.yml` pins `CUDA_VERSION: "13.0.2"`, installs
  `cuda-nvcc-13-0 … libcublas-dev-13-0`, configures `-DCMAKE_CUDA_ARCHITECTURES=120`, and runs "only
  the `cpu`-labelled tests" GPU-less. Our tree has none. Note the architecture subtlety: the FP4 MMA
  path is `sm_120a`-only (upstream `docs/NVFP4.md:140`), so `120` and `120a` are different build
  contracts — a CI pin should record which one it compiles.
- **Engine-unification exactness ladder** (Xolotl `docs/xolotl/BENCHMARKS.md`): "DeepSeek-V4.1's
  logits are bit-identical, GLM-5.3 and Qwen3.8-Flash-Next give the same greedy tokens (within each
  engine's own run-to-run variation), the 27B's teacher-forced log-probabilities differ by 0.015 nats
  on average" — three escalating identities matched to what each engine can honestly provide. Our
  T20 merge validated with MMLU + re-bank; the ladder adds the cheap middle rungs.
- **Token-identity A/B as a bug catcher** (maya `bench/results/2026-10-06-glm-spec/README.md`): a
  double-edit KV bug "found by an A/B that should have been token-identical (decode is now verified
  identical across tier states)". Our F-205 wrapper bug was found by no-op validation; the family
  generalizes: *state the identity a change claims, then test that identity*.
- **The greedy-identity caveat, three-source confirmed**: our F-245 ("stock .2.1 greedy outputs are
  NOT stable across boots at identical env"); Xolotl `DETAILS.md` ("with the IQ models… the same
  prompt at temperature 0 can end in a different (equally good) answer when the drafting, the cache
  state or a resumed conversation differ (issue #152); `STRATA_IQ_MT_MIN=1`… uses the multi-token
  kernels for every group: the answer then no longer depends on the drafting", cost IQ3_S −1..−3 %);
  basalt (identity requires `--adapt-swaps 0` + pinned tiers). Resolution: greedy hashing is a valid
  instrument ONLY with pinned placement + deterministic kernel selection; otherwise use the
  forced-window KLD identity (1.3) which is byte-stable across widths and placements' rounding.

---

## 6. Contradictions and their resolution

1. **Is greedy output a valid instrument?** basalt proves 275/275 bit-identical prompts and 62k
   bit-identical logit rows; our F-245 proves cross-boot greedy hashes are NOT stable on our line.
   Resolution (three sources): identity claims require pinned expert placement and deterministic
   kernel selection; our engine's adaptive cache and IQ3_S CPU tiers (multi-token vs single-token
   rounding depending on drafts-in-window, upstream issue #152) void it. Both camps agree on the
   placement-free alternative: teacher-forced KLD, byte-identical across window widths (basalt
   forced-window table).
2. **Is KLD comparable across references?** No — basalt quantifies 0.01–0.02 KLD of reference noise
   in every stock-base number; maya's wikitext KL 0.934 shows a memorized reference can dominate.
   Resolution: same-reference, same-text, same-engine-path comparisons only; report the reference and
   its noise; flag memorized text.
3. **Per-row vs per-window cost of deeper spec.** T13A's per-row estimate (expert bytes ≈0.55 ms/row)
   vs F-228's measured full-graph T-scaling. The family's controllers settle it by construction:
   `dense_ratio[n]`/`kShape[t]` are per-window-size *measured* curves, with the coupon-collector term
   for shared expert reads. Our planned cost stair (3.5) is the same instrument applied to our box.
4. **Adaptive (EMA) vs closed-form (placement-derived) spec control.** Xolotl/maya adapt online
   (history-dependent, needs interleave discipline); basalt derives windows from pack bytes
   (deterministic, reproducible). Not a contradiction — different invariants prioritized (throughput
   vs reproducibility); both document their choice. For us: instruments reproducible (basalt style),
   control may adapt (controller) but then its verdicts must respect law 1.
5. **"Merge never rebase" (Xolotl) vs our T20 in-place reset.** Xolotl's rule serves public forks
   with collaborators; our reset (F-248) served a solo lane with a provenance archive kept as a lab
   copy. No change needed, but the UPSTREAM_SYNC gate order still applies to our *next* merge.

---

## 7. Open questions

1. Does our IQ3_S pack's CPU-tier rounding (Q8_K-per-256 vs q8_1-per-32) put a quantifiable floor
   under any KL gate we adopt? (basalt measured ~0.8 % of routed experts on CPU in their gate; ours
   differs — measure, don't assume.)
2. What is our verify-window cost curve `ms/window(T)` on the standing (cohort) recipe? (3.5 — the
   one diagnostic boot the depth lane still owes.)
3. Does the v0.2.1 engine expose a per-position draft-confidence we can log for the varlen lever
   (3.2), or is confidence only implicit in min-p filtering? (Determines whether 3.2 is an engine
   feature port or an algorithm port.)
4. Would the controller's lookup priors transfer at all to our lane, given we run suffix drafts off
   and our acceptance profile (F-228: 2.20 offered, ~2.88 tokens/window at min-p 0.5) differs from
   the 0.86-per-position prior? (The EMA learns it; the question is only how fast.)
5. Is upstream's `strata bench run` protocol still draft-status in v0.2.1.4 ("pilot validation
   pending")? If we adopt its numbers, label them draft-protocol until upstream activates.
6. Re-quant lane (ladder item): does an error-feedback/GPTQ pipeline exist for the Qwen-lineage GSQ
   formats we use, or only for maya's GLM experts and Xolotl's DS4 ternaries? (Affects item 15's cost.)

---

## 8. Implications for the campaign

- **Spend MMLU last, not first.** With STRATA_LOGPOS/--short-read already in the tree, a deterministic
  KL yardstick (item 1) converts the law-3 gate from a ±2.8-point lottery into a two-stage gate:
  KL decides whether the distribution moved; MMLU prices the damage only when it did.
- **The next upstream merge should be gated bit-first** (item 4): parity → KL → MMLU → perf, with a
  published edit footprint and a merge-tree census before the first resolution.
- **Bank our own cost curve** (item 14/3.5) before any further depth/verify-window arm; it is the
  cheapest falsifier for the whole lane and the input to the controller (item 2) and varlen (item 13).
- **Price, don't just ban, acceptance trades** (item 5): the TV/token exchange rate gives law 10 a
  market price and kills "free acceptance" hypotheses at zero boot cost.
- **Freeze token ids** for the probe prompts (item 2.3) so a future merge cannot silently move the
  workload; add loop/ctx-fill as a cheap second release gate (item 11).
- **T14B's evaluator gets a spec** (item 12): teacher-forced acceptance + confidence-calibration
  quartiles, anchored at our banked 2.78 tokens/window against the DFlash2 card's 2.50–3.08 stock
  range.
- **What we would NOT adopt**: relaxed acceptance (our canary law and the family's own TV pricing
  argue the trade only pays where diversity is worthless); timing-derived window widths inside
  identity instruments; upstream's quick `strata_bench.py` numbers as comparable (the protocol doc
  itself forbids it).

## Methods note

Productive: reading the four `docs/precision/*.md` files end-to-end (basalt's index/quality/features
are the family's best single artifact); diffing Xolotl's `src/spec/` against maya's (established
direction of evolution: Xolotl added staleness re-probe + chaining); grepping our own tree for
STRATA_LOGPOS/--short-read (turned a "port the engine hooks" item into "port the Python only").
Unproductive / dead ends, recorded to avoid repeats: searching Xolotl's sparse clone for
`bench/results/2026-09-27-spec/window-cost` (bench/ excluded from the clone per MERGE_SCOPE §7 — the
kShape constants in code are the surviving record); searching for a dedicated "verify-cost curve"
doc in maya (it lives in SETTINGS.md prose and the controller code, not a bench report); DeepInfer's
`eval/` (AIME/GPQA/HLE stack — not applicable to our gate class); DFlash2 assets beyond the README
acceptance table (weights are LFS pointers by design). The "CUDA 13.1+120a" hint in the task resolved
to: basalt CI pins CUDA 13.0.2 + sm_120; upstream's FP4 path gates on sm_120a; family bench boxes are
documented as "CUDA 13.1" — the adoptable practice is the pinned compile-only CI, not the literal
version string.

### Source list (primary anchors)

- basalt `fb0944c7`: docs/precision/{index,quality,features,builds}.md; tools/{ppl_gate,greedy_identity}.py;
  data/bench/*.ids; .github/workflows/ci.yml
- Xolotl `cd965e3c`: tools/quant_quality.py; MERGE_SCOPE.md; docs/UPSTREAM_SYNC.md;
  src/spec/{controller,draft_policy,controller_test,draft_policy_test}.cpp; docs/DETAILS.md
  (sampled-drafting, IQ_MT_MIN, PLE-table KL, ringbytes sections); docs/xolotl/BENCHMARKS.md;
  docs/ds4/{V41_1017_QUANT_PLAN,V41_DSPARK_AUDIT,V41_RESUME_PROMPT}.md; docs/RESEARCH_RUNS.md
- maya `35bb2b2` (v1.1.20): docs/QUANT-PLAN.md; docs/SETTINGS.md (SPEC_FOLLOW/SPEC_SAMPLE/MTP_KEEP/
  TRUNK_KEEP/split-pricing rows); bench/results/{MAYA-S.md,2026-10-06-glm-spec/README.md};
  tools/maya_quant/{MODEL_CARD.md,kl_eval.py,loop_test.py}; tools/canonical_xcheck.py;
  src/spec/controller_test.cpp
- Strata upstream `9de04cd4` (v0.2.1.4): docs/BENCHMARK_PROTOCOL.md; bench/prompts/standard-v1/
  {build.json,MANIFEST.sha256,requests/*}; bench/results/…/spec-sweep/{sweep.py,sweep.json};
  bench/micro/fusions_parity.cu; docs/NVFP4.md
- DeepInfer `9bd9cf2`: deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py;
  README.md (DSpark section); deepinfer/environ.py; eval/run_eval.py
- DFlash2 `a22a174`: README.md (acceptance-length table)
- Our tree: AGENTS.md (campaign doctrine + performance law); campaign/GOAL-LINUX-SUPERIORITY.md;
  SVG-CUSTOM.md F-172..F-250 (esp. F-228, F-229, F-230, F-245, F-248, F-249, F-250);
  campaign/{probe-lane,mmlu-pro-gate,mimicbench2,depth-lane,bit-check}.py;
  src/core/verify.cpp:3246; include/strata/core/verify.hpp:330; src/program/generate.cpp:834,999,2696
