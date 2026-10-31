---
type: digest
id: triage-t1
created: 2026-10-31
status: evidence
topic: T1 triage — W1 adaptive speculation + W2 ungated draft chain, priced per law 15 before any run
scope: read-only pricing; no code changed; anchors verified at HEAD a1dd4bb2 (board noted f240701e — mild drift, sites live)
---

# T1 — Spec-core triage: every W1/W2 item priced against THE NEEDLE

Bar (laws 2/4/15): >= +4% on the target lane, no lane or latency term beyond -4%. Inside the band on every lane AND latency term = NOT RUN -> compose or drop.

## Verdict table

| Item | Verdict | Priced expectation | Lane it must clear |
|---|---|---|---|
| W1.1 controller at generate.cpp:16083 | KEEP (gate on free falsifier first) | +35-38% solo IF the M15 22 ms/9-row population is the standing lane; inside band if the 11.5 ms anatomy is (bimodal — see Flag 1) | solo M1 |
| W1.2 dp_ok/dp_tested -> controller EMAs + growth fix | KEEP (composed into W1.1) | enabler, 0 ms alone; without it T* locks shallow and cold positions never learn | solo M1 |
| W1.3 batch per-slot depth (keep[b]) + W18 ramp + drafts_offered fix | DOWNGRADE (compose into W1.1 batch arm) | +1-3% conc — inside band standalone | conc 6-way |
| W1.4 3v3 same-hour battery | KEEP | gate, not needle; protects the -4% floor | all lanes |
| W1.5 MMLU-Pro 2x >= 92.9 + mimic | KEEP | gate; sizing-only -> canary risk low, provenance change -> mimic mandatory | quality |
| W1.6 tree verify STRATA_MTP_TOP2 | DOWNGRADE (phase-2, behind W1.1 only) | -5% to +3% standalone (rows at 0.028 tok/ms vs window avg 0.342); pays only where controller prices chain depth dead | solo M1 |
| W2.1 STRATA_DRAFT_UNGATED default-on (min_p>0) | KEEP | +3.9-6.6% solo and pair (draft term -0.43 to -0.71 ms); conc 0 by construction (batch path already ungated) | solo, pair |
| W2.2 round+steps as ONE graph, mapped staging | KEEP (composed with W2.1) | +1.8-3.7% solo incremental over W2.1; composed total +5.7-9% | solo, pair |
| W2.3 draft ms/window delta vs 0.86-1.42 baseline | KEEP (instrument) | the row's falsifier; proves/refutes "bubbles dominate the draft term" | — |
| W2.4 3v3 + MMLU + bit-identical verify | KEEP (MMLU once, on the composed promotion) | gate; Philox staging is the silent-divergence surface | all lanes |

Count: 8 KEEP, 2 DOWNGRADE, 0 KILL.

## Conversion basis (all arithmetic from mission anchors)

Solo 207.4 t/s = 2.43 tok / 11.5 ms window; pair 198.4; conc 341.5 agg @ 9 rows, 23.8 ms (M8). Wall fit: 8.4 + 1.57xrows ms (M15, R2=0.99). t/s ratio = wall_old/wall_new at held tokens. Draft term 0.86-1.42 ms/window. Drift band ~±4-8% (M1 180-222).

## W1 pricing

- **W1.1** — M15's own math: depth-8 window 22.5 ms -> ~16.3 ms at same tokens: 121.4->167 (+38%), 163.1->221 (+35%); vs 207.4 standing that is 280-286 t/s, 9x the bar. Controller upside is bounded by waste actually present: rows=(11.5-8.4)/1.57=2.0 on the standing anatomy -> row term ~3.1 ms, 2.43 tok / ~3 rows ~ 80% efficient -> nothing to trim, inside band. Leviathan condition holds at every measured acceptance (alpha 0.685-0.89 > c 0.4-0.6), so T=1 collapse needs alpha<~0.5 — unobserved; min_gain guard is exactly that tripwire. Risks: EMA cold start (guard: law-11 discarded warmup leg + prewarm2 replay feeds EMAs), batch over-trim (guard: cost_ms must see batch size — digest w1, TRT-LLM batch-x-rows note; M1 law-4 parity), quality none (sizing-only, lossless; M3b canary + gate W1.5). Falsifier (free, law 4): rows/window from decode-timing lines + per-depth dp_ok/dp_tested dump (banked at generate.cpp:15209) before any wiring.
- **W1.2** — verified in source: growth fix exists (controller.cpp:74-79), controller EMAs unfed from production paths (Controller::choose referenced only by controller_test — dead-code claim confirmed). Compose into W1.1; not independently runnable.
- **W1.3** — batch D=1 already 78-91% row-efficient (M15); depth-2 row on a keep=1 slot yields <=0.5-0.6 tok per 1.57 ms = 0.32-0.38 tok/ms vs window average 0.342 (8.13 tok/23.8 ms) -> +1-3% conc, inside band. M8's 341.5->450+ is refuted by M15's own downgrade. keep[b] wiring shares the controller (compose); drafts_offered telemetry fix is free and unblocks per-request acceptance reads. W18 D=2/3 ramp: env-only, one boot, measure-do-not-plan (M15 verbatim).
- **W1.4/W1.5** — mandatory proof burden (laws 2/3/4/10). No needle.
- **W1.6** — salvage per runner-up row = P(top-1 wrong) 0.11 x P(runner-up right) ~0.4 = 0.044 tok per +1.57 ms = 0.028 tok/ms — 12x below the window average; unconditional adds are lane-negative. Tree evidence (SGLang/TRT-LLM/Medusa/DSBD) lives in lower-acceptance or sublinear-row-cost regimes; ours is 1.57 ms/row linear at alpha 0.89. Re-price trigger: arm a rank-2 prob-mass counter (calibration deciles already log cal_on/cal_n); mean rank-2 mass >> 0.11 at depth >=2 doubles the salvage and reopens it. Compose: tree rows only where the controller prices chain depth dead (DSBD cumulative-mass rule).

## W2 pricing

- **W2.1** — draft term 0.86-1.42 ms; ungated removes 2 host RTTs/step ("GPU idled between steps while the host spun" — code comment, mtp.cpp:1741-1752). Roadmap expects ~half the term: -0.43 to -0.71 ms -> solo 11.5/10.79-11.07 = +6.6/+3.9%; pair same Delta; conc 0 — batch draft_launch is ALREADY ungated unconditionally (mtp.cpp:1884-1895, "the standing rule here", verified). Cost: ~50 us x dead steps (~0.25 ms worst) — net favorable. Bit-identical (same graphs, stream order, Philox). Rollback: env unset. Cheapest item in either row; a default flip, not a build.
- **W2.2** — depends on W2.1 (the host read forces per-step graph separation). Residual post-W2.1: 5-7 cudaGraphLaunch + host loop ~ 0.2-0.4 ms -> one graph removes it: +1.8-3.7% incremental; composed total draft term 0.86-1.42 -> ~0.15-0.3 ms = +5.7-9% solo/pair. Risks: staged Philox in-graph has NO upstream precedent (digest w2: SGLang resolves RNG via capture-time variant graphs; FlashInfer baked-generator trap) — the one surface where bit-identity silently breaks; guard = bitwise parity + mimic + MMLU once on the promotion. Pre-allocate all-step KV slots (SGLang out_cache_loc pattern); boot capture cost folds into W11's sweep item (already lists MTP round/step graphs).
- **W2.3** — the instrument that prices W2.1's split (bubbles vs graph-exec). If post-flip draft ms reads >=0.8, W2.2's premise dies — run it first or simultaneously.
- **W2.4** — gate. Per roadmap Protocol 2, bit-parity moves need parity suites + battery for speed; keep MMLU once on the composed promotion (Philox surface), not per leg.

## Cross-cutting

- **KEEP order by t/s-per-effort:** W2.1 (hours, +4-7%) > W2.2 (days, +1.8-3.7% incremental) > W1.1+W1.2 (days, bimodal 0-38%, the only >10% item; falsifier first) > W1.3 (composed, +1-3%) > W1.6 (conditional). Gates throughout.
- **Highest-leverage composition:** W2.1+W2.2 as one bit-identical promotion (draft-chain rewrite), then W1.1+W1.2 wired on the SAME promoted binary's battery series — sequenced, not simultaneous: the controller's cost_ms EMAs must learn the POST-W2 draft phase (cheaper draft lowers c, which RAISES Leviathan's T*; controller landed first bakes stale costs the W2 landing invalidates). Law 12's stacked-wins clause covers one merged promotion if the operator prefers a single plotted step: graph work + controller + EMA feed + telemetry fix, one 3v3 series + MMLU 2x + mimic + parity.
- **Flag 1 (digest-vs-telemetry contradiction, biggest):** W1 board detail says "2.39-2.47 tok per 8-row solo window, ~5.5 rows discarded" — but 8 rows by the M15 fit = 21 ms, and standing decode-timing reads 11.3-11.7 ms at those tokens; M15's probe population (2.73-3.61 tok @ 8.7-9.0 rows = 30-41%) also disagrees with M2/M23 session WSTAT (~7.2 tok @ 9.6 rows ~ 75%). The +35-38% holds only for the 22 ms/9-row population. Resolution before wiring: the W1.1 falsifier (free).
- **Flag 2:** kShape prior (3.6x at 8 rows, draft_policy.cpp:19) vs M15 fit (2.1x incl. fixed term) — the controller's cold-start prior overstates deep-row cost, biasing T* low; conservative solo, wrong at batch margins; W1.2's EMA feed is the correction.
- **Flag 3:** do not import digest cross-engine percentages (Leviathan oracle +60%, DISCO +10.3%, AdaEAGLE 42%) into the row's Expected field — other-engine populations; local anchors only. Digests label them correctly; this is an over-reading guard.

## Open questions

1. Which window population is the standing solo lane — 22 ms/9-row or 11.5 ms/~3-row? (falsifier: rows/window + per-depth dump from custom-iq3_s.log)
2. Draft-term split: host-RTT bubbles vs graph-exec floor? (falsifier: W2.3 after the W2.1 flip)
3. Rank-2 prob mass of the MTP head at depth >=2? (falsifier: calibration-decile counter; re-opens W1.6)
4. Depth-2 acceptance on the batch lane? (falsifier: W18 ramp, env-only, one boot)

## Sources

- AGENTS.md, The performance campaign law (laws 2, 4, 4b, 9, 12, 13, 15) — read 2026-10-31.
- /home/svg/LLM/.zcode/rich-tracking/strata-perf-roadmap.board.json, rows w1-adaptive-spec / w2-ungated-draft (items + stage baselines).
- campaign/MEASURED-opt-map-2026-10-27.md M2, M8, M10, M15, M23; campaign/ROADMAP-perf-8moves.md §W1/§W2 + Protocol.
- .docs/research/digest/w1-adaptive-speculation.md, w2-ungated-draft-chain.md (2026-10-31).
- Code verified at HEAD a1dd4bb2: src/spec/controller.cpp:36-79 (dead, growth fix present); src/spec/draft_policy.cpp:12-16 (kShape); src/core/mtp.cpp:1745-1757 (ungated env, default off), 1884-1895 (batch path already ungated); src/program/generate.cpp:16083-16089 (fixed-T + min_p cut-only trim). Anchor drift confirmed (board's 12422-12435 now BT/BDONE emission — re-grep before editing).
