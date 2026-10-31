---
type: digest
id: triage-t2
created: 2026-10-31
status: evidence
topic: T2 triage — W3 (batch device acceptance) + W4 (MoE defer combine / foresight / fusion) priced against the needle before any run (law 15)
---

# T2 — W3+W4 items priced against the needle

Bar: >= +4% target lane, no lane or latency term worse than -4% (laws 2/4/15). Pricing convention (M2): dlane% ~ dms/window / window_ms — conc 12.14 ms @ 4.93 rows (1 ms = 8.2%), solo 11.3 ms (8.8%). Row conservation (W3): tokens/window = rows x acceptance; conc today
4.93 x 0.685 = 3.38 tok/window. No KILL verdicts: every sub-bar item composes (law 4b/15); no item's premise is false enough to drop the machinery outright.

## Verdict table

| item | needle math | verdict | conf |
|---|---|---|---|
| W3.1 batch p/q verify, one spec_verify_merge launch | acc 68.5 -> 85-92% = +24-34% tok/window, wall ~0 -> pair/conc +24-34% | KEEP (the row) | high |
| W3.2 TYPICAL_DEVICE unconditional; host rescue deleted | 0 ms standing (BATCH_RESCUE=0, device env=1); <=0.1 ms fallback boots (0.8%) | DOWNGRADE -> fold into 3.1 commit | high |
| W3.3 re-sample in graph (mapped Philox; kill verify.cpp:4565 sync) | sync tail 0.03-0.1 ms = 0.25-0.8%/lane, sub-band ALONE; required for 3.1 graph | KEEP, bundle-only | med |
| W3.4 batch sample_rows one launch | ~8 x 4 us = 0.03 ms = 0.25%, sub-band alone | KEEP, bundle-only | high |
| W3.5 gate: 3v3 + MMLU 2x + mimic | prices the bundle; not an arm | KEEP mandatory | — |
| W4.1 deferred combine (bo_ double-buffer, wait after L+1 mixer) | 30-60% of 2.09 exposed = -0.6-1.25 ms -> conc +5-10% | KEEP | med-high |
| W4.2 device foresight (L+1 router+top-16 during L; early H2D) | -1.0-1.5 ms exposure, +0.5-0.9 ms GEMV (hides; M7 latency-bound) -> conc +4-10% | KEEP, after 4.1 | med |
| W4.3 Mix/Combine-fused HC node | hc 2.39/4.00 ms (light/heavy); precedent +7.6/+5.5% e2e -> +4-10% all lanes | KEEP, nsys FIRST (F-218) | med-high |
| W4.4 router+top-10+plan+bell one node | kernels-only -0.3-0.6 ms = +2.5-5% conc borderline; unlocks host-plan 1.2-1.9 ms (device plan = +10-16%) | KEEP scoped; standalone DOWNGRADE at low estimate | med |
| W4.5 re-measure CPU-expert + router/plan/bell | map stale (F-218 2026-10-16; 12.14 vs 262K-era 13.40 @ 4.81) | KEEP first, one boot | — |
| W4.6 gate: 3v3 conc + M1 no-regression + MMLU 2x | prices W4; law-4 M1 guard | KEEP mandatory | — |

## Per-item notes (risk / guardrail / remediation / falsifier)

**W3.1 — the board's biggest needle.** (a) ~30-40% of paid batch rows die on exact-match disagreement; p/q accepts each at min(1,p/q): +24-34% tokens/window at ~zero added wall — exceeds the row's own +8-20% claim and the bar 5-8x. (b) Risks: output-changing (acceptance semantics); port vLLM's precision set — fp32 logsumexp drift shifted acceptance (#40651 -> #64242): NaN-guard, open-interval uniforms, last-valid-index fallback (scout). F-30b law: p/q is NOT the rescue; STRATA_BATCH_RESCUE stays 0; never reintroduce the rescue under any name. (c) Guardrail: MMLU 2x floor 92.9 + mimic + 3v3; snapshot engine/strata.pre-w3; env-gated. (d) Falsifier (cheapest, law 4): instrument one conc window to log exact-match AND p/q verdicts per row (serial kernel already computes p/q). Recovered mass < 12% (acc < ~77%) drops the needle to the bar -> compose or drop with pricing recorded.

**W3.2 — hygiene, not an arm.** Standing needle 0: BATCH_RESCUE=0 (F-30b) and TYPICAL_DEVICE=1 live (M23) leave the 248K-logit D2H path dormant; law 15 -> NOT RUN alone; land as cleanup in 3.1's gated commit (also permanently removes the F-30b re-arm landmine). Risk: deleting the fallback before the device kernel is proven — same gate covers. Falsifier: counter on generate.cpp:379 hits at standing = 0 (free).

**W3.3 — enabler, sub-band alone.** Every temp>0 request pays post-window sample_tokens + full cudaStreamSynchronize (verify.cpp:4565-4574, verified). Priced 0.03-0.1 ms/window — a real latency term but tiny; its value is graph capture of the sampled path (3.1 needs accept/resample in-graph or the sync returns). Risk: FlashInfer trap (scout) — torch.Generator bakes capture-time state; mapped-staged counter (verify.cpp:2785 pattern) is TRT-LLM-validated; prefer the vLLM/SGLang counter-FREE keyed draw f(seed,pos,col) — zero staging, deterministic under reordering. Falsifier: replay-equivalence diff at fixed seed. **W3.4 — free rider:** per-row launch loop (verify.cpp:4238-4246, verified) -> one grid over S rows, ~0.03 ms/window; never arm alone; rides the 3.1 graph.

**W4.1 — deferred combine.** (a) Exposed CPU experts 2.04-2.09 ms/window; deferral hides the wait under L+1 mixer/attention (hc+attention = 3-8 ms/window of independent work). Capture 30-60% -> -0.6-1.25 ms -> conc +5-10%; bar met if capture >= ~0.5 ms. (b) Risk: MEDIUM graph restructure — buffer-lifetime bug = silent corruption (F-219 class); wait kernel carries its guard (2261-2262). Bit-identical in theory -> parity suite + M3b canary; MMLU 2x insurance. Remediation: env off = rollback; law 13 if regression. (c) Falsifier (free): STRATA_DM_WAIT_STATS boot — spin distribution per layer position; one-straggler tail means deferral hides less -> re-price.

**W4.2 — foresight.** (a) Eliseev trigger (gate current hidden, issue after L's expert load; scout prefers over DWDP completion trigger). Cuts residual exposure after 4.1 by 50-70% -> +4-8% conc incremental; cost one extra router GEMV+top-16/layer ~ 0.5-0.9 ms IF SMs busy — M7 says decode is latency-bound (waiting) so it likely hides. Net +4-10% conc. (b) Risk: stale-hidden router divergence (top-16 must intersect resident table; fs->take exists, expert_source.cpp:4763-4764, but CPU-fed, depth-0, skip-when-busy); copy-stream race with the doorbell path. Bit-identical in theory -> parity + MMLU 2x. (c) Sequence AFTER 4.1 (its guarded wait is the fallback shape). Falsifier: existing "strata batch:" telemetry — if the inter-layer gap < GEMV+H2D time, foresight cannot cover.

**W4.3 — HC Mix/Combine fusion.** (a) hc reads 2.39 light / 4.00 heavy ms/window; SGLang-style fusion measured +7.6/+5.5% e2e on this model class (F-218); 20-30% off hc = +4-10% all lanes. F-217 killed grid surgery (M1 -9.2%); fusion is the surviving hc lever. (b) Risk: fused math (not bitwise) -> MMLU 2x + mimic mandatory; nsys FIRST (F-218 law) — the 2026-10-16 class map is stale post-ATTN_DAILY. (c) Falsifier: the nsys pass itself — if hc ms live in norms/splits that cannot fold into one node, re-price before building.

**W4.4 — router+plan+bell fusion.** (a) Term = 1.43-1.90 ms/window but host plan RTT (25-40 us x 48 = 1.2-1.92 ms) is the dominant slice and kernel fusion does NOT remove it. Standalone needle 0.3-0.6 ms = +2.5-5% conc, borderline — law 15: not run alone at the low estimate. The prize is the device-plan endpoint (vLLM EM upper-bound grid + device early-exit; TRT-LLM bans host copy plans): full 1.2-1.9 ms -> +10-16% conc — phase 2 on 4.2's machinery. (b) Risk: graph-capture constraints (shape-fixed, mask-validity — DeepGEMM masked-GEMM precedent); doorbell semantics touch the CPU-miss path -> races. Parity + MMLU 2x.

**W4.5 / W4.6.** W4.5 runs first: baselines are 2026-10-27-era and F-218's class map is three adoptions old; one boot + two greps ("strata batch:" / "GPU stages") prices 4.1-4.4 honestly. W4.6: conc target lane, M1 no-regression explicit (law 4), MMLU 2x.

## Cross-cutting

Order by t/s-per-effort: (1) W3 bundle [3.1+3.3+3.4, 3.2 cleanup] — needle 5-8x bar on an existing device kernel; (2) W4.5 re-measure; (3) W4.1+4.2 as ONE composed candidate (same 2.09 ms term); (4) nsys -> W4.3+4.4 fusion bundle; (5) device-plan phase 2. W3 and W4 attack different window terms (tokens/window vs wall) — they compose multiplicatively; ladder them as separate promotions (law 10: stack confirmed wins into one promotion so each chart step clears noise). Best with W1 (pending, 0%): W3 first — higher acceptance raises every row's value; W1's adaptive depth then trims the SAME batch windows. W1 before W3 depth-tunes against exact-match waste it cannot see. Law-15 sweep — inside band on every lane and latency term, NOT RUN standalone: W3.2, W3.3-alone, W3.4-alone, W4.4-at-low-estimate; all composed above or dropped with pricing recorded. W4.1 alone viable; 4.2 alone smaller but above bar — 4.1->4.2.

## Contradictions (digest vs telemetry)

1. copy+combine: mission anchor 1.06-1.33 ms vs F-218 1.08 light / 1.41 heavy. Anchor string not found in files searched; W4.5's re-measure settles it — do not price 4.1 off either.
2. Solo claim: row W4 expects solo +3-8%; M4 prices solo CPU-fetch wait at ~0.2% of turn. The solo needle exists only via router+plan+bell (1.43-1.90 ms on an 11.3 ms window = 13-17%), i.e. via 4.4/device-plan — CPU-expert items alone do NOT move solo. Row overstates until fusion lands.
3. Conc reference differs by era: 294.1 rows/s standing (T59) vs 341.5 (M8 6-lane) vs 271.5 (262K-era anchor). Pricing basis kept at window anatomy (12.14 @ 4.93).
4. W3 expected +8-20% is an UNDERestimate vs row conservation (+24-34%); the gap is the unpinned min(1,p/q) mass — W3.1's falsifier measures it before any kernel work.

## Sources

AGENTS.md laws 2/4/4b/9/10/12/13/15; campaign/ROADMAP-perf-8moves.md (Baseline, W3, W4, W13, protocol); campaign/MEASURED-opt-map-2026-10-27.md M2/M4/M7/M8/M15/M23; board rows w3-batch-accept + w4-cpu-rendezvous (/home/svg/LLM/.zcode/rich-tracking/strata-perf-roadmap.board.json); digests w3-batch-device-acceptance.md + w4-moe-rendezvous-fusion.md (scouts 2026-10-31, relaying vLLM/TRT-LLM/SGLang/FlashInfer/DeepGEMM/KTransformers/Eliseev-Mazur primaries); SVG-CUSTOM.md F-217/F-218/F-30b/F-219; code spot-checks this pass: verify.cpp:4565-4574, 2232-2246; generate.cpp:379-392, 12505-12520; expert_source.cpp:4598-4607.

## Methods note

Read-only triage; only this file written; no code changed. Not found at source: mission anchors copy+combine 1.06-1.33 and 262K-era 13.40 @ 4.81 / 271.5 rows/s (searched campaign/*.md, both digests, tracker/lanes.csv — its tail is DeepSeek-era rows; anchors taken as dispatcher-provided, flagged above). Unproductive: no banked measurement of batch-lane min(1,p/q) acceptance mass anywhere in the record — the single number that swings W3.1 between +24% and +34%; measure first.
