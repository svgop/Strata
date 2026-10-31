---
type: digest
id: triage-t5
created: 2026-10-31
status: evidence
scope: T5 triage — price every W8 (7) + W11 (6) acceptance item against THE NEEDLE (AGENTS.md law 15); read-only
method: board rows + MEASURED-opt-map M1/M5/M8/M9/M10/M12/M14/M18/M19/M21/M23/M25 + ROADMAP §W8/§W11/§W17/§W15/§W19 + digests w8/w11 + fresh log pull (custom-iq3_s.log, 2026-10-31) + code checks (server.py:325-339, launcher). Bar: >=+4% target lane or MATERIAL latency-term cut; none beyond −4%; regression remediation = scope/compose/rebuild (law 13).
---

# T5 triage — W8 server lane + W11 cold/jitter, priced per item

## Verdict table (ordered by latency-recovery-per-effort within each row; numbers = Δms/event × events, first-class)

| # | Item | Verdict | Needle math (latency terms first) | t/s side | Conf |
|---|---|---|---|---|---|
| W11-6 | Cache budget for multi-100k tenancy | **KEEP** (first) | Eviction → full cold re-read **11.3 s @80k** (M25, measured) × 7 evictions observed; block-fallback tax up to ~5 s/turn @100k (M21) | none (host RAM only) | HIGH |
| W8-7 | W17 cadence flag at next boot | **KEEP** (first, zero code) | ITL spikes **0.1–0.8 s/event** + read TTFT 3–5× (6.6–6.9 s → ~1.5–2 s implied) per interleave read (M18, this engine); vLLM #44558: P50 ITL −88%, TTFT better, t/s flat | t/s flat (upstream) | HIGH |
| W11-3 | Spare pool above es_demote watermark | **KEEP** | Park flips 0.0 (spare absorbs, M12) → **p50 48.7 ms** real D2H × 1,847 demotes (log pull; pool seen drained to 0 chunks); max 807.9 ms | expert-hit guard below | MED-HIGH |
| W11-2 | Adapt tier-swap budget after idle | **KEEP** | 96-swap storm/round after idle; ~69 ms/swap composition (join 2.46 + park 24–52 + restore 12–16 + promote 12–17) ≈ **seconds on first turn post-idle**; jitter 126–294 t/s intervals (M5) is the same event | none (smoothness) | MED-HIGH |
| W11-1 | Boot-sweep pre-capture of hot layouts | **KEEP**, pricing corrected DOWN | Lazy capture **10–40 ms × first-round-of-layout events**; log: 3,322 events over **207 distinct layouts** → full sweep ≈ 2–8 s boot (hot-set top-20 ≈ 35% of events, <1 s) | cold-first-round −10–25% claimed; TTFT term is the priced needle | HIGH |
| W11-5 | Cold-3v3 battery (gate) | **KEEP** | Instrument, not lever: no W11 item closes without cold-first-round t/s + TTFT p50/p99 (law 1/11/12; warm-only gates are blind to every cold term above) | — | HIGH |
| W8-2 | In-engine PROMOTE at window boundary | **KEEP**, scoped | Solo→pair join **80–150 ms/event** (STOP+drain+full-KV park D2H+restore+promote) × join events (low frequency solo-dominant); elastic path already cheap (p50 2.6 ms) — win is the STOP/park half; upstream: all 6 stacks join with zero park | removes parked-image churn → fewer cache evictions (feeds W11-6) | MED |
| W8-6 | Conc battery 6-way (gate) | **KEEP** (before W8 items) | Instrument; owns the unexplained **2-lane anomaly (aggregate 164 < solo 209)** and TTFT 0.03–0.62 s batch spread (M8); STRATA_SERVE_TELEMETRY=1 | — | HIGH |
| W8-1 | Round-robin chunked reads between admissions | **DOWNGRADE** (scope after W17) | Admission queueing ~0 (p99 ≤0.5 ms, M10) → no admission-order needle on this traffic; the real term is M18's graph-replay eviction, mostly owned by cadence; residual = 2+ queued reads (rare; BYIELD fired 0/3) | 6-lane reads spike clocks 1785 MHz (M24) — cadence owns | MED |
| W11-4 | Background fadvise after drop_file_cache | **DOWNGRADE** (fold into W11-2) | Solo fetch wait 7–22 ms/turn (~0.2%, M4); post-drop re-warm 33 GB observed but read-ahead already "used 100%" (M11) — unpriced residue is only the post-drop window | conc contention term belongs to W4 | LOW-MED |
| W8-4 | Slot-free handoff vs 1 s poll | **DOWNGRADE** (hygiene bundle) | Worst case 1 s/pickup, but admission p99 ≤0.5 ms measured → term ~0 on this traffic; map says the poll is "not yet measured" and the stagger probe missed it | none | MED (term absent) |
| W8-3 | SSE batch emission | **KILL** (latency) | M14: LATENCY-NEUTRAL — client gap ≈ server gap (−3…−18 coalescence); ITL is window wall time. GIL/syscall upside unpriced, not a needle | reopen only >6 lanes or non-loopback | HIGH |
| W8-5 | Per-token lock sampling | **KILL** (perf) | µs-scale takes vs 23.1 ms window wall; M14 showed the *heavier* per-token send adds nothing measurable → no priced needle on any latency term | none | MED |

## Per-item: risks, guardrails, remediation

- **W11-6:** risk = host-RAM ceiling (budget 48 GiB, 18.2 GiB held — headroom is real but sized it); stable hashes required (LMCache/Mooncake restart bugs trace to unstable hashes). Guard: eviction counter → 0 at multi-100k tenancy. Remediation if RAM binds: LRU depth over slot count, never silent eviction (law 14).
- **W8-7:** risk = TTFT p99 regression from delaying prefill admission (vLLM N=3 precedent: p99 TTFT 630→3,175 ms — ship N=2, never sweep N; waiver already spares the lone request). Guard: measure TTFT p50/p99 beside lane ITL, one boot, env rollback. Cadence also flattens the M24 prefill-storm clock spike.
- **W11-3:** risk = expert-cache tail refill evicts hot experts (hit 93–97%). Guard: refill cold-tail only; expert-hit delta within −4% (M4 fetch-wait counter as the tripwire). Falsifier: pool level logged per demote; park ms vs below-watermark correlation.
- **W11-2:** risk = capping swaps starves profile re-learn (F-271 hazard: poisoned profile = 21k swaps/boot, M1 −2.2x silently — control the learned profile in any A/B). Guard: swap count + hit-rate per round; decay-aware cap per the item. Falsifier: idle-gap length vs storm size vs next-turn prompt_ms.
- **W11-1:** risk = VRAM — free is **1,244 MiB** (M12); 207 resident graphs may OOM (vLLM RFC #20098 closed "strong no" on lazy capture for exactly OOM + perf-dip; sweep-over-lazy is the orthodox side). Guard: hot-set first, arena check per capture, prewarm2 hook in the same commit (law 11). Remediation: bucketing (pad-to-nearest) stays available but is optional at 207 layouts.
- **W8-2:** risk = window-boundary transition bugs (law 9: empty cohorts, captures while cohort in flight — F-233 precedent); fence the BADM-path transition. Falsifier (cheapest, law 4): count STOP+drain join events per session from the log before writing the command.
- **W8-1:** only after W17 measures residual multi-read queueing; risk of fighting the cadence waiver; compose, don't stack (law 4b).

## Cross-cutting

- **Composition of M1's 317–934 ms fixed overhead — do not double-claim:** reuse-scan 73→313 ms median (grows with ctx, 2–413 ms band) is **W15's** hash item; the 1,207–2,236 tok/s launch-bound small-read term (~870–934 ms on 1k+ reads) is **W19's**; park/restore is W11-2/3; only the join/promote slice (80–150 ms × join events) is W8-2. M25 kills the "idle ramp" version of the claim: prompt_ms flat 205–208 ms after 0/30/60 s idle.
- **Order:** W11-6 → W17/W8-7 → W11-3 → W11-2 → W11-1 → W8-2 → (W8-6 battery first among W8) → W8-1 residual → W11-4 scoped; W8-3/5 killed as perf items.
- **One-boot env bundle (ROADMAP order-of-attack already says so):** STRATA_PREFILL_CADENCE=1 (N=2) + PSTAT/BSTAT telemetry + STRATA_BATCH_DRAFT_DEPTH ramp — but cadence verdict rides on TTFT p50/p99, not ITL alone.

## Digest-vs-telemetry contradictions (resolved on the record)

1. **Sweep cost overstated 11x:** digest w11 priced "2,361 layouts ≈ 118 s" — that counts *events*; the log shows 3,322 events over **207 distinct layouts** → 2–8 s full / <1 s hot-set. Bucketing is optional, not a prerequisite.
2. **Two promote mechanisms, one label:** row fact "promote 12–17 ms" is the adapt tier-swap path; elastic materialize measured **0.9–17.4 ms, p50 2.6 (n=1,925)** — M23's cheap figure. Row must carry both; W8-2's win is NOT the materialize term.
3. **Demote park term grown:** anchors say 21.6–299 ms; current log (n=1,847) reads **p50 48.7 / p90 58.9 / max 807.9 ms** — the spare pool drains (0 chunks left observed at materialize), which is exactly item W11-3's needle. Anchor is stale-low.
4. **Admission has no needle on this traffic:** p99 ≤0.5 ms (M10) + M8 TTFT spread 0.03–0.62 s ⇒ W8-1/4 have no standing delay term solo; they are burst/multi-client scoped. Round-robin's real term is the graph-replay eviction (cadence's), not admission order.
5. **BYIELD fired 0/3 taken** (M23 + log): the short-prompt rescue never engages; the long-prompt interleave case is uncovered — cadence is the only live fix for M18's disease today.

## Open questions

- Join-event frequency per session (STOP+drain count) — prices W8-2 exactly; log format carries no single "join ms" line (cost is a composition of parts).
- Slot-wait >100 ms count under STRATA_SERVE_TELEMETRY — prices W8-4; zero over a session kills it.
- VRAM bytes per captured batch-window graph — sizes the hot-set sweep against the 1,244 MiB ceiling.
- Cadence N=2 effect on TTFT p99 under OUR chunking (vLLM measured TTFT better; chunkers differ — watch is already encoded).
- Whether W19's small-read path lands before or after W11 (ROADMAP says open only after W8/W11; M22's ~1.0–1.3 s fixed tax is its needle).

## Sources

- Code/telemetry (primary, pulled 2026-10-31): custom-iq3_s.log (materialize n=1,925; demote-park n=1,847; 3,322 captures/207 layouts; BYIELD 3; elastic boot lines); serve/server.py:325-339 (cadence shipped, OFF); launch-custom-iq3s-8081.sh (cadence + BATCH_DRAFT_DEPTH absent — row note verified).
- Campaign records: AGENTS.md laws 15/13/4b/9/11/12; .zcode/rich-tracking/strata-perf-roadmap.board.json rows w8-server-lane (7 items), w11-cold-jitter (6); campaign/MEASURED-opt-map-2026-10-27.md (M1–M25 cited); campaign/ROADMAP-perf-8moves.md §W8/§W11/§W15/§W17/§W19.
- Digests (secondary, upstream-sourced): .docs/research/digest/w8-server-admission-joins.md; w11-cold-jitter-warmup.md (both 2026-10-31).
- Upstream via digests: vLLM PR #44558 (merged 2026-06-16), vLLM RFC #20098 (closed not-planned), LMCache PR #4778 / issue #1175, SGLang breakable-graph buckets, Mooncake load_async.

## Methods note

Unproductive: direct grep for a "join … ms" log line (none exists — the 80–150 ms join cost is a composition of logged parts, not a logged term); live /metrics admission re-check impossible (engine boundary — M10's histogram is the record); restore-ms grep pattern matched nothing (restore appears inside demote lines only). Capture-count drift: row says 2,361 (map snapshot), log now 3,322 events — same mechanism, layouts are the invariant that matters.
