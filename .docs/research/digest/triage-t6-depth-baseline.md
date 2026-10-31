---
type: digest
id: triage-t6
created: 2026-10-31
status: evidence
topic: T6 triage — W15 depth persistence (6 items) + W0 baseline open items (6), priced against law 15's needle
scope: read-only triage, no code touched; prices from MEASURED-opt-map M3/M8/M16/M19-M21/M23/M25, board rows w15-depth/w0-baseline, F-374/F-375, digests w0/w15
---

# T6 — W15 + W0 priced against the needle (AGENTS.md law 15)

Needle: bar ≥ +4% on the targeted lane AND no lane or latency term beyond −4% (band = ledger M1 180-222 until w0-9 derives per-lane floors). Priced expectation inside the band on every lane and latency term = NOT RUN — compose (law 4b) or drop with pricing recorded. Remediation for any regression = scope/compose/rebuild (law 13).

## Verdict table

| Item | Verdict | Priced expectation | Conf |
|---|---|---|---|
| w15-1 elastic KV tier | KEEP (sequenced, capped) | conc @134K gap −40-48% (341-351→179-210), PCIe-contest share unproven; 262K residency 3.2 GiB vs vram_free ~1.2 GiB | Med |
| w15-2 TC decode scores | KEEP (one stages boot gates it) | scan ≥1 ms/window @134K ⇒ +7-16% solo depth | Med |
| w15-3 top-k widen/prune | DOWNGRADE → fold into w15-2 | tens of µs standalone (Quest RAFT 5-10 µs <128K): inside band | High |
| w15-4 cell prefetch | KEEP (after copy-stream look) | hides copy ms, adds PCIe traffic; ceiling = copy share of knee, unmeasured | Low |
| w15-5 reuse-scan hash | KEEP — first W15 lever | −240-306 ms median TTFT @99K (73→313; p90 546); byte-identical | High |
| w15-6 depth battery | KEEP — the gate | no W15 adoption without it; needs w0-7/8/9 | High |
| w0-4 prefill ladder 8K/32K | KEEP | prices W7/W14/W19 (+10-25% claims); one run | High |
| w0-5 pre-roadmap snapshot | KEEP | rollback insurance (law 7); trivial | High |
| w0-6 §Baseline table cells | KEEP | audit surface for every gate; trivial | High |
| w0-7 depth baselines, 3 legs | KEEP — first W0 execution | 2026-10-27/28 depth reads are 5-boot-confounded; bank moved to 262K config | High |
| w0-8 reuse baseline + WSTAT | KEEP | prices w15-5's A/B; separates ctx-linear from variance terms | High |
| w0-9 drift floor + temp/clock trace | KEEP — gate of gates | no ±4% verdict is honest without a derived floor (MLPerf 2%, Triton 10%) | High |

Nothing KILLs: every item prices above the bar somewhere or is a gate prerequisite. Two are sub-ms standalone and fold into other items' batteries (w15-3; part of w15-4's ceiling).

## Per-item pricing, risks, falsifiers

### w15-1 elastic KV resident tier — KEEP, sequenced last of the levers
- Needle: past-tier page misses × lane count contend with shadow-expert fetch on one PCIe (pcie-frac 0.35); conc fell −40-48% at 134K while solo fell −17% — the differential IS contention. If misses are ≥half it, a capped tier takes conc @134K +20-40%; bar clears if miss share ≥ ~10% of the gap. Solo @134K target 200 (+11-15%).
- Cost bound (T58 teeth): full 262K residency 3.2 GiB (1 KB/cell/layer × 12; the roadmap's 1.6 GiB was 131K-era) needs ~1,600 expert-slot demotions (19.6 GiB / 9,958 slots ≈ 2 MB/slot). F-375: reserve-1024 → vram_free 151-498 MiB → HARD STALL, 0 tokens/130.75 s, reverted; its +597 slots measured nowhere. F-271: poisoned expert tier = 6-10 ms GPU-reach waits, M1 −2.2×, hit rate still ~99% (telemetry looks fine; read the adapt line). Per-session pools ×6 = 9.6 GiB — dead on arrival; shared pool only.
- Guardrail: vram_free ≥ 1.5 GiB, routed entries % ≥ 95.5, standard-lane parity (law 2), M1 single-stream protection (law 4). Scope per law 4b: tier grows only toward the deepest ACTIVE conversation, shared pool, demotion capped by the VRAM floor.
- Falsifier (cheapest first): read the pool-allocation code (shared vs per-session — free); then one env boot `--kv-resident 65536` with vram/routed telemetry before any demotion design. Falsified if routed% drops ≥1.5 pts or VRAM < 1.5 GiB.

### w15-2 TC block scores on decode — KEEP, one telemetry boot gates it
- Needle: knee term +4.7-7.6 ms/window above ~80K (M20; p99 22-24→29.5-32.8). Scan = O(ctx/4) × 12 QSA layers (33.6K blocks @134K); TF32 kernel already in-tree (qsa_select.cu:173+, law 5 — wiring, not writing). At 4× on scan share S: save 0.75·S of a ~19 ms window → S=4 ms ⇒ solo 176→204 (+16%); bar clears at S ≥ ~1 ms (+7%).
- Risks: TF32 changes score precision → selection → outputs; law 10 canary (M3b acceptance) + MMLU 2x mandatory. Low-ctx parity: 8.2K blocks @32K must not lose.
- Falsifier: one 134K boot, read GPU-stages kernel-class ms — block_scores < 1 ms/window ⇒ NOT RUN, pricing recorded.

### w15-3 top-k widen/prune — DOWNGRADE, fold into w15-2's battery
- Single 1×1024 CTA over the score row ≈ tens of µs at 33.6K entries (Quest RAFT batched top-k: 5-10 µs <128K) — inside the drift band on every lane ⇒ law 15: NOT RUN standalone.
- Compose: exact multi-CTA ships with w15-2 (one battery). Pooled-of-pooled is approximate — NSA §6.1: heuristic selection showed low recall under native training; arm only if stages show top-k ≥ 1 ms/window AND recall validated vs exact selection.
- GQA union trap (NSA §2.1): per-head selections pay the UNION in KV traffic — sum scores across heads sharing KV BEFORE top-k, or the win silently erases at 6 lanes.

### w15-4 prev-window cell prefetch — KEEP, gated on a copy-stream look
- Needle: hides DMA under compute (competitor doctrine: vLLM async submit_load; HiCache best_effort + timeout; LMCache admission-window prefetch); ceiling = page-copy share of the knee, unmeasured. Batch DMA across 12 layers cuts per-page overhead at conc.
- Risk: prefetch traffic contends with expert fetch on the SAME PCIe (pcie-frac 0.35) — can regress the lane it targets (law 2). Adjacent-window selection overlap is asserted, not measured.
- Falsifier: nsys the copy stream at 134K — idle gaps ≥ copy share ⇒ GO; saturated ⇒ NOT RUN (adds traffic, not overlap).
- Does NOT fix M21's post-eviction whole-32768-block fallback (~5 s/turn at 100K) — that is reuse granularity, W11's shelf; this item may not claim it.

### w15-5 reuse-scan rolling hash — KEEP, first lever (t/s-per-effort champion)
- Needle: TTFT term on EVERY warm turn — median 73→313 ms (9K→99K), p90 546, max 663 (M23); ~40-60% of warm TTFT at 99K (M19/M25 1-token reads). Law 15: delays count like t/s. ≤5 ms @99K ⇒ −~300 ms TTFT, zero lane risk, byte-identical (hash gate + verify on hit).
- Digest upgrade: vLLM chained block hash (sha256(parent hash + block tokens)) — O(1)/block, entry-count-independent; beats a plain rolling hash and SGLang's O(prefix) radix. Maintain incrementally on park/append.
- Named choice the roadmap hides: verify-on-hit keeps ONE linear equal (the winner) — the ≤5 ms target implies trusting sha256 as identity (vLLM precedent, post collision-leak). Name it in the F-entry.
- Contradiction: M23 "roughly linear in ctx" vs M19's 2-413 ms variance at similar ctx — a non-ctx term (contention/cache state) shares the number. w0-8's WSTAT bands must separate them or the A/B credits the hash with a term it does not remove.
- Falsifier: prompt_ms decomposition, 5-run medians (MLPerf shape) — median @99K improvement < 4% of TTFT ⇒ NOT RUN (it will not be).

### w15-6 depth battery — KEEP, the gate, sequenced
- Not a lever: the adoption gate for all five above (law 2 all-lane parity at same depth). The 02:10-03:30 five-boot confound proves depth verdicts are drift-fragile without w0-7/8/9. Verify W1's controller shrink-T-at-depth inside it — W1 is itself a depth-persistence mechanism (free depth win if the cost EMA behaves).

### W0 open items — gate quality; all KEEP
- w0-9 is the law-15 prerequisite: the ±4% band must be a derived per-lane floor, not a ledger anecdote. Adopted shapes: MLPerf 5-run min + 2% materiality + 600 s DVFS floor; Triton max/min ≤10%; TensorRT doctrine — `nvidia-smi -q` before, `dmon -s pcu` during every battery. Our 180-222 band (~21% max/min) is normal floating-clock physics (TensorRT: "tens of percent"), not a bug; M17's ~6% derate is Blackwell hardware — do not chase.
- w0-7: every W15 verdict computes against these; current 134K/256K numbers are confounded-boot reads and the standing bank moved to 262K config (207.4/198.4/294.1) — baselines must match config and protocol, interleaved legs.
- w0-8 prices w15-5 and builds WSTAT; w0-4 prices the prefill row (W7/W14/W19's +10-25% are unpriced without it); w0-5/w0-6 cost minutes — run them first.

## Cross-cutting

- Order by depth t/s-per-effort: w0-9 → w0-7 → w0-8 → w0-5/6 (half a day; makes everything decidable) → w15-5 → w15-2(+3) → w15-4 → w15-1 → w15-6 gates each adoption.
- Composition: W1's controller + W6's multi-CTA resolve (same neighborhood as w15-1/4) + F-270 precedent — stack confirmed wins into one promotion so each chart step clears the noise band (law 12 incline).
- Contradictions flagged: (1) roadmap sizing 1.6 GiB is 131K-era; at `--max-context 262144` it is 3.2 GiB — math here uses 3.2. (2) M23 linearity vs M19 2-413 ms variance — WSTAT separates. (3) "≤5 ms @99K" vs verify-on-hit — name the hash-trust choice. (4) Acceptance RISES with ctx (0.773→0.89, M23): the depth slide is pure wall-time — depth arms get no acceptance tailwind and may not claim one.

## Sources
AGENTS.md laws 15/13/2/4/4b/12; ROADMAP-perf-8moves.md §Baseline/§W15/§W16; MEASURED-opt-map-2026-10-27.md M3/M8/M16/M19/M20/M21/M23/M25; board rows w15-depth + w0-baseline (verified open 2026-10-31); SVG-CUSTOM.md F-374/F-375 (T58 stall + shadow), F-271, F-270; digests w15-depth-persistence.md + w0-measurement-discipline.md (upstream: vLLM prefix-caching/kv_offload docs, SGLang hicache, Quest arXiv:2406.10774, NSA arXiv:2502.11089 §2.1/§6.1, LMCache arXiv:2510.09665, MLPerf rules/conf/audit, Triton perf_analyzer, TensorRT benchmarking); code anchors qsa_select.cu:173+/1003/1032-1038/2579-2580, kv_stream.cu:104-160/230, layer.cpp:575/793, conversation_cache.hpp:233-249/315-330.

Methods note: all directed reads hit (no unproductive searches); T58 stall verified at F-374/F-375, not just briefing; scan/topk ms shares are NOT directly measured anywhere in the record — the stages boot is the cheapest missing number on the row.
