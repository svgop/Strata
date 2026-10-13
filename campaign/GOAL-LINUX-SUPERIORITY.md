# GOAL — LINUX SUPERIORITY OVER THE WINDOWS LANE (the Ultimatum)

Started: 2026-10-13 (operator directive, mid F-172 session). Owner: svg + ZCode.

## The bar (the goal does not stop until this is true)

Every metric of the six-metric suite on the Linux lane (~/LLM/strata-custom-2, port 8081) must beat
the Windows standing lane by AT LEAST 30% — with one physical exception, recorded honestly:

| metric | Windows standing (PERF-CHART.md, sourced) | Linux bar |
|---|---|---|
| M1 single-stream 2000 tok | 145.8 t/s | **≥ 189.5 t/s** |
| M2 pair aggregate | 108.8 t/s | **≥ 141.4 t/s** |
| M2 6-conc aggregate | 140.2 t/s | **≥ 182.3 t/s** |
| short class (220 tok) | ~124-146 t/s class | **≥ +30% vs matched probe** |
| M6a depth prefill (64k) | 1490.7 (contended) / 7590.6 best | **≥ +30% vs matched run** |
| M6b depth decode (64k) | 91.6 (contended) / 145.4 fresh | **≥ +30% vs matched run** |
| M3c banked tokens/window | 2.18-2.41 standing | **≥ 3.0** (this is the M1 lever) |
| M3 accepted/s | (derived) | **≥ +30% vs matched probe** |
| M4 MMLU-Pro 70q | 95.7 | **≥ 95.7** (bounded at 100; hold + no regression — 30% over is physically impossible) |
| M5 mimic 64 | 100% | **= 100%** (ceiling; hold) |
| M3b acceptance rate | 90.1-90.9% | **≥ 90.1%** (bounded; hold + push toward the 94.9 best) |

## The rules (inherited laws, kept)

1. **/mnt/DATA is FROZEN** — a snapshot of before. All work local: ~/LLM/strata-custom-2,
   ~/LLM/Custom-data. Copying FROM /mnt/DATA (mtp, scripts, configs) is allowed; writing TO it is not.
2. **The Proof Law**: every claim carries its probe, its window, its comparator, its source file in
   campaign/results/. A row without those is not a row.
3. **The No-Regression Law**: nothing is adopted that bends M4/M5 or degrades another metric past noise.
4. **One variable per leg.** Matched windows where possible (same boot or back-to-back boots).
5. **The journal decides**: every round lands as an F-entry in SVG-CUSTOM.md (local copy; NTFS master
   frozen).

## The round protocol (every round, going deeper each time)

1. **Study leg A — the code** (subagent): read the engine source deeply; the round's hypothesis must
   cite file:line. Go deeper than the previous round (kernels, tier logic, draft path, DMA, scheduler).
2. **Study leg B — the world** (subagent): arXiv (speculative decoding, MoE serving, memory tiers),
   upstream git (llama.cpp/strata upstreams), Fedora/kernel/CUDA 13.x releases, driver notes, news.
   Deliver ranked, applicable levers with expected magnitude.
3. **Experiment**: one variable, A/B, probe (campaign/probe-lane.py), quality gate when distribution-
   bending.
4. **Record**: results file + F-entry + updated scorecard in this file.

Target: at least 90 rounds. The goal is not done until the bar is met. Sessions resume this file.

## SCORECARD (updated every round; sources in campaign/results/)

| round | date | change | M1 | M2 pair | M2 6-conc | accept | verdict |
|---|---|---|---|---|---|---|---|
| 0 | 2026-10-13 | Linux port, standing recipe (r17 draft) | 109.1 | 132.7 | 162.3 | 63% | baseline (Windows: 145.8/108.8/140.2/90%) |
| 0b | 2026-10-13 | spec-min-p 0.5 (r17) | 108.5 | 134.5 | 175.2 | 50% | reject (rate fell, M1 flat) |
| 1 | 2026-10-13 | **--mtp → stock rt** (r17 = 57%-plateau distilled head per F-104/114/115/116 — never the 145.8 source) | 154.9 | 132.8 | 168.8 | 88-92% | **ADOPT** — reproduces Windows acceptance exactly |
| 2 | 2026-10-13 | spec-min-p 0.5 (stock draft) | 167.1 | 144.7 | 187.6 | 87% | **ADOPT** — pair bar met, conc bar met |
| 3 | 2026-10-13 | --spec 6 | 181.5 | 144.9 | 179.1 | 79% | **ADOPT for M1** (+8.6%), conc dipped 187.6→179.1 — watch |
| 4 | 2026-10-13 | + STRATA_ROUTE_RESIDENT=0.5, STRATA_ROUTE_RESIDENT_MTP=1 (battery arms missing from the port; F-72) | **196.1** | **158.8** | **212.8** | 79.4% | **BARS MET: M1 +34.5%, pair +46%, 6-conc +52%; M4 MMLU 95.71 = Windows 95.7 (gate HELD, mmlu-linux-r4.json). Banked 2.54/window vs their 2.18-2.41. Rate 79.4% is the depth-for-rate trade (diagnostic per the 2026-10-12 amendment). M5 mimic + M6 depth + short-class anchors still open. |

## Round-1 study findings (both agents, 2026-10-13)

CODE (agent): rt-r17 = distilled head, 55.7-57.4% plateau (their own F-104/114/115/116); stock rt = 85-90%
class; draft_offered Y = min-p-filtered window rows; accepted X includes typical-acceptance rescues;
--spec gets rewritten by suffix_draft (spec 6 → S_mtp effective 4 + 2 suffix rows); no RNG/OS numerics in
the draft path (bit-deterministic Philox, position-keyed); Windows battery had ROUTE_RESIDENT arms the
port lacked (fixed in round 4); STRATA_SPEC_MIN_TEMP exists (argmax drafting for lukewarm requests).
RESEARCH (agent): STRATA_SPEC_GUMBEL=1 (+9% acceptance upstream-measured, lossless); CPU isolation/pinning
7.6-20% on desktop-churned boxes (pin to V-cache CCD, SCHED_BATCH, IRQ steering); GPU clock lock + nvidia-smi
-pm 1 (2-5% + determinism; needs sudo); THP/madvise for the 17GiB pinned arena (+2-6% swap-heavy); CUDA 13.x
MMQ-vs-cuBLAS audit + CUDA 12.8 A/B (Blackwell trap, up to 1.4x); L2 persisting window (96MB GB202, +~10%);
green contexts for draft/DMA overlap; adaptive draft length (AdaEDL/SpecDec++ class, +10-25% code-level);
public ceiling for this model class on 5090 ≈ 100-120 t/s — the lane is at 196, already ~65% past public.


## THE ANCHOR (locked 2026-10-13 night; the R4 recipe, 3 full-protocol probes + 2 MMLU gates, fresh boot)

| metric | run1 | run2 | run3 | MEAN | Windows | vs Windows | bar status |
|---|---|---|---|---|---|---|---|
| M1 single 2000 tok | 183.6 | 199.3 | 188.1 | **190.3** | 145.8 | **+30.6%** | at bar (+0.4% over 189.5; run noise ±4%) — needs margin |
| M2 pair | 150.9 | 166.0 | 161.4 | **159.4** | 108.8 | **+46.5%** | MET with margin |
| M2 6-conc | 194.1 | 214.0 | 203.0 | **203.7** | 140.2 | **+45.3%** | MET with margin |
| M3b rate | 80.5 | 81.6 | 79.5 | **80.5%** | 90.1 | below (depth-for-rate trade; diagnostic only per the 2026-10-12 amendment) | open |
| M4 MMLU-Pro | 95.71 | 95.71 | — | **95.71 ±2.42** | 95.7 | held | MET (=) |
| short class (post-fresh) | 177.8 | 191.0 | 187.3 | **185.4** | ~124-146 | +27-49% | matched-anchor pending |

**THE LOCK LAW from here:** every future round re-runs the full probe (3x when within noise of a bar)
plus the MMLU gate when distribution-bending, and must not regress any MEAN by more than run noise
(±4% M1-class). Gains are claimed only on means. Sources: campaign/results/anchor-run{1,2,3}.txt,
mmlu-linux-r4.json, mmlu-anchor.json.

## NEXT ROUNDS QUEUE (deepest-first protocol continues)

- R5: STRATA_SPEC_GUMBEL=1 A/B → M1 + acceptance; gate if bending.
- R6: MMLU gate result → adopt/reject round 4. Port/locate mimic (M5). 
- R7: CPU affinity for the engine (V-cache CCD taskset + SCHED_BATCH) — cgroup cpuset, no isolcpus.
- R8: GPU clocks locked + persistence mode (sudo, user step) — determinism pass, re-anchor all bars.
- R9: THP madvise on the pinned arena; NVMe readahead down for random expert fetch.
- R10: STRATA_SPEC_MIN_TEMP sweep; spec/min-p joint re-tune at the new acceptance level.
- R11: CUDA build audit (MMQ path on sm_120; CUDA 12.8 A/B build).
- R12+: code-level — adaptive draft-length governor, L2 persisting window, green contexts.
- M6 depth metrics (64k prefill/decode) + short-class matched probes still to be measured for the bar table.

