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
| 7 | 2026-10-13 | R7 aux-cpus arm (F-176): baseline 173.7/153.2/194.6 → auto 171.4/146.6/192.1 → 16-23 176.5/149.3/194.8 (3-run means, matched same-night windows) | — | — | — | ~81% | **REVERT** — no arm beats baseline past noise (±4%); draft path, not host preemption, sets the rate. R7c IRQ steering still open (sudo). Evening drift vs anchor noted: re-anchor before next adoption claim. |
| 5 | 2026-10-14 | R5 GUMBEL, full factorial (F-177): leg A 6v6 standing+GUMBEL 171.8/145.4/187.2 vs 168.7/145.6/191.6; leg B rescue-off 3v3: GUMBEL +1.4pts on 58.5% raw | — | — | — | 81.0 vs 79.1 | **REVERT** — mechanism fires (+1.5-2pts) but min-p-filtered offered set leaves no artifact mass (upstream +9% was unfiltered). DISCOVERY: rescue quantified +20.6pts accept / +8.2% M1 → R5b tau sweep queued (MMLU-gated). |
| 10 | 2026-10-14 | **MERGE upstream v0.1.44** (F-180): 557-line mod patch 3-way onto upstream tree, 2 hand resolutions; COMMIT_BATCH+race fix banked | 195.2 | 160.4 | 211.0 | 80.0% | **ADOPT w/ watch** — perf neutral (Δ −0.6..+0.7%), best-ever single readings (215.4 conc / 200.1 M1), MMLU 3v3 = −1 question (92.86 vs 94.29, full overlap). Rollback: engine/strata.0.1.43.1-local-backup. WATCH: re-gate MMLU next round; 3-gate mean <92 escalates. |
| 14 | 2026-10-14 | **F-31 ADOPTED: STRATA_BATCH_MTP=1** (coupled slot drafts; wiring complete in merged tree, zero code) | 175.1 | **162.8** | 193.7 | 79.8% | **ADOPT** — pair **+11.5%** vs same-hour control (146.0), 6-conc +0.5%, M1 −2.2% (noise, serial untouched), **MMLU 97.14±1.99 = campaign-best gate**; slot accept 60-67%; arm live in BOTH 131k standing + 262k real-use launchers (F-184) |

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

## NEXT ROUNDS QUEUE v4 (re-ranked 2026-10-14 ~14:30 by the six-leg deep scan, F-187)

THE ROUND-9 THESIS stands: the engine is EXECUTION-bound — ~1,800 graph nodes/window at ~9.5 µs
spacing carrying ~2.3 µs of bytes each; 24-26% of GDDR7 peak; ~2.9x honest ceiling without
touching bytes. Node spacing before bytes; acceptance gains multiply on top (rows/s ∝ T).
Supersedes the F-179 ranking and F-186 v3 ordering where they conflict. Banked/closed since v3:
R9 overlap-unblock REJECTED (F-185), commit double-buffer BANKED (STRATA_COMMIT_BATCH default-on,
verify.cpp:2039), batch 6 ADOPTED (F-185, 131k lanes only — 262k driver still parallel:4, R0).

**R0 — hygiene (zero risk):** dead arms DELETED from both live launchers (F-187: ADAPT_OVERLAP
refused while DMA_BATCH=1, KEEP_MAPPING dead on Linux). Open: 262k driver parallel:4→6 A/B
(watch per-slot KV/conv-cache + M6b); logrotate drop-in for custom-iq3_s*.log.
**R8 — GPU state (USER SUDO, NOW — variance first):** persistence Disabled + clocks unlocked
measured (F-187) — the likeliest re-anchor-v2 drift mechanism. Block:
`sudo systemctl enable --now nvidia-persistenced; sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc 2835;
sudo nvidia-smi -lmc 14001` (step -lgc to 2745 if `dmon -s pc` shows sw_power/sw_thermal; optional
`-pl 600`; `-rgc`/`-rmc` revert). Then RE-ANCHOR v3 — all means reference post-lock conditions.
**Tier 0 — env A/Bs (after R8):**
- R10b: diagnostic boot VERIFY_PROFILE=1 + DECODE_TIMING=1 + VERIFY_NODES=1 (one boot; sets the
  code order; not perf-representative — prof stamps, df_branch off).
- R10c: PDL STRATA_DF_PDL=1 — corroborated 3 ways (TRT-LLM Blackwell-global; LMSYS Ling-3.0-flash
  bitwise-safe batch-1 MoE+spec; SGLang HPC-ops). In-tree + parity-tested, G==1 windows.
  3×3 + 2000-tok sustained gate + VERIFY_NODES programmatic-edges>0. [+3-8% class]
- R10f: v0.1.44 opt-ins never tested here — STRATA_PIN_BY_NEED=1 + --adapt-async 1 (vendor
  +4.2-5.2% decode on 5070). MMLU-gated (residency stateful).
- R10d: ROUTE_RESIDENT margin sweep 0.5→0.75 (1.0 stays closed, F-69). MMLU-gated.
- R10e: L2_SETASIDE_MIB 60→72 — 60 was the ask, not the cap (112 MiB span, hitRatio 0.536).
- R10a: QFUSE=1 boot-rider ONLY (−48 nodes) — CAUTION: disables T=1 self-commit (verify.cpp:879);
  never pair with a BATCH_MTP adoption boot; low prior (F-67).
**Spec path — the M1 margin levers (all distribution-neutral, F-187 #3):**
- Diagnostics first: STRATA_SPEC_DEPTH=1 boot + one spec-8 boot (--spec 8 --mtp-max-t 8).
  Floor is the binder (F-159: 96/93/90% at depths 1/2/3, chains stop ≤3 at floor 0.5).
- Floor stair 0.5/0.5/0.30… (generate.cpp:12796-12798) → M1 +2-6% class; offer-side only.
- Deeper slot chains: set_max_drafts(1)→3 + draft(2,…)→4 + matched-prefix accept (F-187 map).
  Pair +4-8% class; 6-conc is the guard.
- AdaEDL entropy early-stop (proven lossless) before the F-186 open-lane arbiter.
- Tier-1.5: rejection-sampled slot accept = distribution-safe F-30b substitute. MMLU-gate.
**Tier 1 — code (order set by R10b):** partial device plan (s_bad abort removal; full data-path
map in F-187; STRATA_VERIFY_DEVICE_PLAN_PART=1; exact arithmetic; 9-15% class) → fenced admission
(sync map F-187; measured 54.8-58.3 → ~15 ms) → window width 8→12 (F-187 VRAM math: +1.9 GiB of
21.7 free). Config rider: kv-resident 65k for the M6b/depth lane (+415 MiB).
**Tier 2 — research→code:** v3 items stand (FlashInfer ports, ECHO budget, DuetServe look-ahead,
layered prefill, sentinel-slot graphs, QKV-multistream) + NEW: SP-MoE draft-gated prefetch
(arXiv 2510.10302), LMSYS bitwise-neutral fusion recipes (SwiGLU-in-GEMM epilogue) opening the
node-density program, conditional nodes / metadata-glue graph capture for remaining host syncs.
**Tier 3 — training:** FastMTP recursive re-tune (unchanged).
**Upstream/env:** v0.1.44.1 = hotfix only (merge for hygiene); hold driver 615.71.09; stay on
kernel 7.2.9; numa_balancing=0 small probe (open bar).
**Demoted:** v3 demotions stand + QFUSE-as-headline, overlap-unblock (F-185), hc→Q8 byte cuts.
**Open bar items:** M5 mimic port, M6 depth + short-class matched anchors, NTFS master sync
(F-172..F-187), log rotation, numa_balancing probe.

## RE-ANCHOR v2 (2026-10-14 ~00:15, fresh boot, current conditions — the A/B reference going forward)

| metric | MEAN (3 probes / 2 gates) | locked anchor | note |
|---|---|---|---|
| M1 single | **176.3** | 190.3 | −7.3%; drift is global, not GPU throttle (42 °C, clean) |
| M2 pair | **145.3** | 159.4 | −8.8% |
| M2 6-conc | **190.7** | 203.7 | −6.4% |
| M3b | **81.3%** | 80.5% | +0.8 pt |
| MMLU-Pro | **93.6** (92.86 / 94.29) | 95.71 | ≈1.5 q / 70 — within noise, recipe unchanged |

The locked anchor stays the BAR reference; re-anchor v2 is the matched-window reference for
adoption A/Bs. Candidate drift source: the learned expert profile (re-saved by probe traffic every
30 rounds) — check by restoring data/expert-profile.bin over learned in a spare boot if the gap
persists. Sources: campaign/results/reanchor2-{1,2,3}.txt, mmlu-reanchor2-{a,b}.json.

