---
type: digest
date: 2026-11-09
title: "Residency regression study — did the expert-cache hit rate or routing behavior change since Nov-6?"
claim_under_test: "The GSQ IQ3_S lane loss (-8..-15% solo vs the Nov-6 reads) is caused by a change in expert residency/routing behavior since Nov-6 (learned heat-profile drift, cache-count move, shadow-store tier)."
verdict: RESIDENCY EXONERATED — hit rate did not drop (96.39% era1 vs 96.18% today weighted; worst like-for-like bucket -0.71pp); the CPU-miss path got CHEAPER (3.54%→3.29% of routed, 2.66→2.23 ms/window on matched legs); the learned profile drifted hard but is save-only and never loaded (all 292 boots fill from the static profile); the -8..-15% sits in the GPU run residual (+2.0 ms/window solo-class) and pair-lane GPU-reach wait (+1.8 ms), zero of it in residency.
evidence: logs/custom-iq3_s.log (292 boots, boot-segmented), campaign/results/gsq-clean-r*.out, gsq-restore-r*.out, t22-stock-r1.out, hostmem-definitive-iq3s-r1.out, data/expert-profile.bin vs data/expert-profile-learned.bin (STRP v1 decode), campaign/tracker/lanes.csv rows 123-131; sibling digest 2026-11-09-regression-bank-provenance.md owns the baseline provenance question.
---

# Residency/routing regression study (Nov-6 vs Nov-8 vs Nov-9)

Method: `logs/custom-iq3_s.log` (662,723 lines) segmented by the 292 `[strata] <ts> engine started` boot markers. Every boot in all three eras runs a byte-identical command (same pack `/home/svg/LLM/data/packs/iq3_s`, same GSQ-RCO native shards, `--max-context 262144`, `--batch 6`, 0.95 GiB slot sessions, 0.97 GiB elastic arenas, `--expert-profile data/expert-profile.bin --expert-profile-save data/expert-profile-learned.bin --expert-profile-save-every 30 --resident-experts`). Eras: **E1** = Nov-6+Nov-7 (48 boots, 3,044 windows), **E2** = Nov-8 (4 boots, 175 windows), **E3** = Nov-9 (5 boots, 251 windows). Like-for-like comparisons bucket the per-window batch telemetry by avg rows/window (≈3.0-3.5 = pair-class legs, ≈4.5-5.0 = solo/conc-class legs) because the eras carried different leg mixes.

## 1. Hit rate per era — the direct residency metric did not move

Per-window `decode expert cache hit rate: NN% (H hits / L lookups)` lines, summed over the era:

| era | boots | windows | hit mean (per-window) | hit weighted (hits/lookups) | routed VRAM% | routed CPU% | CPU-expert ms/window | ms/window | rows/s |
|---|---|---|---|---|---|---|---|---|---|
| E1 Nov-6/7 | 48 | 3,044 | 95.78% | **96.39%** (224.67M/233.09M) | 95.73% | 3.54% | 1.05 | 16.16 | 208.8 |
| E2 Nov-8 | 4 | 175 | 95.71% | **97.50%** (7.61M/7.80M) | 96.78% | 2.91% | 0.89 | 18.77 | 165.6 |
| E3 Nov-9 | 5 | 251 | 95.61% | **96.18%** (25.15M/26.15M) | 96.05% | 3.29% | 0.99 | 17.01 | 188.2 |

E3 vs E1: **-0.21pp weighted** (noise; the raw means invert because the leg mix shifted). Routed VRAM share *improved* +0.32pp; routed CPU share *improved* -0.25pp.

Like-for-like battery legs (window-weighted by avg-rows bucket):

| rows bucket | era | n | VRAM hit% | ms/window | GPU-reach wait | CPU stage | run residual (run-wait-cpu) | rows/s |
|---|---|---|---|---|---|---|---|---|
| ~4.8 (solo/conc legs) | E1 | 41 | 95.91 | 14.08 | 0.48 | 2.29 | 9.65 | 257.3 |
| | E2 | 3 | 96.47 | 18.11 | 0.51 | 2.07 | 9.20 | 192.8 |
| | E3 | 2 | 95.20 | 15.87 | 0.45 | 2.28 | **11.65** | 236.0 |
| ~4.6 | E1 | 34 | 95.46 | 15.04 | 1.24 | 2.72 | 9.12 | 204.5 |
| | E3 | 4 | 95.51 | 16.75 | 1.17 | 2.42 | **11.24** | 197.7 |
| ~3.2 (pair legs) | E1 | 44 | 96.68 | 15.62 | 9.86 | 2.87 | 1.06 | 190.9 |
| | E3 | 6 | **97.03** | 16.98 | **11.65** | 2.35 | 1.16 | 170.5 |

Bucket hit deltas: +0.35pp (pair), +0.05pp (4.6), **-0.71pp worst** (4.8) — mixed sign, sub-1pp. Matched single-leg quotes (pair class, E1 13:51 boot vs E3 10:27 boot): CPU entries/window 54.3 → **40.0** (fewer misses today), CPU-expert per-layer-window 1.00 → **0.79 ms**, VRAM hit 96.7 → **97.7%**, while GPU-reach wait 8.60 → **12.43 ms**. The .out per-request lines corroborate: gsq-clean-r1/r2 mean 96.07/96.31% (steady-state legs 97-99%), same warm-up ramp shape as the Nov-6-era stock-binary legs.

## 2. Boot lines per era

| era | cache slots (GiB) typical | pre-filled | resident RAM complement | MemAvailable at boot | profile loads | profile saves | adaptive swaps/window |
|---|---|---|---|---|---|---|---|
| E1 Nov-6/7 | 9,856-10,079 (18.85-19.12) | N/N, 100% | 37.5-37.95 GiB full | 75-130 GiB | static only | 19 exit + 6 periodic | ~490 |
| E2 Nov-8 | 9,836-9,853 (18.66-18.69) | N/N, 100% | boots 2-4 **capped 17.39/19.37/19.80 GiB** (budget bug, fixed e8484c36) | 46-83 GiB | static only | 3 exit | ~240 |
| E3 Nov-9 | 9,863-9,903 (18.71-18.79) | N/N, 100% | 37.85-37.93 GiB full | 46-87 GiB | static only | 4 exit (no periodic — battery legs < 30 windows) | ~240 |

Cache-count attribution (the operator's "9997 → 9872" move, packs disentangled): the 9997-experts/18.97 GiB line is the **Nov-7 19:39:10 boot**, not Nov-8; the Nov-8 engine boots filled 9,836-9,853 (18.66-18.69 GiB; `hostmem-definitive-iq3s-r1.out` = 9,853/18.69); Nov-9 clean = 9,872/18.73, restore = 9,903/18.79. All eras solve slots from free VRAM at the same auto line ("20.7-21.1 GiB free -> 7,525-7,648 slots" before arena reclaim); the ±130-slot wobble (±0.25 GiB) tracks desktop VRAM state, not any config or profile change. The 125-slot delta (ranks ~9,872-9,997 of the 24,576-pair ranking) is deep-tail routing mass — the routed CPU share shows it: unchanged at 2.1-3.5%. `the prompt path's lend region: N of M slots` is **not emitted by this build** (0 hits in the log and all .out files; the lend region exists in code at `src/core/expert_source.cpp:229`) — named boundary.

## 3. data/expert-profile-learned.bin — drifted hard, but never a solver input

- 196,632 bytes = 24-byte STRP v1 header + 24,576 (u16 layer, u16 expert) ranked pairs + 48x512 u32 table. Not git-tracked; mtime Nov-9 10:40:39 (exit save of the last boot), birth = same write (inode recreated).
- Drift vs the static `expert-profile.bin` is **real and large**: top-9,872 overlap 43.23%, top-9,997 overlap 43.55%; static's top-100 experts sit at learned-rank mean 4,907 (median 4,683). Two days of battery traffic did reshape the adaptive tier's heat ranking.
- **But no boot ever reads it.** All 292 `engine started` lines carry `--expert-profile data/expert-profile.bin` (static, Oct-14, unchanged). The learned file is written (periodic every 30 windows / at exit: 25 exit + 9 periodic saves across the three eras) and never loaded — a save-only sink. The boot-time cache composition is therefore identical across eras (top-N of the same static ranking, N = slot count); profile drift is structurally unable to explain any boot-to-boot behavior change.

## 4. Shadow store arm per era (tier map)

| era | boots | tier | wording |
|---|---|---|---|
| E1 Nov-6/7 | all 48 | tmpfs `/dev/shm/strata-shadow-q2_0` | plain mapped store, no memfd wording |
| E2 Nov-8 | 12:58, 13:07 | canonical pack `/home/svg/LLM/data/packs/q2_0` | "2 MiB pool pages (memfd, populated in 3.4-3.5 s)" — memfd wording first appears Nov-8 |
| E2 Nov-8 | 13:10, 13:12 | tmpfs `/dev/shm/strata-shadow-q2_0` | memfd pool, populated 5.5 s |
| E3 Nov-9 | 10:22-10:30 (clean) | tmpfs `/dev/shm/strata-shadow-q2_0` | memfd pool, 5.7 s |
| E3 Nov-9 | 10:38, 10:41 (restore) | canonical pack `/home/svg/LLM/data/packs/q2_0` | memfd pool, 3.4 s |

The shadow tier prices only the CPU-miss compute source (the line's own claim: "hits, the PCIe share and the prompt path do not change"). Consistent with that: the canonical-tier restore legs read lw-CPU 1.23 ms vs clean 0.87 and CPU% 4.06 vs 2.89 — a real but ~0.4 ms/window miss-path tax, and the matched-leg quote above shows the era-wide miss path got *cheaper* anyway.

## 5. Verdict — no residency mechanism; the cost is GPU-side

1. **Hit rate exonerated.** Era weighted hit 96.39% (E1) vs 96.18% (E3), -0.21pp; worst like-for-like bucket -0.71pp; the pair bucket *improved* +0.35pp. Routed CPU share 3.54% → 3.29%.
2. **The counterfactual is quantified and did not happen.** A hit drop of the hypothesized size (e.g. 96.5% → 93.1%, -3.4pp) at ~1,500 routed entries/window and ~150 µs per CPU miss ≈ 7.7 ms/window — roughly half the window, a ~30-45% lane cost. The measured worst delta (-0.71pp) would bound at ~1.6 ms/window *if* every marginal miss cost 150 µs — and the measured CPU-expert stage contradicts even that: flat to down (2.29→2.28 ms solo-class; 2.66→2.23 ms matched pair legs; era CPU-expert ms/window 1.05→0.99).
3. **Where the -8..-15% actually sits:** solo/conc-class legs grew the run residual (run − GPU-reach wait − CPU stage) from 9.65 to 11.65 ms/window (**+2.0 ms, +14-21% of the E1 residual**) at flat hit% and flat wait; pair-class legs grew GPU-reach wait 9.86 → 11.65 ms (+1.8 ms). E2 (Nov-8) additionally carried the adaptive-tier wait spike (adapt wait 2.7-6.0 ms) and the complement capped at 17.4-19.8/37.9 GiB — the budget bug e8484c36 fixed, not today's mechanism. Both live cost movements are GPU device-side (window run/scheduling state), consistent with the sibling digest's conclusion (aged/degraded-box mode, -7..-17% in-family for same-binary A/A re-banks; baseline composite inflated).
4. **Residency improvements available if ever needed** (not the regression): learned profile never loaded (wire it as a load A/B if desired), restore legs' canonical shadow tier costs ~+0.4 ms/window on the miss path vs tmpfs.

Boundaries named: the prompt-path lend-region telemetry is absent from this build; Nov-6 raw probe legs were never preserved (sibling digest gap 1), so E1 hit-rate comparisons rest on the shared engine-log telemetry, which both eras' probes read from the same file and instrument.
