---
type: research-study
date: 2026-11-18
subject: cpu-exp doubling time series — GSQ lane anatomy across every battery in the repo; onset pinned to the 385937a4d rollback, not the 945e747d0 estate move
repo: /home/svg/LLM/strata-custom-2 (HEAD 2fa29b771)
status: complete
---

# cpu-exp time series (2026-11-18)

## 0. Verdict first

The cpu-exp term doubled **at the rollback, not at the estate move**. Last battery with cpu-exp ~2.0: the GSQ POST-REBOOT WARM BENCH, boot `2026-11-10 13:37:59` (mean 2.01, median 2.00, range 1.80-2.24 ms/window). First battery with cpu-exp ~4+: **t66-warm-recheck, boots `2026-11-18 17:49/17:52/17:54`** (warm-lane 4.38/4.93/4.23). **No battery exists between them** — zero GSQ boots Nov 10 16:47 through Nov 18 13:48 (engine-log gap), and the estate-move boots produced zero decode flushes (armed boots NaN'd at the first window; disarmed boots were config-validation). The doubling is the rollback's disarm of `STRATA_SHADOW_EXPERTS`: per-entry CPU-miss cost went 46 → 93 μs at constant entry count, hit, and accept — the 1.93x byte ratio between the Q2_0 shadow tier (1,382,400 B/expert, tmpfs) and the canonical IQ3_S blob (2.66 MB, ntfs3 mmap). The bank-era 219.6/210.4/290.3 was the shadow-armed regime; the ~20% GSQ regression is the price of the NaN rollback.

## 1. Method

- Anatomy source: the engine's own per-flush line `strata batch: N windows, avg R rows, W ms/window = run X (CUDA0 GPU-reach wait G + CPU experts C) ... routed entries per window: VRAM ... (hit%) ... rows/s ... slot drafts accepted ... (accept%)`, quoted into probe output by `campaign/probe-lane.py` since 2026-11-10 (attribution block) and read raw from the live engine log `logs/custom-iq3_s.log` (132,311 lines, boot-stamped, spans 2026-11-09 16:49 → 2026-11-18 18:12).
- Segmentation: 48 `engine started:` markers; per boot, all flush lines parsed (extractor: `/tmp/boot_anatomy.py`, pattern set validated against `campaign/probe-lane.py` regexes).
- Battery labels: bound to boots via `campaign/results/*.log` LEG timestamps and `.out` mtimes.
- Bank trio location: the exact `219.6/210.4/290.3` appears nowhere in results or tracker files; it entered the repo in `campaign/batteries/t66-warm-recheck.sh` (commit `1cdd2851b`, 2026-11-18) as the operator's warm multi-leg anchor of the Nov 10 bank battery. That battery's probe-lane stdout was not captured — `postreboot-warm-1.out` is the raw engine console only, and `campaign/tracker/lanes.csv` rows for it carry tps=0 (unbooked). Engine-level corroboration below.

## 2. Time series (per boot; median over the boot's flushes, all lanes mixed)

| boot (UTC-2) | battery / leg | shadow | flushes | rows | window | gpu-wait | cpu-exp | hit % | accept % | rows/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 11-09 16:49 | gsq-arms-screen ctl | armed tmpfs | 7 | 4.82 | 14.87 | 0.75 | 2.21 | 96.6 | 63.6 | 196.8 |
| 11-09 17:19 | gsq-arms-screen | armed tmpfs | 7 | 4.66 | 14.82 | 0.65 | 2.16 | 96.2 | 62.1 | 202.8 |
| 11-09 17:23 | gsq-condring-ab cr1 (slow arm) | armed tmpfs | 7 | 4.74 | 38.89 | 0.68 | 2.14 | 94.6 | 62.3 | 113.7 |
| 11-09 17:28 | gsq-condring-ab ctl | armed tmpfs | 7 | 4.76 | 15.07 | 0.94 | 2.35 | 96.2 | 63.6 | 201.6 |
| 11-09 17:32 | gsq-condring-ab cr2 (slow arm) | armed tmpfs | 7 | 4.71 | 38.95 | 1.24 | 1.98 | 94.4 | 63.3 | 114.0 |
| 11-09 17:37 | gsq-condring-ab ctl | armed tmpfs | 7 | 4.81 | 16.16 | 0.96 | 2.09 | 96.6 | 63.8 | 184.1 |
| 11-09 17:41 | gsq-condring-ab cr3 (slow arm) | armed tmpfs | 7 | 4.68 | 39.81 | 0.69 | 2.15 | 94.1 | 61.5 | 112.4 |
| 11-09 17:47 | gsq-quiet-record-hour qr-1 | armed tmpfs | 7 | 4.81 | 15.78 | 1.26 | 2.14 | 96.6 | 64.5 | 195.6 |
| 11-09 17:51 | qr-2 | armed tmpfs | 7 | 4.74 | 15.73 | 0.86 | 2.09 | 96.6 | 64.0 | 191.8 |
| 11-09 17:55 | qr-3 | armed tmpfs | 7 | 4.78 | 14.87 | 1.25 | 2.18 | 96.4 | 63.1 | 189.5 |
| 11-09 17:59 | qr-4 | armed tmpfs | 7 | 4.71 | 14.97 | 1.19 | 2.04 | 96.3 | 62.2 | 192.5 |
| 11-09 18:03 | qr-5 | armed tmpfs | 7 | 4.69 | 14.97 | 1.12 | 2.10 | 96.4 | 63.3 | 199.6 |
| 11-09 18:42 | kernel-aa-7215 kaa | armed tmpfs | 7 | 4.76 | 15.68 | 1.11 | 2.10 | 96.9 | 63.3 | 184.6 |
| 11-09 18:48 | gsq-v6-verdict v6-1 | armed tmpfs | 7 | 4.74 | 13.85 | 1.12 | 2.40 | 96.2 | 64.1 | 239.2 |
| 11-09 18:52 | v6-2 | armed tmpfs | 7 | 4.81 | 13.52 | 0.66 | 2.15 | 96.5 | 63.4 | 218.5 |
| 11-09 18:56 | v6-3 | armed tmpfs | 7 | 4.71 | 13.59 | 1.02 | 2.08 | 96.7 | 62.4 | 206.6 |
| 11-09 19:01-19:24 | gsq-pinning-ab pin legs | armed tmpfs | 7-8/boot | 4.63-4.79 | 13.16-14.15 | 1.13-4.99 | 1.86-2.36 | 96.4-96.7 | 62.3-63.9 | 217.9-229.0 |
| 11-09 20:14/20:22 | evening tails | armed tmpfs | 3 | 4.72 | 14.29/13.01 | 1.27/1.08 | 2.49/2.31 | 95.9/96.8 | 64.1/60.0 | 205.0/225.4 |
| 11-09 23:52 | refactor-boot1 | armed tmpfs | 9 | 4.72 | 15.20 | 1.02 | 2.60 | 95.7 | 62.4 | 206.6 |
| 11-10 00:06 | refactor-boot3-ud | armed tmpfs | 3 | 4.71 | 14.55 | 0.79 | 2.96 | 95.8 | 62.2 | 193.2 |
| 11-10 00:10 | refactor-boot4-pass2 | armed tmpfs | 14 | 3.55 | 13.98 | 9.22 | 2.27 | 96.4 | 64.0 | 214.2 |
| 11-10 00:30 | refactor-warm-test | armed tmpfs | 3 | 4.65 | 14.35 | 1.39 | 2.59 | 96.0 | 62.6 | 200.5 |
| 11-10 00:34 | iso-legA (arm) | armed tmpfs | 3 | 4.76 | 27.33 | 0.97 | 6.76 | 94.0 | 62.6 | 120.8 |
| 11-10 00:38 | iso-legB/legC | armed tmpfs | 3 | 4.67 | 13.71 | 0.63 | 2.11 | 96.4 | 62.8 | 211.7 |
| **11-10 13:37** | **GSQ POST-REBOOT WARM BENCH — THE BANK** | **armed /home q2_0** | **11** | **4.70** | **13.70** | **1.16** | **2.00** | **96.8** | **62.8** | **217.6** |
| — | **(gap: no GSQ boot Nov 10 16:47 → Nov 18 13:48)** | — | — | — | — | — | — | — | — | — |
| 11-18 13:48 | estate-move boot 1 (mtp path broken) | none | 0 | — | — | — | — | — | — | — |
| 11-18 16:11 | estate-move boot 2 | none | 0 | — | — | — | — | — | — | — |
| 11-18 16:17/16:42/16:50 | NaN-hunt boots (armed; NaN #879 killed every request at window 1) | armed tmpfs | 0 | — | — | — | — | — | — | — |
| 11-18 16:53 | gen-tee diagnostic boot | none | 0 | — | — | — | — | — | — | — |
| 11-18 17:20 | first post-rollback boot (France proof) | **disarmed** | 1 (cold) | 4.57 | 21.37 | 1.21 | 6.76 | 95.8 | 59.3 | 109.9 |
| 11-18 17:32 | post-rollback validation | disarmed | 1 (cold) | 4.80 | 24.02 | 0.87 | 9.36 | 95.5 | 65.6 | 123.7 |
| **11-18 17:49** | **t66-warm-r1** | **disarmed** | **3** | **4.71** | **17.76** | **0.89** | **4.38** | **96.2** | **62.9** | **170.0** |
| **11-18 17:52** | **t66-warm-r2** | **disarmed** | **3** | **4.68** | **17.98** | **1.34** | **4.93** | **95.8** | **61.0** | **169.9** |
| **11-18 17:54** | **t66-warm-r3** | **disarmed** | **3** | **4.66** | **17.05** | **0.76** | **4.23** | **96.3** | **61.6** | **176.2** |
| 11-18 18:12 | t67 stock engine (v0.2.15; own counting, rows 5.95) | n/a | 1 | 4.74 | 24.09 | 0.93 | 8.54 | 95.9 | 63.5 | 121.5 |

Lane split, bank boot vs t66 legs (warm = rows≥4, conc = rows<4):

| boot | lane | n | window | gpu-wait | cpu-exp | hit % | accept % | rows/s |
|---|---|---|---|---|---|---|---|---|
| 11-10 13:37 bank | warm | 6 | 13.46 | 0.57 | 1.98 | 96.5 | 63.7 | 289.1 |
| 11-10 13:37 bank | conc | 5 | 13.81 | 9.45 | 2.05 | 97.4 | 60.5 | 208.8 |
| 11-18 t66 legs | warm | 2/leg | 17.05-17.98 | 0.50-0.67 | 4.23-4.93 | 95.8-96.3 | 64.3-65.7 | 209.1-228.8 |
| 11-18 t66 legs | conc | 1/leg | 16.30-16.93 | 9.39-9.55 | 3.94-4.41 | 97.0-97.5 | 57.5-60.7 | 169.9-176.2 |

## 3. The bank battery, full anatomy (2026-11-10 13:37:59 boot; engine log lines 117845-124367)

- t/s anchor: `219.6/210.4/290.3` (solo/pair/conc, warm multi-leg; operator anchor recorded in `campaign/batteries/t66-warm-recheck.sh`; probe stdout not captured — named gap).
- Per-request engine lines (warm legs, representative): prompt 81/gen ~111 → 230-289 tok/s; prompt 209/gen 92 → 224-246 tok/s; drafts accepted 59-83 of 77-102.
- Flush-level: 11 flushes (6 warm, 5 conc); window mean 13.77 (median 13.70); per-window cpu-exp mean 2.01 / median 2.00 / range 1.80-2.24; warm gpu-wait median 0.57; hit median 96.8; accept median 62.8; flush rows/s median 217.6.
- Representative warm flush, term by term:

```
strata batch: 719 windows, avg 4.85 rows, 13.32 ms/window = run 11.89 (CUDA0 GPU-reach wait 0.57
  + CPU experts 1.89) + commit 0.35 + emit 0.36 + adapt wait 0.73; per layer-window: CPU experts 0.74;
  routed entries: VRAM 1475.2 (96.8%), PCIe 8.2 (0.5%), CPU 40.7 (2.7%);
  294.1 rows/s; slot drafts accepted 1364 of 2112 (64.6%)
```

- t66 warm flush, same workload class:

```
strata batch: 720 windows, avg 4.84 rows, 17.76 ms/window = run 15.86 (CUDA0 GPU-reach wait 0.53
  + CPU experts 4.38) + commit 0.37 + emit 0.38 + adapt wait 1.16; per layer-window: CPU experts 0.84;
  routed entries: VRAM 1444.2 (96.2%), PCIe 10.4 (0.7%), CPU 47.3 (3.1%);
  209.1 rows/s; slot drafts accepted 1362 of 2114 (64.4%)
```

- Window delta decomposition (bank warm 13.32 → t66 warm 17.76, +4.44 ms): cpu-exp +2.49, adapt-wait +0.43, commit/emit +0.04, gpu-wait −0.04. The regression is the cpu-exp term.

## 4. Attribution: move vs rollback

- **Move timing corrected.** The estate move commit `945e747d0` is dated 2026-11-18 16:10:37 +0200 (not Nov 17); the rollback `385937a4d` 2026-11-18 17:22:44 +0200. The move was executed before the 13:48:36 boot (first `/mnt/DATA` pack paths); the commit landed after validation.
- **No anatomy exists at the move.** The six move-era boots (13:48-16:53) logged zero `strata batch` flushes: the armed boots (16:17/16:42/16:50, armed on `/dev/shm`) NaN'd every request at its first window (`non-finite logits #879`, log lines 124497+); the others died on config errors or were validation boots. Battery evidence cannot place the rise at the move because the move never completed a measurable window.
- **First flush-bearing post-move boot is post-rollback.** 17:20:32 (disarmed; the France-proof boot cited in the rollback commit) already shows cpu-exp 6.76 cold; t66's three legs show 4.23-4.93 warm.
- **Mechanism pins it to the disarm.** Armed shadow was present in every flush-bearing boot of the ~2.0 era (Nov 9 16:49 → Nov 10 13:37; shadow line in each boot segment). `385937a4d` set `STRATA_SHADOW_EXPERTS=""` in both live qwen launchers (`campaign/lib/launch-common.sh` `launch::shadow_export`). Post-rollback, CPU misses compute from the canonical IQ3_S blob (2.66 MB, ntfs3 mmap on `/mnt/DATA`) instead of the Q2_0 shadow tier (1,382,400 B, tmpfs): 1.93x the bytes. Measured per-entry cost: bank 1.89 ms / 40.7 entries = 46 μs; t66 4.38 ms / 47.3 entries = 93 μs — 2.0x per entry at constant entry count (+16%), constant hit class (96.2 vs 96.8), constant accept (64.4 vs 64.6), constant gpu-wait (0.53 vs 0.57).
- **Not a code regression.** The engine binary path is unchanged across the eras; the rollback's own diff is launcher env + config paths only ("unset = nothing takes a new branch, byte-identical"). The doubling is the removal of a performance feature (the shadow compute tier), not a introduced cost.

## 5. Evaluation and boundary

- The t/s regression (bank 219.6/210.4/290.3 → t66 169.1/169.0/225.2, −23/−20/−22 %) is fully accounted for by the window delta at fixed tokens/window (accept and hit unchanged): window 13.4-13.8 → 17.0-18.0 ms is −22 % throughput arithmetic. t66 legs: M1 170.7/176.6/160.0, pair 162.8/169.2/175.1, conc 212.5/230.1/232.9.
- Boundary that would change the conclusion: a probe of the armed shadow on the moved estate post-NaN-fix. If armed+post-move cpu-exp does not return to ~2.0, the ntfs3 canonical-mmap path (or the memfd-from-pack ladder state) carries a second-order cost the disarm is masking. Next executable probe: re-arm `STRATA_SHADOW_EXPERTS` after the ActQ/run_split_multi NaN fix and re-run t66's recipe (3 legs).
- Restore path for the ~20 %: fix the armed CPU-miss NaN, re-arm, expect cpu-exp ~2.0, window ~13.4-13.8, warm lanes back inside the bank band.

## 6. Named gaps

- Bank probe stdout (M1/M2/M3 summary of the Nov 10 bank) not captured anywhere; the trio 219.6/210.4/290.3 is operator-anchored, engine-level evidence consistent (warm flush rows/s 289-294, conc 208).
- `campaign/tracker/lanes.csv` rows 175-177 (the bank) carry tps=0 — unbooked.
- t67 stock anatomy not comparable 1:1: stock engine counts rows 5.95/window and runs its own verify composition (cpu-exp 8.04-8.89 at conc-phase gpu-wait 11.1-11.3; per-row cpu-exp ≈1.4 ms, comparable per-row).

## 7. Evidence index

- Live engine log (all flush anatomy, boot-stamped): `/home/svg/LLM/strata-custom-2/logs/custom-iq3_s.log` (bank boot segment lines 117845-124367; t66 legs 126362-131639; NaN lines 124497+)
- Bank battery console: `/home/svg/LLM/strata-custom-2/campaign/results/postreboot-warm-1.out`
- t66 battery: `/home/svg/LLM/strata-custom-2/campaign/results/t66-warm-recheck.log`, legs `t66-warm-r1/r2/r3.out`
- t67 battery: `/home/svg/LLM/strata-custom-2/campaign/results/t67-stock-headtohead.log`
- Rollback diff: `campaign/lib/launch-common.sh` (`launch::shadow_export`), git `385937a4d`
- Estate move commit: git `945e747d0`; bank-era t66 anchor text: `campaign/batteries/t66-warm-recheck.sh` (git `1cdd2851b`)
- Lane ledger: `/home/svg/LLM/strata-custom-2/campaign/tracker/lanes.csv` (rows 160-177 = Nov 9-10 banks)
