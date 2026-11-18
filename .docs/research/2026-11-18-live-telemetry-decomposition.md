---
type: research
date: 2026-11-18
title: LIVE lane window anatomy decomposition vs bank era and T66 shallow battery
lane: strata-custom-2 (port 8081, engine 0.1.46, qwen3.8-flash-next-iq3_s)
method: READ-ONLY — GET /metrics x19 @10 s, GET /v1/status, GET /v1/models, engine log flush tail. No POST, no restart.
---

# LIVE telemetry decomposition — 2026-11-18

## Method and capture window

- Engine on 8081 stayed live under the operator the whole time (no restart, no POST).
- 19 `GET /metrics` samples, 10 s apart (18:21:26–18:24:26), full JSON sections
  (`engine`, `live`, `requests`, `window`, `elastic`, `totals`, `hardware`).
- WSTAT family read from the batch-flush lines in `logs/custom-iq3_s.log`
  (current boot, 12 flushes) — the per-window anatomy instrument per
  [[strata-live-self-telemetry]].
- `/metrics` `window` WSTAT snapshot: windows 117, rows 1086, p50 35.31 / p99 153.1 /
  max 174.3 ms — refreshes only on an idle flush and covers a co-tenant/deep mixed
  period (9.28 rows/window, prefill-storm-inflated p99). The log flush lines are the
  authoritative anatomy; this snapshot is secondary evidence of the same lag doctrine.
- The operator ran turns at depth 160,878 → 186,075 → 189,933 → an agent session
  growing 153,753 → 167,662 during/after the sampling window.

## LIVE engine section (this boot)

| term | LIVE now | bank-era anchor | note |
|---|---|---|---|
| expert_slots | 9,968 | 10,476 standing (F-355 era) | −488 (−4.7%) |
| expert_cache_mib | 19,364 | not separately banked | — |
| vram_free_mib | 1,232 | ~1,244 (10-27 note) | at bank |
| pool_workers | **7** | **15** (launcher default; F-349 "15 stays") | **−8 workers — config delta, prime cpu-exp suspect** |
| pcie_frac | 0.35 | 0.35 (262K config) | at bank |
| batch_slots | 6 | 6 | at bank |
| spec | 8 (verify window ≤8 rows) | `--spec 6` → ≤8 rows | same family |
| mtp_max | 6 | MTP depth family | same family |
| kv_resident | 32,768 int8 | 32,768 | at bank |
| cvec | project:1-47, steered 47/48 | project:1-47 | at bank |
| version | 0.1.46 | T60/F-376 head (10-30) | newer tree |

## LIVE series (requests, this boot — depth | decode t/s | drafts accepted)

| depth | reused | decode t/s | drafts accepted |
|---|---|---|---|
| 106,678 (cold) | 0 | 153.6 | 3246/4059 = 80.0% |
| 114,814 | 111,912 | 160.1 | 81.7% |
| 119,292 | 118,153 | 164.4 | 82.6% |
| 124,141 | 122,568 | 155.2 | 82.4% |
| 129,375 | 126,839 | 158.2 | 83.3% |
| 138,746 | 136,777 | 160.5 | 83.9% |
| 153,753 | 153,684 | 148.3 | 87.7% |
| 154,931 | 154,034 | 159.7 | 93.0% |
| 155,985 | 155,128 | 147.1 | 89.8% |
| 156,827 | 156,367 | 146.2 | 87.1% |
| 157,993 | 157,348 | 157.0 | 83.6% |
| 158,275 | 158,178 | 161.7 | 84.0% |
| **160,878 (cold)** | 0 | **175.9** | 86.4% |
| 163,214 | 161,076 | 168.7 | 85.6% |
| 167,662 | 167,022 | 157.7 | 81.7% |
| 186,075 | 162,550 | 170.0 | 86.6% |
| 189,933 | 189,724 | 193.0 | 90.4% |

Prefill side: cold 160,878 read at 6,462.8 t/s bulk (marginal ≈ 6,774 t/s after the
~1.15 s fixed tax — at the banked 6.5–7K class); warm reuse reads 69–2,536 tokens in
408 ms–1.2 s (launch-bound class, at bank).

## WSTAT flush classes (current boot, logs/custom-iq3_s.log)

**Shallow class (T66-shallow battery — flushes at log lines 128123/129873/131623):**
716–728 windows, avg rows 4.79–4.87, window 17.05–17.98 ms = run 15.32–15.96
(GPU-reach wait 0.50–0.67 + CPU experts 4.23–4.93) + commit 0.36–0.37 + emit 0.37–0.38
+ adapt wait 1.00–1.27; per layer-window CPU 0.80–0.90, VRAM hits 30.09–30.53, PCIe
0.19–0.22; routed VRAM 1444.2–1465.6 (95.8–96.3%), PCIe 10.1–11.4 (0.7%), CPU
46.5–52.1 (3.1–3.4%); adaptive 87–94 rounds / 5,967–6,646 swapped; 209.1–228.8 rows/s;
slot drafts 64.3–65.7%; MTP 64.0–64.4%.

**Deep class (160K-depth decode — the LIVE-now anatomy; flushes 126089/126905/128678/
130434/132191; 132191 = the 160,878 cold turn's decode, 8,940 ms wall):**
227–234 windows, avg rows 4.66–4.80, window 20.80–24.09 ms = run 18.22–21.22
(GPU-reach wait 0.76–1.34 + CPU experts 6.37–9.36) + commit 0.36–0.38 + emit
0.36–0.39 + adapt wait 1.75–2.11; per layer-window CPU 0.84–0.99, VRAM hits
29.50–32.75, PCIe 0.14–0.18; routed VRAM 1416.0–1572.0 (95.5–96.3%), PCIe 7.3–10.5
(0.5–0.7%), CPU 47.0–61.7 (3.2–3.8%); adaptive 29–30 rounds / 2,249–2,497 swapped;
121.5–131.4 rows/s; slot drafts 61.0–65.6%; MTP 62.6%.

**Serial class (drain-print / serial-lane flushes 127257/129025/130779):**
534–566 windows, rows 3.11–3.24, window 16.30–16.93 ms = run 14.52–14.95 (GPU-reach
wait **9.36–9.55** — the serial-lane signature, healthy class) + CPU experts 3.94–4.41
+ commit 0.04–0.05 + emit 0.60–0.63 + adapt 1.14–1.31; routed VRAM 1859.2–1883.3
(97.0–97.5%); 169.9–176.2 rows/s; slot drafts 57.5–62.9%.

## Depth slope law applied to LIVE

Law ([[strata-live-session-review-2026-10-30]], shallow slope at the 262K config):
~200 t/s @ 20–60K · 192–206 @ 120–140K · 179–184 @ 160–180K · 171 @ 261K. Bands below
are segment-interpolated at the live depth.

| LIVE turn | depth | expected band | LIVE t/s | delta |
|---|---|---|---|---|
| cold long turn | 160,878 | 179–184 | 175.9 | **−1.7…−4.4%** |
| long warm turn | 186,075 | 176.9–180.6 | 170.0 | **−3.9…−5.9%** |
| short warm turn | 189,933 | 176.6–180.1 | 193.0 | **+7.2…+9.3% (above bank)** |
| agent burst set | 153.8–158.3K | 179–185 | 146.2–161.7 | **−9.7…−20%** |
| agent long turn | 163,214 | 178.7–183.6 | 168.7 | −5.6…−8.1% |
| agent turn | 167,662 | 178.2–182.7 | 157.7 | −11.5…−13.7% |
| latest deep turn | 158,275 | 179–185 | 161.7 | −9.7…−12.6% |

Verdict: LIVE deep decode runs under the banked slope at matched depth in 5 of 7
classes (−2 to −20%, worst on short agent bursts); one suffix-draft-friendly short
turn read above bank.

## Per-term table — bank era | T66 shallow | LIVE now

Bank era = 2026-10-30 262K concurrent re-pull (13.40 ms/window @ 4.81 rows → 271.5
rows/s; the memory does not record that flush's depth; its session spanned 20K–261K).
T66 shallow = today's shallow flushes (identical to the task's battery band).
LIVE deep = 160K-depth decode class (the operator's current window).

| term | bank era | T66 shallow | LIVE deep (160K) | verdict |
|---|---|---|---|---|
| rows/window | 4.81 | 4.79–4.87 | 4.66–4.80 | at bank |
| window ms | 13.40 | 17.05–17.98 (+27–34%) | 20.80–24.09 (+55–80%) | **slower, both classes, shallow already regressed** |
| gpu-reach wait | 0.50 | 0.50–0.67 (at bank) | 0.76–1.34 (+52–168%) | shallow at bank; deep slower |
| CPU experts ms/window | 2.04 | 4.23–4.93 (**2.1–2.4×**) | 6.37–9.36 (**3.1–4.6×**) | **slower at ANY depth — the carrier** |
| commit ms | 0.34 | 0.36–0.37 | 0.36–0.38 | at bank (±0.05) |
| emit ms | 0.35 | 0.37–0.38 | 0.36–0.39 | at bank (±0.05) |
| adapt wait | 0.82 | 1.00–1.27 (+22–55%) | 1.75–2.11 (**+113–157%**) | slower, grows with depth |
| per-layer CPU experts | 0.76 | 0.80–0.90 | 0.84–0.99 | slower at depth |
| per-layer VRAM hits | 30.06 | 30.09–30.53 | 29.50–32.75 | at bank |
| VRAM hit % | 96.5 | 95.8–96.3 (−0.2…−0.7pp) | 95.5–96.3 (−0.2…−1.0pp) | at bank (marginal) |
| routed CPU entries/window | 43.0 (2.9%) | 46.5–52.1 (3.1–3.4%) | 47.0–61.7 (3.2–3.8%) | mildly slower (fewer slots) |
| routed PCIe % | 0.6 | 0.7 | 0.5–0.7 | at bank |
| slot-draft acceptance | 62.9–63.9% | 64.3–65.7% (+1.4–2.8pp) | 61.0–65.6% | at/above bank |
| rows/s | 271.5 | 209.1–228.8 (−16…−23%) | 121.5–131.4 (−52…−55%) | slower (depth + shallow regression) |
| adaptive tier | 92 rd / 5,806 swaps | 87–94 rd / 5,967–6,646 | 29–30 rd / 2,249–2,497 (per ~9 s) | at bank when normalized per-second |
| pool_workers (config) | 15 | — | **7** | **config delta** |
| expert_slots (config) | 10,476 | — | 9,968 | −4.7% config delta |
| vram_free_mib | ~1,244 | — | 1,232 | at bank |
| tok/s at depth (slope-law-matched) | 200→171 band | — | −2…−20% under band (1 turn above) | slower on 5/7 classes |

## Slower than banked even at matched depth

1. **CPU experts ms/window — the dominant carrier.** 2.04 → 4.23–4.93 shallow →
   6.37–9.36 deep. The shallow class alone is 2.1–2.4× bank, so this is NOT a depth
   effect. Prime suspect: `pool_workers` 7 vs the banked 15 (F-349: "15 stays"; the
   9950X3D has 16 physical cores — 7 workers halves the miss-row compute fleet), with
   expert_slots 9,968 vs 10,476 adding a smaller miss-rate term (routed CPU entries
   46.5–61.7 vs 43.0).
2. **Window period.** 13.40 → 17.05–17.98 shallow (+27–34% at matched shallow depth).
   Mostly the cpu-exp term plus adapt wait.
3. **Adapt wait.** 0.82 → 1.00–1.27 shallow → 1.75–2.11 deep (+113–157%).
4. **rows/s.** 271.5 → 209–229 shallow → 121–131 deep. Partially depth-expected
   (attention knee), but the shallow class already gives back 16–23%.
5. **Deep-class gpu-reach wait** 0.76–1.34 vs bank 0.50 (shallow class matches bank at
   0.50–0.67, so the deep excess is depth-attributable but the bank lacks a
   depth-labeled reference to clear it fully).
6. **tok/s at depth.** −2…−20% under the slope-law band on 5/7 turn classes.

## At or above bank

- VRAM hit %: 95.5–96.3 vs 96.5 — within 1 pp, at bank.
- Slot-draft acceptance: 61.0–65.6% vs 62.9–63.9 — at/above bank (shallow class above).
- commit/emit tails: 0.36–0.39 vs 0.34–0.35 — at bank.
- rows/window 4.66–4.80 vs 4.81 — at bank.
- Engine static config: pcie_frac, batch_slots, kv_resident, cvec, vram_free — all at
  bank.
- Prefill marginal rate ~6.8K t/s at 160K cold — at the banked 6.5–7K class.
- One 189,933-depth turn decoded at 193.0 t/s — +7–9% above the slope band.

## Flags for the next tier

1. Confirm `pool_workers` intent: launcher default is 15; live boot runs 7. If
   unintended, restoring 15 is the single highest-value rollback candidate for the
   cpu-exp carrier (predicted recovery toward ~2 ms/window at matched depth).
2. expert_slots 9,968 vs standing 10,476 — check what consumed the ~0.9 GiB
   (conversation cache budget 49,152 MiB / 24 slots is a candidate consumer).
3. Adapt wait doubling at depth — F-332's scan thread is slot-gated (<3 slots); deep
   single-request turns qualify, worth a re-read against the current tree.
4. WSTAT flush cadence: no flush line prints during continuous traffic (only at
   batch-idle boundaries); the 186,075/189,933 turns and the current agent session
   have not flushed yet — their anatomy lands at the next idle.

## Raw evidence

- Samples: /tmp/telem/samples.txt (19 full /metrics JSON blobs, 18:21:26–18:24:26).
- Flush lines: logs/custom-iq3_s.log lines 125498–132191 (current boot).
- Request summaries: same log, lines 132203–132457.
