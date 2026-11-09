---
type: research-digest
date: 2026-11-09
question: "Are the Nov-6 GSQ banks and the post-reboot hostmem batteries (hostmem-definitive / gsq-clean / gsq-restore) the same measurement? Is the reported regression flow or engine?"
instruments: campaign/probe-lane.py, campaign/battery-lib.sh, serve/server.py (Prewarm), campaign/batteries/{hostmem-definitive,hostmem-gsq-clean,gsq-restore}.sh, logs/custom-iq3_s.log
banks: campaign/tracker/lanes.csv rows 123-131 (2026-11-06), commits 083a68a7 + 4152f114
---

# Regression vs bench-flow audit: GSQ Nov-6 banks vs the hostmem-era batteries

## Verdict (per lane, gsq-clean Nov-9 vs both Nov-6 banks)

| lane | Nov-6 night (n=2) | Nov-6 midday (n=2) | gsq-clean mean (n=3) | gap | flow-explained? |
|---|---|---|---|---|---|
| solo | 229.1 | 233.1 | 183.1 (175.7/178.4/195.1) | −20/−21% | **NO** — M1 provably ran outside the warm-in window in all 3 legs; engine-side decode carries the same −21% with no probe wall involved |
| pair | 220.7 | 224.8 | 172.0 (156.1/182.8/177.0) | −22/−23% | **NO** — runs after M1, warm-in long terminal in every clean leg |
| conc | 301.6 | 303.5 | 227.9 (221.2/233.7/228.9) | −24/−25% | **NO** — runs last, same position both eras |

The lanes are comparable in flow position and instrument; the deficit is in the engine's
own decode rate on the post-reboot pool architecture. Flow DID corrupt three other readings
(quantified below): the Nov-8 definitive battery (−31..−37%, budget bug), gsq-restore-r1
solo 75.3 (probe fired into the prewarm replay's 6-burst), and every shorts-class number
today (inside the 80-112 GiB files pass).

## Recipe A — the Nov-6 banks (tracker rows 123-131)

- **Runner:** none committed. Ad-hoc console boots of `launch-custom-iq3s-8081.sh` +
  `probe-lane.py --short-n 6`, 2 legs banked as means (deletion policy: run-once harnesses
  leave no corpse). No `.log`/`.out` for these legs exists on disk or in git.
- **Legs:** night spot 03:05:13 / 03:07:45 / 03:09:50 = 2 mask-on legs + 1 off-control
  (banked 03:11:50 in 1553a614: "GSQ battery: mask-on 2-leg means 229.1/220.7/301.6 vs the
  off-control 242.2*/209.9/283.6"). The off-control leg's engine M1 window reads 241.9 t/s
  — the banked 242.2 is that boot. Midday "quiet-box" spot: boots 10:33:48 + 11:51:25
  (2 legs; the 11:37:30 boot crashed at token 0 — the `gguf_hc_is_q8()` gate bug fixed in
  4152f114), banked 11:58:40 as 233.1/224.8/303.5.
- **Warm state:** server-side prewarm (cc05319f, Nov-4) was ARMED — config `custom-iq3_s-8081.json`
  carries `"prewarm": {}` since Nov-4, `server.py` arms it whenever the key is non-null.
  Whether the spot runner called `wait_prewarm` is **unrecorded**; boot→boot cadence
  (152 s / 125 s night) fits load ~60 s + probe ~60 s with **no** prewarm wait. Every
  committed battery of that era (t31-rebank, t44-champ-bank, env/binary ladders) calls
  `wait_prewarm`, and t22 carries the dated warning: "2026-11-05: a probe fired mid-warm-in
  reads the warm-in, not the engine."
- **Box state:** 4K-pinned complement (FileExpertSource one-shot `cudaHostAlloc`, banner
  "0.00/0.01 GiB of 37.69 GiB in 2 MiB pages (THP accepted)"), /dev/shm shadow store
  (31.64 GiB, persistent across boots), no hugepage pool, gpu-lock v2 (mid-lock bounce),
  zram full later that evening. Same-day engine M1 swing: **242-275 (bank window) → 80-93
  (18:44, zram+clock) → 228-236 (20:00)**. The bank rode a favorable window, and its own
  ledger says so: `campaign/results/fork-ud-first-bank.md` §"The regression hunt":
  "The banks (one afternoon's 2-leg means) may also be a favorable-window artifact."
  Note conc had already fallen 350.8 (Oct-27 F-367) → 303.5 (Nov-6) BEFORE any pool work.

## Recipe B — today's batteries

| | hostmem-definitive (Nov-8 13:07) | hostmem-gsq-clean (Nov-9 10:22) | gsq-restore (Nov-9 10:38) |
|---|---|---|---|
| script | batteries/hostmem-definitive.sh | batteries/hostmem-gsq-clean.sh | batteries/gsq-restore.sh |
| legs | 3 boots × probe-lane (iq3s+iq2xs) | 3 boots × probe-lane | 5 boots, bank last 3 |
| wait_prewarm | **NO** (health-wait then immediate probe) | **NO** | **NO** |
| health→probe | same second (LEG line = PROBE line) | same second | same second |
| complement | **19.80 GiB of 37.97 capped** (pool-blind budget bug, e8484c36) | 37.59-37.85 GiB whole (HUGETLB POOL) | 38.00-38.01 GiB whole |
| store | memfd 31.64 GiB + /dev/shm copy **held twice** | same double hold | single-owner (f95455a6): memfd from pack only |
| prewarm targets | 18 files, 111.7 GiB | 17 files, 80.1 GiB | 18 files, 111.7 GiB |
| interference | compact_fail +2222..+3737/boot | none recorded | r2+ collided with operator chat (max_tokens 262049, 9830/23003/54006-tok generations) — legs r2-r5 never landed |

`probe-lane.py` is byte-identical across eras: `git diff 1553a614 HEAD -- campaign/probe-lane.py`
is empty; last touched 2026-11-03 (a60e6f29). Same prompts, same warmup ('Say OK.'),
same `--short-n 6`, same M1/pair/conc shapes and order. `serve/server.py` unchanged since
Nov-5 17:42 (5d17b967); `battery-lib.sh` unchanged; config unchanged. The instrument is
constant; the recipes are not.

## Timestamp tables — prewarm overlap per boot (today)

Probe phases in order: warmup → 6 A.shorts → fresh-long → 6 B.shorts → M1 → pair → 6-conc.
Files pass does NOT yield to traffic; replay yields (any foreign request ends it).

| boot | health=probe start | files pass (X GiB in Y s → ≈ends) | replay | M1/pair/conc vs warm-in |
|---|---|---|---|---|
| gsq-clean r1 | 10:25:58 (load 213 s) | 80.1 GiB/18 s → ~10:26:16 | done 7 s → ~10:26:23 | M1 done-line AFTER replay-done line — **clean**; warmup+all shorts inside files pass |
| gsq-clean r2 | 10:28:42 (65 s) | 80.1/19 s → ~10:29:01 | skipped ("live traffic arrived") | **clean** |
| gsq-clean r3 | 10:31:12 (60 s) | 80.1/17 s | skipped | **clean** |
| hmd-iq3s r1 (Nov-8) | 13:08:19 | 111.7/27 s | skipped | **clean** (solo 165.2 — budget bug, not flow) |
| hmd-iq3s r2 | 13:10:49 | 80.1/22 s | done 8 s | **clean** (solo 120.9 — budget bug) |
| hmd-iq3s r3 | 13:13:22 | 80.1/18 s | skipped | **clean** |
| gsq-restore r1 | 10:39:25 | 111.7/20 s | **done 19 s** | **M1 ran INSIDE the replay tail/burst → solo 75.3** (probe wall 26.57 s vs 10.3-11.4 s in clean legs) — invalid leg |

Warm-in bandwidth on the box today: 80.1 GiB in 17-20 s = **4.0-4.7 GiB/s** sustained
read+page-cache-fill; 111.7 GiB in 18-27 s = **4.1-6.2 GiB/s** (Nov-6-era reference:
80.1 GiB in 8 s = 10 GiB/s, cc05319f's own measurement — the pool reservation shrank the
page cache and slowed the sweep). Any probe leg inside that window shares memory bandwidth
with a 4-6 GiB/s stream. In today's boots that window covers warmup + shorts only; M1 and
beyond escape it in 6/7 boots — and still read −21..−25%.

## Engine-side cross-check (warm-state-immune witness)

M1's thinking window (~1489 tok of the same 2000-tok REAL request, engine's own accounting,
steady-state decode, insensitive to which second the probe fired):

| boot(s) | complement/state | engine M1 t/s |
|---|---|---|
| Nov-6 03:05/03:07/03:09 (night bank legs) | 4K pinned, /dev/shm store | 243.0 / 247.7 / 241.9 |
| Nov-6 10:33 + 11:51 (midday bank legs) | same | 221.9-274.5 (bank legs 274.5/249.6) |
| Nov-6 evening (zram full + v2 clock bounce) | same | **80.5-93** — the box's own −65% swing, same binary |
| Nov-7 16:56-16:59 leg0 (aged box, 6 d uptime, pre-reboot) | same | 232.7 / 233.3 / 214.9 (probe-wall bank 206.3 = −10..−15%) |
| Nov-8 13:07-13:12 definitive | **19.80 GiB capped** + double store hold | 178.1 / 164.5 / 191.3 |
| Nov-9 10:22-10:30 gsq-clean | 37.6-37.9 GiB whole + double store hold | 191.0 / 195.7 / 195.7 |
| Nov-9 10:56/10:59 boots (single-owner store) | 38.01 GiB whole, no double hold | **203.2 / 213.4** |

The engine decode table reproduces the probe-wall table lane for lane. That is the
falsifier: no measurement-flow mechanism can move the engine's own per-window decode
accounting.

## Quantified confound list

1. **Prewarm files-pass overlap** — real, 4.1-6.2 GiB/s for 17-27 s per boot, covers
   warmup+shorts in 7/7 boots, M1 in 0/7 clean boots. Effect on M1/pair/conc: ~0 in
   gsq-clean; cannot explain the lane deficits.
2. **Replay burst collision** — gsq-restore-r1 solo 75.3 = −61% vs the same-day clean
   engine class (191). One line in server.py explains it: the replay's 6-burst shares the
   GPU while M1 decodes. Invalid leg; do not bank it.
3. **Budget-bug complement cap (19.80 of 37.97 GiB)** — the Nov-8 definitive battery is a
   real engine-state defect (e8484c36 names it "the REAL post-reboot GSQ regression"),
   −31..−37% probe-wall. Fixed Nov-9 10:22, before gsq-clean. Not flow, but also not the
   current architecture.
4. **Double store hold during gsq-clean** — boots pre-f95455a6 (10:38) held the store
   twice (memfd + /dev/shm 31.6 GiB); f95455a6 calls it "dead resident weight, page-cache
   squeeze". Post-rule boots recovered 191→203-213 engine-side (+6..9%). gsq-clean's
   −21% carries ~7 points of this tax.
5. **wait_prewarm dropped** — all three hostmem-era batteries violate campaign law 11
   ("warmup is part of the bench", the wording battery-lib itself carries). The era's
   committed banks (t31/t44/ladders) wait. Recipe drift, yes — but it bites shorts and
   first-request legs, not the banked M1/pair/conc of gsq-clean.
6. **First-boot load asymmetry** — 213 s vs 60-65 s (store memfd populate 7.4-15.2 s from
   disk + pool init). Wall-clock cost only; no lane effect.
7. **Operator traffic collision** — gsq-restore r2+ (262049-token chat at 150-198 t/s).
   Legs never landed; the bank that exists is r1 only.
8. **Era-neutral observation** — `NameError: releasing is not defined` (server.py:1728)
   fires 10-17x per boot in every Nov-7/9 battery (0x in the Nov-5/6 t22 .outs). Same in
   all post-reboot batteries, so it does not differentiate the two eras — but it is a live
   serve-layer bug firing per concurrent request.

## Verdict

- The two eras' **lane numbers are comparable in instrument and flow position** (probe
  identical, server identical, M1/pair/conc sequenced identically). They are **not
  comparable in engine architecture or box state**: 4K complement + persistent /dev/shm
  store + gpu-lock v2 vs pool complement + memfd store + gpu-lock v4 + fresh kernel.
- The −8..−15% the operator cites is understated: gsq-clean's means sit −20/−22/−25% vs
  the banks. Of that: ~0 points are flow (M1/pair/conc ran outside the warm-in window);
  ~7 points are the double store hold (removed at 10:38 Nov-9, boots 10:56+ read 203-213);
  the remainder is the post-reboot pool-era engine state versus the Nov-6 window — against
  a bank that (a) was a 2-leg favorable-window spot, (b) rode a box that swung −65% the
  same evening, and (c) had already given back 13% of conc vs Oct-27 before any pool work.
- **The regression is real but mis-banked**: no same-window control exists for the pool-era
  boot vs the Nov-6 boot (the arenas were adopted on leg-3's +7.2% solo a day BEFORE the
  budget bug inflated everything, and the definitive battery that should have certified
  the composition ran cap-blind). The engine-side table is the proof the deficit is in the
  engine, not in how the numbers were taken.

## Closers

1. Finish gsq-restore (5 boots, last-3 bank) with the box quiet — its r3/r4-class boots
   already read 203-213 engine-side; that is the current architecture's honest lane.
2. Re-arm `wait_prewarm` in hostmem-definitive.sh / hostmem-gsq-clean.sh / gsq-restore.sh
   (one line each; battery-lib provides it) or gate the probe on /health's prewarm word.
3. Law-17 close-out: one boot `STRATA_HUGEPAGE_ARENAS=0 STRATA_SHADOW_MEMFD=0` same-window
   3-leg control on today's box — splits (pool composition) from (Nov-6 window selection).
4. Never bank a leg whose probe start precedes prewarm's terminal state; record the
   prewarm lines with the leg (gsq-restore.sh already appends witnesses — keep it).
5. Fix server.py:1728 (`releasing` → `self.slot_releasing`) — per-request exception on the
   concurrent path in every current battery.
