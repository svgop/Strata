---
type: research
status: measured
date: 2026-10-27
lane: strata-custom-2
topic: the ranked lever map — every high-value structural lever, priced from code and measurement
---

# High-value lever synthesis — 2026-10-27

This is the synthesis the code review was for: the three review passes (pool, concurrency,
drafter — the last done in main context), the F-365 fault watch, and the F-366 EXP_MODE matrix
measured on the real shapes, all folded into one corrected cost model and one ranked list.

## 1. Corrected window cost model (solo, 14.15 ms/window, 2.74 tokens/window)

The F-361 anatomy's "expert math" bucket was wrong: the `gr_*` kernels are the hyper-connection
mixers (`hc_attn_`/`hc_ffn_` reads, verify.cpp:1426, 2389-2422), not MoE. Split correctly:

| component | ms/window | share | note |
|---|---:|---:|---|
| expert math (mmvq_il + gu_swiglu + down_multi + mmvq_multi) | ~4.0 | 28% | bench-confirmed: ~90 µs per layer × 48 |
| HC mixers (gr_up75 + gr_down_staged + gr_norm_multi) | ~1.7 | 12% | fused_gr.cu, not MoE |
| spin-wait inside kernels (dm_wait_combine + wait_flag_ge_planned) | ~1.6 | 11% | bimodal; the tail is the pool |
| attention + GDN | ~1.2 | 8% | |
| GPU idle (host-side exposure) | ~1.85 | 13% | idle + spin ≈ 24% of the window |
| sampler + route + act-quantize | ~1.5 | 11% | |
| draft chain (serial after verify) | 1.12 | 8% | |
| remainder | ~1.2 | 9% | |

Host-side exposure (idle + spin ≈ 3.4 ms) is the biggest single structural pool: it is
**the pool's delivery latency, not the GPU's work**.

## 2. New measurements this session

### F-365 — fault watch, one probe leg on the live engine
2,635,199 minor faults during one probe leg while shadow RSS grew only +160 MB: the page cache
is turning over the 31.64 GiB shadow mapping under load (91% resident after boot, evicted during
it). Evicted blobs re-fault from NVMe inside a layer — the `dm_wait_combine` 9.8 ms max is that
tail. `ulimit -l` = 8 GiB rules out mlock; the fix is the store itself: tmpfs.

### F-366 — EXP_MODE matrix on the real shapes (G=32 T=7, layer 20 iq3_xxs/q2_0, layer 36 iq2_s/q2_0)
`native_expert_bench`, bitwise-equal, min-of-3 interleaved, GPU-free window:

| mode | all (µs) | gate/up GB/s | down GB/s |
|---|---:|---:|---:|
| 0 (CUDA, the standing layout) | 85.5–94.5 | 644–709 | 371–394 |
| 2 (R2) | 173.9 | 317 | 286 |
| 3 | 98.3 | 675 | 394 |
| 4 | 173.5 | 308 | 336 |
| 5 | 179.4 | 294 | 390 |
| 6 | 164.3 | 323 | 393 |
| 7 | 97.4 | 638 | 418 |
| 8 | 93.5–97.6 | 682–727 | 396–418 |

**Verdict: mode 0 is already the best layout on the 5090; no env switch closes the gap.**
The grouped kernel streams at 650–700 GB/s (gate/up) and 370–400 GB/s (down) against the
card's ~1.79 TB/s DRAM ceiling — the head kernel runs at 1.18 TB/s on the same card. So the
expert kernel is not DRAM-bound: it is dequant-ALU bound on iq3_xxs/iq2_s and short-row
overhead bound on the down projection (2560 rows × 640 values ≈ 340 B/row). The ~1.5–2 ms
gap to the head's efficiency is a kernel-rewrite project, not a lever.

## 3. The ranked lever list

Ranked by expected ms/window (or lane effect) per unit of engineering risk. The two
"targets" are the standing campaign goals: 250 solo, 350 conc.

1. **Shadow store fault-free — tmpfs (`/dev/shm`)** — expected 0.6–1.0 ms/window solo + the
   conc tail. Evidence: F-365 (2.63M faults/leg), the 9.8 ms combine max, 93 GiB free in
   tmpfs, `ulimit -l` blocking mlock. Cost: 31.64 GiB of RAM made unreclaimable (engine RSS
   169 GiB of 184 — the shadow pages are *inside* that RSS today as reclaimable cache; moving
   to tmpfs makes them permanent but bounded). No capability change. **This is the top lever.**
2. **Conc scheduling: C2 tail/admission overlap + shared ExpertPool across cohorts** —
   concurrency review: +9–18% conc → 355–385, clears the 350 target. Both cohorts share one
   ExpertPool (generate.cpp:12130 `coh_b_drive = drive`): cohort B's pool work serializes
   against A's window tail. Medium-high engineering, lane-local, no capability change.
3. **Drafter acceptance (the structural multiplier)** — throughput is tokens/window ÷ ms/window;
   acceptance ≈0.62 today, 0.68 clears 250 solo at the current ms/window. The F-353 pipeline
   (harvest_sweep → aligned replay → train_cycle → stair_probe) is built; round 2 plateaued at
   0.7932 agreement and was stair-rejected — the next round needs an acceptance-weighted
   objective, not raw agreement. Highest ceiling, highest risk, longest path.
4. **ForesightSwap re-plug** — 0.4–0.8 ms (pool review); the predict→VRAM-slot pipeline exists
   and is hard-refused beside STRATA_VERIFY_DEVICE_PLAN=1 (expert_source.cpp:6238-6245).
   Needs the device-plan conflict resolved before it can be priced.
5. **Per-entry pool overhead** — 0.3–0.6 ms (pool review): the ~150 µs/entry pool delivery
   cost, of which the shadow store already cut ~half; the rest is the job-build/plan layout
   (expert_source.cpp:4949-4968).
6. **Expert kernel rewrite (iq3_xxs/iq2_s dequant + short-row down)** — 1.5–2.0 ms ceiling
   per §2; the F-366 matrix kills every cheap variant. A project, not a lever.
7. **DAgger retrain (drafter round 3)** — owned by lever 3; do not start a second drafter
   front while the acceptance-weighted objective is unwritten.

## 4. Ruled out (do not re-explore without new evidence)

- EXP_MODE layout switch on the 5090 — F-366 table above, mode 0 optimal.
- mmvq_il table retune — −4.7% on the T=2..4 shapes = 0.6% of window (F-361), shelf.
- Shared-expert hoist — G=1 under --batch 6 (F-362), shelf for non-batch lanes.
- Aux-core pinning (STRATA_AUX_CPUS) — M1 −7.7%, conc −8.2% (F-363).
- Batch depth D=2 — F-352 falsified; re-opens only behind the drafter.
- KV-window reduction — F-355 capability trade, operator-shelved ("cheating").
- Ternary b9/c3 — F-354 VRAM-uneconomical.
- BATCH_G2 — F-223, pair −22%, conc −12%.
- Per-node graph spacing — F-361 §3, wait-kernel p50 is the launch floor.

## 5. Execution order

Lever 1 (tmpfs shadow) ADOPTED 10-27 (F-367: M1 +5.5 / pair +4.2 / conc +3.5). Lever 2
(conc scheduling) follows — it is the only lever that clears 350 conc without touching the
model. **Design pass done (`c2a-tail-overlap-design-2026-10-27.md`): the T47-1 headline band
is gated behind a hazard — `draft_finish` blocks on `cudaStreamSynchronize` (mtp.cpp:1922-1936)
and the draft tail does a host D2D copy (generate.cpp:12353), both the issue-#31/F-285a
deadlock class inside another cohort's window. Split into slice 1 (overlap the acceptance +
commit + emit head only; honest +1-3% conc, low risk) and slice 2 (non-blocking draft
completion port, which makes the +8-15% band reachable).** Lever 3 (drafter) is the only lever
that clears 250 solo at the current ms/window; its next round must carry an
acceptance-weighted objective before another boot pair is spent on it.
