# ROADMAP: +30% vs upstream stock (the night-and-day target)

**Baseline (F-198, window 2, same-hour):** stock v0.1.44 = M1 147.1 / pair 149.1 / 6-conc 187.7.
Current = 164.8 / 160.5 / 192.7 (**+12.0 / +7.6 / +2.7%**).

**The target in absolute terms (stock-row × 1.30):** M1 191.2 · pair 193.8 · 6-conc 244.0.
**Still needed ON TOP of current:** M1 +16.0% · pair +20.8% · 6-conc +26.6%.

---

## 1. Where the time actually goes (measured, F-192/F-194)

The 6-concurrency decode window is **31.41 ms** (525-window live mean, 6.64 rows):

| stage | ms | share | what it is |
|---|---|---|---|
| GPU-reach wait | 21.68 | 69% | pure graph execution — the host spins while ~1,800 nodes run at ~9.5 µs spacing (bytes need only ~2.3 µs/node) |
| — QSA attention | 4.40 | #1 line | latency-starved: 64 serial dependent pool reads/thread, NO row sharing across query rows (width growth 1.6→4.4 ms is pure re-gather), 12 warp-shuffles/cell CUDA tax |
| — hc reads | 4.20 | #2 line | parallelism-starved: geometry gives the down-projection only 41 blocks (157 MB/window q8); Little's-law ceiling ~80 GB/s of 1,792 |
| — experts+rest | ~13 | — | MoE grouped reads + dense projections + head (521 MB/window) |
| CPU-expert doorbell | 4.90 | 16% | host runs the 2% CPU-routed experts while the GPU stalls at those layers — all-or-nothing fallback (verify_kernels.cu:1141-1155) |
| emit | 1.87 | 6% | per-slot MTP draft launches + bookkeeping on host (generate.cpp ~10300-10326) |
| commit | 1.22 | 4% | **RECOVERED by F-196** (parity double buffer) |
| adapt | 0.18 | — | healthy |

Round-9's law: the engine is **execution-bound, not bandwidth-bound** (24-26% of GDDR7 peak) —
the honest ceiling without byte cuts is ~2.9×. A 30% cut is well inside it.

**The recoverable pool in one 6-conc window: doorbell 4.9 + attention ~2.2 + hc ~1.8 + emit ~1 ≈
10 ms of 31.4 (32%)** — more than the 6.6 ms the conc target needs; the tiers below harvest it in
risk order. M1's 17 ms single window has the same attention/hc lines plus the draft/acceptance
class (rescue already banked at +20.6 pts accept / +8.2% M1; chain math: banked ≈ (1−a⁷)/(1−a)).

---

## 2. TIER 0 — the env battery (one boot each, ABAB, zero code)

All exist in-tree now; F-195's revised order. Each: fresh boot → 3+3 probes, means, Lock Law.

| # | arm | attacks | expected | gate |
|---|---|---|---|---|
| T0.1 | `STRATA_TYPICAL_DEVICE=1` | rescue moved device-side (parity-proven) | +2-5% sampled M1 | none (parity-proven) |
| T0.2 | `STRATA_VERIFY_ASYNC_LAUNCH=1` | the 69% GPU-reach line — if the launch CALL is the wait, async launch attacks it directly (v0.1.44 HostGraphLaunch) | unknown, possibly large | none |
| T0.3 | `STRATA_ATTN_LANECELL=1` | kills the 12 warp-shuffles/cell in attention | claimed bit-identical; move the 4.40 ms line | phase table self-check |
| T0.4 | `STRATA_HC_SPLIT=0` then `=1` | the 41-block starvation (down = 41 blocks) | move the 4.2 ms line | none |
| T0.5 | `STRATA_GR_V3=1` | split-K hc read | ~1-2% | MMLU (sum-order change) |
| T0.6 | `ROUTE_RESIDENT 0.75` | the CPU-expert tail (2% routed) | ~1-3% | MMLU (routing change) |
| T0.7 | `L2_SETASIDE 72` | GDN state in L2 | ~1% single-stream | none |
| T0.8 | `STRATA_MMQ_KQ=1` | prefill K-quant kernels (prefill metric) | upstream ×1.3-1.7 claim on prefill | none |

A knob that moves its phase-table line but not wall time tells us the line was hidden — equally
valuable. Cheapest first, always same-window toggles (the F-189 law: no comparison across 20+ min).

---

## 3. TIER 1 — surgical code (the ceiling-breakers, measured-value order)

### T1.1 — Partial device plan (THE #1 code item: the 4.90 ms doorbell tail, 9-16% of window)
Mechanism: when a layer-group has ANY missing expert, the resident plan aborts all-or-nothing
(`s_bad`, verify_kernels.cu:1141-1155) and the whole group takes the host round-trip — for a 2%
CPU-routed tail. The per-region machinery ALREADY EXISTS (counts[2]=0 / start2[0]=n layout,
pre-doorbell miss publish verify.cpp:1519-1526; the E-6 plan covers resident regions only).
Rewrite: publish a partial plan per region; only genuinely-missing regions wait on the host.
Exact arithmetic (routing math unchanged) → no MMLU risk class. One quiet-window reboot to land.

### T1.2 — R19c: fold the query rows into the attention block (the 6-conc ceiling)
The window's rows re-gather the SAME context KV independently — width growth 1.6→4.4 ms is pure
re-gather. One block loading the KV cell ONCE for all rows of the slot group: 6 rows × 12 KB sq
fits 228 KB shared. Halves the #1 GPU line (~2.2 ms/window at width 6, more at 8-12).
This is also what makes window width 8→12 (T2.5) affordable later.

### T1.3 — Deeper slot chains + the floor stair (the pair/M1 acceptance class)
Pair: slots verify ONE proposal today (`set_max_drafts(1)` generate.cpp:4736, `draft(2,…)`);
chain to 3 with the matched-prefix accept loop (~:10223). At 63-68% slot acceptance, a second
proposal banks ~0.6 extra rows/window nearly free.
M1: the serial chain stops at depth 3 under the flat min-p 0.5 floor while depth acceptance runs
96/93/90% (F-159) — stair 0.5/0.5/0.30 is offer-side only (serial verify = true rejection
sampling → distribution-neutral). Banked ≈ (1−a⁷)/(1−a): a=0.80→0.87 is +21% banked rows.

### T1.4 — hc read parallelism (the 4.2 ms line)
After T0.4/T0.5 pick the best block layout, the code form: fuse the hc down/up read into one
pass over the row group (QFUSE done properly — the F-187 rejection was the naive form), or split-K
across the 41-block starvation. Target ~1.5-2 ms.

### T1.5 — Emit on the device (1.87 ms)
Per-slot draft launches are host-serialized. A device-driven draft chain (the drafter's round
graphs launched inside/behind the window graph, gated by the commit event) removes the host from
the round trip. Composes with T1.3.

### T1.6 — Fenced admission (the churn hitches, 54.8-58.3 ms waves)
~14-30 cudaDeviceSynchronize per 6-slot admit wave (copy_to_slot/from_slot,
generate.cpp:10049-10140). Stream-ordered events + one batched save/restore → ~15 ms. Not in the
steady-state window, but it is the pair/conc jitter under real traffic.

### T1.7 — Graph pre-capture / sentinel slots (the layout churn)
176 re-captures per 5 boots from BATCH_MTP slot rotation; interim = pre-capture canonical
rotation subsets inside copy_to_slot; full fix = sentinel-slot bucketing (fixed slot count,
zero-length placeholders, ONE graph shape). Also the enabler for cheap window width.

### T1.8 — Exact two-phase head (521 MB/window lm-head read)
4-bit shortlist + full-scan only on disagreement (EXACT, unlike tqllm's shipped form).
~1-2% of window; small prize, cheap code.

---

## 4. TIER 2 — formulation/pathing rewrites

- **T2.1 Layered prefill** — layer-group interleaving across chunks kills the per-chunk
  expert-reload tax; up to 70% TTFT claims in the literature; prefill is priority metric #1.
- **T2.2 ECHO per-row verify budget** — stop drafting rows whose recent acceptance is cold;
  frees window rows for hot slots (concurrent metric).
- **T2.3 DuetServe look-ahead launch + launch-completion events + QKV-multistream-in-graph** —
  the 9.5 µs node-spacing is launch-path latency; overlapping the next window's launch hides it.
  (PDL alone measured no-op F-191 — this is the fuller form.)
- **T2.4 Conditional graph nodes** — mapped spin gates + E-6 plans already scaffold it; NO
  production engine ships conditional nodes today — genuine green field, kills doorbell waits
  inside the graph.
- **T2.5 Window width 8→12** — 1.9 GiB of 21.7 free; only after T1.2 makes width cheap.

## 5. TIER 3 — rebuild/training class

- **TQ3 rotation quants** (~11% more experts/GiB — softens every cache trade; rebuild).
- **~3-bit/trellis KV** (deep-context lane; 12→~4.5 KB/token).
- **DFlash2 port from tqllm** (gated on a sampled-mode tree-acceptance measurement FIRST —
  greedy-only claims don't survive temp 0.7).
- **FastMTP + KV-injection** (training tier, upstream pos-2 0.476→0.616 for our family).

## 6. Cross-cutting (do alongside, not instead)

- **RUN-2 STALL hunt** (one ~97-100 t/s collapsed run per boot; worth several % of every battery
  mean; suspects: expert-profile-save fsync, adapt refill, clock transitions; newest data point:
  it hit the OLDEST engine's FIRST run — F-197).
- **MMLU gates** where marked; single-stream protection law (concurrency-for-M1 trades are shelf
  variants, F-183 precedent).
- **User-side hardware blocks still open**: iGPU cable move (frees ~0.3-0.5 GB VRAM + removes
  compositor contention — top hardware task per F-193), GPU lock state check (F-190 verdict:
  locked-2835 mildly beneficial).

---

## 7. The stacked arithmetic (honest, with overlap)

| metric | banked vs stock | T0 battery | T1 stack | realistic landing |
|---|---|---|---|---|
| M1 | +12.0% | +3-7% (T0.1/T0.3/T0.7) | +8-15% (T1.3 stair, T1.1, T1.2/T1.4 window cuts) | **+25-35%** ✅ |
| pair | +7.6% | +2-4% | +12-20% (T1.3 chains, T1.6, T1.1, T1.2) | **+22-32%** ✅ |
| 6-conc | +2.7% | +2-5% | +15-25% (T1.1 4.9ms, T1.2 2.2ms, T1.4 1.8ms, T1.5 1ms, T1.7) | **+20-33%** (the stretch goal) |

Overlaps are real (cutting one stage shrinks the window the others are measured against); the
recoverable-pool arithmetic in §1 says the pool is deep enough IF the Tier-1 items land. Order of
execution: T0 battery first (days), then T1.1 → T1.2 → T1.3 (the three that carry the target),
each under the Lock Law: fresh-boot ABAB, 3+3 same-hour probes, means, MMLU where marked.
