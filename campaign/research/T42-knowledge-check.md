# T42 — The "Strata Core Knowledge v2" external document, checked against this tree (2026-10-24, findings-only, no boots)

User delivered a ~1,200-line external synthesis ("v2") built partly on our own campaign reporting
(F-276/F-278/E1-E3/C2a/the 323 record are cited correctly) and partly on 2026 frontier research
(Qwen3.8-Flash-Next, DeepSeek Engram/V4.1-Flash/DSpark, MiniMax MSA, LongCat). This file is the
verdict: what it gets RIGHT, what is STALE, what is WRONG for this engine, and — the actionable
core — which of its levers already exist here versus which are genuinely new. Companion maps:
[[T42-r1-miss-path]] (miss path), [[T42-r2-gpu-chain]] (GDN/QSA/HC), [[T42-r3-ple-draft]]
(PLE+draft). R4 (VRAM ledger) died on a provider-network failure; its essentials were recovered
from boot logs + R1 and are folded in here.

## 1. Headline claims, verdict at a glance

| Doc claim | Verdict | Evidence |
|---|---|---|
| Model is Qwen3.8-Flash-Next (v1's Qwen3-Next assumption wrong) | **RIGHT** | pack index: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`; output.weight 2560x248320; 512 experts/layer |
| Geometry: 2560 hidden, 248,320 vocab, 512x top-10, 36 GDN + 12 QSA, PLE at layer 2, MTP 1-layer | **RIGHT** | native_experts.txt (47 layer rows, per-layer quant types), r2/r3 anchors |
| "Binding constraint = DDR bytes of host-resident experts per verify pass" | **HALF RIGHT — binds the TAIL, not the median** | live hit-rate distribution n=4,192 windows: p50 97.2%, p10 88.7%; misses ≈8% of the median ~15.6 ms forward, ≈31% at p10; the 3-6-position verify band hits only **71.0%** (U(P) law real); warmed conc windows back at 97.1% |
| Expert at IQ3_S ≈ 2.11 MB | ~RIGHT | measured avg 2,046,240 B (50,292,326,400 / 24,576); 516 experts/GiB |
| "~21-24 GB for experts ≈ 42-47% resident" | **WRONG for us** | ~13.3 GiB VRAM experts (~6,520, 26.5% by count) + 42.29 GiB page-locked RAM complement at boot (ALL experts resident somewhere; file tier only first-touch, ~5-6K reads/boot decaying to 0) — plus profile-ranked placement + prediction, which is why 26.5% count yields 97% hits |
| Solo ≈ 220 t/s ≈ 14 ms/forward, GPU floor ≈ 3 ms | RIGHT (self-consistent) | 222.4 M1 record; acceptance 74.9% → E[tok/verify]≈3.46 → ~15.6 ms/fwd measured-compatible |
| "Prefill misses compute on CPU — GPU shipping is the unpriced prefill lever" | **WRONG (already built)** | chunks >=1024 tokens stream weights into VRAM ncmoe-style by default (prefill.cpp:179,2864; R1 Q4); our `--prefill auto:32768` always takes it |
| "The draft shares the 248k lm_head (~521 MB/step)" | **WRONG (already trimmed)** | 40,525x2560 Q4_0 shortlist = 55.7 MiB/step (mtp.cpp:248-266, rt/draft_vocab.bin; R3 Q5) |
| "Draft MoE can miss to DDR — constrain its routing" | **MOOT** | all 512 draft experts hard-resident VRAM, 675 MiB, no offload tier (mtp.cpp:341-363; R3 Q6) |
| "PLE/engram: lock in RAM, 4-bit, dedup, sample-time gather, conv as state, reuse draft→verify" | **ALL ALREADY BUILT** | gather at sample time (layer.cpp:1603-1625), IQ4_NL 90 B/row (= the 4-bit floor, 28.8 GB), page dedup + prefetch-skip + row cache, NG_HIST=9 rolled back on rejection + parked (R3 whole) |
| "PLE disabled in draft" | **BACKWARDS** | the draft actively prefetches PLE rows for verify (mtp.cpp:1652-1666) — LongCat's reuse pattern already wired |
| "GDN decode = 48 CTAs = 28% of SMs — row-split it" | **RIGHT observation, FALSIFIED remedy** | verify_kernels.cu:1215 grid (48,1); BUT `STRATA_GDN_SPLIT=1` already tested — "neutral on sm_120, NOT adopted" (journal T1a) — the kernel is not SM-starved in the window path |
| "GDN state fp32, 108 MiB/stream, 216 MiB/token traffic" | **RIGHT** | session.cpp:43-45,106; fp32 only, no bf16 mode (R2 Q2); L2 set-aside (60 MiB) exists as partial mitigation — OUR arm since B8 |
| "Verify pays K+1 sequential passes over GDN state" | **WRONG** | one kernel/layer loops all tokens in-kernel, state in registers, verify writes nothing (R2 Q3) |
| "HC adds ~96 small kernels/forward" | **UNDERCOUNTS 3x** | ~288-291 launches/forward (2 reads/layer x 3 launches each + head + final; R2 Q6) — makes fusion worth MORE than the doc thinks |
| "QSA is pure overhead below ~2051 context" | **PARTLY RIGHT** | the 2051 bound is the codebase's own; selection degenerates to identity BUT 3-4 indexer launches still fire per layer per step — no host-side bypass (R2 Q4). This is a REAL, cheap, exact-output skip |
| "Miss path costs 30-80 µs/layer in handshakes" | ORDER RIGHT, NOT BINDING | ~10-40 µs/layer pure overhead (doorbell + polled flags + batched DMA — R1); real per-fetch wait ~0.2 ms = bytes/compute dominate 10x |
| "Conc clusters ~295/~315 = hidden discrete mode; pin the good mode" | CONSISTENT with our bimodality log | unfalsified; the PDL-race variant is testable cheaply (doc §11.3) |
| 128 GB RAM capacity edge | **STALE** | this box has 184 GB (110 available with the server up) |

Stale-on-arrival (our campaign moved past it): acceptance battery DONE (τ0.15 falsified, SPEC_PROB adopted F-282); template REPLACED (F-283, user's protocol template); E1 DONE (24 slots, 0 evictions); PDL PINNED =0 (F-279); "no >=4% in-tree prefill lever" WRONG (T39: dedup-to-T-rows +460 slots / T15: bit-exact ~4-5%).

## 2. What the doc proposes that ALREADY EXISTS here (inventory)

| Doc item | Our implementation | Anchor |
|---|---|---|
| §6.C predictive expert streaming / pre-gating | RouterLookahead thread: recomputes next-layer router on CPU, warms predicted misses; io-prefetch with confidence admission + byte budgets; recall/precision telemetry at exit | expert_source.cpp:2938-3010, 812-898 (R1 Q5) |
| §6.D GPU-only prefill via expert streaming | default for chunks >=1024 tokens (pinned slot ring, GPU computes) | prefill.cpp:179, 2864 (R1 Q4) |
| §6.B doorbell handshake | mapped-memory doorbell (device writes ring, host spins), device-side flag polling, batched DMA (STRATA_DMA_BATCH), dm_wait_combine fold | verify.cpp:2083-2094, 2232-2315 (R1 path map) |
| §6.E draft-head vocab trim | 40,525-row Q4_0 shortlist + VRAM-adaptive subset (#474) | mtp.cpp:247-266 (R3 Q5) |
| §6.H L2 persisting window over GDN state | STRATA_L2_SETASIDE_MIB=60 standing arm | generate.cpp:9300-9305 (R2 Q2) |
| §6.J two-batch overlap (verify level) | STRATA_COHORTS=2 (B pool under A window, F-233 +24.8% conc) | standing |
| §7.4 n-gram routing prior | the 149-prefetch vote-ranked predictor (softmax-anchored, admission bars) | expert_source.cpp:2837-2990 |
| §7.10 draft-first residency | all draft experts + head hard-resident | mtp.cpp:341-363 |
| §5 engram design items 1-8 | all built (see verdict table) | T42-r3 |
| Residency by measured frequency | expert-profile learned + persisted (`--expert-profile*`), boot-ranked fill, adaptive EWMA swaps (ADAPT_DECAY 0.85, DORMANT 64), thousands of exchanges/boot | expert_cache.cpp:471-660; boot log |
| §6.X KV quantization | int8 KV standing (`--kv int8`, 99.7-99.9% VRAM streaming hit) | config; log |
| DDR/PCIe coordination | PcieAuto + online per-layer share regression + io budgets from measured rates | generate.cpp:2229-2240; expert_source.hpp:395-435 (R1 Q7) |

**The pattern:** the doc's "moves" describe, almost without exception, systems this campaign
already built. Its physics formulas (miss law, U(P), barrier tax, wave quantization, ski rental)
are the durable import — they EXPLAIN why our adopted arms work and how to price the next ones.

## 3. Genuinely NEW levers the doc contributes (all falsifier-first)

1. **QSA below-budget launch skip** — host-side: when n_kv <= width (2051), skip the indexer
   append/query/score/top-k launches (selection is the identity); still store index keys.
   Exact-output, ~36-48 launches/forward removed at chat contexts. NOT in tree (R2 Q4).
   Falsifier: bit-exact parity + 3v3; value = launch-gap x count (waterfall to price).
2. **GDN state bf16 storage** (KL-gated) — no bf16 mode exists (R2 Q2); halves the 216 MiB/token
   access volume and doubles L2 reach with the existing set-aside. Falsifier: KL + long-ctx
   needle + 3v3. (T17's "bf16-native dots" tranche is the same family, banked.)
3. **Miss-aware verify WIDTH** (§7.1, DSpark-style) — targets the measured 3-6pos/71%-hit band
   exactly: cap verify width when predicted unique-miss count is high. Scheduling change only.
   Falsifier: wall-per-accepted-token vs fixed K on identical prompts.
4. **Temporal/adjacent index reuse** (§7.6/7.7) — approximate, KL-gated, only worth it past 2k
   context; low priority here.
5. **ple-io ram** — one-flag: mlock the 28.8 GB PLE table (110 GB free). Robustness (removes
   page-cache dependence under pressure) more than speed. Zero-code.
6. **PDL race screen** (§11.3) — 20x greedy same-prompt with PDL on/off under contention, count
   non-identical outputs; the bimodality hunt's cheapest probe.
7. **Popularity-weighted per-expert precision** (§7.2) — NOT already done: the pack's
   heterogeneity is per-LAYER (47 distinct quant-type rows), not per-expert. Re-quant-class,
   USER-GO-gated, T17-banked recipe applies.

## 4. Corrected physics for THIS box (the numbers that price the next moves)

- Expert blob 2.046 MB avg → **516 experts/GiB**; +1.9 pt hit-rate per +460 slots (T39 measured
  pricing) — the marginal-GiB lever is real but small at 97% median hits.
- Miss economics at median: ~0.2 ms per fetch wait (log: 1477.8 ms / 7504 fetches); pool drains
  34-44 GB/s measured; PCIe share 0.35 by config.
- The forward's ~15.6 ms splits (doc model + our instruments): GPU floor ~3 ms, miss term ~1.2 ms
  median / ~4.8 ms p10, handshakes ~10-40 µs/layer, remainder = the GPU kernel chain (288-291 HC
  launches, 12 QSA indexer launch-quads, MoE groups, drafter) — **the chain, not the bytes, is
  the median binding term on this box**, which is exactly what T15 (hc occupancy, bit-exact
  +8-9%) and T17 (hc ALU-cap mma tranche, +9-17% ceiling) already priced.
- Conc: U(P) inflation is real but the cohort admission + prediction recover it (97.1% on wide
  windows); the conc binding term stays the cohort round structure (E5 ternary, C2a family).

## 5. The merged breakthrough queue (value-ranked, falsifiers named)

Tier A — banked, measured, ready (the record-breakers):
- **A1. hc grid occupancy** (T15: UPM_COLS 16→8 + 82-block staged down; fused_gr.cu:361;
  BIT-EXACT +8-9% M1) — solo record lane. The doc's wave law independently confirms the class.
- **A2. E5 ternary 3+3+3 reactor** (band 355-390 vs record 323.0) — conc record lane.
- **A3. FastMTP retrain T14B** (GO-banked) — acceptance lifts every lane.
Tier B — new from this check, cheap first boots:
- **B1. QSA below-budget skip** (exact-output; biggest launch-count cut available).
- **B2. GDN bf16 state** KL-gated (R2: no mode exists; T17 tranche overlaps — run as one arm).
- **B3. ple-io ram** one-flag robustness boot.
- **B4. PDL-race screen** (20x greedy) + bimodality histogram — settles §11 cheaply.
- **B5. +460-slot prefill dedup** (T39: +1.9 pt hit, bit-exact class).
Tier C — bigger builds (specced, after A/B):
- **C1. TC-hc mma tranche** (T17 +9-17% ceiling; R2's 288-launch count raises its value).
- **C2. Miss-aware verify width** (targets the 71% band).
- **C3. Two-batch overlap on separate streams** (beyond C2a; doc §6.J generalized) — after E5.
- **C4. Per-expert popularity re-quant** (§7.2; USER GO required).

Standing riders unchanged: E2/E3 park pathway, E7 idle scan, A1 diagnostics, drop-tests
(RESCOPED: PREDICT_*/POOL_SPIN_US are now TUNE-tests, not removal tests — they run the doc's
"predictive streaming" tier), SPEC_PROB pooled rider, five-boot re-bank of the full current
recipe (dm+template+slots+PDL-pin+SPEC_PROB) to certify the new standing baseline.
