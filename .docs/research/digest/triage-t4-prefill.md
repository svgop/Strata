---
type: digest
id: triage-t4
created: 2026-10-31
status: evidence
topic: W7 prefill — per-item needle pricing vs law 15 (row w7-prefill-sync, 9 items)
---

# T4 triage: W7 prefill row vs THE NEEDLE (law 15)

Bar: targeted ≥+4%, no priority metric beyond −4% (law 2); drift ±8% across hours (law 1);
priced-in-band-everywhere = NOT RUN → compose (law 4b) or drop. Remediation = scope/compose/rebuild (law 13).

## Verdict table

| # | Item | Verdict | Needle term | Priced Δ | Conf |
|---|------|---------|-------------|----------|------|
| 1 | MMQ device grouping, delete sync prefill.cpp:3741 + counts 3758-3765 | KEEP (scope-corrected) | TTFT tail + fixed tax, sub-1024-chunk reads only | +1-4% blended prefill; +7% class precedent on covered layers | H mech / M size |
| 2 | MMQ host path deleted ENTIRELY | DOWNGRADE | none (hygiene) | 0% | H |
| 3 | GDN prep(i+1)‖scan(i), scratch 2× (kernels.cu:1174-1215) | KEEP (price-first) | marginal rate (GDN layers' share) | unmeasured; = prep_t × GDN share; falsifier free | H struct / L size |
| 4 | cuBLASLt sm_120 top-K autotune (prefill.cpp:1286-1349) | DOWNGRADE (rider) | none steady-state; cold first-read only | 0-2% steady; moves a one-time HIP-measured 1.0-1.2 s library load (CUDA cost unmeasured) | H bound |
| 5 | KV host-mirror block-DMA on CUDA (kv_q8.cu:72-80) | KEEP (price-first) | marginal rate, all bulk chunks | +1.5-3%/32k-chunk (4-8× PCIe-TX amp on 384 MB/chunk); falsifier free | M |
| 6 | Staging pool double-buffer (prefill.cpp:2836/3263) | DOWNGRADE (fold into 5) | marginal rate | ≤1-2%: inline stage ≈16-32 ms per 32k chunk (0.3-0.7%) | M |
| 7 | Battery t42 8K/32K/64K 3v3 | KEEP + scope fix | gate | n/a — lanes must add 512/9k-tail/cold-first | H |
| 8 | MMLU 2× + KV parity suites | KEEP | gate (law 3/10) | n/a | H |
| 9 | W19 small-read fast path (<2k) | KEEP, gated W8→W11 | TTFT tail (fixed tax) | residual after W8/W11 only; vs 2k/1.03 s the W8/W11 share may be the whole fixed term | H gate / L residual |

## Needle math (per item)

**Fixed-tax anatomy first (corrects the anchors):** M22's own three points give fixed = wall − T/6,820 =
1.03−0.30 = 0.73 s (2k), 1.89−1.17 = 0.72 s (8k), 5.34−4.63 = 0.71 s (32k) — the fixed term reads
**~0.72 s, not the "1.0-1.3 s" the anchor text carries** (that band needs a non-constant marginal below 8k;
unproven). Of the 0.72 s, M1's admission machinery (reuse-scan + park/restore + join, 317-934 ms; scan
median 73→313 ms with ctx, M23) is W8/W11-owned and may cover most of it. Price against 0.72-1.3 s, biased
low. Marginal ceiling 6.8k tok/s is PCIe-RX-capped (31 GB/s @ 475 W, M17) — no GEMM/launch item moves it;
only bytes-removed or overlap items do.
1. **Sync/group:** fires only where `T < stream_all_min()` = 1024 (prefill.cpp:179) — tail chunks
   (9k read = 8192+808) and tool-loop reads <1024; the 2k/8K/32K/64K lanes run the FUSED device path today
   (STRATA_PF_FUSED default-on sm120 ≥80 SMs, moe_fused_iq.cu:1409-1421), so Δ≈0 there. Δ = per-layer drain
   bubble × MoE layers × sub-1024 chunks; precedent +7.0% 4K / +5.2% 20K / +3.4% 98K (RTX PRO 5000,
   fused-vs-MMQ, same file) is the ceiling for full replacement, not tail chunks.
4. **Autotune bound (honest):** (i) heuristics queried once per shape, top-1 of 4 fetched and cached
   (gemm.cu:591-651 algo_cache) — benchmarking adds no steady dispatch; (ii) cuBLASLt host dispatch ~5-20 µs
   × ~15 shapes × layers ≈ ≤1% of the 0.72 s tax — dense-GEMM launch/dispatch is a negligible fixed-tax
   share; (iii) battery chunks are wide-M (the 10-30% miss is narrow-M/decode); (iv) marginal is PCIe-capped.
   Real needle = extending warm_gemm_libraries to CUDA (today HIP-only, CUDA branch `(void) device`): moves
   a one-time library load (HIP: 1.2/1.0 s BF16/FP16) off the first prompt. Steady-state 0-2% → NOT RUN alone.
5. **Block-DMA:** append writes host mirror per-cell (1-4 B stores) → 32 B PCIe transactions ≈ 4-8× amp on
   12 QSA layers × 1 KB/cell = 384 MB/chunk@32k → 1.5-3 GB effective ≈ 75-150 ms at ~20 GB/s TX = 1.5-3%
   of the lane, scaling with chunk size. Unmeasured amp — the one number to buy.
6. **Staging:** "~134 MB/layer" is bytes, not time; time = p0-prefix bytes / RX rate, 16-32 ms per 32k chunk
   shared with expert-fetch RX. STRATA_KV_PREFETCH (layer-ahead, kv_copy stream, prefill.cpp:2575-2581)
   exists default-OFF — arming it is the free falsifier for the whole staging family.
9. **W19:** 2k read = 0.73 s fixed + 0.30 s compute; the fixed term is co-owned by W8/W11 admission
   machinery. Fast-path addressable share = fixed − (W8/W11 landings), unknowable today; acceptance
   metric prompt_ms is already visible per request. Gate stands, now with math that justifies it.

## Regression risks + guardrail + remediation

- **1:** cpu-share auto (default-on <1024 chunks, one CUDA GPU) NEEDS the per-layer sync to read its
  GPU-time events (prefill.cpp:3741 region); killing it can re-price small chunks −15-20% (in-code 5090
  note: 600-tok 1,528→1,296 ms with share on). Guardrail: device-timed events replace the sync's
  measurement, or cpu-share re-scoped off where it loses (law 4b). KV parity untouched (routing, not
  values). Remediation: scope to the old MMQ regime; env gate per arm.
- **2:** MMQ is load-bearing OFF this card: peer and layer-split helper paths run MMQ (prefill.cpp:3560s;
  fused not on multi-GPU prompt paths), plus uncovered formats (IQ1_M/FP16). Delete with no fallback =
  binary-only rollback. Guardrail: delete only after the device path carries every mmq_plan consumer;
  snapshot engine/strata.pre-w7 first; sequence LAST.
- **3:** scratch 2×25.6 = 51.2 MB vs vram_free 1,244 MiB (M12) — fits; same kernels, same order →
  bit-identical. Guardrail: GDN lane parity + STRATA_GDN_CHUNKED tests.
- **5:** staged-first-block completeness invariant documented in-code (#579/#613). Guardrail: kv_q8_parity
  + kv_steps_parity + kv_stream_parity suites, env-gated STRATA_KV_HOST_DMA on CUDA.
- **9:** bypass must preserve append semantics + conversation-cache checkpoints; strict <2k threshold,
  watch bulk TTFT for leakage. Remediation: threshold scope (law 4b).
- **4:** boot-time top-K timing lives on a side stream/handle like warm_gemm_libraries (zero startup
  cost); keep heuristic fallback on empty timing.

## Confidence + falsifiers (cheapest first, law 4)

1. **Session log chunk-size histogram + STRATA_PREFILL_TIMING=1 boot** (kPfHostGroup marks exist): prices
   items 1, 5, 6, 9 in ONE diagnostic boot — sub-1024 chunks rare or staging marks <1% of chunk wall → items 1/6 drop to rider status.
2. **nvidia-smi PCIe-TX during appends** (one 32k read): confirms/refutes item 5's 4-8× amp bound;
   **two CUDA events prep vs scan** (one boot) price item 3 — keep only if prep ≥10% of GDN chunk wall;
   **STRATA_KV_PREFETCH=1 leg** (env-only) upper-bounds item 6's residual prize.

## Order by t/s-per-effort + composition

1 → 5 → 3 → (9 after W8/W11) → 6 → 4 → 2. Items 7/8 wrap every promotion.
**Composition (laws 4b/12):** one candidate binary stacking 1+5+3 (each env-gated default-off), promoted
once via `campaign/binary-ladder.sh` ABAB so the stacked step clears the ±8% drift band; lanes 512-tok /
2k / 9k-with-tail / 8K / 32K / 64K + cold-first-read; standard lanes + MMLU 2× + 3 KV parity suites; one F-entry with the pair table.

## Digest-vs-telemetry contradictions

- **C1 (battery design):** row expected "prefill +10-25%" priced as if MMQ ran bulk prefill; on the
  standing recipe the 8K/32K/64K lanes run the fused device path — the spec'd battery cannot see items
  1/2/4; add sub-1024 lanes.
- **C2 (2k-read attribution):** the board ties small-read slowness to the MMQ sync; a 2,051-token auto
  chunk is ≥1024 → fused. The 1,207-2,236 tok/s population is admission+prefill machinery (M1/M9), not
  the sync; the sync's true regime is tail chunks and explicit small chunks.
- **C3 (scout overreach):** w7 scout's "deleting our MMQ host path is parity" holds across vLLM/SGLang/
  DeepGEMM/TRT-LLM/FlashInfer but NOT in-tree: peer/layer-split helpers still require MMQ.
- **C4 (bytes≠time):** item 6's "~134 MB/layer" framing inflates a ≤1-2% time term. **C5 (fixed-tax
  value):** anchor says 1.0-1.3 s; M22's own points compute to ~0.72 s — use 0.72-1.3, biased low;
  re-fit with STRATA_PREFILL_TIMING before pricing W19's residual.

## Open questions

Sub-1024 chunk frequency in real traffic (log-minable today) · CUDA one-time cuBLASLt load cost ·
prep/scan time split · PCIe-TX amp on per-cell mirror writes · can device-timed events carry cpu-share's
controller without the sync.

## Sources

prefill.cpp:179,1281-1349,2575-2581,2832-2860,3255-3300,3560s,3741-3765,4597 · moe_fused_iq.cu:1332,1409-1421 ·
moe_fused.cu:710-726 · kernels.cu:1174-1215 · gemm.cu:591-651 · kv_q8.cu:60-84 · kv_stream.cu:276-285 ·
custom-iq3_s-8081.json · generate.cpp:695-708 · board row w7-prefill-sync · MEASURED-opt-map-2026-10-27.md
M1,M6,M9,M10,M12,M16,M17,M22,M23 · ROADMAP-perf-8moves.md §W7/§W14/§W19 · digest/w7-prefill-moe-autotune.md ·
AGENTS.md laws 1,2,3,4,4b,9-13,15. Methods: no per-layer-count constant found (geometry constants only);
layer count left as ~12 QSA + GDN mix from W15 sizing, not load-bearing.
