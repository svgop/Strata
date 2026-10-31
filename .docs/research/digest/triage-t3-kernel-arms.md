---
type: digest
id: triage-t3
created: 2026-10-31
status: evidence
topic: T3 triage — pricing W5/W6/W9 acceptance items against THE NEEDLE (law 15)
scope: read-only triage; no code changed; prices every item: stage ms x improvement x lane occupancy
---

# T3 triage — W5 dense_tc / W6 QSA decode / W9 env arms, priced per law 15

Anchors (all sourced): solo window 11.3–11.7 ms @ 2.39–2.47 tok, GPU-reach 8.9 ms (ROADMAP baseline); wall ≈ 8.4+1.57×rows, ctx-free <72k (M15); knee +4.7–7.6 ms >80k (M20); stages: VRAM-hit GEMV 1.93–3.46, hc reads 2.39–4.00, attention 0.8–2.5 solo / 4.09 heavy, shared+quant 0.64–1.23 (F-218 map); dense ≈30% of GPU window; nsp=20 → 40 blocks on 340 capacity (12% fill), 20% empty blocks at width 65 chunks; i8v3 MMA 732 vs 233 (3.1×, prefill-only); dense_tc ≥1.0× only from T=4 (0.72–0.9× T=1, verify.cpp:1453 dtc_ok n∈[4,8] — source-confirmed); wide share: solo avg T 2.6, most windows T<4; pair 4–8 rows constant; conc 4.93 rows. Adopt bar ≥+4% target lane, no priority lane <−4%; drift band ±4–8%.

## Verdict table

| Item | Verdict | Priced needle (best-estimate window Δ) | Paying lane | Conf |
|---|---|---|---|---|
| W5.1 STRATA_DENSE_TC=63 export | KEEP* | 0 ms itself; gate = W5 microbench multiplier ≥1.2× at T∈{4,6,8} pre-leg, else NOT RUN | pair | M |
| W5.2 Wide-window share per lane | KEEP | instrumentation = the occupancy term of this math | all | H |
| W5.3 Grouped TC experts T≥4 | DOWNGRADE | 0.65 wide-share × ~2.5 ms expert stage × 15–30% ≈ +2–3% conc — inside band alone | conc | L |
| W5.4 MMLU 2x + mimic | KEEP | contract (output-order: fixed-order fp32 scale sum) | — | H |
| W5.5 3v3 battery pair/conc | KEEP | pair is the only lane that can clear +4% | pair/conc | H |
| W5.6 Posture + snapshot | KEEP | closure cost | — | H |
| W6.1 nsp width-derived + empty-exit hoist | KEEP | attention 4.09 ms heavy × 30–50% fill-fix ≈ +5–8% heavy; ~0 at shallow width (nsp=chunks there) | long-ctx | M |
| W6.2 q prescale hoist to plan | KEEP | per-row fixed term in 40 blocks ≈ 2–5% of attn stage; bitwise-plausible | long-ctx | M |
| W6.3 Multi-CTA resolve <<<1,RT>>> | DOWNGRADE→W15 | resolve ms unmeasured; nsys first; solo inside band if <0.2 ms | depth | L |
| W6.4 i8v3 port to decode | KEEP | knee-only: 3.1× MMA rate pays where MMA-bound (32K+); +5–12% at 80k+; inside band <8K | knee lanes | M |
| W6.5 LSE merge + gate fold | DOWNGRADE→W13 | 2 launches × 12 QSA layers ≈ 60 µs ≈ +0.5% — inside band standalone | all | H |
| W6.6 STRATA_GDN_SPLIT=1 sm_120 | KEEP | bitwise, env-only, gfx1151 precedent 2+ rows +3–13%, 1 row no gain; underfill 48/170 blocks worse on sm_120 → ≥ gfx gain; check 1-row case | solo 2+ rows | M |
| W6.7 Parity suite | KEEP | contract | — | H |
| W6.8 3v3 + long-ctx leg + MMLU 2x | KEEP | long-ctx leg IS the priced lane | — | H |
| W9.1 EXPERT_V2+TSUM export | KEEP-as-composite | re-priced 2–5% (see contradiction 2); kill line: composite <+4% → unset, record negative | all | L |
| W9.2 S26 dispatch confirmed live | KEEP | 5-min check inside the composite leg | — | H |
| W9.3 STRATA_LFUSE default-on | KEEP-as-composite | 48 layers × 1 launch saved ≈ 0.2–0.3 ms ≈ +2% standalone — inside band | all | M |
| W9.4 Bitwise parity expert stage | KEEP | contract (bitwise claim must be proven per-shape, #28111 n=5/8 divergence) | — | H |
| W9.5 3v3 battery + stage lines | KEEP | contract + falsifier; re-measure VRAM-hit GEMV + shared+quant lines | — | H |

*W5.1 conditional: multiplier <1.2× at T≥4 → items 1/3/5 collapse to KILL; row closed with pricing recorded (law 15).

## Needle math per row

- **W5 ceiling, computed from anchors:** wide-window share × 30% dense share × stage cut. Solo: 0.25 wide-share × 30% × 25% cut × 8.9 ms GPU-reach ≈ 0.17 ms ≈ **+1.5% best case** — the roadmap's "+1–3% solo" assumed deep windows are commoner than telemetry shows (solo avg T 2.6); solo is priced OUT (law 15: solo cannot justify the leg). Pair: 1.0 × 30% × 13–23% (1.15–1.3× stage) × ~12 ms ≈ **+3.9–6.9%** — clears bar iff multiplier ≥1.2×. Conc: 0.65 share → **+2.5–4.5%** — clears iff ≥1.25×. The ONLY measured number is "≥1.0× from T=4" — the actual multiplier at T=4–8 is unmeasured; `dense_tc_parity --bench-cases` reads it in minutes (law 4 cheapest falsifier). Donor doctrine supports real upside: llama.cpp int8-MMA MMQ all-M on Turing+ (mmq.cu L435), TRT-LLM exact-M≤8 tiles; our T19 mmvq tables hold 1.3–1.5 GB/s — headroom exists.
- **W6:** split path is live on trunk; nsp=min(2·SMs/(n_q·kvh),20,chunks) pins 20 → 40 blocks vs 340 capacity = 12% fill, splits 17–19 empty, split 16 one chunk (source-confirmed :2307–2309). Fix direction is MORE splits (width-derived → ~130 blocks), corroborated by the FA3 seq-aware-split paper: low-head-count decode (exactly our n_q=1, 2 kvh) recovers 21–24% kernel efficiency. Short-ctx windows already run nsp=chunks (no waste) → this is a wide-width arm, scope per law 4b. **Depth-knee term priced separately:** M20's +4.7–7.6 ms/window above 80k is attention-dominated; W6 retune (1.1+1.2+1.4) cuts the attention share 25–40% ≈ 1.2–2.6 ms ≈ **+6–12% at 80k+**; at the standing 60–76k session ≈ +3–7%; <8K inside band. i8v3's 3.1× MMA rate converts only where the kernel is MMA-issue-bound (long ctx); falsifier: MMA-bound fraction in qsa_rows_bench at 32K/64K <40% → inside band → don't port. GDN split: env-only, bitwise precedent on sibling arch; sm_120 underfill is worse (48 blocks / 170 SMs = 28%), so gfx1151's +3–13% (2+ rows) transfers favorably; solo windows at 2.4–3.6 rows are the paying case; gfx "1 row no gain" means near-zero for 1-row windows.
- **W9:** stage share is real (VRAM-hit GEMV 1.93–3.46 ms, largest stable line) but the improvement fraction is the contested term — see contradiction 2. Composite (EXPERT_V2+TSUM+LFUSE, all bitwise, one battery leg, bisection by env-unset on regression) prices **+2–5%**: straddles the bar. It survives law 15 only on the gfx1151 same-class measured precedent (law 12: production precedent on exact component class) and because the leg costs hours and 3 envs. Kill line enforced.

## Regression risks + guardrails + remediation (law 13)

- **W5 dense_tc / W6 i8v3 / W6.1 nsp: output-order changes** — fixed-order fp32 scale sum; nsp moves split boundaries (fp non-associativity through the LSE merge). The row record assigns MMLU to i8v3 only; **W6.1 also needs parity + MMLU** — gap flagged. Guardrail: MMLU-Pro 2× (floor per doctrine 93.3 — see contradiction 5) + mimic + 3v3. Remediation: scope (n∈[4,8] gate already is the scope; law 4b structural fallback: raise gate to n≥5 if T=4 mixes regress), compose, or rebuild narrow-J tiles (llama.cpp per-type J-width 32/64/128 + per-type crossovers Q3_K 4/5, Q5_K 5, Q6_K 4, IQ4_XS 5).
- **smem bounds sm_120 (#24399 precedent):** upstream crashed OOB in exactly the MMQ MMA write-back epilogue on 5090, plus IQ_K garbage (#28784). i8v3's 3-way split adds operand planes to a 44.7 KB kernel (2 blocks/SM — one plane over 48 KB halves occupancy = regression). Guardrail: static smem audit + dry-run epilogue bounds before unmask. Remediation: scores-only i8v3, register-staged planes, or abort to retune-only.
- **Toolchain (W9/#28581):** S26Split<21>/S26IQ3S use the u32-byte-load pattern nvcc 13.2 miscompiles; box builds 13.4 (safe). Guardrail: pin toolchain check in the parity harness; belt-and-braces `__byte_perm` conversion.
- **Bitwise claims (W9, W6.6):** bitwise on gfx1151 ≠ bitwise on sm_120 — run per-shape parity per #28111 (n=5/8 MMVQ batch divergence); GRP_NC/RPW every configuration.
- **Cold path:** dense_tc pays its first wide window post-boot (graph capture — W11 prewarm owns); i8v3/retune change capture-relevant kernels → cold-3v3 check per W11 discipline; env arms (W9, W6.6) have none.
- **Single-stream protection:** M1 must not regress for ANY change (campaign rule 4) — W9 composite and W6.6 are solo-lane-positive by pricing; W5 is pair-scoped so M1 exposure is the T<4 exclusion gate (already structural). MMLU law: anything token-stream-affecting gates 2×.

## Ordering by t/s-per-effort + composition

1. W6.6 GDN split (env-only, bitwise, hours, precedent). 2. W9 composite leg 1+2+3 (env-only, bitwise, hours, one battery leg — the mission's law-12 citation maps to re-pricing; composition doctrine is law 4b + F-270). 3. W6.1+2 retune (days, long-ctx scoped). 4. W6.4 i8v3 (days, knee-only, smem-audited). 5. W5 microbench falsifier → W5.1+5 pair/conc battery (only if ≥1.2×). 6. W5.3 grouped experts (after 5 reads). Deferred: W6.3→W15, W6.5→W13 fusion stack. One session, four battery legs sharing the 3v3 harness: A=W9 composite, B=GDN split, C=retune, D=i8v3 — each still one variable per leg.

## Contradictions flagged

1. **W9 digest "already armed on CUDA / premise stale" — REFUTED**: arch_defaults table sits behind `gfx_arch_is_gfx1151` (:13/:32), CUDA builds never consult it; nv_defaults arms only sm_86/sm_60. Adjudicated in the row ALIGN; my source read confirms. Arm items stand.
2. **W9 digest "expected delta small (a few %)" vs roadmap "+6–10%"**: mechanism physics sides with the digest on sm_120 — 2 KB iq3s_grid goes L1-resident after first touch; LDS ≈ L1 latency on Blackwell; IQ4_NL side is __byte_perm-emulated (near-zero LUT traffic). The +6–10% was gfx1151 transfer. Re-priced 2–5% composite; the nsys read of the expert stage (item 5) settles it for ~90 minutes.
3. **W6 digest internal: "production formulas land 1–3 splits, not 20" vs our regime** — those formulas assume multi-head batch; at n_q=1/2 kvh the same formulas + the FA3 paper say MORE splits (occupancy-fill). Direction for W6.1: width-derived up, not down.
4. **M15 (window ctx-free <72k) vs stage profile "attention grows 0.8–2.5 ms with ctx"**: resolved by M20 — growth is absorbed below ~72k, bites above.
5. **MMLU floor: doctrine 93.3 (AGENTS.md) vs roadmap protocol 92.9** — pick one before the first output-order leg; doctrine outranks roadmap.

## Open questions

- Dense-stage multiplier of dense_tc at T=4/6/8 on the real Q6_K/Q5_K/Q4_K/IQ4_XS mixture — minutes via existing harness; gates the whole W5 row.
- resolve_kernel per-window ms (W6.3) — one nsys pass; routes the item to W15 or kills it.
- Whether conc windows below 4 rows are common enough to drag W5 conc under the bar — the W5.2 instrumentation answers it.
- i8v3 decode accuracy vs the int8-KV dequant baseline (no surveyed kernel does multi-byte-split MMA; formulation is novel — QServe is the family).

## Implications

Window wall time is the per-token sink (M2: −10% window ≈ −10% lane t/s); all 19 items are decode-window moves — TTFT untouched (W7/W14 own it). Highest certainty×effort: W6.6 then W9 composite. The row records' measured expectations need two downgrades (W5 solo, W9 absolute) and one scope-up (W6 long-ctx), all recorded here so the battery legs read honest kill lines. Nothing here is adoptable unmeasured: every KEEP carries a falsifier that runs before or with its leg.

## Sources

- AGENTS.md laws 12/13/15 + campaign rules 2/4/4b (adopt bar, no-terminal-trade-offs, needle) — /home/svg/LLM/strata-custom-2/AGENTS.md
- .docs/research/digest/w5-dense-tc-smallM.md, w6-qsa-split-attn.md, w9-s26-expert-kernels-sm120.md (2026-10-31, source-read digests)
- /home/svg/LLM/.zcode/rich-tracking/strata-perf-roadmap.board.json rows w5-dense-tc, w6-qsa-i8v3, w9-s26-expert-kernels
- campaign/MEASURED-opt-map-2026-10-27.md M2/M3/M7/M15/M20
- campaign/ROADMAP-perf-8moves.md §W5/§W6/§W9/§W10/§W12/§W13 + protocol + baseline
- Source spot-checks (this pass): src/core/verify.cpp:1447–1457 (dtc_ok gate), src/kernels/cuda/qsa_prompt_attn.cu:2304–2312 (nsp), src/core/arch_defaults.cpp:13–52 (gfx1151 guard + arm table)
- Upstream precedents cited via digests: llama.cpp #24399, #28784, #28581, #28111, #28823, #28572; FA3 seq-aware split arXiv 2604.00028; TRT-LLM XQA/low_m_gemm; FlashInfer decode.cuh

Methods note: unproductive — no per-lane T-histogram exists in campaign/results (W5.2 item supplies it); i8v3 MMA-bound fraction at width unmeasured anywhere; gfx1151 absolute S26 gain figures live in aurora_s23.md (not read this pass — would sharpen W9 pricing one notch).
