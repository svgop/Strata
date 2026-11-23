# LEVERS 2026-11-20 evening — the far-higher-t/s investigation: the wall as one equation, the coverage correction, the priced queue

Method: read-only investigation from the F-411→F-418 recalibrated map + targeted code reads
(this session's sites cited). No boots, no builds. Priced per law 15 (the needle named before
the leg runs); falsifier-first per law 4; scoping per law 4b; capability envelope per law 16.

## 0. The wall (the map every lever prices against)

Solo M1 window ~13.6 ms at load 2-3: GPU-reach idle **8.79 ms at 3.10 rows** vs **0.51 ms at
4.88 rows** (same boot, F-414); CPU experts 2.85-3.27 ms; adapt wait 0.76 ms quiet / **1.19 ms
under load** (live anatomy line); emit 0.35-0.61. The kernels are at 95-98% of the measured
stream anchor (F-411) — they are not the lever. **The wall is one equation: host-stage
exposure × (1 − rows in flight).** Every high-value lever either adds rows (depth, acceptance),
shrinks a host stage (pool granularity, swap traffic), or moves a host stage under the GPU
window (lookahead). The lanes are the same wall at different row counts: the batch anatomy
(4.68 rows) still shows reach idle 1.33 + CPU experts 2.98 + adapt 1.19 — the conc lane is
gated by the same host stages.

## 1. Tier 1 — the +10-25% class (rows × host stages)

**A. Draft depth-2 re-test (armed, mechanism live).** STRATA_BATCH_DRAFT_DEPTH=2: round 1
+6.8% M1 (legs peaked 239.4), M3b canary QUIET (F-416). The ladder is named: quiet window,
five-boot means, STRATA_SPEC_DEPTH published on both arms so position-1 acceptance prices D=3
in the same battery, MMLU ×2 on adoption. The ceiling is the wall itself: rows 2.4→3.0+
converts the 8.79 ms idle. Cheapest big lever on the board — env only, no build.

**B. Lookahead overlap of CPU-miss experts (ceiling ~+21%).** RouterLookahead already runs
host-side per layer and its enable_stats/stats_score already score every prediction against
what the next layer really routes (recall bands at top-k, k+4, k+10). The falsifier is one
boot: STRATA_LOOKAHEAD_STATS=1 + probe leg. High recall → the third consumer (predicted-miss
jobs handed to ExpertPool as speculative work; epoch-claim expires wrong predictions naturally)
moves up to the whole 2.9 ms term under the GPU window. Low recall kills it free.

**C. Pool-granularity surgery (new pricing this session).** The mechanism, read at
src/kernels/cpu/pool.cpp:1040-1051: `run_split` = phase1 → **serial `act_quant_q8_1` barrier
on the service thread** (line 1050; every worker parked between phases) → phase2, ~48
per-layer calls/window, ~1 miss each: the two-phase machinery plus the barrier dominates the
GEMV — 65 µs/miss vs the ~26-30 µs DRAM floor (2.2-2.5×). Arm: fold the activation quant into
the phase-1 task's epilogue (the worker that wrote `ff` quantizes its own `a2`; the barrier
dissolves) or batch the quant loop across the pool. Halves the miss path → ~1.35 ms/window
exposed (F-414's waitCPU) → ~+10% class. **Composes with B**: C halves the term, B hides the
remainder.

**D. Context n-gram draft source (never attempted; T47 §3.7).** Prompt-lookup-style second
proposer spliced into the coupled chain; token-substitution form, lossless by construction.
The falsifier is zero-boot: a hash-lookup hit-rate probe over harvest/operator traces. +5-25%
on copy-heavy real traffic, 0% on open-ended probes — the scoping law names the arm-condition
(traffic class), not a standing trade.

## 2. Tier 2 — the coverage axis: a correction that re-prices F-418's fit

**The correction (boot-line + code evidence):** the resident cache slots are the model's own
pack form — `ExpertCache::slot_stride(lay.blob_bytes(l))` per layer (generate.cpp:6445,
expert_cache.cpp:571), IQ3_S, max blob 2.66 MB. The boot line: **9900 slots = 18.78 GiB =
1.94 MiB/slot average** (the prompt path's borrow line agrees: 5189 slots = 9.80 GiB).
F-418's fit arithmetic used the Q2_0 canonical blob (1.3824 MiB) and priced 16k slots at
22.1 GiB; the true fit is **16k × 1.94 ≈ 30.3 GiB** — the card is 32.6, the engine holds
31.1, free ~1.5. The geometry guard will refuse (the F-409 refusal class); the 16k falsifier
boot is still worth one boot to make the guard name the cap exactly.

**The re-price:** the fit on this card is ~11k slots (+2.1 GiB from reserve + slack) → offline
~92.0% (+1.6pp) → live ~+1% by F-412's translation. The 99% goal (~20.8k slots) needs ~40
GiB — off the envelope on 32 GB; the goal is either a different card or a different form.
The live operator-traffic lines already read 94.9-97.0% lookup hit — the 99% ask is a
coverage ask, and coverage on this card is capped by bytes/slot.

**The real coverage lever: cache form per expert.** The Q2_0 canonical form is 1.3824 MiB
(−29% bytes/slot): the same 18.78 GiB holds ~13.6k slots → offline ~94% (+3.6pp). The GPU
Q2_0 walk exists in the tree (q2_native_kernels); the miss path already computes Q2_0 —
the model's effective quality is already mixed (96% IQ3_S + 4% Q2_0). A pure Q2_0-form cache
moves hits onto Q2_0 — a quality-axis trade (law 16, MMLU-gate class). The bounded shape is
**frequency-tiered requant** (T47 §3.8, MxMoE/MoPEQ class): hot experts IQ3_S, cold Q2_0,
mixed-form cache — coverage per byte up, quality price bounded and measured. This is the
coverage lever that survives the card's ceiling; it is a design, not an arm.

## 3. Tier 3 — the drafter's quality (multiplies Tier 1)

- **M3b era-gap hunt:** 77-78% now vs the 87-92% record era — the slot-depth telemetry
  (F-415) is armed; the hunt is which leg of the draft chain pays (depth-0 acceptance
  62.5%; the MTP chain dp counters read 0/0 — the chain path does not run on the standing).
- **Retrain:** the deployed drafter is Oct 13 (dense.bin 116 MB); the trainer's checkpoint
  home and the harvest corpus are GONE after the estate move (`/home/svg/LLM/Custom-data`
  empty) — the retrain cycle starts with a re-harvest (the standing trainer is
  campaign/retrain/train_cycle.py). The drafter is ours: a deeper/better drafter shrinks
  waste and raises the depth-2/3 payoff (F-229's re-open clause).
- **spec-min-p 0.70:** upstream measured +4% on their engine (F-418); env-ladder arm for
  ours, M3b canary + MMLU ×2 (distribution-adjacent knob).

## 4. Tier 4 — the conc lane (325 → 450 aggregate)

- **C2 admission/tail overlap** (T47-conc-450's top named lever): ~40-50% of the round wall
  is NOT the verify window — finish tails, admission stops, adapt join. The field is
  unanimous that host work must hide under device work at our window lengths.
- **C4 second draft row = the depth-2 arm** (batch-lane corroboration: F-416 conc +2.5
  round 1); verify the kVerifyMaxT headroom at 9-wide before pricing.
- **swaps=80 (1/0.97/80):** same offline share, half the swap traffic; the target metric is
  the adapt wait under load (1.19 vs 0.76 quiet). config-ladder 3v3, cheap, F-418's curve
  already justifies it.
- **Host placement:** F-201's CCD split + IRQ steering + renice of ambient burn (naive
  whole-process taskset falsified, −20%); banks the record under load, quiet-box gain near
  zero — the load band is the operator's norm, so it is worth the leg.
- **Cohorts-3** needs batch≥9 + the b9 lane fix; elastic already delivers 9-wide with zero
  solo tax (F-273), so the capacity is ready — the window machinery is the gate.

## 5. Stability program

1. **F-271 class is structural in the ladders** (re-pin + md5 at DONE, F-417). The remaining
   exposure: the serving boot's own learned profile — the convergence signature is live in
   the log (exchanges 4164→11891 climbing under operator traffic). Boot-invariant candidate:
   an anatomy watchdog — exchange/file-read rate above 2× the live band → LOUD alert line
   (never auto-re-pin; the operator decides).
2. **NaN census root-cause** (the shadow arm under host-memory pressure; ActQ/run_split_multi
   audit) — open defect class with a live bite; canonical-pack-primary is the mitigation, not
   the cure.
3. **Elastic spare pool:** "0 chunks left in the spare pool" at admission → materialize pays
   the release path (6.3 ms) instead of the background promote (2.2-8 ms); price
   STRATA_ELASTIC_WATERMARK=2 — a TTFT lever, small but free.
4. **Estate data home:** Custom-data is empty post-move — re-harvest is a prerequisite for
   the retrain lever, and the workspace law wants the home rebuilt before the first artifact.
5. **Upstream watch:** ADAPT_FETCH re-prices if coverage/depth changes the miss economics
   (F-412's re-open clause); SLOT_READ_WIDE/AHEAD named for the multi-turn TTFT lane;
   POOL_DEP not ported (measured −2.9% window on a 5090 upstream).

## 6. The priced order (law 4/15: falsifier cost, then needle)

| # | Arm | Kind | Cost | Priced needle |
|---|---|---|---|---|
| 1 | Lookahead stats falsifier | boot | 1 boot | decides the +21% class (B) |
| 2 | Depth-2 quiet re-test (+stair, D=3 pricing) | ladder | 5 boots | rows 2.4→3.0: +10-25% (A) |
| 3 | swaps=80 + spec-min-p 0.7 | ladders | 2 ladders | adapt wait 1.19→~0.7; +1-4% |
| 4 | Pool-granularity arm (fold/parallel quant) | build | ~3 files | ~1.35 ms exposed → ~+10% (C) |
| 5 | n-gram draft source | falsifier | offline | +5-25% copy-heavy traffic (D) |
| 6 | 16k-slot falsifier boot | boot | 1 boot | names the coverage cap (expect REFUSE) |
| 7 | M3b era-gap hunt | telemetry | 1 boot | the acceptance leg (§3) |
| 8 | Re-harvest + retrain cycle | training | GPU-exclusive | drafter quality: multiplies 2/7 |
| 9 | C2 admission overlap | build | large | the conc round wall (450) |
| 10 | Mixed-form cache (tiered requant) | design | large | coverage/byte +29% (§2) |

1-3 are behind the card and already armed (the runbook + the two ladders). 4-7 are new or
re-priced this session. The composition A+B+C attacks the same wall from three sides and is
the credible path to solo 250-270 in the current envelope; 8 raises the ceiling on top;
9-10 are the lane/coverage plays beyond.

## 7. CODE STUDY: the cache-hit levers, grounded (same day, post-answer)

Read the residency pathway end to end: `expert_cache.hpp/.cpp` (slots, residency table,
admission), `expert_source.cpp` (the host plan, the miss split, the swaps, ForesightSwap),
`generate.cpp` (the adaptive tier's scan at 10801-10856, the ADAPT_FETCH wiring at 7140),
`verify.cpp` (the grouped launches at 2213-2289), `foresight_swap.hpp`, `shadow_experts.hpp`,
`PcieModel` (expert_source.hpp:400). Live anatomy from the serving log: routed entries per
window VRAM 95.8-97.6%, PCIe 0.4-0.8%, CPU 2.1-4.0%; adaptive tier churns 2374-6190 experts
per request batch; the tier runs every=4, swaps=96, decay=0.85 (config).

**Laws the code states:**
1. `ExpertCache` NEVER evicts (`admit` takes free slots only; header comment: eviction
   policy is a measured question). The live set churns only via the adaptive tier's
   `install_resident`/`replace` and the fetch-admission.
2. The adaptive scan is PER LAYER and swaps within a layer (the victim's slot is the
   candidate's new home) — the per-layer budget is fixed at the profile-fill and can
   never move across layers.
3. `native_expert_grouped` refuses non-i-quant types (`is_iq` in `native_expert_supported`,
   iq_kernels.cu:4100); the canonical Q2_0 form is computed by `moe_grouped_s2` (the
   non-native path, kernel in tree). The plan already carries TWO group sets (VRAM P.ptr,
   PCIe P.ptr2); a third (s2-form resident) is a plan-column change, not a new kernel.
4. The shadow store (`shadow_experts.hpp`, F-359, ARMED) already maps the whole model in
   canonical Q2_0 form — 1,382,400 B blobs, uniform across layers — and the engine already
   computes every CPU miss from it (the quality axis at 3.7% of routings passed the gate).
5. `PcieModel` is ON (pcie_balance() default true + --pcie-frac 0.35) and picks the
   per-layer fetch count to minimize max(GPU, CPU); the measured 5090-class fit says the
   pool is DRAM-bound (d ~ c). pcie_frac is NOT the lever — the model already optimizes
   the split.

**The unarmed arms (zero-build, env-gated):**
- `STRATA_ADAPT_FETCH` (default 0 here; upstream ships =2 on ≤16 GiB cards): the fetched
  PCIe copies land in the victim slot (dst2) instead of staging — the bytes are already
  in flight, the residency is free. Converts ~9 fetches/window into permanent residents
  AND shrinks the tier's own copy churn (the adapt wait 1.19 ms term).
- `STRATA_FS_SLOTS` (ForesightSwap, unset): separate per-layer VRAM slots refilled by
  RouterLookahead predictions + note_miss; a landed slot is a GPU group like a hit
  ("the grouped kernel reads whatever address the plan carries, so no kernel changes").
  Budget 32 copies/card/window covers the ~70 misses. COLLISION: declared "Not with
  STRATA_VERIFY_DEVICE_PLAN" (armed here) — but the device plan only plans all-resident
  layers, and those have no misses; the gate may be over-broad. Falsifier: boot
  FS_SLOTS=4 with the device plan on and read the refusal line; if refused, the small
  build is scoping the gate to host-planned layers.

**The capacity lever, redesigned (supersedes §2's design sketch):**
mixed-form cache = a second ExpertCache arena of canonical Q2_0 slots (1,382,400 B,
uniform across layers), filled from the shadow store's mapping (no requant — the bytes
already exist), consumed by a second grouped launch per layer (moe_grouped_s2; activation
quantization already exists for the canonical path). Q2_0 form is uniform across layers,
so cold slots are a cross-layer pool — the per-layer budget constraint dissolves for the
cold tier. Budget math: 18.78 GiB in IQ3_S buys 9900 slots; 6k IQ3_S + 5.5k Q2_0 = 11.5k
slots in the same bytes, hot experts keep IQ3_S quality. No new kernel; ~4-5 files
(expert_cache form vector, expert_source plan column + fill + admission, verify.cpp third
launch, generate.cpp heat-tier policy). The offline falsifier is free: score a slot mix in
tools/replay_cache_policy.py where cold slots carry the 1.3824 MB cost.

## 8. POLYSTRATA DELTA STUDY (2026-11-23; external intel = re-pricing, law 12)

Lineage: PolyStrata (simplexai-labs) = upstream Strata **0.2.7** + their poly engine program
(388+ poly: commits; 234+ prefill:; hy4flash = a fourth engine with Tree-8 SpecInfer + MLA).
Our fork shares history only to 0.2.1 (90767252f). Their Qwen3.8 lane is upstream's engine
UNCHANGED (0.1.26/0.1.36 builds) — not a competitor on our lane. Clone:
/home/svg/LLM/data/references/PolyStrata (pristine). The 2026-11-20..23 sessions here
(F-419..F-450) hardened our tree independently; F-450 codified the Pareto rule this study
prices under.

**CODE-READ FALSIFIERS ALREADY RESOLVED (this session, zero boots):**
- FATTN-MMA/FDEC: **HAVE.** Our QSA prompt attention IS MMA (qsa_prompt_attn.cu m16n8k16,
  sm_75+ gate, default on sm_120) WITH the hi+lo split (qh/ql register halves = their
  split_bits fp32-faithful form); decode is split-K (qsa_decode_attn under g_fast_attn).
  The vec kernel survives only as the <=256-context short adapter. Their +65-85% is their
  gap, not ours. DEAD as a port.
- Cache-freeze during prompt read (their 3%): **HAVE** — adapt_quiesce fires on prompt lend
  (generate.cpp:11051); the round drains before a prompt borrows slots (11015).
- B3 partial: STRATA_RAM_ADAPT2 already seeds the RAM policy from the prompt's routing
  (generate.cpp:11531).
- Routed-stream skip copies: dead on our box (~96-97.6% VRAM share; <5% non-cached).
- RAM diet / --ram-experts-mib / --lock-experts / card_far: priced ~0 (184 GiB, deliberate
  pinned complement). Shelf only.

**THE PARETO-PRICED QUEUE from their material:**
- KVA (approximate late-layer prompt read, split 36): the only REMOVES-WORK lever, fitted on
  OUR model shape (48L/512E), +12-28% prompt at 14-30K. BUT their own numbers: KL
  0.018-0.020 nats (control floor 0.0013-0.0032; 6-15x), decode acceptance -0-11%, split-24
  failed 0/10 agent runs, split-36 6/10 vs exact 4/7. Quality axis < 0 by their own
  measurement -> **Tier C quarantine** (operator decision only). Priced falsifier ladder:
  (1) one instrumented boot pricing the late-layer skippable share of a 30K cold prompt
  (dead if <8% of TTFT); (2) if >=, KL/acceptance canary on 3 forced-token prompts; (3)
  only then a port behind STRATA_KVA.
- FATTN/#8 split_bits: HAVE (above). GDN-MMA scan: capped by the recurrence share of a 32K
  prompt (one timing boot; we hold GDN_CHUNKED 1.68x banked; their sm_120 class matches).
- Size-class tile launches (their staged +10%, decode 132->241-269 t/s on THEIR engine):
  falsifier = one counters boot histogramming rows-per-expert + launches/window; dead if
  our groups land >=70% tile occupancy.
- Switch-point fitted crossover vs our static floors: one boot logging both sides' ms at
  2-3 prompt lengths; dead if our floor sits within +/-10% of the fit.
- CPU-share link rule (limit = 3.4 x F, F = missing bytes / link GB/s): one boot line. On
  our box a cold 32K prompt streams ~37.9 GiB non-resident over ~55 GB/s -> F ~ 0.7 s,
  limit ~2.4 s: LIVE, not dead; price before any port.
- Draft admission economics (their wants()/takes() marginal-ms gate + ask/no-ask A/B):
  **offline replay** through our banked window anatomy (zero boots) prices an ask-gate
  beside the W1 ADAPTIVE_SPEC controller. Wholesale prompt-lookup adoption (no verify) is
  output-risky: needs the exact-match canary first, else quarantined.
- hy4flash Tree-8 SpecInfer (tree drafting): unexamined this pass; our MTP lane is
  serial-depth. A separate study item, priced only after the drafter retrain.

**SERVE-LAYER PACKAGE (clearest Pareto-positive ports; quality first):**
1. Greedy-card guard (QUALITY FIX, zero perf cost): our merge honors per-key overrides
   only — an explicit temperature:0 request still runs under the qwen4exp instruct card's
   top_k 20 + presence_penalty 1.5 (serve/server.py:5581 + MODEL_CARD_SAMPLING:8610),
   against the merge's own comment "explicit 0 stays greedy". Their fix d7ce708ba
   (#3295/#2940): suppress card sampling when the request asks greedy. One guard.
2. warm_start read-at-start (server-only): capture the agent's first turn at runtime,
   re-read through the engine at server start. Their numbers: agent-start TTFT
   0.24-0.43 s vs 1.49-1.74 s; 37/37 pinned answers identical. (Pinned-session form needs
   an engine leg; later.)
3. --prompt-cache / rewind-keep / spread: ALREADY PORTED in our engine (prompt_cache
   generate.cpp:804; rewind_keep :821 wired :14326 citing upstream fe686835d; spread :837)
   — all default-OFF, none armed in our config. Action = a config ladder for agent-rotation
   traffic, gated on measured re-read counters (their 8-agents measurement: 196K -> 105K
   tokens re-read, evictions 19 -> 0).
4. Shared-prefix tokenization (LACK, confirmed 0 hits): server CPU TTFT on long unchanged
   resubmissions; cheapest port on the list.
5. Tool-call robustness series (LACK: tool_markup_retry, undeclared_tool_calls, R1-R6,
   linear compaction): their measured 98.3% vs 92.9% intended-read on 1,462 agent turns.
   Quality lever aimed at the operator's own traffic shape.
6. Stalled-peer turn release (PARTIAL): their PEER_STALL_S frees the engine turn from a
   client that stops reading; our timeouts cover only the announce-body read.

**RECORD INCONSISTENCY surfaced (operator ruling owed):** standing-line A2 says ROUTE
TAIL SKIP stays 0 everywhere; T63 ADOPTED STRATA_ROUTE_TAIL_SKIP=7 standing (golden
identical, MMLU 98.6/95.7). Their --skip-weight data (weight-gated skip 10-12% faster at
equal KL vs position-gated) is the refinement candidate IF Tier C ever opens — and the
A2-vs-T63 conflict needs one ruling (the cheap resolution is our own golden set + quality
battery on =0 vs =7, our build).

## 9. UPSTREAM ENGINE DELTA via PolyStrata (0.2.2..0.2.20 content; sweep 3)

PolyStrata rebased upstream content beyond its release notes (STRATA_DEFAULTS=026 in-tree).
The 147-verify cc12.0 bitwise set we ALREADY took as sm120_series card rules (SIDE_MIXER=2,
ATTN_DAILY, GR_UP75/DOWN_TG, GU_SWIGLU, GR_V3(+PF), PF_FUSED, SMALL_UNR) — the gap is the
nv_defaults table remainder + CPU-pool/prefill/drafter machinery.

**PRICED QUEUE (their numbers, our lanes):**
- POOL_DEP (default-on upstream, edef565b9->902ea2328): run_split_multi_native's
  gate/up->barrier->down (+serial intermediate quantize) becomes ONE mode-7 batch, per-expert
  counters, first thread quantizes. BIT-EXACT (BASE-05 27/27, pool_determinism 124,656
  checks); CPU-pool window -2.9% on 5090+EPYC. This IS dossier section 4's C arm
  (pool.cpp serial act_quant barrier) — upstream engineered it; the port is the shortest
  intelligent path (law 9). LACK confirmed at pool.cpp:1106.
- Parallel doorbell quant-act + PLE gather_batch (431a26db1): host grouping 81->7 ms on our
  exact card (RTX 5090); our doorbell quant is serial per-token (expert_source.cpp:5286).
  Pairs with POOL_DEP: the serial-quant class.
- Missing cc12.0 defaults, kernels mostly PRESENT in our tree but never selected on CUDA
  cc12.0: Q80_ROWS1 (kernel exists, native_mmvq.cu:1301; our dev_class misses cc12.0),
  EXP_MODE=8, DEQUANT_SIDE (LACK), QFUSE/LFUSE/HC_UPMIX/PROMPT_ATTN_IMMA (opt-in here,
  default there). ZERO-CODE env falsifiers today; Q80_ROWS1/EXP_MODE/DEQUANT_SIDE need
  small card-rule code first. Toolkit gate: ours is CUDA 13.4 — PLE_BATCH's illegal-access
  negative is a 12.8-build class, NOT ours.
- CPU-share link_rule + NT (a620ffb61, 16c3fa0cd; NT=32 in sm_120 table, -26% 3K follow-up
  TTFT on 4090): pure-rule port; falsifier = env-boot with raised CPU_SHARE_MAX first.
- Drafter BF16 TC attention (-0.10..-0.20 ms/round drafting, 2.1x attention): falsifier =
  measure our MTP verify-attention ms/window first (F-228/DRAFT_TIMING data).
- SPEC_COST (sm_121 table, 1.047x decode, 24/24 greedy byte-identical): compare against our
  W1 ADAPTIVE_SPEC controller (both cost-model window pickers); falsifier = F-228 data.
- Conversation-cache swap-block + spill-persist (0f1738e61): LACK; serving TTFT for
  alternating agents; falsifier = measure our park/restore copy ms.
- CPU complement quant dots, AVX-512 VBMI byte-lookup (afdff31ba: 40.5 vs 25.1 GB/s, 0/26000
  dots differ): falsifier = quant_bench cross-comparison on this box (bench, not port) —
  if theirs wins, the CPU-experts term (2.98-3.55 ms/window) shrinks proportionally.
- MMQ_TIER_SORT (240aeb985): falsifier = code read of our staged-walk ordering.
- VERIFY_DEVICE_HITS (a14c910fe, default >16 GiB): CONFLICTS with ForesightSwap in BOTH
  trees; skip unless we trade FS for device planning.
- SEQ_READAHEAD (dc3ace6a0, author-reported 65-76s -> 23-28s cold follow-ups, left OFF
  upstream): our complement prevents the shape; low.
- SLOT_READ_WIDE/AHEAD: --batch-lane only; skip (cohorts standing).

**NEGATIVES BANKED (upstream measured on OUR exact shape — do not adopt):**
- L2_PF default-rejected on sm_120: -8-12% on 1x 5090 IQ3_S (#6431) — OUR pack. The BW-C
  trio's +2.0% does NOT transfer; only Q80_ROWS1+EXP_MODE legs.
- TAIL_SKIP: upstream flipped default OFF for 0.1.63 (answers differ when on). Ours T63
  =7 is golden-identical ON OUR SET — conscious divergence; sharpens the A2-vs-T63 ruling.
- QFUSE on sm_86 regressed partial-VRAM decode ~5% — scope any QFUSE default to
  all-resident, keep GR_UP75 semantics.
- ADAPT_FETCH=2: upstream default ONLY on <=16 GiB cc12.x; NOT bit-exact (forced-token KL
  0.948 [0.830, 1.063]); hit-rate gains cited were AT 54->62% there. REVISION of section 7:
  on our 32 GB card upstream itself takes 0; the arm's honest price is =1 (parity-safe
  semantics) with a forced-token KL canary, expected value below the first pricing.
- ROUTE_PRIOR: upstream owner-reverted to default-off (agent tasks 4.5/10 vs 5.6/10,
  repetition loops). Ours is off standing (verified: launcher has no STRATA_ROUTE_PRIOR).
  Keep off; bank the negative.
- gfx12 miss-split 0.15, PCIe-share auto rule: HAVE/NA. PLE_BATCH@CUDA-12.8: NA (13.4).

**EXECUTION ORDER (all gated on the campaign law; none run on our box):**
(a) zero-code env falsifiers: QFUSE/LFUSE/HC_UPMIX/PROMPT_ATTN_IMMA/PLE_BATCH + raised
CPU_SHARE_MAX (diagnostic boots, interleave, M3b watched); (b) POOL_DEP port (bit-exact
harness pattern exists upstream); (c) link_rule+NT pure-rule port; (d) parallel quant-act +
gather_batch; (e) quant_bench cross-comparison; (f) SPEC_COST after DRAFT_TIMING pricing.

## 10. SERVE-PACKAGE ADJUDICATION (2026-11-23; desk recon, both trees; free falsifiers RUN)

Per-claim verdicts (evidence in the adjudicator's report; corrections to section 8 marked):
- **A Greedy-card guard: CONFIRMED, deeper than priced.** (i) The merge (server.py:5581,
  :7443 /props) lets the card's other keys survive an explicit temperature:0 — REPL PROVEN
  on our code today: card {temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5} +
  {temperature:0} -> GEN "top_p=0.8 top_k=20 penalty_present=1.5 penalty_last_n=64".
  (ii) Engine side: sampler.cu:1259-1269 greedy branch argmaxes over apply_penalties()-
  SHIFTED logits whenever a penalty window exists — the pick is not the true argmax
  whenever it appeared in the last 64 tokens. (iii) OUR OWN F-400 Prewarm replay posts
  temperature 0.0 through the same merge (:4145-4150) — its "greedy" premise is false the
  same way (deterministic, but penalized). (iv) Their fix is TWO commits: d7ce708ba
  (asks_greedy + greedy= param popping CARD_PENALTIES; ~70 LOC port, env kill-switch,
  port their test) and d35353903 (MTP + nothing-asks-sampling -> card={}: DO NOT port
  blindly — flips our whole default-API traffic to greedy). HAZARD: presence 1.5 is our
  incidental anti-degeneration medicine — pair the fix with arming
  reasoning_loop_recovery (default OFF today, :4342) or verify :5914 recovery armed.
  Gates: MMLU + M3b MANDATORY (token-stream-affecting; expect an output JUMP,
  pre-register), 3x probe-lane same-hour.
- **B warm_start: FALSIFY-FIRST.** Their read-at-start is real (numbers verbatim, their
  box); port shape = persist agent's first turn + pin at boot (~120-160 LOC). OUR
  conflicts: F-400 corpus replay ordering (warm GPU reads BEFORE the replay break routing
  determinism — pin-only is safe, re-read is not), probe-prefix contamination (a pinned
  10-15K start parked must not prefix-match probe prompts), 24-slot budget holds the
  ~212-433 MiB state. Cheapest falsifier: one boot, read the FIRST real agent request's
  DONE line vs mid-session — if Prewarm+cache already hold a 4K+ warm prefix, residual
  win is only the system-prompt-shaped prefix.
- **C Tool-call robustness: PARTIAL — the 98.3/92.9 number is tool_call_recovery's,
  WHICH WE ALREADY HAVE, DEFAULT OFF** (serve/frontend.py:931-1469, wired :4346,
  config :9141; not armed in custom-iq3_s-8081.json). Step 1 = config-only arm +
  adherence battery. Step 2 = port _compact_scan_body + tool_markup_retry +
  undeclared_tool_calls (~110 LOC) after their rcall_specimens.json (25-entry corpus,
  IN THEIR TREE) runs against our parser (~60 LOC adapter, no GPU). R1-R6 (~150-200 LOC)
  falsify first: count damaged openers in our logs.
- **D Shared-prefix tokenization: REFUTED — we HAVE it.** The named commit f13a870d5 is
  in OUR history; PromptEncoder (tools/strata_tokenizer.py:446, wired :4359-4368,
  keep=24). My section 8 "LACK" was a wrong-symbol grep. Law 18 in action.
- **E Conversation-cache arms: CONFIRMED ported-but-off; MEASURED, and the arm has a
  live target.** OUR LOG (4,820 prompt events): mean prefix reuse 93.2%, but 193 FULL
  re-reads (4.0%) and 266 prompts <50% reuse (5.5%) — a real returning-agent/eviction
  class. Ladder when the card frees: prompt-cache 12 -> park_gain 256 -> rewind_keep,
  one per leg, gated on the full-reread count falling; read-only skips are
  byte-identical (M3b watch only; MMLU cheap insurance).
- **F PEER_STALL_S: CONFIRMED gap, DO (~25 LOC + test).** Our engine survives a stalled
  reader (stdout pumped to per-slot queues) but the HANDLER THREAD + ITS BATCH SLOT LEAK
  forever — one hung client silently drops 6-conc capacity by 1/6. Port their
  99b1b27b1 form: per-socket-op bound, SSE write loop settimeout, OSError ->
  existing disconnect path; STRATA_PEER_STALL_S=0 rolls back. Falsifier: campaign logs
  for stuck slots; synthetic stalled-reader test on a dev boot.

BANKING LAW (recon): server-only changes need NO binary ladder (that is an engine-binary
gauntlet; precedent F-19 serve-layer default-on with before/after evidence). MMLU gate
only for token-stream-affecting changes: A yes; B/C/F no; E watch-only.

DO-FIRST ORDER (serve): A (~70 LOC, biggest quality win, gates paid) -> E (free
falsifier done, ladder armed when card frees) -> C step 1 (0 LOC + corpus harness) ->
F (~25 LOC) -> B (falsify then port pin-core) -> D dropped.

## 11. ENGINE-PACKAGE ADJUDICATION (2026-11-23; desk recon + BANKED evidence; corrections to my own pricings)

- **A POOL_DEP: CONFIRMED 1:1 (pool.cpp:1106 same structure; no NUMA/invariant variants here
  = simpler port; pack manifest: 40/49 layers engage, 9 Q2_0-down excluded) — but the
  falsifier ALREADY RAN BANKED: campaign/results/f325-stair-*.out reads gate/up 1.301 /
  quantize 0.009 / down 0.716 ms/round (and 1.950/0.015/1.012). The serial-quant term is
  0.009-0.015 ms/round; the overlap win ~0.06-0.10 ms/round = ~0.4-0.7% of our ~15 ms
  window. CORRECTION: section 4's C-arm pricing (~+10%) was WRONG — carried from the
  compaction's 1.35 ms figure; the banked data contradicts it. Law 15: DROP standalone
  (below the +4% bar on every lane); compose-only if ever (env STRATA_POOL_DEP, ~70-85
  LOC; certifiers already ours: pool_tasks_test extend, native_hadamard_test, M3b+MMLU).
- **B parallel doorbell quant / gather_batch: DROP** (banked falsifier fired: our decode
  exposure 0.036-0.090 ms/window; prefill ceiling ~0.5%; their own evidence is comment-
  argument, no parity harness). Re-open only on hardware change.
- **C cc12.0 default-set gap: CONFIRMED, the best-value cluster on our box.** PROMPT_ATTN_
  IMMA already standing (launcher:142). Today's PURE ENV arms (one boot each, identity-
  gated, M3b exact-match + golden hash, MMLU 93.3 gate): STRATA_EXP_MODE=8 (consumer in-
  tree iq_kernels.cu:4725; exact on P100/P40 per our own comments), STRATA_NO_MMVQ_ROWS2=1
  (bit-exact, unscoped — if it reads, write the 10-LOC scoped STRATA_Q80_ROWS1 class-3
  rule mirroring their e6efa2b16), STRATA_HC_UPMIX=1 (+STRATA_HC_UPMIX_CHECK=64: BUILT-IN
  rel-RMS/max|diff| print, prefill.cpp:3172-90), STRATA_LFUSE=1, STRATA_PLE_BATCH=1
  (12.8-hazard N/A on our 13.4), STRATA_QFUSE=1 (watch partial-VRAM decode, #6583 tax).
  Their BW-C trio (Q80_ROWS1+EXP_MODE+L2_PF) decode +2.0% DOES NOT transfer wholesale —
  the L2_PF leg is the measured sm_120 negative; the pair alone is the candidate. Code-
  first later: STRATA_DEQUANT_SIDE (~80-120 LOC + parity), PLE_COMPACT/COPY_STREAM. Ours
  has NO sm_120 nv_defaults table at all (nv_defaults.cpp = sm_86/sm_60 only) — the
  composed-verified set lands as ~15 LOC + test.
- **D CPU-share link_rule + NT: DORMANT on our standing — the arm gate requires
  o.batch <= 0 (generate.cpp:17586-90) and we serve parallel:6 -> the CPU share is OFF
  entirely.** DROP-standing; re-open under a solo/batch<=0 serving arm with the F-print
  falsifier (3.4 x 700ms = 2,380 tokens < our fixed 3072, so the rule would TIGHTEN us —
  note the link-GB/s spread 55/26/12 is tier-dependent; F must come from a live probe,
  not assumption). Port shape when opened: ~60-75 LOC rule + ~20 plumbing + 15 NT knob.
- **E VBMI byte-lookup dots: FALSIFY-FIRST at bench level, no boot.** Their 40.5-vs-25.1
  GB/s is THEIR OWN "from memory" figure, PENDING on their side (only 0/32,000 SIMDe
  parity banked). Ours: iq_avx512.cpp Fmt<21> (AVX-512+VNNI, no vpermi2b in tree); the
  trick touches ~20% of gate/up work (IQ3_S gu on 10/49 layers); contract differs
  (Q8_K vs our ActQ). Step 1 zero-code: kq_cpu_bench --pairs iq3_s/iq4_nl --threads
  1,8,16,32 --nt 1,2,4 baseline. Step 2 ~100-150 LOC vpermi2b variant + parity
  (iq_avx2_parity/expert_parity, 0 differing dots). Expected if it transfers: 1.5-2.5%
  window, LOW confidence.
- **F falsifiers refined:** F4 SPEC_COST -> OFFLINE arithmetic from F-228 banked data
  (depths 3/4/5 offered 35/17/8%, accept ~85/68/75%) through E(tokens)=1+sum(prod p) vs
  step_ms=8.4+1.57T: ZERO boots; optimum >= standing T kills the W1 follow-up free.
  Optional 3-5 LOC decision print at generate.cpp:16645 (STRATA_ADAPTIVE_SPEC log line —
  the controller is invisible today). F2/F3: ONE diagnostic boot, STRATA_PREFILL_TIMING=1
  at 4K/16K/32K — the timing lines already print the gdn recurrence share (prefill.cpp:
  3218-32); +5-15 LOC for the effective-floor/ring print. Their gdn_mma is the F-224
  re-open candidate (final-state-only prefill recurrence), not a coverage hole.
- **G VERIFY_DEVICE_HITS: DROP.** Two corrections to the claim: the >16 GiB default is
  431a26db1 (not a14c910fe), and OUR TREE ALREADY OWNED the class — T1.1
  STRATA_PLAN_PARTIAL, measured CLOSED-NEUTRAL banked at F-222 (-0.6/-1.6/-0.9; the
  flagA round-trip is worth ~0.26 ms/window). A verbatim port is INERT on our boot
  (explicit --pcie-frac 0.35) and would fight PcieAuto on frac-less boots (law 2).
  Re-open clause: a new instrument reads flagA/flagB/plan-publish > ~0.3 ms/window, or
  pcie_frac 0 adopted standing -> then STRATA_PLAN_PARTIAL=1 (fully wired, ONE env)
  before any port.

**NEGATIVES (verified in their tree; our standing respects all six):** L2_PF sm_120
(-8-12% on 1x5090 IQ3_S #6431 — mechanism ABSENT in ours), ADAPT_FETCH=2 <=16 GiB +
not-bit-exact (we carry the plumbing, unset standing), ROUTE_PRIOR owner-off (4.5/10
agent score NOT in their tree — unconfirmed, owner decision 2026-10-29 #4402 is),
QFUSE sm_86 partial-VRAM tax, gfx12 miss-split RDNA4-only, PCIe-share auto rule already
ours (generate.cpp:2273-2324, dormant under explicit 0.35).

**UNIFIED DO-FIRST (serve package section 10 + engine package):**
1. [0 boots] F4 offline SPEC_COST arithmetic; A-correction recorded (done here).
2. [0 LOC] Serve step 1: arm tool_call_recovery + corpus falsifier; DONE-line ladder
   design for E already measured (193 full re-reads).
3. [1 boot] Serve A port (~70 LOC) — greedy-card guard, the proven defect; MMLU+M3b,
   output jump pre-registered.
4. [1 boot each, identity-gated] Engine env arms: EXP_MODE=8 -> NO_MMVQ_ROWS2 ->
   HC_UPMIX(+CHECK) -> LFUSE -> PLE_BATCH -> QFUSE; compose survivors through
   campaign/binary-ladder.sh; then the ~15-LOC sm_120 defaults table.
5. [1 diagnostic boot] STRATA_PREFILL_TIMING=1 stair (F2 floor fit + F3 recurrence
   share) + PEER_STALL_S port (~25 LOC) same window if convenient.
6. [bench, no boot] kq_cpu_bench baseline -> vpermi2b variant + parity.
7. [port] warm_start pin-core after its DONE-line falsifier; conversation-cache ladder
   (prompt-cache 12 -> park_gain 256 -> rewind_keep) gated on full-reread count.
8. [compose-class, when the above read] POOL_DEP + DEQUANT_SIDE + sm_120 table into ONE
   candidate binary through the ladder.
DROP with mechanisms banked: D-standing, G, B, L2_PF, ROUTE_PRIOR, ADAPT_FETCH=2,
shared-prefix tokenize (HAVE), RAM-diet family (~0), R1-R6 (until our logs show the
class), d35353903 (blast radius), KVA (Tier C quarantine, ladder written).
