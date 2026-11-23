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
