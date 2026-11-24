# From Lottery to Contract: the November 2024-class Reliability and Performance Campaign on Strata

**A state-of-the-record paper — achieved, not achieved, learnt, unclear**
Date: 2026-11-24 · Scope: the campaign arc 2026-11-18 → 2026-11-24 (F-395 → F-450 + the standing-quality arc)
Estate: `/home/svg/LLM/strata-custom-2` · Qwen3.8-Flash-Next MoE ("em-oh-ee", mixture of experts), IQ3_S expert
quantization, one RTX 5090 32 GB (sm_120) + 192 GB host RAM, Linux · Standing envelope: 262144 context, vision on,
spec drafting (`--spec 6`), batch 6, int8 KV, 9900 expert slots · Sources: `SVG-CUSTOM.md` F-entries (the journal,
13,739 lines), `campaign/tracker/lanes.csv` (the chart of record), `campaign/results/*` (evidence files),
git history `695bf9c7f…66a0c6beb`.

---

## Abstract

The campaign entered this arc with a boot lottery — identical configs drawing 137–240 t/s across boots — and a
scattered 100–170 t/s serving experience against a banked ~220. It leaves the arc with: **certified-deterministic
boots** (five-boot spread 1.5 %, every organ declared, refusals fatal), a **banked M1 mean of 220.6 t/s** (record leg
228.7, MMLU-Pro 94.29), the prime performance suspect **acquitted by measurement** (the expert GEMV kernels run at
95–98 % of the measured stream roof — one microbench falsified a multi-week port ladder before it was built), the
performance wall **rewritten as one equation** (host-stage exposure × (1 − rows in flight)), a **structurally
hardened engine** (roughly thirty F-entries of decomposition, each certified under Golden Master harnesses), a
**quality axis under a Pareto rule** with a two-table control-vector architecture adopted, and a **priced forward
queue** whose top lever (draft depth) is mechanism-proven but adoption-pending. No priority lane has yet been
promoted above the 220.6 bank: the arc's yield is determinism, map quality, and structural integrity — the levers
the map prices are queued, not yet cashed. This paper is the complete, citation-grounded account of what was
achieved, what was not, what was learnt, and what remains unclear.

---

## 1. Method (why this arc's numbers can be trusted)

Every verdict in this record was produced by the campaign laws (AGENTS.md, "The performance campaign law"):

- **The interleave settles it** — no comparison across boots, boxes, or hours; ABAB / 3v3 alternation inside one
  window, means only (law 1, 17). A single-boot number is a rumor; this arc's adoptions carry interleave evidence.
- **The cheapest falsifier first** — measure the premise before arming the lever (law 4). The arc's largest single
  saving was a one-boot microbench that killed a ×1.5–1.7 port ladder (F-411).
- **Mechanism over accounting** — a millisecond is a question, not a budget line (law 3). The wait-structure
  decomposition (F-414) replaced a bandwidth narrative with a structural one.
- **The canary is the contract** — acceptance (M3b: the fraction of speculative draft rows accepted under
  exact-match verification, pronounced "em-three-bee") watched on every battery; MMLU-Pro ≥ 93.3 gates anything
  token-stream-adjacent (law 3, 10).
- **Negatives are assets** — every falsification is banked with mechanism and a re-open clause (law 8).
- **The record is the engine of compounding** — journal + commit + push; unpushed work does not exist (law 14).
  The 42-commit backlog was published and verified at `a2dce9285` (F-410).

Terms used throughout, defined once: **TTFT** (time to first token); **ITL** (inter-token latency); **M1** (the
single-stream decode lane, tokens/s); **conc** (the 6-concurrent aggregate lane); **the tier** (the VRAM expert
cache, 9900 slots); **the complement** (the 37.86 GiB pinned host-RAM copy of all experts); **GPU-reach wait** (the
service thread's wait for the GPU to reach the queued window — idle GPU time by definition); **Golden Master** (a
harness that executes a previously untested seam and freezes its behavior so later refactors carry a before/after
proof); **interleave** (same-hour alternating legs, ABAB boots or 3v3 rounds).

---

## 2. Where the arc started

Three standing problems defined the arc's mandate:

1. **Restarts were slow and inconsistent.** The #477 learned-profile save/load accumulates a routing profile across
   sessions (saved every 30 rounds); a battery's probe traffic poisons it, and a poisoned profile makes the tier
   thrash — 25,303 expert exchanges and 10,196 file reads on one boot against a healthy 2.7–3.1k (F-271, F-417).
   Separately, boots drew wildly different speeds at identical config: the recorded extremes were 137.2 and 239.9 t/s
   maxima on the same recipe (F-404).
2. **A perf-hunt mandate**: "aggressively hunt down latency bottlenecks, throughput regressions, HBM/PCIe stalls,
   and flawed lane logic," pairing telemetry with external research, under two invariants — exact mathematical and
   behavioral parity across execution paths, and strictly-forward TTFT/ITL/tokens-per-second with no tail-variance
   and no non-deterministic boot states.
3. **A reliability plan** ("100 % reliable and ready to continue improving"), consolidated in
   `campaign/research/ROADMAP-RELIABILITY.md` (commit `608501819`) after three research passes.

---

## 3. What has been achieved

### 3.1 The boot lottery is dead — determinism is now a contract (F-400 → F-409)

The lottery decomposed into four independent mechanisms, each found by instrument, each closed by architecture:

| # | Mechanism | Instrument that found it | Fix | Entry |
|---|---|---|---|---|
| 1 | Tier starting state was sampling luck (prewarm replayed temperature-0.7 text; MoE routing depends on tokens) | fingerprint-identical boots, divergent M1 | Prewarm's first stage = fixed calibration corpus at **temperature 0** — routing becomes a function of config alone | F-400 |
| 2 | Host end-state unchecked (engine could choke the box silently) | operator-run /proc forensics | Boot contract line at READY **and** post-warm-in: engine RSS / shmem / MemAvailable, in /health | F-401/F-403 |
| 3 | **The memory organ flipped per boot** — the 37.86 GiB complement was allocated driver-pinned (cudaHostAlloc → memfd, shmem 62.9 GiB) or paced-registered (THP + cudaHostRegister, shmem 25.0 GiB), decided by free RAM at the allocation instant | the contract line's bimodal shmem + smaps on a live boot | `STRATA_COMPLEMENT_BACKING=hostalloc` declares the fast organ; the flip is gone | F-404/F-405 |
| 4 | A declared organ that failed would silently degrade (resident → mmap) | forced-refusal retest caught the second fallback layer | Source-level refusals + `contract_refused_` gates the soft branches: **a refused organ exits hard**, never degrades | F-409 |

**The certification numbers** (F-405, F-408): the backing flip's speed signature was decisive — driver-pinned boots
maxed 239.9/203.8; paced boots 137.2/195.4/191.3/138.7. Declared `hostalloc`, the five-boot AUTO spread of
**53.1 %** (floor 137) became **9.4 %** at a worse load band (floor 194), and the quiet-window bank closed the arc:

| Boot | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| M1 maxima (t/s) | 226.5 | 225.3 | 226.7 | 227.8 | 228.7 |

**Spread 1.5 %, stdev 1.3 — M1 bank 220.6 (mean, legs 2+), record leg 228.7, MMLU-Pro 94.29 % (66/70, floor
93.3).** The lottery's full arc: 53.1 % → 9.4 % → 1.5 %. The tracker's standing line never stepped down (220.6 vs
the Nov-9 223.6 bank: −1.3 %, inside noise — a within-noise re-bank confirmation, per the incline law).

What is *not* yet contract: the leg-1 warm residual (−11.8 % on boot-leg 1; mechanism narrowed to a routing-set
mismatch — corpus generations route to their own experts; see §5.6) and R3 anchored residency from the boot
contract (still not in tree; ROADMAP Phase 1).

### 3.2 The prime suspect was acquitted — and that is the arc's biggest save (F-411)

The forensic audit (commit `695bf9c7f`, `perf-hunt-2026-11-20.md`) priced suspect #1: the expert GEMV kernels
(`gu_pair_kernel`/`down_pair_kernel`, `src/kernels/cuda/s2_expert_grouped.cu:322-402`) derived ~185 GB/s achieved
against ~1790 GB/s peak — an apparent 10 %-of-roof wall, with an external dossier (`6776d492f`,
`gpu-inference-optimization-dossier-2026-11.md`, 40 findings) pricing a three-rung port ladder (llama.cpp few-row
MMQ #31533 → Marlin-class arXiv 2408.11743 → CUTLASS SM120 TMA-WS #3280) at **M1 ×1.5–1.7**.

The microbench existed nowhere, so it was built: `bench/micro/moe_hit_sat.cu` — standalone, **including the
engine's own `.cu`** (no copied kernels; the owner law), launching the kernels exactly as the engine does, ROTATE
(fresh slots — DRAM-honest) and HOT (L2 ceiling) modes, swept n_hits 13→3328, anchored by a grid-stride uint4
stream over the same allocation.

| Measurement (saturated, ROTATE) | Bandwidth | % of anchor |
|---|---|---|
| uint4 stream anchor | 1493 GB/s | 100 (83 % of the 1790 datasheet) |
| `gu_pair_kernel` | 1445–1470 GB/s | 95–98 % |
| `down_pair_kernel` | 1317–1334 GB/s | 88–89 % |
| HOT at live-like n_hits = 52 (gu) | 3601 GB/s (L2-fed) | — |

**Verdict: the kernels keep pace with a pure stream. The integer-dequant starvation hypothesis is falsified.** The
derivation's error was doubly instructive: the blob is 1,382,400 B (the cache-side S2 form), not the 2.66 MB pack
form the dossier assumed — live unique bytes ≈ 48 layers × ~13 experts × 1.38 MB ≈ **862 MB/window ≈ 0.6 ms of
kernel time inside a ~9 ms wait**. The H2 port ladder was killed before a line of it was written. One boot-sized
instrument retired a multi-week engineering program — law 4 at full scale.

### 3.3 The wall was rewritten: solo starvation, not silicon (F-414)

With the kernels acquitted, the wait structure itself was decomposed (the T26-era `STRATA_DM_WAIT_STATS` instrument,
first read on the current standing). Per group: waitA (host plan) 4.8–5.2 µs — dead; waitB (PCIe) 9.8–24.0 µs
against 52–56 µs of reach — fully covered; waitCPU (rows) 37.9–59.1 µs exposed ≈ **1.35 ms/window**. And the
smoking gun, one boot, two lanes:

| Leg (same boot) | Window | Rows in flight | GPU-reach wait |
|---|---|---|---|
| M1 solo | 13.60 ms | 3.10 | **8.79 ms** |
| 6-conc | 12.94 ms | 4.88 | **0.51 ms** |

**The wall is one equation: host-stage exposure × (1 − rows in flight).** The solo window starves the GPU — too few
accepted draft rows, so the host stages (CPU experts 2.85–3.27 ms + adapt 0.76–1.19 ms + emit 0.35–0.61 ms) expose
as GPU idle. Concurrent traffic supplies rows and the same engine, same boot, same kernels, runs 17× less idle. This
unified the two biggest open numbers on the board: M3c (tokens/window, banked 2.18–2.41 vs target ≥ 3.0) and the M1
wait wall are the same lever. Every remaining high-value arm either **adds rows** (draft depth, acceptance),
**shrinks a host stage** (pool granularity, swap traffic), or **moves a host stage under the GPU window**
(lookahead overlap) — the priced queue in `levers-2026-11-20-evening.md`.

### 3.4 The hypothesis ladder resolved — three negatives banked, one mechanism proven (F-412 → F-416)

| Hypothesis | Result | Mechanism verdict |
|---|---|---|
| H3: adaptive-tier calibration champion (`every=1 decay=0.97 swaps=160`; offline hit share 0.9049 vs 0.8839) | **+1.5 % M1 (bar +4), conc −4.0 % at the floor — NOT adopted** (F-412) | The CPU-expert term the calibration feeds is 19–22 % of a 13.25 ms window: +2 pp of hit share ≈ +0.1 ms/window ≈ +1 %. Offline replay ≠ live window. Re-opens if the miss economics change |
| H4: prewarm corpus extension (shape-matched prompts) | **−1.1 % on the targeted warm leg — falsified** (F-413) | The residual is a routing-**set** mismatch; domain-level shape matching does not close expert-set overlap. Leg 1 is discarded by protocol anyway — the discard IS the fix; banked closed |
| H5: draft depth 2 (`STRATA_BATCH_DRAFT_DEPTH=2`, pre-built and never armed) | **Mechanism live: round-1 +6.8 % M1, legs to 239.4, M3b quiet (76.5/77.1 → 77.8/77.4). Means +2.4 % in a moved load band — NOT adopted on this battery** (F-416) | F-414's physics stands; the re-test ladder is named (quiet five-boot, stair published on both arms, MMLU ×2) |
| (prerequisite) per-depth telemetry | Built, publish-only — and it exposed the drafter's true shape: **the slot drafter offers depth 1 only; positions 1–4 are never tested; the 61–66 % aggregate is depth-0 acceptance** (F-415) | F-228's "depths 4–6 offered" described the MTP-chain era; the standing speculates single proposals. The chain machinery (`keep[b] = acc`, `commit_slot_prefixes`, kVerifyMaxT rows) sits ready and idle |

These are the record's compound-interest entries: each negative cost one battery and permanently priced its
boundary.

### 3.5 The engine was structurally hardened (F-419 → F-449)

The structural audit (F-419, `STRUCTURAL-AUDIT-2026-11-20.md`) named five god-modules (main() 16.3k lines,
run_impl 2473, dispatch_multi 570, the post() env-matrix, _run 534) and four safety-net gaps. The hardening mission
then executed, every step certified against a baseline or a Golden Master:

- **Determinism-by-default**: the time-seeded sampling RNG and per-window `getenv` hoisted and unified
  (F-420); three Qwen4Exp prefill kernels pinned to `kCompiledGeometry` by assert (F-421); ~30 env reads across
  verify/mtp/expert_source consolidated into the owned `strata::env` module with a truthiness safety net
  (F-422–F-430). One detour was **reverted on operator ruling** — added complexity that did not simplify (F-431,
  the env_flag module) — the record polices itself.
- **Decomposition under proof**: the dispatch god-function became a phase list (admission → plan → CPU-miss →
  swap-admission; F-432–F-436), each cut verified against the pre-session baseline (gate V1, F-437), then frozen
  under Golden Masters covering the dispatch seam, single-token dispatch, and the full branch matrix
  (F-438–F-440). `pin_cache_complement` — the very organ the lottery lived in — was extracted in three verbatim
  cuts into `ComplementArena` (F-441–F-443), and the arena's open path got its own Golden Master plus a mechanized
  return-path audit (`campaign/return_audit.py`, F-446–F-447).
- **UB retired as a class**: `-Werror=return-type` is now permanent after the defect fired twice (F-448).
- **Close-out gate**: the full suite runs 92 % serial-green at session close with the failure set proven
  environmental (live-engine VRAM starvation under `-j8`; 25 pre-existing serial failures identified as F-437's
  set) — F-444, F-449.

The readable summary: the engine's load-bearing seams now execute under test, its invariants are executable, and
the next kernel or scheduling change inherits harnesses instead of hope.

### 3.6 Upstream intelligence converted to pricing, not recipes (F-397 → F-418 and the delta studies)

- **The cross-engine calibration run (F-418):** upstream's own tuner, on their stock engine, seeded with our gold
  profile, landed at **163.1 t/s best-tuned** against our standing 210–230 — the fork leads upstream's best-of-box
  by **+29–41 %** at the same card. Their tier shape converged with ours (every=1, decay=0.97); the swaps knob
  saturates at ~80 (0.9042 → 0.9049 across 80–160 — past it, double the copy traffic for +0.07 pp).
- **The 99 % answer:** the replay coverage curve (9900 slots → 90.4 % … 20.8k → ~99 %) proves the knob space is
  exhausted at ~90.5 %; **the slot count is the coverage lever**. The follow-up correction (`levers` §2): slots are
  the pack form at 1.94 MiB average, so 16k slots ≈ 30.3 GiB — the geometry guard will refuse on a 32.6 GiB card;
  the coverage hill now wants a per-expert cache form (the mixed-form Q2_0 cold-tier redesign, commit `05a5c613c`),
  not a bigger number.
- **Ports landed with parity harnesses:** upstream's rewind-keep (follow-up TTFT 6.46 → 1.62 s at 460K in their
  field log; our harness green, adoption owed a battery), the #5697/#2253 eviction stops, MALLOC_ARENA_MAX=2 and
  stdin-clean probes (F-397–F-399), and the complete three-step `STRATA_ADAPT_FETCH` predicted-prefetch port —
  correctly scoped as a **shelf lane** (it refuses resident+DMA, our standing's organs; it pays on non-resident
  recipes) per the scoping law.
- **Adjudications on our exact code:** the serve-package study **proved the greedy defect** on our tree
  (temperature-0 requests were emitted with penalty/top-k sampler settings — `sampler.cu` penalized argmax,
  affecting the F-400 replay class); the engine-package study corrected POOL_DEP pricing to 0.4–0.7 % on f325 banked
  data and killed the E-arm on F-375's byte-bound evidence; the PolyStrata lineage study established our lane is not
  their competitor and resolved the attention/cache-freeze code-read falsifiers.
- **Store closure** (`66a0c6beb`): one data home, one pristine references library — the workspace law enforced.

### 3.7 The quality axis became first-class (F-450 + the vector arc)

Operator directive 2026-11-23 — "better quality with same performance, or better performance with same quality" —
is now the standing acceptance rule, codified in `campaign/standing-line.md`: both axes ≥ 0, one strictly >,
measured A/B on the same build pair, with a determinism canary (`shadow_misses`) and a three-tier ledger.

On that rule, the control-vector arc landed (commits `0bd36187e` → `ba13d1acb`): a two-table vector architecture
(persona add-dose on the answer path; refusal-ablation projection on layers 1–47), a fused-path mode bug found and
fixed (table 1's mode was steering table 2's pointers), the dose walked back 0.45 → 0.3 when the MMLU interleave
found the window's edge, then recalibrated in the block's read domain (harmonic mean of stream norms), then
**phase-scoped** so the persona add lands on the answer phase only — MMLU ties stock, register stronger. The
arc's own lesson is in its commit trail: each dose change was an interleave verdict, not a taste decision.

The same session diagnosed the "same prompt, wildly different results" report: the primary mechanism is
`STRATA_SHADOW_EXPERTS` — when armed, every CPU miss computes from the Q2_0 shadow store whose quantization differs
by design, and the hit/miss boundary moves with cache history, so the per-token quantization mix drifts run to run
(F-450). The falsifier ladder is recorded; greedy collapses it to near-boundary flips.

### 3.8 Instruments built this arc (the map is now measurable)

| Instrument | What it sees | Entry/commit |
|---|---|---|
| `bench/micro/moe_hit_sat.cu` | Kernel saturation vs stream anchor (ROTATE/HOT) | F-411 |
| Slot-drafter per-depth telemetry | accepted/tested per position, publish-only | F-415 |
| Boot contract lines (READY + settled) | engine RSS/shmem/MemAvailable per boot | F-401/F-403 |
| `complement_backing()` organ banner + boot-spread recorder | the memory organ, per boot | F-404 |
| Thread naming (19 sites) | per-thread CPU attribution on burn maps | F-407 |
| config-ladder / env-ladder (ROUNDS, load-per-leg, PUBLISH_ENV, M3b + adopt-bar in summarizers, md5-verified profile re-pin at DONE) | config/env A/B with trap-restore and close-out hygiene | f412/f416/f417 rounds |
| `campaign/standing-line.md` batteries + `shadow_misses` canary | quality/perf Pareto verdicts | F-450 |
| `campaign/return_audit.py` + four Golden Masters | seam behavior frozen under test | F-438–F-447 |
| Ladder refusal path (arena-first OOM names the holder; trap restores config byte-equal) | box-level battery safety | F-412 note |

Session-side (harness, not engine): the operator's web-search tooling was repaired — a credential slot held a
foreign provider's key; a valid key now sits in the harness config. This unblocked the external research passes
that fed §3.2's dossier and §3.6's adjudications.

---

## 4. What has NOT been achieved (the open ledger)

Honesty first: **no priority lane has been promoted above the 220.6 bank in this arc.** The yield was determinism,
map quality, and structure — real, banked, and precursor to every future promotion — but the t/s chart's standing
line is flat within noise since F-408. The concrete open items, each with its owner condition:

| # | Open item | Why it is open | Next action |
|---|---|---|---|
| 1 | **Depth-2 adoption** (the top-priced lever, ceiling = the 8.79 ms idle itself) | F-416's window was not quiet; means landed inside the band with the load moved between arms | The named re-test: quiet window, five-boot means, stair published on both arms (prices D=3), MMLU ×2 on adoption |
| 2 | **Conc quiet re-bank** | F-410's battery had the window close under it (load 0.5 → 3.8 mid-run); law 12 correctly appended nothing | Five-boot quiet protocol; F-414's single diagnostic read (conc 312.5 at load 2.2) says bank-class is likely |
| 3 | **Lookahead overlap arm** (ceiling ~+21 %: hide the 2.9 ms CPU-expert term under GPU work) | The falsifier boot has not run | One boot: `STRATA_LOOKAHEAD_STATS=1` + probe leg — the predictor's recall decides the arm free |
| 4 | **Pool-granularity surgery** (~+10 % class: fold `act_quant_q8_1` into the phase-1 epilogue, dissolve the serial barrier at pool.cpp:1050) | Specced (levers §1C), not built | Build behind a parity harness; composes with #3 |
| 5 | **Coverage beyond 90.5 %** | 16k slots ≈ 30.3 GiB — geometry refuses; the pack form is the constraint | The one-boot falsifier (guard names the cap), then the mixed-form cold-tier design already drafted |
| 6 | **M3c ≥ 3.0 tokens/window** (banked 2.18–2.41) | The lever (depth) is proven but unadopted; position-1+ acceptance unknown | Falls out of #1's stair publish |
| 7 | **R3 anchored residency** (boot contract's remaining plank) | ROADMAP Phase 1, never entered the tree | Design + battery when the perf queue pauses |
| 8 | **spec-min-p 0.70 arm** (upstream measured +4 % on their engine) | Not yet run on ours | One env battery |
| 9 | **Verifier-golden-harness / specdec unlock** | Specced (F-448), not built | Build task as specced |
| 10 | **Q2_0-shadow divergence falsifier ladder** | Diagnosed (F-450), ladder recorded, not run | Run the ladder; likely ends in a scoping rule (shadow off for deterministic lanes) |
| 11 | **n-gram/context draft source** (T47 §3.7; +5–25 % on copy-heavy traffic, 0 % on open-ended) | Never attempted | Zero-boot falsifier first: hash-lookup hit-rate over harvest traces |
| 12 | **Phase-scoped persona battery verdict** | **In flight at time of writing** (`m1-phase-2026-11-24`: A0 legs done, B1 mid-run) | Do not disturb; read the means when it closes |

---

## 5. What has been learnt (the transferable lessons)

1. **The cheapest falsifier scales.** A one-boot microbench (F-411) retired a ×1.5–1.7 port ladder that the
   external dossier had made look rational. External intel re-prices; it never executes (law 12 held).
2. **Derived bandwidths need an anchor.** "185 GB/s achieved" was a division of a mis-measured numerator (2.66 MB
   pack form vs 1.38 MB cache form) over an assumed denominator (kernel-bound wait). The honest comparator is a
   measured uint4 stream on the same allocation (1493 GB/s), not the datasheet.
3. **The wall was structure, not silicon.** The same engine on the same boot idles 8.79 ms/window solo and 0.51 ms
   concurrent — rows in flight, not kernel quality, separate the lanes. Structure beats squeezing, at the OS
   scheduling level as much as the pipeline level.
4. **Determinism is an organ-level property.** The lottery was not one bug but four (sampling-seeded warm-in,
   unchecked host end-state, an allocation-path flip, a fallback layer under refusal) — and the fix that closed it
   was declaring organs and making their failure fatal, not retrying the slow path.
5. **Offline replay prices direction, not magnitude.** Calibration's +2.1 pp offline hit share bought +1.5 % live
   (F-412): the term it feeds was smaller and partially hidden in the live window. Price the live anatomy before
   trusting any replay's ceiling.
6. **Shape is not routing.** Domain-matched warm-in text does not close expert-set overlap (F-413); a routing-set
   warm-in (the profile's ranked pairs pre-staged) is the only form that could — and it is not worth building while
   leg 1 is discarded by protocol.
7. **Instruments rot silently.** The per-depth counters read zero correctly — they instrumented a drafter the
   standing no longer runs (F-414 note, F-415). An instrument that cannot see the path that runs voids the verdict
   it would produce; revalidate the instrument on every standing change.
8. **The battery is itself a hazard.** Batteries poison the learned profile (F-271), and the one miss of the arc —
   a missed re-pin — thrashed the operator's serving boot at exactly the documented 2.2× class (F-417). The
   structural fix (md5-verified re-pin in the ladder's close-out) is now part of the instrument, like the trap
   restore. Tools that depend on memory are defects.
9. **Deployment is verified by hash, not by intent.** A stash-provenance rebuild once deployed a stale binary and
   the telemetry "failed" correctly-silently (F-415). The deployed process is the change; its build is proved by
   hash mismatch, its patch by anchor assertion.
10. **Quality and performance are one Pareto rule, and dose is a measurement.** The vector arc's 0.45 → 0.3
    walk-back, the read-domain recalibration, and the phase scoping were all interleave verdicts; MMLU found the
    edge each time before the user did (F-450 arc).
11. **Retraction is part of the result.** This arc's earlier sessions produced two wrong diagnoses (a
    page-cache/drop-caches theory and a learned-profile-divergence-as-speed-theory) before the organ flip was
    found; both were retracted on the record with cause, and the F-411 acquittal then falsified the arc's own
    initial suspect in turn. The record that admits its misses is the one that can be trusted on its hits.
12. **Upstream is a lever library and a control group.** Their best-tuned stock (163.1) prices our lead (+29–41 %);
    their defaults audit ours (ADAPT_FETCH default on ≤16 GiB cards — production-proven at their end, correctly
    scoped at ours); their field logs port as hazard closure (rewind-keep, eviction stops).

---

## 6. What is NOT clear (open questions, with the experiment that would answer each)

| # | Question | What would answer it | Cost |
|---|---|---|---|
| 1 | Does depth-2 clear the +4 % bar at a quiet box? (Mechanism proven; round-1 +6.8 % vs round-2 −2.0 % flip is unexplained beyond load) | The named re-test battery (five quiet boots, means only) | One battery |
| 2 | What is the slot drafter's position-1+ acceptance? (Depth-0 is 59.8–64.6 %; positions 1–4 have never been tested) | The stair publish inside #1's battery | Free (same boot) |
| 3 | What is RouterLookahead's real recall on the current standing? (The +21 % ceiling assumes it predicts the misses) | `STRATA_LOOKAHEAD_STATS=1` boot | One boot |
| 4 | How much of the 8.79 ms idle can host-stage surgery actually reclaim vs only depth? (waitCPU is ~1.35 ms exposed; the rest is rows) | #1 + #3 + #4's composed verdicts | Three batteries |
| 5 | Is the conc quiet bank ~312 (the F-414 single read) or the 305.9-class envelope? | Five-boot quiet re-bank | One battery |
| 6 | What is the true VRAM ceiling for coverage, and does the mixed-form cold tier clear it? | The 16k-slot refusal boot (guard names the cap), then the tier's parity battery | One boot + build |
| 7 | Does the phase-scoped persona hold MMLU at stock with the calibrated dose? | The in-flight `m1-phase-2026-11-24` battery + its MMLU gate | Running now |
| 8 | Why did the F-416 round-2 control run hot (off2 M1 226.6 against a +6.8 % arm)? | Unresolved; the quiet re-test implicitly tests the "load moved" explanation | — |
| 9 | Dossier items flagged unverified at source and never laundered: the `moores` PR number, NVIDIA graph-launch µs (secondary relay), Chips-and-Cheese L2 tables | Primary-source verification if ever load-bearing | Reading |
| 10 | The exact split of the Q2_0-shadow divergence between shadow quantization, GPU/CPU path numerics, and client seeds | F-450's recorded falsifier ladder | One battery |

None of these blocks the queue's head; items 1–3 are one quiet window's work.

---

## 7. The forward queue (priced, ordered)

Per `levers-2026-11-20-evening.md`, F-418's correction, and the standing-line ledger — each entry priced before it
runs (law 15):

1. **Depth-2 quiet re-test** (env-only, no build; ceiling = the wall itself; MMLU ×2 on adoption).
2. **Lookahead falsifier** (one boot; kills or prices the +21 % overlap arm).
3. **Pool-granularity fold** (~+10 % class; parity harness first; composes with 2).
4. **Conc quiet re-bank** (protocol debt; likely confirmatory).
5. **Coverage: the refusal boot, then the mixed-form cold tier** (the 99 % hill by cache form, not slot count).
6. **spec-min-p 0.70** (one env battery; upstream +4 % precedent).
7. **Verifier-golden-harness** (the specdec unlock, specced in F-448).
8. **R3 anchored residency** (ROADMAP Phase 1) and the standing-line's Tier-B quality items, interleaved as quiet
   windows allow.

## 8. In flight at time of writing

The `m1-phase-2026-11-24` battery (config-A/B, 4 boots × 3 legs, pinned profile) is mid-run on the standing lane —
A0 leg 3 read M1 223.8 / conc 267.7 / M3b 77.2 %; B1 legs 2–3 read 236.2/215.9 with load disclosed 3.4–6.3. Per the
launch law and the interleave law: it is not disturbed, and no verdict is drawn from its partial legs.

## 9. Conclusion

This arc set out to make restarts boring and speed legible. It delivered both: boots are now certified objects
(fingerprint, organ, refusal contract, spread 1.5 %), and the performance problem has been reduced from "scattered
100–170 t/s, cause unknown" to one falsified suspect, one equation, and a priced queue whose head is a single
env-gated battery away from the largest M1 lever on the record. The year's deeper result is methodological: every
number above survived an interleave, every negative is banked with its re-open clause, and the two false starts of
the arc are on the record beside the verdicts that corrected them. The wall is now a doorway with a queue in front
of it — and the queue is one quiet window deep.

---

### References (primary)

- Journal: `SVG-CUSTOM.md` F-395–F-450 (this arc), esp. F-404/F-405/F-408 (determinism), F-411 (acquittal),
  F-412–F-416 (hypothesis ladder), F-417 (profile hazard), F-418 (upstream calibration), F-419–F-449 (hardening),
  F-450 (Pareto rule).
- Artifacts: `campaign/research/perf-hunt-2026-11-20.md` (`695bf9c7f`),
  `campaign/research/gpu-inference-optimization-dossier-2026-11.md` (`6776d492f`),
  `campaign/research/STRUCTURAL-AUDIT-2026-11-20.md`, `campaign/research/levers-2026-11-20-evening.md`,
  `campaign/research/ROADMAP-RELIABILITY.md` (`608501819`), `campaign/standing-line.md`,
  `campaign/tracker/lanes.csv` (chart of record).
- Evidence: `campaign/results/spread-f404-*.txt`, `spread-f405-bank-125807.txt`, `f405-envladder.out`,
  `f409-refusal.out`, `f411-moe-hit-sat.txt`, `f412-h3-calib-v2-ladder.out`, `f413-h4-corpus-ladder.out`,
  `f416-h5-depth2-ladder.out`, `f410-conc-rebank-quiet-v2.txt`, `mmlu-f405-hostalloc.json`.
- Commits: `695bf9c7f`, `6776d492f`, `608501819`, `a2dce9285` (push verification), `05a5c613c`,
  `0bd36187e`→`ba13d1acb` (vector arc), `66a0c6beb` (store closure).
- Instrument sources: `bench/micro/moe_hit_sat.cu`, `src/kernels/cuda/s2_expert_grouped.cu:322-402`,
  `src/kernels/cpu/pool.cpp:1040-1051`, `expert_source.cpp` (organ + refusal sites).
