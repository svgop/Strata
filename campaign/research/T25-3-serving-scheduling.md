# T25-3 — SERVING, SCHEDULING & ADMISSION: what basalt / Xolotl / maya do that beats ours

Date: 2026-10-20. Scope: every serving-layer mechanism in the three reference repos (basalt `fb0944c7`,
Xolotl `cd965e3c`, project-maya `35bb2b2` = v1.1.20; MANIFEST.md pins) that is not already in our tree
(strata-custom-2 standing conc config `campaign/cohorts-iq3_s-8081.json` + `launch-custom-iq3s-8081-conc.sh`),
with mechanism / their evidence (file:line + bench docs) / applicability at our scale (single 5090, 6-12
clients) / expected gain / conflict surface with the COHORTS reactor (generate.cpp:9112-9129, F-233) + F-14
admission (serve/server.py:1141,1655-1690) / cheapest falsifier. READ-ONLY session: no builds, boots, GPU, or
writes outside campaign/research/. Companion dossiers: T17B (our baseline + external papers), T23-* (recon),
T13B (lanes), T23-deepinfer (DSpark varlen — maya implements its cousin, §8).

---

## 0. EXECUTIVE SUMMARY

1. **Our serving layer is the family's most advanced where it matters (slot MTP drafts + cohort overlap);
   the real gaps are three, all basalt's:** (a) **elastic KV that composes with batch slots** — their
   per-session 32k backing + per-request take-back turns the expert-cache tail into a KV/VRAM lender
   (running.md:19-26, generate.cpp:2963-2967); ours refuses (`--kv-grow` hard-requires `o.batch == 0`,
   generate.cpp:5201) and we run fixed `--kv-resident 32768`; (b) **prompt-read-beside-windows shipped
   always-on with the windows' stream at high priority** (commits 81ab8aef, 95f3f535) — the exact mechanism
   T17B priced as candidate 4 (+6-8% conc, admissions = 14% of probe wall); ours has the machinery in-tree
   but opt-in, experimental, and **structurally refused by our own armed `--pcie-mode dma`**
   (generate.cpp:13812-13848); (c) the tier-decay default fix (0.93 single-card, basalt generate.cpp:5721)
   — moot for us: our config already arms `--adapt-decay 0.85`; only `--adapt-every 1` remains untested.
2. **Two T17B leads are closed by the family's own measurements, saving battery slots:** ForesightSwap
   (STRATA_FS_SLOTS) measured 0.91-0.98x everywhere and left off by Xolotl (DETAILS.md:1736-1738) — do not
   boot it; blocking-tier-vs-async-tier is already our banked shape (their async table corroborates, Xolotl
   BATCHING.md:165-180), and our remaining DMA_BATCH-vs-adapt-async trade stays T17B candidate 9 (±1-3%).
3. **Their headline conversation-pool win is NOT a delta:** Xolotl's 47.5 → 3.96 s/turn 5090 bench
   (README.md:57) is the ds41 serve-loop park/unpark + partial-LCP unpark; our Qwen-lineage cache already
   does longest-prefix matching across parked AND active state (conversation_cache.hpp:187,254-259) at 49 GiB
   vs their 8 GiB default. Their genuinely new robustness piece is the **runtime VRAM-aware degradation**
   (calibrated `need(B)` cap, half-width retry, cache-seed floor/skip: gate 3/3 turns vs HTTP 400, ds4
   V41_START_HERE 13c) — we fail instead of degrading on a busy card.
4. **maya's answer to queue pain is fair-slice preemption, not deadlines:** ordered capacity FairLock +
   `STRATA_FAIR_SLICE_S` + engine `PAUSE keep` (server.py:141-235, 2103-2260; generate.cpp:1131-1190;
   measured: short question during a 700-token answer got first word 10 s vs 60 s, 2x V100, CHANGELOG #93).
   Ours has neither short-first armed nor any slice: at 7+ clients the 7th waits a whole answer. Fairness
   only — scoreboard conc is untouched; the short-first env we merged from v0.2.1 is the cheap first step
   and is currently OFF in our launcher.
5. **Cold-start: nobody in the family has a COLD_OVERLAP-class mechanism** (overlapping the cold expert
   refill with serving). Closest: maya's pack-side `expert-profile.bin` written after EVERY answer and read
   at start (SETTINGS.md:439-447) — the warm-boot pattern our T17B candidate 2 already prices (+10-13%
   first leg); and their DISK_SKIP_AUTO "cold mode" (auto-off degradation, SETTINGS.md:404-421) changes
   answers → gate-forbidden here. Request coalescing: **absent from all four trees** (searched, §9).

**Ranked table** (gain bands are at our scale: single 5090, 6 clients, conc scoreboard 315.7
[SVG-CUSTOM.md:8448]; "cost" = port cost in our tree):

| # | mechanism | repo + anchor | their evidence | applicability | expected gain | cost | conflict surface | cheapest falsifier |
|---|---|---|---|---|---|---|---|---|
| 1 | Prompt-read-beside-windows, always-on + windows' stream priority | basalt 81ab8aef, 95f3f535; generate.cpp:5190, 6432-6435; running.md:28-31 | shipped default in the same-model engine; commit texts; no published conc table | **direct**: machinery already in-tree (STRATA_PIPED_CONCURRENT, generate.cpp:13812) but refused by our `--pcie-mode dma` | T17B candidate-4 class: **+6-8% conc** (admissions 14% of wall) | S (arm) to M (drop the dma exclusion) | F-233 staging fences (reads already fenced); adaptive tier stands still during read; borrow refunds after | one env boot `STRATA_PIPED_CONCURRENT=1` with `--pcie-mode kernel` (or STRATA_DMA_BATCH=0 leg); read the refusal line if any |
| 2 | Elastic KV under batch slots (per-session 32k backing, per-request take-back, give-back) | basalt running.md:19-26, generate.cpp:2963-2967 + kv_fit; commits 8ded6974, 7388717e, c9d1705b | their slot math: sessions come "out of the expert cache", elastic line `0.6 / 4.4 GiB (elastic)`; no conc table | **direct and load-bearing**: funds T17B candidate 1 (parallel-8's +1.46 GiB) without shrinking the cache; refines our `--kv-resident 32768` (streams mid-depth; elastic maps on demand) | unlocks parallel-8 (**+15-25%** per T17B #1) and/or −kv-stream tax at 30-60K depth; alone ~+0-2% | M-L | `--kv-grow` gate (`o.batch == 0`), kvg_ensure OFF (T15B), VMM availability, F-183 slot-slope | zero-code: log per-request KV depth in the conc probe (how many sessions exceed 32k cells?) — sizes the stream tax first |
| 3 | `--adapt-every 1` at decay 0.85 (tier follows the conversation fastest) | basalt configuration/index.md tuning note (5090+5060 Ti, 400 W, ABBA) | +3.6% prose after topic switch, +4.2% conversation, neutral structured; decay ≥0.97 −3-6% | **direct** (our GPU, our tier; we arm 0.85/every-4) | **+2-4% conc** on topic-switchy traffic; ~0 on our fixed probe prompts | S (env) | none (count thresholds unchanged; swaps hide behind cohort windows already) | one 3v3 conc boot with `--adapt-every 1` |
| 4 | Runtime VRAM-aware admission: calibrated `need(B)` cap, half-width retry, seed floor/skip + retry-once | Xolotl ds4 V41_START_HERE 13c (gate: 3/3 turns, 0 errors, 4 GiB foreign holder; old binary: HTTP 400 turn 1) | measured gate + `serve: VRAM 11.26 GiB free -> prompt batch 6428 (of 12288)` live line | partial: our card is dedicated, but ctxoff/1M legs and desktop neighbours (their Whisper case) hit the same wall; we fail the request today | no throughput gain; removes a failure class | M | expert-cache sizing order at boot; F-183 slope if cache shrinks live | none cheap in our tree (no runtime cap today); their calibration recipe is the spec |
| 5 | Fair-slice preemption (`STRATA_FAIR_SLICE_S` + engine `PAUSE keep`) behind an ordered capacity FairLock | maya server.py:141-235, 2103-2260; generate.cpp:1131-1190, 1836; CHANGELOG #93, #314 | short question during 700-tok answer: first word 10 s vs 60 s (2x V100) | 7-12-client interactive lanes only; scoreboard conc unaffected | TTFT of queued clients (60→10 s class), **0% conc** | M-L | F-14 tickets (order law), slot set stability under COHORTS staging, our slots keep state only at BDONE/park | none config-only (needs PAUSE verb); first step is arming the already-merged `STRATA_ADMIT_SHORT_FIRST` (S, env) |
| 6 | ds41 park/unpark with size gates + partial-LCP unpark | Xolotl ds4 V41_START_HERE 13b/13c (`STRATA_SERVE_PARK_MIN/MAX`, `_PARTIAL`; run19: reused 21,165 of 21,166 tokens) | README.md:57 (47.5→3.96 s/turn, 5090, 3 agents 30-60K); park gate tokens-IDENTICAL | **not a delta**: our cache LCP-matches parked+active (conversation_cache.hpp:187,254) | — (theirs was ds41-serve-loop-specific) | — | — | — |
| 7 | Conversation-affinity queue jump (a slot-resume admission may go first) | basalt running.md:8-9 | doc'd policy; no numbers | small: our pick_slot (server.py:2107) already routes the turn to its slot AFTER admission; the jump only saves queue waits | tail-TTFT of continuations; ~0 conc | S-M | F-14 strict order (the ticket deadlock is why it exists) | skip unless lane 5 is armed |
| 8 | Verify arbitration controller (none/lookup/MTP + depth k per step, EMA acceptance, min-gain gate) | maya src/spec/controller.cpp (CostModel::step_ms with measured U(n) distinct-expert growth; choose/observe) | wired at generate.cpp:6339-6340, 6866; their parallel k-depth finding: k=2/k=4 REDUCED throughput at 4 requests (SETTINGS.md parallel section) | concept-level: the principled online gate for T17B candidate 8 (MTP-off-under-cohorts ±2-4%) | decides the F-231 generalization without burning boots | M (new component) | exact-match contract (F-30b rescue off), COHORTS collect cap | T17B candidate 8's one config boot first; the controller is the follow-up if the curve is non-trivial |
| 9 | LRU expert eviction for single-conversation lanes (`STRATA_GLM_VRAM_EVICT=lru`) | maya CHANGELOG #210 | 24.8→26.5 tok/s (+6.9%) on 3x V100 128K, ONE conversation; **under `--parallel` the old rule stays (recency hurt)** | conc lane: DOESN'T (their own caveat); M1 solo lane: marginal at 97-99% hit | ≤+1% M1-class | S (env if the knob ports; ours has no such env) | F-183 cache-slope; our tier is multi-conversation | skip at 6 clients |
| 10 | Pack-embedded / pack-side placement profile, saved per answer | basalt model-file.md:13,31 (profile member of `.basalt`; "significantly slower" without); maya SETTINGS.md:439-447 (`<pack>/expert-profile.bin`, saved after each answer, counts ≈2,000 tokens prior) | both ship it as the warm-boot default | ops form of T17B candidate 2 (we save every 30 requests, boot from the static file) | T17B candidate-2 band: **+10-13% first leg, +4-6% five-boot mean** | S (point `--expert-profile` at the learned file; packaging later) | none (F-183 band ±500 slots) | the standing T17B falsifier: one boot loading `data/expert-profile-learned.bin`, read first-request hit rate |
| 11 | Drafter weights off-card during long prompts | Xolotl README.md:33-34 ("draft off the card during long prompts… default since 2026-10-17": 187/317 vs 178/268 tok/s) | ds41 DSpark 2.91 GB → ~5 s of a 20K prompt (V41_START_HERE 13b) | low: our prefill is 3,430 tok/s at 6.5K, drafter ~0.24 s load, prefill not the bottleneck | ~0 (maybe TTFT on 1M legs) | M | verify windows need the drafter back (0.15 s class each way, their number) | skip |
| 12 | ForesightSwap miss slots (STRATA_FS_SLOTS) | in OUR TREE byte-identical (foresight_swap.hpp SAME); Xolotl DETAILS.md:1736-1738 | their A/Bs: 0.95/0.91 (3060), 0.97/1.00 (P100), 0.98 (5070) — **"Left off"**; 0 bad in 2,300+ verify reads | **closed**: 2.0-2.3% miss, pool hidden behind cohort windows; their boxes measured it negative even at higher miss | negative to ~0 | 0 (env exists) | also excludes STRATA_PIPED_CONCURRENT (generate.cpp:13821) | none — do not spend a boot |
| 13 | Request coalescing (merge duplicate concurrent prompts) | — | **absent from all four trees** (kernel-level "coalesce" hits only) | — | — | — | — | — |

---

## 1. OUR BASELINE (what the comparison is against)

Standing conc config (`campaign/cohorts-iq3_s-8081.json` + `launch-custom-iq3s-8081-conc.sh`): parallel 6,
COHORTS=2 two-verifier reactor (generate.cpp:9112-9129; slots [0,3)/[3,6); round = stage A+B → go A → go B →
batch_poll round-robin verify.cpp:3345-3425), slot MTP drafts on (`STRATA_BATCH_MTP=1`, coupled depth-1,
67% acceptance), `--kv int8 --kv-resident 32768`, `--prefill auto:32768`, `--pcie-frac 0.35 --pcie-mode dma`,
`--conversation-cache-mib 49152`, `--adapt-decay 0.85` (every-4 default), profile save armed but boot reads
the STATIC profile (`--expert-profile data/expert-profile.bin`), expert tiers VRAM 9,366 slots + pinned-RAM
41.9 GiB complement, scoreboard conc **315.7** (319.8/311.3/303.3/321.6, SVG-CUSTOM.md:8448).
Admission: F-14 strict FIFO tickets (serve/server.py:1141, 1655-1690; `ctl_ticket`/`ctl_served`/
`ctl_abandoned`); the upstream-merged `STRATA_ADMIT_SHORT_FIRST` + `STRATA_ADMIT_MAX_WAIT_S` + queue
deadline (`QueueWaitExpired`) are present but NOT armed in the launcher; slot lifecycle: pick_slot
held-prefix→empty→LRU (server.py:2107+), orphan-slot watchdog (#1603, server.py:1630-1643), solo re-entry
(SOLO_AGAIN, server.py:1735+). Cold start: boot-time page-cache drop defeats prewarm (T17B §1.2,
expert_source.cpp drop_file_cache) — documented, unfixed.

## 2. basalt — batch slots, tier placement, pack profiles (same-model sibling, sm_120)

### 2.1 Prompt read beside decoding slots, always-on, with stream priority — their sharpest serving delta

- Mechanism: with slots active the prompt path runs on its own thread while the slots' windows keep
  launching; the windows' main stream is raised above the prompt path's kernels (`ver.set_main_priority
  (o.batch > 0)`, basalt generate.cpp:5190; verify.hpp:118); the few steps that touch the draft layer or the
  output run between two windows (running.md:28-31). Commit 81ab8aef removed their overlap_on gate ("with
  --batch a prompt is ALWAYS read beside the decoding slots"); 95f3f535 added the priority.
- Our tree already contains the same machinery as `STRATA_PIPED_CONCURRENT` — opt-in, experimental
  (generate.cpp:13812-13848): prompt thread + doorbell-served pump thread, bell_cv wake semantics
  (:12063-12069), reader-thread CPU unpinning (:13864+). But it REFUSES beside: `STRATA_PREFILL_HELP`,
  `STRATA_SPLIT_MTP_BATCH`, `STRATA_FS_SLOTS`, `--pcie-mode dma`, helper caches, staged/other sources
  (:13838-13845). **Our conc config arms `--pcie-mode dma` (cohorts json) — so today the mechanism
  structurally never arms on our lane.**
- Applicability: direct — this is T17B candidate 4 (admissions 1.6 s of an 11.56 s probe wall = 14%;
  hiding half priced +6-8% conc), now with a second team having shipped it on the same model.
- Expected gain: +6-8% conc (T17B's pricing; their repo publishes no conc table to tighten it — noted in §7).
- Conflict surface: the dma exclusion exists because doorbell service makes a CUDA call (:13819-13821);
  dropping it means moving doorbell DMA to the copy path or the kernel path (pcie-mode kernel) — T13B priced
  DMA_BATCH at 84.9% submit-cycle savings, so the A/B is dma+no-overlap vs kernel+overlap. The adaptive tier
  stands still during a read (their design too) and the prompt's borrowed slots are seen missing by the
  windows (CPU computes them) — same as ours.
- Cheapest falsifier: one 3v3 conc boot with `STRATA_PIPED_CONCURRENT=1` and `--pcie-mode kernel`
  (or `STRATA_DMA_BATCH=0`); the engine prints the refusal reason if any binding exclusion remains.

### 2.2 Elastic KV that composes with batch slots

- Mechanism: a serve on one GPU backs only the first 32,768 cells of each session's K/V at start
  (`kKvElasticFloor`, basalt generate.cpp:2963-2967); the rest of that VRAM holds the expert cache's last
  slots; a request backs prompt+16,384 cells BEFORE its prompt is read by taking those slots back (experts
  become CPU misses served from the RAM copies kept for exactly this), and gives cells back when a later
  request needs less, the cache refilling from RAM (running.md:19-26, `kv_fit`); the drafter's K/V is elastic
  too (c9d1705b); startup line `elastic KV: ...`, every move logged.
- Ours: `--kv-grow` exists but requires `o.batch == 0` (generate.cpp:5198-5201, "the batch slots carve their
  own K/V") and is opt-in; upstream v0.2.1.4 has the same exclusion (Strata generate.cpp:5117-5122). We run
  `--kv-resident 32768` instead: fixed 32k VRAM window, everything deeper streams from pinned RAM
  (kv-stream tax: F-181 −11% decode at 244k depth; T15B).
- Applicability: high-value exactly where T17B is blocked: candidate 1 (parallel-8) needs +1.46 GiB for two
  more slot sessions and refuses to shrink the expert cache (F-183 slope ±513 slots = ±4.9% conc); elastic
  backing is the third way (fewer permanently-resident KV cells per slot, take-back on demand). It also
  beats kv-resident for mid-depth (30-60K) sessions, which today stream every attention.
- Expected gain: enabling-class — parallel-8's +15-25% becomes fundable; standalone vs kv-resident, the
  kv-stream tax on mid-depth sessions (unmeasured at our depths; Xolotl measured kv-streaming +5% decode
  512K but that is the other direction, maya README).
- Conflict surface: VMM availability (ours: `vmm_available()`, layer.cpp:681), kvg_ensure OFF (T15B law),
  the batch carve comment, F-183 band; COHORTS verifiers are untouched (graphs don't own KV sizing).
- Cheapest falsifier (zero-code): instrument the conc probe — per-request max KV depth; if no session ever
  exceeds ~32-48k cells, kv-resident already costs nothing and the elastic port drops to the parallel-8
  funding question only.

### 2.3 Tier placement: decay default and the measured tuning point

- Mechanism/evidence: basalt ships effective default decay 0.93 on one GPU (`fast_decay =
  o.adapt_decay_set ? o.adapt_decay : 0.93f`, generate.cpp:5721-5723 — a basalt-only substitution; upstream
  v0.2.1.4 and our tree apply the raw 0.7 code default). Their tuning note (configuration/index.md, RTX 5090
  + 5060 Ti peer, both 400 W, ABBA 4 processes/arm): `--adapt-every 1 --adapt-decay 0.85` = +3.6% prose
  after a topic switch, +4.2% conversation, neutral structured; decay ≥0.97 measured −3-6% (single runs,
  ~3% noise). Thresholds: candidate needs count ≥2 and 1.5 above the victim; copies land a few windows
  later (their doc = our behavior).
- Ours: 0.85 armed in the conc json; `--adapt-every` default 4. The remaining untested cell is every-1.
- Applicability: direct (their bench GPU = ours). Expected +2-4% on topic-switchy traffic; likely ~0 on our
  fixed probe prompt set (the tier already holds 98%+ hit). Cost S; conflict none; falsifier one 3v3 boot.
- Related, Xolotl's same-shape table (BATCHING.md:165-180, 2x 3080 layer split): async tier 89.2 vs blocking
  74.7 vs frozen-from-learned-profile 69.5 vs off 45.2 tok/s — corroborates (a) the adaptive tier's value in
  streaming regimes, (b) async ≈ +19% over blocking WHEN the blocking round stalls windows 8.4 ms; our adapt
  join is 0.25 ms/window, so our residual is only T17B candidate 9 (adapt-async vs DMA_BATCH, ±1-3%).

### 2.4 Pack-embedded placement profile

- The official `.basalt` pack embeds the expert profile (+hc8): "A pack you convert yourself from a GGUF,
  without the official files' expert profile … will be significantly slower" (model-file.md:13,31). maya
  stores the profile BESIDE the pack (`<pack>/expert-profile.bin`), writes it after EVERY answer, boots from
  it, and seeds fresh packs from a generic mix worth ~2,000 tokens (maya SETTINGS.md:439-447).
- Ours: save side armed (`--expert-profile-save data/expert-profile-learned.bin --expert-profile-save-every
  30`, cohorts json) but the boot still reads the STATIC profile — T17B candidate 2, still open, their
  packaging is the ops-grade form. Gain band stays T17B's: +10-13% first leg, +4-6% on five-boot means.
  Falsifier unchanged: one boot pointing `--expert-profile` at the learned file; read first-request hit rate
  (today 88.1-88.4%) and cold-leg swap counts.

### 2.5 Also in basalt, verified shared or not applicable

- Conversation-affinity queue jump (running.md:8-9): arrival order except a slot-resume admission may go
  first. Our pick_slot gives the affinity AFTER admission; the jump is a small admission-priority exception
  that fights the F-14 ticket law (built to kill the wake-and-race). Skip unless fair-slice work lands.
- `pcie-mode direct` — the grouped kernel reads the mapped arena directly (configuration/index.md); present
  in our tree (generate.cpp:555 ple_io is a different knob; pcie-mode parsing shared). Our config pins dma;
  the T15C untested-pcie-frac note plus this mode choice make ONE pcie-mode A/B (dma vs kernel vs direct)
  the natural companion leg to §2.1's boot.
- Memory floors + serving-time RAM guard (running.md:74-91): per-request MemAvailable check stops the
  request, server keeps serving. Ours: conversation-cache parking has the cgroup-aware floor (shared
  lineage, DETAILS-identical code); the generalization to every request is a robustness nicety.
- Elastic vision borrow (running.md:107-109) — same lend mechanism; we don't serve vision on the conc lane.

## 3. Xolotl — conversation pool, admission control, the arena

### 3.1 The 47.5 → 3.96 s/turn bench: what it actually is

README.md:55-57: "Conversation pool (#19, #20): every conversation is kept at its turn boundary in pinned
host RAM (`--conversation-cache-mib`, default 8192) and a request resumes from the longest matching prefix,
pictures included … measured on a 5090: three agents at 30-60K tokens, 47.5 → 3.96 s a turn." That is the
whole published bench (one README line; the mechanisms live in the ds4 session docs). Our tree already has
the same family: park/restore with longest-prefix matching across parked AND active containers
(conversation_cache.hpp:187 `conversation_prefix`, :254-259 best-prefix pick), turn checkpoints, min-token
parking, cgroup-aware RAM floors — at 49,152 MiB. The ds41-specific fixes (exact-prefix demand → partial
unpark, `STRATA_SERVE_PARK_PARTIAL`, V41_START_HERE 13c; Engram writeback) were repairs to THEIR serve loop,
not capabilities we lack. Verdict: NOT a delta; do not "port the pool".

Their park gating is still worth copying as policy hygiene if our cache ever thrashes: park only if
≥`STRATA_SERVE_PARK_MIN` (2048) tokens and prompt+max ≤`STRATA_SERVE_PARK_MAX` (65,536 positions ~1 GB),
drop the park when a request writes past it (V41_START_HERE 13b). Ours has min-tokens; no max-size gate
(our 49 GiB budget makes it moot at our scale).

### 3.2 Admission control: runtime VRAM-aware degradation (the one piece we lack)

- ds41 serve sizes the layer-major prompt batch from VRAM actually free with a CALIBRATED cost model
  (`need(B) ≈ 4.3 GiB + 0.95 MiB·B`; corrected to a fixed 1,424 MiB B-independent graph buffer + per-column
  state, run20/run21 nsys), retries at HALF width on allocator failure, clamps the chunk stage from free
  VRAM with a 16 MiB floor, and floors/skips the expert-cache seed with one retry with the cache off —
  "never a failed request" (V41_START_HERE 13c; live line `serve: VRAM 11.26 GiB free -> prompt batch 6428
  (of 12288)`).
- Their gate: 3/3 turns, 0 engine errors with a foreign 4 GiB VRAM holder resident, vs HTTP 400 on turn 1
  for the same traffic on the pre-fix binary; quantified neighbour cost 233.9 vs 387 tok/s.
- Applicability: our card is dedicated, but the campaign's own logs show foreign CUDA contexts (T17B-era
  renders) and the ctxoff/1M lanes allocate near the ceiling; today our engine sizes at boot and fails the
  request (their run8's second failure mode). Port cost M; gain robustness, not scoreboard.
- Cheapest falsifier: none config-only in our tree — this is a spec to file, not an env to flip.

### 3.3 The "conversation-adaptive arena", resolved

No object by that name exists. The task's phrase maps to three real things, all measured in their docs:
(a) the VRAM tier following the conversation in batch windows (`--adapt-every`, BATCHING.md:144-155) — we
have it; (b) `--adapt-async 1` (tier round on a helper thread; measured table BATCHING.md:165-180: 89.2 vs
74.7 blocking vs 69.5 static-profile vs 45.2 off) — we have it, structurally disabled while DMA_BATCH is
armed (T17B candidate 9, ±1-3%, one diagnostic boot); (c) the ds41 pinned arena seeded in ROUTING-PROFILE
order with re-seed without file re-reads (ds4/ENGINE_MOE.md:40-44) — our complement pins everything, so the
ordering idea is inert at our scale.

### 3.4 Closed by their measurement: ForesightSwap

`STRATA_FS_SLOTS` (per-layer VRAM slots for missed experts, own copy stream, router look-ahead feed) is
byte-identical in our tree (foresight_swap.hpp diffs SAME; our generate.cpp:6692 already references the
#1348 look-ahead) and UNARMED. Xolotl's own A/Bs: decode median B/A 0.95 (0/6 pairs) misses-only and 0.91
with look-ahead on RTX 3060; 0.97/1.00 P100; 0.98 (1/6) RTX 5070; 2x 3090 same story — "in every box
measured the smaller cache cost more than the slots returned … Left off" (DETAILS.md:1736-1738). Correctness
was fine (0 bad in 2,300+ reads) — it is a pure economics loss. This closes T17B §2.5's "prefetch-predictor
ideas remain for the miss tail" lead WITHOUT a boot: our 2.0-2.3% miss with pool hidden under cohort windows
is strictly more favorable than their boxes. It also removes one exclusion from §2.1's list if ever revisited.

### 3.5 Slot lifecycle, Xolotl form

ds41 batch slots (BATCHING.md:305-366): slab tensors (all slots' rings per layer), windows carry decode rows
PLUS rows of a short prompt being read (`STRATA_SERVE_STEP_MAX` 96, ≤16 rows/window), routed experts run
once over the row union. 5090 ds41 table (8 slots 256K): 94-95 tok/s steady at 8, window 75.1 ms. The
prompt-rows-in-windows idea is their form of admission overlap; the transferable number is that they cap
mixed rows at 16 — OUR cap is `kVerifyMaxT=8` with fixed arrays (verify.hpp:351-374), binding at 10+ clients
(T17B §2.3), so mixing admission rows into cohort windows at 6 clients has no headroom without touching the
cap. Design input for candidate 4, not a port. Qwen-engine slots in Xolotl are the upstream ones we already
run (BATCHING.md's layer-split/`--batch-mtp` tables are upstream-lineage evidence, e.g. batch-MTP +15.4%
serial / −21.9% when experts fit VRAM — consistent with our COHORTS slot-draft economics at 98% hit).

## 4. project-maya — serving layer, queue/deadline path

### 4.1 Queue discipline vs ours (the assigned comparison)

- Ours: F-14 strict FIFO tickets (arrival order, no wake-and-race — built because the ticket deadlock is
  worse than reordering, SVG-CUSTOM-TODO.md:1533) + the v0.2.1 merge: `STRATA_ADMIT_SHORT_FIRST` (short
  prompts jump, 30 s anti-starvation, `serve/server.py:1650-1735`) + `QueueWaitExpired` queue deadline —
  all present, none armed in the conc launcher.
- maya: an ordered capacity FairLock (server.py:141-204: deque of [token, units], `wait_turn` holds place,
  exclusive holders take all units — images/updates/reloads) + `STRATA_FAIR_SLICE_S` time-slicing
  (:230-235, `_sliced` :2103-2260): once an answer has decoded ≥ slice seconds AND someone waits, the server
  sends engine `PAUSE keep` — the engine keeps the conversation's state with what it wrote
  (generate.cpp:1131-1190: ctl map rid→{STOP, PAUSE, PAUSE keep}; :1836 snapshot moves to the last fed
  token), the FIFO goes to the next request, and the paused one requeues at units=1 and continues from
  prompt+produced (only the delta is re-read; "a prompt read is ~30x a decode step's speed"). Heartbeats
  keep the client calm while requeued. No queue deadlines anywhere in maya; a client that goes away leaves
  the queue at once (#314 — ours: ClientGone + ctl_abandoned, equivalent).
- Their evidence: CHANGELOG #93 — a short question asked during a 700-token answer got its first word after
  10 s instead of 60 (2x V100). Test coverage: serve/test_fair.py (order + slices against a scripted
  resuming engine).
- Applicability at 6-12 clients: our slots are 6; client 7+ waits a WHOLE answer (no slice, no short-first).
  maya's mechanism is the strongest form of the fairness fix; the merged-but-dormant short-first env is the
  cheapest form. Neither moves scoreboard conc (the probe is 6 fixed clients; slicing can only add window
  churn). Cost of full slices: M-L (new PAUSE verb through our engine protocol + slot-state continuity
  mid-decode, which our slots do not do today — they keep state at BDONE/park boundaries). Conflict: F-14's
  order law, COHORTS staging assumes a stable active-slot set within a round; a PAUSE must land on a window
  boundary (their engine does the same).
- Cheapest falsifier for the CLASS: arm `STRATA_ADMIT_SHORT_FIRST=<tokens>` (env, S) and measure queued
  TTFT in a 7-client lane; the slice port is only worth it if that lane matters to the product.

### 4.2 The verify arbitration controller (serving-adjacent)

src/spec/controller.cpp: `CostModel::step_ms(k, mtp)` = dense(ratio table) + CPU-miss term with MEASURED
distinct-expert growth U(n) ("tokens per distinct expert = n / U(n)") + draft cost; priors from flyweight
measurement (0.86/draft) and offline lookup replay; EMA-learned per session; `choose()` scans lookup depths
then MTP depths and returns None unless best beats baseline by `min_gain`. Wired per request at
generate.cpp:6339-6340/6866, 7579-7603 (DraftPolicy). Their parallel finding (SETTINGS.md, several-requests
section): at 4 requests, draft depth k=1 ≈ noise vs legacy, **k=2 and k=4 REDUCED throughput** — the family
version of F-231/Su-et-al. Applicability: this is the principled online governor for T17B candidate 8 (is
the marginal slot-draft row still worth it under COHORTS? ±2-4% either way). Our depth is fixed by contract
(depth-1 coupled chains, exact match); the controller's VALUE here is as the measurement harness for the
F-231 generalization, not as a port (their lookup/none modes violate our contract's offer shape). Sequence:
T17B candidate 8's one config boot first.

### 4.3 Tier/lane mechanics (evidence, mostly DOESN'T at our operating point)

- LRU eviction for the single-conversation case (#210, CHANGELOG: 24.8→26.5 tok/s on 3x V100 128K) — with
  the explicit caveat that under `--parallel` the recency rule HURT (the latest conversation pushed out what
  all of them use). At our 6-client lane: DOESN'T by their own finding; M1 lane is at 97-99% hit where
  eviction policy is marginal.
- MAYA_DS4_STREAM (PCIe lane): per-token split of missed experts between CPU and GPU streaming, decided
  from per-PC measured times kept in `lane-times.txt`, re-measured under load, kept only while ≥3% faster
  (SETTINGS.md:366-388). Ours: static `--pcie-frac 0.35`. The transferable idea is measured-fraction
  selection replacing manual sweeps — T15C already lists pcie-frac as untested here; one 3-value sweep is
  the cheap form, the auto-calibrator is overkill for one box.
- MAYA_DS4_PROMPT_SKIP (layers 20-39 skip early prompt tokens): model-specific to DS4's windowed layers AND
  not bit-identical ("same up to expert-output add order") — F-230's prefill-bits law forbids the class
  outright. DOESN'T.
- MAYA_DS4_DISK_SKIP_W/_AUTO: changes the answers (drops light missed experts) — gate-forbidden; but
  DISK_SKIP_AUTO's SHAPE (cold-mode degradation that switches itself off once the SSD-miss share stops
  falling, re-arms only between requests) is the only cold-start-aware serving policy found in the family.
  Record as design context for a hypothetical quality-relaxed cold lane; not actionable under our gates.
- --calibrate (10-15 min hardware self-tune, incl. disk read-chunk sizes with a 3% keep rule; CPU-lane
  thread topology measured at start, 5% keep): an ops pattern, not a serving mechanism; our knob-battery
  scripts already play this role by hand.
- Parallel scaling anchor: x1.41@2 / x1.35@4 (V100, experts mostly streamed), x1.67@2 / x2.02@4 (all
  experts in GPU) — corroborates our F-183/COHORTS economics: gains grow with VRAM residency.

## 5. CROSS-CUTTING: cold-start (COLD_OVERLAP-class), slot lifecycle, coalescing

- **Cold start.** No repo has an admission/decode overlap that serves through the cold expert refill (the
  COLD_OVERLAP idea). The family's answers are: pack-side per-answer profile persistence (maya, §2.4) =
  warm boots; profile-ordered seeding (maya read-ahead order, Xolotl arena profile order) — inert for us,
  the complement pins everything; runtime degradation when memory is short (Xolotl §3.2); and maya's
  cold-mode quality trade (forbidden here). Our T17B package (post-boot prewarm + learned profile + layout
  self-probe) remains the plan; nothing in the references beats it, two pieces corroborate it.
- **Slot lifecycle.** The family's full set: conversation-affinity pick + LRU (ours: server.py:2107+,
  superset with pipeline-hole-fill); slot-as-conversation-cache with 50-60 ms restore (shared); park/
  unpark/drop with size gates + partial-LCP (Xolotl ds41, ours LCP-equivalent); elastic per-slot KV backing
  + give-back (basalt — ours lacks, §2.2); orphan-slot watchdog (ours, #1603 — Xolotl/maya have
  client-gone cleanup, none has the owner-leak watchdog). Nothing else found in any tree.
- **Request coalescing.** Absent from all four trees (only kernel memory-coalescing hits). If the campaign
  ever wants duplicate-prompt merging under 6-12 agents, it would be novel here.

## 6. CONTRADICTIONS AND THEIR RESOLUTIONS

1. **"We run decay 0.7" vs the conc json**: the code default is 0.7 (generate.cpp:764) but our campaign
   config arms `--adapt-decay 0.85` (cohorts json args). Basalt's doc-default-vs-code-default gap
   (0.93 substitution at :5721) initially read as our gap; resolved by reading OUR config. Residual
   candidate: `--adapt-every 1` only.
2. **"basalt reads prompts beside windows; we don't" vs our STRATA_PIPED_CONCURRENT**: both true — the
   machinery is upstream-shared and present in our tree, but opt-in + excluded by our armed
   `--pcie-mode dma` (generate.cpp:13812-13845). The delta is basalt's default-on + stream priority +
   fewer exclusions, not the mechanism.
3. **Xolotl's 10.8x side-call fix number vs their own caveat**: run15's 265.9 s was a capped batch on a
   loaded box, run19's 24.7 s a wide batch on a quiet one — they say so themselves (V41_START_HERE 13c) and
   rest the claim on `reused=0 → reused=21165`. Cited here the honest way; the mechanism, not the wall
   clock, transfers.
4. **mayas's parallel k-depth finding (k≥2 hurts at 4 requests) vs MagicDec/T17B #8 (spec pays when verify
   is bandwidth-bound)**: both hold — their drafter is DFlash2 (block-diffusion, expensive verify rows per
   token), ours is a coupled MTP head at 98% cache hit. Not a contradiction; a reminder that candidate 8's
   answer is empirical and may flip either way under COHORTS.
5. **ForesightSwap present-and-fixed in our tree vs measured-negative in theirs**: the code is identical
   (diff SAME) including #2134; the negative result is economics (cache slots lost > slots served), which
   our higher hit rate makes worse, not better. Closed.

## 7. OPEN QUESTIONS

1. Our KV-depth distribution on the conc probe (drives §2.2's falsifier and the parallel-8 funding path):
   instrument per-session max cells; zero code beyond a log line.
2. Does `STRATA_PIPED_CONCURRENT=1` + `--pcie-mode kernel` hold bit-exactness on the conc lane (their
   claim: window arithmetic unchanged; ours must pass the S3.3 token-equality gate)? One boot answers.
3. basalt publishes no concurrent-throughput table (their 324.6 is a solo decode window, unified-kernel.md:32)
   — our 315.7 conc at 6 clients may already exceed their served number; if a port decision ever needs their
   conc evidence, it does not exist publicly (searched docs/, bench/, README).
4. Whether maya's `PAUSE keep` can land on a COHORTS window boundary without a staging fence change —
   design review only if the fairness lane is armed.
5. maya's U(n) distinct-expert growth table (controller.cpp) — worth extracting as a prior for any future
   cost-model work on our slot drafts (their measurement, our model family).

## 8. IMPLICATIONS

- The two cheapest boots on this board are **`STRATA_PIPED_CONCURRENT=1` (with the pcie-mode trade)** and
  **`--adapt-every 1`** — both env-only, both attacking the two remaining conc terms T17B ranked first
  (admissions 14% of wall; tier responsiveness). They should precede any port.
- **Elastic-KV-under-slots is the enabling mechanism for parallel-8** and the only M-L-cost port on the
  board with a priced payoff (+15-25% class via T17B candidate 1); its zero-code falsifier (KV-depth
  instrumentation) should run first.
- Do NOT spend boots on: ForesightSwap (their negative), the conversation pool (ours is equal/better), cold-
  start alternatives (ours is already the family's best-documented plan), request coalescing (nothing to
  copy), DISK_SKIP/PROMPT_SKIP (gate-forbidden).
- The admission-policy layer (short-first armed, fair slices, affinity jump) is a product-lane decision, not
  a scoreboard lever: nothing here moves conc at 6 fixed clients. If 7-12-client interactive service becomes
  a goal, the order is short-first (env) → maya slices (port) with their measured 60→10 s TTFT anchor.
- Conflict law unchanged: every candidate gates on byte-exact differentials (or the standing 2x MMLU for
  routing changes); F-230 forbids the prefill-bits class regardless of their evidence.

## 9. METHODS NOTE (so nothing is repeated)

- Pins: basalt fb0944c7, Xolotl cd965e3c, project-maya 35bb2b2 (v1.1.20), upstream Strata v0.2.1.4
  (9de04cd4) — all per MANIFEST.md, worktrees clean, read-only.
- What was actually read: line-level — our serve/server.py (admission/slot/pick_slot), our
  generate.cpp (COHORTS init :9112-9129, reactor :10193-10433, PIPED_CONCURRENT :12063/:13805-13875, conv
  cache :9385-9540, kv-grow :5190-5205), our cohorts json + conc launcher; basalt running.md (whole),
  configuration/index.md (whole), model-file.md (greps), unified-kernel.md (whole), generate.cpp (:5190,
  :6420-6470, :2963-2967, :444-452), verify.hpp:118, commits 81ab8aef/95f3f535/8ded6974; Xolotl BATCHING.md
  (whole), DETAILS.md (:660-720, :980-1130, :1730-1745), README.md (:30-90), EXCHANGE_ROTATION.md (whole),
  KV_PREFETCH/DISJOINT_EXPERT_CACHE/MESSAGE_BOUNDARY_CACHE/PROMPT_CACHE_TAIL (headers), ds4 V41_START_HERE
  (:255-460), ENGINE_MOE.md (whole), foresight_swap.hpp (whole), conversation_cache.hpp diff; maya
  server.py (:141-260, :2095-2260), test_fair.py (header), generate.cpp (:1131-1190, :1836, :6339-6340),
  spec/controller.cpp (whole), SETTINGS.md (:267-470), CHANGELOG (greps + :238-250, :604-625), README
  (:135-215).
- Unproductive searches (do not repeat): "conversation-adaptive arena" as a literal — no such object in
  Xolotl (resolved to three mechanisms, §3.3); "COLD_OVERLAP" or any cold-overlap serving mechanism in the
  three repos — none; request coalescing — kernel hits only, no request-level mechanism anywhere; Xolotl
  3.96 bench detail beyond README.md:57 — no fuller write-up exists (docs/xolotl, cenote, HANDOFF greps);
  maya queue deadline — does not exist (fair slices are their answer); basalt concurrent tok/s table — not
  published; `pause`/`PAUSE` in Xolotl — BYIELD (#656) only, PAUSE is maya's.
- Traps avoided: the upstream/basalt/ours decay defaults differ (code 0.7 everywhere; basalt substitutes
  0.93 when unset; OUR CONFIG sets 0.85) — always check the campaign json, not the code default, for "what
  we run". `pcie-mode dma` in our json silently disqualifies two env mechanisms (PIPED_CONCURRENT,
  FS_SLOTS+concurrent) — check exclusions before arming anything.
- All file:line anchors above were read this session at the pinned SHAs; no network sources were needed
  (family-internal study); nothing was executed.

## 10. SOURCES

Local (all primary, read this session):
- OUR TREE: serve/server.py:1141,1630-1735,2107+; src/program/generate.cpp:9112-9129,10193-10433,12063-12069,
  13805-13875,5190-5205,9385-9540,764,2653-2654; src/core/layer.cpp:625-700; src/core/expert_source.cpp:3919;
  include/strata/core/verify.hpp:241; include/strata/core/conversation_cache.hpp:28-32,170-187,254-259;
  include/strata/core/foresight_swap.hpp (== Xolotl's); campaign/cohorts-iq3_s-8081.json;
  launch-custom-iq3s-8081-conc.sh; SVG-CUSTOM.md:8448; SVG-CUSTOM-TODO.md:1533;
  campaign/research/T17B-conc-leap.md (baseline pricing, all of it).
- basalt (fb0944c7): docs/running.md:5-41,62-72,74-91,93-123; docs/configuration/index.md:100-170;
  docs/model-file.md:4-31; docs/unified-kernel.md; src/program/generate.cpp:444-452,5190,2963-2967,
  6420-6470,5721-5723; include/basalt/core/verify.hpp:118; commits 81ab8aef, 95f3f535, 8ded6974, 7388717e,
  c9d1705b, b9147adc, 5ba36168.
- Xolotl (cd965e3c): README.md:30-90 (incl. :57 bench line); docs/BATCHING.md (whole; tables :41-64,:165-180,
  :224-263,:305-366); docs/DETAILS.md:660-720,980-1130,1730-1745; docs/EXCHANGE_ROTATION.md (incl. A/B table
  104.90→113.44); docs/KV_PREFETCH.md; docs/DISJOINT_EXPERT_CACHE.md; docs/MESSAGE_BOUNDARY_CACHE.md;
  docs/PROMPT_CACHE_TAIL.md; docs/ds4/V41_START_HERE.md:255-460; docs/ds4/ENGINE_MOE.md;
  include/strata/core/foresight_swap.hpp; serve/server.py (fifo-era admission, :215,:3137).
- project-maya (35bb2b2): serve/server.py:141-260,2095-2260,3721; serve/test_fair.py;
  src/program/generate.cpp:1131-1190,1836,6339-6340,6866,7579-7603; src/spec/controller.cpp (whole);
  docs/SETTINGS.md:187-470; CHANGELOG.md:26,238-250,604-625,1232,1322; README.md:135-215.
- Upstream reference: Strata v0.2.1.4 serve/server.py:310-317 (short-first provenance);
  src/program/generate.cpp:683,5117-5122 (decay default, kv-grow batch exclusion).
- MANIFEST.md (pins, identities).

Counts: family-internal primary sources 6 (four repos' code+docs+history at pinned SHAs); categories:
code 5 trees read, docs 20+ files, git history (basalt commits, Xolotl foresight history), bench/config
artifacts (our json/launcher/logs via T17B). External web sources: 0 (none needed; every claim is pinned
to read local file:line).
