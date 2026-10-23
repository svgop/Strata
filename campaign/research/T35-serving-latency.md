# T35 — THE SERVING/SCHEDULING/LATENCY LAYER: the request's pathway, measured stage by stage, and the next-tier levers for smoothness, tails, concurrency, and utilization

Date: 2026-10-23. Scope: READ-ONLY code study (serve/server.py 8229 lines in full; the engine's serving
internals in src/program/generate.cpp) + the measured record (campaign/results/t34-elastic.log,
metrics.jsonl, F-266..F-273) + external research mapped to OUR structures. No boots, no builds, no
commits. Builds on and does NOT re-tread: T27-3-serving-ops-stability.md (VTC/TWP/decode-share/prewarm2
verdicts stand), T33-priorart-serving.md + T33-serving-delta-notes.md (Llumnix policy layer), 
T33-dynamic-session-allocation.md v2/v3 (the F-272/273 substrate), T26-4 (dma/PIPED), T26-6 (cold start).
SIBLING FILE, same T35 slot, disjoint scope: T35-elastic-refinement.md (the elastic lifecycle's internal
anatomy from the live log — promotes/demotes/chunks/pins). Where the two touch (stage 2, lever T35-3,
top-3 #3), this file cites it as primary and does not re-derive it.

---

## PART 1 — OUR DESIGN, AS A REQUEST'S PATHWAY (code read 2026-10-23, working tree)

Standing recipe: launch-custom-iq3s-8081.sh + custom-iq3_s-8081.json (`--batch` via `parallel 6`,
`--spec 6`, `--prefill auto:32768`, `--vram-reserve-mib 2000`, `--conversation-cache-mib 49152`,
`--kv int8 --kv-resident 32768`); launcher env: STRATA_COHORTS=2, STRATA_ELASTIC_SESSIONS=1,
STRATA_BATCH_MTP=1, STRATA_TYPICAL_ACCEPT=0.2, decode share default 0.5 (F-263 killed share=1.0).
Scoreboard at this recipe (T34, 3v3 interleaved, prewarmed, single-engine every leg): M1 224.5 /
pair 214.4 / 6-conc 338.8 (elastic means, F-273); 9-conc 322.5 at M1 209.3 (F4, solo in-band).

### Stage 0 — HTTP admission (`Service.reserve_inference`, server.py:3621-3635)
An in-process counter bounds pending inference calls before generation or SSE headers. Cost:
microseconds; unbounded queueing debt is taken HERE by design (see stage 1). Gauges exist for the
outside world (`vllm:num_requests_running/waiting`, prometheus.py:77-79; live `queued/waiting/
admitting`, server.py:4255-4257) and are scraped by probe-metrics.py.

### Stage 1 — Control lines = the admission point (`_take_control`, server.py:1647-1717)
One prompt read at a time, one engine. Default order: strict FIFO arrival tickets (F-14 law, no
wake-and-race). Opt-in `STRATA_ADMIT_SHORT_FIRST=<tokens>` re-orders NEW admissions only
(`_admit_next`, server.py:1719-1729): a prompt at or under the threshold goes first; a longer one that
waited `STRATA_ADMIT_MAX_WAIT_S` (default 30 s) counts as short, so nothing starves. **Status: OFF in
the launcher** (dormant since merge; T27-3 CONFIRMED its value as the trivial member of the VTC
family). A request whose read gave way (BYIELD) re-enters with `after_epoch`, letting later arrivals
go first — the only fairness inversion in the code. Waiting has NO default deadline
(`QueueWaitExpired` fires only when the client sets `_strata_queue_deadline`). This stage is the only
unbounded wait on the critical path, and at 6 fixed probe clients it is ~empty — which is exactly why
the admission-order levers (short-first, VTC-lite, pacing) are 7+-client levers.

### Stage 2 — Materialize (elastic sessions, F-272/273; refined by the same-day companion file
### campaign/research/T35-elastic-refinement.md, whose live-log anatomy supersedes parts of this paragraph)
`BGEN <slot>` on a demoted slot starts `es_promote` on a background thread when the spare chunk pool
holds a demote's chunks (map+zero cannot disturb a running window; generate.cpp:12616-12620); the
admission joins it at its own quiesce point (generate.cpp:16066-16073). Empty pool: the cache
eviction half runs on the request thread at the admission quiesce (`kvg_release_until`, cache floor
respected). MEASURED (t34-elastic.log, F-272): **promote 11-19 ms** (map 12-17 + zero 0.5-0.6; log:
12.2 ms for 373 chunks / 0.73 GiB), **demote 23.5-28.1 ms** (unmap 4.4-7.8; idle scan only, never
inside a window, never mid-conversation, slots < S_min=2 never demote). The companion's live-log
study (59 promotes / 48 demotes in custom-iq3_s.log) CORRECTS two terms: the background path NEVER
fires in practice (the spare pool was empty at 59/59 promotes — every promote ran synchronously on
the request thread), and the idle scan is line-arrival-driven, so the demote work bills the post-idle
ARRIVAL 128 ms (4 slots, measured) to ~200-310 ms (7-slot worst). `give_way` refuses unmapped slots
(server reserved-slot path can never pick one). THE DEVIATION THAT COSTS: demote does
`bs[b] = BSlot{}` — the slot's session KV dies with its arena; a returning conversation RE-READS its
prompt — MEASURED by the companion at **1,361 ms re-read for the returning chat** (routeable to
~60 ms via the existing ConversationCache; their lever 5).

### Stage 3 — Prompt read (chunked, cooperative, decode-share interleaved)
The admission (BGEN) reads the prompt chunk by chunk (`--prefill auto:32768`, PP lines per chunk);
between chunks the live slots decode for `STRATA_BATCH_DECODE_SHARE=0.5` × the chunk's time
(generate.cpp:14020-14026, 14402-14511). A long read yields cooperatively: `BYIELD <slot>` at a chunk
boundary when a waiter has a prompt under half this one's (`_shorter_waiting`, YIELDS_MAX=2,
server.py:1744-1747, 1806-1838; engine 13710+). The read borrows the conversation cache/expert cache;
borrowed-marked experts fall to CPU for the decoding windows (the window "sees the cache without the
slots this prompt borrowed", generate.cpp:14018-14019).

### Stage 4 — First token
The admission itself produces token 1 (`GEN 1` semantics inside BGEN) and lands the slot with
`BADM <slot> <1>`; the server's live tracker stamps `first_token` here. TTFT (probe histogram,
per finished request) = stages 0-4. **MEASURED (T34, 6-conc legs): TTFT p50 306-344 ms, p99
2.02-2.42 s; 9-conc: p50 388 ms / p99 2.39 s.** Anatomy: the probe mix carries one 5,865-token
prompt; at the measured prefill (~3,300 tok/s class) that single read IS ~1.8 s — the p99 is
prompt-SIZE-dominated, not queue-dominated, at 6 fixed clients.

### Stage 5 — Steady windows (the COHORTS reactor)
Slots decode in batch windows: COHORTS=2 splits slots [0,batch/2)/[batch/2,batch) onto two verifiers;
below `batch/2` active slots the second cohort is empty and the round degenerates to one window
(generate.cpp:11809); both cohorts stage/launch/poll in one round — B's pool work hides under A's GPU
window (F-233's escape, +24.8% conc). Windows carry ≤ kVerifyMaxT=8 rows (2 MTP rows/slot); the
suffix drafter + DraftPolicy arbitrate per round on expected committed-tokens/ms (draft_policy.hpp:
MTP EMA vs lookup E(k)/cost, margin 0.03); typical-acceptance rescue exists at both accept sites,
batch-side OFF by quality law (STRATA_BATCH_RESCUE=0), serial-side ON via STRATA_TYPICAL_DEVICE.
The F-233 pl-lane rule (`pl_want = pipe && pl_pw >= 2 && pl_active < 4`, generate.cpp:14845-14848)
keeps the pipelined fast lane preferred under 4 active slots — INERT on this box (pipe not
constructed with --batch; F-272 correction) but semantically armed for elastic multi-GPU splits.
The adaptive expert tier swaps under windows (GPU-reach wait healthy 0.62 ms/window; the F-271
poisoned-profile class showed 6-10 ms/window = the ms/window degradation signal).
**MEASURED: ms/window 7-22 ms per request row, TREND early 11.1-11.3 → late 9.1-9.3 (stable); ITL
p50 4.3-4.8 ms (F-266 anchor); ITL p99 20 ms at 6-conc, 35 ms at 9-conc (t34).** CRITICAL
MEASUREMENT CAVEAT found in code: the ITL histogram observes the request's MEAN gap
`decode_ms/(n-1)` repeated (n-1) times (server.py:5143-5145, prometheus.py Latencies.observe) — the
p99 of PER-TOKEN gaps is invisible; a request with one 500 ms stall over 100 tokens reports ~5 ms.
The 20→35 ms ITL-tail readings are per-request-mean tails. The true per-token tail is unmeasured
today (see lever T35-2).

### Stage 6 — Tails and the restart class
E2E p50 2.7 s / p99 19.6 s at 9-conc (b9ela metrics record). The bad tail is not scheduling, it is
the restart class: an engine death mid-request costs minutes of boot + the first request on the new
engine pays the lazy-capture tax (F-271: ~71 t/s band + TTFT p99 8.4 s — measured under a
death/restart cycle; the numbers are the cycle's anatomy). Guards: silence/stall watchdogs
(ENGINE_SILENCE_S + ENGINE_STALL_S=90 s CPU/IO-frozen check, server.py:302-324), EngineDied → clean
503 with auto-restart (restart(), 3 tries × fixed RESTART_RETRY_S=15 s — NO backoff, NO storm
breaker), orphan-slot reaper (3 s, #1603), slot parking on BGEN refusal (2 s, #1994),
EngineNonFinite state drop (#879). Cold-vs-warm: conc 272.0 cold-boot vs 299.5 warm same-boot (F-237,
~+10%); prewarm2 ADOPTED as standing posture (F-266: M1 +4.5% 3v5, pair +1.7%, conc -1.0% noise,
cost 15-20 s/boot). Golden-hash boot variance: 3 hashes lifetime (a974 / dbb2 / 3b68-on-b9ela) —
the known batch-geometry variance signal, monitored, no action (consistent with the banked
batch-composition determinism analysis).

### Stage 7 — Hardware floor under it all
GPU locked at SM 3135 / mem 14001 (strata-gpu-lock; F-188/204: the lock holds under probes, DVFS
exonerated as a variance source — dmon zero dips through a full battery). VRAM floor: reserve 2000
MiB in config; engine-side #2065 detection = per-request free-VRAM sample with a LOW warning below
kVramSafeMib (512 MiB Linux / 1024 Windows) printing the exact `--vram-reserve-mib` suggestion
(generate.cpp:12641-12656); the WDDM auto-floor 2560 is Windows-HIP-only; the #496 small-card
shrink applies only when the reserve was NOT given. Host: 9950X3D, desktop co-resident (every .out
carries the RLIMIT_RTTIME/rtkit warning); engine has `--aux-cpus` placement for non-pool threads and
the CPU-expert pool; server.py threads are unpinned.

---

## PART 2 — RANKED NEXT-TIER LEVERS

| # | mechanism | stage / tail it improves | expected effect, with arithmetic | risk | cheapest falsifier | code? |
|---|-----------|--------------------------|----------------------------------|------|---------------------|-------|
| T35-1 | Latency-path host isolation: taskset/cpuset pin of server.py + engine aux threads away from desktop cores; nice -10 on the serving pair; GPU/NVMe IRQ affinity to non-serving cores | ITL tail variance, ms/window variance, TTFT jitter (stage 5); throughput under churn | eunomia (2026-05, eBPF): CPU churn cost 8.8-20.5% decode throughput, median 15.3 ms preemption penalty on affected launch pairs; taskset+nice recovered +7.6% (of 8.8). Our box: desktop co-resident, launch thread preemption lands 1:1 in window ms and token gaps | near-zero (revert = unpin) | one 3v3 pinned-vs-standing under identical desktop activity; read ITL p99 + ms/window SPREAD (TREND), not means | zero-code (ops) |
| T35-2 | Per-hop timing upgrade: engine already times every window (bt_run/bt_commit/bt_emit, printed at idle) — emit a per-window ms histogram + ms/window p99 (not the early/late MEAN), per-request admission-wait, restart/incident counters into /metrics | makes the 20→35 ms ITL tail attributable (window vs adaptive-wait vs cohort depth vs capture); prices admission share of wall; detects capture tax | not a direct ms win — the finder of the next ms. Arithmetic for the gap it closes: ITL p99 is currently a per-request-MEAN artifact; true per-token p99 could sit anywhere in 20-500 ms band and no instrument would know | zero scoreboard risk (probe/server-side) | run probe-metrics against one standing boot with the new fields; confirm histogram non-empty, then read one 9-conc leg | S (server.py + probe; engine print exists) |
| T35-3 | Demote parks KV instead of dropping it: route the returning conversation through the EXISTING ConversationCache (the companion file's lever 5) rather than novel host shadows | stage 2/4 for RETURNING conversations: kills the re-read cliff after an idle demote | MEASURED by the companion (T35-elastic-refinement.md): returning-chat re-read **1,361 ms** today → **~60 ms** restore. Arithmetic holds: 5,865 tok re-read ≈ 1.8 s at probe prefill vs 0.12-0.2 ms/token H2D; 20k-token threads save seconds | M engine work; new restore surface — but the cache substrate exists | DONE by the companion (the falsifier this file asked for): 1,361 ms measured — the lever is LIVE, not speculative | M (engine; rides existing ConversationCache) |
| T35-4 | prewarm2 replay at the ACTIVE width: extend the staged burst 6-way → 9-way when ELASTIC_SESSIONS=1 (script rider) | first-request capture tax at 9-wide (stage 5/6): WIDE 9-row graph captures land at first request (F-271: requests-time capture ≈ 3x window cost band) | first 9-conc leg today pays captures inline; burst-forced capture moves it into the 11-19 s prewarm. Expected: first-leg conc up toward steady 322.5 (the F-271 71 t/s first-request band, minus its death-cycle confound — needs clean measurement) | zero (script) | one 9-conc leg with 9-way prewarm vs without; read first-leg conc + capture log lines | zero-code (script) |
| T35-5 | VTC-lite admission fairness (~40 lines, T27-3 B4, still open) + Equinox/FairInference corroboration | queued-TTFT fairness at 7+ clients (stage 1) | FairInference (2609.18112): VTC-class counters held a fair client's p99 TTFT at 1.2-1.3x isolated while plain sharing hit 82-242x; Equinox (2025) same lineage. At our 6 fixed clients: zero value (no queue) | S-M Python; order-only (no preemption) | 7-client lane, two clients of different token weight; per-client TTFT spread from /metrics | S-M (Python) |
| T35-6 | Restart backoff + storm breaker + single-engine receipt: exponential retry (15→30→60 s) after consecutive start failures; `engines_delta` + watchdog-trip counters in /metrics (the F-271 single-engine law made mechanical) | the restart class (stage 6): death loops (#2065-class VRAM starvation) re-boot every 15 s, each paying full cold boot + first-request capture tax | arithmetic: a 3-try storm at 15 s spacing wastes 3 cold boots ≈ 3× minutes + 3 capture taxes; backoff halves the waste and the counters turn "did the leg have an event?" into a field (T27-3 d8) | S Python; touches the recovery path (test coverage exists: test_restart_waiters, test_fatal_recovery) | unit test + one forced-death drill; no battery needed | S (Python) |
| T35-7 | Adaptive decode share (P-PAS-shaped): keep 0.5 at idle, drop toward 0.25 under prefill pressure (or raise under decode-only), two endpoints switched by a pressure predicate | TTFT under bursts AND steady ITL (stages 3-5) | P-PAS (2608.15171, measured ON RTX 5090, vLLM): pressure-switched token budget beat BOTH static endpoints by 8.5-11.3% avg latency, TPOT +36.3% vs the large static budget. F-263 killed the CONSTANT 1.0 (no gain at 6 fixed clients) — consistent with P-PAS's own finding that constants win nowhere; the gain is the SWITCH under pressure | S engine work (env-pred switch); falsifier first in Python (T35-8) | Python admission gate (hold next admission while ms/window EWMA high) — TWP/B3 shape — on a 7-client burst lane; TTFT p99 + ITL from /metrics | S (engine) after a zero-code pre-test |
| T35-8 | TWP-class admission pacing gate (T27-3 B3, still open) — signals all exist (ms/window lines, queued/waiting/admitting); hold admission while window EWMA exceeds threshold | TTFT p99 + ITL variance at 7+ (stage 1) | TWP (2610.17689): stable TTFT, lower p99, less ITL variation vs conventional schedulers (their benches). Zero engine change | near-zero (Python, env-gated) | same falsifier as T35-7 — RUN THIS BEFORE T35-7 | zero-code (Python) |
| T35-9 | Session watermark headroom (k_empty × 0.78 GiB unclaimed at auto-sizing; T33-dynamic-session item 1, still unimplemented) | stage 2 promote fast path + the F-271 draft-head starvation class | cost ≈ ~380 cache slots at the 97.7%-saturated hit term ≈ negligible (F-254 arithmetic); benefit: every promote after the first takes the spare-pool path (12-17 ms) instead of the cache-eviction path on the request thread | S-M engine; pool-size trade measured by one boot | boot log: spare pool non-empty after first promote storm; cache slot count vs standing | S-M (engine) |

Not ranked (context): ADMIT_SHORT_FIRST arming (banked CONFIRMED, env-only, 7+ lane — it is the
zero-th lever, already priced); elastic-session watermark expressible in remaining-decode-iteration
units (Llumnix /B insight, T33-delta-notes D1 — adopt when the watermark above is built).

---

## PART 3 — THE TOP 3, ARGUED

### 1. T35-1 — Host isolation of the latency path (zero-code; the smoothness lever)
Why first: every tail number we bank is measured on a box where the desktop shares the CPU with the
serving pair, and our own .outs carry the rtkit/RLIMIT_RTTIME warning every boot. The eunomia eBPF
study measured the exact mechanism on our shape of system (CUDA launch thread preempted by CFS
slicing + IRQ work): median 15.3 ms penalty per affected launch pair, 8.8-20.5% throughput loss
under churn, and taskset+nice recovered most of the CPU-load case (+7.6% of 8.8%). Our window is
~10 ms; a 15 ms launch-thread preemption IS a doubled window — it lands directly in the ms/window
trend and the true (unmeasured) per-token tail. The machinery partially exists (`--aux-cpus` for
engine non-pool threads; cpu_affinity.py partitions instances), but nothing pins server.py or
steers IRQs today. This is AGENTS.md pathway law in its purest form: the stage that waits is the
GPU waiting on a descheduled host thread; the fix REMOVES the wait rather than adding a mechanism.
Expected, honestly: variance reduction, not a mean shift — the falsifier must read spreads (ITL
p99, ms/window p99 from T35-2, TREND instability), and is one 3v3.

### 2. T35-2 — Per-hop timing upgrade (S code; the "next ms" finder)
Why: the assignment asks what telemetry would expose the next ms; the answer is that most of it is
ALREADY TIMED and thrown away. The engine times every window (bt_run/bt_commit/bt_emit, plus
adaptive-wait and pool terms) and prints a summary only when slots go idle; the server's ITL
histogram records per-request MEAN gaps, so the p99 ITL we bank is a mean-of-gaps artifact — the
real per-token tail (the thing "smoothness" means to a client) has no instrument. SM-occupancy
profiling (CUPTI class) is NOT the cheap path: the engine's own window timers + two CUDA event
pairs per phase give the same attribution at zero device risk. Deliverables: per-window ms
histogram at /metrics, ms/window p99 in the TREND line (T27-3 d7's one-liner, still undone),
admission-wait per request, incident counters (restarts, watchdog trips, BGEN refusals, promotes/
demotes). This converts every other lever's falsifier from "archaeology in .out files" into a
field, and it is the precondition for honest verdicts on T35-7/8 at 9-wide (where ITL p99 moved
20→35 ms and nobody can yet say which term paid).

### 3. T35-3 — Demote parks KV through the existing ConversationCache (M engine; the returning-conversation cliff)
Why: elastic sessions made capacity elastic, but demotion currently destroys the one asset the
session owned — its KV. The designed-for workload (agents with long threads that idle between
turns) is exactly the workload that crosses STRATA_ELASTIC_DEMOTE_S and then returns. This file's
framing asked for the falsifier first; the same-day companion (T35-elastic-refinement.md) RAN it
from the live log: the returning chat's re-read measures **1,361 ms** and routes to **~60 ms** via
the ConversationCache the tree already owns — the largest measured per-event serving cliff on the
board after the restart class, and the only seconds-scale item on either file's list. Pathway law:
the re-read stage exists only because the state was dropped; FOLD: demote = spill to the cache,
promote = restore from it. Companion riders compose: their #1 (timed-wait idle scan) removes the
128-310 ms demote bill the arrival pays today, and their #2 (spare watermark) makes promotes
asynchronous — the three together turn the lifecycle's worst mode (empty spare, synchronous
promotes, arrival-billed demotes) into its designed one.

---

## PART 4 — CONTRADICTIONS WITH BANKED POSITIONS (challenged with primary evidence only)

1. **F-271's mechanism note "server.py's restart path AUTO-RAISES the reserve to 2000" does not
   match today's tree.** Code read: server.py has no reserve auto-raise (reserve flows only from
   the config arg and POST /v1/vram; server.py:2930-2936, 3986-4008, runconfig.py:41). The engine
   KEEPS a user-given reserve as-is (vram_reserve_given; generate.cpp:6165-6174 — the WDDM 2560
   floor is Windows-HIP-only; the #496 shrink applies only when the reserve was NOT given,
   6228-6240). The observed "600→1200/2600" figures are the boot log's reserved line ADDING
   prefill buffers + draft-head bind to the given number (generate.cpp:6187-6194). F-271's
   CONCLUSION stands untouched — a floor exists and reserves below it die at the draft head —
   but the surviving 2000 is the standing CONFIG value, and the engine-side protections are the
   #2065 LOW warning (512 MiB Linux floor) + cache auto-sizing. Why it matters: the "prevention
   formulation beyond the reserve" the assignment asks about already exists as detection
   (#2065, with the exact suggested reserve printed); the actionable residue is T35-6's counters,
   not a new floor.
2. **"Spec decoding does nothing for a busy server" (practitioner folk claim) is contradicted by
   Red Hat's Apr 2026 measurement** (vLLM, GPT-OSS: spec decoding at concurrency 100 improved TTFT
   P95 12.3 s → 10.0 s, -18.7%). This SUPPORTS our banked position (slot MTP drafts stay ON under
   cohorts; MagicDec convergence) and DEMOTES T17B candidate 8 ("MTP-off-under-cohorts") further:
   the priors now say the boundary probe, if ever run, is likelier to find OFF hurts TTFT tail.
   Our own T34 data agrees directionally: 9-conc acceptance 77.98% (highest measured) with ITL
   p99 35 ms — drafts still pay at 9-wide.
3. **The decode-share question re-opens ONLY scoped to pressure (the 4b law applied).** F-263
   falsified the CONSTANT share=1.0 at 6 fixed clients; P-PAS (RTX 5090-measured) shows constants
   lose to a pressure-switched two-endpoint budget on BOTH sides (avg latency +8.5-11.3% over the
   better static endpoint). No contradiction with F-263 — F-263's legs had no pressure to switch
   on — but the banked "share raise staged in t26-a-slate.sh" framing from T27-3 B1 should be
   amended: the constant raise is dead per F-263 + P-PAS's own static-endpoint data; the live form
   is the switch (T35-7), pre-tested zero-code by the pacing gate (T35-8).
4. **ADMIT_SHORT_FIRST's CONFIRMED value: no primary evidence found against it.** The 2025-2026
   admission-order literature (VTC OSDI'24; Equinox 2025; FairInference 2609.18112) converges on
   order-only admission control as the strongest cheap tail lever, and FairInference's numbers
   (VTC 103x vs plain 242x p99-TTFT inflation under a hog tenant) sharpen WHY it pays: it is the
   scheduling half of isolation. Value remains 7+-client-only on this box. No challenge.

---

## PART 5 — HONEST NEGATIVES (searched, priced, dead here — with the condition that re-opens each)

1. **Layered prefill** (arXiv 2510.08055, MLSys 2026: token→layer scheduling axis, MoE chunked
   prefill's redundant expert reloads +39% memory traffic, TTFT -70%): our prompt read's borrow-
   and-mark-missing mechanics (generate.cpp:14018-14019) are the same disease in a different
   organ, but a layer-group read has no object in our window architecture (whole-window verify
   graphs, borrowed slots, kVerifyMaxT accounting). Re-open if the bt_miss/PCIe terms ever price
   the borrow above ~1 ms/window. What transfers today: the PAPER is evidence that our borrow term
   is real and measurable — price it with T35-2's counters first.
2. **Closed-loop pacing at 6 fixed clients** — no queue exists; the controller degenerates to a
   constant (T27-3 e7, reconfirmed). Lives at 7+.
3. **DVFS re-shaping for latency** (GreenLLM 2508.16449, IEEE CAL 2024: SLO-aware DVFS saves
   34-45% energy at SLO cost): energy-optimization direction; our 3135 lock is a latency floor AT
   max boost, F-204 exonerated DVFS as a variance source (zero dips under load). Dead for
   latency; irrelevant to energy goals on this lane.
4. **SM occupancy per phase via CUPTI/profiler**: heavier than the engine's existing window timers
   for the same attribution; profiler attaches perturb the very tails we measure. The cheap form
   (T35-2) precedes any occupancy work.
5. **vLLM-class preemption ladder / retract**: we do not preempt by design (num_preemptions_total
   hardcoded 0, prometheus.py:95-96; preemption exists only as cooperative BYIELD during prompt
   reads). The maya fair-slice PAUSE verb remains the only route to decode-slicing (T27-3 B7,
   charter-gated) — nothing new in 2025-2026 literature supersedes that verdict; FairInference
   reaches isolation WITHOUT preemption (deadlines + cache reservation), which strengthens the
   case that B5/B8-class admission control is the affordable 90%.
6. **InstantInfer/snapshot-restore cold-start class** (2607.18957; Cerebrium CUDA snapshots):
   targets multi-model serverless swap-in; our boot is a dedicated single-model box whose cold
   component prewarm2 already removes. Dead here (same verdict as T27-3 e6, newer sources).
7. **Restart-storm circuit breaking beyond T35-6**: K8s-class orchestration patterns (liveness →
   replace) assume replicas; single-engine law means the only correct degraded mode IS the
   restart. The residue is backoff + counters (T35-6), nothing more.

---

## PART 6 — SOURCES

### Local / code / record (primary for Part 1; all read this session)
- serve/server.py (8229 L): admission 309-324, 1647-1729; slots/parking/orphans 1600-1643;
  batched lane 1749-2150; pick_slot 2107-2150; restart 1249-1305; reserve 2930-2936, 3814-3860,
  3986-4008; metrics observe 5143-5145; gauges at prometheus.py:77-98; cpu_affinity.py (68 L).
- src/program/generate.cpp: COHORTS 11802-12053; elastic lifecycle 5381-5507, 6103-6111,
  11510-11600 (es_promote/es_demote/es_idle_scan), 12616-12620 (bg promote), 16066-16073
  (admission promote); decode share 14020-14026, 14402-14511; F-233 pl gate 14844-14860;
  reserve floors 6160-6280; #2065 LOW 12641-12656, kVramSafeMib 1497-1501; DraftPolicy
  include/strata/spec/draft_policy.hpp; typical rescue 347-420.
- launch-custom-iq3s-8081.sh; custom-iq3_s-8081.json; campaign/probe-metrics.py; serve/prometheus.py.
- campaign/results/t34-elastic.log + t34-{ctl,ela,b9ela}-r*.out; campaign/results/metrics.jsonl
  (T34 records: TTFT/ITL/E2E histograms, acceptance, clocks, golden hashes).
- SVG-CUSTOM.md F-233/237/249/253-254/263/266/271/272/273; AGENTS.md (laws, incl. 4b and
  PATHWAY MINDFULNESS); campaign/research/{T27-3,T33-dynamic-session-allocation,
  T33-priorart-serving,T33-serving-delta-notes,T26-6,GOAL-LINUX-SUPERIORITY};
  **campaign/research/T35-elastic-refinement.md** (same-day companion; its live-log anatomy —
  59/59 synchronous promotes, empty spare, 128-310 ms arrival-billed demotes, 1,361 ms
  returning-chat re-read — is cited as primary evidence in stage 2, lever T35-3, and top-3 #3).

### External (fetched or searched 2026-10-23)
Literature / specs:
1. Layered prefill — "From Tokens to Layers: Redefining Stall-Free Scheduling", arXiv:2510.08055
   (MLSys 2026): MoE chunked prefill reloads experts (+39% memory traffic); TTFT -70%, E2E -41%.
2. P-PAS — "Prefill-Pressure Adaptive Scheduling", arXiv:2608.15171v2: pressure-switched token
   budget, RTX 5090-measured; +8.5-11.3% avg latency over both static MBT endpoints; TPOT +36.3%
   vs MBT 16k; TTFT -18.5% vs large static under pressure.
3. TWP — "Stable Token-level Scheduling for Serverless LLM Inference", arXiv:2610.17689
   (2026-10): closed-loop token-window pacing on ITL/queue signals (banked T27-3 A3; re-anchored).
4. FairInference — "Performance Isolation for Multi-Tenant LLM Serving", arXiv:2609.18112:
   δ-token latency fairness; deadline admission + KV reservation; p99 TTFT 1.2-1.3x isolated vs
   82-242x (VTC/plain) under a hog tenant; single A100.
5. VTC — "Fairness in Serving Large Language Models", OSDI'24 / arXiv:2401.00588 (banked; lineage
   for T35-5). Equinox (2025) and Stanford CS244c enhanced-cost-functions (2026) as the lineage's
   current tail.
6. GreenLLM — arXiv:2508.16449 (SLO-aware DVFS, -34% energy) + IEEE CAL 2024 SLO-aware GPU DVFS
   (-45.5%): the energy-direction evidence behind negative #3.
7. "A First Look at Bugs in LLM Inference Engines", arXiv:2506.09713: engine failure-mode taxonomy
   behind the stability section.
8. InstantInfer — arXiv:2607.18957 (2026-07): FSM snapshot/restore cold start, 7.2x (negative #6).
9. Dean & Barroso, "The Tail at Scale" (CACM 2013) + "Tales of the Tail" (Google, 2014,
   hardware/OS/application tail decomposition): the jitter-decomposition frame for Part 1's tail
   anatomy.
10. "Batch Speculative Decoding Done Right", arXiv:2510.22876 (2026): sync-preserving batch spec —
    background for the spec-under-batch position.
11. LLM serving SLO/latency practice: Spheron SLO guide (2026); NVIDIA NIM metrics docs (ITL/TPOT
    definitions); Anyscale metrics guide.
Practitioner / data:
12. eunomia.dev, "When CPU Noise Slows Down GPU Inference" (2026-05-24, eBPF): the T35-1 numbers
    (15.3 ms median launch-pair penalty; 8.8/2.8/-0.3/20.5% slowdowns; taskset+nice +7.6%
    recovery; isolcpus/nohz_full/IRQ-affinity escalation) — fetched, primary for the lever.
13. Red Hat Developers, "Performance improvements with speculative decoding in vLLM (GPT-OSS)"
    (2026-04-16): TTFT P95 -18.7% at concurrency 100 — contradiction #2's primary source.
14. NVIDIA enterprise IRQ-affinity article (2025-09) + NVIDIA forums NUMA IRQ thread: affinity
    placement practice.
15. vLLM failure archaeology: discuss "V1 engine child process dies unnoticed — check_health is a
    no-op" (2025-06); issues #17676 (startup hang), #35104 (idle worker death, 2026-02): the
    watchdog/restart-storm context behind T35-6.
16. snowflakedb/ArcticInference issue #185 (2025-09): suffix/spec benefit degradation under high
    concurrency — the counterweight kept honest against #13.
17. Session-affinity practice (context for stage 2 / Llumnix continuity): llm-d precise prefix-cache
    routing docs; Ray 2.59 prefix-aware routing docs; vLLM blog "Session-Aware Agentic Routing"
    (2026-06); AWS SageMaker prefix-aware routing (2026-09); Anyscale token-load routing
    (2026-08). All confirm: keep a session's cache where it lands — our pick_slot already is that,
    single-instance.
18. Cold-start practice: Runpod vLLM cold-start anatomy (2026-09); The New Stack compile-cache piece
    (2026-09); Cerebrium CUDA memory snapshots (2026-07) — the 2606.07362 lineage's current state.

---

## METHODS NOTE (so nothing is re-purchased)
Productive: full server.py + engine-window/elastic/reserve code reads (every Part-1 line anchor
opened this session); t34 logs + metrics.jsonl re-parsed (the ITL mean-gap discovery came from
Latencies.observe, not from any doc); F-266..F-273 journal reads; eunomia + P-PAS + FairInference +
layered-prefill full fetches. Mid-write discovery: the INDEX row for T35-elastic-refinement.md
(same-day sibling) — integrated as primary where our scopes touch instead of duplicating its
measurements; its 1,361 ms returning-chat re-read answered this file's planned falsifier for
T35-3. Unproductive / warned-off: "TWC" as an LLM-serving scheduler name —
searched, does not exist in this literature (the assignment's "VTC/TWC lineage" resolves to VTC +
its 2025-26 successors; recorded so nobody searches it again); "latency corridors"/"sync boosts"
as terms — not in the literature, the underlying concepts are Dean-Barroso burst absorption; a
server.py search for the F-271 "auto-raise to 2000" — no such code exists today (contradiction #1);
earlier metrics.jsonl records lack TTFT/ITL histograms (harness predates probe-metrics; only T34+
records carry the percentiles — do not cite ITL tails for t26-t31 legs).

## OPEN QUESTIONS
- What IS our true per-token ITL p99? (T35-2 answers; nothing today measures it.)
- Does the 9-wide ITL p99 35 ms decompose into window cost (rows 8→cap), adaptive-wait, or cohort
  round depth? (T35-2's first customer.)
- The demote transient's exact anatomy ("M1 dip, relearn within one request"): expert-refill-from-
  profile is the F-272-stated mechanism; a per-request window-ms breakdown (T35-2) would settle
  how much is refill vs first-window capture adjacency.
- 9950X3D core topology for the pin (V-cache CCD placement): unmeasured on this box; the falsifier
  can A/B two placements for free.
- ANSWERED by the companion mid-session: returning-chat TTFT after a real idle demote = 1,361 ms
  measured (T35-elastic-refinement.md) — T35-3's falsifier is already banked there.
