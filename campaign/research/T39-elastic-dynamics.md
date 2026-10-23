# T39 — ELASTIC DYNAMICS: churn economics, pathway distance, and where the lifecycle travels unnecessarily

Date: 2026-10-23 (session). Researcher R2, wave T39. HEAD = 3219b765 ("F-278: T38 landed…").
Scope: the elastic lifecycle at full depth — promote/demote/park/restore churn, the watermark's
honest ledger, park/restore pathway distance, the admission→materialize→serve bill, the 9-wide
structural fix, superchunk granularity, and session/slot affinity. READ-ONLY: no boots, no builds,
no git; every internal claim carries a file:line (opened this session at HEAD) or a log line mined
this session from `custom-iq3_s.log` (63,896 lines, 62 boots) and the committed battery logs under
`campaign/results/`. Required reading consumed in full: SVG-CUSTOM-FEATURES.md (§1a lifecycle era),
AGENTS.md (laws 2/3/4/4b/8/9), F-272..F-278, T33-{dynamic-session-allocation, priorart-serving,
priorart-materialization}, T35-elastic-refinement, T37-{scheduling-spec, pathways-memory}. External
sweep: 2025-26 elastic-serving / KV-lifecycle / memory-tier literature and vendor docs (§2, §Sources).

**The user's directive for this wave:** "elasticness could be causing delays, or back and forth, or
travelling in other areas creating unnecessary distance — improve, rewrite, redesign." The headline
answer: **the elastic VMM lifecycle itself is now clean (promote p50 2.3 ms, demote unmap p50
2.4 ms, demotes in true idle) — the unnecessary distance is in the CONVERSATION-CACHE layer that
sits on top of it: a 43 ms whole-snapshot park on the request thread of nearly every request
(1.2 parks/request measured), a four-slot cache that evicts what it just parked, and a restore
that travels main-sessions→slot in two hops when one would do.** The stage to remove is not in
generate.cpp's elastic block; it is in the park/restore path the elastic block calls.

---

## §1 STATE VERIFIED (file:line at HEAD 3219b765) + COUNTERS MINED FROM THE COMMITTED LOGS

### 1.1 The lifecycle state machine, drawn with its measured costs

Anchors all read at HEAD. Costs are the mined distributions of §1.2 (p50 unless noted).

```
                        ┌──────────────────────────────────────────────────────────────┐
  HTTP request arrives  │  serial/batch request path (generate.cpp)                    │
  ────────────────────► │                                                              │
                        │  1. park_current (:9602/:9770, call sites :13565-13590)      │
                        │     fires when !from_live || incoming || slot_source         │
                        │     = whole-snapshot D2H of the OUTGOING conversation        │
                        │     MEASURED p50 42.7 ms (n=1,421; max 137.8) ← ON REQUEST   │
                        │       THREAD, before the prompt read: inside TTFT            │
                        │  2. conversations.best → take incoming (:13499-13509)        │
                        │  3. restore incoming (:13678-13737)                          │
                        │     = snapshot H2D into the MAIN sessions                    │
                        │     MEASURED p50 11.2 ms (n=69; 8.8-14.3) ← inside TTFT      │
                        │  4. prompt read + GEN 1 (TTFT ends here)                     │
                        │  5. admission (:16278-16290): es_bg.join() + es_promote      │
                        │     (:11558-11607) + copy_to_slot (:11856-11896)             │
                        │     promote MEASURED p50 2.3 ms (bg p50 1.1-2.7)             │
                        │     copy_to_slot = per-state checkpoint save/restore with     │
                        │       ≥3 cudaDeviceSynchronize (:11865,:11875,:11879,:11891) │
                        │       — reverse direction measured: gave-back p50 32.6 ms    │
                        │       (n=128; copy_from_slot, same machinery)                │
                        │  6. decode windows …                                         │
                        └──────────────────────────────────────────────────────────────┘
   idle > 10 s (timed scan, wait_for_line :12561-12571; scan :11721-11733)
                        │
                        │  es_demote (:11667-11716):
                        │    a. park_slot_conversation (:11616-11666):
                        │       cudaDeviceSynchronize (:11636) + snapshot D2H from the
                        │       SLOT's sessions  MEASURED p50 43.7 ms (n=149; 22.5-63.0)
                        │       — in TRUE idle (A3 contract holds)
                        │    b. unmap 24 superchunks  MEASURED p50 2.4 ms (0.4-4.5)
                        │    c. watermark (:11692-11700): keep 24 chunks in the spare;
                        │       only the excess goes to kvg_cache_eat_spare (:8108)
                        │    → demote TOTAL p50 59.5 ms, ~93% of it park D2H, all idle
                        │
   BGEN arrival on an unmapped slot (:12828-12833)
                        │  spare ≥ chunks_per → background promote thread (es_bg :11737)
                        │  else → synchronous promote at the admission, allow_release
                        │  MEASURED split: 50 background / 153 admission (n=203)
                        │  admission promote p50 2.3 ms (release 1.3 + map 0.4 + zero 0.5)
                        │  background promote p50 1.1-2.7 ms; ALL promote total
                        │  n=203: min 1.0 / p50 2.3 / p90 7.9 / max 11.7 ms
                        │  … and it runs AFTER GEN 1: the slot lane's TTFT does not
                        │  contain the promote (the copy in step 5 is post-TTFT too).
```

**The conversation cache under it all** (`include/strata/core/conversation_cache.hpp`,
`generate.cpp:799-809`): `conversation_cache_slots = 4` is a hardcoded DEFAULT (generate.cpp:799;
parse :2738; printed at boot "conversation_cache_slots": 4 in the t38-sanity telemetry). Every
parked conversation costs ONE slot whatever its length (:801-803, hpp:240-242); eviction is
oldest-first (`evict_oldest`, hpp:367-377); the byte budget (48,640 MiB configured) is NOT the
binding constraint — the t38-sanity boot parked 615 MB across 4 slots against the 48 GiB budget
while evicting 10 times. The `--conversation-cache-min-tokens` gate (wants(), called from BOTH the
main park :9616 and the elastic park :11626) defaults to 0 = park everything.

### 1.2 Counters mined from the committed logs

Method: `custom-iq3_s.log` split at its 62 "engine started" lines; lifecycle-era boots identified by
the "watermark on" boot line (37 boots, 2026-10-23 14:31→17:04 = the t36 battery legs, the t32b
five-boot legs, the t38 legs, the t38-sanity and t38-final-serve boots). All distributions this
session, ms:

| Event (source line pattern) | n | min | p50 | p90 | max | mean | where it lands |
|---|---|---|---|---|---|---|---|
| promote total ("materialized in") | 203 | 1.0 | 2.3 | 7.9 | 11.7 | 3.3 | admission (post-GEN 1) or background |
| ├ release from cache tail | 203 | 0.0 | 1.3 | 6.9 | 10.8 | 2.1 | synchronous promotes only |
| └ map (24 superchunk ops) | 203 | 0.4 | 0.4 | 1.1 | 7.2 | 0.7 | both modes |
| demote total ("demoted in") | 149 | 25.8 | 59.5 | 71.6 | 82.2 | 55.8 | TRUE idle (timed scan) |
| ├ elastic park D2H | 149 | 22.5 | 43.7 | 51.3 | 63.0 | 42.0 | true idle |
| └ unmap | 149 | 0.4 | 2.4 | 3.4 | 4.5 | 2.2 | true idle |
| main-path park ("parked N tokens", no elastic tag) | 1,421 | 21.5 | 42.7 | 48.3 | 137.8 | 43.8 | **request thread, before the read** |
| elastic park (tagged "(elastic slot") | 153 | 21.4 | 42.7 | 48.1 | 59.6 | 40.3 | true idle |
| restore ("restored N tokens") | 69 | 8.8 | 11.2 | 12.5 | 14.3 | 11.0 | **request thread, before the read** |
| slot gave-back ("gave back N tokens", copy_from_slot) | 128 | 27.2 | 32.6 | 36.4 | 45.1 | 32.6 | request thread, before the read |
| parked tokens per park | 1,574 | 62 | 301 | 2,099 | 7,722 | 631 | — |

Rate structure over the era: **~1.2 main-path parks per request** (40 parks per 33-request probe
leg, stable across 25+ legs: 37-45 parks/leg in every t32b/t38 boot); ~8-11 promotes and ~5-8
demotes per probe leg; restores only 2-4 per leg (the probe's restore count is low because most
re-entries hit the checkpoint-resume path, "reused": 76-of-81-class in the t38-sanity request
records — the 9-15 ms restore line fires only on full parked-image takes). Park size explains
little of the park time (r = 0.21, slope ≈ 2.6 µs/token): the bill is dominated by the FIXED
snapshot bulk (36 GDN recurrence checkpoints + QSA tails + PLE history, ~118 MB per checkpoint
chain per the code's own comment generate.cpp:794-796; observed snapshot_bytes ≈ 240 MB for a
301-token conversation), i.e. ~240 MB D2H at the observed ~5.5 GB/s effective ≈ 43 ms — which is
the measured p50 almost exactly.

**The t38-sanity boot's own elastic + cache counters** (its /metrics JSON, read this session):
promotes 8 / demotes 6 / park_demotes 6 / forget_demotes 0 / last_promote_ms 2.3 (release 1.4 +
map 0.4) / last_demote_ms 39.8 (park 23.4 + unmap 0.4) / spare_chunks 24; conversation cache:
parks 46 / restores 3 / evictions 10 / parked 4 / bytes 615,436,352 / requests 33 /
requests_reused 23 / reused_tokens 1,784. Cohort WSTAT (C5's first live histogram): 363 windows,
p50 23.8 / p99 30.7 / max 42.6 ms.

**9-wide cohort round shapes** (the three parallel-9 boots in the log — t36 b9 class and the two
F-273-era b9 boots; "cohorts: round A.S= B.S=" lines counted this session):

| boot | rounds | (8,8) | (0,8) | (8,4)/(4,8) | other |
|---|---|---|---|---|---|
| @1 (t34 era, batch 9) | 456 | 325 | 0 | 121 | 10 |
| @14326 (b9) | 637 | 340 | 60 | 127 | 110 |
| @27972 (b9ela, T36) | 538 | 339 | 64 | 126 | 9 |

At 9 active clients the dominant round is **(8,8): 8 of 9 slots decode, one A-span slot rotates
out every round** — the collect cap `w.S + (mtp_use?2:1) <= kVerifyMaxT` (generate.cpp:11993) with
span 5×2=10 > 8. The ~60-64 rounds of **(0,8) — cohort A completely idle while B runs 8 rows** —
are the ramp/drain asymmetry the shape cap amplifies. This is the direct, log-level measurement of
the F-273 9c≈6-parity mechanism T37 inferred from code.

### 1.3 Record corrections banked this session (the T35/T37 tradition)

- **C1 — "the sub-millisecond cache-hit pattern" (SVG-CUSTOM-FEATURES.md:49) is wrong for the
  parked path.** A cache-hit turn on the serial lane pays park-out 42.7 + restore-in 11.2 ≈ **54 ms
  before its first token** (t38-sanity request records: first_engine_token_s 0.069-0.075 s for
  76-of-81-reused 220-token shorts; engine line "prompt 81 tokens = 76 reused + 5 read in 68 ms").
  The sub-ms path is the same-session checkpoint resume, a different mechanism. The features doc
  line should read "checkpoint resume sub-ms; parked restore 9-15 ms; plus the park the switch
  pays."
- **C2 — last_demote_ms 39.8 vs the "2.0-2.8 ms demote" claim: both true, different terms.** The
  demote's unmap is 2.4 ms p50 (24 superchunk ops); the demote's TOTAL is park-dominated (park
  43.7 p50 = ~93%). F-276's "demote unmaps from 6-15 ms to 2.0-2.8 ms" was about the unmap term
  and holds; nothing regressed — but the 39.8 ms headline counter is the park, by design in true
  idle. Any future reader of `elastic_last_demote_ms` must read `last_park_ms` beside it.
- **C3 — the timed-scan contract has a small race.** The scan's gates (`in_queue_empty() &&
  !batch_on() && !pipe_inflight()`, generate.cpp:12564) are checked ONCE, before the scan walks
  slots; `es_idle_scan` (:11721-11733) has no per-slot re-check, so an arrival landing inside a
  multi-slot scan (up to ~200 ms at 7 slots × 60 ms worst, park-dominated) waits out the remaining
  scan. Collision odds at ServeGen-class inter-turn gaps (~100 s median-class, §2 E9) ≈
  scan_duration/ITT ≈ 0.2% per arrival — negligible rate, trivial hardening (E7).
- **C4 — the watermark did NOT flip promotes to background, and no longer needs to.** Measured
  split 50 background / 153 at-the-admission (25%): only the FIRST promote of each demote cycle
  finds the spare full; burst promotes 2..n are synchronous by design (the watermark is ONE arena).
  But the T35 arithmetic that made this scary (11.9-21.2 ms synchronous walks) is dead: at 32 MiB
  superchunks the admission promote is **p50 2.3 ms total** (release 1.3 + map 0.4 + zero 0.5).
  The watermark's surviving value is demote-side (single-slot cycles stay unmap-only, no
  cache-eat + ~415 MiB profile refill) — re-priced in E4.
- **C5 — ver_c's memory is measured, not estimated: ~127 MiB window buffers per verifier**
  ("strata verify: window up to 8 tokens, 127.2 MiB of device buffers", boot line this session) +
  graph/exec pools + doorbell buffers — the code comment's "~200-400 MiB" (generate.cpp:9341) is
  the right band for the whole verifier. At ~2 MiB/cache-slot that is 75-200 cache slots ≈
  **0.2-0.6% conc** by the F-273 linear arithmetic (1,300 slots ↔ +4.0%), against a +12.5%
  in-window token gain at 9 clients (E5).
- **C6 — the restore is still two hops.** T35-elastic R5's wiring caveat stands verbatim at HEAD:
  "a restore takes the same path every parked conversation takes (into the main sessions, then
  copy_to_slot into the slot)" (generate.cpp:11610-11612). Returning-to-slot = restore H2D to main
  (11 ms) + copy_to_slot main→slot (32.6 ms p50 by the reverse machinery's measurement; host path,
  ≥3 device syncs :11865/:11875/:11879/:11891). The demote side parks straight from the slot
  (:11633-11646) — asymmetric with the restore side.

---

## §2 EXTERNAL SWEEP — graded (what 2025-26 does with the same problems)

[CORROBORATED] = primary source fetched or search-verified this session. [LIST] = verified to
exist at list level, not fetched. [MEASURED-OURS] = our logs/code. [HYPOTHESIS] = my transfer,
arithmetic shown.

- **E1. KV-cache reuse is heavily skewed; moderate caches capture most of it.** "Characterizing and
  Optimizing KVCache at a Large LLM Provider" (arXiv 2506.02634, USENIX ATC'25) [CORROBORATED,
  abstract fetched]: reuse is concentrated in a small request subset; single-turn reuse matters as
  much as multi-turn; per-category reuse is predictable; the footprint needed for near-ideal hit
  ratio is "moderate." Maps to us: our 4-slot/615-MB-used cache evicts 10×/boot while holding
  1.3% of its 48 GiB budget — we are on the wrong side of their "moderate footprint" finding by a
  sizing DEFAULT, not by bytes (E1).
- **E2. Retention must be a hold, not a hint; LRU under churn is the enemy class.** vLLM RFC
  #37003, context-aware KV-cache retention API (opened 2026-03, open; working implementation,
  PR #38514 referenced) [CORROBORATED, fetched]: tool-call pauses leave agent KV unreferenced and
  LRU evicts it; the proposal is priority + TTL directives; the discussion's production findings —
  "retention must be a hold, not a hint (SLRU captured nothing)"; a 15-40 s gap/residency boundary
  beyond which LRU stops protecting live sessions; hybrid-model (mamba-checkpoint) retention
  concerns. We are a hybrid-GDN engine whose parked snapshots ARE recurrence checkpoints — their
  caveat is our exact object. Our oldest-first 4-slot eviction under a 10 s demote hysteresis is
  the un-annotated-LRU case their data shows failing (E1, E6).
- **E3. Workflow-aware eviction beats LRU ~2.19× in agent interleaving.** KVFlow (arXiv 2507.07400,
  NeurIPS'25) [CORROBORATED, list+abstract]: evict by steps-to-execution, not recency. The
  conversation-cache analog (evict the conversation whose next turn is furthest away) is E6's
  upgrade path; not needed at 4→32 slots, but it is the field's direction.
- **E4. Inter-turn gaps are ~100 s-class with a long tail.** ServeGen (NSDI'26, arXiv 2505.09999)
  [CORROBORATED, fetched]: ITTs "concentrate around 100 seconds, with an extremely long tail";
  ~10% of a reasoning workload is multi-turn, ~3.5 turns/conversation. UniCache (SIGMETRICS'26)
  [LIST]: inter-turn intervals fit log-normal; chat ≫ agentic gaps. Consequences: (a) our
  STRATA_ELASTIC_DEMOTE_S=10 s is ~10× below the typical gap — in real chat serving the demote
  (and its idle park) fires between nearly every turn, which is FINE (idle) but means the
  return-side bill (promote + restore + copy ≈ 24-50 ms) is the recurring per-turn cost to
  optimize; (b) hysteresis is adequate — flap risk (demote→promote inside a conversation) needs
  gaps in the 10-30 s band, the tail of the distribution, not its body.
- **E5. Migrate-vs-recompute is a measured policy axis, and the field's boundary is profiled, not
  assumed.** Mell (arXiv 2501.06709) [CORROBORATED, fetched]: per-request choice between KV
  transfer and token-transfer+re-prefill with offline-profiled boundaries; batched operations cut
  migrations ~25-30%. vLLM PagedAttention's recompute-vs-swap rule (carried T33) said recompute ≤
  20% of swap latency in the small regime. Our arithmetic (below) puts the forget/park break-even
  at ~105-215 tokens — our `--conversation-cache-min-tokens` exists and defaults to 0, i.e. we
  park conversations re-reading would serve faster (E6).
- **E6. The lifecycle's moving parts are now the field's consensus shape.** vLLM sleep mode
  (docs, re-verified): cuMem pool, graphs "replayed, not recaptured" — our stable-VA slots already
  do this by construction. vLLM RFC #34303 (2026-02) [LIST]: sleep/wake graph re-capture ~2-10 s
  is their pain; ours is zero. SGLang Unified Radix Cache (LMSYS 2026-08) + HiCache layering
  [LIST]: GPU/host/disk tiering with radix retention; LMCache (arXiv 2510.09665 tech report)
  [LIST]: cache lifetime decoupled from engine lifetime. Mooncake store: hierarchical eviction;
  without the SSD tier their hit rate collapsed 83%→36% (kvcache.ai engineering blog) [LIST] —
  the "cache too small for the working set" failure our 4-slot default re-enacts at micro scale.
  Nothing in 2025-26 contradicts the parked-restore design; everything says give it enough slots
  and retention signals.
- **E7. No batched VMM map through CUDA 13.4** (carried T37 §2, release notes scanned there);
  per-op GSP cost 29-51 µs is ours (F-276). Op-count stays the only lever; 24 ops/arena at 32 MiB
  is why the map term is now 0.4 ms p50 [MEASURED-OURS]. The 64 MiB question is priced in §4 N1.

---

## §3 IMPLEMENTER-READY CANDIDATES (value-ranked; band honesty vs the pooled records M1 215.2 / pair 206.4 / conc 323.0)

Convention: every candidate names the pathway it shortens and the measured term it removes
(law 9), prices the cheapest falsifier first (law 4), carries law-4b scoping and a one-line
rollback. Pooled records are DECODE-lane numbers; several candidates below are TTFT/interactive-
lane wins that will show in the probe's TTFT p50 (325-375 ms class) and in `prompt_ms`, not in
M1/pair/conc means — stated per candidate, because an honest "no scoreboard movement" is part of
the price.

### E1 (rank 1, zero code). CONVERSATION-CACHE SLOTS 4 → 24-32 — un-starve the cache the lifecycle feeds

- **Mechanism.** `--conversation-cache-slots 24` (flag exists; default 4 at generate.cpp:799).
  The lifecycle parks faithfully (1.2 parks/request, 42.7 ms each) and then the cache throws the
  park away at the fifth distinct conversation: 10 evictions per boot at 4 slots while using 615 MB
  of a 48,640 MiB budget. Each evicted-then-returning conversation pays the full re-read
  (232 µs/token measured: 5,865 tokens = 1,361 ms) that the park was paid to prevent. Slots 4→24
  fits ~24 conversations ≈ 3.7 GB of the budget (at the measured ~154 MB/conversation) and moves
  the binding constraint from count to bytes, which is where the field says it belongs (§2 E1/E2).
- **Expected band.** Battery lanes (M1/pair/conc): ~0 — the probe interleaves only 4-5 distinct
  conversations. The win is the eviction class: evictions/boot 10 → ~0, and every returning
  conversation beyond the 4 most recent stops paying the 100-1,400 ms re-read. On a real
  multi-chat workload this is the difference between the park system existing and working. WTF-
  class on the interactive tail, invisible on decode means — the falsifier reads the telemetry,
  not the probe lanes.
- **Cost anatomy (F-232).** Slot cost = host RAM per parked conversation (measured: 4 conversations
  = 615 MB → ~154 MB each, snapshot bulk = checkpoints + recurrence states, generate.cpp:794-796;
  conversation_snapshot_bytes path). 24 slots ≈ 3.7 GB pinned-class host RAM; box has ~120 GB free
  of 192 (t38-sanity hardware record: ram_used 77.2/197.7 GB). make_room/evict_oldest paths
  unchanged (hpp:346-377); `conversation_cache_min_free_mib` 2560 guard unchanged.
- **Falsifier-first.** One boot with the flag at 24, t38-sanity pattern, read the telemetry JSON:
  evictions must go 10 → 0 with parks unchanged and restores unchanged; RAM delta ≈ +3 GB.
  Cheaper than any battery; if evictions do not fall, the eviction pressure is byte-side, not
  count-side, and the flag dies free.
- **Law-4b scoping.** The flag is per-boot configuration; a revert is the launcher line. No lane
  sees new code paths. Watch: pinned-RAM pressure against the 1.55 GiB flat pins + 36.8 GiB
  complement on low-RAM boots — the min_free_mib guard is the protection; keep it.
- **Rollback.** Remove the flag (default 4).

### E2 (rank 2, S-M code). THE REQUEST-START PARK: stash D2D, drain D2H in the background — the 43 ms leaves TTFT

- **Mechanism (law 9: MOVE the copy under an existing wait).** Today park_current (:9602-9770,
  call sites :13565-13590) runs its ~42.7 ms whole-snapshot D2H **synchronously on the request
  thread before the prompt read**, on every switch/rewind admission (measured 1.2 parks/request;
  the probe's 68 ms cache-hit TTFT is ~54 ms lifecycle). The park's source (main-session device
  state) is about to be overwritten by the incoming restore, which forces the serialization — but
  only against the bytes, not the wall: (1) D2D-stash the outgoing snapshot into a scratch
  session/arena (240 MB D2D at ~8 TB/s ≈ 0.03-0.06 ms + launch), (2) run the incoming restore
  immediately, (3) drain the stash to host on a side thread UNDER the request's own prompt
  read/decode (68-919 ms of cover measured on this workload), with the next park joining any
  still-in-flight drain. The park leaves TTFT; its wall cost is unchanged; its correctness contract
  (the fork-preservation the comment at :13545-13550 names) is unchanged — the stash holds the
  branch until the host image exists.
- **Expected band.** Cache-hit/switch TTFT: −40 ms of a measured 68-75 ms serial short-request
  TTFT (t38-sanity first_engine_token_s) → ~28-35 ms; probe TTFT p50 325-375 → ~285-335 (−10-13%;
  the conc-phase queueing share does not move). Multi-turn chat at ITT ~100 s: every turn's slot
  was demoted, so every turn pays park-on-return-path today via the demote-park's sibling — this
  candidate moves only the REQUEST-START park; the demote park is already idle-side (correctly
  placed). Honest scoreboard expectation: M1/pair/conc ≈ 0 (decode-dominated); TTFT p50 −10% class
  is the targeted metric and it is WTF-visible on interactive turns.
- **Cost anatomy.** Park path: park_current_body → drop_superseded (:9624) → snapshot_bytes →
  conversation_snapshot_save D2H (the 42.7 ms; bulk = checkpoints/recurrence, §1.2) — all
  synchronous, watchdog-named "parking" (:9608-9611). Scratch cost: one ~240 MB device buffer
  (≈100 cache slots ≈ 0.3% conc by F-273 arithmetic; or reuse a pinned-host shadow at H2D speed —
  slower, still hidden). New concurrency surface: ONE side thread + a join; the es_bg pattern
  (:11737-11745) is the in-tree template, including the destructor-join discipline.
- **Falsifier-first.** (a) Zero-code read first: from the existing t38-sanity request records,
  TTFT of requests whose predecessor parked vs not (parked_at in the cache record) — if the
  ΔTTFT is not ~40 ms, the park is not on the path and this dies free. (b) Then the arm behind
  `STRATA_PARK_ASYNC=1`, one 3v3, TTFT p50 + golden hash + M3b (bit-class: the park is a pure
  copy, but the reordering touches state lifetimes — the F-233 transition-law checklist applies;
  the stash must be fenced against the restore's overwrite, which the D2D-then-restore stream
  order gives).
- **Law-4b scoping.** Scoped by the same gate the park already has: only requests that park pay
  any change; continuing requests (from_live, no incoming) are byte-identical. The deeper REMOVE —
  delta/append-only parks (write only cells past the last checkpoint; Llumnix's append-only
  argument, T33-priorart-serving) — is the structural successor but needs snapshot-format work;
  the reuse machinery (`take_reuse`/`retain`, generate.cpp:9642/:13729, hpp:256-270) already
  proves the pattern for the KV share, which is however the SMALL share (240 MB bulk vs ~1 MB KV
  at 300 tokens) — do not expect reuse to shrink parks without extending it to the checkpoint
  bulk.
- **Rollback.** `STRATA_PARK_ASYNC=0` (default off) — or one cp of engine/strata.pre-<label>.

### E3 (rank 3, S code). RESTORE STRAIGHT INTO THE PROMOTED SLOT — finish T35 R5's wiring caveat

- **Mechanism (law 9: REMOVE a hop).** A returning conversation to an elastic slot today: snapshot
  H2D → MAIN sessions (11.2 ms) then copy_to_slot main→slot (reverse machinery measured 32.6 ms
  p50; host path with ≥3 device syncs, :11856-11896). The demote side already reads straight out
  of the slot (:11633-11646) — the restore side never got the symmetric form T35 specified
  ("restore directly into the promoted slot (one H2D)"; the code comment :11610-11612 admits the
  two-hop shape). When the admission's target slot is already promoted (the common return: the
  timed scan demoted it, the BGEN background-promoted it, :12828-12833), the snapshot can restore
  into `*bslot_ss[0][slot]` directly and copy_to_slot is skipped entirely.
- **Expected band.** Returning-chat turn-2+ latency: −(11 + 12..33) ms − syncs ≈ **−25-45 ms per
  return**; combined with E2 the interactive return goes from ~68 ms to ~10-20 ms ≈ 3.5-6× on
  that class. Scoreboard lanes ≈ 0 (same as E2).
- **Cost anatomy.** conversation_snapshot_restore writes into a SessionState& (:13680); the slot's
  SessionState is a legal target by construction (copy_to_slot's `to` is exactly that object,
  :11863). The drafter KV (:13681) and checkpoint chain (:13716-13721) have slot-side twins
  (slot_mtp[b], bs[b].checks). The non-batch path (slot lane not engaged) keeps today's form — the
  gate is the same `es.on && slot live` test the admission already computes (:16278-16286).
- **Falsifier-first.** One boot, two-turn chat probe with a >10 s gap between turns (the T35
  "two-turn chat probe" falsifier, still never run): read the restore line + admission timing +
  golden hash. If the restore-into-slot path fails validation for any slot-affine state (the
  failure mode the fatal-restore handler :13683-13686 guards), the boot shows it as an ERR — kill,
  no battery spent.
- **Law-4b scoping.** Only conversations returning to a live promoted slot take the new path; all
  other restores byte-identical. Watch: the park-then-restore of the SAME conversation inside one
  request cycle (incoming while the slot holds it) must prefer the slot's live copy (the
  slot_source path :13592-13643 already wins where it applies — verify the precedence test order
  in the implementation).
- **Rollback.** `STRATA_RESTORE_TO_SLOT=0` (default off).

### E4 (rank 4, zero code). THE WATERMARK RE-PRICE — one boot decides whether 0.75 GiB still earns its keep

- **Mechanism.** The watermark's promote-side value died with superchunks: the synchronous
  admission promote it was built to avoid is now p50 2.3 ms (release 1.3 + map 0.4), against the
  11.9-21.2 ms T35 priced pre-A4. Its surviving value is demote-side: single-slot cycles skip the
  cache-eat + ~415 MiB profile-refill (the "ate them back and refilled" leg of the demote lines,
  §1 t36 data: it fires on the SECOND demote of a burst, ~+11-19 ms class in idle). Its cost is
  permanent: 24 chunks = 0.75 GiB ≈ 320-375 cache slots ≈ ~1% conc upper bound (T35's own
  arithmetic), plus the burst promotes 2..n are synchronous anyway (25% background share,
  §1.2 — the watermark was never going to cover them at depth 1).
- **Expected band if dropped.** conc: +0-1% (the slots return to the pool); demote tail: +11-19 ms
  in TRUE IDLE on burst cycles only (no request sees it); promote: +1-2 ms at the admission
  (hidden under GEN 1 for prompts ≥ ~20 ms of read — all of them). Honest expectation: inside
  noise on decode lanes; the question is whether the conc giveback is real at the pooled record
  (323.0) — F-276's own note ("the conc-vs-anchor gap gets the watermark-slot-cost tunables if the
  pooled read confirms it") scheduled exactly this question.
- **Cost anatomy.** Boot pre-fill + demote excess logic: generate.cpp:6106-6115 (keep at boot),
  :11689-11704 (eat only the excess), promote's `had >= chunks_per` source test :11567-11574,
  background gate :12828-12833. All already parameterized by `es.watermark` — the off path is the
  pre-T36 code that t34/t36's own ctl legs exercised (the ctl legs in t36-lifestic.log ran
  watermark=0 with 373-chunk cycles, stable).
- **Falsifier-first (the whole candidate).** One 3v3 interleave, arm = standing minus
  STRATA_ELASTIC_WATERMARK (superchunks/park/timed-scan stay ON), read pooled conc + M1 + the
  demote/park lines. Decision rule written in advance: conc pooled delta ≥ +1% with demotes still
  idle-side → drop the watermark; conc inside noise → keep it (the demote-side churn saving is
  then its price, and 0.75 GiB is cheap for a calmer lifecycle). Either way the record gains the
  honest ledger line the features doc currently lacks.
- **Law-4b scoping.** If kept, the honest doc line is "0.75 GiB held for demote-side churn
  avoidance + first-promote backgrounding, promote-side value retired by A4." If dropped, rollback
  is the launcher line only.
- **Rollback.** Re-add `STRATA_ELASTIC_WATERMARK=1` (or remove it — each is one line).

### E5 (rank 5, M code). THE TERNARY REACTOR AT 9-WIDE — extend T37 C3 with this wave's measurements

T37 owns the design (C3: 3+3+3, every slot every round; scoped `o.batch >= 9`). This wave adds the
three facts it lacked:

- **The rotation is now log-measured, not inferred:** at 9 clients the round shape is (8,8) in
  63-76% of rounds and (0,8) — cohort A fully idle — in another ~10-12% (§1.2 table; three batch-9
  boots, 1,631 rounds). Per (8,8) round the engine commits 8 × 1.77 ≈ 14.2 tokens where 9 × 1.77 ≈
  15.9 exist: **+12.5% in-window at identical wall** (6-row windows ≈ 12-13 ms vs 8-row ≈ 13-14 ms,
  c_row ≈ 2.4 ms F-270 + the 13.11 ms/4.99-row boot line) — T37's 355-390 band for 9-conc stands,
  vs the pooled 323.0 record.
- **ver_c's memory is bounded by measurement:** 127.2 MiB window buffers per verifier (boot line)
  + graphs/doorbells → 150-250 MiB class ≈ 75-125 cache slots ≈ 0.2-0.4% conc giveback. The
  init-failure fallback (prints and continues, :9352-class) makes the gate safe.
- **The fence question multiplies by one:** the standing→cohort fence (:11971-11979) is per-ramp,
  not per-round; a third cohort adds no new fence class but adds a third stage/go/poll lane in the
  round (:12043-12058 pattern) and a third finish tail — T37's "mechanical" read confirmed at HEAD.
- **Band vs records:** 9-conc 355-390 [HYPOTHESIS, T37 arithmetic + this shape measurement]; the
  precondition is banking the F-249 void-verdict retraction (T37 §headline 1) and the 9-wide C5
  telemetry leg (still owed, F-278) to read the A/B window asymmetry before the port.
- **Rollback.** Do not export STRATA_COHORTS=3; the launcher gate scopes it to batch ≥ 9 boots.

### E6 (rank 6, zero code). FORGET-VS-PARK THRESHOLD — stop parking what re-reading serves faster

- **Mechanism.** `--conversation-cache-min-tokens 200` (flag exists, default 0; gates BOTH parks
  through wants(), :9616 and :11626). Arithmetic on measured terms: park-and-return bill ≈
  restore 11.2 + copy_to_slot 12.5-33 ≈ 24-44 ms; re-read bill = 232 µs/token (5,865 tokens =
  1,361 ms) → break-even at **105-190 tokens**. Below it, a parked conversation's return is SLOWER
  than its re-read, and the park consumed a slot + 43 ms of idle D2H to achieve that. The probe's
  84-token shorts are exactly this class (t38-sanity: restored 84 tokens in 9-12 ms against a
  ~19 ms re-read — the park is net-negative there, modulo the copy term).
- **Expected band.** Battery lanes ~0. Real-workload effect: frees slots + idle park I/O for the
  short-conversation mass (Mooncake's >50%-never-reused block finding, carried T33; our probe's
  restore:park ratio 3:46 says the return rate is low). Value is modest and workload-dependent —
  ranked below E1 because E1 removes the eviction pathology for ALL conversation sizes while this
  tunes one boundary.
- **Falsifier-first.** Same boot as E1 (both flags together): restores/evictions/parks counters +
  returning-TTFT. If E1 alone fixes the return path, E6's residual value is just the slot relief —
  decide on the same telemetry.
- **Law-4b scoping.** Conversations below the threshold demote with `bs[b] = BSlot{}` exactly as
  pre-A1 (the forget path is the tested default). Long conversations (the ones where park wins
  25-30×) are untouched.
- **Rollback.** Remove the flag.

### E7 (rank 7, trivial code). IDLE-SCAN HARDENING — re-check the queue between slot demotes

`es_idle_scan` (:11721-11733) walks all live slots once entered; a multi-slot scan at ~60 ms/slot
park can hold an arrival up to ~200-400 ms (7-slot worst). Add `if (!in_queue_empty()) return;`
at the top of the per-slot loop (one line; the partially-scanned slots keep their idle_since and
finish on the next wake). Closes C3's race. Falsifier: code-read + the existing telemetry
(admission_wait histogram, prometheus.py:134-136) — no battery; ride any next adoption's M3b
formality.

---

## §4 NEGATIVES (researched and rejected this session, with re-open conditions)

- **N1. 64 MiB superchunks (12 ops/arena): rejected at current churn.** Saves ~0.2 ms/event on a
  map that is already 0.4 ms p50 in a background/at-admission path, and ~0.2 ms on an idle-side
  unmap; coarsens the watermark to 12 chunks and the cache-tail release to ~12-superchunk
  quantization (eviction quantization doubles, still above the 128-slot floor). Event rate ~14
  lifecycle events/probe-leg ≈ 550/hour battery-class → ~0.1-0.2 s/hour total saving. Re-open
  only if arena-churn rises ~10× (T37 V8's condition, re-affirmed with the event-rate now
  measured).
- **N2. `--conversation-cache-park-gain` as the park-cost fix: rejected as a wash at measured
  turn sizes.** Skipping the park when a turn advanced < N tokens keeps the stale copy; next turn
  re-reads the delta. At 220-token turns: park 43 ms + re-read 5 tokens (~1-5 ms) vs skip + re-read
  225 tokens (~52 ms at 232 µs/token) — a wash or a loss. It pays only for turns > ~190 new
  tokens (the same break-even as E6, mirrored), i.e. it is E6's complement, not an independent
  win. Re-open as the tuning pair (min-tokens + park-gain) if E1's telemetry shows long-turn
  re-park churn (the "dropped 1 superseded copy" line is its counter — 1/boot-classes today,
  2/boot at some legs).
- **N3. Extending the KV-reuse machinery to shrink parks: rejected for now, with the reason that
  kills it.** `take_reuse`/`retain` (hpp:256-270) covers only the KV share — measured ~1 MB of a
  ~240 MB snapshot at 300 tokens (reused_kv_bytes=1,155,840 on the one park where it hit, park
  still 42.7 ms). The bulk is checkpoints + GDN recurrence states, which reuse does not touch. A
  delta-snapshot format for the bulk is E2's deeper successor, not a knob.
- **N4. Llumnix-style tail-flip for the slot copy: not applicable on this path (carried T35 C2,
  re-verified).** Promotes ride copy_to_slot from the main sessions at the admission, after GEN 1
  — there is no old-graph decode to flip from; the copy is post-TTFT already. The field's
  migration machinery (Mell, §2 E5; selective transfer arXiv 2607.28150 [LIST]) addresses
  cross-instance moves we do not do.
- **N5. Predictive pre-mapping beyond the watermark (2+ arenas): still rejected (carried T35).**
  The measured promote is 2.3 ms; pre-mapping two arenas holds 1.5 GiB out of the cache to save
  ~2 ms that GEN 1 already hides.
- **N6. Raising STRATA_ELASTIC_DEMOTE_S as hysteresis tuning: still the wrong lever (carried
  T35).** With ITT ~100 s (§2 E4), demote-before-return is nearly guaranteed either way; a longer
  delay trades M1 (smaller cache during gaps) to save idle-side demotes. The hysteresis is
  adequate; flap risk lives in the 10-30 s tail and costs at most one extra cycle (promote 2.3 ms
  + demote 60 ms idle).
- **N7. Park-compression revival: dead per SplitZip (carried T37 §4 N1).** int8 KV and int8-class
  checkpoint states have no exponent redundancy; 1.01× measured is the bound. The park cost is
  PCIe bytes of states that do not compress losslessly.

---

## OPEN QUESTIONS

1. Does the probe-lane TTFT p50 actually carry the ~54 ms lifecycle share into the reported metric,
   or is it diluted by conc-phase queueing? (E2's zero-code falsifier reads existing request
   records; one battery leg would settle the metric level.)
2. The (0,8) cohort-A-idle rounds at 9-wide (60-64 per boot): ramp/drain asymmetry or a slot-
   assignment bias in how the probe's 9 clients map to spans? (The 9-wide C5 telemetry leg reads
   per-cohort windows; owed before E5.)
3. copy_to_slot's direct path (`STRATA_SLOT_COPY_DIRECT=1`, :11858-11859, default off since before
   T35): still unmeasured on this recipe (T35 Q4 open). E3's implementation should ride the same
   parity/golden-hash boot and price both at once.
4. Whether restore-into-slot must clone the drafter KV separately per slot drafter (slot_mtp[b])
   or whether conversation_kv_restore accepts the slot drafter state directly (code read inside
   E3; the fatal-restore guard is the safety net).
5. Host-RAM ceiling for E1 at real scale: 24 slots ≈ 3.7 GB, but a 131k-context workload parks
   bigger images (7,722 tokens = 472 MB observed, one line, 93.3 ms) — the byte budget becomes
   binding around ~15 huge conversations. Cap-by-bytes exists (make_room); the question is whether
   min-tokens (E6) should scale with context to keep slot count and byte budget aligned.
6. Collision-rate of the idle-scan race (C3) on real traffic — the admission_wait histogram now
   measures it for free; read it once on a long-running boot before deciding E7's priority.

## IMPLICATIONS

- The elastic VMM layer (promote/demote/unmap/superchunks/timed-scan) has no measurable disease
  left at these churn rates: every event is p50 ≤ 2.4 ms of on-path work or lives in true idle.
  Further VMM-side work (64 MiB, deeper watermarks, batched maps) is priced out by measurement.
- The lifecycle's remaining unnecessary distance is conversational, not virtual: the park on the
  request thread (43 ms × 1.2/request), the four-slot eviction thrash, and the two-hop restore.
  E1 (zero code) + E2/E3 (small, gated) turn the interactive-turn bill from ~54-68 ms toward
  ~10-20 ms — the same 25-30× class of improvement the park itself delivered over re-reads, now
  applied to the park's own pathway (law 9 applied to law 9's product).
- The 9-wide structural fix (E5) remains the only candidate that moves a pooled scoreboard record
  (conc 323.0 → 355-390 band); its precondition chain (void-verdict retraction → 9-wide C5 leg →
  port) is unchanged from T37, now with the rotation log-measured.
- Telemetry actions: read `elastic_last_demote_ms` only beside `last_park_ms` (C2); add eviction
  count to the battery alarm list (it is the E1 signal); the admission_wait histogram answers C3's
  rate for free.

## SOURCES

### Internal — code (all read at HEAD 3219b765 this session)
1. src/program/generate.cpp:799-809 (cache slots default, min-tokens, park-gain), :2738 (flag
   parse), :5387-5440 (ElasticSessions struct, watermark/timed-scan/park gates), :6097-6139 (boot
   release + watermark prefill + boot line), :7965/:8082-8183 (kvg spare, release_until,
   cache_eat_spare), :9341-9361 (ver_b init; ~200-400 MiB comment), :9602-9770 (park_current_body:
   drop_superseded :9624, snapshot save, watchdog stage), :11545-11607 (BSlot, es_promote: source
   split :11567-11574, map :11583, zero, print), :11608-11666 (park_slot_conversation: sync
   :11636, snapshot from slot, print), :11667-11716 (es_demote: park→unmap→watermark excess),
   :11721-11733 (es_idle_scan), :11737-11745 (EsBg), :11856-11896 (copy_to_slot: syncs
   :11865/:11875/:11879/:11891, direct path :11858), :11900-11940 (copy_from_slot), :11961-12058
   (cohort collect cap :11993, stage/go both), :12544-12571 (es_next_deadline, wait_for_line, scan
   gate :12564), :12779 (legacy arrival scan), :12828-12833 (BGEN background promote), :13499-13590
   (best/take/incoming, park call sites), :13592-13643 (slot_source gave-back), :13644-13737
   (kvg_ensure, borrow, restore incoming, retain :13729, restore print), :16278-16290 (admission
   join + promote + copy_to_slot).
2. include/strata/core/conversation_cache.hpp:240-246 (slot-per-conversation), :256-270
   (retain/take_reuse), :346-377 (make_room/evict_oldest), :393-397 (drop_superseded #342).
3. serve/prometheus.py:132-136, :205-236 (admission_wait, token_gap, elastic counters),
   :218 (bgen_refusals).
4. launch-custom-iq3_s-8081.sh + custom-iq3_s-8081.json (standing recipe: the F-276 six + dm;
   --conversation-cache-mib 49152, slots default).

### Internal — logs (mined this session)
5. custom-iq3_s.log (63,896 lines, 62 boots): all distributions in §1.2; boot lines (watermark on,
   127.2 MiB window buffers, "STRATA_COHORTS=2 on"); 1,574 park lines / 203 promotes / 149 demotes
   / 69 restores / 128 gave-backs; 9-wide round-shape histograms (three batch-9 boots).
6. campaign/results/t36-lifestic.log (leg structure, ctl 373-chunk vs ela 24-chunk cycles, park
   gate off/on demote lines, per-leg metrics).
7. campaign/results/t38-sanity.log + .out (the full /metrics JSON: conversation_cache parks 46 /
   restores 3 / evictions 10 / slots 4; elastic counters; per-request first_engine_token_s,
   prompt_ms, reused; cohort WSTAT 363 windows p50 23.8/p99 30.7/max 42.6; hardware RAM 77/198 GB).
8. campaign/results/t32b-*.out, t38-final-serve.out, t38-tiebreak.out (leg structure; pooled
   records context M1 215.2 / pair 206.4 / conc 323.0).

### Internal — journal, doctrine, prior dossiers (consumed, extended, not restated)
9. SVG-CUSTOM.md F-272 (:9154), F-273 (:9174), F-275 (:9191), F-276 (:9221), F-277 (:9240),
   F-278 (:9310). 10. SVG-CUSTOM-FEATURES.md §1a (the lifecycle era; the :49 claim corrected in
   C1). 11. AGENTS.md laws 2/3/4/4b/8/9. 12. T35-elastic-refinement.md (R1-R8; C1-C5; the
   promote-line anatomy whose A8 fix is now live in the log strings). 13.
   T33-dynamic-session-allocation.md v1-v3. 14. T33-priorart-serving.md (Llumnix, vLLM sleep,
   Mooncake, ServerlessLLM, Sarathi; the recompute-vs-swap rule E6 prices). 15.
   T33-priorart-materialization.md (vAttention Table 3; pool semantics). 16. T37-scheduling-spec.md
   (C3 ternary; C5; the F-249 void verdict). 17. T37-pathways-memory.md (V8 64 MiB negative;
   SplitZip; no batched map through CUDA 13.4).

### External (this session unless marked carried)
18. Characterizing and Optimizing KVCache at a Large LLM Provider — arXiv 2506.02634 (USENIX
    ATC'25), abstract fetched. 19. vLLM RFC #37003 context-aware retention API (2026-03, open) —
    fetched; "hold not a hint", 15-40 s boundary, hybrid-model caveat. 20. KVFlow — arXiv
    2507.07400 (NeurIPS'25), steps-to-execution eviction, 2.19×. 21. ServeGen — NSDI'26, arXiv
    2505.09999, fetched: ITT ~100 s-class, long tail; ~3.5 turns/conversation. 22. Mell — arXiv
    2501.06709, fetched: migrate-vs-recompute boundaries, batching cuts migrations 25-30%. 23.
    UniCache — SIGMETRICS'26 [LIST]: log-normal inter-turn fits; chat ≫ agentic gaps. 24. LMCache
    tech report — arXiv 2510.09665 [LIST]: cache lifetime decoupled from engine lifetime. 25.
    Mooncake — FAST'25 (carried) + kvcache.ai engineering blog [LIST]: hit-rate collapse 83→36%
    without tiering. 26. vLLM sleep-mode docs (carried, re-verified): cuMem pool, graphs replayed
    not recaptured. 27. vLLM RFC #34303 CUDA checkpoint/restore (2026-02) [LIST]: their 2-10 s
    re-capture pain = our zero. 28. vLLM blog, zero-reload model switching (2025-10-26) [LIST].
    29. SGLang Unified Radix Cache (LMSYS 2026-08) + HiCache / llm-d architecture [LIST]. 30.
    HAServe — arXiv 2605.08581, fetched: consumer-count-ranked retention (DART), 5.90× p99 TTFT
    class. 31. Selective KV transfer — arXiv 2607.28150 [LIST]. 32. An Internet for the KV Cache —
    arXiv 2608.01526 [LIST]. 33. Optical KV cache network — arXiv 2610.18444 [LIST]. 34. "A Year in
    LLM Serving" trace study [LIST, via HAServe citations]. 35. "The Five Eras of KVCache,"
    Modular blog 2026-02 [LIST]. 36. vAttention arXiv 2405.04437 (carried): per-call VMM costs,
    background-thread mapping. 37. Llumnix arXiv 2406.03243 (carried): append-only migration, the
    tail-flip negative N4. 38. vLLM PagedAttention arXiv 2309.06180 (carried): recompute ≤ 20% of
    swap — E6's boundary rule. 39. SplitZip arXiv 2605.01708 (carried): the int8 lossless bound
    behind N7.

Source count: 39 numbered (17-18 internal primary sets, 22 external across literature / vendor
docs+RFCs / engineering blogs / carried primaries), plus the code anchors.

## METHODS NOTE

Productive: splitting custom-iq3_s.log at "engine started" and mining per-boot event streams (the
1.2-parks/request rate, the 25% background-promote share, and the 9-wide round-shape histograms
are all session-original measurements); reading the /metrics JSON in t38-sanity.out (the cache
slots=4 default against a 48 GiB budget is the wave's central finding and comes from that file);
the generate.cpp reads that located the park on the request thread (:13565-13590 before the read)
and the two-hop restore comment (:11610-11612); ServeGen/ATC'25/RFC #37003 fetches that anchored
the hysteresis and eviction findings.
Unproductive / do-not-repeat: grepping for "DONE" to count requests (batch completions print
differently — count parks or use the telemetry's requests field); searching the t36-lifestic.log
for restore lines (it is a battery-console capture; the engine-side restores live only in
custom-iq3_s.log); attempting to decompose promote map_ms against release (T35 C3 warned this is
internally consistent only since the A8 timer split — the split is now live, use release/map
fields directly); re-deriving the watermark's value from pre-A4 numbers (the log's release 1.3 ms
supersedes them); "HoME"-class tiering systems and TalleQ (already recorded unfindable, T33).
Estimates are marked [HYPOTHESIS]; every ms in §1/§3 traces to a mined distribution or a code
timer.
