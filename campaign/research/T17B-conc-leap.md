# T17B — THE CONC LEAP 320→400+ and COLD-START FORENSICS

Date: 2026-10-17. Scope: (1) forensic pricing of the ~10% cold-boot warmup loss from code + boot logs;
(2) the next structural escapes beyond STRATA_COHORTS=2 for 6-client concurrent decode;
(3) external harvest (40 sources, each with a verdict against our 6-client single-GPU exact-match MTP setup).
Method: line-level reads of generate.cpp / verify.cpp / expert_source.cpp / pool.cpp at the standing tree, the
five most recent boots in custom-iq3_s.log (incl. tonight's cold 272.0 / warm 299.5 legs), and 40 externally
verified sources (arXiv 2023-2026, engine blogs/docs, kernel and OS docs). READ-ONLY session — no boots, no builds.

---

## 0. EXECUTIVE SUMMARY

1. **The cold-vs-warm signature is real, measured, and mostly NOT mysterious.** Cold leg 267.2 rows/s in-window
   vs warm 301.6 (+12.9%; wall 272.0→299.5 = +10.1%). Three code-confirmed contributors: (a) the engine
   **deliberately drops the model files' page cache at every boot** when it pins the 41.6 GiB RAM complement
   (expert_source.cpp:2847-2860, `MADV_DONTNEED` + `POSIX_FADV_DONTNEED`) — so every fresh boot re-reads
   **44.5-54.8 GB from storage inside the first request(s)** (log, `file tier I/O this request` lines), and
   campaign/prewarm.sh is **structurally defeated** because it runs BEFORE the drop; (b) the CPU-experts pool
   term runs **+33% cold** (2.71 vs 2.04 ms/window, batch summary lines custom-iq3_s.log:37061 vs :38354);
   (c) ~6 lazy batch-graph captures + adaptive-tier refill (~3.4k expert swaps) land inside the first leg.
2. **Cheapest fast-warm lever: run the prewarm AFTER the engine is up** (post complement-pin, pre-probe), plus
   boot the tier from the **learned** profile (`--expert-profile data/expert-profile-learned.bin`, saved every
   30 requests since Oct 14, currently never loaded back — the launcher points at the static
   expert-profile.bin). Both are config/script-class; priced +10-13% on a first leg, ~+4-6% on a five-boot mean.
3. **The rows 8→12 ladder is DEAD at 6 clients — COHORTS already spent it.** Post-COHORTS all 6 slots decode
   every round (cohort collect cap generate.cpp:10214: 3 slots × 2 rows = 6 ≤ kVerifyMaxT=8,
   verify_kernels.hpp:21). The cap first bites at **10+ clients** (5 slots × 2 rows = 10 > 8). T13B C3's
   +31-36% arithmetic was a single-window-era number; do not spend a battery on it at 6 clients.
4. **The measured two-stream contention factor is ~1.45x per-row** (F-233: 2×6-row windows per ~26-29 ms round
   vs one 6.6-row window per 22.9 ms → 3.47→2.29 ms/row). That number prices the next escapes: COHORTS=3
   (+8-15% est., VRAM-cheap: ver_c ≈ +130 MiB vs 1,262 MiB free) and the **8-client lane** (parallel 8,
   config-only, the zero-code falsifier that also measures the contention curve at width).
5. **14% of the probe wall is admissions** (1.6 s of 11.56 s; ~36 × 32-58 ms `slot takes` restores).
   Overlapping admissions with in-flight windows is the largest single line left (+6-12% wall), L-cost.
6. Ranked queue: (1) 8-client config boot [free falsifier], (2) post-boot prewarm + learned profile [S],
   (3) COHORTS=3 [M], (4) admission overlap [L], (5) next-round staging pipelining + cross-cohort tail
   reorder [+2-4%, M], (6) PDL re-arm (F-191's own re-open condition is now met) [free], (7) FastMTP retrain
   (compounds: +4-5% conc). Stacked realistic ceiling at 6 clients: ~380-420.

---

## 1. COLD-START FORENSICS

### 1.1 The boot timeline (custom-iq3_s.log, boot of 2026-10-17 20:55:11, line 37248; identical shape 20:21/20:33/20:39 boots)

| t (relative) | event | log evidence | throughput cost while cold |
|---|---|---|---|
| 0-2 s | weights, projections, drafter | `1466 MiB ... in 0.1 s`; `draft layer loaded ... 0.24 s` | none (before serving) |
| ~3 s | expert cache pre-fill 9,366 slots | `pre-filled 9366 of 9366 slots from the profile in 3.5 s (5505 MB/s)` | none |
| ~3-21 s | 41.63 GiB page-locked complement | `allocating 41.63 GiB page-locked cache complement` → `copied cache complement through layer 48/48 (41.63 GiB, 10 s)` → `ready ... in 18 s` | **the drop happens here** (below); copy runs at 2.3 GB/s = NVMe speed, not RAM speed |
| boot end | solo graphs 1-8 captured | `capturing the 1-token window` … `captured the 8-token window` | none |
| boot end | batch verifiers + COHORTS on | `batch windows of up to 6 sequences` ×2; `STRATA_COHORTS=2 on: slots [0,3) on verifier A, [3,6) on B` | none |
| request 1 | **44.5-54.8 GB read from storage** | `file tier I/O this request: the OS read 44514.2 MB from storage (7 major faults) for 19312.3 MB of expert reads` (20:21 boot); 54803.3 MB (109 faults) on the 01:41 Oct-16 boot; 54803.0 MB (20:55 boot) | storage stall + readahead pressure lands inside the first lane |
| requests 1-10 | hit-rate ramp | decode expert cache hit rate: **88.1% → 95.9% → 97.8% → 98.6% → 99.0% → 99.1%** | first-request miss work is 10x the warm miss rate |
| requests 2-10 | adaptive tier refills | `resident RAM: ... 0 exchanged` → 50 → 89 → 104 → 112 → 117 → 128 → 142 | swaps run between windows (generate.cpp:10306-10307 thread; join :10425-10433) but eat PCIe/DMA share |
| first probe windows | **lazy batch-graph captures** | `strata verify: captured the batch window over slots 0,0 / 1,1,0,0 / 1,1,2,2 / 3,3,2,2 / 3,3 …` (verify.cpp:2867) | ~6 layouts × capture+instantiate+upload, one-time, inside the first 6-conc round(s) |

### 1.2 Mechanism 1 (dominant, code-confirmed): the boot-time page-cache drop defeats prewarm.sh

- Code: `drop_file_cache()` — `madvise(MADV_DONTNEED)` on the engine's own shard maps + `posix_fadvise(fd, 0, 0,
  POSIX_FADV_DONTNEED)` on the shard fds — fires whenever `pin_depends_on_reclaim(free_now, bytes, reserve)`
  is true before the 41.6 GiB `cudaHostAlloc` (expert_source.cpp:2847-2860; message at :2899-2901). #1250's
  comment (:2852-2857) explains why: the driver's one-shot page-lock cannot wait for kernel reclaim; a host was
  OOM-killed.
- Every boot triggers it, because the OS uses spare RAM to cache the shards it just loaded ("36.46 GiB really
  free for a 41.63 GiB … released: 83.75 GiB free", 20:55 boot). `POSIX_FADV_DONTNEED` on 0..EOF drops the
  whole file's cache (madvise(2)/fadvise semantics, man7.org — SOURCES #40).
- Consequence: **campaign/prewarm.sh (F-220) warms the cache before the boot, and the boot throws it away.**
  The 20:55 boot shows no `prefill: the GGUF experts are 100% in the page cache` line (the Oct-16 prewarm-era
  boot showed it, and still read 51.9 GB from storage on request 1 — dropped after the check).
- Then the first prefill routes ~all 24,576 expert pairs through the mmap (50.3 GB of pairs, T12-MEM ledger)
  with a cold cache: the measured 44.5-54.8 GB storage read, with ~2.3-2.7x readahead amplification. The
  engine's own watchdog treats this state as "slow, not stuck" (#1407 file-tier allowance,
  generate.cpp:9829-9857) — the codebase itself acknowledges multi-second silent steps from file reads.
- Open sub-question (see §7): which consumer exactly issues the 44.5 GB (prefill stager mmap reads are the
  prime suspect; the engine has a ready instrument — `src.io_stats()`/`io_prefetch` counters that print
  `page cache held X MB at the read, not Y MB`, generate.cpp:13664-13669, currently silent in our boots).

### 1.3 Mechanism 2 (measured, mechanism ranked): the pool term runs +33% cold

Batch summary lines, same day, same binary recipe:
- cold leg (20:21 boot, log :37061): `713 windows, avg 4.89 rows, 15.11 ms/window = run 14.30 (CUDA0 GPU-reach
  wait 0.65 + CPU experts 2.71) … 267.2 rows/s … slot drafts accepted 67.2%`
- warm leg (20:55 boot, log :38354): `700 windows, avg 4.98 rows, 14.21 ms/window = run 13.33 (CUDA0 wait 0.59
  + CPU experts 2.04) … 301.6 rows/s … 67.6%`
The +0.67 ms/window of CPU-experts work persists across the whole leg (700 windows), so it is not only the
first-window park-storm (POOL_DIET already removed the trailing reparks, F-227; workers park with a
spin-then-sleep after STRATA_POOL_SPIN_US=100, pool.cpp:529-593). Ranked hypotheses: (a) kernel
readahead/page-refill pressure and kswapd activity while 44+ GB flows into page cache DURING the leg;
(b) CPU frequency ramp on freshly-woken workers; (c) first-encounter page-table/THP work on the fresh 41.6 GiB
complement. falsifier: a post-boot prewarm boot that ends the storage read before the leg starts — if the pool
term still reads >2.4 ms, the cause is not the cache refill (one boot, read the summary line).

### 1.4 Mechanism 3 (minor, one-time): lazy captures + tier refill

- Batch graph layouts are captured lazily on first encounter of each active-slot pattern
  (`captured the batch window over slots …`, verify.cpp:2867; the F-233 two-phase stage/launch exists precisely
  so a capture never happens while the other cohort's window is in flight — generate.cpp:10194-10199 fences the
  standing→cohort transition). ~6 layouts seen tonight; each capture is capture+upload+sync on the service
  thread. A boot-time self-probe (one throwaway window per common layout at session-up) would move this off the
  first lane entirely.
- The adaptive tier starts from the STATIC profile (`--expert-profile data/expert-profile.bin` in the launcher)
  and re-learns heat every boot: cold leg shows `59 rounds, 3377 experts swapped in` and still climbing vs warm
  `87 rounds, 4925`. The learned file is written every 30 requests (`expert profile saved to
  data/expert-profile-learned.bin (periodic)`, in every boot log) but never read back — generate.cpp:583
  documents the intended round-trip ("as a profile `--expert-profile` reads").

### 1.5 The priced gap and the fast-warm package

Cold leg 267.2 vs warm 301.6 rows/s in-window = **+12.9%**; wall 272.0→299.5 = **+10.1%**. Attribution:
storage-read stall inside the first lane(s) + readahead pressure (largest, Mechanism 1), pool-term inflation
(Mechanism 2, +4.7% of window time), captures + tier refill (Mechanism 3, small). On a five-boot mean with the
first leg of each boot depressed, fixing the cold start is worth ~**+4-6% on the mean** and ~+10-13% on any
first leg.

Ranked fast-warm levers:
1. **Post-boot page-cache re-warm** (S, script): run prewarm.sh (or a leaner first-20GB pass) AFTER
   `session is up` and before the first probe. Removes the storage pull and the refill pressure from the leg.
   Cheapest falsifier: A/B two boots, first-lane decode rate + the `file tier I/O` line (should read ~0 MB).
2. **Learned-profile boot** (S, config): `--expert-profile data/expert-profile-learned.bin` in both launchers.
   falsifier: one boot, read first-request hit rate (today 88.1-88.4%) and cold-leg adapt swap counts.
3. **Boot-time layout self-probe** (S-M, engine): after COHORTS init, run one throwaway window per common
   layout to force the captures. falsifier: grep `captured the batch window` — should appear before
   `session is up`, not during the probe.
4. Keep the pre-boot prewarm too: it made the complement copy 2x faster historically (8 s / 5.2 GB/s on
   Oct-14 vs 17-18 s / 2.3 GB/s tonight) — boot-time only, no decode effect.
Do NOT arm STRATA_PIN_GUARD=0 to skip the drop: #1250's OOM kill is the documented failure
(expert_source.cpp:2852-2857). Guard=2 (paced, in-step registration) skips the explicit drop but lets kswapd
reclaim the same pages — no expected win, one diagnostic boot at most.

---

## 2. CONC LEAP 320 → 400+: THE POST-COHORTS ANATOMY

### 2.1 Where the time goes now (warm leg, log :38354)

- 700 cohort windows per probe, avg 4.98 banked rows/window, **14.21 ms/window** = run 13.33 (GPU-reach wait
  0.59 + CPU experts 2.04) + commit 0.31 + emit 0.32 + adapt 0.25 → **301.6 rows/s in-window**, wall 299.5.
- Round = stage A + stage B → go A → go B → reactor poll (generate.cpp:10260-10293) → finish(A) → finish(B)
  (:10419-10420) → adapt join (:10425-10433). Two ~5-row windows overlap on the GPU; B's pool work hides under
  A's GPU (the F-223 escape; batch_poll serves each verifier's doorbell ring round-robin,
  verify.cpp:3345-3425, pool call at :3388-3390, flag raises :3395-3405).
- The GPU-reach wait collapsing 16.4 ms → 0.59 ms is COHORTS' win, already banked. What remains per ~28 ms
  round: ~12 input rows of GPU serial + contention (~1.45x/row, from F-233's 3.47→2.29 ms/row), ~2-2.6 ms of
  serial host (staging both cohorts + both tails + adapt join), and **~1.6 s of admissions per 11.56 s probe
  wall (14%)**.

### 2.2 Ranked candidate table

| # | candidate | mechanism | priced magnitude | VRAM / regression risk | cheapest falsifier | sources |
|---|---|---|---|---|---|---|
| 1 | **parallel-8 lane** (`parallel: 8`, `--batch 8`, COHORTS 4+4) | rows/round 10→13.4 at depth-1 MTP (8 rows/cohort window = exactly kVerifyMaxT) | +15-25% conc net of window growth (expert bytes scale ~linearly on the 3.3 ms roofline share; light lines sublinear) | **+2 slot sessions = +1.46 GiB** → cache 9,366→~8,590 slots (−8.2%) ≈ −3 pt hit ≈ −7% conc-class counterpressure + M1 exposure (F-183 slope: ±513 slots = ±4.9% conc / ∓6.9% M1) | **config-only**: edit the conc json (parallel 8, batch 8), 3-leg probe; read rows/s + hit rate + M1 spot-check | §3: S1,S2,S3; F-183 |
| 2 | **post-boot prewarm + learned profile** (§1.5) | removes cold-lane storage stall + pool inflation | +10-13% first leg, +4-6% five-boot mean | none (host-side) | one A/B boot pair | §3: #40; log lines §1.1 |
| 3 | **COHORTS=3** (ver_c, slots {0,1}/{2,3}/{4,5} or 2/2/2) | third graph in flight: more pool-hiding + finer overlap; scaling from the measured 2-graph contention curve | +8-15% conc (2-graph efficiency 75%; assume 3rd adds ~half again, minus staging/tails ×1.5) | ver_b measured ≈ +117-126 MiB (free 1,388→1,262 MiB, boot logs); ver_c ≈ +130 MiB — **fits, cache untouched**; risk: reactor round-robin across 3 rings, capture fences generalize | none config-only; proxy = candidate 1's contention read first, then M-cost build behind STRATA_COHORTS | F-233; verify.cpp:3345-3425 |
| 4 | **admission overlap** (copy_to_slot on a side thread/stream while windows run) | today admissions run on the main thread BETWEEN windows; copies are stream-ordered (conversation_copy.hpp:36) and sync per QSA layer (generate.cpp:10049-10150) | admissions = 14% of probe wall; hiding ~half = **+6-8% conc** | bit-identical; hazard: adapt tier loan vs admission (the code already refuses swaps during prompt reads, generate.cpp:10511) | zero-code read first: sum `slot takes … (copied in X ms)` per results file; then one-thread prototype | F-232 C9; T13B §1.1 |
| 5 | **next-round staging pipelining + cross-cohort tail reorder** | stage A(N+1)/B(N+1) while window N runs needs per-verifier ping-pong buffers (F-196 parity-commit pattern is the template); launch both cohorts' drafts before waiting either (F-206 generalized across cohorts) | +2-4% (hides ~1-2 ms of the ~28 ms round) | bit-identical; the staged-buffer overwrite is exactly the G2 ghost — per-verifier double-buffer makes it structurally impossible (same argument as R2-S3 §1) | STRATA_VERIFY_PROFILE stage table around the between-window seam | T11.R2; F-196; F-206 |
| 6 | **PDL re-arm** (pdl.hpp is complete; F-191 no-op was PRE-cohort) | F-191's own re-open condition — "re-test ONLY if the window structure changes (post-pipelining/G2)" — is now MET: two graphs share the GPU and the host round-trip that hid launch spacing is gone | +0-3% | env/boot only; parity gate per F-191 discipline | env flip, one 3v3 | F-191; §3: #37 |
| 7 | **FastMTP retrain** (T16 W2, unchanged) | slot acceptance 67.6%→~75% lifts banked rows/round 10→10.5+ at constant window cost | +4-5% conc, compounds with all above | different bits → full gate (acceptance + 2x MMLU) | the T14A/B/C G0-G5 ladder | §3: #21; T14C |
| 8 | **MTP-off / MAX_SLOTS re-test under COHORTS** | F-231's "draft rows cost more than they return at 6 slots" was measured pre-COHORTS; under cohorts the marginal row's pool cost hides, so the trade may have moved (either way) | ±2-4% | config-only (STRATA_BATCH_MTP_MAX_SLOTS / =0 boot); MTP-off collapses pair by law — conc launcher only | one boot, read rows/window vs window ms | F-231 |
| 9 | **adapt-async vs DMA_BATCH trade** | log: `adaptive overlap: off: not with STRATA_DMA_BATCH` — the async tier is structurally disabled while DMA_BATCH is armed; DMA_BATCH saved 84.9% DMA submit cycles (T13B) | ±1-3% (adapt join is 0.25 ms/window) | both arms standing envs, reversible | one boot with STRATA_DMA_BATCH=0, read `asynchronous adaptive tier` lines + conc | T13B §1.4; log line |
| 10 | **L2 carve re-tune on the cohort topology** | boot sets `L2 set-aside: 60 MiB window / 60 MiB carve over a 112 MiB span on 3 stream(s)` — the stream count/identity changed with cohorts | ±0-2% | env-only (T15C's untested L2-72 candidate) | env boot, lts__t_sector hit-rate | T15C; §3: #38 guide |
| 11 | **multi-row sampler** (T13B C10, still open) | sample_rows is per-row launches + sync at the last stage (called in batch_poll, verify.cpp:3417) | +0.5-1% | bit-identical class | stage profile around the head | T13B C10 |

### 2.3 The rows-ladder re-price (kill the zombie)

T13B C3 priced "rows 8→12" at +31-36% conc — measured in the single-window era where only 4 of 6 slots
decoded per window. Post-COHORTS (generate.cpp:10193-10236): both cohorts run every round, cohort collect takes
slots while `S + (mtp_use ? 2 : 1) <= kVerifyMaxT` (:10214) → 3 slots × 2 rows = 6 ≤ 8: **the cap is not
binding at 6 clients; all 12 input rows already flow per round.** The cap first binds at 10 clients
(5 × 2 = 10 > 8) or if slot drafts ever deepen past 1 (spec-8 falsifier). The queued "MTP=0 batch-8
falsifier" is still worth one boot — but only as the marginal-row curve for candidate 1 (width 8), not as a
precursor of a 12-row ladder at 6 clients. verify.hpp fixed arrays (brow_[8]/b_out_[8]/last_rows_[8],
:351/:357/:374) + pool `kMaxWindowEntries` would only need touching for the ≥10-client world.

### 2.4 COHORTS=3 detail (the structural candidate)

- Measured basis: F-233's round went 22.9 ms (one 6.6-row window) → ~26-29 ms (two ~6-row windows overlapped)
  = **75% parallel efficiency at 2 in flight**; tonight's warm leg confirms (2×14.21 ms windows + ~2 ms host ≈
  28-30 ms round, 9.96 banked rows).
- Mechanism: a third ~4-row graph (2 slots × 2 rows) shortens per-window GPU serial and adds a third pool-work
  shadow. The reactor loop (generate.cpp:10281-10292) generalizes to three `batch_poll` calls; the pool is a
  single global ExpertPool (15 workers) whose per-round work is ~constant — hiding improves as GPU windows
  shrink. Two risks are priced: (a) SM contention worsens sublinearly (75% → expect ~60-65% at 3);
  (b) host serials ×1.5 (staging, tails) — partially attacked by candidate 5.
- VRAM: ver_b cost read from boots: 1,388 MiB free pre-cohorts vs 1,262-1,271 MiB with everything loaded
  (tonight's boots) ≈ 117-126 MiB ≈ one verifier's 127.2 MiB device buffers (log line) minus shared weight
  views (WeightRef). ver_c ≈ +130 MiB → ~1,130 MiB free; expert cache sizing happens before verifier
  construction and is untouched (9,366 slots tonight, same as pre-cohort boots).
- No-Regression: the gate `cohorts == 2 && n_active > o.batch/2` (:10193) keeps solo/pair byte-identical; a
  COHORTS=3 gate must keep the same property (n_active ≤ batch/3 degenerates to 2 cohorts, ≤ solo to one).
- Cost: M (a third Verifier + partition logic + fences). Sequencing: AFTER candidate 1's contention read.

### 2.5 What the outside world says about this shape (detail on the load-bearing verdicts)

- **Two windows in flight = the right family.** SGLang v0.4's zero-overhead scheduler (future tokens + event
  overlap, SOURCES #8) and vLLM V1's unified scheduler (#10) both keep one batch of host work ahead of the GPU;
  our COHORTS reactor is the same shape with the doorbell protocol standing in for events. SGLang issue #19347
  (#9) documents overlap silently failing when a dependency sneaks back — our analog is any tail that must
  join before the next stage (candidate 5).
- **Spec+batch interaction.** Su et al. (#23) measured spec-decoding's benefit shrink as batch grows (batch
  expansion costs); our F-231 is the same law locally, and MagicDec (#29) shows the benefit can return at
  large batch when drafting is cheap relative to a bandwidth-bound verify — ours IS bandwidth-bound at 98%
  hit, which is why depth-1 drafts still pay (+67% banked rows for ~40% more input rows). EqSpec (#30) is the
  correctness fence: never adopt batch-expansion or tree verify; our exact-match depth-1 is the safe class.
- **Expert offload under concurrency.** Fiddler (#31), ProMoE (#33), MoE-Infinity (#34), DuoServe-MoE (#35),
  KTransformers (#36) all attack the miss/stream we already hid under the GPU; at 2.0-2.3% miss and pool
  hidden behind windows, their mechanisms are mostly spent here — PARTIAL at best (prefetch-predictor ideas
  remain for the miss tail). Pre-gated MoE (#32)'s next-layer prediction is the one portable idea (graph-safe
  L2-prefetch of predicted experts), re-priced DOWN because our misses are already covered.
- **Disaggregation papers (DistServe #5, Splitwise #6, Mooncake #7, Sarathi-Serve #4)**: single GPU →
  physical disaggregation DOESN'T apply; the portable residue is chunked/co-scheduled prefill (Sarathi's
  stall-free mixing) which our read_part/decode-share already approximates (T13B C11), and admission overlap
  (candidate 4) is our local version of their "don't stall decodes on admissions".
- **TRT-LLM blog20 (#13)**: graph-batch ladder padding waste — our layouts are exact-fit per cohort pattern
  (graph key on rows), so the padding class DOESN'T apply directly; the transferable point is keeping the
  captured-layout set small and deliberate (our lazy captures, §1.4).
- **Green contexts (#38, #39)**: SM partitioning between the two cohort streams would cut the 1.45x contention
  factor — APPLIES as a C-cost experiment AFTER candidate 1/3 price the contention curve; NVIDIA power-wall
  caution (T9.A#6) stands; L2 is NOT partitioned so the expert stream shares anyway.

---

## 3. EXTERNAL SOURCES (40; each verified this session)

Format: source — relevance — verdict (APPLIES / PARTIAL / DOESN'T) given our 6-client single-GPU exact-match
MTP, expert-offloaded MoE, CUDA-graph cohort setup.

1. Orca, OSDI'22 — https://www.usenix.org/conference/osdi22/presentation/yu — iteration-level (continuous)
   batching origin — PARTIAL: our batch lane is already iteration-scheduled with slot rotation; the paper's
   selective-batching problem doesn't exist here (single shared graph).
2. PagedAttention / vLLM, SOSP'23 — https://arxiv.org/abs/2309.06180 — paged KV + admission memory math —
   DOESN'T: our KV is fixed-slot (kv int8 + resident cells) with conversation parking; no fragmentation class.
3. vLLM PagedAttention blog — https://vllm.ai/blog/2023-06-20-vllm — same — DOESN'T (reference only).
4. Sarathi-Serve, OSDI'24 — https://arxiv.org/abs/2403.02310 — chunked prefill mixed with decodes, stall-free
   schedules — PARTIAL: our read_part decode-share (generate.cpp:11801-11888) is a coarse version; the
   admission-overlap candidate (#4 in table) is the sharper local form.
5. DistServe, OSDI'24 — https://arxiv.org/abs/2401.09670 — prefill/decode disaggregation + goodput — DOESN'T:
   one GPU; but its goodput framing matches our aggregate-vs-M1 metric tension.
6. Splitwise, ISCA'24 — https://arxiv.org/abs/2311.18677 — phase splitting across machines — DOESN'T (single
   GPU; phases already share our streams).
7. Mooncake, FAST'25 — https://arxiv.org/abs/2407.00079 — KVCache-centric disaggregated pool — DOESN'T for the
   architecture; PARTIAL for the tiering idea (our KV pinned-RAM tier already plays that role locally).
8. SGLang v0.4 zero-overhead scheduler — https://lmsys.org/blog/2024-12-04-sglang-v0-4 — future tokens + event
   overlap, zero GPU idle — APPLIES: the design our COHORTS reactor implements; its remaining lesson (keep
   host work one step ahead) = candidate 5.
9. SGLang issue #19347 (overlap failing) — https://github.com/sgl-project/sglang/issues/19347 — hidden
   sync defeating the scheduler — APPLIES as a warning: our adapt join (:10425) is exactly such a hidden sync.
10. vLLM V1 alpha — https://blog.vllm.ai/2025-01-27-v1-alpha-release.html — unified scheduler, near-zero CPU
    overhead — PARTIAL: validates the target structure; nothing portable beyond what COHORTS already does.
11. TRT-LLM architecture overview (overlap scheduler + CUDA graphs) —
    https://nvidia.github.io/TensorRT-LLM/architecture/overview.html — APPLIES (validation): overlap +
    graph replay is the production-proven pairing we run.
12. TRT-LLM tech blog 12 (guided + spec decoding under overlap) — https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.md —
    spec under overlap scheduler races — APPLIES as a gate template: byte-exact differentials (we already run
    S3.3-style token-equality gates).
13. TRT-LLM blog 20 (CUDA graph batch-size tuning) — https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.md —
    padding-waste economics of graph ladders — DOESN'T: our layouts are exact-fit per pattern; no padding class.
14. NVIDIA spec-decoding blog (TRT-LLM, up to 3.6x) —
    https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x —
    PARTIAL: MTP-class drafting under graphs is our standing state; their two-model drafting modes are
    contract-irrelevant (single drafter, exact verify).
15. Baseten production spec decoding with TRT-LLM — https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm —
    PARTIAL: acceptance-vs-batch operational notes; no mechanism we lack.
16. SGLang MTP blog — https://lmsys.org/blog/2025-07-17-mtp — MTP serving; depth/batch tuning under load —
    APPLIES (corroboration): depth-1 drafts under load is exactly our F-231/COHORTS shape.
17. SGLang Qwen3.8-Flash-Next day-0 — https://www.lmsys.org/blog/2026-08-26-qwen-flash-next — OUR EXACT MODEL
    CLASS served at scale (hybrid GDN+attention+MTP) — APPLIES: their batch-MTP metadata handling is the
    reference for any depth>1 reopening.
18. vLLM Qwen3-Next hybrid blog — https://vllm.ai/blog/2025-09-11-qwen3-next — hybrid linear+full attention
    serving — PARTIAL: architecture kin; no cohort/overlap content beyond #10.
19. PyTorch blog: hybrid models in SGLang — https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention —
    Mamba-pool + KV-pool split for hybrids — PARTIAL: mirrors our GDN state vs QSA KV split; no new lever.
20. DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437 — MTP training + 85-90% second-token
    acceptance + 1.8x decode — APPLIES (target-setting): our MTP is the same mechanism; their acceptance is
    the FastMTP ceiling argument.
21. FastMTP — https://arxiv.org/abs/2509.18362 — inference-aligned MTP head retrain (2.1x) — APPLIES: THE
    acceptance lever (T16 W2); draft-only, output-safe.
22. TurboSpec (goodput control) — https://arxiv.org/abs/2406.14066 — closed-loop depth control under load —
    PARTIAL: our depth is min-p-gated and measured (F-228); the governor idea re-prices AFTER FastMTP.
23. Su et al., spec+batching synergy — https://arxiv.org/abs/2310.18813 — spec benefit shrinks with batch —
    APPLIES (law): matches F-231 locally; sets the expectation that our slot drafts stop paying near 8+ slots.
24. SpecInfer — https://arxiv.org/abs/2305.09781 — token-tree verify serving — DOESN'T: tree verify is
    contract surgery under exact-match (T15C closed the tree lane on c_row economics).
25. EAGLE-3 — https://arxiv.org/abs/2503.01840 — trained drafter frontier — DOESN'T for adoption (we own the
    native MTP head; FastMTP is the same lever); APPLIES as ceiling context.
26. Medusa — https://arxiv.org/abs/2401.10774 — extra heads — DOESN'T: needs training + tree verify.
27. AdaEDL — https://arxiv.org/abs/2410.18351 — entropy-based draft stopping, +10-57% — PARTIAL: our min-p
    gate already trims chains (F-228 measured the curve); an entropy bound could replace min-p AFTER a better
    drafter exists.
28. Sequoia — https://arxiv.org/abs/2402.12374 — DP-optimal tree topologies — DOESN'T: trees are closed here
    (chain-only verdict loop, generate.cpp:12957-class; T15C).
29. MagicDec — https://arxiv.org/abs/2408.11049 — spec pays even at high batch when verify is bandwidth-bound —
    APPLIES (explains why our depth-1 drafts still pay at 6 clients at 98% hit).
30. EqSpec / correctness forensics of batch spec decoding — https://arxiv.org/abs/2510.22876 — ragged-batch
    spec silently corrupts — APPLIES (fence): the reason our exact-match + per-cohort byte gates are
    load-bearing; never adopt batch-expansion verify.
31. Fiddler — https://arxiv.org/abs/2402.07033 — CPU/GPU expert split, window scheduling — DOESN'T: numerics-
    changing CPU experts violate our canaries; pool already hides behind windows.
32. Pre-gated MoE — https://arxiv.org/abs/2308.12066 (note: repo TODO's 2408.13666 is a mis-ID; 2308.12066 is
    the ISCA'24 paper) — next-layer expert prediction to break the routing dependency — PARTIAL: portable as
    L2-prefetch of predicted experts; at 2% miss the EV is small.
33. ProMoE — https://arxiv.org/abs/2410.22134 — proactive expert caching (2.07-2.20x) — PARTIAL: their regime
    is high-miss; ours is 97-99% hit with the tier already adaptive.
34. MoE-Infinity — https://arxiv.org/abs/2401.14361 — activation-aware expert offloading — PARTIAL: same
    high-miss regime assumption; the activation-correlation idea could seed the tier's refill order (we boot
    from a static profile today — candidate 2).
35. DuoServe-MoE — https://arxiv.org/abs/2509.07379 — dual-phase (prefill/decode) expert scheduling + next-
    layer prediction — PARTIAL: phase specialization is real here (prefill routes 100% of experts); the decode
    side is where we already are.
36. KTransformers, SOSP'25 — https://dl.acm.org/doi/pdf/10.1145/3731569.3764843 — CPU/GPU hybrid MoE serving —
    DOESN'T: CPU-expert numerics + different contract (same class as Fiddler).
37. CUDA Programming Guide: Programmatic Dependent Launch —
    https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html —
    APPLIES: the mechanism behind candidate 6 (sm_90+ incl. sm_120; works in captured graphs per F-191's read).
38. CUDA Programming Guide: Green Contexts —
    https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html — PARTIAL:
    SM partition per cohort stream is the contention attack (candidate after 1/3); L2 not partitioned; C-cost.
39. open-infer: co-locating prefill and decode with green contexts — https://open-infer.org/blog/green-ctx —
    APPLIES (evidence): SM-splitting stabilizes decode latency when phases share a GPU — our cohorts share one.
40. madvise(2) — https://man7.org/linux/man-pages/man2/madvise.2.html — MADV_DONTNEED / locked-page semantics —
    APPLIES (explains §1.2): the drop mechanism and why pinned pages are immune; pairs with POSIX_FADV_DONTNEED
    on the shard fds (posix_fadvise(2), same site).

---

## 4. CONTRADICTIONS AND THEIR RESOLUTIONS

1. **T13B C3 (+31-36% for rows 8→12) vs post-COHORTS code**: resolved — C3 was single-window arithmetic;
   the collect cap (:10214) already admits all 6 slots' rows per round. Re-priced to "binds at ≥10 clients"
   (§2.3). The T13-master-queue item 5 stays closed at 6 clients.
2. **F-231's "draft rows lose at 6 slots" vs MagicDec/Su (#23, #29) "spec pays when verify is bandwidth-bound"**:
   both hold — F-231 measured the loss pre-COHORTS when the marginal draft row paid full window time; under
   cohorts the pool-side cost hides and only GPU contention remains. Not resolved analytically; candidate 8's
   one-boot falsifier decides. No adoption without it.
3. **Repo TODO's Pre-gated-MoE ID (2408.13666) vs the real paper**: resolved this session — the ISCA'24
   pre-gated paper is 2308.12066 (verified); 2408.13666 could not be verified as that paper. Cited as 2308.12066.
4. **prewarm.sh "works" (F-220, morning-gap fixed) vs tonight's cold 272.0**: both true — prewarm fixed the
   pre-boot cache; the complement pin then drops it (§1.2). The lever moves to post-boot.
5. **"9,337 slots" (briefing) vs 9,366-9,481 (tonight's logs)**: the cache auto-sizes per boot from free VRAM;
   all within the ±500-slot band F-183 calibrated. No action.

## 5. OPEN QUESTIONS

1. Which path issues the 44-55 GB first-request storage read (prefill stager mmap vs another consumer)?
   One boot with the io_stats/io_prefetch instrument (generate.cpp:13664-13669) names it.
2. The cold pool-term mechanism (+0.67 ms/window): cache-refill pressure vs CPU ramp vs fresh-arena page
   work. The post-boot-prewarm A/B separates refill-pressure from the rest (§1.3 falsifier).
3. The 3-graph contention curve: candidate 1's 8-client boot reads width-scaling; COHORTS=3's go/no-go
   hangs on it.
4. Whether conversation-cache-mib (49,152) is host or VRAM in this build — decides whether parallel-8's
   +1.46 GiB can be reclaimed without touching the expert cache (not found in the reads this session).
5. Whether slot drafts deepen (draft[1] rows) could fit under cohorts at 6 clients with a retrained drafter —
   T14C's c_row economics say no today; re-read after FastMTP.

## 6. IMPLICATIONS

- Tonight's actionable set, in order: (1) 8-client config boot — free, prices width + contention, the biggest
  single number the program lacks; (2) post-boot prewarm + learned profile — script/config, banks the cold
  signature as +4-6% on means; (3) candidate 8's MTP-off-under-cohorts boot — free, closes F-231's open
  generalization. Only then spend code: COHORTS=3 (M), admission overlap (L), staging pipelining (M).
- The No-Regression Law binds candidate 1 through the expert cache: parallel-8 must NOT shrink the cache to
  fit; if the +1.46 GiB cannot be found elsewhere (open question 4), parallel-8 becomes a diagnostic-only lane
  and COHORTS=3 carries the leap instead.
- Quality gates unchanged: everything bit-identical gates on byte-exact differentials; FastMTP on acceptance +
  2x MMLU. The prefill-acceptance tax law (F-230) forbids any prefill-bits change inside this program.

## 7. METHODS NOTE (so nothing is repeated)

- T11/T12 dossiers are sections of SVG-CUSTOM-TODO.md (T13B's methods note already says so — re-confirmed).
- The engine log holds ~50 boots; boot boundaries are `engine started` lines (tonight: 15:37, 20:21, 20:33,
  20:39, 20:55). The 4.9-row/13-15 ms lines are COHORTS 6-conc probes; 3.1-row/14-15 ms lines are pair lanes.
- Unproductive searches: campaign/research has no T11.R2 file (it is SVG-CUSTOM-TODO.md:810-828); no pool.cpp
  under src/core (it is src/kernels/cpu/pool.cpp); "kVerifyMaxT" lives in include/strata/kernels/
  verify_kernels.hpp:21, and the row-cap expressions are generate.cpp:10214/:10440 (T13B's :10144 has shifted).
- External verification corrections: SpecInfer is 2305.09781 (not 2402.05109); TurboSpec/goodput is 2406.14066
  (2310.18813 is the spec+batching synergy paper); Pre-gated MoE is 2308.12066. All three were mis-citable
  from memory — verified before citation.
- No builds, boots, server calls, or git operations were performed; the only file written is this dossier.

## 8. SOURCES

All 40 numbered sources with URLs and verdicts are in §3. Categories: literature/papers 20 (arXiv/USENIX/ACM),
engine blogs & docs 11, NVIDIA/CUDA documentation 4, OS/kernel docs 1, local code+logs 4 (explicit file:line
throughout §1-2). Every external URL was surfaced and confirmed by search this session (2026-10-17); arXiv IDs
were checked against abstract pages or HTML listings, not memory.
