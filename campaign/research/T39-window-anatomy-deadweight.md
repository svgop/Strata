# T39 — WINDOW ANATOMY AT NODE LEVEL (where the ms live now) + THE DEAD-WEIGHT AUDIT (flags, args, and engine code that no longer pay)

Date: 2026-10-23 (wave T39, researcher R1). Scope: READ-ONLY — no boots, no builds, no git, no
batteries (a live server is serving; its /proc environ was read, nothing touched). Every internal
claim carries a file:line or log-line read THIS SESSION at HEAD `3219b765` (F-278's commit).
Required reading consumed in full and BUILT PAST: T37-{kernel-math, scheduling-spec,
pathways-memory}, T26-3 (diet/PDL chain), F-204-node-inventory, SVG-CUSTOM-FEATURES.md, AGENTS.md
(laws 4b, 9, F-232 cost-anatomy, F-263 five-boot pooled), the F-ledger F-200..F-278, the standing
launcher + config, `campaign/pdl-accuracy.json`, and the committed evidence under `campaign/results/`
plus the live boot log `custom-iq3_s.log` (engine stderr; 63,904 lines; boot boundaries found per the
F-277 lesson: `grep "engine started" | tail -1`).
Grading: [MEASURED-OURS] = banked record or read this session; [CORROBORATED] = primary external
source fetched or search-verified this session; [CARRIED] = verified by a prior wave at a stated HEAD
and re-anchor-checked here; [HYPOTHESIS] = my arithmetic from measured terms, marked where the
constant is unmeasured.

---

## 0. SUMMARY (the wave in nine lines)

1. **Five record corrections banked (§1):** (a) **PDL is ARMED BY DEFAULT at HEAD** —
   `verify_kernels.cu:2084` reads `getenv("STRATA_DF_PDL") … : 1`, and the live boot prints
   "strata pdl: armed (STRATA_DF_PDL=1); 1 of 9 families WAR-off" with the env UNSET; three records
   say the opposite and are stale. (b) The **C5 cohort WSTAT sample and the standing WSTAT sample
   cover different phases** — the cohort push (`generate.cpp:12086`) excludes the finish tails the
   standing push (`:12440`) includes; the cohort histogram understates its round by ~1.4-2 ms.
   (c) The tasking's "spec-8 buffer class for a dead spec-8" is **misattributed**: the T=8 arena
   sizing is the BATCH lane's row cap (rows/slot=2, `generate.cpp:9197/:9351`), load-bearing at any
   cohort ≥4 slots — not dead. (d) **LFUSE is dead per F-241 S5** (measured −3.0% F-206 + it
   displaces BOTH adopted folds: `!sg_gated_[grp]` guards hits_in_combine `verify.cpp:2299` and
   dm_fold `:2202`); F-277's "-344 nodes, four one-boot arms" diet tranche must drop LFUSE → three
   arms, ~−200 nodes. (e) **STRATA_VERIFY_PROFILE and STRATA_DM_WAIT_STATS both DISABLE the dm
   fold** (`:2201-2202`) and VERIFY_PROFILE also disables FUSE_HITS (`:2299`) — the F-218 profile
   instrument can no longer re-baseline the standing composition.
2. **The standing window at depth 6 + dm (+PDL, armed) reads 12.9-14.9 ms across the newest five
   boots, row-FLAT**: ~13.4 ms at ~3.2 committed rows (M1 class) and ~13.2-13.7 at ~5 rows (pair
   class) — the memory-bound regime (rows ride the weight reads). Composition (TREND lines,
   `custom-iq3_s.log`): run 11.4-13.8 (inside it: the graph replay is the critical path — GPU-reach
   wait 8.9-9.9 ms on the M1 class where the host has only ~2.5 ms of pool work to hide; 0.32-0.72 on
   the 5-row class) + commit 0.34-0.36 + emit 0.35-0.38 (emit includes the drafters' `draft_finish`
   waits) + adapt 0.41-0.95.
3. **The cohort round's 2× is two memory-bound windows, not bloat**: the stage→windows-done span
   (the only thing C5 measures) is p50 24.3 / p99 37.0 / max 42.3 over the newest 500 rounds (all-time
   n=17,234, mean 24.21). Arithmetic: two ~6-row batch graphs (~2,660 nodes post-dm/FUSE_HITS,
   F-204's 2,900 minus ~192 dm nodes minus ~48 FUSE_HITS) replay **serially on the one stream** —
   F-233's overlap hides pool/doorbell HOST work under the opposite window's GPU, it cannot overlap
   the GPU work itself. The leverage left in the cohort lane is the SERIALIZED FINISH TAILS
   (A-then-B, `:12205-12206`) + admission stops — C2a's program, now with measured terms.
4. **The launch-map diet phase 2 re-priced and re-scoped**: consumed since F-204: dm (−2 nodes/group
   × 2 groups × 48 layers = **−192**), FUSE_HITS (−48), norm_fast (execution, node-neutral),
   POOL_DIET (host waits). Remaining IN-TREE gates: FUSE_WAIT (−144), DF_PLE (−56),
   UseNodePriority (scheduling) → **−200 nodes ≈ −0.10..−0.19 ms/window at the measured 0.48-0.95
   µs/node**; the rest of the −900..−1,000 needs NEW kernels (q8_1 folds −180, router 4→2 −96, hc
   3→2 −96, batch-QSA suite −330 at S=6, small-pairs −48, expert-side −52..−92 residual).
5. **CUDA-graph fixed terms are already near their floor on the standing window**: instantiate+upload
   at capture (`:2636-2639, :3572-3575`), one `cudaGraphLaunch` per window (through
   HostGraphLaunch, F-200), node-replay marginal 0.48-0.95 µs (ours) vs ~1.5-2 µs/node forum class
   and ~4.2 µs/node CAPTURE cost (arXiv 2501.09398, A100). The host-launched conditional graph
   (T37 C1) remains the only structural cut of the draft-loop's fixed terms; nothing in the CUDA
   13.4 surface changes that plan (§2).
6. **The one diagnostic re-baseline boot (§3 A1)**: standing env + `STRATA_VERIFY_NODES=2` (CUDA
   census works, `verify.cpp:2579-2606`), one M1 probe + one 6-conc probe; optionally a second boot
   `STRATA_DUMMY_NODES=16` to re-price the node marginal post-dm. NOT STRATA_VERIFY_PROFILE — it
   measures a fold-less window (correction e).
7. **The dead-weight audit (§3.B)**: 33 env arms + 28 config args, each with gate site and verdict —
   **31 KEEP** (24 with live engagement proof), **2 SUSPECT** worth one five-boot pooled drop-test
   (POOL_SPIN_US=100, v1-era; PREDICT_CREDIT/BUDGET, proof era = 103k), **0 DEAD on the launcher**
   (the F-187/F-267 cleanups already removed the dead ones); engine-side: SpecKeep + falsified
   instruments all zero-standby-cost and retained deliberately; the only actionable engine
   dead-weight is *documentational* (the PDL default-vs-record split) and the uncontrolled
   `expert-profile-learned.bin` accumulation (F-271 hazard, hygiene-law controlled).
8. **Negatives (§4)**: profile-boot re-baseline (poisoned), LFUSE arm, spec-8 buffer removal,
   graph-update-as-a-mechanism (topology-identical constraint), and the carried falsifications
   restated so nobody re-buys them.
9. Records to beat unchanged: five-boot pooled M1 215.2 / pair 206.4 / conc 323.0; best singles
   222.1/228.3/237.4. The user's bar: the biggest honest bands this wave are C2a (+5-8% conc
   arithmetic), the diet tranche-3 (−0.10..−0.19 ms ≈ +1-2%, five-boot territory), and TC-hc /
   two-stage head (+6-12% / +4-6%, T37-kernel-math — not re-derived here).

---

## §1 OUR-STATE VERIFICATION (anchors at HEAD 3219b765; corrections are victories)

### 1.1 The standing machine, proven from the live process

The live server (pid 1444142, booted 2026-10-23 17:04:35, `/proc/1444142/environ` read this session)
carries exactly the launcher's **33 STRATA_ exports** and no more (notably NO STRATA_DF_PDL, NO
LFUSE/FUSE_WAIT/DF_PLE, NO QFUSE, NO SPEC_TRACE, NO GR_NORM_FAST — the last is the code default,
F-241). Config: `--spec 6 --spec-min-p 0.5 --batch 6 --kv int8 --kv-resident 32768 --pcie-frac 0.35
--pcie-mode dma --vram-reserve-mib 2000 --conversation-cache-mib 49152` (+ cvec, profile, PLE args).
Engagement lines in the live boot (log 63778-63904): "STRATA_DM armed" (:63841), "STRATA_COHORTS=2
on: slots [0,3) A, [3,6) B" (:63875), "L2 set-aside: 60 MiB window / 60 MiB carve over a 112 MiB
span" (at the device's 60 MiB persisting cap — F-276's falsified-72 correction visible), "ELASTIC_
SESSIONS on: 6 slot arenas … 2 mapped … 2.25 GiB freed in 7 ms", "chunk unit 32.0 MiB … watermark
on … timed scan on, park-on-demote on", "routing-aware prefetch on" (the PREDICT_CREDIT substrate),
"typical acceptance: 516 of 621 first-rejections rescued (threshold 0.20)" and "2646 of 5186
first-rejections rescued" (TYPICAL_ACCEPT is a LIVE, large term on the serial lane),
"suffix drafts: 192 windows, 562 of 781 drafts accepted" (the suffix/lookup chain fires on real
traffic — the serial T>6 window families are NOT dead on production traffic).

### 1.2 CORRECTION 1 — PDL is armed by default at HEAD (three records stale)

`pdl_supported()` (`src/kernels/cuda/verify_kernels.cu:2084`): `static const int mode = [] { const
char* v = std::getenv("STRATA_DF_PDL"); return v ? std::atoi(v) : 1; }();   // opt-in until the A/B
says otherwise` — **the unset default is 1 = armed** (subject to the per-device sm_90+ PTX check,
`:2088-2091`, which the 5090 passes). The live boot prints `strata pdl: armed (STRATA_DF_PDL=1);
1 of 9 families WAR-off: quantize_q8_1 (defaults)` (log 63841-area, boot of 17:04:35, env unset).
Stale records, all corrected: (i) `campaign/pdl-accuracy.json` env line "unset/0 = PDL entirely off
(the standing recipe; nothing changes)"; (ii) F-278's "STRATA_DF_PDL unset = byte-identical
behavior"; (iii) T37-pathways §1 "STRATA_DF_PDL remains unset in the launcher → standing value 0"
(and `verify.cpp:1208`'s own "opt-in: STRATA_DF_PDL=1" comment). Consequences, priced:
- The armed surface = 8 of 9 families; standing also has STRATA_DENSE_TC unset (=off, boot line
  "mask 0"), so the live sites are native_bf16 ×7 + native_mmvq ×5 + gpu_stamp ×1 ≈ **13 of ~1,860
  window nodes** → T26-3's partial-chain pricing applies: **−0.005..−0.02 ms/window — harmless**.
- But **every standing boot since T38 is an "armed boot" with no entry in pdl-accuracy.json**,
  violating that record's own append protocol; and F-278's OWED item "V2 re-price arm
  STRATA_DF_PDL=1" is moot — the owed leg is `STRATA_DF_PDL=0` (the OFF leg, §3 A2).
- T26-3's old anchor (:1709-1720, default 0 = "opt-in") was correct at bf746391's tree state; the
  flip came with the T38 V2 landing. The inline comment ("opt-in until the A/B says otherwise") is
  now wrong relative to its own code — a comment fix rides any next engine commit.

### 1.3 CORRECTION 2 — the two WSTAT histograms measure different phases

- Standing path: the sample is pushed at `generate.cpp:12440` as `msd(w0, Clock::now())` — AFTER
  `bt_run` (w0→w1), `bt_commit` (w1→w2) and `bt_emit` (w2→now, which includes the per-slot
  `draft_finish` waits at `:12428-12434` and the `fflush`). **Standing WSTAT = window+commit+emit.**
- Cohort path: the sample is pushed at `:12086` as `msd2(w0c, w1c)` where w1c is stamped at `:12073`
  right after "cohorts: windows done" — **stage + launch + poll ONLY; the finish tails (`:12205-12206`),
  the draft waits inside them, and the adapt join (`:12211-12219`) are excluded.** The in-code comment
  (`:12080-12085`) calls this "the same phase the standing sample covers" — it is not.
- Practical effect: the C5 histogram (363 windows, p50 23.8 / p99 30.7 / max 42.6, F-278) and the
  log-wide distribution (n=17,234, mean 24.21, max 83.6; newest 500: p50 24.3 / p99 37.0 / max 42.3)
  UNDERSTATE the cohort round by the tail (~1.4-2.0 ms, §1.6). C2a's instrument must either add the
  tail to the sample (one line: push `msd2(w0c, w2c)`; w2c exists at `:12207`) or push two samples.
  (C5b, §3 A3.)

### 1.4 CORRECTION 3 — the "spec-8 buffer class" is load-bearing, not dead

The tasking's premise ("falsified arms whose buffers still allocate — T=8 buffers for a dead spec-8,
F-270's VRAM ledger") fails three ways. (a) F-270 contains no VRAM ledger; its verdict table is the
T32 tranche judgment (SVG-CUSTOM.md:9070-9094). (b) The T=8 sizing is set by **BATCH_MTP**, not by
any spec-8 arm: both batch verifiers init at `batch_mtp ? kVerifyMaxT : …` (`generate.cpp:9197` for
split stages, `:9351` for ver_b) because the batch lane stages **2 rows/slot up to kVerifyMaxT=8**
(`:11993`, the collect loop; kVerifyMaxT=8 at `include/strata/kernels/verify_kernels.hpp:21`). At 6
clients each cohort is 3 slots × 2 = 6 rows; at 9-wide (5+4) cohort A packs 8 rows and rotates the
5th (T37 C3's measured cap). The two 127.2-MiB window arenas ("window up to 8 tokens, 127.2 MiB of
device buffers", log 63778+) ARE the T=8 sizing — **removing it would break the batch lane's ≥4-slot
cohorts**. (c) What IS depth-8-flavored and dead-on-probe-traffic: the serial lane's `exec_[7]`/
`exec_[8]` family members — but graphs are captured LAZILY per seen T (`capture()`, `verify.cpp:2512-
2514`: `if (exec_t != nullptr) return true;`), so uncaptured T costs **zero VRAM and zero replay**;
and on real traffic the suffix/lookup chain extends T past 6 up to S=8 (`:3270` raises `o.spec` to 8
with lookup chains; live "suffix drafts" lines), so even that class is alive in production. **Verdict:
nothing to remove.** The honest per-verifier figure: ~60 MiB × 2 verifiers is the 8-vs-6 sizing
delta, and it is the price of the 9-wide row class.

### 1.5 CORRECTION 4 — LFUSE is dead (F-241 stands at HEAD); the diet tranche loses an arm

F-277 restates the T37-pathways V3 tranche as "LFUSE/FUSE_WAIT/DF_PLE + UseNodePriority … = −344
nodes ~ −0.17..−0.33 ms, four one-boot arms." That contradicts F-241 S5 (SVG-CUSTOM.md:8271-8281),
which adjudicated LFUSE dead by code read + F-206's measured −3.0%: on the device-plan path LFUSE
sets `sg_gated_[grp]`, and `hits_in_combine` requires `!sg_gated_[grp]` (`verify.cpp:2299`) — arming
it displaces the adopted FUSE_HITS fold and re-adds ~48 moe_hit_add launches — and **the same guard
now also sits on the dm fold** (`:2202`: `… && !g_dmws.live() && !sg_gated_[grp] && …`), so LFUSE
would displace dm too. Verified at HEAD: both guards present. **Tranche re-scoped to three arms:
FUSE_WAIT=1 (−144), DF_PLE=1 (−56), UseNodePriority flag 8 on the three instantiate sites
(`verify.cpp:2776, :3738, :3745` — still flags=0 this session) ≈ −200 nodes ≈ −0.10..−0.19
ms/window** at R1's 0.48-0.95 µs/node marginal (pre-dm number; A1's second boot re-prices it).
LFUSE's re-open clause (F-241): a combine form that keeps hits-added under the gate.

### 1.6 The standing window anatomy at depth 6 + dm (TREND lines, newest boots)

From `custom-iq3_s.log` "strata batch:" TREND lines (the per-window accounting is
`generate.cpp:12478-12501`; run = w0→w1 graph span incl. the host's GPU-reach wait and pool; commit =
w1→w2; emit = w2→end incl. draft waits; adapt separate):

| class (committed rows/window) | ms/window | run | GPU-reach wait | pool (CPU experts) | commit | emit | adapt | source line |
|---|---|---|---|---|---|---|---|---|
| ~3.2 rows (M1 probe class) | 13.41 | 12.14 | **8.97** | 2.51 | 0.02 | 0.57 | 0.68 | log:59963 |
| ~3.2 rows | 14.51 | 13.19 | 9.90 | 2.63 | 0.02 | 0.61 | 0.68 | log:61454 |
| ~5.0 rows (pair class) | 13.68 | 12.41 | **0.46** | 2.09 | 0.34 | 0.36 | 0.57 | log:63758 |
| ~5.0 rows | 13.00 | 11.71 | 0.32 | 2.07 | 0.34 | 0.35 | 0.59 | earlier boot |
| ~5.1 rows | 14.94 | 13.84 | 0.52 | 2.99 | 0.35 | 0.35 | 0.41 | log:62259 |

Readings (arithmetic beats adjectives):
- **The run span is row-FLAT** (11.7-13.8 ms from 3.2 to 5.1 rows): the memory-bound regime — draft
  rows ride the expert-weight reads. This is why keep-rule row-cutting lost solo (F-270's law) and
  why depth 7/8 as *window width* bought nothing (F-262/F-263): the marginal row costs ~0.2-0.5 ms,
  not c_row.
- **The graph replay is the critical path on every lane.** GPU-reach wait 8.9-9.9 ms on the M1 class
  = the host finishes its ~2.5 ms of pool work and then waits for the graph; on the 5-row class the
  wait is 0.3-0.7 because the pool has more rows to compute per window. The F-222-era "GPU-reach wait
  is the dominant line" is CONFIRMED post-adoption-wave, in this sense: the graph itself, not host
  scheduling, is what the window waits on. Host-side levers (isolation, CCD split) move the VARIANCE
  of this wait (T37 V1), not its mean.
- **Inside the graph** (the F-218 map's line structure, adjusted per T37-kernel-math §6): attention
  ~0.42 light-class post-F-221; VRAM expert hits ~1.9-3.5 (byte-floored 87-92%); hc reads ~2.0-2.9
  (the TC-hc target); head ~0.45-0.78 (the two-stage-head target); copy+combine now folded (dm);
  router/plan/bell ~0.7; the rest small. The dm fold removed the waitCPU+copy+combine chain (~0.35-0.5
  ms/window by F-264's split: 74% of ~477 µs of waits + launches). PDL's 13 armed sites shave ≤0.02.
- commit+emit+draft ≈ 0.7-1.2 ms/window on the standing path; the M1 class's commit 0.02 vs the
  5-row class's 0.34 is the single-slot commit skip (one-token self-commit) vs
  `commit_slot_prefixes`.
- **The windows-done→throughput conversion**: at 13.3 ms and ~2.8-3.0 committed tok/window (M1 215 ×
  13.3 ms), the window-transfer law (±20 t/s per ∓1.0 ms/window, T15A) says a −0.19 ms diet win ≈
  +3-4 t/s ≈ +1.5-2% — five-boot-pooled territory, exactly the user's "not +1%" boundary; the diet
  tranche composes but does not lead.

### 1.7 The cohort round's 2× decomposed (the "what is inside 24 ms" answer)

Measured: cohort spans ("cohorts: windows done in X ms", `generate.cpp:12072`) — all-time n=17,234,
mean 24.21, max 83.6; newest 500: **p50 24.3 / p99 37.0 / max 42.3**. C5's first histogram (F-278):
363 windows, p50 23.8 / p99 30.7 / max 42.6. Consistent.

Inside the span (all sites at HEAD):
1. **Two 6-row batch graphs replayed SERIALLY on the one stream.** `batch_go` launches
   `exec_bm_[bkey]` on `cs_` (`verify.cpp:4065-4067`); A's and B's graphs QUEUE — the GPU executes
   them back-to-back; only host/doorbell work overlaps (F-233's mechanism: "B's pool work runs under
   A's GPU" — pool, not SMs). F-204 priced batch S=6 at ~2,900 nodes; post-dm (−2/group × 2 groups ×
   48 layers = −192, per the dm comment `verify.cpp:2194-2199`) and FUSE_HITS (−48): **~2,660
   nodes/window on the cohort lane**. At the 0.48-0.95 µs/node scheduling marginal that is
   1.3-2.5 ms of pure node overhead per window — but the ~10-11 ms body is expert-weight traffic,
   row-flat like the standing window. Two windows ≈ 21-23 ms of GPU; the measured 24.3 says ~2-3 ms
   of exposed host (staging both cohorts `:12043-12050`, poll-loop round trips `:12060-12071`,
   service the doorbells for BOTH cohorts — pool 2.2-3.7 ms/window each, mostly hidden).
2. **NOT in the span (excluded by the C5 phase, correction 2): the finish tails, run SERIALLY
   A-then-B** (`:12205-12206`): per cohort — exact-match acceptance walk (`:12099-12124`),
   `commit_slot_prefixes` (a parity-keyed commit graph launch, `verify.cpp:3965`), per-token BT emits
   + `fflush(stdout)` (`:12148-12202`), per-slot `draft_launch(2,…)` D2D copies + launches
   (`:12166-12185`), then `draft_finish` waits (`:12190-12196`). TREND accounts these as
   bt_commit+bt_emit **split 0.5/0.5 by an accounting approximation** (`:12209-12210` comment: "both
   tails ran inside this span; split evenly"). Estimated tail ≈ 1.4-2.0 ms/round total [HYPOTHESIS —
   the standing path's measured commit+emit 0.7-0.9 ms × 2 cohorts, minus overlap of the draft
   waits]. This is C2a's exact target (finish-A-under-B), and its instrument gap is C5b.
3. **The adapt join inside the round** (`:12211-12219`): 0.41-1.18 ms when it fires (every
   o.adapt_every rounds; the F-271-poisoned class hits 6-10 here FIRST — watch it per-window).
4. **The p99 mass (30.7-42.3)**: admission/ramp rounds, first-round transition fence
   (`cudaDeviceSynchronize`, `:11977`), lazy captures of new bkeys inside stage (gated on "NO window
   in flight", `:12041-12042`), and pool spikes — the F-271 hazard lives here too.

**Conclusion for the program**: the cohort 2× is NOT a fusion target — it is two honest memory-bound
windows. The lane's remaining leverage is (i) the serialized tails (C2a: overlap A's finish under B's
window; band below), (ii) admission stops (C2b/decode-share pricing, T37), (iii) the graph body only
via the shared T-flat lines (TC-hc, two-stage head — T37-kernel-math), and (iv) 9-wide's ternary form
(C3, after C5b + the config-only 9-wide telemetry leg).

### 1.8 Launch-map census: what is capturable-vs-streamed now

F-204's ~2,100-node T=8 doorbell window (36 GDN layers × 39 + 12 QSA × 52 ≈ 2,028 — the GDN=36
correction settled, F-278) has become, at HEAD: **~1,860** (−192 dm, −48 FUSE_HITS; norm_fast
node-neutral, POOL_DIET host-side). Batch cohort windows: ~2,660 (from 2,900). The diet's remaining
IN-TREE gates and their sites, verified this session:
- `STRATA_FUSE_WAIT` (`verify.cpp:317-346`; the CUDA fold consumed at `:2220-2225`, and `:334-348`
  `=1` also folds the CPU-rows wait via `copy_rows_from_mapped_after` — −144 nodes; note the fold
  path requires `!prof_on_ && m_bal_ == nullptr` at `:2220`, consistent with correction (e)).
- `STRATA_DF_PLE` (`verify.cpp:1355-1360`, "the window's post-projection steps in one pass, bitwise
  the per-token loop" — −56).
- `cudaGraphInstantiateFlagUseNodePriority` — **still flags=0** at all three instantiate sites
  (`:2776` commit, `:3738`, `:3745` window/batch via `instantiate_evicting` `:3733-3745`); priority
  streams exist (`:1016-1029`).
- Needs NEW kernels (phase 3, per F-204/T26-3, GDN-scaled where per-layer): q8_1 quantize folds
  (−180; QFUSE exists default-off with the F-187 mirror-trade caution), router 4→2 (−96), hc 3→2
  (−96), batch-QSA per-row suite (−330 at S=6 — the cohort lane's biggest single item), QSA
  small-pairs (−48), expert-side residual (−52..−92), misc (−60..−90).
- Capturable-vs-streamed: the four capture families bake only slot/lane/model-affine memory (F-272
  audit); residency changes swap families (refresh_ar `:2496-2508`, family select `:2513`), never
  recapture. Everything in the window graph is captured; the STREAMED (outside-graph) per-window work
  is: stage_inputs, PLE row prefetching (`:2953-2963`), the pool's doorbell service, staging, commit
  graphs, drafts, emits — the host segments the TREND line already separates.

### 1.9 Engine-side dead-weight inventory (the "still allocates / still resident" list)

| item | site | standby cost | verdict |
|---|---|---|---|
| SpecKeep + calibrated table | generate.cpp:2354-2401 | `apply()` returns T at th≤0 (one compare/window); one static | KEEP-DORMANT — the C1 keep-rule substrate (F-270's retire-to-conditional era) |
| STRATA_DRAFT_UNGATED twin | mtp.cpp:1741-1754 | branch at launch, zero when unset | KEEP (F-229 instrument, banked negative) |
| STRATA_DM_WAIT_STATS | verify.cpp:398-445 | allocates ONLY when armed; `live()` short-circuits on `on` | KEEP instrument; NOTE it disables dm while armed (by design — it measures the standing chain) |
| STRATA_VERIFY_PROFILE | verify.cpp:992, prof_ alloc gated | zero when off | KEEP instrument; NOTE it disables dm+FUSE_HITS (correction e) — never a re-baseline tool again |
| GR_WIDE / GR_TILEV / HC_SPLIT | fused_gr.cu:364, :1151-1154, :1382, :1477 | env-gated variants, zero when unset | KEEP (F-217's explicit instruments) |
| STRATA_DUMMY_NODES | verify.cpp:1431-1432 | capture-time only | KEEP — the node-marginal instrument (A1's second boot) |
| STRATA_VERIFY_NODES | verify.cpp:2539-2606 (CUDA at :2578) | capture-time only | KEEP — the census instrument (A1's first boot) |
| Batch verifiers' T=8 arenas (127.2 MiB ×2) | generate.cpp:9197/:9351 | VRAM | LOAD-BEARING (correction 3) — not dead weight |
| Serial exec_[7]/exec_[8] | verify.cpp:2512-2514 | zero (lazy capture) | not dead weight; alive on suffix traffic |
| pl_/pipeline (multi-GPU) paths | generate.cpp:9124-9215, pl_want :15046-class | init-time only on this box (n_stages=1, pipe=false) | keep — other-hardware surface, compile-gated |
| STRATA_BATCH_MTP_MAX_SLOTS | generate.cpp:11961 | static int | F-231 instrument, dormant, zero cost |
| STRATA_SPEC_RATIO | generate.cpp:12111/:12329/:15942 | static bool | dead while BATCH_RESCUE=0 (its only batch consumer); alive as the serial rescue twin — keep |
| Shelf launchers (spec8/olds7/calib-*) | repo root | already DELETED (F-267; only launch-custom-iq3s-8081.sh remains) | closed — git history is the archive |
| expert-profile-learned.bin | config :49-54, save-every 30 | disk, grows every 30 rounds | the F-271 HAZARD, not removable without losing learning — batteries must pin/reset (law exists); consider a save-interval revisit only with the F-275 refill tunables |

---

## §2 EXTERNAL SWEEP (confidence-graded; primary source on every claim)

**G1. CUDA graph mechanics, CUDA 13.4 Programming Guide (fetched 2026-10-23). [CORROBORATED]**
docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html: (a) "Calling
cudaGraphUpload can separate out the cost of allocation from the launch by performing all mappings
for that graph immediately" — and the caveat "Using different streams for graph upload and graph
launch … likely resulting in remap operations" (we upload on `cs_` and launch on `cs_` — same
stream, `verify.cpp:2639/:3575` vs `:2984/:4065`: correct by construction). (b) `cudaGraphExecUpdate`
"much more efficient than re-instantiation" but requires **topologically identical** graphs; single
-node updates "skip the topology checks". (c) Launch cost: "much of the setup is done in advance …
the graph itself can then be launched repeatedly with very little overhead." (d)
`cudaGraphInstantiateFlagDeviceLaunch` required for device launch; memory-pool aliasing across
non-simultaneously-executing graphs is legal. Relevance: our instantiate→upload→launch-per-family
loop is already the guide's recommended shape; the ONLY structural upgrade left in the graph surface
is conditional nodes (host-launched; T37 C1's verified design) — nothing in 13.4 re-prices it.

**G2. Per-node cost classes. [CORROBORATED + MEASURED-OURS]** (a) CAPTURE-side: "Kernel Batching
with CUDA Graphs" (arXiv 2501.09398, fetched): graph creation cost ≈ **4.2 µs per kernel node**
(fitted k_c, A100/CUDA 12.3; Grace-Hopper cross-check) with base ~0.16-0.42 ms; optimal batch ~100
nodes. This prices OUR lazy-capture tax (new bkey at request time: ~2,600 nodes × ~4 µs ≈ ~10 ms
— the F-271 first-request class, exactly) and motivates keeping the bkey family bounded
(`set_batch_graph_limit(64)`, `verify.cpp:9347`). (b) REPLAY-side: ~1.5-2 µs/node is the forum/
secondary class (NVIDIA forums null-kernel ~5 µs standalone); OUR measured replay marginal is
**0.48-0.95 µs/node** (R1 §1, carried) — the graph scheduler beats the standalone class, and the
diet node-arithmetic uses ours. (c) Non-graph boundary 5-10 µs (Event Tensor, MLSys'26, carried
T27-1) — we have no non-graph kernel boundaries in windows. (d) "Hybrid JIT–CUDA Graph Optimization
for Low-Latency Workloads" (arXiv 2604.23467, search-verified) and **GraCE** (USENIX OSDI'26,
usenix.org/system/files/osdi26-ghosh.pdf, search-verified): compiler-assisted capture/dynamic-shape
graphing — the field is building what our fixed-shape families already are; no import.

**G3. Conditional nodes / device graph launch. [CARRIED + CORROBORATED]** T37-scheduling S11
verified the 120/255/1 limits are DEVICE-launch-only and do not bind host-launched graphs;
re-confirmed the two NVIDIA blogs exist as primaries (device graph launch, 2022-12-12; conditional
nodes, 2024-05-10: IF/WHILE 12.4, ELSE/SWITCH 12.8) and XLA's on-device control-flow RFC
(github.com/openxla/xla/issues/6973) shows production-framework demand for exactly the C1 shape.
No new 13.4.x limit found this session. C1's GATE-0 (th=0 bit-identical) remains the entry.

**G4. The serving-stack analogs (padding vs our exact-shape families). [CORROBORATED]**
(a) vLLM CUDA-graphs design doc + discuss thread: batches **padded to the nearest captured size**;
capture sizes bucketed (powers of two + intermediates) — padding waste between buckets is the
accepted cost. (b) SGLang "Advanced CUDA Graph Techniques" (lmsys.org, 2026-08-17, fetched): three
backends (Full / Breakable CUDA Graph / TC-piecewise); BCG "builds prefill graphs 3.8-5.2× faster"
(no compilation), full capture still fastest at replay (1.93× vs BCG 1.70× vs piecewise 1.45× on
gpt-oss-120b); **"42 shapes across a 78-layer MoE add 2.4 GB of graph memory"**; spec runners =
"draft, draft-extend and frozen-KV MTP draft steps each get their own runner, while target verify is
the decode runner … capturing more than one token per request"; "token padding is expensive (extra
GEMM rows), request-slot padding is cheap." (c) PolyGraph (arXiv 2610.23861, search-verified):
plan-buffered adaptive graphs, names vLLM-style padding as the overhead source. **Mapping to us:**
our bkey'd per-composition batch family + per-T serial family = no padding at all (rows within a
composition are exact; slots rotate rather than pad) — the design the field is converging toward,
purchased with capture count, which the F-271 lazy-capture tax prices. Their 2.4 GB/42-shape memory
class vs our audit-baked families (modest pools, F-272) — no action.

**G5. PDL discipline. [CARRIED]** TRT-LLM's per-kernel WAR list + on-disk accuracy record (T27-1/
T38 V2's model); the CUDA PDL guide's visibility warning (T37-pathways §2); gu_swiglu correction
(not launch_pdl sites at HEAD — pdl-accuracy.json's own note). The NEW fact this wave is the
default flip (§1.2), not new PDL mechanics.

**G6. Nothing new on the memory tier.** No batched cuMemMap through CUDA 13.4 (T37-pathways
negative, carried — release notes unchanged this session per that scan); vAttention ~40 µs/call
class stands; 32 MiB superchunks remain the standing form.

Source count this section: 8 external primary/engineering sources fetched or search-verified +
carried primaries from T37 (verified there at bf746391, re-anchor-checked where cited here).

---

## §3 IMPLEMENTER-READY CANDIDATES (value-ranked) + THE DEAD-WEIGHT REMOVAL SLATE

Every candidate: mechanism → expected band vs the pooled records → cost-anatomy file:line →
falsifier-first → law-4b scoping → env-off rollback. Bands are against five-boot pooled
M1 215.2 / pair 206.4 / conc 323.0; the user's bar is "past the noise range," so sub-+2% solo items
are explicitly five-boot-pooled items and say so.

### A1 (rank 0 — gates everything): THE ONE DIAGNOSTIC RE-BASELINE BOOT(s)

- **Mechanism.** Re-derive the cost map from the CURRENT composition without poisoning it. Boot 1:
  standing env + `STRATA_VERIFY_NODES=2` (census at capture: per-family node counts + kernel-name
  histogram, `verify.cpp:2578-2606`) + one M1 probe + one 6-conc probe; read (i) node census for the
  four families (serial T∈{2..6} doorbell + all-resident, batch 6-row), (ii) both lanes' TREND lines,
  (iii) WSTAT both phases (post-C5b). Boot 2 (only if the diet is headed to a battery):
  `STRATA_DUMMY_NODES=16` (or 32) same probes — re-prices the node marginal post-dm (the 0.48-0.95
  number is pre-dm; F-204's deltas were counted pre-dm too — T37-pathways flagged this gap).
- **Band.** Zero direct; it converts every §3 band from estimate to measurement.
- **Cost anatomy.** §1.6-1.8; the instruments already in-tree and capture-time-only.
- **Falsifier-first.** It IS the falsifier layer: if the census shows the dm fold did NOT remove
  ~192 nodes (e.g. groups≠2 on the cohort lane), the diet arithmetic changes before any arm boots.
- **Law 4b/9.** Diagnostic boots are not arms (campaign law 5). Rollback: unset the two envs.
- **Cost: 2 boots.** NOT `STRATA_VERIFY_PROFILE` (correction e — it measures a fold-less window).

### A2 (rank 1 — closes a live record split): PDL DISPOSITION — pin the standing state

- **Mechanism.** The standing recipe runs PDL-armed by default with ~13 of ~1,860 nodes armed
  (§1.2). Two honest ends: (a) pin `export STRATA_DF_PDL=0` in the launcher — restores the
  documented standing semantics, makes every standing boot match pdl-accuracy.json's baseline entry,
  costs ≤0.02 ms; or (b) accept default-armed and backfill the record (one pdl-accuracy.json entry
  citing the live boot line + the standing golden hash). Either ends the split. THEN, if the F-278
  OWED re-price is still wanted, it is a proper A/B: `STRATA_DF_PDL=0` leg vs standing-armed, five
  boots pooled — expected inside noise (the armed surface prices −0.005..−0.02 ms), which is itself
  the record: "PDL armed-surface = no-op at current chain length; the chain extension (12 more
  files) stays the priced item" (T26-3 band −0.5..−1.5 for the FULL chain, untouched).
- **Band.** 0 ± noise; the win is measurement hygiene and record closure.
- **Cost anatomy.** verify_kernels.cu:2084 (the default), :2049-2054 (family gate), pdl.hpp:61-82.
- **Falsifier-first.** The A2 boot itself.
- **Law 4b.** No lane can regress ±0.02 ms. **Rollback.** Delete the launcher line (or `=1`).
- **Cost: 1 launcher line + (optionally) one 5-boot pooled A/B.**

### A3 (rank 2 — the C2a/C3 instrument): C5b cohort-WSTAT phase fix (one line)

- **Mechanism.** Push the cohort sample over the same phase the standing sample covers:
  `window_ms.push_back((float) msd2(w0c, w2c));` at `generate.cpp:12086` (w2c exists at `:12207`),
  and optionally two extra counters (bt tail split A/B real spans instead of the 0.5/0.5 split at
  `:12209-12210`). Without it, C2a's tail-overlap gain and C3's rotation loss are read off a
  histogram that excludes the very segment being reordered.
- **Band.** Zero direct; instrument for C2a (+5-8% conc arithmetic band, below).
- **Cost anatomy.** :12073/:12086/:12205-12210 (§1.3, §1.7).
- **Falsifier-first.** One 6-conc leg with fields non-empty; check the cohort p50 shifts ~+1.5-2 ms
  vs the current sample (that shift IS the tail, measured).
- **Law 4b.** Instrument, not arm. **Rollback.** Two-hunk revert (the C5 revert shape, F-278).
- **Cost: 2-line engine change + one telemetry leg.**

### A4 (rank 3 — the conc lane's head): C2a-lite, finish-A-under-B (T37 C2a with measured terms)

- **Mechanism.** The poll loop waits for BOTH windows (`:12060-12071`) before EITHER tail runs
  (`:12205-12206`). Reorder: as soon as A's `batch_poll` reports done, run A's finish (acceptance
  walk, commit graph, emits, next drafts) while B's window is still in flight; B's finish under the
  next round's A window. The commit graphs are parity-keyed for this class (`capture_commit_batch`,
  `verify.cpp:3583-3696`); the tail's CUDA calls are stream-ordered behind the window on `cs_` —
  the fence question is exactly TRT-LLM's zero-sync contract (S9): the finish's host segment
  (`draft_finish` waits) must not require B's graph done — it doesn't (drafters are per-slot
  streams).
- **Band.** Tail ≈ 1.4-2.0 ms of a ~26 ms round (24.3 span + tail) → hiding ~half of both tails ≈
  **−1.4..−2.0 ms/round ≈ +5-8% conc arithmetic** (26→24 ms ≈ +8% at 6 clients); pair/M1 untouched
  (the `n_active > o.batch/2` gate `:11972`). Conc record 323.0 → ~340-350 IF the admission stops
  don't dominate — which is why A3 precedes it.
- **Cost anatomy.** §1.7 items 2; the finish lambda is verifier-parameterized (`:12096`).
- **Falsifier-first.** A3's leg first (reads the true tail); then a zero-code pricing leg:
  `STRATA_BATCH_DECODE_SHARE=0.9` 3v3 (T37's falsifier — prices the ADMISSION share, the other
  tail competitor) — only then the reorder (S code, no new threads).
- **Law 4b.** Scoped by the existing cohort gate; standing path byte-identical. **Rollback.**
  `STRATA_COHORT_OVERLAP=0` gate (new), one line.
- **Cost: S code + the A3 leg.** This is the only candidate this wave with a >+4% single-lane band
  that is not already carried by T37.

### A5 (rank 4 — the diet, re-scoped): tranche of three one-boot arms

FUSE_WAIT=1, DF_PLE=1, UseNodePriority(=8) — sites and guards in §1.8. **LFUSE removed** (§1.5).
Node term −200 ≈ −0.10..−0.19 ms/window (A1's boot-2 re-prices with the post-dm marginal); plus each
arm's execution delta (FUSE_WAIT deletes a round trip per group — the F-264 wait-split says the
exposed wait is small post-dm, so the node term IS the arm; DF_PLE is bitwise-equal batched postops).
Band: **+1-2% M1 five-boot pooled** — under the user's bar ALONE; its value is composition (it
enlarges contiguous kernel→kernel spans for the PDL chain extension, T26-3 sequencing). Judge as
one composed tranche, five-boot pooled, M3b+golden-hash on every arm boot; UseNodePriority's watch:
priority inversion vs doorbell spin nodes (nsys the shared-expert fork). Rollback: `=0`/unset ×3.
Cost: 3-4 boots.

### A6 (rank 5 — carried, unchanged by this wave): the T-flat kernel lines

TC-hc rungs (band +6-12% M1, `fused_gr_bench` ≥1.3× gate), two-stage head (+4-6%, parity-harness
first + the sampler constraint), mid_chain/dec_gdn_rec (parity-gated; F-278 stopped mid_chain at
bitwise, reopen clause recorded). Not re-derived — T37-kernel-math owns them; this wave adds only:
the GDN=36 layer count scales their node arithmetic ×1.5 (already applied there post-F-278), and
the A1 census hard-numbers the per-line shares they will be judged against.

### B. THE DEAD-WEIGHT REMOVAL SLATE (the user's explicit ask; each item individually revertible)

Launcher arms (33 exports; gate sites verified; verdicts):

| arm | gate (file:line) | engagement proof (live boot / log) | evidence vintage | verdict |
|---|---|---|---|---|
| ATTN_DAILY=1 | kernel dispatch (PR#540 selector) | bit-exact class; M1/pair/conc +6.0/+7.0/+5.3 (F-221) | F-221, current | KEEP |
| GDN_CHUNKED=1 | kernels.cu prefill dispatch | prefill −4.2% every pair (F-224); decode neutral | F-224, current | KEEP |
| FUSE_HITS=1 | verify.cpp:2299 | in TREND composition (−48 nodes); +4.35% M1 (F-226) | current | KEEP |
| POOL_DIET=1 | pool.cpp:701-702 | +4.06% M1 (F-227) | current | KEEP |
| COHORTS=2 | generate.cpp:9344-9361, gate :11972 | "STRATA_COHORTS=2 on" :63875; conc +24.8% (F-233) | current | KEEP |
| TYPICAL_ACCEPT=0.2 | typical_threshold() generate.cpp:354 | "2646 of 5186 first-rejections rescued (threshold 0.20)" LIVE, log 63904-area | R5-era, RE-PROVEN LIVE | KEEP — a major live acceptance term |
| POOL_SPIN_US=100 | pool.cpp:697-699 (default kSpinBeforeSleep=20 ms, pool.hpp:243) | none printed (silent knob) | v1-era (F-05..F-64), pre-dm, pre-POOL_DIET | **SUSPECT — one five-boot pooled drop-test** (default 20 ms vs 100 µs shapes doorbell-pool wake latency; dm moved the wait into the kernel, so the spin's payer may be gone; cheap A/B) |
| KV_PREFETCH=1 | prefill.cpp:2576-2578 | prefill +5.3/4.1% at 32/64K (F-72 era) | prefill-only, decode nil | KEEP (prefill standing) |
| ADAPT_LAG=2 | generate.cpp (nv_defaults.cpp twin) | kills lag-1 event sync stall | old but structural | KEEP |
| DMA_BATCH=1 | pcie dma path (batch_read.hpp:48 class) | 84.9% fewer DMA submits | current | KEEP; INTERACTION: blocks ADAPT_OVERLAP (F-187, removed) and shapes C2b's reader-thread mode — name it in any C2b spec |
| VERIFY_DEVICE_PLAN=1 | device_plan_ verify.cpp:1113-class | "token graph hit path … decided on the device" :63838-area | current | KEEP (dm composes on this route, F-277) |
| ELASTIC_SESSIONS=1 | generate.cpp:9340-class init | "6 slot arenas … 2.25 GiB freed in 7 ms" | F-272/273 current | KEEP |
| ELASTIC_PARK / _WATERMARK / _TIMED_SCAN=1 | lifecycle gates | "watermark on … timed scan on, park-on-demote on"; park lines LIVE ("parked 84 tokens in 47.2 ms") | F-276 current | KEEP |
| VMM_CHUNK_MIB=32 | vmm chunk unit | "chunk unit 32.0 MiB … 24 driver calls per arena" | F-276 current | KEEP |
| SERVE_TELEMETRY=1 | WSTAT/prometheus plumbing | standing | F-276 current | KEEP |
| DM=1 | verify.cpp:492, arm :2201 | "STRATA_DM armed" :63841; pooled +2.5% (F-277) | current | KEEP |
| PREDICT_CREDIT=1.0 / PREDICT_BUDGET=2 | generate.cpp:10466-10490, expert_source.cpp:2984-2996 | substrate LIVE ("routing-aware prefetch on"); bounded lane (u≥0.5, credit≥1.5, budget 2/round) | proof era = 103k context; standing probes are ~6k | **SUSPECT-light — one five-boot pooled drop-test** (bounded cost, but never re-measured on the short-context standing probes post-superchunks); expect "keep" — the F-16 remediation already bounds its failure mode |
| ADAPT_PINCAP=48 / MIN_AGE=2 / DORMANT=64 | generate.cpp:10208-10214 | silent | B6-era, pre-watermark (the spare pool now holds an arena — victim-protection economics changed) | KEEP but fold into the SAME pooled drop-test leg as PREDICT_* if a boot window allows (one extra leg, not a campaign) |
| L2_SETASIDE_MIB=60 | set-aside setup | "60 MiB window / 60 MiB carve over a 112 MiB span" — at the device cap | current; 72 falsified by the cap (F-276) | KEEP |
| BATCH_MTP=1 | batch_mtp flag | slot drafters live ("6 slot drafters … 51 MiB each") | F-31, current | KEEP |
| BATCH_RESCUE=0 | batch_rescue_on() generate.cpp:363-366 (default OFF) | export is a no-op | documentation of the default | KEEP as documentation (one-line rollback flip is its value); note the cohort rescue branch (:12110) is dead code while =0 |
| SPEC_COUPLED=1 | mtp coupled path | "coupled draft sampling on … over 40525 tokens" ×6 | current | KEEP |
| PLE_BATCH=1 | PLE staging | silent; PLE lines live | current | KEEP |
| TYPICAL_DEVICE=1 | device rescue kernel | live (rescue counts above) | F-200 current | KEEP |
| VERIFY_ASYNC_LAUNCH=1 | verify.cpp:2980-2985 | HostGraphLaunch path | F-200 current | KEEP |
| SPEC_DRAFT_TEMP=0.85 | drafter sampling | "coupled draft sampling" line; proposals-only | F-201 (depth-era); F-263 reverts did not touch it; acceptance 74-87% standing | KEEP (any change = distribution-adjacent → law 10 gates) |
| ROUTE_RESIDENT=0.75 | router prior | "R4 hit path ON"; hit rates 96.8-99.9% | F-200, MMLU-gated | KEEP |
| ROUTE_RESIDENT_MTP=1 | MTP router bias | boot line self-labels "**experimental**" | F-72 era | KEEP (live, cheap), but retire the "experimental" label or the arm — one of the two |

Config args (28): all KEEP except notes — `--spec-min-p 0.5` (0.7 dead, F-265), `--pcie-frac 0.35`
(0.55 dead, F-263/F-265; calibrate.py sweep expected ~0), `--spec 6` (7/8 dead), `--expert-profile-
save*` (the F-271 hazard — hygiene law controls it; do not remove silently),
`--control-vector-*` (user feature, out of scope), `--conversation-cache-mib 49152` (now MORE
load-bearing post-PARK), `--vram-reserve-mib 2000` (the anti-paging floor, F-271 tested).

Engine-side slate: nothing deletable pays. The falsified arms (UNGATED, GR_WIDE, GR_TILEV, SPEC_KEEP,
BATCH_MTP_MAX_SLOTS, SPEC_RATIO, DM_WAIT_STATS, DUMMY_NODES, VERIFY_NODES) are all env-gated
zero-standby-cost instruments that F-217/F-229/F-231/F-270 explicitly banked — deleting them deletes
the falsifier layer. The ONE engine-side change this wave recommends is documentation + A2/A3 code
(pdl comment fix :1208 + verify_kernels.cu:2084's comment; C5b's one-liner).

---

## §4 NEGATIVES — researched and rejected this wave (with reasons; re-open clauses)

1. **Re-baselining the cost map with STRATA_VERIFY_PROFILE.** Poisoned at HEAD: it disables dm
   (`:2201`) and FUSE_HITS (`:2299`) — it re-measures a fold-less window and would overstate every
   line the folds removed (~0.5-0.7 ms/window class). Re-open: never for the standing composition;
   per-line profiling needs a stamp path that coexists with the folds (a dm-aware `stamp()`) — worth
   an S item ONLY if TC-hc/two-stage-head need per-line attribution beyond the census.
2. **Arming LFUSE as part of the diet.** Dead per F-241 S5 (F-206's measured −3.0%; displaces
   FUSE_HITS via `verify.cpp:2299` and dm via `:2202`). Re-open: a combine form keeping hits-added
   under the gate (F-241's clause) — phase-3 kernel work, not an arm.
3. **Removing the "spec-8 buffers."** The T=8 sizing is the batch lane's row cap
   (`generate.cpp:9197/:9351`); removal breaks ≥4-slot cohorts (9-wide). Re-open: never at 6 clients;
   the 9-wide ternary (C3) INCREASES the class's value.
4. **cudaGraphExecUpdate-based "dynamic graph" as a mechanism** (update one exec instead of families).
   Requires topologically identical graphs (G1b); our data-dependent structure is already
   topology-fixed per family with device-side plans — there is nothing to update per window. The
   only live uses: ver_c capture cloning (T37 open q5) and C1's conditional build. Re-open: if a
   future shape axis explodes family count (e.g. per-position bkeys — not on any roadmap).
5. **PDL blanket extension now.** The armed 13 sites read ~0 (§1.2); the FULL chain band
   (−0.5..−1.5 ms) still requires the 12-file trigger/staging work and stays sequenced behind the
   diet (T26-3 (e)); the nsys per-pair probe remains the entry gate (F-278 OWED, now framed as
   A2's optional A/B).
6. **Dropping TYPICAL_ACCEPT to "simplify" the recipe.** Live-proven this session (2,646 rescues in
   one real request; 51% of first rejections). Any change is distribution-adjacent (law 10) — the
   arm is evidence-backed; the drop-test would burn a battery to re-learn R5/F-200.
7. **Carried falsifications restated so nobody re-buys them** (F-263/F-265/F-276 + this session's
   confirmations): decode-share 1.0 constant, --pcie-frac 0.55, --spec 7/8, adapt-tier 1/80/0.97,
   --spec-min-p 0.7, STRATA_GEMM_LT, device-plan falsifier (void), PIPED-as-env-boot, L2=72 (device
   cap 60), CONV_COMPRESS (SplitZip bound: int8 KV has no exponent redundancy), GDN_PRESCALE
   (composed-boot canary confound — the isolated boot + KL pair is still owed), CCD_SPLIT (redo
   owed), reserve-600, static parallel-9, COHORTS=3-at-6 (dead by arithmetic; the F-249 measurement
   is VOID — T37's correction stands).
8. **"The cohort round's 2× is waste to fuse."** Rejected on arithmetic (§1.7): two memory-bound
   windows on one GPU queue; the F-233 overlap hides host work only. The 2× is the lane's honest
   cost; the leverage is tails (A4), admission (C2b), width (C3), and the graph body (A6).

---

## CONTRADICTIONS AND THEIR RESOLUTION

- **"STRATA_DF_PDL unset = byte-identical / standing 0" (F-278, pdl-accuracy.json, T37-pathways)
  vs verify_kernels.cu:2084 default 1 + the live boot line.** Resolved for the code + live process
  (three independent proofs: the code default, the /proc environ, the boot line). The records are
  wrong at HEAD; A2 disposes.
- **F-277's diet tranche (four arms, −344 nodes) vs F-241 S5's LFUSE kill.** Both in the journal;
  F-241 is the adjudicated verdict with the measured negative and the code citations, now re-verified
  at HEAD including the dm interaction. Tranche re-scoped (§1.5, A5).
- **C5's "same phase as the standing sample" (code comment :12080-12085) vs the two push sites.**
  Resolved by reading the spans (§1.3): different phases; C5b.
- **Tasking's "spec-8 buffers for a dead spec-8 (F-270's VRAM ledger)" vs the tree.** No such ledger
  in F-270; the sizing is batch-lane row cap, live. Corrected (§1.4).
- **T37-pathways' "LFUSE implemented + bitwise-checked, never armed — arm it" vs F-241.** T37 missed
  F-241's citations; F-241 stands (this is the same dossier-citation failure mode its own methods
  note warns about).

## OPEN QUESTIONS

1. The TRUE cohort tail split (commit vs emit vs draft-waits) — A3's extra counters; currently a
   0.5/0.5 accounting approximation (`:12209-12210`).
2. The post-dm node-replay marginal (0.48-0.95 is pre-dm) — A1 boot 2.
3. Whether PREDICT_CREDIT/ADAPT knobs pay anything on the short-context standing probes — one
   pooled drop-test leg (B slate); expect "keep," but the record should say so from data, not age.
4. Does the dm fold's internal wait expose any waitCPU residue (the F-264 split measured the
   STANDING chain; the fused kernel's staging is uninstrumented)? Only matters if the cohort p99
   (30.7-42) shows a pool-correlated class — A3's fields would show it.
5. Per-family node census values (the ~1,860/~2,660 figures here are F-204-derived arithmetic, not
   a fresh count) — A1 boot 1 settles every number this dossier derives.

## IMPLICATIONS

- The measurement posture changes in one way immediately: **never re-baseline with
  STRATA_VERIFY_PROFILE**, and treat "PDL off" claims as pre-T38 citations. The launcher should pin
  the PDL state (A2) so record and reality cannot drift again.
- The conc lane's next structural move is A4 after A3 — the tails are the measured residue, not the
  windows. The solo lane's next moves remain T37's kernel lines; this wave's diet (A5) is a
  composition item, not a headline.
- The dead-weight answer to the user: the launcher is CLEAN (31 of 33 KEEP with live evidence, 2
  cheap drop-tests owed, 0 dead — F-187/F-267 already did the cutting); the engine's falsified-arm
  residue is deliberate instrument-keeping at zero standby cost; the two real hygiene items are the
  PDL default-vs-record split and the uncontrolled learned-profile growth (both handled by existing
  laws once named).
- Every arithmetic claim in §1.6-1.8 that rests on F-204-derived node counts is flagged for A1's
  census; the wave's permanent outputs are the five corrections, the re-based anatomy frame, and
  the re-scoped queue.

## SOURCES

### Local (primary for §1; read this session at HEAD 3219b765)
Code: src/core/verify.cpp (:97, :317-346, :992, :1016-1029, :1137, :1208, :1355-1360, :1431-1432,
:1522-1624, :2194-2206, :2207-2263, :2299-2318, :2496-2513, :2512-2643 (capture + VERIFY_NODES CUDA
listing), :2646, :2776, :2921-2998 (run + async launch), :3551-3575, :3583, :3696, :3733-3745,
:3839-3846, :3965, :4040-4137); src/kernels/cuda/verify_kernels.cu (:1979-2070, :2084-2099,
:2102-2160); src/program/generate.cpp (:354, :363-366, :6891-6920, :9124-9215, :9181, :9197,
:9330-9365, :10208-10214, :10460-10500, :11747-11781, :11960-12228, :12205-12225, :12352-12440,
:12478-12501, :15660-15750, :15921-15962); src/core/mtp.cpp (:1316, :1366, :1646-1647, :1735-1782,
:1879); src/kernels/cpu/pool.cpp (:697-702, :776, :801) + include/strata/kernels/cpu/pool.hpp (:243,
:306-311); src/kernels/cuda/fused_gr.cu (:364, :1151-1154, :1382, :1477); src/spec/{controller,
draft_policy}.cpp (dormant harnesses); include/strata/kernels/verify_kernels.hpp:21;
include/strata/kernels/pdl.hpp.
Standing config: launch-custom-iq3s-8081.sh (33 exports), custom-iq3_s-8081.json (28 args),
campaign/pdl-accuracy.json (1 baseline entry + env doc now corrected by §1.2).
Logs/evidence: custom-iq3_s.log (63,904 lines; TREND lines :59963, :60791, :61238, :61454, :62259,
:62718, :62937, :63758; cohort spans 17,234 samples; live boot :63778-63904; engagement lines
individually cited); /proc/1444142/environ (the live process); campaign/results/ (t32b-*, t38-*
file lists; t36-lifestic.log), campaign/research/{F-204-node-inventory, T26-3-prefill-window,
T37-kernel-math, T37-scheduling-spec, T37-pathways-memory}.md.
Record: SVG-CUSTOM.md F-206 (via F-241's citation), F-217 (:7565), F-218 (:7601), F-221 (:7660),
F-222 (:7682), F-229 (:7871), F-231 (:7916), F-233 (:7965), F-241 (:8228), F-249 (:8460),
F-259-F-263 (:8800-8948), F-264 (:8949), F-265 (:8972), F-266 (:8986), F-267 (:9007), F-270
(:9070), F-271 (:9096/:9119), F-272-273 (:9154/:9174), F-275-276 (:9191/:9221), F-277 (:9240),
F-278 (:9310); SVG-CUSTOM-FEATURES.md; AGENTS.md (laws 2, 4, 4b, 5, 8, 9, 10, campaign law,
F-232/F-263 doctrine).

### External (fetched or search-verified 2026-10-23)
1. CUDA C++ Programming Guide, CUDA Graphs chapter (docs.nvidia.com/cuda/cuda-programming-guide/
   04-special-topics/cuda-graphs.html, fetched): cudaGraphUpload semantics + stream caveat;
   cudaGraphExecUpdate topology constraint + cost; instantiate-once/launch-cheap; DeviceLaunch flag;
   memory-pool aliasing rules.
2. "Kernel Batching with CUDA Graphs," arXiv 2501.09398 (fetched): ~4.2 µs/node capture cost
   (A100, CUDA 12.3), ~100-node optimum, T_i < T_a observation.
3. NVIDIA blog, "Enabling Dynamic Control Flow in CUDA Graphs with Device Graph Launch"
   (2022-12-12, search-verified): device fire-and-forget launch.
4. NVIDIA blog, "Dynamic Control Flow in CUDA Graphs with Conditional Nodes" (2024-05-10,
   search-verified): IF/WHILE 12.4, ELSE/SWITCH 12.8.
5. LMSYS, "Advanced CUDA Graph Techniques in SGLang" (2026-08-17, fetched): Full/BCG/TC-piecewise;
   capture-speed and replay figures; 2.4 GB/42-shape graph memory; spec-runner structure; padding
   economics.
6. vLLM CUDA Graphs design doc (docs.vllm.ai/en/stable/design/cuda_graphs) + discuss thread on
   capture-size padding (search-verified): bucket padding as the accepted overhead.
7. vLLM issue #28207 (2025-11, search-verified): prefill branches in decode graphs; capture-size
   filtering trade.
8. PolyGraph, arXiv 2610.23861 (search-verified): plan-buffered adaptive CUDA graphs for serving.
9. GraCE, USENIX OSDI'26 (usenix.org/system/files/osdi26-ghosh.pdf, search-verified):
   compiler-supported CUDA graph capture.
10. "Hybrid JIT–CUDA Graph Optimization for Low-Latency Workloads," arXiv 2604.23467
    (search-verified).
11. XLA RFC #6973 on-device control flow (github.com/openxla/xla/issues/6973, search-verified):
    production demand for conditional graph nodes.
12. SGLang Piecewise CUDA Graph docs (sgl-project.github.io, search-verified).
13. NVIDIA developer forums on kernel-launch latency (~5 µs standalone class, search-verified).
14. Carried primaries, verified at their wave and re-cited: CUDA PDL programming guide (13.4.2);
    TRT-LLM PDL casebook + PR #9695 discipline; Event Tensor (arXiv 2604.13327, 5-10 µs boundaries);
    eunomia CPU-noise study (2026-05-24); CUDA toolkit release notes (no batched VMM map);
    vAttention (arXiv 2405.04437).

Totals: 14 external sources this session (4 fetched in full) + 13 carried primaries + the local
corpus (28 code/config files, 1 live-process environ read, 1 engine log with 17,234 measured cohort
spans and 8 TREND lines, 5 research dossiers, 25+ F-entries).

## METHODS NOTE (so nothing is re-purchased)

Productive: reading the LIVE process's /proc environ before trusting any launcher claim (this is
what caught the PDL default flip — the boot line alone could have been a stale section); the
`grep "engine started" | tail -1` discipline (F-277's lesson) applied before every log read; the
dm_fold guard read at :2201-2202 which simultaneously resolved the LFUSE question and poisoned the
VERIFY_PROFILE re-baseline plan BEFORE a boot was spent on it; computing the cohort-span
distribution directly from all 17,234 log lines instead of citing the single C5 histogram (the two
agree — p50 24.3 vs 23.8 — which is itself a validation of the C5 fix).
Unproductive / do-not-repeat: F-270 contains no VRAM ledger — do not cite one (the tasking's
attribution was wrong; the T=8 sizing question resolves in generate.cpp's init calls, not in any
journal table); `WSTAT` does not appear in campaign/results t38 logs (engine stderr for those boots
was not captured — the live log is the only WSTAT source; future battery boots should tee engine
stderr per the campaign law's tee rule); launch-spec8.sh/launch-olds7.sh no longer exist at repo
root (F-267 deleted them; git history is the archive — do not search the working tree); the t38 .out
files are server-console level only (no TREND/WSTAT content).
Estimates are marked [HYPOTHESIS]; every other number traces to a file:line, a log line, a banked
F-entry, or a fetched source.
