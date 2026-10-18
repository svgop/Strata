# T17E — MoE expert-serving and memory-tiering leaps for the standing regime (2026-10-17)

Topic: what a 2025-2026-grade MoE serving layer would do differently on this box
(RTX 5090 32GB sm_120 + 9950X3D + 184GB DDR5, PCIe 5, no NVLink), given the standing
custom-2 expert pipeline: 48x512 = 24,576 (layer,expert) pairs, avg blob 2.05MB, VRAM
expert cache ~9,337 slots / 17.71GiB = 38% of pairs -> 97.7% hit rate, VRAM-hits line at
87-100% of the GDDR7 roofline (T12-MEM/T12-GPU, SVG-CUSTOM-TODO.md:909-953), standing-warm
pool exposure 0.3-0.7 ms/window (SVG-CUSTOM-TODO.md:973).

Method: read-only code study of the expert pipeline (expert_source.cpp/hpp, native_moe.cu,
pool.cpp, verify.cpp plan/doorbell path, generate.cpp adaptive tier + profiles), then a
45-source external sweep (arXiv 2023-2026, OSDI/SOSP/DAC/ICLR/ICSME/IPDPS papers, SGLang /
vLLM / llama.cpp / KTransformers / TensorRT-LLM engineering, NVIDIA CUDA docs). Every local
claim carries file:line; every external claim carries a numbered SOURCES entry.

---

## 1. EXECUTIVE SUMMARY

1. **The 2024-2026 expert-serving literature attacks a regime we have already exited.**
   Fiddler, MoE-Infinity, Mixtral-offloading, HybriMoE, Pre-gated MoE, Fate, ProMoE,
   AdapMoE and the whole edge-offload class buy their 1.3-19x in the 30-90% hit-rate /
   PCIe-or-NVMe-bound regime. Our standing regime is 97.7% VRAM hits with the hit line at
   the roofline (T12-MEM #1). Re-priced against our own accounting, the TOTAL recoverable
   by eliminating every remaining miss term is the pool exposure + PCIe share
   ~= 0.5-0.9 ms of an 11.9-12.3 ms window ~= 4-7% ceiling, of which the actually-exposed
   part is 0.3-0.7 ms (~+1-3% M1). **No published serving mechanism is a leap here.**

2. **Everything the literature sells, we already implement or have banked a verdict on:**
   cross-layer lookahead = RouterLookahead (expert_source.cpp:2351-2472, submit per layer at
   :3704-3710); expert-activation-trace-driven residency = the learned profile + adaptive
   tier (generate.cpp:8880-8889, 9019-9068); prefetch-during-compute = the cohort reactor
   (F-233) which is the only structural place a prefetch can hide; PCIe fetch batching =
   STRATA_DMA_BATCH (verify.cpp:2614-2625, standing); pinned-RAM tier = the resident
   complement (custom-iq3_s-8081.json:55; expert_source.hpp:559-608); profile re-save by
   traffic = F-178 / --expert-profile-save (custom-iq3_s-8081.json:51-54).

3. **New externally-grounded results this dossier banks:**
   - (a) The strongest external negative: arXiv 2608.12103 (kernel-managed expert caching,
     Aug 2026) measures that router-lookahead advice is worth only 0.3% at 64.7% recall
     (perfect one-layer advice = 5.0%) **on a page-cache path we do not even use** — our
     RAM tier is pinned. This kills the "make the lookahead prefetch harder" lane
     quantitatively, not just by argument. The same paper shows the pinned oracle arena is
     1.09-1.11x FASTER than the page-cache hit/reclaim path — independent validation of the
     standing --resident-experts pinned-complement design.
   - (b) One new integration candidate (C1): prediction-driven VRAM staging executed by the
     cohort reactor in the other cohort's shadow, consumed through the EXISTING
     ForesightSwap::take() hook in the plan path (expert_source.cpp:3899-3902) — which is
     host-plan-only, so it composes with device-plan E-6 instead of fighting it. Priced
     +0.5-2%, M effort, with a ~free falsifier (STRATA_LOOKAHEAD_STATS=1 boot — never run).
   - (c) One new policy candidate (C4): profile domain-shift guard. 2608.12103's central
     finding is that a STATIC oracle profile degrades under domain shift while recency
     stays stable; model knowledge belongs at ADMISSION. Our PINCAP=48 pins profile ranks
     against eviction (generate.cpp:8859-8876) — exactly the mechanism that paper warns
     about. DORMANT=64 (generate.cpp:8866, 9046-9051) already mitigates; the residual is a
     divergence-triggered un-pin / re-save cadence. Variance-class, S effort.
   - (d) A verified-absent report on the big-engine state: vLLM MoE offload is STILL
     RFC-only (#38256, Mar 2026), SGLang expert-granular CPU offload is an open feature
     issue (#14233) and its per-token expert recorder is flagged too slow for production
     (#18587). Nothing to port; the "port vLLM/SGLang offload ideas" door (T7.C) stays shut
     and is now confirmed closed as of 2026-03/06.
   - (e) A verdict ledger (section 5) that saves batteries: draft-gated prefetch,
     compressed-experts-in-RAM with decompress-on-GPU, UVM/cudaMemPrefetchAsync, cacheless
     on-demand loading, kernel page-cache ownership, and AMX-class CPU expert compute are
     all DOESN'T-APPLY with mechanisms, not vibes.

4. **The throughput leaps do not live in expert serving.** At 97.7% hits and roofline-done
   bytes, the remaining leap energy is where the T16 queue already put it: drafter
   acceptance (FastMTP retrain), the decode kernel tranche, and +slots-via-bytes (R22,
   banked — this sweep found three fresh 2025-2026 corroborations for it, section 4.3,
   which strengthen its case without re-pricing it).

---

## 2. THE STANDING PIPELINE AS READ (the anatomy the candidates must fit)

Per verify window (~6 rows, ~48 layer-groups, ~2,554 routed entries, ~2,495 VRAM hits,
~59 misses — T12-MEM numbers, SVG-CUSTOM-TODO.md:914, 971-973):

- **Route + residency bias.** Router runs on GPU; STRATA_ROUTE_RESIDENT=0.75 (F-200) may
  re-pick ranks 6-9 toward resident experts (verify.cpp:1550-1556); MTP router bias likewise
  (generate.cpp:7869-7878).
- **Per layer-group, pre():** device-side `resident_plan` for all-resident groups (E-6,
  verify.cpp:1558-1573, standing STRATA_VERIFY_DEVICE_PLAN=1); doorbell publish writes the
  normed activations + ids (+weights) into mapped host memory and stores the ring value
  (elementwise.cu:355-367; call sites verify.cpp:1577-1590); shared expert + q8_1 quantize
  (verify.cpp:1597-1635).
- **Host service half:** the pool consumes the doorbell, then `expert_pool_dispatch_multi`
  (expert_source.cpp:3694) builds the plan in ONE pass: kind[] per routed entry
  (-2 tail-skip / -1 CPU miss / 0 VRAM hit / 1 PCIe / 2 peer), VRAM groups + PCIe groups
  (:3734-3943), publishes via flagA (seq-cst fence + store, :3944-3946 ->
  verify.cpp:2632-2636), and starts the copy-engine fetch of the pcie_num/256 tail misses
  (:3869-3875, :3948). The PCIe uploads are ONE batched cudaMemcpyBatchAsync under the
  standing STRATA_DMA_BATCH=1 (verify.cpp:2604-2630). CPU misses are quantized, gathered
  into jobs, and computed by the 15-worker pool (pool.cpp run_split_multi; POOL_DIET drops
  the trailing repark wait, pool.cpp:759; park/spin via STRATA_POOL_SPIN_US=100,
  pool.cpp:478, 533-594; batch claim epoch protocol pool.cpp:609-660).
- **Per layer-group, post():** GPU waits flagA -> grouped hit kernel reads VRAM slots
  (verify.cpp:1683-1697), waits flagB -> PCIe-share kernel (:1698-1709), waits flag_ ->
  copies the CPU rows from mapped memory (:1711-1730), then the combine (hits folded in
  under FUSE_HITS, F-226; native_moe.cu:98-140 combine_k10_vec4_hits).
- **Residency management (between rounds):** static preload from the learned profile
  (--expert-profile, custom-iq3_s-8081.json:49-50; read_expert_profile), adaptive tier
  swaps driven by decayed usage `--adapt-decay 0.85` (:47-48): candidates need u>=2.0
  measured (:9027) or the bounded prediction-credit lane (STRATA_PREDICT_CREDIT=1.0 /
  PREDICT_BUDGET=2; credit requires u>=0.5 AND pc>=1.5, :9030-9042), victims = lowest usage
  minus PINCAP-pinned profile ranks, minus MIN_AGE=2-rounds-young, plus DORMANT=64
  never-routed evictables (:8864-8866, 9046-9054). Swaps are blob-exact H2D copies on
  adapt_stream (batched under DMA_BATCH, :9110-9171), pending until adapt_ev, applied by
  apply_pending with non-blocking polls (:8978-9001; call sites :10468, :10827, :13162).
  The learned profile is re-saved on a 30-min cadence (custom-iq3_s-8081.json:52-54; heat =
  cumulative non-decayed counter, generate.cpp:8880-8889, save lambda :9626) — the F-178
  mechanism.
- **Lookahead:** a CPU thread applies NEXT layer's router (BF16 dots, up to 8 tokens) to
  the current layer's MoE input, drops GPU-resident experts, and (i) warms the FILE tier's
  pages (madvise WILLNEED — no-op in the standing pinned-RAM regime since decode never
  touches files: warms() gate expert_source.hpp:725, warm :721-723), (ii) feeds
  ForesightSwap::predict — **structurally dead**: foresight_swap_from_env refuses to init
  beside STRATA_VERIFY_DEVICE_PLAN (expert_source.cpp:5259-5263; T13 ledger,
  SVG-CUSTOM-TODO.md:975), (iii) banks softmax-shared fractional credits drained per adapt
  round (take_credits expert_source.cpp:5009-5013; drained generate.cpp:9010-9012).
  STRATA_LOOKAHEAD_STATS has a full scorer (recall/precision at k, k+4, k+10; vote top-4/8/12
  coverage of the PCIe share, expert_source.cpp:2256-2349) that has NEVER been banked in the
  journal — that is the cheapest falsifier in this dossier.
- **Cohorts:** STRATA_COHORTS=2 (F-233) runs two verifiers on one GPU; they share weights,
  expert cache and slot sessions; each owns buffers/graphs/doorbells
  (generate.cpp:8106-8110). The reactor launches both windows then polls; B's CPU pool work
  runs under A's GPU window (:10187-10190) — the F-223 theorem's only escape, and therefore
  the only structural place an expert prefetch/refill can hide for free.

---

## 3. RANKED CANDIDATE TABLE

| # | Candidate | Mechanism | Priced magnitude | Effort | Cheapest falsifier | Sources |
|---|---|---|---|---|---|---|
| 1 | **C1: prediction-driven VRAM staging in the reactor's shadow** | Reactor enqueues RouterLookahead's predicted next-layer misses for H2D copies into a Foresight-class staging arena while the OTHER cohort's window runs; host-planned groups consume landed slots via the EXISTING `d.fs->take()` hook (expert_source.cpp:3899-3902). ForesightSwap is dead only because foresight_swap_from_env refuses DEVICE_PLAN at INIT (:5259-5263); the take() hook itself is in the host-plan path, so an init-time gate keyed to "host-planned groups only" composes with E-6 device plans. | +0.5-2% M1 (pool exposure 0.3-0.7 ms is the whole prize; conversion is partial because the pool is already mostly hidden); conc tail flattening similar | M-L | STRATA_LOOKAHEAD_STATS=1 boot: if lookahead recall@10 on standing traffic is below the ~54-67% DuoServe exact-top-k bar, the candidate dies before any code [S12][S5][S6] | [5][6][8][11][12][16] |
| 2 | **C4: profile domain-shift guard (hybrid admission)** | 2608.12103: static profiles degrade under domain shift, recency does not; model knowledge belongs at admission, not eviction. Ours: PINCAP=48 pins profile ranks (generate.cpp:8859-8876) — the at-risk mechanism; DORMANT=64 already exempts never-routed pins (:9046-9051). New: divergence-triggered un-pin (per-round decayed-usage-vs-profile-rank disagreement scales PINCAP or pulls the re-save cadence forward — the F-178 re-save made traffic-driven). | Variance class: protects the conc band (F-183: +-500 slots = +-4.9% conc band, SVG-CUSTOM.md:5963); mean ~0-1% | S-M | One battery on a deliberately off-domain prompt set (drift repro) vs standing; then the guard same-boot | [8][9][1][3][26][28] |
| 3 | **C5 rider: adaptive-tier admission budget sweep** | The credit lane already caps promotions at PREDICT_BUDGET=2/round (generate.cpp:9013-9015); HybriMoE/ExpertCache-RL argue admission control over eviction policy in miss-bound regimes; our 2.3%-miss regime makes this a rider, not a lane. Sweep PREDICT_BUDGET {1,2,4} x MEASURED_OK floor {1.5,2.0,3.0} in one boot pair. | 0-1%, conc-tail class | S (env-only) | The sweep itself | [9][29][26] |
| 4 | **R22 rare-expert mixed precision — EXTERNAL CORROBORATION ONLY (banked, do not re-price)** | Rare/cold experts at fewer bits buy slots. Fresh 2025-26 evidence: hot-FP8/cold-INT4 expert quantization [30], bit-sliced experts under miss-rate constraints [13], importance-driven caching/substitution [31], sensitivity-gated AdapMoE [17]. Chain here: +GiB -> +512 slots -> +2.1pt hits (SVG-CUSTOM-TODO.md:913) -> miss-term shrink. | Banked | — | — | [13][30][31][17] |
| 5 | **NEGATIVE RESULT (banked, saves work): lookahead-page-warming and harder prefetch predictors** | 2608.12103 measured router-lookahead readahead at 0.3% (64.7% recall) and perfect one-layer advice at 5.0% — on a page-cache consumer. Our RAM tier is pinned and decode never touches the file tier (warms() gate off: FileExpertSource::warms(), expert_source.hpp:725 — ArenaExpertSource warms() is the base no-op). Prediction accuracy is NOT the binding constraint; the consumer is. | Kills trained-predictor imports (Fate/SpecPrefetch/Pre-attention class) without batteries | — | Already falsified externally; local stats boot only if C1 is pursued | [8][4][5][6][16] |

---

## 4. DEEP-DIVES

### 4.1 What a 2025-2026 serving layer does differently — item by item against our code

**Prefetch during verify windows.** The literature's version (Mixtral-offloading's
speculative expert loading [7], ProMoE's proactive fetch from intermediate router state [6],
AdapMoE prefetch+gating [17], HybriMoE's prefetch+admission cache [9]) hides H2D transfer
behind the CURRENT layer's compute. In our window the equivalent consumer is the pool's
flag_ wait (verify.cpp:1714-1730) and the pool exposure. Three structural facts re-price it:
(i) the transfers to hide are ~59 blobs x 2.05MB/window, mostly already hidden behind 15
pool workers (standing-warm exposure 0.3-0.7 ms, SVG-CUSTOM-TODO.md:973); (ii) the old
ForesightSwap did exactly this and is refused beside DEVICE_PLAN
(expert_source.cpp:5259-5263); (iii) the ADAPT_OVERLAP machinery (copies landing while the
next window runs) is refused under THREE standing arms at once — DMA_BATCH, ADAPT_LAG=2,
and the device-plan adapt gate (generate.cpp:8909-8924, :8930). The one clean integration
left is C1: the cohort reactor's inter-window gap + the other cohort's window as the hiding
place, feeding the host-plan take() hook. Note the take() hook only serves HOST-planned
groups; E-6 all-resident groups (planned on device) never need it — by construction the
staging only has to exist for miss-bearing groups. This is why C1 does not collide with the
standing device plan even though ForesightSwap-the-env did.

**Draft-gated prefetch.** DOESN'T APPLY (verdict, not laziness): the verify window's routed
ids are a per-layer function of embeddings the window itself produces; the MTP drafter's
output gives next-token identities, not layer-0 hidden states, so "gate the prefetch on the
draft" degenerates to predicting layer 0 from the drafter's hidden state — strictly worse
than the RouterLookahead's one-layer-exact predictor, which is already running and already
under-powered for this regime (4.3). No paper found that does better than one-layer gate
prediction without training (Fate's 97.15% IS the one-layer adjacent-gate input, same as
ours) [5][6][16].

**Cross-layer lookahead (pre-gated MoE style).** IMPLEMENTED (expert_source.cpp:2201-2472,
submitted per layer at :3704-3710). Pre-gated MoE [3] adds a trained pre-gate and reports
prefetch accuracy as the win in offload regimes; Fate [5] shows adjacent-layer gate inputs
hit 97.15% prefetch accuracy training-free — the same information our lookahead uses. The
decay chain in OUR regime: 97.15% prediction accuracy x (miss share already ~2.3%) x
(hideable fraction of an already-hidden 0.3-0.7 ms) ~= tenths of a percent. The free
falsifier exists and has never been run: STRATA_LOOKAHEAD_STATS=1 prints recall/precision
at k=10/14/20 and vote coverage of the PCIe share (expert_source.cpp:2324-2348).

**Batching PCIe fetches.** DONE. The per-window miss uploads already go out as one
cudaMemcpyBatchAsync (STRATA_DMA_BATCH=1 standing; verify.cpp:2610-2625; the launcher's
"84.9% fewer DMA submit cycles" note, launch-custom-iq3s-8081.sh:17). RMM independently
confirms batched memcpy has lower submission overhead and less driver lock contention [42].

**Compressed expert formats in RAM, decompress-on-GPU.** DOESN'T APPLY twice over: (a) the
CPU pool is PARALLELISM-limited, not bandwidth-limited — 17-38 GB/s aggregate against
~60-70 GB/s real DRAM (T11.R4(a), SVG-CUSTOM-TODO.md:832-833), so compressed RAM blobs buy
nothing and cost a dequant; (b) the PCIe share is ~0.3-0.4% of the window (T12-MEM #6), so
transfer compression attacks a rounding error. Note the blobs are ALREADY the GPU-native
IQ3_S family — "decompress-on-GPU" would add work to serve a line with no headroom. The
bit-format direction that DOES pay is VRAM-side: fewer bytes per SLOT (more slots/GiB) —
that is R22, banked, now triple-corroborated externally [13][30][31].

**UVM / pinned-RAM policy.** Standing wins. CUDA managed memory migrates at PAGE
granularity with a fault-driven path (cudaMemPrefetchAsync is page-level advice) [40]; our
swaps are blob-exact pinned H2D copies into named slots and our RAM tier is a registered
pinned arena read directly by both the CPU pool and (for the PCIe share) the GPU. On a
PCIe-only consumer part with no NVLink there is no coherent-access upside to UVM, only page
faults. The kernel-vs-user-space question is now measured: 2608.12103 finds the pinned
oracle arena 1.09-1.11x FASTER than the page-cache hit/reclaim path, with recency's only
advantage being robustness under domain shift — an admission/policy matter, not a storage
matter [8]. Our design already sits on the faster side and keeps model knowledge at
admission (profile preload + adaptive tier).

**Profile lifetime management.** The live candidate (C4). External state of the art has
moved from static profiling (MoE-Infinity's activation traces [2]) through decayed
frequency (our --adapt-decay; tqllm's decayed-frequency, banked T1.3) to learned policies
(ExpertCache-RL [29]) and to the 2026 position: recency for eviction, model knowledge for
admission, with static-oracle profiles degrading under domain shift [8]. Ours is already
the hybrid (profile preload at admission + decayed-usage adaptation + PINCAP + MIN_AGE +
DORMANT). The un-banked residual: the PINCAP pinning is unconditional on domain — the
divergence guard is the new content, plus tying the 30-min profile re-save cadence
(custom-iq3_s-8081.json:53-54; F-178 identified the re-save-by-traffic mechanism) to
measured divergence rather than wall time. Expectations must be honest: this is band/
variance protection (F-183's +-4.9% conc band per +-500 slots), not mean throughput.

### 4.2 The engine-side external state, verified 2026-10 (nothing to port)

- vLLM: MoE expert offload = RFC #38256 (Mar 2026), expert weights pinned in host RAM +
  GPU cache — the design we already run; nothing merged; forum threads confirm no dynamic
  offload in releases [21][22][23]. T7.C's "nothing merged; RFC-only" is CONFIRMED still
  true as of 2026-03.
- SGLang: expert-granular CPU offloading is feature issue #14233 (open) [18]; the expert
  distribution recorder's per-token mode is flagged production-infeasible on overhead
  (#18587, Feb 2026) [19] — a caution for any always-on routing telemetry; our usage
  counting is 48x60 adds/window (expert_source.cpp:3731-3733), far below that class.
  Qwen3.8-Flash-Next day-0 support exists (Aug 2026) [20] but its serving wins are
  multi-GPU/batch features (already harvested: IndexShare S10).
- llama.cpp: the two-tier GPU+RAM expert cache thread (#20757, Mar 2026) and the slot-pool
  eviction-policy RFC (#28248, SLRU/probation) are the community converging on the
  decayed-frequency + admission design we already run [24][26][28]; #24528's ADMIT_AFTER
  gating is banked (T7.C.1) [27].
- KTransformers: SOSP'25 paper = AMX CPU expert kernels + layout [14][15]; Zen5 has no AMX
  — the platform transfer is zero, and the CPU-pool lane's remaining arithmetic wins are
  already banked (R4#6 VBMI/VNNI, SVG-CUSTOM-TODO.md:846).
- TensorRT-LLM: the DeepSeek-R1/B200 latency blog's wins are MTP layer extension (ours:
  FastMTP, queued) and expert-overlap scheduling on NVLink datacenter parts [35] —
  the overlap pattern is what the cohort reactor already does; the parts do not transfer.

### 4.3 R22 corroboration detail (banked item, evidence refreshed only)

- Hot-FP8/cold-INT4 expert quantization (arXiv 2511.15015, Nov 2025) does frequency-tiered
  bits inside MoE serving — same direction as our banked "rare experts get bits, not hot"
  [30].
- Bit-sliced expert caching under miss-rate constraints (arXiv 2512.12990) treats expert
  BITS as a cacheable, sliceable resource under a miss-rate budget — the formal version of
  the slots-vs-bits trade T12-MEM quantified (+512 slots per GiB = +2.1pt hits,
  SVG-CUSTOM-TODO.md:913) [13].
- Importance-driven expert caching (arXiv 2508.18983) adds substitution of low-importance
  active experts — the LOSSY class our exact-match verify cannot absorb on the serve path
  (same verdict family as HOBBIT/MoE-Spec, T9.B traps, SVG-CUSTOM-TODO.md:664) [31].

### 4.4 The negative-results ledger (mechanisms priced DOESN'T-APPLY, with why)

| Mechanism (source) | Verdict | Why on this box |
|---|---|---|
| Draft-gated prefetch (no publication found doing better than one-layer gate prediction) | DOESN'T | Verify-window routing is per-layer; drafter gives tokens, not layer-0 states; one-layer predictor already runs [5][6] |
| Trained prefetch predictors (SpecPrefetch adapter [16], Pre-attention prediction [37]) | DOESN'T | Prediction value collapses to the miss term at 97.7% hits; 2608.12103 measured 0.3% at 64.7% recall even with a fault-paying consumer [8] |
| Cacheless / on-demand expert loading (OD-MoE [32]) | DOESN'T | Opposite regime; we are cache-max at the VRAM ceiling; disk tier closed at -70% (T7.C) |
| Kernel page-cache ownership of the expert tier (2608.12103) | DOESN'T (for us) | Their own numbers: pinned arena 1.09-1.11x faster; recency's edge is domain-shift robustness = policy, banked as C4 [8] |
| Substitution / low-importance expert swap (2508.18983 [31], AdapMoE gating [17]) | DOESN'T on verify path | Output-changing under exact-match verify; contract breach (T9.B trap class) |
| UVM / cudaMemPrefetchAsync for experts [40][41] | DOESN'T | Page-granular migration + fault path vs blob-exact pinned swaps; no NVLink coherence upside on PCIe-only consumer |
| Compressed-in-RAM experts + GPU decompress | DOESN'T | Pool parallelism-limited (17-38 vs 60-70 GB/s, T11.R4(a)); PCIe share 0.3-0.4% (T12-MEM #6) |
| AMX-style CPU expert kernels (KTransformers [14][15]) | DOESN'T | Zen5/AM5 has no AMX; remaining CPU-matvec arithmetic already banked (R4#6) |
| Multi-GPU EP/Wide-EP serving patterns (vLLM Wide-EP [37->38], TRT-LLM EP [36], LMSYS EP [39]) | DOESN'T | Single-GPU box; no NVLink; PCIe 5 peer tier dead (F-223-era) |
| Datacenter MoE overlap engineering details (DeepSeek day-6 [34], TRT-LLM B200 blog [35]) | PARTIAL | The overlap principle = our cohort reactor (F-233, adopted); their mechanisms (RDMA/EP) have no local target |

---

## 5. CONTRADICTIONS AND THEIR RESOLUTION

1. **"Sophisticated cache policies win" (MoE-Infinity [2], HybriMoE [9], ExpertCache-RL
   [29]) vs "policy choice barely matters" (2608.12103 [8]).** Resolved by regime: the
   first group measures 30-90% hit-rate caches where each point of hit rate is throughput;
   2608.12103 measures near-saturated caches where eviction policy is worth ~nothing but
   admission (and domain-shift robustness) still is. We are in the second regime; the
   actionable residue is C4 (admission/robustness), not a new eviction formula.
2. **Pre-gated/Fate ~97% prefetch accuracy [3][5] vs 2608.12103's 64.7% measured recall /
   0.3% gain [8].** Resolved: accuracy is measured on the predictor's own distribution;
   value is miss-term x consumer-sensitivity. A page-cache consumer pays faults and still
   gains 0.3%; our pinned consumer pays nothing at all to warm. Prediction-import races
   are therefore NOT a lane here, and C1's ceiling inherits this cap.
3. **Fiddler "CPU compute beats PCIe transfer" [1] vs our banked rejection of CPU-pool
   numerics changes (T1.x SKIP of tqllm's CPU_MISS; T11.R2 rejection of deferred experts).**
   No real conflict: Fiddler's box has NO pool and accepts CPU-GPU bit differences; our
   contract (canaries, M3b, bit-exact A/B law) does not. We already run the pool; the
   remaining pool win was scheduling, adopted as POOL_DIET (F-227).
4. **"Every extra GiB = +2.1pt hits" (T12-MEM) vs F-183's vram-800 arm (+513 slots) reading
   conc +4.9% but M1 -6.9%.** Resolved in F-183 itself: the reserve change moved more than
   slots (KV/VRAM pressure); the slot term is real on conc (wider union) and ~zero on M1
   (already ~99% there). Any C1/C4 pricing must be per-lane, not aggregate.

## 6. OPEN QUESTIONS

- **RouterLookahead recall on standing traffic is UNMEASURED.** STRATA_LOOKAHEAD_STATS=1
  exists (expert_source.cpp:2256-2349) and no journal entry banks its output. One boot
  answers whether C1 survives its own falsifier (compare against DuoServe's 54-67%
  exact-top-k bar [12] and Fate's 97.15% one-layer bar [5]).
- Whether the cohort reactor loop already polls apply_pending(false) between windows or
  only at round boundaries (call sites :10468/:10827/:13162 exist on the batch paths; the
  exact cohort-loop cadence at :10187-10512 was not traced to closure). C1's hiding place
  depends on it.
- The marginal hit-probability of profile ranks beyond slot ~9,337 today (the +2.1pt/GiB
  figure is the historical marginal; ROUTE_RESIDENT=0.75 reshapes the traffic the profile
  was learned on). A /metrics dump of per-slot hit counts would answer it without a battery.
- Whether F-178's evening drift was ever split into "profile pollution by probe traffic"
  vs ambient state (the re-anchor mechanism note, SVG-CUSTOM.md:5596-5599, left it a
  two-hypothesis open). C4's guard would close it as a side effect.

## 7. IMPLICATIONS FOR THE CAMPAIGN

- **Do not spend batteries on prefetch/residency-policy candidates beyond C1's falsifier
  and C4's guard.** The external sweep's center of mass (45 sources) confirms the standing
  architecture is at or past the 2026 serving frontier for this regime; the wins the
  papers report do not survive the 97.7%-hit translation.
- **C1 is the only new code candidate worth its falsifier**, because the falsifier is a
  single stats boot and its integration surface (take() hook + reactor enqueue) is small;
  its ceiling is honest: +0.5-2%.
- **C4 is env/logic-class** and protects the 262k-switch decision (T16 #6): a -7.5% slot
  budget landing on a domain-shifted profile is exactly the scenario 2608.12103 warns
  about; the guard makes the switch safer.
- **The leap energy stays in T16's lanes** (FastMTP retrain, kernel tranche, R22 when
  quality headroom is needed). This dossier's contribution to those lanes is the fresh
  R22 corroboration and the verified absence of any external shortcut.
- Suggested INDEX line: `| 2026-10-17 | T17E-moe-serving-leaps.md | 2024-26 MoE serving-leap sweep vs standing 97.7%-hit pipeline | no external leap survives the regime; C1 lookahead-staging (falsifier = LOOKAHEAD_STATS boot), C4 profile-drift guard; UVM/compressed-RAM/draft-gated prefetch/page-cache-ownership priced DOESN'T | current |`

## SOURCES (45)

1. Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models (Kamahori et al., ICLR'25) — https://arxiv.org/abs/2402.07033 — first-N-experts-on-GPU-rest-on-CPU with a no-pool baseline. Verdict: PARTIAL — we already run a 15-worker CPU pool; their regime is our Phase-2 h=0 past; the CPU-numerics freedom they use violates our canary contract.
2. MoE-Infinity: Efficient MoE Inference on Personal Machines (Xue et al., OSDI'24) — https://arxiv.org/abs/2401.14361 — sequence-level expert-activation traces drive cache+prefetch groups. Verdict: PARTIAL — trace-driven residency = our learned profile + adaptive tier; prefetch pays only in their 30-70% hit regime, not at 97.7%.
3. MoE-Infinity repo — https://github.com/efficientmoe/moe-infinity — engineering detail of EAM traces. Verdict: DOESN'T (as a port; PyTorch stack, batch-offload class).
4. Pre-gated MoE (Hwang et al., ICML'25) — https://arxiv.org/abs/2308.12066 — trained pre-gate predicts next-block experts for prefetch. Verdict: PARTIAL — our RouterLookahead is the training-free form; value capped by the miss term (see [11]).
5. Fate: Fast Edge Inference via Accurate Expert Predictions (Fang et al., 2025) — https://arxiv.org/abs/2502.12224 — adjacent-layer gate inputs give 97.15% one-layer prefetch accuracy, no GPU overhead. Verdict: PARTIAL — same mechanism class our lookahead implements; accuracy is not our binding constraint.
6. ProMoE: Fast MoE-based LLM Serving using Proactive Caching (2024) — https://arxiv.org/abs/2410.22134 — predicts next-layer/next-token usage from intermediate results and fetches early. Verdict: PARTIAL — already in T9.B#3; at 2.3% misses the proactive fetch races a hidden pool.
7. Fast Inference of MoE Language Models with Offloading (Eliseev & Mazur, 2023) — https://arxiv.org/abs/2312.17238 — LRU cache + speculative expert prefetch during current-layer compute. Verdict: PARTIAL/DOESN'T — canonical prefetch-during-compute; our misses are already hidden; their 2-3 tok/s baseline class is far below standing.
8. Who Should Own the Expert Cache? Kernel-Managed Tiering for Trillion-Parameter MoE Inference (Si et al., Aug 2026) — https://arxiv.org/abs/2608.12103 — page cache vs user-space pinned arena; oracle-pinned arena 1.09-1.11x faster than page-cache path; router-lookahead readahead worth 0.3% at 64.7% recall, perfect one-layer advice 5.0%; static oracle degrades under domain shift while recency stays stable. Verdict: APPLIES — the sweep's strongest calibration: validates our pinned complement, caps all prefetch/policy expectations, and motivates C4.
9. HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management (Zhong et al., DAC'25) — https://arxiv.org/abs/2504.05897 — dynamic intra-layer CPU/GPU load balance + admission-gated caching to cut swaps; 1.3-3.2x vs offload baselines. Verdict: PARTIAL — their swap-bound regime is behind us; the admission-control principle feeds C4/C5.
10. HybriMoE repo — https://github.com/PKU-SEC-Lab/HybriMoE — implementation reference. Verdict: DOESN'T (nothing portable past the policy ideas).
11. MoE-Infinity SRC-W'24 abstract — https://uksystems.org/workshop/2024/papers/srcw24-paper13.pdf — imminent-activation detection to fetch only needed experts. Verdict: PARTIAL (same as [2]).
12. DuoServe: Dual-Stream System for MoE inference (2025) — https://arxiv.org/abs/2509.07379 — next-layer expert predictor 54-67% exact top-k, 90-95% >=1-hit; graph-captured DMA slot table; 1.78-7.04x on weaker HW. Verdict: PARTIAL — banked T7.C.2; its predictor bar is C1's falsifier yardstick; its graph-DMA idea remains the bigger (unpriced-here) sibling.
13. Bit-Sliced Expert Caching under Miss-Rate Constraints (2025/26) — https://arxiv.org/abs/2512.12990 — expert bits as a sliceable cache resource under a miss-rate budget. Verdict: APPLIES as external corroboration for banked R22 (bits-vs-slots trade).
14. KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models (Chen et al., SOSP'25) — https://dl.acm.org/doi/10.1145/3731569.3764843 — AMX expert kernels, tiling-aware layout. Verdict: DOESN'T — no AMX on Zen5/AM5; our pool lane's arithmetic alternatives banked (R4#6).
15. KTransformers repo — https://github.com/kvcache-ai/ktransformers — framework state. Verdict: DOESN'T (platform).
16. SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE (Jun 2026) — https://arxiv.org/abs/2607.24787 — trained shared-adapter prefetch predictor, router-preserving. Verdict: DOESN'T — requires training we do not have; prediction value capped by [8]'s measurement.
17. AdapMoE: Adaptive Sensitivity-based Expert Gating and Management (2024) — https://arxiv.org/abs/2408.10284 — sensitivity-gated expert selection + prefetch. Verdict: DOESN'T on the verify path (output-changing gating); PARTIAL for the prefetch scheduling idea.
18. SGLang feature issue: expert-granular CPU offloading (#14233, Dec 2025) — https://github.com/sgl-project/sglang/issues/14233 — open feature, no merged port target. Verdict: APPLIES as state-of-engine evidence (nothing to port).
19. SGLang issue: expert distribution recorder performance (#18587, Feb 2026) — https://github.com/sgl-project/sglang/issues/18587 — per-token expert telemetry too slow for production. Verdict: APPLIES as a caution — keep routing counters window-aggregated (ours are: expert_source.cpp:3731-3733).
20. SGLang day-0 support for Qwen3.8-Flash-Next (LMSYS, Aug 26 2026) — https://www.lmsys.org/blog/2026-08-26-qwen-flash-next — our exact model class in SGLang. Verdict: PARTIAL — features are multi-GPU/batch serving class; IndexShare already harvested (T7.A.1).
21. vLLM RFC #38256: Incremental MoE Expert Offloading — GPU Cache + pinned host RAM (Mar 2026) — https://github.com/vllm-project/vllm/issues/38256 — design mirrors our standing architecture. Verdict: DOESN'T — nothing merged; confirms the T7.C "RFC-only" verdict as of 2026-03.
22. vLLM discuss: Enable Expert Offloading — https://discuss.vllm.ai/t/enable-expert-offloading/1884 — no dynamic expert offload in releases. Verdict: DOESN'T.
23. vLLM discuss: Expert offloading — https://discuss.vllm.ai/t/expert-offloading/1880 — same. Verdict: DOESN'T.
24. llama.cpp issue #20757: Two-tier GPU+RAM expert cache for MoE offload (Mar 2026) — https://github.com/ggml-org/llama.cpp/issues/20757 — community two-tier designs, LFU-cold misses. Verdict: PARTIAL — converges on our design; no mechanism we lack.
25. llama.cpp discussion #24528: MoE expert cache, VRAM caching of hot CPU-resident experts — https://github.com/ggml-org/llama.cpp/discussions/24528 — admission gating (ADMIT_AFTER class). Verdict: PARTIAL — banked (T7.C.1).
26. llama.cpp discussion #28248: Persistent expert slot pool RFC (SLRU/probation policies) — https://github.com/ggml-org/llama.cpp/discussions/28248 — eviction-policy zoo; ~90% of activation mass in few experts. Verdict: PARTIAL — policy class capped by [8].
27. LMCache new architecture blog (Apr 2026) — https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x — 10x MoE serving claims. Verdict: DOESN'T — multi-GPU batch-serving class, KV-tier focus.
28. ExpertCache: GPU-Efficient MoE Inference through Reinforcement Learning (ICSME'25) — https://orbilu.uni.lu/bitstream/10993/65750/1/ICSME25-ExpertCache.pdf — RL two-phase expert caching. Verdict: DOESN'T/PARTIAL — learned policy overhead vs our decayed-frequency+profile in a regime where policy is worth ~0 ([8]).
29. HybriMoE ACM DL record (paywalled anchor; [9] carries the free arXiv) — https://dl.acm.org/doi/10.1109/DAC63849.2025.11133274 — publication record for [9]. Verdict: (same as [9]).
30. Dynamic expert quantization: hot FP8 / cold INT4 (Nov 2025) — https://arxiv.org/abs/2511.15015 — frequency-tiered expert bits inside serving. Verdict: APPLIES as R22 corroboration (banked).
31. Enabling MoE on the Edge via Importance-Driven Expert Caching (Aug 2025) — https://arxiv.org/abs/2508.18983 — online prefetch + substitution of low-importance experts; 48% lower decode latency in their regime. Verdict: DOESN'T on verify (lossy substitution); the importance-ranked admission idea corroborates the profile preload.
32. OD-MoE: On-Demand Expert Loading for Cacheless Edge Inference (Dec 2025) — https://arxiv.org/abs/2512.03927 — cacheless fully-on-demand loading. Verdict: DOESN'T — opposite regime; our cache is the asset.
33. SMOE: hybrid CPU/GPU MoE inference on a single consumer GPU (IPDPS 2026) — https://csdl-downloads.ieeecomputer.org/proceedings/ipdps/2026/0602/00/060200b219.pdf — dynamic expert caching consumer-GPU framework. Verdict: PARTIAL — same box class, miss-bound regime; no mechanism past standing.
34. DeepSeek Open Infra day 6: V3/R1 inference system overview (Feb 2025) — https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md — datacenter PD-disaggregation + large-EP engineering. Verdict: DOESN'T — multi-node; the MTP piece is ours already.
35. TensorRT-LLM: Pushing Latency Boundaries — DeepSeek-R1 on B200 (Mar 2026) — https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html — MTP extension + expert-overlap scheduling on NVLink parts. Verdict: PARTIAL — the overlap principle is the adopted cohort reactor; the parts do not transfer.
36. TensorRT-LLM expert parallelism docs — https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html — EP/TP for MoE. Verdict: DOESN'T (single GPU).
37. Pre-Attention Expert Prediction and Prefetching (Nov 2025) — https://www.alphaxiv.org/abs/2511.10676 — lightweight expert prefetch predictor. Verdict: DOESN'T — prediction value capped ([8]); unverified beyond abstract-level.
38. vLLM DeepSeek @ 2.2k tok/s/H200 Wide-EP (Dec 2025) — https://vllm.ai/blog/2025-12-17-large-scale-serving — async scheduling + dual-batch overlap + wide EP. Verdict: DOESN'T — datacenter EP; the async-scheduling principle is the adopted reactor.
39. LMSYS: Deploying DeepSeek with PD Disaggregation and Large-Scale EP (May 2025) — https://www.lmsys.org/blog/2025-05-05-large-scale-ep — production large-EP. Verdict: DOESN'T.
40. CUDA Programming Guide: Unified and System Memory — https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html — cudaMemPrefetchAsync page-granular migration advice. Verdict: DOESN'T — page granularity + fault path vs blob-exact pinned swaps on a PCIe-only part.
41. OLCF CUDA Unified Memory slides — https://www.olcf.ornl.gov/wp-content/uploads/2019/06/06_Managed_Memory.pdf — prefetch-migration vs on-demand faulting costs. Verdict: DOESN'T (supports [40]).
42. RAPIDS RMM issue #2509: use cudaMemcpyBatchAsync — https://github.com/rapidsai/rmm/issues/2509 — batched memcpy lowers submission overhead/driver lock contention. Verdict: APPLIES — already adopted (STRATA_DMA_BATCH standing).
43. CUDA Runtime API memory management (cudaMemcpyBatchAsync et al.) — https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html — the API itself. Verdict: APPLIES (in use).
44. NVIDIA blog: Qwen3-Next hybrid MoE architecture (Sep 15 2025) — https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform — confirms 512 routed + 1 shared, 10 active (our geometry class). Verdict: PARTIAL — model-class facts only.
45. vLLM blog: Qwen3-Next support (Sep 11 2025) — https://vllm.ai/blog/2025-09-11-qwen3-next — 1:50 activation sparsity framing. Verdict: PARTIAL — model-class facts only.

Also consulted (not numbered, index-level): awesome-moe-inference — https://github.com/MoE-Inf/awesome-moe-inference — survey index used to sweep for missed 2025-26 titles; The Ingestion Tax (arXiv 2608.12114) — https://arxiv.org/abs/2608.12114 — file-backed weight ingestion costs (startup-domain; our prewarm.sh owns that lane); Stack Overflow note on cudaMemcpyBatchAsync stream requirements — https://stackoverflow.com/questions/79684961 (practitioner gotcha, consistent with our named-stream use).

## METHODS NOTE (so nothing is repeated)

- Unproductive/dead-end searches: "AdapMoE arXiv 2504.13780" was a WRONG id (a supply-chain paper); the real AdapMoE is 2408.10284 — recorded here to avoid a repeat. "EdgeServe" as a 2025 MoE paper does not exist (the name belongs to a 2019 DL model-caching poster); the intended class is arXiv 2508.18983. The "Bit-Sliced Expert Caching" paper's ID was cited elsewhere as 2602.11192 and could not be verified there; the verified preprint ID found is 2512.12990 — flagging the discrepancy since secondary citations disagree. An ICLR'26 proceedings PDF on "expert offloading suitability" surfaced without a verifiable title and was dropped rather than cited blind.
- arXiv IDs of two campaign-internal citations were re-verified against memory: Mixtral-offloading is 2312.17238 (NOT 2401.04088, which is Mixtral-of-Experts) — the TODO's T7.B/T9 entries cite it correctly by title only; future entries should use 2312.17238.
- The repo was treated as read-only; no builds, boots, API calls, or git operations were run. The only write is this dossier (the INDEX.md update is proposed in section 7 for whoever owns the ledger).
- Confidence grades for the headline findings: (1) regime claim [HIGH — T12-MEM/T12-GPU roofline + 2608.12103 measurement]; (2) C1 pricing [LOW-MEDIUM — arithmetic on banked exposure numbers, no boot]; (3) C4 direction [MEDIUM — single strong external source + F-183 internal]; (4) negative ledger [HIGH for UVM/compressed-RAM/cacheless (mechanism+internal measurements), MEDIUM for draft-gated (absence-of-publication argument)].
