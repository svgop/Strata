# T25-5 — WINDOW & GRAPH ARCHITECTURE: how basalt runs a 6.675 ms window where ours runs ~11.5 ms

2026-10-20. Read-only subsystem study for the custom-2 campaign. Sources: basalt @ `fb0944c7`
(code + docs read directly this session), our tree @ `425c014e` (with the T24 dense_tc port staged
uncommitted), Xolotl @ `cd965e3c`, project-maya @ `35bb2b2`, plus the campaign's banked R1/T13A/
T23 basalt-portstudy dossiers. No builds, no boots, no GPU; the only file written is this one.
Scope: the structural techniques behind the window-time gap, EXCLUDING dense_tc/hc8 kernel
interiors (T24, already ranked; their window-level interactions are covered here).

---

## 0. Summary

1. **Both engines run the same macro-window**: one captured CUDA graph per width `T` on a main
   stream, replayed once per window, one blocking sync at the end, doorbell flags for a host expert
   pool, a separate async commit graph, per-width `exec_[T]` caches, graphs uploaded at capture.
   The family skeleton is shared (Strata parent). **The 6.675-vs-11.5 gap is NOT graph-vs-stream,
   not capture discipline, not sync count.** It is (a) a ~3x smaller launch map (~640-750 nodes vs
   our ~2,050-2,100), (b) a continuous PDL chain with in-kernel early triggers and constant-read
   staging that overlaps nearly every remaining launch's tail and weight reads, (c) fusion of the
   router/quantize/wait structure into fewer, persistent, item-queue kernels, and (d) a draft side
   that never touches the host mid-window. Every one of these is now code-pinned on both sides.
2. **The launch-map arithmetic (high confidence, code-counted):** basalt's dm-mode GDN layer is
   ~9-11 graph nodes (verify.cpp:1259-1998; hc read 1 + mid_chain 1 + hc read+router 1 + route 1 +
   glue gu/swiglu/down 3 + wait 1 + combine 1, shared expert + ring on side streams); their QSA
   layer ~21-23; window ≈ **~640-750 nodes incl. fork/join events**. Ours is 39/GDN + 52/QSA ≈
   ~2,050-2,100 (F-204, doorbell path). At our measured marginal (0.48-0.95 µs/node, R1 §1) the
   node term alone is **−0.65..−1.3 ms**. The rest of the gap is execution overlap, not dispatch.
3. **The PDL reversal is now fully explained (high confidence, code-pinned both sides).** Our
   F-191 measured the launch ATTRIBUTE (no early dependent exists without in-kernel triggers).
   Basalt's chain has three parts our window lacks: `pdl_launch_dependents()` fired at named points
   INSIDE kernels (at kernel start for hc_fused/combine/rec_norm — the dependent then stages its
   whole preamble during the primary's body; at last-work-unit for dense_tc); constant weight/state
   reads issued BEFORE `pdl_wait()` (cp.async rings, TMA fills, L2 prefetch of the next read's
   norm); and `PdlScope(first_plain)` boundary discipline so the chain breaks only at copies/waits/
   stream forks. Their tree has ~79 launch sites across 20 kernel files, default-on since 36d6f22.
   **Our staged T24 dense_tc port already carries the trigger pattern at the dense_tc sites** — the
   port study's "PORT WITH WORK" is really "finish and extend."
4. **The anti-technique is as valuable as the techniques (high confidence):** basalt BUILT the
   whole-layer persistent kernel, it was bitwise-correct and SLOWER (7.541 vs 6.675 ms), and they
   retired it with root causes that read as a checklist of our R1 S3 risks: cross-CTA dependency
   latency (58 µs vs 30 µs byte floor at T=3), 1.5 MB of SASS stalling gate/up warps on instruction
   fetch, two CTAs/SM not fitting, and a standalone harness that predicted gains the engine did not
   show (ratio 0.1-0.3) because it started CTAs with experts already known. **CLC: zero occurrences
   in their tree.** Nobody in the family has shipped CLC; our F6 falsifier remains decisive.
5. **Ranked answer in one line:** finish T24's dense_tc/hc8 (elsewhere), then port the PDL chain
   (triggers + staging + scope discipline) and the wait/combine restructure (1-block wait kernel,
   combine as its programmatic dependent, pre-wait VRAM loads) — together the two largest
   non-dense/hc lines; then the launch-map diet (router-in-read, quantize-into-producer,
   pair launches, QSA branch forks); `cudaGraphInstantiateFlagUseNodePriority` is a one-flag arm
   our tree lacks; the draft conditional graph + in-graph sampler complete the host-exit.
   Honest composed band on our anatomy: **−2.0..−4.3 ms/window beyond dense_tc/hc8**, bracketing
   the ~4.8 ms gap; the irreducible remainder is our byte-floored expert line (~1.73 ms) plus
   bench-config differences (their 6.675 run carries ~zero CPU-pool exposure; ours 0.3-0.7).

---

## 1. The comparable numbers, and what each side's window contains

| | ours (standing, M1) | basalt (headline) |
|---|---|---|
| window wall | ~11.4-12.0 ms (T13A §2) | 6.675 ms (docs/unified-kernel.md:32) |
| accepted/window | ~2.47 | 2.17 (= 324.6 t/s ÷ 149.8 w/s) |
| per-token | ~4.66 ms | ~3.07 ms |
| drafter | ours is BETTER (74.9% vs 70.1% accepted) | — |
| nodes/window | ~2,050-2,100 (F-204) | ~640-750 (code-counted, §2.2) |
| config | 131k ctx, Zen5 pinned-RAM pool, ~1-2% miss | prose fixture, `--peer-exclusive --resident-experts` (5060 Ti holds overflow; CPU pool empty), short ctx |

Calibration (T23 portstudy, standing): their 8-bit-dense arm on the same fixture ran **213.4 t/s ≈
our 212.5** — engines at parity before their Oct 8-20 wave; the whole ~1.5x accrued in named
commits. Two honesty caveats that survive into every comparison below: (i) their 6.675 window runs
with the CPU tier EMPTY (`--peer-exclusive`), so their window's doorbell/pool exposure ≈ 0 while
ours is 0.3-0.7 ms exposed (T13A §2 row 4) — subtract a config term, not an engineering term;
(ii) their own A/B tables show 10.00→9.24 ms windows for the dense-format change on a build that
by then had the other features (features.md:26-34) — their published windows range 6.675 to ~10 ms
across fixtures/configs, so the headline is their BEST config, and per-feature attributions must
come from their isolated A/Bs, not from the headline.

---

## 2. Their window anatomy (code-pinned @ fb0944c7)

### 2.1 Topology and capture

- **Streams** (verify.cpp:1045-1090): `cs_` main at `cs_prio`; `sh_cs_` shared-expert side stream;
  `sh_hi_` and `ring_hi_` at the **greatest** priority ("side branches at stream priority %d",
  :1063); `qs_[3]` QSA branches at greatest priority; `ple_s_` for layer 1's PLE projections
  forked from the window's start; `cm_` commit stream; `copy_` DMA. Fork/join via
  `ev_fork_/ev_join_/ev_ring_/qev_[5]` **inside the capture** → parallel branches in the graph.
- **Capture** (:2298-2350 `capture`): `cudaStreamBeginCapture(ThreadLocal)` → `record_window` →
  `cudaStreamEndCapture` → **`cudaGraphInstantiateWithFlags(..., cudaGraphInstantiateFlagUseNodePriority)`**
  (:2334) → **`cudaGraphUpload`** (:2341) → sync. Two graph sets per width: `exec_[T]`
  (zero-doorbell, all-resident) and `exec_nr_[T]` (doorbell), chosen per window by `refresh_ar`
  (:2282-2296). Batch graphs reuse the recorder and instantiate with `priority=true` (:3284-3285).
  *Our tree has the upload (:2487) and priority streams (verify.cpp:929) but NOT the
  UseNodePriority flag — inside a captured graph, stream priorities are honored by the graph
  scheduler only when the exec is instantiated with that flag. This is a one-line arm we lack.*
- **Run** (:2744-2960): stage inputs to pinned/mapped buffers → ONE `cudaGraphLaunch` (:2789) →
  one `(void) cudaStreamQuery(cs_)` (:2807, the driver call that flushes submission) → in doorbell
  mode a per-layer mapped-spin loop serving only the CPU pool (:2839-2924, with a 2 ms
  `cudaStreamQuery` flush cadence, :2851-2858) → ONE blocking `cudaStreamSynchronize` (:2933).
  The graph.hs header (graph.hpp:5-27) documents the two measurement-backed invariants: the host
  must poll an EVENT not bare memory (a memory-only spin never runs the kernel — the driver batches
  submission until a call arrives), and a graph re-reads its input buffers at replay but never its
  kernel arguments ("anything that changes per step must be DATA in a device buffer").
- **Commit** (:2388-2449 `capture_commit`): separate graph on `cm_`; the GDN commit claims items
  one at a time from a counter with **`kCommitBlocks = 32` persistent blocks** (:262-264) "so it
  trickles beside the draft instead of filling the card"; the next window opens with a finisher
  claiming the remainder (:1200-1203 `gdn_commit_finish`); the window's first node waits only for
  the commit's device epoch counter, with a 0.5 ms host fallback to event ordering (:2787-2788,
  :2376-2386).
- **Overlap structure** — their own one-paragraph spec (:252-264), worth quoting in compressed
  form: shared expert's side kernels at greatest stream priority in T≥2 windows; z projection
  forked at that priority beside qkv at T≤2; MoE route split (CUDA0 plan+q8 rows on main, peer/CPU
  ring on its own high-priority stream); **combine runs as the wait kernel's programmatic
  dependent**; QSA K/V/indexer on branches beside the query; block scores skip identity windows;
  head hc read does every column in one launch; attention merge writes the out-projection's q8_1
  rows; hc read writes the projections' q8_1 rows; qkv+z are one `dense_tc_pair` launch.

### 2.2 The per-layer launch map (their node count, code-counted)

dm-mode GDN layer with `mid_ok` + `glue_on` (the standing single-GPU path, verify.cpp:1377-1442,
1860-1943, 2004-2124):

| # | node | kernel(s) | notes |
|---|---|---|---|
| 1 | hc read 0 | `hc_fused_read` | norm+down+up+inject in ONE persistent launch (`L.nb = d.sms` blocks, global item queue, hc_fused.cu:968-1010, :1476); writes next stage's q8_1 rows; L2-prefetches next read's w_norm (:1427) |
| 2 | mid chain | `mid_chain` | qkv+z pair + recurrence + out proj in ONE persistent launch (T≤4); all-CTA-resident |
| 3 | hc read 1 + router | `hc_fused_read` (router args) | router GEMV + top-10 fused INTO the read (`w_router/logits/ids/weights`, :1875-1880; `hc_top10` smem, hc_fused.cu:967) |
| 4 | route | `dec_moe_route` mode 1 | builds CUDA0's plan + q8 rows + ring (one launch; :1910-1943) |
| 5-7 | glue | `dec_moe_glue` | gate/up resident kernel + swiglu_q8_1 + down, all PDL-chained (dec_moe.cuh:1741-1785; ring's work inside the gu launch) |
| 8 | wait | `dm_wait_kernel` | ONE block; polls only the peer ring and host flag (dec_moe.cuh:314-329) |
| 9 | combine | `dm_combine4` | programmatic dependent of the wait; loads CUDA0 rows + weights + shared row BEFORE its pdl_wait (:125-159) |
| + | shared expert | `shared_expert_multi` | side stream `sh_hi_`, 1 launch |

≈ **9-11 nodes/GDN layer**. QSA (`qsaf_wide`, :1648-1699): hc0, V proj, idx gemv + kv_post, K proj
+ kv_post, idx-q gemv, q proj, q_post, scores, topk, kv_resolve, fused attn, out proj, hc1+router,
route, gu, swiglu, down, wait, combine ≈ **21-23 + shared + 3 branch forks**. Window total ≈
36×10 + 12×22 + prologue/head/sampler ≈ **~640-750 nodes incl. event nodes** — vs our ~2,050-2,100
(F-204: GDN 39, QSA 52 doorbell). The fusions that buy the difference are enumerable: router+top10
inside the hc read (ours: separate GEMV + top10 + route_resident, 3d ~0.83 ms); q8_1 quantize
absorbed into every producer (hc read, attn merge, rec, route — ours: standalone `quantize` nodes);
qkv+z as one pair launch; one-launch-per-phase persistent kernels (hc read, mid chain) instead of
per-op launches; the E-6-style control complement deleted by device planning (dm) rather than by
conditional nodes — their unplanned fallback is the doorbell graph `exec_nr_[T]`, a WHOLE-GRAPH
switch, not per-group spin nodes.

### 2.3 The PDL chain (the F-191 reversal, mechanism-complete)

Their wrapper (pdl.hpp): `launch_pdl` = `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization`
with a per-(kernel,device) capability cache; `PdlScope(on, first_plain)` = thread-local scope so
recorder call-sites declare where the chain may run and where it must break (first launch after a
copy/wait/fork goes plain); `pdl_wait` = `griddepcontrol.wait`; `pdl_launch_dependents` =
`griddepcontrol.launch_dependents`. `pdl_requested()/pdl2_requested()` are `constexpr true` —
**no env knob, always on** (CONTRIBUTING's "one fast path"). Attributes survive stream capture;
their capture retries plain if instantiation with PDL fails (verify.cpp:2301-2310 `retry_plain`).

The three load-bearing patterns, each code-pinned:

1. **Early triggers at named points.** `hc_fused_kernel`: stages the first down item's rows
   (TMA/cp.async), then `if (L.early) pdl_launch_dependents(); pdl_wait();` (hc_fused.cu:1000-1003)
   — the trigger fires at KERNEL START, so the dependent (mid chain, or next hc read via the
   combine) launches and stages while the entire read runs. `dec_gdn_rec`'s norm kernel:
   `if (a.early_trigger) pdl_launch_dependents()` at start (:335-337) — the out projection stages
   its weights during the whole norm. `dm_combine4`: trigger at start — "the next kernel (the hc
   read) may start its blocks and stage weights while this one waits" (dec_moe.cuh:134-137).
   `dense_tc`: trigger at the LAST work unit (:434), and the pair kernel variant that lets its
   dependents in at its own start (:968-986). Inside dm's glue: non-lead blocks trigger "once the
   plan is out" (dec_moe.cuh:433).
2. **Constant reads before `pdl_wait`.** dense_tc issues S-1 weight chunks through its cp.async
   ring before the wait ("weights are constant"); mid_chain fills the first pair tiles and
   L2-prefetches the rec head's state + conv history before the wait (mid_chain.cu:127-136);
   dm_combine4 loads CUDA0's parts rows, routing weights and the shared-expert row before its wait,
   reading only the mapped tiers' rows after (:135-159); the head kernels load weights before the
   wait (native_mmvq.cu:806, 1337-1344); hc_fused prefetches the NEXT read's w_norm through the
   int8 copy table at the end of the current read (verify.cpp:1427, 1881-1884; redirected to the
   hc8 copies at hc_fused.cu:1483-1495).
3. **Chain topology per layer.** hc0 →(PDL) mid_chain →(PDL) hc1+router → plain → route →(PDL)
   gu →(PDL) swiglu →(PDL) down → wait (PDL-dependent of down; spins on peer/host flags) →(PDL)
   combine →(PDL, dm only) next layer's hc0 (verify.cpp:1408-1412: `after_kernel = pdl && dm &&
   dec_hc && l > lb_`; `PdlScope(pdl, !after_kernel)`). The wait kernel itself was STARTED during
   the down kernel ("its rows are complete once this returns", dec_moe.cuh:327) — even the tier
   wait's launch latency is hidden.

**Why our F-191 no-op and their win are consistent** (confirmed by reading our tree): our pdl.hpp
(1-52) has `pdl_wait`/`pdl_trigger` and the same capture discipline, and the staged T24 port of
dense_tc (src/kernels/cuda/dense_tc.cu, untracked) already carries the trigger-at-last-unit and
pair-start patterns verbatim (our :440/:572/:674/:941/:985-986 ≈ their :434/:566/:668/:935/:979-980).
What our standing window lacks: triggers in the hc/gr reads, rec, attention, combine; pre-wait
constant-read staging in those kernels; and the continuous scope wiring across the layer boundary
in dm mode. The F-191 measurement armed the attribute on kernels that had no early triggers and no
staging dependents — there was nothing for the attribute to overlap.

### 2.4 The persistent pieces that stayed, and the one that was retired

- **`mid_chain`** (mid_chain.hpp:1-20, mid_chain.cu, attn_chain_kernel.cuh + attn_chain_bound.cuh):
  qkv+z pair + recurrence + out projection in ONE launch, one CTA per SM, phases hand over via
  global-memory counters (a pair ring with EVEN slot count — documented failure mode if odd),
  TMA `CUtensorMap` for rec state, mbarrier parity waits; PDL-coupled (dependent of hc0, parent of
  hc1); bitwise vs the three kernels it replaces; scoped to T≤4, GDN only, QSA/batched/profiled
  windows keep separate kernels. Every device wait is bounded by `%%globaltimer` and traps with a
  code+line on timeout instead of hanging (attn_chain_bound.cuh:1-32).
- **Commit finisher** (verify_kernels.cu; kCommitBlocks=32): see §2.1.
- **RETIRED — the whole-layer kernel** (docs/unified-kernel.md:24-56): bitwise-identical, slower
  (7.541 vs 6.675 ms). Per-layer losses: experts half +11 µs (bound by cross-CTA dependency latency:
  58 µs vs 30 µs byte floor at T=3), QSA half +35 µs (serializes projections the separate kernels
  overlap; adds grid barriers). Two CTAs/SM don't fit (99 KB smem, 512 threads, 128 regs); 1.5 MB
  SASS → gate/up warps 20% stalled on instruction fetch; windows 1 and 5-7 gained nothing; the
  fused layer forced the tag protocol off (−1% at T≥5). **The standalone harness predicted gains
  the engine didn't show (ratio 0.1-0.3) because it started CTAs with the experts already known —
  in-engine they arrive ~7 µs after the attention output.** Their Lessons block: check register
  spills on the real binary (`cuobjdump --dump-resource-usage`; `--split-compile` quadrupled spills,
  −9% t/s), confirm dynamic-smem fit (silent fallback), keep per-launch memsets out of hot loops,
  convert per-layer µs to window with **~0.73 × 36 per layer-µs**, and reproduce the engine's start
  state in any harness.
- **CLC (`clusterlaunchcontrol.try_cancel`): zero hits in their tree.** Their persistent kernels
  are fixed-grid + counter barriers + PDL coupling, not CLC work-stealing. Our S3 gamble (R1 §3)
  remains un-proven by anyone; their scoped design is the existence proof for the SCOPED variant
  only (our S3 v1/v2 shape), and their retirement is direct external evidence against our S3 v3
  (multi-layer/whole-layer) ambitions.

### 2.5 The draft side and the host exit

- **Conditional draft graph** (mtp.cpp:1240-1320 `capture_cond`): top graph = round + a chain of
  `cudaGraphNodeTypeConditional` IF nodes (one per step, handles created with
  `cudaGraphCondAssignDefault` so a skipped step leaves the next handle at 0 — fails safe to
  not-run); each body captured with `cudaStreamBeginCaptureToGraph` into the node's `phGraph_out`;
  a tiny `mtp_draft_cond` kernel sets handles from device-resident probabilities (the cumulative
  keep rule `draft_cum.hpp` evaluates the same products host+device in the same order). Zero host
  participation between round and last step; falls back to per-step graphs on any failure
  (`cond_off_`). This is R2's mechanism with a working reference implementation — and it is the
  pattern our S1 needs for the verify side too (they build conditionals EXPLICITLY and capture
  bodies into them, rather than post-capture surgery; both paths are viable, theirs avoids the
  node-matching step).
- **Sampler**: greedy sampling is a captured in-graph kernel (record_window:2212-2218, "no extra
  launch or sync per window"); non-greedy/penalized/biased requests queue the sampler on the same
  stream immediately after the graph launch (:2793-2806) — "queued behind the window now, not after
  a sync". This is our R3 rider, shipped.
- **PLE**: layer 1's projections fork at the window's START (ple_s_, record_window:1176-1196) and
  the host gathers NVMe rows at k==0 AFTER raising layer 0's flags so the reads overlap layer 0's
  attention (:2897-2907).

---

## 3. Gap decomposition: 11.5 → 6.675 ms

Using our T13A segment map as the ledger and pricing each basalt technique on OUR anatomy. Bands
are honest (not additive to the last digit; overlaps noted). dense_tc/hc8 kernel interiors are
listed for completeness but ranked elsewhere (T24).

| # | technique (their anchor) | our line it attacks | expected Δms/window on OUR graph | overlap caveat |
|---|---|---|---|---|
| 0 | dense_tc + Q6_K dense re-quant (features.md:26-34: 10.00→9.24 prose on their stack) | 3b dense 3.3 ms | −0.3..−0.7 (Q8_0-on-TC) + more post-re-quant | T24; staged in-tree |
| 1 | hc8 + hc_fused persistent read (T23 #2) | 3a hc 2.87 ms | −0.9..−1.6 (T17D TC-hc band) | shares #2's fusion |
| 2 | **launch-map diet** to ~700 nodes (router-in-read, quantize-into-producer, pair launches, QSA branch forks; §2.2) | dispatch term ≈ 1,300 nodes × 0.5-1.0 µs (R1 §1) + router slice of 3d (~0.3-0.4) + quant slices of 3e/3f | **−0.8..−1.7** | node term double-counts nothing; fusion execution gains bounded by F-217's lesson |
| 3 | **PDL chain** (triggers + staging + scope; §2.3) | launch-tail + weight-read overlap across the REMAINING ~700 launches; T13A #9's band | **−0.5..−1.5** | overlaps #2 (fewer launches = less to overlap); some of #1's gain arrives via the same mechanism |
| 4 | **wait/combine restructure** (1-block wait; combine as PDL dependent with pre-wait CUDA0 loads; dm whole-graph doorbell switch) | 3f copy+combine 0.82 + part of 3h spins 0.4-0.8 | **−0.2..−0.5** | our E-6/S1 deletes the same spin nodes a different way; dm's ring path needs our pool rework |
| 5 | **priority scheduling inside the graph** (`UseNodePriority` + greatest-priority branches; §2.1) | 3e shared+quant 0.81 (hide behind expert kernels), z fork, QSA branches | −0.1..−0.3 | needs #2's forks to exist first |
| 6 | **two-stage Q6_K head** (features.md:52-75; 521.5→~199 MB) | 3g head 0.45 | −0.2..−0.3 | independent |
| 7 | **draft conditional graph + cum keep rule** (mtp.cpp:1240-1320) | 5 draft 0.95-1.19 (host spin ~50-100 µs/step) | −0.1..−0.3 | R2/F1's mechanism, shipped by them |
| 8 | **in-graph + on-stream sampler** (record_window:2212-2218; run:2793-2806) | 6 tail / R3 | −0.03..−0.08 | independent |
| 9 | commit trickle + finisher (kCommitBlocks=32) | 6 commit | −0.02..−0.06 | our commit already async |
| — | bench config (CPU pool empty, short ctx, resident+peer) | row 4 pool exposure 0.3-0.7 + part of 3h | −0.3..−0.7 **not engineering** | cancels in the 213.4≈212.5 calibration |

Sum of engineering rows ≈ **−3.2..−6.6** naively; de-overlapped (rows 2/3 share, 0/1 share) the
honest joint band beyond dense_tc/hc8 is **−2.0..−4.3 ms**, which brackets the 4.8 ms gap with the
config row inside it. Their per-layer retirement numbers cross-check the scale: the separate-kernel
build's per-window budget at 0.73×36 ≈ 26 µs per layer-µs means their whole-wave win (~4.8 ms)
≈ ~185 µs/layer ≈ ~5 µs/GDN-layer-kernel — consistent with overlapping ~10 launches × ~0.5 µs
dispatch + tails, not with a different physics.

---

## 4. Ranked techniques for OUR window (mechanism / evidence / conflict / effort / falsifier)

Ranked by (expected Δ on our anatomy) ÷ (effort × risk), beyond T24's dense_tc/hc8.

### T-1. PDL chain completion — triggers + staging + scope discipline (PORT NOW, the F-191 reversal)
- **Mechanism:** extend our pdl.hpp discipline (which already has wait/trigger/scope) with: (a)
  `pdl_trigger()` at kernel start of our hc/gr reads, rec-norm, attention gate, and combine — each
  conditioned on a `next_early` arg exactly as their `hc_fused_read` does (verify.cpp:1417-1426);
  (b) pre-`pdl_wait` constant reads in the dense/gemv kernels (our staged dense_tc already does
  this); (c) `PdlScope(first_plain)` at every copy/wait/fork boundary in `record_window`.
- **Their evidence:** default-on since 36d6f22; ~79 launch sites/20 files; the hc3_early commit
  history (b7cb16b → 0212b8d/154b9b3 → ca533a5 → 36d6d22) is the map of which pairs mattered.
- **On our graph:** T13A #9 band −0.5..1.5 ms/window; conc-favored (larger maps, COHORTS shares).
- **Conflict surface:** our in-graph flag spins (E-6) break the chain — PDL edges cannot follow a
  spin node (pdl.hpp: "never after a memcpy... or an event node"); wire after the wait kernels
  exactly where they do (`after_kernel` in dm mode). COHORTS: shared recorder, one wiring. norm_fast
  (fused_gr.cu): independent files, no staging conflict.
- **Effort:** M (kernels + wiring), staged behind `STRATA_PDL2=1`; no numerics change (bit-identical
  class — PDL only moves launch/loads; their wait-semantics reading confirmed by our T17D §E).
- **Cheapest falsifier:** arm 2-3 kernel pairs env-gated; nsys overlap probe on the F-235 infra —
  if measured overlap per pair < ~0.3 µs, the lane is priced and closed in an afternoon.

### T-2. Launch-map diet, basalt-scale (the S1+S2 tranche with their menu as the answer key)
- **Mechanism (their exact items, our targets):** router+top10 fused into hc read 1 (deletes 2-3
  nodes/layer and one mixed_ pass; our router chain is verify.cpp:1519-1557); q8_1 quantize absorbed
  into producers (their attn-merge/route/rec write xq; our F-204 q8_1 producer-fold item, −180);
  qkv+z pair launch (arrives with staged dense_tc_pair — our port carries the pair kernel);
  QSA K/V/indexer branch forks (3 side streams; our QSA is 52 nodes serial-heavy); LFUSE gate
  extension (already ours, R1 S2's cheapest item).
- **Their evidence:** the whole §2.2 map; their retirement lesson bounds the fusion risk (merged
  bodies can LOSE: 1.5 MB SASS, instruction-fetch stalls).
- **On our graph:** −900..−1,300 nodes ≈ −0.5..−1.3 ms at our marginal + the router/quant fusion
  execution terms (−0.2..−0.4); R1's S1 conditional nodes and S2 fusion menu are the SAME ledger —
  basalt is the existence proof the end-state reaches ~10 nodes/GDN-layer without a whole-layer
  kernel.
- **Conflict surface:** COHORTS two-graph batch path shares the recorder (free); COHORTS byte-equality
  gate re-run per F-233; norm_fast file touched only by the hc-read work.
- **Effort:** M per item; the R1 falsifier ladder (F0/F3/F5) stands; add their width-mix constant
  (0.73×36) to convert per-layer µs to windows when pricing fusions offline.
- **Cheapest falsifier:** unchanged from R1 (F0b nsys node-trace re-bank, then LFUSE arm).

### T-3. Wait/combine restructure (their dm tail) (PORT WITH WORK)
- **Mechanism:** 1-block wait kernel that polls ONLY the non-VRAM tiers, with the combine launched
  as its programmatic dependent; the combine's pre-wait loads cover the CUDA0 parts rows + weights +
  shared row so the tier wait exposes only the mapped-row loads (dec_moe.cuh:314-329, 135-159).
  Their doorbell/unplanned fallback is a whole-graph switch (exec vs exec_nr), not per-group spin
  nodes.
- **On our graph:** 3f + part of 3h ≈ −0.2..−0.5 ms; also deletes our always-launched empty-PCIe
  trio by device counts==0 (their `plan->counts` checks inside the wait kernel) — the same nodes R1
  S1 targets with conditionals.
- **Conflict surface:** this is E-6's machinery extended, not replaced; our CPU-pool service loop
  (publish/flags) stays for the pinned-RAM tier; COHORTS unaffected (per-slot rings like theirs).
- **Effort:** M/L. **Falsifier:** instrument the wait kernel's spin duration (`--wait-stats` is
  their wstat pattern, dec_moe.cuh + verify.cpp:2942-2952) for one boot; if our exposed tier wait
  is already < 0.2 ms/window (POOL_DIET may have shrunk it), the restructure prices near zero and
  S1's conditionals suffice.

### T-4. `cudaGraphInstantiateFlagUseNodePriority` + greatest-priority forks (ONE-FLAG ARM)
- **Mechanism:** instantiate all window/batch execs with the flag so the priority streams we
  already create (verify.cpp:929) schedule their nodes ahead of the expert launches inside the
  graph; add the sh/z/QSA forks where missing.
- **Their evidence:** verify.cpp:2334, :3285; the :252-264 comment says the shared-expert priority
  is load-bearing at T≥2 and z-fork at T≤2.
- **On our graph:** −0.1..−0.3 ms (3e hiding) + better tail behavior at conc.
- **Effort:** XS. **Falsifier:** capture A/B with the flag on our standing graph, one nsys pass,
  compare shared-expert overlap. (Watch: priority inversion vs the doorbell spin nodes — their
  wait kernels sit on the MAIN stream, forks only carry independent work.)

### T-5. Draft conditional graph + cumulative keep rule (R2's mechanism, their reference impl)
- **Mechanism:** mtp.cpp:1240-1320 pattern (explicit conditional nodes + BeginCaptureToGraph
  bodies + default-reset handles) + draft_cum.hpp's ~50-line joint-probability keep rule scaled by
  served expert bytes.
- **On our graph:** draft phase 0.95-1.19 → −0.1..−0.3 ms/window; re-opens the depth/min-p stairs
  with a better rule (T16 #4/#5).
- **Conflict surface:** mtp.cpp only; COHORTS untouched; our `chain()`/`propose_after` stay (we are
  ahead on arbitration).
- **Effort:** M. **Falsifier:** F1 (STRATA_DRAFT_TIMING capture boot) then host-side replay of the
  keep rule on logged STRATA_SPEC_PROB data — no boot to first order.

### T-6/T-7. In-graph/on-stream sampler; commit trickle + finisher (riders, XS each)
R3's exact shape is shipped by them (greedy in-graph, sampled queued on-stream after launch);
commit trickle (32 persistent blocks claiming from a counter, next-window finisher, device-epoch
wait with 0.5 ms host fallback) is polish on our already-async commit. Take both opportunistically.

### Anti-techniques (do NOT port)
- **Whole-layer persistent kernel**: retired by its builders with root causes matching our R1 §3
  risk ledger 1:1 (hang class, occupancy, harness mismatch). S3 stays scoped (per-phase/per-group),
  v1 = device-planned groups only, bounded waits (their attn_chain_bound pattern is the template —
  adopt it for ANY device-side spin we add).
- **CLC work-stealing**: absent from the entire family; our F6 proto remains the only decider.
- **`--split-compile`**: quadrupled register spills on their units (−9% t/s) — never on kernel TUs.

---

## 5. Xolotl and maya equivalents (family calibration)

- **Xolotl** (cd965e3c): same window skeleton (verify.cpp:1920/2055 captures, exec_/exec_nr_
  dual graphs, commit graph, cudaGraphUpload :2037). PDL present only as our inherited
  attribute-only form (pdl.hpp + one gpu_stamp site) — **Xolotl has NOT built the trigger chain**;
  its wins are three-tier placement and serving machinery, not window anatomy. No UseNodePriority,
  no persistent mid-chain, no conditional nodes. For window structure, basalt is the only donor.
- **project-maya** (35bb2b2): plain capture + instantiate (verify.cpp:822/873), commit graph,
  cudaGraphUpload :879, **zero PDL anywhere**, no priority streams, no persistent kernels. Their
  decode speed comes from the spec-arbitration controller (src/spec/) and placement, not window
  structure. Confirms the trigger chain is basalt's differentiator inside the family.
- Strata upstream (v0.2.1.4) and our fork carry the family skeleton: per-width captures, doorbell
  flags, async commit, graph upload — i.e., the baseline all three siblings share, and the reason
  the gap decomposes into the techniques above rather than into "different architecture."

---

## 6. Contradictions and premise corrections

1. **"PDL closed (F-191, docs-confirmed)" vs basalt default-on** — resolved in §2.3: the closure
   measured attribute-only launches. Our own pdl.hpp already provides triggers; the staged T24
   dense_tc port carries the trigger pattern. F-191 stays closed as measured; the lane re-opens as
   T-1 with a different mechanism (no re-test of a closed arm).
2. **Their headline 6.675 vs their own A/B windows 9.24-10.00 ms** — not a contradiction: different
   packs/fixtures/sessions, all labeled development-grade. We price from their ISOLATED deltas
   (dense_tc −0.76 ms) and our anatomy, never from the headline (T23's calibration discipline).
3. **"Conditionals need post-capture surgery on our recorder" (R1 S1 path (a))** — basalt shows the
   alternative: build the conditional nodes explicitly and capture bodies via
   `cudaStreamBeginCaptureToGraph` (mtp.cpp:1288). Both remain viable; their path avoids
   node-matching brittleness; R1's fail-safe default-reset design matches theirs
   (`cudaGraphCondAssignDefault`).
4. **"Persistent kernels are the 300-class lever" (T17A R1) vs their retirement** — sharpened, not
   contradicted: the SCOPED persistent kernel (GDN middle, T≤4) landed and is default-on; the
   WHOLE-LAYER version lost. Our S3's honest band (−2..−4 ms) should be re-read as T-2+T-3+T-1's
   composed band with the mid-chain as one ingredient, not as a single-kernel bet.
5. **Node-count estimate for their window** — this dossier's ~640-750 is code-counted from the
   dm-mode launch map (§2.2), not published by them (their docs publish no node count). Our F-204
   count is measured. The 3x ratio is robust to ±15% either way.

---

## 7. Open questions

1. The isolated PDL size on THEIR stack is deliberately unpublished (folded into the chain); our
   T-1 falsifier prices it on ours. Their ps_pdl bit history (13b1a0c) is the map if we want their
   pair-level ordering.
2. Does `UseNodePriority` matter without basalt's fork structure (i.e., on OUR current graph with
   only the shared-expert late fork)? One nsys pass answers; sequenced with F0b.
3. mid_chain on OUR graph path: their residency requirement ("stream drained of kernels that wait
   on later work") must be reconciled with our doorbell spin nodes — on our standing residency the
   mid-chain's PDL coupling may need the dm-style whole-graph switch first (T-3 before mid-chain).
4. Their `--wait-stats` exposure number on the peer-exclusive bench is not published; our T-3
   falsifier measures our own.
5. Whether the hc read's router fusion is bitwise-portable to our gr_read path given our
   ROUTE_RESIDENT tie-order contract (top10 semantics) — R1 S2's router-fusion caveat, unchanged.

---

## 8. Implications

- The window gap is fully attributed to portable engineering: launch-map diet (R1 S1+S2's ledger,
  basalt-scale), the PDL trigger chain (our staged dense_tc already carries its seed), the
  wait/combine restructure, and the draft-side host exit. No exotic machinery: no CLC, no
  whole-layer persistence, no graphs-vs-streams rewrite.
- Sequencing recommendation (consistent with R1's ladder): F0b (nsys node-trace) → T-4 one-flag arm
  → T-1 pairs + nsys overlap → T-2 items in R1's order (LFUSE first) → T-3 → mid-chain last (needs
  T-3's dm-style tail). The composed R1 band (+3.5-8% bankable; 300-class gated) survives and gains
  a second external existence proof for the end-state (~10 nodes/GDN-layer, continuous PDL).
- The retirement postmortem should be folded into S3's risk register as external evidence (risk
  rows 1, 5, 6 gain citations), and their Lessons block (register spills on the real binary, smem
  fit, harness start-state, 0.73×36 conversion) adopted into our battery standards (overlaps T23
  candidate 7).
- COHORTS: every T-item above shares the recorder and lands on the batch lane free — the conc lane's
  launch-map and PDL gains are expected to exceed M1's, as R1 §8 already argued.

---

## 9. Citations

**basalt @ fb0944c7 (code, read this session):**
- src/core/verify.cpp: :252-264 (overlap-structure comment), :264 (kCommitBlocks=32), :1045-1090
  (stream creation + priorities), :1142-2222 (record_window: PLE fork :1176-1196; finisher
  :1200-1203; mid_ok selection :1377-1407; PdlScope/after_kernel :1408-1412; hc read + next_early
  :1413-1431; mid_chain launch :1434-1439; GDN dm path :1462-1509; QSA wide path :1643-1699; router
  in hc read :1871-1889; ring fork/glue :1860-1943; dm wait+combine :2102-2112; captured greedy
  sampler :2212-2218), :2224-2278 (profile names + host-ms table), :2282-2296 (refresh_ar),
  :2298-2350 (capture: retry_plain, UseNodePriority :2334, Upload :2341), :2388-2449 (capture_commit),
  :2744-2960 (run: launch :2789, query :2807, on-stream sampler :2793-2806, doorbell loop
  :2839-2924, flush cadence :2851-2858, single sync :2933), :2937-2941 (plan_err contract),
  :2942-2952 (wstat), :3284-3285 (batch instantiate priority=true).
- include/basalt/core/graph.hpp:5-27 (wait-must-poll-driver; args-are-frozen; fixed addresses),
  graph.cpp:84-102 (wait_ms event query + real-clock deadline).
- include/basalt/kernels/pdl.hpp (whole: wrapper, PdlScope, pdl_wait/launch_dependents, capability
  cache); src/kernels/cuda/: hc_fused.cu:968-1010 (item queue, staging, start trigger), :1466-1506
  (launcher: nb=sms, hc8 redirect, pf redirect); dense_tc.cu:370-440 (S-1 pre-wait chunks, last-unit
  trigger), :968-986 (pair start-trigger); mid_chain.cu:127-136 (pre-wait fills + L2 prefetch, wait,
  immediate trigger); mid_chain.hpp:1-20 (contract, even ring, residency); attn_chain_bound.cuh:1-32
  (bounded waits, trap); dec_gdn_rec.cu:335-337 (norm start trigger); dec_moe.cuh:134-159 (combine4
  DEP + pre-wait loads), :314-329 (dm_wait_kernel), :433 (lead trigger), :1741-1785 (glue 3 launches);
  native_moe.cu:125-159; native_mmvq.cu:806, :837-838, :1337-1344.
- src/core/mtp.cpp:1240-1320 (capture_cond: explicit conditional nodes, BeginCaptureToGraph bodies,
  default-reset handles, fallback); include/basalt/core/draft_cum.hpp (~50-line keep rule).
- docs/unified-kernel.md (whole: what runs, retirement table 6.675/7.541, root causes, Lessons);
  docs/precision/features.md:26-34 (dense A/B), :52-75 (two-stage head), :110-135 (spec-7 identity +
  145.1→239.5 / 183.9→319.8), :200-215 (relaxed drafting speeds 323.8-354.2 exact IQ3_S).

**Our tree @ 425c014e (+ staged T24):** campaign/research/R1-design-dossier.md (§1 marginal A/B,
§2 anatomy, S1-S4); campaign/research/T13A-decode-critical-path.md §2 (segment map);
campaign/research/T23-basalt-portstudy.md (calibration, ranked table, dead-end scorecard);
src/core/verify.cpp:929 (priority streams), :2487 (GraphUpload), :2370 (capture), launch-custom-
iq3s-8081.sh:8-39 (standing env); src/core/mtp.cpp:1697-1715 (min_p gate), :1631-1665 (ungated
twin); include/strata/kernels/pdl.hpp:1-52 (attribute+trigger discipline); staged
src/kernels/cuda/dense_tc.cu (:396-440, :985-986 trigger pattern) + dense_tc_parity.cpp (untracked,
T24-S1, STRATA_DENSE_TC default off); git log 425c014e (T24 queued).

**Siblings:** Xolotl src/core/verify.cpp:1920/2055/2300/2373/2817 (same skeleton), :2037 (upload);
src/kernels/cuda/verify_kernels.cu:1466-1484 (attribute-only PDL); project-maya
src/core/verify.cpp:822/873/879/993 (plain capture+instantiate+upload), zero PDL hits repo-wide.

**Methods note.** Read-only; the only file written is this dossier (INDEX line for the maintainer:
2026-10-20, T25-5 window anatomy, conclusion: gap = launch-map 3x + PDL trigger chain + wait/combine
restructure + draft host-exit; composed band beyond dense_tc/hc8 −2.0..−4.3 ms; status current).
Launch-map counts for basalt are code-derived (§2.2), a stated estimate. Unproductive searches (do
not repeat): no per-phase ms dump of the 6.675 session exists in their docs (the instruments are
in-tree, the tables are not published); no CLC references anywhere in basalt/Xolotl/maya; maya has
no PDL; their standalone mid-chain microbench numbers are in-repo (mid_chain_parity) but not run
here per the no-GPU constraint. Deliberate non-reads: DeepInfer/DFlash2 (out of scope for window
anatomy), the vLLM-fork history inside DeepInfer.
