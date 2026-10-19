# R1 DESIGN DOSSIER — the verify-window node-collapse restructure (staged: conditional nodes → fusion tranche → CLC persistent groups → composition)

2026-10-19. Engineering synthesis from banked facts + code read; no new external research, no
builds/boots. Scope: the M1 serial verify window first (the measured node-tax lane), the COHORTS
batch windows where the machinery is shared. Standing anchors: M1 ~198-208 (STEP 0 control leg
198.0; dtt leg 208.0 = boot noise), late window ~9.2 ms, ~2,470 tok/s-class transfer law
**+20 t/s per −1.0 ms/window** (T15A, T17A §1). Laws honored: No-Regression absolute; one variable
per battery; F-91 (graphs captured once at boot); F-206 ("node savings != wall time" — now
quantified); F-223 (no intra-window pool hiding; per-slot buffers); F-225 (never compare across
boot positions); PDL stays closed (F-191, docs-confirmed T17D §E).

---

## 0. Executive summary

1. **Premise correction (high confidence, code-pinned) — the node tax is real but its per-node
   price is ~0.5-1.0 µs, not ~23 µs.** The t17a-dum48 leg set `STRATA_DUMMY_NODES=48`
   (campaign/t17-dum48.json `"env"`), and the instrument inserts k empty kernel nodes **per layer**
   (verify.cpp:1152-1155, inside the per-layer loop at :1009; n_layers = 48 = 36 GDN + 12 QSA,
   T12-MEM). The leg therefore added **48 × 48 = 2,304 nodes**, not 48. Measured against the control
   leg: late window 9.2 → 10.3 ms (**+1.1 ms ⇒ 0.48 µs/node**), early 13.2 → 15.4 ms (**+2.2 ms ⇒
   0.95 µs/node**), M1 198.0 → 173.3 (**−12.5%**, consistent with the window inflation at flat
   tokens/window). The "23 µs/node" figure divided 1.1 ms by 48 — it is off by the layer factor.
   Sanity: a noop graph node cannot cost 4x the ~6 µs *average* spacing of real nodes; 0.5-1.0 µs
   is the physically consistent reading. The STEP 0 verdict ("tax REAL and material at scale") stands;
   the *magnitude class* changes everything below.
2. **Consequence for the lane:** node-count collapse alone (S1+S2, a realistic −600..900 nodes of
   the ~2,050) prices at **−0.3..0.9 ms/window = +3.5-9% M1** at the measured marginal — a solid
   tranche, not a leap. The remaining gap to the byte floor (late window ~9.2 ms vs ~2.1 ms T=1
   floor [T17A] / ~4.3 ms light-class floor [T12-GPU]) is **small-kernel execution latency + real
   dependency waits**, which only S3 (persistent layer-group absorption) attacks. The ~2x gamble
   survives, but its basis shifts from "delete 6-7 ms of node tax" to "absorb ~3-5 ms of per-kernel
   ramp"; the Cohere 1.58x production precedent (T17A [S32]) supports ~half the theoretical band.
3. **The standing graph is the E-6 device-plan graph, not the zero-doorbell graph.** `all_resident_`
   requires 100% of all 512 experts of every layer resident (verify.cpp:837-851; verify.hpp:442-448)
   — false forever on the standing 9,426-slot/24,576-pair cache — so `ar_on()` is false and the
   window carries the full control complement: per layer-group ~4 early-exit spin/copy nodes
   (waitA_planned, plan-copy-unless, waitB_planned, wait_or; verify.cpp:1685-1716) plus the
   always-launched empty-PCIe trio (staging fetch + rebase + grouped launch, :1700-1709). These are
   exactly S1's raw material, they exist today, and they early-exit in ~1-2 µs when the group was
   device-planned — i.e. S1's saving is the node dispatch itself, priced by the dummy A/B.
4. **Staged verdict:** S1 (conditional nodes) = M effort, +2-4% M1, falsifiable without a server.
   S2 (fusion tranche) = S-M per item, +1.5-4% M1, two items are one-line env arms, two pending
   adjudications are consumed not re-opened. S3 (CLC persistent layer-group) = L-XL, the only
   300-class lever, falsifier ladder starts with a 20-line proto and a 2-layer bench, both
   server-free. S4 (composition) prices R2-R5 (+2.5-7 t/s combined) and the batch-window ledger.
   Every stage: env-gated, revert = one command, bit-exactness contract stated per stage.
5. **The binding unknown** the ledger cannot price yet: the true per-node spacing distribution on
   the CURRENT standing recipe (the ~6/10.7 µs numbers are T12-GPU's, pre FUSE_HITS/POOL_DIET/
   COHORTS). One nsys `--cuda-graph-trace=node` pass (banked T8.C recipe) replaces the estimate
   before S1 surgery. Falsifier F0b below.

---

## 1. The measured basis (what the dummy A/B does and does not establish)

Source legs: campaign/results/t17a-ctl-r1.txt, t17a-dum16-r1.txt, t17a-dum48-r1.txt (F-235 STEP 0,
2026-10-17 22:35-23:03; SVG-CUSTOM-TODO.md "STEP 0 RESULTS").

| Leg | Nodes added | M1 | ΔM1 | Trend early | Trend late |
|---|---|---|---|---|---|
| ctl-r1 | 0 | 198.0 | — | 13.2 | 9.2 |
| dum16 (16/layer = 768) | 768 | 197.6 | −0.2% (flat) | 13.6 | 8.9 |
| dum48 (48/layer = 2,304) | 2,304 | 173.3 | **−12.5%** | 15.4 | 10.3 |

- The instrument: `gpu_noop()` = `noop_kernel<<<1,32>>>` (verify_kernels.cu:1533-1536), inserted as
  a contiguous run per layer inside `record_window` (verify.cpp:1152-1155) — a pure node-count
  intervention: no bytes, no work, same dependencies.
- **Marginal node cost: 0.48 µs/node (late window), 0.95 µs/node (early/cold).** The M1 −12.5% is
  window-inflation-consistent (+1.1 ms on 9.2 late ≈ −10.7% at flat tokens/window; the early legs
  carry the rest) and carries the usual single-leg caveat (F-225 boot variance; the campaign's own
  verdict: "INFO WIN, not adoptable").
- **Why linear extrapolation is NOT valid from the average, and IS (bounded) valid from the
  marginal.** The T12-GPU/T17A picture ("~2,050 nodes × ~6 µs ≈ the whole window") treats the
  average spacing as the per-node price. The A/B falsifies that: 2,304 extra nodes cost 1.1-2.2 ms,
  not 13-22 ms. The ~6 µs average is mostly **per-kernel execution/ramp of real kernels serialized
  by the dependency chain**, plus real waits — not dispatch. What the A/B measures is the pure
  graph-topology term (dispatch + scheduling advance ≈ 0.5-1.0 µs/node). Linear extrapolation from
  the *marginal* is valid for interventions that REMOVE nodes while conserving work (fusions,
  conditional skips of already-early-exiting nodes) — with the caveat that a fusion can also lose
  if the merged kernel's register/smem pressure slows the body (F-217's lesson: the SMEM tile was
  load-bearing). It is NOT valid for S3-style absorption, where the prize is the *execution-ramp*
  term the noop never paid; S3's ceiling must be argued from the byte floor instead.
- dum16 (768 nodes, predicted +0.37 ms) read flat: the effect at that scale is inside the ms-window
  quantization (individual rows are integer ms) and boot noise. Use only the 48-leg for pricing.
- **The average/marginal split also re-prices the two banked framings:** T17A's "W(1)=8-9 ms vs
  ~2.1 ms floor ⇒ ~6-7 ms fixed node-dependency tax" decomposes, on this measurement, into
  ~1.0-1.9 ms of true node-count tax (2,050 × 0.5-0.95 µs) + ~3-5 ms of small-kernel execution
  latency + ~0.7-1.5 ms of real waits/spins/pool exposure (T13A 3h/4). S1+S2 monetize the first
  term; S3 monetizes the second; nothing monetizes the third except host-side work already banked
  (POOL_DIET F-227; T12-HOST lane).

---

## 2. The standing window's control anatomy (what actually exists to delete)

Code-pinned (all read this session; `record_window` = verify.cpp:897-1890, captured by
`cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal)` at :1947, instantiated per T into
`exec_[T]`; the COHORTS batch graphs reuse the SAME recorder with `batch_rec_=true`, :2842-2862,
keyed `bkey(rows,S,0)` at :3116).

Per layer-group (G=1 standing serial; the F-204 inventory: GDN 39 nodes doorbell-path / 28
all-resident, QSA 52/41, window ≈ 2,050-2,100 mixed):

1. **E-6 skip sites (4 nodes, early-exit when the group was device-planned)** —
   `wait_flag_ge_planned(m_flagA_, ring, skip_+grp)` :1685; `copy_i32_from_mapped_unless` :1686;
   `wait_flag_ge_planned(m_flagB_, ...)` :1698; `wait_flag_ge_or(m_flag_, ring, skip_+grp)` :1714.
   Kernel bodies: verify_kernels.cu:1373-1392 — read `*skip`; if `== ring` (fully planned) or
   `== kPlanPartialSkip` return, else spin on the mapped flag. `*skip_` is written by
   `resident_plan_par_kernel` (verify_kernels.cu:1305-1358) in the same graph (:1568-1573).
2. **Empty-PCIe trio (3 nodes, always launched, empty almost always)** — staging
   `fetch_blobs` + `rebase_ptrs` (pcie_mode 2 = dma standing) :1700-1705, and the grouped launch
   `grouped(p_ptr2, ..., kPcieGroupRows, hit_out)` :1709 ("its launch is kPcieGroupRows block rows
   striding over the groups"). Live pcie_share 0.001-0.017 (T13A) — counts[2]==0 nearly every
   layer-group; the trio launches and early-exits anyway.
3. **Real-wait sites (NOT deletable by conditionals)** — on unplanned groups the flagA/flagB/flag_
   spins wait for the host pool's publish: a true data dependency (0.3-0.7 ms/window exposed at
   standing-warm, T13A §2 row 4; shrunk by POOL_DIET F-227). Conditional nodes reshape the node,
   not the wait (see S1 caveat).
4. **Adopted/banked already:** `moe_hit_add` is folded into the combine standing (STRATA_FUSE_HITS=1,
   F-226 ADOPTED, launcher:10 — T13A #4's "check why gated off" is resolved: it is on); POOL_DIET on
   (F-227); COHORTS=2 on the batch lane (F-233, conc +24.8%, launcher:12).
5. **Head/prologue** and the drafter's separate graphs (round + gated step loop, mtp.cpp:1562-1746)
   round out the window; the draft chain is S4/R2 territory.

---

## 3. Staged design

### S1 — Conditional nodes around the flag-spin and empty-PCIe nodes (T17D candidate 5)

**Mechanism.** CUDA 12.4+ conditional graph nodes (IF/WHILE; ELSE/SWITCH in 12.8; CC 9.0+ ⇒ sm_120
eligible; condition read and evaluated ON DEVICE, set by device code via `cudaGraphSetConditional`;
body graphs allow kernel/memcpy(pinned-mapped)/memset/child/conditional; no device-graph-launch
inside bodies; optional per-launch default reset) [T17D §2 table, sources S4/S8/S44]. Convert:

- **Site group A (skip-conditioned, 4 nodes/layer-group):** the E-6 spin/copy quartet becomes one
  condition-setter kernel + IF bodies. The setter (one tiny kernel per layer-group, launched after
  `resident_plan`, reading `*skip_` once) calls `cudaGraphSetConditional` on 4 handles:
  planned (skip == ring or kPlanPartialSkip) → set "skip" → all four body nodes vanish; unplanned →
  set "run" → the bodies execute exactly today's kernels (waitA spin, plan copy, waitB, wait_or).
  Net on planned groups: −4 nodes +1 setter = −3; on unplanned: +1 (bounded, and the unplanned
  fraction is small — per-request hits 98.3-99.3% with ROUTE_RESIDENT 0.75 actively swapping
  residents in).
- **Site group B (PCIe-conditioned, 3 nodes/layer-group):** the empty-PCIe trio becomes
  counts-conditioned: setter handle reads `p_counts[2]` (device plan memory, valid after the plan
  node); empty → the fetch/rebase/grouped nodes vanish; non-empty → they run as today. Expected
  taken-rate ≈ 1 − pcie_share ≈ ~99% of layer-groups.
- **Scaffold:** E-6 device plans and the mapped-spin gates are the existing scaffolding
  (verify_kernels.cu:1373-1392, verify.cpp:1568-1573, 1683-1716); `decode_cluster_parity.cpp:1-15`
  is the in-tree precedent for shipping a parity harness beside new graph machinery (synthetic +
  adversarial inputs + a captured-graph replay with new inputs — exactly the staleness class to
  test).

**Capture semantics on CUDA 13.4/sm_120 (the design's sharpest edge).** Plain stream capture
(what `capture()` uses, verify.cpp:1947) cannot *create* conditional nodes — capture records
launched work; conditionals must be added explicitly. Two viable paths; pick (a):

- (a) **Post-capture surgery (recommended):** keep capture exactly as is; after
  `cudaStreamEndCapture`, walk `cudaGraphGetNodes` + `cudaGraphKernelNodeGetParams`, match the four
  wait/copy kernel function pointers and the PCIe trio per layer-group, then
  `cudaGraphAddNode(cudaGraphNodeTypeConditional)` + rewire (`cudaGraphAddDependencies` /
  `cudaGraphRemoveDependencies`) + `cudaGraphDestroyNode`, then instantiate as today. All boot-time
  (F-91: graphs captured once; the per-T `exec_` cache and batch `bkey` cache absorb the surgery
  cost). One surgery site serves BOTH the serial and COHORTS batch graphs (shared recorder).
- (b) `cudaStreamBeginCaptureToGraph`: re-plate the recorder onto a pre-seeded graph — more
  invasive, no added value for this shape.
- Handle discipline: pass the `cudaGraphConditionalHandle` values as kernel parameters to the
  setter kernels (device-side `cudaGraphSetConditional`); arm the per-launch **default reset with
  default = "run the body"** so any stale/missed set fails safe to today's behavior, never skips a
  wait. (Verify the exact reset flag name/semantics on the 13.4 headers in F0 — T17D records the
  capability, not the flag spelling.)
- Device-graph-launch limits (120 fire-and-forget, 1 pending self-launch, no host updates while
  device-launched) do NOT bind: conditional nodes are host-launched and none of this uses device
  graph launch [T17D §2].

**Why F-191's PDL result does not apply to this mechanism.** Three separations, each load-bearing:
(i) *Different cost pool.* PDL overlaps a secondary kernel's **preamble and launch latency** with
the primary's tail; its wait semantics are "completed and flushed" (T17D §E, docs-confirmed). F-191
measured it a clean no-op because our spacing is **dependency serialization**, and the sanctioned
preamble already exists (pdl.hpp:1-20 discipline, kPdlPrefetch, pdl_wait/pdl_trigger). Conditional
nodes do not overlap anything — they **delete nodes from the executed topology**, i.e. they monetize
exactly the ~0.5-1.0 µs/node topology term the dummy A/B isolated, a term PDL cannot touch.
(ii) *Different decision locus.* F-222's T1.1 measured the **host** flagA round-trip at 0.26 ms and
closed the host-planned variant; conditional evaluation is **device-side with zero host
participation** — the mechanism F-222 tested is orthogonal. (iii) *No re-test of a closed arm:*
PDL stays closed; S1 is a new mechanism that happens to live in the same graph.

**Honest caveat (corrects T17D's pricing note).** T17D priced candidate 5 "+2..5% M1, more at conc
where flags wait longer". The second half is wrong under E-6: on unplanned groups the flagA/flagB
wait is a real data dependency (the pool must publish), and an IF body still waits — conditionals
shave node overhead, not exposure. Conc's larger gains come from its larger node count
(~2,900/window at S=6, F-204) and from COHORTS sharing the surgery, not from hiding waits.

**Anatomy-priced magnitude.** Sites: (4+3) × 48 layer-groups ≈ 336 nodes; net removal ≈ 336 − 48
setters − ~2×7 unplanned bodies ≈ **−270..−290 nodes/window** (serial). Plus each skipped node's
early-exit execution (~0.3-0.5 µs). Price at 0.5-1.0 µs/node (A/B marginal): **−0.16..−0.34
ms/window = +1.7-3.6% M1**; batch windows (S=6, ~2,900 nodes, more groups) similar in ms, larger in
conc %. Matches T17D's −0.2..−0.5 ms band. Sits on top of, not inside, the S2 fusions.

**Effort class.** M (graph surgery + setter kernels + harness; no numerics touched).

**Bit-exactness contract.** Byte-identical outputs by construction (the executed set of kernels and
their order is a subset of today's; every value identical; the setter reads only `*skip_`/counts
already produced). Gates: (a) the standalone harness proves both branch polarities (satisfied and
unsatisfied flags) produce byte-identical results to the plain kernels, including the
capture-once/replay-with-changed-flags sequence (the SageAttention #392 staleness class, T17A [S9]);
(b) boot canary + M3b (bit-identical class ⇒ canary suffices per law); (c) the COHORTS byte-equality
gate from R2-S3 S3.2 re-run with surgery on.

**Cheapest falsifier.** F0 (server-free): a 20-node standalone graph — today's spin kernels vs the
conditional rewrite, satisfied/unsatisfied device flags, replayed with toggled flags; measures
per-site saving and proves the handle/reset semantics on our exact driver/toolchain. T17D's own
candidate-5 falsifier. If the harness cannot express the rewrite (capture-surgery API friction) or
the per-site saving reads <0.3 µs, close S1 for one afternoon's cost.

**Revert.** `STRATA_CONDITIONAL_NODES=0` (the surgery is a post-capture pass; off = the plain graph).
One command, no rebuild path difference.

### S2 — Node-count collapse via the priced fusion groups

F-204's scan priced ~−900..−1,000 nodes from ~2,100; that was written before F-226 (FUSE_HITS)
landed. The surviving menu, each item with its own arm per the F-206 law:

| Item | Nodes | Verdict under "structure beats squeezing" | Evidence |
|---|---|---|---|
| **LFUSE gate extension to the device-plan path** (shared-expert chain 6→3) | −144 | **SURVIVES — cheapest, first.** Implemented + bitwise-checked; standing OFF only because `lfuse_on()` conjuncts `ar_on()` (verify.cpp:929) which is false forever on standing residency. One line + env. Bitwise per S26. | F-204 #2; T12-GPU #6 ("one-line arm", verify.cpp:927-930) |
| **Router+plan+bell fusion** (GEMV + top10 [+ route_resident] one launch; plan stays separate) | −96..144 | **SURVIVES as a small tranche.** All readers of the same logits; bit-exact requires preserving top10 tie order exactly (native_router_top10_multi semantics). Value is launch-map length; the GEMV's ~0.27 ms line is NOT the target (T17D do-not-build: "2x on the GEMV ≈ 0.1 ms"). | T12-GPU #3 (~150 nodes); router_top10.cu:30-33 |
| **PLE row loop → batched postops** | −56 | **SURVIVES — env-class.** `native_ple_postops_batch_snap` exists, bitwise-equal (STRATA_DF_PLE); the standing PLE_BATCH covers key/value only, the history-dependent tail stays row-wise (verify.cpp:1106-1127). | F-204 #7 |
| **QSA small-pairs** (K+V mmvq one launch; attn chunk+merge one kernel at decode n_chunks; kv_resolve 2→1) | −48 | **SURVIVES, M effort.** Bitwise-checkable per pair. | F-204 #8 |
| **Expert-side: rebase into fetch_blobs; PCIe-skip when counts==0** | −24..48 | **PARTIALLY SUBSUMED by S1 site group B** (the conditional skip deletes the nodes; the rebase-into-fetch fusion then deletes one more). | F-204 #9 |
| **q8_1 producer-fold (quantize into hc read / gr epilogue)** | −180 | **CONSUMES, does not re-open.** QFUSE-class gates; F-187's single-form mirror-trade caution; riding the pending QFUSE/HC_Q8 adjudication (T16 #8). Hooks exist (xq_out, q8_mixed, verify.cpp:1142-1146). | F-204 #1; T16 #8 |
| **Fused wait+copy doorbell forms (FUSE_WAIT)** | −144 | **ADJUDICATION-PENDING, shrinking.** CUDA default-off by history not semantics (verify.cpp:264-269); T15A re-dig marks DEAD with citations vs F-205's un-gate; T16 #8 strikes one side. If S1 lands first, most of this item's nodes are already conditional bodies — the residual is the fuse itself. | verify.cpp:245-271; T16 #8 |
| **hc gr_norm fold into down/up** | −96 | **DIES.** T13 re-price: folding xn into the up epilogue saves only the w_norm load (~40 KB vs 6.5 MB ≈ 0.5%); the norm kernel's ~0.34 ms of real work must run somewhere; F-217 proved the kernel's tile structure is load-bearing (wide grid = −9.2% M1). Node saving ≠ wall time, again. | T13 #2 (code-read re-price); F-217 |
| **Head mixer 3→2, miscellany** | −60..90 | **OPPORTUNISTIC ONLY** — priced ~0.5-1 µs/node; take only what falls out of adjacent work. | F-204 #10 |

**Anatomy-priced magnitude.** Surviving tranche ≈ **−250..−400 nodes** (LFUSE −144 + router −96..144
+ PLE −56 + small-pairs −48, minus overlaps) ⇒ at the measured marginal **−0.13..−0.40 ms/window =
+1.5-4% M1**, conc-favored via the shorter launch map. The F-206 discipline applies with the A/B
numbers now behind it: each arm's expected yield is ~0.5-1.0 µs × its node delta; any arm whose
merged body adds measurable execution (register pressure, lost L2 reuse) dies in its own ABAB.

**Effort class.** LFUSE arm: XS (one line + env, parity re-check on the device-plan path). Router
fusion + small-pairs: M. PLE: S (env). 

**Bit-exactness contract.** LFUSE: already bitwise-checked (S26) — re-prove on the device-plan path
(gr_parity-class QFUSE byte-equality) since the combine takes the gated form only on ar windows
today. Router fusion: byte-equal ids/weights incl. tie order (the parity pattern of
decode_cluster_parity). PLE: bitwise claimed in-tree; one canary. All canary-class; nothing here
changes bits.

**Cheapest falsifiers.** F3 = the LFUSE arm boot (XS, prices the biggest single item). F5 = router
fusion behind `STRATA_FUSE_ROUTER=1` with the offline parity binary first.

**Revert.** Each item its own env; off = one command.

### S3 — The CLC persistent layer-group kernel (the ~2x spacing gamble, re-based)

**Mechanism.** One persistent kernel per layer-group (v1: the device-planned expert-side chain —
grouped → wait_or/copy_rows → combine; v2: full pre+post per GDN layer; v3: multi-layer groups)
walks its phases under one grid with device-side counter barriers (the doorbell/flag pattern
formalized), CLC work-stealing (`clusterlaunchcontrol.try_cancel` → 16 B mbarrier response;
`query_cancel`) distributing phase tiles without SM-count introspection [T17D §C, sources S6/S14/
S16/S17]. The ~2,050-node dependency chain becomes ~48-96 kernel nodes; per-kernel launch ramps
(~2-4 µs each) collapse into intra-grid phase barriers (~sub-µs to ~1 µs).

**Why the re-based ceiling still reaches 300-class.** After the S1/S2 correction, the window's gap
to floor (~9.2 ms late vs 2.1-4.3 ms floors) is ~3-5 ms of small-kernel execution latency + ~1-1.9
ms of node dispatch + real waits. S3 monetizes the first two: phase barriers replace both the ramp
and the dispatch. Theoretical band −4..6 ms (T17A R1's "−3..−6 ms ⇒ +60..120 t/s"); the honest band
after subtracting what the noop A/B proved cheap (dispatch was never the whole story) and adding
what Cohere/Hazy/MPK/Forge prove attainable (~60% of theoretical: 1.58x over vLLM at 62% SoL,
T17A [S32-35]): **−2..−4 ms/window = +40..80 t/s (M1 240-290 class)**. The 250→300+ claim stays
gated on the falsifier ladder, exactly as T17's consolidated verdict put it.

**Preemption semantics vs our clock lock.** CLC work-stealing kernels remain preemptable — CLC
failure "can signal preemption by a higher-priority kernel, after which CLC can resume" [T17D §C,
S17] — unlike a hard cooperative pin. This is the fit that matters on our box: the 3135 lock
realizes ~2940 under load with benign 0x400 throttle flags (T16 #1), and the card demonstrably
bounces clocks on sub-ms bursts (T8.B). A persistent window kernel must (a) not spin-wait without
progress bounds (bounded-iteration barrier loops + an error word, the m_plan_err_ pattern, #871),
(b) tolerate a mid-window preemption/resume (CLC gives this; a cooperative grid does not), and
(c) not hold SMs the COHORTS sibling window needs — v1 is serial-lane-only; the batch lane keeps
its graphs until S3-b measures the contention.

**SM12x precedents (what exists).** FlashInfer 0.6.17 SM12x fused-MoE families ship "cooperative
persistent launches"; SonicMoE extends to SM120 with ~500 LoC; the CLC instruction family targets
sm_120f; CUTLASS ships depth-3 CLC pipelines [T17D §C]. **No published 5090/GB202 CLC benchmark
exists** (T17D open question 4 — searched; only B200 measurements) — which is exactly why falsifier
F6 is a 20-line compile-and-run proto and not a design review.

**Minimal viable prototype (in-tree shape).** One layer group = the device-planned expert-side
chain of ONE GDN layer (grouped → combine), no host flags in v1 (planned groups only; unplanned
groups fall back to the graph path via the E-6 skip — the fallback already exists as a device
value). Behind `STRATA_PERSISTENT_LG=1` the capture emits the 1-node group instead of the chain for
planned groups. **Bit-exactness contract:** every global word the graph version writes (parts_out /
hit_out path, bo_, combine outputs) byte-identical — v1 keeps each phase's grid EXACTLY the
phase-kernel's grid (same decomposition, so same reduction order ⇒ bit-exact by construction; any
later grid merge changes bits and goes to the F-230 full gate: M3b + coupled-MTP acceptance battery
+ 2× MMLU). Harness: the decode_cluster_parity pattern (synthetic + adversarial + captured-replay)
plus gr_parity-class byte-equality on QFUSE paths.

**The honest risk ledger.**
1. **Hang class (new):** a persistent kernel whose barrier never rises hangs the whole window (a
   graph's early-exit node cannot hang). Mitigations: v1 runs only on device-planned groups (flags
   raised by construction — resident_plan wrote *skip itself); bounded spins + error word + host
   watchdog that treats a hung window as a hard error (fail loud, fail the request, not the
   process-quiet corruption class).
2. **F-223 G2 ghost:** persistent state shared across windows = the corruption class that killed
   batch G=2. Per-slot buffers and monotonic window tags ((window<<6)|step, T11.R2's pattern) are
   the design rule; S3 v1 sidesteps by owning nothing beyond the layer-group's existing buffers.
3. **Numerics:** none in v1 (same grids, same order). The temptation to merge phase grids for real
   is where bits change — gated, priced, sequenced last.
4. **Profiler blindness:** nsys sees one kernel; the phase table needs in-kernel %globaltimer marks
   (the gpu_stamp pattern, verify_kernels.cu:1506-1532, already PDL-aware). Bank the instrument
   WITH the prototype or S3's arms become unmeasurable.
5. **Occupancy/warp specialization:** the persistent grid must not shrink per-phase parallelism
   (the hc line is issue-bound; T17D §A) — v1 launches the union of phase resources at reduced
   occupancy only if the bench says neutral; else persistent-per-layer-group (48 nodes) not
   persistent-per-window.
6. **COHORTS interaction:** two persistent kernels on two streams = the one number S3-b exists to
   measure (contention factor; the R2-S3 plan's open question, now with higher stakes).
7. **sm_120 CLC unproven on GB202** — F6 exists to kill the whole stage for ~an hour's cost.

**Anatomy-priced magnitude, effort, falsifiers, revert.** −2..−4 ms/window honest band (+40..80
t/s); ceiling −4..6. Effort L-XL (weeks), but the falsifier ladder is: **F6** = 20-line sm_120f
CLC proto under PTXAS 13.4 (server-free; kills the stage); **F7** = 2-layer persistent microbench
vs the captured graph slice (server-free; prices the absorption); **F8** = the in-tree 1-layer-group
prototype + parity + battery ABAB. Revert: `STRATA_PERSISTENT_LG=0` (graph path untouched; the E-6
skip machinery is the switch).

### S4 — Composition with R2-R5 and the COHORTS batch path

- **R2 (conditional-node draft chain, +1..2.5 t/s):** same machinery investment as S1 — the gated
  step loop (mtp.cpp:1725-1743) becomes a WHILE whose condition is `m_prob_[j] >= min_p`
  (device-resident, mtp.cpp:1382), per-step trim at zero host cost, the third policy neither
  F-229 arm implemented. Ceiling derived FROM F-229's −2.73% (the bubble pot ≈ +2.4 t/s).
  **Decided by F1** (the STRATA_DRAFT_TIMING capture boot STEP 0 lost to the log rotation;
  gap_ms < 0.05 ms/window retires it free).
- **R3 (sampler-in-graph, +1..2 t/s):** the second sync + host re-sample (verify.cpp:2509-2541)
  folds into the graph; Philox draws are position-keyed (verify.cpp:2501-2504) so the replay
  objection dies; `pos0` stages through mapped memory like every other per-window scalar.
  Exactness-critical: draw parity via the STRATA_SPEC_PROB harness (the DOOR closed as an
  adoption in STEP 0; the harness remains the parity tool). Sequences naturally after S1 (same
  staged-params discipline).
- **R4 (first-window draft seeding, +0..1% probe mean, TTFT −8-9 ms):** `first_window T=1`
  (generate.cpp:13113-13119, verified) runs `draft(1,[x],p,0)` right after prefill. XS.
- **R5 (L2 set-aside retarget, +0.5..1.5 t/s):** the standing 60 MiB sits on GDN states already at
  floor; the drafter re-reads ~66 MB/step (13.8 MB Q2_0 + ~52 MB Q4_0 head). Env-class on the
  launcher; ncu `lts__t_sector` on the draft phase is the falsifier.
- **COHORTS batch path:** the batch graphs share `record_window` (verify.cpp:2842-2862), so S1's
  surgery and S2's arms land there for free — one surgery site, both lanes; the batch windows are
  the BIGGER ledger (~49/GDN + 91/QSA ≈ ~2,900 nodes at S=6, F-204) and conc is where the launch
  map pays most. S3 stays serial-lane until S3-b prices two persistent cohorts. The two-graph
  batch path itself is banked (F-233, COHORTS=2 standing).
- **What the composed window looks like after all stages (serial):** prologue/stage → per layer:
  [S2-fused hc/dense/router chain] → resident_plan → [S1 setter + conditional bodies] →
  [S3 persistent expert-group kernel] → combine → head → [R3 in-graph sampler]; the drafter runs
  [R2 WHILE chain]. Node-count ledger below; host tail unchanged (the pool serves only unplanned
  groups).

### Node-count ledger (serial M1 window; batch S=6 in parentheses)

| Stage | Nodes/window | Δ vs prev | Priced Δms/window | ΔM1 | Basis |
|---|---|---|---|---|---|
| S0 standing | ~2,050 (~2,900) | — | — | — | T13A/F-204 mix estimate |
| S1 conditional | ~1,760 (~2,500) | −270..−290 | −0.16..−0.34 | +1.7-3.6% | A/B marginal × anatomy |
| S2 fusion tranche | ~1,400-1,500 (~2,100-2,200) | −250..−400 | −0.13..−0.40 | +1.5-4% | F-204 deltas, survivors only |
| S3 persistent groups | ~100-150 (~150-250) | −1,300..−1,400 | −2..−4 (honest band; ceiling −6) | +40..80 t/s class | byte-floor gap − dispatch, Cohere-class attainment |
| S4 riders | unchanged (draft graphs re-shaped, not counted here) | — | −0.15..−0.35 equiv. | +2.5-7 t/s | T17A R2-R5 prices |

S1+S2 combined realistic: **−0.3..−0.7 ms/window = +3.5-8% M1** — the bankable tranche. Cumulative
S0→S3: ~2,050 → ~100-150 nodes; the ledger's honest reading is that S3's win is NOT the last
1,300 nodes' dispatch (~0.7-1.3 ms) but the absorption of their execution ramps (~2-4 ms) — which
is why S3 exists as its own stage with its own falsifiers rather than as "S2 done thoroughly".

---

## 4. Risk register (cross-stage)

| # | Risk | Stage | Mitigation / trigger |
|---|---|---|---|
| 1 | Conditional-handle semantics differ from T17D's summary on 13.4/sm_120 (reset flag, body-graph memcpy rules) | S1 | F0 harness FIRST; fail-safe default = run-the-body |
| 2 | Capture-surgery API friction (node matching by func pointer brittle across instantiations) | S1 | Match on kernel func + a capture-time side table (we build the graph; we know the order); abort surgery → plain graph on ANY mismatch |
| 3 | Replay staleness (SageAttention #392 class): flags change between replays | S1/R2 | The parity harness replays with toggled flags; default-reset handles |
| 4 | Fusion slows the merged body (register/smem pressure) — the F-217 lesson | S2 | One arm per fusion, ABAB, fused_gr_selftest-class harnesses before boots |
| 5 | Persistent-kernel hang on a never-rising barrier | S3 | v1 = device-planned groups only; bounded spins + error word + loud host watchdog |
| 6 | sm_120/GB202 CLC unproven | S3 | F6 proto; stage dies cheap |
| 7 | Two persistent cohorts contend (conc lane regression) | S3-b/S4 | S3 serial-only until measured; batch keeps graphs |
| 8 | Pricing drift: standing numbers predate FUSE_HITS/POOL_DIET/COHORTS | all | F0b: one nsys node-trace pass to re-bank count + spacing distribution before S1 surgery |
| 9 | Boot-variance over-read of the dummy A/B's −12.5% | pricing | Price from the window trend (within-boot, robust), not the cross-boot M1 delta; the dossier does |
| 10 | Adjudication collisions (FUSE_WAIT, QFUSE/HC_Q8 in T16 #8) | S2 | S2 consumes their outcomes; no S2 battery touches an unadjudicated gate |

---

## 5. Falsifier ladder (cheapest first; server-free where stated)

- **F0 (S, server-free):** conditional-node harness — 20-node graph, IF+WHILE, satisfied/unsatisfied
  toggled device flags, replay-with-changed-flags, byte-parity vs the plain kernels; measures
  per-site saving on our driver/toolchain. Decides S1's existence and its exact API shape.
- **F0b (S, one profile pass):** nsys `--cuda-graph-trace=node` on the standing recipe (T8.C
  recipe; wall-minus-event accounting per the SGLang #5593 caution) — re-banks node count and the
  spacing distribution post-FUSE_HITS/POOL_DIET. Decides the ledger's S0 row.
- **F1 (free boot):** STRATA_DRAFT_TIMING=1 capture boot (STEP 0 lost the table). Decides R2.
- **F2 (S, server-free):** S1 parity binary (decode_cluster_parity pattern) — both branch
  polarities × both polarities replayed. Gate for the in-tree arm.
- **F3 (XS boot):** LFUSE device-plan arm (one line + env). Prices S2's biggest item.
- **F4 (M boot):** S1 in-tree behind STRATA_CONDITIONAL_NODES=1; ABAB battery; COHORTS byte-equality
  re-run. 
- **F5 (M):** router fusion + small-pairs tranche, offline parity then ABAB.
- **F6 (S, server-free):** 20-line sm_120f CLC compile+run proto under PTXAS 13.4. Decides S3's
  existence on this card.
- **F7 (M, server-free):** 2-layer persistent microbench vs the captured graph slice. Prices S3's
  absorption before any in-tree surgery.
- **F8 (L):** S3 1-layer-group in-tree prototype + parity + ABAB. Then S3-b (cohort contention).

---

## 6. Contradictions and premise corrections (resolved here; do not re-litigate)

1. **"~23 µs/node marginal" (tasking premise, from 1.1 ms / 48) vs the instrument's per-layer
   semantics.** Resolved by code: 48 nodes per LAYER × 48 layers = 2,304 (verify.cpp:1152-1155 +
   :1009 + t17-dum48.json). True marginal 0.48-0.95 µs/node. Physically corroborated: a noop cannot
   cost 4x the ~6 µs average for real nodes. All pricing above uses the corrected marginal.
2. **"Node spacing ~6 µs ⇒ ~2,050 nodes ≈ the whole window" (T12-GPU/T17A framing) vs the A/B.**
   Resolved: the average spacing conflates dispatch (0.5-1.0 µs, A/B-measured) with per-kernel
   execution ramp (the majority) and real waits. The node-collapse tranche re-prices DOWN (~1-2 ms
   total, not 6-7); the persistent-kernel stage keeps its ceiling via the execution-ramp term.
   F-206's law is thereby quantified, not merely respected.
3. **T17D's "more at conc where flags wait longer" for candidate 5 vs E-6 semantics.** Resolved:
   unplanned-group waits are real data dependencies an IF body still pays; the conc gain is
   node-count and map-length, not wait-hiding (§3-S1 caveat).
4. **T13A #4's "check why combine-hits is gated off in standing" vs the launcher.** Resolved:
   STRATA_FUSE_HITS=1 is ADOPTED standing (F-226, launcher:10, M1 +4.35% in its 3v3). The S2 menu
   excludes it.
5. **T17A's "AR window is already slim / nothing left to delete" vs this dossier's S1.** Not a
   contradiction: T17A's "AR" is the autoregressive serial LANE; its slimness = zero doorbells when
   `ar_on()` (all-resident windows only). Standing never satisfies `all_resident_` (38% residency),
   so the E-6 control complement exists and is S1's target. T17A's R1 conclusion (the structural
   lane = S3's class) is unchanged and re-based in §1.

---

## 7. Open questions

1. F0: the exact conditional-handle reset semantics and body-graph rules on our 13.4 headers
   (T17D banked the capability from docs; the flag spellings and capture-surgery behavior are
   prototype facts).
2. F0b: the standing recipe's true node count and spacing distribution (the ledger's S0 row).
3. The unplanned-group fraction per window at standing (drives S1's +1-node regression term and
   S3's fallback frequency). Readable from one `STRATA_VERIFY_PROFILE=1` or skip-stat boot — no
   instrument exists yet; cheap to add beside F0b.
4. F6/F7: CLC on GB202, and the persistent absorption factor (the two numbers S3 lives or dies on).
5. Whether S2's router fusion can absorb `resident_plan` as a second phase without breaking the
   128-thread par kernel's bit-order (verify_kernels.cu:1305-1358) — prototype detail, not priced.
6. Post-F0: does S1's surgery cost stay boot-bounded for the batch graphs' larger bkey set
   (instantiation count × surgery ms)?

---

## 8. Implications

- The R1 lane is REAL and now correctly priced: a bankable +3.5-8% M1 tranche (S1+S2) with
  server-free falsifiers first, and a correctly-based 300-class gamble (S3) whose first two
  falsifiers cost a prototype, not a project. The corrected per-node marginal protects the campaign
  from the expensive error: burning weeks fusing ~1,000 nodes for ~6 µs each that were never there.
- Sequencing: F0/F0b/F1 (all cheap, two server-free) → F3 (LFUSE arm) → F2/F4 (S1) → F5 →
  F6/F7/F8 (S3) → S4 riders ride their stages. Nothing here touches the T16/T17 queue order
  (re-quant → kernel tranche → retrain); R1 is the structural capstone those lanes sequence INTO,
  per the T17 consolidated verdict.
- The COHORTS batch lane inherits S1/S2 free (shared recorder) — the conc lane's launch-map gain
  may exceed M1's; the S3-b contention measurement is the only new conc-specific risk.
- Quality: S1/S2 are bit-identical class (canary gates); S3 v1 is bit-exact by construction with
  the F-230 full gate held in reserve for any grid merge; the No-Regression Law binds every stage's
  battery.

---

## 9. Citations

**Code (read this session, line-verified):**
- src/core/verify.cpp:1152-1155 (dummy-node instrument, per-layer), :1009 (layer loop), :833-857
  (all_resident_/device_plan_ selection), :837-851 (100%-of-table test), :1947/:2082/:2842/:2883
  (stream captures), :929 (lfuse_on ar_on conjunction), :1015-1036 (pre/stage, PLE flag), :1106-1127
  (PLE row tail), :1150-1173 (gr_read_group + mm), :1519-1557 (router chain), :1558-1573
  (resident_plan AR + E-6), :1577-1590 (doorbell forms), :1677-1736 (post: waitA/copy/waitB/staging/
  PCIe-grouped/wait_or/copy_rows/hit_add), :1740-1763 (combine forms), :1937+ (capture), :2338-2360
  (run: HostGraphLaunch), :2489 (:871 plan_err refresh), :2842-2862 (batch capture reuses
  record_window), :3105-3130 (run_slot_rows, #646/#871), :3630-3648 (zero-doorbell service comment).
- src/core/verify.hpp:442-449 (all_resident_, ar_on).
- src/kernels/cuda/verify_kernels.cu:1305-1358 (resident_plan_par_kernel), :1370-1392
  (wait_or/planned/unless kernels), :1441-1445/:1454-1477 (copy_unless/copy_or_zero/
  copy_rows_or_zero), :1480-1496 (wait_copy fuse), :1487-1490 (wait_flag_ge), :1506-1532 (gpu_stamp,
  PDL-aware), :1533-1536 (noop_kernel/gpu_noop).
- include/strata/kernels/verify_kernels.hpp:75-95 (node-cost comment ~2.4 µs gfx1201; E-6/T1.1
  plan contract).
- include/strata/kernels/pdl.hpp:1-52 (PDL discipline, kPdlPrefetch, pdl_wait/trigger).
- src/program/generate.cpp:13113-13119 (solo loop T trim, first_window), :5876-5881/:8106-8127
  (COHORTS binding), :7962-8044 (batch graphs per layout).
- src/kernels/decode_cluster_parity.cpp:1-15 (parity-harness precedent: synthetic + adversarial +
  captured-graph replay).
- campaign/t17-dum48.json / t17-dum16.json ("env": STRATA_DUMMY_NODES 48/16); campaign/t17-leg.sh:53-58
  (env verification law).
- launch-custom-iq3s-8081.sh:8-39 (standing env: FUSE_HITS F-226, POOL_DIET F-227, COHORTS F-233,
  DEVICE_PLAN, ROUTE_RESIDENT 0.75, L2_SETASIDE 60, ASYNC_LAUNCH, TYPICAL_DEVICE, DRAFT_TEMP 0.85).

**Banked facts (dossiers/journal/results):**
- SVG-CUSTOM-TODO.md: "STEP 0 RESULTS" (F-235: ctl 198.0/9.2/13.2; dum16 197.6 flat; dum48 173.3
  −12.5%, late 10.3/early 15.4; dtt table lost; SPEC_PROB closed; parallel-8 dead; MTP-stays) +
  campaign/results/t17a-{ctl,dum16,dum48,dtt}-r1.txt; "T17 CONSOLIDATED VERDICT"; "T12" (GPU audit:
  launch map ~39/52 nodes, ~2,050/window, spacing ~6/10.7 µs, light floor 4.3 vs 12.32, ranked
  moves incl router+plan+bell ~150 nodes, empty-PCIe green field; MEM audit: 48×512 pairs,
  9,337-9,454 slots, roofline); "T13 MASTER QUEUE" (ledger corrections; FUSE_WAIT adjudication);
  "T16" (#1 clock re-bank; #8 adjudicate LFUSE/FUSE_WAIT) ; "T17D" summary block (candidates 4/5
  pricing).
- campaign/research/T17D-sm120-kernel-frontier.md — §2 platform table (conditional nodes CC 9.0+,
  body-graph rules, device-launch limits, CLC CC 10.0+/sm_120f, PDL semantics); §3 candidates 4/5
  rows (pricing + falsifiers); §4.C (CLC deep dive: preemption, precedents, no-5090-bench); §4.D
  (conditional mechanics: cudaGraphSetConditional, body-graph node types, capture surgery); §4.E
  (PDL closed, docs-confirmed); sources S4/S5/S6/S8/S14/S16/S17/S21/S25/S44.
- campaign/research/T17A-m1-single-leap.md — §1/§3 (R1 lane framing, W(1)=8-9 ms vs ~2.1 ms floor,
  Cohere 1.58x [S32], +20 t/s per −1 ms law); §2 (slim-AR facts, draft chain device-chaining, F-229
  bound); §3 R2-R5 rows (prices + falsifiers); §6.4 (F-229 vs R2 resolution); §10 (methods).
- campaign/research/T13A-decode-critical-path.md — §2 (window anatomy table: 11.4-12.0 ms, 2.47
  tok, segment map incl 3h spins 0.4-0.8, pool exposure 0.3-0.7); §3 #4 (always-on empty nodes);
  §4 (head geometry reconciliation, ONEPASS banked).
- campaign/research/F-204-node-inventory.md — per-layer node counts (GDN 39/28, QSA 52/41, window
  ~2,100/~1,580/~1,800 mix; batch ~2,900 at S=6); the −900..−1,000 fusion menu and per-item node
  deltas.
- campaign/research/R2-S3-two-cohort-impl-plan.md — cohort shape, per-verifier buffers, the
  contention question, byte-equality gates.
- Journal laws cited from the TODO index: F-91 (capture-once), F-191 (PDL no-op), F-205/F-206
  (node-fusion law), F-217 (SMEM tile load-bearing), F-222 (flagA RTT 0.26 ms), F-223 (G2 ghost +
  no intra-window hiding), F-225 (boot positions), F-226 (FUSE_HITS adopted), F-227 (POOL_DIET
  adopted), F-229 (draft ungate negative), F-230 (different-bits gate tax), F-233 (COHORTS adopted),
  F-235 (STEP 0).

## Methods note

Read-only session; the only file written is this dossier. Code read: verify.cpp (830-1200, 1360-1900,
1930-1960, 2330-2360, 2840-2870, 3100-3140, 3630-3660), verify_kernels.cu (1300-1540), verify.hpp
(440-450), pdl.hpp (1-80), decode_cluster_parity.cpp (1-40), generate.cpp (13100-13130, 5870-5885,
8100-8130, 7950-8050), t17-leg.sh, t17-dum48/dum16.json, launch-custom-iq3s-8081.sh. Results read:
t17a-ctl/dum16/dum48/dtt legs in full. Dossiers read in full: T17D, T17A, T13A, F-204, R2-S3 plan;
SVG-CUSTOM-TODO.md in full. Unproductive searches (not repeated): no `ar_on()` all-resident boot
exists on standing ( residency 38% ⇒ never taken) — confirmed by reading the residency test, not by
searching logs; no DUMMY_NODES echo line found in server-t17leg-*.out (the /proc environ grep
prints only when the engine process is found; the config json is the authoritative record).
Deliberately NOT done (constraint): INDEX.md not updated — the tasking restricts writes to this
dossier alone; the maintainer should append one line (date 2026-10-19, topic R1 staged design,
conclusion: node tax real at 0.5-1.0 µs/node ⇒ S1+S2 bankable +3.5-8% M1, S3 re-based and
falsifier-gated; status current).
