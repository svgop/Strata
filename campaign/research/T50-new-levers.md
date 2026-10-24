# T50 — NEW LEVERS FOR SOLO-300 AFTER THE F-287..F-295 CLOSURES: the four open areas resolved
# (spec-shape / verify-window / MoE-decode / serving-compose), two small never-priced levers found,
# one owned rung de-priced by a scale correction, and the honest composed arithmetic — 300 sits AT the
# upper edge of everything that survives

Date: 2026-10-26 (wave T50, research agent). Scope: READ-ONLY. No boots, no builds, no edits outside
campaign/research/, no git. Internal claims carry file:line reads THIS SESSION at the working tree
(mtp.cpp, verify.cpp, device.cu, generate.cpp, launch-custom-iq3s-8081.sh) or banked F-entries /
T-dossiers cited verbatim. External sweep: 12 primary sources search-verified this session (§6);
no deep fetch contradicted a carried verdict, so none is re-fetched.
Grading: **[M]** = measured-ours (banked F-entry/T-dossier or read this session); **[S]** =
search-verified external this session (URL given); **[C]** = carried (verified at a prior wave,
re-checked against this dossier's claims); **[H]** = hypothesis (arithmetic shown, constant marked).

---

## §0 SUMMARY (the honest answer first)

1. **All four tasking areas resolve NEGATIVE except two small never-priced levers.** The confidence-
   shaped window family (dynamic depth by dprob, keep/cut rules, DSpark-class scheduling) is closed by
   THREE independent banked instruments (F-228's offer-stair, F-292's data-empty refit, T26-2's DSpark/
   varlen kills); trees are dead at our acceptance (chains dominate at p ≈ 0.77-0.89, and our graphs are
   row-flat); the "graph relaunch ≈ 35% of window time" premise is a TASKING ERROR (banked solo anatomy:
   83-88% of the window is the in-graph GPU walk, spins+gaps 4-7%, tails ~9.5% — the 35-50% figures live
   on the CONC round wall, T47-conc §1); row-batched MoE across a solo window's rows is structurally
   impossible (the chain is serial); and the 2026 MoE×speculation literature (Expert Budgeting, MoESD,
   SP-MoE, self-speculative-MoE) binds on expert-offload or budget constraints we do not have, or bends
   the TARGET = law-10 dead.
2. **The two new levers (both riders on builds that must happen anyway, both user/quality-gated):**
   (a) **target output-weight requant rider** — the 521.5 MB Q6_K `output.weight` is read ~0.45-0.78
   ms/window and the T26-5 alpha requant explicitly leaves it `=COPY`; a Q4-class pack saves ~0.14-0.24
   ms/window ≈ **+1-2% M1** [H], priced offline FIRST with the F-294b headport harness (the real tensor
   + CPU reference are already extracted in campaign/headport/); (b) **per-position draft-temperature
   schedule** — the drafter's global DT 0.85 (F-201, +13.4% at adoption) is the one C-lane knob no
   instrument closed; deep positions accept 53.1% vs 93.6% shallow (F-235), so a position schedule
   (shallow ≈ 1.0, deep sharper) is untried: **+0-2% M1** [H], one battery, S code. Bands are small —
   neither closes the gap alone.
3. **One owned rung DE-PRICED by a scale correction:** F-288's "next rung = persistent-L2 access-policy
   window pinning w_down/w_up" assumed ~13 MB of hc weights. That is the PER-LAYER size; 48 layers
   carry ~630 MB against a measured 60 MiB persisting carve (device cap, F-276) and ONE policy window
   per stream (device.cu:375-431). Pinning ≤ 4-5 layers saves ≤ ~0.06 ms/window ≈ **+0.3-0.5% M1 —
   below the five-boot bar. NEGATIVE by arithmetic** (§4.5); the hc line's honest residual is F-289's
   un-run wide-grid occupancy form (§3.3).
4. **The composed ladder (§3 table): central ≈ 271, upper ≈ 296** from B = 217.3 — retrain v2 (1.06-1.09)
   × requant GO (1.04-1.07) × head-requant rider (1.01-1.027) × PDL chain (1.01-1.04) × diet+micro
   (1.02-1.04) × DT schedule (1.00-1.02) × depth re-check (1.00-1.03). **300 therefore requires
   upper-band landing on nearly every rung simultaneously, or a C-lane gift beyond the retrain — and
   this sweep found none.** The gap's honest home remains retrain quality (it owns the deficit pool,
   F-292) and the requant GO (the only lever that moves the byte floor). Every sub-4% item is
   five-boot-pooled territory (F-263 law); the ladder's risk is sequencing, not search.
5. **Nothing cheap is left — say it once and stop looking:** after F-287..F-295 plus this sweep, every
   remaining item on the solo lane is either (i) a training/quant build with a user gate (retrain,
   requant, head rider), (ii) a multi-file kernel/scheduling tranche with a pre-declared bench or nsys
   gate (PDL chain, diet, wide-grid down, QSA skip), or (iii) a one-boot dial inside an already-adopted
   arm (DT schedule, depth re-check, TYPICAL threshold). There is no zero-build lever above noise on
   this board.

---

## §1 OUR-STATE VERIFICATION (anchors read this session)

### 1.1 The solo window anatomy the bands hang on (carried T39 §1.6, re-checked for contradictions)

| class (committed rows) | ms/window | run (graph span) | GPU-reach wait | pool | commit | emit | adapt |
|---|---|---|---|---|---|---|---|
| ~3.2 rows (M1) | 13.41 | 12.14 | 8.97 | 2.51 | 0.02 | 0.57 | 0.68 |
| ~5.0 rows (pair) | 13.68 | 12.41 | 0.46 | 2.09 | 0.34 | 0.36 | 0.57 |

Row-FLAT (11.7-13.8 ms from 3.2 to 5.1 rows; marginal row 0.2-0.5 ms); inside the graph: expert hits
1.9-3.5 ms (byte-floored 87-92%), hc 2.0-2.9 ms (CLOSED, F-290), head 0.45-0.78 ms (two-stage DEAD,
F-294b), attention ~0.42, router/plan/bell ~0.7, spins+gaps 0.4-0.8 [M: T13A §1 row 3h + T39 §1.6].
W for the record config = 12.29 ms (T47 §1.1); C = 2.734 rescue-inclusive banked, fresh-trace 2.232 of
3.935 offered [M: F-260/F-292].

### 1.2 The launch cadence per solo window (read this session; the graph surface is at its floor)

- **Draft chain**: `mtp.cpp:1670-1766` — round graph (`round_exec_[T]`) then per-step graphs
  (`step_exec_[j]`) HOST-GATED by the min-p trim (`--spec-min-p 0.5`, config); the gate trims the chain
  at ~2.2 of 6 steps on average; a step graph costs ~50 µs; the UNGATED twin (enqueue-all) measured
  **−2.73% M1** and is banked closed (F-229) — the launch bubbles are already smaller than the waste.
- **Verify window**: one `cudaGraphLaunch(exec_[T])` through HostGraphLaunch (F-200), after
  instantiate+upload at capture (`verify.cpp:2636-2639`, `:3572-3575`, same-stream upload per the CUDA
  guide's caveat — correct by construction [C: T39 §2 G1]); families captured LAZILY per seen T
  (`verify.cpp:2512-2514`) — exec_[7]/exec_[8] cost ZERO when uncaptured [M: T39 §1.4].
- **Commit**: solo single-slot self-commit measures **0.02 ms** (the M1-class row, §1.1) — commit is
  NOT a solo lever (the 0.34 ms `commit_slot_prefixes` graph is batch-lane).
- **Node replay marginal 0.48-0.95 µs/node × ~1,860 nodes ≈ 0.9-1.8 ms/window** — the diet tranche-3
  (FUSE_WAIT/DF_PLE/UseNodePriority, −200 nodes ≈ −0.10-0.19 ms) and the PDL chain extension
  (T26-3: −0.5..−1.5 ms for the FULL 12-file chain, nsys-gated) own this term; both carried [M].
- **PDL standing state**: `STRATA_DF_PDL=0` pinned in the launcher (F-279); the code default is ARMED
  (`verify_kernels.cu:2084`) with only ~13 of ~1,860 nodes on the armed surface = ≤0.02 ms [C: T39 §1.2].
- **L2 machinery**: `device.cu:375-431` — `l2_setaside_arm` takes ONE base+span, sets ONE
  accessPolicyWindow per stream (hitRatio = window/span), persisting carve capped at the device's
  measured 60 MiB (F-276 falsified 72); live arm pins the GDN state over a 112 MiB span at hitRatio
  0.53 (`generate.cpp:9300-9305`, boot line "60 MiB window / 60 MiB carve over a 112 MiB span") [M].

### 1.3 THE TASKING CORRECTION: "35% of window time outside kernels" is not a solo-window fact

Banked instruments: T13A's node-level map reads **83-88% of the window = the GPU's serial walk through
the graph**, with waitA/waitB/waitCPU flag spins + gaps at **0.4-0.8 ms (4-7%)**; commit+emit+adapt
tails add ~0.9-1.3 ms (~9.5%) (§1.1). The 35-50% non-window figures are the CONC ROUND WALL (three
finish tails, admission stops, adapt join — T47-conc §1), a different denominator. **Consequence: no
"graph relaunch overhead reduction" program exists on the solo lane** — the launch terms are already at
their floor (instantiate-once/upload/HostGraphLaunch/0.48-0.95 µs/node, T39 §1.5 §2 G1-G2), and the
only structural graph lever (conditional nodes / device launch) prices at **+0.3-0.9%** with its
consumer rule dead (T26-2: "conditional draft graph… band drops to +0.3-0.9%"; F-292 killed the rule).
Device-graph-launch limits (120/255/1) bind device-launched graphs only and do not change this [C:
T37 S11].

### 1.4 The C-lane instruments (why the confidence-shaped-window family is closed)

- **Offer side (extend)**: F-228's stair — at min-p 0.5, depths 3/4/5 are offered at 35/17/8% of
  windows accepting ~85/68/75%; the min-p **0.35** re-stair (offering MORE rows everywhere) measured
  **−1.75% M1 3v3** and is "CLOSED for flat gate-tuning" [M]. F-263 killed depth 7/8 as *window width*
  [M]. Deep-position acceptance 53.1% vs 93.6% shallow (F-235) is WHY more offers lose: the marginal
  deep row's acceptance ≈ 0.5-0.6 against a 0.2-0.5 ms row + step cost + the F-229 waste mechanism.
- **Cut side (keep)**: the keep-rule refit is **data-empty** — the fresh 3,265-window SPEC_PROB trace's
  decile curve (0.554/0.614/0.610/0.689/0.821) is a property of the drafter-model pair; kept-mean ≈
  offered-mean held; "the deficit pool's lever is RETRAIN v2" (F-292) [M].
- **Trained confidence heads**: DSpark killed serve-side — "dprob is the free confidence; the flat gate
  already consumes it"; DeepInfer varlen killed single-stream (flat c_row — nothing to arbitrate);
  both adjudicated T26-2/INDEX [M].
- **Rescue**: TYPICAL_ACCEPT 0.2 + TYPICAL_DEVICE adopted (F-200/R5: +20.6 pts accept / +8% M1;
  51% of first-rejections rescued live); SPEC_PROB adopted (F-282, +5.8% M1). These are the adopted
  C-lane arms; their DIALS (threshold 0.2; global DT 0.85) were set by one sweep each (DT 0.45-0.90
  peaked 0.85, F-201) and never re-swept post-SPEC_PROB — the one gap §3.2/§3.5 price.

---

## §2 THE FOUR RESEARCH AREAS RESOLVED

### A1 — Speculative depth/window shaping beyond the depth-6 chain

- **Tree windows under the 8-row budget — NEGATIVE.** Two banked kills stand (TreeWY 0.97-0.99x vs
  chains on chain-verifying stacks; SpecLA <1x at p ≤ 0.6 [C: T35/T47]), plus the arithmetic: under
  iid per-row acceptance p, a linear chain of R rows yields E = p(1−p^R)/(1−p); at our p ≈ 0.77-0.89
  (mean 2.232-2.734 accepted of ~4-6 offered) the chain is within ~2-5% of the optimal R-row PREFIX
  policy, and any tree spends rows on siblings that only pay when the chain head rejects — at high p
  the chain head rarely rejects. EAGLE-2's dynamic-tree gain (context-aware expansion, 3.05-4.26x
  speedup regime, arXiv 2406.16858 [S]) decomposes into an EXTEND half (spend rows on confident paths)
  and a BRANCH half (spend rows on uncertain alternatives); the extend half is exactly our min-p gate
  (already in-tree, F-228's stair measured its optimum at 0.5), and the branch half has no bubble to
  hide in a row-flat graph and pays tree-verify row overhead. **No new form survives.**
- **Dynamic per-position depth by dprob — NEGATIVE (three instruments, §1.4).** The offer stair, the
  keep-rule refit, and the DSpark kill triangulate the same fact: per-draft dprob is already consumed
  by the free gate, its conditional information about acceptance is weak (0.55→0.82 across extreme
  deciles), and every re-allocation of rows along it measured ≤ 0 or data-empty. External 2025-26
  additions (vLLM DSL RFC #36657 — shelved upstream [S, carried]; HF dynamic_speculation_lookahead
  scout [S]; ACL 2025 effective-draft-tree pruning [S] = cut-side; "Speculative Verification" ACL
  2026 findings [S] = predict-before-accept = cut-side) all land in the same two dead families.
  Re-open only via the retrain (drafter quality shifts the curve itself — F-292's own conclusion).
- **Position-dependent draft temperature — the survivor (§3.2).** The external temperature literature
  is about the TARGET's sampling temperature (arXiv 2410.10141: speedup peaks near decoding temp 0.2
  [S]) — a different knob. The DRAFTER's proposal-temperature schedule across positions is unpublished
  territory and untouched in-tree (global DT 0.85 only, F-201).

### A2 — The verify window itself (row layout, commit/emit, graph relaunch)

- **Row-layout alternatives within kVerifyMaxT**: 3-drafts/slot and mixed-depth-across-slots are
  BATCH-lane shapes (T47-3 owns the second-row form at kVerifyMaxT 8→9; conc lane, not solo-300).
  On the serial lane the window rows are a causal chain — there is no layout freedom beyond depth
  (dead, §A1) and the tip policy (SPEC_PROB owns it, adopted).
- **Commit/emit path**: solo commit = 0.02 ms (self-commit skip, §1.2) — dead. Emit 0.35-0.61 ms
  includes draft_finish waits; the F-204 emit-batching map and dm fold already consumed the foldable
  share; the residue is per-token BT emits + fflush (host, ~0.3 ms class) — five-boot-noise territory,
  owned by the micro bundle at best.
- **Graph relaunch overhead — the premise is wrong (§1.3).** The graph surface is at its floor; the
  conditional-node/device-launch machinery would serve a dead rule at +0.3-0.9%. The ONLY live items
  on this lane are the carried diet tranche-3 and the PDL full chain (both §3.4).

### A3 — MoE-specific decode wins

- **Batching the routing across a window's rows — structurally impossible solo.** Row r's layer-l
  expert set is known only after row r's hidden state reaches layer l, which requires row r−1's whole
  forward (the chain is serial). The only already-batched components are those whose inputs are
  complete before the walk (the verify head's mmvq IL tables read 521 MB once for all T rows; the hc
  grouped reads take T). On the BATCH lane rows are independent across slots — union-sharing there is
  real but it is conc-lane work (T47/T42's batch-QSA suite territory).
- **The 2026 MoE×speculation literature — wrong constraint or non-exact.** Expert Budgeting
  (arXiv 2602.16052 [S]) budgets the expert union during multi-token verification — a TARGET-side
  constraint = law-10 dead for us; self-speculative MoE (arXiv 2608.15018 [S]), MoESD (OpenReview
  [S]) and DraftExpert (arXiv 2607.24434 [S]) target expert-OFFLOADED tiers where the extra activation
  per draft token is the binding cost — our experts are ~99% resident and byte-floored (hit-rate
  floor argument, T47 §4.7); SP-MoE's prefetching (arXiv 2510.10302 [S]) dies on the same floor. The
  mlx-lm #1132 report (spec HURTS MoE via union growth [S]) is the union-growth term we already
  measure inside the row-flat marginal (0.2-0.5 ms/row).
- **L2 persistence for hc weights — de-priced (§4.5).** The F-288 "next rung" assumed a ~13 MB pin;
  the true working set is 48 × ~13 MB ≈ 630 MB against a 60 MiB carve and a one-window-per-stream
  API — **+0.3-0.5% ceiling [H]**, below the bar. Negative by arithmetic.
- **The surviving hc form is owned**: F-289's wide-grid occupancy instantiation for the down line
  (the one axis the four falsifications did NOT vary — grid shape at T ≥ 4) + the mma rung at
  true-cold (four flats price it low) [C].
- **KV read path at 131k — wrong lane for this record.** Standing probes run ~6k context; KV is int8
  (`--kv int8`, config); the attention line is 0.42 ms at that length. Sparse/quest-class KV read is
  non-exact on the target or quality-gated, and no 300-record leg runs at 131k. Scoped out (§4.6).

### A4 — Serving-world composition with coupled MTP at single-GPU scale

- Re-checked this session: the field's live levers are one-step-ahead host overlap (SGLang v0.4/
  Spec-V2/DFlash, vLLM async_scheduling, TRT-LLM overlap — our C2, conc lane [C: T47-conc S1]),
  confidence-scheduled verify length (DSpark — killed here, dprob free), dynamic spec length per
  batch (vLLM RFC shelved; per-slot dprob is our better signal — carried T37 S12), piecewise/breakable
  CUDA graphs (SGLang BCG — capture-time wins for dynamic shapes; our fixed-shape families are
  already the no-padding design it converges to [C: T39 G4]), PDL (owned, sm_120-supported [C: T47
  S4]). **No new composable mechanism found for the coupled-drafter single-GPU solo lane** — the
  serving world's wins are host-overlap and batching economics; our solo lane has neither in its
  critical path (§1.1: the graph replay is 90% of the window).

---

## §3 CANDIDATES (value-ranked, falsifier-first) — the composed ladder

| # | lever | lane | band (M1) | central × | upper × | owner / falsifier |
|---|---|---|---|---|---|---|
| B | certified base | — | 217.3 (σ 2.2) | — | — | F-293 |
| 1 | retrain v2 (T14B; owns the deficit pool) | C | +6..9% | 1.075 | 1.09 | T14B/T31; trainer flags carry LK/AdaSPEC/replay-rejection/fake-quant |
| 2 | experts requant GO (user-gated) | W | +4..7% | 1.055 | 1.07 | T17F #1; G0.1 bit-identity + m̂ gates |
| 3 | **output-weight requant rider (NEW)** | W | +1..2% (to 2.7 at IQ3-class if the gate allows) | 1.015 | 1.027 | §3.1; offline KL via headport FIRST |
| 4 | PDL full chain + diet tranche-3 + micro bundle | W | +2..4% composed (T26-3 stretch −0.5..−1.5 ms if nsys gates pass) | 1.03 | 1.04 | T26-3/T39 A5/T42 §5 |
| 5 | **per-position DT schedule (NEW)** | C | 0..+2% | 1.01 | 1.02 | §3.2; one battery, DT-machinery rider |
| 6 | depth re-check post-SPEC_PROB (carried) | C | 0..+3% | 1.015 | 1.03 | F-285 rider; F-228 stair instrument |
| — | **COMPOSED** | | | **≈ 271** | **≈ 296** | arithmetic below |

Arithmetic — central: 217.3 ×1.075 = 233.6 → ×1.055 = 246.4 → ×1.015 = 250.1 → ×1.03 = 257.6 →
×1.01 = 260.2 → ×1.015 = **264.1** (with #4 at its +2% floor and every other rung central). With #4 at
+4% and #6 at +3%: 246.4 ×1.015 ×1.04 ×1.01 ×1.03 = **270.8**. Upper: 217.3 ×1.09 = 236.9 → ×1.07 =
253.4 → ×1.027 = 260.3 → ×1.04 = 270.7 → ×1.02 = 276.1 → ×1.03 = **284.4**; admitting the T26-3 PDL
stretch (−1.5 ms ⇒ ×1.11) instead of +4%: 284.4/1.04×1.11 ≈ **303.6**. Read honestly: **central ≈
264-271, upper ≈ 284-296; 300 requires the PDL-chain stretch PLUS upper-band retrain AND requant AND
the rider — i.e. everything lands high at once, or the retrain beats its band.** No double-counting
found among these rungs (the rider's bytes are disjoint from T17F's expert/dense sets; the DT schedule
spends drafter entropy, not the retrain's acceptance pool — but if BOTH land, the post-retrain DT
re-sweep (T39 §3.3: DT→1.0 at q=p) subsumes #5 — sequence #5 BEFORE the retrain or not at all).

### 3.1 Output-weight requant rider (NEW; the best new item)

- **Mechanism.** The verify head `output.weight` [2560×248320], Q6_K, 521,472,000 bytes exact (F-294b's
  own extraction), is read once per window at 0.45-0.78 ms (≈ 0.7-1.1 TB/s, byte-floored class). The
  T26-5 alpha requant leaves `token_embd/output/head =COPY` — the exclusion was draft_vocab-binding
  conservatism (the DRAFTER's vocab subset binds to the target's token IDs; requanting the target head
  changes logits numerics, not IDs), never a measured verdict. A Q4_K-class pack (4.5 bpw vs 6.5625)
  saves 31% of the read ≈ 0.14-0.24 ms/window ≈ **+1.0-1.9% M1**; an IQ4_XS/IQ3-class pack reaches
  +1.3-2.7% with proportionally more KL risk. Full-vocab sampler untouched (all 248,320 logits still
  produced — this is a PACK change, not a shortlist).
- **Falsifier-first (zero boots).** The headport artifacts from F-294a/b ARE the harness: the real
  tensor is already extracted verbatim (campaign/headport/), the CPU Q6_K reference and copy-fit
  machinery exist. Add a requantize-then-score leg: pack candidate → dequant → per-row top-K overlap
  + KL vs the Q6_K reference over the banked activation dump → the same ~1e-4-class gate discipline
  the proxy head failed, now against a floor we control. GO criterion: KL ≤ the T17F requant gate's
  class AND top-1 flip rate ≤ the alpha-requant precedent; then the F-230 ladder (3v3, 2× MMLU,
  before/after single+conc proof).
- **Cost anatomy.** Requant tool: one-tensor assignment in the T17F tool run (the pack is per-tensor
  typed; the load path already dispatches per-tensor types — `mmq::fits` dispatch is per-layer/per-type
  today [C: T47 §3.7's own note]). Rider on the T17F GO build — never block the uniform requant on it.
- **Law-10 frame.** Identical class to the experts requant (the engine IS the modified target;
  self-consistent exactness; quality via MMLU gate). User-gated with the experts GO.

### 3.2 Per-position draft-temperature schedule (NEW; the last live C-lane dial)

- **Mechanism.** Global DT 0.85 was swept ONCE pre-SPEC_PROB (0.45-0.90, peaked 0.85; F-201, +13.4%
  M1 at adoption). Acceptance is strongly position-dependent (93.6% shallow vs 53.1% deep, F-235) and
  the optimum proposal temperature is position-dependent too (q→p locally; one global DT cannot track
  both regimes). A per-step temperature vector (shallow ≈ 0.9-1.0, deep 0.7-0.85) is a candidate.
- **Cost anatomy.** The DT is baked into the captured step graphs' sampling constants; the schedule
  needs the temp read from device memory per step (a kVerifyMaxT-length buffer) — S code at
  `sampler.cu:891` (the SPEC_PROB site) + the step-graph capture in mtp.cpp. Env-gated
  `STRATA_SPEC_DT_SCHED="0.9,0.9,0.85,0.85,0.8,0.8"`, default-off.
- **Falsifier-first.** One 3v3 per candidate schedule (two schedules max); M3b + golden hash on every
  leg (distribution-adjacent to proposals only — exactness preserved by rejection sampling as today).
  GO bar: pooled ≥ +1% M1 with M3b neutral. **Sequence before T14B** (the retrain re-opens DT and
  subsumes this — T39 §3.3); if the retrain is imminent, skip.
- **Band honesty.** 0..+2% [H]: the F-228 stair says the chain averages 2.2 steps, so only the first
  2-3 positions' temps matter much; the win is bounded by how far global 0.85 is from per-position
  optima — unknown, plausibly ~1%.

### 3.3 Wide-grid staged2b down (owned-unrun; the hc line's honest residue)

F-289's own text: the down line's binder at T ≥ 4 is the 41-block grid occupancy, and staged2's
per-thread staging removed the coupling that killed the F-15C wide family — "a 4-warp/82-block staged2
is the un-run form". One `fused_gr_bench` instantiation (the bench and 48-set cold mode exist), engine
A/B only if the bench moves at the T=4-8 rows. Band **0..+3% M1** [H: down ≈ 1.2-1.5 ms of the hc
2.0-2.9 ms line; a 1.5-2x at T ≥ 4 saves ~0.3-0.6 ms/window]. The mma rung stays de-priced (four
flats, F-290).

### 3.4 Carried W-lane rungs (recited, not re-priced — the ladder's mass)

- **Retrain v2** +6..9% (T14C banked; trainer flags already carry the EAGLE-3-class recipe space —
  T47 §3.3's alignment note stands: keep `--fake-quant` + `--replay-rejection` in the first run).
- **Experts requant GO** +4..7% (T17F #1; the largest pending decision; frequency-tiering is its v2,
  T47 §3.7).
- **PDL full chain** (T26-3: −0.5..−1.5 ms, 12-file trigger/staging work, nsys per-pair probe as entry
  gate, sequenced behind the diet) + diet tranche-3 (FUSE_WAIT −144, DF_PLE −56, UseNodePriority;
  −0.10-0.19 ms) + micro bundle (QSA below-2051 skip, GDN bf16 state KL-gated, `--ple-io ram`,
  PDL-race screen) — composed +2..4% central, the T26-3 stretch as upside.
- **Depth re-check post-SPEC_PROB** 0..+3% (one boot on the next battery window; the F-228 stair
  instrument reads it).
- **TYPICAL threshold dial** (0.2 → 0.25/0.3, config-only): distribution-ADJACENT (the rescue is the
  one adopted arm that bends the emitted distribution — F-200 parity-proven at 0.2; any change is an
  F-30b-class disclosure + MMLU gate). Never swept past the on/off. **+0-3%, user-gated, NOT
  exact-class** — list it for completeness; do not book it in the exact ladder.

---

## §4 NEGATIVES (each with its reason and re-open condition; saves boots)

1. **dprob-gated dynamic depth, extend OR cut.** Extend: min-p 0.35 stair measured −1.75% M1 (more
   offers lose at 53% deep acceptance); depth 7/8 dead as width (F-262/F-263). Cut: keep-rule refit
   data-empty (kept-mean ≈ offered-mean on 3,265 fresh windows); DSpark killed (dprob free);
   Speculative-Verification/information-gain class (ACL 2026) = cut-side, same grave. Re-open: only
   through retrain-quality shifting the decile curve (F-292's conclusion). [M ×4]
2. **Tree windows under the 8-row budget.** Chain-optimal at p ≈ 0.77-0.89 (§A1 arithmetic); TreeWY/
   SpecLA carried kills; branch parallelism has no bubble in row-flat graphs; EAGLE-2's dynamic tree
   decomposes into our existing min-p gate + a dead branch half. Re-open: never under a chain-verify
   row-flat window; revisit only if acceptance drops below ~0.6 AND graphs become variable-shape. [C+H]
3. **"Graph relaunch ≈ 35% of window" as a solo lever program.** Premise corrected: 83-88% in-graph
   walk, spins+gaps 4-7%, tails 9.5% (§1.3); the graph surface is at its floor (instantiate/upload/
   HostGraphLaunch/0.48-0.95 µs/node); conditional-node machinery prices +0.3-0.9% with its consumer
   rule dead; device-launch limits don't bind host-launched graphs. The 35-50% figures belong to the
   CONC round wall (C2's program, T47-conc). Re-open: a measured >15% in-graph gap share on a fresh
   nsys of the CURRENT composition (the A1 census boot's rider). [M]
4. **Row-batched MoE / expert-union sharing across a solo window's rows.** Structurally impossible on
   a serial chain (routing needs the row's own hidden state); the 2026 budgeting class (2602.16052,
   MoESD, 2608.15018, SP-MoE, DraftExpert) constrains the TARGET (non-exact) or targets offloaded
   tiers (we are 99% resident, byte-floored); expert prefetch dead at the floor (T47 §4.7). Re-open:
   on the conc lane only, as batch-window union batching (owned question there, not here). [M+S]
5. **Persistent-L2 pin for hc weights (F-288's "next rung").** Scale correction: the ~13 MB is
   per-layer; 48 layers ≈ 630 MB vs the measured 60 MiB persisting cap (F-276) and ONE policy window
   per stream (device.cu:375-431; attributes are static across a replay). Ceiling = (4-5/48 layers)
   × 13-21% × 2.0-2.9 ms ≈ 0.03-0.06 ms ≈ **+0.3-0.5% M1** — below the five-boot bar. Re-open: only a
   mechanism that re-attributes persistence per phase inside one replay (none exists in the CUDA
   surface — accessPolicyWindow is a stream attribute consumed at capture/launch, not a runtime
   schedule). [H, arithmetic shown]
6. **KV long-context decode levers for THIS record.** Standing record lanes run ~6k context; KV int8
   standing; attention line 0.42 ms there. Quest/sparse KV = non-exact or quality-gated and off-lane;
   KV requant below int8 = quality-gated user item with no record-lane payoff. Re-open: if the
   campaign adds a long-context record lane. [M]
7. **Draft-temperature GLOBAL re-sweep post-SPEC_PROB.** Subsumed by §3.2's per-position schedule
   (the schedule's sweep reads the global value for free); do not spend a separate battery on a
   scalar re-sweep. [M]
8. **Carried negatives re-confirmed, one line each** (do not re-buy): two-stage/proxy head (F-294b,
   flat real-tensor tops); hc schedule/prefetch/reassociation forms (F-287..F-290, four flats +
   DRAM-cold closure); fp16-accum/Ozaki compensation (bytes-bound lines, T47 §4.2); NVFP4 repack
   (byte-regressing ≥4 bpw); draft-shortlist precision cuts (34 µs/row); n-gram draft source
   (F-291, drafter beats the oracle); CCD/thread-placement (F-295, 0-for-2); whole-process taskset
   (−20%, F-286); 12-wide ternary for solo (solo crushed 153.5, F-286); DRAFT_UNGATED (F-229);
   cudaGraphExecUpdate-as-mechanism (topology-fixed families, T39 §4.4); profile-boot re-baseline
   (poisons the folds, T39 §4.1).

---

## §5 OPEN QUESTIONS (ranked by what they gate)

1. **Retrain v2 timing** — it owns the C pool, subsumes §3.2 and the keep-rule family, and re-opens
   DT→1.0 + the UNGATED re-test (F-229's conditional re-open) for free. Every week it slips, the
   ladder's central case dies. (User decision on the T14B run.)
2. **Requant GO** (experts + §3.1 rider in ONE tool run) — the byte-floor lever; the headport offline
   KL leg prices the rider before any boot.
3. **The A1 census boot (T39) with an nsys rider** — settles the post-dm node marginal AND the honest
   in-graph gap share (the §4.3 re-open test) in one diagnostic.
4. **Depth stair post-SPEC_PROB** — one battery-window boot (§3.4).
5. **Wide-grid staged2b bench instantiation** — one afternoon, gates the last hc form (§3.3).
6. Does the F-235 deep/shallow acceptance split (93.6/53.1) shift under the fresh recipe? (The
   F-292 trace can answer offline; if deep acceptance rose materially, §3.2's band widens and the
   depth re-check brightens — same boot, free read.)

---

## §6 SOURCES

### Local (read this session)
`src/core/mtp.cpp` (:1670-1766 round/step host-gated launch chain, min-p trim, dt telemetry);
`src/core/verify.cpp` (:2512-2514 lazy capture, :2636-2639/:3572-3575 instantiate+upload, :2913/:2988
exec launches, :3459/:3995 commit launches, :787 max_t gate, :3211-3222 SPEC_PROB device q-lists);
`src/core/device.cu` (:375-431 l2_setaside_arm/disarm — one window per stream, carve cap);
`src/program/generate.cpp` (:9300-9305 GDN-state pin call, :354 typical_threshold, :363-366
BATCH_RESCUE, :11814/:12122-12155/:12432-12443 rescue paths, bt_run/bt_commit/bt_emit :11775/:12467-
12551); `src/kernels/cuda/verify_kernels.cu` (:2084 PDL default); `launch-custom-iq3s-8081.sh` (:18
COHORTS, :19 TYPICAL_ACCEPT, :32 DF_PDL=0, :33 SPEC_PROB, :40 L2_SETASIDE, :45 TYPICAL_DEVICE, :47
SPEC_DRAFT_TEMP, :53-54 ROUTE_RESIDENT); config `custom-iq3_s-8081.json` (--spec 6 --spec-min-p 0.5
--kv int8).
Banked: SVG-CUSTOM.md F-229 (:7871), F-228 (via T26-2 §4), F-235 (deep-position 53.1/93.6), F-260
(W=12.29), F-263/F-265, F-270, F-276 (L2 cap 60), F-279 (PDL pin), F-282 (SPEC_PROB), F-286-F-295;
T13A-decode-critical-path.md (node map: 83-88% graph walk, hc 2.9, expert hits 1.73 at 87-92%
floor, spins+gaps 0.4-0.8); T39-window-anatomy-deadweight.md (§1.2 PDL default flip, §1.4 lazy
families, §1.6 TREND table, §1.8 diet census, §2 G1-G4 CUDA graph surface, §4 negatives);
T26-2-spec-lane.md (§4: keep-rule = 8 bins prod(acc) ≥ th; DSpark/varlen kills; conditional graph
+0.3-0.9%; F-228 stair numbers); T26-5-numerics-quality.md (:54 `token_embd/output/head =COPY`,
draft_vocab binding); T47-solo-300.md + T47-conc-450.md (composition bands, conc round-wall
decomposition, S1-S9); T42-knowledge-check.md (merged queue: QSA skip, bf16 state, miss-aware verify
width, the 71% U(P) band); F-294a/b headport artifacts (campaign/headport/ — real tensor extracted,
CPU reference, copy-fit harness); campaign/t48-spec-trace.txt (3,265-window fresh trace).

### External (search-verified this session unless marked carried)
1. EAGLE-2 — arXiv 2406.16858 (context-aware dynamic draft trees; confidence-ranked expansion) [S].
2. vLLM RFC #36657 "Dynamic Speculation Length with Confidence" (Mar 2026; shelved — carried T37
   S12) [S/C].
3. HF `dynamic_speculation_lookahead` (scout oracle, Oct 2024) [S]; DISCO (Schwartz lab, classifier
   dynamic lookahead) [S].
4. "Faster Speculative Decoding via Effective Draft Tree" (ACL 2025, threshold pruning = cut-side)
   [S]; "Exploiting Information Gain / Speculative Verification" (ACL 2026 Findings
   2026.findings-acl.1809) [S].
5. DSpark — arXiv 2607.05147 via DeepInfer (T23-deepinfer.md; killed here T26-2) [C].
6. "Temperature-Centric Investigation of Speculative Decoding" — arXiv 2410.10141 (target-temperature
   study; peaks near 0.2) [S].
7. "Expert Budgeting for Efficient Speculative Decoding" — arXiv 2602.16052 [S]; "Enabling Efficient
   Self-Speculative Decoding for MoE" — arXiv 2608.15018 [S]; SP-MoE — arXiv 2510.10302 [S]; MoESD
   (OpenReview FAeU7516MR) [S]; DraftExpert — arXiv 2607.24434 [S]; mlx-lm issue #1132 (spec can hurt
   MoE via union growth) [S].
8. CUDA graph surface (device launch, conditional nodes, upload semantics, per-node cost classes) —
   carried from T39 §2 G1-G3 (CUDA 13.4 Programming Guide; NVIDIA device-launch 2022-12 and
   conditional-nodes 2024-05 blogs; arXiv 2501.09398) — re-checked for contradictions, none found [C].
9. Carried external frame: SGLang CUDA-graph techniques + vLLM padding economics (T39 G4); PDL on
   sm_120 (T47-conc S4); SMOE/IPDPS'26 + public 5090 frontier (T47-solo §4.8).

### Arithmetic honesty note
Every band is a banked band or carries [H] with its constant shown. The two genuinely new numbers
this session are measurements: the `=COPY` exclusion of the output head (T26-5 :54) and the one-
window-per-stream L2 arm shape (device.cu:375-431) — the latter de-prices F-288's next rung by the
48-layer scale fact (§4.5). The composition's central (264-271) and upper (284-296) are products of
carried bands; the 300 crossing exists only through the T26-3 PDL stretch plus upper-band retrain
and requant, and the dossier says so rather than smoothing it.
