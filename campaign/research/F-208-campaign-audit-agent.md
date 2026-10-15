# F-208 — CAMPAIGN AUDIT: missed high-value items and mis-prioritizations (audit agent, 2026-10-15)

**Scope audited:** `campaign/ROADMAP-30-VS-STOCK.md`, all 18 files in `campaign/research/`
(F-200-\*, F-201-\*, F-204-\*, T1.1-\*), journal entries F-173..F-207 in `SVG-CUSTOM.md`
(plus targeted greps back to F-118), `AGENTS.md`, `campaign/probe-lane.py`, git log through
74bff7a. Read-only; no builds, no benchmarks, no server contact.

**Standing baseline used throughout:** stock v0.1.44 = M1 147.1 / pair 149.1 / 6-conc 187.7
(F-198 window 2). Current warm class = 179.9-187.8 / 171.0-175.0 / 210.1-219.1
(F-203/F-205 controls). Targets ×1.30: 191.2 / 193.8 / 244.0.

---

## (A) HIGH-VALUE DROPPED ITEMS

Each entry: source → what → expected effect → why it likely got dropped → confidence.

**A1. The F-200 Tier-0 battery's non-adopted arms were never voided and never re-queued.**
Source: F-200 results table (SVG-CUSTOM.md:6908-6917) vs F-205's explicit void list
(:7185-7195). F-205 voided exactly three things: F-203's two rejections and the F-205
LFUSE/DF_PLE interleave. The F-200 battery's other arms (ATTN_LANECELL, HC_SPLIT=1, GR_V3,
L2 72, MMQ_KQ) were measured on the SAME buggy wrapper (bare engine + one var, scored against
a full-standing-battery interpolated control) but appear in neither the STANDS nor the VOID
list — they silently kept their verdicts. Specifics:
- **ATTN_LANECELL** — "shelved (phase-table check pending)" at +3.6/+5.4/+1.9. Note the sign:
  it read POSITIVE while missing the entire standing env battery; the true effect on the full
  stack is plausibly LARGER. It kills the 12 warp-shuffles/cell in the 4.40 ms attention line
  (F-194), i.e. it acts on the exact line the #3 queue item attacks — a zero-code partial
  precursor. The "phase-table self-check" it was shelved pending never happened (no journal
  mention after F-200). Effect: 4.40 ms attention line; possibly free. Confidence: High that
  the verdict is void-class; Medium that it is a real win.
- **GR_V3** (split-K hc read, the 4.2 ms line, MMLU-gated) and **L2 72** (carve headroom per
  the 96 MB premise correction; F-187 measured the 60 MiB ask clamped over a 112 MiB span,
  hitRatio 0.536) — both "neutral today" on a bare base = unmeasured. Confidence: High
  (void), Low-Medium (win).
- **MMQ_KQ** — tested on M1/pair/conc, which are NOT its metric ("MMQ needs a prefill-lane
  test, not these metrics" — F-200's own verdict), and the prefill test never happened. It
  remains the only queued-anywhere prefill arm. Confidence: High (mis-test is documented).
- **HC_SPLIT=1** — neutral on a bare base; the =0 rejection (−15%) stands regardless of
  confound. Confidence: High (void), Low (win).
Why dropped: F-205's triage focused on what to void for fairness of the *adopted* stack and
the two fresh rejections; the battery's rejects were left with stale verdicts by omission.

**A2. T1.3 — deeper slot chains + the floor stair (the pair-lane acceptance lever).**
Source: ROADMAP T1.3 (:72-79); F-187 NEW #3 with file:lines (generate.cpp:12796-12798,
4736, 10317, 10223-10234); F-184 "WHAT THIS OPENS"; F-159 per-position data. Slots verify
ONE proposal today; at 63-68% slot acceptance a second proposal banks ~0.6 rows/window
nearly free; the serial chain starves at floor 0.5 while depths 1-3 accept 96/93/90%.
F-187 estimated pair +4-8%, M1 +2-6% (stair), all offer-side/distribution-neutral by
construction. The F-199 roadmap called T1.1/T1.2/T1.3 "the three that carry the target";
the F-207 recalibration kept T1.1 and T1.2 and **silently dropped T1.3 entirely** — it is in
no queue line and no re-queue note. Pair is the second-worst goal metric (+16%). Confidence:
High (drop is factual; value estimate is F-187's own).

**A3. T1.4 — hc read parallelism (the 4.2 ms line has NO queue item).**
Source: ROADMAP §3 T1.4 (:80-84); F-194 §2 (41-block starvation, 157 MB/window at 37 GB/s
effective, Little's-law ceiling ~80 GB/s); F-202 ROADMAP v3 R37 (hc absorption, exact-static,
n=2-safe per OLMo ablation); F-200-moe/formulations briefs. The roadmap's own recoverable-pool
arithmetic (§1: doorbell 4.9 + attention 2.2 + **hc 1.8** + emit 1 ≈ 10 ms) includes hc; the
F-207 queue covers doorbell, attention, emit — and omits hc. Target ~1.5-2 ms/window on the
#2 GPU line. Confidence: High.

**A4. R28 — GDN one-pass fused identity + log-space decay.**
Source: F-201-formulations-agent.md findings 1-2 (identities EXECUTION-VALIDATED at
1.8e-15 / 9.1e-14; halves the serial dependency chain, register-resident state per FLA,
1.5-2× on the step kernel); F-202 queue line "R28 GDN fused one-pass" (listed FIRST in
ROADMAP v3); F-201 R27 note ("the real item is the step kernel's ILP rewrite… queued with
the attention row-fold"). GDN lines ≈ 2.85 ms VRAM-hit + 0.75 ms rec per 6-conc window
(F-187 phase table). Silently absent from the F-207 queue; the "queued with the row-fold"
promise was not kept. Confidence: High.

**A5. In-graph KV-gather overlap (ProMoE pattern) — the zero-risk version of the attention
fix.**
Source: F-201-kv-agent.md #4 ("gather is independent of expert routing — concurrent graph
branch, event dependency only before the QSA layer… hides nearly all of 4.40 ms… ADOPT (pure
scheduling)"); corroborated F-202 queue line R30's "in-graph overlap". The queue narrowed R30
to the row-fold kernel only. If the independence claim survives a code read, this hides up to
~4 ms behind the ~20+ ms of non-attention window work with no bit risk and no new attention
kernel — and would de-risk/change the ROI of the row-fold. Confidence: Medium (single-source
mechanism claim; scheduling-only).

**A6. T1.6 — fenced admission (the 54.8-58.3 ms admit waves).**
Source: ROADMAP T1.6; F-194/F-195 (~14-30 cudaDeviceSynchronize per 6-slot wave,
generate.cpp:10049-10140); F-201-scheduling-agent #3 (FastSwitch: dispatch is 90-95% of
transmission time; our 4× signature; 1.4-11.2× tail improvements) and #6 (overlap scheduler).
Not in steady-state window means, but it IS the pair/6-conc numbers under real traffic —
and the goal metrics are measured through probe admission churn. Dropped in the F-207
recalibration without a note (R31 shape-keyed graphs, queued #5, fixes re-captures but not
the sync storm). Confidence: High (drop factual), Medium (size of pair/conc effect).

**A7. Acceptance-theory "immediate" trio + R34/R42 — never entered any queue.**
Source: F-201-acceptance-theory-agent.md closing line: "Immediate: (a) Sequoia
depth/width reallocation with c=6-9%; (b) marginal stopping ≥7-9%; (c) truncation ~10 at
temp 0.7. Medium: (d) ERSD first-k arrivals…" — (d) became R34 in F-202's ROADMAP v3 and
(g) δ-KL rescue became R42; none appears in F-200 v2 queue, F-202 v3 survives only as
unexecuted lines, and the F-207 queue has none. (b) is directly load-bearing for the spec-8
re-queue (see C2). Confidence: High (drop factual), Low-Medium (individual value).

**A8. Block Verification (R28 of the F-200 v2 queue — numbering later reused).**
Source: F-200-specdec-agent.md #1 (ICLR'25, provably optimal, distribution-exact, +3-8%,
"drop-in atop exact-match+rescue"); F-200 entry queue line "R28 block verification (lossless
drop-in)". Silently gone by F-207. Acts on every metric via banked rows/window; composes with
A2 (slot chains) and A7. Confidence: High (drop factual), Medium (value — 2026-paper
greedy-only discount applies, F-186).

**A9. Exact two-phase lm-head (T1.8) with the concrete Cauchy-Schwarz construction.**
Source: F-200-quant-agent.md #8 ("NOBODY SHIPS IT… provably exact… 521 MB → 30-70 MB");
ROADMAP T1.8 (~1-2% of window). Dropped from queue. Honest sizing: the F-187 phase table
measures the head line at 0.84 ms, so the realistic prize is ~0.4-0.7 ms — small, but it is
the rare EXACT-no-MMLU-risk byte cut and the algorithm is fully specified. Confidence: High
(drop factual), Medium (value).

**A10. M5 mimicbench port + adherence measurement (a goal lane with no instrument).**
Source: F-173-era note "mimicbench not yet ported — the M5 tool must be located/ported from
the Windows campaign" (:5426-5427); F-178 "Open bar items unchanged: M5 mimic port" (:5661);
AGENTS.md law 3 ("Adherence (mimic) when the tool is ported"). Never done. Last mimic number
on any lane: 95.3-98.4 band, pre-campaign (F-20/F-30 era). Since then the stack adopted at
least one output-affecting change (ROUTE_RESIDENT 0.75 — routing swap of tail ranks, MMLU-
gated only). See (D). Confidence: High.

**A11. Prefill instrument + prefill-vs-stock baseline.**
Source: probe-lane.py contains no prefill measurement at all (grep: only "fresh-long-prompt"
decode); the last stock-vs-ours prefill comparison is pre-campaign (F-30 era: stock 6632 vs
ours 7463-7514, +13%, different recipe); F-198's stock scoreboard recorded no prefill row;
T2.1 layered prefill sits in roadmap Tier 2 with no queue slot; F-200-prefill-agent #8's
must-verify ("VERIFY OUR GDN PREFILL IS CHUNKWISE — sglang-jax #1416") was never executed
(grep "chunkwise": F-200 notes only). See (D). Confidence: High.

**A12. Smaller confirmed drops (one line each):**
- KTransformers one-layer-late CPU-expert deferral (+1.45× class; F-200-moe #3; queued in
  F-200 v2) — dropped; superseded IF T1.1 lands fully, hedge if it does not. Medium.
- Next-layer expert prediction 93-99% recall (F-200-moe #4/#5), HOBBIT T₂-skip, LLRU
  eviction (~5 lines) — all "ADOPT" verdicts, none queued. Low-Medium each.
- DMA megablobs 2.8-10 MB + THP/cudaHostRegister pool + CCD re-pin (R40; ~1.2-1.6 ms of the
  tail's DMA slice + staging path) — dropped; value shrinks after T1.1 but the hugepage
  re-register is cheap and benefits any residual PCIe traffic. Medium.
- Fence audit (13 `__threadfence_system` sites, elementwise.cu; R26 residual) — subsumed
  only PARTIALLY by doorbell inversion; never explicitly queued. Medium.
- AdaEDL early draft-stop (proven lossless; queued twice: F-187 v4, F-193 #7) — never
  tested. Low-Medium.
- Suffix-automaton draft override (TRT ships it; "ADOPT — cheap"; F-200-specdec #11),
  Nightjar length bandits, multi-draft-without-replacement, traversal verification — all
  F-200-adopt verdicts, none queued. Low-Medium each.
- Run-2 stall hunt: fsync (F-203) and DVFS (F-204/F-205) exonerated, then the hunt stopped —
  no next action recorded despite "worth several % of every battery mean" (ROADMAP §6) and
  remaining suspects (adapt refill, expert-profile save path, ASPM/fabric C-states per F-193
  #1). Medium.
- Rejection-sampled slot accept (F-187 #3: "distribution-SAFE substitute for the F-30b
  rescue… Tier-1.5, MMLU-gate mandatory") — never pursued; would let slot rows use rescue
  legally and lift the 63-68% slot-acceptance class. Medium.
- DT sweep peak re-check (F-205: "the peak location may shift on the full base — optional
  re-sweep") — not queued; two boots (0.80/0.90) would settle it. Low.

---

## (B) MIS-PRIORITIZATIONS IN THE CURRENT QUEUE

Queue under audit (F-207): T1.1 partial plan → doorbell inversion → attention row-fold →
batched draft-head → R31 shape-keyed graphs; re-queued DOORBELL_STORE, spec-8.

**B1. T1.1 #1 — CORRECT.** 4.90 ms = 15.6% of the 31.41 ms window; mechanism confirmed twice
(F-194 #4, T1.1 plan's flow read). Highest EV per line of code (~150).

**B2. Doorbell inversion #2 — estimate likely double-counts T1.1.** The 1.8-3.6 ms figure
(F-201-pcie brief; F-201 entry) was computed for ~1,800 poll nodes at 1-2 µs PCIe RTT in the
PRE-T1.1 graph. After T1.1, partial groups wait on one flag with the host publishing misses;
the poll-node population and per-poll count both shrink. Its residual EV should be re-derived
AFTER T1.1 lands, or the two should be co-designed (posted-write publish is arguably part of
T1.1's miss-publish path). Not wrong to keep at #2 — wrong to carry the 1.8-3.6 ms number
uncorrected.

**B3. The queue dropped the roadmap's own "three that carry the target" to two.**
F-199: "the Tier-1 trio T1.1/T1.2/T1.3 carries the target." Stacked arithmetic check at the
warm-class means: 6-conc needs +14-16% on top of ~214 to reach 244; queue EV ≈ T1.1
+10-14% + row-fold +5-7% + inversion-residual +1-3% + batched-head +0.3% — reachable only if
everything lands near the optimistic end AND the overlap discounts are small. Pair needs
+11-13% on top of ~173; the queue's pair EV is thinner (T1.1's tail is ~3 ms at pair windows
per F-192, row-fold helps, but the acceptance-class lever A2 — which F-184/F-187 priced at
+4-8% specifically for pair — is absent). T1.3 was the hedge that made the roadmap's
"+22-32% pair" honest; without it the plan has a single point of failure per budget line.

**B4. hc (1.8 ms of the claimed 10 ms recoverable pool) has no queue owner** (A3) while the
~100 µs batched draft-head residual holds slot #4. That ordering inverts the pool arithmetic
the roadmap itself established.

**B5. R31 (#5) vs fenced admission (A6):** for the goal metrics as measured (probe batteries
that admit slots every run), the 54.8-58.3 ms sync waves are a measured per-run cost; R31
kills re-capture stalls (~35 captures/boot, admission/drain boundaries — F-195), which is the
same stage's smaller half. If only one admission-side item is queued, the sync-storm fix has
the larger measured line.

**B6. spec-8 re-queued WITHOUT its prerequisite (see C2):** re-testing depth 8 on the fixed
wrapper but at floor 0.5 re-creates the exact starvation that made depth 4-6 unmeasurable;
the arm would burn a boot to re-derive a voided number instead of enabling the depth lane
via the stair first.

**B7. Post-F-206 calibration not fully propagated:** the fusion-tranche negative
("node savings ≠ wall time") devalues every spacing-class item still alive in the broader
roadmap (T2.3 DuetServe look-ahead, launch-completion edges, device graph launch, conditional
nodes T2.4). The F-207 queue correctly excludes them — but the roadmap document still carries
them un-annotated, and T2.4's "kills doorbell waits inside the graph" overlaps T1.1. Cheap
fix: mark them "post-F-206: devalued/overlapping" so nobody re-queues them from the doc.

---

## (C) VOID/SUSPECT RESULTS DESERVING RE-TEST ON THE FIXED WRAPPER

**C1. Already voided by F-205 and re-queued (listed for completeness):** DOORBELL_STORE,
spec-8. Note for DOORBELL_STORE: the F-203 "correctness-risk signature" (M3b −13 pts) was the
missing battery, so the CUDA ring-semantics desync THEORY is unproven either way — re-derive
the per-wait-site value semantics before arming, as F-203 itself ordered.

**C2. spec-8's rejection rationale contained an unmeasured inference.** F-203 claimed
"measured 90%@3 → the tail is dead by 6." F-159's 96/93/90% are depths 1-3; depths 4-6 were
never OFFERED (serial chain stops ≤3 under floor 0.5, generate.cpp:12796-12798) — their
acceptance is unknown, not measured to be dead. F-187's own protocol (STRATA_SPEC_DEPTH=1
diagnostic boot to read accepted/tested by depth, then decide) was never run. The re-queue
should be: stair diagnostic → per-depth histogram → THEN depth 6/8 with the stair. High
confidence this sequencing is required; Medium that depth >6 wins.

**C3. The F-200 battery rejects/shelvings (see A1) — ATTN_LANECELL, HC_SPLIT=1, GR_V3, L2 72,
MMQ_KQ.** All bare-based; none voided by F-205; all cheap (env + one prefill-lane boot for
MMQ_KQ). Highest-priority re-tests in this section.

**C4. DT 0.85 peak location.** Re-validated at +4.0% M1 at 0.85 only; the sweep curve that
picked 0.85 was bare-based (F-205: optional re-sweep). Two boots.

**C5. RR 0.75's MMLU gate evidence ran during the buggy window.** The two F-200 gates
(92.86/97.14) were taken on boots from the confounded battery window; the adoption stands on
stack-level evidence, but a single MMLU gate on the true standing stack closes the loop cheaply
(the watch rule from F-180 also still nominally applies: MMLU 3-gate mean < 92 escalates).

**C6. Standing (no action):** F-191 PDL no-op, F-185 overlap rejection, F-187 QFUSE rejection,
F-183 shelf variants, F-118 SPEC_RATIO −21% (note: F-178 listed SPEC_RATIO as "A/B-pending"
while F-118 had already measured and rejected it — stale queue note, not a dropped win),
F-203's fsync strace and F-204/F-205's DVFS dmon (launcher-direct boots), F-206 fusion
negative (fixed wrapper).

---

## (D) UNCOVERED LANES

| lane | tried (campaign era) | current delta vs stock | planned item? |
|---|---|---|---|
| **prefill** | F-200 T0.8 MMQ_KQ — on the WRONG metrics, bare base (void-class); nothing else since the merge. Historical: auto:32768 (+21-35%), KV_PREFETCH (+5.3/4.1%), DMA (record 7514 tok/s); pre-campaign vs stock ≈ +13% | **unknown** — no stock-vs-current prefill row exists post-F-198; probe-lane.py has no prefill line; last comparison is pre-campaign at +13% (stale recipe, stale binary) | T2.1 layered prefill (Tier 2, no queue slot); MMQ_KQ re-test on a prefill metric (nowhere queued) |
| **mimic adherence** | nothing — tool never ported to Linux (M5 open since F-173); last measurement pre-campaign (98.4 best, band 90.6-98.4) | **unmeasurable** on this box today; output-affecting adoptions (RR 0.75) never adherence-checked | NONE anywhere |
| **acceptance rate** | heavily worked internally: DT 0.85 (re-validated +4.0% M1), rescue/typical machinery, M3b canary; warm M3b 74-79% | **undefined vs stock** — stock servers do not print M3b/accept lines (F-198 note 3), so the "+30% acceptance" goal has no denominator | DT re-sweep (optional, F-205); no acceptance-class item in the queue (A2/A7/A8 would be) |
| **MMLU** | gate instrument used throughout (floor 93.3; RR 0.75 adopted at 95.00 mean) | vs stock: stock v0.1.44 never gated on this box (95.71 anchor is our own early-merge class) | gate-only lane by design — but the missing stock row makes "vs stock" unauditable |

**Headline:** MIMIC is the purest uncovered lane (no tool, no number, no plan). PREFILL is a
close second and more damning given it is a stated speed metric with a stale +13% figure and
an arm (MMQ_KQ) that was tested on the wrong lane and voided by implication. Acceptance's gap
is narrower (no stock denominator). If the campaign declares "every metric +30%," two of the
seven lanes currently cannot be evaluated at all.

---

## (E) TOP-10: WHAT THE CAMPAIGN SHOULD DO NEXT THAT IT IS NOT PLANNING

1. **Floor-stair diagnostic + stair adoption (T1.3a).** STRATA_SPEC_DEPTH=1 boot reads
   depths 4-6 (never measured); stair 0.5/0.5/0.30 is offer-side only (outputs unchanged by
   construction, F-187) — the M1 lever (+2-6%) and the precondition that makes the re-queued
   spec-8 test meaningful instead of a repeat of a voided arm.
2. **Deeper slot chains (T1.3b).** set_max_drafts 1→3 + matched-prefix accept loop
   (file:lines in F-187): pair +4-8% on the second-worst goal metric; the roadmap's own
   "three that carry the target" minus one.
3. **hc read parallelism (T1.4).** The 4.2 ms #2 GPU line — 1.8 ms of the roadmap's own
   10 ms recoverable pool — has no queue owner; env knobs (GR_V3 re-test) plus the
   exact-static absorption identity make it low-risk.
4. **Re-run the F-200 void-class arms on the fixed wrapper** (ATTN_LANECELL first — it read
   +3.6/+5.4/+1.9 while missing the whole battery, then GR_V3, L2 72, HC_SPLIT=1): days of
   boots, zero code, possibly masked wins on the two biggest GPU lines.
5. **In-graph KV-gather overlap.** Scheduling-only form of the attention fix (kv brief #4):
   hides up to ~4 ms behind non-attention window work at zero bit risk; validate the
   routing-independence claim by code read first.
6. **GDN one-pass fused identity + log-space decay (R28).** Execution-validated exact
   identities (1.5-2× step kernel) on ~3.6 ms of GDN lines; promised "queued with the
   row-fold" and dropped.
7. **Fenced admission (T1.6).** Event-ordered admission replaces 14-30 device syncs; measured
   54.8-58.3 ms waves vs ~13 ms floor — the pair/6-conc number under real traffic, which the
   goal metrics actually sample.
8. **Prefill lane: instrument, baseline, then attack.** Add a prefill line to probe-lane.py
   (or pin depth-lane.py), take current-vs-stock prefill in one window, verify GDN prefill is
   chunkwise (F-200-prefill #8), then MMQ_KQ on the metric it was built for.
9. **M5 mimicbench port + adherence re-measure on the standing stack.** A goal lane with no
   instrument and no number since before the campaign; also the only quality check
   RR 0.75's output change has never passed through.
10. **Block Verification (lossless, drop-in).** +3-8% class on every metric via banked rows
    (F-200-specdec #1); the exact-lossless item F-200 queued as R28 and F-207 lost; composes
    with items 1-2. (Runners-up: DT re-sweep at 0.80/0.90; run-2 stall next-suspect step;
    fence audit as a rider on the doorbell-inversion work; DMA megablobs + THP pool.)

---

## Contradictions and their resolution

- **F-178 "SPEC_RATIO A/B-pending" vs F-118 (−21%, rejected):** F-118 wins (measured);
  journal note is stale. No action.
- **spec-8 "tail dead by 6" (F-203) vs MTP brief "90% at depth 3 = large headroom":** the
  F-203 claim was an inference over unoffered depths AND is void anyway; resolved by the
  stair diagnostic (E1).
- **Node-spacing class (briefs: 8-10 ms from fusion) vs F-206 negative:** resolved —
  F-206 (fixed wrapper) stands; spacing items devalued; B7 records the doc fix.
- **fsync-class run-2 stall (NVMe brief) vs F-203 strace (zero syncs):** resolved against
  the brief; the stall itself remains open with no owner (A12).

## Open questions

- True per-depth acceptance at depths 4-8 on the full stack (E1 answers).
- Post-T1.1 residual poll-node count (re-prices queue #2).
- Whether QSA gather is genuinely routing-independent (gates E5).
- Whether the RTX 5090's persisting-L2 cap actually allows 72 MB and whether GR_V3's
  sum-order change passes MMLU (E4 answers both cheaply).
- Stock v0.1.44 acceptance/MMLU/prefill rows: one stock boot with instrumented probes closes
  all three denominators.

## Implications

The queue's #1-#3 are the right big rocks; the campaign's exposure is (1) a voided-battery
tail of cheap arms that were never re-queued, (2) a missing acceptance-class hedge on the
pair lane, (3) one recoverable-pool line (hc) with no owner, and (4) three goal lanes
(prefill, mimic, acceptance-vs-stock) that cannot currently be scored against the +30% bar.
Items E1/E2/E4 are boots, not code; they should interleave with T1.1 work immediately.

## Sources

Primary: SVG-CUSTOM.md F-118, F-156..F-159, F-178..F-207 (lines as cited); AGENTS.md;
campaign/ROADMAP-30-VS-STOCK.md; campaign/research/ (all 18 files); campaign/probe-lane.py;
campaign/results/f198-up-3.txt; git log 74bff7a..083f700. Categories: code+history (journal,
git, results, probe source), specs/docs (roadmap, AGENTS law, briefs' primary literature
citations inherited), data (measured budget lines, F-198 scoreboard). ~30 distinct sources
consulted; briefs' external citations were treated as leads, not re-verified.

## Methods note (unproductive searches)

probe-lane.py contains no prefill instrumentation (searched "prefill", "prompt", "tok/s" —
confirmed absent; use depth-lane.py or engine-log parsing instead). No journal evidence that
the F-200 "phase-table self-check" for LANECELL ever ran (searched "phase-table self-check",
"phase table self"). No F-208 file existed prior. "chunkwise", "Nightjar", "suffix-automaton",
"megablob", "hugepage", "KTransformers", "two-phase" each appear ONLY in research-note/queue
contexts — never in an execution entry — which is the grep basis for the never-executed
classifications.
