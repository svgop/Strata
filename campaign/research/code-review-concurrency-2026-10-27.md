---
type: research
status: read-only-code-review
date: 2026-10-27
lane: strata-custom-2 (RTX 5090, standing recipe: --batch 6, STRATA_COHORTS=2, STRATA_BATCH_MTP=1, STRATA_SPEC_COUPLED=1)
topic: batch/cohort lane anatomy, G=2 under batch, conc-lane cost structure, lever ranking
method: code read (verify.cpp / generate.cpp / server.py / session.cpp) + banked campaign evidence; no boots, no benchmarks; server untouched
---

# Code review — the batch/concurrency lane (2026-10-27)

Every claim below carries a `file:line` anchor or a banked F-entry. Measured lanes this window:
solo M1 ~223, pair ~214, conc ~325 t/s (F-356's same-hour audit reads the current binary at
213.3/204.9/300.2 on a loaded box — the band moves with box state, not code: SVG-CUSTOM.md:11482).

## 1. The batch lane: how 6 slots share one window graph

**Capture.** `Verifier::capture_batch` (src/core/verify.cpp:3551) captures ONE CUDA graph over all
rows: it sets `batch_rec_ = true`, copies the slot ids `brow_[t] = rows[t]` (:3560), and calls
`record_window(S, cs_, ...)` (:3562) — the same window recorder the serial path uses, with the
batch flag on. The graph is keyed `bkey(rows, S, hbase)` in `exec_bm_` (:3552) — a captured
execution graph per slot-layout. `stage_batch` (:3758) captures the window graph AND both commit
parities up front (:3784-3786, F-196), so a parity flip never captures mid-serving.

**Rows per slot.** Rows are (slot, proposal) pairs. With `STRATA_BATCH_MTP` and draft depth D=1,
each slot owns a contiguous group of 2 rows — its base token plus one proposal
(generate.cpp:12092-12093: `rows_b = 1 + min(n_draft, batch_draft_depth)`). At batch 6 / cohorts 2:
cohort A = slots [0,3) = 6 rows, cohort B = slots [3,6) = 6 rows (generate.cpp:9387, :12078).
`kVerifyMaxT = 8` (include/strata/kernels/verify_kernels.hpp:21) is the hard row cap per window;
`collect` admits a slot only if `w.S + (mtp_use ? 2 : 1) <= kVerifyMaxT` (generate.cpp:12084).

**Token staging.** `stage_batch` (verify.cpp:3758-3837) writes per-row tokens/steps/positions into
host-mapped staging `h_tok_ / h_step_ / h_pos_` (:3789-3797) — the graph's first nodes copy them
over PCIe (`copy_i32_from_mapped`, record_window :1239, :1247-1248). It also stages PLE n-gram rows
(:3798-3810) and the per-slot-group commit defaults `c[0]=n, c[1]=n-1` + positions (:3813-3824),
then zeroes the doorbell words `h_seq_ / h_flag_ / h_flagA_ / h_flagB_` (:3825-3829).

**KV state per slot.** Inside the graph every row dereferences its OWN session:
`slot_ss(t) = *slots_[brow_[t]]` (verify.cpp:1232). Concretely: GDN conv history and state are read
per slot-group from `sx.gdn_state` (:1544-1547, :1592-1594); each slot's QSA indexer tail is
snapshot before its first proposed row (:1707-1716); each row attends over its own slot's K/V
(:1944); the batch commit graph walks slot groups and commits each slot's accepted prefix into that
slot's `gdn_state` (capture_commit_batch :3624-3653). The service loop's per-(layer,group) pool call
feeds each group's rows only: `pool(user, h_x_ + tb*N, h_ids_ + tb*K, n, K, h_ymiss_ ..., l)`
(verify.cpp:3924-3926).

**Launch and service.** `batch_launch`/`batch_launch_rows` (:4040-4083) = stage + `batch_go`:
launch the exec graph, set `b_steps_ = (le_-lb_) * batch_groups_` (:4073), mark `b_running_`. The
host then serves doorbells: in `run_slot_rows` (:3890-3944) and identically in `batch_serve`
(:4096-4133) / `batch_poll` (:4180+), doorbell `k` = layer `lb_ + k/G`, group `k % G`; the host
spins on `*seq < want`, runs the CPU-expert pool for that (layer, group), publishes the plan and
raises `h_flag_` (:3941). The GPU side, per layer per group: router → device plan → doorbell
publish → resident expert kernels → `wait_flag_ge_planned(m_flagA_/m_flagB_, ring, skip_+grp)` →
combine (:2215-2332), where `ring = (l-lb_)*G + grp + 1` (:2164). `batch_go(with_commit=false)` on
the cohort path (:4059-4061) — the pre-launched commit collided with the tail's parity commit
(F-233 bug 2, SVG-CUSTOM.md:7975-7977).

**Batch window vs the serial T=7 window, per token.** Both are the same recorder; `dec_batch`
multi-row kernels make every row's arithmetic the single-row call's (e.g. router :2025-2034,
conv :1540-1550). The serial T=7 window (`capture(T)`, verify.cpp:2518) runs ONE session: 7 rows
are one stream's MTP chain — weight streaming amortized over 7 rows of one sequence, acceptance
2.74 tok/window at 14.15 ms (campaign/research/decode-anatomy-2026-10-27.md §1). The batch window
runs 6 independent streams in the same row count-class: the per-layer weight stream (expert math =
50% of GPU busy, anatomy §2) is paid ONCE per window either way, so per-token marginal cost is the
row arithmetic — the campaign's c_row ≈ 2.2-2.6 ms/row (T47-conc-450.md §1). Two real differences:
(1) batch tokens are independent across slots, so acceptance is per-slot (~1.77 tok/slot at conc,
T47 §1) with no chain depth; (2) with cohorts=2 the round runs TWO half-width windows, so the
per-round expert-weight stream happens twice (once per cohort graph) — that duplication is the
price of overlap, and it wins because each window's GPU-idle pool phases are filled by the other
window's weight stream: two ~6-row windows per 26-29 ms round moved the in-window rate from ~349
to ~460 rows/s (F-233, SVG-CUSTOM.md:7987-7989).

## 2. Cohorts: STRATA_COHORTS=2, and what =3 needs

**The two-cohort round** (generate.cpp:12054-12242, F-233 adopted 10-17): slots [0, batch/2) on
verifier A, [batch/2, batch) on `ver_b` (bound at generate.cpp:9366-9388 — second verifier, own
buffers/graphs/doorbells, shared weights + expert cache + slot sessions). Both half-width windows
are STAGED while nothing is in flight (:12156-12171 — a lazy capture under flight is the #267 hang
class), both LAUNCHED (:12172-12183), then ONE reactor loop polls both (:12217-12233), each
`batch_poll` serving its own verifier's doorbells. **"B pool work runs under A's GPU"** is literal:
while A's graph executes, B's `batch_poll` returns a rung doorbell and the host thread runs B's
CPU-expert pool for that layer — host pool work overlaps the OTHER cohort's GPU execution. This is
"the one escape from the F-223 theorem" (comment at generate.cpp:6884-6888): within one window, the
synchronous service loop can never overlap host pool work with that same window's GPU (see §3);
across two windows on two streams/graphs, it can. Engagement gate: `cohorts >= 2 && n_active >
o.batch/2` (:12061) — solo and pair take the untouched single-window path byte for byte. Measured:
conc +24.8% (306.8 vs 245.9, non-overlapping arms; F-233, SVG-CUSTOM.md:7983-7989).

**What COHORTS=3 needs.** Code gate: `STRATA_COHORTS >= 3 && o.batch >= 9` (generate.cpp:9393) —
verifier C takes slots [2b/3, b), ternary rounds are 3+3+3 with every window the 6-row class
(:12080, :12114-12118). The **b9 lane problem**: batch 9 + elastic sessions hung in the admission
ramp — slots 5-8 materialize by releasing "1.3 + map from the EXPERT CACHE'S TAIL — 0 chunks left
in the spare pool"; at batch 6 all slots fit the spare pool, at 9 they steal from the cache
(F-284, SVG-CUSTOM.md:9604-9611; #267 "verify timed out at layer 0"). The fix was CONFIG-ONLY:
`STRATA_ELASTIC_S_MIN=9` maps all nine arenas at boot so the admission-ramp expert-tail release
never executes (F-286, SVG-CUSTOM.md:9699-9704; confirmed running clean in F-354,
SVG-CUSTOM.md:11434-11437). So "needs batch>=9 AND the b9 lane fixed" = the code gate at :9393 plus
`STRATA_ELASTIC_S_MIN=9` (or the T45 root-fix of the admission-ramp release path).

**But COHORTS=3 is already priced and economically negative at current VRAM** (F-354,
SVG-CUSTOM.md:11430-11454): b9/c3 reads M1 195.7 / pair 184.6 / conc 254.2 (S_MIN=9) and
199.2/185.3/262.2 (S_MIN=6) vs standing b6/c2 ~215/~205/~301-319. The 9 slot arenas + third
verifier's buffer/graph sets (6.75 GiB arenas, ~200-400 MiB verifier set) shrink the expert cache
from 10,446 to 7,974-8,749 slots (−16..−24%); the miss path (F-344 CPU-rows cost) grows more than
the third window hides. REJECTED for standing. The composition path (ternary + more cache slots) is
BLOCKED at the pack format: per-expert mixed quant (hot IQ3_S / tail low-bit) is NOT representable
in native_experts.txt v3 (per-layer types only, F-355, SVG-CUSTOM.md:11461-11467); the tractable
variants are per-layer mixed quant (a global quality trade) or a low-bit shadow store for misses
(HOBBIT-class engine work). Context for the hope: ter9 beat a 9-wide single-verifier control by
+9.7% pooled (F-286, SVG-CUSTOM.md:9714-9717) — the third window DOES work; it is the VRAM bill
that kills it at 9-wide on a 32 GB card.

## 3. G=2 under batch (STRATA_BATCH_G2): mechanism — and the pricing that closed it

**The gate.** verify.cpp:1179-1180:
`G = (T >= 2 && ((!batch_rec_ && split_) || (batch_rec_ && batch_g2))) ? 2 : 1`. Standing lane:
`batch_rec_` true every decode window, `STRATA_BATCH_G2` unset → **G=1** (F-362's correction,
decode-anatomy §6). The row split is `tb_[2] = {0, (T+1)/2}, te_[2] = {G==2 ? (T+1)/2 : T, T}`
(verify.cpp:1199). The captured G travels in `batch_groups_` (:1207) which every service loop
reads (:3887, :4073, :4096, :4180).

**What G=2 changes per layer** (each item doubles):
- TWO doorbells per layer (`ring = (l-lb_)*G + grp + 1`, :2164) → the host serves two plan/pool
  rounds per layer instead of one (:3890-3944).
- TWO plan handoffs: `set_plan_slot(grp)` re-points the sink at `h_plan_ + grp * plan_i32_`
  (:3279-3296); the GPU reads `plan_ + grp*(plan_i32_+16)` (:2067, :2077, :2166) and waits
  `wait_flag_ge_planned(m_flagA_/m_flagB_, ring, skip_+grp)` (:2215-2245).
- Staging splits: `per = G==2 ? kStagingBlobs/2 : kStagingBlobs` (:2247, :3294).
- TWO combines and two shared-expert invocations — the shared-expert fork (`sh_fork`,
  :2011-2017, late fork :2097-2100, join :2306-2308) is per (layer, group), so G=2 doubles the
  fork/join event pairs and runs the shared expert twice at half width.
- PDL is disabled under G=2 (`pdl_window(G == 1 && ...)`, :1216); the df-branch mixer too (:1221).

**The intended mechanism** (comment at :1175-1178): rows are independent slots, so the group cut
needs no dependency argument; while the host serves group A's CPU-expert pool work, the GPU
executes group B's kernels of the same layer — halving the window's host tail.

**Was it priced? YES — F-223, executed and CLOSED negative** (SVG-CUSTOM.md:7728-7766; also
T13B-concurrency-scheduling.md:28,129; T15C-overlooked-wins.md:361 "swept NEGATIVE+corrupt"). Five
discriminator boots on one binary:
1. G2 + standing (pcie 0.35): NON-FINITE logits under sustained concurrency (#879) — every
   multi-slot request dies after 1-4 tokens.
2. Clean only at `--pcie-frac 0` → the corruption boundary is **batch × G2 × PCIe-share>0**
   (the DMA device-alias/flagB per-group path; root cause never run to ground).
3. Even clean, the numbers: **G2 26.06 ms/window vs ctl 22.72; pair 157.2 vs 201.7 (−22%);
   conc 220.5 vs 251.9 (−12%)**.
4. The mechanism verdict: the pool stage did NOT shrink — the synchronous service loop serializes
   wait-doorbell → pool → raise-flag between doorbells regardless of grouping; the host's pool
   work never overlaps GPU execution from the host's side, while the GPU wait GREW +2.2 ms
   (2× launches at half width). Row-splitting alone cannot hide the CPU-expert stage.
The premise G=2 wanted (overlap pool work with GPU work) is delivered by cohorts at WINDOW
granularity instead (F-233 +24.8%), with no row-splitting and no per-group handoffs. So: on THIS
lane, BATCH_G2 is a characterized negative — re-open only with (a) a decoupled/multi-threaded
service loop (the `batch_serve`/`set_external_serve` machinery exists, verify.cpp:4085, :4144),
(b) a root-caused fix of the flagB/DMA per-group corruption, and (c) an honest expectation of ~0:
the cohorts lane already occupies the overlap headroom G=2 would sell.

## 4. The concurrency lane's cost structure at ~325 t/s

**Admission and the 7th+ request.** serve/server.py: every request takes the control lines through
`_take_control` (:1751-1824) — strict FIFO tickets (F-14, :1759-1760) or opt-in short-first
(:1774-1787). `generate_batched` (:1870+) runs the solo GEN path only when alone (:1907-1926);
otherwise it picks a slot (:2023-2031). A 7th+ concurrent request finds `pick_slot` None, gives the
control lines BACK (`self.ctl.release()`, :2035-2037), and waits on `slot_cv` in ≤1 s steps
(:2038-2052) — counted in `self.admitting` (#1603, :2032-2034) — until a slot frees at BDONE
(:1746), then re-takes the lines and re-queues. Two more admission gates sit in front: the prefill
cadence (`_cadence_open`, :1856-1868 — at most one prompt read per PREFILL_CADENCE_N decode
rounds) and BYIELD give-way (#656, :1933-1943). On the engine side, an idle slot's arena is
demoted; `STRATA_ELASTIC_PARK=1` parks the demoted slot's conversation into the ConversationCache
(generate.cpp:5417, :5458, :11692-11739) so a returning chat restores in ~60 ms instead of a
1,361 ms re-read. src/core/session.cpp is the per-slot state owner (`session_init` :82,
`session_zero` :154, `gdn_point_at` :191) — what a slot admission restores into; it is not a
scheduler.

**Where the 325 t/s goes** (T47-conc-450.md §1, all banked measurements): the standing 6-conc lane
commits ≈ 10.6 tok/round (6 × 1.77); 323-325 t/s ⇔ a **~32.8 ms round wall**, of which the
overlapped window span is 22-25 ms → **~25-30% of the round is NOT the verify window**: the two
per-cohort finish tails (acceptance chain-compare generate.cpp:12264-12294, parity commit :12305,
BT emit, per-slot `draft_launch`/`draft_finish`), admission prompt-read stops (decode-share 0.5
default), the in-round adapt join, and host-feed stalls. Verdict against the three candidates:
- **GPU time: no.** The two cohort windows overlap by construction (§2); the 87%-GPU-busy figure
  (anatomy §2) is the SOLO serial lane. In-window headroom is small (+1-3% class: PDL + launch
  diet, T47-5).
- **Expert pool contention: no.** Cache hit 97-98%; misses are the 2-3% residual (T47 §S7); CPU
  experts at solo are 0.64 entries/layer-window ≈ 2.24 ms/window (anatomy §1), already the object
  of the pool diet; the ~2 s GPU-busy spin-kernel tail (anatomy §2) is in-window and second-order.
- **Scheduling/host tails: yes — the binding term.** The non-window share (tails ×2, admission,
  adapt join, feed preemption) is the only place a 25-30% round-wall chunk lives; the window tax
  under host contention was measured at 10-15% (F-286 WSTAT forensics, SVG-CUSTOM.md:9718-9729)
  and the pre-F-332-vs-current same-hour audit proved box state, not code, moves the band
  (F-356, SVG-CUSTOM.md:11482-11502).

## 5. F-352 "batch depth D=2" — what was falsified and where it lives

F-352 (SVG-CUSTOM.md:11367-11398): the serving lane pinned `set_max_drafts(1)`; multi-proposal was
BUILT (187 lines, D=1 byte-identical) and **D=2 FALSIFIED at current acceptance**: rows/window
+19% but window period +20-22% (17.0 vs 15.25 ms), pair −7% (192.2 vs ~204-206), conc −6% (284.2
vs ~301-305). The marginal draft row's expected yield (~0.77 tokens at 77% per-draft acceptance)
loses to its cost (one more verify row + 2 extra draft step graphs/slot + the CPU-row term).
Re-open condition: per-draft acceptance → ~0.9+ — exactly what the DAgger drafter retrain targets;
but F-353 (SVG-CUSTOM.md:11400-11428) rejected round-2 (offline plateau 0.7932, engine stair WORSE
at every depth: 0.520/1.47 vs live 0.587/2.14) — the offline→engine transfer gap is binding.

In code today: `STRATA_BATCH_DRAFT_DEPTH` (default 1, clamp 1..min(spec-1,3)) at generate.cpp:5375-
5380; `d->set_max_drafts(batch_draft_depth)` at :5638; collect builds `1 + min(n_draft, D)` rows at
:12093 (cohort) and :12428 (single-window); a too-deep slot TRUNCATES rather than widening the
window past `kVerifyMaxT` (:12094-12095); the finish tail chain-compares proposals
(:12276-12292). Concretely, D=2 means 3 rows/slot → 9 rows per 3-slot cohort window > the cap 8
(verify_kernels.hpp:21) — so at b6/c2 every third slot truncates every round unless kVerifyMaxT
goes to 9 (T47-3's pair lever, with the mmvq-IL head tables covering T=2..8 only as a named
sub-cost, T47 §T47-3). The patch is in-tree, env-gated, inert — waiting for the drafter, not for
engineering.

## 6. Lever ranking (conc/solo expected gain, numbers from banked measurements)

Standing base: solo ~223 / pair ~214 / conc ~325 (operator's lanes; F-356's same-hour 213/205/300
is the loaded-box truth of the same binary).

| # | lever | solo | conc | evidence |
|---|---|---|---|---|
| 1 | **C2 tail/admission overlap at cohort scale** — run cohort A's finish the moment A's poll is done, under B's in-flight window; reader-thread for prompt reads; adapt-join deferral. READER-THREAD/PORT DESIGN ONLY — the F-285a C2a-lite lambda relocation hung the lane even arm-off (SVG-CUSTOM.md:9646-9662) | ~0 | **+9-18% → 355-385** | T47-1 (T47-conc-450.md §3); tails ≈ 1/3 of round wall; per-cohort `batch_stage_rows/batch_go/batch_poll` all exist (verify.cpp:4046/:4054/:4137) |
| 2 | **Host placement: CCD split + ambient-burn control** (renice/SCHED_BATCH the monitor/chrome burn; per-thread-class masks — NEVER whole-process taskset, falsified −20%) | protects ~223 | **+5-15% on a contended box; banks any record night** | T47-2; F-286 window tax 10-15%; eunomia 8.8-20.5% contended, +7.6% recovered; F-356 box-state proof |
| 3 | **COHORTS=3 (b9/c3)** — only behind a VRAM unlock: per-layer mixed quant or a low-bit miss shadow store (per-expert mixed quant BLOCKED by the v3 pack format) | **−9% measured** | **−15..−20% today (254-262 vs 301-319); the 355-390 band is unproven and format-blocked** | F-354 (SVG-CUSTOM.md:11430-11454); F-355 format block :11461-11467; ter9-vs-ctl9 +9.7% shows the window works, the cache bill kills it |
| 4 | **BATCH_G2** | ~0 (S=1 stays G=1) | **negative: −12% conc clean; corrupts at pcie-frac>0** | F-223 (SVG-CUSTOM.md:7728-7766); mechanism: sync service loop serializes regardless of grouping; cohorts already own the overlap |
| 5 | **Depth D=2 (F-352 patch)** | noise (209.7) | **−6% measured (284.2); re-opens only at ~0.9+ acceptance behind the DAgger drafter** | F-352 :11367-11398; F-353 round-2 REJECTED :11400-11428 — the drafter transfer gap, not engineering, is the long pole |
| 6 | PDL + launch-map diet inside the windows | +1-3% | +1-3% | T47-5; PDL same-stream only, sm_120 verified |
| 7 | 12-wide ternary | −30%+ (solo crushed to 153.5) | FLAT (281.1 vs 279.3) | F-286b correction, T47-conc-450.md:490-495 |

**Top three, ranked:** (1) C2 tail overlap in the reader-thread/port form — the only engine lever
with a +9-18% measured-class band left on the conc lane; (2) host placement (CCD split + ambient
burn) — zero code, converts the loaded-box 300-325 band into the 320-338 quiet-box band and makes
every other lever measurable; (3) the DAgger drafter retrain — the only lever that raises
tokens/window (1.77) on EVERY lane AND re-opens F-352's depth, currently blocked on the
offline→engine transfer gap, not on code. BATCH_G2 and COHORTS=3-at-current-VRAM are both priced
negatives — do not re-open them without their named preconditions (decoupled service loop +
corruption root-cause; pack-format or shadow-store VRAM unlock respectively).

## Boundary of this review

Read-only; the running server (port 8081) was untouched and no benchmark ran. All performance
numbers are inherited from banked campaign records (SVG-CUSTOM.md F-entries, campaign/research/
T47-conc-450.md, decode-anatomy-2026-10-27.md) — none re-measured here. The F-223 corruption root
cause (batch × G2 × PCIe-share flagB/DMA path) remains un-run-to-ground; F-353's offline→engine
drafter transfer gap is the open blocker on the depth lever.
