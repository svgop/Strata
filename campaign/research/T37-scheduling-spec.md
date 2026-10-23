# T37 — SCHEDULING, SPECULATION, AND CONCURRENCY: the conditional-graph keep rule made implementer-ready, admission overlap into the cohort reactor, the ≥9-wide + COHORTS=3 re-test (its F-249 verdict found VOID), and the lane-scoped depth governor

Date: 2026-10-22. Researcher B, wave T37. Scope: READ-ONLY — every anchor below re-verified at HEAD
`bf746391` ("features doc: the lifecycle era at full depth…") by opening the file this session; no boots,
no builds, no probes, no edits outside this file. Required reading consumed in full: T35-serving-latency,
T33-{priorart-materialization, priorart-serving, dynamic-session-allocation v1-v3, ourtree-verification,
serving-delta-notes}, T31-keeprefit, T26-2-spec-lane, T26-4-serving-conc, T27-2-spec-model-frontier; the
F-ledger (F-232/233/235/249/261-276); SVG-CUSTOM-FEATURES.md; AGENTS.md (laws 4b, 9); the standing
launcher + config. External sweep: 16 primary web sources across papers, CUDA/vendor docs, and
practitioner blogs (fetched or search-verified this session; §Sources).

**THE TWO HEADLINES:**
1. **F-249's "COHORTS=3 measured FLAT" is a VOID VERDICT, not a falsification.** The t21b arm boot
   (`campaign/results/t21b-c3-arm-r1.out`) prints `STRATA_COHORTS=2 on: slots [0,3) on verifier A,
   [3,6) on B` — the engine clamps `STRATA_COHORTS>=2` to the two-verifier mode (generate.cpp:9344,
   `cohorts = 2`; no `ver_c`, no `cohorts == 3` exists in ANY revision — git `-S` sweeps came back
   empty). Both sides of the battery ran the same mode; the "+0.3% inside noise" is an arm measuring
   itself. The third cohort has never been built or measured. The dead-end ledger entry (F-265) and
   every downstream citation ("COHORTS=3 flat at 6 clients", including the T37 task brief) inherit a
   measurement that never happened. The 6-client kill still stands on F-233 ARITHMETIC (below), but
   at 9-wide the question is not merely re-opened — it is FRESH.
2. **The batch/cohort lane offers exactly ONE draft per slot** (generate.cpp:11990-12008,
   `w.nrows[A] = mtp_use && bs[b].draft_ready ? 2 : 1` — seed + one draft). "Speculation under load"
   on this engine is therefore NOT "reduce depth when busy" — the loaded lane's horizon is already
   minimal by construction — and the task brief's "our depth-7-at-all-conc posture is unexamined" is
   wrong on both halves: the standing depth is **6** (F-263 FINAL reverted F-261's spec-7 adoption;
   the +8.5% was the ctl-dip artifact), and the conc lane never runs that chain at all.

---

## §1 OUR-STATE VERIFICATION (anchors re-verified at HEAD bf746391; corrections banked)

### 1.1 The standing machine (launcher + config, read this session)
`launch-custom-iq3s-8081.sh`: STRATA_COHORTS=2 · ELASTIC_SESSIONS=1 · ELASTIC_PARK/WATERMARK/
TIMED_SCAN=1 · VMM_CHUNK_MIB=32 · SERVE_TELEMETRY=1 · BATCH_MTP=1 · BATCH_RESCUE=0 ·
SPEC_COUPLED=1 · SPEC_DRAFT_TEMP=0.85 · TYPICAL_ACCEPT=0.2 · TYPICAL_DEVICE=1 ·
VERIFY_ASYNC_LAUNCH=1 · VERIFY_DEVICE_PLAN=1 · DMA_BATCH=1 · L2_SETASIDE_MIB=60 ·
PREDICT_CREDIT/BUDGET · ADAPT_{LAG,PINCAP,MIN_AGE,DORMANT}. **ADMIT_SHORT_FIRST absent (OFF,
dormant since merge)**. **No dm export — the F-270 dm fold is banked but standing adoption is still
OWED** (five-boot pooled extension; SVG-CUSTOM-TODO line 1773). `custom-iq3_s-8081.json`: `--spec 6
--spec-min-p 0.5`, `parallel 6`, `--prefill auto:32768`, `--pcie-mode dma`, `--kv int8
--kv-resident 32768`, `--vram-reserve-mib 2000`, `--conversation-cache-mib 49152`,
`--adapt-decay 0.85`; CUDA 13.4 lib_dirs. kVerifyMaxT = 8 (include/strata/kernels/verify_kernels.hpp:21).

Scoreboard cross-check: M1 212.2-228.3 / pair ~200-205 / conc-6 314.8-338.8 across F-266/F-270/F-273/
F-276 same-window controls and arms (the band is window-dependent — ±8-14% boot swing is the F-263
law); M3b 76.0-76.1, 9-conc acceptance 77.98% (F-273, highest measured); ITL p99 "20 ms @6 / 35 ms @9"
are the RETIRED mean-of-gaps artifacts (F-276 A6 ended that era; true per-token gaps now in /metrics);
TTFT p99 2.0-2.4 s prompt-size-dominated (the probe's 5,865-token prompt at ~3,300 tok/s ≈ 1.8 s);
E2E p99 19.6 s @9 (b9ela). All task-brief numbers verified or corrected above.

### 1.2 Engine anchors at HEAD (all opened this session; prior-dossier line ranges in parens)

**verify.cpp — the four capture families + doorbell duality:**
- `refresh_ar()` :2494-2508 + family selection `cudaGraphExec_t& exec_t = ar_off_ ? exec_nr_[T] :
  exec_[T]` **:2513** (also :2898 #871 variant rule, :2984/:2988 launch sites). Log strings verbatim:
  "have experts out of VRAM (a prompt loan, a shrink or a swap): the doorbell graph runs the windows"
  / "are 100% VRAM resident again: the zero-doorbell graph". (T33 said 2504-2511.)
- `Verifier::capture(int T)` **:2512** — the serial T-family (`exec_[T]`/`exec_nr_[T]` per offered T,
  ThreadLocal capture, `instantiate_evicting` + `cudaGraphUpload`). (T33 said ~2522.)
- `capture_commit()` **:2646** (T33: 2657). `capture_batch(rows,S,hbase)` **:3551** — keyed
  `exec_bm_[bkey(rows, S, hbase)]`: the BATCH family is per slot-COMPOSITION, not per width.
  (T33: 3554.) `capture_commit_batch(..., parity)` **:3583** — `commit_bm_[ckey(...,parity)]`, the
  F-196/F-233 parity double-buffer. (T33: 3597.)
- `device_plan_ = !all_resident_ && !always_publish_ && env` **:1113**; `Verifier::run(T,...)` **:2921**
  — captures and launches `exec_[T]` for the OFFERED T; `STRATA_VERIFY_ASYNC_LAUNCH` threaded launch
  :2972-2988.
- Batch reactor verbs: `batch_stage_rows` **:4046**, `batch_go(err, with_commit)` **:4054** (launches
  `exec_bm_[bkey(last_rows_, last_t_, row_base_)]`; cohort path passes `with_commit=false` and commits
  from the tail via `commit_slot_prefixes`), `batch_serve` **:4085** (doorbell service; the ar_on PLE
  host-flag raise; empty-plan sink publish), `batch_poll` **:4137** (cudaStreamQuery ≤1/ms on the
  ext_serve path). (T26-4 said 3896-3927/3935-4017.)

**generate.cpp — the cohort reactor and the two spec postures:**
- COHORTS init **:9340-9361**: the gate is `std::atoi(v8c) >= 2` → `cohorts = 2` (MAX). No third
  verifier anywhere (`ver_b` at :9102/:1281 only).
- Cohort round **:11965-12214** (T26-4 said 11503-11602; T35 gate 11809): engagement
  `cohorts == 2 && n_active > o.batch / 2` **:11972**; the F-233 standing→cohort transition fence
  (one `cudaDeviceSynchronize`) :11973-11979; `collect` **:11990-12008** — per slot
  `w.nrows[A] = mtp_use && draft_ready ? 2 : 1` (**ONE draft/slot**), rotation within the cohort span
  (`next_cur` :12007), loop condition `w.S + (mtp_use?2:1) <= kVerifyMaxT` (**the 8-row cap**);
  stage BOTH :12050-12057 (comment: "a lazy capture happens while NO window is in flight — the
  standing flow's capture condition"); `batch_go` BOTH :12058-12065; ONE poll loop :12067-12083; the
  per-cohort `finish` tail :12104-12189 — single-draft exact match (`outb[i] == w.tok[i+1]` → keep 2)
  + optional batch rescue; next-round per-slot drafts `draft_launch(2, …)` **:12168-12171** /
  `draft_first(1, …)` :12173-12177; the ADAPTIVE-TIER THREAD is started after the windows
  (:12089-12093) and **JOINED INSIDE THE ROUND** :12202-12210 (`bt_adapt_wait`).
- Serial lane (solo / ≤3 active): the flat gate **:15695-15700** (`while (T < S_mtp && dprob[T-1] >=
  req_spec_min_p) ++T;`) + **the F-269 keep-rule twin already in-tree**: `T = sk.apply(dprob.data(),
  T)` **:15701** (cut-only, serial-lane); suffix/lookup arbitration :15706-15730; `ver.run(T,…)`;
  acceptance walk + up-to-2 rescues :15921-15960; `STRATA_SPEC_TRACE` emission :15955-15962.
- Decode share: default 0.5 **:14222-14226**; budget application :14604/:14651/:14702-14713.
- pl_ fast lane: `pl_want = pipe && pl_pw >= 2 && pl_snap_ok && pl_active < 4` **:15046-15048** —
  INERT on this box (pipe not constructed with `--batch`; F-272 correction stands).
- Elastic lifecycle: `es_promote` :11558, `es_demote` :11667, `es_idle_scan` :11721-11733, watermark
  comment :6109 ("the spare always holds one promote's worth"), idle-scan call sites :12552/:12767.
- **WSTAT instrument: `window_ms.push_back` exists at exactly ONE site, :12428 — in the STANDING
  single-window path.** The cohort round records `bt_run/bt_commit/bt_emit` and prints "cohorts:
  windows done in %.1f ms" (:12085) but **never feeds the WSTAT histogram**. The F-276 "true
  per-window p50/p99" instrument is cohort-blind today (server-side parse_window_sample,
  server.py:971-981, consumes WSTAT lines). Per-TOKEN gap histograms (latency_sink, server.py:994)
  DO cover cohorts (BT lines are emitted in the cohort tail :12143).

**mtp.cpp / verify_kernels.cu — the draft loop (C1's target):**
- `capture_round` **:1316**, `capture_step` **:1366** (one graph per step so the host can stop when a
  draft is unlikely, :1329); the gated launch loop + `wait_step` spin **:1735-1775**;
  `h_row_[0]=a; h_row_[1]=0;` **:1646-1647** (mapped words; today only two).
- `mtp_select_kernel` **verify_kernels.cu:713-724** (`__threadfence_system()` :724 before the visible
  token write), `mtp_select` :795-799. (T26-2 said 671-686.)
- **ZERO conditional-node symbols in the tree** (`cudaGraphConditional|cudaGraphSetConditional|
  CondType|12040`: grep empty). The port is greenfield but fully templated by basalt (T26-2 §a.3).

**server.py — admission:**
- `ADMIT_SHORT_FIRST` / `ADMIT_MAX_WAIT_S` **:310-317** (OFF); `_take_control` **:1716-1789**
  (F-14 FIFO tickets; the only fairness inversion is BYIELD re-entry `after_epoch`);
  `_admit_next` **:1791-1800**; `_shorter_waiting` :1816; `YIELDS_MAX=2` :1714.
  (T35's 1647-1729 shifted +~70 lines since.)

### 1.3 BANKED CORRECTIONS (prior-dossier claims that fail at HEAD — record victories)
- **C-brief-1 (the task brief itself): "coupled MTP drafter at spec depth 7 (+8.5% M1)".** FALSE at
  HEAD: standing = `--spec 6`. F-261 adopted 7 on a 3v3 (+8.5%); **F-263 FINAL's discriminator read all
  three T28/T29 adoptions statistically indistinguishable and REVERTED spec to 6**; F-264 lists
  "spec 7 as gain" among the six priced falsifications; F-265 moved `--spec 7/8` to the dead-end
  removals list; the spec7 config + launcher were DELETED (F-267). The "depth 8 DEAD — serializes the
  cohorts" half is CORRECT (F-262: conc −6.1%, all three rounds negative; "the 8-token window is
  kVerifyMaxT itself").
- **C-brief-2: "our depth-7-at-all-conc posture is unexamined".** Wrong twice (depth is 6; the conc
  lane runs 1 draft/slot, §headline 2). The real unexamined surface is inverted: can the BATCH lane
  afford a SECOND draft row (never a knob — rows/slot is hardcoded :11998), and is the serial lane's
  flat-0.5 gate the right shape when row-cutting becomes launch-free (C1)?
- **C-new-1 (this session): F-249's COHORTS=3 verdict is VOID** — §headlines and §3.3. The 6-client
  dead survives on arithmetic only; the 9-wide question is fresh.
- **C-new-2 (this session): the WSTAT window histogram is cohort-blind** (only push site :12428,
  standing path). Every prior window-tail claim at ≥4 active slots rests on `bt_*` means in the TREND
  line, not the p50/p99 instrument.
- **C-anchor-1: line drift.** Every file:line above carries the HEAD anchor; the T26-2/T26-4/T33
  ranges are shifted by +2..+250 lines (the F-276 lifecycle commit grew generate.cpp). Future dossiers:
  cite re-verified HEAD anchors only.
- **C-hist-1: T35's citation "Batch Speculative Decoding Done Right, arXiv:2510.22876"** — that ID is
  "Correctness Forensics for Batch Speculative Decoding: Diagnosing the Ragged Tensor Problem"
  (EQSPEC/EXSPEC, Findings of EMNLP 2026; fetched this session). The negative's substance unchanged.
- **C-hist-2: AdaEAGLE is arXiv 2412.18910**, not 2410.18721 (that ID is a physics education paper —
  fetched, confirmed unrelated). Recorded so nobody re-purchases the wrong ID.
- Carried corrections (already banked, restated for the wave): F-271's "server auto-raises the
  reserve" is not in the tree (T35 contradiction #1; F-275 record correction); T35-elastic's
  "59/59 synchronous promotes, the background path NEVER fires" is SUPERSEDED by F-276's watermark
  (promotes now background 2.2-8 ms; demote unmap 2.0-2.8 ms; demotes in true idle).

---

## §2 EXTERNAL SWEEP (confidence-graded; primary source on every claim)

[MEASURED-OURS] = our banked record or code. [CORROBORATED] = primary external source, fetched or
search-verified this session. [HYPOTHESIS] = my transfer estimate with the arithmetic shown.

**S1. Nightjar — "Dynamic Adaptive Speculative Decoding for LLM Serving" (arXiv 2512.22420, v5
2026-06-15; Li, Zhang, Zhang, Wang, Fu, Lai). [CORROBORATED]** Full HTML fetched. Formalization:
**γ\*(B) = argmin_{γ∈{0..Γmax}} μ_{B,γ}** (Eq. 2), μ = mean latency/token under arm γ; loss
L_t(γ_t) = ℓ_t(γ_t) + 𝕀(γ_{t−1}=0 ∧ γ_t>0)·C_switch/γ_t (Eq. 1) — the disable/re-enable switch cost
amortized over post-switch yield; MAB planner with bin-level locking (Õ(√T) regret); draft offload
under memory pressure. **γ\*(B) is non-increasing in B, including γ\*=0**: small batch = memory-bound
(extra draft rows raise arithmetic intensity over the same weight read); large batch = compute-capped
(verification overhead loses). Measured: γ=3 +15.5% at 15 QPS, γ=2 best at 25 QPS, vanilla wins at
high QPS (their SD degraded up to 30.25% there); headline +14.76% throughput / −20.18% latency under
dynamic arrivals. vLLM-based; RTX 4090 (7B pair), A100 (13B), 2×L20 (32B, 128k test). **No CUDA-graph
mention anywhere** — their γ-switch pays a reconfiguration cost (C_switch 17.9-102 ms measured!) that
our exec_[T] family ZEROES (any T replays from a captured family; TODO B3's "our exec_[T] family
zeroes their dominant cost term"). Mapping to us: our γ is the keep rule's joint-product threshold;
their batch-size axis is our ACTIVE-SLOT axis (solo → cohort), not client count.

**S2. Su, Giannoula, Pekhimenko — "The Synergy of Speculative Decoding and Batching" (arXiv
2310.18813; RTX 3090/4090/A100, OPT-6.7B+125M). [CORROBORATED]** Full HTML fetched. Direction
confirmed with the reason: **optimal speculation length DECREASES with batch size** — at batch 32,
length >2 hurt; model t ≈ N·(α_b·s + β + t_S·s)/(c·s^γ + 1), l(s) = 0.9·s^0.548 (sub-linear
acceptance growth), α_b grows with b, and the optimality condition's root in s falls as α_b rises.
2.73× at batch 1, 1.31× at batch 32; adaptive beats every fixed scheme.

**S3. TurboSpec (arXiv 2406.14066, v3 2025-07; Liu et al., UCSD/Berkeley). [CORROBORATED]** Abstract
fetched: closed-loop goodput controller choosing speculation level at runtime in vLLM; "highest
goodput in runtime"; no fixed numbers on the abstract page. The principled harness for any adaptive
horizon (our dormant Controller, include/strata/spec/controller.hpp:23-30, is the same shape).

**S4. MagicDec (arXiv 2408.11049, v5 2025-04; Sadhukhan et al., CMU). [CORROBORATED]** Abstract
fetched: with batch and context growth the bottleneck shifts to the KV cache (draft's KV included),
which is why naive draft length must shrink with batch; with a SPARSE-KV drafter, SD can help MORE at
large batch (up to 2.51×, Llama-3.1-8B, batch 32-256). The counterweight inside the counterweight:
the "spec dies when busy" folk claim is about DRAFTER-KV-bound dense drafting; our drafter shares the
target's chain and KV (2 rows staged in the verify window) — we are structurally on MagicDec's good
side, not Nightjar's bad side, at our widths (6-9 slots ≪ 32).

**S5. Red Hat Developers, "Performance improvements with speculative decoding in vLLM with GPT-OSS"
(2026-04-16). [CORROBORATED]** Search-verified with the correct URL
(developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss;
the /vllm suffix 404s — T35's URL is dead). At concurrency 100, spec decoding holds TTFT P95 at
~10.0 s vs 12.3 s baseline (**−18.7%**); gap persists to 200+ concurrent; ~20% better throughput on
code-heavy workloads. The primary source that spec HELPS the tail under load when the drafter is
cheap relative to the target — our class (marginal row ≈ 2.2-2.6 ms riding the target's weights vs
their 0.5-1.3B separate model).

**S6. ArcticInference issue #185 (2025-09): suffix/spec benefit degrades at high concurrency.
[CORROBORATED]** (Carried from T35's source 16 as the honest counterweight; not re-fetched — a
single practitioner report against S5's measured sweep; resolved in favor of S5 + our own F-273
measurement, see S10.)

**S7. SGLang v0.4 (LMSYS blog, 2024-12-04). [CORROBORATED]** Full blog fetched. Overlap scheduler:
"runs one batch ahead and prepares all the metadata required for the next batch"; "future tokens" +
CUDA event scheduling; Nsight-verified five decode batches with no GPU idle; **1.1× vs v0.3, 1.3× vs
baselines**; on by default (`--disable-overlap`); idea credited to NanoFlow. Gains largest on small
models / high TP — i.e., exactly where host work is a large fraction of a short step (our ~10-13 ms
windows qualify).

**S8. SGLang Spec V2 + DFlash (LMSYS blog, 2026-06-15; Z Lab/Modal/SGLang). [CORROBORATED]** Fetched.
The current shipping form of the same pattern applied to speculation host work: host
`pop_and_process`(batch N−1) (stop detection, metadata) overlaps GPU batch N; host KV allocation in
`prepare_for_decode`(N) overlaps GPU N−1. Spec-V2-with-overlap **+33%** (11.4→15.3 ktok/s, Qwen3-8B,
1×B200, conc 32). Draft worker wraps the target for "parallel verification passes"; KV injection
materializes projections ahead of the draft forward (layer-batched linear + fused norm+RoPE) to keep
radix-cache sharing. This is the closest shipping analog to our C2 (host work of round N+1 under GPU
round N) AND to the batch-draft materialization question.

**S9. vLLM V1 async_scheduling + TRT-LLM overlap scheduler. [CORROBORATED]** vLLM: `async_scheduling`
config (docs.vllm.ai scheduler config) — "avoids gaps in GPU utilization"; default-ON in v0.14.0
(Medium release note; the AWS Neuron design doc describes the same one-step-ahead split: scheduling,
input prep, output processing overlap device execution). TRT-LLM (docs 1.2.0, features/
overlap-scheduler.html, fetched): "At step n, GPU work for step n+1 launches without waiting for step
n's CPU tasks… CPU work (step n) and GPU computation (step n+1) run concurrently"; stated cost: "The
optimization introduces one extra decoding step" (one step of latency); the TRT-LLM onboarding doc
(GitHub, .claude/skills/...) states the contract: the overlap "hides host work behind GPU work only
if all three hold — Contract 1: Zero CPU-GPU syncs inside forward." Both cite NanoFlow (arXiv
2408.12757, fetched: nano-batches overlap heterogeneous resources within one device, 1.91×,
50-72% of optimal) and SGLang v0.4 as the lineage. **The pattern's admission ticket in every one of
them: no synchronous host-GPU round trip inside the overlapped region.** Our cohort round's sync
points are exactly enumerable: the adapt join (:12202-12210), draft_finish waits (:12181-12189),
apply_pending at window entry (:15827-class), and the capture fence (:11973-11979).

**S10. OUR OWN under-load acceptance data. [MEASURED-OURS]** F-273: 9-conc acceptance 77.98% — the
highest ever measured on this engine (solo/pair M3b 75-76%); 9-conc 322.5 at 6-parity with M1 209.3
in-band. T34: 9-conc ITL p99 (mean-of-gaps artifact, but the acceptance numerator is real). The
direction agrees with S5 and contradicts S6: our drafts pay MORE per window when batched (window
amortization: fixed costs spread over 6-8 rows; the draft row's marginal c_row buys 0.78 expected
tokens at 78% — above the conc break-even ~0.55, T14C c_row 4.62 class).

**S11. CUDA conditional nodes + device graph launch (CUDA 13 Programming Guide,
docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html, fetched 2026-10-22;
+ the NVIDIA blog "Dynamic Control Flow in CUDA Graphs with Conditional Nodes", 2024-05-10,
fetched). [CORROBORATED]** Every device-side limit in the task brief VERIFIED against current docs:
- IF/WHILE nodes: CUDA 12.4; ELSE + SWITCH: 12.8 (13.4 has both).
- Handles: `cudaGraphConditionalHandleCreate`; with `cudaGraphCondAssignDefault` the value "is
  initialized to the specified default at the beginning of each graph execution" — a skipped step
  leaves downstream handles at 0 and their bodies do not run.
- Body graphs may contain: kernel, empty, memcpy, memset, child graph, and NESTED conditional nodes.
  NOT allowed: cooperative launches (with MPS), CUDA-array copies, device-side graph launch FROM a
  body kernel. Same single device/context throughout.
- Bodies populated by graph API (`phGraph_out`) or `cudaStreamBeginCaptureToGraph`.
- **The 120 / 255 / 1 limits are DEVICE-LAUNCH limits and do not bind a host-launched graph:**
  "A graph can have up to 120 total fire-and-forget graphs during the course of its execution. This
  total resets between launches of the same parent graph." / "A graph can have up to 255 pending
  tail launches." / self-launch allowed but "a given graph can only have one self-launch enqueued at
  a time." / "launching a device graph from the device while an update is being applied will result
  in undefined behavior." Our design (C1) launches from the HOST and sets handles from kernels INSIDE
  the graph — none of these apply. (Verified this distinction explicitly; the task brief's worry
  list is satisfied by construction.)

**S12. vLLM Dynamic Speculation Length RFC #36657 (opened 2026-03-10, closed not-planned 2026-07-27;
fetched in T27-2). [CORROBORATED via T27-2's fetch]** Batch-mean confidence stopping rule, τ 0.4-0.6;
their named future work: a DraftLengthPolicy abstraction AND "device-side control flow" — the largest
serving stack proposed, then shelved, exactly the door C1 occupies. Their closure reasons
(maintenance burden) price C1's scope discipline: draft-loop only, verify/commit untouched.

**S13. AdaEAGLE (arXiv 2412.18910) + Adaptive Speculative Decoding with RL (arXiv 2603.01639,
2026-03-02). [CORROBORATED, list-level]** Trained draft-length predictors (auxiliary module / RL
policy). Confirm the adaptive-horizon door is crowded; none beats a calibrated joint-product rule on
cost where the per-draft confidence is already free (our dprob; T26-2 §c.3 step 2's within-bin-spread
test remains the only re-opener for a trained head).

**S14. EQSPEC/EXSPEC — "Correctness Forensics for Batch Speculative Decoding: Diagnosing the Ragged
Tensor Problem" (arXiv 2510.22876, Findings of EMNLP 2026; fetched). [CORROBORATED]** Popular batch
spec-decoding implementations "silently produce corrupted outputs (repetitive tokens, <unk>)" while
showing competitive speed; root cause: variable per-request acceptance desynchronizes position IDs,
attention masks, KV cache (the "ragged tensor problem"); their EXSPEC (same-length scheduling)
achieves 3× at batch 8 with **95% exact match** — the remaining 5% attributed to FP
non-determinism. Our contract (exact-match walk + coupled Philox draws, outputs = f(seed)) is the
invariant their paper shows others break; never trade it (§4 #4).

**S15. Su-lineage goodput terms for the governor. [CORROBORATED via S2/S3]** The governor's objective
must be EXPECTED TOKENS/MS per round (goodput), not acceptance rate — T14C's break-even
(marginal > c_row·C0/W0) is the same object; the T31 refit already fits it per-position.

**S16. eunomia CPU-noise study + P-PAS + TWP + FairInference. [CORROBORATED via T35's fetches,
2026-10-23]** Carried unchanged: T35-1 (host isolation), T35-7 (pressure-switched decode share — the
P-PAS two-endpoint form), T35-8 (TWP pacing gate), T35-5 (VTC-lite at 7+). This file COMPOSES with
them and does not re-derive.

---

## §3 IMPLEMENTER-READY CANDIDATES (VALUE-RANKED)

Every candidate: mechanism → expected band per lane (with the window-ms math) → cost-anatomy code-read
(the F-232 law) → falsifier-first design → composition + law-4b scoping → rollback.

---

### C1 (rank 1). THE CONDITIONAL-GRAPH DRAFT LOOP + DEVICE-SIDE KEEP RULE — the solo unlock

**Mechanism.** Replace the serial drafter's launch shape (capture_round + per-step graphs launched
from a host-gated loop with a `wait_step` mapped-memory spin between steps, mtp.cpp:1735-1775) with
ONE host-launched conditional graph:

```
top graph = [round child] → IF h1 { step-1 body; cond-tail kernel j=1 } → IF h2 { step-2 body; cond-tail j=2 } → … → IF h_{S_mtp-1} { step-(S_mtp-1) body; cond-tail }
```

- Handles h1..h_{S-1} created with `cudaGraphCondAssignDefault`, default 0: a step that is not kept
  sets its handle to 0 and every downstream body is skipped ON DEVICE (the guide's default-at-launch
  semantics, S11).
- The branch predicate: computed in a ≤8-instruction tail appended to the step body (extend
  `mtp_select_kernel` verify_kernels.cu:713-724 or a 1-thread kernel after it):
  `c = ∏_{i≤j} draft_cum_acc(hist[i])` in float32, ascending i (hist = the window's mapped
  `out_p[]`/m_prob_ the select kernel already publishes past the threadfence); keep =
  `(c >= th_bits) && (p >= minp_bits)`; then `cudaGraphSetConditional(h[j+1], keep ? 1 : 0)`.
  Thresholds reach the device by MEMORY — two new mapped words beside `h_row_[0..1]`
  (mtp.cpp:1646-1647 → `h_row_[2]=min_p bits, h_row_[3]=th bits`, the basalt h_row_ pattern, T26-2
  §a.3) — so a per-window adaptive threshold costs nothing at capture and the th=0/th≤0 path
  degenerates to today's flat gate bit-for-bit.
- The capture uses `cudaStreamBeginCaptureToGraph` to populate each body from today's
  `capture_step` body contents (record_forward(1,row) + mtp_select), exactly basalt's `capture_cond`
  shape (T26-2 §a.3, donor code read at their :1241-1323).
- **What replays when the row count differs:** the VERIFY window is untouched — it still launches
  `exec_[T]` (capture at verify.cpp:2512, launched at :2988) with the T the rule decided, from the
  T-keyed family that already exists for every T the engine has seen. The conditional graph covers
  the DRAFT LOOP ONLY; the host still reads the produced drafts from mapped memory and stops on the
  −1 sentinel (mtp.cpp already writes the sentinels). This is the same scope line F-229's twin drew.
- **Why this unlocks the keep rule (the F-269/F-270/Nightjar triangle):** F-270 measured the cut-only
  keep rule at M1 −4.4% solo because a cut window still pays the per-window fixed work (stage, launch,
  K/V catch-up, commit, doorbell structure) while losing offered rows linearly — on the memory-bound
  serial tail, rows ride the weight read nearly free (the same roofline Nightjar uses to explain why
  LONG horizons win at batch 1). The conditional graph removes the launch-shape consequence of T:
  (a) the drafter's wasted-step tail vanishes on-device (F-229 measured the ungated twin −2.73%,
  decomposed ~0.19 ms/window GPU tail + ~0.1 ms host bubble — the conditional form keeps the gating
  but deletes both terms); (b) the keep rule becomes SHAPE-FREE, so it can arm where rows DO cost —
  the batch lane (bkey'd per-composition graphs, verify.cpp:3551) and the post-retrain world —
  without paying the solo tax again. That is Nightjar's γ\*(B) with our C_switch ≡ 0.

**Expected band per lane.**
- M1 (solo): **+0.3-0.9%** from the launch shape alone (F-229 decomposition ceiling: ~0.29 ms on an
  ~11.5 ms window ≈ +2.5% gross, realistic half — T26-2's re-priced band, carried). PLUS the keep-rule
  model-side value when composed (T31: +4.9% at th*=0.50 central, ALL of it the F-235 deep-position
  deficit — expected to fade after T14B; the battery decides; never double-count with the retrain).
- Pair: the F-270 both-arm read (keep+dm) cleared the pair bar at **+4.9%** — C1 is the form that
  lets the keep half stand without its solo penalty.
- Conc: ~0 direct (the batch lane's depth is already 1 draft/slot); indirect via dm-composition only.
- All bands [HYPOTHESIS] with the arithmetic above; the T31 model's pooled-bin floor (+0.05%) is the
  honest floor if the position effect is an artifact.

**Cost-anatomy code-read (file:line, this session).** The step bodies are
`capture_step` (mtp.cpp:1366-1391: catch-up K/V + full head layer on the anchor row + one row per
step from the previous step's residual/token); the spin being deleted is `wait_step`
(:1735-1743) inside the gated loop :1745-1782 (STRATA_DRAFT_UNGATED twin :1716-1731 = the −2.73%
measured twin); the mapped-output publication is `mtp_select_kernel` (verify_kernels.cu:713-724,
threadfence :724) + `h_row_` mapping (mtp.cpp:412, :1646-1647); the drafter shares the target's
kernels (q_chain dispatch, verify.cpp:1650-1654) so no new kernel family; no cooperative launches in
the draft path (grep: zero); in-capture memcpys are D2D only. Graph memory: the conditional graph
duplicates the step body ~S_mtp−1 ≈ 5-7 times (few hundred nodes; ~1.5 µs/node instantiate ≈
ms-class boot cost, ~10 MB-class graph-pool per the TRT-LLM graph-size datum T33 carried) while
ROUND graphs (`round_exec_[T]`) plus per-step graphs (`step_exec_[j]`) are REPLACED — net graph count
falls.

**Falsifier-first design (the th=0 bit-identical boot is GATE 0, exactly the model example).**
- **GATE 0 (one boot, kills the machinery if it fails):** conditional graph armed, rule OFF — handles
  implement `keep = p ≥ min_p` (today's flat gate, same offers). Requirement: bit-identical outputs
  (golden hash) + M3b in-band + identical offered-width histogram vs standing on one probe boot. Any
  bit difference = the graph machinery is wrong; kill before touching the rule.
- **GATE 1 (prices the band, one instrument boot):** capture `g_dt.gap_ms` (built, never captured —
  T26-2 unknown #3; STRATA_DRAFT_TIMING). If the inter-step gap < ~0.05 ms/window the launch-shape
  band collapses to ~+0.3% and C1 re-prices to "keep-rule enabler only" (still worth it for the
  composition, at lower rank).
- **GATE 2 (the battery):** five-boot quiet window, arms = {ctl, cond-only (rule off), cond+keep th 0.50,
  cond+keep th 0.40 + dm}, all three lanes, M3b + KL yardstick on every adoption-class arm (F-263
  law governs).

**Composition + law-4b scoping.** Composes with dm (the F-270 both-arm pair +4.9% is the designed
scope); composes with T14B by SEQUENCING, not stacking (the keep half retires after a successful
anti-drift retrain — T31 §6; the launch-shape half survives it). Law 4b: the rule NEVER ships
unscoped solo; the conditional graph is precisely the mechanism that scopes it — the same graph
serves "cut when it pays, extend when it pays" per lane without shape consequences. The
post-retrain EXTENSION side (offering d<0.5 drafts when the calibrated product passes) is
deliberately NOT priced (T31's dark-extension +0.36% has zero calibration data behind it).

**Rollback.** `STRATA_DRAFT_COND=0` (default-off); capture failure latches `cond_off_` → per-step
graphs take over (basalt's fallback, T26-2 §a.3); keep-rule half independently `STRATA_DRAFT_CUM=0`.

---

### C2 (rank 2). ADMISSION OVERLAP INTO THE COHORT REACTOR — hide the read and the tails

**Mechanism (three separable pieces, in value order).**
- **C2a. Finish-A-under-B (window-tail overlap).** Today the poll loop waits for BOTH windows
  (:12067-12083: `while (la || lb)`) before EITHER cohort's tail runs (:12104+). Reorder: run
  cohort A's finish (acceptance walk, commit prefix, emit, next-draft launch) as soon as A's
  `batch_poll` reports done, while B's window is still in flight; B's tail likewise under the NEXT
  round's A window. This is the SGLang v0.4 / vLLM V1 / TRT-LLM one-step-ahead shape (S7/S8/S9)
  applied at our granularity; the commit graphs are already parity-keyed for exactly this class of
  interleaving (capture_commit_batch :3583).
- **C2b. Reader thread (the F-249-named lever).** The prompt read stops the reactor today; between
  pieces the slots decode for `decode_share × piece_ms` (:14702-14713, 0.5 default). Port the T26-4
  §3.1 spec verbatim: side-thread reader (`sp.run` shape, stop_at_chunk for BYIELD), the existing
  Doorbells thread iterated over `{ver, ver_b}`, main thread keeps launching cohort rounds on the
  UNBORROWED slots, adaptive tier stands still during the read (`!in_prompt_read` preserved),
  engagement gated on `n_active > o.batch/2` (solo/pair byte-identical). The doorbell-thread
  contract ("makes no CUDA call") requires `--pcie-mode kernel/direct` on this path
  (batch_read.hpp:48 ← generate.cpp dma flag; T26-4 resolved STRATA_DMA_BATCH=0 does NOT clear it).
- **C2c. Adapt-join deferral (pathway law micro-win).** The round JOINS the adaptive-tier thread
  before returning (:12202-12210) — a wait with no payer inside the round (healthy 0.62 ms/window,
  F-271-poisoned 6-10 ms visible here first). Defer the join to the NEXT round's pre-stage
  (apply-before-next-window is already the #463 rule at the standing path :15827-class): the copies
  keep landing under the next staging, bounded staleness by construction.

**Expected band per lane (window-ms math).** At 6 clients: F-233's round anatomy = two ~6-row windows
overlapped per ~26-29 ms round; committed ≈ 6 slots × 1.77 tok ≈ 10.6/round → in-window ideal
365-408 t/s vs measured 315-340 — a 15-20% gap living in the tails + admission stops + adapt joins +
draft waits. C2a: the finish tail is ~1/3 of round wall (bt_run vs bt_commit+bt_emit split, TREND
line terms); overlapping half of it saves ~2-4 ms/round → **conc +8-14% arithmetic**
(26-29 → 22-25 ms). C2b: admissions ≈ 14% of probe wall pre-cohort (T17B #4; likely higher now,
windows shrank — T26-4's caveat); decode share 0.5 already hides half → the overlap removes the
residual stop → **+3-6% net** of the kernel-mode tax (sub-1% at 2.0-2.3% miss, unmeasured — read it
from the same boot). COMBINED honest band: **conc-6 +5-9%**, 9-wide larger (reads serialize against
more slots). Pair/M1: untouched paths (the n_active gate). All [HYPOTHESIS] with the measured terms
above; C2a's true price is the FIRST read of the cohort-window instrument (C5).

**Cost-anatomy code-read.** The stop is real: the reactor branch returns after the tails
(:12213-12214); `read_part`'s decode-share budgets (:14604/:14651/:14702-14713) are mutually
exclusive with reactor rounds by construction; `batch_stage_rows` → `stage_batch` stages into
per-verifier buffers (verify.cpp:4046-4053) with the lazy capture gated on "NO window in flight"
(:12050 comment) — double-buffering the stage rows per cohort (the F-196 parity pattern, commit_bm_
ckey :3583) removes the capture hazard for KNOWN bkeys; new-bkey capture keeps the fence (F-233 law
9: transitions are where it bleeds). The doorbell thread exists (:13913-13971 T26-4 anchors, now
:14000-class) and `set_external_serve`/batch_serve/batch_poll exist per verifier (verify.cpp:4085/
:4137) — the port is wiring, per law 5 (read what exists).

**Falsifier-first.** (1) ZERO-code pricing boot: `STRATA_BATCH_DECODE_SHARE=0.9` one 3v3 — NOT an
adoption (F-263 killed the constant; this prices the reactor-stop share): if conc does not move
beyond noise, the admission share is dead and C2b dies free; if it moves +2-4%, the stop is real.
(2) C5's cohort-WSTAT one-liner + admission-wait counters (F-276 fields exist) — one leg reads the
tail-vs-stop decomposition directly. (3) THEN the C2a port (S, no new threads — reorders existing
tail code) as its own 3v3; C2b (M) only after C2a's verdict.

**Composition + law-4b.** Composes with C3 (9-wide needs every hidden ms), with T35-7 (once the
reader thread exists, the decode share becomes the pressure knob — the P-PAS switched form gets its
clean substrate), and with T35-1 (host isolation: an overlapped reader thread is one more thread the
pin must place). Law 4b: every piece is scoped to `n_active > batch/2` by the existing cohort gate —
solo/pair never see new code paths. TRT-LLM's contract (S9) is the design review checklist: no sync
inside the overlapped region (the adapt join deferral C2c exists precisely for this).

**Rollback.** `STRATA_COHORT_OVERLAP=0` (C2a/C2c) / the T26-4 gate for C2b; each is one launcher line.

---

### C3 (rank 3). ≥9-WIDE + THE REAL COHORTS=3 — the F-249 re-test, rebuilt from scratch

**Mechanism.** Three facts make 9-wide the right slot for a third verifier:
1. **kVerifyMaxT=8 is structurally lossy at 9 on two cohorts.** 9 slots split 5+4; the collect loop
   stops at `S + 2 ≤ 8` (:11998) → cohort A serves 4 slots/window and the 5th ROTATES and idles a
   round (next_cur :12007); cohort B fits 4×2=8 exactly. F-273's 9-conc 322.5 ≈ 6-parity is this cap,
   measured.
2. **A third cohort restores full width at 9.** 3+3+3 → every cohort a 6-row window (12-13 ms class),
   all 9 slots active every round, and TWO other cohorts' pool/doorbell work to hide under instead of
   one. The reactor structure generalizes: `batch_launch over an ARBITRARY slot set` is already the
   documented primitive (include/strata/core/verify.hpp:241, F-233 note); the round needs a ternary
   collect/stage/go/poll and a third finish tail.
3. **The F-249 kill never tested any of this** (§1.3 C-new-1).

**Expected band (arithmetic, all terms measured).** COHORTS=3 at 9 clients: round ≈ max-window
12-14 ms + tails; committed ≈ 9 × 1.77 ≈ 16 tok/round → in-window ≈ 1,140-1,330 t/s; with the same
15-20% tail gap observed at 6 → 9-conc ≈ **355-390**; composed with C2 (tails + reads) → **380-430**,
i.e. T26-4's 380-420 ceiling band becomes concrete: it was priced for parallel-8 4+4 (kVerifyMaxT
exactly 8); at 9 the same arithmetic arrives via 3×3. At 6 clients COHORTS=3 stays DEAD by F-233
arithmetic (2+2 fills both reactors; 2+2+2 narrows every window without adding a hiding place) —
the F-249 measurement, being void, does not even contribute this; the arithmetic does.

**Cost-anatomy code-read.** `cohorts` is an int (:6866) set only to 2 (:9360); the PV table at :9447
is already 2×2; `ver_b`'s init path (:9340-9361: set_stage, set_batch_graph_limit(64), pcie_mode,
init_slots) is the template a `ver_c` clones; the round's CohortWin struct + collect lambda
(:11979-12008) parameterize on (lo,hi) span — a third span is mechanical; the tails are
verifier-parameterized (:12104). M cost class. Memory: ver_c needs its own graph/exec pools (~the
ver_b footprint; F-249's TODO line 1109 estimated ~+130 MiB — re-price at capture-audit time).

**Falsifier-first.** (1) FIRST bank C5 (the cohort window instrument) — the re-test is unreadable
without it. (2) Config-only 9-wide control leg: parallel 9 + COHORTS=2 with the telemetry fields
(window p50/p99 per cohort once C5 lands, admission waits, adapt swaps, true ITL p99 at 9) — this
boot MEASURES the 5+4 rotation loss directly (A-vs-B window ms asymmetry + per-slot token counts)
and prices the cap before any engine work. (3) Then the ver_c port behind `STRATA_COHORTS=3`, gated
to `o.batch >= 9` in the launcher (law 4b: 6-client boots keep the 2-cohort mode), 3v3 ABAB at 9
clients, single-engine check every leg, fresh expert profile per leg (the F-271 hazard law).

**Composition + law-4b.** Composes with C2 (the third cohort gives C2a more to hide under) and with
the elastic sessions (9 slots exist only while used; the COHORTS=3 build inherits S_min semantics —
check es_demote against the ternary split). Law 4b: scoped to ≥9 by the launcher gate; the 2-cohort
path stays byte-identical.

**Rollback.** Do not export STRATA_COHORTS=3 (the gate refuses <2 verifiers by init failure —
engine prints and falls back to cohorts=1... verify: the :9352 path prints and continues with
`cohorts` remaining 1 → safe).

---

### C4 (rank 4). SPECULATION UNDER LOAD — the lane-scoped governor (law 4b, made concrete)

**What the sweep actually says for OUR two lanes.**
- Serial lane (solo, ≤3 active): memory-bound regime — Su/Nightjar say LONG horizons win (S1/S2);
  our banked F-228 stair + T31 calibration agree (the flat 0.5 gate already harvests most mass; the
  keep rule's joint product is the refinement). The governor here IS C1: shape-free cut/extend.
  No new mechanism; solo keeps depth 6.
- Batch lane (≥4 active): horizon already 1 draft/slot (§headline 2) — the "reduce when busy" move
  is pre-empted by construction, and the field's high-load harm (Nightjar's −30%, S6) applies to
  SEPARATE-drafter stacks whose draft rows carry drafter-KV cost (MagicDec's bottleneck, S4). Ours
  rides the target's window at c_row ≈ 2.2-4.6 ms and acceptance RISES under load (77.98% at 9-wide,
  F-273 = S10) — the Red Hat class (S5). **Spec stays ON under load on this engine; the folk claim
  and its refutation are both now sourced.**
- The genuinely open knob: **a SECOND draft row on the batch lane** (2 drafts/slot → 3 rows/slot,
  9 rows for 3 slots — needs kVerifyMaxT ≥ 9, the ledger's known recipe class: window tables
  s_wsum[4]→[5] etc., SVG-CUSTOM.md:6197). Economics at measured terms: row 2 costs c_row ≈ 2.4 ms
  (batch) and returns pos-1 marginal acceptance ≈ 0.84 exact (T31's conditional table) ≈ 0.84 ×
  4.7 ms/token ≈ 3.9 ms → EV ≈ +1.5 ms/row-slot ≈ +6-12% conc arithmetic MINUS the drafter's extra
  step per slot (draft_launch(3,…)) and the wider commit graphs. [HYPOTHESIS, never priced on this
  stack; the F-262 spec-8 conc −6.1% is NOT this shape (it widened the SERIAL family and the
  drafter/commit graph set at MT=8) — and F-262's mechanism read is flagged UNRESOLVED in §Open.]
- Governor form (if the second-row falsifier passes): Su/TurboSpec-style closed loop on MEASURED
  tokens/ms per lane (the DraftPolicy EMA machinery, draft_policy.cpp:110-155, is the in-tree
  harness; the dormant Controller is the U(n) form), lane-scoped: solo fixed, batch adaptive between
  1 and 2 drafts/slot on the round's realized goodput. NOT a batch-mean confidence rule (vLLM's
  shelved RFC, S12) — our per-slot dprob is the better signal (T26-2's kill of DSpark stands).

**Falsifier-first.** Zero-boot first: extend the T31 simulator to the batch-lane shape (1-2 drafts,
c_row 2.2-4.6, acceptance from the pos-0/pos-1 conditional marginals) — if the EV is negative on the
model, the knob dies free. Then one boot: kVerifyMaxT 9 + batch drafts 2 (the arm is a small collect/
draft_launch change) gated `STRATA_BATCH_MTP_ROWS=2`, 6-conc 3v3, acceptance + ITL true tails read.

**Rollback.** Env-gated default-off (1 draft/slot byte-identical).

---

### C5 (rank 0 for cost, rank 1 for leverage). THE COHORT-WSTAT INSTRUMENT FIX — one line, gates C2/C3

`window_ms.push_back(msd2(w0c, w1c))` in the cohort round (generate.cpp:12085 area, where
"cohorts: windows done" already prints), + the same for the commit/emit span if attribution is
wanted. Without it: C2a's band is arithmetic, C3's rotation loss is invisible, and every future
window-tail claim at ≥4 slots rests on means. This is T35-2's one remaining hole, found in code this
session (§1.2). S cost, zero scoreboard risk, falsifier = one 9-wide leg with fields non-empty.
Pair with surfacing `bt_adapt_wait` and admission waits (fields exist, F-276) into the same readout.

---

### C6 (rank 5). NEW LEVERS THAT COMPOSE (deliverable 5, short list — each one line of pricing)
1. **Cross-cohort tail pipelining beyond C2a**: pre-stage NEXT round's rows during THIS round's
   poll gaps for KNOWN bkeys (double-buffered stage rows, F-196 pattern) — removes the serial
   stage→go host segment (~1-2 ms/round class); only after C2a proves the ordering discipline.
2. **Pressure-switched decode share (P-PAS form, T35-7)** — gets its clean substrate FROM C2b's
   reader thread; the switched two-endpoint form is the only live shape (F-263 killed the constant;
   P-PAS's own static-endpoint data agrees, S16).
3. **ADMIT_SHORT_FIRST arming at the 7+ lane** — still the zero-th fairness lever (banked CONFIRMED,
   env-only, dormant); with C2b it stops interacting with the read (order-only admission under
   overlap).
4. **NOT candidates (this wave's additions to the negative ledger, §4):** window-packing sentinels
   (our windows already pack dynamically via collect; SGLang's sentinel trick solves a padding
   problem we do not have); NanoFlow-style intra-device SM partitioning as a NEW mechanism (our two
   cohort windows already provide the overlap axis; green-context/SM-carving is researcher C's
   pathway territory and carries the PDL-adjacent risk class).

---

## §4 NEGATIVES RESEARCHED-AND-REJECTED (with reasons; conditions that re-open each)

1. **Spec 7 / spec 8 as standing depth.** F-263 FINAL (7: the +8.5% was a ctl-dip artifact; all three
   T28/T29 adoptions reverted) and F-262 (8: conc −6.1%, all rounds negative). Re-open 7 only via
   C1's shape-free form (the keep rule at depth 7 is a DIFFERENT rule than the flat gate at 7); the
   8-token window is kVerifyMaxT itself. FLAG carried: F-262's mechanism sentence ("the batch lane
   pays for the widest window") does not follow from the current collect code (rows/slot fixed at 2,
   cap 8 regardless of --spec) — the real path is drafter/commit graph growth (MT=8 widens
   capture_commit :2646 CB = 2+MT and every per-slot draft graph set). Unpriced; one instrument boot
   would settle it; do not cite the F-262 mechanism as settled.
2. **COHORTS=3 at 6 clients.** The MEASURED flat (F-249) is VOID (the arm ran 2 cohorts; §headline 1).
   Dead at 6 by ARITHMETIC (2+2 fills both reactors; a third verifier at 2+2+2 narrows windows with
   no additional hiding place). Re-open condition unchanged and now honest: `o.batch ≥ 9` (C3).
   Bank the void-verdict correction in the ledger before any c3 work starts.
3. **Decode-share constant raise (share=1.0).** F-263 + P-PAS's own static-endpoint data (constants
   win nowhere; the gain is the pressure switch, S16). Live form: T35-7 after C2b.
4. **Batch-expansion verify / multi-proposal per row / non-exact acceptance (EQSPEC-class, arXiv
   2510.22876).** Their own forensics: variable acceptance under batching desynchronizes position
   IDs/masks/KV — silent corruption at competitive speed; their FIX achieves 95% exact match (S14).
   Our exact-match walk + coupled Philox contract is the invariant the paper shows others break;
   multi-proposal per row buys ≤1 survivor under exact-match (T17C §1.4). Never trade. Re-open
   never; this is a contract, not a trade-off.
5. **"Spec decoding hurts when busy" as an operating rule for THIS engine.** Contradicted by S5
   (Red Hat, TTFT P95 −18.7% at conc-100, cheap-drafter class) and by our own F-273 (9-wide
   acceptance 77.98%, highest measured). TRUE only for weak-separate-drafter stacks at compute-bound
   saturation (Nightjar −30%, S1; Su batch ≥16, S2; S6's practitioner report) — a different point in
   drafter-cost space. Our MTP-off re-test under cohorts (T17B candidate 8) stays demoted: the prior
   now says OFF would hurt the tail.
6. **vLLM-class preemption ladder / retract.** num_preemptions_total = 0 by design (prometheus.py);
   preemption exists only as cooperative BYIELD during reads. FairInference reaches isolation without
   preemption (S16); maya's fair-slice PAUSE remains the only decode-slicing route (charter-gated,
   7-12-client lane).
7. **Layered prefill / PDL-as-host-flip / process split / weight swapping / P-D disaggregation.**
   Carried from T35 §5 and T26-4 §5 with their reasons (no object in the window architecture; F-191
   no-op; Splitwiser loses graphs; AlpaServe seconds-class; single GPU). PDL's per-pair re-price under
   the NEW cohort window structure belongs to researcher C's pathways lens — not duplicated here.
8. **Trained draft-length predictors as a serve-side arm (AdaEAGLE, RL-SD, DSpark).** Our dprob is a
   free per-draft confidence and the gate/keep rule consume it; a trained head pays only if the
   refit's within-bin spread is wide (T26-2 §c.3 step 2 — the standing re-opener). The RL variant
   (S13) additionally violates the bit-exact offers contract risk class unless offers-only.

---

## OPEN QUESTIONS
1. F-262's spec-8 conc mechanism (§4 #1 flag): drafter/commit graph growth vs window width — one
   instrument boot (C5 fields on a spec-8 shelf boot) settles it.
2. The true cohort-window tail at 6 and 9 (C5's first read) — the number every C2/C3 band above
   currently rests on indirectly.
3. Batch-lane second-draft economics on OUR stack (C4's falsifier); the batch lane has no
   STRATA_SPEC_TRACE equivalent — if the knob goes anywhere, the trace instrument needs a batch twin.
4. g_dt.gap_ms (carried T26-2 unknown #3) — C1 GATE 1.
5. Whether cudaGraphExecUpdate-style graph CLONING across identical-shape cohorts could cut ver_c's
   capture/pool footprint (T33's open question, still open; no production prior art found).
6. Does the adaptive tier's round-local join (C2c) ever exceed its healthy 0.62 ms under 9-wide
   churn (the poisoned-profile class says it can hit 6-10 ms — the hygiene law controls it, the
   instrument should watch it per-window).

## IMPLICATIONS
- The conditional-graph keep rule (C1) is ready for an implementer: GATE 0 (th=0 bit-identical) is a
  one-boot kill switch, the CUDA 13.4 limit list in the task brief is verified and does NOT bind a
  host-launched design, and the law-4b scoping is the graph itself (shape-free keep).
- The admission-overlap ladder (C2a → C2b → C2c) replaces "PIPED port" as the conc lane's head with
  a cheaper first step (tail reordering, no new threads) and a zero-code pricing falsifier
  (decode-share 0.9 leg).
- The 380-420 ceiling is concretely reachable only through 9-wide, and 9-wide wants the ternary
  reactor (C3) — whose prior "measured flat" must first be retracted in the ledger (void test).
  The config-only 9-wide telemetry leg prices the kVerifyMaxT rotation loss before any ver_c work.
- Speculation policy on this engine should be stated as: solo long (depth 6, keep-rule refined via
  C1), loaded short-by-construction (1 draft/slot), never off — with the batch second-row knob as
  the only unpriced depth axis.

## SOURCES

### Local (primary for §1; all read this session at HEAD bf746391)
Code: src/core/verify.cpp (:1113, :2494-2513, :2512, :2646, :2921, :2988, :3551, :3583, :4046, :4054,
:4085, :4137); src/program/generate.cpp (:6866, :9340-9361, :11558, :11667, :11721-11733, :11754,
:11965-12214, :12143, :12168-12177, :12202-12210, :12428, :14222-14226, :14604-14713, :15046-15048,
:15695-15701, :15827-class, :15921-15962); src/core/mtp.cpp (:412, :1316, :1366, :1646-1647,
:1716-1782); src/kernels/cuda/verify_kernels.cu (:713-724, :795-799); include/strata/kernels/
verify_kernels.hpp:21; include/strata/core/verify.hpp:241; serve/server.py (:310-317, :943-999,
:1714-1729, :1791-1816).
Record: SVG-CUSTOM.md F-232 (:7942), F-233 (:7965), F-235 (:8076), F-249 (:8460), F-261 (:8850),
F-262 (:8872), F-263/FINAL (:8899/:8921), F-264 (:8948), F-265 (:8975), F-266 (:9000), F-269 (:9047),
F-270 (:9070), F-271 (:9096/:9119), F-272 (:9154), F-273 (:9174), F-275 (:9191), F-276 (:9221);
SVG-CUSTOM-FEATURES.md (whole); AGENTS.md (laws 4b, 9, campaign law); launch-custom-iq3s-8081.sh;
custom-iq3_s-8081.json; SVG-CUSTOM-TODO.md (T35 board incl. B3, lines 1750-1878).
Campaign evidence: campaign/results/t21b-c3-arm-r1.out (the void-verdict boot line);
t21b-cohorts3.log; git history sweeps (-S "ver_c", -S "cohorts == 3", -S "STRATA_COHORTS=3": no
3-cohort code in any revision; F-249 commit 5a3e9356's generate.cpp read directly).
Research files (required reading, integrated not restated): T35-serving-latency, T31-keeprefit,
T33-priorart-serving, T33-priorart-materialization, T33-dynamic-session-allocation (v1-v3),
T33-ourtree-verification, T33-serving-delta-notes, T26-2-spec-lane, T26-4-serving-conc,
T27-2-spec-model-frontier.

### External (fetched or search-verified 2026-10-22 unless noted)
1. Nightjar, arXiv 2512.22420 (abs + HTML v5, fetched): γ\*(B) formalism, C_switch, direction, hardware.
2. Su et al., arXiv 2310.18813 (abs + HTML, fetched): optimal length falls with batch; model + fit.
3. TurboSpec, arXiv 2406.14066 (abs, fetched): closed-loop goodput control.
4. MagicDec, arXiv 2408.11049 (abs, fetched): KV-bound drafting at batch; sparse-KV reversal.
5. Red Hat Developers blog 2026-04-16 (search-verified, correct URL recorded): TTFT P95 −18.7% at
   conc-100; S5's primary source.
6. ArcticInference issue #185 (2025-09; carried from T35): the counterweight, resolved.
7. SGLang v0.4 blog, lmsys.org 2024-12-04 (fetched): one-batch-ahead overlap, 1.1×, NanoFlow credit.
8. SGLang Spec V2 + DFlash blog, lmsys.org 2026-06-15 (fetched): overlap points, +33% at conc-32 B200.
9. vLLM scheduler config docs (async_scheduling) + v0.14.0 release note (Medium) + AWS Neuron
   async-scheduling design doc (search-verified): one-step-ahead as default.
10. TRT-LLM overlap-scheduler docs, nvidia.github.io/TensorRT-LLM/1.2.0 (fetched) + onboarding
    skill doc (GitHub, search) + disagg blog (search): step-n CPU under step-n+1 GPU; one-extra-step
    cost; zero-sync contract.
11. CUDA C++ Programming Guide, CUDA Graphs chapter (docs.nvidia.com/cuda/cuda-programming-guide/
    04-special-topics/cuda-graphs.html, fetched 2026-10-22): conditional-node semantics + the
    120/255/1 device-launch limits + host-update UB — all quoted in §2/S11.
12. NVIDIA blog, "Dynamic Control Flow in CUDA Graphs with Conditional Nodes" (2024-05-10, fetched):
    IF/WHILE 12.4, ELSE/SWITCH 12.8, BeginCaptureToGraph, body whitelist.
13. vLLM RFC #36657 dynamic speculation length (fetched in T27-2, 2026-10-21; carried).
14. AdaEAGLE, arXiv 2412.18910 + Adaptive SD with RL, arXiv 2603.01639 (search-verified, list-level).
15. EQSPEC/EXSPEC, arXiv 2510.22876, Findings of EMNLP 2026 (abs, fetched): the ragged-tensor
    corruption class; 95% exact-match fix.
16. NanoFlow, arXiv 2408.12757 (abs, fetched): intra-device overlap, 1.91×; the lineage anchor for
    S7/S9/S10-class overlap designs.

## METHODS NOTE (so nothing is re-purchased)
Productive: full F-ledger reads around every touched F-entry; every prior-dossier anchor re-verified
at HEAD (all shifts recorded in §1.2 — do not trust old ranges); t21b boot-output archaeology (the
void-verdict discovery came from reading campaign/results/t21b-c3-arm-r1.out, not from any document
— the boot line contradicts the battery's label); git -S sweeps across all revisions for 3-cohort
code; two fetches of the CUDA 13 graphs chapter to separate host-launched conditional-node semantics
from device-launch limits (the task brief's limit list applies to the latter only). Web: 15
fetch/search rounds; every load-bearing claim fetched at least once this session.
Unproductive / warned-off: arXiv 2410.18721 is a physics-education paper — AdaEAGLE is 2412.18910
(do not repeat); the Red Hat URL without the `-gpt-oss` suffix 404s; T35's title for 2510.22876 is
wrong (Correctness Forensics…, not "Batch Speculative Decoding Done Right"); there is NO dormant
ver_c branch or commit — do not look for one; "TWC" as a scheduler name does not exist (T35's
warning stands). The engine-side WSTAT cohort gap and the F-249 void verdict are NEW this session —
if a later dossier contradicts them, it must cite a boot line or a commit, not a prior dossier.
