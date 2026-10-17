# T13B — CONCURRENCY & SCHEDULING AUDIT: how concurrent streams share the engine, and what binds aggregate throughput

Date: 2026-10-16 (audit performed while an ABAB planarm/planctl battery was live on 8081 — untouched; /metrics read once, read-only).
Scope: the concurrency/scheduling path of the standing IQ3_S boot (`--batch 6`, `STRATA_BATCH_MTP=1`, `--spec 6`, sm_120, single GPU, engine 0.1.44.1).
Method: line-level read of generate.cpp / verify.cpp / expert_source.cpp / pool.cpp / expert_cache.cpp / serve/server.py + probe-lane.py, live /metrics, live engine log lines, and ~40 recent battery result files. T11/T12 items are marked **known**; new ground is flagged **NEW**.

---

## 1. THE CONCURRENCY SHARING MAP (file:line evidence)

### 1.1 Intake and queueing
- serve/server.py owns HTTP; the engine is a child process spoken to over stdin/stdout lines. Concurrent requests are assigned to `--batch` slots: server.py builds `slot_q`/`slot_busy` per slot and routes `BT`/`BDONE` lines back to the owning request (serve/server.py:921-932, 972-984). Config `parallel: 6` + `--batch 6` → 6 slots (live /metrics: `batch_slots: 6, parallel: 6`).
- Admission = `BGEN <slot> <max_new> ...` (generate.cpp:10585-10599). The engine thread reads request lines **only between windows** (`try_next_line` over the mutex'd `in_lines` deque, generate.cpp:10041-10047, checked at the top of the main loop :10551-10560).
- Admission of a conversation into a slot = `copy_to_slot` (generate.cpp:10049-10086): checkpoint save/restore **per stage** and **per QSA K/V state with `cudaDeviceSynchronize()` between every save and restore** (:10055, :10065, :10069), plus the slot drafter's K/V image (:10077-10084). Live log: "slot takes 318 tokens (copied in 36.0–55.4 ms)" per admission. **NEW (not in T11/T12): the per-QSA-layer sync chain is a measurable admission stall — 6 admissions ≈ 0.25–0.33 s per 6-conc probe (~2% of a 14.2 s probe; more on short multi-turn traffic).**
- While a prompt is being read, the slots keep decoding in a time-sliced interleave: `read_part` runs batch windows between prompt chunks for `decode_share = 0.5` of the chunk-read time (generate.cpp:11801-11804, 11879-11888). `STRATA_BATCH_YIELD_CHUNKS=8` gates the piece size; BYIELD is cooperative preemption into a slot (:11810-11878).

### 1.2 The single service thread's loop
- ONE thread runs everything GPU-serving: the main loop (generate.cpp:10551) alternates reading request lines and calling `batch_step()` (:10135). `batch_step` collects active slots into rows and issues **one synchronous batch window**: `ver.run_slot_rows(...)` at :10199.
- **Row cap**: `kVerifyMaxT = 8` (include/strata/kernels/verify_kernels.hpp:21). The row-collection loop `S + (batch_mtp ? 2 : 1) <= 8` (generate.cpp:10144) means **with STRATA_BATCH_MTP=1 each slot takes 2 rows → at most 4 slots decode per window**. At 6 concurrent requests, `next_slot` rotation serves cohorts of 4 (e.g. {0,1,2,3}, {4,5,0,1}, {2,3,4,5}; rotation at :10165) — every window is full (8 rows) but each request advances in 2 of 3 windows.
- `run_slot_rows` (verify.cpp:3110-3220): `stage_batch` (staging + graph-layout key + **per-window flag resets**, :3089-3092) → plain `cudaGraphLaunch` of the captured batch graph (:3116) → then the per-(layer,group) service loop :3153-3207:
  - **Wait site 1 (doorbell)**: spin `while (*seq < want) _mm_pause()` on the mapped ring the GPU's `doorbell_publish_value` kernel bumps (verify.cpp:3161-3178; kernel at src/kernels/cuda/elementwise.cu:355-373 — one 1024-thread block, writes the 80 KB fp32 activations + ids + weights + seq per layer-group).
  - **The pool call is synchronous on this thread**: `pool(user, h_x_..., h_ymiss_..., l)` at verify.cpp:3188 → `expert_pool_dispatch_multi` (expert_source.cpp:3694).
  - **Flag raises**: if the pool published no GPU plan, an empty one is published and flagA/flagB raised (:3194-3203); then `*flag = want` (CPU rows ready) at :3204.
  - Accounting: `ms_wait` (spin) + `ms_pool` (pool call) per step (:3205-3206); window ends with `cudaStreamSynchronize` + `cudaStreamSynchronize(copy_)` (:3208-3210).
- In-graph wait sites (the GPU's side of the same handshake): `wait_flag_ge(m_flagA_, ring)` pre-VRAM-GEMM (verify.cpp:1691, or the fused wait+copy at :1689), `wait_flag_ge(m_flagB_, ring)` pre-PCIe-groups (:1698-1699), `wait_flag_ge(m_flag_, ring)` pre-combine (:1719, or fused `copy_rows_from_mapped_after` at :1724). These three spins are why the pool term is *additive*: the GPU cannot run past layer l's flags until the host has served layer l.
- After the window: acceptance (exact-match on the slot's one draft, + the F-28/F-30b rescue **OFF by law**, generate.cpp:10218-10258) → `commit_slot_prefixes` (:10259) with the F-196 parity double-buffer + async commit (verify.cpp:3228-3271; `commitb_pending_` retired at the next commit :3234-3238) → **F-206 de-serialized drafts**: all slots' `draft_launch` first, then the waits (:10275-10337; MtpDrafter contract at include/strata/core/mtp.hpp:109-114) → per-token `printf("BT ...")` + one `fflush` per window (:10297, :10338) → adapt tier join (:10340-10351).
- The pipelined machinery (`pump`, batch_launch/batch_poll, generate.cpp:10432-10530) **cannot engage on this rig**: `piped = o.batch > 0 && o.batch_groups > 1 && n_pipe > 1` and `n_pipe = stages + 1 = 1` on a single GPU (:10407-10408); batch_groups > 1 without a layer split is refused at :4642. `batch_poll`'s doorbell loop exists and is single-GPU-safe (verify.cpp:3327-3380) — the machinery is there but only one window can ever be in flight because **all doorbell/flag buffers (`h_seq_`, `h_flag_`, `h_flagA_`, `h_flagB_`, `h_x_`, `h_ymiss_`) are per-Verifier singletons reset at every `stage_batch`** (verify.cpp:3089-3092).
- STRATA_BATCH_G2 (two groups per window): env-gated, default off (verify.cpp:913-934, `batch_groups_` at verify.hpp:531) — **known**: closed for corruption + regression (campaign/results/g2-arm-r1..3: pair 3.9 t/s = corrupted output; g2-ctl ~255).
- HostGraphLaunch (STRATA_VERIFY_ASYNC_LAUNCH=1, standing ON) is wired into the **serial** `run()` only (verify.cpp:2342-2357); the batch path launches inline at :3116. **NEW confirmation at line level of the T12-HOST aside**: concurrency pays the inline graph-launch + the full staging on the service thread every window.
- Draft/verify GPU streams: window graph on `cs_`, shared-expert side stream `sh_cs_` (fork/join verify.cpp:1591-1594, 1740-1742), df_branch side streams (:948-956, solo/G=1 only), DMA copy stream `copy_` (:3210, :518), one MTP drafter per slot on its own stream (generate.cpp:4704-4745, draft copies at :10316-10321).

### 1.3 The pool call (what the 4.0–4.8 ms is made of)
- `expert_pool_dispatch_multi` (expert_source.cpp:3694-4095): plan build + publish FIRST (:3734-3945: the **STRATA_PLAN_ONEPASS single-pass seen-stamp planner already landed** at :3739-3811, default ON, `STRATA_PLAN_DIFF` self-check — **known (T12-HOST #1), currently being A/B'd live** as planarm/planctl), tail-rank skip (:3828-3860), PCIe share pick (:3866-3945), activation quant (:3982-3994), **O(m²) `std::find` miss list** (:3996-4004), job assembly (:4005-4057), then `pool->run_split_multi_native` (:4060-4062).
- `ExpertPool::run_phase` (pool.cpp:746-756) per phase: `wait_parked` → `begin_batch` → host drains → `wait_done` → **trailing `wait_parked` at pool.cpp:753** (**known** T12-HOST #2). Native multi = 2 phases/layer (gate/up rows mode 5, down rows mode 6, :817-846) → per window: ~4× wait_parked + 2× wait_done barrier pairs × 48 layers. Workers spin on `epoch_` then sleep after `STRATA_POOL_SPIN_US=100` (pool.cpp:529-593). Matvec is only ~0.25–0.35 ms/window (**known** T12-HOST); row-split tasks (mode 5/6 ranges spanning experts, pool.cpp:684-721) already spread 3–5 distinct missed experts over the 15 workers (live /metrics: `pool_workers: 15`).
- T12-HOST's caveat is code-true: ms_wait+ms_pool is an accounting identity; the *exposure* is exactly the in-graph flagA/flag_ gates (:1691/:1719).

### 1.4 Expert cache under multi-stream — NOT thrashing
- Residency is **no-eviction by design**: `ExpertCache::admit` returns kNotResident when full — "full: no eviction, deliberately" (expert_cache.cpp:653) — with per-layer quotas (R4.2g, :638-656, ranges at :624-631). Concurrent streams **cannot** evict each other's experts; there is no LRU to thrash.
- Live 6-conc evidence (batch summary lines): routed entries per window VRAM 97.9–98.0%, CPU 1.7–1.8%, PCIe 0.3–0.4%; 79.3–79.4 VRAM hits per layer-window; hit *rate* at 6-conc (98.0%) ≥ single-stream (97.7%).
- The adaptive tier swaps aggressively in the background (live: "133 rounds, 7263 experts swapped in" over 13.7 s ≈ 1.06 GB/s on refill streams) but its exposed cost is only `adapt wait 0.58 ms/window` (generate.cpp:10177-10184, 10340-10351); it stands still during prompt reads (`in_prompt_read`, :10177).
- kvg_/elastic K/V (`kvg_ensure`, generate.cpp:6881-6982) gives K/V room by giving up cold expert slots with heat-ordered device-to-device moves + `cudaDeviceSynchronize` + residency re-upload; called at admissions/RESTORE (:10831, :11121, :12891). At probe-scale prompts (≤900 tok vs `kv_resident 32768`) it never fires — **not on the conc-probe path**; a latency hazard only for long-context admissions.
- STRATA_KV_PREFETCH=1 and STRATA_DMA_BATCH=1 (standing env) are prefill/DMA-lane arms; STRATA_DMA_BATCH batches the main device's copies as one `cudaMemcpyBatchAsync` (generate.cpp:9076) — 84.9% fewer DMA submit cycles; neither is concurrency-specific.

### 1.5 Acceptance under concurrency (M3b-adjacent)
- Slot drafts verify **ONE proposal per slot per window** (batch_mtp depth 1: `draft[0]` at generate.cpp:10159; `draft_launch(2, ...)` at :10322). Live acceptance: **66.6–67.1% at 4-slot windows; 57.9% at pair** vs M3b (serial, sampled) 74–79% across ~30 recent arms. Rows/window = slots × (1 + accept) = 4 × 1.666 ≈ 6.55 observed.
- So the acceptance "dip" under concurrency is mostly *depth* (1 proposal vs the serial path's depth-6 chain that banks 2.78 tokens/window), not a per-proposal quality cliff.

### 1.6 The measured window anatomy (healthy standing boot, 4-slot windows)
From `custom-iq3s-ctxoff.log` / live log (identical shape in the live boot):
```
532 windows, avg 6.55 rows, 22.32 ms/window = run 21.11 (CUDA0 GPU-reach wait 16.47 + CPU experts 3.95)
+ commit 0.03 + emit 0.60 + adapt wait 0.58; ... 254.9 rows/s over wall (admissions included);
slot drafts accepted 1389 of 2087 (66.6%)
```
Pair windows: 14.20 ms, 3.15 rows, 202.9 rows/s wall. Single serial windows (context): 2.78 banked / 13.2 ms.

## 1.7 BINDING-CONSTRAINT VERDICT

**The binding constraint is the single-window serial chain owned by the one service thread: window time = GPU kernel serial (~16.4 ms) + additive pool term (~4.0 ms, host bookkeeping) + host tail (~1.2 ms emit/adapt/commit/staging) — with structurally only ONE window in flight.** Neither side alone is the wall:

- GPU utilization inside the concurrent lane ≈ 16.47/22.32 = **74%** — the GPU idles ~26% of each window at the flagA/flag_ gates and between windows while the host stages/launches/commits/emits.
- F-223's law holds in code: the GPU may not pass layer l's flags until the host serves layer l (verify.cpp:1691/1699/1719 vs :3188/:3194-3204), so no intra-window reordering can hide the pool — the T1.1 falsification and the flagA 0.26 ms number are consistent with the code shape.
- Expert-cache pressure: **ruled out** (98.0% hit at 6-conc, no-eviction design, 1.7% CPU share).
- Draft acceptance: **ruled out as the binding constraint** (66.6% is healthy for depth-1) — but it sets the rows/window multiplier (1.67), which is the second-order lever.
- GPU window occupancy: the 16.4 ms GPU serial is itself only ~3.3 ms expert bytes (91–100% of roofline, **known** T12-MEM/T12-GPU); the rest is small-kernel latency — so even the GPU side has headroom.

Corroborating arithmetic: removing *only* the host-added terms (pool 3.95 + tail 1.2) bounds the current shape at ~6.55/16.4ms ≈ 400 rows/s in-window — i.e. ≥300 aggregate is reachable by (a) hiding the host terms behind another window in flight, or (b) shrinking them and the GPU serial together, or (c) raising rows/window.

---

## 2. RANKED IMPROVEMENT CANDIDATES (target: concurrent ≥ 300 aggregate)

Baseline for arithmetic: 22.32 ms/window, 6.55 rows → 293.5 rows/s in-window, 254.9 rows/s wall (wall efficiency 0.868 incl. admissions). "≥300 wall" needs in-window ≥ ~345–355 rows/s.

**C1. R2-S3: two cohort Verifiers in flight (pipelined service reactor).** — **known (T11.R2 S3), with new line-level confirmation of the exact structural blocker.**
- Mechanism: two Verifier instances (slots 0-3 / 4-7 cohorts), each with its own `h_seq_`/flags/mapped buffers (today's per-window resets at verify.cpp:3089-3092 are the blocker), one reactor thread serving both via the existing batch_launch/batch_poll doorbell machinery (verify.cpp:3327-3380, generate.cpp:10432-10530).
- Impact: window period → max(GPU serial ≈16.4 ms + two-stream contention, host pipelined ≈5–6 ms) → 17.5–18.5 ms ⇒ 354–374 rows/s in-window ⇒ **307–325 wall (+20–28%)**. The two-stream contention factor is the one number S3 exists to measure.
- Cost L. Risk: the G2 ghost (buffer overwrite) — structurally impossible with per-verifier buffers; byte-exact token-equality + position-contiguity gates (bits unchanged: same graphs, same per-layer service order per cohort).
- Confidence: high (mechanism follows from the measured 26% GPU idle + additive host terms).
- Cheap falsification: none below S1 — but S1 (defer-raise payload parity, F-196-shaped) is the built-in first gate.

**C2. T12-HOST micro-package: pool bookkeeping off the critical path.** — **known (T12-HOST #1–#4), status update + arithmetic.**
- #1 onepass planner **has landed** (expert_source.cpp:3739-3811, default ON, STRATA_PLAN_DIFF verifier) and is in A/B right now (planarm/planctl; early arms r1-r2 read 230-241 vs ctl-history ~250 — inconclusive, do not close on it).
- Remaining: #2 barrier diet (drop trailing `wait_parked`, pool.cpp:753; keep 2-3 workers hot), #3 adapt join/apply off the service thread (generate.cpp:10340-10351), #4 doorbell publishes the q8_1 image the GPU already computes (elementwise.cu:355; host skips act quant, expert_source.cpp:3982-3994), + #5 emit/fflush and the O(m²) miss-list `std::find` (expert_source.cpp:3996-4004).
- Impact: pool 3.95 → ~1.3–1.7, tail −0.5 ⇒ window ~19.3–20.3 ms ⇒ 323–339 in-window ⇒ **280–295 wall alone; with C1: ~385 in-window, 334+ wall**. Bits unchanged except #4 (M3b canary + 2× MMLU-Pro, R4.2h scale-parity rule).
- Cost M total. Confidence: high for #1/#2/#3 (accounting identity verified in code).
- Cheap falsification: one battery arm with `wait_parked`-after env-gated off + `STRATA_POOL_TRACE` + the existing per-phase timers (pool.cpp:809-811, 839-841); read the ms_pool delta straight from the batch summary line.

**C3. Window-rows ladder: kVerifyMaxT 8 → 12 (and MAXT 8 → 12) so batch_mtp serves 6 slots × 2 rows.** — **NEW.**
- Mechanism: today only 4 of 6 concurrent slots decode per window (generate.cpp:10144). 12 rows/window ≈ 9.8 rows at 66.6% acceptance. GPU serial grows by the expert-bytes delta (~+1.7 ms; hits 79→119/layer-window at 2.05 MB on a line already at 91–100% roofline) while the latency-bound light lines scale sub-linearly.
- Impact (standalone, no C1): window ~24.5–25.5 ms, 9.8 rows ⇒ 384–400 in-window ⇒ **333–347 wall (+31–36%)**. With C1: GPU-bound at ~2×(16.4+1.7)=36.2 ms per two 12-row windows ⇒ ~500+ in-window theoretical (contention unknown).
- Cost L: verify.hpp fixed arrays (`brow_[8]`, `b_out_[8]`, `last_rows_[8]`, :342-348, :365), pool MAXT (include/strata/kernels/cpu/expert.hpp:143; `kMaxWindowEntries` 128→240, expert_source.cpp:3673-3674; thread_local `ff[MAXT][FF]` scratch pool.cpp:695), graph captures per new layout (LRU limit 64 already set, generate.cpp:8034), ~1344 MiB free VRAM must absorb the activation growth. Per-token bits can stay identical (exact-mode multi-column MMVQ) but grouping changes → differential gate + M3b.
- Confidence: medium (arithmetic solid, surface area large).
- Cheap falsification (config-only, today): 3-arm battery — [MTP=1, batch 6] (6.55 rows) vs [MTP=0, batch 6] (6 rows) vs [MTP=0, batch 8→ refused? no: --batch 8, MTP=0] (8 rows): measures the marginal rows→window-time curve that C3's projection rests on.

**C4. FastMTP retrain (acceptance lever).** — **known (T11.R3 GO).** Slot accept 66.6% → ~75% ⇒ rows/window 6.55 → 7.0 ⇒ **+7% ⇒ ~273-292 wall alone; compounds with everything** (with C2 alone it clears 300). Cost L + weekend compute; different bits → full gate. (The in-repo alternative — porting the typicality rescue to the batch site — stays OFF by the F-30b law: measured MMLU 91.4-92.9 vs 95.7.)

**C5. Arm LFUSE on the doorbell path.** — **NEW (the gate is dead code in the standing recipe).**
- `lfuse_on` requires `ar_on()` (verify.cpp:928-930) — false forever at 38% residency — AND `STRATA_LFUSE` is not in the launcher env: the shared-expert gate/combine fusion (T12-GPU #4/#6, native_moe_combine_multi_hits_gated path at :1743-1746) never fires. Widening the gate to the doorbell graph is a one-line arm plus the env.
- Impact: fusible small lines (combine 0.93/1.81 + quant) ⇒ +0.5–1.0 ms GPU serial ⇒ **+2–4%**. Different bits (gate application reassociated into the combine) → M3b + 2× MMLU-Pro.
- Cost S. Confidence: medium (T12-GPU measured the lines; the fusion functions exist and are gated).
- Cheap falsification: zero-code — the dead-gate fact is proven by the env/grep above; the value A/B needs the one-line arm behind its own env, then one battery.

**C6. HostGraphLaunch for the batch path.** — **NEW (line-confirmed solo-only).**
- verify.cpp:2344-2349 uses the threaded launcher in serial `run()`; batch launches inline at :3116. Wire the same helper into `run_slot_rows` (and `batch_launch`).
- Impact: the launch+query segment (~0.7–1.0 ms/window, the unexplained part of run 21.11 vs wait+pool 20.42) ⇒ **+3–4%**. Bits identical (launch mechanics only). Cost S–M.
- Cheap falsification: microbench `cudaGraphLaunch` of the captured 1.9–2k-node batch graph off-line vs the HostGraphLaunch path (no server boot needed).

**C7. Emit batching.** — **NEW.**
- `printf("BT %d %d\n")` per token (:10297) + `fflush` per window (:10338) through the pipe to server.py's reader (server.py:972-984): ~8 syscalls + reader wakeups per window = the 0.60 ms `emit` term. Buffer BT lines and write once per window (flush policy on BDONE unchanged for the server's line protocol — lines stay newline-terminated).
- Impact: **+0.4–0.6 ms/window ⇒ +2–3%**. Bits identical. Cost S.
- Cheap falsification: `bt_emit` in the batch summary line is already instrumented — one env-gated arm reads the delta directly.

**C8. Doorbell publish fusion (router epilogue).** — **known (T12-GPU #3), new kernel-level evidence.**
- `doorbell_publish_value_kernel` is a standalone 1-block/1024-thread kernel per (layer,group) (elementwise.cu:355-373) — ~48–96 near-noop launches/window, each paying ~6 µs node spacing. Fold the 80 KB publish into the router kernel's epilogue (~150 nodes/window).
- Impact: +0.3–0.6 ms GPU serial ⇒ **+1.5–3%**. Bits identical (same bytes, same order) → differential gate. Cost M.
- Cheap falsification: STRATA_VERIFY_PROFILE stage table — the publish/plan/bell stage is already a stamped line (kProfPer stamps).

**C9. Admission sync diet.** — **NEW.**
- `copy_to_slot`/`copy_from_slot` sync per QSA K/V image (generate.cpp:10055-10069, 10097-10116): 12 QSA layers × 2–3 `cudaDeviceSynchronize` + checkpoint save/restore = 36–56 ms/admission (logged). Chain the copies on one stream with a single sync per direction (the per-layer calls are already stream-ordered saves/restores).
- Impact: ~2% on the 6-conc probe (6 admissions ≈ 0.25 s of 14.2 s), more on short-turn chat; also shrinks the prompt-read window during which slots see only the 0.5 decode share.
- Bits identical. Cost M. Confidence: high on the mechanism, low-moderate on the metric move.
- Cheap falsification: zero-code — sum the existing "copied in %.1f ms" lines vs probe wall in any results file.

**C10. Multi-row sampler launch.** — **known (T12-HOST #6), sized.** `sample_rows` launches one `sample_tokens` per sampled row + a sync (verify.cpp:3286-3299); at 6-conc every probe row samples (temp 0.7). One multi-row launch ⇒ +0.1–0.3 ms/window on sampled traffic ⇒ **+0.5–1%**. Cost S–M. Falsification: stage profile around the head.

**C11. Decode-share during prompt reads.** — **NEW, minor.** `STRATA_BATCH_DECODE_SHARE` default 0.5 (generate.cpp:11801-11804): during multi-chunk prompt admissions the slots get half the interleaving budget. Raising to ~0.75 is a pure config experiment for admission-heavy traffic; at the probe's ~30-token prompts it is inert. Cost S (config). Falsification: one boot arm, read `strata trace: prompt chunk ... slots decoded %.0f ms` lines.

Not candidates (refuted or ruled out by this audit): expert-cache residency/eviction work for concurrency (no eviction exists; 98% hit at 6-conc); more pool workers (parallelism-limited bookkeeping, not matvec — T11.R4/T12-HOST); intra-window reordering (F-223); STRATA_BATCH_G2 (closed, corruption); kvg_/KV changes for the conc lane (never fires at probe scale).

---

## 3. CONTRADICTIONS vs THE CONTEXT FACTS

1. **"concurrent 210–219 aggregate" is stale.** Thirty-plus recent standing/config arms measure M2 6-conc **244–260** (e.g. standing-final-f224 253.7, ctxoff-boot 260.2, standing-final-g2 252.8, t11 arms 249–256, gdnc2 243–258). The in-flight planarm/planctl ABAB currently reads lower (230–241, r1–r2) — flagging, not resolving; the healthy-boot band is 244–260. The target "≥300" stands, but the baseline gap is ~15–18%, not ~40%.
2. **"pair 171–175" is stale**: recent arms measure pair 194–204 (post F-221/TYPICAL_DEVICE/ASYNC_LAUNCH adoptions). M1 213–215 matches (standing-final-f224 M1=214.7).
3. **Expert cache slots**: context says 9,337; the live boot reports **9,426 slots / 17.88 GiB** (boot log + /metrics). Same ceiling conclusion (38% residency) either way.
4. Context "pool term 4.0–4.8 ms/window" confirmed (3.95–4.29 measured at 4-slot windows); "T1.1 falsified" and "G2 closed" both confirmed in code and results (verify.cpp:913-934 env-off; g2-arm pair 3.9 = corruption).
5. No evidence found for any of the context's structural claims being wrong; the one material update is that **T12-HOST #1 (onepass planner) has already landed** and is mid-A/B — the "known pool fixes" list is partially spent.

## 4. OPEN QUESTIONS
- The two-stream contention factor for C1/S3 (the GPU already runs 6 graphs sharing SMs inside one window; a second verify graph is the unmeasured case). This is THE number S3 exists to produce.
- Why the in-flight planarm/planctl ABAB reads 5–8% below the standing band (thermal, ambient, or the arm genuinely neutral-to-negative) — resolve when the 3v3 completes before adopting onepass.
- Whether 12-row windows keep the CPU-pool entry tables and the drafter's residual copies within the 1,344 MiB free VRAM at all batch layouts (capture-time audit).
- Emitted-token latency impact of C7's flush batching on interactive TTFT/TBT (server.py line protocol is unaffected; perceived latency may shift by <1 window).

## 5. IMPLICATIONS
- The 300 target is reachable by two independent routes: **C1 alone (~307–325)** or **C2+C4 (~300–310)**; the compounding of C1+C2 (~334+) gives headroom. C3 is the largest single lever but an L-cost change best scheduled after S3's contention number exists (it multiplies whichever pipeline shape wins).
- Quality gates: only C2#4, C4, C5, C3 change bits — M3b canary + 2× MMLU-Pro each; C1/C6/C7/C8/C9 are bit-identical and gate on byte-exact differentials.
- The expert cache needs no concurrency work: the no-eviction + per-layer-quota design is load-bearing for the 98% multi-stream hit rate; protect it in any residency overhaul.

## 6. SOURCE LIST (primary = code/read at this commit; measurements = logs/results)
Code: src/program/generate.cpp (:10041-10599, :11800-11900, :4642, :6863-6982, :8034); src/core/verify.cpp (:913-934, :1563-1594, :1691-1731, :2342-2460, :3022-3101, :3110-3299, :3327-3380); include/strata/core/verify.hpp (:189-268, :531); src/core/expert_source.cpp (:3669-4095); src/kernels/cpu/pool.cpp (:529-663, :746-846); include/strata/kernels/cpu/expert.hpp (:143); src/core/expert_cache.cpp (:624-656); src/kernels/cuda/elementwise.cu (:355-373); include/strata/core/mtp.hpp (:109-114); include/strata/kernels/verify_kernels.hpp (:21); serve/server.py (:899-984); campaign/probe-lane.py (:35-98).
Measurements: live /metrics (2026-10-16 01:26); live batch summary lines (custom-iq3s-ctxoff.log tail; server-8081.out); campaign/results/{standing-final-f224,ctxoff-boot,standing-final-g2,t11-*,gdnc2-*,g2-arm/g2-ctl,planarm/planctl-r1-r2}.txt (~40 files surveyed); launch-custom-iq3s-8081.sh (standing env); campaign/battery-{boot,config}.sh; custom-iq3_s-8081.json + campaign/planneroff-iq3_s-8081.json (arm configs).
Prior dossiers folded in (marked **known**): SVG-CUSTOM-TODO.md T11.R2/R3/R4/R5 + T12-HOST/GPU/MEM (2026-10-16).

## 7. METHODS NOTE (unproductive searches, so they are not repeated)
- T11/T12 dossiers are NOT standalone files in campaign/research/ — they are sections of SVG-CUSTOM-TODO.md:764-964; searching campaign/research/ for "T11" wastes time.
- pool.cpp lives at src/kernels/cpu/pool.cpp, not src/core/ (the task brief said src/core/pool.cpp).
- /metrics carries no per-window or acceptance counters (only slot occupancy + totals) — window anatomy must come from the engine stderr "strata batch:" lines.
- vprof-boot-console.txt and prof221 references were not re-derived (T12 calibration, marked known).
