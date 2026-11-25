# F-453 — drafter launch-path audit (read-only, 2026-11-25)

Tree `/home/svg/LLM/strata-custom-2`. Nothing built, run or modified; grep and file reads only.
The solo decode loop is `src/program/generate.cpp:17160` (there is no `src/core/generate.cpp`).
"Measured" quotes the record: F-452 in `SVG-CUSTOM.md`, `campaign/results/f452-2026-11-24/draft-timing-lines.txt` and `nsys-anatomy.md`.
Every number obtained by subtracting two measured numbers, and every mechanism not measured on this tree, is UNCONFIRMED STATIC ANALYSIS.

**Existing mechanism.** `Verifier::make_dev_exec` (verify.cpp:3914-3951) instantiates a twin with
`cudaGraphInstantiateWithFlags(&dev, graph, cudaGraphInstantiateFlagDeviceLaunch)` (:3934) and captures a one-node parent on a private stream (:3938-3946).
`window_dev_launch_kernel` is `(void) cudaGraphLaunch((cudaGraphExec_t) exec, cudaStreamGraphTailLaunch);` (verify_kernels.cu:1930).
Its only caller is `capture(T)` (:2756). Arrays `exec_dev_[9]`/`exec_devp_[9]` (verify.hpp:589-596), accessor `dev_exec(T)` (verify.hpp:603), used at :3045 and :3121-3124; `pl_launch` tests `exec_devp_[T]` directly (:4627).
`HostGraphLaunch` (host_graph_launch.hpp:15-62) is the fallback at :3126-3128; it spawns a thread only under `STRATA_VERIFY_ASYNC_LAUNCH=1` (:3116-3119).

## A. Drafter launch mechanics

**Thread:** the calling thread, inline. `mtp.draft()` (generate.cpp:17160) calls `MtpDrafter::draft` (mtp.cpp:1649) and returns only after the last step's output is on the host.
No helper thread, no drafter thread; every graph is a plain `cudaGraphLaunch(.., cs_)` on the drafter's own stream.

**Caller to round-graph start** (mtp.cpp):
1. :1665 `OnDevice` device switch. No counter.
2. :1672, :1676-1677 `capture_round`/`capture_step`: cache hits (`if (exec) return true;` :1365, :1416). Capture, instantiate, upload and sync (:1331-1340) happen on first use only.
3. :1680-1697 mapped staging (`h_tok_`, `h_step_`, `h_pos_`, `h_row_`), every step's K/V cell included (:1692-1693); :1694 `h_out_[j] = -1`; :1698 seq_cst fence. Counter: `pre_ms`.
4. :1699 `stage_source_R`: one `cudaMemcpyAsync` D2D (:2038), skipped when `src_R_ == nullptr || src_R_ == window_R_` (:2035). Counter: `ms_src` (generate.cpp:16727-16729).
5. :1782 `cudaEventRecord` (diagnostic env only), then :1784 `cudaGraphLaunch(cp ? round_exec_c_[T] : round_exec_[T], cs_)`. Counter: `launch_ms`.

**Round to step 1, and step j to step j+1** (gated path, min_p > 0, :1778):
:1789 event, :1806 `cudaStreamQuery`, :1807 `prefetch_ple` (`ple_ms`), :1808 `progress_at` (no counter),
:1834/:1854 `wait_step`, a spin on `while (((volatile int32_t*) h_out_)[j] < 0)` (:1811) with `_mm_pause()` and `if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;` (:1815) (`wait0_ms`/`waitj_ms`; the query's own time is not split out),
:1836-1837/:1856-1857 read token and prob, gate (:1842), :1843 event,
:1845 `cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_)`, :1850 event, :1851 `cudaStreamQuery`, :1852 `prefetch_ple(drafts[j - 1])`, spin.
**After the last step:** :1864 `cudaStreamSynchronize(cs_)` (`tail_ms`); :1871-1874 `cudaEventElapsedTime` (diagnostic only).

**Not on the steady path:** `cudaMemcpy` D2H (tokens and probs arrive as mapped stores); `cudaEventQuery`/`cudaEventSynchronize` (chain path only, :2118-2145);
allocations (first use only); host sampling or argmax (:1304-1311 are graph nodes); thread wake-ups.
`prefetch_rows` is a mutex-guarded no-op in mmap mode on Linux (ngram.cpp:425-449) and issues an async ticket in Direct mode (:427-434).

**Measured per round** (draft-timing-lines.txt:3-8, ms): staging 0.000-0.001; launch calls 0.009-0.010; PLE prefetch 0.000; wait for draft 0 0.389 / 0.423 / 0.418;
round-graph event pair 0.310 / 0.316 / 0.324; per step 0.202-0.214; gaps 0.070-0.109; tail 0.004-0.005.

**Where the 0.39 ms goes.** It is `wait0_ms`, a host wall time: 0.01 launch calls + 0.31-0.32 round-graph occupancy (`eb[0]` to `ea[0]`, :1782/:1789)
+ ~0.08-0.10 residual (launch return to graph start, graph end to flag visible, polling). nsys puts the round graph's kernels at 0.10-0.14 ms, so 0.17-0.22 ms of its occupancy is not kernel execution.
Ranked by what the code shows:
1. **Plain serial edges inside the graph.** mtp.cpp never sets `pdl_scope()`; the window capture does, `pdl_window(G == 1 && pdl_supported())` (verify.cpp:1261).
   Each node waits for its predecessor to retire, and the shared-expert branch adds cross-stream event nodes (:1165, :1181, :1208). UNCONFIRMED.
2. **GPU shared with the async commit graph.** `ver.commit` (generate.cpp:17086) is launch-and-return on one GPU by default (generate.cpp:4503; verify.cpp:3603, :3619-3624:
   "the drafter (its own stream) reads only this window's final rows and its own K/V"). Nothing orders the drafter's stream behind it, and `wait_commit` is reached only after the loop (generate.cpp:17200).
   Only the round graph overlaps it, and only if drafter and verifier share a card. UNCONFIRMED.
3. The head node reads mapped host memory over PCIe (`copy_from_mapped_multi`, :1388-1391; elementwise.cu:406-416).
4. Launch call, 0.01 ms measured. Not it. 5. Kernels, 0.10-0.14 ms. Not launch path.
The step graphs overlap nothing and still read 0.20-0.21 ms each, so (1) is likely the larger term and (2) additive.

## B. Last step graph to window launch

In `draft()`: :1854 spin on `h_out_[j]` (waits: the step's `mtp_select_kernel` mapped store), :1862 `prefetch_ple`, :1864 `cudaStreamSynchronize(cs_)` (waits: drafter stream drain).
In generate.cpp: :17162-17166 counters, :17167 `pcie_auto_step`,
:17168-17172 `adapt_thr.join()` (waits: this window's adapt thread, :17081-17085, when `(rounds + 1) % o.adapt_every == 0`; measured 0.03),
:17181-17186 policy math, loop head :16742-16745 `T` re-walked from `dprob`, :16746-16778 controller and keep rule, :16785-16807 suffix and chain proposals, :16808-16818 window fill,
:16825-16842 `apply_pending(true)` (waits: `cudaEventSynchronize(adapt_ev)` :10760, then `src.commit_exchanges()` :10769 host memcpy, `res_upload()` :10783) and :16950 `adapt_tick(false)`, together `dt_adapt_wait`, measured 0.24 ms/window,
:16960-16964 `penalty_rows` plus a blocking `cudaMemcpy` H2D (penalized requests only), :16978 `ver.run`.
In `run()` (verify.cpp): :3087 `stage_inputs`, :3089-3099 PLE `prefetch_rows` per token, :3104/:3109 memsets,
:3112-3113 `adapt_gate_wait` (STRATA_ADAPT_OVERLAP flips only; spins on copies-landed, :2996-3010),
:3124 `cudaGraphLaunch(dev, cs_)` (one-node parent; F-452: 7.9 us host, 3.5 us to first kernel), :3136 `cudaStreamQuery`.
**After the launch, before the window's first real kernel:** :3175 PLE `gather_batch`, then :3178 `*flag = 1`, the flag the zero-doorbell graph waits on (:4668).
Acceptance (:16997-17026), commit (:17086) and emit/`fflush` (:17133-17141) all sit **before** `draft()`, so they are not on this path (F-452: commit/emit 0.01 ms).
Measured whole chain, flag on (nsys-anatomy.md:13): median 31 us, p10 21, mean 614, p90 2165. The typical path is already short; the mean is the adapt round's `commit_exchanges` (F-452: 65 of 117 slow-gap samples under it).

## C. Dependencies between step graphs

**No host data dependency.** Token path, all on device: step j runs `argmax_rows`/`sample_tokens` (:1304-1311), `row_top_prob` (:1312-1313), `map_ids` (:1314) into `out_ids_`,
then `mtp_select` (:1436) whose kernel does `*tok_dst = tok;` (verify_kernels.cu:772, the device buffer `tok_`) and copies the residual row into `Rin_` (:768);
step j+1's embedding reads it back with `ne->gather_dev(tok_, T, emb_, cs)` (mtp.cpp:970).
Step j+1's K/V cell and position were staged before the round (:1692-1693) and are read by that step's own first node (:1426-1428). `cudaGraphLaunch(step_exec_[j], cs_)` (:1845) takes no arguments.
The readback to the host is a zero-copy mapped store, not a copy: `((volatile float*) out_p)[j] = probs[row];`, `__threadfence_system();`, `((volatile int32_t*) out)[j] = tok;` (verify_kernels.cu:773-775), polled at mtp.cpp:1811.
It feeds host-only consumers: the gate, the output arrays (:1856-1858), and `prefetch_ple(drafts[j - 1])` (:1852; a hint. The window's `gather_batch`, verify.cpp:3175, reads for itself).
So the host holds **control** (whether step j+1 launches), not **data** (what it computes).
- (i) Back-to-back on one stream: valid and same bits (mtp.cpp:1790-1795; F-229 M3b 75.5 vs 75.4), and **measured negative** (F-229, -2.73% M1).
  At ~1.7-1.9 steps kept of 6 it executes ~4 wasted steps x 0.21-0.27 ms, about 0.85-1.0 ms/window, to save 0.045-0.11 ms of gaps.
- (ii) One graph with conditional nodes: needs the gate on device (`probs_` is device memory before `mtp_select`); the machinery exists (verify.cpp:2568-2624, CUDA >= 12.4).
  Ceiling is the gaps alone. Conditional nodes cannot be device-launched (:2579-2580), so (ii) excludes (iii).
- (iii) Tail-launched like the window: the `make_dev_exec` pattern per `round_exec_[T]` and `step_exec_[j]`. It removes only the launch calls (0.01 ms/round);
  the gaps are the host's gate wait (`waitj_ms`), not launch latency.

## D. F-206 `draft_launch` / `draft_finish`

Call sites: batch lanes only, the cohort tail (generate.cpp:12905, finishes :12924-12930) and the serial batch (:13171, finishes :13192-13198).
Per active slot: a `cudaMemcpyAsync` D2D of the slot's residual rows onto the drafter's stream (:12901-12904), then `draft_launch` (mtp.cpp:1897-1974):
the same staging as `draft()` (:1915-1934), the round launch (:1937), **every** step graph at once (:1949-1954), `cudaStreamQuery` (:1957), PLE prefetch of `tokens[a]` (:1962-1968),
state into `pend_*`, `draft_pending_ = true` (:1969-1971), return.
After all slots are launched, `draft_finish` per slot (:1976-2032): spin on `h_out_[0]` (:1983-1988), `cudaStreamSynchronize` (:1990), readout,
trim at `if (((volatile float*) h_prob_)[n - 1] < pend_min_p_) break;` (:2013). The slots overlap because each drafter owns a stream.
**Solo lane:** there is no call site; generate.cpp:17160 and :18579 call `mtp.draft` unconditionally.
The precondition is the refusal at mtp.cpp:1907: `if (max_steps < 1 || min_p <= 0.0f) { err = "mtp: draft_launch covers the sampled shape only (min_p > 0, a draft cell must exist)"; return false; }` (`draft()` serves that shape at :1739).
Two reasons not to wire it: it is shape (i), negative above; and there is no host work to hide behind (commit/emit 0.01 ms).
It is also not value-identical under `STRATA_SPEC_ENTABORT`: `draft_finish` has no `abort_after` (`draft()` applies it at :1767, :1827, :1842).
(The comment at mtp.cpp:1894-1896 says `max_drafts_ <= 1`; the header at mtp.hpp:117 says "at any max_drafts_". The header matches the code.)

## E. The min_p walk

Two host decisions, both after a readback:
1. Launch gate, per step: `for (int j = 1; j < max_steps && pj >= min_p && !abort_after(j - 1, pj); ++j)` (mtp.cpp:1842), with `pj = ((volatile float*) h_prob_)[j];` (:1857). Yes, a readback between steps.
2. Window size, next loop pass: `while (T < S_mtp && dprob[(size_t) T - 1] >= (float) req_spec_min_p) ++T;` (generate.cpp:16744), from the `dprob` array `draft()` filled. This one decides what is verified; (1) only decides what is executed.
The prob is computed on device (:1312-1313). In the ungated shapes the stop is readout only (:1822-1832; :2011-2017).
`n_drafts` is not passed at :17160, so (2) is the binding rule in the solo lane.

## F. Batch lane's window graphs

**Host-launched batch sites.** No batch exec has a device twin (`make_dev_exec` is called only at verify.cpp:2756).
- :4048 `run_slot_rows`, window graph; caller generate.cpp:13039.
- :4253 `batch_go`, window graph; callers :12719/:12723/:12727 after `batch_stage_rows` (:12705-12713); served by `batch_poll`/`batch_serve` (:12766-12776, :13372, :15481).
- :4258 `batch_go(with_commit)` and :4190 `commit_slot_prefixes` (callers :12856, :13107), commit graphs. The solo commit graph (:3603) is host-launched too.

**Differences from the solo capture.** Same `record_window` (:1204) under `batch_rec_ = true` (:3702), which reshapes about 25 sites (:1225, :1227, :1232, :1239, :1266, ...).
No conditional wrapper (it lives in `capture(T)` :2568-2613 only). Capture mode identical (`cudaStreamCaptureModeThreadLocal` :3698 vs :2615).
Instantiate identical (`cudaGraphInstantiate(&ex, graph, 0)` :3882). One variant per layout where the solo keeps two (`exec_`/`exec_nr_`, :2559). PDL scope identical (:1261).
**Number.** One exec per (rows, S, hbase) in `std::map<std::vector<int>, cudaGraphExec_t> exec_bm_, commit_bm_;` (verify.hpp:397), two commit graphs per layout.
LRU-capped at 64 only with `--batch-mtp` (generate.cpp:9539, :9620, :9691, :9713), else unbounded (verify.hpp:400).
**Re-captured at runtime.** `stage_batch` calls `capture_batch` and both `capture_commit_batch` every window (:3979-3980). A miss is capture, instantiate, upload and `cudaStreamSynchronize(cs_)` (:3716-3720), on:
a new layout; LRU eviction (:3975-3977 to :3845-3875, sync :3856); VRAM eviction inside `instantiate_evicting` (:3884-3891).
**To route through `make_dev_exec`.**
(a) Call it in `capture_batch` between :3716 and :3717 into two layout-keyed maps; destroy them at :734-737, :3861-3863 and :3886.
(b) Upload twin and parent before the :3720 sync, dropping both on failure as :2767-2773 does.
(c) A `dev_exec_bm(key)` accessor for :4048 and :4253. (d) `decline` clears the stage-wide `dev_launch_` (:3931); a declined batch layout must not retire the solo parents.
(e) Cost: a second instantiation per layout at first use, on the serve path (~6 MiB per window size here, F-452; host graphs 20-30 MiB on an L40S, :3879).

## Lever candidates (ranked by ms saved per window)

1. **Commit-overlap test, zero code.** Term: up to the round graph's 0.17-0.22 ms. One boot with `STRATA_COMMIT_SYNC=1` (verify.cpp:3565), `STRATA_COMMIT_TIMING=1` (:3590), `STRATA_DRAFT_TIMING=1` (mtp.cpp:1626).
   If the round's event pair falls toward 0.12-0.14 and commit/emit rises by less, the mechanism is A.2 and the lever is ordering the round graph behind the commit's event, or shrinking the commit graph.
   Bits: yes (ordering only). Risk: none in the test. If nothing moves, the pool belongs to lever 2.
2. **PDL scope on the drafter's captures.** Term: the same 0.17-0.22 ms plus an unmeasured share of each step's 0.21 ms.
   Change: a `PdlScope` guard around `record_front`/`record_rest` in `capture_round` (mtp.cpp:1393-1405) and `capture_step` (:1434-1436); the guard type is local to verify.cpp:1257 and must move to pdl.hpp.
   Reach is partial: by file, `launch_pdl` sites against `<<<` launches are native_mmvq.cu 6 of 41, native_bf16.cu 7 of 14, fused_gr.cu 0 of 120, quantize_act.cu 0 of 11, sampler.cu 0 of 16.
   Bits: claimed for the window (verify.cpp:1255-1256), but pdl.hpp:61-65 carries a WAR list for NaNs. Treat as unproven: deterministic-config identity plus M3b.
   Risk: NaN families; the event nodes at :1165/:1208 cannot take programmatic edges (:2724).
   Ceiling about 1.5-1.9% plus the steps' share of an 11.6-12.2 ms window: under the +4% bar alone, so compose it (law 15).
   Falsifier, zero boots: in the existing node trace, span minus summed kernel time for one round and one step graph. About 0 for the steps kills it.
3. **`wait_step`'s liveness query.** Term: ~0.08-0.10 ms (wait for draft 0 minus the round event pair; by difference). Change: time-base the `cudaStreamQuery` at mtp.cpp:1815 (also :1762, :1987) instead of every 1024 pauses.
   Bits: yes, host only. Risk: later dead-graph detection, bounded by the period.
   Falsifier, zero boots: sum the `cudaStreamQuery` API durations on the main thread between the round's launch and step 1's launch in the existing trace, and read last round kernel end to step-1 launch start.
4. **Adapt round off the serve thread.** Term: `dt_adapt_wait` 0.24 ms mean/window; boundary mean 0.61 against median 0.031.
   Change: move `commit_exchanges()`'s memcpy (generate.cpp:10769) off the serve thread under the `complement_ready`/`soft_commit` protocol (:16832).
   Bits: **not guaranteed**. Residency timing changes GPU/CPU rounding (#463, :16822-16823); M3b required. Risk: eviction and refill consistency.
   Falsifier, one env boot: `STRATA_DECODE_TIMING=1` is already on the record at 0.24. What is open is the median (nsys says typical is about 0), so price it as a latency-tail lever before building.
5. **On-device min_p gate (ii) plus tail launch (iii).** Term: gaps 0.045-0.11 plus launch 0.01. Under 1%; priced out at +0.4% in F-452 correction 2.
6. **Ungated pre-launch (i), or wiring `draft_launch` into the solo lane.** Negative (F-229), now about 5x worse per wasted step. Closed; re-open clause is a deeper mean chain or a step graph near 50 us.
