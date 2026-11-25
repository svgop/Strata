# F-463 — the critical path traced at source level (Paragraphs 27-30 of the directive), reconciled against the F-452..F-461 floor

**Provenance.** Six parallel read-only traces over head `4d3e375ac` (= origin/main `765dd5099` + F-462), 2026-11-25:
router/dispatch, CPU pool, GPU kernels (+ a ptxas -v recompile of the hot TUs from build.ninja flags), verify/MTP,
prefill/KV/PLE, serving + the Python layer.  Log evidence: `logs/custom-iq3_s.log` (793,187 lines; boot 2026-11-22
15:59; engine 0.1.46, i.e. PRE-F-452 - its wall numbers predate the adopted device launch; code citations are current).
Tags: [M] measured, [I] inferred from source, [F-4xx] closed/bounded by that journal entry.

## 0. Corrections to earlier entries (mine)
1. There IS a Python server: `serve/server.py` (9,329 lines, ThreadingHTTPServer) spawns `engine/strata --serve`
   over stdin/stdout pipes (:1180) and pulls one `BT <slot> <id>` line per token.  F-462's "in-engine C++ serve,
   no Python" came from a `find | head -6` that truncated alphabetically before `serve/`.  Same class as F-441.
2. `STRATA_ADAPT_OVERLAP` is not a lever here: the launcher removed it (F-187), F-185 measured the combo -5..-8%
   at 6-conc, and the engine refuses it with DMA_BATCH.  The +5.5..8.3% in the standing line was A4000/P100-class.
3. Prefill on THIS box is not flat: six-point fit 619..30,320 tokens -> **~420 ms + 0.155 ms/token** (3.9K -> 5.9K
   tok/s) [M, log].  The flat curve was the external laptop rig.
4. Solo decode is MTP-amortized: 2.04-3.11 tokens per 11.4-13.5 ms window [M, log:481956-481977] (F-461: 235-248 t/s).
5. Per-request `answering:` rate: p50 **58.5** / p90 188 / max 238.6 tok/s over 4,996 requests [M] - diluted by
   other slots' prompt reads; the 156-205 tail lines quoted earlier were the upper tail.
6. `strata batch:` "ms/window" is per cohort-half (bt_windows counts 2 per round, :12985): round ~= 24.5 ms,
   ~9.8 rows; `GPU-reach wait` is only added on the standing path (verify.cpp:4137), never in `batch_poll`.
7. The "IQ3_S" pack is mixed per layer [M, native_experts.txt]: gate/up IQ2_S x20, IQ3_XXS x17, IQ3_S x10,
   IQ2_XS/IQ4_XS x1 (traces differ on one layer); down IQ4_NL x39, Q2_0 x9.  Avg blob 2.05 MB; gate/up:down
   bytes 1.45:1.  The quality ceiling of this model is set by 2-bit gate/up in 20 layers.

## 1. Critical-path map (file:line)

### 1.1 Byte model, one decode layer-window (k = 10 [M, GGUF expert_used_count -> geometry.hpp:54])
Entries = n x 10.  Shares of entries [M, log:793797..795026]: VRAM 96.2-97.2 %, PCIe 0.4-0.7 %, CPU 2.5-3.3 %
(~1.0-1.8 CPU experts per layer-window; aggregate 1.79).  Duplicates across a window's rows are coalesced
(`plan_build`, expert_source.cpp:4856-4990: one group per distinct (layer, expert), M entry rows).
GDDR7 per window ~= 48 x ~40 distinct hits x 2.05 MB ~= 4 GB -> >= 2.2 ms at 1.79 TB/s [I].  Roofline minimum
is far below the measured window; the lanes are bound by structure, not by a bus (F-461).

### 1.2 The per-layer protocol (decode, graph-captured; 13 nodes/layer [I, verify.cpp:2049-2362])
router GEMV (bf16_gemv_fp32_mmvf_multi, native_bf16.cu:340) -> `route_multi` softmax+top-10
(native_router.cu:100-153) -> `route_resident_k` (ranks 6-9 swapped to resident within 0.75 logits; output-
changing, F-200; :265-318) -> `resident_plan_kernel` (device plan ONLY when every entry is resident, else the
host plan: verify_kernels.cu:1520-1625) -> `doorbell_publish_res_kernel` (GPU writes mapped pinned ids/x +
RMW `m_seq_` over PCIe: elementwise.cu:329-351) -> HOST: spin `*seq<want` (verify.cpp:4380 / 3192), plan, dead
memset of hit rows (expert_source.cpp:5127-5128, see 2.A2), pool(), `*flagA/flag_=want` (sfence; :3488, :4422)
-> GPU: `wait_flag_ge_planned` (spin kernels on mapped host words: verify_kernels.cu:1700-1712) -> grouped
hits (`native_expert_grouped`: ONE launch for gu+SiLU+q8_1, ONE for down: iq_kernels.cu:4822-5039) ->
`dm_wait_combine` (one launch: weights+shared+hits loaded before the spin on the CPU flag, CPU rows read
zero-copy from mapped memory: verify_kernels.cu:1753-1815).  ~60 % of layer-windows carry >= 1 miss and pay the
GPU->host->GPU round trip [I; F-222 ceiling 0.26 ms/window].  No cudaStreamSynchronize per layer on the cohort
lane; one cudaGraphLaunch + cudaStreamQuery per window.  No hot-path allocations (all first-use/boot pools).

### 1.3 CPU pool (pool.cpp; native_expert.cpp)
16 threads (15 workers cpus 1-15 + host on 0; SMT siblings N/N+16 never used).  Two phases (gate/up, down) x
(park-wait, done-wait) = 4 barriers per layer-window; host-serial `native_quant_h` between phases (:1120-1123).
48 tasks/phase ~= 13 rows each (:1066).  **nt ~= 1.12** [M: 96.3 entries / 85.8 experts] -> ~88 % of CPU experts
are single-token -> gate/up and down run ggml 256-bit dots with ZERO software prefetch (objdump: zmm=0,
prefetcht0=0; native_expert.cpp:272-283, 330-337).  The 512-bit `iq512_gu_rows` (zmm=256) runs only at nt>=2.
UMWAIT code is MSVC-only; `STRATA_POOL_UMWAIT*` are no-ops on this build.  `wstate_` is an unpadded atomic array
stored on every claim (:763, :948).  Aux threads (lookahead, esrc, adapt, CUDA driver) are unpinned: aux_cpus
never turned ON in the log -> they float on cpus 16-31 = the workers' SMT siblings [I; F-454 priced siblings].
Bounded by F-458: DRAM-read-bound at the engine's call size (37-45.6 GB/s vs ceiling 48-57).

### 1.4 GPU hit kernels (ptxas -v recompile, sm_120, flags from build.ninja) [M]
| kernel | layers | regs | smem | spills st/ld B | occupancy |
|---|---|---|---|---|---|
| native_gu_swiglu_kernel<22> (IQ2_S) | 20 | 64 | 512 | **104/148** | 2 CTA/SM, spilling at the 64-reg cliff |
| native_gu_swiglu_kernel<18> (IQ3_XXS) | 17 | 64 | 512 | 8/8 | 2 CTA/SM |
| native_gu_swiglu_kernel<21> (IQ3_S) | 10 | 87 | 512 | 0 | **1 CTA/SM = 25 %** |
| native_gu_swiglu_kernel<23> (IQ4_XS) | 1 | 92 | 512 | 0 | 1 CTA/SM |
| native_down_multi_kernel<20|42,true> | 39|9 | 48 | 2880 | 0 | 5 CTA/SM by regs |
| gr_down_staged_kernel (hc read) | all | 64-90 | dyn | 0 | **41 CTAs on 170 SMs (24 %)** |
| gr_up_fast_kernel (hc) | all | 128 | 12304 | 0 | 2 CTA/SM |
`__launch_bounds__(512)` with no minBlocks (iq_kernels.cu:2606) lets ptxas pick a different trade per type.
Grid = 20 x ng CTAs (gu) / 80 x ng (down): M adds no CTAs, only distinct experts do; at n<=3 the grid is short
and latency-bound.  Dequant is register-fused (Split<21>::load: two 16-bit loads + 8 dependent `iq3s_grid`
gathers per 32 values, iq_kernels.cu:999-1034; grid table is global memory, not __constant__).  L2 set-aside
covers ONLY the GDN recurrent state span (device.cu:383-414); the 18.78 GiB expert arena is in no policy window
(single-use weight reads are not marked evict_first).  DENSE_TC default off (output-changing; T=1 0.72-0.9x).

### 1.5 Prefill chunk loop (Prefill::run_impl, prefill.cpp:2562)
Per 32K chunk: `stream_all` queues EVERY non-resident expert of every layer in id order, **routed or not**
(:2905-2950; 14,676 never-resident + the 5189 lent) = 28.6-38.7 GB per chunk through a 96-slot issuer ring
(:3004, ~190 MB run-ahead) into 32-expert fused batches that each wait on their own copies (:3646-3649).  The
fused kernel decodes int8 weights to smem PER 64-row tile (moe_fused_iq.cu:479-493) -> dequant cost grows with
T x K/64 (no amortization).  GDN recurrence is sequential per token, 5 __syncthreads each (kernels.cu:588-640);
a chunked scan exists only in the sm_70 build (:429-436).  PLE gather is one std::async thread under a
recursive_mutex with per-row (90 B) reads (ngram.cpp:539, 611-615); the layer-1 wait is ~0.4 s of a 32K prompt.
The CPU pool does NOT share this path (`cpu_maybe` needs !stream_all, :3683).  Lend/refill: 5189 of 9900 slots
(52 %) are lent to the ring for the read and refilled after (~180 ms, generate.cpp:14953).  Fit: 420 ms + 0.155
ms/token [M].  Checkpoint restore/park: 2 cudaDeviceSynchronize + 38 (+60 per stage) synchronized memcpys
(conversation_state.cpp:196/211; conversation_copy.hpp:54-63); 12.5K-token park 55.8 ms, slot copy 44.5 ms [M].

### 1.6 Serve iteration and the Python layer
C++: `for(;;)` batch_on -> try_next_line else batch_step (generate.cpp:13714); cohorts iff n_active>3
(:12613); per round: stage, 2 graph launches, a busy poll that serves both verifiers' doorbells on cpu0 (:12792-
12833), sample sync per cohort (sampled), commit, `printf("BT")` per token + fflush (:12936, :12990), 2
unbuffered `cohorts:` writes per round (:12729, :12835), a std::thread for adapt every 4th round whose copies
are waited synchronously (`apply_pending(true)`, :10781; 0.75-1.0 ms/window [M]).  **Prompt reads run
synchronously on this single window thread** (:15758-15811): a 32K piece = 4.4-6 s during which every slot
stalls; between pieces slots get 0.5 x piece_ms of decode (:15314-15317).  [M] Log wall time: 11,567 s of prompt
reads vs 7,103 s of cohort rounds.  Python: concurrency reaches the engine (BGEN into 6 slots), but prompt
reads serialize on `ctl`, and a running solo GEN is STOPped and re-admitted at the next arrival (slot copy
mean 118.9 ms x 7,453 = 886 s).  Per token: Queue put/get + status_lock + detok + SSE, off the engine's cores.

### 1.7 Regimes (measured)
| regime | number | binding |
|---|---|---|
| solo M1 (F-461) | 235-248 t/s | the window graph's kernel set (97.6 % kernel time); ~2-3 % from its floor |
| concurrent 6 (F-461) | 289-297 t/s; round ~= 24.5 ms / 9.8 rows | pool DRAM reads (46 % of round) x miss rate |
| per request | p50 58.5 t/s | other slots' prompt reads (1.6) |
| prefill | 420 ms + 0.155 ms/tok | stream-all walk + per-tile dequant + sequential GDN (1.5) - NOT on the F-461 frontier |

## 2. Backlog
### A. New - not on the priced frontier (ranked by gain x certainty / cost)
| # | item | file:line | mechanism (lexicon) | gain basis | cx | fidelity |
|---|---|---|---|---|---|---|
| A1 | **Prefill/decode interleaving**: interim `--prefill auto:8192` when slots are active (lends fewer slots, 4x more decode turns); structural: read thread off the window thread | generate.cpp:15758-15811, 15314-15317, 15429-15462 (exists, refused with --pcie-mode dma :15454) | #58/#59 | 62/38 wall split; p50 58.5 vs 280-355 rows/s pure decode | S cfg / M | none-low |
| A2 | **Fix the IQ2_S gate/up spill + the 1-CTA/SM occupancy**: explicit `__launch_bounds__(512, 2)` per type | iq_kernels.cu:2606 | #6/#7/#8 | ptxas [M]: 104/148 B/thread spills in 20 layers, 25 % occupancy in 11; moves the SOLO floor (the kernel set IS the binding resource) | S | none (register alloc only) |
| A3 | **Routed-only streaming** for small/mid chunks | prefill.cpp:2916-2925, 3634-3637 | #37 | at 2K tokens ~56 % of experts unrouted -> ~45 % fewer streamed bytes of the fixed 420 ms | S-M | none |
| A4 | **Issuer a full layer ahead** (>= ~414 slots or a 2nd ring) | prefill.cpp:3004, 3625 | #12/#14/#57 | copy-bound fixed cost at low T | M | none |
| A5 | **Dead host memset of VRAM-hit rows** (never read: copy_rows_or_zero zeroes listed rows, dm_wait_combine reads only unlisted) | expert_source.cpp:5301 (`!device_plan` term), :5127-5128 | #63/#65 | ~16 MiB/window of pinned writes on the host between doorbell and flag [I 0.3-0.7 ms/window] | S | none - Golden + LOGPOS A/B gate |
| A6 | **Adapt wait in cohorts**: hide the every-4th-round copy wait (STRATA_ADAPT_NOWAIT is unset; overlap cannot arm at n_active>3) | generate.cpp:10781, 12801-12803, 13039 | #32/#63 | 0.75-1.0 ms/window [M] ~= 3-4 % of a round | M | none (staleness -> miss rate; battery) |
| A7 | **Dequant-once + tall grouped GEMM** in the prefill fused kernel | moe_fused_iq.cu:479-493, 1524-1528 | #24/#30 | dequant share of the 0.155 ms/token marginal (U1) | L | low (order) |
| A8 | **Chunked GDN scan for sm_120** (bit-exact v100 variants exist) | kernels.cu:588-640 vs :429-436 | #48 | recurrence share of the marginal [I 5-10 %] | M-L | low |
| A9 | **PLE gather on a pool + whole-chunk WILLNEED** | ngram.cpp:539, 589, 611-615 | #49 | layer-1 wait ~0.4 s per 32K | S | none |
| A10 | **Checkpoint/park packing**: one packed image + one async copy per stage | conversation_copy.hpp:54-63 | #51/#63 | 20-40 ms per park/restore (park 55.8, copy 44.5 ms [M]) | S-M | none |
| A11 | **Telemetry truthing**: time waits in `batch_poll`; fix the per-half ÷2 and the A-only pool | verify.cpp:4397-4426; generate.cpp:12985, 13030-13035 | #65 | none direct; every Amdahl on the cohort lane is blind today | S | none |
| A12 | **hc down grid 41 CTAs** -> split D over blockIdx.y keeping the per-(row,token) add order | fused_gr.cu:45, 1860 | #4/#5 | 6.5 MB bf16/layer at 24 % of SMs | M | low |
| A13 | **Evict-first on expert-arena loads** (`ld.global.L2::evict_first` or a per-layer policy window) | iq_kernels.cu:2626-2627; device.cu:398-404 | #18 | protects dense/activation L2 from ~80 MB/layer of single-use reads; ncu L2 hit rate decides | M | none |
| A14 | **VRAM**: +~600 slots (+6 %) from the idle 1271 MiB; 4 elastic arenas (3.9 GiB) come out of the cache tail at 6-conc (effective ~7.9K slots) | generate.cpp:6598-6618, 12144-12148 | #32 | hit rate = F-461 variable 1 | S cfg | none |
| A15 | **Pin aux threads off the workers' SMT siblings** (`--aux-cpus`) | generate.cpp:15859; pool.cpp:497-585 | #42/#43 | lookahead/adapt/driver threads float on cpus 16-31 [I] | S | none |
| A16 | **Solo lane**: STRATA_DRAFT_UNGATED=1 (bit-identical per mtp.cpp:1791-1794) - verify against F-452's drafter (0.57 ms/step kernel-bound) before pricing | mtp.cpp:1790-1800 | #61/#63 | draft 0.74-1.16 ms/window [M] | S env | none (canary) |
### B. Closed or bounded by F-452..F-461 (do not re-open without their clauses)
- CPU kernel speed / 512-bit single-token dequant / STRATA_IQ_GATHER: F-458 (DRAM-bound; 25 % faster kernel
  <= 4 % of pool).  L3 prefetch: F-460/F-461 (predictor precision 0.112; re-open clause there).  Both-cohorts
  double compute: F-459 (+0.86 %).  SMT siblings: F-454.  PCIe share rounding: F-455/F-456.  gdn_ab fold: F-457.
  Launch mechanics: F-452 (adopted).  PLAN_PARTIAL: F-222 (neutral).  Pool spin threshold: F-346.
- Pool granularity / per-expert pipelining / wstate_ padding (pool.cpp:1066, 1117-1130, 763/948): bounded by
  F-458 to the 10-20 % gap between the pool's 37-45.6 GB/s and the 48-57 ceiling; price only after A6/A11.
### C. Live quality-for-speed knobs (standing line Tier C - battery items, operator sign-off)
- `STRATA_ROUTE_TAIL_SKIP=7` (launcher:35, T63): a missed expert routed at rank >= 7 by every row contributes
  ZERO with no renormalisation (expert_source.cpp:4878-4905, 5115-5118); ~2 skipped entries/window [M].  On a
  32 GB rig at 96 % hits its perf value is ~nil: battery 7 -> 0 on the paired 534-question test + conc lane.
  The campaign's determinism pin says 0; production runs 7 (contract conflict).
- `STRATA_ROUTE_RESIDENT=0.75` (launcher:164, F-200 adopted +8.5..14.9): output-changing routing.  Stays.
- Pack: IQ2_S gate/up in 20 layers - F-461 variable 5 in the quality direction (a LABELLED trade).
- `--expert-profile` loads the Oct-14 base (config:64-65) while batteries pin `expert-profile-gold.bin`
  (F-453 ff.); several operator-dated learned snapshots exist.  Operator question, not a finding.

## 3. Routing / dispatch plan (Mikado order, under the F-452 standing)
1. A11 telemetry truthing (nothing on the cohort lane is honestly attributable before it).
2. A2 + A5 + A16: bitwise-safe, S, on the solo kernel set and the host round trip; Golden fingerprints +
   LOGPOS A/B; ncu before/after on `native_gu_swiglu_kernel<22|18|21>`.
3. A1-interim (chunk-size battery with active slots) + A6 + A14 + A15: config/env battery on the conc lane.
4. A3 + A4 + A9 + A10: the prefill fixed cost (S/M, fidelity none) - the only lane with a double-digit ceiling
   left on this box, and the one the external TTFT gap points at.
5. A7 + A8 + A12 + A13: kernel work in the prefill marginal and the hc read - each needs its own ncu split first.
6. A1-structural (read thread off the window thread): the L item that unlocks the 62 % wall share; design
   after 3-4 land, with the verifier-golden harness (F-448) still the gate for anything inside verify.cpp.

## 4. Patch blueprints (top 5) and verification
**P1 (A2) launch bounds.** iq_kernels.cu:2606 `__global__ void __launch_bounds__(512)` ->
`__launch_bounds__(512, 2)` for the <22>/<18> instantiations (kills the spill by forbidding >64 regs - ptxas
must then re-allocate instead of spilling; if it spills anyway, split the 4-entry register arrays :2632-2655)
and a SEPARATE trial of (512, 2) for <21>/<23> (87/92 regs -> must come out <= 64 with 0 spills, else keep
(512, 1) and shrink the live `W` range in Split<21>::load :999-1025 instead).
  prove: recompile the TU with the build.ninja flags + `-Xptxas -v` (the F-463 method) -> table above must show
  0 spills and regs <= 64 for the types changed; `ncu --set full --kernel-name regex:native_gu_swiglu
  --launch-count 24 engine/strata generate ...` -> local memory traffic 0, achieved occupancy 2x; Golden
  fingerprints (dispatch 0xc90ded77662c664d / 0xf37183119d4d14fa / 0xec8d5c9cdddc564d, arena 0xf2da901296f91c2d)
  unchanged; `STRATA_LOGPOS=1 STRATA_LOGPOS_TOPK=20` A/B identical; M1 + conc battery vs champion 3865e27a.
**P2 (A5) dead memset.** expert_source.cpp:5301: drop `&& !device_plan` from `gpu_zeroes_hits` so the
device-plan route also skips the host memset of kind 0/1 rows (:5127-5128).  prove: `expert_dispatch_golden_test`
under STRATA_VERIFY_DEVICE_PLAN=1 + the serve smoke; LOGPOS A/B; the host per-layer term (`per-layer host`) drops.
  If the Golden moves, the memset is live on the device-plan route - the item closes with that as the reason.
**P3 (A3) routed-only streaming.** prefill.cpp:2915-2950: before `stream_all`, obtain per-expert routed counts
(router output is on device; one 2 KB D2H per layer issued right after `route`, consumed a layer later), and
skip experts with count 0.  Keep the id-order walk for the rest.  prove: prefill tok/s at 1K/2K/4K/8K prompts
(the 420 ms intercept shrinks by the unrouted share); KV/logits parity on a fixed prompt (unrouted weights are
never read: fidelity none by construction); `STRATA_PREFILL_TIMING=1` kPf marks before/after.
**P4 (A1-interim) chunk policy.** config: `--prefill auto:32768` -> battery `auto:8192` and `auto:16384` at
--parallel 6 (fewer lent slots: 11717-11720; decode turns 4x/2x more often).  prove: per-request p50/p90 t/s
(the `answering:` distribution, n >= 200 requests), TTFT at 4K/32K, cohort round time during reads (pool ms).
**P5 (A11) telemetry.** verify.cpp:4397-4426: accumulate `now - ring` per served doorbell into `ms_wait` in
`batch_poll` exactly as run_slot_rows does (:4137); generate.cpp:12985/13030-13035: print per-ROUND values.
  prove: `cohorts: tel` `wait` stops reading +0.00 under load; the sum matches nsys doorbell->doorbell gaps.
Commands (all with the engine the only GPU program; serial): `nsys profile -t cuda,osrt,nvtx --sample=process
--cpuctxsw=process -o /tmp/f463_%p --force-overwrite true engine/strata generate --pack ... -p "<fixed>" -n 256`;
`ncu --set full --kernel-name regex:"native_gu_swiglu|native_down_multi|gr_down_staged" --launch-count 48 ...`;
`perf stat -e cycles,instructions,dtlb_load_misses.miss_causes_miss_walk --per-thread -p <pool tids>`;
`STRATA_COHORT_ROUND_TELEMETRY=1 STRATA_PREFILL_TIMING=1` for one boot; LOGPOS/DBG_NAN for every kernel change.
