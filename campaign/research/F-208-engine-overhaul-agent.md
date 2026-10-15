# F-208 — hostile decode-path review: clean-slate deltas and which are reachable incrementally

Agent review of the decode window (6-conc standing recipe, 31.41 ms/window budget: GPU-reach 21.68,
QSA attention 4.40, hc reads 4.2, CPU-expert doorbell tail 4.90, emit ~1.87 pre-R33, adapt 0.18).
Read-only pass over: `src/core/verify.cpp`, `src/core/expert_source.cpp`, `src/kernels/cpu/pool.cpp`,
`src/program/generate.cpp`, `src/core/mtp.cpp`, `src/kernels/cuda/{verify_kernels,elementwise,
qsa_decode_attn,fused_gr,native_gdn,sampler,native_moe,iq_kernels}.cu`. No engine runs; every ms
number below is an estimate with the arithmetic shown, to be confirmed by STRATA_VERIFY_PROFILE
before any adoption. The already-queued items (T1.1 partial plan, posted-write inversion, row-fold,
batched draft-head GEMM, shape-keyed graph cache) are treated as landed and NOT re-derived.

---

## (A) Structural inefficiencies — mechanism, evidence, expected ms/window

### A1. The 5090 runs the OLDEST attention kernel variant in the tree (est. −1.5 to −2.5 ms of the 4.40 ms QSA line)

Evidence: `src/kernels/cuda/qsa_decode_attn.cu:291-293` — `STRATA_ATTN_PRE75_BUILT` is defined only for
`STRATA_EXPERIMENTAL_SM60` or HIP wave32; `pre75_attn()` (:580-608) returns `cc < 75`, false on
sm_120. The standing launcher sets neither `STRATA_ATTN_PRE75` nor `STRATA_ATTN_LANECELL`
(`launch-custom-iq3s-8081.sh`). So every QSA decode/verify call lands on the base
`attn_chunk_kernel` (:119-282), which has:

- a value loop that is a **dependent scalar load chain per cell** (:258-279): `v = load_v1(...)` for
  one cell, then 12 FMAs, then the next cell's load — the ~300-600 ns L2/DRAM latency of each V row
  read is exposed serially (no 4-cell grouped loads; those exist only in `attn_chunk_kernel_pre75`
  :505-518);
- per-thread **shared-memory broadcast reads in head-major layout** (`sp[h][c]`, :278) — 12 LDS reads
  per element instead of the pre75 kernel's 3 `float4` weight loads per cell (`fold`, :492-501);
- no NC-batching of K loads in the score phase (:227-239); pre75 stages 2-4 cells' K per lane
  (:429-431).

Arithmetic for why this is latency-bound, not bandwidth-bound: cap ≈ `min(n_kv, 2051)` cells
(verify.cpp:619, qsa.hpp:127-134). Per (row, QSA layer): K ≈ 2051×2 kvh×256×1 B ≈ 1.0 MB (q8) +
V ≈ 0.5 MB (q4) ≈ 1.5 MB. A 6-conc window (≤8 rows) × 12 QSA layers ≈ 108-147 MB → **~80-110 us at
achievable HBM bandwidth**. The measured line is 4.40 ms — ~40x the bandwidth floor. The kernel is
sitting on memory latency and LDS throughput, not on bytes.

Fix class: the pre75 kernel is documented bit-for-bit ("same operands, same order", :284-290) and
already in the file. Building it for sm_120 (or porting `fold`+NC into the base kernel so one kernel
serves all archs) is order-preserving. Orthogonal to — and compounding with — the queued row-fold
(row-fold shares K/V bytes across the window's rows; this fixes the per-row ILP/shared traffic).

### A2. One host thread services all 48 doorbells; layer l+1's flags wait out layer l's CPU expert compute (est. −0.5 to −1.5 ms of the 4.90 ms tail, the part T1.1 does NOT remove)

Evidence: the batch path's service loop is `verify.cpp:3080-3130` (`run_slot_rows`) — a single
thread: spin on `*seq` → `pool(user, h_x_, h_ids_, ...)` → raise flags. `pool` is
`expert_pool_dispatch_multi` (expert_source.cpp:3694-4017), which does plan-fill + `P.publish`
(flagA, :3865 — correctly BEFORE the CPU compute) + activation quant (:3904-3916) + prefetch
(:3918-3926) + `pool->run_split_multi_native(...)` (:3983) and returns only when the misses are
computed (`wait_done`, pool.cpp:642-663) — with the servicing thread itself draining
(`host_works_`, pool.cpp:751, 881-891).

Mechanism: batch windows are one group (`verify.cpp:882` — `!batch_rec_` forces G=1), so there is no
intra-window overlap: while the pool computes layer l's misses (ms-scale on a miss layer), the GPU
has typically finished layer l's resident groups and is parked in layer l+1's
`wait_flag_ge_or(m_flagA_/m_flag_)` (verify.cpp:1637/1650/1664) — a `<<<1,1>>>` kernel polling
host-mapped flags over PCIe (verify_kernels.cu:1301-1305). T1.1 removes the plan rebuild and the
flagA wait for partial groups, but layer l+1's **flag_ (CPU rows) still cannot be raised until the
single service thread finishes layer l's drain**. The doorbell tail is therefore pipeline
serialization of the host, not only miss-compute time.

Fix class: split plan-fill from compute. A second pinned thread (or the plan-fill inlined into the
ring spin) publishes flagA(l+1) within tens of us of the ring; the miss compute becomes a queued
stage consumed by the worker pool; flags raised per layer as each drain completes. The plan slots
already exist per group (`set_plan_slot`, verify.cpp:3107/3576). Bit-identical: same plan bytes, same
expert arithmetic, only WHEN the host publishes them.

### A3. hc-read grids cannot feed a 170-SM GPU (est. −1.5 to −3.0 ms of the 4.2 ms hc line)

Evidence: `fused_gr.cu` — `gr_down_multi_kernel` grid = `DOWN_BLOCKS+1` = **41 blocks** of 256
threads (:38, :267-275); `gr_up_multi_kernel` = 80 blocks (:39, :326). Per layer the hc read streams
w_down 324×10240×2 B (6.6 MB) + w_up 10240×320×2 B (6.5 MB) ≈ 13.1 MB; × 48 layers = **629 MB per
window**. 629 MB / 4.2 ms = **150 GB/s achieved** — roughly 10% of the 5090's ~1.8 TB/s. The line is
grid-capped bandwidth: 41 blocks occupy ≤41 of 170 SMs; achieved HBM bandwidth scales with
outstanding loads, i.e. with resident blocks.

Why it is shaped this way: one warp per output row keeps each row's lane-serial dot order
(bit-identity), 8 warps per block → 41 blocks. The order-preserving way to widen the grid is
**1-warp-per-row blocks**: 324 blocks of 32 threads, each warp reading its `xn` tile straight from
global (xn is D×T = 10240×8×4 B ≈ 327 KB — L2-resident after the first rows touch it), dropping the
cooperative shared tile (:286-293). Per-lane chunk order is unchanged, one `warp_sum` per row
unchanged → bitwise identical. D-splitting across blocks is NOT order-preserving (changes the add
tree) — do not go there without gating. Same reshape applies to `gr_up` and to the final head read
(`fused_gr_read_multi`, used at verify.cpp:1764-1797).

### A4. The doorbell publish pays an x-copy over PCIe from a single block, plus a system-fenced RMW of the ring, per miss layer-group (est. −0.4 to −1.0 ms)

Evidence: `elementwise.cu:311-351` — `doorbell_publish_res_kernel` is `<<<1,1024>>>`: it copies
`n×N` f32 activations into mapped memory (8 rows × 2560 × 4 B = 80 KB of posted PCIe writes issued by
ONE SM), then `__threadfence_system()` (a sysmem flush), then thread 0 does
`seq = *(volatile seq) + 1` — a PCIe read RTT (1-2 us) plus a posted write. 48 layers × (RMW ~2-4 us
+ fence ~1-3 us + single-SM 80 KB copy ~20-40 us) ≈ **1.1-2.0 ms of serialized stream time before
the host even sees the ring** on miss layers. F-203's store-not-RMW is a known negative on CUDA —
but two deltas remain untouched:

1. **Publish the q8_1 activation image, not f32 x.** The CPU pool's first act on `h_x_` is
   `native_quant_act`/`act_quant_any` (expert_source.cpp:3909-3915). The GPU already owns the exact
   same quantizer (`native_quantize_q8_1`, used at verify.cpp:1447; the QDEDUP precedent at
   verify.cpp:1563-1566 documents byte-equality of the two quantizers). Quantizing on-device and
   publishing 20 KB instead of 80 KB cuts the copy 4x and removes the host-side serial quant from
   the service thread's critical path.
2. **Widen the copy** to a multi-block grid (or a graph memcpy node) so one SM is not serializing
   80 KB of posted writes.

### A5. CPU expert pipeline: global phase barriers with a serial host-only quantization between phases (est. −0.15 to −0.6 ms/window; the miss-layer tail component)

Evidence: `pool.cpp:817-846` — `run_split_multi_native` per layer-group: `run_phase(5)` =
`wait_parked` → publish epoch → drain → `wait_done` → **`wait_parked` again** (pool.cpp:746-756);
then the HOST ALONE quantizes every expert×token's ff rows (`native_quant_h`, :830-833); then
`run_phase(6)`. Two full park/wake barriers per layer-group (epoch `fetch_add` + workers spinning at
`_mm_pause`, 100 us park threshold armed by the launcher's `STRATA_POOL_SPIN_US=100`) plus a serial
quant pass that idles 15 workers while one thread runs AVX over nb×nt×640 floats.

Mechanism: gate/up of expert e+1 could overlap quant of expert e; down of a row range could start
when that range's quant lands. The protocol's park-before-publish requirement (pool.hpp's
lost-wakeup argument) forces the barriers. The GPU already solved this exact problem in-kernel
(`native_gu_fused_kernel`, iq_kernels.cu:3502-3506, "the q8_1 quantization in the epilogue…
bitwise the separate kernels' hq") — the CPU pool can do the same: make quant part of phase-5's task
epilogue (per expert, by the worker that finished its last row range), and drop the post-phase
`wait_parked` on the multi path by versioning the batch instead of re-parking.

### A6. Batch windows forfeit the group-split overlap machinery (est. −1.0 to −2.5 ms if it measures; the second half of the doorbell-tail story)

Evidence: `verify.cpp:882` — `const int G = (split_ && T >= 2 && !batch_rec_) ? 2 : 1;` — "a batch
window is one group". The 2-group interleave (`post(l,grp)` then `pre(l+1,grp)`, verify.cpp:1731-1735)
exists precisely to hide host service of group 0 under GPU work of group 1 — and it is compiled out
for the exact path the 6-conc metric runs. With G=1 every layer is: publish → host plan+compute →
flags → GPU continues. No layer's host service overlaps any GPU work except through the flagA
early-publish.

What G=2-for-batch would cost: the hc/router/shexp weights are read per group — but one layer's hc
weights (13.1 MB) fit in the ~36 MB of L2 left after the 60 MB set-aside, so group 1's re-reads
mostly hit L2; the MoE dedup degrades (the same expert routed in both halves is read twice —
partially L2-covered, blobs are ~1.3 MB each). What it buys: layer l's miss compute (up to the whole
4.90 ms tail on bad layers) runs under layer l's OTHER half's GPU work. Bit-identity risk is low but
must be proven per multi-row kernel (see D6).

### A7. L2 set-aside (60 MB of 96) starves the lines this review shrinks — re-tune after, not before

Evidence: launcher arms `STRATA_L2_SETASIDE_MIB=60` (B8, a measured win for the GDN state). GDN
state is 48 heads×64 KB×36 layers ≈ 108 MB total; the set-aside persists the hot portion and leaves
~36 MB of L2 for hc weights (629 MB streaming), QSA pools, and expert blobs. B8 was measured with
the CURRENT attention and hc kernels; after A1/A3 land, the marginal value of 60 MB to GDN vs. the
marginal cost to the (now smaller but still streaming) attention/hc lines changes. This is a re-tune
(env sweep), not a redesign.

### A8. `draft_finish` full-stream syncs after the pick is already visible (est. −0.2 to −0.5 ms of emit)

Evidence: `mtp.cpp:1812-1853` — the spin on `h_out_[0] >= 0` (mapped) breaks as soon as the token is
readable, but :1826 then calls `cudaStreamSynchronize(cs_)`, waiting out the graph tail (PLE
prefetch nodes, penalty staging) before returning. R33 already batched the waits per slot; the
remaining per-slot cost is the tail of each drafter graph. The pick+prob are already read from
mapped memory (:1830-1831). A tail event recorded before the PLE-prefetch nodes would bound the
wait to what the next consumer actually needs. (The PLE prefetch is advisory —
`ple.table->prefetch_rows` — and the host does its own at :1832-1837 anyway.)

### A9. CUDA arms neither STRATA_FUSE_WAIT nor STRATA_FUSE_HITS (est. −0.2 to −0.4 ms of graph-node handoffs; cheap A/B)

Evidence: `verify.cpp:247-268, 283` — the 143-amdk wait+copy fusion and the hits-in-combine fold are
gfx12 defaults; on CUDA they require `STRATA_FUSE_WAIT=1` / `STRATA_FUSE_HITS=1`, and the standing
launcher sets neither. Unfused, every served layer-group runs `wait_flag_ge` (1-thread poll kernel)
then a separate copy launch — a graph-node handoff (~2-3 us) × 3 waits × 48 layers. F-206 says node
COUNT buys nothing — but fuse_rows folds the poll into the copy's blocks, removing the handoff
latency on the critical path, which is a serialized-time reduction, not a launch-count reduction.
One-variable A/B, no code.

### A10. Smaller confirmed irritants (each <0.2 ms, listed for completeness)

- `expert_source.cpp:3750-3759` — the plan fill's `first_of` is O(n²) over ≤80 entries per
  layer-group plus a linear `distinct` search in the tail-skip loop (:3770-3774). Tens of us —
  fine today, but it sits on the flagA critical path (A2) where it will matter once the plan thread
  exists.
- pcie_mode 1/2 lets the grouped kernel read host expert blobs directly over PCIe
  (expert_source.hpp:337-340) — uncached sysmem reads from the same SMs running resident groups.
  The DMA-staging modes (0/2, `fetch` beside the CPU work, expert_source.cpp:3867) are strictly
  better whenever staging slots are free; with ROUTE_RESIDENT=0.75 the PCIe share is usually empty,
  so this only matters on cold/adapt windows.
- `sampler.coupled_penalize`/`coupled_merge` are `<<<1,32>>>` warp-serial per draft (sampler.cu:806,
  855-889) — correct and small; the draft-vocab subset keeps them under ~10 us each.
- `generate.cpp:10222` — `std::find` over `eos_ids` per accepted token per window; trivial unless
  eos_ids grows.

---

## (B) Incremental vs rewrite, per finding

| # | Finding | Incremental (fits graphs, bit-identical) | Rewrite-scope | Effort |
|---|---------|------------------------------------------|---------------|--------|
| A1 | Old attention variant on sm_120 | Build `attn_chunk_kernel_pre75` for sm_120 behind `STRATA_ATTN_PRE75=1` (or port `fold`+NC into the base kernel). Parity: the kernel is documented bit-for-bit; still run the M3b canary + a stream-hash A/B. | FA-style persistent attention with tensor-core dot and fixed accumulation order (gated: changes bits → MMLU x2). | **S** (build flag) / M (port) |
| A2 | Single service thread | Dedicated plan thread filling `m_plan_` + flagA as rings fire; miss compute moves to a queued stage the pool drains; flags per layer on completion. Graph-safe: waits read runtime values. | Producer/consumer ring: GPU publishes (ids, q8_1 act) → plan stage → compute stage → copy stage, all double-buffered one layer deep. | **M** |
| A3 | hc grid too small | New `gr_down_multi_wide` variant: 1 warp/block × 324 blocks, xn from global (L2), same lane chunk order; same for up/head. Bit-identical by construction. | bf16 HFMA2 / HMMA hc GEMV (10x ALU) — changes summation order → MMLU-gated, and unnecessary while bandwidth-capped. | **M** |
| A4 | Publish pays f32 x-copy + RMW | (i) quantize on device, publish q8_1 (byte-equal quantizers, QDEDUP precedent); (ii) multi-block copy grid. Keeps the RMW (F-203 law). | Posted-write batched publish — that IS queued item 2; not re-derived. | **M** |
| A5 | Pool phase barriers + serial quant | Quant into phase-5 task epilogue per expert; drop post-phase `wait_parked` on the multi path via batch versioning. Per-expert order unchanged. | Per-expert pipeline (gu→quant→down chained jobs), no global epochs at all. | **M** / L |
| A6 | Batch windows are G=1 | Arm group-split for batch windows behind an env (`STRATA_BATCH_GROUPS=2`); needs per-kernel multi-row split parity (see D6). | Window scheduler that treats host service as a stage with explicit credit, not flags. | **M** |
| A7 | L2 set-aside 60 MB | env re-tune AFTER A1/A3 land (single-variable legs). | Per-kernel `cudaAccessPolicyWindow` instead of a global set-aside. | **S** |
| A8 | draft_finish sync | Record a tail event before the PLE-prefetch nodes; wait that, not the stream. | Verify→accept→commit→draft as ONE captured supergraph (device-side accept). | **S** |
| A9 | FUSE_WAIT/FUSE_HITS unarmed | `STRATA_FUSE_WAIT=1` (+`_HITS=1` separately) in the launcher; one-variable legs. | — | **S** |
| A10 | misc | as noted per item | — | S |

---

## (C) The clean-slate design, and the 80/20 that is reachable now

### If the decode path were rebuilt for 5090 + 9950X3D + 184 GB today

**Weight placement.** VRAM (32 GB): all dense weights (hc 629 MB, router, shexp, head, QKV) + KV
pools + GDN state resident, plus the hot ~70% of experts by credit in the on-device cache — the
current split is already close to right. The clean-slate delta: the CPU-served expert tier stored as
**q8_1-quantized activations + IQ weights in huge pages on the V-Cache CCD**, with the pool pinned
there (pool.cpp pins one-thread-per-core but is CCD-agnostic; on the 9950X3D the V-Cache CCD's extra
L2 halves the DRAM reads of the IQ dequant tables and blob streaming, at ~10% lower clocks — for a
bandwidth-bound drain that is a net win) and host+service threads on the non-V-cache CCD at full
clocks.

**Host-GPU protocol.** One direction of waits only, in device memory. The GPU publishes work with a
single kernel that writes (ids, q8_1 activations, ring) into a pinned ring and rings a device-mem
flag via a batched memop write (queued item 2 completes this half). The host publishes results by
DMA: the pool's rows are written to pinned staging and a dedicated copy thread submits
`cudaMemcpyBatchAsync` + a trailing `cuStreamWriteValue32` on the copy stream — the graph waits on a
DEVICE word, never polls sysmem, and there are zero `<<<1,1>>>` poll kernels in the graph. The plan
fill is a device kernel everywhere (T1.1's resident_plan partial mode is the first half); the host
never builds a plan on the steady-state path.

**Attention.** One fused kernel per QSA layer for the whole window: rows folded (K/V bytes shared
across the window's rows — the queued row-fold), V loads 4-cells-deep (A1), probabilities in
cell-major float4 layout (A1's `fold`), split-K partials kept in L2, and the merge fused into the
last-arriving block via an atomic counter (removes the `attn_merge_kernel` global round-trip of
n_chunks×24×256 partials per row). INT8 tensor-core score path as an opt-in, MMLU-gated arm.

**Spec decode.** Verify→accept→commit→draft as one captured supergraph per (rows, S) shape: the
accept comparison (exact-match) is a device kernel writing `keep[]` to a mapped word; the host's
per-window role shrinks to staging the next tokens and reading picks. The drafter shares the window
graphs' streams/events instead of per-slot sync (R33 pointed here; the tail sync in A8 is the
leftover).

**Graph strategy.** Shape-keyed whole-window graphs (queued R31) + CUDA conditional nodes for the
partial-plan branches (replaces the skip_-value wait variants with graph-visible control flow) +
stream-ordered memops instead of poll kernels. No node-count micro-fusion (F-206 law).

**CPU experts.** A pipelined per-expert job graph (gu→quant→down chained, A5's rewrite) with the
pool as a work-stealing stage; plan thread decoupled (A2's rewrite); no global epochs.

### The 20% of that design delivering 80% of the gain, reachable WITHOUT the rewrite

1. **A1** — arm/port the pre75 attention shape on sm_120 (S/M). Biggest single GPU line.
2. **A3** — wide-grid hc read kernels, order-preserving (M). Second biggest GPU line.
3. **A2** — plan thread decoupling (M). The half of the 4.90 ms doorbell tail that T1.1 cannot touch.
4. **A4** — device-side q8_1 activation publish + wide copy (M). Removes up to ~1 ms of serialized
   pre-ring stream time on miss layers and the host's serial quant.
5. **A6** — batch group-split arm (M). Hides what remains of the tail under the other half's GPU work.
6. **A8/A9** — draft tail event + FUSE_WAIT arm (S each). Cents on the dollar.

Together with the queued items (T1.1 kills the s_bad fallback; posted-write inversion kills the
sysmem polls; row-fold shares attention bytes; R31 kills re-captures), the arithmetic target is:
4.40 → ~1.5 (A1+row-fold), 4.2 → ~1.5 (A3), 4.90 → ~0.5 (T1.1 + A2 + A4 + A6), emit ~1.0 (R33 +
A8). Window ~31.4 → ~21-24 ms ⇒ clears the +30%-vs-stock bar on 6-conc (standing +13% ⇒ needs
~24.2 ms) with margin. Every item above is individually Lock-Law-able (one variable, ABAB, MMLU when
touched).

---

## (D) Risk notes per item

- **A1** — bit-identity is the design constraint of the pre75 kernel (documented at
  qsa_decode_attn.cu:284-290, per-cell fold in ascending order). Still: run the stream-hash A/B and
  the M3b canary; 2x MMLU if the stream hash differs (it must not). Graph capture: launcher
  signature unchanged → same graphs.
- **A2** — no arithmetic change; risk is protocol (a plan published for the wrong layer-slot /
  overwritten while the GPU reads it). The plan slots are per group already; the new risk is the
  compute stage writing `h_ymiss_` rows for layer l while the GPU still reads layer l-1's — needs
  the existing ring-value discipline (flags are per-ring-value) and a ymiss double buffer or
  per-layer offsets. MMLU gate recommended once (stream-affecting class by mechanism, not by math).
- **A3** — per-row order preserved ⇒ bitwise; verify with `native_multi_parity`-style harness plus
  stream hash. Shared-tile removal changes NOTHING per-lane; the risk is a typo in the row↔warp map.
  Graphs: new launch dims are capture-time constants — re-capture required (shape-keyed cache
  handles it).
- **A4(i)** — the two quantizers are documented byte-equal (QDEDUP, verify.cpp:1563-1566) but the
  CPU's `native_quant_act` (expert_layout formats) vs GPU `native_quantize_q8_1` must be re-proven
  for the native IQ pack path specifically (the QDEDUP note covers the q8_1 image case). If bytes
  differ, tokens differ → MMLU x2 + stream hash mandatory.
- **A5** — quant epilogue must produce identical bytes at identical row granularity; the wait_parked
  removal changes the pool's lost-wakeup protocol assumptions (pool.hpp issue #29 notes) — the
  batch-versioning argument must be written down before the wait is dropped. Stall-abort (#29)
  coverage must still fire.
- **A6** — the real risk: multi-row kernels that change accumulation when the row set changes.
  `qsa_decode_attn_batch` is per-row independent (blockIdx.z per query); the grouped IQ GEMMs are
  per-entry independent (iq_kernels.cu:3437-3500); `bf16_gemv_fp32_mmvf_multi` and
  `native_moe_combine_multi` claim per-row order preservation, but `shared_expert_multi` with
  LFUSE/sg_ready and the `hits_in_combine` path are per-GROUP stateful (`sg_gated_[grp]`) — the
  group split must recompute that state per half. Stream hash + M3b canary mandatory; MMLU if any
  hash delta.
- **A7** — pure env re-tune; Lock Law single-variable; re-run only after A1/A3 (a re-tune before
  would optimize against the old kernel mix).
- **A8** — ordering risk only: the next window's drafter staging must not overtake the unread graph
  tail; the tail event must enclose everything the next stage reads (`window_R_`, penalty ring).
- **A9** — fuse variants exist and are HIP-proven; on CUDA they change graph topology (F-206
  territory) — the claim to test is handoff-latency, so measure with STRATA_VERIFY_PROFILE stamps,
  not node counts.
- **Known-negative guard respected** — nothing above re-proposes node-count fusion (F-206), fsync,
  DVFS, or DOORBELL_STORE-on-CUDA. A4 keeps the RMW; A9's justification is serialized handoff time,
  and its acceptance bar is the profile line, not a node count.

## Premortem (top failure modes of THIS plan)

1. A1/A3 estimates are bandwidth-floor arithmetic, not measurements — if the lines are actually
   LDS/ALU-bound elsewhere (e.g. the FWHT pair around QSA, or the indexer top-k), the wins shrink.
   Verify with STRATA_VERIFY_PROFILE stamps 11-14 (attention) and 10/32 (hc read) before committing
   anything downstream.
2. A2's protocol change breaks pipelined windows (the `fl_*` path shares the service) — the plan
   thread must serve BOTH `run_slot_rows` and `service()` call sites or the M1 lane regresses
   (single-stream protection law, F-183 precedent).
3. A6 interacts with the commit graph's per-slot prefix structure (`ckey(rows,S,hbase,parity)`) —
   group-split batch windows change the row layout the commit keys assume; audit `stage_batch`'s
   `rows[]` grouping (verify.cpp:3011-3019) first.
4. Hour drift (±8%) makes the small items (A8/A9) unmeasurable alone — bundle them only after the
   big lines land, never as a basket with A1/A3.
