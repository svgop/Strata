# T15A — the M1 198→250 critical path, re-audited on today's recipe: the fresh budget, every open item priced ENGAGES/DEAD, and the ranked road

Date: 2026-10-17 (post F-233; the five-boot final bench is the baseline). Scope: M1 single-stream
decode only. READ-ONLY: no server touched, nothing built or booted, no file outside this dossier and
the INDEX line written. Baseline (ground truth, all measured): launch-custom-iq3s-8081.sh standing
env (ATTN_DAILY, GDN_CHUNKED, FUSE_HITS, POOL_DIET, COHORTS=2, SPEC_DRAFT_TEMP=0.85, BATCH_MTP=1,
TYPICAL_ACCEPT=0.2, DEVICE_PLAN, ASYNC_LAUNCH, TYPICAL_DEVICE, ROUTE_RESIDENT 0.75, PLE_BATCH,
POOL_SPIN_US=100 …) + config (--spec 6, --spec-min-p 0.5, --kv int8, --kv-resident 32768, --pcie-frac
0.35 dma, --batch 6, --mtp-q4 all, draft_vocab en). Five final-bench boots (standing-restore-f233,
final-bench-r1..r4): **M1 185.7/192.2/211.3/205.5/197.6 = 198.5 mean; pair 197.6; conc 312.4;
acceptance 74.9%**. Method: line-level re-read of the solo loop (`generate.cpp` ~13101-13400 — the
true single-GPU loop; the 12807 loop is the multi-GPU pipeline, inert here), `Verifier::run` tail
(`verify.cpp` 2330-2560), the drafter (`mtp.cpp` 1620-1746), the hc kernels (`fused_gr.cu`), the
mmvq/IL dispatch (`native_mmvq.cu` 2588-2810), the pool/doorbell path — calibrated against the five
fresh boot logs, the Oct-14 diag stage tables (still the only per-stage GPU map banked), and the
F-203..F-233 journal. T13A's map is one recipe old; T14C's c_row is one adoption old; both are
re-derived here. Reconciled with T15C (the env inventory) — shared items are marked, and where this
audit's cell-level reads sharpen T15C's, the delta is stated.

---

## 1. Summary

1. **Today's window is 11.9 ms mean (10.9-13.1 steady rows) carrying ~2.47 tokens; the M1 probe mean
   is 198.5 because the probe's request mix (first windows, the 1500-token request, 16 ms tails) sits
   under the steady rate.** A T=1 window costs **8-9 ms warm** (the gen-1 rows of the same five boots
   — the W(1) intercept T14C lacked). The implied marginal row on TODAY'S recipe is **~2.2-2.6 ms
   (range 2.0-3.5)** — T14C's 3.56 was measured pre-POOL_DIET; the diet and FUSE_HITS adoptions
   landed after the s35 3v3 and lowered the row cost under the acceptance curve. This moves T14C's
   depth-crossing threshold from 0.72 to **~0.48-0.60 tokens/row — at or below the 0.35 arm's
   measured 0.604 blend**: the min-p re-sweep is no longer closed by F-228's arithmetic; the recipe
   changed under the verdict (§3 item 2, §6).
2. **The 250 arithmetic:** need C/W = 0.250 tok/µs vs today's 0.208. The kernel tranche (hc + dense +
   head + rec, §4) is worth **−1.8..−3.2 ms/window** on mechanism-read byte/issue floors — the probe
   mean lands **~235-270** from the tranche alone, so 250 is *reachable but not stable* without the
   acceptance half. The tranche simultaneously cuts c_row to ~1.8-2.4, which drops the depth threshold
   to ~0.42-0.55 and makes both the 0.35 re-sweep and the F-229 ungated re-boot positive WITHOUT the
   retrain — the kernel and acceptance questions are one question, as T14C said, with new constants.
   FastMTP (+5.7% C at fixed shape, T14C) plus the tranche plus the clock lock is the composition
   that makes 250 **stable**: 198.5 → ~240 (tranche) → ~255-265 (retrain) with boot variance folded
   flat by the lock.
3. **Every still-open item was code-read and priced** (§3): the head mmvq question resolved at table
   level (plain kernel runs on ~73% of windows; the fix is env-only, bitwise, instruments already
   built); FUSE_WAIT double-dead (F-206 negative AND both sites unreachable under standing
   DEVICE_PLAN=1); LFUSE double-dead (F-206 −3.0% AND ar_on() false at 2% misses); DOORBELL_STORE
   closed by F-203 (output-changing on CUDA); EcoSpec/tree filling dead by the c_row law (0.16
   tokens/row marginal vs 0.48+ needed); PDL half-landed (intrinsics in 3 kernels, zero coverage in
   the 3.6 ms hc+dense middle); PCIe-empty-skip real but ~0.1 ms (F-205 caps it).
4. **New levers found** (§4): the host-tail package (rescue-probe kernel+sync pairs between run() and
   commit, folded into the window graph; stage-in hoisted into the drafter's wait loop) worth
   −0.25..−0.6 ms/window bitwise; the hc line's true anatomy (serial 320-FMA accumulator chain,
   ~3.8x off its byte floor — the mechanism T12 called "ALU/cvt-bound" is a latency chain, which is
   fixable at three bit-class levels: TSUM bitwise → multi-accumulator sum-order → HC_Q8 different
   bits); and the IL tables having never been measured on this 5090 (the shared table is the
   worse-of-3060/5070 rule).

---

## 2. The ms/token budget as of today's recipe

### 2.1 Fresh measurements (the five final-bench boots, this audit)

| quantity | value | source |
|---|---|---|
| M1 probe mean | 198.5 (185.7-211.3) | standing-restore-f233, final-bench-r1..r4 |
| Steady window (220-tok rows 3-14) | **10.9-13.1 ms, mean ≈ 11.9** | same files, per-request rows |
| Tokens/window C | 2.3-2.5 (accept 69-84% on 149-190 offered) | rows [n] accounting (W = 220 − accepted) |
| **W(1)** — one-row window, warm | **8-9 ms** (7-13 settling) | rows [17]-[34] (gen-1 requests, hit 94-99%) |
| Implied marginal row (today) | (11.9−8.5)/1.47 ≈ **2.3 ms** (range 2.0-3.5) | W(1) anchor + steady regression |
| pair / conc | 197.6 / 312.4 | same boots (COHORTS engages only >3 slots) |
| GPU | RTX 5090 32 GB, SM clock 2827 MHz unlocked (target 3135) | nvidia-smi, this audit |

Reconciliation: t/s(steady) = 2470/11.9 ≈ 207; the probe mean 198.5 = steady × 0.96 (the mix: first
windows at 14-15 ms, request [15] 1502 tok at 13.2, [16] 482 tok at 16.1). Deltas in W apply to the
heavy rows too; the probe-mean transfer coefficient ~0.96 is used throughout.

### 2.2 Segment map (Oct-14 diag warm row = 11.39 ms GPU stages, adjusted by the two adopted arms)

The stage table predates FUSE_HITS/POOL_DIET; the adjustments are the measured arm deltas, not
guesses (F-226: M1 +4.35%; F-227: +4.06% — together ≈ −0.7..−1.0 ms of window-equivalent, part of
which was exposed pool, part combine work):

| # | Segment | ms/window (today, central) | basis | where |
|---|---------|---------------------------|-------|-------|
| 1 | hc mixer reads (gr_read ×2/layer: norm, down, up; GDN 2.10 + QSA 0.69) | **2.8** | unchanged by both adoptions; 24% of the window | fused_gr.cu (gr_down 124-139, gr_up 141-168); mtp.cpp:964/1084/1203/1214 share them |
| 2 | dense/attn lines (qkv, gate/z, ssm_out, QSA q/k/v/attention/out) | **3.5** | unchanged | verify.cpp mm lambda 1163-1175; per-row T-scaling per F-228 |
| 3 | VRAM expert hits (grouped matvec) | **1.75** | byte-floored (roofline, **known**) | verify.cpp grouped() 1671-1681 |
| 4 | head (248,320×2560 Q6_K) | **0.45 avg** | **plain kernel on ~73% of windows** (§3 item 1) | verify.cpp:1862-1868 |
| 5 | router+plan+bell (GEMV, top10, route_resident, resident_plan, doorbell_publish_res ×48) | 0.8 | DEVICE_PLAN skips waitA on fully-resident groups | verify.cpp 1544-1596, 1588 |
| 6 | shared expert + quant | 0.8 | unchanged | verify.cpp 1597-1635 |
| 7 | copy+combine | **0.4** | was 0.66; FUSE_HITS folded hit_add (−0.2..−0.3) | verify.cpp 1737-1761 |
| 8 | PCIe grp + waitB | 0.15 | 48 always-launches, counts[2] usually 0 | verify.cpp 1694-1708 |
| 9 | waitA + gaps + exposed pool | **0.2** | was 0.3-0.5; POOL_DIET took the repark waits | verify.cpp 2410-2468; pool.cpp run_phase |
| 10 | draft phase (gated step loop) | **1.0** (0.95-1.19) | host-latency-shaped; F-229 closed the ungated twin at chain 2.2 | mtp.cpp 1666-1746 |
| 11 | commit (async) + emit + rescue probes + adapt join | 0.4 | probes: 1-2 × (launch+sync+8B D2H) = 0.03-0.06; sampler second sync 0.01-0.03 (sampled traffic) | generate.cpp 13222-13246 (rescue), verify.cpp 3488-3498 (row_typical_stats), 2537 (sync #2) |
| | **Total** | **≈ 11.5** (measured 11.9 steady incl. heavier mix) | | |

What must shrink for 250 at fixed C: **−2.0 ms**. Where it can come from: (1) −2.8 hc, (2) −3.5
dense/attn, (4) −0.45 head, rec (inside 2) −0.25, (10) −1.0 draft, (11) −0.4 host tail. The hits line
(3) is byte-floored and does not move without routing/bytes changes.

---

## 3. The open items: ENGAGES-or-DEAD, priced

**F-232 law honored**: every verdict below is a code-read of what the segment spends, or a measured
arm, never an accounting identity. Prices: Δms/window → Δt/s via probe-transfer ≈ ×(0.96×2470)/(11.9²)
≈ **+20 t/s per −1.0 ms/window** at today's C (±10% over the boot band).

### 1. Head + dense mmvq at small T on sm_120 — ENGAGES, and cheaper than priced (env-only first)
- **Mechanism (cell-level read, this audit — sharpens T15C #2):** `native_mmvq_il_supported` admits
  T=1..4 and Q6_K (type 14), BUT the row count comes from the tables: sm_120 has **no T=1 entry at
  all** (`kIl1Arch` lists only cc 61/86 → returns 0), and the shared table's Q6_K row at T=2,
  n_out≥12288 is **0** — so the 248,320-row head runs the PLAIN `native_mmvq` at T=1, T=2 **and
  T≥5** (ncols>4 unsupported). Only T=3/T=4 windows get IL: **~73% of windows run the plain kernel**
  (T-mix from F-228's offer distribution). Dense Q4_K: plain at T=1 and T≥5, IL r=1 at T=2-4. The
  shared table is the **worse-of-RTX-3060/5070** rule (native_mmvq.cu:2588-2592) — **this 5090 has
  never voted on any cell**. The head line is 0.45 ms/window at 63-68% roofline on the plain kernel
  (T13A); the IL family's measured wins at these shapes are 3-30% per cell.
- **Fix, in cost order:** (a) `STRATA_Q6P_SELFTEST=1` one boot — the built-in bitwise memcmp +
  **paired timings at nc 1/2/4 on the real head shapes** (native_mmvq.cu:2064-2090) answers the
  microbench F-232 queued, free; (b) `STRATA_MMVQ_IL_ROWS="14:1:…;14:2:…"` ABAB — the no-rebuild
  env override turns on T=1/T=2 head IL (parser allows ncols 1-4, consulted BEFORE the kIl1Arch
  branch, native_mmvq.cu:2691-2704); (c) `mmvq_il_parity --bench --emit-table` on this card → paste
  `kIlRows_120` (one rebuild) for head+dense; (d) `STRATA_Q6_PACKED=1`+`STRATA_TSUM=1` (T15C #2,
  bitwise class, AMD defaults).
- **Expected:** head −0.05..−0.15 ms; dense T=1/T≥5 cells −0.1..−0.4 → **+3..+9 t/s**. Cost: XS (a+b)
  → S (c). Risk: LOW (bitwise-claimed family, parity harness exists); Q6_PACKED costs ~521 MiB VRAM
  (≈ −2.6% residency — watch the hit-rate line). Gate: canary only.
- **Falsifier:** the selftest's paired timings; if packed/IL ≤ plain on every nc for the head shape,
  the lane retires for one boot.

### 2. min-p 0.35 re-sweep on TODAY'S recipe — RE-OPENED BY THE BUDGET (was closed by F-228)
- **Mechanism:** F-228's −1.75% and T14C's c_row=3.56 were measured on boots WITHOUT POOL_DIET in the
  launcher (s35 3v3 ran 02:11-02:18; the diet was adopted from the 01:47-01:55 battery). Today's
  W(1)-anchored row cost is 2.0-3.5 (central ~2.3-2.6) → the crossing rule
  c_row·C₀/W₀ = 2.3×2.47/11.9 ≈ **0.48** (range 0.42-0.70) sits at/below the 0.35 arm's measured
  blend **0.604 tokens/row**. The acceptance marginals are bit-preserved by the diet (M3b −0.04pp
  class), so the blend carries over; the cost side moved. Also FUSE_HITS/POOL_DIET shrank the
  non-scaled window W₀, which further lowers the threshold.
- **Step:** one boot, `STRATA_SPEC_DEPTH=1` + 3× M1 probe at `--spec-min-p 0.35` on the standing
  launcher; read dp_ok/dp_tested and the row accounting (the same read T14C's G5 schedules — but it
  is no longer gated on the retrain landing first).
- **Expected:** 0..+4% M1 (at threshold-equality it is a wash; at c_row 2.3 it is ~+2.9%). Cost: XS.
  Risk: output-changing (min-p is a gate, not a bit-change to accepted rows — the F-230 tax does not
  apply; M3b drift gauge per T14C G1). **Falsifier:** the boot itself — if ΔC/Δoffered < 0.45 on the
  stair read, it closes again for free.

### 3. hc mixer line (2.8 ms/window) — ENGAGES; the anatomy is a latency chain, and there are three bit-class rungs
- **Mechanism (new, this audit):** `gr_down_kernel`'s dot loop (fused_gr.cu:128-131) runs ONE fp32
  accumulator per lane: `for (j = lane; j < 1280; j += 32) acc += dot8(...)` = **320 dependent FMAs
  per lane** (`#pragma unroll 4` cannot break the acc chain; each dot8 is itself an 8-FMA chain).
  At ~4-cycle FMA latency that is ~4x the issue floor; measured, the line runs at ~450 GB/s effective
  (26.3 MB/layer hc bytes × 48 layers ≈ 1.26 GB/window vs 2.8 ms) = **3.8x off the byte floor
  (~0.74 ms)**. `gr_up_kernel` is a 40-chunk gather, grid-starved (80 blocks), also latency-bound.
  This refines T12's "ALU/cvt-bound" (the unpack8 cost is real but secondary) and T15C's byte-only
  HC_Q8 premise (bytes halving helps, but the chain is the bigger half).
- **Rungs by bit-class:**
  1. `STRATA_TSUM=1` — bitwise butterfly warp-sums, wired into fused_gr's q8 launch path
     (fused_gr.cu:1969-1985) and the mmvq kernels; AMD ships it on by default. **Never armed here**
     (T15C #2). One env boot.
  2. **Multi-accumulator down** (4 independent acc chains, summed at the end) — same products,
     different FP sum order = the GDN_CHUNKED class (distribution-adjacent, ~1e-7): canary + 2×
     MMLU-Pro per the campaign law, small blast radius, ~15-line kernel change. F-217 law: a fresh
     nsys/`fused_gr_bench` on the line BEFORE surgery.
  3. `STRATA_HC_Q8=1` (T15C #3) — hc reads as Q8_0, halves bytes AND unpack work; different bits
     (RDNA4 precedent +2.5-4%); **check the prefill read path first** (F-230 tax).
- **Expected:** rung 1+2 together −0.7..−1.3 ms → **+14..+26 t/s**; rung 3 adds −0.3..−0.6 more.
  The drafter shares these kernels (mtp.cpp:964/1084/1203/1214) → c_row drops in the same motion
  (~0.3-0.5 ms/row), which feeds item 2's threshold and T14C's G4/G5.
- **Falsifier:** `fused_gr_bench`/`gr_parity` offline microbench of a 4-acc variant at the exact
  (LR=320, D=10240, bf16) shapes vs the stock kernel — no server boot needed; then one
  STRATA_VERIFY_PROFILE probe reading the hc0/hc1 classes.

### 4. FUSE_WAIT — DEAD (twice over)
- F-206 swept it: **−1.3% M1, default-off stays**. Mechanism read on top: both call sites
  (verify.cpp:1688 waitA+plan-copy fusion, 1718 wait+copy_rows fusion) live under `!device_plan_`,
  and **STRATA_VERIFY_DEVICE_PLAN=1 is standing** — the armed env never executes a fused form on
  this recipe (the −1.3% was round noise/structure elsewhere). Re-openable only as NEW fused variants
  of `wait_flag_ge_planned`/`copy_rows_or_zero_from_mapped` (kernels that do not exist yet); prize
  ≤0.1 ms (waitA+waitB warm total is 0.1-0.2 ms post-DIET). Not worth a battery.

### 5. LFUSE — DEAD (twice over)
- F-206 swept it: **−3.0% M1**. And `lfuse_on` requires `ar_on()` = all-resident (verify.cpp:928-930,
  verify.hpp:448) — false forever at 2-3% misses/window. T13B C5's "one-line arm" was written before
  the sweep; do not re-queue. The gated combine variant (`native_moe_combine_multi_hits_gated`,
  verify.cpp:1743) stays dead code at standing.

### 6. STRATA_DOORBELL_STORE — CLOSED (F-203), do not arm
- Measured **−21% M1 with M3b 63.1 vs controls' 75.2-76.9** — the store form desyncs the CUDA wait
  loops' ring semantics (they expect the increment counter, not absolute values). Output-changing =
  correctness-signature class. The remaining doorbell levers: (a) fuse `doorbell_publish_res` into
  the router epilogue — sub-F-205 value (node cut); (b) **q8_1 publish** (T13A #7, real byte cut,
  ~48-96 mapped writes/window → −0.1..−0.3 ms, M cost, needs the shared-expert switch T13A flagged).

### 7. PCIe-empty-node skip — ENGAGES but marginal
- verify.cpp:1700-1708: per layer-group, `wait_flag_ge_planned(m_flagB_)` + `fetch_blobs` (pcie_mode
  DMA staging) + the PCIe `grouped` launch run regardless of counts[2] (live share 0.3-0.5% of
  entries; most layers 0). Empty-grid blocks read a count and exit; cost ≈ launch+read ≈ 1-3 µs × 48
  ≈ **0.05-0.15 ms** — and F-205's law (node cuts inert at our spacing) caps even that. Counts-baked
  grids need capture-time knowledge the device plan does not provide. **+1..2 t/s ceiling.** LOW
  rank; bundle with any other verify.cpp surgery. `STRATA_PCIE_MIN1=1` (T15C #6) is the cheaper
  cousin worth one boot on the pair/conc lanes.

### 8. Sampler single-sync — ENGAGES (still unexecuted; T13A #8 = T15C #8)
- verify.cpp:2486 sync #1 + :2537 sync #2 on sampled/penalized requests (standing probe traffic is
  sampled). Skipping #1 when the sampler will sync (stream order guarantees the graph completed;
  one fence-ordering check on the next window's stage-in). **+1..2 t/s**, XS-S, bits identical.

### 9. PDL — HALF-LANDED; re-open only bundled with the hc/dense kernel work
- Since F-191 the infrastructure landed: `launch_pdl` with programmatic graph edges, `pdl_wait/
  pdl_trigger` intrinsics, restrict-handling (pdl.hpp; verify_kernels.cu:1568-1590). But coverage is
  **3 kernel files** (native_mmvq ×2, native_bf16, the stamp kernel) — **zero in fused_gr and
  native_moe**, i.e. not in the 3.6 ms hc+dense middle where overlap would pay. `STRATA_DF_PDL` is
  opt-in, never A/B'd on this rig. Price if completed there: −0.3..−0.8 ms (launch-tail overlap at
  ~5.6 µs/node), different-schedule same-bits → canary + stress class, HIGH blast radius. Sequence it
  AFTER the hc/dense kernels it would overlap, not before.

### 10. EcoSpec / tree filling (T7.B.4) — DEAD by the c_row law at these rows
- The +5.7% tree-topk evidence is from an engine whose verify row is ~17% of window (5.4/32 ms).
  Ours: 2.0-3.5 ms/row ≈ 20-30% of window. The measured tree shape (chain-3 → steps4/topk4/draft8,
  τ 3.413→4.231) is +0.8 tokens for ~+5 offered rows ≈ **0.16 tokens/row** — vs the ≥0.48-0.60
  needed today (and ≥0.42-0.55 post-tranche). Net ≈ **−25..−40% M1 on this engine**. The draft_policy
  machinery (src/spec/draft_policy.cpp) is EV-correct for lookup sizes and would veto it if it could
  see MTP shapes. Revisits only if c_row ever reaches ~1 ms (not on any priced path). The GDN
  slot-promotion design that would HOST a tree is likewise a depth/acceptance play, not a window-time
  one (rec is 0.23 ms — near floor after the ILP rewrite).

### 11. Clock lock — ENGAGES for the *stable* half (unchanged, still pending user sudo)
- Box at 2827 MHz unlocked; F-203 leg 4 pinned DVFS transitions as the collapsed-run class; the
  185.7-211.3 boot spread is the visible cost. `SM_CLOCK=3135` (T8.B) makes §2's budget boot-invariant.
  Zero mean risk; falsifier: `nvidia-smi dmon -s c` alongside one ABAB pair.

---

## 4. New levers found this audit (not in T13A/T14C/T15C)

### A. The host-tail package: fold the rescue probes, hoist the stage-in, single-sync the sampler (bits identical) — +5..11 t/s for S-M code
- **What the solo loop pays between the window sync and the next launch** (generate.cpp 13206-13360):
  (1) 1-2 typical/ratio rescue probes, each a kernel launch + FULL stream sync + 8-byte D2H
  (verify.cpp:3488-3498 `row_typical_stats`) = 0.03-0.06 ms; (2) `stage_inputs` 0.19-0.48 ms at the
  top of the NEXT run() (verify.cpp ~2320-2336) while the GPU sits idle post-draft; (3) the sampler's
  second sync (item 8). The drafter's gated step loop (mtp.cpp 1697-1746) spends ~0.5-0.7 ms/window
  SPINNING between step graphs — host-idle, GPU-idle time that can absorb (1) and (2): the rescue
  stats for ALL rows computed inside the window graph (a per-row stats kernel reads logits already in
  VRAM, ~1-2 µs/row; results to mapped memory), the next window's rows staged incrementally as each
  draft lands (row 0 is known at the sync; row j+1 when step j's h_out_ lands). Commit stays async;
  emit is pipe-buffered (F-232 C7, confirmed).
- **Expected:** −0.25..−0.6 ms/window → **+5..+11 t/s**. Cost: S for the probes+sampler, M for the
  staging hoist. Risk: LOW (same kernels, precomputed; bitwise class). Falsifier:
  STRATA_DECODE_TIMING=1 one block — the `stage` + rescue columns on the same prompt.

### B. The hc latency-chain fix (§3 item 3) — the top kernel tranche, +14..+26 t/s
(See §3 item 3; new contribution = the accumulator-chain anatomy and the three-rung bit-class ladder,
which turns "bf16-native dots, different bits, full gate" into a stepped, mostly-cheap path.)

### C. The 5090 has never voted on the IL tables (§3 item 1) — the head/dense cells are tunable env-only
(See §3 item 1; new contribution = the exact dead cells: sm_120 has no T=1 IL entry at all and
Q6_K-cls4@T=2 is 0 → ~73% of windows run the plain head kernel; the Q6P_SELFTEST boot measures the
real shapes for free.)

### D. W(1) = 8-9 ms and the post-adoption c_row ≈ 2.2-2.6 (§2.1) — re-opens the 0.35 re-sweep now, not post-retrain
(See §3 item 2; new contribution = the recipe-change-under-the-verdict argument with the boot rows
to back it.)

### E. Closed by this audit (do not queue): solo emit (F-232 C7 confirmed — printf is a memcpy,
fflush one 20-byte syscall/window); GDN state read/write as a time lever (rec 0.23 ms/window ≈ near
floor; the state traffic ~147 MB/window ≈ 0.1 ms at roofline — the ILP rewrite of §3/T15C #9 is the
only piece worth doing, for c_row not W); commit() (async, off-path, generate.cpp:13257 +
wait_commit at request end); the 12807 piped loop (multi-GPU/HIP only, generate.cpp:3607 `on=false`
on CUDA — the standing solo loop is 13101+; T13A's line reference was stale, as T15C also noted).

---

## 5. The 198.5 → 250 composition (ranked shortlist)

Transfer law: **+20 t/s per −1.0 ms/window** (probe-mean terms, ±10%). Three tiers; within a tier,
order is (expected t/s) × (confidence) / (cost).

**Tier A — cheap, env/boot-falsifiable (no code), ~6 boots total:**
1. **Q6P_SELFTEST boot + MMVQ_IL_ROWS head ABAB + parity bench on this card** (§3.1) — +3..9 t/s;
   falsifier: the selftest's own paired timings.
2. **min-p 0.35 re-stair on today's recipe** (§3.2) — 0..+4% (+0..8 t/s); falsifier: the stair read
   (ΔC/Δoffered < 0.45 closes it).
3. **STRATA_TSUM=1** (§3.3 rung 1, bitwise) + the T15C env tranche (L2 {0,72}, PcieAuto, DT
   {0.80,0.90}, PCIE_MIN1, LANECELL=0-on-conc) — +2..6 t/s aggregate; each arm is its own falsifier.
4. **Clock lock (sudo)** — the "stable" half; variance, not mean.

**Tier B — kernel/code work (bitwise or sum-order gated):**
5. **Host-tail package** (§4.A: probes-in-graph, sampler single-sync, stage-into-draft-waits) —
   +5..11 t/s; cheapest code on the board, bits identical.
6. **hc multi-accumulator + TSUM** (§3.3 rung 2) — +14..+26 t/s; gr_parity microbench first (F-217
   law), then canary + 2× MMLU (sum-order class). HC_Q8 as the follow-on rung after the prefill-touch
   check.
7. **Dense/head sm_120 shapes → kIlRows_120 + Q6_PACKED** (§3.1c/d) — +4..9 t/s; bitwise class.
8. **GDN rec ILP rewrite** (T15C #9) — +2..5 t/s AND a c_row cut; gdn_rec_parity harness exists.
9. **PDL into fused_gr/native_moe** (§3.9) — +6..16 t/s potential, HIGH risk, only after #6 lands.

**Tier C — needs the FastMTP retrain (T14A/B/C):**
10. **The retrain** — +5.7% acceptance-side alone (→ ~210 from today, or → ~250-260 on the post-
    tranche base); then the free compounding options re-open at the new constants: G4 ungated
    (crossover O* drops from ~4.0 toward ~3 as waste shrinks), G5 deeper chains (threshold ~0.42-0.55
    post-tranche vs deep marginals 0.80-0.92 expected). The retrain is what converts the tranche's
    good boots into every boots.

**Composition check:** 198.5 + tiers A+B central (−2.4 ms/window, +45-55 t/s steady-equivalent) ≈
**240-255**, with the boot band ±9 until item 4 lands; + retrain (+6..10% on the new base) ≈
**255-270 stable**. 250-stable is the central outcome of items 1-10 minus any single failure; the
first four boots (A tier) decide the shape of the rest for ~one afternoon of batteries.

---

## 6. Contradictions and their resolution

1. **T13A "IL gated T≥2" vs T15C "IL dispatches at T=1 too" vs this audit.** All three describe
   different layers: the DISPATCH gate admits ncols 1-4 (T15C right), but the sm_120 TABLE zeroes
   T=1 (no kIl1Arch entry) and Q6_K@T=2@cls≥12288 (shared table) — so the head's EXECUTION is plain
   on ~73% of windows (this audit's cell read). T15C's "fallback at T≥5 only" was incomplete; T13A's
   question "does mmq_il_supported reject Q6_K?" resolves to "no — the tables do".
2. **T14C c_row = 3.56 vs this audit's 2.2-2.6.** Not a contradiction: 3.56 was measured on the
   pre-POOL_DIET recipe (the s35 3v3 predates the diet's launcher adoption by minutes; FUSE_HITS was
   in, the diet was not). T14C's threshold law stands; its constant is re-scoped to 2.0-3.5
   (W(1)-anchored) with the consequence that the 0.35 re-sweep is at/below threshold TODAY.
3. **T13C/T13B "FUSE_WAIT/LFUSE built, never armed" vs F-206.** They WERE armed and swept negative
   (−1.3%/−3.0%); the tasking's "still-open" status was stale. This audit adds the mechanism: FUSE_WAIT's
   sites are additionally unreachable under standing DEVICE_PLAN=1.
4. **T15C #2's "the packed kernel answers the head question in one boot" vs the ~73%-plain finding.**
   Compatible and stronger together: the selftest measures nc 1/2/4 on the real shape — exactly the
   cells this audit identified as dead in the tables.
5. **F-228 "the depth lever routes through a better drafter" vs §3.2.** F-228's verdict was correct
   for its recipe; the diet changed the cost constant. The re-sweep is a measurement debt, not a
   reversal of F-228's mechanism (the T-scaling of rows is real and was the right call at c_row 3.56).

---

## 7. Open questions

1. What does `mmvq_il_parity --bench --emit-table` print on THIS 5090 for (248320, 2560, Q6_K) at
   nc 1-4 and for the dense Q4_K shapes? (One offline run; decides items 1/7.)
2. Does a 4-accumulator gr_down variant reach the byte floor's neighborhood at the exact (320,
   10240, bf16) shapes? (fused_gr_bench offline; decides item 6's size.)
3. The exact post-adoption c_row: one STRATA_DECODE_TIMING stair (depth 1 vs 3 vs 5, same boot)
   pins it and shrinks §3.2's threshold band from ±0.13 to ±0.04. One boot.
4. Does HC_Q8's load-time replacement touch the prefill read path (the F-230 tax question, T15C §6)?
   One prefill parity probe.
5. Does the packed-Q6 head (+521 MiB) fit beside the cohort arenas without a residency regression?
   (The selftest boot prints it.)
6. PDL: does `STRATA_DF_PDL=1` change anything today given the 3-kernel coverage? (One nsys boot
   answers whether the attribute fires at all before any intrinsic work is priced.)

## 8. Implications

- The single most valuable next action is **Tier A's four boots**: they are all falsifiable for the
  cost of a battery afternoon, two of them (IL cells, TSUM) are bitwise, and one (the 0.35 re-stair)
  is a re-measurement the recipe change owes the ledger regardless of outcome.
- The 250-stable path does NOT require inventing new kernel families: it requires executing the
  already-built instruments (selftest, parity harnesses, gdn_rec_parity) on the card that never got
  to vote, plus one ~15-line accumulator fix in the top kernel line, plus the host-tail plumbing —
  then the retrain converts headroom into floor.
- The conc lane shares segments 1/2/3/4/7 (the tranche) — every bit-safe adoption here also lifts
  the 312 conc toward its next ceiling; the pair lane shares the hc/dense lines.

## 9. Sources

Primary code (all read this audit): src/program/generate.cpp (3600-3640 pipeline gate; 13101-13400
solo loop; 13222-13246 rescue walk; 290-345 typical/ratio + typical_stats; 13257 commit; 13325
draft); src/core/verify.cpp (80-95 envs; 249-310 fuse_wait/fuse_hits; 900-945 lfuse/pdl scope; 1163-
1175 dense mm lambda; 1500-1600 router/plan/doorbell; 1660-1765 grouped/waitB/PCIe/combine; 1840-1886
head dispatch; 2330-2560 run tail + service loop + sampler; 3488-3498 row_typical_stats); src/core/
mtp.cpp (1620-1746 gated/ungated draft); src/core/native_dense.cpp (HC_Q8 plumbing per T15C);
src/kernels/cuda/fused_gr.cu (46-168 unpack8/dot8u/gr_down/gr_up; 1046-1064 GR_V3/MAX4; 1969-1985
TSUM launch); src/kernels/cuda/native_mmvq.cu (1156 g_multi_exact; 2588-2810 IL tables, env rows,
supported(), il dispatch); src/kernels/cuda/gdn.cu (15-33 rec kernel shape); src/kernels/cuda/
elementwise.cu (330-380 doorbell kernels); src/kernels/cuda/shared_expert.cu (45/161-162 native_bf16);
include/strata/kernels/pdl.hpp; include/strata/core/verify.hpp (448 ar_on); launch-custom-iq3s-8081.sh;
custom-iq3_s-8081.json; docs/MMVQ_IL_TABLE.md.
Measured artifacts: campaign/results/standing-restore-f233.txt, final-bench-r1..r4.txt,
final-bench-series.log, diag-phase-table.txt (2026-10-14 warm rows), fbench-lanes.log; nvidia-smi
(this audit).
Journal/dossiers: SVG-CUSTOM.md F-203 (doorbell store + spec-8 + DVFS), F-206 (fusion tranche
negatives), F-217 (GR_WIDE/nsys law), F-222/F-223, F-225-F-233; campaign/research/T13A (superseded
budget), T13B (C5, stale per F-206), T13C (FUSE_WAIT status, stale), T14C (c_row law, re-constant-ed
here), T15C (env inventory — items #1/#2/#3/#4/#5/#6/#7/#8/#9 shared or reconciled above).

## 10. Methods note

Unproductive searches, not repeated: (1) the T13A-cited solo-loop lines 12807-13048 are the
multi-GPU pipeline (auto-on requires multi_gpu and is HIP-gated, generate.cpp:3600-3615) — the
standing solo loop is at 13101+; T13A's and the tasking's line anchors were both stale (T15C's
methods note agrees). (2) No recipe-current STRATA_VERIFY_PROFILE/DECODE_TIMING capture exists
anywhere (diag-phase-table.txt is 2026-10-14; the F-218/F-222 maps are the newest per-stage data) —
the §2 adjustments lean on measured arm deltas instead. (3) The stage table's "hc0/hc1" labels do
not map 1:1 to fused_gr functions; the 2.8 ms total is taken from the six hc class rows, consistent
with T13A's 2.87 on the same table. (4) `STRATA_GR_FAST`/`GR_NORM_FAST` builds exist but have no
CUDA decode dispatch (T15C methods note, confirmed). (5) The draft-head (Q4_0, 40,525 rows) is not
IL-eligible (type 16 not in {23,12,13,14,11}) — no draft-head IL lever exists; irrelevant anyway
while the draft phase is host-gated. Source count: 14 code files, 8 measured artifacts, 6
journal/dossier references.
