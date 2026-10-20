# THE v0.2.1 WAVE, ANALYZED — what produced +15.6/+25.9% M1 on sm_120, and what we cherry-pick (F-244 follow-up; 2026-10-20)

Analyst dossier for the custom-2 campaign. Read-only analysis of `/home/svg/LLM/strata` at v0.2.1
(884 commits, v0.1.45..v0.2.1) against our fork `/home/svg/LLM/strata-custom-2` (v0.1.44 base + five
adoptions + norm_fast + T18 fix; standing scoreboard 203.9 / 202.0 / 306.1, F-242). Every claim
carries a commit hash, file:line anchor, or diff anchor, with a confidence level. No builds, boots,
or API calls were made; both trees untouched.

---

## 1. EXECUTIVE SUMMARY

1. The wave is NOT a scheduling/batching/spec wave. The drafter was untouched (mtp.cpp +3 lines =
   an sm_60 no-op pack call; generate.cpp's only decode-adjacent additions are I/O-tier and
   two-card flags). verify.cpp grew +66 lines, almost all of it one thing: the SIDE_MIXER default
   flip. The conc gain has no structural component upstream — no second window, no cohort, no
   service-loop change exists anywhere in the wave. **Our COHORTS mechanism remains fork-exclusive.**

2. The wave is a **kernel package, default-flipped for sm_120**, concentrated in five commits dated
   Oct 17-18 2026 (ba2c2d5f, 75fabbd9, 44997124, 94104176, 75afe3f4) plus a prefill-side default
   (b1e0fdaa) and an I/O-tier package for Linux (82c80ac4, 8b1139d0, ecf2bdcb). Upstream's own
   end-to-end measurement of the decode package: **+11.9% story / +14.0% code / +13.1% 8K doc on an
   RTX PRO 5000 (sm_120, IQ3_S, all experts in VRAM), same tokens** (docs/DETAILS.md, added in
   44997124). Our measured +15.6% IQ3_S M1 is the same package landing harder because our standing
   runs --spec 6 (windows of 6-8 rows) where the package's biggest single piece — the 5-8-column
   interleaved matvec — pays the most, and our windows carry host-offload waits that the kernel
   savings partly unmask. Confidence: HIGH on the composition, MEDIUM on the exact split.

3. The five decode commits are all **bitwise** (each carries its parity harness and the identity
   gate). That makes them our cheapest possible ports: no MMLU gate, canary + battery only, per the
   campaign law's bit-identity class.

4. Top cherry-picks, ranked by (measured magnitude x portability):
   - **PORT NOW:** mmvq_il 5-8 columns + the sm_120 tables (75fabbd9 + 44997124) — the wave's
     biggest single piece and the conc lever; self-contained in native_mmvq.cu + two caller caps.
   - **PORT WITH CONFLICT WORK:** GR_DOWN_TG per-token block rows (94104176 + the #2070 sm75-series
     machinery it rides); SIDE_MIXER=2 (ffc30b8d + 477ac13c + ba2c2d5f) — our tree already has the
     graph-branch machinery (df_branch_) it runs on.
   - **BENCH FIRST (idea may be absorbed):** GR_UP75 load-hoisting — same hc-read family our
     norm_fast flip already paid; F-241 S4 measured exactly this absorption pattern for gr_up8.
   - **PORT NOW (small):** GU_SWIGLU one-launch gate/up+SwiGLU+q8_1 (75afe3f4), ~+0.5-0.9% decode.
   - **PORT WITH GATES (lossy, prefill):** PF_FUSED default on 80+SM sm_120 (b1e0fdaa), +5-7% prompt.
   - **FALSIFY ON STOCK FIRST:** the Linux I/O-tier defaults (io prefetch, zero-copy, decode gate,
     drive probe) — one boot of stock .2.1 with the four env-offs prices their share of the wave.

5. Convergence (mission item 4): upstream converged on TWO of our adoptions and ZERO of our
   structural wins. STRATA_ATTN_DAILY — upstream made it sm_120-default (ba2c2d5f); we already arm
   it: keep ours, no action. GDN chunked prefill — upstream keeps it opt-in under the SAME env name
   (STRATA_GDN_CHUNKED, src/prefill/kernels.cu:2711-2714) with improved kernels; we run it standing:
   keep ours. STRATA_GR_NORM_FAST — upstream took the mechanism but only as CUDA opt-in (+0.5-0.6%
   on a V100, fec7d1ca); our default flip at pooled +5.2% M1 (F-241) is AHEAD of upstream: keep
   ours, and we now hold evidence-backed-default-flip material for an upstream issue. COHORTS,
   FUSE_HITS, POOL_DIET, and the T18 job_of fix have no upstream counterpart anywhere in the wave
   (verified: zero wave diff lines touch hits_in_combine/repark; dispatch_multi's wave hunks are
   io-prefetch only).

6. Rebase hazard banked: upstream rewrote expert_pool_dispatch_multi's neighborhood for io-prefetch
   (four hunks around old-file :3814/:4012/:4053/:4066). Our T18 guards (job_of reset at call start
   + nt MAXT refusal) are NOT in upstream and sit inside that function — any future merge of
   upstream expert_source.cpp must re-apply them on the new shape.

---

## 2. WAVE COMPOSITION — where the +16-26% comes from

### 2.1 The sm_120 decode package (the M1 story)

One umbrella commit flips four existing bitwise switches default-ON on compute capability 12.0,
governed by a new header (include/strata/core/sm120_series.hpp:20-36: `sm120_default()` = is_sm120
&& !STRATA_SM120_PERF=0; per-device tables so a layer split can mix cards):

- **ba2c2d5f** "sm_120: four bitwise decode switches on by default on a 12.0 card" (2026-10-17):
  STRATA_SIDE_MIXER=2, STRATA_ATTN_DAILY=1, STRATA_GR_UP75=1, STRATA_GR_DOWN_TG=2. Measured on RTX
  PRO 5000 Blackwell (IQ3_S, all experts in VRAM): a fixed 4-token verify round **13.70 -> 12.23 ms
  (-10.7%)** with all four; 2-token rounds 10.37 -> 9.59. Each bitwise (their parity tests + the
  identity gate). Commit-comment per-switch numbers (cumulative runs, so noisy per-switch):
  SIDE_MIXER -3.8%, ATTN_DAILY -2.6%, GR_DOWN_TG pairs -2.4%, GR_UP75 -1.7%.
- **94104176** upgrades GR_DOWN_TG on sm_120 from token pairs to ONE token per block row for every
  launch of 2+ tokens (fused_gr.cu down_tg(): `if (per_dev[dev] == 3) return ct >= 2 ? 1 : 0;` —
  v0.2.1 fused_gr.cu:1246): 2 tok 9.44->9.24, 3 tok 10.81->10.56, 4 tok 11.94->11.90, 5 tok
  14.04->13.70 ms, bitwise (same per-warp sum order).
- **75fabbd9** removes the 5-row cliff in the dense projections: `native_mmvq_il_kernel` gains 5-8
  column cases (R<=2 rows/warp; launch_il_rows maps 4->2 at ncols>4) and sm_120 gets two measured
  tables (kIlRows_120 for 2-4 cols on 80+ SM cards, kIlRows58_120 for 5-8 cols). The cliff it
  kills: a window going T=4 -> T=5 used to fall back to native_mmvq_multi and the dense projections
  jumped **+2.0 ms** (the main head 521 -> 1011 us). With the table: head at 5 cols 899 -> 491 us,
  at 8 cols 1178 -> 667 us. Bitwise (mmvq_il_parity T 1-8, every rows choice).
- **44997124** gates the 2-4 col kIlRows_120 table to 80+ SM cards (5070-class keeps the shared
  table; 5090 = 170 SMs, qualifies) and adds the docs/DETAILS.md block with the package total:
  **+11.9% story / +14.0% code / +13.1% 8K document, same tokens, identity gate passed** — plus a
  spec re-measure showing deeper speculation got cheaper (--spec 6: 1.060/1.177/1.097 of before).
- **75afe3f4** fuses expert gate/up + SwiGLU + q8_1 into one launch (native_gu_swiglu_kernel,
  iq_kernels.cu +192 lines; row_dot2_80_sub16 order preserved): a layer's gate/up+SwiGLU
  35.5+2.4 -> 36.3 us, fixed 4-token rounds 11.92 -> 11.80 ms, decode +0.6/+0.9/+0.5% (6/6 pairs,
  same tokens). Default ON sm_120, STRATA_GU_SWIGLU elsewhere. **8837cb6f** adds the 80-dots-a-row
  guard for trellis formats (the Gyro-S garbage fix — a correctness lesson, not a port item).

Attribution vs our measurement (confidence MEDIUM-HIGH): upstream's package total (+11.9-14.0% at
--spec 4) vs our +15.6% IQ3_S M1 at --spec 6. The gap is coherent: at --spec 6 our windows are 6-8
rows, so the 5-8-column matvec (the largest single piece, ~-0.5-0.7 ms/window on the head alone)
binds MORE for us than in their --spec 4 test. The Q2_0 asymmetry (+25.9% vs +15.6%) reads the same
way: the Q2_0 pack's shorter windows make the same absolute kernel savings a larger fraction, and a
smaller pack keeps more experts VRAM-resident so fewer savings hide under pool waits. Both are
inference from the measured per-piece numbers, not measurements on our card.

### 2.2 The prefill-side default (b1e0fdaa)

PF_FUSED (fused int8 prompt experts) becomes default-ON for sm_120 cards with 80+ SMs
(src/prefill/moe_fused_iq.cu:1391-1428, the `env < 0` CUDA branch: gate/up types IQ2_S/IQ3_XXS/
IQ3_S/IQ4_XS with IQ4_NL/Q2_0 down; `d.s120 && d.sms >= 80`). RTX PRO 5000: prompt **+7.0% 4K /
+5.2% 20K / +3.4% 98K**, teacher-forced KL 0.0002, first-token KL vs FP16 path 0.0118 — LOSSY-tiny.
The 5070 (48 SMs, cache partly in RAM) measured -4.7% on 20K prompts, hence the SM gate. NOT part of
our M1 delta (our probe prompts are short; this is a prompt lane change), but it is part of why
stock .2.1 looks faster on prompt-heavy work, and it is directly relevant to our 262k lane.

### 2.3 The Linux I/O-tier package (a wave component with a caveat for our A-B-A)

82c80ac4 + 8b1139d0 + c3136797 (GGUF read-in-place zero-copy page-cached experts, io prefetch,
decode read-ahead gate), ecf2bdcb (extends the defaults to experts.bin packs when the page cache has
room for >=55% of experts — our 184 GB box qualifies for the ~85 GB pack), 484e06b9 + a7bdbfda (the
#3237 drive probe, default on), e77f1dfc + d8471050 + 8253477b (prefill planner prices host/drive,
request-sized chunks). Upstream's own numbers are low-RAM scenarios (RTX 3060 12 GB: decode up to
x1.38-1.47), which we are not; on our box the plausible effect is cold-boot warm-in (the T17B
finding: boot-time pin drops page cache, first requests re-read 44-55 GB) and the conc lane's higher
miss tier. **This is also a measurement caveat for F-244 itself: the .45 legs of our A-B-A had none
of this, so a small share of the wave delta on a fresh-boot protocol may be I/O-side, not
kernel-side.** Cheapest falsifier, zero code: one boot of the deployed stock .2.1 with
`STRATA_IO_PREFETCH=0 STRATA_COLD_OVERLAP=0 STRATA_IO_PF_GATE=0 STRATA_FILE_TIER_PROBE=0` vs the
default boot, full probe; the delta prices the I/O share on OUR box. Confidence that it is small for
M1 (97.7% VRAM hits, warm by M1 time): MEDIUM.

### 2.4 What it is NOT (checked, so nobody re-litigates)

- **No drafter/spec changes**: wave diff of mtp.cpp/session.cpp/native_head.cpp = pin_budget guard +
  an sm_60 no-op pack call; generate.cpp's draft/spec grep hits are the same I/O lines. The wave's
  spec gain is purely "windows got cheaper," and their spec re-measure (44997124) exploits that.
- **No conc-structural changes**: no cohort/second-window/overlap commits exist (grep across the
  884); the batch path's only perf-relevant wave change is two-card batch-group serial grouping
  (f680a80d, STRATA_BATCH_GROUPS_NV_SERIAL — N/A single card). Conc +17.7% is the kernel package at
  T=8 windows (where kIlRows58 pays max: head -511 us/window) + expert pool worker-count hygiene
  (5ece068f, bec4c838, 88b29d48 — physical-core defaults, E-core counting; on our 9950X3D verify the
  boot banner worker count did not silently change) + possibly the I/O gate on the miss tier.
- **AMD/sycl/sm75/sm86/sm60 work is real but not our lane**: gfx12 WMMA prompt attention + Q2_0
  fused prompt experts, Linux-only defaults (97b680e1, 99f6d472, 961d37cd, 79ac3164); R9700 queue
  ordering (3f1fa67f, b35a9816); iGPU matrix-core prompt sets (eb7eac25, ce968a2f); sm_86/sm_60
  per-arch decode tables (2725c23f; +10.2-14.2% on 3060/A4000/P100); the Volta/Pascal community-PR
  group in 0.1.46 (8b3a9677, 35 commits: pf_wmma, kv_q8 wide gather, V100 fusions). None applies to
  a single 5090. One exception worth a note: kv_q8's 8-wide gather is explicitly below-sm_75 only
  (kv_q8.cu:155-170, pre75_q8_gather) — do not queue it for sm_120.
- Release-note cross-check: 0.2.1 known-limits records "the 0.2.1 tip is 7 to 10% faster in decode
  than 0.1.45" on P100 (sm_60) and A4000 (sm_86) — consistent with a generic-CUDA component
  (GU_SWIGLU is opt-in elsewhere; the generic part is likely code-motion around the new gates) plus
  the per-arch tables. Nothing sm_120-exclusive explains their older-card gain; nothing there is
  portable value for us beyond what is listed above.

---

## 3. RANKED PORT TABLE

| # | Candidate | Mechanism | Evidence | Expected on our box | Conflict surface | Class | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | **mmvq_il 5-8 cols + kIlRows_120/kIlRows58_120** (75fabbd9, 44997124) | interleaved small-batch matvec extended to 5-8 columns (R<=2 rows/warp), measured per-cell tables for sm_120 | bitwise (mmvq_il_parity T1-8); head Q5_K/Q6_K 899->491 / 1178->667 us at 5/8 cols; kills the +2.0 ms T4->T5 cliff; DETAILS.md package total +11.9-14.0% | +5-10% M1 (our windows are 6-8 rows at --spec 6), +6-12% conc (all windows T=8) | LOW: self-contained in native_mmvq.cu; fork caps at 4 cols (src/kernels/cuda/native_mmvq.cu:1556-1566, :2706, :2592 kIlRowsShared, :2666 kIlArch has no 120 row); callers are two sites (verify.cpp:1164-1169, :1863-1865) gated by native_mmvq_il_supported + the interleave cap (native_q8_1_interleave 2<=ncols<=4 -> 8) | **PORT NOW** | server-free: build mmvq_il_parity --bench --emit-table on the 5090, confirm the head's 248,320x2560 Q6_K cells win at our shapes; then one 3v3 ABAB (bitwise -> canary only, no MMLU). Interacts with: nothing (self-contained); benefits scale with our standing --spec 6 |
| 2 | **GR_DOWN_TG per-token block rows** (94104176, riding #2070's sm75-series via 80872d43 + ba2c2d5f's `per_dev==3` arm) | hc down projection: one token per block row at every launch of 2+ tokens (gridDim.y = ct), same per-warp sum order | bitwise; T2-T5 rounds -2.1/-2.3/-0.2/-2.5% vs pairs on PRO 5000; part of the -10.7% four-switch stack | +2-4% M1 (our hc read line 2.87 ms/window) | MEDIUM: fork has NO up75_on/down_tg at all (grep empty in fork fused_gr.cu — the sm75_series machinery landed in 0.1.45, after our base); port = down_tg() + the sm120 arm + whatever launch variants #2070 added; our staged-down experiments (F-241 S2 REJECT) touched the same kernel's THREADS — keep one variable | **PORT WITH CONFLICT WORK** | server-free: fused_gr_bench cold at T in {4,6,8} with/without (in-tree harness); then 3v3. Interacts with: our norm_fast flip (same hc read — but down parallelism is a different axis than the norm's second pass; F-217's "grid shape is load-bearing" cuts both ways, measure) |
| 3 | **SIDE_MIXER=2** (ffc30b8d, 477ac13c Q2_0-guard fix, ba2c2d5f default) | the GDN gates + QSA indexer (keys/snapshot/queries/scores/top-k) run on graph branch streams beside the attention projections; =2 adds gate-projection + query chains on a second branch; bitwise, joins before recurrence/attention | -3.8% 4-token round (cumulative-run number, ba2c2d5f); in the -10.7% stack | +3-4% M1 | MEDIUM and mostly FRIENDLY: the fork already carries df_branch_ (verify.cpp:817-827 setup, :946-958 fork/join lambdas, 13 df_side_ refs = same as upstream's 13); port = g_side_mixer per-device fn + the sidem/side_proj guards (~20 sites, upstream verify.cpp:1041-1050, :1343-1416 GDN, :1587, :1624-1795 QSA) + the `&& !side_mixer` term at :1076. Serial windows only (`!batch_rec_`) — **our COHORTS batch path is untouched by construction**. Our FUSE_HITS block (verify.cpp:1677-1748) is a later stage, untouched. Fork lacks the Q2_0 1-column guard 477ac13c — include it | **PORT WITH CONFLICT WORK** | one boot STRATA_SIDE_MIXER=1, then =2 (bitwise -> M3b stable, watch DF_BRANCH exclusivity; fork's df_branch carries the #905 NVML/GSP caveat comment at :817-821); then 3v3. Interacts with: DF_BRANCH env (mutually exclusive by code — we do not arm it standing); gr_up8 if ever armed (branch grid interleaving) |
| 4 | **GU_SWIGLU** (75afe3f4 + 8837cb6f guard) | gate/up + SwiGLU + q8_1 in one 512-thread launch; 16-lane rows, values bit-identical to the separate kernels | +0.6/+0.9/+0.5% decode, 6/6 pairs, same tokens; rounds 11.92->11.80 ms | +0.5-1% M1 | LOW: fork's iq_kernels.cu already has the dependency (row_dot2_80_sub16 at :1579); port = +192 lines kernel/launcher + gate + sm120 default + the 80-dots guard for non-80-dot formats | **PORT NOW (small)** | gr-style selftest/byte-equality at the arm state, then fold into any scheduled battery leg (cheap rider). Interacts with: nothing (expert gate/up launch only) |
| 5 | **PF_FUSED default, 80+ SM sm_120** (b1e0fdaa) | fused int8 prompt experts default-on for IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS gate-up + IQ4_NL/Q2_0 down | prompt +7.0% 4K / +5.2% 20K / +3.4% 98K (PRO 5000); KL 0.0002 teacher-forced; 5070 counterexample justified the SM gate | +3-7% prefill lane; conc prompt share minor on our probe | LOW-MED code: fork's native_supported (src/prefill/moe_fused_iq.cu:1122-1141) hard-returns false on CUDA default; needs the `env < 0` CUDA branch + DevInfo s120 field (fork DevInfo at :963-967 has sms but no s120) | **PORT WITH GATES** (LOSSY — full law: coupled-MTP acceptance battery + 2x MMLU) | native_expert_parity harness + one prefill-probe A/B before any battery. Interacts with: GDN_CHUNKED (both prefill — separate legs); 262k lane benefits most |
| 6 | **GR_UP75 load-hoisted hc up** (80872d43 + ba2c2d5f arm) | up projection kernel with its rows' loads hoisted (sm75-series kernel, 3+ tokens) | -1.7% round (cumulative number); bitwise | UNKNOWN on our stack — absorption risk: F-241 S4 measured gr_up8 FLAT on top of norm_fast ("once the norm wall is gone, the up grid's extra in-flight bytes buy nothing"); up75 is the same read path | MEDIUM: bring the whole sm75-series up kernel + up75_on(); our UPM_COLS template work (gr_up8, default off) lives in gr_up_multi_kernel — separate kernel, no code clash, but same budget | **BENCH FIRST — take the idea only until priced** | fused_gr_bench cold up at T 6/8, norm_fast ON (our standing), before any port decision |
| 7 | **Linux I/O-tier package** (82c80ac4, 8b1139d0, ecf2bdcb, 484e06b9, planner commits) | zero-copy page-cached experts, io prefetch + decode read-ahead gate, drive probe, request-sized prefill chunks | upstream: x1.14-1.47 decode on 12-GB-card scenarios; unknown at our RAM/expert-hit regime; possibly part of our measured wave delta via cold-boot warm-in | likely small standing; cold-start + conc-miss-tier relevant; overlaps our T17B fast-warm lane | HIGH if ported (expert_source.cpp +842 lines of machinery, and our T18 guards live in the same function); trivial to A/B on stock | **FALSIFY ON STOCK FIRST; idea-level for our cold-start lane** | one stock .2.1 boot with STRATA_IO_PREFETCH=0 STRATA_COLD_OVERLAP=0 STRATA_IO_PF_GATE=0 STRATA_FILE_TIER_PROBE=0 vs default, full probe — prices the wave's I/O share for free |
| 8 | **Improved chunked-GDN kernels** (gdn_chunk.cu +636 new, gdn_chunk_scan_kernel in prefill/kernels.cu:991/:1205-1212) | three-kernel cumsum/kkt/fwd super-block form of the chunked prefill recurrence (wmma sm_70+, FP32 oracle via STRATA_GDN_CHUNK=2) | upstream keeps the whole path opt-in (STRATA_GDN_CHUNKED, prefill/kernels.cu:2711-2714) — no default-flip evidence | our GDN_CHUNKED already took the main prefill recurrence win (F-224 1.68x); residual GDN 26.1% of prefill | MEDIUM (same dispatch site our adoption modified) | **LATER — prefill lane only** | our own gdn_rec_parity --bench against the new kernels, prefill probe only |
| 9 | ATTN_DAILY per-device default (ba2c2d5f) | same kernel we already arm | convergence | — | none | **NO ACTION (converged)** | — |
| 10 | norm_fast (fec7d1ca) | upstream opt-in only, +0.5-0.6% V100 | convergence; WE ARE AHEAD (default flip, pooled +5.2%) | — | none | **NO ACTION — keep ours; bank as upstream-issue evidence** | — |
| 11 | COHORTS / FUSE_HITS / POOL_DIET / T18 fix | no wave counterpart (verified: zero hits in the wave diff for hits_in_combine/repark; dispatch_multi wave hunks are io-prefetch) | — | — | see rebase hazard, §5 | **NO ACTION** | — |

Sequence recommendation: #1 first (biggest, cleanest, server-free falsifier), then #4 as a rider leg,
then #2 and #3 as the fused_gr/verify.cpp conflict-work tranche (one variable per leg, #6 priced by
bench before it earns a port), #5 behind its quality gates when the prefill lane next opens, #7's
falsifier run on the stock install any time the box is otherwise idle.

---

## 4. CONVERGENCE NOTES (mission item 4, direct answers)

- **ATTN_DAILY: YES — converged.** Upstream's ba2c2d5f makes STRATA_ATTN_DAILY=1 the sm_120 default
  (qsa_decode_attn.cu daily_attn() per-device form). We arm it in the standing launcher
  (launch-custom-iq3s-8081.sh:13). Same kernel, same mechanism — we keep our env arm; when merging
  anything else from qsa_decode_attn.cu, prefer upstream's per-device default form to avoid two
  default paths.
- **norm_fast: YES — converged, and we are ahead.** fec7d1ca (in-wave, 0.1.46 Volta group) flips
  STRATA_GR_NORM_FAST to opt-in ON CUDA, measured +0.5-0.6% on a V100+PG500 split. Our F-241 S3
  flip is the same gate with the default inverted, pooled-measured +5.2% M1 / +5.5% conc on sm_120.
  Keep our version, no port. Byproducts: (a) our pooled 6v6 is exactly the evidence-backed
  default-flip case AGENTS.md's contributing rules ask for — candidate upstream issue; (b) if we
  ever merge fused_gr.cu wholesale, the two default directions will conflict — ours wins.
- **GDN chunked prefill: YES — same mechanism, same env name.** Upstream v0.2.1 still gates it
  STRATA_GDN_CHUNKED=1 (prefill/kernels.cu:2711-2714) and improved the kernels in-wave (new
  src/prefill/gdn_chunk.cu; gdn_chunk_scan_kernel). We adopted the mechanism standing-ON (F-224).
  No action now; the improved kernels are a prefill-lane option later (#8 above).
- **COHORTS: NO upstream counterpart.** Nothing in the wave puts a second window in flight; the
  conc gain is kernels + I/O hygiene. Our +24.8% conc mechanism (F-233) remains exclusive — and the
  margin math says it must stay: stock .2.1's 281.4 conc vs our 306.1 is only +8.8% for them to
  close, and ports #1/#2/#3 all feed our conc lane too.
- **FUSE_HITS / POOL_DIET / T18: NO upstream counterpart** (zero wave-diff hits; see §1.6 rebase
  hazard for T18).
- **Partial convergence worth one line:** upstream built a bitwise shared-expert dual-GEMV+SwiGLU
  kernel (s_gemv_pair_silu, src/kernels/cuda/shared_expert.cu:336-344, new s_gemv_pair.cu) — but it
  is gated STRATA_EXPERIMENTAL_SM60 + STRATA_FUSE_DECODE, V100-experimental, not part of any
  measured default. Our T1.1 queue item (shared expert fused into routed launches) stays open; this
  is reference code, not a convergence claim.

---

## 5. CONFLICT SURFACE SUMMARY (fork file:line anchors)

- src/core/verify.cpp (fork): g_mmvq_il :159; fuse_hits_on :283-297; df_branch_ setup :817-827
  (carries the #905 NVML/GSP caveat); record_window :897; sg_gated_ :932/:1624; br + fork/join
  :946-958 (no side_mixer term — the SIDE_MIXER port adds it); mm() il sites :1164-1169 and head
  :1863-1865 (callers of the 4-col cap); hits_in_combine :1677-1748 (FUSE_HITS — later stage,
  untouched by all top ports). Upstream counterpart anchors: g_side_mixer :185-201; df_branch
  :934-940; mixer locals :1041-1050; `&& !side_mixer` :1076; GDN branch sites :1343-1416; side_proj
  interleave :1587; QSA branch sites :1624-1795.
- src/kernels/cuda/native_mmvq.cu (fork): il kernel :1500; launch_il 1-4 cases :1556-1566;
  kIlRowsShared :2592; kIlArch :2666 (no 120 row); rows_for :2706 (caps ncols 4); HIP stub :2809.
  Port = upstream's launch_il 5-8 cases + launch_il_rows 4->2 map + kIlRows_120/kIlRows58_120 +
  native_q8_1_interleave cap 8 + rows_for ncols>4 branch (upstream native_mmvq.cu, commit
  75fabbd9 hunks).
- src/kernels/cuda/fused_gr.cu (fork): gr_norm_split_fast_kernel :762; norm_fast_on :1084-1088
  (our flip); flip use :1131/:1137/:2392. NO up75_on/down_tg (absent — 0.1.45 machinery). Upstream:
  gr_up75_kernel :442; up75_on :1191-1213; down_tg :1223-1247 (per_dev==3 -> `ct >= 2 ? 1 : 0`).
- src/kernels/cuda/iq_kernels.cu (fork): row_dot_80_sub16 :1206; row_dot2_80_sub16 :1579 — the
  GU_SWIGLU dependency exists.
- src/prefill/moe_fused_iq.cu (fork): DevInfo :963-967 (has sms, lacks s120); native_supported
  :1122-1141 (CUDA default hard-false — the PF_FUSED port site). Upstream: native_supported
  :1391-1428.
- src/core/expert_source.cpp (fork): T18 guards in expert_pool_dispatch_multi (F-240). Upstream
  wave touched the same function four times for io-prefetch (old-file hunks at :3814/:4012/:4053/
  :4066) — reapply our guards on any upstream merge of this file.

---

## 6. SOURCES (commits, diffs, docs — the wave, v0.1.45..v0.2.1 unless noted)

1. ba2c2d5f — "sm_120: four bitwise decode switches on by default on a 12.0 card" (sm120_series.hpp
   new; verify.cpp g_side_mixer; fused_gr.cu up75_on/down_tg arms; qsa_decode_attn.cu daily_attn
   per-device). 13.70->12.23 ms 4-token round, all four.
2. 94104176 — "sm_120: the hyper-connection down projection with one token a block row (bitwise)";
   down_tg() final form `ct >= 2 ? 1 : 0`; T2-T5 round numbers.
3. 75fabbd9 — "mmvq_il: 5-8 column interleaved kernels and the sm_120 tables (bitwise)"; +2.0 ms
   T4->T5 cliff; head 899->491 / 1178->667 us; STRATA_MMVQ_IL_WIDE/_120 switches.
4. 44997124 — "mmvq_il: the sm_120 2-4 column table only on 80+ SM cards; docs"; docs/DETAILS.md
   new section: package total +11.9/+14.0/+13.1% story/code/8K, same tokens; spec re-measure table;
   docs/MMVQ_IL_TABLE.md table routing (5090-class -> kIlRows_120).
5. 75afe3f4 — "experts: gate/up, SwiGLU and q8_1 in one launch (bitwise; default on sm_120)";
   35.5+2.4->36.3 us; 11.92->11.80 ms; STRATA_GU_SWIGLU.
6. 8837cb6f — "gu_swiglu: the 80-dots-a-row guard for every format" (Gyro-S Blackwell garbage fix).
7. b1e0fdaa — "sm_120: PF_FUSED default and per-token HC down blocks only on 80+ SM cards"; prompt
   +7.0/+5.2/+3.4%, KL figures, 5070 -4.7% counterexample.
8. ffc30b8d — "Verify window: the mixer's input-only work on graph branches (STRATA_SIDE_MIXER=1/2,
   opt-in)" (Eddoursul-origin; df_branch exclusivity).
9. 477ac13c — "verify: the side-projection q8_1 interleave copies need two or more columns"
   (SIDE_MIXER=2 Q2_0 1-column fix; v0.1.45).
10. 80872d43 — "#2070 sm_75: hc up kernel with batched row loads for 3+ tokens, down token pairs for
    T 4/6" (v0.1.45; the machinery GR_UP75/GR_DOWN_TG flip).
11. 7f1964a6 — "sm_75 series: the bitwise Turing kernels (#2069-#2072 #2081 #2093 #2220-#2222) are
    opt-in with STRATA_SM75_PERF=1" (v0.1.45; fork lacks the whole series).
12. fec7d1ca — "fused_gr: STRATA_GR_NORM_FAST=1 on CUDA too (opt-in; +0.5-0.6% V100)" — the
    norm_fast convergence point.
13. 82c80ac4, 8b1139d0, c3136797, ecf2bdcb, 484e06b9, a7bdbfda, e77f1dfc, d8471050, 8253477b — the
    Linux I/O-tier package (zero-copy page-cached experts, io prefetch + decode gate, pack defaults,
    drive probe, prefill host plan).
14. 5ece068f, bec4c838, 88b29d48 — CPU pool worker defaults (physical cores, E-core counting).
15. 2f44bf51 — STRATA_HC_REQ8 (v0.1.45, NOT in the wave; noted to keep it out of wave attribution).
16. 2725c23f — sm_86/sm_60 per-arch decode tables (cross-arch corroboration only).
17. 97b680e1, 99f6d472, 79ac3164, 961d37cd, 3f1fa67f, b35a9816, eb7eac25 — AMD gfx12/R9700/iGPU
    defaults (N/A single-5090; the Q2_0 fused prompt experts are HIP-side).
18. 132faae7 + kv_q8.cu:155-170 — 8-wide KV gather is below-sm_75 only (do-not-queue note).
19. d284809f, c0d02ba9, s_gemv_pair.cu, shared_expert.cu:336-360 — V100 experimental fusions
    (STRATA_FUSE_DECODE/EXPERIMENTAL_SM60; T1.1 reference only).
20. 7c4ff2de, f1a9eaa4, src/prefill/gdn_chunk.cu, src/prefill/kernels.cu:991/:1205-1212/:2711-2714 —
    improved chunked-GDN prefill, still STRATA_GDN_CHUNKED opt-in.
21. docs/release-notes-0.2.0.md, docs/release-notes-0.2.1.md — default tables, I/O numbers, the
    "+7-10% generic CUDA decode vs 0.1.45" cross-check, known limits.
22. Fork record: SVG-CUSTOM.md F-238..F-244; SVG-CUSTOM-TODO.md T15/T15C/T16/T17/T18 + scoreboard;
    campaign/results/v02-bench.log (the A-B-A); launch-custom-iq3s-8081.sh:8-36 (standing env);
    campaign/results/t15c-*.log, t18-* (adoption evidence).

— Analysis complete. Box untouched; no builds, boots, or checkouts performed; this dossier is the
only file written.
