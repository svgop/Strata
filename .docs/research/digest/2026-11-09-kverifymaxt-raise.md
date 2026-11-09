---
type: research-digest
id: 2026-11-09-kverifymaxt-raise
date: 2026-11-09
scope: kVerifyMaxT 8→12/16 build scoping — the verify window's row cap as a compile option (window_limits.hpp import), every dependent site, the MMVQ 8-column gate, build cost, lane math, quality gates, rollback
subject: include/strata/kernels/verify_kernels.hpp:21; include/strata/kernels/window_limits.hpp (new); src/kernels/cuda/native_mmvq.cu:830; src/kernels/cuda/verify_kernels.cu; src/kernels/cuda/s2_expert_grouped.cu:853; include/strata/core/verify.hpp:395-635; src/core/verify.cpp:816; src/core/mtp.cpp:319; src/program/generate.cpp:3302-9202; include/strata/kernels/cpu/expert.hpp:145; include/strata/spec/controller.hpp:28 (fork-owned)
method: source reads of every kVerifyMaxT/MAXT consumer in strata-custom-2 (grep + per-site reads); upstream commit reads (Niko1221/Strata 9d3c8c36 "windows of up to 16 rows", 64bf6908 "STRATA_WINDOW_MAX_T=12") from the checked-out stock tree /home/svg/LLM/strata; `git apply --check` of both upstream patches against this tree (both fail — guided manual port, not cherry-pick); smem arithmetic for GMAX 12 vs 16 from the shared-array declarations; acceptance-model math from the task's 3.15 rows/window avg and 57-88% acceptance band. No src/ file edited.
---

# kVerifyMaxT 8→12/16 — the window limit is an import, but only its batch half

## 0. Verdict (read this first)

**Feasible in ONE session: the BATCH-window widening (P1). NOT in that session: the SOLO spec-window widening (P2) — and the task's stated payoff (serial-lane ring amortization) lives entirely in P2.**

| # | finding | evidence |
|---|---|---|
| V1 | Upstream resolved exactly this (commits `9d3c8c36` 16-row, `64bf6908` 12-row; header `include/strata/kernels/window_limits.hpp`: `kVerifyNarrowT=8`, `kWindowMaxT=STRATA_VERIFY_MAX_T` macro, CMake `STRATA_WINDOW_MAX_T=auto\|8\|12\|16`). Proven: narrow windows bitwise unchanged (3096/3097 HIP kernels identical; CUDA identical or NVVM-renamed). | /home/svg/LLM/strata (stock tree, checked out at 9f7d977d); commits NOT in our history (`git cat-file -t 9d3c8c36` → not found) |
| V2 | Both patches FAIL `git apply --check` on this tree (all 12 key files diverged from the patch parent 46c152a1: native_mmvq.cu 733 lines, verify.cpp 1656, generate.cpp 4426). This is a **guided manual port** (~19 files edited, 2 new, ~500-700 insertions), with our own static_assert tripwires catching what the eye misses. | /tmp/wide16.patch, /tmp/wide12.patch apply logs |
| V3 | **Upstream deliberately keeps SOLO spec windows at 8**: their generate.cpp REFUSES `--spec > kVerifyNarrowT` ("only a plain --batch forms the wider windows") and mtp.cpp caps the drafter at `kVerifyNarrowT` — the draft layer's kernels (grouped 512-expert head, fused reads) are narrow. The import widens BATCH windows only. | upstream 9d3c8c36 generate.cpp + mtp.cpp diffs |
| V4 | Therefore the serial lane's ~48+ per-layer doorbell rings per window do NOT divide further under P1: solo windows stay ≤ 8 rows. P1's payoff is the pair/conc lanes that form `--batch` windows (fill guaranteed by parallel slots, acceptance-irrelevant). P2 (solo >8) is ~4-6 more file edits — small, but unproven upstream and acceptance-economics-limited (§5). | §5 math |
| V5 | **Target 12, not 16, first**: at 12 every 8→12 cost cliff is avoided — expert tables still fit (12×10=120 ≤ kMaxWindowEntries=128, no 256 doubling), s2 grouped smem fits statically (38.4 KB < 48 KB CUDA static limit; 16 needs upstream's SKIP-second-pass rework of the grouped kernels), MMVQ adds 4 exact-column instantiations per site instead of 8. 12 is an upstream-supported value with measurements (64bf6908, RESULTS-NV2.md on their box). | §3 cost table |
| V6 | Our fork owns two gates upstream lacks: `include/strata/spec/controller.hpp:28` (`static_assert(K_MAX == kVerifyMaxT)` — fires at 12) and the campaign `dec_moe.cuh` copies (NOT in the build — no action). controller.hpp must be relaxed or its cost tables extended; upstream never touched it. | grep, CMakeLists (campaign/ absent) |

**Verdict on the task's five questions:** import feasible in one session = YES for P1 (batch); the minimal viable patch set is §3 P1 (12-first). Solo ring amortization needs P2 as a second session after P1's gates pass. Full upstream port NOT required: upstream's 23-file commit is the map, and our fork's T65 tripwires (static_asserts at mtp.hpp:212-300, verify.hpp:578-637, native_mmvq.cu:833, s2_expert_grouped.cu:856, expert.hpp:146, controller.hpp:28) turn every missed site into a loud compile error.

## 1. Dependency map — every site, and what raising kVerifyMaxT does to it

Dispositions: **AUTO** = follows the constant cleanly (buffer/array sized by kVerifyMaxT; guarded by an existing static_assert that then forces the widen — compile-loud, not silent). **EDIT** = hand edit required. **SILENT** = hardcoded 8 with NO static_assert — would corrupt at T>8 without a compile error. **TEST** = parity/bench harness whose loop widens.

| site (our tree) | what it is | disposition at 12 |
|---|---|---|
| include/strata/kernels/verify_kernels.hpp:21 | `kVerifyMaxT = 8` — the one constant | EDIT → `kWindowMaxT` (upstream form; +`#include window_limits.hpp`) |
| src/kernels/cuda/native_mmvq.cu:830-834 | **THE MMVQ GATE**: `MAX_NCOLS = 8` + static_assert (compile error at T>8 — the T65 "graph-capture throw" wall) | EDIT: `MAX_NCOLS = STRATA_VERIFY_MAX_T > 8 ? STRATA_VERIFY_MAX_T : 8`; switch cases 9..12 behind `#if STRATA_VERIFY_MAX_T > 8` in launch_multi (1517-1524), native_mmvq_il (1770 — fork-added, upstream lacks it), wave_launch (1899+), q8_packed (~2629), q6_packed (~2774); sm60 packed path runs 9..12 in passes of 8 (upstream form) |
| src/kernels/cuda/q8_sm60.cuh:121 | `ncols > 8` refuse — SILENT literal | EDIT → `MAX_COLS=8` + passes (upstream 9d3c8c36) |
| src/kernels/cuda/native_bf16.cu:288,312-314 | rows/multi GEMV `n_tok > 8` refuse — SILENT literal | EDIT: rows kernels stay 8-col (fall through to switch); multi takes `STRATA_MMVF_WIDE(9..12)` exact per-row instantiations (upstream form) |
| src/kernels/cuda/verify_kernels.cu:110,187 | `template <int MAX_T = kVerifyMaxT, EXACT_T>` GDN bodies | EDIT: default to `kVerifyNarrowT` (upstream) so T≤8 windows keep their exact code; dispatch `default:` instantiates `<kVerifyMaxT,false>` — T=9..12 share ONE generic template each (no exact-T 9..12 project) |
| verify_kernels.cu:1131,1181 (case 7/8) | ours routes 7,8 via `default: <kVerifyMaxT,false>` (same as upstream pre-commit) | EDIT: add `case 7: case 8: <kVerifyNarrowT,false>` so narrow stays bitwise (upstream diff) |
| verify_kernels.cu:1234 | gdn_state_replay `<kVerifyMaxT>` | EDIT: narrow/wide split on `n_tok > kVerifyNarrowT` |
| verify_kernels.cu:1502-1507 | resident_plan: 128 threads, `kVerifyMaxT*10 <= kResidentPlanMax`, "at most 4 warps" assert | EDIT: template `<NT>` 128/256, `s_wsum[NT/32]` (upstream form; static_assert upgraded to ≤256, 8 warps) |
| verify_kernels.cu:1855-1859 | dm_wait_combine gate `n_tok > kVerifyMaxT` + alignment | AUTO (follows constant; message text edit cosmetic) |
| src/kernels/cuda/s2_expert_grouped.cu:853-856 | `GMAX = 8` + static_assert (fires at 12 — loud) | EDIT at 12: `GMAX = kVerifyMaxT`. smem: xs_q[12][640]=30.0 KB + xs_d 0.9 KB (887-888); gu_grouped_t xs_w[8·12·80]=30.0 KB + xs_dh 7.5 KB (1023-1024) = **37.5 KB < 48 KB static limit → NO upstream SKIP-pass port needed at 12**. At 16: 50 KB+ → OVER the limit → SKIP pass mandatory (upstream's actual change). Parity-gate the 12-smem form (deviation from upstream's code path) |
| src/kernels/cuda/fused_gr.hpp:68 / fused_gr.cu | fork-owned `kFusedGrMaxT = 8`; upstream added `kFusedGrWideT` + 8-row slices (hyper-connection Q8_0 read skips rows otherwise) | EDIT (upstream form; SILENT-class — verify the Q8_0 read slicing) |
| src/kernels/cuda/shared_expert.cu:219-220 | `n_tok > 8` refuse — SILENT literal | EDIT → kVerifyMaxT |
| src/kernels/cuda/native_moe.cu:178,185 | combine gate `n_tok <= 8` — SILENT literal | EDIT → kVerifyMaxT |
| src/kernels/cuda/iq_kernels.cu:1365,1854 | GRP_NC=4 passes; comments cite kVerifyMaxT=8 | AUTO (pass loop takes 12-entry groups in 3 passes; register budget already ~48 at GRP_NC=4) |
| include/strata/core/verify.hpp:395,401,430,453,563 | `brow_[8]`, `last_rows_[8]`, `b_out_[8]`, `last_pos_b_[8]`, `last_tokens_[8]` — **SILENT fixed arrays, no asserts** | EDIT → kWindowMaxT (exactly the five upstream widened; the batch rows overflow class) |
| verify.hpp:574-575,635 (`exec_[9]`, `exec_nr_[9]`, `groups_[9]`) | behind static_asserts 578-581, 636-637 indexed by kVerifyMaxT+1 | AUTO-LOUD: asserts fire at 12 → widen to `[kVerifyMaxT+1]` |
| verify.hpp:577 prof_t_sum_[9] | per-T profile tables | leave [9] (upstream kept 9 then moved to kVerifyLongT; wider windows unprofiled) — EDIT optional |
| src/core/verify.cpp:816 | window gate `max_t > kVerifyMaxT ‖ max_t > cpu::MAXT` | EDIT: drop the cpu::MAXT conjunct (CPU experts take wider windows as ≤8-token jobs, upstream); max_t bound follows kVerifyMaxT |
| verify.cpp:896, 3054, 3313-3322, 3899, 4385, 4466 | nf flags, ple_rows stacks, d_spec_, pl_ple_rows_ — all sized by kVerifyMaxT | AUTO (mapped/malloc/stack scale: ple 12×16=192 words, trivial) |
| verify.cpp:3862-3864 | handoff default kVerifyMaxT rows | EDIT → kVerifyNarrowT default (upstream; wide only via set_handoff_rows) |
| src/core/mtp.cpp:319 | drafter gate `max_t > kVerifyMaxT` | P1: EDIT → kVerifyNarrowT (upstream: draft kernels narrow). P2: → kVerifyMaxT |
| include/strata/core/mtp.hpp:212-300 | graph arrays indexed by kVerifyMaxT+1, behind static_asserts | AUTO-LOUD (widen by the asserts; sizes trivial: +4 entries × small structs) |
| src/program/generate.cpp:3302,3311 | `o.spec = min(o.spec+2, kVerifyMaxT)` (suffix/lookup auto-grow) | P1 EDIT: cap at kVerifyNarrowT + upstream's `--spec > 8` refusal. P2: raise |
| generate.cpp:5430-5435,5516,5625-5628 | slots_in_windows cap (batch groups ≤ kVerifyMaxT rows) | AUTO semantics, EDIT messages; the batch slot math follows the constant |
| generate.cpp:5670 | `mtp_t = pipeline_windows>=2 ? kVerifyMaxT : o.spec` | EDIT → kVerifyNarrowT (upstream) |
| generate.cpp:9131,9200-9202 | kHistSlots, hrows (hand-off rows) | EDIT (upstream form: narrow default, wide only when a >8 batch forms) |
| include/strata/kernels/cpu/expert.hpp:145-147 + pool.hpp | `cpu::MAXT = 8` + `static_assert(MAXT == kVerifyMaxT)` (fires at 12); stack arrays ff[MAXT][FF] | EDIT: DECOUPLE (upstream keeps CPU jobs at 8; a 9..12-row group = two jobs over the same expert — expert_source job split) |
| src/core/expert_source.cpp:4547-4550 | kMaxWindowEntries=128 + assert `cpu::MAXT*10 ≤ 128` | at 12: **fits (120 ≤ 128), no change**; the nt≥MAXT guard (5017-5031, currently refuses) becomes upstream's job-split; act/nact/jobs resizes (4593-4614) → kWindowMaxT |
| src/core/remote_experts.cpp:21,271; peer_experts.cpp:29; remote_expert_opt.cu:14 | CAP = cpu::MAXT·10, n_tok gates | EDIT → kWindowMaxT·10 (=120) + h_sum_ allocs |
| src/kernels/cpu/kq_avx2.cpp:28 | `MAXT = 8` (CPU qk lane) | keep 8 (upstream never widened CPU); groups form ≤8 |
| include/strata/spec/controller.hpp:28,62 | **FORK-OWNED**: `K_MAX=8`, `static_assert(K_MAX == kVerifyMaxT)` — FIRES at 12 | EDIT: relax to `K_MAX ≥ kVerifyNarrowT` (solo policy never picks >7) or extend dense_ratio/distinct_ratio tables to 12; DraftAbort.kSlots = kVerifyMaxT auto-widens (beta[12], kFloats=24 — fine) |
| src/kernels/gdn_parity.cpp:397,436,528 | parity loop T=1..kVerifyMaxT, memcmp bitwise | TEST — auto-runs T=9..12 after the raise (exactly what we want) |
| src/kernels/dm_plan_parity.cpp:734,782 | hardcoded `T <= 8` fold loops | TEST — EDIT loops → kVerifyMaxT |
| src/kernels/resident_plan_parity.cpp:36 | capx = kVerifyMaxT·K | TEST — auto |
| campaign/headport/** (dec_moe.cuh:862 kDmMaxEnt, basalt verify_kernels.hpp:21) | campaign copies with their own `kVerifyMaxT=8` | NO ACTION — campaign/ is not in CMakeLists; independent constants |
| sycl/src/core/mtp.cpp:269 | SYCL drafter gate | NO ACTION — SYCL compiles shared headers without the define → stays 8 (upstream's macro-default design) |

## 2. The MMVQ gate and upstream's answer (the import's core)

- Our wall: `src/kernels/cuda/native_mmvq.cu:830` — `constexpr int MAX_NCOLS = 8;` + static_assert vs kVerifyMaxT. The family is one generic kernel per format, instantiated per exact column count 2..8; each column is BITWISE the single-column call (verified by mmvq_multi parity). Raising the cap is NOT slow-path generic code — upstream adds exact per-N instantiations 9..N behind `#if STRATA_VERIFY_MAX_T > 8`, so an 8-build compiles byte-identical kernels.
- Upstream 9d3c8c36 (files: native_mmvq.cu +83, verify_kernels.cu +62, s2_expert_grouped.cu +92, fused_gr.cu +30, native_bf16.cu +24, verify.cpp +35, generate.cpp +85, verify.hpp +24, expert_source.cpp +29, mtp.cpp +3, remote/peer/shared/native_moe/q8_sm60 small, CMake +29, window_limits.hpp NEW +27, window_limits_test.cpp NEW +64, pipe_probe.hpp NEW +148 — total 678+/122−). 64bf6908 adds 12: exact instantiations 13..16 built only above 12, `MAX_NCOLS` follows the cap, +22/−9.
- The narrow/wide law (window_limits.hpp:10-11): a window of ≤8 rows runs EXACTLY the code it ran at 8 — same instantiations, same buffers. This is the rollback guarantee and the reason upstream could prove bitwise no-regression.
- CUDA default stays 8 upstream because CUDA_MODULE_LOADING=EAGER loads every kernel at start — wide kernels would tax VRAM on every card before a wide window ever forms. Wide is opt-in: `-DSTRATA_WINDOW_MAX_T=12`.

## 3. Port plan and cost

**P1 — batch-window import, STRATA_WINDOW_MAX_T=12 (one session).** New files 2 (window_limits.hpp, window_limits_test.cpp). Edited ~17: verify_kernels.hpp; CMakeLists (option+define+test target); native_mmvq.cu (+il variant); q8_sm60.cuh; native_bf16.cu; verify_kernels.cu (GDN templates + resident_plan NT); s2_expert_grouped.cu (GMAX=12, smem fits); shared_expert.cu; native_moe.cu; fused_gr.hpp/.cu; core/verify.hpp (5 silent arrays + 3 asserted arrays); core/verify.cpp (816 gate, handoff default); cpu/expert.hpp decouple; expert_source.cpp (job split); remote_experts/peer_experts/remote_expert_opt CAPs; mtp.cpp (drafter → narrow); generate.cpp (spec refusal, mtp_t, kHistSlots, hrows); spec/controller.hpp (fork-owned assert); dm_plan_parity.cpp loops.
**Instantiations added at 12:** MMVQ +4 columns × (8 formats multi + wave + il + q8_packed + q6_packed) ≈ 40-60 kernels; bf16 MMVF +4×row-block variants; GDN +1 generic template ×3 entry points; resident_plan 256-thread variant; sm60 passes (no new kernels). ~50-80 new kernel objects, tens of MiB of EAGER-loaded code — measure module-size delta against the expert cache before arming.
**Build time:** not measured on this box (honest): incremental = 8 .cu + ~10 .cpp recompiled + link, expect ~10-25 min (NVCC ~1-2 min/file); cold full build (223 .cu rules in build-fresh/build.ninja) ~30-60 min. Measure with `time cmake --build build-fresh --target strata`. Configure: `cmake -B build-wide -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_WINDOW_MAX_T=12`.
**P2 — solo >8 (second session, gated on P1):** generate.cpp refusal + spec auto-grow caps → kVerifyMaxT; mtp.cpp:319 → kVerifyMaxT; controller.hpp solo policy tables if extended; draft-side grouped experts already covered by the s2 GMAX edit; +gdn_parity/spec parities at T=9..12 + live same-note diff. ~4-6 file edits, small code, unproven economics (§5).

## 4. Lane effect (honest ranges)

Model: geometric acceptance, per-position `a`; expected rows/window at cap T: `E(T) = (1−a^T)/(1−a)`. Today's 3.15 rows/window avg at cap 8 ⇒ a ≈ 0.70. Rings/token ∝ 1/E (per-window ring count fixed: ~48+ per-layer doorbells).

| workload | a | E(8) today | E(12) | E(16) | rings/token Δ at 12 | at 16 |
|---|---|---|---|---|---|---|
| low-accept (code/agentic) | 0.57 | 2.30 | 2.32 | 2.32 | −0.9% | −1.0% |
| typical text (today's avg) | 0.70 | 3.14 | 3.29 | 3.32 | −4.4% | −5.4% |
| mimic-class (high accept) | 0.88 | 5.34 | 6.54 | 7.25 | −18.4% | −26.5% |

- **Solo (P1): zero — windows stay ≤8 rows.** **Solo (P2):** the extra rows cost +4 draft steps per window (chain 11 vs 7). At a≈0.70 the trade is +0.15 tokens/window for +4 drafts — likely a wash or small loss. At a≈0.88: +1.2 tokens/window for +4 drafts — wins only if a draft step (single-row MTP head forward) is under ~25-30% of the window body's per-row cost. State plainly: **solo T>8 pays only on mimic-class acceptance; on today's average it is not the ring-tax fix the task assumes.**
- **Pair/conc (P1, the real target):** batch windows pack parallel slots' rows — fill is guaranteed, acceptance-irrelevant. Cap 12 = ×1.5 slots per launch at the same launch/ring count; cap 16 = ×2 (upstream's motivation: 32 slots in 4×8 groups was the binding cap; 4×16 = 64 slots). Expect the gain concentrated where --batch forms (the conc/serving legs; the 186.5/272.7 legs), scaling with the fraction of step time that is per-window fixed overhead. Honest range: mid-single-digit percent on lanes already launch-bound, ~0 on lanes compute-bound.
- Witness instrumentation already in-tree: STRATA_DM_WAIT_STATS wait-cost accumulators (verify_kernels.hpp:122-138) + upstream's STRATA_P4_TIMING/TRACE pipe_probe.hpp (port optional).

## 5. Quality gates and rollback

1. **g1 kernel parity (bitwise):** `gdn_parity` (auto-runs T=1..12, memcmp), `mmvq_multi_parity` (per-column bitwise), `s2_expert_grouped_parity` (GMAX=12 smem form — the one in-tree deviation), `resident_plan_parity`, `dm_plan_parity` (widen loops first), `spec_verify_parity`, `batch_rows_parity`, `native_multi_parity`.
2. **g2 narrow-invariance:** `window_limits_test` (port upstream's) + kernel-identity diff of a T=8 run wide-binary vs current banked binary (upstream's 3096/3097 method).
3. **g3 exact-match verify at T>8:** same-note accept semantics live — the verifier's per-row argmax compare is unchanged code; prove via interleaved E-C-E-C fixed-seed text diff (the branch-verdict method) on a >8-row batch window.
4. **g4 quality:** `campaign/mmlu-pro-gate.py` spot; `campaign/mimicbench2.py` + `mimic-compare.py` (also the a≈0.88 lane where 12 pays).
5. **g5 lane A/B:** standing recipe batteries vs the 2026-11-09 anchor 195.8/186.5/272.7 (warm probes 2-3, pool=0 fresh boot, banked-state discipline; dc694709/c4b0d2de are the anchors).
6. **Rollback (compile gate — no env needed):** the wide engine is a separate build dir/binary (`-DSTRATA_WINDOW_MAX_T=8` rebuild reproduces the 0.2.7 code paths by construction); the current banked binary stays the instant rollback; runtime, `--spec` stays ≤8 under P1 (upstream's refusal is the guard). No silent fallback paths are added: every wide-only path is behind the macro.

## 6. Boundary

- Upstream RESULTS-NV2.md (their 12/16 measurements) is not in git (untracked on their box) — their numbers are cited only via commit messages; our own A/B is g5.
- s2_expert_grouped GMAX=12 static-smem form is our deviation from upstream's SKIP-pass code path (theirs is 16-forced): parity-gate it, and if it regresses, port the SKIP pass unchanged from 9d3c8c36.
- Build time and EAGER-load VRAM delta are estimates until first configure+build; budget a rehearsal build before the gated session.
- `native_mmvq_il` (fork-added) has no upstream reference: widen its switch by the same pattern, parity via mmvq_il_parity.
