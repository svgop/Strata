# GR_V3_PF (BW-DENSE) port package — upstream v0.2.10/v0.2.11-3 → fork

Date: 2026-11-06. Source: `/home/svg/LLM/strata` @ 9f7d977d (v0.2.11-3 line). Target base: fork @ 9728436a.
Scratch worktree: `/tmp/grv3-port` (worktree of this repo at 9728436a). The live tree was not touched except this file.
Patch package: `/tmp/grv3-port/grv3-port-package.patch` (685 insertions, 13 deletions, 6 files). Changed files, full paths:

- `/tmp/grv3-port/include/strata/core/bw_dense_defaults.hpp` (new, adapted)
- `/tmp/grv3-port/include/strata/kernels/fused_gr.hpp` (fused_gr_v3_pf / fused_gr_v3_print_resources decls)
- `/tmp/grv3-port/src/kernels/cuda/fused_gr.cu` (the mechanism: kernels + mask plumbing + dispatch + v3 default-on)
- `/tmp/grv3-port/src/kernels/gr_v3_screen.cpp` (new, verbatim upstream HEAD, masks {0, 5, 133, 261})
- `/tmp/grv3-port/src/kernels/gr_parity.cpp` (the STRATA_GR_V3=0 pin in main, ported from upstream)
- `/tmp/grv3-port/CMakeLists.txt` (gr_v3_screen target beside fused_gr_bench)

## What landed

Mechanism ported onto OUR structs (the fork never took upstream's w_inject-era struct evolution — the fork's
FusedGrArgs already carries every field the mechanism touches: R, R_out, apply, bo_prev, inj_prev, w_norm,
w_inject; GrMulti unchanged). Per upstream commit:

- 9abe0d6c — `ldg_v4` (volatile `ld.global.nc.v4.u32` asm, anti-sink), the `PF` template bits on
  `gr_down_v3_kernel` / `gr_up_v3_kernel` (bit 0 batched staging / partial+row loads, bit 1 weight prefetch),
  `g_v3_pf_force` + `fused_gr_v3_pf()`, the PF dispatch templates, `warp_sum8`, gr_v3_screen.
- 3ce5193e — `v3_pf_mask()` in its final form: env read once per process, explicit value wins on any card in both
  directions, unset → per-device default via `strata::gr_v3_pf_default(ma, mi)`.
- 4ab81c79 + 0e8e4d4c — the `RB` template bit on `gr_down_v3_kernel` only (register-blocked dot phase: chunks →
  tokens → R2V rows, activations read from shared once per token for all rows, same chunk dots in the same
  ascending order; epilogue `warp_sum8` for T 5-8). Up kernel blocking is NOT ported — upstream dropped it
  measured-slower (up 9.2 → 10.6 µs T1, 18.8 → 23.9 µs T8). Bit 7 = 2 rows a warp, gated `n_tok >= 5` in the
  dispatch (0e8e4d4c's gate); bit 8 = 4 rows, any window.
- 08b20b10 + 27a3e056 — `bw_dense_defaults.hpp` (adapted, below); default mask 133 = bits 0+2+7 on cc-12.0.
- ee108afe — `set_attr_once_per_device` (`<mutex>`, one flag per device) for `gr_down_pf_launch`,
  `gr_down_rb_launch`; smem opt-in is per DEVICE, not per process.
- The v3 read itself becomes default-on with upstream's 0.2.7 rule (`gr_v3_default_on`: discrete CUDA cc-12.0,
  > 13 GiB; adapted rule `gr_v3_read_default` in the defaults header — no 026 term). STRATA_GR_V3=0/1 keeps
  forcing off/on (the old arm's env name survives with upstream's semantics; the fork's former pure opt-in is
  superseded — upstream's mask system is built on the default-on v3 read, and the +16% conc stake requires the
  v3 read active). REQ8 (s_down/s_up) and Q8 packs decline the v3 branch exactly as before.

## Fork adaptations (deviations, all named)

1. **bw_dense_defaults.hpp**: upstream's carries `small_unr_default` (STRATA_SMALL_UNR: bf16 router read + GDN
   a/b constant-trip loops, wired into native_bf16.cu / verify_kernels.cu) and gates everything on the 026 master
   mode (`defaults_026.hpp`, STRATA_DEFAULTS=026). The fork has neither the 026 mode nor that wiring, both outside
   this port's scope → the adapted header carries `gr_v3_pf_default` (returns 133 on sm120_default, else 0) and
   `gr_v3_read_default` (upstream's gr_v3_default rule without the 026 term; placed here instead of sm120_series.hpp
   to keep the patch in one new header). STRATA_SMALL_UNR is NOT ported — say so if it is wanted later.
2. **gr_down_r2_launch** (mask bits 4-5, plain rows-a-warp 2/4, T ≤ 4) sets its own per-device smem opt-in on
   first use. Upstream leaves that to its boot block, whose STRATA_GR_R2 section the fork never imported; without
   this, bits 4-5 at T3/T4 (60/80 KB dynamic smem) would fail the launch. No arithmetic or launch-geometry change.
3. **Exact-T 3/4-token plain instantiations** get their `MaxDynamicSharedMemorySize` (60/80 KB) in the fork's v3
   boot block. Upstream never sets these either — its default-on v3 read would fail a 3/4-token window launch
   ("invalid argument"); with v3 becoming a default here the windows must launch. Attributes only.
4. **gr_parity** pins STRATA_GR_V3=0 when unset (verbatim upstream): the test compares the plain default read bit
   for bit, and the v3 read is now a default. Its `--v3`-style tolerance leg still runs with STRATA_GR_V3=1.
5. NOT ported from upstream's file, deliberately: `fused_gr_read_multi_slice` + the slice-walking wrapper (the
   fork's single-window `fused_gr_read_multi` stands; the mask dispatch lives in its v3 branch), `gr_r2()` /
   STRATA_GR_R2 (superseded by the mask bits 4/7/8, and the fork never had it), upstream's `dispatch::kernel`
   BW-MEASURE counters, `q32_no_cnt`. None is referenced by the ported mechanism.

No kernel-name collisions: the fork's older GR_V3 arm IS this arm (same `gr_down_v3_kernel` / `gr_up_v3_kernel`
from the F-369 import); the port extends its templates rather than adding a second set. Instantiation census in
the built fatbin confirms: plain `<1,K,true,1,0,false>` K1-8, PF `<1,K,true,1,{1,2,3},false>` K1-8,
r2 `<1,K,true,{2,4},0,false>` K1-4, RB `<1,K,true,{2,4},0,true>` K1-8, up `<1,K,true,{0..3}>` (plain 7/8 absent —
the fork's plain chain never launched exact 7/8; the PF dispatch does, guarded by kFusedGrMaxT). The static-assert
wall is untouched: every window-indexed array still ties to `MAX_T` (default `kFusedGrMaxT` = 8 = `kVerifyMaxT`).

## Env semantics (exact)

- `STRATA_GR_V3_PF` — read once per process. Any set value is the mask (bitwise; =0 restores the plain v3
  kernels; bits: 0 down batched staging, 1 down weight prefetch, 2 up batched loads, 3 up weight prefetch,
  4/5 down rows-a-warp 2/4 (T ≤ 4), 7 down RB 2 rows (windows ≥ 5), 8 down RB 4 rows). Unset → mask 133 on a
  cc-12.0 card unless `STRATA_SM120_PERF=0`, else 0. Per-device cache, 64 devices.
- `STRATA_GR_V3` — 0/1 forces the two-kernel v3 read off/on on any card; unset = on for discrete CUDA cc-12.0
  > 13 GiB. NOTE: default-on changes the read's summation order (answers shift ~1e-7; a greedy tie can flip).
  `STRATA_GR_V3=0` restores the fork's exact pre-port behavior.
- `STRATA_SM120_PERF=0` — turns both defaults off (the set rule in sm120_series.hpp).
- `fused_gr_v3_pf(int)` — bench/test setter for the mask (-1 = env), used by gr_v3_screen.

## Proof

- Build (fresh `/tmp/grv3-port/build`, `/home/svg/LLM/strata-custom-2/.venv/bin/cmake -G Ninja`, Release,
  `CMAKE_CUDA_ARCHITECTURES=120`, `STRATA_ENABLE_CUDA=ON`, nvcc 13.4):
  `[60/60] Linking CXX executable gr_v3_screen`; `[107/107] Linking CXX executable strata` (exit 0);
  gr_parity + fused_gr_bench relink clean. Only warning: pre-existing generate.cpp:13083 unused variable.
- Fidelity diffs of the ported kernel bodies against upstream HEAD show exactly two deltas, both intended:
  upstream's plain-path `warp_sum8` fast writes (T5-8) are absent on the fork's plain path — mask 0 must stay
  the fork's current class; everything else (PF bit 0/1 bodies, RB block, warp_sum8 helper) is upstream-verbatim.

## Verify before banking (operator, GPU serialized)

1. `build/gr_v3_screen` (defaults: 400 iters, T 1-8, 16 sets, 5 rounds): every mask {5, 133, 261} must print zero
   "differs from the plain v3 read" lines across T 1-8, apply/inject 0/1. This re-proves the bitwise claims on the
   fork's plain path (which differs from upstream's plain path only in the T5-8 partial-write reduction: the fork
   keeps per-token `warp_sum`, lane 0 writes; RB uses `warp_sum8` — same butterfly additions in the same order,
   upstream-verified, but the fork owes itself the 0-differing line in-tree).
2. `gr_v3_screen --resources` for the register/smem table of the variants.
3. Engine lanes, mask on (133 default) vs `STRATA_GR_V3_PF=0` (and `STRATA_GR_V3=0` as the pre-port control):
   solo (T1), pair (T2), 6-conc (T6). The stake predicts conc ~272 → ~315 t/s with solo/pair unchanged (bit 7 is
   gated n_tok ≥ 5; bits 0+2 help little at T1-2). Down-kernel µs expectation at T5-8: 16-19 → 13-15 (gpu_stamp
   legs of gr_v3_screen show down/up split per mask).
4. gr_parity still passes (`--selftest`) with the pin (bitwise default-read checks unaffected).
5. Watch the first T3/T4 window the engine ever runs with v3 on — my boot-block opt-in makes it launch, but it is
   a path upstream never exercises, so it is unmeasured territory (correctness comes from the kernels' own
   EXACT_T contract, not from measurement).

Boundary: no GPU work was run in this port (the operator's benches own the card); every claim above that needs
the GPU carries its executable probe.
