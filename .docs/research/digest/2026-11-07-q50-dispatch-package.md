---
type: digest
id: q50-dispatch-package-2026-11-07
created: 2026-11-07
status: patch-delivered
topic: The Q5_0 down dispatch package — the decode-once multi form (`kSplitDown<6>`) adopted behind a host-known per-group entry bound (`max_nc`) and the `STRATA_Q50_MULTI_MAX_NC` ceiling (default 2): the law-17 serial wins (solo +4.8 / pair +2.9) without the 6-column conc loss (-11.2, law 13)
scope: patch built in the scratch worktree /tmp/q50-dispatch/wt (fork cc72b883, never the live tree; patch: /tmp/q50-dispatch/q50-dispatch.patch, 5 files, +67/-10, sha256 bb05d44cba2d58fc); compile-proven in /tmp/q50-dispatch/build (Release, CUDA arch 120, nvcc 13.4, the fork's .venv cmake/ninja); mechanism evidence from static ptxas -v only (no GPU run — the card serves the operator). Siblings: 2026-11-06-ud-dense-conv-package.md (house style), the in-tree law-17 note at src/kernels/cuda/iq_kernels.cu:1127-1134.
---

# Q5_0 down dispatch package (2026-11-07)

Verdict: **the dispatch is built, linked, and delivered in `/tmp/q50-dispatch` (patch: `q50-dispatch.patch`, 5 files, +67/-10).** `Split<6>` (the Q5_0 decode-once down traits, already in-tree bitwise-clean) is now reachable through a down-only trait `kSplitDown<6>` and a runtime gate in `launch_down`: the multi form (`native_down_multi_kernel<6>`) runs **only when the caller hands a host-known bound on any one group's entry count and the bound is within `STRATA_Q50_MULTI_MAX_NC` (default 2)**. The bound exists at every call site — the window's rows bound every group's entries (a group is one expert, at most one entry per row) — so the verify path (`verify.cpp`'s `grouped` lambda, one wiring point over all three launches) passes `n = te - tb` and the dispatch takes the measured serial windows (solo +4.8%, pair +2.9%) while the 6-column conc window keeps the per-entry kernel. `kSplit<6>` **stays false**: the gate/up, mmvq and `gu_split` sides never see the trait, and for GSQ (no type-6 downs) the whole branch is dead code. Both forms are bitwise the same outputs, so the gate moves only the clock, never a bit.

## 1. The three design questions, answered from the code

- **What the multi kernel loops over.** `native_down_multi_kernel` (`iq_kernels.cu:2022`, TD 6 falls to the generic staged path `:2123-2176`): `for (g = blockIdx.y; g < ng; g += gridDim.y)` with `ng = *n_groups` **read on the device**; per group the entries `[grp_start[g], grp_start[g+1])` — **also device-side** — are taken `GRP_NC = 4` at a time (`:1856`; "a group has at most one entry per token of the window (kVerifyMaxT = 8)... 4 takes such windows in one pass"). Per pass the whole 256-thread block cooperatively copies the next n ≤ 4 entries' q8_1 blocks (n × 180 words) into `s_hq_buf` (2880 B static shared) between **two `__syncthreads()`**, then each of the 8 warps runs `row_dot_multi<TD, NC>` (`:1435`): the weight (`Split<TD>::load`) decoded **once per pass**, applied to every staged entry. Q5_0's weight row is nb 20 blocks × 22 B = 440 B (`Fmt<6>`: qk 32, ipb QI5_0/VDR_Q5_0); the per-entry kernel (`native_down_kernel`, `:1988`) re-reads and re-decodes it for every entry, zero shared, zero barriers.
- **Where the per-group expert count lives.** Device-only: `grp_start[]`/`n_groups` are device arrays the plans publish (`verify.cpp` mapped plans; `peer_experts.cpp:279` / `remote_experts.cpp:315` upload host-built copies). Nothing inside `launch_down` (`:3935`) sees a count — its caller passes only the grid `gd = ((n_embd + d_rows - 1)/d_rows, gy)` (`:5006-5007`; TD 6: d_rows 8 → 320 × gy, **identical for both forms**, so grid shape is not the difference).
- **The host-known proxy exists.** Trace: `native_expert_grouped` (`:4823`) ← `verify.cpp:2215` inside the `grouped` lambda (`:2209`), where `post()` has `n = te_[grp] - tb_[grp]` (`:2192`) and passes `cap = n*K` as both capacities. A group is one expert and holds at most one entry per window row, so **`n` bounds every group's NC exactly** — at solo n=1, pair n=2, the law-17 conc case n=6. The same bound covers all three `grouped` call sites (`:2238` ar, `:2266` VRAM, `:2284` PCIe — the PCIe share is the same window's rows). `peer_experts.cpp:299` knows `rows` (and the per-group sizes it builds at `:256-276`); `remote_experts.cpp:341` knows `dst_.size()`. Mechanism (a) — the host-threshold branch — is what the code supports; no in-kernel fallback is needed.

## 2. Why multi loses at conc: the mechanism (static evidence)

ptxas -v, sm_120, nvcc 13.4, Release (full log: `/tmp/q50-dispatch/ptxas_iq.log`):

| TD | form | regs | barriers | smem | stack |
|---|---|---|---|---|---|
| **6 (Q5_0)** | per-entry | **40** | 0 | 0 | 0 |
| **6 (Q5_0)** | multi | **64** | 1 | 2880 B | 16 B |
| 8 (Q8_0) | multi | 48 | 1 | 2880 B | 16 B |
| 13 (Q5_K) | multi | 64 | 1 | 2880 B | 16 B |
| 20 / 23 / 40 / 42 | multi | 48 | 1 | 2880 B | 24-64 B |
| 144 / 146 | multi | 40 / 48 | 1 | 2880 B | 8 B |

- **Occupancy, the primary mechanism.** At 256 threads/block against the 64K-register file: 40 regs → **6 blocks/SM** (1536 threads, 75% warp slots); 64 regs → **4 blocks/SM** (1024 threads, 50%). The Q5_0 multi form gives up **one third of the resident warps** — `row_dot_multi<6, GRP_NC>` keeps the 4-int + 1-float `W` alive across four unrolled applies plus four accumulators, the heaviest register budget of any *default-adopted* down multi. Shared memory is not the limiter (4 × 2880 B = 11.5 KB). The 16 B stack is `off[GRP_NC]`'s dynamic indexing in the n ≥ 3 fallthrough (`:2152-2154`).
- **Barrier serialization, the amplifier.** Per GRP_NC pass the block pays two full-block `__syncthreads()` around the staging copy. At NC 6 that is 2 passes → **4 barriers per group per block**, each draining all 8 warps to the slowest — with a third fewer warps resident to fill the gaps. At NC 1-2 it is one pass with a 180/360-word copy: cheap.
- **Why the flip lands at 6, not 2.** The decode-once saving is a re-read of the *same 440 B row* per entry — after the first entry those are L1 hits plus ~15 int ops per k-step. At conc many groups are live at once (up to n×K per window) and every block streams a different expert blob, so latency hiding depends on resident warps: the occupancy loss and the barrier drains outweigh savings that cache hits made cheap. At NC 1-2 the pass count is 1, the copy is tiny, and the decode-once plus the once-per-block hq staging (720 B read from shared by 8 warps instead of 8 per-warp L1 streams) is what the solo +4.8% banked on. Corroborating precedent in-tree: the only other 64-reg multi (Q5_K, TD 13) is **not** default-adopted either — the whole K-quant multi sits behind `STRATA_KQ_SPLIT=1` (`:816-819`), the same "gate the heavy form until measured" discipline this package gives Q5_0, with a finer key.

## 3. The dispatch

- **`kSplit<6> stays false.** `STRATA_GU_FMTS` includes 6, so flipping the trait would have re-routed a hypothetical Q5_0 *gate/up* to `native_gu_multi_kernel<6>`, `launch_gu_swiglu<6>`, `launch_mmvq<6>` and `gu_split(6)`'s 16-row selection (`:3104-3108`) — none of it measured. Instead the down side gets its own gate, one specialization away from the trait it mirrors:
  ```cpp
  template<int TD> inline constexpr bool kSplitDown = kSplit<TD>;
  template<> inline constexpr bool kSplitDown<6> = true;   // iq_kernels.cu, after Split<6>
  ```
- **The gate (option (a), the host branch).** `launch_down` (`:3935` in cc72b883) gains `int64_t max_nc = 0` and, only under `if constexpr (TD == 6)`: per-entry kernel when `max_nc <= 0 || max_nc > g_q50_multi_max_nc`, otherwise the multi form (SUB_DOWN true/false per `g_no_sub16_gu`, as the other formats). Branch order preserved: `g_old_kernels || split_off<TD>()` still forces per-entry first, so `STRATA_OLD_IQ_MMVQ=1` and the parity harness's old-form runs are untouched. Every other TD dispatches exactly as before — `kSplitDown ≡ kSplit` for all of them.
- **Defaults and switches.** `STRATA_Q50_MULTI_MAX_NC`, read once at startup beside `g_old_kernels` (`:3119`), **default 2** = the measured crossover (only solo/pair/conc are measured; 2 takes both measured wins and refuses the measured loss — law 13). `=0` is the kill switch (never multi). `max_nc <= 0` (unknown) → per-entry: today's bits, so **no caller regresses**; the wins engage only where a caller affirms a small bound. `iq_set_q50_multi_max_nc(int)` (public, beside `iq_set_old_kernels`) lets the parity harness and a bench leg reach the multi form without an env dance.
- **GSQ statement (required).** GSQ's pack carries no type-6 downs: `launch_down<6>`'s TD 6 case is dead code for GSQ, and every d_type GSQ does carry dispatches through a `kSplitDown` value identical to the `kSplit` it tested against — the fork's GSQ behavior is bit-identical, full stop.
- **Wiring map.** `native_expert_grouped` gains the defaulted `max_nc` (header `iq_kernels.hpp:78`; the defaulted param keeps all 12-arg call sites compiling) and threads it to `launch_down` in the `STRATA_DOWN` switch (`:5035`). `verify.cpp:2215` passes `n` — the one wiring point the measured path needs. `native_expert_bench.cpp:122` passes its own `T` (groups of exactly T entries — an exact bound, so the bench IS the per-NC sweep tool). `peer_experts.cpp` / `remote_experts.cpp` deliberately unwired (default 0 → per-entry): named boundary, not a silent substitution — their bound is one line each when the operator wants their tier on the sweep.
- **Graph capture.** The branch is host-side at launch: a captured graph bakes whichever kernel the capture-time `max_nc` chose. The engine's window stages are per-T instantiations (capture-T == replay-T); a stale graph degrades to today's per-entry kernel — never to the unmeasured conc form.

## 4. What the package changes

- `src/kernels/cuda/iq_kernels.cu` — `kSplitDown` trait after `Split<6>` (`kSplit<6>` untouched); `g_q50_multi_max_nc` env global + comment; `launch_down` signature + TD 6 gate; `iq_set_q50_multi_max_nc`; `native_expert_grouped` signature + `STRATA_DOWN` macro passes `max_nc`. The `launch_down<13>` v2k call (`:4882`) compiles on the default (TD 13 unaffected).
- `include/strata/kernels/iq_kernels.hpp` — `native_expert_grouped(..., int64_t grid_groups = 0, int64_t max_nc = 0)` with the bound's contract; `iq_set_q50_multi_max_nc`.
- `src/core/verify.cpp` — the `grouped` lambda passes `n` (comment names the law-17 split).
- `src/kernels/iq_multi_parity.cpp` — Q5_0 added to `name_of`; **`block_values(6) = 32`** (without it the fp16-scale patch stride would have been 176 B, not the 22 B block — invalid random blocks); `Grouped::run` passes `max_nc = cap_ent` (a true bound) on new-form runs; `main` raises the ceiling to 99 before the grouped legs and **adds dt 6 to the 2560×640 model-shape list**, so the multi<6> kernel is checked bitwise against the old per-entry kernel at every group size 1..11 (partial passes, a 0-entry group, unused grid rows, scattered destinations).
- `src/kernels/native_expert_bench.cpp` — `run` passes `max_nc = T`.

## 5. Compile proof

Fresh `/tmp/q50-dispatch/build` — Ninja, the fork's `.venv` cmake 4.4.3 + ninja 1.13.2, Release, `CMAKE_CUDA_ARCHITECTURES=120`, `STRATA_ENABLE_CUDA=ON`, nvcc 13.4 (`/usr/local/cuda-13.4`), gcc `/usr/sbin/c++`, configured from the worktree at cc72b883. All link, zero errors:

```
iq_multi_parity         linked (relinked after the ptxas-capture object rebuild)
native_grouped_parity   linked
strata                  linked (91,879,376 B; strata_core + strata_engine + kernels)
strata_engine           linked (compiles peer_experts.cpp + remote_experts.cpp: the default-arg call sites)
native_expert_bench     linked
```

The only diagnostics are pre-existing `-Wunused-variable` warnings in untouched lines (`layer.cpp:1154/1511`, `verify.cpp:3403`, `generate.cpp:13179`). No GPU execution — binaries built, not run.

## 6. Parity and boot checks for the operator (GPU legs, quiet window)

1. **Bitwise, the new kernel**: `build/iq_multi_parity` (ctest runs it with `STRATA_KQ_SPLIT=1`, `CMakeLists.txt:1037`). Expect 8 new `Q5_0` lines (`IQ*/Q2_0 × Q5_0 2560 x 640`) reading "bitwise equal to the old kernels" and `iq_multi_parity: 0 failures`. This is the multi<6> coverage the harness never had (its dt list skipped 6).
2. **Bitwise, the dispatched default**: covered by construction — `max_nc` unknown/large selects `native_down_kernel<6>`, the exact template that runs today; `native_grouped_parity` runs unchanged (its calls default `max_nc = 0`).
3. **Dispatch engagement, no new boot line needed**: a profile (nsys/nvprof kernels view) on a UD-IQ4_XS boot shows `native_down_multi_kernelILi6ELb1E` on 1-2-row windows and `native_down_kernelILi6E` on the 6-column window. GSQ shows no type-6 down kernel at all, as before.

## 7. The threshold's sweep plan (the law-17 legs)

- **Kernel-only curve (cheapest first)**: `build/native_expert_bench <layer> <fmt> <G> <T> <mref> <mode>` — T is the exact NC; sweep `STRATA_Q50_MULTI_MAX_NC` ∈ {0, 1, 2, 3, 4, 6, 8} at each T 1..8. Reads the per-NC crossover directly; the 0 rows are the control (today's bits).
- **End-to-end**: the law-17 3+3 interleave on the UD-IQ4_XS pack, ceiling ∈ {0 (control), 2 (default), 3, 4}: solo and pair must hold ≈ +4.8 / +2.9 at 2; conc must return to ≈ 0 vs control (it is −11.2 only when the multi form is wrongly adopted there). Ceiling 4 is the first unmeasured point between 2 and 6 (4-row windows feed groups of ≤ 4); promote the default to 3 only if the sweep shows the 3-row stream pair clean at conc; **ceiling ≥ 6 never without a fresh conc measurement**.
- **Peer/remote tiers** (if wanted on the sweep): pass the same bound at `peer_experts.cpp:299` (the max group size the loop already counts) and `remote_experts.cpp:341` (window rows) — one argument each, then re-run the interleave with the pool paths armed.

## 8. Boundaries

- The mechanism analysis is static (ptxas -v, code reading). The occupancy arithmetic (64K regs/SM, 256-thread blocks) is the sm_120/sm_89 shared shape; the solo +4.8 / pair +2.9 / conc −11.2 numbers are the operator's measurements, not reproduced here.
- The ceiling's default (2) rests on three measured points (1, 2, 6). The 3/4/5 region is unmeasured — the sweep, not this package, sets it.
- `STRATA_Q50_MULTI_MAX_NC` is read once at startup like `g_old_kernels`; `iq_set_q50_multi_max_nc` is the runtime write for harnesses. Captured graphs keep the kernel they captured (see §3).
- The 16 B stack frame on `native_down_multi_kernel<6>` (`off[GRP_NC]`) is a candidate micro-optimization if the sweep ever wants the multi form at high NC; not touched — law 13 forbids adopting that regime without measurement.
