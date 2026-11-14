# project-maya (vendored portions)

Source: https://github.com/mw00/project-maya at v1.1.63 (commit 6651b5b), MIT (LICENSE here).
Read-only reference clone: `data/references/project-maya` (device estate). Imported 2026-11-14.

This directory carries the license and this notice for code vendored into the strata-custom-2 tree:

- `src/mm3/` — the MiniMax-M3.1-Flash engine family (mm3_main/model/kernels/exl3/gguf/gv/dspark/
  vision/tp/cpu/host/pack/geo/sample/slots/lane_plan/cands/os/env), ported from `src/mm3/`.
  Include paths re-pointed at our substrate (`../dsv41/` numerics+os, `vendor/` for the rest);
  env prefix renamed `MAYA_` → `STRATA_` (choke point mm3_env.hpp; knob suffixes unchanged).
- `src/mm3/vendor/` — the family's private kernel substrate, compiled into `strata_mm3_kernels`
  and the `strata-mm3` executable only; no symbol is shared with `strata_kernels`:
  - from maya `src/ds4/`: ds4_iq.{cuh,cu}, ds4_iq_cpu.cpp, ds4_gguf_cpu.{hpp,cpp}, ds4_lt.{hpp,cu},
    ds4_ablate.hpp, ds4_iq1_codes.hpp, ds4_iq512_kern.inc, ds4_mx_kern.inc, ds4_pdl.cuh,
    ds4_par_exp.h, ds4_cpu_prepare_rows.hpp (built with MAYA_DS4_IQ_ONLY so dsv41's diverged
    ds4_iq stays untouched — the two forks must never share a namespace or a build)
  - from maya `include/strata/kernels/cpu/` + `src/kernels/cpu/`: the native-expert CPU lattice —
    exl3_cpu.{hpp,cpp} + exl3_cpu_avx512.inc, exl3_ref, expert.{hpp,cpp}, expert_layout,
    iq_avx2/512/512_one, q2_avx2, glm_lane_row4 (+x86/avx2), native_expert — all under the
    `strata::mm3k` namespace cut (ours' dsv41 copy diverged under `strata::dsv41k`; the two
    forks must never meet — see src/mm3/vendor/README.md)
  - from maya `include/strata/spec/` + `src/spec/`: draft_policy.{hpp,cpp} + suffix_drafter.{hpp,cpp}
    (maya's DraftPolicy has kMaxT=16, context bands and trust(), which mm3_main needs and our
    spec's kMaxT=8 policy lacks; vendoring keeps `strata-spec` untouched)
  - `ablate_capture.hpp` is a stub (the direction-fitting read-out is not ported; calls are no-ops)
- `tools/mm3/` (later phases) — pack tooling from maya `tools/mm3/`.

The ggml/llama.cpp-derived layouts inside the ds4_iq kernels remain covered by
third_party/ggml/LICENSE. Metal sources (`src/metal/mm3_*`) are not ported.
