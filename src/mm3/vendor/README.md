# src/mm3/vendor/ — the mm3 family's private kernel substrate

Provenance: Project Maya v1.1.63 (MIT, third_party/project-maya/NOTICE.md), imported 2026-11-14.
Namespace cut: `strata::mm3k` (this whole directory) — **no symbol is shared with `strata_kernels`
or the dsv41 family (`strata::dsv41k`)**. This is the family-isolation law, the same one dsv41's
vendor/ established: both trees vendored the same upstream files and the forks have genuinely
diverged (ds4_iq: maya carries `act/tu/pexp` our dsv41 copy dropped; exl3_cpu: 1,135 diff lines).
Linking the wrong fork still runs but produces different bits — the two directories must never
merge, share a namespace, or share a build target.

Built into `strata_mm3_kernels` (MAYA_DS4_IQ_ONLY=1) and the `strata-mm3` executable only.
Per-file AVX512+BMI2/AVX2 compile flags mirror maya's own build config (CMakeLists.txt).

Contents: maya's ds4_iq fork (GGUF low-bit experts with swigluoai), ds4_lt (hipBLASLt picks),
ds4_ablate, ds4 numerics-adjacent helpers, the exl3_cpu/exl3_ref CPU lane, the native-expert
lattice (expert, expert_layout, iq_avx2/512/512_one, q2_avx2, glm_lane_row4 pair, native_expert),
and maya's spec pair (draft_policy with kMaxT=16 + trust(), suffix_drafter).
`ablate_capture.hpp` is a stub (the capture feature is not ported).
