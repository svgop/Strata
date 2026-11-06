// strata/core/bw_dense_defaults.hpp - defaults of the bitwise load-ordering variants from upstream's Blackwell Qwen
// speed work (BW-DENSE), ported into this fork (2026-11-06, from v0.2.10/v0.2.11-3: 9abe0d6c, 3ce5193e, 4ab81c79,
// 0e8e4d4c, 08b20b10, 27a3e056, ee108afe).
//
// Each is bitwise equal to the kernel it replaces (upstream's screens and engine A/Bs: phaseA-tests/blackwell/decisions/
// E15.md and E21-small-kernel-loads.md), measured on an RTX PRO 5000 Blackwell (compute capability 12.0), so they are on
// there and nowhere else (unmeasured on other cards).  STRATA_SM120_PERF=0 turns them off like the rest of the 12.0 set
// (sm120_series.hpp); an explicit STRATA_GR_V3_PF wins on any card in both directions (=0 restores the plain kernels).
// Fork adaptation: upstream's copy also carries the small_unr_default (STRATA_SMALL_UNR) rule and gates everything on the
// 026 master mode (defaults_026.hpp qwen_defaults_026(), STRATA_DEFAULTS=026); this fork has neither the 026 mode nor the
// STRATA_SMALL_UNR wiring (native_bf16.cu / verify_kernels.cu were not in the port's scope), so the header carries the
// GR_V3_PF default and the v3-read default rule only, without the 026 term.
#pragma once

#include "strata/core/sm120_series.hpp"

namespace strata {

/// STRATA_GR_V3_PF when unset (gr_down_v3 / gr_up_v3 load-ordering variants): bit 0 (gr_down batched staging loads),
/// bit 2 (gr_up batched partial/row loads) and bit 7 (gr_down: windows of 5+ rows take 2 rows a warp with the
/// activations read from shared memory once for both rows; bitwise the same dots, upstream DF2-MORE: down 16.1 / 16.2 /
/// 17.6 / 19.5 -> 13.3 / 14.1 / 14.7 / 15.6 us at T 5-8).  Mask 133 = bits 0 + 2 + 7.
inline int gr_v3_pf_default(int major, int minor) { return sm120_default(major, minor) ? 133 : 0; }

/// STRATA_GR_V3 (the stream-split two-kernel hyper-connection read, another summation order than the default read) when
/// the user set nothing: ON for a discrete compute capability 12.0 card (STRATA_SM120_PERF=0 turns the set off) with more
/// than 13 GiB, CUDA builds only.  Upstream's rule (sm120_series.hpp gr_v3_default, without the 026 term): measured on
/// the RTX PRO 5000 72 GB with every expert resident: code decode +1.2% decode clean; NOT a default on sm_86 (measured
/// -3.0% / -5.9% on 2x RTX A4000, +1.0% / -3.4% on an RTX 3060) and not in HIP builds (a gfx12 card reports 12.0 too).
inline bool gr_v3_read_default(bool hip, int major, int minor, bool integrated, double total_bytes) {
    return !hip && !integrated && sm120_default(major, minor) && total_bytes / 1073741824.0 > 13.0;
}

}  // namespace strata
