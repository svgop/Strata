#pragma once
// The decode miss split between the CPU pool and the PCIe link (PcieAuto): the share of a layer's missed experts the link carries is
// link / (link + k R), R the CPU pool's rate in GB/s of expert bytes, rounded to 0.05 and kept between 0.10 and a ceiling.
// Ported verbatim from upstream Strata 0.2.10.1 (round 0211; include/strata/core/pcie_share_rule.hpp there): the ceiling is 0.95 and k
// falls with R, on a discrete card; STRATA_PCIE_SHARE_MAX and STRATA_PCIE_K_RULE set either by hand.  Pure functions, no GPU.
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace strata {

/// The shipped ceiling and the 0.95 of the rule that follows the pool.
inline constexpr double kPcieCeilingShipped = 0.60;
inline constexpr double kPcieCeilingFollow  = 0.95;

/// The factor k.  1.85 (shipped; fits a strong pool whose reads compete with the uploads for RAM bandwidth) at R >= 30 GB/s of expert bytes,
/// falling linearly to 0.8 at R <= 12 (a compute-bound weak pool: 4 workers settled best at 0.85 where 1.85 gave 0.70).  k_rule false: 1.85.
inline double pcie_pool_k(double r, bool k_rule) {
    if (!k_rule || r >= 30.0) return 1.85;
    if (r <= 12.0) return 0.8;
    return 0.8 + (r - 12.0) / 18.0 * (1.85 - 0.8);
}

/// The share for a link of `link` GB/s against a pool of `r` GB/s, rounded to 0.05, between 0.10 and `ceiling`.
inline double pcie_share_for(double link, double r, double ceiling, bool k_rule) {
    const double raw = link / (link + pcie_pool_k(r, k_rule) * r);
    return std::min(ceiling, std::max(0.10, std::round(raw * 20.0) / 20.0));
}

/// Does PcieAuto also refine a layer split's shares?  `env` is STRATA_PCIE_AUTO_SPLIT (nullptr when unset): an explicit value wins either way.
/// Upstream: on for a discrete CUDA card, off on HIP, off on an integrated GPU and under the 026 master switch.  This fork has no 026 mode;
/// callers pass `v026` false (the split refinement itself stays behind its own switch, as before).
inline bool pcie_auto_split_default(bool hip_build, bool discrete, bool v026, const char* env) {
    if (env != nullptr && env[0] != 0) return std::atoi(env) != 0;
    return !hip_build && discrete && !v026;
}

/// The pool-rate factor on?  STRATA_PCIE_K_RULE (nullptr unset) wins; unset: on for a discrete card, off for an integrated GPU.
inline bool pcie_k_rule_default(bool discrete, bool v026, const char* env) {
    if (env != nullptr && env[0] != 0) return std::atoi(env) != 0;
    return discrete && !v026;
}

/// The ceiling.  STRATA_PCIE_SHARE_MAX (nullptr unset) wins (clamped 0.10..1.0); unset: 0.95 on a discrete card, 0.60 otherwise.
inline double pcie_share_ceiling(bool discrete, bool v026, const char* env) {
    if (env != nullptr && env[0] != 0) return std::min(1.0, std::max(0.10, std::atof(env)));
    return discrete && !v026 ? kPcieCeilingFollow : kPcieCeilingShipped;
}

}  // namespace strata
