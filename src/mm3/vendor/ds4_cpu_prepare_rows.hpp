// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.63).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// Startup selection checks, shared with the CPU-only regression. The callbacks keep their own numerical gates.
#pragma once

namespace strata::ds4::gguf::detail {

template <class Own, class One, class Multi>
void prepare_cpu_rows(int type, int fallback_type, Own own, One one, Multi multi) {
    if (!own(type, 1)) (void) one(type);
    // IQ1_S own rows can be disabled by one_gpu_plan after startup calibration.
    if (!own(type, 2) || type == fallback_type) (void) multi(type);
}

}  // namespace strata::ds4::gguf::detail
