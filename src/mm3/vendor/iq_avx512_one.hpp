// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.63).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// include/strata/kernels/cpu/iq_avx512_one.hpp - one-token AVX-512 dot products for the decode's CPU lane, bit-identical
// to ggml-cpu's dot (see src/kernels/cpu/iq_avx512_one.cpp). IQ2_XXS for now.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::mm3k {

/// true when this CPU runs the one-token rows for `ggml_type` (AVX-512 F/BW/VL/VBMI and BMI2, the OS saving the state)
bool iq512_one_supported(int ggml_type) noexcept;
/// out[t][r] = w_r . a[t], rows [r0, r1), each token on its own (the same shape as iq512_rows)
void iq512_one_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                    float* const* out, int r0, int r1);

}  // namespace strata::mm3k
