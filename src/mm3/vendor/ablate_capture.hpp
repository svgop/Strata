// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.63).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// Stub of maya's src/core/ablate_capture.hpp (the direction-fitting read-out, MAYA_ABLATE_CAPTURE):
// the capture feature is not ported; mm3_model.cu's include stays satisfied and every call is a no-op.
#pragma once
#include <cstddef>
#include <cuda_runtime.h>
namespace strata { namespace ablcap {
inline bool on() { return false; }
inline void put(const float*, int, int, cudaStream_t) {}   // (maya signature; no-op here)
}}  // namespace strata::ablcap
