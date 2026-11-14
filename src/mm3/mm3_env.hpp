// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.63).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/mm3/mm3_env.hpp - maya-mm3's environment settings: MAYA_<name> first (the other engines' prefix), then <name>
// itself (the MM3_* names #477 used, kept as aliases).
#pragma once

#include <cstdlib>
#include <string>

namespace mm3 {

inline const char* env(const char* name) {
    const std::string m = std::string("STRATA_") + name;
    if (const char* v = std::getenv(m.c_str())) return v;
    return std::getenv(name);
}

}  // namespace mm3
