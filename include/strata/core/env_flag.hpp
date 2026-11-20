// include/strata/core/env_flag.hpp - one authoritative home for STRATA_* environment truthiness.
//
// generate.cpp historically carried ~40 bespoke one-line env-helper lambdas, each re-implementing one of
// several incompatible conventions, so a boot with STRATA_FOO="" or STRATA_FOO=0 armed some features and
// disarmed others depending on which lambda read them (the F-419 "boot-state lottery", findings C5/D4).
// This header names each convention explicitly. The *_value() forms are pure predicates (testable with no
// environment); the name-taking forms read getenv. The Golden Master test (src/core/env_flag_test.cpp)
// pins every convention's full truth table - the M1 extraction of generate.cpp's helpers must preserve it.
//
// What arms a flag, per convention:
//   present:          any value at all, including "" and "0"          (getenv != nullptr)
//   truthy:           "" arms, "0" disarms                            (v && v[0] != '0')
//   truthy_strict:    "" and "0" both disarm                          (v && v[0] != '\0' && v[0] != '0')
//   nonzero:          a non-zero integer arms ("" / "0" / non-numeric disarm)  (v && atoi(v) != 0)
//   one:              only a value whose first character is '1'       (v && v[0] == '1')
#pragma once

#include <cstdlib>

namespace strata::env {

// --- pure predicates (value-taking; the Golden Master target) ---
inline bool present_value(const char* v) { return v != nullptr; }
inline bool truthy_value(const char* v) { return v != nullptr && v[0] != '0'; }
inline bool truthy_strict_value(const char* v) { return v != nullptr && v[0] != '\0' && v[0] != '0'; }
inline bool nonzero_value(const char* v) { return v != nullptr && std::atoi(v) != 0; }
inline bool one_value(const char* v) { return v != nullptr && v[0] == '1'; }

// --- getenv readers (the forms the call sites use) ---
inline bool present(const char* name) { return present_value(std::getenv(name)); }
inline bool truthy(const char* name) { return truthy_value(std::getenv(name)); }
inline bool truthy_strict(const char* name) { return truthy_strict_value(std::getenv(name)); }
inline bool nonzero(const char* name) { return nonzero_value(std::getenv(name)); }
inline bool one(const char* name) { return one_value(std::getenv(name)); }

// --- value readers (parse a number; unset or empty returns the default) ---
inline float float_value(const char* v, float dflt) {
    return (v != nullptr && v[0] != '\0') ? (float) std::atof(v) : dflt;
}
inline float float_or(const char* name, float dflt) { return float_value(std::getenv(name), dflt); }

}  // namespace strata::env
