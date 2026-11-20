// src/core/env_flag_test.cpp - Golden Master for strata::env truthiness.
//
// Pins the exact arm/disarm behavior of every convention for the input classes that have historically
// diverged (unset, "", "0", "1", numeric, non-numeric). This is the oracle the M1 extraction of
// generate.cpp's env-helper lambdas must satisfy: a convention change without an operator decision fails
// this test.
#include "strata/core/env_flag.hpp"

#include <cstdio>

namespace {
int failures = 0;
void expect(const char* value, bool got, bool want) {
    if (got != want) {
        std::printf("FAIL value=%-8s got %d want %d\n", value, (int) got, (int) want);
        ++failures;
    }
}
}  // namespace

int main() {
    struct Case {
        const char* v;
        bool p, t, ts, n, o;  // present, truthy, truthy_strict, nonzero, one
    };
    const Case cases[] = {
        //          value     present  truthy  t_strict  nonzero  one
        {nullptr,                false, false,  false,  false, false},  // unset
        {"",                     true,  true,   false,  false, false},  // "" arms present+truthy only
        {"0",                    true,  false,  false,  false, false},
        {"00",                   true,  false,  false,  false, false},
        {"1",                    true,  true,   true,   true,  true},
        {"10",                   true,  true,   true,   true,  true},   // one() reads first char only
        {"2",                    true,  true,   true,   true,  false},
        {"abc",                  true,  true,   true,   false, false},  // atoi("abc") == 0
        {"-1",                   true,  true,   true,   true,  false},  // atoi("-1") == -1
    };
    for (const Case& c : cases) {
        const char* label = c.v ? c.v : "(unset)";
        expect(label, strata::env::present_value(c.v), c.p);
        expect(label, strata::env::truthy_value(c.v), c.t);
        expect(label, strata::env::truthy_strict_value(c.v), c.ts);
        expect(label, strata::env::nonzero_value(c.v), c.n);
        expect(label, strata::env::one_value(c.v), c.o);
    }
    if (failures) {
        std::printf("env_flag_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("env_flag_test: ok\n");
    return 0;
}
