// src/core/pcie_share_rule_test.cpp - the ported miss-share rule's unit proof (upstream's
// own cases): a strong pool's shares do not move, a weak pool's rise to 0.85, and the env
// overrides win over every default.  Run: build && ./build/pcie_share_rule_test
#include "strata/core/pcie_share_rule.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace strata;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }
}  // namespace

int main() {
    // a strong pool (61.2 GB/s): the follow rule keeps the shipped share - our standing configs sit here
    check(near(pcie_share_for(56.1, 61.2, kPcieCeilingShipped, false), 0.35), "strong pool: shipped 0.35");
    check(near(pcie_share_for(56.1, 61.2, kPcieCeilingFollow, true), 0.35), "strong pool: follow keeps 0.35");
    // every rate from 30 GB/s up: k falls no further, the two rules agree
    for (double r = 30.0; r <= 80.0; r += 1.7)
        for (double link = 12.0; link <= 60.0; link += 3.1)
            check(near(pcie_share_for(link, r, kPcieCeilingShipped, false), pcie_share_for(link, r, kPcieCeilingFollow, true)),
                  "r >= 30: the rules agree");
    // a weak pool (10.3 GB/s): the follow rule carries far more of the miss (upstream's 4-worker case, 56 -> 79 tok/s)
    check(near(pcie_share_for(56.3, 10.3, kPcieCeilingShipped, false), 0.60), "weak pool: shipped ceiling 0.60");
    check(near(pcie_share_for(56.3, 10.3, kPcieCeilingFollow, true), 0.85), "weak pool: follow raises 0.85");
    // k itself: 1.85 from 30, 0.8 at 12, linear between; k_rule off pins 1.85
    check(near(pcie_pool_k(45.0, true), 1.85), "k pins 1.85 from 30");
    check(near(pcie_pool_k(10.0, true), 0.8), "k floors 0.8 at 12");
    check(near(pcie_pool_k(10.0, false), 1.85), "k_rule off: 1.85");
    check(pcie_pool_k(21.0, true) > 0.8 && pcie_pool_k(21.0, true) < 1.85, "k linear between");
    // the floor
    check(near(pcie_share_for(1.0, 60.0, kPcieCeilingFollow, true), 0.10), "a slow link floors at 0.10");
    // the defaults and their env overrides (this fork: no 026 mode, the arm is false)
    check(pcie_k_rule_default(true, false, nullptr), "k rule on for a discrete card");
    check(!pcie_k_rule_default(false, false, nullptr), "k rule off for integrated");
    check(!pcie_k_rule_default(true, false, "0"), "STRATA_PCIE_K_RULE=0 wins");
    check(near(pcie_share_ceiling(true, false, nullptr), 0.95), "ceiling 0.95 discrete");
    check(near(pcie_share_ceiling(false, false, nullptr), 0.60), "ceiling 0.60 integrated");
    check(near(pcie_share_ceiling(true, false, "0.70"), 0.70), "STRATA_PCIE_SHARE_MAX wins");
    check(near(pcie_share_ceiling(true, false, "9"), 1.0), "SHARE_MAX clamps at 1.0");
    check(pcie_auto_split_default(false, true, false, nullptr), "auto split on: CUDA discrete");
    check(!pcie_auto_split_default(true, true, false, nullptr), "auto split off on HIP");
    check(pcie_auto_split_default(true, true, false, "1"), "AUTO_SPLIT env wins");

    std::printf("pcie_share_rule_test: %d checks pass\n", checks);
    return 0;
}
