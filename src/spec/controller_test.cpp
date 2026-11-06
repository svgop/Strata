// src/spec/controller_test.cpp - plan v0.3 P6: the speculation controller's decisions (no model, no GPU).
#include "strata/spec/controller.hpp"

#include <cstdio>

using namespace strata::spec;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}
const char* name(Source s) { return s == Source::Mtp ? "mtp" : s == Source::Lookup ? "lookup" : "none"; }
}  // namespace

int main() {
    {   // priors (MTP 0.86 per draft): MTP with a short-to-medium window beats plain decoding
        Controller c;
        const Choice ch = c.choose(0, 0, true);
        check(ch.source == Source::Mtp && ch.k >= 2 && ch.k <= 5, "MTP chosen with k in 2..5 at the prior");
        std::printf("prior: %s k=%d expected %.2f tokens, %.1f tok/s (plain %.1f)\n", name(ch.source), ch.k,
                    ch.expected_tokens, 1000 * ch.tokens_per_ms, 1000.0 / c.cost().step_ms(0, false));
    }
    {   // a drafter that keeps missing its first token is switched off
        Controller c;
        Choice ch = c.choose(0, 0, true);
        for (int i = 0; i < 400; ++i) { c.observe(ch, 0, 0); ch = c.choose(0, 0, true); if (ch.source == Source::None) break; }
        check(ch.source == Source::None, "persistent rejection turns speculation off");
        std::printf("after rejections: %s, p0 %.2f\n", name(ch.source), c.mtp_accept(0));
    }
    {   // a long lookup match beats MTP; how LONG a window pays depends on what a wide verify costs
        Controller c;
        const Choice ch = c.choose(8, 20, true);
        check(ch.source == Source::Lookup, "long lookup match preferred over MTP");
        std::printf("long lookup match, measured costs: %s k=%d expected %.2f, %.1f tok/s\n", name(ch.source), ch.k,
                    ch.expected_tokens, 1000 * ch.tokens_per_ms);
        CostModel cheap;                                    // a verify pass whose cost barely grows with width
        cheap.dense_ratio = {1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16};
        cheap.hit_rate = 0.95;
        Controller c2(cheap);
        const Choice ch2 = c2.choose(8, 20, true);
        check(ch2.source == Source::Lookup && ch2.k == 8, "cheap wide verify takes the whole lookup window");
        std::printf("long lookup match, cheap wide verify: %s k=%d expected %.2f, %.1f tok/s\n", name(ch2.source),
                    ch2.k, ch2.expected_tokens, 1000 * ch2.tokens_per_ms);
    }
    {   // a bare trigram match, no MTP: not worth a verify window
        Controller c;
        const Choice ch = c.choose(8, 3, false);
        check(ch.source == Source::None, "short lookup match alone is not used");
    }
    {   // full acceptance is learned at the positions the window tested; the window does NOT grow past its
        // evidence in the EMA path - the old law pulled untested deep positions toward the deepest observed
        // rate (the decode-control2 §1 hazard) to manufacture that growth, and it is gone.  Growth now comes
        // from per-position evidence: SurvivalEstimator's damped refit (STRATA_SPEC_LOGISTIC, tested below).
        Controller c;
        Choice ch = c.choose(0, 0, true);
        const int k0 = ch.k;
        for (int i = 0; i < 400; ++i) { c.observe(ch, ch.k, 0); ch = c.choose(0, 0, true); }
        check(ch.k == k0 && c.mtp_accept(0) > 0.99, "full acceptance learned; the window stays at its evidence");
        std::printf("after full acceptance: %s k=%d (was %d), p0 %.3f\n", name(ch.source), ch.k, k0, c.mtp_accept(0));
    }
    {   // the extrapolation law is gone: one all-accepted window at the chosen k must NOT pull the untested
        // deep positions toward the deepest observed rate.  Untested positions keep their own estimate.
        Controller c;
        const Choice ch = c.choose(0, 0, true);
        c.observe(ch, ch.k, 0);   // every drafted position accepted
        for (int d = ch.k; d < K_MAX; ++d)
            check(c.mtp_accept(d) == 0.86, "an untested deep position keeps its prior (no extrapolation)");
    }
    {   // SurvivalEstimator: the tables seed the prior; the window's own feature modulates it
        SurvivalEstimator e;
        const double p0[2] = {0.86, 0.70};
        e.set_prior(p0, 2);
        check(e.beta(0) > 0.0 && e.beta(1) < e.beta(0), "betas seeded from the tables' logits");
        const double s_conf = e.survival(0, 2.0), s_mid = e.survival(0, 0.0), s_doubt = e.survival(0, -2.0);
        check(s_conf > s_mid && s_mid > s_doubt, "a confident draft modulates the prior upward");
        check(e.survival(1, 0.0) < e.survival(0, 0.0), "the weaker table sits lower at the same feature");
    }
    {   // SurvivalEstimator: the refit learns a positive slope on evidence, and the damping keeps a
        // data-poor deep position from moving like a data-rich shallow one (the extrapolation fix)
        SurvivalEstimator e;
        for (int w = 0; w < 400; ++w) {   // the truth: p = sigma(1.0 * f + (position 0 ? +1 : -1)); the fit
            double f[K_MAX];              // must find the slope's direction AND the positions' separation
            int y[K_MAX];
            const int seen = 2;
            for (int d = 0; d < seen; ++d) {
                f[d] = (d == 0 ? 2.0 : -1.0) + 0.5 * (double) (w % 3 - 1);   // the feature moves per window
                unsigned r = 2654435761u * (unsigned) (w * 8 + d + 1);
                const double u = (double) (((r = r * 1103515245u + 12345u) >> 8) % 100000) / 100000.0;
                y[d] = u < SurvivalEstimator::sigmoid(1.0 * f[d] + (d == 0 ? 1.0 : -1.0));
            }
            e.observe(f, y, seen);
        }
        check(e.refits() >= 3, "the fit refills on its interval");
        check(e.omega() > 0.2, "the slope learned the feature's direction");
        check(e.beta(0) > e.beta(1), "the intercepts separated on per-position evidence");
    }
    std::printf("\nMTP acceptance -> choice (default cost model):\n");
    for (double p : {0.5, 0.6, 0.7, 0.8, 0.86, 0.9, 0.95}) {
        Controller c;
        Choice ch = c.choose(0, 0, true);
        for (int i = 0; i < 2000; ++i) {                    // feed a stream with iid acceptance p per position
            int acc = 0;
            unsigned r = 2654435761u * (unsigned) (i + 1);
            while (acc < ch.k && ((r = r * 1103515245u + 12345u) >> 8) % 1000 < p * 1000) ++acc;
            c.observe(ch, acc, 0);
            ch = c.choose(0, 0, true);
        }
        std::printf("  p=%.2f: %s k=%d, %.1f tok/s\n", p, name(ch.source), ch.k, 1000 * ch.tokens_per_ms);
    }
    std::printf("controller tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
