// include/strata/spec/controller.hpp - plan v0.3 P6: which drafter, and how many draft tokens, this step.
//
// Each step the controller maximizes  E[tokens committed] / T(step)  over the choices
//     none (k = 0) | suffix lookup with k <= its proposal | MTP with k <= K_MAX
// using running estimates of per-position acceptance and a cost model of a (k+1)-token verify pass. It keeps
// k = 0 unless the best choice beats plain decoding by `min_gain` (plan: 5%).
//
// Acceptance: for MTP, position i's conditional acceptance p_i (EMA). For lookup, one conditional acceptance per
// match-length bucket (a long match is far more reliable than a trigram), also an EMA. `observe` updates them
// from how many drafted tokens the verify pass accepted.
//
// Cost: T(n tokens) = dense(n) + experts(n) + sync + draft. The defaults are the 23 Sep measurements:
// dense ratio per token count from native MMVQ (bench/results/2026-09-23-mmvq-multi), distinct experts per window
// from the routing trace (2026-09-23-spec-economics), extra-token CPU cost 0.2-0.35 (2026-09-23-cpu-expert-multi).
// Replace them with in-engine measurements through `CostModel`.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "strata/kernels/verify_kernels.hpp"

namespace strata::spec {

inline constexpr int K_MAX = 8;
// T65: the spec window IS the verify window's row cap - one constant, one wall for every array either sizes.
static_assert(K_MAX == strata::kernels::kVerifyMaxT,
              "the spec window and the verify row cap must stay one constant (T65)");

struct CostModel {
    double dense_ms = 11.0;                                  // one-token dense pass (plan P3 target)
    std::array<double, K_MAX + 1> dense_ratio{1.0, 1.05, 1.3, 1.45, 1.85, 2.2, 2.6, 3.0, 3.3};   // by n = k+1 (index n-1)
    double cpu_all_miss_ms = 15.8;                           // 480 experts on the CPU
    double hit_rate = 0.55;                                  // share of distinct experts served from VRAM
    std::array<double, K_MAX + 1> distinct_ratio{1.0, 1.70, 2.31, 2.88, 3.40, 3.89, 4.35, 4.80, 5.2};  // U(n)/U(1)
    double extra_use_cost = 0.25;                            // each extra token on an expert, fraction of a read
    double sync_ms = 2.4;
    double mtp_draft_ms = 1.2;                               // per MTP draft token
    double lookup_draft_ms = 0.01;

    /// Step time for verifying n = k + 1 tokens, plus drafting k tokens with the given source.
    double step_ms(int k, bool mtp) const;
};

enum class Source { None, Lookup, Mtp };

struct Choice {
    Source source = Source::None;
    int k = 0;
    double expected_tokens = 1.0;
    double tokens_per_ms = 0.0;
};

/// STRATA_SPEC_ENTABORT: the host-side mid-flight abort's inputs (digest decode-control2 §3.3 rule 3).  The
/// caller refreshes it before each draft round; the drafter's trim loop reads it between launches - plain
/// members only, nothing here is captured.  After each drafted position the chain stops when its modulated
/// survival sigma(omega * logit(q) + beta) falls below `s_min` (the production decay tail's floor) or when
/// AdaEDL's entropy gate fires (1 - gamma * H < lambda; H from the drafter's stats staging, one 1-block
/// reduction per step the graphs already hold).  Defaults are the digest's constants.
struct DraftAbort {
    static constexpr int kSlots = strata::kernels::kVerifyMaxT;   // one slot per drafted position
    static constexpr int kFloats = 2 * kSlots;                    // the drafter's stats staging: p, H per position
    static_assert(kSlots == K_MAX, "the abort's slots are the spec window (T65)");
    bool on = false;
    float omega = 0.5f;     // the estimator's shared slope
    float s_min = 0.10f;    // the survival floor implied by the production decay tail
    float gamma = 0.2f;     // AdaEDL's entropy weight
    float lambda = 0.70f;   // the entropy gate (the EMA of realized acceptance, clamped 0.3..0.9)
    float beta[kSlots] = {};   // per-position intercepts (the estimator's current ones)
};

/// STRATA_SPEC_LOGISTIC: the W1 acceptance tables become logistic priors (digest decode-control2 §3.1, §3.4).
/// Model: per-position p = sigma(omega * logit(q_max) + beta_k) - ONE shared slope, K_MAX intercepts, the
/// form vLLM ships.  The tables are the prior: beta_k starts at the logit of the house depth prior and is
/// re-seeded from the engine's own depth EMAs whenever they have evidence.  Labels fold with the same
/// censoring rule the depth EMAs use (positions <= accepted + 1); every REFIT_INTERVAL windows the K_MAX + 1
/// parameters are refilled by IRLS on the Bernoulli likelihood with the shipped damping - the slope's
/// evidence weight is n/(n + 50), each intercept's its own n_k/(n_k + 50), ridge 1e-3 on the diagonal.  The
/// damping replaces observe()'s deep-position extrapolation: an untested deep position moves only as its own
/// evidence lands, so one lucky deep window cannot over-extend the table.
class SurvivalEstimator {
public:
    static constexpr int REFIT_INTERVAL = 100;   // windows between refits (vLLM's shipped cadence)
    static constexpr double DAMPING = 50.0;      // evidence damping n / (n + 50)
    static constexpr double L2 = 1e-3;           // ridge on the refit's diagonal

    SurvivalEstimator();

    /// Seed the intercepts from the engine's own depth EMAs (the W1 feed).  `p[i] < 0` leaves position i at
    /// its current beta - the engine has not tested it yet this session.
    void set_prior(const double* p, int n);

    /// One window's labels: position i was drafted with feature `f[i]` (logit(q_max) of its draft) and label
    /// `y[i]`; `seen` = min(k, accepted + 1) holds the censoring rule.  Folds into the next refit and refits
    /// when the interval fills.
    void observe(const double* f, const int* y, int seen);

    /// sigma(omega * f + beta_k): the per-position survival the window's own drafts modulate (f = 0 is the
    /// bare prior).
    double survival(int k, double f) const;

    double omega() const { return omega_; }
    double beta(int k) const { return beta_[(size_t) k]; }
    int windows() const { return since_refit_; }
    int refits() const { return refits_; }
    int folds() const { return folds_; }

    /// Clamped at the estimator's numerical floor so a near-certain draw cannot send the fit to infinity.
    static double logit(double q);
    static double sigmoid(double z);

private:
    void refit();

    double omega_ = 0.5;   // vLLM's cold start, the median across five production model pairs
    std::array<double, K_MAX> beta_{};
    struct Event { float f, y; int k; };
    std::vector<Event> events_;   // this refit interval's labels (<= REFIT_INTERVAL * K_MAX, kilobytes)
    int since_refit_ = 0, refits_ = 0, folds_ = 0;
};

class Controller {
public:
    explicit Controller(CostModel cost = {}, double min_gain = 0.05, double ema = 0.05);

    /// `lookup_available` tokens the suffix drafter can propose now, from a match of `lookup_match` tokens.
    Choice choose(int lookup_available, int lookup_match, bool mtp_ready) const;

    /// After verification: `accepted` of the `k` drafted tokens matched (a prefix).
    void observe(const Choice& c, int accepted, int lookup_match);

    /// W1 wiring: seed the per-position acceptance straight from the engine's own depth EMAs
    /// (dp_ok/dp_tested by position).  `p[i] < 0` leaves position i at its prior - the engine has
    /// not tested it yet this session.
    void set_mtp_accept(const double* p, int n) {
        for (int i = 0; i < n && i < K_MAX; ++i)
            if (p[i] >= 0.0) mtp_p_[i] = p[i];
    }

    double mtp_accept(int position) const { return mtp_p_[position]; }     // position 0 = first draft token
    double lookup_accept(int match) const { return lookup_q_[bucket(match)]; }
    const CostModel& cost() const { return cost_; }

private:
    static int bucket(int match) { return match >= 16 ? 3 : match >= 8 ? 2 : match >= 5 ? 1 : 0; }
    static double expected(const double* p, int k, bool conditional_same);

    CostModel cost_;
    double min_gain_, ema_;
    std::array<double, K_MAX> mtp_p_{};
    std::array<double, 4> lookup_q_{};
};

}  // namespace strata::spec
