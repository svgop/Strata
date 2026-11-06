// src/spec/controller.cpp - see include/strata/spec/controller.hpp.
#include "strata/spec/controller.hpp"

#include <algorithm>
#include <cmath>

namespace strata::spec {

double CostModel::step_ms(int k, bool mtp) const {
    const int n = std::clamp(k + 1, 1, K_MAX + 1);
    const double dense = dense_ms * dense_ratio[n - 1];
    // CPU misses: distinct experts grow as U(n); each extra token routed to a missed expert adds a fraction of a
    // read. Tokens per distinct expert = n / U(n)-ratio (U measured in units of one token's 10 experts).
    const double u = distinct_ratio[n - 1];
    const double uses_per_expert = (double) n / u;
    const double cpu = (1.0 - hit_rate) * cpu_all_miss_ms * u * (1.0 + extra_use_cost * (uses_per_expert - 1.0));
    const double draft = k * (mtp ? mtp_draft_ms : lookup_draft_ms);
    return dense + cpu + sync_ms + draft;
}

Controller::Controller(CostModel cost, double min_gain, double ema) : cost_(cost), min_gain_(min_gain), ema_(ema) {
    // Priors: MTP per-position acceptance from flyweight's measurement on this model (0.86 per draft); lookup by
    // match length from the offline replay (2026-09-23-suffix-lookup). Both are then learned per session.
    mtp_p_.fill(0.86);
    lookup_q_ = {0.35, 0.6, 0.8, 0.92};
}

double Controller::expected(const double* p, int k, bool same) {
    double e = 1.0, run = 1.0;                                // the verify pass always commits one token
    for (int i = 0; i < k; ++i) {
        run *= same ? p[0] : p[i];
        e += run;
    }
    return e;
}

Choice Controller::choose(int lookup_available, int lookup_match, bool mtp_ready) const {
    Choice best;
    best.tokens_per_ms = 1.0 / cost_.step_ms(0, false);
    const double baseline = best.tokens_per_ms;
    auto consider = [&](Source s, int k, double e) {
        const double rate = e / cost_.step_ms(k, s == Source::Mtp);
        if (rate > best.tokens_per_ms) best = Choice{s, k, e, rate};
    };
    const int lk = std::min(lookup_available, K_MAX);
    const double q = lookup_q_[bucket(lookup_match)];
    for (int k = 1; k <= lk; ++k) consider(Source::Lookup, k, expected(&q, k, true));
    if (mtp_ready)
        for (int k = 1; k <= K_MAX; ++k) consider(Source::Mtp, k, expected(mtp_p_.data(), k, false));
    if (best.source != Source::None && best.tokens_per_ms < baseline * (1.0 + min_gain_)) {
        Choice none;
        none.tokens_per_ms = baseline;
        return none;
    }
    return best;
}

void Controller::observe(const Choice& c, int accepted, int lookup_match) {
    if (c.source == Source::None || c.k <= 0) return;
    accepted = std::clamp(accepted, 0, c.k);
    // Positions 0..accepted-1 were accepted given their prefix; position `accepted` (if drafted) was rejected.
    const int seen = std::min(c.k, accepted + 1);
    for (int i = 0; i < seen; ++i) {
        const double hit = i < accepted ? 1.0 : 0.0;
        if (c.source == Source::Mtp) mtp_p_[i] += ema_ * (hit - mtp_p_[i]);
        else lookup_q_[bucket(lookup_match)] += ema_ * (hit - lookup_q_[bucket(lookup_match)]);
    }
    // Positions the window never tested keep their own evidence's estimate: no extrapolation.  The old law
    // pulled every unobserved deep position toward the deepest observed rate, so one lucky deep window
    // over-extended the whole table (digest decode-control2 §1, §3.4).  The window now grows only as the
    // damped IRLS refit lands evidence per position - SurvivalEstimator below (STRATA_SPEC_LOGISTIC).
}

// ---- SurvivalEstimator (STRATA_SPEC_LOGISTIC): see controller.hpp for the model and the constants' provenance.

double SurvivalEstimator::logit(double q) {
    q = std::clamp(q, 1e-4, 1.0 - 1e-4);
    return std::log(q / (1.0 - q));
}

double SurvivalEstimator::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-std::clamp(z, -40.0, 40.0)));
}

SurvivalEstimator::SurvivalEstimator() {
    // beta from the house prior (Controller's 0.86 per draft): the tables are the prior, not a parallel system.
    beta_.fill(logit(0.86));
}

void SurvivalEstimator::set_prior(const double* p, int n) {
    for (int i = 0; i < n && i < K_MAX; ++i)
        if (p[i] >= 0.0) beta_[(size_t) i] = logit(p[i]);
}

void SurvivalEstimator::observe(const double* f, const int* y, int seen) {
    for (int i = 0; i < seen && i < K_MAX; ++i)
        events_.push_back({(float) f[i], (float) (y[i] != 0), i});
    ++folds_;
    if (++since_refit_ >= REFIT_INTERVAL) refit();
}

double SurvivalEstimator::survival(int k, double f) const {
    if (k < 0 || k >= K_MAX) return 0.0;
    return sigmoid(omega_ * f + beta_[(size_t) k]);
}

void SurvivalEstimator::refit() {
    const int n = (int) events_.size();
    since_refit_ = 0;
    if (n < 2 * K_MAX) {   // too little evidence since the last refit: the prior stands (and stays damped)
        events_.clear();
        return;
    }
    // Evidence damping (vLLM's shipped law): the slope moves on the session's whole evidence, each intercept
    // only on its own position's.  The cross terms take the slope's damping (any consistent positive scaling
    // only slows the step; this one is the slope block's view of the shared column).
    std::array<double, K_MAX> n_k{};
    for (const Event& e : events_) n_k[(size_t) e.k] += 1.0;
    const double d_w = (double) n / ((double) n + DAMPING);
    std::array<double, K_MAX> d_k{};
    for (int k = 0; k < K_MAX; ++k) d_k[(size_t) k] = n_k[(size_t) k] / (n_k[(size_t) k] + DAMPING);
    double w = omega_, b[K_MAX];
    for (int k = 0; k < K_MAX; ++k) b[k] = beta_[(size_t) k];
    constexpr int P = K_MAX + 1;   // (omega, beta_0..beta_K-1): the arrowhead system
    for (int it = 0; it < 8; ++it) {
        double H[P][P] = {}, g[P] = {};
        for (const Event& e : events_) {
            const double p = sigmoid(w * (double) e.f + b[e.k]);
            const double wt = p * (1.0 - p);
            const double r = (double) e.y - p;
            g[0] += wt * d_w * (double) e.f * r;
            g[1 + e.k] += wt * d_k[(size_t) e.k] * r;
            H[0][0] += wt * d_w * (double) e.f * (double) e.f;
            H[0][1 + e.k] += wt * d_w * (double) e.f;
            H[1 + e.k][1 + e.k] += wt * d_k[(size_t) e.k];
        }
        // The ridge is NOT damped: a position with no evidence keeps its full prior pull (a damped ridge
        // would zero its diagonal and make the system singular - the refit would silently stop moving).
        H[0][0] += L2;
        for (int k = 0; k < K_MAX; ++k) {
            H[1 + k][0] = H[0][1 + k];
            H[1 + k][1 + k] += L2;
        }
        // Gauss with partial pivots, then a damped step (the trust region keeps one wild window from moving
        // the fit the extrapolation law's distance)
        double a[P][P + 1];
        for (int r = 0; r < P; ++r) {
            for (int c = 0; c < P; ++c) a[r][c] = H[r][c];
            a[r][P] = g[r];
        }
        double dx[P] = {};
        bool ok = true;
        for (int col = 0; col < P && ok; ++col) {
            int piv = col;
            for (int r = col + 1; r < P; ++r) if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
            if (std::fabs(a[piv][col]) < 1e-12) { ok = false; break; }
            if (piv != col)
                for (int c = col; c <= P; ++c) std::swap(a[col][c], a[piv][c]);
            for (int r = col + 1; r < P; ++r) {
                const double m = a[r][col] / a[col][col];
                for (int c = col; c <= P; ++c) a[r][c] -= m * a[col][c];
            }
        }
        if (ok) {
            for (int r = P - 1; r >= 0; --r) {
                double s = a[r][P];
                for (int c = r + 1; c < P; ++c) s -= a[r][c] * dx[c];
                dx[r] = s / a[r][r];
            }
        }
        if (!ok) break;
        // The step is damped by the same evidence factors as the system (vLLM's law: a data-poor position
        // keeps learning, just slowly), and the slope stays a positive correction on the prior - the
        // production slopes measured 0.51-0.55; omega <= 0 inverts confidence, omega -> 0 degrades safely
        // to the bare tables.  (Without this, a narrow-feature session - deep positions only ever see
        // low-confidence drafts - walks the fit's near-collinear valley to a nonsense corner.)
        w = std::clamp(w + std::clamp(dx[0] * d_w, -0.25, 0.25), 0.05, 4.0);
        for (int k = 0; k < K_MAX; ++k)
            b[k] = std::clamp(b[k] + std::clamp(dx[1 + k] * d_k[(size_t) k], -0.5, 0.5), -20.0, 20.0);
    }
    omega_ = w;
    for (int k = 0; k < K_MAX; ++k) beta_[(size_t) k] = b[k];
    ++refits_;
    events_.clear();
}

}  // namespace strata::spec
