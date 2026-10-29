/* ref_impl.hpp - independent CPU reference implementation of PRIA.
 *
 * Everything runs in double precision on plain full [N][T][T] arrays (no
 * dense pair-packed layouts), which makes it an independent check of the GPU
 * implementation's layouts, projections and solver loop. The control flow of
 * solve() mirrors pria::solve_impl exactly.
 */
#ifndef PRIA_TESTS_REF_IMPL_HPP
#define PRIA_TESTS_REF_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "pria/pria.h"

namespace ref {

using std::vector;
using i64 = int64_t;

/* ---- PCG32 ------------------------------------------------------------- */
struct Pcg32 {
    uint64_t state, inc;
    explicit Pcg32(uint64_t seed, uint64_t seq = 0xda3e39cb94b95bdbULL)
        : state(0), inc((seq << 1) | 1) {
        next();
        state += seed;
        next();
    }
    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = (uint32_t)(((old >> 18) ^ old) >> 27);
        uint32_t rot = (uint32_t)(old >> 59);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    double uniform(double lo, double hi) {
        return lo + (hi - lo) * (next() * (1.0 / 4294967296.0));
    }
};

/* ---- instances --------------------------------------------------------- */
struct Instance {
    i64 n, t;
    vector<double> k;  /* [n] */
    double k0;
    vector<double> h;  /* [n*t*t], upper triangular */
};

/* Random instance with the same structure as
 * JrpProblem.random_instance: integer costs, zero diagonal, holding
 * costs non-decreasing in t. */
inline Instance random_instance(uint64_t seed, i64 n, i64 t,
                                bool force_joint = false) {
    Pcg32 rng(seed);
    Instance inst;
    inst.n = n;
    inst.t = t;
    inst.h.assign(n * t * t, 0.0);
    inst.k.resize(n);
    for (i64 i = 0; i < n; ++i) {
        double mult = rng.uniform(0.0, 1.0);
        for (i64 s = 0; s < t; ++s)
            for (i64 tt = s + 1; tt < t; ++tt) {
                double eps = rng.uniform(0.0, mult);
                inst.h[(i * t + s) * t + tt] =
                    std::nearbyint(mult * (double)(tt - s) + eps);
            }
    }
    double hi = std::max((double)t, 1.01);
    for (i64 i = 0; i < n; ++i)
        inst.k[i] = std::max(1.0, std::nearbyint(rng.uniform(1.0, hi)));
    inst.k0 = (n > 1 || force_joint)
                  ? std::max(1.0, std::nearbyint(rng.uniform(1.0, hi)))
                  : 0.0;
    return inst;
}

/* ---- projections ------------------------------------------------------- */

/* Sort-based simplex projection with the count-based rho, matching the GPU
 * pipeline (and double_simplex_projection in JAX). target <= 0 maps to the
 * degenerate simplex {0}. */
inline void simplex_project(vector<double>& v, double target) {
    vector<double> u(v), cum(v.size());
    std::sort(u.begin(), u.end(), std::greater<double>());
    double running = 0.0;
    i64 rho = 0;
    for (size_t kk = 0; kk < u.size(); ++kk) {
        running += u[kk];
        cum[kk] = running;
        if (u[kk] * (double)(kk + 1) > running - target) ++rho;
    }
    double theta = rho > 0 ? (cum[rho - 1] - target) / (double)rho
                           : std::numeric_limits<double>::infinity();
    for (auto& x : v) x = std::max(x - theta, 0.0);
}

inline void proj_x(i64 n, i64 t, vector<double>& x) {
    vector<double> seg;
    for (i64 i = 0; i < n; ++i)
        for (i64 tt = 0; tt < t; ++tt) {
            seg.clear();
            for (i64 s = 0; s <= tt; ++s) seg.push_back(x[(i * t + s) * t + tt]);
            simplex_project(seg, 1.0);
            for (i64 s = 0; s <= tt; ++s) x[(i * t + s) * t + tt] = seg[s];
        }
}

inline void proj_y(i64 n, i64 t, vector<double>& y, const vector<double>& k) {
    vector<double> seg;
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < t; ++s) {
            seg.clear();
            for (i64 tt = s; tt < t; ++tt) seg.push_back(y[(i * t + s) * t + tt]);
            simplex_project(seg, k[i]);
            for (i64 tt = s; tt < t; ++tt) y[(i * t + s) * t + tt] = seg[tt - s];
        }
}

inline void proj_z(i64 n, i64 t, vector<double>& z, double k0) {
    vector<double> seg;
    for (i64 s = 0; s < t; ++s) {
        seg.clear();
        for (i64 i = 0; i < n; ++i)
            for (i64 tt = s; tt < t; ++tt) seg.push_back(z[(i * t + s) * t + tt]);
        simplex_project(seg, k0);
        size_t p = 0;
        for (i64 i = 0; i < n; ++i)
            for (i64 tt = s; tt < t; ++tt) z[(i * t + s) * t + tt] = seg[p++];
    }
}

/* ---- one PDHG operator application ------------------------------------- */
struct State {
    vector<double> x, y, z;
};

inline void raw_step(i64 n, i64 t, const vector<double>& h,
                     const vector<double>& k, double k0, double omega,
                     double eta, const State& in, State& out) {
    const double tau = eta / omega, sigma = eta * omega;
    const i64 sz = n * t * t;
    out.x.assign(sz, 0.0);
    out.y.assign(sz, 0.0);
    out.z.assign(sz, 0.0);
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < t; ++s)
            for (i64 tt = s; tt < t; ++tt) {
                i64 g = (i * t + s) * t + tt;
                out.x[g] = in.x[g] - tau * (h[g] + in.y[g] + in.z[g]);
            }
    proj_x(n, t, out.x);
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < t; ++s)
            for (i64 tt = s; tt < t; ++tt) {
                i64 g = (i * t + s) * t + tt;
                double r = sigma * (2.0 * out.x[g] - in.x[g]);
                out.y[g] = in.y[g] + r;
                out.z[g] = in.z[g] + r;
            }
    proj_y(n, t, out.y, k);
    proj_z(n, t, out.z, k0);
}

/* ---- norms and objectives ---------------------------------------------- */

inline double canonical_err(i64 n, i64 t, const State& a, const State& b,
                            double eta, double omega) {
    double a0 = 0, a1 = 0, a2 = 0;
    for (i64 g = 0; g < n * t * t; ++g) {
        double dx = a.x[g] - b.x[g], dy = a.y[g] - b.y[g], dz = a.z[g] - b.z[g];
        a0 += dx * dx;
        a1 += dy * dy + dz * dz;
        a2 += dx * (dy + dz);
    }
    /* The cross term is NEGATIVE.  P is the proximal-point metric of one PDHG
     * step, whose off-diagonal blocks are -A^T, and A is the identity here --
     * it is the norm in which the step is firmly nonexpansive and the restart
     * residual provably monotone.  Flipping this sign leaves a norm that still
     * looks plausible and still decreases, so nothing fails loudly; what
     * drifts is where the solver restarts. */
    double v = (omega / eta) * a0 + (1.0 / (eta * omega)) * a1 - 2.0 * a2;
    return std::sqrt(std::max(v, 0.0));
}

struct Gap {
    double primal, dual, gap;
};

inline Gap objective_gap(i64 n, i64 t, const vector<double>& h,
                         const vector<double>& k, double k0, double scale,
                         const State& s) {
    double term1 = 0.0, term2 = 0.0, term3 = 0.0, dual = 0.0;
    for (i64 i = 0; i < n; ++i)
        for (i64 ss = 0; ss < t; ++ss) {
            double m = -std::numeric_limits<double>::infinity();
            for (i64 tt = ss; tt < t; ++tt)
                m = std::max(m, k[i] * s.x[(i * t + ss) * t + tt]);
            term1 += m;
        }
    for (i64 ss = 0; ss < t; ++ss) {
        double m = -std::numeric_limits<double>::infinity();
        for (i64 i = 0; i < n; ++i)
            for (i64 tt = ss; tt < t; ++tt)
                m = std::max(m, k0 * s.x[(i * t + ss) * t + tt]);
        term2 += m;
    }
    for (i64 g = 0; g < n * t * t; ++g) term3 += s.x[g] * h[g];
    for (i64 i = 0; i < n; ++i)
        for (i64 tt = 0; tt < t; ++tt) {
            double m = std::numeric_limits<double>::infinity();
            for (i64 ss = 0; ss <= tt; ++ss) {
                i64 g = (i * t + ss) * t + tt;
                m = std::min(m, h[g] + s.y[g] + s.z[g]);
            }
            dual += m;
        }
    Gap gi;
    gi.primal = (term1 + term2 + term3) * scale;
    gi.dual = dual * scale;
    gi.gap = (gi.primal - gi.dual) /
             (1.0 + std::fabs(gi.dual) + std::fabs(gi.primal));
    return gi;
}

/* ---- full reference solver --------------------------------------------- */
struct Result {
    vector<double> x, y, z; /* [n*t*t] on the ORIGINAL (unpadded) size */
    i64 iterations = 0;
    double primal = 0, dual = 0, gap = 0;
};

inline Result solve(const Instance& inst, const pria_params& params_in,
                    const State* warm = nullptr) {
    pria_params p = params_in;
    if (p.op_steps < 2) p.op_steps = 2;
    if (p.warm_steps < 0) p.warm_steps = 0;

    const i64 n = inst.n, torig = inst.t;
    const i64 t = torig + (torig & 1);
    const i64 sz = n * t * t;

    /* scale from the unpadded instance */
    double ss = inst.k0 * inst.k0;
    for (double v : inst.k) ss += v * v;
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < torig; ++s)
            for (i64 tt = s; tt < torig; ++tt) {
                double v = inst.h[(i * torig + s) * torig + tt];
                ss += v * v;
            }
    double scale = std::sqrt(ss);
    if (scale == 0.0) scale = 1.0;

    vector<double> h(sz, 0.0), k(n);
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < torig; ++s)
            for (i64 tt = s; tt < torig; ++tt)
                h[(i * t + s) * t + tt] =
                    inst.h[(i * torig + s) * torig + tt] / scale;
    for (i64 i = 0; i < n; ++i) k[i] = inst.k[i] / scale;
    const double k0 = inst.k0 / scale;

    State cur;
    cur.x.assign(sz, 0.0);
    cur.y.assign(sz, 0.0);
    cur.z.assign(sz, 0.0);
    if (warm) {
        /* warm duals are normalized, like the solver's outputs */
        for (i64 i = 0; i < n; ++i)
            for (i64 s = 0; s < torig; ++s)
                for (i64 tt = s; tt < torig; ++tt) {
                    i64 g = (i * t + s) * t + tt;
                    i64 go = (i * torig + s) * torig + tt;
                    cur.x[g] = warm->x[go];
                    cur.y[g] = warm->y[go] * k[i];
                    cur.z[g] = warm->z[go] * k0;
                }
    }
    proj_x(n, t, cur.x);
    proj_y(n, t, cur.y, k);
    proj_z(n, t, cur.z, k0);

    State anchor = cur;
    State step;
    double omega = p.omega;
    const double eta = p.eta;
    i64 iteration = 0;
    raw_step(n, t, h, k, k0, omega, eta, cur, step);

    Gap gi{0.0, 0.0, std::numeric_limits<double>::infinity()};

    auto vanilla_step = [&]() {
        State next;
        raw_step(n, t, h, k, k0, omega, eta, step, next);
        cur = step;
        step = next;
        ++iteration;
    };
    auto run_vanilla = [&](i64 local_max) {
        for (;;) {
            gi = objective_gap(n, t, h, k, k0, scale, cur);
            bool within = (iteration < local_max) || (local_max < 0);
            if (!within || !(gi.gap > p.tolerance)) return;
            for (int s2 = 0; s2 < p.op_steps; ++s2) vanilla_step();
        }
    };

    i64 warm_budget =
        p.use_halpern ? (i64)p.op_steps * p.warm_steps : p.max_iter;
    if (!p.use_halpern || warm_budget != 0) run_vanilla(warm_budget);

    if (iteration > 0) {
        double b0 = 0, b1 = 0;
        for (i64 g = 0; g < sz; ++g) {
            double dx = cur.x[g] - anchor.x[g];
            double dy = cur.y[g] - anchor.y[g];
            double dz = cur.z[g] - anchor.z[g];
            b0 += dx * dx;
            b1 += dy * dy + dz * dz;
        }
        double om = std::sqrt(b1) / std::sqrt(b0);
        omega = std::isnan(om) ? p.omega : om;
    }

    bool report_step = false;
    if (p.use_halpern) {
        double theta = p.theta_initial;
        double check_err = std::numeric_limits<double>::infinity();
        i64 restart_iter = 0;

        anchor = cur;
        double fp_err = canonical_err(n, t, cur, step, eta, omega);
        gi = objective_gap(n, t, h, k, k0, scale, cur);
        bool terminated = !(gi.gap > p.tolerance) ||
                          (p.max_iter >= 0 && iteration >= p.max_iter);

        while (!terminated) {
            report_step = true;
            for (int s2 = 0; s2 < p.op_steps; ++s2) {
                double lw = (double)(restart_iter + 1) / (restart_iter + 2);
                double rw = 1.0 / (restart_iter + 2);
                for (i64 g = 0; g < sz; ++g) {
                    cur.x[g] = lw * (2.0 * step.x[g] - cur.x[g]) + rw * anchor.x[g];
                    cur.y[g] = lw * (2.0 * step.y[g] - cur.y[g]) + rw * anchor.y[g];
                    cur.z[g] = lw * (2.0 * step.z[g] - cur.z[g]) + rw * anchor.z[g];
                }
                State next;
                raw_step(n, t, h, k, k0, omega, eta, cur, next);
                step = next;
                ++restart_iter;
                ++iteration;
            }

            double new_err = canonical_err(n, t, cur, step, eta, omega);
            gi = objective_gap(n, t, h, k, k0, scale, step);
            terminated = !(gi.gap > p.tolerance) ||
                         (p.max_iter >= 0 && iteration >= p.max_iter);

            bool sufficient = new_err <= p.beta_sufficient * fp_err;
            bool necessary = (new_err <= p.beta_necessary * fp_err) &&
                             (new_err > check_err);
            bool artificial =
                (double)restart_iter >= p.beta_artificial * (double)iteration;

            if (terminated) break;

            if (sufficient || necessary || artificial) {
                double b0 = 0, b1 = 0;
                for (i64 g = 0; g < sz; ++g) {
                    double dx = cur.x[g] - anchor.x[g];
                    double dy = cur.y[g] - anchor.y[g];
                    double dz = cur.z[g] - anchor.z[g];
                    b0 += dx * dx;
                    b1 += dy * dy + dz * dz;
                }
                double pn = std::sqrt(b0), dn = std::sqrt(b1);
                if (pn > 1e-16 && dn > 1e-16) {
                    omega = std::exp(theta * (std::log(dn) - std::log(pn)) +
                                     (1.0 - theta) * std::log(omega));
                    theta = p.theta_target +
                            p.theta_decay * (theta - p.theta_target);
                }
                anchor = step;
                State next;
                raw_step(n, t, h, k, k0, omega, eta, step, next);
                cur = step;
                step = next;
                fp_err = canonical_err(n, t, cur, step, eta, omega);
                check_err = std::numeric_limits<double>::infinity();
                restart_iter = 0;
            } else {
                check_err = new_err;
            }
        }
    }

    const State& fin = report_step ? step : cur;
    Result res;
    res.iterations = iteration;
    res.primal = gi.primal;
    res.dual = gi.dual;
    res.gap = gi.gap;
    const i64 szo = n * torig * torig;
    res.x.assign(szo, 0.0);
    res.y.assign(szo, 0.0);
    res.z.assign(szo, 0.0);
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < torig; ++s)
            for (i64 tt = s; tt < torig; ++tt) {
                i64 g = (i * t + s) * t + tt;
                i64 go = (i * torig + s) * torig + tt;
                res.x[go] = fin.x[g];
                res.y[go] = k[i] > 0.0 ? fin.y[g] / k[i] : 0.0;
                res.z[go] = k0 > 0.0 ? fin.z[g] / k0 : 0.0;
            }
    return res;
}

} // namespace ref

#endif /* PRIA_TESTS_REF_IMPL_HPP */
