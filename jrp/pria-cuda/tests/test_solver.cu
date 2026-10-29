/* test_solver.cu - end-to-end tests of the public PRIA API.
 *
 * The decisive check is self-certifying: the returned x is primal feasible
 * and the returned (y, z) are dual feasible, so the objective F(x) and the
 * dual bound G(y, z), both recomputed on the host in double precision on the
 * ORIGINAL (unpadded, unscaled) instance, sandwich the LP optimum. The
 * relative gap between them must be within the requested tolerance.
 */
#include <vector>

#include "pria/pria.h"
#include "ref_impl.hpp"
#include "solver.cuh" /* pria::graphs_disabled() test hook */
#include "test_common.hpp"

using ref::i64;
using std::vector;

/* dual objective G on the original instance from NORMALIZED duals */
static double dual_bound(const ref::Instance& in, const vector<double>& ynorm,
                         const vector<double>& znorm) {
    const i64 n = in.n, t = in.t;
    double g = 0.0;
    for (i64 i = 0; i < n; ++i)
        for (i64 tt = 0; tt < t; ++tt) {
            double m = std::numeric_limits<double>::infinity();
            for (i64 s = 0; s <= tt; ++s) {
                i64 idx = (i * t + s) * t + tt;
                m = std::min(m, in.h[idx] + in.k[i] * ynorm[idx] +
                                    in.k0 * znorm[idx]);
            }
            g += m;
        }
    return g;
}

static void check_certificate(const ref::Instance& in,
                              const vector<double>& x,
                              const vector<double>& y,
                              const vector<double>& z,
                              const pria_result_info& info, double tol,
                              double feas_tol, double cert_slack,
                              double consistency_tol) {
    const i64 n = in.n, t = in.t;

    /* primal feasibility */
    for (i64 i = 0; i < n; ++i)
        for (i64 tt = 0; tt < t; ++tt) {
            double sum = 0.0;
            for (i64 s = 0; s <= tt; ++s) {
                double v = x[(i * t + s) * t + tt];
                CHECK(v >= -feas_tol);
                sum += v;
            }
            CHECK_NEAR(sum, 1.0, feas_tol * (double)t);
        }

    /* Dual feasibility of the normalized duals. For odd t the solver pads a
     * period and the truncated rows keep only part of the mass, so the sums
     * are <= 1 (exactly like the JAX solver's outputs); the dual bound G
     * remains valid with row sums <= K_i. Even t is exact. */
    const bool padded = (t % 2 != 0);
    for (i64 i = 0; i < n; ++i)
        for (i64 s = 0; s < t; ++s) {
            double sum = 0.0;
            for (i64 tt = s; tt < t; ++tt) {
                double v = y[(i * t + s) * t + tt];
                CHECK(v >= -feas_tol);
                sum += v;
            }
            if (in.k[i] > 0.0) {
                CHECK(sum <= 1.0 + feas_tol * (double)t);
                if (!padded) CHECK_NEAR(sum, 1.0, feas_tol * (double)t);
            }
        }
    for (i64 s = 0; s < t; ++s) {
        double sum = 0.0;
        for (i64 i = 0; i < n; ++i)
            for (i64 tt = s; tt < t; ++tt) {
                double v = z[(i * t + s) * t + tt];
                CHECK(v >= -feas_tol);
                sum += v;
            }
        if (in.k0 > 0.0) {
            CHECK(sum <= 1.0 + feas_tol * (double)(n * t));
            if (!padded) CHECK_NEAR(sum, 1.0, feas_tol * (double)(n * t));
        } else {
            CHECK_NEAR(sum, 0.0, feas_tol * (double)(n * t));
        }
    }

    /* Duality-gap certificate on the original instance. F comes from the
     * returned primal; the lower bound is the better of the truncated-dual
     * bound and the solver's padded dual bound (the padded problem has the
     * same optimum, so info.dual_objective <= p* as well; for odd t the
     * truncated duals alone are loose because the padded period keeps some
     * dual mass). */
    double F = pria_objective_f64(n, t, in.k.data(), in.k0, in.h.data(),
                                  x.data());
    double G = std::max(dual_bound(in, y, z), info.dual_objective);
    double gap = (F - G) / (1.0 + std::fabs(F) + std::fabs(G));
    CHECK(G <= F + feas_tol * (1.0 + std::fabs(F)));
    if (!(gap <= cert_slack * tol + 1e-9)) {
        fprintf(stderr, "certificate: F=%.12g G=%.12g gap=%.3e tol=%.1e\n", F,
                G, gap, tol);
        CHECK(gap <= cert_slack * tol + 1e-9);
    }

    /* For even t there is no padding: the host-recomputed objectives must
     * agree with the solver's own (scaled-problem) values. */
    if (t % 2 == 0) {
        CHECK_NEAR(F, info.primal_objective,
                   consistency_tol * (1.0 + std::fabs(F)));
        CHECK_NEAR(dual_bound(in, y, z), info.dual_objective,
                   consistency_tol * (1.0 + std::fabs(info.dual_objective)));
    }
}

static void run_case_f64(uint64_t seed, i64 n, i64 t, double tol,
                         int use_halpern, bool check_ref) {
    ref::Instance in = ref::random_instance(seed, n, t);
    pria_params p = pria_params_auto(-1, tol);
    p.use_halpern = use_halpern;

    const i64 sz = n * t * t;
    vector<double> x(sz), y(sz), z(sz);
    pria_result_info info;
    CHECK(pria_solve_f64(n, t, in.k.data(), in.k0, in.h.data(), &p, nullptr,
                         nullptr, nullptr, x.data(), y.data(), z.data(),
                         &info) == 0);
    CHECK(info.relative_gap <= tol);
    check_certificate(in, x, y, z, info, tol, 1e-8, 1.5, 1e-9);

    if (check_ref) {
        ref::Result r = ref::solve(in, p);
        CHECK(r.gap <= tol);
        if (info.iterations != r.iterations)
            fprintf(stderr,
                    "note: iterations differ gpu=%lld ref=%lld (n=%lld t=%lld)"
                    "\n",
                    (long long)info.iterations, (long long)r.iterations,
                    (long long)n, (long long)t);
        CHECK(std::llabs((long long)(info.iterations - r.iterations)) <=
              (long long)(p.op_steps));
        double denom = 1.0 + std::fabs(r.primal) + std::fabs(info.primal_objective);
        CHECK(std::fabs(info.primal_objective - r.primal) / denom <= 4.0 * tol);
    }

    /* Warm start from the solution. With even t the projected warm start IS
     * the final iterate, so the init check terminates immediately; with odd
     * t the padded period is lost in the round trip and only fast
     * re-convergence can be expected. */
    pria_result_info info2;
    vector<double> x2(sz), y2(sz), z2(sz);
    CHECK(pria_solve_f64(n, t, in.k.data(), in.k0, in.h.data(), &p, x.data(),
                         y.data(), z.data(), x2.data(), y2.data(), z2.data(),
                         &info2) == 0);
    if (t % 2 == 0) CHECK(info2.iterations == 0);
    else CHECK(info2.iterations <= std::max<int64_t>(info.iterations, 1));
    CHECK(info2.relative_gap <= tol);
}

static void run_case_f32(uint64_t seed, i64 n, i64 t, double tol) {
    ref::Instance in = ref::random_instance(seed, n, t);
    pria_params p = pria_params_auto(-1, tol);

    const i64 sz = n * t * t;
    vector<float> kf(n), hf(sz), x(sz), y(sz), z(sz);
    for (i64 i = 0; i < n; ++i) kf[i] = (float)in.k[i];
    for (i64 g = 0; g < sz; ++g) hf[g] = (float)in.h[g];

    pria_result_info info;
    CHECK(pria_solve_f32(n, t, kf.data(), (float)in.k0, hf.data(), &p, nullptr,
                         nullptr, nullptr, x.data(), y.data(), z.data(),
                         &info) == 0);
    CHECK(info.relative_gap <= tol);

    vector<double> xd(x.begin(), x.end()), yd(y.begin(), y.end()),
        zd(z.begin(), z.end());
    check_certificate(in, xd, yd, zd, info, tol, 2e-4, 4.0, 1e-3);
}

/* The graph paths replay exactly the launches the plain path makes and every
 * decision is computed by the same device kernels, so all three execution
 * modes - whole-solve conditional graph (default), host-driven block graphs,
 * plain launches - must match bitwise: same iterations, same objectives,
 * same returned iterates. */
static void run_graphs_equivalence(uint64_t seed, i64 n, i64 t, double tol,
                                   int use_halpern, int op_steps) {
    ref::Instance in = ref::random_instance(seed, n, t);
    pria_params p = pria_params_auto(-1, tol);
    p.use_halpern = use_halpern;
    if (op_steps > 0) p.op_steps = op_steps;
    const i64 sz = n * t * t;
    vector<double> x[3], y[3], z[3];
    pria_result_info info[3];

    for (int mode = 0; mode < 3; ++mode) {
        pria::cond_disabled() = mode >= 1;
        pria::graphs_disabled() = mode == 2;
        x[mode].resize(sz);
        y[mode].resize(sz);
        z[mode].resize(sz);
        CHECK(pria_solve_f64(n, t, in.k.data(), in.k0, in.h.data(), &p,
                             nullptr, nullptr, nullptr, x[mode].data(),
                             y[mode].data(), z[mode].data(),
                             &info[mode]) == 0);
    }
    pria::cond_disabled() = false;
    pria::graphs_disabled() = false;

    for (int mode = 1; mode < 3; ++mode) {
        CHECK(info[0].iterations == info[mode].iterations);
        CHECK(info[0].primal_objective == info[mode].primal_objective);
        CHECK(info[0].dual_objective == info[mode].dual_objective);
        for (i64 g = 0; g < sz; ++g) {
            CHECK(x[0][g] == x[mode][g]);
            CHECK(y[0][g] == y[mode][g]);
            CHECK(z[0][g] == z[mode][g]);
        }
    }

    /* and all must still track the CPU reference solver */
    ref::Result r = ref::solve(in, p);
    CHECK(r.gap <= tol);
    CHECK(std::llabs((long long)(info[0].iterations - r.iterations)) <=
          (long long)p.op_steps);
    double denom =
        1.0 + std::fabs(r.primal) + std::fabs(info[0].primal_objective);
    CHECK(std::fabs(info[0].primal_objective - r.primal) / denom <=
          4.0 * tol);
}

/* The persistent megakernel runs the same per-row device code as the
 * launched-kernel path, but inlining differences let nvcc contract FMAs
 * differently, so the two trajectories agree only to rounding: compare
 * solver-level results with tolerances. Two mega runs, however, must be
 * bitwise identical - any grid-barrier race would break that. */
static void run_mega_equivalence(uint64_t seed, i64 n, i64 t, double tol,
                                 int op_steps) {
    ref::Instance in = ref::random_instance(seed, n, t);
    pria_params p = pria_params_auto(-1, tol);
    if (op_steps > 0) p.op_steps = op_steps;
    const i64 sz = n * t * t;
    vector<double> x[3], y[3], z[3];
    pria_result_info info[3];

    /* 0: mega, 1: launched kernels, 2: mega again (determinism) */
    for (int m = 0; m < 3; ++m) {
        pria::mega_mode() = m == 1 ? 0 : 1;
        x[m].resize(sz);
        y[m].resize(sz);
        z[m].resize(sz);
        CHECK(pria_solve_f64(n, t, in.k.data(), in.k0, in.h.data(), &p,
                             nullptr, nullptr, nullptr, x[m].data(),
                             y[m].data(), z[m].data(), &info[m]) == 0);
    }
    pria::mega_mode() = -1;

    CHECK(info[0].iterations == info[2].iterations);
    CHECK(info[0].primal_objective == info[2].primal_objective);
    CHECK(info[0].dual_objective == info[2].dual_objective);
    for (i64 g = 0; g < sz; ++g) {
        CHECK(x[0][g] == x[2][g]);
        CHECK(y[0][g] == y[2][g]);
        CHECK(z[0][g] == z[2][g]);
    }

    CHECK(std::llabs((long long)(info[0].iterations - info[1].iterations)) <=
          2 * (long long)p.op_steps);
    double denom = 1.0 + std::fabs(info[0].primal_objective) +
                   std::fabs(info[1].primal_objective);
    CHECK(std::fabs(info[0].primal_objective - info[1].primal_objective) /
              denom <= 4.0 * tol);
    for (i64 g = 0; g < sz; ++g) {
        CHECK_NEAR(x[0][g], x[1][g], 100.0 * tol);
        CHECK_NEAR(y[0][g], y[1][g], 100.0 * tol);
        CHECK_NEAR(z[0][g], z[1][g], 100.0 * tol);
    }
}

/* Batched solves must be bitwise identical to solo solves (batch workers
 * never use the megakernel, so the solo reference runs with mega off). */
static void run_batch_equivalence(uint64_t seed, i64 count, i64 n, i64 t,
                                  double tol) {
    const int mega_prev = pria::mega_mode();
    pria::mega_mode() = 0;
    struct Restore {
        int prev;
        ~Restore() { pria::mega_mode() = prev; }
    } restore{mega_prev};
    pria_params p = pria_params_auto(-1, tol);
    const i64 sz = n * t * t;
    vector<double> k(count * n), k0(count), h(count * sz);
    vector<ref::Instance> ins;
    for (i64 b = 0; b < count; ++b) {
        ins.push_back(ref::random_instance(seed + b, n, t));
        std::copy(ins[b].k.begin(), ins[b].k.end(), k.begin() + b * n);
        k0[b] = ins[b].k0;
        std::copy(ins[b].h.begin(), ins[b].h.end(), h.begin() + b * sz);
    }
    vector<double> bx(count * sz), by(count * sz), bz(count * sz);
    vector<pria_result_info> binfo(count);
    CHECK(pria_solve_batch_f64(count, n, t, k.data(), k0.data(), h.data(), &p,
                               bx.data(), by.data(), bz.data(),
                               binfo.data()) == 0);
    for (i64 b = 0; b < count; ++b) {
        vector<double> sx(sz), sy(sz), sz2(sz);
        pria_result_info si;
        CHECK(pria_solve_f64(n, t, ins[b].k.data(), ins[b].k0,
                             ins[b].h.data(), &p, nullptr, nullptr, nullptr,
                             sx.data(), sy.data(), sz2.data(), &si) == 0);
        CHECK(si.iterations == binfo[b].iterations);
        CHECK(si.primal_objective == binfo[b].primal_objective);
        CHECK(si.dual_objective == binfo[b].dual_objective);
        for (i64 g = 0; g < sz; ++g) {
            CHECK(bx[b * sz + g] == sx[g]);
            CHECK(by[b * sz + g] == sy[g]);
            CHECK(bz[b * sz + g] == sz2[g]);
        }
    }
}

int main() {
    /* small instances, compared against the CPU reference solver */
    run_case_f64(101, 1, 1, 1e-4, 1, true);
    run_case_f64(102, 2, 3, 1e-4, 1, true);
    run_case_f64(103, 3, 8, 1e-4, 1, true);
    run_case_f64(104, 5, 7, 1e-4, 1, true);   /* odd t: padding */
    run_case_f64(105, 4, 12, 1e-4, 1, true);
    run_case_f64(106, 2, 5, 1e-4, 0, true);   /* vanilla PDHG path */

    /* single item, zero joint cost (degenerate z) */
    run_case_f64(107, 1, 9, 1e-4, 1, true);

    /* larger, halpern, tighter tolerance */
    run_case_f64(108, 8, 24, 1e-6, 1, false);
    run_case_f64(109, 20, 30, 1e-4, 1, false);

    /* f32 */
    run_case_f32(110, 3, 8, 1e-4);
    run_case_f32(111, 10, 20, 1e-4);
    run_case_f32(112, 30, 40, 1e-4);

    /* batched == solo, bitwise */
    run_batch_equivalence(120, 5, 8, 14, 1e-4);

    /* graph path == plain path, bitwise (restarts exercise both parities) */
    run_graphs_equivalence(114, 12, 26, 1e-5, 1, 0);
    run_graphs_equivalence(115, 6, 15, 1e-4, 0, 0);  /* vanilla, odd t */
    run_graphs_equivalence(116, 5, 12, 1e-4, 1, 7);  /* odd op_steps */
    run_graphs_equivalence(117, 4, 10, 1e-4, 0, 5);  /* odd, vanilla */

    /* both projection backends (sort-free Michelot is the default, sort
     * pipelines behind PRIA_SORTFREE=0): tri-mode bitwise equivalence,
     * batch == solo, and reference accuracy must hold in each (the two
     * backends' trajectories differ at last-ulp level, correctness may not) */
    const int sf_default = pria::sortfree_mode();
    for (int sf = 0; sf < 2; ++sf) {
        pria::sortfree_mode() = sf;
        run_graphs_equivalence(114, 12, 26, 1e-5, 1, 0);
        run_graphs_equivalence(116, 5, 12, 1e-4, 1, 7);
        run_batch_equivalence(121, 5, 8, 14, 1e-4);
        run_case_f64(107, 1, 9, 1e-4, 1, true); /* degenerate z */
    }
    /* sort-free with the reflected term materialized (rfuse off) */
    pria::sortfree_mode() = 1;
    pria::rfuse_mode() = 0;
    run_graphs_equivalence(114, 12, 26, 1e-5, 1, 0);
    pria::rfuse_mode() = 1;

    /* megakernel == launched kernels, bitwise */
    run_mega_equivalence(130, 12, 26, 1e-5, 0);
    run_mega_equivalence(131, 5, 12, 1e-4, 7);
    pria::sortfree_mode() = sf_default;

    /* max_iter is respected */
    {
        ref::Instance in = ref::random_instance(113, 6, 18);
        pria_params p = pria_params_auto(64, 1e-12);
        pria_result_info info;
        CHECK(pria_solve_f64(6, 18, in.k.data(), in.k0, in.h.data(), &p,
                             nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, &info) == 0);
        CHECK(info.iterations == 64);
    }

    test_done("test_solver");
    return 0;
}
