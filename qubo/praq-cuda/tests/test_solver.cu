/* Tests for the PRAQ CUDA solver, against an independent reference.
 *
 * The reference needs no LP solver: the relaxation's vertices are half-integral
 * (roof duality), so on a tiny instance its optimum is the minimum of F(x) over
 * {0, 1/2, 1}^n, which is enumerated here.  F itself is recomputed on the host
 * from the dense definition in QuboProblem.relaxation_value, so it does not
 * share code with the half-edge form the solver uses.
 *
 * The instances mirror harness.instances.random_instance: integer entries in
 * +-[1, 3], off-diagonal ones kept with probability `density`, so that sparse
 * and dense builds are both exercised, down to isolated nodes and no edges.
 *
 * Run with `make test`; exit status is the number of failed checks.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "praq/praq.h"

static int failures = 0;

#define CHECK(condition, ...)                                        \
    do {                                                             \
        if (!(condition)) {                                          \
            ++failures;                                              \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);            \
            printf(__VA_ARGS__);                                     \
            printf("\n");                                            \
        }                                                            \
    } while (0)

struct Instance {
    int64_t n;
    std::vector<double> diag, vals;
    std::vector<int64_t> rows, cols;

    PraqInstance view() const {
        PraqInstance out;
        out.n = n;
        out.m = int64_t(vals.size());
        out.diag = diag.data();
        out.rows = rows.data();
        out.cols = cols.data();
        out.vals = vals.data();
        return out;
    }
};

static Instance random_instance(unsigned seed, int64_t n, double density) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> magnitude(1, 3);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    auto value = [&]() { return double(magnitude(rng)) * (uniform(rng) < 0.5 ? -1.0 : 1.0); };

    Instance out;
    out.n = n;
    for (int64_t i = 0; i < n; ++i) out.diag.push_back(value());
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            if (uniform(rng) < density) {
                out.rows.push_back(i);
                out.cols.push_back(j);
                out.vals.push_back(value());
            }
        }
    }
    return out;
}

/* F(x) of QuboProblem.relaxation_value: the linear term, plus per edge
 * 2 W_ij max(x_i + x_j - 1, 0) when W_ij > 0 and 2 W_ij min(x_i, x_j) when
 * W_ij < 0 -- the factor 2 because the dense sum visits each edge twice. */
static double relaxation_value(const Instance& in, const std::vector<double>& x) {
    double value = 0.0;
    for (int64_t i = 0; i < in.n; ++i) value += in.diag[i] * x[i];
    for (size_t e = 0; e < in.vals.size(); ++e) {
        double xi = x[in.rows[e]], xj = x[in.cols[e]], w = 2.0 * in.vals[e];
        value += w > 0.0 ? w * std::fmax(xi + xj - 1.0, 0.0) : w * std::fmin(xi, xj);
    }
    return value;
}

/* min F over {0, 1/2, 1}^n: the relaxation optimum, by half-integrality. */
static double half_integral_optimum(const Instance& in) {
    std::vector<double> x(size_t(in.n), 0.0);
    double best = INFINITY;
    int64_t total = 1;
    for (int64_t i = 0; i < in.n; ++i) total *= 3;
    for (int64_t code = 0; code < total; ++code) {
        int64_t rest = code;
        for (int64_t i = 0; i < in.n; ++i) { x[i] = 0.5 * double(rest % 3); rest /= 3; }
        best = std::fmin(best, relaxation_value(in, x));
    }
    return best;
}

struct Solved {
    int status;
    PraqResult result;
    std::vector<double> x;
};

static Solved solve(const Instance& in, double tolerance, int64_t max_iter, int use_float32) {
    PraqParameters p = praq_default_parameters();
    p.tolerance = tolerance;
    p.max_iter = max_iter;
    p.use_float32 = use_float32;
    Solved out;
    out.x.assign(size_t(in.n), -1.0);
    PraqInstance view = in.view();
    out.status = praq_solve(&view, &p, out.x.data(), &out.result);
    return out;
}

static double relative_gap(const PraqResult& r) {
    return (r.upper - r.lower) / (1.0 + std::fabs(r.upper) + std::fabs(r.lower));
}

static void test_converges_to_the_half_integral_optimum() {
    printf("converges to the enumerated optimum, float64\n");
    const double tolerance = 1e-9;
    const double densities[] = {1.0, 0.5, 0.2, 0.0};
    unsigned seed = 0;
    for (int64_t n = 1; n <= 8; ++n) {
        for (double density : densities) {
            Instance in = random_instance(seed++, n, density);
            double optimum = half_integral_optimum(in);
            Solved s = solve(in, tolerance, -1, 0);
            double scale = 1.0 + std::fabs(optimum);

            CHECK(s.status == 0 && s.result.status == 0,
                  "n=%lld d=%.1f status %d", (long long)n, density, s.status);
            CHECK(relative_gap(s.result) <= tolerance,
                  "n=%lld d=%.1f gap %.3e", (long long)n, density, relative_gap(s.result));
            CHECK(s.result.lower <= optimum + 1e-9 * scale && s.result.upper >= optimum - 1e-9 * scale,
                  "n=%lld d=%.1f bounds [%.9f, %.9f] miss %.9f",
                  (long long)n, density, s.result.lower, s.result.upper, optimum);
            CHECK(std::fabs(s.result.upper - optimum) <= 1e-7 * scale,
                  "n=%lld d=%.1f upper %.9f vs optimum %.9f",
                  (long long)n, density, s.result.upper, optimum);

            bool boxed = true;
            for (double xi : s.x) boxed = boxed && xi >= 0.0 && xi <= 1.0;
            CHECK(boxed, "n=%lld d=%.1f primal outside [0, 1]", (long long)n, density);
            double value = relaxation_value(in, s.x);
            CHECK(std::fabs(value - s.result.upper) <= 1e-9 * scale,
                  "n=%lld d=%.1f F(x) %.12f is not the reported upper %.12f",
                  (long long)n, density, value, s.result.upper);
        }
    }
}

static void test_bounds_bracket_the_optimum_at_every_budget() {
    /* Both iterates stay feasible, so L_D <= OPT <= L_P has to hold at every
     * point of the run, not only at convergence. */
    printf("bounds bracket the optimum under every iteration cap\n");
    Instance in = random_instance(101, 8, 0.6);
    double optimum = half_integral_optimum(in);
    double scale = 1.0 + std::fabs(optimum);
    const int64_t budgets[] = {1, 4, 16, 64, 256};
    for (int64_t budget : budgets) {
        Solved s = solve(in, 1e-12, budget, 0);
        CHECK(s.status >= 0, "cap %lld status %d", (long long)budget, s.status);
        CHECK(s.result.iterations <= budget + 15,
              "cap %lld ran %lld iterations", (long long)budget, (long long)s.result.iterations);
        CHECK(s.result.lower <= optimum + 1e-9 * scale,
              "cap %lld lower %.9f above optimum %.9f", (long long)budget, s.result.lower, optimum);
        CHECK(s.result.upper >= optimum - 1e-9 * scale,
              "cap %lld upper %.9f below optimum %.9f", (long long)budget, s.result.upper, optimum);
        CHECK(s.result.status == 1 || relative_gap(s.result) <= 1e-12,
              "cap %lld reports convergence at gap %.3e", (long long)budget, relative_gap(s.result));
    }
}

static void test_float32_reaches_its_own_accuracy() {
    printf("float32 iterates reach the accuracy float32 allows\n");
    unsigned seed = 200;
    for (int64_t n = 4; n <= 8; n += 2) {
        Instance in = random_instance(seed++, n, 0.7);
        double optimum = half_integral_optimum(in);
        Solved s = solve(in, 1e-4, -1, 1);
        double scale = 1.0 + std::fabs(optimum);
        CHECK(s.status == 0, "n=%lld status %d", (long long)n, s.status);
        CHECK(std::fabs(s.result.upper - optimum) <= 1e-3 * scale
              && std::fabs(s.result.lower - optimum) <= 1e-3 * scale,
              "n=%lld bounds [%.6f, %.6f] vs optimum %.6f",
              (long long)n, s.result.lower, s.result.upper, optimum);
    }
}

static void test_larger_instance_certifies_itself() {
    /* Beyond enumeration; what can be checked is the certificate: a gap
     * within tolerance between a bound that is F of the returned point and a
     * bound that is a dual value, both at the original scale. */
    printf("a larger sparse instance certifies its own gap\n");
    Instance in = random_instance(300, 400, 0.05);
    Solved s = solve(in, 1e-8, -1, 0);
    CHECK(s.status == 0, "status %d after %lld iterations", s.status, (long long)s.result.iterations);
    CHECK(relative_gap(s.result) <= 1e-8, "gap %.3e", relative_gap(s.result));
    double value = relaxation_value(in, s.x);
    CHECK(std::fabs(value - s.result.upper) <= 1e-9 * (1.0 + std::fabs(value)),
          "F(x) %.12f vs upper %.12f", value, s.result.upper);
    CHECK(s.result.lower <= s.result.upper, "lower %.9f above upper %.9f",
          s.result.lower, s.result.upper);
}

static void test_rejects_what_it_cannot_solve() {
    printf("rejects bad arguments and bad instances\n");
    Instance in = random_instance(1, 3, 1.0);
    PraqParameters p = praq_default_parameters();
    PraqResult result;
    std::vector<double> x(3);
    PraqInstance view = in.view();

    CHECK(praq_solve(nullptr, &p, x.data(), &result) < 0, "null instance accepted");
    CHECK(praq_solve(&view, &p, nullptr, &result) < 0, "null primal accepted");

    Instance loop = in;
    loop.cols[0] = loop.rows[0];
    view = loop.view();
    CHECK(praq_solve(&view, &p, x.data(), &result) == -2, "a self-loop was accepted");

    Instance out_of_range = in;
    out_of_range.cols[0] = 3;
    view = out_of_range.view();
    CHECK(praq_solve(&view, &p, x.data(), &result) == -2, "an index past n was accepted");

    Instance empty = in;
    empty.n = 0;
    empty.diag.clear();
    view = empty.view();
    CHECK(praq_solve(&view, &p, x.data(), &result) == -2, "n = 0 was accepted");
}

static void test_interval_rule() {
    printf("the check-interval rule and op_steps <= 0\n");
    PraqParameters p = praq_default_parameters();
    /* The defaults are 3 log2(n / 50k): 50k -> 0 -> floored at 2; 400k -> 9; 2M -> 16. */
    CHECK(praq_check_interval(50000, &p) == 2, "interval at 50k is %lld", (long long)praq_check_interval(50000, &p));
    CHECK(praq_check_interval(400000, &p) == 9, "interval at 400k is %lld", (long long)praq_check_interval(400000, &p));
    CHECK(praq_check_interval(2000000, &p) == 16, "interval at 2M is %lld", (long long)praq_check_interval(2000000, &p));
    /* power 1/2 with the same scale: a square root, 50k * e^2 -> scale * 2 (e - 1) = 14.9 -> 15. */
    p.interval_power = 0.5;
    CHECK(praq_check_interval(int64_t(50000.0 * 7.38905609893065), &p) == 15,
          "square-root branch gives %lld", (long long)praq_check_interval(int64_t(50000.0 * 7.38905609893065), &p));
    /* op_steps <= 0 solves with the derived interval, which at n = 60 is 2. */
    Instance in = random_instance(3, 60, 1.0);
    PraqParameters q = praq_default_parameters();
    q.tolerance = 1e-8;
    q.op_steps = 0;
    PraqResult result;
    std::vector<double> x(60);
    PraqInstance view = in.view();
    int status = praq_solve(&view, &q, x.data(), &result);
    CHECK(status == 0, "op_steps = 0 returned %d", status);
    CHECK(result.iterations % 2 == 0 && result.iterations > 0,
          "with an interval of 2 the count should be a positive multiple of 2, got %lld",
          (long long)result.iterations);
}

int main() {
    test_interval_rule();
    test_converges_to_the_half_integral_optimum();
    test_bounds_bracket_the_optimum_at_every_budget();
    test_float32_reaches_its_own_accuracy();
    test_larger_instance_certifies_itself();
    test_rejects_what_it_cannot_solve();
    if (failures) {
        printf("%d check(s) failed\n", failures);
    } else {
        printf("test_solver: all tests passed\n");
    }
    return failures;
}
