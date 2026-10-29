/* PRAQ CUDA solver: the public entry points.  The algorithm is in solver.cuh. */
#include "praq/praq.h"
#include "solver.cuh"

#include <cmath>
#include <limits>

extern "C" PraqParameters praq_default_parameters(void) {
    PraqParameters p;
    /* praq-jax's with_auto_step has no default tolerance; this is the
     * harness's Tolerances.optimality, the gap every run here asks for. */
    p.tolerance = 1e-8;
    p.max_iter = -1;
    p.use_float32 = 0;
    p.omega = praq::kOmegaInitial;
    p.beta_sufficient = praq::kBetaSufficient;
    p.beta_necessary = praq::kBetaNecessary;
    p.beta_artificial = praq::kBetaArtificial;
    p.theta_initial = praq::kThetaInitial;
    p.theta_target = praq::kThetaTarget;
    p.theta_decay = praq::kThetaDecay;
    p.op_steps = praq::kOpSteps;
    p.interval_offset = praq::kIntervalOffset;
    p.interval_scale = praq::kIntervalScale;
    p.interval_power = praq::kIntervalPower;
    return p;
}

extern "C" int64_t praq_check_interval(int64_t n, const PraqParameters* parameters) {
    if (!parameters || n <= 0) return 2;
    return praq::check_interval(n, parameters->interval_offset, parameters->interval_scale,
                                parameters->interval_power);
}

extern "C" int praq_solve(const PraqInstance* instance,
                          const PraqParameters* parameters,
                          double* x, PraqResult* result) {
    if (result) {
        result->lower = std::numeric_limits<double>::quiet_NaN();
        result->upper = std::numeric_limits<double>::quiet_NaN();
        result->iterations = 0;
        result->status = praq::kErrorArgument;
    }
    if (!instance || !parameters || !x || !result) return praq::kErrorArgument;
    PraqParameters p = *parameters;
    if (!(p.tolerance >= 0.0) || !(p.omega > 0.0) || !(p.beta_sufficient > 0.0)
        || !(p.beta_necessary > 0.0) || !(p.beta_artificial >= 0.0)
        || !std::isfinite(p.theta_initial) || !std::isfinite(p.theta_target)
        || !std::isfinite(p.theta_decay) || !std::isfinite(p.interval_offset)
        || !std::isfinite(p.interval_scale) || !(p.interval_power >= 0.0)) {
        return praq::kErrorArgument;
    }

    praq::Structure structure;
    int status = structure.build(*instance);
    if (status) {
        result->status = status;
        return status;
    }

    /* <= 0 means no cap: praq-jax runs with max_iter=None and this matches
     * it.  A wall-clock budget is the caller's to enforce, as the harness does
     * around the file CLI.  op_steps <= 0 asks for the rule, as in praq-jax's
     * solve_qubo_problem_warm_start; otherwise it is floored at 2 as
     * Parameters.new does. */
    if (p.max_iter <= 0) p.max_iter = -1;
    if (p.op_steps <= 0) p.op_steps = praq_check_interval(instance->n, &p);
    if (p.op_steps < 2) p.op_steps = 2;

    if (p.use_float32) {
        praq::Solver<float> solver(structure, p);
        status = solver.run(x, result);
    } else {
        praq::Solver<double> solver(structure, p);
        status = solver.run(x, result);
    }
    if (status < 0) result->status = status;
    return status;
}
