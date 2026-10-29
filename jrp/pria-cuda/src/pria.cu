/* pria.cu - public C entry points for the PRIA CUDA solver. */
#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "solver.cuh"

namespace {

/* Batch = plain thread fan-out over independent solves: each worker owns its
 * contexts (streams, graphs), so the per-instance work overlaps on the GPU
 * while every instance computes exactly what a solo solve would. */
template <typename real>
int solve_batch(int64_t count, int64_t n, int64_t t, const real* k,
                const real* k0, const real* h, const pria_params* params,
                real* x, real* y, real* z, pria_result_info* infos) {
    if (count <= 0) return (int)cudaErrorInvalidValue;
    const int64_t sz = n * t * t;
    const int nworkers = (int)std::min<int64_t>(count, 16);
    std::atomic<int64_t> next(0);
    std::atomic<int> err(0);

    auto worker = [&]() {
        /* concurrent instances must not run persistent full-GPU grids
         * against each other; restore for later solo solves on this thread */
        pria::batch_worker() = true;
        for (int64_t b; (b = next.fetch_add(1)) < count;) {
            cudaError_t st = pria::solve_impl<real>(
                n, t, k + b * n, k0[b], h + b * sz, params, nullptr, nullptr,
                nullptr, x ? x + b * sz : nullptr, y ? y + b * sz : nullptr,
                z ? z + b * sz : nullptr, infos ? infos + b : nullptr);
            if (st != cudaSuccess) err.store((int)st);
        }
        pria::batch_worker() = false;
    };

    std::vector<std::thread> threads;
    for (int w = 1; w < nworkers; ++w) threads.emplace_back(worker);
    worker();
    for (auto& th : threads) th.join();
    return err.load();
}

} /* namespace */

extern "C" {

pria_params pria_params_auto(int64_t max_iter, double tolerance) {
    /* Defaults mirror pria-jax/pria/solver.py Parameters.with_auto_step,
     * which owns them; update both together. beta_sufficient = 1/e is the
     * analytic optimum for sufficient-decay restarts in the canonical norm. */
    pria_params p;
    p.omega = 0.10835;
    p.eta = 0.999 / std::sqrt(2.0);
    p.beta_sufficient = std::exp(-1.0);
    p.beta_artificial = 0.2468;
    p.beta_necessary = 0.5706;
    p.theta_initial = 0.81930;
    p.theta_target = 0.42818;
    p.theta_decay = 0.90474;
    p.tolerance = tolerance;
    p.max_iter = max_iter;
    p.use_halpern = 1;
    p.warm_steps = 0;
    p.op_steps = 16;
    p.verbose = 0;
    return p;
}

int pria_solve_f64(int64_t n, int64_t t, const double* order_costs,
                   double joint_cost, const double* holding_costs,
                   const pria_params* params, const double* warm_x,
                   const double* warm_y, const double* warm_z, double* x,
                   double* y, double* z, pria_result_info* info) {
    return (int)pria::solve_impl<double>(n, t, order_costs, joint_cost,
                                         holding_costs, params, warm_x, warm_y,
                                         warm_z, x, y, z, info);
}

int pria_solve_f32(int64_t n, int64_t t, const float* order_costs,
                   float joint_cost, const float* holding_costs,
                   const pria_params* params, const float* warm_x,
                   const float* warm_y, const float* warm_z, float* x, float* y,
                   float* z, pria_result_info* info) {
    return (int)pria::solve_impl<float>(n, t, order_costs, joint_cost,
                                        holding_costs, params, warm_x, warm_y,
                                        warm_z, x, y, z, info);
}

int pria_solve_batch_f64(int64_t count, int64_t n, int64_t t,
                         const double* order_costs, const double* joint_costs,
                         const double* holding_costs,
                         const pria_params* params, double* x, double* y,
                         double* z, pria_result_info* infos) {
    return solve_batch<double>(count, n, t, order_costs, joint_costs,
                               holding_costs, params, x, y, z, infos);
}

int pria_solve_batch_f32(int64_t count, int64_t n, int64_t t,
                         const float* order_costs, const float* joint_costs,
                         const float* holding_costs,
                         const pria_params* params, float* x, float* y,
                         float* z, pria_result_info* infos) {
    return solve_batch<float>(count, n, t, order_costs, joint_costs,
                              holding_costs, params, x, y, z, infos);
}

double pria_objective_f64(int64_t n, int64_t t, const double* order_costs,
                          double joint_cost, const double* holding_costs,
                          const double* x) {
    double obj = 0.0;
    for (int64_t i = 0; i < n; ++i)
        for (int64_t s = 0; s < t; ++s) {
            double m = 0.0;
            for (int64_t tt = 0; tt < t; ++tt)
                m = std::max(m, x[(i * t + s) * t + tt]);
            obj += order_costs[i] * m;
        }
    for (int64_t s = 0; s < t; ++s) {
        double m = 0.0;
        for (int64_t i = 0; i < n; ++i)
            for (int64_t tt = 0; tt < t; ++tt)
                m = std::max(m, x[(i * t + s) * t + tt]);
        obj += joint_cost * m;
    }
    for (int64_t i = 0; i < n; ++i)
        for (int64_t s = 0; s < t; ++s)
            for (int64_t tt = s; tt < t; ++tt)
                obj += holding_costs[(i * t + s) * t + tt] *
                       x[(i * t + s) * t + tt];
    return obj;
}

} /* extern "C" */
