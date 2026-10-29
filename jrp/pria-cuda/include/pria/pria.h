/* pria.h - C API for PRIA, a CUDA solver for the Joint Replenishment
 * Problem (JRP) LP.
 *
 * PRIA solves the LP relaxation of the joint replenishment problem
 *
 *   minimize    sum_s K0*z_s + sum_{i,s} K_i*y[i][s]
 *                + sum_{i,t} sum_{s<=t} H[i][s][t] * x[i][s][t]
 *   subject to  sum_{s<=t} x[i][s][t] = 1          for all i, t
 *               x[i][s][t] <= y[i][s]              for all i, s <= t
 *               x[i][s][t] <= z[s]                 for all i, s <= t
 *               x, y, z >= 0
 *
 * with a restarted Halpern-accelerated reflected primal-dual hybrid gradient
 * method, matching the pria-jax implementation of PRIA. All
 * per-segment sorting inside the simplex projections runs on the GPU
 * through CUB.
 *
 * Indexing: H is a dense row-major [n][t][t] array; only the upper-triangular
 * part (s <= t) is read. Solutions x, y, z are written as dense row-major
 * [n][t][t] arrays that are upper triangular. The returned y and z are the
 * dual iterates normalized by K_i and K0 respectively (rows of y sum to 1
 * over t >= s, and each z group {i, t >= s} sums to 1), exactly like the JAX
 * solver's outputs.
 */
#ifndef PRIA_CUDA_PRIA_H
#define PRIA_CUDA_PRIA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pria_params {
    double omega;            /* initial primal weight */
    double eta;              /* step size */
    double beta_sufficient;  /* sufficient-decay restart threshold */
    double beta_artificial;  /* artificial restart threshold */
    double beta_necessary;   /* necessary restart threshold */
    double theta_initial;    /* initial primal-weight smoothing exponent */
    double theta_target;     /* exponent theta decays toward */
    double theta_decay;      /* fraction of remaining theta gap kept per update */
    double tolerance;        /* relative optimality tolerance */
    int64_t max_iter;        /* maximum operator steps, -1 for unlimited */
    int use_halpern;         /* nonzero: restarted Halpern; zero: vanilla PDHG */
    int warm_steps;          /* vanilla op_steps blocks to run before Halpern */
    int op_steps;            /* operator steps between convergence checks */
    int verbose;             /* nonzero: print per-check progress to stderr */
} pria_params;

/* Defaults matching Parameters.with_auto_step in the JAX implementation. */
pria_params pria_params_auto(int64_t max_iter, double tolerance);

typedef struct pria_result_info {
    int64_t iterations;      /* operator steps performed */
    double primal_objective; /* implied primal objective at the final iterate */
    double dual_objective;   /* implied dual objective at the final iterate */
    double relative_gap;     /* (primal - dual) / (1 + |primal| + |dual|) */
} pria_result_info;

/* Solve one instance in the given precision.
 *
 *   n, t           - number of items / time periods
 *   order_costs    - [n] item order costs K_i
 *   joint_cost     - joint order cost K0
 *   holding_costs  - [n*t*t] holding costs H (upper-triangular part used)
 *   warm_x/y/z     - optional warm start, [n*t*t] each; pass NULL for zeros
 *                    (the JAX default). Warm duals are NORMALIZED, i.e. in
 *                    the same convention as the returned y and z, so feeding
 *                    a returned solution back in warm-starts at that point.
 *   x, y, z        - output buffers, [n*t*t] each (any may be NULL to skip)
 *   info           - optional result info (may be NULL)
 *
 * Returns 0 on success, a nonzero CUDA error code otherwise. */
int pria_solve_f64(int64_t n, int64_t t,
                   const double* order_costs, double joint_cost,
                   const double* holding_costs, const pria_params* params,
                   const double* warm_x, const double* warm_y, const double* warm_z,
                   double* x, double* y, double* z, pria_result_info* info);

int pria_solve_f32(int64_t n, int64_t t,
                   const float* order_costs, float joint_cost,
                   const float* holding_costs, const pria_params* params,
                   const float* warm_x, const float* warm_y, const float* warm_z,
                   float* x, float* y, float* z, pria_result_info* info);

/* Solve a batch of same-shaped instances concurrently, one CUDA stream (and,
 * by default, one whole-solve graph) per instance, so small instances that
 * cannot fill the GPU individually overlap. Inputs/outputs are packed
 * contiguously: order_costs [count*n], joint_costs [count], holding_costs
 * [count*n*t*t]; x, y, z [count*n*t*t] each (any may be NULL); infos [count]
 * (may be NULL). No warm starts. Results are bitwise identical to solving
 * each instance alone. Returns 0 if every instance succeeded, else the CUDA
 * error code of a failing instance. */
int pria_solve_batch_f64(int64_t count, int64_t n, int64_t t,
                         const double* order_costs, const double* joint_costs,
                         const double* holding_costs,
                         const pria_params* params, double* x, double* y,
                         double* z, pria_result_info* infos);

int pria_solve_batch_f32(int64_t count, int64_t n, int64_t t,
                         const float* order_costs, const float* joint_costs,
                         const float* holding_costs,
                         const pria_params* params, float* x, float* y,
                         float* z, pria_result_info* infos);

/* Objective value of a primal solution x on the ORIGINAL problem:
 * sum_{i,s} K_i max_t x[i][s][t] + K0 sum_s max_{i,t} x[i][s][t] + <H, x>.
 * Computed on the host in double precision. */
double pria_objective_f64(int64_t n, int64_t t, const double* order_costs,
                          double joint_cost, const double* holding_costs,
                          const double* x);

#ifdef __cplusplus
}
#endif

#endif /* PRIA_CUDA_PRIA_H */
