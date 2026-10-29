/* PRAQ CUDA solver - public interface.
 *
 * The JAX implementation in ../praq-jax is the specification: this port
 * reproduces its bounds to the tolerance it is asked for.  The instance is
 * taken sparse -- the edge list given here is the half-edge set the solver
 * iterates on, so memory and work are O(n + m) rather than O(n^2), which is
 * what lets an instance of tens of millions of edges fit on one card.
 */
#ifndef PRAQ_PRAQ_H
#define PRAQ_PRAQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A QUBO in minimisation form: min_{x in {0,1}^n} x^T W x, with W symmetric
 * given as its diagonal plus its strict upper triangle in COO. */
typedef struct {
    int64_t n;              /* binary variables                              */
    int64_t m;              /* off-diagonal nonzeros (edges), i < j          */
    const double* diag;     /* [n]                                           */
    const int64_t* rows;    /* [m]                                           */
    const int64_t* cols;    /* [m]                                           */
    const double* vals;     /* [m], W_ij = W_ji                              */
} PraqInstance;

typedef struct {
    double tolerance;       /* relative duality gap at termination           */
    int64_t max_iter;       /* <= 0 for no cap, as praq-jax's max_iter=None  */
    int use_float32;        /* 0 = float64; float32 caps accuracy near 1e-4  */

    /* The restart schedule: praq-jax's Parameters, less the step size, which
     * ||A||_2 = 1 fixes.  praq_default_parameters() fills these with
     * with_auto_step's values; scripts/tune_praq.py is what varies them. */
    double omega;           /* initial primal weight                         */
    double beta_sufficient; /* restart once the residual fell to this ratio  */
    double beta_necessary;  /* ... or to this ratio and then rose again      */
    double beta_artificial; /* ... or after this share of all iterations     */
    double theta_initial;   /* smoothing exponent of the first omega update  */
    double theta_target;    /* what the exponent decays toward               */
    double theta_decay;     /* share of the remaining gap kept per update    */
    int64_t op_steps;       /* iterations between checks, at least 2; <= 0
                             * derives it from n by the rule below           */

    /* The check-interval rule, used when op_steps <= 0 (praq_check_interval):
     *
     *   interval(n) = max(2, round(offset + scale * B(ln(n / 50000), power)))
     *   B(u, p)     = (exp(p u) - 1) / p,  or u itself when p = 0
     *
     * B is the Box-Cox family: p = 0 is a logarithm in n, p = 1/2 a square
     * root, p = 1 linear, so one continuous parameter spans the growth laws.
     * A sweep of the interval by size found the best
     * interval rising from 2 at 100k nodes to 16 at 2M and staying there to
     * 8M, about three per doubling; the defaults are that logarithm. */
    double interval_offset;
    double interval_scale;
    double interval_power;
} PraqParameters;

typedef struct {
    double lower;           /* valid lower bound on the QUBO optimum         */
    double upper;           /* relaxation value at the returned primal       */
    int64_t iterations;
    int status;             /* 0 = converged, 1 = iteration limit, <0 error  */
} PraqResult;

/* Defaults matching praq-jax's Parameters.with_auto_step.  Start from these
 * and override fields, so that a field added later keeps its default. */
PraqParameters praq_default_parameters(void);

/* The check interval the rule gives for n variables (see PraqParameters);
 * praq_solve uses it when op_steps <= 0.  praq-jax's check_interval is the
 * same arithmetic, so the two solvers take the same steps. */
int64_t praq_check_interval(int64_t n, const PraqParameters* parameters);

/* Solve the LP relaxation (the roof dual).  `x` receives the primal, length n.
 *
 * Returns `result->status`: 0 when the relative gap (upper - lower) /
 * (1 + |lower| + |upper|) reached `tolerance`, 1 when `max_iter` ran out first
 * -- the bounds and `x` are still valid then, both iterates being feasible at
 * every step -- and a negative code (-1 argument, -2 instance, -3 CUDA) when
 * nothing was solved.  Rows and columns must be in range and distinct. */
int praq_solve(const PraqInstance* instance, const PraqParameters* parameters,
               double* x, PraqResult* result);

#ifdef __cplusplus
}
#endif
#endif /* PRAQ_PRAQ_H */
