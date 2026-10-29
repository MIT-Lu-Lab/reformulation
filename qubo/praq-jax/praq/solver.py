import math
from dataclasses import dataclass, replace
from functools import partial

import numpy as np

import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax import Array, lax, jit, config

# XLA tuning knobs.  jax_exec_time_optimization_effort and memory_fitting_effort
# were removed in jax 0.11.1 in favour of the CompilerEffortLevel enum, which is
# what jax_optimization_level below now takes -- so the effort request is still
# made, by its current name, and nothing was lost in the migration.
#
# These are set unguarded on purpose.  Swallowing an unrecognised option would
# silently drop the optimisation on a jax that renamed it; CI runs against an
# unpinned jax so the next rename fails loudly and gets migrated instead.
config.update('jax_optimization_level', 'O3')
config.update('eager_constant_folding', True)
config.update('jax_use_simplified_jaxpr_constants', True)

from .qubo_problem import QuboProblem
from .half_edge import HalfEdge
from .projection import project_primal, project_dual
from .halpern_operator import canonical_norm

# ||A||_2 for the PRAQ coupling. The half-edge lift is degree-preconditioned
# PDHG in x-space, and D^{-1/2} S^T S D^{-1/2} = I exactly, so this is 1 for
# every graph and every instance -- no line search, no power iteration, and no
# dependence on the maximum degree. PRIA's coupling X -> (X, X) has sqrt(2)
# instead. See the paper's bound on the coupling norm.
COUPLING_NORM = 1.0


#: The check-interval rule's defaults and reference size, see `check_interval`.
INTERVAL_NODES = 50_000.0
INTERVAL_OFFSET = 0.0
INTERVAL_SCALE = 4.328085122666891   # 3 / ln 2
INTERVAL_POWER = 0.0


data_fields = ["omega",
               "theta"]
meta_fields = ["eta",
               "beta_sufficient",
               "beta_artificial",
               "beta_necessary",
               "theta_target",
               "theta_decay",
               "max_iter",
               "use_halpern",
               "warm_steps",
               "op_steps",
               "tolerance",
               "interval_offset",
               "interval_scale",
               "interval_power"]
@partial(register_dataclass, data_fields=data_fields, meta_fields=meta_fields)
@dataclass(frozen=True)
class Parameters:
    omega: float
    eta: float
    beta_sufficient: float
    beta_artificial: float
    beta_necessary: float
    theta: float
    theta_target: float
    theta_decay: float
    max_iter: int
    use_halpern: bool
    warm_steps: int
    op_steps: int
    tolerance: float
    interval_offset: float
    interval_scale: float
    interval_power: float

    @classmethod
    def new(cls,
            omega: float,
            eta: float,
            beta_sufficient: float,
            beta_artificial: float,
            beta_necessary: float,
            theta_initial: float,
            theta_target: float,
            theta_decay: float,
            max_iter: int | None,
            use_halpern: bool,
            warm_steps: int,
            op_steps: int,
            tolerance: float,
            interval_offset: float = INTERVAL_OFFSET,
            interval_scale: float = INTERVAL_SCALE,
            interval_power: float = INTERVAL_POWER):
        """
        :param omega:           The primal (x) weight
        :param eta:             The step size; the metric is PSD iff eta <= 1
        :param beta_sufficient: The sufficient restart parameter
        :param beta_artificial: The artificial restart parameter
        :param beta_necessary:  The necessary restart parameter
        :param theta_initial:   The starting primal weight smoothing exponent (1.0 = one-shot ratio)
        :param theta_target:    The smoothing exponent theta decays toward
        :param theta_decay:     Fraction of the remaining theta gap kept per omega update
                                (0.0 = jump to theta_target after one update, 1.0 = constant theta_initial)
        :param max_iter:        The maximum iterations
        :param use_halpern:     Whether to use restarted Halpern
        :param warm_steps:      How many op_steps to run vanilla before starting the algorithm
        :param op_steps:        How many steps between each convergence and restart check;
                                <= 0 derives it from the instance size with `check_interval`
        :param tolerance:       The relative optimality tolerance
        :param interval_offset: The check-interval rule's constants, see `check_interval`
        :param interval_scale:
        :param interval_power:
        :return:                The solver parameters
        """
        if max_iter is None:
            max_iter = -1

        op_steps = max(op_steps, 2) if op_steps > 0 else 0
        warm_steps = max(warm_steps, 0)

        return cls(omega, eta, beta_sufficient, beta_artificial, beta_necessary,
                   theta_initial, theta_target, theta_decay, max_iter,
                   use_halpern, warm_steps, op_steps, tolerance,
                   float(interval_offset), float(interval_scale), float(interval_power))

    @classmethod
    def with_auto_step(cls,
                       max_iter: int | None,
                       tolerance: float,
                       use_halpern: bool = True,
                       op_steps: int = 16,
                       # omega = 1 minimizes lambda_max of the canonical metric
                       # (prop:praq-spectrum). The restart logic re-estimates it
                       # from the first anchor onward, so this only sets the
                       # warm phase.
                       omega: float = 1.0,
                       # 1/e minimizes the worst-case complexity of
                       # sufficient-decay restarts in the fixed-omega analysis;
                       # the theory is PRIA's and carries over unchanged, since
                       # the restart tests see the iteration only through the
                       # canonical-norm residual. beta_necessary/artificial and
                       # the theta schedule are PRIA's JRP-tuned values, taken
                       # as starting points and NOT tuned for QUBO.
                       beta_sufficient: float = float(np.exp(-1.0)),
                       beta_necessary: float = 0.5706,
                       beta_artificial: float = 0.2468,
                       theta_initial: float = 0.81930,
                       theta_target: float = 0.42818,
                       theta_decay: float = 0.90474,
                       warm_steps: int = 0,
                       # The check-interval rule, used when op_steps <= 0;
                       # see `check_interval`.
                       interval_offset: float = INTERVAL_OFFSET,
                       interval_scale: float = INTERVAL_SCALE,
                       interval_power: float = INTERVAL_POWER):
        eta = 0.999 / COUPLING_NORM

        return cls.new(omega, eta, beta_sufficient, beta_artificial,
                       beta_necessary, theta_initial, theta_target, theta_decay,
                       max_iter, use_halpern, warm_steps, op_steps, tolerance,
                       interval_offset, interval_scale, interval_power)

    @classmethod
    def stack(cls, parameters: list["Parameters"]) -> "Parameters":
        """
        Converts an array-of-structs `parameters` to a struct-of-arrays. Every
        field except omega and theta is inherited from the first element.

        :param parameters: a list of Parameters singletons to stack
        :return: a Parameters instance with 1D arrays for omega and theta
        """
        first = parameters[0]
        return cls(jnp.array([p.omega for p in parameters]),
                   first.eta,
                   first.beta_sufficient,
                   first.beta_artificial,
                   first.beta_necessary,
                   jnp.array([p.theta for p in parameters]),
                   first.theta_target,
                   first.theta_decay,
                   first.max_iter,
                   first.use_halpern,
                   first.warm_steps,
                   first.op_steps,
                   first.tolerance,
                   first.interval_offset,
                   first.interval_scale,
                   first.interval_power)

    def as_type(self, dtype: jnp.dtype) -> "Parameters":
        return replace(self,
                       omega=jnp.asarray(self.omega, dtype=dtype),
                       theta=jnp.asarray(self.theta, dtype=dtype))


def check_interval(n: int, offset: float = INTERVAL_OFFSET,
                   scale: float = INTERVAL_SCALE, power: float = INTERVAL_POWER) -> int:
    """
    The check interval (op_steps) the rule gives for an instance of `n` variables:

        max(2, round(offset + scale * B(ln(n / 50000), power)))

    with B(u, p) = (exp(p u) - 1) / p, or u itself at p = 0: the Box-Cox
    family, so p = 0 is a logarithm in n, p = 1/2 a square root and p = 1
    linear, and one continuous parameter spans the growth laws. The best interval
    grows with n, by about three per doubling, and the defaults are the
    logarithm fitted to that. praq-cuda's praq_check_interval
    is the same arithmetic, expression for expression, so both solvers take
    the same steps.

    :param n:      the number of variables
    :param offset: the rule's additive constant
    :param scale:  the rule's multiplier
    :param power:  the Box-Cox exponent, >= 0
    :return:       the interval, an int of at least 2
    """
    u = math.log(float(n) / INTERVAL_NODES)
    b = u if power == 0.0 else math.expm1(power * u) / power
    v = math.floor(offset + scale * b + 0.5)
    if not v >= 2.0:
        return 2
    return int(min(v, 1e9))


@register_dataclass
@dataclass(frozen=True)
class StepArguments:
    x: Array
    y: Array

    step_x: Array
    step_y: Array

    x0: Array
    y0: Array

    # error of current x0, y0
    fp_error: Array
    # fixed-point error at the previous restart/convergence check
    check_error: Array
    # number of iterations since last restart
    restart_iter: Array

    half_edge: HalfEdge
    shift: Array

    iteration: int
    parameters: Parameters

    terminate: Array

    # scalar for bound objective scaling
    objective_scale: Array


def _raw_step(x1: Array, y1: Array, half_edge: HalfEdge, shift: Array,
              omega: float, eta: float) -> tuple[Array, Array]:
    """One PDHG step on the PRAQ saddle form."""
    dtype = x1.dtype
    omega = jnp.array(omega).astype(dtype)
    eta = jnp.array(eta).astype(dtype)

    gradient = half_edge.diagonal - half_edge.scatter(y1)
    x2 = project_primal(x1 - (eta / omega) * (gradient / half_edge.degree))

    reflected = 2.0 * x2 - x1
    y2_inner = y1 + eta * omega * (shift - half_edge.gather(reflected))
    y2 = project_dual(y2_inner, half_edge.weights, half_edge.positive)

    return x2, y2


def _reflect_step(t: StepArguments, reflect: bool) -> StepArguments:
    x1, y1 = t.step_x, t.step_y
    omega, eta = t.parameters.omega, t.parameters.eta

    if reflect:
        restart_iter = t.restart_iter + 1

        left_weight = restart_iter / (restart_iter + 1.0)
        right_weight = 1.0 / (restart_iter + 1.0)

        x1 = left_weight * (2 * x1 - t.x) + right_weight * t.x0
        y1 = left_weight * (2 * y1 - t.y) + right_weight * t.y0

    x2, y2 = _raw_step(x1, y1, t.half_edge, t.shift, omega, eta)

    return replace(t, x=x1, y=y1, step_x=x2, step_y=y2,
                   restart_iter=t.restart_iter + 1, iteration=t.iteration + 1)


def _step(t: StepArguments) -> StepArguments:
    return _reflect_step(t, False)


def _h_step(t: StepArguments) -> StepArguments:
    return _reflect_step(t, True)


# returns the state after t.parameters.op_steps steps
def _op_steps(t: StepArguments) -> StepArguments:
    op_steps = t.parameters.op_steps

    if t.parameters.use_halpern:
        step_f = _h_step
    else:
        step_f = _step

    def do_op_step(_: int, ts: StepArguments) -> StepArguments:
        return step_f(ts)

    return lax.fori_loop(0, op_steps, do_op_step, t)


# L_P(x): the best dual response to the current primal, an UPPER bound on the
# relaxation value. This is F(x) of eq:qubo-F, evaluated on half-edges.
def _upper_bound_raw(half_edge: HalfEdge, x: Array) -> Array:
    ends = half_edge.gather(x)
    pair_sum, pair_min = ends[:, 0] + ends[:, 1], jnp.minimum(ends[:, 0], ends[:, 1])

    covering = half_edge.weights * jnp.maximum(pair_sum - 1.0, 0.0)
    packing = -half_edge.weights * pair_min
    edge_terms = jnp.where(half_edge.positive, covering, packing)

    return jnp.dot(half_edge.diagonal, x) + edge_terms.sum()


# L_D(y): the best primal response to the current dual, a LOWER bound. The
# primal set is a box, so the minimization is coordinatewise.
def _lower_bound_raw(half_edge: HalfEdge, y: Array) -> Array:
    reduced_cost = half_edge.diagonal - half_edge.scatter(y)
    shifted = jnp.sum(jnp.where(half_edge.positive, y[:, 0], 0.0))

    return jnp.minimum(reduced_cost, 0.0).sum() + shifted


def _upper_bound(t: StepArguments) -> Array:
    return _upper_bound_raw(t.half_edge, t.x) * t.objective_scale


def _lower_bound(t: StepArguments) -> Array:
    return _lower_bound_raw(t.half_edge, t.y) * t.objective_scale


def relaxation_bounds(instance: QuboProblem, primal: Array, dual: Array
                      ) -> tuple[Array, Array]:
    """
    Bracket the relaxation value from a primal-dual pair, in the instance's own
    units. Both iterates are always feasible, so this is a true certificate at
    every point of the run and not only at convergence.

    :param instance: the QUBO instance
    :param primal: a point of [0, 1]^n
    :param dual: a point of the dual set V
    :return: (lower bound L_D, upper bound L_P)
    """
    half_edge = HalfEdge.build(instance)

    return _lower_bound_raw(half_edge, dual), _upper_bound_raw(half_edge, primal)


def _obj_gap(t: StepArguments) -> Array:
    upper = _upper_bound(t)
    lower = _lower_bound(t)

    return (upper - lower) / (1.0 + jnp.abs(upper) + jnp.abs(lower))


def _check_termination_with_gap(t: StepArguments, relative_gap: Array) -> Array:
    iteration, params = t.iteration, t.parameters

    within_max_iter = (iteration < params.max_iter) | (params.max_iter < 0)
    # negated form, so that a NaN gap counts as NOT converged: `nan > tol` is False
    # and would report success, whereas `~(nan <= tol)` is True and keeps iterating
    # until max_iter, which is the honest outcome.
    outside_tolerance = ~(relative_gap <= params.tolerance)

    return within_max_iter & outside_tolerance


def _check_termination(t: StepArguments) -> Array:
    return _check_termination_with_gap(t, _obj_gap(t))


def _it_err(x, step_x, y, step_y, half_edge, eta, omega):
    return canonical_norm(x - step_x, y - step_y, half_edge, eta, omega)


def _it_err_t(t: StepArguments) -> Array:
    return _it_err(t.x, t.step_x, t.y, t.step_y, t.half_edge,
                   t.parameters.eta, t.parameters.omega)


def _halpern_step(t: StepArguments) -> StepArguments:
    t2 = _op_steps(t)

    new_err = _it_err_t(t2)
    err = t.fp_error

    # vanilla step
    t3 = replace(t2, x=t2.step_x, y=t2.step_y)

    relative_gap = _obj_gap(t3)
    terminated = ~_check_termination_with_gap(t3, relative_gap)

    p_x, p_y, params = t2.x, t2.y, t2.parameters
    restart_iter, iteration = t2.restart_iter, t2.iteration

    # check restart condition; the necessary condition compares against the
    # previous check rather than the previous iterate, matching cuPDLPx
    sufficient = (new_err <= params.beta_sufficient * err)
    necessary = (new_err <= params.beta_necessary * err) & (new_err > t.check_error)
    artificial = (restart_iter >= params.beta_artificial * iteration)
    should_restart = sufficient | necessary | artificial

    def _omega_true():
        d_x = p_x - t2.x0
        d_y = p_y - t2.y0

        # the primal norm is degree-weighted, matching the canonical metric
        primal_norm = jnp.sqrt(jnp.sum(t2.half_edge.degree * d_x * d_x))
        dual_norm = jnp.linalg.norm(d_y.ravel())

        change_omega = (primal_norm > 1e-16) & (dual_norm > 1e-16)

        # exponential smoothing in log space
        theta = t.parameters.theta
        omega_2 = jnp.exp(theta * (jnp.log(dual_norm) - jnp.log(primal_norm))
                          + (1.0 - theta) * jnp.log(t.parameters.omega))
        theta_2 = params.theta_target + params.theta_decay * (theta - params.theta_target)

        return (lax.select(change_omega, omega_2, t.parameters.omega),
                lax.select(change_omega, theta_2, theta))

    new_omega, new_theta = lax.cond(should_restart, _omega_true,
                                    lambda: (t.parameters.omega, t.parameters.theta))

    def _restart_true():
        t2_new = replace(t2, parameters=replace(t2.parameters, omega=new_omega))
        t4 = _step(t2_new)
        err_new = _it_err(t4.x, t4.step_x, t4.y, t4.step_y, t4.half_edge,
                          t4.parameters.eta, new_omega)

        restart_iter_new = 0
        check_err_new = jnp.full_like(err_new, jnp.inf)
        return t4.x, t4.y, t4.step_x, t4.step_y, err_new, check_err_new, restart_iter_new

    def _restart_false():
        return t2.x0, t2.y0, t2.step_x, t2.step_y, err, new_err, restart_iter

    x0, y0, step_x, step_y, err, check_err, restart_iter = lax.cond(
        should_restart, _restart_true, _restart_false)

    def _active_true():
        return (lax.select(should_restart, x0, p_x),
                lax.select(should_restart, y0, p_y))

    def _active_false():
        return t3.x, t3.y

    x2, y2 = lax.cond(~terminated, _active_true, _active_false)

    return replace(t2, x=x2, y=y2, step_x=step_x, step_y=step_y,
                   x0=x0, y0=y0, fp_error=err, check_error=check_err,
                   restart_iter=restart_iter, terminate=terminated,
                   parameters=replace(t.parameters, omega=new_omega, theta=new_theta))


def _should_step(t: StepArguments) -> Array:
    return ~t.terminate


@jit
def _multi_step(t: StepArguments) -> StepArguments:
    if t.parameters.use_halpern:
        return lax.while_loop(_should_step, _halpern_step, t)
    return lax.while_loop(_check_termination, _op_steps, t)


@jit
def solve_qubo_problem_warm_start(instance: QuboProblem,
                                  params: Parameters,
                                  warm_start: tuple[Array, Array]
                                  ) -> tuple[Array, Array, int]:
    """
    Solve the LP relaxation of `instance` with PRAQ.

    The relaxation is the roof dual, so the result is a bound and a
    variable-fixing certificate, not in general a QUBO optimum: by persistency,
    every coordinate that comes out at 0 or 1 takes that value in some optimal
    binary solution.

    Half-integrality is a property of the relaxation's VERTICES. A first-order
    method converges to a point of the optimal face, which when that face is not
    a single point may be fractional beyond {0, 1/2, 1}. This is safe but
    conservative: a coordinate integral at a face-interior point is integral at
    every point of the face, so persistency still applies -- it simply fixes no
    more variables than a vertex would, and possibly fewer.

    :param instance: the QUBO instance
    :param params: the solver parameters
    :param warm_start: (x, y) in the instance's own units; both are projected
                       before use, so any point may be passed
    :return: (primal, dual, iterations), the dual in the instance's own units
    """
    x0, y0 = warm_start
    dtype = instance.matrix.dtype
    x0, y0 = x0.astype(dtype), y0.astype(dtype)
    params = params.as_type(dtype)
    if params.op_steps <= 0:
        params = replace(params, op_steps=check_interval(
            instance.n(), params.interval_offset, params.interval_scale, params.interval_power))

    half_edge = HalfEdge.build(instance)

    # bound objective scaling
    objective_scale = instance.objective_scale()
    half_edge = half_edge.scaled(objective_scale)
    shift = half_edge.shift()

    # the warm start is given in the instance's own units, matching what this
    # function returns; bring the dual into the normalized ones
    x0 = project_primal(x0)
    y0 = project_dual(y0 / objective_scale, half_edge.weights, half_edge.positive)

    x1, y1 = _raw_step(x0, y0, half_edge, shift, params.omega, params.eta)

    starting_params = params
    if params.use_halpern:
        starting_params = replace(params, use_halpern=False,
                                  max_iter=params.op_steps * params.warm_steps)

    t = StepArguments(x0, y0, x1, y1, x0, y0,
                      jnp.array(0.0, dtype=dtype), jnp.array(jnp.inf, dtype=dtype),
                      jnp.array(0), half_edge, shift,
                      0, starting_params, jnp.array(False), objective_scale)

    t = _multi_step(t)

    primal_norm = jnp.sqrt(jnp.sum(half_edge.degree * (t.x - x0) ** 2))
    dual_norm = jnp.linalg.norm((t.y - y0).ravel())
    omega = dual_norm / primal_norm
    omega = jnp.where(jnp.isnan(omega) | (omega <= 0.0),
                      t.parameters.omega * jnp.ones_like(omega), omega)

    t = replace(t, parameters=replace(params, omega=omega))

    err = _it_err_t(t)
    terminated = ~_check_termination(t)
    t = replace(t, fp_error=err, check_error=jnp.full_like(err, jnp.inf),
                x0=t.x, y0=t.y, restart_iter=jnp.array(0), terminate=terminated)

    output = _multi_step(t)

    # un-normalize the dual: V scales linearly with the edge weights, so
    # multiplying by varsigma lands the iterate in the dual set of the original
    # instance. The primal lives in [0, 1]^n and is already scale-free.
    return output.x, output.y * objective_scale, output.iteration


@jit
def solve_qubo_problem(instance: QuboProblem, params: Parameters
                       ) -> tuple[Array, Array, int]:
    """
    Solve the LP relaxation of `instance` from the cold start.

    The start is the centroid of the feasible region: x = 1/2 and, on every
    edge, the centre of its dual set (HalfEdge.centroid). It is the anchor of
    the first Halpern phase, so it matters more than a start does for plain
    PDHG; measured against x = 0 with the projection of 0 as the dual, it takes
    fewer iterations on most instances and never more than about one check
    interval more. On an instance whose diagonal is minus its row sums -- max
    cut form -- the pair is already an optimal certificate, every reduced cost
    being exactly 0, and the solve ends at iteration 0.

    :param instance: the QUBO instance
    :param params: the solver parameters
    :return: (primal, dual, iterations), as solve_qubo_problem_warm_start
    """
    n = instance.n()
    dtype = instance.matrix.dtype

    x0 = jnp.full(n, 0.5, dtype=dtype)
    y0 = HalfEdge.build(instance).centroid()

    return solve_qubo_problem_warm_start(instance, params, (x0, y0))


def persistent_variables(primal: Array, tolerance: float = 1e-6
                         ) -> tuple[Array, Array]:
    """
    Split a relaxation optimum into the coordinates persistency fixes and the
    fractional core that remains.

    By the persistency property of the roof dual, every coordinate at 0 or 1 in
    an optimal relaxation solution takes that value in some optimal binary
    solution, so those variables may be fixed and removed. The rest sit at 1/2
    when the solve lands on a vertex; see solve_qubo_problem_warm_start for why
    a first-order solve need not.

    :param primal: an optimal (or near-optimal) relaxation solution
    :param tolerance: how far from integral still counts as fixed
    :return: (mask of fixed coordinates, their rounded values)
    """
    rounded = jnp.round(primal)
    fixed = jnp.abs(primal - rounded) <= tolerance

    return fixed, rounded
