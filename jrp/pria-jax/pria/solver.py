from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, config, jit, lax, vmap
from jax.tree_util import register_dataclass

# XLA tuning knobs.  jax_exec_time_optimization_effort and memory_fitting_effort
# were removed in jax 0.11.1 in favour of the CompilerEffortLevel enum, which is
# what jax_optimization_level below now takes -- so the effort request is still
# made, by its current name, and nothing was lost in the migration.
#
# These are set unguarded on purpose.  Swallowing an unrecognised option would
# silently drop the optimisation on a jax that renamed it; the checks run
# against an unpinned jax so the next rename fails loudly and gets migrated
# instead.
config.update('jax_optimization_level', 'O3')
config.update('eager_constant_folding', True)
config.update('jax_use_simplified_jaxpr_constants', True)

from .convergence_log import RestartType, log_convergence_iteration, log_restart_iteration
from .halpern_operator import canonical_norm
from .jrp_problem import JrpProblem, primal_objective
from .projection import (
    double_projection_3d_dim2,
    double_projection_upper_triangular,
    double_projection_upper_triangular_row,
)
from .upper_triangular import (
    col_to_dim2,
    col_to_dim2_permutation,
    col_to_row,
    col_to_row_permutation,
    densify_columnwise,
    densify_rowwise,
    densify_tensor_dim2,
    dim2_to_col,
    dim2_to_col_permutation,
    row_to_col,
    row_to_col_permutation,
    sparsify_columnwise,
    sparsify_rowwise,
    sparsify_tensor_dim2,
)

densify_column_vmap = vmap(densify_columnwise)
densify_row_vmap = vmap(densify_rowwise)

sparsify_column_vmap = vmap(sparsify_columnwise)
sparsify_row_vmap = vmap(sparsify_rowwise)


data_fields = ["omega",
               "theta",
               "iid"]
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
               "log"]
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
    iid: int
    log: bool


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
            iid: int = 0,
            log: bool = False):
        """
        :param omega:           The primal (X) weight
        :param eta:             The step size
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
        :param op_steps:        How many steps between each convergence and restart check
        :param tolerance:       The relative optimality tolerance
        :param iid:             A unique identifying integer for the instance, necessary when logging
        :param log:             Whether to log, logging is slow
        :return:                The solver parameters
        """
        if max_iter is None:
            max_iter = -1

        op_steps = max(op_steps, 2)
        warm_steps = max(warm_steps, 0)

        return cls(
            omega,
            eta,
            beta_sufficient,
            beta_artificial,
            beta_necessary,
            theta_initial,
            theta_target,
            theta_decay,
            max_iter,
            use_halpern,
            warm_steps,
            op_steps,
            tolerance,
            iid,
            log)


    @classmethod
    def with_auto_step(cls,
                       max_iter: int | None,
                       tolerance: float,
                       use_halpern: bool = True,
                       op_steps: int = 16,
                       iid: int = 0,
                       log: bool = False,
                       omega: float = 0.10835,
                       # 1/e minimizes the worst-case complexity of
                       # sufficient-decay restarts in the fixed-omega analysis,
                       # the canonical norm being the monotone PPA metric; see
                       # convergence-notes/findings.md.  beta_necessary and
                       # beta_artificial are heuristic.
                       beta_sufficient: float = float(np.exp(-1.0)),
                       beta_necessary: float = 0.5706,
                       beta_artificial: float = 0.2468,
                       theta_initial: float = 0.81930,
                       theta_target: float = 0.42818,
                       theta_decay: float = 0.90474,
                       warm_steps: int = 0):
        eta = 0.999 / np.sqrt(2.0)

        return cls.new(
            omega,
            eta,
            beta_sufficient,
            beta_artificial,
            beta_necessary,
            theta_initial,
            theta_target,
            theta_decay,
            max_iter,
            use_halpern,
            warm_steps,
            op_steps,
            tolerance,
            iid,
            log)


    @classmethod
    def stack(cls, parameters: list["Parameters"]) -> "Parameters":
        """
        Converts an array-of-structs `parameters` to a struct-of-arrays. The
        max_iter and tolerance fields will be inherited from the first
        Parameters object in the list.

        :param parameters: a list of Parameters singletons to stack
        :return: a Parameters instance with 1D arrays for primal_step and dual_step
        """
        omegas       = jnp.array([p.omega for p in parameters])
        etas         = parameters[0].eta
        beta_suf     = parameters[0].beta_sufficient
        beta_art     = parameters[0].beta_artificial
        beta_nec     = parameters[0].beta_necessary
        thetas       = jnp.array([p.theta for p in parameters])
        theta_target = parameters[0].theta_target
        theta_decay  = parameters[0].theta_decay
        iid          = jnp.array([p.iid for p in parameters])
        max_iters    = parameters[0].max_iter
        use_halpern  = parameters[0].use_halpern
        warm_steps   = parameters[0].warm_steps
        op_steps     = parameters[0].op_steps
        tolerances   = parameters[0].tolerance
        log          = parameters[0].log

        return cls(
            omegas,
            etas,
            beta_suf,
            beta_art,
            beta_nec,
            thetas,
            theta_target,
            theta_decay,
            max_iters,
            use_halpern,
            warm_steps,
            op_steps,
            tolerances,
            iid,
            log)


    def as_type(self, dtype) -> "Parameters":
        return Parameters.new(
            jnp.array(self.omega, dtype=dtype),
            self.eta,
            self.beta_sufficient,
            self.beta_artificial,
            self.beta_necessary,
            jnp.array(self.theta, dtype=dtype),
            self.theta_target,
            self.theta_decay,
            self.max_iter,
            self.use_halpern,
            self.warm_steps,
            self.op_steps,
            self.tolerance,
            jnp.array(self.iid, dtype=dtype),
            self.log
        )

@partial(register_dataclass, data_fields=[],
         meta_fields=["row_col_vmap", "col_row_vmap", "dim2_col", "col_dim2"])
@dataclass(frozen=True)
class Permutations:
    row_col_vmap: Callable[[Array], Array]
    col_row_vmap: Callable[[Array], Array]

    dim2_col: Callable[[Array], Array]
    col_dim2: Callable[[Array], Array]

    @classmethod
    def new(cls, n: int, t: int) -> "Permutations":
        col_to_row_p = col_to_row_permutation(t)
        row_to_col_p = row_to_col_permutation(t)

        row_col_vmap_f = vmap(partial(row_to_col, perm=row_to_col_p))
        col_row_vmap_f = vmap(partial(col_to_row, perm=col_to_row_p))

        dim2_to_col_p = dim2_to_col_permutation(n, t)
        col_to_dim2_p = col_to_dim2_permutation(n, t)

        dim2_col_f = partial(dim2_to_col, perm=dim2_to_col_p)
        col_dim2_f = partial(col_to_dim2, perm=col_to_dim2_p)

        return cls(row_col_vmap_f, col_row_vmap_f, dim2_col_f, col_dim2_f)


@register_dataclass
@dataclass(frozen=True)
class StepArguments:
    x: Array
    y: Array
    z: Array

    step_x: Array
    step_y: Array
    step_z: Array

    x0: Array
    y0: Array
    z0: Array

    # error of current x0, y0, z0
    fp_error: Array
    # fixed-point error at the previous restart/convergence check
    check_error: Array
    # number of iterations since last restart
    restart_iter: Array

    h: Array
    k: Array
    k0: float

    iteration: int
    parameters: Parameters

    terminate: Array
    permute: Permutations

    # scalar for bound objective scaling
    objective_scale: Array


# return v[:l].max() + v[l:].max()
def sum_max_partition(v: Array, l: Array) -> Array:
    a_max = v.argmax()
    max_1 = v[a_max]
    max_is_left = (a_max < l)

    less_l = (jnp.arange(v.shape[0]) < l)
    other_side = max_is_left ^ less_l

    max_array = jnp.where(other_side, v, -jnp.inf)
    max_2 = max_array.max()

    return max_1 + max_2


def sum_max_partition_upper_triangular_row(m: Array) -> Array:
    l = jnp.arange(m.shape[1]-1, m.shape[1]-1-m.shape[0], -1)
    return vmap(sum_max_partition, in_axes=(0, 0), out_axes=0)(m, l)


def double_max_2d(m: Array, l: int) -> Array:
    m_vector = m.ravel()
    return sum_max_partition(m_vector, l * m.shape[1])


def double_max_3d_dim2(m: Array) -> Array:
    l = jnp.arange(m.shape[1] - 1, m.shape[1] - 1 - m.shape[0], -1)
    return vmap(double_max_2d, in_axes=(0, 0), out_axes=0)(m, l)


def _upper_bound(t: StepArguments) -> Array:
    """The LP objective at the current primal: the best duals for a fixed x.

    Named for which side of the optimum it sits on, not for which variables
    the maximisation runs over -- an upper bound on a minimisation, which is
    also what pria-cuda's `primal_objective` reports.
    """
    x, h, k, k0 = t.x, t.h, t.k, t.k0

    term3 = jnp.sum(x * h)

    kx = k[:, None, None] * x
    kx = t.permute.col_row_vmap(kx)
    term1 = vmap(sum_max_partition_upper_triangular_row)(kx).sum()

    dx = k0 * t.permute.col_dim2(x)
    term2 = double_max_3d_dim2(dx).sum()

    return (term1 + term2 + term3) * t.objective_scale


# return v[:l].min() + v[l:].min()
def sum_min_partition(v: Array, l: Array) -> Array:
    a_min = v.argmin()
    min_1 = v[a_min]
    min_is_left = (a_min < l)

    less_l = (jnp.arange(v.shape[0]) < l)
    other_side = min_is_left ^ less_l

    min_array = jnp.where(other_side, v, jnp.inf)
    min_2 = min_array.min()

    return min_1 + min_2


def sum_min_partition_upper_triangular(m: Array) -> Array:
    l = jnp.arange(1, m.shape[0]+1)
    return vmap(sum_min_partition, in_axes=(0, 0), out_axes=0)(m, l)


def _lower_bound(t: StepArguments) -> Array:
    """The LP dual's objective at the current duals: the best x for fixed y, z.

    A lower bound on the minimisation, and what pria-cuda's `dual_objective`
    reports.  On a padded instance this covers the pad period too; see
    `solve_jrp_problem_warm_start`.
    """
    y, z, h = t.y, t.z, t.h

    y = t.permute.row_col_vmap(y)
    z = t.permute.dim2_col(z)

    lhs = (z + y + h)

    return vmap(sum_min_partition_upper_triangular)(lhs).sum() * t.objective_scale


x_projection = vmap(double_projection_upper_triangular)
y_projection = vmap(double_projection_upper_triangular_row, in_axes=(0, 0), out_axes=0)


def _raw_step(x1: Array, y1: Array, z1: Array,
              h: Array, k: Array, k0: float,
              omega: float, eta: float, permutations: Permutations) -> tuple[Array, Array, Array]:
    y1_col = permutations.row_col_vmap(y1)
    z1_col = permutations.dim2_col(z1)

    omega = jnp.array(omega).astype(h.dtype)
    eta = jnp.array(eta).astype(h.dtype)

    x2_inner = x1 - (eta / omega) * (h + z1_col + y1_col)
    x2 = x_projection(x2_inner)

    reflected_x = eta * omega * (2 * x2 - x1)
    y1 = y_projection(permutations.col_row_vmap(y1_col + reflected_x), k)
    z1 = double_projection_3d_dim2(permutations.col_dim2(z1_col + reflected_x), k0)

    return x2, y1, z1


def _reflect_step(t: StepArguments, reflect: bool) -> StepArguments:
    x1, y1, z1 = t.step_x, t.step_y, t.step_z
    h, k, k0 = t.h, t.k, t.k0
    omega, eta = t.parameters.omega, t.parameters.eta

    if reflect:
        restart_iter = t.restart_iter + 1

        left_weight = restart_iter / (restart_iter + 1.0)
        right_weight = 1.0 / (restart_iter + 1.0)

        x1 = left_weight * (2 * x1 - t.x) + right_weight * t.x0
        y1 = left_weight * (2 * y1 - t.y) + right_weight * t.y0
        z1 = left_weight * (2 * z1 - t.z) + right_weight * t.z0

    x2, y2, z2 = _raw_step(x1, y1, z1, h, k, k0, omega, eta, t.permute)

    return replace(t, x=x1, y=y1, z=z1,
                   step_x=x2, step_y=y2, step_z=z2,
                   restart_iter=t.restart_iter+1, iteration=t.iteration+1)


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


def _obj_gap(t: StepArguments) -> Array:
    lower = _lower_bound(t)
    upper = _upper_bound(t)

    return (upper - lower) / (1.0 + jnp.abs(lower) + jnp.abs(upper))


def _check_termination_with_gap(t: StepArguments, relative_gap: Array) -> Array:
    iteration, params = t.iteration, t.parameters

    within_max_iter = (iteration < params.max_iter) | (params.max_iter < 0)
    # Negated form, so that a NaN gap counts as NOT converged: `nan > tol` is
    # False and would report success on the spot, whereas `~(nan <= tol)` is
    # True and keeps iterating until max_iter, which is the honest outcome.
    outside_tolerance = ~(relative_gap <= params.tolerance)

    if params.log:
        log_convergence_iteration(params.iid, iteration,
                                  jnp.array(_upper_bound(t)), _lower_bound(t))

    return within_max_iter & outside_tolerance


def _check_termination(t: StepArguments) -> Array:
    relative_gap = _obj_gap(t)
    return _check_termination_with_gap(t, relative_gap)


def _it_err(x, step_x, y, step_y, z, step_z, permute, eta, omega):
    dx = x - step_x
    dy = permute.row_col_vmap(y - step_y)
    dz = permute.dim2_col(z - step_z)

    return canonical_norm(dx, dy, dz, eta, omega)


def _it_err_t(t):
    return _it_err(t.x, t.step_x, t.y, t.step_y, t.z, t.step_z, t.permute,
                   t.parameters.eta, t.parameters.omega)


def _halpern_step(t: StepArguments) -> StepArguments:
    t2 = _op_steps(t)

    new_err = _it_err_t(t2)
    err = t.fp_error

    # vanilla step
    t3 = replace(t2, x=t2.step_x, y=t2.step_y, z=t2.step_z)

    relative_gap = _obj_gap(t3)
    terminated = ~_check_termination_with_gap(t3, relative_gap)

    p_x, p_y, p_z, params = t2.x, t2.y, t2.z, t2.parameters
    restart_iter, iteration = t2.restart_iter, t2.iteration

    # check restart condition; the necessary condition compares against the
    # previous check rather than the previous iterate, matching cuPDLPx
    sufficient = (new_err <= params.beta_sufficient * err)
    necessary = (new_err <= params.beta_necessary * err) & (new_err > t.check_error)
    artificial = (restart_iter >= params.beta_artificial * iteration)
    should_restart = sufficient | necessary | artificial

    def _omega_true():
        d_x = (p_x - t2.x0).ravel()
        d_y = (p_y - t2.y0).ravel()
        d_z = (p_z - t2.z0).ravel()

        primal_norm = jnp.linalg.norm(d_x)
        dual_norm = jnp.sqrt(jnp.dot(d_y, d_y) + jnp.dot(d_z, d_z))

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

    if params.log:
        sufficient_int = RestartType.Sufficient * sufficient.astype(int)
        necessary_int = RestartType.Necessary * necessary.astype(int)
        artificial_int = RestartType.Artificial * artificial.astype(int)
        restart_type = jnp.maximum(jnp.maximum(sufficient_int, necessary_int), artificial_int)

        log_restart_iteration(params.iid, iteration, new_err, restart_type, new_omega)

    def _restart_true():
        t2_new = replace(t2, parameters=replace(t2.parameters, omega=new_omega))
        t4 = _step(t2_new)
        err_new = _it_err(t4.x, t4.step_x, t4.y, t4.step_y, t4.z, t4.step_z,
                          t4.permute, t4.parameters.eta, new_omega)

        restart_iter_new = 0
        check_err_new = jnp.full_like(err_new, jnp.inf)
        return t4.x, t4.y, t4.z, t4.step_x, t4.step_y, t4.step_z, err_new, check_err_new, restart_iter_new

    def _restart_false():
        return t2.x0, t2.y0, t2.z0, t2.step_x, t2.step_y, t2.step_z, err, new_err, restart_iter

    (x0, y0, z0, step_x, step_y, step_z, err, check_err,
     restart_iter) = lax.cond(should_restart, _restart_true, _restart_false)

    def _active_true():
        return (lax.select(should_restart, x0, p_x),
                lax.select(should_restart, y0, p_y),
                lax.select(should_restart, z0, p_z))

    def _active_false():
        return t3.x, t3.y, t3.z

    x2, y2, z2 = lax.cond(~terminated, _active_true, _active_false)

    return replace(t2, x=x2, y=y2, z=z2, step_x=step_x, step_y=step_y, step_z=step_z,
                   x0=x0, y0=y0, z0=z0, fp_error=err, check_error=check_err,
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
def solve_jrp_problem_warm_start(instance: JrpProblem,
                                 params: Parameters,
                                 warm_start: tuple[Array, Array, Array]
                                 ) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Solve from a given starting point, returning the solution and its bracket.

    Returns ``(x, y, z, iterations, lower, upper)``.  `x` is the primal
    solution and `y`, `z` are the dual iterates normalised by K_i and K0, so a
    returned triple warm-starts as a fixed point.  `lower` and `upper` bracket
    the LP value, matching `pria_result_info` on the CUDA side.

    The two bounds are not computed in the same place, and the reason is the
    padding.  An odd number of periods is padded to even, which adds a period
    whose holding costs are zero -- free to serve from any order already
    placed, so the padded LP has the same optimal value.  `lower` is the
    solver's own dual bound *on the padded problem*, which is valid for the
    original one and is the bound it converged against; recomputing it
    outside, on the unpadded duals, throws away the pad row's contribution and
    can come out much weaker (`relaxation_bounds` does exactly that, and says
    so).  `upper` is the objective at the unpadded primal, which is a feasible
    point of the original LP and therefore at least as tight as the padded
    objective would be.
    """
    original = instance
    x0, y0, z0 = warm_start
    dtype = instance.holding_costs.dtype
    x0, y0, z0 = x0.astype(dtype), y0.astype(dtype), z0.astype(dtype)
    params = params.as_type(dtype)

    # pad number of time periods to be even
    _, m, _ = instance.holding_costs.shape
    if m % 2 == 1:
        pad_width = ((0, 0), (0, 1), (0, 1))
        h_pad = jnp.pad(instance.holding_costs, pad_width, mode='constant')
        x0 = jnp.pad(x0, pad_width, mode='constant')
        y0 = jnp.pad(y0, pad_width, mode='constant')
        z0 = jnp.pad(z0, pad_width, mode='constant')

        instance = replace(instance, holding_costs=h_pad)

    k = instance.order_costs
    k0 = jnp.array(instance.joint_order_cost, dtype=dtype)

    # bound objective scaling
    objective_vector = jnp.append(jnp.append(instance.holding_costs.ravel(), k), k0)
    objective_scale = jnp.linalg.norm(objective_vector)

    k = k / objective_scale
    k0 = k0 / objective_scale
    holding = instance.holding_costs / objective_scale

    instance = replace(instance, holding_costs=holding, order_costs=k, joint_order_cost=k0)

    # Set up the denser data structures.  The warm-start duals arrive in the
    # convention the solver returns them in -- normalised, each group summing
    # to one -- so they are scaled back onto the K_i- and K0-simplices before
    # projecting.  Without that the projection of a point summing to one onto
    # a simplex summing to K_i is not that point, and a returned solution
    # would not warm-start as the fixed point it is.  The primal needs no such
    # scaling: its simplices sum to one in both conventions.
    x0 = x_projection(densify_column_vmap(x0))
    y0 = y_projection(densify_row_vmap(y0) * k[:, None, None], k)
    z0 = double_projection_3d_dim2(densify_tensor_dim2(z0) * k0, k0)

    holding = instance.holding_costs
    holding = densify_column_vmap(holding)

    permutations = Permutations.new(instance.n(), instance.t())
    x1, y1, z1 = _raw_step(x0, y0, z0, holding, k, k0, params.omega, params.eta, permutations)

    starting_params = params
    if params.use_halpern:
        starting_params = replace(params, use_halpern=False, max_iter=params.op_steps * params.warm_steps)

    t = StepArguments(x0, y0, z0, x1, y1, z1, x0, y0, z0,
                      jnp.array(0.0), jnp.array(jnp.inf), jnp.array(0),
                      holding, k, k0,
                      0, starting_params, jnp.array(False),
                      permutations, objective_scale)

    t = _multi_step(t)

    primal_norm = jnp.linalg.norm((t.x - x0).ravel())
    dual_norm = jnp.linalg.norm(jnp.append((t.y - y0).ravel(), (t.z - z0).ravel()))
    omega = dual_norm / primal_norm
    omega = jnp.where(jnp.isnan(omega), t.parameters.omega * jnp.ones_like(omega), omega)

    t = replace(t, parameters=replace(params, omega=omega))

    err = _it_err_t(t)
    terminated = ~_check_termination(t)
    t = replace(t, fp_error=err, check_error=jnp.full_like(err, jnp.inf),
                x0=t.x, y0=t.y, z0=t.z, restart_iter=jnp.array(0), terminate=terminated)

    output = _multi_step(t)
    x, y, z = output.x, output.y, output.z
    iteration = output.iteration
    lower = _lower_bound(output)

    # A zero budget projects onto {0}, and normalising by it is 0/0.  The
    # normalisation exists so that a returned dual round trips as a warm
    # start; for a budget of zero the point is zero either way.
    y = jnp.where(k[:, None, None] > 0, y / jnp.where(k[:, None, None] > 0, k[:, None, None], 1.0), 0.0)
    z = jnp.where(k0 > 0, z / jnp.where(k0 > 0, k0, 1.0), 0.0)

    x = sparsify_column_vmap(x)
    y = sparsify_row_vmap(y)
    z = sparsify_tensor_dim2(z)

    # remove padding if the original instance was odd
    if m % 2 == 1:
        x = x[:, :m, :m]
        y = y[:, :m, :m]
        z = z[:, :m, :m]

    return x, y, z, iteration, lower, primal_objective(original, x)


@jit
def solve_jrp_problem(instance: JrpProblem, params: Parameters
                      ) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Solve from zero.  Returns ``(x, y, z, iterations, lower, upper)``;
    see `solve_jrp_problem_warm_start` for what each one is."""
    var_shape = (instance.n(), instance.t(), instance.t())

    starts = jnp.zeros(var_shape, dtype=instance.holding_costs.dtype)
    warm_start = (starts, starts, starts)

    return solve_jrp_problem_warm_start(instance, params, warm_start)


def relaxation_bounds(instance: JrpProblem, primal: Array, y: Array, z: Array
                      ) -> tuple[Array, Array]:
    """
    Bracket the LP value from a primal-dual triple, in the instance's own units.

    An independent recomputation, for a triple from anywhere -- a warm start, a
    saved solution, another implementation's output.  The solver reports its
    own bracket alongside its answer (`solve_jrp_problem`), and that is the one
    to read after a solve; this is the one to check it with.

    Both iterates are feasible at every step -- the primal projections land on
    the per-(i, t) unit simplices and the dual ones on the K_i- and
    K0-simplices -- so both numbers are valid bounds at any point of a run and
    not only at convergence.

    The upper bound is `jrp_problem.primal_objective` at `primal`.  The lower
    bound is the LP dual's objective at ``alpha = K_i y``, ``beta = K0 z``,

        sum_{i,t} min_{s <= t} ( H_ist + alpha_ist + beta_ist ),

    which is the best the multiplier of ``sum_{s <= t} x_ist = 1`` can be once
    the reduced-cost constraints are imposed.  The solver's normalisation is
    what makes that substitution legal: the returned `y` has rows summing to
    one over ``t >= s`` and `z` has each group ``{(i, t >= s)}`` summing to
    one, so scaling by K_i and K0 lands exactly on the dual's two budget
    constraints.  Scaling by less would too -- the budgets are inequalities --
    which is why this stays a valid bound for a triple that has been truncated.

    **On an odd number of periods this lower bound is weaker than the
    solver's.**  The solver pads to an even count and its dual assigns the pad
    period a multiplier of its own; the returned duals have the pad sliced off,
    so that contribution is gone from the sum here.  What is left is still
    dual-feasible and still a bound -- dropping nonnegative mass only lowers it
    -- but it can sit well below the value the solver converged against.  Read
    the solver's own bracket for how well it converged, and this one for
    whether a point is what it claims to be.

    :param instance: the JRP instance
    :param primal: a primal point, dense [N][T][T] and upper triangular
    :param y: the per-item order duals, normalised as the solver returns them
    :param z: the joint-order duals, normalised as the solver returns them
    :return: (lower bound, upper bound)
    """
    holding = jnp.triu(instance.holding_costs)
    reduced = (holding + instance.order_costs[:, None, None] * y
               + instance.joint_order_cost * z)
    periods = jnp.arange(holding.shape[-1])
    # s > t is not a variable of the LP at all, so it must not win the min.
    reduced = jnp.where(periods[:, None] <= periods[None, :], reduced, jnp.inf)

    return reduced.min(axis=1).sum(), primal_objective(instance, primal)
