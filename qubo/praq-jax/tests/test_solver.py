import numpy as np
import jax.numpy as jnp
from jax import config

config.update('jax_enable_x64', True)

from praq.half_edge import HalfEdge
from praq.prelude import *
from praq.brute_force import solve_qubo_brute_force
from reference import reference_relaxation_value
from praq.solver import _lower_bound_raw, _upper_bound_raw

TOLERANCE = 1e-9


def _instances(sizes, seed=0):
    return [QuboProblem.random_instance(seed + k, n, dtype=jnp.float64)
            for k, n in enumerate(sizes)]


def _solve(instance, tolerance=TOLERANCE, max_iter=200_000):
    params = Parameters.with_auto_step(max_iter, tolerance)
    primal, dual, iterations = solve_qubo_problem(instance, params)
    return np.asarray(primal), np.asarray(dual), int(iterations)


def test_relaxation_value_agrees_with_the_half_edge_form():
    # QuboProblem.relaxation_value works from the dense matrix; the solver works
    # from half-edges. They must agree.
    rng = np.random.default_rng(0)
    for instance in _instances([2, 3, 5, 8]):
        half_edge = HalfEdge.build(instance)
        for _ in range(20):
            x = jnp.array(rng.random(instance.n()))
            dense = float(instance.relaxation_value(x))
            lifted = float(_upper_bound_raw(half_edge, x))
            assert abs(dense - lifted) < 1e-10 * (1.0 + abs(dense)), (dense, lifted)


def test_matches_reference_lp():
    print(f"\n{'n':>3} {'HiGHS':>12} {'PRAQ L_P':>13} {'PRAQ L_D':>13} {'rel err':>10} {'iters':>8}")
    for instance in _instances([4, 6, 8, 12, 20]):
        primal, dual, iterations = _solve(instance)
        reference = reference_relaxation_value(instance)

        lower, upper = relaxation_bounds(instance, jnp.array(primal), jnp.array(dual))
        lower, upper = float(lower), float(upper)

        scale = 1.0 + abs(reference)
        error = max(abs(upper - reference), abs(lower - reference)) / scale
        print(f"{instance.n():3d} {reference:12.6f} {upper:13.6f} {lower:13.6f} "
              f"{error:10.2e} {iterations:8d}")
        assert error < 1e-6, (instance.n(), reference, upper, lower)


def test_bounds_bracket_the_optimum_at_every_iterate():
    # Both iterates stay feasible, so L_D <= OPT <= L_P must hold throughout,
    # not only at convergence.
    instance = _instances([10])[0]
    reference = reference_relaxation_value(instance)

    for budget in (1, 4, 16, 64, 256):
        params = Parameters.with_auto_step(budget, TOLERANCE)
        primal, dual, _ = solve_qubo_problem(instance, params)
        lower, upper = relaxation_bounds(instance, primal, dual)

        assert float(lower) <= reference + 1e-9, (budget, float(lower), reference)
        assert float(upper) >= reference - 1e-9, (budget, float(upper), reference)


def test_solution_lands_in_the_optimal_face():
    # Half-integrality is a property of the relaxation's vertices; a first-order
    # method returns a point of the optimal face, which need not be one. What
    # must hold is that the point is optimal.
    for instance in _instances([4, 6, 9, 14], seed=7):
        primal, _, _ = _solve(instance)
        reference = reference_relaxation_value(instance)

        value = float(instance.relaxation_value(jnp.array(primal)))
        assert abs(value - reference) < 1e-6 * (1.0 + abs(reference)), (value, reference)


def test_persistency_fixes_correct_variables():
    for instance in _instances([4, 6, 8, 10], seed=11):
        primal, _, _ = _solve(instance)
        fixed, values = persistent_variables(jnp.array(primal), tolerance=1e-5)
        fixed, values = np.asarray(fixed), np.asarray(values)

        matrix = np.asarray(instance.matrix)
        n = instance.n()
        _, best = solve_qubo_brute_force(instance)

        assignments = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
        objectives = np.einsum("bi,ij,bj->b", assignments, matrix, assignments)
        optimal = assignments[np.abs(objectives - best) < 1e-9]

        agrees = (optimal[:, fixed] == values[fixed]).all(axis=1)
        assert agrees.any(), (instance.n(), primal, values[fixed])


def test_warm_start_from_the_solution_terminates_immediately():
    instance = _instances([10], seed=3)[0]
    primal, dual, _ = _solve(instance)

    params = Parameters.with_auto_step(None, TOLERANCE)
    _, _, iterations = solve_qubo_problem_warm_start(
        instance, params, (jnp.array(primal), jnp.array(dual)))
    assert int(iterations) <= params.op_steps, int(iterations)


def test_dual_stays_feasible():
    instance = _instances([12], seed=5)[0]
    _, dual, _ = _solve(instance)
    half_edge = HalfEdge.build(instance)

    weights = np.asarray(half_edge.weights)
    positive = np.asarray(half_edge.positive)
    dual = np.asarray(dual)

    covering, packing = dual[positive], dual[~positive]
    assert np.abs(covering[:, 0] - covering[:, 1]).max() < 1e-9
    assert (covering[:, 0] <= 1e-9).all() and (covering[:, 0] >= -weights[positive] - 1e-9).all()
    assert np.abs(packing.sum(axis=1) - weights[~positive]).max() < 1e-9
    assert (packing >= -1e-9).all()


if __name__ == "__main__":
    test_relaxation_value_agrees_with_the_half_edge_form()
    test_matches_reference_lp()
    test_bounds_bracket_the_optimum_at_every_iterate()
    test_solution_lands_in_the_optimal_face()
    test_persistency_fixes_correct_variables()
    test_warm_start_from_the_solution_terminates_immediately()
    test_dual_stays_feasible()
    print("test_solver: all tests passed")


def test_check_interval_rule():
    # The defaults are 3 log2(n / 50k): 50k -> 0, floored at 2; 400k -> 9; 2M -> 16.
    assert check_interval(50_000) == 2
    assert check_interval(400_000) == 9
    assert check_interval(2_000_000) == 16
    # power 1/2 with the same scale is a square root: 50k e^2 -> scale * 2 (e - 1) = 14.9 -> 15
    assert check_interval(int(50_000 * np.e ** 2), power=0.5) == 15
    # op_steps <= 0 solves with the derived interval; at n = 60 that is 2, so
    # the count is a positive multiple of 2, and it converges all the same
    instance = _instances([60], seed=3)[0]
    params = Parameters.with_auto_step(200_000, TOLERANCE, op_steps=0)
    primal, dual, iterations = solve_qubo_problem(instance, params)
    assert iterations > 0 and iterations % 2 == 0
    lower, upper = relaxation_bounds(instance, primal, dual)
    assert (upper - lower) / (1.0 + abs(upper) + abs(lower)) <= TOLERANCE
