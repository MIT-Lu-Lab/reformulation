"""The solver, end to end, against an independent LP reference.

The reference is HiGHS through `baselines/`, reached by `tests/reference.py` so
that the dependency runs tests -> baselines and never pria -> baselines.  What
is checked here is the pair of things a solve has to get right: the value it
converges to, and the bracket it reports around it.
"""

import time

from jax import block_until_ready, config, jit, vmap

config.update("jax_enable_x64", True)   # the tolerances below are tighter than float32

import jax.numpy as jnp
from reference import reference_relaxation_value

from pria.prelude import (
    JrpProblem,
    Parameters,
    relaxation_bounds,
    solve_jrp_problem,
    solve_jrp_problem_warm_start,
)

#: What every solve below is asked for, and what its bracket is then held to.
TOLERANCE = 1e-8

#: How far a solve may land from the reference and still be the same answer.
#: Two decades above the tolerance it stopped on, the same margin
#: `harness.config.AGREEMENT_TOLERANCE` keeps for the same reason.
CLOSE = 1e-6

#: (seed, items, periods) of the instances the reference is solved on.  Small,
#: because a disagreement shows up at three periods as plainly as at three
#: hundred and HiGHS builds the LP explicitly.  Both parities are here, since
#: an odd period count is padded and an even one is not, and so is the
#: single-item case, whose joint order cost is zero and whose joint dual set is
#: therefore the degenerate simplex {0}.
SHAPES = ((1, 2, 3), (2, 3, 4), (3, 4, 7), (4, 5, 10), (5, 1, 6), (6, 1, 9))

#: The even-period shapes, where nothing is padded.
UNPADDED = tuple(s for s in SHAPES if s[2] % 2 == 0)


def _instance(shape, dtype=jnp.float64) -> JrpProblem:
    seed, items, periods = shape
    return JrpProblem.random_instance(seed, items, periods, dtype=dtype)


def test_matches_reference() -> None:
    """Every instance converges to the LP optimum HiGHS reports."""
    params = Parameters.with_auto_step(None, TOLERANCE)
    for shape in SHAPES:
        instance = _instance(shape)
        x, _, _, iterations, _, _ = solve_jrp_problem(instance, params)
        ours = instance.objective_value(x)
        theirs = reference_relaxation_value(instance)
        relative = abs(ours - theirs) / (1.0 + abs(theirs))
        assert relative <= CLOSE, (shape, ours, theirs, relative)
        assert iterations > 0, shape


def test_the_reported_bracket_is_a_certificate() -> None:
    """The solver's own bounds straddle the true value, and closely.

    Not merely "the two numbers are close" -- a bound that came out on the
    wrong side of the optimum would still be close and would still be wrong.
    """
    params = Parameters.with_auto_step(None, TOLERANCE)
    for shape in SHAPES:
        instance = _instance(shape)
        x, _, _, _, lower, upper = solve_jrp_problem(instance, params)
        lower, upper = float(lower), float(upper)
        truth = reference_relaxation_value(instance)
        slack = CLOSE * (1.0 + abs(truth))
        assert lower <= truth + slack, (shape, lower, truth)
        assert upper >= truth - slack, (shape, upper, truth)
        gap = (upper - lower) / (1.0 + abs(upper) + abs(lower))
        assert gap <= CLOSE, (shape, gap)
        assert abs(upper - instance.objective_value(x)) <= 1e-9, shape


def test_recomputed_bounds_are_valid_and_tight_when_unpadded() -> None:
    """`relaxation_bounds` bounds any triple, and matches the solver unpadded.

    Two claims, because the function makes two.  It is always a *valid*
    bracket -- that is what lets it check a point from anywhere.  It is as
    tight as the solver's own only when no padding was applied, since the
    padded period's dual contribution is sliced off the returned arrays; the
    docstring says so, and this is what holds it to saying so.
    """
    params = Parameters.with_auto_step(None, TOLERANCE)
    for shape in SHAPES:
        instance = _instance(shape)
        x, y, z, _, solver_lower, _ = solve_jrp_problem(instance, params)
        lower, upper = relaxation_bounds(instance, x, y, z)
        truth = reference_relaxation_value(instance)
        slack = CLOSE * (1.0 + abs(truth))
        assert float(lower) <= truth + slack, (shape, float(lower), truth)
        assert float(upper) >= truth - slack, (shape, float(upper), truth)
        # Never tighter than the bound the solver converged against.
        assert float(lower) <= float(solver_lower) + slack, shape
        if shape in UNPADDED:
            assert abs(float(lower) - float(solver_lower)) <= slack, shape


def test_max_iter_is_honoured() -> None:
    """A capped solve stops at the cap instead of running to convergence.

    The cap is checked once per `op_steps` block, so the count lands within
    one block of it rather than exactly on it.
    """
    instance = _instance(SHAPES[3])
    capped = Parameters.with_auto_step(16, 1e-14)
    _, _, _, iterations, _, _ = solve_jrp_problem(instance, capped)
    assert 0 < iterations <= 16 + capped.op_steps, iterations


def test_a_zero_joint_cost_solves_rather_than_poisoning() -> None:
    """A single item has no joint order cost, and still solves.

    The dual set for the joint orders is then `{w >= 0 : sum w = 0}`, whose
    only point is zero.  Two things have to hold for that to be an ordinary
    run rather than a degenerate one: the projection onto it must give zero
    rather than a division by an empty active set, and a non-finite gap must
    count as not converged -- otherwise the first check passes and the solve
    returns its starting point.
    """
    params = Parameters.with_auto_step(None, TOLERANCE)
    for shape in ((5, 1, 6), (6, 1, 9)):
        instance = _instance(shape)
        assert float(instance.joint_order_cost) == 0.0, shape
        x, _, z, iterations, lower, upper = solve_jrp_problem(instance, params)
        assert iterations > 0, shape
        assert bool(jnp.all(jnp.isfinite(x))) and bool(jnp.all(jnp.isfinite(z))), shape
        assert float(jnp.abs(z).max()) == 0.0, "the zero simplex holds only zero"
        truth = reference_relaxation_value(instance)
        assert abs(float(upper) - truth) <= CLOSE * (1.0 + abs(truth)), shape
        assert float(lower) <= truth + CLOSE * (1.0 + abs(truth)), shape


def test_warm_start_from_a_solution_is_a_fixed_point() -> None:
    """Re-solving from a converged point costs nothing.

    The duals come back normalised, and feeding them straight back in is a
    fixed point only if that convention round trips -- so this is what holds
    the scaling on the way in and the scaling on the way out together.  An
    even period count takes zero iterations, exactly as pria-cuda does; an odd
    one is padded, the pad is sliced off the returned arrays, and the reprojected
    point is near the solution rather than on it, so it only costs less.
    """
    params = Parameters.with_auto_step(None, TOLERANCE)
    for seed, items, periods in ((7, 3, 8), (1, 2, 6), (3, 3, 7)):
        instance = JrpProblem.random_instance(seed, items, periods, dtype=jnp.float64)
        x, y, z, first, _, before = solve_jrp_problem(instance, params)
        _, _, _, again, _, after = solve_jrp_problem_warm_start(instance, params,
                                                                (x, y, z))
        assert abs(float(before) - float(after)) <= CLOSE * (1.0 + abs(float(before)))
        if periods % 2 == 0:
            assert again == 0, (periods, first, again)
        else:
            assert again < first, (periods, first, again)


def test_vmap_over_a_stack() -> None:
    """The batched path solves the same instances to the same answers.

    `JrpProblem.stack` and `Parameters.stack` exist for this, and the solver is
    only usable under `vmap` while every pytree field lines up.
    """
    problems = [JrpProblem.random_instance(seed, 3, 6, dtype=jnp.float64)
                for seed in (11, 12, 13)]
    params = [Parameters.with_auto_step(None, TOLERANCE) for _ in problems]
    stacked, stacked_params = JrpProblem.stack(problems), Parameters.stack(params)

    solve = vmap(solve_jrp_problem, in_axes=(0, 0), out_axes=(0, 0, 0, 0, 0, 0))
    started = time.perf_counter()
    compiled = jit(solve).trace(stacked, stacked_params).lower().compile()
    print(f"  vmap compiled in {time.perf_counter() - started:.1f}s")
    batched = block_until_ready(compiled(stacked, stacked_params))

    for index, instance in enumerate(problems):
        alone = solve_jrp_problem(instance, params[index])[0]
        together = batched[0][index]
        assert abs(instance.objective_value(alone)
                   - instance.objective_value(together)) <= CLOSE


if __name__ == "__main__":
    test_matches_reference()
    test_the_reported_bracket_is_a_certificate()
    test_recomputed_bounds_are_valid_and_tight_when_unpadded()
    test_max_iter_is_honoured()
    test_a_zero_joint_cost_solves_rather_than_poisoning()
    test_warm_start_from_a_solution_is_a_fixed_point()
    test_vmap_over_a_stack()
    print("test_solver: all tests passed")
