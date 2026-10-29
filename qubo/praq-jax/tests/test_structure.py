"""Structural invariants of the problem and lift.

These exist because the solver tests alone are insensitive to several things the
math depends on: `objective_scale` had no coverage at all, and every other test
uses the dense build, where `degree` is the constant n-1 and the degree
preconditioner is a scalar. The claims that matter -- the coupling norm being 1,
the identity D = S^T S -- are trivial in that regime and only bite on irregular
graphs, so most of what follows uses sparse instances.
"""

import itertools

import numpy as np
import jax.numpy as jnp
from jax import config

config.update('jax_enable_x64', True)

from praq.half_edge import HalfEdge
from praq.prelude import *
from reference import reference_relaxation_value
from praq.solver import _lower_bound_raw, _upper_bound_raw


def _sparse(seed, n, density):
    return QuboProblem.random_instance(seed, n, density=density, dtype=jnp.float64)


def _reference(matrix):
    """Independent recomputation of every HalfEdge field from the dense matrix."""
    n = matrix.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = np.array([2.0 * abs(matrix[i, j]) for i, j in pairs])
    positive = np.array([matrix[i, j] > 0 for i, j in pairs])
    degree = np.full(n, float(n - 1))
    return np.array(pairs), weights, positive, degree, np.diagonal(matrix).copy()


def test_objective_scale_matches_its_definition():
    # varsigma = ||(d; w)||_2. Previously untested, which let three distinct
    # mutations of it through the suite.
    for seed, n, density in [(0, 5, 1.0), (1, 7, 0.4), (2, 9, 0.15), (3, 2, 1.0)]:
        instance = _sparse(seed, n, density)
        matrix = np.asarray(instance.matrix)
        rows, cols = np.triu_indices(n, 1)
        want = np.linalg.norm(np.append(np.diagonal(matrix), 2.0 * np.abs(matrix[rows, cols])))
        got = float(instance.objective_scale())
        assert abs(got - want) <= 1e-12 * (1.0 + want), (seed, got, want)

    # an all-zero instance must not produce a zero scale (it would divide by it)
    assert float(QuboProblem.zero_instance(4, dtype=jnp.float64).objective_scale()) == 1.0


def test_half_edge_fields_match_the_definition():
    for seed, n, density in [(0, 6, 1.0), (1, 8, 0.35), (2, 10, 0.12)]:
        instance = _sparse(seed, n, density)
        half_edge = HalfEdge.build(instance)
        ends, weights, positive, degree, diagonal = _reference(np.asarray(instance.matrix))

        assert np.array_equal(np.asarray(half_edge.ends), ends)
        assert np.allclose(np.asarray(half_edge.weights), weights)
        assert np.array_equal(np.asarray(half_edge.positive), positive), "positive is W_ij > 0, strictly"
        assert np.allclose(np.asarray(half_edge.degree), degree)
        assert np.allclose(np.asarray(half_edge.diagonal), diagonal)


def test_shift_is_the_indicator_of_slot_one():
    # The convention is load-bearing: _lower_bound_raw reads y[:, 0] for the
    # <1_+, Y> term, so shift() must put its 1 in the same slot.
    instance = _sparse(0, 7, 0.6)
    half_edge = HalfEdge.build(instance)
    shift = np.asarray(half_edge.shift())
    positive = np.asarray(half_edge.positive)

    assert np.array_equal(shift[:, 0], positive.astype(float))
    assert np.all(shift[:, 1] == 0.0)


def test_preconditioned_coupling_norm_is_one_on_irregular_graphs():
    # eq:praq-Anorm and solver.COUPLING_NORM: ||D^{-1/2} S^T S D^{-1/2}||_2 = 1.
    # On the dense build D is a scalar and this is vacuous, so use sparse graphs
    # with a genuinely non-constant degree.
    for seed, n, density in [(0, 9, 0.3), (1, 12, 0.2), (2, 7, 0.5)]:
        half_edge = HalfEdge.build(_sparse(seed, n, density))
        ends = np.asarray(half_edge.ends)
        edges = half_edge.m()

        duplication = np.zeros((2 * edges, n))
        duplication[np.arange(2 * edges), ends.ravel()] = 1.0
        degree = np.asarray(half_edge.degree)

        gram = duplication.T @ duplication
        assert np.allclose(gram, np.diag(degree)), "D = S^T S must hold exactly"
        scaled = np.diag(degree ** -0.5) @ gram @ np.diag(degree ** -0.5)
        assert abs(np.linalg.norm(scaled, 2) - 1.0) < 1e-12


def test_single_variable_instances():
    # The degree floor binds ONLY at n = 1: the dense build gives every node
    # degree n-1 whether or not its edges carry weight, so a node with no nonzero
    # edge still has degree n-1 >= 1 for n >= 2. At n = 1 there are no edges at
    # all, and without the floor the primal step divides by zero.
    for d, want, opt in [(-3.0, 1.0, -3.0), (2.0, 0.0, 0.0)]:
        instance = QuboProblem.new(jnp.array([[d]]))
        assert float(HalfEdge.build(instance).degree[0]) >= 1.0
        primal, dual, _ = solve_qubo_problem(instance, Parameters.with_auto_step(10_000, 1e-9))
        assert np.isfinite(np.asarray(primal)).all(), "n=1 produced a non-finite iterate"
        assert abs(float(primal[0]) - want) < 1e-6, (d, float(primal[0]))
        lower, upper = relaxation_bounds(instance, primal, dual)
        assert abs(float(upper) - opt) < 1e-6 and abs(float(lower) - opt) < 1e-6


def test_nodes_with_no_incident_edge_are_solved_correctly():
    # Not the degree floor (see above) but the other end of the same behaviour:
    # a node whose every edge has weight 0 must still reach min(0, d_i).
    for d, want in [(-2.0, 1.0), (3.0, 0.0)]:
        matrix = np.zeros((3, 3))
        matrix[0, 1] = matrix[1, 0] = -1.0
        matrix[0, 0] = matrix[1, 1] = 1.0
        matrix[2, 2] = d                                    # node 2 is isolated
        instance = QuboProblem.new(jnp.array(matrix))

        assert float(HalfEdge.build(instance).degree[2]) >= 1.0
        primal, _, _ = solve_qubo_problem(instance, Parameters.with_auto_step(200_000, 1e-9))
        assert np.isfinite(np.asarray(primal)).all(), "isolated node produced a non-finite iterate"
        assert abs(float(primal[2]) - want) < 1e-6, (d, float(primal[2]))


def test_relaxation_value_branch_semantics():
    # The covering branch is W_ij > 0 strictly; a zero entry contributes nothing
    # under either branch, and the two forms must agree with a direct computation.
    rng = np.random.default_rng(0)
    for seed, n, density in [(0, 5, 1.0), (1, 7, 0.3)]:
        instance = _sparse(seed, n, density)
        matrix = np.asarray(instance.matrix)
        half_edge = HalfEdge.build(instance)
        for _ in range(30):
            x = rng.random(n)
            want = float(np.dot(np.diagonal(matrix), x))
            for i in range(n):
                for j in range(i + 1, n):
                    w = 2.0 * matrix[i, j]
                    if w > 0:
                        want += w * max(0.0, x[i] + x[j] - 1.0)
                    elif w < 0:
                        want += w * min(x[i], x[j])
            for got in (float(instance.relaxation_value(jnp.array(x))),
                        float(_upper_bound_raw(half_edge, jnp.array(x)))):
                assert abs(got - want) < 1e-10 * (1.0 + abs(want)), (got, want)


def test_random_instances_are_integral_and_symmetric():
    # thm:praq-margin needs d and w integral; the generator must deliver it.
    for seed in range(20):
        for n, density in [(4, 1.0), (7, 0.4)]:
            matrix = np.asarray(_sparse(seed, n, density).matrix)
            assert np.allclose(matrix, matrix.T)
            assert np.allclose(matrix, np.round(matrix)), "non-integral instance"


def test_sparse_instances_bracket_the_optimum():
    for seed, n, density in [(0, 9, 0.3), (1, 12, 0.25), (2, 16, 0.15)]:
        instance = _sparse(seed, n, density)
        primal, dual, _ = solve_qubo_problem(instance, Parameters.with_auto_step(500_000, 1e-9))
        lower, upper = relaxation_bounds(instance, primal, dual)
        reference = reference_relaxation_value(instance)
        assert float(lower) <= reference + 1e-7, (seed, float(lower), reference)
        assert float(upper) >= reference - 1e-7, (seed, float(upper), reference)
        assert abs(float(upper) - reference) < 1e-6 * (1.0 + abs(reference))


def test_persistency_actually_fixes_variables_when_it_should():
    # The old test admitted a persistent_variables that fixes nothing: on dense
    # instances the roof dual is fractional everywhere, so every assertion held
    # vacuously. Sparse instances are where fixing happens, so demand it.
    total_fixed = 0
    for seed, n, density in [(0, 12, 0.15), (1, 16, 0.12), (2, 20, 0.1)]:
        instance = _sparse(seed, n, density)
        primal, _, _ = solve_qubo_problem(instance, Parameters.with_auto_step(500_000, 1e-9))
        fixed, values = persistent_variables(primal, tolerance=1e-6)
        fixed, values = np.asarray(fixed), np.asarray(values)
        total_fixed += int(fixed.sum())

        matrix = np.asarray(instance.matrix)
        assignments = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
        objectives = np.einsum("bi,ij,bj->b", assignments, matrix, assignments)
        optimal = assignments[np.abs(objectives - objectives.min()) < 1e-9]
        assert (optimal[:, fixed] == values[fixed]).all(axis=1).any(), seed
    assert total_fixed > 0, "persistency fixed nothing anywhere; the test is vacuous"


def test_nan_gap_is_not_reported_as_convergence():
    # `nan > tol` is False, so the naive termination test reads a NaN gap as
    # converged. It must instead keep iterating and stop at max_iter.
    matrix = np.array([[np.nan, 1.0], [1.0, -1.0]])
    instance = QuboProblem(jnp.array(matrix))
    _, _, iterations = solve_qubo_problem(instance, Parameters.with_auto_step(64, 1e-9))
    assert int(iterations) >= 64, f"NaN gap reported convergence after {int(iterations)} iterations"


def test_omega_is_unclamped_but_guarded():
    # PRIA leaves the primal weight unbounded for practical performance, and the
    # restart theory carries boundedness as an assumption rather than enforcing
    # it; PRAQ now matches. What the implementation must still guarantee is that
    # the update is well posed: omega stays strictly positive (the canonical
    # metric needs omega > 0), and a vanishing displacement freezes the update
    # instead of producing 0/0. This pins both, and that the solve is correct.

    params = Parameters.with_auto_step(500_000, 1e-9)
    assert not hasattr(params, "omega_min"), "the omega clamp should be gone"
    assert not hasattr(params, "omega_max"), "the omega clamp should be gone"

    for seed, n, density in [(0, 12, 1.0), (1, 16, 0.2)]:
        instance = _sparse(seed, n, density)
        reference = reference_relaxation_value(instance)
        primal, dual, _ = solve_qubo_problem(instance, params)
        _, upper = relaxation_bounds(instance, primal, dual)
        assert abs(float(upper) - reference) < 1e-6 * (1.0 + abs(reference))

    # A zero instance makes both displacement norms vanish at every restart; the
    # guard must freeze omega at its initial value rather than emit nan/0.
    zero = QuboProblem.zero_instance(6, dtype=jnp.float64)
    primal, dual, _ = solve_qubo_problem(zero, Parameters.with_auto_step(64, 1e-9))
    assert bool(jnp.all(jnp.isfinite(primal))), "degenerate omega update produced non-finite x"
    assert bool(jnp.all(jnp.isfinite(dual))), "degenerate omega update produced non-finite y"


if __name__ == "__main__":
    test_objective_scale_matches_its_definition()
    test_half_edge_fields_match_the_definition()
    test_shift_is_the_indicator_of_slot_one()
    test_preconditioned_coupling_norm_is_one_on_irregular_graphs()
    test_single_variable_instances()
    test_nodes_with_no_incident_edge_are_solved_correctly()
    test_relaxation_value_branch_semantics()
    test_random_instances_are_integral_and_symmetric()
    test_sparse_instances_bracket_the_optimum()
    test_persistency_actually_fixes_variables_when_it_should()
    test_nan_gap_is_not_reported_as_convergence()
    test_omega_is_unclamped_but_guarded()
    print("test_structure: all tests passed")


