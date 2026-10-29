import numpy as np
import jax.numpy as jnp
from jax import config

config.update('jax_enable_x64', True)

from praq.halpern_operator import canonical_norm, omega_argmin
from praq.half_edge import HalfEdge
from praq.qubo_problem import QuboProblem


def _structure(n: int, seed: int = 0) -> HalfEdge:
    return HalfEdge.build(QuboProblem.random_instance(seed, n, dtype=jnp.float64))


def _duplication_matrix(half_edge: HalfEdge) -> np.ndarray:
    """The dense S: R^n -> R^(2m), rows ordered as (e, k) row-major."""
    ends = np.asarray(half_edge.ends)
    s = np.zeros((ends.size, half_edge.n()))
    s[np.arange(ends.size), ends.ravel()] = 1.0
    return s


def _dense_metric(half_edge: HalfEdge, eta: float, omega: float) -> np.ndarray:
    """The proximal-point metric of one PDHG step: off-diagonal blocks +S^T, +S."""
    s = _duplication_matrix(half_edge)
    d = np.diag(np.asarray(half_edge.degree))
    a, b = omega / eta, 1.0 / (eta * omega)

    n, two_m = half_edge.n(), s.shape[0]
    m = np.zeros((n + two_m, n + two_m))
    m[:n, :n] = a * d
    m[n:, n:] = b * np.eye(two_m)
    m[:n, n:] = s.T
    m[n:, :n] = s
    return m


def test_canonical_norm_matches_dense_metric():
    rng = np.random.default_rng(0)
    half_edge = _structure(6)
    n, edges = half_edge.n(), half_edge.m()

    for eta, omega in [(0.999, 1.0), (0.5, 0.25), (0.6, 3.0)]:
        dx = rng.normal(size=n)
        dy = rng.normal(size=(edges, 2))
        v = np.concatenate([dx, dy.ravel()])

        want = np.sqrt(max(v @ _dense_metric(half_edge, eta, omega) @ v, 0.0))
        got = float(canonical_norm(jnp.array(dx), jnp.array(dy), half_edge, eta, omega))
        assert abs(got - want) <= 1e-10 * (1.0 + want), (eta, omega, got, want)


def test_metric_psd_exactly_up_to_step_bound():
    # PSD iff eta <= 1, for every omega -- the sqrt(2) gain over PRIA, which
    # has two dual blocks and so needs eta <= 1/sqrt(2).
    half_edge = _structure(5)
    for omega in (0.01, 0.5, 1.0, 10.0):
        for eta, expect_psd in [(0.99, True), (1.0, True), (1.001, False)]:
            evs = np.linalg.eigvalsh(_dense_metric(half_edge, eta, omega))
            assert (evs.min() >= -1e-10) == expect_psd, (eta, omega, evs.min())


def test_spectrum_matches_proposition():
    # prop:praq-spectrum: eigenvalues relative to diag(D, I) are lambda_pm with
    # multiplicity n each, and b with multiplicity 2m - n.
    half_edge = _structure(5)
    n, two_m = half_edge.n(), 2 * half_edge.m()
    eta, omega = 0.999, 0.7
    a, b = omega / eta, 1.0 / (eta * omega)

    root = np.sqrt(((a - b) / 2.0) ** 2 + 1.0)
    want = np.sort(np.concatenate([
        np.full(n, (a + b) / 2.0 - root),
        np.full(n, (a + b) / 2.0 + root),
        np.full(two_m - n, b),
    ]))

    scale = np.diag(np.concatenate([np.asarray(half_edge.degree), np.ones(two_m)]))
    got = np.sort(np.linalg.eigvals(np.linalg.solve(scale, _dense_metric(half_edge, eta, omega))).real)

    assert np.allclose(got, want, atol=1e-9), (got, want)


def test_step_is_nonexpansive_in_the_metric():
    # THE test for the cross-term sign. Both signs give the same spectrum (they
    # are conjugate by diag(I, -I)), so PSD and eigenvalue checks cannot tell
    # them apart; only nonexpansiveness of the PDHG step can. Regression for the
    # sign that was wrong on first writing.
    from praq.projection import project_dual
    from praq.solver import _raw_step

    rng = np.random.default_rng(0)
    half_edge = _structure(7)
    n, edges = half_edge.n(), half_edge.m()
    shift = half_edge.shift()
    eta, omega = 0.99, 1.0

    def draw():
        x = jnp.array(rng.random(n))
        y = project_dual(jnp.array(rng.normal(size=(edges, 2))),
                         half_edge.weights, half_edge.positive)
        return x, y

    worst = 0.0
    for _ in range(2000):
        x1, y1 = draw()
        x2, y2 = draw()
        before = float(canonical_norm(x1 - x2, y1 - y2, half_edge, eta, omega))
        if before <= 1e-9:
            continue
        u1, v1 = _raw_step(x1, y1, half_edge, shift, omega, eta)
        u2, v2 = _raw_step(x2, y2, half_edge, shift, omega, eta)
        after = float(canonical_norm(u1 - u2, v1 - v2, half_edge, eta, omega))
        worst = max(worst, after / before)
    assert worst <= 1.0 + 1e-9, f"PDHG expands in the canonical metric: ratio {worst}"


def test_omega_argmin_minimizes_the_norm():
    rng = np.random.default_rng(3)
    half_edge = _structure(5)

    dx = jnp.array(rng.normal(size=half_edge.n()))
    dy = jnp.array(rng.normal(size=(half_edge.m(), 2)))

    best = float(omega_argmin(dx, dy, half_edge))
    at_best = float(canonical_norm(dx, dy, half_edge, 0.999, best))

    for factor in (0.5, 0.9, 1.1, 2.0):
        other = float(canonical_norm(dx, dy, half_edge, 0.999, best * factor))
        assert at_best <= other + 1e-12, (factor, at_best, other)


if __name__ == "__main__":
    test_canonical_norm_matches_dense_metric()
    test_metric_psd_exactly_up_to_step_bound()
    test_spectrum_matches_proposition()
    test_step_is_nonexpansive_in_the_metric()
    test_omega_argmin_minimizes_the_norm()
    print("test_halpern_operator: all tests passed")
