import numpy as np
import jax.numpy as jnp
from jax import config
from scipy.optimize import minimize_scalar

config.update('jax_enable_x64', True)

from praq.projection import project_dual, project_primal


def _reference_projection(v: np.ndarray, w: float, positive: bool) -> np.ndarray:
    """Project onto the one-dimensional edge set by direct line search."""
    if positive:
        def point(mu):
            return np.array([mu, mu])
        lo, hi = -w, 0.0
    else:
        def point(a):
            return np.array([a, w - a])
        lo, hi = 0.0, w

    if hi - lo <= 0.0:
        return point(lo)

    res = minimize_scalar(lambda s: float(np.sum((v - point(s)) ** 2)),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-13})
    return point(res.x)


def test_dual_projection_matches_line_search():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(400):
        w = float(rng.choice([0.0, 0.5, 1.0, 4.0]))
        positive = bool(rng.integers(2))
        v = rng.normal(scale=3.0, size=2)

        got = np.asarray(project_dual(jnp.array(v[None, :]),
                                      jnp.array([w]),
                                      jnp.array([positive])))[0]
        worst = max(worst, float(np.abs(got - _reference_projection(v, w, positive)).max()))
    assert worst < 1e-7, worst


def test_dual_projection_lands_in_the_set():
    rng = np.random.default_rng(1)
    w = jnp.array([0.0, 0.5, 1.0, 4.0, 2.5])
    positive = jnp.array([True, False, True, False, True])
    v = jnp.array(rng.normal(scale=3.0, size=(5, 2)))

    p = np.asarray(project_dual(v, w, positive))
    weights, is_pos = np.asarray(w), np.asarray(positive)

    covering, packing = p[is_pos], p[~is_pos]
    assert np.allclose(covering[:, 0], covering[:, 1]), covering
    assert (covering[:, 0] <= 1e-12).all() and (covering[:, 0] >= -weights[is_pos] - 1e-12).all()
    assert np.allclose(packing.sum(axis=1), weights[~is_pos]), packing
    assert (packing >= -1e-12).all(), packing


def test_projections_are_idempotent():
    rng = np.random.default_rng(2)
    w = jnp.array(rng.uniform(0.0, 3.0, size=17))
    positive = jnp.array(rng.integers(2, size=17).astype(bool))
    v = jnp.array(rng.normal(scale=2.0, size=(17, 2)))

    once = project_dual(v, w, positive)
    assert np.allclose(np.asarray(once), np.asarray(project_dual(once, w, positive)), atol=1e-12)

    u = jnp.array(rng.normal(size=23))
    once_primal = project_primal(u)
    assert np.allclose(np.asarray(once_primal), np.asarray(project_primal(once_primal)))


def test_zero_weight_edges_are_pinned_to_the_origin():
    # This is what lets the dense build carry absent edges (half_edge.build).
    v = jnp.array([[3.0, -2.0], [3.0, -2.0]])
    p = np.asarray(project_dual(v, jnp.zeros(2), jnp.array([True, False])))
    assert np.allclose(p, 0.0), p


if __name__ == "__main__":
    test_dual_projection_matches_line_search()
    test_dual_projection_lands_in_the_set()
    test_projections_are_idempotent()
    test_zero_weight_edges_are_pinned_to_the_origin()
    print("test_projection: all tests passed")
