import jax.numpy as jnp
import numpy as np
from jax import config

config.update('jax_enable_x64', True)

from pria.halpern_operator import canonical_norm


def _dense_metric(d: int, eta: float, omega: float) -> np.ndarray:
    """The proximal-point metric of one PDHG step: off-diagonal blocks are -I."""
    a, b = omega / eta, 1.0 / (eta * omega)
    m = np.zeros((3 * d, 3 * d))
    m[:d, :d] = a * np.eye(d)
    m[d:2 * d, d:2 * d] = b * np.eye(d)
    m[2 * d:, 2 * d:] = b * np.eye(d)
    m[:d, d:2 * d] = m[d:2 * d, :d] = -np.eye(d)
    m[:d, 2 * d:] = m[2 * d:, :d] = -np.eye(d)
    return m


def test_canonical_norm_matches_dense_metric():
    rng = np.random.default_rng(0)
    d = 11
    for eta, omega in [(0.999 / np.sqrt(2.0), 0.10835), (0.5, 1.0), (0.6, 3.0)]:
        x, y, z = rng.normal(size=(3, d))
        v = np.concatenate([x, y, z])
        want = np.sqrt(max(v @ _dense_metric(d, eta, omega) @ v, 0.0))
        got = float(canonical_norm(jnp.array(x), jnp.array(y), jnp.array(z), eta, omega))
        assert abs(got - want) <= 1e-10 * (1.0 + want), (eta, omega, got, want)


def test_metric_psd_exactly_up_to_step_bound():
    # PSD iff eta <= 1/sqrt(2), for every omega.
    for omega in (0.01, 0.10835, 1.0, 10.0):
        for eta, expect_psd in [(0.999 / np.sqrt(2.0), True),
                                (1.0 / np.sqrt(2.0), True),
                                (1.001 / np.sqrt(2.0), False)]:
            evs = np.linalg.eigvalsh(_dense_metric(1, eta, omega))
            assert (evs.min() >= -1e-12) == expect_psd, (eta, omega, evs)


def test_cross_term_sign_is_negative():
    # The cross term is negative, so aligned positive x and (y, z) must
    # DECREASE the norm relative to the diagonal-only value.  A flipped sign
    # leaves a norm that still looks plausible, which is why it is pinned
    # here rather than left to the solver's behaviour to reveal.
    d = 4
    x = jnp.ones(d)
    y = jnp.ones(d)
    z = jnp.ones(d)
    eta, omega = 0.999 / np.sqrt(2.0), 0.10835
    a, b = omega / eta, 1.0 / (eta * omega)
    diag_only = np.sqrt(d * (a + 2 * b))
    got = float(canonical_norm(x, y, z, eta, omega))
    assert got < diag_only - 1e-9


if __name__ == "__main__":
    test_canonical_norm_matches_dense_metric()
    test_metric_psd_exactly_up_to_step_bound()
    test_cross_term_sign_is_negative()
    print("test_halpern_operator: all tests passed")
