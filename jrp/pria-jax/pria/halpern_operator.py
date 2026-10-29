import jax.numpy as jnp
from jax import Array


def canonical_norm(x, y, z, eta: float, omega: float) -> jnp.ndarray:
    """Returns || (X, Y, Z) ||_{P(eta, omega)}.

    The cross term is NEGATIVE: P is the proximal-point metric of one PDHG step
    (off-diagonal blocks -I), the norm in which the step is firmly nonexpansive
    and the restart residual provably monotone. See convergence-notes/findings.md.
    """
    xf, yf, zf = x.ravel(), y.ravel(), z.ravel()

    val = (
        (omega / eta)         * jnp.dot(xf, xf)
        + (1.0 / (eta*omega)) * (jnp.dot(yf, yf) + jnp.dot(zf, zf))
        - 2.0                 * (jnp.dot(xf, yf) + jnp.dot(xf, zf))
    )
    return jnp.sqrt(jnp.maximum(val, 0.0))


def omega_argmin(x: Array, y: Array, z: Array) -> jnp.ndarray:
    """Returns for an eta the argmin_{omega} of || (X, Y, Z) ||_{P(eta, omega)}"""
    return jnp.sqrt(((y*y).sum() + (z*z).sum()) / (x*x).sum())

