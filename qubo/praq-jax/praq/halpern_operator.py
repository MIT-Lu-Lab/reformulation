import jax.numpy as jnp
from jax import Array

from .half_edge import HalfEdge


def canonical_norm(dx: Array, dy: Array, half_edge: HalfEdge,
                   eta: float, omega: float) -> Array:
    """
    Returns || (dx, dy) ||_{P(eta, omega)} in the canonical norm of PDHG
    at primal weight omega.

    The cross term is POSITIVE, and the sign is NOT a free convention -- the step
    is nonexpansive in this metric and expansive in the other one (measured:
    ratio 0.976 vs 1.897). P is the proximal-point metric of one PDHG step, whose
    off-diagonal block is +A^T for the template min_x max_y <c,x> + <y, q - Ax>;
    PRAQ has A = S, giving +S^T here.

    Do not "mirror PRIA", which carries -I: PRIA's Lagrangian <H + Y + Z, X> has
    +Au against the template's -Au, so its generic-form A is the NEGATIVE of its
    coupling map and the two minus signs cancel. PRAQ's -Sx is genuine, so the
    sign survives. Both metrics have the same spectrum (they are conjugate by
    diag(I, -I)), so the PSD test and the eigenvalues cannot tell them apart --
    only nonexpansiveness can. See tests/test_halpern_operator.py.

    P is positive semidefinite exactly when eta <= 1, uniformly in omega.
    """
    deg, cross = half_edge.degree, jnp.sum(half_edge.gather(dx) * dy)

    val = (
        (omega / eta)         * jnp.sum(deg * dx * dx)
        + (1.0 / (eta*omega)) * jnp.sum(dy * dy)
        + 2.0                 * cross
    )
    return jnp.sqrt(jnp.maximum(val, 0.0))


def omega_argmin(dx: Array, dy: Array, half_edge: HalfEdge) -> Array:
    """Returns for an eta the argmin_{omega} of || (dx, dy) ||_{P(eta, omega)}."""
    return jnp.sqrt(jnp.sum(dy * dy) / jnp.sum(half_edge.degree * dx * dx))
