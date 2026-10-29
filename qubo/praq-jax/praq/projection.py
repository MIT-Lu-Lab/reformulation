import jax.numpy as jnp
from jax import Array


def project_primal(v: Array) -> Array:
    """
    Euclidean projection onto the box [0, 1]^n.

    This is also the projection in the degree-weighted metric the PRAQ primal
    step uses, since any positive diagonal metric leaves a coordinatewise clip
    unchanged.
    """
    return jnp.clip(v, 0.0, 1.0)


def project_dual(v: Array, weights: Array, positive: Array) -> Array:
    """
    Euclidean projection onto the dual set V of the PRAQ saddle form.

    Per edge this is one of two closed forms, both O(1) with no sort:

      packing edge (W_ij < 0): the weighted 2-simplex {Y >= 0, Y_1 + Y_2 = w_e},
        reached by moving onto the line and clipping the free coordinate;
      covering edge (W_ij > 0): the segment {(mu, mu) : -w_e <= mu <= 0},
        reached by averaging the pair and clipping.

    An edge of weight 0 lands on {(0, 0)} under either branch, which is what
    lets absent edges be carried in the dense build.

    :param v: (..., m, 2) point to project
    :param weights: (..., m) edge weights w_e
    :param positive: (..., m) whether the edge is a covering edge
    :return: the projection, same shape as v
    """
    line = v[..., 0] + 0.5 * (weights - v[..., 0] - v[..., 1])
    first = jnp.clip(line, 0.0, weights)
    simplex = jnp.stack([first, weights - first], axis=-1)

    mu = jnp.clip(0.5 * (v[..., 0] + v[..., 1]), -weights, 0.0)
    segment = jnp.stack([mu, mu], axis=-1)

    return jnp.where(positive[..., None], segment, simplex)
