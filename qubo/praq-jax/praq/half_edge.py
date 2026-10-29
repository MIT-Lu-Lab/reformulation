from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax import Array

from .qubo_problem import QuboProblem


@partial(register_dataclass,
         data_fields=["ends", "weights", "positive", "degree", "diagonal"],
         meta_fields=[])
@dataclass(frozen=True)
class HalfEdge:
    """
    The group-indexed (half-edge) structure of a QUBO instance.

    Each edge e = {i, j} with i < j owns two half-edges, (e, 1) at node i and
    (e, 2) at node j, so arrays over half-edges have shape (m, 2). The
    duplication map S: R^n -> R^(m, 2) is `gather` and its transpose is
    `scatter`; these are the only two non-elementwise operations in the PRAQ
    iteration. See the paper's group-indexed lift.

    The build is dense: every pair i < j gets an edge, and an absent edge simply
    carries weight 0, whose dual set collapses to the single point {(0, 0)}.
    That keeps every shape static under jit at the cost of O(n^2) work.
    """

    ends: Array       # (m, 2) node index of each half-edge
    weights: Array    # (m,)   w_e = 2 |W_ij| >= 0
    positive: Array   # (m,)   whether W_ij > 0
    degree: Array     # (n,)   incident half-edges, floored at 1 (see build)
    diagonal: Array   # (n,)   d_i = W_ii

    def n(self) -> int:
        return self.diagonal.shape[-1]

    def m(self) -> int:
        return self.weights.shape[-1]

    @classmethod
    def build(cls, problem: QuboProblem) -> "HalfEdge":
        """
        Build the half-edge structure of `problem`.

        The degree is floored at 1 so that an isolated node still takes a step.
        Convergence needs only D >= S^T S, which the floor preserves; the two
        agree wherever a node has an edge, which in the dense build is
        everywhere as soon as n >= 2.

        :param problem: the QUBO instance
        :return: its HalfEdge structure
        """
        matrix = problem.matrix
        dtype = matrix.dtype
        n = problem.n()

        rows, cols = jnp.triu_indices(n, k=1)
        ends = jnp.stack([rows, cols], axis=1)

        off_diagonal = matrix[rows, cols]
        weights = 2.0 * jnp.abs(off_diagonal)
        positive = off_diagonal > 0.0

        incident = jnp.zeros(n, dtype=dtype).at[ends].add(jnp.ones_like(weights)[:, None])
        degree = jnp.maximum(incident, 1.0)

        return cls(ends, weights, positive, degree, jnp.diagonal(matrix))

    def gather(self, x: Array) -> Array:
        """S x: broadcast a node vector onto its half-edges."""
        return x[self.ends]

    def scatter(self, v: Array) -> Array:
        """S^T v: accumulate a half-edge array back onto nodes."""
        return jnp.zeros(self.n(), dtype=v.dtype).at[self.ends].add(v)

    def shift(self) -> Array:
        """
        The shift q = 1_+ of the saddle form: 1 in slot (e, 1) of every covering
        edge, 0 elsewhere.
        """
        positive = self.positive.astype(self.weights.dtype)
        return jnp.stack([positive, jnp.zeros_like(positive)], axis=1)

    def centroid(self) -> Array:
        """
        The centre of the dual set V, edge by edge: (w/2, w/2) on a packing
        edge and (-w/2, -w/2) on a covering one, which is (-W_ij, -W_ij) in
        both cases. An absent edge gets (0, 0), the whole of its dual set.
        """
        half = 0.5 * jnp.where(self.positive, -self.weights, self.weights)
        return jnp.stack([half, half], axis=1)

    def scaled(self, scale: Array) -> "HalfEdge":
        """Divide the objective data by `scale`; the structure is unchanged."""
        return HalfEdge(self.ends, self.weights / scale, self.positive,
                        self.degree, self.diagonal / scale)
