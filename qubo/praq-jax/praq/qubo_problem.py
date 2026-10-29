import os
from dataclasses import dataclass
from functools import partial

assertions_enabled = os.getenv('ASSERTIONS_ENABLED') is not None

import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax import random, Array


@partial(register_dataclass, data_fields=["matrix"], meta_fields=[])
@dataclass(frozen=True)
class QuboProblem:
    """An instance of min_{x in {0,1}^n} x^T W x, stored as the dense W."""

    matrix: Array

    def n(self) -> int:
        return self.matrix.shape[-1]

    @classmethod
    def new(cls, matrix: Array) -> "QuboProblem":
        """
        Construct a QUBO instance

            minimize (over x in {0,1}^n)  x^T W x,

        where W == matrix. Only the symmetric part of `matrix` is meaningful, so
        the input is symmetrized on the way in.

        The diagonal is not linearized: because x_i^2 = x_i on {0,1}, W_ii
        contributes the linear term W_ii x_i, and routing it through a product
        variable instead would weaken the relaxation at x_i = 1/2. See
        the paper's remark on the three easily-omitted bounds, item (iii).

        :param matrix: the (n, n) cost matrix W
        :return: the QuboProblem
        """
        matrix = 0.5 * (matrix + matrix.swapaxes(-1, -2))

        if assertions_enabled:
            assert matrix.ndim == 2, "matrix should have 2 dimensions"
            assert matrix.shape[0] == matrix.shape[1], "matrix should be square"
            assert matrix.shape[0] > 0, "matrix is empty"

        return cls(matrix)

    @classmethod
    def stack(cls, instances: list["QuboProblem"]) -> "QuboProblem":
        """
        Convert a struct-of-arrays `instances` to an array-of-structs.

        :param instances: a list of QuboProblems to stack
        :return: a QuboProblem whose field is an array with the input data
        """
        return cls(jnp.stack([i.matrix for i in instances]))

    @classmethod
    def random_instance(cls, seed: int, n: int, density: float = 1.0,
                        magnitude: float = 3.0, dtype=jnp.float32) -> "QuboProblem":
        """
        Construct a random symmetric instance with integer-valued entries drawn
        uniformly from +-[1, magnitude], off-diagonal entries kept with
        probability `density`.

        Note that float32 carries roughly 1e-4 of relative accuracy on these
        problems; request float64 (with jax_enable_x64) for tighter tolerances.

        :param seed: an integer seed for the generation
        :param n: the number of binary variables
        :param density: probability that an off-diagonal entry is nonzero
        :param magnitude: entries are drawn from +-[1, magnitude]
        :param dtype: the type of floats to use
        :return: a random QuboProblem
        """
        value_key, sign_key, mask_key = random.split(random.PRNGKey(seed), 3)

        values = jnp.round(random.uniform(value_key, (n, n), dtype, 1.0, magnitude + 1.0))
        values = jnp.minimum(values, magnitude)
        signs = jnp.where(random.bernoulli(sign_key, 0.5, (n, n)), 1.0, -1.0).astype(dtype)
        keep = random.bernoulli(mask_key, density, (n, n)) | jnp.eye(n, dtype=bool)

        matrix = jnp.where(keep, values * signs, 0.0)
        return cls.new(jnp.triu(matrix) + jnp.triu(matrix, 1).T)

    @classmethod
    def zero_instance(cls, n: int, dtype=jnp.float32) -> "QuboProblem":
        return cls.new(jnp.zeros((n, n), dtype=dtype))

    def objective_value(self, x: Array) -> Array:
        """x^T W x. Meaningful for binary x; use relaxation_value for fractional x."""
        return jnp.einsum("i,ij,j->", x, self.matrix, x)

    def relaxation_value(self, x: Array) -> Array:
        """
        F(x), the objective of the relaxation after the product variables are
        minimized out in closed form. Agrees with
        objective_value on binary x and is the tightest convex piecewise-linear
        underestimator reachable by the standard linearization.
        """
        w = self.matrix
        pair_sum = x[:, None] + x[None, :]
        pair_min = jnp.minimum(x[:, None], x[None, :])

        # Summing over ALL off-diagonal (i, j) double-counts each edge, which is
        # exactly the factor 2 in w_e = 2 |W_ij|.
        off = jnp.where(w > 0.0, w * jnp.maximum(pair_sum - 1.0, 0.0), w * pair_min)
        off = off - jnp.diag(jnp.diagonal(off))

        return jnp.dot(jnp.diagonal(w), x) + off.sum()

    def objective_scale(self) -> Array:
        """
        The norm of the objective data, used to rescale an instance so that the
        domain diameter bound of the paper is dimensionless. Matches
        PRIA's varsigma = ||(vec(H); K; K0)||_2.
        """
        diagonal = jnp.diagonal(self.matrix)
        weights = 2.0 * jnp.abs(jnp.triu(self.matrix, 1))
        scale = jnp.linalg.norm(jnp.append(diagonal, weights.ravel()))
        # an all-zero instance has scale 0; any positive constant is a valid
        # normalization, and 1 keeps the data (all zeros) unchanged
        return jnp.where(scale > 0.0, scale, 1.0)

    def normalized(self) -> "QuboProblem":
        return QuboProblem(self.matrix / self.objective_scale())

    def as_type(self, dtype: jnp.dtype) -> "QuboProblem":
        return QuboProblem(self.matrix.astype(dtype))
