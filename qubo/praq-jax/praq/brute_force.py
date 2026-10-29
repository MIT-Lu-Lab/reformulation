import numpy as np

import jax.numpy as jnp

from .qubo_problem import QuboProblem


def solve_qubo_brute_force(instance: QuboProblem) -> tuple[jnp.ndarray, float]:
    """
    Enumerate {0, 1}^n. Exponential; for tests on small instances only.

    The LP reference lives in the experiment repository's
    `baselines/highs_backend.py`, alongside Gurobi and cuPDLPx; keeping it out
    of the library is what lets `praq` install without scipy.

    :param instance: the QUBO instance
    :return: (an optimal binary solution, its objective value)
    """
    matrix = np.asarray(instance.matrix, dtype=float)
    n = matrix.shape[0]

    assignments = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    values = np.einsum("bi,ij,bj->b", assignments, matrix, assignments)
    best = int(values.argmin())

    return jnp.asarray(assignments[best]), float(values[best])
