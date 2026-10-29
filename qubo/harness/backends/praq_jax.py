"""Run the JAX PRAQ implementation on one instance."""

from __future__ import annotations

import os
import sys
import time

from harness.config import Tolerances, repo_root
from harness.results import Result

#: `QuboProblem` stores W densely, so an instance costs 8 n^2 bytes to load.
#: Above this the adapter reports `unavailable` instead of exhausting memory.
#: Lifting the limit means teaching praq-jax a sparse instance type, not
#: raising this number.
MAX_DENSE_N = 12_000


def _import_praq():
    """Import `praq`, adding the in-repo checkout to the path if it is not installed.

    The retry drops any partially-resolved `praq` from `sys.modules` first.  A
    bare directory named `praq` anywhere on `sys.path` is enough for Python to
    bind the name to an empty namespace package, which then shadows the real
    one; this is why the repository root is not itself called `praq`.
    """
    try:
        import praq.prelude  # noqa: F401
    except ModuleNotFoundError:
        for name in [k for k in sys.modules if k == "praq" or k.startswith("praq.")]:
            del sys.modules[name]
        sys.path.insert(0, str(repo_root() / "praq-jax"))

    from jax import config
    config.update("jax_enable_x64", True)   # tolerances here are tighter than float32

    import praq.prelude as prelude
    return prelude


def solve(instance, *, tolerances: Tolerances, time_limit: float) -> Result:
    """Solve the relaxation of `instance` with praq-jax.

    Only `tolerances.optimality` reaches the solver, as the relative duality gap
    it stops on.  There is no feasibility tolerance to pass: every step projects
    onto [0, 1]^n and onto the dual set, so the iterates are feasible throughout
    and the column stays NaN rather than claiming a number that was never used.

    `time_limit` is recorded but not enforced: the solver's only budget is its
    iteration cap, and it does not check the clock inside the compiled loop.
    """
    common = dict(backend="praq-jax", instance=instance.name,
                  family=instance.family, n=instance.n, edges=instance.edges,
                  optimality_tolerance=tolerances.optimality)

    if instance.n > MAX_DENSE_N:
        return Result(status="unavailable", note=f"n > MAX_DENSE_N ({MAX_DENSE_N})",
                      **common)

    prelude = _import_praq()
    import jax.numpy as jnp
    from jax import block_until_ready

    start = time.perf_counter()
    problem = prelude.QuboProblem.new(jnp.asarray(instance.to_dense(), dtype=jnp.float64))
    params = prelude.Parameters.with_auto_step(max_iter=None,
                                              tolerance=tolerances.optimality)
    block_until_ready(prelude.solve_qubo_problem(problem, params))  # compile
    setup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    primal, dual, iterations = block_until_ready(
        prelude.solve_qubo_problem(problem, params))
    solve_seconds = time.perf_counter() - start

    lower, upper = prelude.relaxation_bounds(problem, primal, dual)
    return Result(status="optimal", objective=float(upper), lower=float(lower),
                  upper=float(upper), iterations=int(iterations),
                  setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                  **common)
