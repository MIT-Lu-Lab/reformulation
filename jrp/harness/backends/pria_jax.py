"""Run the JAX PRIA implementation on one instance."""

from __future__ import annotations

import sys
import time

from harness.config import Tolerances, repo_root
from harness.results import Result


def _import_pria():
    """Import `pria`, adding the in-repo checkout to the path if it is not installed.

    The retry drops any partially-resolved `pria` from `sys.modules` first.  A
    bare directory named `pria` anywhere on `sys.path` is enough for Python to
    bind the name to an empty namespace package, which then shadows the real
    one; this is why the repository root is not itself called `pria`.
    """
    try:
        import pria.prelude  # noqa: F401
    except ModuleNotFoundError:
        for name in [k for k in sys.modules if k == "pria" or k.startswith("pria.")]:
            del sys.modules[name]
        sys.path.insert(0, str(repo_root() / "pria-jax"))

    from jax import config
    config.update("jax_enable_x64", True)   # tolerances here are tighter than float32

    import pria.prelude as prelude
    return prelude


def solve(instance, *, tolerances: Tolerances, time_limit: float) -> Result:
    """Solve the LP of `instance` with pria-jax.

    Only `tolerances.optimality` reaches the solver, as the relative duality
    gap it stops on.  There is no feasibility tolerance to pass: every step
    projects onto the primal and dual simplices, so the iterates are feasible
    throughout and the column stays NaN rather than claiming a number that was
    never used.

    `setup_seconds` is the compile, which jax caches per *shape* -- so it is
    paid once per rung of the ladder and is near zero for every instance after
    the first of its shape.  That is why it is recorded separately and why
    `harness.figures` refuses to compare it across backends.

    `time_limit` is recorded but not enforced: the solver's only budget is its
    iteration cap, and it does not check the clock inside the compiled loop.
    """
    common = dict(backend="pria-jax", instance=instance.name,
                  family=instance.family, items=instance.items,
                  periods=instance.periods, entries=instance.entries,
                  optimality_tolerance=tolerances.optimality)

    prelude = _import_pria()
    import jax.numpy as jnp
    from jax import block_until_ready

    start = time.perf_counter()
    problem = prelude.JrpProblem.new(
        jnp.asarray(instance.order_costs, dtype=jnp.float64),
        float(instance.joint_order_cost),
        jnp.asarray(instance.holding_costs, dtype=jnp.float64))
    params = prelude.Parameters.with_auto_step(max_iter=None,
                                               tolerance=tolerances.optimality)
    block_until_ready(prelude.solve_jrp_problem(problem, params))   # compile
    setup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    _, _, _, iterations, lower, upper = block_until_ready(
        prelude.solve_jrp_problem(problem, params))
    solve_seconds = time.perf_counter() - start

    return Result(status="optimal", objective=float(upper), lower=float(lower),
                  upper=float(upper), iterations=int(iterations),
                  setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                  **common)
