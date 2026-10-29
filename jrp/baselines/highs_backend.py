"""Solve the joint replenishment LP with HiGHS, through scipy's `linprog`.

The cheapest reference to have available: no licence, no GPU, and scipy is
required by `baselines.model` anyway, so every LP baseline already needs it.
It is a baseline like the others, which is why the PRIA library itself does
not carry one.
"""

from __future__ import annotations

import time

from scipy.optimize import linprog

from baselines import model
from harness.config import TOLERANCES, Tolerances
from harness.results import Result

#: Interior point, and still a vertex: HiGHS runs crossover after it.  It is
#: the faster of the two on this LP by a wide margin -- the demand equalities
#: make the basis large, and dual simplex spends its iterations rediscovering
#: a structure the barrier sees at once -- and "highs-ds" is the other spelling
#: if a basis is ever wanted for its own sake.
METHOD = "highs-ipm"

#: HiGHS rejects either tolerance below this, and its idea of rejecting is to
#: warn and quietly keep its own 1e-7 default -- so a request under the floor
#: would otherwise be recorded as asked for and solved at something else.
#: Clamped instead, and the clamped value is what the record carries.  Gurobi
#: has the same arrangement one decade up.
TOLERANCE_FLOOR = 1e-10


def _split(lp):
    """The two-sided rows of `lp` as the `A_ub`/`A_eq` pair linprog wants."""
    csr = lp.matrix.tocsr()
    equality = lp.row_lower == lp.row_upper
    inequality = ~equality
    return (csr[inequality], lp.row_upper[inequality],
            csr[equality], lp.row_upper[equality])


def solve(instance, *, tolerances: Tolerances, time_limit: float) -> Result:
    """Solve the LP of `instance` with HiGHS.

    A vertex solve, so it stops on optimality rather than at a gap; both
    tolerances are still requested, not merely recorded.  HiGHS splits them
    the way Gurobi does under different names -- primal feasibility is
    constraint satisfaction, dual feasibility is the reduced-cost test that
    says a vertex is optimal -- so `feasibility` and `optimality` map onto
    those two, and `optimality` also sets the interior point method's own
    stopping tolerance.  HiGHS defaults all three to 1e-7 or looser.
    """
    optimality = max(tolerances.optimality, TOLERANCE_FLOOR)
    feasibility = max(tolerances.feasibility, TOLERANCE_FLOOR)
    common = dict(backend="highs", instance=instance.name, family=instance.family,
                  items=instance.items, periods=instance.periods,
                  entries=instance.entries,
                  optimality_tolerance=optimality, feasibility_tolerance=feasibility)

    start = time.perf_counter()
    lp = model.build(instance)
    a_ub, b_ub, a_eq, b_eq = _split(lp)
    setup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    outcome = linprog(lp.cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                      bounds=(0.0, None), method=METHOD,
                      options={"time_limit": time_limit,
                               "primal_feasibility_tolerance": feasibility,
                               "dual_feasibility_tolerance": optimality,
                               "ipm_optimality_tolerance": optimality})
    solve_seconds = time.perf_counter() - start

    if outcome.status == 1:
        return Result(status="time_limit", setup_seconds=setup_seconds,
                      solve_seconds=solve_seconds, **common)
    if not outcome.success:
        return Result(status="error", note=str(outcome.message)[:200], **common)

    iterations = int(getattr(outcome, "nit", -1) or -1)
    return Result(status="optimal", objective=float(outcome.fun),
                  lower=float(outcome.fun), upper=float(outcome.fun),
                  iterations=iterations, setup_seconds=setup_seconds,
                  solve_seconds=solve_seconds, **common)


def relaxation_value(instance) -> float:
    """The LP's optimal value, for use as a test reference.

    At the repository's default tolerances: a reference solved to looser ones
    than the thing being checked against it would be the wrong way round.
    """
    outcome = solve(instance, tolerances=TOLERANCES, time_limit=float("inf"))
    if outcome.status != "optimal":
        raise RuntimeError(f"HiGHS reference failed: {outcome.status} {outcome.note}")
    return outcome.objective
