"""Solve the joint replenishment LP with Gurobi."""

from __future__ import annotations

import time

from baselines import model
from harness.config import Tolerances, gurobi_threads
from harness.results import Result

#: Gurobi LP method: 2 = barrier, with the crossover Gurobi runs after it by
#: default, so the returned primal is a vertex.
#:
#: Barrier and not dual simplex, which is what the QUBO sibling of this
#: repository uses.  The two LPs are shaped differently: this one's `N T`
#: demand equalities put a row through every x column, so a basis is enormous
#: and simplex spends its iterations rediscovering structure the barrier's
#: factorisation sees at once.  Concurrent (-1, Gurobi's own default) is not
#: used because it runs several methods at once and the wall clock it reports
#: is then not any one method's.
METHOD_BARRIER = 2

#: Gurobi refuses either tolerance below this.  A request tighter than its own
#: floor is clamped rather than raising, so that one set of tolerances can be
#: handed to every backend and each takes what it can.
TOLERANCE_FLOOR = 1e-9


def solve(instance, *, tolerances: Tolerances, time_limit: float) -> Result:
    """Solve the LP of `instance` with Gurobi.

    A *vertex* solve, so it stops on optimality rather than at a gap, but both
    tolerances are requested: OptimalityTol is the reduced-cost test and
    FeasibilityTol is constraint satisfaction, which is the same split
    `Tolerances` makes.  They are recorded too, so timings against a
    first-order backend can be read with them in mind.

    `setup_seconds` is building the LP and handing it to Gurobi, which is this
    harness's overhead and not the solver's; `solve_seconds` is Gurobi's own
    `Runtime` for the `optimize()` call, which excludes model construction and
    reading the solution back -- neither of which any other backend's reported
    time includes either.
    """
    optimality = max(tolerances.optimality, TOLERANCE_FLOOR)
    feasibility = max(tolerances.feasibility, TOLERANCE_FLOOR)
    common = dict(backend="gurobi", instance=instance.name, family=instance.family,
                  items=instance.items, periods=instance.periods,
                  entries=instance.entries,
                  optimality_tolerance=optimality, feasibility_tolerance=feasibility)
    try:
        import gurobipy as gp
    except ModuleNotFoundError:
        return Result(status="unavailable", note="gurobipy not installed", **common)

    start = time.perf_counter()
    lp = model.build(instance)
    try:
        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model(env=env)
        columns = m.addMVar(lp.columns, lb=lp.column_lower, ub=lp.column_upper,
                            obj=lp.cost)
        matrix = lp.matrix.tocsr()
        # The demand rows are equalities and the coupling rows are "<=", so the
        # two blocks go in separately rather than as one ranged constraint.
        equality = lp.row_lower == lp.row_upper
        m.addConstr(matrix[~equality] @ columns <= lp.row_upper[~equality])
        m.addConstr(matrix[equality] @ columns == lp.row_upper[equality])
        m.setParam("Method", METHOD_BARRIER)
        m.setParam("Threads", gurobi_threads())
        m.setParam("TimeLimit", time_limit)
        m.setParam("OptimalityTol", optimality)
        m.setParam("FeasibilityTol", feasibility)
        setup_seconds = time.perf_counter() - start

        m.optimize()
        solve_seconds = m.Runtime
    except gp.GurobiError as error:
        return Result(status="error", note=str(error)[:200], **common)

    if m.Status == gp.GRB.TIME_LIMIT:
        return Result(status="time_limit", solve_seconds=solve_seconds,
                      setup_seconds=setup_seconds, **common)
    if m.Status != gp.GRB.OPTIMAL:
        return Result(status="error", note=f"gurobi status {m.Status}", **common)

    return Result(status="optimal", objective=m.ObjVal, lower=m.ObjVal,
                  upper=m.ObjVal, iterations=int(m.IterCount),
                  setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                  note=f"barrier iterations {m.BarIterCount}", **common)
