"""Solve the standard linearization with Gurobi's LP solver."""

from __future__ import annotations

import time

from baselines import model
from harness.config import Tolerances
from harness.results import Result

#: Gurobi LP method: 1 = dual simplex, so the returned primal is a vertex and
#: the relaxation's half-integrality applies to it.  Barrier would not give that.
METHOD_DUAL_SIMPLEX = 1

#: Gurobi refuses either tolerance below this.  A request tighter than its own
#: floor is clamped rather than raising, so that one set of tolerances can be
#: handed to every backend and each takes what it can.
TOLERANCE_FLOOR = 1e-9


def solve(instance, *, tolerances: Tolerances, time_limit: float) -> Result:
    """Solve the relaxation of `instance` with Gurobi.

    A *vertex* solve, so it stops on optimality rather than at a gap, but both
    tolerances are requested: OptimalityTol is the reduced-cost test and
    FeasibilityTol is constraint satisfaction, which is the same split
    `Tolerances` makes.  They are recorded too, so timings against a first-order
    backend can be read with them in mind.
    """
    optimality = max(tolerances.optimality, TOLERANCE_FLOOR)
    feasibility = max(tolerances.feasibility, TOLERANCE_FLOOR)
    common = dict(backend="gurobi", instance=instance.name, family=instance.family,
                  n=instance.n, edges=instance.edges,
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
        x = m.addMVar(lp.columns, lb=0.0, ub=1.0, obj=lp.cost)
        m.addConstr(lp.matrix.tocsr() @ x <= lp.rhs)
        m.setParam("Method", METHOD_DUAL_SIMPLEX)
        m.setParam("TimeLimit", time_limit)
        m.setParam("OptimalityTol", optimality)
        m.setParam("FeasibilityTol", feasibility)
        setup_seconds = time.perf_counter() - start

        start = time.perf_counter()
        m.optimize()
        solve_seconds = time.perf_counter() - start
    except gp.GurobiError as error:
        return Result(status="error", note=str(error)[:200], **common)

    if m.Status == gp.GRB.TIME_LIMIT:
        return Result(status="time_limit", solve_seconds=solve_seconds,
                      setup_seconds=setup_seconds, **common)
    if m.Status != gp.GRB.OPTIMAL:
        return Result(status="error", note=f"gurobi status {m.Status}", **common)

    return Result(status="optimal", objective=m.ObjVal, lower=m.ObjVal,
                  upper=m.ObjVal, iterations=int(m.IterCount),
                  setup_seconds=setup_seconds, solve_seconds=solve_seconds, **common)
