"""Solve the joint replenishment LP with cuPDLPx.

cuPDLPx's Python package takes the LP as the NumPy and SciPy arrays
`baselines.model` already builds, so it goes straight from memory into the
solver.

The bindings expose the solver's own `Runtime` and `RescalingTime` but not its
presolve time, so `solve_seconds` is the wall clock of `optimize()`: presolve,
preconditioning, the transfer to the device and the solve.  That is the number
a PRIA backend's `solve_seconds` is comparable to, since neither includes
building the problem; `note` carries cuPDLPx's own `Runtime` so the narrower
figure is still recoverable.
"""

from __future__ import annotations

import math
import os
import time

from baselines import model
from harness import memory
from harness.config import MEMORY_SAMPLE_SECONDS, Tolerances, results_dir
from harness.results import Result

try:
    from cupdlpx import Model
except ImportError as error:        # the package is optional, like a licence
    Model = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None

#: cuPDLPx's termination reason mapped onto `harness.results.STATUSES`.  The
#: reasons left out -- PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED
#: and UNSPECIFIED -- become "error": the LP is feasible and bounded by
#: construction, every column nonnegative with a nonnegative cost and every
#: demand met by the period it falls in, so any of them says something went
#: wrong rather than something about the instance.
TERMINATION = {
    "OPTIMAL": "optimal",
    "FEAS_POLISH_SUCCESS": "optimal",
    "TIME_LIMIT": "time_limit",
    "ITERATION_LIMIT": "iteration_limit",
}

#: Parameters set on every model, beyond the two tolerances and the time limit.
#: `Presolve` is on, which is cuPDLPx's own default and what a reader
#: reproduces from its README -- so its iteration counts are on the presolved
#: LP while PRIA's are on the whole one.  `{"Presolve": False}` is the knob if
#: that comparison has to be exact; it costs iterations and leaves the
#: objective where it was.
SOLVER_PARAMS: dict = {"Presolve": True, "OutputFlag": False}

#: Shape of the throwaway instance solved before anything that is measured.
#: The first `optimize()` of a process carries CUDA context and cuSPARSE
#: set-up on top of its own loop, which would otherwise land entirely on
#: whichever instance sorted first.  It has to be a real LP through
#: `build_model`; a 1-by-1 matrix does not trigger the same set-up.
WARM_UP_SHAPE = (4, 8)

_warmed = False


def _warm_up(tolerances: Tolerances) -> None:
    """Pay CUDA's one-off set-up before anything that is being measured."""
    global _warmed
    if _warmed:
        return
    _warmed = True          # set first: a failure here must not retry every solve
    from harness import instances  # local: only this path needs a generator
    build_model(model.build(instances.random_jrp(0, *WARM_UP_SHAPE)),
                tolerances=tolerances, time_limit=math.inf).optimize()


def export_mps(instance, path=None):
    """Write `instance`'s LP to an MPS file and return the path.

    For staging an instance onto a machine that has cuPDLPx but not this
    harness.
    """
    path = path or results_dir() / "mps" / f"{instance.name}.mps"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.write_mps(model.build(instance), path, name=instance.name)
    return path


def build_model(lp, *, tolerances: Tolerances, time_limit: float) -> Model:
    """A cuPDLPx model for `lp`, a `model.Formulation`.

    The two-sided row bounds go across unchanged: that is the form cuPDLPx
    wants, and it is why `baselines.model` states the LP that way rather than
    turning the demand equalities into pairs of inequalities.  `OptimalityTol`
    is the relative duality gap it stops on, which is also PRIA's criterion and
    is not what the vertex baselines stop on.
    """
    lp_model = Model(objective_vector=lp.cost,
                     constraint_matrix=lp.matrix.tocsr(),
                     constraint_lower_bound=lp.row_lower,
                     constraint_upper_bound=lp.row_upper,
                     variable_lower_bound=lp.column_lower,
                     variable_upper_bound=lp.column_upper)
    lp_model.setParams(OptimalityTol=tolerances.optimality,
                       FeasibilityTol=tolerances.feasibility, **SOLVER_PARAMS)
    if math.isfinite(time_limit):
        lp_model.setParam("TimeLimit", time_limit)
    return lp_model


def solve(instance, *, tolerances: Tolerances, time_limit: float,
          sample_memory: bool = False) -> Result:
    """Solve the LP of `instance` with cuPDLPx, or report why it cannot.

    `setup_seconds` is building the LP and handing it to cuPDLPx, which is this
    harness's overhead and not the solver's; `solve_seconds` is the
    `optimize()` call.  cuPDLPx only looks at the clock every `eval_freq`
    iterations, so a time limit overruns by an iteration block, and in process
    there is no subprocess to kill: a run that overruns is waited out and
    reported as `time_limit` on its own say-so.

    `sample_memory` watches this process while `optimize()` runs and records
    the peak.  It is in process, so the host figure includes the LP this
    harness built and is not the solver's alone; the device figure is, cuPDLPx
    being the only thing here that touches the card.  Sampling costs the solve
    it watches, which is why a run either asks for memory or for timings -- see
    `harness.memory`.
    """
    common = dict(backend="cupdlpx", instance=instance.name, family=instance.family,
                  items=instance.items, periods=instance.periods,
                  entries=instance.entries,
                  optimality_tolerance=tolerances.optimality,
                  feasibility_tolerance=tolerances.feasibility)

    if Model is None:
        return Result(status="unavailable",
                      note=f"cupdlpx not installed ({IMPORT_ERROR}); pip install cupdlpx",
                      **common)

    start = time.perf_counter()
    lp_model = build_model(model.build(instance), tolerances=tolerances,
                           time_limit=time_limit)
    timed = dict(setup_seconds=time.perf_counter() - start, **common)

    _warm_up(tolerances)
    start = time.perf_counter()
    with memory.watch(os.getpid(), MEMORY_SAMPLE_SECONDS,
                      enabled=sample_memory) as sampler:
        lp_model.optimize()
    solve_seconds = time.perf_counter() - start
    timed.update(peak_host_bytes=sampler.peak.host_bytes,
                 peak_device_bytes=sampler.peak.device_bytes)

    reason = lp_model.StatusName
    if reason is None:
        return Result(status="error", solve_seconds=solve_seconds,
                      note="cupdlpx returned no status", **timed)

    note = f"runtime {lp_model.Runtime:.4g}s, rescaling {lp_model.RescalingTime:.4g}s"
    status = TERMINATION.get(reason, "error")
    if status != "optimal":
        # Bounds stay NaN, as they do when the vertex baselines stop early:
        # `objective` means the LP's optimal value and this is not it.  How
        # close it came is the gap, which goes in the note.
        note = f"{reason}, relative gap {lp_model.RelGap:.3g}; " + note
        return Result(status=status, iterations=lp_model.IterCount,
                      solve_seconds=solve_seconds, note=note[:200], **timed)

    # Stopping on a gap rather than at a vertex, cuPDLPx brackets the optimum
    # the way PRIA does: the primal objective is the value of a nearly feasible
    # point, the dual objective that of a nearly feasible dual one.
    return Result(status=status, objective=lp_model.ObjVal, lower=lp_model.DualObj,
                  upper=lp_model.ObjVal, iterations=lp_model.IterCount,
                  solve_seconds=solve_seconds, note=note[:200], **timed)
