"""The record one solver produces on one instance, and how it reaches disk."""

from __future__ import annotations

import csv
import math
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

#: Statuses a backend may report.  "unavailable" means the backend could not
#: run here at all (no GPU, no licence, binary not built) and is distinct from
#: "error", which means it ran and failed.
STATUSES = ("optimal", "time_limit", "iteration_limit", "error", "unavailable")

#: Statuses that stopped a run at a limit rather than at an answer.  What such
#: a run reached is a floor on what it would have taken, so a summary over a
#: population containing one is a lower bound and says so, rather than being
#: taken over the finishers alone -- which would read a solver that gives up on
#: the hard instances as the fast one.  "error" and "unavailable" are not here:
#: neither is a measurement at all.
CENSORED = ("time_limit", "iteration_limit")


@dataclass
class Result:
    """One (backend, instance) measurement.

    `objective` is the LP relaxation's optimal value in *minimisation* form, so
    it is directly comparable across backends.

    `lower`/`upper` bracket the true value, and where `objective` sits inside
    that bracket depends on how the backend arrived at it.  A vertex solve
    (HiGHS, Gurobi) sets all three equal, having no uncertainty to declare.  A
    first-order solve reports its duality gap, with `objective` at `upper`,
    since the primal bound is the value of a point it actually holds.  The
    max-flow baseline is exact but for rounded capacities, so it centres
    `objective` in a bracket of that rounding slack.

    A backend that stopped short always leaves `objective` NaN, the column
    meaning the relaxation's optimal value and that not being what it has; it
    may still record a bracket if its bounds are a true one anyway, which
    praq-cuda's are and cuPDLPx's are not.  `scripts/check_correctness.py`
    compares none of those, but it does read the bracket of the ones it
    compares, so a backend that widens it loosens its own comparisons.

    The two tolerance columns record what the backend was actually asked for, not
    what the run requested, so a backend with no such knob leaves one NaN: PRAQ
    has no feasibility tolerance, its iterates being feasible by construction.

    The two peak columns are filled only by a memory run -- `run_benchmark.py
    --memory`, see `harness.memory` -- and are NaN otherwise.  That is what keeps
    the two kinds of run apart: sampling costs the solve it samples, so a row
    carrying a peak is a row whose timings are not the project's timings, and
    `harness.figures` uses it for the memory table and for nothing else.  The
    host figure is not comparable across backends the way the device figure is:
    praq-cuda's is a subprocess that read the instance from a file, while an
    in-process backend's includes whatever the harness itself is holding.
    """

    backend: str
    instance: str
    family: str
    n: int
    edges: int
    status: str
    objective: float = math.nan
    lower: float = math.nan
    upper: float = math.nan
    iterations: int = -1
    setup_seconds: float = math.nan
    solve_seconds: float = math.nan
    optimality_tolerance: float = math.nan
    feasibility_tolerance: float = math.nan
    peak_host_bytes: float = math.nan
    peak_device_bytes: float = math.nan
    note: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}, expected one of {STATUSES}")


def write_csv(path: Path, results: list[Result]) -> Path:
    """Write `results` to `path`, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f.name for f in fields(Result)]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return path


#: How a `Result` field's text is turned back into a value.  `from __future__
#: import annotations` leaves every field's `type` a string, so the lookup is by
#: spelling; a field of a fourth type would raise here rather than silently
#: arrive as text.
_PARSE = {"str": str, "int": int, "float": float}

#: The columns a file must have to be a benchmark CSV at all: what was measured,
#: on what, and whether it worked.  Every other column falls back to its field's
#: default when the file does not have it, which is what lets a column be added
#: without invalidating the runs already on disk -- an old CSV is still readable
#: and simply has no opinion about the new column.  Unknown columns are ignored,
#: so a newer file also reads here.
REQUIRED_COLUMNS = ("backend", "instance", "family", "n", "edges", "status")


def read_csv(path: Path) -> list[Result]:
    """Read back what `write_csv` wrote.

    The inverse of `write_csv` and the only reader, so the plotting code works
    on typed records instead of strings: a column whose type changes fails here,
    at the read, rather than as a wrong axis three figures later.  A file without
    `REQUIRED_COLUMNS` is rejected by name, which is what distinguishes a
    benchmark CSV from the other files that collect in the results directory.

    An empty cell is the default rather than a parse error, so a CSV somebody
    has edited by hand still reads.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        present = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in present]
        if missing:
            raise ValueError(f"{path} is not a benchmark CSV: no {', '.join(missing)} column")
        wanted = [f for f in fields(Result) if f.name in present]
        out = []
        for row in reader:
            values = {}
            for field in wanted:
                text = row[field.name]
                if text == "" and field.default is not MISSING:
                    continue
                values[field.name] = _PARSE[field.type](text)
            out.append(Result(**values))
        return out


def uncertainty(result: Result) -> float:
    """How far a backend says its own objective may sit from the true value.

    The furthest point of the bracket it reports from the value it reported, so
    zero for a backend that returns a vertex and `lower == upper == objective`.

    `--maxflow` is exact, but for an instance whose capacities were rounded to
    integers rather than for this one, and it reports that rounding slack as a
    bracket; a comparison against it has to allow for the slack rather than
    pretend it is not there.  For the first-order backends the bracket is the
    duality gap, which their optimality tolerance already holds two orders of
    magnitude inside `config.AGREEMENT_TOLERANCE`, so this loosens nothing there.
    """
    if not (math.isfinite(result.lower) and math.isfinite(result.upper)):
        return 0.0
    return max(result.objective - result.lower, result.upper - result.objective, 0.0)


def disagreement(base: Result, other: Result) -> float:
    """By how much, relatively, two records differ beyond their own brackets.

    Zero when the brackets overlap, so a backend that admits its uncertainty is
    judged on what it claims rather than on where inside the claim it landed.
    The `1 +` fixes a unit: the relative difference of two objectives is not
    scale-invariant, and an instance whose optimum is near zero would otherwise
    fail any tolerance at all.

    One home for the comparison, because two places need it to mean the same
    thing: `scripts/check_correctness.py` decides a run passes with it, and
    `harness.figures` decides whether a timing is worth plotting.  Both compare
    against `config.AGREEMENT_TOLERANCE`.
    """
    excess = abs(other.objective - base.objective) - uncertainty(base) - uncertainty(other)
    return max(0.0, excess) / (1.0 + abs(base.objective))
