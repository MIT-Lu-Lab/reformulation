"""The record one solver produces on one instance, and how it reaches disk."""

from __future__ import annotations

import csv
import itertools
import math
import os
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

#: Statuses a backend may report.  "unavailable" means the backend could not
#: run here at all (no GPU, no licence, binary not built) and is distinct from
#: "error", which means it ran and failed.
STATUSES = ("optimal", "time_limit", "iteration_limit", "error", "unavailable")

#: Statuses that stopped a run at a limit instead of at an answer.  What such
#: a run reached is a floor, so a summary over a population containing one is
#: a lower bound and says so; dropping them would read a solver that gives up
#: on the hard instances as the fast one.  "error" and "unavailable" are not
#: measurements at all and are not here.
CENSORED = ("time_limit", "iteration_limit")


@dataclass
class Result:
    """One (backend, instance) measurement, and one row of a benchmark CSV.

    Every column is what the backend reported, so a row says what was asked
    for as well as what came back.  NaN means "not measured", never zero.

    - `objective` is the LP's optimal value in the minimisation form the LP is
      stated in, and is NaN for a backend that stopped short.  `lower` and
      `upper` bracket it: a vertex solve sets all three equal, a first-order
      solve puts `objective` at `upper` and reports its gap.  A backend that
      widens its bracket loosens its own comparisons, since
      `scripts/check_correctness.py` reads it.
    - The tolerance columns are what the backend was given, not what the run
      requested; one is NaN for a backend with no such knob.
    - `items` and `periods` name the shape, `entries` sets the cost; see
      `harness.instances.Instance.entries`.
    - The peak columns are filled only by `run_benchmark.py --memory`.  A row
      carrying one has timings that are not the project's timings, which is
      what keeps the two kinds of run apart; see `harness.memory`.  Only the
      device figure compares across backends, a subprocess backend's host
      figure excluding what this process holds.
    """

    backend: str
    instance: str
    family: str
    items: int
    periods: int
    entries: int
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


def claim_path(path: Path) -> Path:
    """Reserve `path`, or the first "-2", "-3", ... variant free, and return it.

    Created empty and exclusively, so that a caller naming its file after the
    clock gets a file of its own: seconds are all the resolution such a name
    has, and two jobs submitted together start within one.  Sharing a path does
    not merge two runs, it interleaves them into whichever rewrites it last.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for index in itertools.count(1):
        candidate = (path if index == 1 else
                     path.with_name(f"{path.stem}-{index}{path.suffix}"))
        try:
            os.close(os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        return candidate


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
REQUIRED_COLUMNS = ("backend", "instance", "family", "items", "periods", "entries",
                    "status")


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

    For the first-order backends the bracket is the duality gap, which their
    optimality tolerance holds two orders of magnitude inside
    `config.AGREEMENT_TOLERANCE`, so this loosens nothing there; it is here so
    that a backend which declares its uncertainty is judged on what it claims.
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
