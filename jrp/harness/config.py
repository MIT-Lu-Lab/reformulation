"""Paths and shared defaults.

Paths come from the environment and are *not* duplicated here: ``env.sh`` is
their single home, and this module fails loudly rather than guessing.  Solver
defaults that several scripts need (tolerances, seed, time limit) live here, and
the scripts' argparse definitions read them from here instead of restating the
values.
"""

import os
from dataclasses import dataclass
from pathlib import Path

_HINT = "environment not set: run `source env.sh` from the repository root"


def _dir(name: str) -> Path:
    """Return the directory named by environment variable `name`."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{_HINT} ({name} is unset)")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir() -> Path:
    """Where `scripts/run_benchmark.py` writes CSV."""
    return _dir("PRIA_RESULTS_DIR")


def figures_dir() -> Path:
    """Where `scripts/plot_results.py` writes figures and tables.

    A `figs/` inside the results directory, so a figure moves and is cleared
    with the CSV it was drawn from.  `--output` overrides it for a one-off.
    """
    path = results_dir() / "figs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bin_dir() -> Path:
    """Where `scripts/build_cuda.sh` installs compiled binaries."""
    return _dir("PRIA_BIN_DIR")


def repo_root() -> Path:
    return _dir("PRIA_ROOT")


def pria_cuda_bin() -> Path | None:
    """The pria-cuda file CLI, or None if it was never built."""
    path = bin_dir() / "pria_solve_file"
    return path if path.exists() else None


def pria_cuda_lib() -> Path | None:
    """The pria-cuda shared library, or None if it was never built."""
    path = bin_dir() / "libpria.so"
    return path if path.exists() else None


# --- shared solver defaults ------------------------------------------------
# One home each.  Scripts pass these as argparse defaults; help text describes
# them relationally so no number is restated anywhere else.


@dataclass(frozen=True)
class Tolerances:
    """What a run asks its solvers to stop on.

    Two numbers, because every solver here has a separate knob for each and
    they answer different questions.  Asking for a tight gap while leaving
    feasibility loose lets a solver stop at a point whose objective is not the
    objective of anything feasible.

    PRIA takes only `optimality`.  Its iterates are feasible by construction,
    so it has no feasibility knob and its records leave that column NaN.
    """

    optimality: float = 1e-6
    """Relative duality gap a first-order solver may stop at, and the
    reduced-cost tolerance given to the vertex baselines.

    The tightest gap at which every first-order backend here is still
    measuring its own convergence and not its own rounding."""

    feasibility: float = 1e-9
    """Relative primal feasibility a point must reach to count as a solution.

    Pinned rather than matched to `optimality`, so that a first-order baseline
    is not given an easier problem than a vertex one."""


TOLERANCES = Tolerances()
"""The defaults both scripts start from.

No backend reads this: each is handed the tolerances its run resolved, and
records what it was given."""

TIME_LIMIT = 300.0
"""Per-instance wall-clock budget, in seconds."""

SEED = 0
"""RNG seed for the random instances used by the correctness check."""

CHECK_SHAPES = ((3, 6), (5, 10), (8, 16))
"""(items, periods) the correctness check uses without --shapes."""

CHECK_TRIALS = 2
"""Random instances per shape in the correctness check."""

CHECK_MAX_ENTRIES = 100_000
"""Largest instance the correctness check will touch, in x variables.

Instances above it are in the collection to be timed, not checked: a
disagreement shows at the smallest shape as plainly as at the largest, and a
test slow enough to skip is a test nobody runs."""

CHECK_MAX_INSTANCES = 24
"""How many instances the correctness check will run.

`CHECK_MAX_ENTRIES` caps how big one may be; this caps how many there are,
since a bucket holds hundreds.  The selection is spread evenly over the
instances sorted by size, so it samples every shape rather than keeping the
smallest."""

SGM_SHIFT_SECONDS = 1e-2
"""Shift of the shifted geometric mean the summary tables report, in seconds.

A geometric mean of times is the statistic a ratio of two of them is a ratio
of; the shift makes it insensitive to differences well below itself, so one
very fast instance cannot decide a comparison.  This one sits at the floor
below which a solve is CUDA launch and kernel dispatch rather than
convergence -- far under the literature's ten seconds, which is sized for
instances that take minutes.  The tables print the shift they used."""

SGM_SHIFT_ITERATIONS = 10.0
"""Shift of the same mean over iteration counts.

The literature's, and small next to what these solvers take, so it changes no
comparison here; it keeps a zero-iteration instance from taking the mean to
zero."""

MEMORY_SAMPLE_SECONDS = 0.005
"""How often a memory run reads the footprint of the process it watches.

Fast enough to give a short solve the readings `MEMORY_MIN_SAMPLES` asks for,
and cheap enough next to a solve.  It can afford to be fast because a memory
run's timings are not used for anything; see `harness.memory`."""

MEMORY_MIN_SAMPLES = 40
"""How many readings a solve must span before its peak is worth believing.

A sampler misses any peak shorter than its interval, so a solve ending in
fewer readings has a peak that is mostly luck."""

MEMORY_MIN_SOLVE_SECONDS = MEMORY_SAMPLE_SECONDS * MEMORY_MIN_SAMPLES
"""The shortest solve whose sampled peak the memory table will plot.

Derived, so changing the sampling rate moves the floor with it."""

AGREEMENT_TOLERANCE = 1e-4
"""Relative objective difference two implementations may show and still agree.

Looser than `Tolerances.optimality`, which is how far a first-order solver's
objective can sit from a vertex value; a check tighter than the tolerance it
asked for would be testing the tolerance."""


def gurobi_threads() -> int:
    """How many threads the Gurobi backend asks for; 0 is Gurobi's own choice.

    Gurobi sizes its thread pool from the machine rather than from the job's
    cgroup, so under a scheduler it oversubscribes the allocation and the time
    it reports is not the time the allocation bought.  `GUROBI_THREADS` in the
    environment pins it -- `env.sh` exports it when it is set and says nothing
    when it is not, so that what "unset" means is decided here and once.
    """
    return int(os.environ.get("GUROBI_THREADS") or 0)
