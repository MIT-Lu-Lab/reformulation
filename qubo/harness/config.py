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
    return _dir("PRAQ_RESULTS_DIR")


def figures_dir() -> Path:
    """Where `scripts/plot_results.py` writes figures and tables.

    A `figs/` inside the results directory rather than an environment variable
    of its own: a figure belongs with the CSV it was drawn from, so moving the
    results somewhere larger takes the figures along, and one `rm -r` clears
    both.  `--output` overrides it for a one-off.
    """
    path = results_dir() / "figs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bin_dir() -> Path:
    """Where `scripts/build_cuda.sh` installs compiled binaries."""
    return _dir("PRAQ_BIN_DIR")


def repo_root() -> Path:
    return _dir("PRAQ_ROOT")


def praq_cuda_bin() -> Path | None:
    """The praq-cuda file CLI, or None if it was never built."""
    path = bin_dir() / "praq_solve_file"
    return path if path.exists() else None


def praq_cuda_lib() -> Path | None:
    """The praq-cuda shared library, or None if it was never built."""
    path = bin_dir() / "libpraq.so"
    return path if path.exists() else None


# --- shared solver defaults ------------------------------------------------
# One home each.  Scripts pass these as argparse defaults; help text describes
# them relationally so no number is restated anywhere else.


@dataclass(frozen=True)
class Tolerances:
    """What a run asks its solvers to stop on.

    Two numbers and not one, because they answer two different questions and
    every solver here has a separate knob for each: cuPDLPx's `--eps_opt` and
    `--eps_feas`, Gurobi's OptimalityTol and FeasibilityTol, HiGHS's dual and
    primal feasibility tolerances.  Collapsing them is the trap -- ask for a
    tight gap while feasibility stays loose and a solver may stop at a point
    whose objective is not the objective of anything feasible, which makes
    comparing it with another solver's meaningless.

    PRAQ takes only the first.  Its iterates are feasible by construction, every
    step projecting onto [0, 1]^n and onto the dual set, so it has no
    feasibility knob to set and its records leave that column NaN.
    """

    optimality: float = 1e-8
    """Relative duality gap at which a first-order solver may stop, and the
    reduced-cost tolerance -- dual feasibility -- given to the simplex baselines."""

    feasibility: float = 1e-9
    """Relative primal feasibility a point must reach to count as a solution."""


TOLERANCES = Tolerances()
"""The defaults both scripts start from.

`--optimality-tolerance` and `--feasibility-tolerance` override them for a run.
No backend reads this: each is handed the tolerances the run resolved, and
records what it was given.
"""

TIME_LIMIT = 300.0
"""Per-instance wall-clock budget, in seconds.

Sized so that one solver's pass over `instances.HARD_BUCKETS` is bounded at
about an hour, and so that the first-order solvers finish the top rung well
inside it.  `--time-limit` overrides it for a run that wants to see what a
simplex baseline does with the large rungs.
"""

SEED = 0
"""RNG seed for the random instances used by the correctness check."""

CHECK_SIZES = (25, 50, 100)
"""Sizes n used by the correctness check when --sizes is not given."""

CHECK_DENSITY = 1.0
"""Off-diagonal density of the random correctness instances."""

CHECK_TRIALS = 3
"""Random instances per size in the correctness check."""

CHECK_MAX_EDGES = 15_000
"""Largest instance the correctness check will touch.

A test slow enough to skip is a test nobody runs, and this cap is what keeps
the pass well under a minute.  Nothing above it says anything new about
whether two implementations agree -- the small instances disagree just as
visibly -- and the large ones are in the collection to be timed by
`run_benchmark.py`.
"""

CHECK_MAX_INSTANCES = 24
"""How many instances the correctness check will run.

`CHECK_MAX_EDGES` caps how big one instance may be; this caps how many there
are, without which a bucket of `instances.HARD_BUCKETS` could not be checked
at all -- each holds hundreds of instances per size.

The selection is spread evenly over the instances sorted by size, so the cap
samples every size and family rather than keeping the smallest.  The hundredth
instance of one size says nothing new about whether two implementations agree,
the same argument `CHECK_MAX_EDGES` makes about size.  `--limit 0` lifts it
for a run that wants the whole collection.
"""

SGM_SHIFT_SECONDS = 1e-3
"""Shift, in seconds, of the shifted geometric mean the summary tables report.

A geometric mean of times is the standard summary because it is the statistic a
ratio of two of them is a ratio of; the shift is there so that one very fast
instance cannot dominate that ratio, `exp(mean(log(t + s))) - s` being
insensitive to differences well below `s`.  The literature's shift is ten
seconds, sized for instances that take minutes; this one sits at the floor
below which a solve here is process launch and kernel dispatch rather than
convergence.

The tables print the shift they used, so a reader never has to assume it, and
`--shift-seconds` overrides it for a run that wants to see how much of a gap
survives a harsher one.
"""

SGM_SHIFT_ITERATIONS = 10.0
"""Shift, in iterations, for the same mean over iteration counts.

The count is what the first-order literature reports a shifted geometric mean of
at this shift, and it is small next to the hundreds-to-thousands these solvers
take, so it changes no comparison here; it is what keeps a hypothetical
zero-iteration instance from taking the mean to zero.
"""

MEMORY_SAMPLE_SECONDS = 0.005
"""How often a memory run reads the footprint of the process it is watching.

Fast enough that a solve of a fifth of a second is sampled tens of times, which
is what `MEMORY_MIN_SAMPLES` asks for, and slow enough that the reads -- one
`/proc` line and one NVML call -- stay a rounding error next to a solve.  It can
afford to be fast because a memory run's timings are not used for anything: see
`harness.memory` for why the two kinds of run are separate.
"""

MEMORY_MIN_SAMPLES = 40
"""How many readings a solve must span before its peak is worth believing.

A sampler misses any peak shorter than its interval, so a solve that ends in
fewer readings than this has a peak that is mostly luck.  Forty is enough that
the largest reading is near the true maximum for an allocation pattern that
holds its high-water mark for any appreciable part of the solve, which is what a
solver that allocates its working set up front does.
"""

MEMORY_MIN_SOLVE_SECONDS = MEMORY_SAMPLE_SECONDS * MEMORY_MIN_SAMPLES
"""The shortest solve whose sampled peak the memory table will plot.

Derived rather than chosen, so that changing the sampling rate moves the floor
with it instead of leaving a stale number behind.  `--min-solve-seconds`
overrides it for a run that wants to see the fast instances anyway.
"""

AGREEMENT_TOLERANCE = 1e-6
"""Relative objective difference two implementations may show and still agree.

Loose next to `Tolerances.optimality` on purpose: the first-order solvers stop
on a relative duality gap, so their reported objective can sit that far from the
vertex value a simplex baseline returns.
"""
