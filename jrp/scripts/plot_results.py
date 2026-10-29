#!/usr/bin/env python3
"""Draw the benchmark figures and tables from one or more result CSVs.

    source env.sh
    scripts/plot_results.py results/benchmark-pria-cuda-*.csv results/benchmark-highs-*.csv
    scripts/plot_results.py results/benchmark-*.csv --figures ecdf --max-entries 1000000
    scripts/plot_results.py results/benchmark-*.csv --pria-cuda --cupdlpx --format pdf

Name as many CSVs as you like: the GPU baselines need a GPU box and pria-jax
does not, so a comparison is usually assembled from runs made on different
machines.  Where two files hold the same measurement the later one wins.  The
files come first, because `--families` and `--figures` take lists and would
otherwise swallow them.

Six outputs, and `--figures` picks a subset of them:

    ecdf                fraction of instances solved within a time budget
    cost-per-iteration  seconds per iteration against size, with its table
    iterations          iterations to converge against size
    sgm-time            shifted geometric mean of solve time, by size bucket
    sgm-iterations      the same for iterations to converge
    uncertainty         how much of the agreement budget each backend's own
                        bracket spends, by size bucket
    peak-memory         peak device memory, by size bucket, from a memory run

`peak-memory` is the one that is not in the default set: sampling memory costs
the solve it watches, so `run_benchmark.py --memory` is a run of its own and its
rows are used by that table and by nothing else.  Name its CSV and ask for the
table.

Every figure is drawn over the instances each of its backends attempted, and
names that count; every one has a table twin carrying the same numbers.  The
three size axes are filtered separately -- `items` and `periods` are how an
instance is named, `entries` is what sets the cost -- and `--size-axis` picks
which one the figures plot against and the tables bucket by.

Figures are PNG and tables are Markdown, both of which can be looked at as they
are; `--format pdf` is for the figure that is going into the paper and
`--csv-tables` for the table something else has to read back.  Files land in the
figures directory `harness.config` names, a `figs/` beside the CSVs, unless
`--output` says otherwise.

The drawing itself is `harness.figures`, so a notebook can produce the same
figures without going through this script.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path)
import matplotlib

matplotlib.use("Agg")   # every figure here goes to a file, on machines with no display

from matplotlib import pyplot as plt

from harness import config, figures, registry

#: The figures this script can draw, which is also what `--figures` accepts.
CHOICES = ("ecdf", "cost-per-iteration", "iterations", "sgm-time", "sgm-iterations",
           "uncertainty", "peak-memory")

#: What it draws when not told: everything a timing run can support.
#: `peak-memory` is out because it needs a memory run, which is a run of its own
#: -- sampling costs the solve it watches -- so it is asked for by name, with
#: that run's CSV, rather than failing on every ordinary invocation.
DEFAULT_FIGURES = CHOICES[:-1]


def save(ax, stem: Path, fmt: str) -> Path:
    """Write `ax`'s figure to `stem.<fmt>` and close it."""
    path = stem.with_suffix(f".{fmt}")
    ax.figure.savefig(path, format=fmt, facecolor="white", dpi=200)
    plt.close(ax.figure)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", type=Path,
                        help="benchmark CSVs written by run_benchmark.py")
    registry.add_backend_flags(parser)
    parser.add_argument("--families", nargs="+", default=None,
                        help="restrict to these families, spelled as the CSVs spell them")
    # One --min-<size>/--max-<size> pair per size axis, generated from the one
    # home for what the sizes are, so adding an axis there adds its filters here.
    for name, label in figures.AXES.items():
        parser.add_argument(f"--min-{name}", type=int, default=None,
                            help=f"drop instances with fewer {label} than this")
        parser.add_argument(f"--max-{name}", type=int, default=None,
                            help=f"drop instances with more {label} than this")
    parser.add_argument("--figures", nargs="+", choices=CHOICES,
                        default=list(DEFAULT_FIGURES), metavar="NAME",
                        help=f"which to draw; any of {', '.join(CHOICES)}. The last "
                             f"needs a memory run and is not in the default set")
    parser.add_argument("--seconds", choices=tuple(figures.SECONDS), default="solve",
                        help="which time column to plot; setup is not the solver's "
                             "cost and is not comparable across backends")
    parser.add_argument("--size-axis", choices=tuple(figures.AXES), default="entries",
                        help="the size the figures plot against and the tables bucket by")
    parser.add_argument("--agreement-tolerance", type=float,
                        default=config.AGREEMENT_TOLERANCE,
                        help="relative objective gap two backends may show and still "
                             "be plotted against each other")
    parser.add_argument("--shift-seconds", type=float, default=config.SGM_SHIFT_SECONDS,
                        help="shift of the shifted geometric mean over times")
    parser.add_argument("--min-solve-seconds", type=float,
                        default=config.MEMORY_MIN_SOLVE_SECONDS,
                        help="for the memory table, the shortest solve whose sampled "
                             "peak is believed")
    parser.add_argument("--shift-iterations", type=float,
                        default=config.SGM_SHIFT_ITERATIONS,
                        help="shift of the shifted geometric mean over iterations")
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"),
                        help="figure file format; pdf is the one to ask for when the "
                             "figure is going into the paper")
    parser.add_argument("--csv-tables", action="store_true",
                        help="write each table as CSV as well as Markdown")
    parser.add_argument("--output", type=Path, default=None,
                        help="directory for the figures and tables; the default is "
                             "the one harness/config.py names")
    args = parser.parse_args(argv)

    strays = [name for name in (args.families or []) if name.endswith(".csv")]
    if strays:
        parser.error(f"--families swallowed {', '.join(strays)}; name the CSVs first")

    missing = [str(path) for path in args.csv if not path.exists()]
    if missing:
        parser.error(f"no such file: {', '.join(missing)}")

    try:
        return draw(args)
    except ValueError as error:
        # A file that is not a benchmark CSV, or a selection with nothing left in
        # it.  `harness.figures` raises rather than exiting, so that a notebook
        # sees an exception; deciding it is an exit status is this layer's job.
        print(error, file=sys.stderr)
        return 2


def draw(args) -> int:
    """Everything after the arguments are settled."""
    lower = {name: getattr(args, f"min_{name}") for name in figures.AXES
             if getattr(args, f"min_{name}") is not None}
    upper = {name: getattr(args, f"max_{name}") for name in figures.AXES
             if getattr(args, f"max_{name}") is not None}
    data = figures.prepare(args.csv, backends=args.backends, families=args.families,
                           lower=lower, upper=upper,
                           tolerance=args.agreement_tolerance)
    for note in data.notes:
        print(note)
    print()

    out = args.output or config.figures_dir()
    out.mkdir(parents=True, exist_ok=True)
    axis, seconds, written = args.size_axis, args.seconds, []
    tables = []

    if "ecdf" in args.figures:
        written.append(save(figures.ecdf(data, seconds=seconds),
                            out / f"ecdf-{seconds}-time", args.format))
    if "cost-per-iteration" in args.figures:
        written.append(save(figures.cost_per_iteration(data, axis=axis),
                            out / f"cost-per-iteration-by-{axis}", args.format))
        tables.append((figures.cost_per_iteration_table(data, axis=axis),
                       out / f"cost-per-iteration-by-{axis}"))
    if "iterations" in args.figures:
        written.append(save(figures.iterations(data, axis=axis),
                            out / f"iterations-by-{axis}", args.format))
    if "sgm-time" in args.figures:
        tables.append((figures.solve_time_table(data, seconds=seconds, axis=axis,
                                                shift=args.shift_seconds),
                       out / f"sgm-{seconds}-time-by-{axis}"))
    if "sgm-iterations" in args.figures:
        tables.append((figures.iterations_table(data, axis=axis,
                                                shift=args.shift_iterations),
                       out / f"sgm-iterations-by-{axis}"))
    if "uncertainty" in args.figures:
        tables.append((figures.uncertainty_table(
            data, axis=axis, tolerance=args.agreement_tolerance),
            out / f"uncertainty-by-{axis}"))

    if "peak-memory" in args.figures:
        tables.append((figures.peak_memory_table(
            data, axis=axis, min_solve_seconds=args.min_solve_seconds),
            out / f"peak-memory-by-{axis}"))

    for table, stem in tables:
        print(table.markdown())
        written += table.write(stem, comma_separated=args.csv_tables)

    print("wrote")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
