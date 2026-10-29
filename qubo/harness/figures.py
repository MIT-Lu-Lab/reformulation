"""The figures and tables the benchmark CSVs exist to produce.

`scripts/plot_results.py` is the command line over this module.  The drawing is
here and not there so a notebook can import it, which is also why nothing below
reads argv, prints, or decides where a file goes: `prepare` returns the rows to
plot together with the prose saying what it dropped, and the caller decides what
to do with both.

Three properties of the input decide most of what follows.

`status` is not always `optimal`.  `unavailable` means the backend could not run
on this machine at all -- no GPU, no licence, binary not built -- and is not a
population member; dropping it and dropping a timeout are different acts.  A
`time_limit` row *is* a member: a censored observation of a real attempt, which
the ECDF shows as a curve that stops rising and the tables leave out of their
means, since a run that did not converge has no time-to-converge to average.

Timings compare only across backends that ran the same instances, and these
runs are made on different machines -- the GPU baselines need a GPU box, praq-jax
does not -- so every figure draws over the instances *all* of its backends
attempted and says how many that is.  Otherwise a backend that skipped the hard
half of the collection plots as the fast one.

`iterations` means one thing for a first-order method and another for dual
simplex, whose steps are a different object and hundreds of times more expensive
each.  Anything that divides by it or plots it draws only
`registry.FIRST_ORDER`, and names the ones it left out rather than inviting the
comparison.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, PercentFormatter

from harness import config, registry
from harness.results import CENSORED, Result, disagreement, read_csv, uncertainty


@dataclass(frozen=True)
class Look:
    """How one backend is drawn, in every figure it appears in."""

    colour: str
    dashes: str | tuple
    marker: str


#: Colour, dash pattern and marker per backend.  Fixed per backend rather than
#: per curve, so that a backend is the same in every figure and adding one to a
#: plot does not repaint the others.
#:
#: The six colours are slots of a categorical palette checked for colour-vision
#: separation on a white surface with every pair in play, not just neighbouring
#: ones, because any subset of backends can be plotted together: the worst pair
#: is magenta against aqua at deutan dE 6.1 (OKLab x100), which is inside the
#: band that a second, non-colour channel makes legible -- hence a distinct dash
#: pattern and marker each, which also survive a greyscale print.  The three
#: first-order backends, the only ones some figures draw, are the strongest
#: triple of the six: worst pair dE 13.0.  Aqua, yellow and magenta fall under
#: 3:1 contrast against white, so every figure has a table twin carrying the
#: same numbers.
LOOK = {
    "praq-cuda": Look("#2a78d6", "-", "o"),                        # blue
    "praq-jax": Look("#4a3aa7", (0, (5, 1.5)), "s"),               # violet
    "cupdlpx": Look("#1baf7a", (0, (1, 1.3)), "^"),                # aqua
    "maxflow": Look("#008300", (0, (6, 1.5, 1, 1.5)), "D"),        # green
    "highs": Look("#eda100", (0, (3, 1.2, 1, 1.2, 1, 1.2)), "v"),  # yellow
    "gurobi": Look("#e87ba4", (0, (2, 1.2)), "P"),                 # magenta
}

_unstyled = set(registry.BACKENDS) - set(LOOK)
if _unstyled:   # a new backend needs a colour of its own before it can be drawn
    raise RuntimeError(f"harness.figures.LOOK has no entry for {sorted(_unstyled)}")

#: Which time column each `seconds` choice reads, and what to call it on an axis.
#: They are three different measurements and the axis has to say which one it is
#: showing.  `setup` is not the solver's cost and is not comparable across
#: backends: for praq it is compilation and the upload, for cuPDLPx it is this
#: harness exporting an MPS file, which is our overhead and not the solver's.
SECONDS = {
    "solve": "solve time (s)",
    "setup": "setup time (s)",
    "total": "setup + solve time (s)",
}

#: Which backend's objective the others are checked against, most trustworthy
#: first.  The max-flow oracle is exact up to the rounding slack it declares and
#: stops early for no reason at all, so it is the reference wherever it ran; then
#: the two simplex baselines, which return a vertex; then the first-order
#: solvers, which stop on a gap.  The object of study is last on purpose: a
#: reference should not be the thing being measured.
REFERENCE_ORDER = ("maxflow", "highs", "gurobi", "cupdlpx", "praq-cuda", "praq-jax")

# --- chrome ----------------------------------------------------------------
# Hairline grid and axes one shade off the surface, text in ink rather than in
# a series colour, so the marks are the only saturated thing in the frame.

GRID = "#e6e5e1"
INK = "#0b0b0b"
MUTED = "#52514e"
FIGSIZE = (6.4, 4.0)


def axes(ax=None, *, figsize: tuple[float, float] = FIGSIZE):
    """A styled Axes: `ax` if given, otherwise a new figure's."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    return ax


def _finish(ax, *, title: str, subtitle: str, xlabel: str, ylabel: str,
            legend: str) -> None:
    """Title, a smaller line of provenance under it, axis labels, and a legend.

    The legend appears only where there is something to tell apart: a lone curve
    needs no box, and the line under the title names it instead.  That line is
    where the population size lives, because a fraction or a mean is unreadable
    without its denominator and a caption is not guaranteed to carry it.
    """
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 1:
        subtitle = f"{labels[0]}, {subtitle}"
    elif len(handles) > 1:
        ax.legend(loc=legend, frameon=False, fontsize=9, labelcolor=MUTED,
                  handlelength=2.6)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=20)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=MUTED,
            fontsize=9, va="bottom", ha="left")
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)


#: Decades above which a log axis is labelled once per decade, and below which
#: it is labelled at 1, 2 and 5 within each.  Both rules exist because the same
#: figure can span either: two backends over the whole collection put five
#: decades on the time axis, where decade labels are plenty, and two rungs of
#: the ladder put one, where they would leave two labels on the whole axis.
_DENSE_DECADES = 4.0


def _log_axis(axis, *, compact: bool) -> None:
    """Label a log axis, once its limits are known -- so call this last.

    Matplotlib labels the minor ticks of a log axis that spans only a decade or
    two, which puts "2 x 10^3" in a smaller font across half the axis;
    `NullFormatter` on the minors is what stops that, and `_DENSE_DECADES`
    decides how many majors replace them.  `compact` spells a label 500, 10k,
    2M rather than as a power of ten.
    """
    low, high = axis.get_view_interval()
    if low > 0 and math.log10(high / low) < _DENSE_DECADES:
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axis.set_minor_formatter(NullFormatter())
    if compact:
        axis.set_major_formatter(FuncFormatter(lambda v, _: _si(v)))


def _si(value: float) -> str:
    """1000 as "1k", 2e6 as "2M": axis ticks and bucket labels a reader can skim."""
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


# --- getting the rows ------------------------------------------------------


def load(paths) -> list[Result]:
    """Every row of every CSV in `paths`, in the order given."""
    return [record for path in paths for record in read_csv(Path(path))]


def profiled(record: Result) -> bool:
    """Whether this row came from a memory run rather than a timing run.

    It carries a peak, which only `run_benchmark.py --memory` fills in, and
    sampling costs the solve it watches -- so this is also the row's own
    statement that its timings are not the project's timings.  Every figure but
    the memory table asks for rows where this is false, which is how the two
    kinds of run stay apart without anyone having to keep their files sorted.
    """
    return math.isfinite(record.peak_host_bytes) or math.isfinite(record.peak_device_bytes)


def deduplicate(records) -> tuple[list[Result], int]:
    """One row per (backend, instance, kind of run), the last occurrence winning.

    Two files can hold the same measurement -- a run passed alongside the
    re-run that replaces it, or two overlapping passes of one backend -- and
    counting it twice would bend an ECDF towards whichever backend was run
    twice.  The last wins so that naming the newer file second replaces the
    older, and the count of what that dropped comes back with it rather than
    happening quietly.

    A memory run's row for an instance does not displace the timing run's: they
    measure different things, and each is wanted by a different figure.
    """
    kept: dict[tuple[str, str, bool], Result] = {}
    for record in records:
        kept[(record.backend, record.instance, profiled(record))] = record
    return list(kept.values()), len(records) - len(kept)


def select(records, *, backends=None, families=None, min_n=None, max_n=None,
           min_edges=None, max_edges=None) -> list[Result]:
    """The rows matching every restriction given; None means no restriction.

    `n` and `edges` are separate ranges because they measure different things.
    `n` is what the QUBO literature reports and what a reader recognises a
    family by; `edges` is what sets the cost, the standard linearization having
    `n + m` columns and about `3m` rows, so a dense n = 250 instance is a bigger
    LP than a sparse n = 20000 one.  Filtering on one is not a proxy for the
    other, which is why both are here.
    """
    def keep(r: Result) -> bool:
        return ((backends is None or r.backend in backends)
                and (families is None or r.family in families)
                and (min_n is None or r.n >= min_n)
                and (max_n is None or r.n <= max_n)
                and (min_edges is None or r.edges >= min_edges)
                and (max_edges is None or r.edges <= max_edges))

    return [r for r in records if keep(r)]


def cross_check(records, tolerance: float = config.AGREEMENT_TOLERANCE):
    """Instances whose backends do not agree on the objective, and who differed.

    Returns `(reference, suspect)` where `suspect` maps an instance name to the
    backends that disagreed with the reference on it, each with its relative
    difference.  A timing is worth plotting only once the answer behind it is
    the right answer: a solver that is quick because it stopped somewhere else
    has to show up rather than be averaged in.  The test is
    `results.disagreement` against `config.AGREEMENT_TOLERANCE`, the same one
    that decides whether `scripts/check_correctness.py` passes, so a comparison
    the check calls clean cannot be called dirty here.
    """
    solved: dict[str, dict[str, Result]] = {}
    for record in records:
        if record.status == "optimal" and math.isfinite(record.objective):
            solved.setdefault(record.instance, {})[record.backend] = record

    present = {r.backend for r in records}
    reference = next((b for b in REFERENCE_ORDER if b in present), None)
    suspect: dict[str, dict[str, float]] = {}
    for name, byname in solved.items():
        base = byname.get(reference)
        if base is None:
            continue
        for backend, other in byname.items():
            delta = disagreement(base, other)
            if delta > tolerance:
                suspect.setdefault(name, {})[backend] = delta
    return reference, suspect


@dataclass(frozen=True)
class Prepared:
    """Rows ready to plot, and the prose describing how they got that way.

    `notes` is not decoration: it says which rows were dropped and why, and a
    figure drawn from these rows is only as honest as that text is visible.  The
    population is deliberately *not* fixed here -- each figure intersects over
    the backends it actually draws, because a figure restricted to two of them
    should not inherit the smaller population of six.
    """

    records: tuple[Result, ...]
    backends: tuple[str, ...]
    notes: tuple[str, ...]
    reference: str | None

    def population(self, backends=None, *, sampled: bool = False, keep=None
                   ) -> tuple[list[Result], list[str]]:
        """The rows of `backends`, over the instances every one of them attempted.

        An `unavailable` row is not an attempt, and is already gone by here.
        The returned names are the denominator of any fraction drawn from the
        rows, and the caller states it.

        `sampled` picks which kind of run to draw from -- timing rows by default,
        a memory run's rows for the memory table.  `keep` drops rows before the
        intersection rather than after, which matters: a row dropped for one
        backend has to take the instance with it, or the two columns end up over
        different instances and stop being a comparison.
        """
        chosen = [b for b in (backends or self.backends) if b in self.backends]
        rows = [r for r in self.records
                if r.backend in chosen and profiled(r) == sampled
                and (keep is None or keep(r))]
        seen: dict[str, set[str]] = {}
        for record in rows:
            seen.setdefault(record.instance, set()).add(record.backend)
        names = sorted(name for name, got in seen.items() if len(got) == len(chosen))
        return [r for r in rows if r.instance in names], names


def prepare(paths, *, backends=None, families=None, min_n=None, max_n=None,
            min_edges=None, max_edges=None,
            tolerance: float = config.AGREEMENT_TOLERANCE) -> Prepared:
    """Read the CSVs and hand back what may be plotted, with the reasons.

    In order: read, drop duplicate measurements, apply the restrictions, drop
    the backends that could not run here, and drop every instance whose
    objectives do not agree -- all of its rows, not just the offender's, because
    a comparison one backend got wrong is not a comparison the others can be
    ranked on.
    """
    paths = [Path(p) for p in paths]
    rows = load(paths)
    rows, superseded = deduplicate(rows)
    notes = [f"{len(rows) + superseded} rows from {len(paths)} file(s)"]
    if superseded:
        notes.append(f"{superseded} superseded by a later file's row for the "
                     f"same backend and instance")

    rows = select(rows, backends=backends, families=families, min_n=min_n,
                  max_n=max_n, min_edges=min_edges, max_edges=max_edges)
    if not rows:
        raise ValueError("no rows left after the restrictions; widen them")

    absent: dict[str, int] = {}
    for record in rows:
        if record.status == "unavailable":
            absent[record.backend] = absent.get(record.backend, 0) + 1
    rows = [r for r in rows if r.status != "unavailable"]
    if absent:
        spelled = ", ".join(f"{count} for {backend}"
                            for backend, count in sorted(absent.items()))
        notes.append(f"dropped as 'unavailable', the backend not having run on the "
                     f"machine that produced the file, which is not a timeout and "
                     f"not a population member: {spelled}")
    if not rows:
        raise ValueError("every row is 'unavailable': these runs solved nothing")

    reference, suspect = cross_check(rows, tolerance)
    if suspect:
        worst = max(max(d.values()) for d in suspect.values())
        who = sorted({b for d in suspect.values() for b in d})
        notes.append(f"OBJECTIVES DISAGREE on {len(suspect)} instance(s), worst "
                     f"{worst:.1e} relative against {reference}: {', '.join(who)}. "
                     f"Those instances are dropped from every figure below -- a "
                     f"timing is only worth reading once the answer is right.")
        rows = [r for r in rows if r.instance not in suspect]

    present = [b for b in LOOK if b in {r.backend for r in rows}]
    notes.append(f"plotting {', '.join(present)}; {len({r.instance for r in rows})} "
                 f"instance(s) across {len({r.family for r in rows})} famil(ies) appear "
                 f"in at least one row, and each figure below draws over the ones its "
                 f"own backends all attempted; objectives checked against {reference}")
    return Prepared(tuple(rows), tuple(present), tuple(notes), reference)


# --- what a row contributes ------------------------------------------------


def _seconds(record: Result, which: str) -> float:
    if which == "solve":
        return record.solve_seconds
    if which == "setup":
        return record.setup_seconds
    return record.setup_seconds + record.solve_seconds


def _per_iteration(record: Result) -> float:
    if record.iterations <= 0:
        return math.nan
    return record.solve_seconds / record.iterations


def _iterations(record: Result) -> float:
    return float(record.iterations) if record.iterations > 0 else math.nan


def shifted_geometric_mean(values, shift: float) -> float:
    """`exp(mean(log(v + shift))) - shift`, the summary the tables report.

    The mean of a set of times whose ratio is the ratio of their means, made
    insensitive to differences far below `shift` so that one instance solved in
    microseconds cannot decide the comparison.  See `config.SGM_SHIFT_SECONDS`
    for why the shift is what it is.
    """
    values = np.asarray(list(values), dtype=float)
    return float(np.exp(np.mean(np.log(values + shift))) - shift)


def _median(values) -> float:
    return float(np.median(np.asarray(list(values), dtype=float)))


# --- the figures -----------------------------------------------------------


def ecdf(data: Prepared, *, seconds: str = "solve", backends=None, ax=None):
    """(a) Fraction of the population solved within a time budget, per backend.

    The population plot: one step per instance solved, so a curve's height at
    `t` is the fraction the backend finished in `t` seconds or less.  A run that
    hit the time limit is a censored observation -- it is in the denominator and
    never contributes a step -- so a backend that timed out on a third of the
    collection plateaus at two thirds, which is the shape to read the figure by.
    The legend says so for every curve that does not reach the top.
    """
    rows, names = data.population(backends)
    ax = axes(ax)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    times = {}
    for backend in [b for b in LOOK if b in {r.backend for r in rows}]:
        solved = sorted(_seconds(r, seconds) for r in rows
                        if r.backend == backend and r.status == "optimal"
                        and math.isfinite(_seconds(r, seconds)) and _seconds(r, seconds) > 0)
        times[backend] = solved

    finite = [t for solved in times.values() for t in solved]
    if not finite:
        raise ValueError(f"no finite positive {seconds} time in the selection")
    left, right = min(finite) / 1.6, max(finite) * 1.6

    for backend, solved in times.items():
        look = LOOK[backend]
        fraction = len(solved) / len(names)
        label = backend if fraction > 0.995 else f"{backend} ({fraction:.0%} solved)"
        xs = np.concatenate([[left], solved, [right]])
        ys = np.concatenate([[0.0], np.arange(1, len(solved) + 1) / len(names),
                             [fraction]])
        ax.step(xs, ys, where="post", color=look.colour, linestyle=look.dashes,
                linewidth=1.8, label=label)

    ax.set_xlim(left, right)
    ax.set_ylim(0.0, 1.02)
    _log_axis(ax.xaxis, compact=True)
    _finish(ax, title="Solved within a time budget",
            subtitle=f"{len(names)} instances, {SECONDS[seconds].removesuffix(' (s)')}",
            xlabel=SECONDS[seconds], ylabel="instances solved", legend="lower right")
    return ax


def _profile(data: Prepared, *, value, ylabel: str, title: str, axis: str,
             backends, counts: bool = False, ax=None):
    """Median of `value` against instance size, with the quartile band.

    Instances are grouped by family and variable count -- one rung of the
    benchmark's ladder, of which there are hundreds of instances -- and each
    group is drawn at its median size on the chosen axis.  Grouping by that size
    directly would not work: the generated instances draw their edges at random
    and no two land on the same count, so every point would be its own group and
    the figure a smear of single-instance markers.  A family with one instance
    per size is its own group either way, and the band collapses onto the
    marker.

    Both axes are logarithmic: the sizes span four decades by construction, and
    a linear axis would put every rung but the largest on top of the origin.
    """
    rows, names = data.population(backends)
    ax = axes(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")

    drawn = 0
    for backend in [b for b in LOOK if b in {r.backend for r in rows}]:
        groups: dict[tuple[str, int], list[tuple[float, float]]] = {}
        for record in rows:
            if record.backend != backend or record.status != "optimal":
                continue
            v = value(record)
            if math.isfinite(v) and v > 0:
                groups.setdefault((record.family, record.n), []).append(
                    (float(getattr(record, axis)), v))
        if not groups:
            continue
        rungs = sorted(((float(np.median([size for size, _ in group])),
                         np.percentile([v for _, v in group], (25, 50, 75)))
                        for group in groups.values()),
                       key=lambda rung: rung[0])   # by size; the arrays do not compare
        sizes = [size for size, _ in rungs]
        look, drawn = LOOK[backend], drawn + 1
        ax.fill_between(sizes, [q[0] for _, q in rungs], [q[2] for _, q in rungs],
                        color=look.colour, alpha=0.15, linewidth=0)
        ax.plot(sizes, [q[1] for _, q in rungs], color=look.colour,
                linestyle=look.dashes, linewidth=1.8, marker=look.marker,
                markersize=5.5, markeredgecolor="white", markeredgewidth=0.8,
                label=backend)
    if not drawn:
        raise ValueError(f"nothing to plot for {title!r}: no backend in the "
                         f"selection reports it")
    ax.autoscale_view()
    _log_axis(ax.xaxis, compact=True)
    _log_axis(ax.yaxis, compact=counts)
    _finish(ax, title=title, subtitle=f"{len(names)} instances", xlabel=axis,
            ylabel=ylabel, legend="upper left")
    return ax


def cost_per_iteration(data: Prepared, *, axis: str = "edges", ax=None):
    """(b) Seconds per iteration against instance size, first-order backends only.

    The elementwise-iteration claim in one figure: PRAQ's step is projections
    and elementwise arithmetic where cuPDLPx's is two sparse matrix-vector
    products, and this is what that costs per step at each size.  Dividing by
    `iterations` is why only `registry.FIRST_ORDER` is here.
    """
    return _profile(data, value=_per_iteration, ylabel="seconds per iteration",
                    title="Cost of one iteration", axis=axis,
                    backends=registry.FIRST_ORDER, ax=ax)


def iterations(data: Prepared, *, axis: str = "edges", ax=None):
    """(c) Iterations to converge against instance size, first-order backends only.

    Read together with the figure above: the two factors of a solve time are how
    many iterations it takes and what one costs, and a method can win either.
    Both sides stopped on the same relative duality gap, without which the
    counts would not be comparable -- the tables print the tolerance each
    backend was actually given.
    """
    return _profile(data, value=_iterations, ylabel="iterations",
                    title="Iterations to converge", axis=axis, counts=True,
                    backends=registry.FIRST_ORDER, ax=ax)


# --- the tables ------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    """A small table with its own provenance, as Markdown or as CSV.

    The table twin of every figure, which is also what makes the figures legible
    to a reader who cannot separate two of the colours: the numbers are here.
    """

    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...]

    def markdown(self) -> str:
        align = "|:---|" + "|".join(["---:"] * (len(self.columns) - 1)) + "|"
        lines = [f"### {self.title}", "",
                 "| " + " | ".join(self.columns) + " |", align]
        lines += ["| " + " | ".join(row) + " |" for row in self.rows]
        return "\n".join(lines + [""] + [f"- {note}" for note in self.notes]) + "\n"

    def write(self, stem: Path, *, comma_separated: bool = False) -> list[Path]:
        """Write `stem.md`, and `stem.csv` too if asked.

        Markdown is the one that gets read: it carries the notes, and the point
        of a table beside a figure is that a person can read the numbers off it.
        The CSV is the same rows without the prose, for something that wants to
        read them back, and it is off by default because two files per table is
        twice the directory to look through for a copy nothing has asked for.
        """
        written = [stem.with_suffix(".md")]
        written[0].write_text(self.markdown())
        if comma_separated:
            written.append(stem.with_suffix(".csv"))
            with open(written[1], "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(self.columns)
                writer.writerows(self.rows)
        return written


def _excluded(data: Prepared, backends) -> list[str]:
    """Backends in the selection that this figure or table does not draw.

    The tables name them, in a note with room to say why -- the ones left out of
    anything with iterations in it are out because their iterations are a
    different object.  The figures do not: a line of small print about a curve
    that is not there is clutter, and the legend already says what is.
    """
    return [b for b in data.backends if b not in backends]


def _decimal(value: float) -> str:
    return f"{value:.3g}"


def _count(value: float) -> str:
    return f"{value:.0f}"


def _bytes(value: float) -> str:
    """Powers of two, labelled as powers of two: NVML reports bytes."""
    for scale, suffix in ((2 ** 30, "GiB"), (2 ** 20, "MiB"), (2 ** 10, "kiB")):
        if value >= scale:
            scaled = value / scale
            # Not %g, which renders 1009 MiB as "1.01e+03" -- a table is for reading.
            digits = 0 if scaled >= 100 else (1 if scaled >= 10 else 2)
            return f"{scaled:.{digits}f} {suffix}"
    return f"{value:.0f} B"


def _decades(sizes) -> list[tuple[float, float, str]]:
    """The powers-of-ten bands `sizes` occupies, as (low, high, label).

    Buckets a reader can name -- "10k to 100k edges" -- and the coarsest ones
    that still separate the benchmark's rungs, which climb by about 3.2x.  Bands
    with nothing in them are left out rather than printed empty.
    """
    exponents = sorted({int(math.floor(math.log10(s))) for s in sizes if s > 0})
    return [(10.0 ** k, 10.0 ** (k + 1), f"{_si(10 ** k)}-{_si(10 ** (k + 1))}")
            for k in exponents]


def _bucketed(data: Prepared, *, value, summarise, cell, axis: str, backends,
              title: str, notes, sampled: bool = False, keep=None) -> Table:
    rows, names = data.population(backends, sampled=sampled, keep=keep)
    if not names:
        raise ValueError(f"nothing left for {title!r}: no instance that every one of "
                         f"{', '.join(backends)} "
                         + ("measured in a memory run" if sampled else "attempted"))
    size = {r.instance: getattr(r, axis) for r in rows}
    order = [b for b in LOOK if b in {r.backend for r in rows}]
    by_backend = {b: [r for r in rows if r.backend == b] for b in order}

    marked, bounded, table = False, False, []
    for low, high, label in _decades(size[name] for name in names):
        members = {name for name in names if low <= size[name] < high}
        if not members:
            continue
        cells = []
        for backend in order:
            finished, stopped = [], []
            for record in by_backend[backend]:
                if record.instance not in members:
                    continue
                v = value(record)
                if not (math.isfinite(v) and v > 0):
                    continue
                if record.status == "optimal":
                    finished.append(v)
                elif record.status in CENSORED:
                    stopped.append(v)
            if not finished:
                # Nothing converged, so every value here is the limit restated.
                # That is the same number for any solver that gives up, and the
                # per-backend note below already says how many did; a summary of
                # it would look like a measurement without being one.
                cells.append("--")
                continue
            # A run stopped at a limit enters at what it reached, which is a
            # floor -- so the bucket covers all its members and the entry is a
            # lower bound, rather than a mean over the finishers with the
            # hardest instances quietly dropped.
            text = cell(summarise(finished + stopped))
            if stopped:
                text, bounded = ">" + text, True
            if len(finished) + len(stopped) < len(members):
                text, marked = text + "*", True
            cells.append(text)
        table.append((label, str(len(members)), *cells))

    reached = "measured in a memory run" if sampled else "attempted"
    notes = [f"over the {len(names)} instances every backend in this table "
             f"{reached}, bucketed by {axis}", *notes]
    left_out = _excluded(data, backends)
    if left_out:
        notes.append("left out, their iterations not counting the same kind of "
                     f"step: {', '.join(left_out)}")
    if bounded:
        notes.append("> a lower bound: some of the runs summarised stopped at a "
                     "limit rather than at an answer, and enter at the value "
                     "they reached")
    if marked:
        notes.append("* summarises fewer instances than the row counts: the rest "
                     "reported no usable value at all")
    for backend in order:
        counts: dict[str, int] = {}
        for record in by_backend[backend]:
            if record.status != "optimal":
                counts[record.status] = counts.get(record.status, 0) + 1
        if counts:
            spelled = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
            notes.append(f"{backend}: {spelled} of {len(names)} instances")
    tolerances = sorted({r.optimality_tolerance for r in rows if math.isfinite(r.optimality_tolerance)})
    if tolerances:
        notes.append("relative duality gap asked of the first-order solvers: "
                     + ", ".join(f"{t:g}" for t in tolerances))
    return Table(title, (f"{axis}", "instances", *order), tuple(table), tuple(notes))


def solve_time_table(data: Prepared, *, seconds: str = "solve", axis: str = "edges",
                     shift: float = config.SGM_SHIFT_SECONDS) -> Table:
    """(d) Shifted geometric mean of solve time, by size bucket and backend."""
    return _bucketed(data, value=lambda r: _seconds(r, seconds),
                     summarise=lambda v: shifted_geometric_mean(v, shift),
                     cell=_decimal, axis=axis, backends=data.backends,
                     title=f"Shifted geometric mean of {SECONDS[seconds]}",
                     notes=[f"shift {shift:g} s; see harness/config.py for why"])


def iterations_table(data: Prepared, *, axis: str = "edges",
                     shift: float = config.SGM_SHIFT_ITERATIONS) -> Table:
    """(e) Shifted geometric mean of iterations, first-order backends only."""
    return _bucketed(data, value=_iterations,
                     summarise=lambda v: shifted_geometric_mean(v, shift),
                     cell=_count, axis=axis, backends=registry.FIRST_ORDER,
                     title="Shifted geometric mean of iterations to converge",
                     notes=[f"shift {shift:g} iterations; see harness/config.py"])


def cost_per_iteration_table(data: Prepared, *, axis: str = "edges") -> Table:
    """(b) Median seconds per iteration, by size bucket, first-order backends only.

    The median and not a shifted mean: a per-iteration cost is a ratio of two
    numbers the backend reports, in microseconds where a shift sized for solve
    times would swamp it.
    """
    return _bucketed(data, value=_per_iteration, summarise=_median,
                     cell=_decimal, axis=axis, backends=registry.FIRST_ORDER,
                     title="Median seconds per iteration", notes=[])


def _budget_spent(record: Result, tolerance: float) -> float:
    """The bracket a backend declares, against what agreement allows it.

    `harness.results.disagreement` scales an objective difference by
    ``1 + |objective|`` and compares it with `tolerance`, so that product is
    the budget an instance has; this is the share of it the backend's own
    uncertainty takes up.  One is where a backend's declared bracket is as
    wide as the disagreement that would fail a correctness check.
    """
    return uncertainty(record) / (tolerance * (1.0 + abs(record.objective)))


def uncertainty_table(data: Prepared, *, axis: str = "edges",
                      tolerance: float = config.AGREEMENT_TOLERANCE) -> Table:
    """(f) How much of the agreement budget each backend's own bracket spends.

    Not every backend claims an exact answer, and the ones that do not differ
    in how far off they admit they might be: the first-order solvers report a
    duality gap, and the max-flow oracle reports the slack from rounding its
    capacities to integers.  Both widen as instances grow, and a comparison
    against such a backend has to allow for that -- which
    `scripts/check_correctness.py` does -- so what matters is how much of the
    allowance is left, not the raw width.

    The largest in each bucket rather than a typical one, since the question
    is whether any instance is close to the limit.
    """
    return _bucketed(data, value=lambda r: _budget_spent(r, tolerance),
                     summarise=max, cell=_decimal, axis=axis,
                     backends=data.backends,
                     title="Largest declared uncertainty, as a share of the "
                           "agreement budget",
                     notes=[f"1.0 would be a bracket as wide as the objective "
                            f"difference that fails a correctness check, at "
                            f"tolerance {tolerance:g}",
                            "-- here means the backend declared no uncertainty "
                            "at all, which a vertex solve does"])


def peak_memory_table(data: Prepared, *, axis: str = "edges",
                      min_solve_seconds: float = config.MEMORY_MIN_SOLVE_SECONDS
                      ) -> Table:
    """Peak device memory by size bucket, from a memory run.

    Device and not host, because the device figure is the one that compares and
    the one that decides anything: both solvers hold their working set on the
    card, and whether an instance fits there is the question being asked.  The
    host column is in the CSV, and is not here on purpose -- praq-cuda's is a
    child process that read the instance from a file, while cuPDLPx's is this
    process and includes the LP the harness built, so the two are not the same
    measurement.

    The largest peak in each bucket rather than a typical one: what a reader
    wants from a memory number is whether it fits, and the bucket spans a decade
    of size, so the biggest instance in it is the one that answers that.

    Instances solved faster than `min_solve_seconds` are left out, sampling
    being blind to a peak shorter than its interval -- and left out for every
    backend at once, so the columns stay over the same instances.
    """
    return _bucketed(data, value=lambda r: r.peak_device_bytes, summarise=max,
                     cell=_bytes, axis=axis, backends=registry.SAMPLED_MEMORY,
                     sampled=True, keep=lambda r: r.solve_seconds >= min_solve_seconds,
                     title="Peak device memory",
                     notes=[f"largest peak in the bucket, from a memory run "
                            f"(run_benchmark.py --memory)",
                            f"instances solved in under {min_solve_seconds:g} s are left "
                            f"out: too short to sample; see harness/config.py"])
