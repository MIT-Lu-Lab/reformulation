#!/usr/bin/env bash
# Run the project's checks.  This file is the single home for what they are:
# .github/workflows/ci.yml sets up an environment and then calls this script, so
# CI and a local run cannot drift apart, and there is one place to add a check.
#
#     source env.sh && scripts/run_checks.sh            # everything
#     scripts/run_checks.sh docs tests                  # a subset
#     scripts/run_checks.sh --list                      # the names
#
# Run it before opening a PR.
#
# Exit status is 0 only when nothing failed.  A check whose tools are absent is
# SKIPped, not failed -- the CUDA tests need nvcc, shellcheck needs installing --
# but the summary says so, and a skip means the run did not cover that ground.
set -uo pipefail

# Usable standalone: source env.sh if the caller has not.
if [ -z "${PRIA_ROOT:-}" ]; then
    # shellcheck source=/dev/null
    . "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
fi
: "${PRIA_PYTHON:?env.sh did not set PRIA_PYTHON}"

# The order is cheapest-first, so a typo in a shell script fails in seconds
# rather than after the test suite.
ALL_CHECKS="shell lint scipy-free docs results model tuner correctness plots cuda-adapter tests cuda"

SKIP=77   # a check's tools are not present here

# What the correctness check runs on.  Its one home, so the shapes it tests
# cannot drift from the shapes the `model` check builds.  Small on purpose --
# a disagreement shows up at four periods as plainly as at four hundred, and
# this has to stay fast enough to be run before every PR.  Both parities are
# here, since an odd period count is padded and an even one is not, and a
# single item is the zero-joint-cost corner.
CHECK_SHAPES="--shapes 3 6 4 7 1 6 --trials 2"

# --- the checks ------------------------------------------------------------

check_shell() {   # every shell script parses, and is shellcheck-clean
    local script
    for script in "$PRIA_ROOT"/env.sh "$PRIA_ROOT"/scripts/*.sh; do
        bash -n "$script" || return 1
    done
    command -v shellcheck >/dev/null || {
        echo "  shellcheck not installed (brew install shellcheck)"; return $SKIP; }
    shellcheck --severity=warning "$PRIA_ROOT"/env.sh "$PRIA_ROOT"/scripts/*.sh
}

check_lint() {   # ruff, over every Python file in the repository
    # ruff.toml at the root is the configuration; pria-jax/ruff.toml extends
    # it for the library, and ruff applies the nearest one to each file.
    "$PRIA_PYTHON" -m ruff --version >/dev/null 2>&1 || {
        echo "  ruff not installed (see requirements.txt)"; return $SKIP; }
    "$PRIA_PYTHON" -m ruff check "$PRIA_ROOT"
}

check_scipy_free() {   # pria neither declares nor imports scipy
    # scipy is always present -- jax depends on it -- so its absence is not the
    # thing to test.  What keeps the library installable on numpy and jax is
    # that pria does not declare it and does not import it.
    "$PRIA_PYTHON" - <<'PY'
import importlib.metadata as md, sys
declared = [r for r in (md.requires("pria") or []) if "extra ==" not in r]
assert not any(r.lower().startswith("scipy") for r in declared), declared
import pria.prelude
assert "scipy" not in sys.modules, "pria imported scipy at import time"
print("  declared:", declared)
PY
}

check_docs() {   # the documentation builds with warnings as errors
    "$PRIA_PYTHON" -c "import sphinx" 2>/dev/null || {
        echo "  sphinx not installed (pip install -r docs/requirements.txt)"; return $SKIP; }
    "$PRIA_PYTHON" -m sphinx -b html -W "$PRIA_ROOT/docs" "$PRIA_ROOT/docs/_build/html"
}

check_results() {   # a run's rows survive the trip to disk, one file per run
    "$PRIA_PYTHON" - <<'PY'
import math
import os
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

root = os.environ["PRIA_ROOT"]
sys.path[:0] = [root, f"{root}/scripts"]

from harness.results import Result, claim_path, read_csv, write_csv

with tempfile.TemporaryDirectory() as directory:
    # Two passes that pick the same name -- which is what a second's worth of
    # timestamp resolution buys two jobs submitted together -- must not be
    # handed the same file.
    wanted = Path(directory) / "runs" / "benchmark-gurobi-20260101-000000.csv"
    claimed = [claim_path(wanted) for _ in range(3)]
    assert claimed[0] == wanted, claimed[0]
    assert len(set(claimed)) == 3, claimed
    assert all(path.exists() for path in claimed)

    # Writing fills the file that was claimed and leaves the others alone.
    row = Result(backend="gurobi", instance="jrp-medium-n10-t25-s0",
                 family="jrp-medium", items=10, periods=25, entries=3250,
                 status="optimal", objective=1059.0, lower=1059.0, upper=1059.0)
    assert claimed[1].read_text() == ""
    write_csv(claimed[1], [row])
    assert claimed[0].read_text() == ""

    # Field by field, because NaN marks an unmeasured column and does not
    # equal itself, so the dataclass's own == calls every row unequal.
    back, = read_csv(claimed[1])
    for field in fields(Result):
        wrote, got = getattr(row, field.name), getattr(back, field.name)
        same = (math.isnan(got) if isinstance(wrote, float) and math.isnan(wrote)
                else wrote == got)
        assert same, f"{field.name}: wrote {wrote!r}, read {got!r}"

print(f"  {len(claimed)} runs of one name got {len(set(claimed))} files; "
      f"a row round-trips unchanged")
PY
}

check_model() {   # the sparse LP is the LP it claims to be
    # baselines/model.py is the single home for the formulation every LP
    # baseline solves, and it is built with fancy indexing over a flattened
    # upper triangle -- exactly the sort of code that is wrong by an index and
    # still runs.  This rebuilds the same LP the slow, obvious way, dense and
    # in nested loops with every index spelled out, and checks the two agree.
    # A shared builder would make that a tautology, which is why there are two.
    "$PRIA_PYTHON" - <<'PY'
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.environ["PRIA_ROOT"])

from baselines import highs_backend
from harness.instances import random_jrp


def naive(instance):
    """The same LP, dense and looped, with s > t pinned out by its bounds."""
    n, t = instance.items, instance.periods
    holding, order = instance.holding_costs, instance.order_costs
    joint = instance.joint_order_cost
    n_x, n_y = n * t * t, n * t
    total = n_x + n_y + t

    def x_at(i, s, u):
        return i * t * t + s * t + u

    def y_at(i, s):
        return n_x + i * t + s

    def z_at(s):
        return n_x + n_y + s

    cost = np.zeros(total)
    for i in range(n):
        for s in range(t):
            cost[y_at(i, s)] = order[i]
            for u in range(s, t):
                cost[x_at(i, s, u)] = holding[i, s, u]
    for s in range(t):
        cost[z_at(s)] = joint

    upper, upper_rhs, equal, equal_rhs = [], [], [], []
    for i in range(n):
        for u in range(t):
            for s in range(u + 1):
                row = np.zeros(total); row[x_at(i, s, u)] = 1; row[y_at(i, s)] = -1
                upper.append(row); upper_rhs.append(0.0)
                row = np.zeros(total); row[x_at(i, s, u)] = 1; row[z_at(s)] = -1
                upper.append(row); upper_rhs.append(0.0)
            row = np.zeros(total)
            for s in range(u + 1):
                row[x_at(i, s, u)] = 1
            equal.append(row); equal_rhs.append(1.0)

    bounds = [(0.0, None)] * total
    for i in range(n):
        for s in range(t):
            for u in range(s):
                bounds[x_at(i, s, u)] = (0.0, 0.0)   # not a variable of the LP

    outcome = linprog(cost, A_ub=np.array(upper), b_ub=np.array(upper_rhs),
                      A_eq=np.array(equal), b_eq=np.array(equal_rhs),
                      bounds=bounds, method="highs")
    assert outcome.success, outcome.message
    return float(outcome.fun)


SHAPES = ((0, 3, 5), (1, 2, 7), (2, 4, 6), (3, 1, 6), (4, 5, 4))
for seed, items, periods in SHAPES:
    instance = random_jrp(seed, items, periods)
    sparse = highs_backend.relaxation_value(instance)
    dense = naive(instance)
    assert abs(sparse - dense) <= 1e-7 * (1.0 + abs(dense)), \
        f"n={items} t={periods}: sparse {sparse} vs dense {dense}"

# And the shape of what was built, which the value alone would not catch: a
# formulation with the right optimum and the wrong column count is still wrong.
from baselines import model
lp = model.build(random_jrp(0, 3, 5))
assert lp.entries == lp.items * lp.pairs == 3 * 15
assert lp.columns == lp.entries + lp.items * lp.periods + lp.periods
assert lp.rows == 2 * lp.entries + lp.items * lp.periods
assert lp.matrix.nnz == 5 * lp.entries, lp.matrix.nnz

print(f"  {len(SHAPES)} shapes agree with an independent dense construction; "
      f"{lp.rows} rows, {lp.columns} columns, {lp.matrix.nnz} nonzeros at 3x5")
PY
}

check_tuner() {   # the tuner's bookkeeping, without a GPU
    # tune_pria.py only runs where there is a libpria.so and a card, so the
    # half of it that is pure bookkeeping -- which shapes are drawn, how they
    # are weighted, how an earlier run's CSV is read back -- would otherwise be
    # exercised by nothing until a batch job fell over an hour in.  This runs
    # that half on the host.
    "$PRIA_PYTHON" -c "import optuna" 2>/dev/null || {
        echo "  optuna not installed (see requirements.txt)"; return $SKIP; }
    "$PRIA_PYTHON" - <<'PY'
import csv
import os
import sys
import tempfile
from pathlib import Path

root = os.environ["PRIA_ROOT"]
sys.path[:0] = [root, f"{root}/scripts"]

import numpy as np
import optuna

import tune_pria as tuner
from harness import instances

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- the ladder -------------------------------------------------------------
rungs = tuner.rungs_for(list(instances.BENCHMARK), pool_factor=3.0)
assert len(rungs) == sum(len(v) for v in instances.LADDER.values())
assert [r.entries for r in rungs] == sorted(r.entries for r in rungs), "not cheapest first"
assert all(r.pool >= r.draw and r.pool >= 4 for r in rungs)

# --- draws, and that the two seed ranges cannot collide ---------------------
rng = np.random.default_rng(0)
train = tuner.training_draw(rungs, rng)
assert len(train) == sum(r.draw for r in rungs)
plan = tuner.holdout_plan(rungs, factor=2.0)
assert not ({d.name for d in train} & {d.name for d in plan}), \
    "a hold-out instance is also a training instance"
assert all(d.seed >= tuner.TRAINING_SEEDS for d in train)
assert all(d.seed >= tuner.HOLDOUT_SEEDS for d in plan)
# ...and that neither can collide with the benchmark's own seeds.
benchmark = {s.name for s in instances.all_specs()}
assert not (benchmark & {d.name for d in train + plan}), \
    "a training or hold-out instance is in the benchmark"

# --- size_weights takes a Draw during the search and a Spec at the confirm
# step, and has to weight both the same way.
specs = [s for s in instances.all_specs(["jrp-medium"]) if s.name.endswith("-s0")]
for drawn in (train, plan, specs):
    weights = tuner.size_weights(drawn, 0.5)
    assert len(weights) == len(drawn) and np.all(weights > 0)
    # A shape's total weight is entries**power however many seeds it holds.
    sizes = sorted({d.entries for d in drawn})
    for size in sizes:
        total = sum(w for d, w in zip(drawn, weights) if d.entries == size)
        assert abs(total - size ** 0.5) < 1e-9 * size ** 0.5, (size, total)

# --- the name a Draw makes is the name an instance gets --------------------
one = train[0]
built = instances.random_jrp(one.seed, one.items, one.periods, name=one.name,
                             family=one.family)
assert built.name == one.name and built.entries == one.entries
assert tuner.shape_of(one.name) == (one.items, one.periods), \
    "the column parser and the name builder disagree"

# --- the sampler's coordinates round trip ----------------------------------
study = optuna.create_study(direction="minimize")
trial = study.ask(tuner.distributions())
point = tuner.suggest(trial)
assert set(point) == set(tuner.SCHEDULE), sorted(set(point) ^ set(tuner.SCHEDULE))
assert isinstance(point["op_steps"], int) and 2 <= point["op_steps"] <= 64
lo, hi = tuner.necessary_room(point["beta_sufficient"])
assert lo <= point["beta_necessary"] <= hi
back = tuner.to_params(point)
assert abs(back["beta_necessary_frac"]
           - (point["beta_necessary"] - lo) / (hi - lo)) < 1e-9
for name in tuner.searched():
    assert abs(float(back[name]) - float(point[name])) < 1e-12, name

# --- a check costs what the one home says it costs -------------------------
assert tuner.cost_of(100, 16, 0.0) == 100.0
assert abs(tuner.cost_of(100, 10, 1.0) - 110.0) < 1e-9
assert tuner.cost_of(100, 2, 1.0) > tuner.cost_of(100, 32, 1.0), \
    "a smaller interval must cost more, or the charge prices nothing"

# --- an earlier run's CSV reads back into a study --------------------------
with tempfile.TemporaryDirectory() as work:
    path = Path(work) / "tune-made-up.csv"
    names = [d.name for d in train[:4]]
    columns = ["config", *tuner.SCHEDULE, "geomean", "worst", "capped",
               "abandoned", "iterations", "seconds", *names]
    default = dict(omega=0.10835, beta_sufficient=0.3679, beta_necessary=0.5706,
                   beta_artificial=0.2468, theta_initial=0.8193,
                   theta_target=0.4282, theta_decay=0.9047, op_steps=16)
    rows = [dict(config=0, **default, geomean=1.0, worst=1.0, capped=0,
                 abandoned=0, iterations=400, seconds=1.0,
                 **{n: 100 for n in names}),
            dict(config=1, **dict(default, omega=0.2, op_steps=8), geomean=0.9,
                 worst=1.0, capped=0, abandoned=0, iterations=360, seconds=1.0,
                 **{n: 90 for n in names}),
            dict(config=2, **dict(default, omega=0.5, op_steps=32), geomean=2.5,
                 worst=3.0, capped=1, abandoned=1, iterations=900, seconds=1.0,
                 **{n: (220 if n == names[0] else "") for n in names})]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    seeded = optuna.create_study(direction="minimize")
    added = tuner.warm_start(seeded, path, charge=1.0, size_power=0.5)
    assert added == 2, f"the baseline row must not become a trial: {added}"
    states = sorted(t.state.name for t in seeded.trials)
    assert states == ["COMPLETE", "PRUNED"], states
    done = next(t for t in seeded.trials if t.state.name == "COMPLETE")
    # 90 iterations against the baseline's 100, at intervals of 8 and 16:
    # (90 * (1 + 1/8)) / (100 * (1 + 1/16)) = 101.25 / 106.25.
    assert abs(done.value - (101.25 / 106.25)) < 1e-9, done.value
    # The abandoned row scored only one instance, and is pruned at that value.
    partial = next(t for t in seeded.trials if t.state.name == "PRUNED")
    assert abs(partial.intermediate_values[0]
               - (220 * (1 + 1 / 32)) / (100 * (1 + 1 / 16))) < 1e-9

print(f"  {len(rungs)} rungs, {len(train)} training and {len(plan)} hold-out draws, "
      f"seeds disjoint from each other and from the benchmark; weights consistent "
      f"over Draw and Spec; 2 of 3 CSV rows warm-started")
PY
}

check_correctness() {   # two implementations agree on random instances
    # --highs is the only comparison that needs neither a licence nor a GPU,
    # which is why this is the check that always runs.
    # shellcheck disable=SC2086  # CHECK_SHAPES is a word list on purpose
    "$PRIA_PYTHON" "$PRIA_ROOT/scripts/check_correctness.py" \
        --highs --pria-jax $CHECK_SHAPES
}

check_plots() {   # the figures draw, and leave out what they have to leave out
    # Made-up rows rather than a real run: what needs testing is not a timing
    # but the bookkeeping around it, and every case has to be present at once.
    # The CSV below holds a backend that could not run here, a run that hit the
    # time limit, and a backend that disagrees about an objective -- each of
    # which the figures treat differently, and none of which a passing
    # benchmark run would contain.
    "$PRIA_PYTHON" -c "import matplotlib" 2>/dev/null || {
        echo "  matplotlib not installed (see requirements.txt)"; return $SKIP; }
    "$PRIA_PYTHON" - <<'PY'
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

root = os.environ["PRIA_ROOT"]
sys.path[:0] = [root, f"{root}/scripts"]

import plot_results
from harness import config, figures
from harness.instances import entries
from harness.results import Result, read_csv, write_csv

# The mean itself, whose two invariants no figure would reveal if they broke.
assert abs(figures.shifted_geometric_mean([2.0, 8.0], 0.0) - 4.0) < 1e-12
assert abs(figures.shifted_geometric_mean([1.0, 1.0], 5.0) - 1.0) < 1e-12

# Shapes chosen for where their x-variable counts land: (10, 25) is 3,250 and
# (12, 30) is 5,580, so they share the 1k-10k decade; (20, 60) is 36,600;
# (30, 110) is 183,150, alone in 100k-1M; (100, 200) is 2,010,000.
SHAPES = ((10, 25), (12, 30), (20, 60), (30, 110), (100, 150), (100, 200))
DISAGREES, TOO_FAST = (20, 60), (10, 25)
# (12, 30) shares its decade with (10, 25), which finishes, so that bucket is
# part censored; (30, 110) is alone in its decade, so that one is censored
# outright and enters at the limit it reached.  Both are lower bounds.
TIMES_OUT = ((12, 30), (30, 110))
# A pass that stopped before the top rungs.  (100, 150) is 1,132,500 x
# variables and (100, 200) is 2,010,000, so they share the 1M-10M decade and
# gurobi covers one of the two: the cell is partly covered rather than empty,
# and the instance it never ran has to enter at zero for the bound to hold.
NOT_REACHED = ((100, 200),)


def row(backend, shape, **kw):
    items, periods = shape
    size = entries(items, periods)
    fields = dict(objective=-float(size), lower=-float(size), upper=-float(size),
                  optimality_tolerance=1e-6)
    fields.update(kw)
    return Result(backend, f"made-up-n{items}-t{periods}", "made-up",
                  items, periods, size, "optimal", **fields)


def unavailable(backend, shape):
    items, periods = shape
    return Result(backend, f"made-up-n{items}-t{periods}", "made-up",
                  items, periods, entries(items, periods), "unavailable")


rows = []
for k, shape in enumerate(SHAPES):
    size = entries(*shape)
    rows.append(row("pria-cuda", shape, iterations=200 + 50 * k, setup_seconds=0.3,
                    solve_seconds=0.01 * (k + 1)))
    # A first-order backend brackets its answer: objective at `upper`, the
    # dual bound below.  Everything else here claims a vertex and declares
    # none, which is what the uncertainty table must show as a dash.
    rows.append(row("cupdlpx", shape, iterations=600 + 150 * k, setup_seconds=0.2,
                    solve_seconds=0.05 * (k + 1),
                    lower=-float(size) * 1.0000001))
    # Wrong by a percent on one instance: every backend's row for it should go.
    wrong = -float(size) * (1.01 if shape == DISAGREES else 1.0)
    rows.append(row("pria-jax", shape, iterations=300 + 60 * k, setup_seconds=1.2,
                    solve_seconds=0.1 * (k + 1), objective=wrong, lower=wrong,
                    upper=wrong))
    rows.append(unavailable("highs", shape))
    # A memory run over the same instances: separate rows, carrying peaks, whose
    # timings nothing plots.  The smallest is solved too fast to sample, so the
    # memory table must leave it out while the timing tables keep it.
    for backend, scale in (("pria-cuda", 1_000), ("cupdlpx", 2_500)):
        rows.append(row(backend, shape, iterations=100 + 50 * k, setup_seconds=0.3,
                        solve_seconds=0.01 if shape == TOO_FAST else 0.5,
                        peak_host_bytes=float(size) * scale,
                        peak_device_bytes=float(size) * scale))
    if shape in TIMES_OUT:
        items, periods = shape
        rows.append(Result("gurobi", f"made-up-n{items}-t{periods}", "made-up",
                           items, periods, size, "time_limit",
                           setup_seconds=1.0, solve_seconds=300.0))
    elif shape not in NOT_REACHED:
        rows.append(row("gurobi", shape, iterations=90 + 10 * k, setup_seconds=1.0,
                        solve_seconds=1.0 * (k + 1)))

with tempfile.TemporaryDirectory() as work:
    work = Path(work)
    csv = write_csv(work / "made-up.csv", rows)
    assert len(read_csv(csv)) == len(rows), "a CSV did not survive the round trip"

    # A CSV written before a column existed still reads, with that column's
    # default -- otherwise adding one to Result would orphan every run on disk.
    import csv as csv_module
    short = work / "column-short.csv"
    with open(csv, newline="") as handle:
        read = list(csv_module.DictReader(handle))
    keep = [c for c in read[0] if c not in ("optimality_tolerance", "note")]
    with open(short, "w", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=keep)
        writer.writeheader()
        writer.writerows({c: r[c] for c in keep} for r in read)
    aged = read_csv(short)
    assert len(aged) == len(rows) and aged[0].note == "", "a column-short CSV must read"

    def run(*extra):
        printed = io.StringIO()
        with redirect_stdout(printed):
            status = plot_results.main([str(csv), "--output", str(work), *extra])
        printed = printed.getvalue()
        assert status == 0, f"plot_results {extra} exited {status}\n{printed}"
        return printed

    # No --format and no --csv-tables: this pins what the defaults are.
    log = run()
    FIGURES = ("ecdf-solve-time", "cost-per-iteration-by-entries",
               "iterations-by-entries")
    TABLES = ("cost-per-iteration-by-entries", "sgm-solve-time-by-entries",
              "sgm-iterations-by-entries", "uncertainty-by-entries")
    for stem in FIGURES:
        path = (work / stem).with_suffix(".png")
        assert path.exists() and path.stat().st_size > 0, f"{path} not written"
    for stem in TABLES:
        path = (work / stem).with_suffix(".md")
        assert path.exists() and path.stat().st_size > 0, f"{path} not written"
        assert not path.with_suffix(".csv").exists(), f"{path}: CSV is meant to be asked for"
    assert not list(work.glob("*.pdf")), "PDF is meant to be asked for"

    # Every size axis draws, since --size-axis offers all three and an axis
    # that raised on a real CSV would only be found at plotting time.
    for axis in figures.AXES:
        run("--size-axis", axis)

    # The memory table is asked for by name, reads only the memory run's rows,
    # and drops the instance that was solved too fast to sample -- while the
    # timing tables, which never see those rows at all, keep it.
    memory_log = run("--figures", "peak-memory")
    memory = (work / "peak-memory-by-entries").with_suffix(".md")
    assert memory.exists(), "no memory table"
    memory = memory.read_text()
    assert "| 1k-10k | 1 |" in memory, f"the too-fast instance is still in it:\n{memory}"
    assert "MiB" in memory or "GiB" in memory, f"peaks are not in bytes:\n{memory}"
    timing = (work / "sgm-solve-time-by-entries").with_suffix(".md").read_text()
    assert "| 1k-10k | 2 |" in timing, f"a memory row reached a timing table:\n{timing}"

    # gurobi is the last column, and carries all three cases: a bucket it part
    # finished, one where every run hit the limit, and one it never reached.
    # The first two are lower bounds over every member; the third is empty,
    # because a row that does not exist is not a slow one.
    banded = {line.split("|")[1].strip():
              [c.strip() for c in line.strip().strip("|").split("|")]
              for line in timing.splitlines() if line.startswith("|")}
    assert banded["1k-10k"][-1].startswith(">"), \
        f"a part-censored bucket is not marked a lower bound:\n{timing}"
    assert banded["100k-1M"][-1].startswith(">"), \
        f"a wholly censored bucket is not marked a lower bound:\n{timing}"
    assert banded["1M-10M"][2] and not banded["1M-10M"][2].startswith(("-", ">")), \
        f"a backend that covered the bucket should be a plain number:\n{timing}"

    # The bound has to be one that holds, not just a marker.  gurobi covered
    # one of the two instances in the top decade, so its cell is the mean over
    # both with the one it never ran entering at zero -- summarising the
    # covered instance alone would be larger here, and is not a bound at all
    # when the instance left out is the faster one.
    ran = 1.0 * (SHAPES.index((100, 150)) + 1)
    both = figures.shifted_geometric_mean([ran, 0.0], config.SGM_SHIFT_SECONDS)
    covered = figures.shifted_geometric_mean([ran], config.SGM_SHIFT_SECONDS)
    assert banded["1M-10M"][-1] == ">" + f"{both:.3g}", \
        f"a part-covered bucket is not the bound {both:.3g}:\n{timing}"
    assert both < covered, "filling with zero must not raise the summary"
    assert "peak" not in log, "the default figure set should not need a memory run"

    # A limit does not censor a rate: the ratio is over the iterations the run
    # did take.  A bucket holding a stopped run therefore gives a plain
    # per-iteration number, while its solve time over the same bucket is a
    # bound.  Both shapes here are in one decade of entries.
    rate = 0.002
    timed = [Result("cupdlpx", f"rate-n{n}-t{t}", "made-up", n, t, entries(n, t),
                    status, objective=-1.0, lower=-1.0, upper=-1.0,
                    optimality_tolerance=1e-6, setup_seconds=0.1,
                    iterations=its, solve_seconds=rate * its)
             for (n, t), status, its in (((10, 25), "optimal", 100),
                                         ((12, 30), "time_limit", 50))]
    rated = figures.prepare([write_csv(work / "rates.csv", timed)])
    measured = figures.cost_per_iteration_table(rated).rows[0][-1]
    assert measured == f"{rate:.3g}", \
        f"a stopped run leaves the rate a measurement, not {measured!r}"
    bounded = figures.solve_time_table(rated).rows[0][-1]
    assert bounded.startswith(">"), \
        f"solve time over the same bucket is a bound, not {bounded!r}"

    # Declared uncertainty is a column, not prose: the backend that brackets
    # its answer gets a number, and the ones claiming a vertex get a dash.
    declared = (work / "uncertainty-by-entries").with_suffix(".md").read_text()
    banded = [[c.strip() for c in line.strip().strip("|").split("|")]
              for line in declared.splitlines() if line.startswith("| ")]
    columns, body = banded[0], [r for r in banded[1:] if r[1].isdigit()]
    assert body, f"no rows in the uncertainty table:\n{declared}"
    spent = {row[0]: dict(zip(columns[2:], row[2:])) for row in body}
    for band, cells in spent.items():
        assert cells["pria-cuda"] == "-", f"a vertex claim is not a dash:\n{declared}"
        assert 0.0 < float(cells["cupdlpx"]) < 1.0, \
            f"{band}: a declared bracket should spend part of the budget, not "\
            f"none or all of it:\n{declared}"

    # And the two things that have to be asked for arrive when they are.
    run("--csv-tables", "--format", "pdf")
    for stem in FIGURES:
        assert (work / stem).with_suffix(".pdf").exists(), f"{stem}: no PDF from --format"
    for stem in TABLES:
        assert (work / stem).with_suffix(".csv").exists(), f"{stem}: no CSV from --csv-tables"

    expected = [
        # The backend with no GPU here is not a timeout and not a population member.
        ("highs", "not run on the machine"),
        # The disagreement is named, loudly, and takes its instance with it.
        ("pria-jax", "OBJECTIVES DISAGREE"),
        (f"over the {len(SHAPES) - 1} instances", "population dropped the disagreement"),
        # A run that hit the limit stays in the count and in the summary, at
        # the value it reached, which makes that cell a lower bound.
        ("2 time_limit", "the censored runs"),
        ("a lower bound", "the note on a partly censored cell"),
        # Iteration counts are not comparable across the vertex line.
        ("gurobi", "left out of the iteration figures"),
    ]
    for text, what in expected:
        assert text in log, f"{what}: no {text!r} in\n{log}"

    print(f"  {len(rows)} made-up rows, {len(SHAPES)} shapes: png and markdown by "
          f"default, pdf and csv on request, {len(figures.AXES)} size axes, "
          f"1 disagreement dropped, {len(TIMES_OUT)} buckets bounded below by a "
          f"timeout and 1 by a shape the pass never reached, {len(spent)} bands "
          f"of declared uncertainty")
PY
}

check_cuda_adapter() {   # the pria-cuda adapter, against a stand-in binary
    # pria-cuda is driven as a subprocess, and on a machine with no nvcc none of
    # that is exercised: `cuda` skips and the adapter goes untested.  A stand-in
    # `pria_solve_file` that speaks the same two file formats tests this side of
    # the contract -- the launch, the timeout kill, the timings parse, the result
    # parse, the memory sampling -- without a GPU.  It says nothing about the
    # solver, which is what `cuda` is for.
    "$PRIA_PYTHON" - <<'PY'
import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

root = os.environ["PRIA_ROOT"]
sys.path.insert(0, root)

from harness.config import TOLERANCES
from harness.instances import random_jrp

STANDIN = """#!{python}
import struct, sys, time
instance, result, tolerance = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(instance, "rb") as handle:
    magic, n, t = struct.unpack("<3q", handle.read(24))
assert magic == 0x41495250, magic
print("setup_seconds: 0.25")
print("solve_seconds: 1.50")
time.sleep({sleep})
if {exit_code}:
    sys.exit({exit_code})
with open(result, "wb") as handle:
    handle.write(struct.pack("<4q3d", n, t, 1234, 0, -9.5, -9.0, 1e-9))
"""


def standin(directory, *, sleep=0.0, exit_code=0):
    """A fake pria_solve_file in its own PRIA_BIN_DIR, and that directory."""
    binary = Path(directory) / "pria_solve_file"
    binary.write_text(STANDIN.format(python=sys.executable, sleep=sleep,
                                     exit_code=exit_code))
    binary.chmod(0o755)
    os.environ["PRIA_BIN_DIR"] = str(directory)


#: How long the stand-in sleeps when the timeout is under test.  Far longer than
#: the shrunken grace below, so a timeout that waited the child out rather than
#: killing it shows up as a hang rather than as a pass.
SLEEP = 20.0

instance = random_jrp(0, 3, 4)
solved = failed = timed_out = None

with tempfile.TemporaryDirectory() as directory:
    standin(directory)
    import harness.backends.pria_cuda as adapter
    solved = adapter.solve(instance, tolerances=TOLERANCES, time_limit=30.0,
                           sample_memory=True)

with tempfile.TemporaryDirectory() as directory:
    standin(directory, exit_code=3)
    failed = adapter.solve(instance, tolerances=TOLERANCES, time_limit=30.0)

with tempfile.TemporaryDirectory() as directory:
    # The grace has to cover CUDA context creation, so it is a minute; the test
    # shrinks it, the mechanism being what is under test and not the constant.
    standin(directory, sleep=SLEEP)
    adapter.LAUNCH_GRACE = 0.2
    started = time.perf_counter()
    timed_out = adapter.solve(instance, tolerances=TOLERANCES, time_limit=0.3)
    waited = time.perf_counter() - started

assert solved.status == "optimal", solved
assert solved.iterations == 1234, solved
assert (solved.setup_seconds, solved.solve_seconds) == (0.25, 1.5), solved
assert (solved.lower, solved.upper, solved.objective) == (-9.5, -9.0, -9.0), solved
assert (solved.items, solved.periods) == (3, 4), solved
assert solved.entries == 3 * 4 * 5 // 2, solved
# Sampling a process this short-lived may catch nothing; what must hold is that
# the column is a number or an honest NaN, never a zero standing in for one.
assert math.isnan(solved.peak_host_bytes) or solved.peak_host_bytes > 0, solved
assert math.isnan(failed.peak_host_bytes), "an unsampled run must not report a peak"

assert failed.status == "error" and "exit 3" in failed.note, failed

assert timed_out.status == "time_limit", timed_out
assert waited < SLEEP, f"the timeout waited the child out instead of killing it: {waited:.1f}s"

print(f"  stand-in binary: solved, exit 3 reported, killed {waited:.2f}s into a "
      f"{SLEEP:g}s sleep")
PY
}

check_tests() {   # the pria-jax test suite
    # Each file is runnable as a script, which is how the README documents them;
    # running from pria-jax puts tests/ on sys.path so tests/reference.py resolves.
    local test status=0
    cd "$PRIA_ROOT/pria-jax" || return 1
    for test in tests/test_*.py; do
        echo "  == $test"
        "$PRIA_PYTHON" "$test" || status=1
    done
    return $status
}

check_cuda() {   # the pria-cuda test suite, where there is a GPU to run it
    command -v "${NVCC:-nvcc}" >/dev/null 2>&1 || {
        echo "  no nvcc at '${NVCC:-nvcc}'"; return $SKIP; }
    make -C "$PRIA_ROOT/pria-cuda" test
}

# --- driver ----------------------------------------------------------------

run_one() {
    local name="$1" started elapsed result
    printf '\n=== %s ===\n' "$name"
    started=$SECONDS
    "check_${name//-/_}"
    result=$?
    elapsed=$((SECONDS - started))
    case $result in
        0)     printf 'PASS  %s (%ds)\n' "$name" "$elapsed"; passed+=("$name") ;;
        "$SKIP") printf 'SKIP  %s\n' "$name";                  skipped+=("$name") ;;
        *)     printf 'FAIL  %s (%ds)\n' "$name" "$elapsed"; failed+=("$name") ;;
    esac
}

case "${1:-}" in
    --list) printf '%s\n' $ALL_CHECKS; exit 0 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
esac

requested=("$@")
[ ${#requested[@]} -eq 0 ] && read -r -a requested <<< "$ALL_CHECKS"

for name in "${requested[@]}"; do
    case " $ALL_CHECKS " in
        *" $name "*) ;;
        *) echo "unknown check: $name (see --list)" >&2; exit 2 ;;
    esac
done

passed=(); failed=(); skipped=()
for name in "${requested[@]}"; do run_one "$name"; done

printf '\n========================================\n'
printf '%d passed, %d failed, %d skipped\n' \
    "${#passed[@]}" "${#failed[@]}" "${#skipped[@]}"
[ ${#failed[@]} -gt 0 ] && printf 'failed:  %s\n' "${failed[*]}"
if [ ${#skipped[@]} -gt 0 ]; then
    printf 'skipped: %s\n' "${skipped[*]}"
    printf 'NOTE: skipped checks were not run -- this is not a full pass.\n'
fi
[ ${#failed[@]} -eq 0 ]
