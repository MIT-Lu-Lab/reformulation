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
if [ -z "${PRAQ_ROOT:-}" ]; then
    # shellcheck source=/dev/null
    . "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
fi
: "${PRAQ_PYTHON:?env.sh did not set PRAQ_PYTHON}"

# The order is cheapest-first, so a typo in a shell script fails in seconds
# rather than after the test suite.
ALL_CHECKS="shell scipy-free docs correctness maxflow plots cuda-adapter tests cuda"

SKIP=77   # a check's tools are not present here

# What the correctness checks run on.  Their one home: both checks below read
# it, so the two cannot end up testing different instances.  Small on purpose --
# a disagreement shows up at n = 25 as plainly as at n = 2500, and this has to
# stay fast enough to be run before every PR.
CHECK_INSTANCES="--sizes 25 40 --trials 2"

# --- the checks ------------------------------------------------------------

check_shell() {   # every shell script parses, and is shellcheck-clean
    local script
    for script in "$PRAQ_ROOT"/env.sh "$PRAQ_ROOT"/scripts/*.sh; do
        bash -n "$script" || return 1
    done
    command -v shellcheck >/dev/null || {
        echo "  shellcheck not installed (brew install shellcheck)"; return $SKIP; }
    shellcheck --severity=warning "$PRAQ_ROOT"/env.sh "$PRAQ_ROOT"/scripts/*.sh
}

check_scipy_free() {   # praq neither declares nor imports scipy
    # scipy is always present -- jax depends on it -- so its absence is not the
    # thing to test.  What keeps the library installable on numpy and jax is
    # that praq does not declare it and does not import it.
    "$PRAQ_PYTHON" - <<'PY'
import importlib.metadata as md, sys
declared = [r for r in (md.requires("praq") or []) if "extra ==" not in r]
assert not any(r.lower().startswith("scipy") for r in declared), declared
import praq.prelude
assert "scipy" not in sys.modules, "praq imported scipy at import time"
print("  declared:", declared)
PY
}

check_docs() {   # the documentation builds with warnings as errors
    "$PRAQ_PYTHON" -c "import sphinx" 2>/dev/null || {
        echo "  sphinx not installed (pip install -r docs/requirements.txt)"; return $SKIP; }
    "$PRAQ_PYTHON" -m sphinx -b html -W "$PRAQ_ROOT/docs" "$PRAQ_ROOT/docs/_build/html"
}

check_correctness() {   # two implementations agree on random instances
    # --highs is the only comparison that needs neither a licence nor a GPU,
    # which is why this is the check that always runs.
    # shellcheck disable=SC2086  # CHECK_INSTANCES is a word list on purpose
    "$PRAQ_PYTHON" "$PRAQ_ROOT/scripts/check_correctness.py" \
        --highs --praq-jax $CHECK_INSTANCES
}

check_maxflow() {   # the LP solvers and PRAQ agree with the combinatorial oracle
    # Unlike --gurobi and --cupdlpx, this one may go in an automated check: it
    # asks for ortools first and SKIPs, so it cannot pass while testing nothing.
    # The roof dual is a maximum flow, computed here in integers with no
    # tolerance and no iteration count, so it is the one reference in the
    # repository that does not itself stop early -- hence the reference here.
    "$PRAQ_PYTHON" -c "from ortools.graph.python import max_flow" 2>/dev/null || {
        echo "  ortools not installed (see requirements.txt)"; return $SKIP; }
    # shellcheck disable=SC2086  # CHECK_INSTANCES is a word list on purpose
    "$PRAQ_PYTHON" "$PRAQ_ROOT/scripts/check_correctness.py" \
        --maxflow --highs --praq-jax $CHECK_INSTANCES
}

check_plots() {   # the figures draw, and leave out what they have to leave out
    # Made-up rows rather than a real run: what needs testing is not a timing
    # but the bookkeeping around it, and every case has to be present at once.
    # The CSV below holds a backend that could not run here, a run that hit the
    # time limit, and a backend that disagrees about an objective -- each of
    # which the figures treat differently, and none of which a passing
    # benchmark run would contain.
    "$PRAQ_PYTHON" -c "import matplotlib" 2>/dev/null || {
        echo "  matplotlib not installed (see requirements.txt)"; return $SKIP; }
    "$PRAQ_PYTHON" - <<'PY'
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

root = os.environ["PRAQ_ROOT"]
sys.path[:0] = [root, f"{root}/scripts"]

import plot_results
from harness import figures
from harness.results import Result, read_csv, write_csv

# The mean itself, whose two invariants no figure would reveal if they broke.
assert abs(figures.shifted_geometric_mean([2.0, 8.0], 0.0) - 4.0) < 1e-12
assert abs(figures.shifted_geometric_mean([1.0, 1.0], 5.0) - 1.0) < 1e-12

SIZES = (500, 800, 5000, 20_000, 100_000)
DISAGREES, TOO_FAST = 5000, 500
# 800 shares the 1k-10k bucket with 500, so that bucket is part censored and
# gets a lower bound; 20_000 is alone in 100k-1M, so that one is wholly
# censored and has nothing to summarise.
TIMES_OUT = (800, 20_000)


def row(backend, n, **kw):
    fields = dict(objective=-float(n), lower=-float(n), upper=-float(n),
                  optimality_tolerance=1e-8)
    fields.update(kw)
    return Result(backend, f"made-up-n{n}", "made-up", n, 10 * n, "optimal", **fields)


rows = []
for k, n in enumerate(SIZES):
    rows.append(row("praq-cuda", n, iterations=200 + 50 * k, setup_seconds=0.3,
                    solve_seconds=0.001 * (k + 1)))
    # A first-order backend brackets its answer: objective at `upper`, the
    # dual bound below.  Everything else here claims a vertex and declares
    # none, which is what the uncertainty table must show as a dash.
    rows.append(row("cupdlpx", n, iterations=600 + 150 * k, setup_seconds=0.2,
                    solve_seconds=0.005 * (k + 1),
                    lower=-float(n) * 1.0000001))
    # Wrong by a percent on one instance: every backend's row for it should go.
    wrong = -float(n) * (1.01 if n == DISAGREES else 1.0)
    rows.append(row("praq-jax", n, iterations=300 + 60 * k, setup_seconds=1.2,
                    solve_seconds=0.01 * (k + 1), objective=wrong, lower=wrong,
                    upper=wrong))
    rows.append(Result("highs", f"made-up-n{n}", "made-up", n, 10 * n, "unavailable"))
    # A memory run over the same instances: separate rows, carrying peaks, whose
    # timings nothing plots.  The smallest is solved too fast to sample, so the
    # memory table must leave it out while the timing tables keep it.
    for backend, scale in (("praq-cuda", 1_000_000), ("cupdlpx", 2_500_000)):
        rows.append(row(backend, n, iterations=100 + 50 * k, setup_seconds=0.3,
                        solve_seconds=0.01 if n == TOO_FAST else 0.5,
                        peak_host_bytes=float(n) * scale,
                        peak_device_bytes=float(n) * scale))
    if n in TIMES_OUT:
        rows.append(Result("gurobi", f"made-up-n{n}", "made-up", n, 10 * n,
                           "time_limit", setup_seconds=1.0, solve_seconds=300.0))
    else:
        rows.append(row("gurobi", n, iterations=90 + 10 * k, setup_seconds=1.0,
                        solve_seconds=0.5 * (k + 1)))

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
        writer.writerows({c: row[c] for c in keep} for row in read)
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
    FIGURES = ("ecdf-solve-time", "cost-per-iteration-by-edges", "iterations-by-edges")
    TABLES = ("cost-per-iteration-by-edges", "sgm-solve-time-by-edges",
              "sgm-iterations-by-edges", "uncertainty-by-edges")
    for stem in FIGURES:
        path = (work / stem).with_suffix(".png")
        assert path.exists() and path.stat().st_size > 0, f"{path} not written"
    for stem in TABLES:
        path = (work / stem).with_suffix(".md")
        assert path.exists() and path.stat().st_size > 0, f"{path} not written"
        assert not path.with_suffix(".csv").exists(), f"{path}: CSV is meant to be asked for"
    assert not list(work.glob("*.pdf")), "PDF is meant to be asked for"

    # The memory table is asked for by name, reads only the memory run's rows,
    # and drops the instance that was solved too fast to sample -- while the
    # timing tables, which never see those rows at all, keep it.
    memory_log = run("--figures", "peak-memory")
    memory = (work / "peak-memory-by-edges").with_suffix(".md")
    assert memory.exists(), "no memory table"
    memory = memory.read_text()
    assert "| 1k-10k | 1 |" in memory, f"the too-fast instance is still in it:\n{memory}"
    assert "MiB" in memory or "GiB" in memory, f"peaks are not in bytes:\n{memory}"
    timing = (work / "sgm-solve-time-by-edges").with_suffix(".md").read_text()
    assert "| 1k-10k | 2 |" in timing, f"a memory row reached a timing table:\n{timing}"

    # gurobi is the last column; a censored run enters at what it reached, so a
    # bucket holding one is a lower bound over all of its members.  A bucket
    # where every run hit the limit holds no measurement at all -- only the
    # limit restated, which is the same number for anyone who gives up -- so it
    # is dashed, and the per-backend note carries the count instead.
    banded = {line.split("|")[1].strip(): [c.strip() for c in line.strip().strip("|").split("|")]
              for line in timing.splitlines() if line.startswith("|")}
    assert banded["1k-10k"][-1].startswith(">"), \
        f"a part-censored bucket is not marked a lower bound:\n{timing}"
    assert banded["100k-1M"][-1] == "--", \
        f"a wholly censored bucket was summarised anyway:\n{timing}"
    assert "peak" not in log, "the default figure set should not need a memory run"

    # Declared uncertainty is a column, not prose: the backend that brackets
    # its answer gets a number, and the ones claiming a vertex get a dash.
    declared = (work / "uncertainty-by-edges").with_suffix(".md").read_text()
    banded = [[c.strip() for c in line.strip().strip("|").split("|")]
              for line in declared.splitlines() if line.startswith("| ")]
    columns, body = banded[0], [r for r in banded[1:] if r[1].isdigit()]
    assert body, f"no rows in the uncertainty table:\n{declared}"
    spent = {row[0]: dict(zip(columns[2:], row[2:])) for row in body}
    for band, cells in spent.items():
        assert cells["praq-cuda"] == "--", f"a vertex claim is not a dash:\n{declared}"
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
        ("praq-jax", "OBJECTIVES DISAGREE"),
        (f"over the {len(SIZES) - 1} instances", "population dropped the disagreement"),
        # A run that hit the limit stays in the count and in the summary, at
        # the value it reached, which makes that cell a lower bound.
        ("2 time_limit", "the censored runs"),
        ("a lower bound", "the note on a partly censored cell"),
        # Iteration counts are not comparable across the simplex line.
        ("gurobi", "left out of the iteration figures"),
    ]
    for text, what in expected:
        assert text in log, f"{what}: no {text!r} in\n{log}"

    print(f"  {len(rows)} made-up rows, {len(SIZES)} sizes: png and markdown by "
          f"default, pdf and csv on request, 1 disagreement dropped, "
          f"1 bucket bounded below and 1 dashed by {len(TIMES_OUT)} timeouts, "
          f"1 backend declaring a bracket and {len(spent)} bands of it drawn")
PY
}

check_cuda_adapter() {   # the praq-cuda adapter, against a stand-in binary
    # praq-cuda is driven as a subprocess, and on a machine with no nvcc none of
    # that is exercised: `cuda` skips and the adapter goes untested.  A stand-in
    # `praq_solve_file` that speaks the same two file formats tests this side of
    # the contract -- the launch, the timeout kill, the timings parse, the result
    # parse, the memory sampling -- without a GPU.  It says nothing about the
    # solver, which is what `cuda` is for.
    "$PRAQ_PYTHON" - <<'PY'
import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

root = os.environ["PRAQ_ROOT"]
sys.path.insert(0, root)

from harness.config import TOLERANCES
from harness.instances import random_instance

STANDIN = """#!{python}
import struct, sys, time
instance, result, tolerance = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(instance, "rb") as handle:
    magic, n, m = struct.unpack("<3q", handle.read(24))
assert magic == 0x51415250, magic
print("setup_seconds: 0.25")
print("solve_seconds: 1.50")
time.sleep({sleep})
if {exit_code}:
    sys.exit({exit_code})
with open(result, "wb") as handle:
    handle.write(struct.pack("<3q2d", n, 1234, 0, -9.5, -9.0))
    handle.write(struct.pack(f"<{{n}}d", *([0.5] * n)))
"""


def standin(directory, *, sleep=0.0, exit_code=0):
    """A fake praq_solve_file in its own PRAQ_BIN_DIR, and that directory."""
    binary = Path(directory) / "praq_solve_file"
    binary.write_text(STANDIN.format(python=sys.executable, sleep=sleep,
                                     exit_code=exit_code))
    binary.chmod(0o755)
    os.environ["PRAQ_BIN_DIR"] = str(directory)


#: How long the stand-in sleeps when the timeout is under test.  Far longer than
#: the shrunken grace below, so a timeout that waited the child out rather than
#: killing it shows up as a hang rather than as a pass.
SLEEP = 20.0

instance = random_instance(0, 6, 1.0)
solved = failed = timed_out = None

with tempfile.TemporaryDirectory() as directory:
    standin(directory)
    import harness.backends.praq_cuda as adapter
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

check_tests() {   # the praq-jax test suite
    # Each file is runnable as a script, which is how the README documents them;
    # running from praq-jax puts tests/ on sys.path so tests/reference.py resolves.
    local test status=0
    cd "$PRAQ_ROOT/praq-jax" || return 1
    for test in tests/test_*.py; do
        echo "  == $test"
        "$PRAQ_PYTHON" "$test" || status=1
    done
    return $status
}

check_cuda() {   # the praq-cuda test suite, where there is a GPU to run it
    command -v "${NVCC:-nvcc}" >/dev/null 2>&1 || {
        echo "  no nvcc at '${NVCC:-nvcc}'"; return $SKIP; }
    make -C "$PRAQ_ROOT/praq-cuda" test
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
