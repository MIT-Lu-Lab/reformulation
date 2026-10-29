# The experiment harness

Not part of the library. `pria-jax` and `pria-cuda` never import any of this;
it exists to run and compare them.

## Environment

`env.sh` is the single home for every path the project uses. Nothing else
hard-codes one, and the Python side reads the variables rather than restating
their defaults — {py:mod}`harness.config` raises rather than guessing when they
are unset.

```sh
source env.sh
cp env.local.sh.example env.local.sh   # then edit; gitignored
```

Every assignment in `env.sh` is a `:=` default, so an already-exported variable
wins, and `env.local.sh` is sourced last so it wins over both. The defaults
point at `results/` and `bin/` at the repository root, both gitignored, so a
fresh clone runs with no configuration and your own paths never appear in a
diff.

| Variable | Holds |
|---|---|
| `PRIA_RESULTS_DIR` | benchmark output (CSV), figures, MPS staged by `export_mps` |
| `PRIA_BIN_DIR` | compiled CUDA binaries |
| `PRIA_PYTHON` | an interpreter with the dependencies below |
| `CUDA_HOME`, `NVCC`, `CUDA_ARCH` | the CUDA toolchain and target SM |
| `CUPDLPX_DIR` | the cuPDLPx checkout its Python package is built from |
| `GUROBI_THREADS` | pins Gurobi's pool to the allocation; unset leaves it automatic |

`CUDA_ARCH` has no default in `pria-cuda/Makefile` on purpose: `env.sh` owns it,
and the build stops with a message rather than guessing an SM.

## Scripts

```sh
scripts/build_cuda.sh [pria-cuda|cupdlpx]   # into $PRIA_BIN_DIR
scripts/clean.sh [--results] [--figs] [-n]  # generated files; -n to preview

$PRIA_PYTHON scripts/check_correctness.py --highs --pria-jax
$PRIA_PYTHON scripts/run_benchmark.py --pria-jax --families jrp-medium
$PRIA_PYTHON scripts/plot_results.py $PRIA_RESULTS_DIR/benchmark-*.csv
$PRIA_PYTHON scripts/tune_pria.py --buckets jrp-medium --configs 50
sbatch scripts/tune_pria.sbatch                          # every bucket, on a batch GPU
sbatch scripts/run_benchmark.sbatch --pria-cuda          # one pass on a batch GPU
```

`tune_pria.py` searches PRIA's restart schedule with Optuna's multivariate TPE,
running `pria-cuda` in process through {py:mod}`harness.pria_cuda_library` on
the benchmark's own distribution with seeds the benchmark never uses. Every
configuration is scored on its own random draw from per-shape seed pools, so
none can be selected for suiting particular instances; shapes are weighted by
the square root of their $x$-variable count, so the largest rungs lead without
one instance deciding. The search screens broadly and then confirms deeply: a
shortlist of eight is re-scored on a fixed hold-out with disjoint seeds, which
is what makes the per-shape numbers precise enough to reject a schedule that is
good on average but worse at one shape. Only the best few reach the benchmark
at the end. The default schedule is enqueued as a trial of its own rather than
assumed to score 1, and random probes are asked for on top of any warm-started
or enqueued trials rather than being consumed by them.

Unlike its QUBO sibling the check interval is a plain integer in the search
box, not a rule of the instance size: PRIA's `op_steps` does not read the
shape, so there is a number to tune and no law to fit.

A winner is applied by hand, to both implementations at once —
`Parameters.with_auto_step` in `pria-jax/pria/solver.py` and `pria_params_auto`
in `pria-cuda/src/pria.cu`, which say so at both ends.

`check_correctness.py` takes two or more implementations, runs them on random
instances, and treats the first named as the reference. `run_benchmark.py` takes
one or more, runs the generated benchmark unless `--families` says otherwise,
and writes a CSV per run; the file is rewritten after every instance, so an
interrupted run keeps its data.

`run_benchmark.sbatch` is one pass as a batch job. It names no implementation:
which ones a pass is for is what changes between submissions, so that comes
from the command line. One backend per job is the intended shape, because the
backends are not rerun at the same rate — pria-cuda is measured again after
every change to it, cuPDLPx is a fixed baseline measured once — and a combined
job would pay for the baseline every time. The cost of splitting is that each
job generates the instances itself, which for this problem is not negligible:
a JRP instance is a dense $N \times T \times T$ array of holding costs, and at
the top of the ladder drawing one takes about as long as solving it.

A pass that needs no GPU should not hold one. `--highs` and `--gurobi` are the
cases that matter, and a CPU partition has idle nodes when every GPU is spoken
for, so override the directives rather than keeping a second copy of the
script; the script's own header carries the worked example and is the copy to
trust if these two disagree.

`clean.sh` removes build products, Python caches, exported MPS and anything a
killed pria-cuda run left staged, by default. Benchmark results and figures
each need their own flag: results cannot be recreated at all, and figures,
though one `plot_results.py` away, are usually what you are in the middle of
reading.

### Checking versus benchmarking

`check_correctness.py` is the test and `run_benchmark.py` is the benchmark. The
test takes two or more implementations and says only whether they agree; the
benchmark runs them over everything and records what it cost.

```sh
# a pair against random instances -- seconds, and what the checks run
$PRIA_PYTHON scripts/check_correctness.py --highs --pria-jax

# a pair against the benchmark's own buckets, capped
$PRIA_PYTHON scripts/check_correctness.py --highs --cupdlpx --families jrp-medium
```

The second form has a size cap built in ({py:data}`harness.config.CHECK_MAX_ENTRIES`)
and a count cap on top of it ({py:data}`harness.config.CHECK_MAX_INSTANCES`),
spread across the shapes rather than keeping the smallest. Both matter: a test
that takes minutes stops being run, nothing above the size cap says anything new
about agreement, and one bucket alone holds hundreds of instances. Use
`--max-entries 0` and `--limit 0` to lift them, and `run_benchmark.py` when the
question is cost rather than agreement.

## Implementations

Both scripts take `--pria-jax`, `--pria-cuda`, `--highs`, `--gurobi` and
`--cupdlpx`. A backend that cannot run here — no GPU, no licence, binary not
built — reports `unavailable` and does not fail the run. `--highs` needs only
scipy, so it is the reference that always works.

Backends are imported on demand ({py:func}`harness.registry.load`), so a machine
with no Gurobi licence never pays for `gurobipy`.

:::{warning}
HiGHS and Gurobi both solve to a vertex — HiGHS by interior point with
crossover, Gurobi by barrier with crossover — while PRIA and cuPDLPx stop on a
relative duality gap. Objective values are comparable across all four, which is
what the correctness check uses. Iteration counts are comparable nowhere: a
barrier iteration, an IPM iteration and a PDHG step are three different things
and differ by orders of magnitude in both count and cost. Timings compare only
loosely — cuPDLPx is the like-for-like comparison.
:::

Both scripts take `--optimality-tolerance` and `--feasibility-tolerance`, and
{py:class}`harness.config.Tolerances` is where those two defaults live — the only
place either number is written down. Each backend maps them onto its own knobs:
cuPDLPx's `OptimalityTol` and `FeasibilityTol`, Gurobi's OptimalityTol and
FeasibilityTol, HiGHS's dual and primal feasibility tolerances. PRIA takes only
the first, its iterates being feasible by construction. Every record carries
what its backend was actually given, so a CSV says what was asked for rather
than what the run defaulted to, and the column is NaN where a backend has no
such knob.

cuPDLPx is driven through its Python package, which takes the LP as a SciPy
matrix, so no MPS file is written to solve one — which matters here more than it
does for a sparser problem, since the file would be some 60 bytes per nonzero
and the LP has five nonzeros per $x$ variable. `setup_seconds` is building that
LP and handing it over, which is this harness's overhead and not the solver's;
`solve_seconds` is the `optimize()` call, covering presolve, preconditioning,
the transfer to the device and the solve. cuPDLPx's own narrower `Runtime`,
which is the loop alone, goes to the record's `note`.

## Instances

Every instance is generated. {py:func}`harness.instances.random_jrp` is the one
generator, used by the benchmark and by the correctness check alike, and it is
written in numpy rather than imported from `pria-jax` so that a Gurobi-only run
needs no jax — it mirrors the distribution of `JrpProblem.random_instance`, not
jax's bit stream.

| Family | items × periods | $x$ variables |
|---|---|---|
| `jrp-medium` | 10×25 to 50×100 | 3.3k – 253k |
| `jrp-large` | 75×150 to 200×300 | 849k – 9.0M |
| `jrp-huge` | 250×375 to 350×450 | 17.6M – 35.5M |

Four shapes to a family, several seeds to a shape, and
{py:data}`harness.instances.LADDER` says how many of each; together they are
the benchmark and what `run_benchmark.py` runs by default. Both dimensions climb together along the ladder, because the two of
them move different costs — `periods` sets the size of every simplex the solver
projects onto and grows the instance quadratically, while `items` grows it
linearly and adds independent work — so a ladder that raised one alone would be
measuring a shape the problem is never posed in. The counts taper
geometrically, since the cost of a shape grows faster than the information
another seed at it buys, and the top rung's holding costs alone run to
hundreds of megabytes an instance.

The difficulty is in the draw of $K_0$. A study of it found the LP's
fractionality, and with it the iteration count of every first-order method
tried, peaking when the joint order cost is comparable with the item order
costs, which is what drawing both from the same distribution gives: a large
$K_0$ makes ordering jointly obviously right and the LP nearly integral, and a
small one decouples the items into $T$ independent lot-sizing problems.

A run enumerates its instances as {py:class}`harness.instances.Spec` objects,
which carry a name and a size but no arrays, and builds each one only when its
turn comes. That is what keeps a pass to one instance of memory: an instance
costs $8 N T^2$ bytes, so materialising a whole pass before solving any of it
is hundreds of gigabytes. `tune_pria.py` trains on the same shapes with
disjoint seeds and a fresh draw per configuration.

## Dependencies

`requirements.txt` at the repository root declares what `harness` and
`baselines` need: numpy, scipy, optuna and matplotlib. scipy is required, not
optional — {py:func}`baselines.model.build` returns a `scipy.sparse` matrix and
all three LP baselines go through it. `gurobipy` and a licence are needed only
for `--gurobi`, and `cupdlpx` and a GPU only for `--cupdlpx`.

The library declares its own separately in `pria-jax/pyproject.toml`: numpy and
jax at runtime, with scipy and pytest under its `dev` extra.
