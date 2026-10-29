# The experiment harness

Not part of the library. `praq-jax` and `praq-cuda` never import any of this;
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
point at `results/` and `bin/` at the repository root, both gitignored,
so a fresh clone runs with no configuration and your own paths never appear in a
diff.

| Variable | Holds |
|---|---|
| `PRAQ_RESULTS_DIR` | benchmark output (CSV) and MPS staged by `export_mps` |
| `PRAQ_BIN_DIR` | compiled CUDA binaries |
| `PRAQ_PYTHON` | an interpreter with the dependencies below |
| `CUDA_HOME`, `NVCC`, `CUDA_ARCH` | the CUDA toolchain and target SM |
| `CUPDLPX_DIR`, `CUPDLPX_BIN` | the cuPDLPx checkout and executable |

`CUDA_ARCH` has no default in `praq-cuda/Makefile` on purpose: `env.sh` owns it,
and the build stops with a message rather than guessing an SM.

## Scripts

```sh
scripts/build_cuda.sh [praq-cuda|cupdlpx]   # into $PRAQ_BIN_DIR
scripts/clean.sh [--results] [--figs] [-n]  # generated files; -n to preview

$PRAQ_PYTHON scripts/check_correctness.py --highs --praq-jax
$PRAQ_PYTHON scripts/run_benchmark.py --praq-jax --families hard-medium
$PRAQ_PYTHON scripts/tune_praq.py --buckets hard-medium --configs 50  # restart schedule
sbatch scripts/tune_praq.sbatch                                          # every bucket, on a batch GPU
sbatch scripts/run_benchmark.sbatch --praq-cuda          # one pass on a batch GPU
```

`tune_praq.py` searches PRAQ's restart schedule, including the rule that sets the
check interval from the instance size, with Optuna's multivariate TPE, running
`praq-cuda` in process through `harness/praq_cuda_library.py` on the
hard buckets' distribution with seeds the benchmark never uses. Every
configuration is scored on its own random draw from per-size seed pools, so
none can be selected for suiting particular instances; sizes run in a ladder at
about √2 from 100k to 4M nodes and are weighted by √n, so the largest rungs
lead without one instance deciding. The search screens broadly and then
confirms deeply: a shortlist of eight is re-scored on a fixed hold-out drawn
eight times as deep, which is what makes the per-size numbers precise enough to
reject a schedule that is good on average but worse at one size. Only the best
few reach the benchmark at the end.
The default schedule is enqueued as a trial of its own rather than assumed to
score 1, and random probes are asked for on top of any warm-started or
enqueued trials rather than being consumed by them. A winner is applied by
hand, to both implementations at once.

`check_correctness.py` takes two or more implementations, runs them on random
instances, and treats the first named as the reference. `run_benchmark.py` takes
one or more, runs the generated benchmark unless `--families` says otherwise,
and writes a CSV per run; the file is rewritten after every instance, so an
interrupted run keeps its data.

`run_benchmark.sbatch` is one pass as a batch job. It names no
implementation: which ones a pass is for is what changes between submissions,
so that comes from the command line. One backend per job is the intended
shape, because the backends are not rerun at the same rate — praq-cuda is
measured again after every change to it, cuPDLPx is a fixed baseline measured
once — and a combined job would pay for the baseline every time.

That only works if the passes are comparable, which is why the job pins
`--gres=gpu:h100:1` instead of taking whatever is free: a pass is read against
passes submitted days apart, so the card cannot be up to the scheduler. The
cost of splitting is that each job generates the instances itself, which takes
about as long as solving them.

A pass that needs no GPU should not hold one. `--maxflow` is the case that
matters — the roof dual as a maximum flow is a CPU computation — and a CPU
partition has idle nodes when every GPU is spoken for, so override the
directives rather than keeping a second copy of the script:

```sh
sbatch --partition=<cpu-partition> --gres=none \
       scripts/run_benchmark.sbatch --maxflow
```

Pick a CPU partition whose OS matches the one the venv was built on — numpy
built against a newer glibc fails on an older node with `GLIBC_2.27 not
found`. Nodes of one cluster can differ in this.

A CPU baseline that is not first-order needs `--time` raised too, and by a lot:
`--maxflow` took 2h41 over the collection against the script's six-hour
default, but `--gurobi` took 36h51, three quarters of it on instances it
abandoned at the per-instance limit. The script's own header carries the
worked example and is the copy to trust if these two disagree.

The script builds praq-cuda only where there is a card to build for, so it
runs unchanged on a CPU node.

Two passes at once need a dependency rather than a second GPU:

```sh
a=$(sbatch --parsable scripts/run_benchmark.sbatch --praq-cuda)
sbatch --dependency=afterany:$a scripts/run_benchmark.sbatch --cupdlpx
```

A per-user GPU cap counts an interactive session holding one against it, so
an unchained second pass stops on `QOSMaxGRESPerUser` behind your own shell —
which reads as a busy cluster and is not one. Running the passes back to back
is the reliable order.

`clean.sh` removes build products, Python caches and exported MPS by default.
Benchmark results and figures each need their own flag: results cannot be
recreated at all, and figures, though one `plot_results.py` away, are usually
what you are in the middle of reading.

### Checking versus benchmarking

`check_correctness.py` is the test and `run_benchmark.py` is the benchmark. The
test takes two or more implementations and says only whether they agree; the
benchmark runs them over everything and records what it cost.

```sh
# a pair against random instances -- seconds, and what CI runs
$PRAQ_PYTHON scripts/check_correctness.py --highs --praq-jax

# a pair against the benchmark's own buckets, capped -- about 40 seconds
$PRAQ_PYTHON scripts/check_correctness.py --highs --cupdlpx --families hard-medium
```

The second form has the size cap built in (`config.CHECK_MAX_EDGES`), and that
is what keeps it to 40 seconds: a test that takes minutes stops being run.
Nothing above the cap says anything new about agreement — the small instances
disagree just as visibly — while the large ones exist to be *timed*: on the top
rung cuPDLPx takes 28 s against praq-cuda's 3.7 s, and that ratio is the whole
point of them. Use `--max-edges 0` to lift the cap on
the rare occasion that is what you want, and `run_benchmark.py` when the
question is cost rather than agreement.

## Implementations

Both scripts take `--praq-jax`, `--praq-cuda`, `--highs`, `--gurobi` and
`--cupdlpx`. A backend that cannot run here — no GPU, no licence, binary not
built — reports `unavailable` and does not fail the run. `--highs` needs only
scipy, so it is the reference that always works.

Backends are imported on demand ({py:func}`harness.registry.load`), so a machine
with no Gurobi licence never pays for `gurobipy`.

:::{warning}
HiGHS and Gurobi both solve to a vertex — HiGHS by interior point with
crossover, Gurobi by dual simplex — while PRAQ and cuPDLPx stop on a relative
duality gap. Objective values are comparable across all four, which is what the
correctness check uses. Iteration counts are comparable nowhere: an IPM
iteration, a simplex pivot and a PDHG step are three different things, and on
one LP they can number 17, 245k and 1200 respectively. Timings compare only
loosely — cuPDLPx is the like-for-like comparison.
:::

Both scripts take `--optimality-tolerance` and `--feasibility-tolerance`, and
{py:class}`harness.config.Tolerances` is where those two defaults live — the only
place either number is written down. Each backend maps them onto its own knobs:
cuPDLPx's `--eps_opt` and `--eps_feas`, Gurobi's OptimalityTol and FeasibilityTol,
HiGHS's dual and primal feasibility tolerances. PRAQ takes only the first, its
iterates being feasible by construction. Every record carries what its backend
was actually given, so a CSV says what was asked for rather than what the run
defaulted to, and the column is NaN where a backend has no such knob.

cuPDLPx is driven through its Python package, which takes the linearization as a
SciPy matrix, so no MPS file is written to solve one. `setup_seconds` is building
that LP and handing it over, which is this harness's overhead and not the
solver's; `solve_seconds` is the `optimize()` call, covering presolve,
preconditioning, the transfer to the device and the solve. cuPDLPx's own narrower
`Runtime`, which is the loop alone, goes to the record's `note`.

## Instance families

Every instance is generated, and {py:mod}`harness.instances` states them all as
**minimisation** QUBOs so no backend has to know how one was built.

| Family | Instances | $n$ | Density | Source |
|---|---:|---|---|---|
| `hard-medium` | 1464 | 500–5000 | ~10 edges per node | generated, {py:func}`harness.instances.random_hard` |
| `hard-large` | 585 | 10k–100k | ~10 edges per node | generated |
| `hard-huge` | 233 | 200k–2M | ~10 edges per node | generated |

Those 2282 instances are the benchmark and what `run_benchmark.py` runs by
default. They span five orders of magnitude in edge count, which is what makes
the collection a scaling study rather than one ECDF population. They are built
on demand from {py:func}`harness.instances.random_hard`, whose defaults are
the hardest corner of a knob study (log-normal weights with log standard
deviation 2, a quarter negative, diagonal a tenth of the incident weight);
`HARD_BUCKETS` fixes the sizes and seeds. The counts taper geometrically, 500
instances at 500 nodes down to 40 at 2M, which costs about forty minutes of L40S
a pass: twelve minutes of solving and the rest generating, since a 2M instance
takes longer to build than to solve. One instance per size was the old shape and
could not support a claim about anything, the spread between seeds at a fixed
size being tens of percent.

Every instance has degree 20, which is a measured local maximum of difficulty:
at 5k, 20k and 100k nodes the iteration count peaks there and falls by two to
eight times at degree 4 and at degree 500 alike. It also means the edge count is
exactly ten times the variable count, so a fit on the benchmark alone cannot say
which of the two anything depends on. Varying the degree would break that tie,
but only by adding instances that are much easier, so it belongs in a study of
its own rather than in the benchmark.

A run enumerates its instances as {py:class}`harness.instances.Spec` objects,
which carry a name and a size but no arrays, and builds each one only when its
turn comes. That is what keeps a pass to one instance of memory: a generated
instance costs about 25 bytes per edge, so the top rung alone is a gigabyte,
and materialising a whole pass before solving any of it does not scale past the
set we have today. Iterations grow with size, and on the top
rung `praq-cuda` takes half a minute and cuPDLPx over a minute, on a standard
linearization of 60M rows. `tune_praq.py` trains on a finer ladder of the same
distribution with disjoint seeds, a fresh draw per configuration, and sizes
weighted by √n, so the largest rungs lead and the small ones break ties.

## Dependencies

`requirements.txt` at the repository root declares what `harness` and
`baselines` need: numpy and scipy. scipy is required, not optional —
{py:func}`baselines.model.build` returns a `scipy.sparse` matrix and all three LP
baselines go through it. `gurobipy` and a licence are needed only for
`--gurobi`.

The library declares its own separately in `praq-jax/pyproject.toml`: numpy and
jax at runtime, with scipy and pytest under its `dev` extra.
