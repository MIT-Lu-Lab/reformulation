# QUBO experiments

PRAQ (Projected Restarted Algorithm for QUBO) and the baselines it is measured
against, in one repository.

PRAQ solves the LP relaxation of a QUBO's standard linearization — equivalently
its **roof dual** — by restarted reflected Halpern PDHG on a bilinear
reformulation, in a half-edge space where the coupling operator is the identity.
The derivation is in the paper named in the repository root README.

## Installation

```sh
cd qubo                                # from the repository root
source env.sh                          # every path the project uses
```

`env.sh` is the single home for paths and is **sourced, not executed**. Its
defaults put `results/` and `bin/` at the repository root, both gitignored, so
a fresh clone runs with no configuration. Every assignment in it is a `:=`
default, so an already-exported variable wins, and `env.local.sh` is sourced
last and wins over both:

```sh
cp env.local.sh.example env.local.sh   # then edit; gitignored
```

That is where a cluster CUDA install, a scratch results directory or your own
interpreter go, without ever showing up in a diff.

### Python

`PRAQ_PYTHON` must name the interpreter everything below runs under. A
virtualenv at the repository root is the usual choice; point `PRAQ_PYTHON` at
it in `env.local.sh`.

```sh
$PRAQ_PYTHON -m pip install -r requirements.txt   # harness and baselines
$PRAQ_PYTHON -m pip install -e "praq-jax[dev]"    # the library, editable
```

Optional, each needed by one backend or one check:

| Install | Enables |
|---|---|
| `pip install ortools` | `--maxflow`, and the `maxflow` check |
| `pip install gurobipy` + a licence | `--gurobi` |
| `pip install -r docs/requirements.txt` | `make -C docs html`, and the `docs` check |
| `pip install nvidia-ml-py` | the device column of a memory run |

A backend whose dependency is absent reports `unavailable` and does not fail
the run, so none of these blocks anything else.

### CUDA

Needed only for `--praq-cuda` and `--cupdlpx`. `CUDA_ARCH` has no default
anywhere but `env.sh`, so set it there or in `env.local.sh` for the card you
are targeting.

```sh
scripts/build_cuda.sh praq-cuda    # -> $PRAQ_BIN_DIR/praq_solve_file, libpraq.so
```

cuPDLPx is reached through its Python package rather than a binary, and has
to be cloned first — `$CUPDLPX_DIR` says where:

```sh
git clone https://github.com/MIT-Lu-Lab/cuPDLPx "$CUPDLPX_DIR"
scripts/build_cuda.sh cupdlpx      # needs cmake as well as nvcc
```

**Build for the card you will run on.** Both builds are architecture-specific
and `CUDA_ARCH` selects it, so a build for one GPU does not run on another.
That matters most under a scheduler, where `scripts/run_benchmark.sbatch`
pins `--gres=gpu:h100:1` so that passes submitted days apart stay comparable:
build on the same class of card. The sbatch scripts detect `CUDA_ARCH` from
the GPU they land on and rebuild `praq-cuda` themselves, but the cuPDLPx
wheel is installed once, so build it in a job on the target card rather than
on a login node.

### Check the install

```sh
scripts/run_checks.sh
```

`N passed, 0 failed, 0 skipped` means everything above is in place. A skip
names what is missing.

## Experiments

Five, each reproducible from a clean checkout. The first is seconds; the last
two want a GPU and a scheduler. Nothing is stored: every instance comes from a
seed and a size, so a pass regenerates exactly what the last one measured.

### 1. Do the implementations agree?

The test, not the benchmark — it asks only whether two solvers return the same
objective, and is meant to be fast enough to run before every commit.

```sh
# random instances; the pair that needs neither a licence nor a GPU
$PRAQ_PYTHON scripts/check_correctness.py --highs --praq-jax

# the benchmark's own buckets, capped by size and by count
$PRAQ_PYTHON scripts/check_correctness.py --maxflow --highs --families hard-medium
```

The first implementation named is the reference and every other is compared
against it; exit status is 0 only if all agree. Prefer `--maxflow` as the
reference wherever `ortools` is installed: it computes the roof dual as a
maximum flow in integer arithmetic, so it stops at no tolerance, and it shares
no code with `baselines/model.py` — agreement with the LP solvers is therefore
evidence rather than a tautology.

Both caps matter with `--families`. The largest instances are in the
collection to be timed rather than checked, and a bucket holds hundreds of
instances per size, so an uncapped pass over one is half an hour against the
four minutes it is capped. `--max-edges 0` and `--limit 0` lift them.

### 2. How does each solver scale?

The benchmark pass: every instance of the ladder, one CSV per run, rewritten
after every instance so an interrupted job keeps what it measured.

```sh
# a slice, locally
$PRAQ_PYTHON scripts/run_benchmark.py --praq-jax --families hard-medium

# one full pass per backend, on a batch GPU
sbatch scripts/run_benchmark.sbatch --praq-cuda
sbatch scripts/run_benchmark.sbatch --cupdlpx

# a CPU baseline needs no GPU, and wants a longer wall clock and more memory
sbatch --partition=<cpu-partition> --gres=none --time=2-00:00:00 \
       --mem=400G --cpus-per-task=32 --export=ALL,GUROBI_THREADS=32 \
       scripts/run_benchmark.sbatch --gurobi
```

`GUROBI_THREADS` is worth setting on every scheduled Gurobi run. Gurobi sizes
its thread pool from the machine rather than from the job's cgroup, so
unpinned it oversubscribes the allocation and the time it reports is not the
time the allocation bought.

Output: `$PRAQ_RESULTS_DIR/benchmark-<backends>-<timestamp>.csv`.

One backend per job is the intended shape — praq-cuda is remeasured after
every change to it while the baselines are fixed — and the job pins
`--gres=gpu:h100:1` rather than taking what is free, because passes submitted
days apart are read against each other. Submitting two at once needs a
dependency, not a second partition; the script's header has the worked
examples and is the copy to trust.

### 3. How much memory does each GPU solver need?

A **separate run**: sampling costs the solve it watches, so its timings are
not the project's timings and its CSV says so in the name.

```sh
$PRAQ_PYTHON scripts/run_benchmark.py --praq-cuda --cupdlpx --memory
```

Output: `$PRAQ_RESULTS_DIR/benchmark-memory-*.csv`. A row carrying a peak is
used by the memory table and by no other figure, and a row without one never
reaches it, so the two kinds of run cannot be mixed up by hand. Only
`--praq-cuda` and `--cupdlpx` measure themselves; the table reports peak
*device* memory, which is what decides whether an instance fits on the card.

### 4. Draw the figures

```sh
$PRAQ_PYTHON scripts/plot_results.py $PRAQ_RESULTS_DIR/benchmark-*.csv
$PRAQ_PYTHON scripts/plot_results.py $PRAQ_RESULTS_DIR/benchmark-memory-*.csv \
       --figures peak-memory
$PRAQ_PYTHON scripts/plot_results.py $PRAQ_RESULTS_DIR/benchmark-*.csv --format pdf
```

Name as many CSVs as you like — the GPU baselines and praq-jax are usually run
on different machines — and name them **first**, ahead of the list-valued
flags. Output goes to a `figs/` beside the CSVs, as PNG and Markdown; `--format
pdf` is for a figure going into the paper and `--csv-tables` for a table
something else has to read back.

What it draws: an ECDF of solve time, iterations and seconds-per-iteration
against instance size, the size-bucketed tables of both, and how much of the
agreement budget each backend's own bracket spends — a duality gap for the
first-order solvers, rounding slack for `--maxflow`.

Three things it does that a plotting script would not. It cross-checks the
objectives first, with the same test `check_correctness.py` uses, and drops
every instance the backends disagree about — a timing is worth reading only
once the answer behind it is right. It draws each figure over the instances
*every* backend in it attempted, and prints that count, since a backend that
skipped the hard half would otherwise plot as the fast one. And it keeps
`unavailable` apart from `time_limit`: the first never ran and is not in the
population, the second is a censored observation, in the denominator and never
contributing a step, so a backend that timed out on a third of the collection
plateaus at two thirds.

### 5. Tune the restart schedule

```sh
$PRAQ_PYTHON scripts/tune_praq.py --buckets hard-medium --configs 50  # a look
sbatch scripts/tune_praq.sbatch                                       # the real thing
```

Output: `$PRAQ_RESULTS_DIR/tuning/tune-<timestamp>.csv`, rewritten after every
configuration, with a column per instance so a score can be recomputed under
other weights without re-running.

Optuna's multivariate TPE searches the restart ratios, the primal-weight
smoothing and the check-interval rule, scoring in iterations rather than
seconds so the result does not depend on the GPU. Every configuration is
scored on its own draw from per-size seed pools disjoint from the benchmark's,
so none can be selected for suiting particular instances. The search screens
broadly and confirms deeply: a shortlist is re-scored on a fixed hold-out, and
only the best few are scored on the benchmark itself.

**The winner is not applied automatically.** Its values belong in two places
that must move together: `Parameters.with_auto_step` in
`praq-jax/praq/solver.py` and the defaults at the top of
`praq-cuda/src/solver.cuh`.

### Backends, and cleaning up

Every run above takes `--praq-jax`, `--praq-cuda`, `--highs`, `--gurobi`,
`--cupdlpx` and `--maxflow`. A backend that cannot run here reports
`unavailable` and does not fail the run; `--highs` needs only scipy, so it is
the one that always works.

```sh
scripts/clean.sh [--results] [--figs] [--all] [-n]
```

Build products, Python caches and exported MPS go by default. Results and
figures each need their own flag: results cannot be recreated at all, and
figures, though one `plot_results.py` away, are usually what you are in the
middle of reading. `-n` previews.

## Layout

| Directory | What it is |
|---|---|
| `praq-jax/` | **the library.** The JAX implementation, and the reference for everything else. |
| `praq-cuda/` | **the library.** The CUDA port, sparse; a file CLI the harness drives as a subprocess. |
| `baselines/` | *not* the library. HiGHS, Gurobi, cuPDLPx and an OR-Tools max-flow oracle; the only place the standard linearization is built explicitly. |
| `harness/` | *not* the library. Instance readers, the backend registry, result records, the figures, the memory sampler. |
| `scripts/` | the entry points listed below. |

`praq-jax/` and `praq-cuda/` never import from `harness/` or `baselines/`. The
dependency runs one way, so the comparison machinery can be thrown away without
touching the solvers.

## Checks

`scripts/run_checks.sh` is the single home for what the checks are, and
`.github/workflows/ci.yml` calls the same script, so a green local run means a
green CI run. Run it before opening a PR.

Read the summary, not just the exit code: a **skipped** check did not run.
`cuda` skips without `nvcc`, `docs` without sphinx, `maxflow` without `ortools`,
`plots` without matplotlib, `shell`'s lint without shellcheck. Only `N passed, 0 failed, 0 skipped` is a
complete pass.

GitHub Actions is currently **disabled** on this repository to stay inside the
monthly minute budget, so the local run is the only run.

## Documentation

```sh
$PRAQ_PYTHON -m pip install -r docs/requirements.txt
make -C docs html        # or: python -m sphinx -b html -W docs docs/_build/html
```

Built with `-W`, so a broken cross-reference or an unparseable docstring fails
the build. CI (`.github/workflows/ci.yml`) runs that, the `praq-jax` tests, a
`--highs --praq-jax` correctness smoke test, and `shellcheck` on every shell
script. It deliberately does **not** exercise `--gurobi`, `--cupdlpx` or
`--praq-cuda`: those are licence- or GPU-gated and report `unavailable` rather
than failing, so a job reaching for them would go green without testing
anything. `--maxflow` is the exception, because the `maxflow` check asks for
`ortools` itself and reports a skip when it is absent instead of passing
vacuously.

## Instances

The benchmark is generated rather than stored: `random_hard` in
`harness/instances.py` builds instances that are hard for the *relaxation*
rather than for the binary problem -- log-normal edge weights with a wide
spread, a quarter of them negative, and diagonal entries a tenth of the incident
weight -- and `HARD_BUCKETS` fixes three families of them, `hard-medium`
(500 to 5000 nodes), `hard-large` (10k to 100k) and `hard-huge` (200k to 2M),
at about ten edges per node, with counts tapering from 500 instances at the
smallest size to 40 at the largest. Those three are
what `run_benchmark.py` runs by default and what `tune_praq.py` trains on, with
other seeds. On the top rung `praq-cuda` takes half a minute and cuPDLPx over a
minute, on a standard linearization of 60M rows. The two smallest sizes are
under `CHECK_MAX_EDGES`, so the correctness check has them.

That is 2282 instances spanning five orders of magnitude in edge count, which
is what makes it a scaling study rather than a population to draw one ECDF
over. Nothing is stored: a seed and a size are the whole record of an
instance, so a pass regenerates exactly what the last one measured.

## Known gaps

- `praq-cuda` builds the half-edge lift on the instance's own edges, where
  `praq-jax` pads to the complete graph to keep jit shapes static. On a sparse
  instance the two are therefore preconditioned differently: same bounds,
  fewer iterations for the sparse build. Iteration counts between the two are
  comparable only on dense instances.
- cuPDLPx presolves by default, so its iteration counts are on the presolved LP
  while PRAQ's are on the whole one. `SOLVER_FLAGS` in `baselines/cupdlpx_backend.py`
  is the knob; objective values are the same either way.
- `QuboProblem` stores W densely, so `praq-jax` reaches only `hard-medium` and
  the first rung of `hard-large` (`MAX_DENSE_N` in its adapter). Fixing that
  means a sparse instance type in the library, not a bigger constant.
- HiGHS and Gurobi solve to a vertex -- interior point with crossover and dual
  simplex respectively -- while the first-order backends stop on a relative
  duality gap. Fine for the correctness check, which
  compares objective values; for timings, cuPDLPx is the like-for-like comparison.
- The right baseline for this problem is neither of those: roof duality reduces
  to a max-flow, and a QPBO code will beat any LP solver on it. That baseline is
  now `--maxflow`, and it does below about 10k edges — 0.0026 s against
  cuPDLPx's 0.052 s and Gurobi's 0.10 s on the smallest decade, though it is
  overtaken above that — so the LP baselines should be read as measuring
  what the reformulation does to an LP solver, which is the paper's question,
  not PRAQ's standing against the best algorithm for this particular bound.
- Peak memory is sampled, not exact. Five of the six backends run inside the
  harness's own process, whose `ru_maxrss` is one high-water mark for the whole
  run and cannot be split per instance, so the sampler reads the footprint every
  few milliseconds and keeps the largest — which misses any peak shorter than
  that interval. The table therefore drops instances solved faster than
  `MEMORY_MIN_SOLVE_SECONDS`, and the host column is not in it at all:
  praq-cuda's is a child process that read its instance from a file, while
  cuPDLPx's is the harness's process and includes the LP built for it, so the
  two are not the same measurement. Both columns are in the CSV.
- Every figure is drawn over the instances all of its backends attempted, so
  naming a run in progress alongside a complete one shrinks the figure to the
  partial run's coverage. It is the only comparable choice — a curve over a
  different denominator than its neighbour is not a comparison — but the way to
  look at a partial run is to plot it on its own, or to cut the others down to
  it with the size ranges. The `--cupdlpx` comparisons are the ones this bites:
  they are the point of two of the figures and cuPDLPx is the slowest backend to
  finish a pass.
- `--maxflow` needs int64 capacities, so a float instance is solved exactly only
  after rounding. The slack is reported per record and allowed for by
  `check_correctness.py`; it is exactly zero on every integer-weighted instance,
  and on the generated families it stays between four and eight orders of
  magnitude inside `AGREEMENT_TOLERANCE`, worst at the top rung. Generating
  integer-weighted instances to remove it outright was considered and rejected:
  it would change what the benchmark measures to improve an error term that is
  already nowhere near being the limiting factor.
