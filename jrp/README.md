# JRP experiments

PRIA (Projected Restarted Inventory Algorithm) and the baselines it is measured
against, in one repository.

PRIA solves the LP relaxation of the joint replenishment problem by restarted
reflected Halpern PDHG on a saddle reformulation: the order variables `y` and
`z` are eliminated analytically, and what is left is a min-max over simplices
whose coupling operator is the identity, so an iteration is projections and
elementwise arithmetic with no matrix in it. The derivation is in
`docs/reformulation.md`.

## Installation

```sh
cd jrp                                 # from the repository root
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

`PRIA_PYTHON` must name the interpreter everything below runs under. A
virtualenv at the repository root is the usual choice; point `PRIA_PYTHON` at
it in `env.local.sh`.

```sh
$PRIA_PYTHON -m pip install -r requirements.txt   # harness and baselines
$PRIA_PYTHON -m pip install -e "pria-jax[dev]"    # the library, editable
```

For a GPU, install a CUDA-enabled jax over the CPU wheel pip resolves by
default: `pip install -U "jax[cuda13]"`.

Optional, each needed by one backend or one check:

| Install | Enables |
|---|---|
| `pip install gurobipy` + a licence | `--gurobi` |
| `pip install -r docs/requirements.txt` | `make -C docs html`, and the `docs` check |
| `pip install nvidia-ml-py` | the device column of a memory run |

A backend whose dependency is absent reports `unavailable` and does not fail
the run, so none of these blocks anything else.

### CUDA

Needed only for `--pria-cuda` and `--cupdlpx`. `CUDA_ARCH` has no default
anywhere but `env.sh`, so set it there or in `env.local.sh` for the card you
are targeting.

```sh
scripts/build_cuda.sh pria-cuda    # -> $PRIA_BIN_DIR/pria_solve_file, libpria.so
```

`pria_solve_file` is the binary the harness drives; `libpria.so` is what
`tune_pria.py` loads through ctypes.

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
the GPU they land on and rebuild `pria-cuda` themselves, but the cuPDLPx
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
seed and a shape, so a pass regenerates exactly what the last one measured.

### 1. Do the implementations agree?

The test, not the benchmark — it asks only whether two solvers return the same
objective, and is meant to be fast enough to run before every commit.

```sh
# random instances; the pair that needs neither a licence nor a GPU
$PRIA_PYTHON scripts/check_correctness.py --highs --pria-jax

# the benchmark's own buckets, capped by size and by count
$PRIA_PYTHON scripts/check_correctness.py --highs --pria-cuda --families jrp-medium
```

The first implementation named is the reference and every other is compared
against it; exit status is 0 only if all agree. HiGHS is the reference to
prefer: it needs only scipy, it stops at an optimal basis rather than at a
gap, and it goes through `baselines/model.py`, which is the only place the LP
is built explicitly.

Both caps matter with `--families`: the largest instances are in the
collection to be timed rather than checked, and a bucket holds hundreds of
instances per shape. `--max-entries 0` and `--limit 0` lift them.

### 2. How does each solver scale?

The benchmark pass: every instance of the ladder, one CSV per run, rewritten
after every instance so an interrupted job keeps what it measured.

```sh
# a slice, locally
$PRIA_PYTHON scripts/run_benchmark.py --pria-jax --families jrp-medium

# one full pass per backend, on a batch GPU
sbatch scripts/run_benchmark.sbatch --pria-cuda
sbatch scripts/run_benchmark.sbatch --cupdlpx

# a CPU baseline needs no GPU, and wants a longer wall clock and more memory
sbatch --partition=<cpu-partition> --gres=none --time=2-00:00:00 \
       --mem=400G --cpus-per-task=32 --export=ALL,GUROBI_THREADS=32 \
       scripts/run_benchmark.sbatch --gurobi
```

`GUROBI_THREADS` is worth setting on every scheduled Gurobi run. Gurobi sizes
its thread pool from the machine rather than from the job's cgroup, so
unpinned it oversubscribes the allocation and the time it reports is not the
time the allocation bought. The memory is for the explicit LP: the largest
instances have tens of millions of columns and five nonzeros in each.

Output: `$PRIA_RESULTS_DIR/benchmark-<backends>-<timestamp>.csv`.

One backend per job is the intended shape — pria-cuda is remeasured after
every change to it while the baselines are fixed — and the job pins
`--gres=gpu:h100:1` rather than taking what is free, because passes submitted
days apart are read against each other. The cost of splitting is that each job
generates its own instances, which for this problem is a real share of the
pass: a JRP instance is a dense `N x T x T` array of holding costs. The
script's header has the worked examples and is the copy to trust.

### 3. How much memory does each GPU solver need?

A **separate run**: sampling costs the solve it watches, so its timings are
not the project's timings and its CSV says so in the name.

```sh
$PRIA_PYTHON scripts/run_benchmark.py --pria-cuda --cupdlpx --memory
```

Output: `$PRIA_RESULTS_DIR/benchmark-memory-*.csv`. A row carrying a peak is
used by the memory table and by no other figure, and a row without one never
reaches it, so the two kinds of run cannot be mixed up by hand. Only
`--pria-cuda` and `--cupdlpx` measure themselves; the table reports peak
*device* memory, which is what decides whether an instance fits on the card.

### 4. Draw the figures

```sh
$PRIA_PYTHON scripts/plot_results.py $PRIA_RESULTS_DIR/benchmark-*.csv
$PRIA_PYTHON scripts/plot_results.py $PRIA_RESULTS_DIR/benchmark-memory-*.csv \
       --figures peak-memory
$PRIA_PYTHON scripts/plot_results.py $PRIA_RESULTS_DIR/benchmark-*.csv --format pdf
```

Name as many CSVs as you like — the GPU baselines and pria-jax are usually run
on different machines — and name them **first**, ahead of the list-valued
flags. Output goes to a `figs/` beside the CSVs, as PNG and Markdown;
`--format pdf` is for a figure going into the paper and `--csv-tables` for a
table something else has to read back.

What it draws: an ECDF of solve time, iterations and seconds-per-iteration
against instance size, the size-bucketed tables of both, and how much of the
agreement budget each backend's own duality gap spends. `--size-axis` picks
which of `entries`, `items` and `periods` the figures plot against.

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
$PRIA_PYTHON scripts/tune_pria.py --buckets jrp-medium --configs 50  # a look
sbatch scripts/tune_pria.sbatch                                      # the real thing
```

Output: `$PRIA_RESULTS_DIR/tuning/tune-<timestamp>.csv`, rewritten after every
configuration, with a column per instance so a score can be recomputed under
other weights without re-running.

Optuna's multivariate TPE searches the restart ratios, the primal-weight
smoothing and the check interval, scoring in iterations rather than seconds so
the result does not depend on the GPU. Every configuration is scored on its
own draw from per-shape seed pools disjoint from the benchmark's, so none can
be selected for suiting particular instances. The search screens broadly and
confirms deeply: a shortlist is re-scored on a fixed hold-out, and only the
best few are scored on the benchmark itself.

**The winner is not applied automatically.** Its values belong in two places
that must move together: `Parameters.with_auto_step` in
`pria-jax/pria/solver.py` and `pria_params_auto` in `pria-cuda/src/pria.cu`.

### Backends, and cleaning up

Every run above takes `--pria-jax`, `--pria-cuda`, `--highs`, `--gurobi` and
`--cupdlpx`. A backend that cannot run here reports `unavailable` and does not
fail the run; `--highs` needs only scipy, so it is the one that always works.

```sh
scripts/clean.sh [--results] [--figs] [--all] [-n]
```

Build products, Python caches, exported MPS and anything a killed pria-cuda
run left staged go by default. Results and figures each need their own flag:
results cannot be recreated at all, and figures, though one `plot_results.py`
away, are usually what you are in the middle of reading. `-n` previews.

## Layout

| Directory | What it is |
|---|---|
| `pria-jax/` | **the library.** The JAX implementation, and the reference for everything else. |
| `pria-cuda/` | **the library.** The CUDA port; a file CLI the harness drives as a subprocess, and a shared library the tuner loads. |
| `baselines/` | *not* the library. HiGHS, Gurobi and cuPDLPx; the only place the LP is built explicitly. |
| `harness/` | *not* the library. Instance generation, the backend registry, result records, the figures, the memory sampler. |
| `scripts/` | the entry points listed below. |

`pria-jax/` and `pria-cuda/` never import from `harness/` or `baselines/`. The
dependency runs one way, so the comparison machinery can be thrown away without
touching the solvers.

## Checks

`scripts/run_checks.sh` is the single home for what the checks are, and
`.github/workflows/ci.yml` calls the same script, so a green local run means a
green CI run. Run it before opening a PR.

Read the summary, not just the exit code: a **skipped** check did not run.
`cuda` skips without `nvcc`, `docs` without sphinx, `plots` without matplotlib,
`lint` without ruff, `shell`'s lint without shellcheck. Only
`N passed, 0 failed, 0 skipped` is a complete pass.

`lint` is ruff over every Python file. `ruff.toml` at the root configures it,
and `pria-jax/ruff.toml` extends that for the library, which gets more room
per line and keeps `l` as the split point of a two-segment projection.

Two are worth naming. `model` rebuilds the LP the slow, obvious way — dense,
nested loops, every index spelled out — and checks that its optimum matches
the sparse builder every baseline uses; `baselines/model.py` is fancy-indexed
over a flattened upper triangle, which is exactly the sort of code that is
wrong by an index and still runs, and a shared builder would make the
agreement a tautology. `tuner` covers the half of `tune_pria.py` that does not
need a GPU — which shapes are drawn, how they are weighted, how an earlier
run's CSV reads back — because the rest of it cannot be checked until a batch
job is an hour in.

GitHub Actions is currently **disabled** on this repository to stay inside the
monthly minute budget, so the local run is the only run.

## Documentation

```sh
$PRIA_PYTHON -m pip install -r docs/requirements.txt
make -C docs html        # or: python -m sphinx -b html -W docs docs/_build/html
```

Built with `-W`, so a broken cross-reference or an unparseable docstring fails
the build.

## Instances

The benchmark is generated rather than stored: `random_jrp` in
`harness/instances.py` draws holding costs that rise with the wait and order
costs on the same scale as the joint order cost, which is the draw the
difficulty study found hardest — the LP's fractionality peaks when `K0` is
comparable with the `K_i`, a large one making joint ordering obviously right
and a small one decoupling the items into `T` independent lot-sizing problems.

`LADDER` fixes three families: `jrp-medium` (10×25 up to 50×100),
`jrp-large` (75×150 up to 200×300) and `jrp-huge` (250×375 up to 350×450),
several seeds at each shape and fewer of them the larger the shape gets. The
span is four orders of magnitude in x variables, 3.3k to 35.5M, which is what
makes it a scaling study rather than a population to draw one ECDF over.
`harness/instances.py` is where the counts live.

Both dimensions climb together along the ladder. They move different costs —
`periods` sets the size of every simplex the solver projects onto and grows the
instance quadratically, while `items` grows it linearly and adds independent
work — so a ladder that raised one alone would be measuring a shape the problem
is never posed in.

Nothing is stored: a seed and a shape are the whole record of an instance, so a
pass regenerates exactly what the last one measured. A run enumerates its
instances as `Spec` objects, which carry a name and a size but no arrays, and
builds each one only when its turn comes — an instance is `8 N T²` bytes, 570 MB
at the top rung, so materialising a pass before solving any of it is hundreds
of gigabytes.

## Known gaps

- **The ladder has not been run in full.** The pipeline has been exercised end
  to end on an H100 — the CUDA suite, a `--pria-cuda --highs` pass, the
  figures, and the tuner through `libpria.so` — but only over the smallest
  rungs. The other numbers in circulation are `pria-cuda`'s own, from
  `pria_bench` on an L40S at tolerance 1e-4 in float32 (27 ms at 30×75, 1.30 s
  at 200×300, 9.09 s at 350×450); see `pria-cuda/README.md`. Those are not
  this harness's ladder, tolerance or precision, and nothing should be read as
  measured here until `run_benchmark.py` has produced it.
- `--mem` in `scripts/run_benchmark.sbatch` is an estimate from the LP's
  structure, not a measurement. A memory run is how to replace it.
- cuPDLPx presolves by default, so its iteration counts are on the presolved LP
  while PRIA's are on the whole one. `SOLVER_PARAMS` in
  `baselines/cupdlpx_backend.py` is the knob; the objective is the same either
  way.
- HiGHS and Gurobi solve to a vertex — interior point with crossover and
  barrier with crossover respectively — while the first-order backends stop on
  a relative duality gap. Fine for the correctness check, which compares
  objective values; for timings, cuPDLPx is the like-for-like comparison.
- `pria-jax` compiles per shape, so the first instance of every rung pays a
  compile. It is recorded as `setup_seconds` and kept out of the timings, but a
  pass over eleven shapes does spend real time compiling.
- float32 iterates stop improving near a relative gap of 1e-6 beyond the medium
  bucket, which is why the default optimality tolerance is 1e-6 and not the
  1e-8 the QUBO sibling of this repository uses. Tighter needs float64.
- An odd period count is padded to even. The padded LP has the same optimum,
  but the pad period's dual multiplier is sliced off the returned arrays, so
  `relaxation_bounds` recomputed from them afterwards can be weaker than the
  bound the solver converged against. The solver reports its own, and that is
  what the harness records.
- Peak memory is sampled, not exact. A peak shorter than the sampling interval
  is missed entirely, so the table drops instances solved faster than
  `MEMORY_MIN_SOLVE_SECONDS`, and the host column is not in it at all:
  pria-cuda's is a child process that read its instance from a file, while
  cuPDLPx's is the harness's process and includes the LP built for it, so the
  two are not the same measurement. Both columns are in the CSV.
- Every figure is drawn over the instances all of its backends attempted, so
  naming a run in progress alongside a complete one shrinks the figure to the
  partial run's coverage. It is the only comparable choice, but the way to look
  at a partial run is to plot it on its own, or to cut the others down to it
  with the size ranges.
