# jrp

PRIA (Projected Restarted Inventory Algorithm) and the baselines it is measured
against.

PRIA solves the LP relaxation of the joint replenishment problem by restarted
reflected Halpern PDHG on a saddle reformulation in which the order variables
have been eliminated and the coupling operator is the identity.

Two halves, and the boundary between them matters:

- **The library**: `pria-jax` and `pria-cuda`. Installable, self-contained, and
  dependent on nothing else here.
- **Not the library**: `harness` and `baselines`. The experiment machinery, which
  knows about the library but never the other way round.

The dependency runs one way, so the comparison machinery could be deleted without
touching the solvers.

```{toctree}
:maxdepth: 2
:caption: Contents

reformulation
harness
api/pria
api/harness
```

## Quick start

```sh
source env.sh
$PRIA_PYTHON -m pip install -r requirements.txt
$PRIA_PYTHON -m pip install -e "pria-jax[dev]"

$PRIA_PYTHON scripts/check_correctness.py --highs --pria-jax
$PRIA_PYTHON scripts/run_benchmark.py --pria-jax --highs --families jrp-medium
```

## What the result is

The LP's optimal value, with a certificate.

- {py:func}`pria.solver.relaxation_bounds` brackets it, and both bounds are
  valid at every iterate rather than only at convergence — every projection
  lands on a feasible point.
- {py:func}`pria.solver.solve_jrp_problem` returns that bracket alongside the
  solution, which is what the harness records as `lower` and `upper`.
- Unlike the QUBO roof dual, this LP **is** the object of study, not a bound on
  an integer problem. There is no persistency to exploit and no rounding step
  here.

## Known gaps

- cuPDLPx presolves by default, so its iteration counts are on the presolved LP
  while PRIA's are on the whole one. `SOLVER_PARAMS` in
  {py:mod}`baselines.cupdlpx_backend` is the knob; the objective is the same
  either way.
- HiGHS and Gurobi solve to a vertex — interior point with crossover, and
  barrier with crossover respectively — while the first-order backends stop on a
  relative duality gap. Objective values compare; iteration counts do not.
- `pria-jax` compiles per *shape*, so the first instance of every rung pays a
  compile. It is recorded as `setup_seconds` and kept out of the timings, but it
  does mean a pass over many shapes spends real time compiling.
- float32 iterates stop improving near a relative gap of 1e-6 beyond the medium
  bucket, which is why {py:class}`harness.config.Tolerances` asks for 1e-6 and
  not the 1e-8 its QUBO sibling uses. Tighter tolerances need float64.
- An odd period count is padded to even. The padded LP has the same optimum, but
  the padded period's dual multiplier is sliced off the returned arrays, so
  recomputing the lower bound from them afterwards can give a weaker number than
  the one the solver converged against. The solver reports its own; see
  {py:func}`pria.solver.relaxation_bounds`.
