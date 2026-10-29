# qubo

PRAQ (Projected Restarted Algorithm for QUBO) and the baselines it is measured
against.

PRAQ solves the LP relaxation of a QUBO's standard linearization — equivalently
its **roof dual** — by restarted reflected Halpern PDHG on a bilinear
reformulation, in a half-edge space where the coupling operator is the identity.

Two halves, and the boundary between them matters:

- **The library**: `praq-jax` and `praq-cuda`. Installable, self-contained, and
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
api/praq
api/harness
```

## Quick start

```sh
source env.sh
$PRAQ_PYTHON -m pip install -r requirements.txt
$PRAQ_PYTHON -m pip install -e "praq-jax[dev]"

$PRAQ_PYTHON scripts/check_correctness.py --highs --praq-jax
$PRAQ_PYTHON scripts/run_benchmark.py --praq-jax --highs --families hard-medium
```

## What the result is, and is not

QUBO is NP-hard, so the relaxation is a **bound and a variable-fixing engine**,
not a solver.

- {py:func}`praq.solver.relaxation_bounds` gives a valid lower bound.
- {py:func}`praq.solver.persistent_variables` returns the coordinates that came
  out at 0 or 1. By persistency each takes that value in some optimal binary
  solution, so they may be fixed and removed.
- The relaxation is **exact** when the signed graph $\sigma_e = -\mathrm{sign}(W_{ij})$
  is balanced — equivalently when $W$ becomes submodular after complementing some
  subset of the variables. Then it is a min-cut and rounding solves the QUBO
  outright.

## Known gaps

- `praq-cuda` builds the half-edge lift on the instance's own edges, where
  `praq-jax` pads to the complete graph to keep jit shapes static. On a sparse
  instance the two are therefore preconditioned differently: same bounds,
  fewer iterations for the sparse build. Iteration counts between the two are
  comparable only on dense instances.
- cuPDLPx presolves by default, so its iteration counts are on the presolved LP
  while PRAQ's are on the whole one. `SOLVER_PARAMS` in
  {py:mod}`baselines.cupdlpx_backend` is the knob; the objective is the same
  either way, and turning presolve off costs iterations.
- {py:class}`praq.qubo_problem.QuboProblem` stores $W$ densely, so `praq-jax`
  reaches only `hard-medium` and the first rung of `hard-large`. Fixing that
  needs a sparse instance type, not a bigger constant.
