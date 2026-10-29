# pria-jax

PRIA (Projected Restarted Inventory Algorithm) in JAX: the reference
implementation, and what `pria-cuda` is a port of.

It solves the LP relaxation of the joint replenishment problem

```
minimize    sum_s K0 z_s + sum_{i,s} K_i y_is + sum_{i,t} sum_{s<=t} H_ist x_ist
subject to  sum_{s<=t} x_ist = 1            for all i, t
            x_ist <= y_is                   for all i, s <= t
            x_ist <= z_s                    for all i, s <= t
            x, y, z >= 0
```

by eliminating `y` and `z` analytically and running restarted reflected
Halpern PDHG on the saddle problem that remains,

```
min_{x in X} max_{y in Y, z in Z}  <h, x> + <y, x> + <z, x>
```

where `X` is a product of per-`(i, t)` unit simplices, `Y` of per-`(i, s)`
simplices summing to `K_i`, and `Z` of per-period simplices over
`{(i, t) : t >= s}` summing to `K0`.  Each `x_ist` belongs to exactly one
group of each dual block, so the coupling operator is the identity and one
step is projections and elementwise arithmetic — no matrix-vector product,
and a step size that does not depend on the data.

## Install

```sh
pip install -e ".[dev]"          # numpy and jax; scipy and pytest for the tests
pip install -U "jax[cuda13]"     # for a GPU, over the CPU wheel pip resolves
```

## Use

```python
from jax import config
config.update("jax_enable_x64", True)     # for tolerances tighter than float32

from pria.prelude import (JrpProblem, Parameters, relaxation_bounds,
                          solve_jrp_problem)

instance = JrpProblem.random_instance(seed=0, items=30, time_periods=75)
params = Parameters.with_auto_step(max_iter=None, tolerance=1e-8)

x, y, z, iterations = solve_jrp_problem(instance, params)
lower, upper = relaxation_bounds(instance, x, y, z)
```

`solve_jrp_problem` is `jit`ed, so the first call on a new shape compiles.
`jit(solve_jrp_problem).trace(instance, params).lower().compile()` does that
step explicitly, which is what the benchmark harness does to keep compilation
out of a measured solve.

`relaxation_bounds` is a **certificate**: both iterates are feasible at every
step, so `lower <= optimum <= upper` holds throughout the run and not only at
convergence.  `y` and `z` come back normalised — each `y` row sums to one over
`t >= s`, each `z` group over `{(i, t >= s)}` — so feeding a returned solution
back in through `solve_jrp_problem_warm_start` starts at a fixed point.

## Layout

| Module | What it is |
|---|---|
| `pria/prelude.py` | the public surface; everything else is internals |
| `pria/jrp_problem.py` | the instance, its objective, and the random generator |
| `pria/solver.py` | parameters, the Halpern loop, restarts, the bounds |
| `pria/projection.py` | the simplex projections, closed form and sort-based |
| `pria/upper_triangular.py` | the dense packings of the `s <= t` triangle |
| `pria/halpern_operator.py` | the canonical (proximal-point) metric |
| `pria/convergence_log.py` | optional per-iteration logging, off by default |
| `pria/adversarial.py` | constructions of instances that are hard on purpose |

The package installs on **numpy and jax alone**.  The LP references it is
checked against live outside it, in the experiment repository's `baselines/`,
and `tests/reference.py` is the one-way bridge: tests may import baselines,
the shipped package may not.

## Tests

```sh
python tests/test_solver.py      # each file runs as a script
pytest                           # or all of them at once
```

`tests/test_solver.py` needs the experiment repository around it, for the
HiGHS reference.  The rest are self-contained.
