# PRAQ Jax

PRAQ (Projected Restarted Algorithm for QUBO) is the sibling of
[PRIA](../../pria-jax) for quadratic unconstrained binary optimization. Given a
symmetric $W$, QUBO is

$$\min_{x \in \\{0,1\\}^n} \; x^\top W x. \tag{Q}$$

PRAQ solves the LP relaxation of its standard linearization — equivalently, the
**roof dual** of (Q) — by restarted reflected Halpern PDHG on a bilinear
reformulation. The derivation is in the paper named in the repository root README.

## The reformulation

Writing $d_i = W_{ii}$, $w_e = 2|W_{ij}|$ on each edge $e = \\{i,j\\}$, and $S$
for the map that copies a node value onto its two half-edges, PRAQ solves

$$\min_{x \in [0,1]^n} \; \max_{Y \in \mathcal V} \; \langle d, x\rangle + \langle Y, \mathbf 1_+ - Sx\rangle$$

where $\mathcal V$ is a product of one-dimensional per-edge sets: a weighted
2-simplex $\\{Y \geq 0 : Y_1 + Y_2 = w_e\\}$ on each edge with $W_{ij} < 0$, and
a segment $\\{(\mu,\mu) : -w_e \leq \mu \leq 0\\}$ on each edge with
$W_{ij} > 0$. Both projections are closed form on two elements — **no sort**,
unlike PRIA's simplex blocks.

Three consequences drive the implementation:

- **No matrix-vector products.** An iteration is elementwise arithmetic plus one
  gather ($Sx$) and one scatter ($S^\top Y$).
- **$\\|A\\|_2 = 1$ exactly**, for every graph and every instance, so
  $\eta = 0.999$ is admissible for every primal weight — no line search, no
  power iteration, no dependence on the maximum degree. PRIA's coupling has
  $\sqrt 2$ and admits only $\eta < 1/\sqrt 2$.
- **Exact bounds at every iterate.** Both iterates stay feasible, so
  `relaxation_bounds` brackets the optimum from the first step, with no
  feasibility slack to account for.

The half-edge lift is degree-preconditioned PDHG in $x$-space; the primal is
stored as $x \in \mathbb R^n$ rather than the redundant $2m$ copies, and the
canonical norm carries $\mathrm{diag}(\deg)$ on its primal block accordingly.

## Installation

Activate the Python environment you would like to install to, then from this
directory run `pip install -e .`

If you plan to use the project with a GPU, install a CUDA-enabled Jax manually:
`pip uninstall jax jaxlib` then `pip install -U "jax[cuda13]"`.

To test the installation run `python tests/test_solver.py`. Each test file is
runnable as a script.

## Usage

```python
from jax import config, jit, block_until_ready

# enable x64 when you need better than ~1e-4 relative accuracy
config.update('jax_enable_x64', True)

import jax.numpy as jnp
from praq.prelude import *

instance = QuboProblem.new(jnp.array(W))               # W symmetric, (n, n)
params = Parameters.with_auto_step(max_iter=None, tolerance=1e-8)

solve = jit(solve_qubo_problem).trace(instance, params).lower().compile()
primal, dual, iterations = block_until_ready(solve(instance, params))

lower, upper = relaxation_bounds(instance, primal, dual)   # brackets the optimum
fixed, values = persistent_variables(primal)               # variables you may fix
```

Batches of instances of the same size solve under `vmap`, using
`QuboProblem.stack` and `Parameters.stack` as in PRIA.

## What the result is, and is not

QUBO is NP-hard, so the relaxation is a **bound and a variable-fixing engine**,
not a solver:

- `relaxation_bounds` gives a valid lower bound on (Q).
- `persistent_variables` returns the coordinates that came out at 0 or 1. By
  persistency, each takes that value in some optimal binary solution, so they
  may be fixed and removed. On sparse instances this is most of them; on dense
  random instances the roof dual is typically fractional everywhere and fixes
  nothing.
- The relaxation is **exact** when the signed graph
  $\sigma_e = -\mathrm{sign}(W_{ij})$ is balanced — equivalently when $W$ becomes
  submodular after complementing some subset of the variables. Then the
  relaxation is a min-cut and rounding solves (Q) outright.

Half-integrality is a property of the relaxation's vertices. A first-order
method returns a point of the optimal face, which may be more fractional than
that; this is safe but conservative for persistency, and never wrong.

## Layout

| Module | Contents |
|---|---|
| `qubo_problem.py` | `QuboProblem`, instance generation, `relaxation_value` |
| `half_edge.py` | the group-indexed structure; `gather` / `scatter` |
| `projection.py` | the two closed-form projections |
| `halpern_operator.py` | the canonical norm and its `omega` minimizer |
| `solver.py` | `Parameters`, the PDHG step, restarts, Halpern, bounds |
| `brute_force.py` | exhaustive reference, for tests on tiny instances |

## The right baseline is max-flow, not an LP solver

The relaxation PRAQ solves is the roof dual, and roof duality is a *combinatorial*
polynomial-time problem: Hammer–Hansen–Simeone solve it by reduction to a bipartite
maximum flow, and QPBO implementations do exactly that. So a max-flow code is the
correct baseline for this problem and will beat any first-order method on it.

`../scripts/run_benchmark.py` compares against HiGHS, Gurobi and cuPDLPx, which is
informative only about the LP formulation — HiGHS must carry the Θ(n²) product variables that the max-flow
formulation never forms, so the speedups there measure the reformulation's effect on
an LP solver, not PRAQ's standing against the best available algorithm. Read them
that way.

PRAQ's interest is as a second instantiation of the reformulation recipe, and as the
formulation-comparison data point in the paper: for integral `W` the sharpness
constant is `gamma = O(||W||_max * n^{5/2})`, proved — both margins are at least 1/2,
because every vertex of the relaxation and of its dual value LP is half-integral.
The JRP lift admits vertices of denominator 34 with integer data, so it has no such
bound. What the saddle-point
form retains over max-flow is that it survives additions — side constraints on `x`,
higher-order terms — under which roof duality's max-flow equivalence, special to the
quadratic *unconstrained* case, does not.

## Not yet done

Hyperparameters other than the step size are PRIA's JRP-tuned values, taken as
starting points and **not** tuned for QUBO. There is no
crossover to a vertex, and no comparison against a QPBO/max-flow implementation —
which is the comparison that would actually be informative.
