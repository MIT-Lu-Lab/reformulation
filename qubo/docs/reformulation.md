# The reformulation

Given a symmetric $W$, QUBO is

$$\min_{x \in \{0,1\}^n} x^\top W x.$$

## The standard linearization

Writing $d_i = W_{ii}$ and $w_e = 2 W_{ij}$ on each edge $e = \{i,j\}$, the
textbook linearization introduces a product variable $z_e$ per edge:

$$
\begin{aligned}
\min\quad & \langle d, x\rangle + \sum_e w_e z_e \\
\text{s.t.}\quad & x_i + x_j - z_e \le 1 && w_e > 0 \\
& z_e \le x_i,\; z_e \le x_j && w_e < 0 \\
& 0 \le x \le 1,\ 0 \le z \le 1.
\end{aligned}
$$

Three details are easy to get wrong and each changes the optimal value: the
diagonal enters linearly through $x_i^2 = x_i$ rather than through a $z_{ii}$;
the covering block needs $z_e \ge 0$; and $x$ must be boxed explicitly. This is
the model the LP baselines solve, built once in
{py:func}`baselines.model.build`.

PRAQ never forms it.

## The half-edge lift

Applying the max-selection recipe in $x$-space leaves a coupling operator of
incidence type, with $\|A\|_2 = \Theta(\sqrt{\Delta})$ for maximum degree
$\Delta$. The fix is to duplicate each variable once per group it belongs to.

Let $\mathcal H = \{(e,k) : e \in E,\ k \in \{1,2\}\}$ be the half-edge space and
$S$ the map copying a node value onto its two half-edges. PRAQ solves

$$\min_{x \in [0,1]^n}\ \max_{Y \in \mathcal V}\ \langle d, x\rangle + \langle Y, \mathbf 1_+ - Sx\rangle$$

where $\mathcal V$ is a product of per-edge sets: a weighted 2-simplex
$\{Y \ge 0 : Y_1 + Y_2 = w_e\}$ on each edge with $W_{ij} < 0$, and a segment
$\{(\mu,\mu) : -w_e \le \mu \le 0\}$ on each edge with $W_{ij} > 0$.

## Three consequences

**No matrix-vector products.** An iteration is elementwise arithmetic plus one
gather ($Sx$) and one scatter ($S^\top Y$).

**$\|A\|_2 = 1$ exactly**, for every graph and every instance, so $\eta = 0.999$
is admissible for every primal weight — no line search, no power iteration, no
dependence on maximum degree. Both projections are closed form on two elements,
so there is no sort in the inner loop either.

**Exact bounds at every iterate.** Both iterates stay feasible, so
{py:func}`praq.solver.relaxation_bounds` brackets the optimum from the first
step, with no feasibility slack to account for.

The lift is degree-preconditioned PDHG in $x$-space: the primal is stored as
$x \in \mathbb R^n$ rather than the redundant $2m$ copies, and the canonical norm
carries $\mathrm{diag}(\deg)$ on its primal block accordingly.

## Honest accounting

The primal grows from $n$ to $2m$, the same order as the $z$ block that was
eliminated — so the reformulation buys elementwise iterations and a fixed
coupling norm, but not fewer variables. And the consensus projection's per-node
averaging is an implicit $1/\deg$ preconditioner, so the identity coupling is
partly a change of metric rather than free.

## The right baseline is max-flow

Roof duality is a combinatorial polynomial-time problem: Hammer–Hansen–Simeone
solve it by reduction to maximum flow, and QPBO implementations do exactly that.
A max-flow code is the correct baseline and will beat any first-order method
here. {py:mod}`baselines.maxflow_backend` is that baseline, on OR-Tools, and its
docstring derives the network it builds.

It earns its place on exactness rather than on speed. The flow value *equals*
the LP relaxation's optimum, so it is the same number every other backend
reports — reached with no tolerance, no iteration count and no floating-point
arithmetic beyond scaling the capacities to integers. That makes it the
reference the correctness check compares the rest against, since every other
backend here stops on a tolerance of its own.

The LP baselines measure something narrower and deliberate: what the
reformulation does to an LP solver, which must carry the $\Theta(n^2)$ product
variables that the max-flow formulation never forms. Read them that way.

What the saddle-point form retains over max-flow is that it survives additions —
side constraints on $x$, higher-order terms — under which roof duality's
max-flow equivalence, special to the quadratic *unconstrained* case, does not.
