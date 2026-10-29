# The reformulation

The joint replenishment problem asks when to place orders for $N$ products over
$T$ periods. Each period's demand for each product must be served by an order
placed in that period or earlier; an order for product $i$ in period $s$ costs
$K_i$, any period in which some order is placed costs a further $K_0$, and
carrying product $i$'s period-$t$ demand from an order in period $s \le t$ costs
$H_{ist}$.

## The LP

With $x_{ist}$ the fraction of product $i$'s period-$t$ demand served from
period $s$, $y_{is}$ the indicator that $i$ orders in $s$, and $z_s$ the
indicator that anything does:

$$
\begin{aligned}
\min\quad & \sum_s K_0 z_s + \sum_{i,s} K_i y_{is}
           + \sum_{i,t}\sum_{s \le t} H_{ist} x_{ist} \\
\text{s.t.}\quad & \sum_{s \le t} x_{ist} = 1 && \forall i, t \\
& x_{ist} \le y_{is},\quad x_{ist} \le z_s && \forall i,\ s \le t \\
& x, y, z \ge 0.
\end{aligned}
$$

This is the model the baselines solve, built once in
{py:func}`baselines.model.build`. Its size is worth keeping in view: the $x$
block alone has $N T (T+1)/2$ columns, and each of them carries two coupling
rows, so an instance of 200 products over 300 periods is 9.0M columns and 18.1M
rows before the demand equalities.

PRIA never forms it.

## Eliminating $y$ and $z$

At an optimum each order variable sits at the smallest value its own
constraints allow, $y_{is} = \max_{t \ge s} x_{ist}$ and
$z_s = \max_{i,\ t \ge s} x_{ist}$, so both blocks can be substituted out:

$$F(x) = \langle H, x\rangle + \sum_{i,s} K_i \max_{t \ge s} x_{ist}
        + K_0 \sum_s \max_{i,\ t \ge s} x_{ist},$$

convex and piecewise linear over the product of unit simplices
$\mathcal X = \prod_{i,t} \Delta(\{s \le t\})$. Nothing was lost and every
coupling row is gone.

## The saddle form

The support-function identity
$\kappa \max_j v_j = \max\{\langle \lambda, v\rangle : \lambda \ge 0,\
\mathbf 1^\top \lambda = \kappa\}$, applied once per max-group, turns each
maximum back into a linear term against a dual variable on a weighted simplex:

$$\min_{x \in \mathcal X}\ \max_{y \in \mathcal Y,\ z \in \mathcal Z}\
  \langle H, x\rangle + \langle y, x\rangle + \langle z, x\rangle$$

with $\mathcal Y$ a product of per-$(i,s)$ simplices over $\{t \ge s\}$ summing
to $K_i$, and $\mathcal Z$ a product of per-period simplices over
$\{(i,t) : t \ge s\}$ summing to $K_0$.

**The coupling operator is the identity.** Each $x_{ist}$ belongs to exactly one
group of each dual block — one $(i,s)$ order group and one period-$s$ joint
group — so the group-indexed space and the primal space are the same $(i,s,t)$
grid and no lift is needed. That is a property of this problem rather than of
the recipe: where groups overlap, as they do for QUBO, the variables have to be
duplicated into a group-indexed space first, and the consensus constraint that
creates is then absorbed into the primal projection.

With $A = [I; I]$ stacking the two dual blocks, $\|A\|_2 = \sqrt 2$ exactly, for
every instance, which is where the step size $\eta = 0.999/\sqrt 2$ in
{py:meth}`pria.solver.Parameters.with_auto_step` comes from. No power
iteration, no line search, no dependence on the data.

One operator application, at primal weight $\omega$, is

$$
\begin{aligned}
x' &= \mathrm{proj}_{\mathcal X}\!\left(x - \tfrac{\eta}{\omega}(H + y + z)\right)\\
y' &= \mathrm{proj}_{\mathcal Y}\!\left(y + \eta\omega\,(2x' - x)\right)\\
z' &= \mathrm{proj}_{\mathcal Z}\!\left(z + \eta\omega\,(2x' - x)\right)
\end{aligned}
$$

wrapped in a restarted reflected Halpern iteration with a log-space-smoothed
$\omega$ update at each restart.

## Three consequences

**No matrix-vector products.** An iteration is elementwise arithmetic and three
simplex projections. There is no sparse matrix in the inner loop, and nothing in
it depends on the LP's $5 N T (T+1)/2$ nonzeros.

**Fewer variables, not more.** The primal is the $x$ block, and the $y$, $z$
blocks and all $N T (T+1)$ coupling rows are gone. This is where the JRP differs
from the QUBO reformulation, which has to duplicate its primal into half-edge
space and so trades the eliminated block for one of the same order.

**Exact bounds at every iterate.** Both iterates stay feasible — the projections
see to that — so {py:func}`pria.solver.relaxation_bounds` brackets the optimum
from the first step, with no feasibility slack to account for. The upper bound
is $F(x)$; the lower is the LP dual's objective,
$\sum_{i,t} \min_{s \le t} (H_{ist} + \alpha_{ist} + \beta_{ist})$ at
$\alpha = K_i y$ and $\beta = K_0 z$, whose two budget constraints are exactly
what the dual projections enforce.

## What it costs

The projections are onto simplices of up to $T$ elements for $y$ and up to
$N(T-s)$ for $z$, not onto two-element sets, so each one needs a threshold
search rather than a closed form. Both implementations spend most of their time
there. `pria-cuda` finds the threshold with Michelot's finite algorithm — repeated
averaging of the active set, sort-free — which measured about twice as fast end
to end as the `cub::DeviceSegmentedSort` pipeline it replaced.

## The layouts

Only $s \le t$ exists, and a triangular variable does not pack into a dense
array that a segmented projection can stride over. Both implementations use the
same trick: pair column $t = j$ with column $t = T-1-j$ reversed, so every dense
row is exactly full and is the concatenation of exactly **two** projection
segments. {py:mod}`pria.upper_triangular` builds those packings and the
permutations between the three of them — one with the $y$ groups contiguous, one
with the $x$ groups contiguous, one with the $z$ groups contiguous — and
`pria-cuda/src/layouts.cuh` reproduces them as closed-form index maps.

An odd $T$ is padded to even, since the pairing needs an even column count. The
padded period's holding costs are zero, so its demand is served by any order
already placed and the padded LP has the same optimal value; see
{py:func}`pria.solver.solve_jrp_problem_warm_start` for the one place that
distinction is visible.

## What the LP is for

Unlike the QUBO roof dual, this LP is the object of study rather than a bound on
one. Its integrality gap is what the approximation guarantees for the JRP are
built on, and solving it fast is the point — not a step towards solving an
integer problem. That is why every baseline here is an LP solver and there is no
combinatorial oracle among them: there is no max-flow equivalent to reduce this
to.
