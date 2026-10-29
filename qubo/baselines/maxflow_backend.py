"""Compute the roof dual as a maximum flow, with OR-Tools.

Roof duality is not an LP problem.  Hammer, Hansen and Simeone reduce it to a
maximum flow on an implication network, and that is the baseline
`docs/reformulation.md` names as the right one for this problem: a CPU, no
licence, no tolerance and no iteration count.

It is worth more than a timing column.  The flow value *equals* the optimum of
the standard linearization's LP relaxation, so this produces the same number as
`highs_backend`, `gurobi_backend`, `cupdlpx_backend` and both PRAQ backends, by
a combinatorial route that shares no code and no arithmetic with any of them.
That makes it the reference for `scripts/check_correctness.py`: the LP baselines
all stop on a tolerance, and this one does not.

`baselines/model.py` is the model of record for the linearization and stays so.
This module is the second, independent construction; agreement between them is
evidence rather than tautology, which is the whole point of having it.

The construction
----------------

Write the objective as a *posiform*: a constant plus a non-negative combination
of products of literals, where a literal is some ``x_i`` or its complement
``1 - x_i``.  Only two rewritings are needed.  A negative product loses its sign
onto the linear part,

.. math::  w\\, x_i x_j = |w|\\, x_i (1 - x_j) - |w|\\, x_i \\qquad (w < 0),

and a negative linear coefficient becomes a constant plus its complement,
``a x_i = a + |a| (1 - x_i)`` for ``a < 0``.  Every remaining coefficient is
positive.  Linear terms are products too, against a literal that is constantly
true, so the posiform is a set of terms ``a * u * v`` over literals.

Now give every literal a node, add a source node for the constantly-false
literal and a sink node for the constantly-true one, and send each term to a
*pair* of arcs::

    term  a * u * v   ->   arc (NOT u) -> v   and   arc (NOT v) -> u,
                           capacity a/2 each

An s-t cut assigns 0 to the source side and 1 to the sink side, and an arc
``p -> q`` is cut exactly when ``p`` takes 0 and ``q`` takes 1, so it
contributes ``c (1 - p) q``.  On a cut that is *consistent* -- one where each
literal and its complement disagree -- the two arcs of a term contribute
``(a/2) u v`` apiece, so a consistent cut's value is the posiform at that ``x``
less its constant, and the minimum over consistent cuts is
``min_x f(x) - constant``.  Every cut is a candidate and only some are
consistent, so dropping the requirement can only lower the minimum:

.. math::  \\text{constant} + \\text{min-cut} \\le \\min_x x^T W x.

The theorem -- Hammer, Hansen and Simeone (1984), in the implication-network
form of Boros and Hammer's 2002 survey, section 4 -- is that this inequality is
tight against the roof dual rather than something weaker.  Max-flow min-cut then
turns it into a flow computation.

Complementing the endpoints of a negative product gives the other choice of
which endpoint absorbs the leftover linear term, hence a different network; the
bound is the same, and `scripts/check_correctness.py` against the LP baselines
is what says so.

What it costs
-------------

`SimpleMaxFlow` is a single-threaded push-relabel and it reaches the whole
size ladder, well inside `harness.config.TIME_LIMIT` at every rung; its
resident memory is the dominant cost, most of it OR-Tools' own.  The
measurements are in `docs/harness.md`.

Capacities are rounded to integers, so the bound carries a slack that grows
with size as the scale shrinks against the total weight.  It reaches the
record as the `lower`/`upper` bracket, which is what
`scripts/check_correctness.py` allows for and what `plot_results.py
--figures uncertainty` draws against the agreement budget.  An instance with
integer weights has no slack at all, that being a property of the weights
rather than of this backend.

Persistency comes free from residual reachability -- `get_source_side_min_cut`
already returns it -- and is a natural second output, but nothing here needs it
yet, so it is not built.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from harness.config import Tolerances
from harness.results import Result

#: Ceiling on the network's total integer capacity.
#:
#: OR-Tools' max flow is exact and therefore integral: `SimpleMaxFlow` takes
#: int64 capacities, and the weights here are log-normal floats spanning orders
#: of magnitude, so they have to be scaled and rounded.  `build_network` picks
#: the largest power-of-two scale whose total capacity fits under this, which is
#: what makes the rounding error as small as int64 allows instead of a constant
#: guess.  Half of int64's range, so that an intermediate sum of two cuts cannot
#: overflow either -- reaching it at all would take total weight above 2^61.
CAPACITY_BUDGET = 2 ** 62


@dataclass(frozen=True)
class Network:
    """The implication network, ready for `SimpleMaxFlow`.

    Literal ``x_i`` is node ``2i`` and its complement is node ``2i + 1``, so
    complementing a literal is XOR with 1.  `source` is the constantly-false
    literal and `sink` the constantly-true one, numbered ``2n`` and ``2n + 1``
    so that XOR with 1 swaps them too -- which is why linear terms need no
    special case anywhere.

    Capacities are in units of ``1 / (2 * scale)`` of the objective: each of a
    term's two arcs carries half the term's coefficient, and the scale absorbs
    the other factor.  `scale` is a power of two, so dividing the flow by it is
    exact and the only floating-point error left is the flow's own conversion
    from int64.
    """

    tails: np.ndarray           # (arcs,) int32
    heads: np.ndarray           # (arcs,) int32
    capacities: np.ndarray      # (arcs,) int64
    source: int
    sink: int
    constant: float
    scale: float
    slack: float
    """Total rounding error of the capacities, in objective units.

    The exact sum of ``|round(c) - c|`` over the arcs, not a worst-case count:
    any cut crosses a subset of the arcs, so this bounds how far the rounded
    network's min-cut can be from the real one, in either direction.

    It is *zero* on any instance whose weights are integers, since `scale` is
    then an integer too: it drops below one only for a total absolute weight
    above 2^61.
    """

    @property
    def arcs(self) -> int:
        return int(self.capacities.size)


def _posiform(instance):
    """Rewrite `instance` as a quadratic posiform over literals.

    :return: the constant, then the terms as parallel arrays ``(u, v, a)`` of
        literal node indices and non-negative coefficients.  Linear terms
        appear with ``v`` equal to the sink, the constantly-true literal, and
        are the only ones whose coefficient can be zero.
    """
    weights = 2.0 * instance.vals            # x^T W x has 2 W_ij on each edge
    keep = weights != 0.0
    ends_i, ends_j, weights = instance.rows[keep], instance.cols[keep], weights[keep]

    # A negative product becomes "x_i and not x_j", pushing |w| onto the linear
    # coefficient of i -- the diagonal, which is posiformed next.
    down = weights < 0.0
    linear = instance.diag.astype(float, copy=True)
    np.add.at(linear, ends_i[down], weights[down])

    below = linear < 0.0
    sink = 2 * instance.n + 1
    return (float(linear[below].sum()),
            np.concatenate([2 * ends_i, 2 * np.arange(instance.n) + below]),
            np.concatenate([2 * ends_j + down, np.full(instance.n, sink)]),
            np.concatenate([np.abs(weights), np.abs(linear)]))


def build_network(instance, budget: int = CAPACITY_BUDGET) -> Network:
    """Build the implication network of `instance`, with integer capacities.

    Two arcs per edge and two per variable.  The two arcs of a term always carry
    the same capacity, so the rounding happens once per term and the scale is
    chosen from the terms' total: the network's total capacity is
    ``2 * scale * sum(coefficients)``, and this is the largest power of two that
    keeps it under `budget`.

    The variable arcs are kept even where the linear coefficient is zero, which
    costs two arcs of zero capacity and no accuracy -- an exactly-representable
    capacity contributes nothing to `Network.slack`.
    """
    constant, u, v, coefficients = _posiform(instance)

    total = float(coefficients.sum())
    exponent = 0 if total <= 0.0 else math.floor(math.log2(budget / (2.0 * total)))
    scale = 2.0 ** exponent

    scaled = coefficients * scale
    rounded = np.round(scaled)
    slack = float(np.abs(rounded - scaled).sum() / scale)

    capacity = rounded.astype(np.int64)
    return Network(tails=np.concatenate([u ^ 1, v ^ 1]).astype(np.int32),
                   heads=np.concatenate([v, u]).astype(np.int32),
                   capacities=np.concatenate([capacity, capacity]),
                   source=2 * instance.n, sink=2 * instance.n + 1,
                   constant=constant, scale=scale, slack=slack)


def solve(instance, *, tolerances: Tolerances, time_limit: float) -> Result:
    """Compute the roof dual of `instance` as a maximum flow.

    Neither argument reaches the solver and both columns stay NaN, which is the
    point of this baseline rather than an omission: the computation is
    combinatorial and exact, so there is no tolerance to set, and
    `SimpleMaxFlow` has no time limit to give it either -- a run that must be
    bounded has to bound it by instance size instead.  `iterations` keeps its
    -1 for the same reason.

    `Result.lower` and `Result.upper` bracket the true roof dual by
    `Network.slack`, the capacities' rounding error, which is zero whenever the
    instance's weights are integers.
    """
    common = dict(backend="maxflow", instance=instance.name, family=instance.family,
                  n=instance.n, edges=instance.edges)
    try:
        from ortools.graph.python import max_flow
    except ModuleNotFoundError:
        return Result(status="unavailable", note="ortools not installed", **common)

    start = time.perf_counter()
    network = build_network(instance)
    solver = max_flow.SimpleMaxFlow()
    solver.add_arcs_with_capacity(network.tails, network.heads, network.capacities)
    setup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    status = solver.solve(network.source, network.sink)
    solve_seconds = time.perf_counter() - start

    note = (f"scale 2^{int(math.log2(network.scale))}, {network.arcs} arcs, "
            f"rounding slack {network.slack:.3e}")
    if status != solver.OPTIMAL:
        return Result(status="error", note=f"max flow status {status!r}; {note}",
                      setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                      **common)

    objective = network.constant + solver.optimal_flow() / (2.0 * network.scale)
    return Result(status="optimal", objective=objective,
                  lower=objective - network.slack, upper=objective + network.slack,
                  setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                  note=note, **common)
