"""The benchmark's instances, normalised to one object.

Every instance is a joint replenishment problem: `items` products over
`periods` periods, each with a fixed order cost `K_i`, a joint order cost `K0`
paid once per period any order is placed, and a holding cost `H_ist` for
serving period `t`'s demand from an order in period `s <= t`.  The LP this
repository measures is::

    min   sum_s K0 z_s + sum_{i,s} K_i y_is + sum_{i,t,s<=t} H_ist x_ist
    s.t.  sum_{s<=t} x_ist = 1                for all i, t
          x_ist <= y_is,  x_ist <= z_s        for all i, s <= t
          x, y, z >= 0

stored here as its three cost arrays, so no solver backend has to know how it
was built.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np


@dataclass(frozen=True)
class Instance:
    """A joint replenishment instance.

    :param name: identifier, unique within a run (e.g. "jrp-large-n100-t200-s3")
    :param family: the bucket it came from (e.g. "jrp-large")
    :param order_costs: (items,) the per-item order costs K_i
    :param joint_order_cost: the joint order cost K0
    :param holding_costs: (items, periods, periods) H, strictly upper
        triangular in the last two axes -- serving period t from an order in
        period t itself is free, and s > t is not a variable of the LP at all
    """

    name: str
    family: str
    order_costs: np.ndarray
    joint_order_cost: float
    holding_costs: np.ndarray

    @property
    def items(self) -> int:
        """Number of products, N."""
        return int(self.order_costs.size)

    @property
    def periods(self) -> int:
        """Number of time periods, T."""
        return int(self.holding_costs.shape[-1])

    @property
    def entries(self) -> int:
        """Number of `x` variables, ``N T (T + 1) / 2``.

        The size that sets the cost of an instance, and what the figures plot
        against by default; `items` and `periods` name its shape, and neither
        orders two instances on its own.
        """
        return entries(self.items, self.periods)

    @classmethod
    def from_problem(cls, problem, name: str = "problem", family: str = "problem"
                     ) -> Instance:
        """Wrap a `pria.JrpProblem`, whose arrays are jax's.

        The inverse of what `harness.backends.pria_jax` does on the way in.
        Lets pria-jax's own tests reach the LP baselines without the library
        taking a dependency on them.
        """
        return cls(name, family,
                   np.asarray(problem.order_costs, dtype=float),
                   float(problem.joint_order_cost),
                   np.triu(np.asarray(problem.holding_costs, dtype=float), 1))


def entries(items: int, periods: int) -> int:
    """How many `x` variables an instance of this shape has, without building it.

    Enumerating a run needs the size of each instance before deciding to build
    it, which is what this is for.  Exact, unlike the edge count of a random
    graph: the JRP's variable set is determined by its shape.
    """
    return items * periods * (periods + 1) // 2


def random_jrp(seed: int, items: int, periods: int, name: str | None = None,
               family: str = "random") -> Instance:
    """A random instance, in the distribution both the benchmark and the check use.

    Mirrors `JrpProblem.random_instance` in pria-jax -- built here in numpy
    rather than imported so that a Gurobi-only run needs no jax, and so the
    harness hands every backend bit-for-bit the same instance.  It mirrors the
    *distribution*, not jax's bit stream: nothing compares an instance built
    here with one built there.

    Each item draws a holding-cost slope ``a_i ~ U(0, 1)``; the cost of
    carrying its demand from period `s` to period `t` is then
    ``round(a_i (t - s) + U(0, a_i))``, so costs rise with the wait and the
    rounding leaves ties between nearby periods.  Order costs are
    ``max(1, round(U(1, T)))``, and the joint cost K0 is drawn the same way.

    Drawing K0 on the same scale as the item order costs is what makes the
    instances hard: the LP is most fractional when the two are comparable.  A
    large K0 makes ordering jointly obviously right and the LP nearly
    integral; a small one decouples the items into T independent lot-sizing
    problems.  A single
    item is the degenerate corner of that, so K0 is zero there rather than
    drawn: with nothing to share, a joint cost is just a constant added to
    every solution.
    """
    rng = np.random.default_rng(seed)
    grid = np.arange(periods)
    wait = grid[None, :] - grid[:, None]                  # (s, t) -> t - s

    slope = rng.uniform(0.0, 1.0, (items, 1, 1))
    jitter = rng.uniform(0.0, 1.0, (items, periods, periods)) * slope
    holding = np.round(np.triu(slope * wait + jitter, 1))

    drawn = np.maximum(np.round(rng.uniform(1.0, max(periods, 1.01), items + 1)), 1.0)
    joint = float(drawn[-1]) if items > 1 else 0.0

    name = name or f"random-n{items}-t{periods}-s{seed}"
    return Instance(name, family, drawn[:-1], joint, holding)


#: The benchmark: `random_jrp` in three size buckets, as
#: (items, periods, instances) with seeds 0 to instances-1.
#:
#: Both dimensions climb together: `periods` sets the size of every simplex
#: the solver projects onto and grows the instance quadratically, while
#: `items` grows it linearly and adds independent work.
#:
#: The counts taper geometrically, since the cost of a shape grows faster
#: than the information another seed at it buys.  Several seeds per shape all
#: the same: at a fixed shape the spread between seeds is wide enough that one
#: instance settles nothing.
#:
#: What sets the counts' absolute level is that a whole pass has to fit one
#: scheduler slot for the *slowest* backend measured, not the fastest: a
#: vertex solve and a first-order one differ by orders of magnitude on the top
#: rungs, and a ladder only the winner can finish is not a comparison.
#: Widening it again means finding the queue time first, since the shapes are
#: what span the sizes and the seeds are only how sure each point is.
#:
#: The medium bucket's four shapes are under `CHECK_MAX_ENTRIES`, so the
#: correctness check has them too.  The tuning script draws its training set
#: from the same buckets with disjoint seeds.
LADDER = {
    "jrp-medium": ((10, 25, 100), (20, 50, 75), (30, 75, 60), (50, 100, 45)),
    "jrp-large": ((75, 150, 30), (100, 200, 20), (150, 250, 12), (200, 300, 8)),
    "jrp-huge": ((250, 375, 6), (300, 400, 4), (350, 450, 3)),
}

#: Every family there is, which is every bucket of the ladder: what a run gets
#: by default and the only thing `--families` can name.  The one home for the
#: list is `LADDER`, so a bucket added there is selectable without touching
#: anything here.
BENCHMARK = tuple(LADDER)


@dataclass(frozen=True)
class Spec:
    """One instance of a run, named and sized before it exists.

    Enumerating a run must not build it.  An instance costs ``8 N T^2`` bytes
    of holding costs alone, so the ladder's top rung is over half a gigabyte
    and a pass that materialised every instance before solving the first would
    need hundreds of gigabytes of them resident at once -- for a set whose
    whole point is that only one instance is in play at a time.  A spec
    carries what selecting, ordering and reporting need, and `build()` makes
    the instance when its turn comes; the caller drops it again afterwards.
    """

    name: str
    family: str
    items: int
    periods: int
    entries: int
    build: Callable[[], Instance]


def build_ladder(bucket: str, items: int, periods: int, seed: int, name: str
                 ) -> Instance:
    """One generated benchmark instance, named for its bucket."""
    return random_jrp(seed, items, periods, name=name, family=bucket)


def ladder_specs(bucket: str, seed_base: int = 0) -> list[Spec]:
    """One family of generated instances as specs; nothing is built or stored.

    `seed_base` shifts every seed: the benchmark is 0, and a training set that
    must not overlap it uses another.
    """
    out = []
    for items, periods, count in LADDER[bucket]:
        for k in range(count):
            seed = seed_base + k
            name = f"{bucket}-n{items}-t{periods}-s{seed}"
            out.append(Spec(name, bucket, items, periods, entries(items, periods),
                            partial(build_ladder, bucket, items, periods, seed, name)))
    return out


def all_specs(families: list[str] | None = None) -> list[Spec]:
    """Every instance of every requested family as a spec (default: all of them).

    Nothing is built: a spec knows its name and size, and makes its instance
    when its turn comes.
    """
    out = []
    for family in families or list(BENCHMARK):
        out.extend(ladder_specs(family))
    return out


def random_spec(seed: int, items: int, periods: int) -> Spec:
    """A spec for `random_jrp`, so the correctness check enumerates its run
    before building any of it, the way a benchmark pass does."""
    name = f"random-n{items}-t{periods}-s{seed}"
    return Spec(name, "random", items, periods, entries(items, periods),
                partial(random_jrp, seed, items, periods, name))
