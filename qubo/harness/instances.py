"""The benchmark's instances, normalised to one object.

Every instance is a *minimisation* QUBO ``min_{x in {0,1}^n} x^T W x`` with W
symmetric, stored sparsely as its diagonal plus its strict upper triangle, so
no solver backend has to know how it was built.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import numpy as np


@dataclass(frozen=True)
class Instance:
    """A QUBO in minimisation form.

    :param name: identifier, unique within a run (e.g. "hard-large-n20000-s3")
    :param family: the bucket it came from (e.g. "hard-large")
    :param n: number of binary variables
    :param diag: (n,) diagonal of W, the linear coefficients
    :param rows: (m,) row index of each strict-upper-triangle nonzero, i < j
    :param cols: (m,) column index of each, matching `rows`
    :param vals: (m,) value W_ij = W_ji of each
    """

    name: str
    family: str
    n: int
    diag: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray

    @property
    def edges(self) -> int:
        """Number of off-diagonal nonzeros, i.e. edges of the coupling graph."""
        return int(self.vals.size)

    @classmethod
    def from_dense(cls, matrix, name: str = "dense", family: str = "dense") -> "Instance":
        """Wrap a symmetric (n, n) matrix, already in minimisation form.

        The inverse of `to_dense`.  Lets praq-jax's own `QuboProblem` reach the
        LP baselines without the library taking a dependency on them.
        """
        w = np.asarray(matrix, dtype=float)
        n = w.shape[0]
        rows, cols = np.triu_indices(n, k=1)
        keep = w[rows, cols] != 0.0
        return cls(name, family, n, np.diagonal(w).copy(),
                   rows[keep], cols[keep], w[rows, cols][keep])

    def to_dense(self) -> np.ndarray:
        """Materialise the symmetric (n, n) matrix.

        Costs 8 n^2 bytes.  `praq-jax` needs this because `QuboProblem` stores W
        densely; the LP baselines do not and should not call it.
        """
        w = np.zeros((self.n, self.n), dtype=float)
        w[np.arange(self.n), np.arange(self.n)] = self.diag
        w[self.rows, self.cols] = self.vals
        w[self.cols, self.rows] = self.vals
        return w


def random_hard(seed: int, n: int, degree: float = 20.0, negative: float = 0.25,
                linear: float = 0.1, spread: float = 2.0) -> Instance:
    """A random instance built to be hard for the relaxation, not the QUBO.

    The graph is `random_maxcut`'s: about ``n * degree / 2`` uniformly random
    pairs.  Each edge weight is log-normal with median 1 and log-standard
    deviation `spread`, negative with probability `negative`; each diagonal
    entry is `linear` times the node's incident absolute weight times a uniform
    draw from [-1, 1].

    The defaults are the hardest corner of a sweep of these knobs, and
    praq-cuda, cuPDLPx and Gurobi rank that sweep the same way.  What each one
    does:

    - `spread` matters most: a heavy enough tail gives the LP a dynamic range
      the objective scale cannot hide.
    - `negative` at 1 is a submodular instance, whose relaxation is exact; the
      default is well away from that.
    - `linear` at 0 leaves the whole primal at 1/2 and at 1 lets persistency
      fix everything, so the default keeps a fractional core without making it
      trivial.
    - `degree` barely moves the iteration count; `HARD_DEGREE` is where the
      benchmark pins it.
    """
    rng = np.random.default_rng(seed)
    count = int(round(n * degree / 2))
    i, j = rng.integers(0, n, count), rng.integers(0, n, count)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    pairs = np.unique(np.stack([lo[lo != hi], hi[lo != hi]], axis=1), axis=0)
    lo, hi = pairs[:, 0], pairs[:, 1]
    m = lo.size

    magnitude = np.exp(spread * rng.standard_normal(m))
    sign = np.where(rng.random(m) < negative, -1.0, 1.0)
    vals = sign * magnitude

    incident = np.zeros(n, dtype=float)
    np.add.at(incident, lo, magnitude)
    np.add.at(incident, hi, magnitude)
    diag = linear * incident * rng.uniform(-1.0, 1.0, n)
    return Instance(f"hard-n{n}-k{degree:g}-s{seed}", "hard", n, diag, lo, hi, vals)


#: The degree the benchmark's size ladder is generated at, and a local
#: maximum of difficulty: the iteration count peaks here and falls away on
#: both sides.  Instances off it are measuring an easier problem and their
#: counts do not compare with the ladder's.
HARD_DEGREE = 20.0

#: The benchmark: `random_hard` at its defaults in three size buckets, as
#: (n, instances) pairs with seeds 0 to instances-1.
#:
#: The counts taper geometrically, since the cost of a size grows faster than
#: the information another seed at it buys, and at the top rung an instance
#: takes longer to generate than to solve.  Generation is paid once per pass
#: rather than once per backend, because `run_benchmark.py` loops instances
#: outermost.  Several seeds per size all the same: at a fixed size the spread
#: between seeds is wide enough that one instance settles nothing.
#:
#: The medium bucket's two smallest sizes are under `CHECK_MAX_EDGES`, so the
#: correctness check has them too.  The tuning script draws its training set
#: from the same buckets with disjoint seeds, drawn afresh for every
#: configuration it scores.
HARD_BUCKETS = {
    "hard-medium": ((500, 500), (1000, 397), (2000, 316), (5000, 251)),
    "hard-large": ((10_000, 200), (20_000, 159), (50_000, 126), (100_000, 100)),
    "hard-huge": ((200_000, 80), (500_000, 63), (1_000_000, 50), (2_000_000, 40)),
}


def expected_edges(n: int, degree: float = HARD_DEGREE) -> int:
    """How many distinct edges `random_hard` will produce, without producing them.

    It draws ``n * degree / 2`` pairs uniformly and drops duplicates and loops,
    so the yield is the coupon-collector count, ``N (1 - e**(-k/N))`` out of the
    ``N = n (n - 1) / 2`` possible pairs.  At degree 20 that is 99.9% of the
    draws and the distinction hardly matters; as the degree approaches n it
    saturates at 1 - 1/e, and then it matters a great deal.  Enumerating a run
    needs the size of each instance before deciding to build it, which is what
    this is for; it is an expectation, so it can be off by a few edges.
    """
    pairs = n * (n - 1) / 2
    if pairs <= 0:
        return 0
    draws = round(n * degree / 2)
    return int(round(pairs * -math.expm1(-draws / pairs)))


def build_hard(bucket: str, n: int, seed: int, name: str) -> Instance:
    """One generated benchmark instance, named for its bucket."""
    return replace(random_hard(seed, n, degree=HARD_DEGREE), name=name, family=bucket)


def hard_specs(bucket: str, seed_base: int = 0) -> list["Spec"]:
    """One family of generated instances as specs; nothing is built or stored.

    `seed_base` shifts every seed: the benchmark is 0, and a training set that
    must not overlap it uses another.
    """
    out = []
    for n, count in HARD_BUCKETS[bucket]:
        for k in range(count):
            seed = seed_base + k
            name = f"{bucket}-n{n}-s{seed}"
            out.append(Spec(name, bucket, n, expected_edges(n),
                            partial(build_hard, bucket, n, seed, name)))
    return out


#: Every family there is, which is every bucket of the generated benchmark:
#: what a run gets by default and the only thing `--families` can name.  The
#: one home for the list is `HARD_BUCKETS`, so a bucket added there is
#: selectable without touching anything here.
BENCHMARK = tuple(HARD_BUCKETS)


@dataclass(frozen=True)
class Spec:
    """One instance of a run, named and sized before it exists.

    Enumerating a run must not build it.  A generated instance costs about 25
    bytes per edge, so the benchmark's top rung is a gigabyte and a pass that
    materialised every instance before solving the first would need tens of
    gigabytes of them resident at once -- for a set whose whole point is that
    only one instance is in play at a time.  A spec carries what selecting,
    ordering and reporting need, and `build()` makes the instance when its
    turn comes; the caller drops it again afterwards.

    `edges` is an expectation (`expected_edges`), since counting them exactly
    means generating the instance.  It is used to select and order, not
    reported.
    """

    name: str
    family: str
    n: int
    edges: int
    build: Callable[[], Instance]


def all_specs(families: list[str] | None = None) -> list[Spec]:
    """Every instance of every requested family as a spec (default: all of them).

    Nothing is built: a spec knows its name and size, and makes its instance
    when its turn comes.
    """
    out = []
    for family in families or list(BENCHMARK):
        out.extend(hard_specs(family))
    return out


def random_maxcut(seed: int, n: int, degree: float, signed: bool = False) -> Instance:
    """A random sparse max-cut instance, as a minimisation QUBO.

    About ``n * degree / 2`` edges join uniformly random pairs -- duplicates and
    loops are dropped, so the count is at most that -- with weight 1, or +-1
    with `signed`.  Maximising ``cut(x) = sum_E w_ij (x_i + x_j - 2 x_i x_j)``
    is minimising ``x^T W x`` for ``W_ij = w_ij`` off the diagonal and
    ``W_ii = -sum_j w_ij``, which is the diagonal built here.

    This is the shape `random_instance` cannot produce: a max-cut graph is
    sparse enough that a density would round to zero at these sizes, and its
    diagonal is determined by its edges rather than drawn.
    """
    rng = np.random.default_rng(seed)
    count = int(round(n * degree / 2))
    i, j = rng.integers(0, n, count), rng.integers(0, n, count)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    pairs = np.unique(np.stack([lo[lo != hi], hi[lo != hi]], axis=1), axis=0)
    lo, hi = pairs[:, 0], pairs[:, 1]
    w = rng.choice([-1.0, 1.0], lo.size) if signed else np.ones(lo.size)

    diag = np.zeros(n, dtype=float)
    np.add.at(diag, lo, -w)
    np.add.at(diag, hi, -w)
    flavour = "-signed" if signed else ""
    return Instance(f"maxcut-n{n}-k{degree:g}{flavour}-s{seed}", "maxcut", n, diag, lo, hi, w)


def random_spec(seed: int, n: int, density: float) -> Spec:
    """A spec for `random_instance`, so the correctness check enumerates its
    run before building any of it, the way a benchmark pass does.

    The edge count is the expectation, `density` of the n (n - 1) / 2 possible
    pairs; it selects and orders, and the reported count comes from the
    instance itself.
    """
    return Spec(f"random-n{n}-d{density}-s{seed}", "random", n,
                int(round(density * n * (n - 1) / 2)),
                partial(random_instance, seed, n, density))


def random_instance(seed: int, n: int, density: float, magnitude: float = 3.0
                    ) -> Instance:
    """A random symmetric integer instance, for the correctness check.

    Mirrors `QuboProblem.random_instance` in praq-jax: entries drawn uniformly
    from +-[1, magnitude], off-diagonal entries kept with probability `density`.
    Built here rather than imported so that a Gurobi-only run needs no jax.
    """
    rng = np.random.default_rng(seed)

    def values(size):
        return rng.integers(1, magnitude + 1, size) * rng.choice([-1.0, 1.0], size)

    rows, cols = np.triu_indices(n, k=1)
    keep = rng.random(rows.size) < density
    rows, cols = rows[keep], cols[keep]

    return Instance(f"random-n{n}-d{density}-s{seed}", "random", n,
                    values(n).astype(float), rows, cols,
                    values(rows.size).astype(float))
