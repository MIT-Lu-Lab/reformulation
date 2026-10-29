"""The joint replenishment LP, built explicitly.

This is the single home for the model the baselines solve.  `highs_backend`,
`gurobi_backend` and `cupdlpx_backend` all build from here, so a change to the
formulation cannot reach one baseline and miss the others.

PRIA never forms this.  That is the point of the comparison: the reformulation
it solves has the same optimal value and `2 N T(T+1)/2 + N T` fewer rows, and
what these baselines measure is what the explicit LP costs an LP solver.

The model, for an instance with N items and T periods::

    min  sum_{i,s<=t} H_ist x_ist + sum_{i,s} K_i y_is + sum_s K0 z_s
    s.t. x_ist - y_is <= 0                       for all i, s <= t
         x_ist - z_s  <= 0                       for all i, s <= t
         sum_{s<=t} x_ist = 1                    for all i, t
         x, y, z >= 0

Two details of the encoding are load bearing.

**Only `s <= t` exists.**  The x block is one flat axis over the ``T(T+1)/2``
valid pairs, in `np.triu_indices` order, not a `T x T` grid.

**No column has an upper bound.**  `x <= 1` follows from the demand equality
together with `x >= 0`, and `y` and `z` sit at the largest `x` they must
dominate, which that equality already holds at or below one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_array


@dataclass(frozen=True)
class Formulation:
    """The LP above in solver-agnostic sparse form.

    Columns are x (indices 0..E-1, in `np.triu_indices` order within each
    item), then y (E..E+NT-1), then z.  Rows are two-sided: the y and z
    coupling rows are `(-inf, 0]` and the demand rows are `[1, 1]`, which is
    the form cuPDLPx takes directly and which the other two split apart.
    """

    items: int
    periods: int
    pairs: int                   #: T(T+1)/2, the (s, t) pairs one item has
    cost: np.ndarray             # (columns,)
    matrix: coo_array            # (rows, columns)
    row_lower: np.ndarray        # (rows,)
    row_upper: np.ndarray        # (rows,)
    column_lower: np.ndarray     # (columns,)
    column_upper: np.ndarray     # (columns,)
    order_starts: tuple[int, int]  #: where the y and z blocks begin

    @property
    def entries(self) -> int:
        """Number of x columns, N T(T+1)/2."""
        return self.items * self.pairs

    @property
    def columns(self) -> int:
        return int(self.cost.size)

    @property
    def rows(self) -> int:
        return int(self.row_upper.size)


def build(instance) -> Formulation:
    """Build the LP of `instance`.

    Vectorised, and int32-indexed: there are five nonzeros per x column, and
    the largest instances of `harness.instances.LADDER` have too many for a
    Python loop over pairs to finish or for 64-bit indices to be free.

    :param instance: a `harness.instances.Instance`
    :return: the LP, with two-sided rows and nonnegative columns
    """
    items, periods = instance.items, instance.periods
    si, ti = np.triu_indices(periods)
    pairs = si.size
    count = items * pairs

    holding = np.ascontiguousarray(instance.holding_costs[:, si, ti], dtype=float)
    cost = np.concatenate([
        holding.ravel(),
        np.repeat(np.asarray(instance.order_costs, dtype=float), periods),
        np.full(periods, float(instance.joint_order_cost)),
    ])

    y_base = count
    z_base = count + items * periods

    # Every nonzero is indexed off the same (item, pair) enumeration: entry k
    # is item k // pairs at pair k % pairs, whose source period is si and whose
    # demand period is ti.
    x_columns = np.arange(count, dtype=np.int32)
    item_offset = np.repeat(np.arange(items, dtype=np.int32) * periods, pairs)
    source = np.tile(si.astype(np.int32), items)
    demand = np.tile(ti.astype(np.int32), items)

    y_columns = y_base + item_offset + source
    z_columns = np.int32(z_base) + source
    demand_rows = np.int32(2 * count) + item_offset + demand

    rows = np.concatenate([x_columns, x_columns,                    # x in its two
                           np.int32(count) + x_columns,             # coupling rows
                           np.int32(count) + x_columns,
                           demand_rows])
    columns = np.concatenate([x_columns, y_columns,
                              x_columns, z_columns,
                              x_columns])
    values = np.concatenate([
        np.ones(count), np.full(count, -1.0),
        np.ones(count), np.full(count, -1.0),
        np.ones(count),
    ])

    n_rows = 2 * count + items * periods
    n_columns = z_base + periods
    matrix = coo_array((values, (rows, columns)), shape=(n_rows, n_columns))

    row_lower = np.concatenate([np.full(2 * count, -np.inf),
                                np.ones(items * periods)])
    row_upper = np.concatenate([np.zeros(2 * count), np.ones(items * periods)])

    return Formulation(items, periods, int(pairs), cost, matrix, row_lower,
                       row_upper, np.zeros(n_columns), np.full(n_columns, np.inf),
                       (y_base, z_base))


def write_mps(lp: Formulation, path, name: str = "JRP_LP") -> None:
    """Write `lp` as an MPS file, for a solver that takes one on disk.

    Not used to solve anything here -- every baseline is handed the arrays --
    but writing the file is separately useful for staging an instance onto a
    machine that has a solver and not this harness.

    No BOUNDS section: MPS defaults a column to `[0, +inf)`, which is what
    every column of this LP has.  Values are written with %.17g, and numpy
    scalars are converted first, since numpy 2 reprs them as
    "np.float64(...)".
    """
    csc = lp.matrix.tocsc()

    def num(value) -> str:
        return f"{float(value):.17g}"

    def field(text: str, width: int) -> str:
        # Every field ends in a space of its own as well as being padded: a
        # 17-digit value is wider than its column, and without the space it
        # would run straight into the next row name.
        return (text + " ").ljust(width)

    with open(path, "w") as out:
        out.write(f"NAME          {name}\nROWS\n N  COST\n")
        for r in range(lp.rows):
            out.write(f" {'E' if lp.row_lower[r] == lp.row_upper[r] else 'L'}  R{r}\n")

        out.write("COLUMNS\n")
        for c in range(lp.columns):
            pieces = []
            if lp.cost[c] != 0.0:
                pieces.append(("COST", lp.cost[c]))
            lo, hi = csc.indptr[c], csc.indptr[c + 1]
            pieces.extend((f"R{csc.indices[k]}", csc.data[k]) for k in range(lo, hi))
            for k in range(0, len(pieces), 2):
                cells = []
                for row_name, value in pieces[k:k + 2]:
                    cells.append(field(row_name, 12))
                    cells.append(field(num(value), 14))
                out.write("    " + field(f"C{c}", 12) + "".join(cells).rstrip() + "\n")

        out.write("RHS\n")
        for r in range(lp.rows):
            if lp.row_upper[r] != 0.0:
                out.write("    " + field("RHS", 12) + field(f"R{r}", 12)
                          + num(lp.row_upper[r]) + "\n")
        out.write("ENDATA\n")
