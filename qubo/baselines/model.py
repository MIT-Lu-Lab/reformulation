"""The standard linearization of a QUBO, as an explicit LP.

This is the single home for the model the baselines solve.  Both `gurobi_backend`
and `cupdlpx_backend` build from here, so a change to the formulation cannot
reach one baseline and miss the other.

An independent construction of the same LP lives in
`praq-jax/praq/scipy_implementation.py` (HiGHS, used by PRAQ's own tests).  The
two are kept honest by `scripts/check_correctness.py`, which compares objective
values across implementations; they are not shared code on purpose, so that
agreement is evidence rather than tautology.

The model, for W symmetric with d = diag(W) and w_e = 2 W_ij on each edge::

    min  <d, x> + sum_e w_e z_e
    s.t. x_i + x_j - z_e <= 1                    w_e > 0   (covering)
         z_e - x_i <= 0,  z_e - x_j <= 0         w_e < 0   (packing)
         0 <= x <= 1,  0 <= z <= 1

The diagonal enters linearly via x_i^2 = x_i rather than through z_ii; the
covering block carries z_e >= 0 through the variable bound; and x is boxed
explicitly.  Dropping any of the three changes the optimal value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_array


@dataclass(frozen=True)
class StandardLinearization:
    """The LP above in solver-agnostic sparse form.

    Columns are x (indices 0..n-1) then z (indices n..n+m-1), all boxed to
    [0, 1].  Rows are all "<=".
    """

    n: int
    m: int
    cost: np.ndarray            # (n + m,)
    matrix: coo_array           # (rows, n + m)
    rhs: np.ndarray             # (rows,)
    ends_i: np.ndarray          # (m,) first endpoint of each edge
    ends_j: np.ndarray          # (m,) second endpoint

    @property
    def columns(self) -> int:
        return self.n + self.m

    @property
    def rows(self) -> int:
        return int(self.rhs.size)


def build(instance) -> StandardLinearization:
    """Build the standard linearization of `instance`.

    :param instance: a `harness.instances.Instance`, in minimisation form
    :return: the LP, with all rows written as "<=" and all columns in [0, 1]
    """
    n = instance.n
    weights = 2.0 * instance.vals
    keep = weights != 0.0
    ends_i, ends_j, weights = instance.rows[keep], instance.cols[keep], weights[keep]
    m = int(weights.size)

    cost = np.concatenate([instance.diag, weights])

    covering = np.flatnonzero(weights > 0.0)
    packing = np.flatnonzero(weights < 0.0)
    n_rows = covering.size + 2 * packing.size

    row = np.empty(3 * covering.size + 4 * packing.size, dtype=np.int64)
    col = np.empty_like(row)
    val = np.empty(row.size, dtype=float)
    rhs = np.zeros(n_rows, dtype=float)

    # covering, one row each: x_i + x_j - z_e <= 1
    c_rows = np.arange(covering.size)
    head = 3 * covering.size
    row[:head] = np.repeat(c_rows, 3)
    col[:head] = np.stack([ends_i[covering], ends_j[covering], n + covering], 1).ravel()
    val[:head] = np.tile([1.0, 1.0, -1.0], covering.size)
    rhs[:covering.size] = 1.0

    # packing, two rows each: z_e - x_i <= 0 and z_e - x_j <= 0
    p_rows = covering.size + np.arange(2 * packing.size)
    row[head:] = np.repeat(p_rows, 2)
    col[head:] = np.stack([
        np.repeat(n + packing, 2),
        np.stack([ends_i[packing], ends_j[packing]], 1).ravel(),
    ], 1).ravel()
    val[head:] = np.tile([1.0, -1.0], 2 * packing.size)

    matrix = coo_array((val, (row, col)), shape=(n_rows, n + m))
    return StandardLinearization(n, m, cost, matrix, rhs, ends_i, ends_j)


def write_mps(lp: StandardLinearization, path, name: str = "QUBO_LP") -> None:
    """Write `lp` as an MPS file, for solvers that take one on disk.

    cuPDLPx reads MPS, so this is how an instance reaches it without a Gurobi
    licence in the loop.  Values are written with %.17g: numpy scalars must be
    converted first, since numpy 2 reprs them as "np.float64(...)".

    The columns are padded to the classic widths for readability, but every
    field also ends in a space of its own, because a 17-digit value is wider
    than its column: without the space it would run straight into the next row
    name, which is what cuPDLPx's "Failed to read or parse the file" meant on
    the `hard` family, whose weights are the first here that are not integers.
    """
    csc = lp.matrix.tocsc()

    def num(value) -> str:
        return "%.17g" % float(value)

    def field(text: str, width: int) -> str:
        return (text + " ").ljust(width)

    with open(path, "w") as out:
        out.write(f"NAME          {name}\nROWS\n N  COST\n")
        for r in range(lp.rows):
            out.write(f" L  R{r}\n")

        out.write("COLUMNS\n")
        for c in range(lp.columns):
            entries = []
            if lp.cost[c] != 0.0:
                entries.append(("COST", lp.cost[c]))
            lo, hi = csc.indptr[c], csc.indptr[c + 1]
            entries.extend((f"R{csc.indices[k]}", csc.data[k]) for k in range(lo, hi))
            for k in range(0, len(entries), 2):
                fields = []
                for row_name, value in entries[k:k + 2]:
                    fields.append(field(row_name, 12))
                    fields.append(field(num(value), 14))
                out.write("    " + field(f"C{c}", 12) + "".join(fields).rstrip() + "\n")

        out.write("RHS\n")
        for r in range(lp.rows):
            if lp.rhs[r] != 0.0:
                out.write("    " + field("RHS", 12) + field(f"R{r}", 12)
                          + num(lp.rhs[r]) + "\n")

        out.write("BOUNDS\n")
        for c in range(lp.columns):
            out.write(" UP " + field("BND", 12) + field(f"C{c}", 12) + "1\n")
        out.write("ENDATA\n")
