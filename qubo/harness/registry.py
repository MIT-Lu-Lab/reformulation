"""Which solvers the scripts can drive, and where each one lives.

The two PRAQ implementations are the library.  The baselines are not: they live
under `baselines/` and are imported only when actually selected, so a machine
with no Gurobi licence and no GPU can still run the PRAQ backends.
"""

from __future__ import annotations

import importlib
from typing import Callable

#: flag name -> module implementing `solve(instance, *, tolerances, time_limit)`
#: -- taking a `harness.config.Tolerances` -- and returning a
#: `harness.results.Result`.  A backend in `SAMPLED_MEMORY` takes one more
#: keyword, `sample_memory`, and is the only kind a run passes it to.
BACKENDS = {
    "praq-jax": "harness.backends.praq_jax",
    "praq-cuda": "harness.backends.praq_cuda",
    "gurobi": "baselines.gurobi_backend",
    "highs": "baselines.highs_backend",
    "cupdlpx": "baselines.cupdlpx_backend",
    "maxflow": "baselines.maxflow_backend",
}

#: The backends that are the object of study, as opposed to comparisons.
LIBRARY = ("praq-jax", "praq-cuda")

#: The backends whose `iterations` column counts the same kind of step, so that
#: dividing by it or comparing it across them means something.  All three stop
#: on a relative duality gap after cheap iterations; HiGHS and Gurobi reach a
#: vertex by dual simplex, whose iterations are a different object entirely and
#: hundreds of times more expensive each, and the max-flow oracle has no
#: iteration to report at all.  `harness.figures` plots only these on any figure
#: with iterations in it, which is the only way that figure can be read.
FIRST_ORDER = ("praq-jax", "praq-cuda", "cupdlpx")

#: The backends that measure their own peak memory when a run asks for it with
#: `run_benchmark.py --memory`.  The two GPU solvers, because the comparison the
#: memory table is for is theirs: whether an instance fits on the card is the
#: question, and `harness.memory` can attribute device memory to a process but
#: not to a library inside one.  A run that asks the others for it gets nothing,
#: which is why it says so rather than sampling for an hour to no purpose.
SAMPLED_MEMORY = ("praq-cuda", "cupdlpx")


def load(name: str) -> Callable:
    """Return the `solve` callable of backend `name`, importing it on demand."""
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}, expected one of {sorted(BACKENDS)}")
    return importlib.import_module(BACKENDS[name]).solve


def add_backend_flags(parser) -> None:
    """Add one --<backend> flag per entry, so scripts share a spelling."""
    group = parser.add_argument_group("implementations")
    for name in BACKENDS:
        group.add_argument(f"--{name}", dest="backends", action="append_const",
                           const=name, help=f"run the {name} implementation")
