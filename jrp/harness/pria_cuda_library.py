"""In-process access to pria-cuda, through ctypes.

The backend in `harness.backends.pria_cuda` drives the file CLI, which is
right for one solve per instance but pays a process start, a CUDA context and
an `8 N T^2` file round trip each time.  A sweep that solves the same
instances hundreds of times over wants the library instead: this loads
`libpria.so`, mirrors the two structs of `pria-cuda/include/pria/pria.h`, and
calls `pria_solve_f64` directly.  Keep the struct definitions in step with the
header.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_double, c_int, c_int64
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.config import pria_cuda_lib


class PriaParameters(ctypes.Structure):
    """`pria_params` of pria.h, field for field and in its order."""

    _fields_ = [
        ("omega", c_double),
        ("eta", c_double),
        ("beta_sufficient", c_double),
        ("beta_artificial", c_double),
        ("beta_necessary", c_double),
        ("theta_initial", c_double),
        ("theta_target", c_double),
        ("theta_decay", c_double),
        ("tolerance", c_double),
        ("max_iter", c_int64),
        ("use_halpern", c_int),
        ("warm_steps", c_int),
        ("op_steps", c_int),
        ("verbose", c_int),
    ]

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name, _ in self._fields_}


#: The fields a tuning run may vary: the restart schedule and the check
#: interval, and nothing about the stopping rule or the precision.  Unlike its
#: QUBO sibling, `op_steps` here is a plain integer rather than a rule of the
#: instance size -- PRIA's interval does not read the shape, so there is no
#: rule to tune, only the number.
SCHEDULE = ("omega", "beta_sufficient", "beta_necessary", "beta_artificial",
            "theta_initial", "theta_target", "theta_decay", "op_steps")


class PriaResultInfo(ctypes.Structure):
    _fields_ = [
        ("iterations", c_int64),
        ("primal_objective", c_double),
        ("dual_objective", c_double),
        ("relative_gap", c_double),
    ]


@dataclass(frozen=True)
class Solve:
    """What one `pria_solve_f64` call returned.

    `lower` and `upper` are named for which side of the optimum they sit on:
    the library calls them `dual_objective` and `primal_objective`, which name
    the side of the saddle they came from.
    """

    status: int          #: 0 on success, a CUDA error code otherwise
    lower: float
    upper: float
    iterations: int
    gap: float


class Library:
    """`libpria.so`, loaded once; `solve` may be called any number of times."""

    def __init__(self, path: Path | None = None):
        path = path or pria_cuda_lib()
        if path is None or not Path(path).exists():
            raise RuntimeError("libpria.so not built; run scripts/build_cuda.sh pria-cuda")
        self.path = Path(path)
        self._lib = ctypes.CDLL(str(self.path))
        self._lib.pria_params_auto.restype = PriaParameters
        self._lib.pria_params_auto.argtypes = [c_int64, c_double]
        self._lib.pria_solve_f64.restype = c_int
        self._lib.pria_solve_f64.argtypes = [
            c_int64, c_int64, POINTER(c_double), c_double, POINTER(c_double),
            POINTER(PriaParameters),
            POINTER(c_double), POINTER(c_double), POINTER(c_double),
            POINTER(c_double), POINTER(c_double), POINTER(c_double),
            POINTER(PriaResultInfo),
        ]

    def parameters(self, tolerance: float, max_iter: int = -1, **overrides
                   ) -> PriaParameters:
        """`pria_params_auto(max_iter, tolerance)` with `overrides` by field name."""
        parameters = self._lib.pria_params_auto(c_int64(max_iter), c_double(tolerance))
        known = dict(PriaParameters._fields_)
        for name, value in overrides.items():
            if name not in known:
                raise KeyError(f"pria_params has no field {name!r}")
            setattr(parameters, name, value)
        return parameters

    def solve(self, instance, parameters: PriaParameters) -> Solve:
        """Solve `instance` (a `harness.instances.Instance`) in double precision.

        No solution buffers are passed; the C API takes NULL for each, and
        the tuner scores schedules on iteration counts alone.
        """
        order = np.ascontiguousarray(instance.order_costs, dtype=np.float64)
        holding = np.ascontiguousarray(instance.holding_costs, dtype=np.float64)
        info = PriaResultInfo()
        nothing = ctypes.cast(None, POINTER(c_double))
        status = self._lib.pria_solve_f64(
            c_int64(instance.items), c_int64(instance.periods),
            order.ctypes.data_as(POINTER(c_double)),
            c_double(instance.joint_order_cost),
            holding.ctypes.data_as(POINTER(c_double)),
            ctypes.byref(parameters),
            nothing, nothing, nothing, nothing, nothing, nothing,
            ctypes.byref(info))
        return Solve(status, info.dual_objective, info.primal_objective,
                     int(info.iterations), info.relative_gap)
