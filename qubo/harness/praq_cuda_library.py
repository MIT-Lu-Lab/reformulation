"""In-process access to praq-cuda, through ctypes.

The backend in `harness.backends.praq_cuda` drives the file CLI, which is right
for one solve per instance but pays a process start and a CUDA context each
time.  A sweep that solves the same instances hundreds of times over wants the
library instead: this loads `libpraq.so`, mirrors the three structs of
`praq-cuda/include/praq/praq.h`, and calls `praq_solve` directly.  Keep the
struct definitions in step with the header.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_double, c_int, c_int64
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.config import praq_cuda_lib


class PraqParameters(ctypes.Structure):
    """`PraqParameters` of praq.h, field for field."""

    _fields_ = [
        ("tolerance", c_double),
        ("max_iter", c_int64),
        ("use_float32", c_int),
        ("omega", c_double),
        ("beta_sufficient", c_double),
        ("beta_necessary", c_double),
        ("beta_artificial", c_double),
        ("theta_initial", c_double),
        ("theta_target", c_double),
        ("theta_decay", c_double),
        ("op_steps", c_int64),
        ("interval_offset", c_double),
        ("interval_scale", c_double),
        ("interval_power", c_double),
    ]

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name, _ in self._fields_}


#: The fields a tuning run may vary: everything but the stopping rule and the
#: precision.  op_steps <= 0 hands the check interval to the rule the three
#: interval_* constants define (see praq.h).
SCHEDULE = ("omega", "beta_sufficient", "beta_necessary", "beta_artificial",
            "theta_initial", "theta_target", "theta_decay", "op_steps",
            "interval_offset", "interval_scale", "interval_power")


class PraqInstance(ctypes.Structure):
    _fields_ = [
        ("n", c_int64),
        ("m", c_int64),
        ("diag", POINTER(c_double)),
        ("rows", POINTER(c_int64)),
        ("cols", POINTER(c_int64)),
        ("vals", POINTER(c_double)),
    ]


class PraqResult(ctypes.Structure):
    _fields_ = [
        ("lower", c_double),
        ("upper", c_double),
        ("iterations", c_int64),
        ("status", c_int),
    ]


@dataclass(frozen=True)
class Solve:
    """What one `praq_solve` call returned."""

    status: int          #: 0 converged, 1 iteration limit, < 0 error
    lower: float
    upper: float
    iterations: int
    x: np.ndarray

    @property
    def gap(self) -> float:
        return (self.upper - self.lower) / (1.0 + abs(self.upper) + abs(self.lower))


class Library:
    """`libpraq.so`, loaded once; `solve` may be called any number of times."""

    def __init__(self, path: Path | None = None):
        path = path or praq_cuda_lib()
        if path is None or not Path(path).exists():
            raise RuntimeError("libpraq.so not built; run scripts/build_cuda.sh praq-cuda")
        self.path = Path(path)
        self._lib = ctypes.CDLL(str(self.path))
        self._lib.praq_default_parameters.restype = PraqParameters
        self._lib.praq_default_parameters.argtypes = []
        self._lib.praq_solve.restype = c_int
        self._lib.praq_solve.argtypes = [POINTER(PraqInstance), POINTER(PraqParameters),
                                         POINTER(c_double), POINTER(PraqResult)]
        self._lib.praq_check_interval.restype = c_int64
        self._lib.praq_check_interval.argtypes = [c_int64, POINTER(PraqParameters)]

    def parameters(self, **overrides) -> PraqParameters:
        """`praq_default_parameters()` with `overrides` applied, by field name."""
        parameters = self._lib.praq_default_parameters()
        for name, value in overrides.items():
            if name not in dict(PraqParameters._fields_):
                raise KeyError(f"PraqParameters has no field {name!r}")
            setattr(parameters, name, value)
        return parameters

    def check_interval(self, n: int, parameters: PraqParameters) -> int:
        """The check interval `praq_solve` will use for `n` variables: op_steps
        itself when positive, otherwise the rule's value (praq_check_interval)."""
        if parameters.op_steps > 0:
            return max(int(parameters.op_steps), 2)
        return int(self._lib.praq_check_interval(n, ctypes.byref(parameters)))

    def solve(self, instance, parameters: PraqParameters) -> Solve:
        """Solve `instance` (a `harness.instances.Instance`) in place."""
        diag = np.ascontiguousarray(instance.diag, dtype=np.float64)
        rows = np.ascontiguousarray(instance.rows, dtype=np.int64)
        cols = np.ascontiguousarray(instance.cols, dtype=np.int64)
        vals = np.ascontiguousarray(instance.vals, dtype=np.float64)
        view = PraqInstance(instance.n, instance.edges,
                            diag.ctypes.data_as(POINTER(c_double)),
                            rows.ctypes.data_as(POINTER(c_int64)),
                            cols.ctypes.data_as(POINTER(c_int64)),
                            vals.ctypes.data_as(POINTER(c_double)))
        x = np.empty(instance.n, dtype=np.float64)
        result = PraqResult()
        status = self._lib.praq_solve(ctypes.byref(view), ctypes.byref(parameters),
                                      x.ctypes.data_as(POINTER(c_double)),
                                      ctypes.byref(result))
        return Solve(status, result.lower, result.upper, int(result.iterations), x)
