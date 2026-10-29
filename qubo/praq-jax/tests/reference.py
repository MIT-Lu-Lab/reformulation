"""Test-only access to the LP baselines.

The `praq` package must not depend on the baselines; its tests may.  This shim
puts the experiment repository's root on `sys.path` and exposes the HiGHS
reference in the terms the tests want, so the dependency direction stays
one-way: tests -> baselines -> nothing in praq.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines import highs_backend           # noqa: E402
from harness.instances import Instance        # noqa: E402


def reference_relaxation_value(instance) -> float:
    """The LP relaxation optimum of a `QuboProblem`, from HiGHS."""
    return highs_backend.relaxation_value(
        Instance.from_dense(instance.matrix, name="test"))
