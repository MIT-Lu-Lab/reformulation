"""Run the CUDA PRAQ implementation on one instance.

praq-cuda is driven as a separate program, `praq_solve_file`, rather than linked
as a library: the instance goes out as a binary file, the binary runs, and the
result comes back the same way.  The two formats are the contract between this
module and `praq-cuda/tools/praq_solve_file.cu`, defined at both ends; keep the
two in step.
"""

from __future__ import annotations

import math
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from harness import memory
from harness.config import (MEMORY_SAMPLE_SECONDS, Tolerances, praq_cuda_bin,
                            results_dir)
from harness.results import Result

#: Binary instance file written for the CLI, little-endian:
#:     int64  magic = 0x51415250 ("PRAQ")
#:     int64  n, int64 m
#:     double diag[n]
#:     int64  rows[m], int64 cols[m], double vals[m]
#: and the result file it writes back:
#:     int64  n, int64 iterations, int64 status
#:     double lower, double upper
#:     double x[n]
MAGIC = 0x51415250

#: `PraqResult.status`, as the result file carries it, onto `harness.results.STATUSES`.
STATUS = {0: "optimal", 1: "iteration_limit"}

#: How long to wait past `time_limit` before killing the process.  The solver
#: has no clock of its own -- like praq-jax, its only budget is the iteration
#: cap -- so this is what enforces the limit, and it has to cover CUDA context
#: creation and the file round trip on top.
LAUNCH_GRACE = 60.0


def write_instance(instance, path: Path) -> None:
    """Write `instance` in the CLI's format."""
    with open(path, "wb") as handle:
        handle.write(np.array([MAGIC, instance.n, instance.edges], dtype="<i8").tobytes())
        handle.write(np.ascontiguousarray(instance.diag, dtype="<f8").tobytes())
        handle.write(np.ascontiguousarray(instance.rows, dtype="<i8").tobytes())
        handle.write(np.ascontiguousarray(instance.cols, dtype="<i8").tobytes())
        handle.write(np.ascontiguousarray(instance.vals, dtype="<f8").tobytes())


def read_result(path: Path) -> tuple[int, int, float, float, np.ndarray]:
    """Read a result file back as (iterations, status, lower, upper, x)."""
    data = path.read_bytes()
    n, iterations, status = struct.unpack_from("<qqq", data, 0)
    lower, upper = struct.unpack_from("<dd", data, 24)
    x = np.frombuffer(data, dtype="<f8", count=n, offset=40)
    return iterations, status, lower, upper, x


def _timings(stdout: str) -> dict[str, float]:
    """The `*_seconds: value` lines the CLI prints."""
    out = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().endswith("_seconds"):
            try:
                out[key.strip()] = float(value)
            except ValueError:
                pass
    return out


def solve(instance, *, tolerances: Tolerances, time_limit: float,
          sample_memory: bool = False) -> Result:
    """Solve the relaxation of `instance` with praq-cuda, or report why it cannot.

    Same contract as praq-jax: only `tolerances.optimality` reaches the solver,
    as the relative duality gap it stops on, and the feasibility column stays
    NaN because the iterates are feasible by construction.

    `solve_seconds` is the `praq_solve` call as the CLI times it, which includes
    the upload and the download but not CUDA context creation; that, and
    reading the instance file, is the CLI's `setup_seconds`.  The whole
    subprocess's wall clock goes in `note`.

    `sample_memory` watches the child and records its peak.  Being a child, its
    host figure is the solver's own and excludes the instance this process is
    holding, which an in-process backend's does not -- so the host column does
    not compare across the two and the device column does.  Sampling costs the
    solve it watches: see `harness.memory` for why a memory run is a run of its
    own.
    """
    common = dict(backend="praq-cuda", instance=instance.name,
                  family=instance.family, n=instance.n, edges=instance.edges,
                  optimality_tolerance=tolerances.optimality)

    binary = praq_cuda_bin()
    if binary is None:
        return Result(status="unavailable",
                      note="praq_solve_file not built; run scripts/build_cuda.sh",
                      **common)

    limited = math.isfinite(time_limit)
    # On the results volume rather than /tmp, and gone with the call; what a
    # killed run leaves behind goes with `clean.sh`.
    staging = results_dir() / "praq-cuda"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging, prefix="run-") as directory:
        instance_path = Path(directory) / "instance.bin"
        result_path = Path(directory) / "result.bin"
        write_instance(instance, instance_path)

        command = [str(binary), str(instance_path), str(result_path),
                   str(tolerances.optimality)]
        start = time.perf_counter()
        # Popen rather than run(), because a memory run needs the child's pid to
        # watch.  The timeout behaves as run()'s does: kill, then reap, so a
        # solver that ignores the limit cannot leave a process behind.
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
        except OSError as error:
            return Result(status="error", note=str(error)[:200], **common)
        try:
            with memory.watch(process.pid, MEMORY_SAMPLE_SECONDS,
                              enabled=sample_memory) as sampler:
                stdout, stderr = process.communicate(
                    timeout=time_limit + LAUNCH_GRACE if limited else None)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return Result(status="time_limit", solve_seconds=time.perf_counter() - start,
                          note=f"killed {LAUNCH_GRACE:g}s past the {time_limit:g}s limit",
                          **common)
        wall_seconds = time.perf_counter() - start

        if not result_path.exists():
            tail = (stderr or stdout or "").strip().rsplit("\n", 1)[-1]
            return Result(status="error",
                          note=f"exit {process.returncode}, no result: {tail}"[:200],
                          **common)
        iterations, code, lower, upper, _ = read_result(result_path)

    timings = _timings(stdout)
    timed = dict(iterations=iterations,
                 setup_seconds=timings.get("setup_seconds", math.nan),
                 solve_seconds=timings.get("solve_seconds", math.nan),
                 peak_host_bytes=sampler.peak.host_bytes,
                 peak_device_bytes=sampler.peak.device_bytes,
                 **common)
    status = STATUS.get(code, "error")
    note = f"wall {wall_seconds:.3g}s"
    if status != "optimal":
        # The bounds are still a true bracket -- both iterates are feasible at
        # every step -- so they are recorded; `objective` is not, since it
        # means the relaxation's optimal value and this is not it.
        gap = (upper - lower) / (1.0 + abs(upper) + abs(lower))
        return Result(status=status, lower=lower, upper=upper,
                      note=f"status {code}, relative gap {gap:.3g}; {note}"[:200], **timed)
    return Result(status=status, objective=upper, lower=lower, upper=upper,
                  note=note, **timed)
