"""Peak memory of a solve, sampled while it runs.

Not `resource.getrusage`.  Its `ru_maxrss` is a high-water mark for a whole
process, and five of the six backends run inside this one, so the mark a solve
leaves cannot be told from the mark the solve before it left -- and it is a
maximum, so it cannot be differenced either.  Sampling is what is left: read the
process's current footprint often enough, keep the largest reading.

What that costs is honest and has to be said: a peak shorter than the interval
between readings is missed entirely, so a solve of a few milliseconds gets a
number that means nothing.  `config.MEMORY_MIN_SOLVE_SECONDS` is how many
samples a solve has to span before the figures will believe its peak, and the
table drops everything under it rather than plotting noise.

Sampling costs the run it samples -- a thread waking every few milliseconds, and
`/proc` and NVML reads that are cheap but not free -- so it is off unless asked
for, and a run that asks for it is a memory run whose timings are not the
project's timings.  `harness.figures` keeps the two apart by construction: a row
carrying a peak is used by the memory table and by nothing else.
"""

from __future__ import annotations

import contextlib
import math
import os
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class Peak:
    """The largest host and device footprints seen during one solve.

    NaN means not measured rather than zero: no `/proc` on this platform, no
    NVML, or a process NVML does not list because it never touched the device.
    """

    host_bytes: float = math.nan
    device_bytes: float = math.nan


def host_bytes(pid: int) -> float:
    """Resident set size of `pid` right now, in bytes, or NaN.

    From `/proc/<pid>/statm`, which is Linux only -- where the benchmark runs.
    Elsewhere this returns NaN and the column stays empty, rather than the
    harness growing a dependency to sample a platform it is not measured on.
    """
    try:
        with open(f"/proc/{pid}/statm") as handle:
            resident = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return math.nan
    return float(resident * os.sysconf("SC_PAGE_SIZE"))


#: NVML, once: the module and this machine's device handles after a successful
#: `nvmlInit`, or False once it has failed.  Initialising is not free and a
#: sampler asks hundreds of times a second, so the answer -- including "there is
#: no NVML here", which is the answer on a login node and on a laptop -- is
#: worked out once and kept.
_NVML: object = None


def _nvml():
    """`(module, handles)` if NVML is usable here, else None."""
    global _NVML
    if _NVML is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML = (pynvml, [pynvml.nvmlDeviceGetHandleByIndex(index)
                              for index in range(pynvml.nvmlDeviceGetCount())])
        except Exception:   # no module, no driver, no permission, MIG; none fatal
            _NVML = False
    return _NVML or None


def device_bytes(pid: int) -> float:
    """GPU memory `pid` holds right now, summed over devices, in bytes, or NaN.

    Per process and not per device, which is the whole reason for NVML here:
    `cudaMemGetInfo` would report the card, and on a shared node the card is
    carrying other people's work.  A process NVML does not list is holding none,
    but it is reported as NaN rather than zero -- before the first allocation the
    two are indistinguishable from out here.
    """
    nvml = _nvml()
    if nvml is None:
        return math.nan
    module, handles = nvml
    try:
        total, found = 0, False
        for handle in handles:
            for process in module.nvmlDeviceGetComputeRunningProcesses(handle):
                if process.pid == pid and process.usedGpuMemory is not None:
                    total, found = total + int(process.usedGpuMemory), True
    except Exception:
        return math.nan
    return float(total) if found else math.nan


class Sampler:
    """A thread that watches one process's footprint until told to stop.

    `probe` is the pair of readings to take, injectable so the sampling itself
    can be tested without a GPU or a `/proc`.  Readings that come back NaN stay
    NaN in the result: an unmeasurable footprint must not read as a small one.
    """

    def __init__(self, pid: int, interval: float, probe=None):
        self.pid, self.interval = pid, interval
        self.probe = probe or (lambda pid: Peak(host_bytes(pid), device_bytes(pid)))
        self.peak = Peak()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-sampler",
                                        daemon=True)

    def _run(self) -> None:
        while True:
            reading = self.probe(self.pid)
            self.peak = Peak(_larger(self.peak.host_bytes, reading.host_bytes),
                             _larger(self.peak.device_bytes, reading.device_bytes))
            if self._stop.wait(self.interval):
                return

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval * 10 + 1.0)


def _larger(seen: float, reading: float) -> float:
    """The larger of two readings, where NaN means "nothing seen yet"."""
    if math.isnan(reading):
        return seen
    return reading if math.isnan(seen) else max(seen, reading)


@contextlib.contextmanager
def watch(pid: int, interval: float, *, enabled: bool, probe=None):
    """A running `Sampler` over `pid`, or a dormant one if `enabled` is false.

    So that a backend has one shape for both cases and no branch around its
    solve: the dormant sampler is never started, so its peak stays NaN and the
    solve it did not watch costs nothing.
    """
    sampler = Sampler(pid, interval, probe=probe)
    if not enabled:
        yield sampler
        return
    with sampler:
        yield sampler
