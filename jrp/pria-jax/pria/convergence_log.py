from collections import defaultdict, namedtuple
from enum import IntEnum
from queue import Queue

from jax import Array
from jax.debug import callback

_convergence_log_queue = Queue()

ObjectiveBound = namedtuple("ObjectiveBound", ["objective", "bound", "iteration"])

class ConvergenceLogEntry:
    def __init__(self, iid: int, iteration: int, obj: Array, bound: Array):
        self.iid = int(iid)
        self.iteration = int(iteration)
        self.obj = float(obj)
        self.bound = float(bound)


def _log_convergence_iteration(iid: int, iteration: int, obj: Array, bound: Array):
    entry = ConvergenceLogEntry(iid, iteration, obj, bound)
    _convergence_log_queue.put(entry)


def log_convergence_iteration(iid: int, iteration: int, obj: Array, bound: Array):
    callback(_log_convergence_iteration, iid, iteration, obj, bound)


def get_convergence_logs() -> list[list[ObjectiveBound]]:
    log_queue = list(_convergence_log_queue.queue)
    iid_to_entry = defaultdict(list)

    for entry in log_queue:
        iid_to_entry[entry.iid].append((entry.iteration, entry.obj, entry.bound))

    for iid in iid_to_entry:
        dedup_iterations = {t[0]: (t[1], t[2]) for t in iid_to_entry[iid]}
        dedup_iterations = [(k, t[0], t[1]) for k, t in dedup_iterations.items()]
        dedup_iterations.sort(key=lambda x: x[0])

        iid_to_entry[iid] = [ObjectiveBound(i[1], i[2], i[0]) for i in dedup_iterations]

    _convergence_log_queue.queue.clear()
    items = sorted(iid_to_entry.items(), key=lambda x: x[0])
    return [i[1] for i in items]


class RestartType(IntEnum):
    NoRestart = 0
    Artificial = 1
    Necessary = 2
    Sufficient = 3


_restart_log_queue = Queue()

NormRestart = namedtuple("NormRestart", ["norm", "restart", "omega", "iteration"])

class RestartLogEntry:
    def __init__(self, iid: int, iteration: int, canonical_norm: float,
                 restart_type: RestartType, omega: float):
        self.iid = int(iid)
        self.iteration = int(iteration)
        self.canonical_norm = float(canonical_norm)
        self.restart_type = RestartType(int(restart_type))
        self.omega = float(omega)


def _log_restart_iteration(iid: int, iteration: int, canonical_norm: float,
                           restart_type: RestartType, omega: float):
    entry = RestartLogEntry(iid, iteration, canonical_norm, restart_type, omega)
    _restart_log_queue.put(entry)


def log_restart_iteration(iid: int, iteration: int, canonical_norm: float,
                          restart_type: RestartType, omega: float):
    callback(_log_restart_iteration, iid, iteration, canonical_norm, restart_type, omega)


def get_restart_logs() -> list[list[NormRestart]]:
    log_queue = list(_restart_log_queue.queue)
    iid_to_entry = defaultdict(list)

    for entry in log_queue:
        iid_to_entry[entry.iid].append((entry.iteration, entry.canonical_norm,
                                        entry.restart_type, entry.omega))

    for iid in iid_to_entry:
        dedup_iterations = {t[0]: (t[1], t[2], t[3]) for t in iid_to_entry[iid]}
        dedup_iterations = [(k, t[0], t[1], t[2]) for k, t in dedup_iterations.items()]
        dedup_iterations.sort(key=lambda x: x[0])

        iid_to_entry[iid] = [NormRestart(i[1], i[2], i[3], i[0]) for i in dedup_iterations]

    _restart_log_queue.queue.clear()
    items = sorted(iid_to_entry.items(), key=lambda x: x[0])
    return [i[1] for i in items]
