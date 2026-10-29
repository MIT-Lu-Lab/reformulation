#!/usr/bin/env python3
"""Run one or more implementations over the benchmark sets and record timings.

    source env.sh
    scripts/run_benchmark.py --praq-jax
    scripts/run_benchmark.py --praq-jax --gurobi --families hard-medium

Writes one CSV per run into $PRAQ_RESULTS_DIR, and prints a line per instance as
it goes so a long run is watchable.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path)

from harness import config, instances, registry, results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    registry.add_backend_flags(parser)
    parser.add_argument("--families", nargs="+", default=list(instances.BENCHMARK),
                        choices=list(instances.BENCHMARK),
                        help="size buckets to run; the default is all of them")
    parser.add_argument("--optimality-tolerance", type=float,
                        default=config.TOLERANCES.optimality,
                        help="relative duality gap requested of each solver")
    parser.add_argument("--feasibility-tolerance", type=float,
                        default=config.TOLERANCES.feasibility,
                        help="relative primal feasibility requested of each solver")
    parser.add_argument("--time-limit", type=float, default=config.TIME_LIMIT,
                        help="per-instance wall-clock budget in seconds")
    parser.add_argument("--max-n", type=int, default=None,
                        help="skip instances with more variables than this")
    parser.add_argument("--max-edges", type=int, default=None,
                        help="skip instances with more edges than this")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many instances, for a smoke test")
    parser.add_argument("--memory", action="store_true",
                        help="sample peak memory during each solve.  This makes it a "
                             "memory run: sampling costs the solve it watches, so the "
                             "timings it records are not the project's timings and "
                             "nothing plots them.  See harness/memory.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV path; default is a timestamped file in $PRAQ_RESULTS_DIR")
    args = parser.parse_args(argv)

    selected = args.backends or []
    if not selected:
        parser.error("name at least one implementation, e.g. --praq-jax")
    solvers = {name: registry.load(name) for name in selected}
    sampling = [name for name in selected if name in registry.SAMPLED_MEMORY]
    if args.memory:
        if not sampling:
            parser.error(f"--memory needs one of {', '.join(registry.SAMPLED_MEMORY)}; "
                         f"the others cannot measure their own")
        deaf = [name for name in selected if name not in sampling]
        if deaf:
            print(f"note: --memory does nothing for {', '.join(deaf)}, which leave the "
                  f"peak columns empty; see harness/registry.py\n")
    tolerances = config.Tolerances(optimality=args.optimality_tolerance,
                                   feasibility=args.feasibility_tolerance)

    # Specs, not instances: a pass is selected and ordered before anything is
    # built, and each instance is built when its turn comes and dropped after,
    # so a run costs one instance of memory rather than all of them.
    loaded = instances.all_specs(args.families)
    if args.max_n is not None:
        loaded = [i for i in loaded if i.n <= args.max_n]
    if args.max_edges is not None:
        loaded = [i for i in loaded if i.edges <= args.max_edges]
    loaded.sort(key=lambda i: (i.family, i.edges))
    if args.limit is not None:
        loaded = loaded[:args.limit]

    if not loaded:
        print(f"nothing to run: no instance of {args.families} survived "
              f"--max-n/--max-edges", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # "memory" in the name because a memory run's timings are not comparable
    # with a plain run's, and the two should not be mistaken for each other on
    # disk.  What keeps them apart in the figures is the peak columns themselves.
    kind = "benchmark-memory" if args.memory else "benchmark"
    path = args.output or (config.results_dir()
                           / f"{kind}-{'-'.join(selected)}-{stamp}.csv")

    print(f"{len(loaded)} instances x {len(selected)} implementations -> {path}\n")
    header = f"{'instance':>22} {'n':>7} {'edges':>10} {'backend':>10} {'status':>16} {'objective':>18} {'iters':>10} {'seconds':>10}"
    print(header)
    print("-" * len(header))

    records, started = [], time.perf_counter()
    for spec in loaded:
        instance = spec.build()
        for name in selected:
            extra = {"sample_memory": True} if args.memory and name in sampling else {}
            record = solvers[name](instance, tolerances=tolerances,
                                   time_limit=args.time_limit, **extra)
            records.append(record)
            print(f"{instance.name:>22} {instance.n:7d} {instance.edges:10d} "
                  f"{name:>10} {record.status:>16} {record.objective:18.6f} "
                  f"{record.iterations:10d} {record.solve_seconds:10.3f}", flush=True)
            results.write_csv(path, records)   # rewritten each step: a kill keeps the data
        del instance

    solved = sum(1 for r in records if r.status == "optimal")
    print(f"\n{solved}/{len(records)} solved in {time.perf_counter() - started:.1f}s")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
