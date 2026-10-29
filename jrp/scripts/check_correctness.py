#!/usr/bin/env python3
"""Check that two or more implementations agree.

    source env.sh
    scripts/check_correctness.py --highs --pria-jax
    scripts/check_correctness.py --highs --cupdlpx --families jrp-medium

The first implementation named is the reference; every other is compared
against it. Exit status is 0 only if every comparison agrees.

Random instances by default. `--families` checks the benchmark's own buckets
instead, under two caps, because this is the test and not the benchmark.
`config.CHECK_MAX_ENTRIES` drops the instances too big to check, which are in
the collection to be timed rather than checked, and
`config.CHECK_MAX_INSTANCES` thins what is left to a spread across the shapes,
without which one generated bucket alone is several hundred instances.

`run_benchmark.py` is where the whole collection earns its keep; it has no cap.
`--max-entries 0` and `--limit 0` here lift either cap for the rare run that
wants it.
"""

import argparse
import sys

import _bootstrap  # noqa: F401  (sys.path)

from harness import config, instances, registry
from harness.results import disagreement


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    registry.add_backend_flags(parser)
    parser.add_argument("--families", nargs="+", default=None,
                        choices=list(instances.BENCHMARK),
                        help="check the benchmark's buckets instead of random instances")
    parser.add_argument("--max-entries", type=int, default=config.CHECK_MAX_ENTRIES,
                        help="with --families, skip anything larger; 0 means no cap")
    parser.add_argument("--limit", type=int, default=config.CHECK_MAX_INSTANCES,
                        help="with --families, run at most this many, spread over "
                             "the shapes; 0 means all of them")
    parser.add_argument("--shapes", type=int, nargs="+", metavar="ITEMS PERIODS",
                        default=[n for shape in config.CHECK_SHAPES for n in shape],
                        help="items and periods in pairs, without --families")
    parser.add_argument("--trials", type=int, default=config.CHECK_TRIALS,
                        help="random instances per shape, without --families")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="base RNG seed; trial k uses seed + k")
    parser.add_argument("--optimality-tolerance", type=float,
                        default=config.TOLERANCES.optimality,
                        help="relative duality gap requested of each solver")
    parser.add_argument("--feasibility-tolerance", type=float,
                        default=config.TOLERANCES.feasibility,
                        help="relative primal feasibility requested of each solver")
    parser.add_argument("--agreement-tolerance", type=float,
                        default=config.AGREEMENT_TOLERANCE,
                        help="relative objective gap two solvers may show and still agree")
    parser.add_argument("--time-limit", type=float, default=config.TIME_LIMIT,
                        help="per-instance wall-clock budget in seconds")
    args = parser.parse_args(argv)

    selected = args.backends or []
    if len(selected) < 2:
        parser.error("name at least two implementations, e.g. --highs --pria-jax")
    if len(args.shapes) % 2:
        parser.error("--shapes takes items and periods in pairs, so an even count")

    reference, others = selected[0], selected[1:]
    solvers = {name: registry.load(name) for name in selected}
    tolerances = config.Tolerances(optimality=args.optimality_tolerance,
                                   feasibility=args.feasibility_tolerance)

    if args.families:
        # Specs first, so `--max-entries` rejects the big generated instances
        # before anything builds them; see harness.instances.Spec.
        loaded = instances.all_specs(args.families)
        if args.max_entries:
            loaded = [i for i in loaded if i.entries <= args.max_entries]
        loaded.sort(key=lambda i: i.entries)
        if args.limit and len(loaded) > args.limit:
            # Evenly spaced over the sorted list, so the cap samples every shape
            # rather than keeping the smallest instances and calling it a pass.
            step = len(loaded) / args.limit
            loaded = [loaded[int(k * step)] for k in range(args.limit)]
        if not loaded:
            print(f"no instance of {args.families} is at most "
                  f"{args.max_entries} x variables", file=sys.stderr)
            return 2
    else:
        shapes = list(zip(args.shapes[::2], args.shapes[1::2], strict=True))
        loaded = [instances.random_spec(args.seed + trial, items, periods)
                  for items, periods in shapes for trial in range(args.trials)]

    header = (f"{'instance':>26} {'entries':>10} " +
              " ".join(f"{name:>16}" for name in selected) + f" {'verdict':>10}")
    print(header)
    print("-" * len(header))

    failures, skipped = 0, 0
    for spec in loaded:
        instance = spec.build()
        outcomes = {name: solvers[name](instance, tolerances=tolerances,
                                        time_limit=args.time_limit)
                    for name in selected}

        cells = []
        for name in selected:
            r = outcomes[name]
            cells.append(f"{r.objective:16.6f}" if r.status == "optimal"
                         else f"{r.status:>16}")

        base = outcomes[reference]
        if base.status != "optimal":
            verdict, skipped = "skip", skipped + 1
        else:
            deltas = []
            for name in others:
                other = outcomes[name]
                if other.status != "optimal":
                    continue
                deltas.append(disagreement(base, other))
            if not deltas:
                verdict, skipped = "skip", skipped + 1
            elif max(deltas) <= args.agreement_tolerance:
                verdict = "ok"
            else:
                verdict, failures = f"FAIL {max(deltas):.1e}", failures + 1

        print(f"{instance.name:>26} {instance.entries:10d} "
              + " ".join(cells) + f" {verdict:>10}")
        del instance

    total = len(loaded)
    print(f"\n{total - failures - skipped} agreed, {failures} disagreed, "
          f"{skipped} not comparable  (reference: {reference})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
