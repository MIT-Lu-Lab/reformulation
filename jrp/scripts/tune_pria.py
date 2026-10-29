#!/usr/bin/env python3
"""Tune PRIA's restart schedule on the benchmark's distribution, with pria-cuda.

    source env.sh
    scripts/tune_pria.py                                    # every bucket, 200 configurations
    scripts/tune_pria.py --buckets jrp-medium --configs 50  # a quick look
    scripts/tune_pria.py --configs 1000000 --time-budget 10800   # what the batch job runs
    scripts/tune_pria.py --warm-start results/tuning/tune-<earlier>.csv

Optuna's multivariate TPE searches the schedule: the three restart ratios, the
theta smoothing of the primal weight, the initial weight, and the interval
between restart-and-termination checks.  The score is the cost of reaching the
optimality tolerance, in iterations plus `--check-charge` per check -- a check
is two segmented reductions and a synchronous copy, so an iteration count
alone would price a small interval at nothing.  Iterations and not seconds,
because they are deterministic and do not depend on the GPU.

**Scoring.**  Every configuration is scored on its own draw.  Each shape of
the ladder has a pool of seeds (`--pool-factor` times its draw, at least
four), each configuration draws a fresh random subset of every pool, and a
pool instance's baseline is run the first time it is drawn and remembered.  A
configuration therefore cannot be selected for suiting particular instances.
Each instance's ratio to the baseline is weighted by its x-variable count to
`--size-power`, split over the instances at that shape, so a shape's weight
does not depend on how many seeds it has.  The score is the weighted
geometric mean; the worst case is kept beside it.  A run past `--cap-factor`
times the baseline's iterations is cut off and counted at the cap.

Configurations are evaluated bucket by bucket, cheapest first, and abandoned
after the first if its score there is already `--abandon` times the baseline.

**Selection.**  The winner is chosen on the hold-out, not on the training
score, under which the best value belongs to the luckiest trial.  The
`--leaders` best are re-scored on a fixed hold-out drawn with disjoint seeds
and ranked by that, and a leader worse than `--max-size-ratio` at any one
shape is ranked behind those that are not.  The best `--confirm` of those are
scored once on the benchmark itself, so the last number is neither fitted nor
selected on.

**Warm starts.**  `--warm-start` seeds the sampler with everything an earlier
run scored, abandoned configurations included, rescored under the current
charge and weights from the per-instance columns of its CSV.  It also
consumes the start-up phase, which counts every trial in the study, so a warm
start continues a search; a search starts cold.

Results go to a CSV under $PRIA_RESULTS_DIR/tuning, rewritten after every
configuration, with a column per pool instance holding the iteration count
where the configuration drew it.

The winner is not applied automatically.  Its values belong in two places that
must move together: `Parameters.with_auto_step` in pria-jax/pria/solver.py and
`pria_params_auto` in pria-cuda/src/pria.cu.
"""

import argparse
import csv
import math
import sys
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path)
import numpy as np
import optuna

from harness import config, instances, results
from harness.pria_cuda_library import SCHEDULE, Library

#: Seed offsets of the two draws that must not overlap the benchmark's seeds,
#: which start at 0 and stay far below these.
TRAINING_SEEDS = 10_000
HOLDOUT_SEEDS = 20_000

#: The search box: omega in log scale, the rest linear, and op_steps an
#: integer.  Centred on the current values and wide enough that the edges are
#: visibly bad.  beta_necessary is searched as a fraction of the room above
#: beta_sufficient, since below it the necessary test never fires: every
#: proposal is then feasible and the box the sampler learns on is fixed.
#: op_steps reaches down to the solver's floor of 2; a check is two segmented
#: reductions over the whole variable, so a small interval is not free and
#: `--check-charge` is what prices it.
SPACE = {
    "omega": (0.01, 10.0),
    "beta_sufficient": (0.15, 0.8),
    "beta_necessary": (0.3, 0.95),
    "beta_artificial": (0.05, 0.6),
    "theta_initial": (0.3, 1.0),
    "theta_target": (0.1, 1.0),
    "theta_decay": (0.5, 0.99),
    "op_steps": (2, 64),
}

#: The one field of the box that is an integer rather than a float.
INTEGER = ("op_steps",)


def searched() -> list[str]:
    """The names the sampler proposes: the box less beta_necessary, whose
    fraction stands in for it."""
    return [name for name in SPACE if name != "beta_necessary"]


def distributions() -> dict:
    """What the sampler sees: the searched names' boxes plus the fraction."""
    out = {}
    for name in searched():
        lo, hi = SPACE[name]
        if name in INTEGER:
            out[name] = optuna.distributions.IntDistribution(int(lo), int(hi))
        else:
            out[name] = optuna.distributions.FloatDistribution(lo, hi,
                                                               log=(name == "omega"))
    out["beta_necessary_frac"] = optuna.distributions.FloatDistribution(0.0, 1.0)
    return out


def necessary_room(beta_sufficient: float) -> tuple[float, float]:
    """The interval beta_necessary is drawn from, given beta_sufficient."""
    lo, hi = SPACE["beta_necessary"]
    return max(lo, beta_sufficient), hi


def to_params(schedule: dict) -> dict:
    """A schedule in the sampler's coordinates (the inverse of `suggest`)."""
    params = {name: (int(schedule[name]) if name in INTEGER else float(schedule[name]))
              for name in searched()}
    lo, hi = necessary_room(params["beta_sufficient"])
    fraction = (float(schedule["beta_necessary"]) - lo) / (hi - lo)
    params["beta_necessary_frac"] = min(max(fraction, 0.0), 1.0)
    return params


def suggest(trial: optuna.Trial) -> dict:
    """One point of the search box, proposed by the study."""
    point = {}
    for name in searched():
        lo, hi = SPACE[name]
        if name in INTEGER:
            point[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            point[name] = trial.suggest_float(name, lo, hi, log=(name == "omega"))
    lo, hi = necessary_room(point["beta_sufficient"])
    point["beta_necessary"] = lo + (hi - lo) * trial.suggest_float(
        "beta_necessary_frac", 0.0, 1.0)
    return point


def shape_of(column: str) -> tuple[int, int]:
    """The (items, periods) in a column name such as jrp-huge-n350-t450-s10001."""
    items = int(column.split("-n")[1].split("-")[0])
    periods = int(column.split("-t")[1].split("-")[0])
    return items, periods


def cost_of(count: float, op_steps: float, charge: float) -> float:
    """What a run costs: its iterations, plus `charge` per check it made.

    One home, because the baseline, the search, the hold-out and a warm
    start's rescoring all have to price a check the same way or their ratios
    are not comparable.
    """
    return count * (1.0 + charge / max(int(op_steps), 2))


def weighted_geomean(ratios, weights) -> float:
    ratios, weights = np.asarray(ratios), np.asarray(weights)
    return float(np.exp(np.sum(weights * np.log(ratios)) / np.sum(weights)))


def rescore(row: dict, base: dict, columns: list[str], charge: float,
            size_power: float) -> float:
    """A CSV row's score under the current objective, from its per-instance
    iteration counts (the columns present: a row holds only the instances its
    configuration drew, and an abandoned row fewer).  The earlier run's
    instances and baseline, not this run's, but the same charge and weights,
    so its scores mean what this run's do."""
    present = [c for c in columns if row[c] != "" and base[c] != ""]
    sizes = {c: instances.entries(*shape_of(c)) for c in present}
    counts = Counter(sizes.values())
    weights = [sizes[c] ** size_power / counts[sizes[c]] for c in present]
    ratios = [cost_of(float(row[c]), row["op_steps"], charge)
              / cost_of(float(base[c]), base["op_steps"], charge) for c in present]
    return weighted_geomean(ratios, weights)


def warm_start(study: optuna.Study, path: Path, charge: float,
               size_power: float) -> int:
    """Add an earlier run's CSV to the study: scored rows as complete trials,
    abandoned ones as pruned with their first-bucket score, the baseline
    skipped, every score recomputed under the current objective from the
    per-instance columns.  Returns how many were added."""
    added = 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in SPACE if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path} predates the search box: no {', '.join(missing)} column")
        columns = [c for c in reader.fieldnames if c.startswith("jrp-")]
        rows = list(reader)
    base = next(r for r in rows if r["config"] == "0")
    space = distributions()
    for row in rows:
        if row["config"] == "0":
            continue
        params = to_params(row)
        value = rescore(row, base, columns, charge, size_power)
        if int(row["abandoned"]):
            trial = optuna.trial.create_trial(
                params=params, distributions=space,
                state=optuna.trial.TrialState.PRUNED, intermediate_values={0: value})
        else:
            trial = optuna.trial.create_trial(params=params, distributions=space,
                                              value=value)
        study.add_trial(trial)
        added += 1
    return added


@dataclass(frozen=True)
class Rung:
    family: str     #: the benchmark bucket it belongs to
    items: int
    periods: int
    draw: int       #: instances per configuration
    pool: int       #: seeds those are drawn from

    @property
    def entries(self) -> int:
        return instances.entries(self.items, self.periods)


def draws_at(entries: int) -> int:
    """Instances per configuration at a shape of this size: many where they
    are cheap, two where one instance is seconds of GPU and as long again to
    generate, so that no rung is one instance's luck."""
    return 6 if entries <= 1_000_000 else 3 if entries <= 10_000_000 else 2


def rungs_for(buckets: list[str], pool_factor: float) -> list[Rung]:
    """The training shapes for `buckets`, cheapest first."""
    out = []
    for bucket in instances.LADDER:
        if bucket not in buckets:
            continue
        for items, periods, _ in instances.LADDER[bucket]:
            draw = draws_at(instances.entries(items, periods))
            out.append(Rung(bucket, items, periods, draw,
                            max(4, math.ceil(pool_factor * draw))))
    return sorted(out, key=lambda r: r.entries)


@dataclass(frozen=True)
class Draw:
    """Which instance to build, and nothing built.

    This quadruple identifies an instance everywhere here: the pool keys on
    it, the hold-out is a list of them, and the CSV's per-instance columns are
    its `name`.  It travels as one value so that the pool's key, the column
    heading and the instance's own name cannot drift apart -- a warm start
    matches old runs by that name, so they have to agree exactly.
    """

    family: str
    items: int
    periods: int
    seed: int

    @property
    def name(self) -> str:
        return f"{self.family}-n{self.items}-t{self.periods}-s{self.seed}"

    @property
    def entries(self) -> int:
        """The size `size_weights` weights by.

        Named to match `instances.Spec.entries`, because `size_weights` is
        handed Draws during the search and Specs at the confirm step and has
        to weight both the same way.
        """
        return instances.entries(self.items, self.periods)


class Pool:
    """Instances by `Draw`, generated on first use and kept while they fit in
    `budget` bytes, the least recently used going first.

    Eviction only reclaims what the caller has let go of, so the budget is a
    promise the caller keeps: consume `stream` one instance at a time and drop
    each before asking for the next.  `list(pool.stream(...))` holds hard
    references that eviction cannot reclaim, and a JRP instance is `8 N T^2`
    bytes -- over half a gigabyte at the top of the ladder.
    """

    def __init__(self, budget: float):
        self.budget = budget
        self.bytes = 0
        self.cache: OrderedDict[Draw, tuple[instances.Instance, int]] = OrderedDict()

    def stream(self, draws: Iterable[Draw]) -> Iterator[instances.Instance]:
        """Each draw's instance in turn, built or taken from the cache.

        Lazy on purpose: consume it in a loop and let each instance go before
        asking for the next.  Do not collect the result.
        """
        for draw in draws:
            yield self._get(draw)

    def _get(self, draw: Draw) -> instances.Instance:
        if draw in self.cache:
            self.cache.move_to_end(draw)
            return self.cache[draw][0]
        instance = instances.random_jrp(draw.seed, draw.items, draw.periods,
                                        name=draw.name, family=draw.family)
        size = instance.holding_costs.nbytes + instance.order_costs.nbytes
        while self.cache and self.bytes + size > self.budget:
            _, (_, old) = self.cache.popitem(last=False)
            self.bytes -= old
        self.cache[draw] = (instance, size)
        self.bytes += size
        return instance


def training_draw(rungs: list[Rung], rng: np.random.Generator) -> list[Draw]:
    """One configuration's draws: `draw` seeds of each rung's pool, at random
    without replacement, cheapest rung first."""
    out = []
    for rung in rungs:
        for k in sorted(rng.choice(rung.pool, size=rung.draw, replace=False).tolist()):
            out.append(Draw(rung.family, rung.items, rung.periods, TRAINING_SEEDS + k))
    return out


def holdout_plan(rungs: list[Rung], factor: float) -> list[Draw]:
    """The fixed selection set as `Draw`s, building nothing.

    The caller streams it, scoring every leader on each instance before
    letting go -- see `Pool` for why, and for fewer builds than one pass per
    leader would cost.
    """
    return [Draw(rung.family, rung.items, rung.periods, HOLDOUT_SEEDS + k)
            for rung in rungs
            for k in range(max(1, math.ceil(factor * rung.draw)))]


def size_weights(drawn, power: float = 0.5) -> np.ndarray:
    """Each instance's weight: its x-variable count to `power`, split over the
    instances at that size, so a shape's total weight is `entries**power`
    however many seeds it has."""
    sizes = [d.entries for d in drawn]      # Draw or Spec; both carry it
    counts = Counter(sizes)
    return np.array([size ** power / counts[size] for size in sizes])


class Baseline:
    """The default schedule's iterations and cost on every instance it has
    met, run on first request and remembered by name."""

    def __init__(self, library: Library, schedule: dict, tolerance: float,
                 charge: float):
        self.library, self.schedule = library, schedule
        self.tolerance, self.charge = tolerance, charge
        self.counts: dict[str, int] = {}
        self.costs: dict[str, float] = {}
        self.seconds = 0.0

    def __call__(self, instance: instances.Instance) -> tuple[int, float]:
        if instance.name not in self.counts:
            start = time.perf_counter()
            parameters = self.library.parameters(tolerance=self.tolerance, max_iter=-1,
                                                 **self.schedule)
            solve = self.library.solve(instance, parameters)
            if solve.status != 0:
                raise RuntimeError(f"pria_solve failed with status {solve.status} "
                                   f"on {instance.name}")
            self.counts[instance.name] = solve.iterations
            self.costs[instance.name] = cost_of(solve.iterations,
                                                self.schedule["op_steps"], self.charge)
            self.seconds += time.perf_counter() - start
        return self.counts[instance.name], self.costs[instance.name]


@dataclass
class Score:
    geomean: float          #: weighted geometric mean of cost / baseline cost
    worst: float            #: the largest such ratio
    capped: int             #: instances cut off at the cap
    abandoned: bool         #: stopped after the first bucket; geomean is that bucket's
    iterations: int         #: total iterations over what was run
    seconds: float          #: the configuration's own time, baseline runs excluded


def evaluate(library: Library, schedule: dict, pool: Pool, drawn: list[Draw], weights,
             baseline: Baseline, cap_factor: float, tolerance: float,
             abandon: float | None = None, charge: float = 1.0
             ) -> tuple[Score, list[int]]:
    """Run one schedule over `drawn`, one instance resident at a time.

    `baseline` supplies (and runs on demand) each instance's reference, and
    `abandon` the ratio after the first bucket beyond which the rest is not
    run.  Returns the score and the per-instance iteration counts.
    """
    counts, ratios, capped = [], [], 0
    start, base_start = time.perf_counter(), baseline.seconds
    first_bucket = drawn[0].family
    abandoned = False
    for k, instance in enumerate(pool.stream(drawn)):
        if abandon is not None and instance.family != first_bucket and ratios:
            if weighted_geomean(ratios, weights[:k]) > abandon:
                abandoned = True
                break
        base_count, base_cost = baseline(instance)
        cap = max(int(cap_factor * base_count), 64)
        parameters = library.parameters(tolerance=tolerance, max_iter=cap, **schedule)
        solve = library.solve(instance, parameters)
        if solve.status != 0:
            raise RuntimeError(f"pria_solve failed with status {solve.status} "
                               f"on {instance.name}")
        # The library reports no status of its own, so "did not converge" is
        # "stopped at the cap": the gap it reached against the one it wanted.
        hit_cap = not (solve.gap <= tolerance)
        count = cap if hit_cap else solve.iterations
        capped += hit_cap
        counts.append(count)
        ratios.append(cost_of(count, schedule["op_steps"], charge) / base_cost)
    seconds = time.perf_counter() - start - (baseline.seconds - base_start)
    score = Score(weighted_geomean(ratios, weights[:len(ratios)]), float(max(ratios)),
                  capped, abandoned, int(sum(counts)), seconds)
    return score, counts


def describe(schedule: dict) -> str:
    return " ".join(f"{name}={int(schedule[name])}" if name in INTEGER
                    else f"{name}={float(schedule[name]):.4g}" for name in SCHEDULE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=int, default=200,
                        help="configurations to try")
    parser.add_argument("--time-budget", type=float, default=math.inf,
                        help="stop drawing configurations after this many seconds")
    parser.add_argument("--startup", type=int, default=60,
                        help="random configurations before TPE starts proposing, on top "
                             "of any warm-started or enqueued trials")
    parser.add_argument("--no-enqueue-default", action="store_true",
                        help="do not score the default schedule as a trial of its own")
    parser.add_argument("--max-size-ratio", type=float, default=1.05,
                        help="reject a leader whose hold-out ratio at any one shape is "
                             "worse than this, however good its average; 0 to rank on "
                             "the average alone")
    parser.add_argument("--gamma", type=float, default=0.25,
                        help="fraction of the trials TPE treats as good; Optuna's\n"
                             "default is far narrower")
    parser.add_argument("--gamma-cap", type=int, default=60,
                        help="at most this many good trials (Optuna's default is 25)")
    parser.add_argument("--warm-start", type=Path, nargs="+", default=[],
                        help="CSVs of earlier runs whose scores seed the sampler")
    parser.add_argument("--buckets", nargs="+", default=list(instances.BENCHMARK),
                        choices=list(instances.LADDER),
                        help="size buckets to train, hold out and confirm on")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="seed of the sampler and of the per-configuration draws; "
                             "the instance seeds add fixed offsets to it")
    parser.add_argument("--cap-factor", type=float, default=3.0,
                        help="cut a run off at this multiple of the baseline's iterations")
    parser.add_argument("--abandon", type=float, default=2.0,
                        help="drop a configuration whose first-bucket score is this "
                             "many times the baseline's")
    parser.add_argument("--size-power", type=float, default=0.5,
                        help="weight each shape by its x-variable count to this power; "
                             "0 counts shapes equally")
    parser.add_argument("--check-charge", type=float, default=1.0,
                        help="cost of one restart-and-termination check, in iterations; "
                             "0 scores iterations alone")
    parser.add_argument("--pool-factor", type=float, default=3.0,
                        help="seeds in each shape's pool, as a multiple of the\n"
                             "draw, with a floor")
    parser.add_argument("--holdout-factor", type=float, default=4.0,
                        help="hold-out instances per shape, as a multiple of the\n"
                             "training draw")
    parser.add_argument("--cache-gb", type=float, default=40.0,
                        help="host memory for generated instances; beyond it the least "
                             "recently used are regenerated when drawn again")
    parser.add_argument("--optimality-tolerance", type=float,
                        default=config.TOLERANCES.optimality,
                        help="relative duality gap each run stops on")
    parser.add_argument("--leaders", type=int, default=8,
                        help="how many of the best training scores to re-score\n"
                             "on the hold-out")
    parser.add_argument("--confirm", type=int, default=3,
                        help="how many of the best hold-out scores to score on the benchmark")
    parser.add_argument("--confirm-seeds", type=int, default=3,
                        help="seeds per shape of the benchmark subset the leaders are "
                             "confirmed on; the whole benchmark is far longer")
    parser.add_argument("--no-confirm", action="store_true",
                        help="skip scoring the leaders on the benchmark itself")
    parser.add_argument("--library", type=Path, default=None,
                        help="libpria.so to load; default $PRIA_BIN_DIR/libpria.so")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV of every configuration; the default is under\n"
                             "$PRIA_RESULTS_DIR/tuning")
    args = parser.parse_args(argv)

    library = Library(args.library)
    output = args.output or results.claim_path(
        config.results_dir() / "tuning"
        / f"tune-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rungs = rungs_for(args.buckets, args.pool_factor)
    if not rungs:
        raise SystemExit(f"no training shapes in {args.buckets}")
    pool = Pool(args.cache_gb * 2 ** 30)
    rng = np.random.default_rng(args.seed)
    print(f"library {library.path}\n{sum(r.draw for r in rungs)} training instances per "
          f"configuration over {len(rungs)} shapes in {args.buckets}, drawn from pools of "
          f"{sum(r.pool for r in rungs)}; a check costs {args.check_charge:g} iterations; "
          f"shapes weighted by entries^{args.size_power:g}")
    print("  " + "  ".join(f"{r.items}x{r.periods}:{r.draw}/{r.pool}" for r in rungs)
          + "  (items x periods : draw/pool)")

    default = library.parameters(tolerance=args.optimality_tolerance).as_dict()
    default = {name: default[name] for name in SCHEDULE}
    print(f"baseline: {describe(default)}, run on each instance as it is first drawn")
    baseline = Baseline(library, default, args.optimality_tolerance, args.check_charge)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Optuna deprecated `gamma` while still honouring it; without this the
    # warning prints once per run.
    warnings.filterwarnings("ignore", message=".*gamma.*", category=FutureWarning)
    study = optuna.create_study(direction="minimize")
    for path in args.warm_start:
        added = warm_start(study, path, args.check_charge, args.size_power)
        print(f"warm start: {added} configurations from {path}, re-scored under "
              f"this objective")

    # The default is enqueued rather than assumed to score 1: it is 1 by
    # construction against itself, which is not the same as measured on a draw
    # under the same noise as everything else.
    if not args.no_enqueue_default:
        study.enqueue_trial(to_params(default))

    # Start-up counts trials, not draws of its own, so warm-started and
    # enqueued trials would otherwise consume the random exploration budget
    # entirely.  The sampler is therefore built after them, asking for
    # `--startup` random probes on top of whatever is already in the study.
    existing = len(study.trials)
    study.sampler = optuna.samplers.TPESampler(
        multivariate=True, seed=args.seed, n_startup_trials=existing + args.startup,
        gamma=lambda n: min(math.ceil(args.gamma * n), args.gamma_cap))
    print(f"{existing} trials in the study before the search; "
          f"{args.startup} random probes on top of them")

    # One column per pool instance holds the iteration count of the
    # configurations that drew it (empty for the others, and past the point a
    # configuration was abandoned), so the score can be recomputed under other
    # weights or another check charge from the CSV alone; that is what a warm
    # start does.  The baseline's row fills in as the pool is drawn.
    pool_names = [Draw(r.family, r.items, r.periods, TRAINING_SEEDS + k).name
                  for r in rungs for k in range(r.pool)]
    columns = ["config", *SCHEDULE, "geomean", "worst", "capped", "abandoned",
               "iterations", "seconds", *pool_names]
    rows = []

    def flush():
        base_row = dict(config=0, **default, geomean=1.0, worst=1.0, capped=0,
                        abandoned=0,
                        iterations=sum(baseline.counts.get(c, 0) for c in pool_names),
                        seconds=round(baseline.seconds, 3),
                        **{c: baseline.counts[c] for c in pool_names
                           if c in baseline.counts})
        with open(output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows([base_row, *rows])

    flush()
    started = time.perf_counter()
    best = 1.0
    for k in range(1, args.configs + 1):
        if time.perf_counter() - started > args.time_budget:
            print(f"time budget reached after {k - 1} configurations")
            break
        train = training_draw(rungs, rng)
        weights = size_weights(train, args.size_power)
        trial = study.ask()
        schedule = suggest(trial)
        score, counts = evaluate(library, schedule, pool, train, weights, baseline,
                                 args.cap_factor, args.optimality_tolerance,
                                 args.abandon, charge=args.check_charge)
        if score.abandoned:
            # TPE takes a pruned trial at its last reported value: the
            # first-bucket score, above `abandon` and so above any real one.
            trial.report(score.geomean, 0)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, score.geomean)
        rows.append(dict(config=k, **schedule, geomean=round(score.geomean, 4),
                         worst=round(score.worst, 3), capped=score.capped,
                         abandoned=int(score.abandoned), iterations=score.iterations,
                         seconds=round(score.seconds, 3),
                         # An abandoned configuration has counts for only
                         # the buckets it reached.
                         **{d.name: c for d, c in zip(train, counts, strict=False)}))
        flush()
        flag = ""
        if score.abandoned:
            flag = "  abandoned after the first bucket"
        elif score.geomean < best:
            best, flag = score.geomean, "  <-- best so far"
        print(f"[{k:4d}] geomean {score.geomean:.3f} worst {score.worst:.2f} "
              f"capped {score.capped:2d} {score.seconds:6.1f}s  {describe(schedule)}{flag}")
    print(f"baseline runs: {len(baseline.counts)} instances in {baseline.seconds:.0f}s")

    ranked = sorted((r for r in rows if not r["abandoned"]),
                    key=lambda r: r["geomean"])[:args.leaders]
    if not ranked:
        print("no configuration got past the first bucket")
        return 1

    plan = holdout_plan(rungs, args.holdout_factor)
    print(f"\nthe {len(ranked)} best training scores, re-scored on the hold-out "
          f"({len(plan)} instances, seeds disjoint from the pool), ranked by the hold-out "
          "(ratio to baseline; < 1 is better):")

    # One instance at a time: built, scored for the baseline and every leader,
    # then dropped.  See `holdout_plan`.
    leaders = [(row, {name: row[name] for name in SCHEDULE}) for row in ranked]
    shape_counts = Counter(d.entries for d in plan)
    ratios: dict[int, dict[int, list[float]]] = {row["config"]: defaultdict(list)
                                                 for row, _ in leaders}
    for instance in pool.stream(plan):
        size = instance.entries
        base_count, base_cost = baseline(instance)
        cap = max(int(args.cap_factor * base_count), 64)
        for row, schedule in leaders:
            parameters = library.parameters(tolerance=args.optimality_tolerance,
                                            max_iter=cap, **schedule)
            solve = library.solve(instance, parameters)
            if solve.status != 0:
                raise RuntimeError(f"pria_solve failed with status {solve.status} "
                                   f"on {instance.name}")
            count = cap if not (solve.gap <= args.optimality_tolerance) else solve.iterations
            ratios[row["config"]][size].append(
                cost_of(count, schedule["op_steps"], args.check_charge) / base_cost)
        del instance

    scored = []
    for row, _schedule in leaders:
        per = {size: math.exp(float(np.mean(np.log(v))))
               for size, v in sorted(ratios[row["config"]].items())}
        flat = [v for size in per for v in ratios[row["config"]][size]]
        flat_w = [size ** args.size_power / shape_counts[size]
                  for size in per for _ in ratios[row["config"]][size]]
        scored.append((weighted_geomean(flat, flat_w), row, per))

    # Ranked on the average, but a leader that is worse than --max-size-ratio at
    # any one shape is set behind the ones that are not, however good its
    # average: an average winner that regresses on the largest rungs is not a
    # winner, and the average cannot see that by construction.
    def rank(item):
        geomean, _row, per = item
        breached = args.max_size_ratio > 0 and max(per.values()) > args.max_size_ratio
        return (breached, geomean)
    scored.sort(key=rank)

    header = (f"{'config':>6} {'train':>7} {'holdout':>8} {'worst shape':>11} "
              f"{'at':>10}  schedule")
    print(header)
    print("-" * len(header))
    for geomean, row, per in scored:
        schedule = {name: row[name] for name in SCHEDULE}
        worst = max(per, key=per.get)
        flag = "" if not (args.max_size_ratio > 0 and per[worst] > args.max_size_ratio) \
            else f"   rejected: {worst} is {per[worst]:.2f}"
        print(f"{row['config']:6d} {row['geomean']:7.3f} {geomean:8.3f} "
              f"{per[worst]:11.3f} {worst:10d}  {describe(schedule)}{flag}")

    if not args.no_confirm:
        # A fixed subset of the benchmark: its first `--confirm-seeds` seeds at
        # every shape, which spans the ladder without running all of it.  Each
        # instance is built once and every schedule is run on it before it is
        # dropped, rather than the set being materialised or rebuilt per
        # leader: the largest rungs take longer to generate than to solve.
        chosen = [s for s in instances.all_specs(args.buckets)
                  if int(s.name.rsplit("-s", 1)[1]) < args.confirm_seeds]
        leaders = [(row["config"], {name: row[name] for name in SCHEDULE})
                   for _geomean, row, _per in scored[:args.confirm]]
        weights = size_weights(chosen, args.size_power)
        print(f"\n{len(chosen)} benchmark instances ({args.confirm_seeds} seeds per "
              f"shape), scored once, not used for selection:")
        ratios = {config_id: [] for config_id, _ in leaders}
        base_total = 0
        for spec in chosen:
            instance = spec.build()
            base_count, base_cost = baseline(instance)
            base_total += base_count
            cap = max(int(args.cap_factor * base_count), 64)
            for config_id, schedule in leaders:
                parameters = library.parameters(tolerance=args.optimality_tolerance,
                                                max_iter=cap, **schedule)
                solve = library.solve(instance, parameters)
                count = cap if not (solve.gap <= args.optimality_tolerance) \
                    else solve.iterations
                ratios[config_id].append(
                    cost_of(count, schedule["op_steps"], args.check_charge) / base_cost)
            del instance
        print(f"  baseline: {base_total} iterations")
        for config_id, _ in leaders:
            print(f"  config {config_id:4d}: geomean "
                  f"{weighted_geomean(ratios[config_id], weights):.3f} "
                  f"worst {max(ratios[config_id]):.2f}")

    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
