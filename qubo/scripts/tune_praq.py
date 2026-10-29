#!/usr/bin/env python3
"""Tune PRAQ's restart schedule on the benchmark's distribution, with praq-cuda.

    source env.sh
    scripts/tune_praq.py                                     # every bucket, 200 configurations
    scripts/tune_praq.py --buckets hard-medium --configs 50  # a quick look
    scripts/tune_praq.py --configs 1000000 --time-budget 10800   # what the batch job runs
    scripts/tune_praq.py --warm-start results/tuning/tune-<earlier>.csv

Optuna's multivariate TPE searches the schedule: the three restart ratios, the
theta smoothing of the primal weight, the initial weight and the three
constants of the check-interval rule.  The score is the cost of reaching the
optimality tolerance, in iterations plus `--check-charge` per
restart-and-termination check -- a check is a reduction pass and a synchronous
copy, so an iteration count alone would price a small interval at nothing and
the search would drive the interval to its floor.  Iterations and not seconds,
because they are deterministic and do not depend on the GPU.

The interval is a rule of n rather than a number, because its best value grows
with size.  Every proposal sets `op_steps` to 0, which hands the interval to
the rule; the baseline keeps the fixed default so that ratios stay comparable
across runs.  `--interval-power` pins the rule's power -- the choice between a
logarithm, a square root and a line -- when the search should not spend trials
on it.

**Scoring.**  Every configuration is scored on its own draw.  Each size of the
training ladder has a pool of seeds (`--pool-factor` times its draw, at least
four), each configuration draws a fresh random subset of every pool, and a
pool instance's baseline is run the first time it is drawn and remembered.  A
configuration therefore cannot be selected for suiting particular instances,
and the sampler's density over neighbouring trials does the averaging a fixed
set would need many seeds for.  Sizes are the medium and large buckets, then a
ladder geometric in n up past the benchmark's top rung (`LADDER`), with fewer
instances where one costs more.

Each instance's ratio to the baseline is weighted by `n**--size-power`, split
over the instances at that size, so a size's weight does not depend on how
many seeds it has.  Power 0 counts every size equally and lets the medium
bucket outvote the rungs the schedule is for; power 1 tracks the wall clock
and lets the top rung alone decide the run.  The score is the weighted
geometric mean of the ratios, and the worst case is kept beside it.  A run
past `--cap-factor` times the baseline's iterations is cut off and counted at
the cap.

Configurations are evaluated bucket by bucket, cheapest first, and abandoned
after the first if its score there is already `--abandon` times the baseline.
The medium bucket is a poor predictor of the huge one, so the default only
drops configurations that are diverging there, not merely worse.

**Selection.**  The winner is chosen on the hold-out, not on the training
score, under which the best value belongs to the luckiest trial.  The
`--leaders` best are re-scored on a fixed hold-out drawn with disjoint seeds
and ranked by that, and a leader worse than `--max-size-ratio` at any one size
is ranked behind those that are not -- an average winner that regresses on the
largest rungs is not a winner, and the average cannot see that.  The best
`--confirm` of those are scored once on the benchmark itself, so the last
number is neither fitted nor selected on.

**Warm starts.**  `--warm-start` seeds the sampler with everything an earlier
run scored, abandoned configurations included, rescored under the current
charge and weights from the per-instance columns of its CSV.  It also consumes
the start-up phase, which counts every trial in the study, so a warm start
continues a search; a search starts cold, and `--startup` random draws are
what give ten noisy dimensions a chance of showing the sampler more than one
basin.

Results go to a CSV under $PRAQ_RESULTS_DIR/tuning, rewritten after every
configuration, with a column per pool instance holding the iteration count
where the configuration drew it, so a score can be recomputed under other
weights without re-running.

The winner is not applied automatically.  Its values belong in two places that
must move together: `Parameters.with_auto_step` in praq-jax/praq/solver.py and
the defaults at the top of praq-cuda/src/solver.cuh.
"""

import argparse
import csv
import math
import sys
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import optuna

import _bootstrap  # noqa: F401  (sys.path)

from harness import config, instances
from harness.praq_cuda_library import SCHEDULE, Library

#: Seed offsets of the two draws that must not overlap the benchmark's seeds,
#: which start at 0 and stay far below these.
TRAINING_SEEDS = 10_000
HOLDOUT_SEEDS = 20_000

#: The search box: omega in log scale, the rest linear.  Centred on the current
#: values and wide enough that the edges are visibly bad.  beta_necessary is
#: searched as a fraction of the room above beta_sufficient, since below it
#: the necessary test never fires: every proposal is then feasible and the
#: box the sampler learns on is fixed.  The interval rule's box spans the
#: logarithm, the square root and the line alike, and the tuner picks the law
#: by where it puts the power.  The offset may be negative, since for the
#: logarithm it is where the curve crosses zero, and the scale reaches 0,
#: which is a constant interval; both are needed for a fixed interval to be a
#: point of the box, so the default can be enqueued and scored like anything
#: else.
SPACE = {
    "omega": (0.01, 10.0),
    "beta_sufficient": (0.15, 0.8),
    "beta_necessary": (0.3, 0.95),
    "beta_artificial": (0.05, 0.6),
    "theta_initial": (0.3, 1.0),
    "theta_target": (0.1, 1.0),
    "theta_decay": (0.5, 0.99),
    "interval_offset": (-16.0, 48.0),
    "interval_scale": (0.0, 12.0),
    "interval_power": (0.0, 1.0),
}


def searched(fixed_power: float | None) -> list[str]:
    """The names the sampler proposes: the box less beta_necessary (the
    fraction stands in for it) and less the power when it is pinned."""
    return [name for name in SPACE
            if name != "beta_necessary" and not (name == "interval_power" and fixed_power is not None)]


def distributions(fixed_power: float | None) -> dict:
    """What the sampler sees: the searched names' boxes plus the fraction.
    op_steps is not searched: it is 0 in every proposal."""
    out = {name: optuna.distributions.FloatDistribution(*SPACE[name], log=(name == "omega"))
           for name in searched(fixed_power)}
    out["beta_necessary_frac"] = optuna.distributions.FloatDistribution(0.0, 1.0)
    return out


def necessary_room(beta_sufficient: float) -> tuple[float, float]:
    """The interval beta_necessary is drawn from, given beta_sufficient."""
    lo, hi = SPACE["beta_necessary"]
    return max(lo, beta_sufficient), hi


def to_params(schedule: dict, fixed_power: float | None) -> dict:
    """A schedule in the sampler's coordinates (the inverse of `suggest`).  A
    pinned power is dropped: an earlier run's trials then count as if their
    power made no difference, which the interval sweep found to be nearly so."""
    params = {name: float(schedule[name]) for name in searched(fixed_power)}
    lo, hi = necessary_room(params["beta_sufficient"])
    fraction = (float(schedule["beta_necessary"]) - lo) / (hi - lo)
    params["beta_necessary_frac"] = min(max(fraction, 0.0), 1.0)
    return params


def suggest(trial: optuna.Trial, fixed_power: float | None) -> dict:
    """One point of the search box, proposed by the study."""
    point = {}
    for name in searched(fixed_power):
        lo, hi = SPACE[name]
        point[name] = trial.suggest_float(name, lo, hi, log=(name == "omega"))
    if fixed_power is not None:
        point["interval_power"] = fixed_power
    lo, hi = necessary_room(point["beta_sufficient"])
    point["beta_necessary"] = lo + (hi - lo) * trial.suggest_float("beta_necessary_frac", 0.0, 1.0)
    point["op_steps"] = 0      # the rule's, see check_interval
    return point


def default_as_rule(default: dict) -> dict:
    """The default schedule written in the sampler's coordinates.

    Its `op_steps` is a fixed interval, which the rule expresses as a zero
    scale and that interval as the offset: `check_interval` is then
    `max(2, round(offset))` at every size.  Without this the default is not a
    point of the search box and the sampler never learns what the thing it is
    being compared against actually scores -- it is 1 by construction, which
    is not the same as measured on a draw under the same noise.
    """
    return dict(default, interval_offset=float(default["op_steps"]),
                interval_scale=0.0, interval_power=0.0, op_steps=0)


def size_of(column: str) -> int:
    """The n in an instance column name such as hard-huge-n2000000-s10001."""
    return int(column.split("-n")[1].split("-")[0])


def rescore(row: dict, base: dict, columns: list[str], library: Library,
            charge: float, size_power: float) -> float:
    """A CSV row's score under the current objective, from its per-instance
    iteration counts (the columns present: a row holds only the instances its
    configuration drew, and an abandoned row fewer).  The earlier run's
    instances and baseline, not this run's, but the same charge and weights,
    so its scores mean what this run's do."""
    present = [c for c in columns if row[c] != "" and base[c] != ""]
    counts = Counter(size_of(c) for c in present)
    weights = [size_of(c) ** size_power / counts[size_of(c)] for c in present]
    base_p = library.parameters(**{name: float(base[name]) if name != "op_steps"
                                   else int(base[name]) for name in SCHEDULE})
    row_p = library.parameters(**{name: float(row[name]) if name != "op_steps"
                                  else int(row[name]) for name in SCHEDULE})
    ratios = []
    for c in present:
        cost = float(row[c]) * (1.0 + charge / library.check_interval(size_of(c), row_p))
        base_cost = float(base[c]) * (1.0 + charge / library.check_interval(size_of(c), base_p))
        ratios.append(cost / base_cost)
    return weighted_geomean(ratios, weights)


def warm_start(study: optuna.Study, path: Path, library: Library, charge: float,
               size_power: float, fixed_power: float | None) -> int:
    """Add an earlier run's CSV to the study: scored rows as complete trials,
    abandoned ones as pruned with their first-bucket score, the baseline
    skipped, every score recomputed under the current objective from the
    per-instance columns.  Returns how many were added."""
    added = 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in SPACE if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path} predates the search box: no {', '.join(missing)} column")
        columns = [c for c in reader.fieldnames if c.startswith("hard-")]
        rows = list(reader)
    base = next(r for r in rows if r["config"] == "0")
    space = distributions(fixed_power)
    for row in rows:
        if row["config"] == "0" or int(row["op_steps"]) > 0:
            continue   # the baseline, or a fixed-interval configuration from an older search
        params = to_params(row, fixed_power)
        value = rescore(row, base, columns, library, charge, size_power)
        if int(row["abandoned"]):
            trial = optuna.trial.create_trial(
                params=params, distributions=space,
                state=optuna.trial.TrialState.PRUNED, intermediate_values={0: value})
        else:
            trial = optuna.trial.create_trial(params=params, distributions=space, value=value)
        study.add_trial(trial)
        added += 1
    return added


@dataclass(frozen=True)
class Rung:
    family: str     #: the benchmark bucket it belongs to, or hard-huge above them
    n: int
    draw: int       #: instances per configuration
    pool: int       #: seeds those are drawn from


#: The training ladder above the medium and large buckets: the huge bucket's
#: rungs with geometric midpoints between them and one doubling above, so a
#: ratio of about the square root of two throughout.  100k is the large
#: bucket's top rung and keeps its family.  The ladder stops at 4M: an 8M
#: rung was a third of every configuration's time and, drawn once, the
#: noisiest term in the score; the benchmark itself ends at 2M.
LADDER = (100_000, 141_421, 200_000, 316_228, 500_000, 707_107, 1_000_000,
          1_414_214, 2_000_000, 2_828_427, 4_000_000)


def draws_at(n: int) -> int:
    """Instances per configuration at size n of the ladder: many where they
    are cheap, two where an instance is ten seconds or more, so that no rung
    is one instance's luck."""
    return 6 if n <= 500_000 else 3 if n <= 2_000_000 else 2


def rungs_for(buckets: list[str], pool_factor: float) -> list[Rung]:
    """The training sizes for `buckets`, cheapest first."""
    def rung(family, n, draw):
        return Rung(family, n, draw, max(4, math.ceil(pool_factor * draw)))
    out = []
    if "hard-medium" in buckets:
        out += [rung("hard-medium", n, 3) for n, _ in instances.HARD_BUCKETS["hard-medium"]]
    if "hard-large" in buckets:
        out += [rung("hard-large", n, 3) for n, _ in instances.HARD_BUCKETS["hard-large"]
                if n < LADDER[0]]
    for n in LADDER:
        family = "hard-large" if n <= 100_000 else "hard-huge"
        if family in buckets:
            out.append(rung(family, n, draws_at(n)))
    return out


@dataclass(frozen=True)
class Draw:
    """Which instance to build, and nothing built.

    This triple identifies an instance everywhere here: the pool keys on it,
    the hold-out is a list of them, and the CSV's per-instance columns are its
    `name`.  It travels as one value so that the pool's key, the column
    heading and the instance's own name cannot drift apart -- a warm start
    matches old runs by that name, so they have to agree exactly.
    """

    family: str
    n: int
    seed: int

    @property
    def name(self) -> str:
        return f"{self.family}-n{self.n}-s{self.seed}"


class Pool:
    """Instances by `Draw`, generated on first use and kept while they fit in
    `budget` bytes, the least recently used going first.  A 4M instance is a
    gigabyte and takes long enough to generate that the training pool should
    fit whole; the hold-out then replaces it.

    Eviction only reclaims what the caller has let go of, so the budget is a
    promise the caller has to keep.  `stream` is the only way in for that
    reason: it hands out one instance at a time and the loop that consumes it
    drops it before the next.  A `list(pool.stream(...))` defeats the budget
    and is how job 22835493 was OOM-killed at 67 GiB, holding a hold-out of
    24 instances at 2M, 16 at 2.8M and 16 at 4M.
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
        instance = instances.random_hard(draw.seed, draw.n)
        instance = replace(instance, name=draw.name, family=draw.family)
        size = sum(a.nbytes for a in (instance.diag, instance.rows, instance.cols, instance.vals))
        while self.cache and self.bytes + size > self.budget:
            _, (_, old) = self.cache.popitem(last=False)
            self.bytes -= old
        self.cache[draw] = (instance, size)
        self.bytes += size
        return instance


def training_draw(rungs: list[Rung], rng: np.random.Generator) -> list[Draw]:
    """One configuration's draws: `draw` seeds of each rung's pool, at random
    without replacement, cheapest rung first.  Draws, not instances: sixty-odd
    instances of this ladder are some sixteen gigabytes, and nothing needs
    more than one of them at a time."""
    out = []
    for rung in rungs:
        for k in sorted(rng.choice(rung.pool, size=rung.draw, replace=False).tolist()):
            out.append(Draw(rung.family, rung.n, TRAINING_SEEDS + k))
    return out


def holdout_plan(rungs: list[Rung], factor: float) -> list[Draw]:
    """The fixed selection set as `Draw`s, building nothing.

    It must not be materialised.  `Pool` evicts by byte budget, but a list of
    instances holds hard references that eviction cannot reclaim, so a deep
    hold-out becomes resident all at once: at `--holdout-factor 8` that is 24
    instances at 2M, 16 at 2.8M and 16 at 4M, about 73 GiB, and job 22835493
    was OOM-killed at 67 GiB after six hours of search had already finished.
    The caller builds one at a time and scores every leader on it before
    letting go, which is also fewer builds than one pass per leader.
    """
    return [Draw(rung.family, rung.n, HOLDOUT_SEEDS + k)
            for rung in rungs
            for k in range(max(1, math.ceil(factor * rung.draw)))]


def size_weights(drawn, power: float = 0.5) -> np.ndarray:
    """Each instance's weight: its size to `power`, split over the instances at
    that size, so a size's total weight is n**power however many seeds it has.
    Power 0 counts every size equally; 1 tracks the wall clock, which is spent
    on the largest rungs; 1/2 lets the top octave lead without one instance
    deciding the score."""
    counts = Counter(d.n for d in drawn)
    return np.array([d.n ** power / counts[d.n] for d in drawn])


class Baseline:
    """The default schedule's iterations and cost on every instance it has
    met, run on first request and remembered by name."""

    def __init__(self, library: Library, schedule: dict, tolerance: float, charge: float):
        self.library, self.schedule, self.tolerance, self.charge = library, schedule, tolerance, charge
        self.counts: dict[str, int] = {}
        self.costs: dict[str, float] = {}
        self.seconds = 0.0

    def __call__(self, instance: instances.Instance) -> tuple[int, float]:
        if instance.name not in self.counts:
            start = time.perf_counter()
            parameters = self.library.parameters(tolerance=self.tolerance, max_iter=-1, **self.schedule)
            solve = self.library.solve(instance, parameters)
            if solve.status < 0:
                raise RuntimeError(f"praq_solve failed with status {solve.status} on {instance.name}")
            count = solve.iterations
            self.counts[instance.name] = count
            self.costs[instance.name] = count * (
                1.0 + self.charge / self.library.check_interval(instance.n, parameters))
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


def weighted_geomean(ratios, weights) -> float:
    ratios, weights = np.asarray(ratios), np.asarray(weights)
    return float(np.exp(np.sum(weights * np.log(ratios)) / np.sum(weights)))


def evaluate(library: Library, schedule: dict, pool: Pool, drawn: list[Draw], weights,
             baseline: Baseline, cap_factor: float, tolerance: float,
             abandon: float | None = None,
             charge: float = 1.0) -> tuple[Score, list[int]]:
    """Run one schedule over `drawn`, one instance resident at a time.

    `baseline` supplies (and runs on demand) each instance's reference,
    `abandon` the ratio after the first bucket beyond which the rest is not
    run.  A run's cost is its iterations times (1 + charge / interval): the
    checks, charged at `charge` iterations each.  Returns the score and the
    per-instance iteration counts."""
    counts, ratios, capped = [], [], 0
    start, base_start = time.perf_counter(), baseline.seconds
    first_bucket = drawn[0].family
    abandoned = False
    for k, instance in enumerate(pool.stream(drawn)):
        if abandon is not None and instance.family != first_bucket:
            if weighted_geomean(ratios, weights[:k]) > abandon:
                abandoned = True
                break
        base_count, base_cost = baseline(instance)
        cap = max(int(cap_factor * base_count), 64)
        parameters = library.parameters(tolerance=tolerance, max_iter=cap, **schedule)
        solve = library.solve(instance, parameters)
        if solve.status < 0:
            raise RuntimeError(f"praq_solve failed with status {solve.status} on {instance.name}")
        count = solve.iterations if solve.status == 0 else cap
        capped += solve.status != 0
        cost = count * (1.0 + charge / library.check_interval(instance.n, parameters))
        counts.append(count)
        ratios.append(cost / base_cost)
    seconds = time.perf_counter() - start - (baseline.seconds - base_start)
    score = Score(weighted_geomean(ratios, weights[:len(ratios)]), float(max(ratios)),
                  capped, abandoned, int(sum(counts)), seconds)
    return score, counts


def describe(schedule: dict) -> str:
    return " ".join(f"{name}={schedule[name]:.4g}" if name != "op_steps"
                    else f"op_steps={schedule[name]}" for name in SCHEDULE)


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
                        help="reject a leader whose hold-out ratio at any one size is worse "
                             "than this, however good its average; 0 to rank on the average "
                             "alone")
    parser.add_argument("--gamma", type=float, default=0.25,
                        help="fraction of the trials TPE treats as good (Optuna's default is 0.1)")
    parser.add_argument("--gamma-cap", type=int, default=60,
                        help="at most this many good trials (Optuna's default is 25)")
    parser.add_argument("--warm-start", type=Path, nargs="+", default=[],
                        help="CSVs of earlier runs whose scores seed the sampler")
    parser.add_argument("--buckets", nargs="+", default=list(instances.BENCHMARK),
                        choices=list(instances.HARD_BUCKETS),
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
                        help="weight each size by n to this power; 0 counts sizes equally")
    parser.add_argument("--check-charge", type=float, default=1.0,
                        help="cost of one restart-and-termination check, in iterations; "
                             "0 scores iterations alone")
    parser.add_argument("--interval-power", type=float, default=None,
                        help="pin the interval rule's power (0 log, 0.5 sqrt, 1 linear) "
                             "instead of searching it")
    parser.add_argument("--pool-factor", type=float, default=3.0,
                        help="seeds in each size's pool, as a multiple of the draw (at least 4)")
    parser.add_argument("--holdout-factor", type=float, default=8.0,
                        help="hold-out instances per size, as a multiple of the training draw")
    parser.add_argument("--cache-gb", type=float, default=40.0,
                        help="host memory for generated instances; beyond it the least "
                             "recently used are regenerated when drawn again")
    parser.add_argument("--optimality-tolerance", type=float,
                        default=config.TOLERANCES.optimality,
                        help="relative duality gap each run stops on")
    parser.add_argument("--leaders", type=int, default=8,
                        help="how many of the best training scores to re-score on the hold-out")
    parser.add_argument("--confirm", type=int, default=3,
                        help="how many of the best hold-out scores to score on the benchmark")
    parser.add_argument("--confirm-seeds", type=int, default=3,
                        help="seeds per size of the benchmark subset the leaders are "
                             "confirmed on; the whole benchmark is hours")
    parser.add_argument("--no-confirm", action="store_true",
                        help="skip scoring the leaders on the benchmark itself")
    parser.add_argument("--library", type=Path, default=None,
                        help="libpraq.so to load; default $PRAQ_BIN_DIR/libpraq.so")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV of every configuration; default under $PRAQ_RESULTS_DIR/tuning")
    args = parser.parse_args(argv)

    library = Library(args.library)
    output = args.output or (config.results_dir() / "tuning"
                             / f"tune-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rungs = rungs_for(args.buckets, args.pool_factor)
    if not rungs:
        raise SystemExit(f"no training sizes in {args.buckets}")
    pool = Pool(args.cache_gb * 2 ** 30)
    rng = np.random.default_rng(args.seed)
    print(f"library {library.path}\n{sum(r.draw for r in rungs)} training instances per "
          f"configuration over {len(rungs)} sizes in {args.buckets}, drawn from pools of "
          f"{sum(r.pool for r in rungs)}; a check costs {args.check_charge:g} iterations; "
          f"sizes weighted by n^{args.size_power:g}")
    print("  " + "  ".join(f"{r.n}:{r.draw}/{r.pool}" for r in rungs) + "  (size:draw/pool)")

    default = library.parameters().as_dict()
    default = {name: default[name] for name in SCHEDULE}
    print(f"baseline: {describe(default)}, run on each instance as it is first drawn")
    baseline = Baseline(library, default, args.optimality_tolerance, args.check_charge)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # `gamma` is deprecated since Optuna 4.9 and still honoured in 5.0; the
    # warning would otherwise print once per run.
    warnings.filterwarnings("ignore", message=".*gamma.*", category=FutureWarning)
    study = optuna.create_study(direction="minimize")
    for path in args.warm_start:
        added = warm_start(study, path, library, args.check_charge,
                           args.size_power, args.interval_power)
        print(f"warm start: {added} configurations from {path}, re-scored under this objective")

    # Enqueued rather than assumed to score 1: it is 1 against itself by
    # construction, which is not the same as measured on a draw under the same
    # noise as everything else.
    if not args.no_enqueue_default:
        study.enqueue_trial(to_params(default_as_rule(default), args.interval_power))

    # Start-up counts trials, not draws of its own, so warm-started and
    # enqueued trials would otherwise consume the random exploration budget
    # entirely and the search would never leave their basin.  The sampler is
    # built after them, asking for `--startup` probes on top.
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
    pool_names = [Draw(r.family, r.n, TRAINING_SEEDS + k).name
                  for r in rungs for k in range(r.pool)]
    columns = ["config", *SCHEDULE, "geomean", "worst", "capped", "abandoned",
               "iterations", "seconds", *pool_names]
    rows = []

    def flush():
        base_row = dict(config=0, **default, geomean=1.0, worst=1.0, capped=0, abandoned=0,
                        iterations=sum(baseline.counts.get(c, 0) for c in pool_names),
                        seconds=round(baseline.seconds, 3),
                        **{c: baseline.counts[c] for c in pool_names if c in baseline.counts})
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
        schedule = suggest(trial, args.interval_power)
        score, counts = evaluate(library, schedule, pool, train, weights, baseline,
                                 args.cap_factor, args.optimality_tolerance, args.abandon,
                                 charge=args.check_charge)
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
                         **{d.name: c for d, c in zip(train, counts)}))
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
    counts = Counter(draw.n for draw in plan)
    ratios: dict[int, dict[int, list[float]]] = {row["config"]: defaultdict(list) for row, _ in leaders}
    weights: dict[int, float] = {}
    for draw, instance in zip(plan, pool.stream(plan)):
        n = draw.n
        base_count, base_cost = baseline(instance)
        weights[instance.name] = n ** args.size_power / counts[n]
        cap = max(int(args.cap_factor * base_count), 64)
        for row, schedule in leaders:
            parameters = library.parameters(tolerance=args.optimality_tolerance,
                                            max_iter=cap, **schedule)
            solve = library.solve(instance, parameters)
            if solve.status < 0:
                raise RuntimeError(f"praq_solve failed with status {solve.status} on {instance.name}")
            count = solve.iterations if solve.status == 0 else cap
            cost = count * (1.0 + args.check_charge
                            / library.check_interval(n, parameters))
            ratios[row["config"]][n].append(cost / base_cost)
        del instance

    scored = []
    for row, schedule in leaders:
        per = {n: math.exp(float(np.mean(np.log(v)))) for n, v in sorted(ratios[row["config"]].items())}
        flat = [v for n in per for v in ratios[row["config"]][n]]
        flat_w = [n ** args.size_power / counts[n] for n in per for _ in ratios[row["config"]][n]]
        scored.append((weighted_geomean(flat, flat_w), row, per))

    # Ranked on the average, but a leader worse than --max-size-ratio at any
    # one size is set behind those that are not: an average winner that
    # regresses on the largest rungs is not a winner, and the average cannot
    # see that by construction.
    def rank(item):
        geomean, _row, per = item
        breached = args.max_size_ratio > 0 and max(per.values()) > args.max_size_ratio
        return (breached, geomean)
    scored.sort(key=rank)

    header = (f"{'config':>6} {'train':>7} {'holdout':>8} {'worst size':>10} {'at':>8}"
              f" {'>=1M':>6}  schedule")
    print(header)
    print("-" * len(header))
    for geomean, row, per in scored:
        schedule = {name: row[name] for name in SCHEDULE}
        worst_n = max(per, key=per.get)
        big = [v for n, v in per.items() if n >= 1_000_000]
        big_geo = f"{math.exp(sum(map(math.log, big)) / len(big)):.3f}" if big else "-"
        flag = "" if not (args.max_size_ratio > 0 and per[worst_n] > args.max_size_ratio) \
            else f"   rejected: {worst_n} is {per[worst_n]:.2f}"
        print(f"{row['config']:6d} {row['geomean']:7.3f} {geomean:8.3f} "
              f"{per[worst_n]:10.3f} {worst_n:8d} {big_geo:>6}  {describe(schedule)}{flag}")

    if not args.no_confirm:
        # A fixed subset of the benchmark: its first `--confirm-seeds` seeds at
        # every size, which spans the ladder without running all of it.  The
        # benchmark is now some two thousand instances and forty minutes a
        # pass, so scoring the baseline and every leader over the whole thing
        # would be hours of tail on a tuning run.
        #
        # Each instance is built once and every schedule is run on it before it
        # is dropped, rather than the set being materialised or rebuilt per
        # leader: the largest rungs take longer to generate than to solve.
        chosen = [s for s in instances.all_specs(args.buckets)
                  if int(s.name.rsplit("-s", 1)[1]) < args.confirm_seeds]
        leaders = [(row["config"], {name: row[name] for name in SCHEDULE})
                   for _geomean, row, _per in scored[:args.confirm]]
        weights = size_weights(chosen, args.size_power)
        print(f"\n{len(chosen)} benchmark instances ({args.confirm_seeds} seeds per size), "
              f"scored once, not used for selection:")
        ratios = {config_id: [] for config_id, _ in leaders}
        base_total = 0
        for spec, weight in zip(chosen, weights):
            instance = spec.build()
            base_count, base_cost = baseline(instance)
            base_total += base_count
            for config_id, schedule in leaders:
                parameters = library.parameters(tolerance=args.optimality_tolerance,
                                                max_iter=max(int(args.cap_factor * base_count), 64),
                                                **schedule)
                solve = library.solve(instance, parameters)
                count = solve.iterations if solve.status == 0 else parameters.max_iter
                cost = count * (1.0 + args.check_charge
                                / library.check_interval(instance.n, parameters))
                ratios[config_id].append(cost / base_cost)
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
