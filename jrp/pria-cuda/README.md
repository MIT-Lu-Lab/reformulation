# PRIA CUDA

A C/CUDA implementation of **PRIA** — the restarted Halpern-accelerated
reflected primal-dual algorithm for the Joint Replenishment Problem (JRP) LP —
ported from the JAX implementation in `pria-jax` and matching it step for
step. All sorting inside the simplex projections runs through
[CUB](https://nvidia.github.io/cccl/cub/) (`cub::BlockRadixSort` on the fused
fast path, `cub::DeviceSegmentedSort` on the general path).

## Problem

```
minimize    sum_s K0*z_s + sum_{i,s} K_i*y[i][s] + sum_{i,t,s<=t} H[i][s][t]*x[i][s][t]
subject to  sum_{s<=t} x[i][s][t] = 1        for all i, t
            x[i][s][t] <= y[i][s]            for all i, s <= t
            x[i][s][t] <= z[s]               for all i, s <= t
            x, y, z >= 0
```

After eliminating the LP variables y and z, PRIA solves the equivalent
saddle-point problem

```
min_{x in X} max_{y in Y, z in Z}  <h, x> + <y, x> + <z, x>
```

where `X` is a product of per-(i,t) unit simplices, `Y` of per-(i,s) simplices
summing to `K_i`, and `Z` of per-period simplices over `{(i,t) : t >= s}`
summing to `K0`. One operator application (step `eta`, primal weight `omega`)
is

```
x' = proj_X(x - (eta/omega) * (h + y + z))
y' = proj_Y(y + eta*omega * (2x' - x))
z' = proj_Z(z + eta*omega * (2x' - x))
```

wrapped in the Halpern iteration
`v_{k+1} = (k+1)/(k+2) * (2T(v_k) - v_k) + 1/(k+2) * v_anchor`, with
sufficient/necessary/artificial restarts every `op_steps` operator steps and a
log-space-smoothed `omega` update at each restart — exactly the scheme of
`pria-jax/pria/solver.py` (including the objective rescaling by
`||(H, K, K0)||_2`, the padding of odd T to even, and the implied
primal/dual objective gap used for termination).

## Layout and kernels

Variables live on the upper triangle `{s <= t}` in dense pair-packed layouts
mirroring the JAX `densify_*` routines (`src/layouts.cuh`): with `P = T/2`,
`W = T+1`, every dense row is exactly full and is the concatenation of exactly
**two projection segments**:

- **column layout** (x, h): row `(i, j)` holds column `t=j` then column
  `t=T-1-j` reversed;
- **row layout** (y): row `(i, j)` holds row `s=j` reversed then row
  `s=T-1-j`;
- **dim2 layout** (z): the row layout with the item index innermost, so each
  per-period group `{(i, t>=s)}` is contiguous.

All layout conversions are closed-form index maps — the permutation arrays of
the JAX implementation become fused gathers inside the kernels.

The double-simplex projection (`src/projection.cuh`) is **sort-free by
default**: Michelot's finite algorithm finds the simplex threshold by
repeatedly averaging the active set and dropping everything at or below the
average (`theta <- (sum_{v>theta} v - target)/#{v>theta}` until the active
count stops shrinking — typically a handful of rounds, each one strided pass
plus a block reduction in a fixed, deterministic order). Two kernels cover
everything:

- **fused pre+project** (whenever a dense row — or, for z, a dense slice —
  fits in 48 KB of shared memory): computes the pre-projection row once into
  shared, then thresholds and clamps both segments on chip (raising the cap
  to Ada's 99 KB opt-in was measured ~7% slower — 1 block/SM — and
  rejected);
- **per-segment kernel** (everything else): one block per segment iterating
  over global memory — short row segments stay L1-resident across rounds,
  the wide z slices stream from L2/DRAM. Two accelerations, both
  deterministic: **warm-started thresholds** (each segment's converged
  theta persists per geometry and seeds the next call with a 1% backoff;
  the first round doubles as validation, falling back cold when theta*
  dropped — cuts rounds/call from ~5.6-6.3 to ~3.6 with 2-4% fallbacks;
  `PRIA_SF_WARM=0` disables) and **active-set compaction** (the first
  validated round also writes its actives, in segment order via a
  tile-wise block scan, into scratch; later rounds iterate the shrinking
  prefix; auto-enabled when the variable exceeds L2/2 or too few segments
  fill the GPU, forced with `PRIA_SF_COMPACT=1/0`). `PRIA_SF_STATS=1`
  prints calls / rounds / fallbacks at teardown.

`PRIA_SORTFREE=0` restores the previous sort-based pipelines (kept as a
reference backend, and measured ~2x slower end to end): a fused
`cub::BlockRadixSort` + `BlockScan` row kernel where rows fit, and a
device-wide `cub::DeviceSegmentedSort` + head-flag segmented scan +
deterministic integer-atomic rho count + threshold + clamp pipeline
otherwise. The two backends compute the same unique threshold (sort-based:
`rho = #{k : u_k*k > S_k - target}`, `theta = (S_rho - target)/rho`) but sum
in different orders, so their solve trajectories differ at the last ulp;
each backend is individually deterministic and bitwise-stable across all
execution modes. Both define the degenerate `target <= 0` projection (e.g.
`K0 = 0` with a single item) as exactly zero — the JAX implementation
NaN-poisons in that corner (see "Deviations" below).

Convergence checks every `op_steps` iterations compute the implied
primal/dual objectives with `cub::DeviceSegmentedReduce` Max/Min plus
deterministic two-stage reductions that accumulate in `double` even for f32
data (`src/common.cuh`). All restart/termination decisions run **on the
device** (`k_halpern_logic`, `k_restart_update`), and by default the whole
Halpern loop executes as a **single CUDA graph** with a device-evaluated
WHILE conditional node (restarts are a nested IF node) - one launch and one
stream synchronization per solve. Three execution modes exist and are
**bitwise identical** (asserted by tests): the conditional whole-solve graph
(default), a host-driven loop over per-block graphs (`PRIA_NO_COND`), and
plain stream launches (`PRIA_NO_GRAPHS`); `verbose` uses the host-driven
loop.

Four further optimizations engage by size (gates re-tuned empirically for
the sort-free kernels):

- **persistent megakernel** (tiny instances, working set <= 2 MB): the
  Halpern inner iteration — combine, x projection, y/z projections with
  the reflected term inline — runs with software grid-wide barriers
  between phases (atomic counter + generation; plain launches, so they
  capture into the graphs unchanged; deadlock freedom from occupancy-sized
  grids). Two variants, selected empirically by precision: f32 runs the
  **whole op_steps block in one launch** with z segments block-owned from
  global memory (shared holds one row, occupancy maxed; -11% at 30x75),
  f64 runs **one launch per iteration** with z fused in shared (-15% at
  30x75; f64's doubled traffic makes shared-resident z rounds worth the
  occupancy cost, and the in-kernel loop variant loses residency to
  register pressure). Mid sizes reject every variant (+9..+23% at
  75x150/200x300 — not launch-bound), hence the tight gate. Never in
  batch solves (concurrent full-GPU spinning grids measured +63%
  ms/instance). Force with `PRIA_MEGA=1`/`0`.
- **reflect-inline consumers** (working set <= 2 MB): the y/z kernels
  compute `sigma*(2x' - x)` at the gathered position instead of reading a
  materialized reflection, skipping `k_reflect` (~-1% at 30x75; doubling
  the scattered gathers loses +6%/+23% at 200x300, hence the gate).
  `PRIA_SF_RFUSE=0` disables.
- **y/z overlap** (auto: only when the z projection takes the device-wide
  path and the working set is <= 16 MB): the y and z pipelines run on
  forked streams (-3% at 75x150 f64). With everything fused the fork/join
  events cost +15..+22%, so it is otherwise off. `PRIA_OVERLAP=1`/`0`
  forces (`PRIA_NO_OVERLAP` still means off).
- **tiled-transpose mirrors** (large instances, > 96 MB per variable —
  beyond the L40S's L2): the transpose-patterned scattered gathers
  overfetch DRAM badly; 32x32 shared-memory tile mirrors restage y, z and
  the reflected term between layouts so every kernel streams coalesced
  (-20% at 350x450 f32). Sort-free reads each gather source only once per
  iteration, so below that the staging does not amortize (+11% at 200x300
  f64 when forced). Force with `PRIA_MIRRORS=1`/`0`.

## Building

Requires CUDA 12+ (developed against CUDA 13.1, CCCL 3.x) and a GPU.
`CUDA_ARCH` has no default in this Makefile: the repository's `env.sh` is its
single home, so source that first, or pass `make CUDA_ARCH=<sm>`.  Everything
lands under `build/sm$(CUDA_ARCH)`, which is what lets a batch job's build for
one card sit beside an interactive session's build for another.

```
source ../env.sh
make            # the static and the shared library
make tools      # pria_solve_file, the file CLI the harness drives
make lib        # libpria.so, which scripts/tune_pria.py loads through ctypes
make test       # unit + end-to-end tests (needs a GPU)
make bench      # pria_bench, the standalone driver
```

`../scripts/build_cuda.sh` is the usual way in: it builds `tools` and `lib`
and installs both into `$PRIA_BIN_DIR`, where the harness looks for them.

## API

C-callable (`include/pria/pria.h`), f64 and f32 variants:

```c
#include <pria/pria.h>

pria_params p = pria_params_auto(/*max_iter=*/-1, /*tolerance=*/1e-4);
pria_result_info info;
/* K[n], H[n*t*t] row-major (upper triangle used), x/y/z out buffers n*t*t */
int rc = pria_solve_f64(n, t, K, K0, H, &p,
                        NULL, NULL, NULL,      /* warm start (optional) */
                        x, y, z, &info);
```

Outputs mirror the JAX solver: `x` is the primal solution; `y`, `z` are the
dual iterates normalized by `K_i` / `K0`. Warm-start duals use the same
normalized convention, so a returned solution warm-starts as a fixed point
(with even `t`, re-solving from a solution takes 0 iterations).
`pria_params_auto` reproduces `Parameters.with_auto_step` (same tuned
`omega`, `beta_*`, `theta_*`, `eta = 0.999/sqrt(2)`, `op_steps = 16`).

`pria_solve_batch_f64/f32` solves a packed batch of same-shaped instances
concurrently (a worker-thread fan-out, up to 16 workers; each instance owns
its streams and whole-solve graph, so instances overlap on the GPU).
Results are bitwise identical to solo solves - the solver never touches the
legacy stream or synchronizes the device, so concurrent captures from
several threads compose safely. Benchmark with `pria_bench --batch B`.

## Testing

- `tests/ref_impl.hpp` — an independent CPU double-precision reference of the
  whole algorithm on plain `[N][T][T]` arrays (no packed layouts).
- `test_layouts` — bijectivity/coverage of the index maps and segment
  geometry against the offset tables.
- `test_projection` — GPU projections (both backends: sort-free Michelot
  and the sort pipelines, fused and device-wide paths, all tile sizes,
  f32/f64, degenerate targets) against the CPU reference, elementwise.
- `test_step` — one PDHG operator application, a 6-step Halpern sequence, a
  forced restart, and the canonical-norm/gap reductions against the CPU
  reference (1e-9 elementwise).
- `test_solver` — end-to-end through the public API: primal/dual feasibility
  of the returned iterates, a **duality-gap certificate recomputed on the
  host on the original unscaled instance**, agreement with the CPU reference
  solver (identical iteration counts in practice), warm-start round trips,
  the vanilla-PDHG path, `max_iter`, f32, and — under both projection
  backends — tri-mode bitwise equivalence and batch == solo bitwise; the
  megakernel is checked bitwise-deterministic across runs (any grid-barrier
  race would break that) and equivalent to the launched-kernel path to
  rounding (nvcc contracts FMAs differently across inlining contexts).

Agreement with the other implementations is not checked here: it is a
property of the pair, so it lives one level up, in the experiment
repository's `scripts/check_correctness.py`, which drives this solver through
`harness/backends/pria_cuda.py` and compares objectives against HiGHS,
Gurobi, cuPDLPx and pria-jax on the same instances.

Observed result: objectives agree to ~1e-16 relative against pria-jax, and
**iteration counts are identical** on every instance whose joint cost is
nonzero (see "Deviations" for the `K0 = 0` corner).

## Performance

NVIDIA L40S, float32, tolerance 1e-4, `pria_bench` random instances
(end-to-end solve including allocation and transfers; JAX numbers are the
compiled-solver device time on same-size instances):

| items x T  | PRIA CUDA | iters | JAX (per-iter ratio) | PRIA f64 |
|-----------:|----------:|------:|---------------------:|---------:|
|   30 x 75  |   27 ms   |  560  |  43 ms (~1.6x)       |    73 ms |
|   75 x 150 |  138 ms   |  768  | 274 ms (~2.0x)       |   410 ms |
|  200 x 300 |  1.30 s   |  704  | 4.98 s (~3.7x)       |   3.75 s |
|  350 x 450 |  9.09 s   |  864  | 24.5 s (~2.6x)       |          |

(The optimization history on this table: whole-solve conditional graph
~-2% at 30x75; the tiled-transpose mirrors -26% at 350x450; fusing the
pre-projection kernels into the fused projections another
-4%/-6%/-12%/-10%; the sort-free Michelot projections another
-53%/-70%/-55%/-43% — by far the largest single win, the sorts dominated
per-iteration time everywhere; then warm thresholds + compaction for the
device-wide z (-3% at 350x450, -4% f64 200x300), the empirically re-tuned
overlap and mirror gates (-15% at 30x75, -10% f64 200x300 — both gates
had been calibrated against sort-era cost ratios), and the persistent
megakernel (-15% at 30x75 f64, then another -11% at 30x75 f32 from the
one-launch-per-op_steps-block f32 variant).)

Batched throughput (`pria_solve_batch_*`, 30x75 f32 @1e-4): 16.0
ms/instance at batch 8 and 15.4 at batch 32, vs ~39 ms/instance solving
the same instances sequentially - about 2.5x. Under the sort backend the
same batch managed only 1.35x: the sort pipelines' many tiny dependent
kernels set a per-kernel execution floor that concurrent instances could
not fill; the sort-free projections (one kernel per projection) largely
removed it. Batch workers never use the persistent megakernel (concurrent
full-GPU spinning grids destroy throughput), so batch results are bitwise
identical to solo solves made with `PRIA_MEGA=0`.

At tolerance 1e-6 the CUDA solver is far faster in wall clock. Its
convergence reductions accumulate in double precision, so the gap estimate
stays clean and termination fires as soon as the true gap crosses the
tolerance: in f32, 30x75 takes 0.45 s / 7.5k iters and 75x150 1.0 s / 5.6k
iters, vs 20.6 s / 210k iters and 208 s / 534k iters for JAX f32, whose f32
gap estimate plateaus near its precision floor. Beyond ~75x150 the f32
*iterates themselves* hit that floor, so use f64 for tight tolerances
(75x150: 2.7 s / 4.8k iters; 200x300: 12.2 s / 2.0k iters).

The megakernel extension experiment settled the remaining roadmap
question: the op_steps-block-in-one-launch variant with z decoupled from
shared memory was built and won at 30x75 f32 (-11%, adopted), but mid
sizes reject the megakernel with *every* variant (+9..+23% at
75x150/200x300 even with occupancy fixed) - beyond the tiny class,
per-iteration time is execution-bound, not launch-bound, and the launched
kernels' freedom to overlap beats the barrier lockstep. No further
size-class lever is known. Measured and rejected in this series: the
99 KB opt-in shared cap for wider fused rows (1 block/SM, ~7% slower than
the device-wide path it displaces), reflect-inlining beyond tiny sizes
(+6%/+23% at 200x300), block+z-shared and z-first phase ordering in the
f64 megakernel (+30% each), `__launch_bounds__` residency forcing
(register spills), the megakernel in batch mode (+63% ms/instance), and -
for the sort backend, scaffold kept under `PRIA_ZSORT_PROBE=1` - z-sort
permutation reuse (order stability ~0.1% of iterations).

## Deviations from the JAX implementation

Intentional, all verified by tests:

1. **Reductions accumulate in double** (norms, objectives, restart
   distances) even in f32 mode — strictly more accurate termination and
   omega updates.
2. Instance data may not exceed `2^31 - 1` packed entries per variable
   (`N * T(T+1)/2`); the solver returns an error rather than truncating.

Two behaviours the two implementations now share, and that a port to a third
would have to reproduce:

- **A degenerate budget** (`K0 = 0`, which every single-item instance has, or
  any `K_i = 0`) projects onto `{0}`, the only nonnegative vector summing to
  zero.
- **Warm-start duals are normalized**, the same convention as the outputs, so
  a returned solution round-trips as a fixed point: an even-period instance
  re-solves in zero iterations.

## Layout

```
include/pria/pria.h     public C API
src/layouts.cuh         dense layouts, index maps, segment geometry
src/projection.cuh      CUB projection pipelines (fused + device-wide)
src/solver.cuh          kernels + the Halpern/restart solver loop
src/common.cuh          error handling, deterministic reductions
src/pria.cu             extern "C" entry points
tests/                  CPU reference implementation + test binaries
tools/pria_solve_file   file-based CLI (what the harness drives)
bench/pria_bench.cu     standalone timing driver
```
