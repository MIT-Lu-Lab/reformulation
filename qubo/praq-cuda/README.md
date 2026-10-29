# praq-cuda

A CUDA port of [`../praq-jax`](../praq-jax), which is the specification: the
port is correct when `scripts/check_correctness.py --praq-jax --praq-cuda`
agrees on random instances across sizes and densities.

## Building

```sh
source ../env.sh          # sets CUDA_ARCH, NVCC, PRAQ_BIN_DIR
make tools                # build/praq_solve_file, the CLI the harness drives
make test                 # build/test_solver, needs a GPU
```

`CUDA_ARCH` has no default in this Makefile; `env.sh` is its only home, and the
build stops with a message rather than guessing an SM. `../scripts/build_cuda.sh`
does the build and installs the binary into `$PRAQ_BIN_DIR` where the harness
looks for it.

## Layout

| File | What it is |
|---|---|
| `include/praq/praq.h` | the C interface: `PraqInstance`, `PraqParameters`, `PraqResult`, `praq_solve` |
| `src/solver.cuh` | the port, templated on the real type; kernels and the host loop |
| `src/praq.cu` | the entry points: validation, dispatch on precision |
| `tools/praq_solve_file.cu` | the file CLI; its wire format is the contract with `harness/backends/praq_cuda.py` |
| `tests/test_solver.cu` | the solver against an enumerated half-integral optimum |

## What is different from praq-jax

**The build is sparse.** `praq-jax` gives every pair $i < j$ an edge so that jit
sees static shapes, and pads absent edges with weight 0. Here the instance's
own edge list is the half-edge set, so memory and work are $O(n + m)$ and the
degree is the true incident count. Both satisfy $D \succeq S^\top S$, which is
all convergence needs, and $\|A\|_2 = 1$ holds for both. On a dense instance
the two take the same path; on a sparse one this build is the better
conditioned and takes fewer iterations, to the same bounds.

**The scatter is a gather.** Each node keeps a CSR list of its incident
half-edges, so $S^\top y$ is a per-node sum rather than atomics. The list is
strided over a group of lanes sized from the average degree and summed in a
fixed butterfly, so the order of additions is a function of the instance, and
the iterates are bitwise reproducible from run to run.

**The scalar logic runs on the host.** Every `op_steps` iterations (a number,
or with `op_steps <= 0` the size rule `praq_check_interval` in `praq.h`) one
reduction pass produces nine sums and copies them back; the restart tests, the
omega update and the termination test are then evaluated exactly as
`solver.py` writes them. Termination is on the relative gap at the original
scale, both bounds multiplied back by the objective scale first.

**No warm phase.** `with_auto_step` sets `warm_steps = 0` and
`PraqParameters` exposes no way to raise it, so the vanilla warm loop of
`solve_qubo_problem_warm_start` is not ported.

**Precision.** `use_float32` switches the iterates to `float`; the reductions
and the bookkeeping stay in double. `praq-jax` does everything in the array's
dtype, so the two float32 modes are not bitwise alike.
