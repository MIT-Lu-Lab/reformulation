# Reformulation for PDLP

This repository has the code for the paper *Reformulation Techniques for PDLP
with Applications to QUBO and Joint Replenishment*, by Jacob Dentes, Haihao Lu
and Jinwen Yang.

Every PDHG iteration on the original formulation multiplies one vector by the
constraint matrix. It multiplies another vector by the transpose of that
matrix. Those two products are the largest term in the cost of such an
iteration. The number of iterations depends on the sharpness of the saddle
problem and on the tolerance. A reformulation changes the cost of an iteration
and the number of iterations.

The paper studies two LP families. Each reformulation absorbs every constraint
of its family into the projection steps of PDHG. Each subdirectory below has
one solver, the wrappers for the baseline solvers, and a harness. A harness is
the set of scripts that run the solvers and record the times.

| | Problem | Solver | Baselines compared against the solver |
|---|---|---|---|
| [`qubo/`](qubo/) | LP relaxation of the standard linearization of QUBO, whose optimum is the roof dual | **PRAQ**, in CUDA and in JAX | cuPDLPx, Gurobi, HiGHS, an OR-Tools max-flow solver |
| [`jrp/`](jrp/) | LP relaxation of the joint replenishment problem | **PRIA**, in CUDA and in JAX | cuPDLPx, Gurobi, HiGHS |

The baseline solvers are external. This repository has only the wrappers that
call them.

## Getting started

Install each child separately. `qubo/README.md` and `jrp/README.md` give the
steps. Both start here.

```sh
cd qubo          # or: cd jrp
source env.sh    # sourcing sets the variables in the current shell
```

Each child README has sections for installation, experiments, layout, checks,
documentation, instances and known gaps.

In each child, the JAX implementation is the reference for everything else. It
runs on a CPU. The two CUDA implementations and cuPDLPx require a GPU. Gurobi
requires a licence. A backend without its licence or its GPU reports
`unavailable`. The run continues through the other backends. The run notes
name each backend that reported `unavailable`.

## Reproducing the paper

The harness generates the benchmark instances at run time. The generator
produces the same instance from the same family, size and seed. Each pass
regenerates the instances the last pass measured. The `## Instances` section
of each child gives the families and the sizes. The `## Experiments` section
of each child gives the benchmark commands.

Solve times vary with the hardware. Compare two solvers by the ratio of their
times on one machine.

We ran the GPU backends on one NVIDIA H100 80GB. We ran the other backends on
CPU nodes. Each solve ran under the harness time limit, `TIME_LIMIT` in
`harness/config.py`.

## Top-level directories

| Path | Contents |
|---|---|
| [`qubo/`](qubo/) | the QUBO case study. PRAQ, the baseline wrappers and the harness |
| [`jrp/`](jrp/) | the joint replenishment case study. PRIA, the baseline wrappers and the harness |
| `.github/workflows/` | one CI workflow per child |

Each child has its own `env.sh`, `requirements.txt`, `baselines/`, `harness/`,
`docs/`, `scripts/` and two solver directories. No code in `qubo/` imports
code in `jrp/`. The scripts in each child resolve paths against that child. A child
also runs after you copy it elsewhere. Each CI workflow is triggered
by changes under its own child and by changes to the workflow file.

## Cluster runs

Each child has two Slurm files in `scripts/`. One runs a benchmark pass. The
other tunes the restart schedule. Every `#SBATCH --partition` line names
`gpu`. Set that name to a partition of your own cluster before you submit.

Each `scripts/*.sbatch` file wraps a command you can run in a shell. Each
child README gives the shell command before the `sbatch` command. The GPU
backends need a GPU in either form.

## License

The MIT License covers this code. See [LICENSE](LICENSE).
