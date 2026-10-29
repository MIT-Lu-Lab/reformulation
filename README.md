# Reformulation for PDLP

Code for **Reformulation Techniques for PDLP with Applications to QUBO and
Joint Replenishment**, by Jacob Dentes, Haihao Lu and Jinwen Yang.

A PDHG-type LP solver reads its formulation twice over: the cost of an
iteration is the cost of multiplying by the constraint matrix, and the number
of iterations is set by the geometry of the saddle problem. This repository
holds the paper's two case studies. Each is an LP family rewritten so that its
constraints are absorbed into the projection step of PDHG, together with the
harness that measures the rewrite against a GPU LP solver running the original
formulation.

| | Problem | Solver | Backends measured against it |
|---|---|---|---|
| [`qubo/`](qubo/) | LP relaxation of QUBO, equivalently the roof dual | **PRAQ**, in CUDA and in JAX | cuPDLPx, Gurobi, HiGHS, an OR-Tools max-flow oracle |
| [`jrp/`](jrp/) | LP relaxation of the joint replenishment problem | **PRIA**, in CUDA and in JAX | cuPDLPx, Gurobi, HiGHS |

The two share a layout, a set of commands and a harness design, so what you
learn in one carries to the other.

## Getting started

Each child is self-contained and installs on its own:

```sh
cd qubo          # or: cd jrp
source env.sh    # every path the project uses; source it, do not execute it
```

[`qubo/README.md`](qubo/README.md) and [`jrp/README.md`](jrp/README.md) are the
instructions, and each covers the same five things in the same order:
installing the Python and CUDA sides, checking the install, reproducing the
paper's experiments, drawing the figures, and tuning the restart schedule.
Each child's `## Instances` section is how to point its solver at problems of
your own.

A GPU is optional. The JAX implementation is the reference for everything else
and runs on CPU, and each backend that wants a licence or a card reports itself
unavailable, so a pass on a laptop runs the subset it can and says what it
skipped.

## Reproducing the paper

Benchmark instances are generated from fixed seeds rather than stored, so a
pass builds its own: each child's `## Instances` section gives the families and
sizes, and its `## Experiments` section gives the commands. Instances and
iteration counts follow from the seeds. Solve times are measurements of the
machine that took them, so a rerun reproduces the comparison rather than the
seconds.

The paper's runs used one NVIDIA H100 80GB for the GPU backends and CPU nodes
for the rest, under a five-minute per-instance limit.

## Layout

| Path | What it is |
|---|---|
| [`qubo/`](qubo/) | the QUBO case study: PRAQ, its baselines, its harness |
| [`jrp/`](jrp/) | the joint replenishment case study: PRIA, its baselines, its harness |
| `.github/workflows/` | one CI workflow per child, each filtered to that child's paths |

Each child owns its own `env.sh`, `requirements.txt`, `docs/` and `scripts/`,
so neither depends on the other and neither depends on this directory.

## Cluster runs

Both children carry `scripts/*.sbatch` for Slurm. The `#SBATCH --partition`
lines are the one site-local thing in this repository; set them to your
cluster's partitions before the first submission. Every experiment also runs
without a scheduler, which is what the child READMEs show first.

## License

MIT. See [LICENSE](LICENSE).
