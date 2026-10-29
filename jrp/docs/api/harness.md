# API: harness and baselines

Not the library. These modules are not importable from an installed `pria`;
they live at the repository root and are reached by the scripts.

## Configuration

```{eval-rst}
.. automodule:: harness.config
```

## Instances

```{eval-rst}
.. automodule:: harness.instances
```

## Results

```{eval-rst}
.. automodule:: harness.results
```

## Peak memory

```{eval-rst}
.. automodule:: harness.memory
```

## Figures

The figures and tables the benchmark CSVs are read back for. Drawing lives here
rather than in the script so a notebook can import it.

```{eval-rst}
.. automodule:: harness.figures
```

## The backend registry

```{eval-rst}
.. automodule:: harness.registry
```

## PRIA adapters

```{eval-rst}
.. automodule:: harness.backends.pria_jax
```

```{eval-rst}
.. automodule:: harness.backends.pria_cuda
```

```{eval-rst}
.. automodule:: harness.pria_cuda_library
```

## Baselines

The joint replenishment LP is built here, once, and all three baselines go
through it. An independent construction of the same LP — dense, looped, every
index spelled out — lives in the `model` check of `scripts/run_checks.sh`,
which is what lets their agreement count as evidence rather than as a
tautology.

```{eval-rst}
.. automodule:: baselines.model
```

```{eval-rst}
.. automodule:: baselines.highs_backend
```

```{eval-rst}
.. automodule:: baselines.gurobi_backend
```

```{eval-rst}
.. automodule:: baselines.cupdlpx_backend
```
