# API: harness and baselines

Not the library. These modules are not importable from an installed `praq`;
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

## PRAQ adapters

```{eval-rst}
.. automodule:: harness.backends.praq_jax
```

```{eval-rst}
.. automodule:: harness.backends.praq_cuda
```

```{eval-rst}
.. automodule:: harness.praq_cuda_library
```

## Baselines

The standard linearization is built here, once, and all three LP baselines go
through it. The max-flow baseline does not: it is a second, independent
construction of the same bound, which is what lets agreement between them count
as evidence.

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

```{eval-rst}
.. automodule:: baselines.maxflow_backend
```
