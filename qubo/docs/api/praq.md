# API: the library

`praq-jax`. Installable on numpy and jax alone.

{py:mod}`praq.prelude` names the public surface; everything below it is
internals, documented here because the CUDA port has to reproduce it.

## Public surface

```{eval-rst}
.. automodule:: praq.prelude
   :no-members:
```

## Instances

```{eval-rst}
.. automodule:: praq.qubo_problem
```

## The solver

```{eval-rst}
.. automodule:: praq.solver
   :exclude-members: StepArguments
```

## The half-edge structure

```{eval-rst}
.. automodule:: praq.half_edge
```

## Projections

```{eval-rst}
.. automodule:: praq.projection
```

## The canonical metric

```{eval-rst}
.. automodule:: praq.halpern_operator
```

## Exhaustive reference

```{eval-rst}
.. automodule:: praq.brute_force
```
