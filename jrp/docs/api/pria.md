# API: the library

`pria-jax`. Installable on numpy and jax alone.

{py:mod}`pria.prelude` names the public surface; everything below it is
internals, documented here because the CUDA port has to reproduce it.

## Public surface

```{eval-rst}
.. automodule:: pria.prelude
   :no-members:
```

## Instances

```{eval-rst}
.. automodule:: pria.jrp_problem
```

## The solver

```{eval-rst}
.. automodule:: pria.solver
   :exclude-members: StepArguments, Permutations
```

## The dense upper-triangular layouts

```{eval-rst}
.. automodule:: pria.upper_triangular
```

## Projections

```{eval-rst}
.. automodule:: pria.projection
```

## The canonical metric

```{eval-rst}
.. automodule:: pria.halpern_operator
```

## Convergence logging

```{eval-rst}
.. automodule:: pria.convergence_log
```

## Adversarial instances

```{eval-rst}
.. automodule:: pria.adversarial
   :no-members:
```
