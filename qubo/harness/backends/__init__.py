"""Adapters for the PRAQ implementations.

One module per implementation, each exposing
``solve(instance, *, tolerances, time_limit) -> Result``.  A backend listed in
`harness.registry.SAMPLED_MEMORY` takes a further `sample_memory` keyword.
The LP baselines expose the same signature from `baselines/`.
"""
