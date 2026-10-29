"""LP baselines: Gurobi and cuPDLPx.

Deliberately outside the PRAQ library.  `praq-jax/` and `praq-cuda/` never
import this package; it exists so the harness has something to compare against,
and it is the only place the standard linearization is built explicitly --
PRAQ solves a reformulation and never forms this LP at all.
"""
