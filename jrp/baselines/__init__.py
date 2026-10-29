"""LP baselines: HiGHS, Gurobi and cuPDLPx.

Deliberately outside the PRIA library.  `pria-jax/` and `pria-cuda/` never
import this package; it exists so the harness has something to compare
against, and it is the only place the joint replenishment LP is built
explicitly -- PRIA solves a saddle reformulation and never forms this LP at
all.
"""
