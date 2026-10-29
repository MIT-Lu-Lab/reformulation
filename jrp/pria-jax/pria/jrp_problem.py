import os
from dataclasses import dataclass
from functools import partial

assertions_enabled = os.getenv('ASSERTIONS_ENABLED') is not None

import jax.numpy as jnp
from jax import Array, random
from jax.tree_util import register_dataclass


@partial(register_dataclass, data_fields=["order_costs", "holding_costs", "joint_order_cost"], meta_fields=[])
@dataclass(frozen=True)
class JrpProblem:
    order_costs: Array
    holding_costs: Array
    joint_order_cost: float

    def t(self):
        return self.holding_costs.shape[-1]

    def n(self):
        return self.order_costs.shape[0]

    @classmethod
    def new(cls,
                        order_costs: Array,
                        joint_order_cost: float,
                        holding_costs: Array,
                     ) -> "JrpProblem":
        """
        Construct a new Joint Replenishment Problem (JRP). The parameters reference the names
        given by this paper: https://people.orie.cornell.edu/shmoys/pdf/jrp-journal.pdf

        The problem, with K0 the joint order cost, ``K_i == order_costs[i]``
        and ``H_{ist} == holding_costs[i][s][t]``::

            minimize (over x, y, z)
                  sum_{s=1}^{T} K0 * z_s
                + sum_{i=1}^{N} sum_{s=1}^{T} K_i * y_{is}
                + sum_{i=1}^{N} sum_{t=1}^{T} sum_{s=1}^{t} H_{ist} * x_{ist}
            subject to
                sum_{s=1}^{t} x_{ist} = 1,    for all i in [N], t in [T]
                x_{ist} - y_{is} <= 0,        for all i in [N], t in [T], s in [t]
                x_{ist} - z_s    <= 0,        for all i in [N], t in [T], s in [t]
                x_{ist}, y_{is}, z_s >= 0,    for all i in [N], t in [T], s in [t]

        :param joint_order_cost: the scalar joint order cost
        :param order_costs: order_costs[i] == K_i
        :param holding_costs: holding_costs[i][s][t] == h_{st}^i
        """
        holding_costs = jnp.triu(holding_costs)

        if assertions_enabled:
            assert len(order_costs.shape) == 1,   "Order costs should have 1 dimension"
            assert len(holding_costs.shape) == 3, "Holding costs should have 3 dimensions"

            assert order_costs.shape[0] > 0,                          "order_costs is empty"
            assert len(order_costs) == len(holding_costs), "Mismatched first dimension shapes"

            assert holding_costs.shape[1] > 0,                       "holding_costs second dimension is empty"
            assert holding_costs.shape[-1] == holding_costs.shape[-2], "holding_costs dimension mismatch"

            assert jnp.all(holding_costs >= 0.0),                  "holding_costs should be non-negative"
            assert jnp.all(jnp.diff(holding_costs, axis=2) >= 0), \
                "holding costs should be monotonically increasing"

        return cls(order_costs, holding_costs, joint_order_cost)

    @classmethod
    def stack(cls, instances: list["JrpProblem"]) -> "JrpProblem":
        """
        Convert a struct-of-arrays `instances` to an array-of-structs.

        :param instances: a list of JrpProblems to stack
        :return: an JrpProblem where each field is an array with the input data
        """
        order_costs       = jnp.stack([i.order_costs for i in instances])
        holding_costs     = jnp.stack([i.holding_costs for i in instances])
        joint_order_costs = jnp.array([i.joint_order_cost for i in instances])

        return cls(order_costs, holding_costs, joint_order_costs)

    @classmethod
    def random_instance(cls, seed: int, items: int, time_periods: int, dtype=jnp.float32) -> "JrpProblem":
        """
        Construct a random JrpProblem of the desired size. All costs are
        integer-valued (order costs at least 1).

        :param dtype: The type of floats to use
        :param seed: An integer seed for the generation
        :param items: The number of items for the instance
        :param time_periods: The number of time periods of the instance
        :return: A random JrpProblem
        """
        master_key = random.PRNGKey(seed)
        hold_mult_key, hold_eps_key, order_key = random.split(master_key, 3)

        idx_t = jnp.broadcast_to(jnp.arange(time_periods), (time_periods, time_periods))
        diff = jnp.broadcast_to(idx_t - idx_t.T, (items, time_periods, time_periods))

        holding_cost_multipliers = random.uniform(hold_mult_key, (items, 1, 1), dtype, 0, 1)
        holding_cost_addends = random.uniform(
            hold_eps_key, (items, time_periods, time_periods), dtype, 0,
            holding_cost_multipliers)
        holding_costs = jnp.round(jnp.triu(holding_cost_multipliers * diff + holding_cost_addends, 1))

        all_order_costs = jnp.maximum(jnp.round(random.uniform(
            order_key, (items + 1,), dtype, 1.0, max(time_periods, 1.01))), 1.0)
        order_costs = all_order_costs[:-1]
        joint_order_cost = all_order_costs[-1].item() if items > 1 else 0.0

        return JrpProblem.new(order_costs, joint_order_cost, holding_costs)

    def objective_value(self, primal: Array) -> float:
        """The LP objective at `primal`, as a Python float.

        `primal_objective` is where the arithmetic lives; this is the spelling
        for a caller who wants a number rather than a traced array.
        """
        return float(primal_objective(self, primal))

    def as_type(self, dtype: jnp.dtype) -> "JrpProblem":
        return JrpProblem(self.order_costs.astype(dtype),
                                self.holding_costs.astype(dtype),
                                self.joint_order_cost)

    def normalized(self) -> "JrpProblem":
        objective_vector = jnp.append(self.holding_costs.ravel(), self.order_costs)
        scale_val = 1.0 + jnp.linalg.norm(jnp.append(objective_vector, jnp.array([self.joint_order_cost])))
        return JrpProblem(self.order_costs / scale_val,
                                self.holding_costs / scale_val,
                                self.joint_order_cost / scale_val.item())

    @classmethod
    def zero_instance(cls, items: int, time_periods: int, dtype=jnp.float32) -> "JrpProblem":
        holding_costs = jnp.zeros((items, time_periods, time_periods), dtype=dtype)
        order_costs = jnp.zeros(items, dtype=dtype)
        joint_order_cost = 0.0

        return JrpProblem.new(order_costs, joint_order_cost, holding_costs)

def primal_objective(instance: "JrpProblem", primal: Array) -> Array:
    """The LP objective at `primal`, with y and z at their cheapest feasible values.

    The one home for the primal bound, because two callers need it to mean the
    same thing: `JrpProblem.objective_value` for a person holding a solution,
    and `pria.solver.relaxation_bounds` for the certificate the solver
    reports.  Given `x`, the constraints `x_ist <= y_is` and `x_ist <= z_s`
    are satisfied most cheaply at ``y_is = max_t x_ist`` and
    ``z_s = max_{i,t} x_ist``, and the maxima may run over the whole row
    because `x` is nonnegative and zero below the diagonal.

    :param instance: the JRP instance
    :param primal: a primal point, dense [N][T][T] and upper triangular
    :return: the objective value, unreduced to a Python float so this stays
        usable inside a traced computation
    """
    return (jnp.einsum("i,is->", instance.order_costs, primal.max(axis=2))
            + instance.joint_order_cost * primal.max(axis=(0, 2)).sum()
            + jnp.einsum("ist,ist->", primal, instance.holding_costs))
