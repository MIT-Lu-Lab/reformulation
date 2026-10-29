from jax import random

from pria.jrp_problem import JrpProblem


def test_random_instances() -> None:
    seed_keys = random.PRNGKey(0xfedcba)
    items_keys = random.PRNGKey(0xdeadbeef)
    time_period_keys = random.PRNGKey(0xabcdef)
    n = 50

    seeds = random.randint(seed_keys, (n,), 1, 2 ** 16)
    items = random.randint(items_keys, (n,), 1, 10)
    time_periods = random.randint(time_period_keys, (n,), 1, 20)
    for seed, item, time_period in zip(seeds, items, time_periods, strict=True):
        JrpProblem.random_instance(seed, item, time_period)

if __name__ == "__main__":
    test_random_instances()
    print("test_jrp_problem: all tests passed")
