from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, config, random, vmap

config.update('jax_enable_x64', True)

from pria.projection import (
    double_projection_3d_dim2,
    double_projection_upper_triangular,
    double_projection_upper_triangular_row,
    double_simplex_projection,
    simplex_projection,
    simplex_projection_3d_triangular,
    simplex_projection_columnwise,
    simplex_projection_qp,
    simplex_projection_upper_triangular,
    simplex_projection_upper_triangular_row,
)
from pria.upper_triangular import (
    densify_columnwise,
    densify_rowwise,
    densify_tensor_dim2,
    sparsify_columnwise,
    sparsify_rowwise,
    sparsify_tensor_dim2,
)


def approx_equal_arrays(a: Array, b: Array, eps: float = 1e-8) -> bool:
    if a.shape != b.shape:
        return False

    return jnp.all(jnp.abs(a - b) < 2.0 * eps).item()


def compare_to_vectorized(seed: int, n: int, l: int,
                          f: Callable[[Array, float, float, Array], Array]) -> None:
    tolerance = 1e-12
    key1, key2 = random.split(random.PRNGKey(seed), 2)

    vecs = random.normal(key1, (l, n), dtype=jnp.float64)
    s = random.uniform(key2, (n,), jnp.float64, 0.1, 10.0)

    gpu_results = simplex_projection_columnwise(vecs, s)

    f_results_list = [f(v, i, tolerance / 2.0, r) for v, i, r in zip(vecs.T, s, gpu_results.T, strict=True)]
    f_results = jnp.stack(f_results_list, axis=1)

    for inp, v1, v2, i in zip(vecs.T, f_results.T, gpu_results.T, s, strict=True):
        assert jnp.abs(v1.sum() - i) <= tolerance
        assert jnp.abs(v2.sum() - i) <= tolerance

        v1_obj = jnp.linalg.norm(v1 - inp)
        v2_obj = jnp.linalg.norm(v2 - inp)

        if not jnp.abs(v1_obj - v2_obj) <= tolerance:
            print("\nOther: ", v1)
            print("Sum: ", jnp.sum(v1))
            print("\nOurs: ", v2)
            print("Sum: ", jnp.sum(v2))
            print("\nOther - Ours: ", jnp.round(v1 - v2, -int(jnp.log10(tolerance))))
            print("Sum: ", jnp.sum(jnp.abs(v1 - v2)))
            print("\nInput", inp)

            raise AssertionError(inp)

def test_versus_qp() -> None:
    n = 500
    l = 100

    def f(v, s=1.0, tolerance=1e-10, x0=None):
        return simplex_projection_qp(v, s, tolerance, x0=x0)

    compare_to_vectorized(0xbadbead, n, l, f)


def test_upper_triangular() -> None:
    key = random.PRNGKey(0)

    n = 100
    l = 50
    matrices = jnp.triu(random.normal(key, (n, l, l), dtype=jnp.float64))

    def test_matrices(mats):
        result_vectorized_row = vmap(simplex_projection_upper_triangular_row)(mats).block_until_ready()
        result_vectorized = vmap(simplex_projection_upper_triangular)(mats).block_until_ready()

        def apply_non_vectorized(matrix):
            res = [simplex_projection(col[:j+1]) for (j, col) in enumerate(matrix.T)]
            res = [jnp.pad(a, (0, matrix.shape[0] - len(a))) for a in res]
            return jnp.stack(res, axis=1)

        def apply_non_vectorized_row(matrix):
            res = [simplex_projection(row[j:]) for (j, row) in enumerate(matrix)]
            res = [jnp.pad(a, (matrix.shape[1] - len(a), 0)) for a in res]
            return jnp.stack(res, axis=0)

        results_non_vectorized = jnp.stack([apply_non_vectorized(m) for m in mats])
        results_non_vectorized_row = jnp.stack([apply_non_vectorized_row(m) for m in mats])

        for i in range(mats.shape[0]):
            if not approx_equal_arrays(result_vectorized[i], results_non_vectorized[i]):
                print("\n")
                print(mats[i])
                print(result_vectorized[i])
                print(results_non_vectorized[i])

                raise AssertionError("Upper triangular test column failure")

            if not approx_equal_arrays(result_vectorized_row[i], results_non_vectorized_row[i]):
                print("\n")
                print(mats[i])
                print(result_vectorized_row[i])
                print(results_non_vectorized_row[i])

                raise AssertionError("Upper triangular test row failure")

    test_matrices(matrices)
    test_matrices(jnp.abs(matrices))
    test_matrices(-1.0 * jnp.abs(matrices))


def test_3d_projection() -> None:
    count = 10
    n = 5
    t = 10

    tolerance = 1e-4
    seed = random.PRNGKey(0)

    a = jnp.triu(random.normal(seed, (count, n, t, t)))
    a_proj = vmap(simplex_projection_3d_triangular)(a)

    for m in a_proj:
        for j in range(t):
            assert jnp.abs(m[:, j, :].sum() - 1.0) <= tolerance


def test_double_projection() -> None:
    key = random.PRNGKey(0)
    key1, key2, key3, key4 = random.split(key, 4)

    n = 3
    l = 10

    vecs = random.normal(key1, (l, n), dtype=jnp.float64)
    partitions = random.randint(key2, (n,), 1, l-1)
    s_val_1 = random.uniform(key3, (n,), jnp.float64, 0.1, 10.0)
    s_val_2 = random.uniform(key4, (n,), jnp.float64, 0.1, 10.0)

    our_projections = jnp.zeros((l, n))
    for i, (v, part, s1, s2) in enumerate(zip(vecs.T, partitions, s_val_1, s_val_2, strict=True)):
        our_projections = our_projections.at[:, i].set(double_simplex_projection(v, part, s1, s2))

    correct_projections = jnp.zeros((l, n))
    for i, (v, part, s1, s2) in enumerate(zip(vecs.T, partitions, s_val_1, s_val_2, strict=True)):
        v1 = v[:part]
        v2 = v[part:]

        v1_proj = simplex_projection(v1, s1)
        v2_proj = simplex_projection(v2, s2)

        correct_projections = correct_projections.at[:, i].set(jnp.hstack([v1_proj, v2_proj]))

    assert approx_equal_arrays(our_projections, correct_projections, 1e-8)


def test_double_projection_ut() -> None:
    key = random.PRNGKey(0)
    key1, key2, key3, _key4 = random.split(key, 4)

    n = 100
    dim2_count = 10
    l = 20

    matrices = jnp.triu(random.normal(key1, (n, l, l), dtype=jnp.float64))
    s_vals = random.uniform(key3, (n,), jnp.float64, 0.1, 10.0)

    for s_val, matrix in zip(s_vals, matrices, strict=True):
        proj = simplex_projection_upper_triangular(matrix, s_val)
        ut_proj = double_projection_upper_triangular(densify_columnwise(matrix), s_val)
        proj_2 = sparsify_columnwise(ut_proj)

        assert approx_equal_arrays(proj, proj_2)

        proj_row = simplex_projection_upper_triangular_row(matrix, s_val)
        ut_proj_row = double_projection_upper_triangular_row(densify_rowwise(matrix), s_val)
        proj_row_2 = sparsify_rowwise(ut_proj_row)

        assert approx_equal_arrays(proj_row, proj_row_2)

    matrices = jnp.triu(random.normal(key2, (n, dim2_count, l, l), dtype=jnp.float64))
    for s_val, matrix in zip(s_vals, matrices, strict=True):
        proj = simplex_projection_3d_triangular(matrix, s_val)
        ut_proj = double_projection_3d_dim2(densify_tensor_dim2(matrix), s_val)
        proj_2 = sparsify_tensor_dim2(ut_proj)

        assert approx_equal_arrays(proj, proj_2)


if __name__ == '__main__':
    test_versus_qp()
    test_upper_triangular()
    test_3d_projection()
    test_double_projection()
    test_double_projection_ut()
    print("test_projection: all tests passed")
