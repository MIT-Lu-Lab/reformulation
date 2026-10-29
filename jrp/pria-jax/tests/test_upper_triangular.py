from functools import partial

from jax import random

from pria.upper_triangular import *


def test_reshapes() -> None:
    key = random.PRNGKey(0)
    key1, key2 = random.split(key, 2)

    matrices_even = jnp.triu(random.randint(key1, (100, 20, 20), -1000, 1000))
    matrices_odd = jnp.triu(random.randint(key2, (100, 21, 21), -1000, 1000))

    for matrices in [matrices_even, matrices_odd]:
        n = matrices.shape[1]

        col_dense = vmap(densify_columnwise)(matrices)
        row_dense = vmap(densify_rowwise)(matrices)

        col_row_perm = col_to_row_permutation(n)
        col_dense_to_row = vmap(partial(col_to_row, perm=col_row_perm))(col_dense)
        row_col_perm = row_to_col_permutation(n)
        row_dense_to_col = vmap(partial(row_to_col, perm=row_col_perm))(row_dense)

        col_masks = jnp.broadcast_to(dense_columnwise_mask(matrices.shape[1]), row_dense_to_col.shape)
        row_masks = jnp.broadcast_to(dense_rowwise_mask(matrices.shape[1]), col_dense_to_row.shape)

        col_dense = col_dense * col_masks
        row_dense = row_dense * row_masks

        col_dense_to_row = col_dense_to_row * row_masks
        row_dense_to_col = row_dense_to_col * col_masks

        assert jnp.all(col_dense_to_row == row_dense)
        assert jnp.all(row_dense_to_col == col_dense)


def test_densify_sparsify() -> None:
    key = random.PRNGKey(1)
    key1, key2 = random.split(key, 2)

    n = 100
    t = 20

    matrices_even = jnp.triu(random.randint(key1, (n, t, t), -1000, 1000))
    matrices_odd = jnp.triu(random.randint(key2, (n, t+1, t+1), -1000, 1000))

    for matrices in [matrices_even, matrices_odd]:
        col_dense = vmap(densify_columnwise)(matrices)
        row_dense = vmap(densify_rowwise)(matrices)

        col_sparse = vmap(sparsify_columnwise)(col_dense)
        row_sparse = vmap(sparsify_rowwise)(row_dense)

        assert jnp.all(col_sparse == matrices)
        assert jnp.all(row_sparse == matrices)


def test_densify_3d() -> None:
    key = random.PRNGKey(2)
    key1, key2 = random.split(key, 2)

    count = 100
    n = 10
    t = 20

    matrices_even = jnp.triu(random.randint(key1, (count, n, t, t), -1000, 1000))
    matrices_odd = jnp.triu(random.randint(key2, (count, n + 1, t + 1, t + 1), -1000, 1000))

    for matrices in [matrices_even, matrices_odd]:
        n = matrices.shape[1]
        m = matrices.shape[2]

        m_dense = vmap(densify_tensor_dim2)(matrices)
        m_sparse = vmap(sparsify_tensor_dim2)(m_dense)

        assert jnp.all(m_sparse == matrices)

        col_dense = vmap(vmap(densify_columnwise))(matrices)
        dim2_to_col_perm = dim2_to_col_permutation(n, m)
        col_dense_2 = vmap(partial(dim2_to_col, perm=dim2_to_col_perm))(m_dense)

        shapes = col_dense.shape
        masks = jnp.broadcast_to(dense_columnwise_mask(matrices.shape[-1]), shapes)

        assert jnp.all(col_dense * masks == col_dense_2 * masks)

        col_to_dim2_perm = col_to_dim2_permutation(n, m)
        m_dense_2 = vmap(partial(col_to_dim2, perm=col_to_dim2_perm))(col_dense)
        shapes = m_dense.shape
        masks = jnp.broadcast_to(dim2_mask(matrices.shape[1], matrices.shape[-1]), shapes)

        assert jnp.all(m_dense * masks == m_dense_2 * masks)

if __name__ == "__main__":
    test_reshapes()
    test_densify_sparsify()
    test_densify_3d()
    print("test_upper_triangular: all tests passed")
