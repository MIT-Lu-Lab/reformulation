"""
Provides functionality for holding an upper triangular matrix in a denser format
"""
import jax.numpy as jnp
from jax import Array, vmap


def densify_columnwise(v: Array) -> Array:
    """
    Takes the square upper triangular matrix v and stacks it columnwise.
    The last column will be stacked on the first, the second-to-last on the second,
    and so on. If v has an odd number of columns it will pair with itself.
    Each stacked pair lies along the last axis of the output, so the
    simplex vectors are contiguous in memory

    :param v: An n x n 2D Jax array, required to satisfy np.triu(v) == v
    :return: a ((n+1)//2) x (n+1) 2D Jax array containing the upper triangular part of v
    """
    return _densify_columnwise_tall(v).T


def _densify_columnwise_tall(v: Array) -> Array:
    # the transposed, (n+1) x ((n+1)//2) layout used internally
    n = v.shape[0]
    new_cols = (n + 1) // 2

    left_indices = jnp.arange(new_cols)
    right_indices = n - 1 - left_indices

    left = v[:, left_indices]
    left = jnp.vstack([left, jnp.zeros(new_cols, dtype=v.dtype)])
    right = v[::-1, right_indices]
    right = jnp.vstack([jnp.zeros(new_cols, dtype=v.dtype), right])

    out = left + right

    return out


def sparsify_columnwise(v: Array) -> Array:
    """
    The inverse operation to densify_columnwise

    :param v: The ((n+1)//2) x (n+1) column dense 2D array to sparsify
    :return: An upper triangular n x n 2D array representing v sparsified
    """
    v = v.T
    n = v.shape[0] - 1
    v_mirror = v[::-1, ::-1]
    if n % 2 == 1:
        v_mirror = v_mirror[:, 1:]

    return jnp.triu(jnp.hstack([v[:-1, :], v_mirror[:-1, :]]))


def densify_rowwise(v: Array) -> Array:
    """
    Like densify_columnwise but stacks rows instead

    :param v: The n x n 2D array to stack
    :return: The resulting ((n+1)//2) x (n+1) 2D Array
    """
    return jnp.rot90(_densify_columnwise_tall(jnp.rot90(v.T, 2)), -2).T[::-1, :]


def sparsify_rowwise(v: Array) -> Array:
    """
    The inverse operation to densify_rowwise

    :param v: The  ((n+1)//2) x (n+1) row dense 2D array to sparsify
    :return: An upper triangular n x n 2D array representing v sparsified
    """
    perm = row_to_col_permutation(v.shape[1] - 1)
    return sparsify_columnwise(row_to_col(v, perm))


def densify_tensor_dim2(v: Array) -> Array:
    """
    Takes the 3D tensor v with np.triu(v) == v and stacks it along the second dimension.
    The matrix v[:, -1, :] will be stacked on v[:, 0, :], and so on.
    If v[1] is odd there will be a self-pairing.
    The stacking dimension leads and the first dimension of v trails, so each
    output slice out[i] flattens (in C order) to the joint simplex vector

    :param v: An n x m x m 2D Jax array
    :return: A ((m+1)//2) x (m+1) x n 3D Jax array containing the upper triangular part of v
    """
    n = v.shape[0]
    m = v.shape[1]
    new_d2 = (m + 1) // 2

    left_indices = jnp.arange(new_d2)
    right_indices = m - 1 - left_indices

    left = v[:, left_indices, ::-1]
    left = jnp.concatenate([left, jnp.zeros((n, new_d2, 1), dtype=v.dtype)], axis=2)

    right = v[:, right_indices, :]
    right = jnp.concatenate([jnp.zeros((n, new_d2, 1), dtype=v.dtype), right], axis=2)

    out = left + right

    return out.transpose(1, 2, 0)


def sparsify_tensor_dim2(v: Array) -> Array:
    """
    The inverse operation to densify_tensor_dim2

    :param v: The ((m+1)//2) x (m+1) x n dim2 dense 3D array to sparsify
    :return: An upper triangular n x m x m 3D array representing v sparsified
    """
    n = v.shape[2]
    m = v.shape[1] - 1
    dim2_col_perm = dim2_to_col_permutation(n, m)
    return vmap(sparsify_columnwise)(dim2_to_col(v, dim2_col_perm))


def dense_columnwise_mask(n: int) -> Array:
    """
    Given a matrix size n, returns a boolean mask that
    is False at the indices where a dense columnwise matrix
    does not correspond to its upper triangular representation

    :param n: The size of the sparse matrix form is n x n
    :return: A ((n+1)//2) x (n+1) 2D boolean array
    """
    new_cols = (n + 1) // 2
    out = jnp.full((n + 1, new_cols), True)

    # mask for odd dimension
    if n % 2 == 1:
        mask = (jnp.arange(n+1) < new_cols)
        masked_col = out[:, -1] * mask
        out = out.at[:, -1].set(masked_col)

    return out.T


def dense_rowwise_mask(n: int) -> Array:
    """
    Like dense_columnwise_mask but for dense rowwise matrices

    :param n: The size of the sparse matrix form is n x n
    :return: A ((n+1)//2) x (n+1) 2D boolean array
    """
    new_rows = (n + 1) // 2
    out = jnp.full((new_rows, n + 1), True)

    # mask for odd dimension
    if n % 2 == 1:
        mask = (jnp.arange(n+1) < new_rows)
        masked_row = out[-1, :] * mask
        out = out.at[-1, :].set(masked_row)

    return out


def dim2_mask(n: int, m: int) -> Array:
    """
    Like dense_columnwise_mask but for dense dim2 tensors

    :param n: The first dimension
    :param m: The second two dimensions (n x m x m)
    :return: A ((m+1)//2) x (m+1) x n 3D boolean tensor
    """
    new_dim2 = (m + 1) // 2
    out = jnp.full((n, new_dim2, m + 1), True)

    if m % 2 == 1:
        mask = (jnp.arange(m + 1) < new_dim2)
        masked_rows = out[:, -1, :] * mask
        out = out.at[:, -1, :].set(masked_rows)

    return out.transpose(1, 2, 0)


def _permutation_helper(n: int) -> tuple[Array, Array]:
    col_mask = dense_columnwise_mask(n)
    row_mask = dense_rowwise_mask(n)

    s = (n + 1) // 2 + 1 if n % 2 == 1 else 0

    v = jnp.triu(jnp.arange(s, n * n + s).reshape(n, n))

    v_col = densify_columnwise(v) * col_mask
    v_row = densify_rowwise(v) * row_mask

    v_col = (v_col - s).ravel()
    v_row = (v_row - s).ravel()

    return v_col, v_row


def col_to_row_permutation(n: int) -> Array:
    """
    Given a matrix size n, gives the permutation p such that
    densify_columnwise(v).ravel()[p] == densify_rowwise(v).ravel()
    for all n x n matrices v

    :param n: The matrix size
    :return: The 1D index array p
    """
    v_col, v_row = _permutation_helper(n)

    return v_col.argsort()[v_row.argsort().argsort()].astype(jnp.int32)


def row_to_col_permutation(n: int) -> Array:
    """
    Like col_to_row_permutation but converts a rowwise dense array
    to a columnwise dense one instead

    :param n: The matrix size
    :return: The 1D index array p
    """
    v_col, v_row = _permutation_helper(n)

    return v_row.argsort()[v_col.argsort().argsort()].astype(jnp.int32)


def _permutation_helper_3d(n: int, m: int) -> tuple[Array, Array]:
    dense_col_mask = dense_columnwise_mask(m)
    col_mask = jnp.broadcast_to(dense_col_mask, (n, dense_col_mask.shape[0], dense_col_mask.shape[1]))

    s = n * (m + 1) // 2 + 1 if m % 2 == 1 else 0

    v = jnp.triu(jnp.arange(s, n * m * m + s).reshape(n, m, m))

    v_col = vmap(densify_columnwise)(v) * col_mask
    v_dim2 = densify_tensor_dim2(v) * dim2_mask(n, m)

    v_col = (v_col - s).ravel()
    v_dim2 = (v_dim2 - s).ravel()

    return v_col, v_dim2


def dim2_to_col_permutation(n: int, m: int) -> Array:
    """
    Like col_to_row_permutation by converts a 3D dim2 dense array
    to an array of columnwise dense ones instead

    :param n: The 3D first dimension
    :param m: The 3D second two dimensions
    :return: The 1D index array p
    """
    v_col, v_dim2 = _permutation_helper_3d(n, m)

    return v_dim2.argsort()[v_col.argsort().argsort()].astype(jnp.int32)


def col_to_dim2_permutation(n: int, m: int) -> Array:
    """
    The inverse operation of dim2_to_col_permutation

    :param n: The 3D first dimension
    :param m: The 3D second two dimensions
    :return: The 1D index array p
    """
    v_col, v_dim2 = _permutation_helper_3d(n, m)

    return v_col.argsort()[v_dim2.argsort().argsort()].astype(jnp.int32)


def col_to_row(v: Array, perm: jnp.ndarray) -> Array:
    return v.ravel()[perm].reshape(v.shape)


def row_to_col(v: Array, perm: jnp.ndarray) -> Array:
    return v.ravel()[perm].reshape(v.shape)


def dim2_to_col(v: Array, perm: jnp.ndarray) -> Array:
    return v.ravel()[perm].reshape(v.shape[2], v.shape[0], v.shape[1])


def col_to_dim2(v: Array, perm: jnp.ndarray) -> Array:
    return v.ravel()[perm].reshape(v.shape[1], v.shape[2], v.shape[0])
