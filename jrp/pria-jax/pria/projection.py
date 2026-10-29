import jax.numpy as jnp
from jax import Array, jit, vmap


def simplex_projection(v: Array, s: float = 1.0) -> Array:
    """
    Vectorized version of simplex_projection_non_vectorized.
    Algorithm due to Mathieu Blondel, Akinori Fujino, and Naonori Ueda

    :param s: The sum of the resulting array (1 for standard simplex)
    :param v: A 1D JAX array
    :returns: v projected to the s simplex
    """
    v_len = v.shape[0]
    if v_len <= 1:
        return s * jnp.ones(v_len)
    v_sorted = v.sort(stable=False, descending=True)
    sums = jnp.cumsum(v_sorted) - s

    indices = jnp.arange(1, v_len+1)
    greater = v_sorted - sums / indices > 0
    last_greater_index = len(greater) - 1 - jnp.argmax(greater[::-1])

    rho = indices[last_greater_index]
    theta = sums[last_greater_index] / rho

    # As in double_simplex_projection: the zero simplex holds only the zero
    # vector, and no entry clears the threshold test there.
    return jnp.where(jnp.any(greater), jnp.maximum(v - theta, 0.0), 0.0)


simplex_projection_rowwise = vmap(simplex_projection, in_axes=(0, 0), out_axes=0)
simplex_projection_columnwise = vmap(simplex_projection, in_axes=(1, 0), out_axes=1)


def simplex_projection_truncated(v: Array, l: int, s: float = 1.0) -> Array:
    """
    Like simplex_projection but only for the first l elements of v
    The rest of the elements may be changed unpredictably

    :param v: A 1D JAX array
    :param l: An integer with 1 <= l <= len(v)
    :param s: The sum of the resulting array
    :returns: v with the first l elements projected to the s simplex
    """
    v_len = v.shape[0]
    if v_len <= 1:
        return s * jnp.ones(v_len)
    v_arg_sort = v.argsort(stable=False, descending=True)
    original_less_l = (jnp.arange(0, v_len)[v_arg_sort] < l)

    v_sorted = v[v_arg_sort]
    sums = jnp.cumsum(v_sorted) - s

    offset = (v_len - l) * (v_sorted < 0).astype(int)

    indices = jnp.arange(1, v_len+1) - offset
    greater = (v_sorted - sums / indices > 0) & original_less_l
    last_greater_index = len(greater) - 1 - jnp.argmax(greater[::-1])

    rho = indices[last_greater_index]
    theta = sums[last_greater_index] / rho

    return jnp.maximum(v - theta, 0.0)


simplex_projection_trunc_column = vmap(simplex_projection_truncated, in_axes=(1, 0, None), out_axes=1)
simplex_projection_trunc_row = vmap(simplex_projection_truncated, in_axes=(0, 0, None), out_axes=0)


def double_simplex_projection(v: Array, l: int, s1: float = 1.0, s2: float = 1.0) -> Array:
    """
    Projects the first l elements of v to the s1-simplex
    Projects the rest of the elements to the s2-simplex

    :param v: A 1D JAX array
    :param l: An integer with 1 <= l < len(v)
    :param s1: A float representing the sum of the first partition result
    :param s2: A float representing the sum of the second partition result
    :returns: v with the first l elements projected to the s1 simplex and the
        rest of the elements independently projected to the s2 simplex
    """
    v_len = v.shape[0]
    if v_len <= 1:
        return s1 * jnp.ones(v_len)

    # sort v[:l] and v[l:]
    original_less_l = (jnp.arange(v_len) < l)
    original_greater_l = ~original_less_l
    # the offset must exceed the spread of v so the first partition sorts
    # strictly above the second; a power of two proportional to the spread
    # keeps the add/subtract round-off at the data's scale rather than absolute
    eps = jnp.finfo(v.dtype).eps
    spread = jnp.ptp(v)
    scale = jnp.maximum(spread, jnp.max(jnp.abs(v)) * (eps * 4096.0))
    offset = 2.0 * jnp.exp2(jnp.ceil(jnp.log2(jnp.maximum(scale, jnp.finfo(v.dtype).tiny))))
    v_offset = v + offset * original_less_l
    v_sorted = jnp.sort(v_offset, descending=True, stable=False) - offset * original_less_l

    original_sums = jnp.cumsum(v_sorted)
    # fix the cumulative sums of v[l:], then subtract s1, s2 respectively
    sums = (original_sums - original_greater_l * original_sums[l-1]
            - original_less_l * s1 - original_greater_l * s2)

    # correct the indices for v[l:]
    indices = jnp.arange(1, v_len + 1, dtype=v.dtype) - l * original_greater_l

    # the condition is prefix-true within each partition, so rho is the count
    greater = v_sorted * indices > sums
    rho_left = jnp.sum(greater & original_less_l)
    rho_right = jnp.sum(greater) - rho_left

    # A target of zero admits exactly one nonnegative vector, the zero one, and
    # rho counts the entries strictly above the threshold -- of which there are
    # none, so the threshold below would be 0/0.  The max keeps the indexing in
    # range; the where discards the result it produced.  This is the K0 = 0
    # corner, which a single-item instance is always in: with nothing to share,
    # a joint order cost is not a cost at all.
    empty_left, empty_right = rho_left == 0, rho_right == 0
    rho_left = jnp.maximum(rho_left, 1)
    rho_right = jnp.maximum(rho_right, 1)

    theta_left = sums[rho_left - 1] / rho_left
    theta_right = sums[l + rho_right - 1] / rho_right

    v_new = v - theta_left * original_less_l - theta_right * original_greater_l
    v_new = jnp.where(jnp.where(original_less_l, empty_left, empty_right),
                      0.0, v_new)

    return jnp.maximum(v_new, 0.0)


def double_projection_upper_triangular(m: Array, s: float = 1.0) -> Array:
    """
    Project the columnwise-dense upper triangular matrix to the simplex.
    Each stacked pair lies along the last axis (see densify_columnwise),
    so the sort runs along contiguous memory

    :param m: The ((n+1)//2) x (n+1) columnwise-dense 2D JAX array to project
    :param s: The simplices to project to
    :return: m projected
    """
    l = jnp.arange(1, m.shape[0]+1)
    return vmap(double_simplex_projection, in_axes=(0, 0, None, None), out_axes=0)(m, l, s, s)


def double_projection_upper_triangular_row(m: Array, s: float = 1.0) -> Array:
    """
    Project the rowwise upper triangular matrix to the simplex rowwise

    :param m: The upper triangular 2D JAX array to project
    :param s: The simplices to project to
    :return: m projected
    """
    l = jnp.arange(m.shape[1]-1, m.shape[1]-1-m.shape[0], -1)
    return vmap(double_simplex_projection, in_axes=(0, 0, None, None), out_axes=0)(m, l, s, s)


def double_projection_2d(m: Array, l: int, s1: float = 1.0, s2: float = 1.0) -> Array:
    """
    A 2D simplex_projection but for the first l rows of matrix m
    NOTE: This means that a[:l, :].sum() == s1, NOT the rowwise sum
    The rest of the elements are projected in the same way, so a[l:, :].sum() == s2

    :param m: A 2D JAX array
    :param l: An integer with 1 <= l < m.shape[0]
    :param s1: A float representing the first simplex to project to
    :param s2: A float representing the second simplex to project to
    :returns: m with the entries in the first l rows projected to the simplex
    """
    m_vector = m.ravel()
    projected = double_simplex_projection(m_vector, l * m.shape[1], s1, s2)
    return projected.reshape(m.shape)


def double_projection_3d_dim2(m: Array, s: float = 1.0) -> Array:
    """
    Project the dim2-dense upper triangular tensor to the simplex along
    its leading (stacking) dimension (see densify_tensor_dim2)

    :param m: The ((m+1)//2) x (m+1) x n dim2-dense 3D JAX array to project
    :param s: The simplices to project to
    :return: m projected
    """
    l = jnp.arange(m.shape[1]-1, m.shape[1]-1-m.shape[0], -1)
    return vmap(double_projection_2d, in_axes=(0, 0, None, None), out_axes=0)(m, l, s, s)


def simplex_projection_truncated_2d(m: Array, l: int, s: Array | None = None) -> Array:
    """
    A 2D simplex_projection but for the first l columns of matrix m
    NOTE: This means that a[:, :l].sum() == 1.0, NOT the columnwise sum
    The rest of the elements must be zero, and will stay zero

    :param m: A 2D JAX array
    :param l: An integer with 1 <= l <= m.shape[1]
    :param s: An array with the simplices to project to
    :returns: m with the entries in the first l columns projected to the simplex
    """
    if s is None:
        s = jnp.ones(m.shape[1])

    m_vector = m.flatten('F')
    projected = simplex_projection_truncated(m_vector, l * m.shape[0], s)
    reshaped = projected.reshape(m.shape, order='F')

    # zero out columns because this is not guaranteed by trunc
    mask = jnp.arange(reshaped.shape[-1]) < l
    return reshaped * mask


simplex_projection_trunc_2d = vmap(simplex_projection_truncated_2d, in_axes=(1, 0, None), out_axes=1)


@jit
def simplex_projection_upper_triangular(m: Array, s: float = 1.0) -> Array:
    """
    Project the upper-triangular part of m to the simplex columnwise.
    The matrix should be upper triangular.

    :param m: The upper triangular 2D JAX array to project
    :param s: The simplices to project to
    :return: m projected
    """
    l = jnp.arange(1, m.shape[1]+1)
    return jnp.triu(simplex_projection_trunc_column(m, l, s))


@jit
def simplex_projection_upper_triangular_row(m: Array, s: float = 1.0) -> Array:
    """
    Project the upper-triangular part of m to the simplex rowwise.
    The matrix should be upper triangular.

    :param m: The upper triangular 2D JAX array to project
    :param s: The simplices to project to
    :return: m projected
    """
    l = jnp.arange(m.shape[0], 0, -1)
    return jnp.triu(simplex_projection_trunc_row(m[:, ::-1], l, s)[:, ::-1])


def simplex_projection_3d_triangular(m: Array, s: Array | None = None) -> Array:
    """
    For each matrix index [:, i, :] in 3D tensor m created,
    project the last m.shape[2] - i columns according to simplex_projection_truncated_2d

    :param m: a 3d tensor
    :param s: the simplices to project to
    :return: m projected as described
    """
    if s is None:
        s = 1.0

    l = jnp.arange(m.shape[2], 0, -1)
    return simplex_projection_trunc_2d(m[:, :, ::-1], l, s)[:, :, ::-1]


def simplex_projection_qp(v: Array, s: float = 1.0, tolerance: float | None = None,
                          x0: Array | None = None) -> Array:
    """
    Performs simplex_projection but using a QP.
    This is slow and meant for testing correctness only.

    :param tolerance: The solver tolerance
    :param v: A 1D JAX array
    :param s: The simplex to project to
    :param x0: An optional warm start
    :return: v projected to the simplex
    """
    import numpy as np
    from scipy.optimize import minimize

    v = np.asarray(v)
    v_len = v.shape[0]

    def objective(x):
        return np.sum((x - v) ** 2)

    def simplex_constraint(x):
        return np.sum(x) - s

    if x0 is None:
        x0 = np.ones(v_len) / v_len

    non_negative = [(0, None) for _ in range(v_len)]
    constraints = {'type': 'eq', 'fun': simplex_constraint}

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=non_negative,
        constraints=constraints,
        tol=tolerance,
    )

    assert result.success, f"SLSQP failed: {result.message}"

    return jnp.array(result.x)
