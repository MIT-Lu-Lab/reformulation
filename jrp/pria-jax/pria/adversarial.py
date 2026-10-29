"""Constructions of adversarial JRP instances.

Three families, in increasing order of difficulty for PRIA (background, derivations,
and measured iteration counts: convergence-notes/findings.md, §14-§15 and §20):

1. ``fractional_optimum_instance`` -- integer data whose LP relaxation optimum is
   EXACTLY a fractional vertex with denominator ``q = |det Z|`` for a staircase gadget Z.
   Empirically benign (control-speed): at the optimum every reduced-cost slope is 1,
   so the fractional competitors are ~1/q away with ~1/q value gaps and sharpness is
   unharmed. Ships as the negative control: a fractional optimum per se is not hard.

2. ``tie_split_instance`` -- the hard family. Two gadgets Z_A, Z_B on shared demand
   rows; complementary slackness ties every configuration on the face
   {Z_A mu_A + Z_B mu_B = 1, mu >= 0} at value sum(rho) exactly, and integer +1
   "bumps" on support columns split the tie by exact mixed-minor fractions of
   ``[Z_A | Z_B]``. Difficulty is governed by P = dist/margin between the competing
   configurations, which margin alone does not rank; ``HARD_BUMPS`` is the bump set
   that maximises it for the shipped gadget pair.

3. ``jittered_instance`` -- continuous near-ties: add U(0,1)*g to all data, splitting
   the integer data's exact ties into near-ties of scale g. Difficulty rises sharply
   as g falls, until g drops under the solve tolerance and the near-ties stop being
   resolvable at all.

Single-item designs are embedded via ``embed_single_item``: a decoy second item
(serve-at-t optimal by a wide margin) pins the joint max-term mu0_s = 1 for every
s, making the K_0 coupling an exact constant and preserving item 1's margins.
The solver handles K_0 = 0 on its own -- a zero budget projects onto {0} --
so the embedding is not a way around that; pinning the coupling is what it is
for.

All constructors return ready-to-solve ``JrpProblem`` instances plus an info
dict. Exact face-vertex enumeration (``enumerate_face_vertices``) is pure rational
arithmetic; for p = 8 it performs C(16,8) = 12,870 exact solves (~1 minute) -- the
shipped bump sets carry precomputed margins so no enumeration is needed to build
the standard instances.
"""
from fractions import Fraction
from itertools import combinations

import jax.numpy as jnp
import numpy as np

from pria.jrp_problem import JrpProblem

# ---------------------------------------------------------------------------
# Verified gadgets: 0/1 matrices Z with the staircase-vacuous embedding in mind.
# A gadget of size p embeds at T = 2p + 1 (single) or T = 3p + 1 (tie pair), and
# q = |det Z| is the denominator of the designed fractional vertex mu = Z^{-1} 1.
# All dets and interior solves are asserted by tests/test_adversarial.py.
# ---------------------------------------------------------------------------
STAIRCASE_GADGETS = {
    # p=4: J - I, det -3, mu = 1/3 uniform.
    4: ((0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)),
    # p=6: det 9, generic interior solve (ninths).
    6: ((0, 1, 1, 1, 0, 1), (1, 0, 1, 0, 1, 0), (1, 0, 0, 1, 0, 1),
        (0, 0, 0, 0, 1, 1), (1, 1, 0, 1, 1, 0), (1, 1, 0, 0, 0, 1)),
    # p=7: det 15, generic interior solve (fifteenths).
    7: ((0, 1, 0, 0, 1, 1, 1), (1, 0, 1, 0, 0, 1, 1), (1, 0, 0, 1, 1, 1, 0),
        (0, 0, 1, 1, 1, 1, 0), (1, 1, 1, 0, 0, 0, 0), (1, 0, 1, 0, 1, 0, 1),
        (0, 0, 0, 1, 0, 0, 1)),
    # p=8: det 34, generic interior solve (34ths). Tie-split partner of TIE_GADGET_B.
    8: ((1, 1, 1, 0, 0, 0, 0, 0), (1, 1, 0, 1, 1, 0, 0, 1), (0, 1, 1, 0, 1, 1, 0, 1),
        (0, 1, 1, 0, 1, 0, 1, 1), (0, 0, 1, 1, 0, 1, 1, 1), (1, 0, 1, 1, 1, 0, 0, 0),
        (1, 0, 0, 0, 1, 1, 1, 0), (0, 1, 0, 1, 1, 1, 1, 0)),
}

# p=8 partner gadget, det -25 (coprime with 34: mixed-minor margins down to ~1/154).
TIE_GADGET_B = ((1, 0, 0, 0, 1, 1, 0, 1), (0, 1, 1, 0, 1, 0, 1, 0),
                (0, 0, 1, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 0, 0, 1),
                (1, 0, 1, 0, 1, 0, 0, 1), (1, 1, 0, 1, 1, 0, 0, 0),
                (1, 0, 0, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 0, 1, 0))

# p=4 partner (det -2, mu = 1/2 uniform): a fast test-scale tie pair with J - I.
TIE_GADGET_B_P4 = ((1, 0, 0, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 1, 0, 0))

# Bump sets for the shipped p=8 tie pair, as ((column j in [Z_A | Z_B], period t)...)
# with exact margins from full rational enumeration of the 155 face vertices.
# Measured PRIA-M iterations at tol 1e-9 in comments (exact tie: 176).
BUMP_LADDER = (
    (((1, 19), (2, 21), (2, 22), (3, 22)), Fraction(1, 10)),    # 368 iters
    (((0, 23), (7, 22), (12, 22), (15, 24)), Fraction(1, 3)),   # 3,584 iters
    (((1, 25), (6, 25), (14, 25), (15, 21)), Fraction(1, 102)), # 2,272 iters
    (((1, 18), (9, 19), (13, 18), (14, 19)), Fraction(1, 154)), # 5,104 iters
)
# argmax of P = dist/margin over 6,000 candidates; the slowest known: 5,792 iters.
HARD_BUMPS = ((2, 18), (6, 24), (7, 19), (15, 24))
HARD_BUMPS_MARGIN = Fraction(1, 145)


def gadget(p: int) -> np.ndarray:
    """The shipped staircase gadget of size p as an int array."""
    return np.array(STAIRCASE_GADGETS[p], dtype=int)


def search_gadget(p: int, det_min: int = 2, det_max: int = 10 ** 6,
                  exclude: tuple = (), tries: int = 400_000,
                  seed: int = 7) -> np.ndarray | None:
    """Random search for a 0/1 gadget: |det| in [det_min, det_max] \\ exclude and
    mu = Z^{-1} 1 strictly inside (0,1)^p. Returns the largest-|det| hit or None."""
    rng = np.random.default_rng(seed)
    best, best_d = None, 0
    for _ in range(tries):
        Z = rng.integers(0, 2, (p, p))
        d = abs(round(np.linalg.det(Z)))
        if det_min <= d <= det_max and d not in exclude and d > best_d:
            try:
                mu = np.linalg.solve(Z, np.ones(p))
            except np.linalg.LinAlgError:
                continue
            if (mu > 1e-9).all() and (mu < 1 - 1e-9).all():
                best, best_d = Z.copy(), d
    return best


# ---------------------------------------------------------------------------
# KKT inverse designs. Internally H is a 1-based (T+1, T+1) integer array with
# H[s, t] for s <= t; the public constructors slice to the 0-based (T, T) layout
# of JrpProblem (holding_costs[i][s][t]).
# ---------------------------------------------------------------------------
def _distribute(total: int, slots: int) -> list:
    base, extra = divmod(total, slots)
    return [base + (1 if j < extra else 0) for j in range(slots)]


def _single_item_design(Z: np.ndarray) -> tuple:
    """Fractional-optimum design: T = 2p+1, stars {2..p+1}, gadget rows {p+2..2p+1};
    anchor s=1 serves every pre-row demand (mu_1 = 1). Returns (T, K, H_1based)."""
    p = len(Z)
    T, K = 2 * p + 1, p + 1
    stars, rows = range(2, p + 2), range(p + 2, 2 * p + 2)
    lam = {(1, t): 1 for t in range(1, p + 2)}
    support = {(1, t) for t in range(1, p + 2)}
    for si, s in enumerate(stars):
        ones = [t for gi, t in enumerate(rows) if Z[gi, si]]
        for t, l in zip(ones, _distribute(K, len(ones)), strict=True):
            lam[(s, t)] = l
        support.update((s, t) for t in ones)
    for s in rows:
        lam[(s, s)] = K
    H = np.zeros((T + 1, T + 1), dtype=int)
    for t in range(1, T + 1):
        for s in range(1, t + 1):
            slack = 0 if (s, t) in support else (1 if (s in stars and t in rows) else 5)
            H[s, t] = (K + 1) - lam.get((s, t), 0) + slack
    return T, K, H


def _tie_design(Z_a: np.ndarray, Z_b: np.ndarray, bumps: tuple) -> tuple:
    """Tie-split design: T = 3p+1, star blocks A = {2..p+1}, B = {p+2..2p+1}, shared
    gadget rows {2p+2..3p+1}. Pre-bump, every point of the face
    {Z_a mu_A + Z_b mu_B = 1, mu >= 0} attains value sum(rho) = (K+1) * T exactly;
    each bump (j, t) adds +1 to the support column j of [Z_a | Z_b] at period t.
    Returns (T, K, H_1based)."""
    p = len(Z_a)
    assert len(Z_b) == p, "tie gadgets must have equal size"
    T, K = 3 * p + 1, p + 1
    stars_a, stars_b = list(range(2, p + 2)), list(range(p + 2, 2 * p + 2))
    rows = list(range(2 * p + 2, 3 * p + 2))
    all_stars = stars_a + stars_b
    lam = {(1, t): 1 for t in range(1, K + 1)}
    support = {(1, t) for t in range(1, 2 * p + 2)}
    for Z, stars in ((Z_a, stars_a), (Z_b, stars_b)):
        for si, s in enumerate(stars):
            ones = [t for gi, t in enumerate(rows) if Z[gi, si]]
            for t, l in zip(ones, _distribute(K, len(ones)), strict=True):
                lam[(s, t)] = l
            support.update((s, t) for t in ones)
    for s in rows:
        lam[(s, s)] = K
    H = np.zeros((T + 1, T + 1), dtype=int)
    for t in range(1, T + 1):
        for s in range(1, t + 1):
            slack = 0 if (s, t) in support else (1 if (s in all_stars and t in rows) else 5)
            H[s, t] = (K + 1) - lam.get((s, t), 0) + slack
    for (j, t) in bumps:
        s = all_stars[j]
        assert (s, t) in support, f"bump {(j, t)} is not a support column"
        H[s, t] += 1
    return T, K, H


# ---------------------------------------------------------------------------
# Embedding and public constructors.
# ---------------------------------------------------------------------------
def embed_single_item(holding: np.ndarray, order_cost: float,
                      decoy_holding: float = 50.0) -> JrpProblem:
    """Embed a single-item design as item 1 of an N = 2 instance (K_0 > 0 is a solver
    input requirement). The decoy item's serve-at-t optimum (0 diagonal, large
    off-diagonal) pins the joint max-term mu0_s = 1 for all s, so the K_0 = 1 term is
    an exact constant and item 1's margins are preserved.

    ``holding``: 0-based (T, T) upper-triangular array for item 1."""
    T = holding.shape[-1]
    decoy = np.full((T, T), float(decoy_holding))
    np.fill_diagonal(decoy, 0.0)
    hold = np.stack([np.asarray(holding, dtype=float), np.triu(decoy)])
    return JrpProblem.new(jnp.array([float(order_cost), 1.0]), 1.0,
                                jnp.array(hold, dtype=jnp.float64))


def fractional_optimum_instance(p: int = 8) -> tuple:
    """Integer instance whose LP optimum is a fractional vertex of denominator
    q = |det gadget(p)|. Benign for PRIA (the negative control -- see module doc)."""
    Z = gadget(p)
    T, K, H = _single_item_design(Z)
    q = abs(round(np.linalg.det(Z)))
    mu = np.linalg.solve(Z, np.ones(p))
    value = _design_value(T, K, H, Z, mu)
    info = {"family": "fractional_optimum", "p": p, "q": q, "T": T, "K": K,
            "holding_item1": H[1:, 1:].copy(), "lp_value": value,
            "mu_gadget": mu}
    return embed_single_item(H[1:, 1:], K), info


def _design_value(T: int, K: int, H: np.ndarray, Z: np.ndarray, mu: np.ndarray) -> float:
    """Objective value of the designed vertex of the single-gadget instance."""
    p = len(Z)
    value = sum(H[1, t] for t in range(1, p + 2)) + K  # anchor mass + mu_1
    for si, s in enumerate(range(2, p + 2)):
        for gi, t in enumerate(range(p + 2, 2 * p + 2)):
            if Z[gi, si]:
                value += H[s, t] * mu[si]
        value += K * mu[si]
    return float(value)


def tie_split_instance(bumps: tuple = HARD_BUMPS, Z_a: np.ndarray | None = None,
                       Z_b: np.ndarray | None = None) -> tuple:
    """The hard family. Defaults to the shipped p = 8 pair (dets 34, 25) and
    ``HARD_BUMPS`` (exact margin 1/145, the slowest known member: 33x the exact
    tie). ``bumps=()`` gives the exact-tie instance (provably harmless). Margins for
    non-shipped bump sets can be computed with ``enumerate_face_vertices`` +
    ``bump_margin_distance``."""
    Z_a = gadget(8) if Z_a is None else np.asarray(Z_a)
    Z_b = np.array(TIE_GADGET_B, dtype=int) if Z_b is None else np.asarray(Z_b)
    T, K, H = _tie_design(Z_a, Z_b, bumps)
    known = dict(BUMP_LADDER)
    margin = (Fraction(0) if bumps == () else
              HARD_BUMPS_MARGIN if bumps == HARD_BUMPS else known.get(bumps))
    info = {"family": "tie_split", "T": T, "K": K, "bumps": bumps,
            "q_a": abs(round(np.linalg.det(Z_a))), "q_b": abs(round(np.linalg.det(Z_b))),
            "holding_item1": H[1:, 1:].copy(), "tie_value": (K + 1) * T,
            "margin": margin}
    return embed_single_item(H[1:, 1:], K), info


def jittered_instance(base: JrpProblem, g: float, seed: int = 12345) -> JrpProblem:
    """Split the exact ties of ``base`` into near-ties of scale g by adding U(0,1)*g
    to all holding costs (strict upper triangle), order costs, and the joint cost.
    Measured: slowdown ~ 1/g for g between the tolerance and O(1); g below the
    solver tolerance (and g = 0) is provably harmless."""
    T = base.t()
    rng = np.random.RandomState(seed)
    mask = np.triu(np.ones((T, T)), k=1)
    jH = jnp.asarray(rng.uniform(0, 1, np.asarray(base.holding_costs).shape)) * mask[None]
    jK = jnp.asarray(rng.uniform(0, 1, np.asarray(base.order_costs).shape))
    jK0 = float(rng.uniform(0, 1))
    return JrpProblem.new(base.order_costs + g * jK,
                                float(base.joint_order_cost) + g * jK0,
                                base.holding_costs + g * jH)


# ---------------------------------------------------------------------------
# Exact analysis of the tie face (pure rational arithmetic).
# ---------------------------------------------------------------------------
def _frac_solve(cols: tuple, ZAB: np.ndarray) -> list | None:
    n = len(cols)
    A = [[Fraction(int(ZAB[i, j])) for j in cols] + [Fraction(1)] for i in range(n)]
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] != 0), None)
        if piv is None:
            return None
        A[c], A[piv] = A[piv], A[c]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c], strict=True)]
    return [A[i][n] / A[i][i] for i in range(n)]


def enumerate_face_vertices(Z_a: np.ndarray, Z_b: np.ndarray) -> list:
    """All vertices of {Z_a mu_A + Z_b mu_B = 1, mu >= 0} as exact Fraction tuples
    (length 2p). O(C(2p, p)) exact solves: instant at p = 4, ~1 minute at p = 8."""
    ZAB = np.hstack([np.asarray(Z_a), np.asarray(Z_b)])
    p = ZAB.shape[0]
    verts = set()
    for basis in combinations(range(2 * p), p):
        sol = _frac_solve(basis, ZAB)
        if sol is None or any(v < 0 for v in sol):
            continue
        mu = [Fraction(0)] * (2 * p)
        for j, b in enumerate(basis):
            mu[b] = sol[j]
        verts.add(tuple(mu))
    return sorted(verts)


def support_pairs(Z_a: np.ndarray, Z_b: np.ndarray) -> list:
    """Bumpable (column j, period t) pairs of the tie design for this gadget pair."""
    ZAB = np.hstack([np.asarray(Z_a), np.asarray(Z_b)])
    p = ZAB.shape[0]
    rows = list(range(2 * p + 2, 3 * p + 2))
    return [(j, rows[i]) for j in range(2 * p) for i in range(p) if ZAB[i, j]]


def bump_margin_distance(bumps: tuple, verts: list, Z_a: np.ndarray,
                         Z_b: np.ndarray) -> tuple | None:
    """Exact margin, x-space distance between the optimal set and the margin-achieving
    vertex, and the difficulty predictor P = dist/margin. P rank-orders measured
    difficulty (9/10 pairwise on the shipped ladder); margin alone does not."""
    ZAB = np.hstack([np.asarray(Z_a), np.asarray(Z_b)])
    wt = np.sqrt(1.0 + ZAB.sum(axis=0))
    bump_cols = [j for (j, _t) in bumps]
    vals = [sum(v[j] for j in bump_cols) for v in verts]
    order = sorted(set(vals))
    if len(order) < 2:
        return None
    margin = order[1] - order[0]
    V = np.array([[float(x) for x in v] for v in verts])
    opt = V[[i for i, x in enumerate(vals) if x == order[0]]]
    sec = V[[i for i, x in enumerate(vals) if x == order[1]]]
    dist = min(float(np.sqrt((((s - o) * wt) ** 2).sum())) for s in sec for o in opt)
    return margin, dist, dist / float(margin)


def find_hard_bumps(Z_a: np.ndarray, Z_b: np.ndarray, verts: list | None = None,
                    tries: int = 6000, max_bumps: int = 4, seed: int = 5) -> tuple:
    """Search bump sets maximizing P = dist/margin. Returns (bumps, margin, dist, P)."""
    verts = enumerate_face_vertices(Z_a, Z_b) if verts is None else verts
    pairs = support_pairs(Z_a, Z_b)
    rng = np.random.default_rng(seed)
    best = (None, None, None, -1.0)
    for _ in range(tries):
        k = int(rng.integers(1, max_bumps + 1))
        bset = tuple(sorted(pairs[i] for i in rng.choice(len(pairs), k, replace=False)))
        r = bump_margin_distance(bset, verts, Z_a, Z_b)
        if r is not None and r[2] > best[3]:
            best = (bset, r[0], r[1], r[2])
    return best


def verify_lp_optimum(holding: np.ndarray, order_cost: float) -> tuple:
    """Solve the single-item value LP min <H, x> + K sum_s mu_s over the lifted
    polytope with scipy/HiGHS. Returns (value, x). Requires scipy (lazy import)."""
    from scipy.optimize import linprog
    T = holding.shape[-1]
    H = np.zeros((T + 1, T + 1))
    H[1:, 1:] = holding
    pairs = [(s, t) for t in range(1, T + 1) for s in range(1, t + 1)]
    pi = {pr: k for k, pr in enumerate(pairs)}
    nx = len(pairs)
    c = np.array([H[s, t] for (s, t) in pairs] + [order_cost] * T, dtype=float)
    Aeq = np.zeros((T, nx + T))
    for t in range(1, T + 1):
        for s in range(1, t + 1):
            Aeq[t - 1, pi[(s, t)]] = 1
    Aub = np.zeros((nx, nx + T))
    for k, (s, t) in enumerate(pairs):
        Aub[k, pi[(s, t)]] = 1
        Aub[k, nx + s - 1] = -1
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(nx), A_eq=Aeq, b_eq=np.ones(T),
                  bounds=[(0, None)] * (nx + T), method="highs")
    return float(res.fun), res.x[:nx]


if __name__ == "__main__":
    from jax import config as _config
    _config.update("jax_enable_x64", True)  # library code leaves config to the caller

    inst, info = fractional_optimum_instance(8)
    value, _ = verify_lp_optimum(info["holding_item1"], info["K"])
    print(f"fractional_optimum p=8: q={info['q']}, T={info['T']}, "
          f"LP={value:.6f} (designed {info['lp_value']:.6f})")
    for bumps, label in [((), "exact tie"), (HARD_BUMPS, "HARD_BUMPS")]:
        inst, info = tie_split_instance(bumps)
        value, _ = verify_lp_optimum(info["holding_item1"], info["K"])
        print(f"tie_split {label}: T={info['T']}, tie value {info['tie_value']}, "
              f"LP={value:.6f}, margin={info['margin']}")
