import jax.numpy as jnp
import numpy as np
from jax import config

config.update('jax_enable_x64', True)

from pria.adversarial import (
    HARD_BUMPS,
    HARD_BUMPS_MARGIN,
    TIE_GADGET_B,
    TIE_GADGET_B_P4,
    _tie_design,
    bump_margin_distance,
    enumerate_face_vertices,
    fractional_optimum_instance,
    gadget,
    jittered_instance,
    tie_split_instance,
    verify_lp_optimum,
)
from pria.jrp_problem import JrpProblem


def test_gadget_determinants_and_interior_solves():
    expected = {4: 3, 6: 9, 7: 15, 8: 34}
    for p, q in expected.items():
        Z = gadget(p)
        assert abs(round(np.linalg.det(Z))) == q, p
        mu = np.linalg.solve(Z, np.ones(p))
        assert (mu > 0).all() and (mu < 1).all(), p
    for Z, q in [(np.array(TIE_GADGET_B), 25), (np.array(TIE_GADGET_B_P4), 2)]:
        assert abs(round(np.linalg.det(Z))) == q
        mu = np.linalg.solve(Z, np.ones(len(Z)))
        assert (mu > 0).all() and (mu < 1).all()


def test_fractional_optimum_lp_matches_design():
    # p=4: LP optimum equals the designed value and is genuinely 1/3-fractional.
    _inst, info = fractional_optimum_instance(4)
    value, x = verify_lp_optimum(info["holding_item1"], info["K"])
    assert abs(value - info["lp_value"]) < 1e-7
    fracs = [v for v in x if 1e-6 < v < 1 - 1e-6]
    assert fracs and all(abs(v * 3 - round(v * 3)) < 1e-6 for v in fracs)


def test_tie_face_value_identity_p4():
    # Pre-bump, the whole face ties at (K+1) * T exactly; bumps raise the LP value
    # by exactly the minimum bump mass over the enumerated face vertices.
    Z_a, Z_b = gadget(4), np.array(TIE_GADGET_B_P4)
    T, K, H = _tie_design(Z_a, Z_b, ())
    tie_value, _ = verify_lp_optimum(H[1:, 1:], K)
    assert abs(tie_value - (K + 1) * T) < 1e-7

    verts = enumerate_face_vertices(Z_a, Z_b)
    assert len(verts) >= 2
    from pria.adversarial import support_pairs
    pairs = support_pairs(Z_a, Z_b)
    bumps = (pairs[0], pairs[-1])
    r = bump_margin_distance(bumps, verts, Z_a, Z_b)
    if r is not None:
        margin, dist, P = r
        assert margin > 0 and dist > 0 and P > 0
    _T2, K2, H2 = _tie_design(Z_a, Z_b, bumps)
    bumped_value, _ = verify_lp_optimum(H2[1:, 1:], K2)
    min_mass = min(sum(v[j] for (j, _t) in bumps) for v in verts)
    assert abs(bumped_value - ((K + 1) * T + float(min_mass))) < 1e-7


def test_shipped_hard_instance_values():
    _inst, info = tie_split_instance(())
    value, _ = verify_lp_optimum(info["holding_item1"], info["K"])
    assert abs(value - info["tie_value"]) < 1e-7
    _inst, info = tie_split_instance(HARD_BUMPS)
    assert info["margin"] == HARD_BUMPS_MARGIN
    value, _ = verify_lp_optimum(info["holding_item1"], info["K"])
    assert -1e-9 < value - info["tie_value"] <= 1.0 + 1e-9


def test_embedding_contract_and_solver_smoke():
    from pria.solver import Parameters, solve_jrp_problem
    inst, info = fractional_optimum_instance(4)
    # K0 > 0 is not a solver requirement any more -- a zero budget projects
    # onto {0} and solves normally.  It is still what the embedding is for:
    # the decoy item pins the joint max-term, which is what keeps item 1's
    # margins the margins the design computed.
    assert float(inst.joint_order_cost) > 0
    assert (np.asarray(inst.order_costs) > 0).all()
    assert np.asarray(inst.holding_costs).shape == (2, info["T"], info["T"])
    params = Parameters.with_auto_step(max_iter=50_000, tolerance=1e-9, iid=1,
                                       log=False, beta_necessary=0.0,
                                       beta_artificial=10.0, theta_initial=0.0,
                                       theta_target=0.0)
    x, y, z, iters, lower, upper = solve_jrp_problem(inst, params)
    assert 0 < int(iters) < 50_000
    assert not any(np.isnan(np.asarray(r)).any() for r in (x, y, z, lower, upper))


def test_jittered_instance_shapes_and_zero_g():
    base = JrpProblem.random_instance(0, 2, 6, dtype=jnp.float64)
    same = jittered_instance(base, 0.0)
    assert np.allclose(np.asarray(same.holding_costs), np.asarray(base.holding_costs))
    assert np.allclose(np.asarray(same.order_costs), np.asarray(base.order_costs))
    jit = jittered_instance(base, 1e-3)
    dH = np.asarray(jit.holding_costs) - np.asarray(base.holding_costs)
    assert dH.shape == np.asarray(base.holding_costs).shape
    assert np.abs(dH).max() <= 1e-3 + 1e-12
    for i in range(6):
        assert dH[:, i, i].max() == 0.0  # diagonal (and lower triangle) untouched
    assert 0 < float(jit.joint_order_cost) - float(base.joint_order_cost) <= 1e-3


if __name__ == "__main__":
    test_gadget_determinants_and_interior_solves()
    test_fractional_optimum_lp_matches_design()
    test_tie_face_value_identity_p4()
    test_shipped_hard_instance_values()
    test_embedding_contract_and_solver_smoke()
    test_jittered_instance_shapes_and_zero_g()
    print("test_adversarial: all tests passed")
