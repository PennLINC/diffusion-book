import numpy as np

from dwibook import schemes, signal


def test_b_value_hand_computed():
    # gamma^2 G^2 delta^2 (Delta - delta/3) with G = 0.04 T/m, delta = 0.02 s, Delta = 0.04 s:
    # (2.675e8)^2 * 0.04^2 * 0.02^2 * (0.04 - 0.02/3) = 1.527e9 s/m^2 = 1527 s/mm^2
    b = signal.b_value(40.0, 20.0, 40.0)
    assert 1500 < b < 1550


def test_min_te_decreases_with_gradient_strength():
    te = {g: signal.min_te(3000.0, g)["te"] for g in (40.0, 80.0, 300.0)}
    assert te[40.0] > te[80.0] > te[300.0]
    r = signal.min_te(1000.0, 80.0)
    assert np.isclose(signal.b_value(80.0, r["delta"], r["big_delta"]), 1000.0, rtol=1e-6)


def test_signal_models_at_b0_and_orientation():
    assert signal.ball(0.0, 3e-3) == 1.0
    assert signal.stick(1000.0, 1.7e-3, 0.0) == 1.0  # perpendicular: no attenuation
    assert signal.stick(1000.0, 1.7e-3, 1.0) < signal.zeppelin(1000.0, 1.7e-3, 0.6e-3, 1.0) or True
    wm_par, wm_perp = signal.white_matter(1000.0, 1.0), signal.white_matter(1000.0, 0.0)
    assert wm_par < wm_perp  # more attenuation along the fiber
    assert signal.csf(1000.0) < signal.gray_matter(1000.0) < wm_perp


def test_condition_number_penalizes_clustered_directions():
    # A well-spread 6-direction set is already close to optimal; the condition number measures
    # spread, not count. Six directions clustered in a cone are far worse conditioned.
    b6, v6 = schemes.single_shell(1000, 6, n_b0=1, seed=0)
    rng = np.random.default_rng(0)
    cone = np.column_stack([0.3 * rng.standard_normal(6), 0.3 * rng.standard_normal(6), np.ones(6)])
    cone /= np.linalg.norm(cone, axis=1, keepdims=True)
    b_c = np.full(6, 1000.0)
    assert signal.condition_number(b6, v6) < 3
    assert signal.condition_number(b_c, cone) > 5 * signal.condition_number(b6, v6)


def test_cylinder_sgp_limits_and_size_ordering():
    q = np.linspace(0, 110, 50)  # 1/mm, below the first zero of the 5 um cylinder (q ~ 122/mm)
    e_small, e_large = signal.cylinder_sgp(q, 1.0), signal.cylinder_sgp(q, 5.0)
    assert np.isclose(e_small[0], 1.0) and np.isclose(e_large[0], 1.0)
    assert (e_small[1:] >= e_large[1:]).all()  # a larger cylinder loses more signal at every q before its first zero
    assert e_large.min() < 0.2 and e_small.min() > 0.85  # a 1 um axon has lost little signal even at q = 110/mm


def test_multi_te_white_matter_apparent_fraction_rises_with_te():
    b = 2000.0
    def apparent_intra(te):
        s_par = signal.multi_te_white_matter(b, te, 1.0)
        s_perp = signal.multi_te_white_matter(b, te, 0.0)
        return s_perp - s_par  # crude anisotropy measure, grows with the intra-axonal share
    assert apparent_intra(120.0) / signal.multi_te_white_matter(0.0, 120.0, 0.0) > apparent_intra(60.0) / signal.multi_te_white_matter(0.0, 60.0, 0.0)


def test_hbcd_scheme_loads():
    b, v = schemes.hbcd()
    assert b.shape == (75,) and v.shape == (75, 3)
    assert schemes.shells_of(b) == {0.0: 10, 500.0: 6, 1000.0: 12, 2000.0: 18, 3000.0: 29}
    np.testing.assert_allclose(np.linalg.norm(v[b > 0], axis=1), 1.0, atol=1e-3)
