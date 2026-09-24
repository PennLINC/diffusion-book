import numpy as np

from dwibook import phantoms


def test_shepp_logan_shape_and_range():
    p = phantoms.shepp_logan(64)
    assert p.shape == (64, 64)
    assert 0 <= p.min() and p.max() <= 1.0 + 1e-6


def test_crossing_block_overlaps_at_center():
    a, b = phantoms.crossing_block(64, 90)
    assert a[32, 32] == 1 and b[32, 32] == 1
    assert a[4, 32] == 0 and b[32, 4] == 0


def test_bloch_matches_analytic_decay():
    t = np.array([0.0, 80.0, 160.0])
    mxy, mz = phantoms.bloch_free_precession(t, t1=1000, t2=80)
    np.testing.assert_allclose(np.abs(mxy), np.exp(-t / 80))
    np.testing.assert_allclose(mz, 1 - np.exp(-t / 1000))


def test_spin_echo_refocuses_and_gradient_echo_does_not():
    t = np.linspace(0, 100, 2001)
    offsets = phantoms.offresonance_ensemble(3000, t2prime_ms=20.0, seed=0)
    # 90x at t=0 only: FID decays with T2* (T2 = 80, T2' = 20 -> T2* = 16 ms)
    fid, _ = phantoms.bloch_sequence(t, [(0.0, 90.0, 0.0)], 1000.0, 80.0, offsets)
    i40 = np.searchsorted(t, 40.0)
    assert abs(fid[i40]) < 0.15  # e^{-40/16} = 0.08, ensemble-limited
    # 90x then 180y at 25 ms: echo at 50 ms with amplitude e^{-50/80}
    se, mz = phantoms.bloch_sequence(t, [(0.0, 90.0, 0.0), (25.0, 180.0, 90.0)], 1000.0, 80.0, offsets)
    i50 = np.searchsorted(t, 50.0)
    np.testing.assert_allclose(abs(se[i50]), np.exp(-50 / 80), rtol=0.05)
    assert abs(se[i50]) > abs(fid[i50]) * 5
    # longitudinal recovery is untouched by the offsets
    assert 0 < mz[i50] < 1


def test_free_random_walk_msd_is_linear_in_time():
    pos = phantoms.random_walk_2d(4000, 200, step=1.0, seed=0)
    msd = phantoms.mean_squared_displacement(pos)
    # 2-D isotropic walk with unit steps: MSD(n) = n
    np.testing.assert_allclose(msd[[50, 100, 200]], [50, 100, 200], rtol=0.1)


def test_restricted_walk_msd_saturates():
    pos = phantoms.random_walk_2d(2000, 400, step=0.5, seed=0, radius=3.0)
    msd = phantoms.mean_squared_displacement(pos)
    assert np.hypot(pos[-1, :, 0], pos[-1, :, 1]).max() <= 3.0 + 1e-9
    assert msd[-1] < 2 * 3.0**2


def test_channel_walk_is_anisotropic_and_obstacles_hinder():
    ch = phantoms.random_walk_2d(2000, 300, step=0.5, seed=1, radius=2.0, geometry="channel")
    d = ch[-1] - ch[0]
    assert np.abs(ch[:, :, 0]).max() <= 2.0 + 1e-9
    assert np.var(d[:, 1]) > 5 * np.var(d[:, 0])  # free along y, restricted along x
    free = phantoms.random_walk_2d(2000, 300, step=0.5, seed=1)
    obs = phantoms.random_walk_2d(2000, 300, step=0.5, seed=1, radius=1.5, geometry="obstacles", spacing=4.0)
    msd_free = phantoms.mean_squared_displacement(free)[-1]
    msd_obs = phantoms.mean_squared_displacement(obs)[-1]
    assert 0.2 * msd_free < msd_obs < 0.9 * msd_free  # hindered: slower but not bounded
