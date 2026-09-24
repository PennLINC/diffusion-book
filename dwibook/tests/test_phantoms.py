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
