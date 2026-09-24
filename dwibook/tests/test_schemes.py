import numpy as np

from dwibook import schemes


def test_electrostatic_directions_are_unit_and_spread():
    d = schemes.electrostatic_directions(30)
    assert d.shape == (30, 3)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-9)
    # antipodally symmetric spread: 30 points on a hemisphere reach ~15 degrees minimum
    # separation at the default iteration count; a poorly converged set sits below 10.
    cos = np.abs(d @ d.T)
    np.fill_diagonal(cos, 0)
    assert np.degrees(np.arccos(cos.max())) > 12


def test_single_shell_layout():
    b, v = schemes.single_shell(1000, 30, n_b0=3)
    assert b.shape == (33,) and v.shape == (33, 3)
    assert (b[:3] == 0).all() and (b[3:] == 1000).all()
    assert schemes.shells_of(b) == {0.0: 3, 1000.0: 30}


def test_multi_shell_interleaves_shells():
    b, v = schemes.multi_shell({1000: 4, 2000: 4}, n_b0=1)
    assert schemes.shells_of(b) == {0.0: 1, 1000.0: 4, 2000.0: 4}
    assert list(b[1:5]) == [1000, 2000, 1000, 2000]


def test_dsi_grid_point_counts():
    b, v = schemes.dsi_grid(radius=4, b_max=4000)
    assert b.shape == (257,)
    assert b[0] == 0 and np.isclose(b.max(), 4000)
    nz = b > 0
    np.testing.assert_allclose(np.linalg.norm(v[nz], axis=1), 1.0, atol=1e-9)
    assert schemes.dsi_grid(radius=5)[0].shape == (515,)


def test_cs_subset_keeps_b0():
    b, v = schemes.dsi_grid(radius=4, n_b0=3)
    idx = schemes.cs_subset(b, v, 64, seed=1)
    assert len(idx) == 64
    assert (b[idx] < 50).sum() == 3


def test_fsl_round_trip(tmp_path):
    b, v = schemes.single_shell(2000, 8, n_b0=2)
    schemes.write_fsl(tmp_path / "s", b, v)
    b2, v2 = schemes.read_fsl(tmp_path / "s")
    np.testing.assert_allclose(b, b2)
    np.testing.assert_allclose(v, v2, atol=1e-6)
