import numpy as np

from dwibook import truth


def test_truth_map_catalogue_has_27_names():
    assert len(truth.ALL_TRUTH_MAPS) == 27
    assert len(set(truth.ALL_TRUTH_MAPS)) == 27


def test_angular_error_ignores_sign():
    a = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0]])
    b = np.array([[-1.0, 0, 0], [1.0, 1.0, 0], [1.0, 0, 0]])
    err = truth.angular_error_deg(a, b)
    np.testing.assert_allclose(err[:2], [0.0, 45.0], atol=1e-9)
    assert np.isnan(err[2])


def test_summarize_error_basic():
    t = np.arange(10, dtype=float)
    f = t + 1.0
    s = truth.summarize_error(f, t, np.ones(10, bool))
    assert np.isclose(s["bias"], 1.0) and np.isclose(s["rmse"], 1.0) and np.isclose(s["r"], 1.0)
    assert s["n"] == 10
