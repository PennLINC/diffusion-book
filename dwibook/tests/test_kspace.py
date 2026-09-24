import numpy as np

from dwibook import kspace


def test_fft_round_trip_is_identity():
    rng = np.random.default_rng(0)
    img = rng.random((32, 24)) + 1j * rng.random((32, 24))
    np.testing.assert_allclose(kspace.ifft2c(kspace.fft2c(img)), img, atol=1e-12)


def test_orthonormal_transform_preserves_energy():
    rng = np.random.default_rng(1)
    img = rng.random((16, 16))
    assert np.isclose(np.sum(np.abs(img) ** 2), np.sum(np.abs(kspace.fft2c(img)) ** 2))


def test_partial_fourier_keeps_expected_lines():
    m = kspace.partial_fourier_mask(64, 32, 0.75)
    assert m.shape == (64, 32)
    assert m.sum() == 48 * 32
    assert m[:48].all() and not m[48:].any()


def test_regular_mask_has_acs_band():
    m = kspace.regular_undersampling_mask(64, 8, accel=2, acs_lines=24)
    assert m[::2].all()
    assert m[20:44].all()


def test_random_mask_rate_and_acs():
    m = kspace.random_undersampling_mask(256, 4, accel=3.0, acs_lines=16, seed=0)
    rate = m[:, 0].mean()
    assert 0.25 < rate < 0.5
    assert m[120:136].all()
