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
    assert m[16:].all() and not m[:16].any()  # the early (most negative ky) lines are skipped


def test_regular_mask_has_acs_band():
    m = kspace.regular_undersampling_mask(64, 8, accel=2, acs_lines=24)
    assert m[::2].all()
    assert m[20:44].all()


def test_random_mask_rate_and_acs():
    m = kspace.random_undersampling_mask(256, 4, accel=3.0, acs_lines=16, seed=0)
    rate = m[:, 0].mean()
    assert 0.25 < rate < 0.5
    assert m[120:136].all()


def test_epi_trajectory_timing():
    tr = kspace.epi_trajectory(64, 64, echo_spacing_ms=0.5)
    assert tr.lines.size == 64 and np.isclose(tr.readout_ms, 63 * 0.5)
    assert np.isclose(tr.time_to_center_ms, 32 * 0.5)
    pf = kspace.epi_trajectory(64, 64, 0.5, partial_fourier=0.75)
    assert pf.lines.size == 48 and np.isclose(pf.time_to_center_ms, 16 * 0.5)  # center reached 16 lines sooner
    acc = kspace.epi_trajectory(64, 64, 0.5, accel=2)
    assert acc.lines.size == 32 and np.isclose(acc.time_to_center_ms, 16 * 0.5)
    assert (tr.directions[::2] == 1).all() and (tr.directions[1::2] == -1).all()


def _phantom_and_coils(n=32, nc=4):
    from dwibook.phantoms import shepp_logan

    img = shepp_logan(n).astype(complex)
    sens = kspace.ring_coil_sensitivities(nc, n, n)
    return img, sens, kspace.fft2c(sens * img)


def test_coil_combinations_recover_the_object():
    img, sens, ksp = _phantom_and_coils()
    coil_imgs = kspace.ifft2c(ksp)
    roemer = kspace.roemer_combine(coil_imgs, sens)
    np.testing.assert_allclose(np.abs(roemer), np.abs(img), atol=1e-8)
    sos = kspace.sos_combine(coil_imgs)
    np.testing.assert_allclose(sos, np.abs(img) * np.sqrt(np.sum(np.abs(sens) ** 2, 0)), atol=1e-8)


def test_sense_unfolds_noiseless_r2():
    img, sens, ksp = _phantom_and_coils(32, 4)
    mask = kspace.regular_undersampling_mask(32, 32, 2)
    aliased = kspace.ifft2c(np.where(mask, ksp, 0)) * 2  # zero-filling halves the amplitude
    rec = kspace.sense_reconstruct(aliased, sens, 2)
    assert np.max(np.abs(rec - img)) < 1e-2 * np.abs(img).max()


def test_grappa_fills_missing_lines_noiselessly():
    img, sens, ksp = _phantom_and_coils(32, 4)
    mask = kspace.regular_undersampling_mask(32, 32, 2, acs_lines=12)
    rec = kspace.grappa_reconstruct(np.where(mask, ksp, 0), mask, 2, acs=(10, 22))
    filled = np.abs(rec - ksp)[:, 3:-3, 1:-1]
    assert filled.max() < 5e-2 * np.abs(ksp).max()
    assert np.abs(kspace.roemer_combine(kspace.ifft2c(rec), sens) - img).mean() < 1e-2


def test_partial_fourier_methods_beat_zero_fill():
    from dwibook.phantoms import shepp_logan

    n = 64
    y, x = np.mgrid[0:n, 0:n]
    img = shepp_logan(n) * np.exp(1j * (0.02 * x + 0.01 * y))  # smooth object phase
    ksp = kspace.fft2c(img)
    mask = kspace.partial_fourier_mask(n, n, 0.625)
    err = lambda rec: np.abs(np.abs(rec) - np.abs(img)).mean()
    zf, hd, pc = kspace.zero_fill(ksp, mask), kspace.homodyne(ksp, mask), kspace.pocs(ksp, mask, 30)
    assert err(hd) < err(zf) and err(pc) < err(zf)


def test_haar_round_trip_and_orthonormality():
    rng = np.random.default_rng(0)
    x = rng.random((32, 32)) + 1j * rng.random((32, 32))
    c = kspace.haar2(x, 3)
    np.testing.assert_allclose(kspace.ihaar2(c, 3), x, atol=1e-12)
    assert np.isclose(np.sum(np.abs(c) ** 2), np.sum(np.abs(x) ** 2))


def test_cs_reconstruction_reduces_aliasing():
    from dwibook.phantoms import shepp_logan

    img = shepp_logan(64)
    ksp = kspace.fft2c(img)
    mask = kspace.random_undersampling_mask(64, 64, accel=2.5, acs_lines=8, seed=0)
    zf = np.abs(kspace.zero_fill(ksp, mask))
    cs = np.abs(kspace.cs_reconstruct(ksp, mask, lam=0.01, iters=100))
    assert np.abs(cs - img).mean() < 0.75 * np.abs(zf - img).mean()


def test_rician_mean_limits():
    assert np.isclose(kspace.rician_mean(0.0, 1.0), np.sqrt(np.pi / 2))
    a = 50.0
    assert np.isclose(kspace.rician_mean(a, 1.0), np.sqrt(a**2 + 1.0), rtol=1e-4)
    assert kspace.rician_mean(1.0, 1.0) > 1.0


def test_noncentral_chi_pdf_normalizes():
    m = np.linspace(0, 40, 20001)
    for a, L in [(0.0, 1), (5.0, 1), (0.0, 8), (5.0, 8)]:
        p = kspace.noncentral_chi_pdf(m, a, 1.0, L)
        assert np.isclose(np.trapezoid(p, m), 1.0, atol=2e-3)
    p = kspace.noncentral_chi_pdf(m, 5.0, 1.0, 1)
    assert np.isclose(np.trapezoid(m * p, m), kspace.rician_mean(5.0, 1.0), rtol=1e-3)
