import numpy as np

from dwibook import phantoms, schemes, synth


def _small_series():
    t = phantoms.brain_slice()
    bvals, bvecs = schemes.single_shell(1000, 12, n_b0=2)
    return t, bvals, bvecs, synth.synthetic_dwi(t, bvals, bvecs)


def test_package_data_loads():
    s1, s2, v = phantoms.brain_slice(1.0), phantoms.brain_slice(2.0), phantoms.brain_volume()
    assert s1["wm"].shape == (256, 256) and s2["wm"].shape == (128, 128)
    assert v["wm"].shape == (62, 52, 51) and v["mask"][:, :, phantoms.VOLUME_VENTRICLE_SLICE].any()
    assert 0 <= s1["csf"].min() and s1["csf"].max() <= 1.0 + 1e-6


def test_orientation_field_is_unit_and_in_plane():
    t = phantoms.brain_slice()
    o = synth.orientation_field(t["wm"])
    assert o.shape == (128, 128, 3)
    np.testing.assert_allclose(np.linalg.norm(o, axis=-1), 1.0, atol=1e-6)
    assert np.allclose(o[..., 2], 0.0)


def test_synthetic_dwi_decays_and_fits():
    t, bvals, bvecs, dwi = _small_series()
    assert dwi.shape == (128, 128, 14)
    b0 = dwi[..., bvals == 0].mean(-1)
    dw = dwi[..., bvals > 0].mean(-1)
    assert (dw[t["mask"]] < b0[t["mask"]]).all()
    maps = synth.dti_maps(dwi, bvals, bvecs, mask=t["mask"])
    wm = t["wm"] > 0.95
    assert maps["fa"][wm].mean() > 0.6 and maps["fa"][t["csf"] > 0.95].mean() < 0.15
    assert 2.5e-3 < maps["md"][t["csf"] > 0.95].mean() < 3.2e-3


def test_noise_is_rician_and_bias_positive():
    t, bvals, bvecs, dwi = _small_series()
    noisy = synth.add_noise(dwi, 0.02, seed=0)
    bg = ~t["mask"]
    assert noisy[bg].mean() > 0.02  # Rayleigh floor sigma*sqrt(pi/2) ~ 0.025
    assert np.iscomplexobj(synth.add_complex_noise(dwi, 0.02))


def test_displace_along_pe_conserves_signal_and_moves_it():
    t = phantoms.brain_slice()
    img = phantoms.brain_image()
    shift = np.full(img.shape, 3.0)
    moved = synth.displace_along_pe(img, shift)
    assert np.isclose(moved.sum(), img.sum(), rtol=0.03)  # a pure translation conserves signal
    # signal at row y appears at y + 3
    r = np.argmax(img[:, 64] > 0.1)
    assert np.argmax(moved[:, 64] > 0.1) == r + 3
    fmap = synth.synthetic_fieldmap(t["mask"], 2.0, amplitude_hz=100)
    assert fmap[t["mask"]].max() > 80 and np.isfinite(fmap).all()
    assert np.abs(np.diff(fmap, axis=0)).max() < 15  # smooth: no jump at the head boundary


def test_eddy_shift_shapes_and_b0_exempt():
    bvals, bvecs = schemes.single_shell(1000, 6, n_b0=1)
    s = synth.eddy_shift(bvals, bvecs, (128, 128), strength=0.02)
    assert s.shape == (128, 128, 7)
    assert np.allclose(s[..., 0], 0.0)
    assert np.abs(s[..., 1:]).max() > 0.5
    s3 = synth.eddy_shift(bvals, bvecs, (62, 52, 51), strength=0.02)
    assert s3.shape == (62, 52, 51, 7) and np.allclose(s3[..., 0], 0.0)
    # the slice-axis gradient component gives a shift that changes from slice to slice
    v = 1 + int(np.argmax(np.abs(bvecs[1:, 2])))
    assert np.ptp(s3[31, 26, :, v]) > 0.1


def test_motion_and_dropout_on_volume():
    v = phantoms.brain_volume()
    bvals, bvecs = schemes.single_shell(1000, 6, n_b0=1)
    series = synth.synthetic_dwi(v, bvals, bvecs)
    poses = [((0, 0, 0), (0, 0, 0))] * 6 + [((5.0, 0, 0), (0, 1.0, 0))]
    moved, bv = synth.moving_series(v, bvals, bvecs, poses)
    np.testing.assert_allclose(moved[..., 0], series[..., 0], atol=1e-5)
    assert not np.allclose(moved[..., 6], series[..., 6])
    np.testing.assert_allclose(np.linalg.norm(bv, axis=1), np.linalg.norm(bvecs, axis=1), atol=1e-9)
    assert np.allclose(bv[:6], bvecs[:6]) and not np.allclose(bv[6], bvecs[6])
    d = synth.dropout(series, 3, [10, 27, 44], 0.3)
    assert np.isclose(d[:, :, 10, 3].sum(), 0.3 * series[:, :, 10, 3].sum())
    assert np.allclose(d[:, :, 11, 3], series[:, :, 11, 3])


def test_undistort_inverts_displacement():
    t = phantoms.brain_slice()
    img = phantoms.brain_image()
    fmap = synth.synthetic_fieldmap(t["mask"], 2.0, amplitude_hz=60)
    shift = fmap * 0.0917  # voxels = Hz * total readout time (s)
    distorted = synth.displace_along_pe(img, shift)
    recovered = synth.undistort_along_pe(distorted, shift)
    # signal that piled up into a few voxels cannot be fully separated again, so the inverse
    # is not exact; at a 6-voxel maximum shift about a third of the error remains
    assert np.abs(recovered - img)[t["mask"]].mean() < 0.4 * np.abs(distorted - img)[t["mask"]].mean()


def test_gnl_unwarp_inverts_warp():
    img = phantoms.brain_image()
    warped, _ = synth.gnl_warp(img, 0.08, 2.0)
    back = synth.gnl_unwarp(warped, 0.08, 2.0)
    inner = phantoms.brain_slice()["mask"]
    assert np.abs(back - img)[inner].mean() < 0.2 * np.abs(warped - img)[inner].mean()


def test_ghost_and_spike_change_the_image():
    img = phantoms.brain_image()
    g = np.abs(synth.nyquist_ghost(img, 0.3))
    assert not np.allclose(g, img) and np.isclose(np.abs(synth.nyquist_ghost(img, 0.0)).sum(), img.sum())
    s = np.abs(synth.kspace_spike(img, 40, 70, 1.0))  # a spike as large as the k-space peak
    assert np.abs(s - img).max() > 0.05  # spreads over the image as a stripe pattern of amplitude peak / N


def test_register_pe_affine_recovers_eddy_parameters():
    t = phantoms.brain_slice()
    img = phantoms.brain_image()
    truth = (0.02, -0.015, 1.5)
    distorted = synth.displace_along_pe(img, synth.pe_affine_shift(truth, img.shape))
    params, corrected = synth.register_pe_affine(distorted, img, mask=t["mask"])
    np.testing.assert_allclose(params, truth, atol=0.01)  # shear and scale per voxel, translation in voxels
    assert np.abs(corrected - img)[t["mask"]].mean() < 0.3 * np.abs(distorted - img)[t["mask"]].mean()


def test_crossing_series_and_spherical_mean_fit():
    t = phantoms.brain_slice()
    bvals, bvecs = schemes.multi_shell({1000: 12, 2000: 12, 3000: 12}, n_b0=2)
    region = synth.crossing_region(t["mask"].shape)
    series, o1, o2 = synth.synthetic_dwi_crossing(t, bvals, bvecs, region, fraction=0.5)
    assert series.shape == (128, 128, 38)
    assert np.allclose(o2[~region], 0) and np.allclose(np.linalg.norm(o2[region], axis=-1), 1)
    single = synth.synthetic_dwi(t, bvals, bvecs)
    assert np.allclose(series[~region], single[~region])
    _, p1, p2 = synth.synthetic_dwi_crossing(t, bvals, bvecs, region, second="perpendicular")
    dots = np.abs(np.sum(p1 * p2, axis=-1))[region & (t["wm"] > 0.5)]
    assert dots.max() < 1e-6  # the second fiber is at 90 degrees to the first everywhere in the band
    shells, means = synth.spherical_mean(series, bvals)
    assert shells.tolist() == [0, 1000, 2000, 3000] and means.shape == (128, 128, 4)
    fit = synth.smt_fit(shells, means)
    wm = t["wm"] > 0.98
    assert 0.35 < np.median(fit["f"][wm]) < 0.75  # the synthetic intra-axonal fraction is 0.55
    assert 1.3e-3 < np.median(fit["d_par"][wm]) < 2.1e-3


def test_gnl_warp_identity_and_jacobian():
    img = phantoms.brain_image()
    warped, jac = synth.gnl_warp(img, 0.0, 2.0)
    np.testing.assert_allclose(warped, img, atol=1e-6)
    np.testing.assert_allclose(jac, np.broadcast_to(np.eye(2), jac.shape))
    warped, jac = synth.gnl_warp(img, 0.1, 2.0)
    assert np.linalg.det(jac)[64, 64] < np.linalg.det(jac)[10, 10]  # more stretch far from the isocenter
    assert not np.allclose(warped, img)
