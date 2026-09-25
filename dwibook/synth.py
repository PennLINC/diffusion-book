"""Synthetic diffusion series and artifact operators for Part III (toy tier).

The series are built voxel by voxel from the phantom's tissue fractions with the compartment
models of :mod:`dwibook.signal`. White matter needs a fiber orientation, which tissue
fractions do not carry; :func:`orientation_field` assigns one tangent to the local white
matter boundary, which produces plausible, spatially varying orientations (bundles running
along gyri and around the ventricles) without any tractogram. The result is not the phantom
TRXScan simulates, but it has a known answer, so every artifact chapter can measure the error
a correction leaves behind.

Array conventions: a 2-D slice is ``(ny, nx)`` and a 3-D volume ``(ny, nx, nz)`` with the
phase-encode axis first (anterior-posterior), as in :func:`dwibook.phantoms.brain_slice`.
A series appends the volume axis last: ``(..., n_volumes)``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import presets, signal
from .phantoms import PROTON_DENSITY

# ----------------------------------------------------------------------------- series


def orientation_field(wm: np.ndarray, smooth_vox: float = 2.0) -> np.ndarray:
    """Unit fiber orientation per voxel, tangent to the white matter boundary.

    The gradient of the smoothed WM fraction points across the boundary; its cross product
    with the through-plane axis is a direction along the boundary. Where the gradient is
    zero (deep, uniform WM) the field falls back to the anterior-posterior axis. Works for 2-D
    (``(ny, nx)``) and 3-D (``(ny, nx, nz)``) inputs; returns ``(..., 3)`` in the order
    (row, column, slice).
    """
    sm = ndimage.gaussian_filter(np.asarray(wm, float), smooth_vox)
    grads = np.gradient(sm)
    if wm.ndim == 2:
        gy, gx = grads
        gz = np.zeros_like(gy)
    else:
        gy, gx, gz = grads
    n = np.stack([gy, gx, gz], axis=-1)
    z = np.zeros_like(n); z[..., 2] = 1.0
    t = np.cross(n, z)
    norm = np.linalg.norm(t, axis=-1, keepdims=True)
    fallback = np.zeros_like(t); fallback[..., 0] = 1.0
    t = np.where(norm > 1e-6, t / np.maximum(norm, 1e-12), fallback)
    return t


def synthetic_dwi(
    tissue: dict[str, np.ndarray],
    bvals: np.ndarray,
    bvecs: np.ndarray,
    te_ms: float = presets.TE_HBCD_MS,
    preset: str = "adult",
    orientation: np.ndarray | None = None,
) -> np.ndarray:
    """Noise-free diffusion series ``(..., n_volumes)`` from tissue fractions.

    Each voxel is the sum of its WM, GM and CSF fractions, each weighted by proton density
    and T2 decay at ``te_ms`` and attenuated by its compartment model at every (b, direction).
    ``bvecs`` are given in the (row, column, slice) frame of the arrays.
    """
    wm, gm, csf = (np.asarray(tissue[k], float) for k in ("wm", "gm", "csf"))
    if orientation is None:
        orientation = orientation_field(wm)
    t2 = presets.T2_MS[preset]
    s0 = {k: PROTON_DENSITY[k] * np.exp(-te_ms / t2[k]) for k in ("WM", "GM", "CSF")}
    bvals = np.asarray(bvals, float)
    bvecs = np.asarray(bvecs, float)
    cos = np.tensordot(orientation, bvecs, axes=([-1], [1]))  # (..., n_volumes)
    out = wm[..., None] * s0["WM"] * signal.white_matter(bvals, cos)
    out = out + gm[..., None] * s0["GM"] * signal.gray_matter(bvals)
    out = out + csf[..., None] * s0["CSF"] * signal.csf(bvals)
    return out


def crossing_region(shape: tuple[int, ...], center_frac: float = 0.5, half_width_frac: float = 0.12) -> np.ndarray:
    """A band of rows around ``center_frac`` of the row extent, ``half_width_frac`` wide on
    each side: the region in which :func:`synthetic_dwi_crossing` adds a second fiber."""
    rows = np.arange(shape[0])
    c = center_frac * (shape[0] - 1)
    band = np.abs(rows - c) <= half_width_frac * shape[0]
    return np.broadcast_to(band.reshape((-1,) + (1,) * (len(shape) - 1)), shape).copy()


def synthetic_dwi_crossing(
    tissue: dict[str, np.ndarray],
    bvals: np.ndarray,
    bvecs: np.ndarray,
    region: np.ndarray,
    second: tuple[float, float, float] = (0.0, 1.0, 0.0),
    fraction: float = 0.5,
    te_ms: float = presets.TE_HBCD_MS,
    preset: str = "adult",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`synthetic_dwi`, with a second fiber population inside ``region``.

    Inside the region the white matter signal is ``(1 - fraction)`` of the boundary-tangent
    fiber plus ``fraction`` of a second fiber: along the fixed direction ``second`` (row,
    column, slice), or, when ``second`` is a number, at that angle in degrees to the first
    fiber within the slice plane in every voxel (``"perpendicular"`` is 90). Returns the
    series and the two orientation fields (the second is zero outside the region), the answer
    key for Chapter 16's peak comparison.
    """
    wm = np.asarray(tissue["wm"], float)
    o1 = orientation_field(wm)
    o2 = np.zeros_like(o1)
    if isinstance(second, str) and second == "perpendicular":
        second = 90.0
    if isinstance(second, (int, float)):
        a = np.deg2rad(float(second))  # rotate the first fiber by this angle within the slice plane
        rot = np.stack([np.cos(a) * o1[..., 0] - np.sin(a) * o1[..., 1],
                        np.sin(a) * o1[..., 0] + np.cos(a) * o1[..., 1],
                        np.zeros_like(o1[..., 0])], axis=-1)
        o2[region] = rot[region]
    else:
        sec = np.asarray(second, float) / np.linalg.norm(second)
        o2[region] = sec
    t2 = presets.T2_MS[preset]
    s0 = {k: PROTON_DENSITY[k] * np.exp(-te_ms / t2[k]) for k in ("WM", "GM", "CSF")}
    bvals = np.asarray(bvals, float); bvecs = np.asarray(bvecs, float)
    cos1 = np.tensordot(o1, bvecs, axes=([-1], [1]))
    cos2 = np.tensordot(o2, bvecs, axes=([-1], [1]))
    f2 = np.where(region, fraction, 0.0)[..., None]
    wm_sig = (1 - f2) * signal.white_matter(bvals, cos1) + f2 * signal.white_matter(bvals, cos2)
    out = wm[..., None] * s0["WM"] * wm_sig
    out = out + np.asarray(tissue["gm"], float)[..., None] * s0["GM"] * signal.gray_matter(bvals)
    out = out + np.asarray(tissue["csf"], float)[..., None] * s0["CSF"] * signal.csf(bvals)
    return out, o1, o2


def spherical_mean(series: np.ndarray, bvals: np.ndarray, tol: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-shell mean of the signal over directions: ``(shell_bvals, means (..., n_shells))``."""
    bvals = np.asarray(bvals, float)
    keys = np.round(bvals / tol) * tol
    shells = np.unique(keys)
    means = np.stack([series[..., keys == b].mean(axis=-1) for b in shells], axis=-1)
    return shells, means


def smt_fit(shell_bvals: np.ndarray, means: np.ndarray, d_par_grid=None, f_grid=None) -> dict[str, np.ndarray]:
    """Spherical-mean fit of a two-compartment model (intra-axonal stick fraction ``f`` with
    axial diffusivity ``d``, extra-axonal tensor with the same axial and radial ``d (1 - f)``)
    by grid search per voxel. Returns ``f``, ``d_par`` and the fit residual.

    The spherical mean removes the orientation distribution, so the model has two parameters
    regardless of how many fiber populations a voxel holds. This is the idea behind the
    spherical mean technique; the parametrization here is a simplified one.
    """
    from scipy.special import erf

    d_par_grid = np.linspace(0.8e-3, 2.4e-3, 33) if d_par_grid is None else d_par_grid
    f_grid = np.linspace(0.0, 1.0, 41) if f_grid is None else f_grid
    b = np.asarray(shell_bvals, float)
    b0 = b == 0
    ratio = means[..., ~b0] / np.maximum(means[..., b0].mean(axis=-1, keepdims=True), 1e-9)
    bb = b[~b0]
    F, D = np.meshgrid(f_grid, d_par_grid, indexing="ij")  # (nf, nd)
    def stick_mean(bd):
        bd = np.maximum(bd, 1e-9)
        return np.sqrt(np.pi / (4 * bd)) * erf(np.sqrt(bd))
    def zeppelin_mean(b, dpar, dperp):
        return np.exp(-b * dperp) * stick_mean(b * (dpar - dperp))
    model = np.stack([F * stick_mean(bi * D) + (1 - F) * zeppelin_mean(bi, D, D * (1 - F)) for bi in bb], axis=-1)  # (nf, nd, nb)
    flat = ratio.reshape(-1, len(bb))
    cost = ((flat[:, None, None, :] - model[None]) ** 2).sum(-1)  # (nvox, nf, nd)
    best = cost.reshape(len(flat), -1).argmin(axis=1)
    fi, di = np.unravel_index(best, F.shape)
    shape = ratio.shape[:-1]
    return {"f": f_grid[fi].reshape(shape), "d_par": d_par_grid[di].reshape(shape),
            "residual": np.sqrt(cost.reshape(len(flat), -1).min(axis=1) / len(bb)).reshape(shape)}


def add_noise(series: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Magnitude of the series plus complex Gaussian noise (per-component SD ``sigma``)."""
    rng = np.random.default_rng(seed)
    n = rng.normal(scale=sigma, size=series.shape) + 1j * rng.normal(scale=sigma, size=series.shape)
    return np.abs(series + n)


def add_complex_noise(series: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """The series plus complex Gaussian noise, kept complex (for complex-domain denoising)."""
    rng = np.random.default_rng(seed)
    n = rng.normal(scale=sigma, size=series.shape) + 1j * rng.normal(scale=sigma, size=series.shape)
    return series + n


def dti_maps(series: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """FA, MD and the principal direction from a dipy weighted-least-squares tensor fit."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    gtab = gradient_table(np.asarray(bvals, float), bvecs=np.asarray(bvecs, float))
    data = np.asarray(series, float)
    fit = TensorModel(gtab, fit_method="WLS").fit(data, mask=mask)
    return {"fa": fit.fa, "md": fit.md, "evecs": fit.evecs, "evals": fit.evals}


# ----------------------------------------------------------------------------- artifacts


def _resample_along_axis0(img: np.ndarray, target_pos: np.ndarray, jacobian: np.ndarray | None, order: int = 3) -> np.ndarray:
    """Sample ``img`` at fractional row positions ``target_pos`` (same shape as img), times a Jacobian."""
    ny = img.shape[0]
    idx = np.indices(img.shape, dtype=float)
    idx[0] = np.clip(target_pos, 0, ny - 1)
    out = ndimage.map_coordinates(img, idx, order=order, mode="nearest")
    return out * jacobian if jacobian is not None else out


def displace_along_pe(img: np.ndarray, shift_vox: np.ndarray, jacobian: bool = True, order: int = 3, supersample: int = 4) -> np.ndarray:
    """Displace signal along the phase-encode axis (axis 0) by a per-voxel shift in voxels.

    Signal at true row ``y`` appears at row ``y + shift(y)``. The forward model is a
    push-forward: each source row (subdivided ``supersample`` times along the axis) deposits
    its signal at its displaced position with linear weights. Signal is conserved, so it
    piles up where the shift converges and thins where it diverges, as in real EPI
    distortion, and folding (compression by more than one voxel per voxel) sums the
    contributions as the scanner does. ``img`` may carry trailing axes (a series); the shift
    is broadcast over them. ``jacobian`` and ``order`` are accepted for API compatibility
    (the push-forward has no separate Jacobian step; ``order`` sets the interpolation used
    to subdivide the source rows).
    """
    img = np.asarray(img, float)
    shift = np.asarray(shift_vox, float)
    if shift.ndim < img.ndim:
        shift = np.broadcast_to(shift.reshape(shift.shape + (1,) * (img.ndim - shift.ndim)), img.shape)
    ny = img.shape[0]
    s = supersample
    # subdivide each source row into s sub-rows (interpolating image and shift), then deposit
    y_sub = (np.arange(ny * s) + 0.5) / s - 0.5
    idx = np.indices((ny * s,) + img.shape[1:], dtype=float)
    idx[0] = np.clip(y_sub.reshape((-1,) + (1,) * (img.ndim - 1)) * np.ones(idx[0].shape), 0, ny - 1)
    val = ndimage.map_coordinates(img, idx, order=order, mode="nearest") / s
    sh = ndimage.map_coordinates(shift, idx, order=1, mode="nearest")
    y_dst = idx[0] + sh
    lo = np.floor(y_dst).astype(int)
    frac = y_dst - lo
    out = np.zeros_like(img)
    for (target, weight) in ((lo, 1 - frac), (lo + 1, frac)):
        ok = (target >= 0) & (target < ny)
        t_ok = target[ok]
        rest = tuple(np.broadcast_to(r.reshape((1,) + img.shape[1:]), target.shape)[ok] for r in np.indices(img.shape[1:]))
        np.add.at(out, (t_ok,) + rest, (val * weight)[ok])
    return out


def undistort_along_pe(img: np.ndarray, shift_vox: np.ndarray, order: int = 3) -> np.ndarray:
    """Exact inverse of :func:`displace_along_pe` for a known shift field.

    The true value at row ``y`` is the distorted value at ``y + shift(y)`` times the local
    stretch ``1 + d shift / dy``. This is what a fieldmap-based correction computes once the
    field is known; estimating the field is the job of topup or a measured fieldmap.
    """
    img = np.asarray(img, float)
    shift = np.asarray(shift_vox, float)
    if shift.ndim < img.ndim:
        shift = np.broadcast_to(shift.reshape(shift.shape + (1,) * (img.ndim - shift.ndim)), img.shape)
    y = np.indices(img.shape, dtype=float)[0]
    jac = 1.0 + np.gradient(shift, axis=0)
    return _resample_along_axis0(img, y + shift, jac, order)


def pe_affine_shift(params: tuple[float, float, float], shape: tuple[int, int]) -> np.ndarray:
    """Shift field (voxels) along the phase-encode axis for ``(shear, scale, translation)``:
    ``shear * (x - cx) + scale * (y - cy) + translation``, the three eddy-current terms."""
    shear, scale, trans = params
    yy, xx = np.indices(shape, dtype=float)
    return shear * (xx - (shape[1] - 1) / 2) + scale * (yy - (shape[0] - 1) / 2) + trans


def register_pe_affine(moving: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the eddy-current transform (shear, scale, translation along the phase-encode
    axis) that maps ``moving`` onto ``target``, by least squares over the three parameters.

    Returns the parameters and the corrected image. The target is normally a *prediction* of
    the volume from the other volumes (same contrast), which is what makes a
    sum-of-squares metric appropriate; registering a diffusion-weighted volume directly to a
    b=0 image with this metric fails because their contrast differs.
    """
    from scipy.optimize import minimize

    m = np.ones(moving.shape, bool) if mask is None else np.asarray(mask, bool)

    def cost(p):
        est = undistort_along_pe(moving, pe_affine_shift(p, moving.shape), order=1)
        return np.mean((est - target)[m] ** 2)

    res = minimize(cost, np.zeros(3), method="Powell", options={"xtol": 1e-3, "ftol": 1e-9, "maxfev": 400})
    params = res.x
    return params, undistort_along_pe(moving, pe_affine_shift(params, moving.shape))


def nyquist_ghost(img: np.ndarray, phase_rad: float) -> np.ndarray:
    """Add a constant phase to every other k-space line, as a readout timing error does."""
    from .kspace import fft2c, ifft2c

    k = fft2c(np.asarray(img, complex))
    k[1::2] *= np.exp(1j * phase_rad)
    return ifft2c(k)


def kspace_spike(img: np.ndarray, ky: int, kx: int, amplitude: float) -> np.ndarray:
    """Add one spurious bright sample to k-space (an RF spike); amplitude relative to the peak."""
    from .kspace import fft2c, ifft2c

    k = fft2c(np.asarray(img, complex))
    k[ky, kx] += amplitude * np.abs(k).max()
    return ifft2c(k)


def synthetic_fieldmap(mask: np.ndarray, voxel_mm: float, amplitude_hz: float = 120.0, seed: int = 0) -> np.ndarray:
    """A plausible B0 off-resonance map in Hz: a frontal (anterior, superior-ish) susceptibility
    focus plus a smooth low-order background. Not a physical simulation."""
    mask = np.asarray(mask, bool)
    coords = np.indices(mask.shape, dtype=float)
    idx = np.nonzero(mask)
    lo = np.array([i.min() for i in idx], float)
    hi = np.array([i.max() for i in idx], float)
    ext = np.maximum(hi - lo, 1)
    u = [(c - lo[i]) / ext[i] for i, c in enumerate(coords)]  # 0..1 across the brain along each axis
    focus_y = u[0] - 0.15  # 15 % in from the anterior edge
    focus_x = u[1] - 0.5
    r2 = focus_y**2 / 0.02 + focus_x**2 / 0.05
    if mask.ndim == 3:
        r2 = r2 + (u[2] - 0.3) ** 2 / 0.05
    field = amplitude_hz * np.exp(-r2) - 0.25 * amplitude_hz * (u[0] - 0.5) + 0.1 * amplitude_hz * (u[1] - 0.5) ** 2
    return field  # smooth everywhere, including outside the head, as a real field is


def eddy_shift(bvals: np.ndarray, bvecs: np.ndarray, shape: tuple[int, ...], strength: float = 0.02) -> np.ndarray:
    """Per-volume PE-axis shift maps (voxels) from a linear eddy-current model.

    The residual eddy field is proportional to the diffusion gradient. Its component along
    each axis produces, along the phase-encode axis: a shear (proportional to the column
    coordinate) for the gradient along columns, a scale (proportional to the row coordinate)
    for the gradient along rows, and a translation for the gradient along slices.
    ``strength`` is the shift in voxels per unit of ``(b/1000) * g`` per voxel of offset.
    Returns ``(*shape, n_volumes)``.
    """
    bvals = np.asarray(bvals, float); bvecs = np.asarray(bvecs, float)
    coords = np.indices(shape, dtype=float)
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    y, x = coords[0] - cy, coords[1] - cx
    w = strength * bvals / 1000.0
    shear = w * bvecs[:, 1]
    scale = w * bvecs[:, 0]
    trans = 4.0 * w * bvecs[:, 2]
    return x[..., None] * shear + y[..., None] * scale + trans


def rigid_transform(volume: np.ndarray, rotation_deg: tuple[float, float, float], translation_vox: tuple[float, float, float]) -> np.ndarray:
    """Rotate (about the volume center, degrees about the row, column, slice axes) and translate a 3-D volume."""
    ry, rx, rz = np.deg2rad(rotation_deg)
    def rot(a, i, j):
        m = np.eye(3); m[i, i] = m[j, j] = np.cos(a); m[i, j] = -np.sin(a); m[j, i] = np.sin(a); return m
    R = rot(ry, 1, 2) @ rot(rx, 0, 2) @ rot(rz, 0, 1)
    c = (np.array(volume.shape) - 1) / 2
    offset = c - R @ (c + np.array(translation_vox))
    return ndimage.affine_transform(volume, R, offset=offset, order=1, mode="constant", cval=0.0), R


def moving_series(
    tissue: dict[str, np.ndarray],
    bvals: np.ndarray,
    bvecs: np.ndarray,
    poses: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    orientation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A series acquired while the head moves rigidly between volumes.

    For each volume the head is at ``poses[v] = (rotation_deg, translation_vox)``. The
    scanner applies the gradient ``bvecs[v]`` in its own frame, so the rotated head sees it as
    ``R.T @ g``; the volume's signal is computed with that direction and the image is then
    rotated and translated. Returns the moved series and the b-vectors in the head's frame
    (``R.T @ g`` per volume), which is what a fit must use once the images have been
    registered back: the b-vectors must be rotated with the registration
    {cite:p}`leemans2009`.
    """
    bvals = np.asarray(bvals, float); bvecs = np.asarray(bvecs, float)
    wm = np.asarray(tissue["wm"], float)
    if orientation is None:
        orientation = orientation_field(wm)
    out = np.empty(wm.shape + (len(bvals),))
    bv_head = bvecs.copy()
    for v, (rot, tr) in enumerate(poses):
        _, R = rigid_transform(np.zeros((2, 2, 2)), rot, (0, 0, 0))
        g_head = R.T @ bvecs[v]
        bv_head[v] = g_head
        vol = synthetic_dwi(tissue, bvals[v : v + 1], g_head[None, :], orientation=orientation)[..., 0]
        out[..., v], _ = rigid_transform(vol, rot, tr)
    return out, bv_head


def dropout(series: np.ndarray, volume: int, slices: list[int], attenuation: float) -> np.ndarray:
    """Attenuate the given slices of one volume (a within-volume motion event during encoding)."""
    out = series.copy()
    out[:, :, slices, volume] *= attenuation
    return out


def gnl_warp(img: np.ndarray, strength: float, voxel_mm: float, center: tuple[float, float] | None = None, order: int = 3):
    """Toy gradient nonlinearity on a 2-D image: apparent position ``r (1 + a (r/R)^2)``.

    Returns the warped image (with Jacobian intensity scaling) and the per-pixel 2 x 2
    gradient-deviation matrix ``J = d(apparent)/d(true)`` on the true grid, which is what
    a per-voxel b-matrix correction needs. ``strength`` is the fractional displacement at
    ``R = 100 mm`` from the isocenter.
    """
    ny, nx = img.shape[:2]
    cy, cx = center if center is not None else ((ny - 1) / 2, (nx - 1) / 2)
    R = 100.0 / voxel_mm
    yy, xx = np.indices((ny, nx), dtype=float)
    y, x = yy - cy, xx - cx
    r2 = (y**2 + x**2) / R**2
    # forward map phi(r) = r (1 + a r^2 / R^2); invert for the resampling by fixed-point iteration
    ys, xs = y.copy(), x.copy()
    for _ in range(10):
        f = 1 + strength * (ys**2 + xs**2) / R**2
        ys, xs = y / f, x / f
    jac = np.empty((ny, nx, 2, 2))
    f = 1 + strength * r2
    jac[..., 0, 0] = f + 2 * strength * y * y / R**2
    jac[..., 0, 1] = 2 * strength * y * x / R**2
    jac[..., 1, 0] = 2 * strength * x * y / R**2
    jac[..., 1, 1] = f + 2 * strength * x * x / R**2
    det_src = np.linalg.det(jac)
    coords = np.array([np.clip(ys + cy, 0, ny - 1), np.clip(xs + cx, 0, nx - 1)])
    def warp(a):
        w = ndimage.map_coordinates(a, coords, order=order, mode="nearest")
        return w / ndimage.map_coordinates(det_src, coords, order=1, mode="nearest")
    if img.ndim == 2:
        return warp(img), jac
    out = np.stack([warp(img[..., v]) for v in range(img.shape[-1])], axis=-1)
    return out, jac


def gnl_unwarp(warped: np.ndarray, strength: float, voxel_mm: float, center: tuple[float, float] | None = None) -> np.ndarray:
    """Inverse of :func:`gnl_warp` given the known field (what gradwarp does with the
    coefficient file): the true value at ``r`` is the warped value at ``phi(r)`` times ``det J(r)``."""
    ny, nx = warped.shape[:2]
    cy, cx = center if center is not None else ((ny - 1) / 2, (nx - 1) / 2)
    R = 100.0 / voxel_mm
    yy, xx = np.indices((ny, nx), dtype=float)
    y, x = yy - cy, xx - cx
    f = 1 + strength * (y**2 + x**2) / R**2
    _, jac = gnl_warp(np.zeros((ny, nx)), strength, voxel_mm, center)
    det = np.linalg.det(jac)
    coords = np.array([np.clip(y * f + cy, 0, ny - 1), np.clip(x * f + cx, 0, nx - 1)])
    def unwarp(a):
        return ndimage.map_coordinates(a, coords, order=3, mode="nearest") * det
    if warped.ndim == 2:
        return unwarp(warped)
    return np.stack([unwarp(warped[..., v]) for v in range(warped.shape[-1])], axis=-1)
