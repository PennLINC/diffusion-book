"""Toy phantoms and simulators that run at book build time (seconds, numpy only).

These are deliberately small and readable: the point of the toy tier is that a reader can
follow every line. Anything brain-shaped or artifact-realistic comes from TRXScan instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from . import presets

_DATA = Path(__file__).with_name("data")

#: Relative proton density used for the synthetic brain images (arbitrary units).
PROTON_DENSITY = {"WM": 0.70, "GM": 0.85, "CSF": 1.00}


def shepp_logan(n: int = 128) -> np.ndarray:
    """The Shepp-Logan head phantom on an ``n x n`` grid, values in [0, 1]. Used in tests."""
    img = shepp_logan_phantom()
    return resize(img, (n, n), anti_aliasing=True).astype(np.float64)


@lru_cache(maxsize=1)
def brain_slice() -> dict[str, np.ndarray]:
    """One axial slice of the book's phantom as WM/GM/CSF tissue fractions, 128 x 128 at 2 mm.

    The slice passes through the lateral ventricles and deep gray matter. Rows run anterior
    (top) to posterior; columns follow radiological convention (image left = subject right).
    The phase-encode axis of the simulated EPI acquisitions is the row axis (anterior-posterior),
    as in the HBCD protocol. Keys: ``wm``, ``gm``, ``csf`` (fractions), ``mask`` (bool),
    ``voxel_mm``, ``provenance``.
    """
    with np.load(_DATA / "brain_slice.npz") as f:
        out = {k: f[k] for k in ("wm", "gm", "csf")}
        out["voxel_mm"] = float(f["voxel_mm"])
        out["provenance"] = str(f["provenance"])
    out["mask"] = (out["wm"] + out["gm"] + out["csf"]) > 0.5
    return out


def brain_image(te_ms: float = presets.TE_HBCD_MS, preset: str = "adult") -> np.ndarray:
    """Synthetic b=0 magnitude image of the brain slice at echo time ``te_ms``.

    Each tissue contributes its fraction times its proton density times ``exp(-TE / T2)`` with
    the T2 values of the requested TRXScan preset. No noise, no artifacts: this is the object
    the k-space chapters encode and reconstruct.
    """
    s = brain_slice()
    t2 = presets.T2_MS[preset]
    img = np.zeros_like(s["wm"])
    for tissue, key in (("WM", "wm"), ("GM", "gm"), ("CSF", "csf")):
        img += s[key] * PROTON_DENSITY[tissue] * np.exp(-te_ms / t2[tissue])
    return img.astype(np.float64)


def brain_phase(strength: float = 1.0) -> np.ndarray:
    """A smooth phase field (radians) of the kind a real reconstructed image carries.

    A linear ramp plus a low-order bump, scaled by ``strength``. It stands in for the receive
    coil phase and a mild B0 offset; it is not a physical model of either.
    """
    n = brain_slice()["wm"].shape[0]
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    return strength * (2.0 * xx + 1.2 * yy + 1.5 * xx * yy + 0.8 * np.exp(-((xx**2 + yy**2) / 0.08)))


def crossing_block(n: int = 64, angle_deg: float = 90.0) -> tuple[np.ndarray, np.ndarray]:
    """Two straight fiber bundles crossing at ``angle_deg`` inside an ``n x n`` slice.

    Returns ``(fraction_a, fraction_b)``: volume-fraction maps of the two bundles, each 1 inside
    its bundle and 0 outside, overlapping in the crossing region. Used to build single-voxel and
    single-slice signals for the modeling chapters without a tractogram.
    """
    y, x = np.mgrid[0:n, 0:n] - n / 2 + 0.5
    width = n / 6
    a = np.abs(y) < width / 2
    t = np.deg2rad(angle_deg)
    b = np.abs(-np.sin(t) * x + np.cos(t) * y) < width / 2
    return a.astype(np.float64), b.astype(np.float64)


def bloch_free_precession(
    t: np.ndarray, m0: float = 1.0, t1: float = 1000.0, t2: float = 80.0, off_res_hz: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Bloch solution after a 90-degree pulse: transverse (complex) and longitudinal.

    Times in ms. Transverse magnetization decays with T2 and precesses at ``off_res_hz``;
    longitudinal recovers toward ``m0`` with T1. The chapter uses this to draw FID, T2 decay,
    and T1 recovery curves before introducing the spin echo.
    """
    t = np.asarray(t, dtype=float)
    mxy = m0 * np.exp(-t / t2) * np.exp(2j * np.pi * off_res_hz * t / 1000.0)
    mz = m0 * (1.0 - np.exp(-t / t1))
    return mxy, mz


def offresonance_ensemble(n: int, t2prime_ms: float, seed: int = 0) -> np.ndarray:
    """Off-resonance offsets (rad/ms) for ``n`` isochromats whose dephasing decays as exp(-t/T2').

    A Lorentzian (Cauchy) distribution of frequency offsets with half-width 1/T2' produces an
    exactly exponential envelope, so 1/T2* = 1/T2 + 1/T2' holds for the ensemble average.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_cauchy(n) / t2prime_ms


def bloch_sequence(
    t_ms: np.ndarray,
    pulses: list[tuple[float, float, float]],
    t1_ms: float,
    t2_ms: float,
    offsets: np.ndarray,
    m0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-exact Bloch evolution of an isochromat ensemble in the rotating frame.

    ``pulses`` are hard RF pulses ``(time_ms, flip_deg, phase_deg)`` applied as instantaneous
    rotations; between pulses each isochromat precesses at its own offset (rad/ms) and relaxes
    with T1/T2 (the exact solution, so the step size only sets the plot resolution). Returns
    the ensemble-mean transverse magnetization ``Mxy(t)`` (complex) and longitudinal ``Mz(t)``.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    mxy = np.zeros(offsets.shape, dtype=complex)
    mz = np.full(offsets.shape, m0, dtype=float)
    events = sorted(pulses)
    out_xy = np.zeros(t_ms.shape, dtype=complex)
    out_z = np.zeros(t_ms.shape, dtype=float)
    t_now = t_ms[0]
    ev = 0

    def evolve(dt):
        nonlocal mxy, mz
        if dt <= 0:
            return
        mxy = mxy * np.exp(-dt / t2_ms) * np.exp(-1j * offsets * dt)
        mz = m0 + (mz - m0) * np.exp(-dt / t1_ms)

    def pulse(flip_deg, phase_deg):
        nonlocal mxy, mz
        a, p = np.deg2rad(flip_deg), np.deg2rad(phase_deg)
        m = mxy * np.exp(-1j * p)  # rotate so the pulse axis is x
        my, mz_new = m.imag * np.cos(a) - mz * np.sin(a), m.imag * np.sin(a) + mz * np.cos(a)
        mxy = (m.real + 1j * my) * np.exp(1j * p)
        mz = mz_new

    for i, t in enumerate(t_ms):
        while ev < len(events) and events[ev][0] <= t:
            evolve(events[ev][0] - t_now)
            t_now = events[ev][0]
            pulse(events[ev][1], events[ev][2])
            ev += 1
        evolve(t - t_now)
        t_now = t
        out_xy[i] = mxy.mean()
        out_z[i] = mz.mean()
    return out_xy, out_z


def random_walk_2d(
    n_walkers: int, n_steps: int, step: float, seed: int = 0, radius: float | None = None
) -> np.ndarray:
    """Isotropic 2-D random walk, optionally restricted inside a circle of ``radius``.

    Returns positions of shape ``(n_steps + 1, n_walkers, 2)``. Restriction is implemented as
    elastic reflection at the boundary (a step that would leave the circle is rejected and the
    walker stays put), which is enough to show hindered vs. restricted mean squared displacement.
    """
    rng = np.random.default_rng(seed)
    pos = np.zeros((n_steps + 1, n_walkers, 2))
    if radius is not None:
        r = radius * np.sqrt(rng.random(n_walkers))
        th = 2 * np.pi * rng.random(n_walkers)
        pos[0] = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    for i in range(n_steps):
        th = 2 * np.pi * rng.random(n_walkers)
        proposal = pos[i] + step * np.stack([np.cos(th), np.sin(th)], axis=1)
        if radius is not None:
            outside = np.hypot(proposal[:, 0], proposal[:, 1]) > radius
            proposal[outside] = pos[i][outside]
        pos[i + 1] = proposal
    return pos


def mean_squared_displacement(pos: np.ndarray) -> np.ndarray:
    """MSD over time from :func:`random_walk_2d` output, averaged over walkers."""
    d = pos - pos[0]
    return np.mean(np.sum(d**2, axis=-1), axis=1)
