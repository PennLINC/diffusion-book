"""Toy phantoms and simulators that run at book build time (seconds, numpy only).

These are deliberately small and readable: the point of the toy tier is that a reader can
follow every line. Anything brain-shaped or artifact-realistic comes from TRXScan instead.
"""

from __future__ import annotations

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


def shepp_logan(n: int = 128) -> np.ndarray:
    """The Shepp-Logan head phantom on an ``n x n`` grid, values in [0, 1]."""
    img = shepp_logan_phantom()
    return resize(img, (n, n), anti_aliasing=True).astype(np.float64)


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
