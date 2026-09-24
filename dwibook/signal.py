"""Diffusion encoding and single-voxel signal models (Chapters 5-7, toy tier).

Units: b in s/mm^2, diffusivities in mm^2/s, gradient amplitude in mT/m, times in ms. The
compartment models are the ones TRXScan uses (stick, cylindrically symmetric tensor, ball),
so the single-voxel curves here are the curves the phantom's voxels follow.
"""

from __future__ import annotations

import numpy as np

from . import presets

#: Gyromagnetic ratio of 1H in rad / (s T).
GAMMA = 2 * np.pi * presets.GAMMA_BAR_MHZ_PER_T * 1e6


def b_value(g_mt_per_m: float, delta_ms: float, big_delta_ms: float) -> float:
    """b (s/mm^2) of a Stejskal-Tanner pulse pair: gamma^2 G^2 delta^2 (Delta - delta/3)."""
    g = g_mt_per_m * 1e-3  # T/m
    d, D = delta_ms * 1e-3, big_delta_ms * 1e-3  # s
    return GAMMA**2 * g**2 * d**2 * (D - d / 3) * 1e-6  # s/m^2 -> s/mm^2


def q_value(g_mt_per_m: float, delta_ms: float) -> float:
    """q (1/mm) = gamma G delta / 2 pi: the spatial frequency probed by a gradient pulse."""
    return GAMMA * g_mt_per_m * 1e-3 * delta_ms * 1e-3 / (2 * np.pi) * 1e-3


def min_te(
    b: float, gmax_mt_per_m: float, t_180_ms: float = 6.0, t_pre_ms: float = 3.0, t_post_ms: float = 23.0
) -> dict[str, float]:
    """Shortest echo time that reaches ``b`` on a system with amplitude ``gmax``.

    A simplified spin-echo timeline: two rectangular diffusion pulses of duration ``delta``
    placed directly on either side of the refocusing pulse (which occupies ``t_180``), so
    ``Delta = delta + t_180``; ``t_pre`` is the time needed for excitation before the first
    pulse and ``t_post`` the time from the end of the second pulse to the center of the EPI
    readout. The echo time is then ``2 * delta + t_180 + 2 * max(t_pre, t_post)``. Ramp times
    and slew-rate limits are ignored, so real minimum echo times are somewhat longer.
    Returns a dict with ``te``, ``delta``, ``big_delta`` in ms.
    """
    lo, hi = 0.0, 500.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if b_value(gmax_mt_per_m, mid, mid + t_180_ms) < b:
            lo = mid
        else:
            hi = mid
    delta = hi
    return {"te": 2 * delta + t_180_ms + 2 * max(t_pre_ms, t_post_ms), "delta": delta, "big_delta": delta + t_180_ms}


# ----------------------------------------------------------------------------- signal models


def ball(b, d: float):
    """Isotropic Gaussian diffusion: S / S0 = exp(-b d)."""
    return np.exp(-np.asarray(b, float) * d)


def stick(b, d_par: float, cos_theta):
    """Diffusion only along one axis; ``cos_theta`` is the cosine between gradient and axis."""
    return np.exp(-np.asarray(b, float) * d_par * np.asarray(cos_theta, float) ** 2)


def zeppelin(b, d_par: float, d_perp: float, cos_theta):
    """Cylindrically symmetric tensor with axial ``d_par`` and radial ``d_perp``."""
    c2 = np.asarray(cos_theta, float) ** 2
    return np.exp(-np.asarray(b, float) * (d_perp + (d_par - d_perp) * c2))


def white_matter(b, cos_theta, preset: str = "adult"):
    """TRXScan's WM fiber compartment: an intra-axonal stick plus an extra-axonal tensor."""
    f = presets.ADULT_FRACTIONS["WM_intra"]
    d_intra = presets.ADULT_DIFFUSIVITY["WM_intra"]
    d_par, d_perp, _ = presets.ADULT_DIFFUSIVITY["WM_extra"]
    return f * stick(b, d_intra, cos_theta) + (1 - f) * zeppelin(b, d_par, d_perp, cos_theta)


def gray_matter(b):
    """TRXScan's GM compartment: a free ball plus a slowly diffusing soma ball."""
    f = presets.ADULT_FRACTIONS["GM_restricted"]
    return (1 - f) * ball(b, presets.ADULT_DIFFUSIVITY["GM"]) + f * ball(b, presets.ADULT_FRACTIONS["d_soma"])


def csf(b):
    return ball(b, presets.ADULT_DIFFUSIVITY["CSF"])


# ----------------------------------------------------------------------------- DTI design


def dti_design_matrix(bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
    """Rows ``-b [gx^2, gy^2, gz^2, 2gxgy, 2gxgz, 2gygz]`` for the log-linear tensor fit."""
    b = np.asarray(bvals, float)[:, None]
    g = np.asarray(bvecs, float)
    return -b * np.column_stack([g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2,
                                 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]])


def condition_number(bvals: np.ndarray, bvecs: np.ndarray) -> float:
    """Condition number of the DTI design matrix over the diffusion-weighted directions."""
    dw = np.asarray(bvals) > 50
    return float(np.linalg.cond(dti_design_matrix(bvals[dw], bvecs[dw])))
