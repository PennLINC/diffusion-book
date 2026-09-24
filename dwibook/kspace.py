"""Centered Fourier transforms and sampling masks for the k-space chapters (Ch. 2-3).

Conventions: image and k-space arrays are 2-D ``(ny, nx)`` with the phase-encode axis first,
matching how TRXScan/mrsim-acq acquire lines along ``y``. Transforms are orthonormal, so
``ifft2c(fft2c(x)) == x`` and noise variance is preserved between domains.
"""

from __future__ import annotations

import numpy as np


def fft2c(img: np.ndarray) -> np.ndarray:
    """Centered, orthonormal 2-D FFT: image -> k-space."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img), norm="ortho"))


def ifft2c(ksp: np.ndarray) -> np.ndarray:
    """Centered, orthonormal 2-D inverse FFT: k-space -> image."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ksp), norm="ortho"))


def partial_fourier_mask(ny: int, nx: int, fraction: float) -> np.ndarray:
    """Boolean k-space mask keeping the first ``round(ny * fraction)`` phase-encode lines.

    Mirrors the scanner-like "contiguous" rule TRXScan uses: the acquired band is the low-ky
    half plus ``(fraction - 0.5) * ny`` lines of the high-ky side; the rest is skipped.
    """
    if not 0.5 <= fraction <= 1.0:
        raise ValueError("partial Fourier fraction must be in [0.5, 1]")
    n_keep = int(round(ny * fraction))
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:n_keep, :] = True
    return mask


def regular_undersampling_mask(ny: int, nx: int, accel: int, acs_lines: int = 0) -> np.ndarray:
    """GRAPPA-style mask: every ``accel``-th PE line, plus a fully sampled central ACS band."""
    if accel < 1:
        raise ValueError("accel must be >= 1")
    mask = np.zeros((ny, nx), dtype=bool)
    mask[::accel, :] = True
    if acs_lines > 0:
        lo = ny // 2 - acs_lines // 2
        mask[lo : lo + acs_lines, :] = True
    return mask


def random_undersampling_mask(
    ny: int, nx: int, accel: float, acs_lines: int = 0, seed: int = 0
) -> np.ndarray:
    """Compressed-sensing-style mask: random PE lines at rate ``1/accel`` with a central ACS band.

    Lines are drawn with a variable density (more likely near the center) so low frequencies
    stay well sampled, which is what makes the incoherent aliasing CS relies on.
    """
    rng = np.random.default_rng(seed)
    ky = np.arange(ny) - ny / 2
    density = np.exp(-((ky / (0.35 * ny)) ** 2))
    density *= (ny / accel) / density.sum()
    keep = rng.random(ny) < np.clip(density, 0, 1)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[keep, :] = True
    if acs_lines > 0:
        lo = ny // 2 - acs_lines // 2
        mask[lo : lo + acs_lines, :] = True
    return mask


def add_complex_noise(ksp: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add i.i.d. complex Gaussian noise with per-component standard deviation ``sigma``."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=sigma, size=ksp.shape) + 1j * rng.normal(scale=sigma, size=ksp.shape)
    return ksp + noise
