"""k-space, EPI, and reconstruction helpers for Chapters 2-3 (toy tier).

Conventions: image and k-space arrays are 2-D ``(ny, nx)`` with the phase-encode axis first,
matching how TRXScan/mrsim-acq acquire lines along ``y``. Multi-coil arrays are
``(n_coils, ny, nx)``. Transforms are orthonormal, so ``ifft2c(fft2c(x)) == x`` and noise
variance is preserved between domains. Everything here is written to be read: it favors a
short, literal implementation over speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as _gamma
from scipy.special import i0e, i1e, ive

# ----------------------------------------------------------------------------- transforms


def fft2c(img: np.ndarray) -> np.ndarray:
    """Centered, orthonormal 2-D FFT over the last two axes: image -> k-space."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def ifft2c(ksp: np.ndarray) -> np.ndarray:
    """Centered, orthonormal 2-D inverse FFT over the last two axes: k-space -> image."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ksp, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


# ----------------------------------------------------------------------------- sampling masks


def partial_fourier_mask(ny: int, nx: int, fraction: float) -> np.ndarray:
    """Boolean k-space mask keeping the last ``round(ny * fraction)`` phase-encode lines.

    The scanner-like "contiguous" rule: the first ``ny * (1 - fraction)`` lines of the EPI
    train (the most negative ky) are skipped, so the ky = 0 line is reached sooner and the
    minimum TE drops. The acquired band is the symmetric center plus one full side.
    """
    if not 0.5 <= fraction <= 1.0:
        raise ValueError("partial Fourier fraction must be in [0.5, 1]")
    n_keep = int(round(ny * fraction))
    mask = np.zeros((ny, nx), dtype=bool)
    mask[ny - n_keep :, :] = True
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


# ----------------------------------------------------------------------------- EPI timing


@dataclass(frozen=True)
class EpiTrajectory:
    """Acquisition order and timing of a single-shot EPI readout."""

    lines: np.ndarray  # ky indices in the order they are acquired
    directions: np.ndarray  # +1 / -1 readout direction per line (alternating)
    times_ms: np.ndarray  # time of each line's center, ms after the first line
    echo_spacing_ms: float
    ny: int
    nx: int

    @property
    def readout_ms(self) -> float:
        """Duration from the first to the last line (what TotalReadoutTime approximates)."""
        return float(self.times_ms[-1] - self.times_ms[0])

    @property
    def time_to_center_ms(self) -> float:
        """Time from the first line to the ky = 0 line: sets the minimum TE contribution."""
        center = self.ny // 2
        idx = np.flatnonzero(self.lines == center)
        return float(self.times_ms[idx[0]]) if idx.size else float("nan")


def epi_trajectory(
    ny: int, nx: int, echo_spacing_ms: float, partial_fourier: float = 1.0, accel: int = 1
) -> EpiTrajectory:
    """Single-shot EPI: one ky line per echo spacing, alternating readout direction.

    Partial Fourier drops the last ``ny * (1 - fraction)`` lines (contiguous rule); in-plane
    acceleration keeps every ``accel``-th line. Both shorten the readout and move the ky = 0
    line earlier, which is exactly why they shorten the achievable TE.
    """
    keep = np.flatnonzero(partial_fourier_mask(ny, nx, partial_fourier)[:, 0])
    keep = keep[::accel]
    times = np.arange(keep.size) * echo_spacing_ms
    directions = np.where(np.arange(keep.size) % 2 == 0, 1, -1)
    return EpiTrajectory(keep, directions, times, echo_spacing_ms, ny, nx)


# ----------------------------------------------------------------------------- coils


def ring_coil_sensitivities(n_coils: int, ny: int, nx: int, with_phase: bool = True) -> np.ndarray:
    """Receive sensitivities of ``n_coils`` arranged on a ring, ``(n_coils, ny, nx)``.

    The magnitude is the model TRXScan/mrsim-acq use (a Gaussian falloff from each coil position
    plus a floor). A smooth coil-dependent phase is added by default so that magnitude-only and
    complex combinations differ, as they do for real coils.
    """
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    cy, cx = ny / 2, nx / 2
    r = 0.6 * max(ny, nx)
    sigma = 0.9 * max(ny, nx)
    sens = np.zeros((n_coils, ny, nx), dtype=complex)
    for c in range(n_coils):
        ang = 2 * np.pi * c / n_coils
        py, px = cy + r * np.sin(ang), cx + r * np.cos(ang)
        d2 = (x - px) ** 2 + (y - py) ** 2
        mag = np.exp(-d2 / (2 * sigma**2)) + 0.15
        phase = 0.0
        if with_phase:
            phase = 0.6 * np.pi * ((x - px) * np.cos(ang) + (y - py) * np.sin(ang)) / max(ny, nx)
        sens[c] = mag * np.exp(1j * phase)
    return sens


def sos_combine(images: np.ndarray) -> np.ndarray:
    """Root-sum-of-squares of coil images ``(n_coils, ny, nx)``: no sensitivities needed."""
    return np.sqrt(np.sum(np.abs(images) ** 2, axis=0))


def roemer_combine(images: np.ndarray, sens: np.ndarray) -> np.ndarray:
    """Sensitivity-weighted (Roemer) combination: SNR-optimal for uncorrelated equal-variance coils."""
    return np.sum(np.conj(sens) * images, axis=0) / np.maximum(np.sum(np.abs(sens) ** 2, axis=0), 1e-12)


# ----------------------------------------------------------------------------- parallel imaging


def sense_reconstruct(aliased: np.ndarray, sens: np.ndarray, accel: int, reg: float = 1e-3) -> np.ndarray:
    """Cartesian SENSE: unfold ``accel``-fold aliasing along ky with known sensitivities.

    ``aliased`` are the coil images reconstructed from the undersampled k-space (zero-filled,
    so each shows ``accel`` overlapping copies). For each pixel of the reduced FOV the
    ``accel`` overlapping true pixels are solved by least squares.
    """
    nc, ny, nx = aliased.shape
    out = np.zeros((ny, nx), dtype=complex)
    step = ny // accel
    for y in range(step):
        ys = [(y + k * step) % ny for k in range(accel)]
        for x in range(nx):
            s = sens[:, ys, x]  # (nc, accel)
            a = aliased[:, y, x]  # (nc,)
            sol = np.linalg.solve(s.conj().T @ s + reg * np.eye(accel), s.conj().T @ a)
            out[ys, x] = sol
    return out


def grappa_reconstruct(
    ksp: np.ndarray, mask: np.ndarray, accel: int, acs: tuple[int, int], kx_half: int = 1, reg: float = 1e-4
) -> np.ndarray:
    """GRAPPA for a regularly undersampled ``(n_coils, ny, nx)`` k-space with a central ACS band.

    For each missing-line offset ``d`` (1..R-1) a linear kernel predicts the target line of every
    coil from four acquired neighboring lines (two below, two above) and ``2*kx_half+1``
    readout points of every coil. The weights are fitted by least squares inside the ACS band
    ``acs = (ky_lo, ky_hi)`` (exclusive upper bound), then applied to every missing line whose
    four source lines were acquired.
    """
    nc, ny, nx = ksp.shape
    out = ksp.copy()
    acquired = mask[:, 0]
    dx = np.arange(-kx_half, kx_half + 1)
    lo, hi = acs
    for d in range(1, accel):
        src_off = np.array([-d - accel, -d, accel - d, 2 * accel - d])
        # --- calibrate
        S, T = [], []
        for ky in range(lo, hi):
            rows = ky + src_off
            if rows.min() < lo or rows.max() >= hi:
                continue
            for kx in range(kx_half, nx - kx_half):
                S.append(ksp[:, rows][:, :, kx + dx].ravel())
                T.append(ksp[:, ky, kx])
        S, T = np.array(S), np.array(T)
        w = np.linalg.solve(S.conj().T @ S + reg * np.trace(S.conj().T @ S).real / S.shape[1] * np.eye(S.shape[1]), S.conj().T @ T)
        # --- synthesize
        for ky in range(ny):
            if acquired[ky]:
                continue
            below = ky - ((ky - np.flatnonzero(acquired)[0]) % accel)  # nearest acquired line below
            if ky - below != d:
                continue
            rows = ky + src_off
            if rows.min() < 0 or rows.max() >= ny or not acquired[rows].all():
                continue
            for kx in range(kx_half, nx - kx_half):
                out[:, ky, kx] = ksp[:, rows][:, :, kx + dx].ravel() @ w
    return out


# ----------------------------------------------------------------------------- partial Fourier


def _lowres_phase(ksp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Phase of the image from the symmetric, fully sampled central band of a PF acquisition."""
    ny = ksp.shape[-2]
    acquired = np.flatnonzero(mask[:, 0])
    n_asym = ny - acquired.size
    band = np.zeros_like(mask)
    band[n_asym : ny - n_asym, :] = True
    win = np.hanning(ny - 2 * n_asym)[:, None] * np.ones((1, ksp.shape[-1]))
    k = np.zeros_like(ksp)
    k[band] = ksp[band]
    k[n_asym : ny - n_asym, :] *= win
    return np.angle(ifft2c(k))


def zero_fill(ksp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Partial Fourier by zero filling: the simplest reconstruction, blurred along PE."""
    return ifft2c(np.where(mask, ksp, 0))


def homodyne(ksp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Homodyne PF reconstruction: pre-weight k-space (0/1/2) and correct with the low-res phase."""
    ny = ksp.shape[-2]
    acquired = np.flatnonzero(mask[:, 0])
    n_asym = ny - acquired.size
    w = np.zeros(ny)
    w[ny - n_asym :] = 2.0  # acquired lines whose conjugate partners were skipped
    w[n_asym : ny - n_asym] = 1.0  # symmetric band
    phase = _lowres_phase(ksp, mask)
    img = ifft2c(np.where(mask, ksp, 0) * w[:, None])
    return np.real(img * np.exp(-1j * phase))


def pocs(ksp: np.ndarray, mask: np.ndarray, iters: int = 20) -> np.ndarray:
    """POCS PF reconstruction: alternate data consistency with the low-res phase constraint."""
    phase = _lowres_phase(ksp, mask)
    data = np.where(mask, ksp, 0)
    img = ifft2c(data)
    for _ in range(iters):
        est = np.abs(img) * np.exp(1j * phase)
        k = fft2c(est)
        img = ifft2c(np.where(mask, ksp, k))
    return img


# ----------------------------------------------------------------------------- compressed sensing


def haar2(x: np.ndarray, levels: int = 3) -> np.ndarray:
    """Orthonormal 2-D Haar wavelet transform, ``levels`` deep, in the standard nested layout."""
    out = np.array(x, dtype=complex)
    ny, nx = out.shape
    for _ in range(levels):
        a = out[:ny, :nx]
        lo_y = (a[0::2, :] + a[1::2, :]) / np.sqrt(2)
        hi_y = (a[0::2, :] - a[1::2, :]) / np.sqrt(2)
        ll = (lo_y[:, 0::2] + lo_y[:, 1::2]) / np.sqrt(2)
        lh = (lo_y[:, 0::2] - lo_y[:, 1::2]) / np.sqrt(2)
        hl = (hi_y[:, 0::2] + hi_y[:, 1::2]) / np.sqrt(2)
        hh = (hi_y[:, 0::2] - hi_y[:, 1::2]) / np.sqrt(2)
        ny, nx = ny // 2, nx // 2
        out[:ny, :nx], out[:ny, nx : 2 * nx] = ll, lh
        out[ny : 2 * ny, :nx], out[ny : 2 * ny, nx : 2 * nx] = hl, hh
    return out


def ihaar2(c: np.ndarray, levels: int = 3) -> np.ndarray:
    """Inverse of :func:`haar2`."""
    out = np.array(c, dtype=complex)
    ny0, nx0 = out.shape
    ny, nx = ny0 >> levels, nx0 >> levels
    for _ in range(levels):
        ll, lh = out[:ny, :nx].copy(), out[:ny, nx : 2 * nx].copy()
        hl, hh = out[ny : 2 * ny, :nx].copy(), out[ny : 2 * ny, nx : 2 * nx].copy()
        lo_y = np.empty((ny, 2 * nx), dtype=complex)
        hi_y = np.empty((ny, 2 * nx), dtype=complex)
        lo_y[:, 0::2], lo_y[:, 1::2] = (ll + lh) / np.sqrt(2), (ll - lh) / np.sqrt(2)
        hi_y[:, 0::2], hi_y[:, 1::2] = (hl + hh) / np.sqrt(2), (hl - hh) / np.sqrt(2)
        a = np.empty((2 * ny, 2 * nx), dtype=complex)
        a[0::2, :], a[1::2, :] = (lo_y + hi_y) / np.sqrt(2), (lo_y - hi_y) / np.sqrt(2)
        out[: 2 * ny, : 2 * nx] = a
        ny, nx = 2 * ny, 2 * nx
    return out


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    mag = np.abs(x)
    return np.where(mag > lam, x * (1 - lam / np.maximum(mag, 1e-30)), 0)


def cs_reconstruct(
    ksp: np.ndarray, mask: np.ndarray, lam: float = 0.01, iters: int = 100, levels: int = 3
) -> np.ndarray:
    """Compressed-sensing reconstruction: wavelet-sparse image consistent with the sampled k-space.

    Solves ``min_x 0.5 ||M F x - y||^2 + lam ||W x||_1`` by FISTA (accelerated proximal
    gradient) with an orthonormal Haar ``W`` and orthonormal ``F`` (so the step size is 1).
    ``lam`` is relative to the largest wavelet coefficient of the zero-filled image.
    """
    y = np.where(mask, ksp, 0)
    x = ifft2c(y)
    scale = np.abs(haar2(x, levels)).max()
    z, t = x.copy(), 1.0
    for _ in range(iters):
        grad = ifft2c(np.where(mask, fft2c(z) - y, 0))
        x_new = ihaar2(soft_threshold(haar2(z - grad, levels), lam * scale), levels)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


# ----------------------------------------------------------------------------- noise statistics


def rician_mean(a: np.ndarray, sigma: float) -> np.ndarray:
    """Expected magnitude of ``a + complex Gaussian noise`` (per-component SD ``sigma``).

    ``E[M] = sigma * sqrt(pi/2) * L_{1/2}(-a^2 / 2 sigma^2)``, written with the exponentially
    scaled Bessel functions so it is stable at high SNR. At ``a = 0`` it is the Rayleigh mean
    ``sigma * sqrt(pi/2)``; at high SNR it tends to ``sqrt(a^2 + sigma^2)``.
    """
    a = np.asarray(a, dtype=float)
    x = a**2 / (2 * sigma**2)
    return sigma * np.sqrt(np.pi / 2) * ((1 + x) * i0e(x / 2) + x * i1e(x / 2))


def noncentral_chi_pdf(m: np.ndarray, a: float, sigma: float, n_coils: int = 1) -> np.ndarray:
    """PDF of the root-sum-of-squares of ``n_coils`` complex Gaussian channels sharing signal ``a``.

    ``n_coils = 1`` is the Rician distribution; ``a = 0`` is the central chi (Rayleigh for one
    coil). Written with the exponentially scaled Bessel function so large arguments do not
    overflow.
    """
    m = np.asarray(m, dtype=float)
    L = n_coils
    if a == 0:
        return m ** (2 * L - 1) * np.exp(-(m**2) / (2 * sigma**2)) / (2 ** (L - 1) * sigma ** (2 * L) * _gamma(L))
    z = a * m / sigma**2
    return (m**L / (sigma**2 * a ** (L - 1))) * np.exp(-(m**2 + a**2) / (2 * sigma**2) + z) * ive(L - 1, z)
