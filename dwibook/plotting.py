"""House style for figures: slice mosaics, error maps, and fit-vs-truth panels.

Every phantom figure in the book uses the same slice indices and intensity windows so the
reader learns one brain. Colour maps are colourblind-safe: ``gray`` for magnitude,
``twilight`` for phase, ``viridis`` for scalar maps, ``RdBu_r`` (zero-centred) for differences.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

#: Default display slices (fractions of the array extent) so they survive resolution changes.
SLICE_FRACTIONS = {"axial": 0.5, "coronal": 0.5, "sagittal": 0.5}

#: Intensity windows (percentiles) per contrast.
WINDOWS = {"magnitude": (1, 99), "scalar": (1, 99), "diff": (2, 98)}

CMAPS = {"magnitude": "gray", "phase": "twilight", "scalar": "viridis", "diff": "RdBu_r"}


def take_slice(vol: np.ndarray, plane: str = "axial", frac: float | None = None) -> np.ndarray:
    """Extract a 2-D slice from a 3-D (x, y, z) array, oriented for display (anterior up)."""
    frac = SLICE_FRACTIONS[plane] if frac is None else frac
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
    idx = int(round(frac * (vol.shape[axis] - 1)))
    sl = np.take(vol, idx, axis=axis)
    return np.rot90(sl)


def _limits(img: np.ndarray, kind: str) -> tuple[float, float]:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, WINDOWS.get(kind, (1, 99)))
    if kind == "diff":
        m = max(abs(lo), abs(hi))
        return -m, m
    return float(lo), float(hi)


def mosaic(
    vols: dict[str, np.ndarray],
    plane: str = "axial",
    kind: str = "magnitude",
    frac: float | None = None,
    share_window: bool = True,
    ncols: int | None = None,
    figsize_per: float = 3.0,
):
    """Side-by-side slices of several volumes with a shared window; returns ``(fig, axes)``."""
    names = list(vols)
    ncols = ncols or len(names)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * nrows), squeeze=False)
    slices = {n: take_slice(vols[n], plane, frac) for n in names}
    if share_window:
        lo, hi = _limits(np.concatenate([s.ravel() for s in slices.values()]), kind)
    for ax, n in zip(axes.ravel(), names):
        vmin, vmax = (lo, hi) if share_window else _limits(slices[n], kind)
        im = ax.imshow(slices[n], cmap=CMAPS[kind], vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(n, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    return fig, axes


def fit_vs_truth(
    fit: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    name: str,
    plane: str = "axial",
    frac: float | None = None,
    max_points: int = 20000,
    seed: int = 0,
):
    """The book's standard 4-panel: fitted map | truth | difference | scatter with identity line."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    f2, t2, m2 = (take_slice(a, plane, frac) for a in (fit, truth, mask.astype(float)))
    lo, hi = _limits(np.concatenate([t2[m2 > 0], f2[m2 > 0]]), "scalar")
    axes[0].imshow(np.where(m2 > 0, f2, np.nan), cmap=CMAPS["scalar"], vmin=lo, vmax=hi)
    axes[0].set_title(f"{name}: fit", fontsize=9)
    axes[1].imshow(np.where(m2 > 0, t2, np.nan), cmap=CMAPS["scalar"], vmin=lo, vmax=hi)
    axes[1].set_title(f"{name}: truth", fontsize=9)
    diff = np.where(m2 > 0, f2 - t2, np.nan)
    dlo, dhi = _limits(diff, "diff")
    im = axes[2].imshow(diff, cmap=CMAPS["diff"], vmin=dlo, vmax=dhi)
    axes[2].set_title("fit − truth", fontsize=9)
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    for ax in axes[:3]:
        ax.axis("off")
    f, t = fit[mask.astype(bool)], truth[mask.astype(bool)]
    ok = np.isfinite(f) & np.isfinite(t)
    f, t = f[ok], t[ok]
    if f.size > max_points:
        sel = np.random.default_rng(seed).choice(f.size, max_points, replace=False)
        f, t = f[sel], t[sel]
    axes[3].scatter(t, f, s=2, alpha=0.3, rasterized=True)
    axes[3].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[3].set_xlabel("truth"), axes[3].set_ylabel("fit")
    axes[3].set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, axes
