"""House style for figures: slice mosaics, error maps, and fit-vs-truth panels.

Every phantom figure in the book uses the same slice indices and intensity windows so the
reader learns one brain. Color maps are colorblind-safe: ``gray`` for magnitude,
``twilight`` for phase, ``viridis`` for scalar maps, ``RdBu_r`` (zero-centered) for differences.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#: Categorical palette (colorblind-validated, fixed order, never cycled past 8).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#: Tissue colors are fixed for the whole book: WM blue, GM orange, CSF aqua.
TISSUE_COLORS = {"WM": PALETTE[0], "GM": PALETTE[1], "CSF": PALETTE[2]}

#: Text and grid tones (never used for data).
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e4e3df"}


def set_style() -> None:
    """Apply the book's matplotlib style: thin marks, recessive axes, the fixed palette."""
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK["secondary"],
        "axes.labelcolor": INK["primary"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"],
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 110,
        "figure.figsize": (7.0, 3.2),
        "image.cmap": "gray",
        "image.interpolation": "nearest",
    })


def show_image(ax, img: np.ndarray, title: str | None = None, kind: str = "magnitude", **kw):
    """Display a 2-D array with the book's color maps and no axes decoration."""
    if np.iscomplexobj(img):
        img = np.abs(img)
    ax.imshow(img, cmap=CMAPS[kind], **kw)
    ax.set_axis_off()
    if title:
        ax.set_title(title)


def animate(fig, update, frames, alt: str, fps: float = 2, width: int = 480, dpi: float | None = None):
    """Render a matplotlib animation as an inline looping GIF; returns an ``IPython.display.HTML``.

    ``update(frame)`` redraws the figure for one frame. The GIF is embedded as a data URI, so
    the built book needs no extra files and the animation plays in any browser without a kernel.
    ``alt`` is the text description for screen readers; ``width`` is the display width in
    pixels. A lower ``dpi`` keeps long or multi-panel animations small.
    """
    import base64
    import html
    import os
    import tempfile

    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation, PillowWriter

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    fd, path = tempfile.mkstemp(suffix=".gif", prefix="dwibook_")
    os.close(fd)
    try:
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        with open(path, "rb") as f:
            gif_b64 = base64.b64encode(f.read()).decode("ascii")
    finally:
        os.remove(path)
    plt.close(fig)
    return HTML(f'<img src="data:image/gif;base64,{gif_b64}" alt="{html.escape(alt)}" style="width: {width}px; max-width: 100%;">')


def complex_noise_cloud(
    ax_plane,
    ax_hist,
    signal: float,
    sigma: float = 1.0,
    phase: float = 0.0,
    n: int = 3000,
    seed: int = 0,
    show: tuple[str, ...] = ("magnitude",),
    rotated: float = 0.0,
    extent: tuple[float, float, float, float] | None = None,
    hist_xlim: tuple[float, float] | None = None,
    hist_ymax: float | None = None,
    bins: int = 40,
    legend: bool = True,
) -> dict[str, float]:
    """One voxel measured ``n`` times: the complex plane and the histogram of what is kept.

    Each measurement is the true complex value ``signal * exp(i phase)`` plus complex Gaussian
    noise with standard deviation ``sigma`` in each of the real and imaginary parts, so the
    samples form a round cloud around the true point. ``ax_plane`` shows the cloud, the true
    point, the origin, and for one sample the line from the origin whose length is that
    sample's magnitude, swung down onto the positive real axis. ``ax_hist`` shows histograms of
    the quantities named in ``show``: ``"magnitude"`` (the distance from the origin) and
    ``"real"`` (the real part of the cloud as drawn), with the true value and the mean of each
    marked. ``rotated`` in [0, 1] rotates every sample by ``-rotated * phase``: 0 draws the
    cloud as measured, 1 rotates it onto the real axis, which is when its real part is the
    phase-corrected, real-valued measurement. The magnitude does not depend on it.

    ``extent`` is the plane's ``(xmin, xmax, ymin, ymax)``; by default it covers the cloud
    before and after the rotation, and the origin. The histogram's horizontal range defaults
    to the plane's, so a histogram drawn under the plane lines up with its real axis. Pass
    fixed ``extent``, ``hist_xlim`` and ``hist_ymax`` to compare panels or animation frames.

    Colors are fixed so the picture reads the same wherever it appears: samples gray, true
    value black, magnitude ``PALETTE[3]``, real part ``PALETTE[0]``. Returns the means, in the
    units of ``signal``.
    """
    from matplotlib.patches import Arc

    rng = np.random.default_rng(seed)
    noise = sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    z = signal * np.exp(1j * phase) + noise
    mag = np.abs(z)
    shown = z * np.exp(-1j * phase * rotated)
    real = shown.real
    true_pt = signal * np.exp(1j * phase * (1 - rotated))
    colors = {"magnitude": PALETTE[3], "real": PALETTE[0]}

    if extent is None:
        pts = np.array([0, signal * np.exp(1j * phase), signal])
        pad = 3.6 * sigma
        extent = (pts.real.min() - pad, pts.real.max() + pad, pts.imag.min() - pad, pts.imag.max() + pad)
    ax_plane.axhline(0, color=INK["secondary"], lw=0.8, zorder=1)
    ax_plane.axvline(0, color=INK["secondary"], lw=0.8, zorder=1)
    ax_plane.scatter(shown.real, shown.imag, s=3, color="0.55", alpha=0.35, lw=0, rasterized=True, zorder=2)
    # one example sample, placed up and to the right of the true point, to show its distance
    target = (signal + sigma * (0.6 + 1.3j)) * np.exp(1j * phase * (1 - rotated))
    k = int(np.argmin(np.abs(shown - target)))
    ex, r = shown[k], mag[k]
    ang = np.degrees(np.angle(ex))
    ax_plane.plot([0, ex.real], [0, ex.imag], color=colors["magnitude"], lw=1.6, zorder=4)
    ax_plane.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=min(0, ang), theta2=max(0, ang),
                           color=colors["magnitude"], lw=1.2, ls="--", zorder=4))
    ax_plane.plot([ex.real], [ex.imag], "o", ms=5, color=colors["magnitude"], mec="white", mew=0.8, zorder=5)
    ax_plane.plot([r], [0], "|", ms=12, mew=2, color=colors["magnitude"], zorder=5)
    ax_plane.plot([0], [0], "+", ms=10, mew=1.5, color=INK["primary"], zorder=5)
    ax_plane.plot([true_pt.real], [true_pt.imag], "x", ms=9, mew=2.2, color=INK["primary"], zorder=6)
    ax_plane.set(xlim=extent[:2], ylim=extent[2:], xlabel="real part", ylabel="imaginary part")
    ax_plane.set_aspect("equal", adjustable="box")

    out = {"true": float(signal), "mean_magnitude": float(mag.mean()), "mean_real": float(real.mean())}
    values = {"magnitude": mag, "real": real}
    names = {"magnitude": "magnitude", "real": "real part"}
    hist_xlim = hist_xlim if hist_xlim is not None else extent[:2]
    edges = np.linspace(hist_xlim[0], hist_xlim[1], bins + 1)
    for name in show:
        ax_hist.hist(values[name], bins=edges, density=True, color=colors[name], alpha=0.45, lw=0)
    ymax = hist_ymax if hist_ymax is not None else ax_hist.get_ylim()[1] * 1.12
    ax_hist.axvline(signal, color=INK["primary"], lw=1.4, ls="--", zorder=5, label=f"true value: {signal / sigma:.1f} σ")
    for name in show:
        mu = values[name].mean()
        ax_hist.axvline(mu, color=colors[name], lw=2.2, label=f"mean {names[name]}: {mu / sigma:.2f} σ")
    ax_hist.set(xlim=hist_xlim, ylim=(0, ymax), xlabel="value", yticks=[])
    ax_hist.spines["left"].set_visible(False)
    if legend:
        ax_hist.legend(loc="best", fontsize=7, handlelength=1.2)
    return out


def show_kspace(ax, ksp: np.ndarray, title: str | None = None, voxel_mm: float | None = None, phase: bool = False):
    """Display k-space as log magnitude (the only way to see anything beyond the center).

    With ``voxel_mm`` the axes are labeled in spatial frequency: the outermost sample sits at
    k_max = 1 / (2 voxel) cycles per mm, with k_x horizontal and k_y (phase encode) vertical.
    With ``phase=True`` the phase is shown instead of the magnitude.
    """
    if phase:
        data, cmap = np.angle(ksp), "twilight"
    else:
        data, cmap = np.log1p(np.abs(ksp) / (np.abs(ksp).max() + 1e-12) * 1e3), "magma"
    if voxel_mm is None:
        ax.imshow(data, cmap=cmap)
        ax.set_axis_off()
    else:
        ny, nx = ksp.shape[-2:]
        kx, ky = 1 / (2 * voxel_mm), 1 / (2 * voxel_mm)
        ax.imshow(data, cmap=cmap, extent=(-kx, kx * (nx - 2) / nx, ky * (ny - 2) / ny, -ky))
        ax.set(xlabel="$k_x$ (cycles/mm)", ylabel="$k_y$ (cycles/mm)")
        ax.grid(False)
    if title:
        ax.set_title(title)


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
