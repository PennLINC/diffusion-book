"""Gradient-scheme generators and q-space plots (Ch. 6).

All schemes are returned as ``(bvals, bvecs)`` with ``bvals`` of shape ``(n,)`` in s/mm^2 and
unit ``bvecs`` of shape ``(n, 3)``; b=0 volumes carry a zero vector. Writers emit the FSL
``.bval``/``.bvec`` layout TRXScan reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dipy.core.sphere import HemiSphere, disperse_charges


def electrostatic_directions(n: int, iters: int = 5000, seed: int = 0) -> np.ndarray:
    """``n`` unit vectors spread over the hemisphere by electrostatic repulsion."""
    rng = np.random.default_rng(seed)
    theta = np.pi * rng.random(n)
    phi = 2 * np.pi * rng.random(n)
    hs = HemiSphere(theta=theta, phi=phi)
    hs, _ = disperse_charges(hs, iters)
    return hs.vertices


def single_shell(b: float, n_dirs: int, n_b0: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One shell at ``b`` with ``n_dirs`` repulsion directions and ``n_b0`` leading b=0s."""
    dirs = electrostatic_directions(n_dirs, seed=seed)
    bvals = np.concatenate([np.zeros(n_b0), np.full(n_dirs, float(b))])
    bvecs = np.concatenate([np.zeros((n_b0, 3)), dirs])
    return bvals, bvecs


def multi_shell(
    shells: dict[float, int], n_b0: int = 1, seed: int = 0, interleave: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Shells given as ``{b: n_dirs}``; optionally interleaved for motion robustness.

    Each shell gets its own repulsion set (rotated by the seed so shells do not share
    directions). Interleaving cycles through shells volume by volume, which spreads any
    time-localized motion across all shells instead of corrupting one.
    """
    per_shell = []
    for i, (b, n) in enumerate(sorted(shells.items())):
        dirs = electrostatic_directions(n, seed=seed + i)
        per_shell.append((np.full(n, float(b)), dirs))
    if interleave:
        order = []
        idx = [0] * len(per_shell)
        while any(i < len(s[0]) for i, s in zip(idx, per_shell)):
            for k, (bv, _) in enumerate(per_shell):
                if idx[k] < len(bv):
                    order.append((k, idx[k]))
                    idx[k] += 1
        bvals = np.array([per_shell[k][0][j] for k, j in order])
        bvecs = np.array([per_shell[k][1][j] for k, j in order])
    else:
        bvals = np.concatenate([s[0] for s in per_shell])
        bvecs = np.concatenate([s[1] for s in per_shell])
    bvals = np.concatenate([np.zeros(n_b0), bvals])
    bvecs = np.concatenate([np.zeros((n_b0, 3)), bvecs])
    return bvals, bvecs


def dsi_grid(radius: int = 4, b_max: float = 4000.0, n_b0: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian DSI grid: all integer lattice points with ``|q| <= radius``.

    Radius 4 gives the classic 257-point grid (``|q|^2 <= 16``); radius 5 gives 515 points.
    b scales with ``|q|^2`` so the outermost points sit at ``b_max``. The origin counts as one
    b=0; extra b=0s are prepended.
    """
    r = np.arange(-radius, radius + 1)
    qx, qy, qz = np.meshgrid(r, r, r, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1).astype(float)
    q = q[np.linalg.norm(q, axis=1) <= radius + 1e-9]
    norm = np.linalg.norm(q, axis=1)
    bvals = b_max * (norm / radius) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        bvecs = np.where(norm[:, None] > 0, q / norm[:, None], 0.0)
    order = np.argsort(norm, kind="stable")
    bvals, bvecs = bvals[order], bvecs[order]
    if n_b0 > 1:
        bvals = np.concatenate([np.zeros(n_b0 - 1), bvals])
        bvecs = np.concatenate([np.zeros((n_b0 - 1, 3)), bvecs])
    return bvals, bvecs


def cs_subset(bvals: np.ndarray, bvecs: np.ndarray, n_keep: int, seed: int = 0) -> np.ndarray:
    """Indices of a random CS-DSI subset of a full grid that always keeps every b=0 volume."""
    rng = np.random.default_rng(seed)
    b0 = np.flatnonzero(bvals < 50)
    dw = np.flatnonzero(bvals >= 50)
    if n_keep < len(b0):
        raise ValueError("n_keep is smaller than the number of b=0 volumes")
    chosen = rng.choice(dw, size=n_keep - len(b0), replace=False)
    return np.sort(np.concatenate([b0, chosen]))


def shells_of(bvals: np.ndarray, tol: float = 50.0) -> dict[float, int]:
    """Group b-values into shells (centers rounded to ``tol``) with volume counts."""
    keys = np.round(np.asarray(bvals) / tol) * tol
    uniq, counts = np.unique(keys, return_counts=True)
    return {float(u): int(c) for u, c in zip(uniq, counts)}


def write_fsl(stem: str | Path, bvals: np.ndarray, bvecs: np.ndarray) -> None:
    """Write ``<stem>.bval`` and ``<stem>.bvec`` in FSL layout (one row of b, three rows of xyz)."""
    stem = Path(stem)
    np.savetxt(stem.with_suffix(".bval"), np.asarray(bvals)[None, :], fmt="%.0f")
    np.savetxt(stem.with_suffix(".bvec"), np.asarray(bvecs).T, fmt="%.6f")


def read_fsl(stem: str | Path) -> tuple[np.ndarray, np.ndarray]:
    stem = Path(stem)
    return np.loadtxt(stem.with_suffix(".bval")), np.loadtxt(stem.with_suffix(".bvec")).T


def hbcd() -> tuple[np.ndarray, np.ndarray]:
    """The HBCD-style multi-shell scheme bundled with the phantom (75 volumes, AP polarity).

    Ten b=0 volumes and four shells (b = 500, 1000, 2000, 3000 with 6, 12, 18, 29 directions),
    interleaved in acquisition order. This is the scheme TRXScan simulates by default and the
    reference protocol of the book.
    """
    return read_fsl(Path(__file__).with_name("data") / "schemes" / "hbcd_ap")


def scan_time_s(n_volumes: int, tr_s: float) -> float:
    """Acquisition time of a scheme: one TR per volume (dummy scans and calibration excluded)."""
    return n_volumes * tr_s


def analysis_matrix(bvals: np.ndarray, complex_data: bool = False, n_dirs_min: int | None = None) -> list[tuple[str, str, str]]:
    """What a scheme supports: rows of ``(analysis, verdict, reason)`` with verdict in
    ``yes``, ``marginal``, ``no``. The rules are the ones Table 6.1 and Chapter 19 state:
    counts of shells, directions per shell, and the maximum b-value.
    """
    bvals = np.asarray(bvals, float)
    shells = {b: n for b, n in shells_of(bvals).items() if b > 0}
    n_b0 = int((bvals < 50).sum())
    b_max = max(shells) if shells else 0.0
    n_shells = len(shells)
    dirs_low = sum(n for b, n in shells.items() if b <= 1200)
    dirs_high = sum(n for b, n in shells.items() if b >= 1800)
    dirs_total = sum(shells.values())
    rows = []
    def add(name, verdict, reason): rows.append((name, verdict, reason))
    add("mean diffusivity / ADC", "yes" if dirs_total >= 3 and n_b0 >= 1 else "no", f"{dirs_total} directions, {n_b0} b=0")
    if dirs_low >= 30: add("DTI (FA, direction)", "yes", f"{dirs_low} directions at b <= 1200")
    elif dirs_low >= 6: add("DTI (FA, direction)", "marginal", f"only {dirs_low} directions at b <= 1200; precision suffers")
    elif dirs_total >= 6: add("DTI (FA, direction)", "marginal", "no low-b shell; high b breaks the Gaussian assumption")
    else: add("DTI (FA, direction)", "no", "fewer than 6 directions")
    add("diffusion kurtosis", "yes" if n_shells >= 2 and b_max >= 2000 and dirs_total >= 30 else ("marginal" if n_shells >= 2 else "no"),
        f"{n_shells} non-zero shell(s), b_max {b_max:.0f}")
    add("single-shell CSD", "yes" if dirs_high >= 45 else ("marginal" if dirs_total >= 30 else "no"), f"{dirs_high} directions at b >= 1800")
    add("multi-tissue CSD", "yes" if n_shells >= 2 and dirs_high >= 45 else ("marginal" if n_shells >= 2 else "no"), f"{n_shells} shells, {dirs_high} high-b directions")
    add("NODDI / spherical mean / free water", "yes" if n_shells >= 2 and b_max >= 2000 else ("marginal" if n_shells >= 2 else "no"), f"{n_shells} shells, b_max {b_max:.0f}")
    add("MAP-MRI / propagator", "yes" if n_shells >= 3 and dirs_total >= 60 else ("marginal" if n_shells >= 2 else "no"), f"{n_shells} shells, {dirs_total} directions")
    add("DSI (model-free propagator)", "yes" if n_shells >= 5 and dirs_total >= 200 else "no", f"{n_shells} shells, {dirs_total} directions (Cartesian grid needed)")
    add("complex-domain denoising", "yes" if complex_data else "no", "phase saved" if complex_data else "magnitude only")
    return rows


def plot_scheme(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    ax=None,
    title: str | None = None,
    antipodal: bool = True,
    b_max: float | None = None,
    colorbar: bool = True,
):
    """3-D scatter of q-space samples, radius proportional to sqrt(b).

    Each measurement samples both ``q`` and ``-q`` (the signal is symmetric), so by default
    both points are drawn. Points are colored by b-value on a ``viridis`` scale running from
    0 to ``b_max``. Schemes with up to six shells get a legend; schemes with more (DSI grids)
    get a colorbar instead. Pass the same ``b_max`` to every panel of a comparison so that
    axis limits and colors match across schemes; it defaults to this scheme's largest b.
    Set ``colorbar=False`` when the figure draws one shared colorbar itself.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if ax is None:
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(111, projection="3d")
    bvals = np.asarray(bvals, float)
    q = np.sqrt(bvals)[:, None] * np.asarray(bvecs, float)
    if antipodal:
        q, bvals = np.concatenate([q, -q]), np.concatenate([bvals, bvals])
    if b_max is None:
        b_max = bvals.max()
    cmap, norm = plt.get_cmap("viridis"), Normalize(0.0, b_max if b_max > 0 else 1.0)
    shells = shells_of(bvals)
    keys = np.round(bvals / 50.0) * 50.0
    if len(shells) <= 6:
        for b in sorted(shells):
            sel = keys == b
            n = int(sel.sum() // (2 if antipodal else 1))
            ax.scatter(*q[sel].T, s=12, color=cmap(norm(b)), label=f"b = {b:.0f} ({n})", depthshade=False)
        ax.legend(fontsize=7, loc="upper left")
    else:
        sc = ax.scatter(*q.T, s=10, c=bvals, cmap=cmap, norm=norm, depthshade=False)
        if colorbar:
            plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.05, label="b (s/mm²)")
    lim = np.sqrt(b_max) * 1.05 if b_max > 0 else 1.0
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    if title:
        ax.set_title(title)
    return ax
