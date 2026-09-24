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
    """Group b-values into shells (centres rounded to ``tol``) with volume counts."""
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


def plot_scheme(bvals: np.ndarray, bvecs: np.ndarray, ax=None, title: str | None = None):
    """3-D scatter of q-space samples, radius proportional to sqrt(b), coloured by shell."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(111, projection="3d")
    q = np.sqrt(np.asarray(bvals))[:, None] * np.asarray(bvecs)
    shells = shells_of(bvals)
    keys = np.round(np.asarray(bvals) / 50.0) * 50.0
    for i, b in enumerate(sorted(shells)):
        sel = keys == b
        ax.scatter(*q[sel].T, s=12, label=f"b={b:.0f} ({sel.sum()})", depthshade=False)
    lim = np.sqrt(max(bvals)) * 1.05 if max(bvals) > 0 else 1.0
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.legend(fontsize=7, loc="upper left")
    if title:
        ax.set_title(title)
    return ax
