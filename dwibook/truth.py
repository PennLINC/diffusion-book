"""TRXScan ground truth: the 27 analytic microstructure maps and the truth peaks (Part IV).

``trxscan-microstructure`` writes ``<prefix>_<name>.nii.gz`` for each name in
:data:`TRUTH_MAPS`, using dipy's conventions (it is validated against dipy at 1e-8), so a
dipy fit of the simulated data can be compared to these maps directly. ``trxscan
--truth-peaks`` writes a 9-volume image of up to three peak vectors per voxel, each scaled by
its mass fraction.
"""

from __future__ import annotations

import numpy as np

from .data import Dataset

#: Map names in the order ``trxscan-microstructure`` writes them, grouped by family.
TRUTH_MAPS: dict[str, tuple[str, ...]] = {
    "DTI": ("fa", "md", "rd", "ad"),
    "DKI": ("ak", "rk", "mk", "mkt", "kfa"),
    "QTI": ("micro_fa", "coherence", "k_bulk", "k_shear"),
    "MAP-MRI": ("rtop", "rtap", "rtpp", "msd", "qiv", "ng", "ngpar", "ngperp", "pa"),
    "ODF": ("gfa", "qa"),
    "NODDI-style": ("icvf", "odi", "isovf"),
}
ALL_TRUTH_MAPS: tuple[str, ...] = tuple(n for names in TRUTH_MAPS.values() for n in names)


def load_truth(ds: Dataset, prefix: str, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Load truth maps ``<prefix>_<name>.nii.gz`` from a dataset into a ``{name: array}`` dict."""
    names = names or ALL_TRUTH_MAPS
    return {n: ds.volume(f"{prefix}_{n}.nii.gz") for n in names}


def load_truth_peaks(ds: Dataset, name: str) -> np.ndarray:
    """Truth peaks as ``(x, y, z, 3 peaks, 3)``; the vector norm is the peak's mass fraction."""
    v = ds.volume(name)
    if v.shape[-1] != 9:
        raise ValueError(f"expected a 9-volume peaks image, got shape {v.shape}")
    return v.reshape(*v.shape[:3], 3, 3)


def angular_error_deg(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Angle in degrees between vectors along the last axis, ignoring sign (antipodal symmetry)."""
    est = np.asarray(est, float)
    ref = np.asarray(ref, float)
    en = np.linalg.norm(est, axis=-1)
    rn = np.linalg.norm(ref, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.abs(np.sum(est * ref, axis=-1) / (en * rn))
    cos = np.clip(cos, 0.0, 1.0)
    out = np.degrees(np.arccos(cos))
    out[(en == 0) | (rn == 0)] = np.nan
    return out


def summarize_error(fit: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Bias, RMSE, and Pearson correlation of ``fit`` vs ``truth`` inside ``mask``."""
    f = np.asarray(fit, float)[mask]
    t = np.asarray(truth, float)[mask]
    ok = np.isfinite(f) & np.isfinite(t)
    f, t = f[ok], t[ok]
    if f.size == 0:
        return {"bias": np.nan, "rmse": np.nan, "r": np.nan, "n": 0}
    return {
        "bias": float(np.mean(f - t)),
        "rmse": float(np.sqrt(np.mean((f - t) ** 2))),
        "r": float(np.corrcoef(f, t)[0, 1]) if f.size > 1 else np.nan,
        "n": int(f.size),
    }
