"""TRXScan ground truth: the 27 analytic microstructure maps and the truth peaks (Part IV).

``trxscan-microstructure`` writes one map per name in :data:`TRUTH_MAPS`, using dipy's
conventions (it is validated against dipy at 1e-8), so a dipy fit of the simulated data can
be compared to these maps directly. ``trxscan --truth-peaks`` writes a 9-volume image of up
to three peak vectors per voxel, each scaled by its mass fraction. The pipeline stores both
under BIDS derivative names, ``<sub>_model-truth_param-<name>_dwimap.nii.gz``, in the
``derivatives/trxscan`` dataset of a simulated dataset or at the top level of the
``truth`` dataset.
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

#: Where a simulated dataset keeps its truth maps; the ``truth`` dataset keeps them at its root.
TRUTH_PIPELINE = "derivatives/trxscan"


def param_label(name: str) -> str:
    """The BIDS ``param-`` label of a truth-map name (``k_bulk`` becomes ``kbulk``)."""
    return name.replace("_", "")


def truth_filename(sub: str, name: str) -> str:
    """File name of one truth map: ``<sub>_model-truth_param-<label>_dwimap.nii.gz``."""
    return f"{sub}_model-truth_param-{param_label(name)}_dwimap.nii.gz"


def truth_file(ds: Dataset, sub: str, name: str) -> str:
    """Path inside ``ds`` of one truth map, in ``derivatives/trxscan`` or at the dataset root."""
    rel = f"{sub}/dwi/{truth_filename(sub, name)}"
    for root in (TRUTH_PIPELINE, ""):
        candidate = f"{root}/{rel}" if root else rel
        if (ds.path / candidate).exists():
            return candidate
    raise FileNotFoundError(f"truth map {name!r} for {sub} not found in dataset {ds.id!r} at {ds.path}")


def load_truth(ds: Dataset, sub: str, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Load the truth maps of one subject from a dataset into a ``{name: array}`` dict."""
    names = names or ALL_TRUTH_MAPS
    return {n: ds.volume(truth_file(ds, sub, n)) for n in names}


def load_truth_peaks(ds: Dataset, entities: str) -> np.ndarray:
    """Truth peaks of one run (``entities`` like ``sub-0001a_acq-hbcd_dir-AP``) as ``(x, y, z, 3 peaks, 3)``.

    The vector norm is the peak's mass fraction.
    """
    sub = entities.split("_")[0]
    v = ds.volume(f"{TRUTH_PIPELINE}/{sub}/dwi/{entities}_model-truth_param-peaks_dwimap.nii.gz")
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
